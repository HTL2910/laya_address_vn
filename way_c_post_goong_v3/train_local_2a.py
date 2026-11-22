"""Offline real-order local trainer & pure-ONNX exporter for Post-Goong 2A (v3.1).

Runs 100% locally with zero network calls, zero PyTorch, zero `onnx` package,
zero pickle, and zero PII written to any output file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
import unicodedata

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from .features import DEFAULT_ARTIFACTS_DIR, GazetteerV3
    from .features_2a import FEATURE_SETS, FeatureExtractor2A
    from .onnx_export_pure import export_lr_hgb_ensemble_to_onnx
except ImportError:  # pragma: no cover - direct script execution fallback
    from features import DEFAULT_ARTIFACTS_DIR, GazetteerV3
    from features_2a import FEATURE_SETS, FeatureExtractor2A
    from onnx_export_pure import export_lr_hgb_ensemble_to_onnx


# ---------------------------------------------------------------------------
# Standalone dataset loading, schema resolution & label normalization
# ---------------------------------------------------------------------------
_COLUMN_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "text": (
        "raw_text", "dia_chi_goc", "raw_address", "customer_address",
        "dia_chi_khach", "dia_chi", "full_address", "address", "chuoi", "text",
    ),
    "outcome": (
        "is_return", "hoan", "returned", "ket_cuc", "delivery_status",
        "trang_thai_giao", "trang_thai", "outcome", "status", "ket_qua", "label",
    ),
    "rule_d": (
        "rule_d_pass", "pass_rule_d", "luat_d_pass", "rule_d_ok", "d_pass",
        "rule_d", "luat_d", "rule_d_blocked", "rule_d_block", "luat_d_chan",
    ),
    "phone": (
        "customer_phone", "receiver_phone", "so_dien_thoai", "sdt",
        "phone", "customer_id", "khach_hang",
    ),
    "timestamp": (
        "created_at", "order_date", "ngay_tao", "thoi_gian", "timestamp", "date", "time",
    ),
    "goong_ward": (
        "goong_ward", "goong_xa", "goong_commune", "auto_ward",
        "xa_phuong", "phuong_xa", "ward", "phuong", "xa", "commune",
    ),
    "goong_district": (
        "goong_district", "goong_huyen", "auto_district",
        "quan_huyen", "district", "huyen", "quan",
    ),
    "goong_province": (
        "goong_province", "goong_tinh", "auto_province",
        "tinh_tp", "tinh_thanh", "province", "tinh", "city", "tp",
    ),
    "goong_formatted": (
        "goong_formatted", "goong_address", "goong_full", "formatted_address", "auto_address",
    ),
    "goong_score": (
        "goong_score", "goong_confidence", "goong_prob", "confidence", "score",
    ),
    "goong_lat": ("goong_lat", "lat", "latitude", "vi_do"),
    "goong_lng": ("goong_lng", "lng", "lon", "longitude", "kinh_do"),
}

_RETURN_TOKENS = frozenset({
    "1", "1.0", "true", "yes", "y",
    "hoan", "don hoan", "hoan hang", "chuyen hoan", "da hoan",
    "returned", "return", "rto", "fail", "failed", "cancel", "cancelled", "canceled",
    "huy", "huy don", "da huy", "bom", "bom hang", "khong giao duoc",
    "tra hang", "that bai", "giao that bai",
})
_DELIVERED_TOKENS = frozenset({
    "0", "0.0", "false", "no", "n",
    "giao", "da giao", "giao thanh cong", "thanh cong", "hoan thanh",
    "delivered", "deliver", "success", "done", "completed", "ok", "nhan hang", "da nhan",
})
_AMBIG_TRUE = frozenset({"1", "1.0", "true", "yes", "y"})
_AMBIG_FALSE = frozenset({"0", "0.0", "false", "no", "n"})
_EXPLICIT_PASS = frozenset({
    "pass", "passed", "ok", "allow", "allowed", "accept", "accepted",
    "cho qua", "qua", "du", "du den xa", "hop le", "address_sufficient", "sufficient",
})
_EXPLICIT_BLOCK = frozenset({
    "block", "blocked", "reject", "rejected", "fail", "failed", "deny", "denied",
    "chan", "khong qua", "tu choi", "chua du", "not_address", "address_insufficient", "insufficient",
})
_PHONE_IN_TEXT_RE = re.compile(r"(?<!\d)(?:\+?84[\s.\-]?|0)\d(?:[\s.\-]?\d){7,9}(?!\d)")


def _strip_accents(text: str) -> str:
    if not text:
        return ""
    s = str(text).replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def _norm_col(col: str) -> str:
    s = _strip_accents(col).lower().strip()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def _norm_tok(val: object) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = _strip_accents(str(val)).lower().strip()
    s = re.sub(r"[^a-z0-9.\s/_-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _load_dataset(path: str | Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(p, sheet_name=sheet_name or 0, engine="openpyxl", dtype=object).reset_index(drop=True)
    if ext in (".jsonl", ".ndjson"):
        return pd.read_json(p, lines=True).reset_index(drop=True)
    if ext == ".parquet":
        return pd.read_parquet(p).reset_index(drop=True)
    if ext in (".csv", ".tsv", ".txt"):
        last_err: Optional[Exception] = None
        for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
            try:
                if ext == ".tsv":
                    sep = "\t"
                else:
                    with p.open("r", encoding=enc, errors="replace") as f:
                        head = "".join(f.readline() for _ in range(5))
                    counts = {s: head.count(s) for s in (",", "\t", ";")}
                    best_sep = max(counts, key=lambda k: counts[k])
                    sep = best_sep if counts[best_sep] > 0 else ","
                return pd.read_csv(p, encoding=enc, sep=sep, dtype=object, keep_default_na=True).reset_index(drop=True)
            except UnicodeDecodeError as exc:
                last_err = exc
        raise RuntimeError(f"Could not decode CSV file {p}: {last_err}")
    raise ValueError(f"Unsupported dataset format '{ext}'. Supported: .csv, .tsv, .xlsx, .jsonl, .parquet")


def _resolve_columns(df: pd.DataFrame, overrides: Dict[str, Optional[object]]) -> Dict[str, Any]:
    raw_cols = [str(c) for c in df.columns]
    norm_to_raw: Dict[str, str] = {}
    for col in raw_cols:
        nc = _norm_col(col)
        norm_to_raw.setdefault(nc, col)

    used: set[str] = set()
    resolved: Dict[str, Optional[str]] = {}

    for role in _COLUMN_CANDIDATES:
        ov = overrides.get(role)
        if ov is not None and str(ov).strip():
            ov_s = str(ov).strip()
            if ov_s in raw_cols:
                resolved[role] = ov_s
                used.add(ov_s)
            elif _norm_col(ov_s) in norm_to_raw:
                actual = norm_to_raw[_norm_col(ov_s)]
                resolved[role] = actual
                used.add(actual)
            else:
                raise ValueError(f"Column override --{role.replace('_', '-')}-col='{ov_s}' not found in {raw_cols}")

    for role, candidates in _COLUMN_CANDIDATES.items():
        if role in resolved:
            continue
        found: Optional[str] = None
        for cand in candidates:
            if cand in norm_to_raw and norm_to_raw[cand] not in used:
                found = norm_to_raw[cand]
                break
        if found is None and role in ("text", "outcome", "rule_d", "phone", "timestamp"):
            for col in raw_cols:
                if col in used:
                    continue
                nc = _norm_col(col)
                if any(nc == c or nc.endswith("_" + c) or nc.startswith(c + "_") for c in candidates):
                    found = col
                    break
        resolved[role] = found
        if found is not None:
            used.add(found)

    missing_req = [r for r in ("text", "outcome") if not resolved.get(r)]
    if missing_req:
        raise ValueError(f"Cannot auto-detect required columns {missing_req}. Pass --text-col / --outcome-col.")

    rule_d_col = resolved.get("rule_d")
    rule_d_is_block = bool(overrides.get("rule_d_is_block", False))
    if not rule_d_is_block and rule_d_col is not None:
        nc_rd = _norm_col(rule_d_col)
        if any(k in nc_rd for k in ("block", "reject", "chan")) and "pass" not in nc_rd:
            rule_d_is_block = True

    return {"cols": resolved, "rule_d_is_block": rule_d_is_block}


def _parse_is_return(
    series: pd.Series,
    return_values: Optional[Sequence[str]] = None,
    delivered_values: Optional[Sequence[str]] = None,
) -> np.ndarray:
    ret_set = set(_RETURN_TOKENS)
    del_set = set(_DELIVERED_TOKENS)
    if return_values:
        ret_set.update(_norm_tok(v) for v in return_values if str(v).strip())
    if delivered_values:
        del_set.update(_norm_tok(v) for v in delivered_values if str(v).strip())

    out = np.full(len(series), -1, dtype=np.int8)
    for i, val in enumerate(series.tolist()):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if isinstance(val, bool):
            out[i] = 1 if val else 0
            continue
        if isinstance(val, (int, float, np.integer, np.floating)):
            fv = float(val)
            if fv == 1.0:
                out[i] = 1
            elif fv == 0.0:
                out[i] = 0
            continue
        tok = _norm_tok(val)
        if not tok:
            continue
        if tok in ret_set:
            out[i] = 1
        elif tok in del_set:
            out[i] = 0
        elif ("hoan" in tok or "tra hang" in tok or "rto" in tok or "that bai" in tok) and "khong hoan" not in tok and "thanh cong" not in tok:
            out[i] = 1
        elif ("thanh cong" in tok or "da giao" in tok or "delivered" in tok) and "khong" not in tok and "chua" not in tok:
            out[i] = 0
    return out


def _parse_bool_gate(series: pd.Series, invert: bool = False) -> np.ndarray:
    out = np.ones(len(series), dtype=bool)
    for i, val in enumerate(series.tolist()):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if isinstance(val, bool):
            out[i] = (not val) if invert else val
            continue
        if isinstance(val, (int, float, np.integer, np.floating)):
            fv = float(val) >= 0.5
            out[i] = (not fv) if invert else fv
            continue
        tok = _norm_tok(val)
        if not tok:
            continue
        if tok in _EXPLICIT_PASS or ("sufficient" in tok and "insufficient" not in tok) or (
            any(k in tok for k in ("pass", "allow", "accept", "cho qua")) and "not" not in tok and "khong" not in tok
        ):
            out[i] = True
        elif tok in _EXPLICIT_BLOCK or any(k in tok for k in ("block", "reject", "chan", "insufficient", "not_address")):
            out[i] = False
        elif tok in _AMBIG_TRUE:
            out[i] = not invert
        elif tok in _AMBIG_FALSE:
            out[i] = invert
    return out


def _str_list(df: pd.DataFrame, col: Optional[str]) -> List[str]:
    if not col or col not in df.columns:
        return [""] * len(df)
    out: List[str] = []
    for v in df[col].tolist():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append("")
        else:
            s = str(v).strip()
            out.append("" if s.lower() in ("nan", "none", "null") else s)
    return out


def _float_list(df: pd.DataFrame, col: Optional[str]) -> List[Optional[float]]:
    if not col or col not in df.columns:
        return [None] * len(df)
    out: List[Optional[float]] = []
    for v in df[col].tolist():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(None)
            continue
        try:
            fv = float(str(v).replace(",", ".").strip())
            out.append(None if (math.isnan(fv) or math.isinf(fv)) else fv)
        except (TypeError, ValueError):
            out.append(None)
    return out


def _populate_goong_fields(
    df: pd.DataFrame, cols: Dict[str, Optional[str]]
) -> Tuple[List[str], List[str], List[str], List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    gw = _str_list(df, cols.get("goong_ward"))
    gd = _str_list(df, cols.get("goong_district"))
    gp = _str_list(df, cols.get("goong_province"))
    gf = _str_list(df, cols.get("goong_formatted"))
    gs = _float_list(df, cols.get("goong_score"))
    glat = _float_list(df, cols.get("goong_lat"))
    glng = _float_list(df, cols.get("goong_lng"))

    for i in range(len(df)):
        if (not gw[i] and not gd[i] and not gp[i]) and gf[i]:
            fmt = re.sub(r"\b(?:Viet\s*Nam|Vietnam|VN|\d{5,6})\b", "", gf[i], flags=re.IGNORECASE).strip(" ,;-")
            segs = [p.strip() for p in re.split(r"[,;\n]+|\s+-\s+", fmt) if p.strip()]
            if len(segs) >= 3:
                gp[i], gd[i], gw[i] = segs[-1], segs[-2], segs[-3]
            elif len(segs) == 2:
                gp[i], gw[i] = segs[-1], segs[-2]
            elif len(segs) == 1:
                gp[i] = segs[0]
    return gw, gd, gp, gs, glat, glng


def _phone_group_key(phone_val: object, raw_text: str) -> str:
    raw_ph = ""
    if phone_val is not None and not (isinstance(phone_val, float) and np.isnan(phone_val)):
        raw_ph = re.sub(r"\D+", "", str(phone_val))
    if not raw_ph:
        m = _PHONE_IN_TEXT_RE.search(raw_text or "")
        if m:
            raw_ph = re.sub(r"\D+", "", m.group(0))
    if raw_ph.startswith("84") and len(raw_ph) >= 10:
        raw_ph = "0" + raw_ph[2:]
    if len(raw_ph) >= 8:
        h = hashlib.sha256(raw_ph[-9:].encode("utf-8")).hexdigest()[:16]
        return f"p_{h}"
    h = hashlib.sha256(_strip_accents(raw_text).lower().encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"t_{h}"


# ---------------------------------------------------------------------------
# Strict Time/Group CV, Curve Calibration & Forward Frozen Validation
# ---------------------------------------------------------------------------
def _build_cv_folds(
    y: np.ndarray,
    group_keys: Sequence[str],
    timestamps: Optional[pd.Series],
    requested_mode: str,
    n_splits: int,
    seed: int,
) -> Tuple[str, List[Tuple[np.ndarray, np.ndarray]], np.ndarray, Optional[np.ndarray]]:
    n = len(y)
    g_arr = np.asarray(group_keys, dtype=object)
    eff_mode = requested_mode
    parsed_ts: Optional[pd.Series] = None

    if timestamps is not None:
        parsed_ts = pd.to_datetime(timestamps, errors="coerce")
        valid_frac = float(parsed_ts.notna().mean()) if n > 0 else 0.0
        uniq_ts = int(parsed_ts.dropna().nunique())
        if eff_mode == "auto":
            eff_mode = "time" if (valid_frac >= 0.80 and uniq_ts >= max(10, n_splits + 2)) else "group"
        elif eff_mode == "time" and (valid_frac < 0.50 or uniq_ts < n_splits + 2):
            eff_mode = "group"
    elif eff_mode == "auto":
        eff_mode = "group"

    if eff_mode == "time" and parsed_ts is not None:
        ts_filled = parsed_ts.fillna(parsed_ts.median())
        sorted_pos = np.argsort(ts_filled.to_numpy(), kind="mergesort")
        blocks = np.array_split(sorted_pos, n_splits + 1)
        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        oof_mask = np.zeros(n, dtype=bool)
        unseen_mask = np.zeros(n, dtype=bool)
        train_acc: List[np.ndarray] = [blocks[0]]
        seen_phones: set[str] = {str(g_arr[idx]) for idx in blocks[0] if str(g_arr[idx]).startswith("p_")}
        for k in range(1, n_splits + 1):
            va_idx = blocks[k]
            tr_idx = np.concatenate(train_acc)
            folds.append((tr_idx, va_idx))
            oof_mask[va_idx] = True
            for idx in va_idx:
                gk = str(g_arr[idx])
                if gk.startswith("p_") and gk not in seen_phones:
                    unseen_mask[idx] = True
            train_acc.append(va_idx)
            seen_phones.update(str(g_arr[idx]) for idx in va_idx if str(g_arr[idx]).startswith("p_"))
        return "time", folds, oof_mask, unseen_mask

    if eff_mode == "group":
        uniq_g = len(set(group_keys))
        actual = min(n_splits, max(2, uniq_g))
        if actual >= 2 and uniq_g >= actual:
            sgkf = StratifiedGroupKFold(n_splits=actual, shuffle=True, random_state=seed)
            folds = [(tr, va) for tr, va in sgkf.split(np.zeros(n), y, groups=g_arr)]
            return "group", folds, np.ones(n, dtype=bool), None

    actual = min(n_splits, max(2, int(np.sum(y == 1)), int(np.sum(y == 0))))
    skf = StratifiedKFold(n_splits=actual, shuffle=True, random_state=seed)
    return "stratified", list(skf.split(np.zeros(n), y)), np.ones(n, dtype=bool), None


def _compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(1, len(y_true))
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            ece += (cnt / n) * abs(float(np.mean(y_true[mask])) - float(np.mean(probs[mask])))
    return float(ece)


def _eval_mask_metrics(y_true: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    n = len(y_true)
    base_ret = float(np.mean(y_true)) if n > 0 else 0.0
    blk = int(np.sum(mask))
    caught = int(np.sum(mask & (y_true == 1)))
    lost = int(np.sum(mask & (y_true == 0)))
    prec = float(caught / blk) if blk > 0 else 0.0
    lift = float(prec / base_ret) if (blk > 0 and base_ret > 0.0) else None
    return {
        "n_eval": n,
        "blocked": blk,
        "block_rate": round(blk / max(1, n), 4),
        "caught_returns": caught,
        "lost_deliveries": lost,
        "ratio_lost_per_caught": round(lost / caught, 4) if caught > 0 else None,
        "return_rate_blocked": round(prec, 4) if blk > 0 else None,
        "lift_vs_base": round(lift, 3) if lift is not None else None,
    }


def _evaluate_curve(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    n = len(y_true)
    total_ret = int(np.sum(y_true == 1))
    base_rate = float(total_ret / max(1, n))

    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = y_true[order]
    change_idx = np.flatnonzero(np.diff(s_sorted) != 0)
    ends = np.append(change_idx, n - 1)
    cum_ret = np.cumsum(y_sorted == 1)
    cum_del = np.cumsum(y_sorted == 0)

    pts: List[Dict[str, Any]] = []
    for ei in ends:
        blk = int(ei + 1)
        caught = int(cum_ret[ei])
        lost = int(cum_del[ei])
        ratio = float(lost / caught) if caught > 0 else None
        pts.append({
            "threshold": round(float(s_sorted[ei]), 6),
            "blocked": blk,
            "block_rate": round(blk / max(1, n), 4),
            "caught_returns": caught,
            "lost_deliveries": lost,
            "ratio_lost_per_caught": round(ratio, 4) if ratio is not None else None,
            "precision_return": round(caught / max(1, blk), 4),
        })

    def _best_min_caught(mc: int) -> Optional[Dict[str, Any]]:
        eff_mc = min(mc, total_ret) if total_ret > 0 else 1
        cands = [p for p in pts if p["caught_returns"] >= eff_mc and p["ratio_lost_per_caught"] is not None]
        if not cands:
            return None
        return min(cands, key=lambda p: (p["ratio_lost_per_caught"], -p["caught_returns"]))

    def _max_at_ratio(max_ratio: float, min_c: int = 1) -> Optional[Dict[str, Any]]:
        cands = [
            p for p in pts
            if p["caught_returns"] >= min_c and p["ratio_lost_per_caught"] is not None and p["ratio_lost_per_caught"] <= max_ratio
        ]
        if not cands:
            return None
        return max(cands, key=lambda p: (p["caught_returns"], -p["lost_deliveries"]))

    checkpoints: Dict[str, Optional[Dict[str, Any]]] = {}
    for vol in (30, 50, 100, 200):
        cands = [p for p in pts if p["caught_returns"] >= vol]
        checkpoints[f"ge{vol}"] = cands[0] if cands else None
    for pct in (0.01, 0.03, 0.04, 0.05):
        target_blk = max(1, int(round(n * pct)))
        cands = [p for p in pts if p["blocked"] >= target_blk]
        checkpoints[f"top_{int(round(pct * 100))}pct"] = cands[0] if cands else None

    roc_auc = round(float(roc_auc_score(y_true, scores)), 4) if len(np.unique(y_true)) == 2 else None
    pr_auc = round(float(average_precision_score(y_true, scores)), 4) if total_ret > 0 else None
    clipped = np.clip(scores, 0.0, 1.0)
    return {
        "n_eval": n,
        "total_returns": total_ret,
        "base_return_rate": round(base_rate, 4),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": round(float(brier_score_loss(y_true, clipped)), 4),
        "ece": round(_compute_ece(y_true, clipped), 4),
        "best30": _best_min_caught(30),
        "max_at_1_5": _max_at_ratio(1.5, min_c=1),
        "checkpoints": checkpoints,
        "_points": pts,
    }


def _select_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_hard_ratio: float,
    min_hard_caught: int,
    target_soft_ratio: float,
    min_soft_caught: int,
) -> Tuple[float, float, str, str]:
    curve = _evaluate_curve(y_true, scores)
    pts: List[Dict[str, Any]] = curve["_points"]
    total_ret = int(np.sum(y_true == 1))

    eff_hard_min = min(min_hard_caught, max(1, total_ret // 3)) if total_ret > 0 else 1
    hard_cands = [
        p for p in pts
        if p["caught_returns"] >= eff_hard_min
        and p["ratio_lost_per_caught"] is not None
        and p["ratio_lost_per_caught"] <= target_hard_ratio
    ]
    if hard_cands:
        hard_pt = max(hard_cands, key=lambda p: (p["caught_returns"], -p["lost_deliveries"]))
        t_high = float(hard_pt["threshold"])
        hard_src = f"ratio<={target_hard_ratio}"
    else:
        t_high = float(np.quantile(scores, 0.99)) if len(scores) else 0.70
        hard_src = "top_1pct_quantile_fallback"

    eff_soft_min = min(min_soft_caught, max(1, total_ret // 2)) if total_ret > 0 else 1
    soft_cands = [
        p for p in pts
        if p["caught_returns"] >= eff_soft_min
        and p["ratio_lost_per_caught"] is not None
        and p["ratio_lost_per_caught"] <= target_soft_ratio
    ]
    if soft_cands:
        soft_pt = max(soft_cands, key=lambda p: (p["caught_returns"], -p["lost_deliveries"]))
        t_low = float(soft_pt["threshold"])
        soft_src = f"ratio<={target_soft_ratio}"
    else:
        t_low = float(np.quantile(scores, 0.97)) if len(scores) else 0.45
        soft_src = "top_3pct_quantile_fallback"

    t_low = min(t_low, t_high)
    return round(t_low, 6), round(t_high, 6), soft_src, hard_src


def _fit_pair(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    lr_c: float,
    hgb_iters: int,
    hgb_depth: int,
    hgb_lr: float,
    hgb_l2: float,
    seed: int,
) -> Tuple[StandardScaler, LogisticRegression, HistGradientBoostingClassifier]:
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    lr = LogisticRegression(C=lr_c, max_iter=600, solver="lbfgs")
    lr.fit(X_tr_sc, y_tr)

    min_leaf = max(5, min(35, len(y_tr) // 30))
    hgb = HistGradientBoostingClassifier(
        max_iter=hgb_iters,
        max_depth=hgb_depth,
        learning_rate=hgb_lr,
        l2_regularization=hgb_l2,
        min_samples_leaf=min_leaf,
        random_state=seed,
    )
    hgb.fit(X_tr, y_tr)
    return scaler, lr, hgb


def _predict_blend(
    scaler: StandardScaler,
    lr: LogisticRegression,
    hgb: HistGradientBoostingClassifier,
    X: np.ndarray,
    w_lr: float,
    w_hgb: float,
) -> np.ndarray:
    w_sum = max(1e-9, w_lr + w_hgb)
    p_lr = lr.predict_proba(scaler.transform(X))[:, 1]
    p_hgb = hgb.predict_proba(X)[:, 1]
    return (w_lr / w_sum) * p_lr + (w_hgb / w_sum) * p_hgb


def _sr_impact(y_true: np.ndarray, blocked_mask: np.ndarray) -> Dict[str, Optional[float]]:
    n = len(y_true)
    if n == 0:
        return {"sr_before": None, "sr_after": None, "sr_delta": None}
    sr_before = float(np.mean(y_true == 0))
    kept = ~blocked_mask
    if int(np.sum(kept)) == 0:
        return {"sr_before": round(sr_before, 4), "sr_after": None, "sr_delta": None}
    sr_after = float(np.mean(y_true[kept] == 0))
    return {
        "sr_before": round(sr_before, 4),
        "sr_after": round(sr_after, 4),
        "sr_delta": round(sr_after - sr_before, 4),
    }


def _strip_internal(curve_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in curve_dict.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Main training & export pipeline
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train & export local real-order Post-Goong 2A ONNX gate (v3.1).")
    p.add_argument("--data", required=True, help="Path to local DEV orders file (.csv/.tsv/.xlsx/.parquet/.jsonl)")
    p.add_argument("--sheet", default=None, help="Excel sheet name if reading .xlsx")
    p.add_argument("--output-dir", required=True, help="Directory to write local ONNX + calibration + report")
    p.add_argument("--converter-path", default=None, help="Optional path to vn_admin_v3_converter.json")

    # Explicit column overrides
    p.add_argument("--text-col", default=None)
    p.add_argument("--outcome-col", default=None)
    p.add_argument("--return-values", default=None, help="Comma-separated labels treated as return (1)")
    p.add_argument("--delivered-values", default=None, help="Comma-separated labels treated as delivered (0)")
    p.add_argument("--rule-d-col", default=None)
    p.add_argument("--rule-d-is-block", action="store_true")
    p.add_argument("--no-rule-d-filter", action="store_true", help="Train on all valid rows without filtering Rule D")
    p.add_argument("--phone-col", default=None)
    p.add_argument("--time-col", default=None)
    p.add_argument("--goong-ward-col", default=None)
    p.add_argument("--goong-district-col", default=None)
    p.add_argument("--goong-province-col", default=None)
    p.add_argument("--goong-formatted-col", default=None)
    p.add_argument("--goong-score-col", default=None)
    p.add_argument("--goong-lat-col", default=None)
    p.add_argument("--goong-lng-col", default=None)

    # Model & threshold parameters
    p.add_argument("--feature-set", choices=tuple(FEATURE_SETS), default="v31_2a_56")
    p.add_argument("--split-mode", choices=("auto", "time", "group", "stratified"), default="auto")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--calib-folds", type=int, default=3)
    p.add_argument("--hgb-iters", type=int, default=120)
    p.add_argument("--hgb-depth", type=int, default=3)
    p.add_argument("--hgb-lr", type=float, default=0.05)
    p.add_argument("--hgb-l2", type=float, default=2.0)
    p.add_argument("--lr-c", type=float, default=0.5)
    p.add_argument("--w-lr", type=float, default=0.5)
    p.add_argument("--w-hgb", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--target-hard-ratio", type=float, default=0.90)
    p.add_argument("--min-hard-caught", type=int, default=15)
    p.add_argument("--target-soft-ratio", type=float, default=1.50)
    p.add_argument("--min-soft-caught", type=int, default=30)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = _load_dataset(args.data, sheet_name=args.sheet)
    overrides: Dict[str, Optional[object]] = {
        "text": args.text_col,
        "outcome": args.outcome_col,
        "rule_d": args.rule_d_col,
        "rule_d_is_block": args.rule_d_is_block,
        "phone": args.phone_col,
        "timestamp": args.time_col,
        "goong_ward": args.goong_ward_col,
        "goong_district": args.goong_district_col,
        "goong_province": args.goong_province_col,
        "goong_formatted": args.goong_formatted_col,
        "goong_score": args.goong_score_col,
        "goong_lat": args.goong_lat_col,
        "goong_lng": args.goong_lng_col,
    }
    schema_info = _resolve_columns(df_raw, overrides)
    cols: Dict[str, Optional[str]] = schema_info["cols"]
    rule_d_is_block: bool = bool(schema_info["rule_d_is_block"])

    ret_vals = [v.strip() for v in args.return_values.split(",")] if args.return_values else None
    del_vals = [v.strip() for v in args.delivered_values.split(",")] if args.delivered_values else None
    y_all = _parse_is_return(df_raw[str(cols["outcome"])], return_values=ret_vals, delivered_values=del_vals)
    valid_mask = (y_all == 0) | (y_all == 1)

    df_valid = df_raw.loc[valid_mask].copy().reset_index(drop=True)
    y_valid = y_all[valid_mask].astype(np.int64)
    if len(df_valid) < 40:
        raise ValueError(f"Too few labeled delivered/returned rows to train ({len(df_valid)} < 40).")

    rule_d_col = cols.get("rule_d")
    if (not args.no_rule_d_filter) and rule_d_col and rule_d_col in df_valid.columns:
        rd_pass = _parse_bool_gate(df_valid[rule_d_col], invert=rule_d_is_block)
        rule_d_summary = _eval_mask_metrics(y_valid, ~rd_pass)
    else:
        rd_pass = np.ones(len(df_valid), dtype=bool)
        rule_d_summary = None

    df_res = df_valid.loc[rd_pass].copy().reset_index(drop=True)
    y_res = y_valid[rd_pass]
    n_res = len(df_res)
    if n_res < 30 or int(np.sum(y_res == 1)) < 5 or int(np.sum(y_res == 0)) < 5:
        raise ValueError(
            f"Residual slice after Rule D is too small/imbalanced (n={n_res}, returns={int(np.sum(y_res == 1))})."
        )

    raw_texts = _str_list(df_res, cols.get("text"))
    gw, gd, gp, gs, glat, glng = _populate_goong_fields(df_res, cols)

    missing_ward_frac = float(sum(1 for w in gw if not w) / max(1, n_res))
    missing_score_frac = float(sum(1 for s in gs if s is None) / max(1, n_res))
    missing_coord_frac = float(sum(1 for a, b in zip(glat, glng) if a is None or b is None) / max(1, n_res))
    allow_missing_ward = missing_ward_frac > 0.02

    conv_src = GazetteerV3._resolve_converter_path(args.converter_path)
    extractor = FeatureExtractor2A(converter_path=conv_src, feature_set=args.feature_set)
    X_all = extractor.extract_matrix(
        raw_texts=raw_texts,
        goong_wards=gw,
        goong_districts=gd,
        goong_provinces=gp,
        goong_scores=gs,
        goong_lats=glat,
        goong_lngs=glng,
    )

    phone_col = cols.get("phone")
    phone_vals = df_res[phone_col].tolist() if (phone_col and phone_col in df_res.columns) else [None] * n_res
    group_keys = [_phone_group_key(ph, txt) for ph, txt in zip(phone_vals, raw_texts)]
    ts_col = cols.get("timestamp")
    ts_series = df_res[ts_col] if (ts_col and ts_col in df_res.columns) else None

    eff_mode, folds, oof_mask, unseen_full_mask = _build_cv_folds(
        y=y_res,
        group_keys=group_keys,
        timestamps=ts_series,
        requested_mode=args.split_mode,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    oof_scores = np.full(n_res, np.nan, dtype=np.float64)
    oof_fold_id = np.full(n_res, -1, dtype=np.int32)
    last_fold_models: Optional[Tuple[StandardScaler, LogisticRegression, HistGradientBoostingClassifier, np.ndarray]] = None

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        y_tr = y_res[tr_idx]
        if len(np.unique(y_tr)) < 2:
            oof_scores[va_idx] = float(np.mean(y_tr))
            oof_fold_id[va_idx] = fold_i
            continue
        sc_f, lr_f, hgb_f = _fit_pair(
            X_all[tr_idx], y_tr,
            lr_c=args.lr_c, hgb_iters=args.hgb_iters, hgb_depth=args.hgb_depth,
            hgb_lr=args.hgb_lr, hgb_l2=args.hgb_l2, seed=args.seed + fold_i,
        )
        oof_scores[va_idx] = _predict_blend(sc_f, lr_f, hgb_f, X_all[va_idx], args.w_lr, args.w_hgb)
        oof_fold_id[va_idx] = fold_i
        last_fold_models = (sc_f, lr_f, hgb_f, va_idx)

    y_oof = y_res[oof_mask]
    s_oof = oof_scores[oof_mask]
    f_oof = oof_fold_id[oof_mask]
    full_oof_curve = _evaluate_curve(y_oof, s_oof)

    # Strict forward frozen-threshold check (first calib_folds -> later folds)
    frozen_report: Dict[str, Any] = {}
    actual_folds = len(folds)
    if actual_folds >= 3 and 1 <= args.calib_folds < actual_folds:
        calib_m = f_oof < args.calib_folds
        future_m = f_oof >= args.calib_folds
        y_cal, s_cal = y_oof[calib_m], s_oof[calib_m]
        y_fut, s_fut = y_oof[future_m], s_oof[future_m]

        t_low_fr, t_high_fr, soft_src_fr, hard_src_fr = _select_thresholds(
            y_cal, s_cal,
            target_hard_ratio=args.target_hard_ratio, min_hard_caught=args.min_hard_caught,
            target_soft_ratio=args.target_soft_ratio, min_soft_caught=args.min_soft_caught,
        )
        fut_hard = s_fut >= t_high_fr
        fut_soft = (s_fut >= t_low_fr) & (s_fut < t_high_fr)
        fut_comb = s_fut >= t_low_fr
        cal_comb = s_cal >= t_low_fr

        cal_rate = float(np.mean(cal_comb)) if len(s_cal) else 0.0
        fut_rate = float(np.mean(fut_comb)) if len(s_fut) else 0.0
        drift_ratio = (fut_rate / cal_rate) if cal_rate > 1e-6 else None

        quantile_eval: Dict[str, Any] = {}
        for pct in (0.01, 0.03, 0.04):
            q_thr = float(np.quantile(s_cal, 1.0 - pct)) if len(s_cal) else 1.0
            q_m = s_fut >= q_thr
            quantile_eval[f"top_{int(round(pct * 100))}pct_frozen"] = {
                "threshold": round(q_thr, 6),
                **_eval_mask_metrics(y_fut, q_m),
                **_sr_impact(y_fut, q_m),
            }

        unseen_future_eval: Optional[Dict[str, Any]] = None
        if unseen_full_mask is not None:
            unseen_oof = unseen_full_mask[oof_mask]
            uf_m = future_m & unseen_oof
            if int(np.sum(uf_m)) >= 10:
                y_uf, s_uf = y_oof[uf_m], s_oof[uf_m]
                unseen_future_eval = {
                    "n_unseen_phone_future": int(len(y_uf)),
                    "hard_block": _eval_mask_metrics(y_uf, s_uf >= t_high_fr),
                    "soft_confirm_band": _eval_mask_metrics(y_uf, (s_uf >= t_low_fr) & (s_uf < t_high_fr)),
                    "hard_plus_soft": _eval_mask_metrics(y_uf, s_uf >= t_low_fr),
                }

        frozen_report = {
            "calib_folds": f"0..{args.calib_folds - 1}",
            "future_folds": f"{args.calib_folds}..{actual_folds - 1}",
            "n_calib": int(len(y_cal)),
            "n_future": int(len(y_fut)),
            "frozen_t_low": t_low_fr,
            "frozen_t_high": t_high_fr,
            "soft_rule_source": soft_src_fr,
            "hard_rule_source": hard_src_fr,
            "future_hard_block": _eval_mask_metrics(y_fut, fut_hard),
            "future_soft_confirm_band": _eval_mask_metrics(y_fut, fut_soft),
            "future_hard_plus_soft": {**_eval_mask_metrics(y_fut, fut_comb), **_sr_impact(y_fut, fut_comb)},
            "block_rate_drift_ratio": round(drift_ratio, 3) if drift_ratio is not None else None,
            "block_rate_drift_warning": bool(drift_ratio is not None and (drift_ratio > 2.0 or drift_ratio < 0.5)),
            "frozen_quantiles": quantile_eval,
            "unseen_phone_future": unseen_future_eval,
        }

    # Permutation feature importance on last fold (zero PII — numeric feature names only)
    top_features: List[Dict[str, Any]] = []
    if last_fold_models is not None:
        sc_last, lr_last, hgb_last, va_last = last_fold_models
        y_va_last = y_res[va_last]
        if len(np.unique(y_va_last)) == 2 and len(va_last) >= 20:
            X_va_base = X_all[va_last].copy()
            base_auc = float(roc_auc_score(
                y_va_last, _predict_blend(sc_last, lr_last, hgb_last, X_va_base, args.w_lr, args.w_hgb)
            ))
            perm_rng = np.random.default_rng(args.seed)
            n_repeats = 5
            imp_means = np.zeros(X_va_base.shape[1], dtype=np.float64)
            imp_stds = np.zeros(X_va_base.shape[1], dtype=np.float64)
            X_perm = X_va_base.copy()
            for col_i in range(X_va_base.shape[1]):
                col_orig = X_va_base[:, col_i]
                if float(np.ptp(col_orig)) == 0.0:
                    continue
                drops = np.empty(n_repeats, dtype=np.float64)
                for r_i in range(n_repeats):
                    X_perm[:, col_i] = perm_rng.permutation(col_orig)
                    auc_p = float(roc_auc_score(
                        y_va_last, _predict_blend(sc_last, lr_last, hgb_last, X_perm, args.w_lr, args.w_hgb)
                    ))
                    drops[r_i] = base_auc - auc_p
                X_perm[:, col_i] = col_orig
                imp_means[col_i] = float(np.mean(drops))
                imp_stds[col_i] = float(np.std(drops))

            for idx in np.argsort(-imp_means)[:15]:
                top_features.append({
                    "feature": extractor.feature_names[int(idx)],
                    "auc_drop_mean": round(float(imp_means[int(idx)]), 5),
                    "auc_drop_std": round(float(imp_stds[int(idx)]), 5),
                })

    # Calibrate final operating thresholds on full OOF & retrain final production ONNX on 100% residual rows
    t_low, t_high, soft_src, hard_src = _select_thresholds(
        y_oof, s_oof,
        target_hard_ratio=args.target_hard_ratio, min_hard_caught=args.min_hard_caught,
        target_soft_ratio=args.target_soft_ratio, min_soft_caught=args.min_soft_caught,
    )

    sc_full, lr_full, hgb_full = _fit_pair(
        X_all, y_res,
        lr_c=args.lr_c, hgb_iters=args.hgb_iters, hgb_depth=args.hgb_depth,
        hgb_lr=args.hgb_lr, hgb_l2=args.hgb_l2, seed=args.seed,
    )
    onnx_name = "post_goong_verifier_2a_local.onnx"
    onnx_path = out_dir / onnx_name
    export_lr_hgb_ensemble_to_onnx(
        out_path=onnx_path,
        feature_names=extractor.feature_names,
        lr_mean=sc_full.mean_,
        lr_std=sc_full.scale_,
        lr_coef=lr_full.coef_[0],
        lr_intercept=float(lr_full.intercept_[0]),
        hgb_model=hgb_full,
        w_lr=args.w_lr,
        w_hgb=args.w_hgb,
    )

    # Mandatory ONNX Runtime vs scikit-learn numerical parity check
    check_n = min(500, n_res)
    p_sk_chk = _predict_blend(sc_full, lr_full, hgb_full, X_all[:check_n], args.w_lr, args.w_hgb)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    p_ort_chk = sess.run(["risk_prob"], {"features": X_all[:check_n].astype(np.float32)})[0].reshape(-1)
    max_parity_diff = float(np.max(np.abs(p_sk_chk - p_ort_chk)))
    if max_parity_diff >= 1e-4:
        raise AssertionError(
            f"ONNX export parity check failed: max_abs_diff={max_parity_diff:.6f} >= 1e-4"
        )

    # Copy converter JSON next to model so PostGoongVerifier(artifacts_dir=out_dir) works standalone
    shutil.copy2(conv_src, out_dir / "vn_admin_v3_converter.json")

    calib_payload = {
        "version": "3.1.0",
        "model_filename": onnx_name,
        "feature_schema": args.feature_set,
        "feature_names": list(extractor.feature_names),
        "output_type": "probability",
        "temperature": 1.0,
        "t_low": float(t_low),
        "t_high": float(t_high),
        "allow_missing_ward": bool(allow_missing_ward),
        "oof_hard_block_rate": round(float(np.mean(s_oof >= t_high)), 4),
        "oof_total_action_rate": round(float(np.mean(s_oof >= t_low)), 4),
        "soft_threshold_source": soft_src,
        "hard_threshold_source": hard_src,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "calibration.json").write_text(
        json.dumps(calib_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_payload = {
        "laya_version": "3.1.0",
        "feature_schema": args.feature_set,
        "n_features": len(extractor.feature_names),
        "split_mode": eff_mode,
        "n_splits": actual_folds,
        "dataset_summary_no_pii": {
            "total_labeled": int(len(df_valid)),
            "total_returns": int(np.sum(y_valid == 1)),
            "rule_d_gate": rule_d_summary,
            "residual_n": n_res,
            "residual_returns": int(np.sum(y_res == 1)),
            "residual_return_rate": round(float(np.mean(y_res)), 4),
            "oof_n": int(len(y_oof)),
            "missing_goong_ward_frac": round(missing_ward_frac, 4),
            "missing_goong_score_frac": round(missing_score_frac, 4),
            "missing_goong_coord_frac": round(missing_coord_frac, 4),
            "allow_missing_ward": allow_missing_ward,
        },
        "oof_curve": _strip_internal(full_oof_curve),
        "oof_calibrated_operating_point": {
            "t_low": float(t_low),
            "t_high": float(t_high),
            "hard_block": _eval_mask_metrics(y_oof, s_oof >= t_high),
            "soft_confirm_band": _eval_mask_metrics(y_oof, (s_oof >= t_low) & (s_oof < t_high)),
            "hard_plus_soft": {**_eval_mask_metrics(y_oof, s_oof >= t_low), **_sr_impact(y_oof, s_oof >= t_low)},
        },
        "frozen_forward_validation": frozen_report,
        "top_15_permutation_features": top_features,
        "onnx_export": {
            "onnx_path": onnx_name,
            "onnx_size_kb": round(onnx_path.stat().st_size / 1024.0, 2),
            "parity_max_abs_diff": round(max_parity_diff, 8),
        },
    }
    (out_dir / "local_eval_report.json").write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    op_h = report_payload["oof_calibrated_operating_point"]["hard_block"]
    op_s = report_payload["oof_calibrated_operating_point"]["soft_confirm_band"]
    op_c = report_payload["oof_calibrated_operating_point"]["hard_plus_soft"]
    print(f"[LAYA v3.1 LOCAL 2A] Output dir: {out_dir}")
    print(f"  Residual rows: {n_res:,} (OOF: {len(y_oof):,}, split={eff_mode}, feature_set={args.feature_set})")
    print(f"  OOF ROC-AUC={full_oof_curve['roc_auc']}  PR-AUC={full_oof_curve['pr_auc']}  ECE={full_oof_curve['ece']}")
    print(f"  t_high={t_high} hard_block -> blocked={op_h['blocked']} caught={op_h['caught_returns']} lost={op_h['lost_deliveries']} ratio={op_h['ratio_lost_per_caught']}")
    print(f"  [t_low={t_low}, t_high) soft_zns -> blocked={op_s['blocked']} caught={op_s['caught_returns']} lost={op_s['lost_deliveries']} ratio={op_s['ratio_lost_per_caught']}")
    print(f"  hard+soft total -> blocked={op_c['blocked']} caught={op_c['caught_returns']} lost={op_c['lost_deliveries']} ratio={op_c['ratio_lost_per_caught']} SR: {op_c['sr_before']} -> {op_c['sr_after']}")
    print(f"  ONNX parity max_abs_diff={max_parity_diff:.2e}  size={report_payload['onnx_export']['onnx_size_kb']} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
