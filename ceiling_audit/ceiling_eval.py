"""Step 1 Offline Ceiling Measurement & Anti-Confounder Audit for Laya Address VN.

Evaluates on real DEV order data (strictly local, zero network calls):
  - Rule D baseline (full dataset & residual slice remaining after Rule D)
  - Existing Way A v2 & Way B v2 gates (from columns or live via `--score-v2`)
  - Ceiling 1A: Pre-Goong Text (PII-masked word+char TF-IDF + Logistic Regression)
  - Ceiling 1B: Pre-Goong Anti-Confounder Structural (24 domain-invariant numeric features)
  - Baseline Rule: Post-Goong MIT Gazetteer Rule Baseline (deterministic zero-train baseline)
  - Ceiling 2A: Post-Goong Consistency (51 domain-invariant features, ZERO regional IDs)
  - Ceiling 2B: Post-Goong Combined (nested-OOF Pre-Goong logit + Consistency features)

Outputs comparison spreadsheets (.csv, .xlsx), PII-free JSON report, Vietnamese Markdown
executive summary, and OOF predictions CSV for Step 2 stratified 300-sample audit.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from post_goong_features import (
    POST_GOONG_CONSISTENCY_FEATURES,
    PRE_GOONG_STRUCT_FEATURES,
    VietnamAdminIndex,
    extract_feature_dataframe,
)
from schema_io import (
    DatasetSchema,
    load_dataset,
    mask_pii_for_features,
    mask_pii_for_preview,
    normalize_phone_key,
    parse_bool_gate,
    parse_is_return,
    populate_goong_columns,
    resolve_schema,
    strip_accents,
    write_csv_utf8_sig,
)


def sanitize_json(obj: Any) -> Any:
    """Recursively convert numpy types and replace NaN/Inf with None for strict JSON."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        fv = float(obj)
        return None if (math.isnan(fv) or math.isinf(fv)) else fv
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    return str(obj)


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
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


def _ratio(lost: int, caught: int) -> Optional[float]:
    if caught <= 0:
        return None
    return float(lost / caught)


def evaluate_binary_gate(
    y_true: np.ndarray,
    blocked_mask: np.ndarray,
    base_rate: float,
) -> Dict[str, Any]:
    """Compute exact blocked/caught/lost/ratio metrics for a fixed boolean block mask."""
    n = len(y_true)
    blocked = int(np.sum(blocked_mask))
    caught = int(np.sum(blocked_mask & (y_true == 1)))
    lost = int(np.sum(blocked_mask & (y_true == 0)))
    total_ret = int(np.sum(y_true == 1))
    prec = float(caught / blocked) if blocked > 0 else 0.0
    rec = float(caught / total_ret) if total_ret > 0 else 0.0
    lift = float(prec / base_rate) if base_rate > 0 and blocked > 0 else None
    return {
        "n_eval": n,
        "blocked": blocked,
        "block_rate": round(blocked / max(1, n), 4),
        "caught_returns": caught,
        "lost_deliveries": lost,
        "ratio_lost_per_caught": round(lost / caught, 4) if caught > 0 else None,
        "precision_return": round(prec, 4),
        "recall_return": round(rec, 4),
        "lift_vs_base": round(lift, 3) if lift is not None else None,
    }


def bootstrap_ratio_ci(
    y_true: np.ndarray,
    blocked_mask: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float]]:
    if int(np.sum(blocked_mask)) == 0 or int(np.sum(blocked_mask & (y_true == 1))) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    n = len(y_true)
    ratios: List[float] = []
    # Vectorized bootstrap on counts of (caught_in_block, lost_in_block, not_blocked)
    caught_obs = int(np.sum(blocked_mask & (y_true == 1)))
    lost_obs = int(np.sum(blocked_mask & (y_true == 0)))
    other_obs = n - caught_obs - lost_obs
    probs = np.array([caught_obs, lost_obs, other_obs], dtype=np.float64) / max(1, n)
    draws = rng.multinomial(n, probs, size=n_boot)
    for c_b, l_b, _ in draws:
        if c_b > 0:
            ratios.append(float(l_b / c_b))
    if not ratios:
        return None, None
    return round(float(np.percentile(ratios, 2.5)), 4), round(float(np.percentile(ratios, 97.5)), 4)


def evaluate_score_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_ratio: float = 1.5,
    min_caught: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute realizable threshold curve (grouped at unique score values) + CIs."""
    n = len(y_true)
    total_ret = int(np.sum(y_true == 1))
    base_rate = float(total_ret / max(1, n))

    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = y_true[order]

    # Boundary indices where score strictly changes (realizable >= threshold points)
    change_idx = np.flatnonzero(np.diff(s_sorted) != 0)
    thresh_ends = np.append(change_idx, n - 1)

    cum_ret = np.cumsum(y_sorted == 1)
    cum_del = np.cumsum(y_sorted == 0)

    curve: List[Dict[str, Any]] = []
    for end_i in thresh_ends:
        thr = float(s_sorted[end_i])
        blk = int(end_i + 1)
        caught = int(cum_ret[end_i])
        lost = int(cum_del[end_i])
        ratio = float(lost / caught) if caught > 0 else None
        prec = float(caught / blk) if blk > 0 else 0.0
        rec = float(caught / max(1, total_ret))
        lift = float(prec / base_rate) if base_rate > 0 else 0.0
        curve.append({
            "threshold": round(thr, 6),
            "blocked": blk,
            "block_rate": round(blk / max(1, n), 4),
            "caught_returns": caught,
            "lost_deliveries": lost,
            "ratio_lost_per_caught": round(ratio, 4) if ratio is not None else None,
            "precision_return": round(prec, 4),
            "recall_return": round(rec, 4),
            "lift_vs_base": round(lift, 3),
        })

    def _best_at_min_caught(mc: int) -> Optional[Dict[str, Any]]:
        cands = [pt for pt in curve if pt["caught_returns"] >= mc and pt["ratio_lost_per_caught"] is not None]
        if not cands:
            return None
        best = min(cands, key=lambda p: (p["ratio_lost_per_caught"], -p["caught_returns"]))
        best_copy = dict(best)
        blk_mask = scores >= best["threshold"]
        ci_lo, ci_hi = bootstrap_ratio_ci(y_true, blk_mask, seed=seed)
        best_copy["ratio_ci95"] = [ci_lo, ci_hi]
        return best_copy

    best_min_caught = _best_at_min_caught(min_caught)
    best_10plus = _best_at_min_caught(10)

    target_cands = [
        pt for pt in curve
        if pt["caught_returns"] > 0
        and pt["ratio_lost_per_caught"] is not None
        and pt["ratio_lost_per_caught"] <= target_ratio
    ]
    max_caught_target = (
        max(target_cands, key=lambda p: (p["caught_returns"], -p["lost_deliveries"]))
        if target_cands
        else None
    )

    checkpoints: Dict[str, Optional[Dict[str, Any]]] = {}
    for vol_c in (30, 50, 100, 200):
        cands = [pt for pt in curve if pt["caught_returns"] >= vol_c]
        checkpoints[f"caught_ge_{vol_c}"] = cands[0] if cands else None
    for pct in (0.01, 0.02, 0.05):
        target_blk = max(1, int(round(n * pct)))
        cands = [pt for pt in curve if pt["blocked"] >= target_blk]
        checkpoints[f"top_{int(pct * 100)}pct"] = cands[0] if cands else None

    try:
        roc_auc = round(float(roc_auc_score(y_true, scores)), 4) if len(np.unique(y_true)) == 2 else None
    except Exception:
        roc_auc = None
    try:
        pr_auc = round(float(average_precision_score(y_true, scores)), 4) if total_ret > 0 else None
    except Exception:
        pr_auc = None

    clipped_p = np.clip(scores, 0.0, 1.0)
    brier = round(float(brier_score_loss(y_true, clipped_p)), 4)
    ece = round(compute_ece(y_true, clipped_p), 4)

    # Compact sampled curve (~20 points across quantiles + key points) for JSON/Excel
    sampled_curve: List[Dict[str, Any]] = []
    if curve:
        step = max(1, len(curve) // 25)
        sampled_curve = curve[::step]
        if curve[-1] not in sampled_curve:
            sampled_curve.append(curve[-1])

    return {
        "n_eval": n,
        "total_returns": total_ret,
        "base_return_rate": round(base_rate, 4),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "ece": ece,
        "best_ratio_at_min_caught": best_min_caught,
        "best_ratio_at_10plus": best_10plus,
        "max_caught_at_target_ratio": max_caught_target,
        "checkpoints": checkpoints,
        "sampled_curve": sampled_curve,
    }


def build_folds(
    y: np.ndarray,
    group_keys: Sequence[str],
    timestamps: Optional[pd.Series],
    requested_mode: str,
    n_splits: int,
    seed: int,
) -> Tuple[str, List[Tuple[np.ndarray, np.ndarray]], np.ndarray, List[np.ndarray]]:
    """Build strict anti-leakage CV folds.

    Returns:
      (effective_mode, folds_list, oof_eval_mask, fold_unseen_phone_masks_aligned_with_oof)
    """
    n = len(y)
    g_arr = np.asarray(group_keys, dtype=object)

    eff_mode = requested_mode
    parsed_ts: Optional[pd.Series] = None
    if timestamps is not None:
        parsed_ts = pd.to_datetime(timestamps, errors="coerce")
        valid_ts_frac = float(parsed_ts.notna().mean()) if n > 0 else 0.0
        n_unique_ts = int(parsed_ts.dropna().nunique())
        if eff_mode == "auto":
            if valid_ts_frac >= 0.80 and n_unique_ts >= max(10, n_splits + 2):
                eff_mode = "time"
            else:
                eff_mode = "group"
        elif eff_mode == "time" and (valid_ts_frac < 0.50 or n_unique_ts < n_splits + 2):
            print("[WARN] Cột thời gian thiếu/trùng quá nhiều — tự động chuyển sang GroupKFold.", file=sys.stderr)
            eff_mode = "group"
    elif eff_mode == "auto":
        eff_mode = "group"

    if eff_mode == "time" and parsed_ts is not None:
        # Expanding chronological blocks: blocks 0..k-1 -> test block k (k = 1..n_splits)
        fill_val = parsed_ts.median()
        ts_filled = parsed_ts.fillna(fill_val)
        sorted_pos = np.argsort(ts_filled.to_numpy(), kind="mergesort")
        blocks = np.array_split(sorted_pos, n_splits + 1)
        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        oof_mask = np.zeros(n, dtype=bool)
        unseen_phone_mask = np.zeros(n, dtype=bool)
        train_acc: List[np.ndarray] = []
        seen_phones: set[str] = set()
        for k in range(n_splits + 1):
            blk = blocks[k]
            if k == 0:
                train_acc.append(blk)
                seen_phones.update(str(g_arr[idx]) for idx in blk if str(g_arr[idx]).startswith("p_"))
                continue
            tr_idx = np.concatenate(train_acc)
            va_idx = blk
            folds.append((tr_idx, va_idx))
            oof_mask[va_idx] = True
            for idx in va_idx:
                gk = str(g_arr[idx])
                if gk.startswith("p_") and gk not in seen_phones:
                    unseen_phone_mask[idx] = True
            train_acc.append(blk)
            seen_phones.update(str(g_arr[idx]) for idx in blk if str(g_arr[idx]).startswith("p_"))
        return "time", folds, oof_mask, [unseen_phone_mask]

    if eff_mode == "group":
        uniq_groups = len(set(group_keys))
        actual_splits = min(n_splits, max(2, uniq_groups))
        sgkf = StratifiedGroupKFold(n_splits=actual_splits, shuffle=True, random_state=seed)
        folds = []
        oof_mask = np.ones(n, dtype=bool)
        for tr_idx, va_idx in sgkf.split(np.zeros(n), y, groups=g_arr):
            # Strict check: 0 overlap on group_keys
            tr_g = {g_arr[i] for i in tr_idx}
            va_g = {g_arr[i] for i in va_idx}
            overlap = tr_g & va_g
            if overlap:
                raise AssertionError(f"Group leakage detected across CV folds: {len(overlap)} overlapping groups")
            folds.append((tr_idx, va_idx))
        return "group", folds, oof_mask, []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(n), y))
    return "stratified", folds, np.ones(n, dtype=bool), []


def _make_text_vectorizer() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=25000, min_df=3, sublinear_tf=True)),
    ])


def _fit_predict_struct_ensemble(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    seed: int,
    hgb_iters: int = 120,
    hgb_depth: int = 3,
) -> np.ndarray:
    if len(np.unique(y_tr)) < 2:
        return np.full(len(X_va), float(np.mean(y_tr)), dtype=np.float64)

    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500, solver="lbfgs"))
    lr.fit(X_tr, y_tr)
    p_lr = lr.predict_proba(X_va)[:, 1]

    hgb = HistGradientBoostingClassifier(
        max_iter=hgb_iters,
        max_depth=hgb_depth,
        learning_rate=0.05,
        l2_regularization=2.0,
        min_samples_leaf=max(15, min(35, len(y_tr) // 30)),
        random_state=seed,
    )
    hgb.fit(X_tr, y_tr)
    p_hgb = hgb.predict_proba(X_va)[:, 1]
    return 0.5 * p_lr + 0.5 * p_hgb


def compute_post_goong_rule_score(post_df: pd.DataFrame, pre_df: pd.DataFrame) -> np.ndarray:
    """Deterministic zero-training Post-Goong heuristic score in [0.05 .. 0.95]."""
    n = len(post_df)
    scores = np.full(n, 0.05, dtype=np.float64)
    has_chat = pre_df["has_chat_or_food_cue"].to_numpy() >= 0.5
    raw_ward = pre_df["raw_gaz_ward_hits"].to_numpy()
    ward_missing_g = post_df["goong_has_ward"].to_numpy() < 0.5
    contradict = (post_df["raw_prov_contradicts_goong"].to_numpy() >= 0.5) | (
        post_df["raw_dist_contradicts_goong"].to_numpy() >= 0.5
    )
    ambig_halluc = post_df["ambiguous_ward_ungrounded_parent"].to_numpy() >= 0.5
    halluc2 = post_df["hallucinated_levels_count"].to_numpy() >= 2.0
    ward_halluc = post_df["ward_hallucinated"].to_numpy() >= 0.5

    scores = np.where(has_chat & (raw_ward == 0), np.maximum(scores, 0.45), scores)
    scores = np.where(ward_halluc, np.maximum(scores, 0.60), scores)
    scores = np.where(ward_missing_g, np.maximum(scores, 0.70), scores)
    scores = np.where(halluc2, np.maximum(scores, 0.80), scores)
    scores = np.where(ambig_halluc, np.maximum(scores, 0.88), scores)
    scores = np.where(contradict, np.maximum(scores, 0.95), scores)
    return scores


def run_confounder_audit(
    df_oof: pd.DataFrame,
    y_oof: np.ndarray,
    masked_texts_oof: List[str],
    g_wards_oof: List[str],
    g_dists_oof: List[str],
    g_provs_oof: List[str],
    buckets_oof: List[str],
    X_post_all_oof: Optional[np.ndarray],
    post_feat_names: List[str],
    oof_scores: Dict[str, np.ndarray],
    curves: Dict[str, Dict[str, Any]],
    unseen_phone_oof_mask: Optional[np.ndarray],
    target_ratio: float,
    min_caught: int,
    has_goong: bool,
    index: VietnamAdminIndex,
    raw_preview: bool,
    mask_names: bool,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run TF-IDF keyword audit, Post-Goong feature weight audit, geographic lift audit, and unseen-phone audit."""
    audit: Dict[str, Any] = {}

    # 1. Top 25 TF-IDF terms pushing toward return
    prov_keywords = set(index.provinces) | {"hcm", "tphcm", "hn", "tphn", "sg", "saigon", "danang"}
    note_keywords = {"gio hanh chinh", "hanh chinh", "ngoai gio", "goi truoc", "ship", "bom", "cod", "sdt"}
    try:
        vec = _make_text_vectorizer()
        Xt = vec.fit_transform([strip_accents(t).lower() for t in masked_texts_oof])
        lr = LogisticRegression(C=1.0, max_iter=400, solver="lbfgs")
        lr.fit(Xt, y_oof)
        names = vec.get_feature_names_out()
        coefs = lr.coef_[0]
        top_idx = np.argsort(-coefs)[:25]
        top_terms: List[Dict[str, Any]] = []
        prov_term_count = 0
        note_term_count = 0
        for idx in top_idx:
            term = str(names[idx])
            clean_term = term.split("__", 1)[-1].strip()
            is_prov = any(pk in clean_term for pk in prov_keywords if len(pk) >= 4)
            is_note = any(nk in clean_term for nk in note_keywords)
            if is_prov:
                prov_term_count += 1
            if is_note:
                note_term_count += 1
            top_terms.append({
                "feature": term,
                "weight": round(float(coefs[idx]), 4),
                "flag_geo_term": is_prov,
                "flag_note_term": is_note,
            })
        audit["top_tfidf_return_terms"] = top_terms
        audit["tfidf_geo_term_hits_in_top25"] = prov_term_count
        audit["tfidf_note_term_hits_in_top25"] = note_term_count
    except Exception as exc:
        audit["top_tfidf_return_terms"] = []
        audit["tfidf_audit_error"] = str(exc)

    # 2. Standardized LogReg coefficients for Post-Goong Consistency features
    if has_goong and X_post_all_oof is not None and len(np.unique(y_oof)) == 2:
        sc = StandardScaler()
        Xs = sc.fit_transform(X_post_all_oof)
        lr_p = LogisticRegression(C=0.5, max_iter=500)
        lr_p.fit(Xs, y_oof)
        w = lr_p.coef_[0]
        ranked = sorted(
            [{"feature": n, "std_coef": round(float(v), 4)} for n, v in zip(post_feat_names, w)],
            key=lambda x: -abs(float(x["std_coef"])),
        )
        audit["top_post_goong_std_weights"] = ranked[:20]

    # 3. Geographic concentration audit across top ceiling models
    geo_audit: Dict[str, Any] = {}
    geo_confounder_warn = False
    if g_provs_oof and any(p.strip() for p in g_provs_oof):
        canon_provs = []
        for p in g_provs_oof:
            _pid, canon = index.canonicalize_province(p)
            canon_provs.append(canon or "unknown")
        prov_series = pd.Series(canon_provs)
        base_dist = prov_series.value_counts(normalize=True).to_dict()

        for model_key in ("pre_goong_text", "post_goong_consistency", "post_goong_combined"):
            if model_key not in oof_scores or model_key not in curves:
                continue
            pt = curves[model_key].get("best_ratio_at_min_caught") or curves[model_key].get("best_ratio_at_10plus")
            if not pt:
                continue
            blk_mask = oof_scores[model_key] >= pt["threshold"]
            blk_provs = prov_series[blk_mask]
            blk_counts = blk_provs.value_counts().to_dict()
            n_blk = max(1, len(blk_provs))
            top5 = []
            max_lift = 1.0
            flagged_prov = False
            for pv, cnt in list(blk_counts.items())[:5]:
                sh_blk = cnt / n_blk
                sh_base = base_dist.get(pv, 1e-6)
                lift = sh_blk / max(sh_base, 1e-4)
                if cnt >= 10 and sh_base >= 0.005 and lift > max_lift:
                    max_lift = lift
                if cnt >= 10 and sh_blk > 0.35 and lift > 2.5:
                    flagged_prov = True
                top5.append({
                    "province": pv,
                    "blocked_count": int(cnt),
                    "share_in_blocked": round(sh_blk, 4),
                    "share_in_baseline": round(sh_base, 4),
                    "over_rep_lift": round(lift, 2),
                })
            if flagged_prov and model_key == "pre_goong_text":
                geo_confounder_warn = True
            geo_audit[model_key] = {
                "blocked_n": int(np.sum(blk_mask)),
                "top5_share_sum": round(sum(x["share_in_blocked"] for x in top5), 4),
                "max_province_lift": round(max_lift, 2),
                "geographic_concentration_flag": flagged_prov,
                "top5_provinces": top5,
            }
    audit["geographic_concentration"] = geo_audit

    # 4. Signal divergence test (TF-IDF hits <=1.5 while domain-invariant features fail)
    def _hits_target(key: str) -> bool:
        c = curves.get(key, {})
        pt = c.get("best_ratio_at_min_caught")
        return bool(pt and pt["ratio_lost_per_caught"] is not None and pt["ratio_lost_per_caught"] <= target_ratio)

    text_hits = _hits_target("pre_goong_text")
    pre_struct_hits = _hits_target("pre_goong_struct")
    post_cons_hits = _hits_target("post_goong_consistency")
    divergence_warn = bool(text_hits and (not pre_struct_hits) and (not post_cons_hits))
    audit["signal_divergence_warning"] = divergence_warn
    audit["geographic_confounder_warning"] = geo_confounder_warn

    # 5. Unseen-phone audit (time split)
    unseen_audit: Dict[str, Any] = {}
    if unseen_phone_oof_mask is not None and int(np.sum(unseen_phone_oof_mask)) >= 20:
        seen_mask = ~unseen_phone_oof_mask
        base_unseen = float(np.mean(y_oof[unseen_phone_oof_mask]))
        base_seen = float(np.mean(y_oof[seen_mask])) if np.any(seen_mask) else 0.0
        unseen_audit["unseen_phone_n"] = int(np.sum(unseen_phone_oof_mask))
        unseen_audit["unseen_phone_base_return_rate"] = round(base_unseen, 4)
        unseen_audit["seen_phone_base_return_rate"] = round(base_seen, 4)
        models_unseen = {}
        for mkey, sc_arr in oof_scores.items():
            pt = curves.get(mkey, {}).get("best_ratio_at_min_caught") or curves.get(mkey, {}).get("best_ratio_at_10plus")
            if not pt:
                continue
            blk = sc_arr >= pt["threshold"]
            models_unseen[mkey] = {
                "unseen_phone": evaluate_binary_gate(y_oof[unseen_phone_oof_mask], blk[unseen_phone_oof_mask], base_unseen),
                "seen_or_text_key": evaluate_binary_gate(y_oof[seen_mask], blk[seen_mask], base_seen) if np.any(seen_mask) else None,
            }
        unseen_audit["by_model"] = models_unseen
    audit["unseen_phone_audit"] = unseen_audit

    # 6. Top-20 blocked rows preview across key models
    preview_rows: List[Dict[str, Any]] = []
    for model_key in ("post_goong_consistency", "post_goong_combined", "pre_goong_text", "pre_goong_struct"):
        if model_key not in oof_scores:
            continue
        sc = oof_scores[model_key]
        top20 = np.argsort(-sc, kind="mergesort")[:20]
        for rank, r_i in enumerate(top20, start=1):
            raw_txt = str(df_oof["_raw_text_orig"].iloc[r_i])
            prev_txt = raw_txt if raw_preview else mask_pii_for_preview(raw_txt, mask_names=mask_names)
            preview_rows.append({
                "model": model_key,
                "rank": rank,
                "score": round(float(sc[r_i]), 4),
                "is_return": int(y_oof[r_i]),
                "diagnostic_bucket": buckets_oof[r_i],
                "raw_text_preview": prev_txt,
                "goong_ward": g_wards_oof[r_i],
                "goong_district": g_dists_oof[r_i],
                "goong_province": g_provs_oof[r_i],
            })

    preview_df = pd.DataFrame(preview_rows)
    return audit, preview_df


def determine_verdict(
    curves: Dict[str, Dict[str, Any]],
    confounder_audit: Dict[str, Any],
    target_ratio: float,
    min_caught: int,
) -> Dict[str, str]:
    def _hits(k: str) -> bool:
        pt = curves.get(k, {}).get("best_ratio_at_min_caught")
        return bool(pt and pt["ratio_lost_per_caught"] is not None and pt["ratio_lost_per_caught"] <= target_ratio)

    post_cons_ok = _hits("post_goong_consistency")
    post_comb_ok = _hits("post_goong_combined")
    pre_struct_ok = _hits("pre_goong_struct")
    pre_text_ok = _hits("pre_goong_text")
    geo_warn = bool(confounder_audit.get("geographic_confounder_warning", False))
    div_warn = bool(confounder_audit.get("signal_divergence_warning", False))

    if post_cons_ok:
        return {
            "code": "GO_POST_GOONG",
            "headline": f"CÓ DƯ ĐỊA RÕ RỆT Ở HẬU-GOONG (đạt <= {target_ratio} ở >= {min_caught} đơn hoàn chỉ bằng tín hiệu đối khớp sạch).",
            "recommendation": "Hoàn tất Bước 2 gán nhãn 300 mẫu để chốt bản đồ lỗi, sau đó build v3 làm cổng xác minh Hậu-Goong (raw_text + goong_output).",
        }
    if post_comb_ok and not geo_warn:
        return {
            "code": "GO_POST_GOONG_COMBINED",
            "headline": f"CÓ DƯ ĐỊA Ở HẬU-GOONG KẾT HỢP (đạt <= {target_ratio} ở >= {min_caught} đơn hoàn).",
            "recommendation": "Soi Top-20 ca bị chặn và chạy Bước 2 gán nhãn 300 mẫu để kiểm tra xem phần cải thiện đến từ lỗi Goong hay từ vựng đặc thù.",
        }
    if pre_struct_ok:
        return {
            "code": "GO_PRE_GOONG",
            "headline": f"CÓ DƯ ĐỊA Ở TIỀN-GOONG (đạt <= {target_ratio} ở >= {min_caught} đơn hoàn trên đặc trưng cấu trúc sạch).",
            "recommendation": "Chạy tiếp Bước 2 trên 300 mẫu rồi huấn luyện bản mở rộng từ lỗi thật.",
        }
    if pre_text_ok and (div_warn or geo_warn):
        return {
            "code": "CAUTION_CONFOUNDER_SUSPECTED",
            "headline": f"CẢNH BÁO NHIỄU (NHIỄU VÙNG/GHI CHÚ): Chỉ Pre-Goong TF-IDF chạm <= {target_ratio}, còn đặc trưng sạch (Pre-Struct & Post-Goong Consistency) KHÔNG ĐẠT.",
            "recommendation": "Chưa vội build v3! Kiểm tra sheet Top_Blocked_Preview & top TF-IDF terms + làm Bước 2 (300 mẫu) để xác minh có đang chặn nhầm tỉnh COD hoàn cao hay không.",
        }
    if pre_text_ok:
        return {
            "code": "GO_PRE_GOONG_TEXT_VERIFY",
            "headline": f"Pre-Goong TF-IDF chạm <= {target_ratio} (chưa thấy cảnh báo vùng miền tự động).",
            "recommendation": "Soi kỹ Top-20 ca bị chặn và gán nhãn 300 mẫu ở Bước 2 trước khi quyết định build v3.",
        }
    return {
        "code": "NO_GO_FOCUS_FORM_PREVENTION",
        "headline": f"KHÔNG ĐỦ DƯ ĐỊA CỔNG MÔ HÌNH: Mọi đường đo trần đều không chạm <= {target_ratio} ở >= {min_caught} đơn hoàn.",
        "recommendation": "Dừng tối ưu cổng phân loại sau đơn; chuyển nguồn lực sang phòng ngừa từ gốc ngay ở form Webcake và tin nhắn Zalo.",
    }


def write_comparison_xlsx(
    path: Path,
    summary_df: pd.DataFrame,
    curves_df: pd.DataFrame,
    confounder_df: pd.DataFrame,
    preview_df: pd.DataFrame,
) -> None:
    wb = openpyxl.Workbook()
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=10)

    def _populate_sheet(ws, df_in: pd.DataFrame, freeze: str = "A2") -> None:
        ws.freeze_panes = freeze
        if df_in.empty:
            ws.append(["(Không có dữ liệu cho bảng này)"])
            return
        cols = list(df_in.columns)
        ws.append(cols)
        for col_i in range(1, len(cols) + 1):
            cell = ws.cell(row=1, column=col_i)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for row in df_in.itertuples(index=False, name=None):
            ws.append([None if pd.isna(v) else v for v in row])
        sample_rows = list(df_in.iloc[:200].itertuples(index=False, name=None))
        for col_i, cname in enumerate(cols, start=1):
            max_len = max([len(str(cname))] + [len(str(r[col_i - 1] or "")) for r in sample_rows])
            ws.column_dimensions[get_column_letter(col_i)].width = min(48, max(12, max_len + 3))

    ws1 = wb.active
    ws1.title = "Summary_Comparison"
    _populate_sheet(ws1, summary_df)

    ws2 = wb.create_sheet("Operating_Curves")
    _populate_sheet(ws2, curves_df)

    ws3 = wb.create_sheet("Confounder_Audit")
    _populate_sheet(ws3, confounder_df)

    ws4 = wb.create_sheet("Top_Blocked_Preview")
    _populate_sheet(ws4, preview_df)

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Laya Address VN — Step 1 Offline Ceiling Measurement Tool")
    p.add_argument("--data", required=True, help="Đường dẫn file data DEV (.csv, .tsv, .xlsx, .jsonl, .parquet)")
    p.add_argument("--sheet", default=None, help="Tên sheet Excel nếu file .xlsx có nhiều sheet")
    p.add_argument("--output-dir", default=None, help="Thư mục xuất kết quả")

    # Column overrides
    p.add_argument("--order-id-col", default=None)
    p.add_argument("--text-col", default=None)
    p.add_argument("--outcome-col", default=None)
    p.add_argument("--rule-d-col", default=None)
    p.add_argument("--rule-d-is-block", action="store_true", help="Đánh dấu nếu cột Luật D mang nghĩa 1=BỊ CHẶN")
    p.add_argument("--phone-col", default=None)
    p.add_argument("--time-col", default=None)
    p.add_argument("--goong-ward-col", default=None)
    p.add_argument("--goong-district-col", default=None)
    p.add_argument("--goong-province-col", default=None)
    p.add_argument("--goong-score-col", default=None)
    p.add_argument("--goong-formatted-col", default=None)
    p.add_argument("--way-a-col", default=None)
    p.add_argument("--way-b-col", default=None)
    p.add_argument("--return-values", default=None, help="Danh sách giá trị hoàn phân cách dấu phẩy")
    p.add_argument("--delivered-values", default=None, help="Danh sách giá trị giao phân cách dấu phẩy")

    # Evaluation controls
    p.add_argument("--split-mode", choices=["auto", "time", "group", "stratified"], default="auto")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-ratio", type=float, default=1.5)
    p.add_argument("--min-caught-returns", type=int, default=30)
    p.add_argument("--score-v2", action="store_true", help="Chạy trực tiếp Way A v2 & Way B v2 nếu CSV chưa có cột sẵn")
    p.add_argument("--mask-names", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--raw-preview", action="store_true", help="Không che SĐT trong sheet Top_Blocked_Preview")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parent / "outputs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_dataset(args.data, sheet_name=args.sheet)
    overrides = {
        "order_id": args.order_id_col,
        "text": args.text_col,
        "outcome": args.outcome_col,
        "rule_d": args.rule_d_col,
        "rule_d_is_block": args.rule_d_is_block,
        "phone": args.phone_col,
        "timestamp": args.time_col,
        "goong_ward": args.goong_ward_col,
        "goong_district": args.goong_district_col,
        "goong_province": args.goong_province_col,
        "goong_score": args.goong_score_col,
        "goong_formatted": args.goong_formatted_col,
        "way_a": args.way_a_col,
        "way_b": args.way_b_col,
    }
    schema = resolve_schema(df_raw, overrides=overrides)

    ret_vals = [v.strip() for v in args.return_values.split(",")] if args.return_values else None
    del_vals = [v.strip() for v in args.delivered_values.split(",")] if args.delivered_values else None
    is_return_all = parse_is_return(df_raw[schema.outcome], return_values=ret_vals, delivered_values=del_vals)
    valid_mask = (is_return_all == 0) | (is_return_all == 1)

    df_valid = df_raw.loc[valid_mask].copy().reset_index(drop=True)
    y_valid = is_return_all[valid_mask].astype(np.int64)
    if len(df_valid) == 0:
        raise ValueError("Không có dòng nào có nhãn giao/hoàn hợp lệ sau khi chuẩn hóa cột outcome.")

    if schema.order_id and schema.order_id in df_valid.columns:
        df_valid["_order_id"] = df_valid[schema.order_id].astype(str)
    else:
        df_valid["_order_id"] = [f"ROW_{i + 1:06d}" for i in range(len(df_valid))]

    if schema.rule_d and schema.rule_d in df_valid.columns:
        rule_d_pass_valid = parse_bool_gate(df_valid[schema.rule_d], invert=schema.rule_d_is_block)
    else:
        rule_d_pass_valid = np.ones(len(df_valid), dtype=bool)

    total_valid = len(df_valid)
    total_ret = int(np.sum(y_valid == 1))
    total_del = int(np.sum(y_valid == 0))
    overall_ret_rate = total_ret / max(1, total_valid)

    rd_stats = evaluate_binary_gate(y_valid, ~rule_d_pass_valid, overall_ret_rate)

    # Residual slice passed by Rule D
    df_res = df_valid.loc[rule_d_pass_valid].copy().reset_index(drop=True)
    y_res = y_valid[rule_d_pass_valid]
    if len(df_res) < 20 or int(np.sum(y_res == 1)) < 5:
        raise ValueError(
            f"Tập còn lại sau Luật D quá ít để kiểm chéo (n={len(df_res)}, returns={int(np.sum(y_res == 1))})."
        )

    res_ret = int(np.sum(y_res == 1))
    res_rate = res_ret / len(y_res)
    nat_ratio = (1.0 - res_rate) / res_rate if res_rate > 0 else None
    req_prec = 1.0 / (1.0 + args.target_ratio)
    req_lift = (req_prec / res_rate) if res_rate > 0 else None

    raw_texts_res = [
        "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
        for v in df_res[schema.text].tolist()
    ]
    df_res["_raw_text_orig"] = raw_texts_res
    masked_texts_res = [mask_pii_for_features(t, mask_names=args.mask_names) for t in raw_texts_res]

    phone_vals_res = (
        df_res[schema.phone].tolist() if (schema.phone and schema.phone in df_res.columns) else [None] * len(df_res)
    )
    group_keys_res = [normalize_phone_key(ph, txt) for ph, txt in zip(phone_vals_res, raw_texts_res)]
    ts_series_res = df_res[schema.timestamp] if (schema.timestamp and schema.timestamp in df_res.columns) else None

    eff_split_mode, folds, oof_mask, unseen_masks = build_folds(
        y=y_res,
        group_keys=group_keys_res,
        timestamps=ts_series_res,
        requested_mode=args.split_mode,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    unseen_full_mask = unseen_masks[0] if unseen_masks else None

    # Feature extraction (strictly row-local + static MIT gazetteer)
    index = VietnamAdminIndex()
    pre_df, post_df, diag_buckets = extract_feature_dataframe(df_res, schema, index=index, mask_names=args.mask_names)
    X_pre = pre_df.to_numpy(dtype=np.float64)
    X_post_only = post_df.to_numpy(dtype=np.float64)
    X_post_all = np.hstack([X_pre, X_post_only])

    # Optional existing or live Way A v2 & Way B v2 gates
    way_a_pass_res: Optional[np.ndarray] = None
    way_b_pass_res: Optional[np.ndarray] = None
    if schema.way_a and schema.way_a in df_res.columns:
        way_a_pass_res = parse_bool_gate(df_res[schema.way_a])
    if schema.way_b and schema.way_b in df_res.columns:
        way_b_pass_res = parse_bool_gate(df_res[schema.way_b])

    if args.score_v2 and (way_a_pass_res is None or way_b_pass_res is None):
        from v2_scorer import score_way_a_v2_batch, score_way_b_v2_batch

        if way_a_pass_res is None:
            scored_a = score_way_a_v2_batch(raw_texts_res, mode="hybrid")
            if scored_a is not None:
                way_a_pass_res = np.asarray(scored_a["pass"], dtype=bool)
        if way_b_pass_res is None:
            scored_b = score_way_b_v2_batch(raw_texts_res, mode="hybrid")
            way_b_pass_res = np.asarray(scored_b["pass"], dtype=bool)

    # Allocate OOF score arrays on df_res
    n_res = len(df_res)
    oof_pre_text = np.full(n_res, np.nan, dtype=np.float64)
    oof_pre_text_logit = np.full(n_res, np.nan, dtype=np.float64)
    oof_pre_struct = np.full(n_res, np.nan, dtype=np.float64)
    oof_post_cons = np.full(n_res, np.nan, dtype=np.float64)
    oof_post_comb = np.full(n_res, np.nan, dtype=np.float64)

    text_arr_norm = np.asarray([strip_accents(t).lower() for t in masked_texts_res], dtype=object)
    g_arr_res = np.asarray(group_keys_res, dtype=object)

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        y_tr = y_res[tr_idx]

        # 1A. Pre-Goong Text TF-IDF + LogisticRegression
        if len(np.unique(y_tr)) < 2:
            p_txt_va = np.full(len(va_idx), float(np.mean(y_tr)), dtype=np.float64)
        else:
            vec = _make_text_vectorizer()
            Xt_tr = vec.fit_transform(text_arr_norm[tr_idx])
            Xt_va = vec.transform(text_arr_norm[va_idx])
            lr_txt = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
            lr_txt.fit(Xt_tr, y_tr)
            p_txt_va = lr_txt.predict_proba(Xt_va)[:, 1]
        oof_pre_text[va_idx] = p_txt_va
        clipped_va = np.clip(p_txt_va, 1e-5, 1.0 - 1e-5)
        oof_pre_text_logit[va_idx] = np.log(clipped_va / (1.0 - clipped_va))

        # 1B. Pre-Goong Anti-Confounder Structural
        oof_pre_struct[va_idx] = _fit_predict_struct_ensemble(
            X_pre[tr_idx], y_tr, X_pre[va_idx], seed=args.seed + fold_i
        )

        # 2A & 2B. Post-Goong Ceilings (only if Goong output is present)
        if schema.has_goong:
            oof_post_cons[va_idx] = _fit_predict_struct_ensemble(
                X_post_all[tr_idx], y_tr, X_post_all[va_idx], seed=args.seed + 100 + fold_i, hgb_iters=120, hgb_depth=3
            )

            # Nested inner-OOF for leakage-free stacking of Pre-Goong text logit + Consistency features
            inner_splits: List[Tuple[np.ndarray, np.ndarray]] = []
            uniq_inner_g = len(set(g_arr_res[tr_idx]))
            if eff_split_mode != "stratified" and uniq_inner_g >= 3:
                try:
                    inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=args.seed + fold_i)
                    inner_splits = list(inner_cv.split(np.zeros(len(tr_idx)), y_tr, groups=g_arr_res[tr_idx]))
                except ValueError:
                    inner_splits = []
            if not inner_splits:
                n_in = min(3, int(np.sum(y_tr == 1)), int(np.sum(y_tr == 0)))
                if n_in >= 2:
                    inner_cv = StratifiedKFold(n_splits=n_in, shuffle=True, random_state=args.seed + fold_i)
                    inner_splits = list(inner_cv.split(np.zeros(len(tr_idx)), y_tr))

            inner_logits = np.zeros(len(tr_idx), dtype=np.float64)
            for itr, iva in inner_splits:
                if len(np.unique(y_tr[itr])) < 2:
                    ip = np.full(len(iva), float(np.mean(y_tr[itr])), dtype=np.float64)
                else:
                    ivec = _make_text_vectorizer()
                    iXt_tr = ivec.fit_transform(text_arr_norm[tr_idx[itr]])
                    iXt_va = ivec.transform(text_arr_norm[tr_idx[iva]])
                    ilr = LogisticRegression(C=1.0, max_iter=400, solver="lbfgs")
                    ilr.fit(iXt_tr, y_tr[itr])
                    ip = np.clip(ilr.predict_proba(iXt_va)[:, 1], 1e-5, 1.0 - 1e-5)
                inner_logits[iva] = np.log(ip / (1.0 - ip))

            X_comb_tr = np.column_stack([inner_logits, X_post_all[tr_idx]])
            X_comb_va = np.column_stack([oof_pre_text_logit[va_idx], X_post_all[va_idx]])
            oof_post_comb[va_idx] = _fit_predict_struct_ensemble(
                X_comb_tr, y_tr, X_comb_va, seed=args.seed + 200 + fold_i, hgb_iters=120, hgb_depth=3
            )

    # Slice to identical OOF evaluation rows across all models & baselines
    y_oof = y_res[oof_mask]
    oof_base_rate = float(np.mean(y_oof))
    df_oof = df_res.loc[oof_mask].copy().reset_index(drop=True)
    masked_texts_oof = [t for t, keep in zip(masked_texts_res, oof_mask) if keep]
    buckets_oof = [b for b, keep in zip(diag_buckets, oof_mask) if keep]
    g_wards_all, g_dists_all, g_provs_all, _ = populate_goong_columns(df_res, schema)
    g_wards_oof = [w for w, keep in zip(g_wards_all, oof_mask) if keep]
    g_dists_oof = [d for d, keep in zip(g_dists_all, oof_mask) if keep]
    g_provs_oof = [p for p, keep in zip(g_provs_all, oof_mask) if keep]

    oof_scores: Dict[str, np.ndarray] = {
        "pre_goong_text": oof_pre_text[oof_mask],
        "pre_goong_struct": oof_pre_struct[oof_mask],
    }
    post_rule_scores_all: Optional[np.ndarray] = None
    if schema.has_goong:
        post_rule_scores_all = compute_post_goong_rule_score(post_df, pre_df)
        oof_scores["post_goong_rule_baseline"] = post_rule_scores_all[oof_mask]
        oof_scores["post_goong_consistency"] = oof_post_cons[oof_mask]
        oof_scores["post_goong_combined"] = oof_post_comb[oof_mask]

    curves: Dict[str, Dict[str, Any]] = {}
    for mkey, sc_arr in oof_scores.items():
        curves[mkey] = evaluate_score_curve(
            y_oof,
            sc_arr,
            target_ratio=args.target_ratio,
            min_caught=args.min_caught_returns,
            seed=args.seed,
        )

    binary_baselines: Dict[str, Any] = {}
    if way_a_pass_res is not None:
        binary_baselines["way_a_v2_hybrid"] = evaluate_binary_gate(
            y_oof, ~way_a_pass_res[oof_mask], oof_base_rate
        )
    if way_b_pass_res is not None:
        binary_baselines["way_b_v2_hybrid"] = evaluate_binary_gate(
            y_oof, ~way_b_pass_res[oof_mask], oof_base_rate
        )
    if post_rule_scores_all is not None:
        binary_baselines["post_goong_rule_strict"] = evaluate_binary_gate(
            y_oof, post_rule_scores_all[oof_mask] >= 0.50, oof_base_rate
        )

    all_post_names = list(PRE_GOONG_STRUCT_FEATURES) + list(POST_GOONG_CONSISTENCY_FEATURES)
    unseen_oof = unseen_full_mask[oof_mask] if unseen_full_mask is not None else None
    confounder_audit, preview_df = run_confounder_audit(
        df_oof=df_oof,
        y_oof=y_oof,
        masked_texts_oof=masked_texts_oof,
        g_wards_oof=g_wards_oof,
        g_dists_oof=g_dists_oof,
        g_provs_oof=g_provs_oof,
        buckets_oof=buckets_oof,
        X_post_all_oof=(X_post_all[oof_mask] if schema.has_goong else None),
        post_feat_names=all_post_names,
        oof_scores=oof_scores,
        curves=curves,
        unseen_phone_oof_mask=unseen_oof,
        target_ratio=args.target_ratio,
        min_caught=args.min_caught_returns,
        has_goong=schema.has_goong,
        index=index,
        raw_preview=args.raw_preview,
        mask_names=args.mask_names,
    )

    verdict = determine_verdict(curves, confounder_audit, args.target_ratio, args.min_caught_returns)

    # Diagnostic error-bucket cross-tab on Rule-D-passed slice (anonymized counts only!)
    bucket_counts_all = pd.Series(diag_buckets).value_counts().to_dict()
    bucket_ret_rates: Dict[str, Dict[str, Any]] = {}
    b_arr = np.asarray(diag_buckets, dtype=object)
    for bk, cnt in bucket_counts_all.items():
        m = b_arr == bk
        rc = int(np.sum(y_res[m] == 1))
        dc = int(np.sum(y_res[m] == 0))
        bucket_ret_rates[str(bk)] = {
            "count": int(cnt),
            "returns": rc,
            "delivered": dc,
            "return_rate": round(rc / max(1, cnt), 4),
            "ratio_lost_per_caught": round(dc / rc, 3) if rc > 0 else None,
        }

    # --- Build Summary Comparison Table ---
    summary_rows: List[Dict[str, Any]] = [
        {
            "gate_or_ceiling": "Rule D (Full Valid Baseline)",
            "scope": "Full Valid Dataset",
            "n_eval": total_valid,
            "blocked": rd_stats["blocked"],
            "caught_returns": rd_stats["caught_returns"],
            "lost_deliveries": rd_stats["lost_deliveries"],
            "ratio_lost_per_caught": rd_stats["ratio_lost_per_caught"],
            "ratio_ci95_low": None,
            "ratio_ci95_high": None,
            "precision_return": rd_stats["precision_return"],
            "lift_vs_base": rd_stats["lift_vs_base"],
            "roc_auc": None,
            "pr_auc": None,
            "max_caught_at_target": None,
            "threshold_or_note": f"Residual return rate = {res_rate*100:.2f}% (natural {nat_ratio:.2f}:1)",
        }
    ]
    for bkey, bname in (
        ("way_a_v2_hybrid", "Way A v2 (hybrid incremental)"),
        ("way_b_v2_hybrid", "Way B v2 (hybrid incremental)"),
        ("post_goong_rule_strict", "Post-Goong Pure Gazetteer Rule (no training)"),
    ):
        if bkey in binary_baselines:
            st = binary_baselines[bkey]
            summary_rows.append({
                "gate_or_ceiling": bname,
                "scope": f"Rule-D Pass OOF ({eff_split_mode})",
                "n_eval": st["n_eval"],
                "blocked": st["blocked"],
                "caught_returns": st["caught_returns"],
                "lost_deliveries": st["lost_deliveries"],
                "ratio_lost_per_caught": st["ratio_lost_per_caught"],
                "ratio_ci95_low": None,
                "ratio_ci95_high": None,
                "precision_return": st["precision_return"],
                "lift_vs_base": st["lift_vs_base"],
                "roc_auc": None,
                "pr_auc": None,
                "max_caught_at_target": None,
                "threshold_or_note": "Fixed binary gate",
            })

    pretty_names = {
        "pre_goong_text": "Ceiling 1A: Pre-Goong Text (TF-IDF + LR, PII-masked)",
        "pre_goong_struct": "Ceiling 1B: Pre-Goong Structural Only (anti-confounder)",
        "post_goong_rule_baseline": "Post-Goong Gazetteer Score (deterministic curve)",
        "post_goong_consistency": "Ceiling 2A: Post-Goong Consistency Only (NO regional bias)",
        "post_goong_combined": "Ceiling 2B: Post-Goong Combined (Text logit + Consistency)",
    }
    for mkey in ("pre_goong_text", "pre_goong_struct", "post_goong_rule_baseline", "post_goong_consistency", "post_goong_combined"):
        if mkey not in curves:
            continue
        cv = curves[mkey]
        pt = cv.get("best_ratio_at_min_caught") or cv.get("best_ratio_at_10plus")
        max_c = cv.get("max_caught_at_target_ratio")
        ci = pt.get("ratio_ci95", [None, None]) if pt else [None, None]
        summary_rows.append({
            "gate_or_ceiling": pretty_names[mkey],
            "scope": f"Rule-D Pass OOF ({eff_split_mode})",
            "n_eval": cv["n_eval"],
            "blocked": pt["blocked"] if pt else 0,
            "caught_returns": pt["caught_returns"] if pt else 0,
            "lost_deliveries": pt["lost_deliveries"] if pt else 0,
            "ratio_lost_per_caught": pt["ratio_lost_per_caught"] if pt else None,
            "ratio_ci95_low": ci[0],
            "ratio_ci95_high": ci[1],
            "precision_return": pt["precision_return"] if pt else None,
            "lift_vs_base": pt["lift_vs_base"] if pt else None,
            "roc_auc": cv["roc_auc"],
            "pr_auc": cv["pr_auc"],
            "max_caught_at_target": max_c["caught_returns"] if max_c else 0,
            "threshold_or_note": f"opt@caught>={args.min_caught_returns} thr={pt['threshold']}" if pt else "Not enough caught returns",
        })

    summary_df = pd.DataFrame(summary_rows)
    write_csv_utf8_sig(summary_df, out_dir / "comparison_sheet.csv")

    # Build Curves sheet DataFrame
    curve_rows: List[Dict[str, Any]] = []
    for mkey, cv in curves.items():
        for cp_name, cp in cv.get("checkpoints", {}).items():
            if cp is not None:
                curve_rows.append({"model": mkey, "point_type": cp_name, **cp})
        for pt in cv.get("sampled_curve", []):
            curve_rows.append({"model": mkey, "point_type": "curve_grid", **pt})
    curves_df = pd.DataFrame(curve_rows)

    # Build Confounder Audit DataFrame
    conf_rows: List[Dict[str, Any]] = []
    for t in confounder_audit.get("top_tfidf_return_terms", []):
        conf_rows.append({"section": "tfidf_top_return_terms", **t})
    for w in confounder_audit.get("top_post_goong_std_weights", []):
        conf_rows.append({"section": "post_goong_std_weights", **w})
    for mk, ga in confounder_audit.get("geographic_concentration", {}).items():
        for tp in ga.get("top5_provinces", []):
            conf_rows.append({"section": f"geo_top5_{mk}", "geo_flag": ga.get("geographic_concentration_flag"), **tp})
    confounder_df = pd.DataFrame(conf_rows)

    write_comparison_xlsx(out_dir / "comparison_sheet.xlsx", summary_df, curves_df, confounder_df, preview_df)

    # Save OOF predictions CSV cleanly for Step 2 stratification
    oof_out = pd.DataFrame({
        "order_id": df_res["_order_id"],
        "is_return": y_res,
        "in_oof_eval": oof_mask.astype(int),
        "rule_d_pass": True,
        "way_a_v2_pass": way_a_pass_res if way_a_pass_res is not None else np.nan,
        "way_b_v2_pass": way_b_pass_res if way_b_pass_res is not None else np.nan,
        "diagnostic_bucket": diag_buckets,
        "oof_pre_goong_text": oof_pre_text,
        "oof_pre_goong_struct": oof_pre_struct,
        "oof_post_goong_consistency": oof_post_cons,
        "oof_post_goong_combined": oof_post_comb,
    })
    write_csv_utf8_sig(oof_out, out_dir / "oof_predictions.csv")

    # Machine-readable JSON report (PII-free!)
    report = sanitize_json({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema": schema.to_dict(),
        "split_mode": eff_split_mode,
        "n_splits": len(folds),
        "target_ratio": args.target_ratio,
        "min_caught_returns": args.min_caught_returns,
        "dataset_accounting": {
            "raw_rows": len(df_raw),
            "valid_outcome_rows": total_valid,
            "dropped_unknown_outcome_rows": len(df_raw) - total_valid,
            "overall_returns": total_ret,
            "overall_delivered": total_del,
            "overall_return_rate": round(overall_ret_rate, 4),
            "rule_d_baseline": rd_stats,
            "residual_after_rule_d": {
                "count": len(df_res),
                "returns": res_ret,
                "delivered": len(df_res) - res_ret,
                "return_rate": round(res_rate, 4),
                "natural_ratio_lost_per_caught": round(nat_ratio, 4) if nat_ratio is not None else None,
                "required_precision_for_target": round(req_prec, 4),
                "required_lift_for_target": round(req_lift, 3) if req_lift is not None else None,
            },
            "oof_eval_rows": int(np.sum(oof_mask)),
        },
        "binary_baselines": binary_baselines,
        "ceiling_curves": curves,
        "post_goong_error_buckets_residual": bucket_ret_rates,
        "confounder_audit": confounder_audit,
        "verdict": verdict,
    })
    (out_dir / "ceiling_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    # Human-readable Vietnamese Markdown Summary
    md_lines = [
        "# Báo Cáo Đo Trần (Bước 1) — Laya Address VN",
        "",
        f"- **Thời điểm chạy:** `{report['generated_at']}`",
        f"- **Cơ chế chia kiểm chéo:** `{eff_split_mode}` ({len(folds)} folds, không rò rỉ SĐT/thời gian)",
        f"- **Tổng đơn hợp lệ:** `{total_valid:,}` | Tỷ lệ hoàn gốc: `{overall_ret_rate * 100:.2f}%`",
        f"- **Lát cắt sau Luật D (residual):** `{len(df_res):,}` đơn | Tỷ lệ hoàn nền: **`{res_rate * 100:.2f}%`** (tỷ lệ tự nhiên `{nat_ratio:.2f}:1`)" if nat_ratio else f"- **Lát cắt sau Luật D:** `{len(df_res):,}` đơn",
        f"- **Yêu cầu toán học để đạt `giao mất / hoàn <= {args.target_ratio}`:** precision tập chặn phải đạt **`>= {req_prec * 100:.1f}%`** (lift **`>= {req_lift:.2f}x`** so với nền `{res_rate * 100:.2f}%`)." if req_lift else "",
        "",
        f"## Kết Luận Tự Động: `{verdict['code']}`",
        f"> **{verdict['headline']}**",
        f"> *Khuyến nghị:* {verdict['recommendation']}",
        "",
        "## Bảng So Sánh Các Cổng & Đường Đo Trần",
        "",
        "| Cổng / Mô hình | N eval | Chặn | Hoàn bắt | Giao mất | Giao mất / Hoàn | Precision | Lift | PR-AUC | Max Hoàn @ <= target |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        rat = f"{r['ratio_lost_per_caught']:.2f}" if r["ratio_lost_per_caught"] is not None else "-"
        prec_s = f"{r['precision_return'] * 100:.1f}%" if r["precision_return"] is not None else "-"
        lift_s = f"{r['lift_vs_base']:.2f}x" if r["lift_vs_base"] is not None else "-"
        pr_s = f"{r['pr_auc']:.3f}" if r["pr_auc"] is not None else "-"
        mc_s = str(r["max_caught_at_target"]) if r["max_caught_at_target"] is not None else "-"
        md_lines.append(
            f"| {r['gate_or_ceiling']} | {r['n_eval']} | {r['blocked']} | {r['caught_returns']} | {r['lost_deliveries']} | **{rat}** | {prec_s} | {lift_s} | {pr_s} | {mc_s} |"
        )

    md_lines.extend([
        "",
        "## Kiểm Tra Nhiễu (Anti-Confounder Audit)",
        f"- **Cảnh báo lệch tín hiệu (Text đạt nhưng Consistency/Struct không đạt):** `{confounder_audit.get('signal_divergence_warning')}`",
        f"- **Cảnh báo gom theo tỉnh (Geographic concentration >35% & >2.5x):** `{confounder_audit.get('geographic_confounder_warning')}`",
        f"- **Số từ khóa địa danh lọt Top-25 TF-IDF:** `{confounder_audit.get('tfidf_geo_term_hits_in_top25', 0)}/25` | **Số từ ghi chú giao nhận:** `{confounder_audit.get('tfidf_note_term_hits_in_top25', 0)}/25`",
    ])
    if confounder_audit.get("unseen_phone_audit"):
        ua = confounder_audit["unseen_phone_audit"]
        md_lines.append(
            f"- **Kiểm tra SĐT mới hoàn toàn (Time-split Unseen Phones):** `n={ua['unseen_phone_n']}`, tỷ lệ hoàn SĐT mới `{ua['unseen_phone_base_return_rate']*100:.2f}%` vs SĐT đã thấy `{ua['seen_phone_base_return_rate']*100:.2f}%`."
        )

    md_lines.extend([
        "",
        "## Phân Bố Nhóm Lỗi Hậu-Goong (Chỉ Số Đếm Ẩn Danh — An Toàn Gửi Lại)",
        "",
        "| Nhóm cấu trúc | Số đơn | Hoàn | Giao | Tỷ lệ hoàn | Giao / Hoàn |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for bk, info in sorted(bucket_ret_rates.items(), key=lambda kv: -kv[1]["count"]):
        rat_s = f"{info['ratio_lost_per_caught']:.2f}" if info["ratio_lost_per_caught"] is not None else "-"
        md_lines.append(
            f"| `{bk}` | {info['count']} | {info['returns']} | {info['delivered']} | {info['return_rate']*100:.2f}% | {rat_s} |"
        )

    (out_dir / "ceiling_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[DONE] Đã xuất báo cáo Bước 1 vào: {out_dir}")
    print(f"  -> Kết luận: {verdict['code']} — {verdict['headline']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
