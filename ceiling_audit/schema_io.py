"""Flexible dataset loader, schema resolver, label normalizer, and PII masker.

Designed for 100% offline local evaluation of real Vietnamese order logs in
Laya Address VN Step 1 (ceiling audit) and Step 2 (stratified 300-sample audit).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import unicodedata

import numpy as np
import pandas as pd


def strip_accents(text: str) -> str:
    """Strip Vietnamese diacritics while converting đ/Đ -> d/D."""
    if not text:
        return ""
    s = str(text).replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def normalize_col_name(col: str) -> str:
    """Normalize a column header to ascii snake_case for matching."""
    s = strip_accents(str(col)).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def normalize_token_str(val: object) -> str:
    """Normalize a categorical string value for robust VN/EN matching."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = strip_accents(str(val)).lower().strip()
    s = re.sub(r"[^a-z0-9.\s/_-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def sha256_short(text: str, length: int = 16) -> str:
    """Deterministic anonymized identifier hash (used for phone/text grouping)."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


# Ordered candidate lists (highest specificity first so e.g. goong_ward beats ward)
COLUMN_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "order_id": (
        "order_id",
        "ma_don",
        "madon",
        "order_code",
        "id",
        "code",
    ),
    "text": (
        "raw_text",
        "dia_chi_goc",
        "raw_address",
        "customer_address",
        "dia_chi_khach",
        "dia_chi",
        "full_address",
        "address",
        "chuoi",
        "text",
    ),
    "outcome": (
        "is_return",
        "hoan",
        "returned",
        "ket_cuc",
        "delivery_status",
        "trang_thai_giao",
        "trang_thai",
        "outcome",
        "status",
        "ket_qua",
        "label",
    ),
    "rule_d": (
        "rule_d_pass",
        "pass_rule_d",
        "luat_d_pass",
        "rule_d_ok",
        "d_pass",
        "rule_d",
        "luat_d",
        "rule_d_blocked",
        "rule_d_block",
        "luat_d_chan",
    ),
    "phone": (
        "customer_phone",
        "receiver_phone",
        "so_dien_thoai",
        "sdt",
        "phone",
        "customer_id",
        "khach_hang",
    ),
    "timestamp": (
        "created_at",
        "order_date",
        "ngay_tao",
        "thoi_gian",
        "timestamp",
        "date",
        "time",
    ),
    "goong_ward": (
        "goong_ward",
        "goong_xa",
        "goong_commune",
        "auto_ward",
        "xa_phuong",
        "phuong_xa",
        "ward",
        "phuong",
        "xa",
        "commune",
    ),
    "goong_district": (
        "goong_district",
        "goong_huyen",
        "auto_district",
        "quan_huyen",
        "district",
        "huyen",
        "quan",
    ),
    "goong_province": (
        "goong_province",
        "goong_tinh",
        "auto_province",
        "tinh_tp",
        "tinh_thanh",
        "province",
        "tinh",
        "city",
        "tp",
    ),
    "goong_score": (
        "goong_score",
        "goong_confidence",
        "goong_prob",
        "confidence",
        "score",
    ),
    "goong_formatted": (
        "goong_formatted",
        "goong_address",
        "goong_full",
        "formatted_address",
        "auto_address",
    ),
    "way_a": (
        "way_a_v2_pass",
        "way_a_pass",
        "way_a_is_address",
        "way_a_v2",
        "way_a",
    ),
    "way_b": (
        "way_b_v2_pass",
        "way_b_pass",
        "way_b_is_address",
        "way_b_v2",
        "way_b",
    ),
}


@dataclass(frozen=True)
class DatasetSchema:
    order_id: Optional[str] = None
    text: str = ""
    outcome: str = ""
    rule_d: Optional[str] = None
    rule_d_is_block: bool = False
    phone: Optional[str] = None
    timestamp: Optional[str] = None
    goong_ward: Optional[str] = None
    goong_district: Optional[str] = None
    goong_province: Optional[str] = None
    goong_score: Optional[str] = None
    goong_formatted: Optional[str] = None
    way_a: Optional[str] = None
    way_b: Optional[str] = None

    @property
    def has_goong(self) -> bool:
        return bool(
            self.goong_ward
            or self.goong_district
            or self.goong_province
            or self.goong_formatted
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "order_id": self.order_id,
            "text": self.text,
            "outcome": self.outcome,
            "rule_d": self.rule_d,
            "rule_d_is_block": self.rule_d_is_block,
            "phone": self.phone,
            "timestamp": self.timestamp,
            "goong_ward": self.goong_ward,
            "goong_district": self.goong_district,
            "goong_province": self.goong_province,
            "goong_score": self.goong_score,
            "goong_formatted": self.goong_formatted,
            "way_a": self.way_a,
            "way_b": self.way_b,
            "has_goong": self.has_goong,
        }


def _sniff_csv_sep(path: Path, encoding: str) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    with path.open("r", encoding=encoding, errors="replace") as f:
        sample_lines = []
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            sample_lines.append(line)
    header_text = "".join(sample_lines)
    counts = {sep: header_text.count(sep) for sep in (",", "\t", ";")}
    best_sep = max(counts, key=lambda k: counts[k])
    return best_sep if counts[best_sep] > 0 else ","


def load_dataset(path: str | Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load CSV, TSV, XLSX/XLS, JSONL, or Parquet into a DataFrame."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    ext = p.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(p, sheet_name=sheet_name or 0, engine="openpyxl", dtype=object)
        return df.reset_index(drop=True)

    if ext in (".jsonl", ".ndjson"):
        return pd.read_json(p, lines=True).reset_index(drop=True)

    if ext == ".parquet":
        return pd.read_parquet(p).reset_index(drop=True)

    if ext in (".csv", ".tsv", ".txt"):
        last_err: Optional[Exception] = None
        for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
            try:
                sep = _sniff_csv_sep(p, enc)
                df = pd.read_csv(
                    p,
                    encoding=enc,
                    sep=sep,
                    dtype=object,
                    keep_default_na=True,
                )
                return df.reset_index(drop=True)
            except UnicodeDecodeError as exc:
                last_err = exc
                continue
        raise RuntimeError(f"Could not decode CSV file {p} with standard encodings: {last_err}")

    raise ValueError(f"Unsupported dataset format '{ext}'. Supported: .csv, .tsv, .xlsx, .jsonl, .parquet")


def write_csv_utf8_sig(df: pd.DataFrame, path: str | Path) -> None:
    """Write DataFrame as UTF-8-SIG CSV so Excel displays Vietnamese cleanly."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8-sig")


def resolve_schema(
    df: pd.DataFrame,
    overrides: Optional[Dict[str, Optional[object]]] = None,
) -> DatasetSchema:
    """Resolve DataFrame columns using explicit overrides + normalized alias matching."""
    overrides = overrides or {}
    raw_cols = list(df.columns)
    norm_to_raw: Dict[str, str] = {}
    for col in raw_cols:
        nc = normalize_col_name(str(col))
        if nc not in norm_to_raw:
            norm_to_raw[nc] = str(col)

    used_cols: set[str] = set()
    resolved: Dict[str, Optional[str]] = {}

    # 1. Apply explicit user overrides first
    for role in COLUMN_CANDIDATES:
        ov = overrides.get(role)
        if ov is not None and str(ov).strip() != "":
            ov_str = str(ov).strip()
            if ov_str in raw_cols:
                resolved[role] = ov_str
                used_cols.add(ov_str)
            else:
                nc = normalize_col_name(ov_str)
                if nc in norm_to_raw:
                    resolved[role] = norm_to_raw[nc]
                    used_cols.add(norm_to_raw[nc])
                else:
                    raise ValueError(
                        f"Column override --{role.replace('_', '-')}-col='{ov_str}' not found. "
                        f"Available columns: {raw_cols}"
                    )

    # 2. Auto-detect remaining roles by candidate priority order
    for role, candidates in COLUMN_CANDIDATES.items():
        if role in resolved:
            continue
        found: Optional[str] = None
        for cand in candidates:
            if cand in norm_to_raw:
                actual = norm_to_raw[cand]
                if actual not in used_cols:
                    found = actual
                    break
        if found is None:
            # Soft substring fallback only for key roles
            if role in ("text", "outcome", "rule_d", "phone", "timestamp"):
                for col in raw_cols:
                    col_s = str(col)
                    if col_s in used_cols:
                        continue
                    nc = normalize_col_name(col_s)
                    if any(nc == c or nc.endswith("_" + c) or nc.startswith(c + "_") for c in candidates):
                        found = col_s
                        break
        resolved[role] = found
        if found is not None:
            used_cols.add(found)

    missing_required = [r for r in ("text", "outcome") if not resolved.get(r)]
    if missing_required:
        raise ValueError(
            f"Không tự nhận diện được cột bắt buộc {missing_required}. "
            f"Vui lòng truyền --text-col / --outcome-col. Các cột hiện có: {raw_cols}"
        )

    rule_d_col = resolved.get("rule_d")
    rule_d_is_block = bool(overrides.get("rule_d_is_block", False))
    if not rule_d_is_block and rule_d_col is not None:
        nc_rd = normalize_col_name(rule_d_col)
        if any(tok in nc_rd for tok in ("block", "reject", "chan")) and "pass" not in nc_rd:
            rule_d_is_block = True

    return DatasetSchema(
        order_id=resolved.get("order_id"),
        text=str(resolved["text"]),
        outcome=str(resolved["outcome"]),
        rule_d=rule_d_col,
        rule_d_is_block=rule_d_is_block,
        phone=resolved.get("phone"),
        timestamp=resolved.get("timestamp"),
        goong_ward=resolved.get("goong_ward"),
        goong_district=resolved.get("goong_district"),
        goong_province=resolved.get("goong_province"),
        goong_score=resolved.get("goong_score"),
        goong_formatted=resolved.get("goong_formatted"),
        way_a=resolved.get("way_a"),
        way_b=resolved.get("way_b"),
    )


DEFAULT_RETURN_TOKENS = frozenset({
    "1", "1.0", "true", "yes", "y",
    "hoan", "don hoan", "hoan hang", "chuyen hoan", "da hoan",
    "returned", "return", "rto", "fail", "failed", "cancel", "cancelled", "canceled",
    "huy", "huy don", "da huy", "bom", "bom hang", "khong giao duoc",
    "tra hang", "that bai", "giao that bai",
})

DEFAULT_DELIVERED_TOKENS = frozenset({
    "0", "0.0", "false", "no", "n",
    "giao", "da giao", "giao thanh cong", "thanh cong", "hoan thanh",
    "delivered", "deliver", "success", "done", "completed", "ok", "nhan hang", "da nhan",
})


def parse_is_return(
    series: pd.Series,
    return_values: Optional[Sequence[str]] = None,
    delivered_values: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Convert outcome column to int8 array: 1=returned, 0=delivered, -1=unknown/skip."""
    ret_set = set(DEFAULT_RETURN_TOKENS)
    del_set = set(DEFAULT_DELIVERED_TOKENS)
    if return_values:
        ret_set.update(normalize_token_str(v) for v in return_values if str(v).strip())
    if delivered_values:
        del_set.update(normalize_token_str(v) for v in delivered_values if str(v).strip())

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

        tok = normalize_token_str(val)
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


DEFAULT_AMBIGUOUS_TRUE_TOKENS = frozenset({"1", "1.0", "true", "yes", "y"})
DEFAULT_AMBIGUOUS_FALSE_TOKENS = frozenset({"0", "0.0", "false", "no", "n"})

DEFAULT_EXPLICIT_PASS_TOKENS = frozenset({
    "pass", "passed", "ok", "allow", "allowed", "accept", "accepted",
    "cho qua", "qua", "du", "du den xa", "hop le",
    "address_sufficient", "sufficient",
})

DEFAULT_EXPLICIT_BLOCK_TOKENS = frozenset({
    "block", "blocked", "reject", "rejected", "fail", "failed", "deny", "denied",
    "chan", "khong qua", "tu choi", "chua du",
    "not_address", "address_insufficient", "insufficient",
})


def parse_bool_gate(
    series: pd.Series,
    invert: bool = False,
    default_on_na: bool = True,
) -> np.ndarray:
    """Convert gate decision column to bool array where True = PASSED gate.

    When `invert=True` (e.g., `rule_d_blocked`), numeric/boolean flags (1/0, True/False,
    yes/no) are inverted, whereas explicit semantic words ('block'/'chan' vs 'pass'/'qua')
    always retain their direct semantic meaning.
    """
    out = np.full(len(series), default_on_na, dtype=bool)
    for i, val in enumerate(series.tolist()):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            out[i] = default_on_na
            continue
        if isinstance(val, bool):
            out[i] = (not val) if invert else val
            continue
        if isinstance(val, (int, float, np.integer, np.floating)):
            fv = float(val) >= 0.5
            out[i] = (not fv) if invert else fv
            continue

        tok = normalize_token_str(val)
        if not tok:
            out[i] = default_on_na
            continue
        if tok in DEFAULT_EXPLICIT_PASS_TOKENS or ("sufficient" in tok and "insufficient" not in tok) or (
            any(k in tok for k in ("pass", "allow", "accept", "cho qua")) and "not" not in tok and "khong" not in tok
        ):
            out[i] = True
        elif tok in DEFAULT_EXPLICIT_BLOCK_TOKENS or any(k in tok for k in ("block", "reject", "chan", "insufficient", "not_address")):
            out[i] = False
        elif tok in DEFAULT_AMBIGUOUS_TRUE_TOKENS:
            out[i] = not invert
        elif tok in DEFAULT_AMBIGUOUS_FALSE_TOKENS:
            out[i] = invert
        else:
            out[i] = default_on_na
    return out


# VN phone pattern: 0912.345.678, +84 987 654 321, 0168..., 8491...
# Avoids matching slash house numbers like 01/23/45/67/89 because '/' is excluded.
_PHONE_PII_RE = re.compile(
    r"(?<!\d)(?:\+?84[\s.\-]?|0)\d(?:[\s.\-]?\d){7,9}(?!\d)"
)
_EMAIL_PII_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b", re.IGNORECASE)
_URL_PII_RE = re.compile(r"(?:https?://\S+|www\.\S+)", re.IGNORECASE)

# Matches explicit leading recipient labels (e.g. "Anh Nguyễn Văn A - ...", "Người nhận: Chị Hoa, ...")
# Anchored at string start (after optional <PHONE> + separator) so street names like
# "123 Nguyễn Văn Linh" or "Đường Cô Giang, Q1" are NEVER touched.
_LEADING_RECIPIENT_RE = re.compile(
    r"^((?:<PHONE>\s*[\-,:|]?\s*)?)"
    r"(?:nguoi\s*nhan|người\s*nhận|khach\s*hang|khách\s*hàng|khach|khách|"
    r"ten|tên|anh|chi|chị|co|cô|chu|chú|em|bac|bác|mr\.?|ms\.?|mrs\.?)"
    r"(?=\s|[:.])\s*[:.]?\s*[^,\-|\n]{1,40}(?=\s*[\-,:|])",
    re.IGNORECASE,
)


def mask_pii_for_features(text: str, mask_names: bool = True) -> str:
    """Mask phones, emails, URLs, and explicit leading recipient labels."""
    if not text:
        return ""
    s = str(text)
    s = _URL_PII_RE.sub(" <URL> ", s)
    s = _EMAIL_PII_RE.sub(" <EMAIL> ", s)
    s = _PHONE_PII_RE.sub(" <PHONE> ", s)
    if mask_names:
        s_strip = s.lstrip()
        s_masked = _LEADING_RECIPIENT_RE.sub(r"\1<NAME>", s_strip, count=1)
        s = s_masked
    s = re.sub(r"\s+", " ", s).strip()
    return s


_LONG_NUM_RE = re.compile(r"(?<!\d)\d{5,}(?!\d)")


def mask_pii_for_preview(text: str, mask_names: bool = True) -> str:
    """Mask PII + any long >=5-digit numeric tokens for human preview sheets."""
    s = mask_pii_for_features(text, mask_names=mask_names)
    s = _LONG_NUM_RE.sub("<NUM>", s)
    return s


def normalize_phone_key(phone_val: object, fallback_text: str) -> str:
    """Build hashed grouping key for customer phone (falls back to text hash if empty)."""
    raw = "" if phone_val is None or (isinstance(phone_val, float) and np.isnan(phone_val)) else str(phone_val)
    digits = re.sub(r"\D+", "", raw)
    if digits.startswith("84") and len(digits) >= 10:
        digits = "0" + digits[2:]
    if len(digits) >= 7:
        return "p_" + sha256_short(digits)
    # Fallback to phone embedded inside raw_text if any
    m = _PHONE_PII_RE.search(fallback_text or "")
    if m:
        emb = re.sub(r"\D+", "", m.group(0))
        if emb.startswith("84") and len(emb) >= 10:
            emb = "0" + emb[2:]
        if len(emb) >= 7:
            return "p_" + sha256_short(emb)
    norm_txt = re.sub(r"\s+", " ", strip_accents(fallback_text or "").lower()).strip()
    return "t_" + sha256_short(norm_txt)


def parse_goong_formatted(formatted: str) -> Tuple[str, str, str]:
    """Best-effort parse of formatted Goong address into (ward, district, province)."""
    if not formatted:
        return "", "", ""
    parts = [p.strip() for p in str(formatted).split(",") if p.strip()]
    # Drop trailing Vietnam / postal code
    while parts:
        tail_norm = normalize_token_str(parts[-1])
        if tail_norm in ("viet nam", "vietnam", "vn") or re.fullmatch(r"\d{5,6}", tail_norm):
            parts.pop()
        else:
            break
    if not parts:
        return "", "", ""
    if len(parts) == 1:
        return "", "", parts[0]
    if len(parts) == 2:
        # Either "Ward, Province" (2025) or "District, Province"
        first_norm = normalize_token_str(parts[0])
        if first_norm.startswith(("quan ", "huyen ", "thi xa ", "q.", "h.", "tx.")):
            return "", parts[0], parts[1]
        return parts[0], "", parts[1]
    return parts[-3], parts[-2], parts[-1]


def populate_goong_columns(df: pd.DataFrame, schema: DatasetSchema) -> Tuple[List[str], List[str], List[str], np.ndarray]:
    """Extract cleaned (ward, district, province, score) arrays, falling back to goong_formatted."""
    n = len(df)
    wards = [""] * n
    dists = [""] * n
    provs = [""] * n

    def _col_strs(col: Optional[str]) -> List[str]:
        if not col or col not in df.columns:
            return [""] * n
        out: List[str] = []
        for v in df[col].tolist():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out.append("")
            else:
                out.append(str(v).strip())
        return out

    w_raw = _col_strs(schema.goong_ward)
    d_raw = _col_strs(schema.goong_district)
    p_raw = _col_strs(schema.goong_province)
    f_raw = _col_strs(schema.goong_formatted)

    for i in range(n):
        w, d, p = w_raw[i], d_raw[i], p_raw[i]
        if not (w or d or p) and f_raw[i]:
            w, d, p = parse_goong_formatted(f_raw[i])
        elif f_raw[i] and not (w and p):
            fw, fd, fp = parse_goong_formatted(f_raw[i])
            w = w or fw
            d = d or fd
            p = p or fp
        wards[i] = w
        dists[i] = d
        provs[i] = p

    scores = np.full(n, -1.0, dtype=np.float64)
    if schema.goong_score and schema.goong_score in df.columns:
        parsed_s = pd.to_numeric(df[schema.goong_score], errors="coerce").to_numpy(dtype=np.float64)
        scores = np.where(np.isfinite(parsed_s), parsed_s, -1.0)

    return wards, dists, provs, scores
