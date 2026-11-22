"""Domain-invariant Pre-Goong & Post-Goong consistency feature extractor.

Loads the MIT-licensed Vietnam Administrative Gazetteer (`shared_data/vn_admin_gazetteer.json`)
and extracts strictly numeric, zero-PII, non-regional features measuring:
  1) Structural / lexical completeness of `raw_text` (PRE_GOONG_STRUCT_FEATURES, 24 features)
  2) Alignment, grounding, hallucination, contradiction, and nationwide ward ambiguity
     between `raw_text` and Goong auto-fill output (POST_GOONG_CONSISTENCY_FEATURES, 25 features)

Intentionally excludes province/district one-hot IDs so ceiling models cannot cheat on
regional COD return-rate differences.
"""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from schema_io import DatasetSchema, mask_pii_for_features, populate_goong_columns, strip_accents

DEFAULT_GAZETTEER_PATH = (
    Path(__file__).resolve().parent.parent / "shared_data" / "vn_admin_gazetteer.json"
)

_VN_ACCENT_RE = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.IGNORECASE)
_HOUSE_NUM_RE = re.compile(r"(?:^|[\s,])(?:so\s+nha|so|sn)?\s*\d{1,4}[a-z]?(?:/\d{1,3}[a-z]?){0,3}(?=[\s,]|$)", re.IGNORECASE)
_SLASH_HOUSE_RE = re.compile(r"(?:^|[\s,])\d{1,4}[a-z]?(?:/\d{1,3}[a-z]?){1,3}(?=[\s,]|$)", re.IGNORECASE)
_WARD_PFX_RE = re.compile(r"(?:^|[\s,])(?:phuong|xa|thi\s+tran|dac\s+khu|[px]\.|tt\.?|dk\b)", re.IGNORECASE)
_DIST_PFX_RE = re.compile(r"(?:^|[\s,])(?:quan|huyen|thi\s+xa|[qh]\.|tx\.?|\bq\s?\d{1,2}\b)", re.IGNORECASE)
_PROV_PFX_RE = re.compile(r"(?:^|[\s,])(?:tinh|thanh\s+pho|tp\.?|tphcm|tphn)", re.IGNORECASE)
_STREET_KW_RE = re.compile(r"\b(?:duong|pho|ngo|ngach|hem|kiet|dai\s+lo|ql\s*\d*|tl\s*\d*|quoc\s+lo|tinh\s+lo)\b", re.IGNORECASE)
_RURAL_KW_RE = re.compile(r"\b(?:thon|xom|ap|ban|lang|buon|doi|khu\s+pho|kp\.?|to\s+dan\s+pho|tdp\.?|cum|khu\s+\d+)\b", re.IGNORECASE)
_APT_KW_RE = re.compile(r"\b(?:chung\s+cu|cc\.?|tang|phong|block|toa|can\s+ho|kdt|khu\s+do\s+thi|kcn|khu\s+cong\s+nghiep)\b", re.IGNORECASE)
_DELIVERY_NOTE_RE = re.compile(r"\b(?:gio\s+hanh\s+chinh|ngoai\s+gio|goi\s+truoc|truoc\s+khi\s+giao|ship|giao\s+giup|nhan\s+ho|bom|kiem\s+tra\s+hang)\b", re.IGNORECASE)
_PHONE_ANY_RE = re.compile(r"(?<!\d)(?:\+?84|0)\d(?:[\s.\-]?\d){7,9}(?!\d)")
_CHAT_FOOD_RE = re.compile(
    r"\b(?:xin\s+gia|bao\s+gia|gia\s+bao\s+nhieu|phi\s+ship|freeship|giam\s+gia|ok\s+shop|chot\s+don|"
    r"con\s+hang|het\s+hang|kiem\s+tra\s+don|banh\s+mi|muc\s+ca\s+mau|tra\s+sua|ca\s+phe|"
    r"ao\s+thun|quan\s+jean|chai\s+nuong|nuong\s+bbq|inbox|cam\s+on|xin\s+chao)\b",
    re.IGNORECASE,
)

_ATTACHED_PFX_RE = re.compile(r"(^|\s)(tp|tx|tt|kp|tdp|sn|ql|tl|dk|p|q|h|x|t)\.(?=[a-z0-9])")
_COMPACT_NUM_PFX_RE = re.compile(r"(^|\s)(p|q)(\d{1,2})(?=\s|$|,)")
_STREET_CONTEXT_WORDS = frozenset({
    "duong", "pho", "ngo", "ngach", "hem", "kiet", "cau", "cho", "kcn", "kdt", "chung cu", "cc"
})
_HOUSE_TOKEN_RE = re.compile(r"^\d{1,4}[a-z]?(?:/\d{1,3}[a-z]?)*$")

_WARD_PFX_WORDS = ("phuong", "xa", "thi tran", "dac khu", "p", "x", "tt", "dk", "ph")
_DIST_PFX_WORDS = ("quan", "huyen", "thi xa", "thanh pho", "q", "h", "tx", "tp")
_PROV_PFX_WORDS = ("tinh", "thanh pho", "tp", "t")


@lru_cache(maxsize=8192)
def _char_bigrams(s: str) -> FrozenSet[str]:
    compact = s.replace(" ", "")
    if not compact:
        return frozenset()
    if len(compact) == 1:
        return frozenset((compact,))
    return frozenset(compact[i : i + 2] for i in range(len(compact) - 1))


def _jaccard_bigrams(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ba = _char_bigrams(a)
    bb = _char_bigrams(b)
    if not ba or not bb:
        return 0.0
    inter = len(ba & bb)
    union = len(ba | bb)
    return inter / union if union else 0.0


def tokenize_admin_string(raw: str) -> List[str]:
    """Normalize and tokenize VN string, keeping commas ',' as segment boundaries."""
    if not raw:
        return []
    s = strip_accents(str(raw)).lower().strip()
    s = re.sub(r"\bba\s+ria\s*-\s*vung\s+tau\b", "ba ria vung tau", s)
    s = re.sub(r"\bphan\s+rang\s*-\s*thap\s+cham\b", "phan rang thap cham", s)
    s = re.sub(r"[,;|()]+", " , ", s)
    s = re.sub(r"\s+-\s+", " , ", s)
    s = re.sub(r"(^|\s)tp\.?\s*hcm(?=\s|$|,)", r"\1tphcm", s)
    s = re.sub(r"(^|\s)tp\.?\s*hn(?=\s|$|,)", r"\1tphn", s)
    s = re.sub(r"(^|\s)br\s*-\s*vt(?=\s|$|,)", r"\1brvt", s)
    s = _ATTACHED_PFX_RE.sub(r"\1\2. ", s)
    s = _COMPACT_NUM_PFX_RE.sub(r"\1\2 \3", s)
    s = re.sub(r"\.(?=\s|$|,)", "", s)
    s = re.sub(r"[^\w\s/,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    tokens: List[str] = []
    for tok in s.split(" "):
        if tok == "-":
            continue
        # Remove punctuation around slashes only if token has no letters (keep house numbers intact)
        tokens.append(tok)
    return tokens


def strip_admin_prefix_tokens(tokens: List[str], level: str) -> str:
    """Strip administrative level prefix tokens cleanly while keeping numeric ward/dist forms."""
    clean = [t for t in tokens if t != ","]
    if not clean:
        return ""
    joined = " ".join(clean)
    if level == "prov":
        for p in ("thanh pho ", "tinh ", "tp ", "t "):
            if joined.startswith(p):
                rest = joined[len(p) :].strip()
                return rest or joined
        return joined
    if level == "dist":
        for p in ("thanh pho ", "thi xa ", "huyen ", "quan ", "tp ", "tx ", "h ", "q "):
            if joined.startswith(p):
                rest = joined[len(p) :].strip()
                if not rest:
                    return joined
                if rest.isdigit():
                    return f"quan {rest}"
                return rest
        return joined
    if level == "ward":
        for p in ("thi tran ", "dac khu ", "phuong ", "xa ", "tt ", "dk ", "ph ", "p ", "x "):
            if joined.startswith(p):
                rest = joined[len(p) :].strip()
                if not rest:
                    return joined
                if rest.isdigit():
                    return f"phuong {rest}"
                return rest
        return joined
    return joined


class VietnamAdminIndex:
    """Fast MIT-licensed VN Gazetteer index for grounding, ambiguity, and contradiction checks."""

    def __init__(self, gazetteer_path: Path = DEFAULT_GAZETTEER_PATH) -> None:
        if not gazetteer_path.exists():
            raise FileNotFoundError(f"MIT Gazetteer not found: {gazetteer_path}")
        data = json.loads(gazetteer_path.read_text(encoding="utf-8"))

        self.provinces: List[str] = list(data["p"])
        self.districts: List[str] = list(data["d"])
        self.wards: List[str] = list(data["w"])
        self.ward_edges_list: List[Set[Tuple[int, int]]] = [
            {tuple(x) for x in edges} for edges in data["e"]
        ]
        self.dist_provinces_list: List[Set[int]] = [set(ps) for ps in data["dp"]]

        self.prov_map: Dict[str, int] = {}
        self.dist_map: Dict[str, int] = {}
        self.ward_map: Dict[str, int] = {}

        # Track canonical aliases (alias -> id and id -> canonical tokens)
        max_words = 1

        def _reg(tbl: Dict[str, int], key: str, idx: int) -> None:
            nonlocal max_words
            k = re.sub(r"\s+", " ", key.replace("-", " ")).strip()
            if k:
                tbl.setdefault(k, idx)
                max_words = max(max_words, len(k.split(" ")))

        for i, k in enumerate(self.provinces):
            self.prov_map[k] = i
            _reg(self.prov_map, k, i)
            max_words = max(max_words, len(k.split(" ")))

        for alias, canon in (data.get("pa") or {}).items():
            if canon in self.prov_map:
                pid = self.prov_map[canon]
                _reg(self.prov_map, alias, pid)

        for i, k in enumerate(self.districts):
            self.dist_map[k] = i
            _reg(self.dist_map, k, i)
            max_words = max(max_words, len(k.split(" ")))

        for alias, canon in (data.get("da") or {}).items():
            if canon in self.dist_map:
                did = self.dist_map[canon]
                _reg(self.dist_map, alias, did)

        for i, k in enumerate(self.wards):
            self.ward_map[k] = i
            _reg(self.ward_map, k, i)
            max_words = max(max_words, len(k.split(" ")))

        self.max_phrase_words = min(6, max_words)

        # Precompute ambiguity counts nationwide for each ward ID
        self.ward_prov_count: List[int] = [
            len({pid for _did, pid in edges if pid >= 0}) for edges in self.ward_edges_list
        ]
        self.ward_dist_count: List[int] = [
            len(edges) for edges in self.ward_edges_list
        ]
        self.ambiguous_single_provs = frozenset({"vinh"})

    def canonicalize_province(self, raw: str) -> Tuple[Optional[int], str]:
        toks = tokenize_admin_string(raw)
        clean = [t for t in toks if t != ","]
        if not clean:
            return None, ""
        full = " ".join(clean)
        stripped = strip_admin_prefix_tokens(clean, "prov")
        if full in self.prov_map:
            pid = self.prov_map[full]
            return pid, self.provinces[pid].replace("-", " ").strip()
        if stripped in self.prov_map:
            pid = self.prov_map[stripped]
            return pid, self.provinces[pid].replace("-", " ").strip()
        no_hyphen = re.sub(r"\s+", " ", stripped.replace("-", " ")).strip()
        if no_hyphen in self.prov_map:
            pid = self.prov_map[no_hyphen]
            return pid, self.provinces[pid].replace("-", " ").strip()
        return None, no_hyphen

    def canonicalize_district(self, raw: str) -> Tuple[Optional[int], str]:
        toks = tokenize_admin_string(raw)
        clean = [t for t in toks if t != ","]
        if not clean:
            return None, ""
        full = " ".join(clean)
        stripped = strip_admin_prefix_tokens(clean, "dist")
        if full in self.dist_map:
            did = self.dist_map[full]
            return did, self.districts[did].replace("-", " ").strip()
        if stripped in self.dist_map:
            did = self.dist_map[stripped]
            return did, self.districts[did].replace("-", " ").strip()
        no_hyphen = re.sub(r"\s+", " ", stripped.replace("-", " ")).strip()
        if no_hyphen in self.dist_map:
            did = self.dist_map[no_hyphen]
            return did, self.districts[did].replace("-", " ").strip()
        return None, no_hyphen

    def canonicalize_ward(self, raw: str) -> Tuple[Optional[int], str]:
        toks = tokenize_admin_string(raw)
        clean = [t for t in toks if t != ","]
        if not clean:
            return None, ""
        stripped = strip_admin_prefix_tokens(clean, "ward")
        if stripped in self.ward_map:
            wid = self.ward_map[stripped]
            return wid, self.wards[wid].replace("-", " ").strip()
        full = " ".join(clean)
        if full in self.ward_map:
            wid = self.ward_map[full]
            return wid, self.wards[wid].replace("-", " ").strip()
        no_hyphen = re.sub(r"\s+", " ", stripped.replace("-", " ")).strip()
        if no_hyphen in self.ward_map:
            wid = self.ward_map[no_hyphen]
            return wid, self.wards[wid].replace("-", " ").strip()
        return None, no_hyphen

    def _detect_prefix(self, tokens: List[str], i: int) -> Optional[str]:
        if i <= 0:
            return None
        t1 = tokens[i - 1]
        if t1 == ",":
            return None
        if i >= 2 and tokens[i - 2] != ",":
            t2 = f"{tokens[i - 2]} {t1}"
            if t2 in ("thi tran", "dac khu"):
                return "ward"
            if t2 == "thi xa":
                return "dist"
            if t2 == "thanh pho":
                return "dist_or_prov"
        if t1 in ("phuong", "xa", "p", "x", "tt", "dk", "ph"):
            return "ward"
        if t1 in ("quan", "huyen", "q", "h", "tx"):
            return "dist"
        if t1 in ("tinh", "t"):
            return "prov"
        if t1 == "tp":
            return "dist_or_prov"
        return None

    def scan_raw_text(self, tokens: List[str]) -> Dict[str, object]:
        """Scan tokenized raw text once for ward/district/province spans & ambiguity."""
        n = len(tokens)
        ward_hits: List[Tuple[int, str, int, int, Optional[str]]] = []  # (wid, phrase, start, end, pfx)
        dist_hits: List[Tuple[int, str, int, int, Optional[str]]] = []
        prov_hits: List[Tuple[int, str, int, int, Optional[str]]] = []

        for i in range(n):
            if tokens[i] == ",":
                continue
            pfx = self._detect_prefix(tokens, i)
            max_len = min(self.max_phrase_words, n - i)
            for ln in range(max_len, 0, -1):
                sl = tokens[i : i + ln]
                if "," in sl:
                    continue
                phrase = " ".join(sl)

                # Ward span
                if pfx not in ("dist", "prov", "dist_or_prov"):
                    wid = self.ward_map.get(phrase)
                    if wid is not None and not phrase.isdigit():
                        ward_hits.append((wid, phrase, i, i + ln, pfx))
                    elif ln == 1 and phrase.isdigit() and pfx == "ward":
                        num_wid = self.ward_map.get(f"phuong {phrase}")
                        if num_wid is not None:
                            ward_hits.append((num_wid, f"phuong {phrase}", i, i + 1, pfx))

                # District span
                did = self.dist_map.get(phrase)
                if did is None and ln == 1 and phrase.isdigit() and pfx == "dist" and tokens[i - 1] in ("q", "quan"):
                    did = self.dist_map.get(f"quan {phrase}")
                if did is not None:
                    dist_hits.append((did, phrase, i, i + ln, pfx))

                # Province span
                pid = self.prov_map.get(phrase)
                if pid is not None:
                    if not (ln == 1 and phrase in self.ambiguous_single_provs and pfx not in ("prov", "dist_or_prov")):
                        prov_hits.append((pid, phrase, i, i + ln, pfx))

        # Check whether any raw (ward, dist/prov) span pair is valid & span-disjoint
        valid_pair = 0
        min_ambig = 0
        if ward_hits:
            min_ambig = min(max(1, self.ward_prov_count[w[0]]) for w in ward_hits)
            for wid, _ph, ws, we, _wpfx in ward_hits:
                edges = self.ward_edges_list[wid]
                for did, _dph, ds, de, _dpfx in dist_hits:
                    if de <= ws or ds >= we:
                        if any(ed == did for ed, _ep in edges):
                            valid_pair = 1
                            break
                if valid_pair:
                    break
                for pid, _pph, ps, pe, _ppfx in prov_hits:
                    if pe <= ws or ps >= we:
                        if any(ep == pid for _ed, ep in edges):
                            valid_pair = 1
                            break
                if valid_pair:
                    break

        return {
            "ward_hits": ward_hits,
            "dist_hits": dist_hits,
            "prov_hits": prov_hits,
            "raw_ward_ids": {w[0] for w in ward_hits},
            "raw_dist_ids": {d[0] for d in dist_hits},
            "raw_prov_ids": {p[0] for p in prov_hits},
            "valid_pair": valid_pair,
            "min_ward_ambiguity": min_ambig,
        }


PRE_GOONG_STRUCT_FEATURES: Tuple[str, ...] = (
    "char_len",
    "word_count",
    "digit_count",
    "alpha_ratio",
    "uppercase_ratio",
    "comma_count",
    "slash_count",
    "has_house_num",
    "has_slash_house_num",
    "has_ward_prefix",
    "has_dist_prefix",
    "has_prov_prefix",
    "has_street_kw",
    "has_rural_kw",
    "has_apt_kw",
    "has_delivery_note",
    "has_phone_in_text",
    "has_chat_or_food_cue",
    "has_vn_accent",
    "raw_gaz_ward_hits",
    "raw_gaz_dist_hits",
    "raw_gaz_prov_hits",
    "raw_gaz_valid_ward_parent_pair",
    "raw_gaz_min_ward_ambiguity",
)

POST_GOONG_CONSISTENCY_FEATURES: Tuple[str, ...] = (
    "goong_has_ward",
    "goong_has_dist",
    "goong_has_prov",
    "goong_filled_levels",
    "ward_exact_in_raw",
    "ward_token_overlap",
    "ward_fuzzy_sim",
    "dist_exact_or_alias_in_raw",
    "dist_token_overlap",
    "dist_fuzzy_sim",
    "prov_exact_or_alias_in_raw",
    "prov_token_overlap",
    "prov_fuzzy_sim",
    "admin_token_grounding_ratio",
    "hallucinated_levels_count",
    "ward_hallucinated",
    "parent_hallucinated",
    "ambiguous_ward_ungrounded_parent",
    "raw_prov_contradicts_goong",
    "raw_dist_contradicts_goong",
    "goong_hierarchy_valid",
    "goong_ward_prov_count",
    "goong_ward_dist_count",
    "log_goong_ward_ambiguity",
    "goong_score",
    "goong_score_missing",
    "raw_to_goong_token_ratio",
)


def _match_component(
    canon_str: str,
    canon_id: Optional[int],
    raw_ids: Set[int],
    raw_words: List[str],
    raw_word_set: Set[str],
    raw_joined: str,
) -> Tuple[float, float, float, Set[str]]:
    """Return (exact_or_alias, token_overlap, fuzzy_sim, matched_canon_tokens)."""
    if not canon_str:
        return 0.0, 0.0, 0.0, set()

    comp_tokens = [t for t in canon_str.split(" ") if t]
    if not comp_tokens:
        return 0.0, 0.0, 0.0, set()
    comp_set = set(comp_tokens)

    # 1. Exact ID or exact contiguous phrase match
    exact = 0.0
    if canon_id is not None and canon_id in raw_ids:
        exact = 1.0
    elif f" {canon_str} " in f" {raw_joined} ":
        exact = 1.0

    if exact == 1.0:
        return 1.0, 1.0, 1.0, comp_set

    # 2. Token overlap
    matched_toks = comp_set & raw_word_set
    overlap = len(matched_toks) / len(comp_set)

    # 3. Sliding-window character bigram similarity (handles minor typos & omitted spaces)
    m = len(comp_tokens)
    best_sim = 0.0
    n_raw = len(raw_words)
    target_nospace = "".join(comp_tokens)
    for win_len in {max(1, m - 1), m, m + 1}:
        if win_len > n_raw:
            continue
        for i in range(n_raw - win_len + 1):
            cand = " ".join(raw_words[i : i + win_len])
            sim = _jaccard_bigrams(canon_str, cand)
            if sim > best_sim:
                best_sim = sim
            if win_len == 1 and len(comp_tokens) > 1:
                sim2 = _jaccard_bigrams(target_nospace, cand)
                if sim2 > best_sim:
                    best_sim = sim2

    return 0.0, float(overlap), float(best_sim), matched_toks


def _is_street_or_admin_context(tokens: List[str], idx: int) -> bool:
    if idx <= 0:
        return False
    prev1 = tokens[idx - 1]
    if prev1 == ",":
        return False
    if prev1 in _STREET_CONTEXT_WORDS:
        return True
    if _HOUSE_TOKEN_RE.match(prev1):
        return True
    if idx >= 2 and tokens[idx - 2] != ",":
        prev2 = f"{tokens[idx - 2]} {prev1}"
        if prev2 in _STREET_CONTEXT_WORDS:
            return True
    return False


def classify_post_goong_error_bucket(feats: Dict[str, float]) -> str:
    """Assign an anonymized, rule-based structural diagnostic category (zero PII)."""
    if feats.get("has_chat_or_food_cue", 0.0) >= 0.5 and feats.get("raw_gaz_ward_hits", 0.0) == 0.0:
        return "raw_chat_or_food_cue"
    if feats.get("goong_has_ward", 0.0) < 0.5:
        return "goong_missing_ward"
    if feats.get("raw_prov_contradicts_goong", 0.0) >= 0.5 or feats.get("raw_dist_contradicts_goong", 0.0) >= 0.5:
        return "goong_prov_or_dist_contradiction"
    if feats.get("ambiguous_ward_ungrounded_parent", 0.0) >= 0.5:
        return "goong_ambiguous_ward_hallucinated_parent"
    if feats.get("hallucinated_levels_count", 0.0) >= 2.0:
        return "goong_full_hallucination_2plus_levels"
    if feats.get("ward_hallucinated", 0.0) >= 0.5:
        return "goong_ward_not_in_raw"
    if feats.get("raw_gaz_ward_hits", 0.0) == 0.0 and feats.get("word_count", 0.0) <= 4.0:
        return "raw_missing_ward_and_short"
    return "grounded_consistent"


def extract_feature_dataframe(
    df: pd.DataFrame,
    schema: DatasetSchema,
    index: Optional[VietnamAdminIndex] = None,
    mask_names: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Extract (pre_struct_df, post_consist_df, diagnostic_buckets) cleanly for all rows."""
    if index is None:
        index = VietnamAdminIndex()

    raw_texts = [
        "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
        for v in df[schema.text].tolist()
    ]
    g_wards, g_dists, g_provs, g_scores = populate_goong_columns(df, schema)

    pre_rows: List[List[float]] = []
    post_rows: List[List[float]] = []
    buckets: List[str] = []

    for i, raw in enumerate(raw_texts):
        masked = mask_pii_for_features(raw, mask_names=mask_names)
        norm = re.sub(r"\s+", " ", strip_accents(masked).lower()).strip()

        char_len = float(len(raw.strip()))
        tokens_comma = tokenize_admin_string(norm)
        raw_words = [t for t in tokens_comma if t != ","]
        word_count = float(len(raw_words))
        digit_count = float(sum(1 for ch in raw if ch.isdigit()))
        alpha_count = sum(1 for ch in raw if ch.isalpha())
        alpha_ratio = float(alpha_count / max(1, len(raw)))
        upper_count = sum(1 for ch in raw if ch.isupper())
        uppercase_ratio = float(upper_count / max(1, alpha_count))
        comma_count = float(raw.count(","))
        slash_count = float(raw.count("/"))

        has_house_num = 1.0 if _HOUSE_NUM_RE.search(norm) else 0.0
        has_slash_house = 1.0 if _SLASH_HOUSE_RE.search(norm) else 0.0
        has_ward_pfx = 1.0 if _WARD_PFX_RE.search(norm) else 0.0
        has_dist_pfx = 1.0 if _DIST_PFX_RE.search(norm) else 0.0
        has_prov_pfx = 1.0 if _PROV_PFX_RE.search(norm) else 0.0
        has_street_kw = 1.0 if _STREET_KW_RE.search(norm) else 0.0
        has_rural_kw = 1.0 if _RURAL_KW_RE.search(norm) else 0.0
        has_apt_kw = 1.0 if _APT_KW_RE.search(norm) else 0.0
        has_delivery_note = 1.0 if _DELIVERY_NOTE_RE.search(norm) else 0.0
        has_phone_in_text = 1.0 if _PHONE_ANY_RE.search(raw) else 0.0
        has_chat_food = 1.0 if _CHAT_FOOD_RE.search(norm) else 0.0
        has_vn_accent = 1.0 if _VN_ACCENT_RE.search(raw) else 0.0

        scan = index.scan_raw_text(tokens_comma)
        raw_ward_ids: Set[int] = scan["raw_ward_ids"]  # type: ignore[assignment]
        raw_dist_ids: Set[int] = scan["raw_dist_ids"]  # type: ignore[assignment]
        raw_prov_ids: Set[int] = scan["raw_prov_ids"]  # type: ignore[assignment]

        pre_vec = [
            char_len,
            word_count,
            digit_count,
            alpha_ratio,
            uppercase_ratio,
            comma_count,
            slash_count,
            has_house_num,
            has_slash_house,
            has_ward_pfx,
            has_dist_pfx,
            has_prov_pfx,
            has_street_kw,
            has_rural_kw,
            has_apt_kw,
            has_delivery_note,
            has_phone_in_text,
            has_chat_food,
            has_vn_accent,
            float(len(raw_ward_ids)),
            float(len(raw_dist_ids)),
            float(len(raw_prov_ids)),
            float(scan["valid_pair"]),
            float(scan["min_ward_ambiguity"]),
        ]
        pre_rows.append(pre_vec)

        # --- Post-Goong features ---
        g_wid, g_w_bare = index.canonicalize_ward(g_wards[i])
        g_did, g_d_bare = index.canonicalize_district(g_dists[i])
        g_pid, g_p_bare = index.canonicalize_province(g_provs[i])

        goong_has_ward = 1.0 if g_w_bare else 0.0
        goong_has_dist = 1.0 if g_d_bare else 0.0
        goong_has_prov = 1.0 if g_p_bare else 0.0
        goong_filled = goong_has_ward + goong_has_dist + goong_has_prov

        raw_word_set = set(raw_words)
        raw_joined = " ".join(raw_words)

        w_exact, w_ov, w_fuzz, w_matched = _match_component(
            g_w_bare, g_wid, raw_ward_ids, raw_words, raw_word_set, raw_joined
        )
        d_exact, d_ov, d_fuzz, d_matched = _match_component(
            g_d_bare, g_did, raw_dist_ids, raw_words, raw_word_set, raw_joined
        )
        p_exact, p_ov, p_fuzz, p_matched = _match_component(
            g_p_bare, g_pid, raw_prov_ids, raw_words, raw_word_set, raw_joined
        )

        # Admin token grounding ratio across unique tokens of filled Goong components
        all_admin_toks = {t for s in (g_w_bare, g_d_bare, g_p_bare) if s for t in s.split(" ") if t}
        grounded_toks = w_matched | d_matched | p_matched
        admin_ground_ratio = (
            float(len(grounded_toks) / len(all_admin_toks)) if all_admin_toks else 0.0
        )

        w_grounded = (w_exact >= 0.5) or (w_fuzz >= 0.65)
        d_grounded = (d_exact >= 0.5) or (d_fuzz >= 0.65)
        p_grounded = (p_exact >= 0.5) or (p_fuzz >= 0.65)

        ward_hallucinated = 1.0 if (goong_has_ward > 0.5 and not w_grounded) else 0.0
        dist_halluc = 1.0 if (goong_has_dist > 0.5 and not d_grounded) else 0.0
        prov_halluc = 1.0 if (goong_has_prov > 0.5 and not p_grounded) else 0.0
        halluc_count = ward_hallucinated + dist_halluc + prov_halluc

        # Parent hallucinated: at least one parent level returned by Goong, yet NONE grounded in raw
        has_any_parent = (goong_has_dist > 0.5) or (goong_has_prov > 0.5)
        parent_halluc = 1.0 if (has_any_parent and not d_grounded and not p_grounded) else 0.0

        # Nationwide ambiguity of Goong's ward
        if g_wid is not None:
            w_prov_cnt = float(max(1, index.ward_prov_count[g_wid]))
            w_dist_cnt = float(max(1, index.ward_dist_count[g_wid]))
        else:
            w_prov_cnt = 0.0
            w_dist_cnt = 0.0
        log_w_ambig = math.log1p(w_prov_cnt)

        ambig_ward_halluc_parent = (
            1.0
            if (goong_has_ward > 0.5 and w_grounded and parent_halluc > 0.5 and w_prov_cnt >= 2.0)
            else 0.0
        )

        # Detect prefix token start index for spans so prefix words ("thanh pho", "huyen", ...) are accounted for
        def _pfx_span_start(tokens: List[str], i: int, pfx: Optional[str]) -> int:
            if pfx is None or i <= 0:
                return i
            if i >= 2 and f"{tokens[i - 2]} {tokens[i - 1]}" in ("thi tran", "dac khu", "thi xa", "thanh pho"):
                return i - 2
            return i - 1

        # Build token intervals covered by confirmed Goong ward / dist / prov entities (including prefixes!)
        explained_intervals: List[Tuple[int, int]] = []
        ward_hits_all: List[Tuple[int, str, int, int, Optional[str]]] = scan["ward_hits"]  # type: ignore[assignment]
        dist_hits_all: List[Tuple[int, str, int, int, Optional[str]]] = scan["dist_hits"]  # type: ignore[assignment]
        prov_hits_all: List[Tuple[int, str, int, int, Optional[str]]] = scan["prov_hits"]  # type: ignore[assignment]

        # Match longer spans first when marking explained intervals so e.g. "hiep binh phuoc" covers all 3 tokens
        for wid, phrase, ws, we, wpfx in ward_hits_all:
            if (g_wid is not None and wid == g_wid) or (g_w_bare and phrase == g_w_bare):
                explained_intervals.append((_pfx_span_start(tokens_comma, ws, wpfx), we))
        for did, phrase, ds, de, dpfx in dist_hits_all:
            if (g_did is not None and did == g_did) or (g_d_bare and phrase == g_d_bare):
                explained_intervals.append((_pfx_span_start(tokens_comma, ds, dpfx), de))
        for pid, phrase, ps, pe, ppfx in prov_hits_all:
            if (g_pid is not None and pid == g_pid) or (g_p_bare and phrase == g_p_bare):
                explained_intervals.append((_pfx_span_start(tokens_comma, ps, ppfx), pe))

        def _overlaps_explained(s_i: int, e_i: int) -> bool:
            return any(not (e_i <= xs or s_i >= xe) for xs, xe in explained_intervals)

        # Also add raw matched ward/dist/prov intervals matching any non-contradictory parent of g_ward
        # (protects e.g. "Xa An Phu, ..." sub-tokens inside longer raw wards)
        raw_ward_spans_full = [
            (_pfx_span_start(tokens_comma, ws, wpfx), we) for _wid, _ph, ws, we, wpfx in ward_hits_all
        ]

        # Raw contradicts Goong province/district (strictly outside explained Goong intervals!)
        prov_contradicts = 0.0
        if g_pid is not None:
            for pid, phrase, ps, pe, pfx in prov_hits_all:
                if pid == g_pid:
                    continue
                if _overlaps_explained(ps, pe):
                    continue
                if phrase == g_w_bare or phrase == g_d_bare:
                    continue
                if pfx in ("ward", "dist"):
                    continue
                if _is_street_or_admin_context(tokens_comma, ps) and pfx not in ("prov", "dist_or_prov"):
                    continue
                prov_contradicts = 1.0
                break

        dist_contradicts = 0.0
        if g_did is not None:
            for did, phrase, ds, de, pfx in dist_hits_all:
                if did == g_did:
                    continue
                if _overlaps_explained(ds, de):
                    continue
                # Also skip proper substrings inside an unambiguous longer raw ward/prov span
                if any(ds >= ws and de <= we and (we - ws) > (de - ds) for ws, we in raw_ward_spans_full):
                    continue
                if phrase == g_w_bare or phrase == g_p_bare:
                    continue
                if pfx in ("ward", "prov"):
                    continue
                if len(phrase.split(" ")) == 1 and pfx not in ("dist", "dist_or_prov"):
                    continue
                if _is_street_or_admin_context(tokens_comma, ds) and pfx not in ("dist", "dist_or_prov"):
                    continue
                dist_contradicts = 1.0
                break

        # Hierarchy consistency inside MIT gazetteer
        if g_wid is not None and g_pid is not None:
            edges = index.ward_edges_list[g_wid]
            if g_did is not None:
                valid_hier = 1.0 if ((g_did, g_pid) in edges or (-1, g_pid) in edges) else 0.0
            else:
                valid_hier = 1.0 if any(ep == g_pid for _ed, ep in edges) else 0.0
        elif g_did is not None and g_pid is not None:
            valid_hier = 1.0 if g_pid in index.dist_provinces_list[g_did] else 0.0
        else:
            valid_hier = -1.0

        raw_score = float(g_scores[i])
        score_missing = 1.0 if raw_score < 0.0 else 0.0
        norm_score = raw_score if raw_score >= 0.0 else -1.0
        goong_tok_cnt = max(1, len(all_admin_toks))
        raw_to_goong_ratio = float(len(raw_words) / goong_tok_cnt)

        post_vec = [
            goong_has_ward,
            goong_has_dist,
            goong_has_prov,
            goong_filled,
            w_exact,
            w_ov,
            w_fuzz,
            d_exact,
            d_ov,
            d_fuzz,
            p_exact,
            p_ov,
            p_fuzz,
            admin_ground_ratio,
            halluc_count,
            ward_hallucinated,
            parent_halluc,
            ambig_ward_halluc_parent,
            prov_contradicts,
            dist_contradicts,
            valid_hier,
            w_prov_cnt,
            w_dist_cnt,
            log_w_ambig,
            norm_score,
            score_missing,
            raw_to_goong_ratio,
        ]
        post_rows.append(post_vec)

        feat_dict = dict(zip(PRE_GOONG_STRUCT_FEATURES, pre_vec))
        feat_dict.update(dict(zip(POST_GOONG_CONSISTENCY_FEATURES, post_vec)))
        buckets.append(classify_post_goong_error_bucket(feat_dict))

    pre_df = pd.DataFrame(pre_rows, columns=list(PRE_GOONG_STRUCT_FEATURES), index=df.index)
    post_df = pd.DataFrame(post_rows, columns=list(POST_GOONG_CONSISTENCY_FEATURES), index=df.index)
    return pre_df, post_df, buckets
