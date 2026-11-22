"""Feature extraction and conservative VN admin canonicalization for Post-Goong V3."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = PACKAGE_DIR / "artifacts"
_FALLBACK_CONVERTER_PATH = PACKAGE_DIR.parent / "shared_data_v3" / "vn_admin_v3_converter.json"

FEATURE_NAMES: tuple[str, ...] = (
    # Pre-Goong structure (17)
    "char_len",
    "word_count",
    "digit_count",
    "comma_count",
    "has_house_num",
    "has_ward_pfx",
    "has_dist_pfx",
    "has_prov_pfx",
    "has_street_kw",
    "has_rural_kw",
    "has_apt_kw",
    "has_chat_food_cue",
    "raw_ward_hits",
    "raw_dist_hits",
    "raw_prov_hits",
    "raw_valid_pair",
    "raw_min_ward_ambig",
    # Post-Goong grounding (11)
    "goong_has_ward",
    "goong_has_dist",
    "goong_has_prov",
    "goong_filled_levels",
    "ward_exact_grounded",
    "dist_exact_grounded",
    "prov_exact_grounded",
    "ward_char_sim",
    "dist_char_sim",
    "prov_char_sim",
    "admin_token_grounding_ratio",
    # Post-Goong errors (10)
    "hallucinated_levels_count",
    "ward_hallucinated",
    "parent_hallucinated",
    "ambiguous_ward_ungrounded_parent",
    "raw_prov_contradicts_goong",
    "raw_dist_contradicts_goong",
    "goong_ward_prov_count",
    "log_goong_ward_ambiguity",
    "goong_score",
    "goong_score_missing",
    # 2025 equivalence & spatial checks (5)
    "is_legacy_2025_equiv",
    "merged_prov_equiv",
    "spatial_dist_km",
    "spatial_outlier",
    "spatial_missing",
)

_HYPHEN_PROTECT_TOKEN = " _hy_ "

# Vietnamese ASCII syllable structure (for conservative Telex/VNI cleanup)
_VN_SYLLABLE_RE = re.compile(
    r"^(?:b|c|ch|d|g|gh|gi|h|k|kh|l|m|n|ng|ngh|nh|p|ph|qu|r|s|t|th|tr|v|x)?"
    r"(?:[aeiouy]{1,3})"
    r"(?:c|ch|m|n|ng|nh|p|t)?$"
)

# Common English / brand / chat / food words that must NEVER be Telex-collapsed
_ENGLISH_WHITELIST: frozenset[str] = frozenset(
    {
        "goong", "google", "good", "food", "foods", "coffee", "cafe", "tea", "milk",
        "juice", "cake", "cakes", "bread", "rice", "pizza", "burger", "sushi", "beer",
        "ice", "hot", "fresh", "sweet", "box", "pack", "order", "ship", "shipper",
        "grab", "bee", "express", "plus", "bus", "mart", "shop", "store", "mall",
        "center", "plaza", "tower", "villa", "park", "garden", "home", "house",
        "land", "city", "town", "street", "road", "floor", "room", "block", "gate",
        "green", "gold", "silver", "blue", "star", "smart", "fast", "super", "mini",
        "vip", "max", "pro", "cool", "free", "pool", "look", "book", "books", "moon",
        "soon", "noon", "pass", "boss", "miss", "less", "glass", "class", "cross",
        "grass", "press", "dress", "tax", "fax", "gas", "bar", "car", "tar", "mix",
        "six", "fix", "lotus", "campus", "news", "kids", "boys", "girls", "nails",
        "shoes", "clothes", "drinks", "apps", "app", "web", "link", "wifi", "size",
        "sale", "driver", "royal", "lucky", "happy", "smile", "river", "view", "lake",
        "hill", "sun", "sea", "sky", "rose", "iris", "lily", "ruby", "king", "queen",
        "prince", "spa", "gym", "baby", "zalo", "zns", "sms", "lunch", "dinner",
        "vietnam", "hanoi", "saigon", "danang", "haiphong", "cantho", "tphcm",
        "dalat", "halong", "daklak", "daknong", "brvt", "tthue", "texas",
    }
)

# Common address structural tokens protected from VNI trailing digit stripping
_ADDR_NUM_PREFIX_TOKENS: frozenset[str] = frozenset(
    {
        "so", "lo", "ngo", "hem", "ngach", "kiet", "tang", "lau", "phong", "can",
        "nha", "khu", "to", "ap", "thon", "xom", "doi", "cum", "block", "tower",
        "gate", "quan", "phuong", "xa", "duong", "pho", "f", "p", "q",
    }
)

_VN_ADDRESS_KEYWORDS: frozenset[str] = frozenset(
    {
        "phuong", "xa", "thi", "tran", "dac", "khu", "quan", "huyen", "thanh",
        "pho", "tinh", "duong", "hem", "ngo", "ngach", "kiet", "so", "nha",
        "chung", "cu", "toa", "tang", "lau", "can", "ho", "lo", "ap", "thon",
        "xom", "ban", "buon", "doi", "cum", "gan", "cho", "truoc", "sau", "giao",
        "goi", "cong", "nga", "tu", "ba", "nam", "bac", "trung", "dong", "tay",
        "moi", "cu", "lon", "nho", "dai", "quoc",
    }
)

_CHAT_FOOD_RE = re.compile(
    r"\b(?:"
    r"tra sua|tran chau|it da|it duong|nhieu da|khong da|com tam|suon bi|"
    r"banh mi|bun bo|bun dau|pho bo|hu tieu|mi cay|ga ran|tra dao|"
    r"ca phe|cafe|bac xiu|nuoc ep|sinh to|pizza|burger|sushi|"
    r"ao thun|ao khoac|quan jean|vay dam|size [smlx]{1,3}|"
    r"shipper|dat hang|chot don|don hang|kiem tra hang|cho em hoi|hoi ti|"
    r"pass wifi|wifi la|cam on|thank|thanks|xin chao|hi shop|alo|"
    r"bao nhieu tien|gia bao nhieu|con hang khong|giam gia|freeship|"
    r"ly|phan|suat|hop|goi|chai|to|dia"
    r")\b"
)

_ADDRESS_ANCHOR_RE = re.compile(
    r"\b(?:phuong|xa|thi tran|dac khu|quan|huyen|thi xa|thanh pho|tinh|"
    r"duong|hem|ngo|ngach|kiet|chung cu|toa|khu|ap|thon|xom)\b|"
    r"\b[pq]\.?\s*\d{1,2}\b|\b(?:tphcm|hcm|hn|brvt)\b"
)

_PHONE_RE = re.compile(r"(?:\+?84|0)[35789]\d{8}\b")
_HOUSE_NUM_RE = re.compile(
    r"(?:^|[,;\s])(?:so\s+nha\s+|so\s+|nha\s+)?\d{1,5}(?:/\d{1,4})?[a-z]?(?=\s+[a-z])"
)
_STREET_KW_RE = re.compile(
    r"\b(?:duong|pho|dai lo|hem|ngo|ngach|kiet|quoc lo|tinh lo|ql\s*\d+|tl\s*\d+|mat tien|nga tu|vong xoay)\b"
)
_RURAL_KW_RE = re.compile(
    r"\b(?:thon|ap|xom|doi|cum|ban|buon|soc|lang|khu vuc|to\s+\d+|khu\s+\d+|khom|tieu khu)\b"
)
_APT_KW_RE = re.compile(
    r"\b(?:chung cu|cc|can ho|toa|tower|block|tang|lau|phong|lo\s+[a-z0-9]+|khu do thi|kdt|vinhomes|masteri|landmark|ecopark)\b"
)

_ACCENTED_MAP = str.maketrans(
    "àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ",
    "aaaaaaaaaaaaaaaaadeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyy"
    "aaaaaaaaaaaaaaaaadeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyy",
)


_RE_TP_HCM = re.compile(r"\btp\.?\s*hcm\b")
_RE_TP_HN = re.compile(r"\btp\.?\s*hn\b")
_RE_BR_VT = re.compile(r"\bbr\s*-\s*vt\b")
_RE_TP_DOT = re.compile(r"\btp\.\s*")
_RE_TX_DOT = re.compile(r"\btx\.\s*")
_RE_TT_DOT = re.compile(r"\btt\.\s*")
_RE_H_DOT = re.compile(r"\bh\.\s*")
_RE_X_DOT = re.compile(r"\bx\.\s*")
_RE_P_NUM = re.compile(r"\b[pf]\.?\s*(\d{1,2})\b")
_RE_P_DOT = re.compile(r"\bp\.\s*")
_RE_Q_NUM = re.compile(r"\b[qd]\.?\s*(\d{1,2})\b")
_RE_Q_DOT = re.compile(r"\bq\.\s*")
_RE_SPACES = re.compile(r"\s+")
_RE_SPLIT_PARTS = re.compile(r"([A-Za-zÀ-ỹĐđ0-9']+)")
_RE_ALPHA_NUM = re.compile(r"^[A-Za-z0-9]+$")
_RE_SINGLE_CODE = re.compile(r"^[a-z]\d+$")
_RE_NUM_UNIT = re.compile(r"^\d+[a-z]+$")
_RE_VNI_MOD = re.compile(r"(?:[aeou][678]|d9)")
_RE_VNI_MID = re.compile(r"[a-z]{2,}[1-5][a-z]+")
_RE_VNI_END = re.compile(r"^[a-z]{2,}[1-5]$")
_RE_VNI_DIGITS = re.compile(r"[1-5]")
_RE_TELEX_HARD_CHARS = re.compile(r"[fjw]")
_RE_TELEX_SOFT_CHARS = re.compile(r"[rsx]")
_RE_PARENS_END = re.compile(r"\s*\([^)]*\)\s*$")
_RE_ALNUM_TOKENS = re.compile(r"[a-z0-9]+")


def strip_accents(text: str) -> str:
    """Fold Vietnamese diacritics (including đ/Đ) to plain lowercase ASCII."""
    s = unicodedata.normalize("NFC", text or "").translate(_ACCENTED_MAP).lower()
    if s.isascii():
        return s
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")


def _collapse_telex_token(tok: str, extra_vocab: frozenset[str]) -> str:
    """Conservatively decode a single ASCII Telex/VNI typo token."""
    if len(tok) <= 2 or tok in _ENGLISH_WHITELIST or tok in extra_vocab:
        return tok
    if not tok.isalpha():
        # Skip house numbers, short codes (q1, p6, m2, v2, k3), or quantity tokens (50k, 24h)
        if tok[0].isdigit() or _RE_SINGLE_CODE.match(tok) or _RE_NUM_UNIT.match(tok):
            return tok
        has_vni_mod = bool(_RE_VNI_MOD.search(tok))
        has_mid_tone = bool(_RE_VNI_MID.search(tok))
        has_end_tone = bool(_RE_VNI_END.match(tok))
        if not (has_vni_mod or has_mid_tone or has_end_tone):
            return tok
        if has_end_tone and not has_vni_mod and tok[:-1] in _ADDR_NUM_PREFIX_TOKENS:
            return tok
        cand = "d" + tok[2:] if tok.startswith("d9") else tok
        cand = cand.replace("a6", "a").replace("a8", "a").replace("e6", "e")
        cand = cand.replace("o6", "o").replace("o7", "o").replace("u7", "u")
        cand = _RE_VNI_DIGITS.sub("", cand)
        if cand != tok and cand.isalpha() and (_VN_SYLLABLE_RE.match(cand) or cand in extra_vocab):
            return cand
        return tok

    if tok in _VN_ADDRESS_KEYWORDS:
        return tok
    has_hard_telex = (
        tok.startswith("dd")
        or ("f" in tok or "j" in tok or "w" in tok)
        or ("aa" in tok or "aw" in tok or "ee" in tok or "oo" in tok or "ow" in tok or "uw" in tok)
    )
    has_soft_tone = (not has_hard_telex) and len(tok) >= 4 and ("r" in tok[1:] or "s" in tok[1:] or "x" in tok[1:])
    if not (has_hard_telex or has_soft_tone):
        return tok

    cand = "d" + tok[2:] if tok.startswith("dd") else tok
    cand = (
        cand.replace("uw", "u")
        .replace("ow", "o")
        .replace("aw", "a")
        .replace("aa", "a")
        .replace("ee", "e")
        .replace("oo", "o")
    )
    if len(cand) >= 2:
        head, tail = cand[0], cand[1:]
        if head in ("f", "j", "w"):
            return tok
        tail_clean = _RE_TELEX_SOFT_CHARS.sub("", _RE_TELEX_HARD_CHARS.sub("", tail))
        cand = head + tail_clean

    if cand == tok:
        return tok
    if has_hard_telex:
        if _VN_SYLLABLE_RE.match(cand) or cand in extra_vocab or cand in _VN_ADDRESS_KEYWORDS:
            return cand
    elif (cand in extra_vocab or cand in _VN_ADDRESS_KEYWORDS) and _VN_SYLLABLE_RE.match(cand):
        return cand
    return tok


def cleanup_telex_vni(text: str, extra_vocab: frozenset[str] = frozenset()) -> str:
    """Conservatively clean stray Telex/VNI artifacts while preserving valid tokens & diacritics."""
    if not text:
        return ""
    s = unicodedata.normalize("NFC", text)
    parts = _RE_SPLIT_PARTS.split(s)
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        if _RE_ALPHA_NUM.match(part):
            low = part.lower()
            cleaned = _collapse_telex_token(low, extra_vocab)
            out.append(cleaned if cleaned != low else part)
        else:
            out.append(part)
    return "".join(out)


def normalize_vn_ascii(text: str, extra_vocab: frozenset[str] = frozenset()) -> str:
    """Apply conservative Telex/VNI cleanup, lowercase, fold diacritics, and tidy spacing."""
    cleaned = cleanup_telex_vni(text, extra_vocab=extra_vocab)
    ascii_str = strip_accents(cleaned)
    return _RE_SPACES.sub(" ", ascii_str).strip()


def _bigram_counts(s: str) -> tuple[dict[str, int], int]:
    if len(s) < 2:
        return ({s: 1}, 1) if s else ({}, 0)
    counts: dict[str, int] = {}
    n = len(s) - 1
    for i in range(n):
        bg = s[i : i + 2]
        counts[bg] = counts.get(bg, 0) + 1
    return counts, n


def _dice_with_precomputed(ca: dict[str, int], na: int, b: str) -> float:
    if not b or na == 0:
        return 0.0
    cb, nb = _bigram_counts(b)
    inter = 0
    for bg, cnt in cb.items():
        ac = ca.get(bg)
        if ac:
            inter += cnt if cnt < ac else ac
    return (2.0 * inter) / float(na + nb)


def sliding_window_bigram_sim(target: str, haystack_tokens: list[str], segments: list[str]) -> float:
    """Compute max character-bigram similarity of target over token windows & segments."""
    if not target or not haystack_tokens:
        return 0.0
    haystack_joined = " ".join(haystack_tokens)
    if target in haystack_joined and re.search(rf"(?<![a-z0-9]){re.escape(target)}(?![a-z0-9])", haystack_joined):
        return 1.0
    ca, na = _bigram_counts(target)
    k = len(target.split())
    best = 0.0
    for seg in segments:
        if seg:
            if seg == target:
                return 1.0
            sim = _dice_with_precomputed(ca, na, seg)
            if sim > best:
                best = sim
    n = len(haystack_tokens)
    w_min = max(1, k - 1)
    w_max = min(n, k + 1)
    for w in range(w_min, w_max + 1):
        for i in range(0, n - w + 1):
            cand = " ".join(haystack_tokens[i : i + w])
            if cand == target:
                return 1.0
            sim = _dice_with_precomputed(ca, na, cand)
            if sim > best:
                best = sim
                if best >= 0.999:
                    return 1.0
    return best


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points in kilometers."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat * 0.5) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon * 0.5) ** 2
    )
    return 2.0 * r * math.asin(math.sqrt(min(1.0, max(0.0, a))))


@dataclass
class ParsedAdminMentions:
    """Structured entities extracted from raw_text or Goong fields."""

    wards: set[str]
    dists: set[str]
    provs: set[str]
    implied_provs_from_dist: set[str]
    all_segments: list[str]
    tokens: list[str]
    has_ward_pfx: bool
    has_dist_pfx: bool
    has_prov_pfx: bool


@dataclass
class FeatureBundle:
    """Full output of feature extraction for model inference + rule telemetry."""

    vector: np.ndarray
    features: dict[str, float]
    is_legacy_2025_equiv: bool
    merged_prov_equiv: bool
    spatial_dist_km: float
    spatial_outlier: bool
    has_prov_contradiction: bool
    has_dist_contradiction: bool
    has_chat_food_only: bool
    goong_has_ward: bool
    goong_valid_admin: bool


class GazetteerV3:
    """Self-contained VN legacy (63) + 2025 (34) gazetteer and canonicalizer."""

    def __init__(self, converter_path: str | Path | None = None) -> None:
        path = self._resolve_converter_path(converter_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.po: list[str] = data["po"]
        self.po_disp: list[str] = data["po_disp"]
        self.do: list[str] = data["do"]
        self.do_disp: list[str] = data["do_disp"]
        self.wo: list[str] = data["wo"]
        self.wo_disp: list[str] = data["wo_disp"]
        self.pn: list[str] = data["pn"]
        self.pn_disp: list[str] = data["pn_disp"]
        self.wn: list[str] = data["wn"]
        self.wn_disp: list[str] = data["wn_disp"]
        self.pm: list[int] = data["pm"]
        self.dp: list[list[int]] = data["dp"]

        self.po_to_idx: dict[str, int] = {name: i for i, name in enumerate(self.po)}
        self.pn_to_idx: dict[str, int] = {name: i for i, name in enumerate(self.pn)}
        self.do_to_idx: dict[str, int] = {name: i for i, name in enumerate(self.do)}
        self.wo_to_idx: dict[str, int] = {name: i for i, name in enumerate(self.wo)}
        self.wn_to_idx: dict[str, int] = {name: i for i, name in enumerate(self.wn)}

        self.po_set: frozenset[str] = frozenset(self.po)
        self.pn_set: frozenset[str] = frozenset(self.pn)
        self.do_set: frozenset[str] = frozenset(self.do)
        self.wo_set: frozenset[str] = frozenset(self.wo)
        self.wn_set: frozenset[str] = frozenset(self.wn)

        self.pa: dict[str, str] = dict(data.get("pa", {}))
        self.da: dict[str, str] = dict(data.get("da", {}))
        self._augment_aliases()

        # Province aliases whose key is also a legacy district name (e.g., thu duc, nha trang)
        self.pa_district_overlap: frozenset[str] = frozenset(
            k for k in self.pa if k in self.do_set or k in self.wn_set
        )
        self.pa_strict: dict[str, str] = {
            k: v for k, v in self.pa.items() if k not in self.pa_district_overlap
        }

        # Hyphenated admin names that must survive ` - ` segment splitting
        self.protected_hyphenated: tuple[str, ...] = tuple(
            sorted(
                {
                    *(x for x in self.po if " - " in x),
                    *(x for x in self.do if " - " in x),
                    *(x for x in self.pn if " - " in x),
                    *(x for x in self.wn if " - " in x),
                },
                key=len,
                reverse=True,
            )
        )
        self._protected_hyphen_map: dict[str, str] = {}
        alt_parts: list[str] = []
        for phrase in self.protected_hyphenated:
            token_repl = phrase.replace(" - ", _HYPHEN_PROTECT_TOKEN)
            for variant in (phrase, phrase.replace(" - ", "-"), phrase.replace(" - ", " ")):
                self._protected_hyphen_map[variant] = token_repl
                alt_parts.append(re.escape(variant))
        self._protected_hyphen_re = re.compile(
            rf"(?<![a-z0-9])(?:{'|'.join(alt_parts)})(?![a-z0-9])"
        )
        self._canon_ward_cache: dict[str, str] = {}
        self._canon_dist_cache: dict[str, str] = {}
        self._canon_prov_cache: dict[str, str] = {}

        # Build fast admin word vocabulary for conservative Telex/VNI cleanup
        vocab_words: set[str] = set(_VN_ADDRESS_KEYWORDS)
        for coll in (self.po, self.do, self.wo, self.pn, self.wn):
            for name in coll:
                for w in name.replace("-", " ").split():
                    if w:
                        vocab_words.add(w)
        for k in list(self.pa.keys()) + list(self.da.keys()):
            for w in k.replace("-", " ").split():
                if w:
                    vocab_words.add(w)
        self.vocab_words: frozenset[str] = frozenset(vocab_words)

        # Build legacy <-> 2025 equivalence & geometry indices
        self.wo_to_provs: list[set[int]] = [set() for _ in range(len(self.wo))]
        self.wo_to_dists: list[set[int]] = [set() for _ in range(len(self.wo))]
        self.wo_do_set: set[tuple[int, int]] = set()
        self.wo_do_to_provs: dict[tuple[int, int], set[int]] = {}
        self.wo_po_to_dists: dict[tuple[int, int], set[int]] = {}
        self.eo_triplets: set[tuple[int, int, int]] = set()
        self.eo_forward: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
        self.eo_wo_do_forward: dict[tuple[int, int], set[tuple[int, int]]] = {}
        self.eo_wo_po_forward: dict[tuple[int, int], set[tuple[int, int]]] = {}

        for wo_idx, entries in enumerate(data["eo"]):
            for did_o, pid_o, wn_idx, pid_n, _is_div, _is_def in entries:
                self.wo_to_provs[wo_idx].add(pid_o)
                self.wo_to_dists[wo_idx].add(did_o)
                self.wo_do_set.add((wo_idx, did_o))
                self.wo_do_to_provs.setdefault((wo_idx, did_o), set()).add(pid_o)
                self.wo_po_to_dists.setdefault((wo_idx, pid_o), set()).add(did_o)
                self.eo_triplets.add((wo_idx, did_o, pid_o))
                self.eo_forward.setdefault((wo_idx, did_o, pid_o), set()).add((wn_idx, pid_n))
                self.eo_wo_do_forward.setdefault((wo_idx, did_o), set()).add((wn_idx, pid_n))
                self.eo_wo_po_forward.setdefault((wo_idx, pid_o), set()).add((wn_idx, pid_n))

        self.dp_sets: list[set[int]] = [set(p_list) for p_list in self.dp]
        self.do_to_pn_sets: list[set[int]] = [
            {self.pm[p] for p in p_list} for p_list in self.dp
        ]

        self.wn_pn_entries: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
        self.wn_to_provs: list[set[int]] = [set() for _ in range(len(self.wn))]
        for wn_idx, entries in enumerate(data["en"]):
            for pid_n, lat, lon, area, _code, _wt in entries:
                self.wn_to_provs[wn_idx].add(pid_n)
                if lat is not None and lon is not None:
                    self.wn_pn_entries.setdefault((wn_idx, pid_n), []).append(
                        (float(lat), float(lon), float(area or 30.0))
                    )

        self.rev_lookup: dict[tuple[int, int], set[tuple[int, int, int]]] = {}
        self.rev_wo_do: dict[tuple[int, int], set[tuple[int, int]]] = {}
        self.rev_wo_po: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for key, items in data["rev"].items():
            wn_idx_s, pid_n_s = key.split(":")
            k_pair = (int(wn_idx_s), int(pid_n_s))
            tri_set = self.rev_lookup.setdefault(k_pair, set())
            wd_set = self.rev_wo_do.setdefault(k_pair, set())
            wp_set = self.rev_wo_po.setdefault(k_pair, set())
            for wo_idx, did_o, pid_o in items:
                tri_set.add((wo_idx, did_o, pid_o))
                wd_set.add((wo_idx, did_o))
                wp_set.add((wo_idx, pid_o))

        self.isle_do_po_to_wn_pn: dict[tuple[int, int], tuple[int, int]] = {}
        self.isle_wn_pn_to_do_po: dict[tuple[int, int], tuple[int, int]] = {}
        for did_o, pid_o, wn_idx, pid_n in data.get("isle", []):
            self.isle_do_po_to_wn_pn[(did_o, pid_o)] = (wn_idx, pid_n)
            self.isle_wn_pn_to_do_po[(wn_idx, pid_n)] = (did_o, pid_o)

        self.ogeo_lookup: dict[tuple[int, int, int], tuple[float, float]] = {}
        for key, coords in data.get("ogeo", {}).items():
            w_s, d_s, p_s = key.split(":")
            if coords and coords[0] is not None and coords[1] is not None:
                self.ogeo_lookup[(int(w_s), int(d_s), int(p_s))] = (float(coords[0]), float(coords[1]))

        # Link legacy triplet -> target new area for accurate spatial outlier threshold
        self.eo_triplet_area: dict[tuple[int, int, int], float] = {}
        for wo_idx, entries in enumerate(data["eo"]):
            for did_o, pid_o, wn_idx, pid_n, _is_div, _is_def in entries:
                en_list = self.wn_pn_entries.get((wn_idx, pid_n))
                if en_list:
                    self.eo_triplet_area[(wo_idx, did_o, pid_o)] = en_list[0][2]

        # Max n-gram length across gazetteer keys
        self.max_ngram = 6

    @staticmethod
    def _resolve_converter_path(converter_path: str | Path | None) -> Path:
        if converter_path is not None:
            p = Path(converter_path)
            if p.is_dir():
                p = p / "vn_admin_v3_converter.json"
            if p.exists():
                return p
        local_art = DEFAULT_ARTIFACTS_DIR / "vn_admin_v3_converter.json"
        if local_art.exists():
            return local_art
        if _FALLBACK_CONVERTER_PATH.exists():
            return _FALLBACK_CONVERTER_PATH
        raise FileNotFoundError(
            f"Could not find vn_admin_v3_converter.json in {DEFAULT_ARTIFACTS_DIR} or {_FALLBACK_CONVERTER_PATH}"
        )

    def _augment_aliases(self) -> None:
        extra_pa = {
            "tp ho chi minh": "ho chi minh",
            "ho chi minh city": "ho chi minh",
            "tpho chi minh": "ho chi minh",
            "thanh pho ho chi minh": "ho chi minh",
            "tp ha noi": "ha noi",
            "ha noi city": "ha noi",
            "thua thien - hue": "hue",
            "tp da nang": "da nang",
            "tp hai phong": "hai phong",
            "tp can tho": "can tho",
            "br-vt": "ba ria - vung tau",
            "ba ria-vung tau": "ba ria - vung tau",
            "dac nong": "dak nong",
            "dak lak": "dak lak",
            "dac lac": "dak lak",
            "phan rang thap cham": "phan rang - thap cham",
        }
        for k, v in extra_pa.items():
            if k == "phan rang thap cham":
                self.da.setdefault(k, v)
            elif v in self.po_to_idx:
                self.pa.setdefault(k, v)
        extra_da = {
            "tp thu duc": "thu duc",
            "thanh pho thu duc": "thu duc",
            "hai ba trung": "hai ba trung",
            "phan rang thap cham": "phan rang - thap cham",
            "phan rang-thap cham": "phan rang - thap cham",
            "tp bien hoa": "bien hoa",
            "tp nha trang": "nha trang",
            "tp da lat": "da lat",
            "tp vung tau": "vung tau",
            "tp quy nhon": "quy nhon",
            "tp phan thiet": "phan thiet",
            "tp my tho": "my tho",
            "tp vinh": "vinh",
            "tp hue": "hue",
            "tp ha long": "ha long",
            "tp pleiku": "pleiku",
            "tp thu dau mot": "thu dau mot",
        }
        for k, v in extra_da.items():
            if v in self.do_to_idx:
                self.da.setdefault(k, v)

    def _preprocess_segment_text(self, text: str, pre_ascii: str | None = None) -> str:
        s = pre_ascii if pre_ascii is not None else normalize_vn_ascii(text, extra_vocab=self.vocab_words)
        if not s:
            return ""
        # Strip phone numbers before segment parsing
        if "0" in s or "84" in s:
            s = _PHONE_RE.sub(" ", s)
        # Expand compact abbreviations: tp.hcm / p.6 / p6 / q.7 / q7 / tx. / tt. / h. / x.
        if "tp" in s:
            s = _RE_TP_HCM.sub("thanh pho ho chi minh", s)
            s = _RE_TP_HN.sub("thanh pho ha noi", s)
            if "tp." in s:
                s = _RE_TP_DOT.sub("thanh pho ", s)
        if "br" in s:
            s = _RE_BR_VT.sub("ba ria - vung tau", s)
        if "." in s:
            s = _RE_TX_DOT.sub("thi xa ", s)
            s = _RE_TT_DOT.sub("thi tran ", s)
            s = _RE_H_DOT.sub("huyen ", s)
            s = _RE_X_DOT.sub("xa ", s)
        if any(c.isdigit() for c in s) or "." in s:
            s = _RE_P_NUM.sub(r"phuong \1", s)
            s = _RE_P_DOT.sub("phuong ", s)
            s = _RE_Q_NUM.sub(r"quan \1", s)
            s = _RE_Q_DOT.sub("quan ", s)
        # Protect compound hyphenated admin names before splitting on ` - `
        if "-" in s or "ba ria" in s or "phan rang" in s or "da lat" in s or "lang co" in s or "quoc tu" in s:
            s = self._protected_hyphen_re.sub(
                lambda m: self._protected_hyphen_map.get(m.group(0), m.group(0)), s
            )
        return _RE_SPACES.sub(" ", s).strip()

    def canonicalize_province(self, text: str) -> str:
        if not text:
            return ""
        cached = self._canon_prov_cache.get(text)
        if cached is not None:
            return cached
        s = self._preprocess_segment_text(text).replace(_HYPHEN_PROTECT_TOKEN, " - ")
        if s.startswith(("tinh ", "thanh pho ", "tp ")):
            s = re.sub(r"^(?:tinh|thanh pho|tp)\s+", "", s).strip()
        if "(" in s:
            s = _RE_PARENS_END.sub("", s).strip()
        res = s
        if s not in self.po_to_idx and s not in self.pn_to_idx and s in self.pa:
            res = self.pa[s]
        if len(self._canon_prov_cache) < 4096:
            self._canon_prov_cache[text] = res
        return res

    def canonicalize_district(self, text: str) -> str:
        if not text:
            return ""
        cached = self._canon_dist_cache.get(text)
        if cached is not None:
            return cached
        s = self._preprocess_segment_text(text).replace(_HYPHEN_PROTECT_TOKEN, " - ")
        if "(" in s:
            s = _RE_PARENS_END.sub("", s).strip()
        res = s
        if s in self.do_to_idx:
            res = s
        elif s in self.da:
            res = self.da[s]
        else:
            s_strip = re.sub(r"^(?:huyen|thi xa|thanh pho|quan|tx|tp|q|h)\s+", "", s).strip()
            if s_strip in self.do_to_idx:
                res = s_strip
            elif s_strip in self.da:
                res = self.da[s_strip]
            elif s.startswith("quan ") and re.fullmatch(r"quan \d{1,2}", s):
                res = s
            else:
                res = s_strip
        if len(self._canon_dist_cache) < 4096:
            self._canon_dist_cache[text] = res
        return res

    def canonicalize_ward(self, text: str, raw_accented: str | None = None) -> str:
        if not text:
            return ""
        cache_key = text if raw_accented is None or raw_accented == text else None
        if cache_key is not None:
            cached = self._canon_ward_cache.get(cache_key)
            if cached is not None:
                return cached
        s = self._preprocess_segment_text(text).replace(_HYPHEN_PROTECT_TOKEN, " - ")
        if "(" in s:
            s = _RE_PARENS_END.sub("", s).strip()
        res = s
        if re.fullmatch(r"(?:phuong|xa)\s+\d{1,2}", s):
            res = s
        else:
            keep_unstripped = False
            if raw_accented is not None and ("phương" in raw_accented.lower()):
                acc = unicodedata.normalize("NFC", raw_accented).lower().strip()
                if acc.startswith(("phương ", "phươ")) and (s in self.wo_to_idx or s in self.wn_to_idx):
                    keep_unstripped = True
            if not keep_unstripped:
                m = re.match(r"^(?:phuong|xa|thi tran|dac khu|p|x|tt)\s+(.+)$", s)
                if m:
                    rest = m.group(1).strip()
                    if rest.isdigit():
                        res = s
                    elif s in (self.wo_set | self.wn_set) and rest not in (self.wo_set | self.wn_set):
                        res = s
                    else:
                        res = rest
        if cache_key is not None and len(self._canon_ward_cache) < 8192:
            self._canon_ward_cache[cache_key] = res
        return res

    def _split_segments(self, pre_str: str, raw_acc: str) -> tuple[list[str], list[str]]:
        # Split ASCII preprocessed string on comma, semicolon, newline, or ' - '
        raw_parts = re.split(r"[,;\n]+|\s+-\s+", pre_str)
        segs = [p.replace(_HYPHEN_PROTECT_TOKEN, " - ").strip() for p in raw_parts if p.strip()]
        # Best-effort aligned accented segments (only used when lengths match)
        acc_clean = unicodedata.normalize("NFC", raw_acc).lower()
        for phrase in self.protected_hyphenated:
            # No-op if diacritics exist, just split conservatively on , ; \n
            pass
        acc_parts = [p.strip() for p in re.split(r"[,;\n]+|\s+-\s+", acc_clean) if p.strip()]
        if len(acc_parts) != len(segs):
            acc_parts = [""] * len(segs)
        return segs, acc_parts

    def _detect_segment_role(self, seg: str, acc_seg: str) -> str | None:
        """Detect explicit admin prefix role: 'ward', 'dist', 'prov', or None."""
        if acc_seg:
            if re.match(r"^(?:phường|xã|thị trấn|đặc khu|p\.|p\s+\d|x\.|tt\.)\b", acc_seg):
                return "ward"
            if re.match(r"^(?:quận|huyện|thị xã|q\.|q\s*\d|h\.|tx\.)\b", acc_seg):
                return "dist"
            if re.match(r"^(?:tỉnh|t\.)\b", acc_seg):
                return "prov"
        if re.match(r"^(?:phuong\s+\d|phuong\s+|thi tran\s+|dac khu\s+)", seg):
            # Guard 'phuong mai', 'phuong liet', etc. when unaccented and no 'phuong phuong ...'
            if seg in (self.wo_set | self.wn_set):
                return None
            return "ward"
        if re.match(r"^xa\s+", seg):
            rest = seg[3:].strip()
            if rest in (self.wo_set | self.wn_set):
                return "ward"
        if re.match(r"^(?:quan\s+\d|quan\s+|huyen\s+|thi xa\s+)", seg):
            return "dist"
        if re.match(r"^tinh\s+", seg):
            return "prov"
        if seg.startswith(("thanh pho ", "tp ")):
            rest = re.sub(r"^(?:thanh pho|tp)\s+", "", seg).strip()
            if rest in (self.po_set | self.pn_set) or rest in self.pa_strict:
                return "prov"
            if rest in self.do_set or rest in self.da:
                return "dist"
        return None

    @staticmethod
    def _is_street_or_house_segment(seg: str, acc_seg: str) -> bool:
        if re.match(r"^(?:duong|pho|dai lo|hem|ngo|ngach|kiet|quoc lo|tinh lo|ql\d|tl\d)\b", seg):
            return True
        if acc_seg and re.match(r"^(?:đường|phố|đại lộ|hẻm|ngõ|ngách|kiệt)\b", acc_seg):
            return True
        # House number prefix: '45 ...', '123/4 ...', 'so 5 ...', 'nha 12 ...'
        if re.match(r"^(?:so\s+nha\s+|so\s+|nha\s+)?\d{1,5}(?:/\d{1,4})?[a-z]?\b", seg):
            # Exclude 2025 Lam Dong ward names '1 bao loc', '2 bao loc', '3 bao loc'
            if seg not in ("1 bao loc", "2 bao loc", "3 bao loc"):
                return True
        if re.match(r"^(?:lo|can|tang|lau|phong|toa|khu|chung cu)\s+", seg):
            return True
        return False

    def _strip_house_number(self, seg: str) -> str:
        s = re.sub(
            r"^(?:so\s+nha\s+|so\s+|nha\s+)?\d{1,5}(?:/\d{1,4})?[a-z]?\s+",
            "",
            seg,
        ).strip()
        s = re.sub(r"^(?:duong|pho)\s+", "", s).strip()
        return s

    def _lookup_prov(self, cand: str, allow_pa_overlap: bool = True) -> str | None:
        if cand in self.po_set or cand in self.pn_set:
            return cand
        if cand in self.pa_strict:
            return self.pa_strict[cand]
        if allow_pa_overlap and cand in self.pa:
            return self.pa[cand]
        return None

    def _lookup_dist(self, cand: str) -> str | None:
        if cand in self.do_set:
            return cand
        if cand in self.da:
            return self.da[cand]
        return None

    def _lookup_ward(self, cand: str) -> str | None:
        if cand in self.wo_set or cand in self.wn_set:
            return cand
        return None

    def _parse_single_segment_rtl(
        self,
        seg: str,
        wards: set[str],
        dists: set[str],
        provs: set[str],
        implied_provs: set[str],
    ) -> None:
        """Right-to-left whole-token n-gram parser for no-comma addresses."""
        tokens = seg.replace(" - ", " - ").split()
        # Strip trailing delivery note words at end if preceded by admin words
        while len(tokens) > 2 and tokens[-1] in ("nhe", "nha", "a", "oi", "shop", "giao", "truoc"):
            tokens.pop()
        if not tokens:
            return

        # 1) Match province at right end
        matched_prov = False
        for k in range(min(self.max_ngram, len(tokens)), 0, -1):
            phrase = " ".join(tokens[-k:])
            p_clean = re.sub(r"^(?:tinh|thanh pho|tp)\s+", "", phrase).strip()
            has_p_pfx = p_clean != phrase
            # Single-token ambiguous province only matches at end if whole string or explicit prefix
            p_hit = self._lookup_prov(p_clean, allow_pa_overlap=(len(tokens) == k or has_p_pfx))
            if p_hit and (len(tokens) > k or p_hit in self.po_set or p_hit in self.pn_set):
                if p_clean in self.do_set and len(tokens) > k and not has_p_pfx:
                    # e.g., '... Ba Dinh' could be district at end, let dist check handle if no prov
                    continue
                provs.add(p_hit)
                tokens = tokens[:-k]
                matched_prov = True
                break

        # 2) Match district at new right end
        if tokens:
            for k in range(min(self.max_ngram, len(tokens)), 0, -1):
                phrase = " ".join(tokens[-k:])
                d_clean = re.sub(r"^(?:huyen|thi xa|thanh pho|quan|tx|tp)\s+", "", phrase).strip()
                if re.fullmatch(r"quan \d{1,2}", phrase):
                    d_clean = phrase
                d_hit = self._lookup_dist(d_clean)
                if d_hit:
                    # Avoid consuming entire street name as district when house number sits right before it
                    # e.g. '123 Hai Ba Trung' alone without ward/prov
                    if len(tokens) - k == 1 and re.fullmatch(r"\d+[a-z]?", tokens[0]) and not matched_prov:
                        continue
                    # For 1-token bare district ('chu', 'vinh', 'lak'), require >= 2 tokens left or explicit prefix
                    if len(d_clean.split()) == 1 and d_clean == phrase and len(tokens) == k:
                        continue
                    dists.add(d_hit)
                    if d_hit in self.pa_district_overlap:
                        implied_provs.add(self.pa[d_hit])
                    tokens = tokens[:-k]
                    break

        # 3) Match ward in remaining tokens (from right end, skipping house number)
        if tokens:
            # Strip leading house number/street prefix if tokens start with digits
            rem = list(tokens)
            if re.match(r"^(?:\d+(?:/\d+)?[a-z]?|so|nha)$", rem[0]):
                rem = rem[1:]
                if rem and rem[0].isdigit():
                    rem = rem[1:]
            if rem and rem[0] in ("duong", "pho"):
                rem = rem[1:]
            for k in range(min(self.max_ngram, len(rem)), 0, -1):
                phrase = " ".join(rem[-k:])
                w_clean = self.canonicalize_ward(phrase)
                w_hit = self._lookup_ward(w_clean)
                if w_hit:
                    # Avoid 1-word ambiguous ward hit inside long unparsed noise unless whole remaining
                    if len(w_hit.split()) == 1 and w_clean == phrase and len(rem) > 2:
                        continue
                    wards.add(w_hit)
                    break

    def parse_raw_mentions(self, raw_text: str, pre_ascii: str | None = None) -> ParsedAdminMentions:
        """Extract raw ward/district/province mentions carefully avoiding collisions."""
        ascii_raw = pre_ascii if pre_ascii is not None else normalize_vn_ascii(raw_text, self.vocab_words)
        pre = self._preprocess_segment_text(raw_text, pre_ascii=ascii_raw)
        segs, acc_segs = self._split_segments(pre, raw_text)

        wards: set[str] = set()
        dists: set[str] = set()
        provs: set[str] = set()
        implied_provs: set[str] = set()
        has_ward_pfx = False
        has_dist_pfx = False
        has_prov_pfx = False

        # Scan explicit prefix cues over the full preprocessed string
        if re.search(r"\b(?:phuong|xa|thi tran|dac khu)\s+[a-z0-9]", pre):
            has_ward_pfx = True
        if re.search(r"\b(?:quan|huyen|thi xa)\s+[a-z0-9]", pre):
            has_dist_pfx = True
        if re.search(r"\b(?:tinh|thanh pho)\s+[a-z]", pre) or re.search(
            r"\b(?:tphcm|hcm|tphn|brvt)\b", ascii_raw
        ):
            has_prov_pfx = True

        n = len(segs)
        if n == 1:
            role = self._detect_segment_role(segs[0], acc_segs[0])
            if role == "ward":
                has_ward_pfx = True
                w = self.canonicalize_ward(segs[0], acc_segs[0])
                if self._lookup_ward(w):
                    wards.add(w)
            elif role == "dist":
                has_dist_pfx = True
                d = self.canonicalize_district(segs[0])
                if self._lookup_dist(d):
                    dists.add(d)
            elif role == "prov":
                has_prov_pfx = True
                p = self.canonicalize_province(segs[0])
                p_hit = self._lookup_prov(p)
                if p_hit:
                    provs.add(p_hit)
            else:
                self._parse_single_segment_rtl(segs[0], wards, dists, provs, implied_provs)
        elif n > 1:
            # First pass: resolve explicit prefix segments & unambiguous trailing province
            resolved_roles: list[str | None] = [None] * n
            for i, (seg, acc_seg) in enumerate(zip(segs, acc_segs)):
                role = self._detect_segment_role(seg, acc_seg)
                if role == "ward":
                    has_ward_pfx = True
                    w = self.canonicalize_ward(seg, acc_seg)
                    if self._lookup_ward(w):
                        wards.add(w)
                        resolved_roles[i] = "ward"
                elif role == "dist":
                    has_dist_pfx = True
                    d = self.canonicalize_district(seg)
                    if self._lookup_dist(d):
                        dists.add(d)
                        if d in self.pa_district_overlap:
                            implied_provs.add(self.pa[d])
                        resolved_roles[i] = "dist"
                elif role == "prov":
                    has_prov_pfx = True
                    p = self.canonicalize_province(seg)
                    p_hit = self._lookup_prov(p)
                    if p_hit:
                        provs.add(p_hit)
                        resolved_roles[i] = "prov"

            # Second pass: resolve bare segments right-to-left using positional VN hierarchy
            for i in range(n - 1, -1, -1):
                if resolved_roles[i] is not None:
                    continue
                seg = segs[i]
                acc_seg = acc_segs[i]
                is_street = self._is_street_or_house_segment(seg, acc_seg)

                if is_street:
                    # Street or house-numbered segment MUST NOT override resolved dist or prov.
                    if not re.match(r"^(?:duong|pho|dai lo|hem|ngo|ngach|kiet)\b", seg):
                        tail = self._strip_house_number(seg)
                        if tail:
                            if not wards:
                                w_hit = self._lookup_ward(self.canonicalize_ward(tail, acc_seg))
                                if w_hit:
                                    wards.add(w_hit)
                                    resolved_roles[i] = "ward"
                                    continue
                            if not dists and not wards:
                                d_hit = self._lookup_dist(self.canonicalize_district(tail))
                                if d_hit and d_hit not in self.po_set:
                                    dists.add(d_hit)
                                    if d_hit in self.pa_district_overlap:
                                        implied_provs.add(self.pa[d_hit])
                                    resolved_roles[i] = "dist"
                    continue

                # Rightmost bare segment: province or district (or ward if 2-seg like 'So 5 gan cho, Tan Lap')
                if i == n - 1:
                    # Check if seg is in po/pn/pa_strict
                    p_hit = self._lookup_prov(seg, allow_pa_overlap=False)
                    d_hit = self._lookup_dist(seg)
                    w_hit = self._lookup_ward(seg)
                    if p_hit and not d_hit:
                        # e.g. 'Long An' in 'Xa Hoa Binh, Long An' -> strictly province!
                        provs.add(p_hit)
                        resolved_roles[i] = "prov"
                        continue
                    if p_hit and d_hit:
                        # Ambiguous name like 'Hoa Binh' or 'Bac Giang' or 'Ha Tinh'
                        # If preceded by a ward/dist that pairs with d_hit and not p_hit, check; else prov
                        if n == 2 and wards:
                            # If seg[0] ward is in dp[d_hit] and prov is ambiguous, add prov (and implied)
                            provs.add(p_hit)
                            resolved_roles[i] = "prov"
                            continue
                        provs.add(p_hit)
                        resolved_roles[i] = "prov"
                        continue
                    if d_hit:
                        # e.g. 'Loc Ninh' in 'Loc Thanh, Loc Ninh' or 'Nha Trang' in 'Vinh Hai, Nha Trang'
                        dists.add(d_hit)
                        if d_hit in self.pa_district_overlap:
                            implied_provs.add(self.pa[d_hit])
                        resolved_roles[i] = "dist"
                        continue
                    if w_hit:
                        # e.g. 'Tan Lap' in 'So 5 gan cho, Tan Lap'
                        wards.add(w_hit)
                        resolved_roles[i] = "ward"
                        continue
                    # Fallback: rightmost segment might have trailing delivery text ('Ha Noi giao gio hanh chinh')
                    self._parse_single_segment_rtl(seg, wards, dists, provs, implied_provs)
                    continue

                # Non-rightmost bare segment:
                d_hit = self._lookup_dist(seg)
                w_hit = self._lookup_ward(seg)
                if d_hit and not w_hit:
                    dists.add(d_hit)
                    if d_hit in self.pa_district_overlap:
                        implied_provs.add(self.pa[d_hit])
                    resolved_roles[i] = "dist"
                elif w_hit and not d_hit:
                    # If a later segment already claimed ward with explicit prefix and i == 0, ignore street name!
                    if not (i == 0 and any(resolved_roles[j] == "ward" for j in range(1, n))):
                        wards.add(w_hit)
                        resolved_roles[i] = "ward"
                elif w_hit and d_hit:
                    # Both ward and district name exist (e.g. 'Ba Dinh', 'An Phu', 'Tan Binh', 'Loc Ninh')
                    # Use position & neighbor compatibility:
                    if i == 0:
                        # Leading bare segment before district/province -> ward (unless n==2 and seg[1] is prov)
                        if n == 2 and resolved_roles[1] == "prov" and not wards:
                            # Could be 'Ba Dinh, Ha Noi' (dist, prov) or 'Hong Ha, Ha Noi' (ward, prov)
                            p_r = next(iter(provs))
                            pid_o = self.po_to_idx.get(p_r)
                            pid_n = self.pn_to_idx.get(p_r)
                            did_o = self.do_to_idx.get(d_hit)
                            if did_o is not None and pid_o is not None and pid_o in self.dp_sets[did_o]:
                                dists.add(d_hit)
                                resolved_roles[i] = "dist"
                            wn_idx = self.wn_to_idx.get(w_hit)
                            if wn_idx is not None and pid_n is not None and pid_n in self.wn_to_provs[wn_idx]:
                                wards.add(w_hit)
                                resolved_roles[i] = "ward"
                            wo_idx = self.wo_to_idx.get(w_hit)
                            if wo_idx is not None and pid_o is not None and pid_o in self.wo_to_provs[wo_idx]:
                                wards.add(w_hit)
                                resolved_roles[i] = "ward"
                        else:
                            wards.add(w_hit)
                            resolved_roles[i] = "ward"
                    else:
                        # Middle segment (e.g. seg[0]=ward, seg[1]=dist, seg[2]=prov)
                        if any(resolved_roles[j] == "dist" for j in range(i + 1, n)):
                            wards.add(w_hit)
                            resolved_roles[i] = "ward"
                        else:
                            dists.add(d_hit)
                            if d_hit in self.pa_district_overlap:
                                implied_provs.add(self.pa[d_hit])
                            resolved_roles[i] = "dist"
                else:
                    # Might be a bare street name like 'Hai Ba Trung' before 'P.6, Q3', or wrapped with note
                    if i == 0 and any(resolved_roles[j] in ("ward", "dist") for j in range(1, n)):
                        continue
                    # Check if segment contains a province/district/ward via RTL token match only when no role resolved yet
                    if i == n - 2 and not dists and not wards:
                        self._parse_single_segment_rtl(seg, wards, dists, provs, implied_provs)

        tokens = re.findall(r"[a-z0-9]+", pre.replace(_HYPHEN_PROTECT_TOKEN, " "))
        return ParsedAdminMentions(
            wards=wards,
            dists=dists,
            provs=provs,
            implied_provs_from_dist=implied_provs,
            all_segments=[s.replace(_HYPHEN_PROTECT_TOKEN, " - ") for s in segs],
            tokens=tokens,
            has_ward_pfx=has_ward_pfx,
            has_dist_pfx=has_dist_pfx,
            has_prov_pfx=has_prov_pfx,
        )

    def _check_raw_valid_pair(self, parsed: ParsedAdminMentions) -> bool:
        # 1) Legacy (wo, do) or (wo, po) or (do, po)
        for w in parsed.wards:
            wo_idx = self.wo_to_idx.get(w)
            wn_idx = self.wn_to_idx.get(w)
            for d in parsed.dists:
                did_o = self.do_to_idx.get(d)
                if wo_idx is not None and did_o is not None and (wo_idx, did_o) in self.wo_do_set:
                    if not parsed.provs:
                        return True
                    for p in parsed.provs:
                        pid_o = self.po_to_idx.get(p)
                        if pid_o is not None and (wo_idx, did_o, pid_o) in self.eo_triplets:
                            return True
            for p in parsed.provs:
                pid_o = self.po_to_idx.get(p)
                pid_n = self.pn_to_idx.get(p)
                if wn_idx is not None and pid_n is not None and (wn_idx, pid_n) in self.wn_pn_entries:
                    return True
                if wo_idx is not None and pid_o is not None and pid_o in self.wo_to_provs[wo_idx]:
                    return True
        # 2) (do, po) pair (e.g., island or district+prov)
        for d in parsed.dists:
            did_o = self.do_to_idx.get(d)
            if did_o is not None:
                for p in parsed.provs:
                    pid_o = self.po_to_idx.get(p)
                    if pid_o is not None and pid_o in self.dp_sets[did_o]:
                        return True
        return False

    def _raw_min_ward_ambiguity(self, parsed: ParsedAdminMentions) -> float:
        if not parsed.wards:
            return 0.0
        min_amb = 999.0
        for w in parsed.wards:
            wo_idx = self.wo_to_idx.get(w)
            wn_idx = self.wn_to_idx.get(w)
            c_old = len(self.wo_to_provs[wo_idx]) if wo_idx is not None else 0
            c_new = len(self.wn_to_provs[wn_idx]) if wn_idx is not None else 0
            c = c_old if c_old > 0 else c_new
            if c > 0 and c < min_amb:
                min_amb = float(c)
        return 0.0 if min_amb == 999.0 else min_amb

    def _check_2025_and_prov_equiv(
        self,
        parsed: ParsedAdminMentions,
        gw: str,
        gd: str,
        gp: str,
    ) -> tuple[bool, bool]:
        """Return (is_legacy_2025_equiv, merged_prov_equiv)."""
        g_pid_n = self.pn_to_idx.get(gp)
        g_pid_o = self.po_to_idx.get(gp)
        g_wn_idx = self.wn_to_idx.get(gw)
        g_wo_idx = self.wo_to_idx.get(gw)
        g_did_o = self.do_to_idx.get(gd)

        merged_prov_equiv = False
        all_raw_provs = parsed.provs | parsed.implied_provs_from_dist
        if g_pid_n is not None:
            for rp in all_raw_provs:
                r_pid_o = self.po_to_idx.get(rp)
                if r_pid_o is not None and self.pm[r_pid_o] == g_pid_n:
                    merged_prov_equiv = True
                    break
        if not merged_prov_equiv and g_pid_o is not None:
            for rp in all_raw_provs:
                r_pid_n = self.pn_to_idx.get(rp)
                if r_pid_n is not None and self.pm[g_pid_o] == r_pid_n:
                    merged_prov_equiv = True
                    break

        is_equiv = False
        # Direction 1: Raw is legacy -> Goong V2 is 2025 (wn, pn)
        if g_wn_idx is not None and g_pid_n is not None:
            for rw in parsed.wards:
                r_wo_idx = self.wo_to_idx.get(rw)
                if r_wo_idx is None:
                    continue
                if parsed.dists and parsed.provs:
                    for rd in parsed.dists:
                        r_did_o = self.do_to_idx.get(rd)
                        if r_did_o is None:
                            continue
                        for rp in parsed.provs:
                            r_pid_o = self.po_to_idx.get(rp)
                            if r_pid_o is not None:
                                targets = self.eo_forward.get((r_wo_idx, r_did_o, r_pid_o))
                                if targets and (g_wn_idx, g_pid_n) in targets:
                                    is_equiv = True
                                    break
                elif parsed.dists:
                    for rd in parsed.dists:
                        r_did_o = self.do_to_idx.get(rd)
                        if r_did_o is not None:
                            targets = self.eo_wo_do_forward.get((r_wo_idx, r_did_o))
                            if targets and (g_wn_idx, g_pid_n) in targets:
                                is_equiv = True
                                break
                elif parsed.provs:
                    for rp in parsed.provs:
                        r_pid_o = self.po_to_idx.get(rp)
                        if r_pid_o is not None:
                            targets = self.eo_wo_po_forward.get((r_wo_idx, r_pid_o))
                            if targets and (g_wn_idx, g_pid_n) in targets:
                                is_equiv = True
                                break
                else:
                    # Unambiguous legacy ward -> 2025 ward
                    if len(self.wo_to_provs[r_wo_idx]) == 1:
                        for d_cand in self.wo_to_dists[r_wo_idx]:
                            targets = self.eo_wo_do_forward.get((r_wo_idx, d_cand))
                            if targets and (g_wn_idx, g_pid_n) in targets:
                                is_equiv = True
                                break
                if is_equiv:
                    break

            # Island legacy district -> 2025 Dac khu
            if not is_equiv and parsed.dists:
                for rd in parsed.dists:
                    r_did_o = self.do_to_idx.get(rd)
                    if r_did_o is None:
                        continue
                    if parsed.provs:
                        for rp in parsed.provs:
                            r_pid_o = self.po_to_idx.get(rp)
                            if r_pid_o is not None and self.isle_do_po_to_wn_pn.get((r_did_o, r_pid_o)) == (g_wn_idx, g_pid_n):
                                is_equiv = True
                                break
                    else:
                        for p_cand in self.dp_sets[r_did_o]:
                            if self.isle_do_po_to_wn_pn.get((r_did_o, p_cand)) == (g_wn_idx, g_pid_n):
                                is_equiv = True
                                break

        # Direction 2: Raw is 2025 (wn, pn) -> Goong is legacy (wo, do, po)
        if not is_equiv and g_wo_idx is not None and (g_did_o is not None or g_pid_o is not None):
            for rw in parsed.wards:
                r_wn_idx = self.wn_to_idx.get(rw)
                if r_wn_idx is None:
                    continue
                for rp in parsed.provs:
                    r_pid_n = self.pn_to_idx.get(rp)
                    if r_pid_n is None:
                        continue
                    if g_did_o is not None and g_pid_o is not None:
                        if (g_wo_idx, g_did_o, g_pid_o) in self.rev_lookup.get((r_wn_idx, r_pid_n), ()):
                            is_equiv = True
                            break
                    elif g_did_o is not None:
                        if (g_wo_idx, g_did_o) in self.rev_wo_do.get((r_wn_idx, r_pid_n), ()):
                            is_equiv = True
                            break
                    elif g_pid_o is not None:
                        if (g_wo_idx, g_pid_o) in self.rev_wo_po.get((r_wn_idx, r_pid_n), ()):
                            is_equiv = True
                            break
                if is_equiv:
                    break

        if is_equiv and g_pid_n is not None and all_raw_provs:
            merged_prov_equiv = True
        return is_equiv, merged_prov_equiv

    def _compute_spatial_features(
        self,
        gw: str,
        gd: str,
        gp: str,
        goong_lat: float | None,
        goong_lng: float | None,
    ) -> tuple[float, bool, bool]:
        """Return (spatial_dist_km, spatial_outlier, spatial_missing)."""
        if goong_lat is None or goong_lng is None:
            return 0.0, False, True
        try:
            lat = float(goong_lat)
            lng = float(goong_lng)
        except (TypeError, ValueError):
            return 0.0, False, True
        if math.isnan(lat) or math.isnan(lng):
            return 0.0, False, True

        candidates: list[tuple[float, float, float]] = []
        g_wo_idx = self.wo_to_idx.get(gw)
        g_wn_idx = self.wn_to_idx.get(gw)
        g_did_o = self.do_to_idx.get(gd)
        g_pid_o = self.po_to_idx.get(gp)
        g_pid_n = self.pn_to_idx.get(gp)

        if g_wo_idx is not None and g_did_o is not None and g_pid_o is not None:
            coords = self.ogeo_lookup.get((g_wo_idx, g_did_o, g_pid_o))
            if coords is not None:
                area = self.eo_triplet_area.get((g_wo_idx, g_did_o, g_pid_o), 30.0)
                candidates.append((coords[0], coords[1], area))

        if g_wn_idx is not None and g_pid_n is not None:
            for clat, clon, area in self.wn_pn_entries.get((g_wn_idx, g_pid_n), ()):
                candidates.append((clat, clon, area))

        # Fallbacks if district or province omitted in Goong response
        if not candidates and g_wo_idx is not None and g_did_o is not None:
            for p_cand in self.wo_do_to_provs.get((g_wo_idx, g_did_o), ()):
                coords = self.ogeo_lookup.get((g_wo_idx, g_did_o, p_cand))
                if coords is not None:
                    area = self.eo_triplet_area.get((g_wo_idx, g_did_o, p_cand), 30.0)
                    candidates.append((coords[0], coords[1], area))

        if not candidates and g_wn_idx is not None and len(self.wn_to_provs[g_wn_idx]) == 1:
            only_pn = next(iter(self.wn_to_provs[g_wn_idx]))
            for clat, clon, area in self.wn_pn_entries.get((g_wn_idx, only_pn), ()):
                candidates.append((clat, clon, area))

        if not candidates and g_wo_idx is not None and len(self.wo_to_provs[g_wo_idx]) == 1:
            only_po = next(iter(self.wo_to_provs[g_wo_idx]))
            for d_cand in self.wo_po_to_dists.get((g_wo_idx, only_po), ()):
                coords = self.ogeo_lookup.get((g_wo_idx, d_cand, only_po))
                if coords is not None:
                    area = self.eo_triplet_area.get((g_wo_idx, d_cand, only_po), 30.0)
                    candidates.append((coords[0], coords[1], area))

        if not candidates:
            return 0.0, False, False

        best_dist = float("inf")
        best_area = 30.0
        for clat, clon, area in candidates:
            d_km = haversine_km(lat, lng, clat, clon)
            if d_km < best_dist:
                best_dist = d_km
                best_area = area

        thresh_km = max(20.0, 2.5 * math.sqrt(max(0.0, best_area)) + 10.0)
        is_outlier = best_dist > thresh_km
        return float(min(best_dist, 2000.0)), is_outlier, False

    def _is_goong_admin_valid(self, gw: str, gd: str, gp: str) -> bool:
        if not gw:
            return False
        g_wo_idx = self.wo_to_idx.get(gw)
        g_wn_idx = self.wn_to_idx.get(gw)
        g_did_o = self.do_to_idx.get(gd)
        g_pid_o = self.po_to_idx.get(gp)
        g_pid_n = self.pn_to_idx.get(gp)

        if g_did_o is not None and g_pid_o is not None and g_wo_idx is not None:
            if (g_wo_idx, g_did_o, g_pid_o) in self.eo_triplets:
                return True
        if g_wn_idx is not None and g_pid_n is not None and not gd:
            if (g_wn_idx, g_pid_n) in self.wn_pn_entries:
                return True
        # Island legacy district-level match or 2025 Dac khu
        if g_wn_idx is not None and g_pid_n is not None and (g_wn_idx, g_pid_n) in self.wn_pn_entries:
            return True
        return False

    def _is_dist_implied_uniquely(self, parsed: ParsedAdminMentions, gd: str, gp: str) -> bool:
        g_did_o = self.do_to_idx.get(gd)
        if g_did_o is None:
            return False
        g_pid_o = self.po_to_idx.get(gp)
        for rw in parsed.wards:
            wo_idx = self.wo_to_idx.get(rw)
            if wo_idx is None:
                continue
            if parsed.provs:
                for rp in parsed.provs:
                    pid_o = self.po_to_idx.get(rp)
                    if pid_o is not None:
                        cands = self.wo_po_to_dists.get((wo_idx, pid_o), set())
                        if len(cands) == 1 and g_did_o in cands:
                            return True
            else:
                # Ward uniquely determines district across whole country
                if len(self.wo_to_dists[wo_idx]) == 1 and g_did_o in self.wo_to_dists[wo_idx]:
                    if g_pid_o is None or g_pid_o in self.wo_to_provs[wo_idx]:
                        return True
        return False

    def _is_prov_implied_uniquely(self, parsed: ParsedAdminMentions, gd: str, gp: str) -> bool:
        g_pid_o = self.po_to_idx.get(gp)
        g_pid_n = self.pn_to_idx.get(gp)
        # 1) Grounded district uniquely maps to a single legacy/2025 province
        for rd in parsed.dists:
            did_o = self.do_to_idx.get(rd)
            if did_o is None:
                continue
            if g_pid_o is not None and len(self.dp_sets[did_o]) == 1 and g_pid_o in self.dp_sets[did_o]:
                return True
            if g_pid_n is not None and len(self.do_to_pn_sets[did_o]) == 1 and g_pid_n in self.do_to_pn_sets[did_o]:
                return True
            # (raw_ward, raw_dist) pair uniquely implies province
            for rw in parsed.wards:
                wo_idx = self.wo_to_idx.get(rw)
                if wo_idx is not None:
                    provs_o = self.wo_do_to_provs.get((wo_idx, did_o), set())
                    if g_pid_o is not None and len(provs_o) == 1 and g_pid_o in provs_o:
                        return True
                    provs_n = {self.pm[p] for p in provs_o}
                    if g_pid_n is not None and len(provs_n) == 1 and g_pid_n in provs_n:
                        return True
        # 2) Unambiguous ward alone uniquely implies province
        for rw in parsed.wards:
            wo_idx = self.wo_to_idx.get(rw)
            if wo_idx is not None and len(self.wo_to_provs[wo_idx]) == 1:
                only_po = next(iter(self.wo_to_provs[wo_idx]))
                if g_pid_o == only_po or g_pid_n == self.pm[only_po]:
                    return True
            wn_idx = self.wn_to_idx.get(rw)
            if wn_idx is not None and len(self.wn_to_provs[wn_idx]) == 1:
                only_pn = next(iter(self.wn_to_provs[wn_idx]))
                if g_pid_n == only_pn:
                    return True
        return False

    def extract(
        self,
        raw_text: str,
        goong_ward: str = "",
        goong_district: str = "",
        goong_province: str = "",
        goong_score: float | None = None,
        goong_lat: float | None = None,
        goong_lng: float | None = None,
    ) -> FeatureBundle:
        """Compute all domain-invariant verification features for a single Goong fill."""
        raw_s = raw_text or ""
        raw_ascii = normalize_vn_ascii(raw_s, extra_vocab=self.vocab_words)
        raw_no_phone = _PHONE_RE.sub(" ", raw_ascii) if ("0" in raw_ascii or "84" in raw_ascii) else raw_ascii
        parsed = self.parse_raw_mentions(raw_s, pre_ascii=raw_ascii)

        # Pre-Goong structural metrics
        char_len = float(len(raw_s.strip()))
        word_count = float(len(raw_s.strip().split())) if raw_s.strip() else 0.0
        digit_count = float(sum(1 for c in raw_s if c.isdigit()))
        comma_count = float(raw_s.count(","))
        has_house_num = 1.0 if _HOUSE_NUM_RE.search(raw_no_phone) else 0.0
        has_ward_pfx = 1.0 if parsed.has_ward_pfx else 0.0
        has_dist_pfx = 1.0 if parsed.has_dist_pfx else 0.0
        has_prov_pfx = 1.0 if parsed.has_prov_pfx else 0.0
        has_street_kw = 1.0 if _STREET_KW_RE.search(raw_no_phone) else 0.0
        has_rural_kw = 1.0 if _RURAL_KW_RE.search(raw_no_phone) else 0.0
        has_apt_kw = 1.0 if _APT_KW_RE.search(raw_no_phone) else 0.0
        has_chat_cue = bool(_CHAT_FOOD_RE.search(raw_no_phone))
        has_chat_food_cue = 1.0 if has_chat_cue else 0.0

        raw_ward_hits = float(len(parsed.wards))
        raw_dist_hits = float(len(parsed.dists))
        raw_prov_hits = float(len(parsed.provs))
        raw_valid_pair = 1.0 if self._check_raw_valid_pair(parsed) else 0.0
        raw_min_ward_ambig = self._raw_min_ward_ambiguity(parsed)

        # Canonicalize Goong output
        gw = self.canonicalize_ward(goong_ward or "", goong_ward or "")
        gd = self.canonicalize_district(goong_district or "")
        gp = self.canonicalize_province(goong_province or "")

        goong_has_ward = 1.0 if gw else 0.0
        goong_has_dist = 1.0 if gd else 0.0
        goong_has_prov = 1.0 if gp else 0.0
        goong_filled_levels = goong_has_ward + goong_has_dist + goong_has_prov

        # 2025 equivalence & merged province check
        is_equiv, merged_prov = self._check_2025_and_prov_equiv(parsed, gw, gd, gp)

        # Exact/alias grounding against parsed raw entities
        ward_exact = bool(gw and gw in parsed.wards)
        dist_exact = bool(gd and gd in parsed.dists)
        prov_exact = bool(gp and (gp in parsed.provs or gp in parsed.implied_provs_from_dist))
        # Also allow island district-to-ward match (e.g., raw 'Huyen Con Dao' -> Goong 'Dac khu Con Dao')
        if not ward_exact and gw and gw in parsed.dists and self.wn_to_idx.get(gw) is not None:
            ward_exact = True

        ward_sim = 1.0 if ward_exact else sliding_window_bigram_sim(gw, parsed.tokens, parsed.all_segments)
        dist_sim = 1.0 if dist_exact else sliding_window_bigram_sim(gd, parsed.tokens, parsed.all_segments)
        prov_sim = 1.0 if (prov_exact or merged_prov) else sliding_window_bigram_sim(
            gp, parsed.tokens, parsed.all_segments
        )

        # Contradictions (explicit raw prov/dist disagrees with Goong prov/dist, excluding 2025 equiv)
        raw_prov_contradicts = False
        if gp and parsed.provs and not prov_exact and not is_equiv and not merged_prov:
            raw_prov_contradicts = True

        raw_dist_contradicts = False
        if gd and parsed.dists and not dist_exact and not is_equiv:
            raw_dist_contradicts = True

        # Grounding & hallucination analysis
        goong_valid_admin = self._is_goong_admin_valid(gw, gd, gp)

        ward_hallucinated = False
        if gw:
            if is_equiv:
                ward_hallucinated = False
            elif ward_exact or ward_sim >= 0.85:
                ward_hallucinated = False
            else:
                ward_hallucinated = True

        dist_hallucinated = False
        if gd:
            if is_equiv:
                dist_hallucinated = False
            elif dist_exact or dist_sim >= 0.80:
                dist_hallucinated = False
            elif not ward_hallucinated and self._is_dist_implied_uniquely(parsed, gd, gp):
                dist_hallucinated = False
            else:
                dist_hallucinated = True

        prov_hallucinated = False
        if gp:
            if is_equiv or merged_prov:
                prov_hallucinated = False
            elif prov_exact or prov_sim >= 0.80:
                prov_hallucinated = False
            elif self._is_prov_implied_uniquely(parsed, gd, gp):
                prov_hallucinated = False
            else:
                prov_hallucinated = True

        # If Goong returned an invalid admin combination, mark parent hallucinated unless
        # both ward and district are explicitly grounded and form a valid (dist, prov) pair
        # (handles pre-2021 merged numbered wards like Phuong 6, Quan 3, TP.HCM).
        g_did_o = self.do_to_idx.get(gd)
        g_pid_o = self.po_to_idx.get(gp)
        valid_dp_pair = (
            g_did_o is not None and g_pid_o is not None and g_pid_o in self.dp_sets[g_did_o]
        )
        if gw and gp and not goong_valid_admin and not is_equiv:
            if not (ward_exact and dist_exact and valid_dp_pair):
                if gd or self.wn_to_idx.get(gw) is not None:
                    prov_hallucinated = True
            else:
                goong_valid_admin = True

        parent_hallucinated = dist_hallucinated or prov_hallucinated
        hallucinated_levels_count = float(
            int(ward_hallucinated) + int(dist_hallucinated) + int(prov_hallucinated)
        )

        ambiguous_ward_ungrounded_parent = (
            parent_hallucinated
            and raw_ward_hits > 0.0
            and raw_min_ward_ambig > 1.0
            and raw_valid_pair == 0.0
            and raw_prov_hits == 0.0
        )

        # Admin token grounding ratio over Goong's admin levels
        raw_tok_set = set(parsed.tokens)
        g_tokens_total = 0
        g_tokens_grounded = 0
        for level_str, is_lvl_grounded in (
            (gw, (not ward_hallucinated) and bool(gw)),
            (gd, (not dist_hallucinated) and bool(gd)),
            (gp, (not prov_hallucinated) and bool(gp)),
        ):
            if not level_str:
                continue
            toks = [t for t in level_str.replace("-", " ").split() if t]
            g_tokens_total += len(toks)
            if is_lvl_grounded:
                g_tokens_grounded += len(toks)
            else:
                g_tokens_grounded += sum(1 for t in toks if t in raw_tok_set)
        admin_token_grounding_ratio = (
            float(g_tokens_grounded) / float(g_tokens_total) if g_tokens_total > 0 else 0.0
        )

        # Goong ward ambiguity count (number of provinces having this ward name)
        goong_ward_prov_count = 0.0
        if gw:
            wo_i = self.wo_to_idx.get(gw)
            wn_i = self.wn_to_idx.get(gw)
            c_old = len(self.wo_to_provs[wo_i]) if wo_i is not None else 0
            c_new = len(self.wn_to_provs[wn_i]) if wn_i is not None else 0
            if gd and c_old > 0:
                goong_ward_prov_count = float(c_old)
            elif not gd and c_new > 0:
                goong_ward_prov_count = float(c_new)
            else:
                goong_ward_prov_count = float(max(c_old, c_new))
        log_goong_ward_ambiguity = math.log1p(goong_ward_prov_count)

        # Goong score
        if goong_score is None:
            g_score_val = 0.0
            g_score_missing = 1.0
        else:
            try:
                f_score = float(goong_score)
                if math.isnan(f_score):
                    g_score_val = 0.0
                    g_score_missing = 1.0
                else:
                    g_score_val = float(min(1.0, max(0.0, f_score)))
                    g_score_missing = 0.0
            except (TypeError, ValueError):
                g_score_val = 0.0
                g_score_missing = 1.0

        # Spatial check
        spatial_dist_km, spatial_outlier, spatial_missing = self._compute_spatial_features(
            gw, gd, gp, goong_lat, goong_lng
        )

        has_chat_food_only = (
            has_chat_cue
            and raw_ward_hits == 0.0
            and raw_dist_hits == 0.0
            and raw_prov_hits == 0.0
            and not _ADDRESS_ANCHOR_RE.search(raw_no_phone)
        )

        feat_dict: dict[str, float] = {
            "char_len": char_len,
            "word_count": word_count,
            "digit_count": digit_count,
            "comma_count": comma_count,
            "has_house_num": has_house_num,
            "has_ward_pfx": has_ward_pfx,
            "has_dist_pfx": has_dist_pfx,
            "has_prov_pfx": has_prov_pfx,
            "has_street_kw": has_street_kw,
            "has_rural_kw": has_rural_kw,
            "has_apt_kw": has_apt_kw,
            "has_chat_food_cue": has_chat_food_cue,
            "raw_ward_hits": raw_ward_hits,
            "raw_dist_hits": raw_dist_hits,
            "raw_prov_hits": raw_prov_hits,
            "raw_valid_pair": raw_valid_pair,
            "raw_min_ward_ambig": raw_min_ward_ambig,
            "goong_has_ward": goong_has_ward,
            "goong_has_dist": goong_has_dist,
            "goong_has_prov": goong_has_prov,
            "goong_filled_levels": goong_filled_levels,
            "ward_exact_grounded": 1.0 if (ward_exact or is_equiv) else 0.0,
            "dist_exact_grounded": 1.0 if (dist_exact or (is_equiv and not gd)) else 0.0,
            "prov_exact_grounded": 1.0 if (prov_exact or merged_prov or is_equiv) else 0.0,
            "ward_char_sim": float(ward_sim),
            "dist_char_sim": float(dist_sim),
            "prov_char_sim": float(prov_sim),
            "admin_token_grounding_ratio": float(admin_token_grounding_ratio),
            "hallucinated_levels_count": hallucinated_levels_count,
            "ward_hallucinated": 1.0 if ward_hallucinated else 0.0,
            "parent_hallucinated": 1.0 if parent_hallucinated else 0.0,
            "ambiguous_ward_ungrounded_parent": 1.0 if ambiguous_ward_ungrounded_parent else 0.0,
            "raw_prov_contradicts_goong": 1.0 if raw_prov_contradicts else 0.0,
            "raw_dist_contradicts_goong": 1.0 if raw_dist_contradicts else 0.0,
            "goong_ward_prov_count": goong_ward_prov_count,
            "log_goong_ward_ambiguity": float(log_goong_ward_ambiguity),
            "goong_score": g_score_val,
            "goong_score_missing": g_score_missing,
            "is_legacy_2025_equiv": 1.0 if is_equiv else 0.0,
            "merged_prov_equiv": 1.0 if merged_prov else 0.0,
            "spatial_dist_km": spatial_dist_km,
            "spatial_outlier": 1.0 if spatial_outlier else 0.0,
            "spatial_missing": 1.0 if spatial_missing else 0.0,
        }

        vec = np.array([feat_dict[name] for name in FEATURE_NAMES], dtype=np.float32)
        return FeatureBundle(
            vector=vec,
            features=feat_dict,
            is_legacy_2025_equiv=is_equiv,
            merged_prov_equiv=merged_prov,
            spatial_dist_km=spatial_dist_km,
            spatial_outlier=spatial_outlier,
            has_prov_contradiction=raw_prov_contradicts,
            has_dist_contradiction=raw_dist_contradicts,
            has_chat_food_only=has_chat_food_only,
            goong_has_ward=bool(gw),
            goong_valid_admin=goong_valid_admin,
        )


@lru_cache(maxsize=4)
def get_default_gazetteer(converter_path: str | None = None) -> GazetteerV3:
    """Cached GazetteerV3 loader."""
    return GazetteerV3(converter_path=converter_path)


def extract_features(
    raw_text: str,
    goong_ward: str = "",
    goong_district: str = "",
    goong_province: str = "",
    goong_score: float | None = None,
    goong_lat: float | None = None,
    goong_lng: float | None = None,
    gazetteer: GazetteerV3 | None = None,
) -> FeatureBundle:
    """Convenience functional entrypoint for feature extraction."""
    gz = gazetteer if gazetteer is not None else get_default_gazetteer()
    return gz.extract(
        raw_text=raw_text,
        goong_ward=goong_ward,
        goong_district=goong_district,
        goong_province=goong_province,
        goong_score=goong_score,
        goong_lat=goong_lat,
        goong_lng=goong_lng,
    )
