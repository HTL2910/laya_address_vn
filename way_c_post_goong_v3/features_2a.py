"""Domain-invariant 2A feature extractor for local real-order Post-Goong models (v3.1).

Reuses GazetteerV3 (2025 legacy equivalence, Telex/VNI cleanup, zero false
contradictions, Haversine spatial checks) and augments its 43 base features with
the proven 1B/2A structural, token-overlap, hierarchy-validity, and token-ratio
signals validated on 18,371 real orders with zero province-ID or raw-text leaks.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from .features import FeatureBundle, GazetteerV3, normalize_vn_ascii
except ImportError:  # pragma: no cover - direct script execution fallback
    from features import FeatureBundle, GazetteerV3, normalize_vn_ascii


FEATURE_NAMES_1B_24: tuple[str, ...] = (
    "char_len",
    "word_count",
    "digit_count",
    "alpha_ratio",
    "uppercase_ratio",
    "comma_count",
    "slash_count",
    "has_house_num",
    "has_slash_house_num",
    "has_ward_pfx",
    "has_dist_pfx",
    "has_prov_pfx",
    "has_street_kw",
    "has_rural_kw",
    "has_apt_kw",
    "has_delivery_note",
    "has_phone_in_text",
    "has_chat_food_cue",
    "has_vn_accent",
    "raw_ward_hits",
    "raw_dist_hits",
    "raw_prov_hits",
    "raw_valid_pair",
    "raw_min_ward_ambig",
)

FEATURE_NAMES_2A_51: tuple[str, ...] = FEATURE_NAMES_1B_24 + (
    "goong_has_ward",
    "goong_has_dist",
    "goong_has_prov",
    "goong_filled_levels",
    "ward_exact_grounded",
    "ward_token_overlap",
    "ward_char_sim",
    "dist_exact_grounded",
    "dist_token_overlap",
    "dist_char_sim",
    "prov_exact_grounded",
    "prov_token_overlap",
    "prov_char_sim",
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

FEATURE_NAMES_2A_V31: tuple[str, ...] = FEATURE_NAMES_2A_51 + (
    "is_legacy_2025_equiv",
    "merged_prov_equiv",
    "spatial_dist_km",
    "spatial_outlier",
    "spatial_missing",
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "v31_1b_24": FEATURE_NAMES_1B_24,
    "v31_2a_51": FEATURE_NAMES_2A_51,
    "v31_2a_56": FEATURE_NAMES_2A_V31,
}

_VN_ACCENT_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)
_SLASH_HOUSE_RE = re.compile(
    r"(?:^|[\s,])\d{1,4}[a-z]?(?:/\d{1,3}[a-z]?){1,3}(?=[\s,]|$)",
    re.IGNORECASE,
)
_DELIVERY_NOTE_RE = re.compile(
    r"\b(?:gio\s+hanh\s+chinh|ngoai\s+gio|goi\s+truoc|truoc\s+khi\s+giao|ship|"
    r"giao\s+giup|nhan\s+ho|bom|kiem\s+tra\s+hang|alo|toi\s+cong|de\s+o\s+bao\s+ve)\b",
    re.IGNORECASE,
)
_PHONE_ANY_RE = re.compile(r"(?<!\d)(?:\+?84|0)\d(?:[\s.\-]?\d){7,9}(?!\d)")
_RAW_WORD_RE = re.compile(r"[a-z0-9]+")


def _token_overlap(canon_str: str, exact_flag: float, raw_tok_set: set[str]) -> float:
    if not canon_str:
        return 0.0
    if exact_flag >= 0.5:
        return 1.0
    comp_toks = [t for t in canon_str.replace("-", " ").split() if t]
    if not comp_toks:
        return 0.0
    return float(sum(1 for t in comp_toks if t in raw_tok_set) / len(comp_toks))


class FeatureExtractor2A:
    """Extracts domain-invariant 1B/2A/v3.1 feature vectors backed by GazetteerV3."""

    def __init__(
        self,
        converter_path: str | Path | None = None,
        feature_set: str = "v31_2a_56",
        gazetteer: GazetteerV3 | None = None,
    ) -> None:
        if feature_set not in FEATURE_SETS:
            raise ValueError(
                f"Unknown feature_set {feature_set!r}. Expected one of {tuple(FEATURE_SETS)}"
            )
        self.feature_set = feature_set
        self.feature_names = FEATURE_SETS[feature_set]
        self.gazetteer = (
            gazetteer
            if gazetteer is not None
            else GazetteerV3(converter_path=converter_path)
        )

    def _compute_goong_hierarchy_valid(
        self, gw: str, gd: str, gp: str, base_valid: bool
    ) -> float:
        gz = self.gazetteer
        filled = int(bool(gw)) + int(bool(gd)) + int(bool(gp))
        if filled < 2:
            return -1.0
        if gw and base_valid:
            return 1.0

        g_wo_idx = gz.wo_to_idx.get(gw) if gw else None
        g_wn_idx = gz.wn_to_idx.get(gw) if gw else None
        g_did_o = gz.do_to_idx.get(gd) if gd else None
        g_pid_o = gz.po_to_idx.get(gp) if gp else None
        g_pid_n = gz.pn_to_idx.get(gp) if gp else None

        if gw and gd and not gp:
            if g_wo_idx is not None and g_did_o is not None:
                return 1.0 if (g_wo_idx, g_did_o) in gz.wo_do_set else 0.0
            if g_did_o is None or g_wo_idx is None:
                return -1.0

        if not gw and gd and gp:
            if g_did_o is not None and g_pid_o is not None:
                return 1.0 if g_pid_o in gz.dp_sets[g_did_o] else 0.0
            if g_did_o is not None and g_pid_n is not None:
                return 1.0 if g_pid_n in gz.do_to_pn_sets[g_did_o] else 0.0
            return -1.0

        if gw and gp:
            if g_wo_idx is not None and g_did_o is not None and g_pid_o is not None:
                return 1.0 if (g_wo_idx, g_did_o, g_pid_o) in gz.eo_triplets else 0.0
            if g_wn_idx is not None and g_pid_n is not None:
                return 1.0 if (g_wn_idx, g_pid_n) in gz.wn_pn_entries else 0.0
            if g_wo_idx is not None and g_pid_o is not None and not gd:
                return 1.0 if g_pid_o in gz.wo_to_provs[g_wo_idx] else 0.0
            if g_wo_idx is not None and g_pid_n is not None and not gd:
                return 1.0 if any(gz.pm[p] == g_pid_n for p in gz.wo_to_provs[g_wo_idx]) else 0.0
            return -1.0

        return -1.0

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
        """Compute full FeatureBundle whose vector matches `self.feature_names`."""
        gz = self.gazetteer
        base = gz.extract(
            raw_text=raw_text,
            goong_ward=goong_ward,
            goong_district=goong_district,
            goong_province=goong_province,
            goong_score=goong_score,
            goong_lat=goong_lat,
            goong_lng=goong_lng,
        )
        f = dict(base.features)
        raw_s = raw_text or ""
        raw_ascii = normalize_vn_ascii(raw_s, extra_vocab=gz.vocab_words)
        raw_no_phone = _PHONE_ANY_RE.sub(" ", raw_ascii) if ("0" in raw_ascii or "84" in raw_ascii) else raw_ascii
        raw_toks = _RAW_WORD_RE.findall(raw_no_phone)
        raw_tok_set = set(raw_toks)

        alpha_cnt = sum(1 for ch in raw_s if ch.isalpha())
        upper_cnt = sum(1 for ch in raw_s if ch.isupper())

        f["alpha_ratio"] = float(alpha_cnt / max(1, len(raw_s)))
        f["uppercase_ratio"] = float(upper_cnt / max(1, alpha_cnt))
        f["slash_count"] = float(raw_s.count("/"))
        f["has_slash_house_num"] = 1.0 if _SLASH_HOUSE_RE.search(raw_no_phone) else 0.0
        f["has_delivery_note"] = 1.0 if _DELIVERY_NOTE_RE.search(raw_no_phone) else 0.0
        f["has_phone_in_text"] = 1.0 if _PHONE_ANY_RE.search(raw_s) else 0.0
        f["has_vn_accent"] = 1.0 if _VN_ACCENT_RE.search(raw_s) else 0.0

        gw = gz.canonicalize_ward(goong_ward or "", goong_ward or "")
        gd = gz.canonicalize_district(goong_district or "")
        gp = gz.canonicalize_province(goong_province or "")

        f["ward_token_overlap"] = _token_overlap(gw, f["ward_exact_grounded"], raw_tok_set)
        f["dist_token_overlap"] = _token_overlap(gd, f["dist_exact_grounded"], raw_tok_set)
        f["prov_token_overlap"] = _token_overlap(gp, f["prov_exact_grounded"], raw_tok_set)

        # Nationwide district/province fan-out for Goong's selected ward
        w_dist_cnt = 0.0
        if gw:
            wo_i = gz.wo_to_idx.get(gw)
            wn_i = gz.wn_to_idx.get(gw)
            c_old = len(gz.wo_to_dists[wo_i]) if wo_i is not None else 0
            c_new = len(gz.wn_to_provs[wn_i]) if wn_i is not None else 0
            if gd and c_old > 0:
                w_dist_cnt = float(c_old)
            elif not gd and c_new > 0:
                w_dist_cnt = float(c_new)
            else:
                w_dist_cnt = float(max(c_old, c_new))
        f["goong_ward_dist_count"] = w_dist_cnt

        f["goong_hierarchy_valid"] = self._compute_goong_hierarchy_valid(
            gw, gd, gp, base.goong_valid_admin
        )

        all_g_toks = {
            t for s in (gw, gd, gp) if s for t in s.replace("-", " ").split() if t
        }
        f["raw_to_goong_token_ratio"] = float(len(raw_toks) / max(1, len(all_g_toks)))

        vec = np.array([f[name] for name in self.feature_names], dtype=np.float32)
        return FeatureBundle(
            vector=vec,
            features=f,
            is_legacy_2025_equiv=base.is_legacy_2025_equiv,
            merged_prov_equiv=base.merged_prov_equiv,
            spatial_dist_km=base.spatial_dist_km,
            spatial_outlier=base.spatial_outlier,
            has_prov_contradiction=base.has_prov_contradiction,
            has_dist_contradiction=base.has_dist_contradiction,
            has_chat_food_only=base.has_chat_food_only,
            goong_has_ward=base.goong_has_ward,
            goong_valid_admin=base.goong_valid_admin,
        )

    def extract_matrix(
        self,
        raw_texts: Sequence[str],
        goong_wards: Sequence[str] | None = None,
        goong_districts: Sequence[str] | None = None,
        goong_provinces: Sequence[str] | None = None,
        goong_scores: Sequence[float | None] | None = None,
        goong_lats: Sequence[float | None] | None = None,
        goong_lngs: Sequence[float | None] | None = None,
    ) -> np.ndarray:
        """Vectorized convenience builder returning float32 matrix [N, D]."""
        n = len(raw_texts)
        mat = np.empty((n, len(self.feature_names)), dtype=np.float32)
        for i in range(n):
            b = self.extract(
                raw_text=str(raw_texts[i] or ""),
                goong_ward=str(goong_wards[i] or "") if goong_wards is not None else "",
                goong_district=str(goong_districts[i] or "") if goong_districts is not None else "",
                goong_province=str(goong_provinces[i] or "") if goong_provinces is not None else "",
                goong_score=goong_scores[i] if goong_scores is not None else None,
                goong_lat=goong_lats[i] if goong_lats is not None else None,
                goong_lng=goong_lngs[i] if goong_lngs is not None else None,
            )
            mat[i] = b.vector
        return np.nan_to_num(mat, nan=0.0, posinf=1e6, neginf=-1e6)
