"""Vietnam 2025 2-tier administrative unit converter, spatial plausibility checker, and V3 parser."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .telex_fuzzy import SymSpellGazetteer, normalize_text, repair_telex_typos, strip_accents

DEFAULT_CONVERTER_PATH: Path = (
    Path(__file__).resolve().parent.parent / "shared_data_v3" / "vn_admin_v3_converter.json"
)

_EARTH_RADIUS_KM = 6371.0

# Protected hyphenated names that must NEVER be split across ' - '.
_PROTECTED_HYPHEN_RE = re.compile(
    r"(?i)\b(?:"
    r"b[aà]\s+r[iị]a\s*[-–—]\s*v[uũ]ng\s+t[aà]u"
    r"|phan\s+rang\s*[-–—]\s*th[aá]p\s+ch[aà]m"
    r"|v[aă]n\s+mi[eế]u\s*[-–—]\s*qu[oố]c\s+t[uử]\s+gi[aá]m"
    r"|ch[aâ]n\s+m[aâ]y\s*[-–—]\s*l[aă]ng\s+c[oô]"
    r"|lang\s+biang\s*[-–—]\s*[dđ][aà]\s+l[aạ]t"
    r"|l[aâ]m\s+vi[eê]n\s*[-–—]\s*[dđ][aà]\s+l[aạ]t"
    r"|xu[aâ]n\s+h[uươ][oơ]ng\s*[-–—]\s*[dđ][aà]\s+l[aạ]t"
    r"|c[aẩ]m\s+ly\s*[-–—]\s*[dđ][aà]\s+l[aạ]t"
    r"|xu[aâ]n\s+tr[uườ][oơ]ng\s*[-–—]\s*[dđ][aà]\s+l[aạ]t"
    r")\b"
)
_HYPHEN_SENTINEL = "\x00HYPH\x00"
_SEGMENT_SPLIT_RE = re.compile(r"[,;\n]+|\s+[-–—]\s+")
_POSTAL_OR_COUNTRY_RE = re.compile(r"^(?:viet\s*nam|vietnam|vn|\d{5,6})$", re.IGNORECASE)

# Prefix detectors (accented exact cues first, then normalized ASCII cues).
_WARD_ACCENTED_PREFIX_RE = re.compile(
    r"^\s*(?:(phường)|(xã)|(thị\s+trấn)|(đặc\s+khu))\s+(.+)$",
    re.IGNORECASE,
)
_WARD_NORM_PREFIX_RE = re.compile(
    r"^(?:(phuong|p|f)|(xa|x)|(thi\s+tran|tt)|(dac\s+khu|dk))\s+(.+)$"
)
_WARD_COMPACT_NUM_RE = re.compile(r"^(?:phuong|p|f)\s*0*(\d+)$")

_DIST_ACCENTED_PREFIX_RE = re.compile(
    r"^\s*(?:(quận)|(huyện)|(thị\s+xã)|(thành\s+phố))\s+(.+)$",
    re.IGNORECASE,
)
_DIST_NORM_PREFIX_RE = re.compile(
    r"^(?:(quan|q)|(huyen|h)|(thi\s+xa|tx)|(thanh\s+pho|tp))\s+(.+)$"
)
_DIST_SUFFIX_RE = re.compile(r"^(.+?)\s+(?:huyen|thi\s+xa|thanh\s+pho)$")
_DIST_COMPACT_NUM_RE = re.compile(r"^(?:quan|q)\s*0*(\d+)$")

_PROV_PREFIX_RE = re.compile(r"^(?:tinh|thanh\s+pho|tp|t)\s+(.+)$")

_STREET_RURAL_CUES: frozenset[str] = frozenset(
    {
        "so",
        "duong",
        "pho",
        "ngo",
        "ngach",
        "hem",
        "kiet",
        "lo",
        "khu",
        "kcn",
        "kdc",
        "cum",
        "to",
        "tdp",
        "thon",
        "ap",
        "xom",
        "ban",
        "buon",
        "lang",
        "khom",
        "kp",
        "cc",
        "chung",
        "toa",
        "tang",
        "ql",
        "tl",
        "cho",
        "ben",
        "cau",
        "cong",
        "cty",
        "nha",
        "dai",
    }
)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle Haversine distance in kilometers."""
    lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        math.sin(dlat * 0.5) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon * 0.5) ** 2
    )
    return 2.0 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def _has_street_or_rural_cue(text: str) -> bool:
    if not text:
        return False
    if any(ch.isdigit() for ch in text):
        return True
    norm = normalize_text(text)
    if not norm:
        return False
    tokens = norm.split()
    if any(tok in _STREET_RURAL_CUES for tok in tokens):
        return True
    return len(tokens) >= 2


def _extract_ward_candidates(raw_ward: str) -> Tuple[List[str], Optional[int], bool]:
    """Return (candidate_stems, wt_idx_hint, has_explicit_prefix)."""
    s = (raw_ward or "").strip()
    if not s:
        return [], None, False

    m_acc = _WARD_ACCENTED_PREFIX_RE.match(s)
    if m_acc:
        wt_idx = 0 if m_acc.group(1) else 1 if m_acc.group(2) else 2 if m_acc.group(3) else 3
        rest = normalize_text(m_acc.group(5))
        if rest.isdigit():
            return [f"phuong {int(rest)}" if wt_idx == 0 else rest], wt_idx, True
        return [rest], wt_idx, True

    norm = normalize_text(s)
    if not norm:
        return [], None, False

    m_num = _WARD_COMPACT_NUM_RE.match(norm)
    if m_num:
        return [f"phuong {int(m_num.group(1))}"], 0, True

    m_norm = _WARD_NORM_PREFIX_RE.match(norm)
    if m_norm:
        wt_idx = 0 if m_norm.group(1) else 1 if m_norm.group(2) else 2 if m_norm.group(3) else 3
        prefix_tok = m_norm.group(1) or m_norm.group(2) or m_norm.group(3) or m_norm.group(4)
        rest = m_norm.group(5).strip()
        if rest.isdigit():
            return [f"phuong {int(rest)}" if wt_idx == 0 else rest], wt_idx, True
        stems = [rest]
        # Protect proper names starting with 'Phuong' (e.g. 'Phuong Mai') when unaccented.
        if prefix_tok == "phuong":
            stems.append(norm)
        explicit = prefix_tok in {"p", "f", "x", "tt", "thi tran", "dk", "dac khu"}
        return stems, wt_idx, explicit

    return [norm], None, False


def _extract_district_candidates(raw_dist: str) -> Tuple[List[str], bool]:
    """Return (candidate_stems, has_explicit_prefix)."""
    s = (raw_dist or "").strip()
    if not s:
        return [], False

    m_acc = _DIST_ACCENTED_PREFIX_RE.match(s)
    if m_acc:
        rest = normalize_text(m_acc.group(5))
        if m_acc.group(1) and rest.isdigit():
            return [f"quan {int(rest)}"], True
        return [rest], True

    norm = normalize_text(s)
    if not norm:
        return [], False

    m_num = _DIST_COMPACT_NUM_RE.match(norm)
    if m_num:
        return [f"quan {int(m_num.group(1))}"], True

    m_norm = _DIST_NORM_PREFIX_RE.match(norm)
    if m_norm:
        prefix_tok = m_norm.group(1) or m_norm.group(2) or m_norm.group(3) or m_norm.group(4)
        rest = m_norm.group(5).strip()
        if prefix_tok in {"quan", "q"} and rest.isdigit():
            return [f"quan {int(rest)}"], True
        stems = [rest]
        if prefix_tok == "quan":
            stems.append(norm)
        return stems, True

    m_suf = _DIST_SUFFIX_RE.match(norm)
    if m_suf:
        return [m_suf.group(1).strip(), norm], True

    return [norm], False


def _extract_province_candidates(raw_prov: str) -> List[str]:
    norm = normalize_text(raw_prov)
    if not norm:
        return []
    m = _PROV_PREFIX_RE.match(norm)
    if m:
        return [m.group(1).strip(), norm]
    return [norm]


def _protect_and_split_segments(text: str) -> List[str]:
    """Split address into segments by commas/dashes while protecting hyphenated admin names."""
    if not text:
        return []

    def _mask(m: re.Match[str]) -> str:
        return re.sub(r"\s*[-–—]\s*", _HYPHEN_SENTINEL, m.group(0))

    masked = _PROTECTED_HYPHEN_RE.sub(_mask, text)
    parts = [p.replace(_HYPHEN_SENTINEL, " - ").strip() for p in _SEGMENT_SPLIT_RE.split(masked)]
    parts = [p for p in parts if p]
    while parts and _POSTAL_OR_COUNTRY_RE.match(normalize_text(parts[-1])):
        parts.pop()
    return parts


class VietnamAdminV3Index:
    """In-memory index for Vietnam legacy (63-province) <-> 2025 (34-province 2-tier) conversion."""

    def __init__(self, converter_path: Path = DEFAULT_CONVERTER_PATH) -> None:
        self.converter_path = Path(converter_path)
        with self.converter_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        self.wt: List[str] = data["wt"]
        self.po: List[str] = data["po"]
        self.po_disp: List[str] = data["po_disp"]
        self.do: List[str] = data["do"]
        self.do_disp: List[str] = data["do_disp"]
        self.wo: List[str] = data["wo"]
        self.wo_disp: List[str] = data["wo_disp"]
        self.pn: List[str] = data["pn"]
        self.pn_disp: List[str] = data["pn_disp"]
        self.pn_code: List[int] = data["pn_code"]
        self.wn: List[str] = data["wn"]
        self.wn_disp: List[str] = data["wn_disp"]
        self.wn_code_disp: Dict[str, str] = data.get("wn_code_disp", {})
        self.pm: List[int] = data["pm"]
        self.dp: List[List[int]] = data["dp"]
        self.eo: List[List[List[int]]] = data["eo"]
        self.en: List[List[List[Any]]] = data["en"]
        self.isle: List[List[int]] = data["isle"]
        self.rev: Dict[str, List[List[int]]] = data["rev"]
        self.ogeo: Dict[str, List[float]] = data["ogeo"]
        self.pa: Dict[str, str] = data["pa"]
        self.da: Dict[str, str] = data["da"]

        # Exact normalized lookup maps
        self.po_by_name: Dict[str, int] = {name: idx for idx, name in enumerate(self.po)}
        self.pn_by_name: Dict[str, int] = {name: idx for idx, name in enumerate(self.pn)}
        self.do_by_name: Dict[str, int] = {name: idx for idx, name in enumerate(self.do)}
        self.wo_by_name: Dict[str, int] = {name: idx for idx, name in enumerate(self.wo)}
        self.wn_by_name: Dict[str, int] = {name: idx for idx, name in enumerate(self.wn)}

        # Add hyphen-relaxed aliases for hyphenated admin names.
        for name, idx in list(self.po_by_name.items()):
            if " - " in name:
                self.po_by_name[name.replace(" - ", " ")] = idx
                self.po_by_name[name.replace(" - ", "-")] = idx
        for name, idx in list(self.do_by_name.items()):
            if " - " in name:
                self.do_by_name[name.replace(" - ", " ")] = idx
                self.do_by_name[name.replace(" - ", "-")] = idx
        for name, idx in list(self.wn_by_name.items()):
            if " - " in name:
                self.wn_by_name[name.replace(" - ", " ")] = idx
                self.wn_by_name[name.replace(" - ", "-")] = idx

        # Aliases from pa/da
        for alias, target in self.pa.items():
            pid_o = self.po_by_name.get(target)
            if pid_o is not None and alias not in self.po_by_name:
                self.po_by_name[alias] = pid_o
            pid_n = self.pn_by_name.get(target)
            if pid_n is not None and alias not in self.pn_by_name:
                self.pn_by_name[alias] = pid_n

        for alias, target in self.da.items():
            did_o = self.do_by_name.get(target)
            if did_o is None and target in {"quan 2", "quan 9"}:
                did_o = self.do_by_name.get("thu duc")
            if did_o is not None and alias not in self.do_by_name:
                self.do_by_name[alias] = did_o
        if "quan 2" not in self.do_by_name and "thu duc" in self.do_by_name:
            self.do_by_name["quan 2"] = self.do_by_name["thu duc"]
        if "quan 9" not in self.do_by_name and "thu duc" in self.do_by_name:
            self.do_by_name["quan 9"] = self.do_by_name["thu duc"]

        # Reverse pm: pid_n -> set of pid_o
        self.pn_to_po: Dict[int, Set[int]] = defaultdict(set)
        for pid_o, pid_n in enumerate(self.pm):
            self.pn_to_po[pid_n].add(pid_o)

        # Island special zones
        self.isle_by_did: Dict[int, Tuple[int, int, int, int]] = {}
        self.isle_by_wn: Dict[int, Tuple[int, int, int, int]] = {}
        self.isle_by_name: Dict[str, Tuple[int, int, int, int]] = {}
        for did_o, pid_o, wid_n, pid_n in self.isle:
            tup = (did_o, pid_o, wid_n, pid_n)
            self.isle_by_did[did_o] = tup
            self.isle_by_wn[wid_n] = tup
            self.isle_by_name[self.do[did_o]] = tup
            self.isle_by_name[self.wn[wid_n]] = tup

        # District -> 2025 wards & Province -> 2025 wards for disambiguation/suggestions
        self.district_to_2025_wards: Dict[int, Dict[Tuple[int, int], int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for wid_o, rows in enumerate(self.eo):
            for did_o, _pid_o, wid_n, pid_n, _is_div, _is_def in rows:
                self.district_to_2025_wards[did_o][(wid_n, pid_n)] += 1

        self.province_to_2025_wards: Dict[int, List[int]] = defaultdict(list)
        for wid_n, rows in enumerate(self.en):
            seen_p: Set[int] = set()
            for row in rows:
                pid_n = int(row[0])
                if pid_n not in seen_p:
                    seen_p.add(pid_n)
                    self.province_to_2025_wards[pid_n].append(wid_n)

        self.gazetteer: SymSpellGazetteer = SymSpellGazetteer.from_converter_data(data)

    # ------------------------------------------------------------------
    # Internal resolution helpers
    # ------------------------------------------------------------------
    def _resolve_province(
        self, raw_prov: str, allow_fuzzy: bool = True
    ) -> Tuple[Optional[int], Optional[int], Set[int]]:
        """Return (pid_o_exact, pid_n_exact, matching_pid_o_set)."""
        cands = _extract_province_candidates(raw_prov)
        for stem in cands:
            pid_o = self.po_by_name.get(stem)
            pid_n = self.pn_by_name.get(stem)
            if pid_o is not None:
                mapped_pn = self.pm[pid_o]
                # If name exists in both po and pn, pid_n == mapped_pn unless merged
                return pid_o, mapped_pn, {pid_o}
            if pid_n is not None:
                return None, pid_n, set(self.pn_to_po[pid_n])

        if allow_fuzzy:
            for stem in cands:
                if len(stem) >= 4:
                    fuzzy = self.gazetteer.lookup(stem, level="province")
                    if fuzzy:
                        pid_o = self.po_by_name.get(fuzzy)
                        pid_n = self.pn_by_name.get(fuzzy)
                        if pid_o is not None:
                            return pid_o, self.pm[pid_o], {pid_o}
                        if pid_n is not None:
                            return None, pid_n, set(self.pn_to_po[pid_n])

        return None, None, set()

    def _resolve_district(
        self,
        raw_dist: str,
        pid_o_filter: Optional[Set[int]] = None,
        allow_fuzzy: bool = True,
    ) -> Optional[int]:
        cands, _ = _extract_district_candidates(raw_dist)
        for stem in cands:
            did_o = self.do_by_name.get(stem)
            if did_o is not None:
                if not pid_o_filter or (set(self.dp[did_o]) & pid_o_filter):
                    return did_o

        if allow_fuzzy:
            ctx: Optional[List[str]] = None
            if pid_o_filter:
                ctx = [self.po[p] for p in pid_o_filter]
            for stem in cands:
                if len(stem) >= 4:
                    fuzzy = self.gazetteer.lookup(stem, level="district", parent_context=ctx)
                    if fuzzy:
                        did_o = self.do_by_name.get(fuzzy)
                        if did_o is not None and (
                            not pid_o_filter or (set(self.dp[did_o]) & pid_o_filter)
                        ):
                            return did_o
        return None

    def _resolve_wo(
        self,
        raw_ward: str,
        did_o: Optional[int] = None,
        pid_o_filter: Optional[Set[int]] = None,
        allow_fuzzy: bool = True,
    ) -> Optional[int]:
        cands, _, _ = _extract_ward_candidates(raw_ward)
        for stem in cands:
            wid_o = self.wo_by_name.get(stem)
            if wid_o is not None and self._wo_matches_filter(wid_o, did_o, pid_o_filter):
                return wid_o

        if allow_fuzzy:
            ctx: List[str] = []
            if did_o is not None:
                ctx.append(self.do[did_o])
            if pid_o_filter:
                ctx.extend(self.po[p] for p in pid_o_filter)
                ctx.extend(self.pn[self.pm[p]] for p in pid_o_filter)
            for stem in cands:
                if len(stem) >= 4:
                    fuzzy = self.gazetteer.lookup(
                        stem, level="wo", parent_context=ctx if ctx else None
                    )
                    if fuzzy:
                        wid_o = self.wo_by_name.get(fuzzy)
                        if wid_o is not None and self._wo_matches_filter(wid_o, did_o, pid_o_filter):
                            return wid_o
        return None

    def _wo_matches_filter(
        self, wid_o: int, did_o: Optional[int], pid_o_filter: Optional[Set[int]]
    ) -> bool:
        for r_did, r_pid, _, _, _, _ in self.eo[wid_o]:
            if did_o is not None and r_did != did_o:
                continue
            if pid_o_filter and r_pid not in pid_o_filter:
                continue
            return True
        return False

    def _resolve_wn(
        self,
        raw_ward: str,
        pid_n: Optional[int] = None,
        allow_fuzzy: bool = True,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Return (wid_n, wt_idx_hint)."""
        cands, wt_hint, _ = _extract_ward_candidates(raw_ward)
        for stem in cands:
            wid_n = self.wn_by_name.get(stem)
            if wid_n is not None and self._wn_matches_province(wid_n, pid_n, wt_hint=wt_hint):
                return wid_n, wt_hint

        if allow_fuzzy:
            ctx = [self.pn[pid_n]] if pid_n is not None else None
            for stem in cands:
                if len(stem) >= 4:
                    fuzzy = self.gazetteer.lookup(stem, level="wn", parent_context=ctx)
                    if fuzzy:
                        wid_n = self.wn_by_name.get(fuzzy)
                        if wid_n is not None and self._wn_matches_province(wid_n, pid_n, wt_hint=wt_hint):
                            return wid_n, wt_hint
        return None, wt_hint

    def _wn_matches_province(
        self, wid_n: int, pid_n: Optional[int], wt_hint: Optional[int] = None
    ) -> bool:
        if pid_n is None:
            if len(self.en[wid_n]) == 1:
                return True
            if wt_hint is not None:
                return sum(1 for r in self.en[wid_n] if int(r[5]) == wt_hint) == 1
            return False
        return any(int(r[0]) == pid_n for r in self.en[wid_n])

    def _pick_default_pn(self, wid_n: int, wt_hint: Optional[int] = None) -> int:
        if wt_hint is not None:
            typed = [r for r in self.en[wid_n] if int(r[5]) == wt_hint]
            if typed:
                return int(typed[0][0])
        return int(self.en[wid_n][0][0])

    def _format_ward_from_row(self, wid_n: int, row: List[Any]) -> str:
        wt_label = self.wt[int(row[5])]
        disp = self.wn_code_disp.get(str(int(row[4])), self.wn_disp[wid_n])
        return f"{wt_label} {disp}" if wt_label else disp

    def _pick_en_row(
        self,
        wid_n: int,
        pid_n: int,
        wt_hint: Optional[int] = None,
        ref_coords: Optional[Tuple[float, float]] = None,
    ) -> Optional[List[Any]]:
        rows = [r for r in self.en[wid_n] if int(r[0]) == pid_n]
        if not rows:
            return None
        if len(rows) == 1:
            return rows[0]
        if wt_hint is not None:
            typed_rows = [r for r in rows if int(r[5]) == wt_hint]
            if len(typed_rows) == 1:
                return typed_rows[0]
            if typed_rows:
                rows = typed_rows
        if ref_coords is not None:
            rlat, rlon = ref_coords
            return min(rows, key=lambda r: haversine_km(rlat, rlon, float(r[1]), float(r[2])))
        return rows[0]

    def _format_candidate(
        self,
        wid_n: int,
        pid_n: int,
        is_default: bool = True,
        wt_hint: Optional[int] = None,
        ref_coords: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        row = self._pick_en_row(wid_n, pid_n, wt_hint=wt_hint, ref_coords=ref_coords)
        if row is None:
            return {
                "new_ward": self.wn_disp[wid_n],
                "new_province": self.pn_disp[pid_n],
                "new_ward_code": None,
                "new_province_code": self.pn_code[pid_n],
                "lat": None,
                "lon": None,
                "area_km2": None,
                "is_default": is_default,
            }
        return {
            "new_ward": self._format_ward_from_row(wid_n, row),
            "new_province": self.pn_disp[pid_n],
            "new_ward_code": int(row[4]),
            "new_province_code": self.pn_code[pid_n],
            "lat": float(row[1]),
            "lon": float(row[2]),
            "area_km2": float(row[3]),
            "is_default": is_default,
        }

    def _build_conversion_result(
        self, status: str, is_divided: bool, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not candidates:
            return {
                "status": status,
                "is_divided": False,
                "new_ward": None,
                "new_province": None,
                "new_ward_code": None,
                "new_province_code": None,
                "lat": None,
                "lon": None,
                "area_km2": None,
                "candidates": [],
            }
        top = candidates[0]
        return {
            "status": status,
            "is_divided": is_divided,
            "new_ward": top["new_ward"],
            "new_province": top["new_province"],
            "new_ward_code": top["new_ward_code"],
            "new_province_code": top["new_province_code"],
            "lat": top["lat"],
            "lon": top["lon"],
            "area_km2": top["area_km2"],
            "candidates": candidates,
        }

    def _check_isle_match(
        self,
        ward: str,
        district: str,
        pid_o_set: Set[int],
        pid_n: Optional[int],
    ) -> Optional[Tuple[int, int, int, int]]:
        for raw in (ward, district):
            if not raw:
                continue
            for stem in _extract_district_candidates(raw)[0] + _extract_ward_candidates(raw)[0]:
                tup = self.isle_by_name.get(stem)
                if tup is None:
                    continue
                _did_o, i_pid_o, _wid_n, i_pid_n = tup
                if pid_o_set and i_pid_o not in pid_o_set and i_pid_n != pid_n:
                    continue
                if pid_n is not None and i_pid_n != pid_n and i_pid_o not in pid_o_set:
                    continue
                return tup
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def convert_legacy_to_2025(
        self, ward: str, district: str = "", province: str = ""
    ) -> Dict[str, Any]:
        """Convert legacy (ward, district, province) or 2025 input to 2025 2-tier format."""
        w_rep = repair_telex_typos(ward) if ward else ""
        d_rep = repair_telex_typos(district) if district else ""
        p_rep = repair_telex_typos(province) if province else ""

        _pid_o, pid_n, pid_o_set = self._resolve_province(p_rep) if p_rep else (None, None, set())

        # 0. Explicit 2025 'Đặc khu ...' when district is omitted -> already_2025
        if not d_rep and w_rep:
            _, wt_hint_dk, exp_dk = _extract_ward_candidates(w_rep)
            if exp_dk and wt_hint_dk == 3:
                wid_n, wt_hint = self._resolve_wn(w_rep, pid_n=pid_n)
                if wid_n is not None:
                    target_pn = pid_n if pid_n is not None else self._pick_default_pn(wid_n, wt_hint=wt_hint)
                    cand = self._format_candidate(wid_n, target_pn, is_default=True, wt_hint=wt_hint)
                    return self._build_conversion_result("already_2025", False, [cand])

        # 1. Explicit legacy (ward, district) sub-ward match (e.g. 'Xã Thổ Châu, TP Phú Quốc')
        #    must run before district-only island special zone fallback.
        did_o = self._resolve_district(d_rep, pid_o_filter=pid_o_set or None) if d_rep else None
        if not pid_o_set and did_o is not None and len(self.dp[did_o]) == 1:
            pid_o_set = {self.dp[did_o][0]}
            pid_n = self.pm[self.dp[did_o][0]]

        if d_rep and w_rep and did_o is not None:
            wid_o_sub = self._resolve_wo(w_rep, did_o=did_o, pid_o_filter=pid_o_set or None)
            if wid_o_sub is not None:
                sub_rows = [
                    r
                    for r in self.eo[wid_o_sub]
                    if r[0] == did_o and (not pid_o_set or r[1] in pid_o_set)
                ]
                if sub_rows:
                    rows_sorted = sorted(sub_rows, key=lambda r: (-r[5], r[2]))
                    is_divided = any(r[4] == 1 for r in rows_sorted)
                    cands_sub: List[Dict[str, Any]] = []
                    for r_did, r_pid, r_wn, r_pn, _is_div, is_def in rows_sorted:
                        ogeo_pt = self.ogeo.get(f"{wid_o_sub}:{r_did}:{r_pid}")
                        ref = (float(ogeo_pt[0]), float(ogeo_pt[1])) if ogeo_pt else None
                        cands_sub.append(
                            self._format_candidate(
                                r_wn,
                                r_pn,
                                is_default=(bool(is_def) if is_divided else True),
                                ref_coords=ref,
                            )
                        )
                    return self._build_conversion_result("converted", is_divided, cands_sub)

        # 2. Check island special zones (Huyen Con Dao, Hoang Sa, Ly Son, Bach Long Vi, Con Co, Phu Quoc, etc.)
        isle_hit = self._check_isle_match(w_rep, d_rep, pid_o_set, pid_n)
        if isle_hit is not None:
            _, _, i_wid_n, i_pid_n = isle_hit
            cand = self._format_candidate(i_wid_n, i_pid_n, is_default=True, wt_hint=3)
            return self._build_conversion_result("converted", False, [cand])

        # 3. If district is omitted, check 2025 (wn, pn) first to avoid misconverting new 2025 wards
        #    whose names overlap old rural wards (e.g. 'Phuong Hong Ha, Ha Noi').
        if not d_rep and w_rep:
            wid_n, wt_hint = self._resolve_wn(w_rep, pid_n=pid_n)
            if wid_n is not None:
                target_pn = pid_n if pid_n is not None else self._pick_default_pn(wid_n, wt_hint=wt_hint)
                cand = self._format_candidate(wid_n, target_pn, is_default=True, wt_hint=wt_hint)
                return self._build_conversion_result("already_2025", False, [cand])

        # 4. Legacy (wo, do, po) resolution
        if w_rep:
            wid_o = self._resolve_wo(w_rep, did_o=did_o, pid_o_filter=pid_o_set or None)
            if wid_o is not None:
                rows = [
                    r
                    for r in self.eo[wid_o]
                    if (did_o is None or r[0] == did_o)
                    and (not pid_o_set or r[1] in pid_o_set)
                ]
                if rows:
                    # If neither district nor province constrained rows, verify unambiguous target.
                    distinct_src = {(r[0], r[1]) for r in rows}
                    if len(distinct_src) == 1 or did_o is not None or pid_o_set:
                        # Group by (r_did_o, r_pid_o) if multiple districts share name in same province
                        if len(distinct_src) > 1 and did_o is None:
                            # Ambiguous across multiple districts in province
                            cands_amb = []
                            seen_targets: Set[Tuple[int, int]] = set()
                            for r_did, r_pid, r_wn, r_pn, _is_div, is_def in rows:
                                if (r_wn, r_pn) in seen_targets:
                                    continue
                                seen_targets.add((r_wn, r_pn))
                                ogeo_pt = self.ogeo.get(f"{wid_o}:{r_did}:{r_pid}")
                                ref = (float(ogeo_pt[0]), float(ogeo_pt[1])) if ogeo_pt else None
                                cands_amb.append(
                                    self._format_candidate(
                                        r_wn, r_pn, is_default=bool(is_def), ref_coords=ref
                                    )
                                )
                            return self._build_conversion_result("ambiguous", False, cands_amb)

                        rows_sorted = sorted(rows, key=lambda r: (-r[5], r[2]))
                        is_divided = any(r[4] == 1 for r in rows_sorted)
                        cands: List[Dict[str, Any]] = []
                        for r_did, r_pid, r_wn, r_pn, _is_div, is_def in rows_sorted:
                            ogeo_pt = self.ogeo.get(f"{wid_o}:{r_did}:{r_pid}")
                            ref = (float(ogeo_pt[0]), float(ogeo_pt[1])) if ogeo_pt else None
                            cands.append(
                                self._format_candidate(
                                    r_wn,
                                    r_pn,
                                    is_default=(bool(is_def) if is_divided else True),
                                    ref_coords=ref,
                                )
                            )
                        return self._build_conversion_result("converted", is_divided, cands)

        # 4. Fallback to 2025 (wn, pn) if district was non-empty (e.g. mixed 2025 ward + legacy district)
        if w_rep:
            wid_n, wt_hint = self._resolve_wn(w_rep, pid_n=pid_n)
            if wid_n is not None:
                target_pn = pid_n if pid_n is not None else self._pick_default_pn(wid_n, wt_hint=wt_hint)
                cand = self._format_candidate(wid_n, target_pn, is_default=True, wt_hint=wt_hint)
                return self._build_conversion_result("already_2025", False, [cand])

        return self._build_conversion_result("not_found", False, [])

    def _resolve_equivalence_sets(
        self, ward: str, district: str, province: str
    ) -> Tuple[Set[Tuple[Any, ...]], Set[Tuple[int, int]], Set[Tuple[int, int]], Optional[int]]:
        """Return (legacy_keys, legacy_to_2025, direct_2025, resolved_pid_n)."""
        w_rep = repair_telex_typos(ward) if ward else ""
        d_rep = repair_telex_typos(district) if district else ""
        p_rep = repair_telex_typos(province) if province else ""

        _pid_o, pid_n, pid_o_set = self._resolve_province(p_rep) if p_rep else (None, None, set())
        did_o = self._resolve_district(d_rep, pid_o_filter=pid_o_set or None) if d_rep else None
        if not pid_o_set and did_o is not None and len(self.dp[did_o]) == 1:
            pid_o_set = {self.dp[did_o][0]}
            pid_n = self.pm[self.dp[did_o][0]]

        legacy_keys: Set[Tuple[Any, ...]] = set()
        legacy_to_2025: Set[Tuple[int, int]] = set()
        direct_2025: Set[Tuple[int, int]] = set()

        matched_subward_in_eo = False
        if d_rep and w_rep and did_o is not None:
            wid_o = self._resolve_wo(w_rep, did_o=did_o, pid_o_filter=pid_o_set or None)
            if wid_o is not None:
                for r_did, r_pid, r_wn, r_pn, _is_div, _is_def in self.eo[wid_o]:
                    if r_did != did_o:
                        continue
                    if pid_o_set and r_pid not in pid_o_set:
                        continue
                    legacy_keys.add((wid_o, r_did, r_pid))
                    legacy_to_2025.add((r_wn, r_pn))
                    matched_subward_in_eo = True

        if not matched_subward_in_eo:
            is_explicit_dk = False
            if not d_rep and w_rep:
                _, wt_hint_dk, exp_dk = _extract_ward_candidates(w_rep)
                is_explicit_dk = bool(exp_dk and wt_hint_dk == 3)
            if not is_explicit_dk:
                isle_hit = self._check_isle_match(w_rep, d_rep, pid_o_set, pid_n)
                if isle_hit is not None:
                    i_did_o, i_pid_o, i_wid_n, i_pid_n = isle_hit
                    legacy_keys.add(("isle", i_did_o, i_pid_o))
                    legacy_to_2025.add((i_wid_n, i_pid_n))
                    direct_2025.add((i_wid_n, i_pid_n))
                    return legacy_keys, legacy_to_2025, direct_2025, i_pid_n

        if w_rep:
            if not matched_subward_in_eo:
                wid_o = self._resolve_wo(w_rep, did_o=did_o, pid_o_filter=pid_o_set or None)
                if wid_o is not None:
                    for r_did, r_pid, r_wn, r_pn, _is_div, _is_def in self.eo[wid_o]:
                        if did_o is not None and r_did != did_o:
                            continue
                        if pid_o_set and r_pid not in pid_o_set:
                            continue
                        legacy_keys.add((wid_o, r_did, r_pid))
                        legacy_to_2025.add((r_wn, r_pn))

            wid_n, wt_hint = self._resolve_wn(w_rep, pid_n=pid_n)
            if wid_n is not None:
                for row in self.en[wid_n]:
                    r_pn = int(row[0])
                    if pid_n is not None and r_pn != pid_n:
                        continue
                    if wt_hint is not None and any(
                        int(r[0]) == r_pn and int(r[5]) == wt_hint for r in self.en[wid_n]
                    ) and int(row[5]) != wt_hint:
                        continue
                    direct_2025.add((wid_n, r_pn))

        return legacy_keys, legacy_to_2025, direct_2025, pid_n

    def is_legacy_2025_equivalent(
        self,
        raw_w: str,
        raw_d: str,
        raw_p: str,
        target_w: str,
        target_d: str = "",
        target_p: str = "",
    ) -> bool:
        """Check bidirectional equivalence between legacy 3-tier and 2025 2-tier admin tuples."""
        raw_leg, raw_l2n, raw_2025, raw_pn = self._resolve_equivalence_sets(raw_w, raw_d, raw_p)
        tgt_leg, tgt_l2n, tgt_2025, tgt_pn = self._resolve_equivalence_sets(
            target_w, target_d, target_p
        )

        # Direct identical legacy unit
        if raw_leg and tgt_leg and (raw_leg & tgt_leg):
            return True

        raw_has_explicit_leg_dist = bool(raw_d and raw_leg and not (raw_leg & tgt_leg))
        tgt_has_explicit_leg_dist = bool(target_d and tgt_leg and not (raw_leg & tgt_leg))

        # Legacy raw <-> 2025 target
        if raw_leg and tgt_2025 and not tgt_has_explicit_leg_dist:
            if raw_l2n & tgt_2025:
                return True

        # 2025 raw <-> Legacy target
        if tgt_leg and raw_2025 and not raw_has_explicit_leg_dist:
            if tgt_l2n & raw_2025:
                return True

        # Both resolved as 2025 units (neither forced into distinct legacy district)
        if raw_2025 and tgt_2025 and not raw_has_explicit_leg_dist and not tgt_has_explicit_leg_dist:
            if raw_2025 & tgt_2025:
                return True

        # Province-only check when both ward & district are empty
        if not (raw_w or raw_d or target_w or target_d):
            if raw_pn is not None and raw_pn == tgt_pn:
                return True

        return False

    def spatial_plausibility(
        self, ward: str, district: str, province: str, lat: float, lon: float
    ) -> Dict[str, Any]:
        """Validate coordinate plausibility against ward centroid & dynamic area threshold."""
        if lat is None or lon is None or not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return {
                "status": "invalid_coordinates",
                "dist_km": None,
                "max_expected_km": None,
                "is_outlier": None,
                "centroid_lat": None,
                "centroid_lon": None,
                "area_km2": None,
                "matched_ward": None,
                "matched_province": None,
            }

        w_rep = repair_telex_typos(ward) if ward else ""
        d_rep = repair_telex_typos(district) if district else ""
        p_rep = repair_telex_typos(province) if province else ""

        _pid_o, pid_n, pid_o_set = self._resolve_province(p_rep) if p_rep else (None, None, set())
        did_o = self._resolve_district(d_rep, pid_o_filter=pid_o_set or None) if d_rep else None
        if not pid_o_set and did_o is not None and len(self.dp[did_o]) == 1:
            pid_o_set = {self.dp[did_o][0]}
            pid_n = self.pm[self.dp[did_o][0]]

        # Check explicit 2025 'Đặc khu ...' when district omitted
        if not d_rep and w_rep:
            _, wt_hint_dk, exp_dk = _extract_ward_candidates(w_rep)
            if exp_dk and wt_hint_dk == 3:
                wid_n, wt_hint = self._resolve_wn(w_rep, pid_n=pid_n)
                if wid_n is not None:
                    target_pn = pid_n if pid_n is not None else self._pick_default_pn(wid_n, wt_hint=wt_hint)
                    row = self._pick_en_row(wid_n, target_pn, wt_hint=wt_hint, ref_coords=(lat, lon))
                    if row is not None:
                        clat, clon, area_km2 = float(row[1]), float(row[2]), float(row[3])
                        dist_km = haversine_km(lat, lon, clat, clon)
                        max_km = max(15.0, min(60.0, 2.5 * math.sqrt(area_km2) + 8.0))
                        return {
                            "status": "ok",
                            "dist_km": round(dist_km, 3),
                            "max_expected_km": round(max_km, 3),
                            "is_outlier": dist_km > max_km,
                            "centroid_lat": clat,
                            "centroid_lon": clon,
                            "area_km2": area_km2,
                            "matched_ward": self._format_ward_from_row(wid_n, row),
                            "matched_province": self.pn_disp[target_pn],
                        }

        # Check explicit legacy (ward, district) sub-ward before island district fallback
        if d_rep and w_rep and did_o is not None:
            wid_o_sub = self._resolve_wo(w_rep, did_o=did_o, pid_o_filter=pid_o_set or None)
            if wid_o_sub is not None:
                sub_rows = [
                    r
                    for r in self.eo[wid_o_sub]
                    if r[0] == did_o and (not pid_o_set or r[1] in pid_o_set)
                ]
                if sub_rows:
                    rows_sorted = sorted(sub_rows, key=lambda r: (-r[5], r[2]))
                    r_did, r_pid, r_wn, r_pn, _is_div, _is_def = rows_sorted[0]
                    ogeo_pt = self.ogeo.get(f"{wid_o_sub}:{r_did}:{r_pid}")
                    ref = (float(ogeo_pt[0]), float(ogeo_pt[1])) if ogeo_pt else (lat, lon)
                    en_row = self._pick_en_row(r_wn, r_pn, ref_coords=ref)
                    area_km2 = float(en_row[3]) if en_row is not None else 25.0
                    clat = float(ogeo_pt[0]) if ogeo_pt is not None else (float(en_row[1]) if en_row is not None else None)
                    clon = float(ogeo_pt[1]) if ogeo_pt is not None else (float(en_row[2]) if en_row is not None else None)
                    if clat is not None and clon is not None:
                        dist_km = haversine_km(lat, lon, clat, clon)
                        max_km = max(15.0, min(60.0, 2.5 * math.sqrt(area_km2) + 8.0))
                        matched_w = (
                            self._format_ward_from_row(r_wn, en_row)
                            if en_row is not None
                            else self.wn_disp[r_wn]
                        )
                        return {
                            "status": "ok",
                            "dist_km": round(dist_km, 3),
                            "max_expected_km": round(max_km, 3),
                            "is_outlier": dist_km > max_km,
                            "centroid_lat": clat,
                            "centroid_lon": clon,
                            "area_km2": area_km2,
                            "matched_ward": matched_w,
                            "matched_province": self.pn_disp[r_pn],
                        }

        # Check island fallback
        isle_hit = self._check_isle_match(w_rep, d_rep, pid_o_set, pid_n)
        if isle_hit is not None:
            _, _, i_wid_n, i_pid_n = isle_hit
            row = self._pick_en_row(i_wid_n, i_pid_n, wt_hint=3)
            if row is not None:
                clat, clon, area_km2 = float(row[1]), float(row[2]), float(row[3])
                dist_km = haversine_km(lat, lon, clat, clon)
                max_km = max(15.0, min(60.0, 2.5 * math.sqrt(area_km2) + 8.0))
                return {
                    "status": "ok",
                    "dist_km": round(dist_km, 3),
                    "max_expected_km": round(max_km, 3),
                    "is_outlier": dist_km > max_km,
                    "centroid_lat": clat,
                    "centroid_lon": clon,
                    "area_km2": area_km2,
                    "matched_ward": self._format_ward_from_row(i_wid_n, row),
                    "matched_province": self.pn_disp[i_pid_n],
                }

        # Check 2025 direct if no district
        if not d_rep and w_rep:
            wid_n, wt_hint = self._resolve_wn(w_rep, pid_n=pid_n)
            if wid_n is not None:
                target_pn = pid_n if pid_n is not None else self._pick_default_pn(wid_n, wt_hint=wt_hint)
                # If multiple en rows exist in the same province, pick closest to query (lat, lon)
                row = self._pick_en_row(
                    wid_n, target_pn, wt_hint=wt_hint, ref_coords=(lat, lon)
                )
                if row is not None:
                    clat, clon, area_km2 = float(row[1]), float(row[2]), float(row[3])
                    dist_km = haversine_km(lat, lon, clat, clon)
                    max_km = max(15.0, min(60.0, 2.5 * math.sqrt(area_km2) + 8.0))
                    return {
                        "status": "ok",
                        "dist_km": round(dist_km, 3),
                        "max_expected_km": round(max_km, 3),
                        "is_outlier": dist_km > max_km,
                        "centroid_lat": clat,
                        "centroid_lon": clon,
                        "area_km2": area_km2,
                        "matched_ward": self._format_ward_from_row(wid_n, row),
                        "matched_province": self.pn_disp[target_pn],
                    }

        # Check legacy (wo, do, po) using ogeo when present and en for area_km2
        if w_rep:
            wid_o = self._resolve_wo(w_rep, did_o=did_o, pid_o_filter=pid_o_set or None)
            if wid_o is not None:
                rows = [
                    r
                    for r in self.eo[wid_o]
                    if (did_o is None or r[0] == did_o)
                    and (not pid_o_set or r[1] in pid_o_set)
                ]
                if rows:
                    rows_sorted = sorted(rows, key=lambda r: (-r[5], r[2]))
                    r_did, r_pid, r_wn, r_pn, _is_div, _is_def = rows_sorted[0]
                    ogeo_pt = self.ogeo.get(f"{wid_o}:{r_did}:{r_pid}")
                    ref = (float(ogeo_pt[0]), float(ogeo_pt[1])) if ogeo_pt else (lat, lon)
                    en_row = self._pick_en_row(r_wn, r_pn, ref_coords=ref)
                    area_km2 = float(en_row[3]) if en_row is not None else 25.0
                    if ogeo_pt is not None:
                        clat, clon = float(ogeo_pt[0]), float(ogeo_pt[1])
                    elif en_row is not None:
                        clat, clon = float(en_row[1]), float(en_row[2])
                    else:
                        clat, clon = None, None  # type: ignore[assignment]
                    if clat is not None and clon is not None:
                        dist_km = haversine_km(lat, lon, clat, clon)
                        max_km = max(15.0, min(60.0, 2.5 * math.sqrt(area_km2) + 8.0))
                        matched_w = (
                            self._format_ward_from_row(r_wn, en_row)
                            if en_row is not None
                            else self.wn_disp[r_wn]
                        )
                        return {
                            "status": "ok",
                            "dist_km": round(dist_km, 3),
                            "max_expected_km": round(max_km, 3),
                            "is_outlier": dist_km > max_km,
                            "centroid_lat": clat,
                            "centroid_lon": clon,
                            "area_km2": area_km2,
                            "matched_ward": matched_w,
                            "matched_province": self.pn_disp[r_pn],
                        }

        # Fallback: 2025 ward even if district passed
        if w_rep:
            wid_n, wt_hint = self._resolve_wn(w_rep, pid_n=pid_n)
            if wid_n is not None:
                target_pn = pid_n if pid_n is not None else self._pick_default_pn(wid_n, wt_hint=wt_hint)
                row = self._pick_en_row(wid_n, target_pn, wt_hint=wt_hint, ref_coords=(lat, lon))
                if row is not None:
                    clat, clon, area_km2 = float(row[1]), float(row[2]), float(row[3])
                    dist_km = haversine_km(lat, lon, clat, clon)
                    max_km = max(15.0, min(60.0, 2.5 * math.sqrt(area_km2) + 8.0))
                    return {
                        "status": "ok",
                        "dist_km": round(dist_km, 3),
                        "max_expected_km": round(max_km, 3),
                        "is_outlier": dist_km > max_km,
                        "centroid_lat": clat,
                        "centroid_lon": clon,
                        "area_km2": area_km2,
                        "matched_ward": self._format_ward_from_row(wid_n, row),
                        "matched_province": self.pn_disp[target_pn],
                    }

        return {
            "status": "not_found",
            "dist_km": None,
            "max_expected_km": None,
            "is_outlier": None,
            "centroid_lat": None,
            "centroid_lon": None,
            "area_km2": None,
            "matched_ward": None,
            "matched_province": None,
        }

    # ------------------------------------------------------------------
    # Span-disjoint extraction + smart 2025 disambiguation & suggestions
    # ------------------------------------------------------------------
    def _extract_suffix_province(self, seg: str) -> Tuple[str, Optional[str], Optional[int], Set[int]]:
        """Try matching whole segment or right-hand token suffix as Province."""
        pid_o, pid_n, pid_o_set = self._resolve_province(seg, allow_fuzzy=False)
        if pid_n is not None:
            return "", seg, pid_n, pid_o_set

        words = seg.split()
        # Try exact suffix n-grams (for unpunctuated 'Ba Dinh Ha Noi' or 'Ba Ria - Vung Tau')
        max_k = min(5, len(words))
        for k in range(max_k, 0, -1):
            suf = " ".join(words[-k:])
            pid_o, pid_n, pid_o_set = self._resolve_province(suf, allow_fuzzy=False)
            if pid_n is not None:
                prefix = " ".join(words[:-k]).strip()
                return prefix, suf, pid_n, pid_o_set

        # Try SymSpell fuzzy match on full segment first, then suffixes len >= 4
        pid_o, pid_n, pid_o_set = self._resolve_province(seg, allow_fuzzy=True)
        if pid_n is not None:
            return "", seg, pid_n, pid_o_set

        for k in range(min(4, len(words)), 0, -1):
            suf = " ".join(words[-k:])
            if len(normalize_text(suf)) >= 4:
                pid_o, pid_n, pid_o_set = self._resolve_province(suf, allow_fuzzy=True)
                if pid_n is not None:
                    prefix = " ".join(words[:-k]).strip()
                    return prefix, suf, pid_n, pid_o_set

        return seg, None, None, set()

    def _extract_suffix_span(
        self,
        seg: str,
        matcher,
        max_words: int = 5,
    ) -> Tuple[str, Optional[str], Any]:
        """Try matching whole segment first, then right-hand token suffixes via matcher."""
        hit = matcher(seg, allow_fuzzy=False)
        if hit is not None:
            return "", seg, hit

        words = seg.split()
        for k in range(min(max_words, len(words)), 0, -1):
            suf = " ".join(words[-k:])
            hit = matcher(suf, allow_fuzzy=False)
            if hit is not None:
                return " ".join(words[:-k]).strip(), suf, hit

        hit = matcher(seg, allow_fuzzy=True)
        if hit is not None:
            return "", seg, hit

        for k in range(min(4, len(words)), 0, -1):
            suf = " ".join(words[-k:])
            if len(normalize_text(suf)) >= 4:
                hit = matcher(suf, allow_fuzzy=True)
                if hit is not None:
                    return " ".join(words[:-k]).strip(), suf, hit

        return seg, None, None

    def suggest_wards(self, raw_text: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Suggest Top-K 2025 wards when an address is insufficient or uncertain."""
        if limit <= 0 or not raw_text or not raw_text.strip():
            return []

        repaired = repair_telex_typos(raw_text)
        segs = _protect_and_split_segments(repaired)
        if not segs:
            return []

        # Extract province if present
        prefix_last, _prov_txt, pid_n, pid_o_set = self._extract_suffix_province(segs[-1])
        rem = segs[:-1] + ([prefix_last] if prefix_last else [])

        did_o: Optional[int] = None
        ward_query = ""
        if rem:
            cand_last = rem[-1]
            did_o = self._resolve_district(cand_last, pid_o_filter=pid_o_set or None)
            if did_o is None and len(cand_last.split()) > 1:
                _, _, did_o = self._extract_suffix_span(
                    cand_last,
                    lambda s, allow_fuzzy: self._resolve_district(
                        s, pid_o_filter=pid_o_set or None, allow_fuzzy=allow_fuzzy
                    ),
                )
            if did_o is not None and pid_n is None and len(self.dp[did_o]) == 1:
                pid_o = self.dp[did_o][0]
                pid_o_set = {pid_o}
                pid_n = self.pm[pid_o]
            ward_query = normalize_text(cand_last)

        suggestions: List[Dict[str, Any]] = []
        seen_keys: Set[Tuple[int, int]] = set()

        # 1. If a legacy district was matched (e.g. 'Na Hang, Tuyen Quang' -> 5 2025 communes)
        if did_o is not None:
            dist_norm = self.do[did_o]
            ward_counts = self.district_to_2025_wards.get(did_o, {})
            ranked = sorted(
                ward_counts.items(),
                key=lambda kv: (
                    0 if self.wn[kv[0][0]] == dist_norm else 1,
                    -kv[1],
                    self.wn[kv[0][0]],
                ),
            )
            for (wid_n, w_pid_n), count in ranked:
                if pid_n is not None and w_pid_n != pid_n:
                    continue
                if (wid_n, w_pid_n) in seen_keys:
                    continue
                seen_keys.add((wid_n, w_pid_n))
                cand = self._format_candidate(wid_n, w_pid_n)
                is_homonym = self.wn[wid_n] == dist_norm
                suggestions.append(
                    {
                        "new_ward": cand["new_ward"],
                        "new_province": cand["new_province"],
                        "new_ward_code": cand["new_ward_code"],
                        "new_province_code": cand["new_province_code"],
                        "lat": cand["lat"],
                        "lon": cand["lon"],
                        "area_km2": cand["area_km2"],
                        "score": 1.0 if is_homonym else round(0.75 + min(0.2, count * 0.03), 3),
                        "reason": "homonymous_ward" if is_homonym else "district_merger",
                    }
                )
                if len(suggestions) >= limit:
                    return suggestions

        # 2. Match fragment / fuzzy ward within province (or globally)
        if ward_query:
            q_stems, _, _ = _extract_ward_candidates(ward_query)
            q_stem = q_stems[0] if q_stems else ward_query
            q_tokens = set(q_stem.split())

            pool_wids: Iterable[int]
            if pid_n is not None:
                pool_wids = self.province_to_2025_wards.get(pid_n, ())
            else:
                pool_wids = range(len(self.wn))

            scored: List[Tuple[float, int, int]] = []
            for wid_n in pool_wids:
                wn_name = self.wn[wid_n]
                score = 0.0
                if wn_name == q_stem:
                    score = 0.99
                elif wn_name.startswith(q_stem) or q_stem.startswith(wn_name):
                    score = 0.85
                elif q_tokens:
                    w_tokens = set(wn_name.split())
                    inter = len(q_tokens & w_tokens)
                    if inter > 0:
                        score = 0.65 * (inter / len(q_tokens | w_tokens))
                if score > 0.0:
                    target_pn = pid_n if pid_n is not None else int(self.en[wid_n][0][0])
                    if (wid_n, target_pn) not in seen_keys:
                        scored.append((score, wid_n, target_pn))

            scored.sort(key=lambda x: (-x[0], self.wn[x[1]]))
            for score, wid_n, target_pn in scored:
                if (wid_n, target_pn) in seen_keys:
                    continue
                seen_keys.add((wid_n, target_pn))
                cand = self._format_candidate(wid_n, target_pn)
                suggestions.append(
                    {
                        "new_ward": cand["new_ward"],
                        "new_province": cand["new_province"],
                        "new_ward_code": cand["new_ward_code"],
                        "new_province_code": cand["new_province_code"],
                        "lat": cand["lat"],
                        "lon": cand["lon"],
                        "area_km2": cand["area_km2"],
                        "score": round(score, 3),
                        "reason": "fuzzy_ward",
                    }
                )
                if len(suggestions) >= limit:
                    return suggestions

        # 3. Fallback: top wards in matched province
        if pid_n is not None:
            pool_wids = self.province_to_2025_wards.get(pid_n, ())
            ranked_prov = sorted(
                pool_wids,
                key=lambda wid: (
                    int(self._pick_en_row(wid, pid_n)[5]) if self._pick_en_row(wid, pid_n) else 9,
                    -len(self.rev.get(f"{wid}:{pid_n}", ())),
                    self.wn[wid],
                ),
            )
            for wid_n in ranked_prov:
                if (wid_n, pid_n) in seen_keys:
                    continue
                seen_keys.add((wid_n, pid_n))
                cand = self._format_candidate(wid_n, pid_n)
                suggestions.append(
                    {
                        "new_ward": cand["new_ward"],
                        "new_province": cand["new_province"],
                        "new_ward_code": cand["new_ward_code"],
                        "new_province_code": cand["new_province_code"],
                        "lat": cand["lat"],
                        "lon": cand["lon"],
                        "area_km2": cand["area_km2"],
                        "score": 0.4,
                        "reason": "province_ward",
                    }
                )
                if len(suggestions) >= limit:
                    break

        return suggestions[:limit]

    def parse_and_normalize_v3(self, raw_text: str) -> Dict[str, Any]:
        """Parse freeform Vietnamese address -> span-disjoint 2025 2-tier normalization + sufficiency."""
        if not raw_text or not raw_text.strip():
            return {
                "raw_text": raw_text or "",
                "repaired_text": "",
                "street": "",
                "parsed_ward": None,
                "parsed_district": None,
                "parsed_province": None,
                "status": "not_found",
                "sufficiency": "address_insufficient",
                "is_divided": False,
                "new_ward": None,
                "new_province": None,
                "new_ward_code": None,
                "new_province_code": None,
                "lat": None,
                "lon": None,
                "area_km2": None,
                "candidates": [],
                "suggested_wards": [],
            }

        repaired = repair_telex_typos(raw_text)
        segs = _protect_and_split_segments(repaired)
        if not segs:
            return {
                "raw_text": raw_text,
                "repaired_text": repaired,
                "street": "",
                "parsed_ward": None,
                "parsed_district": None,
                "parsed_province": None,
                "status": "not_found",
                "sufficiency": "address_insufficient",
                "is_divided": False,
                "new_ward": None,
                "new_province": None,
                "new_ward_code": None,
                "new_province_code": None,
                "lat": None,
                "lon": None,
                "area_km2": None,
                "candidates": [],
                "suggested_wards": [],
            }

        # Step 1: Span-disjoint province extraction from right
        prefix_last, parsed_province, pid_n, pid_o_set = self._extract_suffix_province(segs[-1])
        if parsed_province is not None:
            rem = segs[:-1] + ([prefix_last] if prefix_last else [])
        else:
            rem = list(segs)

        parsed_district: Optional[str] = None
        parsed_ward: Optional[str] = None
        street_segs: List[str] = []
        ward_has_explicit_prefix = False
        dist_has_explicit_prefix = False

        # Step 2: Check island special zone (unless rem[-2] is an explicit sub-ward inside that
        # island district or rem[-1] has explicit 2025 'Đặc khu' prefix)
        if rem:
            _, w_hint_last, w_exp_last = _extract_ward_candidates(rem[-1])
            isle_tup = self._check_isle_match(rem[-1], "", pid_o_set, pid_n)
            if isle_tup is not None:
                i_did_o, i_pid_o, i_wid_n, i_pid_n = isle_tup
                has_subward = len(rem) >= 2 and (
                    self._resolve_wo(rem[-2], did_o=i_did_o, pid_o_filter={i_pid_o}) is not None
                    or self._extract_suffix_span(
                        rem[-2],
                        lambda s, allow_fuzzy: self._resolve_wo(
                            s, did_o=i_did_o, pid_o_filter={i_pid_o}, allow_fuzzy=allow_fuzzy
                        ),
                    )[2]
                    is not None
                )
                if not has_subward and not (w_exp_last and w_hint_last == 3):
                    parsed_district = rem[-1]
                    street_segs = rem[:-1]
                    cand = self._format_candidate(i_wid_n, i_pid_n, is_default=True, wt_hint=3)
                    conv = self._build_conversion_result("converted", False, [cand])
                    return {
                        "raw_text": raw_text,
                        "repaired_text": repaired,
                        "street": ", ".join(street_segs),
                        "parsed_ward": None,
                        "parsed_district": parsed_district,
                        "parsed_province": parsed_province,
                        **conv,
                        "sufficiency": "address_sufficient",
                        "suggested_wards": [],
                    }

        # Step 3: Extract District and/or Ward strictly left of Province
        if rem:
            last_seg = rem[-1]
            _, ward_hint, ward_explicit = _extract_ward_candidates(last_seg)
            _, dist_explicit = _extract_district_candidates(last_seg)

            if ward_explicit:
                # Explicit ward prefix (Phuong/Xa/Thi tran/Dac khu)
                parsed_ward = last_seg
                ward_has_explicit_prefix = True
                street_segs = rem[:-1]
            elif dist_explicit:
                # Explicit district prefix (Quan/Huyen/Thi xa/TP/Q/H)
                did_o = self._resolve_district(last_seg, pid_o_filter=pid_o_set or None)
                if did_o is not None:
                    parsed_district = last_seg
                    dist_has_explicit_prefix = True
                    if not pid_o_set and len(self.dp[did_o]) == 1:
                        pid_o_set = {self.dp[did_o][0]}
                        pid_n = self.pm[self.dp[did_o][0]]
                    if len(rem) >= 2:
                        wid_o = self._resolve_wo(
                            rem[-2], did_o=did_o, pid_o_filter=pid_o_set or None
                        )
                        if wid_o is not None:
                            parsed_ward = rem[-2]
                            street_segs = rem[:-2]
                        else:
                            # Comma-free suffix inside rem[-2]
                            left_w, w_suf, wid_o_suf = self._extract_suffix_span(
                                rem[-2],
                                lambda s, allow_fuzzy: self._resolve_wo(
                                    s,
                                    did_o=did_o,
                                    pid_o_filter=pid_o_set or None,
                                    allow_fuzzy=allow_fuzzy,
                                ),
                            )
                            if wid_o_suf is not None:
                                parsed_ward = w_suf
                                street_segs = rem[:-2] + ([left_w] if left_w else [])
                            else:
                                street_segs = rem[:-1]
                    else:
                        street_segs = []
                else:
                    street_segs = rem
            else:
                # Bare rightmost remaining segment (could be District, 2025 Ward, Legacy Ward, or Comma-free)
                if len(rem) >= 2:
                    did_o = self._resolve_district(last_seg, pid_o_filter=pid_o_set or None)
                    if did_o is not None:
                        tmp_pid_o = pid_o_set or (
                            {self.dp[did_o][0]} if len(self.dp[did_o]) == 1 else set()
                        )
                        wid_o_left = self._resolve_wo(
                            rem[-2], did_o=did_o, pid_o_filter=tmp_pid_o or None
                        )
                        if wid_o_left is not None:
                            parsed_district = last_seg
                            parsed_ward = rem[-2]
                            street_segs = rem[:-2]
                        else:
                            # rem[-2] is NOT a ward inside did_o -> check if last_seg is a 2025/legacy ward
                            # (e.g. '12 Le Loi, Na Hang, Tuyen Quang'!)
                            wid_n_hit, _ = self._resolve_wn(last_seg, pid_n=pid_n)
                            wid_o_hit = self._resolve_wo(last_seg, pid_o_filter=pid_o_set or None)
                            if wid_n_hit is not None or wid_o_hit is not None:
                                parsed_ward = last_seg
                                street_segs = rem[:-1]
                            else:
                                parsed_district = last_seg
                                street_segs = rem[:-1]
                    else:
                        # last_seg is not a district -> check as ward (2025 or legacy)
                        wid_n_hit, _ = self._resolve_wn(last_seg, pid_n=pid_n)
                        wid_o_hit = self._resolve_wo(last_seg, pid_o_filter=pid_o_set or None)
                        if wid_n_hit is not None or wid_o_hit is not None:
                            parsed_ward = last_seg
                            street_segs = rem[:-1]
                        else:
                            # Try comma-free suffix inside last_seg
                            left_s, w_suf, _ = self._extract_suffix_span(
                                last_seg,
                                lambda s, allow_fuzzy: (
                                    self._resolve_wn(s, pid_n=pid_n, allow_fuzzy=allow_fuzzy)[0]
                                    if self._resolve_wn(s, pid_n=pid_n, allow_fuzzy=allow_fuzzy)[0]
                                    is not None
                                    else self._resolve_wo(
                                        s, pid_o_filter=pid_o_set or None, allow_fuzzy=allow_fuzzy
                                    )
                                ),
                            )
                            if w_suf is not None:
                                parsed_ward = w_suf
                                street_segs = rem[:-1] + ([left_s] if left_s else [])
                            else:
                                street_segs = rem
                else:
                    # Single segment left before province (e.g. 'Na Hang, Tuyen Quang' or comma-free!)
                    whole_did = self._resolve_district(last_seg, pid_o_filter=pid_o_set or None)
                    whole_wn, _ = self._resolve_wn(last_seg, pid_n=pid_n)
                    whole_wo = self._resolve_wo(last_seg, pid_o_filter=pid_o_set or None)

                    if whole_did is not None or whole_wn is not None or whole_wo is not None:
                        if whole_did is not None and not (whole_wn is None and whole_wo is None):
                            # Overlaps district & ward (e.g. 'Na Hang' in 'Na Hang, Tuyen Quang')
                            parsed_district = last_seg
                            parsed_ward = last_seg
                        elif whole_did is not None:
                            parsed_district = last_seg
                        else:
                            parsed_ward = last_seg
                        street_segs = []
                    else:
                        # Comma-free multi-token phrase: extract District (optional) + Ward from right!
                        left_d, d_suf, did_suf = self._extract_suffix_span(
                            last_seg,
                            lambda s, allow_fuzzy: self._resolve_district(
                                s, pid_o_filter=pid_o_set or None, allow_fuzzy=allow_fuzzy
                            ),
                        )
                        if did_suf is not None and left_d:
                            tmp_pid_o = pid_o_set or (
                                {self.dp[did_suf][0]} if len(self.dp[did_suf]) == 1 else set()
                            )
                            left_w, w_suf, wid_suf = self._extract_suffix_span(
                                left_d,
                                lambda s, allow_fuzzy: self._resolve_wo(
                                    s,
                                    did_o=did_suf,
                                    pid_o_filter=tmp_pid_o or None,
                                    allow_fuzzy=allow_fuzzy,
                                ),
                            )
                            if wid_suf is not None:
                                parsed_district = d_suf
                                parsed_ward = w_suf
                                street_segs = [left_w] if left_w else []
                            else:
                                # Check if suffix of last_seg is a 2025/legacy ward with street prefix
                                left_w2, w_suf2, _ = self._extract_suffix_span(
                                    last_seg,
                                    lambda s, allow_fuzzy: (
                                        self._resolve_wn(s, pid_n=pid_n, allow_fuzzy=allow_fuzzy)[0]
                                        if self._resolve_wn(s, pid_n=pid_n, allow_fuzzy=allow_fuzzy)[0]
                                        is not None
                                        else self._resolve_wo(
                                            s, pid_o_filter=pid_o_set or None, allow_fuzzy=allow_fuzzy
                                        )
                                    ),
                                )
                                if w_suf2 is not None:
                                    parsed_ward = w_suf2
                                    street_segs = [left_w2] if left_w2 else []
                                else:
                                    parsed_district = d_suf
                                    street_segs = [left_d]
                        else:
                            left_w2, w_suf2, _ = self._extract_suffix_span(
                                last_seg,
                                lambda s, allow_fuzzy: (
                                    self._resolve_wn(s, pid_n=pid_n, allow_fuzzy=allow_fuzzy)[0]
                                    if self._resolve_wn(s, pid_n=pid_n, allow_fuzzy=allow_fuzzy)[0]
                                    is not None
                                    else self._resolve_wo(
                                        s, pid_o_filter=pid_o_set or None, allow_fuzzy=allow_fuzzy
                                    )
                                ),
                            )
                            if w_suf2 is not None:
                                parsed_ward = w_suf2
                                street_segs = [left_w2] if left_w2 else []
                            else:
                                street_segs = [last_seg]

        street = ", ".join(s for s in street_segs if s)
        has_street_cue = _has_street_or_rural_cue(street)

        # Step 4: Smart 2025 merger disambiguation!
        # E.g. 'Na Hang, Tuyen Quang' (no street/house/rural cue, no explicit ward prefix,
        # overlaps a legacy district spanning multiple 2025 wards) -> address_insufficient!
        if parsed_district is not None and (parsed_ward is None or parsed_ward == parsed_district):
            did_check = self._resolve_district(parsed_district, pid_o_filter=pid_o_set or None)
            if did_check is not None:
                merged_2025_count = len(self.district_to_2025_wards.get(did_check, {}))
                if dist_has_explicit_prefix or (
                    merged_2025_count >= 2 and not has_street_cue and not ward_has_explicit_prefix
                ):
                    prov_disp = self.pn_disp[pid_n] if pid_n is not None else None
                    prov_code = self.pn_code[pid_n] if pid_n is not None else None
                    return {
                        "raw_text": raw_text,
                        "repaired_text": repaired,
                        "street": street,
                        "parsed_ward": None if dist_has_explicit_prefix else parsed_ward,
                        "parsed_district": parsed_district,
                        "parsed_province": parsed_province,
                        "status": "insufficient",
                        "sufficiency": "address_insufficient",
                        "is_divided": False,
                        "new_ward": None,
                        "new_province": prov_disp,
                        "new_ward_code": None,
                        "new_province_code": prov_code,
                        "lat": None,
                        "lon": None,
                        "area_km2": None,
                        "candidates": [],
                        "suggested_wards": self.suggest_wards(raw_text, limit=3),
                    }
            # Has street cue (e.g. '12 Le Loi, Na Hang, Tuyen Quang') -> resolve as 2025 ward!
            if parsed_ward is not None and has_street_cue:
                parsed_district = None

        # Step 5: Convert resolved ward/district/province to 2025 2-tier
        if parsed_ward:
            conv = self.convert_legacy_to_2025(
                parsed_ward, parsed_district or "", parsed_province or ""
            )
            if conv["status"] in {"converted", "already_2025"}:
                return {
                    "raw_text": raw_text,
                    "repaired_text": repaired,
                    "street": street,
                    "parsed_ward": parsed_ward,
                    "parsed_district": parsed_district,
                    "parsed_province": parsed_province,
                    **conv,
                    "sufficiency": "address_sufficient",
                    "suggested_wards": [],
                }
            # Ambiguous or not found -> insufficient + suggestions
            return {
                "raw_text": raw_text,
                "repaired_text": repaired,
                "street": street,
                "parsed_ward": parsed_ward,
                "parsed_district": parsed_district,
                "parsed_province": parsed_province,
                **conv,
                "sufficiency": "address_insufficient",
                "suggested_wards": self.suggest_wards(raw_text, limit=3),
            }

        # Only province (or nothing) resolved
        prov_disp = self.pn_disp[pid_n] if pid_n is not None else None
        prov_code = self.pn_code[pid_n] if pid_n is not None else None
        return {
            "raw_text": raw_text,
            "repaired_text": repaired,
            "street": street,
            "parsed_ward": None,
            "parsed_district": parsed_district,
            "parsed_province": parsed_province,
            "status": "insufficient" if pid_n is not None else "not_found",
            "sufficiency": "address_insufficient",
            "is_divided": False,
            "new_ward": None,
            "new_province": prov_disp,
            "new_ward_code": None,
            "new_province_code": prov_code,
            "lat": None,
            "lon": None,
            "area_km2": None,
            "candidates": [],
            "suggested_wards": self.suggest_wards(raw_text, limit=3),
        }
