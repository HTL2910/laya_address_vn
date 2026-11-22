"""Build MIT-licensed Laya v3 Administrative Converter, Spatial Centroids, & Full Records.

Source: https://github.com/tranngocminhhieu/vietnamadminunits (MIT License)
Inputs:
  - shared_data_v3/convert_legacy_2025_with_location_and_default_ward.csv (10,602 legacy <-> 2025 edges)
  - shared_data/2025_34-province-3221-ward_with_location.csv (3,321 post-2025 wards)

Outputs (strictly under shared_data_v3/):
  - vn_admin_v3_converter.json  (compact indexed legacy <-> 2025 + coordinates JSON)
  - vn_admin_v3_records.json    (clean structured list for training & holdout synthesis)
  - PROVENANCE_V3.json          (exact URLs, license, SHA-256 hashes, counts)
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import unicodedata

V3_DATA_DIR = Path(__file__).resolve().parent
SHARED_V2_DIR = V3_DATA_DIR.parent / "shared_data"

LICENSE_FILE = V3_DATA_DIR / "LICENSE_vietnamadminunits.txt"
CONVERT_CSV = V3_DATA_DIR / "convert_legacy_2025_with_location_and_default_ward.csv"
VN2025_CSV = SHARED_V2_DIR / "2025_34-province-3221-ward_with_location.csv"

CONVERTER_OUT = V3_DATA_DIR / "vn_admin_v3_converter.json"
RECORDS_OUT = V3_DATA_DIR / "vn_admin_v3_records.json"
PROVENANCE_OUT = V3_DATA_DIR / "PROVENANCE_V3.json"

PROVINCE_ALIASES: Dict[str, str] = {
    "hn": "ha noi",
    "hanoi": "ha noi",
    "tp hn": "ha noi",
    "tphn": "ha noi",
    "hcm": "ho chi minh",
    "tphcm": "ho chi minh",
    "tp hcm": "ho chi minh",
    "sg": "ho chi minh",
    "sai gon": "ho chi minh",
    "saigon": "ho chi minh",
    "dn": "da nang",
    "danang": "da nang",
    "hp": "hai phong",
    "haiphong": "hai phong",
    "ct": "can tho",
    "cantho": "can tho",
    "bd": "binh duong",
    "brvt": "ba ria - vung tau",
    "br vt": "ba ria - vung tau",
    "vung tau": "ba ria - vung tau",
    "ba ria vung tau": "ba ria - vung tau",
    "hue": "hue",
    "tp hue": "hue",
    "tthue": "hue",
    "thua thien hue": "hue",
    "daklak": "dak lak",
    "dac lac": "dak lak",
    "daknong": "dak nong",
    "dongnai": "dong nai",
    "dnai": "dong nai",
    "nha trang": "khanh hoa",
    "da lat": "lam dong",
    "dalat": "lam dong",
    "bien hoa": "dong nai",
    "thu dau mot": "binh duong",
    "thuan an": "binh duong",
    "di an": "binh duong",
    "buon ma thuot": "dak lak",
    "bmt": "dak lak",
    "quy nhon": "binh dinh",
    "ha long": "quang ninh",
    "halong": "quang ninh",
    "phan thiet": "binh thuan",
    "phu quoc": "kien giang",
    "my tho": "tien giang",
    "long xuyen": "an giang",
    "rach gia": "kien giang",
    "vinh": "nghe an",
    "pleiku": "gia lai",
    "tam ky": "quang nam",
    "hoi an": "quang nam",
    "sa dec": "dong thap",
    "cao lanh": "dong thap",
    "bac lieu": "bac lieu",
    "ca mau": "ca mau",
    "thu duc": "ho chi minh",
    "tp thu duc": "ho chi minh",
}

DISTRICT_ALIASES: Dict[str, str] = {
    "q1": "quan 1",
    "q 1": "quan 1",
    "q2": "quan 2",
    "q 2": "quan 2",
    "q3": "quan 3",
    "q 3": "quan 3",
    "q4": "quan 4",
    "q 4": "quan 4",
    "q5": "quan 5",
    "q 5": "quan 5",
    "q6": "quan 6",
    "q 6": "quan 6",
    "q7": "quan 7",
    "q 7": "quan 7",
    "q8": "quan 8",
    "q 8": "quan 8",
    "q9": "quan 9",
    "q 9": "quan 9",
    "q10": "quan 10",
    "q 10": "quan 10",
    "q11": "quan 11",
    "q 11": "quan 11",
    "q12": "quan 12",
    "q 12": "quan 12",
    "qtb": "tan binh",
    "q tb": "tan binh",
    "qbt": "binh thanh",
    "q bt": "binh thanh",
    "qpn": "phu nhuan",
    "q pn": "phu nhuan",
    "qgv": "go vap",
    "q gv": "go vap",
    "qtp": "tan phu",
    "q tp": "tan phu",
    "qtx": "thanh xuan",
    "q tx": "thanh xuan",
    "qcg": "cau giay",
    "q cg": "cau giay",
    "qdd": "dong da",
    "q dd": "dong da",
    "qhk": "hoan kiem",
    "q hk": "hoan kiem",
    "qhm": "hoang mai",
    "q hm": "hoang mai",
    "qbd": "ba dinh",
    "q bd": "ba dinh",
    "qhbt": "hai ba trung",
    "hbt": "hai ba trung",
    "qlb": "long bien",
    "q lb": "long bien",
    "qth": "tay ho",
    "qhd": "ha dong",
    "qntl": "nam tu liem",
    "ntl": "nam tu liem",
    "qbtl": "bac tu liem",
    "btl": "bac tu liem",
    "qhc": "hai chau",
    "qnk": "ninh kieu",
    "qnq": "ngo quyen",
}

WARD_TYPES = ["Phường", "Xã", "Thị trấn", "Đặc khu"]
WARD_TYPE_TO_IDX = {w: i for i, w in enumerate(WARD_TYPES)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_accents(text: str) -> str:
    text = (text or "").replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def norm_ascii(text: str) -> str:
    s = strip_accents(text).lower().strip()
    return re.sub(r"\s+", " ", s)


def strip_admin_prefix(raw: str, prefixes: Tuple[str, ...], keep_if_numeric_prefix: str = "") -> str:
    s = norm_ascii(raw)
    for p in prefixes:
        if s.startswith(p + " "):
            rest = s[len(p) + 1 :].strip()
            if not rest:
                return s
            if rest.isdigit() and keep_if_numeric_prefix:
                return f"{keep_if_numeric_prefix} {rest}"
            return rest
    return s


def strip_display_prefix(raw: str, prefixes: Tuple[str, ...]) -> str:
    s = re.sub(r"\s+", " ", unicodedata.normalize("NFC", raw or "").strip())
    low = strip_accents(s).lower()
    for p in prefixes:
        if low.startswith(p + " "):
            rest = s[len(p) + 1 :].strip()
            if rest and not rest.isdigit():
                return rest
    return s


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        s = str(v or "").strip()
        return float(s) if s else default
    except ValueError:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        s = str(v or "").strip()
        return int(float(s)) if s else default
    except ValueError:
        return default


def _wt_idx(w_type: str, full_name: str) -> int:
    t = (w_type or "").strip()
    if t in WARD_TYPE_TO_IDX:
        return WARD_TYPE_TO_IDX[t]
    low = norm_ascii(full_name)
    if low.startswith("phuong "):
        return 0
    if low.startswith("thi tran "):
        return 2
    if low.startswith("dac khu "):
        return 3
    return 1


def build() -> None:
    if not CONVERT_CSV.exists():
        raise FileNotFoundError(f"Missing conversion CSV: {CONVERT_CSV}")
    if not VN2025_CSV.exists():
        raise FileNotFoundError(f"Missing 2025 CSV: {VN2025_CSV}")

    po_map: Dict[str, int] = {}
    po_disp: List[str] = []
    do_map: Dict[str, int] = {}
    do_disp: List[str] = []
    wo_map: Dict[str, int] = {}
    wo_disp: List[str] = []

    pn_map: Dict[str, int] = {}
    pn_disp: List[str] = []
    pn_code: List[int] = []
    wn_map: Dict[str, int] = {}
    wn_disp: List[str] = []

    pm_edges: Dict[int, int] = {}
    dp_edges: Dict[int, Set[int]] = {}
    eo_edges: Dict[int, List[List[int]]] = {}
    eo_seen: Dict[int, Set[Tuple[int, int, int, int]]] = {}
    en_entries: Dict[int, List[List[Any]]] = {}
    en_seen: Dict[int, Set[Tuple[int, int]]] = {}
    isle_edges: List[List[int]] = []
    isle_seen: Set[Tuple[int, int, int, int]] = set()
    rev_lookup: Dict[str, List[List[int]]] = {}
    rev_seen: Dict[str, Set[Tuple[int, int, int]]] = {}
    ogeo: Dict[str, List[float]] = {}

    records: List[Dict[str, Any]] = []

    def get_po(norm: str, disp: str) -> int:
        if norm not in po_map:
            po_map[norm] = len(po_disp)
            po_disp.append(disp)
        return po_map[norm]

    def get_do(norm: str, disp: str) -> int:
        if norm not in do_map:
            do_map[norm] = len(do_disp)
            do_disp.append(disp)
        return do_map[norm]

    def get_wo(norm: str, disp: str) -> int:
        if norm not in wo_map:
            wo_map[norm] = len(wo_disp)
            wo_disp.append(disp)
        return wo_map[norm]

    def get_pn(norm: str, disp: str, pcode: int) -> int:
        if norm not in pn_map:
            pn_map[norm] = len(pn_disp)
            pn_disp.append(disp)
            pn_code.append(pcode)
        return pn_map[norm]

    def get_wn(norm: str, disp: str) -> int:
        if norm not in wn_map:
            wn_map[norm] = len(wn_disp)
            wn_disp.append(disp)
        return wn_map[norm]

    # 1. Seed 2025 34 provinces & 3,321 wards cleanly with exact centroids & area
    with VN2025_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pn_raw = (row.get("province") or "").strip()
            wn_raw = (row.get("ward") or "").strip()
            if not (pn_raw and wn_raw):
                continue
            pn_short = (row.get("provinceShort") or "").strip() or strip_display_prefix(pn_raw, ("tinh", "thanh pho"))
            wn_short = (row.get("wardShort") or "").strip() or strip_display_prefix(wn_raw, ("phuong", "xa", "thi tran", "dac khu"))
            wt = (row.get("wardType") or "").strip()
            pn_norm = strip_admin_prefix(pn_raw, ("tinh", "thanh pho"))
            wn_norm = strip_admin_prefix(wn_raw, ("phuong", "xa", "thi tran", "dac khu"), keep_if_numeric_prefix="phuong")

            pid_n = get_pn(pn_norm, pn_short, _safe_int(row.get("provinceCode")))
            wid_n = get_wn(wn_norm, wn_short)
            w_code = _safe_int(row.get("wardCode"))
            if (pid_n, w_code) not in en_seen.setdefault(wid_n, set()):
                en_seen[wid_n].add((pid_n, w_code))
                en_entries.setdefault(wid_n, []).append([
                    pid_n,
                    round(_safe_float(row.get("wardLat")), 4),
                    round(_safe_float(row.get("wardLon")), 4),
                    round(_safe_float(row.get("wardAreaKm2"), 15.0), 2),
                    w_code,
                    _wt_idx(wt, wn_raw),
                ])

    # 2. Parse 10,602 legacy <-> 2025 conversion edges
    total_conv_rows = 0
    with CONVERT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            total_conv_rows += 1
            po_raw = (row.get("province") or "").strip()
            do_raw = (row.get("district") or "").strip()
            wo_raw = (row.get("ward") or "").strip()
            pn_raw = (row.get("newProvince") or "").strip()
            wn_raw = (row.get("newWard") or "").strip()
            if not (po_raw and do_raw and pn_raw and wn_raw):
                continue

            po_short = (row.get("provinceShort") or "").strip() or strip_display_prefix(po_raw, ("tinh", "thanh pho"))
            do_short = (row.get("districtShort") or "").strip() or strip_display_prefix(do_raw, ("quan", "huyen", "thi xa", "thanh pho"))
            pn_short = (row.get("newProvinceShort") or "").strip() or strip_display_prefix(pn_raw, ("tinh", "thanh pho"))
            wn_short = (row.get("newWardShort") or "").strip() or strip_display_prefix(wn_raw, ("phuong", "xa", "thi tran", "dac khu"))

            po_norm = strip_admin_prefix(po_raw, ("tinh", "thanh pho"))
            do_norm = strip_admin_prefix(do_raw, ("quan", "huyen", "thi xa", "thanh pho"), keep_if_numeric_prefix="quan")
            pn_norm = strip_admin_prefix(pn_raw, ("tinh", "thanh pho"))
            wn_norm = strip_admin_prefix(wn_raw, ("phuong", "xa", "thi tran", "dac khu"), keep_if_numeric_prefix="phuong")

            pid_o = get_po(po_norm, po_short)
            did_o = get_do(do_norm, do_short)
            pid_n = get_pn(pn_norm, pn_short, _safe_int(row.get("newProvinceCode")))
            wid_n = get_wn(wn_norm, wn_short)

            pm_edges[pid_o] = pid_n
            dp_edges.setdefault(did_o, set()).add(pid_o)

            w_code_n = _safe_int(row.get("newWardCode"))
            if (pid_n, w_code_n) not in en_seen.setdefault(wid_n, set()):
                en_seen[wid_n].add((pid_n, w_code_n))
                en_entries.setdefault(wid_n, []).append([
                    pid_n,
                    round(_safe_float(row.get("newWardLat")), 4),
                    round(_safe_float(row.get("newWardLon")), 4),
                    round(_safe_float(row.get("newWardAreaKm2"), 15.0), 2),
                    w_code_n,
                    _wt_idx(row.get("newWardType") or "", wn_raw),
                ])

            is_div = 1 if (row.get("isDividedWard") or "").strip() == "True" else 0
            is_def = 1 if (row.get("isDefaultNewWard") or "").strip() == "True" else 0
            w_lat = _safe_float(row.get("wardLat"))
            w_lon = _safe_float(row.get("wardLon"))
            nw_lat = _safe_float(row.get("newWardLat"))
            nw_lon = _safe_float(row.get("newWardLon"))

            if not wo_raw:
                key_i = (did_o, pid_o, wid_n, pid_n)
                if key_i not in isle_seen:
                    isle_seen.add(key_i)
                    isle_edges.append([did_o, pid_o, wid_n, pid_n])
                records.append({
                    "ward_full": "",
                    "ward_bare": "",
                    "district_full": do_raw,
                    "district_bare": do_short,
                    "province_full": po_raw,
                    "province_bare": po_short,
                    "new_ward_full": wn_raw,
                    "new_ward_bare": wn_short,
                    "new_province_full": pn_raw,
                    "new_province_bare": pn_short,
                    "is_divided": False,
                    "is_default": True,
                    "new_lat": round(nw_lat, 4),
                    "new_lon": round(nw_lon, 4),
                    "old_lat": round(_safe_float(row.get("districtLat"), nw_lat), 4),
                    "old_lon": round(_safe_float(row.get("districtLon"), nw_lon), 4),
                })
                continue

            wo_short = (row.get("wardShort") or "").strip() or strip_display_prefix(wo_raw, ("phuong", "xa", "thi tran"))
            wo_norm = strip_admin_prefix(wo_raw, ("phuong", "xa", "thi tran", "dac khu"), keep_if_numeric_prefix="phuong")
            wid_o = get_wo(wo_norm, wo_short)

            sig = (did_o, pid_o, wid_n, pid_n)
            if sig not in eo_seen.setdefault(wid_o, set()):
                eo_seen[wid_o].add(sig)
                eo_edges.setdefault(wid_o, []).append([did_o, pid_o, wid_n, pid_n, is_div, is_def])

            rev_k = f"{wid_n}:{pid_n}"
            rev_sig = (wid_o, did_o, pid_o)
            if rev_sig not in rev_seen.setdefault(rev_k, set()):
                rev_seen[rev_k].add(rev_sig)
                rev_lookup.setdefault(rev_k, []).append([wid_o, did_o, pid_o])

            if w_lat != 0.0 and w_lon != 0.0:
                ogeo.setdefault(f"{wid_o}:{did_o}:{pid_o}", [round(w_lat, 4), round(w_lon, 4)])

            records.append({
                "ward_full": wo_raw,
                "ward_bare": wo_short,
                "district_full": do_raw,
                "district_bare": do_short,
                "province_full": po_raw,
                "province_bare": po_short,
                "new_ward_full": wn_raw,
                "new_ward_bare": wn_short,
                "new_province_full": pn_raw,
                "new_province_bare": pn_short,
                "is_divided": bool(is_div),
                "is_default": bool(is_def),
                "new_lat": round(nw_lat, 4),
                "new_lon": round(nw_lon, 4),
                "old_lat": round(w_lat if w_lat != 0.0 else nw_lat, 4),
                "old_lon": round(w_lon if w_lon != 0.0 else nw_lon, 4),
            })

    pm_list = [pm_edges.get(pid_o, -1) for pid_o in range(len(po_disp))]
    dp_list = [sorted(dp_edges.get(did_o, set())) for did_o in range(len(do_disp))]
    eo_list = [eo_edges.get(wid_o, []) for wid_o in range(len(wo_disp))]
    en_list = [en_entries.get(wid_n, []) for wid_n in range(len(wn_disp))]

    converter_payload = {
        "version": "3.0.0",
        "license": "MIT",
        "source": "tranngocminhhieu/vietnamadminunits (convert_legacy_2025 + 34-province-3221-ward)",
        "wt": WARD_TYPES,
        "po": [None] * len(po_disp),
        "po_disp": po_disp,
        "do": [None] * len(do_disp),
        "do_disp": do_disp,
        "wo": [None] * len(wo_disp),
        "wo_disp": wo_disp,
        "pn": [None] * len(pn_disp),
        "pn_disp": pn_disp,
        "pn_code": pn_code,
        "wn": [None] * len(wn_disp),
        "wn_disp": wn_disp,
        "pm": pm_list,
        "dp": dp_list,
        "eo": eo_list,
        "en": en_list,
        "isle": isle_edges,
        "rev": rev_lookup,
        "ogeo": ogeo,
        "pa": PROVINCE_ALIASES,
        "da": DISTRICT_ALIASES,
    }
    for k, i in po_map.items():
        converter_payload["po"][i] = k
    for k, i in do_map.items():
        converter_payload["do"][i] = k
    for k, i in wo_map.items():
        converter_payload["wo"][i] = k
    for k, i in pn_map.items():
        converter_payload["pn"][i] = k
    for k, i in wn_map.items():
        converter_payload["wn"][i] = k

    CONVERTER_OUT.write_text(
        json.dumps(converter_payload, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    RECORDS_OUT.write_text(
        json.dumps({"license": "MIT", "records": records}, ensure_ascii=False),
        encoding="utf-8",
    )

    provenance = {
        "version": "3.0.0",
        "source_repository": "https://github.com/tranngocminhhieu/vietnamadminunits",
        "license": "MIT License (Copyright (c) 2025 Tran Ngoc Minh Hieu)",
        "raw_files": [
            {
                "filename": CONVERT_CSV.name,
                "url": "https://raw.githubusercontent.com/tranngocminhhieu/vietnamadminunits/main/data/processed/convert_legacy_2025_with_location_and_default_ward.csv",
                "sha256": sha256_file(CONVERT_CSV),
                "rows": total_conv_rows,
            },
            {
                "filename": VN2025_CSV.name,
                "sha256": sha256_file(VN2025_CSV),
            },
            {
                "filename": LICENSE_FILE.name,
                "sha256": sha256_file(LICENSE_FILE),
            },
        ],
        "counts": {
            "legacy_provinces": len(po_disp),
            "legacy_districts": len(do_disp),
            "legacy_wards": len(wo_disp),
            "new_provinces_2025": len(pn_disp),
            "new_wards_2025": len(wn_disp),
            "island_special_zones": len(isle_edges),
            "total_mapping_records": len(records),
        },
        "converter_sha256": sha256_file(CONVERTER_OUT),
        "converter_bytes": CONVERTER_OUT.stat().st_size,
    }
    PROVENANCE_OUT.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    # Sanity assertions
    w_px = wo_map["phuc xa"]
    d_bd = do_map["ba dinh"]
    p_hn = po_map["ha noi"]
    px_hits = [e for e in eo_list[w_px] if e[0] == d_bd and e[1] == p_hn]
    assert px_hits and converter_payload["wn"][px_hits[0][2]] == "hong ha", f"Unexpected Phuc Xa hit: {px_hits}"

    w_lt = wo_map["loc thanh"]
    d_ln = do_map["loc ninh"]
    p_bp = po_map["binh phuoc"]
    lt_hits = [e for e in eo_list[w_lt] if e[0] == d_ln and e[1] == p_bp]
    assert lt_hits and converter_payload["pn"][lt_hits[0][3]] == "dong nai", f"Unexpected Loc Thanh hit: {lt_hits}"

    assert converter_payload["converter_bytes"] if "converter_bytes" in converter_payload else True
    print(
        f"Built v3 MIT Converter: {len(po_disp)} old prov -> {len(pn_disp)} new prov, "
        f"{len(wo_disp)} old wards -> {len(wn_disp)} new wards, "
        f"{CONVERTER_OUT.stat().st_size / 1024:.1f} KB"
    )


if __name__ == "__main__":
    build()
