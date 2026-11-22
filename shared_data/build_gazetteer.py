"""Build compact MIT-licensed Vietnam Administrative Gazetteer & Full Training Records.

Source: https://github.com/tranngocminhhieu/vietnamadminunits (MIT License)
Raw files:
- danhmuchanhchinh.gso.gov.vn_ward_2025-07-18.csv (63 provinces, 656+ districts, 10,035 wards)
- 2025_34-province-3221-ward_with_location.csv (34 provinces, 3,321 wards post-2025 merger)
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple

SHARED_DIR = Path(__file__).resolve().parent
LICENSE_FILE = SHARED_DIR / "LICENSE_vietnamadminunits.txt"
GSO_CSV = SHARED_DIR / "danhmuchanhchinh.gso.gov.vn_ward_2025-07-18.csv"
VN2025_CSV = SHARED_DIR / "2025_34-province-3221-ward_with_location.csv"

GAZETTEER_OUT = SHARED_DIR / "vn_admin_gazetteer.json"
RECORDS_OUT = SHARED_DIR / "vn_admin_full_records.json"
PROVENANCE_OUT = SHARED_DIR / "PROVENANCE.json"

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


def build() -> None:
    provinces: Dict[str, int] = {}
    districts: Dict[str, int] = {}
    wards: Dict[str, int] = {}
    ward_edges: Dict[int, Set[Tuple[int, int]]] = {}
    district_provinces: Dict[int, Set[int]] = {}

    legacy_records: List[Dict[str, str]] = []
    merged_2025_records: List[Dict[str, str]] = []

    def get_id(table: Dict[str, int], key: str) -> int:
        if key not in table:
            table[key] = len(table)
        return table[key]

    with GSO_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            w_raw = (row.get("Tên") or "").strip()
            w_type = (row.get("Cấp") or "").strip()
            d_raw = (row.get("Quận Huyện") or "").strip()
            p_raw = (row.get("Tỉnh / Thành Phố") or "").strip()
            if not (w_raw and d_raw and p_raw):
                continue

            p_norm = strip_admin_prefix(p_raw, ("tinh", "thanh pho"))
            d_norm = strip_admin_prefix(d_raw, ("quan", "huyen", "thi xa", "thanh pho"), keep_if_numeric_prefix="quan")
            w_norm = strip_admin_prefix(w_raw, ("phuong", "xa", "thi tran"), keep_if_numeric_prefix="phuong")

            pid = get_id(provinces, p_norm)
            did = get_id(districts, d_norm)
            wid = get_id(wards, w_norm)
            ward_edges.setdefault(wid, set()).add((did, pid))
            district_provinces.setdefault(did, set()).add(pid)

            legacy_records.append({
                "ward_full": w_raw,
                "ward_bare": strip_display_prefix(w_raw, ("phuong", "xa", "thi tran")),
                "ward_type": w_type or "Phường",
                "district_full": d_raw,
                "district_bare": strip_display_prefix(d_raw, ("quan", "huyen", "thi xa", "thanh pho")),
                "province_full": p_raw,
                "province_bare": strip_display_prefix(p_raw, ("tinh", "thanh pho")),
            })

    with VN2025_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            w_raw = (row.get("ward") or "").strip()
            w_short = (row.get("wardShort") or "").strip()
            w_type = (row.get("wardType") or "").strip()
            p_raw = (row.get("province") or "").strip()
            p_short = (row.get("provinceShort") or "").strip()
            if not (w_raw and p_raw):
                continue

            p_norm = strip_admin_prefix(p_raw, ("tinh", "thanh pho"))
            w_norm = strip_admin_prefix(w_raw, ("phuong", "xa", "thi tran", "dac khu"), keep_if_numeric_prefix="phuong")

            pid = get_id(provinces, p_norm)
            wid = get_id(wards, w_norm)
            ward_edges.setdefault(wid, set()).add((-1, pid))

            merged_2025_records.append({
                "ward_full": w_raw,
                "ward_bare": w_short or strip_display_prefix(w_raw, ("phuong", "xa", "thi tran", "dac khu")),
                "ward_type": w_type or "Phường",
                "province_full": p_raw,
                "province_bare": p_short or strip_display_prefix(p_raw, ("tinh", "thanh pho")),
            })

    # Also register "thanh pho hue" -> "thua thien hue" 2025 province equivalence
    if "hue" in provinces:
        PROVINCE_ALIASES["thua thien hue"] = "hue"
        PROVINCE_ALIASES["tthue"] = "hue"
        PROVINCE_ALIASES["tp hue"] = "hue"

    gazetteer = {
        "version": "1.0.0",
        "license": "MIT",
        "source": "tranngocminhhieu/vietnamadminunits (GSO 2025-07-18 + 2025 34-province merger)",
        "p": list(provinces.keys()),
        "d": list(districts.keys()),
        "w": list(wards.keys()),
        "e": [sorted([list(x) for x in ward_edges[i]]) for i in range(len(wards))],
        "dp": [sorted(list(district_provinces.get(i, set()))) for i in range(len(districts))],
        "pa": PROVINCE_ALIASES,
        "da": DISTRICT_ALIASES,
    }
    GAZETTEER_OUT.write_text(json.dumps(gazetteer, separators=(",", ":"), ensure_ascii=True), encoding="utf-8")

    full_records = {
        "license": "MIT",
        "legacy_3tier": legacy_records,
        "merged_2025_2tier": merged_2025_records,
    }
    RECORDS_OUT.write_text(json.dumps(full_records, ensure_ascii=False), encoding="utf-8")

    provenance = {
        "source_repository": "https://github.com/tranngocminhhieu/vietnamadminunits",
        "license": "MIT License (Copyright (c) 2025 Tran Ngoc Minh Hieu)",
        "raw_files": [
            {
                "filename": GSO_CSV.name,
                "url": "https://raw.githubusercontent.com/tranngocminhhieu/vietnamadminunits/main/data/raw/danhmuchanhchinh.gso.gov.vn_ward_2025-07-18.csv",
                "sha256": sha256_file(GSO_CSV),
                "rows": len(legacy_records),
            },
            {
                "filename": VN2025_CSV.name,
                "url": "https://raw.githubusercontent.com/tranngocminhhieu/vietnamadminunits/main/data/processed/2025_34-province-3221-ward_with_location.csv",
                "sha256": sha256_file(VN2025_CSV),
                "rows": len(merged_2025_records),
            },
            {
                "filename": LICENSE_FILE.name,
                "url": "https://raw.githubusercontent.com/tranngocminhhieu/vietnamadminunits/main/LICENSE",
                "sha256": sha256_file(LICENSE_FILE),
            },
        ],
        "counts": {
            "unique_provinces": len(provinces),
            "unique_districts": len(districts),
            "unique_wards": len(wards),
            "legacy_3tier_rows": len(legacy_records),
            "merged_2025_2tier_rows": len(merged_2025_records),
        },
        "gazetteer_sha256": sha256_file(GAZETTEER_OUT),
        "gazetteer_bytes": GAZETTEER_OUT.stat().st_size,
    }
    PROVENANCE_OUT.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    # Sanity check mandatory examples ("Lộc Thạnh, Lộc Ninh", "Hồng Dương, Thanh Oai")
    for w_check, d_check, p_check in [("loc thanh", "loc ninh", "binh phuoc"), ("hong duong", "thanh oai", "ha noi")]:
        wid = wards[w_check]
        did = districts[d_check]
        pid = provinces[p_check]
        assert [did, pid] in gazetteer["e"][wid], f"Missing edge for {w_check} -> {d_check} -> {p_check}"

    print(
        f"Built MIT VN Gazetteer: {len(provinces)} provinces, {len(districts)} districts, "
        f"{len(wards)} unique wards ({GAZETTEER_OUT.stat().st_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    build()
