"""Build MIT-licensed Laya v3.2 Administrative Converter, Spatial Centroids, & Full Records.

Permissive MIT-only sources ingested:
  1. tranngocminhhieu/vietnamadminunits (MIT License)
     - convert_legacy_2025_with_location_and_default_ward.csv (10,602 legacy <-> 2025 edges)
     - 2025_34-province-3221-ward_with_location.csv (3,321 post-2025 wards)
     - danhmuchanhchinh.gso.gov.vn_ward_2025-07-18.csv (10,035 GSO legacy wards)
     - alias_district.csv, alias_ward.csv, divided_district.csv (MIT alias bridges)
  2. thanglequoc/vietnamese-provinces-database (MIT License, v5.2.0 / 388/NQ-UBTVQH16)
     - 2026 National Assembly updates (3 city promotions + 22 Xa->Phuong upgrades)
     - Decision 2334/QD-BKHCN 5-digit ward postal codes & province postal prefixes
  3. madnh/hanhchinhvn (MIT License, v1.2 & v1.6.0)
     - Pre-2021 / pre-2025 historical districts & wards (Quan 2, Quan 9, Quan Thu Duc,
       Huyen Tan Thanh, Huyen Loc Ha, Phuong 6/7/8/10/13 Quan 3, etc.)
  4. Verified pre-2014 Huyen Tu Liem (Res. 132/NQ-CP) & pre-2021 GSO historical mappings

Outputs (strictly under shared_data_v3/):
  - vn_admin_v3_converter.json  (compact indexed legacy <-> 2025 + coordinates JSON, < 1.8 MB)
  - vn_admin_v3_records.json    (clean structured list for training & holdout synthesis)
  - mit_supplement_v32.json     (compact MIT delta supplement for standalone rebuilds)
  - PROVENANCE_V3.json          (exact URLs, licenses, SHA-256 hashes, counts)
  - DATA_DIFF_REPORT.json       (exact v3.0.0 -> v3.2.0 diff & hygiene report)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import unicodedata

VERSION = "3.2.0"

V3_DATA_DIR = Path(__file__).resolve().parent
SHARED_V2_DIR = V3_DATA_DIR.parent / "shared_data"

LICENSE_VAU = V3_DATA_DIR / "LICENSE_vietnamadminunits.txt"
LICENSE_TQ = V3_DATA_DIR / "LICENSE_thanglequoc_vietnamese_provinces_database.txt"
LICENSE_MADNH = V3_DATA_DIR / "LICENSE_madnh_hanhchinhvn.txt"

CONVERT_CSV = V3_DATA_DIR / "convert_legacy_2025_with_location_and_default_ward.csv"
VN2025_CSV = SHARED_V2_DIR / "2025_34-province-3221-ward_with_location.csv"
GSO_WARD_CSV = SHARED_V2_DIR / "danhmuchanhchinh.gso.gov.vn_ward_2025-07-18.csv"
SUPPLEMENT_FILE = V3_DATA_DIR / "mit_supplement_v32.json"

CONVERTER_OUT = V3_DATA_DIR / "vn_admin_v3_converter.json"
RECORDS_OUT = V3_DATA_DIR / "vn_admin_v3_records.json"
PROVENANCE_OUT = V3_DATA_DIR / "PROVENANCE_V3.json"
DIFF_REPORT_OUT = V3_DATA_DIR / "DATA_DIFF_REPORT.json"

MAX_CONVERTER_BYTES = 1_800_000  # Strict < 1.8 MB budget

# Baseline v3.0.0 metrics for exact diff reporting
BASELINE_V300: Dict[str, Any] = {
    "version": "3.0.0",
    "converter_sha256": "42201268ecfcd30b2f31bbb65d1efe6454790e42b852936450d4b52f6b0b0f46",
    "converter_bytes": 1100790,
    "counts": {
        "legacy_provinces": 63,
        "legacy_districts": 661,
        "legacy_province_district_pairs": 690,
        "legacy_wards": 6433,
        "new_provinces_2025": 34,
        "new_wards_2025": 2745,
        "new_ward_entries_2025": 3321,
        "island_special_zones": 5,
        "total_mapping_records": 10602,
    },
}

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
    "tp dong nai": "dong nai",
    "tp bac ninh": "bac ninh",
    "tp quang ninh": "quang ninh",
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
    "q thu duc": "thu duc",
    "quan thu duc": "thu duc",
    "phu quy": "phu qui",
    "qui nhon": "quy nhon",
    "phan rang-thap cham": "phan rang - thap cham",
    "phan rang thap cham": "phan rang - thap cham",
    "ia h' drai": "ia h'drai",
    "ia hdrai": "ia h'drai",
}

WARD_TYPES = ["Phường", "Xã", "Thị trấn", "Đặc khu"]
WARD_TYPE_TO_IDX = {w: i for i, w in enumerate(WARD_TYPES)}

# Verified pre-2021 GSO/MIT mappings for the ~20 residual wards in madnh v1.2 that merged prior to alias_ward.csv
_PRE2021_FALLBACK_RESOLUTIONS: Dict[Tuple[str, str, str], Tuple[str, str, str]] = {
    # (province_full, district_full, ward_full) -> (conv_province, conv_district, conv_ward)
    ("Tỉnh Cao Bằng", "Huyện Hà Quảng", "Xã Nà Sác"): ("Tỉnh Cao Bằng", "Huyện Hà Quảng", "Xã Sóc Hà"),
    ("Tỉnh Cao Bằng", "Huyện Hà Quảng", "Xã Hồng Sĩ"): ("Tỉnh Cao Bằng", "Huyện Hà Quảng", "Xã Hồng Sỹ"),
    ("Tỉnh Cao Bằng", "Huyện Trà Lĩnh", "Xã Quang Vinh"): ("Tỉnh Cao Bằng", "Huyện Trùng Khánh", "Xã Quang Vinh"),
    ("Tỉnh Lào Cai", "Huyện Sa Pa", "Xã Tả Giàng Phình"): ("Tỉnh Lào Cai", "Thị xã Sa Pa", "Xã Ngũ Chỉ Sơn"),
    ("Tỉnh Lai Châu", "Huyện Tam Đường", "Xã Sùng Phài"): ("Tỉnh Lai Châu", "Thành phố Lai Châu", "Xã Sùng Phài"),
    ("Tỉnh Hải Dương", "Huyện Kinh Môn", "Xã Phúc Thành B"): ("Tỉnh Hải Dương", "Thị xã Kinh Môn", "Xã Quang Thành"),
    ("Tỉnh Hải Dương", "Huyện Kim Thành", "Xã Phúc Thành A"): ("Tỉnh Hải Dương", "Huyện Kim Thành", "Thị trấn Phú Thái"),
    ("Tỉnh Hà Nam", "Huyện Duy Tiên", "Xã Mộc Nam"): ("Tỉnh Hà Nam", "Thị xã Duy Tiên", "Xã Mộc Hoàn"),
    ("Tỉnh Hà Nam", "Thị xã Duy Tiên", "Xã Mộc Nam"): ("Tỉnh Hà Nam", "Thị xã Duy Tiên", "Xã Mộc Hoàn"),
    ("Tỉnh Nam Định", "Huyện Nghĩa Hưng", "Xã Nghĩa Phúc"): ("Tỉnh Nam Định", "Huyện Nghĩa Hưng", "Xã Phúc Thắng"),
    ("Thành phố Đà Nẵng", "Quận Hải Châu", "Phường Hải Châu II"): ("Thành phố Đà Nẵng", "Quận Hải Châu", "Phường Hải Châu"),
    ("Tỉnh Quảng Ngãi", "Huyện Bình Sơn", "Xã Bình Thanh Đông"): ("Tỉnh Quảng Ngãi", "Huyện Bình Sơn", "Xã Bình Thanh"),
    ("Tỉnh Quảng Ngãi", "Huyện Lý Sơn", "Xã An Vĩnh"): ("Tỉnh Quảng Ngãi", "Huyện Lý Sơn", ""),
    ("Tỉnh Quảng Ngãi", "Huyện Lý Sơn", "Xã An Hải"): ("Tỉnh Quảng Ngãi", "Huyện Lý Sơn", ""),
    ("Tỉnh Quảng Ngãi", "Huyện Lý Sơn", "Xã An Bình"): ("Tỉnh Quảng Ngãi", "Huyện Lý Sơn", ""),
    ("Tỉnh Gia Lai", "Huyện Chư Păh", "Xã Chư Jôr"): ("Tỉnh Gia Lai", "Huyện Chư Păh", "Xã Chư Đang Ya"),
    ("Tỉnh Đồng Nai", "Huyện Long Thành", "Xã Suối Trầu"): ("Tỉnh Đồng Nai", "Huyện Long Thành", "Xã Bình Sơn"),
    ("Tỉnh Bến Tre", "Huyện Châu Thành", "Xã Phú An Hòa"): ("Tỉnh Bến Tre", "Huyện Châu Thành", "Xã An Phước"),
    ("Tỉnh Bến Tre", "Huyện Châu Thành", "Xã An Hóa"): ("Tỉnh Bến Tre", "Huyện Châu Thành", "Xã An Phước"),
}

# Verified pre-2014 Huyen Tu Liem (Ha Noi, Resolution 132/NQ-CP) -> successor wards in Nam/Bac Tu Liem
_PRE2014_TU_LIEM_MAPPINGS: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("Thị trấn Cầu Diễn", [("Quận Nam Từ Liêm", "Phường Cầu Diễn")]),
    ("Xã Mỹ Đình", [("Quận Nam Từ Liêm", "Phường Mỹ Đình 1"), ("Quận Nam Từ Liêm", "Phường Mỹ Đình 2")]),
    ("Xã Mễ Trì", [("Quận Nam Từ Liêm", "Phường Mễ Trì")]),
    ("Xã Trung Văn", [("Quận Nam Từ Liêm", "Phường Trung Văn")]),
    ("Xã Đại Mỗ", [("Quận Nam Từ Liêm", "Phường Đại Mỗ")]),
    ("Xã Tây Mỗ", [("Quận Nam Từ Liêm", "Phường Tây Mỗ")]),
    ("Xã Xuân Phương", [("Quận Nam Từ Liêm", "Phường Xuân Phương"), ("Quận Nam Từ Liêm", "Phường Phương Canh")]),
    ("Xã Phú Diễn", [("Quận Bắc Từ Liêm", "Phường Phú Diễn"), ("Quận Bắc Từ Liêm", "Phường Phúc Diễn")]),
    ("Xã Minh Khai", [("Quận Bắc Từ Liêm", "Phường Minh Khai")]),
    ("Xã Tây Tựu", [("Quận Bắc Từ Liêm", "Phường Tây Tựu")]),
    ("Xã Thượng Cát", [("Quận Bắc Từ Liêm", "Phường Thượng Cát")]),
    ("Xã Liên Mạc", [("Quận Bắc Từ Liêm", "Phường Liên Mạc")]),
    ("Xã Thụy Phương", [("Quận Bắc Từ Liêm", "Phường Thụy Phương")]),
    ("Xã Đông Ngạc", [("Quận Bắc Từ Liêm", "Phường Đông Ngạc"), ("Quận Bắc Từ Liêm", "Phường Đức Thắng")]),
    ("Xã Xuân Đỉnh", [("Quận Bắc Từ Liêm", "Phường Xuân Đỉnh"), ("Quận Bắc Từ Liêm", "Phường Xuân Tảo")]),
    ("Xã Cổ Nhuế", [("Quận Bắc Từ Liêm", "Phường Cổ Nhuế 1"), ("Quận Bắc Từ Liêm", "Phường Cổ Nhuế 2")]),
]

# All 12 district-level island special zones (plus Xa Tho Chau -> Dac khu Tho Chau via eo = 13 total dac khu)
_DISTRICT_LEVEL_DAC_KHU: List[Tuple[str, str, str, str]] = [
    ("huyen bach long vi", "hai phong", "bach long vi", "hai phong"),
    ("huyen con co", "quang tri", "con co", "quang tri"),
    ("huyen hoang sa", "da nang", "hoang sa", "da nang"),
    ("huyen ly son", "quang ngai", "ly son", "quang ngai"),
    ("huyen con dao", "ba ria - vung tau", "con dao", "ho chi minh"),
    ("huyen van don", "quang ninh", "van don", "quang ninh"),
    ("huyen co to", "quang ninh", "co to", "quang ninh"),
    ("huyen cat hai", "hai phong", "cat hai", "hai phong"),
    ("huyen truong sa", "khanh hoa", "truong sa", "khanh hoa"),
    ("huyen phu qui", "binh thuan", "phu quy", "lam dong"),
    ("huyen kien hai", "kien giang", "kien hai", "an giang"),
    ("thanh pho phu quoc", "kien giang", "phu quoc", "an giang"),
]


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


def compact_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", strip_accents(text or "").lower())


def canonical_district_raw(raw: str) -> str:
    s = re.sub(r"\s+", " ", unicodedata.normalize("NFC", raw or "").strip())
    s = s.replace("Phan Rang-Tháp Chàm", "Phan Rang - Tháp Chàm")
    s = s.replace("Ia H' Drai", "Ia H'Drai")
    s = s.replace("Qui Nhơn", "Quy Nhơn")
    return s


def strip_admin_prefix(raw: str, prefixes: Tuple[str, ...], keep_if_numeric_prefix: str = "") -> str:
    s = norm_ascii(raw)
    for p in prefixes:
        if s.startswith(p + " "):
            rest = s[len(p) + 1 :].strip()
            if not rest:
                return s
            if rest.isdigit() and keep_if_numeric_prefix:
                return f"{keep_if_numeric_prefix} {int(rest)}"
            return rest
    return s


def strip_display_prefix(raw: str, prefixes: Tuple[str, ...]) -> str:
    s = re.sub(r"\s+", " ", unicodedata.normalize("NFC", raw or "").strip())
    low = strip_accents(s).lower()
    for p in prefixes:
        if low.startswith(p + " "):
            rest = s[len(p) + 1 :].strip()
            if rest:
                if rest.isdigit():
                    return f"{s[: len(p)]} {int(rest)}"
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


def _wt_idx(w_type: str, full_name: str, override_wt: Optional[str] = None) -> int:
    if override_wt and override_wt in WARD_TYPE_TO_IDX:
        return WARD_TYPE_TO_IDX[override_wt]
    low = norm_ascii(full_name)
    if low.startswith("phuong "):
        return 0
    if low.startswith("xa "):
        return 1
    if low.startswith("thi tran "):
        return 2
    if low.startswith("dac khu "):
        return 3
    t = (w_type or "").strip()
    if t in WARD_TYPE_TO_IDX:
        return WARD_TYPE_TO_IDX[t]
    return 1


def _norm_pk_compact(pk: str) -> str:
    return "thanhphohue" if pk == "tinhthuathienhue" else pk


def _norm_p_bare(pb: str) -> str:
    return "hue" if pb == "thua thien hue" else pb


def _extract_supplement_from_sources(sources_dir: Path) -> Dict[str, Any]:
    """Extract deterministic MIT supplement from raw crawled/fetched MIT sources."""
    leg_dir = sources_dir / "legacy"
    new_dir = sources_dir / "new_2025"

    vau_leg = leg_dir / "vietnamadminunits"
    if not vau_leg.exists():
        vau_leg = sources_dir / "vietnamadminunits"
    madnh_dir = leg_dir / "madnh_hanhchinhvn"
    if not madnh_dir.exists():
        madnh_dir = sources_dir / "madnh_hanhchinhvn"
    tq_new = new_dir / "thanglequoc"
    if not tq_new.exists():
        tq_new = sources_dir / "thanglequoc"

    # 1. Load 2025 thanglequoc v5.2.0 MIT data (2026 decrees + postal codes)
    fj_path = tq_new / "full_json_generated_data_vn_units.json"
    if not fj_path.exists():
        raise FileNotFoundError(f"Missing MIT thanglequoc 2025 JSON: {fj_path}")
    fj = json.loads(fj_path.read_text(encoding="utf-8"))
    meta_path = tq_new / "vn_provinces_metadata.json"
    tq_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    prov_2026_full: Dict[str, str] = {}
    prov_postal_prefix: Dict[str, str] = {}
    ward_2026_wt: Dict[str, str] = {}
    ward_postal_codes: Dict[str, str] = {}

    for p in fj:
        pcode = str(int(p["Code"]))
        prov_2026_full[pcode] = p["FullName"].strip()
        prov_postal_prefix[pcode] = (p.get("PostalCodePrefix") or "").strip()
        for w in p["Wards"]:
            wcode = str(int(w["Code"]))
            wt = (w.get("AdministrativeUnitShortName") or "").strip()
            if wt in WARD_TYPE_TO_IDX:
                ward_2026_wt[wcode] = wt
            pc = (w.get("PostalCode") or "").strip()
            if pc:
                ward_postal_codes[wcode] = pc

    # 2. Load convert_legacy_2025 CSV
    conv_csv = vau_leg / "convert_legacy_2025_with_location_and_default_ward.csv"
    if not conv_csv.exists():
        conv_csv = CONVERT_CSV
    with conv_csv.open("r", encoding="utf-8-sig", newline="") as f:
        conv_rows = list(csv.DictReader(f))

    conv_by_pdw: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    conv_by_bare_pdw: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    conv_by_wcode: Dict[str, List[Dict[str, str]]] = {}
    conv_by_pd: Dict[Tuple[str, str], List[Dict[str, str]]] = {}

    for r in conv_rows:
        pk = _norm_pk_compact(compact_key(r["province"]))
        dk = compact_key(canonical_district_raw(r["district"]))
        wk = compact_key(r["ward"])
        conv_by_pdw.setdefault((pk, dk, wk), []).append(r)
        conv_by_pd.setdefault((pk, dk), []).append(r)
        p_b = _norm_p_bare(strip_admin_prefix(r["province"], ("tinh", "thanh pho")))
        d_b = strip_admin_prefix(canonical_district_raw(r["district"]), ("quan", "huyen", "thi xa", "thanh pho"), "quan")
        w_b = strip_admin_prefix(r["ward"], ("phuong", "xa", "thi tran", "dac khu"), "phuong")
        conv_by_bare_pdw.setdefault((p_b, d_b, w_b), []).append(r)
        wc = (r.get("wardCode") or "").strip()
        if wc:
            conv_by_wcode.setdefault(str(int(float(wc))).zfill(5), []).append(r)

    # 3. Load alias_district.csv and alias_ward.csv (MIT, cleaning tabs & accents)
    ad_lookup: Dict[Tuple[str, str], Set[str]] = {}
    ad_csv = vau_leg / "alias_district.csv"
    if ad_csv.exists():
        with ad_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                pk = _norm_pk_compact(compact_key(r["province_key"]))
                dk = compact_key(r["district_key"])
                ak = compact_key(r["alias_keyword"])
                if pk and dk and ak:
                    ad_lookup.setdefault((pk, ak), set()).add(dk)

    aw_lookup: Dict[Tuple[str, str, str], Set[Tuple[str, str]]] = {}
    aw_prov_lookup: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
    aw_hygiene_cleaned = 0
    aw_csv = vau_leg / "alias_ward.csv"
    if aw_csv.exists():
        with aw_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                raw_wk = r["ward_key"]
                raw_ak = r["alias_keyword"]
                if "\t" in raw_wk or "\t" in raw_ak or not raw_wk.isascii() or not raw_ak.isascii():
                    aw_hygiene_cleaned += 1
                pk = _norm_pk_compact(compact_key(r["province_key"]))
                dk = compact_key(r["district_key"])
                wk = compact_key(raw_wk)
                ak = compact_key(raw_ak)
                if pk and dk and wk and ak:
                    aw_lookup.setdefault((pk, dk, ak), set()).add((dk, wk))
                    aw_prov_lookup.setdefault((pk, ak), set()).add((dk, wk))

    # 4. Load optional legacy ward locations (MIT)
    legacy_loc_by_code: Dict[str, Tuple[float, float]] = {}
    leg_loc_csv = vau_leg / "legacy_63-province-10040-ward_with_location.csv"
    if leg_loc_csv.exists():
        with leg_loc_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                wc = (r.get("wardCode") or "").strip()
                lat = _safe_float(r.get("wardLat"))
                lon = _safe_float(r.get("wardLon"))
                if wc and lat != 0.0 and lon != 0.0:
                    legacy_loc_by_code[str(int(float(wc))).zfill(5)] = (round(lat, 4), round(lon, 4))

    def _rows_to_targets(rows: List[Dict[str, str]], src_tag: str) -> List[List[Any]]:
        """Compact target encoding: [newProvinceCode, newWardCode, isDivided, isDefault, wardLat, wardLon]."""
        out: List[List[Any]] = []
        seen_t: Set[Tuple[int, int]] = set()
        for r in rows:
            p_code = _safe_int(r.get("newProvinceCode"))
            w_code = _safe_int(r.get("newWardCode"))
            key_t = (p_code, w_code)
            if key_t in seen_t:
                continue
            seen_t.add(key_t)
            out.append([
                p_code,
                w_code,
                1 if (r.get("isDividedWard") or "").strip() == "True" else 0,
                1 if (r.get("isDefaultNewWard") or "").strip() == "True" else 0,
                round(_safe_float(r.get("wardLat")), 4),
                round(_safe_float(r.get("wardLon")), 4),
            ])
        return out

    def _resolve_triple(p_raw: str, d_raw: str, w_raw: str, wc5: str = "") -> Tuple[List[List[Any]], str]:
        p_raw = p_raw.strip()
        d_raw = canonical_district_raw(d_raw)
        w_raw = re.sub(r"\s+", " ", unicodedata.normalize("NFC", w_raw or "").strip())
        # Normalize leading zeros on numbered wards (e.g., Phuong 06 -> Phuong 6)
        w_disp_norm = strip_display_prefix(w_raw, ("phuong", "xa", "thi tran"))
        if w_disp_norm != w_raw and re.match(r"^(?:Phường|Xã|Thị trấn)\s+0+\d+$", w_raw, re.IGNORECASE):
            w_raw = w_disp_norm

        pk = _norm_pk_compact(compact_key(p_raw))
        dk = compact_key(d_raw)
        wk = compact_key(w_raw)

        # Direct exact match
        if (pk, dk, wk) in conv_by_pdw:
            return _rows_to_targets(conv_by_pdw[(pk, dk, wk)], "conv_exact"), "conv_exact"

        # Bare match (e.g., Thi xa Go Cong -> TP Go Cong, Xa -> Phuong)
        p_b = _norm_p_bare(strip_admin_prefix(p_raw, ("tinh", "thanh pho")))
        d_b = strip_admin_prefix(d_raw, ("quan", "huyen", "thi xa", "thanh pho"), "quan")
        w_b = strip_admin_prefix(w_raw, ("phuong", "xa", "thi tran"), "phuong")
        if (p_b, d_b, w_b) in conv_by_bare_pdw:
            return _rows_to_targets(conv_by_bare_pdw[(p_b, d_b, w_b)], "conv_bare"), "conv_bare"

        # GSO wardCode match
        if wc5 and wc5 in conv_by_wcode:
            return _rows_to_targets(conv_by_wcode[wc5], "gso_code"), "gso_code"

        # alias_ward under same (pk, dk)
        if (pk, dk, wk) in aw_lookup:
            for tdk, twk in sorted(aw_lookup[(pk, dk, wk)]):
                if (pk, tdk, twk) in conv_by_pdw:
                    return _rows_to_targets(conv_by_pdw[(pk, tdk, twk)], "alias_ward"), "alias_ward"

        # alias_district -> direct or alias_ward
        for tdk in sorted(ad_lookup.get((pk, dk), set())):
            if (pk, tdk, wk) in conv_by_pdw:
                return _rows_to_targets(conv_by_pdw[(pk, tdk, wk)], "alias_district"), "alias_district"
            if (pk, tdk, wk) in aw_lookup:
                for xdk, xwk in sorted(aw_lookup[(pk, tdk, wk)]):
                    if (pk, xdk, xwk) in conv_by_pdw:
                        return _rows_to_targets(conv_by_pdw[(pk, xdk, xwk)], "alias_district_ward"), "alias_district_ward"

        # Province-scoped alias_ward (for wards transferred across districts like Xuan Tho, Loc Ha -> Thach Ha, etc.)
        for xdk, xwk in sorted(aw_prov_lookup.get((pk, wk), set())):
            if (pk, xdk, xwk) in conv_by_pdw:
                return _rows_to_targets(conv_by_pdw[(pk, xdk, xwk)], "alias_ward_prov"), "alias_ward_prov"

        # Verified pre-2021 fallback
        fb = _PRE2021_FALLBACK_RESOLUTIONS.get((p_raw, d_raw, w_raw))
        if fb is not None:
            fpk = _norm_pk_compact(compact_key(fb[0]))
            fdk = compact_key(fb[1])
            fwk = compact_key(fb[2])
            if fwk == "" and (fpk, fdk) in conv_by_pd:
                return _rows_to_targets(conv_by_pd[(fpk, fdk)], "pre2021_verified"), "pre2021_verified"
            if (fpk, fdk, fwk) in conv_by_pdw:
                return _rows_to_targets(conv_by_pdw[(fpk, fdk, fwk)], "pre2021_verified"), "pre2021_verified"

        return [], "unresolved"

    # Collect unique historical edges NOT already present under normalized (po, do, wo) in conv_rows
    existing_norm_triples: Set[Tuple[str, str, str]] = set()
    for r in conv_rows:
        p_n = strip_admin_prefix(r["province"], ("tinh", "thanh pho"))
        d_n = strip_admin_prefix(canonical_district_raw(r["district"]), ("quan", "huyen", "thi xa", "thanh pho"), "quan")
        w_n = strip_admin_prefix(r["ward"], ("phuong", "xa", "thi tran", "dac khu"), "phuong")
        if w_n:
            existing_norm_triples.add((p_n, d_n, w_n))

    added_edges: List[List[Any]] = []
    added_seen: Set[Tuple[str, str, str]] = set()
    method_counts: Dict[str, int] = {}

    def _maybe_add(p_raw: str, d_raw: str, w_raw: str, wc5: str, source_tag: str) -> None:
        d_clean = canonical_district_raw(d_raw)
        w_clean = re.sub(r"\s+", " ", unicodedata.normalize("NFC", w_raw or "").strip())
        if re.match(r"^(?:Phường|Xã|Thị trấn)\s+0+\d+$", w_clean, re.IGNORECASE):
            w_clean = strip_display_prefix(w_clean, ("phuong", "xa", "thi tran"))
        p_n = strip_admin_prefix(p_raw, ("tinh", "thanh pho"))
        d_n = strip_admin_prefix(d_clean, ("quan", "huyen", "thi xa", "thanh pho"), "quan")
        w_n = strip_admin_prefix(w_clean, ("phuong", "xa", "thi tran"), "phuong")
        if not (p_n and d_n and w_n):
            return
        triple = (p_n, d_n, w_n)
        if triple in existing_norm_triples or triple in added_seen:
            return
        targets, _how = _resolve_triple(p_raw, d_clean, w_clean, wc5=wc5)
        if not targets:
            return
        added_seen.add(triple)
        method_counts[source_tag] = method_counts.get(source_tag, 0) + 1
        loc = legacy_loc_by_code.get(wc5)
        if loc:
            for t in targets:
                if t[4] == 0.0:
                    t[4], t[5] = loc
        added_edges.append([p_raw.strip(), d_clean, w_clean, targets])

    # 5A. GSO 2025-07-18 variants (75 ward spelling variants + 2 district punctuation variants)
    gso_csv = vau_leg / "danhmuchanhchinh.gso.gov.vn_ward_2025-07-18.csv"
    if not gso_csv.exists():
        gso_csv = GSO_WARD_CSV
    with gso_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            wc5 = (r.get("Mã") or "").strip().zfill(5)
            w_raw = (r.get("Tên") or "").strip()
            d_raw = (r.get("Quận Huyện") or "").strip()
            p_raw = (r.get("Tỉnh / Thành Phố") or "").strip()
            _maybe_add(p_raw, d_raw, w_raw, wc5, "gso_2025_07_18")

    # 5B. madnh master (v1.6.0) then v1.2 (pre-2021)
    for tag, prefix in (("madnh_v1_6_0", "master"), ("madnh_v1_2", "v1.2")):
        mp_path = madnh_dir / f"{prefix}_tinh_tp.json"
        md_path = madnh_dir / f"{prefix}_quan_huyen.json"
        mw_path = madnh_dir / f"{prefix}_xa_phuong.json"
        if not (mp_path.exists() and md_path.exists() and mw_path.exists()):
            continue
        mp = json.loads(mp_path.read_text(encoding="utf-8"))
        md = json.loads(md_path.read_text(encoding="utf-8"))
        mw = json.loads(mw_path.read_text(encoding="utf-8"))
        for wc, wobj in sorted(mw.items()):
            wc5 = str(wc).zfill(5)
            dobj = md[wobj["parent_code"]]
            pobj = mp[dobj["parent_code"]]
            _maybe_add(pobj["name_with_type"], dobj["name_with_type"], wobj["name_with_type"], wc5, tag)

    # 5C. Pre-2014 Huyen Tu Liem (Ha Noi)
    for tl_ward, succ_list in _PRE2014_TU_LIEM_MAPPINGS:
        combined_rows: List[Dict[str, str]] = []
        for s_dist, s_ward in succ_list:
            pk = compact_key("Thành phố Hà Nội")
            dk = compact_key(s_dist)
            wk = compact_key(s_ward)
            combined_rows.extend(conv_by_pdw.get((pk, dk, wk), ()))
        if combined_rows:
            t_list = _rows_to_targets(combined_rows, "pre2014_tu_liem")
            if len(t_list) > 1:
                has_def = any(t[3] == 1 for t in t_list)
                for i, t in enumerate(t_list):
                    t[2] = 1
                    if not has_def:
                        t[3] = 1 if i == 0 else 0
            added_edges.append(["Thành phố Hà Nội", "Huyện Từ Liêm", tl_ward, t_list])
            method_counts["pre2014_tu_liem"] = method_counts.get("pre2014_tu_liem", 0) + 1

    # Raw MIT file SHA-256 provenance
    raw_mit_files: List[Dict[str, Any]] = []
    for label, fpath, url in [
        ("thanglequoc_full_2026_json", fj_path, "https://raw.githubusercontent.com/thanglequoc/vietnamese-provinces-database/master/json/full_json_generated_data_vn_units.json"),
        ("thanglequoc_license", tq_new / "LICENSE", "https://raw.githubusercontent.com/thanglequoc/vietnamese-provinces-database/master/LICENSE"),
        ("vau_alias_district_csv", ad_csv, "https://raw.githubusercontent.com/tranngocminhhieu/vietnamadminunits/main/data/alias_keywords/legacy/alias_district.csv"),
        ("vau_alias_ward_csv", aw_csv, "https://raw.githubusercontent.com/tranngocminhhieu/vietnamadminunits/main/data/alias_keywords/legacy/alias_ward.csv"),
        ("madnh_v1_2_districts", madnh_dir / "v1.2_quan_huyen.json", "https://raw.githubusercontent.com/madnh/hanhchinhvn/v1.2/dist/quan_huyen.json"),
        ("madnh_v1_2_wards", madnh_dir / "v1.2_xa_phuong.json", "https://raw.githubusercontent.com/madnh/hanhchinhvn/v1.2/dist/xa_phuong.json"),
        ("madnh_v1_6_0_districts", madnh_dir / "master_quan_huyen.json", "https://raw.githubusercontent.com/madnh/hanhchinhvn/master/dist/quan_huyen.json"),
        ("madnh_v1_6_0_wards", madnh_dir / "master_xa_phuong.json", "https://raw.githubusercontent.com/madnh/hanhchinhvn/master/dist/xa_phuong.json"),
    ]:
        if fpath.exists():
            raw_mit_files.append({
                "id": label,
                "filename": fpath.name,
                "url": url,
                "sha256": sha256_file(fpath),
            })

    return {
        "schema_version": VERSION,
        "thanglequoc_metadata": tq_meta,
        "prov_2026_full": prov_2026_full,
        "prov_postal_prefix": prov_postal_prefix,
        "ward_2026_wt": ward_2026_wt,
        "ward_postal_codes": ward_postal_codes,
        "added_historical_edges": added_edges,
        "extraction_stats": {
            "method_counts": method_counts,
            "alias_ward_hygiene_cleaned_rows": aw_hygiene_cleaned,
        },
        "mit_source_hashes": raw_mit_files,
    }


def build(sources_dir: Optional[Path] = None) -> Dict[str, Any]:
    if not CONVERT_CSV.exists():
        raise FileNotFoundError(f"Missing conversion CSV: {CONVERT_CSV}")
    if not VN2025_CSV.exists():
        raise FileNotFoundError(f"Missing 2025 CSV: {VN2025_CSV}")

    if sources_dir is not None and sources_dir.exists():
        supplement = _extract_supplement_from_sources(sources_dir)
        SUPPLEMENT_FILE.write_text(
            json.dumps(supplement, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
    elif SUPPLEMENT_FILE.exists():
        supplement = json.loads(SUPPLEMENT_FILE.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(
            f"Neither --sources-dir nor {SUPPLEMENT_FILE} is available to build v3.2 assets."
        )

    prov_2026_full: Dict[str, str] = supplement["prov_2026_full"]
    prov_postal_prefix: Dict[str, str] = supplement["prov_postal_prefix"]
    ward_2026_wt: Dict[str, str] = supplement["ward_2026_wt"]
    ward_postal_codes: Dict[str, str] = supplement["ward_postal_codes"]
    added_historical_edges: List[Dict[str, Any]] = supplement["added_historical_edges"]

    po_map: Dict[str, int] = {}
    po_disp: List[str] = []
    do_map: Dict[str, int] = {}
    do_disp: List[str] = []
    wo_map: Dict[str, int] = {}
    wo_disp: List[str] = []

    pn_map: Dict[str, int] = {}
    pn_disp: List[str] = []
    pn_full_disp: List[str] = []
    pn_code: List[int] = []
    p_postal: List[str] = []
    wn_map: Dict[str, int] = {}
    wn_disp: List[str] = []
    wn_code_disp: Dict[str, str] = {}
    w_postal: Dict[str, str] = {}

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
    wt_updates_applied: List[Dict[str, Any]] = []

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

    def get_pn(norm: str, disp: str, pcode: int, raw_full: str = "") -> int:
        if norm not in pn_map:
            pn_map[norm] = len(pn_disp)
            pn_disp.append(disp)
            pn_code.append(pcode)
            p_str = str(pcode)
            pn_full_disp.append(prov_2026_full.get(p_str, raw_full or disp))
            p_postal.append(prov_postal_prefix.get(p_str, ""))
        return pn_map[norm]

    def get_wn(norm: str, disp: str, wcode: int = 0) -> int:
        if norm not in wn_map:
            wn_map[norm] = len(wn_disp)
            wn_disp.append(disp)
        wid_n = wn_map[norm]
        if wcode and disp != wn_disp[wid_n]:
            wn_code_disp[str(wcode)] = disp
        return wid_n

    # 1. Seed 2025/2026 34 provinces & 3,321 wards with updated 2026 wardTypes & postal codes
    vn25_by_wcode: Dict[int, Dict[str, Any]] = {}
    with VN2025_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pn_raw = (row.get("province") or "").strip()
            wn_raw = (row.get("ward") or "").strip()
            if not (pn_raw and wn_raw):
                continue
            pn_short = (row.get("provinceShort") or "").strip() or strip_display_prefix(pn_raw, ("tinh", "thanh pho"))
            wn_short = (row.get("wardShort") or "").strip() or strip_display_prefix(wn_raw, ("phuong", "xa", "thi tran", "dac khu"))
            wt_csv = (row.get("wardType") or "").strip()
            w_code = _safe_int(row.get("wardCode"))
            p_code = _safe_int(row.get("provinceCode"))
            override_wt = ward_2026_wt.get(str(w_code))

            wt_idx_csv_old = WARD_TYPE_TO_IDX.get(wt_csv, _wt_idx("", wn_raw))
            wt_idx_new = _wt_idx(wt_csv, wn_raw, override_wt=override_wt)
            if wt_idx_new != wt_idx_csv_old:
                wt_updates_applied.append({
                    "ward_code": str(w_code).zfill(5),
                    "province": pn_short,
                    "ward_short": wn_short,
                    "old_wt": WARD_TYPES[wt_idx_csv_old],
                    "new_wt": WARD_TYPES[wt_idx_new],
                })

            pn_norm = strip_admin_prefix(pn_raw, ("tinh", "thanh pho"))
            wn_norm = strip_admin_prefix(wn_raw, ("phuong", "xa", "thi tran", "dac khu"), keep_if_numeric_prefix="phuong")

            pid_n = get_pn(pn_norm, pn_short, p_code, raw_full=pn_raw)
            wid_n = get_wn(wn_norm, wn_short, wcode=w_code)
            if str(w_code) in ward_postal_codes:
                w_postal[str(w_code)] = ward_postal_codes[str(w_code)]

            nw_lat = round(_safe_float(row.get("wardLat")), 4)
            nw_lon = round(_safe_float(row.get("wardLon")), 4)
            nw_area = round(_safe_float(row.get("wardAreaKm2"), 15.0), 2)
            vn25_by_wcode[w_code] = {
                "pid_n": pid_n,
                "wid_n": wid_n,
                "pn_raw": pn_raw,
                "pn_short": pn_short,
                "wn_raw": f"{WARD_TYPES[wt_idx_new]} {wn_short}",
                "wn_short": wn_short,
                "lat": nw_lat,
                "lon": nw_lon,
                "area": nw_area,
            }

            if (pid_n, w_code) not in en_seen.setdefault(wid_n, set()):
                en_seen[wid_n].add((pid_n, w_code))
                en_entries.setdefault(wid_n, []).append([
                    pid_n,
                    nw_lat,
                    nw_lon,
                    nw_area,
                    w_code,
                    wt_idx_new,
                ])

    # 2. Parse 10,602 base legacy <-> 2025 conversion edges
    total_conv_rows = 0
    with CONVERT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            total_conv_rows += 1
            po_raw = (row.get("province") or "").strip()
            do_raw = canonical_district_raw((row.get("district") or "").strip())
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
            pid_n = get_pn(pn_norm, pn_short, _safe_int(row.get("newProvinceCode")), raw_full=pn_raw)
            w_code_n = _safe_int(row.get("newWardCode"))
            wid_n = get_wn(wn_norm, wn_short, wcode=w_code_n)

            pm_edges[pid_o] = pid_n
            dp_edges.setdefault(did_o, set()).add(pid_o)

            if (pid_n, w_code_n) not in en_seen.setdefault(wid_n, set()):
                en_seen[wid_n].add((pid_n, w_code_n))
                override_wt = ward_2026_wt.get(str(w_code_n))
                en_entries.setdefault(wid_n, []).append([
                    pid_n,
                    round(_safe_float(row.get("newWardLat")), 4),
                    round(_safe_float(row.get("newWardLon")), 4),
                    round(_safe_float(row.get("newWardAreaKm2"), 15.0), 2),
                    w_code_n,
                    _wt_idx(row.get("newWardType") or "", wn_raw, override_wt=override_wt),
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

    # 3. Integrate MIT historical & GSO edges (sorted deterministically)
    historical_districts_added: List[str] = []
    base_do_count = len(do_disp)
    base_dp_pairs = sum(len(s) for s in dp_edges.values())
    base_wo_count = len(wo_disp)
    base_records_count = len(records)

    for po_raw, do_raw_in, wo_raw, targets in added_historical_edges:
        do_raw = canonical_district_raw(do_raw_in)
        if not targets:
            continue

        po_norm = strip_admin_prefix(po_raw, ("tinh", "thanh pho"))
        if po_norm == "thua thien hue":
            po_norm = "hue"
        do_norm = strip_admin_prefix(do_raw, ("quan", "huyen", "thi xa", "thanh pho"), keep_if_numeric_prefix="quan")
        wo_norm = strip_admin_prefix(wo_raw, ("phuong", "xa", "thi tran", "dac khu"), keep_if_numeric_prefix="phuong")

        po_short = strip_display_prefix(po_raw, ("tinh", "thanh pho"))
        do_short = strip_display_prefix(do_raw, ("quan", "huyen", "thi xa", "thanh pho"))
        wo_short = strip_display_prefix(wo_raw, ("phuong", "xa", "thi tran"))

        pid_o = get_po(po_norm, po_short)
        is_new_do = do_norm not in do_map
        did_o = get_do(do_norm, do_short)
        if is_new_do:
            historical_districts_added.append(do_raw)
        wid_o = get_wo(wo_norm, wo_short)

        dp_edges.setdefault(did_o, set()).add(pid_o)

        for _p_code_n, w_code_n, t_div, t_def, w_lat, w_lon in targets:
            meta25 = vn25_by_wcode[w_code_n]
            pid_n = meta25["pid_n"]
            wid_n = meta25["wid_n"]
            pn_raw = meta25["pn_raw"]
            pn_short = meta25["pn_short"]
            wn_raw = meta25["wn_raw"]
            wn_short = meta25["wn_short"]
            nw_lat = meta25["lat"]
            nw_lon = meta25["lon"]

            is_div = 1 if (t_div or len(targets) > 1) else 0
            is_def = 1 if (t_def or len(targets) == 1) else 0

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
                ogeo.setdefault(f"{wid_o}:{did_o}:{pid_o}", [round(float(w_lat), 4), round(float(w_lon), 4)])

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
                "new_lat": nw_lat,
                "new_lon": nw_lon,
                "old_lat": round(float(w_lat), 4) if w_lat != 0.0 else nw_lat,
                "old_lon": round(float(w_lon), 4) if w_lon != 0.0 else nw_lon,
            })

    # 4. Ensure all 12 district-level Dac khu are registered in `isle`
    #    (Xa Tho Chau -> Dac khu Tho Chau is mapped directly via eo & wn = 13 total dac khu)
    for d_full, p_full, wn_stem, pn_stem in _DISTRICT_LEVEL_DAC_KHU:
        d_norm = strip_admin_prefix(d_full, ("quan", "huyen", "thi xa", "thanh pho"))
        did_o = do_map[d_norm]
        pid_o = po_map[p_full]
        wid_n = wn_map[wn_stem]
        pid_n = pn_map[pn_stem]
        key_i = (did_o, pid_o, wid_n, pid_n)
        if key_i not in isle_seen:
            isle_seen.add(key_i)
            isle_edges.append([did_o, pid_o, wid_n, pid_n])

    pm_list = [pm_edges.get(pid_o, -1) for pid_o in range(len(po_disp))]
    dp_list = [sorted(dp_edges.get(did_o, set())) for did_o in range(len(do_disp))]
    eo_list = [eo_edges.get(wid_o, []) for wid_o in range(len(wo_disp))]
    en_list = [en_entries.get(wid_n, []) for wid_n in range(len(wn_disp))]

    converter_payload = {
        "version": VERSION,
        "license": "MIT",
        "source": (
            "tranngocminhhieu/vietnamadminunits + "
            "thanglequoc/vietnamese-provinces-database (v5.2.0) + "
            "madnh/hanhchinhvn (v1.2/v1.6.0) [MIT]"
        ),
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
        # Additive backward-compatible v3.2 metadata
        "wn_code_disp": dict(sorted(wn_code_disp.items(), key=lambda kv: int(kv[0]))),
        "pn_full_disp": pn_full_disp,
        "p_postal": p_postal,
        "w_postal": dict(sorted(w_postal.items(), key=lambda kv: int(kv[0]))),
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

    converter_bytes_str = json.dumps(converter_payload, separators=(",", ":"), ensure_ascii=False)
    CONVERTER_OUT.write_text(converter_bytes_str, encoding="utf-8")
    conv_size = CONVERTER_OUT.stat().st_size
    if conv_size >= MAX_CONVERTER_BYTES:
        raise RuntimeError(
            f"vn_admin_v3_converter.json size {conv_size} B exceeds < {MAX_CONVERTER_BYTES} B budget"
        )

    RECORDS_OUT.write_text(
        json.dumps({"version": VERSION, "license": "MIT", "records": records}, ensure_ascii=False),
        encoding="utf-8",
    )

    conv_sha = sha256_file(CONVERTER_OUT)
    new_counts = {
        "legacy_provinces": len(po_disp),
        "legacy_districts": len(do_disp),
        "legacy_province_district_pairs": sum(len(x) for x in dp_list),
        "legacy_wards": len(wo_disp),
        "new_provinces_2025": len(pn_disp),
        "new_wards_2025": len(wn_disp),
        "new_ward_entries_2025": sum(len(x) for x in en_list),
        "island_special_zones": len(isle_edges),
        "dac_khu_total_2025": sum(1 for rows in en_list for r in rows if r[5] == 3),
        "wn_code_disp_collision_fixes": len(wn_code_disp),
        "postal_wards_covered": len(w_postal),
        "total_mapping_records": len(records),
    }

    provenance = {
        "version": VERSION,
        "license": "MIT Only (strictly permissive; zero GPL or custom-clause data ingested)",
        "ingested_mit_repositories": [
            {
                "name": "tranngocminhhieu/vietnamadminunits",
                "url": "https://github.com/tranngocminhhieu/vietnamadminunits",
                "license": "MIT License (Copyright (c) 2025 Tran Ngoc Minh Hieu)",
                "license_file": LICENSE_VAU.name,
            },
            {
                "name": "thanglequoc/vietnamese-provinces-database",
                "url": "https://github.com/thanglequoc/vietnamese-provinces-database",
                "version": supplement.get("thanglequoc_metadata", {}).get("data_version", "5.2.0"),
                "license": "MIT License (Copyright (c) 2021 Thang Le Quoc)",
                "license_file": LICENSE_TQ.name,
            },
            {
                "name": "madnh/hanhchinhvn",
                "url": "https://github.com/madnh/hanhchinhvn",
                "versions": ["v1.2", "v1.6.0"],
                "license": "MIT License (Copyright (c) Danh Manh)",
                "license_file": LICENSE_MADNH.name,
            },
        ],
        "cross_verified_only_excluded_from_ingested_assets": [
            {
                "repository": "daohoangson/dvhcvn",
                "license": "GPL-3.0",
                "status": "cross_verified_only",
            },
            {
                "repository": "vietmap-company/vietnam_administrative_address",
                "license": "VietMap Custom Open-Source Derivative License",
                "status": "cross_verified_only",
            },
            {
                "repository": "zuydd/vn-geo",
                "license": "No LICENSE file",
                "status": "cross_verified_only",
            },
            {
                "repository": "thanhtrungit97/dvhcvn",
                "license": "No LICENSE file",
                "status": "cross_verified_only",
            },
        ],
        "raw_files": [
            {
                "filename": CONVERT_CSV.name,
                "url": "https://raw.githubusercontent.com/tranngocminhhieu/vietnamadminunits/main/data/processed/convert_legacy_2025_with_location_and_default_ward.csv",
                "sha256": sha256_file(CONVERT_CSV),
                "rows": total_conv_rows,
            },
            {
                "filename": VN2025_CSV.name,
                "url": "https://raw.githubusercontent.com/tranngocminhhieu/vietnamadminunits/main/data/processed/2025_34-province-3221-ward_with_location.csv",
                "sha256": sha256_file(VN2025_CSV),
            },
            {
                "filename": GSO_WARD_CSV.name,
                "url": "https://raw.githubusercontent.com/tranngocminhhieu/vietnamadminunits/main/data/raw/danhmuchanhchinh.gso.gov.vn_ward_2025-07-18.csv",
                "sha256": sha256_file(GSO_WARD_CSV),
            },
            {
                "filename": SUPPLEMENT_FILE.name,
                "sha256": sha256_file(SUPPLEMENT_FILE),
            },
            *[
                {
                    "filename": path.name,
                    "sha256": sha256_file(path),
                }
                for path in (LICENSE_VAU, LICENSE_TQ, LICENSE_MADNH)
                if path.exists()
            ],
            *supplement.get("mit_source_hashes", []),
        ],
        "counts": new_counts,
        "converter_sha256": conv_sha,
        "converter_bytes": conv_size,
    }
    PROVENANCE_OUT.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    diff_report = {
        "version": VERSION,
        "baseline": BASELINE_V300,
        "updated": {
            "version": VERSION,
            "converter_sha256": conv_sha,
            "converter_bytes": conv_size,
            "under_1_8_mb_budget": conv_size < MAX_CONVERTER_BYTES,
            "counts": new_counts,
        },
        "deltas": {
            "legacy_provinces_delta": new_counts["legacy_provinces"] - BASELINE_V300["counts"]["legacy_provinces"],
            "legacy_districts_delta": new_counts["legacy_districts"] - BASELINE_V300["counts"]["legacy_districts"],
            "legacy_province_district_pairs_delta": new_counts["legacy_province_district_pairs"] - BASELINE_V300["counts"]["legacy_province_district_pairs"],
            "legacy_wards_delta": new_counts["legacy_wards"] - BASELINE_V300["counts"]["legacy_wards"],
            "new_provinces_delta": new_counts["new_provinces_2025"] - BASELINE_V300["counts"]["new_provinces_2025"],
            "new_wards_delta": new_counts["new_wards_2025"] - BASELINE_V300["counts"]["new_wards_2025"],
            "island_special_zones_delta": new_counts["island_special_zones"] - BASELINE_V300["counts"]["island_special_zones"],
            "total_mapping_records_delta": new_counts["total_mapping_records"] - BASELINE_V300["counts"]["total_mapping_records"],
            "base_to_updated_do": f"{base_do_count} -> {len(do_disp)}",
            "base_to_updated_dp_pairs": f"{base_dp_pairs} -> {new_counts['legacy_province_district_pairs']}",
            "base_to_updated_wo": f"{base_wo_count} -> {len(wo_disp)}",
            "base_to_updated_records": f"{base_records_count} -> {len(records)}",
        },
        "2026_national_assembly_updates": {
            "province_city_promotions": [
                {"code": 22, "short": "Quảng Ninh", "full_2026": prov_2026_full.get("22"), "decree": "36/2026/QH16"},
                {"code": 24, "short": "Bắc Ninh", "full_2026": prov_2026_full.get("24"), "decree": "39/2026/QH16"},
                {"code": 75, "short": "Đồng Nai", "full_2026": prov_2026_full.get("75"), "decree": "30/2026/QH16"},
            ],
            "ward_type_updates_applied_count": len(wt_updates_applied),
            "ward_type_updates_applied": wt_updates_applied,
        },
        "historical_integration": {
            "extraction_stats": supplement.get("extraction_stats", {}),
            "historical_districts_added_count": len(historical_districts_added),
            "historical_districts_added": sorted(historical_districts_added),
        },
        "dac_khu_13_coverage": {
            "district_level_isle_entries": len(isle_edges),
            "xa_tho_chau_via_eo_and_wn": True,
            "all_13_dac_khu_names": [
                "Đặc khu Phú Quốc", "Đặc khu Côn Đảo", "Đặc khu Bạch Long Vĩ", "Đặc khu Cồn Cỏ",
                "Đặc khu Hoàng Sa", "Đặc khu Trường Sa", "Đặc khu Lý Sơn", "Đặc khu Cô Tô",
                "Đặc khu Vân Đồn", "Đặc khu Cát Hải", "Đặc khu Phú Quý", "Đặc khu Kiên Hải",
                "Đặc khu Thổ Châu",
            ],
        },
        "hygiene_and_accuracy_fixes": {
            "wn_code_disp_collision_fixes": len(wn_code_disp),
            "dangling_da_aliases_resolved": ["q2 -> quan 2", "q 2 -> quan 2", "q9 -> quan 9", "q 9 -> quan 9"],
            "new_da_aliases": ["phu quy -> phu qui", "qui nhon -> quy nhon", "phan rang-thap cham -> phan rang - thap cham", "ia h' drai -> ia h'drai"],
            "postal_codes_decision_2334": {"provinces": len(p_postal), "wards": len(w_postal)},
        },
        "license_audit": {
            "ingested": ["vietnamadminunits (MIT)", "thanglequoc/vietnamese-provinces-database (MIT)", "madnh/hanhchinhvn (MIT)"],
            "cross_verified_only_excluded": ["daohoangson/dvhcvn (GPL-3.0)", "vietmap (custom clause)", "vn-geo (no LICENSE)", "thanhtrungit97/dvhcvn (no LICENSE)"],
        },
    }
    DIFF_REPORT_OUT.write_text(json.dumps(diff_report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Sanity assertions (existing + new v3.2 assertions)
    w_px = wo_map["phuc xa"]
    d_bd = do_map["ba dinh"]
    p_hn = po_map["ha noi"]
    assert any(e[0] == d_bd and e[1] == p_hn and converter_payload["wn"][e[2]] == "hong ha" for e in eo_list[w_px])

    w_lt = wo_map["loc thanh"]
    d_ln = do_map["loc ninh"]
    p_bp = po_map["binh phuoc"]
    assert any(e[0] == d_ln and e[1] == p_bp and converter_payload["pn"][e[3]] == "dong nai" for e in eo_list[w_lt])

    # Historical Quận 2, Quận 9, Huyện Từ Liêm, Phường 6 Quận 3
    assert "quan 2" in do_map and "quan 9" in do_map and "tu liem" in do_map
    assert any(e[0] == do_map["quan 3"] and e[1] == po_map["ho chi minh"] and converter_payload["wn"][e[2]] == "xuan hoa" for e in eo_list[wo_map["phuong 6"]])
    assert any(e[0] == do_map["quan 2"] and e[1] == po_map["ho chi minh"] and converter_payload["wn"][e[2]] == "an khanh" for e in eo_list[wo_map["thu thiem"]])
    assert any(e[0] == do_map["tu liem"] and e[1] == po_map["ha noi"] and converter_payload["wn"][e[2]] == "tu liem" for e in eo_list[wo_map["cau dien"]])

    # All 13 đặc khu present in wn (wt=3) & 12 district-level in isle
    assert len(isle_edges) == 12
    assert new_counts["dac_khu_total_2025"] == 13
    assert wn_code_disp.get("31108") == "Kiên Hải"

    print(
        f"Built v{VERSION} MIT Converter: {len(po_disp)} old prov -> {len(pn_disp)} new prov, "
        f"{len(do_disp)} old dist, {len(wo_disp)} old wards -> {len(wn_disp)} new wards "
        f"({new_counts['new_ward_entries_2025']} entries, {len(isle_edges)} isle / 13 dac khu), "
        f"{conv_size / 1024:.1f} KB"
    )
    return diff_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Laya v3.2 MIT Administrative Converter & Records")
    parser.add_argument(
        "--sources-dir",
        type=Path,
        default=None,
        help="Optional directory with crawled/fetched MIT sources (legacy/ & new_2025/)",
    )
    args = parser.parse_args()
    build(sources_dir=args.sources_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
