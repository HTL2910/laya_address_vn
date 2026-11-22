"""Comprehensive unit tests and throughput benchmark for laya_v3_core."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

# Ensure package parent is importable regardless of pytest invocation cwd.
_PKG_PARENT = str(Path(__file__).resolve().parents[2])
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from laya_v3_core import (  # noqa: E402
    DEFAULT_CONVERTER_PATH,
    SymSpellGazetteer,
    VietnamAdminV3Index,
    haversine_km,
    normalize_text,
    repair_telex_typos,
    strip_accents,
)


@pytest.fixture(scope="module")
def admin_index() -> VietnamAdminV3Index:
    return VietnamAdminV3Index(DEFAULT_CONVERTER_PATH)


@pytest.fixture(scope="module")
def converter_data() -> dict:
    with DEFAULT_CONVERTER_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 1. Normalization & Telex Repair
# ---------------------------------------------------------------------------
def test_strip_accents_and_normalize() -> None:
    assert strip_accents("Phường Đống Đa, Hà Nội") == "Phuong Dong Da, Ha Noi"
    assert normalize_text("Bà Rịa – Vũng Tàu") == "ba ria - vung tau"
    assert normalize_text("  TP. Hồ Chí Minh, Việt Nam! ") == "tp ho chi minh viet nam"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("phuowng", "phuong"),
        ("loocj", "loc"),
        ("binhf", "binh"),
        ("thanhf", "thanh"),
        ("huyeenj", "huyen"),
        ("quaanj", "quan"),
        ("dduowngf", "duong"),
        ("xaax", "xa"),
        ("haf nooij", "ha noi"),
        ("tieenf", "tien"),
        ("thuwowngf", "thuong"),
        ("Xã Loocj Thanhj, H. Loocj Ninhf", "Xã Loc Thanh, H. Loc Ninh"),
        ("Phuowngf Beens Nghes", "Phuong Ben Nghe"),
        ("caafn giuowcs", "can giuoc"),
        ("bawsc giang", "bac giang"),
        ("ddawsk lawsk", "dak lak"),
        ("phu7o7ng2 tie6n2", "phuong tien"),
    ],
)
def test_repair_telex_typos_fixes_artifacts(raw: str, expected: str) -> None:
    assert repair_telex_typos(raw) == expected


@pytest.mark.parametrize(
    "preserved",
    [
        "quan jean",
        "ok shop",
        "hi bbQ size M ship cod",
        "bbq coffee free inbox book food",
        "45/2B",
        "0912345678",
        "Q1 P7 123/4A",
        "https://shopee.vn/phuowngf?id=123",
        "customer.service@gmail.com",
        "Ia Boong, La Ee, Dooc May",
    ],
)
def test_repair_telex_typos_preserves_english_numbers_urls(preserved: str) -> None:
    assert repair_telex_typos(preserved) == preserved


# ---------------------------------------------------------------------------
# 2. SymSpellGazetteer O(1) Fuzzy Matcher
# ---------------------------------------------------------------------------
def test_symspell_build_time_and_lookup(converter_data: dict) -> None:
    t0 = time.perf_counter()
    gaz = SymSpellGazetteer.from_converter_data(converter_data)
    build_ms = (time.perf_counter() - t0) * 1000.0
    assert build_ms < 50.0, f"SymSpell build took {build_ms:.2f} ms (target < 50 ms)"

    # Short phrases (< 4 chars) must not trigger
    assert gaz.lookup("ha") is None
    assert gaz.lookup("q1") is None

    # Exact match first
    assert gaz.lookup("loc thanh") == "loc thanh"

    # Single transposition / deletion / insertion / substitution
    assert gaz.lookup("loc tahnh") == "loc thanh"
    assert gaz.lookup("binh phuc", level="province") == "binh phuoc"
    assert gaz.lookup("ha nio") == "ha noi"
    assert gaz.lookup("ho chi mnih") == "ho chi minh"
    assert gaz.lookup("ba dninh") == "ba dinh"

    # Ambiguous multi-entry tie ignored without context, resolved with parent context
    assert gaz.lookup("son ta") is None
    t_ctx0 = time.perf_counter()
    resolved_ctx = gaz.lookup("son ta", parent_context=["ha noi"])
    ctx_ms = (time.perf_counter() - t_ctx0) * 1000.0
    assert resolved_ctx == "son tay"
    assert ctx_ms < 0.3, f"Cold context lookup took {ctx_ms:.3f} ms (target < 0.3 ms)"

    # Lookup latency over 100 fuzzy probes < 0.3 ms avg
    queries = ["loc tahnh", "ha nio", "ho chi mnih", "ba dninh", "loc nihn"] * 20
    t_start = time.perf_counter()
    for q in queries:
        gaz.lookup(q)
    avg_ms = ((time.perf_counter() - t_start) * 1000.0) / len(queries)
    assert avg_ms < 0.3, f"SymSpell avg lookup {avg_ms:.4f} ms (target < 0.3 ms)"


# ---------------------------------------------------------------------------
# 3. Legacy -> 2025 Conversion & Island / Divided Ward Handling
# ---------------------------------------------------------------------------
def test_convert_legacy_to_2025_standard_and_cross_province(
    admin_index: VietnamAdminV3Index,
) -> None:
    res_hn = admin_index.convert_legacy_to_2025("Phường Phúc Xá", "Quận Ba Đình", "Hà Nội")
    assert res_hn["status"] == "converted"
    assert res_hn["is_divided"] is False
    assert res_hn["new_ward"] == "Phường Hồng Hà"
    assert res_hn["new_province"] == "Hà Nội"
    assert res_hn["new_ward_code"] == 97
    assert res_hn["new_province_code"] == 1
    assert len(res_hn["candidates"]) == 1

    res_dn = admin_index.convert_legacy_to_2025("Lộc Thạnh", "Lộc Ninh", "Bình Phước")
    assert res_dn["status"] == "converted"
    assert res_dn["is_divided"] is False
    assert res_dn["new_ward"] == "Xã Lộc Thạnh"
    assert res_dn["new_province"] == "Đồng Nai"
    assert res_dn["new_ward_code"] == 25280
    assert res_dn["new_province_code"] == 75


def test_convert_legacy_to_2025_island_and_divided_and_edge_cases(
    admin_index: VietnamAdminV3Index,
) -> None:
    # Island special zone (both ward-slot and district-slot invocations)
    for w, d in [("Huyện Côn Đảo", ""), ("", "Huyện Côn Đảo"), ("Côn Đảo", "Côn Đảo")]:
        res_isle = admin_index.convert_legacy_to_2025(w, d, "Bà Rịa - Vũng Tàu")
        assert res_isle["status"] == "converted"
        assert res_isle["is_divided"] is False
        assert res_isle["new_ward"] == "Đặc khu Côn Đảo"
        assert res_isle["new_province"] == "Hồ Chí Minh"

    # Divided ward puts is_default=True first + full candidate list
    res_div = admin_index.convert_legacy_to_2025("Phường Cống Vị", "Ba Đình", "Hà Nội")
    assert res_div["status"] == "converted"
    assert res_div["is_divided"] is True
    assert res_div["new_ward"] == "Phường Ngọc Hà"
    assert res_div["candidates"][0]["is_default"] is True
    cand_wards = {c["new_ward"] for c in res_div["candidates"]}
    assert cand_wards == {"Phường Ngọc Hà", "Phường Giảng Võ"}

    # Numbered ward & abbreviated district/province + Phuong Mai proper name protection
    res_p1 = admin_index.convert_legacy_to_2025("P.1", "Q10", "TPHCM")
    assert res_p1["status"] == "converted"
    assert res_p1["new_ward"] == "Phường Vườn Lài"
    assert res_p1["new_province"] == "Hồ Chí Minh"

    res_pm = admin_index.convert_legacy_to_2025("Phường Phương Mai", "Đống Đa", "Hà Nội")
    assert res_pm["status"] == "converted"
    assert res_pm["new_ward"] == "Phường Kim Liên"

    # Direct 2025 input does not misconvert into same-named legacy rural ward
    res_2025 = admin_index.convert_legacy_to_2025("Phường Hồng Hà", "", "Hà Nội")
    assert res_2025["status"] == "already_2025"
    assert res_2025["new_ward"] == "Phường Hồng Hà"
    assert res_2025["new_province"] == "Hà Nội"

    # Unknown returns not_found cleanly
    res_nf = admin_index.convert_legacy_to_2025("Xã Không Tồn Tại", "Huyện Ảo", "Hà Nội")
    assert res_nf["status"] == "not_found"
    assert res_nf["new_ward"] is None


# ---------------------------------------------------------------------------
# 4. Bidirectional Equivalence
# ---------------------------------------------------------------------------
def test_is_legacy_2025_equivalent(admin_index: VietnamAdminV3Index) -> None:
    # Legacy <-> 2025 both directions
    assert admin_index.is_legacy_2025_equivalent(
        "Phúc Xá", "Ba Đình", "Hà Nội", "Phường Hồng Hà", "", "Hà Nội"
    )
    assert admin_index.is_legacy_2025_equivalent(
        "Phường Hồng Hà", "", "Hà Nội", "Phúc Xá", "Ba Đình", "Hà Nội"
    )

    # Cross-province merger both directions
    assert admin_index.is_legacy_2025_equivalent(
        "Lộc Thạnh", "Lộc Ninh", "Bình Phước", "Xã Lộc Thạnh", "", "Đồng Nai"
    )
    assert admin_index.is_legacy_2025_equivalent(
        "Xã Lộc Thạnh", "", "Đồng Nai", "Lộc Thạnh", "Lộc Ninh", "Bình Phước"
    )

    # Island zone equivalence
    assert admin_index.is_legacy_2025_equivalent(
        "Huyện Côn Đảo", "", "Bà Rịa - Vũng Tàu", "Đặc khu Côn Đảo", "", "Hồ Chí Minh"
    )

    # Divided ward matches either resulting 2025 split
    assert admin_index.is_legacy_2025_equivalent(
        "Cống Vị", "Ba Đình", "Hà Nội", "Phường Giảng Võ", "", "Hà Nội"
    )

    # Non-equivalent cases
    assert not admin_index.is_legacy_2025_equivalent(
        "Phúc Xá", "Ba Đình", "Hà Nội", "Trúc Bạch", "Ba Đình", "Hà Nội"
    )
    assert not admin_index.is_legacy_2025_equivalent(
        "Phúc Xá", "Ba Đình", "Hà Nội", "Chương Dương", "Hoàn Kiếm", "Hà Nội"
    )
    assert not admin_index.is_legacy_2025_equivalent(
        "Đội Cấn", "Ba Đình", "Hà Nội", "Ngọc Hà", "Ba Đình", "Hà Nội"
    )
    assert not admin_index.is_legacy_2025_equivalent(
        "Phúc Xá", "Ba Đình", "Hà Nội", "Bến Nghé", "Quận 1", "Hồ Chí Minh"
    )


# ---------------------------------------------------------------------------
# 5. Spatial Haversine & Plausibility Outlier Detection
# ---------------------------------------------------------------------------
def test_spatial_plausibility(admin_index: VietnamAdminV3Index) -> None:
    # Same-region point in Phúc Xá, Ba Đình (~0-2 km away)
    ok = admin_index.spatial_plausibility(
        "Phường Phúc Xá", "Quận Ba Đình", "Hà Nội", 21.047, 105.848
    )
    assert ok["status"] == "ok"
    assert ok["is_outlier"] is False
    assert ok["dist_km"] < 5.0
    assert 15.0 <= ok["max_expected_km"] <= 60.0

    # Far outlier (> 100 km: Hanoi address with HCMC coordinates)
    far = admin_index.spatial_plausibility(
        "Phường Phúc Xá", "Quận Ba Đình", "Hà Nội", 10.7769, 106.7009
    )
    assert far["status"] == "ok"
    assert far["is_outlier"] is True
    assert far["dist_km"] > 1000.0

    # Haversine sanity check (Hanoi <-> HCMC ~1140-1155 km)
    d_hn_hcm = haversine_km(21.0285, 105.8542, 10.7769, 106.7009)
    assert 1130.0 < d_hn_hcm < 1165.0


# ---------------------------------------------------------------------------
# 6. Smart 2025 Disambiguation, Top-3 Suggestions & V3 Parse Pipeline
# ---------------------------------------------------------------------------
def test_parse_and_normalize_v3_smart_disambiguation_and_suggestions(
    admin_index: VietnamAdminV3Index,
) -> None:
    # Bare district-overlapping name without street/ward prefix -> address_insufficient + 3 suggestions
    insuff = admin_index.parse_and_normalize_v3("Nà Hang, Tuyên Quang")
    assert insuff["sufficiency"] == "address_insufficient"
    assert insuff["new_ward"] is None
    assert insuff["new_province"] == "Tuyên Quang"
    assert len(insuff["suggested_wards"]) == 3
    assert insuff["suggested_wards"][0]["new_ward"] == "Xã Nà Hang"

    # Same phrase WITH street/house cue -> resolves to 2025 Xã Nà Hang as address_sufficient
    suff_street = admin_index.parse_and_normalize_v3("12 Lê Lợi, Nà Hang, Tuyên Quang")
    assert suff_street["sufficiency"] == "address_sufficient"
    assert suff_street["street"] == "12 Lê Lợi"
    assert suff_street["new_ward"] == "Xã Nà Hang"
    assert suff_street["new_province"] == "Tuyên Quang"
    assert suff_street["suggested_wards"] == []

    # Explicit ward prefix resolves cleanly without street cue
    suff_prefix = admin_index.parse_and_normalize_v3("Xã Nà Hang, Tuyên Quang")
    assert suff_prefix["sufficiency"] == "address_sufficient"
    assert suff_prefix["new_ward"] == "Xã Nà Hang"

    # Explicit district prefix without ward -> insufficient
    insuff_dist = admin_index.parse_and_normalize_v3("Huyện Na Hang, Tuyên Quang")
    assert insuff_dist["sufficiency"] == "address_insufficient"
    assert len(insuff_dist["suggested_wards"]) == 3


def test_parse_and_normalize_v3_complex_real_world_inputs(
    admin_index: VietnamAdminV3Index,
) -> None:
    # Telex + abbreviated district/province + street preservation
    r1 = admin_index.parse_and_normalize_v3("123 Nguyễn Huệ, Phuowngf Beens Nghes, Q1, TPHCM")
    assert r1["sufficiency"] == "address_sufficient"
    assert r1["street"] == "123 Nguyễn Huệ"
    assert r1["new_ward"] == "Phường Sài Gòn"
    assert r1["new_province"] == "Hồ Chí Minh"

    # Protected hyphenated province ('Bà Rịa - Vũng Tàu') + slash street number
    r2 = admin_index.parse_and_normalize_v3(
        "Số 45/2B, đường 30/4, Phường 1, TP. Vũng Tàu, Bà Rịa - Vũng Tàu"
    )
    assert r2["sufficiency"] == "address_sufficient"
    assert r2["street"] == "Số 45/2B, đường 30/4"
    assert r2["new_ward"] == "Phường Vũng Tàu"
    assert r2["new_province"] == "Hồ Chí Minh"

    # Protected hyphenated district ('Phan Rang - Tháp Chàm')
    r3 = admin_index.parse_and_normalize_v3(
        "Phường Đạo Long, TP. Phan Rang - Tháp Chàm, Ninh Thuận"
    )
    assert r3["sufficiency"] == "address_sufficient"
    assert r3["new_ward"] == "Phường Phan Rang"
    assert r3["new_province"] == "Khánh Hòa"

    # Comma-free address
    r4 = admin_index.parse_and_normalize_v3("12 Lê Lợi Nà Hang Tuyên Quang")
    assert r4["sufficiency"] == "address_sufficient"
    assert r4["street"] == "12 Lê Lợi"
    assert r4["new_ward"] == "Xã Nà Hang"
    assert r4["new_province"] == "Tuyên Quang"

    # SymSpell 1-edit recovery inside parse pipeline
    r5 = admin_index.parse_and_normalize_v3("Loc Tahnh, Loc Ninh, Binh Phuc")
    assert r5["sufficiency"] == "address_sufficient"
    assert r5["new_ward"] == "Xã Lộc Thạnh"
    assert r5["new_province"] == "Đồng Nai"

    # Telex-only ward + district without province
    r6 = admin_index.parse_and_normalize_v3("Xã Loocj Thanhj, H. Loocj Ninhf")
    assert r6["sufficiency"] == "address_sufficient"
    assert r6["new_ward"] == "Xã Lộc Thạnh"
    assert r6["new_province"] == "Đồng Nai"


def test_throughput_benchmark_500_strings(admin_index: VietnamAdminV3Index) -> None:
    sample_pool = [
        "12 Lê Lợi, Phường Phúc Xá, Quận Ba Đình, Hà Nội",
        "123 Nguyễn Huệ, Phuowngf Beens Nghes, Q1, TPHCM",
        "Xã Loocj Thanhj, H. Loocj Ninhf, Bình Phước",
        "Số 45/2B đường 30/4, Phường 1, TP. Vũng Tàu, Bà Rịa - Vũng Tàu",
        "12 Lê Lợi, Nà Hang, Tuyên Quang",
        "Nà Hang, Tuyên Quang",
        "Huyện Côn Đảo, Bà Rịa - Vũng Tàu",
        "Loc Tahnh, Loc Ninh, Binh Phuoc",
        "Phường Cống Vị, Ba Đình, Hà Nội",
        "Phường Đạo Long, TP. Phan Rang - Tháp Chàm, Ninh Thuận",
    ]
    workload = sample_pool * 50  # 500 strings
    assert len(workload) == 500

    # Warm-up 1 call before timed 500-string batch
    admin_index.parse_and_normalize_v3(workload[0])

    t0 = time.perf_counter()
    results = [admin_index.parse_and_normalize_v3(addr) for addr in workload]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    avg_ms = elapsed_ms / len(workload)

    assert len(results) == 500
    assert avg_ms < 1.0, f"500-string parse throughput avg {avg_ms:.4f} ms/str (target < 1.0 ms)"
