"""Independent 600-Case Vietnamese Address Hold-Out Suite Generator (seed = 20261122).

Uses distinct templates, phrasing, and seed from BOTH Way A v2 and Way B v2 training sets.
Audits zero overlap against the v1 90-case benchmark (`../benchmark/vn_address_benchmark.json`).
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
RECORDS_PATH = ROOT / "shared_data" / "vn_admin_full_records.json"
OLD_BENCH_PATH = ROOT / "benchmark" / "vn_address_benchmark.json"
OUT_PATH = ROOT / "benchmark_v2" / "independent_holdout_v2.json"

SEED = 20261122


def strip_accents(s: str) -> str:
    s = (s or "").replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(s).lower().strip())


def load_old_benchmark_keys() -> Set[str]:
    if not OLD_BENCH_PATH.exists():
        return set()
    raw = json.loads(OLD_BENCH_PATH.read_text(encoding="utf-8"))
    return {norm_key(c["text"]) for c in raw.get("cases", [])}


def mix_case(rng: random.Random, s: str) -> str:
    r = rng.random()
    if r < 0.35:
        return s.upper()
    if r < 0.7:
        return s.lower()
    words = s.split(" ")
    return " ".join(w.upper() if rng.random() < 0.35 else w.lower() for w in words)


def minor_typo(rng: random.Random, s: str) -> str:
    # Realistic VN customer typos: drop space after dot/comma, double space, drop one accent/vowel
    ops = [
        lambda x: x.replace(", ", ","),
        lambda x: x.replace(". ", "."),
        lambda x: re.sub(r"\s+", "  ", x, count=1),
        lambda x: x.replace("Phường", "Phuong").replace("Quận", "Quan"),
    ]
    return rng.choice(ops)(s)


def build_holdout() -> Dict[str, Any]:
    rng = random.Random(SEED)
    records = json.loads(RECORDS_PATH.read_text(encoding="utf-8"))
    legacy = list(records["legacy_3tier"])
    merged = list(records["merged_2025_2tier"])
    rng.shuffle(legacy)
    rng.shuffle(merged)

    old_keys = load_old_benchmark_keys()
    seen_keys: Set[str] = set()
    cases: List[Dict[str, Any]] = []

    def add_case(cid: str, group: str, subgroup: str, suff: str, text: str) -> bool:
        k = norm_key(text)
        if not text.strip() or k in seen_keys or k in old_keys:
            return False
        seen_keys.add(k)
        cases.append({
            "id": cid,
            "group": group,
            "subgroup": subgroup,
            "expected_sufficiency": suff,
            "expected_is_address": (suff == "address_sufficient"),
            "text": text.strip(),
        })
        return True

    # ------------------------------------------------------------------
    # Group 1: full_address (100 cases, address_sufficient)
    # ------------------------------------------------------------------
    streets = [
        "Nguyễn Huệ", "Lê Văn Sỹ", "Võ Văn Kiệt", "Trường Chinh", "Hoàng Văn Thụ",
        "Nguyễn Văn Cừ", "Phan Xích Long", "Lê Đức Thọ", "Tô Hiệu", "Nguyễn Oanh",
        "Lý Tự Trọng", "Phạm Ngũ Lão", "Trần Quốc Toản", "Phan Chu Trinh", "Lê Quý Đôn",
        "Hùng Vương", "Quang Trung", "Nguyễn Đình Chiểu", "Võ Thị Sáu", "Lạc Long Quân",
        "Nguyễn Xiển", "Khuất Duy Tiến", "Tố Hữu", "Ngô Gia Tự", "Huỳnh Tấn Phát",
        "Nguyễn Ái Quốc", "Đồng Khởi", "Lê Hồng Phong", "Nguyễn Thị Thập", "Cộng Hòa",
    ]
    buildings = [
        "Chung cư Moonlight", "Chung cư Sunrise City", "Tòa nhà Lotte", "Chung cư Vinhomes Smart City",
        "Chung cư Goldmark City", "Tòa nhà Saigon Centre", "KĐT Ecopark", "Chung cư Celadon City",
        "Tòa A2 Chung cư An Bình", "Chung cư Saigon Pearl", "KĐT Sala", "Tòa nhà Pearl Plaza",
    ]
    kcn_names = ["KCN Sóng Thần 2", "KCN Amata", "KCN Tân Tạo", "KCN Quế Võ", "KCN Đình Vũ", "KCN Hòa Khánh"]
    person_names = [
        "Anh Minh", "Chị Lan", "Anh Hoàng", "Chị Thảo", "Cô Hạnh", "Chú Bình",
        "Nguyễn Minh Tuấn", "Phạm Thị Hồng", "Lê Quốc Bảo", "Võ Thanh Hà",
    ]
    notes = [
        "gọi trước khi giao 15p", "giao giờ hành chính nha", "ship ngoài giờ giúp mình",
        "nhờ bảo vệ nhận hộ nếu không nghe máy", "giao tận phòng giúp em", "không cần gọi chuông",
    ]
    phones = ["0913445566", "0977889900", "0388112244", "0905667788", "0868223344", "0945112299"]

    idx = 0
    g1_count = 0
    while g1_count < 100:
        r = legacy[idx % len(legacy)]
        idx += 1
        st = rng.choice(streets)
        house = rng.choice([f"{rng.randint(1, 350)}", f"{rng.randint(1, 200)}/{rng.randint(1, 30)}", f"{rng.randint(1, 90)}/{rng.randint(1, 20)}/{rng.randint(1, 15)}", f"{rng.randint(1, 120)}{rng.choice('ABCD')}"])
        sub = g1_count % 5
        if sub == 0:
            # Urban formal / semi-formal
            txt = f"Số {house} Đường {st}, {r['ward_full']}, {r['district_full']}, {r['province_full']}"
            sg = "full_urban"
        elif sub == 1:
            # Rural / industrial
            if g1_count % 2 == 0:
                unit = rng.choice([f"Thôn {rng.randint(1, 9)}", f"Ấp {rng.randint(1, 6)}", f"Xóm {rng.randint(1, 8)}", f"Bản {rng.choice(['Cát', 'Hua', 'Lác', 'Mường'])}", f"Khóm {rng.randint(1, 5)}"])
                txt = f"{unit}, {r['ward_full']}, {r['district_full']}, {r['province_full']}"
            else:
                txt = f"Lô {rng.choice('ABCDEF')}{rng.randint(1, 12)}, {rng.choice(kcn_names)}, {r['ward_full']}, {r['district_full']}, {r['province_full']}"
            sg = "full_rural_industrial"
        elif sub == 2:
            # Apartment / office
            bldg = rng.choice(buildings)
            apt = rng.choice([f"Căn {rng.choice('ABC')}{rng.randint(101, 2512)}", f"Tầng {rng.randint(2, 28)}", f"Phòng {rng.randint(201, 1908)}", f"Block {rng.choice('ABCD')} Lầu {rng.randint(1, 18)}"])
            txt = f"{apt}, {bldg}, {house} {st}, {r['ward_full']}, {r['district_full']}, {r['province_full']}"
            sg = "full_apartment"
        elif sub == 3:
            # Deep alley + typo / mixed case
            alley = rng.choice([f"Hẻm {house} {st}", f"Ngõ {rng.randint(1, 180)}/{rng.randint(1, 25)} {st}", f"Kiệt {rng.randint(1, 90)}/{rng.randint(1, 20)} {st}"])
            txt = minor_typo(rng, f"{alley}, {r['ward_full']}, {r['district_bare']}, {r['province_bare']}")
            if g1_count % 3 == 0:
                txt = mix_case(rng, txt)
            sg = "full_alley_typo"
        else:
            # Delivery freetext with customer name + phone + note
            nm = rng.choice(person_names)
            ph = rng.choice(phones)
            nt = rng.choice(notes)
            tpl = rng.choice([
                f"{nm} - {ph} - {house} {st}, {r['ward_full']}, {r['district_bare']}, {r['province_bare']} ({nt})",
                f"giao cho {nm} ({ph}) ở {house} {st}, {r['ward_bare']}, {r['district_bare']}, {r['province_bare']} - {nt}",
                f"nhờ shop ship tới {house} {st} {r['ward_bare']} {r['district_bare']} {r['province_bare']}, sdt {ph}",
            ])
            txt = strip_accents(tpl).lower() if g1_count % 2 == 0 else tpl
            sg = "full_delivery_freetext"

        if add_case(f"H1_{g1_count+1:03d}", "full_address", sg, "address_sufficient", txt):
            g1_count += 1

    # ------------------------------------------------------------------
    # Group 2: short_valid_address (180 cases = 30 x 6 styles, address_sufficient)
    # ------------------------------------------------------------------
    # Ensure multi-word distinct ward names for unprefixed styles so there is zero inherent single-word ambiguity
    multi_legacy = [r for r in legacy if len(r["ward_bare"].split()) >= 2 and len(r["district_bare"].split()) >= 1]
    multi_2025 = [r for r in merged if len(r["ward_bare"].split()) >= 2]

    # Put mandatory real-world example "Lộc Thạnh, Lộc Ninh" first in style 1
    def _find_rec(w_sub: str, d_sub: str) -> Dict[str, str] | None:
        for r in legacy:
            if w_sub.lower() in r["ward_bare"].lower() and d_sub.lower() in r["district_bare"].lower():
                return r
        return None

    loc_thanh = _find_rec("Lộc Thạnh", "Lộc Ninh")

    style_specs = [
        "short_no_prefix_comma",
        "short_no_prefix_no_comma",
        "short_unaccented_lower",
        "short_abbreviated",
        "short_ward_district_province",
        "short_ward_province_2tier",
    ]
    g2_counter = 0
    cursor = 0
    for s_idx, style in enumerate(style_specs):
        taken = 0
        while taken < 30:
            if s_idx == 0 and taken == 0 and loc_thanh is not None:
                r = loc_thanh
            elif style == "short_ward_province_2tier":
                r = multi_2025[cursor % len(multi_2025)]
                cursor += 1
            else:
                r = multi_legacy[cursor % len(multi_legacy)]
                cursor += 1

            if style == "short_no_prefix_comma":
                txt = f"{r['ward_bare']}, {r['district_bare']}"
            elif style == "short_no_prefix_no_comma":
                txt = f"{r['ward_bare']} {r['district_bare']}"
            elif style == "short_unaccented_lower":
                if taken % 3 == 0:
                    txt = strip_accents(f"{r['ward_bare']} {r['district_bare']} {r['province_bare']}").lower()
                else:
                    txt = strip_accents(f"{r['ward_bare']} {r['district_bare']}").lower()
            elif style == "short_abbreviated":
                w_pref = "p." if r["ward_type"].lower().startswith("ph") else ("tt." if "thị trấn" in r["ward_type"].lower() else "x.")
                d_low = r["district_full"].lower()
                d_pref = "q." if d_low.startswith("quận") else ("tx." if d_low.startswith("thị xã") else ("tp." if d_low.startswith("thành phố") else "h."))
                if taken % 3 == 0:
                    txt = f"{w_pref} {r['ward_bare']}, {d_pref} {r['district_bare']}"
                elif taken % 3 == 1:
                    txt = strip_accents(f"{w_pref}{r['ward_bare']} {d_pref}{r['district_bare']}").lower()
                else:
                    txt = strip_accents(f"{w_pref} {r['ward_bare']} {d_pref} {r['district_bare']} {r['province_bare']}").lower()
            elif style == "short_ward_district_province":
                if taken % 2 == 0:
                    txt = f"{r['ward_bare']}, {r['district_bare']}, {r['province_bare']}"
                else:
                    txt = f"{r['ward_full']}, {r['district_full']}, {r['province_bare']}"
            else:  # short_ward_province_2tier
                if taken % 2 == 0:
                    txt = f"{r['ward_bare']}, {r['province_bare']}"
                else:
                    txt = strip_accents(f"{r['ward_full']}, {r['province_bare']}").lower()

            if add_case(f"H2_{g2_counter+1:03d}", "short_valid_address", style, "address_sufficient", txt):
                taken += 1
                g2_counter += 1

    # ------------------------------------------------------------------
    # Group 3: insufficient_address (120 cases, address_insufficient)
    # ------------------------------------------------------------------
    g3_count = 0
    # 3a: ward-only prefixed (35 cases)
    taken = 0
    while taken < 35:
        r = legacy[(cursor + taken) % len(legacy)]
        cursor += 1
        if taken == 0 and loc_thanh is not None:
            txt = "Xã Lộc Thạnh"
        elif taken % 4 == 0:
            txt = r["ward_full"]
        elif taken % 4 == 1:
            txt = strip_accents(r["ward_full"]).lower()
        elif taken % 4 == 2:
            w_p = "P." if r["ward_type"].lower().startswith("ph") else "X."
            txt = f"{w_p} {r['ward_bare']}"
        else:
            txt = f"{'p.' if r['ward_type'].lower().startswith('ph') else 'x.'}{strip_accents(r['ward_bare']).lower()}"
        if add_case(f"H3_{g3_count+1:03d}", "insufficient_address", "insuf_ward_only", "address_insufficient", txt):
            taken += 1
            g3_count += 1

    # 3b: province-only (25 cases)
    seen_provs: List[str] = []
    for r in legacy:
        for cand in (r["province_full"], r["province_bare"], strip_accents(r["province_bare"]).lower()):
            if cand not in seen_provs:
                seen_provs.append(cand)
        if len(seen_provs) >= 40:
            break
    taken = 0
    for cand in seen_provs:
        if taken >= 25:
            break
        if add_case(f"H3_{g3_count+1:03d}", "insufficient_address", "insuf_province_only", "address_insufficient", cand):
            taken += 1
            g3_count += 1

    # 3c: district-only or district+province without ward/street (20 cases)
    taken = 0
    while taken < 20:
        r = legacy[(cursor + taken * 7) % len(legacy)]
        if taken % 2 == 0:
            txt = r["district_full"]
        else:
            txt = f"{r['district_bare']}, {r['province_bare']}"
        if add_case(f"H3_{g3_count+1:03d}", "insufficient_address", "insuf_district_only", "address_insufficient", txt):
            taken += 1
            g3_count += 1

    # 3d: bare granular fragments (40 cases, must include "ngõ 12", "Thôn 7", "thôn 3", "khu phố 3")
    mandatory_frags = [
        "ngõ 12", "Thôn 7", "thôn 3", "khu phố 3", "xóm 4", "ấp 2",
        "tổ dân phố 5", "hẻm 28", "tầng 4", "đường Lê Lợi", "lô B2", "chung cư A1",
    ]
    for frag in mandatory_frags:
        if add_case(f"H3_{g3_count+1:03d}", "insufficient_address", "insuf_fragment_only", "address_insufficient", frag):
            g3_count += 1
    frag_templates = [
        lambda n: f"Ngõ {n}",
        lambda n: f"ngo {n}",
        lambda n: f"Hẻm {n}/{n % 9 + 1}",
        lambda n: f"Thôn {n % 12 + 1}",
        lambda n: f"Ấp {n % 8 + 1}",
        lambda n: f"Xóm {n % 10 + 1}",
        lambda n: f"Khu phố {n % 9 + 1}",
        lambda n: f"Tổ {n % 20 + 1}",
        lambda n: f"Kiệt {n % 50 + 2}",
        lambda n: f"Số nhà {n}",
        lambda n: f"Tầng {n % 15 + 2} Block {rng.choice('ABCD')}",
        lambda n: f"Đường {rng.choice(streets)}",
        lambda n: f"Ngách {n}/{n % 7 + 1}",
    ]
    k_frag = 12
    while g3_count < 120:
        fn = frag_templates[k_frag % len(frag_templates)]
        k_frag += 1
        txt = fn(rng.randint(2, 240))
        if add_case(f"H3_{g3_count+1:03d}", "insufficient_address", "insuf_fragment_only", "address_insufficient", txt):
            g3_count += 1

    # ------------------------------------------------------------------
    # Group 4: garbage_and_chat (100 cases, not_address)
    # ------------------------------------------------------------------
    mandatory_garbage = [
        "73", "10", "K", "Hi", "Ok", "Jean", "Xom", "Xin giá",
        "Cho mình 3 chai nướng BBQ", "ok shop", "ship nhanh giúp",
    ]
    g4_count = 0
    for g in mandatory_garbage:
        if add_case(f"H4_{g4_count+1:03d}", "garbage_and_chat", "mandatory_garbage", "not_address", g):
            g4_count += 1

    garbage_pool = [
        "1", "99", "0", "abc", "xyz", "test", "hello", "alo", "dạ", "vâng ạ", "ok bạn", "chốt", "hủy",
        "xin chào shop", "inbox giá", "giá sao ạ", "còn hàng không shop", "còn size M không",
        "màu trắng còn không bạn", "cho xem ảnh thật đi", "gửi link giúp mình", "tổng hết bao nhiêu",
        "bao giờ có hàng lại", "shop mở cửa mấy giờ", "đặt 2 cái màu đen", "lấy 1 hộp nha",
        "cho 5 gói loại cay", "ship cod không ạ", "freeship nha shop", "giảm giá đi shop ơi",
        "mã giảm giá là gì", "chờ mình chuyển khoản", "đã ck rồi nhé", "kiểm tra giúp em",
        "sao lâu nhận được vậy", "đơn của em tới đâu rồi", "mã vận đơn VN123456789", "GHN88291032",
        "0901234567", "0356789123", "+84 912 345 678", "muahang@yahoo.com", "shopabc.vn",
        "https://tiktok.com/@shoptest", "180.000đ", "499k", "1tr2", "20/11/2026", "12:30",
        "ở chỗ cũ nha", "giao như tuần trước", "gửi về nhà cũ nhé", "để ngoài cổng giùm",
        "gọi mình trước khi tới", "tầm chiều mình mới về", "bảo vệ ký nhận hộ", "chút gửi lại địa chỉ",
        "tối mình nhắn sau", "hẹn thứ 7 nhé", "không cần gấp đâu", "cảm ơn nhiều nha",
        "ok luôn", "đúng rồi ạ", "sai rồi bạn ơi", "đổi sang màu xanh giúp mình",
        "lấy 2kg nhé", "chai 500ml", "combo 3 món", "quần jean ống suông", "áo khoác kaki",
        "giày sneaker trắng size 41", "túi xách nữ da bò", "son môi màu đỏ cam", "kem chống nắng",
        "nước hoa mini 10ml", "bàn phím cơ bluetooth", "tai nghe không dây", "cáp sạc type c",
        "ốp lưng trong suốt", "thẻ nhớ 128gb", "chuột gaming", "lót chuột cỡ lớn",
        "đèn bàn học sinh", "bình giữ nhiệt 1 lít", "ly sứ có nắp", "hộp cơm văn phòng",
        "nồi chiên không dầu", "máy xay sinh tố mini", "quạt tích điện", "ổ cắm đa năng",
        "sách kỹ năng sống", "vở ô ly 200 trang", "bút bi xanh", "tập gym",
        "chúc shop đắt hàng", "hẹn gặp lại lần sau", "tạm biệt", "bye",
    ]
    for txt in garbage_pool:
        if g4_count >= 100:
            break
        if add_case(f"H4_{g4_count+1:03d}", "garbage_and_chat", "chat_garbage", "not_address", txt):
            g4_count += 1

    # ------------------------------------------------------------------
    # Group 5: hard_negative (100 cases, not_address)
    # ------------------------------------------------------------------
    g5_count = 0
    hard_neg_items = [
        # Food / products containing real VN place names (35)
        ("hard_product_place", "bánh mì Sài Gòn"),
        ("hard_product_place", "mực Cà Mau"),
        ("hard_product_place", "phở Hà Nội"),
        ("hard_product_place", "bún bò Huế"),
        ("hard_product_place", "nước mắm Phú Quốc"),
        ("hard_product_place", "cà phê Buôn Ma Thuột"),
        ("hard_product_place", "nem chua Thanh Hóa"),
        ("hard_product_place", "kẹo dừa Bến Tre"),
        ("hard_product_place", "chè Thái Nguyên"),
        ("hard_product_place", "bánh đậu xanh Hải Dương"),
        ("hard_product_place", "chả mực Hạ Long"),
        ("hard_product_place", "bánh tráng Trảng Bàng Tây Ninh"),
        ("hard_product_place", "nhãn lồng Hưng Yên"),
        ("hard_product_place", "vải thiều Lục Ngạn Bắc Giang"),
        ("hard_product_place", "mì Quảng Đà Nẵng"),
        ("hard_product_place", "cơm tấm Sài Gòn"),
        ("hard_product_place", "bánh pía Sóc Trăng"),
        ("hard_product_place", "cho mình 2 hộp bánh đậu xanh Hải Dương"),
        ("hard_product_place", "đặt 1kg mực một nắng Cà Mau nha shop"),
        ("hard_product_place", "lấy 3 chai nước mắm Phú Quốc loại 1"),
        ("hard_product_place", "ship 2 gói trà Tân Cương Thái Nguyên"),
        ("hard_product_place", "mua 5 bịch kẹo dừa Bến Tre làm quà"),
        ("hard_product_place", "10 cái nem chua Thanh Hóa nhé bạn"),
        ("hard_product_place", "bánh đa cua Hải Phòng giá bao nhiêu 1 hộp"),
        ("hard_product_place", "yến sào Khánh Hòa hộp 6 hũ"),
        ("hard_product_place", "tỏi đen Lý Sơn Quảng Ngãi 500g"),
        ("hard_product_place", "hạt điều rang muối Bình Phước 500gr"),
        ("hard_product_place", "tiêu đen Phú Quốc nguyên hạt"),
        ("hard_product_place", "sầu riêng Ri6 Cần Thơ bao ngon"),
        ("hard_product_place", "xoài cát Hòa Lộc Tiền Giang 3kg"),
        ("hard_product_place", "bưởi da xanh Bến Tre loại 1"),
        ("hard_product_place", "gạo ST25 Sóc Trăng túi 5kg"),
        ("hard_product_place", "rượu sim Phú Quốc chai 750ml"),
        ("hard_product_place", "cá kho làng Vũ Đại Hà Nam"),
        ("hard_product_place", "bánh cốm Hàng Than Hà Nội 10 chiếc"),

        # Person names colliding with streets / districts / wards (25)
        ("hard_person_place", "Nguyễn Trãi"),
        ("hard_person_place", "Lê Lợi"),
        ("hard_person_place", "Trần Hưng Đạo"),
        ("hard_person_place", "Hai Bà Trưng"),
        ("hard_person_place", "Nguyễn Thị Thanh Xuân"),
        ("hard_person_place", "Trần Văn Bình Thạnh"),
        ("hard_person_place", "Lê Hoàng Mai"),
        ("hard_person_place", "Phạm Minh Đức"),
        ("hard_person_place", "Nguyễn Văn Tân Bình"),
        ("hard_person_place", "Võ Thị Phú Nhuận"),
        ("hard_person_place", "Hoàng Văn Long Biên"),
        ("hard_person_place", "Đặng Thị Hồng Hà"),
        ("hard_person_place", "Bùi Quốc Hoàn Kiếm"),
        ("hard_person_place", "Anh Nguyễn Trãi vừa gọi cho mình"),
        ("hard_person_place", "Chị Thanh Xuân đặt 2 áo sơ mi"),
        ("hard_person_place", "Bạn Lộc Ninh nhắn tin hỏi đơn"),
        ("hard_person_place", "Ông Lê Lợi là anh hùng dân tộc"),
        ("hard_person_place", "Bác Hồ từng sống ở Bến Nhà Rồng"),
        ("hard_person_place", "Em tên Hải Châu sinh năm 1998"),
        ("hard_person_place", "Cô Bến Nghé bán tạp hóa gần nhà"),
        ("hard_person_place", "Chú Cầu Giấy mới mua xe hôm qua"),
        ("hard_person_place", "Cháu Đông Anh năm nay học lớp 5"),
        ("hard_person_place", "Anh Bình Chánh là trưởng phòng kỹ thuật"),
        ("hard_person_place", "Nguyễn Thị Minh Khai là chiến sĩ cách mạng"),
        ("hard_person_place", "Vua Quang Trung đại phá quân Thanh"),

        # Shipping / branch / narrative / weather / travel questions mentioning real places (40)
        ("hard_chat_place", "Phí ship về Bình Phước bao nhiêu tiền vậy shop?"),
        ("hard_chat_place", "Shop có giao hàng đi Lộc Ninh không ạ?"),
        ("hard_chat_place", "Ship từ Hà Nội vào TP.HCM mất mấy ngày?"),
        ("hard_chat_place", "Bên mình có cửa hàng ở Quận 7 không?"),
        ("hard_chat_place", "Hôm qua tôi mới đi công tác ở Hải Phòng về"),
        ("hard_chat_place", "Cuối tuần gia đình mình đi du lịch Đà Lạt"),
        ("hard_chat_place", "Thời tiết Đà Nẵng hôm nay nắng đẹp lắm"),
        ("hard_chat_place", "Ở Cần Thơ mùa nước nổi rất đẹp"),
        ("hard_chat_place", "Quận Tân Bình chiều nay mưa ngập đường"),
        ("hard_chat_place", "Đường Nguyễn Huệ tối nay cấm xe đi bộ"),
        ("hard_chat_place", "Bạn mình quê ở Nam Định nhưng làm việc tại Bình Dương"),
        ("hard_chat_place", "Chi nhánh Thủ Đức mở cửa đến mấy giờ tối?"),
        ("hard_chat_place", "Cước vận chuyển GHTK đi Đắk Lắk tính sao shop"),
        ("hard_chat_place", "Tết năm nay mình về quê Thanh Hóa ăn Tết"),
        ("hard_chat_place", "Vé máy bay từ Sài Gòn ra Phú Quốc đang giảm giá"),
        ("hard_chat_place", "Sân bay Nội Bài hôm nay sương mù dày đặc"),
        ("hard_chat_place", "Khu vực Đồng Nai hay cúp điện vào cuối tuần"),
        ("hard_chat_place", "Mình đang tìm phòng trọ giá rẻ gần Gò Vấp"),
        ("hard_chat_place", "Trường đại học ở Thanh Xuân nghỉ lễ 4 ngày"),
        ("hard_chat_place", "Giá đất ở Long An dạo này tăng cao quá"),
        ("hard_chat_place", "Khách sạn ở Nha Trang mùa hè hay hết phòng"),
        ("hard_chat_place", "Xe khách chạy tuyến Hà Nội - Lào Cai mất 5 tiếng"),
        ("hard_chat_place", "Công ty tuyển nhân viên làm việc tại Cầu Giấy"),
        ("hard_chat_place", "Hàng gửi về Bến Tre có thu hộ COD không bạn"),
        ("hard_chat_place", "Bưu cục Quận 10 hôm nay nghỉ trưa lúc 12h"),
        ("hard_chat_place", "Em ở ngoài Bắc còn anh ấy ở trong Sài Gòn"),
        ("hard_chat_place", "Món ăn miền Tây ở Cần Thơ rất hợp khẩu vị"),
        ("hard_chat_place", "Cầu Rồng Đà Nẵng phun lửa vào tối thứ bảy"),
        ("hard_chat_place", "Mùa thu Hà Nội có hương hoa sữa rất thơm"),
        ("hard_chat_place", "Quê ngoại mình ở huyện Củ Chi trồng nhiều rau"),
        ("hard_chat_place", "Tỉnh Lâm Đồng nổi tiếng với nhiều thác nước đẹp"),
        ("hard_chat_place", "Giá vé tàu cao tốc đi Phú Quốc tăng nhẹ"),
        ("hard_chat_place", "Hôm nay kẹt xe suốt đoạn Cộng Hòa Tân Bình"),
        ("hard_chat_place", "Mình học cấp 3 ở Biên Hòa Đồng Nai"),
        ("hard_chat_place", "Lễ hội pháo hoa quốc tế tổ chức tại Đà Nẵng"),
        ("hard_chat_place", "Ship hỏa tốc nội thành TP.HCM hết bao nhiêu"),
        ("hard_chat_place", "Kho hàng của shop đặt ở Bắc Ninh hay Hưng Yên"),
        ("hard_chat_place", "Tối nay tụi mình họp nhóm ở quán cafe Hồ Tây"),
        ("hard_chat_place", "Bệnh viện Chợ Rẫy đông bệnh nhân lắm"),
        ("hard_chat_place", "Chúc các bạn ở Vũng Tàu cuối tuần vui vẻ"),
    ]
    for sg, txt in hard_neg_items:
        if g5_count >= 100:
            break
        if add_case(f"H5_{g5_count+1:03d}", "hard_negative", sg, "not_address", txt):
            g5_count += 1

    assert len(cases) == 600, f"Expected 600 cases, got {len(cases)}"
    payload = {
        "version": "2.0.0",
        "seed": SEED,
        "total_cases": len(cases),
        "group_counts": dict(Counter(c["group"] for c in cases)),
        "sufficiency_counts": dict(Counter(c["expected_sufficiency"] for c in cases)),
        "cases": cases,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Generated", OUT_PATH)
    print("Groups:", payload["group_counts"])
    print("Sufficiency:", payload["sufficiency_counts"])
    return payload


if __name__ == "__main__":
    build_holdout()
