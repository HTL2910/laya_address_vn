"""Dataset v2 Generator for Way A v2 (Laya Address VN — 3-Class Sufficiency).

Generates 90,000 stratified Vietnamese address / fragment / non-address samples
(seed = 101, 80% train / 10% val / 10% test) from MIT-licensed GSO + 2025
administrative unit records (`../shared_data/vn_admin_full_records.json`), with
strict anti-leakage against both v1 (`../benchmark/vn_address_benchmark.json`)
and v2 (`../benchmark_v2/independent_holdout_v2.json`) benchmarks.

Labels:
  0: not_address
  1: address_insufficient
  2: address_sufficient
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

TRAIN_DIR = Path(__file__).resolve().parent
BASE_DIR = TRAIN_DIR.parent
ROOT_DIR = BASE_DIR.parent

RECORDS_PATH = ROOT_DIR / "shared_data" / "vn_admin_full_records.json"
BENCH_V1_PATH = ROOT_DIR / "benchmark" / "vn_address_benchmark.json"
BENCH_V2_PATH = ROOT_DIR / "benchmark_v2" / "independent_holdout_v2.json"
MANIFEST_PATH = TRAIN_DIR / "train_texts_manifest.txt"

SEED = 101
TOTAL_SAMPLES = 90_000

LABEL_NAMES = ("not_address", "address_insufficient", "address_sufficient")

# ---------------------------------------------------------------------------
# Normalization & Anti-Leakage Helpers
# ---------------------------------------------------------------------------


def strip_accents(text: str) -> str:
    s = (text or "").replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def norm_key(text: str) -> str:
    s = strip_accents(text).lower().strip()
    return re.sub(r"\s+", " ", s)


def _token_lists(text: str) -> Tuple[List[str], List[str]]:
    nk = norm_key(text)
    space_tokens = nk.split(" ") if nk else []
    word_tokens = re.findall(r"[a-z0-9]+", nk)
    return space_tokens, word_tokens


def _extract_4grams(text: str) -> Set[Tuple[str, ...]]:
    grams: Set[Tuple[str, ...]] = set()
    for toks in _token_lists(text):
        if len(toks) >= 4:
            for i in range(len(toks) - 3):
                grams.add(tuple(toks[i : i + 4]))
    return grams


def load_eval_leakage_guards() -> Tuple[Set[str], Set[Tuple[str, ...]]]:
    forbidden_exact: Set[str] = set()
    forbidden_4grams: Set[Tuple[str, ...]] = set()

    for path in (BENCH_V1_PATH, BENCH_V2_PATH):
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        for case in raw.get("cases", []):
            txt = case.get("text", "")
            nk = norm_key(txt)
            if nk:
                forbidden_exact.add(nk)
                forbidden_4grams |= _extract_4grams(txt)

    return forbidden_exact, forbidden_4grams


# ---------------------------------------------------------------------------
# Lexical Pools (distinct phrasing & rich coverage)
# ---------------------------------------------------------------------------

PROVINCE_ABBR: Dict[str, List[str]] = {
    "hà nội": ["HN", "hn", "TP.HN", "Hà Nội", "TP. Hà Nội", "TP Hà Nội"],
    "hồ chí minh": ["HCM", "hcm", "TPHCM", "TP.HCM", "SG", "Sài Gòn", "TP. Hồ Chí Minh"],
    "đà nẵng": ["ĐN", "dn", "Đà Nẵng", "TP. Đà Nẵng"],
    "hải phòng": ["HP", "hp", "Hải Phòng", "TP. Hải Phòng"],
    "cần thơ": ["CT", "ct", "Cần Thơ", "TP. Cần Thơ"],
    "bình dương": ["BD", "bd", "Bình Dương", "Tỉnh Bình Dương"],
    "đồng nai": ["ĐNai", "dnai", "Đồng Nai", "Tỉnh Đồng Nai"],
    "bà rịa - vũng tàu": ["BRVT", "BR-VT", "Vũng Tàu", "Bà Rịa - Vũng Tàu"],
    "thừa thiên huế": ["Huế", "TP. Huế", "Thừa Thiên Huế"],
    "huế": ["Huế", "TP. Huế", "tthue"],
    "đắk lắk": ["Đắk Lắk", "Dak Lak", "daklak"],
    "bình phước": ["Bình Phước", "BP", "Tỉnh Bình Phước"],
}

DISTRICT_ABBR: Dict[str, List[str]] = {
    "tân bình": ["Q.TB", "q.tb", "qtb", "Tân Bình", "Quận Tân Bình"],
    "bình thạnh": ["Q.BT", "q.bt", "qbt", "Bình Thạnh", "Quận Bình Thạnh"],
    "phú nhuận": ["Q.PN", "q.pn", "qpn", "Phú Nhuận", "Quận Phú Nhuận"],
    "gò vấp": ["Q.GV", "q.gv", "qgv", "Gò Vấp", "Quận Gò Vấp"],
    "tân phú": ["Q.TP", "q.tp", "qtp", "Tân Phú", "Quận Tân Phú"],
    "thanh xuân": ["Q.TX", "q.tx", "qtx", "Thanh Xuân", "Quận Thanh Xuân"],
    "cầu giấy": ["Q.CG", "q.cg", "qcg", "Cầu Giấy", "Quận Cầu Giấy"],
    "đống đa": ["Q.ĐĐ", "q.dd", "qdd", "Đống Đa", "Quận Đống Đa"],
    "hoàn kiếm": ["Q.HK", "q.hk", "qhk", "Hoàn Kiếm", "Quận Hoàn Kiếm"],
    "hoàng mai": ["Q.HM", "q.hm", "qhm", "Hoàng Mai", "Quận Hoàng Mai"],
    "ba đình": ["Q.BĐ", "q.bd", "qbd", "Ba Đình", "Quận Ba Đình"],
    "hai bà trưng": ["Q.HBT", "HBT", "hbt", "Hai Bà Trưng", "Quận Hai Bà Trưng"],
    "long biên": ["Q.LB", "q.lb", "qlb", "Long Biên", "Quận Long Biên"],
    "tây hồ": ["Q.TH", "q.th", "Tây Hồ", "Quận Tây Hồ"],
    "hà đông": ["Q.HĐ", "q.hd", "Hà Đông", "Quận Hà Đông"],
    "nam từ liêm": ["NTL", "Q.NTL", "Nam Từ Liêm", "Quận Nam Từ Liêm"],
    "bắc từ liêm": ["BTL", "Q.BTL", "Bắc Từ Liêm", "Quận Bắc Từ Liêm"],
    "thủ đức": ["TP. Thủ Đức", "Thủ Đức", "Q. Thủ Đức", "TĐ"],
    "hải châu": ["Q.HC", "Hải Châu", "Quận Hải Châu"],
    "ninh kiều": ["Q.NK", "Ninh Kiều", "Quận Ninh Kiều"],
    "ngô quyền": ["Q.NQ", "Ngô Quyền", "Quận Ngô Quyền"],
}

STREETS = [
    "Nguyễn Trãi", "Lê Lợi", "Trần Hưng Đạo", "Hai Bà Trưng", "Lý Thường Kiệt",
    "Nguyễn Huệ", "Điện Biên Phủ", "Cách Mạng Tháng 8", "CMT8", "Võ Văn Tần",
    "Nguyễn Thị Minh Khai", "Lê Duẩn", "Pasteur", "Nam Kỳ Khởi Nghĩa", "Cộng Hòa",
    "Trường Chinh", "Hoàng Văn Thụ", "Phan Xích Long", "Phan Đăng Lưu",
    "Xô Viết Nghệ Tĩnh", "Ung Văn Khiêm", "Kha Vạn Cân", "Phạm Văn Đồng",
    "Võ Văn Ngân", "Nguyễn Văn Linh", "Huỳnh Tấn Phát", "Nguyễn Hữu Thọ",
    "Lạc Long Quân", "Âu Cơ", "Ba Tháng Hai", "Lý Tự Trọng", "Bùi Viện",
    "Xuân Thủy", "Hồ Tùng Mậu", "Phạm Hùng", "Trần Duy Hưng", "Nguyễn Chí Thanh",
    "Láng Hạ", "Thái Hà", "Chùa Bộc", "Tôn Đức Thắng", "Tây Sơn", "Giải Phóng",
    "Minh Khai", "Đại La", "Bạch Mai", "Phố Huế", "Kim Mã", "Liễu Giai",
    "Đội Cấn", "Thụy Khuê", "Hoàng Quốc Việt", "Nguyễn Văn Cừ", "Ngô Gia Tự",
    "Lê Văn Lương", "Tố Hữu", "Khuất Duy Tiến", "Nguyễn Xiển", "Tam Trinh",
    "Lĩnh Nam", "Trương Định", "Kim Ngưu", "Bà Triệu", "Quang Trung", "Hùng Vương",
    "Bạch Đằng", "Trần Phú", "Võ Nguyên Giáp", "Tôn Thất Tùng", "Phan Chu Trinh",
    "Phan Bội Châu", "Đinh Bộ Lĩnh", "Nguyễn Oanh", "Lê Đức Thọ", "Nguyễn Kiệm",
    "Hoàng Hoa Thám", "Lê Văn Sỹ", "Phan Văn Trị", "Quốc lộ 1A", "QL13", "QL22",
    "Tỉnh lộ 10", "Đại lộ Thăng Long", "Xa lộ Hà Nội", "Võ Chí Công", "Mai Chí Thọ",
    "Nguyễn Duy Trinh", "Đỗ Xuân Hợp", "Lê Văn Việt", "Lê Trọng Tấn", "Duy Tân",
    "Hàm Nghi", "Tô Hiến Thành", "Sư Vạn Hạnh", "An Dương Vương", "Nguyễn Đình Chiểu",
    "Võ Thị Sáu", "Kinh Dương Vương", "Nguyễn Văn Trỗi", "Nguyễn Thị Thập",
    "Đồng Khởi", "Lê Hồng Phong", "Nguyễn Ái Quốc", "Trần Đại Nghĩa", "Võ Văn Kiệt",
]

# Streets unambiguous enough for standalone "house + street" insufficient examples
UNAMBIGUOUS_STREETS = [
    "Hai Bà Trưng", "Lý Thường Kiệt", "Cách Mạng Tháng 8", "Võ Văn Tần",
    "Nguyễn Thị Minh Khai", "Nam Kỳ Khởi Nghĩa", "Phan Xích Long", "Phan Đăng Lưu",
    "Xô Viết Nghệ Tĩnh", "Ung Văn Khiêm", "Kha Vạn Cân", "Võ Văn Ngân",
    "Huỳnh Tấn Phát", "Nguyễn Hữu Thọ", "Ba Tháng Hai", "Trần Duy Hưng",
    "Nguyễn Chí Thanh", "Láng Hạ", "Thái Hà", "Chùa Bộc", "Kim Mã", "Liễu Giai",
    "Hoàng Quốc Việt", "Khuất Duy Tiến", "Nguyễn Xiển", "Bạch Đằng",
    "Phan Chu Trinh", "Phan Bội Châu", "Đinh Bộ Lĩnh", "Tô Hiến Thành",
    "Sư Vạn Hạnh", "Kinh Dương Vương", "Nguyễn Văn Trỗi", "Nguyễn Thị Thập",
]

BUILDINGS = [
    "Chung cư Vinhomes Central Park", "Vinhomes Ocean Park", "Vinhomes Smart City",
    "Chung cư Times City", "Chung cư Royal City", "Chung cư Masteri Thảo Điền",
    "Tòa nhà Keangnam", "Tòa nhà Lotte Center", "Chung cư Sunrise City",
    "KĐT Sala", "KĐT Ecopark", "Chung cư Goldmark City", "Chung cư HH Linh Đàm",
    "Chung cư Eco Green", "Chung cư Imperia Garden", "Chung cư The Manor",
    "Chung cư Estella Heights", "Chung cư Saigon Pearl", "Chung cư Topaz Home",
    "Tòa nhà Viettel", "Tòa nhà FPT", "Tòa nhà Etown", "Chung cư Mizuki Park",
    "Chung cư Akari City", "Chung cư Celadon City", "Chung cư An Bình City",
    "Chung cư Scenic Valley", "Chung cư Gateway Thảo Điền", "Tòa nhà Bitexco",
]

KCN_POOL = [
    "KCN Tân Bình", "KCN VSIP 1", "KCN VSIP 2", "KCN Amata", "KCN Biên Hòa 2",
    "KCN Bắc Thăng Long", "KCN Quang Minh", "KCN Quế Võ", "KCN Yên Phong",
    "KCN Sóng Thần 1", "KCN Sóng Thần 2", "KCN Tân Tạo", "KCN Hiệp Phước",
    "KCX Tân Thuận", "KCN Mỹ Phước 3", "KCN Nhơn Trạch", "KCN Long Hậu",
    "KCN Tràng Duệ", "KCN Hòa Khánh", "KCN Đình Vũ", "KCN Phố Nối A",
]

# Person-name / historical traps that should NEVER be treated as bare ward_only
PERSON_TRAPS = [
    "Nguyễn Trãi", "Lê Lợi", "Trần Hưng Đạo", "Hai Bà Trưng", "Lý Thường Kiệt",
    "Nguyễn Huệ", "Quang Trung", "Đinh Tiên Hoàng", "Phan Bội Châu", "Phan Chu Trinh",
    "Nguyễn Thị Minh Khai", "Tôn Đức Thắng", "Võ Nguyên Giáp", "Lê Duẩn", "Hoàng Diệu",
    "Ngô Quyền", "Lý Tự Trọng", "Nguyễn Du", "Hồ Xuân Hương", "Nguyễn Công Trứ",
    "Nguyễn Văn Cừ", "Trần Phú", "Võ Thị Sáu", "Nguyễn Đình Chiểu", "Lê Quý Đôn",
    "Hùng Vương", "Phạm Ngũ Lão", "Trần Quốc Toản", "Lê Hồng Phong", "Hoàng Hoa Thám",
    "Nguyễn Thị Thanh Xuân", "Trần Văn Phú", "Lê Hoàng Mai", "Phạm Văn Đồng",
]

SURNAMES = [
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ",
    "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý", "Đinh", "Đào", "Trương",
]
MID_M = ["Văn", "Minh", "Hoàng", "Quang", "Đức", "Thanh", "Gia", "Tuấn", "Hữu", "Quốc"]
MID_F = ["Thị", "Ngọc", "Thị Mai", "Thị Thu", "Khánh", "Phương", "Thanh", "Thùy", "Hồng"]
GIVEN_M = [
    "Hùng", "Tuấn", "Dũng", "Minh", "Nam", "Long", "Khang", "Bảo", "Huy", "Đạt",
    "Quân", "Phong", "Tài", "Sơn", "Thành", "Hiếu", "Kiên", "Thắng", "Việt", "Phú",
]
GIVEN_F = [
    "Linh", "Hương", "Lan", "Trang", "Thảo", "Anh", "Ngân", "Vy", "Hà", "Yến",
    "Nhi", "Quỳnh", "Phương", "Hạnh", "Mai", "Châu", "Trinh", "My", "Hoa", "Xuân",
]

FOOD_PRODUCT_PLACE = [
    ("bánh mì", "Sài Gòn"), ("mực", "Cà Mau"), ("mực khô", "Cà Mau"), ("mực một nắng", "Phan Thiết"),
    ("phở", "Hà Nội"), ("bún bò", "Huế"), ("nước mắm", "Phú Quốc"),
    ("kẹo dừa", "Bến Tre"), ("nem chua", "Thanh Hóa"), ("chè", "Thái Nguyên"),
    ("chè tân cương", "Thái Nguyên"), ("trà tân cương", "Thái Nguyên"),
    ("cơm tấm", "Sài Gòn"), ("bánh tráng", "Tây Ninh"), ("cà phê", "Buôn Ma Thuột"),
    ("bánh cuốn", "Thanh Trì"), ("nem nướng", "Nha Trang"), ("hủ tiếu", "Mỹ Tho"),
    ("bánh đậu xanh", "Hải Dương"), ("vải thiều", "Bắc Giang"), ("bánh pía", "Sóc Trăng"),
    ("mắm tôm", "Thanh Hóa"), ("bò tơ", "Củ Chi"), ("dừa sáp", "Trà Vinh"),
    ("tỏi", "Lý Sơn"), ("tỏi đen", "Lý Sơn"), ("bánh đa cua", "Hải Phòng"),
    ("mì quảng", "Quảng Nam"), ("cao lầu", "Hội An"), ("cá kho", "Vũ Đại"),
    ("bún đậu", "Hà Nội"), ("bánh xèo", "Miền Tây"), ("chả mực", "Hạ Long"),
    ("bưởi", "Diễn"), ("cam", "Cao Phong"), ("nhãn lồng", "Hưng Yên"),
    ("xoài cát", "Hòa Lộc"), ("cốm", "Làng Vòng"), ("bánh cốm", "Hà Nội"),
    ("yến sào", "Khánh Hòa"), ("trà sen", "Tây Hồ"), ("gà đồi", "Yên Thế"),
    ("thịt trâu gác bếp", "Sơn La"), ("lạp xưởng", "Cao Bằng"), ("bánh chưng", "Bờ Đậu"),
    ("mè xửng", "Huế"), ("tôm khô", "Cà Mau"), ("cá lóc nướng", "Cần Thơ"),
    ("bún cá", "Châu Đốc"), ("bánh ít lá gai", "Bình Định"), ("rượu cần", "Tây Nguyên"),
    ("rượu sim", "Phú Quốc"), ("hạt điều", "Bình Phước"), ("tiêu đen", "Phú Quốc"),
    ("sầu riêng", "Tiền Giang"), ("thanh long", "Bình Thuận"), ("gạo ST25", "Sóc Trăng"),
    ("cua biển", "Năm Căn"), ("mật ong", "U Minh"), ("bưởi da xanh", "Bến Tre"),
]

ECOM_PRODUCTS_GARMENTS = [
    "áo khoác kaki", "áo sơ mi lụa", "áo thun cotton", "áo len cổ lọ", "áo chống nắng",
    "quần jean ống rộng", "quần tây công sở", "váy hoa nhí", "đầm dự tiệc",
    "túi xách nữ da bò", "ví da nam cầm tay", "balo laptop chống sốc", "thắt lưng da",
    "tai nghe không dây", "sạc dự phòng 20000mah", "cáp sạc nhanh type c",
    "chuột bluetooth không dây", "bàn phím cơ mini", "loa bluetooth mini",
    "ốp lưng điện thoại", "kính cường lực", "giá đỡ laptop nhôm",
    "giày thể thao nam", "giày cao gót mũi nhọn", "dép quai ngang", "mũ lưỡi trai kaki",
    "kính mát thời trang", "đồng hồ dây da", "bình giữ nhiệt inox", "nồi cơm điện mini",
    "máy sấy tóc ion", "kem chống nắng", "sữa rửa mặt dịu nhẹ", "son lì dưỡng môi",
    "khăn choàng cổ len", "vớ cổ cao cotton", "bộ ga gối cotton", "rèm cửa chống nắng",
]

ECOM_ATTRS = [
    "", "chính hãng", "loại 1", "giá rẻ", "cao cấp", "màu đen", "màu trắng",
    "màu be", "màu xanh navy", "size M", "size L", "size XL", "freesize",
    "bản mới", "siêu nhẹ", "chống nước", "nguyên hộp", "fullbox",
]

CHAT_CONFIRMATIONS_SHORT = [
    "vâng ạ", "dạ vâng ạ", "ok shop", "ok bạn nhé", "xin chào shop", "chào bạn ạ",
    "cảm ơn nhiều nha", "cảm ơn shop nhiều ạ", "nhận được rồi ạ", "đã kiểm tra rồi",
    "chốt đơn này nha", "tư vấn giúp mình", "gửi ảnh thật giúp em", "xin giá ạ",
    "còn hàng không shop", "giá bao nhiêu vậy", "mẫu này có sẵn không", "đổi size giúp em",
    "chờ mình chút xíu", "nhắn tin lại sau nhé", "bận họp tới chiều", "ok rồi đó bạn",
    "mình chuyển khoản rồi", "check inbox giúp em", "tính tiền giúp mình", "lấy cái này nha",
]

NON_ADDR_DELIVERY_COMMENTS = [
    "để ngoài cổng giùm", "bảo vệ ký nhận hộ", "gọi trước khi tới nhé",
    "giao giờ hành chính giúp em", "nhờ lễ tân nhận giúp mình", "không bấm chuông cửa",
    "treo ở tay nắm cửa nha", "sau 18h mình mới về nhà", "gửi hàng xóm nhận hộ",
    "tới đầu ngõ gọi mình ra", "cho xem hàng trước khi trả", "đóng gói kỹ giúp mình",
    "hàng dễ vỡ xin nhẹ tay", "ship hỏa tốc giúp em với", "giao cuối tuần giúp mình",
    "tới nơi nhá máy là được", "để ở bàn bảo vệ tầng trệt", "không giao được buổi sáng",
]

SHORT_GARBAGE_TOKENS = [
    "73", "10", "K", "Hi", "Ok", "Jean", "Xom", "ok", "ko", "ib", "check", "test",
    "abc", "xyz", "null", "none", "N/A", "alo", "yes", "no", "shop", "done",
    "chot", "gui", "ship", "huy", "doi", "size", "xl", "xxl", "m", "l", "s",
    "0", "1", "2", "5", "99", "100", "200", "???", "---", "...", "hi shop",
    "ok b", "tks", "cam on", "da gui", "cho xiu", "k co", "chuoi", "khong",
]

VAGUE_NON_ADDRESS = [
    "ở chỗ cũ", "như đơn lần trước", "giao chỗ hôm qua", "địa chỉ cũ nha shop",
    "nhắn địa chỉ sau nhé", "tí mình gửi địa chỉ", "qua lấy trực tiếp ở kho",
    "giao chỗ mọi khi", "vẫn chỗ tuần trước nha", "tới đầu hẻm mình ra lấy",
    "gần nhà văn hóa cũ", "đối diện cổng trường cũ", "không cần giao hàng",
    "đặt tại quầy rồi", "như thông tin cũ trên hệ thống",
]


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------


def _rand_house(rng: random.Random) -> str:
    r = rng.random()
    n1 = rng.randint(1, 780)
    if r < 0.42:
        return str(n1)
    if r < 0.62:
        return f"{n1}{rng.choice(['A', 'B', 'C', 'D', 'a', 'b'])}"
    if r < 0.85:
        n2 = rng.randint(1, 99)
        sfx = rng.choice(["", "", "A", "B", "C"])
        return f"{n1}/{n2}{sfx}"
    n2 = rng.randint(1, 55)
    n3 = rng.randint(1, 30)
    return f"{n1}/{n2}/{n3}"


def _rand_phone(rng: random.Random) -> str:
    pfx = rng.choice(["09", "03", "07", "08", "05"])
    tail = "".join(str(rng.randint(0, 9)) for _ in range(8))
    style = rng.random()
    if style < 0.5:
        return f"{pfx}{tail}"
    if style < 0.8:
        return f"{pfx}{tail[:2]}.{tail[2:5]}.{tail[5:]}"
    return f"+84 {pfx[1]}{tail[:2]} {tail[2:5]} {tail[5:]}"


def _apply_case(rng: random.Random, s: str, p_lower: float = 0.28, p_upper: float = 0.08, p_mixed: float = 0.10) -> str:
    r = rng.random()
    if r < p_lower:
        return s.lower()
    if r < p_lower + p_upper:
        return s.upper()
    if r < p_lower + p_upper + p_mixed:
        words = s.split(" ")
        return " ".join(w.upper() if rng.random() < 0.3 else w.lower() for w in words)
    return s


def _abbr_ward(rng: random.Random, ward_full: str, ward_bare: str) -> str:
    low = norm_key(ward_full)
    bare_clean = ward_bare.strip()
    is_num = bare_clean.isdigit() or low.startswith("phuong ") and low[7:].strip().isdigit()
    if is_num:
        num = re.search(r"\d+", ward_full)
        n = num.group(0) if num else bare_clean
        return rng.choice([f"p.{n}", f"P.{n}", f"p{n}", f"P{n}", f"p. {n}", f"Phường {n}"])
    if low.startswith("xa "):
        return rng.choice([f"x. {bare_clean}", f"x.{bare_clean}", f"X.{bare_clean}", f"xa {bare_clean}"])
    if low.startswith("thi tran "):
        return rng.choice([f"tt. {bare_clean}", f"tt.{bare_clean}", f"TT.{bare_clean}", f"tt {bare_clean}"])
    if low.startswith("dac khu "):
        return rng.choice([f"dk {bare_clean}", f"đk. {bare_clean}", f"Đặc khu {bare_clean}"])
    return rng.choice([f"p. {bare_clean}", f"p.{bare_clean}", f"P.{bare_clean}", f"p {bare_clean}"])


def _abbr_district(rng: random.Random, dist_full: str, dist_bare: str) -> str:
    low = norm_key(dist_full)
    bare_norm = norm_key(dist_bare)
    if bare_norm in DISTRICT_ABBR and rng.random() < 0.75:
        return rng.choice(DISTRICT_ABBR[bare_norm])
    if low.startswith("quan "):
        rest = low[5:].strip()
        if rest.isdigit():
            return rng.choice([f"q.{rest}", f"q{rest}", f"Q.{rest}", f"Q{rest}", f"q. {rest}"])
        return rng.choice([f"q. {dist_bare}", f"q.{dist_bare}", f"Q.{dist_bare}"])
    if low.startswith("huyen "):
        return rng.choice([f"h. {dist_bare}", f"h.{dist_bare}", f"H.{dist_bare}", f"h {dist_bare}"])
    if low.startswith("thi xa "):
        return rng.choice([f"tx. {dist_bare}", f"tx.{dist_bare}", f"TX.{dist_bare}"])
    if low.startswith("thanh pho "):
        return rng.choice([f"tp. {dist_bare}", f"tp.{dist_bare}", f"TP.{dist_bare}"])
    return dist_bare


def _abbr_province(rng: random.Random, prov_full: str, prov_bare: str) -> str:
    key = norm_key(prov_bare)
    if key in PROVINCE_ABBR and rng.random() < 0.80:
        return rng.choice(PROVINCE_ABBR[key])
    low = norm_key(prov_full)
    if low.startswith("thanh pho "):
        return rng.choice([f"TP.{prov_bare}", f"tp {prov_bare}", prov_bare])
    return rng.choice([prov_bare, f"t.{prov_bare}", f"Tỉnh {prov_bare}"])


def _safe_ward_bare(rec: Dict[str, str]) -> str:
    wb = rec["ward_bare"].strip()
    if wb.isdigit():
        return f"Phường {wb}"
    return wb


def _safe_ward_for_comma_pair(rec: Dict[str, str], parent_bare: str) -> str:
    wb = _safe_ward_bare(rec)
    if norm_key(wb) == norm_key(parent_bare):
        return rec["ward_full"].strip()
    return wb


def _safe_dist_for_prov_pair(rec: Dict[str, str]) -> str:
    db = rec["district_bare"].strip()
    pb = rec["province_bare"].strip()
    if norm_key(db) == norm_key(pb):
        return rec["district_full"].strip()
    return db


# ---------------------------------------------------------------------------
# Dataset Generation Engine
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    text: str
    label: int
    subgroup: str


@dataclass
class DatasetSplitsV2:
    train: List[Sample]
    val: List[Sample]
    test: List[Sample]


def _make_person_name(rng: random.Random, allow_admin_trap: bool = False) -> str:
    if allow_admin_trap and rng.random() < 0.35:
        trap = rng.choice(PERSON_TRAPS)
        if rng.random() < 0.3:
            trap = strip_accents(trap)
        return _apply_case(rng, trap, p_lower=0.25, p_upper=0.05, p_mixed=0.05)

    sn = rng.choice(SURNAMES)
    if rng.random() < 0.5:
        mid = rng.choice(MID_M)
        giv = rng.choice(GIVEN_M)
    else:
        mid = rng.choice(MID_F)
        giv = rng.choice(GIVEN_F)
    if allow_admin_trap and rng.random() < 0.3:
        giv = rng.choice(["Thanh Xuân", "Hải Châu", "Long Biên", "Bình Tân", "Minh Khai", "Quang Trung", "Phú Nhuận", "Đông Anh"])
    name = f"{sn} {mid} {giv}"
    if rng.random() < 0.14:
        name = f"{rng.choice(['Anh ', 'Chị ', 'Em ', 'Cô ', 'Chú ', 'Mr. ', 'Ms. '])}{name}"
    if rng.random() < 0.32:
        name = strip_accents(name)
    return _apply_case(rng, name, p_lower=0.25, p_upper=0.05, p_mixed=0.04)


def generate_dataset_v2(
    total_samples: int = TOTAL_SAMPLES,
    seed: int = SEED,
    write_manifest: bool = True,
) -> DatasetSplitsV2:
    rng = random.Random(seed)
    records = json.loads(RECORDS_PATH.read_text(encoding="utf-8"))
    legacy = list(records["legacy_3tier"])
    merged = list(records["merged_2025_2tier"])
    rng.shuffle(legacy)
    rng.shuffle(merged)

    # Deduplicated pools of provinces, districts, wards for clean sampling
    all_provs: List[Tuple[str, str]] = []
    seen_p: Set[str] = set()
    for r in legacy + merged:
        k = norm_key(r["province_bare"])
        if k not in seen_p:
            seen_p.add(k)
            all_provs.append((r["province_full"], r["province_bare"]))

    all_dists: List[Tuple[str, str, str, str]] = []
    seen_d: Set[str] = set()
    for r in legacy:
        k = f"{norm_key(r['district_bare'])}|{norm_key(r['province_bare'])}"
        if k not in seen_d:
            seen_d.add(k)
            all_dists.append((r["district_full"], r["district_bare"], r["province_full"], r["province_bare"]))

    all_wards_bare: List[str] = []
    seen_wb: Set[str] = set()
    person_trap_keys = {norm_key(p) for p in PERSON_TRAPS}
    for r in legacy + merged:
        wb = _safe_ward_bare(r)
        wk = norm_key(wb)
        # Skip 1-word bare names, digits, and famous historical person names for bare ward_only
        if len(wk.split(" ")) >= 2 and wk not in person_trap_keys and wk not in seen_wb:
            seen_wb.add(wk)
            all_wards_bare.append(wb)

    all_wards_full: List[Tuple[str, str]] = []
    seen_wf: Set[str] = set()
    for r in legacy + merged:
        wk = norm_key(r["ward_full"])
        if wk not in seen_wf:
            seen_wf.add(wk)
            all_wards_full.append((r["ward_full"], _safe_ward_bare(r)))

    rng.shuffle(all_provs)
    rng.shuffle(all_dists)
    rng.shuffle(all_wards_bare)
    rng.shuffle(all_wards_full)

    forbidden_exact, forbidden_4grams = load_eval_leakage_guards()
    seen_global: Set[str] = set(forbidden_exact)

    # Scale subgroup targets cleanly to match total_samples
    scale = total_samples / 90_000.0
    subgroups_spec: List[Tuple[int, str, int]] = [
        # --- Label 2: address_sufficient (~45% -> 40,500) ---
        (2, "short_style1_ward_dist_comma", int(round(3200 * scale))),
        (2, "short_style2_ward_dist_nocomma", int(round(3200 * scale))),
        (2, "short_style3_unaccented_lower", int(round(3500 * scale))),
        (2, "short_style4_abbreviated", int(round(3600 * scale))),
        (2, "short_style5_3tier", int(round(3500 * scale))),
        (2, "short_style6_2tier", int(round(3500 * scale))),
        (2, "full_urban", int(round(5500 * scale))),
        (2, "full_rural", int(round(4500 * scale))),
        (2, "full_apartment", int(round(3000 * scale))),
        (2, "full_deep_alley", int(round(3000 * scale))),
        (2, "full_noisy_case", int(round(2000 * scale))),
        (2, "full_delivery_freetext", int(round(2000 * scale))),
        # --- Label 1: address_insufficient (~25% -> 22,500) ---
        (1, "insuff_ward_prefixed", int(round(5200 * scale))),
        (1, "insuff_ward_bare", int(round(3000 * scale))),
        (1, "insuff_province_only", int(round(350 * scale))),
        (1, "insuff_district_only", int(round(2900 * scale))),
        (1, "insuff_district_province", int(round(5050 * scale))),
        (1, "insuff_bare_granular", int(round(4000 * scale))),
        (1, "insuff_house_street_only", int(round(2000 * scale))),
        # --- Label 0: not_address (~30% -> 27,000) ---
        (0, "neg_short_garbage", int(round(3000 * scale))),
        (0, "neg_chat_order", int(round(6500 * scale))),
        (0, "neg_food_product_place", int(round(5500 * scale))),
        (0, "neg_person_name", int(round(3500 * scale))),
        (0, "neg_narrative_weather_travel", int(round(5000 * scale))),
        (0, "neg_structured_id_price_date", int(round(2200 * scale))),
        (0, "neg_vague_non_addr", int(round(1300 * scale))),
    ]

    # Adjust any rounding drift so exact total matches total_samples
    drift = total_samples - sum(c for _, _, c in subgroups_spec)
    if drift != 0:
        lbl, sg, cnt = subgroups_spec[6]
        subgroups_spec[6] = (lbl, sg, cnt + drift)

    # Precompute deterministic province-only surface forms across all 63 provinces + aliases
    prov_only_pool: List[str] = []
    for p_full, p_bare in all_provs:
        key = norm_key(p_bare)
        variants = [p_bare, p_full]
        if norm_key(p_full).startswith("thanh pho "):
            variants.extend([f"tp {p_bare}", f"tp.{p_bare}", f"thành phố {p_bare}"])
        else:
            variants.extend([f"t.{p_bare}", f"tinh {p_bare}"])
        if key in PROVINCE_ABBR:
            variants.extend(PROVINCE_ABBR[key])
        prov_only_pool.extend(variants)
    rng.shuffle(prov_only_pool)

    cursors = defaultdict(int)

    def next_legacy() -> Dict[str, str]:
        i = cursors["legacy"]
        cursors["legacy"] += 1
        return legacy[i % len(legacy)]

    def next_merged() -> Dict[str, str]:
        i = cursors["merged"]
        cursors["merged"] += 1
        return merged[i % len(merged)]

    def gen_one(label: int, sg: str) -> str:
        # -------- Label 2 --------
        if sg == "short_style1_ward_dist_comma":
            r = next_legacy()
            db = r["district_bare"].strip()
            wb = _safe_ward_for_comma_pair(r, db)
            raw = f"{wb}, {db}" if rng.random() < 0.85 else f"{wb},{db}"
            return _apply_case(rng, raw, p_lower=0.18, p_upper=0.08, p_mixed=0.12)

        if sg == "short_style2_ward_dist_nocomma":
            r = next_legacy()
            db = r["district_bare"].strip()
            wb = _safe_ward_for_comma_pair(r, db)
            raw = f"{wb} {db}"
            return _apply_case(rng, raw, p_lower=0.22, p_upper=0.08, p_mixed=0.12)

        if sg == "short_style3_unaccented_lower":
            choice = rng.random()
            if choice < 0.40:
                r = next_legacy()
                db = r["district_bare"].strip()
                wb = _safe_ward_for_comma_pair(r, db)
                raw = f"{wb} {db}"
            elif choice < 0.72:
                r = next_legacy()
                db = _safe_dist_for_prov_pair(r)
                wb = _safe_ward_for_comma_pair(r, db)
                raw = f"{wb} {db} {r['province_bare']}"
            else:
                r = next_merged()
                wb = _safe_ward_for_comma_pair(r, r["province_bare"])
                w_str = r["ward_full"] if rng.random() < 0.45 else wb
                raw = f"{w_str} {r['province_bare']}"
            return strip_accents(raw).lower()

        if sg == "short_style4_abbreviated":
            if rng.random() < 0.14:
                # Classic HCM / HN compact shorthand
                w_num = rng.randint(1, 16)
                p_str = rng.choice([f"p.{w_num}", f"p{w_num}", f"P.{w_num}", f"P{w_num}"])
                q_str = rng.choice(["q.tb", "qtb", "q.bt", "q.gv", "q.pn", "q.tp", "q.1", "q1", "q.3", "q3", "q.5", "q.7", "q.10", "q10"])
                c_str = rng.choice(["", " hcm", " tphcm", " sg"])
                raw = f"{p_str} {q_str}{c_str}".strip()
            else:
                r = next_legacy()
                w_a = _abbr_ward(rng, r["ward_full"], _safe_ward_bare(r))
                d_a = _abbr_district(rng, r["district_full"], r["district_bare"])
                sep = rng.choice([", ", " ", ","])
                if rng.random() < 0.55:
                    raw = f"{w_a}{sep}{d_a}"
                else:
                    p_a = _abbr_province(rng, r["province_full"], r["province_bare"])
                    raw = f"{w_a}{sep}{d_a}{sep}{p_a}"
            if rng.random() < 0.55:
                raw = strip_accents(raw)
            return _apply_case(rng, raw, p_lower=0.45, p_upper=0.06, p_mixed=0.10)

        if sg == "short_style5_3tier":
            r = next_legacy()
            use_full = rng.random()
            if use_full < 0.35:
                raw = f"{r['ward_full']}, {r['district_full']}, {r['province_full']}"
            elif use_full < 0.72:
                db = _safe_dist_for_prov_pair(r)
                wb = _safe_ward_for_comma_pair(r, db)
                raw = f"{wb}, {db}, {r['province_bare']}"
            else:
                db = _safe_dist_for_prov_pair(r)
                wb = _safe_ward_for_comma_pair(r, db)
                raw = f"{wb} {db}, {r['province_bare']}"
            return _apply_case(rng, raw, p_lower=0.20, p_upper=0.06, p_mixed=0.12)

        if sg == "short_style6_2tier":
            r = next_merged() if rng.random() < 0.70 else next_legacy()
            wb = _safe_ward_for_comma_pair(r, r["province_bare"])
            w_part = r["ward_full"] if rng.random() < 0.48 else wb
            p_part = r["province_full"] if rng.random() < 0.25 else _abbr_province(rng, r["province_full"], r["province_bare"])
            sep = rng.choice([", ", ", ", " "])
            raw = f"{w_part}{sep}{p_part}"
            if rng.random() < 0.30:
                raw = strip_accents(raw)
            return _apply_case(rng, raw, p_lower=0.25, p_upper=0.06, p_mixed=0.10)

        if sg == "full_urban":
            r = next_legacy() if rng.random() < 0.75 else next_merged()
            st = rng.choice(STREETS)
            hnum = _rand_house(rng)
            num_pfx = rng.choice(["Số ", "Số nhà ", "SN ", "", "", ""])
            st_pfx = rng.choice(["Đường ", "Phố ", "đ. ", "", ""])
            w_part = r["ward_full"] if rng.random() < 0.55 else _abbr_ward(rng, r["ward_full"], _safe_ward_bare(r))
            p_part = _abbr_province(rng, r["province_full"], r["province_bare"])
            if "district_full" in r:
                d_part = r["district_full"] if rng.random() < 0.6 else _abbr_district(rng, r["district_full"], r["district_bare"])
                if rng.random() < 0.78:
                    raw = f"{num_pfx}{hnum} {st_pfx}{st}, {w_part}, {d_part}, {p_part}"
                else:
                    raw = f"{num_pfx}{hnum} {st_pfx}{st}, {d_part}, {p_part}"
            else:
                raw = f"{num_pfx}{hnum} {st_pfx}{st}, {w_part}, {p_part}"
            if rng.random() < 0.35:
                raw = strip_accents(raw)
            if rng.random() < 0.30:
                raw = raw.replace(", ", " ")
            return _apply_case(rng, raw, p_lower=0.28, p_upper=0.05, p_mixed=0.08)

        if sg == "full_rural":
            r = next_legacy() if rng.random() < 0.72 else next_merged()
            unit = rng.choice([
                f"Thôn {rng.randint(1, 14)}",
                f"Thôn {rng.choice(['Đông', 'Đoài', 'Trung', 'Thượng', 'Hạ', 'Tân Lập', 'Bình Minh', 'Phú Hòa'])}",
                f"Ấp {rng.randint(1, 9)}",
                f"Ấp {rng.choice(['Tân Hòa', 'Bình An', 'Phước Lợi', 'An Khánh'])}",
                f"Xóm {rng.randint(1, 15)}",
                f"Bản {rng.choice(['Lác', 'Pó', 'Sài Khao', 'Mường', 'Hua Tạt', 'Cát Cát'])}",
                f"Khóm {rng.randint(1, 8)}",
                f"Tổ {rng.randint(1, 24)}",
                f"Tổ dân phố {rng.randint(1, 18)}",
                f"KP {rng.randint(1, 9)}",
                f"Lô {rng.choice(['A','B','C','D'])}{rng.randint(1,8)}-{rng.randint(1,15)} {rng.choice(KCN_POOL)}",
            ])
            sn = f"Số {rng.randint(1, 120)} " if rng.random() < 0.35 else ""
            w_part = r["ward_full"] if rng.random() < 0.7 else _abbr_ward(rng, r["ward_full"], _safe_ward_bare(r))
            p_part = r["province_bare"] if rng.random() < 0.7 else r["province_full"]
            if "district_full" in r:
                d_part = r["district_full"] if rng.random() < 0.65 else r["district_bare"]
                raw = f"{sn}{unit}, {w_part}, {d_part}, {p_part}"
            else:
                raw = f"{sn}{unit}, {w_part}, {p_part}"
            if rng.random() < 0.32:
                raw = strip_accents(raw)
            if rng.random() < 0.25:
                raw = raw.replace(", ", " ")
            return _apply_case(rng, raw, p_lower=0.26, p_upper=0.05, p_mixed=0.08)

        if sg == "full_apartment":
            r = next_legacy() if rng.random() < 0.8 else next_merged()
            bldg = rng.choice(BUILDINGS)
            unit = rng.choice([
                f"Căn hộ {rng.choice(['A','B','C','S1','S2'])}.{rng.randint(1,9)}-{rng.randint(101,3512)}",
                f"Phòng {rng.randint(102, 2815)}, Tầng {rng.randint(2, 28)}",
                f"Lầu {rng.randint(1, 18)}, {rng.choice(['Tháp A', 'Tháp B', 'Block C', 'Tòa R3'])}",
                f"Block {rng.choice(['A','B','C','D'])} căn {rng.randint(101, 2410)}",
                f"P{rng.randint(201, 2908)} Tòa {rng.choice(['S1.03', 'S2.01', 'T4', 'A2', 'B1'])}",
            ])
            st = rng.choice(STREETS)
            p_part = _abbr_province(rng, r["province_full"], r["province_bare"])
            if "district_full" in r:
                d_part = r["district_bare"] if rng.random() < 0.55 else r["district_full"]
                if rng.random() < 0.5:
                    raw = f"{unit}, {bldg}, {r['ward_full']}, {d_part}, {p_part}"
                else:
                    raw = f"{unit}, {bldg}, {_rand_house(rng)} {st}, {d_part}, {p_part}"
            else:
                raw = f"{unit}, {bldg}, {r['ward_full']}, {p_part}"
            if rng.random() < 0.30:
                raw = strip_accents(raw)
            return _apply_case(rng, raw, p_lower=0.22, p_upper=0.05, p_mixed=0.08)

        if sg == "full_deep_alley":
            r = next_legacy() if rng.random() < 0.8 else next_merged()
            st = rng.choice(STREETS)
            kw = rng.choice(["Ngõ", "Hẻm", "Ngách", "Kiệt"])
            a1 = rng.randint(1, 350)
            a2 = rng.randint(1, 50)
            a3 = rng.randint(1, 35)
            pattern = rng.random()
            if pattern < 0.35:
                prefix = f"Số {a3} {kw} {a1}/{a2} {st}"
            elif pattern < 0.65:
                prefix = f"{a1}/{a2}/{a3} {st}"
            elif pattern < 0.85:
                prefix = f"Kiệt {a1}/{a2}{rng.choice(['', 'A', 'B'])} {st}"
            else:
                prefix = f"Ngách {a2}/{a3} Ngõ {a1} {st}"
            p_part = _abbr_province(rng, r["province_full"], r["province_bare"])
            if "district_full" in r:
                d_part = r["district_bare"] if rng.random() < 0.5 else r["district_full"]
                raw = f"{prefix}, {r['ward_bare']}, {d_part}, {p_part}" if rng.random() < 0.55 else f"{prefix}, {d_part}, {p_part}"
            else:
                raw = f"{prefix}, {r['ward_full']}, {p_part}"
            if rng.random() < 0.35:
                raw = strip_accents(raw)
            return _apply_case(rng, raw, p_lower=0.30, p_upper=0.06, p_mixed=0.08)

        if sg == "full_noisy_case":
            r = next_legacy()
            st = rng.choice(STREETS)
            hnum = _rand_house(rng)
            raw = f"{hnum} {st}, {r['ward_full']}, {r['district_bare']}, {r['province_bare']}"
            if rng.random() < 0.5:
                # Partial accent strip
                tokens = raw.split(" ")
                raw = " ".join(strip_accents(t) if rng.random() < 0.55 else t for t in tokens)
            else:
                raw = strip_accents(raw)
            raw = raw.replace(", ", rng.choice([",", " ", "  ", "-"]))
            raw = raw.replace(". ", ".")
            return _apply_case(rng, raw, p_lower=0.25, p_upper=0.25, p_mixed=0.45)

        if sg == "full_delivery_freetext":
            r = next_legacy() if rng.random() < 0.78 else next_merged()
            cust = _make_person_name(rng, allow_admin_trap=False)
            phone = _rand_phone(rng)
            note = rng.choice([
                "gọi trước khi giao 15p", "giao giờ hành chính nha",
                "gửi bảo vệ nếu k nghe máy", "ship nhanh trong ngày",
                "nhờ anh shipper bấm chuông", "giao sau 18h giúp mình",
            ])
            st = rng.choice(STREETS)
            hnum = _rand_house(rng)
            if "district_bare" in r:
                addr = (
                    f"{_safe_ward_bare(r)}, {r['district_bare']}, {r['province_bare']}"
                    if rng.random() < 0.35
                    else f"{hnum} {st}, {r['district_bare']}, {r['province_bare']}"
                )
            else:
                addr = f"{hnum} {st}, {r['ward_full']}, {r['province_bare']}"
            tpl = rng.choice([
                f"giao cho {cust} sdt {phone} {addr}",
                f"ship về {addr} ({cust} - {phone}, {note})",
                f"đ/c: {addr}, người nhận {cust} {phone}, {note}",
                f"{cust} {phone} - {addr} - {note}",
            ])
            if rng.random() < 0.45:
                tpl = strip_accents(tpl)
            return _apply_case(rng, tpl, p_lower=0.35, p_upper=0.05, p_mixed=0.08)

        # -------- Label 1: address_insufficient --------
        if sg == "insuff_ward_prefixed":
            i = cursors["w_full"]
            cursors["w_full"] += 1
            w_full, w_bare = all_wards_full[i % len(all_wards_full)]
            style = rng.random()
            if style < 0.42:
                raw = w_full
            elif style < 0.72:
                raw = _abbr_ward(rng, w_full, w_bare)
            else:
                # Numbered ward variants
                n = rng.randint(1, 20)
                raw = rng.choice([f"Phường {n}", f"p.{n}", f"P.{n}", f"p{n}", f"phường {n}"])
            if rng.random() < 0.38:
                raw = strip_accents(raw)
            return _apply_case(rng, raw, p_lower=0.35, p_upper=0.06, p_mixed=0.08)

        if sg == "insuff_ward_bare":
            i = cursors["w_bare"]
            cursors["w_bare"] += 1
            wb = all_wards_bare[i % len(all_wards_bare)]
            if rng.random() < 0.40:
                wb = strip_accents(wb)
            return _apply_case(rng, wb, p_lower=0.35, p_upper=0.08, p_mixed=0.10)

        if sg == "insuff_province_only":
            i = cursors["prov"]
            cursors["prov"] += 1
            raw = prov_only_pool[i % len(prov_only_pool)]
            return _apply_case(rng, raw, p_lower=0.40, p_upper=0.15, p_mixed=0.10)

        if sg == "insuff_district_only":
            i = cursors["dist_only"]
            cursors["dist_only"] += 1
            d_full, d_bare, _, _ = all_dists[i % len(all_dists)]
            style = rng.random()
            if style < 0.45:
                raw = d_full
            elif style < 0.75:
                raw = _abbr_district(rng, d_full, d_bare)
            else:
                raw = d_bare
            if rng.random() < 0.38:
                raw = strip_accents(raw)
            return _apply_case(rng, raw, p_lower=0.35, p_upper=0.08, p_mixed=0.10)

        if sg == "insuff_district_province":
            i = cursors["dist_prov"]
            cursors["dist_prov"] += 1
            d_full, d_bare, p_full, p_bare = all_dists[i % len(all_dists)]
            if norm_key(d_bare) == norm_key(p_bare):
                d_choice = d_full
            else:
                d_choice = rng.choice([d_full, d_bare, _abbr_district(rng, d_full, d_bare)])
            p_choice = rng.choice([p_full, p_bare, _abbr_province(rng, p_full, p_bare)])
            sep = rng.choice([", ", ", ", " ", ","])
            raw = f"{d_choice}{sep}{p_choice}"
            if rng.random() < 0.35:
                raw = strip_accents(raw)
            return _apply_case(rng, raw, p_lower=0.30, p_upper=0.08, p_mixed=0.10)

        if sg == "insuff_bare_granular":
            raw = rng.choice([
                f"ngõ {rng.randint(1, 450)}",
                f"hẻm {rng.randint(1, 300)}",
                f"ngách {rng.randint(1, 150)}",
                f"kiệt {rng.randint(1, 200)}",
                f"Thôn {rng.randint(1, 25)}",
                f"Thôn {rng.choice(['Đông', 'Tây', 'Nam', 'Bắc', 'Trung', 'Tân Lập', 'Hòa Bình'])}",
                f"xóm {rng.randint(1, 20)}",
                f"ấp {rng.randint(1, 15)}",
                f"tổ {rng.randint(1, 45)}",
                f"tổ dân phố {rng.randint(1, 35)}",
                f"khu phố {rng.randint(1, 16)}",
                f"kp {rng.randint(1, 14)}",
                f"tầng {rng.randint(1, 35)}",
                f"lầu {rng.randint(1, 20)}",
                f"phòng {rng.randint(101, 2512)}",
                f"căn hộ {rng.choice(['A','B','C','S1','S2'])}{rng.randint(1,9)}-{rng.randint(101,1912)}",
                f"lô {rng.choice(['A','B','C','D','E'])}{rng.randint(1,9)}",
                f"block {rng.choice(['A','B','C','D'])}{rng.randint(1,5)}",
                f"chung cư {rng.choice(['A1', 'B2', 'C3', 'An Bình', 'Hòa Bình', 'Sunrise', 'HH1A'])}",
                f"đường {rng.choice(STREETS)}",
                f"phố {rng.choice(UNAMBIGUOUS_STREETS)}",
                f"{rng.choice(KCN_POOL)}",
            ])
            if rng.random() < 0.38:
                raw = strip_accents(raw)
            return _apply_case(rng, raw, p_lower=0.35, p_upper=0.08, p_mixed=0.12)

        if sg == "insuff_house_street_only":
            st = rng.choice(UNAMBIGUOUS_STREETS)
            hnum = _rand_house(rng)
            raw = rng.choice([
                f"{hnum} {st}",
                f"số {hnum} {st}",
                f"số nhà {hnum} đường {st}",
                f"ngõ {rng.randint(1, 180)} {st}",
                f"hẻm {rng.randint(1, 120)} đường {st}",
            ])
            if rng.random() < 0.35:
                raw = strip_accents(raw)
            return _apply_case(rng, raw, p_lower=0.30, p_upper=0.06, p_mixed=0.10)

        # -------- Label 0: not_address --------
        if sg == "neg_short_garbage":
            r = rng.random()
            if r < 0.12:
                return rng.choice(SHORT_GARBAGE_TOKENS)
            if r < 0.35:
                return str(rng.randint(0, 9999))
            if r < 0.55:
                return "".join(rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(rng.randint(1, 4)))
            if r < 0.80:
                return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(rng.randint(2, 10)))
            w1 = rng.choice(["ok", "hi", "alo", "jean", "xom", "shop", "check", "ship", "co", "khong", "vui", "dep", "tot"])
            w2 = rng.choice(["nha", "nhe", "a", "oi", "ban", "em", "10", "73", "vip", "pro"])
            return f"{w1} {w2}" if rng.random() < 0.6 else w1.capitalize()

        if sg == "neg_chat_order":
            place_pool_item = rng.choice([
                rng.choice(all_provs)[1],
                rng.choice(all_dists)[1],
                rng.choice(all_wards_bare),
            ])
            r = rng.random()
            if r < 0.28:
                # 2-7 word e-commerce products, garments, electronics (standalone & lightly qualified)
                prod = rng.choice(ECOM_PRODUCTS_GARMENTS)
                attr = rng.choice(ECOM_ATTRS)
                qty = rng.choice(["", "", f"{rng.randint(1, 5)} ", f"{rng.randint(1, 4)} cái ", f"{rng.randint(1, 4)} chiếc "])
                tpl = f"{qty}{prod} {attr}".strip()
            elif r < 0.50:
                # Short polite chat & non-address delivery instructions
                if rng.random() < 0.55:
                    base = rng.choice(CHAT_CONFIRMATIONS_SHORT)
                    extra = rng.choice(["", " nhé", " ạ", " nha", f" đơn #{rng.randint(10, 999)}"])
                    tpl = f"{base}{extra}".strip()
                else:
                    tpl = rng.choice(NON_ADDR_DELIVERY_COMMENTS)
            elif r < 0.68:
                qty = rng.randint(1, 12)
                unit = rng.choice(["chai", "gói", "hộp", "bịch", "túi", "hũ", "cái", "chiếc"])
                prod = rng.choice([
                    "nước tương", "xốt nướng BBQ", "áo thun size L", "quần jean ống suông",
                    "bánh quy bơ", "trà sữa trân châu", "đồ ngủ lụa", "son lì",
                    "hạt điều rang muối", "cà phê rang xay", "khẩu trang y tế", "băng keo trong",
                ])
                tpl = rng.choice([
                    f"Cho mình {qty} {unit} {prod}",
                    f"Lấy giúp em {qty} {unit} {prod} nha shop",
                    f"Đặt {qty} {unit} {prod} {rng.choice(['loại 1', 'chính hãng', 'giá rẻ', ''])}".strip(),
                    f"Xin giá {prod} {rng.choice(['chính hãng', 'loại 1', ''])}".strip(),
                ])
            elif r < 0.86:
                tpl = rng.choice([
                    f"phí ship về {place_pool_item} bao nhiêu vậy shop",
                    f"ship về {place_pool_item} mất mấy ngày ạ",
                    f"có gửi hàng đi {place_pool_item} không bạn",
                    f"đơn về {place_pool_item} hôm nay có giao kịp không",
                    f"từ kho gửi ra {place_pool_item} hết bao nhiêu tiền",
                    f"khách ở {place_pool_item} đặt có được freeship không",
                    f"ship tới {place_pool_item} tính sao vậy shop",
                ])
            else:
                tpl = rng.choice([
                    f"gọi anh {rng.choice(GIVEN_M)} lúc {rng.randint(8, 20)}h giúp em",
                    f"shipper tới gọi {rng.choice(['anh', 'chị'])} {rng.choice(GIVEN_F)} nha",
                    f"cho xem hàng trước khi trả {rng.randint(100, 900)}k",
                    f"tối nay {rng.randint(18, 21)}h mình mới có mặt",
                    f"mã đơn #{rng.randint(1000, 99999)} kiểm tra giúp mình",
                ])
            if rng.random() < 0.35:
                tpl = strip_accents(tpl)
            return _apply_case(rng, tpl, p_lower=0.35, p_upper=0.04, p_mixed=0.05)

        if sg == "neg_food_product_place":
            dish, orig_place = rng.choice(FOOD_PRODUCT_PLACE)
            # Mix canonical dish origins with compositional real VN ward/district/province pairs
            if rng.random() < 0.55:
                place = orig_place
            elif rng.random() < 0.55:
                place = rng.choice(all_provs)[1]
            elif rng.random() < 0.65:
                place = rng.choice(all_dists)[1]
            else:
                place = rng.choice(all_wards_bare)

            qty_u = rng.choice([
                "", "",
                f"{rng.randint(1, 6)} gói ", f"{rng.randint(1, 5)} hộp ",
                f"{rng.randint(1, 4)} chai ", f"{rng.randint(1, 4)} hũ ",
                f"{rng.randint(1, 4)} túi ", f"{rng.randint(1, 4)} bịch ",
                f"{rng.choice([200, 250, 300, 500, 750])}g ",
                f"{rng.choice([250, 500])}gr ",
                f"{rng.randint(1, 5)}kg ",
                f"{rng.choice([250, 500, 750])}ml ",
                f"{rng.randint(1, 3)} lít ",
                f"{rng.randint(1, 6)} phần ",
            ])
            qual = rng.choice([
                "", "", " đặc biệt", " loại 1", " chính gốc", " truyền thống",
                " nguyên hạt", " một nắng", " chính hãng", " giá rẻ", " ít ngọt",
            ])
            r = rng.random()
            if r < 0.45:
                raw = f"{qty_u}{dish} {place}{qual}".strip()
            elif r < 0.80:
                raw = rng.choice([
                    f"Cho mình {qty_u or '2 phần '}{dish} {place}{qual}",
                    f"Mua {qty_u or '2 hộp '}{dish} {place}{qual} làm quà",
                    f"Quán {dish} {place} này ngon nổi tiếng",
                    f"Đặc sản {dish} {place}{qual} giá bao nhiêu vậy shop",
                    f"Ship giúp em {qty_u or '2 gói '}{dish} {place}{qual} nha",
                    f"Đặt {qty_u or '3 chai '}{dish} {place}{qual}",
                ])
            else:
                raw = f"{dish} {place}{qual} - {rng.choice(['50k', '120k', '250k', 'combo 2 người', 'size lớn'])}"
            if rng.random() < 0.35:
                raw = strip_accents(raw)
            return _apply_case(rng, raw, p_lower=0.35, p_upper=0.04, p_mixed=0.05)

        if sg == "neg_person_name":
            name = _make_person_name(rng, allow_admin_trap=True)
            if rng.random() < 0.50:
                return name
            if rng.random() < 0.25:
                # Anh/Chị <District/Ward> narrative/contact
                adm = rng.choice(all_dists)[1] if rng.random() < 0.5 else rng.choice(all_wards_bare)
                tpl = rng.choice([
                    f"Anh {adm} vừa nhắn tin đặt hàng",
                    f"Chị {adm} hỏi giá sản phẩm",
                    f"Trưởng phòng {name} phụ trách {adm}",
                    f"Khách {rng.choice(['anh', 'chị'])} {adm} mới chuyển khoản",
                ])
            else:
                tpl = rng.choice([
                    f"Khách hàng: {name}",
                    f"Anh {name} vừa đặt hàng",
                    f"Liên hệ {name} để xác nhận đơn",
                    f"Người nhận là {name}",
                    f"Chuyển khoản từ {name}",
                    f"{name} sinh năm {rng.randint(1975, 2005)}",
                ])
            if rng.random() < 0.35:
                tpl = strip_accents(tpl)
            return _apply_case(rng, tpl, p_lower=0.28, p_upper=0.04, p_mixed=0.04)

        if sg == "neg_narrative_weather_travel":
            r_p = rng.random()
            if r_p < 0.45:
                place = rng.choice(all_provs)[1]
            elif r_p < 0.80:
                d_full, d_bare, _, p_bare = rng.choice(all_dists)
                place = d_full if rng.random() < 0.45 else f"{d_bare}, {p_bare}" if rng.random() < 0.35 else d_bare
            else:
                place = rng.choice(all_wards_bare)

            place2 = rng.choice(all_provs)[1]
            tpl = rng.choice([
                f"Thời tiết {place} hôm nay nắng đẹp {rng.randint(26, 36)} độ",
                f"Cuối tuần cả nhà mình đi du lịch {place}",
                f"Chi nhánh {place} tạm nghỉ lễ từ thứ 7",
                f"Bưu cục {place} nghỉ trưa từ 12h đến 13h30",
                f"Cửa hàng ở {place} mở cửa tới 21h tối",
                f"Chi nhánh ở {place} đang tuyển nhân viên",
                f"Công ty ở {place} tổ chức sự kiện cuối năm",
                f"Bệnh viện {place} hôm nay đông bệnh nhân",
                f"Trường đại học ở {place} tổ chức lễ khai giảng",
                f"Quê ở {place} nhưng đang thuê phòng trọ đi làm",
                f"Đường {place} cấm xe giờ cao điểm, phải đi bộ",
                f"Giá đất khu vực {place} năm nay tăng cao",
                f"Giá vé tàu cao tốc đi {place} bao nhiêu vậy",
                f"Xe khách chạy tuyến {place} - {place2} mấy giờ xuất bến",
                f"Thác nước ở {place} nổi tiếng đẹp vào mùa thu",
                f"Mùa thu ở {place} hương hoa sữa thơm ngát",
                f"Cầu Rồng ở {place} phun lửa tối cuối tuần",
                f"Các chiến sĩ ở {place} tham gia hỗ trợ người dân",
                f"Ở {place} sáng nay mưa rào rải rác",
                f"Lịch cắt điện khu vực {place} ngày {rng.randint(1, 28)}/{rng.randint(1, 12)}",
            ])
            if rng.random() < 0.35:
                tpl = strip_accents(tpl)
            return _apply_case(rng, tpl, p_lower=0.30, p_upper=0.04, p_mixed=0.05)

        if sg == "neg_structured_id_price_date":
            r = rng.random()
            if r < 0.20:
                return _rand_phone(rng) if rng.random() < 0.7 else f"Zalo: {_rand_phone(rng)}"
            if r < 0.38:
                u = strip_accents(rng.choice(GIVEN_M + GIVEN_F)).lower()
                dom = rng.choice(["gmail.com", "yahoo.com", "outlook.com", "fpt.vn", "company.vn"])
                return f"{u}{rng.randint(10, 9999)}@{dom}"
            if r < 0.54:
                host = rng.choice(["shopee.vn", "tiki.vn", "lazada.vn", "tiktok.com", "vnexpress.net"])
                return f"https://{host}/item/{rng.randint(1000, 999999)}"
            if r < 0.70:
                pfx = rng.choice(["GHN", "SPXVN", "GHTK", "JT", "VTP", "MVD"])
                return f"{pfx}{rng.randint(10000000, 99999999)}"
            if r < 0.85:
                dd, mm, yy = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1990, 2027)
                return f"{dd:02d}/{mm:02d}/{yy}"
            amt = rng.choice([35, 50, 85, 120, 199, 250, 390, 550, 1200, 2500])
            return f"{amt}{rng.choice(['.000đ', '.000 VND', 'k', 'K'])}"

        # neg_vague_non_addr
        base = rng.choice(VAGUE_NON_ADDRESS)
        tail = rng.choice(["", " nha", " giúp mình", " ạ", f" đơn #{rng.randint(100, 9999)}", " nhé shop", " cho nhanh"])
        raw = f"{base}{tail}".strip()
        if rng.random() < 0.35:
            raw = strip_accents(raw)
        return _apply_case(rng, raw, p_lower=0.40, p_upper=0.05, p_mixed=0.05)

    # Collect samples per subgroup, stratified cleanly, with per-label backfill
    all_by_sg: Dict[Tuple[int, str], List[Sample]] = {}
    label_targets: Dict[int, int] = defaultdict(int)
    label_counts: Dict[int, int] = defaultdict(int)
    for label, sg, target_n in subgroups_spec:
        label_targets[label] += target_n
        bucket: List[Sample] = []
        check_4gram = (
            label == 0
            or "freetext" in sg
            or "chat" in sg
        )
        attempts = 0
        max_attempts = max(target_n * 25, 4000)
        while len(bucket) < target_n and attempts < max_attempts:
            attempts += 1
            txt = re.sub(r"\s+", " ", gen_one(label, sg).replace("\n", " ").replace("\r", " ").strip())
            if not txt:
                continue
            nk = norm_key(txt)
            if not nk or nk in seen_global:
                continue
            if check_4gram and (_extract_4grams(txt) & forbidden_4grams):
                continue
            seen_global.add(nk)
            bucket.append(Sample(text=txt, label=label, subgroup=sg))
        all_by_sg[(label, sg)] = bucket
        label_counts[label] += len(bucket)

    # Backfill each label individually from its richest diverse subgroup so exact class ratios hold
    backfill_sg_by_label = {
        0: "neg_chat_order",
        1: "insuff_district_province",
        2: "full_urban",
    }
    for lbl in (0, 1, 2):
        sg = backfill_sg_by_label[lbl]
        while label_counts[lbl] < label_targets[lbl]:
            txt = re.sub(r"\s+", " ", gen_one(lbl, sg).strip())
            nk = norm_key(txt)
            if not nk or nk in seen_global:
                continue
            if lbl == 0 and (_extract_4grams(txt) & forbidden_4grams):
                continue
            seen_global.add(nk)
            all_by_sg[(lbl, sg)].append(Sample(text=txt, label=lbl, subgroup=sg))
            label_counts[lbl] += 1

    # Stratified 80 / 10 / 10 split per subgroup (distribute rounding remainders cleanly)
    train: List[Sample] = []
    val: List[Sample] = []
    test: List[Sample] = []

    rem_toggle = 0
    for key in sorted(all_by_sg.keys()):
        items = all_by_sg[key]
        rng.shuffle(items)
        n = len(items)
        n_tr = (n * 8) // 10
        n_va = n // 10
        n_te = n - n_tr - n_va
        # Keep overall val/test totals balanced when n % 10 != 0
        if n_te - n_va > 1:
            n_va += 1
            n_te -= 1
        elif n_te > n_va and (rem_toggle % 2 == 1):
            n_tr += 1
            n_te -= 1
        rem_toggle += 1
        train.extend(items[:n_tr])
        val.extend(items[n_tr : n_tr + n_va])
        test.extend(items[n_tr + n_va :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    if write_manifest:
        MANIFEST_PATH.write_text(
            "\n".join(s.text for s in train) + "\n",
            encoding="utf-8",
        )

    return DatasetSplitsV2(train=train, val=val, test=test)


if __name__ == "__main__":
    splits = generate_dataset_v2(write_manifest=True)
    for split_name, items in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        counts = {c: sum(1 for x in items if x.label == c) for c in (0, 1, 2)}
        print(f"{split_name:5s}: total={len(items)} | counts={counts}")
    print(f"Manifest written to: {MANIFEST_PATH} ({MANIFEST_PATH.stat().st_size / 1024:.1f} KB)")
