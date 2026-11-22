"""Way B v2 Dataset Generator (75,000 samples, 3-class sufficiency, MIT GSO/2025 + Strict Anti-Leakage).

Generates 75,000 deduplicated Vietnamese samples (seed=202) stratified 80/10/10 into
train (60,000) / val (7,500) / test (7,500) across 3 classes:
  0: not_address          (~30%, 22,500)
  1: address_insufficient (~25%, 18,750)
  2: address_sufficient   (~45%, 33,750)

Data source: ../shared_data/vn_admin_full_records.json (MIT License tranngocminhhieu/vietnamadminunits)
Strict anti-leakage guard rejects any exact normalized overlap against BOTH:
  - ../benchmark/vn_address_benchmark.json
  - ../benchmark_v2/independent_holdout_v2.json
and rejects any negative / chat / delivery-freetext sample sharing a word 4-gram with eval cases.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

from app.tokenizer import strip_accents

SUFFICIENCY_CLASSES: Tuple[str, ...] = (
    "not_address",
    "address_insufficient",
    "address_sufficient",
)

ROOT_DIR = Path(__file__).resolve().parent.parent
SHARED_RECORDS_PATH = ROOT_DIR.parent / "shared_data" / "vn_admin_full_records.json"
BENCH_V1_PATH = ROOT_DIR.parent / "benchmark" / "vn_address_benchmark.json"
BENCH_V2_PATH = ROOT_DIR.parent / "benchmark_v2" / "independent_holdout_v2.json"
MANIFEST_OUT_PATH = ROOT_DIR / "train" / "train_texts_manifest.txt"


@dataclass(frozen=True)
class SampleV2:
    text: str
    label: int  # 0=not_address, 1=address_insufficient, 2=address_sufficient
    group: str
    subgroup: str


def norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(text).lower().strip())


def norm_alnum_words(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", strip_accents(text).lower()).strip()
    return cleaned.split() if cleaned else []


def extract_4grams(text: str) -> Set[Tuple[str, str, str, str]]:
    grams: Set[Tuple[str, str, str, str]] = set()
    w1 = norm_key(text).split()
    for i in range(len(w1) - 3):
        grams.add((w1[i], w1[i + 1], w1[i + 2], w1[i + 3]))
    w2 = norm_alnum_words(text)
    for i in range(len(w2) - 3):
        grams.add((w2[i], w2[i + 1], w2[i + 2], w2[i + 3]))
    return grams


class LeakageGuard:
    """Strict eval anti-leakage filter against v1 & v2 holdout benchmarks."""

    def __init__(self, eval_paths: Sequence[Path]) -> None:
        self.eval_exact: Set[str] = set()
        self.eval_alnum: Set[str] = set()
        self.eval_4grams: Set[Tuple[str, str, str, str]] = set()

        for path in eval_paths:
            if not path.exists():
                raise FileNotFoundError(f"Missing eval benchmark for anti-leakage guard: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            for item in payload.get("cases", []):
                txt = item["text"]
                self.eval_exact.add(norm_key(txt))
                self.eval_alnum.add(" ".join(norm_alnum_words(txt)))
                self.eval_4grams.update(extract_4grams(txt))

    def is_leaked(self, text: str, check_4grams: bool) -> bool:
        k = norm_key(text)
        if not k or k in self.eval_exact:
            return True
        if " ".join(norm_alnum_words(text)) in self.eval_alnum:
            return True
        if check_4grams:
            for g in extract_4grams(text):
                if g in self.eval_4grams:
                    return True
        return False


# ---------------------------------------------------------------------------
# Authentic Vietnamese Lexical Pools (non-overlapping templates)
# ---------------------------------------------------------------------------

STREETS: List[str] = [
    "Lê Lợi", "Nguyễn Huệ", "Trần Hưng Đạo", "Hai Bà Trưng", "Nguyễn Trãi",
    "Lý Thường Kiệt", "Điện Biên Phủ", "Cách Mạng Tháng 8", "Võ Thị Sáu",
    "Nguyễn Thị Minh Khai", "Phan Đình Phùng", "Hoàng Diệu", "Bạch Đằng",
    "Trần Phú", "Hùng Vương", "Quang Trung", "Nguyễn Văn Linh", "Lê Duẩn",
    "Phan Châu Trinh", "Nguyễn Văn Cừ", "Ngô Quyền", "Lạc Long Quân", "Âu Cơ",
    "Hoàng Hoa Thám", "Thụy Khuê", "Kim Mã", "Đội Cấn", "Láng Hạ", "Thái Hà",
    "Chùa Bộc", "Tôn Đức Thắng", "Xã Đàn", "Khâm Thiên", "Giải Phóng",
    "Trường Chinh", "Bạch Mai", "Minh Khai", "Tam Trinh", "Nguyễn Xiển",
    "Khuất Duy Tiến", "Lê Văn Lương", "Tố Hữu", "Trần Duy Hưng", "Nguyễn Chí Thanh",
    "Hoàng Quốc Việt", "Xuân Thủy", "Cầu Giấy", "Hồ Tùng Mậu", "Phạm Hùng",
    "Phạm Văn Đồng", "Võ Chí Công", "Tây Sơn", "Nguyễn Lương Bằng", "Đào Tấn",
    "Liễu Giai", "Văn Cao", "Phan Kế Bính", "Trần Thái Tông", "Trung Kính",
    "Pasteur", "Nam Kỳ Khởi Nghĩa", "Võ Văn Tần", "Nguyễn Đình Chiểu",
    "Lê Văn Sỹ", "Phan Xích Long", "Hoàng Văn Thụ", "Cộng Hòa", "Phổ Quang",
    "Bạch Đằng", "Xô Viết Nghệ Tĩnh", "Đinh Bộ Lĩnh", "Ung Văn Khiêm",
    "Phan Đăng Lưu", "Lê Quang Định", "Nguyễn Oanh", "Phan Văn Trị",
    "Kha Vạn Cân", "Võ Văn Ngân", "Lê Văn Việt", "Đỗ Xuân Hợp", "Xa Lộ Hà Nội",
    "Mai Chí Thọ", "Huỳnh Tấn Phát", "Nguyễn Hữu Thọ", "Nguyễn Thị Thập",
    "Lê Văn Lương", "Tạ Quang Bửu", "Phạm Thế Hiển", "An Dương Vương",
    "Hồng Bàng", "3 Tháng 2", "Sư Vạn Hạnh", "Thành Thái", "Tô Hiến Thành",
    "Lý Thái Tổ", "Nguyễn Tri Phương", "Hậu Giang", "Kinh Dương Vương",
    "Tân Kỳ Tân Quý", "Lũy Bán Bích", "Hòa Bình", "Trịnh Đình Trọng",
    "Đồng Khởi", "Nguyễn Ái Quốc", "Phạm Văn Thuận", "Hà Huy Giáp",
    "30 Tháng 4", "Lê Hồng Phong", "Đại lộ Bình Dương", "Mỹ Phước Tân Vạn",
    "Lê Thánh Tôn", "Tôn Thất Tùng", "Bùi Thị Xuân", "Lý Tự Trọng",
    "Trần Quang Khải", "Hoàng Sa", "Trường Sa", "Võ Nguyên Giáp",
]

BUILDINGS: List[str] = [
    "Chung cư Vinhomes Ocean Park", "Chung cư Times City", "KĐT Royal City",
    "Chung cư Imperia Sky Garden", "Tòa nhà Capital Place", "Chung cư Goldseason",
    "Chung cư Sunshine Riverside", "KĐT Ciputra", "Chung cư Ecohome 3",
    "Tòa nhà Lotte Center", "Chung cư Hateco Apollo", "Chung cư Ecolife Capitol",
    "Chung cư Masteri Waterfront", "Chung cư Masteri West Heights",
    "KĐT Ecopark Westbay", "Chung cư Vinhomes Grand Park", "Chung cư Masteri Centre Point",
    "KĐT Phú Mỹ Hưng", "Chung cư River Panorama", "Chung cư Scenic Valley",
    "Chung cư Saigon South Residences", "Chung cư Mizuki Park", "Chung cư Akari City",
    "Chung cư Opal Boulevard", "Chung cư Bcons Suối Tiên", "Tòa nhà Etown 5",
    "Tòa nhà Deutsches Haus", "Tòa nhà Lim Tower 3", "Tòa nhà Viettel Complex",
    "Chung cư Charmington La Pointe", "Chung cư Hà Đô Centrosa", "Chung cư RichStar",
    "Chung cư Monarchy", "KĐT FPT City", "Chung cư Diamond Riverside",
]

KCN_NAMES: List[str] = [
    "KCN VSIP 1", "KCN VSIP 2", "KCN Mỹ Phước 3", "KCN Sóng Thần 1",
    "KCN Amata", "KCN Biên Hòa 2", "KCN Long Thành", "KCN Nhơn Trạch 2",
    "KCN Tân Tạo", "KCN Vĩnh Lộc", "KCN Hiệp Phước", "KCN Quang Minh",
    "KCN Thăng Long", "KCN Yên Phong", "KCN Quế Võ", "KCN Tràng Duệ",
    "KCN Đình Vũ", "KCN Hòa Khánh", "KCN Liên Chiểu", "KCN Phú Mỹ 1",
    "KCN Long Hậu", "KCN Đức Hòa 3", "KCN Trà Nóc", "KCN Bắc Đồng Phú",
]

VN_SURNAMES: List[str] = [
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ",
    "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý", "Đinh", "Đoàn", "Trương", "Mai",
]
VN_MIDDLES: List[str] = [
    "Văn", "Thị", "Minh", "Hoàng", "Quốc", "Thanh", "Đức", "Ngọc", "Anh",
    "Gia", "Phương", "Thảo", "Hữu", "Xuân", "Tuấn", "Kim", "Bảo", "Nhật",
]
VN_GIVEN: List[str] = [
    "Nam", "Hùng", "Dũng", "Tuấn", "Khang", "Phúc", "Long", "Quân", "Sơn",
    "Tâm", "Trang", "Linh", "Lan", "Hương", "Thảo", "Vy", "Ngọc", "Hà",
    "Nhi", "Phương", "Hạnh", "Tú", "Đạt", "Thịnh", "Khánh", "My",
]
COLLIDING_PERSON_NAMES: List[str] = [
    "Nguyễn Trãi", "Lê Lợi", "Trần Hưng Đạo", "Hai Bà Trưng", "Quang Trung",
    "Hùng Vương", "Lý Thường Kiệt", "Phan Bội Châu", "Phan Chu Trinh",
    "Ngô Quyền", "Đinh Tiên Hoàng", "Lê Thánh Tôn", "Nguyễn Huệ", "Võ Thị Sáu",
    "Nguyễn Thị Minh Khai", "Hoàng Văn Thụ", "Phạm Ngũ Lão", "Tôn Đức Thắng",
    "Nguyễn Thị Thanh Xuân", "Trần Văn Bình Thạnh", "Lê Hoàng Mai",
    "Phạm Văn Phú Nhuận", "Nguyễn Văn Tân Bình", "Võ Thị Hồng Hà",
    "Hoàng Quốc Hoàn Kiếm", "Đặng Minh Cầu Giấy", "Bùi Đức Long Biên",
    "Trần Ngọc Bình Chánh", "Lê Thị Bến Nghé", "Nguyễn Hữu Đông Anh",
    "Phạm Văn Đồng", "Võ Văn Kiệt", "Nguyễn Đình Chiểu", "Lý Tự Trọng",
    "Trần Quốc Toản", "Nguyễn Văn Cừ", "Lê Văn Sỹ", "Nguyễn Chí Thanh",
]

SHIPPER_NOTES: List[str] = [
    "giao buổi sáng", "giao sau 18h giúp mình", "tới cổng alo số này",
    "nhờ lễ tân giữ hộ", "ship khung giờ 9h-11h", "không bấm chuông nhà có em bé",
    "gọi điện trước 10 phút", "giao tận sảnh tòa nhà", "nhận hàng giờ hành chính",
    "giao cuối tuần cũng được", "nhắn zalo nếu không nghe máy", "để bàn bảo vệ cổng B",
]

PROVINCE_ALIASES_LIST: List[Tuple[str, str]] = [
    ("Hà Nội", "HN"), ("TP. Hồ Chí Minh", "TP.HCM"), ("TP. Hồ Chí Minh", "TPHCM"),
    ("TP. Hồ Chí Minh", "HCM"), ("TP. Hồ Chí Minh", "Sài Gòn"), ("Đà Nẵng", "ĐN"),
    ("Hải Phòng", "HP"), ("Cần Thơ", "CT"), ("Bình Dương", "BD"),
    ("Bà Rịa - Vũng Tàu", "Vũng Tàu"), ("Thừa Thiên Huế", "Huế"),
]

DISTRICT_ALIASES_REV: Dict[str, List[str]] = {
    "quan 1": ["Q1", "Q.1", "q1", "q.1"],
    "quan 3": ["Q3", "Q.3", "q3", "q.3"],
    "quan 4": ["Q4", "Q.4", "q4", "q.4"],
    "quan 5": ["Q5", "Q.5", "q5", "q.5"],
    "quan 6": ["Q6", "Q.6", "q6", "q.6"],
    "quan 7": ["Q7", "Q.7", "q7", "q.7"],
    "quan 8": ["Q8", "Q.8", "q8", "q.8"],
    "quan 10": ["Q10", "Q.10", "q10", "q.10"],
    "quan 11": ["Q11", "Q.11", "q11", "q.11"],
    "quan 12": ["Q12", "Q.12", "q12", "q.12"],
    "tan binh": ["Q.TB", "q.tb", "qtb"],
    "binh thanh": ["Q.BT", "q.bt", "qbt"],
    "phu nhuan": ["Q.PN", "q.pn", "qpn"],
    "go vap": ["Q.GV", "q.gv", "qgv"],
    "tan phu": ["Q.TP", "q.tp", "qtp"],
    "thanh xuan": ["Q.TX", "q.tx", "qtx"],
    "cau giay": ["Q.CG", "q.cg", "qcg"],
    "dong da": ["Q.DD", "q.dd", "qdd"],
    "hoan kiem": ["Q.HK", "q.hk", "qhk"],
    "hoang mai": ["Q.HM", "q.hm", "qhm"],
    "ba dinh": ["Q.BD", "q.bd", "qbd"],
    "hai ba trung": ["Q.HBT", "HBT", "hbt"],
    "long bien": ["Q.LB", "q.lb"],
    "nam tu liem": ["NTL", "Q.NTL"],
    "bac tu liem": ["BTL", "Q.BTL"],
    "hai chau": ["Q.HC", "q.hc"],
    "ninh kieu": ["Q.NK", "q.nk"],
    "ngo quyen": ["Q.NQ", "q.nq"],
}

REGIONAL_FOODS: List[Tuple[str, str]] = [
    ("bánh mì", "Sài Gòn"), ("mực một nắng", "Cà Mau"), ("mực khô", "Cà Mau"),
    ("phở bò", "Hà Nội"), ("bún chả", "Hà Nội"), ("bún thang", "Hà Nội"),
    ("bún bò", "Huế"), ("mè xửng", "Huế"), ("tôm chua", "Huế"),
    ("nước mắm", "Phú Quốc"), ("tiêu đen", "Phú Quốc"), ("rượu sim", "Phú Quốc"),
    ("kẹo dừa", "Bến Tre"), ("bưởi da xanh", "Bến Tre"), ("mứt dừa", "Bến Tre"),
    ("nem chua", "Thanh Hóa"), ("chả tôm", "Thanh Hóa"),
    ("chè xanh", "Thái Nguyên"), ("trà móc câu", "Thái Nguyên"),
    ("bánh đậu xanh", "Hải Dương"), ("vải thiều", "Hải Dương"),
    ("chả mực", "Hạ Long"), ("sá sùng", "Quảng Ninh"),
    ("bánh tráng phơi sương", "Tây Ninh"), ("muối tôm", "Tây Ninh"),
    ("nhãn lồng", "Hưng Yên"), ("tương bần", "Hưng Yên"),
    ("vải thiều", "Bắc Giang"), ("mì chũ", "Bắc Giang"),
    ("mì Quảng", "Đà Nẵng"), ("chả bò", "Đà Nẵng"),
    ("cơm tấm", "Sài Gòn"), ("hủ tiếu", "Sài Gòn"),
    ("bánh pía", "Sóc Trăng"), ("lạp xưởng", "Sóc Trăng"),
    ("bánh đa cua", "Hải Phòng"), ("nem cua bể", "Hải Phòng"),
    ("yến sào", "Khánh Hòa"), ("nem nướng", "Nha Trang"),
    ("tỏi cô đơn", "Lý Sơn"), ("mạch nha", "Quảng Ngãi"),
    ("hạt điều rang muối", "Bình Phước"), ("tiêu xanh", "Bình Phước"),
    ("cà phê rang xay", "Buôn Ma Thuột"), ("mật ong hoa cà phê", "Đắk Lắk"),
    ("sầu riêng", "Cần Thơ"), ("bánh tét lá cẩm", "Cần Thơ"),
    ("xoài cát", "Tiền Giang"), ("hủ tiếu", "Mỹ Tho"),
    ("gạo ST25", "Sóc Trăng"), ("cá kho niêu", "Hà Nam"),
    ("bánh cốm", "Hà Nội"), ("ô mai", "Hà Nội"),
    ("bưởi Phúc Trạch", "Hà Tĩnh"), ("kẹo cu đơ", "Hà Tĩnh"),
    ("cam sành", "Hà Giang"), ("thịt trâu gác bếp", "Sơn La"),
    ("bánh cáy", "Thái Bình"), ("nước mắm", "Phan Thiết"),
    ("thanh long ruột đỏ", "Bình Thuận"), ("bánh ít lá gai", "Bình Định"),
    ("rượu Bàu Đá", "Bình Định"), ("nem nướng", "Đà Lạt"),
    ("dâu tây", "Đà Lạt"), ("atisô", "Lâm Đồng"),
    ("cơm cháy", "Ninh Bình"), ("dê núi", "Ninh Bình"),
    ("bánh gai", "Nam Định"), ("phở", "Nam Định"),
    ("cam Vinh", "Nghệ An"), ("tương Nam Đàn", "Nghệ An"),
    ("bò tơ", "Củ Chi"), ("bánh cuốn", "Thanh Trì"),
    ("cốm làng Vòng", "Cầu Giấy"), ("bún cá", "Long Xuyên"),
    ("bánh hỏi", "Bến Tre"), ("khô cá lóc", "Đồng Tháp"),
    ("quýt hồng", "Lai Vung"), ("bánh tằm", "Bạc Liêu"),
]

GENERAL_PRODUCTS: List[str] = [
    "Áo phông cotton unisex", "Quần jean ống rộng", "Áo khoác gió chống nước",
    "Giày chạy bộ đế êm", "Túi đeo chéo canvas", "Balo laptop 15.6 inch",
    "Tai nghe bluetooth chống ồn", "Sạc dự phòng 20000mAh", "Cáp sạc nhanh USB-C",
    "Bàn phím cơ không dây", "Chuột gaming siêu nhẹ", "Giá đỡ laptop hợp kim",
    "Nồi chiên không dầu 6L", "Máy xay cầm tay mini", "Bình giữ nhiệt inox 900ml",
    "Hộp cơm giữ nhiệt văn phòng", "Kem chống nắng kiềm dầu", "Sữa rửa mặt dịu nhẹ",
    "Son lì dưỡng ẩm", "Nước tẩy trang cho da nhạy cảm", "Dầu gội thảo dược",
    "Sổ tay bìa da A5", "Bút gel mực đen", "Đèn học chống cận",
    "Thảm tập yoga TPE", "Dây kháng lực tập gym", "Thức ăn cho mèo hạt khô",
    "Cát vệ sinh cho mèo", "Sữa tươi tiệt trùng ít đường", "Ngũ cốc ăn sáng",
]

SHORT_WORDS: List[str] = [
    "ok", "hi", "hello", "alo", "dạ", "vâng", "chốt", "hủy", "xong", "rồi",
    "tốt", "đẹp", "rẻ", "đắt", "nhanh", "chậm", "giao", "ship", "đặt", "mua",
    "bán", "hỏi", "xem", "gửi", "nhận", "trả", "đổi", "size", "màu", "giá",
    "shop", "bạn", "em", "anh", "chị", "jean", "xom", "k", "m", "l", "xl",
    "xxl", "red", "blue", "test", "admin", "user", "pass", "code", "otp",
    "link", "zalo", "fb", "sale", "free", "cod", "bank", "ck", "stk", "bill",
    "vâng ạ", "ok ạ", "dạ vâng", "chưa ạ", "có ạ", "không ạ", "cảm ơn",
    "xin lỗi", "tạm biệt", "hẹn gặp", "đợi chút", "chờ tí", "xác nhận",
    "hết hàng", "còn hàng", "hàng mới", "hàng lỗi", "bảo hành", "chính hãng",
]


def _rand_phone(rng: random.Random) -> str:
    pref = rng.choice(["09", "03", "07", "08", "05"])
    digits = "".join(str(rng.randint(0, 9)) for _ in range(8))
    return pref + digits


def _rand_house(rng: random.Random, allow_slash: bool = True) -> str:
    r = rng.random()
    if r < 0.45:
        return str(rng.randint(1, 450))
    if allow_slash and r < 0.72:
        return f"{rng.randint(1, 280)}/{rng.randint(1, 45)}"
    if allow_slash and r < 0.88:
        return f"{rng.randint(1, 150)}/{rng.randint(1, 30)}/{rng.randint(1, 20)}"
    return f"{rng.randint(1, 220)}{rng.choice('ABCDEF')}"


def _maybe_case_noise(rng: random.Random, s: str, p_lower: float = 0.25, p_upper: float = 0.08) -> str:
    r = rng.random()
    if r < p_upper:
        return s.upper()
    if r < p_upper + p_lower:
        return s.lower()
    return s


def _apply_typo(rng: random.Random, s: str) -> str:
    ops = [
        lambda x: x.replace(", ", ","),
        lambda x: x.replace(". ", "."),
        lambda x: x.replace("Phường", "Phuong").replace("Quận", "Quan"),
        lambda x: x.replace("Đường", "Dg").replace("Số", "So"),
        lambda x: re.sub(r"\s+", "  ", x, count=1),
        lambda x: x.replace(",", ""),
    ]
    n_ops = rng.choice([1, 2])
    out = s
    for _ in range(n_ops):
        out = rng.choice(ops)(out)
    return out


def _ward_abbrev(rec: Dict[str, str], rng: random.Random) -> Tuple[str, str]:
    wt = rec["ward_type"].lower()
    if wt.startswith("ph"):
        w_pref = rng.choice(["P.", "p.", "P", "p"])
    elif "thị trấn" in wt:
        w_pref = rng.choice(["TT.", "tt.", "TT"])
    elif "đặc khu" in wt:
        w_pref = rng.choice(["ĐK", "Đặc khu"])
    else:
        w_pref = rng.choice(["X.", "x.", "X"])
    wb = rec["ward_bare"]
    if wb.lower().startswith("phường ") and wb.split()[-1].isdigit():
        num = wb.split()[-1]
        return f"{w_pref}{num}" if rng.random() < 0.5 else f"{w_pref} {num}", wb
    sep = "" if (w_pref.endswith(".") and rng.random() < 0.45) else " "
    return f"{w_pref}{sep}{wb}".strip(), wb


def _dist_abbrev(rec: Dict[str, str], rng: random.Random) -> str:
    df_low = strip_accents(rec["district_full"]).lower()
    db = rec["district_bare"]
    db_norm = strip_accents(db).lower()
    if db_norm in DISTRICT_ALIASES_REV and rng.random() < 0.55:
        return rng.choice(DISTRICT_ALIASES_REV[db_norm])
    if df_low.startswith("quan "):
        rest = df_low[5:].strip()
        if rest.isdigit():
            return rng.choice([f"Q.{rest}", f"Q{rest}", f"q.{rest}", f"q{rest}", f"Q. {rest}"])
        pref = rng.choice(["Q.", "q.", "Q "])
    elif df_low.startswith("thi xa"):
        pref = rng.choice(["TX.", "tx.", "TX "])
    elif df_low.startswith("thanh pho"):
        pref = rng.choice(["TP.", "tp.", "TP "])
    else:
        pref = rng.choice(["H.", "h.", "H "])
    sep = "" if (pref.endswith(".") and rng.random() < 0.4) else " "
    return f"{pref}{sep}{db}".strip()


def _prov_short(rec: Dict[str, str], rng: random.Random) -> str:
    pb = rec["province_bare"]
    pf = rec["province_full"]
    pb_norm = strip_accents(pb).lower()
    alias_map = {
        "ha noi": ["HN", "hn", "Hà Nội", "TP.HN", "TP Hà Nội"],
        "ho chi minh": ["TP.HCM", "TPHCM", "HCM", "hcm", "SG", "Sài Gòn", "TP Hồ Chí Minh"],
        "da nang": ["Đà Nẵng", "TP.ĐN", "TP Đà Nẵng", "ĐN"],
        "hai phong": ["Hải Phòng", "HP", "TP Hải Phòng"],
        "can tho": ["Cần Thơ", "CT", "TP Cần Thơ"],
        "hue": ["Huế", "TP Huế", "Thừa Thiên Huế"],
    }
    if pb_norm in alias_map and rng.random() < 0.6:
        return rng.choice(alias_map[pb_norm])
    return rng.choice([pb, pb, pf])


def generate_dataset_v2(
    total_samples: int = 75000,
    seed: int = 202,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    save_manifest: bool = True,
) -> Dict[str, List[SampleV2]]:
    """Generate 75,000 stratified 3-class samples with MIT GSO/2025 coverage & strict anti-leakage."""
    rng = random.Random(seed)
    guard = LeakageGuard([BENCH_V1_PATH, BENCH_V2_PATH])

    raw_records = json.loads(SHARED_RECORDS_PATH.read_text(encoding="utf-8"))
    legacy: List[Dict[str, str]] = list(raw_records["legacy_3tier"])
    merged: List[Dict[str, str]] = list(raw_records["merged_2025_2tier"])

    # Multi-word wards avoid single-syllable lexical collisions when prefixes are dropped
    district_or_prov_keys: Set[str] = set()
    for r in legacy:
        district_or_prov_keys.add(norm_key(r["district_bare"]))
        district_or_prov_keys.add(norm_key(r["province_bare"]))
    for r in merged:
        district_or_prov_keys.add(norm_key(r["province_bare"]))

    multi_legacy = [
        r for r in legacy
        if len(r["ward_bare"].split()) >= 2
        and norm_key(r["ward_bare"]) != norm_key(r["district_bare"])
        and norm_key(r["ward_bare"]) not in district_or_prov_keys
    ]
    all_wards = legacy + merged

    # Distinct province names & forms for province-only generation
    prov_forms_by_prov: Dict[str, List[str]] = {}
    for r in legacy + merged:
        pb = r["province_bare"]
        pf = r["province_full"]
        p_norm = strip_accents(pb).lower()
        forms = prov_forms_by_prov.setdefault(p_norm, [])
        for cand in (
            pf,
            pb,
            strip_accents(pb).lower(),
            strip_accents(pf).lower(),
            f"TP {pb}" if "thành phố" in pf.lower() else f"T.{pb}",
            f"TP.{pb}" if "thành phố" in pf.lower() else f"Tỉnh {pb}",
        ):
            if cand not in forms:
                forms.append(cand)
    for pb_name, al in PROVINCE_ALIASES_LIST:
        p_norm = strip_accents(pb_name).lower()
        for k, lst in prov_forms_by_prov.items():
            if p_norm in k or k in p_norm:
                if al not in lst:
                    lst.append(al)

    # Distinct district records
    seen_dist_keys: Set[Tuple[str, str]] = set()
    unique_districts: List[Dict[str, str]] = []
    for r in legacy:
        dk = (norm_key(r["district_full"]), norm_key(r["province_bare"]))
        if dk not in seen_dist_keys:
            seen_dist_keys.add(dk)
            unique_districts.append(r)

    # ------------------------------------------------------------------
    # Subclass Generators
    # ------------------------------------------------------------------
    def gen_short_ward_district_comma() -> str:
        r = rng.choice(multi_legacy)
        s = f"{r['ward_bare']}, {r['district_bare']}"
        return _maybe_case_noise(rng, s, p_lower=0.15, p_upper=0.05)

    def gen_short_ward_district_nocomma() -> str:
        r = rng.choice(multi_legacy)
        s = f"{r['ward_bare']} {r['district_bare']}"
        return _maybe_case_noise(rng, s, p_lower=0.18, p_upper=0.05)

    def gen_short_unaccented_lower() -> str:
        r = rng.choice(multi_legacy)
        if rng.random() < 0.35:
            s = f"{r['ward_bare']} {r['district_bare']} {r['province_bare']}"
        elif rng.random() < 0.5:
            s = f"{r['ward_bare']}, {r['district_bare']}"
        else:
            s = f"{r['ward_bare']} {r['district_bare']}"
        return strip_accents(s).lower()

    def gen_short_abbreviated() -> str:
        r = rng.choice(legacy)
        w_ab, _ = _ward_abbrev(r, rng)
        d_ab = _dist_abbrev(r, rng)
        sep = rng.choice([", ", " "])
        mode = rng.random()
        if mode < 0.40:
            s = f"{w_ab}{sep}{d_ab}"
        elif mode < 0.72:
            s = f"{w_ab}{sep}{d_ab}{sep}{_prov_short(r, rng)}"
        else:
            # Explicit ward abbreviation + bare district/province or numeric ward
            s = f"{w_ab}{sep}{r['district_bare']}"
        if rng.random() < 0.50:
            s = strip_accents(s).lower()
        return s

    def gen_short_3tier() -> str:
        r = rng.choice(legacy)
        w = r["ward_full"] if rng.random() < 0.45 else r["ward_bare"]
        d = r["district_full"] if rng.random() < 0.45 else r["district_bare"]
        p = r["province_full"] if rng.random() < 0.35 else _prov_short(r, rng)
        sep = rng.choice([", ", ", ", " "])
        s = f"{w}{sep}{d}{sep}{p}"
        if rng.random() < 0.25:
            s = strip_accents(s)
        return _maybe_case_noise(rng, s, p_lower=0.20, p_upper=0.05)

    def gen_short_2tier_merged() -> str:
        use_merged = rng.random() < 0.65
        if use_merged:
            r = rng.choice(merged)
        else:
            r = rng.choice(legacy)
        wb_nk = norm_key(r["ward_bare"])
        can_drop_prefix = (len(r["ward_bare"].split()) >= 2) and (wb_nk not in district_or_prov_keys)
        if not can_drop_prefix:
            w = r["ward_full"] if rng.random() < 0.72 else _ward_abbrev(r, rng)[0]
        else:
            if rng.random() < 0.50:
                w = r["ward_full"]
            elif rng.random() < 0.70:
                w = r["ward_bare"]
            else:
                w, _ = _ward_abbrev(r, rng)
        p = r["province_full"] if rng.random() < 0.30 else _prov_short(r, rng)
        sep = rng.choice([", ", ", ", " "])
        s = f"{w}{sep}{p}"
        if rng.random() < 0.28:
            s = strip_accents(s)
        return _maybe_case_noise(rng, s, p_lower=0.22, p_upper=0.04)

    def _random_addr_tail(r: Dict[str, str], include_ward: bool = True) -> str:
        sep = rng.choice([", ", ", ", " "])
        parts: List[str] = []
        if include_ward:
            parts.append(r["ward_full"] if rng.random() < 0.5 else r["ward_bare"])
        if "district_bare" in r and rng.random() < 0.88:
            parts.append(r["district_full"] if rng.random() < 0.5 else r["district_bare"])
        if rng.random() < 0.85 or not parts:
            parts.append(r["province_full"] if rng.random() < 0.35 else _prov_short(r, rng))
        return sep.join(parts)

    def gen_full_urban() -> str:
        r = rng.choice(legacy if rng.random() < 0.80 else merged)
        st = rng.choice(STREETS)
        house = _rand_house(rng)
        has_pref = rng.random() < 0.45
        st_part = f"{'Số ' if has_pref else ''}{house} {'Đường ' if rng.random() < 0.45 else ''}{st}"
        inc_w = rng.random() < 0.72 if "district_bare" in r else True
        tail = _random_addr_tail(r, include_ward=inc_w)
        s = f"{st_part}{rng.choice([', ', ' '])}{tail}"
        if rng.random() < 0.30:
            s = strip_accents(s)
        return _maybe_case_noise(rng, s, p_lower=0.25, p_upper=0.04)

    def gen_full_rural() -> str:
        r = rng.choice(legacy if rng.random() < 0.75 else merged)
        mode = rng.random()
        if mode < 0.68:
            unit = rng.choice([
                f"Thôn {rng.randint(1, 14)}",
                f"Thôn {rng.choice(['Đoài', 'Đông', 'Nam', 'Tây', 'Bắc', 'Trung', 'Mới', 'Thượng', 'Hạ'])}",
                f"Ấp {rng.randint(1, 9)}",
                f"Ấp {rng.choice(['Tân Lập', 'Mới 1', 'Chợ', 'Bắc', 'Nam', 'Bình Hòa', 'Phước Thạnh'])}",
                f"Xóm {rng.randint(1, 12)}",
                f"Bản {rng.choice(['Lác', 'Cát', 'Hua', 'Mường', 'Poom', 'Nà', 'Khau'])}",
                f"Khóm {rng.randint(1, 8)}",
                f"Tổ {rng.randint(1, 25)}",
                f"Tổ dân phố {rng.randint(1, 18)}",
                f"KP{rng.randint(1, 9)}",
                f"Khu phố {rng.randint(1, 9)}",
            ])
        else:
            lot = rng.choice([
                f"Lô {rng.choice('ABCDEF')}{rng.randint(1, 24)}",
                f"Xưởng {rng.randint(1, 12)}",
                f"Đường số {rng.randint(1, 18)}",
            ])
            unit = f"{lot}, {rng.choice(KCN_NAMES)}"
        tail = _random_addr_tail(r, include_ward=True)
        s = f"{unit}{rng.choice([', ', ' '])}{tail}"
        if rng.random() < 0.25:
            s = strip_accents(s)
        return _maybe_case_noise(rng, s, p_lower=0.22, p_upper=0.04)

    def gen_full_apartment() -> str:
        r = rng.choice(legacy)
        bldg = rng.choice(BUILDINGS)
        st = rng.choice(STREETS)
        house = _rand_house(rng, allow_slash=False)
        apt = rng.choice([
            f"Căn hộ {rng.choice('ABC')}{rng.randint(1, 28)}.{rng.randint(1, 16):02d}",
            f"Phòng {rng.randint(2, 32)}{rng.randint(1, 18):02d}",
            f"P.{rng.randint(2, 30)}{rng.randint(1, 16):02d}",
            f"Tầng {rng.randint(2, 35)}",
            f"Lầu {rng.randint(1, 25)}",
            f"Block {rng.choice('ABCD')}{rng.randint(1, 4)}",
            f"Tháp {rng.choice(['T1', 'T2', 'S1.02', 'S2.05', 'Ruby', 'Park 3'])}",
        ])
        sep = rng.choice([", ", " "])
        mid = f"{house} {st}{sep}" if rng.random() < 0.55 else ""
        tail = _random_addr_tail(r, include_ward=rng.random() < 0.60)
        s = f"{apt}{sep}{bldg}{sep}{mid}{tail}"
        if rng.random() < 0.30:
            s = strip_accents(s)
        return _maybe_case_noise(rng, s, p_lower=0.22, p_upper=0.04)

    def gen_full_alley_typo() -> str:
        r = rng.choice(legacy)
        st = rng.choice(STREETS)
        deep = rng.choice([
            f"Hẻm {rng.randint(1, 250)}/{rng.randint(1, 40)}/{rng.randint(1, 18)} {st}",
            f"Ngõ {rng.randint(1, 220)}/{rng.randint(1, 30)} {st}",
            f"Ngách {rng.randint(1, 60)}/{rng.randint(1, 25)} ngõ {rng.randint(1, 180)} {st}",
            f"Kiệt {rng.randint(1, 120)}/{rng.randint(1, 25)} {st}",
            f"{rng.randint(1, 200)}/{rng.randint(1, 30)}/{rng.randint(1, 15)}{rng.choice('AB')} {st}",
            f"Số {rng.randint(1, 90)}/{rng.randint(1, 30)}{rng.choice('BC')} {st}",
        ])
        tail = _random_addr_tail(r, include_ward=rng.random() < 0.70)
        s = _apply_typo(rng, f"{deep}, {tail}")
        if rng.random() < 0.40:
            s = strip_accents(s)
        return _maybe_case_noise(rng, s, p_lower=0.35, p_upper=0.10)

    def gen_full_delivery_freetext() -> str:
        r = rng.choice(legacy if rng.random() < 0.85 else merged)
        st = rng.choice(STREETS)
        house = _rand_house(rng)
        nm = f"{rng.choice(VN_SURNAMES)} {rng.choice(VN_MIDDLES)} {rng.choice(VN_GIVEN)}"
        short_nm = f"{rng.choice(['anh', 'chị', 'bạn', 'em', 'cô', 'chú'])} {rng.choice(VN_GIVEN).lower()}"
        ph = _rand_phone(rng)
        nt = rng.choice(SHIPPER_NOTES)
        if rng.random() < 0.25 and "district_bare" in r:
            addr = f"{r['ward_bare']}, {r['district_bare']}, {r['province_bare']}"
        else:
            addr = f"{house} {st}, {_random_addr_tail(r, include_ward=rng.random() < 0.7)}"
        d_p = f"{r['district_bare']} {r['province_bare']}" if "district_bare" in r else r["province_bare"]
        no_comma_addr = f"{house} {st} {d_p}"
        bldg = rng.choice(BUILDINGS)
        apt = rng.choice([
            f"phòng {rng.randint(1, 28)}{rng.randint(1, 16):02d}",
            f"căn {rng.choice('ABC')}{rng.randint(1, 20)}.{rng.randint(1, 12):02d}",
            f"tầng {rng.randint(2, 30)}",
        ])
        style = rng.random()
        if style < 0.18:
            s = f"{nm} - {ph} - {addr} ({nt})"
        elif style < 0.34:
            s = f"nhận tại {addr} - {nt}"
        elif style < 0.48:
            s = f"gửi về {addr}, người nhận {nm} {ph}"
        elif style < 0.62:
            s = f"ship qua {addr} giùm mình nha, đt {ph}"
        elif style < 0.74:
            s = f"địa chỉ nhận hàng: {addr} - {nt} (liên hệ {ph})"
        elif style < 0.85:
            s = f"giao cho {short_nm} ở {no_comma_addr} {rng.choice(['nhé shop', 'nha', 'giúp em', 'nhé'])}"
        elif style < 0.93:
            landmark = rng.choice(["đại học", "bệnh viện", "siêu thị", "trường tiểu học", "công viên", "nhà văn hóa"])
            gate = rng.choice(["cổng sau", "cổng chính", "đối diện", "cạnh"])
            s = f"{gate} {landmark} {no_comma_addr}"
        else:
            s = f"{rng.choice(['ship tới', 'giao qua', 'gửi tới'])} {apt} {bldg} {rng.choice([f'gọi {ph}', d_p])}"
        if rng.random() < 0.38:
            s = strip_accents(s)
        return _maybe_case_noise(rng, s, p_lower=0.32, p_upper=0.02)

    # --- Label 1: address_insufficient ---
    def gen_insuf_ward_only() -> str:
        r = rng.choice(all_wards)
        mode = rng.random()
        if mode < 0.45:
            s = r["ward_full"]
        elif mode < 0.72:
            s, _ = _ward_abbrev(r, rng)
        elif mode < 0.88:
            # Numeric wards with explicit p./phường prefix
            n = rng.randint(1, 28)
            s = rng.choice([f"Phường {n}", f"p.{n}", f"p{n}", f"P. {n}"])
        else:
            s = strip_accents(r["ward_full"]).lower()
        if rng.random() < 0.30:
            s = strip_accents(s)
        return _maybe_case_noise(rng, s, p_lower=0.25, p_upper=0.05)

    def gen_insuf_district_only() -> str:
        r = rng.choice(unique_districts)
        mode = rng.random()
        if mode < 0.42:
            s = r["district_full"]
        elif mode < 0.62:
            s = _dist_abbrev(r, rng)
        else:
            # District + Province without ward or street
            sep = rng.choice([", ", " "])
            d_txt = r["district_full"] if rng.random() < 0.35 else r["district_bare"]
            p_txt = r["province_full"] if rng.random() < 0.20 else _prov_short(r, rng)
            s = f"{d_txt}{sep}{p_txt}"
        if rng.random() < 0.35:
            s = strip_accents(s)
        return _maybe_case_noise(rng, s, p_lower=0.25, p_upper=0.05)

    def gen_insuf_fragment_only() -> str:
        n1 = rng.randint(1, 250)
        n2 = rng.randint(1, 45)
        st = rng.choice(STREETS)
        # Ensure street-only & granular-only fragments have balanced representation
        if rng.random() < 0.22:
            s = rng.choice([f"đường {st}", f"phố {st}", f"Đường {st}", f"đại lộ {st}"])
            if rng.random() < 0.30:
                s = strip_accents(s)
            return _maybe_case_noise(rng, s, p_lower=0.35, p_upper=0.05)
        templates = [
            lambda: f"ngõ {n1}",
            lambda: f"Ngõ {n1}/{n2}",
            lambda: f"Ngách {n1}/{n2}",
            lambda: f"hẻm {n1}",
            lambda: f"Hẻm {n1}/{n2}",
            lambda: f"Kiệt {n1}",
            lambda: f"Thôn {rng.randint(1, 15)}",
            lambda: f"thôn {rng.choice(['Đoài', 'Đông', 'Nam', 'Bắc', 'Mới', 'Trung'])} {n2}",
            lambda: f"khu phố {rng.randint(1, 12)}",
            lambda: f"KP {rng.randint(1, 12)}",
            lambda: f"xóm {rng.randint(1, 14)}",
            lambda: f"ấp {rng.randint(1, 10)}",
            lambda: f"Ấp {rng.choice(['Tân Lập', 'Mới', 'Bắc', 'Trung', 'Bình Hòa'])} {n2}",
            lambda: f"tổ {rng.randint(1, 30)}",
            lambda: f"tổ dân phố {rng.randint(1, 20)}",
            lambda: f"tầng {rng.randint(1, 35)}",
            lambda: f"lầu {rng.randint(1, 25)}",
            lambda: f"phòng {rng.randint(1, 25)}{rng.randint(1, 15):02d}",
            lambda: f"căn hộ {rng.choice('ABCD')}{rng.randint(1, 25)}{rng.randint(1, 14):02d}",
            lambda: f"lô {rng.choice('ABCDEF')}{rng.randint(1, 24)}",
            lambda: f"chung cư {rng.choice('ABC')}{rng.randint(1, 15)}",
            lambda: f"tòa nhà {rng.choice('ABCD')}{rng.randint(1, 12)}",
            lambda: f"số {n1} đường {st}",
            lambda: f"số nhà {n1}",
            lambda: f"ngõ {n1} {st}",
        ]
        s = rng.choice(templates)()
        if rng.random() < 0.35:
            s = strip_accents(s)
        return _maybe_case_noise(rng, s, p_lower=0.35, p_upper=0.05)

    # --- Label 0: not_address ---
    def gen_not_short_garbage() -> str:
        r = rng.random()
        if r < 0.25:
            return str(rng.randint(0, 9999))
        if r < 0.45:
            return rng.choice(SHORT_WORDS)
        if r < 0.65:
            w = rng.choice(SHORT_WORDS).upper() if rng.random() < 0.5 else rng.choice(SHORT_WORDS).capitalize()
            return w
        if r < 0.80:
            # Single letter or 2-letter code
            return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") for _ in range(rng.choice([1, 2, 3])))
        # Random gibberish / alphanumeric token
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        return "".join(rng.choice(chars) for _ in range(rng.randint(3, 10)))

    def gen_not_chat_order() -> str:
        rec = rng.choice(all_wards)
        place = rng.choice([rec["province_bare"], rec["district_bare"] if "district_bare" in rec else rec["province_bare"], rec["ward_full"]])
        qty = rng.randint(1, 12)
        item = rng.choice([
            "chai nước ngọt", "chai sốt nướng BBQ", "chai nước ép", "chai trà sữa",
            "ly trà sữa", "áo thun", "hộp bánh", "túi xách", "combo", "gói snack", "cái ốp lưng", "phần cơm nướng",
        ])
        color = rng.choice(["đen", "trắng", "xanh", "đỏ", "be", "xám", "nâu", "hồng"])
        sz = rng.choice(["S", "M", "L", "XL", "39", "40", "41", "42"])
        noun = rng.choice([
            "áo sơ mi", "áo khoác", "quần jean", "giày thể thao", "túi xách", "ví da",
            "son môi", "kem chống nắng", "nước hoa", "tai nghe", "bàn phím", "cáp sạc",
            "ốp lưng", "chuột không dây", "bình giữ nhiệt", "hộp cơm", "đèn bàn", "thảm tập",
        ])
        templates = [
            lambda: rng.choice(["Xin giá", "xin giá ạ", "inbox giá giúp mình", "báo giá sỉ đi shop", "giá sao vậy bạn"]),
            lambda: f"Cho mình {qty} {item} nha shop",
            lambda: f"Cho mình {qty} {item}",
            lambda: f"Lấy {qty} {item} size {sz}",
            lambda: f"{noun} màu {color} còn không {rng.choice(['shop', 'bạn', 'ạ'])}?",
            lambda: f"đặt {qty} cái {noun} màu {color} size {sz}",
            lambda: f"gửi {rng.choice(['ảnh thật', 'bảng màu', 'link sản phẩm', 'video'])} giúp mình nhé",
            lambda: f"{rng.choice(['tổng đơn', 'tổng tiền', 'hết tất cả', 'tiền hàng'])} hết bao nhiêu {rng.choice(['ạ', 'vậy shop', 'nhỉ'])}",
            lambda: f"{rng.choice(['bao giờ', 'khi nào', 'mấy hôm nữa'])} có hàng {noun} lại vậy {rng.choice(['shop', 'bạn'])}?",
            lambda: f"{rng.choice(['đã chuyển khoản', 'ck cọc', 'thanh toán trước'])} {rng.choice(['rồi nhé', 'xong rồi ạ', 'cho shop rồi'])}",
            lambda: f"{rng.choice(['đơn hàng', 'kiện hàng', 'món đồ'])} của em {rng.choice(['tới đâu rồi', 'gửi chưa ạ', 'đang ở kho nào'])}",
            lambda: f"{rng.choice(['ship cod', 'freeship', 'giao nhanh', 'đổi trả'])} {rng.choice(['không ạ', 'nha shop', 'được không bạn'])}",
            lambda: f"{rng.choice(['giảm giá', 'tặng mã', 'ưu đãi'])} thêm đi {rng.choice(['shop ơi', 'bạn nhé', 'ạ'])}",
            lambda: f"{rng.choice(['cảm ơn', 'ok luôn', 'vâng ạ', 'đúng rồi ạ', 'không cần gấp đâu', 'hẹn cuối tuần nhé'])} {rng.choice(['shop', 'bạn ơi', 'nhé', ''])}".strip(),
            lambda: f"phí ship về {place} hết bao nhiêu vậy bạn?",
            lambda: f"cước gửi đi {place} tính thế nào ạ?",
            lambda: f"shop mình có hỗ trợ giao về {place} không?",
            lambda: f"gửi hàng đi {place} khoảng mấy ngày nhận?",
            lambda: f"cửa hàng có chi nhánh tại {place} không ạ?",
            lambda: f"đơn hàng #{rng.randint(10000, 999999)} giao về {place} tới đâu rồi?",
            lambda: f"mã giảm giá đơn từ {rng.choice([99, 150, 200, 300])}k là gì vậy?",
        ]
        s = rng.choice(templates)()
        if rng.random() < 0.35:
            s = strip_accents(s).lower()
        return s

    def gen_not_food_product() -> str:
        core_foods = [
            "bánh mì", "phở", "phở bò", "bún bò", "nước mắm", "kẹo dừa", "nem chua",
            "mực một nắng", "mực khô", "cơm tấm", "hủ tiếu", "mì Quảng", "chè xanh",
            "bánh đậu xanh", "chả mực", "bánh tráng", "bánh pía", "yến sào",
            "hạt điều", "cà phê", "sầu riêng", "xoài cát", "bưởi da xanh", "gạo ST25",
            "bánh cốm", "bánh đa cua", "chả bò", "bánh cuốn", "nem nướng", "tôm chua",
        ]
        food_places = [
            "Hà Nội", "Sài Gòn", "Đà Nẵng", "Huế", "Hải Phòng", "Cần Thơ", "Cà Mau",
            "Phú Quốc", "Bến Tre", "Thanh Hóa", "Thái Nguyên", "Hải Dương", "Hạ Long",
            "Nha Trang", "Đà Lạt", "Buôn Ma Thuột", "Sóc Trăng", "Hưng Yên", "Bắc Giang",
            "Quảng Ngãi", "Bình Phước", "Tiền Giang", "Nam Định", "Nghệ An", "Ninh Bình",
            "Vũng Tàu", "Phan Thiết", "Bình Định", "Hà Tĩnh", "Quy Nhơn", "Long An",
        ]
        if rng.random() < 0.65:
            if rng.random() < 0.55:
                food = rng.choice(core_foods)
                place = rng.choice(food_places)
            else:
                food, place = rng.choice(REGIONAL_FOODS)
            mode = rng.random()
            if mode < 0.56:
                s = f"{food} {place}"
            elif mode < 0.78:
                qty = rng.choice([f"{rng.randint(1, 5)} hộp", f"{rng.randint(1, 5)}kg", f"{rng.randint(1, 6)} chai", f"{rng.randint(1, 10)} gói", f"{rng.randint(2, 8)} bịch"])
                s = f"đặt {qty} {food} {place} loại {rng.choice(['đặc biệt', '1', 'ngon'])}"
            elif mode < 0.90:
                s = f"{food} {place} giá {rng.choice([45, 75, 99, 120, 185, 250, 320])}k"
            else:
                s = f"Món {food} ở {place} ăn rất hợp khẩu vị và thơm ngon"
        else:
            prod = rng.choice(GENERAL_PRODUCTS)
            color = rng.choice(["đen", "trắng", "xanh", "be", "xám", "titan tự nhiên", "đỏ cam", "nâu"])
            mat = rng.choice(["da bò", "cotton", "inox", "hợp kim", "không dây", "cỡ lớn", "trong suốt", "mini"])
            mod = rng.choice([
                f"màu {color} {mat}",
                f"màu {color} size {rng.choice(['S', 'M', 'L', 'XL', '41'])}",
                f"chất liệu {mat} cao cấp",
                f"bảo hành {rng.choice([6, 12, 24])} tháng chính hãng",
                f"bản {rng.choice(['128GB', '256GB', '512GB'])} chính hãng VN/A",
                f"giá {rng.choice([79, 149, 250, 490, 890])}k",
                f"combo {rng.randint(2, 4)} cái",
            ])
            s = f"{prod} {mod}"
        if rng.random() < 0.25:
            s = strip_accents(s).lower()
        return s

    def gen_not_person_bio() -> str:
        name = rng.choice(COLLIDING_PERSON_NAMES)
        full_rand = f"{rng.choice(VN_SURNAMES)} {rng.choice(VN_MIDDLES)} {rng.choice(VN_GIVEN)}"
        mode = rng.random()
        if mode < 0.25:
            s = rng.choice([name, full_rand])
        elif mode < 0.50:
            role = rng.choice(["Anh", "Chị", "Bạn", "Em", "Cô", "Chú", "Ông", "Thầy"])
            action = rng.choice([
                "vừa gọi điện hỏi đơn", "nhờ kiểm tra lại lịch hẹn",
                "đang làm trưởng nhóm dự án", "sinh năm 1995", "mới đặt 2 sản phẩm",
            ])
            s = f"{role} {name} {action}"
        elif mode < 0.78:
            s = rng.choice([
                f"{name} là danh nhân lịch sử tiêu biểu trong sách giáo khoa",
                f"{name} là anh hùng dân tộc được ghi nhớ trong lịch sử",
                f"{name} là chiến sĩ cách mạng kiên trung thời kỳ kháng chiến",
                f"Cuốn tiểu sử viết về cuộc đời của {name}",
                f"Trường THPT {name} vừa công bố điểm chuẩn lớp 10",
                f"Bài thuyết trình hôm nay nói về công lao của {name}",
                f"Bệnh viện {name} mở rộng khu khám bệnh theo yêu cầu",
                f"Học sinh tìm hiểu sự nghiệp của danh nhân {name}",
            ])
        else:
            s = f"{full_rand} - {rng.choice(['Kế toán', 'Kỹ sư phần mềm', 'Nhân viên kinh doanh', 'Giáo viên', 'Bác sĩ'])}"
        if rng.random() < 0.25:
            s = strip_accents(s).lower()
        return s

    def gen_not_narrative_place() -> str:
        r1 = rng.choice(legacy)
        r2 = rng.choice(legacy)
        p1 = r1["province_bare"]
        p2 = r2["province_bare"] if r2["province_bare"] != p1 else "Đà Nẵng"
        d1 = r1["district_full"] if rng.random() < 0.5 else r1["district_bare"]
        d2 = r2["district_bare"] if r2["district_bare"] != r1["district_bare"] else "Thanh Xuân"
        st = rng.choice(STREETS)
        n = rng.randint(2, 6)
        templates = [
            f"Thời tiết tại {p1} sáng nay mát mẻ có nắng nhẹ",
            f"Trời {p1} hôm nay mưa nặng hạt suốt cả buổi chiều",
            f"Mùa mưa ở {p1} chiều nào cũng có gió mạnh và dông",
            f"Khu vực {d1} chiều qua xuất hiện mưa rào rải rác",
            f"Gia đình dự định đi nghỉ mát tại {p1} {n} ngày",
            f"Chuyến công tác ra {p1} tuần sau mới xong việc",
            f"Tuyến xe khách chạy từ {p1} về {p2} tăng chuyến dịp lễ",
            f"Vé tàu xe từ {p1} sang {p2} cuối tuần tăng nhẹ",
            f"Đoạn đường {st} sáng thứ hai thường bị ùn ứ giờ cao điểm",
            f"Ngã tư khu vực {d1} giờ tan tầm hay kẹt cứng xe cộ",
            f"Giá thuê mặt bằng khu vực {d1} biến động nhẹ quý này",
            f"Thị trường bất động sản tại {p1} đầu năm giao dịch chậm",
            f"Doanh nghiệp vừa khai trương thêm văn phòng đại diện tại {p1}",
            f"Công ty mình hiện mở chi nhánh ở cả {d1} lẫn {d2}",
            f"Phòng nhân sự tuyển ứng viên làm việc trực tiếp tại {d1}",
            f"Quê nội ở {p1} nhưng hiện công tác lâu dài ở {p2}",
            f"Dịp lễ sắp tới cả nhà lại về quê {p1} sum họp",
            f"Sinh viên các trường đại học khu vực {d1} bước vào kỳ thi cuối kỳ",
            f"Bạn mình đang tìm phòng trọ sinh viên quanh khu {d1}",
            f"Lễ hội văn hóa ẩm thực cuối tuần tổ chức tại quảng trường {p1}",
            f"Đội bóng đá thi đấu trận giao hữu trên sân vận động {p1}",
            f"Tối thứ bảy có màn bắn pháo hoa nghệ thuật bên sông tại {p1}",
            f"Bưu cục giao nhận tại {d1} tạm nghỉ kiểm kê vào sáng chủ nhật",
            f"Cửa hàng bên {d1} hôm nay đóng cửa sớm để bảo trì",
            f"Khách sạn ven biển ở {p1} mùa hè thường kín phòng rất sớm",
            f"Người dân ở {p1} rất thân thiện còn ở {p2} lại hiền hòa",
            f"Cây cầu bắc qua sông ở {p1} lên đèn rực rỡ mỗi tối",
        ]
        s = rng.choice(templates)
        if rng.random() < 0.30:
            s = strip_accents(s).lower()
        return s

    def gen_not_contact_meta_vague() -> str:
        ph = _rand_phone(rng)
        oid = rng.randint(10000, 999999)
        d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(2023, 2027)
        price = rng.choice([45, 89, 120, 199, 350, 650, 1290])
        mode = rng.random()
        if mode < 0.22:
            return rng.choice([
                ph,
                f"{ph[:4]} {ph[4:7]} {ph[7:]}",
                f"+84 {ph[1:4]} {ph[4:7]} {ph[7:]}",
                f"zalo {ph}",
                f"SĐT: {ph}",
            ])
        if mode < 0.42:
            user = strip_accents(f"{rng.choice(VN_SURNAMES)}{rng.choice(VN_GIVEN)}").lower() + str(rng.randint(1, 999))
            dom = rng.choice(["gmail.com", "yahoo.com", "outlook.com", "company.vn"])
            return rng.choice([
                f"{user}@{dom}",
                f"https://example-shop.vn/don-hang/{oid}",
                f"https://shopee.vn/item/{oid}",
                f"www.giaohang{rng.randint(1, 99)}.vn",
            ])
        if mode < 0.65:
            track = f"{rng.choice(['GHN', 'GHTK', 'SPX', 'VTP', 'JT'])}{rng.randint(10000000, 99999999)}"
            return rng.choice([
                track,
                f"Mã vận đơn {track} đang luân chuyển",
                f"CK tiền cọc đơn #{oid} ngân hàng Techcombank",
                f"Chuyển khoản thanh toán hóa đơn {oid}",
                f"Mã OTP xác nhận giao dịch là {rng.randint(100000, 999999)}",
            ])
        if mode < 0.82:
            return rng.choice([
                f"{d:02d}/{m:02d}/{y}",
                f"{d:02d}-{m:02d}-{y} {rng.randint(8, 21):02d}:30",
                f"{price}.000đ",
                f"{price}.000 VND",
                f"Tổng tiền {price}k (đã bao gồm VAT)",
                f"Kích thước {rng.randint(10, 50)}x{rng.randint(10, 40)}cm",
            ])
        return rng.choice([
            "ở chỗ cũ", "như đơn lần trước", "giao như địa chỉ cũ nhé",
            "nhà sơn màu xanh cạnh tiệm tạp hóa", "đối diện cổng trường cấp 2",
            "tới đầu đường gọi em chạy ra lấy", "tới nơi alo mình xuống sảnh nhận",
            "gửi bảo vệ tòa nhà ký nhận hộ", "nhờ lễ tân nhận giúp kiện hàng",
            "lát nữa mình nhắn địa chỉ cụ thể sau", "chút xem lại rồi gửi địa chỉ sau",
            "hẹn gặp lúc 7h tối ở quán cafe", "tối nay rảnh qua quán trà sữa ngồi nhé",
            "giao ngoài giờ hành chính giúp mình", "chờ mình về nhà rồi hãy giao",
        ])

    # ------------------------------------------------------------------
    # Allocation Plan (Exact 75,000 samples: 45% suff, 25% insuf, 30% not)
    # ------------------------------------------------------------------
    n_suff = int(round(total_samples * 0.45))   # 33,750
    n_insuf = int(round(total_samples * 0.25))  # 18,750
    n_not = total_samples - n_suff - n_insuf    # 22,500

    # Standard generators (strict global deduplication by norm_key across splits)
    plan: List[Tuple[int, str, str, int, Callable[[], str], bool]] = [
        # (label, group, subgroup, count, fn, check_4grams)
        (2, "short_valid", "short_ward_district_comma", 3000, gen_short_ward_district_comma, False),
        (2, "short_valid", "short_ward_district_nocomma", 3000, gen_short_ward_district_nocomma, False),
        (2, "short_valid", "short_unaccented_lower", 3000, gen_short_unaccented_lower, False),
        (2, "short_valid", "short_abbreviated", 3000, gen_short_abbreviated, False),
        (2, "short_valid", "short_3tier", 3000, gen_short_3tier, False),
        (2, "short_valid", "short_2tier_merged", 3000, gen_short_2tier_merged, False),
        (2, "full_address", "full_urban", 3500, gen_full_urban, False),
        (2, "full_address", "full_rural", 3250, gen_full_rural, False),
        (2, "full_address", "full_apartment", 2750, gen_full_apartment, False),
        (2, "full_address", "full_alley_typo", 2750, gen_full_alley_typo, False),
        (2, "full_address", "full_delivery_freetext", 3500, gen_full_delivery_freetext, True),

        (1, "insufficient", "insuf_ward_only", 6000, gen_insuf_ward_only, False),
        # insuf_province_only (1,500) handled separately below for strict key-disjointness
        (1, "insufficient", "insuf_district_only", 5250, gen_insuf_district_only, False),
        (1, "insufficient", "insuf_fragment_only", 6000, gen_insuf_fragment_only, False),

        (0, "not_address", "not_short_garbage", 2800, gen_not_short_garbage, True),
        (0, "not_address", "not_chat_order", 4800, gen_not_chat_order, True),
        (0, "not_address", "not_food_product", 5200, gen_not_food_product, True),
        (0, "not_address", "not_person_bio", 3400, gen_not_person_bio, True),
        (0, "not_address", "not_narrative_place", 3500, gen_not_narrative_place, True),
        (0, "not_address", "not_contact_meta_vague", 2800, gen_not_contact_meta_vague, True),
    ]
    assert sum(p[3] for p in plan if p[0] == 2) == n_suff
    assert sum(p[3] for p in plan if p[0] == 1) + 1500 == n_insuf
    assert sum(p[3] for p in plan if p[0] == 0) == n_not

    seen_norm_keys: Set[str] = set()
    seen_exact_lower: Set[str] = set()
    train_list: List[SampleV2] = []
    val_list: List[SampleV2] = []
    test_list: List[SampleV2] = []

    # 1. Build insuf_province_only (1,500) strictly disjoint by norm_key across splits
    prov_pool: Dict[str, List[str]] = {}
    for forms in prov_forms_by_prov.values():
        for f in forms:
            for variant in (f, f.upper(), f.lower(), f.title()):
                if guard.is_leaked(variant, check_4grams=False):
                    continue
                nk = norm_key(variant)
                prov_pool.setdefault(nk, [])
                if variant not in prov_pool[nk]:
                    prov_pool[nk].append(variant)
    prov_keys = sorted(prov_pool.keys())
    rng.shuffle(prov_keys)
    n_pk = len(prov_keys)
    n_pk_test = max(10, int(round(n_pk * test_ratio)))
    n_pk_val = max(10, int(round(n_pk * val_ratio)))
    pk_splits = {
        "test": prov_keys[:n_pk_test],
        "val": prov_keys[n_pk_test : n_pk_test + n_pk_val],
        "train": prov_keys[n_pk_test + n_pk_val :],
    }
    prov_counts = {"train": 1200, "val": 150, "test": 150}
    for sp_name, target_cnt in prov_counts.items():
        keys = pk_splits[sp_name]
        seen_norm_keys.update(keys)
        sp_samples: List[SampleV2] = []
        idx = 0
        while len(sp_samples) < target_cnt:
            k = keys[idx % len(keys)]
            cand = rng.choice(prov_pool[k])
            sp_samples.append(SampleV2(text=cand, label=1, group="insufficient", subgroup="insuf_province_only"))
            idx += 1
        if sp_name == "train":
            train_list.extend(sp_samples)
        elif sp_name == "val":
            val_list.extend(sp_samples)
        else:
            test_list.extend(sp_samples)

    # 2. Fill all remaining subgroups with strict global uniqueness & leakage guard
    for label, group, subgroup, count, fn, check_4g in plan:
        bucket: List[SampleV2] = []
        attempts = 0
        max_attempts = count * 30
        while len(bucket) < count and attempts < max_attempts:
            attempts += 1
            txt = re.sub(r"\s+", " ", fn()).strip()
            if not txt:
                continue
            if guard.is_leaked(txt, check_4grams=check_4g):
                continue
            nk = norm_key(txt)
            el = txt.lower()
            if nk not in seen_norm_keys:
                seen_norm_keys.add(nk)
                seen_exact_lower.add(el)
                bucket.append(SampleV2(text=txt, label=label, group=group, subgroup=subgroup))
            elif attempts > count * 16 and el not in seen_exact_lower:
                # Only if unique normalized space saturates for small finite pools (e.g. short_garbage)
                seen_exact_lower.add(el)
                bucket.append(SampleV2(text=txt, label=label, group=group, subgroup=subgroup))

        # Partition strictly by norm_key so identical normalized strings never span splits
        by_nk: Dict[str, List[SampleV2]] = {}
        for s in bucket:
            by_nk.setdefault(norm_key(s.text), []).append(s)
        nk_groups = list(by_nk.values())
        rng.shuffle(nk_groups)

        target_test = int(round(len(bucket) * test_ratio))
        target_val = int(round(len(bucket) * val_ratio))

        sp_test: List[SampleV2] = []
        sp_val: List[SampleV2] = []
        sp_train: List[SampleV2] = []
        for grp_items in nk_groups:
            if len(sp_test) < target_test:
                sp_test.extend(grp_items)
            elif len(sp_val) < target_val:
                sp_val.extend(grp_items)
            else:
                sp_train.extend(grp_items)

        # If exact rounding overflowed due to multi-item group, trim or rebalance cleanly
        while len(sp_test) < target_test and sp_train:
            sp_test.append(sp_train.pop())
        while len(sp_val) < target_val and sp_train:
            sp_val.append(sp_train.pop())

        test_list.extend(sp_test)
        val_list.extend(sp_val)
        train_list.extend(sp_train)

    rng.shuffle(train_list)
    rng.shuffle(val_list)
    rng.shuffle(test_list)

    if save_manifest:
        MANIFEST_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MANIFEST_OUT_PATH.open("w", encoding="utf-8") as f:
            for s in train_list:
                f.write(s.text.replace("\r", " ").replace("\n", " ") + "\n")

    return {"train": train_list, "val": val_list, "test": test_list}


if __name__ == "__main__":
    splits = generate_dataset_v2(total_samples=75000, seed=202, save_manifest=True)
    for split_name, items in splits.items():
        cnt = Counter(s.label for s in items)
        print(
            f"{split_name:5s}: total={len(items):5d} | "
            f"0(not)={cnt[0]:5d} ({cnt[0]/len(items)*100:4.1f}%) | "
            f"1(insuf)={cnt[1]:5d} ({cnt[1]/len(items)*100:4.1f}%) | "
            f"2(suff)={cnt[2]:5d} ({cnt[2]/len(items)*100:4.1f}%)"
        )
    print("\nSubgroup counts (train):")
    sg_cnt = Counter((s.label, s.subgroup) for s in splits["train"])
    for (lbl, sg), n in sorted(sg_cnt.items()):
        print(f"  label={lbl} {sg:28s}: {n}")

    # Verify zero norm_key overlap across splits
    t_keys = {norm_key(s.text) for s in splits["train"]}
    v_keys = {norm_key(s.text) for s in splits["val"]}
    te_keys = {norm_key(s.text) for s in splits["test"]}
    print("\nCross-split norm_key overlaps:")
    print("  train & val :", len(t_keys & v_keys))
    print("  train & test:", len(t_keys & te_keys))
    print("  val   & test:", len(v_keys & te_keys))
