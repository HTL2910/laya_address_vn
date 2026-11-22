"""Token-aligned synthetic Vietnamese e-commerce address & holdout generator."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from slot_parser_v3.tokenizer import (
    BIO_TAGS,
    TAG2ID,
    strip_vietnamese_accents,
    tokenize_words,
)

DEFAULT_RECORDS_PATH = (
    Path(__file__).resolve().parents[2]
    / "shared_data_v3"
    / "vn_admin_v3_records.json"
)

# Disjoint pools between train/val/test (seed=401) and independent holdout (seed=20261201).
_TRAIN_STREETS: Tuple[str, ...] = (
    "Nguyễn Trãi",
    "Lê Lợi",
    "Trần Hưng Đạo",
    "Hai Bà Trưng",
    "Nguyễn Huệ",
    "Cách Mạng Tháng 8",
    "Điện Biên Phủ",
    "Nguyễn Thị Minh Khai",
    "Lý Thường Kiệt",
    "Võ Văn Kiệt",
    "Phạm Văn Đồng",
    "Lê Văn Sỹ",
    "Hoàng Văn Thụ",
    "Nguyễn Văn Linh",
    "Trường Chinh",
    "Giải Phóng",
    "Xuân Thủy",
    "Cầu Giấy",
    "Kim Mã",
    "Láng Hạ",
    "Chùa Bộc",
    "Thái Hà",
    "Tây Sơn",
    "Bà Triệu",
    "Phố Huế",
    "Đào Tấn",
    "Nguyễn Chí Thanh",
    "Trần Duy Hưng",
    "Khuất Duy Tiến",
    "Lê Đức Thọ",
    "Võ Thị Sáu",
    "Nam Kỳ Khởi Nghĩa",
    "Pasteur",
    "Hai Tháng Chín",
    "Nguyễn Văn Thoại",
    "Bạch Đằng",
    "Lê Duẩn",
    "Quang Trung",
    "Phan Văn Trị",
    "Nơ Trang Long",
    "Xô Viết Nghệ Tĩnh",
    "Đinh Tiên Hoàng",
    "Nguyễn Hữu Thọ",
    "Nguyễn Xiển",
    "Tố Hữu",
    "Hồ Tùng Mậu",
    "Phạm Hùng",
    "Hoàng Quốc Việt",
    "Đỗ Xuân Hợp",
    "Lã Xuân Oai",
    "Nguyễn Duy Trinh",
    "Lê Văn Việt",
    "Võ Chí Công",
    "Huỳnh Tấn Phát",
    "Nguyễn Tất Thành",
    "Trần Phú",
    "Nguyễn Công Trứ",
    "Lê Hồng Phong",
    "Nguyễn Bỉnh Khiêm",
    "Tô Hiến Thành",
    "Sư Vạn Hạnh",
    "3 Tháng 2",
    "Lý Thái Tổ",
    "Hùng Vương",
    "Nguyễn Khuyến",
    "Tạ Quang Bửu",
    "Bùi Viện",
    "Phạm Ngũ Lão",
    "Cộng Hòa",
    "Trường Sơn",
    "Nguyễn Oanh",
    "Thống Nhất",
    "Kha Vạn Cân",
    "Đoàn Văn Bơ",
    "Vĩnh Viễn",
    "Tân Sơn Nhì",
    "Âu Cơ",
    "Lạc Long Quân",
    "Trần Quang Khải",
    "Mai Chí Thọ",
    "Xa lộ Hà Nội",
    "Quốc lộ 1A",
    "QL1A",
    "Quốc lộ 13",
    "Đường số 3",
    "Đường số 7",
    "Đường số 10",
    "Đường số 12",
    "Đường Láng",
    "Khương Trung",
)

_HOLDOUT_STREETS: Tuple[str, ...] = (
    "Hoàng Hoa Thám",
    "Phan Đình Phùng",
    "Lý Tự Trọng",
    "Yên Phụ",
    "Tôn Đức Thắng",
    "Nguyễn Văn Cừ",
    "Nguyễn Đình Chiểu",
    "Phan Chu Trinh",
    "Phan Bội Châu",
    "Nguyễn Thị Thập",
    "Trương Định",
    "Minh Khai",
    "Lê Thanh Nghị",
    "Trần Đại Nghĩa",
    "Nguyễn An Ninh",
    "Trần Quốc Hoàn",
    "Vũ Trọng Phụng",
    "Ngụy Như Kon Tum",
    "Hoàng Đạo Thúy",
    "Lê Trọng Tấn",
    "Nguyễn Ngọc Nại",
    "Võ Nguyên Giáp",
    "Nguyễn Hoàng",
    "Phạm Ngọc Thạch",
    "Đặng Văn Ngữ",
    "Hồ Văn Huê",
    "Hoàng Sa",
    "Trường Sa",
    "Nguyễn Kiệm",
    "Bình Giã",
    "Trần Não",
    "Thảo Điền",
    "Quốc Hương",
    "Nguyễn Gia Trí",
    "Đặng Thai Mai",
)

_TRAIN_BUILDINGS: Tuple[str, ...] = (
    "Tòa HH1C",
    "Chung cư HH1C",
    "Tòa HH2B",
    "Chung cư Ehome 3",
    "Tòa S1.02",
    "Tòa S2.01",
    "Chung cư Goldmark City",
    "Chung cư Vinhomes Grand Park",
    "Tòa R4 Royal City",
    "Chung cư Times City",
    "KĐT Linh Đàm",
    "Tòa CT1A",
    "Chung cư Masteri",
)

_HOLDOUT_BUILDINGS: Tuple[str, ...] = (
    "Tòa Park 6",
    "Chung cư Eco Green",
    "Tòa C1 Mandarin",
    "Chung cư Sunview Town",
    "Tòa S3.05",
    "Chung cư Sala Sarimi",
    "Tòa Landmark 4",
    "Chung cư Rivergate",
)

_TRAIN_RURAL_UNITS: Tuple[str, ...] = (
    "Thôn 1",
    "Thôn 2",
    "Thôn 3",
    "Thôn Đông",
    "Thôn Trung",
    "Ấp 1",
    "Ấp 2",
    "Ấp 4",
    "Ấp Tân Lập",
    "Xóm 3",
    "Xóm 5",
    "Tổ 4",
    "Tổ 7",
    "Khu phố 2",
    "Khu phố 5",
    "Đội 3",
    "Bản Nậm",
    "Cụm 2",
)

_HOLDOUT_RURAL_UNITS: Tuple[str, ...] = (
    "Thôn 6",
    "Thôn Nam",
    "Ấp 3",
    "Ấp Bình Hòa",
    "Xóm 8",
    "Tổ 12",
    "Khu phố 9",
    "Đội 5",
    "Làng Vòng",
)

_TRAIN_PERSON_CUES: Tuple[str, ...] = (
    "Anh",
    "Chị",
    "Em",
    "Cô",
    "Chú",
    "Bác",
    "Ông",
    "Bà",
    "Bạn",
    "Khách",
    "KH",
    "Mr",
    "Ms",
    "Người nhận",
    "Anh/Chị",
)

_HOLDOUT_PERSON_CUES: Tuple[str, ...] = (
    "Anh",
    "Chị",
    "Ông",
    "Bà",
    "Người nhận",
    "Khách",
)

_TRAIN_FIRST_NAMES: Tuple[str, ...] = (
    "Nam",
    "Tuấn",
    "Minh",
    "Hoa",
    "Lan",
    "Hương",
    "Dũng",
    "Hùng",
    "Nga",
    "Linh",
    "Trang",
    "Phúc",
    "Anh",
    "Tâm",
    "Long",
    "Hà",
    "Mai",
    "Phương",
    "Việt",
    "Ngọc",
    "Đức",
    "Hải",
    "Sơn",
    "Vy",
)

_HOLDOUT_FIRST_NAMES: Tuple[str, ...] = (
    "Quang",
    "Thảo",
    "Bảo",
    "Khang",
    "Uyên",
    "Nhung",
    "Khoa",
    "Trinh",
    "Hiếu",
    "Yến",
    "Thịnh",
    "Đạt",
    "Châu",
    "Nhi",
)

_TRAIN_FULL_NAMES: Tuple[str, ...] = (
    "Trần Thị Hoa",
    "Lê Minh",
    "Nguyễn Văn An",
    "Phạm Minh Tuấn",
    "Hoàng Thu Trang",
    "Đặng Quốc Việt",
    "Vũ Thị Mai",
    "Nguyễn Hoàng Long",
    "Lê Thị Ngọc",
    "Trần Đức Anh",
    "Bùi Minh Tâm",
    "Đinh Thị Hương",
    "Phạm Văn Hùng",
    "Ngô Thị Lan",
)

_HOLDOUT_FULL_NAMES: Tuple[str, ...] = (
    "Đỗ Văn Khang",
    "Bùi Thị Thảo",
    "Trịnh Quốc Bảo",
    "Võ Thị Uyên",
    "Phan Thanh Quang",
    "Lý Thị Nhung",
    "Dương Minh Khoa",
    "Tô Ngọc Trinh",
    "Hồ Anh Đạt",
    "Lương Hải Yến",
)

_TRAIN_NOTES: Tuple[str, ...] = (
    "giao giờ hành chính",
    "gọi trước 15p",
    "ship sau 5h",
    "hàng dễ vỡ",
    "gọi trước khi giao",
    "ship nhanh giúp",
    "giao buổi sáng",
    "giao buổi chiều",
    "để ở bảo vệ",
    "gửi lễ tân",
    "không gọi giờ trưa",
    "cho kiểm hàng",
    "giao ngoài giờ hành chính",
    "nhắn tin trước khi qua",
    "giao tận tay",
    "đến cổng gọi mình",
    "giao trước 17h",
    "giao chủ nhật",
    "hàng dễ vỡ xin nhẹ tay",
    "ship buổi tối giúp em",
)

_HOLDOUT_NOTES: Tuple[str, ...] = (
    "để giỏ hàng cổng bảo vệ",
    "gọi số máy phụ giúp mình",
    "giao sau 19h tối",
    "không giao giờ nghỉ trưa",
    "ship trước 11h trưa",
    "gửi cô tạp hóa đầu ngõ",
    "cho đồng kiểm khi nhận",
    "hàng thủy tinh cẩn thận",
    "gọi trước 30 phút",
    "giao cuối tuần giúp shop",
    "nhận hộ ở sảnh B",
    "không gọi điện nhắn zalo",
)

_TRAIN_NEGATIVES: Tuple[str, ...] = (
    "Cho mình 3 chai nướng BBQ",
    "Xin giá",
    "bánh mì Sài Gòn",
    "phở bò Hà Nội 2 tô",
    "bún bò Huế đặc biệt",
    "Shop ơi còn hàng không ạ",
    "2 cái áo thun trắng size L",
    "5 hộp sữa chua nếp cẩm",
    "10kg gạo ST25 bao nhiêu tiền",
    "Giá bao nhiêu vậy shop",
    "cơm tấm Sài Gòn sườn bì chả",
    "trà sữa Đà Lạt size M ít đường",
    "cà phê Buôn Ma Thuột 500g",
    "nem chua Thanh Hóa 2 chục",
    "mắm Phú Quốc chai 500ml",
    "Ok chốt đơn này giúp em nhé",
    "Còn màu đen size XL không shop",
    "Đổi size M qua size L giúp mình",
    "Ib tư vấn kỹ hơn giúp em nha",
    "Lấy 4 gói mì cay cấp độ 3",
    "Hủy đơn cũ đặt lại mẫu mới ạ",
    "Ship đơn 250k nội thành bao lâu",
    "Bánh pía Sóc Trăng hộp 4 cái",
    "Mình lấy 1 combo dầu gội xả",
    "Chốt 2 đôi giày thể thao trắng",
    "bánh tráng Tây Ninh muối tôm",
    "nem nướng Nha Trang 3 phần",
    "chè Thái Nguyên túi 200g",
)

_NEG_VERBS = (
    "Cho mình",
    "Lấy giúp mình",
    "Đặt thêm",
    "Chốt",
    "Gửi cho em",
    "Mua",
    "Tìm mẫu",
    "Hỏi giá",
    "Đổi sang",
    "Tạm giữ",
    "Check giúp mình",
)
_NEG_UNITS = ("cái", "chiếc", "chai", "hộp", "gói", "túi", "kg", "lít", "hũ", "lon", "tô", "phần", "đôi", "bộ", "suất")
_NEG_PRODUCTS = (
    "áo thun cotton",
    "váy hoa",
    "quần jean",
    "giày chạy bộ",
    "dầu gội thảo mộc",
    "sữa rửa mặt",
    "nước mắm nhĩ",
    "mật ong rừng",
    "trà ô long",
    "cà phê rang xay",
    "yến sào chưng",
    "bánh tráng trộn",
    "nem chua",
    "khoai lang sấy",
    "hạt điều rang muối",
    "nước tương",
    "bánh mì chả",
    "phở bò",
    "bún thịt nướng",
    "cơm gà xối mỡ",
    "nước ngọt",
    "kem chống nắng",
    "túi xách da",
    "tai nghe bluetooth",
)
_NEG_PLACES_ATTR = ("Sài Gòn", "Hà Nội", "Huế", "Đà Lạt", "Nha Trang", "Phú Quốc", "Tây Ninh", "Thái Nguyên", "")
_NEG_ATTRS = (
    "size M",
    "size L",
    "size XL",
    "màu trắng",
    "màu đen",
    "màu kem",
    "màu xanh",
    "ít đường",
    "không đá",
    "không hành",
    "loại đặc biệt",
    "500g",
    "1kg",
    "250ml",
    "500ml",
    "hộp lớn",
    "loại 1",
)
_NEG_TAILS = (
    "giá sao",
    "bao nhiêu tiền vậy",
    "còn hàng không",
    "nhé shop",
    "giúp em ạ",
    "có freeship không",
    "không shop ơi",
    "nha",
    "",
)
_NEG_CHAT_QUERIES = (
    "Giá sao ạ",
    "Còn màu này không shop",
    "Check tin nhắn giúp mình nhé",
    "Gửi bảng giá qua tin nhắn giúp em",
    "Đơn hôm qua đi chưa vậy shop",
    "Tạm hoãn đơn sáng nay nhé",
    "Cao 1m65 nặng 55kg mặc size nào vừa",
    "Đặt thêm 2 phần mang về",
    "Mình muốn đổi mẫu khác",
    "Shop tư vấn kỹ hơn giúp mình",
    "Có hàng sẵn không vậy ạ",
    "Chốt đơn cũ giữ nguyên giúp em",
)

_SURNAMES = ("Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương")
_MIDDLES = ("Văn", "Thị", "Minh", "Quốc", "Thanh", "Thu", "Hoàng", "Ngọc", "Anh", "Hải", "Đức", "")
_NOTE_ACTIONS = ("giao", "ship", "gửi", "để", "nhắn", "gọi", "nhận", "cho")
_NOTE_TARGETS = (
    "giờ hành chính",
    "buổi sáng",
    "buổi chiều",
    "buổi tối",
    "trước 17h",
    "sau 18h",
    "trước 12h",
    "cuối tuần",
    "chủ nhật",
    "bảo vệ",
    "lễ tân",
    "cổng chính",
    "sảnh chung cư",
    "trước khi giao",
    "trước 15p",
    "kiểm tra hàng",
    "nhẹ tay hàng dễ vỡ",
    "hàng thủy tinh",
    "không gọi giờ nghỉ",
    "zalo trước khi qua",
)

_HOLDOUT_NEGATIVES: Tuple[str, ...] = (
    "Lấy giúp anh 6 lon nước tăng lực",
    "Hủ tiếu Nam Vang 2 phần không hành",
    "Bánh cuốn Thanh Trì 1kg chả quế",
    "Check tin nhắn inbox giúp mình với",
    "Mẫu váy hoa nhí còn màu be không",
    "Đặt thêm 3 hũ yến chưng đường phèn",
    "Khoai lang mật Đà Lạt 5kg giá sao",
    "Gửi bảng giá sỉ qua zalo cho mình",
    "Bún chả Hà Nội 3 suất mang về",
    "Mì Quảng Đà Nẵng tô lớn thêm trứng",
    "Tạm ngưng đơn sáng nay nhé shop",
    "Áo khoác gió chống nước size XXL",
)

_METRO_PROVINCE_NAMES = {
    "Hồ Chí Minh",
    "Hà Nội",
    "Đà Nẵng",
    "Bình Dương",
    "Đồng Nai",
    "Hải Phòng",
    "Cần Thơ",
}


@dataclass
class SynthExample:
    """One token-aligned synthetic NER sample."""

    text: str
    tokens: List[str]
    tags: List[str]


def load_admin_records(
    path: Path = DEFAULT_RECORDS_PATH,
) -> List[Dict[str, object]]:
    """Load read-only Vietnam administrative division records."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    return list(records)


def _apply_noise_text(
    text: str,
    rng: random.Random,
    allow_unaccented: bool = True,
    allow_typo: bool = False,
) -> str:
    if allow_unaccented:
        text = strip_vietnamese_accents(text)
    if allow_typo and len(text) >= 4:
        words = text.split(" ")
        idx = rng.randrange(len(words))
        w = words[idx]
        if len(w) >= 4 and w.isalpha():
            mode = rng.random()
            if mode < 0.5:
                pos = rng.randint(1, len(w) - 1)
                words[idx] = w[:pos] + w[pos] + w[pos:]
            else:
                pos = rng.randint(1, len(w) - 2)
                words[idx] = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2 :]
            text = " ".join(words)
    return text


def _apply_case_mode(text: str, case_mode: str) -> str:
    if case_mode == "lower":
        return text.lower()
    if case_mode == "upper":
        return text.upper()
    return text


def _gen_house_num(rng: random.Random) -> str:
    style = rng.random()
    n1 = rng.randint(1, 399)
    n2 = rng.randint(1, 45)
    n3 = rng.randint(1, 20)
    letter = rng.choice(["A", "B", "C", "D", "b", "a"])
    if style < 0.25:
        return f"{n1}/{n2}/{n3}"
    if style < 0.45:
        return f"{n1}/{n2}"
    if style < 0.58:
        return f"{n1}/{n2}{letter}"
    if style < 0.68:
        return f"{n1}/{n2}{letter}/{n3}"
    if style < 0.80:
        prefix = rng.choice(["số", "Số", "số nhà", "sn"])
        return f"{prefix} {n1}{letter if rng.random() < 0.25 else ''}"
    if style < 0.88:
        return f"{n1}-{n1 + 2}"
    if style < 0.94:
        return f"K{n1}/{n2}"
    return f"Lô {letter}{n2}"


def _gen_apt_unit(rng: random.Random) -> str:
    floor = rng.randint(2, 28)
    room = rng.randint(1, 16)
    num = f"{floor}{room:02d}"
    style = rng.random()
    if style < 0.50:
        return f"P.{num}"
    if style < 0.72:
        return f"P{num}"
    if style < 0.88:
        return f"Phòng {num}"
    return f"Căn {num}"


def _gen_phone(rng: random.Random, holdout: bool = False) -> str:
    prefixes = ("091", "098", "090", "093", "097", "038", "036", "070", "083", "086")
    pref = rng.choice(prefixes)
    mid = f"{rng.randint(100, 999)}"
    tail = f"{rng.randint(1000, 9999)}"
    d10 = pref + mid + tail
    style = rng.random()
    if style < 0.42:
        return d10
    if style < 0.64:
        return f"{d10[:4]}.{d10[4:7]}.{d10[7:]}"
    if style < 0.78:
        return f"{d10[:4]}-{d10[4:7]}-{d10[7:]}"
    if style < 0.90:
        return f"{d10[:4]} {d10[4:7]} {d10[7:]}"
    if holdout or style < 0.96:
        return f"+84{d10[1:]}"
    return f"028.{rng.randint(3800, 3999)}.{rng.randint(1000, 9999)}"


def _strip_prefix(text: str, prefixes: Tuple[str, ...]) -> str:
    low = text.lower()
    for pfx in prefixes:
        if low.startswith(pfx.lower()):
            return text[len(pfx) :].strip()
    return text


def _format_ward(
    ward_full: str, ward_bare: str, rng: random.Random, unaccented: bool, case_mode: str
) -> str:
    if not ward_full or not ward_bare:
        return ""
    bare = _strip_prefix(ward_bare, ("Phường ", "Xã ", "Thị trấn ", "Đặc khu "))
    low_full = ward_full.lower()
    style = rng.random()
    if style < 0.44:
        out = ward_full
    elif style < 0.74:
        out = ward_bare
    else:
        if low_full.startswith("phường"):
            abbr = rng.choice(["P.", "p.", "P ", "Ph.", "P"])
        elif low_full.startswith("xã"):
            abbr = rng.choice(["X.", "x.", "X ", "x. "])
        elif low_full.startswith("thị trấn"):
            abbr = rng.choice(["TT.", "tt.", "TT ", "T.T "])
        else:
            abbr = "Đặc khu "
        # If bare is numeric ("1", "12"), allow P12 / P.12 without space.
        if bare.isdigit() and abbr in ("P", "P.", "p."):
            out = f"{abbr}{bare}"
        elif abbr == "P":
            out = f"P {bare}"
        else:
            joiner = "" if abbr.endswith(".") and rng.random() < 0.40 else " "
            out = f"{abbr.rstrip()}{joiner}{bare}"

    out = _apply_noise_text(out, rng, allow_unaccented=unaccented, allow_typo=rng.random() < 0.04)
    return _apply_case_mode(out, case_mode)


def _format_district(
    dist_full: str, dist_bare: str, rng: random.Random, unaccented: bool, case_mode: str
) -> str:
    if not dist_full or not dist_bare:
        return ""
    bare = _strip_prefix(dist_bare, ("Quận ", "Huyện ", "Thị xã ", "Thành phố "))
    low_full = dist_full.lower()
    style = rng.random()
    if style < 0.44:
        out = dist_full
    elif style < 0.72:
        out = dist_bare
    else:
        if low_full.startswith("quận"):
            if bare.isdigit():
                out = rng.choice([f"Q{bare}", f"Q.{bare}", f"q.{bare}", f"Q. {bare}"])
            else:
                abbr = rng.choice(["Q.", "q.", "Q "])
                joiner = "" if abbr.endswith(".") and rng.random() < 0.35 else " "
                out = f"{abbr.rstrip()}{joiner}{bare}"
        elif low_full.startswith("huyện"):
            abbr = rng.choice(["H.", "h.", "H "])
            joiner = "" if abbr.endswith(".") and rng.random() < 0.30 else " "
            out = f"{abbr.rstrip()}{joiner}{bare}"
        elif low_full.startswith("thị xã"):
            abbr = rng.choice(["TX.", "tx.", "TX "])
            out = f"{abbr.rstrip()} {bare}"
        else:
            abbr = rng.choice(["TP.", "tp.", "TP "])
            out = f"{abbr.rstrip()} {bare}"

    out = _apply_noise_text(out, rng, allow_unaccented=unaccented, allow_typo=rng.random() < 0.04)
    return _apply_case_mode(out, case_mode)


def _format_province(
    prov_full: str, prov_bare: str, rng: random.Random, unaccented: bool, case_mode: str
) -> str:
    if not prov_full or not prov_bare:
        return ""
    style = rng.random()
    if prov_bare == "Hồ Chí Minh":
        if style < 0.42:
            out = rng.choice(["TP.HCM", "TPHCM", "TP HCM", "HCM", "tp.hcm"])
        elif style < 0.68:
            out = rng.choice(["TP. Hồ Chí Minh", "Hồ Chí Minh", "Thành phố Hồ Chí Minh"])
        else:
            out = rng.choice(["Sài Gòn", "TP Hồ Chí Minh", "SG"])
    elif prov_bare == "Hà Nội":
        if style < 0.55:
            out = "Hà Nội"
        elif style < 0.80:
            out = rng.choice(["TP. Hà Nội", "Thành phố Hà Nội", "TP Hà Nội"])
        else:
            out = rng.choice(["HN", "hn"])
    elif prov_bare == "Đà Nẵng":
        out = rng.choice(["Đà Nẵng", "TP. Đà Nẵng", "Thành phố Đà Nẵng", "TP Đà Nẵng"])
    elif prov_bare == "Bà Rịa - Vũng Tàu":
        out = rng.choice(["Bà Rịa - Vũng Tàu", "Tỉnh Bà Rịa - Vũng Tàu", "BR-VT", "Vũng Tàu"])
    elif style < 0.35:
        out = prov_full
    elif style < 0.82:
        out = prov_bare
    else:
        prefix = "TP." if prov_full.lower().startswith("thành") else rng.choice(["T.", "t."])
        joiner = " " if rng.random() < 0.7 else ""
        out = f"{prefix}{joiner}{prov_bare}"

    out = _apply_noise_text(out, rng, allow_unaccented=unaccented, allow_typo=rng.random() < 0.03)
    return _apply_case_mode(out, case_mode)


def _format_street_part(
    rng: random.Random,
    streets: Sequence[str],
    buildings: Sequence[str],
    rural_units: Sequence[str],
    rural_mode: bool,
    apt_mode: bool,
    unaccented: bool,
    case_mode: str,
) -> Tuple[str, str]:
    """Return (house_num, street) strings cleanly separated without embedded commas."""
    if apt_mode:
        hno = _gen_apt_unit(rng)
        bld = rng.choice(buildings)
        if rng.random() < 0.35:
            st = f"{bld} {rng.choice(streets)}"
        else:
            st = bld
    elif rural_mode:
        hno = _gen_house_num(rng) if rng.random() < 0.55 else ""
        unit = rng.choice(rural_units)
        if rng.random() < 0.45:
            base_s = rng.choice(streets)
            if base_s.lower().startswith(("đường", "phố", "quốc lộ", "ql", "xa lộ")):
                st = f"{unit} {base_s}"
            else:
                st = f"{unit} đường {base_s}"
        else:
            st = unit
    else:
        hno = _gen_house_num(rng) if rng.random() < 0.82 else ""
        base_street = rng.choice(streets)
        has_prefix = base_street.lower().startswith(
            ("đường", "phố", "quốc lộ", "ql", "xa lộ")
        )
        r = rng.random()
        if r < 0.46:
            st = base_street
        elif r < 0.72 and not has_prefix:
            prefix = rng.choice(["đường", "Đường", "Đ.", "phố", "Phố"])
            st = f"{prefix} {base_street}"
        elif r < 0.90:
            alley_word = rng.choice(["ngõ", "hẻm", "kiệt"])
            alley_num = rng.randint(1, 180)
            st = f"{alley_word} {alley_num} {base_street}"
        else:
            st = f"ngõ {rng.randint(1, 90)}/{rng.randint(1, 30)} {base_street}"

    if hno:
        hno = _apply_noise_text(hno, rng, allow_unaccented=unaccented, allow_typo=False)
        hno = _apply_case_mode(hno, case_mode)
    if st:
        st = _apply_noise_text(st, rng, allow_unaccented=unaccented, allow_typo=rng.random() < 0.04)
        st = _apply_case_mode(st, case_mode)
    return hno, st


def _gen_person(
    rng: random.Random,
    cues: Sequence[str],
    first_names: Sequence[str],
    full_names: Sequence[str],
    unaccented: bool,
    case_mode: str,
    holdout: bool = False,
) -> str:
    if not holdout and rng.random() < 0.35:
        mid = rng.choice(_MIDDLES)
        comp_full = " ".join(
            p for p in (rng.choice(_SURNAMES), mid, rng.choice(first_names)) if p
        )
        if rng.random() < 0.35:
            per = f"{rng.choice(cues)} {comp_full}"
        else:
            per = comp_full
    elif rng.random() < 0.58:
        cue = rng.choice(cues)
        name = rng.choice(full_names) if rng.random() < 0.38 else rng.choice(first_names)
        per = f"{cue} {name}"
    else:
        per = rng.choice(full_names)
    per = _apply_noise_text(per, rng, allow_unaccented=unaccented, allow_typo=False)
    return _apply_case_mode(per, case_mode)


def _gen_train_negative(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.16:
        return rng.choice(_TRAIN_NEGATIVES)
    if r < 0.48:
        qty = rng.randint(1, 20)
        parts = [
            rng.choice(_NEG_VERBS) if rng.random() < 0.6 else "",
            str(qty),
            rng.choice(_NEG_UNITS),
            rng.choice(_NEG_PRODUCTS),
            rng.choice(_NEG_ATTRS) if rng.random() < 0.65 else "",
            rng.choice(_NEG_TAILS),
        ]
        return " ".join(p for p in parts if p)
    if r < 0.74:
        place = rng.choice(_NEG_PLACES_ATTR)
        parts = [
            rng.choice(_NEG_PRODUCTS),
            place,
            f"{rng.randint(1, 6)} {rng.choice(_NEG_UNITS)}" if rng.random() < 0.5 else "",
            rng.choice(_NEG_ATTRS) if rng.random() < 0.55 else "",
            rng.choice(_NEG_TAILS),
        ]
        return " ".join(p for p in parts if p)
    if r < 0.90:
        return rng.choice(_NEG_CHAT_QUERIES)
    return f"Đơn {rng.randint(100, 990)}k {rng.choice(_NEG_PRODUCTS)} {rng.choice(_NEG_TAILS)}".strip()


def _gen_train_note(rng: random.Random) -> str:
    if rng.random() < 0.55:
        return rng.choice(_TRAIN_NOTES)
    tail = rng.choice(["nhé", "giúp shop", "giúp em", "ạ", ""])
    note = f"{rng.choice(_NOTE_ACTIONS)} {rng.choice(_NOTE_TARGETS)} {tail}".strip()
    return note


def _assemble_chunks(
    chunks: Sequence[Tuple[str, str, str]],
) -> SynthExample:
    """
    Build token-aligned SynthExample from (text, slot_type, join_sep_before) chunks.
    `slot_type` is `"O"` or one of `"HNO","STR","WAR","DIS","PRO","PER","PHN","NOTE"`.
    """
    pieces: List[str] = []
    tokens: List[str] = []
    tags: List[str] = []

    for idx, (chunk_text, slot_type, sep_before) in enumerate(chunks):
        chunk_tokens = tokenize_words(chunk_text)
        if not chunk_tokens:
            continue
        if idx > 0 and pieces:
            pieces.append(sep_before)
        pieces.append(chunk_text)

        if slot_type == "O":
            for tok in chunk_tokens:
                tokens.append(tok)
                tags.append("O")
        else:
            for t_idx, tok in enumerate(chunk_tokens):
                tokens.append(tok)
                tags.append(f"B-{slot_type}" if t_idx == 0 else f"I-{slot_type}")

    text = "".join(pieces)
    return SynthExample(text=text, tokens=tokens, tags=tags)


def _sample_record(
    rng: random.Random,
    records: Sequence[Dict[str, object]],
    metro_records: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    if metro_records and rng.random() < 0.35:
        return rng.choice(metro_records)
    return rng.choice(records)


def _build_single_example(
    rng: random.Random,
    rec: Dict[str, object],
    holdout: bool,
    force_negative: bool = False,
) -> SynthExample:
    streets = _HOLDOUT_STREETS if holdout else _TRAIN_STREETS
    buildings = _HOLDOUT_BUILDINGS if holdout else _TRAIN_BUILDINGS
    rural_units = _HOLDOUT_RURAL_UNITS if holdout else _TRAIN_RURAL_UNITS
    cues = _HOLDOUT_PERSON_CUES if holdout else _TRAIN_PERSON_CUES
    first_names = _HOLDOUT_FIRST_NAMES if holdout else _TRAIN_FIRST_NAMES
    full_names = _HOLDOUT_FULL_NAMES if holdout else _TRAIN_FULL_NAMES

    if force_negative:
        neg_text = rng.choice(_HOLDOUT_NEGATIVES) if holdout else _gen_train_negative(rng)
        if rng.random() < 0.25:
            neg_text = strip_vietnamese_accents(neg_text)
        if rng.random() < 0.20:
            neg_text = neg_text.lower()
        return _assemble_chunks([(neg_text, "O", "")])

    unaccented = rng.random() < 0.26
    case_roll = rng.random()
    case_mode = "lower" if case_roll < 0.20 else ("upper" if case_roll < 0.25 else "orig")

    # Select address structure: 3-tier urban, 2025 2-tier, rural, or short/partial.
    pattern_roll = rng.random()
    has_ward = bool(rec.get("ward_full")) and bool(rec.get("ward_bare"))

    use_two_tier = pattern_roll < 0.22
    use_short = 0.22 <= pattern_roll < 0.50
    use_rural = 0.50 <= pattern_roll < 0.66
    use_apt = 0.66 <= pattern_roll < 0.75

    hno = ""
    st = ""
    war = ""
    dis = ""
    pro = ""

    if use_two_tier:
        war = _format_ward(
            str(rec.get("new_ward_full") or ""),
            str(rec.get("new_ward_bare") or ""),
            rng,
            unaccented,
            case_mode,
        )
        pro = _format_province(
            str(rec.get("new_province_full") or ""),
            str(rec.get("new_province_bare") or ""),
            rng,
            unaccented,
            case_mode,
        )
        if rng.random() < 0.52:
            hno, st = _format_street_part(
                rng, streets, buildings, rural_units, False, rng.random() < 0.15, unaccented, case_mode
            )
    elif use_short:
        short_kind = rng.random()
        if short_kind < 0.44 and has_ward:
            # 2-part ward + district (like "Lộc Thạnh, Lộc Ninh")
            if not holdout and rng.random() < 0.45:
                war_raw = _apply_noise_text(
                    str(rec["ward_bare"]), rng, allow_unaccented=unaccented, allow_typo=False
                )
                dis_raw = _apply_noise_text(
                    str(rec["district_bare"]), rng, allow_unaccented=unaccented, allow_typo=False
                )
                war = _apply_case_mode(war_raw, case_mode)
                dis = _apply_case_mode(dis_raw, case_mode)
            else:
                war = _format_ward(
                    str(rec["ward_full"]), str(rec["ward_bare"]), rng, unaccented, case_mode
                )
                dis = _format_district(
                    str(rec["district_full"]), str(rec["district_bare"]), rng, unaccented, case_mode
                )
        elif short_kind < 0.62:
            # 2-part district + province
            dis = _format_district(
                str(rec["district_full"]), str(rec["district_bare"]), rng, unaccented, case_mode
            )
            pro = _format_province(
                str(rec["province_full"]), str(rec["province_bare"]), rng, unaccented, case_mode
            )
        elif short_kind < 0.82 and has_ward:
            # Street + ward + district (no province)
            hno, st = _format_street_part(
                rng, streets, buildings, rural_units, False, False, unaccented, case_mode
            )
            war = _format_ward(
                str(rec["ward_full"]), str(rec["ward_bare"]), rng, unaccented, case_mode
            )
            dis = _format_district(
                str(rec["district_full"]), str(rec["district_bare"]), rng, unaccented, case_mode
            )
        else:
            # Street + district (+ optional province)
            hno, st = _format_street_part(
                rng, streets, buildings, rural_units, False, False, unaccented, case_mode
            )
            dis = _format_district(
                str(rec["district_full"]), str(rec["district_bare"]), rng, unaccented, case_mode
            )
            if rng.random() < 0.50:
                pro = _format_province(
                    str(rec["province_full"]), str(rec["province_bare"]), rng, unaccented, case_mode
                )
    else:
        # Full 3-tier urban, rural, or apartment
        hno, st = _format_street_part(
            rng, streets, buildings, rural_units, use_rural, use_apt, unaccented, case_mode
        )
        if has_ward and rng.random() < 0.92:
            war = _format_ward(
                str(rec["ward_full"]), str(rec["ward_bare"]), rng, unaccented, case_mode
            )
        dis = _format_district(
            str(rec["district_full"]), str(rec["district_bare"]), rng, unaccented, case_mode
        )
        if rng.random() < 0.90:
            pro = _format_province(
                str(rec["province_full"]), str(rec["province_bare"]), rng, unaccented, case_mode
            )

    # Assemble address slot chunks with varied comma / dash / no-comma separators.
    no_comma = rng.random() < 0.16
    tight_comma = (not no_comma) and (rng.random() < 0.12)
    if holdout:
        sep = rng.choice([", ", " - ", " ", ", "])
    elif no_comma:
        sep = " "
    elif tight_comma:
        sep = ","
    else:
        sep = ", " if rng.random() < 0.88 else " - "

    addr_chunks: List[Tuple[str, str, str]] = []
    if hno:
        addr_chunks.append((hno, "HNO", " "))
    if st:
        addr_chunks.append((st, "STR", " "))

    for text_val, slot in ((war, "WAR"), (dis, "DIS"), (pro, "PRO")):
        if text_val:
            if addr_chunks:
                if sep in (", ", ","):
                    addr_chunks.append((",", "O", ""))
                    addr_chunks.append((text_val, slot, " " if sep == ", " else ""))
                elif sep == " - ":
                    addr_chunks.append(("-", "O", " "))
                    addr_chunks.append((text_val, slot, " "))
                else:
                    addr_chunks.append((text_val, slot, " "))
            else:
                addr_chunks.append((text_val, slot, ""))

    # Chat/noise slots: person, phone, delivery note, chat lead-ins.
    add_per = rng.random() < (0.55 if holdout else 0.45)
    add_phn = rng.random() < (0.60 if holdout else 0.50)
    add_note = rng.random() < (0.50 if holdout else 0.40)

    # Keep short 2-part bare addresses clean fairly often so parser excels on pure short queries.
    if use_short and rng.random() < 0.60:
        add_per = False
        add_phn = False
        add_note = False

    per_txt = (
        _gen_person(
            rng, cues, first_names, full_names, unaccented, case_mode, holdout=holdout
        )
        if add_per
        else ""
    )
    phn_txt = _gen_phone(rng, holdout=holdout) if add_phn else ""
    note_txt = ""
    if add_note:
        raw_note = rng.choice(_HOLDOUT_NOTES) if holdout else _gen_train_note(rng)
        note_txt = _apply_noise_text(
            raw_note, rng, allow_unaccented=unaccented, allow_typo=False
        )
        note_txt = _apply_case_mode(note_txt, case_mode)

    final_chunks: List[Tuple[str, str, str]] = []

    if rng.random() < 0.14:
        lead = rng.choice(
            [
                "Ship về",
                "Gửi về",
                "Giao tới:",
                "Giao đến",
                "Đơn về:",
                "Gửi hàng tại:",
                "Địa chỉ:",
                "Đ/c:",
                "Nhận tại",
            ]
            if not holdout
            else ["Đơn giao tới:", "Chuyển về", "Nhận hàng tại:"]
        )
        final_chunks.append((lead, "O", ""))

    # Place leading note occasionally.
    note_leading = bool(note_txt) and rng.random() < (0.15 if holdout else 0.22)
    if note_leading:
        final_chunks.append((note_txt, "NOTE", " "))
        final_chunks.append(("-", "O", " "))

    # Decide prefix vs suffix placement for person/phone.
    suffix_contact = rng.random() < (0.28 if holdout else 0.34)
    if not suffix_contact:
        if per_txt and phn_txt:
            if rng.random() < 0.45:
                final_chunks.append((per_txt, "PER", " "))
                final_chunks.append((phn_txt, "PHN", " "))
            else:
                final_chunks.append((per_txt, "PER", " "))
                final_chunks.append(("-", "O", " "))
                final_chunks.append((phn_txt, "PHN", " "))
            final_chunks.append((("-", "O", " ") if rng.random() < 0.75 else (",", "O", "")))
        elif per_txt:
            final_chunks.append((per_txt, "PER", " "))
            final_chunks.append((("-", "O", " ") if rng.random() < 0.7 else (",", "O", "")))
        elif phn_txt:
            if rng.random() < 0.25:
                final_chunks.append(("SĐT:", "O", " "))
            final_chunks.append((phn_txt, "PHN", " "))
            final_chunks.append((("-", "O", " ") if rng.random() < 0.7 else (",", "O", "")))

    for c_idx, (c_txt, c_slot, c_sep) in enumerate(addr_chunks):
        use_sep = " " if (c_idx == 0 and final_chunks) else c_sep
        final_chunks.append((c_txt, c_slot, use_sep))

    if suffix_contact and (per_txt or phn_txt):
        final_chunks.append((("-", "O", " ") if rng.random() < 0.6 else (",", "O", "")))
        if per_txt and phn_txt:
            final_chunks.append((per_txt, "PER", " "))
            if rng.random() < 0.5:
                final_chunks.append(("-", "O", " "))
            final_chunks.append((phn_txt, "PHN", " "))
        elif per_txt:
            final_chunks.append((per_txt, "PER", " "))
        elif phn_txt:
            final_chunks.append((phn_txt, "PHN", " "))

    if note_txt and not note_leading:
        wrap_style = rng.random()
        if wrap_style < 0.48:
            final_chunks.append(("(", "O", " "))
            final_chunks.append((note_txt, "NOTE", ""))
            final_chunks.append((")", "O", ""))
        elif wrap_style < 0.78:
            final_chunks.append(("-", "O", " "))
            final_chunks.append((note_txt, "NOTE", " "))
        elif wrap_style < 0.92:
            final_chunks.append((",", "O", ""))
            final_chunks.append((note_txt, "NOTE", " "))
        else:
            final_chunks.append(("[", "O", " "))
            final_chunks.append((note_txt, "NOTE", ""))
            final_chunks.append(("]", "O", ""))

    if rng.random() < 0.08:
        tail_chat = rng.choice(["nhé shop", "giúp mình ạ", "cảm ơn shop"])
        final_chunks.append((tail_chat, "O", " "))

    return _assemble_chunks(final_chunks)


def generate_dataset(
    n_samples: int = 50000,
    seed: int = 401,
    train_size: int = 42000,
    val_size: int = 4000,
    test_size: int = 4000,
    records_path: Path = DEFAULT_RECORDS_PATH,
) -> Tuple[List[SynthExample], List[SynthExample], List[SynthExample]]:
    """Generate 50k token-aligned synthetic Vietnamese samples split 42k/4k/4k."""
    if train_size + val_size + test_size != n_samples:
        raise ValueError("Split sizes must sum to n_samples")

    records = load_admin_records(records_path)
    metro_records = [
        r for r in records if str(r.get("province_bare", "")) in _METRO_PROVINCE_NAMES
    ]
    rng = random.Random(seed)

    # Guarantee full coverage over all 10,602 admin records across multiple shuffled passes.
    record_pool: List[Dict[str, object]] = []
    while len(record_pool) < n_samples:
        shuffled = list(records)
        rng.shuffle(shuffled)
        record_pool.extend(shuffled)

    examples: List[SynthExample] = []
    for idx in range(n_samples):
        is_neg = rng.random() < 0.15
        rec = record_pool[idx] if idx < len(records) * 3 else _sample_record(rng, records, metro_records)
        ex = _build_single_example(rng, rec, holdout=False, force_negative=is_neg)
        # Keep token lengths comfortably within max_word_len=64.
        if 1 <= len(ex.tokens) <= 60:
            examples.append(ex)
        else:
            examples.append(_build_single_example(rng, rec, holdout=False, force_negative=False))

    rng.shuffle(examples)
    train = examples[:train_size]
    val = examples[train_size : train_size + val_size]
    test = examples[train_size + val_size :]
    return train, val, test


def generate_holdout(
    n_samples: int = 500,
    seed: int = 20261201,
    records_path: Path = DEFAULT_RECORDS_PATH,
) -> List[SynthExample]:
    """Generate independent 500-case holdout using distinct phrasing and vocab pools."""
    records = load_admin_records(records_path)
    metro_records = [
        r for r in records if str(r.get("province_bare", "")) in _METRO_PROVINCE_NAMES
    ]
    rng = random.Random(seed)
    examples: List[SynthExample] = []
    for _ in range(n_samples):
        is_neg = rng.random() < 0.15
        rec = _sample_record(rng, records, metro_records)
        ex = _build_single_example(rng, rec, holdout=True, force_negative=is_neg)
        examples.append(ex)
    return examples


def verify_bio_tags(tags: Sequence[str]) -> bool:
    """Validate that a tag sequence strictly obeys BIO transitions."""
    prev_slot = ""
    for tag in tags:
        if tag not in TAG2ID:
            return False
        if tag == "O":
            prev_slot = ""
            continue
        prefix, slot = tag.split("-", 1)
        if prefix == "I" and prev_slot != slot:
            return False
        prev_slot = slot
    return True
