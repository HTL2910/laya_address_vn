"""Rich synthetic + curated Vietnamese address dataset generator for Laya Noul."""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Vietnamese diacritic stripping (handles đ/Đ cleanly)
# ---------------------------------------------------------------------------

def strip_diacritics(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")


# ---------------------------------------------------------------------------
# Curated 24-case Vietnamese edge-case evaluation suite (never in train/val)
# ---------------------------------------------------------------------------

CURATED_EDGE_CASES: List[Tuple[str, int, str]] = [
    # --- 12 Positives ---
    ("Số 123 Đường Nguyễn Trãi, Phường Thượng Đình, Quận Thanh Xuân, Hà Nội", 1, "formal_hn"),
    ("45/12B Cách Mạng Tháng 8, Phường 13, Quận 10, TP. Hồ Chí Minh", 1, "formal_hcm"),
    ("123 nguyen trai thanh xuan ha noi", 1, "unaccented_full"),
    ("P.12 Q.TB HCM", 1, "abbrev_ward_dist_city"),
    ("sn 45 ngo 102 truong chinh hn", 1, "abbrev_alley_hn"),
    ("15/2B CMT8 P13 Q10", 1, "abbrev_hcm_cmt8"),
    ("45 Lê Lợi, Q1, HCM", 1, "short_formal"),
    ("ngõ 68 cầu giấy", 1, "short_alley"),
    ("giao anh hung 25 tran hung dao hoan kiem", 1, "delivery_embedded"),
    ("Căn hộ S2.05-1208 Chung cư Vinhomes Ocean Park, Gia Lâm, Hà Nội", 1, "apartment"),
    ("Lô B2-3 KCN VSIP 1, TP. Thuận An, Bình Dương", 1, "industrial_zone"),
    ("Ấp 3, Xã Bình Hưng, Huyện Bình Chánh, TP.HCM", 1, "rural_commune"),
    # --- 12 Hard Negatives ---
    ("Nguyễn Văn Hùng", 0, "person_name"),
    ("Trần Thị Mai Linh", 0, "person_name_female"),
    ("0912345678", 0, "phone_plain"),
    ("+84 987 654 321", 0, "phone_intl"),
    ("hung.nguyen@gmail.com", 0, "email"),
    ("https://shopee.vn/ao-thun-nam-123", 0, "url"),
    ("15/08/2026", 0, "date"),
    ("250.000đ", 0, "price"),
    ("Áo thun nam size XL màu đen", 0, "product_sku"),
    ("Hà Nội hôm nay mưa to quá", 0, "chat_place_hn"),
    ("Quận 1 kẹt xe kinh khủng", 0, "chat_place_q1"),
    ("Em đang đi du lịch Đà Nẵng", 0, "chat_place_dn"),
    ("Anh Hùng sinh năm 1990 ở quê", 0, "chat_bio"),
    ("Gọi cho anh Tuấn lúc 5h chiều", 0, "chat_call"),
    ("iPhone 16 Pro Max 256GB", 0, "product_phone"),
    ("abc123", 0, "random_short"),
]


# ---------------------------------------------------------------------------
# Authentic Vietnamese Geography Knowledge Base (63 Provinces + Districts)
# ---------------------------------------------------------------------------

PROVINCES_63: List[Tuple[str, List[str]]] = [
    ("Hà Nội", ["TP. Hà Nội", "Hà Nội", "HN", "Tp Hà Nội"]),
    ("Hồ Chí Minh", ["TP. Hồ Chí Minh", "TP.HCM", "TPHCM", "HCM", "Sài Gòn", "SG", "TP Hồ Chí Minh"]),
    ("Đà Nẵng", ["TP. Đà Nẵng", "Đà Nẵng", "ĐN", "DN"]),
    ("Hải Phòng", ["TP. Hải Phòng", "Hải Phòng", "HP"]),
    ("Cần Thơ", ["TP. Cần Thơ", "Cần Thơ", "CT"]),
    ("An Giang", ["An Giang", "Tỉnh An Giang"]),
    ("Bà Rịa - Vũng Tàu", ["Bà Rịa - Vũng Tàu", "BR-VT", "Vũng Tàu"]),
    ("Bắc Giang", ["Bắc Giang", "Tỉnh Bắc Giang"]),
    ("Bắc Kạn", ["Bắc Kạn"]),
    ("Bạc Liêu", ["Bạc Liêu"]),
    ("Bắc Ninh", ["Bắc Ninh", "Tỉnh Bắc Ninh"]),
    ("Bến Tre", ["Bến Tre"]),
    ("Bình Định", ["Bình Định"]),
    ("Bình Dương", ["Bình Dương", "Tỉnh Bình Dương", "BD"]),
    ("Bình Phước", ["Bình Phước"]),
    ("Bình Thuận", ["Bình Thuận"]),
    ("Cà Mau", ["Cà Mau"]),
    ("Cao Bằng", ["Cao Bằng"]),
    ("Đắk Lắk", ["Đắk Lắk", "Dak Lak"]),
    ("Đắk Nông", ["Đắk Nông"]),
    ("Điện Biên", ["Điện Biên"]),
    ("Đồng Nai", ["Đồng Nai", "Tỉnh Đồng Nai", "ĐNai"]),
    ("Đồng Tháp", ["Đồng Tháp"]),
    ("Gia Lai", ["Gia Lai"]),
    ("Hà Giang", ["Hà Giang"]),
    ("Hà Nam", ["Hà Nam"]),
    ("Hà Tĩnh", ["Hà Tĩnh"]),
    ("Hải Dương", ["Hải Dương"]),
    ("Hậu Giang", ["Hậu Giang"]),
    ("Hòa Bình", ["Hòa Bình"]),
    ("Hưng Yên", ["Hưng Yên"]),
    ("Khánh Hòa", ["Khánh Hòa"]),
    ("Kiên Giang", ["Kiên Giang"]),
    ("Kon Tum", ["Kon Tum"]),
    ("Lai Châu", ["Lai Châu"]),
    ("Lâm Đồng", ["Lâm Đồng"]),
    ("Lạng Sơn", ["Lạng Sơn"]),
    ("Lào Cai", ["Lào Cai"]),
    ("Long An", ["Long An"]),
    ("Nam Định", ["Nam Định"]),
    ("Nghệ An", ["Nghệ An"]),
    ("Ninh Bình", ["Ninh Bình"]),
    ("Ninh Thuận", ["Ninh Thuận"]),
    ("Phú Thọ", ["Phú Thọ"]),
    ("Phú Yên", ["Phú Yên"]),
    ("Quảng Bình", ["Quảng Bình"]),
    ("Quảng Nam", ["Quảng Nam"]),
    ("Quảng Ngãi", ["Quảng Ngãi"]),
    ("Quảng Ninh", ["Quảng Ninh"]),
    ("Quảng Trị", ["Quảng Trị"]),
    ("Sóc Trăng", ["Sóc Trăng"]),
    ("Sơn La", ["Sơn La"]),
    ("Tây Ninh", ["Tây Ninh"]),
    ("Thái Bình", ["Thái Bình"]),
    ("Thái Nguyên", ["Thái Nguyên"]),
    ("Thanh Hóa", ["Thanh Hóa"]),
    ("Thừa Thiên Huế", ["Thừa Thiên Huế", "TP. Huế", "Huế"]),
    ("Tiền Giang", ["Tiền Giang"]),
    ("Trà Vinh", ["Trà Vinh"]),
    ("Tuyên Quang", ["Tuyên Quang"]),
    ("Vĩnh Long", ["Vĩnh Long"]),
    ("Vĩnh Phúc", ["Vĩnh Phúc"]),
    ("Yên Bái", ["Yên Bái"]),
]

HN_DISTRICTS: List[Tuple[str, List[str]]] = [
    ("Hoàn Kiếm", ["Quận Hoàn Kiếm", "Q. Hoàn Kiếm", "Hoàn Kiếm", "Q.HK"]),
    ("Ba Đình", ["Quận Ba Đình", "Q. Ba Đình", "Ba Đình"]),
    ("Đống Đa", ["Quận Đống Đa", "Q. Đống Đa", "Đống Đa", "Q.ĐĐ"]),
    ("Hai Bà Trưng", ["Quận Hai Bà Trưng", "Q. Hai Bà Trưng", "Hai Bà Trưng", "HBT"]),
    ("Cầu Giấy", ["Quận Cầu Giấy", "Q. Cầu Giấy", "Cầu Giấy", "Q.CG"]),
    ("Thanh Xuân", ["Quận Thanh Xuân", "Q. Thanh Xuân", "Thanh Xuân", "Q.TX"]),
    ("Hoàng Mai", ["Quận Hoàng Mai", "Q. Hoàng Mai", "Hoàng Mai", "Q.HM"]),
    ("Long Biên", ["Quận Long Biên", "Q. Long Biên", "Long Biên", "Q.LB"]),
    ("Tây Hồ", ["Quận Tây Hồ", "Q. Tây Hồ", "Tây Hồ"]),
    ("Nam Từ Liêm", ["Quận Nam Từ Liêm", "Q. Nam Từ Liêm", "Nam Từ Liêm", "NTL"]),
    ("Bắc Từ Liêm", ["Quận Bắc Từ Liêm", "Q. Bắc Từ Liêm", "Bắc Từ Liêm", "BTL"]),
    ("Hà Đông", ["Quận Hà Đông", "Q. Hà Đông", "Hà Đông"]),
    ("Đông Anh", ["Huyện Đông Anh", "H. Đông Anh", "Đông Anh"]),
    ("Gia Lâm", ["Huyện Gia Lâm", "H. Gia Lâm", "Gia Lâm"]),
    ("Thanh Trì", ["Huyện Thanh Trì", "H. Thanh Trì", "Thanh Trì"]),
    ("Hoài Đức", ["Huyện Hoài Đức", "Hoài Đức"]),
    ("Thạch Thất", ["Huyện Thạch Thất", "Thạch Thất"]),
    ("Sóc Sơn", ["Huyện Sóc Sơn", "Sóc Sơn"]),
    ("Sơn Tây", ["Thị xã Sơn Tây", "TX. Sơn Tây", "Sơn Tây"]),
]

HCM_DISTRICTS: List[Tuple[str, List[str]]] = [
    ("Quận 1", ["Quận 1", "Q.1", "Q1", "Q 1"]),
    ("Quận 3", ["Quận 3", "Q.3", "Q3", "Q 3"]),
    ("Quận 4", ["Quận 4", "Q.4", "Q4"]),
    ("Quận 5", ["Quận 5", "Q.5", "Q5"]),
    ("Quận 6", ["Quận 6", "Q.6", "Q6"]),
    ("Quận 7", ["Quận 7", "Q.7", "Q7"]),
    ("Quận 8", ["Quận 8", "Q.8", "Q8"]),
    ("Quận 10", ["Quận 10", "Q.10", "Q10"]),
    ("Quận 11", ["Quận 11", "Q.11", "Q11"]),
    ("Quận 12", ["Quận 12", "Q.12", "Q12"]),
    ("Bình Thạnh", ["Quận Bình Thạnh", "Q. Bình Thạnh", "Bình Thạnh", "Q.BT", "QBT"]),
    ("Tân Bình", ["Quận Tân Bình", "Q. Tân Bình", "Tân Bình", "Q.TB", "QTB"]),
    ("Tân Phú", ["Quận Tân Phú", "Q. Tân Phú", "Tân Phú", "Q.TP"]),
    ("Phú Nhuận", ["Quận Phú Nhuận", "Q. Phú Nhuận", "Phú Nhuận", "Q.PN", "QPN"]),
    ("Gò Vấp", ["Quận Gò Vấp", "Q. Gò Vấp", "Gò Vấp", "Q.GV", "QGV"]),
    ("Bình Tân", ["Quận Bình Tân", "Q. Bình Tân", "Bình Tân", "Q.BTân"]),
    ("Thủ Đức", ["TP. Thủ Đức", "Thành phố Thủ Đức", "Thủ Đức", "Q. Thủ Đức", "Q.2", "Q.9"]),
    ("Bình Chánh", ["Huyện Bình Chánh", "H. Bình Chánh", "Bình Chánh"]),
    ("Hóc Môn", ["Huyện Hóc Môn", "H. Hóc Môn", "Hóc Môn"]),
    ("Nhà Bè", ["Huyện Nhà Bè", "H. Nhà Bè", "Nhà Bè"]),
    ("Củ Chi", ["Huyện Củ Chi", "H. Củ Chi", "Củ Chi"]),
]

OTHER_DISTRICTS: List[Tuple[str, str]] = [
    ("Quận Hải Châu", "Đà Nẵng"), ("Quận Thanh Khê", "Đà Nẵng"), ("Quận Sơn Trà", "Đà Nẵng"),
    ("Quận Ngũ Hành Sơn", "Đà Nẵng"), ("Quận Liên Chiểu", "Đà Nẵng"), ("Quận Cẩm Lệ", "Đà Nẵng"),
    ("Quận Ngô Quyền", "Hải Phòng"), ("Quận Lê Chân", "Hải Phòng"), ("Quận Hồng Bàng", "Hải Phòng"),
    ("Quận Hải An", "Hải Phòng"), ("Huyện Thủy Nguyên", "Hải Phòng"),
    ("Quận Ninh Kiều", "Cần Thơ"), ("Quận Cái Răng", "Cần Thơ"), ("Quận Bình Thủy", "Cần Thơ"),
    ("TP. Biên Hòa", "Đồng Nai"), ("Huyện Long Thành", "Đồng Nai"), ("Huyện Nhơn Trạch", "Đồng Nai"),
    ("TP. Thủ Dầu Một", "Bình Dương"), ("TP. Dĩ An", "Bình Dương"), ("TP. Thuận An", "Bình Dương"),
    ("TP. Bến Cát", "Bình Dương"), ("TP. Tân Uyên", "Bình Dương"),
    ("TP. Nha Trang", "Khánh Hòa"), ("TP. Cam Ranh", "Khánh Hòa"),
    ("TP. Đà Lạt", "Lâm Đồng"), ("TP. Bảo Lộc", "Lâm Đồng"),
    ("TP. Hạ Long", "Quảng Ninh"), ("TP. Cẩm Phả", "Quảng Ninh"), ("TP. Móng Cái", "Quảng Ninh"),
    ("TP. Vinh", "Nghệ An"), ("TX. Cửa Lò", "Nghệ An"),
    ("TP. Vũng Tàu", "Bà Rịa - Vũng Tàu"), ("TP. Bà Rịa", "Bà Rịa - Vũng Tàu"),
    ("TP. Bắc Ninh", "Bắc Ninh"), ("TP. Từ Sơn", "Bắc Ninh"), ("Huyện Yên Phong", "Bắc Ninh"),
    ("TP. Hải Dương", "Hải Dương"), ("TP. Chí Linh", "Hải Dương"),
    ("TP. Thanh Hóa", "Thanh Hóa"), ("TP. Sầm Sơn", "Thanh Hóa"),
    ("TP. Quy Nhơn", "Bình Định"), ("TP. Buôn Ma Thuột", "Đắk Lắk"),
    ("TP. Phan Thiết", "Bình Thuận"), ("TP. Rạch Giá", "Kiên Giang"), ("TP. Phú Quốc", "Kiên Giang"),
    ("TP. Mỹ Tho", "Tiền Giang"), ("TP. Tân An", "Long An"), ("Huyện Bến Lức", "Long An"),
    ("Huyện Đức Hòa", "Long An"), ("TP. Long Xuyên", "An Giang"), ("TP. Châu Đốc", "An Giang"),
    ("TP. Thái Nguyên", "Thái Nguyên"), ("TP. Việt Trì", "Phú Thọ"),
    ("TP. Nam Định", "Nam Định"), ("TP. Thái Bình", "Thái Bình"),
    ("TP. Lào Cai", "Lào Cai"), ("TX. Sa Pa", "Lào Cai"), ("Huyện Văn Giang", "Hưng Yên"),
    ("TP. Bắc Giang", "Bắc Giang"), ("TP. Ninh Bình", "Ninh Bình"),
    ("TP. Huế", "Thừa Thiên Huế"), ("TX. Hương Thủy", "Thừa Thiên Huế"),
    ("TP. Pleiku", "Gia Lai"), ("TP. Tuy Hòa", "Phú Yên"),
    ("TP. Tam Kỳ", "Quảng Nam"), ("TP. Hội An", "Quảng Nam"),
    ("TP. Quảng Ngãi", "Quảng Ngãi"), ("TP. Đồng Hới", "Quảng Bình"),
    ("TP. Đông Hà", "Quảng Trị"), ("TP. Hà Tĩnh", "Hà Tĩnh"),
    ("TP. Vĩnh Yên", "Vĩnh Phúc"), ("TP. Phúc Yên", "Vĩnh Phúc"),
    ("TP. Phủ Lý", "Hà Nam"), ("TP. Cao Lãnh", "Đồng Tháp"),
    ("TP. Bến Tre", "Bến Tre"), ("TP. Vĩnh Long", "Vĩnh Long"),
    ("TP. Sóc Trăng", "Sóc Trăng"), ("TP. Bạc Liêu", "Bạc Liêu"),
    ("TP. Cà Mau", "Cà Mau"), ("TP. Tây Ninh", "Tây Ninh"),
    ("TP. Đồng Xoài", "Bình Phước"), ("TP. Sơn La", "Sơn La"),
    ("TP. Điện Biên Phủ", "Điện Biên"), ("TP. Hòa Bình", "Hòa Bình"),
    ("TP. Lạng Sơn", "Lạng Sơn"), ("TP. Cao Bằng", "Cao Bằng"),
    ("TP. Hà Giang", "Hà Giang"), ("TP. Tuyên Quang", "Tuyên Quang"),
    ("TP. Yên Bái", "Yên Bái"), ("TP. Kon Tum", "Kon Tum"),
    ("TP. Gia Nghĩa", "Đắk Nông"), ("TP. Phan Rang - Tháp Chàm", "Ninh Thuận"),
    ("TP. Trà Vinh", "Trà Vinh"), ("TP. Vị Thanh", "Hậu Giang"),
    ("TP. Bắc Kạn", "Bắc Kạn"), ("TP. Lai Châu", "Lai Châu"),
]

HN_WARDS: List[str] = [
    "Hàng Bạc", "Hàng Đào", "Tràng Tiền", "Cửa Nam", "Kim Mã", "Giảng Võ",
    "Ngọc Khánh", "Láng Hạ", "Trung Liệt", "Ô Chợ Dừa", "Bạch Mai", "Minh Khai",
    "Dịch Vọng", "Dịch Vọng Hậu", "Mai Dịch", "Nghĩa Tân", "Nhân Chính",
    "Thanh Xuân Trung", "Thượng Đình", "Khương Đình", "Khương Mai", "Định Công",
    "Hoàng Liệt", "Đại Kim", "Mỹ Đình 1", "Mỹ Đình 2", "Mễ Trì", "Phú Đô",
    "Cổ Nhuế 1", "Văn Quán", "Mộ Lao", "Quang Trung", "Trung Hòa", "Yên Hòa",
    "Láng Thượng", "Cống Vị", "Bách Khoa", "Vĩnh Tuy", "Phương Mai", "Phương Liệt",
    "Xuân La", "Quảng An", "Bồ Đề", "Ngọc Lâm", "Gia Thụy", "Tân Mai",
]

HCM_NAMED_WARDS: List[str] = [
    "Bến Nghé", "Bến Thành", "Đa Kao", "Tân Định", "Võ Thị Sáu", "Thảo Điền",
    "An Phú", "Hiệp Bình Chánh", "Linh Trung", "Linh Chiểu", "Tân Phong",
    "Tân Phú", "Phú Mỹ", "Hiệp Phú", "Tăng Nhơn Phú A", "Phước Long B",
    "Bình Hưng Hòa", "Bình Trị Đông", "Tây Thạnh", "Sơn Kỳ", "Tân Quý",
    "An Lạc", "Tân Tạo", "Hiệp Thành", "Thạnh Xuân", "Tân Chánh Hiệp",
    "Bình An", "An Khánh", "Linh Đông", "Trường Thọ", "Cát Lái",
]

OTHER_WARDS: List[str] = [
    "Thạch Thang", "Hải Châu 1", "Phước Mỹ", "An Hải Bắc", "Mỹ An", "Hòa Khánh Bắc",
    "Xuân Khánh", "An Hòa", "Cái Khế", "Hưng Lợi", "Tân Hiệp", "Trảng Dài",
    "Tam Hiệp", "Long Bình", "Phú Hòa", "Đông Hòa", "An Bình", "Lái Thiêu",
    "Vĩnh Hải", "Lộc Thọ", "Tân Lập", "Hồng Hải", "Bãi Cháy", "Hưng Dũng",
    "Trường Thi", "Phú Nhuận", "Vĩnh Ninh", "Thắng Tam", "Rạch Dừa", "Suối Hoa",
    "Võ Cường", "Bình Hàn", "Quang Trung", "Lê Lợi", "Nguyễn Văn Cừ", "Tân An",
]

RURAL_COMMUNES: List[str] = [
    "Xã Hòa Lạc", "Xã Tân Lập", "Xã Đông Dư", "Xã Kim Chung", "Xã Đa Tốn",
    "Xã Bình Hưng", "Xã Vĩnh Lộc A", "Xã Vĩnh Lộc B", "Xã Bà Điểm", "Xã Phước Kiển",
    "Xã An Khánh", "Xã Tân Triều", "Xã Nghi Phú", "Xã Xuân Sơn", "Xã Mỹ Hạnh Nam",
    "Xã Tam Phước", "Xã An Phước", "Xã Long Hưng", "Xã Tân Thông Hội", "Xã Trung Chánh",
    "Xã Nhơn Đức", "Xã Tân Kiên", "Xã Đa Phước", "Xã Phú Xuân", "Xã Thới Tam Thôn",
    "Thị trấn Đông Anh", "Thị trấn Trâu Quỳ", "Thị trấn Văn Điển", "Thị trấn Long Thành",
    "Thị trấn Bến Lức", "Thị trấn Đức Hòa", "Thị trấn Củ Chi", "Thị trấn Nhà Bè",
    "Thị trấn Dương Đông", "Thị trấn An Thới", "TT. Chờ", "TT. Lim", "TT. Phùng",
]

STREETS: List[str] = [
    "Nguyễn Trãi", "Lê Lợi", "Trần Hưng Đạo", "Hai Bà Trưng", "Lý Thường Kiệt",
    "Nguyễn Huệ", "Điện Biên Phủ", "Cách Mạng Tháng 8", "CMT8", "Võ Văn Tần",
    "Nguyễn Thị Minh Khai", "Lê Duẩn", "Pasteur", "Nam Kỳ Khởi Nghĩa", "Cộng Hòa",
    "Trường Chinh", "Hoàng Văn Thụ", "Phan Xích Long", "Phan Đăng Lưu",
    "Xô Viết Nghệ Tĩnh", "XVNT", "Ung Văn Khiêm", "Kha Vạn Cân", "Phạm Văn Đồng",
    "Võ Văn Ngân", "Nguyễn Văn Linh", "Huỳnh Tấn Phát", "Nguyễn Hữu Thọ",
    "Lạc Long Quân", "Âu Cơ", "Ba Tháng Hai", "3/2", "Lý Tự Trọng", "Bùi Viện",
    "Đề Thám", "Cầu Giấy", "Xuân Thủy", "Hồ Tùng Mậu", "Phạm Hùng", "Trần Duy Hưng",
    "Nguyễn Chí Thanh", "Láng", "Láng Hạ", "Thái Hà", "Chùa Bộc", "Tôn Đức Thắng",
    "Tây Sơn", "Giải Phóng", "Minh Khai", "Đại La", "Bạch Mai", "Phố Huế",
    "Hàng Bài", "Đinh Tiên Hoàng", "Hoàng Diệu", "Kim Mã", "Liễu Giai", "Đội Cấn",
    "Thụy Khuê", "Hoàng Quốc Việt", "Nguyễn Văn Cừ", "Ngô Gia Tự", "Lê Văn Lương",
    "Tố Hữu", "Khuất Duy Tiến", "Nguyễn Xiển", "Tam Trinh", "Lĩnh Nam", "Trương Định",
    "Kim Ngưu", "Bà Triệu", "Quang Trung", "Hùng Vương", "Bạch Đằng", "Trần Phú",
    "Võ Nguyên Giáp", "Nguyễn Văn Thoại", "Tôn Thất Tùng", "Phan Chu Trinh",
    "Phan Bội Châu", "Đinh Bộ Lĩnh", "Nguyễn Oanh", "Lê Đức Thọ", "Nguyễn Kiệm",
    "Hoàng Hoa Thám", "Lê Văn Sỹ", "Phan Văn Trị", "Quốc lộ 1A", "QL1A", "QL13",
    "QL22", "QL51", "Tỉnh lộ 10", "TL10", "ĐT743", "Đại lộ Thăng Long", "Xa lộ Hà Nội",
    "Võ Chí Công", "Mai Chí Thọ", "Nguyễn Duy Trinh", "Đỗ Xuân Hợp", "Lê Văn Việt",
    "Lê Trọng Tấn", "Ngọc Hà", "Đào Tấn", "Phạm Văn Bạch", "Trung Kính", "Vũ Phạm Hàm",
    "Nguyễn Khang", "Trần Thái Tông", "Duy Tân", "Hàm Nghi", "Nguyễn Cơ Thạch",
    "Tô Hiến Thành", "Sư Vạn Hạnh", "Hồng Bàng", "An Dương Vương", "Trần Quang Khải",
    "Nguyễn Đình Chiểu", "Võ Thị Sáu", "Trần Quốc Thảo", "Kinh Dương Vương",
]

APARTMENTS_BUILDINGS: List[str] = [
    "Chung cư Vinhomes Central Park", "Vinhomes Ocean Park", "Vinhomes Smart City",
    "Times City", "Royal City", "Chung cư Masteri Thảo Điền", "Landmark 81",
    "Tòa nhà Keangnam", "Lotte Center", "Tòa nhà Bitexco", "Chung cư Sunrise City",
    "KĐT Sala Đại Quang Minh", "Khu đô thị Ecopark", "Chung cư Goldmark City",
    "Chung cư HH Linh Đàm", "Chung cư Eco Green", "Imperia Garden", "The Manor",
    "Golden Palace", "Estella Heights", "Saigon Pearl", "Chung cư Carillon",
    "Chung cư Topaz Home", "Chung cư 4S Linh Đông", "Tòa nhà Viettel",
    "Tòa nhà FPT", "Tòa nhà Etown", "Tòa nhà Vincom", "Tòa nhà Charmvit",
    "Chung cư Mizuki Park", "Chung cư Akari City", "Chung cư Phú Mỹ Hưng",
    "Chung cư Gateway Thảo Điền", "Chung cư The Sun Avenue", "Tòa nhà Saigon Centre",
]

INDUSTRIAL_ZONES: List[str] = [
    "KCN Tân Bình", "KCN VSIP 1", "KCN VSIP 2", "KCN Amata", "KCN Biên Hòa 2",
    "KCN Bắc Thăng Long", "KCN Quang Minh", "KCN Quế Võ", "KCN Yên Phong",
    "KCN Sóng Thần 1", "KCN Tân Tạo", "KCN Hiệp Phước", "KCX Tân Thuận",
    "Khu công nghiệp Mỹ Phước 3", "Khu công nghiệp Nhơn Trạch", "Khu công nghệ cao",
    "Cụm CN Lai Xá", "KCN Long Hậu", "KCN Tràng Duệ", "KCN Hòa Khánh",
]

LANDMARKS: List[str] = [
    "Ngã tư Sở", "Ngã tư Hàng Xanh", "Ngã tư Thủ Đức", "Chợ Bến Thành",
    "Chợ Đồng Xuân", "Chợ Ninh Hiệp", "Bến xe Mỹ Đình", "Bến xe Miền Đông",
    "Khu đô thị Phú Mỹ Hưng", "Khu dân cư Trung Sơn", "KĐT Linh Đàm",
    "Khu đô thị Ciputra", "KĐT Văn Phú", "Khu tái định cư Bình Khánh",
]

# ---------------------------------------------------------------------------
# Hard Negative Pools
# ---------------------------------------------------------------------------

SURNAMES = [
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ",
    "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý", "Đinh", "Đào", "Trương",
]
MID_NAMES_M = ["Văn", "Minh", "Hoàng", "Quang", "Đức", "Thanh", "Gia", "Tuấn", "Hữu", "Xuân", "Công", "Quốc"]
MID_NAMES_F = ["Thị", "Ngọc", "Thị Mai", "Thị Thu", "Khánh", "Phương", "Thanh", "Minh", "Thùy", "Hồng", "Kiều"]
GIVEN_M = ["Hùng", "Tuấn", "Dũng", "Minh", "Nam", "Long", "Khang", "Bảo", "Huy", "Đạt", "Quân", "Phong", "Tài", "Sơn", "Thành", "Hiếu", "Kiên", "Thắng", "Việt", "An"]
GIVEN_F = ["Linh", "Hương", "Lan", "Trang", "Thảo", "Anh", "Ngân", "Vy", "Hà", "Yến", "Nhi", "Quỳnh", "Phương", "Hạnh", "Mai", "Châu", "Trinh", "My", "Diệp", "Hoa"]

PERSON_HISTORICAL_TRAPS = [
    "Nguyễn Trãi", "Lê Lợi", "Trần Hưng Đạo", "Hai Bà Trưng", "Lý Thường Kiệt",
    "Nguyễn Huệ", "Quang Trung", "Đinh Tiên Hoàng", "Phan Bội Châu", "Phan Chu Trinh",
    "Nguyễn Thị Minh Khai", "Tôn Đức Thắng", "Võ Nguyên Giáp", "Lê Duẩn", "Hoàng Diệu",
    "Ngô Quyền", "Lý Tự Trọng", "Nguyễn Du", "Hồ Xuân Hương", "Nguyễn Công Trứ",
]

PRODUCTS = [
    "Áo thun nam size XL màu đen", "Áo sơ mi trắng tay dài size M",
    "Quần jean ống rộng màu xanh nhạt size L", "Váy hoa nhí dáng dài size S",
    "Giày thể thao Nike Air Force 1 size 42", "Dép quai ngang Adidas màu trắng size 40",
    "iPhone 16 Pro Max 256GB Titan Sa Mạc", "Samsung Galaxy S25 Ultra 512GB",
    "Laptop Dell XPS 13 16GB/512GB", "Tai nghe AirPods Pro 2 chính hãng",
    "Nồi chiên không dầu Philips 6.2L", "Máy lọc không khí Xiaomi Gen 4",
    "Combo 3 hộp sữa tươi Vinamilk 1L", "Thùng 24 lon bia Heineken 330ml",
    "Kem chống nắng La Roche-Posay 50ml", "Sữa rửa mặt Cetaphil 500ml",
    "Balo laptop chống nước 15.6 inch", "Đồng hồ Casio G-Shock màu đen",
    "Mũ bảo hiểm 3/4 màu đen nhám", "Bàn phím cơ Keychron K2 Pro",
    "Chuột không dây Logitech MX Master 3S", "Bình giữ nhiệt Lock&Lock 800ml",
    "2 ly trà sữa trân châu đường đen ít đá", "1 phần cơm tấm sườn bì chả thêm trứng",
    "3kg gạo ST25 loại 1 túi hút chân không", "Cáp sạc nhanh Type-C 100W dài 2m",
    "Bộ chăn ga gối cotton 4 món 1m8x2m", "Ghế công thái học lưng lưới màu xám",
    "Sách Đắc Nhân Tâm bìa cứng tái bản 2025", "Đơn hàng #DH20260815 - 2 sản phẩm",
]

DELIVERY_NOTES_NO_ADDR = [
    "Giao giờ hành chính giúp em nhé",
    "Gọi trước khi giao 15 phút nha anh shipper",
    "Để ở quầy bảo vệ giúp mình, cảm ơn shop",
    "Không giao được buổi sáng, giao sau 6h tối",
    "Hàng dễ vỡ xin nhẹ tay giúp em",
    "Cho kiểm tra hàng trước khi thanh toán",
    "Ship hỏa tốc trong ngày giúp em với",
    "Gọi anh Hùng 0912345678 ra lấy hàng",
    "Nhắn zalo trước khi qua nhé shop",
    "Tới nơi bấm chuông cổng màu xanh giúp em",
    "Em đã chuyển khoản đủ tiền hàng rồi ạ",
    "Shipper gọi không nghe máy thì gọi số dự phòng nhé",
    "Nhờ shop đóng gói cẩn thận làm quà tặng",
    "Giao thứ 7 chủ nhật giúp mình vì ngày thường đi làm",
    "Đơn này thanh toán COD khi nhận hàng",
    "Vui lòng xuất hóa đơn VAT cho công ty",
    "Nếu không có người nhận thì gửi hàng xóm giúp",
    "Mình bận họp từ 14h đến 16h đừng gọi giờ đó",
]

CHAT_TEMPLATES_WITH_PLACES = [
    "{place} hôm nay mưa to quá",
    "{place} kẹt xe kinh khủng luôn",
    "Em đang đi du lịch {place} với gia đình",
    "Cuối tuần này đi {place} chơi không bạn",
    "{place} dạo này nắng nóng thật sự",
    "Mình quê ở {place} nhưng làm việc xa nhà",
    "Trời {place} mùa này đẹp quá đi mất",
    "Vé máy bay đi {place} tháng sau bao nhiêu tiền",
    "{place} có món ăn gì ngon nổi tiếng vậy mọi người",
    "Hôm qua ở {place} mất điện từ 2h chiều",
    "Bộ phim này quay ở {place} đẹp mê ly",
    "Tuyển nhân viên kinh doanh tại {place} lương 15 triệu",
    "Thời tiết {place} ngày mai có mưa không nhỉ",
    "Xe khách đi {place} chạy lúc mấy giờ tối nay",
    "{place} ngập nước nặng sau cơn mưa lớn",
    "Đi công tác {place} 3 ngày rồi mới về",
    "Năm 2025 em chuyển vào {place} sinh sống",
    "Ở {place} có quán cà phê nào yên tĩnh không bạn",
    "Học đại học ở {place} 4 năm nhiều kỷ niệm",
    "Tối nay rủ bạn đi ăn đặc sản {place} nha",
    "Giá nhà đất khu vực {place} năm nay tăng mạnh",
    "Đội bóng {place} tối qua thắng 2-1 rồi",
    "Từ {place} ra sân bay mất bao lâu vậy anh",
    "Hôm nay đi qua {place} thấy đông người quá",
    "Chiều nay kẹt xe ở {place} mất gần 1 tiếng",
    "Mình thích không khí buổi sáng ở {place}",
    "Đang tìm phòng trọ giá rẻ gần {place}",
    "Bạn mình mới mở quán trà sữa bên {place}",
    "Tết này gia đình mình về {place} ăn Tết",
    "Bão sắp đổ bộ vào khu vực {place} mọi người cẩn thận",
]

CHAT_GENERIC = [
    "Chào shop, sản phẩm này còn hàng không ạ?",
    "Cảm ơn bạn rất nhiều nhé",
    "Giá bao nhiêu vậy shop ơi",
    "Ok chốt đơn này giúp mình nhé",
    "Chúc mừng năm mới 2026 an khang thịnh vượng",
    "Alo anh nghe rõ không ạ",
    "Hẹn gặp lại ngày mai lúc 9h sáng",
    "Mình đã chuyển khoản thành công rồi nhé",
    "Anh Hùng sinh năm 1990 ở quê",
    "Gọi cho anh Tuấn lúc 5h chiều nhé",
    "Hôm nay trời mát mẻ dễ chịu quá",
    "Shop tư vấn giúp em size cho người 65kg cao 1m72",
    "Bao giờ hàng về tới nơi vậy bạn",
    "Mình muốn đổi sang màu xanh dương được không",
    "Tối nay ăn gì bây giờ mọi người ơi",
    "Gửi giúp em báo giá qua email nhé",
    "Đã nhận được hàng, chất lượng rất tốt 5 sao",
    "Cuộc họp dời sang thứ 5 tuần sau lúc 10h",
    "Nhớ mang theo giấy tờ tùy thân khi đi làm thủ tục",
    "Mã vận đơn của đơn này là gì vậy shop",
]

COMPANIES_ORGS = [
    "Công ty TNHH ABC Việt Nam", "Công ty Cổ phần Công nghệ FPT",
    "Ngân hàng Vietcombank", "Ngân hàng Techcombank", "Ngân hàng MB Bank",
    "Tập đoàn Vingroup", "Tập đoàn Viettel", "Công ty Cổ phần Thế Giới Di Động",
    "Trường Đại học Bách Khoa", "Trường Đại học Kinh tế Quốc dân",
    "Bệnh viện Chợ Rẫy", "Bệnh viện Bạch Mai", "Bệnh viện Từ Dũ",
    "Sở Kế hoạch và Đầu tư", "Chi cục Thuế khu vực", "Tổng công ty Điện lực",
    "Nhà xuất bản Kim Đồng", "Hãng hàng không Vietnam Airlines",
]


# ---------------------------------------------------------------------------
# Helpers for realistic Vietnamese noise / formatting
# ---------------------------------------------------------------------------

def _rand_house_number(rng: random.Random) -> str:
    style = rng.random()
    n1 = rng.randint(1, 680)
    if style < 0.45:
        return str(n1)
    elif style < 0.65:
        return f"{n1}{rng.choice(['A', 'B', 'C', 'D', 'a', 'b'])}"
    elif style < 0.85:
        n2 = rng.randint(1, 99)
        suffix = rng.choice(["", "", "", "A", "B", "C"])
        return f"{n1}/{n2}{suffix}"
    else:
        n2 = rng.randint(1, 60)
        n3 = rng.randint(1, 35)
        return f"{n1}/{n2}/{n3}"


def _format_ward(rng: random.Random, ward_name: str, numbered: bool = False) -> str:
    if numbered:
        num = rng.randint(1, 18)
        return rng.choice([f"Phường {num}", f"P.{num}", f"P. {num}", f"P{num}", f"p.{num}", f"p{num}"])
    return rng.choice([f"Phường {ward_name}", f"P. {ward_name}", f"P.{ward_name}", ward_name])


def _apply_vn_noise(rng: random.Random, text: str, force_unaccent: bool = False) -> str:
    out = text
    if force_unaccent or rng.random() < 0.36:
        out = strip_diacritics(out)
    r = rng.random()
    if r < 0.30:
        out = out.lower()
    elif r < 0.38:
        out = out.upper()
    # Occasional missing commas
    if rng.random() < 0.32:
        out = out.replace(", ", " ")
    elif rng.random() < 0.12:
        out = out.replace(", ", " - ")
    elif rng.random() < 0.10:
        out = out.replace(", ", ",")
    # Occasional double space or trimmed space
    if rng.random() < 0.12:
        out = re.sub(r"\s+", "  ", out, count=1)
    return out.strip()


def _pick_hn(rng: random.Random) -> Tuple[str, str, str]:
    dist_base, dist_forms = rng.choice(HN_DISTRICTS)
    ward = rng.choice(HN_WARDS)
    city = rng.choice(PROVINCES_63[0][1])
    return _format_ward(rng, ward), rng.choice(dist_forms), city


def _pick_hcm(rng: random.Random) -> Tuple[str, str, str]:
    dist_base, dist_forms = rng.choice(HCM_DISTRICTS)
    if rng.random() < 0.55:
        ward = _format_ward(rng, "", numbered=True)
    else:
        ward = _format_ward(rng, rng.choice(HCM_NAMED_WARDS))
    city = rng.choice(PROVINCES_63[1][1])
    return ward, rng.choice(dist_forms), city


def _pick_other(rng: random.Random) -> Tuple[str, str, str]:
    dist, prov = rng.choice(OTHER_DISTRICTS)
    ward = _format_ward(rng, rng.choice(OTHER_WARDS))
    # Find province variants
    prov_variants = [prov]
    for base, forms in PROVINCES_63:
        if base == prov:
            prov_variants = forms
            break
    return ward, dist, rng.choice(prov_variants)


def _pick_location(rng: random.Random) -> Tuple[str, str, str]:
    r = rng.random()
    if r < 0.36:
        return _pick_hn(rng)
    elif r < 0.72:
        return _pick_hcm(rng)
    else:
        return _pick_other(rng)


# ---------------------------------------------------------------------------
# Positive Address Generators (10 archetypes)
# ---------------------------------------------------------------------------

def gen_positive(rng: random.Random) -> str:
    archetype = rng.choices(
        range(10),
        weights=[20, 14, 12, 10, 12, 12, 8, 5, 4, 3],
        k=1,
    )[0]

    street = rng.choice(STREETS)
    hnum = _rand_house_number(rng)
    ward, dist, city = _pick_location(rng)

    if archetype == 0:
        # Formal full urban address
        num_pfx = rng.choice(["Số ", "Số nhà ", "SN ", "", "", ""])
        st_pfx = rng.choice(["Đường ", "Phố ", "đ. ", "", ""])
        if rng.random() < 0.85:
            raw = f"{num_pfx}{hnum} {st_pfx}{street}, {ward}, {dist}, {city}"
        else:
            raw = f"{num_pfx}{hnum} {st_pfx}{street}, {dist}, {city}"
        return _apply_vn_noise(rng, raw)

    elif archetype == 1:
        # Alley / Ngõ / Ngách / Hẻm / Kiệt
        alley_kw = rng.choice(["Ngõ", "Hẻm", "Ngách", "Kiệt", "ngo", "hem"])
        alley_no = rng.randint(1, 350)
        sub = f"/{rng.randint(1, 40)}" if rng.random() < 0.4 else ""
        if rng.random() < 0.45:
            # House + alley + street + dist + city
            sn = rng.choice(["Số", "SN", "sn", "Nhà số", ""])
            raw = f"{sn} {rng.randint(1, 80)} {alley_kw} {alley_no}{sub} {street}, {dist}, {city}".strip()
        elif rng.random() < 0.75:
            raw = f"{alley_kw} {alley_no}{sub} {street}, {ward}, {dist}, {city}"
        else:
            raw = f"{alley_kw} {alley_no}{sub} {street}, {dist}"
        return _apply_vn_noise(rng, raw)

    elif archetype == 2:
        # Apartment / Building / Office / Industrial zone
        if rng.random() < 0.75:
            bldg = rng.choice(APARTMENTS_BUILDINGS)
            unit_type = rng.choice([
                f"Căn hộ {rng.choice(['A','B','C','S1','S2'])}{rng.randint(1,9)}-{rng.randint(101,3512)}",
                f"P{rng.randint(201, 2815)} Tòa {rng.choice(['S1.02','S2.05','R4','T5','A1','B2'])}",
                f"Phòng {rng.randint(101, 2208)}, Tầng {rng.randint(2, 25)}",
                f"Tầng {rng.randint(2, 20)}",
                f"Lầu {rng.randint(1, 15)}",
                f"Block {rng.choice(['A','B','C','D'])} căn {rng.randint(101, 2010)}",
            ])
            if rng.random() < 0.5:
                raw = f"{unit_type}, {bldg}, {hnum} {street}, {dist}, {city}"
            else:
                raw = f"{unit_type} {bldg}, {dist}, {city}"
        else:
            iz = rng.choice(INDUSTRIAL_ZONES)
            lot = f"Lô {rng.choice(['A','B','C','D','E','CN'])}{rng.randint(1,9)}-{rng.randint(1,20)}"
            raw = f"{lot}, {iz}, {dist}, {city}"
        return _apply_vn_noise(rng, raw)

    elif archetype == 3:
        # Rural / Hamlet / Commune / Tổ / Khu phố
        hamlet = rng.choice([
            f"Thôn {rng.randint(1, 9)}",
            f"Thôn {rng.choice(['Đông', 'Đoài', 'Trung', 'Thượng', 'Hạ', 'Tân Lập', 'Bình Minh'])}",
            f"Ấp {rng.randint(1, 7)}",
            f"Ấp {rng.choice(['Tân Lập', 'Bình Hòa', 'Phước Thạnh', 'An Hòa', 'Bình Khánh'])}",
            f"Xóm {rng.randint(1, 12)}",
            f"Tổ {rng.randint(1, 25)}, Khu phố {rng.randint(1, 9)}",
            f"Tổ dân phố {rng.randint(1, 18)}",
            f"KP {rng.randint(1, 8)}",
            f"Khu phố {rng.randint(1, 9)}",
        ])
        commune = rng.choice(RURAL_COMMUNES)
        _, dist_r, city_r = _pick_other(rng) if rng.random() < 0.65 else _pick_location(rng)
        if rng.random() < 0.45:
            raw = f"{hamlet}, {commune}, {dist_r}, {city_r}"
        else:
            raw = f"{rng.choice(['Số ', ''])}{hnum} {hamlet}, {commune}, {dist_r}, {city_r}"
        return _apply_vn_noise(rng, raw)

    elif archetype == 4:
        # Short / compact urban addresses
        sub_type = rng.random()
        if sub_type < 0.35:
            raw = f"{hnum} {street}, {dist}, {city}"
        elif sub_type < 0.60:
            raw = f"{hnum} {street}, {dist}"
        elif sub_type < 0.80:
            alley_kw = rng.choice(["ngõ", "hẻm", "ngách"])
            raw = f"{alley_kw} {rng.randint(1, 250)} {street}"
        else:
            raw = f"số {hnum} {street}, {ward}"
        return _apply_vn_noise(rng, raw)

    elif archetype == 5:
        # Abbreviated / shorthand VN address
        w_num = rng.randint(1, 16)
        q_abbr = rng.choice([
            "Q1", "Q.1", "Q3", "Q.3", "Q5", "Q7", "Q.7", "Q10", "Q.10",
            "Q.TB", "Q.BT", "Q.PN", "Q.GV", "Q.TP", "Q.TX", "Q.CG", "Q.DD", "Q.HK", "TD",
        ])
        c_abbr = rng.choice(["HCM", "TPHCM", "HN", "SG", "DN", "HP", ""])
        p_abbr = rng.choice([f"P.{w_num}", f"P{w_num}", f"p.{w_num}", f"p{w_num}"])
        pattern = rng.random()
        if pattern < 0.30:
            raw = f"{p_abbr} {q_abbr} {c_abbr or 'HCM'}"
        elif pattern < 0.65:
            raw = f"{hnum} {street} {p_abbr} {q_abbr} {c_abbr}".strip()
        else:
            raw = f"sn {rng.randint(1, 99)} ngo {rng.randint(1, 250)} {street} {c_abbr or 'hn'}"
        return _apply_vn_noise(rng, raw, force_unaccent=rng.random() < 0.7)

    elif archetype == 6:
        # Delivery-embedded / chat-wrapped address
        name_short = rng.choice(["anh Hùng", "chị Lan", "anh Tuấn", "chị Mai", "anh Minh", "chị Thảo", "a Dũng", "c Hương", "em Nam", "anh Huy", "chị Hà"])
        prefix = rng.choice([
            f"giao {name_short} ",
            f"giao cho {name_short} ",
            f"ship {name_short} ",
            f"ship về ",
            f"ship đến ",
            f"gửi hàng tới ",
            f"địa chỉ: ",
            f"đ/c: ",
            f"dc: ",
            f"nhà mình ở ",
            f"nhận tại ",
            f"giao giờ hành chính ",
            f"ship cod ",
        ])
        suffix = rng.choice(["", "", "", "", " giúp em", " nha shop", " nhé"])
        core_style = rng.random()
        if core_style < 0.45:
            core = f"{hnum} {street}, {dist}, {city}"
        elif core_style < 0.80:
            core = f"{hnum} {street} {dist}"
        else:
            core = f"số {rng.randint(1, 90)} ngõ {rng.randint(1, 200)} {street}, {dist}"
        raw = f"{prefix}{core}{suffix}"
        return _apply_vn_noise(rng, raw, force_unaccent=rng.random() < 0.5)

    elif archetype == 7:
        # Administrative / Landmark / KĐT address without house number
        lm = rng.choice(LANDMARKS)
        if rng.random() < 0.5:
            raw = f"{lm}, {dist}, {city}"
        else:
            raw = f"Đường {street}, {ward}, {dist}, {city}"
        return _apply_vn_noise(rng, raw)

    elif archetype == 8:
        # Unaccented tight spacing / messy punctuation
        raw = f"{hnum} {street} {ward} {dist} {city}"
        out = strip_diacritics(raw).lower()
        out = out.replace("phuong ", rng.choice(["p", "p.", "phuong "]))
        out = out.replace("quan ", rng.choice(["q", "q.", "quan "]))
        return out

    else:
        # All 63 provinces explicit coverage
        prov_base, prov_forms = rng.choice(PROVINCES_63)
        prov_str = rng.choice(prov_forms)
        w_str = _format_ward(rng, rng.choice(OTHER_WARDS))
        raw = f"Số {hnum} {street}, {w_str}, {prov_str}"
        return _apply_vn_noise(rng, raw)


# ---------------------------------------------------------------------------
# Hard Negative Generators (12 archetypes)
# ---------------------------------------------------------------------------

def _rand_person_name(rng: random.Random) -> str:
    sn = rng.choice(SURNAMES)
    if rng.random() < 0.5:
        mid = rng.choice(MID_NAMES_M)
        giv = rng.choice(GIVEN_M)
    else:
        mid = rng.choice(MID_NAMES_F)
        giv = rng.choice(GIVEN_F)
    name = f"{sn} {mid} {giv}"
    if rng.random() < 0.15:
        name = f"{rng.choice(['Anh ', 'Chị ', 'Em ', 'Ông ', 'Bà ', 'Mr. ', 'Ms. '])}{name}"
    if rng.random() < 0.35:
        name = strip_diacritics(name)
    if rng.random() < 0.25:
        name = name.lower()
    return name


def _rand_phone(rng: random.Random) -> str:
    pfx = rng.choice(["09", "03", "07", "08", "05"])
    d = "".join(str(rng.randint(0, 9)) for _ in range(8))
    fmt = rng.random()
    if fmt < 0.40:
        return f"{pfx}{d}"
    elif fmt < 0.65:
        return f"+84 {pfx[1]}{d[:2]} {d[2:5]} {d[5:]}"
    elif fmt < 0.85:
        return f"{pfx}{d[:2]}.{d[2:5]}.{d[5:]}"
    else:
        return f"{rng.choice(['SĐT: ', 'sdt ', 'Zalo ', 'LH: '])}{pfx}{d}"


def gen_negative(rng: random.Random) -> str:
    archetype = rng.choices(
        range(12),
        weights=[14, 10, 8, 10, 12, 18, 9, 6, 5, 3, 2, 3],
        k=1,
    )[0]

    if archetype == 0:
        # Person names (including historical names matching street names)
        if rng.random() < 0.18:
            name = rng.choice(PERSON_HISTORICAL_TRAPS)
            return strip_diacritics(name).lower() if rng.random() < 0.3 else name
        return _rand_person_name(rng)

    elif archetype == 1:
        # Phone numbers
        return _rand_phone(rng)

    elif archetype == 2:
        # Emails, URLs, handles
        r = rng.random()
        user = strip_diacritics(rng.choice(GIVEN_M + GIVEN_F)).lower()
        sn = strip_diacritics(rng.choice(SURNAMES)).lower()
        num = rng.randint(1, 9999)
        if r < 0.45:
            dom = rng.choice(["gmail.com", "yahoo.com", "outlook.com", "fpt.vn", "company.com.vn", "icloud.com"])
            return f"{user}.{sn}{num if rng.random() < 0.5 else ''}@{dom}"
        elif r < 0.85:
            host = rng.choice(["shopee.vn", "tiki.vn", "lazada.vn", "vnexpress.net", "facebook.com", "github.com", "google.com"])
            slug = rng.choice(["product", "order", "item", "san-pham", "tin-tuc", "profile"])
            proto = rng.choice(["https://", "http://", "www.", ""])
            return f"{proto}{host}/{slug}/{num}"
        else:
            return f"@{user}_{sn}{num}"

    elif archetype == 3:
        # Dates, times, prices, bank accounts, IDs
        r = rng.random()
        if r < 0.28:
            dd = rng.randint(1, 28)
            mm = rng.randint(1, 12)
            yy = rng.randint(1985, 2028)
            sep = rng.choice(["/", "-", "."])
            return f"{dd:02d}{sep}{mm:02d}{sep}{yy}"
        elif r < 0.56:
            amt = rng.choice([50, 75, 120, 150, 250, 350, 490, 890, 1200, 2500, 15000])
            unit = rng.choice([".000đ", ".000 VND", ".000 đồng", "k", "K", ",000đ"])
            return f"{amt}{unit}"
        elif r < 0.78:
            hr = rng.randint(6, 22)
            mn = rng.choice(["00", "15", "30", "45"])
            return rng.choice([
                f"{hr}h{mn} ngày {rng.randint(1,28)}/{rng.randint(1,12)}",
                f"lúc {hr}:{mn} chiều nay",
                f"sinh năm {rng.randint(1975, 2006)}",
            ])
        else:
            digits = "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([10, 12, 13])))
            return rng.choice([f"STK {digits} VCB", f"CCCD {digits}", f"MST {digits}", f"MVD {digits}"])

    elif archetype == 4:
        # Products / SKUs / Orders (combinatorial for high unique diversity)
        r = rng.random()
        if r < 0.35:
            item = rng.choice([
                "Áo thun nam", "Áo sơ mi nữ", "Áo khoác gió", "Quần jean ống rộng",
                "Quần tây công sở", "Váy hoa nhí", "Giày thể thao", "Dép quai ngang",
                "Túi xách da", "Balo laptop", "Mũ bảo hiểm 3/4", "Kính mát thời trang",
            ])
            color = rng.choice(["màu đen", "màu trắng", "màu xanh navy", "màu be", "màu xám", "màu đỏ đô", "màu hồng"])
            size = rng.choice(["size S", "size M", "size L", "size XL", "size XXL", "size 38", "size 40", "size 42", "freesize"])
            extra = rng.choice(["", " cotton 100%", " form rộng", " chính hãng", f" x{rng.randint(1,5)}", f" {rng.randint(150,890)}k"])
            txt = f"{item} {size} {color}{extra}"
        elif r < 0.65:
            tech = rng.choice([
                f"iPhone {rng.randint(13,16)} {rng.choice(['Pro','Pro Max','Plus',''])} {rng.choice(['128GB','256GB','512GB','1TB'])}",
                f"Samsung Galaxy S{rng.randint(22,25)} {rng.choice(['Ultra','Plus','FE'])} {rng.choice(['128GB','256GB','512GB'])}",
                f"Laptop {rng.choice(['Dell XPS','MacBook Air','Asus Zenbook','Lenovo ThinkPad','HP Envy'])} {rng.choice(['13','14','15.6'])} inch {rng.choice(['8GB/256GB','16GB/512GB','32GB/1TB'])}",
                f"Tai nghe {rng.choice(['AirPods Pro','Sony WH-1000XM5','Marshall Major','JBL Tune'])} {rng.choice(['chính hãng','màu đen','màu trắng','bản 2025'])}",
                f"{rng.choice(['Nồi chiên không dầu','Máy lọc không khí','Bàn phím cơ','Chuột không dây','Bình giữ nhiệt'])} {rng.choice(['Philips','Xiaomi','Lock&Lock','Logitech','Keychron'])} {rng.choice(['5L','6.2L','Gen 4','Pro','800ml'])}",
            ])
            txt = re.sub(r"\s+", " ", tech).strip()
        elif r < 0.85:
            qty = rng.randint(1, 6)
            food = rng.choice([
                "hộp sữa tươi Vinamilk 1L", "lon bia Heineken 330ml", "ly trà sữa trân châu đường đen",
                "phần cơm tấm sườn bì chả", "kg gạo ST25 loại 1", "chai nước mắm Phú Quốc 500ml",
                "gói cà phê rang xay 500g", "hộp bánh trung thu thập cẩm", "lốc sữa chua không đường",
            ])
            note = rng.choice(["", " ít đá", " 50% đường", " giao nhanh", " freeship", f" #{rng.randint(100,999)}"])
            txt = f"Combo {qty} {food}{note}" if rng.random() < 0.5 else f"{qty} {food}{note}"
        else:
            txt = f"{rng.choice(['SKU', 'SP', 'DH', 'ORD', 'VN', 'ITEM', 'MDH'])}-{rng.randint(100,99999)}-{rng.choice(['XL','M','L','BLK','WHT','256G','PRO'])}"
        if rng.random() < 0.35:
            txt = strip_diacritics(txt).lower()
        return txt

    elif archetype == 5:
        # Casual chat mentioning real VN place names (HARDEST NEGATIVES!)
        place_pool = (
            [p[0] for p in PROVINCES_63[:30]]
            + [d[0] for d in HN_DISTRICTS[:14]]
            + [d[0] for d in HCM_DISTRICTS[:16]]
            + ["Sài Gòn", "TP.HCM", "Đà Lạt", "Nha Trang", "Vũng Tàu", "Phú Quốc", "Hội An", "Sa Pa", "Cầu Giấy", "Hoàn Kiếm", "Thủ Đức"]
        )
        place = rng.choice(place_pool)
        time_opt = rng.choice(["", "hôm nay ", "chiều nay ", "sáng nay ", "tối qua ", "tuần sau ", "tháng 6 ", "dạo này ", "mùa này ", "cuối tuần này "])
        subj = rng.choice(["Em", "Mình", "Anh", "Bạn mình", "Gia đình em", "Bọn mình", "Anh Hùng", "Chị Lan", "Cả nhà"])
        r = rng.random()
        if r < 0.45:
            tpl = rng.choice(CHAT_TEMPLATES_WITH_PLACES)
            tail = rng.choice(["", " nha", " nhé", " luôn á", " mọi người ơi", " quá trời", " thật sự"])
            txt = tpl.format(place=place) + tail
        elif r < 0.75:
            act = rng.choice([
                "đi du lịch", "đi công tác", "về thăm quê ở", "chuyển vào sống tại",
                "tìm quán ăn ngon ở", "đặt vé máy bay đi", "ghé qua chơi ở",
                "học đại học tại", "làm việc ở", "xem dự báo thời tiết",
            ])
            dur = rng.choice(["", f" {rng.randint(2,7)} ngày", " với bạn", " một mình", " vào thứ 6", " tháng sau"])
            txt = f"{time_opt}{subj} đang {act} {place}{dur}".strip()
        else:
            event = rng.choice([
                "mưa to quá", "kẹt xe kinh khủng", "nắng nóng 38 độ", "mất điện lúc 3h chiều",
                "ngập nặng sau mưa", "đông người lắm", "có quán cà phê nào đẹp không",
                "đồ ăn ngon mà rẻ", "giá thuê trọ tăng cao", "không khí mát mẻ dễ chịu",
            ])
            txt = f"Ở {place} {time_opt}{event}".replace("  ", " ").strip()
        if rng.random() < 0.35:
            txt = strip_diacritics(txt)
        if rng.random() < 0.25:
            txt = txt.lower()
        return txt

    elif archetype == 6:
        # Delivery notes WITHOUT an address (combinatorial)
        r = rng.random()
        if r < 0.40:
            base = rng.choice(DELIVERY_NOTES_NO_ADDR)
            extra = rng.choice(["", f" (đơn #{rng.randint(100,9999)})", " nha shop", " giúp mình", " ạ"])
            txt = f"{base}{extra}"
        else:
            verb = rng.choice(["Giao", "Ship", "Gọi", "Nhắn tin", "Để hàng", "Bấm chuông", "Liên hệ"])
            cond = rng.choice([
                f"sau {rng.randint(17,21)}h tối", f"trước {rng.randint(9,12)}h trưa",
                f"từ {rng.randint(8,11)}h đến {rng.randint(13,17)}h", "giờ hành chính thứ 2 đến thứ 6",
                f"trước khi tới {rng.choice([10,15,20,30])} phút", "cho người nhà nhận hộ",
                "ở quầy lễ tân hoặc bảo vệ", f"anh {rng.choice(GIVEN_M)} số {rng.choice(['09','03','08'])}{rng.randint(10000000,99999999)}",
            ])
            polite = rng.choice(["giúp em nhé", "nha anh shipper", "cảm ơn shop nhiều", "với ạ", "hộ mình"])
            txt = f"{verb} {cond} {polite}"
        if rng.random() < 0.35:
            txt = strip_diacritics(txt).lower()
        return txt

    elif archetype == 7:
        # Random strings, short garbage, English phrases, numbers
        r = rng.random()
        if r < 0.35:
            return rng.choice([
                "abc123", "hello world", "123456", "test 123", "ok bạn",
                "không có", "N/A", "none", "null", "asdfgh", "qwerty",
                "xyz789", "thank you", "good morning", "please check",
                "123456789", "000000", "a1b2c3d4", "---", "???",
            ])
        elif r < 0.70:
            chars = "abcdefghijklmnopqrstuvwxyz0123456789"
            length = rng.randint(4, 14)
            return "".join(rng.choice(chars) for _ in range(length))
        else:
            return str(rng.randint(10, 9999999))

    elif archetype == 8:
        # Specs / Math / Slashes / Numbers (false-positive traps for house/alley numbers)
        return rng.choice([
            f"Kích thước {rng.randint(10,90)}x{rng.randint(10,90)}x{rng.randint(5,50)} cm",
            f"Giảm giá {rng.randint(10,70)}% cho đơn từ {rng.randint(100,500)}k",
            f"Bài {rng.randint(1,20)} trang {rng.randint(10,200)} sách giáo khoa",
            f"Tỉ số {rng.randint(1,5)}/{rng.randint(0,4)} hiệp {rng.randint(1,2)}",
            f"Hóa đơn số {rng.randint(10000,99999)} ngày {rng.randint(1,28)}/{rng.randint(1,12)}/2026",
            f"Xe biển số {rng.choice(['29A','30G','51H','43A','59B'])}-{rng.randint(100,999)}.{rng.randint(10,99)}",
            f"Chuyến bay VN{rng.randint(100,999)} khởi hành lúc {rng.randint(6,22)}:30",
            f"Số lượng {rng.randint(2,50)} thùng x {rng.randint(6,24)} hộp",
            f"Mã giảm giá SALE{rng.randint(10,99)}K áp dụng đến {rng.randint(1,28)}/{rng.randint(1,12)}",
            f"Điểm thi {rng.randint(5,10)}/10 môn Toán lớp {rng.randint(6,12)}",
            f"Bàn số {rng.randint(1,30)} gọi {rng.randint(1,5)} phần bún chả",
            f"Lãi suất {rng.randint(5,12)}.5%/năm kỳ hạn {rng.choice([6,12,24])} tháng",
        ])

    elif archetype == 9:
        # Companies / organizations alone
        txt = rng.choice(COMPANIES_ORGS)
        return strip_diacritics(txt).lower() if rng.random() < 0.3 else txt

    elif archetype == 10:
        # Bare standalone province/city/generic word (incomplete, not an address)
        bare = rng.choice(
            [p[0] for p in PROVINCES_63[:15]]
            + ["Việt Nam", "TP.HCM", "Sài Gòn", "Hà Nội", "Đà Nẵng", "Quận 1", "Phường 5", "Địa chỉ", "Số nhà"]
        )
        return strip_diacritics(bare).lower() if rng.random() < 0.35 else bare

    else:
        # Generic Vietnamese chat (combinatorial)
        if rng.random() < 0.40:
            txt = rng.choice(CHAT_GENERIC)
        else:
            person = f"{rng.choice(['anh', 'chị', 'em', 'bạn'])} {rng.choice(GIVEN_M + GIVEN_F)}"
            txt = rng.choice([
                f"Gọi cho {person} lúc {rng.randint(1,11)}h {rng.choice(['sáng','trưa','chiều','tối'])}",
                f"{person.capitalize()} sinh năm {rng.randint(1978,2006)} ở quê",
                f"Chào {rng.choice(['shop','bạn','anh','chị'])}, mẫu #{rng.randint(10,999)} còn hàng không ạ?",
                f"Hẹn {person} {rng.choice(['ngày mai','thứ 5','cuối tuần'])} lúc {rng.randint(8,20)}h{rng.choice(['00','30'])}",
                f"Mình vừa chuyển {rng.randint(50,950)}k cho {person} rồi nhé",
                f"Cảm ơn {person} nhiều lắm, chúc một ngày tốt lành",
                f"Đơn #{rng.randint(1000,99999)} của {person} bao giờ giao tới vậy shop",
            ])
        if rng.random() < 0.30:
            txt = strip_diacritics(txt).lower()
        return txt


# ---------------------------------------------------------------------------
# Dataset Assembly & Split
# ---------------------------------------------------------------------------

@dataclass
class DatasetSplits:
    train: List[Tuple[str, int]]
    val: List[Tuple[str, int]]
    test: List[Tuple[str, int]]


def generate_dataset(
    n_per_class: int = 9000,
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> DatasetSplits:
    """Generate a balanced, deduplicated VN address dataset with zero edge-case leakage."""
    rng = random.Random(seed)
    forbidden = {text.strip().lower() for text, _, _ in CURATED_EDGE_CASES}

    positives: List[str] = []
    seen_pos = set(forbidden)
    attempts = 0
    while len(positives) < n_per_class and attempts < n_per_class * 20:
        attempts += 1
        s = gen_positive(rng).strip()
        key = s.lower()
        if s and key not in seen_pos:
            seen_pos.add(key)
            positives.append(s)

    negatives: List[str] = []
    seen_neg = set(forbidden) | seen_pos
    attempts = 0
    while len(negatives) < n_per_class and attempts < n_per_class * 20:
        attempts += 1
        s = gen_negative(rng).strip()
        key = s.lower()
        if s and key not in seen_neg:
            seen_neg.add(key)
            negatives.append(s)

    # Balance to exact min length if dedup constrained negatives
    n_actual = min(len(positives), len(negatives))
    data: List[Tuple[str, int]] = (
        [(t, 1) for t in positives[:n_actual]]
        + [(t, 0) for t in negatives[:n_actual]]
    )
    rng.shuffle(data)

    total = len(data)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    return DatasetSplits(
        train=data[:n_train],
        val=data[n_train : n_train + n_val],
        test=data[n_train + n_val :],
    )


if __name__ == "__main__":
    splits = generate_dataset()
    print(f"Train: {len(splits.train)}, Val: {len(splits.val)}, Test: {len(splits.test)}")
    print("Sample positives:", [x for x, y in splits.train if y == 1][:5])
    print("Sample negatives:", [x for x, y in splits.train if y == 0][:5])
