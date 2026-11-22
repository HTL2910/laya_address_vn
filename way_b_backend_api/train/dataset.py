"""Rich Vietnamese Address vs Hard-Negative Dataset Generator for Laya Address VN.

Generates >= 22,000 unique, balanced, deterministic samples with stratified train/val/test split:
- Positives:
  1. Full formal VN addresses (North/Central/South/Highlands)
  2. Unaccented addresses (khong dau)
  3. Abbreviated / compact addresses (P.Ben Nghe, Q.1, TP.HCM, KP3, Hẻm 214/19...)
  4. Typos / missing commas / noisy spacing / case variations
  5. Rural addresses (Thôn / Ấp / Xóm / Buôn / Bản / Xã / Thị trấn / Huyện)
  6. Apartment / office / tower / KCN addresses
  7. Chat / delivery embedded addresses (ship giúp mình tới ..., giao qua ...)
- Hard Negatives:
  1. Context-tricky Vietnamese sentences containing REAL city/district/ward/street/landmark names
     that are NOT addresses (weather, travel, sports, traffic, company branches, historical figures,
     meetups, shipping fee questions, product origin names) — both accented & unaccented!
  2. Vietnamese personal names & job titles
  3. Phone numbers, emails, URLs, social handles
  4. Bank transfer notes, order IDs, tracking codes, OTPs
  5. Dates, times, prices, dimensions, quantities
  6. E-commerce product titles (including regional specialties with city names)
  7. General chat, bare single region names, customer support messages
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from app.tokenizer import strip_accents


@dataclass(frozen=True)
class AddressSample:
    text: str
    label: int  # 1 = address, 0 = non-address
    category: str


# ---------------------------------------------------------------------------
# Geographic & Linguistic Knowledge Bases (authentic VN administrative data)
# ---------------------------------------------------------------------------

URBAN_REGIONS: List[Dict[str, object]] = [
    {
        "city_forms": ["Hà Nội", "TP. Hà Nội", "TP Hà Nội", "HN"],
        "districts": [
            ("Ba Đình", ["Kim Mã", "Giảng Võ", "Ngọc Khánh", "Đội Cấn", "Liễu Giai", "Điện Biên", "Quán Thánh", "Trúc Bạch"]),
            ("Hoàn Kiếm", ["Hàng Bài", "Hàng Bông", "Tràng Tiền", "Cửa Nam", "Hàng Trống", "Lý Thái Tổ", "Phan Chu Trinh", "Trần Hưng Đạo"]),
            ("Đống Đa", ["Láng Hạ", "Trung Liệt", "Quang Trung", "Ô Chợ Dừa", "Thịnh Quang", "Ngã Tư Sở", "Khâm Thiên", "Nam Đồng", "Phương Mai"]),
            ("Hai Bà Trưng", ["Bách Khoa", "Minh Khai", "Vĩnh Tuy", "Lê Đại Hành", "Nguyễn Du", "Bạch Mai", "Thanh Nhàn", "Đồng Nhân"]),
            ("Cầu Giấy", ["Dịch Vọng", "Dịch Vọng Hậu", "Trung Hòa", "Quan Hoa", "Mai Dịch", "Nghĩa Tân", "Yên Hòa", "Nghĩa Đô"]),
            ("Thanh Xuân", ["Thượng Đình", "Nhân Chính", "Khương Trung", "Khương Mai", "Thanh Xuân Trung", "Thanh Xuân Bắc", "Hạ Đình", "Kim Giang"]),
            ("Nam Từ Liêm", ["Mỹ Đình 1", "Mỹ Đình 2", "Mễ Trì", "Tây Mỗ", "Đại Mỗ", "Trung Văn", "Phú Đô", "Cầu Diễn"]),
            ("Bắc Từ Liêm", ["Cổ Nhuế 1", "Cổ Nhuế 2", "Xuân Đỉnh", "Phú Diễn", "Đông Ngạc", "Thụy Phương"]),
            ("Hoàng Mai", ["Đại Kim", "Giáp Bát", "Hoàng Liệt", "Mai Động", "Tân Mai", "Định Công", "Thịnh Liệt", "Yên Sở"]),
            ("Hà Đông", ["Văn Quán", "Mộ Lao", "Quang Trung", "Yết Kiêu", "La Khê", "Dương Nội", "Phúc La", "Vạn Phúc"]),
            ("Tây Hồ", ["Quảng An", "Xuân La", "Nhật Tân", "Tứ Liên", "Yên Phụ", "Thụy Khuê", "Bưởi"]),
            ("Long Biên", ["Ngọc Lâm", "Bồ Đề", "Gia Thụy", "Việt Hưng", "Đức Giang", "Thạch Bàn", "Sài Đồng"]),
        ],
        "streets": [
            "Láng Hạ", "Thái Hà", "Chùa Bộc", "Tôn Đức Thắng", "Xã Đàn", "Nguyễn Trãi",
            "Khuất Duy Tiến", "Lê Văn Lương", "Tố Hữu", "Trần Duy Hưng", "Nguyễn Chí Thanh",
            "Kim Mã", "Đội Cấn", "Hoàng Quốc Việt", "Xuân Thủy", "Cầu Giấy", "Hồ Tùng Mậu",
            "Phạm Hùng", "Mễ Trì", "Lê Đức Thọ", "Giải Phóng", "Trường Chinh", "Bạch Mai",
            "Minh Khai", "Tam Trinh", "Nguyễn Xiển", "Hai Bà Trưng", "Lý Thường Kiệt",
            "Trần Hưng Đạo", "Bà Triệu", "Phố Huế", "Hàng Đào", "Hàng Ngang", "Đinh Tiên Hoàng",
            "Lạc Long Quân", "Âu Cơ", "Nguyễn Văn Cừ", "Ngô Gia Tự", "Quang Trung", "Lê Trọng Tấn",
            "Phạm Văn Đồng", "Hoàng Hoa Thám", "Thụy Khuê", "Giảng Võ", "Võ Chí Công", "Tây Sơn",
        ],
        "buildings": [
            "Tòa nhà Keangnam", "Keangnam Landmark 72", "Lotte Center", "Vinhomes Smart City",
            "Vinhomes Ocean Park", "Times City", "Royal City", "Chung cư Goldmark City",
            "Tòa nhà Viettel", "Tòa nhà FPT", "Tòa nhà Peakview", "KĐT Linh Đàm",
            "KĐT Mỹ Đình", "Tòa nhà Charmvit", "Chung cư Mipec", "Tòa nhà Vinaconex",
            "Vinhomes Metropolis", "Tòa nhà Handico", "Chung cư Imperia Garden", "Tòa nhà Capital Place",
        ],
    },
    {
        "city_forms": ["TP. Hồ Chí Minh", "TP.HCM", "TPHCM", "HCM", "TP Hồ Chí Minh", "Sài Gòn"],
        "districts": [
            ("Quận 1", ["Bến Nghé", "Bến Thành", "Đa Kao", "Tân Định", "Nguyễn Thái Bình", "Phạm Ngũ Lão", "Cầu Ông Lãnh", "Cô Giang"]),
            ("Quận 3", ["Võ Thị Sáu", "Phường 1", "Phường 4", "Phường 5", "Phường 9", "Phường 11", "Phường 14"]),
            ("Quận 4", ["Phường 1", "Phường 2", "Phường 4", "Phường 6", "Phường 9", "Phường 13"]),
            ("Quận 5", ["Phường 1", "Phường 4", "Phường 7", "Phường 9", "Phường 11", "Phường 14"]),
            ("Quận 7", ["Tân Phong", "Tân Phú", "Tân Hưng", "Tân Quy", "Phú Mỹ", "Bình Thuận", "Tân Kiểng"]),
            ("Quận 10", ["Phường 1", "Phường 4", "Phường 8", "Phường 10", "Phường 12", "Phường 14", "Phường 15"]),
            ("Quận 12", ["Tân Thới Nhất", "Đông Hưng Thuận", "Hiệp Thành", "Thạnh Lộc", "Trung Mỹ Tây", "An Phú Đông"]),
            ("Bình Thạnh", ["Phường 1", "Phường 2", "Phường 11", "Phường 14", "Phường 17", "Phường 22", "Phường 25", "Phường 26"]),
            ("Phú Nhuận", ["Phường 1", "Phường 2", "Phường 5", "Phường 7", "Phường 9", "Phường 11", "Phường 15"]),
            ("Tân Bình", ["Phường 1", "Phường 2", "Phường 4", "Phường 12", "Phường 13", "Phường 14", "Phường 15"]),
            ("Tân Phú", ["Tân Sơn Nhì", "Tây Thạnh", "Sơn Kỳ", "Tân Quý", "Hòa Thạnh", "Phú Thọ Hòa"]),
            ("Gò Vấp", ["Phường 1", "Phường 3", "Phường 5", "Phường 7", "Phường 10", "Phường 11", "Phường 16"]),
            ("Bình Tân", ["Bình Trị Đông", "Bình Trị Đông A", "An Lạc", "Tân Tạo", "Bình Hưng Hòa"]),
            ("TP Thủ Đức", ["Thảo Điền", "An Phú", "Linh Trung", "Hiệp Bình Chánh", "Trường Thọ", "Linh Tây", "Phước Long B", "Tăng Nhơn Phú A"]),
        ],
        "streets": [
            "Nguyễn Huệ", "Lê Lợi", "Đồng Khởi", "Hai Bà Trưng", "Lý Tự Trọng", "Pasteur",
            "Nam Kỳ Khởi Nghĩa", "Cách Mạng Tháng 8", "CMT8", "Võ Văn Tần", "Nguyễn Đình Chiểu",
            "Điện Biên Phủ", "Nguyễn Thị Minh Khai", "Võ Thị Sáu", "Trần Hưng Đạo", "Nguyễn Trãi",
            "Lê Văn Sỹ", "Phan Xích Long", "Hoàng Văn Thụ", "Cộng Hòa", "Trường Chinh", "Âu Cơ",
            "Lạc Long Quân", "Nguyễn Văn Linh", "Huỳnh Tấn Phát", "Nguyễn Hữu Thọ", "Phạm Văn Đồng",
            "Kha Vạn Cân", "Võ Văn Ngân", "Nguyễn Văn Nguyễn", "Phan Đăng Lưu", "Xô Viết Nghệ Tĩnh",
            "Bạch Đằng", "Quang Trung", "Lê Đức Thọ", "Nguyễn Oanh", "Phan Văn Trị", "3 Tháng 2",
            "Sư Vạn Hạnh", "Lý Thường Kiệt", "Hồng Bàng", "Hùng Vương", "Xa Lộ Hà Nội", "Mai Chí Thọ",
            "Hải Triều", "Nguyễn Công Trứ", "Tôn Đức Thắng", "Bùi Viện", "Đinh Bộ Lĩnh", "Ung Văn Khiêm",
        ],
        "buildings": [
            "Bitexco Financial Tower", "Landmark 81", "Vinhomes Central Park", "Vinhomes Grand Park",
            "Masteri Thảo Điền", "Saigon Centre", "Sunrise City", "Chung cư The Sun Avenue",
            "Chung cư Lexington", "Tòa nhà Etown", "Tòa nhà Vietcombank", "KĐT Phú Mỹ Hưng",
            "KĐT Sala", "Chung cư RiverGate", "Tòa nhà Lim Tower", "Tòa nhà Pearl Plaza",
            "Tòa nhà Deutsches Haus", "Chung cư Vinhomes Golden River", "Tòa nhà MPlaza", "KCN Tân Bình",
        ],
    },
    {
        "city_forms": ["Đà Nẵng", "TP. Đà Nẵng", "TP Đà Nẵng", "ĐN"],
        "districts": [
            ("Hải Châu", ["Hải Châu 1", "Hải Châu 2", "Thạch Thang", "Nam Dương", "Phước Ninh", "Hòa Thuận Đông", "Thuận Phước"]),
            ("Thanh Khê", ["Tân Chính", "Vĩnh Trung", "Thạc Gián", "An Khê", "Thanh Khê Đông", "Xuân Hà"]),
            ("Sơn Trà", ["An Hải Bắc", "An Hải Đông", "Phước Mỹ", "Mân Thái", "Nại Hiên Đông", "Thọ Quang"]),
            ("Ngũ Hành Sơn", ["Mỹ An", "Khuê Mỹ", "Hòa Hải", "Hòa Quý"]),
            ("Liên Chiểu", ["Hòa Khánh Bắc", "Hòa Khánh Nam", "Hòa Minh", "Hòa Hiệp Nam"]),
            ("Cẩm Lệ", ["Khuê Trung", "Hòa Thọ Đông", "Hòa Xuân", "Hòa An"]),
        ],
        "streets": [
            "Bạch Đằng", "Trần Phú", "Nguyễn Văn Linh", "Lê Duẩn", "Hùng Vương", "Phan Châu Trinh",
            "Hoàng Diệu", "Nguyễn Hữu Thọ", "Võ Nguyên Giáp", "Phạm Văn Đồng", "Ngô Quyền",
            "Điện Biên Phủ", "Tôn Đức Thắng", "Nguyễn Tất Thành", "2 Tháng 9", "Cách Mạng Tháng 8",
            "Lê Thanh Nghị", "Xô Viết Nghệ Tĩnh", "Hàm Nghi", "Quang Trung", "Ông Ích Khiêm",
        ],
        "buildings": [
            "Tòa nhà Indochina Riverside", "Chung cư Azura", "Tòa nhà FPT Complex", "KĐT FPT City",
            "Tòa nhà Software Park", "Chung cư Monarchy", "Tòa nhà Hilton Đà Nẵng", "KCN Hòa Khánh",
        ],
    },
    {
        "city_forms": ["Đồng Nai", "Tỉnh Đồng Nai"],
        "districts": [
            ("Biên Hòa", ["Tam Hòa", "Tân Phong", "Trảng Dài", "Tân Hiệp", "Thống Nhất", "Hố Nai", "Long Bình", "An Bình", "Quang Vinh"]),
            ("TP Biên Hòa", ["Tam Hòa", "Tân Mai", "Tam Hiệp", "Bửu Long", "Hiệp Hòa", "Phước Tân"]),
            ("Long Thành", ["Thị trấn Long Thành", "An Phước", "Long Đức", "Bình Sơn"]),
            ("Nhơn Trạch", ["Hiệp Phước", "Phú Hữu", "Đại Phước", "Long Thọ"]),
        ],
        "streets": [
            "Phạm Văn Thuận", "Đồng Khởi", "Nguyễn Ái Quốc", "Võ Thị Sáu", "Hà Huy Giáp",
            "Bùi Văn Hòa", "Quốc lộ 1A", "Quốc lộ 51", "Huỳnh Văn Nghệ", "30 Tháng 4",
        ],
        "buildings": [
            "KCN Biên Hòa 2", "KCN Amata", "Chung cư Topaz Twins", "Tòa nhà Sonadezi", "KCN Long Đức",
        ],
    },
    {
        "city_forms": ["Bình Dương", "Tỉnh Bình Dương"],
        "districts": [
            ("TP Thủ Dầu Một", ["Phú Hòa", "Phú Cường", "Chánh Nghĩa", "Hiệp Thành", "Hòa Phú", "Phú Lợi"]),
            ("TP Thuận An", ["Thuận Giao", "An Phú", "Lái Thiêu", "Bình Chuẩn", "Vĩnh Phú"]),
            ("TP Dĩ An", ["Dĩ An", "Đông Hòa", "Tân Đông Hiệp", "Bình An", "An Bình"]),
            ("TP Bến Cát", ["Mỹ Phước", "Thới Hòa", "Hòa Lợi", "Chánh Phú Hòa"]),
        ],
        "streets": [
            "Đại lộ Bình Dương", "Quốc lộ 13", "Cách Mạng Tháng 8", "Phú Lợi", "Huỳnh Văn Lũy",
            "Lê Hồng Phong", "Nguyễn Thị Minh Khai", "DT743", "Mỹ Phước Tân Vạn", "Trần Hưng Đạo",
        ],
        "buildings": [
            "KCN VSIP 1", "KCN VSIP 2", "KCN Sóng Thần", "Chung cư Sora Gardens", "Becamex Tower",
            "Chung cư Happy One", "Chung cư The Emerald Golf View",
        ],
    },
    {
        "city_forms": ["Hải Phòng", "TP. Hải Phòng", "TP Hải Phòng"],
        "districts": [
            ("Ngô Quyền", ["Lạch Tray", "Cầu Đất", "Máy Tơ", "Đằng Giang", "Gia Viên"]),
            ("Lê Chân", ["An Biên", "Cát Dài", "Hàng Kênh", "Kênh Dương", "Vĩnh Niệm"]),
            ("Hồng Bàng", ["Hoàng Văn Thụ", "Minh Khai", "Phan Bội Châu", "Sở Dầu"]),
            ("Hải An", ["Đằng Hải", "Đằng Lâm", "Cát Bi", "Đông Hải 1"]),
        ],
        "streets": [
            "Lạch Tray", "Điện Biên Phủ", "Trần Phú", "Cầu Đất", "Tô Hiệu", "Lê Hồng Phong",
            "Nguyễn Văn Linh", "Trần Nguyên Hãn", "Đà Nẵng", "Văn Cao", "Quán Nam",
        ],
        "buildings": [
            "Vinhomes Imperia", "Tòa nhà TD Plaza", "Chung cư Hoàng Huy Commerce", "KCN VSIP Hải Phòng",
        ],
    },
    {
        "city_forms": ["Cần Thơ", "TP. Cần Thơ", "TP Cần Thơ"],
        "districts": [
            ("Ninh Kiều", ["Tân An", "An Nghiệp", "An Cư", "Xuân Khánh", "Hưng Lợi", "An Khánh", "Cái Khế"]),
            ("Cái Răng", ["Lê Bình", "Hưng Phú", "Hưng Thạnh", "Thường Thạnh"]),
            ("Bình Thủy", ["Bình Thủy", "An Thới", "Trà Nóc", "Long Hòa"]),
        ],
        "streets": [
            "30 Tháng 4", "3 Tháng 2", "Mậu Thân", "Nguyễn Văn Cừ", "Trần Hưng Đạo",
            "Hòa Bình", "Hai Bà Trưng", "Cách Mạng Tháng 8", "Nguyễn Trãi", "Lý Tự Trọng",
        ],
        "buildings": [
            "Vincom Xuân Khánh", "KCN Trà Nóc", "Tòa nhà STS Tower", "KĐT Nam Long Cần Thơ",
        ],
    },
    {
        "city_forms": ["Khánh Hòa", "Tỉnh Khánh Hòa"],
        "districts": [
            ("TP Nha Trang", ["Lộc Thọ", "Tân Lập", "Phước Hòa", "Vĩnh Hải", "Vĩnh Nguyên", "Xương Huân", "Phương Sài"]),
            ("TP Cam Ranh", ["Cam Lộc", "Cam Phú", "Ba Ngòi", "Cam Nghĩa"]),
        ],
        "streets": [
            "Trần Phú", "Nguyễn Thị Minh Khai", "Lê Thánh Tôn", "Thái Nguyên", "Quang Trung",
            "Hùng Vương", "2 Tháng 4", "Lý Tự Trọng", "Yersin", "Phạm Văn Đồng",
        ],
        "buildings": [
            "Chung cư Mường Thanh Viễn Triều", "Tòa nhà Gold Coast", "Vincom Plaza Trần Phú",
        ],
    },
    {
        "city_forms": ["Thừa Thiên Huế", "Huế", "TP Huế"],
        "districts": [
            ("TP Huế", ["Phú Hội", "Vĩnh Ninh", "Phú Nhuận", "Xuân Phú", "Vỹ Dạ", "An Cựu", "Trường An", "Thuận Hòa"]),
        ],
        "streets": [
            "Lê Lợi", "Hùng Vương", "Nguyễn Huệ", "Hà Nội", "Đống Đa", "Bà Triệu",
            "Trần Hưng Đạo", "Mai Thúc Loan", "Phan Bội Châu", "Bến Nghé",
        ],
        "buildings": [
            "Tòa nhà HCC", "Chung cư Vicoland", "KĐT An Cựu City", "Vincom Plaza Huế",
        ],
    },
    {
        "city_forms": ["Lâm Đồng", "Tỉnh Lâm Đồng"],
        "districts": [
            ("TP Đà Lạt", ["Phường 1", "Phường 2", "Phường 3", "Phường 4", "Phường 8", "Phường 9", "Phường 10"]),
            ("TP Bảo Lộc", ["Phường 1", "Phường 2", "B'Lao", "Lộc Sơn"]),
        ],
        "streets": [
            "Nguyễn Thị Minh Khai", "Trần Phú", "Phan Đình Phùng", "Bùi Thị Xuân", "Hai Bà Trưng",
            "Hùng Vương", "3 Tháng 2", "Phù Đổng Thiên Vương", "Xô Viết Nghệ Tĩnh", "Yersin",
        ],
        "buildings": [
            "Khu Hòa Bình", "Chung cư La Pine", "Tòa nhà Dalat Center",
        ],
    },
    {
        "city_forms": ["Nghệ An", "Tỉnh Nghệ An", "Thanh Hóa", "Quảng Ninh", "Bắc Ninh", "Bà Rịa - Vũng Tàu", "Bình Định"],
        "districts": [
            ("TP Vinh", ["Lê Mao", "Hồng Sơn", "Quang Trung", "Trường Thi", "Hà Huy Tập", "Hưng Dũng"]),
            ("TP Hạ Long", ["Bãi Cháy", "Hồng Gai", "Hồng Hải", "Cao Thắng", "Giếng Đáy"]),
            ("TP Bắc Ninh", ["Suối Hoa", "Tiền An", "Võ Cường", "Ninh Xá", "Đại Phúc"]),
            ("TP Vũng Tàu", ["Phường 1", "Phường 2", "Phường 3", "Phường 7", "Thắng Tam", "Nguyễn An Ninh"]),
            ("TP Quy Nhơn", ["Lê Lợi", "Trần Phú", "Nguyễn Văn Cừ", "Ghềnh Ráng", "Ngô Mây"]),
        ],
        "streets": [
            "Lê Mao", "Nguyễn Văn Cừ", "Quang Trung", "Lê Lợi", "Trần Phú", "Hạ Long",
            "Nguyễn Văn Linh", "Ngô Gia Tự", "Lý Thái Tổ", "Thùy Vân", "Ba Cu", "An Dương Vương",
        ],
        "buildings": [
            "KCN Yên Phong", "KCN Quế Võ", "Chung cư Tecco", "Tòa nhà Dầu Khí", "Vinhomes Dragon Bay",
        ],
    },
]

# Rural administrative tuples: (hamlet_prefix, hamlet_name, commune, district, province)
RURAL_ENTRIES: List[Tuple[str, str, str, str, str]] = [
    ("Thôn", "3", "Xã Ea Kao", "TP Buôn Ma Thuột", "Đắk Lắk"),
    ("Thôn", "5", "Xã Hòa Thắng", "TP Buôn Ma Thuột", "Đắk Lắk"),
    ("Buôn", "Ako Dhong", "Phường Tân Lợi", "TP Buôn Ma Thuột", "Đắk Lắk"),
    ("Buôn", "Kô Siêr", "Phường Tân Lập", "TP Buôn Ma Thuột", "Đắk Lắk"),
    ("Ấp", "Tân Lập", "Xã Tân Thông Hội", "Huyện Củ Chi", "TP.HCM"),
    ("Ấp", "Trung Hòa", "Xã Trung An", "Huyện Củ Chi", "TP.HCM"),
    ("Ấp", "4", "Xã Phước Kiển", "Huyện Nhà Bè", "TP.HCM"),
    ("Ấp", "3", "Xã Xuân Thới Thượng", "Huyện Hóc Môn", "TP.HCM"),
    ("Ấp", "2", "Xã Vĩnh Lộc A", "Huyện Bình Chánh", "TP.HCM"),
    ("Ấp", "Bình Hòa", "Xã Tân Phú Trung", "Huyện Củ Chi", "TP.HCM"),
    ("Thôn", "Phú Mỹ", "Xã Ngọc Mỹ", "Huyện Quốc Oai", "Hà Nội"),
    ("Thôn", "Đông", "Xã Kim Chung", "Huyện Đông Anh", "Hà Nội"),
    ("Thôn", "2", "Xã Phù Đổng", "Huyện Gia Lâm", "Hà Nội"),
    ("Thôn", "Tân Tiến", "Xã Tiên Dược", "Huyện Sóc Sơn", "Hà Nội"),
    ("Thôn", "Hậu Ái", "Xã Vân Canh", "Huyện Hoài Đức", "Hà Nội"),
    ("Xóm", "4", "Xã Xuân Trường", "Huyện Nam Đàn", "Nghệ An"),
    ("Xóm", "Chùa", "Xã Nghi Phú", "TP Vinh", "Nghệ An"),
    ("Xóm", "7", "Xã Diễn Châu", "Huyện Diễn Châu", "Nghệ An"),
    ("Thôn", "Tân An", "Xã Hòa Tiến", "Huyện Hòa Vang", "Đà Nẵng"),
    ("Thôn", "An Ngãi Đông", "Xã Hòa Sơn", "Huyện Hòa Vang", "Đà Nẵng"),
    ("Ấp", "Mỹ Thạnh", "Xã Mỹ Phong", "TP Mỹ Tho", "Tiền Giang"),
    ("Ấp", "Bình Tạo", "Xã Trung An", "TP Mỹ Tho", "Tiền Giang"),
    ("Ấp", "1", "Thị trấn Bến Lức", "Huyện Bến Lức", "Long An"),
    ("Ấp", "Rừng Sến", "Xã Mỹ Hạnh Bắc", "Huyện Đức Hòa", "Long An"),
    ("Ấp", "Phước Hải", "Thị trấn Long Thành", "Huyện Long Thành", "Đồng Nai"),
    ("Tổ", "12", "Xã An Phước", "Huyện Long Thành", "Đồng Nai"),
    ("Ấp", "Bình Chánh", "Xã Khánh Bình", "TP Tân Uyên", "Bình Dương"),
    ("Tổ", "5", "Thị trấn Diên Khánh", "Huyện Diên Khánh", "Khánh Hòa"),
    ("Thôn", "Vĩnh Điềm Thượng", "Xã Vĩnh Hiệp", "TP Nha Trang", "Khánh Hòa"),
    ("Thôn", "1", "Xã Cẩm Thanh", "TP Hội An", "Quảng Nam"),
    ("Thôn", "Đông Yên", "Xã Duy Trinh", "Huyện Duy Xuyên", "Quảng Nam"),
    ("Bản", "Cát Cát", "Xã Hoàng Liên", "Thị xã Sa Pa", "Lào Cai"),
    ("Bản", "Lác", "Xã Chiềng Châu", "Huyện Mai Châu", "Hòa Bình"),
    ("Ấp", "Gành Gió", "Xã Cửa Dương", "TP Phú Quốc", "Kiên Giang"),
    ("Ấp", "Cây Thông Trong", "Xã Cửa Dương", "TP Phú Quốc", "Kiên Giang"),
    ("Khóm", "3", "Phường Mỹ Long", "TP Long Xuyên", "An Giang"),
    ("Ấp", "Tân Thành", "Xã Tân Tập", "Huyện Cần Giuộc", "Long An"),
    ("Thôn", "Quảng Xá", "Xã Tân Ninh", "Huyện Quảng Ninh", "Quảng Bình"),
    ("Thôn", "Hải Ninh", "Xã Bình Thạnh", "Huyện Bình Sơn", "Quảng Ngãi"),
    ("Ấp", "6", "Xã Khánh An", "Huyện U Minh", "Cà Mau"),
    ("Thôn", "An Lạc", "Xã Trưng Trắc", "Huyện Văn Lâm", "Hưng Yên"),
    ("Thôn", "Đình Bảng", "Phường Đình Bảng", "TP Từ Sơn", "Bắc Ninh"),
    ("Thôn", "Ngọc Lặc", "Thị trấn Ngọc Lặc", "Huyện Ngọc Lặc", "Thanh Hóa"),
    ("Thôn", "Tân Lập", "Xã Ea Tu", "TP Buôn Ma Thuột", "Đắk Lắk"),
    ("Thôn", "8", "Xã Quảng Phú", "Huyện Cư M'gar", "Đắk Lắk"),
]

# Context-tricky location lists for hard negatives
ALL_CITIES = [
    "Hà Nội", "Sài Gòn", "TP.HCM", "Đà Nẵng", "Hải Phòng", "Cần Thơ", "Huế", "Đà Lạt",
    "Nha Trang", "Phú Quốc", "Vũng Tàu", "Quy Nhơn", "Hạ Long", "Hội An", "Sa Pa",
    "Buôn Ma Thuột", "Biên Hòa", "Bình Dương", "Đồng Nai", "Nghệ An", "Thanh Hóa",
    "Cà Mau", "Bắc Ninh", "Quảng Ninh", "Khánh Hòa", "Đắk Lắk", "Lâm Đồng", "Tiền Giang",
    "An Giang", "Long An", "Hải Dương", "Nam Định", "Thái Nguyên",
]
ALL_DISTRICTS = [
    "Quận 1", "Quận 3", "Quận 5", "Quận 7", "Quận 10", "Bình Thạnh", "Tân Bình",
    "Phú Nhuận", "Gò Vấp", "Thủ Đức", "Củ Chi", "Cầu Giấy", "Thanh Xuân", "Đống Đa",
    "Hoàn Kiếm", "Ba Đình", "Hai Bà Trưng", "Nam Từ Liêm", "Hà Đông", "Tây Hồ",
    "Long Biên", "Hoàng Mai", "Mỹ Đình", "Hải Châu", "Sơn Trà", "Ninh Kiều",
]
ALL_STREETS_AND_PERSONS = [
    "Nguyễn Trãi", "Hai Bà Trưng", "Lê Lợi", "Trần Hưng Đạo", "Nguyễn Huệ",
    "Lý Thường Kiệt", "Quang Trung", "Võ Thị Sáu", "Nguyễn Thị Minh Khai",
    "Phan Bội Châu", "Phan Châu Trinh", "Nguyễn Đình Chiểu", "Lê Duẩn",
    "Điện Biên Phủ", "Cách Mạng Tháng 8", "Láng Hạ", "Phạm Hùng", "Xuân Thủy",
    "Thái Hà", "Kim Mã", "Trần Phú", "Hùng Vương", "Pasteur", "Đồng Khởi",
]
LANDMARKS = [
    "Mỹ Đình", "Hồ Gươm", "Bến Thành", "Cầu Rồng", "Hàng Xanh", "Suối Tiên",
    "Chợ Rẫy", "Bạch Mai", "Keangnam", "Landmark 81", "Bitexco", "Phố đi bộ Nguyễn Huệ",
    "Công viên Thống Nhất", "Hồ Tây", "Cầu Nhật Tân", "Sân bay Tân Sơn Nhất",
    "Sân bay Nội Bài", "Bến xe Miền Đông", "Ga Hà Nội", "Đầm Sen", "Bà Nà Hills",
]

VN_SURNAMES = [
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ",
    "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý", "Trương", "Đinh",
]
VN_MIDDLES = ["Văn", "Thị", "Minh", "Hoàng", "Quang", "Ngọc", "Thanh", "Thu", "Đức", "Anh", "Gia", "Khánh", "Hữu", "Bảo"]
VN_GIVEN = [
    "An", "Bình", "Cường", "Dũng", "Hà", "Hải", "Hạnh", "Hiếu", "Hòa", "Hùng", "Huy",
    "Hương", "Khanh", "Khoa", "Lan", "Linh", "Long", "Mai", "Nam", "Nga", "Ngân", "Nhung",
    "Phong", "Phúc", "Phương", "Quân", "Quỳnh", "Sơn", "Tâm", "Thảo", "Thành", "Thư", "Trang",
    "Trung", "Tú", "Tuấn", "Tùng", "Uyên", "Vân", "Việt", "Vinh", "Vy", "Yến",
]
JOB_ROLES = [
    "Phòng Kinh Doanh", "Phòng Kế Toán", "Trưởng phòng Nhân sự", "Kỹ sư phần mềm",
    "Quản lý dự án", "Chuyên viên tư vấn", "Giám đốc điều hành", "Nhân viên kho vận",
    "Bộ phận CSKH", "Bác sĩ chuyên khoa", "Kế toán trưởng",
]

PRODUCTS = [
    "Áo khoác gió nam chống nước", "Áo thun cotton form rộng", "Quần jean ống suông nữ",
    "Giày thể thao chạy bộ nam", "Tai nghe Bluetooth chống ồn", "Điện thoại iPhone 16 Pro Max 256GB",
    "Laptop ThinkPad X1 Carbon Gen 12", "Bàn phím cơ không dây switch đỏ", "Chuột gaming siêu nhẹ",
    "Nồi chiên không dầu Philips 6.2L", "Máy lọc không khí Xiaomi", "Bình giữ nhiệt Lock&Lock 500ml",
    "Kem chống nắng nâng tone SPF50+", "Son kem lì màu đỏ đất", "Balo laptop chống sốc 15.6 inch",
    "Đồng hồ thông minh theo dõi sức khỏe", "Pin sạc dự phòng 20000mAh sạc nhanh",
    "Bộ chăn ga gối cotton cao cấp", "Vali kéo du lịch size 24 inch", "Sữa rửa mặt dịu nhẹ cho da dầu",
]
REGIONAL_PRODUCTS = [
    "Cà phê rang xay Buôn Ma Thuột gói 500g", "Bánh đậu xanh Hải Dương hộp 400g",
    "Nước mắm Phú Quốc 40 độ đạm chai 500ml", "Trà ô long Đà Lạt hộp quà 250g",
    "Mè xửng Huế đặc sản gói 300g", "Bánh pía Sóc Trăng trứng muối hộp 4 cái",
    "Chả mực Hạ Long giã tay túi 1kg", "Nem chua Thanh Hóa chục 10 cái",
    "Mực một nắng Nha Trang hút chân không 500g", "Áo dài truyền thống Huế vải lụa tơ tằm",
    "Hạt điều rang muối Bình Phước hũ 500g", "Kẹo dừa Bến Tre vị lá dứa hộp 300g",
    "Phở khô Gia Lai đặc sản gói 500g", "Gạo ST25 Sóc Trăng túi 5kg chính hãng",
    "Bánh tráng phơi sương Tây Ninh combo 3 bịch", "Chả bò Đà Nẵng loại 1 cây 500g",
    "Cốm làng Vòng Hà Nội gói lá sen 300g", "Khô cá lóc An Giang loại ngon 500g",
    "Tỏi Lý Sơn Quảng Ngãi túi 1kg", "Bưởi da xanh Bến Tre quả 1.5kg",
]
BANKS = ["Techcombank", "Vietcombank", "MB Bank", "VPBank", "BIDV", "ACB", "TPBank", "Sacombank", "Agribank", "VietinBank"]
EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "fintech.vn", "company.com.vn", "icloud.com", "edu.vn"]
URL_HOSTS = [
    "https://shopee.vn/product/", "https://tiki.vn/san-pham/", "https://lazada.vn/khuyen-mai/",
    "https://vnexpress.net/thoi-su/", "https://github.com/org/repo/", "https://docs.fintech.vn/api/",
    "www.dienmayxanh.com/may-lanh/", "www.thegioididong.com/dtdd/", "https://maps.example.vn/route/",
]


# ---------------------------------------------------------------------------
# Helpers for realistic Vietnamese address formatting & noise injection
# ---------------------------------------------------------------------------

def _house_num(rng: random.Random) -> str:
    mode = rng.random()
    if mode < 0.42:
        return str(rng.randint(1, 680))
    if mode < 0.68:
        return f"{rng.randint(1, 450)}/{rng.randint(1, 65)}"
    if mode < 0.82:
        return f"{rng.randint(1, 320)}{rng.choice(['A', 'B', 'C', 'D', 'E'])}"
    if mode < 0.93:
        return f"{rng.randint(1, 280)}/{rng.randint(1, 45)}/{rng.randint(1, 25)}"
    return f"{rng.randint(1, 180)}/{rng.randint(1, 30)}{rng.choice(['A', 'B', 'C'])}"


def _format_ward(ward: str, rng: random.Random, compact: bool = False) -> str:
    if ward.startswith(("Phường ", "Xã ", "Thị trấn ")):
        if compact:
            if ward.startswith("Phường "):
                rest = ward[len("Phường "):]
                return rng.choice([f"P.{rest}", f"P. {rest}", f"P{rest}", f"Phường {rest}"])
            if ward.startswith("Xã "):
                rest = ward[len("Xã "):]
                return rng.choice([f"X.{rest}", f"X. {rest}", f"Xã {rest}"])
            rest = ward[len("Thị trấn "):]
            return rng.choice([f"TT.{rest}", f"TT {rest}", f"Thị trấn {rest}"])
        return ward
    if compact:
        return rng.choice([f"P.{ward}", f"P. {ward}", f"P {ward}", ward])
    return rng.choice([f"Phường {ward}", f"P. {ward}", ward])


def _format_district(district: str, rng: random.Random, compact: bool = False) -> str:
    if district.startswith("Quận "):
        num = district[len("Quận "):]
        if compact:
            return rng.choice([f"Q.{num}", f"Q{num}", f"Q. {num}", f"Q {num}"])
        return rng.choice([f"Quận {num}", f"Q.{num}", f"Q{num}"])
    if district.startswith(("TP ", "Huyện ", "Thị xã ")):
        if compact and district.startswith("Huyện "):
            rest = district[len("Huyện "):]
            return rng.choice([f"H.{rest}", f"H. {rest}", rest])
        if compact and district.startswith("TP "):
            rest = district[len("TP "):]
            return rng.choice([f"TP.{rest}", f"TP {rest}", rest])
        return district
    if compact:
        if district == "Bình Thạnh" and rng.random() < 0.35:
            return rng.choice(["Q.BT", "Q.Bình Thạnh", "Bình Thạnh"])
        if district == "Tân Bình" and rng.random() < 0.35:
            return rng.choice(["Q.TB", "Q.Tân Bình", "Tân Bình"])
        return rng.choice([f"Q.{district}", f"Q. {district}", district])
    return rng.choice([f"Quận {district}", f"Q. {district}", district])


def _inject_vn_typo(text: str, rng: random.Random) -> str:
    """Apply realistic Vietnamese typing errors / missing commas / spacing quirks."""
    out = text
    if rng.random() < 0.55:
        out = out.replace(",", "")
    if rng.random() < 0.45:
        out = strip_accents(out)
    typo_pairs = [
        ("Phường", "Phuog"),
        ("phuong", "phuog"),
        ("Nguyễn", "Ngyễn"),
        ("nguyen", "ngyen"),
        ("Quận", "Qận"),
        ("quan", "qan"),
        ("Đường", "Đuog"),
        ("duong", "duog"),
        ("Hà Nội", "HaNoi"),
        ("ha noi", "hanoi"),
        ("TP.HCM", "TPHCM"),
        ("Hồ Chí Minh", "HoChiMinh"),
        ("Tòa nhà", "Toa nha"),
        ("Thôn", "Thon"),
    ]
    rng.shuffle(typo_pairs)
    applied = 0
    for src, dst in typo_pairs:
        if src in out and rng.random() < 0.35:
            out = out.replace(src, dst, 1)
            applied += 1
            if applied >= 2:
                break
    if rng.random() < 0.3:
        out = out.replace(". ", ".")
    if rng.random() < 0.25:
        out = out.lower()
    elif rng.random() < 0.12:
        out = out.upper()
    if rng.random() < 0.2:
        # Double space somewhere
        parts = out.split(" ")
        if len(parts) > 3:
            idx = rng.randint(1, len(parts) - 2)
            parts[idx] = parts[idx] + " "
            out = " ".join(parts)
    return out.strip()


def _pick_urban(rng: random.Random) -> Tuple[str, str, str, str, str]:
    region = rng.choice(URBAN_REGIONS)
    city = rng.choice(region["city_forms"])  # type: ignore[arg-type]
    district, wards = rng.choice(region["districts"])  # type: ignore[arg-type]
    ward = rng.choice(wards)
    street = rng.choice(region["streets"])  # type: ignore[arg-type]
    building = rng.choice(region["buildings"])  # type: ignore[arg-type]
    return city, district, ward, street, building


# ---------------------------------------------------------------------------
# Positive Generators (7 categories)
# ---------------------------------------------------------------------------

def _gen_pos_formal(rng: random.Random) -> str:
    city, dist, ward, street, _ = _pick_urban(rng)
    hnum = _house_num(rng)
    prefix = rng.choice(["Số ", "Số nhà ", "", ""])
    street_prefix = rng.choice(["Đường ", "Phố ", "", ""])
    alley = ""
    if rng.random() < 0.28:
        alley_kw = rng.choice(["Ngõ", "Hẻm", "Ngách", "Kiệt"])
        alley = f", {alley_kw} {rng.randint(1, 250)}" if rng.random() < 0.4 else f" {alley_kw} {rng.randint(1, 250)}"
        if rng.random() < 0.35:
            return f"Số {rng.randint(1, 90)}, {alley_kw} {rng.randint(1, 250)} {street}, {_format_ward(ward, rng)}, {_format_district(dist, rng)}, {city}"
    w_str = _format_ward(ward, rng)
    d_str = _format_district(dist, rng)
    if rng.random() < 0.2:
        return f"{prefix}{hnum} {street_prefix}{street}, {d_str}, {city}"
    return f"{prefix}{hnum}{alley} {street_prefix}{street}, {w_str}, {d_str}, {city}"


def _gen_pos_unaccented(rng: random.Random) -> str:
    mode = rng.random()
    city, dist, ward, street, building = _pick_urban(rng)
    hnum = _house_num(rng)
    if mode < 0.35:
        alley = rng.choice(["ngo", "hem", "ngach"])
        text = f"so {rng.randint(1, 120)} {alley} {rng.randint(1, 260)} {street} {dist} {city}"
    elif mode < 0.68:
        sep = rng.choice([" ", ", "])
        text = f"{rng.choice(['so ', ''])}{hnum} {street}{sep}{_format_ward(ward, rng, compact=rng.random() < 0.5)}{sep}{_format_district(dist, rng, compact=rng.random() < 0.5)}{sep}{city}"
    elif mode < 0.85:
        # Unaccented rural
        hp, hn, comm, rdist, prov = rng.choice(RURAL_ENTRIES)
        sep = rng.choice([" ", ", "])
        text = f"{hp} {hn}{sep}{comm}{sep}{rdist}{sep}{prov}"
    else:
        floor = rng.randint(1, 35)
        text = f"tang {floor} {building} {street} {dist} {city}"
    text = strip_accents(text)
    return text.lower() if rng.random() < 0.78 else text


def _gen_pos_abbreviated(rng: random.Random) -> str:
    city, dist, ward, street, _ = _pick_urban(rng)
    mode = rng.random()
    if mode < 0.25:
        # Ward + District + City compact (e.g. "P.Ben Nghe, Q.1, TP.HCM")
        sep = rng.choice([", ", " "])
        s = f"{_format_ward(ward, rng, compact=True)}{sep}{_format_district(dist, rng, compact=True)}{sep}{city}"
    elif mode < 0.50:
        # KP / Tổ + Ward + District + City (e.g. "KP3 P.Tam Hoa Bien Hoa Dong Nai")
        kp = rng.choice([f"KP{rng.randint(1, 12)}", f"KP.{rng.randint(1, 12)}", f"Tổ {rng.randint(1, 24)}", f"TDP {rng.randint(1, 15)}"])
        sep = rng.choice([" ", ", "])
        s = f"{kp}{sep}{_format_ward(ward, rng, compact=True)}{sep}{_format_district(dist, rng, compact=True)}{sep}{city}"
    elif mode < 0.78:
        # Alley + Street + Compact District (e.g. "Hẻm 214/19 Nguyễn Văn Nguyễn Q1")
        alley_kw = rng.choice(["Hẻm", "Ngõ", "Số", ""])
        hnum = _house_num(rng)
        d_str = _format_district(dist, rng, compact=True)
        city_part = f" {city}" if rng.random() < 0.55 else ""
        s = f"{alley_kw} {hnum} {street} {d_str}{city_part}".strip()
    else:
        # Street + Ward + District compact
        hnum = _house_num(rng)
        s = f"{hnum} {street}, {_format_ward(ward, rng, compact=True)}, {_format_district(dist, rng, compact=True)}, {city}"
    if rng.random() < 0.45:
        s = strip_accents(s)
    return s


def _gen_pos_typo_noisy(rng: random.Random) -> str:
    base = rng.choice([_gen_pos_formal, _gen_pos_abbreviated, _gen_pos_apartment])(rng)
    return _inject_vn_typo(base, rng)


def _gen_pos_rural(rng: random.Random) -> str:
    hp, hn, comm, rdist, prov = rng.choice(RURAL_ENTRIES)
    sep = rng.choice([", ", ", ", " "])
    include_prov = rng.random() < 0.80
    include_to = rng.random() < 0.25
    parts: List[str] = []
    if include_to:
        parts.append(f"Tổ {rng.randint(1, 18)}")
    elif rng.random() < 0.18:
        parts.append(f"Số {rng.randint(1, 240)}")
    parts.append(f"{hp} {hn}")
    parts.append(comm)
    # Sometimes drop "Huyện "
    if rdist.startswith("Huyện ") and rng.random() < 0.40:
        parts.append(rdist[len("Huyện "):])
    else:
        parts.append(rdist)
    if include_prov:
        parts.append(prov)
    out = sep.join(parts)
    if rng.random() < 0.35:
        out = strip_accents(out)
        if rng.random() < 0.6:
            out = out.lower()
    return out


def _gen_pos_apartment(rng: random.Random) -> str:
    city, dist, ward, street, building = _pick_urban(rng)
    mode = rng.random()
    sep = rng.choice([", ", " "])
    if mode < 0.38:
        lvl = rng.choice([f"Tầng {rng.randint(1, 45)}", f"Lầu {rng.randint(1, 35)}"])
        s = f"{lvl}{sep}{building}{sep}{street}{sep}{_format_district(dist, rng, compact=rng.random() < 0.4)}{sep}{city}"
    elif mode < 0.72:
        room = rng.choice([
            f"Phòng {rng.randint(1, 35)}{rng.randint(1, 18):02d}",
            f"Căn hộ {rng.choice(['A', 'B', 'C', 'S1.', 'T'])}{rng.randint(1, 28)}-{rng.randint(1, 16):02d}",
            f"P.{rng.randint(2, 30)}{rng.randint(1, 15):02d}",
            f"CH {rng.randint(5, 25)}{rng.randint(1, 12):02d}",
        ])
        block = rng.choice([f"Tòa {rng.choice(['A', 'B', 'C', 'S1.02', 'T6', 'Park 3'])}", f"Block {rng.choice(['A', 'B', 'C'])}", ""])
        parts = [p for p in [room, block, building, _format_ward(ward, rng, compact=rng.random() < 0.3) if rng.random() < 0.45 else street, _format_district(dist, rng, compact=rng.random() < 0.4), city] if p]
        s = sep.join(parts)
    else:
        # Industrial park / office lot
        lot = f"Lô {rng.choice(['A', 'B', 'C', 'D', 'E'])}{rng.randint(1, 24)}"
        s = f"{lot}{sep}{building}{sep}{_format_district(dist, rng, compact=rng.random() < 0.3)}{sep}{city}"
    if rng.random() < 0.32:
        s = strip_accents(s)
        if rng.random() < 0.5:
            s = s.lower()
    return s


def _gen_pos_chat_embedded(rng: random.Random) -> str:
    addr = rng.choice([
        _gen_pos_formal,
        _gen_pos_unaccented,
        _gen_pos_abbreviated,
        _gen_pos_rural,
        _gen_pos_apartment,
    ])(rng)
    # Occasionally generate shorter street+district delivery address like "88 hai bà trưng hoàn kiếm"
    if rng.random() < 0.35:
        city, dist, _, street, _ = _pick_urban(rng)
        hnum = _house_num(rng)
        dist_clean = _format_district(dist, rng, compact=rng.random() < 0.45)
        addr = f"{hnum} {street} {dist_clean}" + (f" {city}" if rng.random() < 0.4 else "")
        if rng.random() < 0.55:
            addr = strip_accents(addr).lower()

    prefixes = [
        "ship giúp mình tới", "giao hàng qua", "gửi về", "địa chỉ nhận hàng:",
        "nhà mình ở", "mình ở", "ship tới", "giao cho em tới", "shop gửi về địa chỉ",
        "giao qua địa chỉ", "nhận hàng tại", "địa chỉ giao hàng:", "em ở", "cho mình đặt giao về",
    ]
    suffixes = ["nhé", "nha", "giúp mình nha", "giúp em ạ", "nha shop", "nhé shop", "ạ", "", ""]
    pref = rng.choice(prefixes)
    suff = rng.choice(suffixes)
    if rng.random() < 0.45:
        pref = strip_accents(pref)
        suff = strip_accents(suff)
    out = f"{pref} {addr} {suff}".strip()
    return out


# ---------------------------------------------------------------------------
# Hard Negative Generators (7 categories, heavy weight on context-tricky VN)
# ---------------------------------------------------------------------------

def _gen_neg_tricky_location_sentence(rng: random.Random) -> str:
    city = rng.choice(ALL_CITIES)
    city2 = rng.choice([c for c in ALL_CITIES if c != city])
    dist = rng.choice(ALL_DISTRICTS)
    dist2 = rng.choice([d for d in ALL_DISTRICTS if d != dist])
    person_street = rng.choice(ALL_STREETS_AND_PERSONS)
    landmark = rng.choice(LANDMARKS)
    n_days = rng.randint(2, 7)
    n_nights = max(1, n_days - 1)
    hour = rng.randint(6, 22)

    templates = [
        # Weather & mood
        f"{city} hôm nay trở lạnh rồi",
        f"{city} mùa này hay mưa chiều lắm",
        f"Cuối tuần này ở {city} trời nắng đẹp quá",
        f"{city} mấy hôm nay mưa rả rích buồn ghê",
        f"Sáng sớm ở {city} se lạnh thích thật",
        f"Thời tiết {city} dạo này thay đổi thất thường quá",
        f"Mùa thu {city} lúc nào cũng có mùi hoa sữa",
        f"Khu {dist} tối nay trời đổ mưa to quá mọi người ơi",
        # Travel & transport
        f"Mình muốn đi du lịch {city} {n_days} ngày {n_nights} đêm",
        f"Tết này gia đình mình dự định đi {city} chơi",
        f"Cuối tháng có chuyến công tác ra {city} {n_days} ngày",
        f"Vé máy bay khứ hồi {city} đi {city2} bao nhiêu tiền vậy?",
        f"Đi xe khách từ {city} lên {city2} mất khoảng mấy tiếng?",
        f"Mùa hè này bạn có định đi {city} không?",
        f"Hôm qua mình vừa bay từ {city} về {city2} xong",
        f"Lịch trình tham quan {city} tự túc siêu tiết kiệm",
        # Sports, events, traffic, real estate news
        f"Đội tuyển Việt Nam đá ở {landmark} tối nay",
        f"Tối mai có bắn pháo hoa ở {landmark} {city}",
        f"Giá nhà ở {dist} và {dist2} tăng mạnh năm nay",
        f"Kẹt xe kinh khủng đoạn {landmark} với đường {person_street}",
        f"Đường {person_street} tối cuối tuần đông nghịt người đi bộ",
        f"Tuyến metro đi qua {landmark} sắp chính thức vận hành rồi",
        f"Hôm nay {city} tổ chức lễ hội âm nhạc ngoài trời cực lớn",
        f"Nghe nói giá thuê mặt bằng quanh {dist} giảm nhẹ so với năm ngoái",
        # Company footprint & vague location chatter (NOT deliverable addresses)
        f"Công ty mình có chi nhánh ở {dist} và {dist2}",
        f"Bên mình hiện có cửa hàng tại {city}, {city2} và nhiều tỉnh khác",
        f"Bạn ấy quê ở {city} nhưng hiện đang làm việc tại {city2}",
        f"Mình đang tìm phòng trọ quanh khu vực {dist} hoặc {dist2}",
        f"Nhà bạn trai mình ở tận {city} xa lắm",
        f"Em đang học đại học ở {dist} chị ơi",
        f"Chi nhánh {dist} hôm nay nghỉ kiểm kê định kỳ",
        f"Mình sinh ra và lớn lên ở {city} hơn 20 năm rồi",
        f"Bọn mình vừa mở thêm văn phòng đại diện tại {city}",
        f"Tuyển dụng nhân viên kinh doanh khu vực {dist} và {dist2}",
        # Historical figures / schools / hospitals overlapping street names
        f"Anh {person_street} là danh nhân văn hóa nổi tiếng",
        f"{person_street} là vị anh hùng dân tộc trong lịch sử Việt Nam",
        f"Mình mới mua cuốn sách lịch sử viết về {person_street}",
        f"Trường THPT {person_street} năm nay lấy điểm chuẩn rất cao",
        f"Bệnh viện {landmark} sáng nay đông bệnh nhân khám tổng quát",
        f"Hôm nay thầy giáo giảng bài về cuộc đời của {person_street}",
        # Casual meetup / delivery-question WITHOUT address
        f"Gặp nhau lúc {hour}h tối ở quán cà phê nhé",
        f"Hẹn mọi người {hour}h sáng mai trước cổng {landmark} nha",
        f"Lát nữa tan làm đi ăn phở gần {dist} không bạn?",
        f"Đợi mình 15 phút ở sảnh tầng 1 nha",
        f"Shop ơi kiểm tra giúp mình đơn hàng giao về {city} tới đâu rồi",
        f"Phí ship từ {city} vào {city2} hết bao nhiêu vậy shop?",
        f"Gửi hàng đi {city} thì khoảng mấy ngày nhận được ạ?",
        f"Ship cho mình 2 ly trà sữa ít đường ít đá nha",
        f"Bên shop có hỗ trợ giao hỏa tốc nội thành {city} không?",
        f"Đơn giao khu vực {dist} bao giờ bưu tá gọi phát vậy ạ?",
        # Bare region / vague non-address phrases
        f"{city}",
        f"{dist}",
        f"{city} - {city2}",
        f"Mình ở {city}",
        f"Khu vực {dist}",
        f"Chi nhánh {city}",
    ]
    s = rng.choice(templates)
    if rng.random() < 0.38:
        s = strip_accents(s)
        if rng.random() < 0.55:
            s = s.lower()
    return s


def _gen_neg_person_name(rng: random.Random) -> str:
    full = f"{rng.choice(VN_SURNAMES)} {rng.choice(VN_MIDDLES)} {rng.choice(VN_GIVEN)}"
    mode = rng.random()
    if mode < 0.55:
        out = full
    elif mode < 0.80:
        out = f"{rng.choice(['Anh', 'Chị', 'BS.', 'TS.', 'Mr.', 'Ms.'])} {full}"
    else:
        out = f"{full} - {rng.choice(JOB_ROLES)}"
    if rng.random() < 0.35:
        out = strip_accents(out)
    return out


def _gen_neg_contact(rng: random.Random) -> str:
    mode = rng.random()
    if mode < 0.38:
        pref = rng.choice(["09", "03", "07", "08", "05"])
        rest = "".join(str(rng.randint(0, 9)) for _ in range(8))
        num = pref + rest
        fmt = rng.choice([
            num,
            f"{num[:4]} {num[4:7]} {num[7:]}",
            f"{num[:4]}.{num[4:7]}.{num[7:]}",
            f"+84 {num[1:4]} {num[4:7]} {num[7:]}",
            f"SĐT: {num}",
            f"zalo {num[:4]}.{num[4:7]}.{num[7:]}",
        ])
        return fmt
    if mode < 0.68:
        user = strip_accents(f"{rng.choice(VN_SURNAMES)}{rng.choice(VN_GIVEN)}").lower()
        user += str(rng.randint(1, 999)) if rng.random() < 0.6 else ""
        return f"{user}@{rng.choice(EMAIL_DOMAINS)}"
    slug = f"{rng.choice(['item', 'post', 'order', 'v1', 'promo'])}-{rng.randint(1000, 999999)}"
    return f"{rng.choice(URL_HOSTS)}{slug}"


def _gen_neg_bank_order(rng: random.Random) -> str:
    order_id = rng.randint(10000, 999999)
    name = strip_accents(f"{rng.choice(VN_SURNAMES)} {rng.choice(VN_MIDDLES)} {rng.choice(VN_GIVEN)}").upper()
    bank = rng.choice(BANKS)
    stk = "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(10, 14)))
    otp = rng.randint(100000, 999999)
    track = f"{rng.choice(['SPXVN', 'GHN', 'GHTK', 'VTP', 'JT'])}{rng.randint(100000000, 999999999)}"
    templates = [
        f"chuyen tien don hang {order_id}",
        f"{name} chuyen khoan thanh toan hoa don {order_id}",
        f"CK tien coc don #{order_id} thang {rng.randint(1, 12)}",
        f"Thanh toan tien dien thang {rng.randint(1, 12)} ma KH PD{order_id}",
        f"STK {stk} {bank} {name}",
        f"Mã vận đơn {track} đang trung chuyển",
        f"Đơn hàng #DH{order_id} đã giao thành công",
        f"Ma OTP cua ban la {otp}, khong chia se cho bat ky ai",
        f"{track}",
        f"Hoàn tiền giao dịch #{order_id} về tài khoản {bank}",
    ]
    out = rng.choice(templates)
    if rng.random() < 0.35:
        out = strip_accents(out)
    return out


def _gen_neg_date_price(rng: random.Random) -> str:
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(2022, 2027)
    hh, mm = rng.randint(0, 23), rng.choice([0, 15, 30, 45])
    price_k = rng.choice([49, 99, 150, 299, 450, 750, 1250, 3490])
    templates = [
        f"{d:02d}/{m:02d}/{y} {hh:02d}:{mm:02d}",
        f"Ngày {d} tháng {m} năm {y}",
        f"{price_k}.000 VNĐ",
        f"Giá khuyến mãi chỉ còn {price_k}k hôm nay",
        f"Tổng thanh toán: {price_k:,}đ (đã gồm VAT)".replace(",", "."),
        f"Trọng lượng {rng.randint(1, 25) / 2:.1f}kg kích thước {rng.randint(10, 60)}x{rng.randint(10, 50)}x{rng.randint(5, 40)}cm",
        f"Giảm giá {rng.choice([10, 15, 20, 30, 50])}% từ ngày {d}/{m} đến {min(28, d + 5)}/{m}",
        f"{d:02d}-{m:02d}-{y}",
        f"Số lượng: {rng.randint(1, 50)} cái - Đơn giá {price_k}k",
    ]
    out = rng.choice(templates)
    if rng.random() < 0.3:
        out = strip_accents(out)
    return out


def _gen_neg_product(rng: random.Random) -> str:
    if rng.random() < 0.45:
        out = rng.choice(REGIONAL_PRODUCTS)
        if rng.random() < 0.4:
            out += f" giá {rng.choice([85, 120, 199, 250, 380])}k"
    else:
        base = rng.choice(PRODUCTS)
        suffix = rng.choice([
            f"size {rng.choice(['S', 'M', 'L', 'XL', 'XXL'])} màu {rng.choice(['đen', 'trắng', 'xanh navy', 'be'])}",
            f"bảo hành chính hãng {rng.choice([6, 12, 24])} tháng",
            f"combo {rng.randint(2, 5)} hộp tiết kiệm",
            f"phiên bản {rng.randint(2024, 2026)} chính hãng VN/A",
            "",
        ])
        out = f"{base} {suffix}".strip()
    if rng.random() < 0.3:
        out = strip_accents(out)
    return out


def _gen_neg_chat_misc(rng: random.Random) -> str:
    msgs = [
        "Ok bạn nhé",
        "Cảm ơn shop nhiều ạ",
        "Cho mình hủy đơn này nha shop ơi",
        "Sản phẩm này còn màu đen size L không bạn?",
        "Mình đã nhận được hàng rồi, đóng gói rất cẩn thận",
        "Tư vấn giúp mình mẫu này với ạ",
        "Hôm nay shop có làm việc không?",
        "Gửi mình xin hóa đơn đỏ công ty nhé",
        "Kiểm tra giúp mình xem đã nhận được tiền chưa ạ",
        "Lát nữa gọi lại cho mình sau 5h chiều nhé",
        "Đặt trước 2 suất cơm trưa văn phòng nhé",
        "Xin chào, mình cần hỗ trợ đổi trả sản phẩm bị lỗi",
        "Màu thực tế có giống trên hình không shop?",
        "Hẹn gặp lại bạn vào thứ 5 tuần sau nhé",
        "Nhớ nhắn tin trước khi giao giúp mình nha",
        "Vâng ạ, em cảm ơn anh nhiều",
        "Báo giá sỉ cho mình số lượng 100 bộ với",
        "Đăng ký gói cước 4G tốc độ cao theo tháng",
        "Mật khẩu wifi quán mình là gì vậy bạn?",
        "Chúc cả nhà cuối tuần vui vẻ và nhiều sức khỏe",
    ]
    out = rng.choice(msgs)
    if rng.random() < 0.4:
        out = strip_accents(out).lower()
    return out


# ---------------------------------------------------------------------------
# Dataset Builder with Stratified Deterministic Split
# ---------------------------------------------------------------------------

def generate_dataset(
    total_samples: int = 22000,
    seed: int = 42,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> Dict[str, List[AddressSample]]:
    """Generate a balanced, deduplicated, stratified Vietnamese address dataset."""
    rng = random.Random(seed)
    n_pos_target = total_samples // 2
    n_neg_target = total_samples - n_pos_target

    pos_plan: Sequence[Tuple[str, float, object]] = [
        ("pos_formal", 0.20, _gen_pos_formal),
        ("pos_unaccented", 0.18, _gen_pos_unaccented),
        ("pos_abbreviated", 0.16, _gen_pos_abbreviated),
        ("pos_typo_noisy", 0.12, _gen_pos_typo_noisy),
        ("pos_rural", 0.12, _gen_pos_rural),
        ("pos_apartment", 0.11, _gen_pos_apartment),
        ("pos_chat_embedded", 0.11, _gen_pos_chat_embedded),
    ]
    neg_plan: Sequence[Tuple[str, float, object]] = [
        ("neg_tricky_location", 0.36, _gen_neg_tricky_location_sentence),
        ("neg_person_name", 0.11, _gen_neg_person_name),
        ("neg_contact", 0.11, _gen_neg_contact),
        ("neg_bank_order", 0.11, _gen_neg_bank_order),
        ("neg_date_price", 0.10, _gen_neg_date_price),
        ("neg_product", 0.11, _gen_neg_product),
        ("neg_chat_misc", 0.10, _gen_neg_chat_misc),
    ]

    seen_texts: set[str] = set()
    by_category: Dict[str, List[AddressSample]] = {}

    def _fill(plan: Sequence[Tuple[str, float, object]], target_total: int, label: int) -> None:
        allocated = 0
        for idx, (cat, frac, fn) in enumerate(plan):
            count = (target_total - allocated) if idx == len(plan) - 1 else int(round(target_total * frac))
            allocated += count
            bucket: List[AddressSample] = []
            attempts = 0
            max_attempts = count * 15
            while len(bucket) < count and attempts < max_attempts:
                attempts += 1
                txt = fn(rng).strip()  # type: ignore[operator]
                key = txt.lower()
                # Allow slight controlled duplication only if generator space nears saturation
                if key not in seen_texts or attempts > count * 8:
                    seen_texts.add(key)
                    bucket.append(AddressSample(text=txt, label=label, category=cat))
            by_category[cat] = bucket

    _fill(pos_plan, n_pos_target, 1)
    _fill(neg_plan, n_neg_target, 0)

    train_list: List[AddressSample] = []
    val_list: List[AddressSample] = []
    test_list: List[AddressSample] = []

    for cat, samples in by_category.items():
        rng.shuffle(samples)
        n = len(samples)
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        test_list.extend(samples[:n_test])
        val_list.extend(samples[n_test : n_test + n_val])
        train_list.extend(samples[n_test + n_val :])

    rng.shuffle(train_list)
    rng.shuffle(val_list)
    rng.shuffle(test_list)

    return {"train": train_list, "val": val_list, "test": test_list}


if __name__ == "__main__":
    splits = generate_dataset(total_samples=22000, seed=42)
    for name, items in splits.items():
        pos = sum(1 for s in items if s.label == 1)
        neg = len(items) - pos
        print(f"{name:5s}: total={len(items):5d} (pos={pos:5d}, neg={neg:5d})")
    print("\nSample positives:")
    for s in [x for x in splits["test"] if x.label == 1][:8]:
        print(f"  [{s.category:18s}] {s.text}")
    print("\nSample hard negatives:")
    for s in [x for x in splits["test"] if x.label == 0][:8]:
        print(f"  [{s.category:18s}] {s.text}")
