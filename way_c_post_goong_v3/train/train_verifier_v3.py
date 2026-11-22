"""Synthesize 60k training/val/test + 600 holdout pairs, train MLP, calibrate, and export ONNX opset 14."""

from __future__ import annotations

import copy
import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TRAIN_DIR = Path(__file__).resolve().parent
PKG_DIR = TRAIN_DIR.parent
ROOT_DIR = PKG_DIR.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from features import (  # noqa: E402
    DEFAULT_ARTIFACTS_DIR,
    FEATURE_NAMES,
    GazetteerV3,
    haversine_km,
    normalize_vn_ascii,
    strip_accents,
)

SHARED_DIR = ROOT_DIR / "shared_data_v3"
RECORDS_PATH = SHARED_DIR / "vn_admin_v3_records.json"
CONVERTER_SRC_PATH = SHARED_DIR / "vn_admin_v3_converter.json"

TRAIN_SEED = 301
HOLDOUT_SEED = 20261199

TRAIN_STREETS = (
    "Nguyễn Huệ", "Lê Lợi", "Trần Hưng Đạo", "Nguyễn Trãi", "Hai Bà Trưng",
    "Bình Thuận", "Nguyễn Thái Bình", "Hòa Bình", "Điện Biên Phủ", "Trường Chinh",
    "Cách Mạng Tháng 8", "Lê Văn Lương", "Lý Thường Kiệt", "Phan Đình Phùng",
    "Quang Trung", "Bà Triệu", "Phan Văn Trị", "Nguyễn Văn Linh", "Xô Viết Nghệ Tĩnh",
    "Võ Thị Sáu", "Lạc Long Quân", "Âu Cơ", "Giải Phóng", "Nguyễn Xiển",
)
HOLDOUT_STREETS = (
    "Hoàng Diệu", "Ngô Quyền", "Tôn Đức Thắng", "Võ Văn Tần", "Pasteur",
    "Nguyễn Thị Minh Khai", "Phạm Văn Đồng", "Trần Duy Hưng", "Nguyễn Chí Thanh",
    "Lê Đức Thọ", "Hùng Vương", "Thái Hà", "Chùa Bộc", "Nguyễn Văn Cừ",
    "Bình Thuận", "Hai Bà Trưng", "Hòa Bình", "Nguyễn Thái Bình",
)

TRAIN_NAMES = ("Anh Minh", "Chị Lan", "Anh Hùng", "Chị Hương", "Bạn Tuấn", "Cô Mai", "Anh Dũng")
HOLDOUT_NAMES = ("Chú Hải", "Chị Thảo", "Anh Khoa", "Bạn Vy", "Em Nam", "Cô Nga")

TRAIN_NOTES = (
    "giao giờ hành chính", "gọi trước khi giao", "để ở bảo vệ giúp em",
    "nhận buổi sáng", "ship ngoài giờ", "call trước 10 phút",
)
HOLDOUT_NOTES = (
    "giao tận tay giúp mình", "tới cổng alo nhé", "nhận chiều nay",
    "gửi lễ tân tòa nhà", "nhắn zalo khi tới",
)

TRAIN_CHAT_FOOD = (
    "cho em 1 ly trà sữa trân châu đường đen ít đá",
    "2 phần cơm tấm sườn bì chả giao gấp",
    "1 bánh mì thịt nướng không ớt",
    "áo thun nam cotton size L màu đen",
    "pass wifi là gì vậy shop ơi",
    "cảm ơn shop nhiều lắm nhé",
    "shipper tới cổng gọi 0901234567 giúp em",
    "kiểm tra đơn hàng giúp chị với ạ",
    "1 trà đào cam sả size lớn ít ngọt",
    "váy hoa nhí size M còn hàng không shop",
    "bao nhiêu tiền 1 hộp bánh trung thu ạ",
    "chốt đơn 2 chai nước ép ổi không đường",
)
HOLDOUT_CHAT_FOOD = (
    "cho mình 1 tô bún bò huế đặc biệt không hành",
    "anh ơi áo khoác dù nam còn size XL không ạ",
    "pass cửa chung cư là bao nhiêu vậy bạn",
    "giao giúp chị 3 ly cà phê sữa đá ít sữa",
    "đặt 1 pizza hải sản cỡ vừa thêm phô mai",
    "shop ơi đơn này freeship không ạ",
    "2 suất bún đậu mắm tôm đầy đủ mắm pha riêng",
    "em muốn đổi size S sang size M được không",
    "cảm ơn anh shipper nhiệt tình quá ạ",
    "1 hộp sushi cá hồi tươi giao trước 6h tối",
)


def _to_telex_word(word: str) -> str:
    """Synthesize realistic messy Telex typing artifacts from a Vietnamese word."""
    rep = {
        "ă": "aw", "ắ": "aws", "ằ": "awf", "ẳ": "awr", "ẵ": "awx", "ặ": "awj",
        "â": "aa", "ấ": "aas", "ầ": "aaf", "ẩ": "aar", "ẫ": "aax", "ậ": "aaj",
        "ê": "ee", "ế": "ees", "ề": "eef", "ể": "eer", "ễ": "eex", "ệ": "eej",
        "ô": "oo", "ố": "oos", "ồ": "oof", "ổ": "oor", "ỗ": "oox", "ộ": "ooj",
        "ơ": "ow", "ớ": "ows", "ờ": "owf", "ở": "owr", "ỡ": "owx", "ợ": "owj",
        "ư": "uw", "ứ": "uws", "ừ": "uwf", "ử": "uwr", "ữ": "uwx", "ự": "uwj",
        "đ": "dd",
        "á": "as", "à": "af", "ả": "ar", "ã": "ax", "ạ": "aj",
        "é": "es", "è": "ef", "ẻ": "er", "ẽ": "ex", "ẹ": "ej",
        "í": "is", "ì": "if", "ỉ": "ir", "ĩ": "ix", "ị": "ij",
        "ó": "os", "ò": "of", "ỏ": "or", "õ": "ox", "ọ": "oj",
        "ú": "us", "ù": "uf", "ủ": "ur", "ũ": "ux", "ụ": "uj",
        "ý": "ys", "ỳ": "yf", "ỷ": "yr", "ỹ": "yx", "ỵ": "yj",
    }
    tones = ""
    out = []
    for ch in word.lower():
        if ch in rep:
            v = rep[ch]
            if v[-1] in ("s", "f", "r", "x", "j") and len(v) >= 2 and v != "dd":
                out.append(v[:-1])
                tones = v[-1]
            else:
                out.append(v)
        else:
            out.append(ch)
    return "".join(out) + tones


def _to_vni_word(word: str) -> str:
    """Synthesize realistic VNI typing artifacts from a Vietnamese word."""
    rep = {
        "ă": "a8", "ắ": "a81", "ằ": "a82", "ẳ": "a83", "ẵ": "a84", "ặ": "a85",
        "â": "a6", "ấ": "a61", "ầ": "a62", "ẩ": "a63", "ẫ": "a64", "ậ": "a65",
        "ê": "e6", "ế": "e61", "ề": "e62", "ể": "e63", "ễ": "e64", "ệ": "e65",
        "ô": "o6", "ố": "o61", "ồ": "o62", "ổ": "o63", "ỗ": "o64", "ộ": "o65",
        "ơ": "o7", "ớ": "o71", "ờ": "o72", "ở": "o73", "ỡ": "o74", "ợ": "o75",
        "ư": "u7", "ứ": "u71", "ừ": "u72", "ử": "u73", "ữ": "u74", "ự": "u75",
        "đ": "d9",
        "á": "a1", "à": "a2", "ả": "a3", "ã": "a4", "ạ": "a5",
        "é": "e1", "è": "e2", "ẻ": "e3", "ẽ": "e4", "ẹ": "e5",
        "í": "i1", "ì": "i2", "ỉ": "i3", "ĩ": "i4", "ị": "i5",
        "ó": "o1", "ò": "o2", "ỏ": "o3", "õ": "o4", "ọ": "o5",
        "ú": "u1", "ù": "u2", "ủ": "u3", "ũ": "u4", "ụ": "u5",
        "ý": "y1", "ỳ": "y2", "ỷ": "y3", "ỹ": "y4", "ỵ": "y5",
    }
    tone = ""
    out = []
    for ch in word.lower():
        if ch in rep:
            v = rep[ch]
            if v[-1] in "12345":
                out.append(v[:-1])
                tone = v[-1]
            else:
                out.append(v)
        else:
            out.append(ch)
    return "".join(out) + tone


def _apply_telex_or_vni(text: str, rng: random.Random) -> str:
    tokens = text.split(" ")
    use_vni = rng.random() < 0.35
    out = []
    for tok in tokens:
        if any(ord(c) > 127 for c in tok) and rng.random() < 0.75:
            out.append(_to_vni_word(tok) if use_vni else _to_telex_word(tok))
        else:
            out.append(tok)
    return " ".join(out)


def _jitter_lat_lng(
    lat: float | None,
    lng: float | None,
    min_km: float,
    max_km: float,
    rng: random.Random,
) -> tuple[float | None, float | None]:
    if lat is None or lng is None:
        return None, None
    dist = rng.uniform(min_km, max_km)
    bearing = rng.uniform(0.0, 2.0 * math.pi)
    dlat = (dist * math.cos(bearing)) / 111.32
    cos_lat = max(0.2, math.cos(math.radians(lat)))
    dlng = (dist * math.sin(bearing)) / (111.32 * cos_lat)
    return round(lat + dlat, 6), round(lng + dlng, 6)


def _abbreviate_prov(prov_bare: str, prov_full: str, rng: random.Random) -> str:
    low = strip_accents(prov_bare)
    if low == "ho chi minh":
        return rng.choice(["TP.HCM", "TPHCM", "HCM", "TP HCM", "Sài Gòn", "TP. Hồ Chí Minh"])
    if low == "ha noi":
        return rng.choice(["Hà Nội", "HN", "TP.HN", "TP Hà Nội"])
    if low == "da nang":
        return rng.choice(["Đà Nẵng", "TP. Đà Nẵng", "ĐN"])
    if low == "ba ria - vung tau":
        return rng.choice(["Bà Rịa - Vũng Tàu", "BR-VT", "BRVT", "Tỉnh Bà Rịa - Vũng Tàu"])
    return rng.choice([prov_bare, prov_full])


def _abbreviate_dist(dist_bare: str, dist_full: str, rng: random.Random) -> str:
    clean_d = dist_bare.replace(" (Huyện)", "").replace(" (Thành phố)", "").replace(" (Thị xã)", "")
    low = strip_accents(clean_d)
    if low.startswith("quan ") and low[5:].isdigit():
        num = low[5:]
        return rng.choice([f"Q.{num}", f"Q{num}", f"Q {num}", f"Quận {num}"])
    if dist_full.startswith("Quận "):
        return rng.choice([f"Q.{clean_d}", f"Q. {clean_d}", dist_full, clean_d])
    if dist_full.startswith("Huyện "):
        return rng.choice([f"H.{clean_d}", f"H. {clean_d}", dist_full, clean_d])
    if dist_full.startswith("Thị xã "):
        return rng.choice([f"TX.{clean_d}", f"TX {clean_d}", dist_full, clean_d])
    if dist_full.startswith("Thành phố "):
        return rng.choice([f"TP.{clean_d}", f"TP {clean_d}", dist_full, clean_d])
    return clean_d


def _abbreviate_ward(ward_bare: str, ward_full: str, rng: random.Random) -> str:
    low = strip_accents(ward_bare)
    if low.startswith("phuong ") and low[7:].isdigit():
        num = low[7:]
        return rng.choice([f"P.{num}", f"P{num}", f"P {num}", f"Phường {num}"])
    if ward_full.startswith("Phường "):
        return rng.choice([f"P.{ward_bare}", f"P. {ward_bare}", ward_full, ward_bare])
    if ward_full.startswith("Xã "):
        return rng.choice([f"X.{ward_bare}", f"Xã {ward_bare}", ward_bare])
    if ward_full.startswith("Thị trấn "):
        return rng.choice([f"TT.{ward_bare}", f"TT {ward_bare}", ward_full, ward_bare])
    return ward_bare


def _norm_pair_key(raw: str, gw: str, gd: str, gp: str) -> str:
    return "|".join(
        (
            normalize_vn_ascii(raw),
            normalize_vn_ascii(gw),
            normalize_vn_ascii(gd),
            normalize_vn_ascii(gp),
        )
    )


class PairSynthesizer:
    """Synthesize diverse VN address / Goong fill pairs for training or independent holdout."""

    def __init__(self, gz: GazetteerV3, records: list[dict[str, Any]], is_holdout: bool = False) -> None:
        self.gz = gz
        self.records = records
        self.is_holdout = is_holdout
        self.streets = HOLDOUT_STREETS if is_holdout else TRAIN_STREETS
        self.names = HOLDOUT_NAMES if is_holdout else TRAIN_NAMES
        self.notes = HOLDOUT_NOTES if is_holdout else TRAIN_NOTES
        self.chat_food = HOLDOUT_CHAT_FOOD if is_holdout else TRAIN_CHAT_FOOD

        # Precompute helper pools
        self.valid_3tier_recs = [r for r in records if r["ward_bare"] and r["district_bare"] and r["province_bare"]]
        self.unique_pair_recs = []
        self.unique_ward_only_recs = []
        self.low_ambig_ward_recs = []
        self.ambig_ward_recs = []
        self.same_ward_diff_recs: dict[str, list[dict[str, Any]]] = {}
        self.legacy_to_new_diff_recs = []
        for r in self.valid_3tier_recs:
            w_key = strip_accents(r["ward_bare"])
            self.same_ward_diff_recs.setdefault(w_key, []).append(r)
            wo_idx = gz.wo_to_idx.get(w_key)
            d_key = strip_accents(r["district_bare"].replace(" (Huyện)", "").replace(" (Thành phố)", "").replace(" (Thị xã)", ""))
            did_o = gz.do_to_idx.get(d_key)
            if wo_idx is not None and did_o is not None:
                provs_for_wd = gz.wo_do_to_provs.get((wo_idx, did_o), set())
                if len(provs_for_wd) == 1:
                    self.unique_pair_recs.append(r)
                n_wo_provs = len(gz.wo_to_provs[wo_idx])
                if n_wo_provs == 1:
                    self.unique_ward_only_recs.append(r)
                elif n_wo_provs == 2:
                    self.low_ambig_ward_recs.append(r)
                if n_wo_provs >= 3:
                    self.ambig_ward_recs.append(r)
            if strip_accents(r["ward_bare"]) != strip_accents(r["new_ward_bare"]) or strip_accents(r["province_bare"]) != strip_accents(r["new_province_bare"]):
                self.legacy_to_new_diff_recs.append(r)

        self.colliding_ward_recs = [
            x for x in self.valid_3tier_recs
            if strip_accents(x["ward_bare"]) in gz.po_set or strip_accents(x["ward_bare"]) in gz.do_set
        ]

        # Unique 2025 (wn, pn) records
        self.new_2tier_recs = []
        seen_new = set()
        for r in records:
            k = (r["new_ward_full"], r["new_province_full"])
            if r["new_ward_bare"] and k not in seen_new:
                seen_new.add(k)
                self.new_2tier_recs.append(r)

        # Province-colliding wards/districts for street trap generation
        self.colliding_admin_names = sorted(
            (set(gz.po_disp) | set(gz.do_disp)) - {"Quận 1", "Quận 3", "Quận 7"}
        )

    def _pick_coords(self, lat: float | None, lng: float | None, rng: random.Random, outlier: bool = False) -> tuple[float | None, float | None]:
        if lat is None or lng is None:
            return None, None
        if outlier:
            return _jitter_lat_lng(lat, lng, 50.0, 290.0, rng)
        if rng.random() < 0.22:
            return None, None
        return _jitter_lat_lng(lat, lng, 0.05, 5.8, rng)

    def _pick_score(self, rng: random.Random, is_good: bool) -> float | None:
        if rng.random() < 0.12:
            return None
        if is_good:
            return round(rng.uniform(0.68, 0.99), 3)
        return round(rng.uniform(0.25, 0.92), 3)

    def _gen_positive(self, rng: random.Random) -> tuple[str, str, str, str, float | None, float | None, float | None, int, str]:
        """Generate a valid clean match (label=0)."""
        p = rng.random()
        r = rng.choice(self.valid_3tier_recs)
        w_full, w_bare = r["ward_full"], r["ward_bare"]
        d_full = r["district_full"]
        d_bare = r["district_bare"].replace(" (Huyện)", "").replace(" (Thành phố)", "").replace(" (Thị xã)", "")
        p_full, p_bare = r["province_full"], r["province_bare"]
        nw_full, nw_bare = r["new_ward_full"], r["new_ward_bare"]
        np_full, np_bare = r["new_province_full"], r["new_province_bare"]

        house = (
            f"{rng.randint(1, 450)}/{rng.randint(1, 30)}"
            if self.is_holdout and rng.random() < 0.4
            else f"{rng.randint(1, 350)}"
        )
        if self.is_holdout and rng.random() < 0.25:
            house = f"Số nhà {house}"
        street = rng.choice(self.streets)
        sep = rng.choice([", ", " - ", "; "]) if self.is_holdout else ", "

        if p < 0.24:
            # 1) Full 3-tier legacy address
            if rng.random() < 0.55:
                raw = f"{house} {street}{sep}{w_full}{sep}{d_full}{sep}{p_full}"
            else:
                raw = f"{w_bare}{sep}{d_bare}{sep}{p_bare}"
            lat, lng = self._pick_coords(r["old_lat"], r["old_lon"], rng, outlier=False)
            return raw, w_full, d_full, p_full, self._pick_score(rng, True), lat, lng, 0, "pos_3tier"

        if p < 0.52:
            # 2) 6 short valid styles:
            style = rng.randrange(6)
            ru = rng.choice(self.unique_pair_recs)
            rw_f, rw_b = ru["ward_full"], ru["ward_bare"]
            rd_f = ru["district_full"]
            rd_b = ru["district_bare"].replace(" (Huyện)", "").replace(" (Thành phố)", "").replace(" (Thị xã)", "")
            rp_f, rp_b = ru["province_full"], ru["province_bare"]
            lat, lng = self._pick_coords(ru["old_lat"], ru["old_lon"], rng, outlier=False)

            if style == 0:
                # "Lộc Thạnh, Lộc Ninh" style (ward + dist uniquely determining prov)
                raw = f"{rw_b}, {rd_b}" if rng.random() < 0.7 else f"{rw_f}, {rd_b}"
                return raw, rw_f, rd_f, rp_f, self._pick_score(rng, True), lat, lng, 0, "pos_short_wd"
            if style == 1:
                # No-comma style
                raw = f"{rw_b} {rd_b} {rp_b}" if rng.random() < 0.65 else f"{rw_b} {rd_b}"
                if rng.random() < 0.45:
                    raw = strip_accents(raw)
                return raw, rw_f, rd_f, rp_f, self._pick_score(rng, True), lat, lng, 0, "pos_no_comma"
            if style == 2:
                # Unaccented style
                raw = strip_accents(f"{rw_f}, {rd_b}, {rp_b}" if rng.random() < 0.6 else f"{rw_b}, {rd_b}")
                return raw, rw_f, rd_f, rp_f, self._pick_score(rng, True), lat, lng, 0, "pos_unaccented"
            if style == 3:
                # Abbreviated style
                w_ab = _abbreviate_ward(rw_b, rw_f, rng)
                d_ab = _abbreviate_dist(rd_b, rd_f, rng)
                p_ab = _abbreviate_prov(rp_b, rp_f, rng)
                raw = f"{w_ab}, {d_ab}, {p_ab}" if rng.random() < 0.7 else f"{w_ab}, {d_ab}"
                return raw, rw_f, rd_f, rp_f, self._pick_score(rng, True), lat, lng, 0, "pos_abbrev"
            if style == 4:
                # 3-tier with house & street
                raw = f"{house} Đường {street}, {_abbreviate_ward(rw_b, rw_f, rng)}, {_abbreviate_dist(rd_b, rd_f, rng)}, {_abbreviate_prov(rp_b, rp_f, rng)}"
                return raw, rw_f, rd_f, rp_f, self._pick_score(rng, True), lat, lng, 0, "pos_3tier_short"
            # style == 5: 2-tier 2025
            r2 = rng.choice(self.new_2tier_recs)
            nw_f2, nw_b2 = r2["new_ward_full"], r2["new_ward_bare"]
            np_f2, np_b2 = r2["new_province_full"], r2["new_province_bare"]
            raw = f"{nw_f2}, {np_b2}" if rng.random() < 0.6 else f"{nw_b2}, {_abbreviate_prov(np_b2, np_f2, rng)}"
            lat2, lng2 = self._pick_coords(r2["new_lat"], r2["new_lon"], rng, outlier=False)
            return raw, nw_f2, "", np_f2, self._pick_score(rng, True), lat2, lng2, 0, "pos_2tier_2025"

        if p < 0.63:
            # 3) Telex / VNI typos
            ru = rng.choice(self.unique_pair_recs)
            rw_b = ru["ward_bare"]
            rd_b = ru["district_bare"].replace(" (Huyện)", "").replace(" (Thành phố)", "").replace(" (Thị xã)", "")
            rp_b = ru["province_bare"]
            base = f"{ru['ward_full']}, {rd_b}, {rp_b}" if rng.random() < 0.6 else f"{rw_b}, {rd_b}"
            raw = _apply_telex_or_vni(base, rng)
            lat, lng = self._pick_coords(ru["old_lat"], ru["old_lon"], rng, outlier=False)
            return raw, ru["ward_full"], ru["district_full"], ru["province_full"], self._pick_score(rng, True), lat, lng, 0, "pos_telex_vni"

        if p < 0.73:
            # 4) Customer names / phones / delivery notes wrapped around valid address
            name = rng.choice(self.names)
            phone = f"09{rng.randint(10000000, 99999999)}"
            if self.is_holdout:
                phone = f"+84{phone[1:]}"
            note = rng.choice(self.notes)
            addr = f"{house} {street}, {w_full}, {d_full}, {_abbreviate_prov(p_bare, p_full, rng)}"
            if self.is_holdout:
                raw = rng.choice([
                    f"Đ/c: {addr} - {name} {phone} ({note})",
                    f"Ship đến {addr} | {phone} - {note}",
                ])
            else:
                raw = rng.choice([
                    f"{name} {phone} - {addr} ({note})",
                    f"{addr}, {name} {phone}, {note}",
                ])
            lat, lng = self._pick_coords(r["old_lat"], r["old_lon"], rng, outlier=False)
            return raw, w_full, d_full, p_full, self._pick_score(rng, True), lat, lng, 0, "pos_customer_note"

        if p < 0.89:
            # 5) Legacy 3-tier raw + Goong V2 2025 2-tier output (or vice-versa)
            rl = rng.choice(self.legacy_to_new_diff_recs)
            rl_db = rl["district_bare"].replace(" (Huyện)", "").replace(" (Thành phố)", "").replace(" (Thị xã)", "")
            if rng.random() < 0.80:
                # Legacy raw -> 2025 Goong V2
                if rng.random() < 0.5:
                    raw = f"{rl['ward_bare']}, {rl_db}, {rl['province_bare']}"
                else:
                    raw = f"{house} {street}, {_abbreviate_ward(rl['ward_bare'], rl['ward_full'], rng)}, {_abbreviate_dist(rl_db, rl['district_full'], rng)}, {_abbreviate_prov(rl['province_bare'], rl['province_full'], rng)}"
                lat, lng = self._pick_coords(rl["new_lat"], rl["new_lon"], rng, outlier=False)
                return raw, rl["new_ward_full"], "", rl["new_province_full"], self._pick_score(rng, True), lat, lng, 0, "pos_legacy_2025_equiv"
            else:
                # Reverse: 2025 raw -> legacy Goong
                raw = f"{house} {street}, {rl['new_ward_full']}, {_abbreviate_prov(rl['new_province_bare'], rl['new_province_full'], rng)}"
                lat, lng = self._pick_coords(rl["old_lat"], rl["old_lon"], rng, outlier=False)
                return raw, rl["ward_full"], rl["district_full"], rl["province_full"], self._pick_score(rng, True), lat, lng, 0, "pos_2025_to_legacy_equiv"

        if p < 0.91 and self.unique_ward_only_recs:
            # 6) Unambiguous unique-ward-only (raw_min_ward_ambig == 1, e.g. "Phường Trúc Bạch")
            ruw = rng.choice(self.unique_ward_only_recs)
            raw = ruw["ward_full"] if rng.random() < 0.7 else f"{house} {street}, {ruw['ward_full']}"
            lat, lng = self._pick_coords(ruw["old_lat"], ruw["old_lon"], rng, outlier=False)
            return raw, ruw["ward_full"], ruw["district_full"], ruw["province_full"], self._pick_score(rng, True), lat, lng, 0, "pos_unique_ward_only"

        if (not self.is_holdout) and p < 0.935 and self.low_ambig_ward_recs:
            # 6b) Borderline 2-province ward-only where Goong guessed the true parent (~50% Bayes uncertainty)
            rl2 = rng.choice(self.low_ambig_ward_recs)
            raw = rl2["ward_full"]
            lat, lng = self._pick_coords(rl2["old_lat"], rl2["old_lon"], rng, outlier=False)
            return raw, rl2["ward_full"], rl2["district_full"], rl2["province_full"], round(rng.uniform(0.42, 0.65), 3), lat, lng, 0, "pos_borderline_2prov"

        if (not self.is_holdout) and p < 0.95:
            # 6c) Minor 1-char ward typo (e.g. 'Phuc X' -> 'Phuc Xa') where Goong resolved right ward
            w_chars = list(w_bare)
            if len(w_chars) >= 6:
                w_typo = "".join(w_chars[:-1])
                raw = f"{w_typo}, {d_bare}, {p_bare}"
                lat, lng = self._pick_coords(r["old_lat"], r["old_lon"], rng, outlier=False)
                return raw, w_full, d_full, p_full, round(rng.uniform(0.45, 0.68), 3), lat, lng, 0, "pos_borderline_typo"

        # 7) Street / ward name collision traps
        trap_type = rng.random()
        if trap_type < 0.45:
            # Classic street name colliding with a province or district name before explicit ward+dist
            trap_street = rng.choice(self.colliding_admin_names)
            pfx = rng.choice(["", "Đường "])
            raw = f"{house} {pfx}{trap_street}, {_abbreviate_ward(w_bare, w_full, rng)}, {_abbreviate_dist(d_bare, d_full, rng)}, {_abbreviate_prov(p_bare, p_full, rng)}"
            lat, lng = self._pick_coords(r["old_lat"], r["old_lon"], rng, outlier=False)
            return raw, w_full, d_full, p_full, self._pick_score(rng, True), lat, lng, 0, "pos_street_trap"
        elif trap_type < 0.75:
            # House number + ward whose name collides with a province/district (like "45 Bình Thuận, Quận 7, TP.HCM")
            rc = rng.choice(self.colliding_ward_recs)
            raw = f"{house} {rc['ward_bare']}, {rc['district_full']}, {_abbreviate_prov(rc['province_bare'], rc['province_full'], rng)}"
            lat, lng = self._pick_coords(rc["old_lat"], rc["old_lon"], rng, outlier=False)
            return raw, rc["ward_full"], rc["district_full"], rc["province_full"], self._pick_score(rng, True), lat, lng, 0, "pos_ward_prov_trap"
        else:
            # Pre-2021 / proper-substring street trap like "123 Hai Bà Trưng, P.6, Q3"
            num_w = rng.choice(["Phường 1", "Phường 2", "Phường 3", "Phường 4", "Phường 5", "Phường 6", "Phường 9"])
            num_d = rng.choice(["Quận 1", "Quận 3", "Quận 5", "Quận 10"])
            trap_street = rng.choice(["Hai Bà Trưng", "Nguyễn Thái Bình", "Hòa Bình", "Bình Thuận", "Điện Biên Phủ"])
            w_tok = num_w.replace("Phường ", "P.") if rng.random() < 0.5 else num_w
            d_tok = num_d.replace("Quận ", "Q") if rng.random() < 0.5 else num_d
            raw = f"{house} {trap_street}, {w_tok}, {d_tok}"
            lat, lng = self._pick_coords(10.780, 106.690, rng, outlier=False)
            return raw, num_w, num_d, "Thành phố Hồ Chí Minh", self._pick_score(rng, True), lat, lng, 0, "pos_substring_trap"

    def _gen_negative(self, rng: random.Random) -> tuple[str, str, str, str, float | None, float | None, float | None, int, str]:
        """Generate a bad / hallucinated / risky fill (label=1)."""
        p = rng.random()

        if p < 0.24:
            # 1) Ambiguous ward-only where Goong hallucinates ungrounded district & province
            use_low_ambig = (not self.is_holdout) and bool(self.low_ambig_ward_recs) and (rng.random() < 0.14)
            ra = rng.choice(self.low_ambig_ward_recs if use_low_ambig else self.ambig_ward_recs)
            w_key = strip_accents(ra["ward_bare"])
            cands = self.same_ward_diff_recs[w_key]
            if use_low_ambig:
                diff_cands = [x for x in cands if x["province_bare"] != ra["province_bare"]]
                goong_r = rng.choice(diff_cands if diff_cands else cands)
                raw = ra["ward_full"]
                lat, lng = self._pick_coords(goong_r["old_lat"], goong_r["old_lon"], rng, outlier=False)
                score = round(rng.uniform(0.35, 0.62), 3)
                return raw, goong_r["ward_full"], goong_r["district_full"], goong_r["province_full"], score, lat, lng, 1, "neg_ambig_ward"
            goong_r = rng.choice(cands)
            cue = rng.choice(["Số 5 gần chợ", f"Xóm {rng.randint(1, 9)}", f"Tổ {rng.randint(1, 15)}", "Gần ngã ba", f"Hẻm {rng.randint(2, 40)}"])
            if self.is_holdout:
                cue = rng.choice([f"Gần trường học tổ {rng.randint(1, 9)}", f"Đầu ngõ {rng.randint(1, 30)}", f"Khu {rng.randint(1, 6)}"])
            raw = rng.choice([f"{cue}, {ra['ward_bare']}", f"{cue}, {ra['ward_full']}", ra["ward_full"]])
            lat, lng = self._pick_coords(goong_r["old_lat"], goong_r["old_lon"], rng, outlier=False)
            return raw, goong_r["ward_full"], goong_r["district_full"], goong_r["province_full"], self._pick_score(rng, False), lat, lng, 1, "neg_ambig_ward"

        if p < 0.44:
            # 2) Province or district contradiction
            ra = rng.choice(self.ambig_ward_recs)
            w_key = strip_accents(ra["ward_bare"])
            cands = self.same_ward_diff_recs[w_key]
            # Pick a different province (and non-merged!)
            ra_pn = self.gz.pm[self.gz.po_to_idx[strip_accents(ra["province_bare"])]]
            diff_cands = [
                x for x in cands
                if strip_accents(x["province_bare"]) != strip_accents(ra["province_bare"])
                and self.gz.pm[self.gz.po_to_idx[strip_accents(x["province_bare"])]] != ra_pn
            ]
            if diff_cands and rng.random() < 0.72:
                goong_r = rng.choice(diff_cands)
                raw = rng.choice([
                    f"{ra['ward_full']}, {_abbreviate_prov(ra['province_bare'], ra['province_full'], rng)}",
                    f"{ra['ward_full']}, {ra['district_bare']}, {ra['province_bare']}",
                ])
                lat, lng = self._pick_coords(goong_r["old_lat"], goong_r["old_lon"], rng, outlier=False)
                return raw, goong_r["ward_full"], goong_r["district_full"], goong_r["province_full"], self._pick_score(rng, False), lat, lng, 1, "neg_prov_contradict"
            # District contradiction within same province or arbitrary wrong district
            other = rng.choice(self.valid_3tier_recs)
            while strip_accents(other["district_bare"]) == strip_accents(ra["district_bare"]):
                other = rng.choice(self.valid_3tier_recs)
            raw = f"{ra['ward_full']}, {ra['district_full']}, {ra['province_bare']}"
            lat, lng = self._pick_coords(other["old_lat"], other["old_lon"], rng, outlier=False)
            return raw, ra["ward_full"], other["district_full"], ra["province_full"], self._pick_score(rng, False), lat, lng, 1, "neg_dist_contradict"

        if p < 0.62:
            # 3) 2+ hallucinated levels (street-only / landmark-only / mismatched ward+dist+prov)
            house = rng.randint(1, 300)
            street = rng.choice(self.streets)
            goong_r = rng.choice(self.valid_3tier_recs)
            raw = rng.choice([
                f"{house} Đường {street}",
                f"Gần chợ {street}",
                f"{house} {street}, đối diện công viên",
                f"Chung cư {street}, tầng {rng.randint(2, 20)}",
            ])
            lat, lng = self._pick_coords(goong_r["old_lat"], goong_r["old_lon"], rng, outlier=False)
            return raw, goong_r["ward_full"], goong_r["district_full"], goong_r["province_full"], self._pick_score(rng, False), lat, lng, 1, "neg_2plus_halluc"

        if p < 0.74:
            # 4) Empty Goong ward
            r = rng.choice(self.valid_3tier_recs)
            rd_b = r["district_bare"].replace(" (Huyện)", "").replace(" (Thành phố)", "").replace(" (Thị xã)", "")
            raw = f"{rng.randint(1, 200)} {rng.choice(self.streets)}, {r['ward_full']}, {rd_b}, {r['province_bare']}"
            lat, lng = self._pick_coords(r["old_lat"], r["old_lon"], rng, outlier=False)
            if rng.random() < 0.5:
                return raw, "", r["district_full"], r["province_full"], self._pick_score(rng, False), lat, lng, 1, "neg_empty_ward"
            return raw, "", "", r["province_full"], self._pick_score(rng, False), lat, lng, 1, "neg_empty_ward"

        if p < 0.88:
            # 5) Spatial centroid outliers (50-300 km), even with completely clean text!
            r = rng.choice(self.valid_3tier_recs)
            rd_b = r["district_bare"].replace(" (Huyện)", "").replace(" (Thành phố)", "").replace(" (Thị xã)", "")
            if rng.random() < 0.35:
                # Outlier on 2025 2-tier
                raw = f"{rng.randint(1, 250)} {rng.choice(self.streets)}, {r['new_ward_full']}, {r['new_province_bare']}"
                lat, lng = self._pick_coords(r["new_lat"], r["new_lon"], rng, outlier=True)
                if lat is None:
                    lat, lng = _jitter_lat_lng(21.028, 105.854, 65.0, 260.0, rng)
                return raw, r["new_ward_full"], "", r["new_province_full"], self._pick_score(rng, True), lat, lng, 1, "neg_spatial_outlier"
            raw = f"{rng.randint(1, 250)} {rng.choice(self.streets)}, {r['ward_full']}, {rd_b}, {r['province_bare']}"
            lat, lng = self._pick_coords(r["old_lat"], r["old_lon"], rng, outlier=True)
            if lat is None:
                lat, lng = _jitter_lat_lng(10.776, 106.700, 65.0, 260.0, rng)
            return raw, r["ward_full"], r["district_full"], r["province_full"], self._pick_score(rng, True), lat, lng, 1, "neg_spatial_outlier"

        # 6) Slipped chat / food / product strings
        chat = rng.choice(self.chat_food)
        if rng.random() < 0.4:
            chat = f"{rng.choice(self.names)}: {chat}"
        goong_r = rng.choice(self.valid_3tier_recs)
        if rng.random() < 0.25:
            return chat, "", "", "", round(rng.uniform(0.10, 0.45), 3), None, None, 1, "neg_chat_food"
        lat, lng = self._pick_coords(goong_r["old_lat"], goong_r["old_lon"], rng, outlier=False)
        return chat, goong_r["ward_full"], goong_r["district_full"], goong_r["province_full"], round(rng.uniform(0.20, 0.65), 3), lat, lng, 1, "neg_chat_food"

    def generate_unique_dataset(
        self,
        total_count: int,
        seed: int,
        pos_ratio: float = 0.55,
        forbidden_keys: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        seen: set[str] = set(forbidden_keys) if forbidden_keys is not None else set()
        target_pos = int(round(total_count * pos_ratio))
        target_neg = total_count - target_pos

        rows: list[dict[str, Any]] = []
        pos_cnt = 0
        neg_cnt = 0
        attempts = 0
        max_attempts = total_count * 25
        while (pos_cnt < target_pos or neg_cnt < target_neg) and attempts < max_attempts:
            attempts += 1
            want_pos = (pos_cnt < target_pos) and (neg_cnt >= target_neg or rng.random() < pos_ratio)
            raw, gw, gd, gp, score, lat, lng, label, kind = (
                self._gen_positive(rng) if want_pos else self._gen_negative(rng)
            )
            key = _norm_pair_key(raw, gw, gd, gp)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "raw_text": raw,
                    "goong_ward": gw,
                    "goong_district": gd,
                    "goong_province": gp,
                    "goong_score": score,
                    "goong_lat": lat,
                    "goong_lng": lng,
                    "label": label,
                    "kind": kind,
                    "pair_key": key,
                }
            )
            if label == 0:
                pos_cnt += 1
            else:
                neg_cnt += 1
        rng.shuffle(rows)
        return rows


class PostGoongVerifierMLP(nn.Module):
    """Compact calibrated binary risk classifier [in_dim -> 128 -> 64 -> 32 -> 1]."""

    def __init__(self, in_dim: int = len(FEATURE_NAMES), dropout: float = 0.15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact ROC-AUC in pure NumPy (handles ties via Mann-Whitney rank average)."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return 1.0
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    n = len(s)
    while i < n:
        j = i + 1
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[i:j] = avg_rank
        i = j
    pos_rank_sum = float(np.sum(ranks[y_sorted == 1]))
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision / PR-AUC in pure NumPy."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return 1.0
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / float(n_pos)
    prev_recall = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - prev_recall) * precision))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error across equal-width probability bins."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(y))
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            acc = float(np.mean(y[mask]))
            conf = float(np.mean(p[mask]))
            ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def fit_temperature_scaling(logits_val: np.ndarray, y_val: np.ndarray) -> float:
    """Fit scalar temperature T > 0 minimizing validation NLL (bounded to avoid logit saturation)."""
    z_t = torch.tensor(logits_val, dtype=torch.float32)
    y_t = torch.tensor(y_val, dtype=torch.float32)
    log_t = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=80)
    bce = nn.BCEWithLogitsLoss()

    def _closure() -> torch.Tensor:
        opt.zero_grad()
        temp = torch.exp(log_t).clamp(min=0.75, max=5.0)
        loss = bce(z_t / temp, y_t)
        loss.backward()
        return loss

    opt.step(_closure)
    temp_val = float(torch.exp(log_t).clamp(min=0.75, max=5.0).item())
    return round(temp_val, 6)


def select_dual_thresholds(p_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float]:
    """Select t_low (< t_low auto_approve) and t_high (>= t_high hard_block)."""
    best_t_low = 0.25
    best_score_low = -1e9
    for cand in np.linspace(0.15, 0.38, 47):
        fn_rate = float(np.mean(p_val[y_val == 1] < cand))
        fp_rate = float(np.mean(p_val[y_val == 0] >= cand))
        if fn_rate <= 0.015:
            score = -fp_rate - 0.5 * fn_rate + 0.01 * float(cand)
        else:
            score = -10.0 * fn_rate - fp_rate
        if score > best_score_low:
            best_score_low = score
            best_t_low = float(cand)

    best_t_high = max(best_t_low + 0.30, 0.65)
    best_score_high = -1e9
    for cand in np.linspace(max(best_t_low + 0.28, 0.58), 0.85, 55):
        blocked = p_val >= cand
        n_blk = int(np.sum(blocked))
        if n_blk == 0:
            continue
        prec = float(np.mean(y_val[blocked] == 1))
        rec = float(np.sum((y_val == 1) & blocked) / max(1, int(np.sum(y_val == 1))))
        if prec >= 0.96:
            score = rec + 0.2 * prec
        else:
            score = prec - 2.0
        if score > best_score_high:
            best_score_high = score
            best_t_high = float(cand)

    return round(best_t_low, 4), round(best_t_high, 4)


def _featurize_rows(gz: GazetteerV3, rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x_list = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
    y_list = np.empty(len(rows), dtype=np.float32)
    for i, r in enumerate(rows):
        b = gz.extract(
            raw_text=r["raw_text"],
            goong_ward=r["goong_ward"],
            goong_district=r["goong_district"],
            goong_province=r["goong_province"],
            goong_score=r["goong_score"],
            goong_lat=r["goong_lat"],
            goong_lng=r["goong_lng"],
        )
        x_list[i] = b.vector
        y_list[i] = float(r["label"])
    return x_list, y_list


def main() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)

    artifacts_dir = DEFAULT_ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    conv_dest = artifacts_dir / "vn_admin_v3_converter.json"
    shutil.copy2(CONVERTER_SRC_PATH, conv_dest)

    gz = GazetteerV3(converter_path=conv_dest)
    with open(RECORDS_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)["records"]

    print("Synthesizing 60,000 train/val/test pairs (seed=301)...")
    t_syn = time.perf_counter()
    synth_main = PairSynthesizer(gz, records, is_holdout=False)
    all_60k = synth_main.generate_unique_dataset(total_count=60_000, seed=TRAIN_SEED, pos_ratio=0.55)
    assert len(all_60k) == 60_000, f"Expected 60,000 rows, got {len(all_60k)}"

    train_rows = all_60k[:48_000]
    val_rows = all_60k[48_000:54_000]
    test_rows = all_60k[54_000:60_000]

    train_keys = {r["pair_key"] for r in train_rows}
    val_keys = {r["pair_key"] for r in val_rows}
    test_keys = {r["pair_key"] for r in test_rows}
    assert len(train_keys & val_keys) == 0
    assert len(train_keys & test_keys) == 0
    assert len(val_keys & test_keys) == 0
    all_60k_keys = train_keys | val_keys | test_keys

    print("Synthesizing independent 600-pair holdout (seed=20261199)...")
    synth_hold = PairSynthesizer(gz, records, is_holdout=True)
    holdout_rows = synth_hold.generate_unique_dataset(
        total_count=600, seed=HOLDOUT_SEED, pos_ratio=0.55, forbidden_keys=all_60k_keys
    )
    assert len(holdout_rows) == 600, f"Expected 600 holdout rows, got {len(holdout_rows)}"
    holdout_keys = {r["pair_key"] for r in holdout_rows}
    assert len(holdout_keys & all_60k_keys) == 0
    print(f"Synthesis done in {time.perf_counter() - t_syn:.2f}s (0 overlap verified).")

    print("Extracting features...")
    t_feat = time.perf_counter()
    x_train_raw, y_train = _featurize_rows(gz, train_rows)
    x_val_raw, y_val = _featurize_rows(gz, val_rows)
    x_test_raw, y_test = _featurize_rows(gz, test_rows)
    x_hold_raw, y_hold = _featurize_rows(gz, holdout_rows)
    print(f"Feature extraction for 60,600 pairs done in {time.perf_counter() - t_feat:.2f}s.")

    # Fit scaler strictly on 48k train split
    mean = x_train_raw.mean(axis=0)
    std = np.maximum(x_train_raw.std(axis=0), 1e-6)
    scaler_payload = {
        "feature_names": list(FEATURE_NAMES),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
    }
    with open(artifacts_dir / "scaler.json", "w", encoding="utf-8") as f:
        json.dump(scaler_payload, f, indent=2)

    def _scale(x: np.ndarray) -> np.ndarray:
        return ((x - mean) / std).astype(np.float32)

    x_train = _scale(x_train_raw)
    x_val = _scale(x_val_raw)
    x_test = _scale(x_test_raw)
    x_hold = _scale(x_hold_raw)

    model = PostGoongVerifierMLP(in_dim=len(FEATURE_NAMES), dropout=0.20)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params <= 150_000, f"Param budget exceeded: {n_params}"
    print(f"PostGoongVerifierMLP params: {n_params:,}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).unsqueeze(1)),
        batch_size=512,
        shuffle=True,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
    epochs = 20
    label_smooth = 0.015
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None
    xv_t = torch.from_numpy(x_val)
    yv_t = torch.from_numpy(y_val).unsqueeze(1)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            yb_smooth = yb * (1.0 - 2.0 * label_smooth) + label_smooth
            loss = criterion(model(xb), yb_smooth)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(xv_t)
            val_loss = float(criterion(val_logits, yv_t).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        if epoch in (1, 5, 10, 15, 20):
            print(f"  Epoch {epoch:02d}/{epochs}: val_loss={val_loss:.5f}")

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    # Calibration & dual threshold tuning on validation set
    with torch.no_grad():
        logits_val = model(xv_t).squeeze(1).numpy()
        logits_test = model(torch.from_numpy(x_test)).squeeze(1).numpy()

    temp = fit_temperature_scaling(logits_val, y_val)
    p_val = 1.0 / (1.0 + np.exp(-logits_val / temp))
    t_low, t_high = select_dual_thresholds(p_val, y_val)
    calib_payload = {
        "version": "3.0.0",
        "temperature": temp,
        "t_low": t_low,
        "t_high": t_high,
        "feature_names": list(FEATURE_NAMES),
        "n_params": n_params,
    }
    with open(artifacts_dir / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(calib_payload, f, indent=2)
    print(f"Calibrated T={temp:.4f}, t_low={t_low:.4f}, t_high={t_high:.4f}")

    # Export single-file ONNX opset 14
    onnx_path = artifacts_dir / "post_goong_verifier_v3.onnx"
    dummy_in = torch.zeros(1, len(FEATURE_NAMES), dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy_in,),
        str(onnx_path),
        opset_version=14,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    onnx_size_kb = onnx_path.stat().st_size / 1024.0
    opset_ver = onnx_model.opset_import[0].version
    assert opset_ver == 14, f"Expected opset 14, got {opset_ver}"
    assert onnx_size_kb <= 500.0, f"ONNX too large: {onnx_size_kb:.1f} KB"

    # Parity check ONNX Runtime vs PyTorch
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    ort_test_logits = sess.run(["logits"], {"features": x_test})[0].reshape(-1)
    max_diff = float(np.max(np.abs(ort_test_logits - logits_test)))
    assert max_diff < 1e-4, f"ONNX parity mismatch: {max_diff}"

    # Evaluate on Test (6k) and Independent Holdout (600) using ONNX Runtime
    def _eval_split(x_scaled: np.ndarray, y_arr: np.ndarray, raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
        logits = sess.run(["logits"], {"features": x_scaled})[0].reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-logits / temp))
        preds_flagged = probs >= t_low
        preds_blocked = probs >= t_high

        hall_idx = [i for i, r in enumerate(raw_rows) if r["kind"] in ("neg_ambig_ward", "neg_2plus_halluc", "neg_empty_ward")]
        cont_idx = [i for i, r in enumerate(raw_rows) if r["kind"] in ("neg_prov_contradict", "neg_dist_contradict")]
        spat_idx = [i for i, r in enumerate(raw_rows) if r["kind"] == "neg_spatial_outlier"]
        chat_idx = [i for i, r in enumerate(raw_rows) if r["kind"] == "neg_chat_food"]
        leg_idx = [i for i, r in enumerate(raw_rows) if r["kind"] in ("pos_legacy_2025_equiv", "pos_2025_to_legacy_equiv")]

        hard_blk_prec = (
            float(np.mean(y_arr[preds_blocked] == 1)) if np.any(preds_blocked) else 1.0
        )
        return {
            "count": int(len(y_arr)),
            "roc_auc": round(compute_roc_auc(y_arr, probs), 6),
            "pr_auc": round(compute_pr_auc(y_arr, probs), 6),
            "ece": round(compute_ece(y_arr, probs), 6),
            "accuracy_at_t_low": round(float(np.mean((probs >= t_low) == (y_arr >= 0.5))), 6),
            "hard_block_precision": round(hard_blk_prec, 6),
            "hallucination_recall": round(float(np.mean(preds_flagged[hall_idx])), 6) if hall_idx else None,
            "contradiction_recall": round(float(np.mean(preds_flagged[cont_idx])), 6) if cont_idx else None,
            "spatial_outlier_recall": round(float(np.mean(preds_flagged[spat_idx])), 6) if spat_idx else None,
            "chat_food_recall": round(float(np.mean(preds_flagged[chat_idx])), 6) if chat_idx else None,
            "legacy_2025_false_alarm_rate": round(float(np.mean(preds_flagged[leg_idx])), 6) if leg_idx else None,
            "clean_false_alarm_rate": round(float(np.mean(preds_flagged[y_arr == 0])), 6),
        }

    test_metrics = _eval_split(x_test, y_test, test_rows)
    holdout_metrics = _eval_split(x_hold, y_hold, holdout_rows)

    # Measure end-to-end single-item runtime CPU latency in a clean zero-PyTorch subprocess
    bench_payload = [
        {
            "raw_text": r["raw_text"],
            "goong_ward": r["goong_ward"],
            "goong_district": r["goong_district"],
            "goong_province": r["goong_province"],
            "goong_score": r["goong_score"],
            "goong_lat": r["goong_lat"],
            "goong_lng": r["goong_lng"],
        }
        for r in holdout_rows
    ]
    bench_script = (
        "import json, sys, time, numpy as np\n"
        f"sys.path.insert(0, {str(PKG_DIR)!r})\n"
        "from verifier import PostGoongVerifier\n"
        "rows = json.load(sys.stdin)\n"
        f"v = PostGoongVerifier(artifacts_dir={str(artifacts_dir)!r})\n"
        "for r in rows[:25]:\n"
        "    v.verify(**r)\n"
        "lats = []\n"
        "for r in rows:\n"
        "    t0 = time.perf_counter()\n"
        "    v.verify(**r)\n"
        "    lats.append((time.perf_counter() - t0) * 1000.0)\n"
        "arr = np.asarray(lats, dtype=np.float64)\n"
        "print(json.dumps({'p50': round(float(np.percentile(arr, 50)), 4), "
        "'p95': round(float(np.percentile(arr, 95)), 4), "
        "'p99': round(float(np.percentile(arr, 99)), 4), "
        "'mean': round(float(np.mean(arr)), 4)}))\n"
    )
    bench_proc = subprocess.run(
        [sys.executable, "-c", bench_script],
        input=json.dumps(bench_payload),
        text=True,
        capture_output=True,
        check=True,
    )
    cpu_latency_ms = json.loads(bench_proc.stdout.strip().splitlines()[-1])

    report = {
        "model": "PostGoongVerifierMLP",
        "seed_train": TRAIN_SEED,
        "seed_holdout": HOLDOUT_SEED,
        "splits": {"train": 48000, "val": 6000, "test": 6000, "independent_holdout": 600},
        "zero_overlap_verified": True,
        "n_features": len(FEATURE_NAMES),
        "n_params": n_params,
        "onnx_opset": opset_ver,
        "onnx_size_kb": round(onnx_size_kb, 2),
        "onnx_pytorch_max_diff": round(max_diff, 8),
        "calibration": {"temperature": temp, "t_low": t_low, "t_high": t_high},
        "test_6k_metrics": test_metrics,
        "holdout_600_metrics": holdout_metrics,
        "cpu_latency_ms": cpu_latency_ms,
    }
    with open(artifacts_dir / "eval_report_v3.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Ensure zero .pt files anywhere in way_c_post_goong_v3
    for pt_file in PKG_DIR.rglob("*.pt"):
        pt_file.unlink()

    print("\n=== EVAL REPORT (600-pair independent holdout) ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
