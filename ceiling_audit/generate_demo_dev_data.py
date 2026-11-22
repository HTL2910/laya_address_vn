"""Synthetic 2,400-row Vietnamese DEV Order Generator (100% MIT Gazetteer, Zero Real PII).

Creates a realistic synthetic benchmark matching real e-commerce / COD order dynamics for
testing `ceiling_eval.py` (Step 1) and `sample_300_for_labeling.py` + `score_300_labels.py` (Step 2).
Includes:
  - ~10% Rule-D-blocked garbage rows (~85% return rate)
  - ~90% Rule-D-passed rows (~16-18% overall return rate), combining:
      * Valid full & short addresses (`Lộc Thạnh, Lộc Ninh`, no-accent, abbreviated) with ~13% non-address returns
      * Repeat-canceller synthetic phones (tests GroupKFold / Time-split anti-leakage)
      * Regional COD bias on 2 provinces (tests Geographic Confounder Warning)
      * Hidden Goong hallucinations (ambiguous ward -> wrong district/province), contradictions,
        empty Goong ward, and slipped chat/product cues (~68-75% return rate)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from schema_io import strip_accents, write_csv_utf8_sig

RECORDS_PATH = Path(__file__).resolve().parent.parent / "shared_data" / "vn_admin_full_records.json"


def _abbreviate_ward(w_bare: str, rng: np.random.Generator) -> str:
    style = int(rng.integers(0, 4))
    if style == 0:
        return w_bare
    if style == 1:
        return f"Xã {w_bare}" if len(w_bare.split()) <= 2 else f"Phường {w_bare}"
    if style == 2:
        return f"x. {w_bare}"
    return strip_accents(w_bare).lower()


def _format_vn_addr(rec: Dict[str, str], rng: np.random.Generator, short_style: bool = False) -> str:
    wb = rec["ward_bare"]
    db = rec["district_bare"]
    pb = rec["province_bare"]
    street_num = f"{int(rng.integers(1, 250))}" if rng.random() > 0.35 else f"{int(rng.integers(1, 90))}/{int(rng.integers(1, 25))}"
    streets = ["Nguyễn Trãi", "Lê Lợi", "Hùng Vương", "Trần Phú", "Nguyễn Văn Linh", "Quốc lộ 1A", "Thôn 3", "Ấp 2"]
    st = streets[int(rng.integers(0, len(streets)))]

    if short_style:
        mode = int(rng.integers(0, 5))
        if mode == 0:
            return f"{wb}, {db}"
        if mode == 1:
            return strip_accents(f"{wb} {db}").lower()
        if mode == 2:
            return f"x. {wb} h. {db}"
        if mode == 3:
            return f"{wb}, {db}, {pb}"
        return f"{street_num} {st}, {wb}, {db}"

    phone = f"09{int(rng.integers(10000000, 99999999))}"
    r = float(rng.random())
    if r < 0.12:
        return f"Chị Hoa - {street_num} {st}, {rec['ward_full']}, {rec['district_full']}, {rec['province_full']} (gọi trước {phone})"
    if r < 0.25:
        return strip_accents(f"{street_num} {st}, {wb}, {db}, {pb}").lower()
    if r < 0.35:
        return f"{street_num} {st}, {wb}, {db}, {pb} - giao giờ hành chính"
    return f"{street_num} {st}, {rec['ward_full']}, {rec['district_full']}, {rec['province_full']}"


def generate_demo_dataset(
    n_rows: int = 2400,
    seed: int = 20261122,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    """Return synthetic (dev_df, oracle_labels_by_order_id)."""
    if not RECORDS_PATH.exists():
        raise FileNotFoundError(f"Missing MIT admin records: {RECORDS_PATH}")
    records: List[Dict[str, str]] = json.loads(RECORDS_PATH.read_text(encoding="utf-8"))["legacy_3tier"]

    rng = np.random.default_rng(seed)

    # Index records by ward_bare to pick genuinely ambiguous wards (>= 3 provinces) for Goong hallucinations
    by_ward_bare: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    by_prov_bare: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for rec in records:
        k = strip_accents(rec["ward_bare"]).lower().strip()
        pk = strip_accents(rec["province_bare"]).lower().strip()
        by_ward_bare[k].append(rec)
        by_prov_bare[pk].append(rec)

    ambiguous_wards = [
        w for w, lst in by_ward_bare.items()
        if len({strip_accents(x["province_bare"]).lower() for x in lst}) >= 3
        and not w.isdigit()
        and not w.startswith("phuong ")
    ]
    all_provs = list(by_prov_bare.keys())
    biased_provs = set(all_provs[:2])  # Inject slight regional COD return bias on 2 provinces

    # Pool of 350 normal phones + 22 repeat-canceller synthetic phones
    normal_phones = [f"09{int(rng.integers(10000000, 99999999))}" for _ in range(900)]
    repeat_bad_phones = [f"08{int(rng.integers(10000000, 99999999))}" for _ in range(22)]

    start_ts = datetime(2026, 9, 1, 8, 0, 0)
    rows: List[Dict[str, object]] = []
    oracle: Dict[str, Dict[str, str]] = {}

    garbage_samples = [
        "73", "Xin giá shop ơi", "Cho mình 3 chai nướng BBQ", "ngõ 12", "Thôn 7",
        "Hộp 2kg mực Cà Mau loại 1", "Ok shop gửi sớm nhé", "Ship về chỗ cũ giúp em",
        "Bao nhiêu tiền 1 combo ạ?", "Mình hỏi phí ship Hà Nội",
    ]
    slipped_chat = [
        "Ship nhanh giup minh nha shop", "Cho anh dat 2 hop banh dau xanh hai duong",
        "Gui cho chi nhu don tuan truoc", "Ngoai gio hanh chinh goi truoc a",
        "So 5 gan cho moi, giao buoi chieu", "Banh mi Sai Gon dac biet 5 o",
    ]

    for i in range(n_rows):
        oid = f"DEMO_{i + 1:05d}"
        ts = (start_ts + timedelta(minutes=int(i * 18 + rng.integers(0, 12)))).strftime("%Y-%m-%d %H:%M:%S")
        rec = records[int(rng.integers(0, len(records)))]
        prov_norm = strip_accents(rec["province_bare"]).lower().strip()

        is_repeat_bad = rng.random() < 0.03
        phone = repeat_bad_phones[int(rng.integers(0, len(repeat_bad_phones)))] if is_repeat_bad else normal_phones[int(rng.integers(0, len(normal_phones)))]

        r_type = float(rng.random())

        if r_type < 0.10:
            # Rule D blocked garbage
            raw_text = garbage_samples[int(rng.integers(0, len(garbage_samples)))]
            rule_d_pass = 0
            g_w, g_d, g_p = "", "", rec["province_full"]
            g_score = round(float(rng.uniform(0.10, 0.45)), 3)
            way_a_pass = 0
            way_b_pass = 0
            is_ret = 1 if rng.random() < 0.85 else 0
            oracle[oid] = {"du_den_xa": "not_address", "goong_dung": "empty", "loai_loi": "chat_or_garbage"}

        elif r_type < 0.135:
            # Rule D passed, BUT ambiguous ward-only -> Goong hallucinates wrong district & province!
            rule_d_pass = 1
            w_key = ambiguous_wards[int(rng.integers(0, len(ambiguous_wards)))]
            cands = by_ward_bare[w_key]
            true_rec = cands[0]
            wrong_cands = [c for c in cands if c["province_bare"] != true_rec["province_bare"]]
            wrong_rec = wrong_cands[int(rng.integers(0, len(wrong_cands)))]
            house = f"Số {int(rng.integers(1, 80))} gần chợ, " if rng.random() > 0.4 else ""
            raw_text = f"{house}{true_rec['ward_bare']}"
            g_w, g_d, g_p = wrong_rec["ward_full"], wrong_rec["district_full"], wrong_rec["province_full"]
            g_score = round(float(rng.uniform(0.65, 0.92)), 3)
            way_a_pass = 0 if rng.random() < 0.70 else 1
            way_b_pass = 0 if rng.random() < 0.75 else 1
            is_ret = 1 if rng.random() < 0.72 else 0
            oracle[oid] = {"du_den_xa": "insufficient", "goong_dung": "wrong", "loai_loi": "ambiguous_ward_wrong_prov"}

        elif r_type < 0.155:
            # Rule D passed, province contradicts Goong!
            rule_d_pass = 1
            other_rec = records[int(rng.integers(0, len(records)))]
            while strip_accents(other_rec["province_bare"]).lower() == prov_norm:
                other_rec = records[int(rng.integers(0, len(records)))]
            raw_text = f"{int(rng.integers(1, 99))} Lê Lợi, {rec['ward_bare']}, {other_rec['province_bare']}"
            g_w, g_d, g_p = rec["ward_full"], rec["district_full"], rec["province_full"]
            g_score = round(float(rng.uniform(0.55, 0.85)), 3)
            way_a_pass = 1
            way_b_pass = 1
            is_ret = 1 if rng.random() < 0.74 else 0
            oracle[oid] = {"du_den_xa": "insufficient", "goong_dung": "wrong", "loai_loi": "goong_hallucination"}

        elif r_type < 0.175:
            # Rule D passed, Goong missing ward OR slipped chat/product
            rule_d_pass = 1
            if rng.random() < 0.5:
                raw_text = f"Gần UBND huyện {rec['district_bare']}, {rec['province_bare']}"
                g_w, g_d, g_p = "", rec["district_full"], rec["province_full"]
                o_suf, o_goong, o_err = "insufficient", "empty", "missing_ward"
            else:
                raw_text = slipped_chat[int(rng.integers(0, len(slipped_chat)))]
                g_w, g_d, g_p = rec["ward_full"], rec["district_full"], rec["province_full"]
                o_suf, o_goong, o_err = "not_address", "wrong", "chat_or_garbage"
            g_score = round(float(rng.uniform(0.35, 0.70)), 3)
            way_a_pass = 0 if rng.random() < 0.45 else 1
            way_b_pass = 0 if rng.random() < 0.50 else 1
            is_ret = 1 if rng.random() < 0.68 else 0
            oracle[oid] = {"du_den_xa": o_suf, "goong_dung": o_goong, "loai_loi": o_err}

        else:
            # Valid full or short address + Goong correct!
            rule_d_pass = 1
            is_short = rng.random() < 0.14
            raw_text = _format_vn_addr(rec, rng, short_style=is_short)
            g_w, g_d, g_p = rec["ward_full"], rec["district_full"], rec["province_full"]
            g_score = round(float(rng.uniform(0.84, 0.99)), 3)
            # Simulate some residual v2 false-positives on unaccented/short valid forms (4.8 ratio effect)
            way_a_pass = 0 if (is_short and rng.random() < 0.08) or (rng.random() < 0.025) else 1
            way_b_pass = 0 if (is_short and rng.random() < 0.06) or (rng.random() < 0.020) else 1
            base_ret_p = 0.28 if prov_norm in biased_provs else (0.55 if is_repeat_bad else 0.13)
            is_ret = 1 if rng.random() < base_ret_p else 0
            oracle[oid] = {"du_den_xa": "sufficient", "goong_dung": "correct", "loai_loi": "non_address_return" if is_ret else "none"}

        g_fmt = ", ".join(x for x in (g_w, g_d, g_p, "Việt Nam") if x)
        rows.append({
            "order_id": oid,
            "created_at": ts,
            "customer_phone": phone,
            "raw_text": raw_text,
            "rule_d_pass": rule_d_pass,
            "goong_ward": g_w,
            "goong_district": g_d,
            "goong_province": g_p,
            "goong_score": g_score,
            "goong_formatted": g_fmt,
            "way_a_v2_pass": way_a_pass,
            "way_b_v2_pass": way_b_pass,
            "outcome": "Hoàn hàng" if is_ret else "Giao thành công",
        })

    return pd.DataFrame(rows), oracle


def main() -> int:
    p = argparse.ArgumentParser(description="Generate synthetic 2,400-row DEV order CSV for ceiling_audit demo/tests.")
    p.add_argument("--rows", type=int, default=2400)
    p.add_argument("--seed", type=int, default=20261122)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "demo_dev_orders.csv"))
    args = p.parse_args()

    df, _oracle = generate_demo_dataset(n_rows=args.rows, seed=args.seed)
    out_path = Path(args.out)
    write_csv_utf8_sig(df, out_path)
    print(f"[DONE] Đã sinh {len(df)} đơn hàng mẫu tổng hợp tại: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
