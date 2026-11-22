"""Comprehensive offline test suite for `laya_address_vn/ceiling_audit/`."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import openpyxl
import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent.parent
if str(AUDIT_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIT_DIR))

import ceiling_eval
from generate_demo_dev_data import generate_demo_dataset
from post_goong_features import VietnamAdminIndex, extract_feature_dataframe
import sample_300_for_labeling
import schema_io
import score_300_labels
from v2_scorer import score_way_a_v2_batch, score_way_b_v2_batch


def test_schema_io_and_pii_masking(tmp_path: Path) -> None:
    # 1. Phone masking & safe leading-recipient masking (must never corrupt streets!)
    txt1 = "Chị Trần Thị B, 123 Nguyễn Văn Linh, P. Bình Thuận, Q7 (0912.345.678)"
    m1 = schema_io.mask_pii_for_features(txt1, mask_names=True)
    assert "<PHONE>" in m1
    assert "0912" not in m1
    assert m1.startswith("<NAME>, 123 Nguyễn Văn Linh")

    txt2 = "123 Nguyễn Văn Linh, Quận 7, TP.HCM"
    assert schema_io.mask_pii_for_features(txt2, mask_names=True) == txt2

    txt3 = "Đường Cô Giang, Phường Cầu Ông Lãnh, Quận 1"
    assert schema_io.mask_pii_for_features(txt3, mask_names=True) == txt3

    txt4 = "45/2B/10/12 Nguyễn Trãi, Q5, +84987654321"
    m4 = schema_io.mask_pii_for_features(txt4, mask_names=True)
    assert "45/2B/10/12" in m4
    assert "<PHONE>" in m4

    txt5 = "Số CCCD 079095001234 ở 12 Lê Lợi"
    assert "<NUM>" in schema_io.mask_pii_for_preview(txt5)

    # 2. CSV + XLSX loading and VN outcome/gate parsing
    df_sample = pd.DataFrame({
        "ma_don": ["D1", "D2", "D3", "D4"],
        "dia_chi_goc": ["a", "b", "c", "d"],
        "trang_thai": ["Hoàn hàng", "Giao thành công", "1", "Đã giao"],
        "luat_d_chan": [1, 0, "block", "pass"],
        "sdt": ["0912345678", "0987654321", "0901112223", "0904445556"],
    })
    csv_p = tmp_path / "test_in.csv"
    xlsx_p = tmp_path / "test_in.xlsx"
    schema_io.write_csv_utf8_sig(df_sample, csv_p)
    df_sample.to_excel(xlsx_p, index=False, engine="openpyxl")

    for p in (csv_p, xlsx_p):
        loaded = schema_io.load_dataset(p)
        sc = schema_io.resolve_schema(loaded)
        assert sc.order_id == "ma_don"
        assert sc.text == "dia_chi_goc"
        assert sc.outcome == "trang_thai"
        assert sc.rule_d == "luat_d_chan"
        assert sc.rule_d_is_block is True  # Auto-detected from "chan"
        ret = schema_io.parse_is_return(loaded[sc.outcome])
        assert ret.tolist() == [1, 0, 1, 0]
        rd_pass = schema_io.parse_bool_gate(loaded[sc.rule_d], invert=sc.rule_d_is_block)
        assert rd_pass.tolist() == [False, True, False, True]


def test_post_goong_features_gazetteer() -> None:
    index = VietnamAdminIndex()
    df = pd.DataFrame({
        "raw_text": [
            "Lộc Thạnh, Lộc Ninh",
            "P. Bến Nghé, Q1, HCM",
            "Số 5 gần chợ, Tân Lập",
            "Xã Hòa Bình, Long An",
            "45 Bình Thuận, Quận 7, TP.HCM",
        ],
        "goong_ward": [
            "Xã Lộc Thạnh",
            "Phường Bến Nghé",
            "Xã Tân Lập",
            "Xã Hòa Bình",
            "Phường Bình Thuận",
        ],
        "goong_district": [
            "Huyện Lộc Ninh",
            "Quận 1",
            "Huyện Đồng Phú",
            "Huyện Trà Ôn",
            "Quận 7",
        ],
        "goong_province": [
            "Tỉnh Bình Phước",
            "TP. Hồ Chí Minh",
            "Tỉnh Bình Phước",
            "Tỉnh Vĩnh Long",
            "TP. Hồ Chí Minh",
        ],
        "goong_score": [0.95, 0.98, 0.72, 0.65, 0.96],
        "outcome": [0, 0, 1, 1, 0],
    })
    schema = schema_io.resolve_schema(df)
    pre_df, post_df, buckets = extract_feature_dataframe(df, schema, index=index)

    # Row 0: Lộc Thạnh, Lộc Ninh -> Lộc Thạnh / Lộc Ninh / Bình Phước
    assert post_df.loc[0, "ward_exact_in_raw"] == 1.0
    assert post_df.loc[0, "dist_exact_or_alias_in_raw"] == 1.0
    assert post_df.loc[0, "hallucinated_levels_count"] <= 1.0
    assert post_df.loc[0, "goong_hierarchy_valid"] == 1.0
    assert post_df.loc[0, "raw_prov_contradicts_goong"] == 0.0

    # Row 1: P. Bến Nghé, Q1, HCM -> alias grounding
    assert post_df.loc[1, "ward_exact_in_raw"] == 1.0
    assert post_df.loc[1, "dist_exact_or_alias_in_raw"] == 1.0
    assert post_df.loc[1, "prov_exact_or_alias_in_raw"] == 1.0
    assert post_df.loc[1, "admin_token_grounding_ratio"] == 1.0
    assert post_df.loc[1, "raw_prov_contradicts_goong"] == 0.0

    # Row 2: Ambiguous ward + hallucinated parent (Đồng Phú, Bình Phước not in text!)
    assert post_df.loc[2, "ambiguous_ward_ungrounded_parent"] == 1.0
    assert post_df.loc[2, "hallucinated_levels_count"] == 2.0
    assert post_df.loc[2, "goong_ward_prov_count"] >= 10.0
    assert buckets[2] == "goong_ambiguous_ward_hallucinated_parent"

    # Row 3: Province contradiction ("Long An" vs "Vĩnh Long")
    assert post_df.loc[3, "raw_prov_contradicts_goong"] == 1.0
    assert buckets[3] == "goong_prov_or_dist_contradiction"

    # Row 4: Ward named "Bình Thuận" MUST NOT trigger contradiction against Hồ Chí Minh
    assert post_df.loc[4, "raw_prov_contradicts_goong"] == 0.0


def test_v2_live_scorer_smoke() -> None:
    samples = [
        "Lộc Thạnh, Lộc Ninh",
        "Hồng Dương, Thanh Oai",
        "loc thanh loc ninh",
        "x. loc thanh h. loc ninh",
        "Số 10 Nguyễn Trãi, Phường Thượng Đình, Quận Thanh Xuân, Hà Nội",
        "ngõ 12",
        "Thôn 7",
        "Xin giá",
        "Cho mình 3 chai nướng BBQ",
        "73",
        "K",
        "Ok",
    ]
    res_b = score_way_b_v2_batch(samples, mode="hybrid")
    assert len(res_b["pass"]) == len(samples)
    assert res_b["pass"][0] is True  # Lộc Thạnh, Lộc Ninh
    assert res_b["pass"][2] is True  # unaccented no-comma short pair
    assert res_b["pass"][5] is False  # ngõ 12
    assert res_b["pass"][8] is False  # Cho mình 3 chai nướng BBQ

    res_a = score_way_a_v2_batch(samples, mode="hybrid")
    if res_a is not None:
        assert len(res_a["pass"]) == len(samples)
        assert res_a["pass"][0] is True
        assert res_a["pass"][5] is False
        assert res_a["pass"][8] is False


def test_ceiling_eval_end_to_end_and_no_leakage(tmp_path: Path) -> None:
    df_demo, _oracle = generate_demo_dataset(n_rows=2400, seed=20261122)
    demo_csv = tmp_path / "demo_dev_orders.csv"
    schema_io.write_csv_utf8_sig(df_demo, demo_csv)

    # Check feature extraction throughput explicitly (< 5.0s for 2,400 rows)
    sc = schema_io.resolve_schema(df_demo)
    idx = VietnamAdminIndex()
    t0 = time.perf_counter()
    extract_feature_dataframe(df_demo, sc, index=idx)
    feat_sec = time.perf_counter() - t0
    assert feat_sec < 5.0, f"Feature extraction too slow: {feat_sec:.2f}s for 2400 rows"

    # Run Step 1 in GROUP mode
    out_grp = tmp_path / "step1_group"
    rc1 = ceiling_eval.main([
        "--data", str(demo_csv),
        "--output-dir", str(out_grp),
        "--split-mode", "group",
        "--min-caught-returns", "30",
        "--target-ratio", "1.5",
    ])
    assert rc1 == 0

    for fn in ("comparison_sheet.csv", "comparison_sheet.xlsx", "ceiling_report.json", "ceiling_report.md", "oof_predictions.csv"):
        assert (out_grp / fn).exists(), f"Missing output file: {fn}"

    wb = openpyxl.load_workbook(out_grp / "comparison_sheet.xlsx")
    assert set(wb.sheetnames) == {"Summary_Comparison", "Operating_Curves", "Confounder_Audit", "Top_Blocked_Preview"}

    raw_json = (out_grp / "ceiling_report.json").read_text(encoding="utf-8")
    assert "NaN" not in raw_json and "Infinity" not in raw_json
    report = json.loads(raw_json)

    post_curve = report["ceiling_curves"]["post_goong_consistency"]
    pre_struct_curve = report["ceiling_curves"]["pre_goong_struct"]
    assert post_curve["best_ratio_at_min_caught"] is not None
    assert post_curve["best_ratio_at_min_caught"]["ratio_lost_per_caught"] <= 1.5
    assert post_curve["pr_auc"] > pre_struct_curve["pr_auc"]

    # Verify Top_Blocked_Preview masks raw 10-digit phones by default
    ws_prev = wb["Top_Blocked_Preview"]
    preview_blob = "\n".join(str(cell.value or "") for row in ws_prev.iter_rows() for cell in row)
    assert not re.search(r"\b0[89]\d{8}\b", preview_blob)

    # Also test TIME mode end-to-end
    out_time = tmp_path / "step1_time"
    rc2 = ceiling_eval.main([
        "--data", str(demo_csv),
        "--output-dir", str(out_time),
        "--split-mode", "time",
        "--min-caught-returns", "30",
        "--target-ratio", "1.5",
    ])
    assert rc2 == 0
    rep_time = json.loads((out_time / "ceiling_report.json").read_text(encoding="utf-8"))
    assert rep_time["split_mode"] == "time"
    assert rep_time["confounder_audit"]["unseen_phone_audit"]["unseen_phone_n"] > 0


def test_step2_sampling_and_ipw_scoring(tmp_path: Path) -> None:
    df_demo, oracle = generate_demo_dataset(n_rows=2400, seed=20261122)
    demo_csv = tmp_path / "demo_dev_orders.csv"
    schema_io.write_csv_utf8_sig(df_demo, demo_csv)

    out_s1 = tmp_path / "step1"
    ceiling_eval.main([
        "--data", str(demo_csv),
        "--output-dir", str(out_s1),
        "--split-mode", "group",
    ])

    out_s2 = tmp_path / "step2"
    rc = sample_300_for_labeling.main([
        "--data", str(demo_csv),
        "--oof-csv", str(out_s1 / "oof_predictions.csv"),
        "--output-dir", str(out_s2),
        "--n-disagree", "100",
        "--n-return", "100",
        "--n-deliver", "100",
        "--seed", "42",
    ])
    assert rc == 0

    df_blind = schema_io.load_dataset(out_s2 / "blind_label_sheet_300.csv")
    df_key = schema_io.load_dataset(out_s2 / "sample_300_key.csv")
    manifest = json.loads((out_s2 / "sampling_manifest.json").read_text(encoding="utf-8"))

    assert len(df_blind) == 300
    assert len(df_key) == 300

    # Zero leakage into blind sheet
    for forbidden in ("is_return", "stratum", "way_a_v2_pass", "way_b_v2_pass", "oof_post_goong_consistency", "order_id"):
        assert forbidden not in df_blind.columns

    # Exact Horvitz-Thompson IPW invariant
    w_sum = float(pd.to_numeric(df_key["stratum_weight"]).sum())
    assert abs(w_sum - manifest["n_pop_dedup"]) < 1e-5

    # Standalone HTML UI contains 0 external links and valid 300-item JSON
    html = (out_s2 / "label_300_ui.html").read_text(encoding="utf-8")
    assert not re.search(r"<script[^>]+src\s*=\s*['\"]https?://", html, re.IGNORECASE)
    assert not re.search(r"<link[^>]+href\s*=\s*['\"]https?://", html, re.IGNORECASE)
    m = re.search(r"const ITEMS = (\[.*?\]);", html, re.DOTALL)
    assert m is not None
    items_raw = json.loads(m.group(1).replace("<\\/", "</"))
    assert len(items_raw) == 300

    # Simulate human labeling using synthetic oracle
    lbl_df = df_blind.copy()
    oid_map = dict(zip(df_key["sample_id"], df_key["order_id"]))
    lbl_df["du_den_xa"] = [oracle[oid_map[sid]]["du_den_xa"] for sid in lbl_df["sample_id"]]
    lbl_df["goong_dung"] = [oracle[oid_map[sid]]["goong_dung"] for sid in lbl_df["sample_id"]]
    lbl_df["loai_loi"] = [oracle[oid_map[sid]]["loai_loi"] for sid in lbl_df["sample_id"]]
    labeled_csv = out_s2 / "labeled_300.csv"
    schema_io.write_csv_utf8_sig(lbl_df, labeled_csv)

    rc_sc = score_300_labels.main([
        "--labels", str(labeled_csv),
        "--key", str(out_s2 / "sample_300_key.csv"),
        "--ceiling-report", str(out_s1 / "ceiling_report.json"),
        "--output-dir", str(out_s2),
    ])
    assert rc_sc == 0

    assert (out_s2 / "label_300_eval_report.md").exists()
    brief_path = out_s2 / "anonymized_v3_brief.json"
    assert brief_path.exists()
    brief_text = brief_path.read_text(encoding="utf-8")
    brief = json.loads(brief_text)

    # Return decomposition shares sum to 1.0
    shares = brief["ipw_population_decomposition"]["among_returns"]
    total_share = (shares["address_or_goong_caused_share"] or 0.0) + (shares["non_address_baseline_share"] or 0.0) + (shares["unclear_share"] or 0.0)
    assert abs(total_share - 1.0) < 1e-3

    # Strict PII-free check on anonymized_v3_brief.json
    assert not re.search(r"\bDEMO_\d{5}\b", brief_text)
    assert not re.search(r"\b0[89]\d{8}\b", brief_text)
    sample_texts = [str(x) for x in df_blind["raw_text"].tolist() if len(str(x)) >= 10]
    for raw_addr in sample_texts[:50]:
        assert raw_addr not in brief_text
