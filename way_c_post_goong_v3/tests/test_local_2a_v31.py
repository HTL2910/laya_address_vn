"""Comprehensive tests for Laya v3.1 local 2A features, pure-stdlib ONNX export, and trainer CLI."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TESTS_DIR = Path(__file__).resolve().parent
PKG_DIR = TESTS_DIR.parent
ROOT_DIR = PKG_DIR.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from features_2a import (  # noqa: E402
    FEATURE_NAMES_1B_24,
    FEATURE_NAMES_2A_51,
    FEATURE_NAMES_2A_V31,
    FeatureExtractor2A,
)
from onnx_export_pure import export_lr_hgb_ensemble_to_onnx  # noqa: E402
from train_local_2a import main as train_local_main  # noqa: E402
from verifier import PostGoongVerifier  # noqa: E402


def test_feature_extractor_2a_schemas_and_values() -> None:
    assert len(FEATURE_NAMES_1B_24) == 24
    assert len(FEATURE_NAMES_2A_51) == 51
    assert len(FEATURE_NAMES_2A_V31) == 56
    assert len(set(FEATURE_NAMES_2A_V31)) == 56

    ext_1b = FeatureExtractor2A(feature_set="v31_1b_24")
    ext_51 = FeatureExtractor2A(feature_set="v31_2a_51")
    ext_56 = FeatureExtractor2A(feature_set="v31_2a_56")

    b1 = ext_1b.extract("45 Lê Lợi, Hiệp Bình Phước, Thủ Đức, TP.HCM")
    b2 = ext_51.extract("45 Lê Lợi, Hiệp Bình Phước, Thủ Đức, TP.HCM", "Phường Hiệp Bình Phước", "Thành phố Thủ Đức", "Thành phố Hồ Chí Minh")
    b3 = ext_56.extract("Phúc Xá, Ba Đình, Hà Nội", "Phường Hồng Hà", "", "Thành phố Hà Nội")
    assert b1.vector.shape == (24,)
    assert b2.vector.shape == (51,)
    assert b3.vector.shape == (56,)
    assert np.all(np.isfinite(b2.vector))
    assert np.all(np.isfinite(b3.vector))

    # Grounded 3-tier hierarchy + zero false contradiction on 'Hiệp Bình Phước' / 'Bình Phước'
    assert b2.features["goong_hierarchy_valid"] == 1.0
    assert b2.features["goong_ward_dist_count"] >= 1.0
    assert b2.features["raw_prov_contradicts_goong"] == 0.0
    assert b2.features["ward_token_overlap"] == 1.0

    # Legacy <-> 2025 equivalence
    assert b3.is_legacy_2025_equiv is True
    assert b3.features["is_legacy_2025_equiv"] == 1.0
    assert b3.features["goong_hierarchy_valid"] == 1.0

    # Contradiction + invalid hierarchy
    b_bad = ext_56.extract("Xã Hòa Bình, Long An", "Xã Hòa Bình", "Huyện Trà Ôn", "Tỉnh Vĩnh Long")
    assert b_bad.features["raw_prov_contradicts_goong"] == 1.0
    assert b_bad.features["goong_hierarchy_valid"] == 1.0  # Goong filled a real Vĩnh Long pair that contradicts raw Long An

    # Hyphenated Ba Ria - Vung Tau stays valid
    b_brvt = ext_56.extract(
        "Phường Phước Trung, Bà Rịa - Vũng Tàu",
        "Phường Phước Trung",
        "Thành phố Bà Rịa",
        "Tỉnh Bà Rịa - Vũng Tàu",
    )
    assert b_brvt.features["raw_prov_contradicts_goong"] == 0.0
    assert b_brvt.features["goong_hierarchy_valid"] == 1.0

    # Missing Goong levels -> -1 hierarchy validity
    b_single = ext_56.extract("12 Nguyễn Huệ", "", "", "Thành phố Hồ Chí Minh")
    assert b_single.features["goong_hierarchy_valid"] == -1.0


def test_pure_onnx_export_parity_and_modes(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(400, 56)).astype(np.float32)
    X[rng.random(X.shape) < 0.03] = 0.0
    y = ((0.7 * X[:, 0] - 0.5 * (X[:, 3] > 0.1) + 0.4 * X[:, 12]) > 0.15).astype(int)

    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=0.5, max_iter=500).fit(sc.transform(X), y)
    hgb = HistGradientBoostingClassifier(
        max_iter=35, max_depth=3, learning_rate=0.05, l2_regularization=2.0, min_samples_leaf=15, random_state=42
    ).fit(X, y)

    p_lr = lr.predict_proba(sc.transform(X))[:, 1]
    p_hgb = hgb.predict_proba(X)[:, 1]
    p_ens = 0.5 * p_lr + 0.5 * p_hgb

    names = [f"feat_{i}" for i in range(56)]
    for mode, w_lr, w_hgb, hgb_arg, target in (
        ("ens", 0.5, 0.5, hgb, p_ens),
        ("lr", 1.0, 0.0, None, p_lr),
        ("hgb", 0.0, 1.0, hgb, p_hgb),
    ):
        onnx_file = tmp_path / f"model_{mode}.onnx"
        export_lr_hgb_ensemble_to_onnx(
            out_path=onnx_file,
            feature_names=names,
            lr_mean=sc.mean_,
            lr_std=sc.scale_,
            lr_coef=lr.coef_[0],
            lr_intercept=float(lr.intercept_[0]),
            hgb_model=hgb_arg,
            w_lr=w_lr,
            w_hgb=w_hgb,
        )
        sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        p_ort = sess.run(["risk_prob"], {"features": X})[0].reshape(-1)
        assert p_ort.shape == (400,)
        assert float(np.max(np.abs(target - p_ort))) < 1e-4

    assert "torch" not in sys.modules
    assert "onnx" not in sys.modules


def test_train_local_2a_cli_end_to_end_and_zero_pii_leak(tmp_path: Path) -> None:
    rng = random.Random(2026)
    rows = []
    t0 = datetime(2026, 8, 1, 9, 0, 0)

    good_templates = [
        ("12 Nguyễn Trãi, Phường Bến Nghé, Quận 1, TP.HCM", "Phường Bến Nghé", "Quận 1", "Thành phố Hồ Chí Minh"),
        ("45 Lê Lợi, Phường Phúc Xá, Quận Ba Đình, Hà Nội", "Phường Hồng Hà", "", "Thành phố Hà Nội"),
        ("88 Hai Bà Trưng, Phường Tân Định, Quận 1, TP.HCM", "", "Quận 1", "Thành phố Hồ Chí Minh"),
        ("23 Trần Hưng Đạo, Xã Lộc Thạnh, Huyện Lộc Ninh, Bình Phước", "Xã Lộc Thạnh", "Huyện Lộc Ninh", "Tỉnh Bình Phước"),
        ("19 Nguyễn Văn Linh, Phường Phước Trung, Bà Rịa - Vũng Tàu", "Phường Phước Trung", "Thành phố Bà Rịa", "Tỉnh Bà Rịa - Vũng Tàu"),
    ]
    bad_templates = [
        ("gần chợ Tân Lập", "Xã Tân Lập", "Huyện Đồng Phú", "Tỉnh Bình Phước"),
        ("số 5 Hòa Bình", "Xã Hòa Bình", "Huyện Trà Ôn", "Tỉnh Vĩnh Long"),
        ("khu phố 2", "Phường Tân An", "Thành phố Thủ Dầu Một", "Tỉnh Bình Dương"),
        ("nhà trọ Minh", "Phường 1", "Quận 10", "Thành phố Hồ Chí Minh"),
    ]

    for i in range(600):
        ts = (t0 + timedelta(minutes=15 * i)).isoformat()
        phone = f"09{10000000 + i}"
        if i % 5 == 0:
            raw, gw, gd, gp = rng.choice(bad_templates)
            is_ret = "hoan" if rng.random() < 0.78 else "giao"
        else:
            raw, gw, gd, gp = rng.choice(good_templates)
            # Inject canary PII into raw text on some rows; verify zero leak into outputs!
            if i == 17:
                raw = f"CANARY_NAME_NGUYEN_VAN_X CANARY_PII_0987654321 - {raw}"
                phone = "0987654321"
            is_ret = "hoan" if rng.random() < 0.08 else "giao"

        rows.append({
            "order_id": f"ORD_{i:05d}",
            "created_at": ts,
            "customer_phone": phone,
            "raw_address": raw,
            "rule_d_pass": "block" if (i % 25 == 0) else "pass",
            "goong_ward": gw,
            "goong_district": gd,
            "goong_province": gp,
            "delivery_status": is_ret,
        })

    csv_path = tmp_path / "dev_orders.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    model_out = tmp_path / "local_2a_artifacts"
    rc = train_local_main([
        "--data", str(csv_path),
        "--output-dir", str(model_out),
        "--hgb-iters", "40",
        "--min-hard-caught", "10",
        "--min-soft-caught", "15",
    ])
    assert rc == 0

    for fname in (
        "post_goong_verifier_2a_local.onnx",
        "calibration.json",
        "local_eval_report.json",
        "vn_admin_v3_converter.json",
    ):
        assert (model_out / fname).exists(), f"Missing output artifact: {fname}"

    calib_txt = (model_out / "calibration.json").read_text(encoding="utf-8")
    report_txt = (model_out / "local_eval_report.json").read_text(encoding="utf-8")
    for pii in ("CANARY_PII_0987654321", "CANARY_NAME_NGUYEN_VAN_X", "0987654321", "ORD_00017"):
        assert pii not in calib_txt, f"PII leak detected in calibration.json: {pii}"
        assert pii not in report_txt, f"PII leak detected in local_eval_report.json: {pii}"

    calib = json.loads(calib_txt)
    report = json.loads(report_txt)
    assert calib["allow_missing_ward"] is True
    assert calib["feature_schema"] == "v31_2a_56"
    assert calib["output_type"] == "probability"
    assert report["onnx_export"]["parity_max_abs_diff"] < 1e-4
    assert report["oof_curve"]["roc_auc"] >= 0.75

    # Load into production PostGoongVerifier runtime
    v = PostGoongVerifier(artifacts_dir=model_out)
    assert v.allow_missing_ward is True
    assert v.feature_schema == "v31_2a_56"

    test_cases = [
        {"raw_text": "12 Nguyễn Trãi, Phường Bến Nghé, Quận 1, TP.HCM", "goong_ward": "Phường Bến Nghé", "goong_district": "Quận 1", "goong_province": "Thành phố Hồ Chí Minh"},
        {"raw_text": "88 Hai Bà Trưng, Phường Tân Định, Quận 1, TP.HCM", "goong_ward": "", "goong_district": "Quận 1", "goong_province": "Thành phố Hồ Chí Minh"},
        {"raw_text": "gần chợ Tân Lập", "goong_ward": "Xã Tân Lập", "goong_district": "Huyện Đồng Phú", "goong_province": "Tỉnh Bình Phước"},
    ]
    singles = [v.verify(**tc) for tc in test_cases]
    batches = v.verify_batch(test_cases)
    for s_res, b_res in zip(singles, batches):
        assert abs(s_res["risk_score"] - b_res["risk_score"]) < 1e-6
        assert s_res["action"] == b_res["action"]
        assert s_res["reason_code"] != "EMPTY_GOONG_WARD"

    assert singles[1]["action"] == "auto_approve"
    assert singles[2]["risk_score"] > singles[0]["risk_score"]

    # Runtime CPU latency check (100 calls after warmup)
    import time
    for _ in range(10):
        v.verify(**test_cases[0])
    lats = []
    for _ in range(100):
        t_start = time.perf_counter()
        v.verify(**test_cases[0])
        lats.append((time.perf_counter() - t_start) * 1000.0)
    p99 = float(np.percentile(lats, 99))
    assert p99 <= 10.0, f"Local 2A runtime p99 too high: {p99:.3f} ms"

    # Subprocess clean runtime check: zero torch / sklearn / pickle loaded when loading local 2A model
    code = (
        f"import sys; sys.path.insert(0, {str(PKG_DIR)!r}); "
        "from verifier import PostGoongVerifier; "
        f"v = PostGoongVerifier(artifacts_dir={str(model_out)!r}); "
        "assert 'torch' not in sys.modules; "
        "assert 'sklearn' not in sys.modules; "
        "r = v.verify('88 Hai Bà Trưng, Phường Tân Định, Quận 1, TP.HCM', '', 'Quận 1', 'Thành phố Hồ Chí Minh'); "
        "assert r['action'] == 'auto_approve'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Clean runtime check failed: {proc.stderr or proc.stdout}"
