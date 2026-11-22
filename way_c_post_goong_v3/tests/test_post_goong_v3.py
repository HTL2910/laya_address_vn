"""Comprehensive unit, integration, and latency tests for Post-Goong V3."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient

TESTS_DIR = Path(__file__).resolve().parent
PKG_DIR = TESTS_DIR.parent
ROOT_DIR = PKG_DIR.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from features import DEFAULT_ARTIFACTS_DIR, FEATURE_NAMES  # noqa: E402
from server import app  # noqa: E402
from verifier import PostGoongVerifier  # noqa: E402


@pytest.fixture(scope="module")
def verifier() -> PostGoongVerifier:
    return PostGoongVerifier(artifacts_dir=DEFAULT_ARTIFACTS_DIR)


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_artifacts_integrity_and_no_pt_files() -> None:
    pt_files = list(PKG_DIR.rglob("*.pt"))
    assert pt_files == [], f"Unexpected .pt files found in package: {pt_files}"

    onnx_path = DEFAULT_ARTIFACTS_DIR / "post_goong_verifier_v3.onnx"
    assert onnx_path.exists()
    assert onnx_path.stat().st_size <= 500 * 1024
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    assert model.opset_import[0].version == 14

    for fname in ("calibration.json", "scaler.json", "eval_report_v3.json", "vn_admin_v3_converter.json"):
        assert (DEFAULT_ARTIFACTS_DIR / fname).exists(), f"Missing artifact: {fname}"


def test_runtime_zero_torch_import() -> None:
    code = (
        f"import sys; sys.path.insert(0, {str(PKG_DIR)!r}); "
        "from verifier import PostGoongVerifier; "
        "v = PostGoongVerifier(); "
        "assert 'torch' not in sys.modules, 'torch imported in runtime!'; "
        "r = v.verify('Lộc Thạnh, Lộc Ninh', 'Xã Lộc Thạnh', 'Huyện Lộc Ninh', 'Tỉnh Bình Phước', 0.92); "
        "assert r['action'] == 'auto_approve'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Runtime check failed: {proc.stderr or proc.stdout}"


def test_loc_thanh_loc_ninh_short_valid(verifier: PostGoongVerifier) -> None:
    res = verifier.verify(
        raw_text="Lộc Thạnh, Lộc Ninh",
        goong_ward="Xã Lộc Thạnh",
        goong_district="Huyện Lộc Ninh",
        goong_province="Tỉnh Bình Phước",
        goong_score=0.92,
        goong_lat=11.9505,
        goong_lng=106.5231,
    )
    assert res["action"] == "auto_approve"
    assert res["is_valid_fill"] is True
    assert res["risk_score"] < verifier.t_low
    assert res["zns_confirm_message"] is None

    # Also valid under 2025 Goong V2 output (Dong Nai)
    res_v2 = verifier.verify(
        raw_text="Lộc Thạnh, Lộc Ninh",
        goong_ward="Xã Lộc Thạnh",
        goong_district="",
        goong_province="Tỉnh Đồng Nai",
        goong_score=0.90,
        goong_lat=11.9365,
        goong_lng=106.5760,
    )
    assert res_v2["action"] == "auto_approve"
    assert res_v2["is_valid_fill"] is True
    assert res_v2["is_legacy_2025_equiv"] is True


def test_legacy_to_2025_equivalence_auto_approve(verifier: PostGoongVerifier) -> None:
    res = verifier.verify(
        raw_text="Phúc Xá, Ba Đình, Hà Nội",
        goong_ward="Phường Hồng Hà",
        goong_district="",
        goong_province="Thành phố Hà Nội",
        goong_score=0.91,
        goong_lat=21.0567,
        goong_lng=105.8450,
    )
    assert res["action"] == "auto_approve"
    assert res["is_valid_fill"] is True
    assert res["is_legacy_2025_equiv"] is True

    # Reverse direction: 2025 input + Goong legacy 3-tier output
    res_rev = verifier.verify(
        raw_text="Phường Hồng Hà, Hà Nội",
        goong_ward="Phường Phúc Xá",
        goong_district="Quận Ba Đình",
        goong_province="Thành phố Hà Nội",
        goong_score=0.90,
        goong_lat=21.0468,
        goong_lng=105.8481,
    )
    assert res_rev["action"] == "auto_approve"
    assert res_rev["is_valid_fill"] is True
    assert res_rev["is_legacy_2025_equiv"] is True

    # Island legacy district -> 2025 Dac khu
    res_isle = verifier.verify(
        raw_text="Huyện Côn Đảo, Bà Rịa - Vũng Tàu",
        goong_ward="Đặc khu Côn Đảo",
        goong_district="",
        goong_province="Thành phố Hồ Chí Minh",
        goong_score=0.90,
        goong_lat=8.6981,
        goong_lng=106.6115,
    )
    assert res_isle["action"] == "auto_approve"
    assert res_isle["is_valid_fill"] is True
    assert res_isle["is_legacy_2025_equiv"] is True


def test_street_ward_collision_traps_auto_approve(verifier: PostGoongVerifier) -> None:
    res1 = verifier.verify(
        raw_text="45 Bình Thuận, Quận 7, TP.HCM",
        goong_ward="Phường Bình Thuận",
        goong_district="Quận 7",
        goong_province="Thành phố Hồ Chí Minh",
        goong_score=0.95,
        goong_lat=10.7418,
        goong_lng=106.7205,
    )
    assert res1["action"] == "auto_approve"
    assert res1["is_valid_fill"] is True

    res2 = verifier.verify(
        raw_text="123 Hai Bà Trưng, P.6, Q3",
        goong_ward="Phường 6",
        goong_district="Quận 3",
        goong_province="Thành phố Hồ Chí Minh",
        goong_score=0.93,
        goong_lat=10.7810,
        goong_lng=106.6910,
    )
    assert res2["action"] == "auto_approve"
    assert res2["is_valid_fill"] is True


def test_ambiguous_ward_hallucinated_parent_not_auto_approved(verifier: PostGoongVerifier) -> None:
    res = verifier.verify(
        raw_text="Số 5 gần chợ, Tân Lập",
        goong_ward="Xã Tân Lập",
        goong_district="Huyện Đồng Phú",
        goong_province="Tỉnh Bình Phước",
        goong_score=0.78,
    )
    assert res["action"] != "auto_approve"
    assert res["action"] in ("hard_block", "soft_confirm_zns")
    assert res["is_valid_fill"] is False
    assert res["risk_score"] >= 0.50
    if res["action"] == "soft_confirm_zns":
        assert isinstance(res["zns_confirm_message"], str) and len(res["zns_confirm_message"]) > 10


def test_soft_confirm_zns_band(verifier: PostGoongVerifier) -> None:
    # Low-ambiguity (2-province) ungrounded ward lands in the [t_low, t_high) uncertainty band
    res = verifier.verify(
        raw_text="Phường Ngọc Hà",
        goong_ward="Phường Ngọc Hà",
        goong_district="Quận Ba Đình",
        goong_province="Thành phố Hà Nội",
        goong_score=0.50,
    )
    assert res["action"] == "soft_confirm_zns"
    assert res["is_valid_fill"] is False
    assert verifier.t_low <= res["risk_score"] < verifier.t_high
    assert isinstance(res["zns_confirm_message"], str) and "Phường Ngọc Hà" in res["zns_confirm_message"]


def test_province_contradiction_hard_block(verifier: PostGoongVerifier) -> None:
    res = verifier.verify(
        raw_text="Xã Hòa Bình, Long An",
        goong_ward="Xã Hòa Bình",
        goong_district="Huyện Trà Ôn",
        goong_province="Tỉnh Vĩnh Long",
        goong_score=0.85,
    )
    assert res["action"] == "hard_block"
    assert res["is_valid_fill"] is False
    assert res["reason_code"] == "PROVINCE_CONTRADICTION"
    assert res["risk_score"] >= verifier.t_high


def test_spatial_outlier_caught(verifier: PostGoongVerifier) -> None:
    # True Phuc Xa is around (21.0468, 105.8481); +0.65 deg lat is ~72 km away (> 60 km)
    res = verifier.verify(
        raw_text="Phúc Xá, Ba Đình, Hà Nội",
        goong_ward="Phường Phúc Xá",
        goong_district="Quận Ba Đình",
        goong_province="Thành phố Hà Nội",
        goong_score=0.96,
        goong_lat=21.7000,
        goong_lng=105.8481,
    )
    assert res["spatial_dist_km"] > 60.0
    assert res["action"] != "auto_approve"
    assert res["is_valid_fill"] is False
    assert res["reason_code"] == "SPATIAL_OUTLIER"


def test_hyphenated_telex_empty_ward_and_chat_noise(verifier: PostGoongVerifier) -> None:
    res_hy = verifier.verify(
        raw_text="Phường Phước Trung, Bà Rịa - Vũng Tàu",
        goong_ward="Phường Phước Trung",
        goong_district="Thành phố Bà Rịa",
        goong_province="Tỉnh Bà Rịa - Vũng Tàu",
        goong_score=0.92,
    )
    assert res_hy["action"] == "auto_approve"

    res_hy2 = verifier.verify(
        raw_text="Kinh Dinh, Phan Rang - Tháp Chàm, Ninh Thuận",
        goong_ward="Phường Kinh Dinh",
        goong_district="Thành phố Phan Rang - Tháp Chàm",
        goong_province="Tỉnh Ninh Thuận",
        goong_score=0.91,
    )
    assert res_hy2["action"] == "auto_approve"

    res_unique = verifier.verify(
        raw_text="Phường Trúc Bạch",
        goong_ward="Phường Trúc Bạch",
        goong_district="Quận Ba Đình",
        goong_province="Thành phố Hà Nội",
        goong_score=0.88,
    )
    assert res_unique["action"] == "auto_approve"

    res_telex = verifier.verify(
        raw_text="Loojc Thanhj, Loojc Ninh",
        goong_ward="Xã Lộc Thạnh",
        goong_district="Huyện Lộc Ninh",
        goong_province="Tỉnh Bình Phước",
        goong_score=0.90,
    )
    assert res_telex["action"] == "auto_approve"

    res_empty = verifier.verify(
        raw_text="12 Nguyễn Huệ, Bến Nghé, Quận 1, TP.HCM",
        goong_ward="",
        goong_district="Quận 1",
        goong_province="Thành phố Hồ Chí Minh",
        goong_score=0.60,
    )
    assert res_empty["action"] != "auto_approve"
    assert res_empty["reason_code"] == "EMPTY_GOONG_WARD"

    res_chat = verifier.verify(
        raw_text="1 ly trà sữa trân châu đường đen ít đá",
        goong_ward="Phường Bến Nghé",
        goong_district="Quận 1",
        goong_province="Thành phố Hồ Chí Minh",
        goong_score=0.45,
    )
    assert res_chat["action"] != "auto_approve"
    assert res_chat["is_valid_fill"] is False


def test_batch_and_fastapi_endpoints(client: TestClient) -> None:
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "ok"

    single_resp = client.post(
        "/v1/verify",
        json={
            "raw_text": "Lộc Thạnh, Lộc Ninh",
            "goong_ward": "Xã Lộc Thạnh",
            "goong_district": "Huyện Lộc Ninh",
            "goong_province": "Tỉnh Bình Phước",
            "goong_score": 0.92,
        },
    )
    assert single_resp.status_code == 200
    assert single_resp.json()["action"] == "auto_approve"

    batch_resp = client.post(
        "/v1/verify/batch",
        json={
            "items": [
                {
                    "raw_text": "Phúc Xá, Ba Đình, Hà Nội",
                    "goong_ward": "Phường Hồng Hà",
                    "goong_district": "",
                    "goong_province": "Thành phố Hà Nội",
                    "goong_score": 0.91,
                },
                {
                    "raw_text": "Xã Hòa Bình, Long An",
                    "goong_ward": "Xã Hòa Bình",
                    "goong_district": "Huyện Trà Ôn",
                    "goong_province": "Tỉnh Vĩnh Long",
                    "goong_score": 0.85,
                },
            ]
        },
    )
    assert batch_resp.status_code == 200
    body = batch_resp.json()
    assert body["count"] == 2
    assert body["results"][0]["action"] == "auto_approve"
    assert body["results"][0]["is_legacy_2025_equiv"] is True
    assert body["results"][1]["action"] == "hard_block"


def test_holdout_metrics_and_p99_latency(verifier: PostGoongVerifier) -> None:
    with open(DEFAULT_ARTIFACTS_DIR / "eval_report_v3.json", "r", encoding="utf-8") as f:
        report = json.load(f)

    assert len(FEATURE_NAMES) >= 34
    assert report["zero_overlap_verified"] is True
    assert report["n_params"] <= 150_000
    assert report["onnx_opset"] == 14
    assert report["onnx_size_kb"] <= 500.0

    m = report["holdout_600_metrics"]
    assert m["count"] == 600
    assert m["roc_auc"] >= 0.98
    assert m["pr_auc"] >= 0.98
    assert m["ece"] <= 0.05
    assert m["hallucination_recall"] >= 0.95
    assert m["contradiction_recall"] >= 0.97
    assert m["spatial_outlier_recall"] >= 0.97
    assert m["legacy_2025_false_alarm_rate"] <= 0.03
    assert m["hard_block_precision"] >= 0.90

    # Live runtime latency check (200 single calls after warmup)
    sample = ("45 Bình Thuận, Quận 7, TP.HCM", "Phường Bình Thuận", "Quận 7", "Thành phố Hồ Chí Minh", 0.95, 10.7418, 106.7205)
    for _ in range(20):
        verifier.verify(*sample)
    lats = []
    for _ in range(200):
        t0 = time.perf_counter()
        verifier.verify(*sample)
        lats.append((time.perf_counter() - t0) * 1000.0)
    p99 = float(np.percentile(lats, 99))
    assert p99 <= 5.0, f"Runtime p99 latency too high: {p99:.3f} ms"


def test_server_env_override_loads_v31_2a_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import server as srv
    from features_2a import FEATURE_NAMES_2A_V31
    from onnx_export_pure import export_lr_hgb_ensemble_to_onnx
    import shutil

    n_f = len(FEATURE_NAMES_2A_V31)
    onnx_out = tmp_path / "post_goong_verifier_2a_local.onnx"
    export_lr_hgb_ensemble_to_onnx(
        out_path=onnx_out,
        feature_names=FEATURE_NAMES_2A_V31,
        lr_mean=np.zeros(n_f, dtype=np.float32),
        lr_std=np.ones(n_f, dtype=np.float32),
        lr_coef=np.zeros(n_f, dtype=np.float32),
        lr_intercept=-1.5,
        hgb_model=None,
        w_lr=1.0,
        w_hgb=0.0,
    )
    onnx.checker.check_model(onnx.load(str(onnx_out)))
    shutil.copy2(DEFAULT_ARTIFACTS_DIR / "vn_admin_v3_converter.json", tmp_path / "vn_admin_v3_converter.json")
    (tmp_path / "calibration.json").write_text(
        json.dumps({
            "version": "3.1.0",
            "model_filename": "post_goong_verifier_2a_local.onnx",
            "feature_schema": "v31_2a_56",
            "feature_names": list(FEATURE_NAMES_2A_V31),
            "output_type": "probability",
            "temperature": 1.0,
            "t_low": 0.40,
            "t_high": 0.70,
            "allow_missing_ward": True,
        }),
        encoding="utf-8",
    )

    monkeypatch.setenv("LAYA_POST_GOONG_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(srv, "_VERIFIER", None)
    with TestClient(srv.app) as c:
        h = c.get("/health").json()
        assert h["status"] == "ok"
        assert h["version"] == "3.1.0"
        assert h["feature_schema"] == "v31_2a_56"
        assert h["allow_missing_ward"] is True
        r = c.post(
            "/v1/verify",
            json={
                "raw_text": "88 Hai Bà Trưng, Tân Định, Quận 1, TP.HCM",
                "goong_ward": "",
                "goong_district": "Quận 1",
                "goong_province": "Thành phố Hồ Chí Minh",
            },
        )
        assert r.status_code == 200
        assert r.json()["action"] == "auto_approve"
    monkeypatch.setattr(srv, "_VERIFIER", None)
