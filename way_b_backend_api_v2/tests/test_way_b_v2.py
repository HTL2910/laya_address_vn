"""Comprehensive verification, accuracy, anti-leakage & CPU latency suite for Way B v2.

Run from way_b_backend_api_v2 directory:
    .venv\\Scripts\\pytest.exe -v --basetemp=".pytest_tmp"
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import time
import unicodedata

from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.engine import DEFAULT_ARTIFACTS_DIR, LayaAddressEngine, SUFFICIENCY_CLASSES
from app.server import app

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
APP_DIR = BASE_DIR / "app"
TRAIN_MANIFEST = BASE_DIR / "train" / "train_texts_manifest.txt"
BENCH_V1 = BASE_DIR.parent / "benchmark" / "vn_address_benchmark.json"
BENCH_V2 = BASE_DIR.parent / "benchmark_v2" / "independent_holdout_v2.json"

MANDATORY_CASES = [
    # --- address_sufficient (all 6 short styles + full + delivery freetext) ---
    ("Lộc Thạnh, Lộc Ninh", "address_sufficient"),
    ("Lộc Thạnh Lộc Ninh", "address_sufficient"),
    ("loc thanh loc ninh", "address_sufficient"),
    ("x. lộc thạnh, h. lộc ninh", "address_sufficient"),
    ("x.loc thanh h.loc ninh", "address_sufficient"),
    ("p.12 q.tb", "address_sufficient"),
    ("Lộc Thạnh, Lộc Ninh, Bình Phước", "address_sufficient"),
    ("Lộc Thạnh, Bình Phước", "address_sufficient"),
    ("Phường Hồng Hà, Hà Nội", "address_sufficient"),
    ("Số 125 Nguyễn Trãi, Phường Thượng Đình, Quận Thanh Xuân, Hà Nội", "address_sufficient"),
    ("Nguyễn Văn Nam - 0912345678 - 45 Lê Lợi, Bến Nghé, Quận 1, TP.HCM (giao giờ hành chính)", "address_sufficient"),
    ("nhận tại 50 Nguyễn Chí Thanh, Đống Đa, Hà Nội - không gọi chuông", "address_sufficient"),
    ("giao cho anh hùng ở 25 trần hưng đạo hoàn kiếm hà nội nhé shop", "address_sufficient"),
    # --- address_insufficient (ward-only, province-only, district/dist+prov, fragments) ---
    ("Xã Lộc Thạnh", "address_insufficient"),
    ("Phường Bến Nghé", "address_insufficient"),
    ("Thị trấn Đông Anh", "address_insufficient"),
    ("p.12", "address_insufficient"),
    ("Hà Nội", "address_insufficient"),
    ("Tỉnh Bình Phước", "address_insufficient"),
    ("TP.HCM", "address_insufficient"),
    ("binh duong", "address_insufficient"),
    ("Quận 1", "address_insufficient"),
    ("Huyện Lộc Ninh", "address_insufficient"),
    ("Thanh Xuân, Hà Nội", "address_insufficient"),
    ("Lộc Ninh, Bình Phước", "address_insufficient"),
    ("ngõ 12", "address_insufficient"),
    ("Thôn 7", "address_insufficient"),
    ("thôn 3", "address_insufficient"),
    ("khu phố 3", "address_insufficient"),
    ("xóm 4", "address_insufficient"),
    ("ấp 2", "address_insufficient"),
    ("tổ 5", "address_insufficient"),
    ("hẻm 30", "address_insufficient"),
    ("tầng 4", "address_insufficient"),
    ("đường Lê Lợi", "address_insufficient"),
    ("lô B2", "address_insufficient"),
    ("chung cư A1", "address_insufficient"),
    # --- not_address (garbage, chat/order, food+place, person names, narrative/questions) ---
    ("73", "not_address"),
    ("10", "not_address"),
    ("K", "not_address"),
    ("Hi", "not_address"),
    ("Ok", "not_address"),
    ("Jean", "not_address"),
    ("Xom", "not_address"),
    ("Xin giá", "not_address"),
    ("Cho mình 3 chai nướng BBQ", "not_address"),
    ("ok shop", "not_address"),
    ("ship nhanh giúp", "not_address"),
    ("bánh mì Sài Gòn", "not_address"),
    ("mực Cà Mau", "not_address"),
    ("phở Hà Nội", "not_address"),
    ("bún bò Huế", "not_address"),
    ("nước mắm Phú Quốc", "not_address"),
    ("kẹo dừa Bến Tre", "not_address"),
    ("nem chua Thanh Hóa", "not_address"),
    ("Nguyễn Thị Thanh Xuân", "not_address"),
    ("Trần Văn Phú", "not_address"),
    ("Nguyễn Trãi", "not_address"),
    ("Lê Lợi", "not_address"),
    ("Phí ship về Bình Phước bao nhiêu?", "not_address"),
    ("Nguyễn Trãi là danh nhân văn hóa thế giới của Việt Nam", "not_address"),
]


def _norm_key(s: str) -> str:
    s = (s or "").replace("đ", "d").replace("Đ", "D")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s.lower().strip())


@pytest.fixture(scope="module")
def engine() -> LayaAddressEngine:
    return LayaAddressEngine(DEFAULT_ARTIFACTS_DIR)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as tc:
        yield tc


def test_artifacts_self_contained_and_zero_torch_or_pt_in_runtime():
    fp32 = ARTIFACTS_DIR / "laya_address_vn_fp32.onnx"
    int8 = ARTIFACTS_DIR / "laya_address_vn_int8.onnx"
    vocab = ARTIFACTS_DIR / "vocab.json"
    cal = ARTIFACTS_DIR / "calibration.json"
    gaz = ARTIFACTS_DIR / "vn_admin_gazetteer.json"

    for p in (fp32, int8, vocab, cal, gaz):
        assert p.exists(), f"Missing artifact: {p}"

    assert not Path(str(fp32) + ".data").exists(), "FP32 ONNX must be single-file self-contained"
    assert not Path(str(int8) + ".data").exists(), "INT8 ONNX must be single-file self-contained"
    assert int8.stat().st_size < fp32.stat().st_size, "INT8 must be smaller than FP32"

    # Zero .pt/.pth checkpoint files anywhere in way_b_backend_api_v2 (excluding .venv site-packages .pth config)
    pt_files = [
        p for p in list(BASE_DIR.rglob("*.pt")) + list(BASE_DIR.rglob("*.pth"))
        if ".venv" not in p.parts
    ]
    assert not pt_files, f"Found forbidden PyTorch checkpoint files: {pt_files}"

    # Zero torch or pickle imports in app/ runtime
    for py_file in APP_DIR.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in ("torch", "pickle"), f"Forbidden import {alias.name} in {py_file}"
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                assert mod not in ("torch", "pickle"), f"Forbidden import {mod} in {py_file}"

    cal_data = json.loads(cal.read_text(encoding="utf-8"))
    assert 0.1 < cal_data["calibration"]["temperature"] < 10.0
    assert 0.1 < cal_data["calibration"]["sufficiency_temperature"] < 10.0
    assert cal_data["calibration"]["sufficiency_classes"] == list(SUFFICIENCY_CLASSES)


def test_strict_anti_leakage_manifest():
    assert TRAIN_MANIFEST.exists(), "Missing train/train_texts_manifest.txt"
    train_lines = [line.strip() for line in TRAIN_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(train_lines) == 60000, f"Expected 60,000 train lines, got {len(train_lines)}"

    train_keys = {_norm_key(t) for t in train_lines}
    for bench_path in (BENCH_V1, BENCH_V2):
        cases = json.loads(bench_path.read_text(encoding="utf-8"))["cases"]
        eval_keys = {_norm_key(c["text"]) for c in cases}
        overlap = train_keys & eval_keys
        assert not overlap, f"Exact leakage with {bench_path.name}: {overlap}"


@pytest.mark.parametrize("mode", ["hybrid", "model_only", "heuristic_only"])
def test_engine_backward_compatibility_and_sufficiency_contract(engine: LayaAddressEngine, mode: str):
    dec = engine.decide("Lộc Thạnh, Lộc Ninh", mode=mode)
    d = dec.to_dict()
    for attr in (
        "text", "is_address", "decision", "probability", "confidence",
        "sufficiency", "sufficiency_probs", "sufficiency_score", "tier", "reason", "latency_ms",
    ):
        assert hasattr(dec, attr)
        assert attr in d

    assert dec.sufficiency in SUFFICIENCY_CLASSES
    assert dec.is_address is (dec.sufficiency == "address_sufficient")
    assert dec.decision == ("yes" if dec.is_address else "no")
    assert 0.0 < dec.probability < 1.0
    assert abs(dec.probability - dec.sufficiency_probs["address_sufficient"]) <= 1e-4
    assert abs(sum(dec.sufficiency_probs.values()) - 1.0) <= 1e-3
    assert dec.sufficiency_score == dec.probability
    assert dec.is_address is True


def test_mandatory_cases_hybrid_and_model_only(engine: LayaAddressEngine):
    for mode in ("hybrid", "model_only"):
        failures = []
        for txt, expected_suff in MANDATORY_CASES:
            dec = engine.decide(txt, mode=mode)
            assert dec.sufficiency in SUFFICIENCY_CLASSES
            assert dec.is_address == (dec.sufficiency == "address_sufficient")
            if dec.sufficiency != expected_suff:
                failures.append((txt, expected_suff, dec.sufficiency, dec.probability, dec.tier, dec.reason))

        if mode == "hybrid":
            assert not failures, f"Hybrid mode failed mandatory cases: {failures}"
        else:
            acc = (len(MANDATORY_CASES) - len(failures)) / len(MANDATORY_CASES)
            assert acc >= 0.90, f"model_only mandatory accuracy {acc:.3f} too low: {failures}"


def test_cpu_single_item_latency_p99_under_10ms(engine: LayaAddressEngine):
    # Warmup
    for txt, _ in MANDATORY_CASES[:15]:
        engine.decide(txt, mode="model_only")

    sample_inputs = [
        "Lộc Thạnh, Lộc Ninh",
        "loc thanh loc ninh",
        "x.loc thanh h.loc ninh",
        "p.12 q.tb",
        "Xã Lộc Thạnh",
        "ngõ 12",
        "Thôn 7",
        "73",
        "Cho mình 3 chai nướng BBQ",
        "bánh mì Sài Gòn",
        "Nguyễn Thị Thanh Xuân",
        "giao cho anh hùng ở 25 trần hưng đạo hoàn kiếm hà nội nhé shop",
    ]

    for mode in ("model_only", "hybrid"):
        latencies = []
        for i in range(120):
            txt = sample_inputs[i % len(sample_inputs)]
            t0 = time.perf_counter()
            dec = engine.decide(txt, mode=mode)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            assert dec.latency_ms >= 0.0
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        assert p99 <= 10.0, f"{mode} single-item CPU latency p99={p99:.3f}ms > 10ms (p50={p50:.3f}, p95={p95:.3f})"


def test_fastapi_endpoints_and_ui(client: TestClient):
    r_ui = client.get("/")
    assert r_ui.status_code == 200
    assert "text/html" in r_ui.headers["content-type"]
    assert "Laya for Address VN v2" in r_ui.text
    assert "http://" not in r_ui.text and "https://" not in r_ui.text, "No external CDN links allowed in index.html"

    r_health = client.get("/health")
    assert r_health.status_code == 200
    h = r_health.json()
    assert h["status"] == "ok"
    assert h["model_loaded"] is True
    assert h["model_file"] == "laya_address_vn_int8.onnx"
    assert h["sufficiency_temperature"] > 0.0

    r_info = client.get("/v1/info")
    assert r_info.status_code == 200
    info = r_info.json()
    assert info["runtime"]["gazetteer_counts"] == {"provinces": 63, "districts": 656, "wards": 7044}
    assert info["quantization"]["int8_size_bytes"] < info["quantization"]["fp32_size_bytes"]
    assert info["quantization"]["int8_test_sufficiency_metrics"]["accuracy"] >= 0.94

    for txt, exp_suff in [
        ("Lộc Thạnh, Lộc Ninh", "address_sufficient"),
        ("Xã Lộc Thạnh", "address_insufficient"),
        ("ngõ 12", "address_insufficient"),
        ("bánh mì Sài Gòn", "not_address"),
    ]:
        r_dec = client.post("/v1/decide", json={"text": txt, "mode": "hybrid"})
        assert r_dec.status_code == 200
        body = r_dec.json()
        assert body["sufficiency"] == exp_suff
        assert body["is_address"] is (exp_suff == "address_sufficient")
        assert set(body["sufficiency_probs"].keys()) == set(SUFFICIENCY_CLASSES)

    r_batch = client.post(
        "/v1/decide/batch",
        json={"texts": [t for t, _ in MANDATORY_CASES], "mode": "hybrid"},
    )
    assert r_batch.status_code == 200
    batch = r_batch.json()
    assert batch["count"] == len(MANDATORY_CASES)
    assert batch["throughput_req_per_s"] > 100.0
    for item, (_, exp_suff) in zip(batch["results"], MANDATORY_CASES):
        assert item["sufficiency"] == exp_suff
        assert item["is_address"] == (exp_suff == "address_sufficient")
