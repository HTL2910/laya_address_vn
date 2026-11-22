"""Comprehensive verification & CPU benchmark test suite for Way B (Laya Address VN).

Run from way_b_backend_api directory:
    .venv\\Scripts\\pytest.exe -v --basetemp=".pytest_tmp"
"""

from __future__ import annotations

import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
import pytest

from app.server import app

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Curated 24-case Vietnamese suite: 12 diverse positives + 12 context-tricky hard negatives
CURATED_24_CASES = [
    # --- 12 Positives (formal, unaccented, abbreviated, typos, rural, apartment, chat-embedded) ---
    ("Số 125 Nguyễn Trãi, Phường Thượng Đình, Quận Thanh Xuân, Hà Nội", True),
    ("so 18 ngo 25 lang ha dong da ha noi", True),
    ("P.Ben Nghe, Q.1, TP.HCM", True),
    ("KP3 P.Tam Hoa Bien Hoa Dong Nai", True),
    ("Hẻm 214/19 Nguyễn Văn Nguyễn Q1", True),
    ("Thôn 3, Xã Ea Kao, TP Buôn Ma Thuột, Đắk Lắk", True),
    ("Ấp Tân Lập, Xã Tân Thông Hội, Củ Chi", True),
    ("Tầng 12 Tòa nhà Keangnam Phạm Hùng Nam Từ Liêm HN", True),
    ("ship giúp mình tới 88 hai bà trưng hoàn kiếm nhé", True),
    ("45/12/3B Le Van Sy P.14 Q.3 TPHCM", True),
    ("Phòng 1405 Tòa S1.02 Vinhomes Smart City Tây Mỗ Nam Từ Liêm Hà Nội", True),
    ("giao hàng qua số 15 ngõ 42 thái hà đống đa giúp em ạ", True),
    # --- 12 Hard Negatives (context-tricky sentences with real VN places + non-addresses) ---
    ("Hà Nội hôm nay trở lạnh rồi", False),
    ("Sài Gòn mùa này hay mưa chiều", False),
    ("Mình muốn đi du lịch Đà Nẵng 3 ngày 2 đêm", False),
    ("Đội tuyển Việt Nam đá ở Mỹ Đình tối nay", False),
    ("Công ty mình có chi nhánh ở Cầu Giấy và Thanh Xuân", False),
    ("Gặp nhau lúc 7h tối ở quán cà phê nhé", False),
    ("Anh Nguyễn Trãi là danh nhân văn hóa", False),
    ("Giá nhà ở Quận 1 và Quận 3 tăng mạnh năm nay", False),
    ("chuyen tien don hang 12345", False),
    ("0912345678", False),
    ("Cà phê rang xay Buôn Ma Thuột gói 500g", False),
    ("https://shopee.vn/product/123456", False),
]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as tc:
        yield tc


def test_artifacts_exist_and_int8_smaller_than_fp32():
    fp32 = ARTIFACTS_DIR / "laya_address_vn_fp32.onnx"
    int8 = ARTIFACTS_DIR / "laya_address_vn_int8.onnx"
    vocab = ARTIFACTS_DIR / "vocab.json"
    cal = ARTIFACTS_DIR / "calibration.json"

    assert fp32.exists(), "Missing FP32 ONNX artifact"
    assert int8.exists(), "Missing INT8 ONNX artifact"
    assert vocab.exists(), "Missing vocab.json"
    assert cal.exists(), "Missing calibration.json"

    # Must be self-contained single files (no external .data sidecar)
    assert not Path(str(fp32) + ".data").exists()
    assert not Path(str(int8) + ".data").exists()

    fp32_size = fp32.stat().st_size
    int8_size = int8.stat().st_size
    assert int8_size < fp32_size, (
        f"Expected INT8 ({int8_size} B) to be smaller than FP32 ({fp32_size} B)"
    )

    cal_data = json.loads(cal.read_text(encoding="utf-8"))
    temp = cal_data["calibration"]["temperature"]
    assert 0.1 < temp < 10.0


def test_health_info_and_web_ui(client: TestClient):
    r_health = client.get("/health")
    assert r_health.status_code == 200
    h = r_health.json()
    assert h["status"] == "ok"
    assert h["model_loaded"] is True
    assert h["model_file"] == "laya_address_vn_int8.onnx"

    r_info = client.get("/v1/info")
    assert r_info.status_code == 200
    info = r_info.json()
    assert info["quantization"]["int8_size_bytes"] < info["quantization"]["fp32_size_bytes"]
    assert info["calibration"]["temperature"] > 0.0
    assert info["quantization"]["int8_test_metrics"]["accuracy"] >= 0.96

    r_ui = client.get("/")
    assert r_ui.status_code == 200
    assert "text/html" in r_ui.headers["content-type"]
    assert "Laya for Address VN" in r_ui.text


@pytest.mark.parametrize("mode", ["hybrid", "model_only", "heuristic_only"])
def test_decide_single_schema_and_probabilities(client: TestClient, mode: str):
    r = client.post(
        "/v1/decide",
        json={"text": "so 18 ngo 25 lang ha dong da ha noi", "mode": mode},
    )
    assert r.status_code == 200
    body = r.json()
    for field in ("text", "is_address", "decision", "probability", "confidence", "tier", "latency_ms", "mode"):
        assert field in body
    assert body["mode"] == mode
    assert 0.0 < body["probability"] < 1.0
    assert 0.5 <= body["confidence"] < 1.0
    assert body["decision"] in ("yes", "no")
    assert body["is_address"] == (body["decision"] == "yes")


def test_curated_24_case_accuracy_hybrid_and_model_only(client: TestClient):
    texts = [t for t, _ in CURATED_24_CASES]
    expected = [y for _, y in CURATED_24_CASES]

    for mode in ("hybrid", "model_only"):
        r = client.post("/v1/decide/batch", json={"texts": texts, "mode": mode})
        assert r.status_code == 200
        payload = r.json()
        assert payload["count"] == 24
        assert payload["total_latency_ms"] > 0
        assert payload["throughput_req_per_s"] > 0

        results = payload["results"]
        correct = 0
        failures = []
        for item, exp in zip(results, expected):
            assert 0.0 < item["probability"] < 1.0
            if item["is_address"] == exp:
                correct += 1
            else:
                failures.append((item["text"], exp, item["is_address"], item["probability"], item["tier"]))

        acc = correct / len(expected)
        assert acc >= 0.91, (
            f"Mode={mode} accuracy {acc:.3f} ({correct}/24) below threshold. Failures: {failures}"
        )


def test_cpu_single_and_batch_latency_benchmark(client: TestClient):
    # Warmup
    client.post("/v1/decide", json={"text": "Hà Nội hôm nay trở lạnh rồi", "mode": "model_only"})

    sample_texts = [t for t, _ in CURATED_24_CASES]
    single_latencies_ms = []
    for txt in sample_texts:
        t0 = time.perf_counter()
        r = client.post("/v1/decide", json={"text": txt, "mode": "model_only"})
        assert r.status_code == 200
        single_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        assert r.json()["latency_ms"] < 25.0

    mean_single_ms = sum(single_latencies_ms) / len(single_latencies_ms)
    assert mean_single_ms < 25.0, f"Mean single CPU latency {mean_single_ms:.2f}ms exceeded 25ms"

    # Batch 100-item throughput check
    batch_100 = [sample_texts[i % len(sample_texts)] for i in range(100)]
    r_batch = client.post("/v1/decide/batch", json={"texts": batch_100, "mode": "hybrid"})
    assert r_batch.status_code == 200
    b = r_batch.json()
    assert b["count"] == 100
    assert b["avg_latency_ms"] < 10.0, f"Batch avg latency {b['avg_latency_ms']:.3f}ms too high"
    assert b["throughput_req_per_s"] > 100.0
