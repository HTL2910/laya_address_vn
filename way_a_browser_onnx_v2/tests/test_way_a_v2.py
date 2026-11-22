"""Comprehensive verification test suite for Way A v2 (Laya Address VN Browser ONNX)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT_DIR / "train"
WEB_DIR = ROOT_DIR / "web"
MODEL_DIR = WEB_DIR / "model"
VENDOR_DIR = WEB_DIR / "vendor" / "onnxruntime-web"

ONNX_PATH = MODEL_DIR / "laya_address_vn.onnx"
GAZETTEER_PATH = MODEL_DIR / "vn_admin_gazetteer.json"
VOCAB_PATH = MODEL_DIR / "vocab.json"
CALIB_PATH = MODEL_DIR / "calibration.json"
MANIFEST_PATH = TRAIN_DIR / "train_texts_manifest.txt"

WORKSPACE_ROOT = ROOT_DIR.parent
BENCH_V1_PATH = WORKSPACE_ROOT / "benchmark" / "vn_address_benchmark.json"
BENCH_V2_PATH = WORKSPACE_ROOT / "benchmark_v2" / "independent_holdout_v2.json"

sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(ROOT_DIR))

from dataset_v2 import norm_key  # noqa: E402
from serve import create_server  # noqa: E402
from train_export_v2 import encode_batch, sigmoid_np, softmax_np  # noqa: E402


@pytest.fixture(scope="module")
def artifacts():
    assert ONNX_PATH.exists(), f"Missing ONNX model: {ONNX_PATH}"
    assert GAZETTEER_PATH.exists(), f"Missing gazetteer: {GAZETTEER_PATH}"
    assert VOCAB_PATH.exists(), f"Missing vocab.json: {VOCAB_PATH}"
    assert CALIB_PATH.exists(), f"Missing calibration.json: {CALIB_PATH}"

    vocab_data = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    calib_data = json.loads(CALIB_PATH.read_text(encoding="utf-8"))

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(ONNX_PATH), sess_options=opts, providers=["CPUExecutionProvider"])
    return {
        "vocab_data": vocab_data,
        "vocab": vocab_data["char_to_id"],
        "max_length": vocab_data["max_length"],
        "calib": calib_data,
        "sess": sess,
    }


def _run_node_tier1(texts):
    js_path = str(WEB_DIR / "heuristics.js").replace("\\", "\\\\")
    gaz_path = str(GAZETTEER_PATH).replace("\\", "\\\\")
    script = (
        f"const fs = require('fs');"
        f"const H = require('{js_path}');"
        f"H.initGazetteer(JSON.parse(fs.readFileSync('{gaz_path}', 'utf8')));"
        "let buf = '';"
        "process.stdin.on('data', d => buf += d);"
        "process.stdin.on('end', () => {"
        "  const items = JSON.parse(buf);"
        "  process.stdout.write(JSON.stringify(items.map(t => H.evaluateTier1(t))));"
        "});"
    )
    proc = subprocess.run(
        ["node", "-e", script],
        input=json.dumps(texts, ensure_ascii=False),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout)


def test_model_artifacts_size_self_contained_and_wasm_safe():
    ext_data = Path(str(ONNX_PATH) + ".data")
    assert not ext_data.exists(), "ONNX model must be self-contained (no .onnx.data sidecar)"

    onnx_bytes = ONNX_PATH.stat().st_size
    assert onnx_bytes <= 1_572_864, f"ONNX FP32 exceeds 1.5 MB: {onnx_bytes}"

    model_files = [p for p in MODEL_DIR.iterdir() if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in model_files)
    assert total_bytes <= 2_097_152, f"Total web/model/* exceeds 2.0 MB: {total_bytes}"

    calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
    assert calib["model_size_bytes"] == onnx_bytes
    assert calib["calibration_size_bytes"] == CALIB_PATH.stat().st_size
    assert calib["total_artifacts_bytes"] == total_bytes

    assert not list(WEB_DIR.rglob("*.pt")), "No .pt files allowed under web/"

    model = onnx.load(str(ONNX_PATH), load_external_data=False)
    onnx.checker.check_model(model)
    for init in model.graph.initializer:
        assert init.data_location != onnx.TensorProto.EXTERNAL, f"External data initializer: {init.name}"

    wasm_safe_ops = {
        "Gather",
        "Transpose",
        "Conv",
        "Relu",
        "ReduceMax",
        "ReduceMean",
        "GlobalMaxPool",
        "GlobalAveragePool",
        "Concat",
        "Gemm",
        "MatMul",
        "Add",
        "Constant",
        "Squeeze",
        "Unsqueeze",
        "Reshape",
        "Shape",
    }
    used_ops = {n.op_type for n in model.graph.node}
    unsupported = used_ops - wasm_safe_ops
    assert not unsupported, f"Unsupported non-WASM-safe ops: {unsupported}"


def test_dual_outputs_calibration_parity_and_latency(artifacts):
    calib = artifacts["calib"]
    assert calib["output_names"] == ["logit", "sufficiency_logits"]
    assert calib["sufficiency_classes"] == ["not_address", "address_insufficient", "address_sufficient"]
    assert float(calib["temperature"]) > 0.0
    assert float(calib["sufficiency_temperature"]) > 0.0
    assert calib["metrics"]["onnx_pytorch_max_abs_diff"] < 1e-4

    samples = [
        "Lộc Thạnh, Lộc Ninh",
        "Xã Lộc Thạnh",
        "73",
        "Cho mình 3 chai nướng BBQ",
        "",
        "x" * 200,
    ]
    x_np = encode_batch(samples, artifacts["vocab"], artifacts["max_length"])
    gate_logits, suf_logits = artifacts["sess"].run(["logit", "sufficiency_logits"], {"input_ids": x_np})

    assert gate_logits.shape == (len(samples), 1)
    assert suf_logits.shape == (len(samples), 3)

    gate_probs = sigmoid_np(gate_logits.squeeze(-1) / float(calib["temperature"]))
    suf_probs = softmax_np(suf_logits / float(calib["sufficiency_temperature"]))

    assert np.all(np.isfinite(gate_probs))
    assert np.all((gate_probs > 0.0) & (gate_probs < 1.0))
    assert np.all(np.isfinite(suf_probs))
    assert np.allclose(np.sum(suf_probs, axis=1), 1.0, atol=1e-6)

    # Single-item CPU ORT p99 latency <= 10 ms
    one_in = x_np[:1]
    for _ in range(20):
        artifacts["sess"].run(["logit", "sufficiency_logits"], {"input_ids": one_in})
    lats = []
    for _ in range(150):
        t0 = time.perf_counter()
        artifacts["sess"].run(["logit", "sufficiency_logits"], {"input_ids": one_in})
        lats.append((time.perf_counter() - t0) * 1000.0)
    p99 = float(np.percentile(np.array(lats), 99))
    assert p99 <= 10.0, f"CPU ORT single-item p99 latency {p99:.3f} ms > 10 ms"


def test_vendored_onnxruntime_web_integrity_and_zero_cdn():
    version_file = VENDOR_DIR / "VERSION.txt"
    sums_file = VENDOR_DIR / "SHA256SUMS.txt"
    assert version_file.exists(), "Missing VERSION.txt"
    assert sums_file.exists(), "Missing SHA256SUMS.txt"
    assert (VENDOR_DIR / "ort.wasm.min.js").exists()
    assert (VENDOR_DIR / "ort-wasm-simd-threaded.wasm").exists()
    assert (VENDOR_DIR / "ort-wasm-simd-threaded.mjs").exists()

    recorded = {}
    for line in sums_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, fname = line.split(None, 1)
        recorded[fname.strip()] = digest.strip().lower()

    actual_files = {p.name for p in VENDOR_DIR.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt"}
    assert set(recorded.keys()) == actual_files, f"SHA256SUMS mismatch files: {set(recorded.keys()) ^ actual_files}"

    for fname, expected_sha in recorded.items():
        h = hashlib.sha256((VENDOR_DIR / fname).read_bytes()).hexdigest().lower()
        assert h == expected_sha, f"SHA256 mismatch for {fname}: {h} != {expected_sha}"

    for web_file in (WEB_DIR / "index.html", WEB_DIR / "app.js"):
        content = web_file.read_text(encoding="utf-8")
        assert not re.search(r"https?://", content, flags=re.IGNORECASE), f"External CDN/URL found in {web_file}"


def test_js_syntax_check():
    for js_file in (WEB_DIR / "heuristics.js", WEB_DIR / "app.js"):
        res = subprocess.run(["node", "--check", str(js_file)], capture_output=True, text=True)
        assert res.returncode == 0, f"node --check failed for {js_file}: {res.stderr}"


def test_mandatory_hybrid_smoke_cases(artifacts):
    smoke = [
        # Valid sufficient (all 6 short styles + full)
        ("Lộc Thạnh, Lộc Ninh", "address_sufficient", True),
        ("Lộc Thạnh Lộc Ninh", "address_sufficient", True),
        ("loc thanh loc ninh", "address_sufficient", True),
        ("loc thanh loc ninh binh phuoc", "address_sufficient", True),
        ("x. lộc thạnh, h. lộc ninh", "address_sufficient", True),
        ("x.loc thanh h.loc ninh", "address_sufficient", True),
        ("p.12 q.tb", "address_sufficient", True),
        ("x.hồng dương h.thanh oai hn", "address_sufficient", True),
        ("Lộc Thạnh, Lộc Ninh, Bình Phước", "address_sufficient", True),
        ("Lộc Thạnh, Bình Phước", "address_sufficient", True),
        ("Phường Hồng Hà, Hà Nội", "address_sufficient", True),
        ("loc thanh binh phuoc", "address_sufficient", True),
        ("Số 123 Đường Nguyễn Trãi, Phường Thượng Đình, Quận Thanh Xuân, Hà Nội", "address_sufficient", True),
        ("ngõ 68 cầu giấy", "address_sufficient", True),
        ("Chúc Sơn, Chương Mỹ", "address_sufficient", True),
        ("Khe Sanh, Hướng Hóa, Quảng Trị", "address_sufficient", True),
        # Insufficient fragments
        ("Xã Lộc Thạnh", "address_insufficient", False),
        ("Phường Bến Nghé", "address_insufficient", False),
        ("Thị trấn Đông Anh", "address_insufficient", False),
        ("p.12", "address_insufficient", False),
        ("x.tan lap", "address_insufficient", False),
        ("Hà Nội", "address_insufficient", False),
        ("Tỉnh Bình Phước", "address_insufficient", False),
        ("TP.HCM", "address_insufficient", False),
        ("binh duong", "address_insufficient", False),
        ("Quận 1", "address_insufficient", False),
        ("Huyện Lộc Ninh", "address_insufficient", False),
        ("Thành phố Tam Kỳ", "address_insufficient", False),
        ("Thành phố Phúc Yên", "address_insufficient", False),
        ("Thanh Xuân, Hà Nội", "address_insufficient", False),
        ("Lộc Ninh, Bình Phước", "address_insufficient", False),
        ("ngõ 12", "address_insufficient", False),
        ("Thôn 7", "address_insufficient", False),
        ("thôn 3", "address_insufficient", False),
        ("khu phố 3", "address_insufficient", False),
        ("xóm 4", "address_insufficient", False),
        ("ấp 2", "address_insufficient", False),
        ("tổ 5", "address_insufficient", False),
        ("hẻm 30", "address_insufficient", False),
        ("tầng 4", "address_insufficient", False),
        ("đường Nguyễn Trãi", "address_insufficient", False),
        ("chung cư A1", "address_insufficient", False),
        ("lô B2", "address_insufficient", False),
        # Not address traps
        ("73", "not_address", False),
        ("10", "not_address", False),
        ("K", "not_address", False),
        ("Hi", "not_address", False),
        ("Ok", "not_address", False),
        ("Jean", "not_address", False),
        ("Xom", "not_address", False),
        ("vâng ạ", "not_address", False),
        ("xin chào shop", "not_address", False),
        ("áo khoác kaki", "not_address", False),
        ("túi xách nữ da bò", "not_address", False),
        ("để ngoài cổng giùm", "not_address", False),
        ("bảo vệ ký nhận hộ", "not_address", False),
        ("tai nghe không dây", "not_address", False),
        ("cảm ơn nhiều nha", "not_address", False),
        ("Xin giá", "not_address", False),
        ("Cho mình 3 chai nướng BBQ", "not_address", False),
        ("ok shop", "not_address", False),
        ("ship nhanh giúp", "not_address", False),
        ("Phí ship về Bình Phước bao nhiêu?", "not_address", False),
        ("bánh mì Sài Gòn", "not_address", False),
        ("mực Cà Mau", "not_address", False),
        ("500g mực Cà Mau, Phường 1, Cà Mau", "not_address", False),
        ("2 gói kẹo dừa Bến Tre", "not_address", False),
        ("Bưu cục Quận 1 nghỉ trưa", "not_address", False),
        ("Cửa hàng ở Quận 3 TP.HCM", "not_address", False),
        ("phở Hà Nội", "not_address", False),
        ("bún bò Huế", "not_address", False),
        ("nước mắm Phú Quốc", "not_address", False),
        ("kẹo dừa Bến Tre", "not_address", False),
        ("nem chua Thanh Hóa", "not_address", False),
        ("Nguyễn Trãi", "not_address", False),
        ("Lê Lợi", "not_address", False),
        ("Nguyễn Thị Thanh Xuân", "not_address", False),
        ("Trần Văn Phú", "not_address", False),
        ("Lê Hoàng Mai", "not_address", False),
        ("ở chỗ cũ", "not_address", False),
        ("như đơn lần trước", "not_address", False),
    ]

    texts = [t for t, _, _ in smoke]
    t1_list = _run_node_tier1(texts)
    calib = artifacts["calib"]
    classes = calib["sufficiency_classes"]
    t_gate = float(calib["temperature"])
    thr = float(calib["threshold"])
    t_suf = float(calib["sufficiency_temperature"])

    x_np = encode_batch(texts, artifacts["vocab"], artifacts["max_length"])
    gate_logits, suf_logits = artifacts["sess"].run(["logit", "sufficiency_logits"], {"input_ids": x_np})

    for i, (txt, exp_suf, exp_addr) in enumerate(smoke):
        t1 = t1_list[i]
        if t1["decision"] != "ambiguous":
            pred_addr = bool(t1["is_address"])
            pred_suf = t1["sufficiency"]
        else:
            p_gate = float(sigmoid_np(gate_logits[i, 0] / t_gate))
            pred_addr = p_gate >= thr
            p_suf = softmax_np(suf_logits[i] / t_suf)
            pred_suf = classes[int(np.argmax(p_suf))]

        assert pred_addr == exp_addr, f"Gate mismatch on {txt!r}: got {pred_addr}, expected {exp_addr}"
        assert pred_suf == exp_suf, f"Sufficiency mismatch on {txt!r}: got {pred_suf}, expected {exp_suf}"


def test_train_manifest_has_zero_exact_eval_leakage():
    assert MANIFEST_PATH.exists(), "Missing train/train_texts_manifest.txt"
    train_keys = {norm_key(line) for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert len(train_keys) >= 70_000

    for bench_path in (BENCH_V1_PATH, BENCH_V2_PATH):
        if not bench_path.exists():
            continue
        cases = json.loads(bench_path.read_text(encoding="utf-8")).get("cases", [])
        eval_keys = {norm_key(c["text"]) for c in cases}
        overlap = train_keys & eval_keys
        assert not overlap, f"Leakage detected against {bench_path.name}: {overlap}"


def test_static_server_serves_web_vendor_and_model_headers():
    server = create_server("127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/index.html", timeout=5) as resp:
            assert resp.status == 200
            assert "text/html" in resp.headers.get("Content-Type", "")
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
            assert "Laya for Address VN v2" in resp.read().decode("utf-8")

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/model/laya_address_vn.onnx", timeout=5) as resp:
            assert resp.status == 200
            assert "application/octet-stream" in resp.headers.get("Content-Type", "")
            assert len(resp.read()) == ONNX_PATH.stat().st_size

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/vendor/onnxruntime-web/ort.wasm.min.js", timeout=5) as resp:
            assert resp.status == 200
            assert "application/javascript" in resp.headers.get("Content-Type", "")

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/vendor/onnxruntime-web/ort-wasm-simd-threaded.wasm", timeout=5) as resp:
            assert resp.status == 200
            assert "application/wasm" in resp.headers.get("Content-Type", "")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
