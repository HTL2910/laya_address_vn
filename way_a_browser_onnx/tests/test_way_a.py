"""Verification test suite for Way A: Laya for Address VN (Browser ONNX)."""

from __future__ import annotations

import json
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT_DIR / "train"
WEB_DIR = ROOT_DIR / "web"
MODEL_DIR = WEB_DIR / "model"
ONNX_PATH = MODEL_DIR / "laya_address_vn.onnx"
VOCAB_PATH = MODEL_DIR / "vocab.json"
CALIB_PATH = MODEL_DIR / "calibration.json"
PT_WEIGHTS_PATH = TRAIN_DIR / "laya_address_vn.pt"

sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(ROOT_DIR))

from dataset import CURATED_EDGE_CASES, generate_dataset  # noqa: E402
from serve import create_server  # noqa: E402
from train_export import (  # noqa: E402
    EMBED_DIM,
    LayaAddressNoulNet,
    encode_batch,
    sigmoid_np,
)


@pytest.fixture(scope="module")
def artifacts():
    assert ONNX_PATH.exists(), f"Missing ONNX model: {ONNX_PATH}"
    assert VOCAB_PATH.exists(), f"Missing vocab.json: {VOCAB_PATH}"
    assert CALIB_PATH.exists(), f"Missing calibration.json: {CALIB_PATH}"

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        calib_data = json.load(f)

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    return {
        "vocab_data": vocab_data,
        "vocab": vocab_data["char_to_id"],
        "max_length": vocab_data["max_length"],
        "calib": calib_data,
        "sess": sess,
    }


def test_onnx_file_exists_self_contained_and_compact():
    assert ONNX_PATH.exists(), "laya_address_vn.onnx must exist"
    ext_data = Path(str(ONNX_PATH) + ".data")
    assert not ext_data.exists(), "ONNX model must be self-contained (no .onnx.data file)"

    size_bytes = ONNX_PATH.stat().st_size
    assert size_bytes > 10_000, "ONNX model unexpectedly small"
    assert size_bytes < 5 * 1024 * 1024, f"ONNX model exceeds 5 MB: {size_bytes}"
    assert size_bytes < 2 * 1024 * 1024, f"ONNX model exceeds 2 MB target: {size_bytes}"

    model = onnx.load(str(ONNX_PATH), load_external_data=False)
    onnx.checker.check_model(model)
    for init in model.graph.initializer:
        assert init.data_location != onnx.TensorProto.EXTERNAL, f"Initializer {init.name} uses external data"

    wasm_safe_ops = {
        "Gather",
        "Transpose",
        "Conv",
        "Relu",
        "ReduceMax",
        "GlobalMaxPool",
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
    used_ops = {node.op_type for node in model.graph.node}
    unsupported = used_ops - wasm_safe_ops
    assert not unsupported, f"Unexpected non-WASM-safe ONNX ops: {unsupported}"


def test_onnx_matches_pytorch_within_1e4(artifacts):
    assert PT_WEIGHTS_PATH.exists(), "PyTorch weights file missing for parity check"
    vocab = artifacts["vocab"]
    max_len = artifacts["max_length"]

    model = LayaAddressNoulNet(vocab_size=len(vocab), embed_dim=EMBED_DIM)
    state = torch.load(PT_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    samples = [t for t, _, _ in CURATED_EDGE_CASES]
    x_np = encode_batch(samples, vocab, max_len)

    with torch.no_grad():
        pt_logits = model(torch.from_numpy(x_np)).squeeze(-1).numpy()

    ort_logits = artifacts["sess"].run(["logit"], {"input_ids": x_np})[0].squeeze(-1)
    max_diff = float(np.max(np.abs(pt_logits - ort_logits)))
    assert max_diff < 1e-4, f"ONNX vs PyTorch diff too large: {max_diff}"


def test_calibration_and_probability_bounds(artifacts):
    calib = artifacts["calib"]
    temp = float(calib["temperature"])
    assert temp > 0.0, f"Temperature T must be > 0, got {temp}"
    assert calib["model_size_bytes"] == ONNX_PATH.stat().st_size

    samples = [t for t, _, _ in CURATED_EDGE_CASES] + ["", "a", "x" * 200]
    x_np = encode_batch(samples, artifacts["vocab"], artifacts["max_length"])
    logits = artifacts["sess"].run(["logit"], {"input_ids": x_np})[0].squeeze(-1)
    probs = sigmoid_np(logits / temp)

    assert np.all(np.isfinite(probs)), "Probabilities must be finite"
    assert np.all((probs > 0.0) & (probs < 1.0)), f"Probabilities must be strictly in (0, 1): min={probs.min()}, max={probs.max()}"


def test_curated_vietnamese_edge_cases(artifacts):
    vocab = artifacts["vocab"]
    max_len = artifacts["max_length"]
    temp = float(artifacts["calib"]["temperature"])
    threshold = float(artifacts["calib"].get("threshold", 0.5))

    texts = [t for t, _, _ in CURATED_EDGE_CASES]
    labels = np.array([y for _, y, _ in CURATED_EDGE_CASES], dtype=np.int64)
    x_np = encode_batch(texts, vocab, max_len)

    logits = artifacts["sess"].run(["logit"], {"input_ids": x_np})[0].squeeze(-1)
    probs = sigmoid_np(logits / temp)
    preds = (probs >= threshold).astype(np.int64)

    acc = float(np.mean(preds == labels))
    misses = [
        (tag, txt, int(gold), float(p))
        for (txt, gold, tag), pred, p in zip(CURATED_EDGE_CASES, preds, probs)
        if int(pred) != gold
    ]
    assert len(CURATED_EDGE_CASES) >= 20
    assert acc >= 0.90, f"Curated VN edge-case accuracy {acc:.3f} < 0.90. Misses: {misses}"


def test_held_out_test_split_accuracy(artifacts):
    splits = generate_dataset(n_per_class=9000, seed=42)
    texts = [t for t, _ in splits.test]
    labels = np.array([y for _, y in splits.test], dtype=np.int64)

    x_np = encode_batch(texts, artifacts["vocab"], artifacts["max_length"])
    logits = artifacts["sess"].run(["logit"], {"input_ids": x_np})[0].squeeze(-1)
    probs = sigmoid_np(logits / float(artifacts["calib"]["temperature"]))
    preds = (probs >= float(artifacts["calib"].get("threshold", 0.5))).astype(np.int64)
    acc = float(np.mean(preds == labels))
    assert acc >= 0.95, f"Held-out test accuracy {acc:.4f} < 0.95"


def test_static_server_serves_web_and_onnx_headers():
    server = create_server("127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/index.html", timeout=5) as resp:
            assert resp.status == 200
            assert "text/html" in resp.headers.get("Content-Type", "")
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
            body = resp.read().decode("utf-8")
            assert "Laya for Address VN" in body

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/model/laya_address_vn.onnx", timeout=5) as resp:
            assert resp.status == 200
            assert "application/octet-stream" in resp.headers.get("Content-Type", "")
            raw = resp.read()
            assert len(raw) == ONNX_PATH.stat().st_size
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
