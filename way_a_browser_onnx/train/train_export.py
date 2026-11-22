"""Train, temperature-calibrate, export to ONNX, and validate Laya Address VN."""

from __future__ import annotations

import json
import math
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import CURATED_EDGE_CASES, generate_dataset

# ---------------------------------------------------------------------------
# Deterministic Configuration
# ---------------------------------------------------------------------------

SEED = 42
MAX_LENGTH = 96
EMBED_DIM = 48
BATCH_SIZE = 128
EPOCHS = 12
LR = 2.5e-3
WEIGHT_DECAY = 3e-4
LABEL_SMOOTHING = 0.005

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "web" / "model"
ONNX_PATH = MODEL_DIR / "laya_address_vn.onnx"
VOCAB_PATH = MODEL_DIR / "vocab.json"
CALIB_PATH = MODEL_DIR / "calibration.json"
PT_WEIGHTS_PATH = BASE_DIR / "train" / "laya_address_vn.pt"

# Compact character vocabulary: ASCII lowercase + digits + punctuation + VN accented chars
VN_ACCENTED_CHARS = (
    "àáảãạăằắẳẵặâầấẩẫậ"
    "èéẻẽẹêềếểễệ"
    "ìíỉĩị"
    "òóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữự"
    "ỳýỷỹỵđ"
)
ASCII_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789.,/-:#@+()&%_?!'\""


def build_vocab() -> Dict[str, int]:
    vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for ch in ASCII_CHARS + VN_ACCENTED_CHARS:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"\s+", " ", text.strip()).lower()
    return text


def encode_text(text: str, vocab: Dict[str, int], max_length: int = MAX_LENGTH) -> List[int]:
    norm = normalize_text(text)
    unk = vocab["<UNK>"]
    ids = [vocab.get(ch, unk) for ch in norm[:max_length]]
    if len(ids) < max_length:
        ids.extend([0] * (max_length - len(ids)))
    return ids


def encode_batch(texts: Sequence[str], vocab: Dict[str, int], max_length: int = MAX_LENGTH) -> np.ndarray:
    arr = np.zeros((len(texts), max_length), dtype=np.int64)
    for i, t in enumerate(texts):
        arr[i] = encode_text(t, vocab, max_length)
    return arr


# ---------------------------------------------------------------------------
# Laya Noul (Boolean) Multi-Scale Char-CNN Model
# Uses strictly WASM-safe ONNX ops:
#   Gather, Transpose, Conv, Relu, ReduceMax, Concat, Gemm, Add
# ---------------------------------------------------------------------------

class LayaAddressNoulNet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv2 = nn.Conv1d(embed_dim, 48, kernel_size=2, padding=0)
        self.conv3 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(embed_dim, 64, kernel_size=7, padding=3)

        # Two-stage contextual conv branch for compositional phrases (e.g. delivery vs chat)
        self.deep_conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.deep_conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.30)

        pooled_dim = 48 + 64 + 64 + 64 + 64  # 304
        self.fc1 = nn.Linear(pooled_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.head = nn.Linear(32, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, L] int64 -> [B, E, L]
        x = self.embedding(input_ids).transpose(1, 2)

        p2 = torch.amax(self.relu(self.conv2(x)), dim=2)
        p3 = torch.amax(self.relu(self.conv3(x)), dim=2)
        p5 = torch.amax(self.relu(self.conv5(x)), dim=2)
        p7 = torch.amax(self.relu(self.conv7(x)), dim=2)

        d = self.relu(self.deep_conv1(x))
        pd = torch.amax(self.relu(self.deep_conv2(d)), dim=2)

        feat = torch.cat([p2, p3, p5, p7, pd], dim=1)
        h = self.dropout(self.relu(self.fc1(feat)))
        h = self.dropout(self.relu(self.fc2(h)))
        logit = self.head(h)  # [B, 1]
        return logit


# ---------------------------------------------------------------------------
# Metrics & Laya Temperature Calibration
# ---------------------------------------------------------------------------

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x64 = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x64))


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            acc = float(np.mean(labels[mask]))
            conf = float(np.mean(probs[mask]))
            ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def compute_metrics(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0, threshold: float = 0.5) -> Dict[str, float]:
    probs = sigmoid_np(logits / temperature)
    preds = (probs >= threshold).astype(np.int64)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    acc = (tp + tn) / max(1, len(labels))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    eps = 1e-7
    p_clip = np.clip(probs, eps, 1.0 - eps)
    nll = float(-np.mean(labels * np.log(p_clip) + (1.0 - labels) * np.log(1.0 - p_clip)))
    brier = float(np.mean((probs - labels) ** 2))
    ece = compute_ece(probs, labels)
    return {
        "accuracy": round(float(acc), 5),
        "precision": round(float(prec), 5),
        "recall": round(float(rec), 5),
        "f1": round(float(f1), 5),
        "nll": round(nll, 5),
        "brier": round(brier, 5),
        "ece": round(ece, 5),
    }


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Fit positive scalar temperature T minimizing Laplace-regularized validation NLL.

    Uses finite-sample Jeffreys/Laplace target smoothing eps = 1 / (2 * N_val) so T
    properly cools overconfident logits even when validation accuracy is near 100%,
    guaranteeing well-calibrated probabilities strictly inside (0, 1).
    """
    logits_t = torch.tensor(val_logits, dtype=torch.float64)
    n_val = max(1, len(val_labels))
    eps = 1.0 / (2.0 * n_val)
    labels_cal = val_labels.astype(np.float64) * (1.0 - 2.0 * eps) + eps
    labels_t = torch.tensor(labels_cal, dtype=torch.float64)

    log_temp = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.LBFGS([log_temp], lr=0.2, max_iter=150, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temp = torch.exp(log_temp).clamp(min=0.25, max=10.0)
        loss = loss_fn(logits_t / temp, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    best_t = float(torch.exp(log_temp).clamp(min=0.25, max=10.0).item())
    return round(best_t, 5)


# ---------------------------------------------------------------------------
# Training + Calibration + ONNX Export Pipeline
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict_torch(model: nn.Module, x_np: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(x_np), batch_size):
            xb = torch.from_numpy(x_np[i : i + batch_size])
            outs.append(model(xb).squeeze(-1).cpu().numpy())
    return np.concatenate(outs, axis=0)


def train_and_export() -> Dict[str, object]:
    set_seed(SEED)
    torch.set_num_threads(min(8, os.cpu_count() or 4))

    print("[1/5] Generating Vietnamese address dataset...")
    splits = generate_dataset(n_per_class=9000, seed=SEED)
    print(f"      Splits -> train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)}")

    vocab = build_vocab()
    print(f"      Vocab size={len(vocab)}, max_length={MAX_LENGTH}")

    x_train = encode_batch([t for t, _ in splits.train], vocab)
    y_train = np.array([y for _, y in splits.train], dtype=np.float32)
    x_val = encode_batch([t for t, _ in splits.val], vocab)
    y_val = np.array([y for _, y in splits.val], dtype=np.float32)
    x_test = encode_batch([t for t, _ in splits.test], vocab)
    y_test = np.array([y for _, y in splits.test], dtype=np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = LayaAddressNoulNet(vocab_size=len(vocab), embed_dim=EMBED_DIM)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[2/5] Training LayaAddressNoulNet ({param_count:,} params) for {EPOCHS} epochs...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)

    best_val_nll = math.inf
    best_state = None
    t0 = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            yb_smooth = yb * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            loss = criterion(logits, yb_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
        scheduler.step()

        val_logits = predict_torch(model, x_val)
        val_m = compute_metrics(val_logits, y_val, temperature=1.0)
        if val_m["nll"] < best_val_nll:
            best_val_nll = val_m["nll"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch in (1, 4, 8, EPOCHS) or val_m["nll"] == best_val_nll:
            print(
                f"      Epoch {epoch:02d}/{EPOCHS} | train_loss={total_loss/len(x_train):.4f} "
                f"| val_acc={val_m['accuracy']:.4f} | val_f1={val_m['f1']:.4f} | val_nll={val_m['nll']:.4f}"
            )

    train_secs = time.perf_counter() - t0
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    print(f"      Training complete in {train_secs:.1f}s (best val_nll={best_val_nll:.5f})")

    # Save PyTorch weights for parity testing
    PT_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, PT_WEIGHTS_PATH)

    print("[3/5] Fitting Laya temperature calibration on validation split...")
    val_logits = predict_torch(model, x_val)
    val_uncal = compute_metrics(val_logits, y_val, temperature=1.0)
    temperature = fit_temperature(val_logits, y_val)
    val_cal = compute_metrics(val_logits, y_val, temperature=temperature)
    print(f"      Temperature T={temperature:.5f} | val ECE: {val_uncal['ece']:.5f} -> {val_cal['ece']:.5f} | val NLL: {val_uncal['nll']:.5f} -> {val_cal['nll']:.5f}")

    print("[4/5] Exporting single self-contained ONNX model (opset 14, WASM-safe)...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Clean any stale external data file
    ext_data = Path(str(ONNX_PATH) + ".data")
    if ext_data.exists():
        ext_data.unlink()

    dummy_input = torch.zeros((1, MAX_LENGTH), dtype=torch.int64)
    torch.onnx.export(
        model,
        (dummy_input,),
        str(ONNX_PATH),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logit"],
        dynamic_axes={"input_ids": {0: "batch_size"}, "logit": {0: "batch_size"}},
        dynamo=False,
    )

    # Ensure strictly self-contained single protobuf file (no external .onnx.data)
    onnx_model = onnx.load(str(ONNX_PATH), load_external_data=True)
    onnx.checker.check_model(onnx_model)
    onnx.save_model(onnx_model, str(ONNX_PATH), save_as_external_data=False)
    if ext_data.exists():
        ext_data.unlink()

    ops_used = sorted({node.op_type for node in onnx_model.graph.node})
    model_size_bytes = ONNX_PATH.stat().st_size
    print(f"      Exported: {ONNX_PATH} ({model_size_bytes / 1024:.1f} KB)")
    print(f"      ONNX opset={onnx_model.opset_import[0].version}, ops={ops_used}")

    print("[5/5] Validating ONNX Runtime inference, test metrics, and curated edge cases...")
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    # Numerical parity check on test set
    pt_test_logits = predict_torch(model, x_test)
    ort_test_logits = sess.run(["logit"], {"input_ids": x_test})[0].squeeze(-1)
    max_abs_diff = float(np.max(np.abs(pt_test_logits - ort_test_logits)))
    test_metrics = compute_metrics(ort_test_logits, y_test, temperature=temperature)
    test_uncal = compute_metrics(ort_test_logits, y_test, temperature=1.0)

    # Latency benchmark (single-sample, 200 runs)
    sample_in = x_test[:1]
    for _ in range(10):
        sess.run(["logit"], {"input_ids": sample_in})
    t_lat = time.perf_counter()
    n_bench = 200
    for _ in range(n_bench):
        sess.run(["logit"], {"input_ids": sample_in})
    avg_latency_ms = (time.perf_counter() - t_lat) * 1000.0 / n_bench

    # Curated edge-case evaluation
    curated_texts = [t for t, _, _ in CURATED_EDGE_CASES]
    curated_labels = np.array([y for _, y, _ in CURATED_EDGE_CASES], dtype=np.int64)
    x_curated = encode_batch(curated_texts, vocab)
    curated_logits = sess.run(["logit"], {"input_ids": x_curated})[0].squeeze(-1)
    curated_probs = sigmoid_np(curated_logits / temperature)
    curated_preds = (curated_probs >= 0.5).astype(np.int64)
    curated_correct = int(np.sum(curated_preds == curated_labels))
    curated_acc = curated_correct / len(curated_labels)

    print(f"      PyTorch vs ONNX max_abs_diff: {max_abs_diff:.2e}")
    print(
        f"      Test -> acc={test_metrics['accuracy']:.4f}, f1={test_metrics['f1']:.4f}, "
        f"ece={test_metrics['ece']:.5f} (uncal={test_uncal['ece']:.5f}), nll={test_metrics['nll']:.5f}"
    )
    print(f"      Curated edge cases: {curated_correct}/{len(curated_labels)} ({curated_acc*100:.1f}%)")
    print(f"      CPU ONNX single-sample latency: {avg_latency_ms:.3f} ms")

    for (txt, gold, tag), prob, pred in zip(CURATED_EDGE_CASES, curated_probs, curated_preds):
        mark = "OK" if int(pred) == gold else "MISS"
        print(f"        [{mark:4s}] p={prob:6.3f} gold={gold} ({tag:22s}) | {txt}")

    # Export vocab.json
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pad_id": 0,
                "unk_id": 1,
                "max_length": MAX_LENGTH,
                "vocab_size": len(vocab),
                "char_to_id": vocab,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Export calibration.json
    calib_payload = {
        "model_name": "Laya Address VN (Noul Char-CNN)",
        "architecture": "Multi-Scale Char-CNN (k=2,3,5,7 + 2-layer contextual conv) + GlobalMaxPool + MLP",
        "opset": 14,
        "ops": ops_used,
        "param_count": param_count,
        "model_size_bytes": model_size_bytes,
        "max_length": MAX_LENGTH,
        "temperature": temperature,
        "threshold": 0.5,
        "metrics": {
            "val_uncalibrated": val_uncal,
            "val_calibrated": val_cal,
            "test": test_metrics,
            "test_uncalibrated_ece": test_uncal["ece"],
            "curated_edge_accuracy": round(curated_acc, 5),
            "curated_edge_correct": curated_correct,
            "curated_edge_total": len(curated_labels),
            "onnx_pytorch_max_abs_diff": max_abs_diff,
            "cpu_onnx_latency_ms": round(avg_latency_ms, 3),
        },
        "dataset": {
            "train_size": len(splits.train),
            "val_size": len(splits.val),
            "test_size": len(splits.test),
            "seed": SEED,
        },
    }
    with open(CALIB_PATH, "w", encoding="utf-8") as f:
        json.dump(calib_payload, f, ensure_ascii=False, indent=2)

    print(f"      Wrote {VOCAB_PATH} and {CALIB_PATH}")
    return calib_payload


if __name__ == "__main__":
    train_and_export()
