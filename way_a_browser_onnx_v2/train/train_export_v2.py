"""Train, dual-calibrate, export to ONNX, and benchmark Way A v2 (Dual-Output Char-CNN)."""

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
import torch.nn.functional as F

from dataset_v2 import LABEL_NAMES, SEED, Sample, generate_dataset_v2

# ---------------------------------------------------------------------------
# Deterministic Configuration
# ---------------------------------------------------------------------------

MAX_LENGTH = 128
EMBED_DIM = 64
BATCH_SIZE = 256
EPOCHS = 10
LR = 3.5e-3
WEIGHT_DECAY = 2e-4

TRAIN_DIR = Path(__file__).resolve().parent
BASE_DIR = TRAIN_DIR.parent
MODEL_DIR = BASE_DIR / "web" / "model"
ONNX_PATH = MODEL_DIR / "laya_address_vn.onnx"
GAZETTEER_PATH = MODEL_DIR / "vn_admin_gazetteer.json"
VOCAB_PATH = MODEL_DIR / "vocab.json"
CALIB_PATH = MODEL_DIR / "calibration.json"

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


def sample_weight_for(sample: Sample) -> float:
    sg = sample.subgroup
    if sg.startswith("short_style"):
        return 1.55
    if sg.startswith("insuff_"):
        return 1.45
    if sg == "neg_food_product_place":
        return 1.60
    if sg == "neg_narrative_weather_travel":
        return 1.50
    if sg in ("neg_chat_order", "neg_person_name"):
        return 1.45
    if sg in ("neg_short_garbage", "neg_vague_non_addr"):
        return 1.35
    if sg in ("full_delivery_freetext", "full_noisy_case"):
        return 1.20
    return 1.00


# ---------------------------------------------------------------------------
# Dual-Output Multi-Scale Char-CNN (~303k params, <= 1.5 MB ONNX FP32)
# Uses strictly WASM-safe ONNX ops:
#   Gather, Transpose, Conv, Relu, ReduceMax, ReduceMean, Concat, Gemm, Add
# ---------------------------------------------------------------------------


class LayaAddressSufficiencyCharCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv2 = nn.Conv1d(embed_dim, 32, kernel_size=2, padding=0)
        self.conv3 = nn.Conv1d(embed_dim, 48, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, 48, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(embed_dim, 48, kernel_size=7, padding=3)

        self.deep_conv1 = nn.Conv1d(embed_dim, 48, kernel_size=3, padding=1)
        self.deep_conv2 = nn.Conv1d(48, 64, kernel_size=5, padding=2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.22)

        branch_channels = 32 + 48 + 48 + 48 + 64  # 240
        pooled_dim = branch_channels * 2  # 480 (GlobalMaxPool + GlobalMeanPool)
        self.fc1 = nn.Linear(pooled_dim, 320)
        self.fc2 = nn.Linear(320, 128)
        self.gate_head = nn.Linear(128, 1)
        self.sufficiency_head = nn.Linear(128, 3)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_ids: [B, L] int64 -> [B, E, L]
        x = self.embedding(input_ids).transpose(1, 2)

        c2 = self.relu(self.conv2(x))
        c3 = self.relu(self.conv3(x))
        c5 = self.relu(self.conv5(x))
        c7 = self.relu(self.conv7(x))
        cd = self.relu(self.deep_conv2(self.relu(self.deep_conv1(x))))

        max_pool = torch.cat(
            [
                torch.amax(c2, dim=2),
                torch.amax(c3, dim=2),
                torch.amax(c5, dim=2),
                torch.amax(c7, dim=2),
                torch.amax(cd, dim=2),
            ],
            dim=1,
        )
        mean_pool = torch.cat(
            [
                torch.mean(c2, dim=2),
                torch.mean(c3, dim=2),
                torch.mean(c5, dim=2),
                torch.mean(c7, dim=2),
                torch.mean(cd, dim=2),
            ],
            dim=1,
        )
        feat = torch.cat([max_pool, mean_pool], dim=1)
        h = self.dropout(self.relu(self.fc1(feat)))
        h = self.dropout(self.relu(self.fc2(h)))
        logit = self.gate_head(h)  # [B, 1]
        sufficiency_logits = self.sufficiency_head(h)  # [B, 3]
        return logit, sufficiency_logits


# ---------------------------------------------------------------------------
# Calibration & Evaluation Metrics
# ---------------------------------------------------------------------------


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x64 = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x64))


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(np.clip(z, -50.0, 50.0))
    return e / np.sum(e, axis=-1, keepdims=True)


def compute_ece_binary(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(1, len(labels))
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            ece += (cnt / n) * abs(float(np.mean(labels[mask])) - float(np.mean(probs[mask])))
    return float(ece)


def compute_ece_multiclass(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accs = (preds == labels).astype(np.float64)
    return compute_ece_binary(confs, accs, n_bins=n_bins)


def compute_binary_metrics(
    logits: np.ndarray,
    labels_bin: np.ndarray,
    temperature: float = 1.0,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probs = sigmoid_np(logits / temperature)
    preds = (probs >= threshold).astype(np.int64)
    tp = int(np.sum((preds == 1) & (labels_bin == 1)))
    tn = int(np.sum((preds == 0) & (labels_bin == 0)))
    fp = int(np.sum((preds == 1) & (labels_bin == 0)))
    fn = int(np.sum((preds == 0) & (labels_bin == 1)))
    acc = (tp + tn) / max(1, len(labels_bin))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    eps = 1e-7
    p_clip = np.clip(probs, eps, 1.0 - eps)
    nll = float(-np.mean(labels_bin * np.log(p_clip) + (1.0 - labels_bin) * np.log(1.0 - p_clip)))
    brier = float(np.mean((probs - labels_bin) ** 2))
    ece = compute_ece_binary(probs, labels_bin)
    return {
        "accuracy": round(float(acc), 5),
        "precision": round(float(prec), 5),
        "recall": round(float(rec), 5),
        "f1": round(float(f1), 5),
        "nll": round(nll, 5),
        "brier": round(brier, 5),
        "ece": round(ece, 5),
    }


def compute_multiclass_metrics(
    suf_logits: np.ndarray,
    labels_cls: np.ndarray,
    temperature: float = 1.0,
) -> Dict[str, float]:
    probs = softmax_np(suf_logits / temperature)
    preds = np.argmax(probs, axis=1)
    acc = float(np.mean(preds == labels_cls))

    f1s: List[float] = []
    for c in (0, 1, 2):
        tp = int(np.sum((preds == c) & (labels_cls == c)))
        fp = int(np.sum((preds == c) & (labels_cls != c)))
        fn = int(np.sum((preds != c) & (labels_cls == c)))
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1s.append(2 * p * r / max(1e-12, p + r))
    macro_f1 = float(np.mean(f1s))

    eps = 1e-7
    p_true = np.clip(probs[np.arange(len(labels_cls)), labels_cls], eps, 1.0)
    nll = float(-np.mean(np.log(p_true)))
    one_hot = np.eye(3, dtype=np.float64)[labels_cls]
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    ece = compute_ece_multiclass(probs, labels_cls)
    return {
        "accuracy": round(acc, 5),
        "macro_f1": round(macro_f1, 5),
        "f1_not_address": round(f1s[0], 5),
        "f1_insufficient": round(f1s[1], 5),
        "f1_sufficient": round(f1s[2], 5),
        "nll": round(nll, 5),
        "brier": round(brier, 5),
        "ece": round(ece, 5),
    }


def fit_binary_temperature(val_logits: np.ndarray, val_labels_bin: np.ndarray) -> float:
    logits_t = torch.tensor(val_logits, dtype=torch.float64)
    n_val = max(1, len(val_labels_bin))
    eps = 1.0 / (2.0 * n_val)
    labels_cal = val_labels_bin.astype(np.float64) * (1.0 - 2.0 * eps) + eps
    labels_t = torch.tensor(labels_cal, dtype=torch.float64)

    log_temp = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.LBFGS([log_temp], lr=0.2, max_iter=150, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        temp = torch.exp(log_temp).clamp(min=0.25, max=10.0)
        loss = loss_fn(logits_t / temp, labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return round(float(torch.exp(log_temp).clamp(min=0.25, max=10.0).item()), 5)


def fit_multiclass_temperature(val_logits: np.ndarray, val_labels_cls: np.ndarray) -> float:
    logits_t = torch.tensor(val_logits, dtype=torch.float64)
    n_val = max(1, len(val_labels_cls))
    eps = 1.0 / (2.0 * n_val)
    one_hot = np.eye(3, dtype=np.float64)[val_labels_cls]
    targets_cal = one_hot * (1.0 - eps) + (eps / 3.0)
    targets_t = torch.tensor(targets_cal, dtype=torch.float64)

    log_temp = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    opt = torch.optim.LBFGS([log_temp], lr=0.2, max_iter=150, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        temp = torch.exp(log_temp).clamp(min=0.25, max=10.0)
        log_probs = F.log_softmax(logits_t / temp, dim=-1)
        loss = -(targets_t * log_probs).sum(dim=-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return round(float(torch.exp(log_temp).clamp(min=0.25, max=10.0).item()), 5)


def tune_binary_threshold(
    val_logits: np.ndarray,
    val_samples: Sequence[Sample],
    temperature: float,
) -> Tuple[float, Dict[str, float]]:
    probs = sigmoid_np(val_logits / temperature)
    y_bin = np.array([1 if s.label == 2 else 0 for s in val_samples], dtype=np.int64)
    is_short = np.array([s.subgroup.startswith("short_style") for s in val_samples], dtype=bool)
    is_garbage = np.array([s.label == 0 for s in val_samples], dtype=bool)

    best_thr = 0.5
    best_score = -1.0
    best_stats: Dict[str, float] = {}

    for thr in np.linspace(0.30, 0.70, 81):
        preds = (probs >= thr).astype(np.int64)
        short_fb = float(np.mean(preds[is_short] == 0)) if np.any(is_short) else 0.0
        garb_blk = float(np.mean(preds[is_garbage] == 0)) if np.any(is_garbage) else 1.0
        tp = int(np.sum((preds == 1) & (y_bin == 1)))
        fp = int(np.sum((preds == 1) & (y_bin == 0)))
        fn = int(np.sum((preds == 0) & (y_bin == 1)))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        feasible = (short_fb <= 0.05) and (garb_blk >= 0.95)
        score = (10.0 if feasible else 0.0) + f1 - 0.01 * abs(float(thr) - 0.5)
        if score > best_score:
            best_score = score
            best_thr = round(float(thr), 4)
            best_stats = {
                "short_address_false_block_rate": round(short_fb, 5),
                "garbage_block_rate": round(garb_blk, 5),
                "val_f1_at_threshold": round(float(f1), 5),
                "constraints_satisfied": feasible,
            }

    return best_thr, best_stats


# ---------------------------------------------------------------------------
# Training, Dual Calibration, ONNX Export & Verification
# ---------------------------------------------------------------------------


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict_torch(
    model: nn.Module, x_np: np.ndarray, batch_size: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    bin_outs: List[np.ndarray] = []
    suf_outs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(x_np), batch_size):
            xb = torch.from_numpy(x_np[i : i + batch_size])
            logit, suf = model(xb)
            bin_outs.append(logit.squeeze(-1).cpu().numpy())
            suf_outs.append(suf.cpu().numpy())
    return np.concatenate(bin_outs, axis=0), np.concatenate(suf_outs, axis=0)


def train_and_export() -> Dict[str, object]:
    set_seed(SEED)
    torch.set_num_threads(min(8, os.cpu_count() or 8))

    print("[1/5] Generating 90,000-sample v2 dataset (MIT GSO + 2025 2-tier, seed=101)...")
    splits = generate_dataset_v2(total_samples=90_000, seed=SEED, write_manifest=True)
    print(f"      Splits -> train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)}")

    vocab = build_vocab()
    print(f"      Vocab size={len(vocab)}, max_length={MAX_LENGTH}, embed_dim={EMBED_DIM}")

    x_train = encode_batch([s.text for s in splits.train], vocab)
    y_train_cls = np.array([s.label for s in splits.train], dtype=np.int64)
    y_train_bin = (y_train_cls == 2).astype(np.float32)
    w_train = np.array([sample_weight_for(s) for s in splits.train], dtype=np.float32)

    x_val = encode_batch([s.text for s in splits.val], vocab)
    y_val_cls = np.array([s.label for s in splits.val], dtype=np.int64)
    y_val_bin = (y_val_cls == 2).astype(np.float32)

    x_test = encode_batch([s.text for s in splits.test], vocab)
    y_test_cls = np.array([s.label for s in splits.test], dtype=np.int64)
    y_test_bin = (y_test_cls == 2).astype(np.float32)

    x_train_t = torch.from_numpy(x_train)
    y_bin_t = torch.from_numpy(y_train_bin)
    y_cls_t = torch.from_numpy(y_train_cls)
    w_train_t = torch.from_numpy(w_train)
    n_train = len(x_train_t)

    model = LayaAddressSufficiencyCharCNN(vocab_size=len(vocab), embed_dim=EMBED_DIM)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[2/5] Training LayaAddressSufficiencyCharCNN ({param_count:,} params) for {EPOCHS} epochs...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)

    best_val_loss = math.inf
    best_state = None
    t0 = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        perm = torch.randperm(n_train)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            xb = x_train_t[idx]
            yb_bin = y_bin_t[idx]
            yb_cls = y_cls_t[idx]
            wb = w_train_t[idx]
            optimizer.zero_grad()
            logit, suf_logits = model(xb)
            bce = F.binary_cross_entropy_with_logits(logit.squeeze(-1), yb_bin, reduction="none")
            ce = F.cross_entropy(suf_logits, yb_cls, reduction="none")
            loss = ((bce + ce) * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
        scheduler.step()

        val_bin_logits, val_suf_logits = predict_torch(model, x_val)
        val_bin_m = compute_binary_metrics(val_bin_logits, y_val_bin, temperature=1.0)
        val_suf_m = compute_multiclass_metrics(val_suf_logits, y_val_cls, temperature=1.0)
        joint_val_nll = val_bin_m["nll"] + val_suf_m["nll"]
        if joint_val_nll < best_val_loss:
            best_val_loss = joint_val_nll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"      Epoch {epoch:02d}/{EPOCHS} | loss={total_loss/len(x_train):.4f} "
            f"| val_bin_acc={val_bin_m['accuracy']:.4f} | val_suf_acc={val_suf_m['accuracy']:.4f} "
            f"| val_joint_nll={joint_val_nll:.4f}"
        )

    train_secs = time.perf_counter() - t0
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    print(f"      Training complete in {train_secs:.1f}s (best joint_val_nll={best_val_loss:.5f})")

    print("[3/5] Fitting dual temperatures (T and T_suf) & tuning binary threshold...")
    val_bin_logits, val_suf_logits = predict_torch(model, x_val)
    val_bin_pre = compute_binary_metrics(val_bin_logits, y_val_bin, temperature=1.0)
    val_suf_pre = compute_multiclass_metrics(val_suf_logits, y_val_cls, temperature=1.0)

    temperature = fit_binary_temperature(val_bin_logits, y_val_bin)
    suf_temperature = fit_multiclass_temperature(val_suf_logits, y_val_cls)
    threshold, thr_stats = tune_binary_threshold(val_bin_logits, splits.val, temperature)

    val_bin_post = compute_binary_metrics(val_bin_logits, y_val_bin, temperature=temperature, threshold=threshold)
    val_suf_post = compute_multiclass_metrics(val_suf_logits, y_val_cls, temperature=suf_temperature)

    print(
        f"      Gate T={temperature:.5f}, threshold={threshold:.4f} "
        f"| short_false_block={thr_stats['short_address_false_block_rate']:.4f}, "
        f"garbage_block={thr_stats['garbage_block_rate']:.4f}"
    )
    print(f"      Gate val ECE: {val_bin_pre['ece']:.5f} -> {val_bin_post['ece']:.5f} | NLL: {val_bin_pre['nll']:.5f} -> {val_bin_post['nll']:.5f}")
    print(f"      3-Class T_suf={suf_temperature:.5f} | val ECE: {val_suf_pre['ece']:.5f} -> {val_suf_post['ece']:.5f} | NLL: {val_suf_pre['nll']:.5f} -> {val_suf_post['nll']:.5f}")

    print("[4/5] Exporting self-contained dual-output ONNX model (opset 14)...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
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
        output_names=["logit", "sufficiency_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "logit": {0: "batch_size"},
            "sufficiency_logits": {0: "batch_size"},
        },
        dynamo=False,
    )

    onnx_model = onnx.load(str(ONNX_PATH), load_external_data=True)
    onnx.checker.check_model(onnx_model)
    onnx.save_model(onnx_model, str(ONNX_PATH), save_as_external_data=False)
    if ext_data.exists():
        ext_data.unlink()

    ops_used = sorted({node.op_type for node in onnx_model.graph.node})
    model_size_bytes = ONNX_PATH.stat().st_size
    print(f"      Exported: {ONNX_PATH} ({model_size_bytes / 1024:.1f} KB)")
    print(f"      Ops: {ops_used}")

    print("[5/5] Verifying Torch vs ONNX parity (in-memory), test metrics & latency...")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(ONNX_PATH), sess_options=sess_opts, providers=["CPUExecutionProvider"])

    pt_test_bin, pt_test_suf = predict_torch(model, x_test)
    ort_bin_raw, ort_suf = sess.run(["logit", "sufficiency_logits"], {"input_ids": x_test})
    ort_test_bin = ort_bin_raw.squeeze(-1)

    max_diff_bin = float(np.max(np.abs(pt_test_bin - ort_test_bin)))
    max_diff_suf = float(np.max(np.abs(pt_test_suf - ort_suf)))
    max_abs_diff = max(max_diff_bin, max_diff_suf)
    assert max_abs_diff < 1e-4, f"Torch vs ONNX max abs diff too high: {max_abs_diff}"

    test_bin_pre = compute_binary_metrics(ort_test_bin, y_test_bin, temperature=1.0, threshold=threshold)
    test_bin_post = compute_binary_metrics(ort_test_bin, y_test_bin, temperature=temperature, threshold=threshold)
    test_suf_pre = compute_multiclass_metrics(ort_suf, y_test_cls, temperature=1.0)
    test_suf_post = compute_multiclass_metrics(ort_suf, y_test_cls, temperature=suf_temperature)

    # Single-sample latency distribution (p50, p95, p99) over 300 runs
    sample_in = x_test[:1]
    for _ in range(25):
        sess.run(["logit", "sufficiency_logits"], {"input_ids": sample_in})
    lats_ms: List[float] = []
    for _ in range(300):
        ts = time.perf_counter()
        sess.run(["logit", "sufficiency_logits"], {"input_ids": sample_in})
        lats_ms.append((time.perf_counter() - ts) * 1000.0)
    lat_arr = np.array(lats_ms, dtype=np.float64)
    p50_ms = float(np.percentile(lat_arr, 50))
    p95_ms = float(np.percentile(lat_arr, 95))
    p99_ms = float(np.percentile(lat_arr, 99))

    print(f"      Torch vs ONNX max_abs_diff={max_abs_diff:.2e}")
    print(
        f"      Test Gate  -> acc={test_bin_post['accuracy']:.4f}, f1={test_bin_post['f1']:.4f}, "
        f"ece={test_bin_post['ece']:.5f} (pre={test_bin_pre['ece']:.5f})"
    )
    print(
        f"      Test 3-Cls -> acc={test_suf_post['accuracy']:.4f}, macro_f1={test_suf_post['macro_f1']:.4f}, "
        f"ece={test_suf_post['ece']:.5f} (pre={test_suf_pre['ece']:.5f})"
    )
    print(f"      CPU ORT single-item latency: p50={p50_ms:.3f}ms, p95={p95_ms:.3f}ms, p99={p99_ms:.3f}ms")

    # Save vocab.json
    VOCAB_PATH.write_text(
        json.dumps(
            {
                "pad_id": 0,
                "unk_id": 1,
                "max_length": MAX_LENGTH,
                "vocab_size": len(vocab),
                "char_to_id": vocab,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Compute total artifacts size cleanly (write calibration.json twice to include exact byte length)
    gaz_bytes = GAZETTEER_PATH.stat().st_size if GAZETTEER_PATH.exists() else 0
    vocab_bytes = VOCAB_PATH.stat().st_size

    calib_payload: Dict[str, object] = {
        "model_name": "Laya Address VN v2 (Dual-Output Char-CNN)",
        "architecture": "Multi-Scale Char-CNN (k=2,3,5,7 + 2-layer contextual conv) + GlobalMaxPool + GlobalMeanPool + Dual MLP Heads",
        "opset": 14,
        "ops": ops_used,
        "param_count": param_count,
        "max_length": MAX_LENGTH,
        "embed_dim": EMBED_DIM,
        "temperature": temperature,
        "threshold": threshold,
        "sufficiency_temperature": suf_temperature,
        "sufficiency_classes": list(LABEL_NAMES),
        "output_names": ["logit", "sufficiency_logits"],
        "model_size_bytes": model_size_bytes,
        "gazetteer_size_bytes": gaz_bytes,
        "vocab_size_bytes": vocab_bytes,
        "calibration_size_bytes": 0,
        "total_artifacts_bytes": 0,
        "threshold_tuning": thr_stats,
        "metrics": {
            "val_gate_pre": val_bin_pre,
            "val_gate_post": val_bin_post,
            "val_sufficiency_pre": val_suf_pre,
            "val_sufficiency_post": val_suf_post,
            "test_gate_pre": test_bin_pre,
            "test_gate_post": test_bin_post,
            "test_sufficiency_pre": test_suf_pre,
            "test_sufficiency_post": test_suf_post,
            "onnx_pytorch_max_abs_diff": max_abs_diff,
            "cpu_onnx_latency_ms": {
                "p50": round(p50_ms, 3),
                "p95": round(p95_ms, 3),
                "p99": round(p99_ms, 3),
            },
        },
        "dataset": {
            "seed": SEED,
            "train_size": len(splits.train),
            "val_size": len(splits.val),
            "test_size": len(splits.test),
        },
    }

    # Fixed-point convergence so calibration_size_bytes & total_artifacts_bytes are exact
    for _ in range(3):
        calib_text = json.dumps(calib_payload, ensure_ascii=False, indent=2)
        calib_bytes = len(calib_text.encode("utf-8"))
        calib_payload["calibration_size_bytes"] = calib_bytes
        calib_payload["total_artifacts_bytes"] = model_size_bytes + gaz_bytes + vocab_bytes + calib_bytes

    with CALIB_PATH.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(calib_payload, ensure_ascii=False, indent=2))
    total_bytes = sum(p.stat().st_size for p in MODEL_DIR.iterdir() if p.is_file())
    assert model_size_bytes <= 1_572_864, f"ONNX FP32 exceeds 1.5 MB: {model_size_bytes}"
    assert total_bytes <= 2_097_152, f"Total web/model/* exceeds 2.0 MB: {total_bytes}"
    print(f"      Total web/model/* size: {total_bytes / 1024:.1f} KB ({total_bytes / (1024*1024):.3f} MB)")
    return calib_payload


if __name__ == "__main__":
    train_and_export()
