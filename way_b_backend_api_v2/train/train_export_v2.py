"""Way B v2 Training (Dynamic Batch Padding), Dual Calibration, ONNX FP32 Export & INT8 Quantization.

Run from way_b_backend_api_v2 directory:
    .venv\\Scripts\\python.exe -m train.train_export_v2
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader, Dataset

from app.tokenizer import DualGranularityTokenizer
from train.dataset_v2 import (
    SHARED_RECORDS_PATH,
    SUFFICIENCY_CLASSES,
    SampleV2,
    generate_dataset_v2,
)
from train.model_v2 import LayaDualGranularitySufficiencyModel

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
GAZETTEER_PATH = ARTIFACTS_DIR / "vn_admin_gazetteer.json"


def set_seed(seed: int = 202) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TokenizedSufficiencyDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SampleV2],
        tokenizer: DualGranularityTokenizer,
    ) -> None:
        self.labels = [s.label for s in samples]
        self.texts = [s.text for s in samples]
        self.encoded = [tokenizer.encode_single(s.text) for s in samples]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], int]:
        w, c = self.encoded[idx]
        return w, c, self.labels[idx]


class LengthBucketedBatchSampler(BatchSampler):
    """Stochastic mega-batch length bucketing so dynamic batch padding minimizes wasted PAD work."""

    def __init__(
        self,
        dataset: TokenizedSufficiencyDataset,
        batch_size: int,
        seed: int = 202,
        pool_mult: int = 24,
    ) -> None:
        self.lengths = [len(w) * 4 + len(c) for w, c in dataset.encoded]
        self.n = len(self.lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.pool_size = batch_size * pool_mult
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = list(range(self.n))
        rng.shuffle(indices)

        batches: List[List[int]] = []
        for start in range(0, self.n, self.pool_size):
            pool = indices[start : start + self.pool_size]
            pool.sort(key=self.lengths.__getitem__)
            for b_start in range(0, len(pool), self.batch_size):
                batches.append(pool[b_start : b_start + self.batch_size])
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size


def make_dynamic_collate_fn(pad_id: int, char_pad_id: int, min_char_len: int = 8):
    """Pad each mini-batch strictly to the batch's actual max word & char length."""

    def collate(batch: Sequence[Tuple[List[int], List[int], int]]):
        bsz = len(batch)
        w_len = max(len(w) for w, _, _ in batch)
        c_len = max(min_char_len, max(len(c) for _, c, _ in batch))

        word_ids = torch.full((bsz, w_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, w_len), dtype=torch.long)
        char_ids = torch.full((bsz, c_len), char_pad_id, dtype=torch.long)
        labels = torch.zeros((bsz,), dtype=torch.long)

        for i, (w, c, y) in enumerate(batch):
            word_ids[i, : len(w)] = torch.tensor(w, dtype=torch.long)
            attention_mask[i, : len(w)] = 1
            char_ids[i, : len(c)] = torch.tensor(c, dtype=torch.long)
            labels[i] = y

        return word_ids, attention_mask, char_ids, labels

    return collate


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def softmax_np(x: np.ndarray) -> np.ndarray:
    x_shift = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(np.clip(x_shift, -40.0, 40.0))
    return e / np.sum(e, axis=-1, keepdims=True)


def compute_binary_metrics(
    logits: np.ndarray,
    labels_3class: np.ndarray,
    temperature: float = 1.0,
    threshold: float = 0.5,
    n_bins: int = 15,
) -> Dict[str, Any]:
    y_bin = (labels_3class == 2).astype(np.float64)
    probs = sigmoid_np(logits / max(temperature, 1e-4))
    eps = 1e-7
    probs_clip = np.clip(probs, eps, 1.0 - eps)

    nll = float(-np.mean(y_bin * np.log(probs_clip) + (1.0 - y_bin) * np.log(1.0 - probs_clip)))
    brier = float(np.mean((probs - y_bin) ** 2))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_bin)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (probs >= lo) & (probs <= hi) if b == n_bins - 1 else (probs >= lo) & (probs < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            ece += (cnt / n) * abs(float(np.mean(y_bin[mask])) - float(np.mean(probs[mask])))

    preds = (probs >= threshold).astype(np.int64)
    y_int = y_bin.astype(np.int64)
    tp = int(np.sum((preds == 1) & (y_int == 1)))
    tn = int(np.sum((preds == 0) & (y_int == 0)))
    fp = int(np.sum((preds == 1) & (y_int == 0)))
    fn = int(np.sum((preds == 0) & (y_int == 1)))

    acc = (tp + tn) / max(1, n)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2.0 * prec * rec / max(1e-12, prec + rec)

    return {
        "accuracy": round(float(acc), 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1": round(float(f1), 6),
        "nll": round(nll, 6),
        "brier": round(brier, 6),
        "ece": round(float(ece), 6),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_sufficiency_metrics(
    suf_logits: np.ndarray,
    labels_3class: np.ndarray,
    temperature: float = 1.0,
    n_bins: int = 15,
) -> Dict[str, Any]:
    probs = softmax_np(suf_logits / max(temperature, 1e-4))
    n = len(labels_3class)
    eps = 1e-7
    probs_clip = np.clip(probs, eps, 1.0 - eps)

    y_int = labels_3class.astype(np.int64)
    nll = float(-np.mean(np.log(probs_clip[np.arange(n), y_int])))
    one_hot = np.eye(len(SUFFICIENCY_CLASSES), dtype=np.float64)[y_int]
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))

    preds = np.argmax(probs, axis=-1)
    confs = np.max(probs, axis=-1)
    correct = (preds == y_int).astype(np.float64)
    acc = float(np.mean(correct))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (confs >= lo) & (confs <= hi) if b == n_bins - 1 else (confs >= lo) & (confs < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            ece += (cnt / n) * abs(float(np.mean(correct[mask])) - float(np.mean(confs[mask])))

    per_class_f1: Dict[str, float] = {}
    for cls_idx, cls_name in enumerate(SUFFICIENCY_CLASSES):
        tp = int(np.sum((preds == cls_idx) & (y_int == cls_idx)))
        fp = int(np.sum((preds == cls_idx) & (y_int != cls_idx)))
        fn = int(np.sum((preds != cls_idx) & (y_int == cls_idx)))
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        per_class_f1[cls_name] = round(2.0 * p * r / max(1e-12, p + r), 6)

    macro_f1 = sum(per_class_f1.values()) / len(SUFFICIENCY_CLASSES)
    return {
        "accuracy": round(acc, 6),
        "macro_f1": round(float(macro_f1), 6),
        "nll": round(nll, 6),
        "brier": round(brier, 6),
        "ece": round(float(ece), 6),
        "per_class_f1": per_class_f1,
    }


def fit_binary_temperature(val_logits: np.ndarray, val_labels_3class: np.ndarray) -> float:
    logits_t = torch.tensor(val_logits, dtype=torch.float64)
    labels_t = torch.tensor((val_labels_3class == 2).astype(np.float64), dtype=torch.float64)
    criterion = nn.BCEWithLogitsLoss()

    grid = np.concatenate([np.linspace(0.25, 2.0, 71), np.linspace(2.05, 5.0, 60)])
    best_t = 1.0
    best_loss = float("inf")
    with torch.no_grad():
        for cand in grid:
            loss = float(criterion(logits_t / float(cand), labels_t).item())
            if loss < best_loss:
                best_loss = loss
                best_t = float(cand)

    log_t = torch.tensor([math.log(best_t)], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.25, max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        t = torch.exp(log_t).clamp(min=0.2, max=6.0)
        loss = criterion(logits_t / t, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    refined_t = float(torch.exp(log_t).clamp(min=0.2, max=6.0).item())
    refined_loss = float(criterion(logits_t / refined_t, labels_t).item())
    return round(refined_t if refined_loss <= best_loss else best_t, 5)


def fit_sufficiency_temperature(val_suf_logits: np.ndarray, val_labels_3class: np.ndarray) -> float:
    logits_t = torch.tensor(val_suf_logits, dtype=torch.float64)
    labels_t = torch.tensor(val_labels_3class.astype(np.int64), dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

    grid = np.concatenate([np.linspace(0.25, 2.0, 71), np.linspace(2.05, 5.0, 60)])
    best_t = 1.0
    best_loss = float("inf")
    with torch.no_grad():
        for cand in grid:
            loss = float(criterion(logits_t / float(cand), labels_t).item())
            if loss < best_loss:
                best_loss = loss
                best_t = float(cand)

    log_t = torch.tensor([math.log(best_t)], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.25, max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        t = torch.exp(log_t).clamp(min=0.2, max=6.0)
        loss = criterion(logits_t / t, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    refined_t = float(torch.exp(log_t).clamp(min=0.2, max=6.0).item())
    refined_loss = float(criterion(logits_t / refined_t, labels_t).item())
    return round(refined_t if refined_loss <= best_loss else best_t, 5)


def tune_binary_threshold(
    val_logits: np.ndarray,
    val_labels_3class: np.ndarray,
    temperature: float,
) -> float:
    y_bin = (val_labels_3class == 2).astype(np.int64)
    probs = sigmoid_np(val_logits / max(temperature, 1e-4))
    best_thresh = 0.50
    best_f1 = -1.0
    best_acc = -1.0
    # Check 0.50 first so ties stay centered at 0.50
    candidates = [0.50] + [round(x, 2) for x in np.linspace(0.35, 0.65, 31) if abs(x - 0.50) > 1e-6]
    for th in candidates:
        preds = (probs >= th).astype(np.int64)
        tp = int(np.sum((preds == 1) & (y_bin == 1)))
        tn = int(np.sum((preds == 0) & (y_bin == 0)))
        fp = int(np.sum((preds == 1) & (y_bin == 0)))
        fn = int(np.sum((preds == 0) & (y_bin == 1)))
        acc = (tp + tn) / max(1, len(y_bin))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2.0 * prec * rec / max(1e-12, prec + rec)
        if f1 > best_f1 + 1e-6 or (abs(f1 - best_f1) <= 1e-6 and acc > best_acc + 1e-6):
            best_f1 = f1
            best_acc = acc
            best_thresh = float(th)
    return round(best_thresh, 3)


@torch.no_grad()
def predict_torch(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    bin_list: List[np.ndarray] = []
    suf_list: List[np.ndarray] = []
    lbl_list: List[np.ndarray] = []
    for word_ids, attn_mask, char_ids, labels in loader:
        logits, suf_logits = model(word_ids, attn_mask, char_ids)
        bin_list.append(logits.cpu().numpy())
        suf_list.append(suf_logits.cpu().numpy())
        lbl_list.append(labels.cpu().numpy())
    return (
        np.concatenate(bin_list, axis=0),
        np.concatenate(suf_list, axis=0),
        np.concatenate(lbl_list, axis=0),
    )


def create_ort_session(model_path: Path, intra_threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def predict_ort(
    session: ort.InferenceSession,
    tokenizer: DualGranularityTokenizer,
    texts: Sequence[str],
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    bin_out: List[np.ndarray] = []
    suf_out: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        w_ids, mask, c_ids = tokenizer.encode_batch(chunk)
        logits, suf_logits = session.run(
            ["logits", "sufficiency_logits"],
            {
                "word_ids": w_ids,
                "attention_mask": mask,
                "char_ids": c_ids,
            },
        )
        bin_out.append(np.asarray(logits, dtype=np.float32).reshape(-1))
        suf_out.append(np.asarray(suf_logits, dtype=np.float32))
    return np.concatenate(bin_out, axis=0), np.concatenate(suf_out, axis=0)


def benchmark_ort_session(
    session: ort.InferenceSession,
    tokenizer: DualGranularityTokenizer,
    texts: Sequence[str],
    single_runs: int = 300,
    batch_size: int = 32,
    batch_iters: int = 40,
) -> Dict[str, float]:
    # Warmup
    for txt in texts[:30]:
        w_ids, mask, c_ids = tokenizer.encode_batch([txt])
        session.run(
            ["logits", "sufficiency_logits"],
            {"word_ids": w_ids, "attention_mask": mask, "char_ids": c_ids},
        )

    single_latencies_ms: List[float] = []
    n_texts = len(texts)
    for i in range(single_runs):
        txt = texts[i % n_texts]
        t0 = time.perf_counter()
        w_ids, mask, c_ids = tokenizer.encode_batch([txt])
        session.run(
            ["logits", "sufficiency_logits"],
            {"word_ids": w_ids, "attention_mask": mask, "char_ids": c_ids},
        )
        single_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(single_latencies_ms, dtype=np.float64)
    single_mean = float(np.mean(arr))
    single_p50 = float(np.percentile(arr, 50))
    single_p95 = float(np.percentile(arr, 95))
    single_p99 = float(np.percentile(arr, 99))

    batch_slice = list(texts[:batch_size])
    for _ in range(5):
        w_ids, mask, c_ids = tokenizer.encode_batch(batch_slice)
        session.run(
            ["logits", "sufficiency_logits"],
            {"word_ids": w_ids, "attention_mask": mask, "char_ids": c_ids},
        )

    t0 = time.perf_counter()
    for _ in range(batch_iters):
        w_ids, mask, c_ids = tokenizer.encode_batch(batch_slice)
        session.run(
            ["logits", "sufficiency_logits"],
            {"word_ids": w_ids, "attention_mask": mask, "char_ids": c_ids},
        )
    elapsed_s = time.perf_counter() - t0
    total_items = batch_size * batch_iters
    batch_ms_per_batch = (elapsed_s * 1000.0) / batch_iters
    batch_ms_per_req = (elapsed_s * 1000.0) / total_items
    batch_req_per_s = total_items / max(elapsed_s, 1e-9)

    return {
        "single_mean_ms": round(single_mean, 3),
        "single_p50_ms": round(single_p50, 3),
        "single_p95_ms": round(single_p95, 3),
        "single_p99_ms": round(single_p99, 3),
        "single_req_per_s": round(1000.0 / max(single_mean, 1e-6), 1),
        "batch_size": batch_size,
        "batch_mean_ms": round(batch_ms_per_batch, 3),
        "batch_ms_per_req": round(batch_ms_per_req, 4),
        "batch_req_per_s": round(batch_req_per_s, 1),
    }


def export_self_contained_onnx(
    model: nn.Module,
    fp32_path: Path,
    opset_version: int = 17,
) -> None:
    """Export dual-head model to a single self-contained FP32 ONNX file (opset 17, dynamic axes)."""
    model.eval()
    dummy_w = torch.tensor(
        [[2, 5, 6, 7, 8, 9, 0, 0], [2, 10, 11, 12, 0, 0, 0, 0]],
        dtype=torch.long,
    )
    dummy_m = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]],
        dtype=torch.long,
    )
    dummy_c = torch.randint(1, 20, (2, 24), dtype=torch.long)

    for stale in [fp32_path, Path(str(fp32_path) + ".data")]:
        if stale.exists():
            stale.unlink()

    exported = False
    try:
        torch.onnx.export(
            model,
            (dummy_w, dummy_m, dummy_c),
            str(fp32_path),
            input_names=["word_ids", "attention_mask", "char_ids"],
            output_names=["logits", "sufficiency_logits"],
            dynamic_axes={
                "word_ids": {0: "batch_size", 1: "word_len"},
                "attention_mask": {0: "batch_size", 1: "word_len"},
                "char_ids": {0: "batch_size", 1: "char_len"},
                "logits": {0: "batch_size"},
                "sufficiency_logits": {0: "batch_size"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )
        exported = True
    except Exception as legacy_exc:
        print(f"[ONNX] Legacy exporter note ({legacy_exc}); falling back to dynamo exporter...")

    if not exported:
        batch_dim = torch.export.Dim("batch_size", min=1, max=4096)
        word_dim = torch.export.Dim("word_len", min=1, max=64)
        char_dim = torch.export.Dim("char_len", min=8, max=160)
        onnx_prog = torch.onnx.export(
            model,
            (dummy_w, dummy_m, dummy_c),
            input_names=["word_ids", "attention_mask", "char_ids"],
            output_names=["logits", "sufficiency_logits"],
            dynamic_shapes={
                "word_ids": {0: batch_dim, 1: word_dim},
                "attention_mask": {0: batch_dim, 1: word_dim},
                "char_ids": {0: batch_dim, 1: char_dim},
            },
            opset_version=max(opset_version, 18),
            dynamo=True,
            external_data=False,
        )
        onnx_prog.save(str(fp32_path))

    data_sidecar = Path(str(fp32_path) + ".data")
    if data_sidecar.exists():
        proto = onnx.load(str(fp32_path), load_external_data=True)
        onnx.save_model(proto, str(fp32_path), save_as_external_data=False)
        data_sidecar.unlink()

    proto = onnx.load(str(fp32_path))
    onnx.checker.check_model(proto)


def collect_admin_texts() -> List[str]:
    """Collect all accented & unaccented Vietnamese admin names so vocab covers 100% of syllables."""
    texts: List[str] = []
    records = json.loads(SHARED_RECORDS_PATH.read_text(encoding="utf-8"))
    for rec in records.get("legacy_3tier", []):
        texts.append(f"{rec['ward_full']} {rec['district_full']} {rec['province_full']}")
    for rec in records.get("merged_2025_2tier", []):
        texts.append(f"{rec['ward_full']} {rec['province_full']}")
    if GAZETTEER_PATH.exists():
        gaz = json.loads(GAZETTEER_PATH.read_text(encoding="utf-8"))
        texts.extend(gaz.get("p", []))
        texts.extend(gaz.get("d", []))
        texts.extend(gaz.get("w", []))
        texts.extend(list(gaz.get("pa", {}).keys()))
        texts.extend(list(gaz.get("da", {}).keys()))
    return texts


def verify_dynamic_shapes(session: ort.InferenceSession, tokenizer: DualGranularityTokenizer) -> None:
    """Verify ONNX model handles variable batch sizes and sequence lengths (including fixed padded)."""
    for batch_inputs in (
        (["73"], False),
        (["Lộc Thạnh, Lộc Ninh"], False),
        (["Lộc Thạnh, Lộc Ninh", "Số 18 Ngõ 25 Láng Hạ, Đống Đa, Hà Nội"], False),
        (["p.12 q.tb", "Xã Lộc Thạnh", "bánh mì Sài Gòn", "ngõ 12"], True),
    ):
        texts, fixed = batch_inputs
        w, m, c = tokenizer.encode_batch(texts, fixed_length=fixed)
        logits, suf = session.run(
            ["logits", "sufficiency_logits"],
            {"word_ids": w, "attention_mask": m, "char_ids": c},
        )
        assert logits.shape == (len(texts),), f"Unexpected logits shape: {logits.shape}"
        assert suf.shape == (len(texts), 3), f"Unexpected suf shape: {suf.shape}"


def main() -> None:
    t_start = time.perf_counter()
    set_seed(202)

    train_threads = 12
    torch.set_num_threads(train_threads)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Generating 75,000-sample Vietnamese Sufficiency dataset (seed=202)...")
    splits = generate_dataset_v2(total_samples=75000, seed=202, val_ratio=0.10, test_ratio=0.10, save_manifest=True)
    train_samples = splits["train"]
    val_samples = splits["val"]
    test_samples = splits["test"]
    print(f"      Split sizes: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    print("[2/6] Building DualGranularityTokenizer (max_word_vocab=6500, full admin syllable coverage)...")
    admin_texts = collect_admin_texts()
    tokenizer = DualGranularityTokenizer.build_from_corpus(
        [s.text for s in train_samples],
        max_word_vocab=6500,
        min_word_freq=2,
        max_char_vocab=220,
        min_char_freq=3,
        max_word_len=64,
        max_char_len=160,
        admin_texts=admin_texts,
    )
    vocab_path = ARTIFACTS_DIR / "vocab.json"
    tokenizer.save(vocab_path)
    print(f"      Saved {vocab_path} (word_vocab={tokenizer.word_vocab_size}, char_vocab={tokenizer.char_vocab_size})")

    train_ds = TokenizedSufficiencyDataset(train_samples, tokenizer)
    val_ds = TokenizedSufficiencyDataset(val_samples, tokenizer)
    test_ds = TokenizedSufficiencyDataset(test_samples, tokenizer)

    collate_fn = make_dynamic_collate_fn(tokenizer.pad_id, tokenizer.char_pad_id, tokenizer.min_char_len)
    train_sampler = LengthBucketedBatchSampler(train_ds, batch_size=256, seed=202)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate_fn)
    # Sort val/test loaders by sequence length (with matching index) so eval dynamic padding is 3x faster
    val_order = sorted(range(len(val_ds)), key=lambda i: len(val_ds.encoded[i][0]) * 4 + len(val_ds.encoded[i][1]))
    test_order = sorted(range(len(test_ds)), key=lambda i: len(test_ds.encoded[i][0]) * 4 + len(test_ds.encoded[i][1]))
    val_ds_sorted = TokenizedSufficiencyDataset([val_samples[i] for i in val_order], tokenizer)
    test_ds_sorted = TokenizedSufficiencyDataset([test_samples[i] for i in test_order], tokenizer)
    val_loader = DataLoader(val_ds_sorted, batch_size=256, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds_sorted, batch_size=256, shuffle=False, collate_fn=collate_fn)

    model_cfg = {
        "architecture": "LayaDualGranularitySufficiencyModel (2-Layer Word Transformer + Multi-Scale Char-CNN + Dual Heads)",
        "word_vocab_size": tokenizer.word_vocab_size,
        "char_vocab_size": tokenizer.char_vocab_size,
        "max_word_len": tokenizer.max_word_len,
        "max_char_len": tokenizer.max_char_len,
        "word_emb_dim": 64,
        "d_model": 160,
        "n_heads": 4,
        "d_ff": 512,
        "n_layers": 2,
        "char_emb_dim": 32,
        "char_channels": 64,
        "num_sufficiency_classes": len(SUFFICIENCY_CLASSES),
        "dropout": 0.15,
    }
    model = LayaDualGranularitySufficiencyModel(
        word_vocab_size=model_cfg["word_vocab_size"],
        char_vocab_size=model_cfg["char_vocab_size"],
        max_word_len=model_cfg["max_word_len"],
        word_emb_dim=model_cfg["word_emb_dim"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        d_ff=model_cfg["d_ff"],
        n_layers=model_cfg["n_layers"],
        char_emb_dim=model_cfg["char_emb_dim"],
        char_channels=model_cfg["char_channels"],
        num_sufficiency_classes=model_cfg["num_sufficiency_classes"],
        dropout=model_cfg["dropout"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[3/6] Training LayaDualGranularitySufficiencyModel ({total_params:,} params, batch_size=256, epochs=8, threads={train_threads})...")

    epochs = 8
    bce_loss_fn = nn.BCEWithLogitsLoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1.5e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        n_seen = 0
        for word_ids, attn_mask, char_ids, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, suf_logits = model(word_ids, attn_mask, char_ids)
            bin_targets = (labels == 2).to(dtype=torch.float32)
            loss = bce_loss_fn(logits, bin_targets) + ce_loss_fn(suf_logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bsz = labels.size(0)
            running_loss += float(loss.item()) * bsz
            n_seen += bsz
        scheduler.step()

        v_bin, v_suf, v_lbl = predict_torch(model, val_loader)
        v_bin_m = compute_binary_metrics(v_bin, v_lbl, temperature=1.0)
        v_suf_m = compute_sufficiency_metrics(v_suf, v_lbl, temperature=1.0)
        val_joint = v_bin_m["nll"] + v_suf_m["nll"]
        ep_sec = time.perf_counter() - ep_t0
        print(
            f"      Epoch {epoch:02d}/{epochs} ({ep_sec:4.1f}s) | "
            f"train_loss={running_loss / n_seen:.4f} | "
            f"val_joint={val_joint:.4f} | "
            f"bin_acc={v_bin_m['accuracy']:.4f} | suf_acc={v_suf_m['accuracy']:.4f}"
        )
        if val_joint < best_val_loss:
            best_val_loss = val_joint
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    print("[4/6] Fitting Dual Temperature Calibration (T and T_suf) + Binary Threshold on validation set...")
    val_bin, val_suf, val_lbl = predict_torch(model, val_loader)
    test_bin, test_suf, test_lbl = predict_torch(model, test_loader)

    val_bin_pre = compute_binary_metrics(val_bin, val_lbl, temperature=1.0, threshold=0.5)
    val_suf_pre = compute_sufficiency_metrics(val_suf, val_lbl, temperature=1.0)

    temperature = fit_binary_temperature(val_bin, val_lbl)
    threshold = tune_binary_threshold(val_bin, val_lbl, temperature=temperature)
    suf_temperature = fit_sufficiency_temperature(val_suf, val_lbl)

    val_bin_post = compute_binary_metrics(val_bin, val_lbl, temperature=temperature, threshold=threshold)
    val_suf_post = compute_sufficiency_metrics(val_suf, val_lbl, temperature=suf_temperature)
    test_bin_pre = compute_binary_metrics(test_bin, test_lbl, temperature=1.0, threshold=0.5)
    test_bin_post = compute_binary_metrics(test_bin, test_lbl, temperature=temperature, threshold=threshold)
    test_suf_pre = compute_sufficiency_metrics(test_suf, test_lbl, temperature=1.0)
    test_suf_post = compute_sufficiency_metrics(test_suf, test_lbl, temperature=suf_temperature)

    print(f"      Optimal Binary T = {temperature:.5f} (threshold={threshold:.3f}) | Sufficiency T_suf = {suf_temperature:.5f}")
    print(
        f"      Val  Binary : pre ECE={val_bin_pre['ece']:.5f} NLL={val_bin_pre['nll']:.5f} -> "
        f"post ECE={val_bin_post['ece']:.5f} NLL={val_bin_post['nll']:.5f} Acc={val_bin_post['accuracy']:.4f}"
    )
    print(
        f"      Val  3-Class: pre ECE={val_suf_pre['ece']:.5f} NLL={val_suf_pre['nll']:.5f} -> "
        f"post ECE={val_suf_post['ece']:.5f} NLL={val_suf_post['nll']:.5f} Acc={val_suf_post['accuracy']:.4f}"
    )
    print(
        f"      Test Binary : Acc={test_bin_post['accuracy']:.4f} F1={test_bin_post['f1']:.4f} ECE={test_bin_post['ece']:.5f} | "
        f"Test 3-Class: Acc={test_suf_post['accuracy']:.4f} MacroF1={test_suf_post['macro_f1']:.4f} ECE={test_suf_post['ece']:.5f}"
    )

    print("[5/6] Exporting FP32 ONNX (opset 17) & applying Dynamic INT8 Quantization (QUInt8)...")
    fp32_path = ARTIFACTS_DIR / "laya_address_vn_fp32.onnx"
    int8_path = ARTIFACTS_DIR / "laya_address_vn_int8.onnx"
    export_self_contained_onnx(model, fp32_path, opset_version=17)

    if int8_path.exists():
        int8_path.unlink()
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QUInt8,
    )

    fp32_bytes = fp32_path.stat().st_size
    int8_bytes = int8_path.stat().st_size
    fp32_mb = fp32_bytes / (1024 * 1024)
    int8_mb = int8_bytes / (1024 * 1024)
    size_reduction_pct = (1.0 - int8_bytes / max(1, fp32_bytes)) * 100.0
    print(
        f"      FP32 size: {fp32_mb:.3f} MB ({fp32_bytes:,} B) | "
        f"INT8 size: {int8_mb:.3f} MB ({int8_bytes:,} B) | "
        f"Reduction: {size_reduction_pct:.1f}%"
    )

    print("[6/6] Benchmarking FP32 vs INT8 ONNX Runtime CPU inference...")
    test_texts_sorted = [test_samples[i].text for i in test_order]
    test_texts_bench = [s.text for s in test_samples]

    # Pick optimal intra_op_threads (1 vs 2) minimizing single-sample CPU p99
    best_threads = 1
    best_p99 = float("inf")
    for cand_t in (1, 2):
        s_tmp = create_ort_session(int8_path, intra_threads=cand_t)
        b_tmp = benchmark_ort_session(s_tmp, tokenizer, test_texts_bench, single_runs=150)
        print(f"      Thread probe intra_op_threads={cand_t}: p50={b_tmp['single_p50_ms']:.3f}ms p95={b_tmp['single_p95_ms']:.3f}ms p99={b_tmp['single_p99_ms']:.3f}ms")
        if b_tmp["single_p99_ms"] < best_p99:
            best_p99 = b_tmp["single_p99_ms"]
            best_threads = cand_t

    ort_threads = best_threads
    sess_fp32 = create_ort_session(fp32_path, intra_threads=ort_threads)
    sess_int8 = create_ort_session(int8_path, intra_threads=ort_threads)
    verify_dynamic_shapes(sess_fp32, tokenizer)
    verify_dynamic_shapes(sess_int8, tokenizer)

    fp32_bin_logits, fp32_suf_logits = predict_ort(sess_fp32, tokenizer, test_texts_sorted)
    int8_bin_logits, int8_suf_logits = predict_ort(sess_int8, tokenizer, test_texts_sorted)

    max_torch_fp32_diff = float(np.max(np.abs(test_bin - fp32_bin_logits)))
    max_fp32_int8_diff = float(np.max(np.abs(fp32_bin_logits - int8_bin_logits)))
    max_suf_fp32_int8_diff = float(np.max(np.abs(fp32_suf_logits - int8_suf_logits)))

    fp32_test_bin_m = compute_binary_metrics(fp32_bin_logits, test_lbl, temperature=temperature, threshold=threshold)
    int8_test_bin_m = compute_binary_metrics(int8_bin_logits, test_lbl, temperature=temperature, threshold=threshold)
    fp32_test_suf_m = compute_sufficiency_metrics(fp32_suf_logits, test_lbl, temperature=suf_temperature)
    int8_test_suf_m = compute_sufficiency_metrics(int8_suf_logits, test_lbl, temperature=suf_temperature)

    fp32_bench = benchmark_ort_session(sess_fp32, tokenizer, test_texts_bench)
    int8_bench = benchmark_ort_session(sess_int8, tokenizer, test_texts_bench)

    print(f"      Max |Torch - FP32 ONNX| bin diff: {max_torch_fp32_diff:.6f}")
    print(f"      Max |FP32  - INT8 ONNX| bin diff: {max_fp32_int8_diff:.6f} | suf diff: {max_suf_fp32_int8_diff:.6f}")
    print(
        f"      FP32 Test: BinAcc={fp32_test_bin_m['accuracy']:.4f} SufAcc={fp32_test_suf_m['accuracy']:.4f} | "
        f"Single p50={fp32_bench['single_p50_ms']:.3f}ms p95={fp32_bench['single_p95_ms']:.3f}ms p99={fp32_bench['single_p99_ms']:.3f}ms | "
        f"Batch32={fp32_bench['batch_req_per_s']:.0f} req/s"
    )
    print(
        f"      INT8 Test: BinAcc={int8_test_bin_m['accuracy']:.4f} SufAcc={int8_test_suf_m['accuracy']:.4f} | "
        f"Single p50={int8_bench['single_p50_ms']:.3f}ms p95={int8_bench['single_p95_ms']:.3f}ms p99={int8_bench['single_p99_ms']:.3f}ms | "
        f"Batch32={int8_bench['batch_req_per_s']:.0f} req/s"
    )

    calibration_payload = {
        "model_name": "laya-address-vn-way-b-v2-int8",
        "version": "2.0.0",
        "architecture": model_cfg,
        "total_parameters": total_params,
        "dataset": {
            "source": "tranngocminhhieu/vietnamadminunits (MIT) + synthetic negative/chat/fragment corpus",
            "total_samples": len(train_samples) + len(val_samples) + len(test_samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "seed": 202,
            "sufficiency_classes": list(SUFFICIENCY_CLASSES),
        },
        "calibration": {
            "temperature": temperature,
            "threshold": threshold,
            "sufficiency_temperature": suf_temperature,
            "sufficiency_classes": list(SUFFICIENCY_CLASSES),
            "val_pre_calibration": val_bin_pre,
            "val_post_calibration": val_bin_post,
            "test_pre_calibration": test_bin_pre,
            "test_post_calibration": test_bin_post,
            "val_sufficiency_pre_calibration": val_suf_pre,
            "val_sufficiency_post_calibration": val_suf_post,
            "test_sufficiency_pre_calibration": test_suf_pre,
            "test_sufficiency_post_calibration": test_suf_post,
        },
        "onnx_comparison": {
            "opset_version": 17,
            "ort_intra_op_threads": ort_threads,
            "fp32_file": fp32_path.name,
            "int8_file": int8_path.name,
            "fp32_size_bytes": fp32_bytes,
            "int8_size_bytes": int8_bytes,
            "fp32_size_mb": round(fp32_mb, 3),
            "int8_size_mb": round(int8_mb, 3),
            "size_reduction_percent": round(size_reduction_pct, 2),
            "max_torch_vs_fp32_logit_diff": round(max_torch_fp32_diff, 6),
            "max_fp32_vs_int8_logit_diff": round(max_fp32_int8_diff, 6),
            "max_fp32_vs_int8_suf_logit_diff": round(max_suf_fp32_int8_diff, 6),
            "fp32_test_metrics": fp32_test_bin_m,
            "int8_test_metrics": int8_test_bin_m,
            "fp32_test_sufficiency_metrics": fp32_test_suf_m,
            "int8_test_sufficiency_metrics": int8_test_suf_m,
            "fp32_cpu_benchmark": fp32_bench,
            "int8_cpu_benchmark": int8_bench,
        },
        "training_wall_seconds": round(time.perf_counter() - t_start, 2),
    }

    cal_path = ARTIFACTS_DIR / "calibration.json"
    cal_path.write_text(json.dumps(calibration_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone in {calibration_payload['training_wall_seconds']}s! Saved {cal_path}")


if __name__ == "__main__":
    main()
