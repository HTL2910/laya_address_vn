"""Training, Laya Temperature Calibration, ONNX FP32 Export & Dynamic INT8 Quantization.

Run from way_b_backend_api directory:
    .venv\\Scripts\\python.exe -m train.train_export
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.tokenizer import DualGranularityTokenizer
from train.dataset import AddressSample, generate_dataset
from train.model import LayaDualGranularityNoulModel

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TokenizedAddressDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[AddressSample],
        tokenizer: DualGranularityTokenizer,
    ) -> None:
        self.labels = [s.label for s in samples]
        self.categories = [s.category for s in samples]
        self.texts = [s.text for s in samples]
        self.encoded = [tokenizer.encode_single(s.text) for s in samples]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], int]:
        w, c = self.encoded[idx]
        return w, c, self.labels[idx]


def make_collate_fn(pad_id: int, char_pad_id: int, min_char_len: int = 8):
    def collate(batch: Sequence[Tuple[List[int], List[int], int]]):
        bsz = len(batch)
        w_len = max(len(w) for w, _, _ in batch)
        c_len = max(min_char_len, max(len(c) for _, c, _ in batch))

        word_ids = torch.full((bsz, w_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, w_len), dtype=torch.long)
        char_ids = torch.full((bsz, c_len), char_pad_id, dtype=torch.long)
        labels = torch.zeros((bsz,), dtype=torch.float32)

        for i, (w, c, y) in enumerate(batch):
            word_ids[i, : len(w)] = torch.tensor(w, dtype=torch.long)
            attention_mask[i, : len(w)] = 1
            char_ids[i, : len(c)] = torch.tensor(c, dtype=torch.long)
            labels[i] = float(y)

        return word_ids, attention_mask, char_ids, labels

    return collate


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float = 1.0,
    threshold: float = 0.5,
    n_bins: int = 15,
) -> Dict[str, float]:
    probs = sigmoid_np(logits / max(temperature, 1e-4))
    eps = 1e-7
    probs_clip = np.clip(probs, eps, 1.0 - eps)

    nll = float(-np.mean(labels * np.log(probs_clip) + (1.0 - labels) * np.log(1.0 - probs_clip)))
    brier = float(np.mean((probs - labels) ** 2))

    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            avg_conf = float(np.mean(probs[mask]))
            avg_acc = float(np.mean(labels[mask]))
            ece += (cnt / n) * abs(avg_acc - avg_conf)

    preds = (probs >= threshold).astype(np.int64)
    y_int = labels.astype(np.int64)
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


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Optimize scalar temperature T > 0 on validation logits minimizing NLL."""
    logits_t = torch.tensor(val_logits, dtype=torch.float64)
    labels_t = torch.tensor(val_labels, dtype=torch.float64)
    criterion = nn.BCEWithLogitsLoss()

    # Coarse grid search for robust global basin
    grid = np.concatenate([np.linspace(0.25, 2.0, 71), np.linspace(2.05, 5.0, 60)])
    best_t = 1.0
    best_loss = float("inf")
    with torch.no_grad():
        for cand in grid:
            loss = float(criterion(logits_t / float(cand), labels_t).item())
            if loss < best_loss:
                best_loss = loss
                best_t = float(cand)

    # Fine L-BFGS refinement around grid optimum
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
    final_t = refined_t if refined_loss <= best_loss else best_t
    return round(final_t, 5)


@torch.no_grad()
def predict_torch_logits(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    for word_ids, attn_mask, char_ids, labels in loader:
        logits = model(word_ids, attn_mask, char_ids)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


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


def predict_ort_logits(
    session: ort.InferenceSession,
    tokenizer: DualGranularityTokenizer,
    texts: Sequence[str],
    batch_size: int = 64,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        w_ids, mask, c_ids = tokenizer.encode_batch(chunk)
        (logits,) = session.run(
            ["logits"],
            {
                "word_ids": w_ids,
                "attention_mask": mask,
                "char_ids": c_ids,
            },
        )
        outputs.append(np.asarray(logits, dtype=np.float32))
    return np.concatenate(outputs, axis=0)


def benchmark_ort_session(
    session: ort.InferenceSession,
    tokenizer: DualGranularityTokenizer,
    texts: Sequence[str],
    single_runs: int = 200,
    batch_size: int = 32,
    batch_iters: int = 40,
) -> Dict[str, float]:
    # Warmup
    for txt in texts[:20]:
        w_ids, mask, c_ids = tokenizer.encode_batch([txt])
        session.run(["logits"], {"word_ids": w_ids, "attention_mask": mask, "char_ids": c_ids})

    # Single-request end-to-end (tokenize + ORT run)
    single_latencies_ms: List[float] = []
    n_texts = len(texts)
    for i in range(single_runs):
        txt = texts[i % n_texts]
        t0 = time.perf_counter()
        w_ids, mask, c_ids = tokenizer.encode_batch([txt])
        session.run(["logits"], {"word_ids": w_ids, "attention_mask": mask, "char_ids": c_ids})
        single_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(single_latencies_ms, dtype=np.float64)
    single_mean = float(np.mean(arr))
    single_p50 = float(np.percentile(arr, 50))
    single_p95 = float(np.percentile(arr, 95))
    single_p99 = float(np.percentile(arr, 99))

    # Batched end-to-end throughput
    batch_slice = list(texts[:batch_size])
    # Warmup batch
    for _ in range(5):
        w_ids, mask, c_ids = tokenizer.encode_batch(batch_slice)
        session.run(["logits"], {"word_ids": w_ids, "attention_mask": mask, "char_ids": c_ids})

    t0 = time.perf_counter()
    for _ in range(batch_iters):
        w_ids, mask, c_ids = tokenizer.encode_batch(batch_slice)
        session.run(["logits"], {"word_ids": w_ids, "attention_mask": mask, "char_ids": c_ids})
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
    """Export model to a single self-contained FP32 ONNX file with dynamic axes."""
    model.eval()
    dummy_w = torch.tensor([[2, 5, 6, 7, 8, 9, 0, 0], [2, 10, 11, 12, 0, 0, 0, 0]], dtype=torch.long)
    dummy_m = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)
    dummy_c = torch.randint(1, 20, (2, 24), dtype=torch.long)

    # Remove stale artifacts / external data if re-exporting
    for stale in [fp32_path, fp32_path.with_suffix(".onnx.data")]:
        if stale.exists():
            stale.unlink()

    exported = False
    # Try standard TorchScript exporter first for guaranteed opset 17 + single-file + dynamic_axes
    try:
        torch.onnx.export(
            model,
            (dummy_w, dummy_m, dummy_c),
            str(fp32_path),
            input_names=["word_ids", "attention_mask", "char_ids"],
            output_names=["logits"],
            dynamic_axes={
                "word_ids": {0: "batch_size", 1: "word_len"},
                "attention_mask": {0: "batch_size", 1: "word_len"},
                "char_ids": {0: "batch_size", 1: "char_len"},
                "logits": {0: "batch_size"},
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
            output_names=["logits"],
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

    # Ensure strictly single self-contained file (inline all initializers, remove any sidecar .data)
    data_sidecar = Path(str(fp32_path) + ".data")
    if data_sidecar.exists():
        proto = onnx.load(str(fp32_path), load_external_data=True)
        onnx.save_model(proto, str(fp32_path), save_as_external_data=False)
        data_sidecar.unlink()

    proto = onnx.load(str(fp32_path))
    onnx.checker.check_model(proto)


def main() -> None:
    t_start = time.perf_counter()
    set_seed(42)

    cpu_count = os.cpu_count() or 4
    train_threads = max(2, min(cpu_count, 8))
    torch.set_num_threads(train_threads)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Generating Vietnamese Address dataset (22,000 samples, seed=42)...")
    splits = generate_dataset(total_samples=22000, seed=42, val_ratio=0.10, test_ratio=0.10)
    train_samples = splits["train"]
    val_samples = splits["val"]
    test_samples = splits["test"]
    print(
        f"      Split sizes: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
    )

    print("[2/6] Building dual-granularity tokenizer on train split...")
    tokenizer = DualGranularityTokenizer.build_from_corpus(
        [s.text for s in train_samples],
        max_word_vocab=5000,
        min_word_freq=2,
        max_char_vocab=160,
        min_char_freq=5,
        max_word_len=64,
        max_char_len=160,
    )
    vocab_path = ARTIFACTS_DIR / "vocab.json"
    tokenizer.save(vocab_path)
    print(
        f"      Saved {vocab_path} (word_vocab={tokenizer.word_vocab_size}, char_vocab={tokenizer.char_vocab_size})"
    )

    train_ds = TokenizedAddressDataset(train_samples, tokenizer)
    val_ds = TokenizedAddressDataset(val_samples, tokenizer)
    test_ds = TokenizedAddressDataset(test_samples, tokenizer)

    collate_fn = make_collate_fn(tokenizer.pad_id, tokenizer.char_pad_id, tokenizer.min_char_len)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

    model_cfg = {
        "architecture": "LayaDualGranularityNoulModel (2-Layer Word Transformer + Multi-Scale Char-CNN + Noul Head)",
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
        "dropout": 0.15,
    }
    model = LayaDualGranularityNoulModel(
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
        dropout=model_cfg["dropout"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[3/6] Training LayaDualGranularityNoulModel ({total_params:,} params) on CPU ({train_threads} threads)...")

    epochs = 9
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=1.5e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    best_val_nll = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        n_seen = 0
        for word_ids, attn_mask, char_ids, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(word_ids, attn_mask, char_ids)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bsz = labels.size(0)
            running_loss += float(loss.item()) * bsz
            n_seen += bsz
        scheduler.step()

        val_logits, val_labels = predict_torch_logits(model, val_loader)
        val_m = compute_metrics(val_logits, val_labels, temperature=1.0)
        ep_sec = time.perf_counter() - ep_t0
        print(
            f"      Epoch {epoch:02d}/{epochs} ({ep_sec:4.1f}s) | "
            f"train_loss={running_loss / n_seen:.4f} | "
            f"val_nll={val_m['nll']:.4f} | val_acc={val_m['accuracy']:.4f} | val_f1={val_m['f1']:.4f}"
        )
        if val_m["nll"] < best_val_nll:
            best_val_nll = val_m["nll"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    print("[4/6] Fitting Laya Temperature Calibration on validation set...")
    val_logits, val_labels = predict_torch_logits(model, val_loader)
    test_logits, test_labels = predict_torch_logits(model, test_loader)

    val_pre = compute_metrics(val_logits, val_labels, temperature=1.0)
    temperature = fit_temperature(val_logits, val_labels)
    val_post = compute_metrics(val_logits, val_labels, temperature=temperature)
    test_pre = compute_metrics(test_logits, test_labels, temperature=1.0)
    test_post = compute_metrics(test_logits, test_labels, temperature=temperature)

    print(f"      Optimal Temperature T = {temperature:.5f}")
    print(
        f"      Val  pre-cal : NLL={val_pre['nll']:.5f}, Brier={val_pre['brier']:.5f}, ECE={val_pre['ece']:.5f}"
    )
    print(
        f"      Val  post-cal: NLL={val_post['nll']:.5f}, Brier={val_post['brier']:.5f}, ECE={val_post['ece']:.5f}"
    )
    print(
        f"      Test post-cal: Acc={test_post['accuracy']:.4f}, F1={test_post['f1']:.4f}, "
        f"NLL={test_post['nll']:.5f}, Brier={test_post['brier']:.5f}, ECE={test_post['ece']:.5f}"
    )

    print("[5/6] Exporting FP32 ONNX & applying Dynamic INT8 Quantization...")
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
        f"      FP32 size: {fp32_mb:.3f} MB ({fp32_bytes:,} bytes) | "
        f"INT8 size: {int8_mb:.3f} MB ({int8_bytes:,} bytes) | "
        f"Reduction: {size_reduction_pct:.1f}%"
    )

    print("[6/6] Benchmarking FP32 vs INT8 ONNX Runtime CPU inference...")
    ort_threads = min(4, max(1, cpu_count // 4))
    sess_fp32 = create_ort_session(fp32_path, intra_threads=ort_threads)
    sess_int8 = create_ort_session(int8_path, intra_threads=ort_threads)

    test_texts = [s.text for s in test_samples]
    fp32_test_logits = predict_ort_logits(sess_fp32, tokenizer, test_texts)
    int8_test_logits = predict_ort_logits(sess_int8, tokenizer, test_texts)

    max_torch_fp32_diff = float(np.max(np.abs(test_logits - fp32_test_logits)))
    max_fp32_int8_diff = float(np.max(np.abs(fp32_test_logits - int8_test_logits)))

    fp32_test_metrics = compute_metrics(fp32_test_logits, test_labels, temperature=temperature)
    int8_test_metrics = compute_metrics(int8_test_logits, test_labels, temperature=temperature)

    fp32_bench = benchmark_ort_session(sess_fp32, tokenizer, test_texts)
    int8_bench = benchmark_ort_session(sess_int8, tokenizer, test_texts)

    print(f"      Max |Torch - FP32 ONNX| logit diff: {max_torch_fp32_diff:.6f}")
    print(f"      Max |FP32  - INT8 ONNX| logit diff: {max_fp32_int8_diff:.6f}")
    print(
        f"      FP32 Test: Acc={fp32_test_metrics['accuracy']:.4f}, F1={fp32_test_metrics['f1']:.4f} | "
        f"Single={fp32_bench['single_mean_ms']:.2f}ms ({fp32_bench['single_req_per_s']:.0f} req/s) | "
        f"Batch32={fp32_bench['batch_ms_per_req']:.3f}ms/req ({fp32_bench['batch_req_per_s']:.0f} req/s)"
    )
    print(
        f"      INT8 Test: Acc={int8_test_metrics['accuracy']:.4f}, F1={int8_test_metrics['f1']:.4f} | "
        f"Single={int8_bench['single_mean_ms']:.2f}ms ({int8_bench['single_req_per_s']:.0f} req/s) | "
        f"Batch32={int8_bench['batch_ms_per_req']:.3f}ms/req ({int8_bench['batch_req_per_s']:.0f} req/s)"
    )

    calibration_payload = {
        "model_name": "laya-address-vn-way-b-int8",
        "version": "1.0.0",
        "architecture": model_cfg,
        "total_parameters": total_params,
        "dataset": {
            "total_samples": len(train_samples) + len(val_samples) + len(test_samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "seed": 42,
        },
        "calibration": {
            "temperature": temperature,
            "threshold": 0.5,
            "val_pre_calibration": val_pre,
            "val_post_calibration": val_post,
            "test_pre_calibration": test_pre,
            "test_post_calibration": test_post,
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
            "fp32_test_metrics": fp32_test_metrics,
            "int8_test_metrics": int8_test_metrics,
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
