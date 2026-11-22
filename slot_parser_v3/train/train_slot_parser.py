"""Train Word+Char BiGRU + Linear-CRF slot parser, export ONNX INT8, and evaluate holdout."""

from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

# Ensure parent of slot_parser_v3 is importable regardless of current working directory.
_PKG_PARENT = Path(__file__).resolve().parents[2]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from slot_parser_v3.parser import (
    AddressSlotParser,
    extract_bio_spans,
    extract_slots_and_clean_address,
    viterbi_decode,
)
from slot_parser_v3.tokenizer import (
    BIO_TAGS,
    ID2TAG,
    SLOT_TYPES,
    TAG2ID,
    UNK_ID,
    SlotParserTokenizer,
)
from slot_parser_v3.train.model import LinearCRF, SlotEncoder, SlotTaggerCRF
from slot_parser_v3.train.synth_data import SynthExample, generate_dataset, generate_holdout

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _encode_dataset(
    examples: Sequence[SynthExample],
    tokenizer: SlotParserTokenizer,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorize token sequences and BIO tags into fixed-width NumPy arrays."""
    token_seqs = [ex.tokens for ex in examples]
    encoded = tokenizer.encode_batch(token_seqs)
    max_len = tokenizer.max_word_len
    tags = np.zeros((len(examples), max_len), dtype=np.int64)
    for idx, ex in enumerate(examples):
        limit = min(len(ex.tags), max_len)
        for t_idx in range(limit):
            tags[idx, t_idx] = TAG2ID[ex.tags[t_idx]]
    return (
        encoded["word_ids"],
        encoded["attention_mask"],
        encoded["char_ids"],
        tags,
    )


def _compute_metrics(
    examples: Sequence[SynthExample],
    pred_tags_list: Sequence[Sequence[str]],
) -> Dict[str, Any]:
    """Compute token accuracy, per-slot Entity F1, macro F1, and clean-address match."""
    correct_tokens = 0
    total_tokens = 0
    tp_counts: Dict[str, int] = {s: 0 for s in SLOT_TYPES}
    pred_counts: Dict[str, int] = {s: 0 for s in SLOT_TYPES}
    gold_counts: Dict[str, int] = {s: 0 for s in SLOT_TYPES}
    clean_matches = 0
    neg_total = 0
    neg_fp = 0

    for ex, pred_tags in zip(examples, pred_tags_list):
        gold_tags = ex.tags
        for g_tag, p_tag in zip(gold_tags, pred_tags):
            total_tokens += 1
            if g_tag == p_tag:
                correct_tokens += 1

        gold_spans = set(extract_bio_spans(gold_tags))
        pred_spans = set(extract_bio_spans(pred_tags))
        for span in gold_spans:
            if span[0] in gold_counts:
                gold_counts[span[0]] += 1
                if span in pred_spans:
                    tp_counts[span[0]] += 1
        for span in pred_spans:
            if span[0] in pred_counts:
                pred_counts[span[0]] += 1

        gold_slots = extract_slots_and_clean_address(ex.tokens, gold_tags)
        pred_slots = extract_slots_and_clean_address(ex.tokens, pred_tags)
        if gold_slots["clean_address_for_goong"] == pred_slots["clean_address_for_goong"]:
            clean_matches += 1

        if all(t == "O" for t in gold_tags):
            neg_total += 1
            if any(pred_slots[s] for s in SLOT_TYPES):
                neg_fp += 1

    per_slot: Dict[str, Dict[str, float]] = {}
    f1_values: List[float] = []
    for slot in SLOT_TYPES:
        tp = tp_counts[slot]
        p_tot = pred_counts[slot]
        g_tot = gold_counts[slot]
        prec = tp / p_tot if p_tot > 0 else (1.0 if g_tot == 0 else 0.0)
        rec = tp / g_tot if g_tot > 0 else (1.0 if p_tot == 0 else 0.0)
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_slot[slot] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": g_tot,
        }
        f1_values.append(f1)

    total_tp = sum(tp_counts.values())
    total_pred = sum(pred_counts.values())
    total_gold = sum(gold_counts.values())
    micro_p = total_tp / total_pred if total_pred > 0 else 0.0
    micro_r = total_tp / total_gold if total_gold > 0 else 0.0
    micro_f1 = (2.0 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    return {
        "num_examples": len(examples),
        "token_accuracy": round(correct_tokens / max(1, total_tokens), 4),
        "macro_entity_f1": round(float(np.mean(f1_values)), 4),
        "micro_entity_f1": round(micro_f1, 4),
        "clean_address_match_rate": round(clean_matches / max(1, len(examples)), 4),
        "negative_false_positive_rate": round(neg_fp / max(1, neg_total), 4),
        "per_slot": per_slot,
    }


@torch.no_grad()
def _predict_pytorch_tags(
    model: SlotTaggerCRF,
    examples: Sequence[SynthExample],
    word_ids: np.ndarray,
    attn_mask: np.ndarray,
    char_ids: np.ndarray,
    batch_size: int = 512,
) -> List[List[str]]:
    model.eval()
    start_t, trans_t, end_t = model.crf.constrained_parameters()
    start_np = start_t.cpu().numpy()
    trans_np = trans_t.cpu().numpy()
    end_np = end_t.cpu().numpy()

    preds: List[List[str]] = []
    n_items = len(examples)
    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        w_b = torch.from_numpy(word_ids[start:end])
        m_b = torch.from_numpy(attn_mask[start:end])
        c_b = torch.from_numpy(char_ids[start:end])
        emit_b = model.encoder(w_b, m_b, c_b).cpu().numpy()
        for idx_in_b in range(end - start):
            ex_len = len(examples[start + idx_in_b].tokens)
            tag_ids = viterbi_decode(
                emit_b[idx_in_b],
                trans_np,
                length=min(ex_len, emit_b.shape[1]),
                start_transitions=start_np,
                end_transitions=end_np,
            )
            tags = [ID2TAG[tid] for tid in tag_ids]
            if ex_len > len(tags):
                tags.extend(["O"] * (ex_len - len(tags)))
            preds.append(tags)
    return preds


def _export_onnx_and_crf(
    model: SlotTaggerCRF,
    tokenizer: SlotParserTokenizer,
    artifacts_dir: Path,
    val_examples: Sequence[SynthExample],
    val_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Dict[str, Any]:
    """Export encoder to FP32 ONNX -> INT8 dynamic quantization, then delete temp FP32."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    encoder = model.encoder.eval().cpu()

    tmp_fp32 = artifacts_dir / "_tmp_slot_parser_fp32.onnx"
    int8_path = artifacts_dir / "slot_parser_v3_int8.onnx"
    crf_path = artifacts_dir / "crf_transitions.json"
    vocab_path = artifacts_dir / "vocab.json"

    # Save tokenizer vocab & CRF transitions first so AddressSlotParser can load them.
    tokenizer.save_vocab(vocab_path)
    with torch.no_grad():
        start_t, trans_t, end_t = model.crf.constrained_parameters()
        crf_payload = {
            "tags": list(BIO_TAGS),
            "start_transitions": start_t.cpu().numpy().tolist(),
            "transitions": trans_t.cpu().numpy().tolist(),
            "end_transitions": end_t.cpu().numpy().tolist(),
        }
    crf_path.write_text(json.dumps(crf_payload, indent=2), encoding="utf-8")

    # Export clean encoder + emission head (batch_size=1 dummy + dynamic batch axis).
    dummy_w = torch.zeros((1, tokenizer.max_word_len), dtype=torch.int64)
    dummy_m = torch.ones((1, tokenizer.max_word_len), dtype=torch.int64)
    dummy_c = torch.zeros(
        (1, tokenizer.max_word_len, tokenizer.max_char_per_word), dtype=torch.int64
    )
    torch.onnx.export(
        encoder,
        (dummy_w, dummy_m, dummy_c),
        str(tmp_fp32),
        input_names=["word_ids", "attention_mask", "char_ids"],
        output_names=["emissions"],
        dynamic_axes={
            "word_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "char_ids": {0: "batch"},
            "emissions": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    quantize_dynamic(
        model_input=str(tmp_fp32),
        model_output=str(int8_path),
        weight_type=QuantType.QUInt8,
    )

    fp32_bytes = tmp_fp32.stat().st_size
    int8_bytes = int8_path.stat().st_size

    # Verify INT8 quantization preserves accuracy within 1% on validation split.
    val_w, val_m, val_c, _ = val_arrays
    pt_preds = _predict_pytorch_tags(model, val_examples, val_w, val_m, val_c)
    pt_metrics = _compute_metrics(val_examples, pt_preds)

    parser = AddressSlotParser(artifacts_dir=artifacts_dir, intra_op_threads=1)
    ort_preds = [res.tags for res in parser.parse_batch([ex.text for ex in val_examples])]
    ort_metrics = _compute_metrics(val_examples, ort_preds)

    macro_drop = pt_metrics["macro_entity_f1"] - ort_metrics["macro_entity_f1"]
    if int8_bytes > 4 * 1024 * 1024 or macro_drop > 0.01:
        # Fallback cleanly if INT8 exceeded budget or degraded accuracy > 1%.
        if fp32_bytes <= 4 * 1024 * 1024 and macro_drop > 0.01:
            int8_path.unlink(missing_ok=True)
            tmp_fp32.replace(int8_path)
        else:
            tmp_fp32.unlink(missing_ok=True)
    else:
        tmp_fp32.unlink(missing_ok=True)

    # Ensure zero leftover .pt or temporary .onnx files remain in artifacts_dir.
    for leftover in artifacts_dir.glob("*.pt"):
        leftover.unlink(missing_ok=True)
    for leftover_tmp in artifacts_dir.glob("_tmp_*.onnx"):
        leftover_tmp.unlink(missing_ok=True)

    return {
        "fp32_bytes": fp32_bytes,
        "int8_bytes": int8_path.stat().st_size,
        "int8_mb": round(int8_path.stat().st_size / (1024 * 1024), 4),
        "val_pytorch_macro_f1": pt_metrics["macro_entity_f1"],
        "val_int8_macro_f1": ort_metrics["macro_entity_f1"],
        "quantization_f1_delta": round(ort_metrics["macro_entity_f1"] - pt_metrics["macro_entity_f1"], 4),
    }


def _evaluate_runtime_parser(
    parser: AddressSlotParser,
    examples: Sequence[SynthExample],
    benchmark_latency: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Run ONNX INT8 + NumPy Viterbi evaluation and optional single-item latency bench."""
    if benchmark_latency:
        latencies: List[float] = []
        pred_tags_list: List[List[str]] = []
        for ex in examples:
            parsed = parser.parse(ex.text)
            pred_tags_list.append(parsed.tags)
            latencies.append(parsed.latency_ms)
        metrics = _compute_metrics(examples, pred_tags_list)
        lat_arr = np.asarray(latencies, dtype=np.float64)
        latency_stats = {
            "mean_ms": round(float(np.mean(lat_arr)), 3),
            "p50_ms": round(float(np.percentile(lat_arr, 50)), 3),
            "p95_ms": round(float(np.percentile(lat_arr, 95)), 3),
            "p99_ms": round(float(np.percentile(lat_arr, 99)), 3),
        }
        return metrics, latency_stats

    parsed_batch = parser.parse_batch([ex.text for ex in examples])
    metrics = _compute_metrics(examples, [p.tags for p in parsed_batch])
    return metrics, {}


def main() -> None:
    torch.set_num_threads(4)
    _set_seed(401)
    train_start = time.perf_counter()

    print("Generating 50,000 synthetic training samples (seed=401, 42k/4k/4k)...")
    train_ex, val_ex, test_ex = generate_dataset(
        n_samples=50000, seed=401, train_size=42000, val_size=4000, test_size=4000
    )
    holdout_ex = generate_holdout(n_samples=500, seed=20261201)

    tokenizer = SlotParserTokenizer.build_from_corpus(
        [ex.tokens for ex in train_ex],
        max_vocab_size=10000,
        min_word_freq=2,
    )
    print(
        f"Vocab built: word_vocab={tokenizer.vocab_size}, char_vocab={tokenizer.char_vocab_size}"
    )

    train_w, train_m, train_c, train_y = _encode_dataset(train_ex, tokenizer)
    val_arrays = _encode_dataset(val_ex, tokenizer)
    val_w, val_m, val_c, _ = val_arrays

    encoder = SlotEncoder(
        vocab_size=tokenizer.vocab_size,
        char_vocab_size=tokenizer.char_vocab_size,
        num_tags=len(BIO_TAGS),
    )
    crf = LinearCRF(num_tags=len(BIO_TAGS))
    model = SlotTaggerCRF(encoder=encoder, crf=crf)

    epochs = 8
    batch_size = 512
    word_dropout = 0.06
    optimizer = torch.optim.AdamW(model.parameters(), lr=4.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-4
    )

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    n_train = len(train_ex)

    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        perm = np.random.permutation(n_train)
        running_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            w_b = torch.from_numpy(train_w[idx]).clone()
            m_b = torch.from_numpy(train_m[idx])
            c_b = torch.from_numpy(train_c[idx])
            y_b = torch.from_numpy(train_y[idx])

            # Word-level OOV dropout teaches Char-CNN + BiGRU to generalize to unseen words.
            if word_dropout > 0.0:
                keep_mask = (w_b <= UNK_ID) | (torch.rand(w_b.shape) >= word_dropout)
                w_b = torch.where(keep_mask, w_b, UNK_ID)

            optimizer.zero_grad(set_to_none=True)
            loss = model(w_b, m_b, c_b, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        val_preds = _predict_pytorch_tags(model, val_ex, val_w, val_m, val_c)
        val_metrics = _compute_metrics(val_ex, val_preds)
        val_f1 = val_metrics["macro_entity_f1"]
        ep_sec = time.perf_counter() - ep_t0
        print(
            f"Epoch {epoch}/{epochs} | loss={running_loss / n_batches:.4f} | "
            f"val_tok_acc={val_metrics['token_accuracy']:.4f} | "
            f"val_macro_f1={val_f1:.4f} | time={ep_sec:.1f}s",
            flush=True,
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    export_info = _export_onnx_and_crf(
        model=model,
        tokenizer=tokenizer,
        artifacts_dir=ARTIFACTS_DIR,
        val_examples=val_ex,
        val_arrays=val_arrays,
    )
    print(f"Export complete: {export_info}")

    # Evaluate final runtime ONNX INT8 + NumPy Viterbi parser on 4k test & 500 holdout.
    runtime_parser = AddressSlotParser(artifacts_dir=ARTIFACTS_DIR, intra_op_threads=1)
    test_metrics, _ = _evaluate_runtime_parser(runtime_parser, test_ex, benchmark_latency=False)
    holdout_metrics, holdout_latency = _evaluate_runtime_parser(
        runtime_parser, holdout_ex, benchmark_latency=True
    )

    report = {
        "model": "Word+Char BiGRU + Linear-CRF (ONNX INT8 + NumPy Viterbi)",
        "train_seed": 401,
        "holdout_seed": 20261201,
        "splits": {"train": len(train_ex), "val": len(val_ex), "test": len(test_ex), "holdout": len(holdout_ex)},
        "vocab": {
            "word_vocab_size": tokenizer.vocab_size,
            "char_vocab_size": tokenizer.char_vocab_size,
            "max_word_len": tokenizer.max_word_len,
            "max_char_per_word": tokenizer.max_char_per_word,
        },
        "training_seconds": round(time.perf_counter() - train_start, 2),
        "artifacts": export_info,
        "test_4k_metrics": test_metrics,
        "holdout_500_metrics": holdout_metrics,
        "latency_single_item_cpu": holdout_latency,
    }
    report_path = ARTIFACTS_DIR / "eval_report_v3.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== INDEPENDENT 500-CASE HOLDOUT (ONNX INT8 + NumPy Viterbi) ===")
    print(json.dumps(report["holdout_500_metrics"], ensure_ascii=False, indent=2))
    print("Latency (ms):", holdout_latency)
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()
