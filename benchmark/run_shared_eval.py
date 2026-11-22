"""Cross-evaluation runner comparing Way A (Browser ONNX) & Way B (CPU Backend API)
against the held-out 90-case Vietnamese Customer Address Benchmark.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = ROOT_DIR / "benchmark" / "vn_address_benchmark.json"
REPORT_PATH = ROOT_DIR / "benchmark" / "shared_eval_report.json"

WAY_A_DIR = ROOT_DIR / "way_a_browser_onnx"
WAY_B_DIR = ROOT_DIR / "way_b_backend_api"


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
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


def summarize(cases: List[Dict[str, Any]], preds: List[bool], probs: List[float], latencies_ms: List[float]) -> Dict[str, Any]:
    y_true = np.array([1 if c["expected"] else 0 for c in cases], dtype=np.int64)
    y_pred = np.array([1 if p else 0 for p in preds], dtype=np.int64)
    p_arr = np.array(probs, dtype=np.float64)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    acc = (tp + tn) / max(1, len(cases))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    ece = compute_ece(p_arr, y_true)

    by_cat: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    mistakes: List[Dict[str, Any]] = []
    for c, pred, prob in zip(cases, preds, probs):
        cat = c["category"]
        by_cat[cat]["total"] += 1
        if pred == c["expected"]:
            by_cat[cat]["correct"] += 1
        else:
            mistakes.append({
                "id": c["id"],
                "category": cat,
                "expected": c["expected"],
                "predicted": pred,
                "probability": round(float(prob), 4),
                "text": c["text"],
            })

    return {
        "total": len(cases),
        "correct": tp + tn,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "ece": round(ece, 4),
        "fp": fp,
        "fn": fn,
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 3),
        "per_category": {k: dict(v) for k, v in by_cat.items()},
        "mistakes": mistakes,
    }


def eval_way_a(cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    vocab = json.loads((WAY_A_DIR / "web" / "model" / "vocab.json").read_text(encoding="utf-8"))
    calib = json.loads((WAY_A_DIR / "web" / "model" / "calibration.json").read_text(encoding="utf-8"))
    char_to_id = vocab["char_to_id"]
    unk_id = vocab["unk_id"]
    max_len = vocab["max_length"]
    temp = float(calib["temperature"])
    thresh = float(calib["threshold"])

    sess = ort.InferenceSession(
        str(WAY_A_DIR / "web" / "model" / "laya_address_vn.onnx"),
        providers=["CPUExecutionProvider"],
    )

    def encode_one(text: str) -> np.ndarray:
        norm = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "").strip()).lower()
        arr = np.zeros((1, max_len), dtype=np.int64)
        for i, ch in enumerate(norm[:max_len]):
            arr[0, i] = char_to_id.get(ch, unk_id)
        return arr

    # Warmup
    sess.run(["logit"], {"input_ids": encode_one("45 Lê Lợi, Q1, TP.HCM")})

    # 1. Pure ONNX
    onnx_preds: List[bool] = []
    onnx_probs: List[float] = []
    onnx_lats: List[float] = []
    for c in cases:
        t0 = time.perf_counter()
        x = encode_one(c["text"])
        logit = float(sess.run(["logit"], {"input_ids": x})[0].reshape(-1)[0])
        prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit / temp))))
        onnx_lats.append((time.perf_counter() - t0) * 1000.0)
        onnx_probs.append(prob)
        onnx_preds.append(prob >= thresh)

    # 2. Hybrid (Tier-1 Node heuristics.js + Tier-2 ONNX)
    js_path = str(WAY_A_DIR / "web" / "heuristics.js").replace("\\", "\\\\")
    node_script = (
        f"const H = require('{js_path}');"
        "let buf = '';"
        "process.stdin.on('data', d => buf += d);"
        "process.stdin.on('end', () => {"
        "  const items = JSON.parse(buf);"
        "  const out = items.map(t => H.evaluateTier1(t));"
        "  process.stdout.write(JSON.stringify(out));"
        "});"
    )
    proc = subprocess.run(
        ["node", "-e", node_script],
        input=json.dumps([c["text"] for c in cases], ensure_ascii=False),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    t1_results = json.loads(proc.stdout)

    hyb_preds: List[bool] = []
    hyb_probs: List[float] = []
    hyb_lats: List[float] = []
    for i, t1 in enumerate(t1_results):
        if t1["decision"] != "ambiguous":
            hyb_preds.append(bool(t1["is_address"]))
            hyb_probs.append(float(t1["probability"]))
            hyb_lats.append(float(t1["latency_ms"]))
        else:
            hyb_preds.append(onnx_preds[i])
            hyb_probs.append(onnx_probs[i])
            hyb_lats.append(float(t1["latency_ms"]) + onnx_lats[i])

    return {
        "way_a_model_only": summarize(cases, onnx_preds, onnx_probs, onnx_lats),
        "way_a_hybrid": summarize(cases, hyb_preds, hyb_probs, hyb_lats),
    }


def eval_way_b(cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    sys.path.insert(0, str(WAY_B_DIR))
    from app.engine import LayaAddressEngine

    engine = LayaAddressEngine(artifacts_dir=WAY_B_DIR / "artifacts")
    results: Dict[str, Dict[str, Any]] = {}
    for mode in ("model_only", "hybrid"):
        preds: List[bool] = []
        probs: List[float] = []
        lats: List[float] = []
        for c in cases:
            dec = engine.decide(c["text"], mode=mode)
            preds.append(dec.is_address)
            probs.append(dec.probability)
            lats.append(dec.latency_ms)
        results[f"way_b_{mode}"] = summarize(cases, preds, probs, lats)
    return results


def main() -> int:
    data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    cases = data["cases"]
    report = {
        "benchmark_count": len(cases),
        "positives": sum(1 for c in cases if c["expected"]),
        "negatives": sum(1 for c in cases if not c["expected"]),
    }
    report.update(eval_way_a(cases))
    report.update(eval_way_b(cases))
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"=== Shared 90-Case VN Customer Address Benchmark ({report['positives']} pos / {report['negatives']} neg) ===")
    for key in ("way_a_model_only", "way_a_hybrid", "way_b_model_only", "way_b_hybrid"):
        m = report[key]
        print(
            f"{key:18s} | Acc: {m['accuracy']*100:6.2f}% ({m['correct']}/{m['total']}) "
            f"| F1: {m['f1']*100:6.2f}% | FP: {m['fp']:2d} | FN: {m['fn']:2d} "
            f"| ECE: {m['ece']:.4f} | Lat: {m['mean_latency_ms']:.3f} ms"
        )
        if m["mistakes"]:
            for err in m["mistakes"]:
                print(f"    MISS [{err['id']}|{err['category']}] exp={err['expected']} p={err['probability']:.4f} | {err['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
