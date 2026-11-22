"""Full 3-Mode Cross-Evaluation for Laya Address VN v2 (Way A v2 & Way B v2).

Evaluates:
- heuristic_only, model_only, hybrid for BOTH Way A v2 and Way B v2
- 600-case independent holdout (3x3 confusion matrix, per-group & per-6-short-style false-block,
  11 mandatory garbage strings, insufficient recall, ECE, CPU p50/p95/p99 latency)
- Old 90-case benchmark AND de-leaked 85-case benchmark (excluding C01, NN02, NN03, NN09, NN10)
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
from typing import Any, Callable, Dict, List

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
HOLD_OUT_PATH = ROOT / "benchmark_v2" / "independent_holdout_v2.json"
OLD_BENCH_PATH = ROOT / "benchmark" / "vn_address_benchmark.json"
REPORT_OUT = ROOT / "benchmark_v2" / "eval_report_v2.json"

WAY_A_V2_DIR = ROOT / "way_a_browser_onnx_v2"
WAY_B_V2_DIR = ROOT / "way_b_backend_api_v2"

CLASSES = ["not_address", "address_insufficient", "address_sufficient"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
V1_LEAKED_IDS = {"C01", "NN02", "NN03", "NN09", "NN10"}

MANDATORY_GARBAGE = [
    "73", "10", "K", "Hi", "Ok", "Jean", "Xom",
    "Thôn 7", "Xin giá", "ngõ 12", "Cho mình 3 chai nướng BBQ",
]


def compute_binary_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
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


def compute_multiclass_ece(probs_3: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = np.max(probs_3, axis=1)
    pred = np.argmax(probs_3, axis=1)
    correct = (pred == labels).astype(np.float64)
    return compute_binary_ece(conf, correct, n_bins=n_bins)


def summarize_holdout(
    cases: List[Dict[str, Any]],
    preds: List[Dict[str, Any]],
) -> Dict[str, Any]:
    n = len(cases)
    y_true_3 = np.array([CLASS_TO_IDX[c["expected_sufficiency"]] for c in cases], dtype=np.int64)
    y_pred_3 = np.array([CLASS_TO_IDX[p["sufficiency"]] for p in preds], dtype=np.int64)
    y_true_bin = np.array([1 if c["expected_is_address"] else 0 for c in cases], dtype=np.int64)
    y_pred_bin = np.array([1 if p["is_address"] else 0 for p in preds], dtype=np.int64)

    p_bin = np.array([float(p["probability"]) for p in preds], dtype=np.float64)
    p_3 = np.array([p["sufficiency_probs"] for p in preds], dtype=np.float64)
    lats = np.array([float(p["latency_ms"]) for p in preds], dtype=np.float64)

    # 3x3 confusion matrix [true][pred]
    cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for yt, yp in zip(y_true_3, y_pred_3):
        cm[int(yt)][int(yp)] += 1

    # Macro F1
    f1s = []
    for k in range(3):
        tp_k = cm[k][k]
        fp_k = sum(cm[r][k] for r in range(3) if r != k)
        fn_k = sum(cm[k][c] for c in range(3) if c != k)
        pr_k = tp_k / max(1, tp_k + fp_k)
        rc_k = tp_k / max(1, tp_k + fn_k)
        f1s.append(2 * pr_k * rc_k / max(1e-12, pr_k + rc_k))

    tp = int(np.sum((y_pred_bin == 1) & (y_true_bin == 1)))
    tn = int(np.sum((y_pred_bin == 0) & (y_true_bin == 0)))
    fp = int(np.sum((y_pred_bin == 1) & (y_true_bin == 0)))
    fn = int(np.sum((y_pred_bin == 0) & (y_true_bin == 1)))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    bin_f1 = 2 * prec * rec / max(1e-12, prec + rec)

    # Group & subgroup metrics
    by_group: Dict[str, Dict[str, Any]] = {}
    short_styles: Dict[str, Dict[str, Any]] = {}
    insuf_subgroups: Dict[str, Dict[str, Any]] = {}

    for c, p in zip(cases, preds):
        grp = c["group"]
        sg = c["subgroup"]
        g_entry = by_group.setdefault(
            grp,
            {"total": 0, "correct_3class": 0, "correct_binary": 0, "passed_gate": 0, "cm": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]},
        )
        g_entry["total"] += 1
        yt_i = CLASS_TO_IDX[c["expected_sufficiency"]]
        yp_i = CLASS_TO_IDX[p["sufficiency"]]
        g_entry["cm"][yt_i][yp_i] += 1
        if yt_i == yp_i:
            g_entry["correct_3class"] += 1
        if p["is_address"] == c["expected_is_address"]:
            g_entry["correct_binary"] += 1
        if p["is_address"]:
            g_entry["passed_gate"] += 1

        if grp == "short_valid_address":
            se = short_styles.setdefault(sg, {"total": 0, "passed": 0, "blocked": 0, "correct_3class": 0})
            se["total"] += 1
            if p["is_address"]:
                se["passed"] += 1
            else:
                se["blocked"] += 1
            if yt_i == yp_i:
                se["correct_3class"] += 1

        if grp == "insufficient_address":
            ie = insuf_subgroups.setdefault(sg, {"total": 0, "tagged_insufficient": 0, "blocked_gate": 0})
            ie["total"] += 1
            if p["sufficiency"] == "address_insufficient":
                ie["tagged_insufficient"] += 1
            if not p["is_address"]:
                ie["blocked_gate"] += 1

    for se in short_styles.values():
        se["false_block_rate"] = round(se["blocked"] / max(1, se["total"]), 4)
        se["pass_rate"] = round(se["passed"] / max(1, se["total"]), 4)
    for g_entry in by_group.values():
        g_entry["acc_3class"] = round(g_entry["correct_3class"] / max(1, g_entry["total"]), 4)
        g_entry["acc_binary"] = round(g_entry["correct_binary"] / max(1, g_entry["total"]), 4)

    # Mandatory garbage checks
    text_to_pred = {c["text"]: p for c, p in zip(cases, preds)}
    mandatory_results = {}
    for m_txt in MANDATORY_GARBAGE:
        mp = text_to_pred.get(m_txt)
        if mp is not None:
            mandatory_results[m_txt] = {
                "blocked": (not mp["is_address"]),
                "sufficiency": mp["sufficiency"],
                "probability": round(float(mp["probability"]), 4),
            }

    # Garbage & Hard Negative block rates
    garbage_cases = [(c, p) for c, p in zip(cases, preds) if c["group"] == "garbage_and_chat"]
    hard_neg_cases = [(c, p) for c, p in zip(cases, preds) if c["group"] == "hard_negative"]
    insuf_cases = [(c, p) for c, p in zip(cases, preds) if c["group"] == "insufficient_address"]
    short_cases = [(c, p) for c, p in zip(cases, preds) if c["group"] == "short_valid_address"]

    mistakes = []
    for c, p in zip(cases, preds):
        if p["sufficiency"] != c["expected_sufficiency"] or p["is_address"] != c["expected_is_address"]:
            mistakes.append({
                "id": c["id"],
                "group": c["group"],
                "subgroup": c["subgroup"],
                "expected_suf": c["expected_sufficiency"],
                "pred_suf": p["sufficiency"],
                "expected_addr": c["expected_is_address"],
                "pred_addr": p["is_address"],
                "prob": round(float(p["probability"]), 4),
                "tier": p["tier"],
                "text": c["text"],
            })

    return {
        "total": n,
        "acc_3class": round(float(np.mean(y_true_3 == y_pred_3)), 4),
        "macro_f1_3class": round(float(np.mean(f1s)), 4),
        "acc_binary": round((tp + tn) / max(1, n), 4),
        "f1_binary": round(float(bin_f1), 4),
        "fp_binary": fp,
        "fn_binary": fn,
        "ece_binary": round(compute_binary_ece(p_bin, y_true_bin), 4),
        "ece_3class": round(compute_multiclass_ece(p_3, y_true_3), 4),
        "confusion_matrix_3x3": cm,
        "short_valid_pass_rate": round(sum(1 for _, p in short_cases if p["is_address"]) / max(1, len(short_cases)), 4),
        "short_valid_false_block_rate": round(sum(1 for _, p in short_cases if not p["is_address"]) / max(1, len(short_cases)), 4),
        "short_styles": short_styles,
        "insufficient_tagged_recall": round(sum(1 for _, p in insuf_cases if p["sufficiency"] == "address_insufficient") / max(1, len(insuf_cases)), 4),
        "insufficient_gate_block_rate": round(sum(1 for _, p in insuf_cases if not p["is_address"]) / max(1, len(insuf_cases)), 4),
        "insufficient_subgroups": insuf_subgroups,
        "garbage_block_rate": round(sum(1 for _, p in garbage_cases if not p["is_address"]) / max(1, len(garbage_cases)), 4),
        "hard_negative_block_rate": round(sum(1 for _, p in hard_neg_cases if not p["is_address"]) / max(1, len(hard_neg_cases)), 4),
        "mandatory_garbage_checks": mandatory_results,
        "latency_ms": {
            "mean": round(float(np.mean(lats)), 3),
            "p50": round(float(np.percentile(lats, 50)), 3),
            "p95": round(float(np.percentile(lats, 95)), 3),
            "p99": round(float(np.percentile(lats, 99)), 3),
        },
        "per_group": by_group,
        "mistake_count": len(mistakes),
        "mistakes_sample": mistakes[:25],
    }


def summarize_old_benchmark(cases: List[Dict[str, Any]], preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _eval_subset(sub_pairs):
        y_t = np.array([1 if c["expected"] else 0 for c, _ in sub_pairs], dtype=np.int64)
        y_p = np.array([1 if p["is_address"] else 0 for _, p in sub_pairs], dtype=np.int64)
        tp = int(np.sum((y_p == 1) & (y_t == 1)))
        tn = int(np.sum((y_p == 0) & (y_t == 0)))
        fp = int(np.sum((y_p == 1) & (y_t == 0)))
        fn = int(np.sum((y_p == 0) & (y_t == 1)))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        misses = [
            {"id": c["id"], "expected": c["expected"], "pred": p["is_address"], "suf": p["sufficiency"], "prob": round(float(p["probability"]), 4), "text": c["text"]}
            for c, p in sub_pairs if p["is_address"] != c["expected"]
        ]
        return {
            "total": len(sub_pairs),
            "correct": tp + tn,
            "accuracy": round((tp + tn) / max(1, len(sub_pairs)), 4),
            "f1": round(float(f1), 4),
            "fp": fp,
            "fn": fn,
            "mistakes": misses,
        }

    pairs_all = list(zip(cases, preds))
    pairs_deleaked = [(c, p) for c, p in pairs_all if c["id"] not in V1_LEAKED_IDS]
    return {
        "full_90": _eval_subset(pairs_all),
        "deleaked_85": _eval_subset(pairs_deleaked),
    }


# ------------------------------------------------------------------
# Way A v2 Runner (Node heuristics.js + ORT Single-Thread Python)
# ------------------------------------------------------------------
class WayAV2Runner:
    def __init__(self) -> None:
        model_dir = WAY_A_V2_DIR / "web" / "model"
        vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
        calib = json.loads((model_dir / "calibration.json").read_text(encoding="utf-8"))
        self.char_to_id = vocab["char_to_id"]
        self.unk_id = vocab["unk_id"]
        self.max_len = int(vocab["max_length"])
        self.temp = float(calib["temperature"])
        self.threshold = float(calib["threshold"])
        self.t_suf = float(calib["sufficiency_temperature"])
        self.classes = calib["sufficiency_classes"]

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            str(model_dir / "laya_address_vn.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        # Warmup
        self._run_onnx_one("Lộc Thạnh, Lộc Ninh")

    def _encode_one(self, text: str) -> np.ndarray:
        norm = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "").strip()).lower()
        arr = np.zeros((1, self.max_len), dtype=np.int64)
        for i, ch in enumerate(norm[: self.max_len]):
            arr[0, i] = self.char_to_id.get(ch, self.unk_id)
        return arr

    def _run_onnx_one(self, text: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        x = self._encode_one(text)
        logit_arr, suf_arr = self.sess.run(["logit", "sufficiency_logits"], {"input_ids": x})
        logit = float(logit_arr.reshape(-1)[0])
        suf_raw = suf_arr.reshape(-1).astype(np.float64)

        prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit / self.temp))))
        scaled = suf_raw / self.t_suf
        scaled -= np.max(scaled)
        exps = np.exp(np.clip(scaled, -30.0, 30.0))
        suf_probs = (exps / np.sum(exps)).tolist()
        best_idx = int(np.argmax(suf_probs))
        suf_label = self.classes[best_idx]
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "is_address": bool(prob >= self.threshold),
            "probability": prob,
            "sufficiency": suf_label,
            "sufficiency_probs": suf_probs,
            "tier": "Tier 2 (ONNX)",
            "latency_ms": lat_ms,
        }

    def _run_node_heuristics(self, texts: List[str], fn_name: str) -> List[Dict[str, Any]]:
        heur_js = str(WAY_A_V2_DIR / "web" / "heuristics.js").replace("\\", "\\\\")
        gaz_json = str(WAY_A_V2_DIR / "web" / "model" / "vn_admin_gazetteer.json").replace("\\", "\\\\")
        script = (
            f"const fs = require('fs');"
            f"const H = require('{heur_js}');"
            f"H.initGazetteer(JSON.parse(fs.readFileSync('{gaz_json}', 'utf8')));"
            "let buf = '';"
            "process.stdin.on('data', d => buf += d);"
            "process.stdin.on('end', () => {"
            "  const arr = JSON.parse(buf);"
            f"  const out = arr.map(t => H.{fn_name}(t));"
            "  process.stdout.write(JSON.stringify(out));"
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

    def predict_batch(self, texts: List[str], mode: str) -> List[Dict[str, Any]]:
        if mode == "model_only":
            return [self._run_onnx_one(t) for t in texts]
        if mode == "heuristic_only":
            raw = self._run_node_heuristics(texts, "evaluateHeuristicOnly")
            return [
                {
                    "is_address": bool(r["is_address"]),
                    "probability": float(r["probability"]),
                    "sufficiency": r["sufficiency"],
                    "sufficiency_probs": r["sufficiency_probs"],
                    "tier": r["tier"],
                    "latency_ms": float(r["latency_ms"]),
                }
                for r in raw
            ]
        # hybrid
        t1_list = self._run_node_heuristics(texts, "evaluateTier1")
        out: List[Dict[str, Any]] = []
        for t, r in zip(texts, t1_list):
            if r["decision"] != "ambiguous":
                out.append({
                    "is_address": bool(r["is_address"]),
                    "probability": float(r["probability"]),
                    "sufficiency": r["sufficiency"],
                    "sufficiency_probs": r["sufficiency_probs"],
                    "tier": r["tier"],
                    "latency_ms": float(r["latency_ms"]),
                })
            else:
                t2 = self._run_onnx_one(t)
                t2["latency_ms"] += float(r["latency_ms"])
                t2["tier"] = "Tier 2 (ONNX)"
                out.append(t2)
        return out


# ------------------------------------------------------------------
# Way B v2 Runner (LayaAddressEngine INT8 + Gazetteer)
# ------------------------------------------------------------------
def make_way_b_runner():
    sys.path.insert(0, str(WAY_B_V2_DIR))
    from app.engine import LayaAddressEngine

    engine = LayaAddressEngine(artifacts_dir=WAY_B_V2_DIR / "artifacts")

    def predict(texts: List[str], mode: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for t in texts:
            d = engine.decide(t, mode=mode)
            sp = [
                float(d.sufficiency_probs["not_address"]),
                float(d.sufficiency_probs["address_insufficient"]),
                float(d.sufficiency_probs["address_sufficient"]),
            ]
            out.append({
                "is_address": bool(d.is_address),
                "probability": float(d.probability),
                "sufficiency": d.sufficiency,
                "sufficiency_probs": sp,
                "tier": d.tier,
                "latency_ms": float(d.latency_ms),
            })
        return out

    return predict


def main() -> int:
    holdout_cases = json.loads(HOLD_OUT_PATH.read_text(encoding="utf-8"))["cases"]
    old_cases = json.loads(OLD_BENCH_PATH.read_text(encoding="utf-8"))["cases"]

    hold_texts = [c["text"] for c in holdout_cases]
    old_texts = [c["text"] for c in old_cases]

    runner_a = WayAV2Runner()
    runner_b = make_way_b_runner()

    configs: List[tuple[str, Callable[[List[str], str], List[Dict[str, Any]]], str]] = [
        ("way_a_v2_heuristic_only", runner_a.predict_batch, "heuristic_only"),
        ("way_a_v2_model_only", runner_a.predict_batch, "model_only"),
        ("way_a_v2_hybrid", runner_a.predict_batch, "hybrid"),
        ("way_b_v2_heuristic_only", runner_b, "heuristic_only"),
        ("way_b_v2_model_only", runner_b, "model_only"),
        ("way_b_v2_hybrid", runner_b, "hybrid"),
    ]

    report: Dict[str, Any] = {
        "holdout_total": len(holdout_cases),
        "old_total": len(old_cases),
        "deleaked_old_total": sum(1 for c in old_cases if c["id"] not in V1_LEAKED_IDS),
        "results": {},
    }

    print("=== Independent 600-Case Hold-Out & Old 90/85 Benchmark ===")
    for name, fn, mode in configs:
        preds_h = fn(hold_texts, mode)
        preds_o = fn(old_texts, mode)
        res_h = summarize_holdout(holdout_cases, preds_h)
        res_o = summarize_old_benchmark(old_cases, preds_o)
        report["results"][name] = {"holdout_600": res_h, "old_benchmark": res_o}

        print(
            f"\n[{name}]\n"
            f"  600-Holdout: 3cls={res_h['acc_3class']*100:.2f}% | bin={res_h['acc_binary']*100:.2f}% "
            f"| ShortBlk={res_h['short_valid_false_block_rate']*100:.2f}% "
            f"| InsufRec={res_h['insufficient_tagged_recall']*100:.2f}% "
            f"| GarbBlk={res_h['garbage_block_rate']*100:.2f}% "
            f"| HardNegBlk={res_h['hard_negative_block_rate']*100:.2f}% "
            f"| ECE_bin={res_h['ece_binary']:.4f} | p99={res_h['latency_ms']['p99']:.3f}ms"
        )
        print("  6-Style Short Block Rates:", {k: f"{v['blocked']}/{v['total']} ({v['false_block_rate']*100:.1f}%)" for k, v in res_h["short_styles"].items()})
        print(
            f"  Old Bench : full_90={res_o['full_90']['correct']}/90 ({res_o['full_90']['accuracy']*100:.2f}%) "
            f"| deleaked_85={res_o['deleaked_85']['correct']}/85 ({res_o['deleaked_85']['accuracy']*100:.2f}%)"
        )

    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved full report to {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
