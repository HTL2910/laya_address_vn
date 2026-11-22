"""Thread-safe Laya Address VN v2 Inference Engine (ONNX Runtime INT8 + Gazetteer Heuristics).

Supports:
- 3-class sufficiency: 'not_address' | 'address_insufficient' | 'address_sufficient'
- Backward-compatible binary decision: is_address == True iff sufficiency == 'address_sufficient'
- mode = "hybrid" (Tier-1 Gazetteer filter -> Tier-2 INT8 ONNX model for ambiguous inputs)
- mode = "model_only" (always invoke INT8 ONNX model)
- mode = "heuristic_only" (pure Tier-1 Gazetteer heuristic scoring, < 0.3 ms)
- Zero PyTorch or pickle imports at runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort

from app.heuristics import (
    GazetteerIndex,
    SufficiencyLabel,
    evaluate_heuristic,
)
from app.tokenizer import DualGranularityTokenizer

InferenceMode = Literal["hybrid", "model_only", "heuristic_only"]
DecisionTier = Literal[
    "heuristic_reject",
    "heuristic_insufficient",
    "heuristic_accept",
    "heuristic_only",
    "onnx_int8",
]

SUFFICIENCY_CLASSES: Tuple[SufficiencyLabel, ...] = (
    "not_address",
    "address_insufficient",
    "address_sufficient",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS_DIR = BASE_DIR / "artifacts"


@dataclass(frozen=True)
class NoulDecision:
    text: str
    is_address: bool
    decision: Literal["yes", "no"]
    probability: float
    confidence: float
    sufficiency: SufficiencyLabel
    sufficiency_probs: Dict[str, float]
    sufficiency_score: float
    tier: DecisionTier
    reason: str
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _softmax_array(x: np.ndarray) -> np.ndarray:
    x_shift = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(np.clip(x_shift, -40.0, 40.0))
    return e / np.sum(e, axis=-1, keepdims=True)


class LayaAddressEngine:
    """Thread-safe CPU inference engine backed by ONNX Runtime INT8 + VN Admin Gazetteer."""

    def __init__(
        self,
        artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
        model_filename: str = "laya_address_vn_int8.onnx",
        intra_op_threads: Optional[int] = None,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.model_path = self.artifacts_dir / model_filename
        self.vocab_path = self.artifacts_dir / "vocab.json"
        self.calibration_path = self.artifacts_dir / "calibration.json"
        self.gazetteer_path = self.artifacts_dir / "vn_admin_gazetteer.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Tokenizer vocab not found: {self.vocab_path}")
        if not self.calibration_path.exists():
            raise FileNotFoundError(f"Calibration metadata not found: {self.calibration_path}")
        if not self.gazetteer_path.exists():
            raise FileNotFoundError(f"Gazetteer not found: {self.gazetteer_path}")

        self.tokenizer = DualGranularityTokenizer.load(self.vocab_path)
        self.gazetteer = GazetteerIndex(self.gazetteer_path)
        self.metadata: Dict[str, Any] = json.loads(
            self.calibration_path.read_text(encoding="utf-8")
        )
        cal = self.metadata.get("calibration", {})
        self.temperature = float(cal.get("temperature", 1.0))
        self.threshold = float(cal.get("threshold", 0.5))
        self.suf_temperature = float(cal.get("sufficiency_temperature", 1.0))

        cpu_count = os.cpu_count() or 4
        if intra_op_threads is None:
            intra_op_threads = int(
                self.metadata.get("onnx_comparison", {}).get(
                    "ort_intra_op_threads",
                    min(2, max(1, cpu_count // 4)),
                )
            )
        self.intra_op_threads = intra_op_threads

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = self.intra_op_threads
        sess_opts.inter_op_num_threads = 1
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._lock = threading.RLock()

        # Warmup once on startup so first live request has zero cold JIT jitter
        self._run_model_batch(["Lộc Thạnh, Lộc Ninh"])

    def _run_model_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Run vectorized ONNX INT8 inference and return calibrated 3-class probabilities [N, 3]."""
        if not texts:
            return np.zeros((0, 3), dtype=np.float64)
        word_ids, attention_mask, char_ids = self.tokenizer.encode_batch(texts)
        with self._lock:
            _bin_logits, suf_logits = self._session.run(
                ["logits", "sufficiency_logits"],
                {
                    "word_ids": word_ids,
                    "attention_mask": attention_mask,
                    "char_ids": char_ids,
                },
            )
        suf_arr = np.asarray(suf_logits, dtype=np.float64)
        return _softmax_array(suf_arr / max(self.suf_temperature, 1e-4))

    @staticmethod
    def _build_decision_from_probs(
        text: str,
        probs_3: Tuple[float, float, float],
        tier: DecisionTier,
        reason: str,
        latency_ms: float,
    ) -> NoulDecision:
        p0, p1, p2 = float(probs_3[0]), float(probs_3[1]), float(probs_3[2])
        total = max(1e-9, p0 + p1 + p2)
        p0, p1, p2 = p0 / total, p1 / total, p2 / total

        cls_idx = int(np.argmax([p0, p1, p2]))
        sufficiency: SufficiencyLabel = SUFFICIENCY_CLASSES[cls_idx]
        is_addr = sufficiency == "address_sufficient"

        prob_suff = min(1.0 - 1e-6, max(1e-6, p2))
        conf = min(1.0 - 1e-6, max(1e-6, [p0, p1, p2][cls_idx]))

        return NoulDecision(
            text=text,
            is_address=is_addr,
            decision="yes" if is_addr else "no",
            probability=round(prob_suff, 6),
            confidence=round(conf, 6),
            sufficiency=sufficiency,
            sufficiency_probs={
                "not_address": round(p0, 6),
                "address_insufficient": round(p1, 6),
                "address_sufficient": round(p2, 6),
            },
            sufficiency_score=round(prob_suff, 6),
            tier=tier,
            reason=reason,
            latency_ms=round(latency_ms, 3),
        )

    def decide(self, text: str, mode: InferenceMode = "hybrid") -> NoulDecision:
        """Evaluate a single Vietnamese string with calibrated 3-class sufficiency + Noul decision."""
        t0 = time.perf_counter()

        if mode == "heuristic_only":
            h = evaluate_heuristic(text, gazetteer=self.gazetteer)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            sp = h.sufficiency_probs
            return self._build_decision_from_probs(
                text=text,
                probs_3=(sp["not_address"], sp["address_insufficient"], sp["address_sufficient"]),
                tier="heuristic_only",
                reason=h.reason,
                latency_ms=elapsed_ms,
            )

        if mode == "hybrid":
            h = evaluate_heuristic(text, gazetteer=self.gazetteer)
            if h.route == "clear_reject":
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                sp = h.sufficiency_probs
                return self._build_decision_from_probs(
                    text=text,
                    probs_3=(sp["not_address"], sp["address_insufficient"], sp["address_sufficient"]),
                    tier="heuristic_reject",
                    reason=h.reason,
                    latency_ms=elapsed_ms,
                )
            if h.route == "clear_insufficient":
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                sp = h.sufficiency_probs
                return self._build_decision_from_probs(
                    text=text,
                    probs_3=(sp["not_address"], sp["address_insufficient"], sp["address_sufficient"]),
                    tier="heuristic_insufficient",
                    reason=h.reason,
                    latency_ms=elapsed_ms,
                )
            if h.route == "clear_sufficient":
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                sp = h.sufficiency_probs
                return self._build_decision_from_probs(
                    text=text,
                    probs_3=(sp["not_address"], sp["address_insufficient"], sp["address_sufficient"]),
                    tier="heuristic_accept",
                    reason=h.reason,
                    latency_ms=elapsed_ms,
                )

        # Tier-2 ONNX INT8 model inference (mode == "model_only" or hybrid ambiguous)
        probs = self._run_model_batch([text])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        p0, p1, p2 = float(probs[0, 0]), float(probs[0, 1]), float(probs[0, 2])
        return self._build_decision_from_probs(
            text=text,
            probs_3=(p0, p1, p2),
            tier="onnx_int8",
            reason="onnx_int8_calibrated_sufficiency",
            latency_ms=elapsed_ms,
        )

    def decide_batch(
        self,
        texts: Sequence[str],
        mode: InferenceMode = "hybrid",
        chunk_size: int = 64,
    ) -> Dict[str, Any]:
        """Evaluate a batch of strings with vectorized ONNX INT8 inference."""
        t0 = time.perf_counter()
        n = len(texts)
        if n == 0:
            return {
                "mode": mode,
                "count": 0,
                "total_latency_ms": 0.0,
                "avg_latency_ms": 0.0,
                "throughput_req_per_s": 0.0,
                "tier_counts": {},
                "sufficiency_counts": {},
                "results": [],
            }

        results: List[Optional[NoulDecision]] = [None] * n
        model_indices: List[int] = []
        model_texts: List[str] = []

        if mode == "heuristic_only":
            for idx, txt in enumerate(texts):
                ti = time.perf_counter()
                h = evaluate_heuristic(txt, gazetteer=self.gazetteer)
                sp = h.sufficiency_probs
                results[idx] = self._build_decision_from_probs(
                    text=txt,
                    probs_3=(sp["not_address"], sp["address_insufficient"], sp["address_sufficient"]),
                    tier="heuristic_only",
                    reason=h.reason,
                    latency_ms=(time.perf_counter() - ti) * 1000.0,
                )
        elif mode == "hybrid":
            route_to_tier: Dict[str, DecisionTier] = {
                "clear_reject": "heuristic_reject",
                "clear_insufficient": "heuristic_insufficient",
                "clear_sufficient": "heuristic_accept",
            }
            for idx, txt in enumerate(texts):
                ti = time.perf_counter()
                h = evaluate_heuristic(txt, gazetteer=self.gazetteer)
                if h.route in route_to_tier:
                    sp = h.sufficiency_probs
                    results[idx] = self._build_decision_from_probs(
                        text=txt,
                        probs_3=(sp["not_address"], sp["address_insufficient"], sp["address_sufficient"]),
                        tier=route_to_tier[h.route],
                        reason=h.reason,
                        latency_ms=(time.perf_counter() - ti) * 1000.0,
                    )
                else:
                    model_indices.append(idx)
                    model_texts.append(txt)
        else:
            model_indices = list(range(n))
            model_texts = list(texts)

        if model_texts:
            for start in range(0, len(model_texts), chunk_size):
                sub_texts = model_texts[start : start + chunk_size]
                sub_indices = model_indices[start : start + chunk_size]
                tc = time.perf_counter()
                probs = self._run_model_batch(sub_texts)
                per_item_ms = ((time.perf_counter() - tc) * 1000.0) / max(1, len(sub_texts))
                for idx, txt, row in zip(sub_indices, sub_texts, probs):
                    results[idx] = self._build_decision_from_probs(
                        text=txt,
                        probs_3=(float(row[0]), float(row[1]), float(row[2])),
                        tier="onnx_int8",
                        reason="onnx_int8_calibrated_sufficiency",
                        latency_ms=per_item_ms,
                    )

        total_ms = (time.perf_counter() - t0) * 1000.0
        final_results = [r for r in results if r is not None]
        tier_counts: Dict[str, int] = {}
        sufficiency_counts: Dict[str, int] = {}
        for r in final_results:
            tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1
            sufficiency_counts[r.sufficiency] = sufficiency_counts.get(r.sufficiency, 0) + 1

        return {
            "mode": mode,
            "count": n,
            "total_latency_ms": round(total_ms, 3),
            "avg_latency_ms": round(total_ms / n, 4),
            "throughput_req_per_s": round((n / max(total_ms / 1000.0, 1e-9)), 1),
            "tier_counts": tier_counts,
            "sufficiency_counts": sufficiency_counts,
            "results": [r.to_dict() for r in final_results],
        }

    def get_info(self) -> Dict[str, Any]:
        """Return architecture, quantization size comparison, calibration, and CPU benchmark stats."""
        onnx_cmp = self.metadata.get("onnx_comparison", {})
        cal = self.metadata.get("calibration", {})
        return {
            "service": "Laya for Address VN v2 — Way B CPU Backend Microservice",
            "model_name": self.metadata.get("model_name", "laya-address-vn-way-b-v2-int8"),
            "version": self.metadata.get("version", "2.0.0"),
            "runtime": {
                "engine": "onnxruntime",
                "version": ort.__version__,
                "provider": "CPUExecutionProvider",
                "active_model": self.model_path.name,
                "intra_op_threads": self.intra_op_threads,
                "gazetteer_counts": {
                    "provinces": len(self.gazetteer.provinces),
                    "districts": len(self.gazetteer.districts),
                    "wards": len(self.gazetteer.wards),
                },
            },
            "architecture": self.metadata.get("architecture", {}),
            "total_parameters": self.metadata.get("total_parameters", 0),
            "dataset": self.metadata.get("dataset", {}),
            "calibration": {
                "temperature": self.temperature,
                "threshold": self.threshold,
                "sufficiency_temperature": self.suf_temperature,
                "sufficiency_classes": cal.get("sufficiency_classes", list(SUFFICIENCY_CLASSES)),
                "val_pre_calibration": cal.get("val_pre_calibration", {}),
                "val_post_calibration": cal.get("val_post_calibration", {}),
                "test_post_calibration": cal.get("test_post_calibration", {}),
                "val_sufficiency_pre_calibration": cal.get("val_sufficiency_pre_calibration", {}),
                "val_sufficiency_post_calibration": cal.get("val_sufficiency_post_calibration", {}),
                "test_sufficiency_post_calibration": cal.get("test_sufficiency_post_calibration", {}),
            },
            "quantization": {
                "fp32_size_mb": onnx_cmp.get("fp32_size_mb"),
                "int8_size_mb": onnx_cmp.get("int8_size_mb"),
                "fp32_size_bytes": onnx_cmp.get("fp32_size_bytes"),
                "int8_size_bytes": onnx_cmp.get("int8_size_bytes"),
                "size_reduction_percent": onnx_cmp.get("size_reduction_percent"),
                "fp32_test_metrics": onnx_cmp.get("fp32_test_metrics", {}),
                "int8_test_metrics": onnx_cmp.get("int8_test_metrics", {}),
                "fp32_test_sufficiency_metrics": onnx_cmp.get("fp32_test_sufficiency_metrics", {}),
                "int8_test_sufficiency_metrics": onnx_cmp.get("int8_test_sufficiency_metrics", {}),
                "fp32_cpu_benchmark": onnx_cmp.get("fp32_cpu_benchmark", {}),
                "int8_cpu_benchmark": onnx_cmp.get("int8_cpu_benchmark", {}),
            },
        }
