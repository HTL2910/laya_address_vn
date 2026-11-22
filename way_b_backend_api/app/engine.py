"""Thread-safe Laya Address VN Inference Engine (ONNX Runtime INT8 + Tier-1 Heuristics).

Supports:
- mode = "hybrid" (Tier-1 fast filter -> Tier-2 INT8 ONNX model for ambiguous inputs)
- mode = "model_only" (always invoke INT8 ONNX model)
- mode = "heuristic_only" (pure Tier-1 heuristic scoring, < 0.2 ms)
- Single & vectorized batch inference with calibrated probability sigmoid(logit / T).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import onnxruntime as ort

from app.heuristics import evaluate_heuristic
from app.tokenizer import DualGranularityTokenizer

InferenceMode = Literal["hybrid", "model_only", "heuristic_only"]
DecisionTier = Literal["heuristic_reject", "heuristic_accept", "heuristic_only", "onnx_int8"]

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS_DIR = BASE_DIR / "artifacts"


@dataclass(frozen=True)
class NoulDecision:
    text: str
    is_address: bool
    decision: Literal["yes", "no"]
    probability: float
    confidence: float
    tier: DecisionTier
    reason: str
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sigmoid_scalar(x: float) -> float:
    x = max(-40.0, min(40.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _sigmoid_array(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


class LayaAddressEngine:
    """Thread-safe CPU inference engine backed by ONNX Runtime INT8 + calibrated Noul head."""

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

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Tokenizer vocab not found: {self.vocab_path}")
        if not self.calibration_path.exists():
            raise FileNotFoundError(f"Calibration metadata not found: {self.calibration_path}")

        self.tokenizer = DualGranularityTokenizer.load(self.vocab_path)
        self.metadata: Dict[str, Any] = json.loads(
            self.calibration_path.read_text(encoding="utf-8")
        )
        cal = self.metadata.get("calibration", {})
        self.temperature = float(cal.get("temperature", 1.0))
        self.threshold = float(cal.get("threshold", 0.5))

        cpu_count = os.cpu_count() or 4
        if intra_op_threads is None:
            intra_op_threads = int(
                self.metadata.get("onnx_comparison", {}).get(
                    "ort_intra_op_threads",
                    min(4, max(1, cpu_count // 4)),
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
        # Reentrant lock ensures safe concurrent thread access if needed while keeping
        # single/batch calls serialized cleanly against ORT scratch buffers on CPU.
        self._lock = threading.RLock()

        # Warmup once on startup so first live request has zero cold JIT jitter
        self._run_model_batch(["Số 18 Ngõ 25 Láng Hạ, Đống Đa, Hà Nội"])

    def _run_model_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Run vectorized ONNX INT8 inference and return calibrated probabilities."""
        if not texts:
            return np.zeros((0,), dtype=np.float64)
        word_ids, attention_mask, char_ids = self.tokenizer.encode_batch(texts)
        with self._lock:
            (logits,) = self._session.run(
                ["logits"],
                {
                    "word_ids": word_ids,
                    "attention_mask": attention_mask,
                    "char_ids": char_ids,
                },
            )
        logits_arr = np.asarray(logits, dtype=np.float64).reshape(-1)
        return _sigmoid_array(logits_arr / self.temperature)

    @staticmethod
    def _build_decision(
        text: str,
        prob: float,
        threshold: float,
        tier: DecisionTier,
        reason: str,
        latency_ms: float,
    ) -> NoulDecision:
        # Clamp strictly into (0, 1) for calibrated probability guarantees
        prob_clamped = min(1.0 - 1e-6, max(1e-6, float(prob)))
        is_addr = prob_clamped >= threshold
        conf = prob_clamped if is_addr else (1.0 - prob_clamped)
        return NoulDecision(
            text=text,
            is_address=is_addr,
            decision="yes" if is_addr else "no",
            probability=round(prob_clamped, 6),
            confidence=round(conf, 6),
            tier=tier,
            reason=reason,
            latency_ms=round(latency_ms, 3),
        )

    def decide(self, text: str, mode: InferenceMode = "hybrid") -> NoulDecision:
        """Evaluate a single Vietnamese string with calibrated Noul decision primitives."""
        t0 = time.perf_counter()

        if mode == "heuristic_only":
            h = evaluate_heuristic(text)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return self._build_decision(
                text=text,
                prob=h.probability,
                threshold=0.5,
                tier="heuristic_only",
                reason=h.reason,
                latency_ms=elapsed_ms,
            )

        if mode == "hybrid":
            h = evaluate_heuristic(text)
            if h.route == "clear_reject":
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return self._build_decision(
                    text=text,
                    prob=h.probability,
                    threshold=0.5,
                    tier="heuristic_reject",
                    reason=h.reason,
                    latency_ms=elapsed_ms,
                )
            if h.route == "clear_address":
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return self._build_decision(
                    text=text,
                    prob=h.probability,
                    threshold=0.5,
                    tier="heuristic_accept",
                    reason=h.reason,
                    latency_ms=elapsed_ms,
                )

        # Tier-2 ONNX INT8 model inference (mode == "model_only" or hybrid ambiguous)
        probs = self._run_model_batch([text])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._build_decision(
            text=text,
            prob=float(probs[0]),
            threshold=self.threshold,
            tier="onnx_int8",
            reason="onnx_int8_calibrated_noul",
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
                "results": [],
            }

        results: List[Optional[NoulDecision]] = [None] * n
        model_indices: List[int] = []
        model_texts: List[str] = []

        if mode == "heuristic_only":
            for idx, txt in enumerate(texts):
                ti = time.perf_counter()
                h = evaluate_heuristic(txt)
                results[idx] = self._build_decision(
                    text=txt,
                    prob=h.probability,
                    threshold=0.5,
                    tier="heuristic_only",
                    reason=h.reason,
                    latency_ms=(time.perf_counter() - ti) * 1000.0,
                )
        elif mode == "hybrid":
            for idx, txt in enumerate(texts):
                ti = time.perf_counter()
                h = evaluate_heuristic(txt)
                if h.route == "clear_reject":
                    results[idx] = self._build_decision(
                        text=txt,
                        prob=h.probability,
                        threshold=0.5,
                        tier="heuristic_reject",
                        reason=h.reason,
                        latency_ms=(time.perf_counter() - ti) * 1000.0,
                    )
                elif h.route == "clear_address":
                    results[idx] = self._build_decision(
                        text=txt,
                        prob=h.probability,
                        threshold=0.5,
                        tier="heuristic_accept",
                        reason=h.reason,
                        latency_ms=(time.perf_counter() - ti) * 1000.0,
                    )
                else:
                    model_indices.append(idx)
                    model_texts.append(txt)
        else:
            model_indices = list(range(n))
            model_texts = list(texts)

        # Vectorized chunked ONNX INT8 inference for all routed items
        if model_texts:
            for start in range(0, len(model_texts), chunk_size):
                sub_texts = model_texts[start : start + chunk_size]
                sub_indices = model_indices[start : start + chunk_size]
                tc = time.perf_counter()
                probs = self._run_model_batch(sub_texts)
                per_item_ms = ((time.perf_counter() - tc) * 1000.0) / max(1, len(sub_texts))
                for idx, txt, prob in zip(sub_indices, sub_texts, probs):
                    results[idx] = self._build_decision(
                        text=txt,
                        prob=float(prob),
                        threshold=self.threshold,
                        tier="onnx_int8",
                        reason="onnx_int8_calibrated_noul",
                        latency_ms=per_item_ms,
                    )

        total_ms = (time.perf_counter() - t0) * 1000.0
        final_results = [r for r in results if r is not None]
        tier_counts: Dict[str, int] = {}
        for r in final_results:
            tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1

        return {
            "mode": mode,
            "count": n,
            "total_latency_ms": round(total_ms, 3),
            "avg_latency_ms": round(total_ms / n, 4),
            "throughput_req_per_s": round((n / max(total_ms / 1000.0, 1e-9)), 1),
            "tier_counts": tier_counts,
            "results": [r.to_dict() for r in final_results],
        }

    def get_info(self) -> Dict[str, Any]:
        """Return architecture, quantization size comparison, calibration, and CPU benchmark stats."""
        onnx_cmp = self.metadata.get("onnx_comparison", {})
        cal = self.metadata.get("calibration", {})
        return {
            "service": "Laya for Address VN — Way B CPU Backend Microservice",
            "model_name": self.metadata.get("model_name", "laya-address-vn-way-b-int8"),
            "version": self.metadata.get("version", "1.0.0"),
            "runtime": {
                "engine": "onnxruntime",
                "version": ort.__version__,
                "provider": "CPUExecutionProvider",
                "active_model": self.model_path.name,
                "intra_op_threads": self.intra_op_threads,
            },
            "architecture": self.metadata.get("architecture", {}),
            "total_parameters": self.metadata.get("total_parameters", 0),
            "dataset": self.metadata.get("dataset", {}),
            "calibration": {
                "temperature": self.temperature,
                "threshold": self.threshold,
                "val_pre_calibration": cal.get("val_pre_calibration", {}),
                "val_post_calibration": cal.get("val_post_calibration", {}),
                "test_post_calibration": cal.get("test_post_calibration", {}),
            },
            "quantization": {
                "fp32_size_mb": onnx_cmp.get("fp32_size_mb"),
                "int8_size_mb": onnx_cmp.get("int8_size_mb"),
                "fp32_size_bytes": onnx_cmp.get("fp32_size_bytes"),
                "int8_size_bytes": onnx_cmp.get("int8_size_bytes"),
                "size_reduction_percent": onnx_cmp.get("size_reduction_percent"),
                "fp32_test_metrics": onnx_cmp.get("fp32_test_metrics", {}),
                "int8_test_metrics": onnx_cmp.get("int8_test_metrics", {}),
                "fp32_cpu_benchmark": onnx_cmp.get("fp32_cpu_benchmark", {}),
                "int8_cpu_benchmark": onnx_cmp.get("int8_cpu_benchmark", {}),
            },
        }
