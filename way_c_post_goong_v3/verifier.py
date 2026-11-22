"""Pure NumPy + ONNX Runtime Post-Goong V3 fill verifier (zero PyTorch, zero pickle)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import onnxruntime as ort

try:
    from .features import DEFAULT_ARTIFACTS_DIR, FEATURE_NAMES, FeatureBundle, GazetteerV3
except ImportError:  # pragma: no cover - direct script execution fallback
    from features import DEFAULT_ARTIFACTS_DIR, FEATURE_NAMES, FeatureBundle, GazetteerV3


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _sigmoid_array(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


class PostGoongVerifier:
    """Calibrated ONNX Runtime verifier for Goong V1/V2 address fills."""

    def __init__(self, artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        onnx_path = self.artifacts_dir / "post_goong_verifier_v3.onnx"
        calib_path = self.artifacts_dir / "calibration.json"
        scaler_path = self.artifacts_dir / "scaler.json"
        conv_path = self.artifacts_dir / "vn_admin_v3_converter.json"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found at {calib_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

        with open(calib_path, "r", encoding="utf-8") as f:
            calib = json.load(f)
        with open(scaler_path, "r", encoding="utf-8") as f:
            scaler = json.load(f)

        calib_features = tuple(calib.get("feature_names", FEATURE_NAMES))
        scaler_features = tuple(scaler.get("feature_names", FEATURE_NAMES))
        if calib_features != FEATURE_NAMES or scaler_features != FEATURE_NAMES:
            raise ValueError("Feature schema mismatch between runtime code and artifacts.")

        self.temperature: float = float(calib["temperature"])
        self.t_low: float = float(calib["t_low"])
        self.t_high: float = float(calib["t_high"])

        self.mean: np.ndarray = np.asarray(scaler["mean"], dtype=np.float32)
        self.inv_std: np.ndarray = (1.0 / np.maximum(np.asarray(scaler["std"], dtype=np.float32), 1e-6)).astype(
            np.float32
        )

        self.gazetteer = GazetteerV3(converter_path=conv_path if conv_path.exists() else None)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name: str = self.session.get_inputs()[0].name
        self._output_name: str = self.session.get_outputs()[0].name

        # Warm up ONNX Runtime execution graph once at load time
        dummy = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)
        self.session.run([self._output_name], {self._input_name: dummy})

    def _build_decision(
        self,
        bundle: FeatureBundle,
        risk_score: float,
        raw_text: str,
        goong_ward: str,
        goong_district: str,
        goong_province: str,
        latency_ms: float,
    ) -> dict[str, Any]:
        if risk_score < self.t_low:
            action = "auto_approve"
        elif risk_score >= self.t_high:
            action = "hard_block"
        else:
            action = "soft_confirm_zns"

        is_valid_fill = action == "auto_approve"
        f = bundle.features

        if is_valid_fill:
            sufficiency_hint = "SUFFICIENT"
            reason_code = "VALID_LEGACY_2025_EQUIV" if bundle.is_legacy_2025_equiv else "VALID_MATCH"
            zns_confirm_message = None
        else:
            if not bundle.goong_has_ward:
                reason_code = "EMPTY_GOONG_WARD"
                sufficiency_hint = "MISSING_WARD_RESOLUTION"
            elif bundle.has_chat_food_only:
                reason_code = "CHAT_OR_FOOD_NOISE"
                sufficiency_hint = "NON_ADDRESS_INPUT"
            elif bundle.has_prov_contradiction:
                reason_code = "PROVINCE_CONTRADICTION"
                sufficiency_hint = "CONTRADICTORY_PROVINCE"
            elif bundle.has_dist_contradiction:
                reason_code = "DISTRICT_CONTRADICTION"
                sufficiency_hint = "CONTRADICTORY_DISTRICT"
            elif bundle.spatial_outlier:
                reason_code = "SPATIAL_OUTLIER"
                sufficiency_hint = "SPATIAL_COORDINATE_MISMATCH"
            elif f["ambiguous_ward_ungrounded_parent"] >= 0.5:
                reason_code = "AMBIGUOUS_WARD_UNGROUNDED_PARENT"
                sufficiency_hint = "NEED_DISTRICT_OR_PROVINCE"
            elif f["hallucinated_levels_count"] >= 2.0:
                reason_code = "HALLUCINATED_LEVELS"
                sufficiency_hint = "INSUFFICIENT_ADMIN_CONTEXT"
            elif f["ward_hallucinated"] >= 0.5:
                reason_code = "WARD_UNGROUNDED"
                sufficiency_hint = "CHECK_WARD_NAME"
            elif f["parent_hallucinated"] >= 0.5:
                reason_code = "PARENT_UNGROUNDED"
                sufficiency_hint = "NEED_DISTRICT_OR_PROVINCE"
            elif action == "soft_confirm_zns":
                reason_code = "LOW_CONFIDENCE_SOFT_CONFIRM"
                sufficiency_hint = "CONFIRM_ADDRESS_WITH_USER"
            else:
                reason_code = "HIGH_RISK_FILL"
                sufficiency_hint = "INSUFFICIENT_OR_MISMATCHED_ADDRESS"

            if action == "soft_confirm_zns":
                parts = [p.strip() for p in (goong_ward, goong_district, goong_province) if p and p.strip()]
                resolved_label = ", ".join(parts) if parts else raw_text.strip()
                zns_confirm_message = (
                    f"Quý khách vui lòng xác nhận địa chỉ nhận hàng: {resolved_label}. "
                    "Nếu chưa đúng, vui lòng bổ sung rõ phường/xã, quận/huyện và tỉnh/thành phố."
                )
            else:
                zns_confirm_message = None

        return {
            "action": action,
            "is_valid_fill": bool(is_valid_fill),
            "risk_score": round(float(risk_score), 6),
            "sufficiency_hint": sufficiency_hint,
            "reason_code": reason_code,
            "is_legacy_2025_equiv": bool(bundle.is_legacy_2025_equiv),
            "spatial_dist_km": round(float(bundle.spatial_dist_km), 3),
            "zns_confirm_message": zns_confirm_message,
            "latency_ms": round(float(latency_ms), 4),
        }

    def verify(
        self,
        raw_text: str,
        goong_ward: str = "",
        goong_district: str = "",
        goong_province: str = "",
        goong_score: float | None = None,
        goong_lat: float | None = None,
        goong_lng: float | None = None,
    ) -> dict[str, Any]:
        """Verify a single Goong address fill and return action, risk score, and telemetry."""
        t0 = time.perf_counter()
        bundle = self.gazetteer.extract(
            raw_text=raw_text,
            goong_ward=goong_ward,
            goong_district=goong_district,
            goong_province=goong_province,
            goong_score=goong_score,
            goong_lat=goong_lat,
            goong_lng=goong_lng,
        )
        x_scaled = ((bundle.vector - self.mean) * self.inv_std).reshape(1, -1)
        logit = float(self.session.run([self._output_name], {self._input_name: x_scaled})[0][0, 0])
        risk_score = _sigmoid(logit / self.temperature)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return self._build_decision(
            bundle=bundle,
            risk_score=risk_score,
            raw_text=raw_text,
            goong_ward=goong_ward,
            goong_district=goong_district,
            goong_province=goong_province,
            latency_ms=latency_ms,
        )

    def verify_batch(self, items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Vectorized batch verification over a sequence of Goong fill payloads."""
        if not items:
            return []
        t0 = time.perf_counter()
        n = len(items)
        bundles: list[FeatureBundle] = []
        mat = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)

        for i, item in enumerate(items):
            b = self.gazetteer.extract(
                raw_text=str(item.get("raw_text", "")),
                goong_ward=str(item.get("goong_ward", "") or ""),
                goong_district=str(item.get("goong_district", "") or ""),
                goong_province=str(item.get("goong_province", "") or ""),
                goong_score=item.get("goong_score"),
                goong_lat=item.get("goong_lat"),
                goong_lng=item.get("goong_lng"),
            )
            bundles.append(b)
            mat[i] = b.vector

        x_scaled = (mat - self.mean) * self.inv_std
        logits = self.session.run([self._output_name], {self._input_name: x_scaled})[0].reshape(-1)
        probs = _sigmoid_array(logits / self.temperature)
        total_ms = (time.perf_counter() - t0) * 1000.0
        per_item_ms = total_ms / float(n)

        results: list[dict[str, Any]] = []
        for i, item in enumerate(items):
            results.append(
                self._build_decision(
                    bundle=bundles[i],
                    risk_score=float(probs[i]),
                    raw_text=str(item.get("raw_text", "")),
                    goong_ward=str(item.get("goong_ward", "") or ""),
                    goong_district=str(item.get("goong_district", "") or ""),
                    goong_province=str(item.get("goong_province", "") or ""),
                    latency_ms=per_item_ms,
                )
            )
        return results
