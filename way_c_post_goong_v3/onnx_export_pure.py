"""Pure Python stdlib ONNX protobuf exporter for LR + HistGradientBoosting ensembles.

Zero dependencies on `torch`, `onnx`, `skl2onnx`, or `pickle`.
Produces valid ONNX ModelProto (IR v8, opset 14 + ai.onnx.ml v1) directly runnable
on CPU with `onnxruntime.InferenceSession`.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np


# Wire types in Protocol Buffers
_WIRE_VARINT = 0
_WIRE_LEN = 2

# ONNX TensorProto.DataType
_ONNX_FLOAT = 1

# ONNX AttributeProto.AttributeType
_ATTR_FLOAT = 1
_ATTR_INT = 2
_ATTR_STRING = 3
_ATTR_FLOATS = 6
_ATTR_INTS = 7
_ATTR_STRINGS = 8


def _varint(n: int) -> bytes:
    if n < 0:
        n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _tag(field_number: int, wire_type: int) -> bytes:
    return _varint((field_number << 3) | wire_type)


def _bytes_field(field_number: int, data: bytes) -> bytes:
    return _tag(field_number, _WIRE_LEN) + _varint(len(data)) + data


def _string_field(field_number: int, text: str) -> bytes:
    return _bytes_field(field_number, text.encode("utf-8"))


def _int_field(field_number: int, val: int) -> bytes:
    return _tag(field_number, _WIRE_VARINT) + _varint(int(val))


def _float_field(field_number: int, val: float) -> bytes:
    return _tag(field_number, 5) + struct.pack("<f", float(val))


def _packed_ints(field_number: int, vals: Sequence[int]) -> bytes:
    if not vals:
        return b""
    buf = b"".join(_varint(int(v)) for v in vals)
    return _bytes_field(field_number, buf)


def _packed_floats(field_number: int, vals: Sequence[float]) -> bytes:
    if not vals:
        return b""
    buf = struct.pack(f"<{len(vals)}f", *(float(v) for v in vals))
    return _bytes_field(field_number, buf)


def _tensor_f32(name: str, arr: np.ndarray) -> bytes:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    parts: List[bytes] = [
        _packed_ints(1, list(a.shape)),   # dims
        _int_field(2, _ONNX_FLOAT),       # data_type
        _string_field(8, name),           # name
        _bytes_field(9, a.tobytes()),     # raw_data (little-endian f32)
    ]
    return b"".join(parts)


def _attr_int(name: str, val: int) -> bytes:
    return (
        _string_field(1, name)
        + _int_field(3, val)
        + _int_field(20, _ATTR_INT)
    )


def _attr_string(name: str, val: str) -> bytes:
    return (
        _string_field(1, name)
        + _bytes_field(4, val.encode("utf-8"))
        + _int_field(20, _ATTR_STRING)
    )


def _attr_ints(name: str, vals: Sequence[int]) -> bytes:
    return (
        _string_field(1, name)
        + _packed_ints(8, vals)
        + _int_field(20, _ATTR_INTS)
    )


def _attr_floats(name: str, vals: Sequence[float]) -> bytes:
    return (
        _string_field(1, name)
        + _packed_floats(7, vals)
        + _int_field(20, _ATTR_FLOATS)
    )


def _attr_strings(name: str, vals: Sequence[str]) -> bytes:
    parts = [_string_field(1, name)]
    for s in vals:
        parts.append(_bytes_field(9, s.encode("utf-8")))
    parts.append(_int_field(20, _ATTR_STRINGS))
    return b"".join(parts)


def _node(
    op_type: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    name: str,
    domain: str = "",
    attrs: Iterable[bytes] = (),
) -> bytes:
    parts: List[bytes] = []
    for inp in inputs:
        parts.append(_string_field(1, inp))
    for out in outputs:
        parts.append(_string_field(2, out))
    if name:
        parts.append(_string_field(3, name))
    parts.append(_string_field(4, op_type))
    for attr in attrs:
        parts.append(_bytes_field(5, attr))
    if domain:
        parts.append(_string_field(7, domain))
    return b"".join(parts)


def _dim_value(val: int) -> bytes:
    return _int_field(1, val)


def _dim_param(param: str) -> bytes:
    return _string_field(2, param)


def _value_info_f32(name: str, dims: Sequence[int | str]) -> bytes:
    shape_parts: List[bytes] = []
    for d in dims:
        dim_msg = _dim_param(d) if isinstance(d, str) else _dim_value(int(d))
        shape_parts.append(_bytes_field(1, dim_msg))
    tensor_shape_msg = b"".join(shape_parts)
    tensor_type_msg = _int_field(1, _ONNX_FLOAT) + _bytes_field(2, tensor_shape_msg)
    type_proto_msg = _bytes_field(1, tensor_type_msg)
    return _string_field(1, name) + _bytes_field(2, type_proto_msg)


def _opset_import(domain: str, version: int) -> bytes:
    parts: List[bytes] = []
    if domain:
        parts.append(_string_field(1, domain))
    parts.append(_int_field(2, version))
    return b"".join(parts)


def _extract_hgb_tree_attrs(hgb_model: Any) -> List[bytes]:
    if not hasattr(hgb_model, "_predictors") or not hasattr(hgb_model, "_baseline_prediction"):
        raise TypeError("hgb_model does not expose fitted scikit-learn HistGradientBoosting attributes.")
    base_val = float(np.asarray(hgb_model._baseline_prediction).ravel()[0])

    nodes_treeids: List[int] = []
    nodes_nodeids: List[int] = []
    nodes_featureids: List[int] = []
    nodes_values: List[float] = []
    nodes_hitrates: List[float] = []
    nodes_truenodeids: List[int] = []
    nodes_falsenodeids: List[int] = []
    nodes_missing_tracks_true: List[int] = []
    nodes_modes: List[str] = []

    target_treeids: List[int] = []
    target_nodeids: List[int] = []
    target_ids: List[int] = []
    target_weights: List[float] = []

    for tree_id, pred_arr in enumerate(hgb_model._predictors):
        nodes = pred_arr[0].nodes
        for node_id, nd in enumerate(nodes):
            if int(nd["is_categorical"]):
                raise ValueError("Categorical HGB splits are not supported in ONNX TreeEnsembleRegressor export.")
            nodes_treeids.append(tree_id)
            nodes_nodeids.append(node_id)
            nodes_hitrates.append(1.0)
            if int(nd["is_leaf"]):
                nodes_modes.append("LEAF")
                nodes_featureids.append(0)
                nodes_values.append(0.0)
                nodes_truenodeids.append(0)
                nodes_falsenodeids.append(0)
                nodes_missing_tracks_true.append(0)

                target_treeids.append(tree_id)
                target_nodeids.append(node_id)
                target_ids.append(0)
                target_weights.append(float(nd["value"]))
            else:
                nodes_modes.append("BRANCH_LEQ")
                nodes_featureids.append(int(nd["feature_idx"]))
                nodes_values.append(float(nd["num_threshold"]))
                nodes_truenodeids.append(int(nd["left"]))
                nodes_falsenodeids.append(int(nd["right"]))
                nodes_missing_tracks_true.append(int(bool(nd["missing_go_to_left"])))

    return [
        _attr_string("aggregate_function", "SUM"),
        _attr_floats("base_values", [base_val]),
        _attr_int("n_targets", 1),
        _attr_ints("nodes_falsenodeids", nodes_falsenodeids),
        _attr_ints("nodes_featureids", nodes_featureids),
        _attr_floats("nodes_hitrates", nodes_hitrates),
        _attr_ints("nodes_missing_value_tracks_true", nodes_missing_tracks_true),
        _attr_strings("nodes_modes", nodes_modes),
        _attr_ints("nodes_nodeids", nodes_nodeids),
        _attr_ints("nodes_treeids", nodes_treeids),
        _attr_ints("nodes_truenodeids", nodes_truenodeids),
        _attr_floats("nodes_values", nodes_values),
        _attr_string("post_transform", "NONE"),
        _attr_ints("target_ids", target_ids),
        _attr_ints("target_nodeids", target_nodeids),
        _attr_ints("target_treeids", target_treeids),
        _attr_floats("target_weights", target_weights),
    ]


def export_lr_hgb_ensemble_to_onnx(
    out_path: str | Path,
    feature_names: Sequence[str],
    lr_mean: Sequence[float] | np.ndarray,
    lr_std: Sequence[float] | np.ndarray,
    lr_coef: Sequence[float] | np.ndarray,
    lr_intercept: float,
    hgb_model: Any | None = None,
    w_lr: float = 0.5,
    w_hgb: float = 0.5,
    doc_string: str = "Laya Address VN v3.1 Post-Goong 2A Local Verifier",
) -> Path:
    """Serialize `(w_lr * Sigmoid(LR(x)) + w_hgb * Sigmoid(HGB(x)))` directly to ONNX bytes."""
    n_feat = len(feature_names)
    if n_feat <= 0:
        raise ValueError("feature_names must not be empty.")

    use_lr = w_lr > 0.0
    use_hgb = hgb_model is not None and w_hgb > 0.0
    if not use_lr and not use_hgb:
        raise ValueError("At least one of LR or HGB branch must have positive weight.")

    weight_sum = (w_lr if use_lr else 0.0) + (w_hgb if use_hgb else 0.0)
    eff_w_lr = float(w_lr / weight_sum) if use_lr else 0.0
    eff_w_hgb = float(w_hgb / weight_sum) if use_hgb else 0.0

    initializers: List[bytes] = []
    nodes: List[bytes] = []

    if use_lr:
        mean_arr = np.asarray(lr_mean, dtype=np.float32).reshape(1, n_feat)
        std_arr = np.maximum(np.asarray(lr_std, dtype=np.float32).reshape(1, n_feat), 1e-6)
        inv_std_arr = (1.0 / std_arr).astype(np.float32)
        coef_arr = np.asarray(lr_coef, dtype=np.float32).reshape(1, n_feat)
        bias_arr = np.asarray([float(lr_intercept)], dtype=np.float32)

        initializers.append(_tensor_f32("lr_mean", mean_arr))
        initializers.append(_tensor_f32("lr_inv_std", inv_std_arr))
        initializers.append(_tensor_f32("lr_weight", coef_arr))
        initializers.append(_tensor_f32("lr_bias", bias_arr))

        nodes.append(_node("Sub", ["features", "lr_mean"], ["lr_centered"], name="lr_sub"))
        nodes.append(_node("Mul", ["lr_centered", "lr_inv_std"], ["lr_scaled"], name="lr_scale"))
        nodes.append(
            _node(
                "Gemm",
                ["lr_scaled", "lr_weight", "lr_bias"],
                ["lr_logit"],
                name="lr_gemm",
                attrs=[_attr_int("transB", 1)],
            )
        )
        lr_prob_name = "lr_prob" if use_hgb else "risk_prob"
        nodes.append(_node("Sigmoid", ["lr_logit"], [lr_prob_name], name="lr_sigmoid"))

    if use_hgb:
        tree_attrs = _extract_hgb_tree_attrs(hgb_model)
        nodes.append(
            _node(
                "TreeEnsembleRegressor",
                ["features"],
                ["hgb_logit"],
                name="hgb_trees",
                domain="ai.onnx.ml",
                attrs=tree_attrs,
            )
        )
        hgb_prob_name = "hgb_prob" if use_lr else "risk_prob"
        nodes.append(_node("Sigmoid", ["hgb_logit"], [hgb_prob_name], name="hgb_sigmoid"))

    if use_lr and use_hgb:
        initializers.append(_tensor_f32("w_lr_scalar", np.asarray([eff_w_lr], dtype=np.float32)))
        initializers.append(_tensor_f32("w_hgb_scalar", np.asarray([eff_w_hgb], dtype=np.float32)))
        nodes.append(_node("Mul", ["lr_prob", "w_lr_scalar"], ["lr_weighted"], name="mul_lr_w"))
        nodes.append(_node("Mul", ["hgb_prob", "w_hgb_scalar"], ["hgb_weighted"], name="mul_hgb_w"))
        nodes.append(_node("Add", ["lr_weighted", "hgb_weighted"], ["risk_prob"], name="ensemble_add"))

    graph_parts: List[bytes] = []
    for n in nodes:
        graph_parts.append(_bytes_field(1, n))
    graph_parts.append(_string_field(2, "laya_post_goong_2a_v31"))
    for init in initializers:
        graph_parts.append(_bytes_field(5, init))
    graph_parts.append(_bytes_field(11, _value_info_f32("features", ["batch", n_feat])))
    graph_parts.append(_bytes_field(12, _value_info_f32("risk_prob", ["batch", 1])))
    graph_bytes = b"".join(graph_parts)

    model_parts: List[bytes] = [
        _int_field(1, 8),  # ir_version = 8
        _string_field(2, "laya_address_vn_pure_onnx"),
        _string_field(3, "3.1.0"),
        _string_field(6, doc_string),
        _bytes_field(7, graph_bytes),
        _bytes_field(8, _opset_import("", 14)),
    ]
    if use_hgb:
        model_parts.append(_bytes_field(8, _opset_import("ai.onnx.ml", 1)))

    dst = Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"".join(model_parts))
    return dst
