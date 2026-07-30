"""Deployable B3 quantization engine with correct ablations and integer requant."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from b3_hw_golden import B3Shapes, B3Weights, forward_b3_layered

INT8_MIN, INT8_MAX = -128, 127
INT16_MIN, INT16_MAX = -32768, 32767


@dataclass
class QuantAblation:
    weight_bits: int = 8          # 8 | 16 | 32
    activation_bits: int = 8      # 8 | 16 | 32
    E_bits: int = 8               # 8 | 16 | 32
    integer_requant: bool = True
    name: str = "w8_a8_e8"


@dataclass
class IntegerRequant:
    mult: np.ndarray
    shift: int

    def apply(self, acc: np.ndarray, *, apply_relu: bool, qmin: int, qmax: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        mult = self.mult.astype(np.int64)
        while mult.ndim < acc.ndim:
            mult = mult.reshape(*([1] * (acc.ndim - mult.ndim)), *mult.shape)
        prod = acc.astype(np.int64) * mult
        if self.shift > 0:
            rounded = (prod + (1 << (self.shift - 1))) >> self.shift
        else:
            rounded = prod
        if apply_relu:
            rounded = np.maximum(rounded, 0)
        q = np.clip(rounded, qmin, qmax)
        sat = int(np.sum(rounded != q))
        return q.astype(np.int16 if qmax > 127 else np.int8), {
            "pre_clip_min": int(rounded.min()) if rounded.size else 0,
            "pre_clip_max": int(rounded.max()) if rounded.size else 0,
            "saturation_count": sat,
            "saturation_rate": float(sat / rounded.size) if rounded.size else 0.0,
        }


@dataclass
class FrozenActivationScales:
    input_scale_i8: float
    input_scale_i16: float
    pre_out_scale_i8: float
    pre_out_scale_i16: float
    mean_scale_i8: float
    mean_scale_i16: float
    post0_out_scale_i8: float
    post0_out_scale_i16: float
    post1_out_scale_i8: float
    post1_out_scale_i16: float
    E_int8_scale: float
    E_int16_scale: float
    source_sessions: List[str] = field(default_factory=list)

    def input_scale(self, bits: int) -> float:
        return self.input_scale_i16 if bits == 16 else self.input_scale_i8

    def pre_out_scale(self, bits: int) -> float:
        return self.pre_out_scale_i16 if bits == 16 else self.pre_out_scale_i8

    def mean_scale(self, bits: int) -> float:
        return self.mean_scale_i16 if bits == 16 else self.mean_scale_i8

    def post0_out_scale(self, bits: int) -> float:
        return self.post0_out_scale_i16 if bits == 16 else self.post0_out_scale_i8

    def post1_out_scale(self, bits: int) -> float:
        return self.post1_out_scale_i16 if bits == 16 else self.post1_out_scale_i8


@dataclass
class QuantLayerSpec:
    name: str
    w_fp: np.ndarray
    b_fp: np.ndarray
    w_q: Optional[np.ndarray]
    w_scale: np.ndarray
    bias_i32: np.ndarray
    in_scale: float
    out_scale: float
    requant: IntegerRequant


@dataclass
class QuantEngineBundle:
    shapes: B3Shapes
    scales: FrozenActivationScales
    layers: List[QuantLayerSpec]
    reciprocal: int
    reciprocal_shift: int
    ablation: QuantAblation


def _safe_scale(max_abs: float, qmax: float) -> float:
    return max(float(max_abs) / qmax, 1e-8)


def _q_bounds(bits: int, relu: bool) -> Tuple[int, int]:
    if bits == 16:
        return (0, INT16_MAX) if relu else (INT16_MIN, INT16_MAX)
    if bits == 8:
        return (0, INT8_MAX) if relu else (INT8_MIN, INT8_MAX)
    raise ValueError(bits)


def quantize_tensor(x_fp: np.ndarray, bits: int, scale: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    if bits == 32:
        return x_fp.astype(np.float32), {"clip_count": 0}
    qmin, qmax = _q_bounds(bits, relu=False)
    pre = np.round(x_fp.astype(np.float64) / scale)
    clip = int(np.sum((pre < qmin) | (pre > qmax)))
    q = np.clip(pre, qmin, qmax)
    dtype = np.int16 if bits == 16 else np.int8
    return q.astype(dtype), {"clip_count": clip, "q_min": int(q.min()), "q_max": int(q.max())}


def quantize_weight_per_channel(w_fp: np.ndarray, bits: int) -> Tuple[np.ndarray, np.ndarray]:
    qmax = 127.0 if bits == 8 else 32767.0
    qmin = -128 if bits == 8 else -32768
    out_f, _ = w_fp.shape
    dtype = np.int8 if bits == 8 else np.int16
    w_q = np.zeros_like(w_fp, dtype=dtype)
    scales = np.zeros((out_f,), dtype=np.float64)
    for o in range(out_f):
        row = w_fp[o].astype(np.float64)
        scale = _safe_scale(np.max(np.abs(row)), qmax)
        scales[o] = scale
        w_q[o] = np.clip(np.round(row / scale), qmin, qmax).astype(dtype)
    return w_q, scales


def float_ratio_to_requant(eff_scale: np.ndarray, out_scale: float, shift: int = 31) -> IntegerRequant:
    ratio = eff_scale.astype(np.float64) / max(out_scale, 1e-12)
    mult = np.round(ratio * (1 << shift)).astype(np.int64)
    return IntegerRequant(mult=mult, shift=shift)


def calibrate_frozen_scales(
    weights: B3Weights,
    calib_sessions: List[np.ndarray],
    session_names: List[str],
    side_feature_sessions: Optional[List[np.ndarray]] = None,
) -> FrozenActivationScales:
    if side_feature_sessions is not None and len(side_feature_sessions) != len(calib_sessions):
        raise ValueError("side_feature_sessions must align one-to-one with calib_sessions")
    max_abs = {k: 0.0 for k in ["input", "pre_relu", "mean", "post0_relu", "post1_relu", "E"]}
    for index, calib in enumerate(calib_sessions):
        side = None if side_feature_sessions is None else side_feature_sessions[index]
        st = forward_b3_layered(calib, weights, side_features=side)
        max_abs["input"] = max(max_abs["input"], float(np.max(np.abs(calib))))
        max_abs["pre_relu"] = max(max_abs["pre_relu"], float(np.max(np.maximum(st["feat"], 0))))
        max_abs["mean"] = max(max_abs["mean"], float(np.max(np.maximum(st["mean_feat"], 0))))
        max_abs["post0_relu"] = max(max_abs["post0_relu"], float(np.max(np.maximum(st["post0_relu"], 0))))
        max_abs["post1_relu"] = max(max_abs["post1_relu"], float(np.max(np.maximum(st["post1_relu"], 0))))
        max_abs["E"] = max(max_abs["E"], float(np.max(np.abs(st["E"]))))
    return FrozenActivationScales(
        input_scale_i8=_safe_scale(max_abs["input"], 127.0),
        input_scale_i16=_safe_scale(max_abs["input"], 32767.0),
        pre_out_scale_i8=_safe_scale(max_abs["pre_relu"], 127.0),
        pre_out_scale_i16=_safe_scale(max_abs["pre_relu"], 32767.0),
        mean_scale_i8=_safe_scale(max_abs["mean"], 127.0),
        mean_scale_i16=_safe_scale(max_abs["mean"], 32767.0),
        post0_out_scale_i8=_safe_scale(max_abs["post0_relu"], 127.0),
        post0_out_scale_i16=_safe_scale(max_abs["post0_relu"], 32767.0),
        post1_out_scale_i8=_safe_scale(max_abs["post1_relu"], 127.0),
        post1_out_scale_i16=_safe_scale(max_abs["post1_relu"], 32767.0),
        E_int8_scale=_safe_scale(max_abs["E"], 127.0),
        E_int16_scale=_safe_scale(max_abs["E"], 32767.0),
        source_sessions=list(session_names),
    )


def _pack_layer(name: str, w_fp: np.ndarray, b_fp: np.ndarray, in_scale: float, out_scale: float, ablation: QuantAblation) -> QuantLayerSpec:
    wb = ablation.weight_bits
    if wb in (8, 16):
        w_q, w_scale = quantize_weight_per_channel(w_fp, wb)
        eff = in_scale * w_scale
        bias_i32 = np.round(b_fp.astype(np.float64) / np.maximum(eff, 1e-12)).astype(np.int32)
        requant = float_ratio_to_requant(eff, out_scale) if ablation.integer_requant else IntegerRequant(np.ones(w_fp.shape[0], dtype=np.int64), 0)
    else:
        w_q = None
        w_scale = np.ones((w_fp.shape[0],), dtype=np.float64)
        bias_i32 = np.zeros_like(b_fp, dtype=np.int32)
        requant = IntegerRequant(np.ones(w_fp.shape[0], dtype=np.int64), 0)
    return QuantLayerSpec(name, w_fp.astype(np.float32), b_fp.astype(np.float32), w_q, w_scale.astype(np.float32), bias_i32, float(in_scale), float(out_scale), requant)


def build_quant_engine_bundle(weights: B3Weights, shapes: B3Shapes, scales: FrozenActivationScales, ablation: QuantAblation, recip_shift: int = 20) -> QuantEngineBundle:
    abits = ablation.activation_bits
    recip = int(round((1 << recip_shift) / shapes.M))
    layers = [
        _pack_layer("pre_pool", weights.pre_w, weights.pre_b, scales.input_scale(abits), scales.pre_out_scale(abits), ablation),
        _pack_layer("post0", weights.post0_w, weights.post0_b, scales.mean_scale(abits), scales.post0_out_scale(abits), ablation),
        _pack_layer("post1", weights.post1_w, weights.post1_b, scales.post0_out_scale(abits), scales.post1_out_scale(abits), ablation),
        _pack_layer("post2", weights.post2_w, weights.post2_b, scales.post1_out_scale(abits), scales.E_int8_scale if ablation.E_bits == 8 else scales.E_int16_scale, ablation),
    ]
    return QuantEngineBundle(shapes, scales, layers, recip, recip_shift, ablation)


def _linear_mixed(x: np.ndarray, layer: QuantLayerSpec, ablation: QuantAblation) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    wb, ab = ablation.weight_bits, ablation.activation_bits
    diag: Dict[str, Any] = {}

    # W8A32 / W16A32: FP32 activation × dequant INT weight, FP32 accumulator
    if wb in (8, 16) and ab == 32:
        w_deq = layer.w_q.astype(np.float64) * layer.w_scale.reshape(-1, 1)
        x_fp = x.astype(np.float64)
        acc_fp = x_fp @ w_deq.T + layer.b_fp.astype(np.float64)
        acc64 = acc_fp.astype(np.int64)
        return acc_fp, acc64, {"acc_i32_overflow": 0, "path": "w8_a32"}

    # W32A8/A16: quantized activation dequant × FP32 weight
    if wb == 32 and ab in (8, 16):
        x_fp = x.astype(np.float64) * layer.in_scale
        acc_fp = x_fp @ layer.w_fp.T + layer.b_fp.astype(np.float64)
        acc64 = acc_fp.astype(np.int64)
        return acc_fp, acc64, {"acc_i32_overflow": 0, "path": "w32_ax"}

    # W8/W16 A8/A16 integer MAC
    if wb in (8, 16) and ab in (8, 16) and layer.w_q is not None:
        x32 = x.astype(np.int32)
        w32 = layer.w_q.astype(np.int32)
        acc64 = x32.astype(np.int64) @ w32.T.astype(np.int64) + layer.bias_i32.astype(np.int64)
        overflow = int(np.sum((acc64 < np.iinfo(np.int32).min) | (acc64 > np.iinfo(np.int32).max)))
        diag.update({"acc_i64_min": int(acc64.min()), "acc_i64_max": int(acc64.max()), "acc_i32_overflow": overflow, "path": "w8_a8"})
        return acc64.astype(np.int32), acc64, diag

    # FP32
    x_fp = x.astype(np.float64)
    acc_fp = x_fp @ layer.w_fp.T + layer.b_fp.astype(np.float64)
    return acc_fp, acc_fp.astype(np.int64), {"acc_i32_overflow": 0, "path": "fp32"}


def _requant(x_acc: np.ndarray, layer: QuantLayerSpec, ablation: QuantAblation, relu: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
    ab = ablation.activation_bits
    if ab == 32:
        y_fp = x_acc.astype(np.float64)
        if relu:
            y_fp = np.maximum(y_fp, 0.0)
        return y_fp.astype(np.float32), {"saturation_count": 0, "path": "fp32_act"}

    qmin, qmax = _q_bounds(ab, relu)
    if ablation.weight_bits in (8, 16) and ab in (8, 16) and ablation.integer_requant:
        return layer.requant.apply(x_acc, apply_relu=relu, qmin=qmin, qmax=qmax)

    # float-scale requant reference
    if ablation.weight_bits in (8, 16):
        eff = layer.in_scale * layer.w_scale
    else:
        eff = np.ones(layer.w_fp.shape[0])
    shape = [1] * (x_acc.ndim - 1) + [eff.shape[0]]
    y_fp = x_acc.astype(np.float64) * eff.reshape(shape)
    if relu:
        y_fp = np.maximum(y_fp, 0.0)
    pre = np.round(y_fp / layer.out_scale)
    q = np.clip(pre, qmin, qmax)
    sat = int(np.sum(pre != q))
    dtype = np.int16 if ab == 16 else np.int8
    return q.astype(dtype), {"saturation_count": sat, "saturation_rate": float(sat / pre.size) if pre.size else 0.0, "path": "float_requant"}


def mean_int32_via_reciprocal(sum_i32: np.ndarray, recip: int, shift: int) -> np.ndarray:
    prod = sum_i32.astype(np.int64) * int(recip)
    if shift > 0:
        return ((prod + (1 << (shift - 1))) >> shift).astype(np.int32)
    return prod.astype(np.int32)


def forward_quant_engine(
    calib_fp: np.ndarray,
    bundle: QuantEngineBundle,
    side_features: np.ndarray | None = None,
) -> Dict[str, Any]:
    ab = bundle.ablation
    abits = ab.activation_bits
    M, T, N = calib_fp.shape
    D = bundle.shapes.D

    if ab.activation_bits == 32:
        calib_act = calib_fp.astype(np.float32)
        diag = {"input": {"clip_count": 0}, "pre_pool": {}, "layers": {}}
    else:
        calib_q, in_diag = quantize_tensor(calib_fp, abits, bundle.scales.input_scale(abits))
        calib_act = calib_q
        diag = {"input": in_diag, "pre_pool": {}, "layers": {}}

    pre = bundle.layers[0]
    feat = np.zeros((M, N, D), dtype=np.float32 if abits == 32 else (np.int16 if abits == 16 else np.int8))
    pre_acc = np.zeros((M, N, D), dtype=np.int32)

    for m in range(M):
        for n in range(N):
            x = calib_fp[m, :, n] if ab.activation_bits == 32 else calib_act[m, :, n]
            acc32, acc64, ld = _linear_mixed(x, pre, ab)
            q, rq = _requant(acc32, pre, ab, relu=True)
            pre_acc[m, n] = acc32
            feat[m, n] = q
            diag["pre_pool"].setdefault("acc_i64_max", acc64.max())
            diag["pre_pool"]["acc_i64_max"] = max(diag["pre_pool"].get("acc_i64_max", 0), int(acc64.max()))
            diag["pre_pool"]["acc_i32_overflow"] = diag["pre_pool"].get("acc_i32_overflow", 0) + ld.get("acc_i32_overflow", 0)

    if ab.activation_bits in (8, 16):
        sum_feat = feat.astype(np.int32).sum(axis=0)
        mean_i32 = mean_int32_via_reciprocal(sum_feat, bundle.reciprocal, bundle.reciprocal_shift)
        mean_fp_est = np.maximum(mean_i32.astype(np.float64) * bundle.scales.pre_out_scale(abits), 0.0)
        mean_act, mean_diag = quantize_tensor(mean_fp_est, abits, bundle.scales.mean_scale(abits))
        diag["mean"] = mean_diag
        diag["mean"]["mean_i32_min"] = int(mean_i32.min())
        diag["mean"]["mean_i32_max"] = int(mean_i32.max())
        diag["mean"]["sum_feat_i32_min"] = int(np.min(sum_feat))
        diag["mean"]["sum_feat_i32_max"] = int(np.max(sum_feat))
    else:
        sum_feat = feat.sum(axis=0)
        mean_act = np.maximum(sum_feat.astype(np.float64) / float(M), 0.0).astype(np.float32)
        mean_i32 = mean_act.astype(np.int32)
        diag["mean"] = {"path": "fp32_mean"}

    required_side_dim = int(bundle.layers[1].w_fp.shape[1]) - D
    if required_side_dim < 0:
        raise ValueError(
            f"post0 in_features {bundle.layers[1].w_fp.shape[1]} is smaller than D={D}"
        )
    if required_side_dim:
        if side_features is None:
            raise ValueError(
                f"post0 requires side_dim={required_side_dim}, but side_features is missing"
            )
        side_features = np.asarray(side_features, dtype=np.float32)
        if side_features.shape != (N, required_side_dim):
            raise ValueError(
                f"side_features must be {(N, required_side_dim)}, got {side_features.shape}"
            )
        if ab.activation_bits in (8, 16):
            side_act, side_diag = quantize_tensor(
                side_features, abits, bundle.scales.mean_scale(abits)
            )
        else:
            side_act = side_features
            side_diag = {"clip_count": 0, "path": "fp32_side"}
        # Hardware contract: pooled activity and T4 use one shared post0 input
        # scale, then concatenate in the integer domain.  There is no FP side
        # bypass around the real 68->64 T4 layer.
        mean_act = np.concatenate([mean_act, side_act], axis=-1)
        diag["side"] = {
            **side_diag,
            "side_dim": required_side_dim,
            "shared_post0_input_scale": float(bundle.scales.mean_scale(abits)),
        }
    elif side_features is not None and np.asarray(side_features).shape[-1] != 0:
        raise ValueError("Plain B3 bundle does not accept non-empty side_features")

    def run_layer(x_in, layer, relu, key):
        acc32, acc64, ld = _linear_mixed(x_in, layer, ab)
        q, rq = _requant(acc32, layer, ab, relu=relu)
        diag["layers"][key] = {**ld, **rq, "acc_i64_min": int(acc64.min()), "acc_i64_max": int(acc64.max())}
        return q, acc32

    post0_q, _ = run_layer(mean_act, bundle.layers[1], True, "post0")
    post1_q, post2_acc = run_layer(post0_q, bundle.layers[2], True, "post1")
    post2_q, post2_acc2 = run_layer(post1_q, bundle.layers[3], False, "post2")

    if ab.E_bits == 32:
        if ab.activation_bits == 32:
            E_fp = post2_acc2.astype(np.float64)
        elif ab.weight_bits in (8, 16) and ab.activation_bits in (8, 16):
            eff = bundle.layers[3].in_scale * bundle.layers[3].w_scale
            E_fp = post2_acc2.astype(np.float64) * eff
        else:
            E_fp = post2_acc2.astype(np.float64)
        E_out = E_fp.astype(np.float32)
        E_dequant = E_out
    else:
        if ab.weight_bits in (8, 16) and ab.activation_bits in (8, 16) and ab.integer_requant:
            q, ed = bundle.layers[3].requant.apply(post2_acc2, apply_relu=False, qmin=_q_bounds(ab.E_bits, False)[0], qmax=_q_bounds(ab.E_bits, False)[1])
            diag["layers"]["E"] = ed
            E_out = q
        else:
            eff = bundle.layers[3].in_scale * bundle.layers[3].w_scale
            shape = [1] * (post2_acc2.ndim - 1) + [eff.shape[0]]
            E_fp_est = post2_acc2.astype(np.float64) * eff.reshape(shape)
            E_out, _ = quantize_tensor(E_fp_est, ab.E_bits, bundle.scales.E_int16_scale if ab.E_bits == 16 else bundle.scales.E_int8_scale)
        scale = bundle.scales.E_int16_scale if ab.E_bits == 16 else bundle.scales.E_int8_scale
        E_dequant = E_out.astype(np.float64) * scale

    return {
        "E": E_out,
        "E_dequant": E_dequant.astype(np.float32),
        "diagnostics": diag,
        "feat": feat,
        "sum_feat_i32": sum_feat,
        "mean_i32": mean_i32,
        "post0_input": mean_act,
    }


def identity_metrics(E_ref: np.ndarray, E_pred: np.ndarray) -> Dict[str, float]:
    diff = np.abs(E_ref.astype(np.float64) - E_pred.astype(np.float64))
    a = E_ref.astype(np.float64).ravel()
    b = E_pred.astype(np.float64).ravel()
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "cosine": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)),
    }


ABLATION_PRESETS: Dict[str, QuantAblation] = {
    "fp32": QuantAblation(32, 32, 32, False, "fp32"),
    "w8_a32": QuantAblation(8, 32, 32, False, "w8_a32"),
    "w32_a8": QuantAblation(32, 8, 32, False, "w32_a8"),
    "w8_a8_e8": QuantAblation(8, 8, 8, True, "w8_a8_e8"),
    "w8_a8_e16": QuantAblation(8, 8, 16, True, "w8_a8_e16"),
    "w8_a16_e16": QuantAblation(8, 16, 16, True, "w8_a16_e16"),
    "w16_a8_e8": QuantAblation(16, 8, 8, True, "w16_a8_e8"),
    "w16_a16_e16": QuantAblation(16, 16, 16, True, "w16_a16_e16"),
}
