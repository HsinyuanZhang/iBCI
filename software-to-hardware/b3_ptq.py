"""Bounded PTQ utilities: activation scale search and cross-layer equalization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from b3_hw_golden import B3Weights, forward_b3_layered
from b3_quant_engine import FrozenActivationScales, _safe_scale, identity_metrics

TENSOR_KEYS = ("input", "pre_relu", "mean", "post0_relu", "post1_relu", "E")


@dataclass
class ActivationStats:
    input_abs: np.ndarray
    pre_relu: np.ndarray
    mean: np.ndarray
    post0_relu: np.ndarray
    post1_relu: np.ndarray
    E_abs: np.ndarray

    def tensor(self, key: str) -> np.ndarray:
        if key == "input":
            return self.input_abs
        if key == "E":
            return self.E_abs
        return getattr(self, key)


def copy_weights(weights: B3Weights) -> B3Weights:
    return B3Weights(
        pre_w=weights.pre_w.copy(),
        pre_b=weights.pre_b.copy(),
        post0_w=weights.post0_w.copy(),
        post0_b=weights.post0_b.copy(),
        post1_w=weights.post1_w.copy(),
        post1_b=weights.post1_b.copy(),
        post2_w=weights.post2_w.copy(),
        post2_b=weights.post2_b.copy(),
    )


def collect_activation_stats(weights: B3Weights, calib_sessions: Sequence[np.ndarray]) -> ActivationStats:
    chunks = {k: [] for k in TENSOR_KEYS}
    for calib in calib_sessions:
        st = forward_b3_layered(calib, weights)
        chunks["input"].append(np.abs(calib).astype(np.float64).ravel())
        chunks["pre_relu"].append(np.maximum(st["feat"], 0).astype(np.float64).ravel())
        chunks["mean"].append(np.maximum(st["mean_feat"], 0).astype(np.float64).ravel())
        chunks["post0_relu"].append(np.maximum(st["post0_relu"], 0).astype(np.float64).ravel())
        chunks["post1_relu"].append(np.maximum(st["post1_relu"], 0).astype(np.float64).ravel())
        chunks["E"].append(np.abs(st["E"]).astype(np.float64).ravel())
    return ActivationStats(
        input_abs=np.concatenate(chunks["input"]),
        pre_relu=np.concatenate(chunks["pre_relu"]),
        mean=np.concatenate(chunks["mean"]),
        post0_relu=np.concatenate(chunks["post0_relu"]),
        post1_relu=np.concatenate(chunks["post1_relu"]),
        E_abs=np.concatenate(chunks["E"]),
    )


def _mse_optimal_scale(values: np.ndarray, qmax: float, *, n_grid: int = 64) -> float:
    """Search quantization scale s (real units per q-level), not raw activation magnitude."""
    vals = np.abs(values.astype(np.float64))
    if vals.size == 0:
        return 1e-8
    vmax = float(vals.max())
    p9999 = float(np.percentile(vals, 99.99))
    # Scale space: max_abs / qmax is the usual upper bound; search a band below/around it.
    hi = max(p9999 / qmax * 1.5, vmax / qmax, 1e-8)
    lo = max(vmax / qmax * 0.05, hi * 0.05, 1e-8)
    if lo >= hi:
        hi = lo * 2.0
    scales = np.linspace(lo, hi, n_grid)
    best_s, best_mse = scales[0], float("inf")
    for s in scales:
        q = np.clip(np.round(vals / s), 0.0, qmax)
        mse = float(np.mean((vals - q * s) ** 2))
        if mse < best_mse:
            best_mse, best_s = mse, float(s)
    return max(best_s, 1e-8)


def scale_from_values(
    values: np.ndarray,
    method: str,
    qmax: float,
    *,
    mult: float = 1.0,
) -> float:
    vals = np.abs(values.astype(np.float64))
    if vals.size == 0:
        return 1e-8
    if method == "max_abs":
        s = float(vals.max()) / qmax
    elif method == "p999":
        s = float(np.percentile(vals, 99.9)) / qmax
    elif method == "p9999":
        s = float(np.percentile(vals, 99.99)) / qmax
    elif method == "mse_opt":
        s = _mse_optimal_scale(vals, qmax)
    else:
        raise ValueError(f"Unknown scale method: {method}")
    return max(s * float(mult), 1e-8)


def calibrate_scales_from_stats(
    stats: ActivationStats,
    method: str,
    *,
    mult: float = 1.0,
    source_sessions: Sequence[str] | None = None,
) -> FrozenActivationScales:
    per_i8: Dict[str, float] = {}
    per_i16: Dict[str, float] = {}
    for key in TENSOR_KEYS:
        tensor = stats.tensor(key)
        per_i8[key] = scale_from_values(tensor, method, 127.0, mult=mult)
        per_i16[key] = scale_from_values(tensor, method, 32767.0, mult=mult)
    return FrozenActivationScales(
        input_scale_i8=per_i8["input"],
        input_scale_i16=per_i16["input"],
        pre_out_scale_i8=per_i8["pre_relu"],
        pre_out_scale_i16=per_i16["pre_relu"],
        mean_scale_i8=per_i8["mean"],
        mean_scale_i16=per_i16["mean"],
        post0_out_scale_i8=per_i8["post0_relu"],
        post0_out_scale_i16=per_i16["post0_relu"],
        post1_out_scale_i8=per_i8["post1_relu"],
        post1_out_scale_i16=per_i16["post1_relu"],
        E_int8_scale=per_i8["E"],
        E_int16_scale=per_i16["E"],
        source_sessions=list(source_sessions or []),
    )


def _channel_max_positive(tensor: np.ndarray) -> np.ndarray:
    return np.max(np.maximum(tensor.astype(np.float64), 0.0), axis=tuple(range(tensor.ndim - 1)))


def _equalization_vector(act_ch_max: np.ndarray, next_w_in_max: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    act = np.maximum(act_ch_max.astype(np.float64), eps)
    wgt = np.maximum(next_w_in_max.astype(np.float64), eps)
    s = np.sqrt(act / wgt)
    s = s / np.exp(np.mean(np.log(s)))
    return s.astype(np.float64)


def apply_cross_layer_equalization(weights: B3Weights, calib_sessions: Sequence[np.ndarray]) -> B3Weights:
    """Balance adjacent Linear–ReLU channel ranges without changing FP32 function."""
    w = copy_weights(weights)
    mean_stack, post0_stack, post1_stack = [], [], []
    for calib in calib_sessions:
        st = forward_b3_layered(calib, w)
        mean_stack.append(st["mean_feat"])
        post0_stack.append(st["post0_relu"])
        post1_stack.append(st["post1_relu"])
    mean_feat = np.max(np.stack(mean_stack, axis=0), axis=0)
    post0_relu = np.max(np.stack(post0_stack, axis=0), axis=0)
    post1_relu = np.max(np.stack(post1_stack, axis=0), axis=0)

    feat_ch = _channel_max_positive(
        np.max(np.stack([forward_b3_layered(c, w)["feat"] for c in calib_sessions], axis=0), axis=0)
    )

    # pre_pool[D,T] out ch i -> post0[D,D] in ch i
    s0 = _equalization_vector(feat_ch, np.max(np.abs(w.post0_w), axis=0))
    w.pre_w /= s0.reshape(-1, 1)
    w.pre_b /= s0
    w.post0_w *= s0.reshape(1, -1)

    # post0 out -> post1 in
    p0_ch = _channel_max_positive(post0_relu)
    s1 = _equalization_vector(p0_ch, np.max(np.abs(w.post1_w), axis=0))
    w.post0_w /= s1.reshape(-1, 1)
    w.post0_b /= s1
    w.post1_w *= s1.reshape(1, -1)

    # post1 out -> post2 in
    p1_ch = _channel_max_positive(post1_relu)
    s2 = _equalization_vector(p1_ch, np.max(np.abs(w.post2_w), axis=0))
    w.post1_w /= s2.reshape(-1, 1)
    w.post1_b /= s2
    w.post2_w *= s2.reshape(1, -1)

    return w


def mean_identity_rmse(calib_sessions: Sequence[np.ndarray], weights: B3Weights, scales: FrozenActivationScales, shapes, ablation) -> float:
    from b3_quant_engine import build_quant_engine_bundle, forward_quant_engine

    rmses = []
    for calib in calib_sessions:
        ref = forward_b3_layered(calib, weights)["E"]
        bundle = build_quant_engine_bundle(weights, shapes, scales, ablation)
        pred = forward_quant_engine(calib, bundle)["E_dequant"]
        rmses.append(identity_metrics(ref, pred)["rmse"])
    return float(np.mean(rmses))


def apply_equalization_to_early_pool(encoder, calib_sessions: Sequence[np.ndarray]) -> None:
    """In-place cross-layer equalization on EarlyPoolEncoder Linear weights."""
    import torch

    from b3_ckpt_loader import load_b3_weights_from_ckpt  # noqa: F401
    from b3_hw_golden import B3Weights

    def _to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32)

    w = B3Weights(
        pre_w=_to_np(encoder.pre_pool[0].weight),
        pre_b=_to_np(encoder.pre_pool[0].bias),
        post0_w=_to_np(encoder.post_pool[0].weight),
        post0_b=_to_np(encoder.post_pool[0].bias),
        post1_w=_to_np(encoder.post_pool[2].weight),
        post1_b=_to_np(encoder.post_pool[2].bias),
        post2_w=_to_np(encoder.post_pool[4].weight),
        post2_b=_to_np(encoder.post_pool[4].bias),
    )
    w_eq = apply_cross_layer_equalization(w, calib_sessions)
    with torch.no_grad():
        encoder.pre_pool[0].weight.copy_(torch.from_numpy(w_eq.pre_w))
        encoder.pre_pool[0].bias.copy_(torch.from_numpy(w_eq.pre_b))
        encoder.post_pool[0].weight.copy_(torch.from_numpy(w_eq.post0_w))
        encoder.post_pool[0].bias.copy_(torch.from_numpy(w_eq.post0_b))
        encoder.post_pool[2].weight.copy_(torch.from_numpy(w_eq.post1_w))
        encoder.post_pool[2].bias.copy_(torch.from_numpy(w_eq.post1_b))
        encoder.post_pool[4].weight.copy_(torch.from_numpy(w_eq.post2_w))
        encoder.post_pool[4].bias.copy_(torch.from_numpy(w_eq.post2_b))


def iter_scale_candidates(
    *,
    methods: Sequence[str] = ("max_abs", "p999", "p9999", "mse_opt"),
    mult_grid: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2),
) -> Iterable[Tuple[str, str, float]]:
    for method in methods:
        yield (f"{method}_x1.0", method, 1.0)
    for mult in mult_grid:
        if abs(mult - 1.0) < 1e-9:
            continue
        yield (f"max_abs_x{mult:g}", "max_abs", float(mult))
