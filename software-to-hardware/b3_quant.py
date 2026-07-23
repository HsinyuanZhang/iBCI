"""INT8 fixed-point primitives for B3 EarlyPool (hardware baseline scheme).

Quantization policy (round 1):
  - activations: signed INT8, per-tensor symmetric
  - weights: signed INT8, per-output-channel symmetric
  - bias: INT32 in accumulator domain
  - dot accumulator: INT32
  - ReLU output: INT8 requant
  - SUM_feat across trials: INT32
  - SUM/M: fixed-point reciprocal + requant
  - final E: INT8 and INT16 variants
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from b3_hw_golden import B3Shapes, B3Weights, forward_b3_layered

INT8_MIN, INT8_MAX = -128, 127
INT16_MIN, INT16_MAX = -32768, 32767


@dataclass
class QuantTensor:
    """Symmetric quantized tensor with a single scale."""

    values: np.ndarray
    scale: float
    dtype: str

    def dequant(self) -> np.ndarray:
        return self.values.astype(np.float64) * self.scale


@dataclass
class QuantLinearParams:
    weight_q: np.ndarray  # [out, in] int8
    weight_scale: np.ndarray  # [out] float, per-output-channel
    bias_i32: np.ndarray  # [out] int32
    input_scale: float
    output_scale: float
    name: str


@dataclass
class ReciprocalParams:
    M: int
    shift: int
    recip: int  # round(2^shift / M)


@dataclass
class B3QuantBundle:
    shapes: B3Shapes
    input_scale: float
    layers: List[QuantLinearParams]
    reciprocal: ReciprocalParams
    mean_feat_scale: float
    E_int8_scale: float
    E_int16_scale: float


def _safe_scale(max_abs: float, qmax: float = 127.0, floor: float = 1e-8) -> float:
    return max(float(max_abs) / qmax, floor)


def quantize_symmetric_int8(x_fp: np.ndarray, scale: Optional[float] = None) -> Tuple[np.ndarray, float]:
    x_fp = x_fp.astype(np.float64)
    if scale is None:
        scale = _safe_scale(np.max(np.abs(x_fp)))
    q = np.round(x_fp / scale)
    q = np.clip(q, INT8_MIN, INT8_MAX).astype(np.int8)
    return q, float(scale)


def quantize_symmetric_int16(x_fp: np.ndarray, scale: Optional[float] = None) -> Tuple[np.ndarray, float]:
    x_fp = x_fp.astype(np.float64)
    if scale is None:
        scale = _safe_scale(np.max(np.abs(x_fp)), qmax=32767.0)
    q = np.round(x_fp / scale)
    q = np.clip(q, INT16_MIN, INT16_MAX).astype(np.int16)
    return q, float(scale)


def quantize_weight_per_channel(w_fp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """w_fp [out, in] -> int8 weight + float scale [out]."""
    out_f, _ = w_fp.shape
    w_q = np.zeros_like(w_fp, dtype=np.int8)
    scales = np.zeros((out_f,), dtype=np.float64)
    for o in range(out_f):
        row = w_fp[o].astype(np.float64)
        scale = _safe_scale(np.max(np.abs(row)))
        scales[o] = scale
        w_q[o] = np.clip(np.round(row / scale), INT8_MIN, INT8_MAX).astype(np.int8)
    return w_q, scales


def pack_linear_params(
    w_fp: np.ndarray,
    b_fp: np.ndarray,
    input_scale: float,
    output_scale: float,
    name: str,
) -> QuantLinearParams:
    w_q, w_scale = quantize_weight_per_channel(w_fp)
    eff = input_scale * w_scale
    bias_i32 = np.round(b_fp.astype(np.float64) / np.maximum(eff, 1e-12)).astype(np.int32)
    return QuantLinearParams(
        weight_q=w_q,
        weight_scale=w_scale.astype(np.float32),
        bias_i32=bias_i32,
        input_scale=float(input_scale),
        output_scale=float(output_scale),
        name=name,
    )


def linear_int8_to_int32(
    x_q: np.ndarray,
    layer: QuantLinearParams,
) -> np.ndarray:
    """x_q [..., in] int8 -> acc int32 [..., out]."""
    x64 = x_q.astype(np.int32)
    w64 = layer.weight_q.astype(np.int32)
    acc = x64 @ w64.T + layer.bias_i32.astype(np.int32)
    return acc.astype(np.int32)


def requant_int8_from_acc(
    acc_i32: np.ndarray,
    layer: QuantLinearParams,
    apply_relu: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """INT32 accumulator -> INT8 activation requant."""
    eff = layer.input_scale * layer.weight_scale
    y_fp = acc_i32.astype(np.float64) * eff[None, :] if acc_i32.ndim == 2 else acc_i32.astype(np.float64) * eff
    if apply_relu:
        y_fp = np.maximum(y_fp, 0.0)
    out_scale = layer.output_scale
    q = np.round(y_fp / out_scale)
    if apply_relu:
        q = np.clip(q, 0, INT8_MAX)
    else:
        q = np.clip(q, INT8_MIN, INT8_MAX)
    return q.astype(np.int8), y_fp.astype(np.float32)


def requant_int8_from_acc_nd(acc_i32: np.ndarray, layer: QuantLinearParams, apply_relu: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """General rank acc -> same rank int8."""
    eff = (layer.input_scale * layer.weight_scale).astype(np.float64)
    # broadcast eff on last dim
    shape = [1] * (acc_i32.ndim - 1) + [eff.shape[0]]
    y_fp = acc_i32.astype(np.float64) * eff.reshape(shape)
    if apply_relu:
        y_fp = np.maximum(y_fp, 0.0)
    q = np.round(y_fp / layer.output_scale)
    if apply_relu:
        q = np.clip(q, 0, INT8_MAX)
    else:
        q = np.clip(q, INT8_MIN, INT8_MAX)
    return q.astype(np.int8), y_fp.astype(np.float32)


def make_reciprocal(M: int, shift: int = 20) -> ReciprocalParams:
    if M <= 0:
        raise ValueError("M must be > 0")
    recip = int(round((1 << shift) / M))
    return ReciprocalParams(M=M, shift=shift, recip=recip)


def mean_int32_via_reciprocal(sum_i32: np.ndarray, recip: ReciprocalParams) -> np.ndarray:
    """sum_i32 * round(2^s/M) >> s  with round-to-nearest."""
    prod = sum_i32.astype(np.int64) * int(recip.recip)
    if recip.shift > 0:
        bias = 1 << (recip.shift - 1)
        return ((prod + bias) >> recip.shift).astype(np.int32)
    return prod.astype(np.int32)


def saturation_count(q: np.ndarray, pre_clip: np.ndarray, signed: bool = True) -> int:
    if signed:
        lo, hi = INT8_MIN, INT8_MAX
    else:
        lo, hi = 0, INT8_MAX
    rounded = np.round(pre_clip)
    return int(np.sum((rounded < lo) | (rounded > hi)))


def build_quant_bundle(
    weights: B3Weights,
    calib_fp: np.ndarray,
    shapes: B3Shapes,
    fp_stages: Optional[Dict[str, np.ndarray]] = None,
    recip_shift: int = 20,
) -> B3QuantBundle:
    """Calibrate per-tensor activation scales from FP32 reference activations."""
    if fp_stages is None:
        fp_stages = forward_b3_layered(calib_fp, weights)

    _, calib_scale = quantize_symmetric_int8(calib_fp)

    # Per-layer output scales from FP32 tensor max abs (post-ReLU uses nonneg max)
    pre_relu_fp = fp_stages["feat"]  # after ReLU
    _, pre_out_scale = quantize_symmetric_int8(np.maximum(pre_relu_fp, 0.0))

    post0_fp = fp_stages["post0_relu"]
    _, post0_out_scale = quantize_symmetric_int8(np.maximum(post0_fp, 0.0))

    post1_fp = fp_stages["post1_relu"]
    _, post1_out_scale = quantize_symmetric_int8(np.maximum(post1_fp, 0.0))

    mean_fp = fp_stages["mean_feat"]
    _, mean_scale = quantize_symmetric_int8(np.maximum(mean_fp, 0.0))

    E_fp = fp_stages["E"]
    _, E_int8_scale = quantize_symmetric_int8(E_fp)
    _, E_int16_scale = quantize_symmetric_int16(E_fp)

    layers = [
        pack_linear_params(weights.pre_w, weights.pre_b, calib_scale, pre_out_scale, "pre_pool"),
        pack_linear_params(weights.post0_w, weights.post0_b, mean_scale, post0_out_scale, "post0"),
        pack_linear_params(weights.post1_w, weights.post1_b, post0_out_scale, post1_out_scale, "post1"),
        pack_linear_params(weights.post2_w, weights.post2_b, post1_out_scale, E_int8_scale, "post2"),
    ]

    return B3QuantBundle(
        shapes=shapes,
        input_scale=calib_scale,
        layers=layers,
        reciprocal=make_reciprocal(shapes.M, shift=recip_shift),
        mean_feat_scale=mean_scale,
        E_int8_scale=E_int8_scale,
        E_int16_scale=E_int16_scale,
    )


def forward_b3_int8_layered(
    calib_fp: np.ndarray,
    bundle: B3QuantBundle,
) -> Dict[str, Any]:
    """Integer forward matching the round-1 HW policy; returns int + dequant views."""
    M, T, N = calib_fp.shape
    D = bundle.shapes.D
    W = bundle.shapes.W

    calib_q, _ = quantize_symmetric_int8(calib_fp, bundle.input_scale)
    pre = bundle.layers[0]

    feat_trials_i8 = np.zeros((M, N, D), dtype=np.int8)
    pre_acc_trials = np.zeros((M, N, D), dtype=np.int32)
    pre_fp_trials = np.zeros((M, N, D), dtype=np.float32)

    for m in range(M):
        for n in range(N):
            x_q = calib_q[m, :, n]
            acc = linear_int8_to_int32(x_q, pre)
            feat_i8, feat_fp = requant_int8_from_acc(acc[None, :], pre, apply_relu=True)
            pre_acc_trials[m, n] = acc
            feat_trials_i8[m, n] = feat_i8[0]
            pre_fp_trials[m, n] = feat_fp[0]

    sum_feat_i32 = feat_trials_i8.astype(np.int32).sum(axis=0)  # [N, D]
    mean_i32 = mean_int32_via_reciprocal(sum_feat_i32, bundle.reciprocal)

    pre_out_scale = bundle.layers[0].output_scale
    mean_fp_est = np.maximum(mean_i32.astype(np.float64) * pre_out_scale, 0.0)
    mean_q, _ = quantize_symmetric_int8(mean_fp_est, bundle.mean_feat_scale)

    post0, post1, post2 = bundle.layers[1], bundle.layers[2], bundle.layers[3]

    post0_acc = linear_int8_to_int32(mean_q, post0)
    post0_i8, post0_fp = requant_int8_from_acc_nd(post0_acc, post0, apply_relu=True)

    post1_acc = linear_int8_to_int32(post0_i8, post1)
    post1_i8, post1_fp = requant_int8_from_acc_nd(post1_acc, post1, apply_relu=True)

    post2_acc = linear_int8_to_int32(post1_i8, post2)
    _, E_fp_from_int = requant_int8_from_acc_nd(post2_acc, post2, apply_relu=False)

    E_int8, _ = quantize_symmetric_int8(E_fp_from_int, bundle.E_int8_scale)
    E_int16, _ = quantize_symmetric_int16(E_fp_from_int, bundle.E_int16_scale)

    sum_after = np.zeros((M, N, D), dtype=np.int32)
    running = np.zeros((N, D), dtype=np.int32)
    for m in range(M):
        running = running + feat_trials_i8[m].astype(np.int32)
        sum_after[m] = running

    return {
        "calib_q": calib_q,
        "pre_acc": pre_acc_trials,
        "feat_i8": feat_trials_i8,
        "feat_fp": pre_fp_trials,
        "sum_after_trial_i32": sum_after,
        "sum_feat_i32": sum_feat_i32,
        "mean_i32": mean_i32,
        "mean_fp_est": mean_fp_est.astype(np.float32),
        "mean_q": mean_q,
        "post0_acc": post0_acc,
        "post0_i8": post0_i8,
        "post0_fp": post0_fp,
        "post1_acc": post1_acc,
        "post1_i8": post1_i8,
        "post1_fp": post1_fp,
        "post2_acc": post2_acc,
        "E_fp": E_fp_from_int,
        "E_int8": E_int8,
        "E_int16": E_int16,
        "E_int8_dequant": E_int8.astype(np.float64) * bundle.E_int8_scale,
        "E_int16_dequant": E_int16.astype(np.float64) * bundle.E_int16_scale,
    }


def compare_fp32_vs_int8(
    fp_stages: Dict[str, np.ndarray],
    int8_out: Dict[str, Any],
    bundle: B3QuantBundle,
) -> Dict[str, Any]:
    """Error report vs FP32 golden."""
    E_fp = fp_stages["E"]
    reports = {}
    for key, pred in [
        ("E_int8_dequant", int8_out["E_int8_dequant"]),
        ("E_int16_dequant", int8_out["E_int16_dequant"]),
        ("E_fp_int_path", int8_out["E_fp"].astype(np.float32)),
    ]:
        diff = np.abs(pred.astype(np.float64) - E_fp.astype(np.float64))
        reports[key] = {
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
        }

    stage_pairs = [
        ("feat", "feat_fp"),
        ("mean_feat", None),  # special: compare mean_i32 float proxy
        ("post0_relu", "post0_fp"),
        ("post1_relu", "post1_fp"),
    ]
    for fp_key, int_key in stage_pairs:
        if int_key is None:
            ref = fp_stages["mean_feat"]
            pred = int8_out["mean_fp_est"]
        else:
            ref = fp_stages[fp_key]
            pred = int8_out[int_key]
        diff = np.abs(pred.astype(np.float64) - ref.astype(np.float64))
        reports[f"stage_{fp_key}"] = {
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
        }

    # SUM_feat: int path sums int8; compare to float sum of quantized feats
    ref_sum = fp_stages["sum_feat"]
    pred_sum = int8_out["sum_feat_i32"].astype(np.float64)
    diff_sum = np.abs(pred_sum - ref_sum)
    reports["sum_feat_i32_vs_fp"] = {"max_abs": float(diff_sum.max()), "mean_abs": float(diff_sum.mean())}

    return reports


def bundle_to_jsonable(bundle: B3QuantBundle) -> Dict[str, Any]:
    return {
        "shapes": asdict(bundle.shapes),
        "input_scale": bundle.input_scale,
        "mean_feat_scale": bundle.mean_feat_scale,
        "E_int8_scale": bundle.E_int8_scale,
        "E_int16_scale": bundle.E_int16_scale,
        "reciprocal": asdict(bundle.reciprocal),
        "layers": [
            {
                "name": layer.name,
                "input_scale": layer.input_scale,
                "output_scale": layer.output_scale,
                "weight_scale": layer.weight_scale.tolist(),
                "bias_i32": layer.bias_i32.tolist(),
            }
            for layer in bundle.layers
        ],
    }


def save_quant_run(
    out: Path,
    bundle: B3QuantBundle,
    int8_stages: Dict[str, Any],
    fp_stages: Dict[str, np.ndarray],
    report: Dict[str, Any],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    qdir = out / "quant_weights"
    qdir.mkdir(exist_ok=True)

    for idx, layer in enumerate(bundle.layers):
        np.save(qdir / f"{layer.name}_w_int8.npy", layer.weight_q)
        np.save(qdir / f"{layer.name}_w_scale.npy", layer.weight_scale)
        np.save(qdir / f"{layer.name}_bias_i32.npy", layer.bias_i32)

    sdir = out / "stages_int8"
    sdir.mkdir(exist_ok=True)
    save_map = {
        "calib_q": "calib_q",
        "feat_i8": "feat",
        "sum_feat_i32": "sum_feat_i32",
        "sum_after_trial_i32": "sum_after_trial_i32",
        "mean_i32": "mean_i32",
        "mean_q": "mean_q",
        "post0_acc": "post0_acc",
        "post0_i8": "post0_relu",
        "post1_acc": "post1_acc",
        "post1_i8": "post1_relu",
        "post2_acc": "post2_acc",
        "E_int8": "E_int8",
        "E_int16": "E_int16",
    }
    for src, name in save_map.items():
        np.save(sdir / f"{name}.npy", int8_stages[src])

    np.save(out / "E_int8.npy", int8_stages["E_int8"])
    np.save(out / "E_int16.npy", int8_stages["E_int16"])
    np.save(out / "E_int8_dequant.npy", int8_stages["E_int8_dequant"].astype(np.float32))
    np.save(out / "E_int16_dequant.npy", int8_stages["E_int16_dequant"].astype(np.float32))

    meta = {
        "variant": "B3_EarlyPool_INT8_baseline",
        "quant_policy": bundle_to_jsonable(bundle),
        "validation_vs_fp32": report,
        "notes": [
            "Input: signed INT8 per-tensor symmetric.",
            "Weights: signed INT8 per-output-channel symmetric.",
            "Bias INT32 in accumulator domain; dot INT32; ReLU then INT8 requant.",
            "SUM_feat INT32 across trials; mean via reciprocal shift.",
            "Export E as INT8 and INT16 with separate scales.",
        ],
    }
    (out / "quant_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        "B3 INT8 baseline validation",
        f"M={bundle.shapes.M} T={bundle.shapes.T} D={bundle.shapes.D} W={bundle.shapes.W} N={bundle.shapes.N}",
        f"input_scale={bundle.input_scale:.6e}",
        f"reciprocal: recip={bundle.reciprocal.recip} shift={bundle.reciprocal.shift} (M={bundle.reciprocal.M})",
        f"E_int8_scale={bundle.E_int8_scale:.6e}  E_int16_scale={bundle.E_int16_scale:.6e}",
        "vs FP32 E:",
    ]
    for k, v in report.items():
        if k.startswith("E_"):
            lines.append(f"  {k}: max_abs={v['max_abs']:.6e} rmse={v.get('rmse', v.get('mean_abs', 0)):.6e}")
    (out / "quant_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
