#!/usr/bin/env python3
"""B3 EarlyPool golden vector exporter for RTL bring-up (self-contained).

Produces, for a chosen shape profile:
  - FP32 layered stages  (S0..S7)          via b3_hw_golden.forward_b3_layered
  - INT8 W8A8 integer stages + diagnostics  via b3_quant_engine.forward_quant_engine
  - flat testbench dumps (.hex / .dec) for weights, bias, requant, and E

Scales here are CALIBRATED ON THE PROVIDED (random or supplied) INPUT for
bring-up only. This is NOT a model sign-off release; for sign-off use the
frozen model_release/ package described in ../01_OPERATORS.md.

Examples
--------
  python export_golden.py --profile tiny --out vectors/tiny
  python export_golden.py --profile d64  --out vectors/d64
  python export_golden.py --profile full_m33_d64 --out vectors/full --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from b3_hw_golden import (
    PROFILES,
    B3Shapes,
    forward_b3_layered,
    random_calib,
    random_weights,
)
from b3_quant_engine import (
    QuantAblation,
    build_quant_engine_bundle,
    calibrate_frozen_scales,
    forward_quant_engine,
    identity_metrics,
)


def _dump_int_array(path: Path, arr: np.ndarray, hexdigits: int) -> None:
    """Write a flat one-value-per-line integer file (.dec and .hex)."""
    flat = np.asarray(arr).reshape(-1).astype(np.int64)
    dec = "\n".join(str(int(v)) for v in flat) + "\n"
    path.with_suffix(".dec").write_text(dec, encoding="utf-8")
    mask = (1 << (4 * hexdigits)) - 1
    hx = "\n".join(f"{int(v) & mask:0{hexdigits}x}" for v in flat) + "\n"
    path.with_suffix(".hex").write_text(hx, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=list(PROFILES), default="tiny")
    ap.add_argument("--out", type=str, default="vectors/out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recip-shift", type=int, default=20)
    args = ap.parse_args()

    base = dict(PROFILES[args.profile])
    base["seed"] = args.seed
    shapes = B3Shapes(**base)
    rng = np.random.default_rng(shapes.seed)

    weights = random_weights(shapes, rng)
    calib = random_calib(shapes, rng)

    # ---- FP32 layered golden ----
    fp = forward_b3_layered(calib, weights)

    # ---- INT8 W8A8 integer engine ----
    scales = calibrate_frozen_scales(weights, [calib], [f"{args.profile}_seed{args.seed}"])
    ablation = QuantAblation(8, 8, 8, True, "w8_a8_e8")
    bundle = build_quant_engine_bundle(weights, shapes, scales, ablation, recip_shift=args.recip_shift)
    q = forward_quant_engine(calib, bundle)

    out = Path(args.out)
    (out / "fp32_stages").mkdir(parents=True, exist_ok=True)
    (out / "int8_stages").mkdir(parents=True, exist_ok=True)
    (out / "coeff").mkdir(parents=True, exist_ok=True)

    # FP32 stages
    for k in ("pre_linear", "feat", "sum_feat", "mean_feat",
              "post0_relu", "post1_relu", "E"):
        np.save(out / "fp32_stages" / f"{k}.npy", fp[k])

    # INT8 stages (contract-relevant)
    np.save(out / "int8_stages" / "feat_i8.npy", q["feat"])
    np.save(out / "int8_stages" / "sum_feat_i32.npy", q["sum_feat_i32"])
    np.save(out / "int8_stages" / "mean_i32.npy", q["mean_i32"])
    np.save(out / "int8_stages" / "E_i8.npy", q["E"])
    np.save(out / "int8_stages" / "E_dequant.npy", q["E_dequant"])

    # Coefficient image: per-layer int8 weights, int32 bias, requant mult/shift.
    layer_meta = []
    for layer in bundle.layers:
        d = out / "coeff" / layer.name
        d.mkdir(parents=True, exist_ok=True)
        _dump_int_array(d / "weight_i8", layer.w_q, hexdigits=2)
        _dump_int_array(d / "bias_i32", layer.bias_i32, hexdigits=8)
        _dump_int_array(d / "requant_mult_i32", layer.requant.mult, hexdigits=8)
        np.save(d / "weight_i8.npy", layer.w_q)
        np.save(d / "bias_i32.npy", layer.bias_i32)
        np.save(d / "requant_mult_i32.npy", layer.requant.mult)
        layer_meta.append({
            "name": layer.name,
            "weight_shape": list(layer.w_q.shape),
            "requant_shift": int(layer.requant.shift),
            "in_scale": float(layer.in_scale),
            "out_scale": float(layer.out_scale),
        })

    # Input calib as int8 (activation quant of the input edge)
    calib_i8 = np.clip(np.round(calib / scales.input_scale_i8), -128, 127).astype(np.int8)
    _dump_int_array(out / "calib_i8", calib_i8, hexdigits=2)
    np.save(out / "calib_i8.npy", calib_i8)
    _dump_int_array(out / "E_i8", q["E"], hexdigits=2)

    metrics = identity_metrics(fp["E"], q["E_dequant"])
    manifest = {
        "note": "BRING-UP golden (input-calibrated scales). NOT a sign-off release.",
        "profile": args.profile,
        "shapes": {"T": shapes.T, "D": shapes.D, "W": shapes.W,
                   "N": shapes.N, "M": shapes.M, "seed": shapes.seed},
        "ablation": ablation.name,
        "reciprocal": bundle.reciprocal,
        "reciprocal_shift": bundle.reciprocal_shift,
        "requant_shift": 31,
        "rounding": "(product + 2^(shift-1)) >>> shift  (arithmetic right shift)",
        "layers": layer_meta,
        "scales": {
            "input_i8": scales.input_scale_i8,
            "pre_out_i8": scales.pre_out_scale_i8,
            "mean_i8": scales.mean_scale_i8,
            "post0_out_i8": scales.post0_out_scale_i8,
            "post1_out_i8": scales.post1_out_scale_i8,
            "E_i8": scales.E_int8_scale,
        },
        "fp32_vs_integer_identity": metrics,
        "integer_diagnostics": q["diagnostics"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"Wrote golden vectors to {out}")
    print(f"  shapes: T={shapes.T} D={shapes.D} W={shapes.W} N={shapes.N} M={shapes.M}")
    print(f"  E int8 shape: {q['E'].shape}")
    print(f"  FP32 vs integer(dequant) E: max_abs={metrics['max_abs']:.4e} "
          f"cosine={metrics['cosine']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
