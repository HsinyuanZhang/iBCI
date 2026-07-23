#!/usr/bin/env python3
"""Validate B3 INT8 baseline quantization before RTL.

Exports integer golden (weights, scales, layered int8/int32 activations) and
reports error vs FP32 reference from b3_hw_golden.py.

Run inside software-to-hardware/ after copy:
  python b3_int8_validate.py --profile tiny --out runs/tiny_int8
  python b3_int8_validate.py --profile d64 --fp-ref runs/d64_sw --out runs/d64_int8
  python b3_int8_validate.py --compare runs/tiny_int8 runs/tiny_int8_rtl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from b3_hw_golden import (  # noqa: E402
    PROFILES,
    B3Shapes,
    B3Weights,
    forward_b3_layered,
    load_weights,
    parse_shapes as parse_fp_shapes,
    random_calib,
    random_weights,
)
from b3_quant import (  # noqa: E402
    build_quant_bundle,
    compare_fp32_vs_int8,
    forward_b3_int8_layered,
    save_quant_run,
)


INT8_STAGE_KEYS = [
    "calib_q",
    "feat",
    "sum_feat_i32",
    "sum_after_trial_i32",
    "mean_i32",
    "mean_q",
    "post0_acc",
    "post0_relu",
    "post1_acc",
    "post1_relu",
    "post2_acc",
    "E_int8",
    "E_int16",
]


def load_fp_reference(ref_dir: Path) -> Tuple[np.ndarray, B3Weights, Dict[str, np.ndarray]]:
    weights = load_weights(ref_dir / "weights")
    calib = np.load(ref_dir / "stages" / "calib.npy").astype(np.float32)
    fp_stages = {k: np.load(ref_dir / "stages" / f"{k}.npy") for k in [
        "feat", "sum_feat", "mean_feat", "post0_relu", "post1_relu", "E",
    ]}
    fp_stages["calib"] = calib
    fp_stages["sum_feat"] = np.load(ref_dir / "stages" / "sum_feat.npy")
    return calib, weights, fp_stages


def max_abs_rel(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    denom = np.maximum(np.abs(b.astype(np.float64)), 1e-12)
    return float(diff.max()), float((diff / denom).max())


def compare_int8_dirs(ref_dir: Path, dut_dir: Path, atol: float, rtol: float) -> int:
    ref_stages = ref_dir / "stages_int8"
    dut_stages = dut_dir / "stages_int8"
    failed: List[str] = []

    for key in INT8_STAGE_KEYS:
        ref_path = ref_stages / f"{key}.npy"
        dut_path = dut_stages / f"{key}.npy"
        if not ref_path.exists():
            continue
        if not dut_path.exists():
            print(f"[SKIP] {key}: missing in DUT")
            continue
        ref = np.load(ref_path)
        dut = np.load(dut_path)
        if ref.shape != dut.shape:
            print(f"[FAIL] {key}: shape {dut.shape} != ref {ref.shape}")
            failed.append(key)
            continue
        if np.issubdtype(ref.dtype, np.integer):
            ok = np.array_equal(ref, dut)
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {key}: integer exact match shape={ref.shape}")
        else:
            mad, mrd = max_abs_rel(dut, ref)
            ok = bool(np.allclose(dut, ref, atol=atol, rtol=rtol))
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {key}: max_abs={mad:.6e} max_rel={mrd:.6e}")
        if not ok:
            failed.append(key)

    for e_key in ("E_int8", "E_int16"):
        for root, name in ((ref_dir, "REF"), (dut_dir, "DUT")):
            p = root / f"{e_key}.npy"
            if not p.exists():
                print(f"[WARN] {name} missing {e_key}")
        ref_e = np.load(ref_dir / f"{e_key}.npy")
        dut_e = np.load(dut_dir / f"{e_key}.npy")
        ok = np.array_equal(ref_e, dut_e)
        print(f"[{'PASS' if ok else 'FAIL'}] {e_key} top-level exact")
        if not ok:
            failed.append(e_key)

    if failed:
        print("FAILED:", failed)
        return 1
    print("All compared INT8 stages PASS")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    shapes = parse_fp_shapes(args)
    rng = np.random.default_rng(shapes.seed)

    if args.fp_ref:
        ref_dir = Path(args.fp_ref)
        calib, weights, fp_stages = load_fp_reference(ref_dir)
        if calib.shape != (shapes.M, shapes.T, shapes.N):
            raise SystemExit(f"fp_ref calib {calib.shape} != profile {(shapes.M, shapes.T, shapes.N)}")
    else:
        if args.weights_dir:
            weights = load_weights(Path(args.weights_dir))
        else:
            weights = random_weights(shapes, rng)
        if args.calib:
            calib = np.load(args.calib).astype(np.float32)
        else:
            calib = random_calib(shapes, rng)
        fp_stages = forward_b3_layered(calib, weights)

    bundle = build_quant_bundle(weights, calib, shapes, fp_stages, recip_shift=args.recip_shift)
    int8_out = forward_b3_int8_layered(calib, bundle)
    report = compare_fp32_vs_int8(fp_stages, int8_out, bundle)

    out = Path(args.out)
    save_quant_run(out, bundle, int8_out, fp_stages, report)

    print(f"Wrote INT8 golden to {out}")
    print((out / "quant_summary.txt").read_text(encoding="utf-8"))

    e8 = report["E_int8_dequant"]
    e16 = report["E_int16_dequant"]
    print(f"PASS criteria hint: E_int8 max_abs={e8['max_abs']:.4e}, E_int16 max_abs={e16['max_abs']:.4e}")

    if args.fail_if_e_max_abs is not None:
        if e8["max_abs"] > args.fail_if_e_max_abs:
            print(f"FAIL: E_int8 max_abs {e8['max_abs']} > {args.fail_if_e_max_abs}")
            return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=list(PROFILES), default="tiny")
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--D", type=int, default=None)
    parser.add_argument("--W", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="runs/default_int8")
    parser.add_argument("--fp-ref", type=str, default=None, help="Existing FP32 golden dir (weights+stages)")
    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--calib", type=str, default=None)
    parser.add_argument("--recip-shift", type=int, default=20, help="Fixed-point shift for 1/M")
    parser.add_argument("--fail-if-e-max-abs", type=float, default=None)
    parser.add_argument("--compare", nargs=2, metavar=("REF_DIR", "DUT_DIR"))
    parser.add_argument("--atol", type=float, default=0.0, help="float compare only; integers are exact")
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()

    if args.compare:
        return compare_int8_dirs(Path(args.compare[0]), Path(args.compare[1]), args.atol, args.rtol)
    return cmd_validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
