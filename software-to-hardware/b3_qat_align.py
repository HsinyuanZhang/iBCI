#!/usr/bin/env python3
"""Verify QAT integer forward is bit-exact vs b3_quant_engine layer-by-layer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _export_weights(enc: QATEarlyPoolEncoder) -> "B3Weights":
    from b3_hw_golden import B3Weights
    def _np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32)

    return B3Weights(
        pre_w=_np(enc.pre_linear.weight),
        pre_b=_np(enc.pre_linear.bias),
        post0_w=_np(enc.post_linears[0].weight),
        post0_b=_np(enc.post_linears[0].bias),
        post1_w=_np(enc.post_linears[1].weight),
        post1_b=_np(enc.post_linears[1].bias),
        post2_w=_np(enc.post_linears[2].weight),
        post2_b=_np(enc.post_linears[2].bias),
    )


def _compare(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "stage": name,
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "exact": bool(np.max(diff) == 0) if diff.size else True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--scale-method", default="mse_opt")
    parser.add_argument("--scale-mult", type=float, default=1.0)
    parser.add_argument("--out", default="runs/b3_qat_align")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    if str(exp_root.resolve()) not in sys.path:
        sys.path.insert(0, str(exp_root.resolve()))

    from b3_ckpt_loader import load_b3_weights_from_ckpt
    from b3_eval_protocol import (
        build_loso_datamodule,
        get_full_calib_pool,
        load_split_manifest,
        load_student_from_ckpt,
        sample_calib_draw,
        stable_session_seed,
    )
    from b3_fake_quant import B3QATScales
    from b3_hw_golden import B3Shapes, B3Weights
    from b3_ptq import calibrate_scales_from_stats, collect_activation_stats
    from b3_qat_encoder import QATEarlyPoolEncoder, wrap_early_pool_with_qat
    from b3_quant_engine import ABLATION_PRESETS, build_quant_engine_bundle, forward_quant_engine, quantize_tensor

    ckpt = Path(args.ckpt)
    split = load_split_manifest(ckpt.parent.parent / "split_manifest.json")
    dm = build_loso_datamodule(exp_root, Path(args.data_dir), split.fold_id)
    dm.setup("fit")
    calibs = []
    for sess in split.train_sessions:
        pool = get_full_calib_pool(dm.train_dataset, sess)
        _, c = sample_calib_draw(pool, num_trials=33, seed=stable_session_seed(42, sess))
        calibs.append(c)
    heldout_calib = get_full_calib_pool(dm.val_heldin_dataset, split.heldout_session)[:33]

    student = load_student_from_ckpt(ckpt, exp_root)
    base = student.id_encoder
    w = load_b3_weights_from_ckpt(ckpt)
    stats = collect_activation_stats(w, calibs)
    frozen = calibrate_scales_from_stats(stats, args.scale_method, mult=args.scale_mult, source_sessions=split.train_sessions)
    scales = B3QATScales(
        input=frozen.input_scale_i8,
        pre_out=frozen.pre_out_scale_i8,
        mean=frozen.mean_scale_i8,
        post0_out=frozen.post0_out_scale_i8,
        post1_out=frozen.post1_out_scale_i8,
        E=frozen.E_int8_scale,
    )
    qat = wrap_early_pool_with_qat(base, scales, num_trials=33)
    qat.eval()

    calib_t = torch.from_numpy(heldout_calib).unsqueeze(0)
    with torch.no_grad():
        stages_t = qat.forward_integer_with_stages(calib_t)

    weights = _export_weights(qat)
    shapes = B3Shapes(T=heldout_calib.shape[1], D=64, W=50, N=heldout_calib.shape[2], M=33)
    bundle = build_quant_engine_bundle(weights, shapes, qat.to_frozen_scales(split.train_sessions), ABLATION_PRESETS["w8_a8_e8"])
    stages_np = forward_quant_engine(heldout_calib, bundle)

    calib_q_np, _ = quantize_tensor(heldout_calib, 8, bundle.scales.input_scale_i8)

    rows = [
        _compare("calib_q", stages_t["calib_q"].squeeze(0).cpu().numpy(), calib_q_np),
        _compare("sum_feat", stages_t["sum_feat"].squeeze(0).cpu().numpy(), stages_np["sum_feat_i32"].astype(np.float32)),
        _compare("mean_q", stages_t["mean_q"].squeeze(0).cpu().numpy(), _mean_act_from_engine(stages_np, bundle)),
        _compare("E_q", stages_t["E_q"].squeeze(0).cpu().numpy(), stages_np["E"].astype(np.float32)),
        _compare("E_dequant", stages_t["E_dequant"].squeeze(0).cpu().numpy(), stages_np["E_dequant"]),
    ]
    all_exact = all(r["exact"] for r in rows)
    report = {"rows": rows, "all_exact": all_exact, "pass": all_exact}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "align_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all_exact else 1


def _mean_act_from_engine(stages_np: dict, bundle) -> np.ndarray:
    from b3_quant_engine import quantize_tensor

    abits = bundle.ablation.activation_bits
    mean_i32 = stages_np["mean_i32"]
    mean_fp_est = np.maximum(mean_i32.astype(np.float64) * bundle.scales.pre_out_scale(abits), 0.0)
    mean_act, _ = quantize_tensor(mean_fp_est, abits, bundle.scales.mean_scale(abits))
    return mean_act.astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
