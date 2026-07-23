#!/usr/bin/env python3
"""Four-path QAT evaluation: anchor / shadow / fake-quant / integer engine."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from b3_eval_protocol import (
    check_baseline_r2,
    collect_session_windows,
    get_full_calib_pool,
    session_r2_fp32_encoder,
    session_r2_with_E,
)
from b3_hw_golden import B3Shapes, B3Weights
from b3_quant_engine import ABLATION_PRESETS, build_quant_engine_bundle, forward_quant_engine, identity_metrics


def _export_weights(qat_encoder) -> B3Weights:
    def _np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32)

    return B3Weights(
        pre_w=_np(qat_encoder.pre_linear.weight),
        pre_b=_np(qat_encoder.pre_linear.bias),
        post0_w=_np(qat_encoder.post_linears[0].weight),
        post0_b=_np(qat_encoder.post_linears[0].bias),
        post1_w=_np(qat_encoder.post_linears[1].weight),
        post1_b=_np(qat_encoder.post_linears[1].bias),
        post2_w=_np(qat_encoder.post_linears[2].weight),
        post2_b=_np(qat_encoder.post_linears[2].bias),
    )


def evaluate_four_paths(
    lit,
    *,
    student_anchor,
    val_ds,
    heldout_session: str,
    behavior_scale: float,
    expected_anchor_r2: float = 0.63024879,
    batch_size: int = 32,
) -> Dict[str, Any]:
    lit.eval()
    device = next(lit.qat_encoder.parameters()).device
    lit.anchor_encoder.to(device)
    neural, behavior = collect_session_windows(val_ds, heldout_session, batch_size=batch_size)
    calib = get_full_calib_pool(val_ds, heldout_session)[:33].astype(np.float32)
    calib_t = torch.from_numpy(calib).unsqueeze(0).to(device)

    with torch.no_grad():
        e_anchor = lit.anchor_encoder.forward_batch(calib_t).squeeze(0).cpu().numpy()
        e_shadow = lit.qat_encoder.forward_fp32(calib_t).squeeze(0).cpu().numpy()
        e_fake = lit.qat_encoder.forward_batch(calib_t).squeeze(0).cpu().numpy()

    shapes = B3Shapes(T=calib.shape[1], D=64, W=50, N=calib.shape[2], M=33)
    bundle = build_quant_engine_bundle(
        _export_weights(lit.qat_encoder),
        shapes,
        lit.qat_encoder.to_frozen_scales([]),
        ABLATION_PRESETS["w8_a8_e8"],
    )
    e_int = forward_quant_engine(calib, bundle)["E_dequant"]

    r2_anchor = session_r2_fp32_encoder(student_anchor, neural, behavior, calib, behavior_scale=behavior_scale)
    r2_shadow = session_r2_with_E(student_anchor, neural, behavior, e_shadow, behavior_scale=behavior_scale)
    r2_fake = session_r2_with_E(student_anchor, neural, behavior, e_fake, behavior_scale=behavior_scale)
    r2_int = session_r2_with_E(student_anchor, neural, behavior, e_int, behavior_scale=behavior_scale)
    e_align = identity_metrics(e_fake, e_int)

    return {
        "heldout_session": heldout_session,
        "baseline_self_check": check_baseline_r2(r2_anchor, expected_anchor_r2, session=heldout_session),
        "r2_anchor_fp32": r2_anchor,
        "r2_shadow_fp32": r2_shadow,
        "r2_fake_quant": r2_fake,
        "r2_integer": r2_int,
        "delta_r2_shadow": r2_shadow - r2_anchor,
        "delta_r2_fake_quant": r2_fake - r2_anchor,
        "delta_r2_integer": r2_int - r2_anchor,
        "fake_vs_integer_E": e_align,
        "fake_integer_E_exact": e_align["max_abs"] == 0.0,
        "shadow_matches_anchor": abs(r2_shadow - r2_anchor) < 1e-5,
    }


def main() -> int:
    import argparse

    from b3_ckpt_loader import load_hyperparams_from_ckpt
    from b3_eval_protocol import build_loso_datamodule, load_split_manifest, load_student_from_ckpt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qat-ckpt", required=True)
    parser.add_argument("--anchor-ckpt", required=True)
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="runs/b3_qat_eval_paths")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    qat_ckpt = Path(args.qat_ckpt)
    anchor_ckpt = Path(args.anchor_ckpt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from b3_qat_module import B3QATLitModule

    init_json = next(p / "qat_init.json" for p in (qat_ckpt.parent, *qat_ckpt.parents) if (p / "qat_init.json").is_file())
    qat_init = json.loads(init_json.read_text())
    qat_scales = qat_init.get("qat_scales") or qat_init.get("qat_scales_init")
    if not qat_scales:
        raise KeyError(f"Missing qat_scales in {init_json}")
    learnable = bool(qat_init.get("learnable_scales", False))

    lit = B3QATLitModule.load_from_checkpoint(
        str(qat_ckpt), map_location="cpu", weights_only=False,
        qat_scales=qat_scales, calib_sessions=None, apply_equalization=False,
        learnable_scales=learnable,
    )
    split = load_split_manifest(anchor_ckpt.parent.parent / "split_manifest.json")
    hp = load_hyperparams_from_ckpt(anchor_ckpt)
    dm = build_loso_datamodule(exp_root, Path(args.data_dir), split.fold_id)
    student_anchor = load_student_from_ckpt(anchor_ckpt, exp_root)

    report = evaluate_four_paths(
        lit,
        student_anchor=student_anchor,
        val_ds=dm.val_heldin_dataset,
        heldout_session=split.heldout_session,
        behavior_scale=float(hp.get("behavior_scaling_factor", 5.0)),
    )
    (out / "eval_paths_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
