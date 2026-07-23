#!/usr/bin/env python3
"""Evaluate QAT checkpoint: anchor/shadow/fake-quant/integer engine on LOSO heldout."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from b3_ckpt_loader import load_hyperparams_from_ckpt
from b3_eval_protocol import (
    build_loso_datamodule,
    check_baseline_r2,
    collect_session_windows,
    get_full_calib_pool,
    load_split_manifest,
    load_student_from_ckpt,
    session_r2_fp32_encoder,
    session_r2_with_E,
)
from b3_hw_golden import B3Shapes
from b3_quant_engine import ABLATION_PRESETS, FrozenActivationScales, build_quant_engine_bundle, forward_quant_engine, identity_metrics


def _export_weights_from_qat_encoder(qat_encoder) -> tuple:
    from b3_hw_golden import B3Weights

    def _np(t):
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


def _scales_from_qat_encoder(qat_encoder, source_sessions: list[str]) -> FrozenActivationScales:
    return qat_encoder.to_frozen_scales(source_sessions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qat-ckpt", required=True, help="Lightning QAT checkpoint")
    parser.add_argument("--anchor-ckpt", required=True, help="FP32 anchor for baseline R²")
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--expected-heldout-r2", type=float, default=0.63024879)
    parser.add_argument("--out", default="runs/b3_qat_eval")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    qat_ckpt = Path(args.qat_ckpt)
    anchor_ckpt = Path(args.anchor_ckpt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.split_manifest) if args.split_manifest else anchor_ckpt.parent.parent / "split_manifest.json"
    split = load_split_manifest(manifest_path)
    hp = load_hyperparams_from_ckpt(anchor_ckpt)

    from b3_qat_module import B3QATLitModule

    def _find_qat_init(ckpt_path: Path) -> Path:
        for parent in (ckpt_path.parent, *ckpt_path.parents):
            cand = parent / "qat_init.json"
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"qat_init.json not found near {ckpt_path}")

    init_json = _find_qat_init(qat_ckpt)
    qat_init = json.loads(init_json.read_text()) if init_json.is_file() else {}
    qat_scales = qat_init.get("qat_scales")
    if not qat_scales:
        raise FileNotFoundError(f"Missing qat_scales in {init_json}")

    lit = B3QATLitModule.load_from_checkpoint(
        str(qat_ckpt),
        map_location="cpu",
        weights_only=False,
        qat_scales=qat_scales,
        calib_sessions=None,
        apply_equalization=False,
    )
    lit.eval()

    dm = build_loso_datamodule(exp_root, Path(args.data_dir), fold_id=split.fold_id)
    batch_size = int(getattr(dm.hparams, "batch_size", 32) or 32)
    student_anchor = load_student_from_ckpt(anchor_ckpt, exp_root)

    heldout = split.heldout_session
    val_ds = dm.val_heldin_dataset
    neural, behavior = collect_session_windows(val_ds, heldout, batch_size=batch_size)
    calib = get_full_calib_pool(val_ds, heldout)[:33].astype(np.float32)
    shapes = B3Shapes(T=calib.shape[1], D=64, W=50, N=calib.shape[2], M=33)
    behavior_scale = float(hp.get("behavior_scaling_factor", 5.0))

    r2_anchor = session_r2_fp32_encoder(student_anchor, neural, behavior, calib, behavior_scale=behavior_scale)
    baseline_check = check_baseline_r2(r2_anchor, args.expected_heldout_r2, session=heldout)

    calib_t = torch.from_numpy(calib).unsqueeze(0)
    with torch.no_grad():
        e_shadow = lit.qat_encoder.forward_fp32(calib_t).squeeze(0).cpu().numpy()
        e_fake = lit.qat_encoder.forward_batch(calib_t).squeeze(0).cpu().numpy()

    weights = _export_weights_from_qat_encoder(lit.qat_encoder)
    scales = _scales_from_qat_encoder(lit.qat_encoder, split.train_sessions)
    bundle = build_quant_engine_bundle(weights, shapes, scales, ABLATION_PRESETS["w8_a8_e8"])
    e_int = forward_quant_engine(calib, bundle)["E_dequant"]

    r2_shadow = session_r2_with_E(student_anchor, neural, behavior, e_shadow, behavior_scale=behavior_scale)
    r2_fake = session_r2_with_E(student_anchor, neural, behavior, e_fake, behavior_scale=behavior_scale)
    r2_int = session_r2_with_E(student_anchor, neural, behavior, e_int, behavior_scale=behavior_scale)
    e_align = identity_metrics(e_fake, e_int)

    report: Dict[str, Any] = {
        "qat_ckpt": str(qat_ckpt.resolve()),
        "anchor_ckpt": str(anchor_ckpt.resolve()),
        "heldout_session": heldout,
        "baseline_self_check": baseline_check,
        "r2_anchor_fp32": r2_anchor,
        "r2_shadow_fp32": r2_shadow,
        "r2_fake_quant": r2_fake,
        "r2_integer": r2_int,
        "delta_r2_shadow": r2_shadow - r2_anchor,
        "delta_r2_fake_quant": r2_fake - r2_anchor,
        "delta_r2_integer": r2_int - r2_anchor,
        "fake_vs_integer_E": e_align,
        "fake_integer_E_exact": e_align["max_abs"] == 0.0,
        "pass_fake_quant": (r2_fake - r2_anchor) >= -0.01,
        "pass_integer": (r2_int - r2_anchor) >= -0.01,
        "exported_scales": lit.qat_encoder.export_scales_dict(),
        "qat_init": qat_init,
    }
    (out / "qat_eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "B3 QAT evaluation (LOSO heldout)",
        f"anchor FP32 R²:  {r2_anchor:.8f} (check {baseline_check['pass']})",
        f"shadow FP32 R²:  {r2_shadow:.8f} (Δ {report['delta_r2_shadow']:.6f})",
        f"fake-quant R²:   {r2_fake:.8f} (Δ {report['delta_r2_fake_quant']:.6f})",
        f"integer R²:      {r2_int:.8f} (Δ {report['delta_r2_integer']:.6f})",
        f"fake vs int E max_abs: {e_align['max_abs']:.6g} exact={report['fake_integer_E_exact']}",
        f"pass integer: {report['pass_integer']}",
    ]
    print("\n".join(lines))
    return 0 if report["pass_integer"] and report["fake_integer_E_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
