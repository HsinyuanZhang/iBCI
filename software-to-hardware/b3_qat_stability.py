#!/usr/bin/env python3
"""Short QAT stability probe: N train steps without full epoch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightning as L
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from b3_qat_eval_paths import evaluate_four_paths
from b3_qat_train import (
    _collect_train_calibs,
    _resolve_teacher_ckpt,
    _scales_to_dict,
)
from b3_ckpt_loader import load_b3_weights_from_ckpt, load_hyperparams_from_ckpt
from b3_eval_protocol import build_loso_datamodule, load_split_manifest, load_student_from_ckpt
from b3_ptq import apply_cross_layer_equalization, calibrate_scales_from_stats, collect_activation_stats
from b3_qat_module import B3QATLitModule


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--scale-method", default="p9999")
    parser.add_argument("--out", default="runs/b3_qat_stability")
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    exp_root = Path(args.exp_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    split = load_split_manifest(ckpt.parent.parent / "split_manifest.json")
    hp = load_hyperparams_from_ckpt(ckpt)
    dm = build_loso_datamodule(exp_root, Path(args.data_dir), split.fold_id)
    dm.setup("fit")
    calibs, calib_names = _collect_train_calibs(dm, split.train_sessions)

    weights_np = load_b3_weights_from_ckpt(ckpt)
    weights_np = apply_cross_layer_equalization(weights_np, calibs)
    stats = collect_activation_stats(weights_np, calibs)
    frozen = calibrate_scales_from_stats(stats, args.scale_method, source_sessions=calib_names)

    lit = B3QATLitModule(
        task=hp.get("task", "m2"),
        teacher_ckpt_path=_resolve_teacher_ckpt(hp, exp_root),
        init_student_ckpt_path=str(ckpt.resolve()),
        exp_root=str(exp_root.resolve()),
        qat_scales=_scales_to_dict(frozen),
        calib_sessions=calibs,
        apply_equalization=True,
        loss_mode="anchor",
        lr=args.lr,
    )
    student_anchor = load_student_from_ckpt(ckpt, exp_root)
    behavior_scale = float(hp.get("behavior_scaling_factor", 5.0))

    before = evaluate_four_paths(
        lit, student_anchor=student_anchor, val_ds=dm.val_heldin_dataset,
        heldout_session=split.heldout_session, behavior_scale=behavior_scale,
    )

    lit.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit.to(device)
    opt = torch.optim.Adam([p for p in lit.qat_encoder.parameters() if p.requires_grad], lr=args.lr)
    loader = dm.train_dataloader()
    it = iter(loader)
    step_logs = []
    for step in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        neural, behavior, calib, _ = batch
        neural = neural.to(device)
        behavior = behavior.to(device)
        calib = calib.to(device)
        opt.zero_grad()
        loss, components = lit._training_loss(neural, behavior, calib)
        loss.backward()
        grad_norm = sum(p.grad.norm(2).item() ** 2 for p in lit.qat_encoder.parameters() if p.grad is not None) ** 0.5
        torch.nn.utils.clip_grad_norm_(lit.qat_encoder.parameters(), 1.0)
        opt.step()
        if step % max(args.steps // 10, 1) == 0:
            step_logs.append({
                "step": step,
                "loss": float(loss.item()),
                "grad_norm": grad_norm,
                **{f"loss_{k}": float(v.item()) for k, v in components.items()},
            })

    lit.eval()
    lit.to("cpu")
    after = evaluate_four_paths(
        lit, student_anchor=student_anchor, val_ds=dm.val_heldin_dataset,
        heldout_session=split.heldout_session, behavior_scale=behavior_scale,
    )
    report = {"steps": args.steps, "lr": args.lr, "before": before, "after": after, "step_logs": step_logs}
    (out / "stability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "shadow_delta_before": before["delta_r2_shadow"],
        "shadow_delta_after": after["delta_r2_shadow"],
        "fake_delta_before": before["delta_r2_fake_quant"],
        "fake_delta_after": after["delta_r2_fake_quant"],
        "E_exact_after": after["fake_integer_E_exact"],
    }, indent=2))
    ok = (
        after["fake_integer_E_exact"]
        and after["delta_r2_shadow"] >= -0.005
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
