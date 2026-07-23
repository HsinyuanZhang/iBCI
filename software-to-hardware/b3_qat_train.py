#!/usr/bin/env python3
"""Train B3 W8A8 QAT on LOSO fold=0 (QAT-A v2: p9999 scales, equalized weight ref)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import lightning as L

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from b3_ckpt_loader import load_b3_weights_from_ckpt, load_hyperparams_from_ckpt
from b3_eval_protocol import (
    build_loso_datamodule,
    load_split_manifest,
    load_student_from_ckpt,
)
from b3_ptq import apply_cross_layer_equalization, calibrate_scales_from_stats, collect_activation_stats
from b3_qat_eval_paths import evaluate_four_paths
from b3_qat_module import B3QATLitModule


def _collect_train_calibs(dm, train_sessions: List[str], seed: int = 42):
    from b3_eval_protocol import get_full_calib_pool, sample_calib_draw, stable_session_seed

    ds = dm.train_dataset
    calibs, names = [], []
    for sess in train_sessions:
        pool = get_full_calib_pool(ds, sess)
        _idx, calib = sample_calib_draw(pool, num_trials=33, seed=stable_session_seed(seed, sess))
        calibs.append(calib)
        names.append(sess)
    return calibs, names


def _scales_to_dict(scales) -> Dict[str, float]:
    return {
        "input": scales.input_scale_i8,
        "pre_out": scales.pre_out_scale_i8,
        "mean": scales.mean_scale_i8,
        "post0_out": scales.post0_out_scale_i8,
        "post1_out": scales.post1_out_scale_i8,
        "E": scales.E_int8_scale,
    }


def _resolve_teacher_ckpt(hp: Dict[str, Any], exp_root: Path) -> str:
    teacher_path = Path(hp.get("teacher_ckpt_path", ""))
    if teacher_path.is_file():
        return str(teacher_path.resolve())
    for cand in [
        exp_root / hp.get("teacher_ckpt_path", ""),
        exp_root.parent / "SPINT-main/logs/train/runs/2026-07-07-16-05-16/checkpoints/best_ckpt/epoch_034.ckpt",
    ]:
        if cand.is_file():
            return str(cand.resolve())
    raise FileNotFoundError("teacher checkpoint not found")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="B3 anchor student checkpoint")
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--out", default="runs/b3_qat_a_v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--equalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scale-method", default="p9999", choices=["max_abs", "p999", "p9999", "mse_opt"])
    parser.add_argument("--scale-mult", type=float, default=1.0)
    parser.add_argument("--learnable-scales", action="store_true")
    parser.add_argument("--lambda-y", type=float, default=0.75)
    parser.add_argument("--lambda-E", type=float, default=0.075)
    parser.add_argument("--lambda-weight", type=float, default=1e-4)
    parser.add_argument("--stability-steps", type=int, default=0, help="run N-step probe before training (0=skip)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    exp_root = Path(args.exp_root)
    data_dir = Path(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.split_manifest) if args.split_manifest else ckpt.parent.parent / "split_manifest.json"
    split = load_split_manifest(manifest_path)
    hp = load_hyperparams_from_ckpt(ckpt)
    teacher_ckpt = _resolve_teacher_ckpt(hp, exp_root)

    dm = build_loso_datamodule(exp_root, data_dir, fold_id=split.fold_id)
    dm.hparams.batch_size = args.batch_size
    dm.setup("fit")

    calibs, calib_names = _collect_train_calibs(dm, split.train_sessions, seed=args.seed)

    weights_np = load_b3_weights_from_ckpt(ckpt)
    if args.equalize:
        weights_np = apply_cross_layer_equalization(weights_np, calibs)
    stats = collect_activation_stats(weights_np, calibs)
    frozen = calibrate_scales_from_stats(
        stats, args.scale_method, mult=args.scale_mult, source_sessions=calib_names,
    )
    qat_scales = _scales_to_dict(frozen)

    lit = B3QATLitModule(
        task=hp.get("task", "m2"),
        teacher_ckpt_path=teacher_ckpt,
        init_student_ckpt_path=str(ckpt.resolve()),
        exp_root=str(exp_root.resolve()),
        qat_scales=qat_scales,
        calib_sessions=calibs if args.equalize else None,
        apply_equalization=args.equalize,
        learnable_scales=args.learnable_scales,
        loss_mode="anchor",
        lambda_y=args.lambda_y,
        lambda_E=args.lambda_E,
        lambda_weight=args.lambda_weight,
        behavior_scaling_factor=float(hp.get("behavior_scaling_factor", 5.0)),
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
    )

    student_anchor = load_student_from_ckpt(ckpt, exp_root)
    behavior_scale = float(hp.get("behavior_scaling_factor", 5.0))

    init_report = {
        "init_ckpt": str(ckpt.resolve()),
        "split": {
            "fold_id": split.fold_id,
            "train_sessions": split.train_sessions,
            "heldout_session": split.heldout_session,
        },
        "qat_scales": qat_scales,
        "equalize": args.equalize,
        "scale_method": args.scale_method,
        "scale_mult": args.scale_mult,
        "lr": args.lr,
        "loss_mode": "anchor",
        "weight_ref": "equalized_shadow_init",
        "weight_penalty": "normalized_relative_mse",
    }
    (out / "qat_init.json").write_text(json.dumps(init_report, indent=2), encoding="utf-8")

    epoch_m1 = evaluate_four_paths(
        lit,
        student_anchor=student_anchor,
        val_ds=dm.val_heldin_dataset,
        heldout_session=split.heldout_session,
        behavior_scale=behavior_scale,
    )
    (out / "eval_epoch-1.json").write_text(json.dumps(epoch_m1, indent=2), encoding="utf-8")
    print("epoch=-1 eval:", json.dumps({
        "r2_anchor": epoch_m1["r2_anchor_fp32"],
        "r2_shadow": epoch_m1["r2_shadow_fp32"],
        "r2_fake": epoch_m1["r2_fake_quant"],
        "r2_integer": epoch_m1["r2_integer"],
        "shadow_matches_anchor": epoch_m1["shadow_matches_anchor"],
        "E_exact": epoch_m1["fake_integer_E_exact"],
    }, indent=2))

    if args.stability_steps > 0:
        import subprocess
        cmd = [
            sys.executable, str(_HERE / "b3_qat_stability.py"),
            "--ckpt", str(ckpt), "--exp-root", str(exp_root), "--data-dir", str(data_dir),
            "--steps", str(args.stability_steps), "--lr", str(args.lr),
            "--scale-method", args.scale_method, "--out", str(out / "stability"),
        ]
        subprocess.run(cmd, check=False)

    L.seed_everything(args.seed, workers=True)

    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(out / "checkpoints"),
        filename="qat-{epoch:02d}-{val_integer_engine/r2_mean:.4f}",
        monitor="val_integer_engine/r2_mean",
        mode="max",
        save_top_k=3,
    )
    early_stop = EarlyStopping(monitor="val_integer_engine/r2_mean", patience=8, mode="max")

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=args.gradient_clip_val,
        default_root_dir=str(out),
        callbacks=[checkpoint_cb, early_stop],
        log_every_n_steps=10,
    )

    init_ckpt = out / "checkpoints" / "qat-epoch=-1-init.ckpt"
    init_ckpt.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(
        {
            "state_dict": lit.state_dict(),
            "hyper_parameters": dict(lit.hparams),
            "epoch": -1,
            "global_step": 0,
        },
        init_ckpt,
    )

    trainer.fit(lit, datamodule=dm)

    init_report["best_ckpt"] = checkpoint_cb.best_model_path
    init_report["epoch_m1_eval"] = epoch_m1
    (out / "qat_init.json").write_text(json.dumps(init_report, indent=2), encoding="utf-8")
    print(f"QAT training done. Best checkpoint: {checkpoint_cb.best_model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
