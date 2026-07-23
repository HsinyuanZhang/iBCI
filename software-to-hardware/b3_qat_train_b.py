#!/usr/bin/env python3
"""QAT-B: shared learnable scales (LSQ-style) on top of QAT-A weights."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import lightning as L
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from b3_ckpt_loader import load_b3_weights_from_ckpt, load_hyperparams_from_ckpt
from b3_eval_protocol import build_loso_datamodule, load_split_manifest, load_student_from_ckpt
from b3_ptq import apply_cross_layer_equalization, calibrate_scales_from_stats, collect_activation_stats
from b3_qat_eval_paths import evaluate_four_paths
from b3_qat_module import B3QATLitModule
from b3_qat_train import _collect_train_calibs, _resolve_teacher_ckpt, _scales_to_dict


def _load_qat_encoder_state(lit: B3QATLitModule, qat_ckpt: Path) -> None:
    ckpt = torch.load(qat_ckpt, map_location="cpu", weights_only=False)
    enc_sd = {
        k.replace("qat_encoder.", "", 1): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("qat_encoder.")
    }
    missing, unexpected = lit.qat_encoder.load_state_dict(enc_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys loading QAT encoder: {unexpected[:8]}")
    lit._shadow_weight_reference = {
        name: param.detach().clone()
        for name, param in lit._encoder_named_linears(lit.qat_encoder)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="B3 FP32 anchor checkpoint")
    parser.add_argument("--init-qat-ckpt", default=None, help="QAT-A checkpoint to warm-start weights/scales")
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="runs/b3_qat_b_loso0")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-weight", type=float, default=5e-6)
    parser.add_argument("--lr-scale", type=float, default=1e-5)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--scale-method", default="p9999")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    exp_root = Path(args.exp_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    split = load_split_manifest(ckpt.parent.parent / "split_manifest.json")
    hp = load_hyperparams_from_ckpt(ckpt)
    teacher_ckpt = _resolve_teacher_ckpt(hp, exp_root)

    dm = build_loso_datamodule(exp_root, Path(args.data_dir), split.fold_id)
    dm.hparams.batch_size = args.batch_size
    dm.setup("fit")

    calibs, calib_names = _collect_train_calibs(dm, split.train_sessions, seed=args.seed)
    weights_np = apply_cross_layer_equalization(load_b3_weights_from_ckpt(ckpt), calibs)
    stats = collect_activation_stats(weights_np, calibs)
    frozen = calibrate_scales_from_stats(stats, args.scale_method, source_sessions=calib_names)
    qat_scales = _scales_to_dict(frozen)

    if args.init_qat_ckpt:
        init_json = next(
            p / "qat_init.json" for p in (Path(args.init_qat_ckpt).parent, *Path(args.init_qat_ckpt).parents)
            if (p / "qat_init.json").is_file()
        )
        qat_scales = json.loads(init_json.read_text())["qat_scales"]

    lit = B3QATLitModule(
        task=hp.get("task", "m2"),
        teacher_ckpt_path=teacher_ckpt,
        init_student_ckpt_path=str(ckpt.resolve()),
        exp_root=str(exp_root.resolve()),
        qat_scales=qat_scales,
        calib_sessions=calibs,
        apply_equalization=not bool(args.init_qat_ckpt),
        learnable_scales=True,
        loss_mode="anchor",
        lr=args.lr_weight,
        lr_scale=args.lr_scale,
        behavior_scaling_factor=float(hp.get("behavior_scaling_factor", 5.0)),
        warmup_epochs=args.warmup_epochs,
    )

    if args.init_qat_ckpt:
        _load_qat_encoder_state(lit, Path(args.init_qat_ckpt))

    student_anchor = load_student_from_ckpt(ckpt, exp_root)
    behavior_scale = float(hp.get("behavior_scaling_factor", 5.0))

    init_report: Dict[str, Any] = {
        "phase": "QAT-B",
        "init_ckpt": str(ckpt.resolve()),
        "init_qat_ckpt": str(Path(args.init_qat_ckpt).resolve()) if args.init_qat_ckpt else None,
        "qat_scales_init": qat_scales,
        "lr_weight": args.lr_weight,
        "lr_scale": args.lr_scale,
        "learnable_scales": True,
    }
    (out / "qat_init.json").write_text(json.dumps(init_report, indent=2), encoding="utf-8")

    epoch_m1 = evaluate_four_paths(
        lit, student_anchor=student_anchor, val_ds=dm.val_heldin_dataset,
        heldout_session=split.heldout_session, behavior_scale=behavior_scale,
    )
    (out / "eval_epoch-1.json").write_text(json.dumps(epoch_m1, indent=2), encoding="utf-8")
    print("epoch=-1:", json.dumps({
        "r2_fake": epoch_m1["r2_fake_quant"],
        "shadow_delta": epoch_m1["delta_r2_shadow"],
        "E_exact": epoch_m1["fake_integer_E_exact"],
    }))

    L.seed_everything(args.seed, workers=True)

    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(out / "checkpoints"),
        filename="qat-b-{epoch:02d}-{val_integer_engine/r2_mean:.4f}",
        monitor="val_integer_engine/r2_mean",
        mode="max",
        save_top_k=3,
    )
    early_stop = EarlyStopping(monitor="val_integer_engine/r2_mean", patience=10, mode="max")

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=args.gradient_clip_val,
        default_root_dir=str(out),
        callbacks=[checkpoint_cb, early_stop],
        log_every_n_steps=10,
    )

    init_ckpt = out / "checkpoints" / "qat-b-epoch=-1-init.ckpt"
    init_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": lit.state_dict(), "hyper_parameters": dict(lit.hparams), "epoch": -1, "global_step": 0},
        init_ckpt,
    )

    trainer.fit(lit, datamodule=dm)

    init_report["best_ckpt"] = checkpoint_cb.best_model_path
    init_report["epoch_m1_eval"] = epoch_m1
    (out / "qat_init.json").write_text(json.dumps(init_report, indent=2), encoding="utf-8")
    print(f"QAT-B done. Best: {checkpoint_cb.best_model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
