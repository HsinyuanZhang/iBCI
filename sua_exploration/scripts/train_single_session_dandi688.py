"""P3 Step 0: single-session end-to-end upper-bound check on DANDI 000688.

Trains encoder + decoder END-TO-END (freeze_decoder=False, loss_mode="task_only")
on ONE session's train trials and evaluates on held-out trials of the SAME
session. This isolates pipeline correctness from cross-session generalization.
The MC_Maze teacher checkpoint is used only to define the decoder architecture
and warm-start weights; it imposes NO distillation constraint here.

Reference upper bound: POYO single-session CO R2 ~= 0.935.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.regression import R2Score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc_maze.multisession_datamodule import discover_nwb_files
from mc_maze.single_session_datamodule import Dandi688SingleSessionDataModule

_sce_root = Path(__file__).resolve().parents[2] / "streaming_calibration_exp"
sys.path.insert(0, str(_sce_root))
from src.models.streaming_calibration_module import StreamingCalibrationLitModule

DEFAULT_TEACHER = (
    Path(__file__).resolve().parents[1]
    / "checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
)
POYO_SINGLE_SESSION_CO_R2 = 0.935


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def configure_single_session_metrics(model, session_name: str) -> None:
    for attr in ("val_heldin_r2", "val_heldout_r2", "test_heldin_r2", "test_heldout_r2"):
        setattr(
            model,
            attr,
            nn.ModuleDict({session_name: R2Score(multioutput="variance_weighted")}),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", type=str, default=str(DEFAULT_TEACHER))
    parser.add_argument("--variant", type=str, default="B3", choices=["B3", "B15P", "B15D", "B15", "B16"])
    parser.add_argument(
        "--data_dir",
        type=str,
        default="sua_exploration/data/dandi_000688/sub-C",
    )
    parser.add_argument("--task", type=str, default="CO", choices=["CO", "RT"])
    parser.add_argument(
        "--nwb_path",
        type=str,
        default=None,
        help="Explicit session file. If omitted, use the earliest CO/RT session.",
    )
    parser.add_argument("--session_index", type=int, default=0)
    parser.add_argument("--out_name", type=str, default=None)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_decoder", action="store_true")
    parser.add_argument("--loss_mode", type=str, default="task_only")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    teacher_ckpt = Path(args.teacher_ckpt).expanduser().resolve()
    if not teacher_ckpt.is_file():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {teacher_ckpt}")

    if args.nwb_path:
        nwb_path = Path(args.nwb_path).expanduser().resolve()
    else:
        data_dir = Path(args.data_dir).expanduser().resolve()
        files = discover_nwb_files(data_dir, task=args.task)
        nwb_path = files[args.session_index].resolve()
    if not nwb_path.is_file():
        raise FileNotFoundError(f"Session NWB does not exist: {nwb_path}")

    session_stem = nwb_path.name.replace("_behavior+ecephys.nwb", "")
    out_name = args.out_name or f"{args.variant.lower()}_ss_{session_stem}"
    output_dir = Path(__file__).resolve().parents[1] / "checkpoints" / out_name
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata_path = output_dir / "run_metadata.json"

    dm = Dandi688SingleSessionDataModule(
        nwb_path=str(nwb_path),
        batch_size=32,
        window_size=50,
        calibration_n_trials=10,
        max_trial_length=100,
        bin_size_ms=20,
        num_workers=4,
        train_frac=args.train_frac,
        seed=args.seed,
    )
    dm.setup()

    run_metadata = {
        "schema_version": 1,
        "status": "initialized",
        "step": "P3_step0_single_session_upper_bound",
        "created_at": datetime.now().astimezone().isoformat(),
        "variant": args.variant,
        "seed": args.seed,
        "teacher_checkpoint": str(teacher_ckpt),
        "teacher_sha256": sha256_file(teacher_ckpt),
        "teacher_role": "architecture_and_warmstart_only_no_distillation",
        "session": session_stem,
        "nwb_path": str(nwb_path),
        "task": args.task,
        "train_frac": args.train_frac,
        "poyo_single_session_co_r2_reference": POYO_SINGLE_SESSION_CO_R2,
        "training": {
            "max_epochs": args.max_epochs,
            "learning_rate": args.lr,
            "batch_size": 32,
            "window_size": 50,
            "calibration_n_trials": 10,
            "trial_length": 100,
            "bin_size_ms": 20,
            "loss_mode": args.loss_mode,
            "freeze_decoder": args.freeze_decoder,
            "decode_last_timestep_only": True,
            "behavior_scaling_factor": 5.0,
            "deterministic": True,
        },
    }
    write_json(run_metadata_path, run_metadata)

    optimizer = partial(torch.optim.Adam, lr=args.lr, weight_decay=0.0)
    model = StreamingCalibrationLitModule(
        task="mc_maze",
        variant=args.variant,
        teacher_ckpt_path=str(teacher_ckpt),
        window_size=50,
        trial_length=100,
        id_hidden_dim=128,
        hidden_dim=64,
        pad_value=-1.0,
        freeze_decoder=args.freeze_decoder,
        loss_mode=args.loss_mode,
        lambda_y=1.0,
        lambda_E=0.1,
        decode_last_timestep_only=True,
        predict_scaled_behavior=True,
        behavior_scaling_factor=5.0,
        optimizer=optimizer,
        scheduler=None,
        compile=False,
    )
    model.setup("fit")
    configure_single_session_metrics(model, dm.session_name)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="best-{epoch:03d}-{val_heldin/r2_mean:.4f}",
        monitor="val_heldin/r2_mean",
        mode="max",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_heldin/r2_mean",
        mode="max",
        patience=15,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb],
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        default_root_dir=str(output_dir),
        deterministic=True,
    )

    trainer.fit(model, datamodule=dm)

    best_val_r2 = float(checkpoint_cb.best_model_score) if checkpoint_cb.best_model_score is not None else float("nan")
    run_metadata.update({
        "status": "completed",
        "completed_at": datetime.now().astimezone().isoformat(),
        "best_checkpoint": str(Path(checkpoint_cb.best_model_path).resolve()),
        "best_val_r2": best_val_r2,
        "reached_poyo_reference_fraction": (
            best_val_r2 / POYO_SINGLE_SESSION_CO_R2 if best_val_r2 == best_val_r2 else None
        ),
    })
    write_json(run_metadata_path, run_metadata)
    write_json(results_dir / f"p3_step0_{out_name}_seed{args.seed}.json", run_metadata)

    print(f"Session: {session_stem}")
    print(f"Best val R2 (single-session, held-out trials): {best_val_r2:.4f}")
    print(f"POYO single-session CO reference: {POYO_SINGLE_SESSION_CO_R2:.3f}")
    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
