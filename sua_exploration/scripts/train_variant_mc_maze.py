"""Train any encoder variant (B3/B15/B16) on MC_Maze with shared teacher.

Usage:
    cd /home/xinyuan/Work_host/SPINT
    conda run -n spint python sua_exploration/scripts/train_variant_mc_maze.py \
        --teacher_ckpt <path> --variant B15 --out_name b15_mc_maze
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from mc_maze.datamodule import MCMazeDataModule

_sce_root = Path(__file__).resolve().parents[2] / "streaming_calibration_exp"
sys.path.insert(0, str(_sce_root))
from src.models.streaming_calibration_module import StreamingCalibrationLitModule


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--variant", type=str, default="B15", choices=["B3", "B15", "B16"])
    parser.add_argument("--out_name", type=str, default=None)
    parser.add_argument("--data_dir", type=str,
                        default="sua_exploration/data/000128/sub-Jenkins")
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    teacher_ckpt = Path(args.teacher_ckpt).expanduser().resolve()
    if not teacher_ckpt.is_file():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {teacher_ckpt}")
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    out_name = args.out_name or f"{args.variant.lower()}_mc_maze"
    output_dir = Path(__file__).resolve().parents[1] / "checkpoints" / out_name
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata_path = output_dir / "run_metadata.json"

    run_metadata = {
        "schema_version": 1,
        "status": "initialized",
        "created_at": datetime.now().astimezone().isoformat(),
        "variant": args.variant,
        "seed": args.seed,
        "teacher_checkpoint": str(teacher_ckpt),
        "teacher_sha256": sha256_file(teacher_ckpt),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir.resolve()),
        "training": {
            "max_epochs": args.max_epochs,
            "learning_rate": args.lr,
            "batch_size": 32,
            "window_size": 50,
            "calibration_n_trials": 10,
            "trial_length": 100,
            "bin_size_ms": 20,
            "loss_mode": "task_plus_y_plus_E",
            "lambda_y": 1.0,
            "lambda_E": 0.1,
            "freeze_decoder": True,
            "decode_last_timestep_only": True,
            "behavior_scaling_factor": 5.0,
            "deterministic": True,
        },
    }
    write_json(run_metadata_path, run_metadata)

    dm = MCMazeDataModule(
        data_dir=str(data_dir),
        batch_size=32,
        window_size=50,
        calibration_n_trials=10,
        max_trial_length=100,
        bin_size_ms=20,
        num_workers=4,
    )
    dm.setup()

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
        freeze_decoder=True,
        loss_mode="task_plus_y_plus_E",
        lambda_y=1.0,
        lambda_E=0.1,
        decode_last_timestep_only=True,
        predict_scaled_behavior=True,
        behavior_scaling_factor=5.0,
        optimizer=optimizer,
        scheduler=None,
        compile=False,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="best-{epoch:03d}-{val_heldin/r2_mean:.4f}",
        monitor="val_heldin/r2_mean",
        mode="max",
        save_top_k=3,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_heldin/r2_mean",
        mode="max",
        patience=10,
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
    run_metadata.update({
        "status": "completed",
        "completed_at": datetime.now().astimezone().isoformat(),
        "best_checkpoint": str(Path(checkpoint_cb.best_model_path).resolve()),
        "best_checkpoint_sha256": sha256_file(Path(checkpoint_cb.best_model_path)),
        "best_checkpoint_validation_r2": float(checkpoint_cb.best_model_score),
    })
    write_json(run_metadata_path, run_metadata)
    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val R2: {checkpoint_cb.best_model_score:.4f}")
    print(f"Run metadata: {run_metadata_path}")


if __name__ == "__main__":
    main()
