"""P0: Train B3 student on MC_Maze with MC_Maze teacher.

Usage:
    cd /home/xinyuan/Work_host/SPINT
    conda run -n spint python sua_exploration/scripts/train_b3_mc_maze.py \
        --teacher_ckpt <path_to_mc_maze_teacher.ckpt>
"""
import argparse
import importlib.util
import sys
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str,
                        default="sua_exploration/data/000128/sub-Jenkins")
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parents[1] / "checkpoints" / "b3_mc_maze"
    output_dir.mkdir(parents=True, exist_ok=True)

    dm = MCMazeDataModule(
        data_dir=args.data_dir,
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
        variant="B3",
        teacher_ckpt_path=args.teacher_ckpt,
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
    )

    trainer.fit(model, datamodule=dm)
    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val R2: {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
