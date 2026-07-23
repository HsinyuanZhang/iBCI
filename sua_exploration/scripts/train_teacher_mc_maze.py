"""Train SPINT teacher on MC_Maze sorted SUA data.

Usage:
    cd /home/xinyuan/Work_host/SPINT
    conda run -n spint python sua_exploration/scripts/train_teacher_mc_maze.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from mc_maze.datamodule import MCMazeDataModule
from src.models.components.spint import SpintModel
from src.models.falcon_module import FalconLitModule


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data" / "000128" / "sub-Jenkins"
    output_dir = Path(__file__).resolve().parents[1] / "checkpoints" / "teacher_mc_maze"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    net = SpintModel(
        model_dim=512,
        num_covariates=2,
        window_size=50,
        num_heads=64,
        num_layers=1,
        num_id_layers=3,
        use_learnable_id=True,
        learnable_id_type="mlp",
        learnable_rep=True,
        dropout_rate=0.0,
        dynamic_dropout=True,
        dynamic_dropout_low=0.0,
        dynamic_dropout_high=1.0,
        tf_drop_rate=0.1,
        readin_layer_type="mlp",
    )

    from functools import partial
    optimizer = partial(torch.optim.Adam, lr=5e-5, weight_decay=0.0)

    model = FalconLitModule(
        net=net,
        optimizer=optimizer,
        scheduler=None,
        compile=False,
        task="mc_maze",
        decode_last_timestep_only=True,
        predict_scaled_behavior=True,
        behavior_scaling_factor=5.0,
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
        patience=20,
    )

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb],
        check_val_every_n_epoch=2,
        log_every_n_steps=50,
        default_root_dir=str(output_dir),
    )

    trainer.fit(model, datamodule=dm)
    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val R2: {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
