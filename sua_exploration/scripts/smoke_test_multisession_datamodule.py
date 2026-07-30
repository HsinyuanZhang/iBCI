"""Smoke test: DANDI 000688 MultiSessionDataModule forward + backward on sub-J."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule

_sce_root = Path(__file__).resolve().parents[2] / "streaming_calibration_exp"
sys.path.insert(0, str(_sce_root))
from src.models.streaming_calibration_module import StreamingCalibrationLitModule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="sua_exploration/data/dandi_000688/sub-J",
    )
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--variant", type=str, default="B3")
    args = parser.parse_args()

    dm = Dandi688MultiSessionDataModule(
        data_dir=args.data_dir,
        task="CO",
        split_counts=(1, 1, 1),
        batch_size=4,
        num_workers=0,
    )
    dm.setup("fit")
    print("Session splits:", dm.session_splits)
    print("Train windows:", len(dm.train_dataset))
    print("Val windows:", len(dm.val_dataset))

    teacher_ckpt = Path(args.teacher_ckpt).expanduser().resolve()
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
        optimizer=torch.optim.Adam,
        scheduler=None,
        compile=False,
    )
    model.train()
    model.setup("fit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    neural, behavior, calib, session_names = batch
    neural = neural.to(device)
    behavior = behavior.to(device)
    calib = calib.to(device)
    print(
        "Batch shapes:",
        neural.shape,
        behavior.shape,
        calib.shape,
        "session",
        session_names[0],
        "N",
        neural.shape[-1],
    )
    assert len(set(session_names)) == 1, "batch must be single-session"

    out = model.model_step((neural, behavior, calib, session_names))
    loss = out["loss"]
    loss.backward()
    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    print(f"loss={loss.item():.4f} grad_norm={grad_norm:.4f}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
