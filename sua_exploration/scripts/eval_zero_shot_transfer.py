"""P2: Zero-shot transfer evaluation.

Tests whether B3 encoder trained on FALCON M2 (MUA) can extract useful identity
from MC_Maze (SUA) calibration data, evaluated via the MC_Maze teacher decoder.

Usage:
    cd /home/xinyuan/Work_host/SPINT
    conda run -n spint python sua_exploration/scripts/eval_zero_shot_transfer.py \
        --teacher_ckpt <path_to_mc_maze_teacher.ckpt>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streaming_calibration_exp"))

import numpy as np
import torch
from torchmetrics.regression import R2Score

from mc_maze.datamodule import MCMazeDataModule
from src.models.components.spint import SpintModel
from src.models.falcon_module import FalconLitModule


def load_b3_encoder(b3_ckpt_path: str, device: torch.device):
    """Load B3 EarlyPool encoder from streaming calibration checkpoint."""
    ckpt = torch.load(b3_ckpt_path, map_location=device, weights_only=False)
    hparams = ckpt["hyper_parameters"]

    from src.models.components.streaming_encoders import build_encoder

    encoder = build_encoder(
        variant=hparams["variant"],
        window_size=hparams["window_size"],
        trial_length=hparams["trial_length"],
        hidden_dim=hparams["hidden_dim"],
        id_hidden_dim=hparams.get("id_hidden_dim", 128),
        pad_value=hparams.get("pad_value", -1.0),
    )

    # Extract encoder weights from student state dict
    student_state = {
        k.replace("student.id_encoder.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("student.id_encoder.")
    }
    encoder.load_state_dict(student_state)
    encoder.to(device).eval()
    return encoder


def extract_identity_teacher(teacher: SpintModel, calib: torch.Tensor) -> torch.Tensor:
    """Extract identity using teacher's built-in ID estimation.

    calib: [B, M, T, N]
    Returns: [B, N, W] identity embedding
    """
    with torch.no_grad():
        calib_perm = calib.permute(0, 1, 3, 2)  # [B, M, N, T]
        B, M, N, T = calib_perm.shape
        x = calib_perm.reshape(B * M * N, 1, T)
        x = teacher.fc_id_in(x)  # [B*M*N, 1, model_dim]
        x = x.reshape(B, M, N, -1)
        x = x.mean(dim=1)  # [B, N, model_dim]
        x = x.permute(0, 2, 1)  # [B, model_dim, N]
        identity = teacher.fc_id_out(x.permute(0, 2, 1).unsqueeze(1).squeeze(1))
        # Actually use the teacher's forward path properly
    # Simpler: just run teacher forward and intercept identity
    with torch.no_grad():
        src = calib[:, 0, :, :]  # dummy neural window [B, T, N] -> use first trial
        # Use teacher's identity extraction directly
        calib_perm = calib.permute(0, 1, 3, 2)  # [B, M, N, T]
        B, M, N, T = calib_perm.shape
        flat = calib_perm.reshape(B * M, N, T)
        id_in = teacher.fc_id_in(flat)  # [B*M, N, model_dim]
        id_in = id_in.reshape(B, M, N, -1).mean(dim=1)  # [B, N, model_dim]
        identity = teacher.fc_id_out(id_in)  # [B, N, W]
    return identity


def extract_identity_b3(encoder, calib: torch.Tensor) -> torch.Tensor:
    """Extract identity using B3 encoder.

    calib: [B, M, T, N]
    Returns: [B, N, W] identity embedding
    """
    with torch.no_grad():
        identity = encoder.forward_batch(calib)  # [B, N, W]
    return identity


def decode_with_identity(teacher: SpintModel, neural: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
    """Run teacher decoder with given identity (bypassing ID estimation).

    neural: [B, W, N]
    identity: [B, N, W]
    Returns: [B, W, C] behavior prediction
    """
    with torch.no_grad():
        src = neural.permute(0, 2, 1)  # [B, N, W]
        src = src + identity  # add identity
        src = teacher.fc_in(src)  # [B, N, model_dim]
        rep = teacher.fc_in(teacher.rep).to(src)  # [1, C, model_dim]
        transformer_output, _ = teacher.transformer(
            rep.repeat(src.size(0), 1, 1), src
        )
        output = teacher.fc_out(transformer_output)  # [B, C, W]
        behavior_pred = output.permute(0, 2, 1)  # [B, W, C]
    return behavior_pred


def evaluate(
    teacher: SpintModel,
    identity_fn,
    dm: MCMazeDataModule,
    behavior_scaling_factor: float,
    device: torch.device,
    max_batches: int = 100,
) -> float:
    """Evaluate R² with given identity extraction function."""
    r2 = R2Score(multioutput="variance_weighted").to(device)
    dl = dm.val_dataloader()[0]

    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        neural, behavior, calib, _ = batch
        neural, behavior, calib = neural.to(device), behavior.to(device), calib.to(device)

        identity = identity_fn(calib)
        pred = decode_with_identity(teacher, neural, identity)

        # Last timestep only, scaled
        pred_last = pred[:, -1:, :] / behavior_scaling_factor
        target_last = behavior[:, -1:, :]

        r2.update(pred_last.reshape(-1, 2), target_last.reshape(-1, 2))

    return r2.compute().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", type=str, required=True,
                        help="Path to MC_Maze teacher checkpoint")
    parser.add_argument("--b3_ckpt", type=str,
                        default="streaming_calibration_exp/outputs/streaming_calibration/b3_d64_anchor_s42_20260711_011020/checkpoints/best.ckpt",
                        help="Path to FALCON M2 B3 checkpoint")
    parser.add_argument("--data_dir", type=str,
                        default="sua_exploration/data/000128/sub-Jenkins")
    parser.add_argument("--behavior_scaling_factor", type=float, default=5.0)
    parser.add_argument("--max_batches", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
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

    # Load MC_Maze teacher
    print(f"Loading teacher from: {args.teacher_ckpt}")
    teacher_module = FalconLitModule.load_from_checkpoint(args.teacher_ckpt, weights_only=False)
    teacher = teacher_module.net.to(device).eval()

    # Load B3 encoder (trained on FALCON M2)
    print(f"Loading B3 from: {args.b3_ckpt}")
    b3_encoder = load_b3_encoder(args.b3_ckpt, device)

    # Evaluate: Teacher's own identity (upper bound)
    print("\n--- Evaluating: Teacher identity (upper bound) ---")
    r2_teacher = evaluate(
        teacher, lambda calib: extract_identity_teacher(teacher, calib), dm,
        args.behavior_scaling_factor, device, args.max_batches
    )
    print(f"Teacher identity R²: {r2_teacher:.4f}")

    # Evaluate: B3 zero-shot identity (transfer)
    print("\n--- Evaluating: B3 zero-shot identity (MUA->SUA transfer) ---")
    r2_b3 = evaluate(
        teacher, lambda calib: extract_identity_b3(b3_encoder, calib), dm,
        args.behavior_scaling_factor, device, args.max_batches
    )
    print(f"B3 zero-shot R²: {r2_b3:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("P2 Zero-Shot Transfer Results")
    print("=" * 60)
    print(f"Teacher (upper bound):  R² = {r2_teacher:.4f}")
    print(f"B3 zero-shot (M2→MC):  R² = {r2_b3:.4f}")
    if r2_teacher > 0:
        print(f"Transfer ratio:         {r2_b3/r2_teacher*100:.1f}%")
    print(f"\nConclusion: {'Transfer viable' if r2_b3 > 0.1 else 'Transfer NOT viable'}")


if __name__ == "__main__":
    main()
