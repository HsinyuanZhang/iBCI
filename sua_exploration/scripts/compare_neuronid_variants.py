"""Compare NeuronID variants (B3/B15/B16) on MC_Maze.

Evaluates task R², identity quality (normalized MSE vs teacher), and
representation similarity (cosine, Pearson) across variants.

Usage:
    cd /home/xinyuan/Work_host/SPINT
    conda run -n spint python sua_exploration/scripts/compare_neuronid_variants.py \
        --teacher_ckpt <path> --b3_ckpt <path> --b15_ckpt <path> --b16_ckpt <path>
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streaming_calibration_exp"))

import torch
from torchmetrics.regression import R2Score

from mc_maze.datamodule import MCMazeDataModule
from src.models.components.spint import SpintModel
from src.models.components.streaming_encoders import build_encoder
from src.models.falcon_module import FalconLitModule


FAIRNESS_HPARAMS = (
    "window_size",
    "trial_length",
    "id_hidden_dim",
    "hidden_dim",
    "pad_value",
    "freeze_decoder",
    "loss_mode",
    "lambda_y",
    "lambda_E",
    "decode_last_timestep_only",
    "predict_scaled_behavior",
    "behavior_scaling_factor",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_metadata(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }


def load_encoder(ckpt_path: str, device: torch.device):
    resolved_ckpt = Path(ckpt_path).expanduser().resolve()
    ckpt = torch.load(resolved_ckpt, map_location=device, weights_only=False)
    hparams = ckpt["hyper_parameters"]
    encoder = build_encoder(
        variant=hparams["variant"],
        window_size=hparams["window_size"],
        trial_length=hparams["trial_length"],
        hidden_dim=hparams["hidden_dim"],
        id_hidden_dim=hparams.get("id_hidden_dim", 128),
        pad_value=hparams.get("pad_value", -1.0),
    )
    student_state = {
        k.replace("student.id_encoder.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("student.id_encoder.")
    }
    encoder.load_state_dict(student_state)
    encoder.to(device).eval()
    return encoder, hparams


def validate_fairness(student_hparams: dict, teacher_artifact: dict, reference_hparams: dict | None) -> dict:
    recorded_teacher = student_hparams.get("teacher_ckpt_path")
    if not recorded_teacher:
        raise ValueError("Student checkpoint does not record teacher_ckpt_path")

    recorded_teacher_path = Path(recorded_teacher).expanduser().resolve()
    if not recorded_teacher_path.is_file():
        raise FileNotFoundError(
            "Cannot verify the student's recorded teacher checkpoint: "
            f"{recorded_teacher_path}"
        )
    recorded_teacher_sha256 = sha256_file(recorded_teacher_path)
    if recorded_teacher_sha256 != teacher_artifact["sha256"]:
        raise ValueError(
            "Student checkpoint was trained against a different teacher: "
            f"{recorded_teacher_path}"
        )

    signature = {key: student_hparams.get(key) for key in FAIRNESS_HPARAMS}
    missing = [key for key, value in signature.items() if value is None]
    if missing:
        raise ValueError(f"Student checkpoint is missing fairness hyperparameters: {missing}")
    if reference_hparams is not None and signature != reference_hparams:
        differences = {
            key: {"reference": reference_hparams[key], "student": signature[key]}
            for key in FAIRNESS_HPARAMS
            if signature[key] != reference_hparams[key]
        }
        raise ValueError(f"Student checkpoints do not share the same configuration: {differences}")

    return {
        "recorded_teacher_checkpoint": str(recorded_teacher_path),
        "recorded_teacher_sha256": recorded_teacher_sha256,
        "fairness_hyperparameters": signature,
    }


def extract_identity_teacher(teacher: SpintModel, calib: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        calib_perm = calib.permute(0, 1, 3, 2)  # [B, M, N, T]
        B, M, N, T = calib_perm.shape
        flat = calib_perm.reshape(B * M, N, T)
        id_in = teacher.fc_id_in(flat)
        id_in = id_in.reshape(B, M, N, -1).mean(dim=1)
        identity = teacher.fc_id_out(id_in)
    return identity


def decode_with_identity(teacher: SpintModel, neural: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        src = neural.permute(0, 2, 1) + identity
        src = teacher.fc_in(src)
        rep = teacher.fc_in(teacher.rep).to(src)
        transformer_output, _ = teacher.transformer(
            rep.repeat(src.size(0), 1, 1), src
        )
        output = teacher.fc_out(transformer_output)
    return output.permute(0, 2, 1)


def evaluate_variant(teacher, encoder, dm, behavior_scaling_factor, device, max_batches=None):
    r2 = R2Score(multioutput="variance_weighted").to(device)
    dl = dm.val_dataloader()[0]

    identity_stats = {
        "count": 0,
        "student_sum": 0.0,
        "teacher_sum": 0.0,
        "student_squared_sum": 0.0,
        "teacher_squared_sum": 0.0,
        "cross_sum": 0.0,
        "squared_error_sum": 0.0,
    }

    for i, batch in enumerate(dl):
        if max_batches is not None and i >= max_batches:
            break
        neural, behavior, calib, _ = batch
        neural, behavior, calib = neural.to(device), behavior.to(device), calib.to(device)

        with torch.no_grad():
            id_student = encoder.forward_batch(calib)
            id_teacher = extract_identity_teacher(teacher, calib)

        pred = decode_with_identity(teacher, neural, id_student)
        pred_last = pred[:, -1:, :] / behavior_scaling_factor
        target_last = behavior[:, -1:, :]
        r2.update(pred_last.reshape(-1, 2), target_last.reshape(-1, 2))

        id_student_double = id_student.double()
        id_teacher_double = id_teacher.double()
        identity_stats["count"] += id_student_double.numel()
        identity_stats["student_sum"] += id_student_double.sum().item()
        identity_stats["teacher_sum"] += id_teacher_double.sum().item()
        identity_stats["student_squared_sum"] += id_student_double.square().sum().item()
        identity_stats["teacher_squared_sum"] += id_teacher_double.square().sum().item()
        identity_stats["cross_sum"] += (id_student_double * id_teacher_double).sum().item()
        identity_stats["squared_error_sum"] += (
            id_student_double - id_teacher_double
        ).square().sum().item()

    r2_val = r2.compute().item()

    if identity_stats["count"] == 0:
        raise ValueError("Validation dataloader produced no batches")
    teacher_squared_sum = max(identity_stats["teacher_squared_sum"], 1e-12)
    norm_mse = identity_stats["squared_error_sum"] / teacher_squared_sum
    cosine_denom = max(
        (identity_stats["student_squared_sum"] * teacher_squared_sum) ** 0.5,
        1e-12,
    )
    cos_sim = max(-1.0, min(1.0, identity_stats["cross_sum"] / cosine_denom))

    count = identity_stats["count"]
    centered_cross = identity_stats["cross_sum"] - (
        identity_stats["student_sum"] * identity_stats["teacher_sum"] / count
    )
    centered_student_squared = max(
        identity_stats["student_squared_sum"]
        - identity_stats["student_sum"] ** 2 / count,
        0.0,
    )
    centered_teacher_squared = max(
        teacher_squared_sum - identity_stats["teacher_sum"] ** 2 / count,
        0.0,
    )
    pearson_denom = (centered_student_squared * centered_teacher_squared) ** 0.5
    pearson_r = centered_cross / pearson_denom if pearson_denom > 0.0 else 0.0

    return {
        "r2": r2_val,
        "identity_norm_mse": norm_mse,
        "identity_cosine": cos_sim,
        "identity_pearson": max(-1.0, min(1.0, pearson_r)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--b3_ckpt", type=str, required=True)
    parser.add_argument("--b15_ckpt", type=str, default=None)
    parser.add_argument("--b16_ckpt", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="sua_exploration/data/000128/sub-Jenkins")
    parser.add_argument("--behavior_scaling_factor", type=float, default=5.0)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Limit validation batches for exploratory runs; default evaluates all batches.",
    )
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    teacher_artifact = artifact_metadata(Path(args.teacher_ckpt))

    dm = MCMazeDataModule(
        data_dir=args.data_dir, batch_size=32, window_size=50,
        calibration_n_trials=10, max_trial_length=100, bin_size_ms=20, num_workers=4,
    )
    dm.setup()

    teacher_module = FalconLitModule.load_from_checkpoint(teacher_artifact["path"], weights_only=False)
    teacher = teacher_module.net.to(device).eval()

    variants = [("B3", args.b3_ckpt)]
    if args.b15_ckpt:
        variants.append(("B15", args.b15_ckpt))
    if args.b16_ckpt:
        variants.append(("B16", args.b16_ckpt))

    print(f"\n{'Variant':<8} {'R²':<8} {'ID MSE':<10} {'Cosine':<10} {'Pearson':<10}")
    print("-" * 50)

    results = {}
    reference_hparams = None
    for name, ckpt_path in variants:
        student_artifact = artifact_metadata(Path(ckpt_path))
        encoder, hparams = load_encoder(student_artifact["path"], device)
        variant_tag = hparams["variant"]
        if variant_tag != name:
            raise ValueError(f"Expected {name}, got {variant_tag}")
        fairness = validate_fairness(hparams, teacher_artifact, reference_hparams)
        if reference_hparams is None:
            reference_hparams = fairness["fairness_hyperparameters"]
        if hparams["behavior_scaling_factor"] != args.behavior_scaling_factor:
            raise ValueError(
                "Evaluation behavior_scaling_factor does not match the student checkpoint: "
                f"{args.behavior_scaling_factor} != {hparams['behavior_scaling_factor']}"
            )
        metrics = evaluate_variant(teacher, encoder, dm, args.behavior_scaling_factor, device, args.max_batches)
        results[name] = {
            "checkpoint": student_artifact,
            **fairness,
            "metrics": metrics,
        }
        print(f"{name:<8} {metrics['r2']:<8.4f} {metrics['identity_norm_mse']:<10.4f} {metrics['identity_cosine']:<10.4f} {metrics['identity_pearson']:<10.4f}")

    # Teacher upper bound
    print("-" * 50)
    r2_teacher = R2Score(multioutput="variance_weighted").to(device)
    dl = dm.val_dataloader()[0]
    for i, batch in enumerate(dl):
        if args.max_batches is not None and i >= args.max_batches:
            break
        neural, behavior, calib, _ = batch
        neural, behavior, calib = neural.to(device), behavior.to(device), calib.to(device)
        id_t = extract_identity_teacher(teacher, calib)
        pred = decode_with_identity(teacher, neural, id_t)
        pred_last = pred[:, -1:, :] / args.behavior_scaling_factor
        target_last = behavior[:, -1:, :]
        r2_teacher.update(pred_last.reshape(-1, 2), target_last.reshape(-1, 2))
    teacher_r2 = r2_teacher.compute().item()
    print(f"{'Teacher':<8} {teacher_r2:<8.4f} {'0.0000':<10} {'1.0000':<10} {'1.0000':<10}")

    created_at = datetime.now().astimezone()
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
    else:
        output_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / f"neuronid_comparison_{created_at.strftime('%Y%m%d_%H%M%S')}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "created_at": created_at.isoformat(),
        "device": str(device),
        "teacher": {
            "checkpoint": teacher_artifact,
            "metrics": {
                "r2": teacher_r2,
                "identity_norm_mse": 0.0,
                "identity_cosine": 1.0,
                "identity_pearson": 1.0,
            },
        },
        "evaluation": {
            "protocol": "mc_maze_internal_validation",
            "data_dir": str(Path(args.data_dir).expanduser().resolve()),
            "validation_loader_index": 0,
            "batch_size": 32,
            "window_size": 50,
            "calibration_n_trials": 10,
            "trial_length": 100,
            "bin_size_ms": 20,
            "behavior_scaling_factor": args.behavior_scaling_factor,
            "max_batches": args.max_batches,
        },
        "students": results,
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nJSON report: {output_path}")


if __name__ == "__main__":
    main()
