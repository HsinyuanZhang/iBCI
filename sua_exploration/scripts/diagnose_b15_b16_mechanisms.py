"""Run inference-only mechanism diagnostics for trained B15 and B16 encoders.

B15 is evaluated with full attention, diagonal-only self attention, and with
the attention output removed while retaining LayerNorm. B16 is evaluated with
its learned variance input and with that input set to zero.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streaming_calibration_exp"))

import torch
from torch import nn

from mc_maze.datamodule import MCMazeDataModule
from scripts.compare_neuronid_variants import (
    artifact_metadata,
    evaluate_variant,
    load_encoder,
    validate_fairness,
)
from src.models.falcon_module import FalconLitModule


class B15DiagnosticPath(nn.Module):
    def __init__(self, encoder: nn.Module, mode: str) -> None:
        super().__init__()
        if mode not in {"full", "self_only", "no_attention"}:
            raise ValueError(f"Unsupported B15 diagnostic mode: {mode}")
        self.encoder = encoder
        self.mode = mode

    def forward_batch(self, calib_trials: torch.Tensor) -> torch.Tensor:
        if self.mode == "full":
            return self.encoder.forward_batch(calib_trials)

        trials = calib_trials.permute(0, 1, 3, 2)
        mean_feat = self.encoder.pre_pool(trials).mean(dim=1)
        if self.mode == "self_only":
            num_neurons = mean_feat.shape[1]
            attention_mask = torch.full(
                (num_neurons, num_neurons),
                float("-inf"),
                device=mean_feat.device,
                dtype=mean_feat.dtype,
            )
            attention_mask.fill_diagonal_(0.0)
            attention_output, _ = self.encoder.cross_neuron_attn(
                mean_feat,
                mean_feat,
                mean_feat,
                attn_mask=attention_mask,
                need_weights=False,
            )
            mean_feat = self.encoder.attn_norm(mean_feat + attention_output)
        else:
            mean_feat = self.encoder.attn_norm(mean_feat)
        return self.encoder.post_pool(mean_feat)


class B16DiagnosticPath(nn.Module):
    def __init__(self, encoder: nn.Module, zero_variance: bool) -> None:
        super().__init__()
        self.encoder = encoder
        self.zero_variance = zero_variance

    def forward_batch(self, calib_trials: torch.Tensor) -> torch.Tensor:
        if not self.zero_variance:
            return self.encoder.forward_batch(calib_trials)

        trials = calib_trials.permute(0, 1, 3, 2)
        mean_feat = self.encoder.pre_pool(trials).mean(dim=1)
        combined = torch.cat([mean_feat, torch.zeros_like(mean_feat)], dim=-1)
        return self.encoder.post_pool(combined)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--b15_ckpt", required=True)
    parser.add_argument("--b16_ckpt", required=True)
    parser.add_argument(
        "--data_dir",
        default="sua_exploration/data/000128/sub-Jenkins",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=200,
        help="Number of validation batches per path; use 0 for the full loader.",
    )
    parser.add_argument(
        "--behavior_scaling_factor",
        type=float,
        default=5.0,
    )
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    max_batches = args.max_batches or None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_artifact = artifact_metadata(Path(args.teacher_ckpt))
    b15_artifact = artifact_metadata(Path(args.b15_ckpt))
    b16_artifact = artifact_metadata(Path(args.b16_ckpt))

    b15, b15_hparams = load_encoder(b15_artifact["path"], device)
    b16, b16_hparams = load_encoder(b16_artifact["path"], device)
    if b15_hparams["variant"] != "B15":
        raise ValueError(f"Expected B15 checkpoint, got {b15_hparams['variant']}")
    if b16_hparams["variant"] != "B16":
        raise ValueError(f"Expected B16 checkpoint, got {b16_hparams['variant']}")

    b15_fairness = validate_fairness(b15_hparams, teacher_artifact, None)
    b16_fairness = validate_fairness(
        b16_hparams,
        teacher_artifact,
        b15_fairness["fairness_hyperparameters"],
    )
    for name, hparams in (("B15", b15_hparams), ("B16", b16_hparams)):
        if hparams["behavior_scaling_factor"] != args.behavior_scaling_factor:
            raise ValueError(
                f"{name} behavior_scaling_factor does not match evaluation: "
                f"{hparams['behavior_scaling_factor']} != {args.behavior_scaling_factor}"
            )

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
    teacher_module = FalconLitModule.load_from_checkpoint(
        teacher_artifact["path"],
        weights_only=False,
    )
    teacher = teacher_module.net.to(device).eval()

    b15_paths = {
        mode: evaluate_variant(
            teacher,
            B15DiagnosticPath(b15, mode).to(device).eval(),
            dm,
            args.behavior_scaling_factor,
            device,
            max_batches,
        )
        for mode in ("full", "self_only", "no_attention")
    }
    b16_paths = {
        mode: evaluate_variant(
            teacher,
            B16DiagnosticPath(b16, zero_variance=(mode == "zero_variance"))
            .to(device)
            .eval(),
            dm,
            args.behavior_scaling_factor,
            device,
            max_batches,
        )
        for mode in ("full", "zero_variance")
    }

    created_at = datetime.now().astimezone()
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
    else:
        output_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / f"b15_b16_mechanisms_{created_at.strftime('%Y%m%d_%H%M%S')}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "created_at": created_at.isoformat(),
        "device": str(device),
        "evaluation": {
            "protocol": "mc_maze_internal_validation_inference_ablation",
            "data_dir": str(Path(args.data_dir).expanduser().resolve()),
            "max_batches": max_batches,
            "batch_size": 32,
            "calibration_n_trials": 10,
            "trial_length": 100,
            "bin_size_ms": 20,
            "behavior_scaling_factor": args.behavior_scaling_factor,
        },
        "teacher": teacher_artifact,
        "B15": {
            "checkpoint": b15_artifact,
            **b15_fairness,
            "paths": b15_paths,
            "cross_neuron_r2_gain": (
                b15_paths["full"]["r2"] - b15_paths["self_only"]["r2"]
            ),
            "attention_path_r2_gain": (
                b15_paths["full"]["r2"] - b15_paths["no_attention"]["r2"]
            ),
        },
        "B16": {
            "checkpoint": b16_artifact,
            **b16_fairness,
            "paths": b16_paths,
            "variance_r2_gain": (
                b16_paths["full"]["r2"] - b16_paths["zero_variance"]["r2"]
            ),
        },
    }
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for variant, paths in (("B15", b15_paths), ("B16", b16_paths)):
        print(variant)
        for mode, metrics in paths.items():
            print(
                f"  {mode:<14} r2={metrics['r2']:.6f} "
                f"id_mse={metrics['identity_norm_mse']:.6f}"
            )
    print(f"JSON report: {output_path}")


if __name__ == "__main__":
    main()
