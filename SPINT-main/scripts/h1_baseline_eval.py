"""Post-run terminal held-in evaluator for the isolated SPINT H1 baseline.

The evaluator receives an explicit checkpoint and never searches validation
metrics.  It accepts only a checkpoint whose Lightning metadata says that 50
epochs completed, loads only the 13 held-in minival recordings through the
H1-only datamodule, and writes one immutable receipt with per-session and
pooled variance-weighted R2.  Formal held-out/EvalAI paths fail closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.h1_baseline_datamodule import H1BaselineDataModule, h1_session_date
from src.models.falcon_module import FalconLitModule


EXPECTED_EPOCHS = 50
EXPECTED_VARIANTS = {
    "paper_lr_1e-5": "appendix_paper_reference",
    "released_code_lr_5e-5": "implementation_primary",
}
EXPECTED_LRS = {
    "paper_lr_1e-5": 1.0e-5,
    "released_code_lr_5e-5": 5.0e-5,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def module_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _variance_weighted_r2(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 7:
        raise ValueError(f"H1 R2 arrays must have shape [N,7], got {prediction.shape}/{target.shape}")
    denominator = float(np.square(target - target.mean(axis=0, keepdims=True)).sum())
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("H1 R2 denominator is non-positive/non-finite")
    score = 1.0 - float(np.square(prediction - target).sum()) / denominator
    if not np.isfinite(score):
        raise ValueError("H1 R2 is non-finite")
    return float(score)


class _FullSessionBatchSampler:
    """Session-homogeneous sampler that keeps the final short batch.

    The released training sampler intentionally drops incomplete batches.  A
    terminal evaluator must score every held-in minival window, so it uses the
    same session grouping/order but retains each remainder.
    """

    def __init__(self, dataset: Any, batch_size: int) -> None:
        self.session_to_indices: dict[str, list[int]] = {}
        for index, (session, _start) in enumerate(dataset.window_indices):
            self.session_to_indices.setdefault(str(session), []).append(index)
        self.batch_size = int(batch_size)

    def __iter__(self):
        for indices in self.session_to_indices.values():
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.session_to_indices.values())


def _validate_config(config_path: Path, expected_variant: str) -> dict[str, Any]:
    if any(token in str(config_path).lower() for token in ("held-out", "heldout", "evalai", "formal_test")):
        raise ValueError(f"H1 baseline evaluator refuses held-out/formal config path {config_path}")
    # A Hydra-written ``.hydra/config.yaml`` is structurally resolved but may
    # legitimately retain runtime-only nodes such as
    # ``${hydra:runtime.output_dir}`` under ``paths``/checkpoint ``dirpath``.
    # Outside a Hydra app there is no initialized HydraConfig, so resolving the
    # *entire* tree here makes a valid terminal config unreadable.  The frozen
    # scientific fields checked below are literal values; keep unrelated
    # runtime nodes opaque and validate every field used by this evaluator.
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    if not isinstance(config, dict):
        raise ValueError("resolved H1 baseline config must be a mapping")
    variant = str(config.get("protocol_variant", ""))
    if variant != expected_variant or variant not in EXPECTED_VARIANTS:
        raise ValueError(f"config variant {variant!r} does not match requested {expected_variant!r}")
    if config.get("protocol_id") != "spint_h1_baseline_reproduction_v1":
        raise ValueError("unexpected H1 baseline protocol id")
    if config.get("comparison_role") != EXPECTED_VARIANTS[expected_variant]:
        raise ValueError("H1 baseline comparison role does not match the frozen variant name")
    if config.get("seed") != 42:
        raise ValueError("H1 baseline terminal evaluator requires seed=42")
    if config.get("test") is not False or config.get("clean_teacher_mode") is not False:
        raise ValueError("H1 baseline terminal evaluator requires test=false and root clean_teacher_mode=false")
    data = config.get("data", {})
    model = config.get("model", {})
    trainer = config.get("trainer", {})
    if data.get("_target_") != "src.data.h1_baseline_datamodule.H1BaselineDataModule":
        raise ValueError("H1 baseline evaluator requires the isolated held-in-only datamodule")
    required_data = {
        "calibration_n_trials": 2,
        "random_calibration": True,
        "window_size": 700,
        "max_trial_length": 1024,
        "batch_size": 32,
        "num_workers": 0,
    }
    for key, expected in required_data.items():
        if data.get(key) != expected:
            raise ValueError(f"H1 baseline config {key}={data.get(key)!r}, expected {expected!r}")
    if model.get("clean_teacher") is not True or model.get("scheduler_monitor") != "val_heldin/r2_mean":
        raise ValueError("H1 baseline config must suppress unpopulated held-out metrics via clean_teacher model flag")
    if float(model.get("optimizer", {}).get("lr", -1.0)) != EXPECTED_LRS[expected_variant]:
        raise ValueError("H1 baseline optimizer LR does not match its frozen variant name")
    if trainer.get("max_epochs") != EXPECTED_EPOCHS or trainer.get("min_epochs") != EXPECTED_EPOCHS:
        raise ValueError("H1 baseline terminal evaluator requires fixed 50-epoch trainer")
    if trainer.get("deterministic") is not False:
        raise ValueError("H1 baseline terminal evaluator requires released trainer deterministic=false")
    callbacks = config.get("callbacks", {})
    fixed = callbacks.get("fixed_epoch50")
    if not isinstance(fixed, dict) or fixed.get("monitor") is not None or fixed.get("every_n_epochs") != EXPECTED_EPOCHS:
        raise ValueError("H1 baseline terminal evaluator requires fixed_epoch50 checkpoint with no metric monitor")
    if "early_stopping" in callbacks:
        raise ValueError("H1 baseline terminal evaluator rejects EarlyStopping")
    if config.get("optimized_metric") is not None:
        raise ValueError("H1 baseline terminal evaluator rejects validation-selected optimized_metric")
    return config


def _validate_terminal_checkpoint(checkpoint_path: Path, expected_variant: str) -> dict[str, Any]:
    if any(token in str(checkpoint_path).lower() for token in ("held-out", "heldout", "evalai", "formal_test")):
        raise ValueError(f"H1 baseline evaluator refuses held-out/formal checkpoint path {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("H1 baseline checkpoint must be a Lightning mapping")
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int) or epoch + 1 != EXPECTED_EPOCHS:
        raise ValueError(f"H1 baseline checkpoint metadata must report epoch=49 for 50 completed epochs, got {epoch!r}")
    global_step = checkpoint.get("global_step")
    if not isinstance(global_step, int) or global_step <= 0:
        raise ValueError(f"H1 baseline checkpoint has invalid global_step {global_step!r}")
    hyper = checkpoint.get("hyper_parameters")
    if not isinstance(hyper, dict) or hyper.get("task") != "h1":
        raise ValueError("H1 baseline checkpoint hyperparameters are not an H1 FalconLitModule")
    if hyper.get("clean_teacher") is not True:
        raise ValueError("H1 baseline checkpoint must use the held-in-only metric flag clean_teacher=true")
    expected_lr = EXPECTED_LRS[expected_variant]
    optimizer_states = checkpoint.get("optimizer_states")
    if not isinstance(optimizer_states, list) or not optimizer_states:
        raise ValueError("H1 baseline terminal checkpoint must carry optimizer_states for LR-arm binding")
    observed_lrs = {
        float(group["lr"])
        for state in optimizer_states
        if isinstance(state, dict)
        for group in state.get("param_groups", [])
        if isinstance(group, dict) and "lr" in group
    }
    if observed_lrs != {expected_lr}:
        raise ValueError(
            f"H1 baseline checkpoint optimizer LR {sorted(observed_lrs)} does not match "
            f"variant {expected_variant}={expected_lr}"
        )
    hyper_optimizer = hyper.get("optimizer")
    hyper_lr = None
    if hasattr(hyper_optimizer, "keywords") and isinstance(hyper_optimizer.keywords, dict):
        hyper_lr = hyper_optimizer.keywords.get("lr")
        if hyper_lr is not None and float(hyper_lr) != expected_lr:
            raise ValueError("H1 baseline checkpoint hyperparameter optimizer LR disagrees with variant")
    return {
        "epochs_completed": int(epoch + 1),
        "checkpoint_epoch_zero_based": int(epoch),
        "global_step": int(global_step),
        "hyperparameters": hyper,
        "optimizer_lr": expected_lr,
        "hyperparameter_optimizer_lr": None if hyper_lr is None else float(hyper_lr),
    }


def evaluate_terminal_checkpoint(
    *,
    data_dir: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    expected_variant: str,
    output_path: str | Path,
    device: str = "cpu",
    batch_size: int = 32,
) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable H1 terminal receipt {output}")
    checkpoint = Path(checkpoint_path).resolve()
    config = Path(config_path).resolve()
    config_tree = _validate_config(config, expected_variant)
    checkpoint_meta = _validate_terminal_checkpoint(checkpoint, expected_variant)
    if int(batch_size) != 32:
        raise ValueError("H1 baseline protocol fixes batch_size=32; do not override it at evaluation")

    module = H1BaselineDataModule(
        task="h1",
        data_dir=str(data_dir),
        heldin_session_names=tuple(config_tree["data"]["heldin_session_names"]),
        batch_size=int(config_tree["data"]["batch_size"]),
        window_size=int(config_tree["data"]["window_size"]),
        calibration_n_trials=int(config_tree["data"]["calibration_n_trials"]),
        random_calibration=True,
        smooth_calibration=False,
        max_trial_length=int(config_tree["data"]["max_trial_length"]),
        standardize_covariates=False,
        use_intertrials=True,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        remove_still_times=False,
        remove_calib_still_times=False,
        use_calib_active_segments=False,
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        pad_value=-1.0,
        num_workers=0,
        pin_memory=False,
    )
    module.setup("fit")
    manifest = module.input_manifest()
    if manifest.get("formal_heldout_opened") is not False:
        raise ValueError("H1 terminal evaluator manifest reports formal held-out access")
    if any("held-out" in row["path"] or "heldout" in row["path"] for row in manifest["files"]):
        raise ValueError("H1 terminal evaluator found held-out input path")

    model = FalconLitModule.load_from_checkpoint(
        str(checkpoint), map_location="cpu", weights_only=False
    )
    model.to(device)
    model.eval()
    if model.training:
        raise RuntimeError("H1 terminal evaluator failed to enter evaluation mode")
    state_before = module_state_sha256(model)
    full_sampler = _FullSessionBatchSampler(module.val_heldin_dataset, int(batch_size))
    predictions: dict[str, list[np.ndarray]] = {session: [] for session in full_sampler.session_to_indices}
    targets: dict[str, list[np.ndarray]] = {session: [] for session in full_sampler.session_to_indices}
    loader = DataLoader(
        module.val_heldin_dataset,
        batch_sampler=full_sampler,
        num_workers=0,
        pin_memory=False,
    )
    with torch.no_grad():
        for neural, behavior, calibration, session_names in loader:
            names = [str(value) for value in session_names]
            if len(set(names)) != 1:
                raise ValueError("H1 terminal evaluator requires session-homogeneous batches")
            session = names[0]
            neural = neural.to(device=device, dtype=torch.float32)
            calibration = calibration.to(device=device, dtype=torch.float32)
            output_value = model(neural, calib_trialized_neural_features=calibration)
            if model.hparams.decode_last_timestep_only:
                output_value = output_value[:, -1:, :]
                behavior = behavior[:, -1:, :]
            if model.hparams.predict_scaled_behavior:
                output_value = output_value / float(model.hparams.behavior_scaling_factor)
            predictions[session].append(output_value[:, -1, :].detach().cpu().numpy())
            targets[session].append(behavior[:, -1, :].detach().cpu().numpy())
    state_after = module_state_sha256(model)
    if state_before != state_after:
        raise RuntimeError("H1 terminal evaluator changed model state")

    per_session: dict[str, dict[str, Any]] = {}
    pooled_predictions = []
    pooled_targets = []
    for session in sorted(predictions):
        if not predictions[session]:
            raise RuntimeError(f"H1 terminal evaluator received no held-in minival windows for {session}")
        prediction = np.concatenate(predictions[session], axis=0)
        target = np.concatenate(targets[session], axis=0)
        per_session[session] = {
            "query_windows": int(target.shape[0]),
            "r2_variance_weighted": _variance_weighted_r2(prediction, target),
        }
        pooled_predictions.append(prediction)
        pooled_targets.append(target)
    pooled_prediction = np.concatenate(pooled_predictions, axis=0)
    pooled_target = np.concatenate(pooled_targets, axis=0)
    session_scores = np.asarray([row["r2_variance_weighted"] for row in per_session.values()], dtype=np.float64)
    day_predictions: dict[str, list[np.ndarray]] = {}
    day_targets: dict[str, list[np.ndarray]] = {}
    day_recordings: dict[str, list[str]] = {}
    for session in sorted(predictions):
        day = h1_session_date(session)
        day_predictions.setdefault(day, []).extend(predictions[session])
        day_targets.setdefault(day, []).extend(targets[session])
        day_recordings.setdefault(day, []).append(session)
    per_day: dict[str, dict[str, Any]] = {}
    for day in sorted(day_predictions):
        day_prediction = np.concatenate(day_predictions[day], axis=0)
        day_target = np.concatenate(day_targets[day], axis=0)
        per_day[day] = {
            "recordings": sorted(day_recordings[day]),
            "query_windows": int(day_target.shape[0]),
            "r2_variance_weighted": _variance_weighted_r2(day_prediction, day_target),
        }
    day_scores = np.asarray([row["r2_variance_weighted"] for row in per_day.values()], dtype=np.float64)
    receipt = {
        "schema": "spint_h1_baseline_terminal_heldin_eval_v1",
        "status": "PASS_H1_BASELINE_HELDIN_TERMINAL_EVAL",
        "protocol": "spint_h1_baseline_reproduction_v1",
        "protocol_variant": expected_variant,
        "comparison_role": EXPECTED_VARIANTS[expected_variant],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_path": str(config),
        "config_sha256": sha256_file(config),
        "input_manifest_sha256": canonical_sha256(manifest),
        "input_manifest": manifest,
        "epochs_completed": checkpoint_meta["epochs_completed"],
        "checkpoint_epoch_zero_based": checkpoint_meta["checkpoint_epoch_zero_based"],
        "global_step": checkpoint_meta["global_step"],
        "checkpoint_optimizer_lr": checkpoint_meta["optimizer_lr"],
        "checkpoint_hyperparameter_optimizer_lr": checkpoint_meta["hyperparameter_optimizer_lr"],
        "session_count": len(per_session),
        "per_session": per_session,
        "day_group_count": len(per_day),
        "per_day": per_day,
        "aggregate": {
            "equal_session_mean_r2": float(session_scores.mean()),
            "equal_session_std_r2_sample": float(session_scores.std(ddof=1)),
            "official_day_mean_r2": float(day_scores.mean()),
            # The released FALCON challenge evaluator calls np.std without a
            # ddof argument.  That population convention is the primary
            # comparable day-group dispersion; retain sample std only as a
            # clearly labelled secondary diagnostic.
            "official_day_std_r2_population": float(day_scores.std(ddof=0)),
            "official_day_std_r2_sample": float(day_scores.std(ddof=1)),
            "official_day_grouping": "falcon_challenge.evaluator.compute_metrics_regression convention: recordings sharing h1_session_date (YYYYMMDD) are concatenated before variance-weighted R2, then the six day R2 values are equally averaged; np.std uses ddof=0 population convention",
            "pooled_query_windows": int(pooled_target.shape[0]),
            "pooled_r2_variance_weighted": _variance_weighted_r2(pooled_prediction, pooled_target),
            "definition": "R2=1-SSE/sum((y-y_mean)^2), summed over seven velocity dimensions; official primary is equal mean and population std (np.std ddof=0) over six date-group R2 values after same-day recording concatenation; equal-session, sample std, and pooled values are secondary.",
        },
        "guards": {
            "formal_heldout_opened": False,
            "validation_epoch_selection": False,
            "optimizer_present": False,
            "backpropagation": False,
            "torch_no_grad": True,
            "model_training_mode_after_load": bool(model.training),
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "model_state_unchanged": True,
        },
        "packaging_boundary": {
            "spint_baseline_input": "neural_only_calibration_features",
            "dense_velocity": "target covariates used for held-in scoring only in this baseline evaluator",
            "afc4_future_boundary": "a first-2-trial neural+velocity descriptor would be supervised few-shot and must not be called GF-FSU; hidden evaluation labels remain unavailable",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.chmod(0o444)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True, choices=sorted(EXPECTED_VARIANTS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    receipt = evaluate_terminal_checkpoint(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        expected_variant=args.variant,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(Path(args.output).resolve()),
                "official_day_mean_r2": receipt["aggregate"]["official_day_mean_r2"],
                "official_day_std_r2_population": receipt["aggregate"]["official_day_std_r2_population"],
                "official_day_std_r2_sample": receipt["aggregate"]["official_day_std_r2_sample"],
                "equal_session_mean_r2": receipt["aggregate"]["equal_session_mean_r2"],
                "pooled_r2_variance_weighted": receipt["aggregate"]["pooled_r2_variance_weighted"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
