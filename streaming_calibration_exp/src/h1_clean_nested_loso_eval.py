"""Post-selection one-shot evaluator for H1 clean nested-LOSO.

The fit DataModule never exposes an outer loader.  This module first fits and
hashes the source-only fold plan, validates the selected checkpoint/monitor,
and only then opens the left-out held-in calibration support plus matching
minival query.  The core evaluator also accepts an in-memory predictor, which
keeps CPU tests independent of GPU checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.callbacks.h1_nested_selection import (
    build_h1_selection_contract,
    validate_h1_checkpoint_monitor,
)
from src.data.falcon_h1_afc4_features import H1_ARMS, H1_NUM_NEURONS, H1SessionRecord, index_h1_heldin_pairs, load_h1_record
from src.data.falcon_h1_clean_nested_loso_datamodule import (
    H1AFC4Dataset,
    H1CleanNestedLossoDataModule,
    _source_plan_sha256,
    build_h1_outer_target_dataset,
    h1_nested_loso_partition,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _variance_weighted_r2(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError(f"H1 outer R2 inputs must match [samples,7], got {pred.shape}/{truth.shape}")
    denominator = float(np.square(truth - truth.mean(axis=0, keepdims=True)).sum())
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("H1 outer R2 denominator is non-positive/non-finite")
    score = 1.0 - float(np.square(pred - truth).sum()) / denominator
    if not np.isfinite(score):
        raise ValueError("H1 outer R2 is non-finite")
    return float(score)


def evaluate_h1_outer_predictor(
    dataset: H1AFC4Dataset,
    predictor: Callable[..., torch.Tensor | np.ndarray],
    *,
    include_side_features: bool,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate a predictor on one post-selection outer minival dataset.

    ``predictor`` receives ``(neural, calib)`` for a teacher and
    ``(neural, calib, side_features)`` for a student.  It may return a full
    ``[B,W,7]`` sequence or only ``[B,1,7]``; the final timestep is the one
    scored by the official continuous-decoding objective.
    """

    if bool(dataset.include_side_features) != bool(include_side_features):
        raise ValueError("H1 evaluator dataset/predictor side-feature arity disagrees")
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    prediction_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    for batch in loader:
        if len(batch) == 5:
            neural, behavior, calib, _session_names, side = batch
        elif len(batch) == 4:
            neural, behavior, calib, _session_names = batch
            side = None
        else:
            raise ValueError(f"unexpected H1 outer batch arity {len(batch)}")
        neural = neural.to(device=device, dtype=torch.float32)
        behavior = behavior.to(device=device, dtype=torch.float32)
        calib = calib.to(device=device, dtype=torch.float32)
        if side is not None:
            side = side.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            output = predictor(neural, calib, side) if include_side_features else predictor(neural, calib)
        output = torch.as_tensor(output)
        if output.ndim == 2:
            output = output[:, None, :]
        if output.ndim != 3 or output.shape[0] != behavior.shape[0] or output.shape[-1] != behavior.shape[-1]:
            raise ValueError(f"H1 outer predictor output shape is invalid: {tuple(output.shape)}")
        prediction_chunks.append(output[:, -1, :].detach().cpu().numpy().astype(np.float64))
        target_chunks.append(behavior[:, -1, :].detach().cpu().numpy().astype(np.float64))
    if not prediction_chunks:
        raise RuntimeError("H1 outer evaluator received no query windows")
    prediction = np.concatenate(prediction_chunks, axis=0)
    target = np.concatenate(target_chunks, axis=0)
    score = _variance_weighted_r2(prediction, target)
    return {
        "r2_variance_weighted": score,
        "query_windows": int(target.shape[0]),
        "prediction_shape": list(prediction.shape),
        "target_shape": list(target.shape),
        "include_side_features": bool(include_side_features),
    }


def _module_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(str(name).encode("utf-8"))
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _load_checkpoint_predictor(
    checkpoint_path: str | Path,
    *,
    teacher_checkpoint_path: str | Path | None,
    include_side_features: bool,
    device: str,
) -> tuple[Callable[..., torch.Tensor], torch.nn.Module]:
    """Load either the standard teacher or streaming student checkpoint."""

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("H1 selected checkpoint must be a Lightning mapping")
    hyper = checkpoint.get("hyper_parameters", {})
    is_student = bool(isinstance(hyper, Mapping) and hyper.get("variant")) or any(
        str(key).startswith("student.") for key in checkpoint.get("state_dict", {})
    )
    if is_student:
        from src.models.streaming_calibration_module import StreamingCalibrationLitModule

        hyperparameters = dict(hyper) if isinstance(hyper, Mapping) else {}
        if teacher_checkpoint_path is not None:
            hyperparameters["teacher_ckpt_path"] = str(teacher_checkpoint_path)
        # StreamingCalibrationLitModule creates teacher/student networks in
        # ``setup``.  Lightning's normal load_from_checkpoint tries to apply
        # state_dict before setup, so every ``teacher.*``/``student.*`` key is
        # reported unexpected.  Reconstruct from saved hyperparameters,
        # initialize first, then perform one strict state load.
        model = StreamingCalibrationLitModule(**hyperparameters)
        model.setup("predict")
        state_result = model.load_state_dict(checkpoint["state_dict"], strict=True)
        if state_result.missing_keys or state_result.unexpected_keys:
            raise RuntimeError(
                f"H1 student checkpoint strict state load failed: missing={state_result.missing_keys}, "
                f"unexpected={state_result.unexpected_keys}"
            )
        model.to(device)
        model.eval()
        if model.student is None:
            raise RuntimeError("H1 student checkpoint did not initialize student network")

        def _student(neural: torch.Tensor, calib: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
            output, _identity = model.student(neural, calib_trials=calib, side_features=side)
            output, _target = model._slice_last_timestep(output, output)
            return output

        return _student, model

    from src.models.falcon_module import FalconLitModule

    model = FalconLitModule.load_from_checkpoint(str(checkpoint_path), map_location="cpu", weights_only=False)
    model.to(device)
    model.eval()
    scale = float(model.hparams.behavior_scaling_factor) if bool(model.hparams.predict_scaled_behavior) else 1.0

    def _teacher(neural: torch.Tensor, calib: torch.Tensor) -> torch.Tensor:
        output = model(neural, calib_trialized_neural_features=calib)
        return output[:, -1:, :] / scale

    return _teacher, model


def evaluate_h1_outer_checkpoint(
    *,
    data_dir: str | Path,
    outer_loso_fold: int,
    checkpoint_path: str | Path,
    arm: str,
    teacher_checkpoint_path: str | Path | None = None,
    expected_teacher_checkpoint_sha256: str | None = None,
    expected_source_plan_sha256: str | None = None,
    expected_monitor: str | None = None,
    selection_manifest_path: str | Path | None = None,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run exactly one selected checkpoint on one left-out outer target."""

    canonical_arm = str(arm).lower()
    is_teacher = canonical_arm == "teacher"
    if not is_teacher and canonical_arm not in H1_ARMS:
        raise ValueError(f"unsupported H1 outer arm {arm!r}")
    if selection_manifest_path is None:
        raise ValueError("H1 outer evaluator requires the immutable fit manifest")
    if not is_teacher and teacher_checkpoint_path is None:
        raise ValueError("H1 student outer evaluation requires an explicit selected teacher checkpoint")
    pairs = index_h1_heldin_pairs(data_dir)
    split = h1_nested_loso_partition(tuple(pairs), int(outer_loso_fold))
    source_records = {
        name: load_h1_record(pairs[name][0], split="calib") for name in split.inner_train_sessions
    }
    # Build the source plan without constructing a Lightning DataModule.  The
    # private hash helper is shared with the fit manifest and is deterministic.
    from src.data.falcon_h1_afc4_features import H1AFC4SourcePlan

    plan = H1AFC4SourcePlan(source_records, shuffle_seed=42)
    plan_sha = _source_plan_sha256(plan, excluded_outer_target=split.outer_target_session)
    if expected_source_plan_sha256 is not None and str(expected_source_plan_sha256) != plan_sha:
        raise ValueError("H1 outer evaluator source-plan SHA disagrees with selected fit manifest")
    selection_manifest: dict[str, Any] = json.loads(Path(selection_manifest_path).read_text(encoding="utf-8"))
    if selection_manifest.get("schema") != "h1_afc4_clean_nested_loso_split_manifest_v1":
        raise ValueError("H1 outer evaluator selection manifest schema is not the fit manifest")
    if int(selection_manifest.get("outer_loso_fold", -1)) != int(outer_loso_fold):
        raise ValueError("H1 outer evaluator selection manifest fold mismatch")
    if selection_manifest.get("outer_target_session") != split.outer_target_session:
        raise ValueError("H1 outer evaluator selection manifest target mismatch")
    if selection_manifest.get("outer_target_loaded_during_fit") is not False or selection_manifest.get(
        "outer_target_query_labels_read_during_fit"
    ) is not False:
        raise ValueError("H1 outer evaluator selection manifest records outer-target fit access")
    if selection_manifest.get("source_plan_sha256") != plan_sha:
        raise ValueError("H1 outer evaluator source-plan SHA disagrees with fit manifest")
    expected_loaded = set(split.inner_train_sessions) | {split.inner_validation_session}
    if set(selection_manifest.get("loaded_fit_sessions", ())) != expected_loaded:
        raise ValueError("H1 outer evaluator fit manifest loaded-session set disagrees with nested split")
    source_manifest = selection_manifest.get("source_only_basis_or_normalizer", {})
    if source_manifest.get("outer_target_used_for_basis_or_normalizer") is not False:
        raise ValueError("H1 outer evaluator fit manifest permits outer target in source transforms")
    manifest_monitor = selection_manifest.get("checkpoint_selection", {}).get("monitor")
    expected_monitor = manifest_monitor if expected_monitor is None else expected_monitor
    if not is_teacher and selection_manifest.get("arm") != canonical_arm:
        raise ValueError("H1 student outer evaluator arm disagrees with fit manifest")
    manifest_teacher_sha = selection_manifest.get("teacher_checkpoint_sha256")
    if not is_teacher and not manifest_teacher_sha:
        raise ValueError("H1 student fit manifest lacks selected teacher checkpoint SHA")
    checkpoint_sha = sha256_file(checkpoint_path)
    observed_teacher_sha = None
    if teacher_checkpoint_path is not None:
        observed_teacher_sha = sha256_file(teacher_checkpoint_path)
        if expected_teacher_checkpoint_sha256 is not None and observed_teacher_sha != str(expected_teacher_checkpoint_sha256):
            raise ValueError("H1 outer evaluator teacher checkpoint SHA disagrees with binding")
        if not is_teacher and observed_teacher_sha != str(manifest_teacher_sha):
            raise ValueError("H1 outer evaluator teacher checkpoint SHA disagrees with fit manifest")
    contract = build_h1_selection_contract(split, arm=canonical_arm, role="teacher" if is_teacher else canonical_arm)
    if expected_monitor is not None:
        validate_h1_checkpoint_monitor(
            expected_monitor,
            inner_validation_session=split.inner_validation_session,
            outer_target_session=split.outer_target_session,
        )
    if selection_manifest is not None:
        if selection_manifest.get("checkpoint_selection", {}).get("outer_target_used") is not False:
            raise ValueError("H1 outer evaluator fit manifest permits outer-target checkpoint selection")
    # Post-selection boundary: the target calibration/minival pair is opened
    # only after plan/checkpoint/monitor checks above.
    dataset, _split, _support, _query = build_h1_outer_target_dataset(
        data_dir=data_dir,
        outer_loso_fold=int(outer_loso_fold),
        source_records=source_records,
        side_feature_group="zero4" if is_teacher else canonical_arm,
        side_feature_shuffle_seed=42,
        window_size=700,
        max_trial_length=1024,
        pad_value=-1.0,
        include_side_features=not is_teacher,
    )
    predictor, model = _load_checkpoint_predictor(
        checkpoint_path,
        teacher_checkpoint_path=teacher_checkpoint_path,
        include_side_features=not is_teacher,
        device=device,
    )
    state_sha_before = _module_state_sha256(model)
    metrics = evaluate_h1_outer_predictor(
        dataset,
        predictor,
        include_side_features=not is_teacher,
        batch_size=batch_size,
        device=device,
    )
    state_sha_after = _module_state_sha256(model)
    model_state_unchanged = state_sha_before == state_sha_after
    if not model_state_unchanged:
        raise RuntimeError("H1 outer evaluator changed model state")
    return {
        "schema": "h1_afc4_clean_nested_loso_outer_eval_v1",
        "status": "PASS_OUTER_ONE_SHOT",
        "outer_loso_fold": int(outer_loso_fold),
        "outer_target_session": split.outer_target_session,
        "outer_target_loaded_after_selection": True,
        "outer_target_used_for_basis_or_normalizer": False,
        "outer_target_used_for_checkpoint_selection": False,
        "arm": canonical_arm,
        "checkpoint_sha256": checkpoint_sha,
        "teacher_checkpoint_sha256": observed_teacher_sha,
        "source_plan_sha256": plan_sha,
        "selection_contract": contract.as_dict(),
        "selection_manifest_path": None if selection_manifest_path is None else str(Path(selection_manifest_path).resolve()),
        "checkpoint_selection_binding": {
            "fit_manifest_selection_scope": "inner_validation_session_only",
            "passed_checkpoint_assumed_selected_best": True,
            "caller_must_resolve_best_checkpoint_path": True,
            "outer_target_used_for_checkpoint_selection": False,
        },
        "h1_hardware_cost_contract": {
            "num_neurons": H1_NUM_NEURONS,
            "window_size": 700,
            "calibration_n_trials": 1,
            "generic_train_py_cost_profile_num_neurons": 96,
            "generic_profile_used_for_h1": False,
        },
        "evaluation_guards": {
            "torch_no_grad": True,
            "backpropagation": False,
            "optimizer_present": False,
            "training_mode": bool(model.training),
            "model_state_sha256_before": state_sha_before,
            "model_state_sha256_after": state_sha_after,
            "model_state_unchanged": model_state_unchanged,
        },
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--outer-loso-fold", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--expected-teacher-sha256")
    parser.add_argument("--expected-source-plan-sha256")
    parser.add_argument("--expected-monitor")
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = evaluate_h1_outer_checkpoint(
        data_dir=args.data_dir,
        outer_loso_fold=args.outer_loso_fold,
        checkpoint_path=args.checkpoint,
        arm=args.arm,
        teacher_checkpoint_path=args.teacher_checkpoint,
        expected_teacher_checkpoint_sha256=args.expected_teacher_sha256,
        expected_source_plan_sha256=args.expected_source_plan_sha256,
        expected_monitor=args.expected_monitor,
        selection_manifest_path=args.selection_manifest,
        batch_size=args.batch_size,
        device=args.device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "output": str(output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
