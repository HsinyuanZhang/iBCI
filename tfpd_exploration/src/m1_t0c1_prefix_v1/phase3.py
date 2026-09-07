"""Phase-3 readout: the 2x2x2 {arm} x {deployment} x {surface} paired table.

Static deployment replays the exact Phase-1 forward (repeated per-row M10
calibration, batch 128, eval mode).  The CDM activity-FIFO deployment applies
the frozen ROLLING_FIXED_M causal selection law on completed trials through
the cached-identity path (algebraically the production identity path).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tfpd_exploration.src.cross_session_worst_group_v1 import source_physical as base_physical
from tfpd_exploration.src.cross_session_worst_group_v1 import source_reader
from tfpd_exploration.src.m1_h1_activity_headroom_v1 import core as headroom_core

from . import plan


class Phase3Error(RuntimeError):
    """Fail closed for the 2x2x2 readout."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase3Error(message)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


WINDOW = plan.MODEL_SHAPE["window"]
UNITS = plan.MODEL_SHAPE["units"]
OUTPUTS = plan.MODEL_SHAPE["outputs"]
BATCH = plan.EVAL_BATCH_SIZE
SUPPORT_TRIALS = 10


@dataclass(frozen=True)
class ArmBinding:
    arm: str
    root_relative: str
    checkpoint_sha256: str
    checkpoint_state_sha256: str
    terminal_sha256: str


def load_arm_binding(root: Path, arm: str) -> ArmBinding:
    """Descriptor-read the completed arm terminal/manifest bindings."""
    import json

    _require(arm in plan.ARMS, "phase3 arm drift")
    directory = Path(root).absolute() / plan.ARM_ROOT_RELATIVE[arm]
    _require(directory.is_dir(), f"phase3 arm root absent: {arm}")
    for name in ("checkpoint_manifest.json", "terminal.json"):
        body = (directory / name).read_bytes()
        digest = _sha(body)
        sidecar = (directory / f"{name}.sha256").read_text(encoding="ascii")
        _require(sidecar == f"{digest}  {name}\n", f"phase3 arm sidecar drift: {arm}/{name}")
    terminal_digest = _sha((directory / "terminal.json").read_bytes())
    _require(terminal_digest == plan.SEALED_ARM_TERMINAL_SHA256[arm],
             f"phase3 sealed arm predecessor terminal digest drift: {arm}")
    manifest = json.loads((directory / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    terminal = json.loads((directory / "terminal.json").read_text(encoding="utf-8"))
    _require(terminal.get("status") == plan.SEALED_ARM_STATUS
             and manifest.get("arm") == arm
             and manifest.get("swa_enabled") is False,
             f"phase3 arm terminal semantics drift: {arm}")
    best = manifest["checkpoints"]["best_source_train_loss"]
    return ArmBinding(
        arm=arm,
        root_relative=plan.ARM_ROOT_RELATIVE[arm],
        checkpoint_sha256=best["sha256"],
        checkpoint_state_sha256=best["state_sha256"],
        terminal_sha256=terminal_digest,
    )


def open_session_dataset(root: Path, source_root: Path, session_id: str) -> dict[str, Any]:
    """Parse one sealed session into the native dataset + trialized activity."""
    from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1

    metadata = v1.load_m1_metadata_manifest_authority(Path(root))
    rows = metadata.get("source_rows")
    row = next(item for item in rows if item.get("session_id") == session_id)
    relative = plan.safe_relative(row["relative_path"])
    target = Path(source_root) / relative
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    _require(digest == row["sha256"], f"phase3 session body digest drift: {session_id}")
    runtime = source_reader.load_native_m1_runtime(Path(root))
    recipe = source_reader.NATIVE_M1_READER_RECIPE
    datamodule = runtime.falcon_datamodule_type(**recipe.datamodule_kwargs(data_dir=Path(source_root)))
    record = datamodule.prepare_session_data(
        target, runtime.task,
        standardize_covariates=recipe.standardize_covariates,
        covariates_mean=None, covariates_std=None,
        use_intertrials=recipe.use_intertrials,
        include_trial_targets=False, include_trial_obj_ids=False,
    )
    dataset = runtime.falcon_dataset_type(
        sessions_dict={session_id: record}, calib_sessions_dict={session_id: record},
        **recipe.dataset_kwargs(),
    )
    trials = np.ascontiguousarray(
        dataset.calib_trialized_neural_features[session_id], dtype=np.float32)
    _require(trials.ndim == 3 and trials.shape[1:] == (1024, UNITS) and trials.shape[0] >= SUPPORT_TRIALS,
             f"phase3 trialized activity topology drift: {session_id}")
    return {"dataset": dataset, "trials": trials, "session_id": session_id}


def output_trial_indices(dataset: Any, session_id: str) -> tuple[int, ...]:
    starts = np.asarray(dataset.trial_start_indices[session_id], dtype=np.int64)
    result: list[int] = []
    for session, start in dataset.window_indices:
        _require(session == session_id, "phase3 query session drift")
        result.append(int(np.searchsorted(starts, int(start) + WINDOW - 1, side="right") - 1))
    _require(len(result) == len(dataset), "phase3 output-trial mapping drift")
    return tuple(result)


def score_static(model: Any, opened: Mapping[str, Any], *, device: str) -> dict[str, object]:
    """Phase-1 exact forward: repeated per-row M10 calibration, batch 128."""
    import torch

    dataset = opened["dataset"]
    session_id = opened["session_id"]
    support = np.ascontiguousarray(opened["trials"][:SUPPORT_TRIALS], dtype=np.float32)
    prediction = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    target = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    forwards = 0
    with torch.no_grad():
        for start_row in range(0, len(dataset), BATCH):
            rows = tuple(range(start_row, min(start_row + BATCH, len(dataset))))
            xs, ys = [], []
            for row in rows:
                _session, start = dataset.window_indices[row]
                end = int(start) + WINDOW
                xs.append(dataset.neural_data[session_id][int(start):end])
                ys.append(dataset.covariate_data[session_id][end - 1])
            x = np.ascontiguousarray(np.stack(xs), dtype=np.float32)
            calibration = np.ascontiguousarray(np.stack([support] * len(rows)), dtype=np.float32)
            output = model(
                torch.as_tensor(x, dtype=torch.float32, device=device),
                calib_trialized_neural_features=torch.as_tensor(
                    calibration, dtype=torch.float32, device=device),
            )
            prediction[np.asarray(rows, dtype=np.int64)] = np.ascontiguousarray(
                output[:, -1, :].detach().cpu().numpy(), dtype=np.float32)
            target[np.asarray(rows, dtype=np.int64)] = np.ascontiguousarray(
                np.stack(ys), dtype=np.float32)
            forwards += 1
    return _cell_result(prediction, target, {
        "forward_path": "phase1_repeated_B128_calibration", "forward_batches": forwards,
        "unique_activity_states": 1, "activity_cardinality": SUPPORT_TRIALS,
    })


def score_cdm_fifo(model: Any, opened: Mapping[str, Any], *, device: str) -> dict[str, object]:
    """CDM activity-FIFO: causal last-10-completed-trials identity."""
    import torch

    dataset = opened["dataset"]
    session_id = opened["session_id"]
    trials = opened["trials"]
    output_trials = output_trial_indices(dataset, session_id)
    selections = tuple(
        headroom_core.selection_for_output_trial(
            headroom_core.ActivityArm.ROLLING_FIXED_M,
            output_trial_index=trial, total_trials=int(trials.shape[0]),
            support_trials=SUPPORT_TRIALS,
        ) for trial in output_trials
    )
    prediction = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    target = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    forwards = 0
    with torch.no_grad():
        for selection, rows in headroom_core.grouped_indices(selections):
            identity = headroom_core.identity_from_raw_trials(
                model, trials, selection, family="m1", device=device)
            for offset in range(0, len(rows), BATCH):
                batch_rows = rows[offset:offset + BATCH]
                xs, ys = [], []
                for row in batch_rows:
                    _session, start = dataset.window_indices[row]
                    end = int(start) + WINDOW
                    xs.append(dataset.neural_data[session_id][int(start):end])
                    ys.append(dataset.covariate_data[session_id][end - 1])
                x = np.ascontiguousarray(np.stack(xs), dtype=np.float32)
                output = headroom_core.forward_with_cached_identity(model, x, identity)
                prediction[np.asarray(batch_rows, dtype=np.int64)] = np.ascontiguousarray(
                    output[:, -1, :].detach().cpu().numpy(), dtype=np.float32)
                target[np.asarray(batch_rows, dtype=np.int64)] = np.ascontiguousarray(
                    np.stack(ys), dtype=np.float32)
                forwards += 1
    return _cell_result(prediction, target, {
        "forward_path": "cdm_activity_fifo_cached_identity",
        "forward_batches": forwards,
        "unique_activity_states": len(set(selections)),
        "activity_cardinality_min": min(map(len, selections)),
        "activity_cardinality_max": max(map(len, selections)),
        "causal": True, "label_free": True,
    })


def _cell_result(prediction: np.ndarray, target: np.ndarray, extra: Mapping[str, object]) -> dict[str, object]:
    from tfpd_exploration.src.m1_heldin_heldout_gap_v1.physical import variance_weighted_last_bin_r2

    _require(prediction.shape == target.shape and prediction.shape[1] == OUTPUTS,
             "phase3 cell prediction topology drift")
    r2 = variance_weighted_last_bin_r2(prediction, target)
    return {
        "metric": plan.METRIC_LABEL,
        "last_bin_only": True,
        "eval_mode": True,
        "no_grad": True,
        "dynamic_dropout_disabled": True,
        "governing_r2": r2,
        "n_windows": int(prediction.shape[0]),
        "prediction_sha256": _array_digest(prediction),
        "target_sha256": _array_digest(target),
        "target_optimizer_steps": 0,
        "target_labels_used_only_for_metric": True,
        **dict(extra),
    }


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = plan.canonical_json_bytes({"dtype": str(array.dtype), "shape": list(array.shape)})
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def strict_load_arm_model(root: Path, binding: ArmBinding, *, device: str) -> tuple[Any, str]:
    """Strict reload of the arm's best checkpoint with the state digest pinned."""
    path = Path(root).absolute() / binding.root_relative / "checkpoint_best_source_train_loss.pt"
    captured: list[Any] = []

    def factory() -> Any:
        model = base_physical.load_exact_m1_spint_model(Path(root)).to(device)
        base_physical.materialize_exact_m1_model(model, device=device)
        captured.append(model)
        return model

    observed = base_physical.strict_reload_checkpoint_bytes(
        path.read_bytes(), expected_state_sha256=binding.checkpoint_state_sha256,
        model_factory=factory, device=device,
    )
    _require(len(captured) == 1 and observed == binding.checkpoint_state_sha256,
             f"phase3 strict arm checkpoint reload drift: {binding.arm}")
    model = captured[0]
    model.eval()
    _require(model.training is False, "phase3 arm model must be in eval mode")
    return model, observed


def build_table(cells: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Aggregate the 2x2x2 cells into equal-session arm means and deltas."""
    rows = []
    for arm in plan.ARMS:
        for deployment in plan.PHASE3_DEPLOYMENTS:
            for surface, sessions in (
                ("heldin_training", plan.HELDIN_TRAINING_SESSIONS),
                ("heldout_fold", plan.HELDOUT_FOLD_SESSIONS),
            ):
                values = {}
                for session_id in sessions:
                    key = f"{arm}_{deployment}_{session_id}"
                    _require(key in cells, f"phase3 table cell absent: {key}")
                    values[session_id] = float(cells[key]["governing_r2"])
                mean = float(np.mean([values[s] for s in sessions]))
                sd = float(np.std([values[s] for s in sessions]))
                rows.append({
                    "arm": arm, "deployment": deployment, "surface": surface,
                    "sessions": list(sessions), "equal_session_mean": mean,
                    "equal_session_sd": sd, "per_session": values,
                })
    def lookup(arm, deployment, surface):
        return next(row["equal_session_mean"] for row in rows
                    if row["arm"] == arm and row["deployment"] == deployment
                    and row["surface"] == surface)
    deltas = []
    for deployment in plan.PHASE3_DEPLOYMENTS:
        for surface in plan.PHASE3_SURFACES:
            delta_fifo_static = lookup("c1", deployment, surface) - lookup("t0", deployment, surface)
            deltas.append({
                "deployment": deployment, "surface": surface,
                "c1_minus_t0": delta_fifo_static,
            })
        for arm in plan.ARMS:
            deltas.append({
                "arm": arm, "deployment": deployment,
                "cdm_fifo_minus_static": {
                    "heldin_training": lookup(arm, "cdm_activity_fifo_m10", "heldin_training")
                    - lookup(arm, "static_m10", "heldin_training"),
                    "heldout_fold": lookup(arm, "cdm_activity_fifo_m10", "heldout_fold")
                    - lookup(arm, "static_m10", "heldout_fold"),
                },
            })
    return {
        "schema": "m1_t0c1_phase3_table_v1",
        "cell": plan.CELL,
        "phase": plan.PHASE,
        "arms": list(plan.ARMS),
        "deployments": list(plan.PHASE3_DEPLOYMENTS),
        "surfaces": list(plan.PHASE3_SURFACES),
        "rows": rows,
        "deltas": deltas,
        "phase1_motivation": plan.PHASE1_MOTIVATION,
        "cdm_fifo_law": plan.CDM_FIFO_LAW,
        "metric": plan.METRIC_LABEL,
        "formal_benchmark_verdict": False,
    }


__all__ = (
    "Phase3Error", "ArmBinding", "load_arm_binding", "open_session_dataset", "score_static",
    "score_cdm_fifo", "strict_load_arm_model", "build_table", "output_trial_indices",
)
