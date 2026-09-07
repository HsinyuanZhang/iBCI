"""Physical frozen-checkpoint evaluator for the M2 activity-budget screen."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import core, plan


def _support_indices(dataset: Any, session: str, budget: int) -> np.ndarray:
    angles = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
    core.require(angles.size >= plan.ACTIVITY_HORIZON, f"{session} lacks first-30 target metadata")
    if budget != 4:
        selected = np.arange(budget, dtype=np.int64)
        core.require(int(np.isfinite(angles[selected]).sum()) >= 3,
                     f"{session} M{budget} has fewer than three directional trials")
        return selected
    from sua_exploration.mc_maze.d_optimal_calibration_design import (
        greedy_forward_d_optimal_indices,
    )

    candidates = np.flatnonzero(np.isfinite(angles[:plan.ACTIVITY_HORIZON])).astype(np.int64)
    core.require(candidates.size >= budget, f"{session} lacks four directional first-30 candidates")
    local = greedy_forward_d_optimal_indices(angles[candidates], budget)
    selected = np.sort(candidates[local]).astype(np.int64, copy=False)
    core.require(selected.size == budget and int(selected.max()) < plan.ACTIVITY_HORIZON,
                 f"{session} M4 D-opt selection drift")
    return np.ascontiguousarray(selected)


def _ridge_side(
    dataset: Any, session: str, selected: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    from tfpd_exploration.src.calibration_budget_comparators_v1 import fit_ridge_t4

    sums = np.asarray(dataset.calib_trial_spike_sums[session][selected], dtype=np.float64)
    lengths = np.asarray(dataset.calib_trial_lengths[session][selected], dtype=np.float64)
    angles = np.asarray(dataset.calib_trial_target_angles[session][selected], dtype=np.float64)
    usable = np.isfinite(angles)
    core.require(
        int(usable.sum()) >= 3,
        f"{session} M{int(selected.size)} has fewer than three directional trials",
    )
    rates = sums[usable] / lengths[usable, None]
    raw, evidence = fit_ridge_t4(
        rates,
        angles[usable],
        normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA,
    )
    mean = np.asarray(dataset.side_feature_mean, dtype=np.float32)
    std = np.asarray(dataset.side_feature_std, dtype=np.float32)
    core.require(mean.shape == (4,) and std.shape == (4,), "frozen T4 normalizer shape drift")
    side = np.ascontiguousarray((raw - mean) / std, dtype=np.float32)
    core.require(side.shape == (plan.CHANNELS, 4) and np.isfinite(side).all(), "normalized ridge T4 drift")
    return side, {
        **evidence,
        "budget": int(selected.size),
        "selection": "doptimal_noncentre_first30" if selected.size == 4 else "chronological_first_m",
        "selected_indices": selected.tolist(),
        "selected_indices_sha256": core.array_sha256(selected),
        "usable_directional_trials": int(usable.sum()),
        "raw_t4_sha256": core.array_sha256(raw),
        "normalized_t4_sha256": core.array_sha256(side),
    }


def _session_starts(dataset: Any, session: str, surface: str) -> np.ndarray:
    starts = np.asarray(
        [start for name, start in dataset.window_indices if name == session],
        dtype=np.int64,
    )
    core.require(starts.size > 0 and np.all(np.diff(starts) > 0), "query window order drift")
    if surface == "within_post30":
        return core.select_common_post30_window_starts(
            starts, dataset.trial_start_indices[session]
        )
    core.require(surface == "external_official_query", "unknown surface")
    return np.ascontiguousarray(starts)


def _score_session(
    *,
    torch: Any,
    student: Any,
    dataset: Any,
    session: str,
    surface: str,
    cell: core.CellSpec,
    device: Any,
    batch_size: int,
) -> dict[str, object]:
    calibration = np.asarray(dataset.calib_trialized_neural_features[session], dtype=np.float32)
    selected = _support_indices(dataset, session, cell.carrier_budget)
    activity = core.select_activity_rows(
        calibration,
        selected_indices=selected,
        activity_budget=cell.activity_budget,
    )
    side, ridge_evidence = _ridge_side(dataset, session, selected)
    support_tensor = torch.from_numpy(activity).unsqueeze(0).to(device)
    side_tensor = torch.from_numpy(side).unsqueeze(0).to(device)
    starts = _session_starts(dataset, session, surface)
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        identity = student.compute_identity(support_tensor, side_features=side_tensor)
        core.require(tuple(identity.shape) == (1, plan.CHANNELS, plan.WINDOW_SIZE), "identity shape drift")
        for offset in range(0, starts.size, batch_size):
            chunk_starts = starts[offset : offset + batch_size]
            neural_np = np.stack(
                [
                    dataset.neural_data[session][start : start + plan.WINDOW_SIZE]
                    for start in chunk_starts
                ],
                axis=0,
            ).astype(np.float32, copy=False)
            target_np = np.stack(
                [
                    dataset.covariate_data[session][start + plan.WINDOW_SIZE - 1]
                    for start in chunk_starts
                ],
                axis=0,
            ).astype(np.float32, copy=False)
            neural = torch.from_numpy(neural_np).to(device)
            prediction, _ = student(neural, identity=identity)
            prediction_np = (
                prediction[:, -1, :].detach().cpu().numpy().astype(np.float32)
                / plan.BEHAVIOR_SCALE
            )
            predictions.append(prediction_np)
            targets.append(target_np)
    target = np.ascontiguousarray(np.concatenate(targets, axis=0), dtype=np.float32)
    prediction = np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32)
    return {
        "session": session,
        "surface": surface,
        "cell": cell.name,
        "carrier_budget": cell.carrier_budget,
        "activity_budget": cell.activity_budget,
        "window_count": int(starts.size),
        "ordered_window_starts_sha256": core.array_sha256(starts),
        "activity_sha256": core.array_sha256(activity),
        "side_evidence": ridge_evidence,
        "target_sha256": core.array_sha256(target),
        "prediction_sha256": core.array_sha256(prediction),
        "r2": core.variance_weighted_r2(target, prediction),
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def execute(repo_root: Path, *, gpu_index: int = 1, batch_size: int = plan.DEFAULT_BATCH_SIZE) -> dict[str, object]:
    if isinstance(gpu_index, bool) or gpu_index < 0:
        raise core.ScreenError("gpu_index must be a non-negative integer")
    if isinstance(batch_size, bool) or batch_size < 1:
        raise core.ScreenError("batch_size must be a positive integer")
    expected_cvd = str(gpu_index)
    core.require(os.environ.get("CUDA_VISIBLE_DEVICES") == expected_cvd, "CUDA_VISIBLE_DEVICES mismatch")
    core.require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER mismatch")
    root = plan.result_root(repo_root)
    core.require(not root.exists(), f"result root already exists: {root}")

    import torch

    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data

    core.require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    started = time.monotonic()
    model, data_module, _task_config, metadata = load_frozen_model_and_data()
    core.require(metadata["checkpoint_sha256"] == plan.CHECKPOINT_SHA256, "checkpoint drift")
    core.require(metadata["normalization_sha256"] == plan.NORMALIZATION_SHA256, "normalizer drift")
    student = model.student.to(device).eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)

    datasets = {
        "within_post30": data_module.train_dataset,
        "external_official_query": data_module.val_heldout_dataset,
    }
    core.require(datasets["within_post30"] is not None, "within dataset missing")
    core.require(datasets["external_official_query"] is not None, "external dataset missing")
    expected_counts = {
        "within_post30": plan.EXPECTED_WITHIN_SESSIONS,
        "external_official_query": plan.EXPECTED_EXTERNAL_SESSIONS,
    }
    rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    contrasts: dict[str, dict[str, object]] = {}
    for surface, dataset in datasets.items():
        sessions = sorted(dataset.calib_trialized_neural_features)
        core.require(len(sessions) == expected_counts[surface], f"{surface} session count drift")
        surface_cells: dict[str, dict[str, float]] = {}
        for cell in core.CELL_SPECS:
            cell_values: dict[str, float] = {}
            for session in sessions:
                row = _score_session(
                    torch=torch,
                    student=student,
                    dataset=dataset,
                    session=session,
                    surface=surface,
                    cell=cell,
                    device=device,
                    batch_size=batch_size,
                )
                rows.append(row)
                cell_values[session] = float(row["r2"])
            surface_cells[cell.name] = cell_values
            summaries[f"{surface}:{cell.name}"] = core.summarize_sessions(cell_values)
        for budget in (10, 4):
            contrasts[f"{surface}:activity30_minus_static_m{budget}"] = core.paired_contrast(
                surface_cells[f"ridge_activity30_m{budget}"],
                surface_cells[f"ridge_static_m{budget}"],
            )

    payload = {
        "schema": plan.SCHEMA,
        "status": "TERMINAL",
        "scientific_role": "inference_only_activity_axis_screen__not_full_cdm",
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "normalization_sha256": metadata["normalization_sha256"],
        "gpu": {
            "cuda_visible_devices": expected_cvd,
            "logical_device": "cuda:0",
            "name": torch.cuda.get_device_name(0),
            "batch_size": batch_size,
        },
        "target_gradients": 0,
        "parameter_updates": 0,
        "elapsed_seconds": float(time.monotonic() - started),
        "cell_order": [cell.name for cell in core.CELL_SPECS],
        "rows": rows,
        "summaries": summaries,
        "paired_contrasts": contrasts,
    }
    _atomic_json(root / "score.json", payload)
    return payload
