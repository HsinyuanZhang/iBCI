"""Physical frozen-weight executor for the M2 same-query comparator matrix."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from . import core, plan


_PARENT_CELL_MAP = {
    "t4_ridge_static_m30": "ridge_static_m30",
    "t4_ridge_static_m10": "ridge_static_m10",
    "t4_ridge_activity30_m10": "ridge_activity30_m10",
    "t4_ridge_static_m4": "ridge_static_m4",
    "t4_ridge_activity30_m4": "ridge_activity30_m4",
}


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _support_indices(dataset: Any, session: str, budget: int) -> np.ndarray:
    if budget != 4:
        return core.chronological_indices(budget)
    angles = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
    core.require(angles.size >= plan.ACTIVITY_HORIZON, f"{session}: fewer than 30 target labels")
    from sua_exploration.mc_maze.d_optimal_calibration_design import (
        greedy_forward_d_optimal_indices,
    )

    candidates = np.flatnonzero(np.isfinite(angles[: plan.ACTIVITY_HORIZON])).astype(np.int64)
    core.require(candidates.size >= 4, f"{session}: fewer than four finite first-30 directions")
    local = greedy_forward_d_optimal_indices(angles[candidates], 4)
    selected = np.sort(candidates[local]).astype(np.int64, copy=False)
    core.require(selected.size == 4 and np.unique(selected).size == 4, "D-opt M4 cardinality drift")
    return np.ascontiguousarray(selected)


def _session_starts(dataset: Any, session: str, surface: str) -> np.ndarray:
    starts = np.asarray([start for name, start in dataset.window_indices if name == session], dtype=np.int64)
    core.require(starts.size > 0 and np.all(np.diff(starts) > 0), f"{session}: query order drift")
    if surface == "within_post30":
        from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.core import (
            select_common_post30_window_starts,
        )

        starts = select_common_post30_window_starts(starts, dataset.trial_start_indices[session])
    else:
        core.require(surface == "external_official_query", "unknown surface")
    return np.ascontiguousarray(starts, dtype=np.int64)


def _query_arrays(dataset: Any, session: str, surface: str) -> tuple[np.ndarray, np.ndarray]:
    starts = _session_starts(dataset, session, surface)
    targets = np.ascontiguousarray(
        np.stack(
            [dataset.covariate_data[session][start + plan.WINDOW_SIZE - 1] for start in starts],
            axis=0,
        ),
        dtype=np.float32,
    )
    core.require(targets.shape == (starts.size, plan.OUTPUT_DIM) and np.isfinite(targets).all(),
                 f"{session}: target shape/nonfinite drift")
    return starts, targets


def _activity(dataset: Any, session: str, indices: Sequence[int]) -> np.ndarray:
    calibration = np.asarray(dataset.calib_trialized_neural_features[session], dtype=np.float32)
    selected = np.asarray(indices, dtype=np.int64)
    core.require(calibration.ndim == 3 and calibration.shape[1:] == (plan.TRIAL_LENGTH, plan.CHANNELS),
                 f"{session}: B3S calibration shape drift")
    core.require(int(selected.min()) >= 0 and int(selected.max()) < calibration.shape[0],
                 f"{session}: B3S selection outside calibration")
    result = np.ascontiguousarray(calibration[selected], dtype=np.float32)
    core.require(np.isfinite(result).all(), f"{session}: B3S activity is nonfinite")
    return result


def _ols_side(dataset: Any, session: str, selected: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    from streaming_calibration_exp.src.data.falcon_t4_features import t4_from_trial_sums

    sums = np.asarray(dataset.calib_trial_spike_sums[session][selected], dtype=np.float64)
    lengths = np.asarray(dataset.calib_trial_lengths[session][selected], dtype=np.float64)
    angles = np.asarray(dataset.calib_trial_target_angles[session][selected], dtype=np.float64)
    raw = t4_from_trial_sums(sums, lengths, angles, source=f"{session} selected-M OLS")
    mean = np.asarray(dataset.side_feature_mean, dtype=np.float32)
    std = np.asarray(dataset.side_feature_std, dtype=np.float32)
    core.require(mean.shape == std.shape == (4,) and np.all(std > 0), "T4 normalizer drift")
    side = np.ascontiguousarray((raw - mean) / std, dtype=np.float32)
    core.require(side.shape == (plan.CHANNELS, 4) and np.isfinite(side).all(), "OLS T4 drift")
    return side, {
        "estimator": "production_ols_cosine",
        "budget": int(selected.size),
        "selected_indices": selected.tolist(),
        "selected_indices_sha256": core.array_sha256(selected),
        "finite_direction_trials": int(np.isfinite(angles).sum()),
        "raw_t4_sha256": core.array_sha256(raw),
        "normalized_t4_sha256": core.array_sha256(side),
    }


def _mean_side(selected: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    side = np.zeros((plan.CHANNELS, 4), dtype=np.float32)
    return side, {
        "estimator": "normalized_zero_equals_frozen_train_population_mean",
        "budget": int(selected.size),
        "selected_indices": selected.tolist(),
        "selected_indices_sha256": core.array_sha256(selected),
        "normalized_t4_sha256": core.array_sha256(side),
        "raw_zero_claim": False,
    }


def _teacher_identity(torch: Any, teacher: Any, activity_tensor: Any) -> Any:
    values = activity_tensor.permute(0, 1, 3, 2)
    encoded = teacher.fc_id_in(values)
    pooled = encoded.mean(dim=1)
    identity = teacher.fc_id_out(pooled)
    core.require(tuple(identity.shape) == (1, plan.CHANNELS, plan.WINDOW_SIZE),
                 "SPINT cached identity shape drift")
    return identity


def _manual_teacher_decode(teacher: Any, neural: Any, identity: Any) -> Any:
    source = neural.permute(0, 2, 1) + identity
    source = teacher.fc_in(source)
    query = teacher.fc_in(teacher.rep).to(source)
    transformed, _ = teacher.transformer(query.repeat(source.shape[0], 1, 1), source)
    return teacher.fc_out(transformed).permute(0, 2, 1)


def _network_row(
    *,
    torch: Any,
    cell: core.CellSpec,
    student: Any,
    teacher: Any,
    dataset: Any,
    session: str,
    surface: str,
    starts: np.ndarray,
    target: np.ndarray,
    device: Any,
    batch_size: int,
) -> dict[str, object]:
    matched = _support_indices(dataset, session, cell.budget)
    selected = core.chronological_indices(cell.budget) if cell.support_selection == "chronological" else matched
    activity = _activity(dataset, session, selected)
    support = torch.from_numpy(activity).unsqueeze(0).to(device)
    side_evidence: dict[str, object] | None = None
    with torch.inference_mode():
        if cell.family == "spint":
            identity = _teacher_identity(torch, teacher, support)
        else:
            if cell.family == "t4_ols":
                side, side_evidence = _ols_side(dataset, session, selected)
            else:
                core.require(cell.family == "t4_mean", "unexpected network family")
                side, side_evidence = _mean_side(selected)
            identity = student.compute_identity(
                support, side_features=torch.from_numpy(side).unsqueeze(0).to(device)
            )
            core.require(tuple(identity.shape) == (1, plan.CHANNELS, plan.WINDOW_SIZE),
                         "T4 cached identity shape drift")

        predictions: list[np.ndarray] = []
        parity: float | None = None
        for offset in range(0, starts.size, batch_size):
            chunk_starts = starts[offset : offset + batch_size]
            neural_np = np.stack(
                [dataset.neural_data[session][start : start + plan.WINDOW_SIZE] for start in chunk_starts],
                axis=0,
            ).astype(np.float32, copy=False)
            neural = torch.from_numpy(neural_np).to(device)
            if cell.family == "spint":
                prediction = _manual_teacher_decode(teacher, neural, identity)
                if parity is None:
                    direct = teacher(neural, calib_trialized_neural_features=support)
                    parity = float((direct - prediction).abs().max().item())
                    core.require(parity == 0.0, "SPINT cached-identity reduction is not bit-exact")
            else:
                prediction, _ = student(neural, identity=identity)
            predictions.append(
                np.ascontiguousarray(
                    prediction[:, -1, :].detach().cpu().numpy().astype(np.float32)
                    / plan.BEHAVIOR_SCALE,
                    dtype=np.float32,
                )
            )
    predicted = np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32)
    core.require(predicted.shape == target.shape, "network prediction shape drift")
    return {
        "session": session,
        "surface": surface,
        "cell": cell.name,
        "family": cell.family,
        "budget": cell.budget,
        "support_selection": cell.support_selection,
        "supervision": cell.supervision,
        "source": "new_forward",
        "window_count": int(starts.size),
        "ordered_window_starts_sha256": core.array_sha256(starts),
        "target_sha256": core.array_sha256(target),
        "activity_sha256": core.array_sha256(activity),
        "selected_indices": selected.tolist(),
        "selected_indices_sha256": core.array_sha256(selected),
        "side_evidence": side_evidence,
        "cached_identity_direct_max_abs": parity,
        "prediction_sha256": core.array_sha256(predicted),
        "r2": core.variance_weighted_r2(target, predicted),
        "parameter_updates": 0,
    }


def _dense_support(
    dataset: Any, session: str, selected: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    neural = np.asarray(dataset.calib_neural[session], dtype=np.float32)
    behavior = np.asarray(dataset.calib_covariates[session], dtype=np.float32)
    starts = np.asarray(dataset.calib_trial_start_indices[session], dtype=np.int64)
    core.require(neural.shape[0] == behavior.shape[0], f"{session}: calibration axes disagree")
    lengths = np.asarray(
        [
            (int(starts[index + 1]) if int(index) + 1 < starts.size else int(neural.shape[0]))
            - int(starts[index])
            for index in selected
        ],
        dtype=np.int64,
    )
    eligible = np.ascontiguousarray(selected[lengths >= plan.WINDOW_SIZE], dtype=np.int64)
    short = np.ascontiguousarray(selected[lengths < plan.WINDOW_SIZE], dtype=np.int64)
    endpoints = core.dense_support_target_bins(starts, neural.shape[0], selected)
    expected_rows = int(np.maximum(lengths - plan.WINDOW_SIZE + 1, 0).sum())
    core.require(endpoints.size == expected_rows, f"{session}: dense support row count drift")
    features = core.materialize_windows(neural, endpoints)
    targets = np.ascontiguousarray(behavior[endpoints], dtype=np.float32)
    core.require(targets.shape == (endpoints.size, plan.OUTPUT_DIM) and np.isfinite(targets).all(),
                 f"{session}: dense support targets invalid")
    evidence = {
        "selected_trial_lengths": lengths.tolist(),
        "selected_trial_lengths_sha256": core.array_sha256(lengths),
        "dense_eligible_selected_indices": eligible.tolist(),
        "dense_eligible_selected_count": int(eligible.size),
        "short_selected_indices": short.tolist(),
        "short_selected_count": int(short.size),
        "short_trial_policy": "zero_dense_rows_no_padding_no_cross_trial_history",
    }
    return features, targets, endpoints, evidence


def _direct_ridge_row(
    *,
    torch: Any,
    cell: core.CellSpec,
    dataset: Any,
    session: str,
    surface: str,
    starts: np.ndarray,
    target: np.ndarray,
    device: Any,
    batch_size: int,
) -> dict[str, object]:
    from sua_exploration.mc_maze.native_m2_m24_ridge_w50 import (
        compile_raw_ridge,
        fit_dual_ridge_w50,
    )

    selected = _support_indices(dataset, session, cell.budget)
    support_features, support_targets, endpoints, support_evidence = _dense_support(
        dataset, session, selected
    )
    readout = fit_dual_ridge_w50(
        support_features,
        support_targets,
        normalized_lambda=plan.DIRECT_RIDGE_NORMALIZED_LAMBDA,
        device="cuda:0",
    )
    compiled = compile_raw_ridge(readout)
    weight = torch.from_numpy(compiled.raw_weights).to(device)
    intercept = torch.from_numpy(compiled.intercept).to(device)
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, starts.size, batch_size):
            chunk_starts = starts[offset : offset + batch_size]
            target_bins = chunk_starts + plan.WINDOW_SIZE - 1
            features = core.materialize_windows(dataset.neural_data[session], target_bins)
            value = torch.from_numpy(features).to(device) @ weight + intercept
            predictions.append(value.detach().cpu().numpy().astype(np.float32))
    predicted = np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32)
    return {
        "session": session,
        "surface": surface,
        "cell": cell.name,
        "family": cell.family,
        "budget": cell.budget,
        "support_selection": cell.support_selection,
        "supervision": cell.supervision,
        "source": "new_closed_form_fit",
        "window_count": int(starts.size),
        "ordered_window_starts_sha256": core.array_sha256(starts),
        "target_sha256": core.array_sha256(target),
        "selected_indices": selected.tolist(),
        "selected_indices_sha256": core.array_sha256(selected),
        "dense_support_rows": int(endpoints.size),
        "dense_support_endpoints_sha256": core.array_sha256(endpoints),
        "dense_velocity_coordinates_used": int(endpoints.size * plan.OUTPUT_DIM),
        "dense_support_evidence": support_evidence,
        "normalized_lambda": plan.DIRECT_RIDGE_NORMALIZED_LAMBDA,
        "prediction_sha256": core.array_sha256(predicted),
        "r2": core.variance_weighted_r2(target, predicted),
        "parameter_updates": 0,
    }


def _population_vector_row(
    *,
    cell: core.CellSpec,
    dataset: Any,
    session: str,
    surface: str,
    starts: np.ndarray,
    target: np.ndarray,
) -> dict[str, object]:
    from sua_exploration.mc_maze.population_vector_comparator import (
        build_pv_readout_from_trial_data,
        predict_pv_from_readout,
    )

    selected = _support_indices(dataset, session, cell.budget)
    _features, _support_targets, endpoints, support_evidence = _dense_support(
        dataset, session, selected
    )
    neural = np.asarray(dataset.calib_neural[session], dtype=np.float32)
    behavior = np.asarray(dataset.calib_covariates[session], dtype=np.float32)
    trial_starts = np.asarray(dataset.calib_trial_start_indices[session], dtype=np.int64)
    angles = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
    labeled_bounds: list[tuple[int, int]] = []
    labeled_angles: list[float] = []
    for index in selected:
        angle = float(angles[index])
        if not np.isfinite(angle):
            continue
        start = int(trial_starts[index])
        stop = int(trial_starts[index + 1]) if int(index) + 1 < trial_starts.size else int(neural.shape[0])
        labeled_bounds.append((start, stop))
        labeled_angles.append(angle)
    core.require(len(labeled_angles) >= 3, f"{session}: PV has fewer than three directional trials")
    readout, detail = build_pv_readout_from_trial_data(
        neural=neural,
        behavior=behavior,
        support_trial_bounds=labeled_bounds,
        support_direction_angles=labeled_angles,
        support_window_target_bins=endpoints,
    )
    query_target_bins = np.ascontiguousarray(starts + plan.WINDOW_SIZE - 1, dtype=np.int64)
    predicted = np.ascontiguousarray(
        predict_pv_from_readout(dataset.neural_data[session], readout, query_target_bins),
        dtype=np.float32,
    )
    return {
        "session": session,
        "surface": surface,
        "cell": cell.name,
        "family": cell.family,
        "budget": cell.budget,
        "support_selection": cell.support_selection,
        "supervision": cell.supervision,
        "source": "new_closed_form_fit",
        "window_count": int(starts.size),
        "ordered_window_starts_sha256": core.array_sha256(starts),
        "target_sha256": core.array_sha256(target),
        "selected_indices": selected.tolist(),
        "selected_indices_sha256": core.array_sha256(selected),
        "dense_support_rows": int(endpoints.size),
        "dense_support_endpoints_sha256": core.array_sha256(endpoints),
        "direction_labels_used": int(detail["target_direction_labels_used"]),
        "dense_velocity_coordinates_used": int(endpoints.size * plan.OUTPUT_DIM),
        "dense_support_evidence": support_evidence,
        "affine_gain_rank": int(detail["affine_gain_rank"]),
        "prediction_sha256": core.array_sha256(predicted),
        "r2": core.variance_weighted_r2(target, predicted),
        "parameter_updates": 0,
    }


def _load_parent(repo_root: Path) -> tuple[dict[tuple[str, str, str], Mapping[str, object]], str]:
    path = plan.parent_score(repo_root)
    core.require(path.is_file(), "accepted M2 activity parent score missing")
    observed_sha = core.file_sha256(path)
    core.require(observed_sha == plan.PARENT_SCORE_SHA256, "accepted M2 activity parent score drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    core.require(payload.get("schema") == "m2_t4_activity_budget_screen_v1", "parent schema drift")
    core.require(payload.get("status") == "TERMINAL" and payload.get("parameter_updates") == 0,
                 "parent is not a zero-update terminal")
    rows = payload.get("rows")
    core.require(isinstance(rows, list) and len(rows) == 65, "parent row cardinality drift")
    indexed: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for row in rows:
        core.require(isinstance(row, Mapping), "malformed parent row")
        key = (str(row.get("surface")), str(row.get("session")), str(row.get("cell")))
        core.require(key not in indexed, "duplicate parent row")
        indexed[key] = row
    return indexed, observed_sha


def _reused_parent_row(
    *,
    cell: core.CellSpec,
    surface: str,
    session: str,
    starts: np.ndarray,
    target: np.ndarray,
    parent: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> dict[str, object]:
    parent_name = _PARENT_CELL_MAP[cell.name]
    row = parent.get((surface, session, parent_name))
    core.require(row is not None, f"missing parent row {surface}/{session}/{parent_name}")
    core.require(row.get("ordered_window_starts_sha256") == core.array_sha256(starts),
                 "parent/current query start digest mismatch")
    core.require(row.get("target_sha256") == core.array_sha256(target),
                 "parent/current target digest mismatch")
    core.require(row.get("window_count") == int(starts.size), "parent/current query count mismatch")
    score = float(row["r2"])
    core.require(np.isfinite(score), "parent R2 nonfinite")
    return {
        **dict(row),
        "cell": cell.name,
        "family": cell.family,
        "budget": cell.budget,
        "support_selection": cell.support_selection,
        "supervision": cell.supervision,
        "source": "reused_parent_score_row",
        "parent_cell": parent_name,
        "parent_score_sha256": plan.PARENT_SCORE_SHA256,
        "parameter_updates": 0,
    }


def _publish_atomic(root: Path, payload: object) -> None:
    core.require(not root.exists(), f"result root already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        body = _canonical_json_bytes(payload)
        score = temporary / "score.json"
        score.write_bytes(body)
        digest = hashlib.sha256(body).hexdigest()
        sidecar = temporary / "score.json.sha256"
        sidecar.write_text(f"{digest}  score.json\n", encoding="ascii")
        score.chmod(0o444)
        sidecar.chmod(0o444)
        temporary.rename(root)
        root.chmod(0o555)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def execute(repo_root: Path, *, gpu_index: int = 1, batch_size: int = plan.DEFAULT_BATCH_SIZE) -> dict[str, object]:
    core.require(isinstance(gpu_index, int) and not isinstance(gpu_index, bool) and gpu_index >= 0,
                 "gpu_index must be a nonnegative integer")
    core.require(isinstance(batch_size, int) and not isinstance(batch_size, bool) and batch_size > 0,
                 "batch_size must be positive")
    core.require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu_index), "CUDA_VISIBLE_DEVICES mismatch")
    core.require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER mismatch")
    core.require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "deterministic CUBLAS workspace missing")
    root = plan.result_root(repo_root)
    core.require(not root.exists(), f"result root already exists: {root}")
    parent, parent_sha = _load_parent(repo_root)

    import torch

    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data

    core.require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    started = time.monotonic()
    model, data_module, _task_config, metadata = load_frozen_model_and_data()
    core.require(metadata["checkpoint_sha256"] == plan.T4_CHECKPOINT_SHA256, "T4 checkpoint drift")
    core.require(metadata["teacher_checkpoint_sha256"] == plan.SPINT_CHECKPOINT_SHA256, "SPINT checkpoint drift")
    core.require(metadata["normalization_sha256"] == plan.NORMALIZATION_SHA256, "T4 normalization drift")
    student = model.student.to(device).eval()
    teacher = model.teacher.to(device).eval()
    for module in (student, teacher):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    datasets = {
        "within_post30": data_module.train_dataset,
        "external_official_query": data_module.val_heldout_dataset,
    }
    expected_counts = {
        "within_post30": plan.EXPECTED_WITHIN_SESSIONS,
        "external_official_query": plan.EXPECTED_EXTERNAL_SESSIONS,
    }
    rows: list[dict[str, object]] = []
    cell_values: dict[str, dict[str, dict[str, float]]] = {}
    for surface, dataset in datasets.items():
        core.require(dataset is not None, f"{surface}: dataset missing")
        sessions = sorted(dataset.calib_trialized_neural_features)
        core.require(len(sessions) == expected_counts[surface], f"{surface}: session count drift")
        surface_values: dict[str, dict[str, float]] = {cell.name: {} for cell in core.CELL_SPECS}
        for session in sessions:
            starts, target = _query_arrays(dataset, session, surface)
            for cell in core.CELL_SPECS:
                if cell.source == "reused_parent":
                    row = _reused_parent_row(
                        cell=cell, surface=surface, session=session, starts=starts, target=target, parent=parent
                    )
                elif cell.family in {"spint", "t4_mean", "t4_ols"}:
                    row = _network_row(
                        torch=torch, cell=cell, student=student, teacher=teacher, dataset=dataset,
                        session=session, surface=surface, starts=starts, target=target,
                        device=device, batch_size=batch_size,
                    )
                elif cell.family == "direct_ridge":
                    row = _direct_ridge_row(
                        torch=torch, cell=cell, dataset=dataset, session=session, surface=surface,
                        starts=starts, target=target, device=device, batch_size=batch_size,
                    )
                else:
                    core.require(cell.family == "pv", "unreachable cell family")
                    row = _population_vector_row(
                        cell=cell, dataset=dataset, session=session, surface=surface,
                        starts=starts, target=target,
                    )
                rows.append(row)
                surface_values[cell.name][session] = float(row["r2"])
        cell_values[surface] = surface_values

    summaries = {
        f"{surface}:{cell.name}": core.summarize_sessions(cell_values[surface][cell.name])
        for surface in datasets
        for cell in core.CELL_SPECS
    }
    contrasts: dict[str, dict[str, object]] = {}
    contrast_pairs = []
    for budget in plan.BUDGETS:
        contrast_pairs.extend([
            (f"t4_ols_minus_spint_m{budget}", f"t4_ols_static_m{budget}", f"spint_chronological_m{budget}"),
            (f"t4_ridge_minus_ols_m{budget}", f"t4_ridge_static_m{budget}", f"t4_ols_static_m{budget}"),
            (f"direct_ridge_minus_t4_ridge_m{budget}", f"direct_ridge_m{budget}", f"t4_ridge_static_m{budget}"),
            (f"pv_minus_t4_ridge_m{budget}", f"population_vector_m{budget}", f"t4_ridge_static_m{budget}"),
        ])
    contrast_pairs.extend([
        ("activity30_minus_static_m10", "t4_ridge_activity30_m10", "t4_ridge_static_m10"),
        ("activity30_minus_static_m4", "t4_ridge_activity30_m4", "t4_ridge_static_m4"),
        ("spint_matched_minus_chronological_m4", "spint_matched_doptimal_m4", "spint_chronological_m4"),
    ])
    for surface in datasets:
        for label, candidate, reference in contrast_pairs:
            contrasts[f"{surface}:{label}"] = core.paired_contrast(
                cell_values[surface][candidate], cell_values[surface][reference]
            )

    expected_rows = sum(expected_counts.values()) * len(core.CELL_SPECS)
    core.require(len(rows) == expected_rows, "final row cardinality drift")
    payload = {
        "schema": plan.SCHEMA,
        "status": "TERMINAL",
        "scientific_role": "frozen_weight_same_query_comparator_matrix",
        "parent_score_sha256": parent_sha,
        "t4_checkpoint_sha256": metadata["checkpoint_sha256"],
        "spint_checkpoint_sha256": metadata["teacher_checkpoint_sha256"],
        "normalization_sha256": metadata["normalization_sha256"],
        "surfaces": list(datasets),
        "cell_order": [cell.name for cell in core.CELL_SPECS],
        "row_count": len(rows),
        "new_forward_or_fit_rows": int(sum(row["source"] != "reused_parent_score_row" for row in rows)),
        "reused_parent_rows": int(sum(row["source"] == "reused_parent_score_row" for row in rows)),
        "target_gradients": 0,
        "parameter_updates": 0,
        "query_target_used_for_selection": False,
        "gpu": {
            "cuda_visible_devices": str(gpu_index),
            "logical_device": "cuda:0",
            "name": torch.cuda.get_device_name(0),
            "batch_size": batch_size,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "elapsed_seconds": float(time.monotonic() - started),
        "rows": rows,
        "summaries": summaries,
        "paired_contrasts": contrasts,
    }
    _publish_atomic(root, payload)
    return payload
