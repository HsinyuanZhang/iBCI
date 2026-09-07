"""Physical driver of the M2 re-blocking cell (first-10 pool, post-10 stream).

CPU-only, inference-only, one launch.  The receipt law mirrors the sealed
routes and the chrono4 precedent: ``attempt.json`` (reserved before any data),
``replay.json`` (the governing grid), ``terminal.json`` (receipt-only
composition).  Nothing under any frozen result root is created or modified;
every sealed executor is reused by import.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import laws, plan


class ReblockPhysicalError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReblockPhysicalError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(path: Path, payload: Mapping[str, Any]) -> str:
    body = (json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        os.write(descriptor, body)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(body).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    descriptor = os.open(sidecar, flags, 0o444)
    try:
        os.write(descriptor, f"{digest}  {path.name}\n".encode("ascii"))
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest


def _validate_environment() -> None:
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
             "the re-blocking cell is CPU-only (CUDA_VISIBLE_DEVICES must be empty)")
    _require(os.environ.get("PYTHONNOUSERSITE") == "1",
             "PYTHONNOUSERSITE=1 is part of the environment law")
    for name, value in plan.ENVIRONMENT_LAW["blas_threads_env"].items():
        _require(os.environ.get(name) == value, f"{name}={value} is part of the environment law")


def _verify_sidecar(path: Path) -> str:
    digest = _sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    _require(sidecar.exists(), f"missing sidecar: {sidecar}")
    _require(sidecar.read_text(encoding="ascii").strip() == f"{digest}  {path.name}",
             f"sidecar drift for {path.name}")
    return digest


def _bind_namespaces(repo_root: Path) -> None:
    """The sealed namespace binding (tfpd src + streaming src coexistence)."""
    for entry in (str(repo_root), str(repo_root / "tfpd_exploration"),
                  str(repo_root / "sua_exploration")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from src.cdm_p1_m2_local_v1.physical import _bind_namespaces as sealed_bind

    sealed_bind(repo_root)


def _load_sealed_static_rows(repo_root: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    path = repo_root / plan.COMPARATOR_SCORE_RELATIVE
    _require(_sha256_file(path) == plan.COMPARATOR_SCORE_SHA256,
             "sealed comparator score drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == "m2_same_query_comparator_v1"
             and payload.get("status") == "TERMINAL", "sealed comparator schema drift")
    rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in payload["rows"]:
        if row.get("cell") == "t4_ridge_static_m4":
            rows[(str(row["surface"]), str(row["session"]))] = row
    _require(len(rows) == plan.EXPECTED_WITHIN_SESSIONS + plan.EXPECTED_EXTERNAL_SESSIONS,
             "sealed static m4 anchor row topology drift")
    return rows


def _load_sealed_cdm_rows(repo_root: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    path = repo_root / plan.CDM_SCREEN_SCORE_RELATIVE
    _require(_sha256_file(path) == plan.CDM_SCREEN_SCORE_SHA256,
             "sealed CDM screen score drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == "m2_precision_cdm_v2_screen_v1"
             and payload.get("status") == "TERMINAL", "sealed CDM screen schema drift")
    rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in payload["rows"]:
        if row.get("cell") == "m4_activity_only":
            rows[(str(row["surface"]), str(row["session_id"]))] = row
    _require(len(rows) == plan.EXPECTED_WITHIN_SESSIONS + plan.EXPECTED_EXTERNAL_SESSIONS,
             "sealed cdm m4 anchor row topology drift")
    return rows


# ---------------------------------------------------------------------------
# Session window starts (the re-blocked partition law).
# ---------------------------------------------------------------------------


def _session_windows(dataset: Any, session: str) -> np.ndarray:
    starts = np.asarray(
        [start for name, start in dataset.window_indices if name == session],
        dtype=np.int64,
    )
    _require(starts.size > 0 and np.all(np.diff(starts) > 0),
             f"{session}: query window order drift")
    return starts


def _static_surface_starts(dataset: Any, session: str, surface: str) -> np.ndarray:
    """The static family's scored starts: post10, or a sealed fidelity mode."""
    windows = _session_windows(dataset, session)
    trials = np.asarray(dataset.trial_start_indices[session], dtype=np.int64)
    if surface in plan.SURFACES_STATIC:
        return laws.select_post_h_window_starts(windows, trials, plan.BLOCK_HORIZON)
    if surface == "within_post30":
        return laws.select_post_h_window_starts(windows, trials, plan.OLD_HORIZON)
    _require(surface == "external_official_query", f"unknown surface {surface}")
    return np.ascontiguousarray(windows)


# ---------------------------------------------------------------------------
# The static family executor (the sealed t4_ridge_static law, support swapped).
# ---------------------------------------------------------------------------


def static_cell(
    *, torch: Any, student: Any, dataset: Any, session: str, starts: np.ndarray,
    selected: np.ndarray, digest_fn: Any, r2_fn: Any, batch_size: int,
    selection_label: str,
) -> dict[str, Any]:
    """The sealed ``t4_ridge_static_m4`` decode law over an arbitrary support.

    ``k == 4`` binds the sealed ``select_activity_rows`` law verbatim; a k-row
    (KALL) support uses the same selected-M arithmetic spelled directly (the
    sealed helper guards cardinality to M4/M10/M30 -- disclosed deviation).
    """
    from tfpd_exploration.src.calibration_budget_comparators_v1 import fit_ridge_t4
    from src.m2_t4_activity_budget_screen_v1.core import select_activity_rows

    support = np.asarray(selected, dtype=np.int64).reshape(-1)
    _require(support.size >= plan.M4 and int(support.max()) < plan.ACTIVITY_HORIZON,
             "the static support must lie in the first-30 pool")
    calibration = np.asarray(dataset.calib_trialized_neural_features[session], dtype=np.float32)
    if support.size == plan.M4:
        activity = select_activity_rows(
            calibration, selected_indices=support, activity_budget=plan.M4,
        )
        activity_law = "sealed_select_activity_rows_m4"
    else:
        activity = np.ascontiguousarray(calibration[support], dtype=np.float32)
        activity_law = "selected_rows_direct (the sealed selected-M arithmetic; the cardinality guard admits only M4/M10/M30)"
    _require(activity.shape == (support.size, plan.TRIAL_LENGTH, plan.CHANNELS)
             and np.isfinite(activity).all(), "static activity stack drift")
    sums = np.asarray(dataset.calib_trial_spike_sums[session][support], dtype=np.float64)
    lengths = np.asarray(dataset.calib_trial_lengths[session][support], dtype=np.float64)
    angles = np.asarray(dataset.calib_trial_target_angles[session][support], dtype=np.float64)
    usable = np.isfinite(angles)
    _require(int(usable.sum()) >= 3,
             f"{session}: the re-blocked support has fewer than three directional trials")
    rates = np.ascontiguousarray(sums[usable] / lengths[usable, None], dtype=np.float64)
    raw, evidence = fit_ridge_t4(
        rates, angles[usable], normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA,
    )
    mean = np.asarray(dataset.side_feature_mean, dtype=np.float32)
    std = np.asarray(dataset.side_feature_std, dtype=np.float32)
    _require(mean.shape == std.shape == (4,) and np.all(std > 0), "frozen T4 normalizer drift")
    side = np.ascontiguousarray((raw - mean) / std, dtype=np.float32)
    _require(side.shape == (plan.CHANNELS, 4) and np.isfinite(side).all(),
             "normalized re-blocked T4 drift")
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    support_tensor = torch.from_numpy(activity).unsqueeze(0)
    side_tensor = torch.from_numpy(side).unsqueeze(0)
    with torch.inference_mode():
        identity = student.compute_identity(support_tensor, side_features=side_tensor)
        _require(tuple(identity.shape) == (1, plan.CHANNELS, plan.WINDOW_SIZE),
                 "identity shape drift")
        for offset in range(0, starts.size, batch_size):
            chunk = starts[offset : offset + batch_size]
            neural = torch.from_numpy(np.ascontiguousarray(np.stack(
                [dataset.neural_data[session][start : start + plan.WINDOW_SIZE]
                 for start in chunk], axis=0, dtype=np.float32)))
            target = np.ascontiguousarray(np.stack(
                [dataset.covariate_data[session][start + plan.WINDOW_SIZE - 1]
                 for start in chunk], axis=0, dtype=np.float32))
            prediction, _ = student(neural, identity=identity)
            predictions.append(
                prediction[:, -1, :].detach().numpy().astype(np.float32, copy=False)
                / plan.BEHAVIOR_SCALE)
            targets.append(target)
    target = np.ascontiguousarray(np.concatenate(targets, axis=0), dtype=np.float32)
    prediction = np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32)
    return {
        "window_count": int(starts.size),
        "ordered_window_starts_sha256": digest_fn(starts),
        "target_sha256": digest_fn(target),
        "prediction_sha256": digest_fn(prediction),
        "activity_sha256": digest_fn(activity),
        "budget": int(support.size),
        "support_positions": support.tolist(),
        "activity_law": activity_law,
        "side_evidence": {
            **{key: evidence[key] for key in (
                "normalized_lambda", "design_rank", "design_condition", "gcv")},
            "selection": selection_label,
            "selected_indices": support.tolist(),
            "selected_indices_sha256": digest_fn(support),
            "usable_directional_trials": int(usable.sum()),
            "raw_t4_sha256": digest_fn(raw),
            "normalized_t4_sha256": digest_fn(side),
        },
        "r2": float(r2_fn(target, prediction)),
        "parameter_updates": 0,
    }


# ---------------------------------------------------------------------------
# The CDM family executor (the sealed m4_activity_only law, support swapped,
# over the extended post10 query stream).
# ---------------------------------------------------------------------------


def reblock_query_trial_rows(
    dataset: Any, session: str, *, raw_neural: np.ndarray, raw_starts: np.ndarray,
    activities: np.ndarray, horizon: int,
) -> tuple[dict[str, Any], ...]:
    """The sealed ``_query_trial_rows`` law with the boundary parameterized.

    ``horizon=10`` is the re-block's evidence stream (completed trials at
    positions 10+ become FIFO evidence); ``horizon=30`` is the sealed fidelity
    mode.  The metric/causal arithmetic is the sealed law verbatim.
    """
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import plan as cdm_plan

    neural_padded = np.asarray(dataset.neural_data[session], dtype=np.float32)
    padded_starts = np.asarray(dataset.trial_start_indices[session], dtype=np.int64)
    windows = _session_windows(dataset, session)
    common = laws.select_post_h_window_starts(windows, padded_starts, horizon)
    _require(
        padded_starts.shape == raw_starts.shape and activities.shape[0] == raw_starts.size
        and np.array_equal(padded_starts, raw_starts + (cdm_plan.WINDOW_BINS - 1)),
        "re-block padded/raw trial chronology drift",
    )
    rows: list[dict[str, Any]] = []
    joined: list[np.ndarray] = []
    for position in range(int(horizon), raw_starts.size):
        raw_start = int(raw_starts[position])
        raw_stop = int(raw_starts[position + 1]) if position + 1 < raw_starts.size \
            else int(raw_neural.shape[0])
        padded_start = int(padded_starts[position])
        padded_stop = int(padded_starts[position + 1]) if position + 1 < padded_starts.size \
            else int(neural_padded.shape[0])
        endpoints = common + cdm_plan.WINDOW_BINS - 1
        metric = np.ascontiguousarray(
            common[(endpoints >= padded_start) & (endpoints < padded_stop)], dtype=np.int64,
        )
        causal = np.arange(padded_start, padded_stop - cdm_plan.WINDOW_BINS + 1, dtype=np.int64)
        _require(raw_stop > raw_start, "re-block query trial interval drift")
        joined.append(metric)
        rows.append({
            "position": position, "trial_id": f"{session}:trial:{position}",
            "raw_start": raw_start, "raw_stop": raw_stop,
            "padded_start": padded_start, "padded_stop": padded_stop,
            "metric_starts": metric, "causal_starts": np.ascontiguousarray(causal, dtype=np.int64),
            "activity": np.ascontiguousarray(activities[position], dtype=np.float32),
        })
    _require(rows and np.array_equal(np.concatenate(joined), common),
             "re-block query trials do not partition the post-%d metric surface" % horizon)
    return tuple(rows)


def _kall_carrier_and_activity(
    support: Mapping[str, Any], selected: np.ndarray,
) -> tuple[np.ndarray, Any]:
    """The k-row carrier/activity binding (the sealed arithmetic, unguarded).

    The sealed ``fit_initial_carrier``/``make_activity_memory`` guard the
    support cardinality to M4/M10/M30; the activity-only law they implement is
    (a) the fixed-ridge-by-trial carrier fit over the support rows and (b) a
    B3S activity memory of support k + FIFO capacity 30-k (the frozen 30-trial
    stack ceiling).  This binding spells exactly that for any k in [4, 10].
    """
    from tfpd_exploration.src.causal_dual_memory_cell_d_v1 import core as tfpd_cdm_core
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core as pseudo_core

    indices = np.asarray(selected, dtype=np.int64).reshape(-1)
    k = int(indices.size)
    rates = np.ascontiguousarray(
        support["support_rates_hz30"][indices], dtype=np.float64)
    theta = np.ascontiguousarray(support["theta30"][indices], dtype=np.float64)
    _require(bool(np.isfinite(theta).all()), "the k-row support includes a non-directional row")
    directions = pseudo_core.canonical_direction_indices(theta)
    fitted = tfpd_cdm_core.fit_carriers_from_trial_table(
        rates, directions,
        mode=tfpd_cdm_core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA,
    )
    carrier_hz = np.asarray(fitted, dtype=np.float64)
    activity = tfpd_cdm_core.ActivityMemory.initialize(
        tuple(support["support_b3s"][index] for index in indices),
        channel_ids=support["channels"],
        fifo_capacity=plan.ACTIVITY_STACK_LIMIT - k,
    )
    return carrier_hz, activity


def cdm_activity_cell(
    *, torch: Any, model: Any, dataset: Any, session: str, surface: str,
    query_rows: tuple[dict[str, Any], ...], support: Mapping[str, Any],
    raw_neural: np.ndarray, selected: np.ndarray, side_mean: np.ndarray,
    side_std: np.ndarray, device: Any, batch_size: int, digest_fn: Any,
    r2_fn: Any,
) -> dict[str, Any]:
    """The sealed ``m4_activity_only`` law over an arbitrary first-10 support.

    The activity FIFO is initialized on the selected support rows and advances
    on EVERY completed query trial of the given stream (the sealed V3 law);
    the frozen canonical fixed-ridge carrier never moves (activity-only).
    """
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical as cdm_physical
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import plan as cdm_plan
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import physical as shared_physical

    indices = np.asarray(selected, dtype=np.int64).reshape(-1)
    k = int(indices.size)
    first10_support = (plan.M4 <= k <= plan.BLOCK_HORIZON
                       and int(indices.max()) < plan.BLOCK_HORIZON)
    oldlaw_support = (k == plan.M4 and int(indices.max()) < plan.ACTIVITY_HORIZON)
    _require(first10_support or oldlaw_support,
             "the CDM support must be a first-10 pool support (k rows, k <= 10) "
             "or the OLD-law 4-row first-30 support")
    if k == plan.M4:
        memory = cdm_physical._build_memory(
            system="activity_only", budget=plan.M4, selected=indices,
            support_rates_hz30=support["support_rates_hz30"],
            theta_first30=support["theta30"], support_b3s=support["support_b3s"],
            channel_ids=support["channels"], valid_mask=support["valid_mask"],
        )
        binding = "sealed_build_memory_m4"
        kall_carrier = None
    else:
        carrier_hz_fixed, activity = _kall_carrier_and_activity(support, indices)
        binding = "generalized_k_row_binding"
        kall_carrier = np.ascontiguousarray(carrier_hz_fixed, dtype=np.float64)
    neural = np.asarray(dataset.neural_data[session], dtype=np.float32)
    behavior = np.asarray(dataset.covariate_data[session], dtype=np.float32)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    starts_joined: list[np.ndarray] = []
    short_trials = 0
    # The frozen B3S memory law rejects a duplicate trial id: a trial cannot
    # be both immutable support and FIFO query evidence.  First-10 supports
    # never overlap the post10 stream (supports lie inside positions 0-9); an
    # OLD-law (from-30) support can, and those positions stay as support.
    support_positions = {int(item) for item in indices.tolist()}
    skipped: list[int] = []
    advance_events = 0
    for row in query_rows:
        metric_starts = np.asarray(row["metric_starts"], dtype=np.int64)
        if k == plan.M4:
            activity_stack, carrier_hz = memory.prediction_inputs()
        else:
            activity_stack, carrier_hz = activity.stack(), carrier_hz_fixed
        if metric_starts.size:
            prediction = shared_physical._predict(
                torch=torch, model=model,
                neural_windows=cdm_physical._windows(neural, metric_starts),
                activity=activity_stack,
                raw_t4=np.ascontiguousarray(
                    carrier_hz * cdm_plan.MODEL_BIN_SECONDS, dtype=np.float32),
                side_mean=side_mean, side_std=side_std, device=device,
                batch_size=batch_size,
            )
            predictions.append(prediction)
            targets.append(np.ascontiguousarray(
                behavior[metric_starts + cdm_plan.WINDOW_BINS - 1], dtype=np.float32))
            starts_joined.append(metric_starts)
        if np.asarray(row["causal_starts"], dtype=np.int64).size == 0:
            short_trials += 1
        position = int(row["position"])
        if position in support_positions:
            skipped.append(position)
            continue
        b3s, _native, _validity = cdm_physical._trial_capabilities(
            session=session, row=row, raw_neural=raw_neural,
            channel_ids=support["channels"],
        )
        advance_events += 1
        if k == plan.M4:
            memory.commit_activity_only(b3s)
        else:
            activity = activity.after_completed_trial(b3s)
    if int(indices.max()) < plan.BLOCK_HORIZON:
        _require(not skipped,
                 "a first-10 support overlapped the post10 evidence stream")
    prediction = np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32)
    target = np.ascontiguousarray(np.concatenate(targets, axis=0), dtype=np.float32)
    starts = np.ascontiguousarray(np.concatenate(starts_joined), dtype=np.int64)
    windows = _session_windows(dataset, session)
    expected = laws.select_post_h_window_starts(
        windows, np.asarray(dataset.trial_start_indices[session], dtype=np.int64),
        int(query_rows[0]["position"]) if query_rows else plan.BLOCK_HORIZON,
    )
    _require(np.array_equal(starts, expected),
             "the scored query differs from its post-H partition authority")
    return {
        "window_count": int(starts.size),
        "ordered_window_starts_sha256": digest_fn(starts),
        "query_starts_sha256": digest_fn(starts),
        "target_sha256": digest_fn(target),
        "prediction_sha256": digest_fn(prediction),
        "budget": k,
        "support_positions": indices.tolist(),
        "memory_binding": binding,
        "fifo_capacity": plan.ACTIVITY_STACK_LIMIT - k,
        "fifo_advance_events": int(advance_events),
        "fifo_skipped_support_positions": [int(item) for item in skipped],
        "query_row_count": len(query_rows),
        "short_query_trials_without_complete_50_bin_window": int(short_trials),
        "carrier_hz_sha256": digest_fn(
            kall_carrier if kall_carrier is not None
            else np.asarray(memory.carrier.active_t4, dtype=np.float64)),
        "r2": float(r2_fn(target, prediction)),
        "parameter_updates": 0,
        "accepted_carrier_updates": 0,
    }


# ---------------------------------------------------------------------------
# The governing grid.
# ---------------------------------------------------------------------------


def execute(repo_root: Path, *, batch_size: int = plan.BATCH_SIZE) -> dict[str, Any]:
    _validate_environment()
    repo_root = Path(repo_root).absolute()
    root = plan.result_root(repo_root)
    _require(not (root / "replay.json").exists(), "the replay receipt already exists")
    _require(not (root / "terminal.json").exists(), "the terminal receipt already exists")
    _require((root / "attempt.json").exists(), "reserve the attempt receipt first")
    attempt_digest = _verify_sidecar(root / "attempt.json")
    attempt = json.loads((root / "attempt.json").read_text(encoding="utf-8"))
    for relative, digest in attempt["predecessor_sha256s"].items():
        _require(_sha256_file(repo_root / relative) == digest,
                 f"an immutable predecessor drifted: {relative}")
    for relative, digest in attempt["owned_sha256s"].items():
        _require(_sha256_file(repo_root / relative) == digest,
                 f"an owned re-block module drifted: {relative}")

    _bind_namespaces(repo_root)

    import torch

    torch.set_num_threads(int(plan.ENVIRONMENT_LAW["torch_num_threads"]))
    try:
        torch.set_num_interop_threads(int(plan.ENVIRONMENT_LAW["torch_num_interop_threads"]))
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    _require(not torch.cuda.is_available(),
             "the CPU-only law broke: CUDA is visible to this process")
    _require(torch.get_num_threads() == int(plan.ENVIRONMENT_LAW["torch_num_threads"]),
             "torch thread binding drift")
    device = torch.device("cpu")

    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data

    model, data_module, _task_config, metadata = load_frozen_model_and_data()
    _require(metadata["checkpoint_sha256"] == plan.T4_CHECKPOINT_SHA256, "T4 checkpoint drift")
    _require(metadata["teacher_checkpoint_sha256"] == plan.SPINT_CHECKPOINT_SHA256,
             "SPINT teacher checkpoint drift")
    _require(metadata["normalization_sha256"] == plan.NORMALIZATION_SHA256,
             "T4 normalization drift")
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    student = model.student
    side_mean = np.asarray(metadata["normalization_mean"], dtype=np.float32)
    side_std = np.asarray(metadata["normalization_std"], dtype=np.float32)

    from src.cdm_p1_m2_local_v1 import replay as g_replay
    from src.m2_same_query_comparator_v1.core import array_sha256 as static_digest
    from src.m2_same_query_comparator_v1.core import variance_weighted_r2 as static_r2
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical as cdm_physical
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core as pseudo_core

    sealed_static = _load_sealed_static_rows(repo_root)
    sealed_cdm = _load_sealed_cdm_rows(repo_root)

    train_ds = data_module.train_dataset
    heldout_ds = data_module.val_heldout_dataset
    _require(train_ds is not None and heldout_ds is not None,
             "the frozen M2 runtime must expose both datasets")
    for dataset in (train_ds, heldout_ds):
        _require(np.array_equal(np.asarray(dataset.side_feature_mean, dtype=np.float32), side_mean)
                 and np.array_equal(np.asarray(dataset.side_feature_std, dtype=np.float32), side_std),
                 "dataset/metadata T4 normalizer disagreement")
    static_datasets = {"within_post10": train_ds, "external_post10_query": heldout_ds}
    cdm_datasets = {
        "within_post10": (train_ds, data_module.train_calib_heldin_sessions),
        "external_post10_local": (heldout_ds, data_module.val_calib_heldout_sessions),
    }
    expected_counts = {
        "within_post10": plan.EXPECTED_WITHIN_SESSIONS,
        "external_post10_query": plan.EXPECTED_EXTERNAL_SESSIONS,
        "external_post10_local": plan.EXPECTED_EXTERNAL_SESSIONS,
    }
    sealed_static_surface = {
        "within_post10": "within_post30",
        "external_post10_query": "external_official_query",
    }
    sealed_cdm_surface = {"within_post10": "within_post30",
                          "external_post10_local": "external_post30_local"}
    fidelity_roster_static = (
        [("external_post10_query", name) for name in sorted(
            heldout_ds.calib_trialized_neural_features)[:2]]
        + [("within_post10", name) for name in sorted(
            train_ds.calib_trialized_neural_features)[:1]]
    )
    fidelity_roster_cdm = (
        [("external_post10_local", name) for name in sorted(
            heldout_ds.calib_trialized_neural_features)[:2]]
        + [("within_post10", name) for name in sorted(
            train_ds.calib_trialized_neural_features)[:1]]
    )
    started = time.monotonic()

    # -- Phase A: supports, D-opt proofs, surface agreements -----------------
    supports: dict[str, dict[str, dict[str, Any]]] = {"static": {}, "cdm": {}}
    dopt_proofs: dict[str, Any] = {}
    cdm_materials: dict[str, dict[str, Any]] = {}
    for family, datasets in (("static", static_datasets), ("cdm", cdm_datasets)):
        for surface, binding in datasets.items():
            dataset = binding[0] if family == "cdm" else binding
            raw_sessions = binding[1] if family == "cdm" else None
            sessions = sorted(dataset.calib_trialized_neural_features)
            _require(len(sessions) == expected_counts[surface], f"{surface}: session count drift")
            for session in sessions:
                key = f"{family}|{surface}|{session}"
                theta = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
                payload = laws.support_payload(theta)
                if family == "cdm":
                    _require(session in raw_sessions, f"{surface}/{session}: raw authority absent")
                    views = g_replay._g_session_views(raw_sessions, session=session)
                    support_base = g_replay.g_support_material(session=session, views=views)
                    cdm_materials[f"{surface}|{session}"] = {
                        "views": views, "support": support_base,
                    }
                    theta30 = np.asarray(support_base["theta30"], dtype=np.float64)
                    payload_views = laws.support_payload(theta30)
                    _require(payload_views == payload,
                             f"{key}: the two angle spellings disagree on the re-blocked support")
                    dopt30_views = laws.dopt30_support(theta30)
                    sealed_row = sealed_cdm[(sealed_cdm_surface[surface], session)]
                    proof = laws.dopt_proof(dopt30_views, sealed_row["support_indices"])
                    dopt_proofs[f"cdm|{surface}|{session}"] = {
                        **proof,
                        "sealed_law_selection": "doptimal_noncentre_first30",
                        "g_support_material_agrees": bool(np.array_equal(
                            np.asarray(support_base["selected"], dtype=np.int64), dopt30_views)),
                    }
                    _require(proof["exact_match"] and dopt_proofs[f"cdm|{surface}|{session}"][
                        "g_support_material_agrees"],
                             f"the sealed cdm m4 indices are not the recomputed D-opt law at {key}")
                else:
                    dopt30 = laws.dopt30_support(theta)
                    sealed_row = sealed_static[(sealed_static_surface[surface], session)]
                    proof = laws.dopt_proof(dopt30, sealed_row["side_evidence"]["selected_indices"])
                    dopt_proofs[f"static|{surface}|{session}"] = {
                        **proof,
                        "sealed_law_selection": str(sealed_row["side_evidence"]["selection"]),
                    }
                    _require(proof["exact_match"],
                             f"the sealed static m4 indices are not the recomputed D-opt law at {key}")
                supports[family][f"{surface}|{session}"] = {
                    **payload,
                    "dopt30_support_positions": [int(item) for item in (
                        dopt30_views if family == "cdm" else laws.dopt30_support(theta))],
                }
                _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                         "the hard timeout fired during the support phase")

    support_agreement = laws.cross_family_support_agreement(
        supports["static"], supports["cdm"])

    # -- Phase B: the static family ------------------------------------------
    static_rows: dict[str, dict[str, dict[str, Any]]] = {
        surface: {} for surface in plan.SURFACES_STATIC}
    static_pairing: dict[str, Any] = {}
    for surface, dataset in static_datasets.items():
        for session in sorted(dataset.calib_trialized_neural_features):
            starts = _static_surface_starts(dataset, session, surface)
            entry = supports["static"][f"{surface}|{session}"]
            cells = {
                "BLOCK10_STATIC_K4": (np.asarray(entry["k4_support_positions"], dtype=np.int64),
                                      "greedy_doptimal_4_within_first10_candidates"),
                "BLOCK10_STATIC_KALL": (np.asarray(entry["kall_support_positions"], dtype=np.int64),
                                        "all_first10_directional_candidates"),
                "OLDLAW_STATIC": (np.asarray(entry["dopt30_support_positions"], dtype=np.int64),
                                  "doptimal_noncentre_first30 (the sealed OLD law, re-scored)"),
            }
            for cell, (support, label) in cells.items():
                row = static_cell(
                    torch=torch, student=student, dataset=dataset, session=session,
                    starts=starts, selected=support, digest_fn=static_digest,
                    r2_fn=static_r2, batch_size=batch_size, selection_label=label,
                )
                static_rows[surface].setdefault(session, {})[cell] = {
                    "surface": surface, "session": session, "cell": cell, **row}
            trio = static_rows[surface][session]
            for cell in ("BLOCK10_STATIC_K4", "BLOCK10_STATIC_KALL"):
                matches = {
                    "window_count": trio[cell]["window_count"] == trio["OLDLAW_STATIC"]["window_count"],
                    "ordered_window_starts_sha256": (
                        trio[cell]["ordered_window_starts_sha256"]
                        == trio["OLDLAW_STATIC"]["ordered_window_starts_sha256"]),
                    "target_sha256": (trio[cell]["target_sha256"]
                                      == trio["OLDLAW_STATIC"]["target_sha256"]),
                }
                static_pairing[f"{surface}|{session}|{cell}"] = {
                    "field_matches": matches,
                    "exact_match": all(bool(value) for value in matches.values()),
                }
                _require(static_pairing[f"{surface}|{session}|{cell}"]["exact_match"],
                         f"the post10 pure-data pairing failed at {surface}/{session}/{cell}")
            _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                     "the hard timeout fired during the static family")

    # -- Phase C: the CDM family ----------------------------------------------
    cdm_rows: dict[str, dict[str, dict[str, Any]]] = {
        surface: {} for surface in plan.SURFACES_CDM}
    cdm_pairing: dict[str, Any] = {}
    kall_parity: dict[str, Any] = {}
    for surface, (dataset, raw_sessions) in cdm_datasets.items():
        for session in sorted(dataset.calib_trialized_neural_features):
            material = cdm_materials[f"{surface}|{session}"]
            support_base = material["support"]
            views = material["views"]
            raw_neural, raw_starts, _theta, _rates, activities = views
            entry = supports["cdm"][f"{surface}|{session}"]
            query_rows = reblock_query_trial_rows(
                dataset, session, raw_neural=raw_neural, raw_starts=raw_starts,
                activities=activities, horizon=plan.BLOCK_HORIZON,
            )
            cells = {
                "BLOCK10_CDM_K4": np.asarray(entry["k4_support_positions"], dtype=np.int64),
                "BLOCK10_CDM_KALL": np.asarray(entry["kall_support_positions"], dtype=np.int64),
                "OLDLAW_CDM": np.asarray(entry["dopt30_support_positions"], dtype=np.int64),
            }
            for cell, support in cells.items():
                row = cdm_activity_cell(
                    torch=torch, model=model, dataset=dataset, session=session,
                    surface=surface, query_rows=query_rows, support=support_base,
                    raw_neural=raw_neural, selected=support,
                    side_mean=side_mean, side_std=side_std,
                    device=device, batch_size=batch_size,
                    digest_fn=pseudo_core.array_sha256,
                    r2_fn=pseudo_core.variance_weighted_r2,
                )
                cdm_rows[surface].setdefault(session, {})[cell] = {
                    "surface": surface, "session": session, "cell": cell, **row}
                _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                         "the hard timeout fired during the cdm family")
            # the KALL binding parity anchor: driven with the K4 support it must
            # reproduce the sealed _build_memory carrier bit-for-bit
            sealed_memory = cdm_physical._build_memory(
                system="activity_only", budget=plan.M4,
                selected=np.asarray(entry["k4_support_positions"], dtype=np.int64),
                support_rates_hz30=support_base["support_rates_hz30"],
                theta_first30=support_base["theta30"],
                support_b3s=support_base["support_b3s"],
                channel_ids=support_base["channels"],
                valid_mask=support_base["valid_mask"],
            )
            kall_carrier, _activity = _kall_carrier_and_activity(
                support_base, np.asarray(entry["k4_support_positions"], dtype=np.int64))
            parity = {
                "law": plan.ANCHORS["kall_binding_parity"]["law"],
                "bitwise_equal": bool(np.array_equal(
                    np.ascontiguousarray(kall_carrier, dtype=np.float64),
                    np.ascontiguousarray(
                        np.asarray(sealed_memory.carrier.active_t4), dtype=np.float64))),
            }
            kall_parity[f"{surface}|{session}"] = parity
            _require(parity["bitwise_equal"],
                     f"the KALL carrier binding drifted from the sealed law at {surface}/{session}")
            trio = cdm_rows[surface][session]
            for cell in ("BLOCK10_CDM_K4", "BLOCK10_CDM_KALL"):
                matches = {
                    "window_count": trio[cell]["window_count"] == trio["OLDLAW_CDM"]["window_count"],
                    "query_starts_sha256": (trio[cell]["query_starts_sha256"]
                                            == trio["OLDLAW_CDM"]["query_starts_sha256"]),
                    "target_sha256": (trio[cell]["target_sha256"]
                                      == trio["OLDLAW_CDM"]["target_sha256"]),
                }
                cdm_pairing[f"{surface}|{session}|{cell}"] = {
                    "field_matches": matches,
                    "exact_match": all(bool(value) for value in matches.values()),
                }
                _require(cdm_pairing[f"{surface}|{session}|{cell}"]["exact_match"],
                         f"the post10 pure-data pairing failed at {surface}/{session}/{cell}")

    # the two external spellings bind the SAME window set
    for session in sorted(heldout_ds.calib_trialized_neural_features):
        left = _static_surface_starts(heldout_ds, session, "external_post10_query")
        right = laws.select_post_h_window_starts(
            _session_windows(heldout_ds, session),
            np.asarray(heldout_ds.trial_start_indices[session], dtype=np.int64),
            plan.BLOCK_HORIZON)
        _require(np.array_equal(left, right),
                 f"the two external post10 spellings disagree at {session}")

    # -- Phase D: executor fidelity on the sealed partitions ------------------
    static_fidelity: dict[str, Any] = {}
    for surface, session in fidelity_roster_static:
        dataset = static_datasets[surface]
        sealed_surface = sealed_static_surface[surface]
        starts = _static_surface_starts(dataset, session, sealed_surface)
        entry = supports["static"][f"{surface}|{session}"]
        row = static_cell(
            torch=torch, student=student, dataset=dataset, session=session,
            starts=starts, selected=np.asarray(entry["dopt30_support_positions"], dtype=np.int64),
            digest_fn=static_digest, r2_fn=static_r2, batch_size=batch_size,
            selection_label="doptimal_noncentre_first30",
        )
        row.update({"surface": sealed_surface, "session": session,
                    "cell": "STATIC_FIDELITY_DOPT"})
        anchored = laws.fidelity_anchor(
            row, sealed_static[(sealed_surface, session)],
            r2_tolerance=float(plan.ANCHORS["executor_fidelity"]["r2_tolerance"]))
        static_fidelity[f"{sealed_surface}|{session}"] = {"row": row, "anchor": anchored}
        _require(anchored["exact_match"],
                 f"the static executor fidelity anchor failed at {sealed_surface}/{session}")
        _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                 "the hard timeout fired during the static fidelity anchors")
    cdm_fidelity: dict[str, Any] = {}
    for surface, session in fidelity_roster_cdm:
        dataset, raw_sessions = cdm_datasets[surface]
        sealed_surface = sealed_cdm_surface[surface]
        material = cdm_materials[f"{surface}|{session}"]
        raw_neural, raw_starts, _theta, _rates, activities = material["views"]
        entry = supports["cdm"][f"{surface}|{session}"]
        query_rows = reblock_query_trial_rows(
            dataset, session, raw_neural=raw_neural, raw_starts=raw_starts,
            activities=activities, horizon=plan.OLD_HORIZON)
        row = cdm_activity_cell(
            torch=torch, model=model, dataset=dataset, session=session,
            surface=sealed_surface, query_rows=query_rows, support=material["support"],
            raw_neural=raw_neural,
            selected=np.asarray(entry["dopt30_support_positions"], dtype=np.int64),
            side_mean=side_mean, side_std=side_std, device=device, batch_size=batch_size,
            digest_fn=pseudo_core.array_sha256, r2_fn=pseudo_core.variance_weighted_r2,
        )
        row.update({"surface": sealed_surface, "session": session,
                    "cell": "CDM_FIDELITY_DOPT"})
        anchored = laws.fidelity_anchor(
            row, sealed_cdm[(sealed_surface, session)],
            r2_tolerance=float(plan.ANCHORS["executor_fidelity"]["r2_tolerance"]))
        cdm_fidelity[f"{sealed_surface}|{session}"] = {"row": row, "anchor": anchored}
        _require(anchored["exact_match"],
                 f"the cdm executor fidelity anchor failed at {sealed_surface}/{session}")
        _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                 "the hard timeout fired during the cdm fidelity anchors")

    # -- Phase E: the evidence-stream census ----------------------------------
    census: dict[str, dict[str, Any]] = {"within": {}, "external": {}}
    for session in sorted(train_ds.calib_trialized_neural_features):
        trials = np.asarray(train_ds.trial_start_indices[session], dtype=np.int64)
        census["within"][session] = {
            **laws.evidence_census(int(trials.size)),
            **laws.window_census(_session_windows(train_ds, session), trials),
        }
    for session in sorted(heldout_ds.calib_trialized_neural_features):
        trials = np.asarray(heldout_ds.trial_start_indices[session], dtype=np.int64)
        entry = {
            **laws.evidence_census(int(trials.size)),
            **laws.window_census(_session_windows(heldout_ds, session), trials),
        }
        entry["official_query_unfiltered_windows"] = int(
            _session_windows(heldout_ds, session).size)
        census["external"][session] = entry
    external_ratios = [item["extension_ratio_new_over_old"]
                       for item in census["external"].values()]
    census_summary = {
        "external_extension_ratio_min": float(min(external_ratios)),
        "external_extension_ratio_max": float(max(external_ratios)),
        "external_extension_ratio_mean": float(sum(external_ratios) / len(external_ratios)),
        "pre_registered_expectation": "2-3x on external (the work order's claim)",
    }

    # -- summaries, contrasts, gates, verdicts --------------------------------
    def _cell_values(family_rows: Mapping[str, Any], surface: str, cell: str) -> dict[str, float]:
        return {session: float(family_rows[surface][session][cell]["r2"])
                for session in sorted(family_rows[surface])}

    static_summaries = {
        surface: {cell: {
            "equal_session_mean": laws.equal_session_mean(_cell_values(static_rows, surface, cell)),
            "per_session_r2": _cell_values(static_rows, surface, cell),
        } for cell in ("BLOCK10_STATIC_K4", "BLOCK10_STATIC_KALL", "OLDLAW_STATIC")}
        for surface in plan.SURFACES_STATIC
    }
    cdm_summaries = {
        surface: {cell: {
            "equal_session_mean": laws.equal_session_mean(_cell_values(cdm_rows, surface, cell)),
            "per_session_r2": _cell_values(cdm_rows, surface, cell),
        } for cell in ("BLOCK10_CDM_K4", "BLOCK10_CDM_KALL", "OLDLAW_CDM")}
        for surface in plan.SURFACES_CDM
    }
    static_contrasts = {
        surface: {
            cell: laws.paired_contrast(
                _cell_values(static_rows, surface, cell),
                _cell_values(static_rows, surface, "OLDLAW_STATIC"))
            for cell in ("BLOCK10_STATIC_K4", "BLOCK10_STATIC_KALL")
        }
        for surface in plan.SURFACES_STATIC
    }
    cdm_contrasts = {
        surface: {
            cell: laws.paired_contrast(
                _cell_values(cdm_rows, surface, cell),
                _cell_values(cdm_rows, surface, "OLDLAW_CDM"))
            for cell in ("BLOCK10_CDM_K4", "BLOCK10_CDM_KALL")
        }
        for surface in plan.SURFACES_CDM
    }
    primary_gate = laws.gate_evaluation(
        cdm_contrasts["external_post10_local"]["BLOCK10_CDM_K4"])
    secondary_gates = {
        "cdm_kall_external": laws.gate_evaluation(
            cdm_contrasts["external_post10_local"]["BLOCK10_CDM_KALL"]),
        "static_k4_external": laws.gate_evaluation(
            static_contrasts["external_post10_query"]["BLOCK10_STATIC_K4"]),
        "static_kall_external": laws.gate_evaluation(
            static_contrasts["external_post10_query"]["BLOCK10_STATIC_KALL"]),
    }
    secondary_gates["expected_sign_disclosure"] = (
        "the static contrasts are expected NEGATIVE: a static deployment cannot "
        "exploit the extended evidence stream, so the smaller first-10 support "
        "can only cost accuracy (reported, never gated)")
    within_disclosures = {
        "cdm_k4_within": laws.gate_evaluation(
            cdm_contrasts["within_post10"]["BLOCK10_CDM_K4"]),
        "cdm_kall_within": laws.gate_evaluation(
            cdm_contrasts["within_post10"]["BLOCK10_CDM_KALL"]),
        "static_k4_within": laws.gate_evaluation(
            static_contrasts["within_post10"]["BLOCK10_STATIC_K4"]),
        "static_kall_within": laws.gate_evaluation(
            static_contrasts["within_post10"]["BLOCK10_STATIC_KALL"]),
    }

    sealed_context = {
        "disclosure": (
            "cross-partition sealed receipt values (the OLD law on its own "
            "sealed post30/official-query surfaces); reported as context ONLY, "
            "never mixed into any contrast or gate"),
        "static_t4_ridge_static_m4": {
            surface: {
                "equal_session_mean": laws.equal_session_mean({
                    session: float(sealed_static[(sealed_surface, session)]["r2"])
                    for session in sorted(static_rows[surface])
                }),
            }
            for surface, sealed_surface in (("within_post10", "within_post30"),
                                            ("external_post10_query", "external_official_query"))
        },
        "cdm_m4_activity_only": {
            surface: {
                "equal_session_mean": laws.equal_session_mean({
                    session: float(sealed_cdm[(sealed_surface, session)]["r2"])
                    for session in sorted(cdm_rows[surface])
                }),
            }
            for surface, sealed_surface in (("within_post10", "within_post30"),
                                            ("external_post10_local", "external_post30_local"))
        },
    }

    pairing_all_exact = all(item["exact_match"] for item in static_pairing.values()) and \
        all(item["exact_match"] for item in cdm_pairing.values())
    fidelity_all_exact = all(item["anchor"]["exact_match"] for item in static_fidelity.values()) \
        and all(item["anchor"]["exact_match"] for item in cdm_fidelity.values())
    dopt_proofs_all_exact = all(item["exact_match"] for item in dopt_proofs.values())
    kall_parity_all = all(item["bitwise_equal"] for item in kall_parity.values())
    stop_conditions = {
        "anchor_or_binding_failure": {
            "expression": "any pure-data pairing, executor-fidelity, D-opt proof, KALL parity or support-agreement check failed",
            "fired": not (pairing_all_exact and fidelity_all_exact and dopt_proofs_all_exact
                          and kall_parity_all
                          and all(item["agree"] for item in support_agreement.values())),
            "evidence": {
                "pure_data_pairing_all_exact": bool(pairing_all_exact),
                "executor_fidelity_all_exact": bool(fidelity_all_exact),
                "dopt_proofs_all_exact": bool(dopt_proofs_all_exact),
                "kall_binding_parity_all": bool(kall_parity_all),
                "cross_family_support_agreement": bool(
                    all(item["agree"] for item in support_agreement.values())),
                "fidelity_max_abs_r2_delta": max(
                    [abs(item["anchor"]["r2_delta"]) for item in static_fidelity.values()]
                    + [abs(item["anchor"]["r2_delta"]) for item in cdm_fidelity.values()]),
            },
        },
    }
    governing = [name for name, item in stop_conditions.items() if item["fired"]]

    replay_payload = {
        "schema": f"{plan.SCHEMA}_replay_v1",
        "stage": "inference_only_first10_pool_post10_stream_local_m2_cpu",
        "status": "REPLAY_COMPLETE",
        "attempt_sha256": attempt_digest,
        "authority": (
            "operator work order 2026-09-02: the M2 re-blocking cell (first-10 "
            "labeled pool, post-10 query/evidence stream), CPU-only, sealed "
            "anchors read from receipts"
        ),
        "foundation": {
            "sealed_comparator_score": plan.COMPARATOR_SCORE_RELATIVE,
            "sealed_comparator_score_sha256": plan.COMPARATOR_SCORE_SHA256,
            "sealed_cdm_screen_score": plan.CDM_SCREEN_SCORE_RELATIVE,
            "sealed_cdm_screen_score_sha256": plan.CDM_SCREEN_SCORE_SHA256,
            "sealed_g_package": plan.SEALED_G_PACKAGE,
            "chrono4_precedent": plan.CHRONO4_PACKAGE,
            "reuse_law": "frozen modules imported verbatim; nothing edited",
        },
        "environment": {
            **plan.ENVIRONMENT_LAW,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "batch_size": int(batch_size),
            "cpu_count_visible_to_torch": int(torch.get_num_threads()),
        },
        "surface": {"law": dict(plan.SURFACE_LAW),
                    "static_surfaces": list(plan.SURFACES_STATIC),
                    "cdm_surfaces": list(plan.SURFACES_CDM)},
        "cells": {key: (dict(value) if isinstance(value, dict) else value)
                  for key, value in plan.CELLS.items()},
        "reblock_law": dict(plan.REBLOCK_LAW),
        "frozen_everything_else": dict(plan.FROZEN_EVERYTHING_ELSE),
        "supports": {
            "static": dict(sorted(supports["static"].items())),
            "cdm": dict(sorted(supports["cdm"].items())),
        },
        "summaries": {"static_family": static_summaries, "cdm_family": cdm_summaries},
        "contrasts": {
            "expression": "BLOCK10 cell minus OLDLAW cell on identical post10 windows",
            "static_family": static_contrasts,
            "cdm_family": cdm_contrasts,
        },
        "gates": {
            "law": dict(plan.GATES),
            "primary_cdm_k4_external": primary_gate,
            "secondary": secondary_gates,
            "within_disclosures": within_disclosures,
        },
        "census": {
            "law": plan.REBLOCK_LAW["evidence_census"],
            "within": dict(sorted(census["within"].items())),
            "external": dict(sorted(census["external"].items())),
            "summary": census_summary,
        },
        "sealed_context": sealed_context,
        "rows": {
            "static_family": {surface: dict(sorted(sessions.items()))
                              for surface, sessions in static_rows.items()},
            "cdm_family": {surface: dict(sorted(sessions.items()))
                           for surface, sessions in cdm_rows.items()},
        },
        "anchors": {
            "law": {key: (dict(value) if isinstance(value, dict) else value)
                    for key, value in plan.ANCHORS.items()},
            "static_pairing": static_pairing,
            "cdm_pairing": cdm_pairing,
            "static_executor_fidelity": static_fidelity,
            "cdm_executor_fidelity": cdm_fidelity,
            "dopt_proofs": dopt_proofs,
            "kall_binding_parity": kall_parity,
            "support_agreement_across_families": support_agreement,
            "pure_data_pairing_all_exact": bool(pairing_all_exact),
            "executor_fidelity_all_exact": bool(fidelity_all_exact),
            "dopt_proofs_all_exact": bool(dopt_proofs_all_exact),
            "kall_binding_parity_all": bool(kall_parity_all),
            "fidelity_roster": {"static": fidelity_roster_static, "cdm": fidelity_roster_cdm},
        },
        "stop_conditions": {
            "any_fired": bool(governing),
            "fired": governing,
            "conditions": stop_conditions,
        },
        "target_optimizer_backward_update": 0,
        "target_backward_calls": 0,
        "target_parameter_update_calls": 0,
        "decoder_training": False,
        "model_or_checkpoint_updated": False,
        "wall_seconds": float(time.monotonic() - started),
        "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    _require(pairing_all_exact, "at least one pure-data pairing failed")
    _require(fidelity_all_exact, "at least one executor-fidelity anchor failed")
    _require(dopt_proofs_all_exact, "at least one D-opt selection proof failed")
    _require(kall_parity_all, "at least one KALL binding parity check failed")
    replay_digest = _publish(root / "replay.json", replay_payload)
    terminal_payload = {
        "schema": f"{plan.SCHEMA}_terminal_v1",
        "status": "TERMINAL",
        "scope": "inference_only_first10_pool_post10_stream_local_m2_cpu",
        "replay_sha256": replay_digest,
        "attempt_sha256": attempt_digest,
        "equal_session_means": {
            "static_family": {
                surface: {cell: static_summaries[surface][cell]["equal_session_mean"]
                          for cell in static_summaries[surface]}
                for surface in static_summaries
            },
            "cdm_family": {
                surface: {cell: cdm_summaries[surface][cell]["equal_session_mean"]
                          for cell in cdm_summaries[surface]}
                for surface in cdm_summaries
            },
        },
        "primary_gate_cdm_k4_external": primary_gate,
        "primary_verdict": primary_gate["verdict"],
        "secondary_gates": secondary_gates,
        "within_disclosures": within_disclosures,
        "paired_contrasts": {
            "cdm_k4_external": cdm_contrasts["external_post10_local"]["BLOCK10_CDM_K4"],
            "cdm_kall_external": cdm_contrasts["external_post10_local"]["BLOCK10_CDM_KALL"],
            "static_k4_external": static_contrasts["external_post10_query"]["BLOCK10_STATIC_K4"],
            "static_kall_external": static_contrasts["external_post10_query"]["BLOCK10_STATIC_KALL"],
        },
        "census_summary": census_summary,
        "census_external": dict(sorted(census["external"].items())),
        "supports_headline": {
            f"static|{key}": {
                "k4": value["k4_support_positions"],
                "kall": value["kall_support_positions"],
                "candidates": value["first10_candidates"],
            }
            for key, value in sorted(supports["static"].items())
        },
        "sealed_context": sealed_context,
        "anchors": {
            "pure_data_pairing_all_exact": bool(pairing_all_exact),
            "executor_fidelity_all_exact": bool(fidelity_all_exact),
            "dopt_proofs_all_exact": bool(dopt_proofs_all_exact),
            "kall_binding_parity_all": bool(kall_parity_all),
            "fidelity_max_abs_r2_delta": stop_conditions["anchor_or_binding_failure"][
                "evidence"]["fidelity_max_abs_r2_delta"],
        },
        "stop_conditions": replay_payload["stop_conditions"],
        "official_contract_claimed": False,
        "wall_seconds": replay_payload["wall_seconds"],
    }
    _publish(root / "terminal.json", terminal_payload)
    return {
        "status": terminal_payload["status"],
        "replay_sha256": replay_digest,
        "primary_verdict": primary_gate["verdict"],
        "primary_delta": primary_gate["delta"],
        "primary_positive_sessions": primary_gate["positive_sessions"],
        "cdm_kall_verdict": secondary_gates["cdm_kall_external"]["verdict"],
        "cdm_kall_delta": secondary_gates["cdm_kall_external"]["delta"],
        "static_k4_delta": secondary_gates["static_k4_external"]["delta"],
        "static_kall_delta": secondary_gates["static_kall_external"]["delta"],
        "external_extension_ratio_range": [
            census_summary["external_extension_ratio_min"],
            census_summary["external_extension_ratio_max"],
        ],
        "stop_conditions_fired": governing,
        "wall_seconds": replay_payload["wall_seconds"],
    }
