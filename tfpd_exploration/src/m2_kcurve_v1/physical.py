"""Physical driver of the M2 labeled-pair k-curve (B30 pool, D-opt prefix).

CPU-only, inference-only, one launch.  The receipt law mirrors the sealed
routes and the chrono4/reblock10 precedents: ``attempt.json`` (reserved
before any data), ``replay.json`` (the governing grid), ``terminal.json``
(receipt-only composition).  Nothing under any frozen result root is created
or modified; every sealed executor is reused by import.
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


class KCurvePhysicalError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KCurvePhysicalError(message)


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
             "the k-curve is CPU-only (CUDA_VISIBLE_DEVICES must be empty)")
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


# ---------------------------------------------------------------------------
# Sealed anchor receipts (read-only).
# ---------------------------------------------------------------------------


def _load_sealed_static_rows(repo_root: Path) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    path = repo_root / plan.BUDGET_SCREEN_SCORE_RELATIVE
    _require(_sha256_file(path) == plan.BUDGET_SCREEN_SCORE_SHA256,
             "sealed activity-budget screen score drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == "m2_t4_activity_budget_screen_v1"
             and payload.get("status") == "TERMINAL",
             "sealed activity-budget screen schema drift")
    rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in payload["rows"]:
        if row.get("cell") in ("ridge_activity30_m4", "ridge_static_m30"):
            rows[(str(row["cell"]), str(row["surface"]), str(row["session"]))] = row
    _require(len(rows) == 2 * (plan.EXPECTED_WITHIN_SESSIONS + plan.EXPECTED_EXTERNAL_SESSIONS),
             "sealed static anchor row topology drift")
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


def _load_g_context(repo_root: Path) -> dict[str, Any]:
    path = repo_root / plan.G_TERMINAL_RELATIVE
    _require(_sha256_file(path) == plan.G_TERMINAL_SHA256,
             "sealed G-family terminal receipt drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == "cdm_p1_m2_local_v1_terminal_v1"
             and payload.get("status") == "TERMINAL", "sealed G-family schema drift")
    g00m = payload["g_family_equal_session_means"]
    return {
        "g00m_equal_session_means": {
            surface: float(g00m[surface]["G00m"]["equal_session_mean"])
            for surface in ("within_post30", "external_post30_local")
        },
        "g01m_equal_session_means": {
            surface: float(g00m[surface]["G01m"]["equal_session_mean"])
            for surface in ("within_post30", "external_post30_local")
        },
    }


# ---------------------------------------------------------------------------
# Session window starts (the sealed post-30 partition law).
# ---------------------------------------------------------------------------


def _session_windows(dataset: Any, session: str) -> np.ndarray:
    starts = np.asarray(
        [start for name, start in dataset.window_indices if name == session],
        dtype=np.int64,
    )
    _require(starts.size > 0 and np.all(np.diff(starts) > 0),
             f"{session}: query window order drift")
    return starts


def _post30_starts(dataset: Any, session: str) -> np.ndarray:
    """The sealed post-30 partition (both families' scored surface law)."""
    from src.m2_t4_activity_budget_screen_v1.core import (
        select_common_post30_window_starts,
    )

    return select_common_post30_window_starts(
        _session_windows(dataset, session),
        np.asarray(dataset.trial_start_indices[session], dtype=np.int64),
    )


def _official_query_starts(dataset: Any, session: str) -> np.ndarray:
    """The sealed static family's unfiltered external fidelity surface."""
    return np.ascontiguousarray(_session_windows(dataset, session))


# ---------------------------------------------------------------------------
# The STATIC family executor (the sealed ridge_activity30 law, carrier at k).
# ---------------------------------------------------------------------------


def static_act30_cell(
    *, torch: Any, student: Any, dataset: Any, session: str, starts: np.ndarray,
    selected: np.ndarray, digest_fn: Any, r2_fn: Any, batch_size: int,
    selection_label: str,
) -> dict[str, Any]:
    """The sealed act30 law: full first-30 activity, carrier from the k pairs.

    The activity is k-independent (the sealed ``select_activity_rows`` act30
    branch spelled directly); ONLY the carrier support moves with k.  The
    prefix supports are all-directional by construction, so the sealed
    usable-mask inside the fit is the identity here.
    """
    from tfpd_exploration.src.calibration_budget_comparators_v1 import fit_ridge_t4

    indices = np.asarray(selected, dtype=np.int64).reshape(-1)
    _require(indices.size >= plan.M4 and int(indices.max()) < plan.ACTIVITY_HORIZON,
             "the static k-curve support must lie in the first-30 pool")
    calibration = np.asarray(dataset.calib_trialized_neural_features[session], dtype=np.float32)
    activity = laws.act30_activity_rows(calibration)
    _require(activity.shape == (plan.ACTIVITY_HORIZON, plan.TRIAL_LENGTH, plan.CHANNELS)
             and np.isfinite(activity).all(), "static act30 activity stack drift")
    sums = np.asarray(dataset.calib_trial_spike_sums[session][indices], dtype=np.float64)
    lengths = np.asarray(dataset.calib_trial_lengths[session][indices], dtype=np.float64)
    angles = np.asarray(dataset.calib_trial_target_angles[session][indices], dtype=np.float64)
    _require(bool(np.isfinite(angles).all()) and int(angles.size) == int(indices.size) >= 3,
             f"{session}: the k-curve support must be all-directional with >= 3 rows")
    rates = np.ascontiguousarray(sums / lengths[:, None], dtype=np.float64)
    raw, evidence = fit_ridge_t4(
        rates, angles, normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA,
    )
    mean = np.asarray(dataset.side_feature_mean, dtype=np.float32)
    std = np.asarray(dataset.side_feature_std, dtype=np.float32)
    _require(mean.shape == std.shape == (4,) and np.all(std > 0),
             "frozen T4 normalizer drift")
    side = np.ascontiguousarray((raw - mean) / std, dtype=np.float32)
    _require(side.shape == (plan.CHANNELS, 4) and np.isfinite(side).all(),
             "normalized k-curve T4 drift")
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
        "budget": int(indices.size),
        "support_positions": indices.tolist(),
        "activity_law": "act30_full_first30_block (the sealed select_activity_rows act30 branch spelled directly)",
        "side_evidence": {
            **{key: evidence[key] for key in (
                "normalized_lambda", "design_rank", "design_condition", "gcv")},
            "selection": selection_label,
            "selected_indices": indices.tolist(),
            "selected_indices_sha256": digest_fn(indices),
            "usable_directional_trials": int(indices.size),
            "raw_t4_sha256": digest_fn(raw),
            "normalized_t4_sha256": digest_fn(side),
        },
        "r2": float(r2_fn(target, prediction)),
        "parameter_updates": 0,
    }


# ---------------------------------------------------------------------------
# The CDM family executor (the sealed m4_activity_only law, support at k).
# ---------------------------------------------------------------------------


def _krow_carrier_and_activity(
    support: Mapping[str, Any], indices: np.ndarray,
) -> tuple[np.ndarray, Any]:
    """The generalized k-row carrier/activity binding (the sealed arithmetic).

    The sealed ``_build_memory`` path guards the support cardinality to
    M4/M10/M30; the activity-only law it implements is (a) the
    fixed-ridge-by-trial carrier fit over the support rows (Hz domain) and
    (b) a B3S activity memory of support k + FIFO capacity 30-k (the frozen
    30-trial stack ceiling).  This binding spells exactly that for any k in
    [4, 30] and is parity-anchored bitwise to the sealed binding at k=4.
    """
    from tfpd_exploration.src.causal_dual_memory_cell_d_v1 import core as tfpd_cdm_core
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core as pseudo_core

    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    k = int(selected.size)
    rates = np.ascontiguousarray(
        support["support_rates_hz30"][selected], dtype=np.float64)
    theta = np.ascontiguousarray(support["theta30"][selected], dtype=np.float64)
    _require(bool(np.isfinite(theta).all()), "the k-row support includes a non-directional row")
    directions = pseudo_core.canonical_direction_indices(theta)
    fitted = tfpd_cdm_core.fit_carriers_from_trial_table(
        rates, directions,
        mode=tfpd_cdm_core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA,
    )
    carrier_hz = np.asarray(fitted, dtype=np.float64)
    activity = tfpd_cdm_core.ActivityMemory.initialize(
        tuple(support["support_b3s"][index] for index in selected),
        channel_ids=support["channels"],
        fifo_capacity=plan.ACTIVITY_STACK_LIMIT - k,
    )
    return carrier_hz, activity


def cdm_k_cell(
    *, torch: Any, model: Any, dataset: Any, session: str, surface: str,
    query_rows: tuple[dict[str, Any], ...], support: Mapping[str, Any],
    raw_neural: np.ndarray, selected: np.ndarray, side_mean: np.ndarray,
    side_std: np.ndarray, device: Any, batch_size: int, digest_fn: Any,
    r2_fn: Any,
) -> dict[str, Any]:
    """The sealed ``m4_activity_only`` law over a D-opt-prefix k support.

    k=4 binds the sealed ``_build_memory`` path verbatim; k != 4 binds the
    generalized k-row arithmetic (parity-anchored at k=4).  The FIFO advances
    on EVERY completed post-30 query trial (the sealed V3 short-trial law);
    supports lie inside positions 0-29, so the support/stream overlap is
    empty by construction (asserted).
    """
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical as cdm_physical
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import plan as cdm_plan
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import physical as shared_physical

    indices = np.asarray(selected, dtype=np.int64).reshape(-1)
    k = int(indices.size)
    _require(plan.M4 <= k <= plan.ACTIVITY_STACK_LIMIT
             and int(indices.max()) < plan.ACTIVITY_HORIZON,
             "the CDM k-curve support must lie in the first-30 pool")
    if k == plan.M4:
        memory = cdm_physical._build_memory(
            system="activity_only", budget=plan.M4, selected=indices,
            support_rates_hz30=support["support_rates_hz30"],
            theta_first30=support["theta30"], support_b3s=support["support_b3s"],
            channel_ids=support["channels"], valid_mask=support["valid_mask"],
        )
        binding = "sealed_build_memory_m4"
        krow_carrier = None
    else:
        carrier_hz_fixed, activity = _krow_carrier_and_activity(support, indices)
        binding = "generalized_k_row_binding"
        krow_carrier = np.ascontiguousarray(carrier_hz_fixed, dtype=np.float64)
    neural = np.asarray(dataset.neural_data[session], dtype=np.float32)
    behavior = np.asarray(dataset.covariate_data[session], dtype=np.float32)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    starts_joined: list[np.ndarray] = []
    short_trials = 0
    support_positions = {int(item) for item in indices.tolist()}
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
        _require(position not in support_positions,
                 "a first-30 support position entered the post-30 evidence stream")
        b3s, _native, _validity = cdm_physical._trial_capabilities(
            session=session, row=row, raw_neural=raw_neural,
            channel_ids=support["channels"],
        )
        advance_events += 1
        if k == plan.M4:
            memory.commit_activity_only(b3s)
        else:
            activity = activity.after_completed_trial(b3s)
    prediction = np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32)
    target = np.ascontiguousarray(np.concatenate(targets, axis=0), dtype=np.float32)
    starts = np.ascontiguousarray(np.concatenate(starts_joined), dtype=np.int64)
    expected = cdm_physical._surface_starts(dataset, session)
    _require(np.array_equal(starts, expected),
             f"{surface}: the scored query differs from the sealed post30 authority")
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
        "query_row_count": len(query_rows),
        "short_query_trials_without_complete_50_bin_window": int(short_trials),
        "carrier_hz_sha256": digest_fn(
            krow_carrier if krow_carrier is not None
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
                 f"an owned k-curve module drifted: {relative}")

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
    g_context = _load_g_context(repo_root)

    train_ds = data_module.train_dataset
    heldout_ds = data_module.val_heldout_dataset
    _require(train_ds is not None and heldout_ds is not None,
             "the frozen M2 runtime must expose both datasets")
    for dataset in (train_ds, heldout_ds):
        _require(np.array_equal(np.asarray(dataset.side_feature_mean, dtype=np.float32), side_mean)
                 and np.array_equal(np.asarray(dataset.side_feature_std, dtype=np.float32), side_std),
                 "dataset/metadata T4 normalizer disagreement")
    static_datasets = {"within_post30": train_ds, "external_post30_local": heldout_ds}
    cdm_datasets = {
        "within_post30": (train_ds, data_module.train_calib_heldin_sessions),
        "external_post30_local": (heldout_ds, data_module.val_calib_heldout_sessions),
    }
    expected_counts = {
        "within_post30": plan.EXPECTED_WITHIN_SESSIONS,
        "external_post30_local": plan.EXPECTED_EXTERNAL_SESSIONS,
    }
    started = time.monotonic()

    # -- Phase A: angle pools, usable census, then the prefix supports --------
    supports: dict[str, dict[str, dict[str, Any]]] = {"static": {}, "cdm": {}}
    dopt_proofs: dict[str, Any] = {}
    cdm_materials: dict[str, dict[str, Any]] = {}
    thetas: dict[str, np.ndarray] = {}
    usable_census: dict[str, int] = {}
    for family, datasets in (("static", static_datasets), ("cdm", cdm_datasets)):
        for surface, binding in datasets.items():
            dataset = binding[0] if family == "cdm" else binding
            raw_sessions = binding[1] if family == "cdm" else None
            sessions = sorted(dataset.calib_trialized_neural_features)
            _require(len(sessions) == expected_counts[surface], f"{surface}: session count drift")
            for session in sessions:
                theta = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
                thetas[f"static|{surface}|{session}"] = theta
                usable_census[f"static|{surface}|{session}"] = laws.usable_count(theta)
                if family == "cdm":
                    _require(session in raw_sessions, f"{surface}/{session}: raw authority absent")
                    views = g_replay._g_session_views(raw_sessions, session=session)
                    support_base = g_replay.g_support_material(session=session, views=views)
                    cdm_materials[f"{surface}|{session}"] = {
                        "views": views, "support": support_base,
                    }
                    theta30 = np.asarray(support_base["theta30"], dtype=np.float64)
                    _require(theta30.shape == (plan.ACTIVITY_HORIZON,),
                             f"{surface}|{session}: the raw theta30 topology drifted")
                    thetas[f"cdm|{surface}|{session}"] = theta30
                    usable_census[f"cdm|{surface}|{session}"] = laws.usable_count(theta30)

    # the usable-cap law runs BEFORE the prefix law so a short roster is a
    # disclosed cap, never an opaque raise
    cap = laws.usable_cap(usable_census)
    k_grid = [int(item) for item in cap["effective_grid"]]
    endpoint_k = int(cap["endpoint_k"])
    _require(k_grid == sorted(k_grid) and k_grid[0] >= plan.M4 and endpoint_k == k_grid[-1],
             "the effective k grid must ascend from the M4 anchor to the endpoint")

    # the static anchors' sealed surface spelling (the budget screen's
    # external rows live on external_official_query, this run's external
    # surface is external_post30_local)
    sealed_static_surface = {
        "within_post30": "within_post30",
        "external_post30_local": "external_official_query",
    }

    for key, theta_pool in sorted(thetas.items()):
        spelling, surface, session = key.split("|", 2)
        entry = laws.prefix_nesting(theta_pool, k_grid)
        _require(entry["prefix_exact"] and entry["prefix_nested"],
                 f"{key}: the D-opt prefix law failed")
        _require(entry["k4_prefix_is_sealed_m4_law"],
                 f"{key}: the first-4 prefix is not the sealed M4 D-opt law")
        endpoint = laws.endpoint_support(theta_pool)
        _require(int(entry["usable"]) == usable_census[key],
                 f"{key}: the usable census disagrees with the prefix law")
        _require(np.array_equal(
            np.asarray(entry["supports"][str(endpoint_k)], dtype=np.int64), endpoint),
            f"{key}: the endpoint grid support is not the all-usable set")
        if spelling == "static":
            sealed_row = sealed_static[
                ("ridge_activity30_m4", sealed_static_surface[surface], session)]
            proof = laws.dopt_proof(
                entry["supports"][str(plan.M4)],
                sealed_row["side_evidence"]["selected_indices"])
            sealed_selection = str(sealed_row["side_evidence"]["selection"])
        else:
            sealed_row = sealed_cdm[(surface, session)]
            proof = laws.dopt_proof(
                entry["supports"][str(plan.M4)], sealed_row["support_indices"])
            proof["g_support_material_agrees"] = bool(np.array_equal(
                np.asarray(cdm_materials[f"{surface}|{session}"]["support"]["selected"],
                           dtype=np.int64),
                np.asarray(entry["supports"][str(plan.M4)], dtype=np.int64)))
            sealed_selection = "doptimal_noncentre_first30"
        dopt_proofs[key] = {
            **proof,
            "sealed_law_selection": sealed_selection,
            "greedy_order_positions": entry["greedy_order_positions"],
            "usable": entry["usable"],
        }
        _require(dopt_proofs[key]["exact_match"],
                 f"the sealed M4 indices are not the first-4 D-opt prefix at {key}")
        _require(dopt_proofs[key].get("g_support_material_agrees", True),
                 f"{key}: g_support drift")
        supports[spelling][f"{surface}|{session}"] = entry
        _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                 "the hard timeout fired during the support phase")

    support_agreement = laws.cross_family_support_agreement(
        supports["static"], supports["cdm"])

    # -- Phase B: the STATIC family ------------------------------------------
    static_rows: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        surface: {} for surface in plan.SURFACES}
    act30_parity: dict[str, Any] = {}
    for surface, dataset in static_datasets.items():
        for session in sorted(dataset.calib_trialized_neural_features):
            starts = _post30_starts(dataset, session)
            entry = supports["static"][f"{surface}|{session}"]
            order = np.asarray(entry["greedy_order_positions"], dtype=np.int64)
            calibration = np.asarray(
                dataset.calib_trialized_neural_features[session], dtype=np.float32)
            for k in k_grid:
                if k in (plan.M4, plan.M10):
                    parity = laws.act30_parity_with_sealed_helper(
                        calibration, np.sort(order[:k]))
                    _require(parity["bitwise_equal"],
                             f"{surface}/{session}: act30 direct spelling drifted at k={k}")
                    act30_parity[f"{surface}|{session}|k{k}"] = parity
                row = static_act30_cell(
                    torch=torch, student=student, dataset=dataset, session=session,
                    starts=starts, selected=np.sort(order[:k]),
                    digest_fn=static_digest, r2_fn=static_r2, batch_size=batch_size,
                    selection_label=f"doptimal_prefix_k{k}_first30",
                )
                static_rows[surface].setdefault(session, {})[f"STATIC_K{k}"] = {
                    "surface": surface, "session": session,
                    "cell": f"STATIC_K{k}", **row}
                _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                         "the hard timeout fired during the static family")

    # -- Phase C: the CDM family ----------------------------------------------
    cdm_rows: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        surface: {} for surface in plan.SURFACES}
    krow_parity: dict[str, Any] = {}
    for surface, (dataset, raw_sessions) in cdm_datasets.items():
        for session in sorted(dataset.calib_trialized_neural_features):
            material = cdm_materials[f"{surface}|{session}"]
            support_base = material["support"]
            views = material["views"]
            raw_neural, raw_starts, _theta, _rates, activities = views
            entry = supports["cdm"][f"{surface}|{session}"]
            order = np.asarray(entry["greedy_order_positions"], dtype=np.int64)
            query_rows = cdm_physical._query_trial_rows(
                dataset, session, raw_neural=raw_neural,
                raw_starts=raw_starts, activities=activities,
            )
            for k in k_grid:
                row = cdm_k_cell(
                    torch=torch, model=model, dataset=dataset, session=session,
                    surface=surface, query_rows=query_rows, support=support_base,
                    raw_neural=raw_neural, selected=np.sort(order[:k]),
                    side_mean=side_mean, side_std=side_std,
                    device=device, batch_size=batch_size,
                    digest_fn=pseudo_core.array_sha256,
                    r2_fn=pseudo_core.variance_weighted_r2,
                )
                cdm_rows[surface].setdefault(session, {})[f"CDM_K{k}"] = {
                    "surface": surface, "session": session,
                    "cell": f"CDM_K{k}", **row}
                _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                         "the hard timeout fired during the cdm family")
            # the k-row binding parity anchor: driven with the k=4 prefix
            # support it must reproduce the sealed _build_memory carrier AND
            # the initial activity stack bit-for-bit
            sealed_memory = cdm_physical._build_memory(
                system="activity_only", budget=plan.M4,
                selected=np.sort(order[:plan.M4]),
                support_rates_hz30=support_base["support_rates_hz30"],
                theta_first30=support_base["theta30"],
                support_b3s=support_base["support_b3s"],
                channel_ids=support_base["channels"],
                valid_mask=support_base["valid_mask"],
            )
            krow_carrier, krow_activity = _krow_carrier_and_activity(
                support_base, np.sort(order[:plan.M4]))
            parity = {
                "law": plan.ANCHORS["krow_binding_parity"]["law"],
                "carrier_bitwise_equal": bool(np.array_equal(
                    np.ascontiguousarray(krow_carrier, dtype=np.float64),
                    np.ascontiguousarray(
                        np.asarray(sealed_memory.carrier.active_t4), dtype=np.float64))),
                "initial_activity_stack_bitwise_equal": bool(np.array_equal(
                    krow_activity.stack(), sealed_memory.activity.stack())),
            }
            krow_parity[f"{surface}|{session}"] = parity
            _require(parity["carrier_bitwise_equal"]
                     and parity["initial_activity_stack_bitwise_equal"],
                     f"the generalized k-row binding drifted at {surface}/{session}")

    # -- Phase D: anchors ------------------------------------------------------
    tolerance = float(plan.ANCHORS["direct_surface_anchors"]["r2_tolerance"])

    def _anchor(cell_row: Mapping[str, Any], sealed_row: Mapping[str, Any]) -> dict[str, Any]:
        anchored = laws.fidelity_anchor(cell_row, sealed_row, r2_tolerance=tolerance)
        _require(anchored["exact_match"], (
            "anchor failed: "
            f"{cell_row.get('surface')}/{cell_row.get('session')}/{cell_row.get('cell')} "
            f"vs sealed r2={sealed_row.get('r2')}"))
        return anchored

    direct_anchors: dict[str, Any] = {}
    for session in sorted(static_rows["within_post30"]):
        direct_anchors[f"static_k4_within|{session}"] = _anchor(
            static_rows["within_post30"][session][f"STATIC_K{plan.K_ANCHOR_LOW}"],
            sealed_static[("ridge_activity30_m4", "within_post30", session)])
        direct_anchors[f"static_kall_within|{session}"] = _anchor(
            static_rows["within_post30"][session][f"STATIC_K{endpoint_k}"],
            sealed_static[("ridge_static_m30", "within_post30", session)])
    for surface in plan.SURFACES:
        for session in sorted(cdm_rows[surface]):
            direct_anchors[f"cdm_k4_{surface}|{session}"] = _anchor(
                cdm_rows[surface][session][f"CDM_K{plan.K_ANCHOR_LOW}"],
                sealed_cdm[(surface, session)])
    static_fidelity: dict[str, Any] = {}
    for session in sorted(heldout_ds.calib_trialized_neural_features):
        entry = supports["static"][f"external_post30_local|{session}"]
        order = np.asarray(entry["greedy_order_positions"], dtype=np.int64)
        starts = _official_query_starts(heldout_ds, session)
        for label, k, sealed_cell in (
            ("k4", plan.K_ANCHOR_LOW, "ridge_activity30_m4"),
            ("kall", endpoint_k, "ridge_static_m30"),
        ):
            row = static_act30_cell(
                torch=torch, student=student, dataset=heldout_ds, session=session,
                starts=starts, selected=np.sort(order[:k]),
                digest_fn=static_digest, r2_fn=static_r2, batch_size=batch_size,
                selection_label=f"doptimal_prefix_k{k}_first30",
            )
            row.update({"surface": "external_official_query", "session": session,
                        "cell": f"STATIC_FIDELITY_{label.upper()}"})
            static_fidelity[f"{label}|{session}"] = {
                "row": row,
                "anchor": _anchor(row, sealed_static[(sealed_cell, "external_official_query", session)]),
            }
            _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                     "the hard timeout fired during the static fidelity anchors")
    # the sealed G-family lineage: G00m means == the sealed CDM screen means
    g_means = g_context["g00m_equal_session_means"]
    sealed_cdm_means = {
        surface: laws.equal_session_mean({
            session: float(sealed_cdm[(surface, session)]["r2"])
            for session in sorted(cdm_rows[surface])
        })
        for surface in plan.SURFACES
    }
    g_lineage = {
        surface: {
            "g00m_mean": float(g_means[surface]),
            "sealed_cdm_screen_mean": sealed_cdm_means[surface],
            "exact_match": bool(abs(float(g_means[surface]) - sealed_cdm_means[surface]) <= 1.0e-12),
        }
        for surface in plan.SURFACES
    }
    _require(all(item["exact_match"] for item in g_lineage.values()),
             "the sealed G-family lineage drifted from the CDM screen anchor")

    # pure-data pairing within every family/surface/session across all k
    pairing: dict[str, Any] = {}
    for family, family_rows in (("static", static_rows), ("cdm", cdm_rows)):
        for surface, sessions in family_rows.items():
            for session, cells in sessions.items():
                names = sorted(cells)
                reference = cells[names[0]]
                for name in names[1:]:
                    matches = {
                        "window_count": cells[name]["window_count"] == reference["window_count"],
                        "starts_digest": (
                            cells[name]["ordered_window_starts_sha256"]
                            == reference["ordered_window_starts_sha256"]),
                        "target_sha256": (cells[name]["target_sha256"]
                                          == reference["target_sha256"]),
                    }
                    pairing[f"{family}|{surface}|{session}|{name}"] = {
                        "field_matches": matches,
                        "exact_match": all(bool(value) for value in matches.values()),
                    }
                    _require(pairing[f"{family}|{surface}|{session}|{name}"]["exact_match"],
                             f"the pure-data pairing failed at {family}/{surface}/{session}/{name}")

    # -- Phase E: the pre-registered readouts ---------------------------------
    def _cell_values(family_rows: Mapping[str, Any], surface: str, cell: str) -> dict[str, float]:
        return {session: float(family_rows[surface][session][cell]["r2"])
                for session in sorted(family_rows[surface])}

    kcurves: dict[str, Any] = {}
    for family, family_rows in (("static", static_rows), ("cdm", cdm_rows)):
        for surface in plan.SURFACES:
            per_k = {
                int(k): _cell_values(family_rows, surface, f"{family.upper()}_K{k}")
                for k in k_grid
            }
            kcurves[f"{family}|{surface}"] = {
                "k_curve_table": laws.kcurve_table(per_k, endpoint_k=endpoint_k),
                "saturation": laws.saturation_point(per_k, endpoint_k=endpoint_k),
                "monotonicity": laws.monotonicity(per_k, endpoint_k=endpoint_k),
            }
    verdicts = {
        key: {
            "k_star": value["saturation"]["k_star"],
            "k_star_deficit": value["saturation"]["k_star_deficit"],
            "endpoint_mean": value["saturation"]["endpoint_mean"],
            "verdict": value["saturation"]["verdict"],
        }
        for key, value in kcurves.items()
    }

    anchors_summary = {
        "dopt_proofs_all_exact": bool(all(
            item["exact_match"] for item in dopt_proofs.values())),
        "direct_surface_anchors_all_exact": bool(all(
            item["exact_match"] for item in direct_anchors.values())),
        "static_fidelity_all_exact": bool(all(
            item["anchor"]["exact_match"] for item in static_fidelity.values())),
        "krow_binding_parity_all": bool(all(
            item["carrier_bitwise_equal"] and item["initial_activity_stack_bitwise_equal"]
            for item in krow_parity.values())),
        "act30_binding_parity_all": bool(all(
            item["bitwise_equal"] for item in act30_parity.values())),
        "pure_data_pairing_all_exact": bool(all(
            item["exact_match"] for item in pairing.values())),
        "cross_family_support_agreement": bool(
            all(item["agree"] for item in support_agreement.values())),
        "g_family_lineage_exact": bool(all(
            item["exact_match"] for item in g_lineage.values())),
        "fidelity_max_abs_r2_delta": max(
            [abs(item["anchor"]["r2_delta"]) for item in static_fidelity.values()]
            + [abs(item["r2_delta"]) for item in direct_anchors.values()]),
        "direct_anchor_max_abs_r2_delta": max(
            abs(item["r2_delta"]) for item in direct_anchors.values()),
    }
    stop_conditions = {
        "anchor_or_binding_failure": {
            "expression": (
                "any D-opt prefix proof, direct/fidelity anchor, k-row parity, "
                "act30 parity, pure-data pairing, support agreement or G-family "
                "lineage check failed"),
            "fired": not all(bool(value) for key, value in anchors_summary.items()
                             if key.endswith(("_all_exact", "_all", "_exact"))),
            "evidence": anchors_summary,
        },
    }
    governing = [name for name, item in stop_conditions.items() if item["fired"]]

    sealed_context = {
        "disclosure": (
            "sealed receipt equal-session means (read-only context; the "
            "k-curve endpoints are the MEASURED cells on their own surfaces)"
        ),
        "static_k4_ridge_activity30_m4": {
            surface: laws.equal_session_mean({
                session: float(sealed_static[("ridge_activity30_m4", surface, session)]["r2"])
                for session in sorted(
                    train_ds.calib_trialized_neural_features
                    if surface == "within_post30"
                    else heldout_ds.calib_trialized_neural_features)
            })
            for surface in ("within_post30", "external_official_query")
        },
        "static_kall_ridge_static_m30": {
            surface: laws.equal_session_mean({
                session: float(sealed_static[("ridge_static_m30", surface, session)]["r2"])
                for session in sorted(
                    train_ds.calib_trialized_neural_features
                    if surface == "within_post30"
                    else heldout_ds.calib_trialized_neural_features)
            })
            for surface in ("within_post30", "external_official_query")
        },
        "cdm_k4_m4_activity_only": sealed_cdm_means,
        "g00m_means": g_means,
    }

    replay_payload = {
        "schema": f"{plan.SCHEMA}_replay_v1",
        "stage": "inference_only_b30_pool_dopt_prefix_kcurve_local_m2_cpu",
        "status": "REPLAY_COMPLETE",
        "attempt_sha256": attempt_digest,
        "authority": (
            "operator work order 2026-09-02: the M2 labeled-pair k-curve (B30 "
            "calibration block fixed, D-optimal selection at every k), "
            "CPU-only, sealed anchors read from receipts"
        ),
        "foundation": {
            "sealed_budget_screen_score": plan.BUDGET_SCREEN_SCORE_RELATIVE,
            "sealed_budget_screen_score_sha256": plan.BUDGET_SCREEN_SCORE_SHA256,
            "sealed_comparator_score": plan.COMPARATOR_SCORE_RELATIVE,
            "sealed_comparator_score_sha256": plan.COMPARATOR_SCORE_SHA256,
            "sealed_cdm_screen_score": plan.CDM_SCREEN_SCORE_RELATIVE,
            "sealed_cdm_screen_score_sha256": plan.CDM_SCREEN_SCORE_SHA256,
            "sealed_g_terminal": plan.G_TERMINAL_RELATIVE,
            "sealed_g_terminal_sha256": plan.G_TERMINAL_SHA256,
            "sealed_g_package": plan.SEALED_G_PACKAGE,
            "chrono4_precedent": plan.CHRONO4_PACKAGE,
            "reblock10_precedent": plan.REBLOCK10_PACKAGE,
            "reuse_law": "frozen modules imported verbatim; nothing edited",
        },
        "environment": {
            **plan.ENVIRONMENT_LAW,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "batch_size": int(batch_size),
            "cpu_count_visible_to_torch": int(torch.get_num_threads()),
        },
        "surface": dict(plan.SURFACE_LAW),
        "kcurve_law": dict(plan.KCURVE_LAW),
        "frozen_everything_else": dict(plan.FROZEN_EVERYTHING_ELSE),
        "usable_cap": cap,
        "k_grid": k_grid,
        "endpoint_k": endpoint_k,
        "supports": {
            "static": dict(sorted(supports["static"].items())),
            "cdm": dict(sorted(supports["cdm"].items())),
        },
        "dopt_prefix_proofs": dict(sorted(dopt_proofs.items())),
        "kcurves": {
            key: {
                "k_curve_table": value["k_curve_table"],
                "saturation": value["saturation"],
                "monotonicity": value["monotonicity"],
            }
            for key, value in kcurves.items()
        },
        "verdicts": verdicts,
        "readout_law": dict(plan.READOUT_LAW),
        "rows": {
            "static_family": {surface: dict(sorted(sessions.items()))
                              for surface, sessions in static_rows.items()},
            "cdm_family": {surface: dict(sorted(sessions.items()))
                           for surface, sessions in cdm_rows.items()},
        },
        "anchors": {
            "law": {key: (dict(value) if isinstance(value, dict) else value)
                    for key, value in plan.ANCHORS.items()},
            "direct_surface_anchors": dict(sorted(direct_anchors.items())),
            "static_executor_fidelity": dict(sorted(static_fidelity.items())),
            "krow_binding_parity": dict(sorted(krow_parity.items())),
            "act30_binding_parity": dict(sorted(act30_parity.items())),
            "pure_data_pairing": dict(sorted(pairing.items())),
            "support_agreement_across_families": support_agreement,
            "g_family_lineage": g_lineage,
            **anchors_summary,
        },
        "sealed_context": sealed_context,
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
    _require(not governing, f"stop conditions fired before publication: {governing}")
    replay_digest = _publish(root / "replay.json", replay_payload)
    terminal_payload = {
        "schema": f"{plan.SCHEMA}_terminal_v1",
        "status": "TERMINAL",
        "scope": "inference_only_b30_pool_dopt_prefix_kcurve_local_m2_cpu",
        "replay_sha256": replay_digest,
        "attempt_sha256": attempt_digest,
        "k_grid": k_grid,
        "endpoint_k": endpoint_k,
        "usable_cap": {
            "min_usable": cap["min_usable"],
            "capped": cap["capped"],
            "dropped_by_cap": cap["dropped_by_cap"],
            "endpoint_is_all_usable": cap["endpoint_is_all_usable"],
        },
        "equal_session_means": {
            family: {
                surface: {
                    str(k): kcurves[f"{family}|{surface}"]["k_curve_table"][
                        "equal_session_means"][str(k)]
                    for k in k_grid
                }
                for surface in plan.SURFACES
            }
            for family in ("static", "cdm")
        },
        "endpoint_means": {
            f"{family}|{surface}": kcurves[f"{family}|{surface}"]["saturation"]["endpoint_mean"]
            for family in ("static", "cdm") for surface in plan.SURFACES
        },
        "deficits_to_endpoint": {
            f"{family}|{surface}": kcurves[f"{family}|{surface}"]["k_curve_table"][
                "deficit_to_endpoint"]
            for family in ("static", "cdm") for surface in plan.SURFACES
        },
        "saturation": {
            f"{family}|{surface}": {
                "k_star": kcurves[f"{family}|{surface}"]["saturation"]["k_star"],
                "k_star_deficit": kcurves[f"{family}|{surface}"]["saturation"]["k_star_deficit"],
                "tolerance": kcurves[f"{family}|{surface}"]["saturation"]["tolerance"],
                "qualifying_ks": kcurves[f"{family}|{surface}"]["saturation"]["qualifying_ks"],
            }
            for family in ("static", "cdm") for surface in plan.SURFACES
        },
        "monotonicity": {
            f"{family}|{surface}": {
                "mean_monotone_nondecreasing": kcurves[f"{family}|{surface}"]["monotonicity"][
                    "mean_monotone_nondecreasing"],
                "dips": kcurves[f"{family}|{surface}"]["monotonicity"]["dips"],
                "adjacent_deltas": kcurves[f"{family}|{surface}"]["monotonicity"][
                    "adjacent_deltas"],
            }
            for family in ("static", "cdm") for surface in plan.SURFACES
        },
        "verdicts": verdicts,
        "headline_verdict": verdicts["cdm|external_post30_local"]["verdict"],
        "per_session_r2": {
            f"{family}|{surface}": kcurves[f"{family}|{surface}"]["k_curve_table"][
                "per_session_r2"]
            for family in ("static", "cdm") for surface in plan.SURFACES
        },
        "anchors": anchors_summary,
        "anchor_receipt_values": sealed_context,
        "stop_conditions": replay_payload["stop_conditions"],
        "official_contract_claimed": False,
        "wall_seconds": replay_payload["wall_seconds"],
    }
    _publish(root / "terminal.json", terminal_payload)
    return {
        "status": terminal_payload["status"],
        "replay_sha256": replay_digest,
        "k_grid": k_grid,
        "endpoint_k": endpoint_k,
        "verdicts": {key: value["verdict"] for key, value in verdicts.items()},
        "headline_verdict": terminal_payload["headline_verdict"],
        "equal_session_means": terminal_payload["equal_session_means"],
        "saturation": {key: value["k_star"] for key, value in terminal_payload["saturation"].items()},
        "monotone_everywhere": all(
            value["mean_monotone_nondecreasing"]
            for value in terminal_payload["monotonicity"].values()),
        "anchor_max_abs_r2_delta": anchors_summary["fidelity_max_abs_r2_delta"],
        "stop_conditions_fired": governing,
        "wall_seconds": replay_payload["wall_seconds"],
    }
