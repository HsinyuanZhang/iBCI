"""Physical driver of the M2 activity-memory-law scan (CPU-only, inference-only).

One launch, zero training, zero target access.  The receipt law mirrors the
sealed routes: ``attempt.json`` (reserved before any data), ``replay.json``
(the governing grid), ``terminal.json`` (receipt-only composition).  Nothing
under any frozen result root is created or modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from . import gates, memory as memory_law, plan


class MemoryLawScanError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MemoryLawScanError(message)


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
             "the memory-law scan is CPU-only (CUDA_VISIBLE_DEVICES must be empty)")
    _require(os.environ.get("PYTHONNOUSERSITE") == "1",
             "PYTHONNOUSERSITE=1 is part of the environment law")


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


def anchor_matches(cell_row: Mapping[str, Any], sealed_row: Mapping[str, Any],
                    *, r2_tolerance: float) -> dict[str, Any]:
    """The UNIFORM_CAP30 tolerance anchor against one sealed GPU row.

    Pure-data fields (query starts, targets, window count) must match exactly;
    R2 must sit within the pre-registered CPU/GPU tolerance; the prediction
    digest is reported informationally (device-dependent, never expected to
    match across CPU and GPU).
    """
    matches = {
        "query_starts_sha256": (
            str(cell_row["query_starts_sha256"]) == str(sealed_row["query_starts_sha256"])),
        "target_sha256": (
            str(cell_row["target_sha256"]) == str(sealed_row["target_sha256"])),
        "window_count": int(cell_row["window_count"]) == int(sealed_row["window_count"]),
        "r2_within_tolerance": abs(
            float(cell_row["r2"]) - float(sealed_row["r2"])) <= float(r2_tolerance),
    }
    return {
        "field_matches": matches,
        "exact_match": all(bool(value) for value in matches.values()),
        "r2_cpu": float(cell_row["r2"]),
        "r2_sealed_gpu": float(sealed_row["r2"]),
        "r2_delta": float(cell_row["r2"]) - float(sealed_row["r2"]),
        "prediction_sha256_match_across_devices": (
            str(cell_row["prediction_sha256"]) == str(sealed_row["prediction_sha256"])),
    }


def _load_sealed_rows(repo_root: Path) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    path = repo_root / plan.CDM_SCREEN_SCORE_RELATIVE
    _require(_sha256_file(path) == plan.CDM_SCREEN_SCORE_SHA256,
             "sealed CDM screen score drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == "m2_precision_cdm_v2_screen_v1"
             and payload.get("status") == "TERMINAL",
             "sealed CDM screen schema drift")
    rows: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in payload["rows"]:
        if row.get("cell") in ("m4_activity_only", "m10_activity_only"):
            rows[(str(row["surface"]), str(row["session_id"]), int(row["budget"]))] = row
    _require(len(rows) == (plan.EXPECTED_SESSIONS_BY_SURFACE["within_post30"]
                           + plan.EXPECTED_SESSIONS_BY_SURFACE["external_post30_local"])
             * len(plan.BUDGETS),
             "sealed activity_only anchor row topology drift")
    return rows


def _support_for_budget(
    support_base: Mapping[str, Any], budget: int, cdm_physical: Any,
) -> dict[str, Any]:
    support = dict(support_base)
    if budget == plan.M4:
        selected = np.asarray(support_base["selected"], dtype=np.int64)
    elif budget == plan.M10:
        selected = np.asarray(
            cdm_physical._finite_m10_indices(support_base["theta30"]), dtype=np.int64,
        )
    else:
        raise MemoryLawScanError(f"unsupported budget {budget}")
    theta = np.asarray(support_base["theta30"], dtype=np.float64)
    _require(selected.size == budget and int(selected.max()) < plan.ACTIVITY_STACK_LIMIT
             and bool(np.isfinite(theta[selected]).all()),
             f"support selection topology drift at M{budget}")
    support["selected"] = selected
    return support


def _predict_with_identity(
    *, torch: Any, student: Any, neural_windows: np.ndarray, identity: Any,
    batch_size: int,
) -> np.ndarray:
    """The sealed ``_predict`` decode path with a precomputed identity."""
    _require(getattr(student, "decoder_mode", None) == "coupled",
             "the scan requires the coupled decoder")
    _require(getattr(student, "fixed_slot_router", None) is None,
             "the scan forbids fixed-slot routing")
    _require(not hasattr(student.id_encoder, "forward_batch_with_gate"),
             "the scan forbids hidden identity gate semantics")
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, neural_windows.shape[0], batch_size):
            neural = torch.from_numpy(
                np.ascontiguousarray(neural_windows[offset : offset + batch_size]),
            )
            prediction = student.decode_with_identity(neural, identity)
            _require(bool(torch.isfinite(prediction).all().item()),
                     "decode produced nonfinite output")
            values.append(
                prediction[:, -1, :].detach().numpy().astype(np.float32, copy=False)
                / plan.BEHAVIOR_SCALE
            )
    result = np.ascontiguousarray(np.concatenate(values, axis=0), dtype=np.float32)
    _require(result.shape == (neural_windows.shape[0], 2) and np.isfinite(result).all(),
             "prediction topology drift")
    return result


def _side_tensor(torch: Any, carrier_hz: np.ndarray, side_mean: np.ndarray,
                 side_std: np.ndarray) -> Any:
    raw_t4 = np.ascontiguousarray(
        np.asarray(carrier_hz, dtype=np.float64) * plan.MODEL_BIN_SECONDS, dtype=np.float32,
    )
    side_np = np.ascontiguousarray((raw_t4 - side_mean) / side_std, dtype=np.float32)
    _require(side_np.shape == (plan.CHANNELS, 4) and np.isfinite(side_np).all(),
             "static T4 side-feature drift")
    return torch.from_numpy(side_np).unsqueeze(0)


def _uniform_cap30_cell(
    *, torch: Any, model: Any, dataset: Any, session: str, views: tuple,
    support: Mapping[str, Any], query_rows: tuple, side_mean: np.ndarray,
    side_std: np.ndarray, device: Any, batch_size: int, budget: int,
    cdm_physical: Any, pseudo_core: Any, use_sealed_g00m: bool,
) -> dict[str, Any]:
    """UNIFORM_CAP30: the sealed activity_only law, sealed executors verbatim."""
    if use_sealed_g00m:
        from src.cdm_p1_m2_local_v1 import replay as g_replay

        result = g_replay.rollout_g00m(
            torch=torch, model=model, ds=dataset, session=session, views=views,
            support=support, query_rows=query_rows, side_mean=side_mean,
            side_std=side_std, device=device, batch_size=batch_size,
        )
    else:
        memory = cdm_physical._build_memory(
            system="activity_only", budget=budget, selected=support["selected"],
            support_rates_hz30=support["support_rates_hz30"],
            theta_first30=support["theta30"],
            support_b3s=support["support_b3s"],
            channel_ids=support["channels"], valid_mask=support["valid_mask"],
        )
        result = cdm_physical._score_system(
            torch=torch, model=model, dataset=dataset, session=session,
            memory=memory, query_rows=query_rows, side_mean=side_mean,
            side_std=side_std, raw_neural=views[0], device=device,
            batch_size=batch_size,
        )
    completed = len(query_rows)
    fifo_capacity = plan.ACTIVITY_STACK_LIMIT - int(budget)
    return {
        "policy": "UNIFORM_CAP30",
        "law": dict(plan.POLICIES["UNIFORM_CAP30"]),
        "budget": int(budget),
        "r2": float(result["r2"]),
        "window_count": int(result["window_count"]),
        "query_starts_sha256": str(result["query_starts_sha256"]),
        "target_sha256": str(result["target_sha256"]),
        "prediction_sha256": str(result["prediction_sha256"]),
        "completed_trials": int(completed),
        "fifo_capacity": int(fifo_capacity),
        "final_query_count": int(min(completed, fifo_capacity)),
        "final_pool_count": int(budget + min(completed, fifo_capacity)),
        "per_row_receipts": "not instrumented: the sealed path is run verbatim",
        "carrier_updates": int(result["accepted_carrier_updates"]),
        "parameter_updates": 0,
    }


def _reweighted_cell(
    *, torch: Any, model: Any, student: Any, dataset: Any, session: str,
    views: tuple, support: Mapping[str, Any], query_rows: tuple,
    side_mean: np.ndarray, side_std: np.ndarray, device: Any, batch_size: int,
    budget: int, policy: str, started: float, cdm_physical: Any,
    pseudo_core: Any,
) -> dict[str, Any]:
    """UNIFORM_UNCAPPED and the EMA cells: the frozen encoder's own pooling
    arithmetic fed with the policy's pooled feature state."""
    memory = cdm_physical._build_memory(
        system="activity_only", budget=budget, selected=support["selected"],
        support_rates_hz30=support["support_rates_hz30"],
        theta_first30=support["theta30"], support_b3s=support["support_b3s"],
        channel_ids=support["channels"], valid_mask=support["valid_mask"],
    )
    carrier_hz = np.asarray(memory.carrier.active_t4, dtype=np.float64)
    side_tensor = _side_tensor(torch, carrier_hz, side_mean, side_std)
    support_activities = [
        np.ascontiguousarray(support["support_b3s"][index].activity, dtype=np.float32)
        for index in support["selected"]
    ]
    channels = int(np.asarray(support["channels"]).size)
    pool_kwargs = dict(
        id_encoder=student.id_encoder, support_activities=support_activities,
        channels=channels, device=device, dtype=torch.float32, torch=torch,
    )
    if policy == "UNIFORM_UNCAPPED":
        pool: Any = memory_law.FrozenB3SUniformPool(**pool_kwargs)
    elif policy in plan.EMA_ALPHAS:
        pool = memory_law.FrozenB3SEmaPool(alpha=plan.EMA_ALPHAS[policy], **pool_kwargs)
    else:
        raise MemoryLawScanError(f"unknown reweighted policy {policy}")
    neural = np.asarray(dataset.neural_data[session], dtype=np.float32)
    behavior = np.asarray(dataset.covariate_data[session], dtype=np.float32)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    starts_joined: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    first_scored: Optional[dict[str, Any]] = None
    for position, row in enumerate(query_rows):
        metric_starts = np.asarray(row["metric_starts"], dtype=np.int64)
        identity_sha: Optional[str] = None
        if metric_starts.size:
            with torch.inference_mode():
                identity = pool.identity(side_tensor)
                identity_np = identity.detach().numpy().astype(np.float32, copy=False)
                identity_sha = pseudo_core.array_sha256(identity_np)
            prediction = _predict_with_identity(
                torch=torch, student=student,
                neural_windows=cdm_physical._windows(neural, metric_starts),
                identity=identity, batch_size=batch_size,
            )
            predictions.append(prediction)
            targets.append(np.ascontiguousarray(
                behavior[metric_starts + plan.WINDOW_BINS - 1], dtype=np.float32,
            ))
            starts_joined.append(metric_starts)
            if first_scored is None:
                first_scored = {
                    "row_index": int(position),
                    "commits_before": int(position),
                    "identity_sha256": str(identity_sha),
                }
        pool_before = int(pool.pool_count)
        with torch.inference_mode():
            pool.commit(np.asarray(row["activity"], dtype=np.float32))
        receipts.append({
            "p": int(position),
            "mw": int(metric_starts.size),
            "pcb": pool_before,
            "pca": int(pool.pool_count),
            "id": identity_sha,
        })
        _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                 "the hard timeout fired during a reweighted cell")
    prediction = np.ascontiguousarray(np.concatenate(predictions), dtype=np.float32)
    target = np.ascontiguousarray(np.concatenate(targets), dtype=np.float32)
    starts = np.ascontiguousarray(np.concatenate(starts_joined), dtype=np.int64)
    expected = cdm_physical._surface_starts(dataset, session)
    _require(np.array_equal(starts, expected),
             "reweighted cell scored query differs from the post30 authority")
    with torch.inference_mode():
        final_identity = pool.identity(side_tensor)
        final_identity_sha = pseudo_core.array_sha256(
            final_identity.detach().numpy().astype(np.float32, copy=False))
        final_pool_features_sha = pseudo_core.array_sha256(pool.pooled_features_numpy())
    law_payload = pool.law_payload()
    return {
        "policy": policy,
        "law": dict(plan.POLICIES[policy]),
        "budget": int(budget),
        "r2": float(pseudo_core.variance_weighted_r2(target, prediction)),
        "window_count": int(starts.size),
        "query_starts_sha256": pseudo_core.array_sha256(starts),
        "target_sha256": pseudo_core.array_sha256(target),
        "prediction_sha256": pseudo_core.array_sha256(prediction),
        "completed_trials": int(len(query_rows)),
        "first_scored": first_scored,
        "final_identity_sha256": str(final_identity_sha),
        "final_pool_features_sha256": str(final_pool_features_sha),
        "law_state": law_payload,
        "per_row_receipts": receipts,
        "carrier_updates": 0,
        "parameter_updates": 0,
    }


def _stack_identity_digest(
    *, torch: Any, student: Any, side_tensor: Any,
    support_activities: list[np.ndarray], completed_activities: list[np.ndarray],
    pseudo_core: Any,
) -> str:
    """The sealed path's identity over an explicit stack (the authority the
    streaming pools must reproduce bitwise)."""
    stack = np.ascontiguousarray(
        np.stack(support_activities + completed_activities, axis=0), dtype=np.float32,
    )
    _require(stack.ndim == 3 and stack.shape[1:] == (100, plan.CHANNELS),
             "identity authority stack topology drift")
    with torch.inference_mode():
        identity = student.compute_identity(
            torch.from_numpy(stack).unsqueeze(0), side_features=side_tensor,
        )
    return pseudo_core.array_sha256(
        identity.detach().numpy().astype(np.float32, copy=False))


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
                 f"an owned scan module drifted: {relative}")

    _bind_namespaces(repo_root)
    from src.cdm_p1_m2_local_v1 import plan as g_plan

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
    _require(metadata["checkpoint_sha256"] == g_plan.T4_CHECKPOINT_SHA256,
             "T4 checkpoint drift")
    _require(metadata["teacher_checkpoint_sha256"] == g_plan.SPINT_CHECKPOINT_SHA256,
             "SPINT teacher checkpoint drift")
    _require(metadata["normalization_sha256"] == g_plan.NORMALIZATION_SHA256,
             "T4 normalization drift")
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    student = model.student
    _require(getattr(student, "id_encoder", None) is not None
             and getattr(student.id_encoder, "variant", None) == "B3S",
             "the frozen student must expose the B3S id encoder")
    side_mean = np.asarray(metadata["normalization_mean"], dtype=np.float32)
    side_std = np.asarray(metadata["normalization_std"], dtype=np.float32)

    from src.cdm_p1_m2_local_v1 import replay as g_replay
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical as cdm_physical
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core as pseudo_core

    sealed_rows = _load_sealed_rows(repo_root)
    surfaces = {
        "within_post30": (data_module.train_dataset, data_module.train_calib_heldin_sessions),
        "external_post30_local": (
            data_module.val_heldout_dataset, data_module.val_calib_heldout_sessions,
        ),
    }
    _require(data_module.train_dataset is not None
             and data_module.val_heldout_dataset is not None,
             "the frozen M2 runtime must expose both held-in and held-out datasets")
    equivalence_roster = (
        [("external_post30_local", session) for session in sorted(
            data_module.val_heldout_dataset.calib_trialized_neural_features)]
        + [("within_post30", session) for session in sorted(
            data_module.train_dataset.calib_trialized_neural_features)[:2]]
    )
    started = time.monotonic()

    rows: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {
        (surface, budget, policy): {}
        for surface in plan.SURFACES for budget in plan.BUDGETS
        for policy in plan.POLICY_ORDER
    }
    anchors_cap30: dict[str, Any] = {}
    anchors_equivalence: dict[str, Any] = {}
    identity_agreements: dict[str, Any] = {}
    session_meta: dict[str, Any] = {}

    for surface in plan.SURFACES:
        dataset, raw_sessions = surfaces[surface]
        _require(dataset is not None, f"{surface}: dataset missing")
        sessions = sorted(dataset.calib_trialized_neural_features)
        _require(len(sessions) == plan.EXPECTED_SESSIONS_BY_SURFACE[surface],
                 f"{surface}: session count drift")
        for session in sessions:
            _require(session in raw_sessions, f"{surface}/{session}: raw authority absent")
            views = g_replay._g_session_views(raw_sessions, session=session)
            support_base = g_replay.g_support_material(session=session, views=views)
            query_rows = g_replay.g_query_rows(ds=dataset, session=session, views=views)
            completed_activities = [
                np.ascontiguousarray(row["activity"], dtype=np.float32)
                for row in query_rows
            ]
            session_meta[f"{surface}|{session}"] = {
                "completed_query_trials": int(len(query_rows)),
                "scored_windows": int(sum(
                    np.asarray(row["metric_starts"], dtype=np.int64).size
                    for row in query_rows)),
                "short_trials_without_complete_window": int(sum(
                    1 for row in query_rows
                    if np.asarray(row["causal_starts"], dtype=np.int64).size == 0)),
            }
            for budget in plan.BUDGETS:
                support = _support_for_budget(support_base, budget, cdm_physical)
                support_activities = [
                    np.ascontiguousarray(support["support_b3s"][index].activity,
                                         dtype=np.float32)
                    for index in support["selected"]
                ]
                # -- UNIFORM_CAP30 (the sealed law) ------------------------
                row = _uniform_cap30_cell(
                    torch=torch, model=model, dataset=dataset, session=session,
                    views=views, support=support, query_rows=query_rows,
                    side_mean=side_mean, side_std=side_std, device=device,
                    batch_size=batch_size, budget=budget,
                    cdm_physical=cdm_physical, pseudo_core=pseudo_core,
                    use_sealed_g00m=(budget == plan.M4),
                )
                rows[(surface, budget, "UNIFORM_CAP30")][session] = row
                sealed = sealed_rows[(surface, session, budget)]
                anchors_cap30[f"{surface}|{session}|m{budget}"] = anchor_matches(
                    row, sealed,
                    r2_tolerance=float(
                        plan.ANCHORS["uniform_cap30_vs_sealed_activity_only_rows"]["r2_tolerance"]),
                )
                _require(anchors_cap30[f"{surface}|{session}|m{budget}"]["exact_match"],
                         f"the UNIFORM_CAP30 anchor failed at {surface}/{session}/m{budget}")
                # -- the executor-fidelity anchor (generalized vs sealed) ---
                if budget == plan.M4 and (surface, session) in equivalence_roster:
                    generalized = _uniform_cap30_cell(
                        torch=torch, model=model, dataset=dataset, session=session,
                        views=views, support=support, query_rows=query_rows,
                        side_mean=side_mean, side_std=side_std, device=device,
                        batch_size=batch_size, budget=budget,
                        cdm_physical=cdm_physical, pseudo_core=pseudo_core,
                        use_sealed_g00m=False,
                    )
                    equal = {
                        "prediction_sha256": (
                            generalized["prediction_sha256"] == row["prediction_sha256"]),
                        "target_sha256": generalized["target_sha256"] == row["target_sha256"],
                        "query_starts_sha256": (
                            generalized["query_starts_sha256"] == row["query_starts_sha256"]),
                        "r2": float(generalized["r2"]) == float(row["r2"]),
                        "window_count": (
                            generalized["window_count"] == row["window_count"]),
                    }
                    anchors_equivalence[f"{surface}|{session}"] = {
                        "field_matches": equal,
                        "exact_match": all(bool(value) for value in equal.values()),
                    }
                    _require(anchors_equivalence[f"{surface}|{session}"]["exact_match"],
                             f"the generalized CAP30 executor drifted from rollout_g00m "
                             f"at {surface}/{session}")
                # -- the reweighted policies -------------------------------
                memory_probe = cdm_physical._build_memory(
                    system="activity_only", budget=budget,
                    selected=support["selected"],
                    support_rates_hz30=support["support_rates_hz30"],
                    theta_first30=support["theta30"],
                    support_b3s=support["support_b3s"],
                    channel_ids=support["channels"],
                    valid_mask=support["valid_mask"],
                )
                side_tensor = _side_tensor(
                    torch, np.asarray(memory_probe.carrier.active_t4, dtype=np.float64),
                    side_mean, side_std,
                )
                for policy in plan.POLICY_ORDER[1:]:
                    cell = _reweighted_cell(
                        torch=torch, model=model, student=student, dataset=dataset,
                        session=session, views=views, support=support,
                        query_rows=query_rows, side_mean=side_mean, side_std=side_std,
                        device=device, batch_size=batch_size, budget=budget,
                        policy=policy, started=started, cdm_physical=cdm_physical,
                        pseudo_core=pseudo_core,
                    )
                    rows[(surface, budget, policy)][session] = cell
                    # identity agreements against the sealed stack-path authority
                    _require(cell["first_scored"] is not None,
                             f"{surface}/{session}/m{budget}/{policy}: no scored query row")
                    record: dict[str, Any] = {
                        "policy": policy,
                        "first_scored": cell["first_scored"],
                        "final_identity_sha256": cell["final_identity_sha256"],
                    }
                    if policy == "UNIFORM_UNCAPPED":
                        first = cell["first_scored"]
                        k = int(first["commits_before"])
                        uncapped_first = _stack_identity_digest(
                            torch=torch, student=student, side_tensor=side_tensor,
                            support_activities=support_activities,
                            completed_activities=completed_activities[:k],
                            pseudo_core=pseudo_core,
                        )
                        capacity = plan.ACTIVITY_STACK_LIMIT - budget
                        capped_members = (
                            completed_activities[:k] if k <= capacity
                            else completed_activities[k - capacity : k])
                        capped_first = _stack_identity_digest(
                            torch=torch, student=student, side_tensor=side_tensor,
                            support_activities=support_activities,
                            completed_activities=capped_members,
                            pseudo_core=pseudo_core,
                        )
                        final_stack = _stack_identity_digest(
                            torch=torch, student=student, side_tensor=side_tensor,
                            support_activities=support_activities,
                            completed_activities=completed_activities,
                            pseudo_core=pseudo_core,
                        )
                        record.update({
                            "first_identity_matches_uncapped_stack": (
                                uncapped_first == first["identity_sha256"]),
                            "first_identity_matches_capped_stack": (
                                capped_first == first["identity_sha256"]),
                            "final_identity_matches_full_stack": (
                                final_stack == cell["final_identity_sha256"]),
                        })
                        _require(record["first_identity_matches_uncapped_stack"]
                                 and record["final_identity_matches_full_stack"],
                                 f"the uncapped streaming law drifted from the frozen "
                                 f"stack mean at {surface}/{session}/m{budget}")
                    identity_agreements[f"{surface}|{session}|m{budget}|{policy}"] = record
                _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                         "the hard timeout fired during the governing grid")

    _require(all(item["exact_match"] for item in anchors_cap30.values()),
             "at least one UNIFORM_CAP30 anchor failed")
    _require(all(item["exact_match"] for item in anchors_equivalence.values()),
             "the generalized executor equivalence anchor failed")

    r2_view: dict[str, dict[int, dict[str, dict[str, float]]]] = {
        surface: {
            budget: {
                policy: {
                    session: float(rows[(surface, budget, policy)][session]["r2"])
                    for session in rows[(surface, budget, policy)]
                }
                for policy in plan.POLICY_ORDER
            }
            for budget in plan.BUDGETS
        }
        for surface in plan.SURFACES
    }
    summaries: dict[str, dict[int, dict[str, Any]]] = {}
    deltas: dict[str, dict[int, dict[str, Any]]] = {}
    for surface in plan.SURFACES:
        summaries[surface] = {}
        deltas[surface] = {}
        for budget in plan.BUDGETS:
            summaries[surface][budget] = {}
            deltas[surface][budget] = {}
            for policy in plan.POLICY_ORDER:
                values = r2_view[surface][budget][policy]
                summaries[surface][budget][policy] = {
                    "equal_session_mean": gates.equal_session_mean(values),
                    "session_sd": gates.session_sd(values),
                    "per_session_r2": dict(sorted(values.items())),
                }
                if policy != plan.BASELINE_POLICY:
                    deltas[surface][budget][policy] = gates.paired_delta(
                        values, r2_view[surface][budget][plan.BASELINE_POLICY],
                    )

    policy_results = {
        policy: gates.evaluate_policy_gates(
            policy=policy,
            within_m4=r2_view["within_post30"][plan.M4][policy],
            baseline_within_m4=r2_view["within_post30"][plan.M4][plan.BASELINE_POLICY],
            external_by_budget={
                budget: r2_view["external_post30_local"][budget][policy]
                for budget in plan.BUDGETS
            },
            baseline_external_by_budget={
                budget: r2_view["external_post30_local"][budget][plan.BASELINE_POLICY]
                for budget in plan.BUDGETS
            },
            gates_law=plan.GATES,
        )
        for policy in plan.POLICY_ORDER[1:]
    }
    verdict = gates.select_verdict(
        policy_results=policy_results, policy_order=plan.POLICY_ORDER[1:],
    )
    within_completed = {
        session: session_meta[f"within_post30|{session}"]["completed_query_trials"]
        for session in r2_view["within_post30"][plan.M4][plan.BASELINE_POLICY]
    }
    drift = gates.drift_reading(
        session_completed_trials=within_completed,
        within_m4_deltas_by_policy={
            policy: deltas["within_post30"][plan.M4][policy]["per_session_delta"]
            for policy in plan.POLICY_ORDER[1:]
        },
    )

    safety_all_pass = all(item["safety"]["passed"] for item in policy_results.values())
    stop_conditions = {
        "anchor_or_binding_failure": {
            "expression": "any CAP30 anchor, executor-equivalence or identity-agreement law failed",
            "fired": not (
                all(item["exact_match"] for item in anchors_cap30.values())
                and all(item["exact_match"] for item in anchors_equivalence.values())
                and all(
                    (record["policy"] != "UNIFORM_UNCAPPED")
                    or (record.get("first_identity_matches_uncapped_stack", False)
                        and record.get("final_identity_matches_full_stack", False))
                    for record in identity_agreements.values()
                )
            ),
            "evidence": {
                "cap30_anchors_all_exact": all(
                    item["exact_match"] for item in anchors_cap30.values()),
                "executor_equivalence_all_exact": all(
                    item["exact_match"] for item in anchors_equivalence.values()),
            },
        },
        "no_challenger_beats_baseline": {
            "expression": "no policy passes the pre-registered primary gate",
            "fired": verdict["verdict"] == "MEMORY_LAW_UNIFORM_RETAINED",
            "evidence": {
                name: {
                    "within_m4_delta": item["primary"]["delta"]["equal_session_mean_delta"],
                    "positive_sessions": item["primary"]["delta"]["positive_sessions"],
                    "primary_passed": item["primary"]["passed"],
                }
                for name, item in policy_results.items()
            },
        },
        "safety_floor_breached": {
            "expression": (
                "any policy - UNIFORM_CAP30 external mean < -0.02 at any budget"),
            "fired": not safety_all_pass,
            "evidence": {
                name: {
                    f"m{budget}": item["safety"]["bounds"][f"m{budget}"]["delta"][
                        "equal_session_mean_delta"]
                    for budget in plan.BUDGETS
                }
                for name, item in policy_results.items()
            },
        },
    }
    governing = [name for name, item in stop_conditions.items() if item["fired"]]

    replay_payload = {
        "schema": f"{plan.SCHEMA}_replay_v1",
        "stage": "inference_only_memory_law_scan_local_m2_cpu",
        "status": "REPLAY_COMPLETE",
        "attempt_sha256": attempt_digest,
        "foundation": {
            "sealed_g_family_package": plan.SEALED_G_PACKAGE,
            "sealed_cdm_screen_score": plan.CDM_SCREEN_SCORE_RELATIVE,
            "sealed_cdm_screen_score_sha256": plan.CDM_SCREEN_SCORE_SHA256,
            "reuse_law": "frozen modules imported verbatim; nothing edited",
        },
        "environment": {
            **plan.ENVIRONMENT_LAW,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "batch_size": int(batch_size),
            "seeds": [42],
            "cpu_count_visible_to_torch": int(torch.get_num_threads()),
        },
        "surface": {
            "law": dict(plan.SURFACE_LAW),
            "within_sessions": sorted(r2_view["within_post30"][plan.M4][plan.BASELINE_POLICY]),
            "external_sessions": sorted(
                r2_view["external_post30_local"][plan.M4][plan.BASELINE_POLICY]),
            "session_meta": session_meta,
        },
        "cells": dict(plan.POLICIES),
        "memory_law": dict(plan.MEMORY_LAW),
        "frozen_everything_else": dict(plan.FROZEN_EVERYTHING_ELSE),
        "summaries": summaries,
        "paired_deltas_vs_uniform_cap30": deltas,
        "rows": {
            f"{surface}|m{budget}|{policy}": dict(sorted(sessions.items()))
            for (surface, budget, policy), sessions in sorted(rows.items())
        },
        "anchors": {
            "uniform_cap30_vs_sealed_rows": anchors_cap30,
            "all_exact": all(item["exact_match"] for item in anchors_cap30.values()),
            "max_abs_r2_delta_vs_sealed": max(
                abs(item["r2_delta"]) for item in anchors_cap30.values()),
            "m4_executor_equivalence": anchors_equivalence,
            "m4_executor_equivalence_all_exact": all(
                item["exact_match"] for item in anchors_equivalence.values()),
            "identity_agreements": identity_agreements,
            "law": {
                key: (dict(value) if isinstance(value, dict) else value)
                for key, value in plan.ANCHORS.items()
            },
        },
        "gates": {
            "law": dict(plan.GATES),
            "policy_results": policy_results,
            "verdict": verdict,
        },
        "drift_reading": drift,
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
    replay_digest = _publish(root / "replay.json", replay_payload)
    terminal_payload = {
        "schema": f"{plan.SCHEMA}_terminal_v1",
        "status": "TERMINAL",
        "scope": "inference_only_memory_law_scan_local_m2_cpu",
        "replay_sha256": replay_digest,
        "attempt_sha256": attempt_digest,
        "verdict": verdict,
        "gates": {
            "primary_expression": plan.GATES["primary"]["expression"],
            "safety_expression": plan.GATES["safety"]["expression"],
            "policy_results": {
                name: {
                    "within_m4_delta": item["primary"]["delta"]["equal_session_mean_delta"],
                    "within_m4_positive_sessions": item["primary"]["delta"]["positive_sessions"],
                    "primary_passed": item["primary"]["passed"],
                    "external_m4_delta": item["safety"]["bounds"]["m4"]["delta"][
                        "equal_session_mean_delta"],
                    "external_m10_delta": item["safety"]["bounds"]["m10"]["delta"][
                        "equal_session_mean_delta"],
                    "safety_passed": item["safety"]["passed"],
                    "disposition": item["disposition"],
                }
                for name, item in policy_results.items()
            },
        },
        "equal_session_means": {
            surface: {
                f"m{budget}": {
                    policy: summaries[surface][budget][policy]["equal_session_mean"]
                    for policy in plan.POLICY_ORDER
                }
                for budget in plan.BUDGETS
            }
            for surface in plan.SURFACES
        },
        "anchors": {
            "cap30_all_exact": replay_payload["anchors"]["all_exact"],
            "max_abs_r2_delta_vs_sealed": replay_payload["anchors"]["max_abs_r2_delta_vs_sealed"],
            "m4_executor_equivalence_all_exact": replay_payload["anchors"][
                "m4_executor_equivalence_all_exact"],
        },
        "drift_reading": drift,
        "stop_conditions": replay_payload["stop_conditions"],
        "official_contract_claimed": False,
        "wall_seconds": replay_payload["wall_seconds"],
    }
    _publish(root / "terminal.json", terminal_payload)
    return {
        "status": terminal_payload["status"],
        "replay_sha256": replay_digest,
        "verdict": verdict["verdict"],
        "winner": verdict["winner"],
        "stop_conditions_fired": governing,
        "wall_seconds": replay_payload["wall_seconds"],
    }
