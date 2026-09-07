"""Physical driver of the M2 post-fusion identity-memory probe (CPU-only,
inference-only, frozen weights).

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

from . import gates, memory as identity_pools, plan


class PostFusionProbeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PostFusionProbeError(message)


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
             "the post-fusion probe is CPU-only (CUDA_VISIBLE_DEVICES must be empty)")
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


def _load_sealed_rows(repo_root: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    path = repo_root / plan.CDM_SCREEN_SCORE_RELATIVE
    _require(_sha256_file(path) == plan.CDM_SCREEN_SCORE_SHA256,
             "sealed CDM screen score drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == "m2_precision_cdm_v2_screen_v1"
             and payload.get("status") == "TERMINAL",
             "sealed CDM screen schema drift")
    rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in payload["rows"]:
        if row.get("cell") == "m4_activity_only":
            rows[(str(row["surface"]), str(row["session_id"]))] = row
    _require(len(rows) == sum(plan.EXPECTED_SESSIONS_BY_SURFACE.values()),
             "sealed m4_activity_only anchor row topology drift")
    return rows


def anchor_matches(cell_row: Mapping[str, Any], sealed_row: Mapping[str, Any],
                   *, r2_tolerance: float) -> dict[str, Any]:
    """The POOLED anchor against one sealed GPU row (exact pure-data fields,
    R2 within the pre-registered 1e-5 CPU/GPU tolerance)."""
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


def _predict_with_identity(
    *, torch: Any, student: Any, neural_windows: np.ndarray, identity: Any,
    batch_size: int,
) -> np.ndarray:
    """The decode path with a precomputed identity.

    The sealed ``_predict`` law (``pseudo_mua_precision_cdm_v2_screen_v1``)
    with the identity supplied by the caller instead of recomputed from a
    stack -- the ``m2_memory_law_scan_v1`` mirror, arithmetic verbatim.
    """
    _require(getattr(student, "decoder_mode", None) == "coupled",
             "the probe requires the coupled decoder")
    _require(getattr(student, "fixed_slot_router", None) is None,
             "the probe forbids fixed-slot routing")
    _require(not hasattr(student.id_encoder, "forward_batch_with_gate"),
             "the probe forbids hidden identity gate semantics")
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
    _require(side_np.shape == (plan.CHANNELS, plan.SIDE_DIM) and np.isfinite(side_np).all(),
             "static T4 side-feature drift")
    return torch.from_numpy(side_np).unsqueeze(0)


def _verify_identity_path_structure(*, student: Any, torch: Any) -> dict[str, Any]:
    """Bind the plan's identity-path law to the loaded frozen encoder."""
    encoder = student.id_encoder
    _require(getattr(encoder, "variant", None) == "B3S",
             "the frozen student must expose the B3S id encoder")
    _require(int(encoder.trial_length) == plan.TRIAL_LENGTH, "encoder trial_length drift")
    _require(int(encoder.hidden_dim) == plan.ID_HIDDEN_DIM, "encoder hidden_dim drift")
    _require(int(encoder.side_dim) == plan.SIDE_DIM, "encoder side_dim drift")
    _require(int(encoder.window_size) == plan.WINDOW_BINS, "encoder window_size drift")
    pre = list(encoder.pre_pool)
    _require(isinstance(pre[0], torch.nn.Linear)
             and pre[0].in_features == plan.TRIAL_LENGTH
             and pre[0].out_features == plan.ID_HIDDEN_DIM
             and isinstance(pre[1], torch.nn.ReLU),
             "pre_pool must be Linear(100->64)+ReLU")
    linears = [layer for layer in encoder.post_pool if isinstance(layer, torch.nn.Linear)]
    _require(len(linears) == plan.POST_POOL_LAYERS,
             "post_pool must be a 3-layer affine stack")
    _require(linears[0].in_features == plan.ID_HIDDEN_DIM + plan.SIDE_DIM
             and linears[-1].out_features == plan.WINDOW_BINS,
             "post_pool input/output dimension drift")
    return {
        "variant": "B3S",
        "trial_length": int(encoder.trial_length),
        "hidden_dim": int(encoder.hidden_dim),
        "side_dim": int(encoder.side_dim),
        "window_size": int(encoder.window_size),
        "pre_pool": "Linear(100->64)+ReLU",
        "post_pool": "68->64->64->50 affine stack (3 Linear layers, ReLU between)",
    }


def _gap_stats(delta: Any, torch: Any) -> dict[str, float]:
    values = delta.detach().numpy().astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(values))),
        "mean_abs": float(np.mean(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values ** 2))),
    }


def _postfusion_cells(
    *, torch: Any, student: Any, dataset: Any, session: str,
    support_identities: list[Any], query_rows: tuple, side_tensor: Any,
    batch_size: int, started: float, cdm_physical: Any, pseudo_core: Any,
) -> dict[str, dict[str, Any]]:
    """The three post-fusion readouts over ONE shared per-trial identity stream."""
    with torch.inference_mode():
        pools: dict[str, Any] = {
            "POSTFUSION_MEAN": identity_pools.PostFusionUniformIdentityPool(
                support_identities=support_identities,
                capacity=plan.ACTIVITY_STACK_LIMIT,
            ),
            "POSTFUSION_ACCUM_UNCAPPED": identity_pools.PostFusionUniformIdentityPool(
                support_identities=support_identities, capacity=None,
            ),
            "POSTFUSION_EMA_A090": identity_pools.PostFusionEmaIdentityPool(
                support_identities=support_identities, alpha=plan.EMA_ALPHA,
            ),
        }
    neural = np.asarray(dataset.neural_data[session], dtype=np.float32)
    behavior = np.asarray(dataset.covariate_data[session], dtype=np.float32)
    channels = int(plan.CHANNELS)
    device = torch.device("cpu")
    committed_identities: list[Any] = []
    first_scored_agreement: Optional[dict[str, Any]] = None
    cell_state: dict[str, dict[str, Any]] = {
        name: {
            "predictions": [], "targets": [], "starts_joined": [],
            "receipts": [], "first_scored": None,
        }
        for name in pools
    }
    for position, row in enumerate(query_rows):
        metric_starts = np.asarray(row["metric_starts"], dtype=np.int64)
        if metric_starts.size:
            if first_scored_agreement is None:
                # the first-scored identity agreement anchor: both uniform
                # pools must equal the from-scratch sequential mean over the
                # CURRENT members (support + the identities committed so far;
                # eviction cannot have occurred unless 26+ trials completed
                # before the first scored window)
                with torch.inference_mode():
                    reference = identity_pools.sequential_mean(
                        list(support_identities) + list(committed_identities))
                    capped_now = pools["POSTFUSION_MEAN"].deployed()
                    uncapped_now = pools["POSTFUSION_ACCUM_UNCAPPED"].deployed()
                    eviction_occurred = pools["POSTFUSION_MEAN"].evictions > 0
                    if eviction_occurred:
                        capped_own = pools["POSTFUSION_MEAN"].from_scratch_mean()
                        capped_drift = float(torch.max(
                            torch.abs(capped_now - capped_own)).item())
                    else:
                        capped_drift = 0.0
                    agreement = {
                        "row_index": int(position),
                        "commits_before": int(len(committed_identities)),
                        "eviction_occurred_before_first_scored": bool(eviction_occurred),
                        "uncapped_equals_reference_bitwise": bool(
                            torch.equal(uncapped_now, reference)),
                        "capped_equals_uncapped_bitwise": (
                            None if eviction_occurred
                            else bool(torch.equal(capped_now, uncapped_now))),
                        "capped_equals_reference_bitwise": (
                            None if eviction_occurred
                            else bool(torch.equal(capped_now, reference))),
                        "capped_max_abs_drift_vs_own_members": capped_drift,
                    }
                _require(
                    agreement["uncapped_equals_reference_bitwise"]
                    and (agreement["capped_equals_reference_bitwise"] is True
                         if not eviction_occurred
                         else agreement["capped_max_abs_drift_vs_own_members"] <= 1.0e-4),
                    f"{session}: the first-scored identity agreement law broke")
                first_scored_agreement = agreement
            windows = cdm_physical._windows(neural, metric_starts)
            targets = np.ascontiguousarray(
                behavior[metric_starts + plan.WINDOW_BINS - 1], dtype=np.float32,
            )
            for name, pool in pools.items():
                with torch.inference_mode():
                    identity = pool.deployed()
                prediction = _predict_with_identity(
                    torch=torch, student=student, neural_windows=windows,
                    identity=identity, batch_size=batch_size,
                )
                state = cell_state[name]
                state["predictions"].append(prediction)
                state["targets"].append(targets)
                state["starts_joined"].append(metric_starts)
                if state["first_scored"] is None:
                    with torch.inference_mode():
                        identity_np = identity.detach().numpy().astype(np.float32, copy=False)
                    state["first_scored"] = {
                        "row_index": int(position),
                        "commits_before": int(position),
                        "identity_sha256": pseudo_core.array_sha256(identity_np),
                        "identity": identity,
                    }
        # the completed trial's identity, computed ONCE and fed to every pool
        with torch.inference_mode():
            trial_identity = identity_pools.per_trial_identity(
                student.id_encoder, torch,
                np.asarray(row["activity"], dtype=np.float32), side_tensor,
                channels=channels, device=device, dtype=torch.float32,
            )
            for pool in pools.values():
                pool.commit(trial_identity)
            committed_identities.append(trial_identity)
        for name, pool in pools.items():
            cell_state[name]["receipts"].append({
                "p": int(position),
                "mw": int(metric_starts.size),
                "pcb": int(pool.completed_count) - 1,
                "pca": int(pool.completed_count),
                "members": int(pool.pool_count) if pool.family == "uniform" else None,
            })
        _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                 "the hard timeout fired during a post-fusion cell")
    rows: dict[str, dict[str, Any]] = {}
    for name, pool in pools.items():
        state = cell_state[name]
        _require(bool(state["predictions"]), f"{name}: no scored query row")
        prediction = np.ascontiguousarray(np.concatenate(state["predictions"]), dtype=np.float32)
        target = np.ascontiguousarray(np.concatenate(state["targets"]), dtype=np.float32)
        starts = np.ascontiguousarray(np.concatenate(state["starts_joined"]), dtype=np.int64)
        expected = cdm_physical._surface_starts(dataset, session)
        _require(np.array_equal(starts, expected),
                 f"{name} scored query differs from the post30 authority")
        with torch.inference_mode():
            final_identity = pool.deployed()
            final_identity_np = final_identity.detach().numpy().astype(np.float32, copy=False)
            first_identity = state["first_scored"]["identity"]
            first_identity_np = first_identity.detach().numpy().astype(
                np.float32, copy=False)
            if pool.family == "uniform":
                from_scratch = pool.from_scratch_mean()
                audit_bitwise = bool(torch.equal(final_identity, from_scratch))
                audit_max_abs = float(torch.max(torch.abs(final_identity - from_scratch)).item())
            else:
                audit_bitwise = None
                audit_max_abs = None
        rows[name] = {
            "policy": name,
            "law": dict(plan.CELLS[name]),
            "budget": int(plan.BUDGET),
            "r2": float(pseudo_core.variance_weighted_r2(target, prediction)),
            "window_count": int(starts.size),
            "query_starts_sha256": pseudo_core.array_sha256(starts),
            "target_sha256": pseudo_core.array_sha256(target),
            "prediction_sha256": pseudo_core.array_sha256(prediction),
            "completed_trials": int(len(query_rows)),
            "first_scored_agreement": (
                dict(first_scored_agreement) if name == "POSTFUSION_MEAN" else None),
            "first_scored": {
                "row_index": state["first_scored"]["row_index"],
                "commits_before": state["first_scored"]["commits_before"],
                "identity_sha256": str(state["first_scored"]["identity_sha256"]),
            },
            "first_scored_identity": first_identity,
            "final_identity_sha256": pseudo_core.array_sha256(final_identity_np),
            "final_accumulation_audit": {
                "bitwise_equal_to_from_scratch": audit_bitwise,
                "max_abs_drift_vs_from_scratch": audit_max_abs,
                "law": dict(plan.ANCHORS["final_accumulation_audit"]),
            },
            "law_state": pool.law_payload(),
            "per_row_receipts": state["receipts"],
            "carrier_updates": 0,
            "parameter_updates": 0,
        }
    return rows


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
                 f"an owned probe module drifted: {relative}")

    _bind_namespaces(repo_root)
    from src.cdm_p1_m2_local_v1 import plan as g_plan
    from src.cdm_p1_m2_local_v1 import replay as g_replay

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
    _require(getattr(student, "id_encoder", None) is not None,
             "the frozen student must expose the id encoder")
    identity_path = _verify_identity_path_structure(student=student, torch=torch)
    side_mean = np.asarray(metadata["normalization_mean"], dtype=np.float32)
    side_std = np.asarray(metadata["normalization_std"], dtype=np.float32)

    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical as cdm_physical
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core as pseudo_core

    # the sealed precedent receipts are pinned read-only context (never anchors
    # beyond the disclosed provenance of the G00m/k4 anchor means)
    _require(_sha256_file(repo_root / plan.KCURVE_TERMINAL_RELATIVE)
             == plan.KCURVE_TERMINAL_SHA256, "sealed k-curve terminal drift")
    _require(_sha256_file(repo_root / plan.MEMORY_LAW_SCAN_TERMINAL_RELATIVE)
             == plan.MEMORY_LAW_SCAN_TERMINAL_SHA256, "sealed memory-law scan drift")
    kcurve = json.loads((repo_root / plan.KCURVE_TERMINAL_RELATIVE).read_text(encoding="utf-8"))
    _require(
        kcurve["anchor_receipt_values"]["g00m_means"] == plan.G00M_ANCHOR_MEANS,
        "the pinned G00m/k4 anchor means disagree with the sealed k-curve receipt",
    )

    sealed_rows = _load_sealed_rows(repo_root)
    surfaces = {
        "external_post30_local": (
            data_module.val_heldout_dataset, data_module.val_calib_heldout_sessions),
        "within_post30": (
            data_module.train_dataset, data_module.train_calib_heldin_sessions),
    }
    _require(data_module.train_dataset is not None
             and data_module.val_heldout_dataset is not None,
             "the frozen M2 runtime must expose both held-in and held-out datasets")
    started = time.monotonic()

    rows: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        surface: {cell: {} for cell in plan.CELL_ORDER} for surface in plan.SURFACES
    }
    anchors_pooled: dict[str, Any] = {}
    anchors_identity: dict[str, Any] = {}
    jensen_gaps: dict[str, Any] = {}
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
            support = g_replay.g_support_material(session=session, views=views)
            query_rows = g_replay.g_query_rows(ds=dataset, session=session, views=views)
            session_meta[f"{surface}|{session}"] = {
                "completed_query_trials": int(len(query_rows)),
                "scored_windows": int(sum(
                    np.asarray(row["metric_starts"], dtype=np.int64).size
                    for row in query_rows)),
                "short_trials_without_complete_window": int(sum(
                    1 for row in query_rows
                    if np.asarray(row["causal_starts"], dtype=np.int64).size == 0)),
                "support_selected": [int(item) for item in support["selected"]],
            }
            # -- POOLED: the sealed law verbatim --------------------------
            pooled = g_replay.rollout_g00m(
                torch=torch, model=model, ds=dataset, session=session, views=views,
                support=support, query_rows=query_rows, side_mean=side_mean,
                side_std=side_std, device=device, batch_size=batch_size,
            )
            pooled_row = {
                "policy": "POOLED",
                "law": dict(plan.CELLS["POOLED"]),
                "budget": int(plan.BUDGET),
                "r2": float(pooled["r2"]),
                "window_count": int(pooled["window_count"]),
                "query_starts_sha256": str(pooled["query_starts_sha256"]),
                "target_sha256": str(pooled["target_sha256"]),
                "prediction_sha256": str(pooled["prediction_sha256"]),
                "completed_trials": int(len(query_rows)),
                "source": "sealed_m4_activity_only_law_verbatim",
                "carrier_updates": 0,
                "parameter_updates": 0,
            }
            rows[surface]["POOLED"][session] = pooled_row
            sealed = sealed_rows[(surface, session)]
            anchors_pooled[f"{surface}|{session}"] = anchor_matches(
                pooled_row, sealed,
                r2_tolerance=float(
                    plan.ANCHORS["pooled_vs_sealed_g00m_k4_rows"]["r2_tolerance"]),
            )
            _require(anchors_pooled[f"{surface}|{session}"]["exact_match"],
                     f"the POOLED anchor failed at {surface}/{session}")

            # -- the shared post-fusion machinery -------------------------
            memory_probe = cdm_physical._build_memory(
                system="activity_only", budget=plan.BUDGET,
                selected=support["selected"],
                support_rates_hz30=support["support_rates_hz30"],
                theta_first30=support["theta30"], support_b3s=support["support_b3s"],
                channel_ids=support["channels"], valid_mask=support["valid_mask"],
            )
            carrier_hz = np.asarray(memory_probe.carrier.active_t4, dtype=np.float64)
            side_tensor = _side_tensor(torch, carrier_hz, side_mean, side_std)
            support_activities = [
                np.ascontiguousarray(support["support_b3s"][index].activity,
                                     dtype=np.float32)
                for index in support["selected"]
            ]
            with torch.inference_mode():
                support_identities = [
                    identity_pools.per_trial_identity(
                        student.id_encoder, torch, activity, side_tensor,
                        channels=plan.CHANNELS, device=device, dtype=torch.float32,
                    )
                    for activity in support_activities
                ]
                support_stack = np.ascontiguousarray(
                    np.stack(support_activities, axis=0), dtype=np.float32)
                pooled_first_identity = student.compute_identity(
                    torch.from_numpy(support_stack).unsqueeze(0),
                    side_features=side_tensor,
                )
            postfusion_rows = _postfusion_cells(
                torch=torch, student=student, dataset=dataset, session=session,
                support_identities=support_identities, query_rows=query_rows,
                side_tensor=side_tensor, batch_size=batch_size, started=started,
                cdm_physical=cdm_physical, pseudo_core=pseudo_core,
            )
            for name, row in postfusion_rows.items():
                rows[surface][name][session] = {
                    key: value for key, value in row.items()
                    if key != "first_scored_identity"
                }
            # -- the first-scored identity agreement anchor ----------------
            # (verified inside the cell against the from-scratch sequential
            # mean over the members present at the first scored row)
            anchors_identity[f"{surface}|{session}"] = dict(
                postfusion_rows["POSTFUSION_MEAN"]["first_scored_agreement"])
            _require(
                anchors_identity[f"{surface}|{session}"]["uncapped_equals_reference_bitwise"]
                and (anchors_identity[f"{surface}|{session}"]["capped_equals_reference_bitwise"]
                     is True
                     or anchors_identity[f"{surface}|{session}"][
                         "capped_max_abs_drift_vs_own_members"] <= 1.0e-4),
                f"the first-scored identity agreement law broke at {surface}/{session}")
            # -- the support-pool Jensen gap (descriptive) -----------------
            capped = postfusion_rows["POSTFUSION_MEAN"]["first_scored_identity"]
            with torch.inference_mode():
                gap = capped - pooled_first_identity
            jensen_gaps[f"{surface}|{session}"] = {
                **_gap_stats(gap, torch),
                "postfusion_first_identity_rms": float(np.sqrt(np.mean(
                    capped.detach().numpy().astype(np.float64) ** 2))),
                "pooled_first_identity_rms": float(np.sqrt(np.mean(
                    pooled_first_identity.detach().numpy().astype(np.float64) ** 2))),
            }
            _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                     "the hard timeout fired during the governing grid")

    # -- summaries, paired deltas, gates --------------------------------------
    r2_view: dict[str, dict[str, dict[str, float]]] = {
        surface: {
            cell: {session: float(rows[surface][cell][session]["r2"])
                   for session in rows[surface][cell]}
            for cell in plan.CELL_ORDER
        }
        for surface in plan.SURFACES
    }
    summaries: dict[str, dict[str, Any]] = {}
    deltas: dict[str, dict[str, Any]] = {}
    for surface in plan.SURFACES:
        summaries[surface] = {}
        deltas[surface] = {}
        for cell in plan.CELL_ORDER:
            values = r2_view[surface][cell]
            summaries[surface][cell] = {
                "equal_session_mean": gates.equal_session_mean(values),
                "session_sd": gates.session_sd(values),
                "per_session_r2": dict(sorted(values.items())),
            }
            if cell != plan.BASELINE_CELL:
                deltas[surface][cell] = gates.paired_delta(
                    values, r2_view[surface][plan.BASELINE_CELL])

    verdict_surface = plan.GATES["primary"]["surface"]
    gated_result = gates.evaluate_verdict(
        external_delta=deltas[verdict_surface][plan.GATED_CELL],
        gates_law=plan.GATES,
    )
    mean_anchor_deltas = {
        surface: {
            "pooled_mean": summaries[surface]["POOLED"]["equal_session_mean"],
            "sealed_anchor_mean": float(plan.G00M_ANCHOR_MEANS[surface]),
            "abs_delta": abs(
                summaries[surface]["POOLED"]["equal_session_mean"]
                - float(plan.G00M_ANCHOR_MEANS[surface])),
        }
        for surface in plan.SURFACES
    }
    for surface in plan.SURFACES:
        _require(mean_anchor_deltas[surface]["abs_delta"] <= float(
                     plan.ANCHORS["pooled_vs_sealed_g00m_k4_rows"]["mean_tolerance"]),
                 f"the POOLED equal-session mean missed the sealed G00m/k4 anchor "
                 f"on {surface}")

    safety_all_pass = all(item["exact_match"] for item in anchors_pooled.values())

    def _identity_agreement_pass(item: Mapping[str, Any]) -> bool:
        return bool(
            item["uncapped_equals_reference_bitwise"]
            and (item["capped_equals_reference_bitwise"] is True
                 or item["capped_max_abs_drift_vs_own_members"] <= 1.0e-4))

    stop_conditions = {
        "pooled_anchor_failure": {
            "expression": "any POOLED row missed the sealed m4_activity_only anchor",
            "fired": not safety_all_pass,
        },
        "identity_law_drift": {
            "expression": (
                "any first-scored identity agreement law failed (either uniform "
                "pool vs the from-scratch sequential mean over the members "
                "present at the first scored row)"),
            "fired": not all(_identity_agreement_pass(item)
                             for item in anchors_identity.values()),
        },
        "uncapped_accumulation_drift": {
            "expression": (
                "any UNCAP/CAP final running mean drifted bitwise from the "
                "from-scratch sequential sum (the uncapped pools)"),
            "fired": any(
                rows[surface]["POSTFUSION_ACCUM_UNCAPPED"][session][
                    "final_accumulation_audit"]["bitwise_equal_to_from_scratch"] is False
                for surface in plan.SURFACES
                for session in rows[surface]["POSTFUSION_ACCUM_UNCAPPED"]),
        },
    }
    governing = [name for name, item in stop_conditions.items() if item["fired"]]

    replay_payload = {
        "schema": f"{plan.SCHEMA}_replay_v1",
        "stage": "inference_only_postfusion_identity_probe_local_m2_cpu",
        "status": "REPLAY_COMPLETE",
        "attempt_sha256": attempt_digest,
        "foundation": {
            "sealed_g_family_package": plan.SEALED_G_PACKAGE,
            "sealed_cdm_screen_score": plan.CDM_SCREEN_SCORE_RELATIVE,
            "sealed_cdm_screen_score_sha256": plan.CDM_SCREEN_SCORE_SHA256,
            "sealed_kcurve_terminal": plan.KCURVE_TERMINAL_RELATIVE,
            "sealed_kcurve_terminal_sha256": plan.KCURVE_TERMINAL_SHA256,
            "sealed_memory_law_scan_terminal": plan.MEMORY_LAW_SCAN_TERMINAL_RELATIVE,
            "sealed_memory_law_scan_terminal_sha256": plan.MEMORY_LAW_SCAN_TERMINAL_SHA256,
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
            "sessions": {
                surface: sorted(r2_view[surface][plan.BASELINE_CELL])
                for surface in plan.SURFACES
            },
            "session_meta": session_meta,
        },
        "cells": dict(plan.CELLS),
        "identity_path": {
            **identity_path,
            **{key: value for key, value in plan.IDENTITY_PATH_LAW.items()},
        },
        "frozen_everything_else": dict(plan.FROZEN_EVERYTHING_ELSE),
        "summaries": summaries,
        "paired_deltas_vs_pooled": deltas,
        "rows": {
            f"{surface}|{cell}": dict(sorted(rows[surface][cell].items()))
            for surface in plan.SURFACES for cell in plan.CELL_ORDER
        },
        "anchors": {
            "pooled_vs_sealed_g00m_k4_rows": anchors_pooled,
            "all_exact": safety_all_pass,
            "max_abs_r2_delta_vs_sealed": max(
                abs(item["r2_delta"]) for item in anchors_pooled.values()),
            "mean_anchor_deltas": mean_anchor_deltas,
            "first_scored_identity_agreement": anchors_identity,
            "law": {
                key: (dict(value) if isinstance(value, dict) else value)
                for key, value in plan.ANCHORS.items()
            },
        },
        "gates": {
            "law": dict(plan.GATES),
            "gated_cell": plan.GATED_CELL,
            "verdict_surface": verdict_surface,
            "result": gated_result,
            "report_only": {
                cell: deltas[verdict_surface][cell]
                for cell in ("POSTFUSION_ACCUM_UNCAPPED", "POSTFUSION_EMA_A090")
            },
        },
        "jensen_gap_reading": {
            "question": (
                "how large is the identity-path Jensen gap at the first "
                "scored row (mean after MLP minus MLP after mean over the "
                "same pool members)?"
            ),
            "role": "descriptive only; never selects, never gates",
            "law": dict(plan.ANCHORS["support_pool_jensen_gap"]),
            "per_session": jensen_gaps,
            "max_abs_over_sessions": max(
                item["max_abs"] for item in jensen_gaps.values()),
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
    _require(replay_payload["anchors"]["all_exact"],
             "at least one POOLED anchor failed")
    _require(not governing, f"a stop condition fired: {governing}")
    replay_digest = _publish(root / "replay.json", replay_payload)
    terminal_payload = {
        "schema": f"{plan.SCHEMA}_terminal_v1",
        "status": "TERMINAL",
        "scope": "inference_only_postfusion_identity_probe_local_m2_cpu",
        "replay_sha256": replay_digest,
        "attempt_sha256": attempt_digest,
        "verdict": gated_result["verdict"],
        "verdict_detail": {
            "cell": plan.GATED_CELL,
            "surface": verdict_surface,
            "equal_session_mean_delta": gated_result["delta"]["equal_session_mean_delta"],
            "positive_sessions": gated_result["delta"]["positive_sessions"],
            "breadth_denominator": int(plan.GATES["primary"]["breadth_denominator"]),
            "mean_pass_breadth_fail": gated_result["mean_pass_breadth_fail"],
            "per_session_delta": gated_result["delta"]["per_session_delta"],
            "disposition": (
                "a training-side per-trial-exposure cell becomes worth considering"
                if gated_result["verdict"] == "POSTFUSION_PROMISING"
                else "the placement question closes on frozen weights"
            ),
        },
        "equal_session_means": {
            surface: {
                cell: summaries[surface][cell]["equal_session_mean"]
                for cell in plan.CELL_ORDER
            }
            for surface in plan.SURFACES
        },
        "paired_delta_summary": {
            surface: {
                cell: {
                    "equal_session_mean_delta":
                        deltas[surface][cell]["equal_session_mean_delta"],
                    "positive_sessions": deltas[surface][cell]["positive_sessions"],
                }
                for cell in deltas[surface]
            }
            for surface in plan.SURFACES
        },
        "anchors": {
            "pooled_all_exact": safety_all_pass,
            "max_abs_r2_delta_vs_sealed":
                replay_payload["anchors"]["max_abs_r2_delta_vs_sealed"],
            "pooled_mean_anchor_abs_deltas": {
                surface: mean_anchor_deltas[surface]["abs_delta"]
                for surface in plan.SURFACES
            },
            "first_scored_identity_agreement_all": all(
                _identity_agreement_pass(item) for item in anchors_identity.values()),
        },
        "jensen_gap_reading": {
            "max_abs_over_sessions": replay_payload["jensen_gap_reading"][
                "max_abs_over_sessions"],
        },
        "stop_conditions": replay_payload["stop_conditions"],
        "official_contract_claimed": False,
        "wall_seconds": replay_payload["wall_seconds"],
    }
    _publish(root / "terminal.json", terminal_payload)
    return {
        "status": terminal_payload["status"],
        "replay_sha256": replay_digest,
        "verdict": gated_result["verdict"],
        "external_delta": gated_result["delta"]["equal_session_mean_delta"],
        "external_positive_sessions": gated_result["delta"]["positive_sessions"],
        "stop_conditions_fired": governing,
        "wall_seconds": replay_payload["wall_seconds"],
    }
