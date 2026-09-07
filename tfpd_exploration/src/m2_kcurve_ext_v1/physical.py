"""Physical driver of the M2 k-curve extension (law x B x k grid).

CPU-only, inference-only, one launch.  The receipt law mirrors the sealed
routes: ``attempt.json`` (reserved before any data), ``replay.json`` (the
governing grid), ``terminal.json`` (receipt-only composition).  Nothing
under any frozen result root is created or modified; every sealed executor
is reused by import (the k-curve cells verbatim for FIFO30/STATIC, the
memory-law-scan accumulation law verbatim for UNCAPPED).
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

from . import laws, plan


class KCurveExtPhysicalError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KCurveExtPhysicalError(message)


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
             "the k-curve extension is CPU-only (CUDA_VISIBLE_DEVICES must be empty)")
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


# ---------------------------------------------------------------------------
# Sealed receipts (read-only).
# ---------------------------------------------------------------------------


def _load_sealed_kcurve(repo_root: Path) -> dict[str, Any]:
    path = repo_root / plan.KCURVE_REPLAY_RELATIVE
    _require(_sha256_file(path) == plan.KCURVE_REPLAY_SHA256,
             "sealed k-curve replay drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == "m2_kcurve_v1_replay_v1"
             and payload.get("status") == "REPLAY_COMPLETE",
             "sealed k-curve replay schema drift")
    cdm_rows: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    static_rows: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for surface, sessions in payload["rows"]["cdm_family"].items():
        for session, cells in sessions.items():
            for cell, row in cells.items():
                cdm_rows[(surface, session, int(row["budget"]))] = row
    for surface, sessions in payload["rows"]["static_family"].items():
        for session, cells in sessions.items():
            for cell, row in cells.items():
                static_rows[(surface, session, int(row["budget"]))] = row
    return {
        "cdm_rows": cdm_rows,
        "static_rows": static_rows,
        "supports": payload["supports"],
        "k_grid": [int(item) for item in payload["k_grid"]],
        "endpoint_k": int(payload["endpoint_k"]),
    }


def _load_sealed_memory_scan(repo_root: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    path = repo_root / plan.MEMORY_SCAN_REPLAY_RELATIVE
    _require(_sha256_file(path) == plan.MEMORY_SCAN_REPLAY_SHA256,
             "sealed memory-law-scan replay drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == "m2_memory_law_scan_v1_replay_v1"
             and payload.get("status") == "REPLAY_COMPLETE",
             "sealed memory-law-scan replay schema drift")
    rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for key, sessions in payload["rows"].items():
        surface, budget, policy = key.split("|")
        if budget == "m4" and policy == "UNIFORM_UNCAPPED":
            for session, row in sessions.items():
                rows[(surface, session)] = row
    _require(len(rows) == plan.EXPECTED_WITHIN_SESSIONS + plan.EXPECTED_EXTERNAL_SESSIONS,
             "sealed UNIFORM_UNCAPPED m4 anchor row topology drift")
    return rows


def _load_sealed_reblock10(repo_root: Path) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    path = repo_root / plan.REBLOCK10_REPLAY_RELATIVE
    _require(_sha256_file(path) == plan.REBLOCK10_REPLAY_SHA256,
             "sealed reblock10 replay drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == "m2_reblock10_v1_replay_v1"
             and payload.get("status") == "REPLAY_COMPLETE",
             "sealed reblock10 replay schema drift")
    supports: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    # the sealed reblock10 receipt spells the SAME post-10 windows differently
    # per family (SURFACES_STATIC vs SURFACES_CDM); both spellings must be
    # bound or the B10 proof fails loud, never silently
    surface_maps = {
        "cdm": {"within_post30": "within_post10",
                "external_post30_local": "external_post10_local"},
        "static": {"within_post30": "within_post10",
                   "external_post30_local": "external_post10_query"},
    }
    for spelling in ("cdm", "static"):
        for key, entry in payload["supports"][spelling].items():
            surface, session = key.split("|", 1)
            supports[(spelling, surface, session)] = entry
    return {"supports": supports, "surface_maps": surface_maps}


# ---------------------------------------------------------------------------
# The UNCAPPED cell (the memory-law-scan accumulation law, imported verbatim).
# ---------------------------------------------------------------------------


def uncapped_cell(
    *, torch: Any, model: Any, student: Any, dataset: Any, session: str,
    surface: str, query_rows: tuple[dict[str, Any], ...],
    support: Mapping[str, Any], raw_neural: np.ndarray, selected: np.ndarray,
    B: int, side_mean: np.ndarray, side_std: np.ndarray, device: Any,
    batch_size: int, digest_fn: Any, r2_fn: Any, started: float,
) -> dict[str, Any]:
    """The m2_memory_law_scan_v1 accumulation law over a (B,k) support.

    The pool is the sealed ``FrozenB3SUniformPool`` (the frozen B3S
    encoder's own streaming accumulation: the k support rows first, then
    every completed post-30 query trial, no eviction ever); the decode is
    the sealed ``_predict_with_identity``; the carrier is the SAME (B,k)
    binding as the FIFO30 cell (k=4 the sealed ``_build_memory`` carrier,
    k!=4 the k-row fixed-ridge-by-trial fit).
    """
    from src.m2_kcurve_v1 import physical as kcurve_physical
    from src.m2_memory_law_scan_v1 import memory as memory_law
    from src.m2_memory_law_scan_v1 import physical as scan_physical
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical as cdm_physical

    indices = np.asarray(selected, dtype=np.int64).reshape(-1)
    k = int(indices.size)
    _require(plan.M4 <= k <= plan.ACTIVITY_STACK_LIMIT and int(indices.max()) < B,
             "the UNCAPPED support must lie inside the first-B pool")
    if k == plan.M4:
        sealed_memory = cdm_physical._build_memory(
            system="activity_only", budget=plan.M4, selected=indices,
            support_rates_hz30=support["support_rates_hz30"],
            theta_first30=support["theta30"], support_b3s=support["support_b3s"],
            channel_ids=support["channels"], valid_mask=support["valid_mask"],
        )
        carrier_hz = np.ascontiguousarray(
            np.asarray(sealed_memory.carrier.active_t4), dtype=np.float64)
        binding = "sealed_build_memory_m4_carrier"
    else:
        carrier_hz, _activity = kcurve_physical._krow_carrier_and_activity(support, indices)
        carrier_hz = np.ascontiguousarray(carrier_hz, dtype=np.float64)
        binding = "generalized_k_row_binding"
    side_tensor = scan_physical._side_tensor(torch, carrier_hz, side_mean, side_std)
    pool = memory_law.FrozenB3SUniformPool(
        id_encoder=student.id_encoder,
        support_activities=[
            np.ascontiguousarray(support["support_b3s"][index].activity, dtype=np.float32)
            for index in indices.tolist()
        ],
        channels=int(np.asarray(support["channels"]).size), device=device,
        dtype=torch.float32, torch=torch,
    )
    neural = np.asarray(dataset.neural_data[session], dtype=np.float32)
    behavior = np.asarray(dataset.covariate_data[session], dtype=np.float32)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    starts_joined: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    short_trials = 0
    support_positions = {int(item) for item in indices.tolist()}
    first_scored: Optional[dict[str, Any]] = None
    for position, row in enumerate(query_rows):
        metric_starts = np.asarray(row["metric_starts"], dtype=np.int64)
        identity_sha: Optional[str] = None
        if metric_starts.size:
            with torch.inference_mode():
                identity = pool.identity(side_tensor)
                identity_np = identity.detach().numpy().astype(np.float32, copy=False)
                identity_sha = digest_fn(identity_np)
            prediction = scan_physical._predict_with_identity(
                torch=torch, student=student,
                neural_windows=cdm_physical._windows(neural, metric_starts),
                identity=identity, batch_size=batch_size,
            )
            predictions.append(prediction)
            targets.append(np.ascontiguousarray(
                behavior[metric_starts + cdm_physical.plan.WINDOW_BINS - 1],
                dtype=np.float32))
            starts_joined.append(metric_starts)
            if first_scored is None:
                first_scored = {
                    "row_index": int(position),
                    "commits_before": int(position),
                    "identity_sha256": str(identity_sha),
                }
        if np.asarray(row["causal_starts"], dtype=np.int64).size == 0:
            short_trials += 1
        position_value = int(row["position"])
        _require(position_value not in support_positions,
                 "a first-30 support position entered the post-30 evidence stream")
        pool_before = int(pool.pool_count)
        with torch.inference_mode():
            pool.commit(np.asarray(row["activity"], dtype=np.float32))
        receipts.append({
            "p": int(position), "mw": int(metric_starts.size),
            "pcb": pool_before, "pca": int(pool.pool_count), "id": identity_sha,
        })
        _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                 "the hard timeout fired during an uncapped cell")
    prediction = np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32)
    target = np.ascontiguousarray(np.concatenate(targets, axis=0), dtype=np.float32)
    starts = np.ascontiguousarray(np.concatenate(starts_joined), dtype=np.int64)
    expected = cdm_physical._surface_starts(dataset, session)
    _require(np.array_equal(starts, expected),
             f"{surface}: the scored query differs from the sealed post30 authority")
    with torch.inference_mode():
        final_identity = pool.identity(side_tensor)
        final_identity_sha = digest_fn(
            final_identity.detach().numpy().astype(np.float32, copy=False))
        final_pool_features_sha = digest_fn(pool.pooled_features_numpy())
    return {
        "law": "UNCAPPED",
        "B": int(B),
        "budget": k,
        "support_positions": indices.tolist(),
        "memory_binding": binding,
        "window_count": int(starts.size),
        "ordered_window_starts_sha256": digest_fn(starts),
        "query_starts_sha256": digest_fn(starts),
        "target_sha256": digest_fn(target),
        "prediction_sha256": digest_fn(prediction),
        "fifo_advance_events": int(len(query_rows)),
        "query_row_count": int(len(query_rows)),
        "short_query_trials_without_complete_50_bin_window": int(short_trials),
        "carrier_hz_sha256": digest_fn(carrier_hz),
        "first_scored": first_scored,
        "final_identity_sha256": str(final_identity_sha),
        "final_pool_features_sha256": str(final_pool_features_sha),
        "law_state": pool.law_payload(),
        "per_row_receipts": receipts,
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
                 f"an owned extension module drifted: {relative}")

    from src.m2_kcurve_v1.physical import _bind_namespaces as kcurve_bind

    kcurve_bind(repo_root)

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
    from src.m2_kcurve_v1 import physical as kcurve_physical
    from src.m2_kcurve_v1 import laws as kcurve_laws
    from src.m2_same_query_comparator_v1.core import array_sha256 as static_digest
    from src.m2_same_query_comparator_v1.core import variance_weighted_r2 as static_r2
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical as cdm_physical
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core as pseudo_core

    sealed_kcurve = _load_sealed_kcurve(repo_root)
    sealed_uncapped = _load_sealed_memory_scan(repo_root)
    sealed_reblock10 = _load_sealed_reblock10(repo_root)

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

    # -- Phase A: angle pools, usable census per B, grids, prefix supports ---
    thetas: dict[str, np.ndarray] = {}
    cdm_materials: dict[str, dict[str, Any]] = {}
    static_sessions: dict[str, list[str]] = {}
    for surface, dataset in static_datasets.items():
        sessions = sorted(dataset.calib_trialized_neural_features)
        _require(len(sessions) == expected_counts[surface], f"{surface}: session count drift")
        static_sessions[surface] = sessions
        for session in sessions:
            thetas[f"static|{surface}|{session}"] = np.asarray(
                dataset.calib_trial_target_angles[session], dtype=np.float64)
    for surface, (dataset, raw_sessions) in cdm_datasets.items():
        for session in sorted(dataset.calib_trialized_neural_features):
            _require(session in raw_sessions, f"{surface}/{session}: raw authority absent")
            views = g_replay._g_session_views(raw_sessions, session=session)
            support_base = g_replay.g_support_material(session=session, views=views)
            theta30 = np.asarray(support_base["theta30"], dtype=np.float64)
            _require(theta30.shape == (plan.ACTIVITY_HORIZON,),
                     f"{surface}|{session}: the raw theta30 topology drifted")
            thetas[f"cdm|{surface}|{session}"] = theta30
            cdm_materials[f"{surface}|{session}"] = {"views": views, "support": support_base}

    usable_census: dict[int, dict[str, int]] = {B: {} for B in plan.B_AXIS}
    for key, theta_pool in thetas.items():
        for B in plan.B_AXIS:
            usable_census[B][key] = laws.usable_count_B(theta_pool, B)
    grids = {B: laws.b_grid(usable_census[B], B) for B in plan.B_AXIS}
    for B in plan.B_AXIS:
        _require(grids[B]["effective_grid"] == sorted(set(grids[B]["effective_grid"]))
                 and grids[B]["effective_grid"][0] == plan.K_FLOOR
                 and grids[B]["endpoint_k"] == grids[B]["effective_grid"][-1],
                 f"B={B}: the effective k grid must ascend from the M4 anchor to the endpoint")
    b30_grid = grids[plan.B_REFERENCE]["effective_grid"]

    supports: dict[str, dict[str, dict[int, dict[str, Any]]]] = {
        spelling: {} for spelling in plan.FAMILIES}
    for key, theta_pool in sorted(thetas.items()):
        spelling, surface, session = key.split("|", 2)
        for B in plan.B_AXIS:
            entry = laws.prefix_payload(theta_pool, B, grids[B]["effective_grid"])
            _require(entry["prefix_exact"] and entry["prefix_nested"],
                     f"{key}|B{B}: the D-opt prefix law failed")
            if B == plan.B_REFERENCE:
                _require(entry["b30_order_is_sealed_kcurve_law"]
                         and entry["b30_usable_matches_sealed"],
                         f"{key}: the B30 column is not the sealed k-curve pool law")
            supports[spelling].setdefault(f"{surface}|{session}", {})[B] = entry
        _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                 "the hard timeout fired during the support phase")

    support_proofs: dict[str, Any] = {}
    for spelling, per_session in supports.items():
        sealed_supports = sealed_kcurve["supports"][spelling]
        for key, by_b in sorted(per_session.items()):
            surface, session = key.split("|", 1)
            sealed_entry = sealed_supports[key]
            b30 = by_b[plan.B_REFERENCE]
            proof = {
                "b30_greedy_order_matches_sealed": (
                    b30["greedy_order_positions"] == sealed_entry["greedy_order_positions"]),
                "b30_k4_support_matches_sealed": (
                    b30["supports"][str(plan.M4)] == sealed_entry["supports"][str(plan.M4)]),
                "b30_endpoint_support_matches_sealed": (
                    b30["supports"][str(grids[plan.B_REFERENCE]["endpoint_k"])]
                    == sealed_entry["supports"][str(sealed_kcurve["endpoint_k"])]),
            }
            if spelling == "static":
                sealed_row = sealed_kcurve["static_rows"][(
                    surface, session, plan.M4)]
                proof["b30_k4_support_matches_sealed_row"] = (
                    b30["supports"][str(plan.M4)] == sealed_row["support_positions"])
            else:
                sealed_row = sealed_kcurve["cdm_rows"][(surface, session, plan.M4)]
                proof["b30_k4_support_matches_sealed_row"] = (
                    b30["supports"][str(plan.M4)] == sealed_row["support_positions"])
            reblock_surface = sealed_reblock10["surface_maps"][spelling][surface]
            reblock_entry = sealed_reblock10["supports"].get(
                (spelling, reblock_surface, session))
            _require(reblock_entry is not None,
                     f"the sealed reblock10 receipt lacks the B10 anchor at "
                     f"{spelling}/{reblock_surface}/{session}")
            proof["b10_k4_support_matches_reblock10"] = (
                by_b[10]["supports"][str(plan.M4)]
                == reblock_entry["k4_support_positions"])
            support_proofs[f"{spelling}|{key}"] = proof
            _require(all(value for value in proof.values()),
                     f"the support identity proof failed at {spelling}|{key}")

    cross_family: dict[str, Any] = {}
    for surface in plan.SURFACES:
        for session in static_sessions[surface]:
            key = f"{surface}|{session}"
            for B in plan.B_AXIS:
                agreement = laws.support_agreement(
                    supports["static"][key][B], supports["cdm"][key][B])
                cross_family[f"{key}|B{B}"] = agreement
                _require(agreement["agree"],
                         f"the two families' (B={B}) supports disagree at {key}")

    # -- Phase B: the CDM family (law x B x k) -------------------------------
    cdm_rows: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        law: {surface: {} for surface in plan.SURFACES} for law in plan.LAWS}
    krow_parity: dict[str, Any] = {}
    for surface, (dataset, raw_sessions) in cdm_datasets.items():
        for session in sorted(dataset.calib_trialized_neural_features):
            material = cdm_materials[f"{surface}|{session}"]
            support_base = material["support"]
            views = material["views"]
            raw_neural, raw_starts, _theta, _rates, activities = views
            entry_by_b = supports["cdm"][f"{surface}|{session}"]
            query_rows = cdm_physical._query_trial_rows(
                dataset, session, raw_neural=raw_neural,
                raw_starts=raw_starts, activities=activities,
            )
            for law in plan.LAWS:
                for B in plan.B_AXIS:
                    order = np.asarray(entry_by_b[B]["greedy_order_positions"], dtype=np.int64)
                    for k in grids[B]["effective_grid"]:
                        selected = np.sort(order[:k])
                        if law == "FIFO30":
                            row = kcurve_physical.cdm_k_cell(
                                torch=torch, model=model, dataset=dataset,
                                session=session, surface=surface,
                                query_rows=query_rows, support=support_base,
                                raw_neural=raw_neural, selected=selected,
                                side_mean=side_mean, side_std=side_std,
                                device=device, batch_size=batch_size,
                                digest_fn=pseudo_core.array_sha256,
                                r2_fn=pseudo_core.variance_weighted_r2,
                            )
                            row = {"law": "FIFO30", "B": int(B), **row}
                        else:
                            row = uncapped_cell(
                                torch=torch, model=model, student=student,
                                dataset=dataset, session=session, surface=surface,
                                query_rows=query_rows, support=support_base,
                                raw_neural=raw_neural, selected=selected, B=B,
                                side_mean=side_mean, side_std=side_std,
                                device=device, batch_size=batch_size,
                                digest_fn=pseudo_core.array_sha256,
                                r2_fn=pseudo_core.variance_weighted_r2,
                                started=started,
                            )
                        cell = f"CDM_{law}_B{B}_K{k}"
                        row["cell"] = cell
                        cdm_rows[law][surface].setdefault(session, {})[cell] = {
                            "surface": surface, "session": session, **row}
                        _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                                 "the hard timeout fired during the cdm family")
            # the k-row binding parity anchor on the B30/k=4 prefix support
            sealed_memory = cdm_physical._build_memory(
                system="activity_only", budget=plan.M4,
                selected=np.sort(np.asarray(
                    entry_by_b[plan.B_REFERENCE]["greedy_order_positions"],
                    dtype=np.int64)[: plan.M4]),
                support_rates_hz30=support_base["support_rates_hz30"],
                theta_first30=support_base["theta30"],
                support_b3s=support_base["support_b3s"],
                channel_ids=support_base["channels"],
                valid_mask=support_base["valid_mask"],
            )
            krow_carrier, krow_activity = kcurve_physical._krow_carrier_and_activity(
                support_base, np.sort(np.asarray(
                    entry_by_b[plan.B_REFERENCE]["greedy_order_positions"],
                    dtype=np.int64)[: plan.M4]))
            parity = {
                "law": plan.ANCHORS["krow_binding_parity_b30_k4"]["law"],
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
                     f"the k-row binding drifted at {surface}/{session}")

    # -- Phase C: the STATIC family ------------------------------------------
    static_rows: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        surface: {} for surface in plan.SURFACES}
    static_grid: dict[int, list[int]] = {
        plan.B_REFERENCE: b30_grid,
        **{B: [plan.K_FLOOR] for B in plan.B_PROBE_ROWS_STATIC},
    }
    for surface, dataset in static_datasets.items():
        for session in static_sessions[surface]:
            starts = kcurve_physical._post30_starts(dataset, session)
            entry_by_b = supports["static"][f"{surface}|{session}"]
            for B, ks in static_grid.items():
                order = np.asarray(entry_by_b[B]["greedy_order_positions"], dtype=np.int64)
                for k in ks:
                    row = kcurve_physical.static_act30_cell(
                        torch=torch, student=student, dataset=dataset,
                        session=session, starts=starts, selected=np.sort(order[:k]),
                        digest_fn=static_digest, r2_fn=static_r2,
                        batch_size=batch_size,
                        selection_label=f"doptimal_prefix_k{k}_first{B}",
                    )
                    cell = f"STATIC_B{B}_K{k}"
                    static_rows[surface].setdefault(session, {})[cell] = {
                        "surface": surface, "session": session, "B": int(B),
                        "cell": cell, **row}
                    _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                             "the hard timeout fired during the static family")

    # -- Phase D: anchors -----------------------------------------------------
    tolerance = float(plan.ANCHORS["exact_cpu_matcher"]["r2_tolerance"])

    def _anchor(cell_row: Mapping[str, Any], sealed_row: Mapping[str, Any],
                label: str) -> dict[str, Any]:
        anchored = laws.exact_cpu_anchor(cell_row, sealed_row, r2_tolerance=tolerance)
        _require(anchored["exact_match"], f"anchor failed: {label}")
        return anchored

    fifo30_anchors: dict[str, Any] = {}
    static_anchors: dict[str, Any] = {}
    uncapped_anchors: dict[str, Any] = {}
    for surface in plan.SURFACES:
        for session in sorted(cdm_rows["FIFO30"][surface]):
            for k in b30_grid:
                cell_row = cdm_rows["FIFO30"][surface][session][f"CDM_FIFO30_B{plan.B_REFERENCE}_K{k}"]
                sealed_row = sealed_kcurve["cdm_rows"][(surface, session, k)]
                fifo30_anchors[f"{surface}|{session}|k{k}"] = _anchor(
                    cell_row, sealed_row, f"fifo30/b30/k{k}/{surface}/{session}")
        for session in sorted(static_rows[surface]):
            for k in b30_grid:
                cell_row = static_rows[surface][session][f"STATIC_B{plan.B_REFERENCE}_K{k}"]
                sealed_row = sealed_kcurve["static_rows"][(surface, session, k)]
                static_anchors[f"{surface}|{session}|k{k}"] = _anchor(
                    cell_row, sealed_row, f"static/b30/k{k}/{surface}/{session}")
    for surface in plan.SURFACES:
        for session in sorted(cdm_rows["UNCAPPED"][surface]):
            cell_row = cdm_rows["UNCAPPED"][surface][session][
                f"CDM_UNCAPPED_B{plan.B_REFERENCE}_K{plan.M4}"]
            sealed_row = sealed_uncapped[(surface, session)]
            uncapped_anchors[f"{surface}|{session}"] = _anchor(
                cell_row, sealed_row, f"uncapped/b30/k4/{surface}/{session}")

    # pure-data pairing within every family/surface/session across ALL cells
    pairing: dict[str, Any] = {}
    for family, family_rows in (("cdm_fifo30", cdm_rows["FIFO30"]),
                                ("cdm_uncapped", cdm_rows["UNCAPPED"]),
                                ("static", static_rows)):
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

    # the capacity census and the external law-equivalence anchor
    capacity_census: dict[str, Any] = {}
    law_equivalence: dict[str, Any] = {}
    for surface in plan.SURFACES:
        for session in sorted(cdm_rows["FIFO30"][surface]):
            advance = int(cdm_rows["FIFO30"][surface][session][
                f"CDM_FIFO30_B{plan.B_REFERENCE}_K{plan.M4}"]["fifo_advance_events"])
            capacities = {str(k): plan.ACTIVITY_STACK_LIMIT - int(k)
                          for k in grids[plan.B_REFERENCE]["effective_grid"]}
            capacity_census[f"{surface}|{session}"] = {
                "advance_events": advance,
                "fifo_capacity_at_k": capacities,
                "eviction_free_at_every_grid_k": all(
                    advance <= capacity for capacity in capacities.values()),
            }
            for B in plan.B_AXIS:
                for k in grids[B]["effective_grid"]:
                    fifo = cdm_rows["FIFO30"][surface][session][f"CDM_FIFO30_B{B}_K{k}"]
                    uncapped = cdm_rows["UNCAPPED"][surface][session][f"CDM_UNCAPPED_B{B}_K{k}"]
                    identical = {
                        "prediction_sha256": (fifo["prediction_sha256"]
                                              == uncapped["prediction_sha256"]),
                        "r2_bitwise": float(fifo["r2"]) == float(uncapped["r2"]),
                    }
                    entry = {
                        "capacity": plan.ACTIVITY_STACK_LIMIT - int(k),
                        "eviction_free": advance <= plan.ACTIVITY_STACK_LIMIT - int(k),
                        "identical": identical,
                        "r2_delta": float(uncapped["r2"]) - float(fifo["r2"]),
                    }
                    law_equivalence[f"{surface}|{session}|B{B}K{k}"] = entry
                    if entry["eviction_free"]:
                        _require(all(identical.values()),
                                 f"the laws disagree where no eviction can occur: "
                                 f"{surface}/{session}/B{B}/K{k}")
    anchors_summary = {
        "b30_support_identity_all_exact": bool(all(
            all(value for value in proof.values())
            for proof in support_proofs.values())),
        "b10_k4_vs_reblock10_all_exact": bool(all(
            proof.get("b10_k4_support_matches_reblock10", False)
            for proof in support_proofs.values())),
        "cross_family_support_agreement": bool(all(
            item["agree"] for item in cross_family.values())),
        "fifo30_b30_all_exact": bool(all(
            item["exact_match"] for item in fifo30_anchors.values())),
        "static_b30_all_exact": bool(all(
            item["exact_match"] for item in static_anchors.values())),
        "uncapped_b30_k4_all_exact": bool(all(
            item["exact_match"] for item in uncapped_anchors.values())),
        "krow_binding_parity_all": bool(all(
            item["carrier_bitwise_equal"] and item["initial_activity_stack_bitwise_equal"]
            for item in krow_parity.values())),
        "pure_data_pairing_all_exact": bool(all(
            item["exact_match"] for item in pairing.values())),
        "external_law_equivalence_holds": bool(all(
            (not entry["eviction_free"]) or all(entry_identical for entry_identical
                                               in entry["identical"].values())
            for entry in law_equivalence.values())),
        "fifo30_max_abs_r2_delta": max(
            (abs(item["r2_delta"]) for item in fifo30_anchors.values()), default=0.0),
        "static_max_abs_r2_delta": max(
            (abs(item["r2_delta"]) for item in static_anchors.values()), default=0.0),
        "uncapped_max_abs_r2_delta": max(
            (abs(item["r2_delta"]) for item in uncapped_anchors.values()), default=0.0),
    }
    boolean_flags = [value for value in anchors_summary.values() if isinstance(value, bool)]
    stop_conditions = {
        "anchor_or_binding_failure": {
            "expression": (
                "any support-identity proof, reblock10 B10 proof, cross-family "
                "agreement, receipt anchor (FIFO30/STATIC/UNCAPPED), k-row "
                "parity, pure-data pairing or external law-equivalence check "
                "failed"),
            "fired": not all(boolean_flags),
            "evidence": anchors_summary,
        },
    }
    governing = [name for name, item in stop_conditions.items() if item["fired"]]

    # -- Phase E: the pre-registered readouts --------------------------------
    def _cdm_values(law: str, surface: str, B: int, k: int) -> dict[str, float]:
        return {session: float(cdm_rows[law][surface][session][
            f"CDM_{law}_B{B}_K{k}"]["r2"])
            for session in sorted(cdm_rows[law][surface])}

    def _static_values(surface: str, B: int, k: int) -> dict[str, float]:
        return {session: float(static_rows[surface][session][
            f"STATIC_B{B}_K{k}"]["r2"])
            for session in sorted(static_rows[surface])}

    q1_paired: dict[str, Any] = {}
    for surface in plan.SURFACES:
        for B in plan.B_AXIS:
            for k in grids[B]["effective_grid"]:
                q1_paired[f"{surface}|B{B}K{k}"] = laws.law_paired_delta(
                    _cdm_values("UNCAPPED", surface, B, k),
                    _cdm_values("FIFO30", surface, B, k))
    b30 = plan.B_REFERENCE
    b30_endpoint = grids[b30]["endpoint_k"]
    slopes = {
        "FIFO30": laws.decline_slope(
            {k: _cdm_values("FIFO30", "external_post30_local", b30, k)
             for k in grids[b30]["effective_grid"]},
            k_low=plan.M4, k_high=b30_endpoint),
        "UNCAPPED": laws.decline_slope(
            {k: _cdm_values("UNCAPPED", "external_post30_local", b30, k)
             for k in grids[b30]["effective_grid"]},
            k_low=plan.M4, k_high=b30_endpoint),
        "STATIC": laws.decline_slope(
            {k: _static_values("external_post30_local", b30, k)
             for k in static_grid[b30]},
            k_low=plan.M4, k_high=b30_endpoint),
    }
    q1 = {
        "decline_slopes_external_b30_k4_to_all_usable": slopes,
        **laws.q1_verdict(slope_fifo30=slopes["FIFO30"],
                          slope_uncapped=slopes["UNCAPPED"],
                          slope_static=slopes["STATIC"]),
        "paired_deltas_uncapped_minus_fifo30": q1_paired,
        "external_law_invariance": {
            key: entry for key, entry in law_equivalence.items()
            if key.startswith("external_post30_local|")},
        "capacity_census": capacity_census,
        "within_surface_law_contrast": {
            key: entry for key, entry in q1_paired.items()
            if key.startswith("within_post30|")},
    }

    q2: dict[str, Any] = {}
    for surface in plan.SURFACES:
        for B in plan.B_PROBE_ROWS_STATIC:
            for k in laws.matched_ks(grids[B]["effective_grid"],
                                     grids[b30]["effective_grid"]):
                q2[f"{surface}|B{B}_minus_B30|k{k}"] = laws.law_paired_delta(
                    _cdm_values("UNCAPPED", surface, B, k),
                    _cdm_values("UNCAPPED", surface, b30, k))

    external_means = {
        f"{law}|B{B}|K{k}": laws.equal_session_mean(
            _cdm_values(law, "external_post30_local", B, k))
        for law in plan.LAWS for B in plan.B_AXIS for k in grids[B]["effective_grid"]
    }
    q3 = laws.best_cell(external_means)

    kcurve_tables: dict[str, Any] = {}
    for law in plan.LAWS:
        for surface in plan.SURFACES:
            for B in plan.B_AXIS:
                per_k = {k: _cdm_values(law, surface, B, k)
                         for k in grids[B]["effective_grid"]}
                kcurve_tables[f"cdm|{law}|B{B}|{surface}"] = {
                    "k_curve_table": kcurve_laws.kcurve_table(
                        per_k, endpoint_k=grids[B]["endpoint_k"]),
                    "monotonicity": kcurve_laws.monotonicity(
                        per_k, endpoint_k=grids[B]["endpoint_k"]),
                }

    replay_payload = {
        "schema": f"{plan.SCHEMA}_replay_v1",
        "stage": "inference_only_law_bpool_kcurve_extension_local_m2_cpu",
        "status": "REPLAY_COMPLETE",
        "attempt_sha256": attempt_digest,
        "authority": (
            "operator work order 2026-09-02: the M2 k-curve extension (law "
            "FIFO30 vs UNCAPPED x pool B in {10,20,30} x D-opt prefix k), "
            "CPU-only, sealed anchors read from receipts"
        ),
        "foundation": {
            "sealed_kcurve_replay": plan.KCURVE_REPLAY_RELATIVE,
            "sealed_kcurve_replay_sha256": plan.KCURVE_REPLAY_SHA256,
            "sealed_kcurve_terminal": plan.KCURVE_TERMINAL_RELATIVE,
            "sealed_kcurve_terminal_sha256": plan.KCURVE_TERMINAL_SHA256,
            "sealed_memory_scan_replay": plan.MEMORY_SCAN_REPLAY_RELATIVE,
            "sealed_memory_scan_replay_sha256": plan.MEMORY_SCAN_REPLAY_SHA256,
            "sealed_reblock10_replay": plan.REBLOCK10_REPLAY_RELATIVE,
            "sealed_reblock10_replay_sha256": plan.REBLOCK10_REPLAY_SHA256,
            "kcurve_package": plan.KCURVE_PACKAGE,
            "memory_scan_package": plan.MEMORY_SCAN_PACKAGE,
            "reblock10_package": plan.REBLOCK10_PACKAGE,
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
        "b_pool_law": dict(plan.B_POOL_LAW),
        "memory_laws": dict(plan.MEMORY_LAWS),
        "law_equivalence_disclosure": plan.LAW_EQUIVALENCE_DISCLOSURE,
        "frozen_everything_else": dict(plan.FROZEN_EVERYTHING_ELSE),
        "b_axis": list(plan.B_AXIS),
        "grids": {str(B): grids[B] for B in plan.B_AXIS},
        "static_grid": {str(B): list(ks) for B, ks in static_grid.items()},
        "supports": {
            spelling: {key: {str(B): entry for B, entry in by_b.items()}
                       for key, by_b in sorted(per_session.items())}
            for spelling, per_session in supports.items()
        },
        "support_proofs": dict(sorted(support_proofs.items())),
        "cross_family_support_agreement": cross_family,
        "readout_law": dict(plan.READOUT_LAW),
        "kcurve_tables": kcurve_tables,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "anchors": {
            "law": {key: (dict(value) if isinstance(value, dict) else value)
                    for key, value in plan.ANCHORS.items()},
            "fifo30_b30": dict(sorted(fifo30_anchors.items())),
            "static_b30": dict(sorted(static_anchors.items())),
            "uncapped_b30_k4": dict(sorted(uncapped_anchors.items())),
            "krow_binding_parity": dict(sorted(krow_parity.items())),
            "pure_data_pairing": dict(sorted(pairing.items())),
            "law_equivalence": dict(sorted(law_equivalence.items())),
            **anchors_summary,
        },
        "rows": {
            "cdm_fifo30": {surface: dict(sorted(sessions.items()))
                           for surface, sessions in cdm_rows["FIFO30"].items()},
            "cdm_uncapped": {surface: dict(sorted(sessions.items()))
                             for surface, sessions in cdm_rows["UNCAPPED"].items()},
            "static": {surface: dict(sorted(sessions.items()))
                       for surface, sessions in static_rows.items()},
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
    _require(not governing, f"stop conditions fired before publication: {governing}")
    replay_digest = _publish(root / "replay.json", replay_payload)

    def _cdm_mean_table(law: str) -> dict[str, dict[str, float]]:
        table: dict[str, dict[str, float]] = {}
        for surface in plan.SURFACES:
            for B in plan.B_AXIS:
                for k in grids[B]["effective_grid"]:
                    table.setdefault(surface, {})[f"B{B}K{k}"] = laws.equal_session_mean(
                        _cdm_values(law, surface, B, k))
        return table

    def _static_mean_table() -> dict[str, dict[str, float]]:
        table: dict[str, dict[str, float]] = {}
        for surface in plan.SURFACES:
            for B, ks in static_grid.items():
                for k in ks:
                    table.setdefault(surface, {})[f"B{B}K{k}"] = laws.equal_session_mean(
                        _static_values(surface, B, k))
        return table

    terminal_payload = {
        "schema": f"{plan.SCHEMA}_terminal_v1",
        "status": "TERMINAL",
        "scope": "inference_only_law_bpool_kcurve_extension_local_m2_cpu",
        "replay_sha256": replay_digest,
        "attempt_sha256": attempt_digest,
        "b_axis": list(plan.B_AXIS),
        "k_grids": {str(B): grids[B]["effective_grid"] for B in plan.B_AXIS},
        "static_grid": {str(B): list(ks) for B, ks in static_grid.items()},
        "equal_session_means": {
            "cdm": {law: _cdm_mean_table(law) for law in plan.LAWS},
            "static": _static_mean_table(),
        },
        "per_session_r2": {
            "cdm": {
                law: {
                    surface: {
                        f"B{B}K{k}": _cdm_values(law, surface, B, k)
                        for B in plan.B_AXIS for k in grids[B]["effective_grid"]
                    } for surface in plan.SURFACES
                } for law in plan.LAWS
            },
            "static": {
                surface: {
                    f"B{B}K{k}": _static_values(surface, B, k)
                    for B, ks in static_grid.items() for k in ks
                } for surface in plan.SURFACES
            },
        },
        "q1": {
            "verdict": q1["verdict"],
            "verdict_rule": plan.READOUT_LAW["q1_law_contrast"]["verdict_rule"],
            "decline_slopes_external_b30_k4_to_all_usable": slopes,
            "excess_decline_uncapped_vs_static": q1["excess_decline_uncapped_vs_static"],
            "decline_removed_fraction": q1["decline_removed_fraction"],
            "paired_delta_means_uncapped_minus_fifo30": {
                key: entry["equal_session_mean_delta"]
                for key, entry in q1_paired.items()
                if isinstance(entry, dict)
            },
            "external_law_invariance_all_identical": all(
                all(entry["identical"].values())
                for key, entry in law_equivalence.items()
                if key.startswith("external_post30_local|")),
            "capacity_census": capacity_census,
        },
        "q2": {
            "expression": plan.READOUT_LAW["q2_pool_narrowing"]["expression"],
            "mean_deltas": {
                key: entry["equal_session_mean_delta"]
                for key, entry in q2.items()
            },
            "per_contrast": q2,
        },
        "q3": {
            "best_external_cell": {
                "law": q3["law"], "B": q3["B"], "k": q3["k"], "value": q3["value"],
            },
            "ranking": q3["ranking"],
        },
        "anchors": anchors_summary,
        "stop_conditions": replay_payload["stop_conditions"],
        "official_contract_claimed": False,
        "wall_seconds": replay_payload["wall_seconds"],
    }
    _publish(root / "terminal.json", terminal_payload)
    return {
        "status": terminal_payload["status"],
        "replay_sha256": replay_digest,
        "q1_verdict": q1["verdict"],
        "q1_slopes": slopes,
        "q2_mean_deltas": terminal_payload["q2"]["mean_deltas"],
        "q3_best": terminal_payload["q3"]["best_external_cell"],
        "anchors_all_exact": all(boolean_flags),
        "stop_conditions_fired": governing,
        "wall_seconds": replay_payload["wall_seconds"],
    }
