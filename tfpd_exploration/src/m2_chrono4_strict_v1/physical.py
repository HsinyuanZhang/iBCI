"""Physical driver of the M2 chrono4 strict-caliber early-start cell.

CPU-only, inference-only, one launch.  The receipt law mirrors the sealed
routes: ``attempt.json`` (reserved before any data), ``replay.json`` (the
governing grid), ``terminal.json`` (receipt-only composition).  Nothing under
any frozen result root is created or modified; every sealed executor is
reused by import.
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


class Chrono4StrictError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Chrono4StrictError(message)


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
             "the chrono4 strict cell is CPU-only (CUDA_VISIBLE_DEVICES must be empty)")
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


def _load_sealed_static_rows(repo_root: Path) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    path = repo_root / plan.COMPARATOR_SCORE_RELATIVE
    _require(_sha256_file(path) == plan.COMPARATOR_SCORE_SHA256,
             "sealed comparator score drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == "m2_same_query_comparator_v1"
             and payload.get("status") == "TERMINAL", "sealed comparator schema drift")
    rows: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in payload["rows"]:
        if row.get("cell") in ("t4_ridge_static_m4", "t4_ridge_static_m10", "t4_ridge_static_m30"):
            rows[(str(row["surface"]), str(row["session"]), int(row["budget"]))] = row
    _require(len(rows) == (plan.EXPECTED_WITHIN_SESSIONS + plan.EXPECTED_EXTERNAL_SESSIONS)
             * len((plan.M4, plan.M10, plan.M30)),
             "sealed static anchor row topology drift")
    return rows


def _load_sealed_cdm_rows(repo_root: Path) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    path = repo_root / plan.CDM_SCREEN_SCORE_RELATIVE
    _require(_sha256_file(path) == plan.CDM_SCREEN_SCORE_SHA256, "sealed CDM screen score drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == "m2_precision_cdm_v2_screen_v1"
             and payload.get("status") == "TERMINAL", "sealed CDM screen schema drift")
    rows: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in payload["rows"]:
        if row.get("cell") in ("m4_activity_only", "m10_activity_only"):
            rows[(str(row["surface"]), str(row["session_id"]), int(row["budget"]))] = row
    _require(len(rows) == (plan.EXPECTED_WITHIN_SESSIONS + plan.EXPECTED_EXTERNAL_SESSIONS) * 2,
             "sealed activity_only anchor row topology drift")
    return rows


# ---------------------------------------------------------------------------
# The static family: the sealed t4_ridge_static law, support swapped.
# ---------------------------------------------------------------------------


def static_cell(
    *, torch: Any, student: Any, dataset: Any, session: str, surface: str,
    selected: np.ndarray, digest_fn: Any, variance_weighted_r2: Any,
    select_activity_rows: Any, session_starts: Any, batch_size: int,
) -> dict[str, Any]:
    """The sealed ``t4_ridge_static_m4`` decode law over an arbitrary 4-support."""
    from tfpd_exploration.src.calibration_budget_comparators_v1 import fit_ridge_t4

    support = np.asarray(selected, dtype=np.int64).reshape(-1)
    _require(support.size == plan.M4, "the static chrono4 cell needs exactly four trials")
    calibration = np.asarray(dataset.calib_trialized_neural_features[session], dtype=np.float32)
    activity = select_activity_rows(
        calibration, selected_indices=support, activity_budget=plan.M4,
    )
    sums = np.asarray(dataset.calib_trial_spike_sums[session][support], dtype=np.float64)
    lengths = np.asarray(dataset.calib_trial_lengths[session][support], dtype=np.float64)
    angles = np.asarray(dataset.calib_trial_target_angles[session][support], dtype=np.float64)
    usable = np.isfinite(angles)
    _require(int(usable.sum()) >= 3,
             f"{session}: the four-trial support has fewer than three directional trials")
    rates = np.ascontiguousarray(sums[usable] / lengths[usable, None], dtype=np.float64)
    raw, evidence = fit_ridge_t4(
        rates, angles[usable], normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA,
    )
    mean = np.asarray(dataset.side_feature_mean, dtype=np.float32)
    std = np.asarray(dataset.side_feature_std, dtype=np.float32)
    _require(mean.shape == std.shape == (4,) and np.all(std > 0), "frozen T4 normalizer drift")
    side = np.ascontiguousarray((raw - mean) / std, dtype=np.float32)
    _require(side.shape == (plan.CHANNELS, 4) and np.isfinite(side).all(),
             "normalized chrono4 T4 drift")
    starts = session_starts(dataset, session, surface)
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
        "surface": surface, "session": session, "cell": "STATIC_CHRONO4",
        "budget": plan.M4,
        "support_positions": support.tolist(),
        "window_count": int(starts.size),
        "ordered_window_starts_sha256": digest_fn(starts),
        "target_sha256": digest_fn(target),
        "prediction_sha256": digest_fn(prediction),
        "activity_sha256": digest_fn(activity),
        "side_evidence": {
            **{key: evidence[key] for key in (
                "normalized_lambda", "design_rank", "design_condition", "gcv")},
            "selection": "chronological_first4_directional",
            "selected_indices": support.tolist(),
            "selected_indices_sha256": digest_fn(support),
            "usable_directional_trials": int(usable.sum()),
            "raw_t4_sha256": digest_fn(raw),
            "normalized_t4_sha256": digest_fn(side),
        },
        "r2": float(variance_weighted_r2(target, prediction)),
        "parameter_updates": 0,
    }


# ---------------------------------------------------------------------------
# The CDM family: the sealed m4_activity_only law, support swapped.
# ---------------------------------------------------------------------------


def cdm_cell(
    *, torch: Any, model: Any, dataset: Any, session: str, surface: str,
    selected: np.ndarray, views: tuple, support: Mapping[str, Any],
    query_rows: tuple, side_mean: np.ndarray, side_std: np.ndarray,
    device: Any, batch_size: int, cdm_physical: Any,
) -> dict[str, Any]:
    """The sealed ``m4_activity_only`` law over an arbitrary 4-support."""
    memory = cdm_physical._build_memory(
        system="activity_only", budget=plan.M4, selected=np.asarray(selected, dtype=np.int64),
        support_rates_hz30=support["support_rates_hz30"],
        theta_first30=support["theta30"], support_b3s=support["support_b3s"],
        channel_ids=support["channels"], valid_mask=support["valid_mask"],
    )
    result = cdm_physical._score_system(
        torch=torch, model=model, dataset=dataset, session=session,
        memory=memory, query_rows=query_rows, side_mean=side_mean, side_std=side_std,
        raw_neural=views[0], device=device, batch_size=batch_size,
    )
    return {
        "surface": surface, "session": session, "cell": "CDM_CHRONO4",
        "budget": plan.M4,
        "support_positions": [int(item) for item in np.asarray(selected, dtype=np.int64)],
        "window_count": int(result["window_count"]),
        "ordered_window_starts_sha256": str(result["query_starts_sha256"]),
        "target_sha256": str(result["target_sha256"]),
        "prediction_sha256": str(result["prediction_sha256"]),
        "r2": float(result["r2"]),
        "accepted_carrier_updates": int(result["accepted_carrier_updates"]),
        "carrier_updates": int(result["accepted_carrier_updates"]),
        "parameter_updates": 0,
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
                 f"an owned chrono4 module drifted: {relative}")

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
    from src.m2_same_query_comparator_v1.core import (
        array_sha256 as static_digest,
    )
    from src.m2_same_query_comparator_v1.core import variance_weighted_r2 as static_r2
    from src.m2_t4_activity_budget_screen_v1.core import select_activity_rows
    from src.m2_t4_activity_budget_screen_v1.physical import _session_starts
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical as cdm_physical
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core as pseudo_core

    sealed_static = _load_sealed_static_rows(repo_root)
    sealed_cdm = _load_sealed_cdm_rows(repo_root)

    static_datasets = {
        "within_post30": data_module.train_dataset,
        "external_official_query": data_module.val_heldout_dataset,
    }
    cdm_datasets = {
        "within_post30": (data_module.train_dataset, data_module.train_calib_heldin_sessions),
        "external_post30_local": (
            data_module.val_heldout_dataset, data_module.val_calib_heldout_sessions),
    }
    _require(data_module.train_dataset is not None
             and data_module.val_heldout_dataset is not None,
             "the frozen M2 runtime must expose both datasets")
    for dataset in (data_module.train_dataset, data_module.val_heldout_dataset):
        _require(np.array_equal(np.asarray(dataset.side_feature_mean, dtype=np.float32), side_mean)
                 and np.array_equal(np.asarray(dataset.side_feature_std, dtype=np.float32), side_std),
                 "dataset/metadata T4 normalizer disagreement")
    expected_counts = {
        "within_post30": plan.EXPECTED_WITHIN_SESSIONS,
        "external_official_query": plan.EXPECTED_EXTERNAL_SESSIONS,
        "external_post30_local": plan.EXPECTED_EXTERNAL_SESSIONS,
    }
    fidelity_roster = {
        "static": (
            [("external_official_query", name) for name in sorted(
                data_module.val_heldout_dataset.calib_trialized_neural_features)[:2]]
            + [("within_post30", name) for name in sorted(
                data_module.train_dataset.calib_trialized_neural_features)[:1]]
        ),
        "cdm": (
            [("external_post30_local", name) for name in sorted(
                data_module.val_heldout_dataset.calib_trialized_neural_features)[:2]]
            + [("within_post30", name) for name in sorted(
                data_module.train_dataset.calib_trialized_neural_features)[:1]]
        ),
    }
    started = time.monotonic()

    static_rows: dict[str, dict[str, dict[str, Any]]] = {
        surface: {} for surface in plan.SURFACES_STATIC
    }
    static_pairing: dict[str, Any] = {}
    static_fidelity: dict[str, Any] = {}
    static_supports: dict[str, Any] = {}
    dopt_proofs: dict[str, Any] = {}
    scatter_evidence: dict[str, Any] = {}

    for surface in plan.SURFACES_STATIC:
        dataset = static_datasets[surface]
        sessions = sorted(dataset.calib_trialized_neural_features)
        _require(len(sessions) == expected_counts[surface], f"{surface}: session count drift")
        for session in sessions:
            angles = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
            chrono = laws.chrono4_support(angles)
            chrono_support = np.asarray(chrono["support_positions"], dtype=np.int64)
            dopt_indices = laws.dopt_support_indices(angles)
            sealed_m4 = sealed_static[(surface, session, plan.M4)]
            proof = laws.dopt_proof(dopt_indices, sealed_m4["side_evidence"]["selected_indices"])
            dopt_proofs[f"static|{surface}|{session}"] = {
                **proof,
                "sealed_law_selection": str(sealed_m4["side_evidence"]["selection"]),
            }
            _require(proof["exact_match"],
                     f"the sealed static m4 indices are not the recomputed D-opt law at "
                     f"{surface}/{session}")
            scatter_evidence[f"static|{surface}|{session}"] = {
                "dopt": {
                    **laws.index_scatter(dopt_indices),
                    **laws.scatter_statistics(angles[dopt_indices]),
                    "pool_depth_required_trials": plan.ACTIVITY_HORIZON,
                    "earliest_decode_trial_1based": plan.ACTIVITY_HORIZON + 1,
                },
                "chrono4": {
                    **laws.index_scatter(chrono_support),
                    **laws.scatter_statistics(angles[chrono_support]),
                    "pool_depth_required_trials": chrono["pool_depth_required_trials"],
                    "earliest_decode_trial_1based": chrono["earliest_decode_trial_1based"],
                },
            }
            static_supports[f"{surface}|{session}"] = chrono
            row = static_cell(
                torch=torch, student=student, dataset=dataset, session=session,
                surface=surface, selected=chrono_support, digest_fn=static_digest,
                variance_weighted_r2=static_r2, select_activity_rows=select_activity_rows,
                session_starts=_session_starts, batch_size=batch_size,
            )
            row["support_payload"] = chrono
            static_rows[surface][session] = row
            static_pairing[f"{surface}|{session}"] = laws.pure_data_pairing(row, sealed_m4)
            _require(static_pairing[f"{surface}|{session}"]["exact_match"],
                     f"the STATIC_CHRONO4 pure-data pairing failed at {surface}/{session}")
            if (surface, session) in fidelity_roster["static"]:
                fidelity = static_cell(
                    torch=torch, student=student, dataset=dataset, session=session,
                    surface=surface, selected=dopt_indices, digest_fn=static_digest,
                    variance_weighted_r2=static_r2,
                    select_activity_rows=select_activity_rows,
                    session_starts=_session_starts, batch_size=batch_size,
                )
                fidelity["cell"] = "STATIC_FIDELITY_DOPT"
                anchored = laws.fidelity_anchor(
                    fidelity, sealed_m4,
                    r2_tolerance=float(plan.ANCHORS["executor_fidelity"]["r2_tolerance"]),
                )
                static_fidelity[f"{surface}|{session}"] = {
                    "row": fidelity, "anchor": anchored,
                }
                _require(anchored["exact_match"],
                         f"the static executor fidelity anchor failed at {surface}/{session}")
            _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                     "the hard timeout fired during the static family")

    cdm_rows: dict[str, dict[str, dict[str, Any]]] = {
        surface: {} for surface in plan.SURFACES_CDM
    }
    cdm_pairing: dict[str, Any] = {}
    cdm_fidelity: dict[str, Any] = {}
    cdm_supports: dict[str, Any] = {}

    for surface in plan.SURFACES_CDM:
        dataset, raw_sessions = cdm_datasets[surface]
        sessions = sorted(dataset.calib_trialized_neural_features)
        _require(len(sessions) == expected_counts[surface], f"{surface}: session count drift")
        for session in sessions:
            _require(session in raw_sessions, f"{surface}/{session}: raw authority absent")
            views = g_replay._g_session_views(raw_sessions, session=session)
            support_base = g_replay.g_support_material(session=session, views=views)
            query_rows = g_replay.g_query_rows(ds=dataset, session=session, views=views)
            theta30 = np.asarray(support_base["theta30"], dtype=np.float64)
            chrono = laws.chrono4_support(theta30)
            chrono_support = np.asarray(chrono["support_positions"], dtype=np.int64)
            dopt_indices = np.asarray(support_base["selected"], dtype=np.int64)
            sealed_m4 = sealed_cdm[(surface, session, plan.M4)]
            unified = laws.dopt_support_indices(theta30)
            sealed_law = pseudo_core.select_m4_support(theta30)
            proof = laws.dopt_proof(unified, sealed_m4["support_indices"])
            dopt_proofs[f"cdm|{surface}|{session}"] = {
                **proof,
                "sealed_law_selection": "doptimal_noncentre_first30",
                "pseudo_core_law_agrees": bool(
                    np.array_equal(np.asarray(sealed_law, dtype=np.int64), unified)),
                "g_support_material_agrees": bool(np.array_equal(dopt_indices, unified)),
            }
            _require(proof["exact_match"] and dopt_proofs[f"cdm|{surface}|{session}"][
                "pseudo_core_law_agrees"],
                     f"the sealed cdm m4 indices are not the recomputed D-opt law at "
                     f"{surface}/{session}")
            scatter_evidence[f"cdm|{surface}|{session}"] = {
                "dopt": {
                    **laws.index_scatter(dopt_indices),
                    **laws.scatter_statistics(theta30[dopt_indices]),
                    "pool_depth_required_trials": plan.ACTIVITY_HORIZON,
                    "earliest_decode_trial_1based": plan.ACTIVITY_HORIZON + 1,
                },
                "chrono4": {
                    **laws.index_scatter(chrono_support),
                    **laws.scatter_statistics(theta30[chrono_support]),
                    "pool_depth_required_trials": chrono["pool_depth_required_trials"],
                    "earliest_decode_trial_1based": chrono["earliest_decode_trial_1based"],
                },
            }
            cdm_supports[f"{surface}|{session}"] = chrono
            row = cdm_cell(
                torch=torch, model=model, dataset=dataset, session=session, surface=surface,
                selected=chrono_support, views=views, support=support_base,
                query_rows=query_rows, side_mean=side_mean, side_std=side_std,
                device=device, batch_size=batch_size, cdm_physical=cdm_physical,
            )
            row["support_payload"] = chrono
            cdm_rows[surface][session] = row
            cdm_pairing[f"{surface}|{session}"] = laws.pure_data_pairing(row, sealed_m4)
            _require(cdm_pairing[f"{surface}|{session}"]["exact_match"],
                     f"the CDM_CHRONO4 pure-data pairing failed at {surface}/{session}")
            if (surface, session) in fidelity_roster["cdm"]:
                fidelity = cdm_cell(
                    torch=torch, model=model, dataset=dataset, session=session,
                    surface=surface, selected=dopt_indices, views=views,
                    support=support_base, query_rows=query_rows, side_mean=side_mean,
                    side_std=side_std, device=device, batch_size=batch_size,
                    cdm_physical=cdm_physical,
                )
                fidelity["cell"] = "CDM_FIDELITY_DOPT"
                anchored = laws.fidelity_anchor(
                    fidelity, sealed_m4,
                    r2_tolerance=float(plan.ANCHORS["executor_fidelity"]["r2_tolerance"]),
                )
                cdm_fidelity[f"{surface}|{session}"] = {
                    "row": fidelity, "anchor": anchored,
                }
                _require(anchored["exact_match"],
                         f"the cdm executor fidelity anchor failed at {surface}/{session}")
            _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                     "the hard timeout fired during the cdm family")

    # -- the chrono4/D-opt support agreement across families ----------------
    support_agreement = {}
    for surface, session in [(s, name) for s in plan.SURFACES_STATIC
                             for name in static_rows[s]]:
        key = f"{surface}|{session}"
        cdm_key = None
        for cdm_surface in plan.SURFACES_CDM:
            if session in cdm_rows[cdm_surface]:
                cdm_key = f"{cdm_surface}|{session}"
        if cdm_key is not None:
            support_agreement[key] = {
                "static_support": static_supports[key]["support_positions"],
                "cdm_support": cdm_supports[cdm_key]["support_positions"],
                "agree": (static_supports[key]["support_positions"]
                          == cdm_supports[cdm_key]["support_positions"]),
            }
    _require(all(item["agree"] for item in support_agreement.values()),
             "the two families' chronological-first-4 supports disagree")

    # -- summaries, sealed reuse, selection contributions, verdicts ----------
    def _session_values(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
        return {name: float(rows[name]["r2"]) for name in sorted(rows)}

    def _sealed_static_values(surface: str, budget: int) -> dict[str, float]:
        return {
            session: float(sealed_static[(surface, session, budget)]["r2"])
            for session in sorted(static_rows[surface])
        }

    def _sealed_cdm_values(surface: str, budget: int) -> dict[str, float]:
        return {
            session: float(sealed_cdm[(surface, session, budget)]["r2"])
            for session in sorted(cdm_rows[surface])
        }

    static_summaries = {
        surface: {
            "STATIC_CHRONO4": laws.equal_session_mean(_session_values(static_rows[surface])),
            "sealed_dopt_m4": laws.equal_session_mean(_sealed_static_values(surface, plan.M4)),
            "sealed_chronological_m10": laws.equal_session_mean(
                _sealed_static_values(surface, plan.M10)),
            "sealed_chronological_m30": laws.equal_session_mean(
                _sealed_static_values(surface, plan.M30)),
        }
        for surface in plan.SURFACES_STATIC
    }
    cdm_summaries = {
        surface: {
            "CDM_CHRONO4": laws.equal_session_mean(_session_values(cdm_rows[surface])),
            "sealed_dopt_m4": laws.equal_session_mean(_sealed_cdm_values(surface, plan.M4)),
            "sealed_chronological_m10": laws.equal_session_mean(
                _sealed_cdm_values(surface, plan.M10)),
        }
        for surface in plan.SURFACES_CDM
    }
    static_contributions = {
        surface: laws.paired_selection_contribution(
            _sealed_static_values(surface, plan.M4), _session_values(static_rows[surface]))
        for surface in plan.SURFACES_STATIC
    }
    cdm_contributions = {
        surface: laws.paired_selection_contribution(
            _sealed_cdm_values(surface, plan.M4), _session_values(cdm_rows[surface]))
        for surface in plan.SURFACES_CDM
    }
    verdicts = {
        "static_external_official_query_m4": laws.verdict_class(
            static_contributions["external_official_query"]["equal_session_mean_delta"]),
        "static_within_post30_m4": laws.verdict_class(
            static_contributions["within_post30"]["equal_session_mean_delta"]),
        "cdm_external_post30_local_m4": laws.verdict_class(
            cdm_contributions["external_post30_local"]["equal_session_mean_delta"]),
        "cdm_within_post30_m4": laws.verdict_class(
            cdm_contributions["within_post30"]["equal_session_mean_delta"]),
    }
    headline = {
        "verdict": verdicts["cdm_external_post30_local_m4"]["verdict"],
        "surface": "external_post30_local",
        "family": "CDM_CHRONO4 (the online early-start system)",
        "delta": verdicts["cdm_external_post30_local_m4"]["delta"],
        "true_early_start_number": cdm_summaries["external_post30_local"]["CDM_CHRONO4"],
        "true_early_start_expression": (
            "CDM_CHRONO4 external_post30_local equal-session mean R2 "
            "(chronological-first-4 initialization, decoding starts at the "
            "realized earliest decodable trial)"),
        "earliest_decode": {
            "chrono4": sorted({
                item["earliest_decode_trial_1based"]
                for item in cdm_supports.values()
            }),
            "dopt": plan.ACTIVITY_HORIZON + 1,
        },
        "agreement_disclosure": (
            "the static-family external verdict is reported alongside and never "
            "averaged into this headline"
        ),
    }
    earliest_decode_evidence = {
        "chrono4_supports": {
            key: {
                "support_positions": item["support_positions"],
                "directional_among_positions_0_to_3": item[
                    "directional_among_positions_0_to_3"],
                "strict_branch_taken": item["strict_branch_taken"],
                "earliest_decode_trial_1based": item["earliest_decode_trial_1based"],
            }
            for key, item in sorted({**static_supports, **cdm_supports}.items())
        },
        "dopt_pool_depth_trials": plan.ACTIVITY_HORIZON,
    }

    pairing_all_exact = all(item["exact_match"] for item in static_pairing.values()) and \
        all(item["exact_match"] for item in cdm_pairing.values())
    fidelity_all_exact = all(
        item["anchor"]["exact_match"] for item in static_fidelity.values()) and all(
        item["anchor"]["exact_match"] for item in cdm_fidelity.values())
    dopt_proofs_all_exact = all(item["exact_match"] for item in dopt_proofs.values())
    stop_conditions = {
        "anchor_or_binding_failure": {
            "expression": "any pure-data pairing, executor-fidelity or D-opt proof failed",
            "fired": not (pairing_all_exact and fidelity_all_exact and dopt_proofs_all_exact),
            "evidence": {
                "pure_data_pairing_all_exact": bool(pairing_all_exact),
                "executor_fidelity_all_exact": bool(fidelity_all_exact),
                "dopt_proofs_all_exact": bool(dopt_proofs_all_exact),
                "fidelity_max_abs_r2_delta": max(
                    [abs(item["anchor"]["r2_delta"]) for item in static_fidelity.values()]
                    + [abs(item["anchor"]["r2_delta"]) for item in cdm_fidelity.values()]),
            },
        },
    }
    governing = [name for name, item in stop_conditions.items() if item["fired"]]

    replay_payload = {
        "schema": f"{plan.SCHEMA}_replay_v1",
        "stage": "inference_only_true_early_start_chrono4_local_m2_cpu",
        "status": "REPLAY_COMPLETE",
        "attempt_sha256": attempt_digest,
        "authority": (
            "operator work order 2026-09-02: strict-caliber true-early-start cell, "
            "CPU-only, sealed D-opt anchors read from receipts"
        ),
        "foundation": {
            "sealed_comparator_score": plan.COMPARATOR_SCORE_RELATIVE,
            "sealed_comparator_score_sha256": plan.COMPARATOR_SCORE_SHA256,
            "sealed_cdm_screen_score": plan.CDM_SCREEN_SCORE_RELATIVE,
            "sealed_cdm_screen_score_sha256": plan.CDM_SCREEN_SCORE_SHA256,
            "sealed_g_package": plan.SEALED_G_PACKAGE,
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
        "cells": dict(plan.CELLS),
        "chrono4_support_law": dict(plan.CHRONO4_SUPPORT_LAW),
        "frozen_everything_else": dict(plan.FROZEN_EVERYTHING_ELSE),
        "summaries": {
            "static_family": static_summaries,
            "cdm_family": cdm_summaries,
        },
        "selection_contribution_dopt_minus_chrono4": {
            "static_family": static_contributions,
            "cdm_family": cdm_contributions,
        },
        "verdicts": verdicts,
        "headline": headline,
        "true_early_start": {
            "number": cdm_summaries["external_post30_local"]["CDM_CHRONO4"],
            "surface": "external_post30_local",
            "cell": "CDM_CHRONO4",
            "evidence": earliest_decode_evidence,
        },
        "dopt_selection_proof": {
            "law": plan.ANCHORS["dopt_selection_proof"]["law"],
            "all_exact": bool(dopt_proofs_all_exact),
            "proofs": dopt_proofs,
            "scatter_evidence": scatter_evidence,
        },
        "rows": {
            "static_family": {
                surface: dict(sorted(sessions.items()))
                for surface, sessions in static_rows.items()
            },
            "cdm_family": {
                surface: dict(sorted(sessions.items()))
                for surface, sessions in cdm_rows.items()
            },
        },
        "anchors": {
            "law": {
                key: (dict(value) if isinstance(value, dict) else value)
                for key, value in plan.ANCHORS.items()
            },
            "static_pairing": static_pairing,
            "cdm_pairing": cdm_pairing,
            "static_executor_fidelity": static_fidelity,
            "cdm_executor_fidelity": cdm_fidelity,
            "pure_data_pairing_all_exact": bool(pairing_all_exact),
            "executor_fidelity_all_exact": bool(fidelity_all_exact),
            "fidelity_roster": fidelity_roster,
        },
        "support_agreement_across_families": support_agreement,
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
    replay_digest = _publish(root / "replay.json", replay_payload)
    terminal_payload = {
        "schema": f"{plan.SCHEMA}_terminal_v1",
        "status": "TERMINAL",
        "scope": "inference_only_true_early_start_chrono4_local_m2_cpu",
        "replay_sha256": replay_digest,
        "attempt_sha256": attempt_digest,
        "equal_session_means": {
            "static_family": static_summaries,
            "cdm_family": cdm_summaries,
        },
        "selection_contribution_dopt_minus_chrono4": {
            "static_family": {
                surface: {
                    "equal_session_mean_delta": item["equal_session_mean_delta"],
                    "dopt_equal_session_mean": item["dopt_equal_session_mean"],
                    "chrono4_equal_session_mean": item["chrono4_equal_session_mean"],
                    "positive_sessions_dopt_better": item["positive_sessions_dopt_better"],
                    "breadth_denominator": item["breadth_denominator"],
                    "per_session_delta": item["per_session_delta"],
                }
                for surface, item in static_contributions.items()
            },
            "cdm_family": {
                surface: {
                    "equal_session_mean_delta": item["equal_session_mean_delta"],
                    "dopt_equal_session_mean": item["dopt_equal_session_mean"],
                    "chrono4_equal_session_mean": item["chrono4_equal_session_mean"],
                    "positive_sessions_dopt_better": item["positive_sessions_dopt_better"],
                    "breadth_denominator": item["breadth_denominator"],
                    "per_session_delta": item["per_session_delta"],
                }
                for surface, item in cdm_contributions.items()
            },
        },
        "verdicts": verdicts,
        "headline": headline,
        "true_early_start": replay_payload["true_early_start"],
        "dopt_selection_proof": {
            "all_exact": bool(dopt_proofs_all_exact),
            "static_m4_indices": {
                key: item["sealed_indices"] for key, item in dopt_proofs.items()
                if key.startswith("static|")
            },
            "cdm_m4_indices": {
                key: item["sealed_indices"] for key, item in dopt_proofs.items()
                if key.startswith("cdm|")
            },
        },
        "anchors": {
            "pure_data_pairing_all_exact": bool(pairing_all_exact),
            "executor_fidelity_all_exact": bool(fidelity_all_exact),
            "fidelity_max_abs_r2_delta": stop_conditions["anchor_or_binding_failure"][
                "evidence"]["fidelity_max_abs_r2_delta"],
            "dopt_proofs_all_exact": bool(dopt_proofs_all_exact),
        },
        "stop_conditions": replay_payload["stop_conditions"],
        "official_contract_claimed": False,
        "wall_seconds": replay_payload["wall_seconds"],
    }
    _publish(root / "terminal.json", terminal_payload)
    return {
        "status": terminal_payload["status"],
        "replay_sha256": replay_digest,
        "headline_verdict": headline["verdict"],
        "true_early_start_number": headline["true_early_start_number"],
        "static_external_delta": verdicts["static_external_official_query_m4"]["delta"],
        "cdm_external_delta": verdicts["cdm_external_post30_local_m4"]["delta"],
        "stop_conditions_fired": governing,
        "wall_seconds": replay_payload["wall_seconds"],
    }
