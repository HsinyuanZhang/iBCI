"""AC3-0 stage 1: materialize the frozen source-session four-group trajectories.

This is the ONLY GPU touch of AC3-0 and it is inference-only: the accepted
frozen Cell-D evaluator of P2' (imported, never rebuilt) is driven over the six
SOURCE sessions at the deployment budget M4 on the never-commit parent line --
exactly the P2' sub-study line -- and the four complementary-group velocity
trajectories of every completed query trial are persisted with digests.

The stage is gated by the AC3-0 process gate: it runs only when both GPUs are
idle (checked immediately before and after), its wall time must stay inside the
brief-inference budget, and it writes a receipt with the nvidia-smi evidence.
No decoder parameter is ever updated and the external-15 roster is never
opened; the materialization loop parses the within-surface assets only and
asserts that fact on the runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from src.causal_dual_memory_cell_d_score_v1 import physical as v1physical
from src.causal_dual_memory_cell_d_score_v1 import score as v1score
from src.learned_gate_p2prime_v1 import physical as p2physical
from src.learned_gate_p2prime_v1 import plan as p2plan
from src.learned_gate_p2prime_v1 import policy as p2policy
from src.precision_aware_causal_dual_memory_cell_d_score_v1 import score as predecessor_score

from . import plan


class AC3MaterializeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AC3MaterializeError(message)


def nvidia_smi_snapshot() -> dict[str, object]:
    """Raw utilization/compute-process evidence for the process-gate receipt."""
    utilization = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    compute = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    lines = [line.strip() for line in utilization.stdout.splitlines() if line.strip()]
    both_idle = bool(lines) and all(
        int(line.split(",")[1].strip()) == 0 for line in lines if len(line.split(",")) >= 2
    )
    compute_processes = [line for line in compute.stdout.splitlines() if line.strip()]
    return {
        "utilization_stdout": lines,
        "compute_apps_stdout": compute_processes,
        "all_gpus_zero_utilization": both_idle,
        "no_compute_processes": not compute_processes,
        "both_gpus_idle": bool(both_idle and not compute_processes),
        "captured_at_epoch": time.time(),
    }


def _publish(path: Path, payload: Mapping[str, object]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if path.exists():
        raise AC3MaterializeError(f"refusing to overwrite an existing receipt: {path}")
    path.write_text(body + "\n", encoding="utf-8")
    (path.parent / f"{path.name}.sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def _array_payload(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": str(contiguous.dtype),
        "shape": [int(item) for item in contiguous.shape],
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _materialize_session(
    runtime: p2physical.P2PrimeOracleMatrixRuntime,
    session: v1physical.PreparedEvaluationSession,
    *,
    budget: int,
) -> dict[str, object]:
    """The never-commit four-group trajectory line for one source session."""
    activity_rollout = runtime._rollout_activity(session=session, budget=budget)
    runtime._memory_mode = "cdm"
    memory, initial = runtime._initial_memory(session=session, budget=budget)
    query_ids = list(session.query_trial_ids[budget])
    records: list[dict[str, object]] = []
    initial_parent = {
        "activity_sha256": activity_rollout["initial_activity_sha256"],
        "carrier_sha256": activity_rollout["initial_carrier_sha256"],
    }
    for index, trial_id in enumerate(query_ids):
        trial = session.trials_by_id[trial_id]
        parent = activity_rollout["post_commit_state_digests"][index - 1] if index >= 1 else initial_parent
        _require(
            memory.state.activity.digest == parent["activity_sha256"]
            and memory.state.carrier.digest == parent["carrier_sha256"],
            "AC3-0 trajectory line drifted from the activity-only parent trajectory",
        )
        group_predictions, _evidence = runtime._group_predictions_dispatch(trial=trial, memory=memory)
        _require(len(group_predictions) == plan.GROUP_COUNT, "the frozen evaluator must return four views")
        true_rows = runtime._true_velocity_rows(session, trial)
        true_reference = p2policy.true_direction_payload(
            true_rows, trial.velocity_validity, config=memory.state.carrier.config,
        )
        pendings: dict[str, Any] = {}
        for construction in plan.CONSTRUCTIONS_MATERIALIZED:
            views, _meta = p2policy.build_construction_predictions(
                group_predictions=group_predictions, construction=construction,
            )
            pendings[construction] = memory.observe_completed_trial(
                b3s_trial_activity=trial.b3s_activity, carrier_trial_counts=trial.native_counts,
                complementary_predictions=views,
            )
        record: dict[str, object] = {
            "trial_id": trial_id,
            "n_rows": int(group_predictions[0].velocity.shape[0]),
            "valid_mask": np.asarray(group_predictions[0].validity.valid_mask, dtype=bool),
            "velocity": np.stack(
                [np.asarray(item.velocity, dtype=np.float64) for item in group_predictions], axis=2,
            ),  # [P, 2, 4]
            "true_velocity": np.asarray(true_rows, dtype=np.float64),
            "true_direction": dict(true_reference),
        }
        for construction in plan.CONSTRUCTIONS_MATERIALIZED:
            pending = pendings[construction]
            record[f"pseudo_{construction}"] = {
                "accepted": np.asarray([item.accepted for item in pending.pseudo_directions], dtype=bool),
                "theta": np.asarray(
                    [np.nan if item.theta_raw_rad is None else float(item.theta_raw_rad)
                     for item in pending.pseudo_directions], dtype=np.float64),
                "theta_index": np.asarray(
                    [-1 if item.theta_index is None else int(item.theta_index)
                     for item in pending.pseudo_directions], dtype=np.int64),
                "movement_bins": np.asarray(
                    [int(item.movement_bins) for item in pending.pseudo_directions], dtype=np.int64),
                "displacement_norm": np.asarray(
                    [np.nan if item.displacement_norm is None else float(item.displacement_norm)
                     for item in pending.pseudo_directions], dtype=np.float64),
                "mean_speed": np.asarray(
                    [np.nan if item.mean_speed is None else float(item.mean_speed)
                     for item in pending.pseudo_directions], dtype=np.float64),
                "reason": [None if item.reason is None else item.reason.value
                           for item in pending.pseudo_directions],
                "b8_accepted": bool(pending.carrier_transition_accepted),
            }
        records.append(record)
        rejected = pendings[plan.CONSTRUCTIONS_MATERIALIZED[0]]
        if rejected.carrier_transition_accepted:
            rejected = p2policy.oracle_rejected_pending(memory, rejected)
        transition = runtime._commit_dispatch(memory=memory, pending=rejected, trial_id=trial_id)
        _require(
            transition.carrier_before_sha256 == transition.carrier_after_sha256,
            "the AC3-0 trajectory line must never commit a carrier transition",
        )
        runtime._bump_trial_counters()
    return {
        "session": session.session,
        "records": records,
        "activity_rows": {
            "A0_matrix_r2": float(activity_rollout["rows"]["A0"]["matrix_r2"]),
            "A1_matrix_r2": float(activity_rollout["rows"]["A1"]["matrix_r2"]),
            "A1_prediction_sha256": activity_rollout["rows"]["A1"]["prediction_sha256_raw"],
        },
    }


def _flatten(records: Sequence[Mapping[str, object]]) -> dict[str, np.ndarray]:
    """CSR layout: one flat row array plus per-trial offsets."""
    row_starts: list[int] = []
    row_counts: list[int] = []
    velocity_chunks: list[np.ndarray] = []
    true_chunks: list[np.ndarray] = []
    valid_chunks: list[np.ndarray] = []
    offset = 0
    for record in records:
        count = int(record["n_rows"])
        row_starts.append(offset)
        row_counts.append(count)
        velocity_chunks.append(np.asarray(record["velocity"], dtype=np.float64))
        true_chunks.append(np.asarray(record["true_velocity"], dtype=np.float64))
        valid_chunks.append(np.asarray(record["valid_mask"], dtype=bool))
        offset += count
    return {
        "row_starts": np.asarray(row_starts, dtype=np.int64),
        "row_counts": np.asarray(row_counts, dtype=np.int64),
        "velocity_flat": np.concatenate(velocity_chunks, axis=0),
        "true_flat": np.concatenate(true_chunks, axis=0),
        "valid_flat": np.concatenate(valid_chunks, axis=0),
    }


def _pseudo_arrays(
    records: Sequence[Mapping[str, object]], construction: str,
) -> dict[str, np.ndarray]:
    """Per-trial pseudo-direction arrays under the trial-set key convention."""
    payload = {
        key: np.stack([record[f"pseudo_{construction}"][key] for record in records], axis=0)
        for key in ("accepted", "theta", "theta_index", "movement_bins",
                    "displacement_norm", "mean_speed")
    }
    payload["theta_raw_rad"] = payload.pop("theta")
    payload["b8_accepted"] = np.asarray(
        [bool(record[f"pseudo_{construction}"]["b8_accepted"]) for record in records], dtype=bool,
    )
    return payload


def materialize(
    root: Path,
    *,
    gpu_index: int,
    budget: int = plan.BUDGET,
    output_root: Optional[Path] = None,
    idle_required: bool = True,
) -> Mapping[str, object]:
    """Run the source-session materialization and persist the trajectory cache."""
    base = Path(root).absolute()
    output = Path(output_root) if output_root is not None else base / plan.RESULT_ROOT_RELATIVE
    _require((output / "attempt.json").exists(), "the AC3-0 attempt must be reserved before data access")
    started = time.monotonic()
    before = nvidia_smi_snapshot()
    if idle_required:
        _require(before["both_gpus_idle"], "the AC3-0 process gate requires both GPUs idle")
    runtime, meta, identity = _runtime_for(base, gpu_index=gpu_index)
    try:
        runtime.prepare(identity=identity)
        fixed = v1score.derive_fixed_evaluation_authority(base)
        _require(len(fixed.within) == plan.SOURCE_SESSION_COUNT, "source (within) roster drift")
        for asset in fixed.within:
            prepared = runtime._parse_session(asset=asset)
            runtime._require_state().sessions[(asset.surface, asset.session)] = prepared
        state = runtime._require_state()
        _require(
            not runtime._external_opened
            and all(key[0] == plan.SOURCE_SURFACE for key in state.sessions)
            and len(state.sessions) == plan.SOURCE_SESSION_COUNT,
            "AC3-0 must materialize the source (within) sessions only",
        )
        sessions_payload: list[dict[str, object]] = []
        for key in _ordered_session_keys(runtime, plan.SOURCE_SURFACE):
            session = state.sessions[key]
            sessions_payload.append(_materialize_session(runtime, session, budget=budget))
    finally:
        runtime.close()
    after = nvidia_smi_snapshot()
    wall = float(time.monotonic() - started)
    sessions = [str(item["session"]) for item in sessions_payload]
    trial_ids: list[str] = []
    trial_session: list[int] = []
    for index, payload in enumerate(sessions_payload):
        trial_ids.extend(str(record["trial_id"]) for record in payload["records"])
        trial_session.extend([index] * len(payload["records"]))
    flat_parts = [_flatten(payload["records"]) for payload in sessions_payload]
    flat = {
        "row_counts": np.concatenate([part["row_counts"] for part in flat_parts], axis=0),
        "velocity_flat": np.concatenate([part["velocity_flat"] for part in flat_parts], axis=0),
        "true_flat": np.concatenate([part["true_flat"] for part in flat_parts], axis=0),
        "valid_flat": np.concatenate([part["valid_flat"] for part in flat_parts], axis=0),
    }
    # CSR offsets must be global across sessions, not per session.
    global_starts: list[int] = []
    offset = 0
    for part in flat_parts:
        for count in part["row_counts"].tolist():
            global_starts.append(offset)
            offset += int(count)
    flat["row_starts"] = np.asarray(global_starts, dtype=np.int64)
    pseudo = {
        construction: _pseudo_arrays(
            [record for payload in sessions_payload for record in payload["records"]], construction,
        )
        for construction in plan.CONSTRUCTIONS_MATERIALIZED
    }
    true_direction = {
        "accepted": np.asarray(
            [bool(record["true_direction"]["accepted"]) for payload in sessions_payload
             for record in payload["records"]], dtype=bool),
        "theta_raw_rad": np.asarray(
            [np.nan if record["true_direction"]["theta_raw_rad"] is None
             else float(record["true_direction"]["theta_raw_rad"])
             for payload in sessions_payload for record in payload["records"]], dtype=np.float64),
        "theta_index": np.asarray(
            [-1 if record["true_direction"]["theta_index"] is None
             else int(record["true_direction"]["theta_index"])
             for payload in sessions_payload for record in payload["records"]], dtype=np.int64),
    }
    arrays: dict[str, np.ndarray] = {
        "trial_session": np.asarray(trial_session, dtype=np.int32),
        **flat,
        **{f"pseudo_{name}_{key}": value for name, payload in pseudo.items()
           for key, value in payload.items()},
        **{f"true_direction_{key}": value for key, value in true_direction.items()},
    }
    cache_path = output / "trajectories.npz"
    _require(not cache_path.exists(), "the AC3-0 trajectory cache already exists")
    np.savez(cache_path, **arrays)
    manifest = {
        "schema": f"{plan.SCHEMA}_materialize_v1",
        "stage": "source_session_trajectory_materialization",
        "budget": int(budget),
        "sessions": sessions,
        "trial_ids": trial_ids,
        "n_trials": len(trial_ids),
        "surface": plan.SOURCE_SURFACE,
        "external_roster_opened": False,
        "gpu": {
            "index": int(gpu_index),
            "before": before,
            "after": after,
            "wall_seconds": wall,
            "inference_only": True,
            "within_process_gate": bool(wall <= plan.PROCESS_GATE["gpu_inference_seconds"]),
        },
        "sealed_cell_d_swa_sha256": plan.SEALED_CELL_D_SWA_SHA256,
        "identity_sha256": meta["identity_sha256"],
        "activity_rows": {
            str(payload["session"]): payload["activity_rows"] for payload in sessions_payload
        },
        "array_digests": {key: _array_payload(value) for key, value in sorted(arrays.items())},
        "reason_counts": {
            construction: _reason_counts([
                reason for payload in sessions_payload for record in payload["records"]
                for reason in record[f"pseudo_{construction}"]["reason"]
            ])
            for construction in plan.CONSTRUCTIONS_MATERIALIZED
        },
        "target_optimizer_backward_update": 0,
        "model_or_checkpoint_updated": False,
        "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    digest = _publish(output / "materialize.json", manifest)
    return {
        "stage": "materialize",
        "materialize_sha256": digest,
        "cache": str(cache_path),
        "wall_seconds": wall,
        "n_trials": len(trial_ids),
        "sessions": sessions,
        "manifest": manifest,
    }


def _reason_counts(reasons: Sequence[Optional[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reason in reasons:
        if reason is None:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _runtime_for(base: Path, *, gpu_index: int):
    environment = p2physical.validate_environment(gpu_index=gpu_index)
    profile = __import__(
        "src.causal_dual_memory_cell_d_score_v1.plan", fromlist=["COMPATIBLE_DEVICE_PROFILES"],
    ).COMPATIBLE_DEVICE_PROFILES[f"gpu{gpu_index}"]
    predecessor_score.validate_completed_v8_predecessor(base)
    identity = predecessor_score.build_reviewed_identity(base, selected_device_profile=profile)
    runtime = p2physical.P2PrimeOracleMatrixRuntime(root=base, selected_device_profile=profile)
    return runtime, {"environment": environment, "identity_sha256": identity.sha256}, identity


def _ordered_session_keys(runtime: p2physical.P2PrimeOracleMatrixRuntime, surface: str):
    state = runtime._require_state()
    return tuple(key for key, item in state.sessions.items() if key[0] == surface)
