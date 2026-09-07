"""Physical frozen-weight Stage-0 evaluator for H1-CAC V1."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from m1_h1_activity_headroom_v1.core import (
    array_digest,
    forward_with_cached_identity,
    identity_from_raw_trials,
    variance_weighted_r2,
)
from h1_date_lodo_activity_headroom_v1.evaluate import (
    _DateLodoDatasetAdapter,
    _load_bound_authority,
    _load_model,
)
from h1_date_lodo_activity_headroom_v1.plan import AUTHORITIES

from .core import CandidateSchedule, array_sha256, build_chunk_schedule, pool_cardinality, require
from .plan import (
    ARM_ORDER, BATCH_SIZE, CHUNK_LENGTH_BY_OUTER_DATE, DATE_ORDER, MAX_MEMBERS,
    SCHEMA, STAGE0_SUPPORT, WINDOW, decide,
)


DATA_RELATIVE = "SPINT-main/data/000954"


def _first_raw_bin(record: Any, trial_value: float) -> int:
    indices = np.flatnonzero(np.isfinite(record.trial_num) & (record.trial_num == float(trial_value)))
    require(indices.size > 0, "phase-origin trial is absent")
    return int(indices[0])


def _last_raw_end(record: Any, trial_value: float) -> int:
    indices = np.flatnonzero(np.isfinite(record.trial_num) & (record.trial_num == float(trial_value)))
    require(indices.size > 0, "true-trial completion is absent")
    return int(indices[-1]) + 1


def _true_trial_schedule(record: Any, *, support: int) -> CandidateSchedule:
    from src.data.h1_m4_eb_pilot import interpolate_trial_identity

    values = tuple(record.trial_values[support:MAX_MEMBERS])
    members = tuple(np.ascontiguousarray(interpolate_trial_identity(record, value), dtype=np.float32)
                    for value in values)
    completion = tuple(_last_raw_end(record, value) for value in values)
    energies = tuple(float(np.asarray(record.eval_trial_neural(value), dtype=np.float64).mean(dtype=np.float64))
                     for value in values)
    return CandidateSchedule(
        members=members,
        completion_end_exclusive=completion,
        candidate_end_exclusive=completion,
        candidate_energies=energies,
        accepted=tuple(True for _ in values),
    )


def _session_material(record: Any, *, chunk_length: int, support: int) -> dict[str, Any]:
    from src.data.h1_m4_eb_pilot import interpolate_trial_identity

    require(len(record.trial_values) > support, "H1-CAC query trial is absent")
    initial = tuple(np.ascontiguousarray(interpolate_trial_identity(record, value), dtype=np.float32)
                    for value in record.trial_values[:support])
    origin = _first_raw_bin(record, record.trial_values[support])
    trial = _true_trial_schedule(record, support=support)
    fixed = build_chunk_schedule(
        record.neural, origin=origin, chunk_length=chunk_length,
        initial_members=support, energy_gated=False,
    )
    energy = build_chunk_schedule(
        record.neural, origin=origin, chunk_length=chunk_length,
        initial_members=support, energy_gated=True,
    )
    return {"initial": initial, "origin": origin, "B-TRIAL7": trial, "C-FIX7": fixed, "D-EMED7": energy}


def _evaluate_arm(
    *, model: Any, dataset: Any, material: Mapping[str, Mapping[str, Any]], arm: str, device: str,
    support: int = STAGE0_SUPPORT,
) -> dict[str, Any]:
    import torch

    require(arm in ARM_ORDER, "unknown H1-CAC arm")
    prediction = np.empty((len(dataset), 7), dtype=np.float32)
    target = np.empty((len(dataset), 7), dtype=np.float32)
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    cardinality_trace = np.empty(len(dataset), dtype=np.int16)
    for row, (session, start) in enumerate(dataset.window_indices):
        endpoint = int(start) + WINDOW - 1
        if arm == "A-STATIC":
            count = support
        else:
            count = pool_cardinality(material[session][arm], initial_members=support,
                                     endpoint_inclusive=endpoint)
        cardinality_trace[row] = count
        groups[(session, count)].append(row)
    forwards = 0
    identity_digests: dict[str, str] = {}
    started = time.monotonic()
    with torch.no_grad():
        for (session, count), rows in groups.items():
            row_material = material[session]
            if arm == "A-STATIC":
                pool = row_material["initial"]
            else:
                schedule = row_material[arm]
                pool = row_material["initial"] + schedule.members
            require(support <= count <= len(pool) <= MAX_MEMBERS, "H1-CAC pool state drift")
            stack = np.ascontiguousarray(np.stack(pool[:count]), dtype=np.float32)
            carrier = np.asarray(dataset.support[session].carriers["full"], dtype=np.float32)
            net = getattr(model, "net", model)
            identity = identity_from_raw_trials(
                net, stack, tuple(range(count)), family="h1", device=device, carrier=carrier,
            )
            identity_digests[f"{session}|M{count}"] = array_sha256(identity.detach().cpu().numpy())
            for offset in range(0, len(rows), BATCH_SIZE):
                batch_rows = rows[offset:offset + BATCH_SIZE]
                xs, ys = [], []
                for row in batch_rows:
                    current_session, start = dataset.window_indices[row]
                    require(current_session == session, "grouped H1-CAC session drift")
                    record = dataset.records[session]
                    end = int(start) + WINDOW
                    xs.append(record.neural[int(start):end])
                    ys.append(record.velocity[end - 1])
                output = forward_with_cached_identity(
                    net, np.ascontiguousarray(np.stack(xs), dtype=np.float32), identity,
                )
                scaling = float(getattr(getattr(model, "hparams", None), "behavior_scaling_factor", 20.0))
                pred = np.ascontiguousarray(
                    output[:, -1, :].detach().cpu().numpy() / np.float32(scaling),
                    dtype=np.float32,
                )
                index = np.asarray(batch_rows, dtype=np.int64)
                prediction[index] = pred
                target[index] = np.ascontiguousarray(np.stack(ys), dtype=np.float32)
                forwards += 1
    sessions = tuple(session for session, _ in dataset.window_indices)
    per_recording = {}
    for session in dataset.records:
        mask = np.asarray([value == session for value in sessions], dtype=bool)
        per_recording[session] = {
            "n_windows": int(mask.sum()),
            "r2": variance_weighted_r2(prediction[mask], target[mask]),
        }
    return {
        "arm": arm,
        "causal": True,
        "officially_deployable": arm != "B-TRIAL7",
        "label_free_state_update": True,
        "equal_recording_mean_r2": float(np.mean([row["r2"] for row in per_recording.values()], dtype=np.float64)),
        "pooled_r2": variance_weighted_r2(prediction, target),
        "per_recording": per_recording,
        "n_windows": len(dataset),
        "prediction_sha256": array_digest(prediction),
        "target_sha256": array_digest(target),
        "cardinality_trace_sha256": array_sha256(cardinality_trace),
        "activity_cardinality_min": int(cardinality_trace.min()),
        "activity_cardinality_max": int(cardinality_trace.max()),
        "identity_state_sha256": identity_digests,
        "forward_batches": forwards,
        "elapsed_seconds": time.monotonic() - started,
        "_prediction": prediction,
        "_target": target,
    }


def _evaluate_date(root: Path, outer_date: str, *, device: str) -> dict[str, Any]:
    from src.data.h1_carrierid_date_lodo_target import (
        H1CarrierIdDateLodoStrictTargetDataset,
        load_outer_date_target_records,
        load_target_dependencies,
    )
    from src.h1_m4_cce_contract import state_hash

    authority = AUTHORITIES[outer_date]
    terminal, checkpoint_path, config_path, source_manifest_path = _load_bound_authority(root, authority)
    plan, normalizer, source_manifest = load_target_dependencies(source_manifest_path, outer_date=outer_date)
    require(source_manifest.get("outer_date") == outer_date, "source manifest outer-date drift")
    model, state_before = _load_model(checkpoint_path, config_path, terminal, device=device)
    records = load_outer_date_target_records(root / DATA_RELATIVE, outer_date=outer_date)
    strict = H1CarrierIdDateLodoStrictTargetDataset(records, plan, normalizer, outer_date=outer_date)
    require(strict.manifest() == terminal["target"]["strict_dataset"], "strict target surface drift")
    dataset = _DateLodoDatasetAdapter(strict)
    chunk_length = CHUNK_LENGTH_BY_OUTER_DATE[outer_date]
    material = {session: _session_material(record, chunk_length=chunk_length, support=STAGE0_SUPPORT)
                for session, record in records.items()}
    results = [_evaluate_arm(model=model, dataset=dataset, material=material, arm=arm, device=device)
               for arm in ARM_ORDER]
    targets = {row["target_sha256"] for row in results}
    require(len(targets) == 1, "four-arm target authority drift")
    # A uses the same frozen cached-identity arithmetic as the accepted static evaluator.
    accepted = float(authority.accepted_static_pooled_r2)
    require(abs(float(results[0]["pooled_r2"]) - accepted) <= 2.0e-7,
            "A-STATIC does not reproduce the accepted H-C checkpoint score")
    state_after = state_hash(model.state_dict())
    require(state_before == state_after, "H1-CAC changed frozen model state")
    traces = {}
    for session, row in material.items():
        traces[session] = {
            "phase_origin": row["origin"],
            "B-TRIAL7": row["B-TRIAL7"].receipt(),
            "C-FIX7": row["C-FIX7"].receipt(),
            "D-EMED7": row["D-EMED7"].receipt(),
        }
    for row in results:
        row.pop("_prediction")
        row.pop("_target")
    return {
        "outer_date": outer_date,
        "sessions": list(records),
        "chunk_length": chunk_length,
        "authority": {
            "terminal_sha256": authority.terminal_sha256,
            "checkpoint_sha256": authority.checkpoint_sha256,
            "config_sha256": authority.config_sha256,
            "source_manifest_sha256": authority.source_manifest_sha256,
            "query_window_indices_sha256": authority.query_window_indices_sha256,
            "accepted_static_pooled_r2": accepted,
        },
        "state_traces": traces,
        "results": results,
        "model_state_before_sha256": state_before,
        "model_state_after_sha256": state_after,
        "model_state_immutable": True,
    }


def run(root: Path, *, device: str) -> dict[str, Any]:
    import torch

    root = Path(root).resolve()
    require(device.startswith("cuda") and torch.cuda.is_available(), "H1-CAC Stage 0 requires CUDA")
    date_rows = [_evaluate_date(root, outer_date, device=device) for outer_date in DATE_ORDER]
    decision = decide(date_rows)
    return {
        "schema": f"{SCHEMA}_stage0_score",
        "status": "COMPLETE_H1_CAC_STAGE0_FROZEN_WEIGHT_SCORE",
        "stage": 0,
        "surface": "five_date_lodo_strict_post_m4_common_query",
        "device": device,
        "date_order": list(DATE_ORDER),
        "arm_order": list(ARM_ORDER),
        "date_results": date_rows,
        "decision": decision,
        "verdict": decision["verdict"],
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
        "formal_heldout_opened": False,
        "minival_opened": False,
        "evalai_opened": False,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }


__all__ = ("run",)
