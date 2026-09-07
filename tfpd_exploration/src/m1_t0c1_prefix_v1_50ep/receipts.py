"""Immutable receipt lifecycles for the 50-epoch smoke, arms, probe, and Phase 3.

``run_stage``, ``StageSpec``, and ``assert_fresh`` are reused from the sealed
20-epoch receipts module.  Stage-name lists and attempt payloads are this
lane's own: the predecessor ``arm_stage_names`` is hardcoded to the old
``plan.EPOCHS`` (20) and cannot name 50 epoch rows or the five epoch
checkpoints.
"""
from __future__ import annotations

from typing import Mapping

from tfpd_exploration.src.m1_t0c1_prefix_v1.receipts import (  # noqa: F401
    ReceiptError,
    StageSpec,
    assert_fresh,
    run_stage,
)

from . import plan


class ReceiptError50(ReceiptError):
    """Fail closed for 50-epoch stage receipt drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError50(message)


def _sha_of(value: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, "sha literal drift")
    return value


def _leaf_pairs(bodies: list[str]) -> tuple[str, ...]:
    return tuple(item for body in bodies for item in (body, f"{body}.sha256"))


def smoke_stage_names() -> tuple[str, ...]:
    bodies = ("attempt.json", "launch.json", "t0_smoke.json", "c1_smoke.json",
              "equality.json", "terminal.json")
    return _leaf_pairs(list(bodies))


def arm_stage_names() -> tuple[str, ...]:
    bodies = [
        "attempt.json", "launch.json", "source_authority.json", "stream_head.json",
        "training.json", "checkpoint_manifest.json", "terminal.json",
        plan.BEST_CHECKPOINT_FILENAME,
    ]
    bodies += [f"epoch_{index:02d}.json" for index in range(plan.EPOCHS)]
    bodies += [plan.epoch_checkpoint_filename(index) for index in plan.CHECKPOINT_EPOCH_INDICES]
    return _leaf_pairs(bodies)


def probe_stage_names() -> tuple[str, ...]:
    bodies = ["attempt.json", "launch.json"]
    for arm in plan.ARMS:
        for epoch_index in plan.CHECKPOINT_EPOCH_INDICES:
            for session_id in plan.PROBE_SESSIONS:
                bodies.append(f"score_{arm}_epoch_{epoch_index:02d}_{session_id}.json")
    bodies += ["opened_sessions.json", "curve.json", "verdict.json", "selection.json", "terminal.json"]
    return _leaf_pairs(bodies)


def phase3_stage_names() -> tuple[str, ...]:
    bodies = ["attempt.json", "launch.json"]
    for label in ("epoch50", "selected"):
        for arm in plan.ARMS:
            for deployment in plan.PHASE3_DEPLOYMENTS:
                for session_id in plan.SCORE_ORDER:
                    bodies.append(f"score_{label}_{arm}_{deployment}_{session_id}.json")
    bodies += ["table_epoch50.json", "table_selected.json", "reference_delta.json", "terminal.json"]
    return _leaf_pairs(bodies)


def arm_concurrency_receipt(
    stage: str, concurrent_sibling_stage: str | None,
) -> dict[str, object]:
    """Bind the GPU-1 concurrent-pair disclosure onto a t0/c1 receipt."""
    _require(stage in plan.CONCURRENT_ARM_STAGES,
             "arm_concurrency_receipt is for t0/c1 only")
    if concurrent_sibling_stage is not None:
        other = "c1" if stage == "t0" else "t0"
        _require(concurrent_sibling_stage == other,
                 f"concurrent_sibling_stage for {stage} must be {other!r}, "
                 f"not {concurrent_sibling_stage!r}")
    return {
        "arm_concurrency": plan.ARM_CONCURRENCY,
        "concurrent_sibling_stage": concurrent_sibling_stage,
        "arm_concurrency_timing_disclosure": plan.ARM_CONCURRENCY_TIMING_DISCLOSURE,
    }


def stage_attempt_payload(
    stage: str, pair_spec_sha256: str, closure: Mapping[str, object],
    *, concurrent_sibling_stage: str | None = None,
) -> dict[str, object]:
    """Attempt payload for this 50-epoch lane.

    The sealed 20-epoch plan froze ``SEALED_SMOKE_TERMINAL_SHA256`` /
    ``SEALED_ARM_TERMINAL_SHA256`` as hex literals because that lane's smoke
    (and later its arms) had already been sealed when the plan was written.
    This 50-epoch lane's own smoke and arm terminals do not exist until the
    orchestrator runs, so the same field names are modeled as
    ``verified_at_run_time`` descriptors.  Drivers check this lane's
    filesystem (regular file, mode ``0o444``, sidecar text, status OK)
    rather than comparing to a plan.py hex literal.  The 20-epoch
    predecessor digests ARE frozen literals and are bound on every stage as
    provenance (spec §2.8 / §6).
    """
    _require(stage in {"smoke", "t0", "c1", "probe", "phase3"} and pair_spec_sha256,
             "stage attempt payload drift")
    if stage not in plan.CONCURRENT_ARM_STAGES:
        _require(concurrent_sibling_stage is None,
                 "only t0/c1 attempts may declare a concurrent sibling")
    payload = {
        "schema": "m1_t0c1_prefix_v1_50ep_stage_attempt_v1",
        "cell": plan.CELL,
        "phase": plan.PHASE,
        "stage": stage,
        "status": "ATTEMPT_RESERVED",
        "pair_spec_sha256": _sha_of(pair_spec_sha256),
        "closure_sha256": closure.get("closure_sha256"),
        "dropout_proof": plan.DROPOUT_PROOF,
        "cycle_law": plan.CYCLE_LAW,
        "lr_schedule_law": dict(plan.LR_SCHEDULE_LAW),
        "source_preparation_law": plan.SOURCE_PREPARATION_LAW,
        "derivative_scan_omission": plan.DERIVATIVE_SCAN_OMISSION,
        "launch_envelope": dict(plan.LAUNCH_ENVELOPE),
        "arm_concurrency_law": dict(plan.ARM_CONCURRENCY_LAW),
        "predecessor_20ep_terminal_sha256": dict(plan.PREDECESSOR_20EP_ARM_TERMINAL_SHA256),
        "predecessor_20ep_smoke_terminal_sha256": plan.PREDECESSOR_20EP_SMOKE_TERMINAL_SHA256,
        "predecessor_20ep_phase3_terminal_sha256": plan.PREDECESSOR_20EP_PHASE3_TERMINAL_SHA256,
        "sealed_smoke_terminal_sha256": dict(plan.THIS_LANE_SMOKE_BINDING)
        if stage in {"t0", "c1", "probe", "phase3"} else None,
        "sealed_arm_terminal_sha256": dict(plan.THIS_LANE_ARM_BINDING)
        if stage in {"probe", "phase3"} else None,
        "operator_resolution": plan.OPERATOR_RESOLUTION,
        "device_identity_law": dict(plan.DEVICE_IDENTITY_LAW),
        "spint_resource_parity": dict(plan.SPINT_RESOURCE_PARITY),
        "val_heldout_access_law": dict(plan.VAL_HELDOUT_ACCESS_LAW)
        if stage in {"probe", "phase3"} else None,
        "frozen_val_heldout_body_sha256": dict(plan.VAL_HELDOUT_BODY_SHA256)
        if stage in {"probe", "phase3"} else None,
        "observed_val_heldout_body_sha256": None if stage in {"probe", "phase3"} else None,
        "observed_body_sha256_published_after_open": True if stage in {"probe", "phase3"} else None,
        "phase1_motivation": plan.PHASE1_MOTIVATION,
        "data_or_model_accessed": False,
        "stage_kind": ("source_training" if stage in {"smoke", "t0", "c1"} else "metric_only_scoring"),
        "target_optimizer_backward_update": 0,
        "gradient_updates": 0 if stage in {"probe", "phase3"} else None,
        "optimizer_steps": 0 if stage in {"probe", "phase3"} else None,
        "labels_used_for": "metric only" if stage in {"probe", "phase3"} else "training",
    }
    if stage in plan.CONCURRENT_ARM_STAGES:
        payload.update(arm_concurrency_receipt(stage, concurrent_sibling_stage))
    return payload


__all__ = (
    "ReceiptError", "ReceiptError50", "StageSpec", "assert_fresh", "run_stage",
    "smoke_stage_names", "arm_stage_names", "probe_stage_names", "phase3_stage_names",
    "stage_attempt_payload", "arm_concurrency_receipt",
)
