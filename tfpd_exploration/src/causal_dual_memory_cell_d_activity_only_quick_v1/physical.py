"""Physical activity-only composition over the accepted V8 evaluator.

This module changes no model or parser behavior.  It replaces V8's target
carrier proposal with one typed activity-only transition and never executes a
complementary-group forward.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import time
from typing import Any, Mapping, Sequence

from src.causal_dual_memory_cell_d_score_v1 import physical as v1physical
from src.causal_dual_memory_cell_d_score_v1 import plan as v1plan
from src.causal_dual_memory_cell_d_score_v1 import score as v1score
from src.causal_dual_memory_cell_d_score_v5 import plan as v5plan
from src.causal_dual_memory_cell_d_score_v5 import score as v5score
from src.causal_dual_memory_cell_d_score_v8 import physical as v8physical
from src.causal_dual_memory_cell_d_v1 import core
from src.precision_aware_causal_dual_memory_cell_d_score_v1 import score as predecessor_score

from . import plan


class ActivityOnlyQuickError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActivityOnlyQuickError(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")).encode("utf-8") + b"\n"


def _digest(value: object) -> str:
    return hashlib.sha256(plan.canonical_json_bytes(value)).hexdigest()


class ActivityOnlyMemory(core.IndependentActivityCausalDualMemory):
    """Independent activity state with an explicit no-carrier-evidence law."""

    def observe_completed_trial(
        self,
        *,
        b3s_trial_activity: Any,
        carrier_trial_counts: Any,
        complementary_predictions: Sequence[Any],
    ) -> core.IndependentActivityPendingTrialUpdate:
        del carrier_trial_counts, complementary_predictions
        reason = self.state.activity.validate_complete_trial(b3s_trial_activity)
        if reason is not None:
            return self._activity_invalid_pending(
                reason,
                evidence={"activity_only_policy": True, "carrier_proposal_attempted": False},
            )
        candidate, changed = self._activity_candidate(b3s_trial_activity)
        return self._carrier_rejected_pending(
            activity_candidate=candidate,
            activity_fifo_changed=changed,
            reason=core.UpdateRejectionReason.INSUFFICIENT_EVIDENCE,
            evidence={
                "activity_only_policy": True,
                "carrier_proposal_attempted": False,
                "carrier_input_consumed": False,
            },
        )


class ActivityOnlyRuntime(v8physical.V8ReviewedCDMScoreRuntime):
    """V8 parser/model/metric runtime with only an activity transition."""

    @staticmethod
    def _make_dual_memory(*, core: Any, activity: Any, carrier: Any) -> Any:
        memory = ActivityOnlyMemory(activity=activity, carrier=carrier)
        _require(type(memory) is ActivityOnlyMemory, "activity-only memory type drift")
        return memory

    @staticmethod
    def _group_predictions(*, trial: Any, memory: Any) -> tuple[tuple[None, ...], tuple[dict[str, object], ...]]:
        del trial
        _require(type(memory) is ActivityOnlyMemory, "activity-only group bypass memory drift")
        evidence = tuple(
            {
                "group_index": index,
                "forward_chunk_count": 0,
                "activity_only_bypass": True,
                "carrier_proposal_attempted": False,
            }
            for index in range(core.GROUP_COUNT)
        )
        return (None,) * core.GROUP_COUNT, evidence

    def _commit_completed_transition(
        self, *, memory: Any, pending: Any, trial_id: str,
    ) -> v1physical.CompletedTrialTransition:
        _require(type(memory) is ActivityOnlyMemory, "activity-only commit memory drift")
        _require(
            isinstance(pending, core.IndependentActivityPendingTrialUpdate)
            and pending.carrier_transition_accepted is False
            and pending.carrier_proposal is None,
            "activity-only route observed a carrier proposal",
        )
        outcome = memory.commit_independent(pending)
        core.validate_independent_activity_outcome_payload(outcome.payload())
        _require(
            outcome.carrier_transition_committed is False
            and outcome.carrier_before_sha256 == outcome.carrier_after_sha256,
            "activity-only route changed carrier state",
        )
        return v1physical.CompletedTrialTransition(
            trial_id=trial_id,
            activity_transition_committed=outcome.activity_transition_committed,
            activity_fifo_changed=outcome.activity_fifo_changed,
            carrier_transition_committed=False,
            activity_rejection_reason=self._reason_value(outcome.activity_rejection_reason),
            carrier_rejection_reason=self._reason_value(outcome.carrier_rejection_reason),
            state_before_sha256=outcome.state_before_sha256,
            state_after_sha256=outcome.state_after_sha256,
            activity_before_sha256=outcome.activity_before_sha256,
            activity_after_sha256=outcome.activity_after_sha256,
            carrier_before_sha256=outcome.carrier_before_sha256,
            carrier_after_sha256=outcome.carrier_after_sha256,
        )

    def score_activity_budget(
        self, *, budget: int, input_authority_sha256: str,
    ) -> tuple[dict[str, object], ...]:
        _require(budget in plan.BUDGETS, "activity-only budget drift")
        state = self._require_state()
        cells: list[dict[str, object]] = []
        for surface in plan.SURFACES:
            sessions = tuple(item.session for item in state.sessions.values() if item.surface == surface)
            _require(len(sessions) == v1plan.expected_session_count(surface), "activity-only surface roster drift")
            raw_rows = tuple(
                self._score_session(
                    session=state.sessions[(surface, session)],
                    budget=budget,
                    system=v5plan.SYSTEM_CDMD,
                )
                for session in sessions
            )
            rows: list[dict[str, object]] = []
            for item, session_name in zip(raw_rows, sessions, strict=True):
                _require(isinstance(item, v5score.IndependentSessionScore), "activity-only session codec drift")
                prepared = state.sessions[(surface, session_name)]
                payload = item.payload(
                    budget=budget,
                    system=v5plan.SYSTEM_CDMD,
                    query_trial_ids=prepared.query_trial_ids[budget],
                )
                transitions = payload["transition_records"]
                _require(
                    payload["carrier_transition_committed_count"] == 0
                    and payload["group_forward_count"] == 0
                    and all(row["carrier_transition_committed"] is False for row in transitions)
                    and all(row["carrier_before_sha256"] == row["carrier_after_sha256"] for row in transitions)
                    and all(row["activity_transition_committed"] is True for row in transitions)
                    and all(row["carrier_rejection_reason_or_null"] == "insufficient_evidence" for row in transitions),
                    "activity-only transition law drift",
                )
                payload = dict(payload)
                payload.update({
                    "schema": "causal_dual_memory_cell_d_activity_only_session_v1",
                    "system": plan.SYSTEM,
                    "carrier_policy": "no_carrier_evidence_no_proposal_frozen_support_initializer",
                    "carrier_proposal_attempt_count": 0,
                })
                rows.append(payload)
            cells.append({
                "schema": "causal_dual_memory_cell_d_activity_only_cell_v1",
                "budget": budget,
                "surface": surface,
                "system": plan.SYSTEM,
                "input_authority_sha256": input_authority_sha256,
                "sessions": rows,
                "resources": dict(self._resources()),
                "eval_mode": True,
                "no_grad": True,
                "dropout_disabled": True,
                "target_optimizer_backward_update": 0,
                "group_forward_count": 0,
            })
        return tuple(cells)


def _mean(values: Sequence[float]) -> float:
    _require(bool(values) and all(math.isfinite(value) for value in values), "finite nonempty mean required")
    return float(sum(values) / len(values))


def _paired(reference: Sequence[Mapping[str, object]], candidate: Sequence[Mapping[str, object]]) -> dict[str, object]:
    _require(
        tuple(row["session"] for row in reference) == tuple(row["session"] for row in candidate),
        "activity-only paired roster drift",
    )
    deltas = [float(new["governing_r2"]) - float(old["governing_r2"]) for old, new in zip(reference, candidate, strict=True)]
    rng = random.Random(42)
    draws = []
    for _ in range(10_000):
        draws.append(_mean([deltas[rng.randrange(len(deltas))] for _ in deltas]))
    draws.sort()
    return {
        "mean_delta": _mean(deltas),
        "n_positive": sum(value > 0.0 for value in deltas),
        "n_sessions": len(deltas),
        "deltas": deltas,
        "bootstrap_ci95": [draws[249], draws[9749]],
        "bootstrap_draws": 10_000,
        "bootstrap_seed": 42,
    }


def _cell(binding: Any, *, budget: int, surface: str, system: str) -> Mapping[str, object]:
    summary = binding.score_payload["budget_summaries"][str(budget)]
    matches = [
        row for row in summary["cells"]
        if row["budget"] == budget and row["surface"] == surface and row["system"] == system
    ]
    _require(len(matches) == 1, "V8 comparator cell topology drift")
    return matches[0]


def summarize(
    *, cells: Sequence[Mapping[str, object]], binding: Any, input_sha256: str,
) -> dict[str, object]:
    expected = [(budget, surface) for budget in plan.BUDGETS for surface in plan.SURFACES]
    _require([(row["budget"], row["surface"]) for row in cells] == expected, "activity-only cell order drift")
    summaries: dict[str, object] = {}
    outlier_status: dict[str, object] = {}
    all_external_worst_safe = True
    within_safe = True
    for budget in plan.BUDGETS:
        surfaces: dict[str, object] = {}
        for surface in plan.SURFACES:
            activity = next(row for row in cells if row["budget"] == budget and row["surface"] == surface)
            sealed = _cell(binding, budget=budget, surface=surface, system=v5plan.SYSTEM_SEALED)
            full = _cell(binding, budget=budget, surface=surface, system=v5plan.SYSTEM_CDMD)
            activity_rows = activity["sessions"]
            sealed_rows = sealed["sessions"]
            full_rows = full["sessions"]
            for fresh, old in zip(activity_rows, sealed_rows, strict=True):
                _require(
                    all(fresh[key] == old[key] for key in (
                        "session", "input_record_sha256", "support_trial_ids_sha256",
                        "initial_carrier_sha256", "group_assignment_sha256", "group_valid_mask_sha256",
                        "initial_activity_sha256", "target_last_bin_sha256", "valid_mask_sha256",
                        "valid_last_bin_count", "sealed_model_load_proof_sha256",
                    )),
                    "activity-only/V8 same-input initial-state drift",
                )
            paired_sealed = _paired(sealed_rows, activity_rows)
            paired_full = _paired(full_rows, activity_rows)
            surfaces[surface] = {
                "sealed_mean_r2": _mean([float(row["governing_r2"]) for row in sealed_rows]),
                "v8_full_cdm_mean_r2": _mean([float(row["governing_r2"]) for row in full_rows]),
                "activity_only_mean_r2": _mean([float(row["governing_r2"]) for row in activity_rows]),
                "activity_only_minus_sealed": paired_sealed,
                "activity_only_minus_v8_full_cdm": paired_full,
            }
            if surface == "within":
                within_safe = within_safe and paired_sealed["mean_delta"] >= -0.01
            else:
                all_external_worst_safe = all_external_worst_safe and min(paired_sealed["deltas"]) >= -0.20
                names = [row["session"] for row in activity_rows]
                outlier = "sub-M_ses-CO-20140626"
                _require(outlier in names, "predeclared external outlier absent")
                index = names.index(outlier)
                outlier_status[str(budget)] = {
                    "activity_only_minus_sealed": paired_sealed["deltas"][index],
                    "activity_only_minus_v8_full_cdm": paired_full["deltas"][index],
                }
        summaries[str(budget)] = {"budget": budget, "surfaces": surfaces}
    m10 = summaries["10"]["surfaces"]["external"]["activity_only_minus_sealed"]
    m4 = summaries["4"]["surfaces"]["external"]["activity_only_minus_sealed"]
    outlier_pass = all(
        row["activity_only_minus_v8_full_cdm"] >= 0.30 or row["activity_only_minus_sealed"] >= -0.20
        for row in outlier_status.values()
    )
    gates = {
        "m10_external": m10["mean_delta"] >= 0.02 and m10["n_positive"] >= 10,
        "m4_external": m4["mean_delta"] >= 0.05 and m4["n_positive"] >= 10,
        "within_safety": within_safe,
        "external_worst_session_safety": all_external_worst_safe,
        "predeclared_outlier_safety": outlier_pass,
    }
    return {
        "schema": "causal_dual_memory_cell_d_activity_only_summary_v1",
        "input_authority_sha256": input_sha256,
        "budget_summaries": summaries,
        "outlier": outlier_status,
        "gates": gates,
        "verdict": "ADVANCE_ACTIVITY_ONLY" if all(gates.values()) else "STOP_ACTIVITY_ONLY",
    }


def _publish(path: Path, payload: Mapping[str, object]) -> str:
    body = _json_bytes(payload)
    digest = hashlib.sha256(body).hexdigest()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        os.write(descriptor, body)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    sidecar = path.with_name(path.name + ".sha256")
    descriptor = os.open(sidecar, flags, 0o444)
    try:
        os.write(descriptor, f"{digest}  {path.name}\n".encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest


def execute(root: Path, *, gpu_index: int = 1) -> Mapping[str, object]:
    base = Path(root).absolute()
    output = base / plan.RESULT_ROOT_RELATIVE
    _require(gpu_index in (0, 1), "activity-only GPU index must be 0 or 1")
    _require(not output.exists(), "activity-only quick result root already exists")
    profile = v1plan.COMPATIBLE_DEVICE_PROFILES[f"gpu{gpu_index}"]
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu_index), "activity-only CVD drift")
    _require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "activity-only PCI order drift")
    binding = predecessor_score.validate_completed_v8_predecessor(base)
    identity = predecessor_score.build_reviewed_identity(base, selected_device_profile=profile)
    owned = plan.owned_sha256s(base)
    output.mkdir(mode=0o755, parents=False, exist_ok=False)
    attempt = {
        "schema": "causal_dual_memory_cell_d_activity_only_attempt_v1",
        "status": "ATTEMPT_RESERVED",
        "cell": plan.CELL,
        "identity_sha256": identity.sha256,
        "precision_successor_identity_used_only_as_current_v8_runtime_closure": True,
        "v8_predecessor": binding.payload(),
        "owned_sha256s": owned,
        "selected_device_profile": dict(profile),
        "budgets": list(plan.BUDGETS),
        "surfaces": list(plan.SURFACES),
        "target_optimizer_backward_update": 0,
    }
    attempt_sha = _publish(output / "attempt.json", attempt)
    runtime: ActivityOnlyRuntime | None = None
    input_sha: str | None = None
    stage = "prepare"
    started = time.monotonic()
    try:
        runtime = ActivityOnlyRuntime(root=base, selected_device_profile=profile)
        runtime.prepare(identity=identity)
        stage = "materialize_inputs"
        fixed = v1score.derive_fixed_evaluation_authority(base)
        authority = runtime.materialize_inputs(identity=identity, authority=fixed)
        input_payload = authority.payload(identity=identity)
        predecessor_score.validate_v8_input_equivalence(input_payload, identity)
        input_sha = _digest(input_payload)
        stage = "score"
        cells: list[Mapping[str, object]] = []
        for budget in plan.BUDGETS:
            cells.extend(runtime.score_activity_budget(budget=budget, input_authority_sha256=input_sha))
        summary = summarize(cells=cells, binding=binding, input_sha256=input_sha)
        result = {
            "schema": "causal_dual_memory_cell_d_activity_only_quick_result_v1",
            "status": "TERMINAL",
            "scope": "non_governing_matched_engineering_screen",
            "attempt_sha256": attempt_sha,
            "input_authority_sha256": input_sha,
            "v8_input_authority_sha256": plan.V8_SHAS["input_authority.json"],
            "v8_score_sha256": plan.V8_SHAS["score.json"],
            "cells": cells,
            "summary": summary,
            "wall_seconds": float(time.monotonic() - started),
            "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "model_or_checkpoint_updated": False,
            "target_optimizer_backward_update": 0,
            "carrier_proposal_attempt_count": 0,
        }
        result_sha = _publish(output / "result.json", result)
        terminal = {
            "schema": "causal_dual_memory_cell_d_activity_only_quick_terminal_v1",
            "status": "TERMINAL",
            "attempt_sha256": attempt_sha,
            "input_authority_sha256": input_sha,
            "result_sha256": result_sha,
            "verdict": summary["verdict"],
            "target_optimizer_backward_update": 0,
        }
        terminal_sha = _publish(output / "terminal.json", terminal)
        os.chmod(output, 0o555)
        return {"attempt_sha256": attempt_sha, "result_sha256": result_sha, "terminal_sha256": terminal_sha, **summary}
    except BaseException as error:
        failure = {
            "schema": "causal_dual_memory_cell_d_activity_only_quick_failure_v1",
            "status": "FAILED",
            "attempt_sha256": attempt_sha,
            "input_authority_sha256_or_null": input_sha,
            "stage": stage,
            "error_class": type(error).__name__,
            "error_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
            "terminal": False,
        }
        _publish(output / "failure.json", failure)
        os.chmod(output, 0o555)
        raise
    finally:
        if runtime is not None:
            runtime.close()

