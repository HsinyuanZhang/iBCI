"""Stage-P replay: the P0--P5 cells over the frozen activity-only runtime.

Design §5 (measurement), §6 (gate), §7.2 (matrix), §9 (receipts).  The frozen
``src/learned_gate_p2prime_v1`` machinery is reused exactly as the sealed
drivers and the Stage-O replay used it, and never edited:

* ``p2physical._runtime_for`` / ``prepare`` / ``materialize_inputs`` -- the
  sealed parser, chronology, normalizer and Cell-D SWA checkpoint;
* ``runtime._rollout_activity`` -- P0 is the Stage-O O0 arm VERBATIM (the
  sealed activity-only loop), which makes the P0 bit-anchor a no-op proof of
  this route's loop;
* ``runtime._governing_forward(trial, inputs)`` -- the sealed forward
  primitive reading the current ``T4_active``;
* ``runtime._group_predictions_dispatch(trial, memory)`` -- the frozen
  four-group held-unit forward of the CURRENT pre-reveal state; prediction
  ``h`` excludes group ``h``'s units, which is exactly the complementary-group
  exclusion the deployable measurement needs.

The Stage-O support anchor, trust region and c_M calibration law are imported
from the proven Stage-O package.  Per trial of a P row: (1) forward with the
current state, (2) finalize, (3) run the frozen four-group forwards and measure
the deployable pseudo directions, (4) evaluate the three-factor gate,
(5) commit the block's labels into the bank or reject, (6) move through the
Stage-O trust-region law, and (7) set the next-trial ``T4_active``.  An update
committed from trial i can never affect trial i's own predictions: every
forward reads the state digest recorded before the measurement, and the
receipt chain asserts ``state_after[i-1] == forward_state[i]``.
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

try:
    from src.causal_dual_memory_cell_d_v1 import core as cdm_core
    from src.support_anchored_t4_stage_o_v1 import replay as stage_o_replay
    from src.support_anchored_t4_stage_o_v1 import trust_region
    from src.support_anchored_t4_stage_o_v1.anchor import SupportAnchor
except ModuleNotFoundError as error:
    if not (error.name == "src" or str(error.name).startswith("src.")): raise
    from tfpd_exploration.src.causal_dual_memory_cell_d_v1 import core as cdm_core
    from tfpd_exploration.src.support_anchored_t4_stage_o_v1 import replay as stage_o_replay
    from tfpd_exploration.src.support_anchored_t4_stage_o_v1 import trust_region
    from tfpd_exploration.src.support_anchored_t4_stage_o_v1.anchor import SupportAnchor

from . import direction_estimator, gate, plan


class StagePReplayError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StagePReplayError(message)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_receipt(path: Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    sidecar = Path(path).with_name(Path(path).name + ".sha256")
    _require(sidecar.exists(), f"missing sidecar: {sidecar}")
    _require(
        sidecar.read_text(encoding="ascii").strip() == f"{digest}  {Path(path).name}",
        f"sidecar drift for {Path(path).name}",
    )
    return digest


def _p2_filters():
    from src.learned_gate_p2prime_v1 import filters as p2filters

    return p2filters


# ---------------------------------------------------------------------------
# Row specs and cell hyperparameters.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowSpec:
    """One P row's measurement law (frozen in plan.ROWS)."""

    row_id: str
    law: str
    weight_law: str
    binding: str

    def payload(self) -> dict[str, Any]:
        return {
            "row": self.row_id,
            "direction_law": self.law,
            "weight_law": self.weight_law,
            "binding": self.binding,
        }


ROW_SPECS: dict[str, RowSpec] = {
    "P1": RowSpec("P1", "cross_group_circular", "resultant_length", "complementary_exclusion"),
    "P2": RowSpec("P2", "cross_group_trajectory", "resultant_times_early_late_agreement",
                  "complementary_exclusion"),
    "P3": RowSpec("P3", "cross_group_trajectory", "constant_one", "complementary_exclusion"),
    "P4": RowSpec("P4", "cross_group_trajectory", "resultant_times_early_late_agreement",
                  "rotated_group_and_confidence"),
    "P5": RowSpec("P5", "same_group", "constant_one", "complementary_exclusion"),
}
MEASURED_ROWS = ("P1", "P2", "P3", "P4", "P5")


@dataclass(frozen=True)
class CellHyperparameters:
    """One cell's movement law and source-selected gate thresholds."""

    rho_M: float
    alpha_M: float
    c_M: Optional[float]
    thresholds: gate.GateThresholds

    def payload(self) -> dict[str, Any]:
        return {
            "rho_M": float(self.rho_M),
            "alpha_M": float(self.alpha_M),
            "c_M": None if self.c_M is None else float(self.c_M),
            "thresholds": self.thresholds.payload(),
        }


def enumerate_gate_grid() -> list[gate.GateThresholds]:
    """The pre-registered stage-1 grid, in the pre-registered enumeration order."""
    return [
        gate.GateThresholds(tau_d=tau_d, r_max=r_max, d_min=d_min, max_mass_relative=max_mass)
        for tau_d in plan.TAU_D_CANDIDATES
        for r_max in plan.R_MAX_CANDIDATES
        for d_min in plan.D_MIN_CANDIDATES
        for max_mass in plan.MAX_PSEUDO_MASS_RELATIVE_CANDIDATES
    ]


def enumerate_mass_grid() -> list[tuple[float, float]]:
    """The pre-registered stage-2 ``(rho_M, alpha_M)`` grid, in order."""
    return [(rho, alpha) for rho in plan.RHO_M_CANDIDATES for alpha in plan.ALPHA_M_CANDIDATES]


def select_vector(scored: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Pick the FIRST maximum (ties keep the earlier, less-moving vector)."""
    _require(bool(scored), "selection needs at least one scored vector")
    best: Optional[Mapping[str, Any]] = None
    best_mean = float("-inf")
    for entry in scored:
        mean = float(entry["mean_r2"])
        if mean > best_mean:
            best, best_mean = entry, mean
    _require(best is not None, "selection failed to choose a vector")
    return best


def _anchored_carrier(
    *, groups: Any, initial_raw_t4: np.ndarray, statistics: Any, config: Any, active_t4: np.ndarray,
) -> Any:
    return cdm_core.CarrierMemory(
        groups=groups,
        initial_raw_t4=initial_raw_t4,
        statistics=statistics,
        ordinary_t4=active_t4,
        fixed_ridge_t4=active_t4,
        config=config,
    )


def _frozen_pseudo_directions(measurement: direction_estimator.TrialMeasurement) -> tuple[Any, ...]:
    """The frozen per-group PseudoDirection objects of the four held views."""
    return tuple(
        cdm_core.PseudoDirection(
            accepted=bool(item["accepted"]),
            reason=None if item["reason"] is None else cdm_core.UpdateRejectionReason(item["reason"]),
            theta_raw_rad=item["theta_raw_rad"],
            theta_index=item["theta_index"],
            theta_canonical_rad=item["theta_canonical_rad"],
            canonical_distance_rad=item["canonical_distance_rad"],
            movement_bins=int(item["movement_bins"]),
            displacement_norm=item["displacement_norm"],
            mean_speed=item["mean_speed"],
        )
        for item in measurement.group_evidence
    )


# ---------------------------------------------------------------------------
# The P rollout (the route-owned anchored loop under the three-factor gate).
# ---------------------------------------------------------------------------


def rollout_p(
    runtime: Any,
    *,
    session: Any,
    budget: int,
    spec: RowSpec,
    hp: CellHyperparameters,
    slim: bool = False,
) -> dict[str, Any]:
    """Run one P row over one session under the three-factor commit gate."""
    p2filters = _p2_filters()
    runtime._memory_mode = "cdm"
    memory, initial = runtime._initial_memory(session=session, budget=budget)
    carrier0 = memory.state.carrier
    config = carrier0.config
    rates, directions, support_ids = stage_o_replay._support_binding(session, budget)
    anchor = SupportAnchor.from_labeled_support(
        groups=carrier0.groups,
        support_trial_rates=rates,
        support_direction_indices=directions,
        support_t4=carrier0.active_t4,
    )
    initial_raw = np.asarray(carrier0.initial_raw_t4)
    thresholds = hp.thresholds
    query_ids = list(session.query_trial_ids[budget])
    per_trial_raw: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    bank = gate.EvidenceBankP.empty()
    bank_digest_chain: list[str] = [bank.digest]
    active = np.asarray(carrier0.active_t4)
    expected_state = memory.state.digest
    movements: list[float] = []
    d2_values: list[float] = []
    proposed_d2_values: list[float] = []
    projected_count = 0
    group_forward_chunks = 0
    measurement_reasons: dict[str, int] = {}
    gate_reasons: dict[str, int] = {}
    committed_label_counts = [0] * cdm_core.GROUP_COUNT
    direction_histograms = [np.zeros(len(cdm_core.CANONICAL_DIRECTIONS_RAD), dtype=np.int64)
                            for _ in range(cdm_core.GROUP_COUNT)]
    weight_sums = [0.0] * cdm_core.GROUP_COUNT
    for trial_id in query_ids:
        trial = session.trials_by_id[trial_id]
        inputs = memory.read_prediction_inputs()
        state_before = memory.state.digest
        _require(inputs.state_digest == state_before, f"{spec.row_id} forward state digest drift")
        _require(state_before == expected_state,
                 f"{spec.row_id} causality chain drift: state before != previous commit state")
        _require(np.array_equal(np.asarray(inputs.active_t4), active),
                 f"{spec.row_id} active T4 drift before forward")
        per_trial_raw.append(runtime._governing_forward(trial=trial, inputs=inputs))
        # (2)/(3): finalize, then measure with the CURRENT pre-reveal state.
        views, group_evidence = runtime._group_predictions_dispatch(trial=trial, memory=memory)
        group_forward_chunks += sum(2 * int(item["forward_chunk_count"]) for item in group_evidence)
        measurement = direction_estimator.measure_trial(
            views, config=config, law=spec.law, weight_law=spec.weight_law,
            binding=spec.binding, tau_d=thresholds.tau_d,
        )
        scalar_rates = cdm_core.scalar_rates_from_native_rewarded_counts(trial.native_counts)
        facts = gate.TrialFacts(
            session_id=session.session,
            trial_id=trial_id,
            chronology_position=int(trial.chronology_position),
            scalar_rates=scalar_rates,
            rate_sha256=cdm_core.array_digest(scalar_rates),
            native_counts_sha256=cdm_core.array_digest(trial.native_counts.counts),
            channel_order_sha256=cdm_core.channel_order_digest(session.channel_ids),
            valid_mask_sha256=cdm_core.array_digest(carrier0.groups.valid_mask),
        )
        activity_candidate, fifo_changed = memory._activity_candidate(trial.b3s_activity)
        decision: Optional[gate.BlockDecision] = None
        next_carrier = None
        moved = False
        movement_this: Optional[float] = None
        pseudo = _frozen_pseudo_directions(measurement)
        if not measurement.accepted:
            route_reason = "measurement_rejected:" + str(measurement.reason)
            measurement_reasons[route_reason] = measurement_reasons.get(route_reason, 0) + 1
        else:
            decision = gate.evaluate_block(
                anchor, bank, measurement.per_group,
                facts=facts, measurement_sha256=measurement.digest,
                rho_M=float(hp.rho_M), c_M=hp.c_M, thresholds=thresholds,
                block_index=len(bank),
            )
            if decision.committed:
                bank = bank.with_row(decision.row)
                outcome = trust_region.active_carrier(
                    anchor, decision.coefficients, alpha_M=float(hp.alpha_M), c_M=hp.c_M,
                )
                carrier = _anchored_carrier(
                    groups=carrier0.groups, initial_raw_t4=initial_raw,
                    statistics=anchor.statistics, config=config,
                    active_t4=outcome.active_t4,
                )
                if carrier.digest == memory.state.carrier.digest:
                    # The block committed but alpha_M = 0 (or a measure-zero zero
                    # delta) reproduces the sealed support carrier bit-exactly:
                    # the frozen V3 no-evidence fallback records the transition.
                    route_reason = "block_committed_zero_movement"
                else:
                    next_carrier = carrier
                    moved = True
                    route_reason = "block_committed"
                movements.append(outcome.movement_frobenius)
                d2_values.append(outcome.d2_unprojected)
                movement_this = float(outcome.movement_frobenius)
                projected_count += 1 if outcome.projected else 0
                for group, label in enumerate(decision.labels):
                    if label is not None:
                        committed_label_counts[group] += 1
                        direction_histograms[group][int(label.theta_index)] += 1
                        weight_sums[group] += float(label.weight)
            else:
                route_reason = str(decision.block_reason)
                gate_reasons[route_reason] = gate_reasons.get(route_reason, 0) + 1
            if decision.d2_unprojected is not None:
                proposed_d2_values.append(float(decision.d2_unprojected))
        completed_evidence: dict[str, Any] = {
            "schema": "support_anchored_t4_stage_p_completed_trial_evidence_v1",
            "row": spec.row_id,
            "measurement_sha256": measurement.digest,
            "measurement_law": {
                "direction_law": spec.law, "weight_law": spec.weight_law, "binding": spec.binding,
            },
            "commit_law": "three_factor_gate_block_one_completed_trial",
            "block_reason": route_reason,
            "bank_sha256": bank.digest,
            "anchor_a0_b0_sha256": anchor.a0_b0_digest,
            "rho_M": float(hp.rho_M),
            "alpha_M": float(hp.alpha_M),
            "c_M": None if hp.c_M is None else float(hp.c_M),
            "gate_thresholds": thresholds.payload(),
            "gate_factors": (
                None if decision is None else dict(decision.factor_evidence.get("factors", {}))
            ),
            "d2_unprojected": None if decision is None else decision.d2_unprojected,
        }
        coverage_rows = (
            () if decision is None else tuple(decision.factor_evidence.get("coverage_rows", ()))
        )
        if next_carrier is None:
            candidate_state = cdm_core.DualMemoryState(
                activity=activity_candidate.activity,
                carrier=memory.state.carrier,
                committed_query_trials=activity_candidate.committed_query_trials,
            )
            pending = cdm_core.IndependentActivityPendingTrialUpdate(
                base_state_digest=state_before,
                activity_transition_ready=True,
                activity_fifo_changed=fifo_changed,
                carrier_transition_accepted=False,
                activity_rejection_reason=None,
                carrier_rejection_reason=cdm_core.UpdateRejectionReason.INSUFFICIENT_EVIDENCE,
                pseudo_directions=pseudo,
                scalar_rates=scalar_rates,
                carrier_proposal=None,
                candidate_state=candidate_state,
                fallback=memory.read_prediction_inputs(),
                completed_trial_evidence=completed_evidence,
            )
        else:
            candidate_state = cdm_core.DualMemoryState(
                activity=activity_candidate.activity,
                carrier=next_carrier,
                committed_query_trials=activity_candidate.committed_query_trials,
            )
            proposal = cdm_core.CarrierProposal(
                True, None, next_carrier, coverage_rows,
                cdm_core._departure_ratios(initial_raw, next_carrier.active_t4, carrier0.groups),
            )
            pending = cdm_core.IndependentActivityPendingTrialUpdate(
                base_state_digest=state_before,
                activity_transition_ready=True,
                activity_fifo_changed=fifo_changed,
                carrier_transition_accepted=True,
                activity_rejection_reason=None,
                carrier_rejection_reason=None,
                pseudo_directions=pseudo,
                scalar_rates=scalar_rates,
                carrier_proposal=proposal,
                candidate_state=candidate_state,
                fallback=memory.read_prediction_inputs(),
                completed_trial_evidence=completed_evidence,
            )
        transition = runtime._commit_dispatch(memory=memory, pending=pending, trial_id=trial_id)
        if moved:
            active = np.asarray(next_carrier.active_t4)
        receipt: dict[str, Any] = {
            "trial_id": trial_id,
            "acc": bool(transition.carrier_transition_committed),
            "reason": route_reason,
            "bd": bank.digest,
            "bn": len(bank),
            "bmg": [bank.accepted_mass(group) for group in range(cdm_core.GROUP_COUNT)],
            "fs": inputs.state_digest,
            "sb": transition.state_before_sha256,
            "sa": transition.state_after_sha256,
            "ab": transition.activity_before_sha256,
            "aa": transition.activity_after_sha256,
            "cb": transition.carrier_before_sha256,
            "ca": transition.carrier_after_sha256,
        }
        if decision is not None and decision.d2_unprojected is not None:
            receipt["d2"] = float(decision.d2_unprojected)
            receipt["tr"] = bool(
                decision.factor_evidence.get("factors", {}).get("support_trust_region")
            )
        if movement_this is not None:
            receipt["mv"] = float(movement_this)
        if not slim:
            receipt["m"] = {
                "law": measurement.law, "weight_law": measurement.weight_law,
                "binding": measurement.binding, "acc": bool(measurement.accepted),
                "reason": measurement.reason,
                "g": [None if item is None else item.payload() for item in measurement.per_group],
                "frozen": [dict(item) for item in measurement.group_evidence],
            }
            receipt["gate"] = {
                "committed": bool(decision.committed) if decision is not None else False,
                "block_reason": None if decision is None else decision.block_reason,
                "per_group_reasons": [] if decision is None else list(decision.per_group_reasons),
                "d2": None if decision is None else decision.d2_unprojected,
                "coverage_rows": list(coverage_rows),
            }
        receipts.append(receipt)
        bank_digest_chain.append(bank.digest)
        expected_state = transition.state_after_sha256
        runtime._bump_trial_counters()
    per_trial_filtered = [p2filters.apply_output_filter_one_trial(item) for item in per_trial_raw]
    return {
        "row": spec.row_id,
        "per_trial_raw": per_trial_raw,
        "per_trial_filtered": per_trial_filtered,
        "receipts": receipts,
        "initial": initial,
        "anchor": anchor,
        "final_carrier_digest": memory.state.carrier.digest,
        "movements": movements,
        "d2_values": d2_values,
        "proposed_d2_values": proposed_d2_values,
        "projected_count": projected_count,
        "bank_digest_chain": bank_digest_chain,
        "committed_rows": len(bank),
        "committed_label_counts": committed_label_counts,
        "direction_histograms": [list(map(int, item)) for item in direction_histograms],
        "weight_sums": weight_sums,
        "measurement_reasons": measurement_reasons,
        "gate_reasons": gate_reasons,
        "group_forward_chunks": group_forward_chunks,
        "hyperparameters": hp.payload(),
        "row_spec": spec.payload(),
        "leakage_flags": dict(plan.LEAKAGE_FLAGS_BY_ROW[spec.row_id]),
        "estimator": f"{spec.law} + anchored block refit + trust region under the three-factor gate",
        "slim": bool(slim),
    }


# ---------------------------------------------------------------------------
# Session-row assembly (§9 receipts).
# ---------------------------------------------------------------------------


def _session_row(
    runtime: Any,
    *,
    row_id: str,
    budget: int,
    session: Any,
    targets: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    rollout: Mapping[str, Any],
    hp: Optional[CellHyperparameters] = None,
    governing: bool = True,
) -> dict[str, Any]:
    matrix_r2, joined = runtime._matrix_r2(rollout["per_trial_filtered"], targets, masks)
    row: dict[str, Any] = {
        "schema": "support_anchored_t4_stage_p_row_session_v1",
        "row": row_id,
        "budget": int(budget),
        "surface": session.surface,
        "session": session.session,
        "governing": bool(governing),
        "n_windows": int(joined.shape[0]),
        "matrix_r2": matrix_r2,
        "house_raw_r2": runtime._house_raw_r2(rollout["per_trial_raw"], targets, masks),
        "prediction_sha256_raw": runtime._raw_prediction_digest(rollout["per_trial_raw"], masks),
        "filtered_prediction_sha256": hashlib.sha256(joined.tobytes()).hexdigest(),
        "output_filter": plan.OUTPUT_FILTER["family"] + f"_a{plan.OUTPUT_FILTER['alpha']}",
        "initial_carrier_sha256": str(rollout["initial"]["initial_carrier_sha256"]),
        "initial_activity_sha256": str(rollout["initial"]["initial_activity_sha256"]),
        "group_assignment_sha256": str(rollout["initial"]["group_assignment_sha256"]),
        "leakage_flags": dict(rollout["leakage_flags"]),
        "estimator": rollout["estimator"],
        "row_spec": dict(rollout.get("row_spec") or {}),
        "target_optimizer_backward_update": 0,
        "target_backward_calls": 0,
        "target_parameter_update_calls": 0,
        "normalizer_update_calls": 0,
    }
    receipts = rollout.get("receipts") or rollout.get("per_trial_receipts")
    row["per_trial_receipts"] = list(receipts)
    row["causality_state_chain_verified"] = stage_o_replay.verify_causality_receipts(receipts)
    accepted = [item for item in receipts if item.get("acc")]
    row["carrier_transitions_committed"] = len(accepted)
    row["activity_transitions_committed"] = len(receipts)
    row["committed_evidence_rows"] = int(rollout.get("committed_rows") or 0)
    row["final_carrier_sha256"] = rollout.get("final_carrier_digest")
    row["carrier_rejection_counts"] = {
        str(key): int(value) for key, value in {
            **(rollout.get("gate_reasons") or {}), **(rollout.get("measurement_reasons") or {}),
        }.items()
    }
    if "transitions" in rollout:
        transitions = rollout["transitions"]
        row["carrier_transitions_committed"] = sum(
            1 for item in transitions if item.carrier_transition_committed
        )
        row["final_carrier_sha256"] = transitions[-1].carrier_after_sha256 if transitions else None
    anchor_obj = rollout.get("anchor")
    if anchor_obj is not None:
        row["support_anchor"] = {
            "anchor_sha256": anchor_obj.digest,
            "a0_b0_sha256": anchor_obj.a0_b0_digest,
            "statistics_sha256": anchor_obj.statistics.digest,
            "support_t4_sha256": cdm_core.array_digest(np.asarray(anchor_obj.support_t4)),
            "coefficient_parity": dict(anchor_obj.support_coefficient_parity),
        }
        movements = rollout.get("movements") or []
        d2 = rollout.get("d2_values") or []
        proposed = rollout.get("proposed_d2_values") or []
        published_hp = dict(rollout["hyperparameters"])
        row["movement"] = {
            "rho_M": float(published_hp["rho_M"]),
            "alpha_M": float(published_hp["alpha_M"]),
            "c_M": published_hp["c_M"],
            "n_committed": len(movements),
            "median_frobenius": float(np.median(movements)) if movements else None,
            "max_frobenius": float(np.max(movements)) if movements else None,
            "projected_commit_count": int(rollout.get("projected_count") or 0),
            "median_d2_unprojected": float(np.median(d2)) if d2 else None,
            "max_d2_unprojected": float(np.max(d2)) if d2 else None,
            "n_proposed_d2": len(proposed),
            "max_proposed_d2": float(np.max(proposed)) if proposed else None,
        }
        row["committed_label_counts"] = list(rollout.get("committed_label_counts") or [])
        row["direction_histograms"] = list(rollout.get("direction_histograms") or [])
        row["accepted_weight_sums"] = list(rollout.get("weight_sums") or [])
        row["bank_final_sha256"] = rollout["bank_digest_chain"][-1]
        row["group_forward_chunk_count"] = int(rollout.get("group_forward_chunks") or 0)
        row["hyperparameters"] = dict(rollout["hyperparameters"])
    return row


# ---------------------------------------------------------------------------
# Source-only hyperparameter selection (within-6 folds) and c_M calibration.
# ---------------------------------------------------------------------------


def _within_rollout_score(
    runtime: Any, *, session: Any, budget: int, spec: RowSpec, hp: CellHyperparameters,
) -> tuple[float, dict[str, Any]]:
    rollout = rollout_p(runtime, session=session, budget=budget, spec=spec, hp=hp, slim=True)
    targets, masks, _sst = runtime._session_target_views(session, budget)
    matrix_r2, _joined = runtime._matrix_r2(rollout["per_trial_filtered"], targets, masks)
    return float(matrix_r2), {
        "bank_final_sha256": rollout["bank_digest_chain"][-1],
        "committed_rows": rollout["committed_rows"],
        "committed_label_counts": list(rollout["committed_label_counts"]),
        "d2_values": [float(item) for item in rollout["d2_values"]],
        "proposed_d2_values": [float(item) for item in rollout["proposed_d2_values"]],
    }


def _selection_pass(
    runtime: Any,
    *,
    budget: int,
    within_sessions: Sequence[Any],
    started: float,
) -> dict[str, Any]:
    """The pre-registered factored within-6 selection for one low budget."""
    spec = ROW_SPECS["P2"]
    stage1: list[dict[str, Any]] = []
    winners: dict[str, Any] = {}
    for thresholds in enumerate_gate_grid():
        hp = CellHyperparameters(
            rho_M=float(plan.GATE_STAGE_FIXED["rho_M"]),
            alpha_M=float(plan.GATE_STAGE_FIXED["alpha_M"]),
            c_M=None, thresholds=thresholds,
        )
        per_session: dict[str, float] = {}
        for session in within_sessions:
            score, _detail = _within_rollout_score(
                runtime, session=session, budget=budget, spec=spec, hp=hp,
            )
            per_session[session.session] = score
            _require((time.monotonic() - started) <= plan.HARD_TIMEOUT_SECONDS,
                     "the Stage-P hard timeout fired during selection stage 1")
        stage1.append({
            "vector": thresholds.payload(),
            "fixed": dict(plan.GATE_STAGE_FIXED),
            "per_session": per_session,
            "mean_r2": float(sum(per_session.values()) / len(per_session)),
        })
    best_stage1 = select_vector(stage1)
    selected_thresholds = gate.GateThresholds(
        tau_d=float(best_stage1["vector"]["tau_d_rad"]),
        r_max=int(best_stage1["vector"]["r_max_repetition_per_direction"]),
        d_min=int(best_stage1["vector"]["d_min_distinct_directions"]),
        max_mass_relative=float(best_stage1["vector"]["max_pseudo_mass_relative_to_support_rows"]),
    )
    winners["stage1_selected"] = best_stage1["vector"]
    stage2: list[dict[str, Any]] = []
    stage2_details: dict[str, dict[str, Any]] = {}
    for rho_M, alpha_M in enumerate_mass_grid():
        hp = CellHyperparameters(
            rho_M=float(rho_M), alpha_M=float(alpha_M), c_M=None, thresholds=selected_thresholds,
        )
        per_session: dict[str, float] = {}
        details: dict[str, Any] = {}
        for session in within_sessions:
            score, detail = _within_rollout_score(
                runtime, session=session, budget=budget, spec=spec, hp=hp,
            )
            per_session[session.session] = score
            details[session.session] = detail
            _require((time.monotonic() - started) <= plan.HARD_TIMEOUT_SECONDS,
                     "the Stage-P hard timeout fired during selection stage 2")
        stage2.append({
            "vector": {"rho_M": float(rho_M), "alpha_M": float(alpha_M),
                       "thresholds": selected_thresholds.payload()},
            "per_session": per_session,
            "mean_r2": float(sum(per_session.values()) / len(per_session)),
        })
        stage2_details[f"rho{rho_M}_alpha{alpha_M}"] = details
    best_stage2 = select_vector(stage2)
    winner_key = f"rho{best_stage2['vector']['rho_M']}_alpha{best_stage2['vector']['alpha_M']}"
    winning_hp = CellHyperparameters(
        rho_M=float(best_stage2["vector"]["rho_M"]),
        alpha_M=float(best_stage2["vector"]["alpha_M"]),
        c_M=None, thresholds=selected_thresholds,
    )
    pooled_d2: list[float] = []
    for session in within_sessions:
        pooled_d2.extend(stage2_details[winner_key][session.session]["d2_values"])
    calibration: dict[str, Any] = {"law": dict(plan.C_M_CALIBRATION)}
    if pooled_d2:
        c_M = trust_region.calibrate_c_M(pooled_d2)
        winning_hp = CellHyperparameters(
            rho_M=winning_hp.rho_M, alpha_M=winning_hp.alpha_M, c_M=c_M,
            thresholds=selected_thresholds,
        )
        calibration.update({
            "c_M": float(c_M),
            "n_d2_values": len(pooled_d2),
            "d2_min": float(np.min(pooled_d2)),
            "d2_median": float(np.median(pooled_d2)),
            "d2_max": float(np.max(pooled_d2)),
            "sessions": [session.session for session in within_sessions],
        })
    else:
        calibration["note"] = "no committed block under the winning configuration; c_M unset"
    winners["stage2_selected"] = best_stage2["vector"]
    return {
        "law": dict(plan.SELECTION),
        "budget": int(budget),
        "objective_cell": "P2",
        "stage1_grid_size": len(stage1),
        "stage2_grid_size": len(stage2),
        "stage1_scored": stage1,
        "stage2_scored": stage2,
        "stage1_selected": winners["stage1_selected"],
        "stage2_selected": winners["stage2_selected"],
        "selected": winning_hp.payload(),
        "c_M_calibration": calibration,
        "fold_view_note": plan.SELECTION["fold_collapse_disclosure"],
        "tie_break": plan.SELECTION["tie_break"],
        "receipts_note": (
            "selection rollouts run SLIM (scores, bank digests and D2 values only); "
            "the governing grid re-derives every row with full §9 receipts"
        ),
    }


# ---------------------------------------------------------------------------
# The stage-O GO anchor (the binding precondition).
# ---------------------------------------------------------------------------


def verify_stage_o_go_anchor(base: Path) -> dict[str, Any]:
    """Re-read the sealed Stage-O terminal receipt and check the GO anchor."""
    terminal = _read_json(Path(base).absolute() / plan.STAGE_O_GO_ANCHOR["terminal_relative"])
    digest = hashlib.sha256(
        (Path(base).absolute() / plan.STAGE_O_GO_ANCHOR["terminal_relative"]).read_bytes()
    ).hexdigest()
    decision = str(terminal.get("decision"))
    summary = terminal.get("gate_summary", {})
    m4 = float(summary.get("m4_external_o2_minus_o0"))  # type: ignore[arg-type]
    m10 = float(summary.get("m10_external_o2_minus_o0"))  # type: ignore[arg-type]
    matches = (
        decision == plan.STAGE_O_GO_ANCHOR["decision"]
        and m4 == float(plan.STAGE_O_GO_ANCHOR["m4_external_o2_minus_o0"])
        and m10 == float(plan.STAGE_O_GO_ANCHOR["m10_external_o2_minus_o0"])
    )
    _require(matches, "the sealed Stage-O GO anchor no longer matches the work order binding")
    return {
        "terminal_relative": plan.STAGE_O_GO_ANCHOR["terminal_relative"],
        "terminal_sha256": digest,
        "decision": decision,
        "m4_external_o2_minus_o0": m4,
        "m10_external_o2_minus_o0": m10,
        "law": plan.STAGE_O_GO_ANCHOR["law"],
        "verified": True,
    }


# ---------------------------------------------------------------------------
# The stage driver.
# ---------------------------------------------------------------------------


def run_stage_p_replay(
    base: Path,
    *,
    gpu_index: int,
    output_root: Path,
) -> Mapping[str, object]:
    """Select on within-6, calibrate c_M, then run the P0--P5 grid once."""
    base = Path(base).absolute()
    output = Path(output_root)
    _require(not (output / "terminal.json").exists(), "the Stage-P terminal receipt already exists")
    _require(not (output / "replay.json").exists(),
             "the Stage-P replay receipt already exists; run --stage terminal, never a second replay")
    _require((output / "attempt.json").exists(), "reserve the Stage-P attempt first")
    attempt_sha = _verify_receipt(output / "attempt.json")
    attempt = _read_json(output / "attempt.json")
    predecessors = plan.predecessor_sha256s(base)
    _require(
        predecessors == attempt["predecessor_sha256s"],
        "an immutable predecessor drifted between attempt and replay",
    )
    owned = plan.owned_sha256s(base)
    _require(owned == attempt["owned_sha256s"], "an owned Stage-P module drifted between attempt and replay")
    go_anchor = verify_stage_o_go_anchor(base)

    started = time.monotonic()
    from src.causal_dual_memory_cell_d_score_v1 import score as v1score
    from src.learned_gate_p2prime_v1 import physical as p2physical

    environment = p2physical.validate_environment(gpu_index=gpu_index)
    runtime, meta, identity = p2physical._runtime_for(base, gpu_index=gpu_index)
    try:
        runtime.prepare(identity=identity)
        fixed = v1score.derive_fixed_evaluation_authority(base)
        runtime.materialize_inputs(identity=identity, authority=fixed)
        state = runtime._require_state()
        arm = state.modules["arm_common"]
        model_digest_before = arm.state_sha256(state.model)
        within_sessions = [state.sessions[key] for key in p2physical._ordered_session_keys(runtime, "within")]
        _require(len(within_sessions) == plan.WITHIN_SESSION_COUNT,
                 "the within surface must carry exactly six sessions")

        # -- within-6 selection + c_M calibration (external-15 never selects) ---
        selection: dict[str, Any] = {"law": dict(plan.SELECTION)}
        selected_hp: dict[int, CellHyperparameters] = {}
        for budget in plan.LOW_BUDGETS:
            payload = _selection_pass(
                runtime, budget=budget, within_sessions=within_sessions, started=started,
            )
            selection[f"m{budget}"] = payload
            chosen = payload["selected"]
            selected_hp[int(budget)] = CellHyperparameters(
                rho_M=float(chosen["rho_M"]),
                alpha_M=float(chosen["alpha_M"]),
                c_M=(None if chosen["c_M"] is None else float(chosen["c_M"])),
                thresholds=gate.GateThresholds(
                    tau_d=float(chosen["thresholds"]["tau_d_rad"]),
                    r_max=int(chosen["thresholds"]["r_max_repetition_per_direction"]),
                    d_min=int(chosen["thresholds"]["d_min_distinct_directions"]),
                    max_mass_relative=float(
                        chosen["thresholds"]["max_pseudo_mass_relative_to_support_rows"]
                    ),
                ),
            )
            _require((time.monotonic() - started) <= plan.HARD_TIMEOUT_SECONDS,
                     "the Stage-P hard timeout fired during selection")
        # M30: the exact no-op (alpha_M = 0), inheriting the M10 thresholds.
        m30_thresholds = selected_hp[10].thresholds
        selected_hp[30] = CellHyperparameters(
            rho_M=1.0, alpha_M=0.0, c_M=None, thresholds=m30_thresholds,
        )
        selection["m30"] = {
            "law": plan.HYPERPARAMETERS["m30"]["law"],
            "alpha_M": 0.0,
            "rho_M": 1.0,
            "c_M": None,
            "inherits_thresholds_from": "m10 (within-6 source-selected; movement is zero by law)",
            "thresholds": m30_thresholds.payload(),
        }

        # -- the governing grid ---------------------------------------------------
        matrix: dict[str, Any] = {}
        anchors_sealed: dict[str, Any] = {}
        anchors_filter_line: dict[str, Any] = {}
        causality_all = True
        movement_medians_by_session: dict[str, dict[str, dict[str, Optional[float]]]] = {}
        m30_noop_all = True
        trust_region_no_committed_drift = True
        sealed_cells = stage_o_replay._sealed_activity_cells(base)
        p0_m30_raw: dict[str, list[np.ndarray]] = {}
        for row_id in plan.ROW_ORDER:
            for budget in plan.BUDGETS:
                hp = selected_hp[int(budget)]
                for surface in plan.SURFACES:
                    session_rows: list[dict[str, Any]] = []
                    for key in p2physical._ordered_session_keys(runtime, surface):
                        session = state.sessions[key]
                        targets, masks, _sst = runtime._session_target_views(session, budget)
                        if row_id == "P0":
                            rollout = stage_o_replay.rollout_o0(runtime, session=session, budget=budget)
                            raw_row = rollout["rows"]["O0_raw"]
                            governing = rollout["rows"]["O0"]
                            row = _session_row(
                                runtime, row_id="P0", budget=budget, session=session,
                                targets=targets, masks=masks,
                                rollout={
                                    "per_trial_filtered": rollout["per_trial_filtered"],
                                    "per_trial_raw": rollout["per_trial_raw"],
                                    "initial": rollout["initial"],
                                    "leakage_flags": rollout["leakage_flags"],
                                    "estimator": "frozen support T4 (activity-only CDM; the Stage-O O0 arm)",
                                    "per_trial_receipts": rollout["per_trial_receipts"],
                                    "transitions": rollout["transitions"],
                                },
                                hp=None,
                            )
                            _require(
                                row["house_raw_r2"] == float(raw_row["house_raw_r2"])
                                and row["prediction_sha256_raw"] == str(raw_row["prediction_sha256_raw"])
                                and row["matrix_r2"] == float(governing["matrix_r2"]),
                                "P0 row disagrees with the sealed activity-only row payload",
                            )
                            anchors_sealed[f"{session.session}:m{budget}:{surface}"] = (
                                stage_o_replay.anchor_o0_vs_sealed(
                                    row=row,
                                    sealed_row=sealed_cells[(int(budget), surface)][session.session],
                                    label=f"P0 vs sealed activity-only m{budget} {surface} {session.session}",
                                )
                            )
                            anchors_filter_line[f"{session.session}:m{budget}:{surface}"] = (
                                stage_o_replay.anchor_o0_vs_filter_line_cache(
                                    base=base, flat_raw=rollout["flat_raw"], surface=surface,
                                    session=session.session, budget=budget,
                                )
                            )
                            if budget == 30:
                                p0_m30_raw[(surface, session.session)] = list(rollout["per_trial_raw"])
                        else:
                            rollout = rollout_p(
                                runtime, session=session, budget=budget,
                                spec=ROW_SPECS[row_id], hp=hp,
                            )
                            row = _session_row(
                                runtime, row_id=row_id, budget=budget, session=session,
                                targets=targets, masks=masks, rollout=rollout, hp=hp,
                                governing=row_id not in plan.DIAGNOSTIC_ROWS,
                            )
                            # No COMMITTED block may sit outside the trust region.
                            for receipt in rollout["receipts"]:
                                if receipt.get("acc") and receipt.get("d2") is not None:
                                    if hp.c_M is not None and float(receipt["d2"]) > float(hp.c_M):
                                        trust_region_no_committed_drift = False
                            if budget == 30:
                                # The exact no-op law: bit-equal predictions, a
                                # frozen carrier and zero movement on every trial.
                                reference = p0_m30_raw.get((surface, session.session))
                                _require(reference is not None,
                                         "the P0 M30 pass must precede the P rows at M30")
                                for left, right in zip(rollout["per_trial_raw"], reference):
                                    _require(np.array_equal(np.asarray(left), np.asarray(right)),
                                             f"M30 no-op law broken by {row_id} in {session.session}")
                                for receipt in rollout["receipts"]:
                                    _require(receipt["cb"] == receipt["ca"],
                                             f"M30 carrier moved under {row_id} in {session.session}")
                                    _require(not receipt["acc"],
                                             f"M30 carrier transition committed under {row_id}")
                                row["m30_noop_verified"] = True
                                m30_noop_all = m30_noop_all and True
                            if row_id in plan.CANDIDATE_ROWS:
                                movement_medians_by_session.setdefault(row_id, {}).setdefault(
                                    (surface, session.session), {}
                                )[f"m{budget}"] = row["movement"]["median_frobenius"]
                        causality_all = causality_all and bool(row["causality_state_chain_verified"])
                        _require(row["causality_state_chain_verified"],
                                 f"causality receipt chain broken for {row_id} {session.session} m{budget}")
                        session_rows.append(row)
                        _require(
                            (time.monotonic() - started) <= plan.HARD_TIMEOUT_SECONDS,
                            "the Stage-P hard timeout fired during the governing grid",
                        )
                    per_session = {
                        str(item["session"]): float(item["matrix_r2"]) for item in session_rows
                    }
                    entry = {
                        "mean_r2": float(sum(per_session.values()) / len(per_session)),
                        "per_session": per_session,
                        "sessions": [dict(item) for item in session_rows],
                        "full_per_trial_state_receipts": not any(
                            bool(item.get("slim")) for item in session_rows
                        ),
                    }
                    matrix.setdefault(f"m{budget}", {}).setdefault(surface, {})[row_id] = entry

        model_digest_after = arm.state_sha256(runtime._require_state().model)
        _require(
            model_digest_before == model_digest_after,
            "the sealed Cell-D model state mutated during the Stage-P replay",
        )
        movement_by_budget: dict[str, object] = {}
        for row_id in plan.CANDIDATE_ROWS:
            for budget in plan.BUDGETS:
                values = [
                    movement_medians_by_session[row_id][session][f"m{budget}"]
                    for session in sorted(movement_medians_by_session[row_id])
                    if movement_medians_by_session[row_id][session].get(f"m{budget}") is not None
                ]
                if values:
                    movement_by_budget[f"{row_id}_m{budget}"] = float(np.median(values))
        movement_by_budget["m4"] = movement_by_budget.get("P2_m4", 0.0)
        movement_by_budget["m10"] = movement_by_budget.get("P2_m10", 0.0)
        movement_by_budget["m30"] = 0.0  # alpha_M = 0: zero by construction
        resources = dict(runtime._resources())
    finally:
        runtime.close()

    payload = {
        "schema": f"{plan.SCHEMA}_replay_v1",
        "stage": "p0_to_p5_deployable_pseudo_direction_grid",
        "status": "REPLAY_COMPLETE",
        "attempt_sha256": attempt_sha,
        "identity_sha256": meta["identity_sha256"],
        "environment": environment,
        "gpu_index": int(gpu_index),
        "budgets": list(plan.BUDGETS),
        "surfaces": list(plan.SURFACES),
        "rows": list(plan.ROW_ORDER),
        "commit_law": dict(plan.COMMIT_LAW),
        "hyperparameters": dict(plan.HYPERPARAMETERS),
        "hyperparameter_selection": selection,
        "matrix": matrix,
        "anchors": {
            "sealed_activity_only": anchors_sealed,
            "sealed_activity_only_all_exact": all(
                bool(item["exact_match"]) for item in anchors_sealed.values()
            ),
            "filter_line_cache": anchors_filter_line,
            "filter_line_cache_all_bitwise": all(
                bool(item.get("raw_bitwise_equal")) for item in anchors_filter_line.values()
                if item.get("available")
            ),
            "stage_o_go": go_anchor,
            "m30_noop_all_rows": bool(m30_noop_all),
        },
        "movement_by_budget_medians": movement_by_budget,
        "movement_by_budget_note": plan.MOVEMENT_ORDERING["statistic"],
        "trust_region_no_committed_drift": bool(trust_region_no_committed_drift),
        "causality_state_chains_all_rows": bool(causality_all),
        "model_state_digest_before_sha256": model_digest_before,
        "model_state_digest_after_sha256": model_digest_after,
        "model_state_digest_unchanged": model_digest_before == model_digest_after,
        "resources": resources,
        "wall_seconds": float(time.monotonic() - started),
        "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "model_or_checkpoint_updated": False,
        "decoder_training": False,
        "target_optimizer_backward_update": 0,
        "target_backward_calls": 0,
        "target_parameter_update_calls": 0,
        "normalizer_update_calls": 0,
    }
    _require(
        payload["anchors"]["sealed_activity_only_all_exact"],
        "the P0 anchor failed at least one sealed activity-only session row",
    )
    _require(payload["anchors"]["m30_noop_all_rows"], "the M30 exact no-op law failed")
    return payload


# ---------------------------------------------------------------------------
# Terminal composition (receipt-only; no data, model or CUDA access).
# ---------------------------------------------------------------------------


def build_terminal(*, replay_payload: Mapping[str, object]) -> Mapping[str, object]:
    from src.tfpd_lane.matched_scorer import paired_session_stats

    from . import gates as gates_module

    matrix = replay_payload["matrix"]
    _require(isinstance(matrix, Mapping), "terminal needs the replay matrix")
    view = {
        budget_key: {
            surface: {
                row: {
                    "mean_r2": float(entry["mean_r2"]),
                    "per_session": {
                        str(session): float(value)
                        for session, value in entry["per_session"].items()
                    },
                }
                for row, entry in rows.items()
            }
            for surface, rows in surfaces.items()
        }
        for budget_key, surfaces in matrix.items()
    }
    anchors = replay_payload["anchors"]
    selection = replay_payload["hyperparameter_selection"]
    safety = {
        "zero_target_updates": bool(
            replay_payload.get("target_optimizer_backward_update") == 0
            and replay_payload.get("target_backward_calls") == 0
            and replay_payload.get("target_parameter_update_calls") == 0
            and replay_payload.get("normalizer_update_calls") == 0
            and replay_payload.get("model_state_digest_unchanged") is True
        ),
        "p0_bit_anchor": bool(anchors["sealed_activity_only_all_exact"]),
        "causality_state_chains": bool(replay_payload["causality_state_chains_all_rows"]),
        "within_regression_bound": True,   # evaluated by the promotion gate from the matrix
        "m30_exact_noop": bool(anchors["m30_noop_all_rows"]),
        "hyperparameters_source_selected_only": bool(
            selection.get("law", {}).get("surface") == plan.SELECTION["surface"]
        ),
        "trust_region_no_committed_drift": bool(
            replay_payload["trust_region_no_committed_drift"]
        ),
    }
    promotion = gates_module.evaluate_promotion_gate(view, safety=safety)
    continuity = gates_module.continuity_attribution(view)
    confidence = gates_module.confidence_attribution(view)
    stops = gates_module.stop_conditions(view, promotion, safety=promotion["safety"])
    interpretation = gates_module.interpretation_rows(view, promotion)
    ordering = gates_module.movement_ordering(replay_payload["movement_by_budget_medians"])
    paired: dict[str, object] = {}
    contrasts = (
        ("P1", "P0"), ("P2", "P0"), ("P3", "P0"), ("P4", "P0"), ("P5", "P0"),
        ("P2", "P1"), ("P2", "P3"), ("P2", "P4"),
    )
    for budget in plan.BUDGETS:
        for surface in plan.SURFACES:
            rows = view[f"m{budget}"][surface]
            for candidate, baseline in contrasts:
                paired[f"{candidate}_minus_{baseline}_m{budget}_{surface}"] = paired_session_stats([
                    float(rows[candidate]["per_session"][session])
                    - float(rows[baseline]["per_session"][session])
                    for session in sorted(rows[candidate]["per_session"])
                ])
    return {
        "gate": promotion,
        "continuity_attribution": continuity,
        "confidence_attribution": confidence,
        "stop_conditions": stops,
        "interpretation_rows": interpretation,
        "movement_ordering": ordering,
        "paired_contrasts": paired,
        "hyperparameter_selection_summary": {
            key: value.get("selected") for key, value in selection.items()
            if isinstance(value, Mapping) and "selected" in value
        },
        "stage_o_go_anchor": anchors["stage_o_go"],
    }
