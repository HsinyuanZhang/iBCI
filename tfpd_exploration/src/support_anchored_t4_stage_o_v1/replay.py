"""Stage-O replay: the O0/O1/O2 cells over the frozen activity-only runtime.

Design §3 (cells), §3.2 (causal discipline), §4 (anchored refit), §7.1.
The frozen ``src/learned_gate_p2prime_v1`` machinery is reused exactly as the
sealed drivers used it and never edited:

* ``p2physical._runtime_for`` / ``prepare`` / ``materialize_inputs`` -- the
  sealed parser, chronology, normalizer and Cell-D SWA checkpoint;
* ``runtime._rollout_activity`` -- O0 is the SEALED activity-only loop itself
  (A0 raw row + A1 filtered row), which is what makes the O0 bit-anchor a
  no-op proof of this route's loop;
* ``runtime._governing_forward(trial, inputs)`` -- the sealed forward
  primitive, called with inputs whose side reflects the current ``T4_active``
  exactly as ``_horizon_counterfactual`` builds its branches;
* ``policy.build_construction_predictions(construction="true")`` -- the frozen
  TRUE completed-trial direction construction;
* ``core.IndependentActivityCausalDualMemory`` -- the frozen independent
  activity law for O1 (observe -> propose -> commit-always, the C0 law with a
  true direction) and the activity transitions of O2.

O2 is the route-owned anchored loop: per trial (1) forward with the current
``T4_active``, (2) finalize, (3) reveal the TRUE direction (leakage-labelled),
(4) append the evidence row, (5) recompute the block refit from ``A0/b0`` +
bank, (6) project through the trust region, (7) set the next-trial
``T4_active``.  An update committed from trial i can never affect trial i's
own predictions: every forward reads the state digest recorded before the
reveal, and the receipt chain asserts ``state_after[i-1] == forward_state[i]``.
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

try:
    from src.causal_dual_memory_cell_d_v1 import core as cdm_core
except ModuleNotFoundError as error:
    if not (error.name == "src" or str(error.name).startswith("src.")): raise
    from tfpd_exploration.src.causal_dual_memory_cell_d_v1 import core as cdm_core

from . import plan
from .anchor import SupportAnchor
from .block_refit import EvidenceBank, EvidenceRow, refit_from_anchor
from .trust_region import active_carrier, calibrate_c_M


class StageOReplayError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StageOReplayError(message)


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


# ---------------------------------------------------------------------------
# Frozen-law helpers.
# ---------------------------------------------------------------------------


def _p2_policy():
    from src.learned_gate_p2prime_v1 import policy as p2policy

    return p2policy


def _p2_filters():
    from src.learned_gate_p2prime_v1 import filters as p2filters

    return p2filters


def _v1_plan():
    from src.causal_dual_memory_cell_d_score_v1 import plan as v1plan

    return v1plan


def _support_binding(session: Any, budget: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The exact support rates/directions the frozen ``_initial_memory`` uses."""
    np_module = np
    support_ids = tuple(session.support_trial_ids[budget])
    rates = np_module.ascontiguousarray(
        np_module.stack([session.support_rates[item] for item in support_ids]), dtype=np_module.float64,
    )
    directions = np_module.ascontiguousarray(
        np_module.asarray([session.support_direction_indices[item] for item in support_ids], dtype=np_module.int64),
    )
    _require(rates.shape == (budget, session.channel_ids.size) and directions.shape == (budget,),
             f"M{budget} support rate/label topology drift")
    return rates, directions, list(support_ids)


def true_direction_for_trial(
    session: Any, trial: Any, *, config: Any,
) -> tuple[Any, Mapping[str, Any]]:
    """The frozen TRUE completed-trial direction of one query trial.

    Exactly the P2' oracle law: restore the sealed z-scored behavior rows to
    physical velocity units, then run ``pseudo_direction_from_velocity`` under
    the session carrier's own frozen config.
    """
    v1plan = _v1_plan()
    p2filters = _p2_filters()
    restored, padded = p2filters.true_physical_velocity_one_trial(
        np.asarray(session.behavior[trial.endpoint_bins]),
        behavior_mean=v1plan.SEALED_BEHAVIOR_MEAN,
        behavior_std=v1plan.SEALED_BEHAVIOR_STD,
    )
    direction = cdm_core.pseudo_direction_from_velocity(
        restored, trial.velocity_validity.valid_mask, config=config,
    )
    meta = {"padded_true_rows": int(np.sum(padded))}
    return direction, meta


def true_views_for_trial(session: Any, trial: Any) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    """The frozen ``construction="true"`` complementary views of one trial.

    The four wrapper inputs carry only the trial's validity evidence; the
    frozen construction replaces the velocity content with the TRUE restored
    rows, so no complementary-group forward is needed for a true direction
    (recorded in the receipt; the O0 arm never runs group forwards either).
    """
    v1plan = _v1_plan()
    p2filters = _p2_filters()
    p2policy = _p2_policy()
    behavior_rows = session.behavior[trial.endpoint_bins]
    restored, _padded = p2filters.true_physical_velocity_one_trial(
        np.asarray(behavior_rows),
        behavior_mean=v1plan.SEALED_BEHAVIOR_MEAN,
        behavior_std=v1plan.SEALED_BEHAVIOR_STD,
    )
    wrappers = tuple(
        cdm_core.CompletedVelocityPrediction(restored, trial.velocity_validity)
        for _ in range(cdm_core.GROUP_COUNT)
    )
    views, meta = p2policy.build_construction_predictions(
        group_predictions=wrappers, construction="true", behavior_rows=behavior_rows,
        behavior_mean=v1plan.SEALED_BEHAVIOR_MEAN, behavior_std=v1plan.SEALED_BEHAVIOR_STD,
    )
    return views, meta


def _direction_row_payload(trial: Any, direction: Any, meta: Mapping[str, Any]) -> dict[str, object]:
    return {
        "trial_id": trial.trial_id,
        "chronology_position": int(trial.chronology_position),
        "n_windows": int(np.asarray(trial.endpoint_bins).size),
        "accepted": bool(direction.accepted),
        "reason": None if direction.reason is None else direction.reason.value,
        "theta_raw_rad": direction.theta_raw_rad,
        "theta_index": None if direction.theta_index is None else int(direction.theta_index),
        "theta_canonical_rad": direction.theta_canonical_rad,
        "canonical_distance_rad": direction.canonical_distance_rad,
        "movement_bins": int(direction.movement_bins),
        "displacement_norm": direction.displacement_norm,
        "mean_speed": direction.mean_speed,
        "padded_true_rows": int(meta["padded_true_rows"]),
    }


def _binding_payload(budget: int, support_ids: Sequence[str], initial: Mapping[str, Any], config: Any) -> dict[str, object]:
    return {
        "budget": int(budget),
        "support_trial_ids": list(support_ids),
        "initial_carrier_sha256": str(initial["initial_carrier_sha256"]),
        "initial_activity_sha256": str(initial["initial_activity_sha256"]),
        "group_assignment_sha256": str(initial["group_assignment_sha256"]),
        "config_payload_sha256": hashlib.sha256(
            json.dumps(config.payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _verify_direction_cell(
    *,
    direction_rows: list[Mapping[str, Any]],
    binding: Mapping[str, Any],
    table_cell: Mapping[str, Any],
    label: str,
) -> None:
    from . import directions as directions_module

    digest = directions_module.direction_cell_digest(direction_rows, binding)
    _require(
        digest == str(table_cell["cell_digest"]),
        f"true-direction table drift for {label}: recomputed cell digest differs",
    )


def _reason_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    candidate = getattr(value, "value", value)
    return candidate if isinstance(candidate, str) and candidate else "unknown"


def _counts_by_reason(reasons: Sequence[Optional[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reason in reasons:
        if reason is None:
            continue
        counts[str(reason)] = counts.get(str(reason), 0) + 1
    return counts


# ---------------------------------------------------------------------------
# O0: the sealed activity-only loop (the no-op proof).
# ---------------------------------------------------------------------------


def rollout_o0(runtime: Any, *, session: Any, budget: int) -> dict[str, Any]:
    """Run the sealed activity-only rollout and stage its receipts."""
    p2filters = _p2_filters()
    rollout = runtime._rollout_activity(session=session, budget=budget)
    transitions = rollout["transitions"]
    per_trial = [
        {
            "trial_id": item.trial_id,
            "fs": item.state_before_sha256,
            "sb": item.state_before_sha256,
            "sa": item.state_after_sha256,
            "ab": item.activity_before_sha256,
            "aa": item.activity_after_sha256,
            "cb": item.carrier_before_sha256,
            "ca": item.carrier_after_sha256,
            "acc": False,
            "reason": item.carrier_rejection_reason,
        }
        for item in transitions
    ]
    rows = {
        "O0_raw": rollout["rows"]["A0"],
        "O0": rollout["rows"]["A1"],
    }
    per_trial_raw = rollout["per_trial_raw"]
    return {
        "row": "O0",
        "rows": rows,
        "per_trial_raw": per_trial_raw,
        "per_trial_filtered": rollout["per_trial_filtered"],
        "transitions": transitions,
        "per_trial_receipts": per_trial,
        "initial": rollout["initial"],
        "flat_raw": _flat_raw(per_trial_raw),
        "output_filter_note": (
            "O0 governing row is the sealed filtered (causal EMA a=0.25) A1 row; the "
            "raw A0 row carries the sealed activity-only bit-anchor fields"
        ),
        "group_forward_count": 0,
        "leakage_flags": dict(plan.LEAKAGE_FLAGS_BY_ROW["O0"]),
    }


def _flat_raw(per_trial_raw: Sequence[np.ndarray]) -> np.ndarray:
    flat = np.ascontiguousarray(
        np.concatenate([np.asarray(item, dtype=np.float32) for item in per_trial_raw], axis=0),
    )
    flat.setflags(write=False)
    return flat


# ---------------------------------------------------------------------------
# O1: the existing recursive incremental estimator fed the TRUE direction.
# ---------------------------------------------------------------------------


def rollout_o1(runtime: Any, *, session: Any, budget: int, table_cell: Mapping[str, Any]) -> dict[str, Any]:
    p2filters = _p2_filters()
    runtime._memory_mode = "cdm"
    memory, initial = runtime._initial_memory(session=session, budget=budget)
    config = memory.state.carrier.config
    targets, masks, _sst = runtime._session_target_views(session, budget)
    query_ids = list(session.query_trial_ids[budget])
    per_trial_raw: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    direction_rows: list[Mapping[str, Any]] = []
    initial_state_digest = memory.state.digest
    expected_state = initial_state_digest
    for trial_id in query_ids:
        trial = session.trials_by_id[trial_id]
        inputs = memory.read_prediction_inputs()
        state_before = memory.state.digest
        _require(inputs.state_digest == state_before, "O1 forward state digest drift")
        _require(state_before == expected_state, "O1 causality chain drift: state before != previous commit state")
        per_trial_raw.append(runtime._governing_forward(trial=trial, inputs=inputs))
        direction, direction_meta = true_direction_for_trial(session, trial, config=config)
        direction_rows.append(_direction_row_payload(trial, direction, direction_meta))
        views, _views_meta = true_views_for_trial(session, trial)
        pending = memory.observe_completed_trial(
            b3s_trial_activity=trial.b3s_activity,
            carrier_trial_counts=trial.native_counts,
            complementary_predictions=views,
        )
        transition = runtime._commit_dispatch(memory=memory, pending=pending, trial_id=trial_id)
        accepted = bool(transition.carrier_transition_committed)
        receipts.append({
            "trial_id": trial_id,
            "fs": inputs.state_digest,
            "sb": transition.state_before_sha256,
            "sa": transition.state_after_sha256,
            "ab": transition.activity_before_sha256,
            "aa": transition.activity_after_sha256,
            "cb": transition.carrier_before_sha256,
            "ca": transition.carrier_after_sha256,
            "acc": accepted,
            "reason": transition.carrier_rejection_reason,
            "d_idx": None if direction.theta_index is None else int(direction.theta_index),
            "d_acc": bool(direction.accepted),
        })
        expected_state = transition.state_after_sha256
        runtime._bump_trial_counters()
    rates, directions, support_ids = _support_binding(session, budget)
    _verify_direction_cell(
        direction_rows=direction_rows,
        binding=_binding_payload(budget, support_ids, initial, config),
        table_cell=table_cell,
        label=f"O1 {session.surface} M{budget} {session.session}",
    )
    per_trial_filtered = [p2filters.apply_output_filter_one_trial(item) for item in per_trial_raw]
    return {
        "row": "O1",
        "per_trial_raw": per_trial_raw,
        "per_trial_filtered": per_trial_filtered,
        "receipts": receipts,
        "initial": initial,
        "leakage_flags": dict(plan.LEAKAGE_FLAGS_BY_ROW["O1"]),
        "estimator": "existing recursive incremental update (observe -> propose -> commit-always)",
        "group_forward_count": 0,
        "true_construction_group_forwards_skipped": True,
    }


# ---------------------------------------------------------------------------
# O2: the support-anchored block refit with trust-region projection.
# ---------------------------------------------------------------------------


def _anchored_carrier(
    *,
    groups: Any, initial_raw_t4: np.ndarray, statistics: Any, config: Any, active_t4: np.ndarray,
) -> Any:
    return cdm_core.CarrierMemory(
        groups=groups,
        initial_raw_t4=initial_raw_t4,
        statistics=statistics,
        ordinary_t4=active_t4,
        fixed_ridge_t4=active_t4,
        config=config,
    )


def rollout_o2(
    runtime: Any,
    *,
    session: Any,
    budget: int,
    alpha_M: float,
    c_M: Optional[float],
    table_cell: Mapping[str, Any],
) -> dict[str, Any]:
    """The route-owned anchored loop (always-commit + trust-region projection)."""
    p2filters = _p2_filters()
    runtime._memory_mode = "cdm"
    memory, initial = runtime._initial_memory(session=session, budget=budget)
    carrier0 = memory.state.carrier
    config = carrier0.config
    rates, directions, support_ids = _support_binding(session, budget)
    anchor = SupportAnchor.from_labeled_support(
        groups=carrier0.groups,
        support_trial_rates=rates,
        support_direction_indices=directions,
        support_t4=carrier0.active_t4,
    )
    initial_raw = np.asarray(carrier0.initial_raw_t4)
    rho_M = float(plan.HYPERPARAMETERS["rho_M"])
    targets, masks, _sst = runtime._session_target_views(session, budget)
    query_ids = list(session.query_trial_ids[budget])
    per_trial_raw: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    direction_rows: list[Mapping[str, Any]] = []
    bank = EvidenceBank.empty()
    active = np.asarray(carrier0.active_t4)
    expected_state = memory.state.digest
    movements: list[float] = []
    d2_values: list[float] = []
    projected_count = 0
    bank_digest_chain: list[str] = [bank.digest]
    for index, trial_id in enumerate(query_ids):
        trial = session.trials_by_id[trial_id]
        inputs = memory.read_prediction_inputs()
        state_before = memory.state.digest
        _require(inputs.state_digest == state_before, "O2 forward state digest drift")
        _require(state_before == expected_state, "O2 causality chain drift: state before != previous commit state")
        _require(np.array_equal(np.asarray(inputs.active_t4), active), "O2 active T4 drift before forward")
        per_trial_raw.append(runtime._governing_forward(trial=trial, inputs=inputs))
        # (2)/(3): finalize, then reveal the TRUE direction.
        direction, direction_meta = true_direction_for_trial(session, trial, config=config)
        direction_rows.append(_direction_row_payload(trial, direction, direction_meta))
        pseudo = tuple(
            cdm_core.pseudo_direction_from_velocity(
                view.velocity, view.validity.valid_mask, config=config,
            )
            for view in true_views_for_trial(session, trial)[0]
        )
        scalar_rates = cdm_core.scalar_rates_from_native_rewarded_counts(trial.native_counts)
        failed = next((item.reason for item in pseudo if not item.accepted), None)
        activity_candidate, fifo_changed = memory._activity_candidate(trial.b3s_activity)
        commit_reason = None
        tr_payload: Optional[Mapping[str, Any]] = None
        if failed is None:
            row = EvidenceRow(
                session_id=session.session,
                trial_id=trial_id,
                chronology_position=int(trial.chronology_position),
                block_index=len(bank),
                direction_indices=tuple(int(item.theta_index) for item in pseudo),
                theta_raw_rad=tuple(float(item.theta_raw_rad) for item in pseudo),
                canonical_distance_rad=tuple(
                    0.0 if item.canonical_distance_rad is None else float(item.canonical_distance_rad)
                    for item in pseudo
                ),
                movement_bins=tuple(int(item.movement_bins) for item in pseudo),
                displacement_norm=tuple(
                    0.0 if item.displacement_norm is None else float(item.displacement_norm)
                    for item in pseudo
                ),
                mean_speed=tuple(
                    0.0 if item.mean_speed is None else float(item.mean_speed)
                    for item in pseudo
                ),
                scalar_rates=scalar_rates,
                rate_sha256=cdm_core.array_digest(scalar_rates),
                native_counts_sha256=cdm_core.array_digest(trial.native_counts.counts),
                channel_order_sha256=cdm_core.channel_order_digest(session.channel_ids),
                valid_mask_sha256=cdm_core.array_digest(carrier0.groups.valid_mask),
            )
            bank = bank.with_row(row)
            refit = refit_from_anchor(anchor, bank, rho_M=rho_M)
            outcome = active_carrier(anchor, refit.coefficients, alpha_M=alpha_M, c_M=c_M)
            next_carrier = _anchored_carrier(
                groups=carrier0.groups, initial_raw_t4=initial_raw, statistics=anchor.statistics,
                config=config, active_t4=outcome.active_t4,
            )
            zero_movement = next_carrier.digest == memory.state.carrier.digest
            if zero_movement:
                # alpha_M = 0 (or a measure-zero zero delta) rebuilds the sealed
                # support carrier bit-exactly.  The evidence row is still
                # committed to the bank, but the frozen V3 law cannot represent
                # an "accepted" carrier transition that changes no bytes, so the
                # transition is recorded through the frozen no-evidence fallback
                # (the same convention P2' oracle_rejected_pending uses).
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
                    fallback=inputs,
                    completed_trial_evidence={
                        "schema": "support_anchored_t4_stage_o_completed_trial_evidence_v1",
                        "evidence_row_sha256": row.digest,
                        "block_refit_sha256": refit.digest,
                        "anchor_a0_b0_sha256": anchor.a0_b0_digest,
                        "commit_law": "always_commit_trust_region_projection_only",
                        "zero_movement": True,
                        "alpha_M": float(alpha_M),
                    },
                )
                commit_reason = "always_commit_zero_movement"
            else:
                candidate_state = cdm_core.DualMemoryState(
                    activity=activity_candidate.activity,
                    carrier=next_carrier,
                    committed_query_trials=activity_candidate.committed_query_trials,
                )
                proposal = cdm_core.CarrierProposal(
                    True, None, next_carrier,
                    tuple(dict(item) for item in refit.coverage_rows),
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
                    fallback=inputs,
                    completed_trial_evidence={
                        "schema": "support_anchored_t4_stage_o_completed_trial_evidence_v1",
                        "evidence_row_sha256": row.digest,
                        "block_refit_sha256": refit.digest,
                        "anchor_a0_b0_sha256": anchor.a0_b0_digest,
                        "commit_law": "always_commit_trust_region_projection_only",
                    },
                )
                active = np.asarray(next_carrier.active_t4)
                commit_reason = "always_commit"
            tr_payload = outcome.payload()
            movements.append(outcome.movement_frobenius)
            d2_values.append(outcome.d2_unprojected)
            projected_count += 1 if outcome.projected else 0
        else:
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
                carrier_rejection_reason=failed,
                pseudo_directions=pseudo,
                scalar_rates=scalar_rates,
                carrier_proposal=None,
                candidate_state=candidate_state,
                fallback=inputs,
                completed_trial_evidence={
                    "schema": "support_anchored_t4_stage_o_completed_trial_evidence_v1",
                    "commit_law": "always_commit_trust_region_projection_only",
                    "no_evidence_reason": failed.value,
                },
            )
            commit_reason = failed.value
        transition = runtime._commit_dispatch(memory=memory, pending=pending, trial_id=trial_id)
        # §9: every scored session records the state-before/state-after digests
        # of every completed trial (and the activity/carrier split) for every
        # row, sensitivity rows included.
        receipt: dict[str, Any] = {
            "trial_id": trial_id,
            "acc": bool(transition.carrier_transition_committed),
            "reason": commit_reason,
            "bd": bank.digest,
            "bm": bank.accepted_mass,
            "fs": inputs.state_digest,
            "sb": transition.state_before_sha256,
            "sa": transition.state_after_sha256,
            "ab": transition.activity_before_sha256,
            "aa": transition.activity_after_sha256,
            "cb": transition.carrier_before_sha256,
            "ca": transition.carrier_after_sha256,
        }
        if tr_payload is not None:
            receipt.update({
                "d2": tr_payload["d2_unprojected"],
                "proj": tr_payload["projected"],
                "mv": tr_payload["movement_frobenius"],
            })
        receipts.append(receipt)
        bank_digest_chain.append(bank.digest)
        expected_state = transition.state_after_sha256
        runtime._bump_trial_counters()
    _verify_direction_cell(
        direction_rows=direction_rows,
        binding=_binding_payload(budget, support_ids, initial, config),
        table_cell=table_cell,
        label=f"O2 {session.surface} M{budget} {session.session}",
    )
    per_trial_filtered = [p2filters.apply_output_filter_one_trial(item) for item in per_trial_raw]
    final_carrier = memory.state.carrier
    return {
        "row": "O2",
        "per_trial_raw": per_trial_raw,
        "per_trial_filtered": per_trial_filtered,
        "receipts": receipts,
        "initial": initial,
        "anchor": anchor,
        "final_carrier_digest": final_carrier.digest,
        "movements": movements,
        "d2_values": d2_values,
        "projected_count": projected_count,
        "bank_digest_chain": bank_digest_chain,
        "committed_rows": len(bank),
        "leakage_flags": dict(plan.LEAKAGE_FLAGS_BY_ROW["O2"]),
        "estimator": "support-anchored block refit + trust-region projection (always-commit)",
        "group_forward_count": 0,
        "true_construction_group_forwards_skipped": True,
    }


# ---------------------------------------------------------------------------
# Anchors against sealed receipts.
# ---------------------------------------------------------------------------


def _sealed_activity_cells(base: Path) -> dict[tuple[int, str], dict[str, Mapping[str, object]]]:
    """The sealed O0 anchor rows per (budget, surface).

    M4/M10 anchor against the sealed activity-only quick_v2 receipt (its cells
    cover exactly those budgets).  M30 anchors against the V8 matched score's
    SEALED STATIC cells -- the activity-only M30 rollout is bit-identical to the
    sealed static system there because the capacity-zero FIFO never mutates the
    activity stack, which is exactly the anchor law P2' stage-A used for A0@M30.
    """
    base = Path(base).absolute()
    cells: dict[tuple[int, str], dict[str, Mapping[str, object]]] = {}
    activity_only = _read_json(base / plan.ACTIVITY_ONLY_V2_RESULT_RELATIVE)
    for cell in activity_only["cells"]:
        budget, surface = int(cell["budget"]), str(cell["surface"])
        _require(budget in plan.LOW_BUDGETS, "sealed activity-only anchor covers only the low budgets")
        key = (budget, surface)
        _require(key not in cells, "sealed activity-only anchor cell topology drift")
        cells[key] = {str(row["session"]): row for row in cell["sessions"]}
    v8_score = _read_json(base / plan.V8_SCORE_RELATIVE)
    sealed_system = "sealed_cell_d_swa"
    for cell in v8_score["budget_summaries"]["30"]["cells"]:
        budget, surface = int(cell["budget"]), str(cell["surface"])
        if budget != 30 or str(cell["system"]) != sealed_system:
            continue
        key = (budget, surface)
        _require(key not in cells, "sealed M30 anchor cell topology drift")
        cells[key] = {str(row["session"]): row for row in cell["sessions"]}
    _require(
        set(cells) == {(budget, surface) for budget in plan.BUDGETS for surface in plan.SURFACES},
        "the sealed O0 anchor does not cover the full Stage-O grid",
    )
    return cells


def anchor_o0_vs_sealed(
    *, row: Mapping[str, object], sealed_row: Mapping[str, object], label: str,
) -> dict[str, object]:
    digest_match = str(row["prediction_sha256_raw"]) == str(sealed_row["prediction_sha256"])
    r2_gap = abs(float(row["house_raw_r2"]) - float(sealed_row["governing_r2"]))
    windows_match = int(row["n_windows"]) == int(sealed_row["valid_last_bin_count"])
    exact = digest_match and r2_gap == 0.0 and windows_match
    return {
        "label": label,
        "prediction_sha256_match": bool(digest_match),
        "house_raw_r2_gap": float(r2_gap),
        "n_windows_match": bool(windows_match),
        "exact_match": bool(exact),
        "replayed_house_raw_r2": float(row["house_raw_r2"]),
        "sealed_governing_r2": float(sealed_row["governing_r2"]),
    }


def anchor_o0_vs_filter_line_cache(
    *, base: Path, flat_raw: np.ndarray, surface: str, session: str, budget: int,
) -> dict[str, object]:
    manifest = _read_json(Path(base).absolute() / plan.FILTER_LINE_MANIFEST_RELATIVE)
    entry = manifest["entries"].get(f"cdm:{surface}:{session}:m{int(budget)}")
    if entry is None:
        return {"available": False, "label": f"{surface} {session} m{budget}"}
    path = Path(base).absolute() / "tfpd_exploration/cache/learnable_output_filter_v1" / str(entry["relative"])
    npz_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as data:
        cached_raw = np.ascontiguousarray(data["raw"], dtype=np.float32)
    bitwise = bool(np.array_equal(cached_raw, flat_raw)) and cached_raw.shape == flat_raw.shape
    return {
        "available": True,
        "label": f"{surface} {session} m{budget}",
        "manifest_sha256_match": bool(npz_sha == str(entry["sha256"])),
        "raw_bitwise_equal": bitwise,
        "n_windows": int(flat_raw.shape[0]),
        "manifest_n_windows": int(entry["n_windows"]),
    }


def p2prime_cross_reference(base: Path) -> dict[str, object]:
    """Read-only cross-reference: the sealed P2' coherent oracle upper bound."""
    stage_cop = _read_json(Path(base).absolute() / plan.P2PRIME_STAGE_COP_RELATIVE)
    matrix = stage_cop["matrix"]
    rows: dict[str, object] = {}
    for budget in plan.LOW_BUDGETS:
        external = matrix[f"m{budget}"]["external"]
        rows[f"m{budget}_external"] = {
            key: float(external[key]["mean_r2"])
            for key in ("A0", "A1", "O0", "O1", "O2")
            if key in external
        }
        rows[f"m{budget}_external"]["O2_minus_O0"] = float(external["O2"]["mean_r2"]) - float(external["O0"]["mean_r2"])
    return {
        "source": plan.P2PRIME_STAGE_COP_RELATIVE,
        "read_only": True,
        "role": plan.ANCHORS["p2prime_cross_reference"]["role"],
        "rows": rows,
        "m4_external_o2_minus_o0_expected_approx": 0.092,
        "note": (
            "the P2' O2 row is true-direction construction plus FUTURE-READING "
            "oracle accept (u_j over a horizon of future trials); Stage-O O2 is a "
            "deterministic always-commit law, so this row bounds its loss"
        ),
    }


# ---------------------------------------------------------------------------
# Session-row assembly and the causality receipt chain.
# ---------------------------------------------------------------------------


def verify_causality_receipts(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """``forward_state[i] == state_before[i] == state_after[i-1]`` for every trial."""
    previous_after: Optional[str] = None
    for receipt in receipts:
        forward_state = receipt.get("fs")
        state_before = receipt.get("sb")
        state_after = receipt.get("sa")
        if forward_state is None or state_before is None or state_after is None:
            return False
        if forward_state != state_before:
            return False
        if previous_after is not None and state_before != previous_after:
            return False
        previous_after = state_after
    return True


def _session_row(
    runtime: Any,
    *,
    row_id: str,
    budget: int,
    session: Any,
    targets: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    rollout: Mapping[str, Any],
    alpha_M: Optional[float],
    c_M: Optional[float],
) -> dict[str, Any]:
    matrix_r2, joined = runtime._matrix_r2(rollout["per_trial_filtered"], targets, masks)
    row: dict[str, Any] = {
        "schema": "support_anchored_t4_stage_o_row_session_v1",
        "row": row_id,
        "budget": int(budget),
        "surface": session.surface,
        "session": session.session,
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
        "target_optimizer_backward_update": 0,
        "target_backward_calls": 0,
        "target_parameter_update_calls": 0,
        "normalizer_update_calls": 0,
    }
    receipts = rollout.get("receipts") or rollout.get("per_trial_receipts")
    row["per_trial_receipts"] = list(receipts)
    row["causality_state_chain_verified"] = verify_causality_receipts(receipts)
    if "transitions" in rollout:
        transitions = rollout["transitions"]
        row["carrier_transitions_committed"] = sum(
            1 for item in transitions if item.carrier_transition_committed
        )
        row["activity_transitions_committed"] = sum(
            1 for item in transitions if item.activity_transition_committed
        )
        row["carrier_rejection_counts"] = _counts_by_reason(
            item.carrier_rejection_reason for item in transitions
        )
        row["final_carrier_sha256"] = transitions[-1].carrier_after_sha256 if transitions else None
    else:
        accepted = [item for item in receipts if item.get("acc")]
        row["carrier_transitions_committed"] = len(accepted)
        row["activity_transitions_committed"] = len(receipts)
        row["carrier_rejection_counts"] = _counts_by_reason(
            item.get("reason") for item in receipts if not item.get("acc")
        )
        row["final_carrier_sha256"] = rollout.get("final_carrier_digest")
    if alpha_M is not None:
        row["alpha_M"] = float(alpha_M)
    if c_M is not None:
        row["c_M"] = float(c_M)
    anchor = rollout.get("anchor")
    if anchor is not None:
        row["support_anchor"] = {
            "anchor_sha256": anchor.digest,
            "a0_b0_sha256": anchor.a0_b0_digest,
            "statistics_sha256": anchor.statistics.digest,
            "support_t4_sha256": cdm_core.array_digest(np.asarray(anchor.support_t4)),
            "coefficient_parity": dict(anchor.support_coefficient_parity),
        }
        movements = rollout.get("movements") or []
        row["movement"] = {
            "rho_M": float(plan.HYPERPARAMETERS["rho_M"]),
            "n_committed": len(movements),
            "median_frobenius": float(np.median(movements)) if movements else None,
            "max_frobenius": float(np.max(movements)) if movements else None,
            "projected_commit_count": int(rollout.get("projected_count") or 0),
            "median_d2_unprojected": float(np.median(rollout["d2_values"])) if rollout.get("d2_values") else None,
            "max_d2_unprojected": float(np.max(rollout["d2_values"])) if rollout.get("d2_values") else None,
        }
        row["committed_evidence_rows"] = int(rollout.get("committed_rows") or 0)
        row["bank_final_sha256"] = rollout["bank_digest_chain"][-1]
    return row


# ---------------------------------------------------------------------------
# The stage driver.
# ---------------------------------------------------------------------------


def run_stage_o_replay(
    base: Path,
    *,
    gpu_index: int,
    output_root: Path,
) -> Mapping[str, object]:
    """Calibrate c_M on within-6, then run the O0/O1/O2 grid once."""
    base = Path(base).absolute()
    output = Path(output_root)
    _require(not (output / "terminal.json").exists(), "the Stage-O terminal receipt already exists")
    _require((output / "attempt.json").exists(), "reserve the Stage-O attempt first")
    _require((output / "directions.json").exists(), "run the Stage-O directions stage first")
    attempt_sha = _verify_receipt(output / "attempt.json")
    directions_sha = _verify_receipt(output / "directions.json")
    attempt = _read_json(output / "attempt.json")
    directions_payload = _read_json(output / "directions.json")
    _require(
        str(directions_payload.get("attempt_sha256")) == attempt_sha,
        "the directions receipt was produced under a different attempt",
    )
    predecessors = plan.predecessor_sha256s(base)
    _require(
        predecessors == attempt["predecessor_sha256s"],
        "an immutable predecessor drifted between attempt and replay",
    )
    owned = plan.owned_sha256s(base)
    _require(owned == attempt["owned_sha256s"], "an owned Stage-O module drifted between attempt and replay")

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
        direction_cells = directions_payload["cells"]

        # -- c_M calibration: within-6 only, projection disabled -----------------
        calibration: dict[str, Any] = {"law": dict(plan.C_M_CALIBRATION)}
        calibration_chains: dict[str, list[str]] = {}
        for budget in plan.BUDGETS:
            d2_values: list[float] = []
            chains: dict[str, list[str]] = {}
            for key in p2physical._ordered_session_keys(runtime, "within"):
                session = state.sessions[key]
                cell = direction_cells[f"within:{session.session}:m{budget}"]
                rollout = rollout_o2(
                    runtime, session=session, budget=budget,
                    alpha_M=float(plan.HYPERPARAMETERS["alpha_M_primary"]),
                    c_M=None, table_cell=cell,
                )
                d2_values.extend(float(item) for item in rollout["d2_values"])
                chains[session.session] = list(rollout["bank_digest_chain"])
                _require(
                    (time.monotonic() - started) <= plan.HARD_TIMEOUT_SECONDS,
                    "the Stage-O hard timeout fired during calibration",
                )
            threshold = calibrate_c_M(d2_values)
            calibration[f"m{budget}"] = {
                "c_M": threshold,
                "n_d2_values": len(d2_values),
                "d2_min": float(np.min(d2_values)),
                "d2_median": float(np.median(d2_values)),
                "d2_max": float(np.max(d2_values)),
                "sessions": sorted(chains),
            }
            calibration_chains[f"m{budget}"] = chains  # type: ignore[assignment]
        calibration["selection_surface"] = "within-6 ONLY; external-15 never selects"
        c_M_by_budget = {int(key[1:]): float(value["c_M"]) for key, value in calibration.items()
                         if isinstance(value, dict) and "c_M" in value}

        # -- the governing grid ---------------------------------------------------
        matrix: dict[str, Any] = {}
        anchors_sealed: dict[str, Any] = {}
        anchors_filter_line: dict[str, Any] = {}
        causality_all = True
        movement_medians_by_session: dict[str, dict[str, float]] = {}
        sensitivity_means: dict[str, Any] = {}
        sealed_cells = _sealed_activity_cells(base)
        row_specs = [
            ("O0", None),
            ("O1", None),
            ("O2", float(plan.HYPERPARAMETERS["alpha_M_primary"])),
        ] + [(row, float(plan.ALPHA_BY_ROW[row])) for row in ("O2A0125", "O2A025", "O2A100")]
        for row_id, alpha_M in row_specs:
            for budget in plan.BUDGETS:
                c_M = c_M_by_budget.get(int(budget)) if alpha_M is not None else None
                for surface in plan.SURFACES:
                    session_rows: list[dict[str, Any]] = []
                    for key in p2physical._ordered_session_keys(runtime, surface):
                        session = state.sessions[key]
                        cell = direction_cells[f"{surface}:{session.session}:m{budget}"]
                        targets, masks, _sst = runtime._session_target_views(session, budget)
                        if row_id == "O0":
                            rollout = rollout_o0(runtime, session=session, budget=budget)
                            raw_row = rollout["rows"]["O0_raw"]
                            governing = rollout["rows"]["O0"]
                            row = _session_row(
                                runtime, row_id="O0", budget=budget, session=session,
                                targets=targets, masks=masks,
                                rollout={
                                    "per_trial_filtered": rollout["per_trial_filtered"],
                                    "per_trial_raw": rollout["per_trial_raw"],
                                    "initial": rollout["initial"],
                                    "leakage_flags": rollout["leakage_flags"],
                                    "estimator": "frozen support T4 (activity-only CDM)",
                                    "per_trial_receipts": rollout["per_trial_receipts"],
                                    "transitions": rollout["transitions"],
                                },
                                alpha_M=None, c_M=None,
                            )
                            _require(
                                row["house_raw_r2"] == float(raw_row["house_raw_r2"])
                                and row["prediction_sha256_raw"] == str(raw_row["prediction_sha256_raw"])
                                and row["matrix_r2"] == float(governing["matrix_r2"]),
                                "O0 row disagrees with the sealed activity-only row payload",
                            )
                            anchors_sealed[f"{session.session}:m{budget}:{surface}"] = anchor_o0_vs_sealed(
                                row=row, sealed_row=sealed_cells[(int(budget), surface)][session.session],
                                label=f"O0 vs sealed activity-only m{budget} {surface} {session.session}",
                            )
                            anchors_filter_line[f"{session.session}:m{budget}:{surface}"] = (
                                anchor_o0_vs_filter_line_cache(
                                    base=base, flat_raw=rollout["flat_raw"], surface=surface,
                                    session=session.session, budget=budget,
                                )
                            )
                        elif row_id == "O1":
                            rollout = rollout_o1(runtime, session=session, budget=budget, table_cell=cell)
                            row = _session_row(
                                runtime, row_id="O1", budget=budget, session=session,
                                targets=targets, masks=masks, rollout=rollout,
                                alpha_M=None, c_M=None,
                            )
                        else:
                            rollout = rollout_o2(
                                runtime, session=session, budget=budget, alpha_M=alpha_M,
                                c_M=c_M, table_cell=cell,
                            )
                            row = _session_row(
                                runtime, row_id=row_id, budget=budget, session=session,
                                targets=targets, masks=masks, rollout=rollout,
                                alpha_M=alpha_M, c_M=c_M,
                            )
                            if row_id == "O2" and surface == "within":
                                chain = calibration_chains[f"m{budget}"][session.session]
                                _require(
                                    rollout["bank_digest_chain"] == chain,
                                    f"O2 bank-digest chain differs from the calibration chain "
                                    f"for {session.session} m{budget}",
                                )
                            if row_id == "O2":
                                movement_medians_by_session.setdefault(session.session, {})[f"m{budget}"] = (
                                    row["movement"]["median_frobenius"]
                                )
                        causality_all = causality_all and bool(row["causality_state_chain_verified"])
                        _require(row["causality_state_chain_verified"],
                                 f"causality receipt chain broken for {row_id} {session.session} m{budget}")
                        session_rows.append(row)
                        _require(
                            (time.monotonic() - started) <= plan.HARD_TIMEOUT_SECONDS,
                            "the Stage-O hard timeout fired during the governing grid",
                        )
                    per_session = {
                        str(item["session"]): float(item["matrix_r2"]) for item in session_rows
                    }
                    entry = {
                        "mean_r2": float(sum(per_session.values()) / len(per_session)),
                        "per_session": per_session,
                        # §9 receipts: every row (the non-governing alpha
                        # sensitivities included) publishes the full per-trial
                        # state/block/movement digest records.
                        "sessions": [dict(item) for item in session_rows],
                        "full_per_trial_state_receipts": True,
                    }
                    matrix.setdefault(f"m{budget}", {}).setdefault(surface, {})[row_id] = entry
                    if row_id in ("O2A0125", "O2A025", "O2A100"):
                        sensitivity_means[f"{row_id}_m{budget}_{surface}"] = entry["mean_r2"]

        model_digest_after = arm.state_sha256(runtime._require_state().model)
        _require(
            model_digest_before == model_digest_after,
            "the sealed Cell-D model state mutated during the Stage-O replay",
        )
        movement_by_budget: dict[str, object] = {}
        for budget in plan.BUDGETS:
            values = [
                movement_medians_by_session[session][f"m{budget}"]
                for session in sorted(movement_medians_by_session)
                if movement_medians_by_session[session].get(f"m{budget}") is not None
            ]
            _require(bool(values), f"no committed movement was recorded at M{budget}")
            movement_by_budget[f"m{budget}"] = float(np.median(values))
        resources = dict(runtime._resources())
    finally:
        runtime.close()

    payload = {
        "schema": f"{plan.SCHEMA}_replay_v1",
        "stage": "o0_o1_o2_oracle_headroom_grid",
        "status": "REPLAY_COMPLETE",
        "attempt_sha256": attempt_sha,
        "directions_sha256": directions_sha,
        "identity_sha256": meta["identity_sha256"],
        "environment": environment,
        "gpu_index": int(gpu_index),
        "budgets": list(plan.BUDGETS),
        "surfaces": list(plan.SURFACES),
        "rows": list(plan.ROW_ORDER),
        "o2_commit_law": dict(plan.O2_COMMIT_LAW),
        "hyperparameters": dict(plan.HYPERPARAMETERS),
        "c_M_calibration": calibration,
        "matrix": matrix,
        "sensitivity_means_non_governing": sensitivity_means,
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
            "p2prime_cross_reference": p2prime_cross_reference(base),
        },
        "movement_by_budget_medians": movement_by_budget,
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
        "the O0 anchor failed at least one sealed activity-only session row",
    )
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
    safety = {
        "zero_target_updates": bool(
            replay_payload.get("target_optimizer_backward_update") == 0
            and replay_payload.get("target_backward_calls") == 0
            and replay_payload.get("target_parameter_update_calls") == 0
            and replay_payload.get("normalizer_update_calls") == 0
            and replay_payload.get("model_state_digest_unchanged") is True
        ),
        "o0_bit_anchor": bool(replay_payload["anchors"]["sealed_activity_only_all_exact"]),
        "causality_state_chains": bool(replay_payload["causality_state_chains_all_rows"]),
        "within_regression_bound": True,  # evaluated by the gate from the matrix
        "other_low_budget_bound": True,   # evaluated by the gate from the matrix
    }
    gate = gates_module.evaluate_oracle_gate(view, safety=safety)
    interpretation = gates_module.interpretation_rows(view)
    ordering = gates_module.movement_ordering(replay_payload["movement_by_budget_medians"])
    paired: dict[str, object] = {}
    for budget in plan.BUDGETS:
        for surface in plan.SURFACES:
            rows = view[f"m{budget}"][surface]
            for candidate, baseline in (("O2", "O0"), ("O1", "O0"), ("O2", "O1")):
                paired[f"{candidate}_minus_{baseline}_m{budget}_{surface}"] = paired_session_stats([
                    float(rows[candidate]["per_session"][session])
                    - float(rows[baseline]["per_session"][session])
                    for session in sorted(rows[candidate]["per_session"])
                ])
    return {
        "gate": gate,
        "interpretation_rows": interpretation,
        "movement_ordering": ordering,
        "paired_contrasts": paired,
        "sensitivity_means_non_governing": replay_payload.get("sensitivity_means_non_governing"),
        "p2prime_cross_reference": replay_payload["anchors"]["p2prime_cross_reference"],
        "c_M_calibration": replay_payload.get("c_M_calibration"),
    }
