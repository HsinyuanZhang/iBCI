"""Policy layer of the P2' oracle-policy decomposition matrix.

This module owns everything that is NOT the frozen CDM state machine:

* construction of the complementary-prediction views fed to ``observe``
  (raw / smoothed / TRUE completed-trial direction);
* the oracle/precision veto that turns an accepted pending proposal into the
  canonical carrier-rejected pending (the exact wrapper pattern of the frozen
  Precision V2 route);
* the counterfactual utility arithmetic of section 10.7 (session-SST
  normalized SSE, horizon isolation, accept rule);
* the kill-criterion verdict functions of section 10.12 with the exact
  boundary semantics pre-registered in ``plan.KILL_CRITERIA``.

The state machine itself (validation, pseudo integration, B8 proposal gates,
sufficient-statistics updates, commit law) is imported from
``src.causal_dual_memory_cell_d_v1.core`` and is never reimplemented here.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.causal_dual_memory_cell_d_v1 import core

from . import filters, plan


class P2PrimePolicyError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P2PrimePolicyError(message)


CONSTRUCTIONS = ("raw", "smoothed_causal", "smoothed_zero_phase", "true")


# ---------------------------------------------------------------------------
# Pseudo-direction construction: transform the complementary prediction views.
# ---------------------------------------------------------------------------


def build_construction_predictions(
    *,
    group_predictions: Sequence[Any],
    construction: str,
    behavior_rows: Any = None,
    behavior_mean: Sequence[float] = (),
    behavior_std: Sequence[float] = (),
    alpha: float = filters.ALPHA,
) -> Tuple[Tuple[Any, ...], Mapping[str, object]]:
    """Return the complementary-prediction views for one construction.

    ``raw`` and the two smoothed kernels transform each group's velocity
    trajectory in place (same validity evidence object, so the frozen
    ``observe`` law sees an identical capability type).  ``true`` replaces the
    predicted trajectory with the TRUE completed-trial physical velocity from
    the sealed behavior rows; the same displacement/movement-mask/snap
    pipeline of the state machine then produces the true direction.
    """
    _require(construction in CONSTRUCTIONS, f"unknown pseudo construction: {construction}")
    rows = tuple(group_predictions)
    _require(len(rows) == core.GROUP_COUNT, "construction needs exactly four group predictions")
    meta: dict[str, object] = {"construction": construction, "alpha": alpha, "groups": len(rows)}
    if construction == "true":
        _require(behavior_rows is not None, "true construction requires the trial's behavior rows")
        restored, padded = filters.true_physical_velocity_one_trial(
            behavior_rows, behavior_mean=behavior_mean, behavior_std=behavior_std,
        )
        meta["true_construction_padded_rows"] = int(np.sum(padded))
        return tuple(
            core.CompletedVelocityPrediction(restored, item.validity) for item in rows
        ), meta
    transformed = tuple(
        core.CompletedVelocityPrediction(
            filters.smooth_velocity_trajectory_one_trial(
                item.velocity, construction=construction, alpha=alpha,
            ),
            item.validity,
        )
        for item in rows
    )
    return transformed, meta


def true_direction_payload(
    restored_velocity: np.ndarray, validity: Any, config: Any,
) -> Mapping[str, object]:
    """Reference truth: the frozen direction pipeline on the true trajectory.

    The sub-study's direction-error table compares each construction's
    pseudo-direction payload against THIS object, computed by the exact
    ``pseudo_direction_from_velocity`` call the state machine makes, under
    the session carrier's own frozen config.
    """
    pseudo = core.pseudo_direction_from_velocity(restored_velocity, validity.valid_mask, config=config)
    return pseudo.payload()


# ---------------------------------------------------------------------------
# Oracle veto (section 10.11 rows O0/O1/O2; M30 no-op uses the same object).
# ---------------------------------------------------------------------------


def oracle_rejected_pending(
    memory: core.IndependentActivityCausalDualMemory,
    pending: core.IndependentActivityPendingTrialUpdate,
) -> core.IndependentActivityPendingTrialUpdate:
    """Canonical carrier rejection that preserves the activity transition.

    Follows the exact reconstruction law of the frozen Precision V2 wrapper
    (``_precision_rejected_pending``): the activity candidate is retained, the
    carrier is restored to the current exact state, and only the typed
    rejection reason changes.  The reason is the frozen
    ``INSUFFICIENT_EVIDENCE`` value -- the state machine's own no-evidence
    fallback -- because P2' deliberately adds no new enum member to the
    frozen core.
    """
    _require(isinstance(memory, core.IndependentActivityCausalDualMemory),
             "oracle veto requires the independent activity memory")
    _require(pending.activity_transition_ready and pending.carrier_transition_accepted,
             "oracle veto may reject only an accepted carrier proposal")
    _require(isinstance(pending.candidate_state, core.DualMemoryState),
             "oracle veto accepted proposal lacks a candidate state")
    candidate = core.DualMemoryState(
        activity=pending.candidate_state.activity,
        carrier=memory.state.carrier,
        committed_query_trials=pending.candidate_state.committed_query_trials,
    )
    return core.IndependentActivityPendingTrialUpdate(
        base_state_digest=pending.base_state_digest,
        activity_transition_ready=True,
        activity_fifo_changed=pending.activity_fifo_changed,
        carrier_transition_accepted=False,
        activity_rejection_reason=None,
        carrier_rejection_reason=core.UpdateRejectionReason.INSUFFICIENT_EVIDENCE,
        pseudo_directions=pending.pseudo_directions,
        scalar_rates=pending.scalar_rates,
        carrier_proposal=pending.carrier_proposal,
        candidate_state=candidate,
        fallback=pending.fallback,
        completed_trial_evidence=pending.completed_trial_evidence,
    )


def branch_states_for_counterfactual(
    memory: core.IndependentActivityCausalDualMemory,
    pending: core.IndependentActivityPendingTrialUpdate,
    *,
    independent_activity: "core.ActivityMemory | None" = None,
) -> Tuple[core.DualMemoryState, core.DualMemoryState, Mapping[str, object]]:
    """The same-parent accept/reject branch states of one decision.

    Both branches leave the identical post-activity state; they differ ONLY
    in the carrier.  When the caller supplies an activity memory rebuilt
    independently through the frozen law (``after_completed_trial``), both
    branches are additionally required to match it, which makes the parity
    check non-tautological.
    """
    _require(pending.activity_transition_ready, "counterfactual requires a ready activity transition")
    _require(isinstance(pending.candidate_state, core.DualMemoryState),
             "counterfactual requires a candidate state")
    accept = pending.candidate_state
    if pending.carrier_transition_accepted:
        reject = core.DualMemoryState(
            activity=accept.activity,
            carrier=memory.state.carrier,
            committed_query_trials=accept.committed_query_trials,
        )
    else:
        # The proposal already failed the frozen B8 checks: both branches are
        # the activity-only candidate and no counterfactual difference exists.
        reject = accept
    accept_stack = accept.activity.stack()
    reject_stack = reject.activity.stack()
    independent_digest = (
        independent_activity.digest if independent_activity is not None else accept.activity.digest
    )
    parity = {
        "activity_digest_equal": accept.activity.digest == reject.activity.digest,
        "activity_stack_bitwise_equal": bool(np.array_equal(accept_stack, reject_stack)),
        "activity_independent_digest_match": accept.activity.digest == independent_digest,
        "committed_query_trials": int(accept.committed_query_trials),
        "carrier_accept_sha256": accept.carrier.digest,
        "carrier_reject_sha256": reject.carrier.digest,
        "carrier_reject_matches_parent": reject.carrier.digest == memory.state.carrier.digest,
        "carrier_branches_differ": accept.carrier.digest != reject.carrier.digest,
        "proposal_accepted_by_b8": bool(pending.carrier_transition_accepted),
    }
    _require(
        parity["activity_digest_equal"]
        and parity["activity_stack_bitwise_equal"]
        and parity["activity_independent_digest_match"]
        and parity["carrier_reject_matches_parent"],
        "counterfactual branches differ outside the carrier commit",
    )
    return accept, reject, parity


# ---------------------------------------------------------------------------
# Counterfactual utility (section 10.7).
# ---------------------------------------------------------------------------


def session_sst(targets: Sequence[np.ndarray]) -> float:
    """SST over all valid query windows of one session (branch-independent)."""
    stacked = np.concatenate([np.asarray(item, dtype=np.float64) for item in targets], axis=0)
    _require(stacked.ndim == 2 and stacked.shape[0] >= 1, "session SST needs [W, 2] targets")
    mean = stacked.mean(axis=0)
    return float(np.sum((stacked - mean[None, :]) ** 2))


def trial_nsse(
    prediction: np.ndarray, target: np.ndarray, *, sst: float,
) -> float:
    """Session-SST-normalized SSE of one future trial (float64 path)."""
    values = np.asarray(prediction, dtype=np.float64)
    labels = np.asarray(target, dtype=np.float64)
    _require(values.shape == labels.shape and values.ndim == 2,
             "nSSE prediction/target shape drift")
    _require(sst > 0.0 and np.isfinite(sst), "session SST must be positive and finite")
    sse = float(np.sum((values - labels) ** 2))
    return sse / sst


def horizon_utility(
    *,
    reject_losses: Sequence[float],
    accept_losses: Sequence[float],
) -> Mapping[str, object]:
    """u_j = mean_t [nSSE_t(reject) - nSSE_t(accept)] over the fixed horizon.

    Horizon isolation is structural: the caller supplies losses for trials
    j+1..j+H only.  This function never sees trial j.
    """
    reject = [float(item) for item in reject_losses]
    accept = [float(item) for item in accept_losses]
    _require(len(reject) == len(accept) and len(reject) >= 1,
             "horizon branch losses must be paired and nonempty")
    deltas = [left - right for left, right in zip(reject, accept)]
    return {
        "u_j": float(sum(deltas) / len(deltas)),
        "horizon_trials": len(deltas),
        "per_trial_deltas": deltas,
    }


def oracle_decision(utility: Mapping[str, object]) -> bool:
    """Coherent greedy oracle: commit accept iff u_j > 0 (ties reject)."""
    return float(utility["u_j"]) > 0.0


# ---------------------------------------------------------------------------
# Kill-criterion verdicts (section 10.12; boundary semantics pre-registered).
# ---------------------------------------------------------------------------


def _mean_delta(
    matrix: Mapping[str, Mapping[str, Mapping[str, float]]],
    row: str,
    baseline: str,
    *,
    budget: int,
    surface: str,
) -> Optional[float]:
    rows = matrix.get(f"m{budget}", {}).get(surface, {})
    if row not in rows or baseline not in rows:
        return None
    return float(rows[row]["mean_r2"]) - float(rows[baseline]["mean_r2"])


#: Pre-registered numerical guard: the kill boundaries are compared after
#: widening by this epsilon so that two arithmetic paths computing the same
#: real number (e.g. ``(0.2 + 0.02) - 0.2``) cannot flip a verdict.
KILL_BOUNDARY_EPSILON = 1.0e-12


def kill_criterion_verdicts(matrix: Mapping[str, object]) -> Mapping[str, object]:
    """Evaluate the five pre-registered kill criteria on the matrix summary.

    ``matrix`` is ``{"m<budget>": {"<surface>": {"<row>": {"mean_r2": x}}}}``
    plus the coherent-oracle rows keyed by row id.  Boundaries follow the
    pre-registration exactly: KC1 fires on strict ``< +0.02``; KC2 fires only
    when BOTH deployment budgets' external deltas are strictly below
    ``+0.005``; KC3 requires KC1 to fire and O2 - A1 to reach ``+0.02``.
    Every strict comparison carries ``KILL_BOUNDARY_EPSILON`` so a boundary
    value itself never fires.
    """
    m4_external_o1_a1 = _mean_delta(matrix, "O1", "A1", budget=4, surface="external")
    m10_external_o1_o0 = _mean_delta(matrix, "O1", "O0", budget=10, surface="external")
    m4_external_o1_o0 = _mean_delta(matrix, "O1", "O0", budget=4, surface="external")
    m4_external_o2_a1 = _mean_delta(matrix, "O2", "A1", budget=4, surface="external")

    kc1_value = None if m4_external_o1_a1 is None else m4_external_o1_a1
    kc1_fired = kc1_value is not None and kc1_value < 0.02 - KILL_BOUNDARY_EPSILON
    kc2_fired = (
        m4_external_o1_o0 is not None
        and m10_external_o1_o0 is not None
        and m4_external_o1_o0 < 0.005 - KILL_BOUNDARY_EPSILON
        and m10_external_o1_o0 < 0.005 - KILL_BOUNDARY_EPSILON
    )
    kc3_fired = (
        bool(kc1_fired)
        and m4_external_o2_a1 is not None
        and m4_external_o2_a1 >= 0.02 - KILL_BOUNDARY_EPSILON
    )

    if kc1_fired:
        verdict = "STOP_LEARNED_GATE"
    elif kc3_fired:
        verdict = "PSEUDO_BIAS_BOTTLENECK"
    elif kc2_fired:
        verdict = "SMOOTHING_NO_CARRIER_VALUE_KEEP_OUTPUT_FILTER_ONLY"
    else:
        verdict = "PASS_KC1_KC2_KC3_EXTERNAL_MATCHED_SCORE_AUTHORIZED"

    return {
        "KC1_STOP_LEARNED_GATE": {
            "fired": bool(kc1_fired),
            "expression": plan.KILL_CRITERIA["KC1_STOP_LEARNED_GATE"]["expression"],
            "O1_minus_A1_m4_external": kc1_value,
            "threshold": 0.02,
            "boundary": "fires on strict < +0.02",
        },
        "KC2_SMOOTHING_NO_CARRIER_VALUE": {
            "fired": bool(kc2_fired),
            "expression": plan.KILL_CRITERIA["KC2_SMOOTHING_NO_CARRIER_VALUE"]["expression"],
            "O1_minus_O0_m4_external": m4_external_o1_o0,
            "O1_minus_O0_m10_external": m10_external_o1_o0,
            "threshold": 0.005,
            "boundary": "fires only when BOTH deployment budgets are strictly below +0.005",
        },
        "KC3_PSEUDO_BIAS_BOTTLENECK": {
            "fired": bool(kc3_fired),
            "expression": plan.KILL_CRITERIA["KC3_PSEUDO_BIAS_BOTTLENECK"]["expression"],
            "O2_minus_A1_m4_external": m4_external_o2_a1,
            "requires_kc1": bool(kc1_fired),
        },
        "KC4_HEADROOM_NOT_IDENTIFIABLE": {
            "fired": False,
            "status": "PENDING_NOT_TESTABLE_IN_P2PRIME",
            "note": "no learned policy is fitted in P2'; adjudicated in P4 per section 10.12",
        },
        "KC5_EXTERNAL_MATCHED_SCORE_AUTHORIZATION": {
            "fired": not (kc1_fired or kc2_fired or kc3_fired),
            "expression": plan.KILL_CRITERIA["KC5_EXTERNAL_MATCHED_SCORE_AUTHORIZATION"]["expression"],
        },
        "verdict": verdict,
    }
