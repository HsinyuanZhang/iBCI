"""Static contract and pre-registrations for the P2' oracle-policy matrix.

Everything that could bias the decomposition is frozen HERE, in module bytes
that the attempt receipt pins by SHA-256 *before* any evaluation row runs:

* the fixed output filter (causal EMA, alpha = 0.25, trial-local with
  trial-boundary reset) shared by every row except the raw anchor A0;
* the pseudo-direction constructions compared by the sub-study, and the rule
  that picks the winner for rows C1/O1;
* the counterfactual utility definition (normalized-SSE delta on a fixed
  future horizon, session-SST normalizer, trial j excluded from its own
  utility);
* the two oracle levels (NONCOHERENT one-step-switch ceiling evaluated
  without commit; COHERENT greedy oracle that commits and advances);
* the kill criteria of section 10.12 with their exact boundary semantics.

Disclosures that this module pins:

* P1 (continuity_probe_v1) selected K/alpha arms on the external surface.
  That selection is exploratory and is NOT used here.  alpha = 0.25 is fixed
  a priori as the mid档 of the EMA family, applied trial-locally -- a
  deliberately different object from P1's concatenated-window-stream kernel.
* Rows O0/O1/O2 and the sub-study direction-error table read target labels
  through the utility/true-direction machinery and are leakage diagnostics,
  never deployment policies.
* The beta* distance is not used as a label anywhere in this route.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from src.causal_dual_memory_cell_d_score_v1 import plan as v1plan


CELL = "LEARNED_GATE_P2PRIME_ORACLE_POLICY_DECOMPOSITION_V1"
SCHEMA = "learned_gate_p2prime_v1"
DESIGN_RELATIVE = "tfpd_exploration/docs/DESIGN_LEARNED_GATE_SMOOTHED_PSEUDOLABEL_20260828.md"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/learned_gate_p2prime_v1"
SMOKE_RESULT_ROOT_RELATIVE = "tfpd_exploration/results/learned_gate_p2prime_v1_smoke"

SEALED_CELL_D_SWA_SHA256 = v1plan.SEALED_CELL_D_SWA_SHA256
ACTIVITY_ONLY_V2_RESULT_RELATIVE = (
    "tfpd_exploration/results/causal_dual_memory_cell_d_activity_only_quick_v2"
)
V8_SCORE_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_matched_score_v8/score.json"
PRECISION_V2_SCORE_RELATIVE = (
    "tfpd_exploration/results/precision_aware_causal_dual_memory_cell_d_matched_score_v2/score.json"
)

EXACT_DATA_ROOT_ENV = {
    "SUBC_DATA_ROOT": "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C",
    "SUBM_DATA_ROOT": "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M",
}

# ---------------------------------------------------------------------------
# Pre-registration 1: the fixed output filter.
# ---------------------------------------------------------------------------

OUTPUT_FILTER = {
    "family": "causal_ema",
    "alpha": 0.25,
    "scope": "per_completed_query_trial_window_sequence",
    "trial_boundary_reset": True,
    "applied_to": "governing_bin predictions, all windows of the trial (validity mask applied after filtering)",
    "dtype": "float64",
    "rows_using_it": ["A1", "C0", "C1", "O0", "O1", "O2", "P", "NONCOHERENT", "M30 diagnostic"],
    "rows_not_using_it": ["A0 (raw anchor)"],
    "p1_k_on_external_selection_used": False,
    "disclosure": (
        "P1's external-best K/alpha arm selection is exploratory and is not consumed here. "
        "alpha=0.25 is fixed a priori; the kernel runs within each trial and resets at trial "
        "boundaries, so no prediction of trial k+1 ever reads a prediction of trial k."
    ),
}

# ---------------------------------------------------------------------------
# Pre-registration 2: pseudo-direction constructions (sub-study of section 10.4).
# ---------------------------------------------------------------------------

PSEUDO_CONSTRUCTIONS = {
    "raw": {
        "label": "(a) raw completed-trial trajectory",
        "kernel": "identity",
    },
    "smoothed_causal": {
        "label": "(b) trial-local causal EMA",
        "kernel": "causal_ema",
        "alpha": 0.25,
        "reset": "per trial per complementary group",
    },
    "smoothed_zero_phase": {
        "label": "(c) trial-complete zero-phase two-pass EMA",
        "kernel": "zero_phase_two_pass_ema",
        "alpha": 0.25,
        "padding": "none",
        "legality": "whole trial is past at update time (section 10.4)",
        "reset": "per trial per complementary group",
    },
    "true": {
        "label": "TRUE completed-trial direction (oracle reference, leakage diagnostic)",
        "kernel": "true_velocity_from_behavior_restored_to_physical_units",
        "movement_mask": "same validity mask as the pseudo pipeline",
        "invalid_target_bins_rule": "zero-filled true velocity (recorded when it triggers)",
    },
}

SUB_STUDY_WINNER_RULE = {
    "eligible": ["smoothed_causal", "smoothed_zero_phase"],
    "primary": "larger mean per-decision utility u_j on M4 external (deployment primary)",
    "tie_condition": "bootstrap 95% CI of the paired difference contains 0",
    "tiebreak_1": "lower mean pseudo-direction circular error vs TRUE direction on M4 external",
    "tiebreak_2": "prefer smoothed_causal (legal in both the update-time and output-time clocks)",
    "raw_can_win": False,
    "disclosure": (
        "If both smoothed constructions show mean u_j at or below the raw construction, the "
        "sub-study records that fact; C1/O1 still run with the rule's winner so the "
        "pre-specified matrix rows exist, and kill criterion 2 adjudicates the value."
    ),
}

# ---------------------------------------------------------------------------
# Pre-registration 3: counterfactual utility (section 10.7).
# ---------------------------------------------------------------------------

UTILITY = {
    "primary_horizon_H": 5,
    "sensitivity_horizon_H": 10,
    "sensitivity_scope": "O1 coherent oracle, budget 4, both surfaces",
    "definition": (
        "u_j = mean over t in {j+1..j+min(H, remaining)} of "
        "[nSSE_t(reject) - nSSE_t(accept)]"
    ),
    "nsse": (
        "nSSE_t(branch) = sum over valid windows of trial t of "
        "||filtered_prediction - target||^2 divided by SST_session"
    ),
    "sst_session": (
        "sum over ALL valid query windows of the session of ||target - session_target_mean||^2; "
        "computed once per session/budget, identical for both branches and all rows"
    ),
    "branch_law": (
        "accept and reject branches leave the SAME parent state after trial j completes, share "
        "the exact activity trajectory/neural inputs/chronology/output filter, and differ ONLY "
        "in the carrier commit at trial j; no further carrier action is taken inside a horizon"
    ),
    "self_exclusion": "trial j never appears in its own utility window (t starts at j+1)",
    "accept_rule": "coherent oracle commits accept iff u_j > 0 (ties and u_j == 0 reject)",
    "secondary_diagnostic": (
        "constrained/Mahalanobis beta* distance is NOT computed as a label; the frozen departure "
        "ratio of the B8 proposal check is the only mechanism diagnostic retained"
    ),
}

# ---------------------------------------------------------------------------
# Pre-registration 4: oracle levels (section 10.8).
# ---------------------------------------------------------------------------

ORACLE_LEVELS = {
    "NONCOHERENT_ONE_STEP_SWITCH_CEILING": {
        "law": (
            "at each trial j, from the same parent state, evaluate both actions' predictions "
            "over j+1..j+H and take the better one per future trial; commit neither"
        ),
        "parent_trajectory": "reject-all line (the A1 activity trajectory, carrier never commits)",
        "session_score": (
            "each trial t takes the better branch prediction of its most recent parent decision "
            "j = t-1 (one-step switch); the first query trial uses the reject branch"
        ),
        "status": "loose diagnostic ceiling, not an executable policy",
    },
    "NONCOHERENT_LOOSE_MIN_OVER_PARENTS": {
        "law": (
            "secondary variant: each trial t additionally may use accept_j candidates from any "
            "covering parent j in [t-H, t-1]; the minimum-loss candidate wins"
        ),
        "status": "looser diagnostic only, never governs",
    },
    "COHERENT_GREEDY_ORACLE": {
        "law": (
            "compute u_j from the fixed-horizon counterfactual, commit the positive-utility "
            "action, actually advance state, continue"
        ),
        "status": "executable greedy policy upper reference, not a global optimum bound",
    },
    "beam_search": "not implemented (trial counts do not require it; a beam would be approximate anyway)",
}

# ---------------------------------------------------------------------------
# Pre-registration 5: kill criteria (section 10.12) -- exact boundary semantics.
# ---------------------------------------------------------------------------

KILL_CRITERIA = {
    "KC1_STOP_LEARNED_GATE": {
        "expression": "COHERENT(O1) - A1 < +0.02 at M4 external (session-mean paired delta)",
        "boundary": "exactly +0.02 does NOT fire (strict less-than with a 1e-12 numerical guard)",
        "action": "stop the learned gate line",
    },
    "KC2_SMOOTHING_NO_CARRIER_VALUE": {
        "expression": "O1 - O0 < +0.005 at BOTH M4 external and M10 external",
        "boundary": "either surface reaching +0.005 or more does NOT fire (same 1e-12 guard)",
        "action": "pseudo-label smoothing adds no carrier-action value; keep the output filter only",
    },
    "KC3_PSEUDO_BIAS_BOTTLENECK": {
        "expression": "KC1 fires AND COHERENT(O2) - A1 >= +0.02 at M4 external",
        "boundary": "O2 delta of exactly +0.02 counts as high (same 1e-12 guard)",
        "action": "bottleneck is circular pseudo-label bias; fix the pseudo direction, fit no gate",
    },
    "KC4_HEADROOM_NOT_IDENTIFIABLE": {
        "expression": (
            "oracle high but realized learned-policy utility near zero under session-grouped CV"
        ),
        "status": "NOT TESTABLE in P2' (no learned policy is fitted here); recorded as PENDING for P4",
    },
    "KC5_EXTERNAL_MATCHED_SCORE_AUTHORIZATION": {
        "expression": "KC1, KC2, KC3 all pass",
        "action": "external matched score authorized as a separate work order",
    },
}

# ---------------------------------------------------------------------------
# Pre-registration 6: the compose/overlap reading for A1 - A0.
# ---------------------------------------------------------------------------

COMPOSE_READING = {
    "question": (
        "does the fixed causal output filter compose with the growing activity memory, or has "
        "the activity memory already consumed the removable output jitter?"
    ),
    "statistic": "A1 - A0 paired session delta per budget x surface",
    "static_comparators_causal_ema_alpha_025_external": {"4": 0.0081, "10": 0.0163, "30": 0.0394},
    "static_comparators_note": (
        "continuity_probe_v1 causal_window_exp_a0.25 external deltas on the static sealed "
        "system; the review's quoted +0.0088/+0.0191/+0.0365 are the external-best arms of the "
        "same probe (K4 mean / alpha 0.25 EMA), reported for context only"
    ),
    "reading_rule": (
        "if A1-A0 on the CDM is comparable to the static deltas the axes compose; if much "
        "smaller the activity memory already removed the jitter; recorded either way"
    ),
}

# ---------------------------------------------------------------------------
# Row topology of the section 10.11 matrix.
# ---------------------------------------------------------------------------

ROWS = ("A0", "A1", "C0", "C1", "O0", "O1", "O2", "P")
ROW_SPECS = {
    "A0": {"pseudo": "none", "carrier_action": "reject all", "output_filter": "raw"},
    "A1": {"pseudo": "none", "carrier_action": "reject all", "output_filter": "fixed"},
    "C0": {"pseudo": "raw", "carrier_action": "always accept / frozen B8 checks", "output_filter": "fixed"},
    "C1": {"pseudo": "sub-study winner", "carrier_action": "always accept / frozen B8 checks", "output_filter": "fixed"},
    "O0": {"pseudo": "raw", "carrier_action": "oracle accept/reject", "output_filter": "fixed"},
    "O1": {"pseudo": "sub-study winner", "carrier_action": "oracle accept/reject", "output_filter": "fixed"},
    "O2": {"pseudo": "true direction", "carrier_action": "oracle accept/reject", "output_filter": "fixed"},
    "P": {"pseudo": "raw", "carrier_action": "frozen Precision V2 rule", "output_filter": "fixed"},
}
BUDGETS = (4, 10, 30)
DEPLOYMENT_BUDGETS = (4, 10)
SURFACES = ("within", "external")
M30_LAW = {
    "carrier": "no-op for every row; oracle rows keep an oracle-only diagnostic",
    "bitwise_check": "carrier state digest identical before/after every trial and equal to the initial digest",
}

CONTRASTS = (
    ("O1", "A1", "matched learned-gate opportunity after holding output smoothing fixed"),
    ("O1", "O0", "value of pseudo-label smoothing inside the gate problem"),
    ("O2", "O1", "remaining pseudo-label bias/quality gap"),
    ("C1", "C0", "effect of smoothing under always-accept dynamics"),
    ("P", "A1", "hand-gate comparator"),
    ("A1", "A0", "output-filter axis (compose/overlap reading)"),
)

OWNED_PATHS = (
    DESIGN_RELATIVE,
    "tfpd_exploration/src/learned_gate_p2prime_v1/__init__.py",
    "tfpd_exploration/src/learned_gate_p2prime_v1/plan.py",
    "tfpd_exploration/src/learned_gate_p2prime_v1/filters.py",
    "tfpd_exploration/src/learned_gate_p2prime_v1/policy.py",
    "tfpd_exploration/src/learned_gate_p2prime_v1/physical.py",
    "tfpd_exploration/scripts/run_learned_gate_p2prime_v1.py",
    "tfpd_exploration/tests/test_learned_gate_p2prime_v1.py",
)

SMOKE_PARAMETERS = {
    "budgets": (4,),
    "surfaces": ("within", "external"),
    "sessions_per_surface": 1,
    "horizon_H": 2,
    "c1_o1_construction": "smoothed_causal",
    "scope": "non_governing_pipeline_smoke",
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def owned_sha256s(root: Path) -> dict[str, str]:
    base = Path(root).absolute()
    return {
        relative: hashlib.sha256((base / relative).read_bytes()).hexdigest()
        for relative in OWNED_PATHS
    }


def pre_registration_payload() -> dict[str, object]:
    return {
        "output_filter": dict(OUTPUT_FILTER),
        "pseudo_constructions": {k: dict(v) for k, v in PSEUDO_CONSTRUCTIONS.items()},
        "sub_study_winner_rule": dict(SUB_STUDY_WINNER_RULE),
        "utility": dict(UTILITY),
        "oracle_levels": {
            k: (dict(v) if isinstance(v, dict) else v) for k, v in ORACLE_LEVELS.items()
        },
        "kill_criteria": {k: dict(v) for k, v in KILL_CRITERIA.items()},
        "compose_reading": dict(COMPOSE_READING),
        "row_specs": {k: dict(v) for k, v in ROW_SPECS.items()},
        "m30_law": dict(M30_LAW),
        "contrasts": [list(item) for item in CONTRASTS],
    }


def validate_pre_registration(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    if result.get("output_filter", {}).get("alpha") != 0.25:
        raise ValueError("P2' output-filter alpha pre-registration drift")
    if result.get("utility", {}).get("primary_horizon_H") != 5:
        raise ValueError("P2' utility horizon pre-registration drift")
    if tuple(result.get("row_specs", ())) != ROWS:
        raise ValueError("P2' row topology pre-registration drift")
    if set(result.get("kill_criteria", ())) != set(KILL_CRITERIA):
        raise ValueError("P2' kill-criterion pre-registration drift")
    return result


def dry_plan() -> dict[str, object]:
    return {
        "schema": f"{SCHEMA}_dry_plan",
        "cell": CELL,
        "design_authority": DESIGN_RELATIVE,
        "sealed_cell_d_swa_sha256": SEALED_CELL_D_SWA_SHA256,
        "rows": list(ROWS),
        "budgets": list(BUDGETS),
        "deployment_budgets": list(DEPLOYMENT_BUDGETS),
        "surfaces": list(SURFACES),
        "pre_registration": pre_registration_payload(),
        "required_data_root_environment": dict(EXACT_DATA_ROOT_ENV),
        "anchors": {
            "A0_M4_M10": ACTIVITY_ONLY_V2_RESULT_RELATIVE,
            "C0_M4_M10": V8_SCORE_RELATIVE,
            "A0_M30": V8_SCORE_RELATIVE,
            "P_M4_M10": PRECISION_V2_SCORE_RELATIVE,
        },
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "smoke_result_root_relative": SMOKE_RESULT_ROOT_RELATIVE,
        "smoke_parameters": dict(SMOKE_PARAMETERS),
        "inference_only": True,
        "no_torch_training": True,
        "target_optimizer_backward_update": 0,
    }
