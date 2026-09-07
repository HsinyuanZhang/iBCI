"""Static contract for Support-Anchored Causal T4 Memory, Stage P (V1).

Design authority: ``DESIGN_SUPPORT_ANCHORED_CAUSAL_T4_MEMORY_20260830.md``
§5 (deployable direction measurement), §6 (three-factor gate), §7.2 (the
P0--P5 matrix), §8 (promotion / attribution / stopping), §9 (receipts).

Work order: ``WORKORDER_SUPPORT_ANCHORED_T4_STAGE_P_V1_20260830.md``, which
embeds the SEALED Stage-O GO anchor (O2 - O0 external: M4 +0.1571351238257375,
M10 +0.05767866103494063; decision ``GO_DEPLOYABLE_PSEUDO_DIRECTION_PROGRAM``).
Stage P runs only under that GO.

Additive reuse: the immutable support anchor, the trust-region law and the
sealed activity-only loop are imported from the proven Stage-O package and are
pinned here as SHA-256 predecessors; no frozen package is edited.  Everything
that could bias the Stage-P measurement is frozen HERE, in module bytes the
attempt receipt pins before any data, model or CUDA access.  Score-only
successor: no training, no model change, no normalizer refit, no decoder-state
update.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Mapping, Tuple

try:
    from src.support_anchored_t4_stage_o_v1 import plan as stage_o_plan
except ModuleNotFoundError as error:
    if not (error.name == "src" or str(error.name).startswith("src.")): raise
    from tfpd_exploration.src.support_anchored_t4_stage_o_v1 import plan as stage_o_plan

CELL = "SUPPORT_ANCHORED_T4_STAGE_P_DEPLOYABLE_PSEUDO_DIRECTION_V1"
SCHEMA = "support_anchored_t4_stage_p_v1"

DESIGN_RELATIVE = "tfpd_exploration/docs/DESIGN_SUPPORT_ANCHORED_CAUSAL_T4_MEMORY_20260830.md"
WORK_ORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_SUPPORT_ANCHORED_T4_STAGE_P_V1_20260830.md"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/support_anchored_t4_stage_p_v1"

# ---------------------------------------------------------------------------
# The sealed Stage-O GO anchor (the binding precondition for Stage P).
# ---------------------------------------------------------------------------

STAGE_O_RESULT_ROOT_RELATIVE = "tfpd_exploration/results/support_anchored_t4_stage_o_v1"
STAGE_O_GO_ANCHOR = {
    "terminal_relative": f"{STAGE_O_RESULT_ROOT_RELATIVE}/terminal.json",
    "decision": "GO_DEPLOYABLE_PSEUDO_DIRECTION_PROGRAM",
    "m4_external_o2_minus_o0": 0.1571351238257375,
    "m10_external_o2_minus_o0": 0.05767866103494063,
    "law": (
        "Stage P is authorized only while the sealed Stage-O terminal receipt "
        "still carries this GO decision and these exact external deltas"
    ),
}

STAGE_O_RECEIPT_FILES = (
    f"{STAGE_O_RESULT_ROOT_RELATIVE}/attempt.json",
    f"{STAGE_O_RESULT_ROOT_RELATIVE}/directions.json",
    f"{STAGE_O_RESULT_ROOT_RELATIVE}/replay.json",
    f"{STAGE_O_RESULT_ROOT_RELATIVE}/terminal.json",
)

#: The Stage-O package this route imports from (drift = fail-closed).
STAGE_O_IMPORTED_MODULES = tuple(stage_o_plan.OWNED_PATHS)

PREDECESSOR_FILES: Tuple[str, ...] = (
    *stage_o_plan.PREDECESSOR_FILES,
    *STAGE_O_RECEIPT_FILES,
    *STAGE_O_IMPORTED_MODULES,
)

SEALED_ACTIVITY_ONLY_RESULT_SHA256 = stage_o_plan.SEALED_ACTIVITY_ONLY_RESULT_SHA256
ACTIVITY_ONLY_V2_RESULT_RELATIVE = stage_o_plan.ACTIVITY_ONLY_V2_RESULT_RELATIVE
V8_SCORE_RELATIVE = stage_o_plan.V8_SCORE_RELATIVE
P2PRIME_STAGE_COP_RELATIVE = stage_o_plan.P2PRIME_STAGE_COP_RELATIVE
FILTER_LINE_MANIFEST_RELATIVE = stage_o_plan.FILTER_LINE_MANIFEST_RELATIVE

EXACT_DATA_ROOT_ENV = dict(stage_o_plan.EXACT_DATA_ROOT_ENV)

# ---------------------------------------------------------------------------
# Cells (design §7.2).
# ---------------------------------------------------------------------------

BUDGETS = (4, 10, 30)
LOW_BUDGETS = (4, 10)
SURFACES = ("within", "external")
WITHIN_SESSION_COUNT = 6
EXTERNAL_SESSION_COUNT = 15

ROWS = {
    "P0": {
        "activity_state": "activity-only CDM (frozen B3S FIFO law)",
        "direction_measurement": "none",
        "carrier_update": "activity-only, frozen T4 (no carrier evidence)",
        "leakage": {"target_label_leakage": False, "self_referential_leakage": False},
        "governing": True,
        "role": (
            "reference; the Stage-O O0 arm VERBATIM -- must reproduce the sealed "
            "activity-only rows bit-exactly on every session/budget"
        ),
    },
    "P1": {
        "activity_state": "same as P0",
        "direction_measurement": (
            "cross-group circular ensemble theta_hat_minus_g = circular mean of the "
            "OTHER groups' completed-trial directions (group g never labels group g)"
        ),
        "carrier_update": "anchored block refit + trust region under the three-factor gate",
        "leakage": {"target_label_leakage": False, "self_referential_leakage": False},
        "governing": True,
        "role": "deployable candidate 1 (complementary-group exclusion only)",
    },
    "P2": {
        "activity_state": "same as P0",
        "direction_measurement": (
            "trajectory-level aggregation: d_hat_minus_g = sum_{h != g} sum_t v_hat_h[t]*dt, "
            "atan2 AFTER the displacement sum, plus the early/late displacement "
            "agreement statistic (confidence and coherence factor)"
        ),
        "carrier_update": "anchored block refit + trust region under the three-factor gate",
        "leakage": {"target_label_leakage": False, "self_referential_leakage": False},
        "governing": True,
        "role": "deployable candidate 2 (the primary cell; drives source selection)",
    },
    "P3": {
        "activity_state": "same as P0",
        "direction_measurement": "P2 direction with CONSTANT confidence (weight = 1)",
        "carrier_update": "anchored block refit + trust region under the three-factor gate",
        "leakage": {"target_label_leakage": False, "self_referential_leakage": False},
        "governing": True,
        "role": "constant-confidence control (confidence-attribution baseline)",
    },
    "P4": {
        "activity_state": "same as P0",
        "direction_measurement": (
            "P2 direction with the deterministic group/confidence row shuffle: group g "
            "receives group (g+1) mod 4's trajectory-law record and group (g+2) mod 4's "
            "confidence, same trial (marginal-preserving, causal, no tunable randomness)"
        ),
        "carrier_update": "anchored block refit + trust region under the three-factor gate",
        "leakage": {"target_label_leakage": False, "self_referential_leakage": False},
        "governing": True,
        "role": "shuffle control (semantic-binding baseline)",
    },
    "P5": {
        "activity_state": "same as P0",
        "direction_measurement": (
            "same-group pseudo direction (group g's own held-group forward labels "
            "group g); SELF-REFERENTIAL by construction"
        ),
        "carrier_update": "anchored block refit + trust region under the three-factor gate",
        "leakage": {"target_label_leakage": False, "self_referential_leakage": True},
        "governing": False,
        "role": "self-labeling diagnostic only; never a promotion or attribution baseline",
    },
}
ROW_ORDER = ("P0", "P1", "P2", "P3", "P4", "P5")
CANDIDATE_ROWS = ("P1", "P2")
CONTROL_ROWS = ("P3", "P4")
DIAGNOSTIC_ROWS = ("P5",)

LEAKAGE_FLAGS_BY_ROW = {
    row: {
        "target_label_leakage": bool(spec["leakage"]["target_label_leakage"]),
        "self_referential_leakage": bool(spec["leakage"]["self_referential_leakage"]),
        "checkpoint_selection_from_target": False,
        "hyperparameter_selection_from_target": False,
    }
    for row, spec in ROWS.items()
}

# ---------------------------------------------------------------------------
# Pre-registration 1: the measurement laws (design §5).
# ---------------------------------------------------------------------------

MEASUREMENT_LAW = {
    "group_prediction": (
        "the frozen four-group held-unit forward of the CURRENT pre-reveal state "
        "(ReviewedCDMScoreRuntime._group_predictions through the frozen P2' dispatch); "
        "prediction h excludes group h's units by construction"
    ),
    "per_group_direction": (
        "core.pseudo_direction_from_velocity under the session carrier's own frozen "
        "CDMDConfig, exactly as the Stage-O replay called it; ALL FOUR per-group "
        "directions must pass the frozen gates for the trial to yield any label "
        "(the frozen CDM all-four law, reused verbatim)"
    ),
    "p1_cross_group_circular": (
        "theta_hat_minus_g = atan2(mean sin, mean cos) over the accepted per-group "
        "raw angles h != g; dispersion_minus_g = max pairwise circular distance; "
        "resultant R_minus_g = mean resultant length"
    ),
    "p2_cross_group_trajectory": (
        "d_hat_minus_g = sum over h != g of (sum_t v_hat_h[t] * dt over the frozen "
        "movement mask); theta = atan2 AFTER the displacement sum; early/late split "
        "at movement_bins // 2 of the same rows; agreement = max(0, 1 - "
        "circular_distance(theta_early, theta_late)/(pi/2))"
    ),
    "confidence_weights": {
        "P1": "R_minus_g (complementary-group coherence only)",
        "P2": "R_minus_g * early/late agreement (predeclared §5.3 functions)",
        "P3": "constant 1.0 (the control)",
        "P4": "the rotated P2 confidence of group (g+2) mod 4",
        "P5": "constant 1.0 (no complementary ensemble exists)",
    },
    "frozen_local_gates": (
        "the production displacement / mean-speed / movement-bin / canonical-snap "
        "thresholds of the session's own CDMDConfig are never retuned; the "
        "aggregate displacement gate reuses config.minimum_displacement and the "
        "aggregate canonical margin reuses config.max_canonical_distance_rad"
    ),
    "measurement_confidence_factor": (
        "a group yields a label only if dispersion_minus_g <= tau_d, the aggregate "
        "displacement is nondegenerate, the aggregate canonical margin passes and "
        "(trajectory law only) the early/late agreement is strictly positive"
    ),
    "p4_shuffle": (
        "within-trial deterministic rotation (group g <- record of (g+1) mod 4 with "
        "the confidence of (g+2) mod 4): causal (no future trial), deterministic "
        "(no tunable randomness), marginal-preserving within the trial, and it "
        "restores exactly the self-labeling binding the deployable law forbids"
    ),
    "p5_same_group": (
        "group g's own frozen per-group direction labels group g; recorded as "
        "self_referential_leakage = true and never governing"
    ),
    "forbidden_inputs": (
        "no query behavior labels, no target R2, no future trials, no carrier "
        "outcomes, no learned confidence model (design §5.3)"
    ),
}

# ---------------------------------------------------------------------------
# Pre-registration 2: the commit law and the three-factor gate (design §4/§6).
# ---------------------------------------------------------------------------

COMMIT_LAW = {
    "block": "1 completed trial (the binding work-order law)",
    "per_group_labels": (
        "an evidence row carries an OPTIONAL label per complementary group; group "
        "g's sufficient statistics only include rows where group g has a label"
    ),
    "recomputation": (
        "T4_candidate = solve(A0 + rho_M * S, b0 + rho_M * t) recomputed ALWAYS from "
        "the immutable anchor plus the accepted bank; a rejected block therefore "
        "reproduces the pre-block state byte-for-byte (the Stage-O bank law)"
    ),
    "movement": (
        "T4_active = T4_support + alpha_M * project(candidate - support, D2 <= c_M) "
        "(the Stage-O trust-region law, reused verbatim); M30 uses alpha_M = 0, the "
        "exact no-op that returns the sealed support carrier bit-exactly"
    ),
    "three_factors": {
        "measurement_confidence": (
            "§6.1: cross-group dispersion below the source-frozen tau_d, "
            "nondegenerate aggregate displacement, canonical margin, and the "
            "trajectory-coherence pass for the trajectory law"
        ),
        "direction_coverage": (
            "§6.2: strictly positive logdet increase of X^T X + ridge under the "
            "candidate row, nondecreasing minimum eigenvalue, evidence-only "
            "repetition <= r_max per canonical direction per group, the "
            "no-deadlock distinct-direction warmup (a block commits only if its "
            "direction is NOVEL for the group or the bank already spans >= d_min "
            "distinct canonical directions -- a repeated high-confidence direction "
            "is not enough to update a full cosine tuning carrier), and cumulative "
            "pseudo mass rho_M * sum w <= max_mass_relative * support rows; no row "
            "is ever silently dropped (every rejection carries a typed reason)"
        ),
        "support_trust_region": (
            "§6.3 with the Stage-O D2 <= c_M law: a block whose AGGREGATE "
            "unprojected design-precision distance D2 = delta^T P_support delta "
            "exceeds the source-calibrated c_M is rejected whole; c_M is the "
            "within-6 median of the per-commit unprojected D2 (the Stage-O "
            "calibration law) and is never described as a credible region"
        ),
    },
    "gate_rejection_encoding": (
        "route-owned rejections are recorded with the frozen state-machine fallback "
        "reason INSUFFICIENT_EVIDENCE plus the precise factor in the receipt; "
        "frozen-law rejections keep their frozen enum value "
        "(complementary_disagreement / low_displacement / canonical_direction_too_far)"
    ),
}

# ---------------------------------------------------------------------------
# Pre-registration 3: hyperparameters and the source-only selection law.
# ---------------------------------------------------------------------------

TAU_D_CANDIDATES = (math.pi / 8.0, math.pi / 4.0)
R_MAX_CANDIDATES = (4, 8)
D_MIN_CANDIDATES = (3, 4)
MAX_PSEUDO_MASS_RELATIVE_CANDIDATES = (1.0, 2.0)
RHO_M_CANDIDATES = (0.5, 1.0)
ALPHA_M_CANDIDATES = (0.0, 0.125, 0.25, 0.5)

#: Fixed while the gate thresholds are selected (pre-registered stage order).
GATE_STAGE_FIXED = {"rho_M": 1.0, "alpha_M": 0.5}

HYPERPARAMETERS = {
    "selected_family": {
        "tau_d_rad": list(TAU_D_CANDIDATES),
        "r_max_repetition_per_direction": list(R_MAX_CANDIDATES),
        "d_min_distinct_directions": list(D_MIN_CANDIDATES),
        "max_pseudo_mass_relative_to_support_rows": list(MAX_PSEUDO_MASS_RELATIVE_CANDIDATES),
        "rho_M": list(RHO_M_CANDIDATES),
        "alpha_M": list(ALPHA_M_CANDIDATES),
        "c_M": "within-6 median of the per-commit unprojected D2 (Stage-O law)",
    },
    "fixed_predeclared": {
        "block_size_completed_trials": 1,
        "early_late_split": "movement_bins // 2 (first/second half of the frozen movement rows)",
        "early_late_agreement_scale_rad": math.pi / 2.0,
        "weight_law_shapes": "see MEASUREMENT_LAW confidence_weights; predeclared, never tuned",
        "alpha_m_candidate_set_note": (
            "design §4.3: the candidate set for alpha_M begins with {0, 0.125, 0.25, 0.5}"
        ),
    },
    "m30": {
        "alpha_M": 0.0,
        "rho_M": 1.0,
        "c_M": None,
        "law": (
            "M30 is an exact no-op in the deployable route (design §4.3/§8.1); the "
            "gate still runs for receipts and inherits the M10 source-selected "
            "thresholds (movement is zero by law, so the thresholds affect only "
            "which rows are receipted as committed evidence)"
        ),
    },
    "arithmetic": (
        "sufficient statistics, solves and trust-region arithmetic in float64; the "
        "active T4 is rebuilt in the production [a, c, hypot(a,c), b] mirror in the "
        "sealed support carrier's own dtype before any decoder forward; no silent "
        "FP32/FP64 substitution"
    ),
}

SELECTION = {
    "surface": "within-6 ONLY; external-15 never selects",
    "objective_cell": "P2 (the primary deployable candidate)",
    "objective": "the equal-session mean governing matrix R2 of the P2 rollout",
    "folds": "leave-one-session-out over the six within-surface sessions",
    "fold_collapse_disclosure": (
        "a session's rollout depends only on that session's own assets and the "
        "frozen runtime, so the fold mean equals the pooled within-6 mean; the "
        "per-fold values are still computed and published"
    ),
    "factored_stages": [
        {
            "stage": 1,
            "selects": ["tau_d_rad", "r_max", "d_min", "max_pseudo_mass_relative"],
            "grid_size": len(TAU_D_CANDIDATES) * len(R_MAX_CANDIDATES)
            * len(D_MIN_CANDIDATES) * len(MAX_PSEUDO_MASS_RELATIVE_CANDIDATES),
            "fixed": dict(GATE_STAGE_FIXED),
        },
        {
            "stage": 2,
            "selects": ["rho_M", "alpha_M"],
            "grid_size": len(RHO_M_CANDIDATES) * len(ALPHA_M_CANDIDATES),
            "uses": "the stage-1 selected gate thresholds",
        },
    ],
    "tie_break": "the first vector in the pre-registered enumeration order (less movement first)",
    "trust_region_during_selection": (
        "disabled (c_M unbounded); c_M is calibrated AFTER selection from the "
        "winning configuration's pooled within-6 per-commit unprojected D2 by the "
        "Stage-O median law, so the governing pass may reject blocks the selection "
        "pass committed -- a pre-registered, non-iterated, one-shot law"
    ),
    "per_budget": "independent selection per low budget (M4 and M10)",
    "hyperparameter_selection_from_target": False,
}

C_M_CALIBRATION = {
    "rule": (
        "per budget, the MEDIAN of the pooled per-commit unprojected aggregate D2 "
        "over the winning selection configuration's within-6 P2 rollouts "
        "(trust-region factor disabled during selection, exactly as Stage O calibrated)"
    ),
    "selection_surface": "within-6 ONLY; external-15 never selects",
    "statistic": "median",
    "geometry_claim_forbidden": (
        "c_M is a source-calibrated geometric trust threshold; it must never be "
        "described as a nominal 95% credible region (design §4.3)"
    ),
    "m30": "not calibrated; alpha_M = 0 is the exact no-op and D2 is still receipted",
    "frozen_before_target_scoring": True,
}

# ---------------------------------------------------------------------------
# Pre-registration 4: surfaces, filter, metric (inherited from Stage O).
# ---------------------------------------------------------------------------

OUTPUT_FILTER = dict(stage_o_plan.OUTPUT_FILTER)
METRIC = dict(stage_o_plan.METRIC)

# ---------------------------------------------------------------------------
# Pre-registration 5: the design §8 gates, VERBATIM (epsilon = 1e-12 band).
# ---------------------------------------------------------------------------

#: Program epsilon: margin comparisons are exact >= on the float64 equal-session
#: means; the epsilon only surfaces a disclosed within-band flag and never
#: flips a verdict (the Stage-O convention).
GATE_BOUNDARY_EPSILON = 1.0e-12

PROMOTION_EXTERNAL_DELTA = 0.01
PROMOTION_POSITIVE_EXTERNAL_SESSIONS = 10
OTHER_LOW_BUDGET_FLOOR = -0.01
WITHIN_EVERY_BUDGET_FLOOR = -0.02
CONTINUITY_EXTERNAL_DELTA = 0.01

DESIGN_8_1_VERBATIM = {
    "conditions": [
        "candidate - activity_only >= +0.01 equal-session external R2",
        "positive external sessions >= 10/15",
        "other low budget >= activity_only - 0.01",
        "within every budget >= activity_only - 0.02",
        "M30 carrier state is exact no-op",
        "target optimizer/backward/model/normalizer updates = 0",
    ],
    "must_also_beat": [
        "constant confidence",
        "deterministic confidence/group shuffle",
        "the same support anchor with no pseudo evidence",
    ],
    "note": "Beating sealed static but losing to activity-only CDM is a failed carrier result.",
}

DESIGN_8_2_VERBATIM = {
    "conditions": [
        "P2 - P1 >= +0.01 external R2",
        "positive sessions >= 10/15",
        "P2 beats constant and shuffle controls",
    ],
    "note": (
        "If P1 is positive and P2 is null, the carrier method may survive, but the "
        "continuity-specific claim is closed."
    ),
}

DESIGN_8_3_VERBATIM = {
    "conditions": [
        "the correct binding beats both constant confidence and a deterministic "
        "row/group shuffle on the same prediction records",
    ],
    "note": "A lower update count is not itself a positive result.",
}

DESIGN_8_4_VERBATIM = {
    "stop_conditions": [
        "O2 receives a STOP decision after the initial run or the one permitted HOLD sensitivity check",
        "oracle refit improves only the current trial through a causality leak",
        "P1/P2 fail to beat activity-only CDM",
        "gains disappear under constant/shuffle controls",
        "positive aggregate gain is carried by fewer than 10/15 external sessions",
        "M4/M10 gains require target-selected thresholds",
        "carrier updates require target gradients or decoder changes",
        "repeated carrier proposals drift outside the support trust region",
    ]
}

GATES = {
    "promotion": DESIGN_8_1_VERBATIM,
    "continuity_attribution": DESIGN_8_2_VERBATIM,
    "confidence_attribution": DESIGN_8_3_VERBATIM,
    "stop_conditions": DESIGN_8_4_VERBATIM,
    "boundary_epsilon": GATE_BOUNDARY_EPSILON,
    "boundary_rule": (
        "exact >= on float64 equal-session means; a miss inside the 1e-12 program "
        "epsilon is disclosed as within_epsilon_band_of_boundary and never flips "
        "the verdict"
    ),
    "beats_control_rule": (
        "'beats' a control = a strictly positive paired equal-session external "
        "delta on the driving budget (the +0.01 margins belong to the promotion "
        "and continuity claims); the comparison rides the SAME prediction records"
    ),
    "never_average_m30": "M30 is never averaged with M4/M10 to rescue or reject a low-budget effect",
}

DISPOSITION_PROMOTE = "PROMOTE_DEPLOYABLE_CARRIER_STATE"
DISPOSITION_STOP = "STOP_CONTINUOUS_T4_LINE"

INTERPRETATION_ROWS = {
    "p1_positive_p2_null": (
        "P1 positive and P2 null: the carrier method may survive, but the "
        "continuity-specific claim is closed (design §8.2)"
    ),
    "confidence_null": (
        "P2 fails to beat P3/P4: the confidence mapping carries no semantic "
        "information; a lower update count is not a positive result (design §8.3)"
    ),
    "headroom_without_deployable_measurement": (
        "O2 GO but P1/P2 null: continuous carrier adaptation has headroom, but the "
        "current deployable direction measurement is inadequate (design §11)"
    ),
    "gate_scope": (
        "a promotion authorizes the bounded carrier state of design §7.3 only "
        "after beating activity-only CDM and every semantic control"
    ),
}

#: Design §6.4 required precision ordering diagnostic (medians of carrier movement).
MOVEMENT_ORDERING = {
    "expression": "median carrier movement at M30 <= median at M10 <= median at M4",
    "statistic": (
        "per-session median over committed trials of the float64 Frobenius norm of "
        "(T4_active - T4_support) of the P2 row; the reported ordering uses the "
        "across-session median per budget; M30 is zero by construction (alpha_M = 0)"
    ),
    "epsilon": GATE_BOUNDARY_EPSILON,
    "status": "required diagnostic, non-governing for the promotion decision",
}

# ---------------------------------------------------------------------------
# Anchors (P0 bit-anchors to the sealed activity-only rows; Stage-O GO pinned).
# ---------------------------------------------------------------------------

ANCHORS = {
    "governing": {
        "path": ACTIVITY_ONLY_V2_RESULT_RELATIVE,
        "sha256_at_design": SEALED_ACTIVITY_ONLY_RESULT_SHA256,
        "law": (
            "per session/budget/surface, P0's raw prediction digest must equal the "
            "sealed activity-only prediction_sha256 and P0's house_raw_r2 must equal "
            "the sealed governing_r2 exactly (gap == 0.0); P0 is the Stage-O O0 arm "
            "VERBATIM, so this is also the proof that this route's loop is the "
            "sealed activity-only loop"
        ),
        "m30_path": V8_SCORE_RELATIVE,
    },
    "stage_o_go": dict(STAGE_O_GO_ANCHOR),
    "m30_noop": {
        "law": (
            "every P1--P5 row at M30 must reproduce P0@M30's raw predictions "
            "bit-exactly, keep carrier_before == carrier_after on every trial, and "
            "record zero movement (alpha_M = 0 is the exact no-op)"
        ),
    },
    "filter_line_cache": {
        "path": FILTER_LINE_MANIFEST_RELATIVE,
        "law": "secondary byte-anchor of P0's flat float32 raw stream, as in Stage O",
    },
    "p2prime_cross_reference": {
        "path": P2PRIME_STAGE_COP_RELATIVE,
        "law": "read-only cross-reference; never an anchor equality",
    },
}

# ---------------------------------------------------------------------------
# Process.
# ---------------------------------------------------------------------------

GPU_INDEX = 1
#: Runtime estimate (from the sealed Stage-O replay wall time): the selection
#: pass is ~24 within-6 P2 rollouts per low budget and the governing grid adds
#: the six P rows over the full grid, each P1--P5 trial costing one governing
#: plus four held-group forwards.  Expect roughly 10--11 GPU-hours; the hard
#: timeout is pre-registered at 15 h and can only change by a fresh attempt.
HARD_TIMEOUT_SECONDS = 54000

OWNED_PATHS = (
    DESIGN_RELATIVE,
    WORK_ORDER_RELATIVE,
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/__init__.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/plan.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/direction_estimator.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/gate.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/replay.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/gates.py",
    "tfpd_exploration/scripts/run_support_anchored_t4_stage_p_v1.py",
    "tfpd_exploration/tests/test_support_anchored_t4_stage_p_v1.py",
)


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def owned_sha256s(root: Path) -> dict[str, str]:
    base = Path(root).absolute()
    return {relative: _digest_file(base / relative) for relative in OWNED_PATHS}


def predecessor_sha256s(root: Path) -> dict[str, str]:
    base = Path(root).absolute()
    return {relative: _digest_file(base / relative) for relative in PREDECESSOR_FILES}


def hyperparameter_grid_payload() -> dict[str, object]:
    return {
        "tau_d_rad": list(TAU_D_CANDIDATES),
        "r_max": list(R_MAX_CANDIDATES),
        "d_min": list(D_MIN_CANDIDATES),
        "max_pseudo_mass_relative": list(MAX_PSEUDO_MASS_RELATIVE_CANDIDATES),
        "rho_M": list(RHO_M_CANDIDATES),
        "alpha_M": list(ALPHA_M_CANDIDATES),
    }


def pre_registration_payload() -> dict[str, object]:
    return {
        "cells": {key: dict(value) for key, value in ROWS.items()},
        "row_order": list(ROW_ORDER),
        "candidate_rows": list(CANDIDATE_ROWS),
        "control_rows": list(CONTROL_ROWS),
        "diagnostic_rows": list(DIAGNOSTIC_ROWS),
        "budgets": list(BUDGETS),
        "low_budgets": list(LOW_BUDGETS),
        "surfaces": list(SURFACES),
        "measurement_law": dict(MEASUREMENT_LAW),
        "commit_law": dict(COMMIT_LAW),
        "hyperparameters": dict(HYPERPARAMETERS),
        "hyperparameter_grid": hyperparameter_grid_payload(),
        "selection": dict(SELECTION),
        "c_m_calibration": dict(C_M_CALIBRATION),
        "output_filter": dict(OUTPUT_FILTER),
        "metric": dict(METRIC),
        "gates": {
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in GATES.items()
        },
        "interpretation_rows": dict(INTERPRETATION_ROWS),
        "movement_ordering": dict(MOVEMENT_ORDERING),
        "anchors": {
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in ANCHORS.items()
        },
        "leakage_flags_by_row": {
            row: dict(value) for row, value in LEAKAGE_FLAGS_BY_ROW.items()
        },
    }


def validate_pre_registration(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    hyper = result.get("hyperparameters", {})
    if hyper.get("fixed_predeclared", {}).get("block_size_completed_trials") != 1:
        raise ValueError("Stage-P block size pre-registration drift (the work order binds 1 trial)")
    grid = result.get("hyperparameter_grid", {})
    if list(grid.get("alpha_M", ())) != list(ALPHA_M_CANDIDATES):
        raise ValueError("Stage-P alpha_M candidate set drift (design §4.3 binds {0, 0.125, 0.25, 0.5})")
    if list(grid.get("rho_M", ())) != list(RHO_M_CANDIDATES):
        raise ValueError("Stage-P rho_M candidate set drift")
    if hyper.get("m30", {}).get("alpha_M") != 0.0:
        raise ValueError("Stage-P M30 must be the exact no-op (alpha_M = 0)")
    if tuple(result.get("row_order", ())) != ROW_ORDER:
        raise ValueError("Stage-P row topology pre-registration drift")
    if result.get("gates", {}).get("boundary_epsilon") != GATE_BOUNDARY_EPSILON:
        raise ValueError("Stage-P gate boundary epsilon drift")
    if result.get("selection", {}).get("surface") != "within-6 ONLY; external-15 never selects":
        raise ValueError("Stage-P hyperparameter selection surface drift")
    if result.get("c_m_calibration", {}).get("selection_surface") != "within-6 ONLY; external-15 never selects":
        raise ValueError("Stage-P c_M calibration surface drift")
    if result.get("commit_law", {}).get("block") != "1 completed trial (the binding work-order law)":
        raise ValueError("Stage-P commit law block pre-registration drift")
    return result


def dry_plan() -> dict[str, object]:
    return {
        "schema": f"{SCHEMA}_dry_plan",
        "cell": CELL,
        "design_authority": DESIGN_RELATIVE,
        "work_order": WORK_ORDER_RELATIVE,
        "stage_o_go_anchor": dict(STAGE_O_GO_ANCHOR),
        "pre_registration": pre_registration_payload(),
        "required_data_root_environment": dict(EXACT_DATA_ROOT_ENV),
        "predecessor_files": list(PREDECESSOR_FILES),
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "gpu_index": GPU_INDEX,
        "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
        "inference_only": True,
        "no_torch_training": True,
        "decoder_training": False,
        "model_or_checkpoint_updated": False,
        "target_optimizer_backward_update": 0,
    }
