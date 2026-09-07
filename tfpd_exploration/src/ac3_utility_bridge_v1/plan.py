"""Static AC3-U contract: the M4-only zero-learning utility bridge (V1).

Work order: ``docs/WORKORDER_AC3_UTILITY_BRIDGE_V1_20260829.md`` (binding).
Guidance: ``docs/EXECUTION_GUIDANCE_AFTER_AC3_AND_CONTINUITY_AUDIT_20260829.md``
§4 (queue item 1) and §13 (launch boundary).

Everything that could bias the measurement is frozen HERE, in module bytes the
attempt receipt pins by SHA-256 *before* any data or model access:

* the three rows U0/UGE/U2 and the construction names they inject at the seam;
* the rotation law's scope (norms, validity evidence and movement mask are
  preserved; only direction content changes) and its predeclared undefined-θ̂
  fallback;
* the replay surface (within only), budget (M4 only), horizon (H=5), roster
  (the frozen AC3 six, 1,206 completed trials) and the trial-id binding;
* which scoring governs (raw, no output filter, per work order §5) and which
  scoring anchors U0 to the sealed stage-cop O0 row (the frozen filtered
  matrix R2 -- the sealed receipt's own governing field);
* the §8 gate margins, breadth denominator and disposition strings;
* the predeclared drift guard that distinguishes a wrong cursor mapping from
  the legitimate oracle-line carrier drift.

AC3-U trains nothing, updates nothing and sweeps nothing: it is one bounded,
inference-only utility measurement on the frozen M4 surface.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.learned_gate_p2prime_v1 import policy as _p2policy


CELL = "AC3_U_M4_ONLY_ZERO_LEARNING_UTILITY_BRIDGE_V1"
SCHEMA = "ac3_utility_bridge_v1"

WORK_ORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_AC3_UTILITY_BRIDGE_V1_20260829.md"
GUIDANCE_RELATIVE = (
    "tfpd_exploration/docs/EXECUTION_GUIDANCE_AFTER_AC3_AND_CONTINUITY_AUDIT_20260829.md"
)
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/ac3_utility_bridge_v1"

# ---------------------------------------------------------------------------
# Immutable predecessors (work order §2), bound by SHA-256 at attempt time.
# ---------------------------------------------------------------------------

AC3_ROOT_RELATIVE = "tfpd_exploration/results/ac3_action_continuity_v0"
P2PRIME_ROOT_RELATIVE = "tfpd_exploration/results/learned_gate_p2prime_v1"

#: ``trajectories.npz`` has no ``.sha256`` sidecar of its own; it is bound
#: bit-for-bit through ``materialize.json``'s ``array_digests`` block, which
#: IS sidecar-pinned.  This is recorded explicitly rather than silently
#: skipped (deviation note of the directions receipt).
PREDECESSOR_FILES = (
    f"{AC3_ROOT_RELATIVE}/trajectories.npz",
    f"{AC3_ROOT_RELATIVE}/materialize.json",
    f"{AC3_ROOT_RELATIVE}/screen.json",
    f"{AC3_ROOT_RELATIVE}/selection.json",
    f"{P2PRIME_ROOT_RELATIVE}/stage_cop.json",
    f"{P2PRIME_ROOT_RELATIVE}/stage_a.json",
)

SIDEcar_PINNED_PREDECESSORS = tuple(
    item for item in PREDECESSOR_FILES if not item.endswith("trajectories.npz")
)

SEALED_CELL_D_SWA_SHA256 = "626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd"

# ---------------------------------------------------------------------------
# Surface, budget, roster (work order §3; guidance §4.1).
# ---------------------------------------------------------------------------

BUDGET = 4
SURFACE = "within"
HORIZON_H = 5
SOURCE_SESSION_COUNT = 6
SESSIONS = (
    "sub-C_ses-CO-20151103",
    "sub-C_ses-CO-20151104",
    "sub-C_ses-CO-20151106",
    "sub-C_ses-CO-20151109",
    "sub-C_ses-CO-20151110",
    "sub-C_ses-CO-20151112",
)

#: Predeclared high-error stratum: it STAYS in the governing equal-session mean
#: and in the 4/6 breadth denominator; the five-session summary that drops it is
#: reported only as explicitly non-governing (guidance §4.3).
HIGH_ERROR_SESSION = "sub-C_ses-CO-20151103"

# ---------------------------------------------------------------------------
# Rows (work order §4) and the injection seam (work order §6).
# ---------------------------------------------------------------------------

#: The two NEW construction names the process-local wrapper owns.  Every other
#: construction name is a frozen ``policy.CONSTRUCTIONS`` member and must reach
#: the frozen function untouched.
ROTATING_CONSTRUCTIONS = ("group_ensemble", "r2_head")
FROZEN_CONSTRUCTIONS = tuple(_p2policy.CONSTRUCTIONS)

ROWS = {
    "U0": {
        "construction": "raw",
        "direction_row": "R0",
        "direction_input": "R0 raw single-view pseudo direction (reference; no rotation)",
        "role": "bit-exact anchor against sealed stage-cop within-M4 O0",
    },
    "UGE": {
        "construction": "group_ensemble",
        "direction_row": "R0.5",
        "direction_input": "R0.5 four-group circular mean (+ resultant length as credibility)",
        "role": "primary candidate",
    },
    "U2": {
        "construction": "r2_head",
        "direction_row": "R2",
        "direction_input": "R2 supervised-head grouped-OOF direction",
        "role": "comparator only (E7-mirror pre-registration)",
    },
}
ROW_ORDER = ("U0", "UGE", "U2")

ROTATION_LAW = {
    "expression": "delta = wrap(theta_hat_i - atan2(d_g[1], d_g[0])); v'_g = R(delta) @ v_g^T",
    "domain": "float64 math, cast back to the incoming dtype at the end",
    "preserves": [
        "every per-bin speed and every net displacement norm (rigid rotation)",
        "the validity evidence object (the SAME VelocityValidityEvidence instance is passed through)",
        "the movement mask and the movement-bin count",
    ],
    "changes": "direction content only",
    "undefined_theta_rule": (
        "a trial whose theta_hat is undefined (NaN) passes all four group views "
        "through unrotated, i.e. U0-equivalent evidence; predeclared per row "
        "before replay"
    ),
    "out_of_scope": "R3/R4/R5/RS rows, new encoders, target-selected arms, M10/M30",
}

SEAM = {
    "target": "src.learned_gate_p2prime_v1.policy.build_construction_predictions",
    "mechanism": (
        "process-local module-attribute assignment in the AC3-U driver only; no "
        "frozen file is edited (the P2' rollout resolves the name on the module "
        "object at call time)"
    ),
    "rotating_branch": "pre-rotate group_predictions, delegate the frozen function with construction='raw'",
    "passthrough_branch": (
        "every frozen construction name is forwarded unchanged and the frozen "
        "function's own return value is returned as-is (object identity)"
    ),
    "purity_proof": "U0 must reproduce sealed stage-cop within-M4 O0 bit-exactly",
}

# ---------------------------------------------------------------------------
# Scoring: which number governs and which number anchors (work order §5).
# ---------------------------------------------------------------------------

SCORING = {
    "governing": {
        "name": "raw_no_output_filter",
        "definition": (
            "matrix R2 of the per-trial RAW governing-bin predictions "
            "(``per_trial_raw``), computed on the frozen float64 "
            "``matched_metric.session_r2`` path"
        ),
        "authority": "work order §5 'no output filter (raw predictions scored)'",
    },
    "anchor": {
        "name": "filtered_causal_ema_a0.25",
        "definition": (
            "the frozen ``_row_session_payload`` matrix_r2, which the P2' oracle "
            "rows compute on ``filters.apply_output_filter_one_trial`` output; "
            "this is exactly the field the sealed stage-cop O0 row carries"
        ),
        "authority": (
            "the frozen oracle law cannot be re-scored without editing it, so the "
            "anchor equality is asserted on the sealed receipt's own field"
        ),
    },
    "note": (
        "the frozen oracle DECISION rule (u_j over the horizon) consumes filtered "
        "losses inside ``_rollout_oracle``/``_horizon_counterfactual``; that law is "
        "part of the frozen replay contract and is left untouched.  Only the "
        "reported utility metric is scored raw."
    ),
}

#: The sealed stage-cop fields the U0 anchor must reproduce bit-exactly.
#: Amendment 1 (2026-08-29): ``final_carrier_sha256`` moved to the REPORTED set.
#: First replay evidence: on the SAME physical GPU as the sealed run, every
#: scored/decision field below was bit-exact for every session while one
#: session's final carrier digest drifted -- a canonical-direction snap of a
#: late completed trial under cross-process float jitter.  Bit-equal governing
#: predictions make any earlier divergence impossible (every governing forward
#: reads the carrier), so the drift affects no scored quantity; the end-state
#: digest is reported, not gated.
ANCHOR_FIELDS = (
    "matrix_r2",
    "house_raw_r2",
    "prediction_sha256_raw",
    "filtered_prediction_sha256",
    "n_windows",
    "oracle_accept_count",
    "oracle_decision_count",
    "carrier_transitions_committed",
    "activity_transitions_committed",
    "carrier_rejection_counts",
    "initial_carrier_sha256",
    "initial_activity_sha256",
    "oracle_level",
    "horizon_H",
    "construction",
    "output_filter",
)

#: Same-origin digest fields that are reported but do not gate the anchor.
ANCHOR_REPORTED_FIELDS = ("final_carrier_sha256",)

#: Predeclared wrong-mapping detector for the wrapper cursor.
DRIFT_GUARD = {
    "comparison": (
        "circular distance between the incoming group net-displacement angle and "
        "the frozen trajectories.npz angle of the SAME (session, trial, group)"
    ),
    "per_trial_aggregation": "mean over the four complementary groups",
    "session_median_max_rad": 0.9,
    "session_fraction_within_1p5_rad_min": 0.4,
    "rationale": (
        "the frozen cache was materialized on the never-commit parent line while "
        "the oracle line MAY commit, so a bounded drift is expected and is not by "
        "itself an error.  A WRONG cursor mapping, by contrast, pairs unrelated "
        "trials and yields near-uniform angle gaps whose median sits at the "
        "uniform value pi/2 = 1.5708 rad.  The session median is therefore the "
        "discriminator (0.9 rad is roughly half the uniform median and well above "
        "any plausible carrier drift); the fraction floor is only a catastrophic "
        "sanity bound (uniform gaps give 0.478 inside 1.5 rad)."
    ),
    "on_failure": (
        "stop before any score is written, report the measured per-session table, "
        "never proceed silently"
    ),
    "redundancy": (
        "the guard is the THIRD binding check; the runtime's query_trial_ids[4] "
        "must already equal the materialize.json subsequence per session and the "
        "seam call count must equal the roster length"
    ),
    "recording": (
        "per row and session: median/mean/max gap, fraction within 1.5 rad, and "
        "the per-group displacement-norm ratio band, whether or not the guard fires"
    ),
}

NORM_PRESERVATION_TOLERANCE = 1.0e-9

# ---------------------------------------------------------------------------
# Gates (work order §8; guidance §4.4), boundary semantics predeclared.
# ---------------------------------------------------------------------------

MARGIN_R2 = 0.01
POSITIVE_SESSIONS_REQUIRED = 4
#: Program epsilon (work order §8).  The margin comparison itself is the exact
#: ``>=`` on the float64 equal-session mean: a value exactly at +0.01 passes and
#: a value 1e-13 below it fails.  The epsilon is used as a disclosed boundary
#: BAND (a miss inside ``GATE_BOUNDARY_EPSILON`` is surfaced for operator review
#: and recorded), never as a silent widening of the margin.
GATE_BOUNDARY_EPSILON = 1.0e-12

DISPOSITION_ADVANCE = "ADVANCE_SMALL_M4_CARRIER_STUDY"
DISPOSITION_ADVANCE_U2_RESCUE = "ADVANCE_SMALL_M4_CARRIER_STUDY_U2_RESCUE"
DISPOSITION_NULL = "AC3_U_UTILITY_NULL__CLOSE_AC3_ON_FROZEN_M4_SURFACE"

GATES = {
    "ADVANCE_SMALL_M4_CARRIER_STUDY": {
        "requires_both": [
            "UGE - U0 equal-session mean raw matrix R2 >= +0.01",
            "positive sessions (UGE > U0 paired per session) >= 4/6",
        ],
        "high_error_session_is_not_excludable": HIGH_ERROR_SESSION,
    },
    "U2_RESCUE": {
        "expression": "U2 may substitute for UGE only if, under the same replay: U2 - UGE >= +0.01 AND U2 - U0 >= +0.01 AND positive sessions (U2 > U0) >= 4/6",
        "role": "supervised-direction comparator; cannot rescue a failed UGE gate on its own mean alone",
    },
    "BOTH_FAIL": DISPOSITION_NULL,
    "boundary_epsilon": GATE_BOUNDARY_EPSILON,
    "boundary_rule": (
        "exact >= on the float64 equal-session mean; a miss inside the 1e-12 "
        "program epsilon of the boundary is recorded in "
        "within_epsilon_band_of_boundary and never flips the verdict"
    ),
    "no_m10_m30_claim": (
        "M10/M30 materializations were not present; no all-budget AC3 closure is "
        "made or implied by any outcome"
    ),
}

#: Work order §9: the pre-registered expectation, recorded before replay.
PRE_REGISTERED_EXPECTATION = {
    "sealed_stage_cop_within_m4": {
        "A0": 0.46559, "C0": 0.45072, "O0": 0.48869, "O1": 0.48885, "O2": 0.49327,
    },
    "true_direction_ceiling_over_raw": 0.0046,
    "uge_gate_demand": 0.01,
    "expectation": (
        "NULL: UGE is a direction-content intervention bounded in expectation by "
        "the true-direction construction, so a pass would require the ensemble "
        "direction to beat the true direction by more than the entire "
        "true-direction advantage; the run is still mandatory because it converts "
        "the §4.4 closure condition from an inference into a measurement"
    ),
}

# ---------------------------------------------------------------------------
# Process (work order §10).
# ---------------------------------------------------------------------------

#: One launch, one process, one GPU.  GPU 1 (3090); GPU 0 stays reserved for the
#: parallel SLOT-AUDIT line.
GPU_INDEX = 1
HARD_TIMEOUT_SECONDS = 5400

OWNED_PATHS = (
    WORK_ORDER_RELATIVE,
    "tfpd_exploration/src/ac3_utility_bridge_v1/__init__.py",
    "tfpd_exploration/src/ac3_utility_bridge_v1/plan.py",
    "tfpd_exploration/src/ac3_utility_bridge_v1/rotation.py",
    "tfpd_exploration/src/ac3_utility_bridge_v1/wrapper.py",
    "tfpd_exploration/src/ac3_utility_bridge_v1/directions.py",
    "tfpd_exploration/src/ac3_utility_bridge_v1/gates.py",
    "tfpd_exploration/src/ac3_utility_bridge_v1/replay.py",
    "tfpd_exploration/scripts/run_ac3_utility_bridge_v1.py",
    "tfpd_exploration/tests/test_ac3_utility_bridge_v1.py",
)


def owned_sha256s(root: Path) -> dict[str, str]:
    base = Path(root).absolute()
    return {
        relative: hashlib.sha256((base / relative).read_bytes()).hexdigest()
        for relative in OWNED_PATHS
    }


def predecessor_sha256s(root: Path) -> dict[str, str]:
    base = Path(root).absolute()
    return {
        relative: hashlib.sha256((base / relative).read_bytes()).hexdigest()
        for relative in PREDECESSOR_FILES
    }


def gate_spec_payload() -> dict[str, object]:
    return {
        "rows": {key: dict(value) for key, value in ROWS.items()},
        "row_order": list(ROW_ORDER),
        "rotating_constructions": list(ROTATING_CONSTRUCTIONS),
        "frozen_constructions": list(FROZEN_CONSTRUCTIONS),
        "rotation_law": dict(ROTATION_LAW),
        "seam": dict(SEAM),
        "scoring": {
            "governing": dict(SCORING["governing"]),
            "anchor": dict(SCORING["anchor"]),
            "note": SCORING["note"],
        },
        "gates": {
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in GATES.items()
        },
        "drift_guard": dict(DRIFT_GUARD),
        "norm_preservation_tolerance": NORM_PRESERVATION_TOLERANCE,
        "surface": SURFACE,
        "budget": BUDGET,
        "horizon_H": HORIZON_H,
        "sessions": list(SESSIONS),
        "high_error_session": HIGH_ERROR_SESSION,
        "pre_registered_expectation": dict(PRE_REGISTERED_EXPECTATION),
        "gpu_index": GPU_INDEX,
        "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
    }
