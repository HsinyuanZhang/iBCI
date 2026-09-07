"""Static contract for Support-Anchored Causal T4 Memory, Stage O (V1).

Design authority: ``DESIGN_SUPPORT_ANCHORED_CAUSAL_T4_MEMORY_20260830.md``
(§3 cells, §4 block refit / trust region, §7.1 stage O, §9 receipts, §10
implementation boundaries, §13 execution order).

Work order: ``WORKORDER_SUPPORT_ANCHORED_T4_STAGE_O_V1_20260830.md`` including
its BINDING operator amendment 1 (pre-registered there before this module was
written):

* O2's commit law is **always-commit + trust-region projection only**; the
  three-factor gate's rejection semantics belong to Stage P;
* ``rho_M = 1.0`` (true directions need no shrinkage);
* block size = 1 completed trial;
* ``alpha_M = 0.5`` primary, ``{0.125, 0.25, 1.0}`` non-governing sensitivity;
* ``c_M`` source-calibrated on within-6 ONLY (external-15 never selects).

Everything that could bias the Stage-O measurement is frozen HERE, in module
bytes the attempt receipt pins by SHA-256 *before* any data, model or CUDA
access.  Score-only successor: no training, no model change, no normalizer
refit, no decoder-state update.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

try:
    from src.causal_dual_memory_cell_d_v1.core import RIDGE_NORMALIZED_LAMBDA
except ModuleNotFoundError as error:
    if not (error.name == "src" or str(error.name).startswith("src.")): raise
    from tfpd_exploration.src.causal_dual_memory_cell_d_v1.core import RIDGE_NORMALIZED_LAMBDA


CELL = "SUPPORT_ANCHORED_T4_STAGE_O_ORACLE_HEADROOM_V1"
SCHEMA = "support_anchored_t4_stage_o_v1"

DESIGN_RELATIVE = "tfpd_exploration/docs/DESIGN_SUPPORT_ANCHORED_CAUSAL_T4_MEMORY_20260830.md"
WORK_ORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_SUPPORT_ANCHORED_T4_STAGE_O_V1_20260830.md"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/support_anchored_t4_stage_o_v1"

# ---------------------------------------------------------------------------
# Immutable predecessors, bound by SHA-256 at attempt time (design §1.1).
# ---------------------------------------------------------------------------

ACTIVITY_ONLY_V2_RESULT_RELATIVE = (
    "tfpd_exploration/results/causal_dual_memory_cell_d_activity_only_quick_v2/result.json"
)
ACTIVITY_ONLY_V2_TERMINAL_RELATIVE = (
    "tfpd_exploration/results/causal_dual_memory_cell_d_activity_only_quick_v2/terminal.json"
)
V8_SCORE_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_matched_score_v8/score.json"
P2PRIME_STAGE_COP_RELATIVE = "tfpd_exploration/results/learned_gate_p2prime_v1/stage_cop.json"
P2PRIME_STAGE_A_RELATIVE = "tfpd_exploration/results/learned_gate_p2prime_v1/stage_a.json"
FILTER_LINE_MANIFEST_RELATIVE = "tfpd_exploration/cache/learnable_output_filter_v1/manifest.json"

SEALED_ACTIVITY_ONLY_RESULT_SHA256 = (
    "48d658c866b0bcb45a9f03c6e0c84204ecb87286e940d850c846a46f67b33bb8"
)

PREDECESSOR_FILES = (
    DESIGN_RELATIVE,
    WORK_ORDER_RELATIVE,
    ACTIVITY_ONLY_V2_RESULT_RELATIVE,
    ACTIVITY_ONLY_V2_TERMINAL_RELATIVE,
    V8_SCORE_RELATIVE,
    P2PRIME_STAGE_COP_RELATIVE,
    P2PRIME_STAGE_A_RELATIVE,
    FILTER_LINE_MANIFEST_RELATIVE,
)

SEALED_CELL_D_SWA_SHA256 = "626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd"

EXACT_DATA_ROOT_ENV = {
    "SUBC_DATA_ROOT": "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C",
    "SUBM_DATA_ROOT": "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M",
}

# ---------------------------------------------------------------------------
# Cells (design §3.3) over the frozen activity-only CDM runtime.
# ---------------------------------------------------------------------------

BUDGETS = (4, 10, 30)
LOW_BUDGETS = (4, 10)
SURFACES = ("within", "external")
WITHIN_SESSION_COUNT = 6
EXTERNAL_SESSION_COUNT = 15

ROWS = {
    "O0": {
        "activity_state": "activity-only CDM (frozen B3S FIFO law)",
        "direction_measurement": "none",
        "carrier_estimator": "frozen support T4 (no carrier evidence)",
        "target_label_leakage": False,
        "role": "reference; must reproduce the sealed activity-only rows bit-exactly",
    },
    "O1": {
        "activity_state": "same as O0",
        "direction_measurement": "TRUE completed-trial direction (leakage-labelled)",
        "carrier_estimator": "the EXISTING recursive incremental update (observe -> propose -> commit)",
        "target_label_leakage": True,
        "role": "diagnostic: does the old estimator fail even with correct directions",
    },
    "O2": {
        "activity_state": "same as O0",
        "direction_measurement": "TRUE completed-trial direction (leakage-labelled)",
        "carrier_estimator": "support-anchored block refit (design §4.1/§4.2) + trust-region projection (§4.3)",
        "target_label_leakage": True,
        "role": "the actual headroom test",
    },
    "O2A0125": {
        "activity_state": "same as O0",
        "direction_measurement": "TRUE completed-trial direction (leakage-labelled)",
        "carrier_estimator": "support-anchored block refit + trust-region projection, alpha_M=0.125",
        "target_label_leakage": True,
        "role": "non-governing alpha sensitivity",
    },
    "O2A025": {
        "activity_state": "same as O0",
        "direction_measurement": "TRUE completed-trial direction (leakage-labelled)",
        "carrier_estimator": "support-anchored block refit + trust-region projection, alpha_M=0.25",
        "target_label_leakage": True,
        "role": "non-governing alpha sensitivity",
    },
    "O2A100": {
        "activity_state": "same as O0",
        "direction_measurement": "TRUE completed-trial direction (leakage-labelled)",
        "carrier_estimator": "support-anchored block refit + trust-region projection, alpha_M=1.0",
        "target_label_leakage": True,
        "role": "non-governing alpha sensitivity",
    },
}
ROW_ORDER = ("O0", "O1", "O2", "O2A0125", "O2A025", "O2A100")
PRIMARY_ROWS = ("O0", "O1", "O2")
ALPHA_BY_ROW = {"O0": 0.0, "O1": None, "O2": 0.5, "O2A0125": 0.125, "O2A025": 0.25, "O2A100": 1.0}

LEAKAGE_FLAGS_BY_ROW = {
    row: {
        "target_label_leakage": bool(spec["target_label_leakage"]),
        "checkpoint_selection_from_target": False,
        "hyperparameter_selection_from_target": False,
    }
    for row, spec in ROWS.items()
}

# ---------------------------------------------------------------------------
# Pre-registration 1: the O2 commit law and hyperparameters (amendment 1).
# ---------------------------------------------------------------------------

O2_COMMIT_LAW = {
    "commit_rule": "always-commit",
    "movement_rule": "trust-region projection only",
    "rejection_semantics": "NONE in Stage O; the three-factor gate belongs to Stage P",
    "rationale": (
        "a gated O2 could turn gate miscalibration into a false STOP; the "
        "trust region alone bounds movement"
    ),
    "measurement_validity_still_required": (
        "a trial whose TRUE direction is not measurable under the frozen "
        "pseudo_direction_from_velocity law (displacement/speed/canonical "
        "snap gates) contributes no evidence row and moves nothing; the typed "
        "rejection reason is recorded"
    ),
    "old_estimator_gates_in_o2": (
        "the recursive estimator's B8 design/departure proposal checks are "
        "recorded as non-governing diagnostics only in O2"
    ),
}

HYPERPARAMETERS = {
    "rho_M": 1.0,
    "block_size_completed_trials": 1,
    "alpha_M_primary": 0.5,
    "alpha_M_sensitivity": (0.125, 0.25, 1.0),
    "alpha_M_sensitivity_governs": False,
    "evidence_weight_w": 1.0,
    "candidate_set_design_note": (
        "design §4.3 candidate set for alpha_M begins with {0, 0.125, 0.25, 0.5}; "
        "Stage O fixes 0.5 primary (alpha 0 is the O0 no-op identity, recorded as "
        "a law and a test, not as a scored row)"
    ),
    "m30_note": (
        "M30 is a mandatory diagnostic/safety column in Stage O and runs the same "
        "law; the DEPLOYABLE route keeps alpha_M = 0 at M30 (design §4.3)"
    ),
    "arithmetic": (
        "sufficient statistics, solves and trust-region arithmetic in float64; "
        "the active T4 is rebuilt in the production [a, c, hypot(a,c), b] mirror "
        "in the sealed support carrier's own dtype before any decoder forward; "
        "no silent FP32/FP64 substitution"
    ),
}

# ---------------------------------------------------------------------------
# Pre-registration 2: the support anchor (design §4.1) -- exact production law.
# ---------------------------------------------------------------------------

SUPPORT_ANCHOR_LAW = {
    "A0": "Xs^T Xs + diag(n_s * lambda, n_s * lambda, 0)",
    "b0": "Xs^T rs (per complementary group, response restricted to the group's units)",
    "T4_support": "solve(A0, b0) -- parity-checked against the sealed fit_carriers_from_trial_table carrier",
    "lambda": float(RIDGE_NORMALIZED_LAMBDA),
    "lambda_note": (
        "design §4.1 writes 'lambda I'; the exact production ridge law of "
        "fit_carriers_from_trial_table / _fit_group_from_statistics penalizes "
        "the cos/sin columns with n * 0.1 and leaves the intercept unpenalized; "
        "the production law governs (the work order binds the EXACT convention)"
    ),
    "design_columns": "[cos(theta_i), sin(theta_i), 1] with theta_i = CANONICAL_DIRECTIONS_RAD[direction_index]",
    "rate_domain": (
        "support: exact-duration rates (the frozen initializer domain); online "
        "evidence: mean native 20-ms counts over the full rewarded interval / 0.020"
    ),
    "immutability": (
        "A0/b0/support T4/statistics digests never change after construction; "
        "all movement lives in the separate evidence bank"
    ),
    "invalid_units": (
        "units outside the valid mask keep the sealed initial raw T4 rows; only "
        "valid units of each complementary group are refit (production _reconstruct law)"
    ),
}

# ---------------------------------------------------------------------------
# Pre-registration 3: the block refit (design §4.2).
# ---------------------------------------------------------------------------

BLOCK_REFIT_LAW = {
    "At": "A0 + rho_M * sum_i w_i x(theta_i) x(theta_i)^T  (per complementary group)",
    "bt": "b0 + rho_M * sum_i w_i x(theta_i) r_i^T         (per complementary group)",
    "T4_candidate": "solve(At, bt)",
    "recomputation": (
        "ALWAYS recomputed from A0/b0 plus the accepted evidence bank; never from "
        "a pseudo-updated carrier; a removed block reproduces the pre-block state "
        "exactly (digest-verified)"
    ),
    "evidence_row_binding": (
        "finalized trial ID, chronology position, session, per-group direction "
        "index + raw angle + canonical distance + displacement/speed/movement "
        "bins, scalar-rate digest, native-count digest, direction digest, "
        "channel-order digest, valid-mask digest, confidence rule, weight"
    ),
    "confidence_rule_stage_o": "true_direction_oracle (confidence = 1.0)",
    "rho_meaning": (
        "with rho_M = 1.0 and w_i = 1 each committed trial contributes exactly "
        "one unpenalized design row per group, mirroring after_pseudo_trial's "
        "single-count row; the anchor's ridge is NOT rescaled (that rescaling is "
        "the recursive estimator's law, deliberately not reused)"
    ),
}

# ---------------------------------------------------------------------------
# Pre-registration 4: the trust region (design §4.3).
# ---------------------------------------------------------------------------

TRUST_REGION_LAW = {
    "delta": "T4_candidate - T4_support in per-unit coefficient space (a, c, b)",
    "P_support": "A0 (the support design precision Xs^T Xs + production ridge penalty)",
    "D2": "sum over groups and valid units of delta_u^T P_support_g delta_u (aggregate)",
    "projection": "radial scaling delta * sqrt(c_M / D2) applied iff D2 > c_M",
    "T4_active": "T4_support + alpha_M * projected_delta, rebuilt as [a, c, hypot(a,c), b] in the sealed support carrier's dtype",
    "geometry": "support DESIGN precision; NOT a sandwich covariance, NOT a target-fitted posterior",
    "coverage_claim_forbidden": (
        "c_M is a source-calibrated geometric trust threshold; it must not be "
        "described as a nominal 95% credible region"
    ),
    "alpha_zero_identity": "alpha_M = 0 returns T4_support itself, byte-identical (the O0 no-op law)",
    "quantization_note": (
        "D2 and the projection act on the float64 coefficient delta; the published "
        "carrier is the production float32-quantized estimand, whose own D2 is "
        "bounded by c_M only down to the float32 quantization floor.  Receipts "
        "record the exact pre-quantization D2 and the published movement"
    ),
}

C_M_CALIBRATION = {
    "rule": "per budget, the MEDIAN of the pooled per-commit unprojected aggregate D2 over ALL committed evidence rows of the six within-surface sessions",
    "selection_surface": "within-6 ONLY; external-15 never selects",
    "invariance": (
        "the evidence bank never depends on the active carrier, so the per-commit "
        "D2 sequence is invariant to alpha_M and c_M; the calibration pass "
        "(c_M unbounded) therefore yields the exact D2 sequence the governing "
        "pass re-derives, asserted by bank-digest equality per trial"
    ),
    "statistic": "median",
    "quantile_rationale": (
        "the median is the pre-registered robust center of the within-6 geometric "
        "movement distribution; roughly half of within commits may project, which "
        "keeps the threshold a movement bound rather than a near-veto"
    ),
    "m30": "M30 calibrates its own c_M from the within-6 M30 diagnostic rollouts",
    "frozen_before_target_scoring": True,
}

# ---------------------------------------------------------------------------
# Pre-registration 5: surfaces, filter, metric (design §3.4).
# ---------------------------------------------------------------------------

OUTPUT_FILTER = {
    "family": "causal_ema",
    "alpha": 0.25,
    "scope": "per_completed_query_trial_window_sequence",
    "trial_boundary_reset": True,
    "governing_domain": "every Stage-O row (O0/O1/O2 and sensitivities) is scored on the filtered float64 matrix R2",
    "raw_domain": (
        "the raw per-trial governing predictions are retained per session; O0's "
        "raw digest and raw house R2 are the fields the sealed activity-only "
        "receipt carries and are the bit-anchor"
    ),
    "source": "inherited pre-registration of src/learned_gate_p2prime_v1/plan.py OUTPUT_FILTER",
}

METRIC = {
    "matrix_r2": "src.learned_gate_p2prime_v1.physical._matrix_r2 (float64 matched_metric.session_r2)",
    "house_raw_r2": "the frozen _house_raw_r2 float32 path on RAW predictions",
    "paired_contrast": "src.tfpd_lane.matched_scorer.paired_session_stats on per-session matrix_r2 deltas",
}

# ---------------------------------------------------------------------------
# Pre-registration 6: the §3.5 oracle headroom gate (exact boundary semantics).
# ---------------------------------------------------------------------------

#: Program epsilon (design-adjacent guard shared with the lane's convention).
#: The margin comparisons are exact >= on the float64 equal-session means; the
#: epsilon only surfaces a disclosed within-band flag and never flips a verdict.
GATE_BOUNDARY_EPSILON = 1.0e-12

GO_EXTERNAL_DELTA = 0.03
HOLD_EXTERNAL_DELTA_MIN = 0.01
GO_POSITIVE_EXTERNAL_SESSIONS = 10
HOLD_POSITIVE_EXTERNAL_SESSIONS = 9
EXTERNAL_SESSION_TOTAL = EXTERNAL_SESSION_COUNT
OTHER_LOW_BUDGET_FLOOR = -0.01
WITHIN_EVERY_BUDGET_FLOOR = -0.02

DISPOSITION_GO = "GO_DEPLOYABLE_PSEUDO_DIRECTION_PROGRAM"
DISPOSITION_HOLD = "HOLD_ONE_SOURCE_SELECTED_SENSITIVITY_ONLY"
DISPOSITION_STOP = "STOP_CLOSE_CONTINUOUS_T4_FOR_CURRENT_SYSTEM"

GATES = {
    "GO": {
        "requires_all_on_at_least_one_low_budget": [
            "O2 - O0 >= +0.03 equal-session external R2",
            "positive external sessions >= 10/15 (paired per session)",
            "the other low budget external >= O0 - 0.01",
            "within EVERY budget (incl. M30) >= O0 - 0.02",
            "all target updates/backward/optimizer steps = 0",
        ],
    },
    "HOLD": {
        "requires_all": [
            "primary external delta in [+0.01, +0.03)",
            "at least 9/15 external sessions positive",
            "no safety condition fails",
        ],
        "authorizes": "ONE source-selected block-size/shrinkage sensitivity check only",
    },
    "STOP": {
        "fires_when": (
            "O2 below +0.01 on both low budgets, or fewer than 9/15 external "
            "sessions positive, or the result depends on a causality leak, or a "
            "safety condition fails"
        ),
    },
    "safety_conditions": {
        "zero_target_updates": "optimizer/backward/parameter/normalizer update counts all zero",
        "o0_bit_anchor": "O0 reproduces the sealed activity-only rows bit-exactly on every session/budget",
        "causality_state_chains": (
            "per-trial forward state digest == state digest after the previous "
            "commit; no update can affect its own supplying trial"
        ),
        "within_regression_bound": "within-surface O2 - O0 >= -0.02 on every budget",
        "other_low_budget_bound": "the non-driving low budget external O2 - O0 >= -0.01",
    },
    "boundary_epsilon": GATE_BOUNDARY_EPSILON,
    "boundary_rule": (
        "exact >= on float64 equal-session means; a miss inside the 1e-12 program "
        "epsilon is disclosed as within_epsilon_band_of_boundary and never flips "
        "the verdict"
    ),
    "never_average_m30": "M30 is never averaged with M4/M10 to rescue or reject a low-budget effect",
}

INTERPRETATION_ROWS = {
    "recursive_estimator_at_fault": (
        "O2 > O0 and O1 <= O0: the recursive update is wrong, but carrier "
        "adaptation has headroom"
    ),
    "both_help": "O2 > O1 > O0: both the measurement and the estimator can help",
    "close_continuous_t4": "O1 and O2 both <= O0: close continuous T4 for the current decoder/activity system",
    "m4_only": "positive M4 only: continue as an M4-only carrier extension",
    "m10_only": "positive M10 only: treat M4 pseudo-direction quality as the likely bottleneck",
    "nonpositive_m30": "nonpositive M30: keep M30 carrier movement exactly disabled",
    "gate_scope": "the oracle gate decides whether headroom exists; it cannot establish a deployable method claim",
}

#: Design §6.4 required precision ordering diagnostic (medians of carrier movement).
MOVEMENT_ORDERING = {
    "expression": "median carrier movement at M30 <= median at M10 <= median at M4",
    "statistic": "per-session median over committed trials of the float64 Frobenius norm of (T4_active - T4_support); the reported ordering uses the across-session median per budget",
    "epsilon": GATE_BOUNDARY_EPSILON,
    "status": "required diagnostic, non-governing for the GO/HOLD/STOP decision",
}

# ---------------------------------------------------------------------------
# Anchors (work order: O0 bit-anchor + P2' cross-reference, read-only).
# ---------------------------------------------------------------------------

ANCHORS = {
    "governing": {
        "path": ACTIVITY_ONLY_V2_RESULT_RELATIVE,
        "sha256_at_design": SEALED_ACTIVITY_ONLY_RESULT_SHA256,
        "fields": ("prediction_sha256", "governing_r2"),
        "law": (
            "per session/budget/surface, O0's raw prediction digest must equal the "
            "sealed activity-only prediction_sha256 and O0's house_raw_r2 must equal "
            "the sealed governing_r2 exactly (gap == 0.0)"
        ),
        "m30_law": (
            "the sealed activity-only receipt covers M4/M10 only; O0@M30 anchors "
            "against the V8 matched score's sealed STATIC cells (the activity-only "
            "M30 rollout is bit-identical to sealed static there because the "
            "capacity-zero FIFO never mutates the activity stack -- the same anchor "
            "law P2' stage-A used for A0@M30)"
        ),
        "m30_path": V8_SCORE_RELATIVE,
    },
    "filter_line_cache": {
        "path": FILTER_LINE_MANIFEST_RELATIVE,
        "law": (
            "secondary: the flat float32 [W,2] raw stream of O0 per "
            "(surface, session, budget) must be byte-identical to the filter-line "
            "cdm cache npz 'raw' array bound by the manifest"
        ),
    },
    "p2prime_cross_reference": {
        "path": P2PRIME_STAGE_COP_RELATIVE,
        "law": "read-only cross-reference; never an anchor equality",
        "role": (
            "the sealed P2' coherent oracle O2 (true-direction construction + "
            "future-reading oracle accept) is the UPPER bound bounding the "
            "deterministic always-commit law's loss; external M4 O2 - O0 is the "
            "quoted ~+0.092 figure"
        ),
    },
}

# ---------------------------------------------------------------------------
# Process.
# ---------------------------------------------------------------------------

GPU_INDEX = 1
HARD_TIMEOUT_SECONDS = 21600

OWNED_PATHS = (
    DESIGN_RELATIVE,
    WORK_ORDER_RELATIVE,
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/__init__.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/plan.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/anchor.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/block_refit.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/trust_region.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/gates.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/directions.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/replay.py",
    "tfpd_exploration/scripts/run_support_anchored_t4_stage_o_v1.py",
    "tfpd_exploration/tests/test_support_anchored_t4_stage_o_v1.py",
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


def pre_registration_payload() -> dict[str, object]:
    return {
        "cells": {key: dict(value) for key, value in ROWS.items()},
        "row_order": list(ROW_ORDER),
        "primary_rows": list(PRIMARY_ROWS),
        "budgets": list(BUDGETS),
        "low_budgets": list(LOW_BUDGETS),
        "surfaces": list(SURFACES),
        "o2_commit_law": dict(O2_COMMIT_LAW),
        "hyperparameters": dict(HYPERPARAMETERS),
        "support_anchor_law": dict(SUPPORT_ANCHOR_LAW),
        "block_refit_law": dict(BLOCK_REFIT_LAW),
        "trust_region_law": dict(TRUST_REGION_LAW),
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
    if result.get("hyperparameters", {}).get("rho_M") != 1.0:
        raise ValueError("Stage-O rho_M pre-registration drift (amendment 1 binds 1.0)")
    if result.get("hyperparameters", {}).get("alpha_M_primary") != 0.5:
        raise ValueError("Stage-O alpha_M primary pre-registration drift (amendment 1 binds 0.5)")
    if result.get("hyperparameters", {}).get("block_size_completed_trials") != 1:
        raise ValueError("Stage-O block size pre-registration drift (amendment 1 binds 1 trial)")
    if tuple(result.get("row_order", ())) != ROW_ORDER:
        raise ValueError("Stage-O row topology pre-registration drift")
    if result.get("gates", {}).get("boundary_epsilon") != GATE_BOUNDARY_EPSILON:
        raise ValueError("Stage-O gate boundary epsilon drift")
    if result.get("c_m_calibration", {}).get("selection_surface") != "within-6 ONLY; external-15 never selects":
        raise ValueError("Stage-O c_M calibration surface drift")
    return result


def dry_plan() -> dict[str, object]:
    return {
        "schema": f"{SCHEMA}_dry_plan",
        "cell": CELL,
        "design_authority": DESIGN_RELATIVE,
        "work_order": WORK_ORDER_RELATIVE,
        "sealed_cell_d_swa_sha256": SEALED_CELL_D_SWA_SHA256,
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
