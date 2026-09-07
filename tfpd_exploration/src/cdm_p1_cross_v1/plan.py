"""Static contract for the CDM x P1 cross-dataset factorial (Part A, V1).

Work order: ``docs/WORKORDER_CDM_P1_CROSS_DATASET_V1_20260831.md`` section 2
(Part A -- the DANDI 000688 inference-only F00/F10/F01/F11 factorial).

Everything that could bias the measurement is frozen HERE, in module bytes the
attempt receipt pins before any data, model or CUDA access:

* the four cells and their weight/carrier factors;
* the frozen activity-only runtime and the identical trial order (the sealed
  ``src/learned_gate_p2prime_v1`` materialization reused verbatim);
* the P1 online carrier law -- the sealed Stage-P ``rollout_p`` machinery with
  the hyperparameters of the SEALED stage-P within-6 selection, never
  re-selected here (pinned as values AND re-verified against the sealed
  stage-P receipts at attempt and replay time);
* the C1 weight arm -- the sealed ``cal_aug_v1`` prefix-cycle SWA
  (``swa_final4.pt``), swapped in by the strict-load + state-digest proof
  pattern of ``src/cal_aug_v1/deployment._swap_runtime_model``;
* the work-order section 2 gate boundaries with the Stage-O/P program epsilon
  (1e-12) as a disclosed boundary band only.

Score-only successor: no training, no model change beyond the disclosed
between-arm swap, no normalizer refit, no decoder-state update.  Nothing under
any frozen result root is created or modified.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Mapping, Tuple

from src.support_anchored_t4_stage_p_v1 import plan as stage_p_plan

CELL = "CDM_P1_CROSS_DATASET_FACTORIAL_PART_A_V1"
SCHEMA = "cdm_p1_cross_v1"

WORK_ORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CDM_P1_CROSS_DATASET_V1_20260831.md"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/cdm_p1_cross_v1"

# ---------------------------------------------------------------------------
# Sealed predecessors: the Stage-P route (P0/P1 laws + receipts) and the C1
# weight artifact with its sealed deployment binding.
# ---------------------------------------------------------------------------

STAGE_P_RESULT_ROOT_RELATIVE = stage_p_plan.RESULT_ROOT_RELATIVE
STAGE_P_RECEIPT_FILES = (
    f"{STAGE_P_RESULT_ROOT_RELATIVE}/attempt.json",
    f"{STAGE_P_RESULT_ROOT_RELATIVE}/replay.json",
    f"{STAGE_P_RESULT_ROOT_RELATIVE}/terminal.json",
)
#: The Stage-P package bytes this route imports (drift = fail-closed).
STAGE_P_IMPORTED_MODULES = tuple(stage_p_plan.OWNED_PATHS)

CAL_AUG_RESULT_ROOT_RELATIVE = "tfpd_exploration/results/cal_aug_v1"
C1_SWA_RELATIVE = f"{CAL_AUG_RESULT_ROOT_RELATIVE}/c1_prefix_cycle/swa_final4.pt"
C1_TRAIN_TERMINAL_RELATIVE = f"{CAL_AUG_RESULT_ROOT_RELATIVE}/c1_prefix_cycle/terminal.json"
C1_DEPLOYMENT_TERMINAL_RELATIVE = f"{CAL_AUG_RESULT_ROOT_RELATIVE}/deployment/terminal.json"

#: The sealed C1 SWA artifact (bit-identical file the sealed deployment scored).
C1_SWA_SHA256 = "5cc24676777cb1eacb0ce5fd5174c55dc2efd5ca9ede69d24a85f935f8e89f8d"
#: ``arm_state_sha256`` the sealed cal_aug_v1 deployment recorded for the C1
#: arm after its strict load (device-independent digest law).
C1_ARM_STATE_SHA256 = "4e7ab197bc7935e9970e67b6df567cce8219e2ee95c73665e26520bfd7bc6dd8"
C1_DEPLOYMENT_STATUS = "DEPLOYMENT_SCORING_COMPLETE"
C1_TRAIN_TERMINAL_STATUS = "CAL_AUG_CELL_TERMINAL"

PREDECESSOR_FILES: Tuple[str, ...] = (
    *stage_p_plan.PREDECESSOR_FILES,
    *STAGE_P_IMPORTED_MODULES,
    *STAGE_P_RECEIPT_FILES,
    WORK_ORDER_RELATIVE,
    C1_SWA_RELATIVE,
    C1_TRAIN_TERMINAL_RELATIVE,
    C1_DEPLOYMENT_TERMINAL_RELATIVE,
)

SEALED_ACTIVITY_ONLY_RESULT_SHA256 = stage_p_plan.SEALED_ACTIVITY_ONLY_RESULT_SHA256
ACTIVITY_ONLY_V2_RESULT_RELATIVE = stage_p_plan.ACTIVITY_ONLY_V2_RESULT_RELATIVE
EXACT_DATA_ROOT_ENV = dict(stage_p_plan.EXACT_DATA_ROOT_ENV)

# ---------------------------------------------------------------------------
# The sealed Stage-P promotion anchor (the binding precondition for reusing P1).
# ---------------------------------------------------------------------------

STAGE_P_GO_ANCHOR = {
    "terminal_relative": f"{STAGE_P_RESULT_ROOT_RELATIVE}/terminal.json",
    "decision": "PROMOTE_DEPLOYABLE_CARRIER_STATE",
    "driving_cell": "P1@m4",
    "m4_external_p1_minus_p0": 0.020463084852528825,
    "law": (
        "Part A reuses the P1 online carrier only while the sealed Stage-P "
        "terminal receipt still carries this promotion decision, this driving "
        "cell and this exact external M4 delta"
    ),
}

# ---------------------------------------------------------------------------
# Cells (work order section 2).
# ---------------------------------------------------------------------------

BUDGETS = (4, 10, 30)
SURFACES = ("within", "external")
WITHIN_SESSION_COUNT = 6
EXTERNAL_SESSION_COUNT = 15

WEIGHT_ARMS = {
    "sealed": {
        "weights": "the sealed Cell-D SWA of the frozen activity-only runtime",
        "swap": "none (the runtime's own strict-loaded sealed model)",
    },
    "c1": {
        "weights": "the sealed cal_aug_v1 C1 prefix-cycle SWA (swa_final4.pt)",
        "swap": (
            "strict-load into a fresh pop_robust population-robustness model "
            "(seed 42, cell D) followed by the state-digest proof pattern of "
            "src/cal_aug_v1/deployment._swap_runtime_model, applied to BOTH the "
            "runtime state model and the executor state model the four held-group "
            "forwards read"
        ),
    },
}

CELLS = {
    "F00": {
        "weights": "sealed",
        "carrier": "frozen T4 (the Stage-P P0 / Stage-O O0 arm VERBATIM)",
        "role": "baseline; must reproduce the sealed Stage-P P0 rows bit-exactly",
        "governing": True,
        "leakage": {"target_label_leakage": False, "self_referential_leakage": False},
    },
    "F10": {
        "weights": "c1",
        "carrier": "frozen T4 (the same O0 law under the C1 weight arm)",
        "role": "CDM (x) C1: the weight factor alone",
        "governing": True,
        "leakage": {"target_label_leakage": False, "self_referential_leakage": False},
    },
    "F01": {
        "weights": "sealed",
        "carrier": "P1 online (the sealed Stage-P P1 rollout under its sealed selection)",
        "role": "CDM (x) P1: the carrier factor alone; must reproduce the sealed Stage-P P1 rows bit-exactly",
        "governing": True,
        "leakage": {"target_label_leakage": False, "self_referential_leakage": False},
    },
    "F11": {
        "weights": "c1",
        "carrier": "P1 online (the same sealed P1 law under the C1 weight arm)",
        "role": "the full stack",
        "governing": True,
        "leakage": {"target_label_leakage": False, "self_referential_leakage": False},
    },
}
CELL_ORDER = ("F00", "F10", "F01", "F11")
BASELINE_CELL = "F00"
CANDIDATE_CELLS = ("F10", "F01", "F11")
CELL_BY_ARM = {
    "sealed": ("F00", "F01"),
    "c1": ("F10", "F11"),
}
CARRIER_LAW_BY_CELL = {
    "F00": "frozen_t4",
    "F10": "frozen_t4",
    "F01": "p1_online",
    "F11": "p1_online",
}

LEAKAGE_FLAGS_BY_CELL = {
    cell: {
        "target_label_leakage": bool(spec["leakage"]["target_label_leakage"]),
        "self_referential_leakage": bool(spec["leakage"]["self_referential_leakage"]),
        "checkpoint_selection_from_target": False,
        "hyperparameter_selection_from_target": False,
        "hyperparameter_selection_reused_not_reselected": True,
    }
    for cell, spec in CELLS.items()
}

# ---------------------------------------------------------------------------
# Pre-registration 1: the P1 hyperparameters, VERBATIM from the sealed
# stage-P within-6 selection (no re-selection in this route).
# ---------------------------------------------------------------------------

_M10_THRESHOLDS = {
    "tau_d_rad": math.pi / 4.0,
    "r_max_repetition_per_direction": 4,
    "d_min_distinct_directions": 4,
    "max_pseudo_mass_relative_to_support_rows": 2.0,
}

SEALED_P1_HYPERPARAMETERS = {
    4: {
        "rho_M": 0.5,
        "alpha_M": 0.5,
        "c_M": 286.77449403760136,
        "thresholds": {
            "tau_d_rad": math.pi / 4.0,
            "r_max_repetition_per_direction": 8,
            "d_min_distinct_directions": 4,
            "max_pseudo_mass_relative_to_support_rows": 2.0,
        },
    },
    10: {
        "rho_M": 1.0,
        "alpha_M": 0.5,
        "c_M": 426.5520051748518,
        "thresholds": dict(_M10_THRESHOLDS),
    },
    30: {
        "rho_M": 1.0,
        "alpha_M": 0.0,
        "c_M": None,
        "thresholds": dict(_M10_THRESHOLDS),
        "inherits_thresholds_from": "m10 (the sealed Stage-P M30 exact-no-op law)",
    },
}

HYPERPARAMETER_LAW = {
    "source": (
        "the sealed Stage-P terminal receipt's hyperparameter_selection_summary "
        "(m4/m10) and the sealed Stage-P replay receipt's m30 no-op law"
    ),
    "re_selection": "FORBIDDEN -- the values are pinned here and re-verified "
                    "against the sealed receipts at attempt and replay time",
    "selection_surface": "within-6 ONLY; external-15 never selected (sealed law)",
    "p1_row_spec": "cross_group_circular + resultant_length + complementary_exclusion",
}

# ---------------------------------------------------------------------------
# Pre-registration 2: the weight-swap law.
# ---------------------------------------------------------------------------

WEIGHT_SWAP_LAW = {
    "pattern": "src/cal_aug_v1/deployment._swap_runtime_model",
    "steps": (
        "re-hash the sealed C1 SWA artifact and compare to the pinned digest; "
        "build a FRESH pop_robust population-robustness model (seed 42, cell D); "
        "load_state_dict(payload['state_dict'], strict=True); move to the "
        "runtime device and eval(); take arm_common.state_sha256 of the loaded "
        "model and compare to the sealed deployment's c1 arm_state_sha256; run "
        "one zeros double-forward purity proof; then replace BOTH "
        "state.model (governing forwards) and state.executor_state.model (the "
        "four held-group pseudo forwards)"
    ),
    "restore": (
        "the sealed model object is retained and restored after the C1 arm, with "
        "its state digest re-verified; the sealed arm's digest is additionally "
        "checked before and after every phase"
    ),
    "inference_only": True,
    "training_under_swap": "FORBIDDEN",
}

# ---------------------------------------------------------------------------
# Pre-registration 3: the work-order section 2 gates (epsilon 1e-12 band).
# ---------------------------------------------------------------------------

GATE_BOUNDARY_EPSILON = 1.0e-12
PROMOTION_EXTERNAL_DELTA = 0.01
PROMOTION_POSITIVE_EXTERNAL_SESSIONS = 10
ADDITIVITY_SLACK = 0.0
WITHIN_EVERY_BUDGET_FLOOR = -0.02
M30_FLOOR = -0.02
DRIVING_BUDGET = 4

GATES = {
    "per_cell_promotion": {
        "expression": (
            "cell - F00 >= +0.01 equal-session external M4 R2 AND positive "
            "external M4 sessions >= 10/15"
        ),
        "cells": list(CANDIDATE_CELLS),
        "driving_budget": DRIVING_BUDGET,
    },
    "additivity": {
        "expression": "F11 external M4 equal-session mean >= max(F10, F01) - 0",
        "slack": ADDITIVITY_SLACK,
    },
    "safety": {
        "within_every_budget_floor": WITHIN_EVERY_BUDGET_FLOOR,
        "m30_floor": M30_FLOOR,
        "expression": (
            "per candidate cell: within delta vs F00 >= -0.02 at EVERY budget "
            "(M4/M10/M30) and external M30 delta vs F00 >= -0.02"
        ),
    },
    "boundary_epsilon": GATE_BOUNDARY_EPSILON,
    "boundary_rule": (
        "exact >= on float64 equal-session means; a miss inside the 1e-12 program "
        "epsilon is disclosed as within_epsilon_band_of_boundary and never flips "
        "the verdict (the Stage-O/P convention)"
    ),
    "never_average_m30": "M30 is never averaged with M4/M10 to rescue or reject a low-budget effect",
    "nulls_reported_as_nulls": True,
}

DISPOSITION_CELL_PROMOTED = "CELL_IMPROVES_OVER_F00"
DISPOSITION_CELL_FAILED = "CELL_FAILS_THE_F00_GATE"
DISPOSITION_ADDITIVITY_HELD = "ADDITIVITY_HELD"
DISPOSITION_ADDITIVITY_VIOLATED = "ADDITIVITY_VIOLATED"

# ---------------------------------------------------------------------------
# Anchors.
# ---------------------------------------------------------------------------

ANCHORS = {
    "f00_vs_sealed_activity_only": {
        "path": ACTIVITY_ONLY_V2_RESULT_RELATIVE,
        "sha256_at_design": SEALED_ACTIVITY_ONLY_RESULT_SHA256,
        "law": (
            "the Stage-O anchor law (anchor_o0_vs_sealed): F00's raw prediction "
            "digest equals the sealed activity-only prediction_sha256 and F00's "
            "house_raw_r2 equals the sealed governing_r2 exactly on every "
            "session/budget/surface"
        ),
    },
    "f00_f01_vs_sealed_stage_p": {
        "path": f"{STAGE_P_RESULT_ROOT_RELATIVE}/replay.json",
        "law": (
            "per session/budget/surface, F00 reproduces the sealed Stage-P P0 row "
            "and F01 reproduces the sealed Stage-P P1 row bit-exactly "
            "(house_raw_r2, prediction_sha256_raw, matrix_r2, "
            "filtered_prediction_sha256 and n_windows all equal)"
        ),
    },
    "stage_p_go": dict(STAGE_P_GO_ANCHOR),
    "m30_noop_per_arm": {
        "law": (
            "within EACH weight arm, the P1 cell at M30 reproduces that arm's "
            "frozen-T4 cell at M30 bit-exactly, keeps carrier_before == "
            "carrier_after on every trial and records zero movement (alpha_M = 0)"
        ),
    },
    "initial_carrier_invariance": {
        "law": (
            "the initial carrier/activity/group-assignment digests are identical "
            "across weight arms for every session/budget (the deployment-recipe "
            "initial carrier is model-independent)"
        ),
    },
    "c1_weight_binding": {
        "artifact_sha256": C1_SWA_SHA256,
        "arm_state_sha256": C1_ARM_STATE_SHA256,
        "law": (
            "the swapped model's state digest equals the sealed cal_aug_v1 "
            "deployment's c1 arm_state_sha256 and the artifact hash equals the "
            "sealed deployment's artifact_sha256"
        ),
    },
}

# ---------------------------------------------------------------------------
# Process.
# ---------------------------------------------------------------------------

GPU_INDEX = 1
#: Runtime estimate: four cells over 3 budgets x 21 sessions at the sealed
#: Stage-P per-trial cost (one governing + four held-group double-pass
#: forwards).  The sealed Stage-P governing grid alone (6 rows) ran ~4 h on
#: this GPU, so expect roughly 2.5-3.5 GPU-hours; the hard timeout is
#: pre-registered at 6 h (the binding work-order ceiling).
HARD_TIMEOUT_SECONDS = 21600

OWNED_PATHS = (
    "tfpd_exploration/src/cdm_p1_cross_v1/__init__.py",
    "tfpd_exploration/src/cdm_p1_cross_v1/plan.py",
    "tfpd_exploration/src/cdm_p1_cross_v1/weights.py",
    "tfpd_exploration/src/cdm_p1_cross_v1/replay.py",
    "tfpd_exploration/src/cdm_p1_cross_v1/gates.py",
    "tfpd_exploration/scripts/run_cdm_p1_cross_v1.py",
    "tfpd_exploration/tests/test_cdm_p1_cross_v1.py",
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


def hyperparameters_payload() -> dict[str, object]:
    return {
        f"m{budget}": dict(value) for budget, value in SEALED_P1_HYPERPARAMETERS.items()
    }


def pre_registration_payload() -> dict[str, object]:
    return {
        "cells": {key: dict(value) for key, value in CELLS.items()},
        "cell_order": list(CELL_ORDER),
        "baseline_cell": BASELINE_CELL,
        "candidate_cells": list(CANDIDATE_CELLS),
        "cell_by_arm": {arm: list(cells) for arm, cells in CELL_BY_ARM.items()},
        "carrier_law_by_cell": dict(CARRIER_LAW_BY_CELL),
        "weight_arms": {arm: dict(value) for arm, value in WEIGHT_ARMS.items()},
        "budgets": list(BUDGETS),
        "surfaces": list(SURFACES),
        "within_session_count": WITHIN_SESSION_COUNT,
        "external_session_count": EXTERNAL_SESSION_COUNT,
        "hyperparameters": hyperparameters_payload(),
        "hyperparameter_law": dict(HYPERPARAMETER_LAW),
        "weight_swap_law": dict(WEIGHT_SWAP_LAW),
        "gates": {
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in GATES.items()
        },
        "anchors": {
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in ANCHORS.items()
        },
        "leakage_flags_by_cell": {
            cell: dict(value) for cell, value in LEAKAGE_FLAGS_BY_CELL.items()
        },
    }


def validate_pre_registration(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    if tuple(result.get("cell_order", ())) != CELL_ORDER:
        raise ValueError("Part-A cell topology pre-registration drift")
    cells = result.get("cells", {})
    for cell in CELL_ORDER:
        if cell not in cells:
            raise ValueError(f"Part-A pre-registration is missing cell {cell}")
        weights = str(cells[cell].get("weights"))
        expected_arm = "c1" if cell in ("F10", "F11") else "sealed"
        if weights != expected_arm:
            raise ValueError(f"Part-A cell {cell} weight-arm pre-registration drift")
    expected_carrier = {
        "F00": "frozen_t4", "F10": "frozen_t4", "F01": "p1_online", "F11": "p1_online",
    }
    if dict(result.get("carrier_law_by_cell", {})) != expected_carrier:
        raise ValueError("Part-A carrier-law pre-registration drift")
    hyper = result.get("hyperparameters", {})
    if set(hyper) != {"m4", "m10", "m30"}:
        raise ValueError("Part-A hyperparameter budget topology drift")
    for key, budget in (("m4", 4), ("m10", 10)):
        pinned = SEALED_P1_HYPERPARAMETERS[budget]
        if hyper[key] != dict(pinned):
            raise ValueError(f"Part-A sealed P1 hyperparameter drift at {key}")
    if hyper["m30"]["alpha_M"] != 0.0 or hyper["m30"]["c_M"] is not None:
        raise ValueError("Part-A M30 must inherit the sealed exact-no-op law")
    if hyper["m30"]["thresholds"] != hyper["m10"]["thresholds"]:
        raise ValueError("Part-A M30 threshold inheritance drift")
    gates = result.get("gates", {})
    if gates.get("boundary_epsilon") != GATE_BOUNDARY_EPSILON:
        raise ValueError("Part-A gate boundary epsilon drift")
    if gates.get("per_cell_promotion", {}).get("driving_budget") != DRIVING_BUDGET:
        raise ValueError("Part-A driving budget drift")
    if gates.get("per_cell_promotion", {}).get("cells") != list(CANDIDATE_CELLS):
        raise ValueError("Part-A candidate cell drift")
    if gates.get("additivity", {}).get("slack") != 0.0:
        raise ValueError("Part-A additivity slack drift (the work order binds - 0)")
    if gates.get("safety", {}).get("within_every_budget_floor") != WITHIN_EVERY_BUDGET_FLOOR:
        raise ValueError("Part-A within floor drift")
    if gates.get("safety", {}).get("m30_floor") != M30_FLOOR:
        raise ValueError("Part-A M30 floor drift")
    return result


def dry_plan() -> dict[str, object]:
    return {
        "schema": f"{SCHEMA}_dry_plan",
        "cell": CELL,
        "work_order": WORK_ORDER_RELATIVE,
        "part": "A (DANDI 000688 inference-only factorial; Part B is out of scope)",
        "stage_p_go_anchor": dict(STAGE_P_GO_ANCHOR),
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
