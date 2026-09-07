"""Frozen constants and laws for the CDM x P1 SUA transfer (Part B1, V1).

The SUA surface is the ``sua_exploration`` matched-scorer runtime: the V9
``shared_t4`` checkpoints (seeds 42/43/44) over the 15 external sub-M
sessions, with the sub-C development-validation six as the source (selection)
roster -- the same two rosters Part A used, on a different frozen decoder
family.  Only the P1 carrier law is transferred; no training happens.
"""

from __future__ import annotations

import math
from pathlib import Path

SCHEMA = "cdm_p1_sua_v1"
WORKORDER_RELATIVE = (
    "tfpd_exploration/docs/WORKORDER_CDM_P1_CROSS_DATASET_V1_20260831.md"
)
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/cdm_p1_sua_v1"

# ---------------------------------------------------------------------------
# Frozen runtime bindings (all re-verified from the sealed paired screen).
# ---------------------------------------------------------------------------

V9_ROOT_RELATIVE = (
    "sua_exploration/results/dandi_000688_subm_co_three_arm_v9_formal_20260805"
)
V9_MANIFEST_SHA256 = (
    "f8f9c0371bdf98d72807bb0563041b384c4ed8e39b0b167e6ebde976b4f455b0"
)
V9_CONTRACT_SHA256 = (
    "9f06357a46544312c3a8b405fa6fa8800c2bb5edb5c344764156294e773e1ca6"
)
NWB_ROOT_RELATIVE = "sua_exploration/data/dandi_000688"
LABEL_BUDGET_AGGREGATE_RELATIVE = (
    "sua_exploration/results/dandi_000688_subm_v9_t4_label_budget_v1_full/"
    "aggregate/endpoint_aggregate_torchmetrics151.json"
)
LABEL_BUDGET_AGGREGATE_SHA256 = (
    "12c4aead244631ed55e5a3eae7f99c87c4cfc4131ac37aa1f84aa55e5e0d4cc2"
)
SEALED_PAIRED_SCORE_RELATIVE = (
    "tfpd_exploration/results/sua_paired_activity_budget_screen_v1/score.json"
)
SEALED_ANCHOR_CELL = "ridge_m4_activity30"

VIEW = "sua"
SEEDS = (42, 43, 44)
SELECTION_SEED = 42
EXPECTED_EXTERNAL_SESSIONS = 15
EXPECTED_WITHIN_SESSIONS = 6
HISTORY_BINS = 50
TRIAL_LENGTH = 100
ACTIVITY_HORIZON = 30
SUPPORT_TRIALS = 50
RIDGE_NORMALIZED_LAMBDA = 0.1
BEHAVIOR_SCALE = 5.0
DEFAULT_BATCH_SIZE = 1024
EXPECTED_QUERY_WINDOWS = 708_795

#: The within (source/selection) roster: the paired-view C1 development
#: validation split -- the same six sub-C sessions Part A used as within-6,
#: never backpropagated by the SUA runtime's training.
WITHIN_SESSIONS = (
    "sub-C_ses-CO-20151103",
    "sub-C_ses-CO-20151104",
    "sub-C_ses-CO-20151106",
    "sub-C_ses-CO-20151109",
    "sub-C_ses-CO-20151110",
    "sub-C_ses-CO-20151112",
)

BUDGETS = (4, 30)
M4 = 4
M30 = 30
M4_ACTIVITY_BUDGET = ACTIVITY_HORIZON
M30_ACTIVITY_BUDGET = ACTIVITY_HORIZON
EXTERNAL_ROSTER_KEY = "external"
WITHIN_ROSTER_KEY = "within"

#: The M10 OLS production carrier (grouped-direction ``lstsq`` over per-
#: direction means) is not an ``A0/b0``-solvable bitwise mirror of the sealed
#: row: reproducing it would require a different solver inside the anchored
#: refit and would break the zero-evidence fallback law.  M10 is therefore
#: NOT RUN in this transfer and the sealed M10 SUA row stands unchanged.
M10_DISCLOSURE = (
    "M10 cells are not run: the sealed SUA M10 carrier is the production "
    "grouped-direction OLS (np.linalg.lstsq over per-direction means), whose "
    "anchored A0/b0 mirror cannot reproduce the sealed carrier bit-for-bit "
    "under the stage-P refit solver; the Part-A M10 surface is therefore "
    "represented by the sealed row only"
)

# ---------------------------------------------------------------------------
# The transferred P1 law.
# ---------------------------------------------------------------------------

#: The frozen direction/measurement/gate machinery is ORCHESTRATED, never
#: reimplemented: ``stage_p_gate`` (bank + three-factor gate), ``gate``'s
#: trust region, and ``direction_estimator`` run verbatim.  Only the anchor's
#: support design carries the SUA sealed carrier's own continuous-angle law.
CARRIER_LAW = {
    "F00s": "frozen T4 (the sealed M4 ridge carrier; activity30 law)",
    "F00s_anchor": "the sealed-law whole-session decode that must reproduce the sealed paired-screen row bit-exactly",
    "F01s": "P1 online (anchored block refit under the three-factor gate, hyperparameters re-selected on the SUA within source folds)",
}
SELECTION_LAW = {
    "surface": "within-6 sub-C development-validation ONLY; external-15 never selects",
    "seed": SELECTION_SEED,
    "objective_cell": "P2 (the sealed stage-P selection objective)",
    "stage1": "the 16 pre-registered gate-threshold vectors (tau_d x r_max x d_min x max_mass)",
    "stage2": "the 8 pre-registered (rho_M, alpha_M) mass vectors",
    "c_M": "within-6 median of the per-commit unprojected D2 (the stage-O law)",
    "tie_break": "first maximum in the pre-registered enumeration order",
    "grids_source": "src/support_anchored_t4_stage_p_v1.replay.enumerate_gate_grid / enumerate_mass_grid",
    "re_selection_of_dandi_values": "FORBIDDEN -- the DANDI stage-P rho/alpha/c_M are never ported as claims",
}
M30_NOOP_LAW = {
    "alpha_M": 0.0,
    "rho_M": 1.0,
    "c_M": None,
    "thresholds_inherited_from": "the selected M4 vector (the Part-A M30 inheritance shape)",
    "proof": "zero committed movement: carrier_before == carrier_after on every trial and the per-trial decode is the F00s loop itself, so F01s@M30 is bitwise the F00s@M30 row",
}
GROUP_VALIDITY_LAW = (
    "valid units are finite rows with positive modulation m > 0; invalid units "
    "keep their sealed support rows in every rebuild"
)
EVIDENCE_RATE_LAW = (
    "per-channel mean firing rate (Hz) over the completed trial's "
    "[start_time, stop_time) spike support, the sealed "
    "unit_side_features._pool_trial_rate_matrix law, computed once per session "
    "for every rewarded trial"
)

# ---------------------------------------------------------------------------
# Gates (mirroring Part A on this surface's own rosters).
# ---------------------------------------------------------------------------

GATES = {
    "driving_cell": "F01s",
    "driving_budget": M4,
    "driving_surface": "external",
    "promotion": {
        "expression": (
            "F01s - F00s >= +0.01 equal-session external M4 R2 "
            "(after seed averaging) AND positive external M4 sessions "
            ">= 10/15 after seed averaging"
        ),
        "delta_floor": 0.01,
        "breadth_min": 10,
        "breadth_denominator": 15,
    },
    "safety": {
        "expression": (
            "F01s - F00s >= -0.02 on the within (source) surface at EVERY rung "
            "and on external M30"
        ),
        "within_floor": -0.02,
        "external_m30_floor": -0.02,
    },
    "boundary_epsilon": 1.0e-12,
    "boundary_rule": (
        "exact >= on float64 equal-session means; a miss inside the 1e-12 "
        "program epsilon is disclosed as within_epsilon_band_of_boundary and "
        "never flips the verdict (the Stage-O/P convention)"
    ),
    "nulls_reported_as_nulls": True,
    "never_average_m30": "M30 is never averaged with M4 to rescue or reject a low-budget effect",
}

HARD_TIMEOUT_SECONDS = 21_600
GPU_INDEX = 1

ENVIRONMENT_LAW = {
    "cuda_visible_devices": "1",
    "cuda_device_order": "PCI_BUS_ID",
    "cublas_workspace_config": ":4096:8",
    "tf32_matmul": False,
    "tf32_cudnn": False,
    "cudnn_benchmark": False,
    "deterministic_algorithms": True,
    "gpu0_policy": "never touched (another live route owns GPU 0)",
}


def result_root(repo_root: Path) -> Path:
    return Path(repo_root) / RESULT_ROOT_RELATIVE


def nwb_path(repo_root: Path, session_id: str) -> Path:
    subject = session_id.split("_")[0]
    return (
        Path(repo_root) / NWB_ROOT_RELATIVE / subject
        / f"{session_id}_behavior+ecephys.nwb"
    )
