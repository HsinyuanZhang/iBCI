"""Frozen constants and laws for the LOCAL-M2 CDM x P1 transfer (Part B2, V1).

Authority: user 2026-08-31 ("本地 M2 立刻开始") on work order
``docs/WORKORDER_CDM_P1_CROSS_DATASET_V1_20260831.md`` section 3 (B2), on top
of the SEALED prerequisite audit ``results/cdm_p1_m2_v1/audit.json`` whose
verdict is ``NOT_EVALUABLE_OFFICIAL_CONTRACT``.  This route therefore runs on
the LOCAL M2 protocol only: the frozen same-query comparator's surfaces and
rosters, where completed-trial boundaries (``trial_change`` /
``trial_start_indices``) are available locally, and the audit's PASS artifacts
are reused as the foundation:

* prerequisite (b) PASS  -- the T4 anchor law ``solve(A0, b0)`` reproduces the
  sealed ``fit_ridge_t4`` carrier bit-for-bit on all 13 M2 sessions;
* prerequisite (c) PASS  -- the direction law is frozen: continuous centre-out
  target angle snapped to the 8 canonical directions ``-3pi/4 + k*pi/4``;
* prerequisite (a) FAIL  -- the OFFICIAL evaluator hides boundaries, so nothing
  here is an official-contract claim; every row is local-protocol evidence.

Two cell families:

* ``F`` (primary, carrier axis): F00m frozen sealed T4 ridge carrier vs F01m
  P1 online carrier, both on the frozen M2 runtime with the sealed static
  activity law, bit-anchored to the sealed ``m2_same_query_comparator_v1``
  ``t4_ridge_static_*`` rows;
* ``G`` (secondary, full champion stack mirroring DANDI's F01): the sealed
  Native-M2 CDM activity memory (``m2_precision_cdm_v2_screen_v1``
  ``m4_activity_only`` law) with the P1 carrier on top, bit-anchored to the
  sealed activity-only rows on that screen's own surfaces.

P1 hyperparameters are RE-SELECTED on M2's OWN source folds (the within-7
``within_post30`` surface); DANDI's are never ported as claims.
"""

from __future__ import annotations

from pathlib import Path

SCHEMA = "cdm_p1_m2_local_v1"
WORKORDER_RELATIVE = (
    "tfpd_exploration/docs/WORKORDER_CDM_P1_CROSS_DATASET_V1_20260831.md"
)
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/cdm_p1_m2_local_v1"

# ---------------------------------------------------------------------------
# The sealed audit this route builds on (fail-closed predecessor).
# ---------------------------------------------------------------------------

AUDIT_RELATIVE = "tfpd_exploration/results/cdm_p1_m2_v1/audit.json"
AUDIT_SHA256 = "8ee508e6973da25dc27c29f7c245c265ea3231d8579e68de2d091a5c4b0055a3"
AUDIT_VERDICT = "NOT_EVALUABLE_OFFICIAL_CONTRACT"
AUDIT_REUSED = {
    "b_t4_equivalent_carrier": (
        "all_sessions_bitwise_parity = True: solve(A0, b0) -> "
        "[a, c, sqrt(a*a+c*c), b] float32 reproduces the sealed fit_ridge_t4 "
        "carrier on all 13 M2 sessions"
    ),
    "c_direction_parametrization": (
        "continuous centre-out target angle snapped to the 8 canonical "
        "directions (-3pi/4 + k*pi/4); calibration-pool angles available "
        "locally, query angles never read"
    ),
    "a_trial_boundaries": (
        "NOT_EVALUABLE under the official evaluator contract; the LOCAL "
        "protocol reads trial_change/trial_start_indices, which the sealed "
        "local screens also consume"
    ),
}

# ---------------------------------------------------------------------------
# Frozen runtime bindings (identical to the sealed M2 screens).
# ---------------------------------------------------------------------------

T4_CHECKPOINT_SHA256 = (
    "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
)
SPINT_CHECKPOINT_SHA256 = (
    "fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec"
)
NORMALIZATION_SHA256 = (
    "d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e"
)
SAME_QUERY_SCORE_RELATIVE = (
    "tfpd_exploration/results/m2_same_query_comparator_v1/score.json"
)
SAME_QUERY_SCORE_SHA256 = (
    "5efa738ba536ee7cfc624a9dd79fd4ceedb36ae54f482cdcd48e59e355dc3d8b"
)
PARENT_SCORE_RELATIVE = (
    "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json"
)
PARENT_SCORE_SHA256 = (
    "6bdad93328ba26c12b8aa3afffcbc3490c312dfe22939b3d3bb3c5ec5b9005ce"
)
CDM_SCREEN_SCORE_RELATIVE = (
    "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1/score.json"
)
CDM_SCREEN_SCORE_SHA256 = (
    "455485bd854a36392f17ec5ed029b1a4044a01a8dd7dfeae95c7763b8327f5f6"
)

CHANNELS = 96
WINDOW_SIZE = 50
TRIAL_LENGTH = 100
ACTIVITY_HORIZON = 30
BEHAVIOR_SCALE = 5.0
BIN_SECONDS = 0.02
RIDGE_NORMALIZED_LAMBDA = 0.1
DEFAULT_BATCH_SIZE = 1024
EXPECTED_WITHIN_SESSIONS = 7
EXPECTED_EXTERNAL_SESSIONS = 6

# ---------------------------------------------------------------------------
# Surfaces (verbatim mirror of the sealed local screens).
# ---------------------------------------------------------------------------

#: F-family surfaces: exactly the same-query comparator's definitions.
SURFACES_F = ("within_post30", "external_official_query")
#: G-family surfaces: exactly the Native-M2 CDM screen's definitions (its
#: external surface is the post-30 local query, where its rows were sealed).
SURFACES_G = ("within_post30", "external_post30_local")
SURFACE_QUERY_LAW = {
    "within_post30": (
        "train-dataset window_indices filtered to starts >= trial_start_indices[30] "
        "(select_common_post30_window_starts)"
    ),
    "external_official_query": (
        "held-out-dataset window_indices, unfiltered (the sealed official-query row)"
    ),
    "external_post30_local": (
        "held-out-dataset window_indices filtered to starts >= trial_start_indices[30] "
        "(the sealed Native-M2 CDM screen surface)"
    ),
}

# ---------------------------------------------------------------------------
# Budgets, cells, and the online evidence stream.
# ---------------------------------------------------------------------------

BUDGETS_F = (4, 10, 30)
BUDGET_G = 4
M4 = 4
M10 = 10
M30 = 30

#: The calibration pool is the first 30 trials (the comparator's first-30
#: law; the M4 D-opt support and the M10/M30 chronological support are drawn
#: from it).  The online evidence stream therefore begins with completed
#: trial 30: every earlier completed trial belongs to the labeled calibration
#: phase and never enters the bank.
EVIDENCE_START_POSITION = 30

CARRIER_AXIS_LAW = {
    "F00m_anchor": (
        "the sealed same-query decode (whole-session 1024-chunk batching); must "
        "reproduce the sealed t4_ridge_static_* rows bit-exactly on every "
        "surface/budget/session"
    ),
    "F00m": (
        "the sealed t4_ridge_static carrier law (fit_ridge_t4 over the selected "
        "support, side = (raw-mean)/std), per-trial decode -- the gate baseline"
    ),
    "F01m": (
        "the same frozen runtime and static activity, with the P1 online "
        "carrier: anchored block refit under the three-factor gate"
    ),
    "activity_axis": (
        "frozen at the sealed static law (select_activity_rows: selected-M "
        "activity for static cells); the F00m/F01m contrast isolates the carrier"
    ),
}
FULL_STACK_LAW = {
    "G00m": (
        "the sealed Native-M2 CDM activity-only law verbatim "
        "(m2_precision_cdm_v2_screen_v1 m4_activity_only): linear-B3S activity "
        "FIFO advancing on every completed valid trial, frozen canonical "
        "fixed-ridge carrier -- must reproduce the sealed rows bit-exactly"
    ),
    "G01m": (
        "the same activity FIFO with the P1 online carrier on top (the DANDI "
        "F01 mirror: CDM activity memory + P1 carrier)"
    ),
    "g_surfaces": "the sealed CDM screen's own surfaces (within_post30, external_post30_local)",
    "anchor_disclosure": (
        "G00m is bit-anchored to the sealed m4_activity_only rows on both G "
        "surfaces; G01m has no sealed same-law row and is reported as new"
    ),
}

# ---------------------------------------------------------------------------
# The velocity unit law (pre-registered before any GPU run).
# ---------------------------------------------------------------------------

#: M2 covariates are ``finger_vel`` in SI m/s (median query speed ~0.007,
#: median per-trial displacement ~0.006).  The frozen Stage-O/P direction
#: gates (minimum_displacement = minimum_mean_speed = 0.05) were frozen in the
#: mc_maze centimetre family.  The exact SI conversion m/s -> cm/s (x100) is
#: applied to every velocity view BEFORE the frozen direction law; dt is
#: exactly the 20-ms M2 bin.  No gate threshold is rescaled.
VELOCITY_UNIT_LAW = {
    "covariate_source": "finger_vel (SI m/s)",
    "view_scale": 100.0,
    "view_units": "cm/s",
    "dt_seconds": BIN_SECONDS,
    "frozen_gates_untouched": (
        "minimum_movement_bins=3, minimum_displacement=0.05, "
        "minimum_mean_speed=0.05, max_canonical_distance_rad=pi/8 "
        "(cdm_core.CDMDConfig defaults; only support_budget_m is set)"
    ),
    "empirical_check_disclosed": (
        "true query movement directions snap within pi/8 on 100% of labeled "
        "query trials on both rosters (median ~4-5 deg) under this law"
    ),
}

#: The evidence-rate domain law: the F-family anchor is fit in the sealed
#: carrier's own counts-per-20ms-bin domain (spike_sums / trial_lengths), so
#: the evidence rows' scalar rates are mean counts per bin over the completed
#: trial; the G-family anchor is fit in the sealed CDM screen's Hz domain, so
#: its evidence rates are mean counts per bin / BIN_SECONDS.
EVIDENCE_RATE_LAW = {
    "F": "mean counts per 20-ms bin over the completed trial (neural_data[start:stop].mean(axis=0))",
    "G": "mean Hz over the completed trial (neural_data[start:stop].mean(axis=0) / 0.02)",
}

#: Short query trials (fewer than one complete 50-bin window) contribute no
#: direction measurement and no commit; the carrier stays exact.  In the
#: G-family the activity FIFO still advances (the sealed V3 law).
SHORT_TRIAL_POLICY = (
    "no_complete_window_means_no_measurement_no_commit__no_padding_no_"
    "cross_trial_history"
)

# ---------------------------------------------------------------------------
# The transferred P1 law (machinery orchestrated verbatim).
# ---------------------------------------------------------------------------

MACHINERY = {
    "direction_estimator": "src.support_anchored_t4_stage_p_v1.direction_estimator.measure_trial (verbatim)",
    "three_factor_gate": "src.support_anchored_t4_stage_p_v1.gate.evaluate_block / EvidenceBankP (verbatim)",
    "trust_region": "src.support_anchored_t4_stage_o_v1.trust_region.active_carrier (verbatim)",
    "selection_grids": "src.support_anchored_t4_stage_p_v1.replay.enumerate_gate_grid / enumerate_mass_grid / select_vector (verbatim)",
    "f_anchor": "the continuous-angle A0/b0 mirror of fit_ridge_t4 (sqrt modulation, the audit's bitwise law)",
    "g_anchor": "src.support_anchored_t4_stage_o_v1.anchor.SupportAnchor.from_labeled_support (canonical indices, hypot mirror)",
}

SELECTION_LAW = {
    "surface": "within_post30 ONLY (the 7 M2 OWN source sessions); external never selects",
    "budgets": [M4, M10],
    "objective_cell": "P2 (the sealed stage-P selection objective)",
    "stage1": "the 16 pre-registered gate-threshold vectors (tau_d x r_max x d_min x max_mass), rho_M=1.0 alpha_M=0.5 fixed",
    "stage2": "the 8 pre-registered (rho_M, alpha_M) mass vectors under the stage-1 thresholds",
    "c_M": "within-7 median of the per-commit unprojected D2 pooled over the winning configuration (the stage-O law)",
    "tie_break": "first maximum in the pre-registered enumeration order",
    "re_selection_of_dandi_values": "FORBIDDEN -- the DANDI/SUA stage-P values are never ported as claims",
    "disclosure": "selection rollouts run SLIM (scores, bank digests, D2 values only)",
}

M30_NOOP_LAW = {
    "alpha_M": 0.0,
    "rho_M": 1.0,
    "c_M": None,
    "thresholds_inherited_from": "the selected M4 vector (the Part-A M30 inheritance shape)",
    "proof": (
        "zero committed movement: carrier_before == carrier_after on every "
        "trial and the per-trial decode is the F00m loop itself, so F01m@M30 "
        "is bitwise the F00m@M30 row"
    ),
}

# ---------------------------------------------------------------------------
# Gates (the Part-A law mirrored on M2's own rosters).
# ---------------------------------------------------------------------------

GATES = {
    "driving_cell": "F01m",
    "driving_budget": M4,
    "driving_surface": "external_official_query",
    "promotion": {
        "expression": (
            "F01m - F00m >= +0.01 equal-session external_official_query M4 R2 "
            "AND positive external M4 sessions >= 4/6"
        ),
        "delta_floor": 0.01,
        "breadth_min": 4,
        "breadth_denominator": 6,
        "breadth_rationale": (
            "the Part-A breadth was 10/15 (two thirds); the M2 external roster "
            "is 6 sessions, so the mirrored two-thirds breadth is 4/6"
        ),
    },
    "secondary_full_stack": {
        "expression": (
            "G01m - G00m >= +0.01 equal-session external_post30_local M4 R2 "
            "AND positive external M4 sessions >= 4/6"
        ),
        "role": "secondary disclosure; the primary verdict never rests on it",
    },
    "safety": {
        "expression": (
            "F01m - F00m >= -0.02 on within_post30 at EVERY rung (M4/M10/M30) "
            "and on external_official_query M30"
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
    "never_average_m30": "M30 is never averaged with M4/M10 to rescue or reject a low-budget effect",
    "official_contract_disclaimer": (
        "every number here is LOCAL-protocol evidence; the official M2 "
        "evaluation contract hides completed-trial boundaries and the sealed "
        "audit records P1-M2 NOT_EVALUABLE_OFFICIAL_CONTRACT"
    ),
}

# ---------------------------------------------------------------------------
# Anchors.
# ---------------------------------------------------------------------------

ANCHORS = {
    "f00m_vs_sealed_same_query_rows": {
        "path": SAME_QUERY_SCORE_RELATIVE,
        "sha256_at_design": SAME_QUERY_SCORE_SHA256,
        "cells": {
            "within_post30": {
                budget: f"t4_ridge_static_m{budget}" for budget in BUDGETS_F
            },
            "external_official_query": {
                budget: f"t4_ridge_static_m{budget}" for budget in BUDGETS_F
            },
        },
        "law": (
            "F00m_anchor reproduces the sealed row bit-exactly: prediction "
            "digest, target digest, r2, window count, ordered starts digest, "
            "activity digest, selected support, raw and normalized T4 digests"
        ),
    },
    "f_anchor_zero_evidence_parity": {
        "law": (
            "per selected support and session: per-group solve(A0, b0) -> "
            "[a, c, sqrt(a*a+c*c), b] float32 equals the sealed fit_ridge_t4 "
            "carrier bit-for-bit (the audit's prerequisite-(b) law on the "
            "SELECTED M4/M10/M30 supports)"
        ),
    },
    "g00m_vs_sealed_cdm_rows": {
        "path": CDM_SCREEN_SCORE_RELATIVE,
        "law": (
            "G00m reproduces the sealed m2_precision_cdm_v2_screen_v1 "
            "m4_activity_only rows bit-exactly (prediction/target digests, r2, "
            "window count) on within_post30 and external_post30_local"
        ),
    },
    "m30_exact_noop": {
        "law": (
            "F01m@M30 (alpha_M = 0) is bitwise the F00m@M30 row, with "
            "carrier_before == carrier_after on every trial"
        ),
    },
}

# ---------------------------------------------------------------------------
# Process.
# ---------------------------------------------------------------------------

GPU_INDEX = 1
HARD_TIMEOUT_SECONDS = 21_600

ENVIRONMENT_LAW = {
    "cuda_visible_devices": "1",
    "cuda_device_order": "PCI_BUS_ID",
    "cublas_workspace_config": ":4096:8",
    "tf32_matmul": False,
    "tf32_cudnn": False,
    "cudnn_benchmark": False,
    "deterministic_algorithms": True,
    "gpu0_policy": "never touched (another live route owns GPU 0)",
    "python": "/home/xinyuan/miniconda3/envs/spint/bin/python",
    "no_user_site": True,
}

OWNED_PATHS = (
    "tfpd_exploration/src/cdm_p1_m2_local_v1/__init__.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/anchor.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/gates.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/plan.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/replay.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/physical.py",
    "tfpd_exploration/scripts/run_cdm_p1_m2_local_v1.py",
    "tfpd_exploration/tests/test_cdm_p1_m2_local_v1.py",
)


def result_root(repo_root: Path) -> Path:
    return Path(repo_root) / RESULT_ROOT_RELATIVE
