"""Frozen constants and laws for the M2 chrono4 strict-caliber early-start cell.

Authority: operator work order 2026-09-02 (verbatim task): run the
strict-caliber TRUE-EARLY-START cell on M2, CPU-ONLY (``CUDA_VISIBLE_DEVICES=''``,
torch threads 4, 0 workers), asking two questions:

1. what is the true early-start deployment number -- chronological-first-4
   initialization, no D-opt selection, decoding can start as soon as the
   fourth directional calibration trial completes -- and
2. how much does the D-opt selection law contribute over that cell?

Everything is inference-only on frozen weights; the sealed machinery is reused
BY IMPORT ONLY (``src/cdm_p1_m2_local_v1`` G-family executors + surface laws,
``src/m2_t4_activity_budget_screen_v1`` digest/selection laws, the sealed
screens' own physical executors); no frozen code is edited and no frozen result
root is created or modified.  The sealed D-opt rows are READ FROM RECEIPTS and
never rerun as anchors.
"""

from __future__ import annotations

from pathlib import Path

SCHEMA = "m2_chrono4_strict_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_chrono4_strict_v1"

# ---------------------------------------------------------------------------
# The sealed foundations (verified by SHA-256 at attempt and at launch).
# ---------------------------------------------------------------------------

#: The sealed same-query comparator: its ``t4_ridge_static_m{4,10,30}`` rows are
#: the static family's D-opt/chronological anchors (external 0.2227-class).
COMPARATOR_SCORE_RELATIVE = "tfpd_exploration/results/m2_same_query_comparator_v1/score.json"
COMPARATOR_SCORE_SHA256 = (
    "5efa738ba536ee7cfc624a9dd79fd4ceedb36ae54f482cdcd48e59e355dc3d8b"
)
#: The sealed Native-M2 CDM screen: its ``m4_activity_only`` /
#: ``m10_activity_only`` rows are the CDM family's anchors (external
#: 0.2991-class), and its physical executors ARE the sealed activity-FIFO law.
CDM_SCREEN_SCORE_RELATIVE = "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1/score.json"
CDM_SCREEN_SCORE_SHA256 = (
    "455485bd854a36392f17ec5ed029b1a4044a01a8dd7dfeae95c7763b8327f5f6"
)
#: The sealed LOCAL-M2 transfer route whose replay helpers bind the frozen M2
#: runtime and whose constants pin the checkpoints/normalizer.
SEALED_G_PACKAGE = "tfpd_exploration/src/cdm_p1_m2_local_v1"

T4_CHECKPOINT_SHA256 = (
    "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
)
SPINT_CHECKPOINT_SHA256 = (
    "fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec"
)
NORMALIZATION_SHA256 = (
    "d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e"
)

CHANNELS = 96
WINDOW_SIZE = 50
TRIAL_LENGTH = 100
ACTIVITY_HORIZON = 30
BEHAVIOR_SCALE = 5.0
BIN_SECONDS = 0.02
MODEL_BIN_SECONDS = 0.02
RIDGE_NORMALIZED_LAMBDA = 0.1
DEFAULT_BATCH_SIZE = 1024
EXPECTED_WITHIN_SESSIONS = 7
EXPECTED_EXTERNAL_SESSIONS = 6
ACTIVITY_STACK_LIMIT = 30
M4 = 4
M10 = 10
M30 = 30

# ---------------------------------------------------------------------------
# Surfaces: each family is scored on exactly the surfaces where its sealed
# D-opt anchor rows were sealed (the paired same-law comparison surface).
# ---------------------------------------------------------------------------

SURFACES_STATIC = ("within_post30", "external_official_query")
SURFACES_CDM = ("within_post30", "external_post30_local")
SURFACE_LAW = {
    "within_post30": (
        "train-dataset window starts filtered to >= trial_start_indices[30] "
        "(the sealed select_common_post30_window_starts law)"
    ),
    "external_official_query": (
        "held-out-dataset window starts, unfiltered (the sealed comparator "
        "official-query row; the static family's anchor surface)"
    ),
    "external_post30_local": (
        "held-out-dataset window starts filtered to >= trial_start_indices[30] "
        "(the sealed Native-M2 CDM screen surface; the CDM family's anchor "
        "surface)"
    ),
    "pairing_disclosure": (
        "the scored query surfaces are the sealed ones so that every "
        "selection-law delta is a paired same-surface contrast; the early-start "
        "claim is about the CALIBRATION DEPTH (four directional trials instead "
        "of the full first-30 pool), not about rescoring earlier windows"
    ),
}

# ---------------------------------------------------------------------------
# The one law this cell contributes: the chronological-first-4 initialization.
# ---------------------------------------------------------------------------

CHRONO4_SUPPORT_LAW = {
    "law": (
        "the chronological-first-4 initialization is the FIRST FOUR completed "
        "calibration trials of the first-30 pool that carry a direction label "
        "(finite target angle).  When the first four chronological trials are "
        "all directional this is exactly [0, 1, 2, 3] and decoding can start "
        "at trial 5; otherwise the deployment waits for the fourth "
        "directional trial and the realized positions are disclosed"
    ),
    "precedent": (
        "the sealed CDM screen's own chronological M-law "
        "(``_finite_m10_indices``: the first ten finite-direction trials); "
        "both sealed families' T4 fits consume only directional trials, so a "
        "non-directional trial contributes nothing to either initialization"
    ),
    "single_law_for_both_families": (
        "STATIC_CHRONO4 and CDM_CHRONO4 bind the IDENTICAL support indices, "
        "so the static and online cells share one initialization"
    ),
    "strict_branch_disclosure": (
        "per session the count of directional trials among positions 0-3 is "
        "reported, together with the realized support positions and the "
        "earliest decodable trial"
    ),
    "dopt_pool_law": (
        "the sealed D-opt M4 law needs the whole first-30 pool before it can "
        "select (greedy forward D-opt over the finite-angle candidates), so "
        "its earliest decodable trial is 31; the chronological cell's depth "
        "is the position of its fourth directional trial + 1"
    ),
}

# ---------------------------------------------------------------------------
# The cells.
# ---------------------------------------------------------------------------

CELLS = {
    "STATIC_CHRONO4": {
        "law": (
            "the sealed ``t4_ridge_static_m4`` law verbatim (fit_ridge_t4 with "
            "normalized lambda 0.1 over the support, side = (raw - mean) / std "
            "under the frozen normalizer, identity from the support activity "
            "stack, batched decode, variance-weighted R2) with the ONE change: "
            "the support AND the activity pool both come from the "
            "chronological-first-4 law above (no D-opt selection)"
        ),
        "surfaces": list(SURFACES_STATIC),
        "anchor": (
            "sealed t4_ridge_static_m4 rows (D-opt init): external_official_query "
            "equal-session mean 0.2227-class; paired per session"
        ),
        "chronological_reuse": (
            "the sealed t4_ridge_static_m10 / m30 rows ARE the static family's "
            "chronological law at M10/M30 (chronological_first_m) and are read "
            "from the receipt, never rerun"
        ),
        "convention_disclosure": (
            "the sealed static M-law carries non-directional rows inside its "
            "strict chronological support (masked out of the fit, at least "
            "three usable required); at M4 that strict law is NOT evaluable "
            "when fewer than three of positions 0-3 are directional, so this "
            "cell's single pre-registered law skips non-directional rows and "
            "initializes on the first four DIRECTIONAL trials (the sealed CDM "
            "screen's own M-law convention), identically in both families; "
            "every realized support and the strict-branch status are disclosed "
            "per session"
        ),
    },
    "CDM_CHRONO4": {
        "law": (
            "the sealed ``m4_activity_only`` law verbatim "
            "(``m2_precision_cdm_v2_screen_v1`` executors: linear-B3S activity "
            "FIFO of capacity 30 - 4 initialized on the support and advanced on "
            "every completed query trial, frozen canonical fixed-ridge carrier "
            "in the Hz domain) with the ONE change: the carrier support AND the "
            "FIFO's initial four rows both come from the "
            "chronological-first-4 law above (no D-opt selection)"
        ),
        "surfaces": list(SURFACES_CDM),
        "anchor": (
            "sealed m4_activity_only rows (D-opt init): external_post30_local "
            "equal-session mean 0.2991-class; paired per session"
        ),
        "chronological_reuse": (
            "the sealed m10_activity_only rows ARE the CDM family's "
            "chronological law at M10 (first ten finite-direction) and are read "
            "from the receipt, never rerun"
        ),
    },
}

FROZEN_EVERYTHING_ELSE = {
    "decoder": "the frozen T4 student checkpoint (sha256 pinned by the sealed exporter)",
    "surfaces_and_rosters": "the sealed surfaces, 7 within + 6 external sessions",
    "scoring": "the sealed variance-weighted R2 law, equal-session means",
    "cdm_fifo_law": (
        "capacity 30 - 4, advance on EVERY completed query trial including "
        "short trials without a complete 50-bin window (the sealed V3 law)"
    ),
    "lambda": RIDGE_NORMALIZED_LAMBDA,
    "seeds": [42],
    "parameter_updates": 0,
    "target_gradients": 0,
}

# ---------------------------------------------------------------------------
# Anchors and the D-opt selection-law proof.
# ---------------------------------------------------------------------------

ANCHORS = {
    "sealed_dopt_rows_read_from_receipts": {
        "static_m4": {
            "path": COMPARATOR_SCORE_RELATIVE,
            "sha256_at_design": COMPARATOR_SCORE_SHA256,
            "cells": {"within_post30": "t4_ridge_static_m4",
                      "external_official_query": "t4_ridge_static_m4"},
        },
        "cdm_m4": {
            "path": CDM_SCREEN_SCORE_RELATIVE,
            "sha256_at_design": CDM_SCREEN_SCORE_SHA256,
            "cell": "m4_activity_only",
        },
        "law": "read from receipts, never rerun; they are the D-opt baselines",
    },
    "pure_data_pairing": {
        "law": (
            "every CHRONO4 row must reproduce its sealed D-opt counterpart's "
            "pure-data fields exactly (ordered window starts digest, target "
            "digest, window count): same surface, same sessions, same "
            "partition -- only the initialization differs"
        ),
    },
    "executor_fidelity": {
        "law": (
            "on a pre-registered roster this package's executors, run with the "
            "SEALED D-opt supports, must reproduce the sealed D-opt rows: "
            "pure-data fields exact and |Delta R2| <= 1e-5 (CPU vs the sealed "
            "GPU receipts; the sealed CPU precedent realized ~1.7e-7)"
        ),
        "roster": (
            "per family: the first two sorted external sessions and the first "
            "sorted within session"
        ),
        "r2_tolerance": 1.0e-5,
        "disclosure": (
            "this is an executor-fidelity assertion only; the anchor numbers "
            "themselves stay the sealed receipt values"
        ),
    },
    "dopt_selection_proof": {
        "law": (
            "the sealed receipts' M4 selected_indices are recomputed from the "
            "first-30 angles with the sealed greedy forward D-opt law and must "
            "match EXACTLY (the sealed receipts are D-opt by construction); "
            "the indices plus their angular scatter statistics are reported "
            "as the selection-law evidence against the chronological indices"
        ),
    },
}

# ---------------------------------------------------------------------------
# The pre-registered readout and verdict law.
# ---------------------------------------------------------------------------

VERDICT_LAW = {
    "selection_contribution": (
        "sealed D-opt cell minus CHRONO4 cell, per surface, per family, "
        "float64 equal-session means with paired per-session deltas and "
        "breadth (sessions where the D-opt cell is higher)"
    ),
    "thresholds": {"large_strictly_above": 0.02, "moderate_floor": 0.005},
    "classes": {
        "SELECTION_CONTRIBUTION_LARGE": "delta > 0.02",
        "SELECTION_CONTRIBUTION_MODERATE": "0.005 <= delta <= 0.02",
        "SELECTION_CONTRIBUTION_SMALL": "delta < 0.005",
    },
    "headline": (
        "the HEADLINE verdict is the CDM family's EXTERNAL verdict (the "
        "online early-start system on its deployment surface); the static "
        "family's external verdict and both within verdicts are reported "
        "alongside and never averaged into it"
    ),
    "boundary_epsilon": 1.0e-12,
    "boundary_rule": (
        "exact comparisons on float64 deltas; a delta inside the 1e-12 band "
        "of a class boundary is disclosed as within_epsilon_band_of_boundary "
        "and never flips the class (the Stage-O/P convention)"
    ),
    "nulls_reported_as_nulls": True,
    "true_early_start_number": (
        "CDM_CHRONO4 external_post30_local equal-session mean, reported as the "
        "deployable strict-caliber true-early-start figure, together with the "
        "realized earliest decodable trial"
    ),
    "official_contract_disclaimer": (
        "every number is LOCAL-protocol evidence; the official M2 evaluation "
        "contract hides completed-trial boundaries and no official-contract "
        "claim is made"
    ),
}

# ---------------------------------------------------------------------------
# Environment and process (CPU-only, per operator).
# ---------------------------------------------------------------------------

ENVIRONMENT_LAW = {
    "cuda_visible_devices": "",
    "device": "cpu",
    "cpu_only_rationale": (
        "operator instruction 2026-09-02: CPU-ONLY with CUDA_VISIBLE_DEVICES "
        "empty, torch threads 4 and 0 workers"
    ),
    "torch_num_threads": 4,
    "torch_num_interop_threads": 1,
    "blas_threads_env": {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                         "OPENBLAS_NUM_THREADS": "4"},
    "dataloader_workers": 0,
    "dataloader_workers_rationale": (
        "no DataLoader is ever constructed; every dataset access is direct "
        "numpy indexing on the frozen runtime (the sealed screens' law)"
    ),
    "deterministic_algorithms": True,
    "python": "/home/xinyuan/miniconda3/envs/spint/bin/python",
    "no_user_site": True,
    "one_launch_one_process": True,
}

RECEIPT_LAW = {
    "attempt_before_any_data_or_model_access": True,
    "receipt_permissions": "0444 plus a sha256 sidecar per receipt",
    "atomicity": (
        "replay.json and terminal.json are published O_EXCL only after the "
        "whole grid and every anchor assert pass; any failure leaves only the "
        "attempt receipt (fail-closed)"
    ),
    "frozen_roots_never_modified": [
        "tfpd_exploration/results/m2_same_query_comparator_v1",
        "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1",
        "tfpd_exploration/results/m2_t4_activity_budget_screen_v1",
        "tfpd_exploration/results/cdm_p1_m2_local_v1",
        "tfpd_exploration/src/cdm_p1_m2_local_v1",
        "tfpd_exploration/src/m2_t4_activity_budget_screen_v1",
        "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1",
        "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1",
    ],
}

HARD_TIMEOUT_SECONDS = 21_600
BATCH_SIZE = DEFAULT_BATCH_SIZE

OWNED_PATHS = (
    "tfpd_exploration/src/m2_chrono4_strict_v1/__init__.py",
    "tfpd_exploration/src/m2_chrono4_strict_v1/plan.py",
    "tfpd_exploration/src/m2_chrono4_strict_v1/laws.py",
    "tfpd_exploration/src/m2_chrono4_strict_v1/physical.py",
    "tfpd_exploration/scripts/run_m2_chrono4_strict_v1.py",
    "tfpd_exploration/tests/test_m2_chrono4_strict_v1.py",
)

PREDECESSOR_RELATIVE = (
    "tfpd_exploration/results/m2_same_query_comparator_v1/score.json",
    "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1/score.json",
    "tfpd_exploration/results/cdm_p1_m2_local_v1/attempt.json",
    "tfpd_exploration/results/cdm_p1_m2_local_v1/replay.json",
    "tfpd_exploration/results/cdm_p1_m2_local_v1/terminal.json",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/__init__.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/plan.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/physical.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/replay.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/__init__.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/plan.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py",
    "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/__init__.py",
    "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/plan.py",
    "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/physical.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/__init__.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/core.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/plan.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/physical.py",
    "tfpd_exploration/src/m2_same_query_comparator_v1/core.py",
    "tfpd_exploration/src/m2_same_query_comparator_v1/plan.py",
    "tfpd_exploration/src/calibration_budget_comparators_v1.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py",
)


def result_root(repo_root: Path) -> Path:
    return Path(repo_root) / RESULT_ROOT_RELATIVE
