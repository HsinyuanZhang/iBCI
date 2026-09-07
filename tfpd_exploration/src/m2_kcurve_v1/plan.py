"""Frozen constants and pre-registered laws for the M2 labeled-pair k-curve V1.

Authority: operator work order 2026-09-02 (verbatim task): run the
labeled-pair k-curve on M2, CPU-ONLY (``CUDA_VISIBLE_DEVICES=''``, torch
threads 4, 0 workers), inference-only, asking: with the B30 calibration block
FIXED and D-optimal selection at every k, how does performance scale with the
number of labeled pairs k -- where does it saturate relative to the
all-usable (M30-class) endpoint?

* the labeled pool is the sealed first-30 calibration block (unchanged); the
  support at every k is the FIRST k of the frozen greedy forward D-opt order
  over the first-30 finite-angle candidates (the same sealed law that picked
  the sealed M4 supports; every session's usable count is 15, so k=15 IS the
  all-usable endpoint, verified per session and capped at min(usable));
* two deployments share one support per (session, k): STATIC (the sealed
  ``ridge_activity30`` law: activity = the FULL first-30 block, carrier T4
  fit from the k selected pairs only) and CDM (the sealed G-family
  ``m4_activity_only`` law: activity FIFO initialized on the k selected rows
  with capacity 30-k, growing on every completed post-30 query trial, frozen
  canonical fixed-ridge carrier in the Hz domain);
* both families are scored on the same two post-30 surfaces:
  ``within_post30`` and ``external_post30_local``.

Pre-registered readouts (from the work order):

1. the k-curve table per family/surface with paired per-session values;
2. the saturation point k* = smallest k whose equal-session mean is within
   0.005 of the all-usable endpoint (per family/surface);
3. the monotonicity check (any non-monotone dip disclosed; the k10
   chronological artifact should NOT appear since selection is uniform
   D-opt at every k);
4. the verdict strings ``SATURATES_AT_<k*>`` per family/surface.

Sealed machinery is reused BY IMPORT ONLY (``src/m2_t4_activity_budget_screen_v1``
D-opt/digest/R2 laws and isfinite candidate mask, ``src/m2_chrono4_strict_v1``
alternative-support precedent + anchor matchers, ``src/m2_reblock10_v1``
k-row binding precedent, ``src/cdm_p1_m2_local_v1`` G-family executors and
``build_session_material`` surface law).  No frozen code is edited and no
frozen result root is created or modified; sealed anchors are read from
receipts only.
"""

from __future__ import annotations

from pathlib import Path

SCHEMA = "m2_kcurve_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_kcurve_v1"

# ---------------------------------------------------------------------------
# The sealed foundations (verified by SHA-256 at attempt and at launch).
# ---------------------------------------------------------------------------

#: The sealed activity-budget screen: its ``ridge_activity30_m4`` rows are the
#: STATIC family's k=4 anchors (act30 activity + D-opt-4 carrier) and its
#: ``ridge_static_m30`` rows are the all-usable endpoint anchors (act30
#: activity + carrier over every usable first-30 row) on both of its surfaces.
BUDGET_SCREEN_SCORE_RELATIVE = (
    "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json"
)
BUDGET_SCREEN_SCORE_SHA256 = (
    "6bdad93328ba26c12b8aa3afffcbc3490c312dfe22939b3d3bb3c5ec5b9005ce"
)
#: The sealed same-query comparator (the same static rows re-issued; pinned as
#: a cross-receipt agreement disclosure, never as a second anchor source).
COMPARATOR_SCORE_RELATIVE = "tfpd_exploration/results/m2_same_query_comparator_v1/score.json"
COMPARATOR_SCORE_SHA256 = (
    "5efa738ba536ee7cfc624a9dd79fd4ceedb36ae54f482cdcd48e59e355dc3d8b"
)
#: The sealed Native-M2 CDM screen: its ``m4_activity_only`` rows are the CDM
#: family's k=4 anchors (the G00m law) on both G surfaces, and its physical
#: executors ARE the sealed activity-FIFO law on the sealed post-30 stream.
CDM_SCREEN_SCORE_RELATIVE = (
    "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1/score.json"
)
CDM_SCREEN_SCORE_SHA256 = (
    "455485bd854a36392f17ec5ed029b1a4044a01a8dd7dfeae95c7763b8327f5f6"
)
#: The sealed LOCAL-M2 G-family receipts: read-only context proving the CDM
#: k=4 anchor lineage (G00m == the sealed m4_activity_only rows).
G_TERMINAL_RELATIVE = "tfpd_exploration/results/cdm_p1_m2_local_v1/terminal.json"
G_TERMINAL_SHA256 = (
    "414f1e825a0ac0067e7c44cb2b34ea3c024381def041de77c45fc5d7d6ec3c82"
)
#: The sealed packages whose pure laws/executors this run imports verbatim.
SEALED_G_PACKAGE = "tfpd_exploration/src/cdm_p1_m2_local_v1"
CHRONO4_PACKAGE = "tfpd_exploration/src/m2_chrono4_strict_v1"
REBLOCK10_PACKAGE = "tfpd_exploration/src/m2_reblock10_v1"
BUDGET_SCREEN_PACKAGE = "tfpd_exploration/src/m2_t4_activity_budget_screen_v1"

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
ACTIVITY_STACK_LIMIT = 30
BEHAVIOR_SCALE = 5.0
BIN_SECONDS = 0.02
MODEL_BIN_SECONDS = 0.02
RIDGE_NORMALIZED_LAMBDA = 0.1
DEFAULT_BATCH_SIZE = 1024
EXPECTED_WITHIN_SESSIONS = 7
EXPECTED_EXTERNAL_SESSIONS = 6
M4 = 4
M10 = 10
M30 = 30

# ---------------------------------------------------------------------------
# The k-curve law.
# ---------------------------------------------------------------------------

#: The requested labeled-pair grid (the work order's own list).  k=15 is the
#: expected all-usable endpoint (every M2 session carries exactly 15 usable
#: directional trials among its first 30 in the sealed receipts); the binding
#: cap law verifies the per-session usable counts and caps the grid at
#: min(usable) if any session falls short (disclosed, never silent).
K_GRID_REQUESTED = (4, 5, 6, 8, 10, 12, 15)
#: The anchor cardinalities inside the grid: k=4 (the sealed D-opt-4 rows) and
#: k=all-usable (the sealed M30-class endpoint rows).
K_ANCHOR_LOW = 4
K_ALL_USABLE_EXPECTED = 15

KCURVE_LAW = {
    "labeled_pool": (
        "the sealed FIRST-30 completed-calibration-trial pool (the B30 block, "
        "unchanged); candidates are the finite-angle positions within it "
        "(the sealed isfinite mask; 15 per session in the sealed receipts)"
    ),
    "selection": (
        "the frozen greedy forward D-opt law "
        "(sua_exploration.mc_maze.d_optimal_calibration_design."
        "greedy_forward_d_optimal_indices) over the candidate angles; the "
        "support at k is the FIRST k of that greedy order (positions sorted "
        "for binding), so every k-curve cell is a prefix of one frozen order "
        "and the k=4 prefix IS the sealed D-opt-4 selection"
    ),
    "prefix_nesting": (
        "selection(k) must equal the first k of the k=all-usable order on "
        "every session (asserted at runtime), and the first-4 prefix must "
        "equal the sealed receipts' M4 indices EXACTLY (the dopt proof)"
    ),
    "usable_cap": (
        "k never exceeds min over sessions of the usable candidate count; "
        "k = that minimum is the all-usable endpoint cell"
    ),
    "static_deployment": (
        "the sealed ``ridge_activity30`` law verbatim: activity = the FULL "
        "first-30 block (the sealed select_activity_rows act30 branch), "
        "carrier T4 = fit_ridge_t4 (normalized lambda 0.1) over the k "
        "selected pairs only, side = (raw - mean)/std under the frozen "
        "normalizer, identity from the 30-row activity stack, batched "
        "decode, variance-weighted R2"
    ),
    "cdm_deployment": (
        "the sealed G-family ``m4_activity_only`` law verbatim "
        "(m2_precision_cdm_v2_screen_v1 executors): linear-B3S activity FIFO "
        "initialized on the k selected rows with capacity 30-k (the frozen "
        "30-trial stack ceiling), advanced on EVERY completed post-30 query "
        "trial (the sealed V3 short-trial law), frozen canonical fixed-ridge "
        "carrier in the Hz domain fit from the k selected rows"
    ),
    "k_row_binding": (
        "k=4 binds the sealed _build_memory path bit-for-bit; k != 4 uses the "
        "generalized k-row binding (fit_carriers_from_trial_table + "
        "ActivityMemory.initialize(fifo=30-k)), parity-anchored bitwise to "
        "the sealed carrier and initial activity stack at k=4 on every "
        "session (the reblock10 KALL precedent)"
    ),
    "query_stream": (
        "the sealed post-30 stream: scored windows are the sealed "
        "select_common_post30_window_starts partition of each surface's "
        "dataset, and (CDM family) the FIFO advances on every completed "
        "trial at positions >= 30; supports lie inside positions 0-29 so "
        "support/stream overlap is empty by construction (asserted)"
    ),
}

SURFACES = ("within_post30", "external_post30_local")
SURFACE_LAW = {
    "within_post30": (
        "train-dataset window starts filtered to >= trial_start_indices[30] "
        "(the sealed select_common_post30_window_starts law)"
    ),
    "external_post30_local": (
        "held-out-dataset window starts filtered to >= trial_start_indices[30] "
        "(the sealed Native-M2 CDM screen surface; the deployment surface of "
        "the sealed G-family rows)"
    ),
    "static_external_disclosure": (
        "the STATIC family's sealed anchor surface was external_official_query "
        "(unfiltered); this run scores static cells on external_post30_local "
        "per the work order, so the static EXTERNAL anchor is enforced in the "
        "executor-fidelity mode on the sealed unfiltered surface, while the "
        "k=4 and all-usable within_post30 anchors are enforced directly on "
        "the k-curve cells themselves (same surface)"
    ),
    "partition_hygiene": (
        "every cell of both families is scored on exactly the same two "
        "surfaces; within a family/surface/session all k cells carry "
        "IDENTICAL window counts, starts digests and target digests"
    ),
}

CELLS = {
    "STATIC_K{k}": {
        "family": "static",
        "law": KCURVE_LAW["static_deployment"],
        "anchor": (
            "k=4: sealed ridge_activity30_m4 (within_post30 direct, "
            "external_official_query fidelity); k=all-usable: sealed "
            "ridge_static_m30 (within_post30 direct, external_official_query "
            "fidelity)"
        ),
    },
    "CDM_K{k}": {
        "family": "cdm",
        "law": KCURVE_LAW["cdm_deployment"],
        "anchor": (
            "k=4: sealed m4_activity_only == G00m on BOTH surfaces directly "
            "(the CDM family's sealed external surface IS "
            "external_post30_local)"
        ),
    },
}

FROZEN_EVERYTHING_ELSE = {
    "decoder": "the frozen T4 student checkpoint (sha256 pinned by the sealed exporter)",
    "surfaces_and_rosters": "the sealed datasets, 7 within + 6 external sessions",
    "scoring": "each family's sealed variance-weighted R2 law, equal-session means",
    "cdm_valid_mask": (
        "the sealed _raw_fixed_ridge_m30 valid-unit mask (a frozen property of "
        "the session's first-30 rates/labels; reused verbatim)"
    ),
    "short_trial_policy": (
        "no complete 50-bin window contributes no scored window; the activity "
        "FIFO still advances on the completed trial (the sealed V3 law)"
    ),
    "lambda": RIDGE_NORMALIZED_LAMBDA,
    "seeds": [42],
    "parameter_updates": 0,
    "target_gradients": 0,
}

# ---------------------------------------------------------------------------
# Anchors (read from receipts; executor fidelity on a pre-registered roster).
# ---------------------------------------------------------------------------

ANCHORS = {
    "sealed_rows_read_from_receipts": {
        "static_k4": {
            "path": BUDGET_SCREEN_SCORE_RELATIVE,
            "sha256_at_design": BUDGET_SCREEN_SCORE_SHA256,
            "cell": "ridge_activity30_m4",
            "expected_means": {
                "within_post30": 0.677220,
                "external_official_query": 0.290992,
            },
        },
        "static_kall": {
            "path": BUDGET_SCREEN_SCORE_RELATIVE,
            "sha256_at_design": BUDGET_SCREEN_SCORE_SHA256,
            "cell": "ridge_static_m30",
            "expected_means": {
                "within_post30": 0.689261,
                "external_official_query": 0.295220,
            },
        },
        "cdm_k4": {
            "path": CDM_SCREEN_SCORE_RELATIVE,
            "sha256_at_design": CDM_SCREEN_SCORE_SHA256,
            "cell": "m4_activity_only",
            "expected_means": {
                "within_post30": 0.654086,
                "external_post30_local": 0.299057,
            },
        },
        "g_family_context": {
            "path": G_TERMINAL_RELATIVE,
            "sha256_at_design": G_TERMINAL_SHA256,
            "law": (
                "the sealed G00m equal-session means must equal the sealed "
                "m4_activity_only means EXACTLY (receipt-level lineage proof, "
                "read-only context)"
            ),
        },
        "law": "read from receipts, never rerun as anchors",
    },
    "dopt_prefix_proof": {
        "law": (
            "per session and per angle spelling: the first-4 of the frozen "
            "greedy order over the first-30 finite-angle candidates must "
            "equal the sealed receipts' M4 selected_indices EXACTLY, and "
            "selection(k) must equal the first k of the all-usable order for "
            "every grid k (the prefix law)"
        ),
    },
    "endpoint_support_proof": {
        "law": (
            "the k=all-usable support must be exactly the sorted finite-angle "
            "positions of the first-30 pool on every session (the endpoint "
            "support IS every usable row), and its static carrier fit input "
            "set equals the sealed M30 row's usable rows"
        ),
    },
    "direct_surface_anchors": {
        "law": (
            "wherever a k-curve cell's surface coincides with its sealed "
            "anchor row's surface, the CELL ITSELF must reproduce the sealed "
            "row: pure-data fields exact and |Delta R2| <= 1e-5 (CPU law vs "
            "the sealed GPU receipts).  Covered: static k=4 and k=all-usable "
            "on within_post30 (all 7 sessions), CDM k=4 on BOTH surfaces "
            "(all 13 sessions)"
        ),
        "r2_tolerance": 1.0e-5,
    },
    "executor_fidelity": {
        "law": (
            "the static family's EXTERNAL anchor runs in the fidelity mode on "
            "the sealed unfiltered surface (external_official_query): k=4 vs "
            "sealed ridge_activity30_m4 and k=all-usable vs sealed "
            "ridge_static_m30 on ALL 6 external sessions, pure-data fields "
            "exact and |Delta R2| <= 1e-5"
        ),
        "r2_tolerance": 1.0e-5,
    },
    "krow_binding_parity": {
        "law": (
            "driven with the k=4 prefix support, the generalized k-row CDM "
            "binding (fit_carriers_from_trial_table + "
            "ActivityMemory.initialize(fifo=26)) must reproduce the sealed "
            "_build_memory carrier AND the initial activity stack "
            "bit-for-bit on every session (the k>4 path is the sealed "
            "arithmetic, unguarded)"
        ),
    },
    "act30_binding_parity": {
        "law": (
            "for the guard-admissible cardinalities k in {4, 10} the direct "
            "act30 spelling (values[:30]) must equal the sealed "
            "select_activity_rows return bitwise; other k spell the same "
            "branch directly (the sealed helper guards selected-size to "
            "M4/M10/M30 -- disclosed deviation, tested)"
        ),
    },
    "pure_data_pairing": {
        "law": (
            "within a family/surface/session, ALL k cells must carry "
            "IDENTICAL window counts, starts digests and target digests -- "
            "only the support (and the memory it feeds) differs"
        ),
    },
    "cross_family_support_agreement": {
        "law": (
            "the static and CDM families bind IDENTICAL prefix supports per "
            "session (the two angle spellings of one roster agree)"
        ),
    },
}

# ---------------------------------------------------------------------------
# The pre-registered readouts and verdicts.
# ---------------------------------------------------------------------------

READOUT_LAW = {
    "k_curve_table": (
        "per family/surface: equal-session mean and paired per-session R2 at "
        "every grid k, in ascending k order"
    ),
    "saturation": {
        "expression": (
            "k* = the smallest grid k whose equal-session mean is within 0.005 "
            "of the all-usable endpoint mean of the SAME family/surface, in "
            "the deficit sense: endpoint_mean - mean_k <= 0.005 (a mean above "
            "the endpoint has negative deficit and qualifies)"
        ),
        "tolerance": 0.005,
        "endpoint": (
            "the k=all-usable cell of the same family/surface (the measured "
            "endpoint on that surface, never a sealed cross-surface value)"
        ),
    },
    "monotonicity": (
        "adjacent mean deltas over the ascending grid are reported; every "
        "non-monotone dip (negative adjacent delta) is disclosed with its "
        "size; the k10-chronological artifact should NOT appear because "
        "selection is the uniform frozen D-opt law at every k"
    ),
    "verdicts": {
        "format": "SATURATES_AT_<k*>",
        "per_family_surface": True,
        "headline": (
            "the CDM family's external_post30_local verdict is the headline "
            "(the online deployment surface); all four verdicts are reported "
            "and never averaged"
        ),
    },
    "nulls_reported_as_nulls": True,
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
        "tfpd_exploration/results/m2_t4_activity_budget_screen_v1",
        "tfpd_exploration/results/m2_same_query_comparator_v1",
        "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1",
        "tfpd_exploration/results/cdm_p1_m2_local_v1",
        "tfpd_exploration/results/m2_chrono4_strict_v1",
        "tfpd_exploration/results/m2_reblock10_v1",
        "tfpd_exploration/src/cdm_p1_m2_local_v1",
        "tfpd_exploration/src/m2_t4_activity_budget_screen_v1",
        "tfpd_exploration/src/m2_chrono4_strict_v1",
        "tfpd_exploration/src/m2_reblock10_v1",
        "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1",
        "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1",
        "tfpd_exploration/src/m2_same_query_comparator_v1",
    ],
}

HARD_TIMEOUT_SECONDS = 21_600
BATCH_SIZE = DEFAULT_BATCH_SIZE

OWNED_PATHS = (
    "tfpd_exploration/src/m2_kcurve_v1/__init__.py",
    "tfpd_exploration/src/m2_kcurve_v1/plan.py",
    "tfpd_exploration/src/m2_kcurve_v1/laws.py",
    "tfpd_exploration/src/m2_kcurve_v1/physical.py",
    "tfpd_exploration/scripts/run_m2_kcurve_v1.py",
    "tfpd_exploration/tests/test_m2_kcurve_v1.py",
)

PREDECESSOR_RELATIVE = (
    "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json",
    "tfpd_exploration/results/m2_same_query_comparator_v1/score.json",
    "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1/score.json",
    "tfpd_exploration/results/cdm_p1_m2_local_v1/terminal.json",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/__init__.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/plan.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/physical.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/replay.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/__init__.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/plan.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py",
    "tfpd_exploration/src/m2_chrono4_strict_v1/__init__.py",
    "tfpd_exploration/src/m2_chrono4_strict_v1/plan.py",
    "tfpd_exploration/src/m2_chrono4_strict_v1/laws.py",
    "tfpd_exploration/src/m2_chrono4_strict_v1/physical.py",
    "tfpd_exploration/src/m2_reblock10_v1/__init__.py",
    "tfpd_exploration/src/m2_reblock10_v1/plan.py",
    "tfpd_exploration/src/m2_reblock10_v1/laws.py",
    "tfpd_exploration/src/m2_reblock10_v1/physical.py",
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
    "sua_exploration/mc_maze/d_optimal_calibration_design.py",
)


def result_root(repo_root: Path) -> Path:
    return Path(repo_root) / RESULT_ROOT_RELATIVE
