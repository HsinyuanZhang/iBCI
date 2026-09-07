"""Frozen constants and pre-registered laws for the M2 re-blocking cell V1.

Authority: operator work order 2026-09-02 (verbatim task): run the M2
re-blocking cell, CPU-ONLY (``CUDA_VISIBLE_DEVICES=''``, torch threads 4,
0 workers), inference-only, shrinking the labeled calibration pool from the
first-30 positions to the FIRST-10 positions:

* the labeled pool is the first 10 completed calibration trials; the support
  is selected by the frozen greedy D-opt law within the finite-angle (=
  finite target-angle) candidates of those 10 positions (~5 usable), or is
  ALL of those candidates (the KALL row);
* everything after position 10 (0-based) is the query stream: scored windows
  AND online completed-trial evidence (the CDM activity FIFO advances on
  every completed trial of the extended stream).

Pre-registered protocol points (from the work order):

1. the query partition is post-position-10 for EVERY cell (the new post10
   surfaces); sealed post30 R2 values are NEVER mixed into the contrasts --
   the OLD law (D-opt-4-from-30 support) is RE-SCORED on the post10 window
   set for both static and CDM, isolating the query-extension effect from
   the support-change effect;
2. the evidence-stream census is reported per session (completed query
   trials old vs new), proving the 2-3x extension on external;
3. the K4 and KALL rows are BOTH reported, no post-hoc pick.

Sealed machinery is reused BY IMPORT ONLY (``src/m2_t4_activity_budget_screen_v1``
D-opt/digest/R2 laws and isfinite candidate mask, ``src/cdm_p1_m2_local_v1``
G-family CDM activity-FIFO executors and ``build_session_material`` surface
law, ``src/m2_precision_cdm_v2_screen_v1`` / ``src/pseudo_mua_precision_cdm_v2_screen_v1``
activity-only runtime laws, the ``src/m2_chrono4_strict_v1`` precedent for
alternative-support cells).  No frozen code is edited and no frozen result
root is created or modified; sealed anchors are read from receipts only.
"""

from __future__ import annotations

from pathlib import Path

SCHEMA = "m2_reblock10_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_reblock10_v1"

# ---------------------------------------------------------------------------
# The sealed foundations (verified by SHA-256 at attempt and at launch).
# ---------------------------------------------------------------------------

#: The sealed same-query comparator: its ``t4_ridge_static_m4`` rows are the
#: static family's D-opt-4-from-30 anchors (executor fidelity + D-opt proofs).
COMPARATOR_SCORE_RELATIVE = "tfpd_exploration/results/m2_same_query_comparator_v1/score.json"
COMPARATOR_SCORE_SHA256 = (
    "5efa738ba536ee7cfc624a9dd79fd4ceedb36ae54f482cdcd48e59e355dc3d8b"
)
#: The sealed Native-M2 CDM screen: its ``m4_activity_only`` rows are the CDM
#: family's D-opt-4-from-30 anchors, and its physical executors ARE the
#: sealed activity-FIFO law.
CDM_SCREEN_SCORE_RELATIVE = "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1/score.json"
CDM_SCREEN_SCORE_SHA256 = (
    "455485bd854a36392f17ec5ed029b1a4044a01a8dd7dfeae95c7763b8327f5f6"
)
#: The sealed LOCAL-M2 transfer route (G-family executors, constants) and the
#: chrono4 precedent (the alternative-support cell pattern this package
#: mirrors, whose pure laws are imported verbatim).
SEALED_G_PACKAGE = "tfpd_exploration/src/cdm_p1_m2_local_v1"
CHRONO4_PACKAGE = "tfpd_exploration/src/m2_chrono4_strict_v1"

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
# The re-blocking law: the first-10 labeled pool and the post-10 stream.
# ---------------------------------------------------------------------------

#: The new labeled-pool boundary: positions 0-9 (0-based) are the labeled
#: calibration pool; every completed trial from position 10 on belongs to the
#: query stream (scored windows and online evidence).
BLOCK_HORIZON = 10
#: The sealed labeled-pool boundary (the OLD law's pool depth).
OLD_HORIZON = 30

REBLOCK_LAW = {
    "labeled_pool": (
        "the FIRST 10 completed calibration trials (positions 0-9, 0-based); "
        "the finite-angle candidates within those 10 positions are the "
        "support-selection pool (~5 usable under the roster's alternating "
        "directional pattern)"
    ),
    "k4_support": (
        "the frozen greedy forward D-opt law (sua_exploration.mc_maze."
        "d_optimal_calibration_design.greedy_forward_d_optimal_indices) over "
        "the finite-angle candidates of the first-10 pool, selecting 4"
    ),
    "kall_support": (
        "ALL finite-angle candidates within the first-10 pool (the 'not "
        "limited to 4' row); the realized cardinality k is disclosed per "
        "session and must be >= 4"
    ),
    "query_stream": (
        "every completed trial from position 10 on: scored windows are the "
        "dataset window starts >= trial_start_indices[10] on every surface, "
        "and (CDM family) the activity FIFO advances on EVERY completed "
        "query trial of the extended stream, positions 10+"
    ),
    "oldlaw_comparator": (
        "the OLD law (sealed D-opt-4-from-30 support) RE-SCORED on the same "
        "post10 window set / query stream, for both static and CDM; sealed "
        "post30 R2 values are never mixed into any contrast; the CDM "
        "comparator's activity FIFO skips the old support's own positions "
        "(already in memory as the immutable support; the frozen B3S law "
        "rejects duplicate trial ids), while the first-10 cells never overlap "
        "the stream (their supports lie inside positions 0-9)"
    ),
    "evidence_census": (
        "per session: completed query trials under the old boundary "
        "(positions >= 30) vs the new boundary (positions >= 10), the "
        "extension ratio, and the window counts of each partition"
    ),
}

# ---------------------------------------------------------------------------
# Surfaces: every cell scores the post10 partition of its family's dataset.
# ---------------------------------------------------------------------------

SURFACES_STATIC = ("within_post10", "external_post10_query")
SURFACES_CDM = ("within_post10", "external_post10_local")
SURFACE_LAW = {
    "within_post10": (
        "train-dataset window starts filtered to >= trial_start_indices[10] "
        "(the sealed select_common_post30_window_starts law with the boundary "
        "moved from 30 to 10)"
    ),
    "external_post10_query": (
        "held-out-dataset window starts filtered to >= trial_start_indices[10] "
        "(the static family's sealed external_official_query surface with the "
        "boundary moved to 10)"
    ),
    "external_post10_local": (
        "held-out-dataset window starts filtered to >= trial_start_indices[10] "
        "(the sealed CDM-screen external_post30_local surface with the "
        "boundary moved to 10)"
    ),
    "same_window_set_disclosure": (
        "external_post10_query and external_post10_local are the SAME window "
        "set (same dataset, same filter); both spellings are kept so each "
        "family reports on its own sealed surface's name, and the run asserts "
        "the two starts arrays are identical"
    ),
    "partition_hygiene": (
        "no cell ever mixes partitions: every cell (BLOCK10 and OLDLAW alike) "
        "is scored on post10 windows only; sealed post30 values appear ONLY "
        "in the cross-partition sealed-context block and never in a gate"
    ),
}

# ---------------------------------------------------------------------------
# The cells.
# ---------------------------------------------------------------------------

CELLS = {
    "BLOCK10_STATIC_K4": {
        "family": "static",
        "law": (
            "the sealed ``t4_ridge_static_m4`` decode law verbatim "
            "(fit_ridge_t4 with normalized lambda 0.1 over the support, "
            "side = (raw - mean)/std under the frozen normalizer, identity "
            "from the support activity stack, batched decode, "
            "variance-weighted R2) with the support AND activity pool from "
            "the first-10 D-opt law; static deployment over post10 windows"
        ),
        "surfaces": list(SURFACES_STATIC),
    },
    "BLOCK10_STATIC_KALL": {
        "family": "static",
        "law": (
            "the same sealed static law with the support = ALL finite-angle "
            "candidates within the first-10 pool (k rows, k >= 4); the "
            "sealed select_activity_rows guards cardinality to M4/M10/M30, so "
            "the k-row activity stack is the same selected-M arithmetic "
            "spelled directly (disclosed deviation, tested)"
        ),
        "surfaces": list(SURFACES_STATIC),
    },
    "BLOCK10_CDM_K4": {
        "family": "cdm",
        "law": (
            "the sealed ``m4_activity_only`` law verbatim "
            "(m2_precision_cdm_v2_screen_v1 executors: linear-B3S activity "
            "FIFO of capacity 30-4 initialized on the support and advanced on "
            "every completed query trial, frozen canonical fixed-ridge carrier "
            "in the Hz domain) with the support from the first-10 D-opt law; "
            "the FIFO grows over the EXTENDED post10 stream (positions 10+)"
        ),
        "surfaces": list(SURFACES_CDM),
    },
    "BLOCK10_CDM_KALL": {
        "family": "cdm",
        "law": (
            "the same activity-only law with the support = ALL finite-angle "
            "candidates within the first-10 pool (k rows); the B3S stack law "
            "keeps the sealed 30-trial ceiling (support k + FIFO capacity "
            "30-k), and the carrier is the sealed fit_carriers_from_trial_table "
            "fixed-ridge arithmetic over the k support rows (the CDMDConfig "
            "M4/M10/M30 budget guard does not bind the activity-only path; "
            "disclosed deviation, parity-anchored to the sealed binding)"
        ),
        "surfaces": list(SURFACES_CDM),
    },
    "OLDLAW_STATIC": {
        "family": "static",
        "law": (
            "the sealed ``t4_ridge_static_m4`` law VERBATIM (D-opt-4-from-30 "
            "support, the sealed receipt's own indices) re-scored on the "
            "post10 window set of each static surface"
        ),
        "surfaces": list(SURFACES_STATIC),
        "role": "reference comparator on the same post10 windows",
    },
    "OLDLAW_CDM": {
        "family": "cdm",
        "law": (
            "the sealed ``m4_activity_only`` law VERBATIM (D-opt-4-from-30 "
            "support, the sealed receipt's own indices, FIFO capacity 26) with "
            "the FIFO advancing on every completed trial of the extended "
            "post10 stream EXCEPT the old support's own positions: those "
            "trials are already permanently in memory as the immutable "
            "support block, and the frozen B3S memory law rejects a duplicate "
            "trial id (a trial cannot be both calibration support and query "
            "evidence).  On the sealed post30 stream the overlap was empty "
            "(all support positions <= 29 < 30); on the extended stream the "
            "realized skip count is disclosed per session.  Scored windows "
            "still cover the whole post10 window set"
        ),
        "surfaces": list(SURFACES_CDM),
        "role": "reference comparator on the same post10 windows",
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
    "sealed_dopt_rows_read_from_receipts": {
        "static_m4": {
            "path": COMPARATOR_SCORE_RELATIVE,
            "sha256_at_design": COMPARATOR_SCORE_SHA256,
            "cell": "t4_ridge_static_m4",
        },
        "cdm_m4": {
            "path": CDM_SCREEN_SCORE_RELATIVE,
            "sha256_at_design": CDM_SCREEN_SCORE_SHA256,
            "cell": "m4_activity_only",
        },
        "law": "read from receipts, never rerun as anchors; they prove the OLD law's supports",
    },
    "pure_data_pairing": {
        "law": (
            "every OLDLAW row shares its family's post10 pure data: within a "
            "family/surface/session, all cells (BLOCK10 K4/KALL and OLDLAW) "
            "must carry IDENTICAL window counts, starts digests and target "
            "digests -- only the support (and the memory it feeds) differs"
        ),
    },
    "executor_fidelity": {
        "law": (
            "on a pre-registered roster this package's executors, run with the "
            "SEALED D-opt-4-from-30 supports on the SEALED partitions "
            "(within_post30 / external_official_query unfiltered for static; "
            "within_post30 / external_post30_local for CDM), must reproduce "
            "the sealed rows: pure-data fields exact and |Delta R2| <= 1e-5 "
            "(CPU vs the sealed GPU receipts; the chrono4 precedent realized "
            "~1.7e-7)"
        ),
        "roster": (
            "per family: the first two sorted external sessions and the first "
            "sorted within session"
        ),
        "r2_tolerance": 1.0e-5,
    },
    "dopt_selection_proof": {
        "law": (
            "the sealed receipts' M4 support indices are recomputed from the "
            "first-30 angles with the sealed greedy forward D-opt law and must "
            "match EXACTLY (the OLD-law supports this run re-scores)"
        ),
    },
    "kall_binding_parity": {
        "law": (
            "the KALL carrier binding (fit_carriers_from_trial_table over the "
            "support rows) must reproduce the sealed _build_memory carrier "
            "bit-for-bit on every session when driven with the K4 support "
            "(the KALL path is the sealed arithmetic, unguarded)"
        ),
    },
    "cross_family_support_agreement": {
        "law": (
            "the static and CDM families bind IDENTICAL first-10 supports per "
            "session (the two angle spellings of one roster agree)"
        ),
    },
}

# ---------------------------------------------------------------------------
# The pre-registered gates and verdicts.
# ---------------------------------------------------------------------------

GATES = {
    "primary": {
        "expression": (
            "BLOCK10_CDM_K4 - OLDLAW_CDM >= +0.01 external_post10_local "
            "equal-session mean R2 AND positive external sessions >= 4/6"
        ),
        "delta_floor": 0.01,
        "breadth_min": 4,
        "breadth_denominator": EXPECTED_EXTERNAL_SESSIONS,
        "question": (
            "does the extended query stream pay for the smaller selection "
            "pool (the CDM activity FIFO growing over positions 10+ vs the "
            "old support, both on post10 windows)"
        ),
    },
    "secondary": {
        "expressions": {
            "cdm_kall": (
                "BLOCK10_CDM_KALL - OLDLAW_CDM >= +0.01 external_post10_local "
                "with >= 4/6 breadth (the same contrast on the kall support)"
            ),
            "static_k4": (
                "BLOCK10_STATIC_K4 - OLDLAW_STATIC on external_post10_query "
                "(expected NEGATIVE: a static deployment cannot exploit the "
                "extended evidence stream -- reported, never gated)"
            ),
            "static_kall": (
                "BLOCK10_STATIC_KALL - OLDLAW_STATIC on external_post10_query "
                "(expected NEGATIVE -- reported, never gated)"
            ),
        },
        "role": "secondary disclosure; the primary verdict never rests on it",
    },
    "boundary_epsilon": 1.0e-12,
    "boundary_rule": (
        "exact >= on float64 equal-session means; a miss inside the 1e-12 "
        "program epsilon below the floor with breadth met is disclosed as "
        "REBLOCK_BORDERLINE with numbers and never silently passes "
        "(the Stage-O/P convention)"
    ),
    "verdicts": {
        "REBLOCK_CDM_NET_POSITIVE": "delta >= +0.01 exactly AND breadth >= 4/6",
        "REBLOCK_BORDERLINE": (
            "delta within 1e-12 below +0.01 AND breadth >= 4/6 (numbers "
            "disclosed)"
        ),
        "REBLOCK_CDM_NOT_WORTH": "everything else (numbers disclosed)",
    },
    "nulls_reported_as_nulls": True,
    "both_rows_reported": (
        "the K4 and KALL rows are both reported in full; no post-hoc pick"
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
        "tfpd_exploration/results/m2_chrono4_strict_v1",
        "tfpd_exploration/src/cdm_p1_m2_local_v1",
        "tfpd_exploration/src/m2_t4_activity_budget_screen_v1",
        "tfpd_exploration/src/m2_chrono4_strict_v1",
        "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1",
        "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1",
        "tfpd_exploration/src/m2_same_query_comparator_v1",
    ],
}

HARD_TIMEOUT_SECONDS = 21_600
BATCH_SIZE = DEFAULT_BATCH_SIZE

OWNED_PATHS = (
    "tfpd_exploration/src/m2_reblock10_v1/__init__.py",
    "tfpd_exploration/src/m2_reblock10_v1/plan.py",
    "tfpd_exploration/src/m2_reblock10_v1/laws.py",
    "tfpd_exploration/src/m2_reblock10_v1/physical.py",
    "tfpd_exploration/scripts/run_m2_reblock10_v1.py",
    "tfpd_exploration/tests/test_m2_reblock10_v1.py",
)

PREDECESSOR_RELATIVE = (
    "tfpd_exploration/results/m2_same_query_comparator_v1/score.json",
    "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1/score.json",
    "tfpd_exploration/results/m2_chrono4_strict_v1/attempt.json",
    "tfpd_exploration/results/m2_chrono4_strict_v1/replay.json",
    "tfpd_exploration/results/m2_chrono4_strict_v1/terminal.json",
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
