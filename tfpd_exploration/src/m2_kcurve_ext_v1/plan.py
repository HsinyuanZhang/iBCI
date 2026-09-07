"""Frozen constants and pre-registered laws for the M2 k-curve EXTENSION V1.

Authority: operator work order 2026-09-02 (verbatim task): run the M2
k-curve EXTENSION, CPU-ONLY (``CUDA_VISIBLE_DEVICES=''``, torch threads 4,
0 workers), inference-only, on top of the just-sealed k-curve
(``results/m2_kcurve_v1/``), asking two operator questions:

* Q1: does the external k-decline of the CDM family survive UNBOUNDED
  accumulation?  The sealed curve used the FIFO-cap-30 law (each support row
  displaces query-evidence slots -- the capacity-competition attribution);
  under UNCAPPED (the ``m2_memory_law_scan_v1`` accumulation law, imported
  verbatim) that mechanism vanishes.
* Q2: does the pool size B interact with k?  D-opt prefix selected from the
  finite-angle candidates within the FIRST-B completed calibration trials
  for B in {10, 20, 30} (B30 column = the sealed curve), all cells scored on
  the SAME sealed post-30 window partition (support pool and evidence stream
  differ; windows identical -- unlike the reblock10 cell, which changed the
  partition).

Grid: law in {FIFO30, UNCAPPED} x B in {10,20,30} x k in
{4, min(5,usable_B), 8, min(10,usable_B), all-usable_B}, capped at the
per-session usable within B (the cap is min over sessions; the endpoint cell
k IS that minimum).  The B30/k=4 and B30/all-usable cells must reproduce the
sealed k-curve values (anchors).  STATIC rows only for B30 (the sealed
column, re-measured as anchors) plus B10/B20 k=4 (cheap pool-narrowing
probes); the static family has no memory so Q1 does not apply to it and Q2
needs only a few points -- the static B30 column doubles as the no-memory
carrier-axis control for the Q1 verdict.

Pre-registered readouts (from the work order):

1. UNCAPPED vs FIFO30 paired delta at each (B,k).  Verdict
   ``CAPACITY_COMPETITION_CONFIRMED`` if the external k-decline
   flattens/disappears under UNCAPPED -- operationalized as: on
   external_post30_local at B30, the decline slope (mean at all-usable minus
   mean at k=4) of the UNCAPPED column is not steeper than the STATIC
   no-memory control's decline slope by more than the 0.005 program
   tolerance (the sealed k-curve's own saturation tolerance); otherwise
   ``CARRIER_OVERFIT_DOMINANT``.
2. the B-effect: paired B10/B20 vs B30 at matched k under UNCAPPED (pool
   narrowing cost, selection-quality axis; descriptive, no gate).
3. the best external cell overall (law, B, k) with its value.

Sealed machinery is reused BY IMPORT ONLY (``src/m2_kcurve_v1`` cells and
binding laws, ``src/m2_memory_law_scan_v1`` accumulation law and decode
helpers, ``src/m2_reblock10_v1`` first-B candidate precedent,
``src/cdm_p1_m2_local_v1`` G-family orchestration,
``src/m2_precision_cdm_v2_screen_v1`` /
``src/pseudo_mua_precision_cdm_v2_screen_v1`` activity-only runtime).  No
frozen code is edited and no frozen result root is created or modified;
sealed anchors are read from receipts only.
"""

from __future__ import annotations

from pathlib import Path

SCHEMA = "m2_kcurve_ext_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_kcurve_ext_v1"

# ---------------------------------------------------------------------------
# The sealed foundations (verified by SHA-256 at attempt and at launch).
# ---------------------------------------------------------------------------

#: The sealed k-curve: the B30 column's law, executor (``cdm_k_cell`` /
#: ``static_act30_cell`` / ``_krow_carrier_and_activity``) and anchor rows.
KCURVE_REPLAY_RELATIVE = "tfpd_exploration/results/m2_kcurve_v1/replay.json"
KCURVE_REPLAY_SHA256 = (
    "f3c1a55fb6207162dcfa4761a83fe2904e5b3d434b368168f47c88a13bff1e12"
)
KCURVE_TERMINAL_RELATIVE = "tfpd_exploration/results/m2_kcurve_v1/terminal.json"
KCURVE_TERMINAL_SHA256 = (
    "77c1bef7df115aac25e38327803e9c613fb5520ff943a2997a5df8365de1214c"
)
#: The sealed memory-law scan: the UNCAPPED accumulation law
#: (``memory.FrozenB3SUniformPool``) plus ``_side_tensor`` /
#: ``_predict_with_identity``, and the UNIFORM_UNCAPPED m4 anchor rows.
MEMORY_SCAN_REPLAY_RELATIVE = "tfpd_exploration/results/m2_memory_law_scan_v1/replay.json"
MEMORY_SCAN_REPLAY_SHA256 = (
    "428be13ebbb38b8f85db36a65a6e4e4ac8f40896d11c187850974892fe20f284"
)
#: The sealed reblock10 cell: the independent sealing of the first-10
#: candidate law (its ``k4_support_positions`` anchor this run's B10/k=4
#: support selection; its post10 window partition is NOT used here).
REBLOCK10_REPLAY_RELATIVE = "tfpd_exploration/results/m2_reblock10_v1/replay.json"
REBLOCK10_REPLAY_SHA256 = (
    "2ae7be22265203a888ed5e826254bce39873d725c3ae520483017c127bed2495"
)
#: The sealed packages whose pure laws/executors this run imports verbatim.
KCURVE_PACKAGE = "tfpd_exploration/src/m2_kcurve_v1"
MEMORY_SCAN_PACKAGE = "tfpd_exploration/src/m2_memory_law_scan_v1"
REBLOCK10_PACKAGE = "tfpd_exploration/src/m2_reblock10_v1"
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
ACTIVITY_STACK_LIMIT = 30
BEHAVIOR_SCALE = 5.0
MODEL_BIN_SECONDS = 0.02
RIDGE_NORMALIZED_LAMBDA = 0.1
DEFAULT_BATCH_SIZE = 1024
EXPECTED_WITHIN_SESSIONS = 7
EXPECTED_EXTERNAL_SESSIONS = 6
M4 = 4
M30 = 30

# ---------------------------------------------------------------------------
# The B axis and the k grids.
# ---------------------------------------------------------------------------

#: B = the number of LEADING completed calibration positions (the block
#: horizon).  The candidate pool for support selection at column B is the
#: finite-angle positions within ``theta[:B]`` -- the reblock10 first-10
#: precedent generalized; B=30 IS the sealed first-30 pool, so the B30 column
#: reproduces the sealed k-curve by construction (asserted against receipts).
B_AXIS = (10, 20, 30)
B_REFERENCE = 30
B_PROBE_ROWS_STATIC = (10, 20)

#: The requested k grid template (the work order's own list, resolved against
#: the column's usable cap): {4, min(5,usable_B), 8, min(10,usable_B),
#: all-usable_B}.  At B=30 with the roster's 15 usable this is exactly
#: {4,5,8,10,15}; at B=20 (10 usable) {4,5,8,10}; at B=10 (5 usable) {4,5}.
K_GRID_TEMPLATE = ("4", "min(5,usable_B)", "8", "min(10,usable_B)", "all-usable_B")
K_FLOOR = M4

B_POOL_LAW = {
    "b_axis": (
        "B in {10,20,30} is the LEADING completed-calibration block horizon; "
        "the candidate pool is the finite-angle positions within the first-B "
        "positions (the sealed isfinite mask over theta[:B], the reblock10 "
        "first-10 precedent)"
    ),
    "usable_within_b": (
        "the per-session count of finite-angle positions among the first B; "
        "the census is disclosed per (family, surface, session, B) and the "
        "column's cap is the MINIMUM over the roster"
    ),
    "selection": (
        "the frozen greedy forward D-opt law "
        "(sua_exploration.mc_maze.d_optimal_calibration_design."
        "greedy_forward_d_optimal_indices) over the first-B candidates; the "
        "support at k is the FIRST k of that one frozen per-(session,B) order "
        "(positions sorted for binding), so every cell is a prefix of one "
        "order (the sealed k-curve prefix law at B=30)"
    ),
    "b30_identity": (
        "the B=30 column's candidate pool, greedy order and every grid "
        "support must equal the sealed k-curve receipts EXACTLY (orders, "
        "k=4 supports and all-usable supports asserted per session)"
    ),
    "partition": (
        "every cell of every column is scored on the SEALED post-30 window "
        "partition (the k-curve/post30 authority); supports lie inside "
        "positions 0..29 and the evidence stream starts at position 30 for "
        "every B, so support/stream overlap is empty by construction and the "
        "windows are IDENTICAL across all cells (unlike reblock10)"
    ),
}

# ---------------------------------------------------------------------------
# The two accumulation laws (Q1's contrast).
# ---------------------------------------------------------------------------

LAWS = ("FIFO30", "UNCAPPED")

MEMORY_LAWS = {
    "FIFO30": {
        "law": (
            "the sealed k-curve CDM law verbatim "
            "(src.m2_kcurve_v1.physical.cdm_k_cell, imported): linear-B3S "
            "activity FIFO initialized on the k selected rows with capacity "
            "30-k, advancing on EVERY completed post-30 query trial; k=4 "
            "through the sealed _build_memory path, k!=4 through the "
            "generalized k-row binding (parity-anchored bitwise at k=4)"
        ),
        "eviction": "oldest completed trial evicted once the FIFO is full "
                    "(capacity 30-k)",
        "origin": KCURVE_PACKAGE,
    },
    "UNCAPPED": {
        "law": (
            "the m2_memory_law_scan_v1 accumulation law imported VERBATIM "
            "(memory.FrozenB3SUniformPool + physical._side_tensor + "
            "physical._predict_with_identity): the frozen B3S encoder's own "
            "streaming accumulation (reset_stream/push_trial/finalize_"
            "identity), the k support rows first, then every completed "
            "post-30 query trial in arrival order, NO eviction ever; the "
            "carrier is the SAME (B,k) carrier binding as the FIFO30 cell "
            "(k=4 the sealed _build_memory carrier, k!=4 the k-row "
            "fixed-ridge-by-trial fit)"
        ),
        "eviction": "none, ever",
        "origin": MEMORY_SCAN_PACKAGE,
    },
}

#: The structural census the verdict interpretation leans on, disclosed at
#: execute from measured advance-event counts (never assumed): the external
#: roster carries at most 13 completed post-30 trials per session while the
#: FIFO capacity is 30-k >= 15 on the whole B30 grid, so NO eviction can
#: occur on the external surface at any grid k -- there the two laws must be
#: bitwise identical (asserted as the external law-equivalence anchor), and
#: the Q1 contrast lives on the within surface (174-309 completed trials).
LAW_EQUIVALENCE_DISCLOSURE = (
    "on external_post30_local the FIFO30 and UNCAPPED cells must agree "
    "bitwise whenever the completed-trial count never exceeds the FIFO "
    "capacity 30-k (no eviction can have occurred); this is measured per "
    "cell from the capacity census and asserted, never assumed"
)

SURFACES = ("within_post30", "external_post30_local")
SURFACE_LAW = {
    "within_post30": (
        "the sealed post-30 partition of the train dataset (the k-curve/"
        "budget-screen select_common_post30_window_starts law); SECONDARY "
        "surface"
    ),
    "external_post30_local": (
        "the sealed post-30 partition of the held-out dataset (the sealed "
        "Native-M2 CDM screen surface; the deployment surface); PRIMARY "
        "surface"
    ),
    "partition_hygiene": (
        "every cell of every (law, B, k, family) is scored on exactly the "
        "same two partitions; within a surface/session ALL cells carry "
        "IDENTICAL window counts, starts digests and target digests "
        "(asserted; the paired-delta pairing law)"
    ),
}

FAMILIES = ("cdm", "static")

CELLS = {
    "CDM_{law}_B{B}_K{k}": {
        "family": "cdm",
        "law": MEMORY_LAWS,
        "grid": "law x B x k (the full grid)",
    },
    "STATIC_B{B}_K{k}": {
        "family": "static",
        "law": (
            "the sealed static act30 law verbatim "
            "(src.m2_kcurve_v1.physical.static_act30_cell, imported): "
            "activity = the FULL first-30 block (unchanged by B and k), "
            "carrier T4 fit from the k selected pairs only"
        ),
        "grid": (
            "B30: every grid k (the sealed column, re-measured as anchors); "
            "B10/B20: k=4 only (the cheap pool-narrowing probes)"
        ),
    },
}

FROZEN_EVERYTHING_ELSE = {
    "decoder": "the frozen T4 student checkpoint (sha256 pinned by the sealed exporter)",
    "surfaces_and_rosters": "the sealed datasets, 7 within + 6 external sessions",
    "carrier": (
        "the sealed fixed-ridge-by-trial carrier fit over the (B,k) support "
        "rows (Hz domain); no carrier adaptation in any cell"
    ),
    "scoring": "each family's sealed variance-weighted R2 law, equal-session means",
    "cdm_valid_mask": (
        "the sealed _raw_fixed_ridge_m30 valid-unit mask (a frozen property of "
        "the session's first-30 rates/labels; reused verbatim)"
    ),
    "short_trial_policy": (
        "the sealed V3 law: every completed post-30 trial advances the "
        "activity memory, including trials with no complete 50-bin window"
    ),
    "lambda": RIDGE_NORMALIZED_LAMBDA,
    "seeds": [42],
    "parameter_updates": 0,
    "target_gradients": 0,
}

# ---------------------------------------------------------------------------
# Anchors (read from receipts; every hard anchor is a stop condition).
# ---------------------------------------------------------------------------

ANCHORS = {
    "fifo30_b30_vs_sealed_kcurve_rows": {
        "path": KCURVE_REPLAY_RELATIVE,
        "sha256_at_design": KCURVE_REPLAY_SHA256,
        "cells": "CDM_K{k} for every B30 grid k (both surfaces, all sessions)",
        "law": (
            "pure-data fields exact, R2 exact within 1e-9 (CPU==CPU on the "
            "pinned environment) and prediction digests EQUAL (the same "
            "imported executor on the same machine)"
        ),
    },
    "static_b30_vs_sealed_kcurve_rows": {
        "path": KCURVE_REPLAY_RELATIVE,
        "sha256_at_design": KCURVE_REPLAY_SHA256,
        "cells": "STATIC_K{k} for every B30 grid k (both surfaces, all sessions)",
        "law": "the same exact-CPU matcher as the FIFO30 anchor",
    },
    "uncapped_b30_k4_vs_sealed_memory_scan_rows": {
        "path": MEMORY_SCAN_REPLAY_RELATIVE,
        "sha256_at_design": MEMORY_SCAN_REPLAY_SHA256,
        "cells": "UNIFORM_UNCAPPED m4 rows (both surfaces, all sessions)",
        "law": (
            "the import-verbatim UNCAPPED law at B30/k=4 (the sealed M4 "
            "support) must reproduce the sealed UNIFORM_UNCAPPED m4 rows: "
            "pure-data fields exact, R2 within 1e-9, prediction digests equal"
        ),
    },
    "b30_support_identity": {
        "law": (
            "the B30 greedy orders, k=4 supports and all-usable supports must "
            "equal the sealed k-curve receipts EXACTLY per session (both "
            "family spellings)"
        ),
    },
    "b10_k4_support_vs_reblock10": {
        "path": REBLOCK10_REPLAY_RELATIVE,
        "sha256_at_design": REBLOCK10_REPLAY_SHA256,
        "law": (
            "the B10/k=4 supports must equal the sealed reblock10 "
            "k4_support_positions per session (an independent sealing of the "
            "same first-10 candidate law; both family spellings)"
        ),
    },
    "krow_binding_parity_b30_k4": {
        "law": (
            "driven with the B30/k=4 prefix support, the generalized k-row "
            "binding must reproduce the sealed _build_memory carrier AND the "
            "initial activity stack bit-for-bit on every session (the sealed "
            "k-curve parity law, re-run on this grid)"
        ),
    },
    "external_law_equivalence": {
        "law": (
            "on external_post30_local, wherever the capacity census shows the "
            "completed-trial count never exceeds the FIFO capacity 30-k, the "
            "FIFO30 and UNCAPPED cells must agree bitwise (prediction "
            "digests and R2 equal) -- the end-to-end executor-equivalence "
            "anchor for the import-verbatim UNCAPPED law"
        ),
    },
    "cross_family_support_agreement": {
        "law": (
            "the static and CDM spellings bind IDENTICAL (B,k) supports per "
            "session at every B (the two angle spellings of one roster)"
        ),
    },
    "pure_data_pairing": {
        "law": (
            "within a family/surface/session, ALL cells (both laws, every B "
            "and k) carry IDENTICAL window counts, starts digests and target "
            "digests -- only the support (and the memory it feeds) differs"
        ),
    },
    "exact_cpu_matcher": {
        "r2_tolerance": 1.0e-9,
        "law": (
            "pure-data fields (starts digest, target digest, window count) "
            "exact; |Delta R2| <= 1e-9; prediction digest equality required "
            "and reported (same executor, same pinned CPU environment)"
        ),
    },
}

# ---------------------------------------------------------------------------
# The pre-registered readouts and verdicts.
# ---------------------------------------------------------------------------

READOUT_LAW = {
    "q1_law_contrast": {
        "paired_delta": (
            "UNCAPPED minus FIFO30 paired per-session delta at every (B,k) "
            "on BOTH surfaces, plus the equal-session mean delta and breadth"
        ),
        "decline_slope": (
            "on external_post30_local at B30: the decline slope is the "
            "equal-session mean at all-usable minus the mean at k=4, for "
            "FIFO30, UNCAPPED and the STATIC no-memory control"
        ),
        "verdict_rule": (
            "CAPACITY_COMPETITION_CONFIRMED iff "
            "slope_UNCAPPED >= slope_STATIC - 0.005 (the uncapped external "
            "k-decline is not steeper than the static no-memory control's "
            "decline by more than the 0.005 program tolerance, i.e. removing "
            "the capacity cap flattens the decline to the carrier-axis "
            "background); otherwise CARRIER_OVERFIT_DOMINANT (the decline "
            "survives unbounded accumulation)"
        ),
        "tolerance": 0.005,
        "disclosures": (
            "the capacity census per session (completed trials vs FIFO "
            "capacity 30-k), the external law-invariance measurement and the "
            "within-surface deltas are all reported; the verdict is computed "
            "from the pre-registered rule only"
        ),
    },
    "q2_pool_narrowing": {
        "expression": (
            "paired (B10 - B30) and (B20 - B30) deltas at every MATCHED k "
            "present in both columns' grids, under UNCAPPED, on both surfaces"
        ),
        "role": (
            "descriptive pool-narrowing cost (selection-quality axis); no "
            "gate, no selection; every matched k reported"
        ),
    },
    "q3_best_cell": {
        "expression": (
            "argmax over ALL CDM (law, B, k) cells of the "
            "external_post30_local equal-session mean, with its value; ties "
            "broken by (law order, B ascending, k ascending); the full "
            "ranking is reported, no claim beyond description"
        ),
    },
    "nulls_reported_as_nulls": True,
    "official_contract_disclaimer": (
        "every number is LOCAL-protocol evidence; the official M2 evaluation "
        "contract hides completed-trial boundaries and no official-contract "
        "claim is made"
    ),
}

VERDICTS = ("CAPACITY_COMPETITION_CONFIRMED", "CARRIER_OVERFIT_DOMINANT")

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
        "tfpd_exploration/results/m2_kcurve_v1",
        "tfpd_exploration/results/m2_memory_law_scan_v1",
        "tfpd_exploration/results/m2_reblock10_v1",
        "tfpd_exploration/results/m2_t4_activity_budget_screen_v1",
        "tfpd_exploration/results/m2_same_query_comparator_v1",
        "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1",
        "tfpd_exploration/results/cdm_p1_m2_local_v1",
        "tfpd_exploration/src/m2_kcurve_v1",
        "tfpd_exploration/src/m2_memory_law_scan_v1",
        "tfpd_exploration/src/m2_reblock10_v1",
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
    "tfpd_exploration/src/m2_kcurve_ext_v1/__init__.py",
    "tfpd_exploration/src/m2_kcurve_ext_v1/plan.py",
    "tfpd_exploration/src/m2_kcurve_ext_v1/laws.py",
    "tfpd_exploration/src/m2_kcurve_ext_v1/physical.py",
    "tfpd_exploration/scripts/run_m2_kcurve_ext_v1.py",
    "tfpd_exploration/tests/test_m2_kcurve_ext_v1.py",
)

PREDECESSOR_RELATIVE = (
    "tfpd_exploration/results/m2_kcurve_v1/attempt.json",
    "tfpd_exploration/results/m2_kcurve_v1/replay.json",
    "tfpd_exploration/results/m2_kcurve_v1/terminal.json",
    "tfpd_exploration/results/m2_memory_law_scan_v1/attempt.json",
    "tfpd_exploration/results/m2_memory_law_scan_v1/replay.json",
    "tfpd_exploration/results/m2_memory_law_scan_v1/terminal.json",
    "tfpd_exploration/results/m2_reblock10_v1/attempt.json",
    "tfpd_exploration/results/m2_reblock10_v1/replay.json",
    "tfpd_exploration/results/m2_reblock10_v1/terminal.json",
    "tfpd_exploration/src/m2_kcurve_v1/__init__.py",
    "tfpd_exploration/src/m2_kcurve_v1/plan.py",
    "tfpd_exploration/src/m2_kcurve_v1/laws.py",
    "tfpd_exploration/src/m2_kcurve_v1/physical.py",
    "tfpd_exploration/src/m2_memory_law_scan_v1/__init__.py",
    "tfpd_exploration/src/m2_memory_law_scan_v1/plan.py",
    "tfpd_exploration/src/m2_memory_law_scan_v1/gates.py",
    "tfpd_exploration/src/m2_memory_law_scan_v1/memory.py",
    "tfpd_exploration/src/m2_memory_law_scan_v1/physical.py",
    "tfpd_exploration/src/m2_reblock10_v1/__init__.py",
    "tfpd_exploration/src/m2_reblock10_v1/plan.py",
    "tfpd_exploration/src/m2_reblock10_v1/laws.py",
    "tfpd_exploration/src/m2_reblock10_v1/physical.py",
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
