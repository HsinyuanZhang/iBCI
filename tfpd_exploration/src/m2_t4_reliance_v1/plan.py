"""Frozen constants and laws for the M2 T4-reliance diagnostic.

Authority: operator 2026-09-01 -- ONE inference-time, zero-training, CPU-only
diagnostic on top of the SEALED M2 static family
``results/m2_t4_activity_budget_screen_v1`` (frozen checkpoint + ``ridge_static``
recipe) replayed through the sealed same-query/M2 machinery
``src/cdm_p1_m2_local_v1``.  Scientific question: how much does the DEPLOYED
frozen model actually USE the T4 side feature at decode time -- measured by
perturbing ONLY the ``[N, 4]`` normalized T4 side input to
``compute_identity`` (activity support held verbatim) and scoring the frozen
decoder on the sealed query surfaces.

Mandatory disclosure (binding): an EVAL-TIME perturbation measures the trained
model's reliance on the T4 side feature; it is NOT the train-time ablation
counterfactual.  A model trained without T4 could learn compensating
structure, so these deltas bound the deployed network's use of the input, not
the information value of T4 itself.

This run can NEVER select a hyperparameter, promote a cell, or update any
model/checkpoint: the held-out surfaces are scored post-hoc and every row is
labelled as a post-hoc diagnostic row.
"""

from __future__ import annotations

from pathlib import Path


SCHEMA = "m2_t4_reliance_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_t4_reliance_v1"
AUTHORITY = (
    "operator 2026-09-01: one inference-time CPU-only T4-side-reliance "
    "diagnostic on the sealed m2_t4_activity_budget static family, never "
    "selecting"
)

# ---------------------------------------------------------------------------
# The sealed foundation (read-only; sha256-verified before any data access).
# ---------------------------------------------------------------------------

PARENT_ROOT_RELATIVE = "tfpd_exploration/results/m2_t4_activity_budget_screen_v1"
SAME_QUERY_SCORE_RELATIVE = "tfpd_exploration/results/m2_same_query_comparator_v1/score.json"

#: Files whose sha256 is frozen into the attempt receipt BEFORE any data or
#: model access: the sealed static-family score, the sealed same-query receipt
#: (whose t4_mean_side rows are the pre-seeded T4_ZERO anchor and whose
#: t4_ridge_static rows are the BASELINE anchor), the frozen packages this run
#: imports, and the frozen exporter the runtime reconstructs through.
PREDECESSOR_RELATIVE = (
    "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json",
    SAME_QUERY_SCORE_RELATIVE,
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/__init__.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/plan.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/__init__.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/anchor.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/gates.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/plan.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/replay.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/physical.py",
    "tfpd_exploration/src/m2_same_query_comparator_v1/core.py",
    "tfpd_exploration/src/calibration_budget_comparators_v1.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py",
    "sua_exploration/evalai_t4_m2/export_t4_payload.py",
    "sua_exploration/mc_maze/d_optimal_calibration_design.py",
)

OWNED_PATHS = (
    "tfpd_exploration/src/m2_t4_reliance_v1/__init__.py",
    "tfpd_exploration/src/m2_t4_reliance_v1/plan.py",
    "tfpd_exploration/src/m2_t4_reliance_v1/physical.py",
    "tfpd_exploration/scripts/run_m2_t4_reliance_v1.py",
    "tfpd_exploration/tests/test_m2_t4_reliance_v1.py",
)

# ---------------------------------------------------------------------------
# The diagnostic grid (pre-registered).
# ---------------------------------------------------------------------------

SURFACES = ("within_post30", "external_official_query")
BUDGETS = (4, 10, 30)
EXPECTED_WITHIN_SESSIONS = 7
EXPECTED_EXTERNAL_SESSIONS = 6

CHANNELS = 96
SIDE_DIM = 4
WINDOW_SIZE = 50
BEHAVIOR_SCALE = 5.0

#: The four briefed cells.  Everything except the ``[N, 4]`` normalized side
#: input to ``compute_identity`` is held verbatim to the sealed static-family
#: recipe (frozen checkpoint, D-opt-4/chronological support, ridge T4 fit with
#: normalized lambda 0.1, train-only normalization, cached-identity decode).
CELLS = ("BASELINE", "T4_ZERO", "T4_SHUFFLE_UNITS", "T4_SHUFFLE_COLS")

CELL_LAWS = {
    "BASELINE": (
        "side = the sealed ridge_static_m{budget} normalized T4 carrier, "
        "verbatim (the deployed recipe)"
    ),
    "T4_ZERO": (
        "side = zeros([96, 4]); in normalized space zeros equal the frozen "
        "train-population mean raw T4 (the sealed comparator's t4_mean_side "
        "intervention)"
    ),
    "T4_SHUFFLE_UNITS": (
        "side[i] = sealed_side[UNIT_PERMUTATION[i]] for every unit i: the "
        "96 per-unit T4 rows are permuted across units, so each unit reads "
        "another unit's T4 row; activity, columns and magnitudes are preserved"
    ),
    "T4_SHUFFLE_COLS": (
        "side[:, j] = sealed_side[:, COL_PERMUTATION[j]] for every column j: "
        "the four T4 columns are permuted, destroying the "
        "[a(cos), c(sin), m(modulation), b(baseline)] column semantics while "
        "preserving every per-unit value multiset"
    ),
}

#: The fixed seed-42 unit permutation: the FIRST draw of
#: ``numpy.random.default_rng(42).permutation(96)``.  It is a fixed-point-free
#: permutation (a derangement: 0 of 96 units keep their own T4 row), frozen
#: here as integers so no runtime RNG can drift it.
UNIT_PERMUTATION = (
    57, 21, 75, 18, 33, 40, 93, 27, 51, 39, 2, 25, 24, 50, 94, 95, 76, 59,
    4, 85, 90, 37, 28, 69, 7, 61, 63, 26, 72, 74, 38, 0, 82, 68, 10, 29, 42,
    52, 17, 87, 73, 5, 91, 1, 46, 80, 88, 3, 79, 89, 20, 83, 43, 71, 32, 56,
    23, 92, 70, 15, 48, 55, 60, 45, 9, 62, 31, 16, 44, 78, 67, 34, 30, 58,
    84, 54, 49, 6, 86, 11, 81, 41, 22, 19, 35, 64, 47, 12, 53, 14, 13, 66,
    36, 65, 77, 8,
)

#: The fixed seed-42 column permutation: the FIRST draw of
#: ``numpy.random.default_rng(42).permutation(4)`` = (3, 2, 1, 0), the
#: reversal [a, c, m, b] -> [b, m, c, a].  Non-identity by construction.
COL_PERMUTATION = (3, 2, 1, 0)

PERMUTATION_LAW = {
    "seed": 42,
    "generator": "numpy.random.default_rng(42).permutation(n)",
    "scope": (
        "ONE fixed permutation per axis, drawn once and applied unchanged to "
        "every session, surface and budget cell (no per-session reseeding)"
    ),
    "unit_permutation_length": len(UNIT_PERMUTATION),
    "unit_permutation_fixed_points": 0,
    "column_permutation": list(COL_PERMUTATION),
}

# ---------------------------------------------------------------------------
# Anchors (pre-registered; CPU tolerance law).
# ---------------------------------------------------------------------------

#: The CPU anchor law inherited from the sealed alpha/oracle CPU diagnostics:
#: GPU receipts cannot be reproduced bitwise on CPU (float-reduction order),
#: so every reproduction assert is a per-session R2 tolerance at 1e-5 while
#: window counts, query-start digests and target digests remain bit-exact.
ANCHOR_TOLERANCE_R2 = 1.0e-5

ANCHORS = {
    "carrier_support_binding": {
        "law": (
            "the sealed same-query carrier law re-derived per surface/session/"
            "budget must still bind the sealed t4_ridge_static evidence "
            "(selected support, raw/normalized T4 digests, activity digest) "
            "and the cdm_p1_m2_local_v1 M2SupportAnchor zero-evidence "
            "fallback parity must be bitwise, exactly as the governing "
            "physical driver asserted"
        ),
    },
    "baseline_vs_sealed_static_rows": {
        "law": (
            "every BASELINE row reproduces the sealed same-query "
            "t4_ridge_static_m{budget} row (= the sealed m2_t4_activity_budget "
            "ridge_static parent row) per session: window_count, "
            "ordered_window_starts_sha256 and target_sha256 equal, and "
            "|R2_mine - R2_sealed| < 1e-5 (the CPU tolerance law)"
        ),
    },
    "t4_zero_vs_sealed_t4_mean_rows": {
        "law": (
            "every T4_ZERO row reproduces the sealed comparator's "
            "t4_mean_side_m{budget} row per session (the same intervention "
            "already run on GPU): window_count, ordered_window_starts_sha256 "
            "and target_sha256 equal, and |R2_mine - R2_sealed| < 1e-5"
        ),
    },
    "perturbation_structure_within_run": {
        "law": (
            "inside THIS run: BASELINE side digest equals the sealed "
            "normalized carrier digest; the T4_ZERO side is exactly zeros; "
            "the shuffled sides restore bit-exactly under the inverse "
            "permutation; the unit shuffle preserves the row multiset and "
            "the column shuffle preserves the per-column value multisets"
        ),
    },
}

# ---------------------------------------------------------------------------
# The pre-registered verdict.
# ---------------------------------------------------------------------------

VERDICT_STRONG = "T4_RELIANCE_STRONG"
VERDICT_MODERATE = "T4_RELIANCE_MODERATE"
VERDICT_NEGLIGIBLE = "T4_RELIANCE_NEGLIGIBLE"

VERDICT_LAW = {
    "primary_cell": "T4_ZERO",
    "statistic": (
        "primary = mean over the 6 (surface, budget) grid cells of "
        "equal_session_mean_delta(T4_ZERO - BASELINE) (paired per session, "
        "equal-session means)"
    ),
    VERDICT_STRONG: "primary <= -0.05 (zeroing T4 costs at least 0.05 R2)",
    VERDICT_MODERATE: "-0.05 < primary <= -0.005",
    VERDICT_NEGLIGIBLE: "primary > -0.005 (zeroing T4 costs less than 0.005 R2)",
    "secondary_cells": ("T4_SHUFFLE_UNITS", "T4_SHUFFLE_COLS"),
    "secondary_law": (
        "the same statistic is reported for the two structure shuffles; it "
        "grades WHICH structure the model uses (unit-resolved rows vs column "
        "semantics) but never flips the verdict string"
    ),
    "breadth_law": (
        "breadth is reported, never governing: per (surface, budget, cell) "
        "the count of sessions with delta < 0, and per cell the pooled "
        "negative share over all 78 session-cells"
    ),
    "threshold_grounding_disclosure": (
        "the bands were fixed BEFORE this run's cells executed, with full "
        "knowledge of the sealed comparator's GPU t4_mean_side deltas "
        "(0.048-0.105 R2 across the six grid cells) and the alpha "
        "diagnostic's small-effect scale (~0.002-0.008 R2): below 0.005 the "
        "deployed model is materially T4-indifferent, above 0.05 the T4 path "
        "carries a substantial share of decode quality"
    ),
    "never_governs": (
        "this verdict is a POST-HOC DIAGNOSTIC statement only: it can never "
        "select a hyperparameter, promote a cell, or change any sealed run"
    ),
}

# ---------------------------------------------------------------------------
# Leakage discipline and the mandatory scientific-role disclosure (binding).
# ---------------------------------------------------------------------------

SCIENTIFIC_ROLE = (
    "inference_time_side_feature_perturbation_reliance__not_train_time_ablation"
)

EVAL_TIME_VS_TRAIN_TIME_DISCLOSURE = (
    "eval-time perturbation measures the trained frozen model's USE of the "
    "T4 side feature at decode time, NOT the train-time ablation "
    "counterfactual: a model trained without T4 could learn compensating "
    "structure, so these deltas bound the deployed network's reliance on the "
    "input, not the information value of T4 itself"
)

LEAKAGE_LAW = {
    "all_rows": {
        "target_label_leakage": False,
        "checkpoint_selection_eligible": False,
        "deployment_eligible": False,
        "post_hoc_diagnostic": True,
        "reason": (
            "the side features come only from the sealed first-30 calibration "
            "recipe (calibration-trial direction labels, never query "
            "targets); the perturbations add no label access; the held-out "
            "surfaces are nevertheless scored post-hoc and can never select "
            "or promote anything"
        ),
    },
    "receipt_statement": (
        "this run is a post-hoc diagnostic: it can NEVER select a "
        "hyperparameter, promote a cell, or update any model or checkpoint"
    ),
}

# ---------------------------------------------------------------------------
# Process (CPU-only; the GPUs are owned by other live routes).
# ---------------------------------------------------------------------------

Torch_THREADS = 4
HARD_TIMEOUT_SECONDS = 7_200  # the operator's 2h hard bound
BATCH_SIZE = 1024
DECODE_CHUNK = 4096

ENVIRONMENT_LAW = {
    "device": "cpu",
    "torch_num_threads": Torch_THREADS,
    "omp_threads_env": "OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=4",
    "gpu_policy": (
        "no GPU is touched: CUDA_VISIBLE_DEVICES must be empty; both GPUs "
        "are owned by other live routes; this run is CPU-only by operator "
        "instruction 2026-09-01"
    ),
    "cuda_visible_devices_required": "",
    "tf32_matmul": False,
    "tf32_cudnn": False,
    "cudnn_benchmark": False,
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
        "whole grid and every anchor assert pass; any failure leaves only "
        "the attempt receipt (fail-closed, the governing run's own law)"
    ),
    "frozen_roots_never_modified": [
        "tfpd_exploration/results/m2_t4_activity_budget_screen_v1",
        "tfpd_exploration/results/m2_same_query_comparator_v1",
        "tfpd_exploration/results/cdm_p1_m2_local_v1",
        "tfpd_exploration/src/m2_t4_activity_budget_screen_v1",
        "tfpd_exploration/src/cdm_p1_m2_local_v1",
        "tfpd_exploration/src/m2_same_query_comparator_v1",
        "streaming_calibration_exp",
        "sua_exploration",
    ],
}

DEVIATIONS = [
    {
        "id": "attempt1_digest_fn_abort",
        "disclosure": (
            "the first reserved attempt aborted fail-closed before any cell "
            "was scored or any receipt published: the T4_ZERO anchor compared "
            "the carrier's cdm_core.array_digest activity digest against the "
            "sealed rows' comparator array_sha256 digest (two different digest "
            "laws); the aborted attempt root was renamed to "
            "results/m2_t4_reliance_v1_attempt1_digest_fn_abort and this fresh "
            "attempt was reserved with the corrected sealed-law digest "
            "comparison (the same trap the sealed alpha diagnostic recorded as "
            "its own attempt1)"
        ),
    },
    {
        "id": "cpu_instead_of_gpu",
        "disclosure": (
            "the operator moved this diagnostic to CPU because both GPUs are "
            "owned by other live routes; the parent screen ran on GPU 1"
        ),
    },
    {
        "id": "tolerance_anchor_instead_of_bitwise",
        "disclosure": (
            "CPU float-reduction order cannot bitwise-match GPU receipts, so "
            "the pre-registered anchors are per-session |Delta R2| < 1e-5 "
            "against the sealed rows (the sealed alpha/oracle CPU precedent); "
            "window_count, query-start and target digests remain bit-exact"
        ),
    },
    {
        "id": "eval_time_not_train_time",
        "disclosure": EVAL_TIME_VS_TRAIN_TIME_DISCLOSURE,
    },
]


def result_root(repo_root: Path) -> Path:
    return Path(repo_root) / RESULT_ROOT_RELATIVE
