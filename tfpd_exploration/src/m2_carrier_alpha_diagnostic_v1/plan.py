"""Frozen constants and laws for the M2 P1-carrier alpha_M post-hoc diagnostic.

Authority: operator 2026-09-01 -- ONE post-hoc, inference-only, zero-training
diagnostic on top of the SEALED governing run ``results/cdm_p1_m2_local_v1``
(attempt/replay/terminal sealed).  The governing run selected ``alpha_M = 0``
on the within-7 ``within_post30`` folds (its stage-2 mass grid scanned
``rho_M in {0.5, 1.0}`` x ``alpha_M in {0, 0.125, 0.25, 0.5}`` and the
within means fell monotonically in alpha for both budgets) and executed F01m
as a bitwise-equal F00m no-op.  This diagnostic replays the SAME F01m carrier
law at ``alpha_M in {0.125, 0.5}`` (plus ``alpha_M = 0`` as the reproduction
anchor) on BOTH surfaces -- the held-out ``external_official_query`` roster
and the within ``within_post30`` roster -- with every other hyperparameter
bound verbatim to the governing selection, to test whether the within-fold
choice of alpha=0 matched held-out reality.

This run can NEVER select a hyperparameter, promote a cell, or update any
model/checkpoint: every ``alpha_M > 0`` row is a target-label-leaking
post-hoc row (the held-out scores are read after the governing verdict was
sealed) and is labelled as such; nothing here feeds any selection.

CPU note (operator 2026-09-01): the replay runs on CPU with
``torch.set_num_threads(4)``; GPU 0 and GPU 1 are both owned by other live
routes and are never touched.  Because CPU float-reduction order cannot
bitwise-match the GPU receipts, the pre-registered anchor is a TOLERANCE
anchor (per-session |Delta R2| < 1e-5 against the sealed rows) rather than a
bitwise anchor; the diagnostic's question -- the SIGN and MONOTONICITY of the
alpha effect, which lives at the 0.005-0.05 R2 scale -- is unaffected.
"""

from __future__ import annotations

from pathlib import Path

SCHEMA = "m2_carrier_alpha_diagnostic_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_carrier_alpha_diagnostic_v1"
AUTHORITY = (
    "operator 2026-09-01: one post-hoc inference-only alpha_M replay of the "
    "sealed cdm_p1_m2_local_v1 governing run, CPU-only, never selecting"
)

# ---------------------------------------------------------------------------
# The sealed foundation (read-only; verified by sha256 before any data access).
# ---------------------------------------------------------------------------

GOVERNING_ROOT_RELATIVE = "tfpd_exploration/results/cdm_p1_m2_local_v1"
GOVERNING_ATTEMPT_SHA256 = (
    "28440774a260d6f9ee916b2533bc4be892b7e5f2d8c2a68eb500b6d2606c33b2"
)
GOVERNING_REPLAY_SHA256 = (
    "2dcd92fed7ebbf544f903c250758281e0f42e259f9bec5439a290696843be0b7"
)
GOVERNING_TERMINAL_SHA256 = (
    "414f1e825a0ac0067e7c44cb2b34ea3c024381def041de77c45fc5d7d6ec3c82"
)
AUDIT_RELATIVE = "tfpd_exploration/results/cdm_p1_m2_v1/audit.json"
AUDIT_SHA256 = "8ee508e6973da25dc27c29f7c245c265ea3231d8579e68de2d091a5c4b0055a3"

#: Files whose sha256 is frozen into the attempt receipt BEFORE any data or
#: model access (the sealed receipts, the frozen packages this run imports,
#: and the frozen exporter the runtime reconstructs through).
PREDECESSOR_RELATIVE = (
    "tfpd_exploration/results/cdm_p1_m2_local_v1/attempt.json",
    "tfpd_exploration/results/cdm_p1_m2_local_v1/replay.json",
    "tfpd_exploration/results/cdm_p1_m2_local_v1/terminal.json",
    "tfpd_exploration/results/cdm_p1_m2_v1/audit.json",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/__init__.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/anchor.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/gates.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/plan.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/replay.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/physical.py",
    "tfpd_exploration/scripts/run_cdm_p1_m2_local_v1.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/anchor.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/trust_region.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/gate.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/replay.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/direction_estimator.py",
    "tfpd_exploration/src/support_anchored_t4_stage_p_v1/plan.py",
    "tfpd_exploration/src/m2_same_query_comparator_v1/core.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py",
    "tfpd_exploration/src/calibration_budget_comparators_v1.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py",
    "sua_exploration/evalai_t4_m2/export_t4_payload.py",
)

OWNED_PATHS = (
    "tfpd_exploration/src/m2_carrier_alpha_diagnostic_v1/__init__.py",
    "tfpd_exploration/src/m2_carrier_alpha_diagnostic_v1/plan.py",
    "tfpd_exploration/src/m2_carrier_alpha_diagnostic_v1/physical.py",
    "tfpd_exploration/scripts/run_m2_carrier_alpha_diagnostic_v1.py",
    "tfpd_exploration/tests/test_m2_carrier_alpha_diagnostic_v1.py",
)

# ---------------------------------------------------------------------------
# The governing selection, bound VERBATIM (frozen from the sealed terminal
# receipt; the execute stage additionally re-derives them from the sealed
# replay.json and fails closed on any drift).
# ---------------------------------------------------------------------------

GOVERNING_M4 = {
    "rho_M": 0.5,
    "alpha_M": 0.0,
    "c_M": 0.2677145247749466,
    "thresholds": {
        "tau_d_rad": 0.7853981633974483,
        "r_max_repetition_per_direction": 4,
        "d_min_distinct_directions": 4,
        "max_pseudo_mass_relative_to_support_rows": 1.0,
    },
}
GOVERNING_M10 = {
    "rho_M": 0.5,
    "alpha_M": 0.0,
    "c_M": 0.2695510718175138,
    "thresholds": {
        "tau_d_rad": 0.39269908169872414,
        "r_max_repetition_per_direction": 8,
        "d_min_distinct_directions": 3,
        "max_pseudo_mass_relative_to_support_rows": 2.0,
    },
}
GOVERNING_BY_BUDGET = {4: GOVERNING_M4, 10: GOVERNING_M10}

# ---------------------------------------------------------------------------
# The diagnostic cells (pre-registered).
# ---------------------------------------------------------------------------

#: The alpha axis: the anchor (the governing value) plus the two replay
#: points.  Both scan points are elements of the sealed stage-2 mass grid's
#: ALPHA_M_CANDIDATES = (0.0, 0.125, 0.25, 0.5); nothing outside the
#: pre-registered grid is ever run.
ALPHA_ANCHOR = 0.0
ALPHA_SCAN = (0.125, 0.5)
ALPHA_ALL = (ALPHA_ANCHOR,) + ALPHA_SCAN

SURFACES = ("within_post30", "external_official_query")
HELD_OUT_SURFACE = "external_official_query"
SELECTION_SURFACE = "within_post30"
BUDGETS = (4, 10)
M4 = 4
M10 = 10

EXPECTED_WITHIN_SESSIONS = 7
EXPECTED_EXTERNAL_SESSIONS = 6

CELL_FAMILIES = {
    "D": {
        "name": "governing_law_alpha_replay",
        "law": (
            "rollout_session_f(cell=F01m/p1_online, row_spec=ROW_SPECS['P1'], "
            "hp = the governing vector with alpha_M replaced and rho_M, c_M, "
            "tau_d, r_max, d_min, max_mass verbatim)"
        ),
        "surfaces": list(SURFACES),
        "alphas": list(ALPHA_ALL),
        "role": (
            "the deployment-faithful question: at the governing vector with "
            "only alpha moved, is the held-out delta non-positive?"
        ),
    },
    "R": {
        "name": "selection_grid_reproduction",
        "law": (
            "rollout_session_f(cell=F01m/p1_online, row_spec=ROW_SPECS['P2'], "
            "hp = the stage-2 mass-grid binding: rho_M=0.5, c_M=None, the "
            "governing thresholds, alpha_M varied)"
        ),
        "surfaces": [SELECTION_SURFACE],
        "alphas": list(ALPHA_SCAN),
        "role": (
            "the reproduction check: per-session R2 must equal the recorded "
            "stage2_scored values at (rho_M=0.5, alpha_M) within the CPU "
            "tolerance anchor; the selection grid scanned the P2 objective "
            "cell with c_M=None, so only that binding reproduces the receipt"
        ),
    },
}

#: The selection receipts' own binding, disclosed because the reproduction
#: check must replay exactly what was scanned, not what was deployed: the
#: governing run's stage-2 grid scored the P2 measurement law with c_M=None
#: (the winner's c_M was calibrated only AFTER selection), and the deployed
#: F01m cells ran the P1 measurement law with the calibrated c_M.
SELECTION_GRID_BINDING_DISCLOSURE = {
    "objective_row_spec": "P2 (the sealed stage-P selection objective)",
    "deployed_row_spec": "P1 (the governing F01m cell law)",
    "stage2_c_M": (
        "None for every scanned vector (factor 3 passes unconditionally); the "
        "winner's c_M is the within-7 median of the pooled per-commit D2"
    ),
    "consequence": (
        "the R family replays (P2, c_M=None) to reproduce the receipt; the D "
        "family replays (P1, calibrated c_M) to answer the held-out question"
    ),
}

# ---------------------------------------------------------------------------
# Anchors (pre-registered, CPU tolerance law).
# ---------------------------------------------------------------------------

#: The CPU anchor law: GPU receipts cannot be reproduced bitwise on CPU
#: (float-reduction order), so every reproduction assert is a per-session
#: R2 tolerance at 1e-5 -- three orders of magnitude below the smallest
#: effect scale this diagnostic asks about (the within alpha steps are
#: ~0.002-0.008 R2, the external cell means live at ~0.22-0.23 R2).
ANCHOR_TOLERANCE_R2 = 1.0e-5

ANCHORS = {
    "d0_vs_sealed_f01m_rows": {
        "law": (
            "every D@alpha=0 row reproduces the sealed replay.json F01m row "
            "per session: window_count and query_starts_sha256 equal, and "
            "|R2_replay - R2_sealed| < 1e-5 (the CPU tolerance law; the "
            "governing GPU run additionally had bitwise prediction digests, "
            "which CPU arithmetic cannot promise)"
        ),
        "also": (
            "D@alpha=0 == D@alpha=0 implies predictions equal the sealed F00m "
            "row within the same tolerance (alpha_M = 0 is the no-op law)"
        ),
    },
    "d0_noop_within_run": {
        "law": (
            "inside THIS run: every D@alpha=0 receipt has cb == ca on every "
            "trial (zero carrier movement), so the alpha=0 cells are exact "
            "no-ops by construction regardless of CPU arithmetic"
        ),
    },
    "r_vs_sealed_stage2_scores": {
        "law": (
            "every R@alpha>0 row's per-session R2 equals the sealed "
            "hyperparameter_selection[m{b}].stage2_scored vector "
            "(rho_M=0.5, alpha_M) per_session value within 1e-5"
        ),
    },
    "carrier_support_binding": {
        "law": (
            "the sealed same-query carrier law re-derived per surface/session/"
            "budget must still bind the sealed t4_ridge_static evidence "
            "(selected support, raw/normalized T4 digests, activity digest), "
            "exactly as the governing physical driver asserted"
        ),
    },
}

# ---------------------------------------------------------------------------
# The pre-registered verdict.
# ---------------------------------------------------------------------------

VERDICT_MATCHED = "WITHIN_SELECTION_MATCHED_HELDOUT"
VERDICT_MISMATCH = "SELECTION_SURFACE_MISMATCH"

VERDICT_LAW = {
    "rows_considered": (
        "the D family on the HELD-OUT surface external_official_query, both "
        "budgets m4/m10, both scan alphas {0.125, 0.5}"
    ),
    "statistic": (
        "delta(alpha, budget) = equal_session_mean(D@alpha) - "
        "equal_session_mean(D@alpha=0) over the 6 held-out sessions "
        "(the alpha=0 mean is the sealed F00m/F01m no-op mean)"
    ),
    VERDICT_MATCHED: "every delta(alpha, budget) <= 0 within the epsilon band",
    VERDICT_MISMATCH: "some delta(alpha, budget) > 0 beyond the epsilon band",
    "boundary_epsilon": 1.0e-12,
    "boundary_rule": (
        "a positive delta inside the 1e-12 band of zero is disclosed as "
        "within_epsilon_band_of_zero and does NOT flip the verdict (the "
        "Stage-O/P convention); alpha=0 deltas are exactly 0.0 by the no-op "
        "law and are never verdict-relevant"
    ),
    "never_governs": (
        "this verdict is a POST-HOC DIAGNOSTIC statement only: it can never "
        "select a hyperparameter, promote a cell, or reopen the sealed "
        "governing run's promotion gate"
    ),
}

MONOTONICITY_LAW = {
    "question": (
        "is the held-out alpha-scan also monotonically non-positive in alpha "
        "(like the within scan, whose means fell monotonically 0 -> 0.125 -> "
        "0.25 -> 0.5), or does some alpha > 0 help held-out?"
    ),
    "per_budget_flags": {
        "nonincreasing_in_alpha": (
            "mean(0) >= mean(0.125) >= mean(0.5) on the equal-session means"
        ),
        "all_alphas_nonpositive_vs_anchor": (
            "both scan deltas <= 0 within the epsilon band"
        ),
    },
}

# ---------------------------------------------------------------------------
# Leakage discipline (binding).
# ---------------------------------------------------------------------------

LEAKAGE_LAW = {
    "alpha_gt_zero_rows": {
        "target_label_leakage": True,
        "checkpoint_selection_eligible": False,
        "deployment_eligible": False,
        "post_hoc_diagnostic": True,
        "reason": (
            "the held-out scores are read AFTER the governing verdict was "
            "sealed and the alpha axis was chosen with knowledge of the "
            "within-fold selection outcome; these rows can never select or "
            "promote anything"
        ),
    },
    "alpha_zero_rows": {
        "target_label_leakage": False,
        "checkpoint_selection_eligible": False,
        "deployment_eligible": False,
        "post_hoc_diagnostic": True,
        "reason": (
            "the alpha=0 rows are reproduction anchors of already-sealed "
            "numbers; they carry no new selection signal and remain "
            "ineligible for anything"
        ),
    },
    "receipt_statement": (
        "this run is a post-hoc diagnostic: it can NEVER select a "
        "hyperparameter, promote a cell, or update any model or checkpoint"
    ),
}

# ---------------------------------------------------------------------------
# Process.
# ---------------------------------------------------------------------------

Torch_THREADS = 4
HARD_TIMEOUT_SECONDS = 14_400
BATCH_SIZE = 1024

ENVIRONMENT_LAW = {
    "device": "cpu",
    "torch_num_threads": Torch_THREADS,
    "omp_threads_env": "OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=4",
    "gpu_policy": (
        "no GPU is touched: GPU 0 and GPU 1 are owned by other live routes; "
        "this run is CPU-only by operator instruction 2026-09-01"
    ),
    "dataloader_workers_max": 2,
    "dataloader_workers_actual": 0,
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
        "whole grid and every anchor assert pass; any failure leaves only the "
        "attempt receipt (fail-closed, the governing run's own law)"
    ),
    "frozen_roots_never_modified": [
        "tfpd_exploration/results/cdm_p1_m2_local_v1",
        "tfpd_exploration/results/cdm_p1_m2_v1",
        "tfpd_exploration/src/cdm_p1_m2_local_v1",
    ],
}

DEVIATIONS = [
    {
        "id": "cpu_instead_of_gpu1",
        "disclosure": (
            "the operator moved this diagnostic to CPU because both GPUs are "
            "owned by other live routes; the governing run ran on GPU 1"
        ),
    },
    {
        "id": "tolerance_anchor_instead_of_bitwise",
        "disclosure": (
            "CPU float-reduction order cannot bitwise-match GPU receipts, so "
            "the pre-registered anchor is per-session |Delta R2| < 1e-5 "
            "against the sealed rows (bitwise was the original brief); "
            "window_count/query_starts_sha256 and the alpha=0 no-op law are "
            "still exact"
        ),
    },
    {
        "id": "selection_grid_binding",
        "disclosure": (
            "the within reproduction check replays the selection grid's own "
            "binding (P2 objective row, c_M=None) because that is what the "
            "receipt recorded; the deployed F01m law (P1 row, calibrated c_M) "
            "is the D family. Both are run and both are labelled."
        ),
    },
    {
        "id": "attempt1_digest_fn_abort",
        "disclosure": (
            "the first reserved attempt aborted fail-closed before any cell "
            "ran: the sealed-row binding used cdm_core.array_digest where the "
            "sealed rows use the comparator's array_sha256; the aborted "
            "attempt root was renamed to "
            "results/m2_carrier_alpha_diagnostic_v1_attempt1_digest_fn_abort "
            "and this fresh attempt was reserved with the corrected digest law"
        ),
    },
    {
        "id": "attempt2_sealed_shadowing_abort",
        "disclosure": (
            "the second reserved attempt aborted fail-closed AFTER the whole "
            "D family ran but BEFORE any receipt was published: a loop-local "
            "'sealed' rebinding shadowed the sealed governing payload and "
            "broke the R-family lookup; the aborted attempt root was renamed "
            "to results/m2_carrier_alpha_diagnostic_v1_attempt2_sealed_"
            "shadowing_abort and this fresh attempt was reserved with the "
            "shadowing removed (no cell result was written or read)"
        ),
    },
]


def result_root(repo_root: Path) -> Path:
    return Path(repo_root) / RESULT_ROOT_RELATIVE
