"""Frozen constants and laws for the M2 Stage-O oracle carrier headroom test.

Authority: operator 2026-09-01 -- ONE inference-only, zero-training, CPU-only
oracle-headroom diagnostic on top of the SEALED governing run
``results/cdm_p1_m2_local_v1`` (attempt/replay/terminal sealed) and its sealed
post-hoc alpha replay ``results/m2_carrier_alpha_diagnostic_v1``.  Design:
``docs/DESIGN_SUPPORT_ANCHORED_CAUSAL_T4_MEMORY_20260830.md`` section 3 (the
causal oracle carrier headroom test) transferred to the LOCAL-M2 surface with
the binding Stage-O operator amendment (always-commit + trust-region
projection only; ``rho_M = 1.0``; block = 1 trial).

Scientific question: with TRUE completed-trial directions (target-label
leakage, diagnostic only), does the support-anchored block-refit carrier
create headroom over the activity baseline on M2 -- i.e. what is the ceiling
ANY direction denoiser (learned or not) could chase?

This run can NEVER select a hyperparameter, promote a cell, or update any
model/checkpoint: every O2m row reads the true target direction of a
completed trial and is labelled as target-label leakage.
"""

from __future__ import annotations

from pathlib import Path

SCHEMA = "m2_carrier_oracle_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_carrier_oracle_v1"
AUTHORITY = (
    "operator 2026-09-01: one inference-only CPU-only M2 Stage-O oracle "
    "carrier headroom test on the sealed cdm_p1_m2_local_v1 governing run, "
    "never selecting"
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

#: The sealed M2 predecessor receipts: the governing run itself, the sealed
#: post-hoc alpha replay (the CPU tolerance-anchor precedent this run reuses),
#: and the sealed same-query score that binds the carrier law.
PREDECESSOR_RELATIVE = (
    "tfpd_exploration/results/cdm_p1_m2_local_v1/attempt.json",
    "tfpd_exploration/results/cdm_p1_m2_local_v1/replay.json",
    "tfpd_exploration/results/cdm_p1_m2_local_v1/terminal.json",
    "tfpd_exploration/results/cdm_p1_m2_v1/audit.json",
    "tfpd_exploration/results/m2_carrier_alpha_diagnostic_v1/attempt.json",
    "tfpd_exploration/results/m2_carrier_alpha_diagnostic_v1/replay.json",
    "tfpd_exploration/results/m2_carrier_alpha_diagnostic_v1/terminal.json",
    "tfpd_exploration/results/m2_same_query_comparator_v1/score.json",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/__init__.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/anchor.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/gates.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/plan.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/replay.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/physical.py",
    "tfpd_exploration/scripts/run_cdm_p1_m2_local_v1.py",
    "tfpd_exploration/src/m2_carrier_alpha_diagnostic_v1/__init__.py",
    "tfpd_exploration/src/m2_carrier_alpha_diagnostic_v1/plan.py",
    "tfpd_exploration/src/m2_carrier_alpha_diagnostic_v1/physical.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/__init__.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/anchor.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/block_refit.py",
    "tfpd_exploration/src/support_anchored_t4_stage_o_v1/plan.py",
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
    "tfpd_exploration/src/m2_carrier_oracle_v1/__init__.py",
    "tfpd_exploration/src/m2_carrier_oracle_v1/plan.py",
    "tfpd_exploration/src/m2_carrier_oracle_v1/physical.py",
    "tfpd_exploration/scripts/run_m2_carrier_oracle_v1.py",
    "tfpd_exploration/tests/test_m2_carrier_oracle_v1.py",
)

# ---------------------------------------------------------------------------
# Surfaces and budgets.
# ---------------------------------------------------------------------------

SURFACES = ("within_post30", "external_official_query")
SELECTION_SURFACE = "within_post30"
HELD_OUT_SURFACE = "external_official_query"
BUDGETS = (4, 10)
M30 = 30
ALL_BUDGETS = (4, 10, 30)

EXPECTED_WITHIN_SESSIONS = 7
EXPECTED_EXTERNAL_SESSIONS = 6
HELD_OUT_BREADTH_MIN = 4  # >= 4/6 positive held-out sessions

# ---------------------------------------------------------------------------
# The cells (pre-registered).
# ---------------------------------------------------------------------------

#: The Stage-O amendment hyperparameters (the binding operator amendment of
#: WORKORDER_SUPPORT_ANCHORED_T4_STAGE_O_V1, mirrored on M2 verbatim).
RHO_M = 1.0
BLOCK_SIZE_COMPLETED_TRIALS = 1
EVIDENCE_WEIGHT_W = 1.0
ALPHA_M_PRIMARY = 0.5
ALPHA_M_SENSITIVITY = (0.125, 0.25, 1.0)
ALPHA_M_SENSITIVITY_GOVERNS = False
ALPHA_M_ALL = (ALPHA_M_PRIMARY,) + ALPHA_M_SENSITIVITY
ALPHA_M_M30_NOOP = 0.0

CELLS = {
    "O0m": {
        "name": "activity_baseline_sealed_f00m_law",
        "law": (
            "rollout_session_f(cell=F00m/frozen_t4_per_trial) -- the sealed "
            "F00m law verbatim: frozen T4, frozen sealed static activity, "
            "per-trial decode of the metric windows"
        ),
        "surfaces": list(SURFACES),
        "budgets": list(ALL_BUDGETS),
        "target_label_leakage": False,
        "role": "the baseline any direction denoiser must beat",
    },
    "O2m": {
        "name": "oracle_stage_o_anchored_block_refit",
        "law": (
            "same activity state as O0m + the TRUE completed-trial direction "
            "under the frozen true-direction law, fed to the Stage-O "
            "support-anchored block refit (EvidenceRow/EvidenceBank, rho_M=1.0, "
            "w=1, block=1 trial) with ALWAYS-COMMIT + trust-region projection "
            "only; no three-factor rejection semantics"
        ),
        "surfaces": list(SURFACES),
        "budgets": list(BUDGETS),
        "alphas": list(ALPHA_M_ALL),
        "alpha_primary": ALPHA_M_PRIMARY,
        "target_label_leakage": True,
        "role": "the actual headroom test (the primary row is alpha_M=0.5)",
    },
    "O2m_m30_noop": {
        "name": "oracle_m30_noop_row",
        "law": (
            "the O2m law at M30 with alpha_M = 0 (the M30 no-op law: the "
            "deployable route keeps alpha_M = 0 at M30); commits are recorded "
            "but the carrier is byte-identical to the sealed support carrier"
        ),
        "surfaces": list(SURFACES),
        "budgets": [M30],
        "alphas": [ALPHA_M_M30_NOOP],
        "target_label_leakage": True,
        "role": "the mandatory M30 diagnostic/safety column, run as a no-op",
    },
}

# ---------------------------------------------------------------------------
# The frozen true-direction law (target-label leakage, diagnostic only).
# ---------------------------------------------------------------------------

TRUE_DIRECTION_LAW = {
    "source": (
        "the query stream's own covariate rows (dataset.covariate_data[session] "
        "= finger_vel in SI m/s) at the endpoints of the trial's complete "
        "causal windows (causal_starts + WINDOW_SIZE - 1) -- the exact "
        "integration domain of the decoded measurement"
    ),
    "unit_law": (
        "the governing VELOCITY_UNIT_LAW view_scale m/s -> cm/s (x100) applied "
        "before the frozen direction gates; no gate threshold is rescaled"
    ),
    "measurement": (
        "cdm_core.pseudo_direction_from_velocity(restored_true_rows, "
        "all-ones movement mask over the trial's causal windows, "
        "config=CDMDConfig(support_budget_m=budget)) -- continuous angle -> "
        "8 canonical snap (max pi/8), the cdm_p1_m2 audit's frozen direction "
        "law, the same law the decoded measurement uses"
    ),
    "group_binding": (
        "the TRUE direction is group-independent: every complementary group "
        "receives the same snapped direction index (direction_indices = "
        "(k, k, k, k)); no group forward is run for the measurement"
    ),
    "causal_discipline": (
        "the direction is revealed only after the trial's metric windows are "
        "scored with the pre-trial state; the committed evidence can affect "
        "only later trials (the receipts bind cb -> ca per trial)"
    ),
    "empirical_check_disclosed_by_the_audit": (
        "true query movement directions snap within pi/8 on 100% of labeled "
        "query trials on both rosters (median ~4-5 deg) under this law"
    ),
    "leakage": (
        "reading the query stream's covariates is TARGET-LABEL LEAKAGE: it is "
        "never a deployable input; every O2m row is labelled as such"
    ),
}

O2M_COMMIT_LAW = {
    "commit_rule": "always-commit",
    "movement_rule": "trust-region projection only",
    "rejection_semantics": (
        "NONE in Stage O; the three-factor gate belongs to Stage P and is NOT "
        "imported here"
    ),
    "measurement_validity_still_required": (
        "a trial whose TRUE direction is not measurable under the frozen "
        "pseudo_direction_from_velocity law (displacement/speed/canonical snap "
        "gates) or that has no complete causal window contributes no evidence "
        "row and moves nothing; the typed rejection reason is recorded"
    ),
    "rationale": (
        "a gated O2 could turn gate miscalibration into a false STOP; the "
        "trust region alone bounds movement"
    ),
}

# ---------------------------------------------------------------------------
# The trust-region calibration law (the Stage-O law mirrored on within-7).
# ---------------------------------------------------------------------------

C_M_CALIBRATION = {
    "rule": (
        "per budget, the MEDIAN of the pooled per-commit unprojected aggregate "
        "D2 over ALL committed evidence rows of the seven within_post30 "
        "sessions, collected in a dedicated alpha_M=0/c_M=None calibration "
        "pass (the Stage-O law; external never selects)"
    ),
    "invariance": (
        "the evidence bank, the refit and D2 never depend on the active "
        "carrier, so the calibration pass yields the exact per-commit D2 "
        "sequence every governing pass re-derives; the calibration rollouts "
        "double as the O2m alpha=0 no-op anchor against O0m"
    ),
    "statistic": "median",
    "coverage_claim_forbidden": (
        "c_M is a source-calibrated geometric trust threshold; it must not be "
        "described as a nominal 95% credible region"
    ),
    "frozen_before_target_scoring": True,
    "m30": "M30 runs alpha_M = 0 (the no-op row); no M30 c_M is calibrated",
}

# ---------------------------------------------------------------------------
# Anchors (pre-registered, CPU tolerance law).
# ---------------------------------------------------------------------------

ANCHOR_TOLERANCE_R2 = 1.0e-5

ANCHORS = {
    "o0m_vs_sealed_f00m_rows": {
        "law": (
            "every O0m row reproduces the sealed replay.json F00m row per "
            "session/budget/surface: window_count, query_starts_sha256 and "
            "target_sha256 equal, and |R2_oracle - R2_sealed| < 1e-5 (the CPU "
            "tolerance-anchor convention of the sealed alpha diagnostic)"
        ),
    },
    "o2m_alpha0_noop": {
        "law": (
            "inside THIS run: the calibration pass (alpha_M=0, c_M=None) and "
            "the M30 no-op row have cb == ca on every trial and their "
            "prediction digests equal the O0m prediction digests bitwise (the "
            "alpha_M = 0 identity holds regardless of CPU arithmetic)"
        ),
    },
    "carrier_support_binding": {
        "law": (
            "the sealed carrier law re-derived per surface/session/budget must "
            "still bind the sealed t4_ridge_static evidence (selected support, "
            "raw/normalized T4 digests, activity digest) and the M2 anchor's "
            "zero-evidence parity must be bitwise, exactly as the governing "
            "physical driver asserted"
        ),
    },
    "empty_bank_parity": {
        "law": (
            "per surface/session/budget, the Stage-O refit with an EMPTY "
            "evidence bank reproduces the anchor's own solve(A0, b0) support "
            "coefficients within the float32 quantization floor of the sealed "
            "carrier (max abs difference <= 1e-6 in float64 coefficient space; "
            "the bitwise law is the anchor's own rebuilt_rows_bitwise_equal "
            "parity, asserted separately)"
        ),
    },
}

# ---------------------------------------------------------------------------
# The pre-registered verdict.
# ---------------------------------------------------------------------------

VERDICT_GO = "GO_M2_DENOISER_TARGET_EXISTS"
VERDICT_NULL = "ORACLE_NULL__M2_CARRIER_CEILING_IS_ZERO"

VERDICT_LAW = {
    "rows_considered": (
        "the PRIMARY O2m row (alpha_M = 0.5) on the HELD-OUT surface "
        "external_official_query at both budgets m4/m10, versus O0m"
    ),
    "statistic": (
        "delta(budget) = equal_session_mean(O2m@0.5) - equal_session_mean(O0m) "
        "over the 6 held-out sessions, paired per session"
    ),
    VERDICT_GO: (
        "some budget has delta >= +0.01 AND at least 4/6 held-out sessions "
        "with a strictly positive per-session delta at that budget"
    ),
    VERDICT_NULL: "otherwise",
    "boundary_rule": (
        "exact >= on the float64 equal-session means; a miss inside the "
        "1e-12 program epsilon band is disclosed and never flips the verdict"
    ),
    "boundary_epsilon": 1.0e-12,
    "sensitivity_rows": (
        "alpha_M in {0.125, 0.25, 1.0} and the within surface are recorded as "
        "NON-GOVERNING sensitivity/context rows; they can never flip the "
        "verdict"
    ),
    "never_governs": (
        "this verdict is a diagnostic ceiling statement only: it can never "
        "select a hyperparameter, promote a cell, or reopen the sealed "
        "governing run's promotion gate"
    ),
}

#: The pre-registered wall classifier for the NULL branch: which wall explains
#: a zero oracle ceiling -- too few usable commit points, or true directions
#: that carry no carrier information.
WALL_CLASSIFIER = {
    "commit_point_scarcity_wall": (
        "at BOTH budgets the pooled held-out accepted oracle commits "
        "(measurement-eligible completed trials whose TRUE direction passed "
        "the frozen gates and therefore committed) number fewer than 24 = "
        "4/session -- the evidence stream itself is too sparse for any "
        "estimator to express a per-session effect"
    ),
    "no_information_in_true_directions": (
        "otherwise: commit points exist but even TRUE directions do not move "
        "the carrier to a better decode -- the ceiling is informational, not "
        "a commit-point artifact"
    ),
    "always_reported_alongside": (
        "the eligible pool (evidence-eligible trials and those with at least "
        "one complete causal window), the per-session commit counts, the "
        "oracle direction-acceptance rate, and the sealed decoded-direction "
        "rejection census are reported on every row regardless of the verdict"
    ),
}

# ---------------------------------------------------------------------------
# The evidence-stream census.
# ---------------------------------------------------------------------------

CENSUS_LAW = {
    "question": (
        "does even the ORACLE hit the commit-point scarcity wall (the sealed "
        "decoded-direction stream rejected ~72% of measurement attempts as "
        "low_displacement and only 48 external trials per budget even reach "
        "the measurement), or are commit points plentiful under TRUE "
        "directions?"
    ),
    "oracle_side": (
        "per held-out session and budget from the PRIMARY O2m receipts: "
        "evidence-eligible trials, trials with >= 1 complete causal window, "
        "accepted (committed) true directions, and the typed rejection "
        "reasons of the frozen direction gates"
    ),
    "decoded_side": (
        "the sealed governing F01m receipts' measurement_reasons and "
        "committed_rows per session/budget, read from the sealed replay "
        "(alpha_M = 0 no-op rows whose measurement machinery ran and was "
        "recorded); never recomputed, never re-selected"
    ),
    "rates": (
        "oracle acceptance = accepted / with_complete_window; decoded "
        "acceptance = committed / with_complete_window; the decoded "
        "low_displacement share = low_displacement / with_complete_window"
    ),
}

# ---------------------------------------------------------------------------
# Leakage discipline (binding).
# ---------------------------------------------------------------------------

LEAKAGE_LAW = {
    "o2m_rows": {
        "target_label_leakage": True,
        "checkpoint_selection_eligible": False,
        "deployment_eligible": False,
        "post_hoc_diagnostic": True,
        "reason": (
            "the O2m measurement reads the query stream's true covariates "
            "(the target labels) after each trial is finalized; this is "
            "diagnostic leakage that no deployable system can have"
        ),
    },
    "o0m_rows": {
        "target_label_leakage": False,
        "checkpoint_selection_eligible": False,
        "deployment_eligible": False,
        "post_hoc_diagnostic": True,
        "reason": (
            "the O0m rows reproduce the already-sealed F00m law and numbers "
            "within the CPU tolerance anchor; they carry no new selection "
            "signal and remain ineligible for anything"
        ),
    },
    "receipt_statement": (
        "this run is a target-label-leaking diagnostic: it can NEVER select a "
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
        "no GPU is touched: GPU 0 and GPU 1 are owned by other live routes "
        "(a 50-epoch M1 training pair occupies GPU 1); this run is CPU-only "
        "by operator instruction 2026-09-01"
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
        "tfpd_exploration/results/m2_carrier_alpha_diagnostic_v1",
        "tfpd_exploration/src/cdm_p1_m2_local_v1",
        "tfpd_exploration/src/support_anchored_t4_stage_o_v1",
        "tfpd_exploration/src/support_anchored_t4_stage_p_v1",
    ],
}

DEVIATIONS = [
    {
        "id": "cpu_instead_of_any_gpu",
        "disclosure": (
            "the operator moved this run to CPU because GPU 0 and GPU 1 are "
            "owned by other live routes (a 50-epoch M1 training pair occupies "
            "GPU 1); the governing run ran on GPU 1. CPU-only with "
            "torch threads = 4 and 0 dataloader workers, exactly like the "
            "sealed m2_carrier_alpha_diagnostic_v1 CPU run"
        ),
    },
    {
        "id": "tolerance_anchor_instead_of_bitwise",
        "disclosure": (
            "CPU float-reduction order cannot bitwise-match GPU receipts, so "
            "the pre-registered anchor is per-session |Delta R2| < 1e-5 "
            "against the sealed rows (the sealed alpha diagnostic' "
            "convention); window_count/query_starts_sha256/target_sha256 and "
            "the alpha = 0 no-op law are still exact"
        ),
    },
    {
        "id": "anchor_adapter_for_the_stage_o_laws",
        "disclosure": (
            "the frozen Stage-O block_refit.refit_from_anchor reads "
            "anchor.statistics.counts, which the M2SupportAnchor exposes as "
            "per_group[g].counts; this package wraps the M2 anchor in a "
            "read-only duck-typed adapter that supplies exactly that view. "
            "No frozen package is edited; the adapter adds no state"
        ),
    },
    {
        "id": "o1m_not_run",
        "disclosure": (
            "the design's O1 diagnostic (the existing recursive incremental "
            "estimator fed the true direction) has no M2 F-family "
            "counterpart: the sealed M2 governing selection executed F01m as "
            "a bitwise F00m no-op (alpha_M = 0), so there is no live recursive "
            "M2 estimator to feed; the brief's O2m-vs-O0m contrast is the "
            "headroom question"
        ),
    },
    {
        "id": "c_m_recalibrated_under_the_oracle_law",
        "disclosure": (
            "c_M is re-calibrated per budget as the within-7 median of the "
            "pooled per-commit D2 UNDER THE ORACLE law (the Stage-O "
            "calibration law mirrored on the M2 within roster), not reused "
            "from the governing decoded-direction calibration: the governing "
            "c_M was calibrated on a D2 distribution produced by 7-commit "
            "decoded banks, which would mis-scale the oracle movement. The "
            "calibration surface is within_post30 ONLY; external never "
            "selects"
        ),
    },
]


def result_root(repo_root: Path) -> Path:
    return Path(repo_root) / RESULT_ROOT_RELATIVE
