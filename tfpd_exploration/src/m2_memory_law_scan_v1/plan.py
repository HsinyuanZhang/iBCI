"""Frozen constants and laws for the M2 activity-memory-law scan (V1).

Authority: user work order 2026-09-01 (inference-only, label-free memory-law
scan on M2, CPU-ONLY because both GPUs belong to other agents).  The
foundation is the SEALED G-family machinery of ``src/cdm_p1_m2_local_v1``
(G00m = the sealed ``m2_precision_cdm_v2_screen_v1`` ``m4_activity_only``
law: uniform mean over the growing completed-trial FIFO capped at the B3S
30-trial stack) plus that screen's frozen executors, reused by import only.

The scan varies exactly one thing -- the recency policy of the completed-trial
activity pool -- with the decoder, the T4 carrier, the surfaces, the rosters,
the supports and the scoring law all frozen:

* UNIFORM_CAP30   the sealed law verbatim (anchor cell);
* UNIFORM_UNCAPPED same uniform mean, no eviction ever;
* EMA_A090 / EMA_A095 / EMA_A080  exponential recency on the pool mean.

Every number is LOCAL-protocol evidence; no official-contract claim is made.
"""

from __future__ import annotations

from pathlib import Path

SCHEMA = "m2_memory_law_scan_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_memory_law_scan_v1"

# ---------------------------------------------------------------------------
# The sealed foundations this scan builds on (verified by SHA-256 at launch).
# ---------------------------------------------------------------------------

#: The sealed G-family route whose replay helpers orchestrate the frozen M2
#: runtime (``_g_session_views`` / ``g_support_material`` / ``g_query_rows`` /
#: ``rollout_g00m``) and whose constants bind the surfaces and budgets.
SEALED_G_PACKAGE = "tfpd_exploration/src/cdm_p1_m2_local_v1"
#: The sealed Native-M2 CDM screen: its ``score.json`` holds the
#: ``m4_activity_only`` / ``m10_activity_only`` anchor rows, and its
#: ``physical`` executors (``_build_memory`` / ``_score_system`` / surface
#: laws) are the sealed activity-memory law itself.
CDM_SCREEN_SCORE_RELATIVE = "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1/score.json"
CDM_SCREEN_SCORE_SHA256 = (
    "455485bd854a36392f17ec5ed029b1a4044a01a8dd7dfeae95c7763b8327f5f6"
)

SURFACES = ("within_post30", "external_post30_local")
SURFACE_LAW = {
    "within_post30": (
        "the sealed CDM screen's within surface: train-dataset common post-30 "
        "windows, 7 M2 held-in sessions (the ONLY selection surface)"
    ),
    "external_post30_local": (
        "the sealed CDM screen's external surface: held-out-dataset common "
        "post-30 windows, 6 M2 held-out sessions (reported once, never selected)"
    ),
}
EXPECTED_SESSIONS_BY_SURFACE = {"within_post30": 7, "external_post30_local": 6}

BUDGETS = (4, 10)
M4 = 4
M10 = 10

#: The B3S activity stack limit (the sealed ``ACTIVITY_STACK_LIMIT``): support
#: M plus a completed-trial FIFO of capacity 30 - M.  "Capped 30" therefore
#: means the total pool never exceeds 30 trials (26 FIFO rows at M4, 20 at
#: M10).
ACTIVITY_STACK_LIMIT = 30

#: The first ``ACTIVITY_STACK_LIMIT`` trials are the calibration pool; the
#: completed-trial evidence stream starts at position 30 and every row of the
#: query partition advances the activity memory (the sealed V3 law, including
#: short trials without a complete 50-bin window).
EVIDENCE_START_POSITION = 30

WINDOW_BINS = 50
CHANNELS = 96
BEHAVIOR_SCALE = 5.0
MODEL_BIN_SECONDS = 0.020

# ---------------------------------------------------------------------------
# The cells (the memory laws under comparison).
# ---------------------------------------------------------------------------

BASELINE_POLICY = "UNIFORM_CAP30"

EMA_ALPHAS = {"EMA_A090": 0.90, "EMA_A095": 0.95, "EMA_A080": 0.80}

POLICY_ORDER = (
    "UNIFORM_CAP30",
    "UNIFORM_UNCAPPED",
    "EMA_A090",
    "EMA_A095",
    "EMA_A080",
)

POLICIES = {
    "UNIFORM_CAP30": {
        "family": "uniform",
        "law": (
            "the sealed m4/m10 activity_only law verbatim: linear-B3S support "
            "trials (D-opt four at M4, first-ten-finite-direction at M10) plus "
            "a completed-trial FIFO of capacity 30 - M, uniform mean inside "
            "the frozen B3S pooling"
        ),
        "eviction": "oldest completed trial evicted once the FIFO is full",
    },
    "UNIFORM_UNCAPPED": {
        "family": "uniform",
        "law": (
            "the same uniform mean over the same support, but completed "
            "trials are never evicted: the pool grows to M + (all completed "
            "trials)"
        ),
        "eviction": "none, ever",
    },
    "EMA_A090": {
        "family": "ema", "alpha": 0.90, "effective_pool": 10,
        "law": "E <- 0.90*E + 0.10*feat per completed trial",
    },
    "EMA_A095": {
        "family": "ema", "alpha": 0.95, "effective_pool": 20,
        "law": "E <- 0.95*E + 0.05*feat per completed trial",
    },
    "EMA_A080": {
        "family": "ema", "alpha": 0.80, "effective_pool": 5,
        "law": "E <- 0.80*E + 0.20*feat per completed trial",
    },
}

#: Where the memory law lives inside the frozen model.  The B3S identity is
#: ``post_pool(concat(mean_over_pool(pre_pool(trial)), side))``; the completed-
#: trial pool reaches the decoder ONLY through that mean, so every policy in
#: this scan is a reweighting of one and the same per-trial feature stream.
MEMORY_LAW = {
    "identity_domain": (
        "the frozen student's id_encoder pre-pool feature space phi = "
        "pre_pool(trial) of shape [N, hidden]; the pool enters the decoder "
        "only through mean_over_pool(phi), then post_pool with the frozen "
        "static T4 side features"
    ),
    "bypass_law": (
        "UNCAPPED and EMA identities are produced by the frozen encoder's own "
        "``finalize_identity`` arithmetic (sum/count -> concat side -> "
        "post_pool) fed with the policy's pooled feature state; UNIFORM_CAP30 "
        "is the sealed path itself (full stack -> ``compute_identity``)"
    ),
    "arithmetic": (
        "float32 on CPU, sequential accumulation in pool order (support "
        "first, then completed trials in arrival order); UNCAPPED's running "
        "sum reproduces the frozen stack-mean accumulation order exactly"
    ),
    "ema_initialization": (
        "E0 = the uniform mean of the M-budget support phi features (the "
        "M-budget support mean); the completed-trial count is reported "
        "separately from the mean law"
    ),
    "ema_closed_form": (
        "E_k = alpha^k * E0 + (1 - alpha) * sum_{i=1..k} alpha^(k-i) * phi_i; "
        "the initial support mean carries weight alpha^k after k completed "
        "trials and the effective pool is 1/(1 - alpha)"
    ),
    "distribution_shift_disclosure": (
        "the B3S encoder was trained on uniform means of phi; UNIFORM_UNCAPPED "
        "stays inside that family (uniform weights, larger pool), while the "
        "EMA cells are an inference-side soft reweighting of the same pool -- "
        "a mild distribution shift on the pooling input, disclosed, never "
        "target-selected"
    ),
    "short_trial_policy": (
        "inherited from the sealed V3 law: every completed query trial "
        "advances the activity memory, including trials with no complete "
        "50-bin window; the carrier never moves (activity_only system, zero "
        "carrier updates in every cell)"
    ),
}

FROZEN_EVERYTHING_ELSE = {
    "decoder": "the frozen T4 student checkpoint (sha256 pinned by the sealed exporter)",
    "carrier": (
        "the sealed static fixed-ridge T4 carrier over the selected support "
        "(Hz domain, scaled to counts-per-bin by MODEL_BIN_SECONDS); no "
        "carrier adaptation in any cell"
    ),
    "supports": (
        "M4: greedy forward D-opt four from the first-30 angles; M10: the "
        "first ten finite-direction trials of the first 30 (the sealed laws)"
    ),
    "scoring": (
        "variance-weighted R2 over the sealed CDM screen's own surfaces; "
        "equal-session means across the frozen rosters"
    ),
}

# ---------------------------------------------------------------------------
# Gates (pre-registered before any data run).
# ---------------------------------------------------------------------------

GATES = {
    "primary": {
        "expression": (
            "policy - UNIFORM_CAP30 >= +0.01 equal-session within_post30 mean "
            "R2 at M4 AND positive within sessions >= 5/7"
        ),
        "delta_floor": 0.01,
        "breadth_min": 5,
        "breadth_denominator": 7,
        "surface": "within_post30",
        "budget": M4,
        "selection_disclosure": (
            "alpha variants and the uncapped row are SELECTED ONLY on the "
            "within surface; the external surface is reported once and never "
            "selects"
        ),
    },
    "safety": {
        "expression": (
            "policy - UNIFORM_CAP30 >= -0.02 equal-session "
            "external_post30_local mean R2 at EVERY budget (M4 and M10)"
        ),
        "floor": -0.02,
        "budgets": [M4, M10],
    },
    "boundary_epsilon": 1.0e-12,
    "boundary_rule": (
        "exact >= on float64 equal-session means; a miss inside the 1e-12 "
        "program epsilon is disclosed as within_epsilon_band_of_boundary and "
        "never flips the verdict (the Stage-O/P convention)"
    ),
    "verdict_law": {
        "MEMORY_LAW_UNIFORM_RETAINED": (
            "no challenger passes primary (with safety); the sealed "
            "UNIFORM_CAP30 law is retained"
        ),
        "EMA_RECENCY_WINS(name alpha)": (
            "an EMA policy passes primary and safety and has the largest "
            "within-M4 delta among passers (tie broken by POLICY_ORDER)"
        ),
        "UNCAPPED_WINS": (
            "UNIFORM_UNCAPPED passes primary and safety and has the largest "
            "within-M4 delta among passers (tie broken by POLICY_ORDER)"
        ),
        "primary_pass_safety_fail": (
            "a policy that passes primary but fails safety is disclosed and "
            "excluded from selection; selection then continues among the "
            "remaining passers, and falls back to "
            "MEMORY_LAW_UNIFORM_RETAINED if none remain"
        ),
    },
    "nulls_reported_as_nulls": True,
    "never_average_budgets": (
        "M4 and M10 are reported separately; no budget averaging rescues or "
        "rejects a policy"
    ),
    "official_contract_disclaimer": (
        "every number here is LOCAL-protocol evidence on the sealed CDM "
        "screen's own surfaces; the official M2 evaluation contract hides "
        "completed-trial boundaries and no official-contract claim is made"
    ),
}

# ---------------------------------------------------------------------------
# Anchors (the scan's binding to the sealed rows).
# ---------------------------------------------------------------------------

ANCHORS = {
    "uniform_cap30_vs_sealed_activity_only_rows": {
        "path": CDM_SCREEN_SCORE_RELATIVE,
        "sha256_at_design": CDM_SCREEN_SCORE_SHA256,
        "cells": {M4: "m4_activity_only", M10: "m10_activity_only"},
        "exact_fields": (
            "query_starts_sha256, target_sha256, window_count (pure-data "
            "fields, device independent)"
        ),
        "r2_tolerance": 1.0e-3,
        "r2_tolerance_rationale": (
            "the sealed rows were produced on GPU; this scan is CPU-only, so "
            "float32 decode paths differ in the last bits.  Any law error "
            "(wrong support, wrong surface, wrong memory) moves R2 by orders "
            "of magnitude more than 1e-3, while CPU/GPU float drift sits far "
            "below it; the realized max |delta R2| is disclosed either way"
        ),
        "prediction_digest_disclosure": (
            "prediction_sha256 equality across devices is NOT expected and "
            "is reported informationally"
        ),
    },
    "m4_executor_equivalence": {
        "law": (
            "on the equivalence roster (all 6 external sessions and the first "
            "2 within sessions) the generalized UNIFORM_CAP30 executor "
            "(cdm_physical._build_memory + _score_system at any budget) must "
            "reproduce the sealed cdm_p1_m2_local_v1.replay.rollout_g00m "
            "output bit-exactly (prediction/target/starts digests, r2, "
            "window_count) at M4; the M4 cell itself is rollout_g00m verbatim"
        ),
    },
    "first_scored_identity_agreement": {
        "law": (
            "UNIFORM_CAP30 and UNIFORM_UNCAPPED share the identical first "
            "scored-query identity digest on every session/budget (no "
            "eviction can have occurred yet, so the two uniform policies must "
            "agree bitwise)"
        ),
    },
}

# ---------------------------------------------------------------------------
# The drift reading (pre-registered descriptive statistic, never a gate).
# ---------------------------------------------------------------------------

DRIFT_READING = {
    "question": "does recency help where sessions are long?",
    "law": (
        "per policy: Spearman rank correlation between the per-session "
        "within_post30 M4 delta (policy - UNIFORM_CAP30) and the session's "
        "completed-query-trial count; plus a median split of the within "
        "sessions at the roster's median completed-trial count (long half vs "
        "short half mean delta).  The external roster (few completed trials "
        "per session) is reported as the short-memory regime"
    ),
    "tie_handling": "average ranks",
    "role": "descriptive only; never selects, never gates",
}

# ---------------------------------------------------------------------------
# Environment and process.
# ---------------------------------------------------------------------------

ENVIRONMENT_LAW = {
    "cuda_visible_devices": "",
    "device": "cpu",
    "cpu_only_rationale": "both GPUs belong to other agents; this scan never touches CUDA",
    "torch_num_threads": 4,
    "torch_num_interop_threads": 1,
    "deterministic_algorithms": True,
    "python": "/home/xinyuan/miniconda3/envs/spint/bin/python",
    "no_user_site": True,
}

BATCH_SIZE = 1024
HARD_TIMEOUT_SECONDS = 21_600

OWNED_PATHS = (
    "tfpd_exploration/src/m2_memory_law_scan_v1/__init__.py",
    "tfpd_exploration/src/m2_memory_law_scan_v1/plan.py",
    "tfpd_exploration/src/m2_memory_law_scan_v1/gates.py",
    "tfpd_exploration/src/m2_memory_law_scan_v1/memory.py",
    "tfpd_exploration/src/m2_memory_law_scan_v1/physical.py",
    "tfpd_exploration/scripts/run_m2_memory_law_scan_v1.py",
    "tfpd_exploration/tests/test_m2_memory_law_scan_v1.py",
)

PREDECESSOR_RELATIVE = (
    "tfpd_exploration/results/cdm_p1_m2_local_v1/attempt.json",
    "tfpd_exploration/results/cdm_p1_m2_local_v1/replay.json",
    "tfpd_exploration/results/cdm_p1_m2_local_v1/terminal.json",
    "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1/score.json",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/__init__.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/anchor.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/gates.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/plan.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/physical.py",
    "tfpd_exploration/src/cdm_p1_m2_local_v1/replay.py",
    "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/physical.py",
    "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/plan.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/core.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/physical.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/plan.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py",
    "tfpd_exploration/src/calibration_budget_comparators_v1.py",
)


def result_root(repo_root: Path) -> Path:
    return Path(repo_root) / RESULT_ROOT_RELATIVE
