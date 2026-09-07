"""Frozen constants and pre-registered laws for the M2 post-fusion identity
probe (V1).

Authority: operator work order 2026-09-02 (verbatim task): run the
post-fusion identity-memory probe on M2, CPU-ONLY (``CUDA_VISIBLE_DEVICES=''``,
torch threads 4, 0 workers), inference-only on frozen weights, asking whether
averaging identities AFTER the fusion MLP (per-trial readout then average)
beats the current pooled readout (average then MLP) -- i.e. the Jensen gap of
the identity path and whether output-side identity memory has any value on
frozen weights.

Foundation (import-only, never edited): the sealed G-family machinery of
``src/cdm_p1_m2_local_v1`` (``_g_session_views`` / ``g_support_material`` /
``g_query_rows`` / ``rollout_g00m`` = the sealed ``m4_activity_only`` law),
that screen's frozen executors, and the precedent laws/anchors of
``results/m2_kcurve_v1`` (the sealed G00m/k4 champion values) and
``results/m2_memory_law_scan_v1`` (the uncapped-accumulator and EMA pooling
precedents).  This probe varies exactly one thing -- WHERE the uniform mean of
the completed-trial pool is taken relative to the frozen B3S post-pool MLP --
with the decoder, the T4 carrier, the surfaces, the rosters, the D-opt-4
support and the scoring law all frozen.

Cells (all on external_post30_local + within_post30, the k=4 D-opt support
law = the sealed champion shape):

* POOLED (anchor)  the sealed law verbatim: identity from the mean activity
  pool (mean of pre-pool features, then the frozen fusion MLP);
* POSTFUSION_MEAN  the user's proposed readout: per completed trial i,
  identity_i = MLP(concat(phi(trial_i), T4)) with the SAME frozen weights;
  the deployed identity is the running mean of identity_i over the pool
  members of the sealed champion shape (the D-opt-4 support identities plus
  the identities of the last 30 - 4 completed query trials, the CAP30 FIFO
  membership mirrored in identity space).  This isolates the placement
  question with pool membership held fixed against POOLED.
* POSTFUSION_ACCUM_UNCAPPED  the uncapped accumulator law of
  ``m2_memory_law_scan_v1`` translated to identity space: init at the
  support-trial identity mean, then accumulate every query-trial identity,
  no eviction ever (disclosed challenger, never the gated head-to-head).
* POSTFUSION_EMA_A090  a small EMA row (alpha 0.9) in identity space as the
  recency sensitivity -- report only.

Key disclosures: the frozen MLP was trained on pooled means, so per-trial
inputs are out-of-distribution; this probe measures the NET effect (Jensen
gap + OOD penalty) on frozen weights.  A positive result makes a training-side
cell (PIT-style per-trial exposure) worth considering; a null/harmful result
closes the placement question on frozen weights.

Every number is LOCAL-protocol evidence; no official-contract claim is made.
"""

from __future__ import annotations

from pathlib import Path

SCHEMA = "m2_postfusion_probe_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_postfusion_probe_v1"

# ---------------------------------------------------------------------------
# The sealed foundations this probe builds on (verified by SHA-256 at launch).
# ---------------------------------------------------------------------------

#: The sealed G-family route whose replay helpers orchestrate the frozen M2
#: runtime and whose ``rollout_g00m`` IS the sealed ``m4_activity_only`` law
#: (the POOLED anchor cell, run verbatim).
SEALED_G_PACKAGE = "tfpd_exploration/src/cdm_p1_m2_local_v1"
#: The sealed Native-M2 CDM screen: its ``score.json`` holds the
#: ``m4_activity_only`` anchor rows and its ``physical`` executors are the
#: sealed activity-memory law itself.
CDM_SCREEN_SCORE_RELATIVE = "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1/score.json"
CDM_SCREEN_SCORE_SHA256 = (
    "455485bd854a36392f17ec5ed029b1a4044a01a8dd7dfeae95c7763b8327f5f6"
)
#: The sealed k-curve receipts: their ``anchor_receipt_values.g00m_means``
#: are the sealed G00m/k4 equal-session means this probe's POOLED anchor must
#: reproduce within 1e-5 (read-only context, pinned by digest).
KCURVE_TERMINAL_RELATIVE = "tfpd_exploration/results/m2_kcurve_v1/terminal.json"
KCURVE_TERMINAL_SHA256 = (
    "77c1bef7df115aac25e38327803e9c613fb5520ff943a2997a5df8365de1214c"
)
#: The sealed memory-law scan receipts: the uncapped-accumulator and EMA
#: pooling precedents translated into identity space by this probe.
MEMORY_LAW_SCAN_TERMINAL_RELATIVE = "tfpd_exploration/results/m2_memory_law_scan_v1/terminal.json"
MEMORY_LAW_SCAN_TERMINAL_SHA256 = (
    "42462644b27eea8f9186b1990393b3472cca99779c4da64c1e28bc8c749e974b"
)

#: The sealed G00m/k4 equal-session means (provenance: the sealed
#: ``m4_activity_only`` rows of the CDM screen, cross-issued verbatim as the
#: k-curve terminal's ``anchor_receipt_values.g00m_means``).
G00M_ANCHOR_MEANS = {
    "external_post30_local": 0.2990573453320357,
    "within_post30": 0.6540862067123892,
}

T4_CHECKPOINT_SHA256 = (
    "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
)
SPINT_CHECKPOINT_SHA256 = (
    "fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec"
)
NORMALIZATION_SHA256 = (
    "d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e"
)

# ---------------------------------------------------------------------------
# Surfaces, roster, budget (the sealed champion shape).
# ---------------------------------------------------------------------------

SURFACES = ("external_post30_local", "within_post30")
SURFACE_LAW = {
    "external_post30_local": (
        "the sealed CDM screen's external surface: held-out-dataset common "
        "post-30 windows, 6 M2 held-out sessions (THE VERDICT SURFACE, "
        "pre-registered)"
    ),
    "within_post30": (
        "the sealed CDM screen's within surface: train-dataset common post-30 "
        "windows, 7 M2 held-in sessions (disclosed paired delta, never the "
        "verdict surface)"
    ),
}
EXPECTED_SESSIONS_BY_SURFACE = {"external_post30_local": 6, "within_post30": 7}

#: The single support law of this probe: the sealed M4 greedy forward D-opt
#: four over the first-30 finite-angle candidates (the sealed champion shape).
M4 = 4
BUDGET = M4
BUDGETS = (M4,)

#: The B3S activity stack limit (the sealed law): support 4 plus a
#: completed-trial FIFO of capacity 30 - 4 = 26; the total pool never exceeds
#: 30 trials.
ACTIVITY_STACK_LIMIT = 30
FIFO_CAPACITY = ACTIVITY_STACK_LIMIT - M4

#: Every completed post-30 query trial advances the pool (the sealed V3 law,
#: including short trials without a complete 50-bin window).
EVIDENCE_START_POSITION = 30

CHANNELS = 96
WINDOW_BINS = 50
TRIAL_LENGTH = 100
#: The frozen B3S identity path: pre_pool per-trial Linear(trial_length=100 ->
#: hidden=64) + ReLU, mean over the pool, concat the T4 side [N, 4], then the
#: 3-layer post_pool MLP -> identity [N, window=50].
ID_HIDDEN_DIM = 64
SIDE_DIM = 4
POST_POOL_LAYERS = 3

BEHAVIOR_SCALE = 5.0
MODEL_BIN_SECONDS = 0.020

# ---------------------------------------------------------------------------
# The cells.
# ---------------------------------------------------------------------------

BASELINE_CELL = "POOLED"
GATED_CELL = "POSTFUSION_MEAN"
EMA_ALPHA = 0.90

CELL_ORDER = (
    "POOLED",
    "POSTFUSION_MEAN",
    "POSTFUSION_ACCUM_UNCAPPED",
    "POSTFUSION_EMA_A090",
)

CELLS = {
    "POOLED": {
        "role": "anchor",
        "law": (
            "the sealed m4_activity_only law verbatim (rollout_g00m): D-opt-4 "
            "support activities + completed-trial FIFO capped at 30, uniform "
            "mean of the pre-pool features INSIDE the frozen B3S pooling, "
            "then the frozen fusion MLP"
        ),
        "mean_placement": "before_the_fusion_mlp",
    },
    "POSTFUSION_MEAN": {
        "role": "gated_challenger",
        "law": (
            "per completed trial i (and per support trial), identity_i = "
            "post_pool(concat(pre_pool(trial_i), T4)) with the SAME frozen "
            "weights (the frozen single-trial finalize arithmetic, exact); "
            "the deployed identity is the running float32 mean of identity_i "
            "over the sealed champion pool membership: the 4 support "
            "identities plus the identities of the last 26 completed query "
            "trials (the CAP30 FIFO mirrored in identity space; support "
            "identities are never evicted)"
        ),
        "mean_placement": "after_the_fusion_mlp",
        "pool_membership": "identical_to_POOLED_by_construction",
    },
    "POSTFUSION_ACCUM_UNCAPPED": {
        "role": "disclosed_challenger_report_only",
        "law": (
            "the m2_memory_law_scan_v1 UNIFORM_UNCAPPED accumulator law "
            "translated to identity space: init at the uniform mean of the "
            "support identities, then accumulate every completed query-trial "
            "identity, no eviction ever"
        ),
        "mean_placement": "after_the_fusion_mlp",
        "pool_membership": "support_plus_all_completed_trials",
    },
    "POSTFUSION_EMA_A090": {
        "role": "report_only_recency_sensitivity",
        "law": (
            "E <- 0.90*E + 0.10*identity_i per completed trial in identity "
            "space, E0 = the uniform mean of the support identities (the "
            "EMA_A090 precedent law in identity space)"
        ),
        "mean_placement": "after_the_fusion_mlp",
        "alpha": EMA_ALPHA,
    },
}

IDENTITY_PATH_LAW = {
    "structure": (
        "identity = post_pool(concat(mean_over_pool(pre_pool(trial)), T4)); "
        "pre_pool is Linear(100 -> 64) + ReLU per trial, post_pool is a "
        "3-layer affine stack 68 -> 64 -> 64 -> 50; verified structurally "
        "against the loaded frozen encoder at launch"
    ),
    "per_trial_identity": (
        "identity_i = the frozen encoder's own single-trial finalize "
        "arithmetic: push_trial(state) then finalize_identity with "
        "trial_count = 1 (the division by one is exact), so identity_i IS "
        "post_pool(concat(phi(trial_i), T4)) bitwise"
    ),
    "accumulation": (
        "float32 on CPU; the running identity sum accumulates in arrival "
        "order (support first, then completed trials) with FIFO eviction of "
        "the oldest COMPLETED identity at capacity 30; the deployed identity "
        "is sum / count, the same mean arithmetic the frozen pooling uses"
    ),
    "jensen_gap_disclosure": (
        "mean(MLP(phi_i)) - MLP(mean(phi_i)) is the Jensen gap of the "
        "identity path; it is nonzero whenever the frozen MLP is nonlinear "
        "on the pool dispersion, and it is reported per session at the first "
        "scored query row (the support-only pool on 11/13 sessions; the two "
        "late-start within sessions carry one committed identity)"
    ),
    "ood_disclosure": (
        "the frozen MLP was trained on pooled means of pre-pool features, so "
        "per-trial inputs are out-of-distribution; this probe measures the "
        "NET effect (Jensen gap + OOD penalty) on frozen weights.  If the "
        "gated cell is positive, a training-side cell (PIT-style per-trial "
        "exposure) becomes worth considering; if null/harmful, the placement "
        "question closes on frozen weights"
    ),
    "shortcut_law": (
        "no shortcut exists: pooling before a linear map would commute "
        "exactly, but post_pool is a 3-layer nonlinear MLP, so the two "
        "readouts differ materially -- that difference is the entire probe"
    ),
}

FROZEN_EVERYTHING_ELSE = {
    "decoder": "the frozen T4 student checkpoint (sha256 pinned by the sealed exporter)",
    "carrier": (
        "the sealed static canonical fixed-ridge T4 carrier over the D-opt-4 "
        "support (Hz domain, scaled to counts-per-bin by MODEL_BIN_SECONDS); "
        "no carrier adaptation in any cell"
    ),
    "support": (
        "the sealed M4 greedy forward D-opt four over the first-30 "
        "finite-angle candidates (the sealed champion shape, every cell)"
    ),
    "scoring": (
        "variance-weighted R2 over the sealed CDM screen's own surfaces; "
        "equal-session means across the frozen rosters"
    ),
    "training": "zero parameter updates, zero gradients, zero target access",
}

# ---------------------------------------------------------------------------
# Gates (pre-registered before any data run).
# ---------------------------------------------------------------------------

GATES = {
    "primary": {
        "expression": (
            "POSTFUSION_MEAN - POOLED paired equal-session mean delta on "
            "external_post30_local >= +0.005 AND positive external sessions "
            ">= 4/6 -> POSTFUSION_PROMISING"
        ),
        "delta_floor": 0.005,
        "breadth_min": 4,
        "breadth_denominator": 6,
        "surface": "external_post30_local",
        "cell": GATED_CELL,
    },
    "null_band": {
        "expression": "|delta| < 0.005 -> POSTFUSION_NULL",
        "width": 0.005,
    },
    "harmful": {
        "expression": "delta <= -0.005 -> POSTFUSION_HARMFUL",
        "floor": -0.005,
    },
    "boundary_epsilon": 1.0e-12,
    "boundary_rule": (
        "exact >= on float64 equal-session means; a miss inside the 1e-12 "
        "program epsilon is disclosed as within_epsilon_band_of_boundary and "
        "never flips the verdict (the Stage-O/P convention)"
    ),
    "verdict_law": {
        "POSTFUSION_PROMISING": (
            "the gated cell meets the external delta floor (+0.005) AND the "
            "external breadth floor (4/6 positive sessions); a training-side "
            "per-trial-exposure cell becomes worth considering"
        ),
        "POSTFUSION_NULL": (
            "|external delta| < 0.005 (or the delta floor is met but the "
            "breadth floor is not, disclosed as mean_pass_breadth_fail); the "
            "placement question closes on frozen weights"
        ),
        "POSTFUSION_HARMFUL": (
            "external delta <= -0.005; the post-fusion readout is actively "
            "harmful on frozen weights and the placement question closes"
        ),
    },
    "report_only_rows": (
        "POSTFUSION_ACCUM_UNCAPPED and POSTFUSION_EMA_A090 are disclosed "
        "challengers: paired deltas are reported on both surfaces but they "
        "never gate and never select"
    ),
    "within_surface_role": (
        "the within_post30 paired delta is disclosed for every cell and "
        "never selects; the verdict surface is external_post30_local"
    ),
    "nulls_reported_as_nulls": True,
    "official_contract_disclaimer": (
        "every number here is LOCAL-protocol evidence on the sealed CDM "
        "screen's own surfaces; the official M2 evaluation contract hides "
        "completed-trial boundaries and no official-contract claim is made"
    ),
}

# ---------------------------------------------------------------------------
# Anchors (the probe's binding to the sealed rows).
# ---------------------------------------------------------------------------

ANCHORS = {
    "pooled_vs_sealed_g00m_k4_rows": {
        "path": CDM_SCREEN_SCORE_RELATIVE,
        "sha256_at_design": CDM_SCREEN_SCORE_SHA256,
        "cell": "m4_activity_only",
        "exact_fields": (
            "query_starts_sha256, target_sha256, window_count (pure-data "
            "fields, device independent)"
        ),
        "r2_tolerance": 1.0e-5,
        "mean_tolerance": 1.0e-5,
        "tolerance_rationale": (
            "the sealed rows and the sealed G00m/k4 anchor means were "
            "produced on GPU; this probe is CPU-only, so float32 decode "
            "paths differ in the last bits (the memory-law scan realized "
            "1.7e-7 on the same law); any law error (wrong support, wrong "
            "surface, wrong memory) moves R2 by orders of magnitude more "
            "than 1e-5, and the realized max delta is disclosed either way"
        ),
        "anchor_means": dict(G00M_ANCHOR_MEANS),
        "anchor_means_provenance": (
            "the sealed m4_activity_only equal-session means, cross-issued "
            "as results/m2_kcurve_v1 anchor_receipt_values.g00m_means"
        ),
        "prediction_digest_disclosure": (
            "prediction_sha256 equality across devices is NOT expected and "
            "is reported informationally"
        ),
    },
    "first_scored_identity_agreement": {
        "law": (
            "at the first scored query row of every session, both uniform "
            "identity pools deploy bitwise the from-scratch sequential "
            "float32 mean over the members present at that row (the D-opt-4 "
            "support identities plus any query-trial identities committed "
            "before the first scored window -- 11/13 sessions have the "
            "support-only pool, the two late-start sessions have one "
            "committed identity); eviction cannot have occurred unless 26+ "
            "trials complete before the first scored window, in which case "
            "the capped pool is held to its own from-scratch mean within the "
            "disclosed 1e-4 float32 drift bound"
        ),
    },
    "support_pool_jensen_gap": {
        "law": (
            "descriptive only: at the first scored query row, "
            "mean_i(MLP(concat(phi_i, T4))) - MLP(concat(mean_i phi_i, T4)) "
            "(the deployed post-fusion identity minus the sealed pooled "
            "identity over the same members), reported as max/mean abs and "
            "RMS together with both identity norms; never gates"
        ),
    },
    "final_accumulation_audit": {
        "law": (
            "at the end of every post-fusion cell the incremental running "
            "sum is audited against a from-scratch sequential float32 sum "
            "over the final pool members; the uncapped pools must agree "
            "bitwise, the capped pool's eviction drift is disclosed"
        ),
    },
}

# ---------------------------------------------------------------------------
# Environment and process.
# ---------------------------------------------------------------------------

ENVIRONMENT_LAW = {
    "cuda_visible_devices": "",
    "device": "cpu",
    "cpu_only_rationale": (
        "operator instruction 2026-09-02: CPU-ONLY (CUDA_VISIBLE_DEVICES "
        "empty, torch threads 4, 0 workers) -- try on CPU first whether this "
        "is worth anything"
    ),
    "torch_num_threads": 4,
    "torch_num_interop_threads": 1,
    "dataloader_workers": 0,
    "dataloader_workers_rationale": (
        "no DataLoader is ever constructed; every dataset access is direct "
        "numpy indexing on the frozen runtime (the sealed screens' law)"
    ),
    "deterministic_algorithms": True,
    "python": "/home/xinyuan/miniconda3/envs/spint/bin/python",
    "no_user_site": True,
}

BATCH_SIZE = 1024
HARD_TIMEOUT_SECONDS = 21_600

OWNED_PATHS = (
    "tfpd_exploration/src/m2_postfusion_probe_v1/__init__.py",
    "tfpd_exploration/src/m2_postfusion_probe_v1/plan.py",
    "tfpd_exploration/src/m2_postfusion_probe_v1/gates.py",
    "tfpd_exploration/src/m2_postfusion_probe_v1/memory.py",
    "tfpd_exploration/src/m2_postfusion_probe_v1/physical.py",
    "tfpd_exploration/scripts/run_m2_postfusion_probe_v1.py",
    "tfpd_exploration/tests/test_m2_postfusion_probe_v1.py",
)

PREDECESSOR_RELATIVE = (
    "tfpd_exploration/results/cdm_p1_m2_local_v1/attempt.json",
    "tfpd_exploration/results/cdm_p1_m2_local_v1/replay.json",
    "tfpd_exploration/results/cdm_p1_m2_local_v1/terminal.json",
    "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1/score.json",
    "tfpd_exploration/results/m2_kcurve_v1/attempt.json",
    "tfpd_exploration/results/m2_kcurve_v1/replay.json",
    "tfpd_exploration/results/m2_kcurve_v1/terminal.json",
    "tfpd_exploration/results/m2_memory_law_scan_v1/attempt.json",
    "tfpd_exploration/results/m2_memory_law_scan_v1/replay.json",
    "tfpd_exploration/results/m2_memory_law_scan_v1/terminal.json",
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
)


def result_root(repo_root: Path) -> Path:
    return Path(repo_root) / RESULT_ROOT_RELATIVE
