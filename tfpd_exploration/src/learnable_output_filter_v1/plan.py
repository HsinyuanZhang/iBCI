"""Frozen plan of the learnable-causal-output-filter route (P0-P4).

Authority: ``docs/DESIGN_LEARNABLE_CAUSAL_OUTPUT_FILTER_20260829.md`` (§2-§11).
This module holds every pre-registered constant of the frozen-output route so
that no later stage can select a parameter from external labels:

* the sealed receipts this route binds (P2' stage A = the B0/B1 cells; the
  continuity probe = the static raw anchor);
* the fixed P1 filter (trial-local causal EMA alpha=0.25, the a-priori choice
  frozen in the P2' pre-registration — NOT the continuity probe's
  external-best arm);
* the F2 alpha grid (also the P3 oracle gain grid);
* the FIR-K4 simplex filter;
* the F4 feature whitelist, scales and parameter count;
* the §8.3 oracle disposition thresholds and the §10.2 FIR gate.

Nothing in this module reads data.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent

SCHEMA = "learnable_output_filter_v1"
DESIGN_RELATIVE = "tfpd_exploration/docs/DESIGN_LEARNABLE_CAUSAL_OUTPUT_FILTER_20260829.md"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/learnable_output_filter_v1"
CACHE_ROOT_RELATIVE = "cache/learnable_output_filter_v1"  # relative to tfpd_exploration

BUDGETS = (4, 10, 30)
SURFACES = ("external", "within")
VELOCITY_DIMS = 2

# ---------------------------------------------------------------------------
# Sealed anchors this route binds (immutable evidence; §1 of the design).
# ---------------------------------------------------------------------------

SEALED_STAGE_A_REL = "tfpd_exploration/results/learned_gate_p2prime_v1/stage_a.json"
SEALED_STAGE_A_SHA256 = (
    "ba290a7688006696f514ac35cb5b971aaf8fb00ec15a2d197d5a9c16398b3bee"
)
SEALED_PROBE_REL = "tfpd_exploration/results/continuity_probe_v1/continuity_probe_v1.json"
SEALED_PROBE_SHA256 = (
    "8afaa9109f2dbeb1fb68e1044c4e7ae275771113c4cb3daf5c5565ea77c588b1"
)

# ---------------------------------------------------------------------------
# Reset contract (§3).
# ---------------------------------------------------------------------------

RESET_POLICY_PRIMARY = "TRIAL_RESET"
RESET_POLICY_NAMED_NOT_RUN = "STREAM_GAP_RESET"
RESET_DISCLOSURE = (
    "TRIAL_RESET is primary on both surfaces: every query trial head is a hard "
    "filter-state reset. On the activity-only CDM stream the trial IDs come from "
    "the immutable input authority; on the static stream the trial blocks are "
    "recovered exactly from the window-start stream (within-trial stride is "
    "exactly 1, every cross-trial stride is >1, and the recovered block count "
    "must equal the loader's query-trial count or the stage fails closed). "
    "STREAM_GAP_RESET is named per §3.2 but not run: the formal adapter audit "
    "of §9 P7 has not happened."
)

# ---------------------------------------------------------------------------
# The P1 fixed filter (the one filter shared by all four factorial cells).
# ---------------------------------------------------------------------------

P1_FIXED_FILTER = {
    "level": "F2",
    "family": "causal_ema",
    "alpha": 0.25,
    "reset": RESET_POLICY_PRIMARY,
    "selection": (
        "alpha=0.25 is the a-priori value frozen in the sealed P2' "
        "pre-registration (its receipt records that the continuity probe's "
        "external-best K/alpha arm selection is exploratory and is not "
        "consumed). The B0/B1 cells of this route are therefore the sealed P2' "
        "stage-A A0/A1 rows, and the static A1 cell applies the exact same "
        "kernel with trial-local resets."
    ),
}

# ---------------------------------------------------------------------------
# F2 alpha grid / P3 gain grid (frozen before any external scoring; §4/§6).
# ---------------------------------------------------------------------------

ALPHA_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00)
GAIN_GRID = ALPHA_GRID  # the §8 oracle grid contains every selectable alpha
FIR_K = 4
FIR_TAPS = 4

# ---------------------------------------------------------------------------
# F4 adaptive scalar gain (§4): whitelist, frozen feature scales, params.
# ---------------------------------------------------------------------------

F4_FEATURES = (
    "innovation_norm",
    "first_difference_norm",
    "trailing_dispersion",
    "predicted_speed",
    "direction_change_angle",
    "budget",
    "activity_progress",
    "boundary_flag",
)
F4_FEATURE_SCALES = {
    "innovation_norm": 2.0,
    "first_difference_norm": 2.0,
    "trailing_dispersion": 2.0,
    "predicted_speed": 3.0,
    "direction_change_angle": 3.141592653589793,
    "budget": 30.0,
    "activity_progress": 1.0,
    "boundary_flag": 1.0,
}
F4_TRAILING_L = 8
F4_PARAM_COUNT = len(F4_FEATURES) + 1  # 8 weights + 1 bias = 9
F4_OPTIMIZER = {
    "method": "scipy.optimize.minimize/L-BFGS-B",
    "init": "zeros",
    "gradient": "analytic adjoint recursion through the scalar-gain state (exact; unit-tested against finite differences)",
    "maxiter": 200,
}

FIR_OPTIMIZER = {
    "method": "scipy.optimize.minimize/L-BFGS-B",
    "init": "uniform simplex (zero logits)",
    "gradient": "2-point finite differences (deterministic, no RNG)",
    "maxiter": 200,
}

# ---------------------------------------------------------------------------
# Gates and dispositions (§8.3, §10.2).
# ---------------------------------------------------------------------------

ORACLE_STOP_BELOW = 0.005
ORACLE_PROCEED_ABOVE = 0.015
FIR_GATE_OVER_F1_F2 = 0.005
P4_GATE_OVER_FIXED = 0.005  # source-grouped OOF null threshold (§10.3 context)

# ---------------------------------------------------------------------------
# Conventions.
# ---------------------------------------------------------------------------

BOOTSTRAP_SEED = 42
BOOTSTRAP_DRAWS = 10000
STATIC_ANCHOR_TOLERANCE = 0.0  # CPU-to-CPU reproduction is measured exactly 0
CDM_ANCHOR_TOLERANCE = 0.0    # same GPU/profile => bit-equal R2 required
GPUS_ALLOWED_FOR_CDM_CACHE = (0, 1)
DATA_ROOT_ENV = {
    "SUBC_DATA_ROOT": "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C",
    "SUBM_DATA_ROOT": "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M",
}

CLOSURE_PATTERNS = (
    "src/learnable_output_filter_v1/plan.py",
    "src/learnable_output_filter_v1/ladder.py",
    "src/learnable_output_filter_v1/metrics.py",
    "src/learnable_output_filter_v1/selection.py",
    "src/learnable_output_filter_v1/oracle.py",
    "src/learnable_output_filter_v1/streams.py",
    "src/learnable_output_filter_v1/audit.py",
    "src/learnable_output_filter_v1/stages.py",
    "src/learnable_output_filter_v1/receipts.py",
    "scripts/run_learnable_output_filter_v1.py",
    "tests/test_learnable_output_filter_v1.py",
)


def pre_registration_payload() -> dict[str, object]:
    return {
        "design_authority": DESIGN_RELATIVE,
        "route": "frozen_output_P0_P4",
        "budgets": list(BUDGETS),
        "surfaces": list(SURFACES),
        "reset_policy_primary": RESET_POLICY_PRIMARY,
        "reset_policy_named_not_run": RESET_POLICY_NAMED_NOT_RUN,
        "reset_disclosure": RESET_DISCLOSURE,
        "p1_fixed_filter": dict(P1_FIXED_FILTER),
        "alpha_grid": list(ALPHA_GRID),
        "gain_grid": list(GAIN_GRID),
        "fir": {"K": FIR_K, "taps": FIR_TAPS, "optimizer": dict(FIR_OPTIMIZER)},
        "f4": {
            "features": list(F4_FEATURES),
            "feature_scales": dict(F4_FEATURE_SCALES),
            "trailing_L": F4_TRAILING_L,
            "param_count": F4_PARAM_COUNT,
            "optimizer": dict(F4_OPTIMIZER),
            "so2_rule": "one scalar gain shared by both velocity dimensions",
        },
        "gates": {
            "oracle_stop_below": ORACLE_STOP_BELOW,
            "oracle_proceed_above": ORACLE_PROCEED_ABOVE,
            "fir_gate_over_f1_f2": FIR_GATE_OVER_F1_F2,
            "p4_null_threshold_over_fixed": P4_GATE_OVER_FIXED,
        },
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS},
        "sealed_anchors": {
            "stage_a": {"rel": SEALED_STAGE_A_REL, "sha256": SEALED_STAGE_A_SHA256},
            "continuity_probe": {"rel": SEALED_PROBE_REL, "sha256": SEALED_PROBE_SHA256},
        },
        "inference_only": True,
        "target_optimizer_backward_update": 0,
        "j2_j3_training": "NOT AUTHORIZED in this route; spec only",
    }


def owned_sha256s(base: Path) -> dict[str, str]:
    """SHA256 of every route-owned module (attempt-before-data binding)."""
    owned: dict[str, str] = {}
    for relative in CLOSURE_PATTERNS:
        path = base / "tfpd_exploration" / relative
        owned[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return owned
