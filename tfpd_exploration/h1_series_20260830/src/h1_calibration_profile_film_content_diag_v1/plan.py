"""Frozen constants for the H1 FiLM content diagnostic (V5)."""
from __future__ import annotations

SCHEMA = "h1_calibration_profile_film_content_diag_v1"
MODES = ("full", "empty", "rowshuffle")

# EP-ROWSHUFFLE: one fixed unit-row permutation (176 units), seed-fixed,
# applied identically at training and scoring.
PERM_SEED = 20260904
PERM_UNITS = 176

# Sealed V2 references (the numbers the diagnostic is paired against).
V2_ROOT_RELATIVE = "tfpd_exploration/h1_series_20260830/results/h1_calibration_profile_film_v2"
V2_SCORE_SHA256 = "8b56b105bf1cd965ffa9d74e323dbce7b5d2e2410a8abfb88b5eaa2151494813"
SEALED_V2_OOF_MEAN_GAIN = 0.02392937742418233
SEALED_V2_OOF_NONNEGATIVE_DATES = 4
V2_INITIAL_FILM_STATE_SHA256 = "ff2263b5705a499caeb5fe807f9889479fdbc31f806b42de82b11d351ba210cf"
# V2's sealed first-fold FINAL film states (strong drift canary; informational
# under cross-GPU kernels).
V1_FIRST_FOLD_FILM_STATE_SHA256 = {
    "EP-FILM": "60eb1161d9621b0830d226e0eaf07aefb0ee1f56eb72d42880e95aa518bc6b58",
    "LP-FILM": "fba420f403b248ee0a6fe6c1898f704dc6ae4be0be03d3a92324771afe979736",
}

# Preregistered decision thresholds (WORKORDER V5 §5).
DRIFT_TOLERANCE = 0.005          # |rerun FILM gain - sealed| bound
CONTENT_FLOOR = 0.002            # below this, profile content is absent
GATE_MEAN = 0.005                # V2 gate constants, reused verbatim
GATE_NONNEGATIVE_DATES = 4
GATE_WORST = -0.010
LP_CANARY_MEAN_TOLERANCE = 0.002
LP_CANARY_MAX_TOLERANCE = 0.005

RESULT_ROOT_RELATIVE = "tfpd_exploration/h1_series_20260830/results/h1_calibration_profile_film_content_diag_v1"

__all__ = ("SCHEMA",)
