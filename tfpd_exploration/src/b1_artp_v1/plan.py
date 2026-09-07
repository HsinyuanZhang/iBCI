"""Frozen constants for the first B1 ARTP source screen."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = REPO_ROOT / "tfpd_exploration" / "results" / "b1_artp_v1"
SCHEMA = "b1-artp-source-screen-v1"

M = 3
SIGNATURE = "population-time-mean-std-l2-v1"
TAU = 0.05
RELIABILITY_STRENGTH = 2.0
MIXING = 1.0
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 10_000

# These nearby settings were frozen with the primary to show that the source
# result is not an isolated grid point.  They are descriptive, never selected
# per held-out date, and may not replace the primary after hidden results.
NEIGHBOR_CONFIGS = (
    (0.02, 4.0, 1.0),
    (0.005, 16.0, 1.0),
)

HELD_IN_DATES = ("20210626", "20210627", "20210628")
HELD_OUT_DATES = ("20210630", "20210701", "20210705")
ALL_DATES = HELD_IN_DATES + HELD_OUT_DATES

PRIMARY_GATES = {
    "gain_vs_tpl_m3_median_strictly_positive_dates": 3,
    "correct_vs_cyclic_query_signature_strictly_positive_dates": 3,
    "correct_vs_neural_free_reliability_strictly_positive_dates": 3,
}
