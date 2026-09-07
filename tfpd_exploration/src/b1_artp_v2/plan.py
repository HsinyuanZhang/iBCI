"""Frozen constants for B1 ARTP-P source screen V2."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = REPO_ROOT / "tfpd_exploration" / "results" / "b1_artp_v2"
SCHEMA = "b1-artp-profile-source-screen-v2"

M = 3
TAU = 0.05
RELIABILITY_STRENGTH = 2.0
RAW_PROFILE_MIX = 0.5
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 10_000
HELD_IN_DATES = ("20210626", "20210627", "20210628")

PRIMARY_GATES = {
    "gain_vs_tpl": 3,
    "correct_vs_cyclic": 3,
    "correct_vs_neural_free": 3,
}
