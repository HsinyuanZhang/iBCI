"""Named revision for S5/S6. Does not restore P parent-carrier identity."""

from __future__ import annotations

from pathlib import Path

from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import plan as fold_plan
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_config import REPO_ROOT, S_FIX_PATH, S_FIX_SHA256

__all__ = [
    "REPO_ROOT",
    "REVISION",
    "CARRIER_NAME",
    "SCHEMA",
    "RESULT_ROOT",
    "BANK_NPZ",
    "BANK_RECEIPT",
]

REVISION = "M1-TEMPORAL-v2"
CARRIER_NAME = "rSyn3-refit-v1"
SCHEMA = "m1_temporal_v2_rsyn3_refit_v1"
RESULT_ROOT = REPO_ROOT / "tfpd_exploration/results/m1_temporal_v2/20260905_183000"
BANK_NPZ = RESULT_ROOT / "rsyn3_refit_v1_source_bank.npz"
BANK_RECEIPT = RESULT_ROOT / "rsyn3_refit_v1_source_bank.receipt.json"
OLD_TARGET_M10_DIGEST = fold_plan.FOLD0_M10_RAW_CARRIER_DIGEST
OLD_D0_DIGEST = "984f234fa54d3daeb933c9e80c6fffe8df2470525557fb1f980bddc32632fcd5"
SOURCE_SESSIONS = tuple(fold_plan.FOLD0_SOURCE_SESSIONS)
TARGET_SESSION = fold_plan.FOLD0_TARGET_SESSION
SUPPORT_TRIALS = fold_plan.SUPPORT_TRIALS
SEED = 42
OMP_THREADS = 1
S_FIX_PATH = S_FIX_PATH
S_FIX_SHA256 = S_FIX_SHA256
DATA_DIR = REPO_ROOT / fold_plan.DATA_DIR_RELATIVE
