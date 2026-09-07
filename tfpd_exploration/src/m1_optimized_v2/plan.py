"""Static contract for the new M1 carrier revision (not a sealed-P repair)."""
from __future__ import annotations

from pathlib import Path

from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import plan as old
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_config import REPO_ROOT, S_FIX_PATH, S_FIX_SHA256

DECODER_REVISION = "m1-optimized-v2"
CARRIER_REVISION = "rSyn3-refit-v1"
SCHEMA = "m1_optimized_v2_rsyn3_refit_v1"
SOURCE_SESSIONS = ("ses-20120926", "ses-20120927", "ses-20120928")
OUTER_SESSION = "ses-20120924"  # identity only; its path must never be resolved during fit.
WINDOW, N_UNITS, OUT_DIM, SUPPORT_TRIALS = 100, 64, 16, 10
SEED, OMP_THREADS = 42, 1
OLD_STAGE0_D0_SHA256 = "984f234fa54d3daeb933c9e80c6fffe8df2470525557fb1f980bddc32632fcd5"
OLD_TARGET_M10_SHA256 = old.FOLD0_M10_RAW_CARRIER_DIGEST
RESULT_ROOT = REPO_ROOT / "tfpd_exploration/results/decoder_validation_v2/20260905_190000/m1"
BANK_NPZ = RESULT_ROOT / "rSyn3-refit-v1.source-only.npz"
BANK_RECEIPT = RESULT_ROOT / "rSyn3-refit-v1.source-only.receipt.json"
DATA_DIR = REPO_ROOT / "SPINT-main/data/000941"

__all__ = [name for name in globals() if name.isupper()]
