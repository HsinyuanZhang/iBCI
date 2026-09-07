"""Frozen B1-SFCJ V1 constants. Do not change without a new work order."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "SPINT-main" / "data" / "001046"
RESULTS_ROOT = REPO_ROOT / "tfpd_exploration" / "results" / "b1_sfcj_v1"
DESIGN_RELATIVE = "tfpd_exploration/docs/DESIGN_B1_SPECTRAL_FUNCTIONAL_CARRIER_JR1_V1_20260903.md"

N_CHANNELS = 85
N_FREQ = 158
N_SPEC_FRAMES = 880
N_NEURAL_SAMPLES = 27000
N_MS_BINS = 900
SAMPLES_PER_MS = 30
FS_NEURAL = 30000
NEURAL_DT = 1.0 / FS_NEURAL
VALID_START = 90
VALID_END = 790
N_VALID = 700
SPEC_T0 = 0.01024
SPEC_DT = 0.001
RAW_SPECTROGRAM_MIN = 1.0
CARRIER_DIM = 9
D_MODEL = 512
N_HEADS = 64
N_LAYERS = 1
FREQ_RANGE_HZ = (292.96875, 7958.984375)

LAG_GRID_MS = (0, 10, 20, 30, 40, 50, 60)
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)
SFC_RIDGE_LAMBDA = 1.0
BOX_MS = 5
SEEDS = (42, 43)

HELD_IN_DATES = ("20210626", "20210627", "20210628")
HELD_OUT_DATES = ("20210630", "20210701", "20210705")
ALL_DATES = HELD_IN_DATES + HELD_OUT_DATES

CALIB_TRIAL_COUNTS = {
    "20210626": 11,
    "20210627": 29,
    "20210628": 8,
    "20210630": 3,
    "20210701": 3,
    "20210705": 3,
}
MINIVAL_TRIAL_COUNTS = {"20210626": 2, "20210627": 2, "20210628": 2}

FOLDS = (
    {
        "fold": 0,
        "train_dates": ("20210627", "20210628"),
        "val_date": "20210626",
        "k_train_max": 28,
        "n_query": 10,
        "n_in_range": 10,
        "n_ood": 0,
    },
    {
        "fold": 1,
        "train_dates": ("20210626", "20210628"),
        "val_date": "20210627",
        "k_train_max": 10,
        "n_query": 28,
        "n_in_range": 8,
        "n_ood": 20,
    },
    {
        "fold": 2,
        "train_dates": ("20210626", "20210627"),
        "val_date": "20210628",
        "k_train_max": 28,
        "n_query": 7,
        "n_in_range": 7,
        "n_ood": 0,
    },
)

ARM_SPECS = (
    ("A0-NATIVE", "native", "zero"),
    ("N-SFC4", "native", "sfc4"),
    ("N-SFC9", "native", "sfc9"),
    ("J0", "jr1", "zero"),
    ("J-SFC4", "jr1", "sfc4"),
    ("J-SFC9", "jr1", "sfc9"),
)

TPL_AGGREGATIONS = ("raw_mean", "log_mean", "median")

EXPECTED_FALCON_SHA256 = {
    "config.py": "3565e4449b7fc1f55a57e7c202083bfc6094ba55e37df2956851b7865a1c6ca9",
    "dataloaders.py": "7638d839ec2085c7dde71ec9ca9676cbf1256c79fdcb327c4ac6487bfe5c9b05",
    "evaluator.py": "2b848f84ef620eac82ce53d331d03727d9ae4ea9a5f2e5edb6e6557537ce4418",
    "interface.py": "781dae89351d2c5e5914c8c9b47654254edd4f5a308a3bda85b4d91a755063e6",
}

FALCON_README_URL = "https://raw.githubusercontent.com/snel-repo/falcon-challenge/main/README.md"
EVALAI_OVERVIEW_URL = "https://eval.ai/web/challenges/challenge-page/2319/overview"
