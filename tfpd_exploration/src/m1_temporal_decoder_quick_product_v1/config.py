"""M1 temporal product constants. New root; do not overwrite P or S-Fix results."""

from __future__ import annotations

from pathlib import Path

from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_config import (
    FOLD0_SOURCES,
    FOLD0_TARGET,
    M1_TEMPORAL,
    S_FIX_PATH,
    S_FIX_SHA256,
)

PRODUCT_SCHEMA = "m1_temporal_decoder_quick_product_v1"
CONTRACT_VERSION = 1
SEED = 42
EFFECTIVE_BATCH = 32
LR = 1.0e-4
WEIGHT_DECAY = 1.0e-2
GRAD_CLIP = 1.0
EMA_DECAY = 0.9995
EPOCHS_TARGET = 12
WARMUP_EPOCHS = 1
PREDICTION_DIVISOR = 1.0
WINDOW = M1_TEMPORAL.window
N_UNITS = M1_TEMPORAL.n_units
OUT_DIM = M1_TEMPORAL.out_dim

PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = REPO_ROOT / "tfpd_exploration/results/m1_temporal_decoder_quick_product_v1"
SLOT_ROOT = REPO_ROOT / "tfpd_exploration/results/six_evalai_slots_v1/20260905_155800"
PROFILE_PATH = RESULT_ROOT / "m1_cost_profile.json"

TRAIN_SESSIONS = FOLD0_SOURCES
OUTER_SESSION = FOLD0_TARGET
