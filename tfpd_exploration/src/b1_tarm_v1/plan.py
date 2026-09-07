"""Frozen constants for the first B1 TARM source-only pilot."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = REPO_ROOT / "tfpd_exploration" / "results" / "b1_tarm_v1"

SCHEMA = "b1-tarm-v1-pilot"
PILOT_FOLD = 0
PILOT_SEED = 42
PILOT_EPOCHS = 12
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0
UNIT_DROPOUT_P = 0.10
RESIDUAL_LOSS_WEIGHT = 1e-4

# Keep the SPINT geometry while reducing the over-parameterized B1 prototype
# enough for the small source-LODO sample count.
D_MODEL = 256
N_HEADS = 16
N_LAYERS = 1

PROFILE_Q = 8
PROFILE_GAMMA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

# Complete 2x2: native/J-R1 x zero/source-prior SFC9.
ARMS = (
    ("TA-N0", "native", "zero"),
    ("TA-NS9", "native", "spsfc9"),
    ("TA-J0", "jr1", "zero"),
    ("TA-JS9", "jr1", "spsfc9"),
)

PRIMARY_LAW = "GROWING"
CONTROL_LAW = "FIXED3"

