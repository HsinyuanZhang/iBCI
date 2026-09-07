"""Frozen framewise B1 TARM pilot constants."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = REPO_ROOT / "tfpd_exploration" / "results" / "b1_tarm_frame_v1"
SCHEMA = "b1-tarm-frame-v1"

FOLD = 0
SEED = 42
EPOCHS = 12
WINDOW = 64
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
UNIT_DROPOUT_P = 0.10
GRAD_CLIP_NORM = 1.0
RESIDUAL_LOSS_WEIGHT = 1e-4

ARMS = (
    ("F-TA-N0", "native", "zero"),
    ("F-TA-NS9", "native", "spsfc9"),
    ("F-TA-J0", "jr1", "zero"),
    ("F-TA-JS9", "jr1", "spsfc9"),
)

