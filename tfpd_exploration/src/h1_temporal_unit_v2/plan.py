"""Named H1 revision: train in the same units used at deploy.

Old product trained on raw y and divided by 20 only at score/container time.
This revision keeps the deploy /20 convention and trains against 20y.
Existing e5/e12 checkpoints stay read-only diagnostic material.
"""

from __future__ import annotations

from pathlib import Path

from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.config import (
    HELDIN_SESSIONS,
    RESULT_ROOT as OLD_PRODUCT_ROOT,
    SEED,
    WINDOW,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REVISION = "H1-TEMPORAL-unit-v2"
SCHEMA = "h1_temporal_unit_v2"
RESULT_ROOT = REPO_ROOT / "tfpd_exploration/results/h1_temporal_unit_v2/20260905_184500"
OLD_PRODUCT_ROOT = OLD_PRODUCT_ROOT
DEPLOY_DIVISOR = 20.0
TRAIN_TARGET_SCALE = 20.0
SEED = SEED
WINDOW = WINDOW
HELDIN_SESSIONS = HELDIN_SESSIONS
PROBE_SESSION = "ses-19250101T111740"
PROBE_N = 16
OVERFIT_STEPS = 400
NO_SUBMIT = True
