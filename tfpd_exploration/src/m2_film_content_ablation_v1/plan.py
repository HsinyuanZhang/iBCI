"""Frozen constants for the official-grid M2 FiLM content ablation."""

from pathlib import Path

from tfpd_exploration.src.m2_hold_film_probe_v1 import plan as probe_plan
from tfpd_exploration.src.m2_means_squeeze_v1 import plan as squeeze_plan

SCHEMA = "m2_film_content_ablation_v3"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_film_content_ablation_v3"

CHECKPOINT_SHA256 = probe_plan.CHECKPOINT_SHA256
NORMALIZATION_SHA256 = probe_plan.NORMALIZATION_SHA256
P0_M33_EXTERNAL = squeeze_plan.P0_M33_EXTERNAL
OFFICIAL_CONFIG_NAME = "means_ep12_lr3e4_t4_plus_contrast"
OFFICIAL_EVALAI_SUBMISSION = 581801
OFFICIAL_EVALAI_HELDOUT_R2 = 0.3203
LOCAL_CANONICAL_REAL_SEED42 = 0.3210151173118348
LOCAL_REPRO_TOLERANCE = 1.0e-6

HORIZON = 33
T4_MODE = "native"
FILM_INPUT = "t4_plus_contrast"
CONTRAST_MASK = (1.0, 1.0, 0.0, 0.0)
EMPTY_MASK = (0.0, 0.0, 0.0, 0.0)
EPOCHS = 12
LEARNING_RATE = 3.0e-4
WINDOWS_PER_SESSION = 256
SEEDS = (42, 43, 44)
GPU_INDEX = 0


def result_root(repo_root: Path) -> Path:
    return repo_root / RESULT_ROOT_RELATIVE


def seed_root(repo_root: Path, seed: int) -> Path:
    return result_root(repo_root) / f"seed{int(seed)}"
