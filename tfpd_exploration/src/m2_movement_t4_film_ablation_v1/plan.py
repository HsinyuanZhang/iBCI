"""Constants for the matched MOVE-T4 FiLM ablation."""

from pathlib import Path

from tfpd_exploration.src.m2_film_content_ablation_v1 import plan as whole_plan

SCHEMA = "m2_movement_t4_film_ablation_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_movement_t4_film_ablation_v1"
MOVE_P0_EXTERNAL = 0.32545164641426183
MOVE_P0_TOLERANCE = 1.0e-6

SEEDS = whole_plan.SEEDS
HORIZON = whole_plan.HORIZON
FILM_INPUT = whole_plan.FILM_INPUT
CONTRAST_MASK = whole_plan.CONTRAST_MASK
EMPTY_MASK = whole_plan.EMPTY_MASK
EPOCHS = whole_plan.EPOCHS
LEARNING_RATE = whole_plan.LEARNING_RATE
WINDOWS_PER_SESSION = whole_plan.WINDOWS_PER_SESSION
GPU_INDEX = whole_plan.GPU_INDEX


def result_root(repo_root: Path) -> Path:
    return repo_root / RESULT_ROOT_RELATIVE


def seed_root(repo_root: Path, seed: int) -> Path:
    return result_root(repo_root) / f"seed{int(seed)}"

