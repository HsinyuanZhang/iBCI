"""Frozen constants for the post-pool co-adaptation successor."""
from pathlib import Path

from mc_maze.dandi688_cp_film_v1 import plan as parent

SCHEMA = "dandi688_cp_film_postpool_v1"
DESIGN_RELATIVE = "sua_exploration/docs/DESIGN_DANDI_000688_CP_FILM_POSTPOOL_COADAPT_V1_20260904.md"
DESIGN_SHA256 = "463ba1ef07ecf37307c03f20bec1bf6ec04d9dac7fc3fd89dd7e1d93b31a4411"
WORKORDER_RELATIVE = "sua_exploration/docs/WORKORDER_DANDI_000688_CP_FILM_POSTPOOL_COADAPT_V1_20260904.md"
WORKORDER_SHA256 = "ec37d69dcacb21a417ac753399f6f62132e9316ae8f53258d55a492ea7d2e358"
RESULT_PARENT_RELATIVE = "sua_exploration/results/dandi688_cp_film_postpool_v1"

SEEDS = parent.SEEDS
ARMS = parent.ARMS
EPOCHS = parent.EPOCHS
AVERAGE_EPOCHS = (9, 10, 11, 12)
BATCH_SIZE = parent.BATCH_SIZE
LEARNING_RATE = parent.LEARNING_RATE
BEHAVIOR_SCALE = parent.BEHAVIOR_SCALE
VALIDATION_SESSIONS = parent.VALIDATION_SESSIONS
FILM_PARAMETERS = parent.FILM_PARAMETERS
POST_POOL_PARAMETERS = 11826
TRAINABLE_PARAMETERS = FILM_PARAMETERS + POST_POOL_PARAMETERS
PROFILE_UTILITY_DELTA_VS_EMPTY = 0.002
POSITIVE_SESSIONS = 4
SOLID_DELTA_VS_NATIVE = 0.010


def result_root(repo_root: Path, seed: int) -> Path:
    return Path(repo_root) / RESULT_PARENT_RELATIVE / f"seed{seed}"

