from pathlib import Path

from mc_maze.dandi688_cp_film_postpool_v1 import plan as parent
from mc_maze.dandi688_cp_film_v1 import plan as substrate

SCHEMA = "dandi688_cp_film_postpool_v2"
DESIGN_RELATIVE = "sua_exploration/docs/DESIGN_DANDI_000688_CP_FILM_POSTPOOL_COADAPT_V2_20260904.md"
DESIGN_SHA256 = "63151aa942ee4a0570284171c01216d394126a84d4310109125fc8ae42d56a33"
WORKORDER_RELATIVE = "sua_exploration/docs/WORKORDER_DANDI_000688_CP_FILM_POSTPOOL_COADAPT_V2_20260904.md"
WORKORDER_SHA256 = "2d9470392bb814c738b267ce5f65f65bc309209fc47802d4ffedb1a3bb82c321"
INCIDENT_RELATIVE = "sua_exploration/docs/INCIDENT_DANDI_000688_CP_FILM_POSTPOOL_V1_FROZEN_POSTPOOL_20260904.md"
INCIDENT_SHA256 = "64df63efcaab813ee6a0519adbbd734b7231cb03ba38ecde2d76c2fae7c71e25"
V1_ROOT_RELATIVE = "sua_exploration/results/dandi688_cp_film_postpool_v1/seed42"
V1_BODY_SHA256 = {
    "attempt.json": "f69cd7fd6078cca46769dc3094c6a17df241d7a5cbde3b0cdb57dcb8edc2d7e4",
    "source_authority.json": "da3ebf1ff02a2397165c08f55905ec32ac8f09581caf8232e5a239015bb85907",
    "training.json": "8ae1ef8af47dd34ae234ba242696cc4e200eb94a8f624a45cdccf9c897d0328d",
    "score.json": "89f40f3e3e89cda1877b95115839cad09fb7c34b531e9da5192199847665efcb",
}
RESULT_PARENT_RELATIVE = "sua_exploration/results/dandi688_cp_film_postpool_v2"

SEEDS = parent.SEEDS
ARMS = parent.ARMS
AVERAGE_EPOCHS = parent.AVERAGE_EPOCHS
TRAINABLE_PARAMETERS = parent.TRAINABLE_PARAMETERS
POST_POOL_PARAMETERS = parent.POST_POOL_PARAMETERS


def result_root(repo_root: Path, seed: int) -> Path:
    return Path(repo_root) / RESULT_PARENT_RELATIVE / f"seed{seed}"

