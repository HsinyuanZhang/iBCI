"""Optimize the sealed P1 hold-vs-reach FiLM toward an EvalAI-legal cached identity."""

from pathlib import Path

SCHEMA = "m2_p1_contrast_opt_v1"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_P1_CONTRAST_OPT_V1_20260903.md"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_p1_contrast_opt_v1"
PROBE_RESULT_RELATIVE = "tfpd_exploration/results/m2_hold_film_probe_v1"
SEALED_P0_EXTERNAL = 0.29521985196829853
SEALED_P1_EXTERNAL = 0.30862991970180154
SEALED_P1_GAIN = SEALED_P1_EXTERNAL - SEALED_P0_EXTERNAL
OFFICIAL_T4_578221_EXTERNAL = 0.30324395
OFFICIAL_T4_578221_LATENCY = 0.04290260
P1_FULL_ABS_TOLERANCE = 1.0e-6
KEEP_FRACTION = 0.80
GATE_MEAN = 0.005
GATE_POSITIVE = 4
M33_HORIZON = 33
DEFAULT_GPU_INDEX = 1
FILM_PARAMS = 1224
FILM_MAC_PER_SESSION = 96 * (8 * 8 + 8 * 128)

MASKS = {
    "p1_full": (1.0, 1.0, 1.0, 1.0),
    "delta": (1.0, 0.0, 0.0, 0.0),
    "logratio": (0.0, 1.0, 0.0, 0.0),
    "holdstd": (0.0, 0.0, 1.0, 0.0),
    "reachstd": (0.0, 0.0, 0.0, 1.0),
    "means": (1.0, 1.0, 0.0, 0.0),
    "stds": (0.0, 0.0, 1.0, 1.0),
    "drop_delta": (0.0, 1.0, 1.0, 1.0),
    "drop_logratio": (1.0, 0.0, 1.0, 1.0),
    "drop_holdstd": (1.0, 1.0, 0.0, 1.0),
    "drop_reachstd": (1.0, 1.0, 1.0, 0.0),
}


def result_root(repo_root: Path) -> Path:
    return repo_root / RESULT_ROOT_RELATIVE


def probe_root(repo_root: Path) -> Path:
    return repo_root / PROBE_RESULT_RELATIVE
