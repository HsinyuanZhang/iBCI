"""Follow-up catalog for contrast-only FiLM on native M33."""

from pathlib import Path

from tfpd_exploration.src.m2_hold_film_probe_v1 import plan as probe_plan
from tfpd_exploration.src.m2_means_squeeze_v1 import plan as squeeze_plan

SCHEMA = "m2_contrast_only_squeeze_v1"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_CONTRAST_ONLY_SQUEEZE_V1_20260904.md"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_contrast_only_squeeze_v1"

CHECKPOINT_SHA256 = probe_plan.CHECKPOINT_SHA256
NORMALIZATION_SHA256 = probe_plan.NORMALIZATION_SHA256
P0_M33_EXTERNAL = squeeze_plan.P0_M33_EXTERNAL
PRIOR_CONTRAST_ONLY_EXTERNAL = 0.31072571859539394
WINNER_T4_PLUS_CONTRAST_EXTERNAL = 0.3210151173118348

M33_HORIZON = squeeze_plan.M33_HORIZON
T4_MODE = squeeze_plan.T4_MODE
DEFAULT_GPU_INDEX = 1
MASK = squeeze_plan.MASKS["means"]
FILM_INPUT = "contrast_only"


def _row(
    name: str,
    *,
    epochs: int,
    learning_rate: float,
    seed: int = 42,
) -> dict[str, object]:
    return {
        "name": name,
        "mask": "means",
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "film_input": FILM_INPUT,
        "windows_per_session": 256,
        "horizon": M33_HORIZON,
        "t4_mode": T4_MODE,
    }


CONFIGS: tuple[dict[str, object], ...] = (
    _row("contrast_only_ep12_lr3e4_s42", epochs=12, learning_rate=3.0e-4),
    _row("contrast_only_ep12_lr1e4_s43", epochs=12, learning_rate=1.0e-4, seed=43),
    _row("contrast_only_ep18_lr1e4_s42", epochs=18, learning_rate=1.0e-4),
)


def result_root(repo_root: Path) -> Path:
    return repo_root / RESULT_ROOT_RELATIVE


def config_by_name(name: str) -> dict[str, object]:
    for row in CONFIGS:
        if row["name"] == name:
            return row
    raise KeyError(name)
