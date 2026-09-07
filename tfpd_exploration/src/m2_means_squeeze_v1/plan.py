"""Frozen constants and the EvalAI-max squeeze catalog."""

from pathlib import Path

from tfpd_exploration.src.m2_hold_film_probe_v1 import plan as probe_plan

SCHEMA = "m2_means_squeeze_v2"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_MEANS_SQUEEZE_V1_20260903.md"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_means_squeeze_v2"
OPT_V1_ROOT_RELATIVE = "tfpd_exploration/results/m2_p1_contrast_opt_v1"
PROBE_RESULT_RELATIVE = "tfpd_exploration/results/m2_hold_film_probe_v1"

CHECKPOINT_SHA256 = probe_plan.CHECKPOINT_SHA256
NORMALIZATION_SHA256 = probe_plan.NORMALIZATION_SHA256
P0_M33_EXTERNAL = 0.2991329172968798
MEANS_M33_BASELINE = 0.3190522122162025
OFFICIAL_T4_578221_EXTERNAL = 0.30324395

M33_HORIZON = 33
T4_MODE = "native"
DEFAULT_GPU_INDEX = 1
GATE_MEAN = 0.005
GATE_POSITIVE = 4
SHUFFLE_IS_REJECT = False
EVALAI_PUSH_THIS_CELL = True
WARM_START_T4_PLUS_CONTRAST = "probe_p0"

MASKS = {
    "means": (1.0, 1.0, 0.0, 0.0),
    "full4d": (1.0, 1.0, 1.0, 1.0),
    "delta": (1.0, 0.0, 0.0, 0.0),
    "logratio": (0.0, 1.0, 0.0, 0.0),
    "means_reachstd": (1.0, 1.0, 0.0, 1.0),
}

MASK_COMPLEXITY = {
    "delta": 1,
    "logratio": 1,
    "means": 2,
    "means_reachstd": 3,
    "full4d": 4,
}

FILM_INPUT_COMPLEXITY = {
    "t4_plus_contrast": 0,
    "contrast_only": 1,
}


def _row(
    name: str,
    *,
    mask: str,
    epochs: int,
    learning_rate: float,
    seed: int = 42,
    film_input: str = "t4_plus_contrast",
    windows_per_session: int = 256,
) -> dict[str, object]:
    return {
        "name": name,
        "mask": mask,
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "film_input": film_input,
        "windows_per_session": int(windows_per_session),
        "horizon": M33_HORIZON,
        "t4_mode": T4_MODE,
    }


# Provisional catalog; Fable 5.1 may swap rows but keep <=12 and the replica first.
CONFIGS: tuple[dict[str, object], ...] = (
    _row("means_ep6_lr1e4_s42", mask="means", epochs=6, learning_rate=1.0e-4),
    _row("means_ep12_lr1e4_s42", mask="means", epochs=12, learning_rate=1.0e-4),
    _row("means_ep18_lr1e4_s42", mask="means", epochs=18, learning_rate=1.0e-4),
    _row("means_ep6_lr3e4_s42", mask="means", epochs=6, learning_rate=3.0e-4),
    _row("means_ep12_lr3e4_s42", mask="means", epochs=12, learning_rate=3.0e-4),
    _row("full4d_ep6_lr1e4_s42", mask="full4d", epochs=6, learning_rate=1.0e-4),
    _row("full4d_ep12_lr1e4_s42", mask="full4d", epochs=12, learning_rate=1.0e-4),
    _row("means_contrast_only_ep12_lr1e4_s42", mask="means", epochs=12, learning_rate=1.0e-4, film_input="contrast_only"),
    _row("delta_ep12_lr1e4_s42", mask="delta", epochs=12, learning_rate=1.0e-4),
    _row("logratio_ep12_lr1e4_s42", mask="logratio", epochs=12, learning_rate=1.0e-4),
    _row("means_reachstd_ep12_lr1e4_s42", mask="means_reachstd", epochs=12, learning_rate=1.0e-4),
    _row("means_ep12_lr1e4_s43", mask="means", epochs=12, learning_rate=1.0e-4, seed=43),
)


def result_root(repo_root: Path) -> Path:
    return repo_root / RESULT_ROOT_RELATIVE


def opt_v1_root(repo_root: Path) -> Path:
    return repo_root / OPT_V1_ROOT_RELATIVE


def config_by_name(name: str) -> dict[str, object]:
    for row in CONFIGS:
        if row["name"] == name:
            return row
    if name == "sealed_means_m33":
        return _row("sealed_means_m33", mask="means", epochs=6, learning_rate=1.0e-4)
    raise KeyError(name)
