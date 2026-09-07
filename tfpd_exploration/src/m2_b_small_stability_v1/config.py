"""Immutable SmallConfig and frozen paths for m2_b_small_stability_v1.

Do not assign into ``m2_dual_track_v1.plan`` globals. Widths live here only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SCHEMA = "m2_b_small_stability_v1"
CONTRACT_VERSION = 1
WORKORDER_RELATIVE = (
    "tfpd_exploration/docs/"
    "WORKORDER_M2_B_SMALL_TRANSFORMER_STABILIZATION_INTERIM_V1_20260905.md"
)
WORKORDER_SHA256 = "bb0df697f353e109b0d470e1f3030146b79a74f05683e4ff4ee4b0a012cd578b"

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = REPO_ROOT / "tfpd_exploration/results/m2_b_small_stability_v1"
INTERIM_RUN_ROOT = RESULT_ROOT / "20260905_123000"
LR_CONTRAST_RUN_ROOT = RESULT_ROOT / "20260905_131500"
ACTIVE_RUN_ROOT = INTERIM_RUN_ROOT
OLD_ROOT = REPO_ROOT / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500"
OLD_MANIFEST_24_PATH = OLD_ROOT / "sampler" / "shuffled_batch_manifest_24.json"
OLD_MANIFEST_12_PATH = OLD_ROOT / "sampler" / "shuffled_batch_manifest.json"

OLD_ANALYSIS_SHA256 = "af64ee40fa5dc3c0a4036f1d67c2ea48e92866b33d5803d1f48309d2ce780d60"
OLD_COMPARISON_SHA256 = "94ab3271896a01feb68b090c1896e4a893c92ad46556d803b4b122965977a95a"

MANIFEST_24_DIGEST = "a95255fa339ea06f1e1cc3ef9f53f3d49d15799bf95e4579f672417fd878d79a"
MANIFEST_12_DIGEST = "0c1808baa4d6722742a3705963c60b55fd7fcc7f3b2d03c2c416e3e0d2937a95"

CHAMPION_CKPT_SHA256 = "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
SELECTED_HEAD_STATE_SHA256 = "13551d3fc33d1cc296670c577c11519c04abdb42fa081f7d061191bb5610e6c5"
CHAMPION_NORMALIZATION_SHA256 = "d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e"

CELL_S0 = "S0-SMALL-LEGACY"
CELL_S1 = "S1-SMALL-COS"
CELL_N0 = "N0-SMALL-LEGACY-1E4"
CELL_N1 = "N1-SMALL-COS-1E4"
PRIMARY_CANDIDATE = "S1-SMALL-COS/EMA"
PRIMARY_CANDIDATE_LR = "N1-SMALL-COS-1E4/EMA"
SCHEDULE_LEGACY = (CELL_S0, CELL_N0)
SCHEDULE_COSINE = (CELL_S1, CELL_N1)
SEED = 42

DECODER_PARAMS = 3_543_010
TEMPORAL_PARAMS = 2_108_416
FRONTEND_READOUT_PARAMS = 1_434_594
SHARED_CALIB_PARAMS = 19_514
DECODER_PLUS_CALIB_PARAMS = 3_562_524

UPDATES_PER_EPOCH = 3165
TOTAL_UPDATES = 75960
EPOCHS = 24

LR_MAX = 3.0e-4
LR_MIN = 3.0e-5
LR_MAX_1E4 = 1.0e-4
LR_MIN_1E4 = 1.0e-5
WEIGHT_DECAY = 1.0e-2
GRAD_CLIP = 1.0
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1.0e-8
EMA_DECAY = 0.9995

R_REF_SESSION = 0.3582396424175502
R_REF_DATE = 0.3279795072

TRAIN_ENV_FLAG = "M2_SMALL_TRAIN"

PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"


@dataclass(frozen=True)
class SmallConfig:
    """Explicit small Transformer widths. Not derived from plan.B_TEMPORAL_WIDTH."""

    conv_channels: int = 16
    conv_kernel: int = 5
    token_in: int = 70
    identity_dim: int = 50
    t4_dim: int = 4
    set_dim: int = 256
    slots: int = 8
    heads: int = 8
    layers: int = 4
    temporal_width: int = 256
    ffn: int = 512
    slot_proj_in: int = 2048
    slot_proj_out: int = 256
    readout: tuple[int, int, int] = (256, 128, 2)
    window: int = 50
    unit_dropout: float = 0.10
    out_dim: int = 2
    channels: int = 96


SMALL = SmallConfig()


class SmallStabilityError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmallStabilityError(message)


def cell_lr_bounds(cell: str) -> tuple[float, float]:
    if cell in (CELL_N0, CELL_N1):
        return LR_MAX_1E4, LR_MIN_1E4
    require(cell in (CELL_S0, CELL_S1), f"unknown cell {cell}")
    return LR_MAX, LR_MIN


def run_root_for_cell(cell: str) -> Path:
    if cell in (CELL_N0, CELL_N1):
        return LR_CONTRAST_RUN_ROOT
    require(cell in (CELL_S0, CELL_S1), f"unknown cell {cell}")
    return INTERIM_RUN_ROOT
