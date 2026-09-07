"""M1 small-B temporal widths. W=100, 16-d EMG, B3 identity 100-d, rSyn3 4-col."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tfpd_exploration.src.cross_dataset_functional_calibration_v1.plan import (
    S_FIX_EPOCH011_RELATIVE,
    S_FIX_EPOCH011_SHA256,
)
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import plan as fold_plan


@dataclass(frozen=True)
class M1TemporalConfig:
    conv_channels: int = 16
    conv_kernel: int = 5
    e0_dim: int = 100  # frozen B3 id_encoder post_pool, not M2's 50-d token
    hc_dim: int = 4
    set_dim: int = 256
    slots: int = 8
    heads: int = 8
    layers: int = 4
    temporal_width: int = 256
    ffn: int = 512
    readout_hidden: int = 128
    out_dim: int = 16
    window: int = 100
    n_units: int = 64
    unit_dropout: float = 0.10
    pe_max_len: int = 100
    route_key_dim: int = 32
    prediction_divisor: float = 1.0

    @property
    def token_in(self) -> int:
        return self.conv_channels + self.e0_dim + self.hc_dim

    @property
    def local_dim(self) -> int:
        return self.conv_channels


M1_TEMPORAL = M1TemporalConfig()

FRONTEND_RNG_DOMAIN = 0
TEMPORAL_RNG_DOMAIN = 1_000_003
ROUTING_RNG_DOMAIN = 7_700_042

REPO_ROOT = Path(__file__).resolve().parents[4]
S_FIX_PATH = REPO_ROOT / S_FIX_EPOCH011_RELATIVE
S_FIX_SHA256 = S_FIX_EPOCH011_SHA256
FOLD0_SOURCES = fold_plan.FOLD0_SOURCE_SESSIONS
FOLD0_TARGET = fold_plan.FOLD0_TARGET_SESSION
SUPPORT_TRIALS = fold_plan.SUPPORT_TRIALS
QUERY_START = fold_plan.QUERY_START
QUERY_STOP_EXCLUSIVE = fold_plan.QUERY_STOP_EXCLUSIVE
