"""Frozen constants for m2_dual_track_v1.

Coordinator-owned. Workers must import these values; do not duplicate or
override them in owner files. Bump CONTRACT_VERSION in contracts.py if the
shared API changes.
"""

from __future__ import annotations

from pathlib import Path

SCHEMA = "m2_dual_track_v1"
CONTRACT_VERSION = 1
WORKORDER_RELATIVE = (
    "tfpd_exploration/docs/"
    "WORKORDER_CALIBRATION_MEMORY_AND_TEMPORAL_DECODER_PARALLEL_V1_20260905.md"
)
DESIGN_RELATIVE = (
    "tfpd_exploration/docs/DESIGN_DUAL_CALIBRATED_TEMPORAL_DECODERS_V1_20260905.md"
)
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_dual_track_v1"
ACTIVE_RUN_RELATIVE = f"{RESULT_ROOT_RELATIVE}/20260905_101500"

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Authority (byte SHA unless marked tensor-state digest)
# ---------------------------------------------------------------------------
CHAMPION_CKPT_RELATIVE = (
    "streaming_calibration_exp/outputs/streaming_calibration/"
    "m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806/checkpoints/best.ckpt"
)
CHAMPION_CKPT_SHA256 = "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
CHAMPION_NORMALIZATION_SHA256 = "d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e"
CHAMPION_RUN_RELATIVE = (
    "streaming_calibration_exp/outputs/streaming_calibration/"
    "m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806"
)
FILM_STATES_RELATIVE = "tfpd_exploration/results/m2_hold_film_probe_v1/film_states.pt"
SELECTED_HEAD_RELATIVE = (
    "tfpd_exploration/results/m2_movement_t4_empty_epoch_pick_v1/selected_head.pt"
)
SELECTED_HEAD_STATE_SHA256 = "13551d3fc33d1cc296670c577c11519c04abdb42fa081f7d061191bb5610e6c5"
M33_SPLIT_MANIFEST_RELATIVE = (
    "streaming_calibration_exp/outputs/streaming_calibration/"
    "m2_m33_disjoint_replay_correction_v1_t4_m2_final1_f1_s42_20260801_120203/"
    "split_manifest.json"
)
OFFICIAL_HISTORICAL_HO_R2 = 0.3494526364
HISTORICAL_SIX_SESSION_LOCAL_R2 = 0.3604492265  # NOT a governing baseline
ADAPTER_SEED = 44
ADAPTER_EPOCH_ONE_BASED = 8

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
CHANNELS = 96
WINDOW = 50
CALIB_TRIAL_LENGTH = 100
SUPPORT_HORIZON = 33
OUT_DIM = 2
HIDDEN_DIM = 64
T4_DIM = 4
IDENTITY_DIM = 50
BEHAVIOR_SCALE = 5.0
MOVE_T4_START_BIN = 5
MOVE_T4_STOP_BIN = 30
RIDGE_NORMALIZED_LAMBDA = 0.1

# Training target space is decoder_raw (student last-bin output).
# Evaluator divides by BEHAVIOR_SCALE before variance-weighted R2 vs native covariates.
TRAINING_TARGET_SPACE = "decoder_raw"
SCORING_TARGET_SPACE = "native_covariate"

# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
HELDIN_SESSIONS = (
    "ses-2020-10-19-Run1",
    "ses-2020-10-19-Run2",
    "ses-2020-10-20-Run1",
    "ses-2020-10-20-Run2",
    "ses-2020-10-27-Run1",
    "ses-2020-10-27-Run2",
    "ses-2020-10-28-Run1",
)
EXTERNAL_SESSIONS = (
    "ses-2020-10-30-Run1",
    "ses-2020-10-30-Run2",
    "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1",
    "ses-2020-11-24-Run1",
    "ses-2020-11-24-Run2",
)
EXT4_SESSIONS = (
    "ses-2020-10-30-Run1",
    "ses-2020-10-30-Run2",
    "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1",
)
EXT4_DATES = (
    "2020-10-30",
    "2020-11-18",
    "2020-11-19",
)
EXT4_EXPECTED_WINDOWS = {
    "ses-2020-10-30-Run1": 519,
    "ses-2020-10-30-Run2": 490,
    "ses-2020-11-18-Run1": 425,
    "ses-2020-11-19-Run1": 635,
}
EXT4_EXPECTED_QUERY_TRIALS = {
    "ses-2020-10-30-Run1": 10,
    "ses-2020-10-30-Run2": 8,
    "ses-2020-11-18-Run1": 8,
    "ses-2020-11-19-Run1": 10,
}
EXCLUDED_EXTERNAL_SESSIONS = (
    "ses-2020-11-24-Run1",
    "ses-2020-11-24-Run2",
)

# ---------------------------------------------------------------------------
# A-QMEM first-round
# ---------------------------------------------------------------------------
A_HEADS = 2
A_HEAD_DIM = 16
A_VALUE_DIM = 16
A_MLP_HIDDEN = 64
A_LR = 1.0e-4
A_WEIGHT_DECAY = 1.0e-2
A_GRAD_CLIP = 1.0
A_WARMUP_FRAC = 0.05
A_PRECISION = "fp32"

# ---------------------------------------------------------------------------
# B first-round
# ---------------------------------------------------------------------------
B_CONV_CHANNELS = 16
B_CONV_KERNEL = 5
B_SET_DIM = 256
B_SLOTS = 8
B_SLOT_HEADS = 8
B_TEMPORAL_WIDTH = 512
B_MAMBA_D_STATE = 64
B_MAMBA_EXPAND = 2
B_MAMBA_HEADDIM = 64
B_MAMBA_NGROUPS = 1
B_MAMBA_D_CONV = 4
B_MAMBA_CHUNK = 64
B_MAMBA_LAYERS = 4
B_TRANSFORMER_HEADS = 8
B_TRANSFORMER_FFN = 1024
B_TRANSFORMER_LAYERS = 4
B_READOUT_HIDDEN = 128
B_LR = 3.0e-4
B_WEIGHT_DECAY = 1.0e-2
B_GRAD_CLIP = 1.0
B_UNIT_DROPOUT = 0.10
B_PRECISION = "bf16"
B_WARMUP_EPOCHS = 1

# ---------------------------------------------------------------------------
# Shared training
# ---------------------------------------------------------------------------
SEED_PRIMARY = 42
SEED_CONFIRM = 43
EPOCHS = 12
EPOCHS_EXTENDED = 24
EFFECTIVE_BATCH = 32
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1.0e-8
TIE_EPS = 1.0e-10
JOB_GPU_HOUR_LIMIT = 6.0
TOTAL_DISK_BUDGET_GIB = 40.0
MEM_WORKING_SET_GIB = 44.0
MEM_AVAILABLE_FLOOR_GIB = 12.0
MEM_PAUSE_GIB = 10.0
MEM_STOP_GIB = 8.0
GPU_PEAK_GIB = 20.0
LOADER_WORKERS = 0
TORCH_THREADS = 4
COMPILE_MAX_JOBS = 2

# GPU lease
GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
GPU1_UUID = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"
GPU0_DEFAULT_OWNER = "A"
GPU1_DEFAULT_OWNER = "B"

# Numeric policy
A_ZERO_EXACT = True
PERM_ATOL = 1.0e-5
PERM_RTOL = 1.0e-4
B_RECURRENCE_LENGTHS = (1, 4, 49, 50, 63, 64, 65, 129)

# Historical numbers that must NEVER be subtracted from new scores.
FORBIDDEN_BASELINE_SUBTRACTIONS = (
    HISTORICAL_SIX_SESSION_LOCAL_R2,
    OFFICIAL_HISTORICAL_HO_R2,
)

PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
REQUIRED_ENV = {
    "PYTHONNOUSERSITE": "1",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
}


def repo_root() -> Path:
    return REPO_ROOT


def result_root() -> Path:
    return REPO_ROOT / RESULT_ROOT_RELATIVE


def active_run_root() -> Path:
    return REPO_ROOT / ACTIVE_RUN_RELATIVE


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DualTrackError(message)


class DualTrackError(RuntimeError):
    pass
