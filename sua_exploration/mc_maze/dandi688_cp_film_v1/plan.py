"""Frozen constants for the DANDI 000688 CP-FiLM source screen."""
from __future__ import annotations

from pathlib import Path

SCHEMA = "dandi688_cp_film_v1"
DESIGN_RELATIVE = "sua_exploration/docs/DESIGN_DANDI_000688_CALIBRATION_PROFILE_FILM_V1_20260904.md"
DESIGN_SHA256 = "5aa3d8a6abb7a627bb710e49154673caa2e0e829d66f6c867cdb595fae63ba42"
WORKORDER_RELATIVE = "sua_exploration/docs/WORKORDER_DANDI_000688_CALIBRATION_PROFILE_FILM_V1_20260904.md"
WORKORDER_SHA256 = "2002c974793e836af1e34bd610ad715015bf51842817f2e1faa6dcb1ff23df93"
MANIFEST_RELATIVE = "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
DATA_RELATIVE = "sua_exploration/data/dandi_000688/sub-C"
CACHE_RELATIVE = "sua_exploration/cache/dandi688_subc_co_v1"
TEACHER_RELATIVE = "sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
TEACHER_SHA256 = "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d"
ANCHOR_RELATIVE = {
    42: "sua_exploration/checkpoints/sua_spint_t4_mainline_fp32_v1_t4_dandi688_co_s42/epoch_ckpts/epoch_011.ckpt",
    43: "sua_exploration/checkpoints/sua_spint_t4_mainline_fp32_v1_t4_dandi688_co_s43/epoch_ckpts/epoch_011.ckpt",
    44: "sua_exploration/checkpoints/sua_spint_t4_mainline_fp32_v1_t4_dandi688_co_s44/epoch_ckpts/epoch_011.ckpt",
}
ANCHOR_SHA256 = {
    42: "2a50fb2072f0fc76057c22639615a37a21cfa9b83c282cd7fb5e5482b40a199d",
    43: "2126ee843d1b90ce2273f52be81f414f1cd9bc9241f965203445a86981fa7ec5",
    44: "654b5423e3f4450d9cf6f4135ddbe1d425e3a2f5e114948d4fb2792f029aa5a7",
}
RESULT_PARENT_RELATIVE = "sua_exploration/results/dandi688_cp_film_v1"

TRAIN_SESSIONS = 27
VALIDATION_SESSIONS = 6
FORMAL_TEST_SESSIONS = 6
ACTIVITY_SUPPORT = 30
T4_SUPPORT = 30
QUERY_START = 50
PROFILE_HORIZONS = (10, 30)
PROFILE_DIM = 4
PROFILE_MASK = (1.0, 1.0, 0.0, 0.0)
T4_DIM = 4
CONTEXT_DIM = 8
FILM_RANK = 8
HIDDEN_DIM = 64
FILM_PARAMETERS = 1224
WINDOW = 50
TRIAL_LENGTH = 100
BIN_MS = 20
BEHAVIOR_SCALE = 5.0

ARMS = ("CP10", "CP30", "SHUFFLE10", "EMPTY")
SEEDS = (42, 43, 44)
EPOCHS = 12
BATCH_SIZE = 32
LEARNING_RATE = 3.0e-4
WINDOWS_PER_SESSION_PER_EPOCH = 1024
PROFILE_SHUFFLE_DOMAIN = "DANDI688_CP_FILM_V1/PROFILE_ROW_SHUFFLE"
TRAIN_SAMPLE_DOMAIN = "DANDI688_CP_FILM_V1/Q50_TRAIN_WINDOWS"

NONINFERIOR_MEAN = -0.005
NONINFERIOR_WORST = -0.030
POSITIVE_SESSIONS = 4
SOLID_MEAN = 0.010
EXPECTED_GPU_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"


def path(repo_root: Path, relative: str) -> Path:
    return Path(repo_root) / relative


def result_root(repo_root: Path, seed: int) -> Path:
    return Path(repo_root) / RESULT_PARENT_RELATIVE / f"seed{seed}"
