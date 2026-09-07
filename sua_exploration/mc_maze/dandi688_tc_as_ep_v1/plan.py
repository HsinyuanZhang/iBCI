"""Frozen Stage-0 constants for TC-AS-EP V1.

This module deliberately has no NumPy, Torch, NWB, or Lightning import so the
route contract can be inspected before any data/model runtime is opened.
"""
from __future__ import annotations

from pathlib import Path


ROUTE_NAME = "dandi688_tc_as_ep_v1"
SCHEMA_VERSION = 1

CANDIDATE_POOL_N = 50
ACTIVITY_SUPPORT_N = 10
QUERY_START_TRIAL = 50
TRIAL_LENGTH_BINS = 100
WINDOW_SIZE_BINS = 50
BIN_SIZE_MS = 20
NUM_DIRECTIONS = 8
MAX_PER_DIRECTION = 2

ROBUST_MAD_FACTOR = 1.4826
ROBUST_SCALE_EPSILON = 1e-12
TIE_REL_TOL = 1e-12
TIE_ABS_TOL = 1e-12
POINT_BISERIAL_MAX_ABS = 0.20
JACKKNIFE_MEDIAN_JACCARD_MIN = 2.0 / 3.0

TRAINING_SEEDS = (42, 43, 44)
PRIMARY_SEED = 42
MAX_EPOCHS = 12
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

RANDOM_DOMAIN = "DANDI688_TC_AS_EP_V1/COV_RAND10/PER_SAMPLE"
DESIGN_RELATIVE = (
    "sua_exploration/docs/"
    "DESIGN_DANDI_000688_ACTIVITY_SELECTED_EARLY_POOLING_V1_20260904.md"
)
DESIGN_SHA256 = "3e04774363f8e35ff7ebb158d72a6492f04ce317fc7ed46903accba1db3d8752"
STAGE0_ATTEMPT2_WORKORDER_RELATIVE = (
    "sua_exploration/docs/"
    "WORKORDER_DANDI_000688_TC_AS_EP_STAGE0_ATTEMPT2_20260904.md"
)
STAGE0_ATTEMPT2_WORKORDER_SHA256 = (
    "9f53c7891e0ef52758c560fb16d2965b9216b852d569571ff0074208ac103229"
)
STRICT_MANIFEST_RELATIVE = (
    "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
)
STRICT_MANIFEST_SHA256 = (
    "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
)
TEACHER_RELATIVE = (
    "sua_exploration/checkpoints/teacher_mc_maze/"
    "best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
)
TEACHER_SHA256 = (
    "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d"
)

# The selected-architecture manifest points at three checkpoint copies that are
# absent in this workspace.  Stage-0 operator parity does not select a model or
# compare performance, so it uses this available, immutable historical M30
# B3S/T4 epoch-11 checkpoint and records its distinct T4@30 provenance.
PARITY_CHECKPOINT_RELATIVE = (
    "sua_exploration/checkpoints/"
    "sua_spint_t4_mainline_fp32_v1_t4_dandi688_co_s42/"
    "epoch_ckpts/epoch_011.ckpt"
)
PARITY_CHECKPOINT_SHA256 = (
    "2a50fb2072f0fc76057c22639615a37a21cfa9b83c282cd7fb5e5482b40a199d"
)

SOURCE_DATA_RELATIVE = "sua_exploration/data/dandi_000688/sub-C"
STAGE0_RESULT_RELATIVE = "sua_exploration/results/dandi688_tc_as_ep_v1/stage0_attempt2"

STAGE0_ATTEMPT1_RELATIVE = (
    "sua_exploration/results/dandi688_tc_as_ep_v1/stage0_attempt1"
)
STAGE0_ATTEMPT1_BODY_SHA256 = {
    "attempt.json": "b69383410265ceb6fc0cc23d4609ab9ae51d23908f12d731f8329f3d6058d7b8",
    "selector_authority.json": "e529fd7adf4d9cbab93f40eb0d7a384e1c30f0f8bf988c008c668887196f1fd3",
    "selector_decision.json": "1d0f2bac594452e7b637eeda851b57296faa9079a38845c4cba0c6f84d3c0f90",
}


def repo_path(repo_root: Path, relative: str) -> Path:
    return repo_root / relative
