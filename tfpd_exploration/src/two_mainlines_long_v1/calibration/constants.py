"""Owned literals for the repaired P pair. Science matches the sealed plan."""
from __future__ import annotations

from pathlib import Path

from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import plan as sealed_plan
from tfpd_exploration.src.two_mainlines_long_v1.gpu_lease import GPU1

REPO_ROOT = Path("/home/xinyuan/Work_host/SPINT")
OWNED_RESULT_ROOT = REPO_ROOT / "tfpd_exploration/results/two_mainlines_long_v1/20260905_155800"
SLOT_ROOT = REPO_ROOT / "tfpd_exploration/results/six_evalai_slots_v1/20260905_155800"
OLD_STAGE0_ROOT = REPO_ROOT / sealed_plan.STAGE0_ROOT_RELATIVE
OLD_PREFLIGHT_ROOT = REPO_ROOT / sealed_plan.P_PREFLIGHT_ROOT_RELATIVE

GPU_INDEX = 1
GPU_UUID = GPU1
CUDA_VISIBLE_DEVICES = "1"

S_FIX_SHA256 = sealed_plan.S_FIX_EPOCH011_SHA256
S_FIX_RELATIVE = sealed_plan.S_FIX_EPOCH011_RELATIVE
F_ETA_NAME = sealed_plan.P_F_ETA_NAME
F_ETA_SHAPE = sealed_plan.P_F_ETA_SHAPE
SOURCES = sealed_plan.M1_FOLD0_SOURCES
OUTER = sealed_plan.M1_FOLD0_TARGET
SUPPORT_TRIALS = sealed_plan.M1_SUPPORT_TRIALS
QUERY_START = sealed_plan.M1_QUERY_START
QUERY_STOP_EXCLUSIVE = sealed_plan.M1_QUERY_STOP_EXCLUSIVE
RIDGE_LAMBDA = sealed_plan.M1_RIDGE_LAMBDA
LR = sealed_plan.P_LR
CLIP = sealed_plan.P_CLIP
WEIGHT_DECAY = sealed_plan.P_WEIGHT_DECAY
WARMUP_EPOCHS = sealed_plan.P_WARMUP_EPOCHS
EPOCHS = sealed_plan.P_EPOCHS
BATCH = sealed_plan.P_BATCH
FORMAL_SEED = sealed_plan.P_SEED
PROFILE_SEED = sealed_plan.P_DISPOSABLE_PROFILE_SEED
CARRIER_ATOL = sealed_plan.P_CARRIER_ATOL
CARRIER_RTOL = sealed_plan.P_CARRIER_RTOL
CONSUMER_ATOL = sealed_plan.P_CONSUMER_ATOL
CONSUMER_RTOL = sealed_plan.P_CONSUMER_RTOL

PROFILE_WARMUP_STEPS = 20
PROFILE_PAIRED_UPDATES = 100

SEALED_D0_DIGEST = "984f234fa54d3daeb933c9e80c6fffe8df2470525557fb1f980bddc32632fcd5"
SEALED_TARGET_M10_DIGEST = "2d1a638e4816aa3254f5b1817601ddc4515cd6a24f4d5aa03e61ee7a8513ffe7"
SEALED_SCALE_DIGEST = "164d6e8c93e5d294af007b7fea9e7386a8bec9a59b5d3cf9aa7877c8adddca21"
SEALED_NMF_RECON_DIGEST = "83da4033d5c33117ed29e9595ce278b47026a00305a10de4cf611a6aaa7b2143"

FORBIDDEN_OLD_FACTORY = (
    "tfpd_exploration.src.cross_dataset_functional_calibration_v1.model_adapter.build_p_pair"
)

T0_LOCAL = "2026-09-05T15:57:50+08:00"
SNAPSHOT_PRIMARY_LOCAL = "2026-09-05T20:27:50+08:00"
SNAPSHOT_FALLBACK_LOCAL = "2026-09-05T21:27:50+08:00"
MIN_SNAPSHOT_EPOCHS = 4
