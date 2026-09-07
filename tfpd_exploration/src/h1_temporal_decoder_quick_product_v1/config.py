"""H1 temporal product constants. New root; do not overwrite old H1 results."""

from __future__ import annotations

from pathlib import Path

PRODUCT_SCHEMA = "h1_temporal_decoder_quick_product_v1"
CONTRACT_VERSION = 1
SEED = 42
STRIDE = 4
EFFECTIVE_BATCH = 32
LR = 1.0e-4
WEIGHT_DECAY = 1.0e-2
GRAD_CLIP = 1.0
EMA_DECAY = 0.9995
EPOCHS_TARGET = 12
WARMUP_EPOCHS = 1
PREDICTION_DIVISOR = 20.0
WINDOW = 700
N_UNITS = 176
OUT_DIM = 7

GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = REPO_ROOT / "tfpd_exploration/results/h1_temporal_decoder_quick_product_v1"
SLOT_ROOT = REPO_ROOT / "tfpd_exploration/results/six_evalai_slots_v1/20260905_155800"
C2_PAYLOAD = REPO_ROOT / "tfpd_exploration/submissions/evalai_h1_c2_ho_epoch15_v1/artifacts/decoder.pt"
H1_DATA_DIR = REPO_ROOT / "SPINT-main/data/000954"

HELDIN_SESSIONS: tuple[str, ...] = (
    "ses-19250101T111740",
    "ses-19250101T112404",
    "ses-19250108T110520",
    "ses-19250108T111022",
    "ses-19250108T111455",
    "ses-19250113T120811",
    "ses-19250113T121303",
    "ses-19250115T110633",
    "ses-19250115T111328",
    "ses-19250119T113543",
    "ses-19250119T114045",
    "ses-19250120T115044",
    "ses-19250120T115537",
)
