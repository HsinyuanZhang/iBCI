"""Frozen S1/S2 pick constants and official M2 tag roster."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path("/home/xinyuan/Work_host/SPINT")
SLOT_ROOT = REPO_ROOT / "tfpd_exploration/results/six_evalai_slots_v1/20260905_155800"
PICKS_PATH = SLOT_ROOT / "m2_product_picks.json"
BANK_CACHE = SLOT_ROOT / "m2_runtime_banks"
DUAL_CACHE = REPO_ROOT / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache"
M2_DATA_DIR = REPO_ROOT / "SPINT-main/data/000953"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"

S1_CKPT = (
    REPO_ROOT
    / "tfpd_exploration/results/m2_b_small_stability_v1/20260905_123000"
    / "S1_SMALL_COS/seed42/epoch_019.pt"
)
S2_CKPT = (
    REPO_ROOT
    / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500"
    / "arms/B-TRANSFORMER/seed42_shuffled_e13_24/epoch_020.pt"
)
S1_CKPT_SCHEMA = "m2_b_small_stability_v1_ckpt"
S2_CKPT_SCHEMA = "m2_dual_track_v1_full_ckpt_v1"

WINDOW = 50
CHANNELS = 96
BEHAVIOR_SCALE = 5.0
SUPPORT_HORIZON = 33
IDENTITY_DIM = 50
T4_DIM = 4
OUT_DIM = 2
OFFICIAL_BATCH = 7
EXPECTED_SESSION_COUNT = 13

S1_EXPECTED_MEAN = 0.4515131891854879
S1_EXPECTED_PER_SESSION = {
    "ses-2020-10-30-Run1": 0.5790014751890533,
    "ses-2020-10-30-Run2": 0.41235621832480873,
    "ses-2020-11-18-Run1": 0.45454332465708325,
    "ses-2020-11-19-Run1": 0.36015173857100613,
}
S2_EXPECTED_MEAN = 0.3719539787006045
S2_EXPECTED_PER_SESSION = {
    "ses-2020-10-30-Run1": 0.5010050821030114,
    "ses-2020-10-30-Run2": 0.3810312032351536,
    "ses-2020-11-18-Run1": 0.35919232451499583,
    "ses-2020-11-19-Run1": 0.24658730494925707,
}
S2_WORST_SESSION = "ses-2020-11-19-Run1"
S2_WORST_R2 = 0.24658730494925707

HELDIN_SESSIONS = (
    "ses-2020-10-19-Run1",
    "ses-2020-10-19-Run2",
    "ses-2020-10-20-Run1",
    "ses-2020-10-20-Run2",
    "ses-2020-10-27-Run1",
    "ses-2020-10-27-Run2",
    "ses-2020-10-28-Run1",
)
EXT4_SESSIONS = (
    "ses-2020-10-30-Run1",
    "ses-2020-10-30-Run2",
    "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1",
)
NOV24_SESSIONS = (
    "ses-2020-11-24-Run1",
    "ses-2020-11-24-Run2",
)
ALL_OFFICIAL_SESSIONS = HELDIN_SESSIONS + EXT4_SESSIONS + NOV24_SESSIONS

OFFICIAL_HELDIN_TAGS = (
    "Run1_20201019",
    "Run2_20201019",
    "Run1_20201020",
    "Run2_20201020",
    "Run1_20201027",
    "Run2_20201027",
    "Run1_20201028",
)
OFFICIAL_HELDOUT_TAGS = (
    "Run1_20201030",
    "Run2_20201030",
    "Run1_20201118",
    "Run1_20201119",
    "Run1_20201124",
    "Run2_20201124",
)
OFFICIAL_TAGS = OFFICIAL_HELDIN_TAGS + OFFICIAL_HELDOUT_TAGS

KNOWN_PAYLOAD_SHA256 = {
    "91ef13cc94b9ab865c8f926dcbb6d33e9628bf9cbc9b757665d10e33cfda144a",
    "f2f8cd4c046a5880e9716d61981cee4aa0652d33e411212fa7be217cef05b051",
    "523d3d2e55a8fd4620a3f0a94dea6a99ae6ff53cefe807c3ff809a30b1d2479d",
    "f3e64950b00193949f6993d1a9022ba98fb65b759198427d666483d0c6bc49c1",
    "0cc4ab769c8752191c87b6d8947b32768e15c51a49a097c3b6335b416fa81ca5",
    "33181fa3c1543b9af777a17571d9c98d427b7e65d49375716d624e80c99603f3",
    "b3d19967b258fe94584ad4f84a7d394f8c2c83f85bdc3e198b59fc8178b05af5",
    "d60f38d5cefe45ea8fdccf9853d5034a15d7896b81b0fff3ebe983b7bd7b78a8",
    "e4ff17e857c0bab9bbd900bc737ca7c48476a44ed03725cefc5377d92b959261",
    "4f68b7b8d0a64c66b53810dd73a125b9021b0b029dc57f00bb888b783151ecf4",
    "4e4dae8f7239582a26d44cdb449e674710f28223523dd691dd4f8758b05220e0",
}

CHAMPION_CKPT_SHA256 = "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
BASE_IMAGE = "spint-m2:e8-epoch027-76f0fb2"
BASE_IMAGE_ID = "sha256:b179efcba8e36202aec688b3104397919576e30d4344c309febcf692d4d7aaf8"

S1_METHOD_LABEL = (
    "M2 small causal Transformer (S1-SMALL-COS seed42 EMA e19); "
    "visible-product pick on local ext-4; frozen M33 MOVE-T4/E0 banks; no TTA"
)
S1_METHOD_NAME = "M2 small Transformer S1-SMALL-COS EMA e19 visible-product"
S1_METHOD_DESCRIPTION = (
    "New M2 Falcon runtime that executes a 3.54M-parameter causal Transformer "
    "(shared set frontend, 4-layer temporal stack, 256-128-2 readout), not the "
    "old static SPINT decoder. Weights are the seed-42 S1-SMALL-COS EMA view at "
    "epoch 19, selected by a sealed visible-product rule on local M33-disjoint "
    "ext-4 (equal-session mean 0.4515). Per-session banks reuse the training-time "
    "native E0 and MOVE-T4 (bins 5:30, source-only normalizer, selected EMPTY head) "
    "for all 13 official held-in and held-out dataset tags. Online path is a 50-bin "
    "causal window with output /5, unit mask, full-window forward (no cross-window "
    "KV reuse), and no test-time adaptation."
)
S2_METHOD_LABEL = (
    "M2 large causal Transformer (B-TRANSFORMER seed42 RAW e20); "
    "architecture probe; visible-product pick on local ext-4; worst-session "
    "0.2466 on 11-19; frozen M33 MOVE-T4/E0 banks; no TTA"
)
S2_METHOD_NAME = "M2 large Transformer B-TRANSFORMER RAW e20 architecture probe"
S2_METHOD_DESCRIPTION = (
    "Architecture probe, not a capacity-controlled ablation versus the small "
    "Transformer. New M2 Falcon runtime that executes the large B-TRANSFORMER "
    "(shared set frontend, 4-layer 512-wide temporal stack, 512-128-2 readout), "
    "not the old static SPINT decoder. Weights are seed-42 RAW epoch 20 from the "
    "shuffled 1-24 B-TRANSFORMER run, selected by the sealed visible-product rule "
    "on local M33-disjoint ext-4 (equal-session mean 0.372). Worst session is "
    "0.2466 on ses-2020-11-19-Run1 and is disclosed as a local development "
    "limitation. Per-session banks reuse training-time native E0 and MOVE-T4 for "
    "all 13 official tags. Online path is a 50-bin causal window with output /5, "
    "unit mask, full-window forward (no cross-window KV reuse), and no TTA."
)
