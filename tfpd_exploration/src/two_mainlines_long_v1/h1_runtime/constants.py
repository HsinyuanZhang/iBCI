"""H1 temporal Transformer EvalAI constants. Snapshot E_H=5, EMA governing."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path("/home/xinyuan/Work_host/SPINT")
SLOT_ROOT = REPO_ROOT / "tfpd_exploration/results/six_evalai_slots_v1/20260905_155800"
RESULT_ROOT = REPO_ROOT / "tfpd_exploration/results/h1_temporal_decoder_quick_product_v1"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
C2_PAYLOAD = REPO_ROOT / "tfpd_exploration/submissions/evalai_h1_c2_ho_epoch15_v1/artifacts/decoder.pt"
C2_CKPT_SHA256 = "ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215"
C2_PAYLOAD_SHA256 = "91ef13cc94b9ab865c8f926dcbb6d33e9628bf9cbc9b757665d10e33cfda144a"
CKPT_SCHEMA = "h1_temporal_decoder_quick_product_v1_ckpt"
EPOCH = 5
EPOCHS_TARGET = 12
SEED = 42
VIEW = "EMA"

WINDOW = 700
CHANNELS = 176
OUT_DIM = 7
BEHAVIOR_SCALE = 20.0
OFFICIAL_BATCH = 8
EXPECTED_SESSION_COUNT = 27
PE_MAX_LEN = 700

FLAT_CKPT = RESULT_ROOT / "flat" / "epoch_005.pt"
ROUTE_CKPT = RESULT_ROOT / "route" / "epoch_005.pt"
FLAT_SUB = REPO_ROOT / "tfpd_exploration/submissions/evalai_h1_temporal_flat_e5_v1"
ROUTE_SUB = REPO_ROOT / "tfpd_exploration/submissions/evalai_h1_temporal_route_e5_v1"

BASE_IMAGE = "h1-epfilm-c1:no-readout-v1-523d3d2e"
BASE_IMAGE_ID = "sha256:0408012c36a0a15a6a1fc21de4760847d7f64e2ac64139002cf7adc30b06f6b9"

OFFICIAL_TAGS: tuple[str, ...] = (
    "S0_set_1",
    "S0_set_2",
    "S1_set_1",
    "S1_set_2",
    "S1_set_3",
    "S2_set_1",
    "S2_set_2",
    "S3_set_1",
    "S3_set_2",
    "S4_set_1",
    "S4_set_2",
    "S5_set_1",
    "S5_set_2",
    "S6_set_1",
    "S6_set_2",
    "S7_set_1",
    "S7_set_2",
    "S8_set_1",
    "S8_set_2",
    "S9_set_1",
    "S9_set_2",
    "S10_set_1",
    "S10_set_2",
    "S11_set_1",
    "S11_set_2",
    "S12_set_1",
    "S12_set_2",
)

KNOWN_PAYLOAD_SHA256 = {
    C2_PAYLOAD_SHA256,
    "523d3d2e55a8fd4620a3f0a94dea6a99ae6ff53cefe807c3ff809a30b1d2479d",
    "df71cb9329a87b5073242044b2933391ced7bf866d1f481e994f71dbd97ba2d7",
    "37071f3e718913f994b5015e13aa203026fb630339e87921ddb1c35de9d2bf62",
}

FLAT_METHOD_LABEL = (
    "H1 new temporal Transformer FLAT; snapshot epoch 5 of intended 12; "
    "EMA; known-source development; not C2 identity swap"
)
ROUTE_METHOD_LABEL = (
    "H1 new temporal Transformer ROUTE; snapshot epoch 5 of intended 12; "
    "EMA; known-source development; not C2 identity swap"
)
FLAT_METHOD_NAME = "H1 temporal Transformer FLAT EMA e5 of 12 known-source"
ROUTE_METHOD_NAME = "H1 temporal Transformer ROUTE EMA e5 of 12 known-source"
FLAT_METHOD_DESCRIPTION = (
    "New temporal Transformer Falcon runtime for H1, not a C2 identity swap "
    "and not the old C2/581920/581900 decoder. Architecture: shared causal "
    "Conv1→16 k5, concat local+E0+H-C, 8-slot set attention d=256, 4-layer "
    "causal Transformer d=256/FFN=512, 256→128→7 last-bin readout, output /20. "
    "Weights are the EMA view of snapshot epoch 5 of the intended 12-epoch "
    "FLAT run (seed 42). This is known-source development, not clean LODO. "
    "E0 is 700-d C2 fused identity plus raw 4-d H-C from the frozen C2 "
    "epoch-15 calibration encoder (SHA "
    "ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215) used "
    "for materialization only. PE covers 700. on_done is a no-op (continual). "
    "No unproven cross-window KV cache. All 27 official H1 dataset tags."
)
ROUTE_METHOD_DESCRIPTION = (
    "New temporal Transformer Falcon runtime for H1 with static calibration "
    "routing, not a C2 identity swap and not the old C2/581920/581900 decoder. "
    "Same stack as FLAT (Conv1→16 k5, concat local+E0+H-C, 8-slot set attn "
    "d=256, 4-layer causal Transformer, 256→128→7 last-bin, output /20) plus "
    "per-head routing bonuses on slot-to-unit logits. Weights are the EMA view "
    "of snapshot epoch 5 of the intended 12-epoch ROUTE run (seed 42). This is "
    "known-source development, not clean LODO. C2 epoch-15 SHA "
    "ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215 is "
    "calibration materialization only. PE covers 700. on_done is a no-op. "
    "No unproven cross-window KV cache. All 27 official H1 dataset tags."
)
