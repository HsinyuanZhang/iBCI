"""Config, SHA, and budget contracts for m2_b_small_stability_v1."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from tfpd_exploration.src.m2_dual_track_v1 import plan
from tfpd_exploration.src.m2_b_small_stability_v1 import config as cfg
from tfpd_exploration.src.m2_b_small_stability_v1.decoder import (
    SmallTransformerDecoder,
    count_decoder_parameters,
)


REPO = Path(__file__).resolve().parents[2]
WORKORDER = (
    REPO
    / "tfpd_exploration/docs/WORKORDER_M2_B_SMALL_TRANSFORMER_STABILIZATION_INTERIM_V1_20260905.md"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_workorder_sha_matches_mandate() -> None:
    assert _sha256(WORKORDER) == cfg.WORKORDER_SHA256
    assert cfg.WORKORDER_SHA256 == "bb0df697f353e109b0d470e1f3030146b79a74f05683e4ff4ee4b0a012cd578b"


def test_small_config_is_immutable_and_not_512_wide() -> None:
    small = cfg.SMALL
    assert small.temporal_width == 256
    assert small.ffn == 512
    assert small.slot_proj_in == 2048
    assert small.slot_proj_out == 256
    assert small.set_dim == 256
    assert small.slots == 8
    assert small.heads == 8
    assert small.layers == 4
    assert small.token_in == 70
    assert small.readout == (256, 128, 2)
    assert small.conv_channels == 16
    assert small.conv_kernel == 5
    assert small.unit_dropout == 0.10
    assert small.window == 50
    try:
        small.temporal_width = 512  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("SmallConfig must be immutable")
    assert small.temporal_width == 256
    assert plan.B_TEMPORAL_WIDTH == 512
    assert plan.B_TRANSFORMER_FFN == 1024


def test_does_not_mutate_old_plan_globals() -> None:
    assert plan.B_TEMPORAL_WIDTH == 512
    assert plan.B_TRANSFORMER_FFN == 1024
    assert "B_TEMPORAL_WIDTH=256" not in Path(
        REPO / "tfpd_exploration/src/m2_b_small_stability_v1/config.py"
    ).read_text(encoding="utf-8")


def test_decoder_param_budget_is_exact() -> None:
    model = SmallTransformerDecoder(seed=42)
    counts = count_decoder_parameters(model)
    assert counts["decoder_total"] == 3_543_010
    assert counts["temporal"] == 2_108_416
    assert counts["frontend_plus_readout"] == 1_434_594
    assert counts["decoder_plus_shared_calib"] == 3_562_524
    assert counts["shared_calib"] == 19_514
    named = dict(model.named_parameters())
    assert sum(p.numel() for p in named.values()) == 3_543_010
    assert model.frontend.slot_proj.out_features == 256
    assert model.frontend.slot_proj.in_features == 2048
    assert model.readout[0].in_features == 256
    assert model.temporal.blocks[0].attn.qkv.in_features == 256


def test_manifest_update_counts_and_digests() -> None:
    from tfpd_exploration.src.m2_dual_track_v1.sampler import load_manifest

    path = cfg.OLD_MANIFEST_24_PATH
    assert path.is_file()
    manifest = load_manifest(path)
    assert manifest["digest"] == cfg.MANIFEST_24_DIGEST
    assert manifest["parent_12_digest"] == cfg.MANIFEST_12_DIGEST
    n_ep = [len(manifest["batches"][str(epoch)]) for epoch in range(1, 25)]
    assert n_ep[0] == 3165
    assert all(n == 3165 for n in n_ep)
    assert sum(n_ep) == 75960
    assert cfg.UPDATES_PER_EPOCH == 3165
    assert cfg.TOTAL_UPDATES == 75960


def test_cells_and_primary_candidate_are_declared() -> None:
    assert cfg.CELL_S0 == "S0-SMALL-LEGACY"
    assert cfg.CELL_S1 == "S1-SMALL-COS"
    assert cfg.PRIMARY_CANDIDATE == "S1-SMALL-COS/EMA"
    assert cfg.EMA_DECAY == 0.9995
    assert cfg.LR_MAX == 3.0e-4
    assert cfg.LR_MIN == 3.0e-5
    assert cfg.SEED == 42
