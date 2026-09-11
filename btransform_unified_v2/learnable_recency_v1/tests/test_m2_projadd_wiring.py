from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from btransform_unified_v2.model import RiftDecoder
from learnable_recency_v1.config import DEFAULT_HALF_LIVES, add_learnable_flags, dataset_config, scaled_half_lives
from learnable_recency_v1.wrap import (
    LearnableRiftDecoder,
    assert_shared_byte_equal,
    install_temporal,
    trainable_new_parameter_count,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import m2_projadd_learnable_score as score_mod  # noqa: E402
import m2_projadd_learnable_train as train_mod  # noqa: E402


def test_default_config_is_m2_scaled_ladder_per_layer() -> None:
    cfg = dataset_config("m2")
    assert cfg.tier == "learned_slope"
    assert cfg.task == "m2" and cfg.context_bins == 50
    assert cfg.per_layer is True
    assert cfg.learn_flat_heads is False
    assert cfg.layers == 4
    assert cfg.ladder == "scaled"
    assert cfg.half_life_seconds == scaled_half_lives(50, 4)
    assert cfg.half_life_seconds != DEFAULT_HALF_LIVES
    assert tuple(cfg.temporal_config.windows) == (13, 12, 12, 12)


def test_scripts_import_and_flag_defaults() -> None:
    assert train_mod.SEED == 42 and train_mod.CONTEXT == 50 and train_mod.EPOCHS == 24 and train_mod.BATCH == 32
    assert train_mod.PROJ_DIM == 16
    assert train_mod.SCHEMA == "m2_rift_projadd_learnable_train_v1"
    assert train_mod.CHECKPOINT_SCHEMA == "m2_rift_projadd_learnable_epoch_checkpoint_v1"
    assert train_mod.SMOKE_RECEIPT_SCHEMA == "m2_rift_projadd_learnable_smoke_receipt_v1"
    assert train_mod.TRAIN_RECEIPT_SCHEMA == "m2_rift_projadd_learnable_train_receipt_v1"
    assert train_mod.REFERENCE.name == "m2_r50_recency_s42_formal_v1"
    assert score_mod.SELECTION_SCHEMA == "m2_rift_projadd_learnable_ext6_epoch_pick_v1_selection"
    assert score_mod.PROJ_DIM == 16
    assert score_mod.REFERENCE.name == "m2_r50_recency_s42_formal_v1"
    parser = argparse.ArgumentParser()
    add_learnable_flags(parser)
    args = parser.parse_args([])
    assert args.tier == "learned_slope"
    assert args.per_layer is True
    assert args.learn_flat_heads is False
    assert args.layers == 4
    assert args.ladder == "scaled"
    assert args.lr_multiplier == 1.0
    assert args.cable_nw is False
    assert args.half_lives is None


def test_decoder_factory_builds_learnable_projadd_decoder_on_cpu() -> None:
    model = train_mod.learnable_decoder(torch.device("cpu"), dataset_config("m2"))
    assert isinstance(model, LearnableRiftDecoder)
    assert isinstance(model, RiftDecoder)
    assert model.proj_dim == 16
    assert model.bias_mode == "learnable_learned_slope"
    assert tuple(model.temporal_config.windows) == (13, 12, 12, 12)
    assert tuple(model.temporal.slope_log.shape) == (4, 8)
    assert model.temporal._attention_backend == "local"
    assert trainable_new_parameter_count(model) == 24


def test_shared_parameters_byte_equal_with_same_seed_stock_rift_decoder() -> None:
    device = torch.device("cpu")
    cfg = dataset_config("m2")
    model = train_mod.learnable_decoder(device, cfg)
    stock = RiftDecoder("m2", context_bins=50, bias_mode="recency", seed=42, proj_dim=16)
    shared = assert_shared_byte_equal(model, stock)
    assert shared
    assert not torch.equal(model.temporal.recency_slopes.cpu(), stock.temporal.recency_slopes.cpu())
    init = train_mod.assert_paired_initialization(model, device, cfg)
    assert init["named_parameters_byte_equal"] is True
    assert init["new_parameter_names"] == ["temporal.slope_log"]
    assert init["new_parameter_counts"] == {"temporal.slope_log": 32}
    assert init["trainable_new_parameter_count"] == 24
    assert init["init_forward_max_abs"] <= 1e-6
    assert init["recency_slopes_differ_from_stock"] is True
    ladder_ref = RiftDecoder("m2", context_bins=50, bias_mode="recency", seed=42, proj_dim=16)
    install_temporal(ladder_ref, dataset_config("m2", tier="fixed"), 42)
    ladder_ref.temporal.set_attention_backend("local")
    torch.manual_seed(11)
    z = torch.randn(2, 50, 256)
    mask = torch.ones(2, 50, dtype=torch.bool)
    mask[0, :3] = False
    with torch.inference_mode():
        torch.testing.assert_close(model.temporal(z, mask), ladder_ref.temporal(z, mask), atol=1e-6, rtol=1e-6)
