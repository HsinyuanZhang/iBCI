from __future__ import annotations

import torch

from btransform_unified_v2.concat_model import RiftConcatDecoder
from btransform_unified_v2.config import DEFAULT_HALF_LIVES
from learnable_recency_v1.config import DATASET_TABLE, dataset_config, scaled_half_lives
from learnable_recency_v1.wrap import (
    LearnableRiftConcatDecoder,
    assert_shared_byte_equal,
    install_temporal,
    new_parameter_names,
    trainable_new_parameter_count,
)

from helpers import make_pair, scramble_new_params, stream


def test_default_learned_slope_has_24_new_params() -> None:
    cfg = dataset_config("m2")
    assert cfg.per_layer is True
    assert cfg.learn_flat_heads is False
    assert cfg.layers == 4
    model = LearnableRiftConcatDecoder("m2", cfg, seed=42)
    assert trainable_new_parameter_count(model) == 24
    assert tuple(model.temporal.slope_log.shape) == (4, 8)
    assert DATASET_TABLE["m2"].per_layer is True
    assert DATASET_TABLE["h1"].per_layer is True
    assert DATASET_TABLE["m1"].per_layer is True


def test_scaled_half_lives_three_tasks() -> None:
    h1 = scaled_half_lives(300, 4)
    assert h1 == DEFAULT_HALF_LIVES
    m1 = scaled_half_lives(100, 4)
    m2 = scaled_half_lives(50, 4)
    assert m1 == tuple(None if half is None else half * (25 / 75) for half in DEFAULT_HALF_LIVES)
    assert m2 == tuple(None if half is None else half * (13 / 75) for half in DEFAULT_HALF_LIVES)
    assert dataset_config("h1").half_life_seconds == DEFAULT_HALF_LIVES
    assert dataset_config("m1").half_life_seconds == m1
    assert dataset_config("m2").half_life_seconds == m2


def test_layers_3_windows() -> None:
    assert tuple(dataset_config("m2", layers=3).temporal_config.windows) == (16, 16, 16)
    assert tuple(dataset_config("h1", layers=3).temporal_config.windows) == (100, 99, 99)
    assert tuple(dataset_config("m1", layers=3).temporal_config.windows) == (33, 33, 32)


def test_fixed_tier_zero_new_params_and_scaled_buffer() -> None:
    cfg = dataset_config("m2", tier="fixed")
    model = LearnableRiftConcatDecoder("m2", cfg, seed=42).eval()
    stock = RiftConcatDecoder("m2", context_bins=50, bias_mode="recency", seed=42).eval()
    assert_shared_byte_equal(model, stock)
    assert new_parameter_names(model) == []
    assert trainable_new_parameter_count(model) == 0
    assert not torch.equal(model.temporal.recency_slopes.cpu(), stock.temporal.recency_slopes.cpu())
    ladder_ref = RiftConcatDecoder("m2", context_bins=50, bias_mode="recency", seed=42).eval()
    install_temporal(ladder_ref, cfg, 42)
    torch.testing.assert_close(model.temporal.recency_slopes, ladder_ref.temporal.recency_slopes)
    z = torch.randn(2, 20, 256)
    mask = torch.ones(2, 20, dtype=torch.bool)
    torch.testing.assert_close(model.temporal(z, mask), ladder_ref.temporal(z, mask), atol=1e-6, rtol=1e-6)


def test_per_layer_init_parity_and_streaming() -> None:
    fixed, learnable = make_pair(50, "learned_slope", seed=7, per_layer=True)
    assert_shared_byte_equal(fixed, learnable)
    z = torch.randn(2, 14, 16)
    mask = torch.ones(2, 14, dtype=torch.bool)
    mask[0, :3] = False
    torch.testing.assert_close(learnable(z, mask), fixed(z, mask), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(stream(learnable, z, mask), stream(fixed, z, mask), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(learnable(z, mask, backend="dense"), fixed(z, mask, backend="dense"), atol=1e-6, rtol=1e-6)


def test_default_fox_cable_param_counts() -> None:
    fox = LearnableRiftConcatDecoder("m2", dataset_config("m2", tier="fox_gate"), seed=42)
    cable = LearnableRiftConcatDecoder("m2", dataset_config("m2", tier="cable"), seed=42)
    assert trainable_new_parameter_count(fox) == 8224
    assert trainable_new_parameter_count(cable) == 25216
