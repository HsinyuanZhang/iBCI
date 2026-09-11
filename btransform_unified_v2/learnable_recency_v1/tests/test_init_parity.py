from __future__ import annotations

import pytest
import torch

from btransform_unified_v2.concat_model import RiftConcatDecoder
from btransform_unified_v2.config import RiftTemporalConfig
from btransform_unified_v2.temporal import RiftTemporal
from learnable_recency_v1.config import dataset_config
from learnable_recency_v1.temporal import LearnableRecencyTemporal
from learnable_recency_v1.wrap import (
    LearnableRiftConcatDecoder,
    assert_shared_byte_equal,
    install_temporal,
    new_parameter_names,
    trainable_new_parameter_count,
)

from helpers import TIERS, make_pair, recency_config, scramble_new_params, temporal_config


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("context_bins", [50, 100, 300])
def test_shared_parameters_byte_equal_and_outputs_match(tier: str, context_bins: int) -> None:
    width = 16 if context_bins == 300 else 32
    fixed, learnable = make_pair(context_bins, tier, seed=42, width=width, heads=4)
    shared = assert_shared_byte_equal(fixed, learnable)
    assert shared
    extra = set(dict(learnable.named_parameters())) - set(dict(fixed.named_parameters()))
    assert extra == set(learnable.new_parameter_names())
    torch.manual_seed(9)
    time = 11 if context_bins == 300 else 16
    z = torch.randn(2, time, width)
    mask = torch.tensor([[False, False] + [True] * (time - 2), [True] * time])
    torch.testing.assert_close(learnable(z, mask), fixed(z, mask), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(learnable(z, mask, backend="dense"), fixed(z, mask, backend="dense"), atol=1e-6, rtol=1e-6)


def test_new_parameters_do_not_consume_rng() -> None:
    config = temporal_config(50, width=16, heads=4)
    torch.manual_seed(3)
    _ = RiftTemporal(config)
    after_fixed = torch.get_rng_state()
    torch.manual_seed(3)
    sibling = RiftTemporal(config)
    _ = LearnableRecencyTemporal.from_initialized(sibling, recency_config("cable", 50, heads=4))
    after_wrap = torch.get_rng_state()
    assert torch.equal(after_fixed, after_wrap)


def test_decoder_init_parity_m2_all_tiers() -> None:
    torch.manual_seed(0)
    batch = torch.randn(2, 50, 256)
    mask = torch.ones(2, 50, dtype=torch.bool)
    mask[0, :4] = False
    stock = RiftConcatDecoder("m2", context_bins=50, bias_mode="recency", seed=42).eval()
    ladder_ref = RiftConcatDecoder("m2", context_bins=50, bias_mode="recency", seed=42).eval()
    install_temporal(ladder_ref, dataset_config("m2", tier="fixed"), seed=42)
    ladder_ref.eval()
    for tier in TIERS:
        learnable = LearnableRiftConcatDecoder("m2", dataset_config("m2", tier=tier), seed=42).eval()
        assert_shared_byte_equal(stock, learnable)
        names = new_parameter_names(learnable)
        assert names
        assert all(name.startswith("temporal.") for name in names)
        torch.testing.assert_close(
            learnable.temporal(batch, mask), ladder_ref.temporal(batch, mask), atol=1e-6, rtol=1e-6
        )


def test_learned_slope_per_layer_init_parity() -> None:
    fixed, learnable = make_pair(50, "learned_slope", seed=5, per_layer=True)
    assert_shared_byte_equal(fixed, learnable)
    assert tuple(learnable.slope_log.shape) == (4, 4)
    z = torch.randn(1, 12, 16)
    torch.testing.assert_close(learnable(z), fixed(z), atol=1e-6, rtol=1e-6)


def test_cable_nw_init_parity() -> None:
    fixed, learnable = make_pair(50, "cable", seed=6, cable_nw=True)
    assert_shared_byte_equal(fixed, learnable)
    assert "cable_gate_weight" not in learnable.new_parameter_names()
    z = torch.randn(2, 10, 16)
    torch.testing.assert_close(learnable(z), fixed(z), atol=1e-6, rtol=1e-6)
