"""Wiring tests for the M1 proj_add P=16 learnable-recency runner (CPU only)."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from btransform_unified_v2.config import DEFAULT_HALF_LIVES
from btransform_unified_v2.model import RiftDecoder
from learnable_recency_v1.config import dataset_config
from learnable_recency_v1.temporal import LearnableRecencyTemporal
from learnable_recency_v1.wrap import (
    assert_shared_byte_equal,
    new_parameter_names,
    trainable_new_parameter_count,
)

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "m1_projadd_learnable_train.py"
_spec = importlib.util.spec_from_file_location("_m1_projadd_learnable_train", RUNNER_PATH)
assert _spec is not None and _spec.loader is not None
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

DEVICE = torch.device("cpu")


def _full_named_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def learned_m1():
    cfg = dataset_config("m1", tier="learned_slope")
    return runner.learnable_decoder(DEVICE, cfg), cfg


def test_m1_config_defaults_scaled_ladder():
    cfg = dataset_config("m1")
    assert cfg.task == "m1" and cfg.tier == "learned_slope"
    assert cfg.context_bins == 100 and cfg.layers == 4
    assert cfg.per_layer is True and cfg.learn_flat_heads is False
    assert cfg.ladder == "scaled"
    factor = 25 / 75  # first-layer window W0 over the H1 anchor 75
    assert cfg.half_life_seconds == tuple(None if half is None else half * factor for half in DEFAULT_HALF_LIVES)
    metadata = cfg.ladder_metadata()
    assert metadata["first_layer_window"] == 25
    assert metadata["scale_vs_h1"] == pytest.approx(1 / 3)
    assert metadata["windows"] == [25, 25, 25, 24]
    assert cfg.temporal_config.windows == (25, 25, 25, 24)
    assert list(metadata["half_life_bins"][:6]) == pytest.approx([4 / 3, 8 / 3, 16 / 3, 32 / 3, 64 / 3, 128 / 3])
    assert metadata["half_life_bins"][6] is None and metadata["half_life_bins"][7] is None


def test_decoder_factory_learned_slope(learned_m1):
    model, _cfg = learned_m1
    assert isinstance(model.temporal, LearnableRecencyTemporal)
    assert model.temporal._attention_backend == "local"
    assert tuple(model.temporal_config.windows) == (25, 25, 25, 24)
    assert model.context_bins == 100 and model.proj_dim == 16
    assert new_parameter_names(model) == ["temporal.slope_log"]
    slope_log = dict(model.named_parameters())["temporal.slope_log"]
    assert tuple(slope_log.shape) == (4, 8)
    assert torch.count_nonzero(slope_log).item() == 0
    assert trainable_new_parameter_count(model) == 24  # 4 layers x 6 active heads


def test_decoder_factory_fixed_tier_uses_plain_scaled_temporal():
    model = runner.learnable_decoder(DEVICE, dataset_config("m1", tier="fixed"))
    assert not isinstance(model.temporal, LearnableRecencyTemporal)
    assert new_parameter_names(model) == []
    assert trainable_new_parameter_count(model) == 0
    assert tuple(model.temporal_config.windows) == (25, 25, 25, 24)
    # Scaled M1 ladder (factor 1/3): first head half-life 0.08*25/75 seconds, flat heads stay 0.
    # recency_slopes is float32, so compare at float32 precision.
    assert model.temporal.recency_slopes[0].item() == pytest.approx(math.log(2.0) * 0.02 / (0.08 * (25 / 75)), rel=1e-6)
    assert model.temporal.recency_slopes[6].item() == 0.0
    assert model.temporal.recency_slopes[7].item() == 0.0


def test_shared_parameters_byte_equal_and_init_forward_parity(learned_m1):
    model, cfg = learned_m1
    stock = RiftDecoder("m1", context_bins=100, bias_mode="recency", seed=42, proj_dim=16)
    stock.temporal.set_attention_backend("local")
    shared = assert_shared_byte_equal(model, stock)
    assert shared, "shared parameter set must be nonempty"
    assert "temporal.blocks.0.qkv.weight" in shared
    assert set(new_parameter_names(model)).isdisjoint(shared)
    # Only the recency_slopes buffer differs (scaled vs DEFAULT ladder); parameters do not.
    assert not torch.equal(model.temporal.recency_slopes, stock.temporal.recency_slopes)
    expected_sha = _full_named_sha(stock)
    assert runner.shared_init_sha(model, shared) == expected_sha
    report = runner.assert_paired(model, DEVICE, {"initialization_sha256": expected_sha}, cfg)
    assert report["shared_parameter_names"] == shared
    assert report["trainable_new_parameter_count"] == 24
    assert report["init_forward_max_abs"] <= 1e-6
