"""CPU-only contracts for the DANDI 688 v2 MOVE-T4/RIFT components."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT.parent, ROOT / "src", ROOT / "learnable_recency_v1" / "src", ROOT.parent / "btransform_unified_v1" / "src", ROOT.parent / "streaming_calibration_exp"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.data.falcon_t4_features import t4_from_trial_sums
from dandi688_bench_v2.carrier import MOVE_BINS, MOVE_DURATION_SECONDS, apply_carrier_normalizer, estimate_move_t4, fit_carrier_normalizer
from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder
from dandi688_bench_v2.model import B3SIdentityEncoder, DandiRiftDecoder
from learnable_recency_v1.config import dataset_config
from learnable_recency_v1.wrap import LearnableRiftDecoder
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_calibration import FrozenC2Materializer


def test_move_t4_matches_formal_m2_ols_oracle_and_source_normalizer() -> None:
    rng = np.random.default_rng(8)
    counts = rng.poisson(3.0, size=(10, 5)).astype(np.float32)
    angles = np.linspace(-2.7, 2.8, 10, dtype=np.float64)
    got = estimate_move_t4(counts, angles)
    expected = t4_from_trial_sums(counts, np.full(10, MOVE_BINS), angles, source="oracle")
    assert np.array_equal(got, expected)
    other = got + np.float32(1.0)
    state = fit_carrier_normalizer([got, other])
    joined = np.concatenate([got, other], axis=0)
    expected_mean, expected_std = joined.mean(0).astype(np.float32), joined.std(0).astype(np.float32)
    expected_std[expected_std <= 1e-6] = 1.0
    assert np.array_equal(state["mean"], expected_mean)
    assert np.array_equal(state["std"], expected_std)
    assert np.array_equal(apply_carrier_normalizer(got, state), ((got - expected_mean) / expected_std).astype(np.float32))


def test_move_t4_rejects_nonformal_window_and_rank_deficient_angles() -> None:
    counts = np.ones((4, 2), dtype=np.float32)
    angles = np.asarray([0.0, np.pi / 2, np.pi, -np.pi / 2])
    with pytest.raises(ValueError, match="fixed"):
        estimate_move_t4(counts, angles, duration_s=MOVE_DURATION_SECONDS / 2)
    with pytest.raises(ValueError, match="rank"):
        estimate_move_t4(counts, np.zeros(4))


def test_move_t4_matches_formal_oracle_when_support_contains_unlabeled_trials() -> None:
    rng = np.random.default_rng(41)
    counts = rng.poisson(2.0, size=(7, 3)).astype(np.float32)
    angles = np.asarray([0.0, np.nan, np.pi / 2, np.pi, np.nan, -np.pi / 2, .7])
    expected = t4_from_trial_sums(counts, np.full(7, MOVE_BINS), angles, source="nan-oracle")
    assert np.array_equal(estimate_move_t4(counts, angles), expected)


def test_common_m2_parameters_and_optimizer_group_contract() -> None:
    cfg = dataset_config("m2", tier="learned_slope", ladder="default")
    m2, dandi = LearnableRiftDecoder("m2", cfg, seed=42, proj_dim=16), DandiRiftDecoder("full", n_pad=4, seed=42, recency_cfg=cfg)
    assert tuple(dandi.temporal_config.windows) == (13, 12, 12, 12)
    assert dandi.temporal._attention_backend == "local"
    reference = dict(m2.named_parameters())
    for name, value in dandi.named_parameters():
        if not name.startswith("encoder.") and name in reference:
            assert value.shape == reference[name].shape and torch.equal(value, reference[name]), name
    groups = dandi.optimizer_param_groups(lr=1e-3, weight_decay=0.02, learned_lr_multiplier=3.0)
    assert [group["lr_multiplier"] for group in groups] == [1.0, 3.0]
    assert groups[0]["weight_decay"] == 0.02 and groups[1]["weight_decay"] == 0.0


def test_b3s_matches_canonical_m1_math_after_state_copy() -> None:
    torch.manual_seed(9)
    canonical = SideFeatureEarlyPoolEncoder(trial_length=100, window_size=50, hidden_dim=64,
                                            side_dim=4, num_post_layers=3)
    local = B3SIdentityEncoder(side_dim=4, seed=99)
    local.load_state_dict(canonical.state_dict())
    support, side = torch.randn(3, 100, 4), torch.randn(4, 4)
    state = canonical.reset_stream(batch_size=1, num_neurons=4, device=support.device, dtype=support.dtype)
    state["side_features"] = side.unsqueeze(0)
    for trial in support:
        state = canonical.push_trial(state, trial)
    expected = canonical.finalize_identity(state).squeeze(0)
    assert torch.equal(local(support, side), expected)


def test_b3s_side_zero_initialization_and_no_film_parameters() -> None:
    encoder = B3SIdentityEncoder(side_dim=4, seed=7)
    support, side = torch.randn(3, 100, 5), torch.randn(5, 4)
    first = encoder.post_pool[0]
    assert isinstance(first, torch.nn.Linear)
    assert torch.count_nonzero(first.weight[:, 64:]) == 0
    pooled = encoder.pre_pool(support.permute(0, 2, 1)).mean(0)
    assert torch.equal(encoder(support, side), encoder.post_pool(torch.cat((pooled, torch.zeros_like(side)), dim=-1)))
    names = {name for name, _ in encoder.named_parameters()}
    assert not any("film" in name or "contrast" in name for name in names)


def test_b3s_side_columns_are_trainable_after_zero_initialization() -> None:
    encoder = B3SIdentityEncoder(side_dim=4, seed=10).train()
    output = encoder(torch.randn(3, 100, 4), torch.randn(4, 4))
    output.square().mean().backward()
    first = encoder.post_pool[0]
    assert isinstance(first, torch.nn.Linear) and first.weight.grad is not None
    assert torch.count_nonzero(first.weight.grad[:, 64:]) > 0


def test_b3s_concat_matches_h1_frozen_c2_materializer_graph_with_nonzero_side_weights() -> None:
    """Synthetic H1-sized modules prove the shared post-pool concat graph.

    No checkpoint is loaded: the test copies randomly initialized weights into
    the legacy materializer's public computation path, then makes the side
    columns nonzero so zero initialization cannot hide a wrong fusion point.
    """
    torch.manual_seed(44)
    local = B3SIdentityEncoder(side_dim=4, support_bins=1024, hidden_dim=32, e0_dim=700,
                                num_post_layers=3, seed=4).eval()
    with torch.no_grad():
        first = local.post_pool[0]
        assert isinstance(first, torch.nn.Linear)
        first.weight[:, 32:].normal_()
    legacy = FrozenC2Materializer(
        pre_pool=torch.nn.Sequential(torch.nn.Linear(1024, 32), torch.nn.ReLU()),
        post_pool=torch.nn.Sequential(torch.nn.Linear(36, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32),
                                      torch.nn.ReLU(), torch.nn.Linear(32, 700)),
        checkpoint_sha256="synthetic",
    ).eval()
    legacy.pre_pool.load_state_dict(local.pre_pool.state_dict())
    legacy.post_pool.load_state_dict(local.post_pool.state_dict())
    activity, side = torch.randn(3, 1024, 176), torch.randn(176, 4)
    with torch.inference_mode():
        expected = legacy.fused_identity(activity.unsqueeze(0), side).squeeze(0)
        got = local(activity, side)
        without_side = local(activity, torch.zeros_like(side))
    assert torch.equal(got, expected)
    assert not torch.equal(got, without_side)


def _arm_prediction(arm: str, *, permute: bool = False) -> torch.Tensor:
    torch.manual_seed(21)
    n = 4
    model = DandiRiftDecoder(arm, n_pad=n, seed=7).eval()
    x, activity, carrier = torch.randn(2, 50, n), torch.randn(3, 100, n), torch.randn(n, 4)
    mask = torch.tensor([True, True, False, False])
    dropout = torch.tensor([True, False, True, True])
    if permute:
        # Padded payload changes arbitrarily and real rows permute jointly;
        # the set frontend must retain the same prediction.
        order = torch.tensor([1, 0, 3, 2])
        x, activity, carrier, mask, dropout = x[:, :, order], activity[:, :, order], carrier[order], mask[order], dropout[order]
        x[:, :, 2:] += 1000.0
        activity[:, :, 2:] -= 1000.0
        carrier[2:] += 1000.0
    with torch.inference_mode():
        return model(x, activity=activity if arm != "raw_set" else None,
                     carrier=carrier if arm == "full" else None, unit_mask=mask,
                     dropout_keep=dropout)


def test_padding_and_real_row_permutation_invariant_for_all_arms() -> None:
    for arm in ("full", "activity", "raw_set"):
        assert torch.allclose(_arm_prediction(arm), _arm_prediction(arm, permute=True), atol=1e-6, rtol=0.0), arm


def test_gradient_routing_for_full_activity_and_raw_set() -> None:
    for arm in ("full", "activity", "raw_set"):
        model = DandiRiftDecoder(arm, n_pad=4, seed=13).train()
        x, activity, carrier = torch.randn(2, 50, 4), torch.randn(3, 100, 4), torch.randn(4, 4)
        output = model(x, activity=activity if arm != "raw_set" else None,
                       carrier=carrier if arm == "full" else None, unit_mask=torch.tensor([True, True, True, False]))
        output.square().mean().backward()
        if arm == "full":
            assert all(parameter.grad is None for parameter in model.encoder.parameters())
        elif arm == "activity":
            assert any(parameter.grad is not None for parameter in model.encoder.parameters())
        else:
            assert model.global_g.grad is not None
