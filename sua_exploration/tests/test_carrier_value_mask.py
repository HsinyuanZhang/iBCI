from __future__ import annotations

import pytest
import torch

from mc_maze.carrier_value_mask import (
    CarrierValueMaskError,
    decode_with_unit_mask,
    first_layer_value_weighted_importance,
    masked_share,
    unit_mask,
)
from src.models.components.spint import SpintModel
from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder
from src.models.components.streaming_spint import StreamingSpintModel


def _student() -> StreamingSpintModel:
    torch.manual_seed(91)
    decoder = SpintModel(
        model_dim=16,
        num_covariates=2,
        window_size=6,
        num_heads=4,
        num_layers=1,
        num_id_layers=1,
        dropout_rate=0.0,
        dynamic_dropout=False,
        tf_drop_rate=0.0,
    )
    encoder = SideFeatureEarlyPoolEncoder(
        trial_length=8, window_size=6, hidden_dim=5, side_dim=4, num_post_layers=2
    )
    return StreamingSpintModel(decoder=decoder, id_encoder=encoder).eval()


def _batch():
    torch.manual_seed(92)
    neural = torch.randn(2, 6, 8)
    calib = torch.randn(2, 3, 8, 8)
    side = torch.randn(2, 8, 4)
    side[0, :, 2] = torch.arange(8)
    side[1, :, 2] = torch.arange(8).flip(0)
    return neural, calib, side


def test_low_and_high_gain_masks_are_exact_quartiles() -> None:
    _neural, _calib, side = _batch()
    low = unit_mask(side, fraction=0.25, mode="low_gain")
    high = unit_mask(side, fraction=0.25, mode="high_gain")
    assert low.sum(dim=1).tolist() == [2, 2]
    assert high.sum(dim=1).tolist() == [2, 2]
    assert low[0].nonzero().flatten().tolist() == [0, 1]
    assert high[0].nonzero().flatten().tolist() == [6, 7]
    assert low[1].nonzero().flatten().tolist() == [6, 7]


def test_random_mask_is_deterministic_and_count_matched() -> None:
    _neural, _calib, side = _batch()
    left = unit_mask(side, fraction=0.25, mode="random", seed=123)
    right = unit_mask(side, fraction=0.25, mode="random", seed=123)
    other = unit_mask(side, fraction=0.25, mode="random", seed=125)
    assert torch.equal(left, right)
    assert left.sum(dim=1).tolist() == [2, 2]
    assert torch.equal(left[0], left[1])
    assert not torch.equal(left, other)


def test_masked_forward_and_value_weighted_probe_are_finite() -> None:
    student = _student()
    neural, calib, side = _batch()
    mask = unit_mask(side, fraction=0.25, mode="low_gain")
    with torch.no_grad():
        prediction = decode_with_unit_mask(student, neural, calib, side, mask)
        importance = first_layer_value_weighted_importance(student, neural, calib, side)
    assert prediction.shape == (2, 6, 2)
    assert torch.isfinite(prediction).all()
    assert importance.attention_mass.shape == (2, 8)
    assert importance.projected_value_norm.shape == (2, 8)
    assert importance.value_weighted_mass.shape == (2, 8)
    share = masked_share(importance.value_weighted_mass, mask)
    assert torch.isfinite(share).all()
    assert ((share >= 0) & (share <= 1)).all()


def test_all_unit_mask_and_training_mode_fail_closed() -> None:
    student = _student()
    neural, calib, side = _batch()
    all_mask = torch.ones(2, 8, dtype=torch.bool)
    with pytest.raises(CarrierValueMaskError, match="every unit"):
        decode_with_unit_mask(student, neural, calib, side, all_mask)
    student.train()
    with pytest.raises(CarrierValueMaskError, match="eval mode"):
        first_layer_value_weighted_importance(student, neural, calib, side)
