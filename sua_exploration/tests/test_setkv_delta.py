from __future__ import annotations

import copy

import pytest
import torch

from mc_maze.setkv_delta import (
    SetKVDeltaError,
    carrier_hidden_tokens,
    decode_setkv_delta,
    setkv_delta_cost_receipt,
)
from src.models.components.spint import SpintModel
from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder
from src.models.components.streaming_spint import StreamingSpintModel


def _student() -> StreamingSpintModel:
    torch.manual_seed(17)
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
        trial_length=8,
        window_size=6,
        hidden_dim=5,
        side_dim=4,
        num_post_layers=2,
    )
    # Production B3S initializes the side columns to zero.  Make them
    # deterministically nonzero so the synthetic content controls are live.
    first_linear = next(module for module in encoder.post_pool if isinstance(module, torch.nn.Linear))
    with torch.no_grad():
        first_linear.weight[:, encoder.hidden_dim :].copy_(
            torch.arange(first_linear.out_features * 4, dtype=first_linear.weight.dtype).reshape(first_linear.out_features, 4) / 37.0
        )
    return StreamingSpintModel(decoder=decoder, id_encoder=encoder).eval()


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(23)
    return (
        torch.randn(2, 6, 7),
        torch.randn(2, 3, 8, 7),
        torch.randn(2, 7, 4),
    )


def test_baseline_path_is_bitwise_production_forward() -> None:
    model = _student()
    neural, calib, side = _batch()
    with torch.no_grad():
        expected, expected_identity = model(neural, calib_trials=calib, side_features=side)
        observed = decode_setkv_delta(
            model, neural, calib, side, decode_mode="baseline"
        )
    assert torch.equal(observed.prediction, expected)
    assert torch.equal(observed.identity, expected_identity)
    assert observed.appended_tokens is None


def test_zero_carrier_token_is_bitwise_zero_after_both_bias_cancellations() -> None:
    model = _student()
    _neural, _calib, side = _batch()
    with torch.no_grad():
        token = carrier_hidden_tokens(model, torch.zeros_like(side))
    assert torch.equal(token, torch.zeros_like(token))


def test_aligned_carrier_tokens_are_live_and_register_no_parameters() -> None:
    model = _student()
    before = tuple((name, id(value)) for name, value in model.named_parameters())
    neural, calib, side = _batch()
    with torch.no_grad():
        result = decode_setkv_delta(
            model, neural, calib, side, decode_mode="carrier", carrier_mode="aligned"
        )
    after = tuple((name, id(value)) for name, value in model.named_parameters())
    assert before == after
    assert result.appended_tokens is not None
    assert result.appended_tokens.shape == (2, 7, 16)
    assert torch.count_nonzero(result.appended_tokens).item() > 0
    assert torch.isfinite(result.prediction).all()
    assert result.attention_scores[0].shape[-1] == 14


def test_row_shuffle_only_permutes_carrier_attachment() -> None:
    model = _student()
    neural, calib, side = _batch()
    with torch.no_grad():
        aligned = decode_setkv_delta(
            model, neural, calib, side, decode_mode="carrier", carrier_mode="aligned"
        )
        shuffled = decode_setkv_delta(
            model,
            neural,
            calib,
            side,
            decode_mode="carrier",
            carrier_mode="row_shuffle",
            permutation_seed=20260814,
        )
    assert shuffled.permutation is not None
    assert torch.equal(shuffled.identity, aligned.identity)
    assert torch.equal(shuffled.activity_tokens, aligned.activity_tokens)
    assert torch.equal(
        shuffled.appended_tokens,
        aligned.appended_tokens.index_select(1, shuffled.permutation),
    )


def test_duplicate_activity_is_a_distinct_set_size_control() -> None:
    model = _student()
    neural, calib, side = _batch()
    with torch.no_grad():
        result = decode_setkv_delta(
            model, neural, calib, side, decode_mode="duplicate_activity"
        )
    assert torch.equal(result.appended_tokens, result.activity_tokens)
    assert result.attention_scores[0].shape[-1] == 14


def test_training_mode_and_missing_shuffle_seed_fail_closed() -> None:
    model = _student()
    neural, calib, side = _batch()
    model.train()
    with pytest.raises(SetKVDeltaError, match="eval mode"):
        decode_setkv_delta(model, neural, calib, side, decode_mode="carrier")
    model.eval()
    with pytest.raises(SetKVDeltaError, match="permutation seed"):
        decode_setkv_delta(
            model, neural, calib, side, decode_mode="carrier", carrier_mode="row_shuffle"
        )


def test_cost_receipt_is_explicit_and_zero_parameter() -> None:
    receipt = setkv_delta_cost_receipt(num_units=96, num_queries=2, model_dim=512)
    assert receipt["parameter_delta"] == 0
    assert receipt["extra_token_count"] == 96
    assert receipt["persistent_carrier_token_state_bytes"] == 96 * 512 * 4
    assert receipt["extra_dense_mha_macs_per_window"] == (
        2 * 96 * 512 * 512 + 2 * 2 * 96 * 512
    )
