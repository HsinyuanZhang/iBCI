"""CPU contracts for the teacher-head-preserving streaming adapter."""
from __future__ import annotations

import torch
from torch import nn

from src.models.components.spint import SpintModel
from src.models.components.streaming_spint_head_oracle_adapter import (
    TeacherHeadOracleStreamingSpint,
)


class _IdentityEncoder(nn.Module):
    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.window_size = window_size

    def forward_batch(
        self,
        calib_trials,
        side_features=None,
        electrode_ids=None,
    ):
        del side_features, electrode_ids
        return calib_trials.mean(dim=1).permute(0, 2, 1) * self.scale


def _decoder() -> SpintModel:
    torch.manual_seed(23)
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
    decoder.fc_id_in(torch.zeros(1, 1, 1, 6))
    return decoder


def _model(mode: str = "e_t4"):
    return TeacherHeadOracleStreamingSpint(
        decoder=_decoder(),
        id_encoder=_IdentityEncoder(6),
        key_mode=mode,
        key_permutation_seed=41 if mode == "e_ts4" else None,
    )


def test_cached_and_training_paths_match_without_online_identity_readin(
    monkeypatch,
):
    model = _model().eval()
    neural = torch.randn(3, 6, 5)
    identity = torch.randn(1, 5, 6)
    calls = []
    original = model.decoder.fc_in.forward

    def capture(value):
        calls.append(tuple(value.shape))
        return original(value)

    monkeypatch.setattr(model.decoder.fc_in, "forward", capture)
    direct = model.decode_with_head_oracle_identity(neural, identity)
    assert calls == [(3, 5, 6), (1, 2, 6), (3, 5, 6)]

    calls.clear()
    state = model.derive_head_oracle_state(identity)
    assert calls == [(1, 5, 6)]
    calls.clear()
    cached = model.decode_with_head_oracle_state(neural, state)
    assert calls == [(3, 5, 6), (1, 2, 6)]
    torch.testing.assert_close(direct, cached, rtol=1e-6, atol=1e-6)


def test_ts4_changes_only_decoder_key_identity_rows():
    aligned = _model("e_t4").eval()
    shuffled = _model("e_ts4").eval()
    shuffled.load_state_dict(aligned.state_dict(), strict=False)
    identity = torch.arange(30, dtype=torch.float32).view(1, 5, 6)
    expected_order = torch.tensor([1, 2, 4, 3, 0])
    assert torch.equal(
        shuffled._identity_key_input(identity),
        identity[:, expected_order],
    )
    assert torch.equal(aligned._identity_key_input(identity), identity)


def test_forward_keeps_encoder_t4_aligned_and_has_no_direct_branch():
    model = _model("e_ts4").eval()
    neural = torch.randn(2, 6, 5)
    calib = torch.randn(2, 3, 6, 5)
    side = torch.randn(2, 5, 4)
    observed = []
    original = model.id_encoder.forward_batch

    def capture(*args, **kwargs):
        observed.append(kwargs["side_features"])
        return original(*args, **kwargs)

    model.id_encoder.forward_batch = capture
    prediction, identity = model(
        neural, calib_trials=calib, side_features=side
    )
    assert prediction.shape == (2, 6, 2)
    assert identity.shape == (2, 5, 6)
    assert observed == [side]
    try:
        model(
            neural,
            calib_trials=calib,
            side_features=side,
            decoder_key_features=side,
        )
    except ValueError as error:
        assert "no direct decoder T4 branch" in str(error)
    else:
        raise AssertionError("direct decoder features must fail closed")


def test_actual_reference_cost_and_initialization_receipts():
    decoder = SpintModel(
        model_dim=512,
        num_covariates=2,
        window_size=50,
        num_heads=64,
        num_layers=1,
        num_id_layers=1,
        dropout_rate=0.0,
        dynamic_dropout=False,
        tf_drop_rate=0.1,
    )
    decoder.fc_id_in(torch.zeros(1, 1, 1, 50))
    model = TeacherHeadOracleStreamingSpint(
        decoder=decoder,
        id_encoder=_IdentityEncoder(50),
        key_mode="e_t4",
    )
    cost = model.oracle_cost_receipt()
    assert cost["online_macs_per_window"]["total"] == 41_193_472
    assert cost["calibration_only_macs"]["total"] == 35_192_832
    assert cost["persistent_state"]["bytes_fp32"] == 131_072
    assert cost["online_mac_reduction_fraction_vs_coupled"] > 0.2894
    receipt = model.oracle_initialization_receipt
    assert receipt["teacher_head_count"] == 64
    assert receipt["teacher_headwise_softmax_preserved"] is True
    assert receipt["legacy_transformer_active"] is False
