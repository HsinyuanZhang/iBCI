"""CPU contracts for the A1 hidden-space carrier adapter.

W-add remains the default student path. H-add applies ``fc_in`` to the
matched activity identity and adds a bias-free ``P(carrier)`` in hidden
space. Decoder and teacher tensors stay strict-loadable and are never
replaced.
"""
from __future__ import annotations

import copy
import hashlib

import pytest
import torch
from torch import nn

from src.models.components.spint import SpintModel
from src.models.components.streaming_spint import StreamingSpintModel
from src.models.components.streaming_spint_hidden_carrier_adapter import (
    HiddenCarrierState,
    HiddenSpaceCarrierStreamingSpint,
    ZeroInitializedHiddenCarrierMap,
)


class _IdentityEncoder(nn.Module):
    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.window_size = window_size
        self.observed_side_features: list[torch.Tensor | None] = []

    def forward_batch(
        self,
        calib_trials,
        side_features=None,
        electrode_ids=None,
    ):
        del electrode_ids
        self.observed_side_features.append(
            None if side_features is None else side_features.detach().clone()
        )
        return calib_trials.mean(dim=1).permute(0, 2, 1) * self.scale


def _decoder(
    *,
    model_dim: int = 16,
    window_size: int = 6,
    num_heads: int = 4,
) -> SpintModel:
    torch.manual_seed(13)
    decoder = SpintModel(
        model_dim=model_dim,
        num_covariates=2,
        window_size=window_size,
        num_heads=num_heads,
        num_layers=1,
        num_id_layers=1,
        dropout_rate=0.0,
        dynamic_dropout=False,
        tf_drop_rate=0.0,
    )
    decoder.fc_id_in(torch.zeros(1, 1, 1, window_size))
    return decoder


def _state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _w_add(
    decoder: SpintModel | None = None,
) -> StreamingSpintModel:
    return StreamingSpintModel(
        decoder=decoder if decoder is not None else _decoder(),
        id_encoder=_IdentityEncoder(6),
        decoder_mode="coupled",
    )


def _h_add(
    *,
    decoder: SpintModel | None = None,
    add_site: str = "hidden",
    attachment_mode: str = "aligned",
    permutation_seed: int | None = None,
) -> HiddenSpaceCarrierStreamingSpint:
    return HiddenSpaceCarrierStreamingSpint(
        decoder=decoder if decoder is not None else _decoder(),
        id_encoder=_IdentityEncoder(6),
        add_site=add_site,
        attachment_mode=attachment_mode,
        permutation_seed=permutation_seed,
    )


def test_disabled_adapter_is_bitwise_identical_to_w_add() -> None:
    decoder = _decoder()
    w_add = _w_add(copy.deepcopy(decoder)).eval()
    disabled = _h_add(decoder=copy.deepcopy(decoder), add_site="waveform").eval()
    neural = torch.randn(3, 6, 5)
    identity = torch.randn(1, 5, 6)
    t4 = torch.randn(1, 5, 4)

    expected = w_add.decode_with_identity(neural, identity)
    actual = disabled.decode_with_identity(neural, identity)
    assert torch.equal(actual, expected)

    calib = torch.randn(3, 2, 6, 5)
    expected_out, expected_id = w_add(neural, calib_trials=calib, side_features=t4)
    actual_out, actual_id = disabled(neural, calib_trials=calib, side_features=t4)
    assert torch.equal(actual_out, expected_out)
    assert torch.equal(actual_id, expected_id)
    assert disabled.add_site == "waveform"
    # A W-add compatibility wrapper must not register a dead adapter factor.
    assert disabled.hidden_carrier_map is None


def test_p_of_zero_is_exactly_zero_even_after_nonzero_weights() -> None:
    projection = ZeroInitializedHiddenCarrierMap(carrier_dim=4, hidden_dim=16)
    zeros = torch.zeros(2, 5, 4)
    assert torch.equal(projection(zeros), torch.zeros(2, 5, 16))
    with torch.no_grad():
        projection.projection.weight.fill_(0.25)
    assert torch.equal(projection(zeros), torch.zeros(2, 5, 16))
    assert projection.bias_parameters == 0


def test_zero_initialized_h_add_matches_w_add_with_activity_identity() -> None:
    decoder = _decoder()
    w_add = _w_add(copy.deepcopy(decoder)).eval()
    h_add = _h_add(decoder=copy.deepcopy(decoder), add_site="hidden").eval()
    neural = torch.randn(3, 6, 5)
    activity_identity = torch.randn(1, 5, 6)
    t4 = torch.randn(1, 5, 4)

    state = h_add.derive_hidden_carrier_state(t4)
    assert torch.count_nonzero(state.hidden_add).item() == 0
    expected = w_add.decode_with_identity(neural, activity_identity)
    actual = h_add.decode_with_hidden_carrier_state(
        neural, activity_identity, state
    )
    assert torch.equal(actual, expected)


def test_h_add_z4_matches_w_add_z4_after_p_is_trained() -> None:
    decoder = _decoder()
    w_add = _w_add(copy.deepcopy(decoder)).eval()
    h_add = _h_add(decoder=copy.deepcopy(decoder), add_site="hidden")
    with torch.no_grad():
        h_add.hidden_carrier_map.projection.weight.fill_(0.07)
    h_add.eval()

    neural = torch.randn(2, 6, 5)
    activity_identity = torch.randn(1, 5, 6)
    z4 = torch.zeros(1, 5, 4)
    expected = w_add.decode_with_identity(neural, activity_identity)
    actual = h_add.decode_with_hidden_carrier(
        neural, activity_identity, z4
    )
    assert torch.equal(actual, expected)

    def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            neural.detach().clone().requires_grad_(True),
            activity_identity.detach().clone().requires_grad_(True),
        )

    neural_w, identity_w = _inputs()
    w_add.decode_with_identity(neural_w, identity_w).square().mean().backward()
    neural_h, identity_h = _inputs()
    h_add.decode_with_hidden_carrier(
        neural_h, identity_h, z4
    ).square().mean().backward()
    assert torch.equal(neural_w.grad, neural_h.grad)
    assert torch.equal(identity_w.grad, identity_h.grad)


def test_decoder_and_teacher_state_remain_strict_loadable_and_untouched() -> None:
    teacher = _decoder()
    teacher_sha = _state_sha256(teacher)
    student = _h_add(decoder=copy.deepcopy(teacher), add_site="hidden")
    before = _state_sha256(student.decoder)
    student.decoder.load_state_dict(teacher.state_dict(), strict=True)
    after_load = _state_sha256(student.decoder)
    neural = torch.randn(2, 6, 5)
    identity = torch.randn(1, 5, 6)
    t4 = torch.randn(1, 5, 4)
    student.eval()
    student.decode_with_hidden_carrier(neural, identity, t4)
    after_forward = _state_sha256(student.decoder)
    assert before == teacher_sha
    assert after_load == teacher_sha
    assert after_forward == teacher_sha
    extra = {
        name: tensor
        for name, tensor in student.state_dict().items()
        if not name.startswith("decoder.")
    }
    assert extra
    teacher.load_state_dict(teacher.state_dict(), strict=True)


def test_h_add_encoder_receives_activity_only_identity() -> None:
    model = _h_add(add_site="hidden").eval()
    neural = torch.randn(2, 6, 5)
    calib = torch.randn(2, 3, 6, 5)
    t4 = torch.arange(40, dtype=torch.float32).view(2, 5, 4)
    _, identity = model(neural, calib_trials=calib, side_features=t4)
    observed = model.id_encoder.observed_side_features[0]
    assert observed is not None
    assert torch.equal(observed, torch.zeros_like(t4))
    assert identity.shape == (2, 5, 6)


def test_w_add_mode_encoder_still_receives_carrier() -> None:
    model = _h_add(add_site="waveform").eval()
    neural = torch.randn(2, 6, 5)
    calib = torch.randn(2, 3, 6, 5)
    t4 = torch.randn(2, 5, 4)
    model(neural, calib_trials=calib, side_features=t4)
    observed = model.id_encoder.observed_side_features[0]
    assert observed is not None
    assert torch.equal(observed, t4)


def test_ts4_shuffled_control_changes_only_carrier_attachment() -> None:
    aligned = _h_add(add_site="hidden", attachment_mode="aligned").eval()
    shuffled = _h_add(
        add_site="hidden",
        attachment_mode="shuffled",
        permutation_seed=41,
    ).eval()
    shuffled.load_state_dict(aligned.state_dict(), strict=False)
    with torch.no_grad():
        aligned.hidden_carrier_map.projection.weight.fill_(0.11)
        shuffled.hidden_carrier_map.projection.weight.fill_(0.11)
    neural = torch.randn(2, 6, 5)
    calib = torch.randn(2, 3, 6, 5)
    t4 = torch.arange(40, dtype=torch.float32).view(2, 5, 4)

    aligned_pred, aligned_identity = aligned(
        neural, calib_trials=calib, side_features=t4
    )
    shuffled_pred, shuffled_identity = shuffled(
        neural, calib_trials=calib, side_features=t4
    )
    assert torch.equal(aligned_identity, shuffled_identity)
    assert torch.equal(
        aligned.id_encoder.observed_side_features[0],
        torch.zeros_like(t4),
    )
    assert torch.equal(
        shuffled.id_encoder.observed_side_features[0],
        torch.zeros_like(t4),
    )
    assert not torch.equal(
        aligned.derive_hidden_carrier_state(t4).hidden_add,
        shuffled.derive_hidden_carrier_state(t4).hidden_add,
    )
    assert not torch.equal(aligned_pred, shuffled_pred)


def test_support_and_query_tensors_are_identical_across_add_sites() -> None:
    decoder = _decoder()
    w_add = _h_add(decoder=copy.deepcopy(decoder), add_site="waveform")
    h_add = _h_add(decoder=copy.deepcopy(decoder), add_site="hidden")
    neural = torch.randn(2, 6, 5)
    calib = torch.randn(2, 3, 6, 5)
    t4 = torch.randn(2, 5, 4)
    provenance = {
        "query_neural_sha256": hashlib.sha256(
            neural.detach().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "support_calib_sha256": hashlib.sha256(
            calib.detach().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "carrier_sha256": hashlib.sha256(
            t4.detach().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }
    w_receipt = w_add.support_query_provenance_receipt(
        neural, calib_trials=calib, side_features=t4
    )
    h_receipt = h_add.support_query_provenance_receipt(
        neural, calib_trials=calib, side_features=t4
    )
    for key, value in provenance.items():
        assert w_receipt[key] == value
        assert h_receipt[key] == value
    assert w_receipt["query_neural_sha256"] == h_receipt["query_neural_sha256"]
    assert w_receipt["support_calib_sha256"] == h_receipt["support_calib_sha256"]
    assert w_receipt["carrier_sha256"] == h_receipt["carrier_sha256"]
    w_add(neural, calib_trials=calib, side_features=t4)
    h_add(neural, calib_trials=calib, side_features=t4)
    assert w_receipt == w_add.support_query_provenance_receipt(
        neural, calib_trials=calib, side_features=t4
    )


def test_parameter_mac_and_state_accounting_are_fixed() -> None:
    decoder = _decoder(model_dim=512, window_size=50, num_heads=64)
    model = HiddenSpaceCarrierStreamingSpint(
        decoder=decoder,
        id_encoder=_IdentityEncoder(50),
        add_site="hidden",
    )
    cost = model.hidden_carrier_cost_receipt(batch_size=1, num_units=64)
    assert cost["trainable_parameter_count"] == 4 * 512
    assert cost["calibration_only_hidden_map_macs"] == 64 * 4 * 512
    assert cost["online_increment"]["additional_linear_macs"] == 0
    assert cost["online_increment"]["hidden_additions"] == 1 * 64 * 512
    assert cost["persistent_additional_state"]["bytes_fp32"] == 64 * 512 * 4
    assert cost["decoder_parameter_count"] == sum(
        parameter.numel() for parameter in decoder.parameters()
    )
    assert cost["teacher_tensors_unmodified"] is True
    receipt = model.hidden_carrier_receipt
    assert receipt["p_of_zero_is_zero"] is True
    assert receipt["bias_parameters"] == 0
    assert receipt["teacher_coupled_activity_identity_readin_preserved"] is True
    assert receipt["new_teacher_required"] is False


def test_guards_fail_closed() -> None:
    with pytest.raises(ValueError, match="add_site"):
        HiddenSpaceCarrierStreamingSpint(
            decoder=_decoder(),
            id_encoder=_IdentityEncoder(6),
            add_site="keys",
        )
    with pytest.raises(ValueError, match="permutation seed"):
        HiddenSpaceCarrierStreamingSpint(
            decoder=_decoder(),
            id_encoder=_IdentityEncoder(6),
            add_site="hidden",
            attachment_mode="shuffled",
        )
    with pytest.raises(ValueError, match="forbids"):
        HiddenSpaceCarrierStreamingSpint(
            decoder=_decoder(),
            id_encoder=_IdentityEncoder(6),
            add_site="hidden",
            attachment_mode="aligned",
            permutation_seed=2,
        )
    with pytest.raises(ValueError, match="positive"):
        ZeroInitializedHiddenCarrierMap(carrier_dim=4, hidden_dim=0)
    with pytest.raises(ValueError, match="shape"):
        HiddenCarrierState(torch.zeros(4, 16))
    model = _h_add(add_site="hidden")
    with pytest.raises(ValueError, match="requires aligned"):
        model(torch.randn(1, 6, 5), calib_trials=torch.randn(1, 2, 6, 5))
    with pytest.raises(ValueError, match="owns"):
        model(
            torch.randn(1, 6, 5),
            calib_trials=torch.randn(1, 2, 6, 5),
            side_features=torch.randn(1, 5, 4),
            decoder_key_features=torch.randn(1, 5, 4),
        )
