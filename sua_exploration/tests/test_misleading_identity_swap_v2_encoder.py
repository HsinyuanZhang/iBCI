from __future__ import annotations

import torch

from mc_maze import misleading_identity_swap_v2_core as core
from mc_maze.misleading_identity_swap import apply_activity_identity_swap
from src.models.components.misleading_identity_swap_v2_encoder import (
    MisleadingIdentitySwapV2Encoder,
)
from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder


def _authority() -> core.VerifiedMatchingAuthority:
    descriptor = torch.tensor([
        [0.0, 0.0, 0.0, 1.0],
        [0.1, 0.0, 0.1, 1.1],
        [2.0, 2.0, 2.8, 3.0],
        [2.1, 2.0, 2.9, 3.1],
        [4.0, 4.0, 5.6, 5.0],
        [4.1, 4.0, 5.7, 5.1],
        [6.0, 6.0, 8.5, 7.0],
        [6.1, 6.0, 8.6, 7.1],
    ], dtype=torch.float64)
    return core.VerifiedMatchingAuthority.from_payload(
        core.build_matching_authority({"source_session": descriptor})
    )


def _parent_and_wrapper() -> tuple[SideFeatureEarlyPoolEncoder, MisleadingIdentitySwapV2Encoder]:
    torch.manual_seed(19)
    parent = SideFeatureEarlyPoolEncoder(
        trial_length=6,
        window_size=5,
        hidden_dim=7,
        side_dim=4,
        num_post_layers=2,
    )
    # Make visible carrier contribution nonzero so a carrier-row swap would be observable.
    first = next(layer for layer in parent.post_pool if isinstance(layer, torch.nn.Linear))
    with torch.no_grad():
        first.weight[:, parent.hidden_dim :].copy_(
            torch.arange(first.out_features * 4, dtype=first.weight.dtype).reshape(first.out_features, 4) / 31.0
        )
    wrapper = MisleadingIdentitySwapV2Encoder.from_parent(parent, authority=_authority())
    parent.eval()
    wrapper.eval()
    return parent, wrapper


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(23)
    calib = torch.randn(2, 3, 6, 8)
    # Unique per-row carrier makes any accidental row permutation immediately visible.
    side = torch.arange(2 * 8 * 4, dtype=torch.float32).reshape(2, 8, 4) / 17.0
    return calib, side


def test_clean_wrapper_is_bitwise_exact_parent_null() -> None:
    parent, wrapper = _parent_and_wrapper()
    calib, side = _batch()
    wrapper.configure_runtime(
        session="source_session", lightning_current_epoch=4, phase="eval_clean"
    )
    with torch.no_grad():
        expected = parent.forward_batch(calib, side_features=side)
        observed = wrapper.forward_batch(calib, side_features=side)
    assert torch.equal(observed, expected)
    assert wrapper.last_swap_trace is not None
    assert wrapper.last_swap_trace["swap_applied"] is False


def test_swap_occurs_after_mean_feat_and_before_unmoved_carrier_concat() -> None:
    _parent, wrapper = _parent_and_wrapper()
    calib, side = _batch()
    side_before = side.clone()
    state = wrapper.reset_stream(2, 8, calib.device, calib.dtype)
    state["side_features"] = side
    for trial in range(calib.shape[1]):
        state = wrapper.push_trial(state, calib[:, trial])
    mean = state["sum_feat"] / state["trial_count"]
    swap = _authority().swap_for("source_session", 4)
    expected_input = torch.cat(
        [apply_activity_identity_swap(mean, swap), side], dim=-1
    )
    expected = wrapper.post_pool(expected_input)

    wrapper.configure_runtime(
        session="source_session",
        lightning_current_epoch=4,
        phase="eval_swapped_diagnostic",
    )
    with torch.no_grad():
        observed = wrapper.forward_batch(calib, side_features=side)
    assert torch.equal(observed, expected)
    assert torch.equal(side, side_before)
    trace = wrapper.last_swap_trace
    assert trace is not None and trace["swap_applied"] is True
    assert trace["mean_feat_before_sha256"] != trace["mean_feat_after_sha256"]
    assert trace["visible_side_before_sha256"] == trace["visible_side_after_sha256"]
    assert trace["hidden_descriptor_forwarded_to_post_pool"] is False


def test_z4_visible_carrier_stays_exact_zero_while_hidden_mapping_is_live() -> None:
    _parent, wrapper = _parent_and_wrapper()
    calib, _side = _batch()
    z4 = torch.zeros(2, 8, 4)
    wrapper.configure_runtime(
        session="source_session", lightning_current_epoch=7, phase="train_swap"
    )
    observed = wrapper.forward_batch(calib, side_features=z4)
    assert torch.isfinite(observed).all()
    assert torch.count_nonzero(z4).item() == 0
    trace = wrapper.last_swap_trace
    assert trace is not None
    assert trace["visible_side_before_sha256"] == trace["visible_side_after_sha256"]
    assert not any("descriptor" in name for name, _ in wrapper.named_parameters())
    assert not any("descriptor" in name for name, _ in wrapper.named_buffers())


def test_wrong_session_epoch_or_unit_count_fails_closed() -> None:
    _parent, wrapper = _parent_and_wrapper()
    calib, side = _batch()
    wrapper.configure_runtime(
        session="missing_session", lightning_current_epoch=4, phase="train_swap"
    )
    try:
        wrapper.forward_batch(calib, side_features=side)
    except core.SwapV2ContractError:
        pass
    else:
        raise AssertionError("missing authority session did not fail closed")

    wrapper.configure_runtime(
        session="source_session", lightning_current_epoch=4, phase="train_swap"
    )
    try:
        wrapper.forward_batch(calib[:, :, :, :6], side_features=side[:, :6])
    except ValueError as exc:
        assert "unit count drift" in str(exc)
    else:
        raise AssertionError("authority/unit count drift did not fail closed")

