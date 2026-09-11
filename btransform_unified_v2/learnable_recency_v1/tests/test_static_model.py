from __future__ import annotations

import inspect

import pytest
import torch

from learnable_recency_v1.config import dataset_config
from learnable_recency_v1.static_model import StaticLearnableRiftDecoder, StaticRiftStreamDecoder
from learnable_recency_v1.wrap import LearnableRiftDecoder


def _model(task: str = "m2", *, seed: int = 23) -> StaticLearnableRiftDecoder:
    # Production learned-slope policy: per-layer parameters and the mainline
    # default ladder, rather than a reduced test-only configuration.
    return StaticLearnableRiftDecoder(
        task,
        dataset_config(task, tier="learned_slope", ladder="default", per_layer=True),
        seed=seed,
    )


@pytest.mark.parametrize("task", ("m1", "m2", "h1"))
def test_static_is_calibration_free_and_preserves_full_retained_initialization(task: str) -> None:
    static = _model(task, seed=29).eval()
    full = LearnableRiftDecoder(
        task,
        dataset_config(task, tier="learned_slope", ladder="default", per_layer=True),
        seed=29,
    ).eval()
    static_params = dict(static.named_parameters())
    full_params = dict(full.named_parameters())
    assert "static_identity" in static_params
    assert torch.count_nonzero(static.static_identity)
    assert not any("e0_proj" in name for name in static_params)
    assert not any("e0_proj" in name for name in static.state_dict())
    retained = set(static_params) & set(full_params)
    assert retained
    for name in retained:
        assert torch.equal(static_params[name], full_params[name]), name
    assert sum(parameter.numel() for parameter in static.parameters()) == (
        sum(parameter.numel() for parameter in full.parameters())
        - full.frontend.e0_proj.weight.numel()
        + static.static_identity.numel()
    )
    assert "bank" not in inspect.signature(static.forward_scores).parameters
    assert "bank" not in inspect.signature(static.frontend_tokens).parameters
    assert torch.equal(static.static_carrier, torch.zeros_like(static.static_carrier))
    if task == "m1":
        assert tuple(static.static_identity.shape) == (64, 16)
        assert tuple(static.static_carrier.shape) == (64, 4)
        assert static.context_bins == 100
        assert sum(parameter.numel() for parameter in static.parameters()) == sum(
            parameter.numel() for parameter in full.parameters()
        ) - 576


@pytest.mark.parametrize("task", ("m1", "m2"))
def test_static_frontend_temporal_gradients_and_explicit_keep_are_final(task: str) -> None:
    model = _model(task).train()
    x = torch.randn(2, 8, model.units, requires_grad=True)
    keep = torch.ones(2, model.units, dtype=torch.bool)
    score = model.forward_scores(x, dropout_keep=keep)
    score.square().mean().backward()
    for gradient in (
        model.static_identity.grad,
        model.frontend.local_conv.conv.weight.grad,
        model.temporal.blocks[0].qkv.weight.grad,
        model.temporal.slope_log.grad,
    ):
        assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
    # Flat heads stay exactly flat under the mainline default policy.
    slopes = model.temporal.effective_slopes()
    assert torch.equal(slopes[:, ~model.temporal._learn_mask], torch.zeros_like(slopes[:, ~model.temporal._learn_mask]))

    unit_mask = torch.ones(model.units, dtype=torch.bool)
    unit_mask[0] = False
    explicit = torch.ones(model.units, dtype=torch.bool)
    explicit[1] = False
    # Explicit keep is caller-owned and must not make a second random .1 draw.
    first = model.frontend_tokens(x.detach(), unit_mask, torch.Generator().manual_seed(3), explicit)
    second = model.frontend_tokens(x.detach(), unit_mask, torch.Generator().manual_seed(999), explicit)
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
    expected_keep = explicit & unit_mask
    torch.testing.assert_close(first, model.frontend_tokens(x.detach(), dropout_keep=expected_keep), atol=0.0, rtol=0.0)
    with pytest.raises(ValueError, match="empty"):
        model.frontend_tokens(x.detach(), unit_mask=torch.zeros(model.units, dtype=torch.bool))
    with pytest.raises(ValueError, match="empty"):
        model.frontend_tokens(x.detach(), dropout_keep=torch.zeros(model.units, dtype=torch.bool))
    with pytest.raises(ValueError, match="unit_mask"):
        model.frontend_tokens(x.detach(), unit_mask=torch.tensor(True))
    with pytest.raises(ValueError, match="dropout_keep"):
        model.frontend_tokens(x.detach(), dropout_keep=torch.tensor(True))


@pytest.mark.parametrize("task,time", (("m1", 105), ("m2", 55), ("h1", 305)))
def test_static_stream_window_parity_reorder_reset_mutation_and_invalidation(task: str, time: int) -> None:
    model = _model(task, seed=31).eval()
    # Prove packing preserves independent learned per-layer increment caches.
    with torch.no_grad():
        model.temporal.slope_log[1].add_(0.17)
    assert not torch.equal(model.temporal.slope_log[0], model.temporal.slope_log[1])
    x = torch.randn(2, time, model.units)
    valid = torch.ones(2, time, dtype=torch.bool)
    valid[0, :3] = False
    offline = model.forward_scores(x, input_valid_mask=valid)
    stream = StaticRiftStreamDecoder(model)
    stepped = torch.empty_like(offline)
    ids = ("first", "second")
    for index in range(time):
        order = (0, 1) if index % 2 else (1, 0)
        staging = x[list(order), index].clone()
        output = stream.stream_step(staging, [ids[row] for row in order], valid_mask=valid[list(order), index])
        # The adapter must retain the consumed bin, not this caller-owned
        # staging tensor which is routinely reused for the next read.
        staging.zero_()
        for output_row, source_row in enumerate(order):
            stepped[source_row, index] = output[output_row]
    torch.testing.assert_close(stepped, offline, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(stream.predict("first"), offline[0, -1], atol=1e-5, rtol=1e-5)
    stream.reset("first")
    with pytest.raises(KeyError):
        stream.predict("first")
    stream.stream_step(x[:1, 0].clone(), ["first"], valid_mask=torch.tensor([True]))
    with torch.no_grad():
        model.static_identity.add_(0.001)
    with pytest.raises(RuntimeError, match="weights changed"):
        stream.stream_step(x[:1, 1].clone(), ["first"], valid_mask=torch.tensor([True]))
    stream.reset("first")
    stream.stream_step(x[:1, 0].clone(), ["first"], valid_mask=torch.tensor([True]))
    with pytest.raises(RuntimeError, match="unit mask"):
        stream.stream_step(
            x[:1, 1].clone(), ["first"], unit_mask=torch.zeros(1, model.units, dtype=torch.bool), valid_mask=torch.tensor([True])
        )
