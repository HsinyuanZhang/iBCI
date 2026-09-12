"""CPU contracts for the neural-only SPINT-style identity branch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
STREAMING = ROOT / "streaming_calibration_exp"
if str(STREAMING) not in sys.path:
    sys.path.insert(0, str(STREAMING))
from src.models.components.streaming_encoders import EarlyPoolEncoder

from learnable_recency_v1.activity_model import (
    ActivityIdentityEncoder,
    ActivityLearnableRiftDecoder,
    DEFAULT_IDENTITY_HIDDEN,
    DEFAULT_SUPPORT_BINS,
)
from learnable_recency_v1.config import dataset_config
from learnable_recency_v1.wrap import LearnableRiftDecoder


def _model(task: str, seed: int = 17) -> ActivityLearnableRiftDecoder:
    return ActivityLearnableRiftDecoder(
        task,
        dataset_config(task, tier="learned_slope", ladder="default", context_bins=5),
        context_bins=5,
        seed=seed,
    )


@pytest.mark.parametrize("task", ("m1", "m2", "h1"))
def test_activity_decoder_preserves_p16_rift_initialization_and_joint_gradients(task: str) -> None:
    model = _model(task).train()
    reference = LearnableRiftDecoder(task, dataset_config(task, tier="learned_slope", ladder="default", context_bins=5), context_bins=5, seed=17)
    activity = torch.randn(2, DEFAULT_SUPPORT_BINS[task], model.units)
    x = torch.randn(2, 6, model.units)
    keep = torch.ones(model.units, dtype=torch.bool)
    score = model(x, activity, dropout_keep=keep)
    score.square().mean().backward()
    for parameter in (model.identity_encoder.pre_pool[0].weight, model.identity_encoder.post_pool[-1].weight, model.frontend.e0_proj.weight, model.readout[0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad)
    shared = set(dict(model.named_parameters())) & set(dict(reference.named_parameters()))
    assert shared
    for name in shared:
        assert torch.equal(dict(model.named_parameters())[name], dict(reference.named_parameters())[name]), name
    assert torch.equal(model.zero_carrier, torch.zeros_like(model.zero_carrier))


@pytest.mark.parametrize("task", ("m1", "m2", "h1"))
def test_eval_trial_and_joint_unit_permutation_invariance(task: str) -> None:
    model = _model(task, seed=23).eval()
    activity = torch.randn(3, DEFAULT_SUPPORT_BINS[task], model.units)
    x = torch.randn(2, 6, model.units)
    keep = torch.ones(model.units, dtype=torch.bool)
    with torch.no_grad():
        expected = model(x, activity, dropout_keep=keep)
        # Floating reductions have a different addition order after a trial
        # permutation, so invariance is numerical rather than byte identity.
        torch.testing.assert_close(expected, model(x, activity[torch.tensor([2, 0, 1])], dropout_keep=keep), atol=1e-6, rtol=1e-6)
        permutation = torch.randperm(model.units)
        actual = model(x[:, :, permutation], activity[:, :, permutation], dropout_keep=keep[permutation])
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("task", ("m1", "m2", "h1"))
def test_trial_and_unit_masks_calibration_and_state_roundtrip(task: str) -> None:
    model = _model(task, seed=31).eval()
    activity = torch.randn(3, DEFAULT_SUPPORT_BINS[task], model.units)
    x = torch.randn(1, 6, model.units)
    trial_mask = torch.tensor([True, False, True])
    unit_mask = torch.ones(model.units, dtype=torch.bool)
    unit_mask[0] = False
    activity_changed = activity.clone()
    activity_changed[1].fill_(1234.0)
    x_changed = x.clone()
    x_changed[:, :, 0].fill_(-999.0)
    with torch.no_grad():
        identity = model.calibrate(activity, trial_mask)
        assert not identity.requires_grad
        torch.testing.assert_close(identity, model.calibrate(activity_changed, trial_mask), atol=0.0, rtol=0.0)
        left = model(x, identity=identity, unit_mask=unit_mask)
        right = model(x_changed, identity=identity, unit_mask=unit_mask)
        torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
    model.train()
    with pytest.raises(RuntimeError, match="eval-only"):
        model.calibrate(activity)
    with pytest.raises(ValueError, match="exactly one"):
        model(x, None)
    with pytest.raises(ValueError, match="exactly one"):
        model(x, activity, identity=identity)
    restored = _model(task, seed=999).eval()
    restored.load_state_dict(model.eval().state_dict())
    with torch.no_grad():
        torch.testing.assert_close(restored(x, activity, trial_mask=trial_mask), model(x, activity, trial_mask=trial_mask), atol=0.0, rtol=0.0)


def test_identity_encoder_requires_nonlinear_trial_pooling_and_valid_mask() -> None:
    encoder = ActivityIdentityEncoder(4, 3, hidden_dim=5, seed=5).eval()
    activity = torch.randn(3, 4, 2)
    with torch.no_grad():
        expected = encoder(activity, torch.tensor([True, False, True]))
        changed = activity.clone(); changed[1].fill_(999.0)
        torch.testing.assert_close(expected, encoder(changed, torch.tensor([True, False, True])), atol=0.0, rtol=0.0)
    with pytest.raises(ValueError, match="selects no"):
        encoder(activity, torch.zeros(3, dtype=torch.bool))


def test_task_defaults_are_existing_activity_early_pool_geometries() -> None:
    assert DEFAULT_SUPPORT_BINS == {"m1": 1024, "m2": 100, "h1": 1024}
    assert DEFAULT_IDENTITY_HIDDEN == {"m1": 64, "m2": 64, "h1": 32}
    for task in ("m1", "m2", "h1"):
        model = _model(task)
        encoder = model.identity_encoder
        assert encoder.input_bins == DEFAULT_SUPPORT_BINS[task]
        assert encoder.hidden_dim == DEFAULT_IDENTITY_HIDDEN[task]
        assert sum(isinstance(layer, torch.nn.Linear) for layer in encoder.pre_pool) == 1
        assert sum(isinstance(layer, torch.nn.Linear) for layer in encoder.post_pool) == 3
        assert list(dict(encoder.named_parameters())) == [
            "pre_pool.0.weight", "pre_pool.0.bias",
            "post_pool.0.weight", "post_pool.0.bias",
            "post_pool.2.weight", "post_pool.2.bias",
            "post_pool.4.weight", "post_pool.4.bias",
        ]


def test_activity_encoder_is_numerically_equivalent_to_b3_activity_trunk() -> None:
    activity = torch.randn(3, 7, 5)
    encoder = ActivityIdentityEncoder(7, 11, hidden_dim=13, seed=5).eval()
    reference = EarlyPoolEncoder(trial_length=7, window_size=11, hidden_dim=13, num_post_layers=3).eval()
    reference.pre_pool.load_state_dict(encoder.pre_pool.state_dict())
    reference.post_pool.load_state_dict(encoder.post_pool.state_dict())
    with torch.no_grad():
        expected = reference.forward_batch(activity.unsqueeze(0))[0]
        torch.testing.assert_close(encoder(activity), expected, atol=0.0, rtol=0.0)


def test_zero_carrier_state_load_rejects_nonzero_buffer() -> None:
    model = _model("m2").eval()
    state = model.state_dict()
    state["zero_carrier"] = torch.ones_like(state["zero_carrier"])
    with pytest.raises(RuntimeError, match="literal zero activity carrier"):
        _model("m2", seed=99).load_state_dict(state)
