"""Inference-only acceptance checks for the FULL all-zero-recency control."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from btransform_unified_v1.bank import TaskBank, make_synthetic_bank
from btransform_unified_v2.cpu_temporal import CpuRiftTemporalRuntime
from btransform_unified_v2.model import RiftDecoder
from learnable_recency_v1.config import dataset_config
from learnable_recency_v1.flat_control import flat_config
from learnable_recency_v1.wrap import LearnableRiftDecoder, LearnableRiftStreamDecoder, new_parameter_names


TASKS = ("m1", "m2", "h1")


def _learned(task: str) -> LearnableRiftDecoder:
    return LearnableRiftDecoder(
        task,
        dataset_config(task, tier="learned_slope", ladder="default", per_layer=True),
        seed=42,
    ).eval()


def _flat(task: str) -> LearnableRiftDecoder:
    return LearnableRiftDecoder(task, flat_config(task), seed=42).eval()


def _bank(task: str, seed: int) -> TaskBank:
    if task != "h1":
        return make_synthetic_bank(task, n_windows=2, seed=seed)
    generator = np.random.default_rng(seed)
    return TaskBank(
        "synthetic-h1",
        generator.normal(size=(176, 700)).astype(np.float32),
        generator.normal(size=(176, 4)).astype(np.float32),
        np.ones(176, dtype=bool),
        np.zeros((1, 300, 176), dtype=np.float32),
        np.zeros((1, 7), dtype=np.float32),
        np.array([0], dtype=np.int64),
        {
            "shape": (176, 700),
            "trial_count": 1,
            "estimator": "synthetic",
            "array_sha256": "synthetic",
            "carrier_sha256": "synthetic",
            "budget": 1,
        },
    )


@pytest.mark.parametrize("task", TASKS)
@torch.no_grad()
def test_full_flat_keeps_full_interface_and_shared_initialization(task: str) -> None:
    learned = _learned(task)
    flat = _flat(task)
    learned_params = dict(learned.named_parameters())
    flat_params = dict(flat.named_parameters())
    shared = sorted(set(learned_params) & set(flat_params))

    assert shared
    assert all(torch.equal(learned_params[name], flat_params[name]) for name in shared)
    assert set(learned_params) - set(flat_params) == {"temporal.slope_log"}
    assert new_parameter_names(flat) == []
    assert flat.frontend.e0_proj is not None
    assert "bank" in inspect.signature(flat.forward_scores).parameters
    assert "bank" in inspect.signature(flat.frontend_tokens).parameters

    bank = _bank(task, seed=17)
    x = torch.randn(1, 7, flat.units)
    out = flat.forward_scores(x, bank)
    assert tuple(out.shape) == (1, 7, flat.out_dim)
    assert bank.E0.shape == (flat.units, flat.geometry["e0_dim"])
    assert bank.carrier.shape == (flat.units, 4)


@pytest.mark.parametrize("task", TASKS)
@torch.no_grad()
def test_full_flat_is_zero_bias_and_matches_explicit_flat_over_context_and_stream(task: str) -> None:
    flat = _flat(task)
    reference = RiftDecoder(task, context_bins=flat.context_bins, bias_mode="flat", seed=42, proj_dim=16).eval()
    bank = _bank(task, seed=31)
    time = flat.context_bins + 5
    x = torch.randn(1, time, flat.units)
    valid = torch.ones(1, time, dtype=torch.bool)
    valid[:, :3] = False

    assert torch.equal(flat.temporal.recency_slopes, torch.zeros(8, dtype=torch.float32))
    assert tuple(flat.temporal.recency_slopes.shape) == (8,)
    assert flat.temporal.config.layers == 4

    z = flat.frontend_tokens(torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x)), bank)
    flat.temporal.set_attention_backend("local")
    local_hidden = flat.temporal(z, valid)
    flat.temporal.set_attention_backend("dense")
    dense_hidden = flat.temporal(z, valid)
    torch.testing.assert_close(local_hidden, dense_hidden, atol=1e-5, rtol=1e-5)
    assert torch.equal(local_hidden[:, :3], torch.zeros_like(local_hidden[:, :3]))

    changed = x.clone()
    changed[:, time // 2 :] = torch.randn_like(changed[:, time // 2 :])
    changed_z = flat.frontend_tokens(torch.where(valid.unsqueeze(-1), changed, torch.zeros_like(changed)), bank)
    flat.temporal.set_attention_backend("local")
    changed_hidden = flat.temporal(changed_z, valid)
    torch.testing.assert_close(local_hidden[:, : time // 2], changed_hidden[:, : time // 2], atol=3e-6, rtol=3e-6)

    flat.temporal.set_attention_backend("dense")
    scores = flat.forward_scores(x, bank, input_valid_mask=valid)
    reference_scores = reference.forward_scores(x, bank, input_valid_mask=valid)
    torch.testing.assert_close(scores, reference_scores, atol=1e-5, rtol=1e-5)

    flat.temporal.set_attention_backend("local")
    stream = LearnableRiftStreamDecoder(flat)
    stepped = torch.stack(
        [stream.stream_step(x[:, index], bank, ["stream"], valid_mask=valid[:, index]) for index in range(time)],
        dim=1,
    )
    torch.testing.assert_close(stepped, flat.forward_scores(x, bank, input_valid_mask=valid), atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_m2_full_flat_uses_unchanged_cpu_runtime_with_zero_slopes() -> None:
    flat = _flat("m2")
    bank = _bank("m2", seed=53)
    x = torch.randn(1, 55, flat.units)
    valid = torch.ones(1, 55, dtype=torch.bool)
    valid[:, :4] = False
    z = flat.frontend_tokens(torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x)), bank)
    runtime = CpuRiftTemporalRuntime(flat.temporal, batch_size=1)
    state = flat.temporal.init_state(1, "cpu", torch.float32)

    for index in range(z.shape[1]):
        got = runtime.step(z[:, index], valid[:, index])
        want, state = flat.temporal.step(z[:, index], state, valid[:, index])
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
