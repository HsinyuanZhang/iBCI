"""Exact CPU equivalence and invalidation checks for cached M2 QueryAge."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tfpd_exploration.src.family_runtime_v1.m2_family_causal import M2RuntimeBank
from tfpd_exploration.src.family_runtime_v1.m2_family_queryage import M2FamilyQueryAgeRuntime
from tfpd_exploration.src.family_runtime_v1.m2_family_queryage_cached import M2FamilyQueryAgeCachedRuntime
from tfpd_exploration.src.m2_queryage_family_v1.model import make_paired_queryage_decoders


def _bank(batch: int, *, heterogeneous: bool):
    gen = torch.Generator().manual_seed(801 + batch + int(heterogeneous))
    shape = (batch, 96) if heterogeneous else (96,)
    return M2RuntimeBank(
        torch.randn((*shape, 50), generator=gen) * .1,
        torch.randn((*shape, 4), generator=gen) * .1,
        torch.rand(shape, generator=gen) > .1,
    )


@pytest.mark.parametrize("arm,batch,heterogeneous", [(0, 1, False), (1, 7, True)])
def test_cached_matches_uncached_and_direct_for_110_startup_rollover_steps(arm, batch, heterogeneous):
    torch.set_num_threads(1)
    model = make_paired_queryage_decoders(42)[arm].eval()
    if arm:
        with torch.no_grad(): model.frontend.attn.routing.g.fill_(.2)
    bank = _bank(batch, heterogeneous=heterogeneous)
    plain, cached = M2FamilyQueryAgeRuntime(model, bank, batch_size=batch), M2FamilyQueryAgeCachedRuntime(model, bank, batch_size=batch)
    raw = torch.zeros(batch, 50, 96); values = np.random.default_rng(99 + batch).normal(0, .2, (110, batch, 96)).astype(np.float32)
    for step, value in enumerate(values):
        raw = torch.cat((raw[:, 1:], torch.from_numpy(value)[:, None]), 1)
        got, reference = cached.predict(value), plain.predict(value)
        np.testing.assert_allclose(got, reference, atol=1e-5, rtol=1e-5)
        if step in (0, 4, 49, 50, 109):
            direct = (model.forward_last(raw, bank, bank.unit_mask) / 5).detach().numpy()
            np.testing.assert_allclose(got, direct, atol=1e-5, rtol=1e-5)
    assert cached.memory.last_rebuild_reason is None
    assert cached.state_bytes()["query_memory_cache"] == 460_800 * batch


def test_cache_projects_only_five_changed_tokens_in_steady_state():
    torch.set_num_threads(1)
    model = make_paired_queryage_decoders(42)[0].eval(); bank = _bank(1, heterogeneous=False)
    calls, original = [], model.temporal.project_memory
    def spy(z): calls.append(z.shape[1]); return original(z)
    model.temporal.project_memory = spy
    runtime = M2FamilyQueryAgeCachedRuntime(model, bank)
    assert calls == [50]
    for _ in range(5): runtime.predict(np.zeros((1, 96), dtype=np.float32))
    assert calls == [50, 5, 5, 5, 5, 5]


def test_active_lane_padding_lifecycle_mutation_and_cache_tamper_refusal():
    torch.set_num_threads(1)
    model = make_paired_queryage_decoders(42)[1].eval(); bank = _bank(7, heterogeneous=True)
    runtime = M2FamilyQueryAgeCachedRuntime(model, bank, batch_size=7)
    partial = M2RuntimeBank(bank.E0[:1].clone(), bank.T[:1].clone(), bank.unit_mask[:1].clone())
    runtime.reset(partial, batch=1)
    out = runtime.predict(np.ones((1, 96), dtype=np.float32)); assert out.shape == (7, 2) and np.all(out[1:] == 0)
    before = runtime.raw.clone()
    with pytest.raises(ValueError): runtime.predict(np.zeros((2, 96), dtype=np.float32))
    assert torch.equal(before, runtime.raw)
    runtime.on_done(np.array([True]))
    with torch.no_grad(): partial.E0.add_(.01)
    runtime.predict(np.zeros((1, 96), dtype=np.float32))
    assert runtime.memory.last_rebuild_reason is None  # reset, then legitimate advance
    runtime.memory.memories[0][0].add_(.01)
    with pytest.raises(RuntimeError, match="cached QueryAge"):
        runtime.on_done(np.array([True]))


def test_temporal_replacement_rebinds_cache_and_reset_restores_integrity():
    torch.set_num_threads(1)
    model = make_paired_queryage_decoders(42)[0].eval(); bank = _bank(1, heterogeneous=False)
    runtime = M2FamilyQueryAgeCachedRuntime(model, bank)
    old = runtime.memory.model
    replacement = make_paired_queryage_decoders(42)[0].temporal.eval()
    model.temporal = replacement
    runtime.predict(np.zeros((1, 96), dtype=np.float32))
    assert runtime.memory.model is replacement and runtime.memory.model is not old
    runtime.memory.z.add_(.01)
    with pytest.raises(RuntimeError, match="cached QueryAge"):
        runtime.on_done(np.array([True]))
    runtime.reset(bank)
    runtime.predict(np.zeros((1, 96), dtype=np.float32))
