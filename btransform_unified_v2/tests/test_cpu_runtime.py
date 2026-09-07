from __future__ import annotations

import torch
import pytest
from btransform_unified_v1 import make_synthetic_bank
from btransform_unified_v2 import RiftDecoder, RiftStreamDecoder
from btransform_unified_v2.cpu_runtime import CpuRiftRuntime


def test_cpu_runtime_matches_reference_stream_and_reorder_reset() -> None:
    torch.manual_seed(71); model=RiftDecoder("m2",context_bins=8,seed=42).eval(); bank=make_synthetic_bank("m2",seed=4)
    x=torch.randn(2,12,96); reference=RiftStreamDecoder(model); actual=CpuRiftRuntime(model,[bank,bank],["a","b"])
    for t in range(8):
        got=actual.advance(x[:,t]); want=reference.stream_step(x[:,t],bank,["a","b"])
        torch.testing.assert_close(got,want,rtol=1e-5,atol=1e-5)
    got=actual.advance(x.flip(0)[:,8],["b","a"]); want=reference.stream_step(x.flip(0)[:,8],bank,["b","a"])
    torch.testing.assert_close(got,want,rtol=1e-5,atol=1e-5)
    actual.reset_rows(["a"]); reference.reset("a")
    got=actual.advance(x.flip(0)[:,9]); want=reference.stream_step(x.flip(0)[:,9],bank,["b","a"])
    torch.testing.assert_close(got,want,rtol=1e-5,atol=1e-5)


def test_cpu_runtime_rejects_empty_bank_and_weight_mutation() -> None:
    model = RiftDecoder("m2", context_bins=8, seed=42).eval()
    bank = make_synthetic_bank("m2", seed=4)
    bank.unit_mask[:] = False
    with pytest.raises(ValueError, match="unit mask became empty"):
        CpuRiftRuntime(model, [bank], ["a"])

    bank = make_synthetic_bank("m2", seed=4)
    runtime = CpuRiftRuntime(model, [bank], ["a"])
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    with pytest.raises(RuntimeError, match="parameters changed"):
        runtime.advance(torch.zeros(1, 96))


def test_cpu_runtime_cached_temporal_matches_state_through_reorder_and_reset() -> None:
    torch.manual_seed(72)
    model = RiftDecoder("m2", context_bins=8, seed=42).eval()
    bank = make_synthetic_bank("m2", seed=5)
    x = torch.randn(2, 12, 96)
    reference = CpuRiftRuntime(model, [bank, bank], ["a", "b"])
    cached = CpuRiftRuntime(model, [bank, bank], ["a", "b"], temporal_backend="cached")
    for t in range(8):
        torch.testing.assert_close(cached.advance(x[:, t]), reference.advance(x[:, t]), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(cached.advance(x.flip(0)[:, 8], ["b", "a"]), reference.advance(x.flip(0)[:, 8], ["b", "a"]), rtol=1e-5, atol=1e-5)
    cached.reset_rows(["a"]); reference.reset_rows(["a"])
    torch.testing.assert_close(cached.advance(x.flip(0)[:, 9]), reference.advance(x.flip(0)[:, 9]), rtol=1e-5, atol=1e-5)


def test_cpu_runtime_valid_mask_matches_streaming_and_does_not_advance_inactive_row() -> None:
    torch.manual_seed(73)
    model=RiftDecoder("m2",context_bins=8,seed=42).eval(); bank=make_synthetic_bank("m2",seed=6)
    x=torch.randn(2,8,96); reference=RiftStreamDecoder(model); runtime=CpuRiftRuntime(model,[bank,bank],["a","b"])
    for t in range(x.shape[1]):
        valid=torch.tensor([True,t not in (2,3,6)])
        want=reference.stream_step(x[:,t],bank,["a","b"],valid_mask=valid)
        got=runtime.advance(x[:,t],valid_mask=valid)
        torch.testing.assert_close(got,want,rtol=1e-5,atol=1e-5)
