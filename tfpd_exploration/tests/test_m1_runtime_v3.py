"""Parity and lifecycle tests for the isolated heterogeneous M1 runtime."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tfpd_exploration.src.m1_runtime_v3 import BankBatch, HeterogeneousCurrentQueryStream


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "submissions/evalai_m1_runtime_v3_selected_t_v1/m1_trf_falcon_decoder.py"
PAYLOAD = PACKAGE.parent / "artifacts/m1_optimized_v2_t_ema_e6.pkl"
if str(PACKAGE.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE.parent))


def _frozen_package():
    name = "_m1_runtime_v3_frozen_package"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PACKAGE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(batch: int = 4):
    package = _frozen_package()
    payload = package.load_payload(PAYLOAD)
    model = package.build_decoder("current_query").eval()
    model.load_state_dict(
        {name: torch.as_tensor(value, dtype=torch.float32) for name, value in payload["state_dict"].items()},
        strict=True,
    )
    rows = list(payload["bank_by_dataset_tag"].items())[:batch]
    bank = BankBatch(
        E0=torch.stack([torch.as_tensor(row["E0"], dtype=torch.float32) for _, row in rows]),
        T=torch.stack([torch.as_tensor(row["T"], dtype=torch.float32) for _, row in rows]),
        unit_mask=torch.stack([torch.as_tensor(row["unit_mask"], dtype=torch.bool) for _, row in rows]),
        session_ids=tuple(tag for tag, _ in rows),
        unit_ids=tuple(tuple(range(64)) for _ in rows),
    )
    return package, model, bank


def _reference_last(package, model, runtime, bank):
    frozen_bank = package.SessionBank(bank.E0, bank.T, bank.unit_mask)
    return model.forward_last(runtime.raw, frozen_bank, bank.unit_mask)


def test_b4_heterogeneous_exact_startup_boundary_and_long_replay():
    torch.set_num_threads(1)
    package, model, bank = _fixture()
    runtime = HeterogeneousCurrentQueryStream(model, bank)
    inputs = torch.randn(384, 4, 64, generator=torch.Generator().manual_seed(8107))
    # 0 is represented by the zero raw state at construction.  Explicitly
    # exercise starts 1/3/4/5 and W-1/W after their observations.
    maximum_error = 0.0
    initial = runtime.current_prediction()
    initial_reference = _reference_last(package, model, runtime, bank)
    maximum_error = max(maximum_error, float((initial - initial_reference).abs().max()))
    assert bool(torch.all((initial - initial_reference).abs() <= 1e-5 + 1e-5 * initial_reference.abs()))
    for step, value in enumerate(inputs, start=1):
        actual = torch.from_numpy(runtime.predict(value.numpy()))
        expected = _reference_last(package, model, runtime, bank)
        error = (actual - expected).abs()
        maximum_error = max(maximum_error, float(error.max()))
        assert bool(torch.all(error <= 1e-5 + 1e-5 * expected.abs()))
    # A record of the observed numerical margin makes accidental tolerance
    # relaxation visible in pytest failure output without changing predictions.
    assert maximum_error <= 1e-5


def test_inactive_rows_do_not_receive_padding_observations_and_on_done_is_a_noop():
    torch.set_num_threads(1)
    _, model, bank = _fixture()
    runtime = HeterogeneousCurrentQueryStream(model, bank)
    two_bank = BankBatch(bank.E0[:2], bank.T[:2], bank.unit_mask[:2], bank.session_ids[:2], bank.unit_ids[:2])
    two_lanes = HeterogeneousCurrentQueryStream(model, two_bank)
    first = torch.randn(4, 64, generator=torch.Generator().manual_seed(1))
    runtime.observe(first)
    two_lanes.observe(first[:2])
    before = runtime.raw[2:].clone()
    out = torch.from_numpy(runtime.predict(torch.ones(2, 64).numpy(), active=2))
    expected = torch.from_numpy(two_lanes.predict(torch.ones(2, 64).numpy()))
    assert out.shape == (2, 16)
    assert bool(torch.all((out - expected).abs() <= 1e-5 + 1e-5 * expected.abs()))
    assert torch.equal(runtime.raw[2:], before)
    raw = runtime.raw.clone()
    assert runtime.on_done([True, False, True, False]) is None
    assert torch.equal(runtime.raw, raw)


def test_row_and_unit_permutations_preserve_independent_lanes():
    torch.set_num_threads(1)
    package, model, bank = _fixture()
    canonical = HeterogeneousCurrentQueryStream(model, bank)
    row_order = torch.tensor([2, 0, 3, 1])
    unit_order = torch.tensor(list(reversed(range(64))))
    permuted = BankBatch(
        bank.E0[row_order][:, unit_order], bank.T[row_order][:, unit_order], bank.unit_mask[row_order][:, unit_order],
        tuple(bank.session_ids[i] for i in row_order.tolist()),
        tuple(tuple(range(63, -1, -1)) for _ in range(4)),
    )
    reordered = HeterogeneousCurrentQueryStream(model, permuted)
    values = torch.randn(7, 4, 64, generator=torch.Generator().manual_seed(900))
    for index, value in enumerate(values):
        expected = torch.from_numpy(canonical.predict(value.numpy()))
        got = torch.from_numpy(reordered.predict(value[row_order][:, unit_order].numpy()))
        ref = expected[row_order]
        assert bool(torch.all((got - ref).abs() <= 1e-5 + 1e-5 * ref.abs()))


def test_mask_roster_and_lifecycle_mutation_fail_closed_until_refresh():
    torch.set_num_threads(1)
    _, model, bank = _fixture()
    runtime = HeterogeneousCurrentQueryStream(model, bank)
    with pytest.raises(ValueError, match="unique"):
        BankBatch(bank.E0, bank.T, bank.unit_mask, bank.session_ids, ((0,) * 64,) * 4).validate(4, 64)
    with torch.no_grad():
        bank.unit_mask[0, 0].logical_not_()
    with pytest.raises(RuntimeError, match="fail closed"):
        runtime.current_prediction()
    runtime.refresh_state()
    assert runtime.current_prediction().shape == (4, 16)
    assert runtime.state_bytes == runtime.rolling_state_bytes + runtime.static_cache_bytes
    assert runtime.static_cache_bytes > 0
    # The same tensor storage is not a safe bank replacement when its explicit
    # session/roster authority changes.
    runtime.bank = BankBatch(bank.E0, bank.T, bank.unit_mask, tuple("replacement" for _ in range(4)), bank.unit_ids)
    with pytest.raises(RuntimeError, match="fail closed"):
        runtime.current_prediction()
    runtime.refresh_state()
    with torch.no_grad():
        next(model.parameters()).add_(0.001)
    with pytest.raises(RuntimeError, match="fail closed"):
        runtime.current_prediction()
    runtime.refresh_state()
    model.double()
    with pytest.raises(RuntimeError, match="fail closed"):
        runtime.current_prediction()
    with pytest.raises(ValueError, match="dtype"):
        runtime.refresh_state()


def test_public_api_requires_direct_float32_contiguous_input():
    torch.set_num_threads(1)
    _, model, bank = _fixture()
    runtime = HeterogeneousCurrentQueryStream(model, bank)
    good = torch.zeros(4, 64).numpy()
    assert runtime.predict(good).dtype.name == "float32"
    with pytest.raises(ValueError, match="C-contiguous"):
        runtime.predict(torch.zeros(4, 128).numpy()[:, ::2])
    with pytest.raises(ValueError, match="float32"):
        runtime.predict(good.astype("float64"))


def _official_like_uint8_batch(steps: int = 5, batch: int = 4):
    """Match Falcon M1: bin_units uint8 then pad_sequence(..., batch_first=False).numpy()[t]."""
    counts = np.random.default_rng(581980).poisson(0.3, (steps, batch, 64)).astype(np.uint8)
    stacked = np.ascontiguousarray(counts)
    assert stacked.dtype == np.uint8 and stacked.flags.c_contiguous
    return stacked


def test_falcon_adapter_accepts_official_uint8_float64_and_noncontiguous():
    from falcon_challenge.config import FalconConfig, FalconTask

    torch.set_num_threads(1)
    package = _frozen_package()
    fixture = np.load(PACKAGE.parent / "fixtures/source_b4_public_parity.npz", allow_pickle=False)
    stems = [str(x) for x in fixture["stems"]]
    cfg = FalconConfig(task=FalconTask.m1)
    official = package.M1TemporalFalconDecoder(cfg, str(PAYLOAD), batch_size=4)
    reference = package.M1TemporalFalconDecoder(cfg, str(PAYLOAD), batch_size=4)
    stacked = _official_like_uint8_batch()
    official.reset(stems)
    reference.reset(stems)
    for row in stacked:
        got = official.predict(row)
        ref = reference.predict(np.ascontiguousarray(row.astype(np.float32)))
        assert got.shape == (4, 16) and got.dtype == np.float32 and got.flags.owndata
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)
    official.reset(stems)
    reference.reset(stems)
    for row in stacked.astype(np.float64):
        got = official.predict(row)
        ref = reference.predict(np.ascontiguousarray(row.astype(np.float32)))
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)
    wide = np.zeros((4, 128), dtype=np.uint8)
    wide[:, 0::2] = stacked[0]
    sliced = wide[:, 0::2]
    assert sliced.dtype == np.uint8 and not sliced.flags.c_contiguous
    official.reset(stems)
    reference.reset(stems)
    got = official.predict(sliced)
    ref = reference.predict(np.ascontiguousarray(sliced.astype(np.float32)))
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)
    before = official._stream.raw.clone()
    bad = np.full((4, 64), np.nan, dtype=np.float32)
    with pytest.raises(ValueError):
        official.predict(bad)
    assert torch.equal(before, official._stream.raw)
