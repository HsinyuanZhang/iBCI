"""CPU-only oracle and lifecycle tests for the isolated M1 ROUTE stream."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tfpd_exploration.src.family_runtime_v1.m1_route_lifted import RouteLiftedFiveTokenCurrentQueryStream
from tfpd_exploration.src.m1_runtime_v3 import BankBatch, HeterogeneousCurrentQueryStream
from tfpd_exploration.src.m1_optimized_v2.model import build
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import M1Bank

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "submissions/evalai_m1_runtime_v3_selected_t_v1/m1_trf_falcon_decoder.py"
PAYLOAD = PACKAGE.parent / "artifacts/m1_optimized_v2_t_ema_e6.pkl"
ROUTE_E1 = ROOT / "results/decoder_validation_v2/20260905_190000/m1/family_v1/p1_queryage16_pair_chron80_v2/route/epoch_001.pt"
if str(PACKAGE.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE.parent))


def _package():
    name = "_route_runtime_payload"
    if name not in sys.modules or not hasattr(sys.modules[name], "load_payload"):
        sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location(name, PACKAGE); assert spec and spec.loader
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return sys.modules[name]


def _bank(batch=4):
    p = _package(); payload = p.load_payload(PAYLOAD)
    rows = list(payload["bank_by_dataset_tag"].items())[:batch]
    return BankBatch(torch.stack([torch.as_tensor(v["E0"], dtype=torch.float32) for _,v in rows]), torch.stack([torch.as_tensor(v["T"], dtype=torch.float32) for _,v in rows]), torch.stack([torch.as_tensor(v["unit_mask"], dtype=torch.bool) for _,v in rows]), tuple(k for k,_ in rows), tuple(tuple(range(64)) for _ in rows))


def _fresh_pair():
    flat, route = build("flat").eval(), build("route").eval()
    route_state = route.state_dict()
    for name, value in flat.state_dict().items():
        assert name in route_state
        torch.testing.assert_close(value, route_state[name], rtol=0, atol=0)
    return flat, route


def _checkpoint_route():
    if not ROUTE_E1.exists(): pytest.skip("P1 ROUTE checkpoint has not been produced yet")
    model = build("route").eval()
    model.load_state_dict(torch.load(ROUTE_E1, map_location="cpu", weights_only=False)["model"], strict=True)
    assert float(model.frontend.attn.g.abs().max()) > 0.0
    return model


def _oracle(model, stream):
    b = M1Bank(stream.bank.E0, stream.bank.T, stream.bank.unit_mask)
    return model.forward_last(stream.raw, b, stream.bank.unit_mask)


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("gate", [0.0, 0.2])
def test_fresh_pair_strict_native_oracles_through_window_100(batch, gate):
    torch.set_num_threads(1)
    flat, model = _fresh_pair(); bank = _bank(batch)
    with torch.no_grad():
        model.frontend.attn.g.fill_(gate)
        if gate: model.frontend.attn.q_cal.add_(0.03)
    # Gate is set before construction, so refresh W=100 must exercise the
    # route-aware lifted frontend as well as five-token repair.
    stream = RouteLiftedFiveTokenCurrentQueryStream(model, bank)
    flat_bank = M1Bank(bank.E0, bank.T, bank.unit_mask)
    torch.testing.assert_close(stream.current_prediction(), _oracle(model, stream), rtol=1e-5, atol=1e-5)
    if gate == 0:
        torch.testing.assert_close(stream.current_prediction(), flat.forward_last(stream.raw, flat_bank, bank.unit_mask), rtol=1e-5, atol=1e-5)
    values = torch.randn(101, batch, 64, generator=torch.Generator().manual_seed(990 + batch))
    for step, value in enumerate(values):
        got = stream.predict_tensor(value)
        want = _oracle(model, stream)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
        if gate == 0:
            torch.testing.assert_close(want, flat.forward_last(stream.raw, flat_bank, bank.unit_mask), rtol=1e-5, atol=1e-5)
        if gate and step == 3:
            # Explicit history rebuild is the other route-aware frontend path.
            stream.refresh_state(history=stream.raw)
            torch.testing.assert_close(stream.current_prediction(), _oracle(model, stream), rtol=1e-5, atol=1e-5)


def test_actual_p1_route_checkpoint_oracle_partial_lanes_masks_and_state_bytes():
    torch.set_num_threads(1)
    model, bank = _checkpoint_route(), _bank(4)
    stream = RouteLiftedFiveTokenCurrentQueryStream(model, bank)
    assert stream.static_cache_bytes > 0
    assert stream.state_bytes == stream.rolling_state_bytes + stream.static_cache_bytes
    single = RouteLiftedFiveTokenCurrentQueryStream(model, BankBatch(bank.E0[:1],bank.T[:1],bank.unit_mask[:1],bank.session_ids[:1],bank.unit_ids[:1]))
    values = torch.randn(101, 4, 64, generator=torch.Generator().manual_seed(818))
    frozen_inactive = {
        "raw": stream.raw[1:].clone(), "z": stream.z[1:].clone(),
        "counts": stream.observation_counts[1:].clone(),
        "memories": [(k[1:].clone(), v[1:].clone()) for k, v in stream.memories],
    }
    for index, value in enumerate(values, start=1):
        got = stream.predict_tensor(value[:1], active=1)
        one = single.predict_tensor(value[:1])
        torch.testing.assert_close(got, one, rtol=1e-5, atol=1e-5)
        assert torch.equal(stream.raw[1:], frozen_inactive["raw"])
        assert torch.equal(stream.z[1:], frozen_inactive["z"])
        assert torch.equal(stream.observation_counts[1:], frozen_inactive["counts"])
        for (k, v), (old_k, old_v) in zip(stream.memories, frozen_inactive["memories"], strict=True):
            assert torch.equal(k[1:], old_k) and torch.equal(v[1:], old_v)
        if index in {1, 4, 99, 100, 101}:
            torch.testing.assert_close(stream.current_prediction()[:1], _oracle(model, stream)[:1], rtol=1e-5, atol=1e-5)
    assert stream.on_done([True, False, False, False]) is None
    public_input = np.zeros((4, 64), dtype=np.float32)
    public_output = stream.predict(public_input)
    assert public_output.dtype == np.float32 and public_output.flags.owndata
    public_input.fill(123.)
    assert np.isfinite(public_output).all()


def test_route_lifecycle_bank_model_and_mask_mutations_require_refresh():
    torch.set_num_threads(1)
    _flat, model = _fresh_pair(); bank = _bank(1)
    stream = RouteLiftedFiveTokenCurrentQueryStream(model, bank)
    def refresh_and_numeric():
        prior = stream.raw.clone()
        stream.refresh_state(history=prior)
        torch.testing.assert_close(stream.current_prediction(), _oracle(model, stream), rtol=1e-5, atol=1e-5)
        got = stream.predict_tensor(torch.zeros(1, 64))
        torch.testing.assert_close(got, _oracle(model, stream), rtol=1e-5, atol=1e-5)
    with torch.no_grad(): bank.T[0,0,0].add_(0.01)
    with pytest.raises(RuntimeError, match="fail closed"): stream.current_prediction()
    refresh_and_numeric()
    with torch.no_grad(): model.frontend.attn.g.add_(0.1)
    with pytest.raises(RuntimeError, match="fail closed"): stream.current_prediction()
    refresh_and_numeric()
    with torch.no_grad(): bank.unit_mask[0,0].logical_not_()
    with pytest.raises(RuntimeError, match="fail closed"): stream.current_prediction()
    refresh_and_numeric()
