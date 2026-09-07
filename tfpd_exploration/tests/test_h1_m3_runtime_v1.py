"""No-data tests for the frozen native M3 readout stream decorator."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from tfpd_exploration.src.h1_m3_runtime_v1 import FrozenM3Readout, M3NativeReadoutStream


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha(value):
    value = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def _map(offset):
    return {
        "family": "MAT7", "ridge": 0.0,
        "p_mean": np.arange(7, dtype=np.float64) * 0.1,
        "p_scale": np.arange(1, 8, dtype=np.float64),
        "y_mean": np.full(7, offset, dtype=np.float64),
        "y_scale": np.linspace(0.5, 1.1, 7, dtype=np.float64),
        "weight": np.eye(7, dtype=np.float64) * (1.0 + offset),
        "intercept": np.full(7, offset / 10.0, dtype=np.float64),
    }


def _artifacts(tmp_path):
    checkpoint, plain = tmp_path / "model.pt", tmp_path / "plain.pt"
    checkpoint.write_bytes(b"checkpoint")
    model = Model(); torch.save(model.state_dict(), plain)
    bank = Bank()
    maps = {"a": _map(0.0), "b": _map(2.0)}
    maps_file = tmp_path / "per_session_maps.pt"
    torch.save({"schema": "v4full24_canonical_m3_mat7_maps_v1", "family": "MAT7", "ridge": 0.0,
                "scale_floor": 1e-6, "maps": maps}, maps_file)
    sessions = {}
    for session, mapping in maps.items():
        sessions[session] = {"map_array_sha256": {name: _array_sha(mapping[name]) for name in
                                                  ("p_mean", "p_scale", "y_mean", "y_scale", "weight", "intercept")}}
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"schema": "v4full24_canonical_m3_mat7_readout_v1", "status": "FITTED_FIXED_CONTRACT",
        "family": "MAT7", "ridge": 0.0, "fit_dtype": "float64", "scale_floor": 1e-6,
        "checkpoint_sha256": _sha(checkpoint), "frozen_plain_ema_state_sha256": _sha(plain),
        "maps": str(maps_file), "maps_sha256": _sha(maps_file), "sessions": sessions,
        "source_cache_authority": {"arrays": {"train": {
            session: {"bank_e0_sha256": _array_sha(bank.E0), "bank_hc_sha256": _array_sha(bank.T)} for session in maps
        }}}}))
    return receipt, maps_file, checkpoint, plain, maps, bank


class Engine:
    def __init__(self, prediction, bank, session_id="a"):
        self.prediction = np.asarray(prediction, dtype=np.float32)
        self.session_id = session_id; self.observations = []; self.resets = []; self.done = 0; self.state_bytes = 123
        self.model = Model(); self.bank = bank; self.unit_ids = tuple(range(bank.E0.shape[-2])); self.task = "h1"; self.divisor = 20.0

    def observe(self, observation, *, session_id=None):
        if session_id is not None and session_id != self.session_id: raise ValueError("engine session")
        self.observations.append(np.asarray(observation).copy())

    def current_prediction(self): return torch.as_tensor(self.prediction)

    def predict(self, observation, *, session_id=None):
        self.observe(observation, session_id=session_id); return self.prediction.copy()

    def reset(self, *, session_id=None, **kwargs):
        self.session_id = session_id; self.resets.append((session_id, kwargs)); self.observations.clear()

    def set_bank(self, bank): self.bank = bank

    def on_done(self, **kwargs): self.done += 1; return "kept"


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.linear = torch.nn.Linear(2, 2); self.dropout = torch.nn.Dropout(p=0.9)
        with torch.no_grad(): self.linear.weight.fill_(0.25); self.linear.bias.fill_(-0.1)
        self.register_buffer("frontend_contract_version", torch.tensor(4, dtype=torch.int64)); self.eval()


class Bank:
    def __init__(self):
        self.E0 = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        self.T = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        self.unit_mask = torch.ones(4, dtype=torch.bool)


def _loaded(tmp_path):
    receipt, maps_file, checkpoint, plain, maps, bank = _artifacts(tmp_path)
    return FrozenM3Readout.load(receipt, checkpoint_path=checkpoint, plain_ema_state_path=plain), maps, bank


def _manual(mapping, x):
    x = np.asarray(x, dtype=np.float64)
    return (((x - mapping["p_mean"]) / mapping["p_scale"]) @ mapping["weight"] + mapping["intercept"]) * mapping["y_scale"] + mapping["y_mean"]


def test_b1_shape_finite_and_manual_float64_map_parity(tmp_path):
    readout, maps, bank = _loaded(tmp_path); raw = np.arange(7, dtype=np.float32)[None]
    stream = M3NativeReadoutStream(Engine(raw, bank), readout, session_id="a")
    got = stream.predict(np.ones((1, 3), dtype=np.float32))
    assert got.shape == (1, 7) and got.dtype == np.float32 and np.isfinite(got).all()
    np.testing.assert_array_equal(got, _manual(maps["a"], raw).astype(np.float32))
    assert stream.state_bytes == 123


def test_session_rebind_selects_different_map_and_unknown_reset_is_isolated(tmp_path):
    readout, _, bank = _loaded(tmp_path); engine = Engine(np.ones((1, 7), dtype=np.float32), bank, "a")
    stream = M3NativeReadoutStream(engine, readout, session_id="a")
    before = stream.current_prediction().copy()
    with pytest.raises(KeyError): stream.reset(session_id="missing", unit_ids=(1, 2))
    assert engine.resets == [] and stream.session_id == "a"
    stream.reset(session_id="b", unit_ids=(5, 6))
    assert engine.resets == [("b", {"unit_ids": (5, 6)})]
    assert stream.session_id == "b"
    assert not np.array_equal(before, stream.current_prediction())


def test_checkpoint_mismatch_rejected_before_map_decode(tmp_path):
    receipt, _, checkpoint, plain, _, _ = _artifacts(tmp_path)
    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="checkpoint hash"):
        FrozenM3Readout.load(receipt, checkpoint_path=checkpoint, plain_ema_state_path=plain)


def test_dropout_engine_is_not_fitted_or_modified(tmp_path):
    readout, _, bank = _loaded(tmp_path); engine = Engine(np.ones((1, 7), dtype=np.float32), bank)
    before = {key: value.detach().clone() for key, value in engine.model.state_dict().items()}
    stream = M3NativeReadoutStream(engine, readout, session_id="a")
    stream.predict(np.zeros((1, 3), dtype=np.float32))
    assert engine.model.training is False
    assert engine.model.state_dict().keys() == before.keys()
    for key, value in before.items(): torch.testing.assert_close(engine.model.state_dict()[key], value)


def test_no_second_h1_divisor_and_current_prediction_is_copied_float32(tmp_path):
    readout, maps, bank = _loaded(tmp_path); raw = np.full(7, 20.0, dtype=np.float32)
    stream = M3NativeReadoutStream(Engine(raw, bank), readout, session_id="a")
    got = stream.current_prediction()
    np.testing.assert_array_equal(got, _manual(maps["a"], raw).astype(np.float32))
    assert not np.allclose(got, _manual(maps["a"], raw / 20.0).astype(np.float32))
    got[:] = 0.0
    assert not np.array_equal(stream.current_prediction(), got)


def test_observe_gaps_on_done_keeps_history_and_session_change_requires_reset(tmp_path):
    readout, _, bank = _loaded(tmp_path); engine = Engine(np.ones((1, 7), dtype=np.float32), bank)
    stream = M3NativeReadoutStream(engine, readout, session_id="a")
    stream.observe(np.zeros((1, 3), dtype=np.float32)); stream.observe(np.ones((1, 3), dtype=np.float32))
    assert len(engine.observations) == 2 and stream.on_done() == "kept" and len(engine.observations) == 2
    with pytest.raises(ValueError, match="explicit reset"):
        stream.predict(np.zeros((1, 3), dtype=np.float32), session_id="b")


def test_wrong_live_model_and_later_parameter_mutation_rejected(tmp_path):
    readout, _, bank = _loaded(tmp_path)
    wrong = Engine(np.ones((1, 7), dtype=np.float32), bank)
    with torch.no_grad(): wrong.model.linear.weight.add_(1.0)
    with pytest.raises(ValueError, match="live model differs"):
        M3NativeReadoutStream(wrong, readout, session_id="a")
    engine = Engine(np.ones((1, 7), dtype=np.float32), bank)
    stream = M3NativeReadoutStream(engine, readout, session_id="a")
    with torch.no_grad(): engine.model.linear.bias.add_(0.25)
    with pytest.raises(RuntimeError, match="model changed"):
        stream.current_prediction()
