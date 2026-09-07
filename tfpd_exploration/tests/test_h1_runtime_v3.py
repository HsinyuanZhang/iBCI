"""Compact-map binding preserves V1 maps and mutation guards, without NWB."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tfpd_exploration.src.h1_m3_runtime_v1 import M3NativeReadoutStream
from tfpd_exploration.src.h1_runtime_v3 import CompactM3NativeReadoutStream, load_frozen_readout
from tfpd_exploration.tests.test_h1_m3_runtime_v1 import Engine, _loaded, _artifacts, _sha


def test_exact_map_and_no_retained_checkpoint_copy(tmp_path):
    readout, _, bank = _loaded(tmp_path)
    raw = np.arange(7, dtype=np.float32)[None]
    old = M3NativeReadoutStream(Engine(raw, bank), readout, session_id="a")
    new = CompactM3NativeReadoutStream(Engine(raw, bank), readout, session_id="a")
    assert new._expected_state is None
    expected_removed = sum(v.numel()*v.element_size() for v in old._expected_state.values())
    assert new.memory_breakdown["duplicate_checkpoint_tensor_bytes_removed_vs_v1"] == expected_removed > 0
    np.testing.assert_array_equal(new.predict(np.ones((1, 3))), old.predict(np.ones((1, 3))))
    assert new.state_bytes == 123 + new.readout_array_bytes


def test_no_receipt_file_read_on_predict_and_exact_session_reset(tmp_path, monkeypatch):
    readout, _, bank = _loaded(tmp_path)
    raw = np.ones((1, 7), dtype=np.float32)
    stream = CompactM3NativeReadoutStream(Engine(raw, bank), readout, session_id="a")
    def forbidden(*args, **kwargs):
        raise AssertionError("steady prediction must not reopen receipt")
    monkeypatch.setattr(type(readout.receipt_path), "read_text", forbidden)
    first = stream.predict(np.ones((1, 3)))
    stream.reset(session_id="b")
    second = stream.predict(np.ones((1, 3)))
    assert not np.array_equal(first, second)
    history = list(stream.engine.observations)
    stream.on_done()
    assert stream.engine.observations == history


def test_numpy_alias_bank_mutation_is_still_detected(tmp_path):
    readout, _, bank = _loaded(tmp_path)
    stream = CompactM3NativeReadoutStream(Engine(np.ones((1, 7)), bank), readout, session_id="a")
    bank.E0.numpy()[0, 0] += 1.0
    with pytest.raises(ValueError, match="bank does not match"):
        stream.current_prediction()


def test_model_mutation_rejected_before_session_reset(tmp_path):
    readout, _, bank = _loaded(tmp_path)
    engine = Engine(np.ones((1, 7)), bank)
    stream = CompactM3NativeReadoutStream(engine, readout, session_id="a")
    with torch.no_grad():
        engine.model.linear.bias.add_(0.1)
    with pytest.raises(RuntimeError, match="model changed"):
        stream.reset(session_id="b")
    assert engine.resets == [] and engine.session_id == "a"


def test_units_and_mask_mutation_rejected(tmp_path):
    readout, _, bank = _loaded(tmp_path)
    engine = Engine(np.ones((1, 7)), bank)
    stream = CompactM3NativeReadoutStream(engine, readout, session_id="a")
    engine.divisor = 1.0
    with pytest.raises(RuntimeError, match="unit contract"):
        stream.current_prediction()
    engine.divisor = 20.0
    bank.unit_mask.numpy()[0] = False
    with pytest.raises(RuntimeError, match="bank/roster changed"):
        stream.current_prediction()


def test_explicit_v4_v6_tag_pairing_and_map_parity(tmp_path):
    import json
    receipt, maps_file, checkpoint, plain, _, _ = _artifacts(tmp_path)
    v4 = load_frozen_readout(receipt, checkpoint_path=checkpoint, plain_ema_state_path=plain)
    record = json.loads(receipt.read_text())
    record["schema"] = "v6_canonical_m3_mat7_readout_v1"
    record["plain_ema_sha256"] = record.pop("frozen_plain_ema_state_sha256")
    record.pop("fit_dtype")
    for session in record["sessions"].values():
        session.pop("map_array_sha256")
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="maps tag"):
        load_frozen_readout(receipt, checkpoint_path=checkpoint, plain_ema_state_path=plain)
    payload = torch.load(maps_file, weights_only=False)
    payload["schema"] = "v6_canonical_m3_mat7_maps_v1"
    payload.pop("scale_floor")
    torch.save(payload, maps_file)
    record["maps_sha256"] = _sha(maps_file)
    receipt.write_text(json.dumps(record))
    v6 = load_frozen_readout(receipt, checkpoint_path=checkpoint, plain_ema_state_path=plain)
    x = np.arange(7, dtype=np.float32)[None]
    np.testing.assert_array_equal(v4.apply("a", x), v6.apply("a", x))


@pytest.mark.parametrize('arm,contract', [
    ('full_v4', 'v4_causal_full_window_no_query_temporal_contract'),
    ('t_v6', 'v6_recency_query_temporal_contract4'),
])
def test_v7_separate_arm_binding_and_actual_float64_maps(tmp_path, arm, contract):
    import json
    receipt, maps_file, checkpoint, plain, _, _ = _artifacts(tmp_path)
    record = json.loads(receipt.read_text())
    record.update(schema='v7_canonical_m3_mat7_readout_v1', arm=arm, operator_contract=contract)
    record['plain_ema_sha256'] = record.pop('frozen_plain_ema_state_sha256')
    payload = torch.load(maps_file, weights_only=False)
    payload.update(schema='v7_canonical_m3_mat7_maps_v1', arm=arm)
    payload.pop('scale_floor')
    torch.save(payload, maps_file)
    record['maps_sha256'] = _sha(maps_file)
    receipt.write_text(json.dumps(record))
    load_frozen_readout(receipt, checkpoint_path=checkpoint, plain_ema_state_path=plain)
    payload['arm'] = 't_v6' if arm == 'full_v4' else 'full_v4'
    torch.save(payload, maps_file)
    record['maps_sha256'] = _sha(maps_file)
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='arm/operator/map'):
        load_frozen_readout(receipt, checkpoint_path=checkpoint, plain_ema_state_path=plain)
    payload['arm'] = arm
    payload['maps']['a']['weight'] = payload['maps']['a']['weight'].astype(np.float32)
    torch.save(payload, maps_file)
    record['maps_sha256'] = _sha(maps_file)
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='float64'):
        load_frozen_readout(receipt, checkpoint_path=checkpoint, plain_ema_state_path=plain)
