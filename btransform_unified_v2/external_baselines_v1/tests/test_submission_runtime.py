import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


RUNTIME = Path(__file__).parents[1] / "submission" / "falcon_decoder.py"


class FakeBCIDecoder:
    def __init__(self, task_config, batch_size):
        self._task_config = task_config


class FakeTask:
    def __init__(self):
        self.name = "m2"


class FakeConfig:
    task = FakeTask()
    n_channels = 2
    n_outputs = 1

    @staticmethod
    def hash_dataset(stem):
        return "hash-" + stem


@pytest.fixture()
def decoder_module(monkeypatch):
    falcon = types.ModuleType("falcon_challenge")
    interface = types.ModuleType("falcon_challenge.interface")
    interface.BCIDecoder = FakeBCIDecoder
    monkeypatch.setitem(sys.modules, "falcon_challenge", falcon)
    monkeypatch.setitem(sys.modules, "falcon_challenge.interface", interface)
    spec = importlib.util.spec_from_file_location("external_gf_test_runtime", RUNTIME)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_payload(tmp_path, method="coral"):
    manifest = {
        "schema": "external_gf_falcon_payload_v1", "task": "m2", "method": method,
        "max_batch": 2, "sessions": {
            "hash-a": {"array_prefix": "a"}, "hash-b": {"array_prefix": "b"},
        },
        "expected_prediction_fields": {"channels": 2, "outputs": 1, "history": 10},
    }
    arrays = {
        "wf_mean": np.zeros(20), "wf_scale": np.ones(20),
        # coefficient picks only the oldest and newest first transformed feature plus intercept
        "wf_coef": np.r_[np.array([[2.0], *([[0.0]] * 17), [3.0], [0.0], [5.0]]),].reshape(21, 1),
    }
    if method == "coral":
        for prefix, shift in (("a", 10.0), ("b", 20.0)):
            arrays[prefix + "_target_mean"] = np.array([shift, shift + 1])
            arrays[prefix + "_source_mean"] = np.array([1.0, 2.0])
            arrays[prefix + "_target_invroot"] = np.eye(2)
            arrays[prefix + "_source_root"] = np.eye(2)
    else:
        for prefix in ("a", "b"):
            arrays[prefix + "_target_fa_mean"] = np.zeros(2)
            arrays[prefix + "_Wpsi"] = np.eye(2)
            arrays[prefix + "_cov_z"] = np.eye(2)
            arrays[prefix + "_rotation"] = np.eye(2)
    np.savez(tmp_path / "payload.npz", **arrays)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))


def test_causal_history_transformed_zero_padding_and_reset(decoder_module, tmp_path):
    write_payload(tmp_path, "coral")
    decoder = decoder_module.ExternalGFFalconDecoder(FakeConfig(), tmp_path / "payload.npz", manifest_path=tmp_path / "manifest.json", batch_size=2)
    decoder.reset([Path("a"), Path("b")])
    # First output: padding is literal transformed zero, so only newest [1, 2] matters.
    got = decoder.predict(np.array([[10, 11], [20, 21]], dtype=np.uint8))
    np.testing.assert_allclose(got, [[8.0], [8.0]])  # 3 * feature[0] + intercept
    # After nine more bins, the first bin reaches oldest history position (coefficient 2).
    for _ in range(8):
        decoder.predict(np.zeros((2, 2), dtype=np.uint8))
    got = decoder.predict(np.zeros((2, 2), dtype=np.uint8))
    np.testing.assert_allclose(got, [[-20.0], [-50.0]])
    decoder.reset([Path("a")])
    np.testing.assert_allclose(decoder.predict(np.array([[10, 11]], dtype=np.uint8)), [[8.0]])


def test_observe_advances_exactly_once_and_unknown_tag_fails(decoder_module, tmp_path):
    write_payload(tmp_path, "aligned_fa")
    decoder = decoder_module.ExternalGFFalconDecoder(FakeConfig(), tmp_path / "payload.npz", manifest_path=tmp_path / "manifest.json", batch_size=2)
    with pytest.raises(ValueError, match="no transform"):
        decoder.reset([Path("missing")])
    decoder.reset([Path("a")])
    decoder.observe(np.array([[4, 5]], dtype=np.uint8))
    # predict advances a second bin, and the newest feature is zero.
    np.testing.assert_allclose(decoder.predict(np.zeros((1, 2), dtype=np.uint8)), [[5.0]])
    before = decoder._history.copy()
    decoder.on_done(np.array([True]))
    np.testing.assert_array_equal(decoder._history, before)
    decoder.set_batch_size(1)
    assert decoder.batch_size == 1 and decoder._active == 0


def test_predict_returns_an_owned_step_snapshot(decoder_module, tmp_path):
    write_payload(tmp_path, "coral")
    decoder = decoder_module.ExternalGFFalconDecoder(FakeConfig(), tmp_path / "payload.npz", manifest_path=tmp_path / "manifest.json", batch_size=1)
    decoder.reset([Path("a")])
    first = decoder.predict(np.array([[10, 11]], dtype=np.uint8))
    second = decoder.predict(np.zeros((1, 2), dtype=np.uint8))
    assert not np.shares_memory(first, second)
    np.testing.assert_allclose(first, [[8.0]])
    np.testing.assert_allclose(second, [[-22.0]])
    decoder.reset([Path("a")])
    np.testing.assert_allclose(first, [[8.0]])
