"""Synthetic contracts for the gated frozen-original M1 scorer.

No test imports an original image payload, opens source data, or runs a real
decoder.  The fake is intentionally stateful so replay ordering is testable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tfpd_exploration.src.family_runtime_v1 import score_original_m1_frozen as scorer


class _Calib:
    shape = (10, 1024, 64)

    def unsqueeze(self, _):
        return self

    def to(self, _):
        return self


class _State:
    def __init__(self, device="cpu"):
        self.device = type("Device", (), {"type": device})()


class _Classifier:
    def __init__(self, device="cpu"):
        self._state = _State(device)
    def parameters(self):
        return [self._state]
    def buffers(self):
        return []


class FakeEngine:
    window_size = 100
    behavior_scaling_factor = 1.0
    smooth_calibration = False
    clf = _Classifier()
    local_clf = type("M", (), {"modules": lambda self: [self], "training": False})()
    local_calib_trial_features = [_Calib()]

    def __init__(self):
        self.observation_buffer = np.zeros((100, 1, 64), np.float32)
        self.tags, self.predict_calls = [], 0

    def reset(self, tags):
        self.tags.append(tags)
        self.device = type("D", (), {"type": "cpu"})()
        self.observation_buffer.fill(0)

    def predict(self, value):
        self.predict_calls += 1
        self.observation_buffer[:-1] = self.observation_buffer[1:]
        self.observation_buffer[-1] = value
        # owning C-order FP32 public payload
        return np.full((1, 16), value[0, 0], np.float32)


def _row(starts=(0, 2, 104), *, nonzero_history=False):
    raw = np.zeros((320, 64), np.float32)
    if nonzero_history:
        raw[:, 0] = np.arange(len(raw), dtype=np.float32)
    else:
        raw[99:, 0] = np.arange(len(raw) - 99, dtype=np.float32)
    return {"raw": raw, "starts": np.asarray(starts, np.int64),
            "target": np.zeros((len(starts), 16), np.float32)}


def test_tag_and_public_input_contracts():
    assert str(scorer.tag_for("ses-20120926")).endswith("ses-20120926_behavior+ecephys")
    with pytest.raises(RuntimeError):
        scorer.tag_for("ses-nope")
    with pytest.raises(RuntimeError):
        scorer.assert_public(np.zeros((1, 16), np.float64))
    with pytest.raises(RuntimeError):
        scorer.assert_input(np.zeros((1, 64), np.float64))
    with pytest.raises(RuntimeError):
        scorer.assert_input(np.full((1, 64), np.nan, np.float32))
    assert scorer.assert_public(np.zeros((1, 16), np.float32)).flags.owndata


def test_constructor_cpu_contract_does_not_assume_pre_reset_device():
    engine = FakeEngine()
    assert not hasattr(engine, "device")
    scorer.assert_constructor_classifier_cpu(engine)
    engine.clf = _Classifier("cuda")
    with pytest.raises(RuntimeError, match="non-CPU"):
        scorer.assert_constructor_classifier_cpu(engine)


def test_replay_sparse_prime_gap_rollover_counts_and_native_oracle(monkeypatch):
    engine, row = FakeEngine(), _row()

    def native(fake, history):
        # The fake public value uses the latest raw column zero too.
        return np.full((1, 16), history[0, -1, 0], np.float32)

    monkeypatch.setattr(scorer, "_native", native)
    result = scorer.replay_session(engine, row, "ses-20120926")
    # endpoints at starts 0,2,104 mean a raw replay through endpoint 203.
    assert result["public_calls"] == 104
    assert result["initial_native_predictions"] == 1
    assert result["direct_native_count"] >= 4
    assert result["max_native_abs_error"] == 0
    assert result["prediction"].shape == (3, 16)
    assert engine.predict_calls == 104
    assert np.array_equal(engine.observation_buffer.transpose(1, 0, 2), row["raw"][104:204][None])


def test_replay_nonzero_start_primes_actual_history_before_native(monkeypatch):
    engine, row = FakeEngine(), _row((5, 7, 109), nonzero_history=True)
    seen = []
    def native(_, history):
        seen.append(history.copy())
        return np.full((1, 16), history[0, -1, 0], np.float32)
    monkeypatch.setattr(scorer, "_native", native)
    result = scorer.replay_session(engine, row, "ses-20120926")
    assert np.array_equal(seen[0], row["raw"][5:105][None])
    assert seen[0][0, 0, 0] == 5 and seen[0][0, -1, 0] == 104
    assert result["prediction"][0, 0] == 104
    assert result["actual_calibration_shape"] == [10, 1024, 64]
    assert np.array_equal(engine.observation_buffer.transpose(1, 0, 2), row["raw"][109:209][None])


def test_invalid_input_is_rejected_before_engine_state_advances(monkeypatch):
    engine, row = FakeEngine(), _row((0, 1))
    monkeypatch.setattr(scorer, "_native", lambda _, h: np.zeros((1, 16), np.float32))
    # The second public bin is nonfinite.  replay must reject it before predict.
    row["raw"][100, 0] = np.nan
    with pytest.raises(RuntimeError, match="replay raw"):
        scorer.replay_session(engine, row, "ses-20120926")
    assert engine.predict_calls == 0
    assert engine.tags == []
    assert np.array_equal(engine.observation_buffer, np.zeros((100, 1, 64), np.float32))


def test_invalid_native_output_and_buffer_are_rejected(monkeypatch):
    engine, row = FakeEngine(), _row((0, 1))
    monkeypatch.setattr(scorer, "_native", lambda _, h: np.zeros((1, 16), np.float64))
    with pytest.raises(RuntimeError, match="public output"):
        scorer.replay_session(engine, row, "ses-20120926")
    engine, row = FakeEngine(), _row((0, 1))
    monkeypatch.setattr(scorer, "_native", lambda _, h: np.zeros((1, 16), np.float32))
    original = engine.predict
    def bad_buffer(value):
        result = original(value)
        engine.observation_buffer = engine.observation_buffer[:-1]
        return result
    engine.predict = bad_buffer
    with pytest.raises(RuntimeError, match="raw-history"):
        scorer.replay_session(engine, row, "ses-20120926")


def test_gate_precedes_any_completion_audit(monkeypatch, tmp_path):
    monkeypatch.delenv(scorer.GO, raising=False)
    monkeypatch.setattr(scorer, "artifact_audit", lambda _: pytest.fail("audit should follow gate"))
    with pytest.raises(RuntimeError, match="review gate"):
        scorer.run(tmp_path / "run", tmp_path / "out", tmp_path / "proof", "0" * 64)


def test_proof_rejects_bad_batch_nan_and_negative_error(tmp_path):
    audit = {"selected": {"flat": {"x": 1}, "route": {"x": 2}}}
    base = {"status": "PASS_IMPLEMENTATION_EQUIVALENCE_ONLY", "outer_query_opened": False,
            "parameter_updates": 0, "batch": 1, "pre_artifact": audit, "post_artifact": audit,
            "pre_source": {"files": {}}, "post_source": {"files": {}}, "arms": {}}
    for arm in ("flat", "route"):
        base["arms"][arm] = {"selected": audit["selected"][arm], "scored_count": scorer.COUNT,
                              "public_calls": sum(scorer.PUBLIC_CALLS), "initial_current_predictions": 3,
                              "max_abs_error": 0.0}
    path = tmp_path / "proof.json"
    import json, hashlib
    def save(value):
        path.write_text(json.dumps(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    for field, value in (("batch", 4), ("max_abs_error", float("nan")), ("max_abs_error", -1.0)):
        item = json.loads(json.dumps(base))
        if field == "batch": item[field] = value
        else: item["arms"]["flat"][field] = value
        with pytest.raises(RuntimeError): scorer.require_proof(path, save(item), audit)


def test_image_closure_rejects_strict_hash_drift_before_loader(monkeypatch, tmp_path):
    """A changed expected image file is detectable without loading a wrapper."""
    image = tmp_path / "image.py"; image.write_text("x = 1\n")
    proof = tmp_path / "proof.json"; proof.write_text("{}")
    monkeypatch.setattr(scorer, "EXPECTED", {image: scorer.sha(image)})
    first = scorer.image_and_helper_closure(proof)
    image.write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="SHA drift"):
        scorer.image_and_helper_closure(proof)
    assert str(image) in first


def test_postclosure_detects_proof_mutation(monkeypatch, tmp_path):
    image = tmp_path / "image.py"; image.write_text("x = 1\n")
    proof = tmp_path / "proof.json"; proof.write_text("first")
    monkeypatch.setattr(scorer, "EXPECTED", {image: scorer.sha(image)})
    before = scorer.image_and_helper_closure(proof)
    proof.write_text("second")
    assert before != scorer.image_and_helper_closure(proof)


def test_constants_bind_original_image_not_h1_or_local_teacher():
    assert scorer.IMAGE.endswith("f5af9eb29b7f86616d898261070193b1b0777db75567848c62d7f888ce3d76cd")
    assert scorer.EXPECTED[scorer.PAYLOAD] == "052e9eab7be2af8bacd5298e348d414cce60dad767d6a0880b58dd39b27279f6"
    assert (scorer.W, scorer.UNITS, scorer.OUTPUTS, scorer.SCALE, scorer.COUNT) == (100, 64, 16, 1.0, 31252)
