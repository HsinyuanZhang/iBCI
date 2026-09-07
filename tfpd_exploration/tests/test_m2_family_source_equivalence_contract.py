import numpy as np
import pytest

from tfpd_exploration.src.family_runtime_v1 import complete_m2_family_source as proof


def test_raw_authority_requires_padding_order_and_native_geometry():
    raw = np.zeros((60, 96), np.float32)
    starts = np.array([0, 4, 10], np.int64)
    target = np.zeros((3, 2), np.float32)
    proof.validate_arrays(raw, starts, target, 3)
    with pytest.raises(RuntimeError):
        proof.validate_arrays(raw, starts[::-1], target, 3)
    with pytest.raises(RuntimeError):
        proof.validate_arrays(raw, starts, target.astype(np.float64), 3)
    raw[0, 0] = 1
    with pytest.raises(RuntimeError):
        proof.validate_arrays(raw, starts, target, 3)


def test_archive_checks_sessions_starts_targets_and_predictions():
    arrays = {"prediction": np.ones((2, 2), np.float32), "target": np.zeros((2, 2), np.float32),
              "session": np.array(["a", "b"]), "start": np.array([0, 4], np.int64)}
    assert proof.validate_archive(arrays, arrays) == 0
    for key in arrays:
        altered = {k: v.copy() for k, v in arrays.items()}
        altered[key][0] = "c" if key == "session" else 2
        with pytest.raises((AssertionError, RuntimeError)):
            proof.validate_archive(arrays, altered)


def test_requires_finalizer_before_model_construction(tmp_path, monkeypatch):
    monkeypatch.setattr(proof, "FINAL", tmp_path)
    with pytest.raises(RuntimeError, match="absent"):
        proof.require_final()
    (tmp_path / "receipt.json").write_text('{"status":"RUNNING"}')
    (tmp_path / "selection_freeze.json").write_text('{}')
    with pytest.raises(RuntimeError, match="status"):
        proof.require_final()
