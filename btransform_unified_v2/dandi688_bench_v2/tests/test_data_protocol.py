from __future__ import annotations

import numpy as np
import pytest

from dandi688_bench_v2 import data, protocol


def test_protocol_is_exact_2015_18_6_6_and_excludes_2013():
    assert len(protocol.TRAIN_SESSIONS) == 18
    assert len(protocol.DEV_SESSIONS) == len(protocol.FINAL_SESSIONS) == 6
    assert all("-2015" in name for name in protocol.TRAIN_SESSIONS)
    assert protocol.split_for(protocol.TRAIN_SESSIONS[0]) == "train"


def test_final_rejected_before_any_nwb_open(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("NWB reader must not run")
    monkeypatch.setattr(data, "NWBHDF5IO", forbidden)
    with pytest.raises(PermissionError):
        data.load_pair(protocol.FINAL_SESSIONS[0], purpose="development")


def test_padded_windows_and_limits():
    neural = np.repeat(np.arange(60, dtype=np.float32)[:, None], 2, axis=1)
    velocity = np.column_stack((np.arange(60, dtype=np.float32), -np.arange(60, dtype=np.float32)))
    record = data.SessionData("x", "train", "sua", neural, velocity,
        np.arange(61), np.array([50]), np.array([1]), np.array([1]), np.zeros((protocol.ACTIVITY_TRIALS, 100, 2), np.float32),
        np.zeros((protocol.CARRIER_TRIALS, 2), np.float32), np.zeros(protocol.CARRIER_TRIALS), np.array([0, 1]), np.array([0, 1]), np.zeros(96, bool), {})
    x, tv, um = data.padded_windows(record, record.query_indices, n_pad=3)
    assert x.shape == (1, 50, 3) and tv.all() and um.tolist() == [True, True, False]
    # The window ends at the returned query index, and its target is the same row.
    assert np.array_equal(x[0, :, 0], np.arange(1, 51, dtype=np.float32))
    assert np.array_equal(record.velocity[record.query_indices], np.array([[50., -50.]], np.float32))
    with pytest.raises(ValueError): data.padded_windows(record, np.array([11]))


def test_source_smoke_pair_conserves_raw_counts_and_separates_support_query(tmp_path):
    pair = data.load_pair("sub-C_ses-CO-20150716", purpose="source")
    sua, pmua = pair["sua"], pair["pmua"]
    assert sua.neural.shape[1] <= 100 and pmua.neural.shape[1] <= 100
    assert np.array_equal(sua.neural.sum(1), pmua.neural.sum(1))
    assert sua.carrier_counts.shape == (protocol.CARRIER_TRIALS, sua.neural.shape[1])
    assert np.array_equal(sua.carrier_counts.sum(1), pmua.carrier_counts.sum(1))
    assert sua.activity.shape == (protocol.ACTIVITY_TRIALS, 100, sua.neural.shape[1])
    assert set(sua.support_indices).isdisjoint(set(sua.query_indices))
    assert np.all(sua.carrier_indices >= 0)
    cache = tmp_path / "source.npz"; data.save_session(sua, cache)
    restored = data.load_cached_session(cache)
    assert restored.metadata["array_sha256"] == sua.metadata["array_sha256"]
    with np.load(cache, allow_pickle=False) as archive:
        broken = {name: archive[name] for name in archive.files}
    broken["neural"] = broken["neural"].copy(); broken["neural"][0, 0] += 1
    np.savez_compressed(cache, **broken)
    with pytest.raises(ValueError, match="hash mismatch"):
        data.load_cached_session(cache)


def test_m33_ols_keeps_nan_angle_rows_and_is_permutation_invariant():
    angles = np.array([0., np.nan, np.pi / 2, np.pi, -np.pi / 2] + [np.nan] * (protocol.CARRIER_TRIALS - 5))
    audit = data.validate_carrier_angles(angles)
    permuted = angles[np.random.default_rng(42).permutation(protocol.CARRIER_TRIALS)]
    assert audit == data.validate_carrier_angles(permuted) == {"finite_direction_count": 4, "direction_design_rank": 3}
