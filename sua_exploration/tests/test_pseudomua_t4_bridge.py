"""Fast invariants for the pseudo-MUA T4 bridge (no NWB fixture required)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc_maze.multisession_datamodule import (  # noqa: E402
    electrode_ids_from_units,
    pool_spikes_by_electrode,
)
from mc_maze.unit_side_features import (  # noqa: E402
    CANONICAL_DIRECTIONS_RAD,
    _side_feature_cache_path,
    _unit_tuning_features,
    permute_side_feature_rows,
    pool_trial_rates_by_electrode,
)
from mc_maze.multisession_datamodule import _cache_key, _source_fingerprint  # noqa: E402


def _t4_from_rates(rates: np.ndarray, direction_indices: np.ndarray) -> np.ndarray:
    """Exercise the same cosine-fit primitive used by the production T4 path."""
    present = sorted(set(direction_indices.tolist()))
    return np.stack(
        [_unit_tuning_features(row, direction_indices, present)[0] for row in rates]
    )


def test_pseudomua_spike_pool_conserves_every_bin_and_is_unit_order_invariant():
    spikes = np.array(
        [[1, 2, 4, 0], [3, 0, 1, 5], [0, 7, 2, 1]], dtype=np.float32
    )
    ids = np.array([9, 3, 9, 3], dtype=np.int64)
    pooled, channels = pool_spikes_by_electrode(spikes, ids)
    assert channels.tolist() == [3, 9]
    assert np.array_equal(pooled.sum(axis=1), spikes.sum(axis=1))

    order = np.array([2, 0, 3, 1])
    reordered, reordered_channels = pool_spikes_by_electrode(spikes[:, order], ids[order])
    assert np.array_equal(reordered_channels, channels)
    assert np.array_equal(reordered, pooled)


def test_singleton_electrodes_are_an_identity_pool():
    spikes = np.array([[2, 1, 0], [0, 4, 3]], dtype=np.float32)
    ids = np.array([12, 4, 7], dtype=np.int64)
    pooled, channels = pool_spikes_by_electrode(spikes, ids)
    assert channels.tolist() == [4, 7, 12]
    assert np.array_equal(pooled, spikes[:, [1, 2, 0]])

    rates = spikes.T.astype(np.float64)
    pooled_rates, rate_channels = pool_trial_rates_by_electrode(rates, ids)
    assert np.array_equal(rate_channels, channels)
    assert np.array_equal(pooled_rates, rates[[1, 2, 0]])


def test_electrode_ids_require_exactly_one_electrode_per_sorted_unit():
    class Region:
        def __init__(self, index):
            self.index = index

    valid = pd.DataFrame({"electrodes": [Region([4]), Region([9])]})
    assert np.array_equal(electrode_ids_from_units(valid), np.array([4, 9]))
    invalid = pd.DataFrame({"electrodes": [Region([4, 5])]})
    with pytest.raises(ValueError, match="exactly one electrode"):
        electrode_ids_from_units(invalid)


def test_pseudomua_t4_is_channel_sized_finite_and_unit_order_invariant():
    # Four rewarded calibration trials over four distinct canonical directions.
    directions = np.array([0, 1, 2, 3], dtype=np.int64)
    rates = np.array(
        [[3.0, 5.0, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0], [7.0, 0.0, 1.0, 2.0]],
        dtype=np.float64,
    )
    ids = np.array([8, 8, 2], dtype=np.int64)
    pooled_rates, channel_ids = pool_trial_rates_by_electrode(rates, ids)
    t4 = _t4_from_rates(pooled_rates, directions)
    assert t4.shape == (channel_ids.size, 4)
    assert channel_ids.size == 2
    assert np.isfinite(t4).all()

    order = np.array([2, 0, 1])
    reordered_rates, reordered_ids = pool_trial_rates_by_electrode(rates[order], ids[order])
    assert np.array_equal(reordered_ids, channel_ids)
    assert np.allclose(_t4_from_rates(reordered_rates, directions), t4)


def test_ts4_permutation_changes_rows_only_and_preserves_each_column_distribution():
    features = np.arange(24, dtype=np.float32).reshape(6, 4)
    permuted = permute_side_feature_rows(features, permutation_seed=43)
    assert permuted.shape == features.shape
    assert not np.array_equal(permuted, features)
    assert np.array_equal(np.sort(permuted, axis=0), np.sort(features, axis=0))
    assert np.array_equal(permuted.mean(axis=0), features.mean(axis=0))
    assert np.array_equal(permuted.std(axis=0), features.std(axis=0))


def test_sua_legacy_cache_key_is_preserved_and_never_crosses_pseudomua(monkeypatch, tmp_path):
    """SUA retains its legacy payload; pseudo-MUA is separately namespaced."""
    nwb_path = tmp_path / "sub-C_ses-CO-20151103_behavior+ecephys.nwb"
    nwb_path.touch()
    monkeypatch.setattr(
        "mc_maze.unit_side_features._electrode_mapping_fingerprint", lambda _: "mapping-v1"
    )
    kwargs = dict(feature_group="t4", pool_size=50, bin_size_ms=20, window_size=50, trial_result_filter="R")
    sua_default = _side_feature_cache_path(tmp_path, nwb_path, **kwargs)
    sua_explicit = _side_feature_cache_path(tmp_path, nwb_path, signal_view="sua", **kwargs)
    pseudomua = _side_feature_cache_path(tmp_path, nwb_path, signal_view="pseudo_mua", **kwargs)
    legacy_payload = {
        "cache_format_version": 1,
        "kind": "unit_side_features",
        "feature_group": "t4",
        "pool_size": 50,
        "bin_size_ms": 20,
        "window_size": 50,
        "trial_result_filter": "R",
        "source": _source_fingerprint(nwb_path),
    }
    legacy = tmp_path / "side_features" / f"{nwb_path.name.replace('_behavior+ecephys.nwb', '')}_{_cache_key(legacy_payload)[:20]}.npz"
    assert sua_default == sua_explicit
    assert sua_default == legacy
    assert sua_default != pseudomua
