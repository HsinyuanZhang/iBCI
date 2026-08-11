"""Focused CPU tests for the production sparse endpoint carrier."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/data/rt_sparse_endpoint_loader.py"
spec = importlib.util.spec_from_file_location("rt_sparse_endpoint_loader_test", MODULE)
assert spec and spec.loader
loader = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = loader
spec.loader.exec_module(loader)


def test_endpoint_estimator_returns_exact_t4d_pad_without_velocity():
    # Three 200-ms reaches with different directions, adequate for rank-3 OLS.
    starts = np.array([0.0, 0.2, 0.4])
    stops = np.array([0.2, 0.4, 0.6])
    cues = np.array([[0.0], [0.2], [0.4]])
    target_count = np.ones(3)
    t = np.arange(0.0, 0.61, 0.01)
    p = np.zeros((t.size, 2))
    first = t < .2; second = (t >= .2) & (t < .4); third = t >= .4
    p[first, 0] = t[first] * 10
    p[second, 0] = 2.; p[second, 1] = (t[second] - .2) * 10
    p[third, 0] = 2. - (t[third] - .4) * 10; p[third, 1] = 2.
    raw, audit = loader._carrier_from_endpoint_payload(
        starts=starts, stops=stops, cues=cues, target_count=target_count,
        position_times=t, positions=p, spike_times=[np.arange(.01, .60, .04), np.arange(.02, .60, .05)],
    )
    assert raw.shape == (2, 4)
    assert np.array_equal(raw[:, 2:], np.zeros((2, 2), dtype=np.float32))
    assert audit["dense_velocity_read"] is False
    assert audit["access_log"][-1] == "carrier_frozen"


def test_estimator_rejects_rank_deficient_endpoint_directions():
    starts = np.array([0., .2, .4]); stops = starts + .2
    with np.testing.assert_raises(ValueError):
        loader._carrier_from_endpoint_payload(
            starts=starts, stops=stops, cues=starts[:, None], target_count=np.ones(3),
            position_times=np.arange(0., .61, .01),
            positions=np.column_stack([np.arange(0., .61, .01), np.zeros(61)]),
            spike_times=[np.arange(.01, .6, .04)],
        )


def test_production_path_has_ordered_dense_boundary_and_no_exploratory_import():
    source = MODULE.read_text()
    assert "sua_exploration" not in source
    assert source.index("raw_feature, audit = _carrier_from_endpoint_payload") < source.index("dense_raw = load_rt_session")
    assert "carrier_unchanged_after_dense_target" in source
    assert "pos.data[:]" not in source
    assert "unique_endpoint_coordinate_samples" in source
    assert "_canonical_unit(pos.unit) != \"cm\"" in source
    dataset = (ROOT / "src/data/falcon_datamodule.py").read_text()
    assert "precomputed_rt_sparse_endpoint_t4d" in dataset
    assert "dense_velocity_k4_estimator_called\": False" in dataset
    assert "values[:, 2:] = 0.0" in dataset


def test_exact_matched_config_contract_is_declared():
    config = (ROOT / "configs/experiment/rt_sparse_endpoint_t4d_clean_nested_loso_m24.yaml").read_text()
    assert "side_feature_group: rt_sparse_endpoint_t4d" in config
    assert "max_epochs: 35" in config


def test_existing_full_zero_xls_surfaces_remain_declared():
    nested = (ROOT / "src/data/rt_nested_loso_datamodule.py").read_text()
    falcon = (ROOT / "src/data/falcon_datamodule.py").read_text()
    assert '"afc4_vel"' in nested and '"zero4"' in nested and '"afc4_xls_v2"' in nested
    assert "afc4_xls_v2" in falcon and "mask_normalized_k4_components" in falcon
