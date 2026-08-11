"""Synthetic tests for isolated RT T4d Stage-2 carrier boundary."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/rt_sparse_endpoint_stage2_adapter.py"


def module():
    spec = importlib.util.spec_from_file_location("rt_sparse_endpoint_stage2_adapter", SCRIPT)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec); sys.modules[spec.name] = value; spec.loader.exec_module(value)
    return value


def endpoint_payload() -> dict[str, np.ndarray]:
    return {"trial_index": np.array([0, 1, 2, 3]), "theta_rad": np.array([0.0, np.pi / 2, np.pi, -np.pi / 2]),
            "left_bin": np.array([0, 10, 20, 30]), "right_bin": np.array([8, 18, 28, 38])}


def test_one_reach_row_carrier_has_exact_zero_pad_and_no_dense_schema():
    m = module(); neural = np.zeros((40, 2)); neural[0:5, 0] = 1; neural[10:15, 1] = 2
    result = m.construct_t4d_carrier(reaches=endpoint_payload(), neural_bins=neural)
    assert result.reach_rows == 4 and result.design_rank == 3
    assert np.array_equal(result.raw_feature[:, 2:], np.zeros((2, 2), dtype=np.float32))
    assert result.access_log == ("trial_events", "cursor_position_endpoints", "spikes")
    with pytest.raises(ValueError, match="schema"):
        m.construct_t4d_carrier(reaches={**endpoint_payload(), "dense": np.zeros(4)}, neural_bins=neural)


def test_source_only_13_session_normalizer_and_exact_zero_pad():
    m = module(); names = tuple(f"s{index}" for index in range(13))
    features = {name: np.array([[float(index + 1), float(index + 2), 0.0, 0.0]], dtype=np.float32) for index, name in enumerate(names)}
    norm = m.fit_t4d_normalizer(features, inner_train_sessions=names)
    applied = m.apply_t4d_normalizer(features[names[0]], norm)
    assert norm.fit_sessions == names and np.array_equal(norm.mean[2:], np.zeros(2, dtype=np.float32))
    assert np.array_equal(applied[:, 2:], np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="13"):
        m.fit_t4d_normalizer(features, inner_train_sessions=names[:-1])


def test_decoder_target_can_load_only_after_carrier_freeze_and_state_stays_equal():
    m = module(); carrier = m.EndpointCarrierResult(np.zeros((2, 4), dtype=np.float32), 3, 3, 1.0, ("trial_events", "cursor_position_endpoints", "spikes"))
    log = []
    outcome = m.outer_target_after_carrier_freeze(carrier, lambda: log.append("decoder_dense_target") or {"rows": 5})
    assert log == ["decoder_dense_target"]
    assert outcome["carrier_state_equal"] is True
    assert outcome["carrier_sha256_before_decoder_target"] == outcome["carrier_sha256_after_decoder_target"]


def test_matched_training_contract_binds_epochs_optimizer_lr_metric_window_pool_and_seed():
    m = module(); base = {"epochs": 35, "optimizer": "adamw", "learning_rate": 1e-3, "checkpoint_metric": "val_heldin/r2_mean", "window_size": 50, "query_start_trial": 24, "session_window_budget": 4096, "seed": 42}
    contract = m.matched_training_contract(base)
    assert contract == {key: base[key] for key in sorted(base)}
    with pytest.raises(ValueError, match="training spec drift"):
        m.matched_training_contract({**base, "epochs": 34})


def test_adapter_has_no_dense_stream_access_or_gpu_import():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in source and "NWBHDF5IO" not in source
    assert "cursor_vel" not in source and "Velocity" not in source
