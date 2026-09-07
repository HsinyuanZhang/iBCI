"""Focused deterministic tests for the M1 support-encoder CPU audit."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/audit_m1_support_encoder_gate_a.py"


def _module():
    spec = importlib.util.spec_from_file_location("m1_support_encoder_gate_a", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _session(module, name: str, scale: float = 1.0):
    n_trials, n_channels = 230, 3
    labels = np.tile(np.asarray([1, 2, 3, 4]), 58)[:n_trials]
    lengths = np.linspace(80, 120, n_trials)
    base = np.asarray([0.2, 0.5, 0.8])[None, :]
    tuning = labels[:, None] * np.asarray([0.05, 0.03, 0.02])[None, :]
    rates = scale * (base + tuning)
    sums = rates * lengths[:, None]
    return module.SessionData(name, sums, lengths, labels, Path(f"/{name}.nwb"))


def test_category_means_keep_fixed_level_and_channel_axes():
    module = _module()
    labels = np.asarray([1, 2, 3, 4, 1, 2, 3, 4])
    rates = np.column_stack([labels, labels * 10.0])
    means = module.category_means(rates, labels)
    assert means.shape == (2, 4)
    assert np.array_equal(means[0], [1, 2, 3, 4])
    assert np.array_equal(means[1], [10, 20, 30, 40])


def test_support_features_use_exactly_first_ten_trials():
    module = _module()
    data = _session(module, "s")
    x_before, d4_before = module.support_features(data.sums, data.lengths, data.labels, "stats")
    changed = data.sums.copy()
    changed[10:] += 1.0e9
    x_after, d4_after = module.support_features(changed, data.lengths, data.labels, "stats")
    assert np.array_equal(x_before, x_after)
    assert np.array_equal(d4_before, d4_after)


def test_ridge_prediction_is_finite_and_four_dimensional():
    module = _module()
    sessions = [_session(module, f"s{i}", 1.0 + 0.1 * i) for i in range(3)]
    model = module.fit_ridge(sessions[:2], "stats", "residual", 0.1)
    prediction = module.predict_ridge(model, sessions[2])
    assert prediction.shape == (3, 4)
    assert np.all(np.isfinite(prediction))
    assert np.all(prediction >= 0)


def test_permutation_and_label_controls_are_deterministic_nonidentity():
    module = _module()
    first = module.nonidentity_permutation(64, "control", 42)
    second = module.nonidentity_permutation(64, "control", 42)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, np.arange(64))
    session = _session(module, "s")
    labels, permutation = module.shuffled_labels(session)
    assert not np.array_equal(labels, session.labels[:10])
    assert sorted(labels.tolist()) == sorted(session.labels[:10].tolist())
    assert not np.array_equal(permutation, np.arange(10))


def test_paired_ratio_reports_geometric_ratio_and_t_interval():
    module = _module()
    result = module.paired_ratio([0.5, 1.0, 1.5, 2.0], [1.0, 2.0, 3.0, 4.0], "unit")
    assert result["geometric_mean_ratio"] == pytest.approx(0.5)
    assert result["n_numerator_better"] == 4
    assert result["paired_t_95_ratio_interval"] == pytest.approx([0.5, 0.5])


def test_nonfinite_first_candidate_is_never_selected():
    module = _module()
    records = [
        {"status": "invalid_nonfinite", "score": None, "name": "bad"},
        {"status": "ok", "score": 2.0, "name": "finite"},
        {"status": "ok", "score": 3.0, "name": "worse"},
    ]
    selected = module.select_finite_record(records, "score", ("name",), "unit test")
    assert selected["name"] == "finite"


def test_strict_json_replaces_nonfinite_values():
    module = _module()
    assert module.strict_json({"nan": np.nan, "inf": np.inf, "ok": 1.5}) == {
        "nan": None,
        "inf": None,
        "ok": 1.5,
    }
