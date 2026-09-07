from __future__ import annotations

import numpy as np

from tfpd_exploration.h1_series_20260830.src.h1_m3_readout_calibration_v1.core import apply, fit, template
from tfpd_exploration.h1_series_20260830.src.h1_m3_readout_calibration_v1.plan import DATE_ORDER, decide


def test_full_affine_recovers_known_mapping() -> None:
    rng = np.random.default_rng(42)
    p = rng.normal(size=(200, 7))
    weight = rng.normal(size=(7, 7))
    bias = rng.normal(size=7)
    y = p @ weight + bias
    model = fit(p, y, family="MAT7", ridge=0.0)
    assert np.max(np.abs(apply(model, p) - y)) < 2e-5


def test_diagonal_affine_does_not_mix_outputs() -> None:
    rng = np.random.default_rng(7)
    p = rng.normal(size=(100, 7))
    y = p * np.arange(1, 8)[None] + np.arange(7)[None]
    model = fit(p, y, family="DIA7", ridge=0.0)
    assert np.count_nonzero(model.weight - np.diag(np.diag(model.weight))) == 0
    assert np.max(np.abs(apply(model, p) - y)) < 2e-5


def test_template_is_neural_free_calibration_mean() -> None:
    y = np.arange(70, dtype=np.float64).reshape(10, 7)
    value = template(y, 3)
    assert value.shape == (3, 7)
    assert np.allclose(value[0], y.mean(0)) and np.array_equal(value[0], value[2])


def test_decision_requires_gain_and_template_gates() -> None:
    passing = [{"outer_date": d, "base_r2": .3, "selected_r2": .31, "template_r2": .29} for d in DATE_ORDER]
    assert decide(passing)["pass"] is True
    bad = list(passing)
    bad[0] = {"outer_date": DATE_ORDER[0], "base_r2": .3, "selected_r2": .25, "template_r2": .24}
    assert decide(bad)["pass"] is False
