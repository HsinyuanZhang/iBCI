from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
H1_SRC = ROOT / "tfpd_exploration" / "h1_series_20260830" / "src"
for value in (str(H1_SRC), str(ROOT / "SPINT-main")):
    if value not in sys.path:
        sys.path.insert(0, value)

from h1_postpool_output_residual_v1.plan import DATE_ORDER, apply_beta, decide_oof, fit_beta


def _row(date: str, scale: float) -> dict[str, object]:
    target = np.asarray([[0.2, -0.1], [0.4, 0.5], [-0.3, 0.7]], dtype=np.float32) * scale
    static = target - np.asarray([[0.1, -0.2], [0.2, 0.1], [-0.1, 0.3]], dtype=np.float32)
    direction = (target - static) / np.float32(0.4)
    return {"date": date, "target": target, "static": static, "post": static + direction}


def test_closed_form_beta_recovers_known_scalar_under_equal_date_weighting() -> None:
    rows = [_row("d1", 1.0), _row("d1", 2.0), _row("d2", 0.7)]
    fit = fit_beta(rows)
    assert fit["beta"] == pytest.approx(0.4, abs=2e-8)
    assert fit["denominator"] > 1.0e-12
    assert fit["source_date_count"] == 2.0


def test_zero_beta_is_direct_static_return() -> None:
    static = np.arange(12, dtype=np.float32).reshape(6, 2)
    post = static + 3.0
    output = apply_beta(static, post, 0.0)
    assert output is static
    assert np.array_equal(output, static)


def test_nonzero_beta_uses_declared_output_residual() -> None:
    static = np.asarray([[1.0, 2.0]], dtype=np.float32)
    post = np.asarray([[3.0, -2.0]], dtype=np.float32)
    assert np.array_equal(apply_beta(static, post, -0.5), np.asarray([[0.0, 4.0]], dtype=np.float32))


def test_oof_gate_is_not_relaxed() -> None:
    passing = [
        {"outer_date": date, "delta_vs_static": delta}
        for date, delta in zip(DATE_ORDER, (0.010, 0.008, 0.006, 0.003, -0.001), strict=True)
    ]
    assert decide_oof(passing)["pass"] is True
    failing = [dict(row) for row in passing]
    failing[0]["delta_vs_static"] = -0.02
    assert decide_oof(failing)["pass"] is False
