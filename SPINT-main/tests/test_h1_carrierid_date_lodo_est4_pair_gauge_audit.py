"""Focused no-NWB contracts for the EST4 minimal-pair audit route."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from src.data.h1_carrierid_date_lodo_est4 import (
    EST4_ARMS,
    EST4_PAIR_ARMS,
    EST4_PAIR_PREFLIGHT_SCHEMA,
    EST4_PAIR_PREFLIGHT_STATUS,
)


ROOT = Path(__file__).resolve().parents[1]
PAIR_PREFLIGHT = ROOT / "scripts/h1_carrierid_date_lodo_est4_pair_preflight.py"
AUDIT_PATH = ROOT / "scripts/h1_carrierid_date_lodo_est4_pair_posttraining_gauge_audit.py"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


PREFLIGHT = _module("est4_pair_preflight", PAIR_PREFLIGHT)
AUDIT = _module("est4_pair_gauge_audit", AUDIT_PATH)


def test_pair_protocol_is_explicit_and_does_not_mutate_six_arm_protocol() -> None:
    assert EST4_PAIR_ARMS == ("B-C", "L-C")
    assert tuple(EST4_ARMS) == ("B-C", "B-LS", "L-C", "L-C0", "L-LS", "L-RS")
    assert PREFLIGHT.PAIR_ARMS == EST4_PAIR_ARMS
    assert PREFLIGHT.PREFLIGHT_SCHEMA == EST4_PAIR_PREFLIGHT_SCHEMA
    assert PREFLIGHT.PREFLIGHT_STATUS == EST4_PAIR_PREFLIGHT_STATUS
    text = PAIR_PREFLIGHT.read_text(encoding="utf-8")
    assert "same_source_schedule" in text
    assert "source_training_loss_is_heldout_evidence" in text
    assert "target_recordings_opened" in text


def test_gauge_metrics_are_invariant_to_declared_latent_and_output_gauges() -> None:
    rng = np.random.default_rng(71)
    pcs = rng.normal(size=(16, 176))
    u = rng.normal(size=(7, 4))
    mu = rng.normal(size=(4,))
    left, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    right, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    alpha, beta = 2.75, 0.41
    first = AUDIT.gauge_aware_estimator_metrics(
        trained_pcs=pcs, trained_u=u, trained_mu=mu, trained_lambda=3.0, trained_tau2=0.2,
        frozen_pcs=pcs, frozen_u=u,
    )
    second = AUDIT.gauge_aware_estimator_metrics(
        trained_pcs=alpha * left @ pcs,
        trained_u=beta * u @ right,
        trained_mu=beta * mu @ right,
        trained_lambda=alpha * alpha * 3.0,
        trained_tau2=beta * beta * 0.2,
        frozen_pcs=pcs, frozen_u=u,
    )
    assert second["pcs"]["condition_number"] == pytest.approx(first["pcs"]["condition_number"])
    assert second["U"]["condition_number"] == pytest.approx(first["U"]["condition_number"])
    assert second["pcs"]["normalized_singular_values"] == pytest.approx(first["pcs"]["normalized_singular_values"])
    assert second["U"]["normalized_singular_values"] == pytest.approx(first["U"]["normalized_singular_values"])
    assert second["gauge_invariant_scalar_ratios"] == pytest.approx(first["gauge_invariant_scalar_ratios"])
    assert second["frozen_to_trained_subspace"]["pcs_row_space"]["max_principal_angle_degrees"] == pytest.approx(0.0, abs=3e-6)
    assert second["frozen_to_trained_subspace"]["U_column_space"]["max_principal_angle_degrees"] == pytest.approx(0.0, abs=3e-6)


def test_pair_preflight_cli_requires_explicit_source_access_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["h1_carrierid_date_lodo_est4_pair_preflight.py"])
    with pytest.raises(SystemExit, match="refusing implicit source/NWB access"):
        PREFLIGHT.main()


def test_posttraining_audit_cli_requires_explicit_source_access_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["h1_carrierid_date_lodo_est4_pair_posttraining_gauge_audit.py"])
    with pytest.raises(SystemExit, match="refusing implicit source/NWB access"):
        AUDIT.main()
