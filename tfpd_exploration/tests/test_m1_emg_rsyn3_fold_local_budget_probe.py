"""CPU contracts for fold-local EMG-rSyn3 budget-sweep token probe. No live NWB/GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest

from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import plan
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.budget_probe import (
    BudgetProbeError,
    apply_read_rule,
    build_budget_carrier_bank,
    carrier_reliability,
    redundancy_fraction,
)


ROOT = Path(__file__).resolve().parents[2]
BUDGET_PROBE_CLI = ROOT / "tfpd_exploration/scripts/run_m1_emg_rsyn3_fold_local_budget_probe.py"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
_CLI_ENV = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": str(ROOT),
}


def _row(
    budget: int,
    *,
    weights_r2: float,
    advantage: float,
    carrier_reliability: float | None,
    redundancy_fraction: float | None,
    null_mean: float = 0.0,
    positive_sessions: int = 4,
) -> dict[str, float | int | None]:
    return {
        "budget": budget,
        "weights_r2": weights_r2,
        "advantage": advantage,
        "null_mean": null_mean,
        "carrier_reliability": carrier_reliability,
        "redundancy_fraction": redundancy_fraction,
        "positive_sessions": positive_sessions,
    }


def test_budget_probe_plan_literals() -> None:
    assert plan.BUDGET_PROBE_ROOT_RELATIVE.startswith(plan.RESULT_ROOT_RELATIVE)
    assert plan.BUDGET_PROBE_ROOT_RELATIVE == f"{plan.RESULT_ROOT_RELATIVE}/budget_probe_v1"
    assert plan.BUDGET_PROBE_OWN_TOKEN in plan.OWN_TOKENS
    assert plan.BUDGET_PROBE_OWN_TOKEN == "run_m1_emg_rsyn3_fold_local_budget_probe.py"
    assert plan.BUDGET_PROBE_BUDGETS == plan.CARRIER_BUDGETS
    assert plan.BUDGET_PROBE_BUDGETS == (10, 6, 4, 2)
    assert plan.BUDGET_PROBE_ARM == "Z-Fix"
    assert plan.BUDGET_PROBE_RELIABILITY_FLOOR == 0.30
    rule = plan.BUDGET_PROBE_READ_RULE
    assert rule["primary_arm"] == "Z-Fix"
    assert rule["primary_metric"] == "weights_r2"
    assert "REDUNDANCY_PERSISTS" in rule
    assert "REDUNDANCY_BREAKS_AT_LOW_BUDGET" in rule
    assert "INCONCLUSIVE_UNRELIABLE_TARGET" in rule
    assert "PARTIAL_DECAY" in rule
    assert rule["REDUNDANCY_PERSISTS"]["min_advantage"] == 0.10
    assert rule["REDUNDANCY_PERSISTS"]["min_redundancy_fraction"] == 0.60
    assert rule["REDUNDANCY_BREAKS_AT_LOW_BUDGET"]["m10_min_weights_r2"] == 0.50
    assert rule["REDUNDANCY_BREAKS_AT_LOW_BUDGET"]["m10_min_advantage"] == 0.10
    assert rule["REDUNDANCY_BREAKS_AT_LOW_BUDGET"]["reliability_floor"] == 0.30
    assert rule["REDUNDANCY_BREAKS_AT_LOW_BUDGET"]["max_redundancy_fraction_at_smallest_reliable"] == 0.40
    assert rule["INCONCLUSIVE_UNRELIABLE_TARGET"]["reliability_floor"] == 0.30
    assert rule["INCONCLUSIVE_UNRELIABLE_TARGET"]["budget_below"] == 10
    assert "tfpd_exploration/scripts/run_m1_emg_rsyn3_fold_local_budget_probe.py" in plan.OWNED_PATHS
    assert "tfpd_exploration/tests/test_m1_emg_rsyn3_fold_local_budget_probe.py" in plan.OWNED_PATHS
    payload = plan.dry_cli_payload()
    assert payload["prospective_roots"]["budget_probe"] == plan.BUDGET_PROBE_ROOT_RELATIVE


def test_apply_read_rule_covers_all_four_branches() -> None:
    rule = plan.BUDGET_PROBE_READ_RULE
    persists = [
        _row(10, weights_r2=0.70, advantage=0.20, carrier_reliability=0.90, redundancy_fraction=0.78),
        _row(6, weights_r2=0.65, advantage=0.18, carrier_reliability=0.85, redundancy_fraction=0.76),
        _row(4, weights_r2=0.60, advantage=0.15, carrier_reliability=0.80, redundancy_fraction=0.75),
        _row(2, weights_r2=0.50, advantage=0.12, carrier_reliability=0.70, redundancy_fraction=0.71),
    ]
    assert apply_read_rule(persists, rule) == "REDUNDANCY_PERSISTS"

    breaks = [
        _row(10, weights_r2=0.55, advantage=0.20, carrier_reliability=0.90, redundancy_fraction=0.61),
        _row(6, weights_r2=0.40, advantage=0.12, carrier_reliability=0.70, redundancy_fraction=0.57),
        _row(4, weights_r2=0.20, advantage=0.08, carrier_reliability=0.50, redundancy_fraction=0.40),
        _row(2, weights_r2=0.10, advantage=0.04, carrier_reliability=0.35, redundancy_fraction=0.29),
    ]
    assert apply_read_rule(breaks, rule) == "REDUNDANCY_BREAKS_AT_LOW_BUDGET"

    inconclusive = [
        _row(10, weights_r2=0.55, advantage=0.20, carrier_reliability=0.90, redundancy_fraction=0.61),
        _row(6, weights_r2=0.10, advantage=0.04, carrier_reliability=0.20, redundancy_fraction=0.50),
        _row(4, weights_r2=0.05, advantage=0.02, carrier_reliability=0.10, redundancy_fraction=0.50),
        _row(2, weights_r2=0.01, advantage=0.00, carrier_reliability=0.05, redundancy_fraction=0.20),
    ]
    assert apply_read_rule(inconclusive, rule) == "INCONCLUSIVE_UNRELIABLE_TARGET"

    partial = [
        _row(10, weights_r2=0.55, advantage=0.20, carrier_reliability=0.90, redundancy_fraction=0.61),
        _row(6, weights_r2=0.45, advantage=0.14, carrier_reliability=0.80, redundancy_fraction=0.56),
        _row(4, weights_r2=0.35, advantage=0.11, carrier_reliability=0.60, redundancy_fraction=0.58),
        _row(2, weights_r2=0.25, advantage=0.08, carrier_reliability=0.50, redundancy_fraction=0.50),
    ]
    assert apply_read_rule(partial, rule) == "PARTIAL_DECAY"


def test_carrier_reliability_high_for_linear_rates_near_zero_for_noise() -> None:
    rng = np.random.default_rng(0)
    n_trials = 8
    bins_per = 24
    n_units = 20
    rank = 3
    trial_ids = np.repeat(np.arange(n_trials, dtype=np.int64), bins_per)
    scores = rng.normal(size=(n_trials * bins_per, rank))
    weights = rng.normal(size=(n_units, rank))
    intercepts = rng.normal(size=(n_units,))
    linear_rates = scores @ weights.T + intercepts
    linear = carrier_reliability(scores, linear_rates, trial_ids=trial_ids)
    assert linear is not None
    assert float(linear) > 0.90

    noise_rates = rng.normal(size=linear_rates.shape)
    noise = carrier_reliability(scores, noise_rates, trial_ids=trial_ids)
    assert noise is not None
    assert abs(float(noise)) < 0.35


def test_build_budget_carrier_bank_rejects_out_of_range_budget_before_data() -> None:
    missing = Path("/nonexistent/spint-budget-probe-does-not-exist")
    with pytest.raises(BudgetProbeError, match="budget"):
        build_budget_carrier_bank(missing, budget=0)
    with pytest.raises(BudgetProbeError, match="budget"):
        build_budget_carrier_bank(missing, budget=11)
    with pytest.raises(BudgetProbeError, match="budget"):
        build_budget_carrier_bank(missing, budget=1.5)  # type: ignore[arg-type]
    with pytest.raises(BudgetProbeError, match="budget"):
        build_budget_carrier_bank(missing, budget="10")  # type: ignore[arg-type]


def test_redundancy_fraction_null_when_reliability_nonpositive() -> None:
    assert redundancy_fraction(0.50, 0.0) is None
    assert redundancy_fraction(0.50, -0.1) is None
    assert redundancy_fraction(0.50, None) is None
    assert redundancy_fraction(0.50, 0.80) == pytest.approx(0.50 / 0.80)


def test_budget_probe_public_cli_is_dry_and_execute_gpu_errors() -> None:
    dry = subprocess.run(
        [PYTHON, str(BUDGET_PROBE_CLI)],
        check=True, capture_output=True, text=True, cwd=str(ROOT),
        env=_CLI_ENV,
    )
    payload = json.loads(dry.stdout)
    assert payload["cli"] == "run_m1_emg_rsyn3_fold_local_budget_probe.py"
    assert payload["prospective_roots"]["budget_probe"] == plan.BUDGET_PROBE_ROOT_RELATIVE
    assert payload["public_gpu_capability"] is False
    gpu = subprocess.run(
        [PYTHON, str(BUDGET_PROBE_CLI), "--execute-gpu"],
        capture_output=True, text=True, cwd=str(ROOT),
        env=_CLI_ENV,
    )
    assert gpu.returncode != 0
    assert "budget probe is CPU-only and mints no GPU capability" in (gpu.stderr + gpu.stdout)
    execute = subprocess.run(
        [PYTHON, str(BUDGET_PROBE_CLI), "--execute"],
        capture_output=True, text=True, cwd=str(ROOT),
        env={**_CLI_ENV, "PYTHONNOUSERSITE": "0"},
    )
    assert execute.returncode != 0
    assert "PYTHONNOUSERSITE=1" in (execute.stderr + execute.stdout)
