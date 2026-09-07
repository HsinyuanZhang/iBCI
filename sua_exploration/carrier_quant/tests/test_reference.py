"""Headline equivalence test against fit_frozen_carrier."""
from __future__ import annotations

import numpy as np
import pytest

from carrier_quant.reference import accumulate_blocks, fit_carrier_batch, solve_carrier
from data.h1_m4_eb_pilot import fit_frozen_carrier

from carrier_quant.tests.synthetic_helpers import (
  legacy_hat_trace,
  make_h1_pilot_record,
  make_synthetic_calibration,
  make_synthetic_plan,
)

TOL = 1e-12


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
@pytest.mark.parametrize("n_blocks", [120, 200])
def test_reference_matches_fit_frozen_carrier(seed: int, n_blocks: int) -> None:
  plan = make_synthetic_plan(seed=seed + 1000)
  cal = make_synthetic_calibration(plan, seed=seed, n_blocks=n_blocks)
  rates, labels, plan = cal.rates, cal.labels, cal.plan
  record = make_h1_pilot_record(rates, labels)

  legacy = fit_frozen_carrier(record, plan, record.trial_values)
  candidate = fit_carrier_batch(rates, labels, plan)

  for key in ("carrier", "raw_carrier", "raw_rows", "beta", "weight", "projected_variance"):
    diff = np.max(np.abs(legacy[key] - candidate[key]))
    assert diff < TOL, f"{key} max abs diff {diff}"

  hat_legacy = legacy_hat_trace(rates, labels, plan)
  assert abs(candidate["hat_trace"] - hat_legacy) < TOL


def test_o2_sufficient_statistics_match_batch() -> None:
  plan = make_synthetic_plan(seed=99)
  cal = make_synthetic_calibration(plan, seed=5, n_blocks=180)
  rates, labels, plan = cal.rates, cal.labels, cal.plan
  state = accumulate_blocks(rates, labels, plan=plan)
  solved = solve_carrier(state, plan, num_neurons=rates.shape[1])
  batch = fit_carrier_batch(rates, labels, plan)
  for key in ("carrier", "beta", "hat_trace"):
    assert np.max(np.abs(solved[key] - batch[key])) < TOL


def test_rss_identity() -> None:
  plan = make_synthetic_plan(seed=3)
  cal = make_synthetic_calibration(plan, seed=11, n_blocks=64)
  rates, labels, plan = cal.rates, cal.labels, cal.plan
  state = accumulate_blocks(rates, labels, plan=plan)
  solved = solve_carrier(state, plan, num_neurons=rates.shape[1])
  z = ((rates - plan.mean[None, :]) / plan.scale[None, :]) @ plan.pcs[: plan.q].T
  design = np.column_stack((np.ones(z.shape[0]), z))
  beta = solved["beta"]
  rss_direct = np.square(labels - design @ beta).sum(axis=0)
  assert np.max(np.abs(rss_direct - (np.diag(state.yty) - 2 * np.sum(beta * state.dty, axis=0) + np.sum(beta * (state.dtd @ beta), axis=0)))) < TOL
