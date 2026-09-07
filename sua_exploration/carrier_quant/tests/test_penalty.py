"""Penalty parameterization and system-matrix tests."""
from __future__ import annotations

import numpy as np
import pytest

from carrier_quant.penalty import PenaltySpec, build_regularizer, build_system_matrix
from carrier_quant.reference import accumulate_blocks, solve_carrier

from carrier_quant.tests.synthetic_helpers import (
  DEFAULT_EB_WEIGHT_MEAN,
  DEFAULT_LABEL_SNR,
  DEFAULT_SPECTRUM_RATIO,
  make_synthetic_calibration,
  make_synthetic_plan,
  measure_calibration_stats,
  pc_spectrum_stds,
)

TOL = 1e-12
IDENTITY_TOL = 1e-10
N_REF = 627
OUTPUT_KEYS = (
  "beta",
  "carrier",
  "raw_carrier",
  "raw_rows",
  "weight",
  "projected_variance",
  "hat_trace",
)


def test_synthetic_fixture_spectrum_snr_and_eb_weights() -> None:
  plan = make_synthetic_plan(seed=101)
  cal = make_synthetic_calibration(plan, seed=202, n_blocks=1200)
  stats = measure_calibration_stats(cal)

  designed = pc_spectrum_stds(plan.q, DEFAULT_SPECTRUM_RATIO)
  designed_ratio = float(np.max(designed) / np.min(designed))
  assert np.isclose(designed_ratio, DEFAULT_SPECTRUM_RATIO)

  assert stats["spectrum_ratio"] >= 30.0
  assert 5.0 <= stats["label_snr"] <= 20.0

  assert 0.45 <= stats["eb_weight_mean"] <= 0.85
  assert abs(stats["eb_weight_mean"] - DEFAULT_EB_WEIGHT_MEAN) < 0.15
  assert 0.10 <= stats["eb_weight_std"] <= 0.40
  assert stats["eb_weight_min"] < 0.15
  assert stats["eb_weight_max"] > 0.85
  assert stats["carrier_rms"] > 1.0e-4
  assert stats["raw_carrier_rms"] > 1.0e-6


def test_fixed_prior_matches_legacy_system() -> None:
  plan = make_synthetic_plan(seed=21)
  cal = make_synthetic_calibration(plan, seed=4, n_blocks=100)
  rates, labels, plan = cal.rates, cal.labels, cal.plan
  state = accumulate_blocks(rates, labels, plan=plan)
  spec = PenaltySpec.fixed_prior(plan.ridge_lambda)
  system = build_system_matrix(state.dtd, state.n, spec)
  reg = build_regularizer(spec, state.n)
  assert np.max(np.abs(system - (state.dtd + reg))) < 1e-15


@pytest.mark.parametrize("n_blocks", [300, 627, 800])
def test_constant_per_sample_equals_fixed_prior_with_effective_lambda(n_blocks: int) -> None:
  """constant_per_sample(lam, n_ref) at n == fixed_prior(lam * n / n_ref)."""
  plan = make_synthetic_plan(seed=24)
  cal = make_synthetic_calibration(plan, seed=9, n_blocks=n_blocks)
  rates, labels, plan = cal.rates, cal.labels, cal.plan
  state = accumulate_blocks(rates, labels, plan=plan)
  lam = plan.ridge_lambda
  lam_eff = lam * float(state.n) / float(N_REF)

  scaled = solve_carrier(
    state,
    plan,
    penalty=PenaltySpec.constant_per_sample(lam, n_ref=float(N_REF)),
    num_neurons=rates.shape[1],
  )
  fixed_eff = solve_carrier(
    state,
    plan,
    penalty=PenaltySpec.fixed_prior(lam_eff),
    num_neurons=rates.shape[1],
  )

  for key in OUTPUT_KEYS:
    diff = np.max(np.abs(np.asarray(scaled[key]) - np.asarray(fixed_eff[key])))
    assert diff < IDENTITY_TOL, f"n={n_blocks} {key} max abs diff {diff}"


@pytest.mark.parametrize("n_blocks", [300, 800])
def test_constant_per_sample_shrinkage_direction_off_n_ref(n_blocks: int) -> None:
  plan = make_synthetic_plan(seed=25)
  cal = make_synthetic_calibration(plan, seed=10, n_blocks=n_blocks)
  rates, labels, plan = cal.rates, cal.labels, cal.plan
  state = accumulate_blocks(rates, labels, plan=plan)

  fixed = solve_carrier(
    state,
    plan,
    penalty=PenaltySpec.fixed_prior(plan.ridge_lambda),
    num_neurons=rates.shape[1],
  )
  scaled = solve_carrier(
    state,
    plan,
    penalty=PenaltySpec.constant_per_sample(plan.ridge_lambda, n_ref=float(N_REF)),
    num_neurons=rates.shape[1],
  )

  fixed_shrinkage = np.linalg.norm(fixed["beta"][1:])
  scaled_shrinkage = np.linalg.norm(scaled["beta"][1:])
  if n_blocks > N_REF:
    assert fixed_shrinkage > scaled_shrinkage
  else:
    assert fixed_shrinkage < scaled_shrinkage


def test_fixed_prior_path_unchanged_vs_fit_frozen_carrier() -> None:
  """Default fixed_prior path must remain bit-identical to the sealed producer."""
  from data.h1_m4_eb_pilot import fit_frozen_carrier

  plan = make_synthetic_plan(seed=31)
  cal = make_synthetic_calibration(plan, seed=12, n_blocks=200)
  rates, labels, plan = cal.rates, cal.labels, cal.plan
  from carrier_quant.tests.synthetic_helpers import make_h1_pilot_record

  record = make_h1_pilot_record(rates, labels)
  legacy = fit_frozen_carrier(record, plan, record.trial_values)
  candidate = solve_carrier(
    accumulate_blocks(rates, labels, plan=plan),
    plan,
    penalty=PenaltySpec.fixed_prior(plan.ridge_lambda),
    num_neurons=rates.shape[1],
  )

  for key in ("carrier", "raw_carrier", "raw_rows", "beta", "weight", "projected_variance"):
    assert np.max(np.abs(legacy[key] - candidate[key])) < TOL


def test_diagonal_penalty_changes_solution() -> None:
  plan = make_synthetic_plan(seed=23)
  cal = make_synthetic_calibration(plan, seed=8, n_blocks=120)
  rates, labels, plan = cal.rates, cal.labels, cal.plan
  state = accumulate_blocks(rates, labels, plan=plan)
  iso = solve_carrier(state, plan, penalty=PenaltySpec.fixed_prior(plan.ridge_lambda), num_neurons=rates.shape[1])
  diag = np.linspace(0.8, 1.2, plan.q)
  noniso = solve_carrier(
    state,
    plan,
    penalty=PenaltySpec.fixed_prior(plan.ridge_lambda, diagonal=diag),
    num_neurons=rates.shape[1],
  )
  assert np.max(np.abs(iso["carrier"] - noniso["carrier"])) > 1e-9
