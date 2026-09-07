"""Exactness tests for R1 rotation and S1 SmoothQuant re-split."""
from __future__ import annotations

import numpy as np
import pytest

from carrier_quant.folds import (
  apply_rotation_fold,
  apply_smoothquant_resplit,
  hadamard_rotation,
  random_orthogonal,
  smoothquant_scales,
)
from carrier_quant.penalty import PenaltySpec
from carrier_quant.reference import fit_carrier_batch

from carrier_quant.tests.synthetic_helpers import make_synthetic_calibration, make_synthetic_plan

TOL = 1e-12


@pytest.mark.parametrize("seed", [0, 3, 11])
@pytest.mark.parametrize("q", [8, 16])
def test_r1_rotation_fold_exact(seed: int, q: int) -> None:
  plan = make_synthetic_plan(seed=seed, q=q)
  cal = make_synthetic_calibration(plan, seed=seed + 1, n_blocks=150)
  rates, labels, plan = cal.rates, cal.labels, cal.plan
  base = fit_carrier_batch(rates, labels, plan)

  r_h = hadamard_rotation(q)
  folded_h = apply_rotation_fold(plan, r_h, penalty=PenaltySpec.fixed_prior(plan.ridge_lambda))
  out_h = fit_carrier_batch(rates, labels, folded_h)

  r_rand = random_orthogonal(q, seed=seed + 99)
  folded_r = apply_rotation_fold(plan, r_rand, penalty=PenaltySpec.fixed_prior(plan.ridge_lambda))
  out_r = fit_carrier_batch(rates, labels, folded_r)

  for out in (out_h, out_r):
    for key in ("carrier", "raw_carrier", "raw_rows", "weight", "projected_variance", "hat_trace"):
      assert np.max(np.abs(base[key] - out[key])) < TOL, key


@pytest.mark.parametrize("seed", [2, 5, 17])
def test_s1_smoothquant_resplit_exact(seed: int) -> None:
  plan = make_synthetic_plan(seed=seed)
  cal = make_synthetic_calibration(plan, seed=seed + 7, n_blocks=160)
  rates, labels, plan = cal.rates, cal.labels, cal.plan
  base = fit_carrier_batch(rates, labels, plan)

  act_max = np.max(np.abs(rates - plan.mean[None, :]), axis=0)
  w_max = np.max(np.abs(plan.pcs[: plan.q]), axis=0)
  target_scale = smoothquant_scales(act_max, w_max, alpha=0.5)
  folded = apply_smoothquant_resplit(plan, target_scale)
  out = fit_carrier_batch(rates, labels, folded)

  z_base = ((rates - plan.mean[None, :]) / plan.scale[None, :]) @ plan.pcs[: plan.q].T
  z_fold = ((rates - folded.mean[None, :]) / folded.scale[None, :]) @ folded.pcs[: folded.q].T
  assert np.max(np.abs(z_base - z_fold)) < TOL

  for key in ("carrier", "raw_carrier", "raw_rows", "weight", "projected_variance", "hat_trace"):
    assert np.max(np.abs(base[key] - out[key])) < TOL, key


def test_rotation_rejects_non_isotropic_penalty() -> None:
  plan = make_synthetic_plan(seed=8)
  diag = np.linspace(0.5, 2.0, plan.q)
  penalty = PenaltySpec.fixed_prior(plan.ridge_lambda, diagonal=diag)
  r = hadamard_rotation(plan.q)
  with pytest.raises(ValueError, match="isotropic"):
    apply_rotation_fold(plan, r, penalty=penalty)
