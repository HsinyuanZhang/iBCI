"""Fixed-point simulation and B1 bit allocation tests."""
from __future__ import annotations

import numpy as np

from carrier_quant.fixedpoint import (
  QuantConfig,
  allocate_bits_reliability_weighted,
  allocate_bits_uniform,
  dequantize,
  ldl_decompose,
  ldl_solve,
  quantize_to_codes,
  simulate_quantized_carrier,
)
from carrier_quant.penalty import PenaltySpec, build_system_matrix
from carrier_quant.reference import AccumulationState, _solver_moments

from carrier_quant.tests.synthetic_helpers import make_synthetic_calibration, make_synthetic_plan


def test_ldl_matches_numpy_solve() -> None:
  rng = np.random.default_rng(0)
  a = rng.standard_normal((17, 17))
  spd = a @ a.T + np.eye(17)
  b = rng.standard_normal((17, 7))
  ref = np.linalg.solve(spd, b)
  got = ldl_solve(spd.astype(np.float32), b.astype(np.float32))
  assert np.max(np.abs(ref - got)) < 1e-4


def test_ldl_sqrt_free_structure() -> None:
  a = np.array([[4.0, 1.0, 0.0], [1.0, 4.0, 2.0], [0.0, 2.0, 4.0]], dtype=np.float32)
  l, d = ldl_decompose(a)
  assert l.dtype == np.float32
  assert d.dtype == np.float32
  recon = l @ np.diag(d) @ l.T
  assert np.max(np.abs(recon.astype(np.float64) - a.astype(np.float64))) < 1e-5
  assert np.allclose(np.diag(l), 1.0)


def test_bit_allocation_budget() -> None:
  n = 176
  total = 8 * n
  uniform = allocate_bits_uniform(n, total)
  weighted = allocate_bits_reliability_weighted(np.linspace(0.1, 1.0, n), total)
  assert int(uniform.sum()) == total
  assert int(weighted.sum()) == total
  assert np.all(uniform >= 4)
  assert np.all(weighted >= 4)


def test_quantize_codes_round_trip() -> None:
  values = np.array([[0.25, -0.5], [1.0, -1.0]], dtype=np.float64)
  scale = 0.5
  codes = quantize_to_codes(values, 8, scale)
  assert np.issubdtype(codes.dtype, np.integer)
  recovered = dequantize(codes, scale)
  assert np.max(np.abs(recovered - np.round(values / scale) * scale)) < 1e-12


def test_quantized_domain_honesty() -> None:
  plan = make_synthetic_plan(seed=50)
  cal = make_synthetic_calibration(plan, seed=51, n_blocks=60)
  out = simulate_quantized_carrier(cal.rates, cal.labels, cal.plan)
  for key in ("norm_codes", "pcs_codes", "label_codes"):
    assert np.issubdtype(out[key].dtype, np.integer), key
  for key in ("dtd_int", "dty_int", "yty_int"):
    assert np.issubdtype(out[key].dtype, np.integer), key
  assert out["ldl_L"].dtype == np.float32
  assert out["ldl_D"].dtype == np.float32
  assert out["system"].dtype == np.float64


def test_scale_algebra_round_trip() -> None:
  """Integer-accumulated moments match float reference up to input quantization."""
  rng = np.random.default_rng(7)
  n_blocks = 3
  q = 2
  n_neurons = 4
  mean = rng.standard_normal(n_neurons)
  scale = np.abs(rng.standard_normal(n_neurons)) + 0.1
  pcs = rng.standard_normal((q + 2, n_neurons))
  plan = make_synthetic_plan(seed=1, n_neurons=n_neurons, q=q)
  plan = plan.__class__(
    **{
      **plan.__dict__,
      "mean": mean,
      "scale": scale,
      "pcs": pcs,
      "q": q,
    }
  )
  rates = rng.standard_normal((n_blocks, n_neurons))
  labels = rng.standard_normal((n_blocks, 7))

  out = simulate_quantized_carrier(rates, labels, plan, config=QuantConfig(input_bits=8))
  design_scales = out["design_scales"]
  label_scale = out["label_scale"]

  dtd_ref = _dequant_dtd(out["dtd_int"], design_scales)
  dty_ref = _dequant_dty(out["dty_int"], design_scales, label_scale)
  yty_ref = _dequant_yty(out["yty_int"], label_scale)
  assert np.max(np.abs(dtd_ref - out["DtD"])) < 1e-12
  assert np.max(np.abs(dty_ref - out["Dty"])) < 1e-12
  assert np.max(np.abs(yty_ref - out["yty"])) < 1e-12

  normalized = (rates - plan.mean[None, :]) / plan.scale[None, :]
  norm_scale = float(np.max(np.abs(normalized))) / 127.0
  norm_codes = quantize_to_codes(normalized, 8, norm_scale)
  pcs_scale = float(np.max(np.abs(plan.pcs[: q]))) / 127.0
  pcs_codes = quantize_to_codes(plan.pcs[: q], 8, pcs_scale)
  z_scale = norm_scale * pcs_scale
  z_float = dequantize(
    _int_accumulate_matmul_reference(norm_codes, pcs_codes),
    z_scale,
  )
  design_float = np.column_stack((np.ones(n_blocks), z_float))
  labels_float = dequantize(out["label_codes"], label_scale)

  dtd_direct = design_float.T @ design_float
  dty_direct = design_float.T @ labels_float
  yty_direct = labels_float.T @ labels_float
  assert np.max(np.abs(dtd_direct - out["DtD"])) < 1e-9
  assert np.max(np.abs(dty_direct - out["Dty"])) < 1e-9
  assert np.max(np.abs(yty_direct - out["yty"])) < 1e-9


def test_beta_drives_carrier_outputs() -> None:
  plan = make_synthetic_plan(seed=60)
  cal = make_synthetic_calibration(plan, seed=61, n_blocks=80)
  rates, labels = cal.rates, cal.labels

  out8 = simulate_quantized_carrier(rates, labels, cal.plan, config=QuantConfig(input_bits=8))
  out4 = simulate_quantized_carrier(rates, labels, cal.plan, config=QuantConfig(input_bits=4))

  beta_ldl = out8["beta"]
  system32 = out8["system"].astype(np.float32)
  beta_from_ldl = ldl_solve(system32, out8["Dty"].astype(np.float32))
  assert np.max(np.abs(beta_ldl - beta_from_ldl)) < 1e-5

  assert out8["carrier"].shape == (rates.shape[1], 4)
  assert np.max(np.abs(out8["carrier"] - out4["carrier"])) > 1e-8


def test_quantized_simulation_runs() -> None:
  plan = make_synthetic_plan(seed=50)
  cal = make_synthetic_calibration(plan, seed=51, n_blocks=60)
  rates, labels, plan = cal.rates, cal.labels, cal.plan
  out = simulate_quantized_carrier(rates, labels, plan)
  assert out["carrier"].shape == (rates.shape[1], 4)
  assert "beta" in out
  stress = simulate_quantized_carrier(rates, labels, plan, int16_accumulator_stress=True)
  assert stress["int16_stress_test"] is True
  assert stress["accumulator_bits"] == 16


def test_quantized_constant_per_sample_uses_normalized_rhs() -> None:
  n_ref = 627.0
  plan = make_synthetic_plan(seed=70)
  cal = make_synthetic_calibration(plan, seed=71, n_blocks=80)
  penalty = PenaltySpec.constant_per_sample(plan.ridge_lambda, n_ref=n_ref)
  out = simulate_quantized_carrier(cal.rates, cal.labels, cal.plan, penalty=penalty)

  state = AccumulationState(
    design_dim=out["DtD"].shape[0],
    n=out["n"],
    dtd=out["DtD"],
    dty=out["Dty"],
  )
  _, dty_view, _ = _solver_moments(state, penalty)
  system32 = build_system_matrix(out["DtD"], out["n"], penalty).astype(np.float32)

  beta_correct = ldl_solve(system32, dty_view.astype(np.float32))
  beta_wrong = ldl_solve(system32, out["Dty"].astype(np.float32))

  assert np.max(np.abs(out["beta"] - beta_correct)) < 1e-5
  assert np.max(np.abs(out["beta"] - beta_wrong)) > 1e-3
  assert np.max(np.abs(beta_correct - beta_wrong)) > 1e-3


def test_quantized_cps_matches_fixed_prior_effective_lambda() -> None:
  n_ref = 627.0
  plan = make_synthetic_plan(seed=72)
  cal = make_synthetic_calibration(plan, seed=73, n_blocks=80)
  rates, labels, plan = cal.rates, cal.labels, cal.plan
  lam = plan.ridge_lambda
  n = rates.shape[0]
  lam_eff = lam * float(n) / n_ref

  cps = simulate_quantized_carrier(
    rates,
    labels,
    plan,
    penalty=PenaltySpec.constant_per_sample(lam, n_ref=n_ref),
  )
  fixed_eff = simulate_quantized_carrier(
    rates,
    labels,
    plan,
    penalty=PenaltySpec.fixed_prior(lam_eff),
  )

  for key in ("beta", "carrier"):
    diff = np.max(np.abs(cps[key] - fixed_eff[key]))
    assert diff < 1e-3, f"{key} max abs diff {diff}"


def _dequant_dtd(dtd_int: np.ndarray, scales: np.ndarray) -> np.ndarray:
  s = np.asarray(scales, dtype=np.float64)
  return dtd_int.astype(np.float64) * s[:, None] * s[None, :]


def _dequant_dty(dty_int: np.ndarray, design_scales: np.ndarray, label_scale: float) -> np.ndarray:
  s = np.asarray(design_scales, dtype=np.float64)
  return dty_int.astype(np.float64) * s[:, None] * float(label_scale)


def _dequant_yty(yty_int: np.ndarray, label_scale: float) -> np.ndarray:
  s = float(label_scale)
  return yty_int.astype(np.float64) * s * s


def _int_accumulate_matmul_reference(lhs_codes: np.ndarray, rhs_codes: np.ndarray) -> np.ndarray:
  out = np.zeros((lhs_codes.shape[0], rhs_codes.shape[0]), dtype=np.int64)
  for b in range(lhs_codes.shape[0]):
    for j in range(rhs_codes.shape[0]):
      for k in range(lhs_codes.shape[1]):
        out[b, j] += int(lhs_codes[b, k]) * int(rhs_codes[j, k])
  return out
