"""Shared synthetic fixtures for carrier_quant tests."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SPECTRUM_RATIO = 100.0
DEFAULT_LABEL_SNR = 10.0
DEFAULT_EB_WEIGHT_MEAN = 0.6554


@dataclass(frozen=True)
class SyntheticFrozenEBPlan:
  outer_date: str
  source_sessions: tuple[str, ...]
  source_input_sha256: tuple[str, ...]
  mean: np.ndarray
  scale: np.ndarray
  pcs: np.ndarray
  q: int
  ridge_lambda: float
  U: np.ndarray
  mu: np.ndarray
  tau2: float
  raw_plan_sha256: str = "synthetic"
  raw_receipt_sha256: str = "synthetic"
  eb_receipt_sha256: str = "synthetic"
  transform_sha256: str = "synthetic"


@dataclass(frozen=True)
class SyntheticCalibration:
  rates: np.ndarray
  labels: np.ndarray
  plan: SyntheticFrozenEBPlan


def pc_spectrum_stds(q: int, spectrum_ratio: float) -> np.ndarray:
  """Per-component standard deviations with max/min = spectrum_ratio (geometric decay)."""
  if q <= 0:
    raise ValueError("q must be positive")
  if spectrum_ratio < 1.0:
    raise ValueError("spectrum_ratio must be >= 1")
  if q == 1:
    return np.ones(1, dtype=np.float64)
  exponents = np.linspace(0.0, 1.0, q, dtype=np.float64)
  return np.power(float(spectrum_ratio), -exponents)


def derive_tau2_for_target_weight_mean(
  projected_variance: np.ndarray,
  *,
  target_mean: float = DEFAULT_EB_WEIGHT_MEAN,
) -> float:
  """Choose ``tau2`` so EB weights ``tau2/(tau2+pv)`` match a target mean on realized data."""
  pv = np.asarray(projected_variance, dtype=np.float64).reshape(-1)
  if pv.size == 0 or np.any(~np.isfinite(pv)) or np.any(pv < 0.0):
    raise ValueError("projected_variance must be finite and nonnegative")

  def mean_weight(tau2: float) -> float:
    return float(np.mean(tau2 / (tau2 + pv)))

  lo = 1.0e-30
  hi = max(float(np.max(pv)) * 1.0e6, 1.0e-12)
  if mean_weight(lo) >= target_mean:
    return lo
  if mean_weight(hi) <= target_mean:
    return hi

  for _ in range(80):
    mid = (lo + hi) / 2.0
    if mean_weight(mid) < target_mean:
      lo = mid
    else:
      hi = mid
  return (lo + hi) / 2.0


def make_synthetic_plan(
  *,
  seed: int,
  n_neurons: int = 176,
  q: int = 16,
  ridge_lambda: float = 100.0,
  tau2: float = 1.0,
) -> SyntheticFrozenEBPlan:
  """Build a plan shell; ``tau2`` is recalibrated from data in ``make_synthetic_calibration``."""
  rng = np.random.default_rng(seed)
  mean = rng.standard_normal(n_neurons) * 0.5
  scale = np.abs(rng.standard_normal(n_neurons)) + 0.1
  pcs = rng.standard_normal((32, n_neurons))
  u, _, vt = np.linalg.svd(rng.standard_normal((7, 7)), full_matrices=False)
  u_mat = (u @ vt)[:, :4]
  mu = rng.standard_normal(4) * 1e-6
  return SyntheticFrozenEBPlan(
    outer_date="19250108",
    source_sessions=("ses-synthetic",),
    source_input_sha256=("synthetic",),
    mean=mean,
    scale=scale,
    pcs=pcs,
    q=q,
    ridge_lambda=ridge_lambda,
    U=u_mat,
    mu=mu,
    tau2=tau2,
  )


def make_synthetic_calibration(
  plan: SyntheticFrozenEBPlan,
  *,
  seed: int,
  n_blocks: int = 200,
  spectrum_ratio: float = DEFAULT_SPECTRUM_RATIO,
  label_snr: float = DEFAULT_LABEL_SNR,
  eb_weight_mean: float = DEFAULT_EB_WEIGHT_MEAN,
) -> SyntheticCalibration:
  """Generate calibration data with realistic PC spectrum, label SNR, and EB weights.

  ``z`` uses a geometric spectrum (default 100x). Labels follow ``z @ coef + noise``
  at the requested SNR. ``tau2`` is derived from the realized ``projected_variance``
  so EB shrinkage weights land near the H1 Q1 reliability regime by default.
  """
  from carrier_quant.penalty import PenaltySpec
  from carrier_quant.reference import accumulate_blocks, solve_carrier

  rng = np.random.default_rng(seed)
  z_scales = pc_spectrum_stds(plan.q, spectrum_ratio)
  z = rng.standard_normal((n_blocks, plan.q)) * z_scales[None, :]
  rates = z @ plan.pcs[: plan.q] * plan.scale[None, :] + plan.mean[None, :]

  coef = rng.standard_normal((plan.q, 7))
  signal = z @ coef
  noise = rng.standard_normal((n_blocks, 7))
  signal_power = float(np.mean(np.square(signal)))
  noise_power = float(np.mean(np.square(noise)))
  if noise_power <= 0.0 or signal_power <= 0.0:
    raise ValueError("synthetic calibration requires positive signal and noise power")
  noise_scale = np.sqrt(signal_power / (label_snr * noise_power))
  labels = signal + noise_scale * noise
  rates = rates.astype(np.float64)
  labels = labels.astype(np.float64)

  pilot = replace(plan, tau2=1.0)
  state = accumulate_blocks(rates, labels, plan=pilot)
  projected_variance = solve_carrier(
    state,
    pilot,
    penalty=PenaltySpec.fixed_prior(pilot.ridge_lambda),
    num_neurons=rates.shape[1],
  )["projected_variance"]
  tau2 = derive_tau2_for_target_weight_mean(projected_variance, target_mean=eb_weight_mean)
  calibrated_plan = replace(plan, tau2=float(tau2))
  return SyntheticCalibration(rates=rates, labels=labels, plan=calibrated_plan)


def projected_z(rates: np.ndarray, plan: SyntheticFrozenEBPlan) -> np.ndarray:
  return ((rates - plan.mean[None, :]) / plan.scale[None, :]) @ plan.pcs[: plan.q].T


def measure_calibration_stats(
  calibration: SyntheticCalibration | tuple[np.ndarray, np.ndarray, SyntheticFrozenEBPlan],
  *,
  label_coef: np.ndarray | None = None,
) -> dict[str, Any]:
  """Summarize realized spectrum, label SNR, EB weights, and carrier scale."""
  from carrier_quant.penalty import PenaltySpec
  from carrier_quant.reference import accumulate_blocks, solve_carrier

  if isinstance(calibration, SyntheticCalibration):
    rates, labels, plan = calibration.rates, calibration.labels, calibration.plan
  else:
    rates, labels, plan = calibration

  z = projected_z(rates, plan)
  variances = np.var(z, axis=0)
  positive = variances[variances > 0]
  spectrum_ratio = float(np.max(positive) / np.min(positive)) if positive.size else float("nan")

  if label_coef is None:
    coef_hat = np.linalg.lstsq(z, labels, rcond=None)[0]
    signal = z @ coef_hat
  else:
    signal = z @ label_coef
  residual = labels - signal
  signal_power = float(np.mean(np.square(signal)))
  noise_power = float(np.mean(np.square(residual)))
  label_snr = signal_power / noise_power if noise_power > 0 else float("inf")

  solved = solve_carrier(
    accumulate_blocks(rates, labels, plan=plan),
    plan,
    penalty=PenaltySpec.fixed_prior(plan.ridge_lambda),
    num_neurons=rates.shape[1],
  )
  weight = np.asarray(solved["weight"], dtype=np.float64)
  carrier = np.asarray(solved["carrier"], dtype=np.float64)
  raw_carrier = np.asarray(solved["raw_carrier"], dtype=np.float64)

  return {
    "spectrum_ratio": spectrum_ratio,
    "label_snr": float(label_snr),
    "eb_weight_mean": float(np.mean(weight)),
    "eb_weight_std": float(np.std(weight)),
    "eb_weight_min": float(np.min(weight)),
    "eb_weight_max": float(np.max(weight)),
    "carrier_rms": float(np.sqrt(np.mean(np.square(carrier)))),
    "raw_carrier_rms": float(np.sqrt(np.mean(np.square(raw_carrier)))),
    "tau2": float(plan.tau2),
    "z_variances": variances,
  }


def make_h1_pilot_record(
  rates: np.ndarray,
  labels: np.ndarray,
  *,
  session_name: str = "ses-19250108T110520",
  trial_values: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
):
  from data.h1_m4_eb_pilot import H1PilotRecord, TrialBlocks

  blocks_per_trial = rates.shape[0] // len(trial_values)
  trials = []
  offset = 0
  for value in trial_values:
    end = offset + blocks_per_trial
    trial_rates = rates[offset:end]
    trial_labels = labels[offset:end]
    trials.append(
      TrialBlocks(
        trial_number=float(value),
        rates=trial_rates,
        velocity=trial_labels,
        block_indices=np.arange(offset, end, dtype=np.int64),
      )
    )
    offset = end
  date = session_name.split("-")[1][:8]
  return H1PilotRecord(
    session_name=session_name,
    date=date,
    path=Path("/tmp/synthetic.nwb"),
    input_sha256="synthetic",
    neural=np.zeros((0, rates.shape[1]), dtype=np.float32),
    velocity=np.zeros((0, 7), dtype=np.float32),
    trial_change=np.zeros(0, dtype=bool),
    eval_mask=np.zeros(0, dtype=bool),
    trial_num=np.zeros(0, dtype=np.float64),
    trial_values=trial_values,
    trials=tuple(trials),
  )


def legacy_hat_trace(rates: np.ndarray, labels: np.ndarray, plan) -> float:
  """Replicate the O(B^2) hat_trace from fit_frozen_carrier for regression."""
  z = projected_z(rates, plan)
  design = np.column_stack((np.ones(z.shape[0]), z))
  regularizer = np.eye(design.shape[1]) * plan.ridge_lambda
  regularizer[0, 0] = 0.0
  system = design.T @ design + regularizer
  return float(np.trace(design @ np.linalg.solve(system, design.T)))
