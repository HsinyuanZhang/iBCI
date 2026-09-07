"""O1/O2 reference carrier fit: sufficient statistics and optimized solve.

Numerically equivalent to ``fit_frozen_carrier`` in ``h1_m4_eb_pilot.py`` when
using the default ``PenaltySpec.fixed_prior`` parameterization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .penalty import PenaltySpec, build_system_matrix

EPS = 1.0e-12


class CarrierPlan(Protocol):
  mean: np.ndarray
  scale: np.ndarray
  pcs: np.ndarray
  q: int
  ridge_lambda: float
  U: np.ndarray
  mu: np.ndarray
  tau2: float


@dataclass
class AccumulationState:
  """Sufficient statistics for the ridge carrier fit."""

  design_dim: int = 17
  n: int = 0
  dtd: np.ndarray | None = None
  dty: np.ndarray | None = None
  yty: np.ndarray = field(default_factory=lambda: np.zeros((7, 7), dtype=np.float64))

  def __post_init__(self) -> None:
    if self.dtd is None:
      self.dtd = np.zeros((self.design_dim, self.design_dim), dtype=np.float64)
    if self.dty is None:
      self.dty = np.zeros((self.design_dim, 7), dtype=np.float64)

  def copy(self) -> AccumulationState:
    return AccumulationState(
      design_dim=self.design_dim,
      n=self.n,
      dtd=self.dtd.copy(),
      dty=self.dty.copy(),
      yty=self.yty.copy(),
    )


def empty_state(design_dim: int = 17) -> AccumulationState:
  return AccumulationState(design_dim=design_dim)


def _project_block(rates: np.ndarray, plan: CarrierPlan) -> np.ndarray:
  return ((rates - plan.mean[None, :]) / plan.scale[None, :]) @ plan.pcs[: plan.q].T


def accumulate(
  state: AccumulationState,
  rates_block: np.ndarray,
  labels_block: np.ndarray,
  *,
  plan: CarrierPlan,
) -> AccumulationState:
  """Update sufficient statistics from one labelled calibration block."""
  rates = np.asarray(rates_block, dtype=np.float64)
  labels = np.asarray(labels_block, dtype=np.float64)
  design_dim = plan.q + 1
  if state.n == 0 and state.design_dim != design_dim:
    state = AccumulationState(design_dim=design_dim)
  elif state.design_dim != design_dim:
    raise ValueError(f"design dimension mismatch: state={state.design_dim}, plan q={plan.q}")
  z = _project_block(rates, plan)
  design = np.column_stack((np.ones(z.shape[0]), z))
  state.n += int(design.shape[0])
  state.dtd += design.T @ design
  state.dty += design.T @ labels
  state.yty += labels.T @ labels
  return state


def accumulate_blocks(
  rates: np.ndarray,
  labels: np.ndarray,
  *,
  plan: CarrierPlan,
) -> AccumulationState:
  """Canonical row-major accumulation over all blocks at once."""
  state = empty_state(plan.q + 1)
  return accumulate(state, rates, labels, plan=plan)


def _cho_solve_many(factor: tuple, rhs: np.ndarray) -> np.ndarray:
  if rhs.ndim == 1:
    return cho_solve(factor, rhs)
  return cho_solve(factor, rhs)


def _solve_with_factor(factor: tuple, rhs: np.ndarray) -> np.ndarray:
  return _cho_solve_many(factor, rhs)


def _hat_trace(factor: tuple, dtd: np.ndarray) -> float:
  """O1: trace(solve(system, DtD)) without forming design @ solve(system, design.T)."""
  solved = _solve_with_factor(factor, dtd)
  return float(np.trace(solved))


def _compute_g(factor: tuple, dtd: np.ndarray) -> np.ndarray:
  """O1: G = solve(system, DtD) @ inv(system) via two solves, no explicit inverse."""
  temp = _solve_with_factor(factor, dtd)
  # G = temp @ inv(system)  =>  G.T = inv(system) @ temp.T  =>  system @ G.T = temp.T
  g_t = _solve_with_factor(factor, temp.T)
  g = g_t.T
  return (g + g.T) / 2.0


def _rss_summed(beta: np.ndarray, state: AccumulationState) -> np.ndarray:
  """Summed residual sum of squares per output dimension."""
  cross = np.sum(beta * state.dty, axis=0)
  quad = np.sum(beta * (state.dtd @ beta), axis=0)
  return np.diag(state.yty) - 2.0 * cross + quad


def _solver_moments(
  state: AccumulationState,
  penalty: PenaltySpec,
) -> tuple[np.ndarray, np.ndarray, float]:
  """Return (dtd_view, dty_view, g_scale) for the active penalty convention.

  ``fixed_prior`` uses summed sufficient statistics unchanged.
  ``constant_per_sample`` uses per-sample means ``(DtD/n, Dty/n)`` for the
  normal equations, hat trace, and ``G`` construction. The latter's sandwich
  term picks up an explicit ``1/n`` factor so downstream EB quantities match
  ``fixed_prior`` at ``n = n_ref``.
  """
  if penalty.mode == "fixed_prior":
    return state.dtd, state.dty, 1.0
  if penalty.mode == "constant_per_sample":
    n = float(state.n)
    return state.dtd / n, state.dty / n, 1.0 / n
  raise ValueError(f"unknown penalty mode: {penalty.mode}")


def solve_carrier(
  state: AccumulationState,
  plan: CarrierPlan,
  *,
  penalty: PenaltySpec | None = None,
  num_neurons: int | None = None,
) -> dict[str, Any]:
  """Solve carrier from sufficient statistics; raw calibration data not required."""
  if state.n < 2:
    raise ValueError("accumulation state must contain at least two samples")
  penalty = penalty or PenaltySpec.fixed_prior(plan.ridge_lambda)
  dtd_view, dty_view, g_scale = _solver_moments(state, penalty)
  system = build_system_matrix(state.dtd, state.n, penalty)
  factor = cho_factor(system, lower=True, check_finite=False)

  beta = _solve_with_factor(factor, dty_view)
  rss = _rss_summed(beta, state)
  hat_trace = _hat_trace(factor, dtd_view)
  denominator = float(state.n - hat_trace)
  if not np.isfinite(denominator) or denominator <= EPS:
    raise ValueError("ridge residual covariance degrees of freedom undefined")

  sigma2 = rss / denominator
  g = _compute_g(factor, dtd_view) * g_scale

  raw_rows = (plan.pcs[: plan.q].T @ beta[1:]) / plan.scale[:, None]
  raw_carrier = raw_rows @ plan.U
  projection = plan.pcs[: plan.q].T
  channel_factor = ((projection @ g[1:, 1:]) * projection).sum(axis=1) / np.square(plan.scale)
  projected_covariance = plan.U.T @ np.diag(sigma2) @ plan.U
  projected_variance = channel_factor * np.trace(projected_covariance) / 4.0

  if (
    np.any(~np.isfinite(projected_variance))
    or np.any(projected_variance < 0.0)
    or not np.isfinite(plan.tau2)
    or plan.tau2 <= EPS
  ):
    raise ValueError("analytic EB projected variance/prior undefined")

  weight = plan.tau2 / (plan.tau2 + projected_variance)
  carrier = plan.mu[None, :] + weight[:, None] * (raw_carrier - plan.mu[None, :])

  n_neurons = num_neurons if num_neurons is not None else raw_carrier.shape[0]
  if carrier.shape != (n_neurons, 4) or not np.isfinite(carrier).all():
    raise ValueError("deployable EB carrier must be finite [N,4]")

  return {
    "carrier": np.asarray(carrier, dtype=np.float64),
    "raw_carrier": np.asarray(raw_carrier, dtype=np.float64),
    "raw_rows": np.asarray(raw_rows, dtype=np.float64),
    "beta": np.asarray(beta, dtype=np.float64),
    "G": np.asarray(g, dtype=np.float64),
    "sigma2": np.asarray(sigma2, dtype=np.float64),
    "projected_variance": np.asarray(projected_variance, dtype=np.float64),
    "weight": np.asarray(weight, dtype=np.float64),
    "hat_trace": np.asarray(hat_trace, dtype=np.float64),
    "system": np.asarray(system, dtype=np.float64),
    "n": state.n,
    "DtD": np.asarray(state.dtd, dtype=np.float64),
    "Dty": np.asarray(state.dty, dtype=np.float64),
    "yty": np.asarray(state.yty, dtype=np.float64),
  }


def fit_carrier_batch(
  rates: np.ndarray,
  labels: np.ndarray,
  plan: CarrierPlan,
  *,
  penalty: PenaltySpec | None = None,
) -> dict[str, Any]:
  """Convenience: accumulate then solve in canonical order."""
  state = accumulate_blocks(rates, labels, plan=plan)
  return solve_carrier(state, plan, penalty=penalty, num_neurons=rates.shape[1])
