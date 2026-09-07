"""Closed-form recursive least-squares carrier updater (CPU-only, no I/O).

Updates the cosine-tuning carrier ``[a, c, b]`` from scalar trial rates and
labelled or pseudo-labelled directions.  Intended as a standalone numerical
component for B8 screening; not wired into any training loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_FORGETTING_FACTOR = 1.0
DEFAULT_DEPARTURE_THRESHOLD = 2.0
NORM_EPSILON = 1.0e-12


class CarrierRLSError(ValueError):
    """Raised for invalid RLS inputs or state."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CarrierRLSError(message)


def design_row(theta_rad: float) -> np.ndarray:
    """Row ``[1, cos(theta), sin(theta)]`` for one trial."""
    theta = float(theta_rad)
    _require(math.isfinite(theta), "theta must be finite")
    return np.array([1.0, math.cos(theta), math.sin(theta)], dtype=np.float64)


def batch_ols_carrier(
    thetas_rad: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    """Batch OLS fit ``rate = b + a*cos(theta) + c*sin(theta)``; returns ``[a,c,m,b]``."""
    theta = np.asarray(thetas_rad, dtype=np.float64).reshape(-1)
    response = np.asarray(rates, dtype=np.float64).reshape(-1)
    _require(theta.size == response.size and theta.size >= 1, "thetas and rates length mismatch")
    _require(np.isfinite(theta).all() and np.isfinite(response).all(), "nonfinite OLS inputs")
    design = np.column_stack(
        [np.ones(theta.size, dtype=np.float64), np.cos(theta), np.sin(theta)]
    )
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    b, a, c = (float(value) for value in coefficients)
    m = float(math.hypot(a, c))
    return np.array([a, c, m, b], dtype=np.float64)


@dataclass
class CarrierRLSState:
    """Per-unit RLS state for ``[b, a, c]`` stored as ``beta = [b, a, c]``."""

    beta: np.ndarray
    precision: np.ndarray
    initial_ac: np.ndarray
    frozen: bool = False
    update_count: int = 0

    @property
    def carrier_t4(self) -> np.ndarray:
        """Return ``[a, c, m, b]`` matching production T4 layout."""
        b, a, c = (float(self.beta[0]), float(self.beta[1]), float(self.beta[2]))
        m = float(math.hypot(a, c))
        return np.array([a, c, m, b], dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta.tolist(),
            "initial_ac": self.initial_ac.tolist(),
            "frozen": bool(self.frozen),
            "update_count": int(self.update_count),
            "carrier_t4": self.carrier_t4.tolist(),
        }


class CarrierRecursiveEstimator:
    """RLS updater with optional forgetting and fail-closed departure freeze."""

    def __init__(
        self,
        beta_init: np.ndarray,
        *,
        precision_init: np.ndarray | None = None,
        forgetting_factor: float = DEFAULT_FORGETTING_FACTOR,
        departure_threshold: float = DEFAULT_DEPARTURE_THRESHOLD,
    ) -> None:
        beta = np.asarray(beta_init, dtype=np.float64).reshape(3)
        _require(np.isfinite(beta).all(), "beta_init must be finite")
        _require(
            0.0 < forgetting_factor <= 1.0 and math.isfinite(forgetting_factor),
            "forgetting_factor must be in (0, 1]",
        )
        _require(departure_threshold > 0.0 and math.isfinite(departure_threshold), "invalid threshold")
        if precision_init is None:
            precision = np.eye(3, dtype=np.float64) * 1.0e6
        else:
            precision = np.asarray(precision_init, dtype=np.float64)
            _require(precision.shape == (3, 3) and np.isfinite(precision).all(), "invalid precision_init")
        self.forgetting_factor = float(forgetting_factor)
        self.departure_threshold = float(departure_threshold)
        self._state = CarrierRLSState(
            beta=beta.copy(),
            precision=precision.copy(),
            initial_ac=beta[1:3].copy(),
            frozen=False,
            update_count=0,
        )

    @property
    def state(self) -> CarrierRLSState:
        return self._state

    def _departure_ratio(self) -> float:
        current = self._state.beta[1:3]
        initial = self._state.initial_ac
        norm_initial = float(np.linalg.norm(initial))
        norm_delta = float(np.linalg.norm(current - initial))
        if norm_initial <= NORM_EPSILON:
            return norm_delta if norm_delta > NORM_EPSILON else 0.0
        return norm_delta / norm_initial

    def maybe_freeze(self) -> bool:
        """Freeze if ``[a,c]`` departed too far from the initial fit."""
        if self._state.frozen:
            return True
        if self._departure_ratio() > self.departure_threshold:
            self._state.frozen = True
        return self._state.frozen

    def update(self, theta_rad: float, rate: float) -> bool:
        """Apply one RLS step; return False if frozen or inputs invalid."""
        if self._state.frozen:
            return False
        x = design_row(theta_rad)
        y = float(rate)
        _require(math.isfinite(y), "rate must be finite")
        lam = self.forgetting_factor
        precision = self._state.precision
        beta = self._state.beta
        px = precision @ x
        denom = lam + float(x @ px)
        _require(denom > 0.0 and math.isfinite(denom), "degenerate RLS denominator")
        gain = px / denom
        innovation = y - float(x @ beta)
        self._state.beta = beta + gain * innovation
        self._state.precision = (precision - np.outer(gain, px)) / lam
        self._state.update_count += 1
        self.maybe_freeze()
        return not self._state.frozen

    def run_sequence(self, thetas_rad: np.ndarray, rates: np.ndarray) -> CarrierRLSState:
        """Update over a sequence and return the terminal state."""
        theta = np.asarray(thetas_rad, dtype=np.float64).reshape(-1)
        response = np.asarray(rates, dtype=np.float64).reshape(-1)
        _require(theta.size == response.size, "sequence length mismatch")
        for t, r in zip(theta, response):
            self.update(float(t), float(r))
        return self._state
