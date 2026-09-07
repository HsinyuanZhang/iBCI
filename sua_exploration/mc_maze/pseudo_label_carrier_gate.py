"""Pseudo-label carrier constructibility gate (CPU-only scaffolding).

Measures whether a carrier refit from decoder-output pseudo-directions
recovers the true ``[a,c]`` block relative to a shuffle baseline.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mc_maze.d_optimal_calibration_design import (
    ac_cosine_per_unit,
    direction_indices_from_thetas,
    fit_carriers_from_selected_trials,
)
from mc_maze.d_optimal_calibration_design import CANONICAL_DIRECTIONS_RAD

# RT Stage-1 split-half reference (HANDOFF_MAINLINE_CLOSURE_20260811).
RT_SPLIT_HALF_REFERENCE_MEDIAN = 0.787119

# Frozen pass gates for the B8 CPU screen.
FROZEN_GATE_MEDIAN_COSINE_GE = 0.50
FROZEN_GATE_CORRECT_MINUS_SHUFFLE_GE = 0.01
FROZEN_GATE_FRACTION_GE_040_GE = 0.50

DEFAULT_BUDGET_M = 30
DEFAULT_SHUFFLE_SEED = 42
SHUFFLE_NAMESPACE = "pseudo-label-carrier-gate-v1"


class PseudoLabelGateError(ValueError):
    """Raised for invalid pseudo-label gate inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PseudoLabelGateError(message)


def deterministic_label_shuffle(
    n_trials: int,
    *,
    session_name: str,
    seed: int = DEFAULT_SHUFFLE_SEED,
) -> np.ndarray:
    """Deterministic pseudo-label permutation distinct from identity when possible."""
    _require(n_trials >= 2, "shuffle needs at least two trials")
    digest = hashlib.sha256(f"{SHUFFLE_NAMESPACE}:{session_name}:{seed}".encode()).digest()
    shift = 1 + int.from_bytes(digest[:8], "little") % (n_trials - 1)
    order = np.roll(np.arange(n_trials, dtype=np.int64), shift)
    _require(not np.array_equal(order, np.arange(n_trials)), "shuffle retained identity")
    return order


def thetas_from_direction_indices(direction_indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(direction_indices, dtype=np.int64).reshape(-1)
    return np.array([CANONICAL_DIRECTIONS_RAD[int(index)] for index in indices], dtype=np.float64)


@dataclass(frozen=True)
class PseudoLabelGateResult:
    budget_m: int
    session_name: str
    median_cosine_correct: float | None
    median_cosine_shuffled: float | None
    correct_minus_shuffle: float | None
    fraction_ge_040_correct: float | None
    defined_units_correct: int
    defined_units_shuffled: int
    shuffle_seed: int
    shuffle_shift: int
    gates: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget_m": int(self.budget_m),
            "session_name": self.session_name,
            "median_cosine_correct": self.median_cosine_correct,
            "median_cosine_shuffled": self.median_cosine_shuffled,
            "correct_minus_shuffle": self.correct_minus_shuffle,
            "fraction_ge_040_correct": self.fraction_ge_040_correct,
            "defined_units_correct": int(self.defined_units_correct),
            "defined_units_shuffled": int(self.defined_units_shuffled),
            "shuffle_seed": int(self.shuffle_seed),
            "shuffle_shift": int(self.shuffle_shift),
            "gates": dict(self.gates),
            "rt_split_half_reference_median": RT_SPLIT_HALF_REFERENCE_MEDIAN,
        }


def evaluate_pseudo_label_gate(
    trial_rates: np.ndarray,
    true_direction_indices: np.ndarray,
    pseudo_direction_indices: np.ndarray,
    *,
    budget_m: int = DEFAULT_BUDGET_M,
    session_name: str = "synthetic",
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
) -> PseudoLabelGateResult:
    """Compare true-label vs pseudo-label carrier ``[a,c]`` cosines."""
    rates = np.asarray(trial_rates, dtype=np.float64)
    true_dirs = np.asarray(true_direction_indices, dtype=np.int64).reshape(-1)
    pseudo_dirs = np.asarray(pseudo_direction_indices, dtype=np.int64).reshape(-1)
    _require(
        rates.ndim == 2 and true_dirs.size == rates.shape[0] and pseudo_dirs.size == rates.shape[0],
        "trial_rates and direction indices must align",
    )
    _require(budget_m > 0 and budget_m <= rates.shape[0], "invalid budget_m")

    selected = np.arange(budget_m, dtype=np.int64)
    true_carriers = fit_carriers_from_selected_trials(rates, true_dirs, selected)
    pseudo_carriers = fit_carriers_from_selected_trials(rates, pseudo_dirs, selected)

    shuffle_order = deterministic_label_shuffle(budget_m, session_name=session_name, seed=shuffle_seed)
    budget_dirs = pseudo_dirs[selected]
    shuffled_budget_dirs = budget_dirs[shuffle_order]
    shuffled_carriers = fit_carriers_from_selected_trials(
        rates[selected], shuffled_budget_dirs, np.arange(budget_m, dtype=np.int64)
    )

    cos_correct = ac_cosine_per_unit(true_carriers[:, :2], pseudo_carriers[:, :2])
    cos_shuffled = ac_cosine_per_unit(true_carriers[:, :2], shuffled_carriers[:, :2])
    finite_correct = cos_correct[np.isfinite(cos_correct)]
    finite_shuffled = cos_shuffled[np.isfinite(cos_shuffled)]

    median_correct = float(np.median(finite_correct)) if finite_correct.size else None
    median_shuffled = float(np.median(finite_shuffled)) if finite_shuffled.size else None
    delta = (
        float(median_correct - median_shuffled)
        if median_correct is not None and median_shuffled is not None
        else None
    )
    fraction_ge = (
        float(np.mean(finite_correct >= 0.40)) if finite_correct.size else None
    )

    gates = {
        "median_cosine_ge_050": bool(
            median_correct is not None and median_correct >= FROZEN_GATE_MEDIAN_COSINE_GE
        ),
        "correct_minus_shuffle_ge_001": bool(
            delta is not None and delta >= FROZEN_GATE_CORRECT_MINUS_SHUFFLE_GE
        ),
        "fraction_ge_040_ge_050": bool(
            fraction_ge is not None and fraction_ge >= FROZEN_GATE_FRACTION_GE_040_GE
        ),
        "all_predeclared_gates": False,
    }
    gates["all_predeclared_gates"] = all(
        gates[key] for key in (
            "median_cosine_ge_050",
            "correct_minus_shuffle_ge_001",
            "fraction_ge_040_ge_050",
        )
    )

    digest = hashlib.sha256(f"{SHUFFLE_NAMESPACE}:{session_name}:{shuffle_seed}".encode()).digest()
    shift = 1 + int.from_bytes(digest[:8], "little") % max(1, budget_m - 1)

    return PseudoLabelGateResult(
        budget_m=budget_m,
        session_name=session_name,
        median_cosine_correct=median_correct,
        median_cosine_shuffled=median_shuffled,
        correct_minus_shuffle=delta,
        fraction_ge_040_correct=fraction_ge,
        defined_units_correct=int(finite_correct.size),
        defined_units_shuffled=int(finite_shuffled.size),
        shuffle_seed=int(shuffle_seed),
        shuffle_shift=int(shift),
        gates=gates,
    )


def synthetic_tuning_rates(
    thetas_rad: np.ndarray,
    units: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Generate synthetic trial rates from known cosine tuning per unit."""
    theta = np.asarray(thetas_rad, dtype=np.float64).reshape(-1)
    rng = np.random.Generator(np.random.PCG64(seed))
    rates = np.zeros((theta.size, units), dtype=np.float64)
    for unit in range(units):
        a = float(rng.normal(3.0, 1.0))
        c = float(rng.normal(2.0, 1.0))
        b = float(rng.normal(5.0, 0.5))
        rates[:, unit] = b + a * np.cos(theta) + c * np.sin(theta)
    return rates


FROZEN_GATE_PARAMETERS = {
    "budget_m_default": DEFAULT_BUDGET_M,
    "shuffle_seed": DEFAULT_SHUFFLE_SEED,
    "shuffle_namespace": SHUFFLE_NAMESPACE,
    "gates": {
        "median_cosine_ge": FROZEN_GATE_MEDIAN_COSINE_GE,
        "correct_minus_shuffle_ge": FROZEN_GATE_CORRECT_MINUS_SHUFFLE_GE,
        "fraction_ge_040_ge": FROZEN_GATE_FRACTION_GE_040_GE,
    },
    "rt_split_half_reference_median": RT_SPLIT_HALF_REFERENCE_MEDIAN,
}
