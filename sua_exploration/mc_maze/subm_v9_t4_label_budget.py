"""Small numerical guards and summaries for the frozen external sub-M T4 budget curve.

This module deliberately has no NWB/checkpoint/result I/O.  The runner uses it
to make the deployment question explicit: the B3 activity tensor is always
the first 30 rewarded trials; only the chronological label prefix used to fit
the already-trained T4 carrier changes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


HISTORY_BINS = 50
BUDGETS = (10, 15, 20, 30, 40)
REFERENCE_BUDGET = 50


class LabelBudgetError(RuntimeError):
    """Raised for an invalid/underidentified target-session T4 prefix."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LabelBudgetError(message)


@dataclass(frozen=True)
class T4DesignAudit:
    budget: int
    present_direction_indices: tuple[int, ...]
    direction_counts: Mapping[str, int]
    design_rank: int
    design_condition: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_t4_design(
    trials: Sequence[Mapping[str, Any]], budget: int, *, nearest_direction: Any, canonical_directions: Sequence[float]
) -> T4DesignAudit:
    """Fail closed unless the exact first-M cosine design has rank three.

    The production loader averages rate by canonical direction before fitting,
    hence the audit has the same unique-direction design rather than treating
    repeated trials as additional design rows.  It intentionally refuses a
    missing or non-finite target label instead of silently dropping it.
    """
    require(budget > 0 and len(trials) >= budget, f"T4 budget {budget} exceeds usable rewarded trials")
    indices: list[int] = []
    for ordinal, trial in enumerate(trials[:budget]):
        try:
            theta = float(trial["target_dir"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LabelBudgetError(f"trial {ordinal} has no finite target direction") from exc
        require(np.isfinite(theta), f"trial {ordinal} has nonfinite target direction")
        index = int(nearest_direction(theta))
        require(0 <= index < len(canonical_directions), f"trial {ordinal} direction mapping out of range")
        indices.append(index)
    present = tuple(sorted(set(indices)))
    counts = {str(index): int(indices.count(index)) for index in present}
    theta = np.asarray([float(canonical_directions[index]) for index in present], dtype=np.float64)
    design = np.column_stack((np.ones(theta.size, dtype=np.float64), np.cos(theta), np.sin(theta)))
    rank = int(np.linalg.matrix_rank(design))
    condition = float(np.linalg.cond(design)) if rank == 3 else float("inf")
    require(rank == 3 and np.isfinite(condition), f"T4 M={budget} design is underidentified: directions={present}, rank={rank}")
    return T4DesignAudit(
        budget=int(budget), present_direction_indices=present, direction_counts=counts,
        design_rank=rank, design_condition=condition,
    )


def hierarchical_bootstrap(delta: np.ndarray, *, seed: int = 688205, draws: int = 100_000) -> dict[str, float | int]:
    """Session-then-seed nonparametric bootstrap for a [session,seed] contrast."""
    values = np.asarray(delta, dtype=np.float64)
    require(values.ndim == 2 and values.shape[0] > 0 and values.shape[1] > 0, "bootstrap expects [session,seed]")
    require(np.isfinite(values).all(), "bootstrap delta contains nonfinite values")
    require(draws > 0, "bootstrap draws must be positive")
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(draws, dtype=np.float64)
    chunk = 10_000
    n_session, n_seed = values.shape
    for start in range(0, draws, chunk):
        stop = min(draws, start + chunk)
        session_index = rng.integers(0, n_session, size=(stop - start, n_session))
        seed_index = rng.integers(0, n_seed, size=(stop - start, n_session, n_seed))
        selected = values[session_index]
        selected = np.take_along_axis(selected, seed_index, axis=2)
        samples[start:stop] = selected.mean(axis=(1, 2), dtype=np.float64)
    lower, upper = np.quantile(samples, (0.025, 0.975), method="linear")
    return {"seed": int(seed), "draws": int(draws), "lower_95": float(lower), "upper_95": float(upper)}


def sign(value: float) -> str:
    return "+" if value > 0.0 else "-" if value < 0.0 else "0"


def within_003_summary(delta: np.ndarray) -> dict[str, Any]:
    """Predeclared deployment adequacy summaries for M versus frozen M50."""
    values = np.asarray(delta, dtype=np.float64)
    require(values.ndim == 2 and np.isfinite(values).all(), "invalid within-.03 delta")
    mean = float(values.mean(dtype=np.float64))
    seed_means = values.mean(axis=0, dtype=np.float64)
    return {
        "threshold_r2": -0.03,
        "mean_delta_at_least_minus_0_03": bool(mean >= -0.03),
        "all_three_seed_means_at_least_minus_0_03": bool(np.all(seed_means >= -0.03)),
        "point_and_three_seed_stable": bool(mean >= -0.03 and np.all(seed_means >= -0.03)),
    }
