"""D-optimal calibration trial selection for cosine tuning (CPU-only scaffolding).

Selects labelled calibration trials using only target-direction geometry.  Neural
rates enter only after selection, for carrier estimation endpoints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

TUNING_NUM_DIRECTIONS = 8
CANONICAL_DIRECTIONS_RAD: tuple[float, ...] = tuple(
    -3.0 * math.pi / 4.0 + k * (math.pi / 4.0) for k in range(TUNING_NUM_DIRECTIONS)
)

SelectionArm = Literal[
    "chronological_first_m",
    "d_optimal_prefix_k",
    "random_m_prefix_k",
    "random_m_span_matched_k",
]

ALGORITHM_ID = "greedy_forward_d_optimal_v1"
MIN_DISTINCT_DIRECTIONS = 3
SPAN_MATCH_TOLERANCE_FRACTION = 0.05
SPAN_MATCH_MIN_TRIALS_TOLERANCE = 1

# Primary B9 gate: paired D-optimal minus chronological carrier fidelity at low M.
FROZEN_PRIMARY_GATE_BUDGETS: tuple[int, ...] = (10, 15)
FROZEN_PRIMARY_GATE_MIN_MEDIAN_DELTA = 0.03

# Fields that must never influence trial selection (leakage boundary).
LEAKAGE_FORBIDDEN_TRIAL_KEYS = frozenset(
    {
        "trial_rates",
        "rates",
        "rate",
        "unit_rates",
        "query_rates",
        "decoder_output",
        "decoder_direction",
        "pseudo_label",
        "pseudo_direction",
        "neural",
        "spikes",
        "activity",
        "query_window",
        "held_out",
    }
)


class DOptimalDesignError(ValueError):
    """Raised for invalid design or selection requests."""


class LeakageViolationError(DOptimalDesignError):
    """Raised when forbidden observables are passed to the selector."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DOptimalDesignError(message)


def assert_leakage_boundary(
    candidate_trials: Sequence[Mapping[str, Any]] | None = None,
    *,
    extra_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed if selection inputs include neural/decoder observables."""
    if candidate_trials is not None:
        for ordinal, trial in enumerate(candidate_trials):
            if not isinstance(trial, Mapping):
                continue
            forbidden = sorted(key for key in trial.keys() if key in LEAKAGE_FORBIDDEN_TRIAL_KEYS)
            if forbidden:
                raise LeakageViolationError(
                    f"trial {ordinal}: selector received forbidden keys {forbidden}; "
                    "D-optimal selection must use label geometry only"
                )
    if extra_kwargs is not None:
        forbidden = sorted(key for key in extra_kwargs if key in LEAKAGE_FORBIDDEN_TRIAL_KEYS)
        if forbidden:
            raise LeakageViolationError(
                f"selector kwargs contain forbidden keys {forbidden}"
            )


def _nearest_canonical_direction_index(target_dir_rad: float) -> int:
    directions = np.asarray(CANONICAL_DIRECTIONS_RAD, dtype=np.float64)
    wrapped = (directions - target_dir_rad + math.pi) % (2.0 * math.pi) - math.pi
    return int(np.argmin(np.abs(wrapped)))


def _fit_cosine_tuning(
    directions_rad: np.ndarray, mean_rates: np.ndarray
) -> tuple[float, float, float, float]:
    design = np.stack(
        [np.ones_like(directions_rad), np.cos(directions_rad), np.sin(directions_rad)], axis=1
    )
    coefficients, *_ = np.linalg.lstsq(design, mean_rates, rcond=None)
    b, a, c = (float(value) for value in coefficients)
    m = float(math.hypot(a, c))
    return a, c, m, b


def nearest_direction_index(theta_rad: float) -> int:
    return _nearest_canonical_direction_index(float(theta_rad))


def design_matrix_from_thetas(thetas_rad: np.ndarray) -> np.ndarray:
    """Build ``[1, cos(theta), sin(theta)]`` rows for each trial."""
    theta = np.asarray(thetas_rad, dtype=np.float64).reshape(-1)
    _require(theta.size > 0 and np.isfinite(theta).all(), "invalid thetas for design matrix")
    return np.column_stack(
        [np.ones(theta.size, dtype=np.float64), np.cos(theta), np.sin(theta)]
    )


def design_metrics(design: np.ndarray) -> dict[str, float | int | bool]:
    """Report ``det(X'X)``, condition number, and ``tr([(X'X)^-1]_{ac})``."""
    matrix = np.asarray(design, dtype=np.float64)
    _require(matrix.ndim == 2 and matrix.shape[1] == 3, "design must be [n_trials, 3]")
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != 3:
        return {
            "design_rank": rank,
            "design_condition": float("inf"),
            "det_xtx": 0.0,
            "covariance_trace_ac": float("inf"),
            "invertible": False,
        }
    gram = matrix.T @ matrix
    det = float(np.linalg.det(gram))
    condition = float(np.linalg.cond(matrix))
    inv_gram = np.linalg.inv(gram)
    trace_ac = float(np.trace(inv_gram[1:3, 1:3]))
    return {
        "design_rank": rank,
        "design_condition": condition,
        "det_xtx": det,
        "covariance_trace_ac": trace_ac,
        "invertible": bool(det > 0.0 and math.isfinite(condition)),
    }


def direction_indices_from_thetas(thetas_rad: np.ndarray) -> np.ndarray:
    theta = np.asarray(thetas_rad, dtype=np.float64).reshape(-1)
    return np.array([nearest_direction_index(float(value)) for value in theta], dtype=np.int64)


def require_min_distinct_directions(
    direction_indices: Sequence[int],
    *,
    min_distinct: int = MIN_DISTINCT_DIRECTIONS,
    session_name: str = "synthetic",
) -> int:
    """Refuse degenerate direction coverage (P5a fail-closed, >=3 for rank-3 design)."""
    present = len({int(index) for index in direction_indices})
    if present < 2:
        raise DOptimalDesignError(
            f"{session_name}: directional design degenerate: "
            f"present_directions={present} < 2; refusing silent all-zero carrier (P5a fail-closed)"
        )
    _require(
        present >= min_distinct,
        f"{session_name}: need >= {min_distinct} distinct directions, got {present}",
    )
    return present


def log_det_gram(design: np.ndarray, ridge: float = 1.0e-12) -> float:
    """Log-det of ``X'X`` with tiny ridge so greedy steps stay finite below rank 3."""
    matrix = np.asarray(design, dtype=np.float64)
    if matrix.size == 0:
        return 0.0
    gram = matrix.T @ matrix + ridge * np.eye(3, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(gram)
    if sign <= 0:
        return -math.inf
    return float(logdet)


def greedy_forward_d_optimal_indices(
    thetas_rad: np.ndarray,
    m: int,
) -> np.ndarray:
    """Greedy forward selection maximizing ``det(X'X)`` (via log-det gain)."""
    theta = np.asarray(thetas_rad, dtype=np.float64).reshape(-1)
    n = int(theta.size)
    _require(1 <= m <= n, f"cannot select m={m} from n={n} candidates")
    selected: list[int] = []
    remaining = set(range(n))
    for _ in range(m):
        best_index = -1
        best_score = -math.inf
        base_design = (
            design_matrix_from_thetas(theta[np.asarray(selected, dtype=np.int64)])
            if selected
            else None
        )
        base_score = log_det_gram(base_design) if base_design is not None else 0.0
        for index in remaining:
            trial_indices = selected + [index]
            trial_design = design_matrix_from_thetas(
                theta[np.asarray(trial_indices, dtype=np.int64)]
            )
            gain = log_det_gram(trial_design) - base_score
            if gain > best_score:
                best_score = gain
                best_index = index
        _require(best_index >= 0, "greedy D-optimal failed to select a trial")
        selected.append(best_index)
        remaining.remove(best_index)
    return np.asarray(selected, dtype=np.int64)


def random_subset_indices(n_candidates: int, m: int, seed: int) -> np.ndarray:
    _require(1 <= m <= n_candidates, "invalid random subset size")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    return np.sort(rng.choice(n_candidates, size=m, replace=False))


def index_span(selected_indices: np.ndarray) -> int:
    """Chronological span of selected pool trials: ``max(index) - min(index)``."""
    selected = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    _require(selected.size > 0, "span requires at least one selected trial")
    return int(selected.max() - selected.min())


def span_tolerance(target_span: int) -> int:
    relative = int(math.ceil(SPAN_MATCH_TOLERANCE_FRACTION * max(target_span, 1)))
    return max(SPAN_MATCH_MIN_TRIALS_TOLERANCE, relative)


def spans_match(selected_span: int, target_span: int) -> bool:
    tolerance = span_tolerance(target_span)
    return abs(int(selected_span) - int(target_span)) <= tolerance


def span_matched_random_indices(
    n_candidates: int,
    m: int,
    target_span: int,
    seed: int,
    *,
    max_attempts: int = 20_000,
) -> np.ndarray:
    """Random ``M`` subset whose index span matches ``target_span`` within tolerance."""
    _require(1 <= m <= n_candidates, "invalid span-matched subset size")
    if m <= 1:
        return random_subset_indices(n_candidates, m, seed)
    feasible_min = 0
    feasible_max = n_candidates - 1
    _require(
        target_span <= feasible_max - feasible_min,
        f"target span {target_span} infeasible for pool size {n_candidates}",
    )
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    tolerance = span_tolerance(target_span)
    for _ in range(max_attempts):
        low = rng.integers(0, n_candidates - target_span)
        high = low + target_span
        if high >= n_candidates:
            continue
        interior = rng.choice(np.arange(low + 1, high), size=m - 2, replace=False)
        selected = np.sort(np.concatenate(([low], interior, [high])))
        if selected.size != m:
            continue
        if spans_match(index_span(selected), target_span):
            return selected.astype(np.int64, copy=False)
    raise DOptimalDesignError(
        f"failed to draw span-matched random subset: m={m}, target_span={target_span}, "
        f"tolerance={tolerance}"
    )


def predict_rates_from_carrier(
    carriers: np.ndarray,
    thetas_rad: np.ndarray,
) -> np.ndarray:
    """Predict per-unit rates from ``[a,c,m,b]`` rows and trial directions."""
    t4 = np.asarray(carriers, dtype=np.float64)
    theta = np.asarray(thetas_rad, dtype=np.float64).reshape(-1)
    _require(t4.ndim == 2 and t4.shape[1] == 4, "carriers must be [units, 4]")
    units = int(t4.shape[0])
    predicted = np.zeros((theta.size, units), dtype=np.float64)
    for unit in range(units):
        a, c, b = float(t4[unit, 0]), float(t4[unit, 1]), float(t4[unit, 3])
        predicted[:, unit] = b + a * np.cos(theta) + c * np.sin(theta)
    return predicted


def per_unit_rmse(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    obs = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    _require(obs.shape == pred.shape, "RMSE shape mismatch")
    return np.sqrt(np.mean((obs - pred) ** 2, axis=0))


def evaluate_carrier_fidelity(
    trial_rates: np.ndarray,
    direction_indices: np.ndarray,
    thetas_rad: np.ndarray,
    selected_indices: np.ndarray,
    *,
    pool_size: int,
    reference_indices: np.ndarray | None = None,
    post_pool_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Held-out carrier fidelity vs full-pool reference and prediction RMSE."""
    rates = np.asarray(trial_rates, dtype=np.float64)
    directions = np.asarray(direction_indices, dtype=np.int64).reshape(-1)
    thetas = np.asarray(thetas_rad, dtype=np.float64).reshape(-1)
    selected = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    pool_size = int(pool_size)
    _require(rates.shape[0] >= pool_size, "rates shorter than declared pool")
    pool_mask = np.arange(pool_size, dtype=np.int64)
    ref_indices = (
        np.asarray(reference_indices, dtype=np.int64).reshape(-1)
        if reference_indices is not None
        else pool_mask
    )
    reference_carrier = fit_carriers_from_selected_trials(rates, directions, ref_indices)
    fit_carrier = fit_carriers_from_selected_trials(rates, directions, selected)
    cosine_ac = ac_cosine_per_unit(fit_carrier[:, :2], reference_carrier[:, :2])
    finite_cosine = cosine_ac[np.isfinite(cosine_ac)]

    unselected_pool = np.setdiff1d(pool_mask, selected, assume_unique=True)
    held_out_pool_rmse = None
    if unselected_pool.size > 0:
        obs = rates[unselected_pool]
        pred = predict_rates_from_carrier(fit_carrier, thetas[unselected_pool])
        rmse = per_unit_rmse(obs, pred)
        held_out_pool_rmse = {
            "median_per_unit_rmse": float(np.median(rmse)),
            "mean_per_unit_rmse": float(np.mean(rmse)),
            "trial_count": int(unselected_pool.size),
        }

    post_pool_rmse = None
    if post_pool_indices is not None:
        post = np.asarray(post_pool_indices, dtype=np.int64).reshape(-1)
        if post.size > 0:
            obs = rates[post]
            pred = predict_rates_from_carrier(fit_carrier, thetas[post])
            rmse = per_unit_rmse(obs, pred)
            post_pool_rmse = {
                "median_per_unit_rmse": float(np.median(rmse)),
                "mean_per_unit_rmse": float(np.mean(rmse)),
                "trial_count": int(post.size),
            }

    median_cosine = float(np.median(finite_cosine)) if finite_cosine.size else None
    mean_rmse = None
    if held_out_pool_rmse is not None:
        mean_rmse = float(held_out_pool_rmse["median_per_unit_rmse"])
    elif post_pool_rmse is not None:
        mean_rmse = float(post_pool_rmse["median_per_unit_rmse"])
    fidelity_score = None
    if median_cosine is not None and mean_rmse is not None:
        fidelity_score = float(median_cosine - mean_rmse)
    elif median_cosine is not None:
        fidelity_score = float(median_cosine)

    return {
        "median_cosine_ac_vs_reference": median_cosine,
        "fraction_cosine_ge_040": (
            float(np.mean(finite_cosine >= 0.40)) if finite_cosine.size else None
        ),
        "defined_units": int(finite_cosine.size),
        "held_out_pool_rmse": held_out_pool_rmse,
        "post_pool_rmse": post_pool_rmse,
        "carrier_fidelity_score": fidelity_score,
        "reference_trial_count": int(ref_indices.size),
        "selected_trial_count": int(selected.size),
    }


def time_span_record(
    selected_indices: np.ndarray,
    *,
    trial_times: np.ndarray | None = None,
) -> dict[str, float | int | list[int]]:
    """Expose selected indices and temporal coverage for coverage-confound audits."""
    selected = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    record: dict[str, float | int | list[int]] = {
        "selected_indices": [int(value) for value in selected],
        "index_span": index_span(selected),
        "min_index": int(selected.min()),
        "max_index": int(selected.max()),
    }
    if trial_times is not None:
        times = np.asarray(trial_times, dtype=np.float64).reshape(-1)
        selected_times = times[selected]
        record.update(
            {
                "min_time": float(selected_times.min()),
                "max_time": float(selected_times.max()),
                "time_span": float(selected_times.max() - selected_times.min()),
            }
        )
    return record


@dataclass(frozen=True)
class TrialSelectionResult:
    arm: SelectionArm
    selected_indices: tuple[int, ...]
    candidate_pool_size: int
    budget_m: int
    algorithm_id: str
    direction_indices: tuple[int, ...]
    design_metrics: dict[str, float | int | bool]
    temporal_coverage: dict[str, float | int | list[int]]
    leakage_assertion_passed: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "selected_indices": list(self.selected_indices),
            "candidate_pool_size": int(self.candidate_pool_size),
            "budget_m": int(self.budget_m),
            "algorithm_id": self.algorithm_id,
            "direction_indices": list(self.direction_indices),
            "design_metrics": dict(self.design_metrics),
            "temporal_coverage": dict(self.temporal_coverage),
            "leakage_assertion_passed": bool(self.leakage_assertion_passed),
        }


def select_calibration_trials(
    candidate_thetas_rad: np.ndarray,
    budget_m: int,
    arm: SelectionArm,
    *,
    seed: int = 42,
    session_name: str = "synthetic",
    candidate_trials_for_leakage_check: Sequence[Mapping[str, Any]] | None = None,
    span_match_target: int | None = None,
    trial_times: np.ndarray | None = None,
) -> TrialSelectionResult:
    """Select trial indices from a label-only candidate pool."""
    assert_leakage_boundary(candidate_trials_for_leakage_check)
    thetas = np.asarray(candidate_thetas_rad, dtype=np.float64).reshape(-1)
    n = int(thetas.size)
    _require(budget_m > 0 and n >= budget_m, f"pool size {n} < budget {budget_m}")
    direction_indices = direction_indices_from_thetas(thetas)
    require_min_distinct_directions(direction_indices, session_name=session_name)

    if arm == "chronological_first_m":
        selected = np.arange(budget_m, dtype=np.int64)
        algorithm_id = "chronological_first_m"
    elif arm == "d_optimal_prefix_k":
        selected = greedy_forward_d_optimal_indices(thetas, budget_m)
        algorithm_id = ALGORITHM_ID
    elif arm == "random_m_prefix_k":
        selected = random_subset_indices(n, budget_m, seed)
        algorithm_id = "random_m_prefix_k"
    elif arm == "random_m_span_matched_k":
        _require(span_match_target is not None, "span-matched arm requires span_match_target")
        selected = span_matched_random_indices(n, budget_m, int(span_match_target), seed)
        algorithm_id = "random_m_span_matched_k"
    else:
        raise DOptimalDesignError(f"unknown arm {arm!r}")

    selected_thetas = thetas[selected]
    selected_directions = direction_indices[selected]
    require_min_distinct_directions(
        selected_directions,
        min_distinct=min(MIN_DISTINCT_DIRECTIONS, min(budget_m, n)),
        session_name=f"{session_name}:{arm}:M{budget_m}",
    )
    metrics = design_metrics(design_matrix_from_thetas(selected_thetas))
    coverage = time_span_record(selected, trial_times=trial_times)
    return TrialSelectionResult(
        arm=arm,
        selected_indices=tuple(int(value) for value in selected),
        candidate_pool_size=n,
        budget_m=budget_m,
        algorithm_id=algorithm_id,
        direction_indices=tuple(int(value) for value in selected_directions),
        design_metrics=metrics,
        temporal_coverage=coverage,
        leakage_assertion_passed=True,
    )


def fit_carriers_from_selected_trials(
    trial_rates: np.ndarray,
    direction_indices: np.ndarray,
    selected_indices: np.ndarray,
) -> np.ndarray:
    """Fit ``[a,c,m,b]`` per unit from selected trials via per-direction means."""
    rates = np.asarray(trial_rates, dtype=np.float64)
    directions = np.asarray(direction_indices, dtype=np.int64).reshape(-1)
    selected = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    _require(
        rates.ndim == 2 and directions.size == rates.shape[0],
        "trial_rates must be [trials, units] aligned with direction_indices",
    )
    sel_dirs = directions[selected]
    sel_rates = rates[selected]
    units = int(rates.shape[1])
    carriers = np.zeros((units, 4), dtype=np.float64)
    present = sorted({int(index) for index in sel_dirs})
    thetas = np.array([CANONICAL_DIRECTIONS_RAD[index] for index in present], dtype=np.float64)
    for unit in range(units):
        per_direction = []
        for direction_index in present:
            mask = sel_dirs == direction_index
            per_direction.append(float(sel_rates[mask, unit].mean()))
        a, c, m, b = _fit_cosine_tuning(thetas, np.asarray(per_direction, dtype=np.float64))
        carriers[unit] = [a, c, m, b]
    return carriers


def ac_cosine_per_unit(first_ac: np.ndarray, second_ac: np.ndarray) -> np.ndarray:
    """Cosine similarity between ``[a,c]`` pairs per unit."""
    left = np.asarray(first_ac, dtype=np.float64)
    right = np.asarray(second_ac, dtype=np.float64)
    _require(left.shape == right.shape and left.ndim == 2 and left.shape[1] == 2, "invalid ac shape")
    norm_left = np.linalg.norm(left, axis=1)
    norm_right = np.linalg.norm(right, axis=1)
    defined = (norm_left > 1.0e-12) & (norm_right > 1.0e-12)
    cosines = np.full(left.shape[0], np.nan, dtype=np.float64)
    cosines[defined] = np.sum(left[defined] * right[defined], axis=1) / (
        norm_left[defined] * norm_right[defined]
    )
    return cosines


def split_half_carrier_reproducibility(
    trial_rates: np.ndarray,
    direction_indices: np.ndarray,
    selected_indices: np.ndarray,
) -> dict[str, Any]:
    """Split selected trials in half; compare ``[a,c]`` cosine per unit."""
    selected = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    _require(selected.size >= 4, "split-half needs at least four selected trials")
    midpoint = selected.size // 2
    first = selected[:midpoint]
    second = selected[midpoint:]
    carriers_first = fit_carriers_from_selected_trials(trial_rates, direction_indices, first)
    carriers_second = fit_carriers_from_selected_trials(trial_rates, direction_indices, second)
    cosines = ac_cosine_per_unit(carriers_first[:, :2], carriers_second[:, :2])
    finite = cosines[np.isfinite(cosines)]
    return {
        "first_half_size": int(first.size),
        "second_half_size": int(second.size),
        "defined_units": int(finite.size),
        "median_cosine_ac": float(np.median(finite)) if finite.size else None,
        "fraction_ge_040": float(np.mean(finite >= 0.40)) if finite.size else None,
    }


# Frozen budget ladder for B9 replay (matches measured label-budget curve sessions).
DEFAULT_BUDGET_LADDER: tuple[int, ...] = (10, 15, 20, 30, 50)
DEFAULT_CANDIDATE_POOL_K = 50

FROZEN_ALGORITHM_PARAMETERS = {
    "algorithm_id": ALGORITHM_ID,
    "selection_method": "greedy_forward_logdet",
    "min_distinct_directions": MIN_DISTINCT_DIRECTIONS,
    "candidate_pool_k": DEFAULT_CANDIDATE_POOL_K,
    "budget_ladder": list(DEFAULT_BUDGET_LADDER),
    "arms": [
        "chronological_first_m",
        "d_optimal_prefix_k",
        "random_m_prefix_k",
        "random_m_span_matched_k",
    ],
    "random_null_seeds": [42, 43, 44],
    "span_match_tolerance_fraction": SPAN_MATCH_TOLERANCE_FRACTION,
    "span_match_min_trials_tolerance": SPAN_MATCH_MIN_TRIALS_TOLERANCE,
    "primary_gate": {
        "endpoint": "carrier_fidelity_score_delta_d_optimal_minus_chronological",
        "budgets": list(FROZEN_PRIMARY_GATE_BUDGETS),
        "minimum_median_session_delta": FROZEN_PRIMARY_GATE_MIN_MEDIAN_DELTA,
        "negative_result_closes_branch": True,
    },
    "implementation_sanity_checks": {
        "det_xtx_gain_positive": "D-optimal greedy maximizes det by construction; not a primary gate",
        "condition_improves": "condition ratio expected >1 when det improves; sanity only",
        "random_null_det_separation": "sanity only",
    },
}
