"""H1 event-conditioning sweep: fixed k, varied latent-space support diversity."""
from __future__ import annotations

import hashlib
import importlib.util
from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2


SCHEMA = "h1_event_conditioning_sweep_v1"
PROTOCOL = "h1_event_conditioning_sweep_20260812_v1"
MODULE_NAME = "h1_event_conditioning_sweep"
BUDGET_TRIALS = 4
SUBSET_SIZES: tuple[int, ...] = (10, 14)
SELECTION_RULES: tuple[str, ...] = (
    "first",
    "maxspread",
    "minspread",
    "random_s0",
    "random_s1",
    "random_s2",
    "random_s3",
    "random_s4",
)
HSE5_TOLERANCE = 1.0e-10
_SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
if _SCIPY_AVAILABLE:
    from scipy import stats as scipy_stats


def _tie_break_lowest(scores: np.ndarray, *, higher_is_better: bool) -> int:
    if higher_is_better:
        best = np.max(scores)
        candidates = np.flatnonzero(scores == best)
    else:
        best = np.min(scores)
        candidates = np.flatnonzero(scores == best)
    return int(np.min(candidates))


def select_first(_z: np.ndarray, k: int) -> np.ndarray:
    v1._need(k >= 1, "first selection requires k >= 1")
    return np.arange(k, dtype=np.int64)


def select_maxspread(z: np.ndarray, k: int) -> np.ndarray:
    count = z.shape[0]
    v1._need(1 <= k <= count, f"maxspread selection requires 1 <= k <= {count}")
    norms = np.linalg.norm(z, axis=1)
    selected = [ _tie_break_lowest(norms, higher_is_better=True) ]
    while len(selected) < k:
        remaining = np.asarray([index for index in range(count) if index not in selected], dtype=np.int64)
        min_distances = np.asarray([
            np.min(np.linalg.norm(z[index] - z[selected], axis=1))
            for index in remaining
        ], dtype=np.float64)
        order = np.lexsort((remaining, -min_distances))
        selected.append(int(remaining[order[0]]))
    return np.asarray(selected, dtype=np.int64)


def select_minspread(z: np.ndarray, k: int) -> np.ndarray:
    count = z.shape[0]
    v1._need(1 <= k <= count, f"minspread selection requires 1 <= k <= {count}")
    pairwise = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
    summed = pairwise.sum(axis=1)
    selected = [ _tie_break_lowest(summed, higher_is_better=False) ]
    while len(selected) < k:
        remaining = np.asarray([index for index in range(count) if index not in selected], dtype=np.int64)
        centroid = z[selected].mean(axis=0)
        distances = np.linalg.norm(z[remaining] - centroid[None, :], axis=1)
        order = np.lexsort((remaining, distances))
        selected.append(int(remaining[order[0]]))
    return np.asarray(selected, dtype=np.int64)


def _random_seed(*, session: str, budget: int, k: int, replicate: int) -> int:
    key = f"{MODULE_NAME}:random:{session}:M{budget}:k{k}:s{replicate}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


def select_random(z: np.ndarray, k: int, *, session: str, budget: int, replicate: int) -> np.ndarray:
    count = z.shape[0]
    v1._need(1 <= k <= count, f"random selection requires 1 <= k <= {count}")
    rng = np.random.default_rng(_random_seed(session=session, budget=budget, k=k, replicate=replicate))
    return np.sort(rng.choice(count, size=k, replace=False).astype(np.int64))


def selection_indices(
    rule: str,
    z: np.ndarray,
    *,
    session: str,
    budget: int,
    k: int,
) -> np.ndarray:
    if rule == "first":
        return select_first(z, k)
    if rule == "maxspread":
        return select_maxspread(z, k)
    if rule == "minspread":
        return select_minspread(z, k)
    if rule.startswith("random_s"):
        replicate = int(rule.removeprefix("random_s"))
        v1._need(0 <= replicate <= 4, f"unknown random replicate {rule!r}")
        return select_random(z, k, session=session, budget=budget, replicate=replicate)
    raise v1.SparseEventEndpointError(f"unknown selection rule {rule!r}")


def median_carrier_weight_cosine(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    w_ref = np.asarray(reference, dtype=np.float64)[:, : v2.LATENT_DIM]
    w_cand = np.asarray(candidate, dtype=np.float64)[:, : v2.LATENT_DIM]
    n_ref = np.linalg.norm(w_ref, axis=1)
    n_cand = np.linalg.norm(w_cand, axis=1)
    defined = (n_ref > v1.NORM_FLOOR) & (n_cand > v1.NORM_FLOOR)
    if not np.any(defined):
        return None
    cosines = np.sum(w_ref[defined] * w_cand[defined], axis=1) / (n_ref[defined] * n_cand[defined])
    return float(np.median(cosines))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        average = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    v1._need(x.size == y.size and x.size >= 2, "Pearson correlation requires at least two paired values")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    v1._need(denominator > 0.0, "Pearson correlation denominator is zero")
    return float(np.sum(x_centered * y_centered) / denominator)


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_correlation(_average_ranks(x), _average_ranks(y))


def rank_correlations(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if _SCIPY_AVAILABLE:
        spearman_value = float(scipy_stats.spearmanr(x_arr, y_arr).statistic)
        pearson_value = float(scipy_stats.pearsonr(x_arr, y_arr).statistic)
        implementation = "scipy.stats"
    else:
        spearman_value = spearman_correlation(x_arr, y_arr)
        pearson_value = pearson_correlation(x_arr, y_arr)
        implementation = "numpy"
    return {
        "implementation": implementation,
        "spearman": spearman_value,
        "pearson": pearson_value,
        "n": int(x_arr.size),
    }


def evaluate_full_support(
    session: v1.EventSession,
    basis: v2.EndpointBasisV2,
    *,
    budget: int = BUDGET_TRIALS,
) -> dict[str, Any]:
    support = v2.select_trial_range(session, start_index=0, budget=budget)
    later = tuple(event for event in session.events if event.trial_index >= budget)
    if len(support) < 8 or len(later) < 4:
        return {
            "status": "undefined_event_count",
            "support_events": len(support),
            "later_events": len(later),
        }
    z, response = v2.event_arrays(support, basis)
    design = np.column_stack((np.ones(z.shape[0]), z))
    design_rank = int(np.linalg.matrix_rank(design))
    if design_rank != v2.CARRIER_DIM:
        return {
            "status": "undefined_rank_deficient",
            "support_events": len(support),
            "design_rank": design_rank,
        }
    carrier = v2.fit_carrier_arrays(z, response)
    z_later, observed = v2.event_arrays(later, basis)
    support_mean = np.mean(np.stack([event.log_rates for event in support]), axis=0)
    r_correct = v1.r2_by_channel(observed, v2.predict(carrier, z_later))
    r_intercept = v1.r2_by_channel(observed, np.broadcast_to(support_mean, observed.shape))
    defined = np.isfinite(r_correct) & np.isfinite(r_intercept)
    condition_number = float(np.linalg.cond(design))
    return {
        "status": "defined" if np.any(defined) else "undefined_channel_variance",
        "support_events": len(support),
        "later_events": len(later),
        "defined_channels": int(np.sum(defined)),
        "design_rank": design_rank,
        "condition_number": condition_number,
        "log10_condition_number": float(np.log10(condition_number)),
        "median_r2_correct": float(np.median(r_correct[defined])) if np.any(defined) else None,
        "median_delta_intercept": float(np.median((r_correct - r_intercept)[defined])) if np.any(defined) else None,
        "carrier_sha256": v1.array_sha256(carrier),
    }


def evaluate_subset_cell(
    session: v1.EventSession,
    basis: v2.EndpointBasisV2,
    *,
    k: int,
    rule: str,
    indices: np.ndarray,
    reference_carrier: np.ndarray,
    budget: int = BUDGET_TRIALS,
) -> dict[str, Any]:
    support = v2.select_trial_range(session, start_index=0, budget=budget)
    later = tuple(event for event in session.events if event.trial_index >= budget)
    if len(support) < k:
        return {
            "status": "undefined_insufficient_events",
            "session": session.session_name,
            "k": k,
            "rule": rule,
            "support_events_available": len(support),
        }
    if len(later) < 4:
        return {
            "status": "undefined_event_count",
            "session": session.session_name,
            "k": k,
            "rule": rule,
            "support_events": len(support),
            "later_events": len(later),
        }
    subset = tuple(support[int(index)] for index in indices)
    z, response = v2.event_arrays(subset, basis)
    design = np.column_stack((np.ones(z.shape[0]), z))
    design_rank = int(np.linalg.matrix_rank(design))
    if design_rank != v2.CARRIER_DIM:
        return {
            "status": "undefined_rank_deficient",
            "session": session.session_name,
            "k": k,
            "rule": rule,
            "support_events": len(subset),
            "design_rank": design_rank,
            "selected_indices": indices.tolist(),
        }
    try:
        carrier = v2.fit_carrier_arrays(z, response)
    except v1.SparseEventEndpointError as error:
        return {
            "status": "undefined_rank_deficient",
            "session": session.session_name,
            "k": k,
            "rule": rule,
            "support_events": len(subset),
            "design_rank": design_rank,
            "reason": str(error),
            "selected_indices": indices.tolist(),
        }
    z_later, observed = v2.event_arrays(later, basis)
    support_mean = np.mean(np.stack([event.log_rates for event in subset]), axis=0)
    r_correct = v1.r2_by_channel(observed, v2.predict(carrier, z_later))
    r_intercept = v1.r2_by_channel(observed, np.broadcast_to(support_mean, observed.shape))
    defined = np.isfinite(r_correct) & np.isfinite(r_intercept)
    condition_number = float(np.linalg.cond(design))
    return {
        "status": "defined" if np.any(defined) else "undefined_channel_variance",
        "session": session.session_name,
        "k": k,
        "rule": rule,
        "support_events": len(subset),
        "later_events": len(later),
        "defined_channels": int(np.sum(defined)),
        "selected_indices": indices.tolist(),
        "design_rank": design_rank,
        "condition_number": condition_number,
        "log10_condition_number": float(np.log10(condition_number)),
        "median_r2_correct": float(np.median(r_correct[defined])) if np.any(defined) else None,
        "median_delta_intercept": float(np.median((r_correct - r_intercept)[defined])) if np.any(defined) else None,
        "carrier_fidelity": median_carrier_weight_cosine(reference_carrier, carrier),
    }


def verify_full_support_reproduction(
    sessions: Mapping[str, v1.EventSession],
    bases: Mapping[str, v2.EndpointBasisV2],
    full_support_rows: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float = HSE5_TOLERANCE,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for name in v1.H1_HELDIN_SESSIONS:
        expected = v2.forward_transfer(sessions[name], bases[v1.session_date(name)], budget=BUDGET_TRIALS)
        observed = full_support_rows[name]
        v1._need(expected.get("status") == "defined", f"H-SE5 forward transfer undefined for {name}")
        v1._need(observed.get("status") == "defined", f"full-support reference undefined for {name}")
        for field in ("median_r2_correct", "median_delta_intercept"):
            delta = abs(float(observed[field]) - float(expected[field]))
            comparisons.append({
                "session": name,
                "field": field,
                "absolute_difference": delta,
                "within_tolerance": delta <= tolerance,
            })
            v1._need(delta <= tolerance, f"full-support reproduction failed for {name}.{field}: {delta}")
    maximum = max(row["absolute_difference"] for row in comparisons)
    return {
        "passed": True,
        "tolerance": tolerance,
        "maximum_absolute_difference": maximum,
        "comparisons": comparisons,
    }


def _aggregate_correlations(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    defined = [
        cell for cell in cells
        if cell.get("status") == "defined"
        and cell.get("log10_condition_number") is not None
        and cell.get("median_delta_intercept") is not None
        and cell.get("carrier_fidelity") is not None
    ]
    if len(defined) < 2:
        return {
            "defined_cells": len(defined),
            "log10_condition_number_vs_median_delta_intercept": None,
            "log10_condition_number_vs_carrier_fidelity": None,
        }
    log10_cond = np.asarray([float(cell["log10_condition_number"]) for cell in defined], dtype=np.float64)
    intercept = np.asarray([float(cell["median_delta_intercept"]) for cell in defined], dtype=np.float64)
    fidelity = np.asarray([float(cell["carrier_fidelity"]) for cell in defined], dtype=np.float64)
    return {
        "defined_cells": len(defined),
        "log10_condition_number_vs_median_delta_intercept": rank_correlations(log10_cond, intercept),
        "log10_condition_number_vs_carrier_fidelity": rank_correlations(log10_cond, fidelity),
    }


def _rule_aggregates(cells_by_session: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows_intercept: list[tuple[str, float | None]] = []
    rows_fidelity: list[tuple[str, float | None]] = []
    for name in v1.H1_HELDIN_SESSIONS:
        cell = cells_by_session[name]
        if cell.get("status") == "defined":
            rows_intercept.append((name, cell.get("median_delta_intercept")))
            rows_fidelity.append((name, cell.get("carrier_fidelity")))
    return {
        "defined_sessions": sum(1 for name in v1.H1_HELDIN_SESSIONS if cells_by_session[name].get("status") == "defined"),
        "median_delta_intercept": v1.paired_summary(rows_intercept),
        "carrier_fidelity": v1.paired_summary(rows_fidelity),
    }


def _paired_maxspread_minus_minspread(
    per_session: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    k: int,
) -> dict[str, Any]:
    key = str(k)
    rows_intercept: list[tuple[str, float | None]] = []
    rows_fidelity: list[tuple[str, float | None]] = []
    for name in v1.H1_HELDIN_SESSIONS:
        max_cell = per_session[name][key]["maxspread"]
        min_cell = per_session[name][key]["minspread"]
        if max_cell.get("status") == "defined" and min_cell.get("status") == "defined":
            max_intercept = max_cell.get("median_delta_intercept")
            min_intercept = min_cell.get("median_delta_intercept")
            max_fidelity = max_cell.get("carrier_fidelity")
            min_fidelity = min_cell.get("carrier_fidelity")
            if max_intercept is not None and min_intercept is not None:
                rows_intercept.append((name, float(max_intercept) - float(min_intercept)))
            if max_fidelity is not None and min_fidelity is not None:
                rows_fidelity.append((name, float(max_fidelity) - float(min_fidelity)))
    return {
        "median_delta_intercept": v1.paired_summary(rows_intercept),
        "carrier_fidelity": v1.paired_summary(rows_fidelity),
    }


def run_screen(sessions: Mapping[str, v1.EventSession]) -> dict[str, Any]:
    v1._need(tuple(sessions) == v1.H1_HELDIN_SESSIONS, "event conditioning sweep session allowlist/order drift")
    bases = {date: v2.fit_source_all_event_basis(sessions, outer_date=date) for date in v1.H1_DATES}
    full_support_rows: dict[str, dict[str, Any]] = {}
    reference_carriers: dict[str, np.ndarray] = {}
    per_session: dict[str, dict[str, dict[str, Any]]] = {}
    cells: list[dict[str, Any]] = []

    for name in v1.H1_HELDIN_SESSIONS:
        session = sessions[name]
        basis = bases[session.date]
        support = v2.select_trial_range(session, start_index=0, budget=BUDGET_TRIALS)
        z_support, response_support = v2.event_arrays(support, basis)
        full_row = evaluate_full_support(session, basis)
        full_support_rows[name] = full_row
        if full_row.get("status") == "defined":
            reference_carriers[name] = v2.fit_carrier_arrays(z_support, response_support)
        per_session[name] = {}
        for k in SUBSET_SIZES:
            key = str(k)
            per_session[name][key] = {}
            for rule in SELECTION_RULES:
                indices = selection_indices(rule, z_support, session=name, budget=BUDGET_TRIALS, k=k)
                cell = evaluate_subset_cell(
                    session,
                    basis,
                    k=k,
                    rule=rule,
                    indices=indices,
                    reference_carrier=reference_carriers.get(name, np.zeros((v1.EXPECTED_NEURONS, v2.CARRIER_DIM))),
                )
                per_session[name][key][rule] = cell
                cells.append(cell)

    reproduction = verify_full_support_reproduction(sessions, bases, full_support_rows)
    correlation_by_k: dict[str, Any] = {}
    rule_aggregates_by_k: dict[str, Any] = {}
    paired_contrast_by_k: dict[str, Any] = {}
    for k in SUBSET_SIZES:
        key = str(k)
        k_cells = [cell for cell in cells if cell.get("k") == k]
        correlation_by_k[key] = _aggregate_correlations(k_cells)
        rule_aggregates_by_k[key] = {
            rule: _rule_aggregates({name: per_session[name][key][rule] for name in v1.H1_HELDIN_SESSIONS})
            for rule in SELECTION_RULES
        }
        paired_contrast_by_k[key] = _paired_maxspread_minus_minspread(per_session, k=k)

    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "budget_trials": BUDGET_TRIALS,
        "subset_sizes": list(SUBSET_SIZES),
        "selection_rules": list(SELECTION_RULES),
        "rank_correlation_implementation": "scipy.stats" if _SCIPY_AVAILABLE else "numpy",
        "basis_by_outer_date": {date: bases[date].manifest() for date in v1.H1_DATES},
        "full_support_reference": full_support_rows,
        "cells": cells,
        "sessions": per_session,
        "correlation_by_k": correlation_by_k,
        "rule_aggregates_by_k": rule_aggregates_by_k,
        "paired_maxspread_minus_minspread_by_k": paired_contrast_by_k,
        "full_support_reproduction": reproduction,
        "scope": {
            "cuda_used": False,
            "decoder_constructed": False,
            "dense_velocity_series_opened": False,
            "public_held_in_calibration_nwbs_opened": 13,
        },
    }
