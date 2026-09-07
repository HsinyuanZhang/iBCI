"""Pure/source-scoped statistics for the Step-2A T4 cross-budget audit.

The module deliberately does not know a formal-test path or a decoder.  Its NWB
count reader accepts an already-receipted first-M trial list; all later functions
operate on arrays and emit summary statistics only.  Raw count matrices are
caller-owned RAM objects and are never written by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from pynwb import NWBHDF5IO

from mc_maze.multisession_datamodule import electrode_ids_from_units
from mc_maze.t4_cross_budget_protocol import T4PrefixFit, fit_t4_prefix


REPEATABILITY_SEEDS: tuple[int, ...] = (1201, 1207, 1213, 1217, 1223, 1229, 1231, 1237)
SPLIT_PARTITIONS = 2
RIDGE_LAMBDA = 1.0e-4
_EPS = 1.0e-12


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def deterministic_nonidentity_row_shuffle(
    values: np.ndarray, *, session_name: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle complete descriptor rows with a deterministic non-identity permutation.

    The returned permutation, rather than value inequality alone, is the contract: duplicated
    descriptor rows can make a valid non-identity attachment shuffle look numerically unchanged.
    A one-unit session has no possible non-identity control and is rejected.
    """
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError("row-shuffle control requires at least two descriptor rows")
    rng = np.random.default_rng(stable_seed("step2a-row-shuffle", session_name, seed))
    identity = np.arange(array.shape[0])
    for _ in range(32):
        permutation = rng.permutation(array.shape[0])
        if not np.array_equal(permutation, identity):
            return array[permutation].copy(), permutation.astype(np.int64, copy=False)
    raise RuntimeError("failed to draw deterministic non-identity descriptor permutation")


def pool_trial_count_matrix_from_receipt(
    nwb_path: Path,
    *,
    trial_start_times: np.ndarray,
    trial_stop_times: np.ndarray,
    signal_view: str,
) -> np.ndarray:
    """Extract integer counts for exactly the already-receipted prefix into RAM.

    This is intentionally separate from the descriptor cache.  It does not inspect the NWB
    trial table, discover another trial, or write counts to disk.  The half-open interval is the
    same convention as `_pool_trial_rate_matrix`.
    """
    starts = np.asarray(trial_start_times, dtype=np.float64)
    stops = np.asarray(trial_stop_times, dtype=np.float64)
    if starts.ndim != 1 or stops.shape != starts.shape or starts.size == 0:
        raise ValueError("receipt start/stop vectors must be aligned nonempty rank-1 arrays")
    if np.any(~np.isfinite(starts)) or np.any(~np.isfinite(stops)) or np.any(stops <= starts):
        raise ValueError("receipt contains invalid trial boundaries")
    if np.any(np.diff(starts) < 0.0):
        raise ValueError("receipt is not chronological")
    if signal_view not in {"sua", "pseudo_mua"}:
        raise ValueError("signal_view must be 'sua' or 'pseudo_mua'")
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        if nwb.units is None:
            raise ValueError(f"NWB file has no units table: {nwb_path}")
        units = nwb.units.to_dataframe()
        counts = np.empty((len(units), starts.size), dtype=np.int64)
        for unit_idx, row in enumerate(units.itertuples(index=False)):
            spike_times = np.asarray(getattr(row, "spike_times"), dtype=np.float64)
            if spike_times.size and np.any(spike_times[:-1] > spike_times[1:]):
                raise ValueError(f"unit {unit_idx} spike times are not non-decreasing")
            counts[unit_idx] = np.searchsorted(spike_times, stops) - np.searchsorted(spike_times, starts)
        if signal_view == "pseudo_mua":
            electrode_ids = electrode_ids_from_units(units)
            channels, inverse = np.unique(electrode_ids, return_inverse=True)
            pooled = np.zeros((channels.size, starts.size), dtype=np.int64)
            np.add.at(pooled, inverse, counts)
            counts = pooled
    return counts


def disjoint_trial_count_partitions(
    counts: np.ndarray, *, seed: int, n_partitions: int = SPLIT_PARTITIONS
) -> np.ndarray:
    """Partition every trial's integer spikes into disjoint, equal-probability views.

    The views preserve each trial, its target label and chronological placement.  They are scaled
    back by `n_partitions` before fitting, so each rate estimate is unbiased for the full-rate
    T4 while the two views have no shared spikes.
    """
    input_counts = np.asarray(counts)
    if input_counts.ndim != 2 or not np.issubdtype(input_counts.dtype, np.integer):
        raise ValueError("counts must be a nonnegative integer [units,trials] matrix")
    if np.any(input_counts < 0) or n_partitions < 2:
        raise ValueError("counts must be nonnegative and require at least two partitions")
    rng = np.random.default_rng(seed)
    output = np.zeros((n_partitions,) + input_counts.shape, dtype=np.int64)
    # Vectorising multinomial over arbitrary count matrices is not supported by NumPy; this is
    # CPU-only and runs only for the at-most-50-trial calibration receipt.
    for unit_idx in range(input_counts.shape[0]):
        for trial_idx in range(input_counts.shape[1]):
            output[:, unit_idx, trial_idx] = rng.multinomial(
                int(input_counts[unit_idx, trial_idx]), np.full(n_partitions, 1.0 / n_partitions)
            )
    if not np.array_equal(output.sum(axis=0), input_counts):
        raise AssertionError("count partition failed conservation")
    return output


def _component_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 2:
        return None
    a, b = a[valid], b[valid]
    if np.std(a) <= _EPS or np.std(b) <= _EPS:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def split_half_t4_repeatability(
    counts: np.ndarray,
    durations_s: np.ndarray,
    direction_indices: np.ndarray,
    *,
    budget: int,
    seeds: Sequence[int] = REPEATABILITY_SEEDS,
) -> dict:
    """Summary-only disjoint within-trial T4 repeatability for one actual prefix.

    Any rank-deficient prefix/partition is marked undefined and excluded from component
    correlations; the output counts exclusions explicitly rather than imputing zeros.
    """
    durations = np.asarray(durations_s, dtype=np.float64)
    directions = np.asarray(direction_indices, dtype=np.int64)
    if counts.shape[1] != durations.size or directions.shape != durations.shape:
        raise ValueError("counts/durations/directions must share the trial axis")
    if budget > counts.shape[1]:
        raise ValueError("repeatability budget exceeds supplied receipt")
    records: list[dict] = []
    for seed in seeds:
        parts = disjoint_trial_count_partitions(counts[:, :budget], seed=int(seed))
        fits = [
            fit_t4_prefix(
                parts[index].astype(np.float64) * SPLIT_PARTITIONS / durations[None, :budget],
                durations[:budget], directions[:budget], budget=budget,
            )
            for index in range(SPLIT_PARTITIONS)
        ]
        if not all(fit.fit_defined for fit in fits):
            records.append({"seed": int(seed), "status": "undefined_rank"})
            continue
        left, right = fits[0].t4, fits[1].t4
        records.append(
            {
                "seed": int(seed), "status": "defined",
                "ac_flattened_pearson": _component_correlation(left[:, :2], right[:, :2]),
                "m_pearson": _component_correlation(left[:, 2], right[:, 2]),
                "b_pearson": _component_correlation(left[:, 3], right[:, 3]),
            }
        )
    defined = [row for row in records if row["status"] == "defined"]
    def median(name: str) -> float | None:
        values = [float(row[name]) for row in defined if row[name] is not None]
        return float(np.median(values)) if values else None
    return {
        "budget": int(budget), "partition": "disjoint_multinomial_two_way_rescaled_rate",
        "seed_count": len(records), "defined_replicates": len(defined),
        "undefined_replicates": len(records) - len(defined), "median": {
            "ac_flattened_pearson": median("ac_flattened_pearson"),
            "m_pearson": median("m_pearson"), "b_pearson": median("b_pearson"),
        }, "replicates": records,
    }


def descriptor_error_to_t50(low: T4PrefixFit, reference: T4PrefixFit) -> np.ndarray:
    """Per-unit squared descriptor error, with undefined fits rejected rather than imputed."""
    if not low.fit_defined or not reference.fit_defined:
        raise ValueError("descriptor error is undefined when either T4 fit is undefined")
    if low.t4.shape != reference.t4.shape:
        raise ValueError("low/reference T4 shapes differ")
    return np.mean(np.square(low.t4.astype(np.float64) - reference.t4.astype(np.float64)), axis=1)


def _fit_linear_ridge(x: np.ndarray, y: np.ndarray, ridge: float = RIDGE_LAMBDA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    mean = x.mean(axis=0); std = np.maximum(x.std(axis=0), _EPS)
    z = (x - mean) / std
    design = np.column_stack([np.ones(z.shape[0]), z])
    penalty = np.eye(design.shape[1]) * ridge; penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return mean, std, weights


def _predict_linear(model: tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    mean, std, weights = model
    design = np.column_stack([np.ones(len(x)), (np.asarray(x) - mean) / std])
    return design @ weights


@dataclass(frozen=True)
class CrossBudgetSessionAudit:
    """Source-session-only inputs for nested q-to-error prediction."""
    name: str
    fits: Mapping[int, T4PrefixFit]


def load_frozen_source_development_manifest(manifest_path: Path) -> dict:
    """Validate the frozen SUA development manifest without resolving/opening any NWB path.

    The returned nested-fold source set is the declared `train` split.  Validation names are
    retained as an unopened development boundary for a later, separately approved target-free
    application receipt; formal-test names are merely sealed strings and are never turned into
    paths here.
    """
    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read frozen source development manifest: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("frozen source development manifest must have schema_version=1")
    splits = payload.get("session_splits")
    if not isinstance(splits, dict):
        raise ValueError("frozen source development manifest lacks session_splits")
    expected = {"train": 27, "val": 6, "test": 6}
    clean: dict[str, tuple[str, ...]] = {}
    for split, count in expected.items():
        names = splits.get(split)
        if not isinstance(names, list) or len(names) != count or not all(isinstance(name, str) for name in names):
            raise ValueError(f"manifest {split} split must contain {count} string names")
        if len(set(names)) != count:
            raise ValueError(f"manifest {split} split contains duplicate names")
        clean[split] = tuple(names)
    if len(set(clean["train"] + clean["val"] + clean["test"])) != 39:
        raise ValueError("manifest splits must be pairwise disjoint")
    digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    return {
        "manifest_path": str(Path(manifest_path).resolve()), "manifest_sha256": digest,
        "nested_source_session_names": clean["train"], "development_validation_names": clean["val"],
        "sealed_formal_test_names": clean["test"], "formal_test_paths_resolved": False,
    }


def paired_mde(deltas: Sequence[float]) -> float | None:
    """Normal-approximation 80%-power, two-sided 5% paired MDE in endpoint units."""
    values = np.asarray(deltas, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    return float((1.959963984540054 + 0.8416212335729143) * values.std(ddof=1) / math.sqrt(values.size))


def _rows_for_q_error(sessions: Sequence[CrossBudgetSessionAudit], budgets: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[np.ndarray, float, float]] = []
    for session in sessions:
        reference = session.fits[50]
        for budget in budgets:
            budget = int(budget)
            if budget >= 50:
                continue
            low = session.fits[budget]
            if not (low.fit_defined and reference.fit_defined):
                continue
            for q, target in zip(low.reliability, descriptor_error_to_t50(low, reference), strict=True):
                rows.append((q.astype(np.float64), float(budget), float(target)))
    if not rows:
        raise ValueError("no finite low-budget/reference descriptor-error rows")
    return (
        np.stack([row[0] for row in rows]),
        np.asarray([row[1] for row in rows], dtype=np.float64)[:, None],
        np.asarray([row[2] for row in rows], dtype=np.float64),
    )


def fit_source_q_error_models(
    sessions: Sequence[CrossBudgetSessionAudit], *, budgets: Sequence[int]
) -> dict:
    """Fit source-only q-error baselines for a later target-free validation application."""
    q, m, target = _rows_for_q_error(sessions, budgets)
    return {
        "constant": float(target.mean()), "m_only": _fit_linear_ridge(m, target),
        "q_plus_m": _fit_linear_ridge(np.column_stack([q, m]), target),
    }


def score_target_free_q_error_models(
    models: Mapping[str, object], session: CrossBudgetSessionAudit, *, budgets: Sequence[int]
) -> dict:
    """Score source-fitted q models; T4@50 is used only after prediction as offline target."""
    q, m, target = _rows_for_q_error([session], budgets)
    predictions = {
        "constant": np.full(target.shape, float(models["constant"])),
        "m_only": _predict_linear(models["m_only"], m),  # type: ignore[arg-type]
        "q_plus_m": _predict_linear(models["q_plus_m"], np.column_stack([q, m])),  # type: ignore[arg-type]
    }
    return {
        "status": "defined", "test_rows": int(target.size), "target_free_application": True,
        "mse": {key: float(np.mean(np.square(value - target))) for key, value in predictions.items()},
    }


def nested_source_q_error_audit(
    sessions: Sequence[CrossBudgetSessionAudit], *, budgets: Sequence[int]
) -> dict:
    """Nested source-LOSO q→T4@50 error prediction versus constant and M-only controls.

    T4@50 is used exclusively as the offline target.  At left-out application the learned q
    predictor receives only the left-out `T4@M` reliability and its known M code; it never gets
    that session's T4@50 vector.  The returned MSE is a descriptor proxy, not decoder R².
    """
    ordered = tuple(int(m) for m in budgets if int(m) < 50)
    if len(sessions) < 3 or not ordered:
        raise ValueError("nested source audit requires >=3 sessions and a low budget")
    per_session: dict[str, dict] = {}
    for heldout in sessions:
        train = [session for session in sessions if session.name != heldout.name]
        try:
            models = fit_source_q_error_models(train, budgets=ordered)
            scored = score_target_free_q_error_models(models, heldout, budgets=ordered)
        except ValueError:
            per_session[heldout.name] = {"status": "undefined_no_valid_t4"}
            continue
        per_session[heldout.name] = scored
    defined = [row for row in per_session.values() if row["status"] == "defined"]
    return {
        "endpoint": "per_unit_mean_squared_T4M_to_T450_descriptor_error",
        "formal_test_used": False, "nested_source_loso": True,
        "target_free_leftout_application": True, "per_session": per_session,
        "defined_session_count": len(defined),
        "mean_session_mse": {name: (float(np.mean([row["mse"][name] for row in defined])) if defined else None) for name in ("constant", "m_only", "q_plus_m")},
        "paired_mde_80pct_two_sided_5pct": {
            comparison: paired_mde([row["mse"][comparison] - row["mse"]["m_only"] for row in defined])
            for comparison in ("q_plus_m", "constant")
        },
    }
