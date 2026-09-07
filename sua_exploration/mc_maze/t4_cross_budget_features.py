"""Versioned NWB/cache boundary for Step-2A causal T4 prefix views.

Nothing in this module registers a side-feature token or touches a training datamodule.  It is
an isolated cache builder for the later CPU audit.  Cache entries contain fitted descriptors and
audit fields, never raw M-by-T support arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from pynwb import NWBHDF5IO

from mc_maze.multisession_datamodule import (
    _cache_key,
    _exclusive_cache_lock,
    _source_fingerprint,
    _write_npz_atomically,
    electrode_ids_from_units,
    list_datamodule_rewarded_trials,
    session_name_from_path,
)
from mc_maze.t4_cross_budget_protocol import (
    DEFAULT_T4_BUDGETS,
    T4_CROSS_BUDGET_FIT_SEMANTICS_VERSION,
    T4PrefixFit,
    canonical_direction_indices,
    fit_t4_prefixes,
    validate_budgets,
)
from mc_maze.unit_side_features import (
    _electrode_mapping_fingerprint,
    _pool_trial_rate_matrix,
    pool_trial_rates_by_electrode,
)


T4_CROSS_BUDGET_CACHE_NAMESPACE = "sua_t4_cross_budget_v2"
T4_CROSS_BUDGET_FEATURE_VERSION = 2


@dataclass(frozen=True)
class CrossBudgetFeatureBundle:
    """Cached causal T4 views, keyed by their actual first-M budgets."""

    source_path: str
    signal_view: str
    cache_key: str
    budgets: tuple[int, ...]
    fits: dict[int, T4PrefixFit]
    cache_payload: dict
    trial_ordinals: np.ndarray  # [max_budget], chronological zero-based rewarded-trial ordinal.
    trial_start_times: np.ndarray  # [max_budget], seconds.
    trial_stop_times: np.ndarray  # [max_budget], seconds.
    trial_target_dirs_rad: np.ndarray  # [max_budget], NaN denotes no finite label.
    trial_direction_indices: np.ndarray  # [max_budget], -1 denotes no finite/snap-valid label.


def _validate_signal_view(signal_view: str) -> None:
    if signal_view not in {"sua", "pseudo_mua"}:
        raise ValueError(f"signal_view must be 'sua' or 'pseudo_mua', got {signal_view!r}")


def cross_budget_cache_payload(
    nwb_path: Path,
    *,
    budgets: Sequence[int],
    bin_size_ms: int,
    window_size: int,
    trial_result_filter: str,
    signal_view: str,
) -> dict:
    """Full versioned cache provenance; signal view is always explicit."""
    checked = validate_budgets(budgets)
    _validate_signal_view(signal_view)
    payload: dict = {
        "cache_namespace": T4_CROSS_BUDGET_CACHE_NAMESPACE,
        "feature_version": T4_CROSS_BUDGET_FEATURE_VERSION,
        "fit_semantics_version": T4_CROSS_BUDGET_FIT_SEMANTICS_VERSION,
        "budgets": list(checked),
        "source": _source_fingerprint(nwb_path),
        "signal_view": signal_view,
        "bin_size_ms": int(bin_size_ms),
        "window_size": int(window_size),
        "trial_result_filter": str(trial_result_filter),
        "canonical_direction_count": 8,
    }
    if signal_view == "pseudo_mua":
        payload["electrode_mapping"] = _electrode_mapping_fingerprint(nwb_path)
    return payload


def cross_budget_cache_path(cache_dir: Path, nwb_path: Path, **kwargs: object) -> Path:
    payload = cross_budget_cache_payload(nwb_path, **kwargs)
    return (
        Path(cache_dir)
        / T4_CROSS_BUDGET_CACHE_NAMESPACE
        / "sessions"
        / f"{session_name_from_path(nwb_path)}_{_cache_key(payload)[:20]}.npz"
    )


def _trial_receipt_from_pool(pool_trials: Sequence[dict]) -> dict[str, np.ndarray]:
    """Return a compact, raw-rate-free chronology receipt for a selected prefix.

    The datamodule selector is the authority for trial membership/order.  We still fail closed if
    that order is not chronological, because the Step-2 hypothesis is explicitly about prefixes.
    """
    starts = np.asarray([float(row["start_time"]) for row in pool_trials], dtype=np.float64)
    stops = np.asarray([float(row["stop_time"]) for row in pool_trials], dtype=np.float64)
    if np.any(~np.isfinite(starts)) or np.any(~np.isfinite(stops)) or np.any(stops <= starts):
        raise ValueError("selected rewarded trials require finite strictly positive durations")
    if np.any(np.diff(starts) < 0.0):
        raise ValueError("datamodule rewarded-trial selector returned non-chronological starts")
    raw_dirs = np.asarray(
        [float(row["target_dir"]) if row.get("target_dir") is not None else np.nan for row in pool_trials],
        dtype=np.float64,
    )
    direction_indices = canonical_direction_indices(raw_dirs.tolist())
    return {
        "trial_ordinals": np.arange(len(pool_trials), dtype=np.int64),
        "trial_start_times": starts,
        "trial_stop_times": stops,
        "trial_target_dirs_rad": raw_dirs,
        "trial_direction_indices": direction_indices,
    }


def _fit_bundle_from_pool(
    nwb_path: Path,
    *,
    pool_trials: Sequence[dict],
    budgets: tuple[int, ...],
    signal_view: str,
    cache_key: str,
    cache_payload: dict,
) -> CrossBudgetFeatureBundle:
    receipt = _trial_receipt_from_pool(pool_trials)
    rates, _ = _pool_trial_rate_matrix(nwb_path, pool_trials)
    if signal_view == "pseudo_mua":
        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwb = io.read()
            if nwb.units is None:
                raise ValueError(f"NWB file has no units table: {nwb_path}")
            electrode_ids = electrode_ids_from_units(nwb.units.to_dataframe())
        rates, _ = pool_trial_rates_by_electrode(rates, electrode_ids)
    durations = np.asarray(
        [float(trial["stop_time"]) - float(trial["start_time"]) for trial in pool_trials],
        dtype=np.float64,
    )
    directions = receipt["trial_direction_indices"]
    fits = fit_t4_prefixes(rates, durations, directions, budgets=budgets)
    return CrossBudgetFeatureBundle(
        source_path=str(nwb_path.resolve()),
        signal_view=signal_view,
        cache_key=cache_key,
        budgets=budgets,
        fits=fits,
        cache_payload=cache_payload,
        **receipt,
    )


def compute_cross_budget_features_uncached(
    nwb_path: Path,
    *,
    budgets: Sequence[int] = DEFAULT_T4_BUDGETS,
    bin_size_ms: int = 20,
    window_size: int = 50,
    trial_result_filter: str = "R",
    signal_view: str = "sua",
) -> CrossBudgetFeatureBundle:
    """Compute real first-M refits using the datamodule's rewarded-trial selector."""
    checked = validate_budgets(budgets)
    _validate_signal_view(signal_view)
    pool_trials = list_datamodule_rewarded_trials(
        nwb_path,
        bin_size_ms=bin_size_ms,
        window_size=window_size,
        trial_result_filter=trial_result_filter,
    )
    if len(pool_trials) < checked[-1]:
        raise ValueError(
            f"{session_name_from_path(nwb_path)} has only {len(pool_trials)} rewarded trials; "
            f"largest T4 budget is {checked[-1]}"
        )
    payload = cross_budget_cache_payload(
        nwb_path,
        budgets=checked,
        bin_size_ms=bin_size_ms,
        window_size=window_size,
        trial_result_filter=trial_result_filter,
        signal_view=signal_view,
    )
    return _fit_bundle_from_pool(
        nwb_path,
        pool_trials=pool_trials[: checked[-1]],
        budgets=checked,
        signal_view=signal_view,
        cache_key=_cache_key(payload),
        cache_payload=payload,
    )


def _bundle_arrays(bundle: CrossBudgetFeatureBundle) -> dict[str, np.ndarray]:
    metadata = {
        "cache_namespace": T4_CROSS_BUDGET_CACHE_NAMESPACE,
        "feature_version": T4_CROSS_BUDGET_FEATURE_VERSION,
        "fit_semantics_version": T4_CROSS_BUDGET_FIT_SEMANTICS_VERSION,
        "source_path": bundle.source_path,
        "signal_view": bundle.signal_view,
        "cache_key": bundle.cache_key,
        "budgets": list(bundle.budgets),
        "cache_payload": bundle.cache_payload,
    }
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "budgets": np.asarray(bundle.budgets, dtype=np.int64),
        "trial_ordinals": bundle.trial_ordinals.astype(np.int64, copy=False),
        "trial_start_times": bundle.trial_start_times.astype(np.float64, copy=False),
        "trial_stop_times": bundle.trial_stop_times.astype(np.float64, copy=False),
        "trial_target_dirs_rad": bundle.trial_target_dirs_rad.astype(np.float64, copy=False),
        "trial_direction_indices": bundle.trial_direction_indices.astype(np.int64, copy=False),
    }
    for budget, fit in bundle.fits.items():
        prefix = f"m{budget}_"
        arrays.update(
            {
                prefix + "t4": fit.t4.astype(np.float32, copy=False),
                prefix + "reliability": fit.reliability.astype(np.float32, copy=False),
                prefix + "fit_defined": np.asarray(int(fit.fit_defined), dtype=np.int8),
                prefix + "design_rank": np.asarray(fit.design_rank, dtype=np.int64),
                prefix + "design_condition": np.asarray(fit.design_condition, dtype=np.float64),
                prefix + "direction_counts": fit.direction_counts.astype(np.int64, copy=False),
                prefix + "direction_balance": np.asarray(fit.direction_balance, dtype=np.float64),
                prefix + "zero_spike_unit_count": np.asarray(fit.zero_spike_unit_count, dtype=np.int64),
                prefix + "zero_modulation_unit_count": np.asarray(fit.zero_modulation_unit_count, dtype=np.int64),
            }
        )
    return arrays


def _load_bundle(cache_path: Path, *, expected_payload: dict | None = None) -> CrossBudgetFeatureBundle:
    with np.load(cache_path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata_json"].item()))
        if metadata.get("cache_namespace") != T4_CROSS_BUDGET_CACHE_NAMESPACE:
            raise ValueError(f"wrong cross-budget cache namespace: {cache_path}")
        if metadata.get("feature_version") != T4_CROSS_BUDGET_FEATURE_VERSION:
            raise ValueError(f"wrong cross-budget feature version: {cache_path}")
        if metadata.get("fit_semantics_version") != T4_CROSS_BUDGET_FIT_SEMANTICS_VERSION:
            raise ValueError(f"wrong cross-budget fit semantics: {cache_path}")
        payload = metadata.get("cache_payload")
        if not isinstance(payload, dict):
            raise ValueError(f"cross-budget cache omits canonical payload: {cache_path}")
        actual_key = _cache_key(payload)
        if metadata.get("cache_key") != actual_key:
            raise ValueError(f"cross-budget cache key/payload mismatch: {cache_path}")
        if expected_payload is not None and payload != expected_payload:
            raise ValueError(f"cross-budget cache payload differs from request: {cache_path}")
        budgets = tuple(int(value) for value in cache["budgets"].tolist())
        fits: dict[int, T4PrefixFit] = {}
        for budget in budgets:
            prefix = f"m{budget}_"
            fits[budget] = T4PrefixFit(
                budget=budget,
                t4=cache[prefix + "t4"].astype(np.float32, copy=False),
                reliability=cache[prefix + "reliability"].astype(np.float32, copy=False),
                fit_defined=bool(cache[prefix + "fit_defined"].item()),
                design_rank=int(cache[prefix + "design_rank"].item()),
                design_condition=float(cache[prefix + "design_condition"].item()),
                direction_counts=cache[prefix + "direction_counts"].astype(np.int64, copy=False),
                direction_balance=float(cache[prefix + "direction_balance"].item()),
                zero_spike_unit_count=int(cache[prefix + "zero_spike_unit_count"].item()),
                zero_modulation_unit_count=int(cache[prefix + "zero_modulation_unit_count"].item()),
            )
        return CrossBudgetFeatureBundle(
            source_path=str(metadata["source_path"]),
            signal_view=str(metadata["signal_view"]),
            cache_key=str(metadata["cache_key"]),
            budgets=budgets,
            fits=fits,
            cache_payload=payload,
            trial_ordinals=cache["trial_ordinals"].astype(np.int64, copy=False),
            trial_start_times=cache["trial_start_times"].astype(np.float64, copy=False),
            trial_stop_times=cache["trial_stop_times"].astype(np.float64, copy=False),
            trial_target_dirs_rad=cache["trial_target_dirs_rad"].astype(np.float64, copy=False),
            trial_direction_indices=cache["trial_direction_indices"].astype(np.int64, copy=False),
        )


def load_or_compute_cross_budget_features(
    nwb_path: Path,
    *,
    cache_dir: Path,
    budgets: Sequence[int] = DEFAULT_T4_BUDGETS,
    bin_size_ms: int = 20,
    window_size: int = 50,
    trial_result_filter: str = "R",
    signal_view: str = "sua",
) -> tuple[CrossBudgetFeatureBundle, Path, bool]:
    """Load/cache a bundle atomically.  Returns ``(bundle, path, cache_hit)``."""
    checked = validate_budgets(budgets)
    expected_payload = cross_budget_cache_payload(
        nwb_path,
        budgets=checked,
        bin_size_ms=bin_size_ms,
        window_size=window_size,
        trial_result_filter=trial_result_filter,
        signal_view=signal_view,
    )
    path = cross_budget_cache_path(
        cache_dir,
        nwb_path,
        budgets=checked,
        bin_size_ms=bin_size_ms,
        window_size=window_size,
        trial_result_filter=trial_result_filter,
        signal_view=signal_view,
    )
    with _exclusive_cache_lock(path):
        if path.is_file():
            return _load_bundle(path, expected_payload=expected_payload), path, True
        bundle = compute_cross_budget_features_uncached(
            nwb_path,
            budgets=checked,
            bin_size_ms=bin_size_ms,
            window_size=window_size,
            trial_result_filter=trial_result_filter,
            signal_view=signal_view,
        )
        _write_npz_atomically(path, **_bundle_arrays(bundle))
        return bundle, path, False
