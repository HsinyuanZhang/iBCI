#!/usr/bin/env python3
"""Read-only Stage-0 audit for calibration-only SUA auxiliary mechanisms.

The command defaults to the occupied sub-C/CO/27-6-6 split.  It opens neural,
waveform, and trial contents only for train + validation sessions.  Test files
are consulted at most by the existing discovery helper for their unit-table row
counts, never loaded into this audit; their names are emitted as an isolation
receipt but they are not cache inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
from pynwb import NWBHDF5IO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mc_maze.multisession_datamodule import (  # noqa: E402
    _cache_key, _source_fingerprint, _write_npz_atomically, chronological_session_split,
    discover_nwb_files, electrode_ids_from_units, list_datamodule_rewarded_trials,
    session_name_from_path,
)
from mc_maze.sua_auxiliary_stage0 import (  # noqa: E402
    STAGE0_FEATURE_VERSION, calibration_quality_features, design_rank_and_condition,
)
from mc_maze.unit_side_features import (  # noqa: E402
    CANONICAL_DIRECTIONS_RAD, WAVEFORM_SAMPLES, _fit_cosine_tuning,
    _nearest_canonical_direction_index, _pool_trial_rate_matrix, _read_unit_waveform_block,
    _scalar_features_from_template, _unit_spike_bounds,
)

# Stage-0's waveform columns are read-only negative diagnostics, not a primary
# representation.  Cap their per-unit I/O deterministically so a high-rate
# early training session cannot turn a scope audit into a data-scale screen.
MAX_DIAGNOSTIC_WAVEFORMS_PER_UNIT = 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_cache_path(cache_dir: Path, source: Path, *, pool_size: int) -> Path:
    payload = {
        "stage0_feature_version": STAGE0_FEATURE_VERSION,
        "kind": "sua_auxiliary_stage0_session",
        "pool_size": pool_size,
        "bin_size_ms": 20,
        "window_size": 50,
        "trial_result_filter": "R",
        "signal_view": "sua",
        "source": _source_fingerprint(source),
    }
    return cache_dir / "sua_auxiliary_stage0_v1" / "sessions" / (
        f"{session_name_from_path(source)}_{_cache_key(payload)[:20]}.npz"
    )


def _waveform_quality(nwb_path: Path, pool_end_time: float, n_units: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute F1-compatible amplitudes plus calibration-template variation/stability."""
    p2p = np.zeros(n_units, dtype=np.float32)
    noise = np.zeros(n_units, dtype=np.float32)
    snr = np.zeros(n_units, dtype=np.float32)
    residual_cv = np.zeros(n_units, dtype=np.float32)
    template_drift = np.zeros(n_units, dtype=np.float32)
    exposure = np.zeros(n_units, dtype=np.float32)
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        units = nwb.units.to_dataframe()
    with h5py.File(nwb_path, "r") as handle:
        waveforms = handle["units/waveforms"]
        waveforms_index = handle["units/waveforms_index"]
        index_index = np.asarray(handle["units/waveforms_index_index"])
        for unit in range(n_units):
            spike_times = np.asarray(units.iloc[unit]["spike_times"], dtype=np.float64)
            count = int(np.searchsorted(spike_times, pool_end_time, side="right"))
            exposure[unit] = count
            if count == 0:
                continue
            start, _ = _unit_spike_bounds(index_index, unit)
            diagnostic_count = min(count, MAX_DIAGNOSTIC_WAVEFORMS_PER_UNIT)
            stacked = _read_unit_waveform_block(waveforms, waveforms_index, start, diagnostic_count)
            template = stacked.mean(axis=0)
            residuals = stacked - template
            values = _scalar_features_from_template(template, residuals.reshape(-1))
            p2p[unit], noise[unit], snr[unit] = values["p2p"], values["noise_std"], values["snr"]
            residual_cv[unit] = float(residuals.std() / max(values["p2p"], 1e-6))
            if diagnostic_count >= 2:
                midpoint = diagnostic_count // 2
                first, second = stacked[:midpoint].mean(axis=0), stacked[midpoint:].mean(axis=0)
                template_drift[unit] = float(np.sqrt(np.mean((first - second) ** 2)) / max(values["p2p"], 1e-6))
    return p2p, noise, snr, residual_cv, template_drift, exposure


def _t4_and_residual(rates: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """T4 plus each unit's rate-normalised residual using only the same pool."""
    t4 = np.zeros((rates.shape[0], 4), dtype=np.float32)
    residual = np.zeros(rates.shape[0], dtype=np.float32)
    present = sorted({int(x) for x in directions if x >= 0})
    if len(present) < 3:
        return t4, residual
    theta = np.asarray([CANONICAL_DIRECTIONS_RAD[index] for index in present])
    for unit, unit_rates in enumerate(rates):
        means = np.asarray([unit_rates[directions == index].mean() for index in present])
        a, c, m, b = _fit_cosine_tuning(theta, means)
        prediction = b + a * np.cos(np.asarray([CANONICAL_DIRECTIONS_RAD[max(i, 0)] for i in directions])) + c * np.sin(np.asarray([CANONICAL_DIRECTIONS_RAD[max(i, 0)] for i in directions]))
        valid = directions >= 0
        denom = float(np.var(unit_rates[valid])) if np.any(valid) else 0.0
        t4[unit] = (a, c, m, b)
        residual[unit] = float(np.mean((unit_rates[valid] - prediction[valid]) ** 2) / max(denom, 1e-6)) if np.any(valid) else 0.0
    return t4, residual


def _within_group_dispersion(values: np.ndarray, memberships: np.ndarray) -> tuple[float, float]:
    """Return unit-weighted multi-unit dispersion and fraction of rows supporting it."""
    weighted_sum = 0.0
    weighted_count = 0
    for electrode in np.unique(memberships):
        rows = values[memberships == electrode]
        if rows.shape[0] <= 1:
            continue
        weighted_sum += float(np.sum((rows - rows.mean(axis=0, keepdims=True)) ** 2))
        weighted_count += int(rows.size)
    return (weighted_sum / weighted_count if weighted_count else 0.0,
            weighted_count / values.size if values.size else 0.0)


def _session_payload(nwb_path: Path, pool_size: int) -> dict[str, np.ndarray]:
    rewarded = list_datamodule_rewarded_trials(nwb_path, bin_size_ms=20, window_size=50, trial_result_filter="R")
    if len(rewarded) < pool_size:
        raise ValueError(f"{nwb_path.name}: requires {pool_size} rewarded trials, found {len(rewarded)}")
    trials = rewarded[:pool_size]
    directions = np.asarray([
        _nearest_canonical_direction_index(trial["target_dir"]) if trial.get("target_dir") is not None else -1
        for trial in trials
    ], dtype=np.int64)
    rates, n_units = _pool_trial_rate_matrix(nwb_path, trials)
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        electrode_ids = electrode_ids_from_units(nwb.units.to_dataframe())
    pool_end = float(trials[-1]["stop_time"])
    p2p, noise, snr, waveform_cv, waveform_drift, exposure = _waveform_quality(nwb_path, pool_end, n_units)
    t4, t4_residual = _t4_and_residual(rates, directions)
    rank, condition = design_rank_and_condition(directions, CANONICAL_DIRECTIONS_RAD)
    quality = calibration_quality_features(
        p2p=p2p, noise_std=noise, snr=snr, waveform_residual_cv=waveform_cv,
        waveform_template_drift=waveform_drift, spike_exposure=exposure,
        t4_relative_residual=t4_residual, design_condition=condition, rank_valid=(rank == 3),
    )
    return {
        "electrode_ids": electrode_ids.astype(np.int64), "rates": rates.astype(np.float32),
        "directions": directions, "t4": t4, "quality": quality,
        "p2p": p2p, "noise_std": noise, "snr": snr, "waveform_residual_cv": waveform_cv,
        "waveform_template_drift": waveform_drift, "spike_exposure": exposure,
        "t4_relative_residual": t4_residual,
        "design_rank": np.asarray(rank), "design_condition": np.asarray(condition),
    }


def _load_or_build(cache_dir: Path, nwb_path: Path, pool_size: int) -> dict[str, np.ndarray]:
    path = _feature_cache_path(cache_dir, nwb_path, pool_size=pool_size)
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    payload = _session_payload(nwb_path, pool_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_npz_atomically(path, **payload)
    return payload


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=ROOT / "data/dandi_000688/sub-C")
    parser.add_argument("--cache_dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--output", type=Path, default=ROOT / "results/sua_auxiliary_stage0/audit.json")
    parser.add_argument("--pool_size", type=int, default=50)
    args = parser.parse_args()
    files = discover_nwb_files(args.data_dir, "CO", max_units_exclusive=100)
    train_files, val_files, test_files = chronological_session_split(files, (27, 6, 6), max_units_exclusive=100)
    # Deliberately build/cache only allowed source files.  Test names are receipt-only.
    allowed = [("train", path) for path in train_files] + [("val", path) for path in val_files]
    sessions = {session_name_from_path(path): (split, _load_or_build(args.cache_dir, path, args.pool_size)) for split, path in allowed}
    train_quality = np.concatenate([payload["quality"] for split, payload in sessions.values() if split == "train"])
    train_t4 = np.concatenate([payload["t4"] for split, payload in sessions.values() if split == "train"])
    quality_mean, quality_std = train_quality.mean(0), train_quality.std(0).clip(1e-6)
    t4_mean, t4_std = train_t4.mean(0), train_t4.std(0).clip(1e-6)
    summary: dict[str, dict] = {}
    for name, (split, payload) in sessions.items():
        ids, rates, t4, quality = payload["electrode_ids"], payload["rates"], payload["t4"], payload["quality"]
        _, group_sizes = np.unique(ids, return_counts=True)
        electrode_values, group_sizes = np.unique(ids, return_counts=True)
        multi = group_sizes[group_sizes > 1]
        multi_mask = np.isin(ids, electrode_values[group_sizes > 1])
        # Avoid an absolute-ID feature: ids enter only equality groups and coverage counts.
        activity_by_unit = np.log1p(rates)
        metrics = {
            "split": split, "units": int(ids.size), "electrodes": int(group_sizes.size),
            "multi_unit_electrodes": int(multi.size), "multi_unit_electrode_fraction": float(multi.size / group_sizes.size),
            "multi_unit_unit_fraction": float(np.sum(group_sizes[group_sizes > 1]) / ids.size),
            "max_group_size": int(group_sizes.max()), "mean_units_per_electrode": float(ids.size / group_sizes.size),
            "rank": int(payload["design_rank"]), "condition": float(payload["design_condition"]),
            "t4_dispersion": _within_group_dispersion((t4 - t4_mean) / t4_std, ids)[0],
            # SNR/template stability remain audit-only negatives: this row is
            # reported but is deliberately excluded from eligibility and model context.
            "read_only_waveform_diagnostic_dispersion": _within_group_dispersion((quality - quality_mean) / quality_std, ids)[0],
            "relative_amplitude_dispersion": _within_group_dispersion(np.log1p(payload["p2p"])[:, None], ids)[0],
            "activity_dispersion": _within_group_dispersion(activity_by_unit, ids)[0],
            "t4_residual_multi_mean": float(payload["t4_relative_residual"][multi_mask].mean()) if np.any(multi_mask) else 0.0,
            "t4_residual_singleton_mean": float(payload["t4_relative_residual"][~multi_mask].mean()) if np.any(~multi_mask) else 0.0,
            "read_only_waveform_diagnostic_feature_names": list(__import__("mc_maze.sua_auxiliary_stage0", fromlist=["READONLY_DIAGNOSTIC_FEATURE_NAMES"]).READONLY_DIAGNOSTIC_FEATURE_NAMES),
        }
        summary[name] = metrics
    bridge = json.loads((ROOT / "results/pseudomua_t4_bridge_v1/summary.json").read_text())
    val_names = [session_name_from_path(path) for path in val_files]
    session_gaps = {
        arm: {
            name: bridge["views"]["sua"]["arm_scores"][arm]["per_session_seed_mean"][name]
            - bridge["views"]["pseudo_mua"]["arm_scores"][arm]["per_session_seed_mean"][name]
            for name in val_names
        } for arm in ("F0", "T4")
    }
    correlations = {
        arm: {metric: _pearson([summary[name][metric] for name in val_names], [session_gaps[arm][name] for name in val_names])
              for metric in ("multi_unit_unit_fraction", "t4_dispersion", "relative_amplitude_dispersion", "activity_dispersion")}
        for arm in ("F0", "T4")
    }
    eligibility = {
        "predeclared_rule": "relation route is eligible only when >=5/6 validation sessions have multi-unit unit share >=0.50 and multi-unit T4/activity dispersion is non-zero; relative amplitude is secondary and requires membership shuffle. SNR/template stability are read-only negatives. R2-gap correlations are descriptive only (n=6).",
        "coverage_sessions_ge_50pct": int(sum(summary[name]["multi_unit_unit_fraction"] >= 0.50 for name in val_names)),
        "all_val_rank3": bool(all(summary[name]["rank"] == 3 for name in val_names)),
        "eligible": bool(sum(summary[name]["multi_unit_unit_fraction"] >= 0.50 for name in val_names) >= 5 and all(summary[name]["t4_dispersion"] > 0 and summary[name]["activity_dispersion"] > 0 for name in val_names)),
    }
    output = {
        "schema_version": 1, "stage0_feature_version": STAGE0_FEATURE_VERSION,
        "purpose": "validation-development, calibration-only SUA quality/electrode Stage-0 audit",
        "source_scope": {"task": "CO", "max_units_exclusive": 100, "split_counts": [27, 6, 6], "pool_size": args.pool_size, "opened_neural_trial_sessions": [name for name, _ in sessions.items()], "test_sessions_not_opened": [session_name_from_path(path) for path in test_files]},
        "train_only_normalizers": {"quality_mean": quality_mean.tolist(), "quality_std": quality_std.tolist(), "t4_mean": t4_mean.tolist(), "t4_std": t4_std.tolist()},
        "sessions": summary, "existing_view_gap": session_gaps, "descriptive_gap_correlations": correlations,
        "relation_eligibility": eligibility,
        "provenance": {"bridge_summary": str((ROOT / "results/pseudomua_t4_bridge_v1/summary.json").resolve()), "bridge_summary_sha256": _sha256(ROOT / "results/pseudomua_t4_bridge_v1/summary.json"), "cache_namespace": str((args.cache_dir / "sua_auxiliary_stage0_v1").resolve())},
        "formal_test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "eligibility": eligibility, "val_sessions": val_names}, indent=2))


if __name__ == "__main__":
    main()
