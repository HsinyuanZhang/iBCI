#!/usr/bin/env python
"""CPU-only diagnostic: quantify cross-session drift in unit_side_features F2 scalars.

Context: side_feature_ablation_v2 (sua_exploration/results/side_feature_ablation_v2/)
came back "indeterminate" for both F1 and F2, but every point estimate was negative and
the shuffled controls beat the real features. Working hypothesis: the six waveform/
amplitude scalars (p2p, noise_std, snr, pt_width, pt_ratio, repol_slope) drift
systematically across recording days (electrode impedance change, different
spike-sorting thresholds, ...), and because they are z-scored with train-session-only
statistics (UNIT_SIDE_FEATURE_ABLATION.md section 6.1), any train-to-validation
distribution shift injects a session-dependent bias into the identity vector.

This script ONLY measures that drift. It does not train anything, does not touch the
GPU, and does not modify unit_side_features.py, the ablation results, or any threshold
in the governing docs.

Reuse discipline: all feature computation is delegated to
sua_exploration/mc_maze/unit_side_features.py (compute_unit_side_features_uncached /
load_unit_side_features / fit_side_feature_stats / _fit_robust_stats /
_in_pool_spike_prefix) and sua_exploration/mc_maze/multisession_datamodule.py
(calibration_pool_end_time / nwb_unit_count / _source_fingerprint). Nothing in the
feature-scalar math (p2p / noise_std / snr / pt_width / pt_ratio / repol_slope, or the
train-only z-scoring contract) is reimplemented here.

Hard constraint: this script NEVER opens an NWB file for one of the 6 held-out test
sessions of the sub-C/CO/27-6-6 (N<100) split. The session list is built exclusively
from session_splits["train"] + session_splits["val"] read out of
sua_exploration/results/side_feature_ablation_v2/aggregate.json; an explicit assertion
below checks the constructed file list is disjoint from the test-session paths before
any file is touched. This diagnostic does not need test-session unit counts either, so
it does not open test NWBs at all (not even for the permitted row-count read).

Usage:
    /home/xinyuan/miniconda3/envs/spint/bin/python \
        sua_exploration/scripts/side_feature_drift_diagnostic.py
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = ROOT / "sua_exploration"
sys.path.insert(0, str(SUA_ROOT))

from mc_maze.unit_side_features import (  # noqa: E402
    FEATURE_GROUPS,
    FEATURE_VERSION,
    _fit_robust_stats,
    _in_pool_spike_prefix,
    _side_feature_cache_path,
    _side_stats_cache_path,
    compute_unit_side_features_uncached,  # noqa: F401  (documented as reused; wrapped by load_unit_side_features)
    fit_side_feature_stats,
    load_unit_side_features,
    side_feature_stats_sha256,
)
from mc_maze.multisession_datamodule import (  # noqa: E402
    _source_fingerprint,
    calibration_pool_end_time,
    nwb_unit_count,
)
from pynwb import NWBHDF5IO  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("side_feature_drift_diagnostic")

DATA_DIR = SUA_ROOT / "data" / "dandi_000688" / "sub-C"
CACHE_DIR = SUA_ROOT / "cache" / "dandi688_subc_co_v1"
AGGREGATE_PATH = SUA_ROOT / "results" / "side_feature_ablation_v2" / "aggregate.json"
OUT_JSON = SUA_ROOT / "results" / "side_feature_drift_diagnostic.json"
OUT_MD = SUA_ROOT / "docs" / "SIDE_FEATURE_DRIFT_DIAGNOSTIC.md"

FEATURE_GROUP = "f2"
FEATURE_NAMES = FEATURE_GROUPS[FEATURE_GROUP]  # (p2p, noise_std, snr, pt_width, pt_ratio, repol_slope)
POOL_SIZE = 50
BIN_SIZE_MS = 20
WINDOW_SIZE = 50
TRIAL_RESULT_FILTER = "R"
DRIFT_FLAG_ABS_Z = 0.5


def nwb_path_for(session_name: str) -> Path:
    return DATA_DIR / f"{session_name}_behavior+ecephys.nwb"


def session_date(session_name: str) -> datetime:
    digits = session_name.split("-")[-1]
    if len(digits) != 8 or not digits.isdigit():
        raise ValueError(f"Cannot parse date from session name {session_name!r}")
    return datetime.strptime(digits, "%Y%m%d")


def null_spike_counts_per_unit(nwb_path: Path, pool_end_time: float) -> np.ndarray:
    """Per-unit in-pool spike count: a feature that should NOT drift with electrode
    impedance / spike-sorting-threshold changes the way waveform shape does (it is
    driven mostly by firing rate and pool duration, both already accounted for by the
    fixed pool_size=50-trial definition). Reuses unit_side_features._in_pool_spike_prefix
    (the identical in-pool definition used for the real features) but never touches the
    waveforms dataset, only units/spike_times."""
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        units_df = nwb.units.to_dataframe()
        counts = np.zeros(len(units_df), dtype=np.float64)
        for unit_idx in range(len(units_df)):
            spike_times = np.asarray(units_df.iloc[unit_idx]["spike_times"], dtype=np.float64)
            num_in_pool, _ = _in_pool_spike_prefix(spike_times, pool_end_time)
            counts[unit_idx] = num_in_pool
    return counts


def linreg(x_days: np.ndarray, y: np.ndarray) -> dict:
    x_days = np.asarray(x_days, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = int(x_days.size)
    if n < 2:
        return {
            "n_sessions": n, "slope_per_day": None, "slope_per_year": None,
            "intercept": None, "r_value": None, "r_squared": None, "p_value": None,
            "stderr": None, "date_range_days": None, "total_drift_z": None,
        }
    result = sstats.linregress(x_days, y)
    date_range = float(x_days.max() - x_days.min())
    slope = float(result.slope)
    return {
        "n_sessions": n,
        "slope_per_day": slope,
        "slope_per_year": slope * 365.25,
        "intercept": float(result.intercept),
        "r_value": float(result.rvalue),
        "r_squared": float(result.rvalue ** 2),
        "p_value": float(result.pvalue),
        "stderr": float(result.stderr),
        "date_range_days": date_range,
        "total_drift_z": slope * date_range,
    }


def between_within_variance(values_by_session: dict) -> dict:
    """One-way decomposition of unit-level values with session as the grouping factor.
    Reported both as raw sum-of-squares fractions and as an (unbalanced) ICC. This
    fraction is invariant to any single global affine rescaling (e.g. the train-only
    z-scoring used elsewhere in this script): if z = (x - m)/s with m, s constant across
    all sessions, both SS_between and SS_within scale by 1/s**2, so their ratio to
    SS_total is unchanged. Values are therefore passed in RAW (unnormalized)."""
    sessions = list(values_by_session.keys())
    arrays = [np.asarray(values_by_session[s], dtype=np.float64) for s in sessions]
    all_values = np.concatenate(arrays)
    grand_mean = float(all_values.mean())
    total_ss = float(np.sum((all_values - grand_mean) ** 2))
    between_ss = 0.0
    within_ss = 0.0
    for arr in arrays:
        session_mean = float(arr.mean())
        between_ss += arr.size * (session_mean - grand_mean) ** 2
        within_ss += float(np.sum((arr - session_mean) ** 2))
    n_total = int(all_values.size)
    n_sessions = len(sessions)
    df_between = n_sessions - 1
    df_within = n_total - n_sessions
    ms_between = between_ss / df_between if df_between > 0 else float("nan")
    ms_within = within_ss / df_within if df_within > 0 else float("nan")
    icc = None
    if df_between > 0 and df_within > 0 and n_total > n_sessions and ms_within + ms_between > 0:
        sum_sq_sizes = sum(arr.size ** 2 for arr in arrays)
        n_bar = (n_total - (sum_sq_sizes / n_total)) / df_between
        if n_bar > 0:
            var_between = max(0.0, (ms_between - ms_within) / n_bar)
            denom = var_between + ms_within
            icc = float(var_between / denom) if denom > 0 else 0.0
    return {
        "n_sessions": n_sessions,
        "n_total_units": n_total,
        "total_ss": total_ss,
        "between_session_ss": between_ss,
        "within_session_ss": within_ss,
        "between_fraction_of_ss": (between_ss / total_ss) if total_ss > 0 else 0.0,
        "within_fraction_of_ss": (within_ss / total_ss) if total_ss > 0 else 0.0,
        "df_between": df_between,
        "df_within": df_within,
        "ms_between": None if np.isnan(ms_between) else ms_between,
        "ms_within": None if np.isnan(ms_within) else ms_within,
        "f_statistic": (ms_between / ms_within) if (ms_within and ms_within > 0 and not np.isnan(ms_between)) else None,
        "icc_session_level_unbalanced": icc,
    }


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def main() -> None:
    started_at = datetime.now().astimezone().isoformat()
    aggregate = json.loads(AGGREGATE_PATH.read_text())
    splits = aggregate["session_splits"]
    train_sessions = list(splits["train"])
    val_sessions = list(splits["val"])
    test_sessions = list(splits["test"])
    assert len(train_sessions) == 27, f"expected 27 train sessions, got {len(train_sessions)}"
    assert len(val_sessions) == 6, f"expected 6 val sessions, got {len(val_sessions)}"
    assert len(test_sessions) == 6, f"expected 6 test sessions, got {len(test_sessions)}"

    used_sessions = train_sessions + val_sessions
    used_paths = {nwb_path_for(s) for s in used_sessions}
    test_paths = {nwb_path_for(s) for s in test_sessions}
    assert used_paths.isdisjoint(test_paths), "REFUSING TO RUN: constructed file list touches a test-session NWB"
    for s in used_sessions:
        p = nwb_path_for(s)
        assert p.is_file(), f"Missing NWB file for session {s}: {p}"
    logger.info(
        "Session list OK: %d train + %d val = %d sessions used; %d test sessions excluded and never opened",
        len(train_sessions), len(val_sessions), len(used_sessions), len(test_sessions),
    )

    train_files = [nwb_path_for(s) for s in train_sessions]

    print(f"[1/6] Fitting train-only {FEATURE_GROUP} stats over {len(train_files)} train sessions "
          f"(fit_side_feature_stats, cache_dir={CACHE_DIR})...")
    mean_f2, std_f2 = fit_side_feature_stats(
        train_files, feature_group=FEATURE_GROUP, pool_size=POOL_SIZE, cache_dir=CACHE_DIR,
        bin_size_ms=BIN_SIZE_MS, window_size=WINDOW_SIZE, trial_result_filter=TRIAL_RESULT_FILTER,
    )
    stats_sha = side_feature_stats_sha256(mean_f2, std_f2)
    stats_cache_path = _side_stats_cache_path(
        CACHE_DIR, train_files, feature_group=FEATURE_GROUP, pool_size=POOL_SIZE,
        bin_size_ms=BIN_SIZE_MS, window_size=WINDOW_SIZE, trial_result_filter=TRIAL_RESULT_FILTER,
    )
    print(f"  mean = {mean_f2.tolist()}")
    print(f"  std  = {std_f2.tolist()}")
    print(f"  stats sha256 = {stats_sha}")
    print(f"  stats cache path = {stats_cache_path} (exists={stats_cache_path.is_file()})")

    raw_by_session: dict[str, np.ndarray] = {}
    meta_by_session: dict[str, dict] = {}
    null_by_session: dict[str, np.ndarray] = {}
    source_fp_by_session: dict[str, dict] = {}
    feature_cache_path_by_session: dict[str, str] = {}
    unit_counts: dict[str, int] = {}
    dates: dict[str, datetime] = {}

    print(f"\n[2/6] Loading raw {FEATURE_GROUP} features (cache-reused, no reimplementation) "
          f"+ null in-pool-spike-count feature for {len(used_sessions)} sessions...")
    zero_vec = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    one_vec = np.ones(len(FEATURE_NAMES), dtype=np.float32)
    for s in used_sessions:
        p = nwb_path_for(s)
        raw, meta = load_unit_side_features(
            p, feature_group=FEATURE_GROUP, pool_size=POOL_SIZE,
            mean=zero_vec, std=one_vec,  # identity normalization -> recovers literal raw cached values
            cache_dir=CACHE_DIR, permutation_seed=None,
            bin_size_ms=BIN_SIZE_MS, window_size=WINDOW_SIZE, trial_result_filter=TRIAL_RESULT_FILTER,
        )
        raw_by_session[s] = raw.astype(np.float64)
        meta_by_session[s] = {
            "feature_group": meta.feature_group,
            "feature_version": meta.feature_version,
            "pool_size": meta.pool_size,
            "cache_key": meta.cache_key,
            "degenerate_unit_count": meta.degenerate_unit_count,
            "zero_spike_unit_count": meta.zero_spike_unit_count,
            "single_spike_unit_count": meta.single_spike_unit_count,
            "zero_noise_std_unit_count": meta.zero_noise_std_unit_count,
            "zero_template_max_unit_count": meta.zero_template_max_unit_count,
        }
        source_fp_by_session[s] = _source_fingerprint(p)
        feature_cache_path_by_session[s] = str(_side_feature_cache_path(
            CACHE_DIR, p, feature_group=FEATURE_GROUP, pool_size=POOL_SIZE,
            bin_size_ms=BIN_SIZE_MS, window_size=WINDOW_SIZE, trial_result_filter=TRIAL_RESULT_FILTER,
        ))

        pool_end_time = calibration_pool_end_time(
            p, pool_size=POOL_SIZE, bin_size_ms=BIN_SIZE_MS, window_size=WINDOW_SIZE,
            trial_result_filter=TRIAL_RESULT_FILTER,
        )
        null_by_session[s] = null_spike_counts_per_unit(p, pool_end_time)

        actual_unit_count = nwb_unit_count(p)
        assert actual_unit_count == raw.shape[0], (
            f"{s}: unit count mismatch {actual_unit_count} vs {raw.shape[0]}"
        )
        unit_counts[s] = int(raw.shape[0])
        dates[s] = session_date(s)
        print(f"  {s} ({'train' if s in train_sessions else 'val '}, {dates[s].date()}): "
              f"units={raw.shape[0]:3d} degenerate={meta.degenerate_unit_count:2d} "
              f"mean_pool_spikes/unit={null_by_session[s].mean():.1f}")

    degenerate_mask_by_session = {s: (null_by_session[s] <= 1) for s in used_sessions}

    # ---------------------------------------------------------------- Task 1
    print("\n[3/6] Task 1: per-session raw feature distributions...")
    task1 = {}
    for s in used_sessions:
        arr = raw_by_session[s]
        task1[s] = {
            "split": "train" if s in train_sessions else "val",
            "date": dates[s].strftime("%Y-%m-%d"),
            "n_units": int(arr.shape[0]),
            "n_degenerate_units": int(degenerate_mask_by_session[s].sum()),
            "features": {
                name: {
                    "mean": float(arr[:, i].mean()),
                    "std": float(arr[:, i].std()),
                    "median": float(np.median(arr[:, i])),
                    "min": float(arr[:, i].min()),
                    "max": float(arr[:, i].max()),
                }
                for i, name in enumerate(FEATURE_NAMES)
            },
        }

    # ---------------------------------------------------------------- Task 2
    print("[4/6] Task 2: train-only z-scoring, per-session mean z, train->val shift...")
    z_by_session = {s: (raw_by_session[s] - mean_f2) / std_f2 for s in used_sessions}
    task2 = {
        "train_only_stats": {"mean": mean_f2.tolist(), "std": std_f2.tolist(), "sha256": stats_sha},
        "drift_flag_abs_meanz_threshold": DRIFT_FLAG_ABS_Z,
        "per_session_mean_z": {},
        "per_feature_val_summary": {},
    }
    for s in used_sessions:
        arr = z_by_session[s]
        arr_clean = arr[~degenerate_mask_by_session[s]]
        task2["per_session_mean_z"][s] = {
            "split": "train" if s in train_sessions else "val",
            "date": dates[s].strftime("%Y-%m-%d"),
            "mean_z": {name: float(arr[:, i].mean()) for i, name in enumerate(FEATURE_NAMES)},
            "mean_z_excl_degenerate_units": (
                {name: float(arr_clean[:, i].mean()) for i, name in enumerate(FEATURE_NAMES)}
                if arr_clean.shape[0] > 0 else None
            ),
        }
    for i, name in enumerate(FEATURE_NAMES):
        val_means = np.array([z_by_session[s][:, i].mean() for s in val_sessions])
        train_means = np.array([z_by_session[s][:, i].mean() for s in train_sessions])
        task2["per_feature_val_summary"][name] = {
            "per_val_session_mean_z": {s: float(z_by_session[s][:, i].mean()) for s in val_sessions},
            "mean_over_val_sessions": float(val_means.mean()),
            "std_over_val_sessions": float(val_means.std()),
            "n_val_sessions_abs_meanz_gt_threshold": int(np.sum(np.abs(val_means) > DRIFT_FLAG_ABS_Z)),
            "train_session_mean_z_spread_mean": float(train_means.mean()),
            "train_session_mean_z_spread_std": float(train_means.std()),
            "n_train_sessions_abs_meanz_gt_threshold": int(np.sum(np.abs(train_means) > DRIFT_FLAG_ABS_Z)),
        }
        print(f"  {name:12s} val mean_z={val_means.mean():+.3f} (session std={val_means.std():.3f}, "
              f"train-session spread std={train_means.std():.3f}), "
              f"{int(np.sum(np.abs(val_means) > DRIFT_FLAG_ABS_Z))}/6 val sessions |mean z|>{DRIFT_FLAG_ABS_Z}")

    # ---------------------------------------------------------------- Task 3
    print("\n[5/6] Task 3: chronological regression of per-session mean z on session date...")
    t0 = min(dates.values())
    days = {s: float((dates[s] - t0).days) for s in used_sessions}
    task3 = {"reference_date": t0.strftime("%Y-%m-%d"), "features": {}}
    x_all = np.array([days[s] for s in used_sessions])
    x_val = np.array([days[s] for s in val_sessions])
    for i, name in enumerate(FEATURE_NAMES):
        y_all = np.array([z_by_session[s][:, i].mean() for s in used_sessions])
        y_val = np.array([z_by_session[s][:, i].mean() for s in val_sessions])
        reg_all = linreg(x_all, y_all)
        reg_val = linreg(x_val, y_val)
        task3["features"][name] = {"all_sessions_n33": reg_all, "val_only_n6": reg_val}
        print(f"  {name:12s} all-33 slope/yr={reg_all['slope_per_year']:+.3f} R2={reg_all['r_squared']:.3f} "
              f"total_drift_z={reg_all['total_drift_z']:+.3f} | val-only(n=6) slope/yr={reg_val['slope_per_year']}")

    # ---------------------------------------------------------------- Task 4 (null)
    print("\n[6/6] Task 4: null-feature comparison (in-pool spike count per unit)...")
    train_null_stacked = np.concatenate([null_by_session[s] for s in train_sessions]).reshape(-1, 1)
    null_mean_arr, null_std_arr, null_clipped_count = _fit_robust_stats(train_null_stacked)
    null_mean, null_std = float(null_mean_arr[0]), float(null_std_arr[0])
    null_z_by_session = {s: (null_by_session[s] - null_mean) / null_std for s in used_sessions}

    y_all_null = np.array([null_z_by_session[s].mean() for s in used_sessions])
    y_val_null = np.array([null_z_by_session[s].mean() for s in val_sessions])
    reg_all_null = linreg(x_all, y_all_null)
    reg_val_null = linreg(x_val, y_val_null)
    val_means_null = np.array([null_z_by_session[s].mean() for s in val_sessions])
    train_means_null = np.array([null_z_by_session[s].mean() for s in train_sessions])

    unit_count_y = np.array([unit_counts[s] for s in used_sessions], dtype=np.float64)
    reg_unit_count = linreg(x_all, unit_count_y)

    task4 = {
        "null_feature": "in_pool_spike_count_per_unit",
        "null_feature_definition": (
            "Per-unit spike count within the same pool_size=50-rewarded-trial calibration "
            "pool used for the real features (reuses unit_side_features._in_pool_spike_prefix "
            "and multisession_datamodule.calibration_pool_end_time); never reads waveforms."
        ),
        "train_only_stats": {"mean": null_mean, "std": null_std, "clipped_scalar_count": int(null_clipped_count)},
        "per_session_raw": {
            s: {
                "mean": float(null_by_session[s].mean()),
                "std": float(null_by_session[s].std()),
                "median": float(np.median(null_by_session[s])),
            }
            for s in used_sessions
        },
        "per_session_mean_z": {s: float(null_z_by_session[s].mean()) for s in used_sessions},
        "val_summary": {
            "mean_over_val_sessions": float(val_means_null.mean()),
            "std_over_val_sessions": float(val_means_null.std()),
            "n_val_sessions_abs_meanz_gt_threshold": int(np.sum(np.abs(val_means_null) > DRIFT_FLAG_ABS_Z)),
            "train_session_mean_z_spread_std": float(train_means_null.std()),
        },
        "chronological": {"all_sessions_n33": reg_all_null, "val_only_n6": reg_val_null},
        "variance_decomposition": between_within_variance(null_by_session),
        "secondary_null_unit_count_per_session": {
            "note": (
                "Session-level scalar (one value per session, not per-unit), so it supports "
                "only the chronological-trend sub-task, not the within/between "
                "variance-decomposition sub-task. Not z-scored (no natural per-unit train-only "
                "normalization for a session-level count)."
            ),
            "per_session": unit_counts,
            "chronological_all_sessions_raw_units": reg_unit_count,
        },
    }
    print(f"  null val mean_z={val_means_null.mean():+.3f} (session std={val_means_null.std():.3f}), "
          f"chron slope/yr={reg_all_null['slope_per_year']:+.3f} R2={reg_all_null['r_squared']:.3f}")

    # ---------------------------------------------------------------- Task 5
    print("\nTask 5: within vs between-session variance decomposition (raw values; "
          "fraction is invariant to the global train-only z-scoring, see docstring)...")
    task5 = {"features": {}, "null_feature_in_pool_spike_count": between_within_variance(null_by_session)}
    for i, name in enumerate(FEATURE_NAMES):
        values_by_session = {s: raw_by_session[s][:, i] for s in used_sessions}
        decomp = between_within_variance(values_by_session)
        task5["features"][name] = decomp
        print(f"  {name:12s} between-session fraction of SS = {decomp['between_fraction_of_ss']:.3f}  "
              f"(ICC={decomp['icc_session_level_unbalanced']})")
    print(f"  {'[null]':12s} between-session fraction of SS = "
          f"{task5['null_feature_in_pool_spike_count']['between_fraction_of_ss']:.3f}  "
          f"(ICC={task5['null_feature_in_pool_spike_count']['icc_session_level_unbalanced']})")

    # ---------------------------------------------------------------- assemble + write
    result = {
        "purpose": "side_feature_drift_diagnostic",
        "status": "diagnostic_only_no_modeling",
        "hypothesis_under_test": (
            "Waveform/amplitude side-feature scalars (F2: p2p, noise_std, snr, pt_width, "
            "pt_ratio, repol_slope) drift systematically across recording sessions; because "
            "they are z-scored with train-session-only statistics, this injects a "
            "session-dependent bias into validation-session identity vectors."
        ),
        "provenance": {
            "generated_at": started_at,
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "python_executable": sys.executable,
            "git_commit": git_commit(),
            "source_split_artifact": str(AGGREGATE_PATH.relative_to(ROOT)),
            "split_name": "sub-C/CO/27-6-6 chronological, N<100 (old unit-count regime)",
            "session_splits": {
                "train": train_sessions,
                "val": val_sessions,
                "test_excluded_never_opened": test_sessions,
            },
            "feature_group": FEATURE_GROUP,
            "feature_names": list(FEATURE_NAMES),
            "feature_version": FEATURE_VERSION,
            "pool_size": POOL_SIZE,
            "bin_size_ms": BIN_SIZE_MS,
            "window_size": WINDOW_SIZE,
            "trial_result_filter": TRIAL_RESULT_FILTER,
            "cache_dir": str(CACHE_DIR.relative_to(ROOT)),
            "side_feature_stats_cache_path": str(stats_cache_path),
            "side_feature_stats_sha256": stats_sha,
            "per_session_feature_cache_path": feature_cache_path_by_session,
            "per_session_source_fingerprint": source_fp_by_session,
            "per_session_feature_metadata": meta_by_session,
            "hard_constraints": {
                "test_sessions_never_opened": True,
                "no_training": True,
                "no_gpu_used": True,
                "cpu_only": True,
                "unit_side_features_py_modified": False,
                "ablation_results_modified": False,
                "p3_formal_test_receipt_touched": False,
            },
        },
        "task1_per_session_raw_distribution": task1,
        "task2_train_to_val_zscore_shift": task2,
        "task3_chronological_trend": task3,
        "task4_null_comparison": task4,
        "task5_variance_decomposition": task5,
    }

    # ---------------------------------------------------------------- Supplementary checks
    print("\n[extra] Supplementary check A: F1's 3 scalars are a numerically identical subset "
          "of F2's 6 (verifies the F2-only analysis above already covers F1)...")
    mean_f1, std_f1 = fit_side_feature_stats(
        train_files, feature_group="f1", pool_size=POOL_SIZE, cache_dir=CACHE_DIR,
        bin_size_ms=BIN_SIZE_MS, window_size=WINDOW_SIZE, trial_result_filter=TRIAL_RESULT_FILTER,
    )
    f1_matches_f2_prefix = bool(np.allclose(mean_f1, mean_f2[:3]) and np.allclose(std_f1, std_f2[:3]))
    print(f"  fit_side_feature_stats('f1') mean/std == fit_side_feature_stats('f2')[:3] mean/std: "
          f"{f1_matches_f2_prefix}")

    print("\n[extra] Supplementary check B: exploratory correlation between per-val-session "
          "drift magnitude (L2 norm of the 6-dim mean-z vector) and the side_feature_ablation_v2 "
          "F2-F0 / F1-F0 R2 deltas actually observed for that session (NOT one of the 5 "
          "requested sub-tasks; n=6, clearly exploratory/underpowered; reads but does not "
          "modify sua_exploration/results/side_feature_ablation_v2/aggregate.json)...")
    drift_l2_by_val_session = {
        s: float(np.linalg.norm([z_by_session[s][:, i].mean() for i in range(len(FEATURE_NAMES))]))
        for s in val_sessions
    }
    r2_delta_f2_f0 = aggregate["paired_deltas"]["F2_minus_F0"]["per_session_seed_mean"]
    r2_delta_f1_f0 = aggregate["paired_deltas"]["F1_minus_F0"]["per_session_seed_mean"]
    x_drift = np.array([drift_l2_by_val_session[s] for s in val_sessions])
    y_f2f0 = np.array([r2_delta_f2_f0[s] for s in val_sessions])
    y_f1f0 = np.array([r2_delta_f1_f0[s] for s in val_sessions])
    pearson_f2f0 = sstats.pearsonr(x_drift, y_f2f0)
    pearson_f1f0 = sstats.pearsonr(x_drift, y_f1f0)
    spearman_f2f0 = sstats.spearmanr(x_drift, y_f2f0)
    for s in val_sessions:
        print(f"  {s}: drift_L2={drift_l2_by_val_session[s]:.3f}  "
              f"F2-F0_R2delta={r2_delta_f2_f0[s]:+.3f}  F1-F0_R2delta={r2_delta_f1_f0[s]:+.3f}")
    print(f"  Pearson r(drift_L2, F2-F0 delta) = {pearson_f2f0.statistic:+.3f} (p={pearson_f2f0.pvalue:.3f}, n=6)")
    print(f"  Pearson r(drift_L2, F1-F0 delta) = {pearson_f1f0.statistic:+.3f} (p={pearson_f1f0.pvalue:.3f}, n=6)")

    permutation_demo_rng = np.random.RandomState(42)
    permutation_demo_data = np.random.RandomState(0).normal(size=(45, 6)).astype(np.float32)
    permutation_demo_perm = permutation_demo_rng.permutation(permutation_demo_data.shape[0])
    permutation_preserves_session_mean = bool(np.allclose(
        permutation_demo_data.mean(axis=0), permutation_demo_data[permutation_demo_perm].mean(axis=0)
    ))

    result["supplementary_checks"] = {
        "f1_equivalence": {
            "claim": (
                "F1 = {p2p, noise_std, snr} is the first 3 columns of F2 = {p2p, noise_std, snr, "
                "pt_width, pt_ratio, repol_slope}; compute_unit_side_features_uncached computes "
                "the same per-unit raw scalars regardless of which feature_group is requested, so "
                "fitting train-only stats independently for 'f1' vs 'f2' must give bit-identical "
                "mean/std on the 3 shared columns. Verified empirically (not just by code reading)."
            ),
            "f1_mean": mean_f1.tolist(),
            "f1_std": std_f1.tolist(),
            "f2_mean_first_3": mean_f2[:3].tolist(),
            "f2_std_first_3": std_f2[:3].tolist(),
            "matches": f1_matches_f2_prefix,
            "conclusion": (
                "All drift results reported above for p2p/noise_std/snr apply identically to F1; "
                "no separate F1 computation is needed."
            ),
        },
        "shuffled_control_permutation_invariance": {
            "claim": (
                "FS1/FS2 are built by permuting F1/F2's per-unit z-vectors ACROSS UNITS WITHIN "
                "THE SAME SESSION (mc_maze/unit_side_features.py load_unit_side_features: "
                "'perm = generator.permutation(normalized.shape[0])' operates on one session's "
                "own row axis, never mixing values across sessions). Therefore FS1/FS2 share "
                "F1/F2's per-session mean, per-session distribution, and between/within-session "
                "variance decomposition EXACTLY, by construction -- a permutation of a finite set "
                "of numbers cannot change that set's mean. Consequently, ANY session-level bias "
                "(drift or otherwise) that affects F1/F2 affects FS1/FS2 identically, so a "
                "session-level-drift mechanism cannot -- even in principle, regardless of the "
                "empirical drift magnitude -- explain the observed F1<FS1 / F2<FS2 gaps in "
                "side_feature_ablation_v2. It could in principle still explain part of the F1<F0 "
                "/ F2<F0 gaps, since F0 carries no side features at all and so is immune to any "
                "such bias."
            ),
            "verified_numerically": permutation_preserves_session_mean,
        },
        "exploratory_drift_vs_ablation_r2_delta": {
            "caveat": (
                "NOT one of the 5 requested sub-tasks; included as a cheap cross-check using "
                "already-computed numbers. n=6 (one point per validation session): severely "
                "underpowered, not a hypothesis test."
            ),
            "per_val_session_drift_l2": drift_l2_by_val_session,
            "f2_minus_f0_r2_delta": {s: r2_delta_f2_f0[s] for s in val_sessions},
            "f1_minus_f0_r2_delta": {s: r2_delta_f1_f0[s] for s in val_sessions},
            "pearson_r_drift_vs_f2_minus_f0": {
                "r": float(pearson_f2f0.statistic), "p": float(pearson_f2f0.pvalue), "n": 6,
            },
            "pearson_r_drift_vs_f1_minus_f0": {
                "r": float(pearson_f1f0.statistic), "p": float(pearson_f1f0.pvalue), "n": 6,
            },
            "spearman_drift_vs_f2_minus_f0": {
                "rho": float(spearman_f2f0.statistic), "p": float(spearman_f2f0.pvalue), "n": 6,
            },
        },
    }
    print(f"  permutation invariance verified numerically: {permutation_preserves_session_mean}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
