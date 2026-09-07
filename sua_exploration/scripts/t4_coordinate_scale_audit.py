"""Descriptive scale audit of the center-out T4 identity-token descriptor.

Protocol: ``sua_exploration/docs/T4_COORDINATE_SCALE_AUDIT_PROTOCOL_20260814.md`` (frozen
2026-08-14, before any session was opened).

The question is the numerical scale of ``T4_i = [a_hat, c_hat, m, b_hat]`` per session and how far
it moves between sessions.  Three of the four coordinates carry session-specific firing-rate units,
so a session whose overall rate is higher presents a uniformly inflated descriptor to the identity
encoder.  Nothing here fits, trains, or scores a model.

The estimator is the sealed one in ``mc_maze/unit_side_features.py``; this module imports it and
never modifies it.  The per-session pass is assembled from that module's own helpers rather than
from ``compute_unit_side_features_uncached`` alone because the mean-firing-rate statistic needs the
same pool-trial rate matrix the cosine fit consumes, and opening each NWB twice would double the
I/O.  ``test_t4_coordinate_scale_audit.py`` asserts the assembled pass is bit-identical to the
sealed public entry point on real sessions; that test is the licence for this shortcut.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SUA_ROOT = REPO_ROOT / "sua_exploration"
if str(SUA_ROOT) not in sys.path:
    sys.path.insert(0, str(SUA_ROOT))


AUDIT_ID = "t4_coordinate_scale_audit_20260814"
SCHEMA_VERSION = 1
PROTOCOL_PATH = SUA_ROOT / "docs" / "T4_COORDINATE_SCALE_AUDIT_PROTOCOL_20260814.md"
MANIFEST_PATH = SUA_ROOT / "configs" / "subc_co_27_6_strict_train_val_manifest.json"
SUBC_DATA_ROOT = SUA_ROOT / "data" / "dandi_000688" / "sub-C"
SUBM_DATA_ROOT = SUA_ROOT / "data" / "dandi_000688" / "sub-M"
SOURCE_CACHE_ROOT = SUA_ROOT / "cache" / "dandi688_subc_co_v1"
DEFAULT_RESULT_ROOT = SUA_ROOT / "results" / AUDIT_ID

# Frozen estimator settings, taken from a2_matched_subject_shift_v2_core.py.
POOL_SIZE = 30
BIN_SIZE_MS = 20
WINDOW_SIZE = 50
TRIAL_RESULT_FILTER = "R"
SIGNAL_VIEW = "sua"
FEATURE_GROUP = "t4"

COORDINATES: tuple[str, ...] = ("a_hat", "c_hat", "m", "b_hat")

# Predeclared read rule, protocol section 5.  Not revised after seeing any number.
CROSS_SESSION_NEGLIGIBLE_RATIO = 1.5
CROSS_SESSION_SUBSTANTIAL_RATIO = 3.0
CROSS_SESSION_MIN_SUBSTANTIAL_COORDINATES = 3
H1_IMBALANCE_BAND: tuple[float, float] = (32.0, 61.0)
IMBALANCE_MATERIAL_MIN = 3.0
RATE_TRACKING_SPEARMAN_MIN = 0.7
RATE_PROPORTIONALITY_CV_MAX = 0.25

# The six formal-test sub-C sessions.  Named here only so the audit can prove it excluded them.
SEALED_TEST_SESSIONS: tuple[str, ...] = (
    "sub-C_ses-CO-20151113",
    "sub-C_ses-CO-20151116",
    "sub-C_ses-CO-20151117",
    "sub-C_ses-CO-20151119",
    "sub-C_ses-CO-20151120",
    "sub-C_ses-CO-20151201",
)

# Sealed implementation files whose bytes determine this audit's numbers.
SEALED_IMPLEMENTATION_PATHS: Mapping[str, Path] = {
    "unit_side_features": SUA_ROOT / "mc_maze" / "unit_side_features.py",
    "multisession_datamodule": SUA_ROOT / "mc_maze" / "multisession_datamodule.py",
    "a2_core": SUA_ROOT / "mc_maze" / "a2_matched_subject_shift_v2_core.py",
    "gpu_contract_common": SUA_ROOT / "mc_maze" / "gpu_contract_common.py",
    "manifest": MANIFEST_PATH,
    "protocol": PROTOCOL_PATH,
    "runner": Path(__file__).resolve(),
}


class AuditError(RuntimeError):
    """A frozen audit invariant was violated."""


# --------------------------------------------------------------------------------------
# Pure statistics.  No data access, so the test suite can exercise every branch offline.
# --------------------------------------------------------------------------------------


def _finite_1d(values: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise AuditError(f"{label}: empty")
    if not np.isfinite(array).all():
        raise AuditError(f"{label}: non-finite values")
    return array


def rms(values: np.ndarray, *, label: str = "values") -> float:
    array = _finite_1d(values, label=label)
    return float(np.sqrt(np.mean(array * array)))


def coordinate_scale_stats(values: np.ndarray, *, label: str = "coordinate") -> dict[str, float]:
    """Scale descriptors for one coordinate across a session's units.

    ``rms`` is the protocol's primary statistic: it is what sets the magnitude the coordinate
    presents at the injection MLP.  ``iqr`` and ``mad`` are the robust spreads; ``mad`` is the
    unscaled median absolute deviation about the median, not rescaled to a Gaussian sd.
    """
    array = _finite_1d(values, label=label)
    median = float(np.median(array))
    quartile_low, quartile_high = (float(value) for value in np.quantile(array, [0.25, 0.75]))
    return {
        "n_units": int(array.size),
        "rms": float(np.sqrt(np.mean(array * array))),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array)),
        "median": median,
        "iqr": quartile_high - quartile_low,
        "mad": float(np.median(np.abs(array - median))),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0 or not np.isfinite(denominator) or not np.isfinite(numerator):
        return None
    return float(numerator / denominator)


def within_session_imbalance(t4: np.ndarray) -> dict[str, float | None]:
    """Intercept scale versus signed-tuning scale inside one session.

    The pooled signed-tuning RMS is taken over the 2N concatenated ``a_hat`` and ``c_hat``
    values, which is the quantity H1's 32-61x figure was formed against.
    """
    values = np.asarray(t4, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise AuditError(f"T4 must be [units,4], got {values.shape}")
    a_hat, c_hat, modulation, b_hat = (values[:, index] for index in range(4))
    pooled_ac = rms(np.concatenate([a_hat, c_hat]), label="ac")
    return {
        "rms_ac_pooled": pooled_ac,
        "b_over_ac_pooled": _safe_ratio(rms(b_hat, label="b"), pooled_ac),
        "b_over_a": _safe_ratio(rms(b_hat, label="b"), rms(a_hat, label="a")),
        "b_over_c": _safe_ratio(rms(b_hat, label="b"), rms(c_hat, label="c")),
        "m_over_ac_pooled": _safe_ratio(rms(modulation, label="m"), pooled_ac),
    }


def cross_session_scale_ratio(scale_by_session: Mapping[str, float]) -> dict[str, Any]:
    """Largest-to-smallest session scale for one coordinate: the size of the gauge problem."""
    if not scale_by_session:
        raise AuditError("cross-session ratio needs at least one session")
    names = sorted(scale_by_session)
    values = np.asarray([float(scale_by_session[name]) for name in names], dtype=np.float64)
    if not np.isfinite(values).all():
        raise AuditError("cross-session ratio received non-finite session scales")
    argmax, argmin = int(np.argmax(values)), int(np.argmin(values))
    return {
        "n_sessions": len(names),
        "max": float(values[argmax]),
        "max_session": names[argmax],
        "min": float(values[argmin]),
        "min_session": names[argmin],
        "max_over_min": _safe_ratio(float(values[argmax]), float(values[argmin])),
        "median": float(np.median(values)),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(array.size, dtype=np.float64)
    ranks[order] = np.arange(1, array.size + 1, dtype=np.float64)
    # Ties take the mean of the ranks they span, which is what Spearman requires.
    sorted_values = array[order]
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = np.mean(ranks[order[start:stop]])
        start = stop
    return ranks


def spearman_rho(x_values: np.ndarray, y_values: np.ndarray) -> float | None:
    """Rank correlation without a SciPy dependency; ``None`` when a rank vector is constant."""
    x_array = _finite_1d(x_values, label="spearman x")
    y_array = _finite_1d(y_values, label="spearman y")
    if x_array.size != y_array.size:
        raise AuditError("spearman inputs must be the same length")
    if x_array.size < 3:
        raise AuditError("spearman needs at least three paired observations")
    x_ranks = _average_ranks(x_array) - np.mean(_average_ranks(x_array))
    y_ranks = _average_ranks(y_array) - np.mean(_average_ranks(y_array))
    denominator = float(np.sqrt(np.sum(x_ranks**2) * np.sum(y_ranks**2)))
    if denominator == 0.0:
        return None
    return float(np.sum(x_ranks * y_ranks) / denominator)


def coefficient_of_variation(values: np.ndarray) -> float | None:
    array = _finite_1d(values, label="cv")
    mean = float(np.mean(array))
    if mean == 0.0:
        return None
    return float(np.std(array) / abs(mean))


def classify_cross_session(ratio: float | None) -> str:
    if ratio is None or not np.isfinite(ratio):
        return "undefined"
    if ratio >= CROSS_SESSION_SUBSTANTIAL_RATIO:
        return "substantial"
    if ratio >= CROSS_SESSION_NEGLIGIBLE_RATIO:
        return "moderate"
    return "negligible"


def classify_imbalance(ratio: float | None) -> str:
    if ratio is None or not np.isfinite(ratio):
        return "undefined"
    if ratio >= H1_IMBALANCE_BAND[0]:
        return "reproduces_h1"
    if ratio >= IMBALANCE_MATERIAL_MIN:
        return "material_below_h1"
    return "balanced"


def evaluate_rate_tracking(
    mean_rates: Sequence[float],
    scale_by_coordinate: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Protocol 5.3: rank association and proportionality of scale to mean firing rate."""
    rates = _finite_1d(np.asarray(mean_rates, dtype=np.float64), label="mean rates")
    per_coordinate: dict[str, Any] = {}
    for name, scales in scale_by_coordinate.items():
        scale_array = _finite_1d(np.asarray(scales, dtype=np.float64), label=f"{name} scale")
        if scale_array.size != rates.size:
            raise AuditError(f"{name}: scale/rate length mismatch")
        rho = spearman_rho(rates, scale_array)
        ratios = np.divide(
            scale_array,
            rates,
            out=np.full(scale_array.shape, np.nan),
            where=rates != 0.0,
        )
        cv = coefficient_of_variation(ratios) if np.isfinite(ratios).all() else None
        per_coordinate[name] = {
            "spearman_rho_with_mean_rate": rho,
            "rank_association_passes": bool(rho is not None and rho >= RATE_TRACKING_SPEARMAN_MIN),
            "scale_over_mean_rate_mean": float(np.mean(ratios)) if np.isfinite(ratios).all() else None,
            "scale_over_mean_rate_cv": cv,
            "proportionality_passes": bool(cv is not None and cv < RATE_PROPORTIONALITY_CV_MAX),
        }
    intercept = per_coordinate.get("b_hat", {})
    per_coordinate["_declared_tracks_mean_rate"] = bool(
        intercept.get("rank_association_passes") and intercept.get("proportionality_passes")
    )
    return per_coordinate


def select_readings(
    *,
    cross_session_by_coordinate: Mapping[str, Mapping[str, Any]],
    median_raw_imbalance: float | None,
    median_standardized_imbalance: float | None,
    tracks_mean_rate: bool,
) -> dict[str, Any]:
    """Apply protocol 5.4.  More than one reading may fire; none is suppressed."""
    substantial = [
        name
        for name in COORDINATES
        if classify_cross_session(cross_session_by_coordinate[name].get("max_over_min"))
        == "substantial"
    ]
    negligible = all(
        classify_cross_session(cross_session_by_coordinate[name].get("max_over_min")) == "negligible"
        for name in COORDINATES
    )
    cross_session_substantial = len(substantial) >= CROSS_SESSION_MIN_SUBSTANTIAL_COORDINATES
    raw_band = classify_imbalance(median_raw_imbalance)
    standardized_band = classify_imbalance(median_standardized_imbalance)
    gauge = bool(cross_session_substantial and tracks_mean_rate)
    intercept_dominance = bool(
        raw_band == "reproduces_h1" or standardized_band == "reproduces_h1"
    )
    h1_special = bool(negligible and raw_band == "balanced" and standardized_band == "balanced")
    supported = [
        name
        for name, fired in (
            ("gauge_reading", gauge),
            ("intercept_dominance_reading", intercept_dominance),
            ("h1_is_special_reading", h1_special),
        )
        if fired
    ]
    return {
        "substantial_coordinates": substantial,
        "cross_session_substantial": cross_session_substantial,
        "all_coordinates_negligible": negligible,
        "raw_imbalance_band": raw_band,
        "standardized_imbalance_band": standardized_band,
        "tracks_mean_rate": tracks_mean_rate,
        "gauge_reading": gauge,
        "intercept_dominance_reading": intercept_dominance,
        "h1_is_special_reading": h1_special,
        "supported_readings": supported,
        "verdict": "none" if not supported else "+".join(supported),
    }


def standardize(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply the train-only z-score exactly as ``load_unit_side_features`` does."""
    return ((np.asarray(raw, dtype=np.float32) - mean) / std).astype(np.float32)


# --------------------------------------------------------------------------------------
# Data access.  Every numerical step below is a sealed function imported unchanged.
# --------------------------------------------------------------------------------------


def session_t4_and_rates(nwb_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """One NWB pass yielding raw T4 ``[units,4]`` and pool-trial rates ``[units,trials]`` in Hz.

    Mirrors ``_compute_tuning_features_uncached``'s SUA/``t4`` branch step for step, using that
    module's own helpers, so the returned T4 is the sealed estimator's output and the returned
    rates are the exact substrate it was fitted on.
    """
    from mc_maze.multisession_datamodule import (
        list_datamodule_rewarded_trials,
        session_name_from_path,
    )
    from mc_maze.unit_side_features import (
        _nearest_canonical_direction_index,
        _pool_trial_rate_matrix,
        _unit_tuning_features,
        enforce_direction_degeneracy_policy,
    )

    pool_trials = list_datamodule_rewarded_trials(
        nwb_path,
        bin_size_ms=BIN_SIZE_MS,
        window_size=WINDOW_SIZE,
        trial_result_filter=TRIAL_RESULT_FILTER,
    )
    if len(pool_trials) < POOL_SIZE:
        raise AuditError(
            f"{session_name_from_path(nwb_path)}: only {len(pool_trials)} rewarded trials pass the "
            f"datamodule filter; pool_size={POOL_SIZE} required"
        )
    pool_trials = pool_trials[:POOL_SIZE]

    direction_indices = np.array(
        [
            _nearest_canonical_direction_index(trial["target_dir"])
            if trial.get("target_dir") is not None
            else -1
            for trial in pool_trials
        ],
        dtype=np.int64,
    )
    present_directions = sorted({int(index) for index in direction_indices if index >= 0})

    rates, num_units = _pool_trial_rate_matrix(nwb_path, pool_trials)
    if len(present_directions) < 2:
        # Default policy is fail-closed; call the sealed gate rather than deciding here.
        enforce_direction_degeneracy_policy(
            present_directions=len(present_directions),
            num_channels=num_units,
            session_name=session_name_from_path(nwb_path),
        )

    t4 = np.zeros((num_units, 4), dtype=np.float32)
    zero_spike = zero_modulation = 0
    for unit_index in range(num_units):
        unit_t4, _t8, is_zero_spike, is_zero_modulation = _unit_tuning_features(
            rates[unit_index], direction_indices, present_directions
        )
        t4[unit_index] = unit_t4
        zero_spike += int(is_zero_spike)
        zero_modulation += int(is_zero_modulation)

    metadata = {
        "n_units": int(num_units),
        "n_pool_trials": int(len(pool_trials)),
        "present_directions": present_directions,
        "n_present_directions": len(present_directions),
        "n_unlabelled_pool_trials": int(np.sum(direction_indices < 0)),
        "zero_spike_unit_count": zero_spike,
        "zero_modulation_unit_count": zero_modulation,
    }
    return t4, np.asarray(rates, dtype=np.float64), metadata


def session_row(session: str, nwb_path: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Every per-session number the protocol asks for, plus the raw descriptor itself."""
    t4, rates, metadata = session_t4_and_rates(nwb_path)
    per_unit_mean_rate = rates.mean(axis=1)
    row: dict[str, Any] = {
        "session": session,
        "nwb_name": nwb_path.name,
        **metadata,
        "mean_rate_hz": float(np.mean(per_unit_mean_rate)),
        "median_rate_hz": float(np.median(per_unit_mean_rate)),
        "rate_hz_sd_across_units": float(np.std(per_unit_mean_rate)),
        "raw_coordinates": {
            name: coordinate_scale_stats(t4[:, index], label=f"{session}:{name}")
            for index, name in enumerate(COORDINATES)
        },
        "raw_imbalance": within_session_imbalance(t4),
    }
    return row, t4


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment_fingerprint() -> dict[str, Any]:
    import h5py
    import pynwb

    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "pynwb": pynwb.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor_count": os.cpu_count(),
        "thread_env": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
                "PYTHONNOUSERSITE",
            )
        },
        "nice": os.nice(0),
    }


def resolve_rosters() -> dict[str, list[str]]:
    from mc_maze.gpu_contract_common import SUBM_EXTERNAL_SESSIONS

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["session_splits"]
    rosters = {
        "subC_train": list(manifest["train"]),
        "subC_val": list(manifest["val"]),
        "subM_external": list(SUBM_EXTERNAL_SESSIONS),
    }
    sealed = set(SEALED_TEST_SESSIONS)
    if tuple(manifest["test"]) != SEALED_TEST_SESSIONS:
        raise AuditError("sealed formal-test roster drifted from the manifest")
    for name, sessions in rosters.items():
        overlap = sorted(set(sessions) & sealed)
        if overlap:
            raise AuditError(f"{name} would open sealed formal-test sessions: {overlap}")
    if len(rosters["subC_train"]) != 27 or len(rosters["subC_val"]) != 6:
        raise AuditError("sub-C roster counts drifted from the frozen 27/6 manifest")
    if len(rosters["subM_external"]) != 15:
        raise AuditError("sub-M external roster must contain exactly 15 sessions")
    return rosters


def session_path(session: str) -> Path:
    root = SUBC_DATA_ROOT if session.startswith("sub-C") else SUBM_DATA_ROOT
    path = (root / f"{session}_behavior+ecephys.nwb").resolve()
    if path.parent != root.resolve() or not path.is_file():
        raise AuditError(f"missing NWB for {session}: {path}")
    return path


def _scale_map(rows: Sequence[Mapping[str, Any]], block: str, coordinate: str) -> dict[str, float]:
    return {str(row["session"]): float(row[block][coordinate]["rms"]) for row in rows}


def _cohort_summary(rows: Sequence[Mapping[str, Any]], block: str, imbalance_block: str) -> dict[str, Any]:
    if not rows:
        return {}
    imbalances = [
        float(row[imbalance_block]["b_over_ac_pooled"])
        for row in rows
        if row[imbalance_block]["b_over_ac_pooled"] is not None
    ]
    return {
        "n_sessions": len(rows),
        "cross_session_scale_ratio": {
            name: cross_session_scale_ratio(_scale_map(rows, block, name)) for name in COORDINATES
        },
        "median_rms": {
            name: float(np.median([float(row[block][name]["rms"]) for row in rows]))
            for name in COORDINATES
        },
        "imbalance_b_over_ac_pooled": {
            "median": float(np.median(imbalances)) if imbalances else None,
            "min": float(np.min(imbalances)) if imbalances else None,
            "max": float(np.max(imbalances)) if imbalances else None,
            "band": classify_imbalance(float(np.median(imbalances)) if imbalances else None),
        },
    }


def build_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Cohort and pooled aggregates plus the predeclared verdict."""
    by_cohort = {
        cohort: [row for row in rows if row["cohort"] == cohort]
        for cohort in ("subC_train", "subC_val", "subM_external")
    }
    sub_c_rows = by_cohort["subC_train"] + by_cohort["subC_val"]
    analysis: dict[str, Any] = {"raw": {}, "standardized": {}}
    for block, imbalance_block, key in (
        ("raw_coordinates", "raw_imbalance", "raw"),
        ("standardized_coordinates", "standardized_imbalance", "standardized"),
    ):
        if block not in rows[0]:
            analysis.pop(key)
            continue
        analysis[key] = {
            "pooled_all_sessions": _cohort_summary(list(rows), block, imbalance_block),
            "subC_development": _cohort_summary(sub_c_rows, block, imbalance_block),
            "subM_external": _cohort_summary(by_cohort["subM_external"], block, imbalance_block),
        }

    mean_rates = [float(row["mean_rate_hz"]) for row in rows]
    analysis["firing_rate"] = {
        "cross_session_mean_rate_hz": cross_session_scale_ratio(
            {str(row["session"]): float(row["mean_rate_hz"]) for row in rows}
        ),
        "raw_scale_tracking": evaluate_rate_tracking(
            mean_rates,
            {name: [float(row["raw_coordinates"][name]["rms"]) for row in rows] for name in COORDINATES},
        ),
    }

    pooled_raw = analysis["raw"]["pooled_all_sessions"]
    standardized_median = None
    if "standardized" in analysis:
        standardized_median = analysis["standardized"]["pooled_all_sessions"][
            "imbalance_b_over_ac_pooled"
        ]["median"]
    analysis["readings"] = select_readings(
        cross_session_by_coordinate=pooled_raw["cross_session_scale_ratio"],
        median_raw_imbalance=pooled_raw["imbalance_b_over_ac_pooled"]["median"],
        median_standardized_imbalance=standardized_median,
        tracks_mean_rate=bool(analysis["firing_rate"]["raw_scale_tracking"]["_declared_tracks_mean_rate"]),
    )
    return analysis


def normalizer_inventory(mean: np.ndarray, std: np.ndarray, train_sessions: Sequence[str]) -> dict[str, Any]:
    """Protocol 4.6: what normalization the center-out path applies to T4, and its scope."""
    return {
        "per_session_normalizer": "none",
        "applied_normalizer": "single train-pooled per-column z-score",
        "fitted_by": "mc_maze.unit_side_features.fit_side_feature_stats",
        "fit_scope": "source sub-C train sessions only; never refit per session or per cohort",
        "fit_session_count": len(train_sessions),
        "robustification": "values clipped to per-column [0.01, 0.99] train quantiles before mean/std",
        "applied_at": "mc_maze.unit_side_features.load_unit_side_features, (raw - mean) / std",
        "reused_unchanged_on_external_cohort": True,
        "column_order": list(COORDINATES),
        "mean": [float(value) for value in np.asarray(mean).reshape(-1)],
        "std": [float(value) for value in np.asarray(std).reshape(-1)],
    }


def run_audit(*, result_root: Path, hash_inputs: str, verify_normalizer: bool) -> dict[str, Any]:
    from mc_maze.unit_side_features import _fit_robust_stats

    started = time.time()
    rosters = resolve_rosters()
    rows: list[dict[str, Any]] = []
    raw_by_session: dict[str, np.ndarray] = {}
    input_records: list[dict[str, Any]] = []

    for cohort, sessions in rosters.items():
        for session in sessions:
            path = session_path(session)
            row, t4 = session_row(session, path)
            row["cohort"] = cohort
            row["raw_t4_sha256"] = _array_sha256(t4)
            rows.append(row)
            raw_by_session[session] = t4
            stat = path.stat()
            record = {
                "session": session,
                "cohort": cohort,
                "path": str(path),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
            if hash_inputs == "full":
                record["sha256"] = _sha256_file(path)
            input_records.append(record)
            print(
                f"[{len(rows):2d}/48] {session}  units={row['n_units']:3d}  "
                f"rate={row['mean_rate_hz']:6.2f}Hz  "
                f"rms_b={row['raw_coordinates']['b_hat']['rms']:7.3f}  "
                f"b/ac={row['raw_imbalance']['b_over_ac_pooled']:6.2f}  "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )

    train_stack = np.concatenate([raw_by_session[name] for name in rosters["subC_train"]], axis=0)
    mean, std, clipped = _fit_robust_stats(train_stack)

    normalizer_check: dict[str, Any] = {"verified_against_fit_side_feature_stats": False}
    if verify_normalizer:
        from mc_maze.unit_side_features import fit_side_feature_stats

        sealed_mean, sealed_std = fit_side_feature_stats(
            [session_path(name) for name in rosters["subC_train"]],
            feature_group=FEATURE_GROUP,
            pool_size=POOL_SIZE,
            cache_dir=SOURCE_CACHE_ROOT,
            bin_size_ms=BIN_SIZE_MS,
            window_size=WINDOW_SIZE,
            trial_result_filter=TRIAL_RESULT_FILTER,
            signal_view=SIGNAL_VIEW,
        )
        if not (
            np.array_equal(np.asarray(sealed_mean), mean) and np.array_equal(np.asarray(sealed_std), std)
        ):
            raise AuditError(
                "refit train normalizer differs from the sealed fit_side_feature_stats output: "
                f"mean {sealed_mean} vs {mean}, std {sealed_std} vs {std}"
            )
        normalizer_check = {
            "verified_against_fit_side_feature_stats": True,
            "sealed_mean": [float(value) for value in np.asarray(sealed_mean).reshape(-1)],
            "sealed_std": [float(value) for value in np.asarray(sealed_std).reshape(-1)],
        }

    for row in rows:
        standardized = standardize(raw_by_session[row["session"]], mean, std)
        row["standardized_coordinates"] = {
            name: coordinate_scale_stats(standardized[:, index], label=f"{row['session']}:z:{name}")
            for index, name in enumerate(COORDINATES)
        }
        row["standardized_imbalance"] = within_session_imbalance(standardized)

    analysis = build_analysis(rows)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "kind": "t4_coordinate_scale_audit",
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "utc_completed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - started, 1),
        "estimator": {
            "source": "mc_maze/unit_side_features.py (sealed, imported unchanged)",
            "feature_group": FEATURE_GROUP,
            "pool_size": POOL_SIZE,
            "bin_size_ms": BIN_SIZE_MS,
            "window_size": WINDOW_SIZE,
            "trial_result_filter": TRIAL_RESULT_FILTER,
            "signal_view": SIGNAL_VIEW,
            "coordinate_order": list(COORDINATES),
            "rate_units": "Hz (spikes / trial duration in seconds)",
            "fit_target": "per-direction mean rates over the calibration-prefix trials",
        },
        "predeclared_thresholds": {
            "cross_session_negligible_ratio": CROSS_SESSION_NEGLIGIBLE_RATIO,
            "cross_session_substantial_ratio": CROSS_SESSION_SUBSTANTIAL_RATIO,
            "cross_session_min_substantial_coordinates": CROSS_SESSION_MIN_SUBSTANTIAL_COORDINATES,
            "h1_imbalance_band": list(H1_IMBALANCE_BAND),
            "imbalance_material_min": IMBALANCE_MATERIAL_MIN,
            "rate_tracking_spearman_min": RATE_TRACKING_SPEARMAN_MIN,
            "rate_proportionality_cv_max": RATE_PROPORTIONALITY_CV_MAX,
        },
        "rosters": rosters,
        "sealed_sessions_excluded": list(SEALED_TEST_SESSIONS),
        "sealed_sessions_opened": [],
        "inputs": input_records,
        "input_hash_mode": hash_inputs,
        "sealed_implementation_sha256": {
            name: _sha256_file(path) for name, path in SEALED_IMPLEMENTATION_PATHS.items()
        },
        "environment": _environment_fingerprint(),
        "normalizer": {
            **normalizer_inventory(mean, std, rosters["subC_train"]),
            "train_clipped_scalar_count": int(clipped),
            **normalizer_check,
        },
        "per_session": rows,
        "analysis": analysis,
    }

    from mc_maze.a2_matched_subject_shift_v2_core import write_immutable_json

    result_root.mkdir(parents=True, exist_ok=True)
    body, sidecar, digest = write_immutable_json(result_root / "receipt.json", receipt)
    print(f"\nreceipt: {body}\nsidecar: {sidecar}\nsha256:  {digest}", flush=True)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--input-hash", choices=("full", "fingerprint"), default="full")
    parser.add_argument("--skip-normalizer-verification", action="store_true")
    args = parser.parse_args(argv)
    receipt = run_audit(
        result_root=args.result_root,
        hash_inputs=args.input_hash,
        verify_normalizer=not args.skip_normalizer_verification,
    )
    readings = receipt["analysis"]["readings"]
    print(f"verdict: {readings['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
