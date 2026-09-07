"""H-U: a zero-learned-parameter, label-free 4-D per-unit descriptor for H1.

This module is a standalone numerical builder.  It never reads a behaviour
array.  The public builder :func:`hu_from_support_neural` accepts only neural
counts, the eval-valid mask, and TrialNum used as a *segmentation* key — never
as a regression target.  Velocity, kinematics, direction, and position are
absent from the signature.

The four components are rotation-invariant per-unit scalars (width-matched to
the H32 CarrierID carrier).  No principal components, SVD, or other
session-defined frames are used::

    [mean_rate, fano, log_isi_cv, autocorr_tau_s]

The fourth component is the exponential timescale ``-dt / log(acf_1)`` of the
20 ms count series.  A discrete 1/e lag-crossing saturates at one bin for
every live H1 unit and would make the descriptor rank-deficient.

Silent units (zero spike sum on the support) emit a documented zero row rather
than NaN; they are recorded in the per-call audit.  Channel 66 is dead on the
public H1 held-in set (H-NF D1); that is a property of the data, not a fill
heuristic applied to live units.

Normalizer policy is *not* implemented here.  The H32 harness applies the
existing source-only scalar RMS
``s_src=sqrt(mean(C_src_raw**2)); C_norm=C_raw/max(s_src,1e-12)`` to the
stacked H-U source cache.  Per-column z-scoring is not used, matching H-C /
H-LS / H-C0.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


CARRIER_WIDTH = 4
H1_NUM_NEURONS = 176
BIN_SECONDS = 0.02  # native FALCON H1 bin; 5 bins = 100 ms H1 M=4 blocks
SUPPORT_TRIALS = 4
EPS = 1.0e-12
FANO_MEAN_FLOOR = 1.0e-6
MIN_ISIS = 3
MIN_ACF_BINS = 8

FEATURE_NAMES = ("mean_rate", "fano", "log_isi_cv", "autocorr_tau_s")
VERSION = "HU-v1"
NO_LABEL_PROOF_METHOD_SIGNATURE = "signature_has_no_behaviour_parameter"
NO_LABEL_PROOF_METHOD_NAN_IDENTITY = "identical_output_when_behaviour_arrays_are_nan"
NO_LABEL_PROOF_METHODS = (
    NO_LABEL_PROOF_METHOD_SIGNATURE,
    NO_LABEL_PROOF_METHOD_NAN_IDENTITY,
)
BEHAVIOUR_PARAMETER_NAMES = frozenset(
    {
        "labels",
        "velocity",
        "kinematics",
        "behavior",
        "behaviour",
        "covariates",
        "target_kinematics",
        "direction",
        "position",
        "cursor_vel",
        "cursor_pos",
    }
)

# Frozen after the CPU dry-run on the 11 fold-0 source recordings: channel 66
# has a zero spike sum on every public held-in calibration recording.
DECLARED_DEAD_CHANNELS: frozenset[int] = frozenset({66})


class CarrierHuError(ValueError):
    """Fail-closed violation of the H-U label-free descriptor contract."""


@dataclass(frozen=True)
class HuAudit:
    """Per-call provenance: no behaviour in, which units were silent."""

    n_bins: int
    n_channels: int
    n_silent_channels: int
    silent_channels: tuple[int, ...]
    declared_dead_channels: tuple[int, ...]
    feature_names: tuple[str, ...]
    version: str
    used_velocity: bool
    used_behaviour_labels: bool
    bin_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_bins": self.n_bins,
            "n_channels": self.n_channels,
            "n_silent_channels": self.n_silent_channels,
            "silent_channels": list(self.silent_channels),
            "declared_dead_channels": list(self.declared_dead_channels),
            "feature_names": list(self.feature_names),
            "version": self.version,
            "used_velocity": self.used_velocity,
            "used_behaviour_labels": self.used_behaviour_labels,
            "bin_seconds": self.bin_seconds,
            "no_label_proof_methods": list(NO_LABEL_PROOF_METHODS),
        }


def assert_no_behaviour_in_signature(fn: Any) -> None:
    """Provenance assertion: the builder must not accept a behaviour array."""

    names = set(inspect.signature(fn).parameters)
    leaked = sorted(names & BEHAVIOUR_PARAMETER_NAMES)
    if leaked:
        raise CarrierHuError(
            f"H-U builder {getattr(fn, '__name__', fn)} accepts behaviour "
            f"parameter(s) {leaked}; {NO_LABEL_PROOF_METHOD_SIGNATURE}"
        )


def _as_finite_f64(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise CarrierHuError(f"{name} must be {ndim}-D, got {array.ndim}-D {array.shape}")
    if not np.isfinite(array).all():
        raise CarrierHuError(f"{name} must be finite")
    return array


def neural_support_counts(
    neural: np.ndarray,
    eval_mask: np.ndarray,
    trial_num: np.ndarray,
    trial_values: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Select eval-valid support bins using TrialNum as a segmentation key only.

    Velocity is not an argument.  ``eval_mask`` is the FALCON eval-valid mask,
    not a kinematic label.
    """

    spikes = _as_finite_f64(neural, name="neural", ndim=2)
    mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
    labels = np.asarray(trial_num, dtype=np.float64).reshape(-1)
    if mask.shape != (spikes.shape[0],) or labels.shape != (spikes.shape[0],):
        raise CarrierHuError("neural/eval_mask/trial_num length mismatch")
    values = tuple(float(value) for value in trial_values)
    if len(values) != SUPPORT_TRIALS:
        raise CarrierHuError(f"H-U support requires exactly {SUPPORT_TRIALS} TrialNum values")
    in_support = np.isin(labels, np.asarray(values, dtype=np.float64))
    keep = mask & np.isfinite(labels) & in_support
    if int(keep.sum()) < MIN_ACF_BINS:
        raise CarrierHuError(
            f"H-U support has {int(keep.sum())} eval-valid bins, need >= {MIN_ACF_BINS}"
        )
    return spikes[keep], labels[keep]


def _isis_seconds(counts: np.ndarray, *, bin_seconds: float) -> np.ndarray | None:
    """Approximate ISIs by placing ``n`` events uniformly inside a bin of count n."""

    times: list[float] = []
    for index, raw in enumerate(np.asarray(counts, dtype=np.float64)):
        if raw <= 0.0:
            continue
        n_spikes = int(np.round(raw))
        if n_spikes <= 0:
            continue
        if n_spikes == 1:
            times.append((index + 0.5) * bin_seconds)
            continue
        for k in range(n_spikes):
            times.append((index + (k + 0.5) / n_spikes) * bin_seconds)
    if len(times) < MIN_ISIS + 1:
        return None
    intervals = np.diff(np.asarray(times, dtype=np.float64))
    if intervals.size < MIN_ISIS or not np.isfinite(intervals).all() or np.any(intervals <= 0.0):
        return None
    return intervals


def _log_isi_cv(counts: np.ndarray, *, bin_seconds: float) -> float | None:
    intervals = _isis_seconds(counts, bin_seconds=bin_seconds)
    if intervals is None:
        return None
    mean = float(intervals.mean())
    if mean <= EPS:
        return None
    cv = float(intervals.std(ddof=0) / mean)
    return float(np.log(max(cv, EPS)))


def _autocorr_tau_seconds(counts: np.ndarray, *, bin_seconds: float) -> float | None:
    """Exponential autocorrelation timescale from lag-1 ACF.

    On 20 ms H1 bins the discrete 1/e-crossing saturates at lag 1 for essentially
    every live unit, which would make this component constant and the 4-D
    descriptor rank-deficient.  The continuous fit ``tau = -dt / log(acf_1)``
    (with non-positive ACF mapped to 0) is the same estimand and varies.
    """

    series = np.asarray(counts, dtype=np.float64)
    if series.size < MIN_ACF_BINS:
        return None
    centered = series - float(series.mean())
    variance = float(np.dot(centered, centered))
    if variance <= EPS:
        return None
    acf1 = float(np.dot(centered[:-1], centered[1:]) / variance)
    if not np.isfinite(acf1):
        return None
    if acf1 <= EPS:
        return 0.0
    acf1 = min(acf1, 1.0 - 1.0e-12)
    return float(-bin_seconds / np.log(acf1))


def hu_from_counts(
    counts: np.ndarray,
    *,
    bin_seconds: float = BIN_SECONDS,
    declared_dead_channels: frozenset[int] | set[int] = frozenset(),
) -> tuple[np.ndarray, HuAudit]:
    """Build the 4-D per-unit descriptor from a [T, N] count tensor.

    Signature contains no behaviour parameter.  Each column is an independent
    per-unit statistic: mutating unit j cannot change unit i's row.
    """

    assert_no_behaviour_in_signature(hu_from_counts)
    matrix = _as_finite_f64(counts, name="counts", ndim=2)
    if matrix.shape[0] < MIN_ACF_BINS or matrix.shape[1] <= 0:
        raise CarrierHuError(f"H-U counts must be [T>= {MIN_ACF_BINS}, N>0], got {matrix.shape}")
    if np.any(matrix < -EPS):
        raise CarrierHuError("H-U counts must be nonnegative")
    n_bins, n_channels = matrix.shape
    dead = frozenset(int(ch) for ch in declared_dead_channels)
    features = np.zeros((n_channels, CARRIER_WIDTH), dtype=np.float64)
    silent: list[int] = []
    for channel in range(n_channels):
        column = matrix[:, channel]
        spike_sum = float(column.sum())
        if channel in dead or spike_sum <= EPS:
            silent.append(channel)
            continue
        mean_count = float(column.mean())
        mean_rate = mean_count / float(bin_seconds)
        if mean_count <= FANO_MEAN_FLOOR:
            silent.append(channel)
            continue
        fano = float(column.var(ddof=0) / mean_count)
        log_cv = _log_isi_cv(column, bin_seconds=bin_seconds)
        tau = _autocorr_tau_seconds(column, bin_seconds=bin_seconds)
        if log_cv is None or tau is None or not np.isfinite([mean_rate, fano, log_cv, tau]).all():
            silent.append(channel)
            continue
        features[channel] = (mean_rate, fano, log_cv, tau)
    if not np.isfinite(features).all():
        raise CarrierHuError("H-U descriptor produced a non-finite value")
    live = n_channels - len(silent)
    if live <= 0:
        raise CarrierHuError("H-U descriptor is empty: every unit was silent or degenerate")
    audit = HuAudit(
        n_bins=n_bins,
        n_channels=n_channels,
        n_silent_channels=len(silent),
        silent_channels=tuple(silent),
        declared_dead_channels=tuple(sorted(dead)),
        feature_names=FEATURE_NAMES,
        version=VERSION,
        used_velocity=False,
        used_behaviour_labels=False,
        bin_seconds=float(bin_seconds),
    )
    return features, audit


def hu_from_support_neural(
    neural: np.ndarray,
    eval_mask: np.ndarray,
    trial_num: np.ndarray,
    trial_values: Sequence[float],
    *,
    bin_seconds: float = BIN_SECONDS,
    declared_dead_channels: frozenset[int] | set[int] = frozenset(),
) -> tuple[np.ndarray, HuAudit]:
    """Public H-U builder.  No behaviour array is in the signature."""

    assert_no_behaviour_in_signature(hu_from_support_neural)
    counts, _trial_ids = neural_support_counts(neural, eval_mask, trial_num, trial_values)
    return hu_from_counts(
        counts,
        bin_seconds=bin_seconds,
        declared_dead_channels=declared_dead_channels,
    )


def hu_from_record(
    record: Any,
    trial_values: Sequence[float],
    *,
    declared_dead_channels: frozenset[int] | set[int] = DECLARED_DEAD_CHANNELS,
) -> tuple[np.ndarray, HuAudit]:
    """Record adapter.  Reads ``neural``, ``eval_mask``, ``trial_num`` only."""

    return hu_from_support_neural(
        record.neural,
        record.eval_mask,
        record.trial_num,
        trial_values,
        declared_dead_channels=declared_dead_channels,
    )


def descriptor_rank(features: np.ndarray, *, silent: Sequence[int] = ()) -> int:
    """Matrix rank of the live-unit submatrix.  Dead rows are excluded."""

    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != CARRIER_WIDTH:
        raise CarrierHuError(f"rank requires [N,4], got {matrix.shape}")
    keep = np.ones(matrix.shape[0], dtype=bool)
    for index in silent:
        keep[int(index)] = False
    live = matrix[keep]
    if live.shape[0] == 0:
        return 0
    centered = live - live.mean(axis=0, keepdims=True)
    return int(np.linalg.matrix_rank(centered, tol=1.0e-8))


def per_unit_column_stats(features: np.ndarray, *, silent: Sequence[int] = ()) -> dict[str, Any]:
    matrix = np.asarray(features, dtype=np.float64)
    keep = np.ones(matrix.shape[0], dtype=bool)
    for index in silent:
        keep[int(index)] = False
    live = matrix[keep]
    body: dict[str, Any] = {
        "n_live": int(live.shape[0]),
        "n_silent": int(matrix.shape[0] - live.shape[0]),
        "rank_live": descriptor_rank(matrix, silent=silent),
        "columns": {},
    }
    for dim, name in enumerate(FEATURE_NAMES):
        column = live[:, dim]
        body["columns"][name] = {
            "mean": float(column.mean()),
            "std": float(column.std(ddof=0)),
            "min": float(column.min()),
            "max": float(column.max()),
            "n_unique": int(np.unique(np.round(column, 12)).size),
            "finite": bool(np.isfinite(column).all()),
            "degenerate_constant": bool(column.std(ddof=0) <= EPS),
        }
    return body


def pearson_corr_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """[4,4] Pearson correlation of columns.  Frame-free; no SVD."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != CARRIER_WIDTH:
        raise CarrierHuError(f"correlation requires matching [N,4] arrays, got {a.shape} vs {b.shape}")
    corr = np.empty((CARRIER_WIDTH, CARRIER_WIDTH), dtype=np.float64)
    for i in range(CARRIER_WIDTH):
        for j in range(CARRIER_WIDTH):
            x = a[:, i]
            y = b[:, j]
            if x.std(ddof=0) <= EPS or y.std(ddof=0) <= EPS:
                corr[i, j] = 0.0
                continue
            corr[i, j] = float(np.corrcoef(x, y)[0, 1])
    return corr


def canonical_hu_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


assert_no_behaviour_in_signature(hu_from_counts)
assert_no_behaviour_in_signature(hu_from_support_neural)
assert_no_behaviour_in_signature(neural_support_counts)
