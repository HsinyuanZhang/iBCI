"""Aligned, source-only rSyn3 carrier primitives for an independent M1 variant.

This module is deliberately not imported by the frozen cross-session program.
It reproduces the rSyn3 source-fit law while using the installed Falcon
``bin_units`` implementation on the M1 EMG end-timestamp clock.  The carrier
rows consequently retain the exact NWB Units row order used for the binned
neural array.  It never opens target query covariates or labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


BIN_SECONDS = 0.02
SUPPORT_TRIALS = 10
RANK = 3
CARRIER_DIM = 4
CAUSAL_WINDOWS = (1, 5, 10, 25, 50, 100)


class AlignedCarrierError(RuntimeError):
    """Raised when source-only or unit/time alignment contracts drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AlignedCarrierError(message)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _unit_id_sha256(unit_ids: np.ndarray) -> str:
    """Hash IDs without relying on object-array memory representation."""
    values = np.asarray(unit_ids).reshape(-1)
    payload = json.dumps([str(value) for value in values], separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rectify(emg: np.ndarray) -> np.ndarray:
    values = np.asarray(emg, dtype=np.float64)
    _require(values.ndim == 2 and values.size > 0 and np.isfinite(values).all(),
             "EMG must be finite, nonempty, and two-dimensional")
    return np.maximum(values, 0.0)


def _movement_bounds(row: Mapping[str, object]) -> tuple[float, float]:
    onset = float(row["move_onset_time"])
    if not math.isfinite(onset):
        onset = float(row["start_time"])
    stop = float(row["stop_time"])
    _require(math.isfinite(onset) and math.isfinite(stop) and stop > onset,
             "trial movement bounds are invalid")
    return onset, stop


def _trial_rows(timestamps: np.ndarray, trials: Sequence[Mapping[str, object]],
                trial_stop: int) -> tuple[np.ndarray, np.ndarray]:
    """Return only movement rows of declared trials, preserving global indices."""
    _require(1 <= trial_stop <= len(trials), "declared trial range is invalid")
    selected: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    for trial_id, row in enumerate(trials[:trial_stop]):
        onset, stop = _movement_bounds(row)
        # M1's official clock treats each supplied EMG timestamp as a bin end.
        # Match the legacy support slicing exactly: [searchsorted(onset, left),
        # searchsorted(stop, right)).
        left = int(np.searchsorted(timestamps, onset, side="left"))
        right = int(np.searchsorted(timestamps, stop, side="right"))
        _require(0 <= left < right <= len(timestamps),
                 f"trial {trial_id} has no valid EMG-clock movement rows")
        selected.append(np.arange(left, right, dtype=np.int64))
        ids.append(np.full(right - left, trial_id, dtype=np.int64))
    return np.concatenate(selected), np.concatenate(ids)


@dataclass(frozen=True)
class AlignedSupport:
    """Declared M1 support/training EMG plus M10 rates on one shared clock.

    ``emg`` can span the declared EMG prefix (e.g. source trials ``[0,310)``),
    while ``counts`` and ``rates`` span only ``[0, neural_trial_stop)``.  The
    respective ``*_global_time_index`` fields index the original M1 EMG
    timestamp clock; they allow a caller to prove the row correspondence.
    """

    path: Path
    unit_ids: np.ndarray
    unit_id_sha256: str
    timestamps: np.ndarray
    emg: np.ndarray
    emg_trial_ids: np.ndarray
    emg_global_time_index: np.ndarray
    counts: np.ndarray
    rates: np.ndarray
    rate_trial_ids: np.ndarray
    rate_global_time_index: np.ndarray
    channel_names: tuple[str, ...]
    emg_trial_stop: int
    neural_trial_stop: int

    def metadata(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "unit_count": int(self.unit_ids.size),
            "unit_id_sha256": self.unit_id_sha256,
            "timestamp_sha256": _array_sha256(self.timestamps),
            "emg_sha256": _array_sha256(self.emg),
            "counts_sha256": _array_sha256(self.counts),
            "rates_sha256": _array_sha256(self.rates),
            "emg_global_time_index_sha256": _array_sha256(self.emg_global_time_index),
            "rate_global_time_index_sha256": _array_sha256(self.rate_global_time_index),
            "emg_trial_stop": int(self.emg_trial_stop),
            "neural_trial_stop": int(self.neural_trial_stop),
            "clock": "official_falcon_bin_units_end_at_t",
        }


def load_aligned_support(path: Path, *, emg_trial_stop: int | None,
                         neural_trial_stop: int = SUPPORT_TRIALS) -> AlignedSupport:
    """Load declared M1 trial prefixes without accessing query labels.

    The official M1 loader passes EMG timestamps to ``bin_units`` with its
    default ``is_timestamp_bin_start=False``.  This function does the same,
    bins the chronological timestamp prefix through M10, and then selects
    movement rows.  Binning a contiguous prefix before slicing avoids the
    helper's discontinuity behavior and keeps each retained count aligned to
    its original global EMG-clock row.
    """
    from pynwb import NWBHDF5IO
    from falcon_challenge.dataloaders import bin_units

    resolved = Path(path).resolve()
    _require(resolved.is_file(), f"missing NWB file: {resolved}")
    with NWBHDF5IO(str(resolved), "r", load_namespaces=True) as io:
        nwb = io.read()
        trials_frame = nwb.trials.to_dataframe().reset_index()
        trials = [dict(row) for _, row in trials_frame.iterrows()]
        n_trials = len(trials)
        emg_stop = n_trials if emg_trial_stop is None else int(emg_trial_stop)
        _require(1 <= neural_trial_stop <= emg_stop <= n_trials,
                 "declared EMG/neural trial stops are inconsistent")

        raw_emg = nwb.acquisition["preprocessed_emg"]
        channel_names = tuple(raw_emg.time_series.keys())
        _require(channel_names, "M1 preprocessed_emg has no channels")
        first = raw_emg.get_timeseries(channel_names[0])
        timestamps = np.asarray(first.timestamps[:], dtype=np.float64)
        _require(timestamps.ndim == 1 and timestamps.size > 1 and np.isfinite(timestamps).all()
                 and np.all(np.diff(timestamps) > 0.0), "invalid M1 EMG timestamp clock")
        for channel in channel_names[1:]:
            candidate = np.asarray(raw_emg.get_timeseries(channel).timestamps[:], dtype=np.float64)
            _require(np.array_equal(candidate, timestamps),
                     "M1 EMG channels do not share the official timestamp clock")

        emg_indices, emg_trial_ids = _trial_rows(timestamps, trials, emg_stop)
        rate_indices, rate_trial_ids = _trial_rows(timestamps, trials, neural_trial_stop)
        # Only declared movement rows are used to materialize EMG values.
        emg = np.column_stack([
            np.asarray(raw_emg.get_timeseries(channel).data[emg_indices], dtype=np.float64)
            for channel in channel_names
        ])
        _require(emg.shape == (emg_indices.size, len(channel_names)) and np.isfinite(emg).all(),
                 "invalid declared EMG values")

        units = nwb.units.to_dataframe()
        _require("spike_times" in units and len(units) > 0, "NWB Units lacks spike_times")
        unit_ids = np.asarray(units.index.to_numpy(copy=True))
        _require(unit_ids.ndim == 1 and unit_ids.size == len(units), "invalid NWB unit IDs")

        # Construct only the contiguous prefix ending with the final allowed
        # neural-support trial.  `bin_units` may read full spike lists, as its
        # official API requires, but it receives no target-query time bins.
        _, neural_stop_time = _movement_bounds(trials[neural_trial_stop - 1])
        prefix_stop = int(np.searchsorted(timestamps, neural_stop_time, side="right"))
        _require(prefix_stop > int(rate_indices[-1]), "M10 clock prefix misses support rows")
        official_counts = np.asarray(
            bin_units(units, bin_size_s=BIN_SECONDS,
                      bin_timestamps=timestamps[:prefix_stop],
                      is_timestamp_bin_start=False),
            dtype=np.float64,
        )
        _require(official_counts.shape == (prefix_stop, unit_ids.size)
                 and np.isfinite(official_counts).all() and np.all(official_counts >= 0.0),
                 "official M1 bin_units output contract drift")
        counts = np.ascontiguousarray(official_counts[rate_indices])
        rates = np.ascontiguousarray(counts / BIN_SECONDS)
        _require(counts.shape[0] == rate_trial_ids.size and rates.shape == counts.shape,
                 "support count/trial alignment drift")

    return AlignedSupport(
        path=resolved,
        unit_ids=np.ascontiguousarray(unit_ids),
        unit_id_sha256=_unit_id_sha256(unit_ids),
        timestamps=np.ascontiguousarray(timestamps),
        emg=np.ascontiguousarray(emg),
        emg_trial_ids=np.ascontiguousarray(emg_trial_ids),
        emg_global_time_index=np.ascontiguousarray(emg_indices),
        counts=counts,
        rates=rates,
        rate_trial_ids=np.ascontiguousarray(rate_trial_ids),
        rate_global_time_index=np.ascontiguousarray(rate_indices),
        channel_names=channel_names,
        emg_trial_stop=emg_stop,
        neural_trial_stop=int(neural_trial_stop),
    )


@dataclass(frozen=True)
class FittedSourceCarriers:
    """Fold-local rSyn3 quantities, frozen before target M10 projection."""

    basis: object
    normalizer_mean: np.ndarray
    normalizer_scale: np.ndarray
    source_carriers: Mapping[str, np.ndarray]
    raw_source_carriers: Mapping[str, np.ndarray]
    source_unit_ids: Mapping[str, np.ndarray]
    source_unit_id_sha256: Mapping[str, str]
    channel_names: tuple[str, ...]
    raw_support_meta: Mapping[str, Mapping[str, object]]


def fit_source_carriers(sources: Sequence[str],
                        path_resolver: Callable[[str], Path]) -> FittedSourceCarriers:
    """Fit NNMF/normalizer only from three declared source sessions.

    Source NNMF sees rectified EMG through trial 309.  Per-unit encoding and
    carrier normalization use only each source M10.  The returned basis and
    normalizer are the only quantities ``project_target_carrier`` may use.
    """
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3

    names = tuple(str(name) for name in sources)
    _require(len(names) == 3 and len(set(names)) == len(names),
             "a LOSO M1 fold requires exactly three unique sources")
    support = {
        name: load_aligned_support(path_resolver(name), emg_trial_stop=310,
                                   neural_trial_stop=SUPPORT_TRIALS)
        for name in names
    }
    channel_layouts = {value.channel_names for value in support.values()}
    _require(len(channel_layouts) == 1, "source EMG channel layouts differ")
    _require(all(value.unit_ids.size == 64 for value in support.values()),
             "source unit count must match the M1 64-unit decoder face")
    basis = syn3.fit_source_nmf(np.concatenate([_rectify(support[name].emg) for name in names], axis=0))
    raw: dict[str, np.ndarray] = {}
    for name in names:
        value = support[name]
        scores = syn3.project_basis(_rectify(value.emg[value.emg_trial_ids < SUPPORT_TRIALS]), basis)
        _require(scores.shape[0] == value.rates.shape[0],
                 f"{name}: M10 EMG/rate bin count mismatch")
        weights, intercepts = syn3.fit_all_units(scores, value.rates)
        raw[name] = np.ascontiguousarray(syn3.carrier_from_encoding(weights, intercepts), dtype=np.float64)
        _require(raw[name].shape == (value.unit_ids.size, CARRIER_DIM) and np.isfinite(raw[name]).all(),
                 f"{name}: raw carrier contract drift")
    mean, scale = syn3.source_normalizer(list(raw.values()))
    normalized = {
        name: np.ascontiguousarray(syn3.normalize_carriers(raw[name], mean, scale), dtype=np.float32)
        for name in names
    }
    for name, carrier in normalized.items():
        _require(carrier.shape == raw[name].shape and np.isfinite(carrier).all(),
                 f"{name}: normalized carrier is invalid")
    return FittedSourceCarriers(
        basis=basis,
        normalizer_mean=np.ascontiguousarray(mean, dtype=np.float64),
        normalizer_scale=np.ascontiguousarray(scale, dtype=np.float64),
        source_carriers=normalized,
        raw_source_carriers=raw,
        source_unit_ids={name: support[name].unit_ids for name in names},
        source_unit_id_sha256={name: support[name].unit_id_sha256 for name in names},
        channel_names=next(iter(channel_layouts)),
        raw_support_meta={name: support[name].metadata() for name in names},
    )


def project_target_carrier(target_path: Path, frozen: FittedSourceCarriers) -> tuple[np.ndarray, dict[str, object]]:
    """Project target M10 only through frozen source basis and normalizer."""
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3

    target = load_aligned_support(target_path, emg_trial_stop=SUPPORT_TRIALS,
                                  neural_trial_stop=SUPPORT_TRIALS)
    _require(target.unit_ids.size == 64, "target unit count must match the M1 64-unit decoder face")
    _require(target.channel_names == frozen.channel_names,
             "target EMG channel layout differs from frozen source basis")
    scores = syn3.project_basis(_rectify(target.emg), frozen.basis)
    _require(scores.shape[0] == target.rates.shape[0], "target M10 EMG/rate bin count mismatch")
    weights, intercepts = syn3.fit_all_units(scores, target.rates)
    raw = np.ascontiguousarray(syn3.carrier_from_encoding(weights, intercepts), dtype=np.float64)
    _require(raw.shape == (target.unit_ids.size, CARRIER_DIM) and np.isfinite(raw).all(),
             "target raw carrier contract drift")
    carrier = np.ascontiguousarray(
        syn3.normalize_carriers(raw, frozen.normalizer_mean, frozen.normalizer_scale), dtype=np.float32)
    _require(carrier.shape == raw.shape and np.isfinite(carrier).all(),
             "target normalized carrier is invalid")
    return carrier, {**target.metadata(), "raw_carrier_sha256": _array_sha256(raw),
                     "carrier_sha256": _array_sha256(carrier),
                     "fit": "target M10 only; frozen source NNMF and normalizer"}


def causal_carrier_projection(neural_counts: np.ndarray, normalized_carrier: np.ndarray,
                               *, range_start: np.ndarray | None = None,
                               windows: Sequence[int] = CAUSAL_WINDOWS) -> np.ndarray:
    """Return a 24-D causal, normalized-carrier conditional projection.

    For each timepoint and each width in ``windows``, this computes the mean of
    that width's current-and-past count rows, with zero padding at each range
    start, then calculates ``mean_counts @ normalized_carrier / 64``.  A true
    value in ``range_start`` begins an independent left-zero-padded range.  No
    value at a future timepoint is read.  Unit rows are never permuted.

    The result is a normalized-carrier conditional projection, not an exact
    EMG decoder.
    """
    counts = np.asarray(neural_counts, dtype=np.float64)
    carrier = np.asarray(normalized_carrier, dtype=np.float64)
    widths = tuple(int(width) for width in windows)
    _require(counts.ndim == 2 and counts.shape[0] >= 1 and counts.shape[1] == 64
             and np.isfinite(counts).all() and np.all(counts >= 0.0),
             "neural_counts must be finite nonnegative [time,64]")
    _require(carrier.shape == (64, CARRIER_DIM) and np.isfinite(carrier).all(),
             "normalized_carrier must be finite [64,4] in matching unit order")
    _require(widths == CAUSAL_WINDOWS, "projection windows must be (1,5,10,25,50,100)")
    if range_start is None:
        starts = np.zeros(counts.shape[0], dtype=bool)
        starts[0] = True
    else:
        starts = np.asarray(range_start, dtype=bool)
        _require(starts.shape == (counts.shape[0],) and bool(starts[0]),
                 "range_start must begin at timepoint zero")

    output = np.empty((counts.shape[0], len(widths) * CARRIER_DIM), dtype=np.float32)
    segment_left = 0
    for segment_right in np.r_[np.flatnonzero(starts[1:]) + 1, counts.shape[0]]:
        segment = counts[segment_left:segment_right]
        prefix = np.vstack((np.zeros((1, counts.shape[1]), dtype=np.float64),
                            np.cumsum(segment, axis=0, dtype=np.float64)))
        local_time = np.arange(segment.shape[0])
        pieces = []
        for width in widths:
            left = np.maximum(0, local_time + 1 - width)
            sums = prefix[local_time + 1] - prefix[left]
            # Divide by the full width: omitted history is the required left-zero pad.
            pieces.append((sums / float(width)) @ carrier / 64.0)
        output[segment_left:segment_right] = np.concatenate(pieces, axis=1)
        segment_left = int(segment_right)
    _require(np.isfinite(output).all() and output.shape[1] == 24,
             "causal carrier projection is invalid")
    return output
