"""RT AFC4 data loading and fail-closed reach segmentation for DANDI 000688.

The RT files are deliberately *not* treated as ordinary FALCON trial files.
Each NWB trial may contain up to four reaches, whose starts are stored in
``trials.go_cue_time_array``.  A K4/AFC4 velocity block is admissible only
when both its five neural 20-ms bins and its 40-ms-lagged five behaviour bins
belong to one accepted reach segment.  There is no fallback to a whole-trial
block when go cues are missing or malformed.

The loader returns the ordinary ``FalconDataset`` arrays plus:

``k4_segment_id``
    An integer reach ID per raw 20-ms bin; ``-1`` marks an ineligible bin.
    IDs are unique across the session, so equality proves both same-reach and
    same-trial membership.
``rt_segment_audit``
    A JSON-serialisable, per-trial accounting of the fixed M24 trial budget,
    complete-cue trials, accepted reach segments, and all exclusions.

Cursor velocity remains in the NWB-declared unit (currently ``cm/s``).  This
module does not standardise, rescale, or impute it.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pynwb


BIN_SIZE_S = 0.02  # 20 ms, matching the frozen K4 raw-bin contract.
SEGMENT_INVALID = -1
RT_EXPECTED_SESSION_COUNT = 15


def _ceil_bin(time_s: float, *, bin_size_s: float = BIN_SIZE_S) -> int:
    """First 20-ms bin whose *start* is at or after ``time_s``.

    Requiring whole bins inside an event interval is intentionally
    conservative: a bin that starts before a go cue, even if it overlaps the
    cue by a microsecond, is not labelled as belonging to that reach.
    """
    return int(np.ceil(float(time_s) / bin_size_s - 1.0e-10))


def _floor_bin(time_s: float, *, bin_size_s: float = BIN_SIZE_S) -> int:
    """Exclusive end for whole 20-ms bins ending at or before ``time_s``."""
    return int(np.floor(float(time_s) / bin_size_s + 1.0e-10))


def _json_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _as_go_cue_matrix(values: Any, *, n_trials: int) -> np.ndarray:
    """Read fixed-width or ragged NWB go cues without inventing missing cues."""
    raw = np.asarray(values)
    if raw.ndim == 2:
        matrix = np.asarray(raw, dtype=np.float64)
    elif raw.ndim == 1 and raw.dtype == object:
        rows = [np.asarray(item, dtype=np.float64).reshape(-1) for item in raw]
        if len(rows) != n_trials:
            raise ValueError("go_cue_time_array row count does not match trials")
        width = max((row.size for row in rows), default=0)
        matrix = np.full((n_trials, width), np.nan, dtype=np.float64)
        for index, row in enumerate(rows):
            matrix[index, : row.size] = row
    else:
        raise ValueError(
            "go_cue_time_array must be a [trials,reaches] numeric matrix or a ragged object vector"
        )
    if matrix.shape[0] != n_trials or matrix.shape[1] < 1:
        raise ValueError(
            f"Invalid go_cue_time_array shape {matrix.shape}; expected [{n_trials}, >=1]"
        )
    return matrix


def build_rt_reach_segments(
    *,
    n_bins: int,
    trial_start_times: np.ndarray,
    trial_stop_times: np.ndarray,
    go_cue_time_array: np.ndarray,
    num_targets: np.ndarray,
    velocity_bin_valid: np.ndarray,
    bin_size_s: float = BIN_SIZE_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Create conservative, event-qualified reach IDs from real RT metadata.

    A trial is ``complete_cue`` only if its declared number of cues is finite,
    all required cue values are finite and strictly increasing, no undeclared
    cue slot is finite, and every cue lies inside the raw trial interval.
    Accepted segments are ``[cue_j, cue_{j+1})`` and the final
    ``[cue_last, stop_time)``.  A segment is rejected, rather than clipped,
    if conservative whole-bin alignment leaves it empty or any velocity bin in
    it is absent.  This makes missing metadata and timing uncertainty visible
    in the receipt instead of silently crossing a reach boundary.
    """
    starts = np.asarray(trial_start_times, dtype=np.float64).reshape(-1)
    stops = np.asarray(trial_stop_times, dtype=np.float64).reshape(-1)
    cues = np.asarray(go_cue_time_array, dtype=np.float64)
    targets = np.asarray(num_targets).reshape(-1)
    velocity_valid = np.asarray(velocity_bin_valid, dtype=bool).reshape(-1)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if starts.shape != stops.shape or starts.shape != targets.shape:
        raise ValueError("RT trial start/stop/num_targets lengths must match")
    if cues.ndim != 2 or cues.shape[0] != starts.size:
        raise ValueError("RT go_cue_time_array must have one row per trial")
    if velocity_valid.shape != (n_bins,):
        raise ValueError("velocity_bin_valid must have one value per raw 20-ms bin")
    if not np.isfinite(starts).all() or not np.isfinite(stops).all():
        raise ValueError("RT trial start/stop times must be finite")
    if not np.all(stops > starts):
        raise ValueError("RT trial stop_time must be strictly greater than start_time")

    trial_change = np.zeros(n_bins, dtype=bool)
    trial_start_bins: list[int] = []
    for index, start in enumerate(starts):
        raw_bin = _ceil_bin(start, bin_size_s=bin_size_s)
        if not 0 <= raw_bin < n_bins:
            raise ValueError(
                f"RT trial {index} start {start:.9f}s maps outside velocity range ({n_bins} bins)"
            )
        if trial_change[raw_bin]:
            raise ValueError(
                f"RT trials share conservative 20-ms start bin {raw_bin}; refusing ambiguous chronology"
            )
        trial_change[raw_bin] = True
        trial_start_bins.append(raw_bin)

    segment_id = np.full(n_bins, SEGMENT_INVALID, dtype=np.int64)
    eval_mask = np.zeros(n_bins, dtype=bool)
    trial_exclusions: Counter[str] = Counter()
    segment_exclusions: Counter[str] = Counter()
    trial_records: list[dict[str, Any]] = []
    declared_segments = 0
    accepted_segments = 0
    complete_cue_trials = 0
    trials_with_accepted_segments = 0
    next_segment_id = 0

    for trial_index, (start, stop, target_count) in enumerate(zip(starts, stops, targets)):
        record: dict[str, Any] = {
            "trial_index": int(trial_index),
            "trial_start_bin": int(trial_start_bins[trial_index]),
            "declared_num_targets": None,
            "complete_cue": False,
            "declared_segments": 0,
            "accepted_segments": 0,
            "excluded_segments": 0,
            "exclusion_reason": None,
            "segment_exclusion_reasons": {},
        }
        if not np.isfinite(target_count):
            reason = "nonfinite_num_targets"
            trial_exclusions[reason] += 1
            record["exclusion_reason"] = reason
            trial_records.append(record)
            continue
        target_count_int = int(target_count)
        record["declared_num_targets"] = target_count_int
        if target_count_int < 1 or target_count_int > cues.shape[1]:
            reason = "num_targets_out_of_go_cue_width"
            trial_exclusions[reason] += 1
            record["exclusion_reason"] = reason
            trial_records.append(record)
            continue
        trial_cues = cues[trial_index, :target_count_int]
        unused_cues = cues[trial_index, target_count_int:]
        if not np.isfinite(trial_cues).all():
            reason = "nonfinite_required_go_cue"
            trial_exclusions[reason] += 1
            record["exclusion_reason"] = reason
            trial_records.append(record)
            continue
        if np.isfinite(unused_cues).any():
            reason = "finite_undeclared_go_cue"
            trial_exclusions[reason] += 1
            record["exclusion_reason"] = reason
            trial_records.append(record)
            continue
        if not np.all(np.diff(trial_cues) > 0.0):
            reason = "nonmonotonic_go_cues"
            trial_exclusions[reason] += 1
            record["exclusion_reason"] = reason
            trial_records.append(record)
            continue
        if trial_cues[0] < start or trial_cues[-1] >= stop:
            reason = "go_cue_outside_trial"
            trial_exclusions[reason] += 1
            record["exclusion_reason"] = reason
            trial_records.append(record)
            continue

        complete_cue_trials += 1
        record["complete_cue"] = True
        record["declared_segments"] = target_count_int
        declared_segments += target_count_int
        segment_ends = np.concatenate([trial_cues[1:], np.asarray([stop])])
        per_trial_segment_exclusions: Counter[str] = Counter()
        for reach_index, (segment_start, segment_end) in enumerate(zip(trial_cues, segment_ends)):
            left = _ceil_bin(segment_start, bin_size_s=bin_size_s)
            right = _floor_bin(segment_end, bin_size_s=bin_size_s)
            if left < 0 or right > n_bins:
                reason = "segment_outside_velocity_range"
                segment_exclusions[reason] += 1
                per_trial_segment_exclusions[reason] += 1
                record["excluded_segments"] += 1
                continue
            if right <= left:
                reason = "segment_empty_after_whole_bin_alignment"
                segment_exclusions[reason] += 1
                per_trial_segment_exclusions[reason] += 1
                record["excluded_segments"] += 1
                continue
            if not velocity_valid[left:right].all():
                reason = "segment_has_missing_velocity_bin"
                segment_exclusions[reason] += 1
                per_trial_segment_exclusions[reason] += 1
                record["excluded_segments"] += 1
                continue
            if np.any(segment_id[left:right] != SEGMENT_INVALID):
                raise RuntimeError(
                    f"RT segment overlap after conservative alignment in trial {trial_index}, reach {reach_index}"
                )
            segment_id[left:right] = next_segment_id
            eval_mask[left:right] = True
            next_segment_id += 1
            accepted_segments += 1
            record["accepted_segments"] += 1
        record["segment_exclusion_reasons"] = _json_counter(per_trial_segment_exclusions)
        if record["accepted_segments"]:
            trials_with_accepted_segments += 1
        elif target_count_int:
            # The cue metadata was valid, but no segment survived timing/data
            # qualification.  Keep this distinct from malformed cues.
            record["exclusion_reason"] = "no_accepted_reach_segment"
        trial_records.append(record)

    audit: dict[str, Any] = {
        "segment_policy": "go_cue_to_next_go_cue__last_go_cue_to_trial_stop__whole_20ms_bins_only",
        "bin_size_ms": int(round(bin_size_s * 1000.0)),
        "trials_total": int(starts.size),
        "complete_cue_trials": int(complete_cue_trials),
        "trials_with_accepted_segments": int(trials_with_accepted_segments),
        "excluded_trials": int(starts.size - complete_cue_trials),
        "trial_exclusion_reasons": _json_counter(trial_exclusions),
        "declared_reach_segments": int(declared_segments),
        "accepted_reach_segments": int(accepted_segments),
        "excluded_reach_segments": int(declared_segments - accepted_segments),
        "segment_exclusion_reasons": _json_counter(segment_exclusions),
        "valid_velocity_bins": int(velocity_valid.sum()),
        "event_qualified_bins": int(eval_mask.sum()),
        "trial_records": trial_records,
    }
    return trial_change, segment_id, eval_mask, audit


def summarize_rt_trial_budget(
    segment_audit: dict[str, Any], *, budget_trials: int
) -> dict[str, Any]:
    """Summarise the exact chronological support prefix without relabelling it.

    ``budget_trials=24`` always means trial indices ``[0,24)``.  It does not
    claim that 24 cue-complete trials, 96 reaches, or 24 valid labels survived;
    those quantities are separately counted here for each session receipt.
    """
    if budget_trials < 1:
        raise ValueError("budget_trials must be positive")
    records = list(segment_audit.get("trial_records", []))
    if len(records) < budget_trials:
        raise ValueError(
            f"RT session has {len(records)} trial records, below budget_trials={budget_trials}"
        )
    prefix = records[:budget_trials]
    trial_reasons: Counter[str] = Counter()
    segment_reasons: Counter[str] = Counter()
    complete = 0
    accepted_trial_count = 0
    declared_segments = 0
    accepted_segments = 0
    excluded_segments = 0
    for record in prefix:
        complete += int(bool(record["complete_cue"]))
        accepted_trial_count += int(int(record["accepted_segments"]) > 0)
        declared_segments += int(record["declared_segments"])
        accepted_segments += int(record["accepted_segments"])
        excluded_segments += int(record["excluded_segments"])
        if record.get("exclusion_reason"):
            trial_reasons[str(record["exclusion_reason"])] += 1
        for reason, value in dict(record.get("segment_exclusion_reasons", {})).items():
            segment_reasons[str(reason)] += int(value)
    return {
        "budget_trials": int(budget_trials),
        "trial_index_range": [0, int(budget_trials)],
        "complete_cue_trials_within_budget": int(complete),
        "trials_with_accepted_segments_within_budget": int(accepted_trial_count),
        "excluded_trials_within_budget": int(budget_trials - complete),
        "trial_exclusion_reasons_within_budget": _json_counter(trial_reasons),
        "declared_reach_segments_within_budget": int(declared_segments),
        "accepted_reach_segments_within_budget": int(accepted_segments),
        "excluded_reach_segments_within_budget": int(excluded_segments),
        "segment_exclusion_reasons_within_budget": _json_counter(segment_reasons),
    }


def _bin_cursor_velocity(
    data: np.ndarray, timestamps: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Average raw cursor velocity into 20-ms bins without scaling or imputation."""
    values = np.asarray(data, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"RT cursor_vel must be [time,2], got {values.shape}")
    if times.shape != (values.shape[0],):
        raise ValueError("RT cursor_vel timestamps must have one entry per velocity sample")
    if times.size < 1 or not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError("RT cursor_vel contains empty or non-finite timestamps/data")
    if times[0] < -1.0e-9 or np.any(np.diff(times) <= 0.0):
        raise ValueError("RT cursor_vel timestamps must be strictly increasing from a non-negative origin")

    bin_index = np.floor(times / BIN_SIZE_S + 1.0e-10).astype(np.int64)
    n_bins = int(bin_index.max()) + 1
    sums = np.zeros((n_bins, 2), dtype=np.float64)
    np.add.at(sums, bin_index, values)
    counts = np.bincount(bin_index, minlength=n_bins).astype(np.int64)
    valid = counts > 0
    binned = np.zeros((n_bins, 2), dtype=np.float64)
    binned[valid] = sums[valid] / counts[valid, None]
    deltas = np.diff(times)
    audit = {
        "source_sample_count": int(times.size),
        "source_timestamp_start_s": float(times[0]),
        "source_timestamp_end_s": float(times[-1]),
        "source_median_dt_s": float(np.median(deltas)) if deltas.size else None,
        "output_bin_count": int(n_bins),
        "output_valid_bin_count": int(valid.sum()),
        "output_missing_bin_count": int((~valid).sum()),
        "resampling": "mean_of_raw_samples_with_timestamp_in_each_20ms_bin",
        "missing_bin_policy": "excluded_from_event_segments_and_query_mask__numeric_zero_never_eligible",
    }
    return binned.astype(np.float32), valid, audit


def load_rt_session(nwb_path: str | Path) -> dict[str, Any]:
    """Load one sorted-SUA RT session at raw 20-ms resolution.

    The M24 budget remains chronological trial indices ``[0,24)`` even when
    some early trials have invalid cue metadata.  The accompanying audit makes
    the distinction between budgeted trials and usable movement blocks explicit.
    """
    nwb_path = Path(nwb_path)
    with pynwb.NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwb = io.read()
        try:
            vel_ts = nwb.processing["behavior"].data_interfaces["Velocity"].time_series[
                "cursor_vel"
            ]
        except KeyError as error:
            raise KeyError(f"RT NWB lacks behavior/Velocity/cursor_vel: {nwb_path}") from error
        if vel_ts.timestamps is None:
            raise ValueError("RT cursor_vel must carry explicit timestamps")
        vel_20ms, velocity_bin_valid, velocity_audit = _bin_cursor_velocity(
            np.asarray(vel_ts.data[:]), np.asarray(vel_ts.timestamps[:])
        )
        n_bins = int(vel_20ms.shape[0])

        if nwb.units is None or "spike_times" not in nwb.units.colnames:
            raise ValueError(f"RT NWB lacks sorted units.spike_times: {nwb_path}")
        spike_times_list = nwb.units["spike_times"][:]
        n_units = len(spike_times_list)
        if n_units < 2:
            raise ValueError("RT AFC4 row-shuffle control requires at least two sorted SUA units")
        neural = np.zeros((n_bins, n_units), dtype=np.float32)
        for channel, spike_times in enumerate(spike_times_list):
            spikes = np.asarray(spike_times, dtype=np.float64).reshape(-1)
            if not np.isfinite(spikes).all():
                raise ValueError(f"RT unit {channel} contains non-finite spike times")
            spike_bins = np.floor(spikes / BIN_SIZE_S + 1.0e-10).astype(np.int64)
            valid_spikes = (spike_bins >= 0) & (spike_bins < n_bins)
            np.add.at(neural[:, channel], spike_bins[valid_spikes], 1.0)

        if nwb.trials is None:
            raise ValueError(f"RT NWB lacks trials table: {nwb_path}")
        trials = nwb.trials
        required_columns = {"start_time", "stop_time", "go_cue_time_array", "num_targets"}
        missing_columns = sorted(required_columns.difference(trials.colnames))
        if missing_columns:
            raise ValueError(f"RT NWB trials table lacks required columns: {missing_columns}")
        starts = np.asarray(trials["start_time"][:], dtype=np.float64)
        stops = np.asarray(trials["stop_time"][:], dtype=np.float64)
        num_targets = np.asarray(trials["num_targets"][:])
        go_cues = _as_go_cue_matrix(trials["go_cue_time_array"][:], n_trials=starts.size)
        trial_change, segment_id, eval_mask, segment_audit = build_rt_reach_segments(
            n_bins=n_bins,
            trial_start_times=starts,
            trial_stop_times=stops,
            go_cue_time_array=go_cues,
            num_targets=num_targets,
            velocity_bin_valid=velocity_bin_valid,
        )

        unit = str(vel_ts.unit)
        conversion = float(vel_ts.conversion)
        offset = float(vel_ts.offset)

    stem = nwb_path.name.removesuffix("_behavior+ecephys.nwb")
    session_name = stem.removeprefix("sub-C_")
    return {
        "neural": neural,
        "covariates": vel_20ms,
        "trial_change": trial_change,
        "eval_mask": eval_mask,
        "k4_segment_id": segment_id,
        "session_name": session_name,
        "nwb_path": str(nwb_path),
        "rt_segment_audit": segment_audit,
        "rt_velocity_audit": {
            **velocity_audit,
            "nwb_unit": unit,
            "nwb_conversion": conversion,
            "nwb_offset": offset,
            "loader_output_unit": unit,
            "loader_standardization": "none",
        },
    }


def find_rt_sessions(data_dir: str | Path) -> list[Path]:
    """Find RT session NWBs, excluding CO files and any unrelated artifacts."""
    data_dir = Path(data_dir)
    return sorted(data_dir.glob("sub-C_ses-RT-*_behavior+ecephys.nwb"))


# Pre-declared T4G/RT protocol parameters.  The aliases are explicit so a
# receipt never mislabels a four-wide zero input as feature-free F0.
RT_PROTOCOL = {
    "signal_type": "sorted_SUA",
    "decode_target": "2D cursor velocity",
    "cursor_velocity_unit": "NWB-declared (currently cm/s), no loader scaling",
    "split": "development_clean_nested_outer_LOSO",
    "support_budget_trials": 24,
    "arms": [
        "zero4",
        "afc4_vel",
        "afc4_rs",
        "afc4_ls",
        "afc4_b4",
        "afc4_w4",
    ],
    "arm_aliases": {
        "zero4": "width-matched all-zero [N,4] side path",
        "afc4_vel": "k4 implementation: aligned movement velocity descriptor",
        "afc4_rs": "ks4 implementation: complete descriptor-row shuffle",
        "afc4_ls": "segment-preserving continuous velocity label-association null",
        "afc4_b4": "normalized [0,0,0,b] baseline-rate component ablation",
        "afc4_w4": "normalized [wx,wy,0,0] velocity-weight component ablation",
    },
    "bin_size_ms": 20,
    "expected_sessions": RT_EXPECTED_SESSION_COUNT,
    "decoder_training": (
        "FALCON-M2 checkpoint initialization followed by joint RT-source retraining; "
        "outer target receives no backpropagation"
    ),
}

RT_GATES = {
    "content": "afc4_vel minus afc4_b4 isolates velocity-conditioned carrier content",
    "system": "afc4_vel minus zero4 tests the complete carrier against a width-matched null",
    "specificity": (
        "afc4_rs and afc4_ls separately test row attachment and velocity-label association"
    ),
    "no_formal_heldout": "RT T4G is a development LOSO program; no official held-out endpoint is opened",
}
