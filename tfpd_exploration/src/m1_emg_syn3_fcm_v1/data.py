"""Allow-listed M1 held-in-calib loaders. Query/formal/minival paths fail closed."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from pynwb import NWBHDF5IO

from . import plan


class DataError(RuntimeError):
    """Fail closed for M1 source path/chronology drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataError(message)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def require_source_path(path: Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (repo_root() / resolved).resolve()
    else:
        resolved = resolved.resolve()
    text = str(resolved).lower()
    if any(token in text for token in plan.FORBIDDEN_PATH_TOKENS):
        raise DataError(f"forbidden source path token: {resolved}")
    if "sub-monkeyl-held-in-calib" not in text:
        raise DataError(f"source leaves held-in-calib: {resolved}")
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _onset_stop(row: Any) -> tuple[float, float]:
    onset = float(row["move_onset_time"])
    stop = float(row["stop_time"])
    if not math.isfinite(onset):
        onset = float(row["start_time"])
    _require(math.isfinite(onset) and math.isfinite(stop) and stop > onset, "trial times")
    return onset, stop


@dataclass(frozen=True)
class SessionBins:
    session: str
    path: Path
    path_sha256: str
    emg: np.ndarray
    emg_trial_ids: np.ndarray
    rates: np.ndarray
    rate_trial_ids: np.ndarray
    n_trials: int
    channel_names: tuple[str, ...]
    signal_view: dict[str, object]


def load_support_bins(
    path: Path, *, emg_trial_stop: int | None, neural_trial_stop: int,
) -> SessionBins:
    resolved = require_source_path(path)
    _require(resolved.is_file(), f"missing source file {resolved}")
    _require(neural_trial_stop >= 1, "neural trial stop")
    digest = file_sha256(resolved)
    with NWBHDF5IO(str(resolved), "r", load_namespaces=True) as io:
        nwb = io.read()
        trials = nwb.trials.to_dataframe()
        raw = nwb.acquisition["preprocessed_emg"]
        channel_names = tuple(raw.time_series.keys())
        _require(len(channel_names) > 0, "no EMG channels")
        first = raw.get_timeseries(channel_names[0])
        timestamps = np.asarray(first.timestamps[:], dtype=np.float64)
        n_trials = int(len(trials))
        _require(n_trials > plan.SUPPORT_TRIALS, "no post-support trials")
        emg_stop = int(n_trials if emg_trial_stop is None else emg_trial_stop)
        _require(1 <= emg_stop <= n_trials and neural_trial_stop <= n_trials, "stop past recording")
        _require(neural_trial_stop <= emg_stop, "neural stop exceeds EMG materialization")
        diffs = np.diff(timestamps)
        median_dt = float(np.median(diffs)) if len(diffs) else plan.BIN_SECONDS
        alignment_ok = bool(abs(median_dt - plan.BIN_SECONDS) < 1.0e-3)
        support_start, _ = _onset_stop(trials.iloc[0])
        _, support_stop = _onset_stop(trials.iloc[neural_trial_stop - 1])
        spike_times = [
            np.asarray(
                nwb.units.get_unit_spike_times(unit_index, in_interval=(support_start, support_stop)),
                dtype=np.float64,
            )
            for unit_index in range(len(nwb.units.id))
        ]
        emg_rows: list[np.ndarray] = []
        rate_rows: list[np.ndarray] = []
        trial_ids: list[int] = []
        for trial_index in range(emg_stop):
            onset, stop = _onset_stop(trials.iloc[trial_index])
            left = int(np.searchsorted(timestamps, onset, side="left"))
            right = int(np.searchsorted(timestamps, stop, side="right"))
            _require(right > left, f"{resolved.name} empty movement window at trial {trial_index}")
            segment = np.column_stack(
                [np.asarray(raw.get_timeseries(channel).data[left:right], dtype=np.float64)
                 for channel in channel_names]
            )
            _require(np.isfinite(segment).all() and segment.shape[0] > 0, "invalid EMG segment")
            bin_times = timestamps[left:right]
            for bin_index, (emg_row, bin_time) in enumerate(zip(segment, bin_times)):
                emg_rows.append(emg_row)
                trial_ids.append(trial_index)
                if trial_index < neural_trial_stop:
                    bin_end = bin_time + plan.BIN_SECONDS
                    counts = np.array(
                        [np.count_nonzero((times >= bin_time) & (times < bin_end))
                         for times in spike_times],
                        dtype=np.float64,
                    )
                    rate_rows.append(counts / plan.BIN_SECONDS)
        emg = np.vstack(emg_rows)
        rates = np.vstack(rate_rows) if rate_rows else np.zeros((0, len(spike_times)))
        ids = np.asarray(trial_ids, dtype=np.int64)
        rate_ids = ids[ids < neural_trial_stop]
        finite = emg[np.isfinite(emg)]
        signal_view = {
            "session": _session_name(resolved),
            "n_trials": n_trials,
            "emg_trial_range": [0, emg_stop],
            "neural_trial_range": [0, neural_trial_stop],
            "query_trials_structurally_available": n_trials > plan.SUPPORT_TRIALS,
            "query_neural_or_emg_values_read": False,
            "median_bin_seconds": median_dt,
            "alignment_lag_bins": plan.LAG_BINS,
            "recorded_time_alignment_ok": alignment_ok,
            "finite_count": int(np.isfinite(emg).sum()),
            "minimum": float(np.min(finite)) if finite.size else None,
            "maximum": float(np.max(finite)) if finite.size else None,
            "zero_fraction": float(np.mean(emg == 0.0)),
            "negative_fraction": float(np.mean(emg < 0.0)),
            "valid_bins": int(emg.shape[0]),
            "valid_seconds": float(emg.shape[0] * plan.BIN_SECONDS),
            "contributing_trials": int(len(np.unique(ids))),
            "n_units": int(rates.shape[1]),
            "n_emg_channels": int(emg.shape[1]),
            "channel_names": list(channel_names),
            "rectification": "none_stored_view",
            "dev": int(resolved.stat().st_dev),
            "inode": int(resolved.stat().st_ino),
        }
        _require(not bool(signal_view["query_neural_or_emg_values_read"]), "query leak")
        return SessionBins(
            session=_session_name(resolved),
            path=resolved,
            path_sha256=digest,
            emg=emg,
            emg_trial_ids=ids,
            rates=rates,
            rate_trial_ids=rate_ids,
            n_trials=n_trials,
            channel_names=channel_names,
            signal_view=signal_view,
        )


def _session_name(path: Path) -> str:
    try:
        return f"ses-{path.name.split('_ses-')[1].split('_behavior')[0]}"
    except IndexError as error:
        raise DataError(f"cannot parse session from {path.name}") from error


def allowlisted_paths() -> dict[str, Path]:
    root = repo_root()
    result: dict[str, Path] = {}
    for session, relative in plan.SOURCE_RELATIVE.items():
        path = require_source_path(root / relative)
        _require(path.is_file(), f"missing {relative}")
        digest = file_sha256(path)
        _require(digest == plan.SOURCE_FILE_SHA256[session], f"source sha drift {session}")
        result[session] = path
    return result
