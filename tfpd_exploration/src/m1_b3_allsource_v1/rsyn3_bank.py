"""All-source EMG-rSyn3 M10 carriers. Query values are unread.

Fits a rank-3 NNMF dictionary on the four held-in calibration sessions after
the frozen ReLU projection, then encodes chronological M10 support into the
4-d unit carrier (3 synergy weights + intercept). Later-day public calib
files may be encoded with the frozen dictionary; they never enter the fit.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from pynwb import NWBHDF5IO

from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import syn3 as rsyn3
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as parent_data
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import plan as syn3_plan
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3 as parent_syn3

from . import plan


class RSyn3BankError(RuntimeError):
    """Fail closed for all-source rSyn3 carrier construction."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RSyn3BankError(message)


SUPPORT_TRIALS = int(syn3_plan.SUPPORT_TRIALS)
SOURCE_CALIB_DIR = "sub-MonkeyL-held-in-calib"
LATER_CALIB_DIR = "sub-MonkeyL-held-out-calib"
_READ_FORBIDDEN = ("minival", "formal", "evalai", "test")


def _session_name(path: Path) -> str:
    try:
        return f"ses-{path.name.split('_ses-')[1].split('_behavior')[0]}"
    except IndexError as error:
        raise RSyn3BankError(f"cannot parse session from {path.name}") from error


def _onset_stop(row: Any) -> tuple[float, float]:
    onset = float(row["move_onset_time"])
    stop = float(row["stop_time"])
    if not math.isfinite(onset):
        onset = float(row["start_time"])
    _require(math.isfinite(onset) and math.isfinite(stop) and stop > onset, "trial times")
    return onset, stop


def _encode_record(
    record: parent_data.SessionBins,
    basis: parent_syn3.SourceBasis,
    *,
    budget: int = SUPPORT_TRIALS,
    selected_trial_ids: np.ndarray | None = None,
) -> np.ndarray:
    if selected_trial_ids is None:
        selected = np.arange(int(budget), dtype=np.int64)
    else:
        selected = np.asarray(selected_trial_ids, dtype=np.int64).reshape(-1)
        _require(selected.size >= 1 and selected.ndim == 1, "selected trial ids")
        _require(int(np.min(selected)) >= 0, "negative selected trial")
        unique = np.unique(selected)
        _require(unique.size == selected.size, "duplicate selected trials")
        selected = unique
    emg_mask = np.isin(record.emg_trial_ids, selected)
    rate_mask = np.isin(record.rate_trial_ids, selected)
    emg_b = record.emg[emg_mask]
    rates_b = record.rates[rate_mask]
    ids_b = record.rate_trial_ids[rate_mask]
    _require(emg_b.shape[0] == rates_b.shape[0] > 0, "EMG/rate bin mismatch after selection")
    present = np.unique(ids_b)
    _require(set(present.tolist()) == set(selected.tolist()), "selected trials missing bins")
    rsyn3.require_support_bins(ids_b, budget=int(selected.max()) + 1)
    scores = rsyn3.project_basis(emg_b, basis)
    weights, intercepts = rsyn3.fit_all_units(scores, rates_b)
    carrier = rsyn3.carrier_from_encoding(weights, intercepts)
    _require(carrier.shape[1] == 4, f"rSyn3 carrier last dim {carrier.shape}")
    return np.asarray(carrier, dtype=np.float64)


def load_source_session(path: Path) -> parent_data.SessionBins:
    """Full-session EMG, M10 neural. Held-in calibration only."""
    record = parent_data.load_support_bins(
        path, emg_trial_stop=None, neural_trial_stop=SUPPORT_TRIALS,
    )
    _require(int(record.emg_trial_ids.max()) >= SUPPORT_TRIALS, "source EMG is not full-session")
    _require(int(record.rate_trial_ids.max()) == SUPPORT_TRIALS - 1, "source neural left support")
    return record


def load_public_calib_support(path: Path, *, support_trials: int = SUPPORT_TRIALS) -> parent_data.SessionBins:
    """EMG+neural from a public calibration NWB, including later-day 10-trial files."""
    resolved = Path(path).resolve()
    text = str(resolved).lower()
    if any(token in text for token in _READ_FORBIDDEN):
        raise RSyn3BankError(f"forbidden calib path token: {resolved}")
    _require(
        SOURCE_CALIB_DIR.lower() in text or LATER_CALIB_DIR.lower() in text,
        f"public calib must be a FALCON calibration NWB, got {resolved}",
    )
    _require(resolved.is_file(), f"missing calib file {resolved}")
    _require(int(support_trials) >= 1, "support_trials")
    digest = parent_data.file_sha256(resolved)
    with NWBHDF5IO(str(resolved), "r", load_namespaces=True) as io:
        nwb = io.read()
        trials = nwb.trials.to_dataframe()
        raw = nwb.acquisition["preprocessed_emg"]
        channel_names = tuple(raw.time_series.keys())
        _require(len(channel_names) > 0, "no EMG channels")
        first = raw.get_timeseries(channel_names[0])
        timestamps = np.asarray(first.timestamps[:], dtype=np.float64)
        n_trials = int(len(trials))
        _require(n_trials >= int(support_trials), f"{resolved.name} has fewer than {support_trials} trials")
        emg_stop = int(support_trials)
        neural_stop = int(support_trials)
        diffs = np.diff(timestamps)
        median_dt = float(np.median(diffs)) if len(diffs) else syn3_plan.BIN_SECONDS
        alignment_ok = bool(abs(median_dt - syn3_plan.BIN_SECONDS) < 1.0e-3)
        support_start, _ = _onset_stop(trials.iloc[0])
        _, support_stop = _onset_stop(trials.iloc[neural_stop - 1])
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
            for emg_row, bin_time in zip(segment, bin_times):
                emg_rows.append(emg_row)
                trial_ids.append(trial_index)
                if trial_index < neural_stop:
                    bin_end = bin_time + syn3_plan.BIN_SECONDS
                    counts = np.array(
                        [np.count_nonzero((times >= bin_time) & (times < bin_end))
                         for times in spike_times],
                        dtype=np.float64,
                    )
                    rate_rows.append(counts / syn3_plan.BIN_SECONDS)
        emg = np.vstack(emg_rows)
        rates = np.vstack(rate_rows)
        ids = np.asarray(trial_ids, dtype=np.int64)
        rate_ids = ids[ids < neural_stop]
        finite = emg[np.isfinite(emg)]
        signal_view = {
            "session": _session_name(resolved),
            "n_trials": n_trials,
            "emg_trial_range": [0, emg_stop],
            "neural_trial_range": [0, neural_stop],
            "query_neural_or_emg_values_read": False,
            "median_bin_seconds": median_dt,
            "recorded_time_alignment_ok": alignment_ok,
            "finite_count": int(np.isfinite(emg).sum()),
            "minimum": float(np.min(finite)) if finite.size else None,
            "maximum": float(np.max(finite)) if finite.size else None,
            "n_units": int(rates.shape[1]),
            "n_emg_channels": int(emg.shape[1]),
            "role": "public_calib_support",
        }
        _require(not bool(signal_view["query_neural_or_emg_values_read"]), "query leak")
        return parent_data.SessionBins(
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


def build_all_source_bank(source_paths: Mapping[str, Path]) -> dict[str, Any]:
    """Fit NMF on four source sessions; encode each session's M10 carrier."""
    names = tuple(plan.SOURCE_SESSION_NAMES)
    _require(tuple(source_paths) == names, f"all-source rSyn3 needs {names}, got {tuple(source_paths)}")
    records = OrderedDict(
        (name, load_source_session(Path(source_paths[name]))) for name in names
    )
    source_emg = np.concatenate([record.emg for record in records.values()], axis=0)
    basis = rsyn3.fit_source_nmf(source_emg)
    _require(basis.kind == "nnmf", "all-source rSyn3 dictionary must be NNMF")
    raw = OrderedDict((name, _encode_record(record, basis)) for name, record in records.items())
    norm_mean, norm_scale = rsyn3.source_normalizer(list(raw.values()))
    normalized = OrderedDict(
        (name, np.asarray(rsyn3.normalize_carriers(carrier, norm_mean, norm_scale), dtype=np.float32))
        for name, carrier in raw.items()
    )
    for name, carrier in normalized.items():
        n_units = int(records[name].rates.shape[1])
        _require(carrier.shape == (n_units, 4), f"{name} carrier shape {carrier.shape}")
    return {
        "schema": "m1_all_source_rsyn3_bank_v1",
        "source_sessions": list(names),
        "support_trials": SUPPORT_TRIALS,
        "query_values_read": False,
        "nmf_fit_sessions": list(names),
        "later_day_in_fit": False,
        "basis_kind": basis.kind,
        "basis_scale": np.asarray(basis.scale, dtype=np.float64),
        "basis_dictionary": np.asarray(basis.dictionary, dtype=np.float64),
        "basis_order": list(basis.order),
        "normalizer_mean": np.asarray(norm_mean, dtype=np.float64),
        "normalizer_scale": np.asarray(norm_scale, dtype=np.float64),
        "raw": {name: np.asarray(value, dtype=np.float64) for name, value in raw.items()},
        "normalized": {name: value for name, value in normalized.items()},
        "source_files": {
            name: {"path": str(Path(source_paths[name]).resolve()), "sha256": records[name].path_sha256}
            for name in names
        },
    }


def _basis_from_bank(bank: Mapping[str, Any]) -> parent_syn3.SourceBasis:
    return parent_syn3.SourceBasis(
        kind=str(bank["basis_kind"]),
        scale=np.asarray(bank["basis_scale"], dtype=np.float64),
        dictionary=np.asarray(bank["basis_dictionary"], dtype=np.float64),
        activations=np.zeros((1, syn3_plan.RANK), dtype=np.float64),
        order=tuple(int(item) for item in bank["basis_order"]),
        reconstruction_digest="unused",
        library={},
        extra={},
    )


def encode_public_session(
    path: Path, bank: Mapping[str, Any], *, support_trials: int = SUPPORT_TRIALS,
) -> np.ndarray:
    """Encode one public calib session with the frozen all-source dictionary."""
    basis = _basis_from_bank(bank)
    record = load_public_calib_support(path, support_trials=int(support_trials))
    raw = _encode_record(record, basis, budget=int(support_trials))
    syn = rsyn3.normalize_carriers(
        raw,
        np.asarray(bank["normalizer_mean"], dtype=np.float64),
        np.asarray(bank["normalizer_scale"], dtype=np.float64),
    )
    return np.asarray(syn, dtype=np.float32)


def encode_record_selected(
    record: parent_data.SessionBins,
    bank: Mapping[str, Any],
    selected_trial_ids: np.ndarray,
    *,
    pool_trials: int = SUPPORT_TRIALS,
) -> np.ndarray:
    """Fit unit ridge on selected calib trials from an already-loaded record."""
    selected = np.asarray(selected_trial_ids, dtype=np.int64).reshape(-1)
    _require(selected.size >= 1, "empty selected trials")
    _require(int(selected.min()) >= 0 and int(selected.max()) < int(pool_trials), "selection outside pool")
    basis = _basis_from_bank(bank)
    raw = _encode_record(record, basis, selected_trial_ids=selected)
    syn = rsyn3.normalize_carriers(
        raw,
        np.asarray(bank["normalizer_mean"], dtype=np.float64),
        np.asarray(bank["normalizer_scale"], dtype=np.float64),
    )
    return np.asarray(syn, dtype=np.float32)


def encode_public_session_selected(
    path: Path,
    bank: Mapping[str, Any],
    selected_trial_ids: np.ndarray,
    *,
    pool_trials: int = SUPPORT_TRIALS,
) -> np.ndarray:
    """Fit unit ridge on selected calib trials; NMF dictionary stays frozen."""
    record = load_public_calib_support(path, support_trials=int(pool_trials))
    return encode_record_selected(
        record, bank, selected_trial_ids, pool_trials=int(pool_trials),
    )


def manifest_payload(bank: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-safe bank receipt. Full arrays stay in hydra_run sidecar via numpy lists."""
    carriers = {
        name: np.asarray(value, dtype=np.float32).tolist()
        for name, value in bank["normalized"].items()
    }
    body = {
        "schema": bank["schema"],
        "source_sessions": list(bank["source_sessions"]),
        "support_trials": int(bank["support_trials"]),
        "query_values_read": False,
        "nmf_fit_sessions": list(bank["nmf_fit_sessions"]),
        "later_day_in_fit": False,
        "basis_kind": bank["basis_kind"],
        "basis_scale": np.asarray(bank["basis_scale"], dtype=np.float64).tolist(),
        "basis_dictionary": np.asarray(bank["basis_dictionary"], dtype=np.float64).tolist(),
        "basis_order": [int(item) for item in bank["basis_order"]],
        "normalizer_mean": np.asarray(bank["normalizer_mean"], dtype=np.float64).tolist(),
        "normalizer_scale": np.asarray(bank["normalizer_scale"], dtype=np.float64).tolist(),
        "normalized_carriers": carriers,
        "source_files": bank["source_files"],
        "carrier_sha256": {
            name: hashlib.sha256(
                np.ascontiguousarray(np.asarray(value, dtype=np.float32)).tobytes()
            ).hexdigest()
            for name, value in bank["normalized"].items()
        },
    }
    body["bank_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return body


def bank_from_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload.get("schema") == "m1_all_source_rsyn3_bank_v1", "rSyn3 bank schema drift")
    _require(payload.get("later_day_in_fit") is False, "later-day entered the NMF fit")
    normalized = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in payload["normalized_carriers"].items()
    }
    return {
        "schema": payload["schema"],
        "source_sessions": list(payload["source_sessions"]),
        "support_trials": int(payload["support_trials"]),
        "query_values_read": False,
        "nmf_fit_sessions": list(payload["nmf_fit_sessions"]),
        "later_day_in_fit": False,
        "basis_kind": payload["basis_kind"],
        "basis_scale": np.asarray(payload["basis_scale"], dtype=np.float64),
        "basis_dictionary": np.asarray(payload["basis_dictionary"], dtype=np.float64),
        "basis_order": list(payload["basis_order"]),
        "normalizer_mean": np.asarray(payload["normalizer_mean"], dtype=np.float64),
        "normalizer_scale": np.asarray(payload["normalizer_scale"], dtype=np.float64),
        "normalized": normalized,
        "source_files": payload["source_files"],
    }
