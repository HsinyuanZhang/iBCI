"""Offline-only M1 AFC4 packaging helpers.

The training AFC4 loader deliberately rejects a file with exactly ten trials,
because a training/LOSO query contract expects a strict post-M10 query.  Public
held-out *calibration* NWBs are the one legal exception for packaging: they
contain exactly M10 and no query file.  This module is a separate, explicit
packaging path that accepts only those three exact held-out calibration files,
materializes their first ten EMG/neural support trials, and never opens a
future/evaluation file.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from pynwb import NWBHDF5IO

from falcon_challenge.config import FalconConfig, FalconTask
from src.data.falcon_datamodule import FalconDataModule, FalconDataset
from src.data.falcon_emg_afc4_features import (
    AFC4_DIM,
    AFC4_Q,
    AFC4_RIDGE_ALPHA,
    AFC4_SUPPORT_TRIALS,
    M1EMGSupport,
    SourceFrozenEMGAFC4Plan,
    _canonical_session,
)


_CALIB_DIRS = {
    "sub-MonkeyL-held-in-calib": True,
    "sub-MonkeyL-held-out-calib": False,
}
_FORBIDDEN = ("minival", "formal", "evalai", "test", "eval", "query")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_calibration_path(path: Path, *, source_for_pca: bool) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    parent = resolved.parent.name
    if parent not in _CALIB_DIRS:
        raise ValueError(
            "packaging may open only sub-MonkeyL-held-in-calib or "
            f"sub-MonkeyL-held-out-calib, got {resolved.parent}"
        )
    if bool(source_for_pca) != _CALIB_DIRS[parent]:
        raise ValueError(
            f"source_for_pca={source_for_pca} is incompatible with calibration directory {parent}"
        )
    text = str(resolved).lower()
    if any(token in text for token in _FORBIDDEN):
        raise ValueError(f"packaging path contains a forbidden future/query token: {resolved}")
    if "calib" not in resolved.name.lower() or "held-out-calib" not in resolved.name.lower() and not source_for_pca:
        raise ValueError(f"not an exact calibration NWB: {resolved}")
    return resolved


def _support_checksum(session: str, trials) -> str:
    """Receipt checksum without reading any post-support samples."""
    boundary = {
        "session": session,
        "support_trial_range": [0, AFC4_SUPPORT_TRIALS],
        "query_trial_range": [AFC4_SUPPORT_TRIALS, int(len(trials))],
        "query_values_read": False,
    }
    return hashlib.sha256(
        json.dumps(boundary, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_m1_emg_support_for_package(path: Path, *, source_for_pca: bool) -> M1EMGSupport:
    """Load only legal source rows or a public held-out M10 support prefix."""
    path = _require_calibration_path(path, source_for_pca=source_for_pca)
    session = _canonical_session(path)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        trials = nwb.trials.to_dataframe()
        if len(trials) < AFC4_SUPPORT_TRIALS:
            raise ValueError(f"{session} has fewer than M10 calibration trials")
        if not source_for_pca and len(trials) != AFC4_SUPPORT_TRIALS:
            raise ValueError(f"held-out packaging calibration must contain exactly M10, got {len(trials)}")
        raw = nwb.acquisition["preprocessed_emg"]
        channel_names = tuple(raw.time_series.keys())
        if not channel_names:
            raise ValueError(f"{session}: no preprocessed EMG channels")
        first = raw.get_timeseries(channel_names[0])
        timestamps = np.asarray(first.timestamps[:], dtype=np.float64)
        emg_trial_count = len(trials) if source_for_pca else AFC4_SUPPORT_TRIALS
        full_rows: list[np.ndarray] = []
        for index in range(emg_trial_count):
            onset = float(trials.iloc[index]["move_onset_time"])
            stop = float(trials.iloc[index]["stop_time"])
            if not math.isfinite(onset):
                onset = float(trials.iloc[index]["start_time"])
            left = int(np.searchsorted(timestamps, onset, side="left"))
            right = int(np.searchsorted(timestamps, stop, side="right"))
            if right <= left:
                raise ValueError(f"{session}: empty movement-window EMG for trial {index}")
            segment = np.column_stack(
                [
                    np.asarray(raw.get_timeseries(channel).data[left:right], dtype=np.float64)
                    for channel in channel_names
                ]
            )
            if not np.isfinite(segment).all():
                raise ValueError(f"{session}: invalid movement-window EMG for trial {index}")
            full_rows.append(segment.mean(axis=0))

        support_rates = np.zeros((AFC4_SUPPORT_TRIALS, len(nwb.units.id)), dtype=np.float64)
        support_start = float(trials.iloc[0]["move_onset_time"])
        if not math.isfinite(support_start):
            support_start = float(trials.iloc[0]["start_time"])
        support_stop = float(trials.iloc[AFC4_SUPPORT_TRIALS - 1]["stop_time"])
        support_spike_times = [
            np.asarray(
                nwb.units.get_unit_spike_times(
                    unit_index, in_interval=(support_start, support_stop)
                ),
                dtype=np.float64,
            )
            for unit_index in range(len(nwb.units.id))
        ]
        for index in range(AFC4_SUPPORT_TRIALS):
            onset = float(trials.iloc[index]["move_onset_time"])
            stop = float(trials.iloc[index]["stop_time"])
            if not math.isfinite(onset):
                onset = float(trials.iloc[index]["start_time"])
            duration = max(stop - onset, 1.0e-12)
            for unit_index, times in enumerate(support_spike_times):
                support_rates[index, unit_index] = np.count_nonzero(
                    (times >= onset) & (times < stop)
                ) / duration
    full = np.vstack(full_rows)
    return M1EMGSupport(
        session_name=session,
        path=path,
        full_trial_emg=full,
        support_emg=full[:AFC4_SUPPORT_TRIALS].copy(),
        support_rates=support_rates,
        n_trials=int(len(trials)),
        query_checksum=_support_checksum(session, trials),
        source_for_pca=bool(source_for_pca),
        emg_trials_materialized=int(emg_trial_count),
    )


class M1AFC4PackagePlan:
    """Shared source basis/normalizer plus exact M10 target descriptors."""

    def __init__(self, source_paths: Mapping[str, Path], *, shuffle_seed: int) -> None:
        if not source_paths:
            raise ValueError("packaging AFC4 plan needs explicit source paths")
        expected = {"ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928"}
        if set(source_paths) != expected:
            raise ValueError(
                "packaging AFC4 plan requires all four held-in source sessions, "
                f"got {tuple(sorted(source_paths))}"
            )
        # Construct the audited source plan first.  It verifies the four source
        # files through the training module's strict held-in path guard.
        self._plan = SourceFrozenEMGAFC4Plan(source_paths, shuffle_seed=int(shuffle_seed))
        self.source_session_names = tuple(sorted(source_paths))

    @property
    def basis(self):
        return self._plan.basis

    @property
    def mean(self):
        return self._plan.mean

    @property
    def std(self):
        return self._plan.std

    @property
    def support(self):
        return self._plan.support

    def add_target(self, path: Path) -> str:
        record = load_m1_emg_support_for_package(path, source_for_pca=False)
        if record.session_name in self._plan.support:
            raise ValueError(f"duplicate AFC4 package session {record.session_name}")
        self._plan.support[record.session_name] = record
        self._plan.target_fit_calls[record.session_name] = 0
        return record.session_name

    def normalized(self, session_name: str, *, arm: str) -> np.ndarray:
        return self._plan.normalized(session_name, arm=arm)

    def receipt(self, *, arm: str) -> dict[str, object]:
        receipt = self._plan.receipt(arm=arm)
        receipt["packaging_scope"] = {
            "source_sessions": list(self.source_session_names),
            "target_sessions": [
                name for name, record in sorted(self.support.items()) if not record.source_for_pca
            ],
            "target_trial_range": [0, AFC4_SUPPORT_TRIALS],
            "target_future_query_values_read": False,
            "target_requires_exact_m10": True,
        }
        return receipt


def load_public_m1_calibration_dataset(paths: Sequence[Path]) -> FalconDataset:
    """Build identity tensors from calibration NWBs only.

    The helper uses the audited FALCON trialization code but receives an
    explicit seven-file list.  It never searches for ``eval``/``query`` files.
    """
    if not paths:
        raise ValueError("calibration dataset needs explicit paths")
    loader = object.__new__(FalconDataModule)
    task = FalconConfig(task=FalconTask.m1).task
    records: OrderedDict[str, dict[str, object]] = OrderedDict()
    covariates_mean = covariates_std = None
    for index, path in enumerate(paths):
        resolved = Path(path).resolve()
        # Both source and held-out files are public calibration inputs.  The
        # NWB reader is invoked here only after an explicit parent-directory
        # check, so a caller cannot pass an EvalAI hidden query file.
        source_for_pca = resolved.parent.name == "sub-MonkeyL-held-in-calib"
        _require_calibration_path(resolved, source_for_pca=source_for_pca)
        record = loader.prepare_session_data(
            resolved,
            task,
            standardize_covariates=False,
            covariates_mean=covariates_mean,
            covariates_std=covariates_std,
            use_intertrials=True,
        )
        if index == 0:
            covariates_mean, covariates_std = record["covariates_mean"], record["covariates_std"]
        records[_canonical_session(resolved)] = record
    dataset = FalconDataset(
        sessions_dict=records,
        calib_sessions_dict=records,
        window_size=100,
        split="package_calibration",
        calibration_n_trials=AFC4_SUPPORT_TRIALS,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=1024,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        pad_value=-1.0,
        side_feature_group="none",
        query_start_trial=0,
    )
    return dataset


__all__ = [
    "M1AFC4PackagePlan",
    "load_m1_emg_support_for_package",
    "load_public_m1_calibration_dataset",
]
