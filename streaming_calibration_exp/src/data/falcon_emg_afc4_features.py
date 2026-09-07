"""Isolated M1 q=3 source-frozen EMG analytic functional carrier (AFC4).

This module has no RT/K4/N4 dependency and never discovers held-out/minival or
EvalAI files.  A ``SourceFrozenEMGAFC4Plan`` is fit from explicit M1 source
paths only.  It freezes source EMG mean/scale/PCA and a source-only descriptor
normalizer, then builds one of five fixed-width inputs for a target session:

* ``full``  : normalized ``[w1,w2,w3,b]``;
* ``zero4`` : four exact zeros, without a target descriptor fit;
* ``rs4``   : full rows under one deterministic complete-row permutation; and
* ``b4``    : normalized baseline coordinate only (post-normalization mask);
* ``ls4``   : full rows after a deterministic derangement of the ten EMG-score
  rows relative to the ten per-channel firing-rate rows.

The target fit consumes precisely its first ten movement-window EMG/neural
trials.  Query data is never an input to descriptor construction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from pynwb import NWBHDF5IO


AFC4_DIM = 4
AFC4_Q = 3
AFC4_SUPPORT_TRIALS = 10
AFC4_RIDGE_ALPHA = 1.0
AFC4_VERSION = "m1_source_frozen_emg_afc4_v1"
_EPS = 1.0e-12
_ARMS = ("full", "zero4", "rs4", "b4", "ls4")
_FORBIDDEN_PATH_TOKENS = ("held-out", "minival", "evalai", "formal", "test")


def _finite_2d(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) <= 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty [rows, columns] matrix")
    return array


def _canonical_session(path: Path) -> str:
    try:
        # The base FALCON datamodule identifies M1 sessions as ``ses-YYYYMMDD``.
        # Preserve that exact namespace rather than silently dropping ``ses-``
        # while parsing the NWB filename.
        return f"ses-{path.name.split('_ses-')[1].split('_behavior')[0]}"
    except IndexError as exc:
        raise ValueError(f"cannot parse M1 session from {path.name}") from exc


def _require_source_path(path: Path) -> None:
    resolved = path.resolve()
    text = str(resolved).lower()
    if "spint-main/data/000941/sub-monkeyl-held-in-calib/" not in text:
        raise ValueError(f"AFC4 source leaves exact M1 held-in-calib scope: {resolved}")
    if any(token in text for token in _FORBIDDEN_PATH_TOKENS):
        raise ValueError(f"AFC4 source contains a forbidden path token: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)


def deterministic_afc4_row_permutation(num_channels: int, *, session_name: str, seed: int) -> np.ndarray:
    """Return a complete-row, deterministic, nonidentity AFC4 schedule."""
    if int(num_channels) < 2:
        raise ValueError("AFC4-RS4 needs at least two channels")
    payload = f"m1-emg-afc4-rs4-v1:{int(seed)}:{session_name}:{int(num_channels)}".encode()
    generator = np.random.RandomState(int.from_bytes(hashlib.sha256(payload).digest()[:4], "little"))
    order = generator.permutation(int(num_channels))
    if np.array_equal(order, np.arange(int(num_channels))):
        order = np.roll(order, 1)
    if np.any(order == np.arange(int(num_channels))):
        # Fixed points are permitted for ordinary row shuffles, but an all-row
        # nonidentity null is stronger and remains deterministic.
        order = np.roll(np.arange(int(num_channels)), 1)
    return order.astype(np.int64, copy=False)


def deterministic_afc4_label_derangement(*, session_name: str, seed: int) -> np.ndarray:
    """Return a deterministic, complete M10 trial-label derangement.

    A nonzero cyclic shift is deliberately used instead of repeatedly sampling
    permutations until fixed points disappear.  It is auditable, guarantees
    that every firing-rate row receives the EMG score from another trial, and
    preserves the exact number, exposure and marginal distribution of the ten
    vector-valued EMG labels.
    """
    payload = f"m1-emg-afc4-ls4-v1:{int(seed)}:{session_name}".encode()
    shift = 1 + int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") % (
        AFC4_SUPPORT_TRIALS - 1
    )
    order = np.roll(np.arange(AFC4_SUPPORT_TRIALS, dtype=np.int64), shift)
    if np.any(order == np.arange(AFC4_SUPPORT_TRIALS)):
        raise RuntimeError("AFC4-LS4 schedule is not a complete derangement")
    return order


@dataclass(frozen=True)
class M1EMGSupport:
    session_name: str
    path: Path
    full_trial_emg: np.ndarray
    support_emg: np.ndarray
    support_rates: np.ndarray
    n_trials: int
    query_checksum: str
    source_for_pca: bool
    emg_trials_materialized: int


@dataclass(frozen=True)
class SourceFrozenEMGBasis:
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray  # [3, emg_dims]
    singular_values: np.ndarray
    explained_energy: np.ndarray
    sign_anchor_indices: np.ndarray

    def project(self, emg: np.ndarray) -> np.ndarray:
        values = _finite_2d(emg, name="emg")
        if values.shape[1] != self.mean.size:
            raise ValueError("EMG dimensionality differs from source-frozen basis")
        return ((values - self.mean) / self.scale) @ self.components.T


def _fit_basis(source_emg: Sequence[np.ndarray]) -> SourceFrozenEMGBasis:
    matrices = tuple(_finite_2d(value, name=f"source_emg[{index}]") for index, value in enumerate(source_emg))
    if len(matrices) < 1 or len({matrix.shape[1] for matrix in matrices}) != 1:
        raise ValueError("AFC4 requires nonempty source EMG with a shared dimension")
    stacked = np.concatenate(matrices, axis=0)
    if min(stacked.shape) < AFC4_Q:
        raise ValueError("source EMG cannot support q=3 PCA")
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0, ddof=0)
    scale[scale <= _EPS] = 1.0
    standardized = (stacked - mean) / scale
    _u, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    components = vt[:AFC4_Q].copy()
    anchors = np.argmax(np.abs(components), axis=1).astype(np.int64)
    for index, anchor in enumerate(anchors):
        if components[index, anchor] < 0.0:
            components[index] *= -1.0
    energy = np.square(singular_values[:AFC4_Q]) / float(np.square(singular_values).sum())
    return SourceFrozenEMGBasis(mean, scale, components, singular_values[:AFC4_Q].copy(), energy, anchors)


def _fit_encoding(scores: np.ndarray, rates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z, y = _finite_2d(scores, name="support_scores"), _finite_2d(rates, name="support_rates")
    if z.shape[0] != AFC4_SUPPORT_TRIALS or y.shape[0] != AFC4_SUPPORT_TRIALS:
        raise ValueError("AFC4 target fit requires exactly M10 support rows")
    if z.shape[1] != AFC4_Q:
        raise ValueError("AFC4 target fit requires exactly q=3 source scores")
    design = np.column_stack((np.ones(AFC4_SUPPORT_TRIALS, dtype=np.float64), z))
    rank = int(np.linalg.matrix_rank(design))
    if rank != AFC4_Q + 1:
        raise ValueError(f"AFC4 M10 affine design must have rank 4, got {rank}")
    penalty = np.eye(AFC4_Q + 1, dtype=np.float64) * AFC4_RIDGE_ALPHA
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    weights, baseline = beta[1:].T, beta[0]
    raw = np.column_stack((weights[:, 0], weights[:, 1], weights[:, 2], baseline))
    if not np.isfinite(raw).all():
        raise RuntimeError("AFC4 target fit produced non-finite descriptor")
    return raw.astype(np.float32), design


def _query_checksum(session_name: str, trials, *, support: int) -> str:
    if len(trials) <= support:
        raise ValueError(f"{session_name} has no strict post-M10 query")
    boundary = {
        "session": session_name,
        "support_trial_range": [0, support],
        "query_trial_range": [support, int(len(trials))],
        "support_last_stop": float(trials.iloc[support - 1]["stop_time"]),
        "query_first_start": float(trials.iloc[support]["start_time"]),
        "query_last_stop": float(trials.iloc[-1]["stop_time"]),
    }
    return hashlib.sha256(json.dumps(boundary, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_m1_emg_support(path: Path, *, source_for_pca: bool) -> M1EMGSupport:
    """Read source PCA rows or a target's M10-only AFC4 calibration values.

    Source sessions may contribute all of their movement-window EMG rows to
    the fold PCA.  A left-out target is deliberately narrower: this function
    materializes EMG values for exactly trials ``[0,10)`` and requests raw
    spike times only inside that same temporal support.  Later target EMG and
    neural values are never loaded by this AFC4 construction path.
    """
    path = Path(path)
    _require_source_path(path)
    session = _canonical_session(path)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        trials = nwb.trials.to_dataframe()
        raw = nwb.acquisition["preprocessed_emg"]
        channel_names = tuple(raw.time_series.keys())
        if not channel_names:
            raise ValueError(f"{session}: no preprocessed EMG channels")
        first = raw.get_timeseries(channel_names[0])
        timestamps = np.asarray(first.timestamps[:], dtype=np.float64)
        if len(trials) <= AFC4_SUPPORT_TRIALS:
            raise ValueError(f"{session} violates M1 trial/query contract")
        emg_trial_count = len(trials) if source_for_pca else AFC4_SUPPORT_TRIALS
        full_rows: list[np.ndarray] = []
        for index in range(emg_trial_count):
            onset, stop = float(trials.iloc[index]["move_onset_time"]), float(trials.iloc[index]["stop_time"])
            if not math.isfinite(onset):
                onset = float(trials.iloc[index]["start_time"])
            left = int(np.searchsorted(timestamps, onset, side="left"))
            right = int(np.searchsorted(timestamps, stop, side="right"))
            if right <= left:
                raise ValueError(f"{session}: empty movement-window EMG for trial {index}")
            # Slice the HDF5 datasets before conversion.  For a target this is
            # never evaluated at a query-trial index.
            segment = np.column_stack(
                [np.asarray(raw.get_timeseries(channel).data[left:right], dtype=np.float64) for channel in channel_names]
            )
            if segment.shape[0] == 0 or not np.isfinite(segment).all():
                raise ValueError(f"{session}: invalid movement-window EMG for trial {index}")
            full_rows.append(segment.mean(axis=0))

        support_rates = np.zeros((AFC4_SUPPORT_TRIALS, len(nwb.units.id)), dtype=np.float64)
        support_start = float(trials.iloc[0]["move_onset_time"])
        if not math.isfinite(support_start):
            support_start = float(trials.iloc[0]["start_time"])
        support_stop = float(trials.iloc[AFC4_SUPPORT_TRIALS - 1]["stop_time"])
        # HDMF applies this interval before returning each unit's ragged spike
        # vector, so target query spike values are not exposed to this loader.
        support_spike_times = [
            np.asarray(nwb.units.get_unit_spike_times(unit_index, in_interval=(support_start, support_stop)), dtype=np.float64)
            for unit_index in range(len(nwb.units.id))
        ]
        for index in range(AFC4_SUPPORT_TRIALS):
            onset, stop = float(trials.iloc[index]["move_onset_time"]), float(trials.iloc[index]["stop_time"])
            if not math.isfinite(onset):
                onset = float(trials.iloc[index]["start_time"])
            duration = max(stop - onset, 1.0e-12)
            for unit_index, times in enumerate(support_spike_times):
                support_rates[index, unit_index] = np.count_nonzero((times >= onset) & (times < stop)) / duration
    if len(full_rows) < AFC4_SUPPORT_TRIALS:
        raise ValueError(f"{session} violates M1 trial/query contract")
    full = np.vstack(full_rows)
    return M1EMGSupport(
        session, path.resolve(), full, full[:AFC4_SUPPORT_TRIALS].copy(), support_rates,
        int(len(trials)), _query_checksum(session, trials, support=AFC4_SUPPORT_TRIALS),
        bool(source_for_pca), int(emg_trial_count),
    )


class SourceFrozenEMGAFC4Plan:
    """One source-joint q=3 plan shared by Full/Zero4/B4/RS4/LS4."""

    def __init__(self, source_paths: Mapping[str, Path], *, shuffle_seed: int) -> None:
        if not source_paths:
            raise ValueError("AFC4 plan needs explicit source paths")
        self.shuffle_seed = int(shuffle_seed)
        self.support = {name: load_m1_emg_support(path, source_for_pca=True) for name, path in source_paths.items()}
        if len(set(self.support)) != len(self.support):
            raise ValueError("AFC4 source sessions are not unique")
        self.source_session_names = tuple(sorted(self.support))
        self.basis = _fit_basis([record.full_trial_emg for record in self.support.values()])
        self.raw_full: dict[str, np.ndarray] = {}
        self.raw_ls4: dict[str, np.ndarray] = {}
        self.target_fit_calls: dict[str, int] = {name: 0 for name in self.support}
        for name, record in self.support.items():
            self.raw_full[name] = self._raw_full(record)
        source_rows = np.concatenate(list(self.raw_full.values()), axis=0)
        self.mean = source_rows.mean(axis=0).astype(np.float32)
        self.std = source_rows.std(axis=0).astype(np.float32)
        self.std[self.std <= 1.0e-6] = 1.0

    def _raw_full(self, record: M1EMGSupport) -> np.ndarray:
        self.target_fit_calls[record.session_name] += 1
        scores = self.basis.project(record.support_emg)
        raw, _design = _fit_encoding(scores, record.support_rates)
        return raw

    def _raw_ls4(self, record: M1EMGSupport) -> np.ndarray:
        self.target_fit_calls[record.session_name] += 1
        scores = self.basis.project(record.support_emg)
        order = deterministic_afc4_label_derangement(
            session_name=record.session_name, seed=self.shuffle_seed
        )
        raw, _design = _fit_encoding(scores[order], record.support_rates)
        return raw

    def add_target(self, path: Path) -> str:
        record = load_m1_emg_support(path, source_for_pca=False)
        if record.session_name not in self.support:
            self.support[record.session_name] = record
            self.target_fit_calls[record.session_name] = 0
        return record.session_name

    def normalized(self, session_name: str, *, arm: str) -> np.ndarray:
        if arm not in _ARMS:
            raise ValueError(f"unsupported AFC4 arm {arm!r}")
        if session_name not in self.support:
            raise KeyError(f"AFC4 session has not been explicitly added: {session_name}")
        record = self.support[session_name]
        if arm == "zero4":
            return np.zeros((record.support_rates.shape[1], AFC4_DIM), dtype=np.float32)
        if arm == "ls4":
            if session_name not in self.raw_ls4:
                self.raw_ls4[session_name] = self._raw_ls4(record)
            raw = self.raw_ls4[session_name]
        else:
            if session_name not in self.raw_full:
                self.raw_full[session_name] = self._raw_full(record)
            raw = self.raw_full[session_name]
        # All arms use the normalizer fitted from correctly paired source
        # carriers.  Refitting it on LS4 would mix the pairing intervention with
        # a different coordinate system.
        values = ((raw - self.mean) / self.std).astype(np.float32)
        if arm == "b4":
            values[:, :3] = 0.0
        schedule = deterministic_afc4_row_permutation(values.shape[0], session_name=session_name, seed=self.shuffle_seed)
        if arm == "rs4":
            values = values[schedule]
        return values

    def receipt(self, *, arm: str) -> dict[str, object]:
        if arm not in _ARMS:
            raise ValueError("unknown AFC4 arm")
        schedules = {
            name: deterministic_afc4_row_permutation(record.support_rates.shape[1], session_name=name, seed=self.shuffle_seed)
            for name, record in self.support.items()
        }
        label_schedules = {
            name: deterministic_afc4_label_derangement(
                session_name=name, seed=self.shuffle_seed
            )
            for name in self.support
        }
        return {
            "version": AFC4_VERSION,
            "arm": arm,
            "q": AFC4_Q,
            "support_trials": AFC4_SUPPORT_TRIALS,
            "ridge_alpha": AFC4_RIDGE_ALPHA,
            "basis_source_sessions": list(self.source_session_names),
            "pca": {
                "mean": self.basis.mean.tolist(), "scale": self.basis.scale.tolist(),
                "components": self.basis.components.tolist(), "singular_values": self.basis.singular_values.tolist(),
                "explained_energy": self.basis.explained_energy.tolist(),
                "sign_anchor_indices": self.basis.sign_anchor_indices.tolist(),
                "sign_rule": "largest-absolute loading nonnegative; lowest loading index breaks ties",
            },
            "normalizer": {"mean": self.mean.tolist(), "std": self.std.tolist(), "fit_source_sessions": list(self.source_session_names)},
            "query_checksums": {name: record.query_checksum for name, record in self.support.items()},
            "materialization_scope": {
                name: {
                    "source_for_pca": record.source_for_pca,
                    "emg_trial_range": [0, record.emg_trials_materialized],
                    "raw_spike_trial_range": [0, AFC4_SUPPORT_TRIALS],
                    "target_query_emg_or_neural_values_read": False if not record.source_for_pca else None,
                }
                for name, record in self.support.items()
            },
            "row_permutation_schedule": {name: order.tolist() for name, order in schedules.items()},
            "label_derangement_schedule": {
                name: order.tolist() for name, order in label_schedules.items()
            },
            "zero4_target_fit_calls": 0 if arm == "zero4" else None,
            "b4_mask": "post_normalization_coordinates_0_to_2_zero" if arm == "b4" else None,
            "rs4": "complete_normalized_row_permutation" if arm == "rs4" else None,
            "ls4": (
                "M10 EMG-score rows deterministically deranged before ridge; firing-rate rows, "
                "source PCA and correct-carrier normalizer unchanged"
                if arm == "ls4" else None
            ),
        }
