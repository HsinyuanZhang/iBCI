"""Source-only date-LODO samples for matched B1 causal-memory training."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tfpd_exploration.src.b1_sfcj_v1 import data as b1data
from tfpd_exploration.src.b1_sfcj_v1.acoustic_basis import AcousticBasis, fit_acoustic_basis
from tfpd_exploration.src.b1_sfcj_v1.constants import FOLDS, N_CHANNELS, N_FREQ, N_MS_BINS, N_SPEC_FRAMES
from tfpd_exploration.src.b1_sfcj_v1.metric import log_from_raw

from .profiles import ProfileAuthority, fit_source_prior_profiles


@dataclass(frozen=True)
class Sample:
    key: str
    date: str
    stream_index: int
    current: np.ndarray
    history: tuple[np.ndarray, ...]
    carrier: np.ndarray
    template_stdlog: np.ndarray
    target_stdlog: np.ndarray
    target_raw: np.ndarray
    in_range: bool


@dataclass(frozen=True)
class FoldData:
    fold: int
    train_dates: tuple[str, ...]
    val_date: str
    basis: AcousticBasis
    profiles: ProfileAuthority
    train_samples: tuple[Sample, ...]
    val_samples: tuple[Sample, ...]
    template_raw_by_date: dict[str, np.ndarray]


def _counts(trial) -> np.ndarray:
    out = b1data.bin_tx_counts(trial.tx).astype(np.float32, copy=False)
    if out.shape != (N_MS_BINS, N_CHANNELS):
        raise RuntimeError(out.shape)
    return out


def _stdlog(raw: np.ndarray, basis: AcousticBasis) -> np.ndarray:
    out = basis.transform_log(log_from_raw(raw).T).T.astype(np.float32)
    if out.shape != (N_FREQ, N_SPEC_FRAMES) or not np.isfinite(out).all():
        raise RuntimeError("invalid standardized-log spectrogram")
    return out


def _template(date: str, basis: AcousticBasis) -> tuple[np.ndarray, np.ndarray]:
    raw = np.median(np.stack([t.spectrogram for t in b1data.first_m3(date)], axis=0), axis=0)
    return raw.astype(np.float32), _stdlog(raw, basis)


def _train_samples_for_date(
    date: str,
    basis: AcousticBasis,
    carrier: np.ndarray,
    template_stdlog: np.ndarray,
) -> list[Sample]:
    trials = b1data.calib_trials(date)
    activities = [_counts(t) for t in trials]
    out = []
    for idx in range(3, len(trials)):
        out.append(
            Sample(
                key=f"train:{date}:calib:{idx}",
                date=date,
                stream_index=idx - 3,
                current=activities[idx],
                history=tuple(activities[:idx]),
                carrier=carrier.astype(np.float32),
                template_stdlog=template_stdlog,
                target_stdlog=_stdlog(trials[idx].spectrogram, basis),
                target_raw=trials[idx].spectrogram.astype(np.float32),
                in_range=True,
            )
        )
    return out


def _val_samples_for_date(
    date: str,
    basis: AcousticBasis,
    carrier: np.ndarray,
    template_stdlog: np.ndarray,
    n_in_range: int,
) -> list[Sample]:
    seed = b1data.calib_trials(date)[:3]
    queries = b1data.query_trials(date)
    history = [_counts(t) for t in seed]
    out = []
    for stream_index, trial in enumerate(queries):
        current = _counts(trial)
        out.append(
            Sample(
                key=f"val:{date}:{trial.split}:{trial.trial_index}",
                date=date,
                stream_index=stream_index,
                current=current,
                history=tuple(history),
                carrier=carrier.astype(np.float32),
                template_stdlog=template_stdlog,
                target_stdlog=_stdlog(trial.spectrogram, basis),
                target_raw=trial.spectrogram.astype(np.float32),
                in_range=stream_index < int(n_in_range),
            )
        )
        # Decode-before-commit: only future samples observe this activity.
        history.append(current)
    return out


def build_fold_data(fold: int) -> FoldData:
    spec = FOLDS[int(fold)]
    train_dates = tuple(spec["train_dates"])
    val_date = str(spec["val_date"])
    basis_trials = []
    for date in train_dates:
        basis_trials.extend(b1data.calib_trials(date))
    basis = fit_acoustic_basis(basis_trials, train_dates)

    all_dates = train_dates + (val_date,)
    profiles = fit_source_prior_profiles(
        train_dates=train_dates,
        all_dates=all_dates,
        basis=basis,
        first_m3=b1data.first_m3,
        held_calib=lambda date: b1data.calib_trials(date)[3:],
    )
    templates = {date: _template(date, basis) for date in all_dates}
    train_samples = []
    for date in train_dates:
        train_samples.extend(
            _train_samples_for_date(
                date,
                basis,
                profiles.standardized_by_date[date],
                templates[date][1],
            )
        )
    val_samples = _val_samples_for_date(
        val_date,
        basis,
        profiles.standardized_by_date[val_date],
        templates[val_date][1],
        int(spec["n_in_range"]),
    )
    if len(train_samples) == 0 or len(val_samples) != int(spec["n_query"]):
        raise RuntimeError("fold sample count mismatch")
    return FoldData(
        fold=int(fold),
        train_dates=train_dates,
        val_date=val_date,
        basis=basis,
        profiles=profiles,
        train_samples=tuple(train_samples),
        val_samples=tuple(val_samples),
        template_raw_by_date={date: templates[date][0] for date in all_dates},
    )


def collate(samples: list[Sample], *, law: str = "GROWING") -> dict:
    if law not in ("GROWING", "FIXED3"):
        raise ValueError(law)
    if not samples:
        raise ValueError("empty batch")
    histories = [sample.history if law == "GROWING" else sample.history[:3] for sample in samples]
    kmax = max(len(h) for h in histories)
    hist = np.zeros((len(samples), kmax, N_MS_BINS, N_CHANNELS), dtype=np.float32)
    mask = np.zeros((len(samples), kmax), dtype=np.float32)
    for row, values in enumerate(histories):
        hist[row, : len(values)] = np.stack(values, axis=0)
        mask[row, : len(values)] = 1.0
    return {
        "keys": [s.key for s in samples],
        "current": np.stack([s.current for s in samples], axis=0),
        "history": hist,
        "history_mask": mask,
        "carrier": np.stack([s.carrier for s in samples], axis=0),
        "template_stdlog": np.stack([s.template_stdlog for s in samples], axis=0),
        "target_stdlog": np.stack([s.target_stdlog for s in samples], axis=0),
        "target_raw": np.stack([s.target_raw for s in samples], axis=0),
        "in_range": np.asarray([s.in_range for s in samples], dtype=bool),
    }
