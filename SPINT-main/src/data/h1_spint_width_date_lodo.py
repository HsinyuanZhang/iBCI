"""Activity-only source and strict-target views for the H1 SPINT width follow-up.

This module is deliberately additive.  It consumes the previously immutable
five-date source bundles to reproduce their H-S schedule, but it never emits
or consumes a carrier tensor: every source/target batch is exactly
``(neural, velocity, M4_activity_identity, session)``.  The Phase-1 carrier
cache is read only by the shared bundle verifier so that the already sealed
source schedule can be authenticated; it is not an input to this experiment.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.data.h1_carrierid_date_lodo_phase2 import (
    H1CarrierIdDateLodoSchedule,
    Phase2SourceBinding,
    load_phase2_source_binding,
)
from src.data.h1_carrierid_date_lodo_source import target_sessions_for_date
from src.data.h1_m4_eb_pilot import (
    EXPECTED_NEURONS,
    H1PilotRecord,
    MAX_TRIAL_LENGTH,
    SUPPORT_TRIALS,
    index_heldin_calib,
    interpolate_trial_identity,
    load_record,
)
from src.h1_m4_cce_contract import (
    CONFIRMATORY_DATES,
    FIXED_EPOCHS,
    FIXED_SEED,
    WINDOW_SIZE,
    array_sha256,
    canonical_sha256,
)


DATE_SOURCE_SCHEMA = "h1_spint_identity_width_date_lodo_source_manifest_v1"
DATE_TARGET_SCHEMA = "h1_spint_identity_width_date_lodo_strict_target_view_v1"


class H1SpintWidthDateLodoError(ValueError):
    """Raised when the activity-only date-LODO protocol drifts."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise H1SpintWidthDateLodoError(message)


class H1SpintWidthDateLodoSourceDataset(Dataset):
    """Exact sealed source windows with an activity-only M=4 identity field."""

    def __init__(self, binding: Phase2SourceBinding) -> None:
        self.binding = binding
        self.records = {name: binding.records[name] for name in binding.source_sessions}
        self.window_indices = binding.source_windows
        self.neural_data: dict[str, np.ndarray] = {}
        self.target_data: dict[str, np.ndarray] = {}
        self._trial_identity: dict[tuple[str, float], np.ndarray] = {}
        prehistory = WINDOW_SIZE - 1
        for name in binding.source_sessions:
            record = self.records[name]
            self.neural_data[name] = np.pad(record.neural, ((prehistory, 0), (0, 0)), constant_values=0.0)
            self.target_data[name] = np.pad(record.velocity, ((prehistory, 0), (0, 0)), constant_values=0.0)
            for start in binding.cache.starts_by_session[name]:
                for value in binding.cache.get(name, start).trial_values:
                    self._trial_identity.setdefault((name, float(value)), interpolate_trial_identity(record, value))
        _need(bool(self.window_indices), "date width source dataset has no valid windows")

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, request: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        _need(isinstance(request, tuple) and len(request) == 2, "date width source samples require (window,start)")
        index, calibration_start = int(request[0]), int(request[1])
        session, start = self.window_indices[index]
        entry = self.binding.cache.get(session, calibration_start)
        expected = tuple(self.records[session].trial_values[calibration_start : calibration_start + SUPPORT_TRIALS])
        _need(entry.trial_values == expected, "sealed source cache and M=4 trial values drift")
        identity = np.stack([self._trial_identity[(session, float(value))] for value in entry.trial_values], axis=0)
        _need(identity.shape == (SUPPORT_TRIALS, MAX_TRIAL_LENGTH, EXPECTED_NEURONS), "source M=4 identity shape drift")
        return (
            self.neural_data[session][start : start + WINDOW_SIZE],
            self.target_data[session][start : start + WINDOW_SIZE],
            identity,
            session,
        )


class H1SpintWidthDateLodoSourceDataModule(pl.LightningDataModule):
    """One date's sealed M=4 source binding with no carrier batch route."""

    def __init__(
        self, *, task: str, data_dir: str, phase1_preflight_path: str, outer_date: str,
        batch_size: int = 32, window_size: int = WINDOW_SIZE, calibration_n_trials: int = SUPPORT_TRIALS,
        max_trial_length: int = MAX_TRIAL_LENGTH, seed: int = FIXED_SEED, fixed_epochs: int = FIXED_EPOCHS,
        num_workers: int = 0, pin_memory: bool = False,
    ) -> None:
        super().__init__()
        contract = {
            "task": str(task).lower() == "h1", "outer_date": str(outer_date) in CONFIRMATORY_DATES,
            "batch_size": int(batch_size) == 32, "window_size": int(window_size) == WINDOW_SIZE,
            "calibration_n_trials": int(calibration_n_trials) == SUPPORT_TRIALS,
            "max_trial_length": int(max_trial_length) == MAX_TRIAL_LENGTH, "seed": int(seed) == FIXED_SEED,
            "fixed_epochs": int(fixed_epochs) == FIXED_EPOCHS, "num_workers": int(num_workers) == 0,
        }
        _need(all(contract.values()), f"date width source contract violated: {contract}")
        self.save_hyperparameters(logger=False)
        self._setup_done = False

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit"}:
            raise RuntimeError("date width DataModule permits source fit only")
        if self._setup_done:
            return
        if self.trainer is not None and self.trainer.world_size != 1:
            raise H1SpintWidthDateLodoError("date width protocol fixes one device and one source schedule")
        binding = load_phase2_source_binding(
            data_dir=self.hparams.data_dir,
            phase1_preflight_path=self.hparams.phase1_preflight_path,
            outer_date=self.hparams.outer_date,
        )
        dataset = H1SpintWidthDateLodoSourceDataset(binding)
        # The schedule itself is exactly the immutable phase-1 schedule.  It
        # observes only window_indices, so the activity-only four-field batch
        # interface cannot acquire a carrier through this reuse.
        sampler = H1CarrierIdDateLodoSchedule(dataset, binding)
        source = binding.manifest()
        self.binding, self.train_dataset, self.train_batch_sampler = binding, dataset, sampler
        self._manifest = {
            "schema": DATE_SOURCE_SCHEMA,
            "outer_date": binding.outer_date,
            "phase1_source_manifest_path": str(binding.source_manifest_path),
            "phase1_source_manifest_sha256": binding.source_manifest_sha256,
            "phase1_preflight_path": str(binding.preflight_path),
            "phase1_preflight_sha256": binding.preflight_sha256,
            "phase2_source_binding_sha256": canonical_sha256(source),
            "source_sessions": list(binding.source_sessions),
            "source_files": source["source_files"],
            "source_window_indices_sha256": binding.source_window_indices_sha256,
            "batch_order_sha256": binding.batch_order_sha256,
            "calibration_schedule_sha256": binding.calibration_schedule_sha256,
            "batches_per_epoch": len(sampler),
            "epochs": FIXED_EPOCHS,
            "calibration_n_trials": SUPPORT_TRIALS,
            "window_size": WINDOW_SIZE,
            "target_recordings_opened_during_training_setup": False,
            "target_bytes_read_during_training_setup": 0,
            "minival_or_heldout_enumerated": False,
            "carrier_path": "absent_from_dataset_and_model_inputs",
            "schedule_authentication": "shared immutable phase1 M4 schedule; carrier cache values are not emitted or model inputs",
        }
        self._manifest_sha256 = canonical_sha256(self._manifest)
        self._setup_done = True

    def train_dataloader(self) -> DataLoader[Any]:
        if not self._setup_done:
            raise RuntimeError("call setup('fit') before date width train_dataloader")
        return DataLoader(self.train_dataset, batch_sampler=self.train_batch_sampler, num_workers=0,
                          pin_memory=bool(self.hparams.pin_memory))

    def val_dataloader(self):
        return []

    def test_dataloader(self):
        raise RuntimeError("formal/organizer data are forbidden")

    def predict_dataloader(self):
        raise RuntimeError("strict target prediction exists only in the one-shot evaluator")

    def source_manifest(self) -> dict[str, Any]:
        if not self._setup_done:
            raise RuntimeError("call setup before source manifest")
        return dict(self._manifest)

    @property
    def source_manifest_sha256(self) -> str:
        if not self._setup_done:
            raise RuntimeError("call setup before source manifest SHA")
        return self._manifest_sha256

    @property
    def phase1_manifest_sha256(self) -> str:
        if not self._setup_done:
            raise RuntimeError("call setup before phase1 source manifest SHA")
        return self.binding.source_manifest_sha256


def load_spint_width_date_target_records(data_dir: str | Path, *, outer_date: str) -> dict[str, H1PilotRecord]:
    """Open the public outer-date recordings only inside terminal evaluation."""

    date = str(outer_date)
    _need(date in CONFIRMATORY_DATES, "target date is not a development date")
    paths = index_heldin_calib(data_dir)
    sessions = tuple(target_sessions_for_date(date))
    records = {name: load_record(paths[name]) for name in sessions}
    _need(tuple(records) == sessions and all(record.date == date for record in records.values()), "target partition drift")
    return records


def _first_valid_bin(record: H1PilotRecord, fifth_trial: float) -> int:
    bins = np.flatnonzero(record.eval_mask & np.isfinite(record.trial_num) & (record.trial_num == fifth_trial))
    _need(bins.size > 0, f"{record.session_name}: fifth trial has no eval-valid bin")
    return int(bins[0])


@dataclass(frozen=True)
class DateWidthTargetSupport:
    session_name: str
    support_trials: tuple[float, float, float, float]
    fifth_trial: float
    query_first_bin: int
    identity: np.ndarray
    support_sha256: str


class H1SpintWidthDateLodoStrictTargetDataset(Dataset):
    """Post-fifth-trial M=4 query windows, with activity identity only."""

    def __init__(self, records: Mapping[str, H1PilotRecord], *, outer_date: str) -> None:
        date, expected = str(outer_date), tuple(target_sessions_for_date(str(outer_date)))
        _need(date in CONFIRMATORY_DATES and tuple(records) == expected, "strict target records/date order drift")
        self.records, self.outer_date = {name: records[name] for name in expected}, date
        self.support: dict[str, DateWidthTargetSupport] = {}
        self.window_indices: list[tuple[str, int]] = []
        for name in expected:
            record = self.records[name]
            _need(len(record.trial_values) >= SUPPORT_TRIALS + 1, f"{name}: fewer than M4 plus first query trial")
            values = tuple(float(value) for value in record.trial_values[:SUPPORT_TRIALS])
            fifth = float(record.trial_values[SUPPORT_TRIALS])
            _need(all(record.blocks_for(value).rates.shape[0] >= 2 for value in values), f"{name}: M4 support underspecified")
            boundary = _first_valid_bin(record, fifth)
            identity = np.stack([interpolate_trial_identity(record, value) for value in values], axis=0).astype(np.float32)
            _need(identity.shape == (SUPPORT_TRIALS, MAX_TRIAL_LENGTH, EXPECTED_NEURONS) and np.isfinite(identity).all(),
                  f"{name}: target identity finite/shape drift")
            digest = hashlib.sha256()
            digest.update(np.asarray(values, dtype=np.float64).tobytes()); digest.update(identity.tobytes())
            for value in values:
                trial = record.blocks_for(value)
                digest.update(trial.rates.tobytes()); digest.update(trial.velocity.tobytes()); digest.update(trial.block_indices.tobytes())
            self.support[name] = DateWidthTargetSupport(name, values, fifth, boundary, identity, digest.hexdigest())
            for start in range(boundary, record.neural.shape[0] - WINDOW_SIZE + 1):
                if bool(record.eval_mask[start + WINDOW_SIZE - 1]):
                    self.window_indices.append((name, int(start)))
        _need(bool(self.window_indices), "strict date target has no post-support windows")
        self.window_indices_sha256 = canonical_sha256(self.window_indices)

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        session, start = self.window_indices[int(index)]
        record, support = self.records[session], self.support[session]
        end = start + WINDOW_SIZE
        _need(start >= support.query_first_bin and end <= record.neural.shape[0] and bool(record.eval_mask[end - 1]),
              "strict target crossed support boundary or invalid endpoint")
        return (np.asarray(record.neural[start:end], dtype=np.float32), np.asarray(record.velocity[start:end], dtype=np.float32),
                support.identity, session)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": DATE_TARGET_SCHEMA,
            "outer_date": self.outer_date,
            "sessions": list(self.records),
            "window_indices_sha256": self.window_indices_sha256,
            "samples": len(self.window_indices),
            "support": {
                name: {"support_trials": list(item.support_trials), "fifth_trial": item.fifth_trial,
                       "query_first_bin": item.query_first_bin, "support_sha256": item.support_sha256,
                       "identity_sha256": array_sha256(item.identity)}
                for name, item in self.support.items()
            },
            "all_query_histories_start_at_or_after_fifth_trial": True,
            "carrier_input": "absent",
        }
