"""Activity-only H1 M=4 data contract for the SPINT identity-width sweep.

This is intentionally separate from the H1 CarrierID data modules.  It reuses
the public H1 M=4 record/identity primitives and reproduces the historical
fold-0 source window order and calibration-start schedule, but a batch contains
only ``(neural, target, identity, session)``.  No carrier is created, returned,
or accepted by the model.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler

from src.data.h1_m4_eb_pilot import (
    FOLD0_DATE,
    H1_M4_FOLD0_SOURCE,
    H1_M4_FOLD0_TARGET,
    H1PilotRecord,
    MAX_TRIAL_LENGTH,
    PilotDataError,
    WINDOW,
    _window_manifest_hash,
    canonical_sha256,
    interpolate_identity,
    interpolate_trial_identity,
    legal_contiguous_starts,
    load_source_records,
    load_target_records,
)
from src.models.components.spint_identity_width import identity_dense_macs, identity_parameter_count


SUPPORT_TRIALS = 4
FIXED_SEED = 42
FIXED_EPOCHS = 50
BATCH_SIZE = 32
EXPECTED_SOURCE_WINDOW_SHA256 = "e7ff7468925335cafcfe6e8f84160549bdee251303be6b0b1e6f8963d6219b44"
EXPECTED_BATCH_ORDER_SHA256 = "14fc810050fb5066137095201786c4a905374600b129598c88efbf1126dd12e4"
EXPECTED_CALIBRATION_SCHEDULE_SHA256 = "96ffdee73186e1b05ddc90a998a059e72d12f5f2d0b529fc1d0ec585e12b7dd4"
EXPECTED_QUERY_WINDOW_SHA256 = "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"


class H1SpintWidthDataError(ValueError):
    """Fail-closed error for the activity-only H1 SPINT width protocol."""


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise H1SpintWidthDataError(message)


class H1SpintWidthSourceDataset(Dataset):
    """Exact M=4 SPINT source windows without any carrier-valued field."""

    def __init__(self, records: Mapping[str, H1PilotRecord]) -> None:
        self.records = {name: records[name] for name in H1_M4_FOLD0_SOURCE}
        self.neural_data: dict[str, np.ndarray] = {}
        self.target_data: dict[str, np.ndarray] = {}
        self.eval_mask: dict[str, np.ndarray] = {}
        self.starts_by_session: dict[str, tuple[int, ...]] = {}
        self._trial_identity: dict[tuple[str, float], np.ndarray] = {}
        self.window_indices: list[tuple[str, int]] = []
        prehistory = WINDOW - 1
        for name in H1_M4_FOLD0_SOURCE:
            record = self.records[name]
            starts = tuple(int(start) for start in legal_contiguous_starts(record))
            _need(bool(starts), f"{name}: no legal contiguous M=4 support start")
            self.starts_by_session[name] = starts
            self.neural_data[name] = np.pad(record.neural, ((prehistory, 0), (0, 0)), constant_values=0.0)
            self.target_data[name] = np.pad(record.velocity, ((prehistory, 0), (0, 0)), constant_values=0.0)
            self.eval_mask[name] = np.pad(record.eval_mask, (prehistory, 0), constant_values=False)
            for calibration_start in starts:
                values = tuple(record.trial_values[calibration_start : calibration_start + SUPPORT_TRIALS])
                _need(len(values) == SUPPORT_TRIALS, f"{name}: M=4 support range drift")
                for value in values:
                    key = (name, float(value))
                    if key not in self._trial_identity:
                        self._trial_identity[key] = interpolate_trial_identity(record, value)
            for start in range(self.neural_data[name].shape[0] - WINDOW + 1):
                if self.eval_mask[name][start + WINDOW - 1]:
                    self.window_indices.append((name, start))
        _need(bool(self.window_indices), "activity-only width source dataset has no eval-valid windows")
        self.window_indices_sha256 = _window_manifest_hash(self.window_indices)
        _need(
            self.window_indices_sha256 == EXPECTED_SOURCE_WINDOW_SHA256,
            "source window index differs from the established H1 M=4 H-S schedule",
        )

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, request: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        _need(isinstance(request, tuple) and len(request) == 2, "source samples require (window_index, calibration_start)")
        index, calibration_start = int(request[0]), int(request[1])
        session, start = self.window_indices[index]
        _need(calibration_start in self.starts_by_session[session], "source schedule selected an illegal M=4 start")
        record = self.records[session]
        values = tuple(record.trial_values[calibration_start : calibration_start + SUPPORT_TRIALS])
        _need(len(values) == SUPPORT_TRIALS, "scheduled support does not contain exactly M=4 trials")
        identity = np.stack([self._trial_identity[(session, float(value))] for value in values], axis=0)
        _need(identity.shape == (SUPPORT_TRIALS, MAX_TRIAL_LENGTH, record.num_neurons), "M=4 identity shape drift")
        end = start + WINDOW
        return self.neural_data[session][start:end], self.target_data[session][start:end], identity, session


class H1SpintWidthBatchSampler(Sampler[list[tuple[int, int]]]):
    """Literal public H1 M=4 source ordering/start schedule, without carriers."""

    def __init__(self, dataset: H1SpintWidthSourceDataset) -> None:
        self.dataset = dataset
        grouped: dict[str, list[int]] = {name: [] for name in H1_M4_FOLD0_SOURCE}
        for index, (session, _start) in enumerate(dataset.window_indices):
            grouped[session].append(index)
        batches: list[list[int]] = []
        for name in H1_M4_FOLD0_SOURCE:
            indices = random.Random(FIXED_SEED).sample(grouped[name], len(grouped[name]))
            batches.extend(
                indices[offset : offset + BATCH_SIZE]
                for offset in range(0, len(indices), BATCH_SIZE)
                if len(indices[offset : offset + BATCH_SIZE]) == BATCH_SIZE
            )
        self.batches = tuple(tuple(batch) for batch in random.Random(FIXED_SEED).sample(batches, len(batches)))
        self.flat_indices = np.asarray([index for batch in self.batches for index in batch], dtype=np.int64)
        self.batch_order_sha256 = _array_sha256(self.flat_indices)
        flat_sessions = np.asarray([dataset.window_indices[int(index)][0] for index in self.flat_indices], dtype=object)
        schedule = np.empty((FIXED_EPOCHS, len(self.flat_indices)), dtype=np.int16)
        for name in H1_M4_FOLD0_SOURCE:
            positions = np.flatnonzero(flat_sessions == name)
            legal = np.asarray(dataset.starts_by_session[name], dtype=np.int16)
            token = hashlib.sha256(f"{FIXED_SEED}|m4-schedule|{name}".encode()).digest()
            generator = np.random.default_rng(int.from_bytes(token[:8], "big"))
            schedule[:, positions] = legal[generator.integers(0, len(legal), size=(FIXED_EPOCHS, len(positions)))]
        self.schedule = schedule
        self.schedule_sha256 = _array_sha256(schedule)
        _need(self.batch_order_sha256 == EXPECTED_BATCH_ORDER_SHA256, "source batch order differs from established H1 M=4 schedule")
        _need(
            self.schedule_sha256 == EXPECTED_CALIBRATION_SCHEDULE_SHA256,
            "source calibration schedule differs from established H1 M=4 schedule",
        )
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.batches)

    def reset_epoch(self) -> None:
        self._epoch = 0

    def __iter__(self):
        if self._epoch >= FIXED_EPOCHS:
            raise RuntimeError("fixed width-sweep M=4 schedule exhausted after epoch 49")
        starts = self.schedule[self._epoch]
        offset = 0
        for batch in self.batches:
            selected = starts[offset : offset + len(batch)]
            offset += len(batch)
            yield [(index, int(calibration_start)) for index, calibration_start in zip(batch, selected)]
        self._epoch += 1


class H1SpintWidthFold0DataModule(pl.LightningDataModule):
    """Source-only fold-0 DataModule for a strictly activity-only H-S sweep."""

    def __init__(
        self,
        *,
        task: str,
        data_dir: str,
        batch_size: int = BATCH_SIZE,
        window_size: int = WINDOW,
        calibration_n_trials: int = SUPPORT_TRIALS,
        max_trial_length: int = MAX_TRIAL_LENGTH,
        random_calibration: bool = True,
        smooth_calibration: bool = False,
        interpolate_trials: bool = True,
        interpolate_trials_kind: str = "cubic",
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = FIXED_SEED,
        fixed_epochs: int = FIXED_EPOCHS,
    ) -> None:
        super().__init__()
        fixed = {
            "task": str(task).lower() == "h1",
            "batch_size": int(batch_size) == BATCH_SIZE,
            "window_size": int(window_size) == WINDOW,
            "calibration_n_trials": int(calibration_n_trials) == SUPPORT_TRIALS,
            "max_trial_length": int(max_trial_length) == MAX_TRIAL_LENGTH,
            "random_calibration": bool(random_calibration),
            "smooth_calibration": not bool(smooth_calibration),
            "interpolate_trials": bool(interpolate_trials),
            "interpolate_trials_kind": str(interpolate_trials_kind) == "cubic",
            "num_workers": int(num_workers) == 0,
            "seed": int(seed) == FIXED_SEED,
            "fixed_epochs": int(fixed_epochs) == FIXED_EPOCHS,
        }
        _need(all(fixed.values()), f"H1 activity-only width data contract violated: {fixed}")
        self.save_hyperparameters(logger=False)
        self.batch_size_per_device = BATCH_SIZE
        self._setup_done = False

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit"}:
            raise RuntimeError("H1 activity-only width DataModule permits source fit only")
        if self._setup_done:
            return
        if self.trainer is not None and self.trainer.world_size != 1:
            raise H1SpintWidthDataError("width sweep fixes one device and one ordered source schedule")
        records = load_source_records(self.hparams.data_dir)
        dataset = H1SpintWidthSourceDataset(records)
        sampler = H1SpintWidthBatchSampler(dataset)
        self.records, self.train_dataset, self.train_batch_sampler = records, dataset, sampler
        self._manifest = {
            "schema": "h1_spint_identity_width_fold0_source_manifest_v1",
            "fold_date": FOLD0_DATE,
            "source_sessions": list(H1_M4_FOLD0_SOURCE),
            "target_sessions_not_opened": list(H1_M4_FOLD0_TARGET),
            "files": [
                {
                    "role": "source_heldin_calib",
                    "session": name,
                    "date": records[name].date,
                    "sha256": records[name].input_sha256,
                    "size_bytes": records[name].path.stat().st_size,
                }
                for name in H1_M4_FOLD0_SOURCE
            ],
            "source_window_indices_sha256": dataset.window_indices_sha256,
            "batch_order_sha256": sampler.batch_order_sha256,
            "calibration_schedule_sha256": sampler.schedule_sha256,
            "batches_per_epoch": len(sampler),
            "scheduled_samples_per_epoch": int(sampler.flat_indices.size),
            "epochs": FIXED_EPOCHS,
            "calibration_n_trials": SUPPORT_TRIALS,
            "window_size": WINDOW,
            "target_nwb_opened_during_training_setup": False,
            "minival_or_heldout_enumerated": False,
            "carrier_path": "absent_from_dataset_and_model_inputs",
        }
        self._manifest_sha256 = canonical_sha256(self._manifest)
        self._setup_done = True

    def train_dataloader(self) -> DataLoader[Any]:
        if not self._setup_done:
            raise RuntimeError("call setup('fit') before train_dataloader")
        return DataLoader(self.train_dataset, batch_sampler=self.train_batch_sampler, num_workers=0, pin_memory=bool(self.hparams.pin_memory))

    def val_dataloader(self):
        return []

    def test_dataloader(self):
        raise RuntimeError("formal/organizer test data are forbidden")

    def predict_dataloader(self):
        raise RuntimeError("target prediction is available only through the terminal evaluator")

    def source_manifest(self) -> dict[str, Any]:
        if not self._setup_done:
            raise RuntimeError("call setup('fit') before requesting source manifest")
        return dict(self._manifest)

    @property
    def source_manifest_sha256(self) -> str:
        if not self._setup_done:
            raise RuntimeError("call setup('fit') before requesting source manifest SHA")
        return self._manifest_sha256


@dataclass(frozen=True)
class TargetSupport:
    session: str
    trial_values: tuple[float, float, float, float]
    fifth_trial: float
    query_first_bin: int
    identity: np.ndarray
    support_sha256: str


class H1SpintWidthStrictTargetDataset(Dataset):
    """Fold-0 strict post-support query windows with activity identity only."""

    def __init__(self, records: Mapping[str, H1PilotRecord]) -> None:
        self.records = {name: records[name] for name in H1_M4_FOLD0_TARGET}
        self.support: dict[str, TargetSupport] = {}
        self.window_indices: list[tuple[str, int]] = []
        for name in H1_M4_FOLD0_TARGET:
            record = self.records[name]
            values = tuple(float(value) for value in record.trial_values[:SUPPORT_TRIALS])
            _need(len(values) == SUPPORT_TRIALS, f"{name}: target lacks four support trials")
            fifth = float(record.trial_values[SUPPORT_TRIALS])
            fifth_bins = np.flatnonzero(record.eval_mask & np.isfinite(record.trial_num) & (record.trial_num == fifth))
            _need(fifth_bins.size > 0, f"{name}: fifth trial has no eval-valid bin")
            boundary = int(fifth_bins[0])
            identity = interpolate_identity(record, values)
            digest = hashlib.sha256()
            digest.update(np.asarray(values, dtype=np.float64).tobytes())
            digest.update(identity.tobytes())
            self.support[name] = TargetSupport(name, values, fifth, boundary, identity, digest.hexdigest())
            for start in range(boundary, record.neural.shape[0] - WINDOW + 1):
                if record.eval_mask[start + WINDOW - 1]:
                    self.window_indices.append((name, start))
        _need(bool(self.window_indices), "strict target query has no post-support windows")
        self.window_indices_sha256 = _window_manifest_hash(self.window_indices)
        _need(
            self.window_indices_sha256 == EXPECTED_QUERY_WINDOW_SHA256,
            "strict target query index differs from established H1 M=4 endpoint",
        )

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        session, start = self.window_indices[int(index)]
        record, support = self.records[session], self.support[session]
        end = start + WINDOW
        _need(start >= support.query_first_bin and record.eval_mask[end - 1], "query window crossed M=4 support boundary")
        return record.neural[start:end], record.velocity[start:end], support.identity, session

    def support_hashes(self) -> dict[str, Any]:
        return {
            name: {
                "trial_values": list(value.trial_values),
                "fifth_trial": value.fifth_trial,
                "query_first_bin": value.query_first_bin,
                "support_sha256": value.support_sha256,
            }
            for name, value in self.support.items()
        }


def load_fold0_strict_target(data_dir: str) -> H1SpintWidthStrictTargetDataset:
    """Open target recordings only when explicitly called by terminal evaluation."""

    return H1SpintWidthStrictTargetDataset(load_target_records(data_dir))


def width_accounting_manifest(widths: Sequence[int]) -> dict[str, Any]:
    """Stable identity-path count table used by CPU preflight and receipts."""

    table = {
        str(int(width)): {
            "identity_width": int(width),
            "identity_parameters": identity_parameter_count(int(width)),
            "identity_dense_macs_m4_n176": identity_dense_macs(int(width)),
        }
        for width in widths
    }
    if table.get("1024", {}).get("identity_parameters") != 5_965_500:
        raise H1SpintWidthDataError("H-S identity parameter reference arithmetic drift")
    if table.get("1024", {}).get("identity_dense_macs_m4_n176") != 2_709_848_064:
        raise H1SpintWidthDataError("H-S identity MAC reference arithmetic drift")
    return table
