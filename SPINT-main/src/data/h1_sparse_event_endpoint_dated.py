"""Date-parameterised H-SE5 source training and strict target evaluation.

This is an additive leave-one-date-out implementation.  It deliberately does
not alter the sealed fold-0 H-SE5 module: the only admissible source recordings
are the public held-in recordings whose date differs from ``fold_date`` and a
target data set is constructed only through the explicit evaluation builder.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sua_exploration.mc_maze import h1_sparse_event_endpoint as event_v1  # noqa: E402
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as event_v2  # noqa: E402
from src.data.h1_m4_eb_pilot import (  # noqa: E402
    MAX_TRIAL_LENGTH,
    PilotDataError,
    WINDOW,
    _window_manifest_hash,
    index_heldin_calib,
    interpolate_identity,
    interpolate_trial_identity,
    legal_contiguous_starts,
    load_record,
)


SCHEMA = "h1_sparse_event_endpoint_dated_source_training_v1"
PROTOCOL = "h1_sparse_event_endpoint_q4_ridge3_lodo_m4_date2_v1"
M4_BUDGET = 4
SE5_DIM = 5
FIXED_SEED = 42
FIXED_EPOCHS = 50
FIXED_BATCH_SIZE = 32
NORMALIZER_FLOOR = 1.0e-12


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise PilotDataError(message)


def lodo_source_sessions(fold_date: str) -> tuple[str, ...]:
    _need(fold_date in event_v1.H1_DATES, f"invalid H-SE5 outer date {fold_date!r}")
    values = tuple(name for name in event_v1.H1_HELDIN_SESSIONS if event_v1.session_date(name) != fold_date)
    _need(values and all(event_v1.session_date(name) != fold_date for name in values), "H-SE5 dated source split leaked target date")
    return values


def lodo_target_sessions(fold_date: str) -> tuple[str, ...]:
    _need(fold_date in event_v1.H1_DATES, f"invalid H-SE5 outer date {fold_date!r}")
    values = tuple(name for name in event_v1.H1_HELDIN_SESSIONS if event_v1.session_date(name) == fold_date)
    _need(values and all(event_v1.session_date(name) == fold_date for name in values), "H-SE5 dated target split drift")
    return values


@dataclass(frozen=True)
class DatedSparseCarrierEntry:
    session_name: str
    start_index: int
    trial_values: tuple[float, float, float, float]
    carrier: np.ndarray
    carrier_sha256: str


class DatedSparseCarrierCache:
    def __init__(
        self,
        entries: Iterable[DatedSparseCarrierEntry],
        *,
        basis: event_v2.EndpointBasisV2,
        source_sessions: Sequence[str],
    ) -> None:
        self.entries = tuple(entries)
        self.source_sessions = tuple(source_sessions)
        self._by_key = {(item.session_name, item.start_index): item for item in self.entries}
        self.starts_by_session = {
            name: tuple(item.start_index for item in self.entries if item.session_name == name)
            for name in self.source_sessions
        }
        _need(len(self.entries) == len(self._by_key) and all(self.starts_by_session.values()),
              "dated H-SE5 cache has duplicate or missing legal source supports")
        rows = [
                {"session": item.session_name, "start_index": item.start_index,
                 "trial_values": list(item.trial_values), "carrier_sha256": item.carrier_sha256}
                for item in self.entries
        ]
        # Compatibility serialization is intentionally exercised by the dated
        # code itself.  It yields the sealed fold-0 cache digest; other dates
        # carry explicit date/protocol provenance.
        body = ({"schema": "h1_sparse_event_endpoint_m4_fold0_cache_v1", "basis_sha256": basis.basis_sha256,
                 "carrier_dim": SE5_DIM, "ridge_lambda": event_v2.RIDGE_LAMBDA, "entries": rows}
                if basis.outer_date == "19250101" else
                {"schema": "h1_sparse_event_endpoint_m4_dated_cache_v1", "protocol": PROTOCOL,
                 "outer_date": basis.outer_date, "basis_sha256": basis.basis_sha256, "carrier_dim": SE5_DIM,
                 "ridge_lambda": event_v2.RIDGE_LAMBDA, "source_sessions": list(self.source_sessions), "entries": rows})
        body["cache_sha256"] = event_v1.canonical_sha256(body)
        self.manifest = body

    def get(self, session_name: str, start_index: int) -> DatedSparseCarrierEntry:
        try:
            return self._by_key[(str(session_name), int(start_index))]
        except KeyError as error:
            raise PilotDataError(f"uncached dated H-SE5 source support {session_name}:{start_index}") from error


@dataclass(frozen=True)
class DatedSparseScalarNormalizer:
    s_src: float
    source_cache_sha256: str
    normalizer_sha256: str

    @property
    def denominator(self) -> float:
        return max(float(self.s_src), NORMALIZER_FLOOR)

    def normalize(self, carrier: np.ndarray) -> np.ndarray:
        values = np.asarray(carrier, dtype=np.float64)
        _need(values.shape[-2:] == (event_v1.EXPECTED_NEURONS, SE5_DIM) and np.isfinite(values).all(),
              f"invalid dated H-SE5 carrier {values.shape}")
        return values / self.denominator

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "formula": "s_src=sqrt(mean(source_cache^2)); carrier_norm=carrier/max(s_src,1e-12)",
            "s_src": self.s_src, "denominator": self.denominator,
            "source_cache_sha256": self.source_cache_sha256,
            "normalizer_sha256": self.normalizer_sha256,
        }


def build_dated_sparse_source_assets(
    *, data_dir: str | Path, fold_date: str,
) -> tuple[dict[str, Any], dict[str, event_v1.EventSession], event_v2.EndpointBasisV2, DatedSparseCarrierCache, DatedSparseScalarNormalizer]:
    source_names = lodo_source_sessions(fold_date)
    indexed = index_heldin_calib(data_dir)
    # Indexing paths is permitted; opening any target NWB is not.
    records = {name: load_record(indexed[name]) for name in source_names}
    event_sessions = {name: event_v1.load_event_session(indexed[name]) for name in source_names}
    _need(set(records) == set(event_sessions) == set(source_names), "dated H-SE5 source load split drift")
    basis = event_v2.fit_source_all_event_basis(event_sessions, outer_date=fold_date)
    _need(tuple(basis.source_sessions) == source_names, "dated H-SE5 source basis session order drift")
    entries: list[DatedSparseCarrierEntry] = []
    for name in source_names:
        record, event_session = records[name], event_sessions[name]
        _need(tuple(record.trial_values) == tuple(event_session.trial_values), f"{name}: SPINT/event trial order drift")
        for start in legal_contiguous_starts(record):
            trial_values = tuple(float(value) for value in record.trial_values[start : start + M4_BUDGET])
            carrier, fit = event_v2.fit_session_range(event_session, basis, start_index=start, budget=M4_BUDGET)
            _need(carrier.shape == (event_v1.EXPECTED_NEURONS, SE5_DIM), "dated H-SE5 source carrier shape drift")
            entries.append(DatedSparseCarrierEntry(name, start, trial_values, carrier, str(fit["carrier_sha256"])))
    cache = DatedSparseCarrierCache(entries, basis=basis, source_sessions=source_names)
    stacked = np.stack([item.carrier for item in cache.entries])
    scalar = float(np.sqrt(np.mean(np.square(stacked, dtype=np.float64), dtype=np.float64)))
    _need(np.isfinite(scalar) and scalar > 0.0, "dated H-SE5 source normalizer is undefined")
    normalizer_body = {
        "formula": "s_src=sqrt(mean(source_cache^2)); carrier_norm=carrier/max(s_src,1e-12)",
        "s_src": scalar, "source_cache_sha256": cache.manifest["cache_sha256"], "shape": list(stacked.shape),
    }
    normalizer = DatedSparseScalarNormalizer(scalar, cache.manifest["cache_sha256"], event_v1.canonical_sha256(normalizer_body))
    return records, event_sessions, basis, cache, normalizer


class H1SparseEventDatedSourceDataset(Dataset):
    def __init__(self, records: Mapping[str, Any], cache: DatedSparseCarrierCache, normalizer: DatedSparseScalarNormalizer) -> None:
        self.source_sessions, self.records, self.cache, self.normalizer = cache.source_sessions, dict(records), cache, normalizer
        self.neural_data: dict[str, np.ndarray] = {}; self.covariate_data: dict[str, np.ndarray] = {}; self.eval_mask: dict[str, np.ndarray] = {}
        self.window_indices: list[tuple[str, int]] = []; self._trial_identity: dict[tuple[str, float], np.ndarray] = {}
        prehistory = WINDOW - 1
        for name in self.source_sessions:
            record = self.records[name]
            self.neural_data[name] = np.pad(record.neural, ((prehistory, 0), (0, 0)), constant_values=0.0)
            self.covariate_data[name] = np.pad(record.velocity, ((prehistory, 0), (0, 0)), constant_values=0.0)
            self.eval_mask[name] = np.pad(record.eval_mask, (prehistory, 0), constant_values=False)
            for calibration_start in cache.starts_by_session[name]:
                for value in cache.get(name, calibration_start).trial_values:
                    self._trial_identity.setdefault((name, float(value)), interpolate_trial_identity(record, value))
            self.window_indices.extend((name, start) for start in range(self.neural_data[name].shape[0] - WINDOW + 1)
                                       if self.eval_mask[name][start + WINDOW - 1])
        _need(bool(self.window_indices), "dated H-SE5 source has no eval-valid windows")
        self.window_indices_sha256 = _window_manifest_hash(self.window_indices)

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, request: tuple[int, int]):
        if not isinstance(request, tuple) or len(request) != 2:
            raise PilotDataError("dated H-SE5 source requires scheduled (window, calibration-start) request")
        index, calibration_start = (int(value) for value in request)
        name, start = self.window_indices[index]; end = start + WINDOW; entry = self.cache.get(name, calibration_start)
        _need(tuple(entry.trial_values) == tuple(self.records[name].trial_values[calibration_start : calibration_start + M4_BUDGET]),
              "dated H-SE5 carrier/identity support alignment drift")
        identity = np.stack([self._trial_identity[(name, float(value))] for value in entry.trial_values])
        return (self.neural_data[name][start:end], self.covariate_data[name][start:end], identity, name,
                self.normalizer.normalize(entry.carrier).astype(np.float32))


class H1SparseEventDatedBatchSampler(Sampler[list[tuple[int, int]]]):
    def __init__(self, dataset: H1SparseEventDatedSourceDataset, *, batch_size: int = FIXED_BATCH_SIZE,
                 seed: int = FIXED_SEED, max_epochs: int = FIXED_EPOCHS, cache_dir: str | Path | None = None,
                 frozen_schedule: np.ndarray | None = None) -> None:
        _need((batch_size, seed, max_epochs) == (FIXED_BATCH_SIZE, FIXED_SEED, FIXED_EPOCHS),
              "dated H-SE5 fixes batch=32, seed=42, epochs=50")
        self.dataset, self.source_sessions, self.max_epochs = dataset, dataset.source_sessions, max_epochs
        grouped: dict[str, list[int]] = {name: [] for name in self.source_sessions}
        for index, (name, _start) in enumerate(dataset.window_indices): grouped[name].append(index)
        batches: list[list[int]] = []
        for name in self.source_sessions:
            indices = random.Random(seed).sample(grouped[name], len(grouped[name]))
            batches.extend(indices[offset : offset + batch_size] for offset in range(0, len(indices), batch_size)
                           if len(indices[offset : offset + batch_size]) == batch_size)
        self.batches = random.Random(seed).sample(batches, len(batches))
        self.flat_indices = np.asarray([index for batch in self.batches for index in batch], dtype=np.int64)
        self.batch_order_sha256 = event_v1.array_sha256(self.flat_indices)
        schedule = np.empty((max_epochs, len(self.flat_indices)), dtype=np.int16)
        flat_sessions = np.asarray([dataset.window_indices[int(index)][0] for index in self.flat_indices], dtype=object)
        for name in self.source_sessions:
            positions, legal = np.flatnonzero(flat_sessions == name), np.asarray(dataset.cache.starts_by_session[name], dtype=np.int16)
            token = hashlib.sha256(f"{seed}|m4-schedule|{name}".encode()).digest()
            schedule[:, positions] = legal[np.random.default_rng(int.from_bytes(token[:8], "big")).integers(0, len(legal), size=(max_epochs, len(positions)))]
        if frozen_schedule is not None:
            frozen = np.asarray(frozen_schedule, dtype=np.int16)
            _need(frozen.shape == schedule.shape and np.array_equal(frozen, schedule), "dated H-SE5 frozen schedule mismatches reconstructed legal schedule")
            schedule = frozen.copy()
        self.schedule, self.schedule_sha256, self._epoch = schedule, event_v1.array_sha256(schedule), 0
        if cache_dir is not None:
            self._persist(Path(cache_dir))

    def _persist(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Preserve sealed fold-0 filenames/schema to prove direct parity.
        stem = "fold0_source" if self.source_sessions == tuple(event_v1.H1_HELDIN_SESSIONS[2:]) else "dated_source"
        schedule_path = cache_dir / f"{stem}_calibration_schedule.npy"
        manifest_path = cache_dir / f"{stem}_schedule.manifest.json"
        body = {"schema": "h1_m4_eb_fold0_shared_50epoch_schedule_v1" if stem == "fold0_source" else "h1_sparse_event_endpoint_dated_shared_50epoch_schedule_v1",
                "seed": FIXED_SEED, "epochs": FIXED_EPOCHS, "batch_size": FIXED_BATCH_SIZE,
                "batches_per_epoch": len(self.batches), "scheduled_samples_per_epoch": int(self.flat_indices.size),
                "source_window_indices_sha256": self.dataset.window_indices_sha256, "batch_order_sha256": self.batch_order_sha256,
                "calibration_schedule_sha256": self.schedule_sha256, "carrier_cache_sha256": self.dataset.cache.manifest["cache_sha256"],
                "selection": "one dedicated RNG draw selecting a legal contiguous M=4 start per scheduled sample"}
        if schedule_path.exists() or manifest_path.exists():
            _need(schedule_path.is_file() and manifest_path.is_file(), "partial dated H-SE5 schedule cache exists")
            _need(stat.S_IMODE(schedule_path.stat().st_mode) == 0o444 and stat.S_IMODE(manifest_path.stat().st_mode) == 0o444,
                  "dated H-SE5 schedule cache must be immutable")
            _need(json.loads(manifest_path.read_text()) == body and np.array_equal(np.load(schedule_path, allow_pickle=False), self.schedule),
                  "dated H-SE5 schedule cache drift")
        else:
            np.save(schedule_path, self.schedule, allow_pickle=False)
            manifest_path.write_bytes(event_v1.canonical_json_bytes(body))
            schedule_path.chmod(0o444); manifest_path.chmod(0o444)
        self.manifest = body

    def __iter__(self):
        if self._epoch >= self.max_epochs: raise RuntimeError("dated H-SE5 schedule exhausted after fixed epoch 50")
        row = self.schedule[self._epoch]; offset = 0
        for batch in self.batches:
            starts = row[offset : offset + len(batch)]; offset += len(batch)
            yield [(int(index), int(start)) for index, start in zip(batch, starts)]
        self._epoch += 1

    def __len__(self) -> int:
        return len(self.batches)


class H1SparseEventDatedDataModule(pl.LightningDataModule):
    """Source-only dated H-SE5 DataModule; target access is impossible here."""
    def __init__(self, task: str, data_dir: str, cache_dir: str, fold_date: str, source_snapshot_receipt: str = "", batch_size: int = 32,
                 window_size: int = 700, calibration_n_trials: int = 4, max_trial_length: int = 1024,
                 num_workers: int = 0, pin_memory: bool = False, seed: int = 42, fixed_epochs: int = 50) -> None:
        super().__init__()
        fixed = {"task": str(task).lower() == "h1", "fold_date": str(fold_date) in event_v1.H1_DATES,
                 "batch_size": int(batch_size) == 32, "window_size": int(window_size) == WINDOW,
                 "calibration_n_trials": int(calibration_n_trials) == M4_BUDGET,
                 "max_trial_length": int(max_trial_length) == MAX_TRIAL_LENGTH, "num_workers": int(num_workers) == 0,
                 "seed": int(seed) == FIXED_SEED, "fixed_epochs": int(fixed_epochs) == FIXED_EPOCHS}
        _need(all(fixed.values()), f"dated H-SE5 fixed contract violated: {fixed}")
        self.save_hyperparameters(logger=False); self.batch_size_per_device = 32; self._setup_done = False

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit"}: raise RuntimeError("dated H-SE5 DataModule permits source fit only")
        if self._setup_done: return
        if self.trainer is not None and self.trainer.world_size != 1: raise PilotDataError("dated H-SE5 fixes one GPU/device per arm")
        fold_date = str(self.hparams.fold_date)
        snapshot_receipt = str(self.hparams.source_snapshot_receipt)
        if snapshot_receipt:
            # Authority is exact arrays from CPU-preflight snapshot, not a fresh
            # SVD/cache reconstruction in the independent Full/Zero processes.
            from src.data.h1_sparse_event_source_snapshot_dated import load_snapshot
            snapshot = load_snapshot(snapshot_receipt)
            _need(snapshot.manifest["fold_date"] == fold_date and snapshot.basis.outer_date == fold_date,
                  "dated H-SE5 snapshot fold binding drift")
            indexed = index_heldin_calib(self.hparams.data_dir)
            records = {name: load_record(indexed[name]) for name in snapshot.cache.source_sessions}
            expected_files = list(snapshot.manifest.get("files", ()))
            observed_files = [{"session": name, "nwb_sha256": records[name].input_sha256} for name in snapshot.cache.source_sessions]
            _need(list(snapshot.manifest.get("source_sessions", ())) == list(snapshot.cache.source_sessions)
                  and observed_files == expected_files, "dated H-SE5 snapshot source roster/NWB hash drift")
            event_sessions: dict[str, event_v1.EventSession] = {}
            basis, cache, normalizer = snapshot.basis, snapshot.cache, snapshot.normalizer
        else:
            records, event_sessions, basis, cache, normalizer = build_dated_sparse_source_assets(data_dir=self.hparams.data_dir, fold_date=fold_date)
        dataset = H1SparseEventDatedSourceDataset(records, cache, normalizer)
        sampler = H1SparseEventDatedBatchSampler(dataset, cache_dir=self.hparams.cache_dir,
            frozen_schedule=snapshot.schedule if snapshot_receipt else None)
        targets = lodo_target_sessions(fold_date)
        manifest = (dict(snapshot.manifest) if snapshot_receipt else ({"schema": "h1_sparse_event_endpoint_fold0_source_training_v1", "fold_date": fold_date, "source_sessions": list(cache.source_sessions),
                     "target_sessions_not_opened": list(targets), "files": [{"session": name, "nwb_sha256": records[name].input_sha256} for name in cache.source_sessions],
                     "event_parser_sha256": event_v1.sha256_file(Path(event_v1.__file__)), "event_estimator_sha256": event_v1.sha256_file(Path(event_v2.__file__)),
                     "basis": basis.manifest(), "carrier_cache_sha256": cache.manifest["cache_sha256"],
                     "normalized_cache_sha256": event_v1.canonical_sha256({"cache": cache.manifest["cache_sha256"], "normalizer": normalizer.normalizer_sha256}),
                     "normalizer": normalizer.manifest, "normalizer_sha256": normalizer.normalizer_sha256,
                     "source_window_indices_sha256": dataset.window_indices_sha256, "batch_order_sha256": sampler.batch_order_sha256,
                     "calibration_schedule_sha256": sampler.schedule_sha256, "carrier_dim": SE5_DIM, "calibration_n_trials": M4_BUDGET,
                     "fixed_epochs": FIXED_EPOCHS, "deployment_carrier_dense_velocity_opened": False,
                     "offline_decoder_training_behavior_targets_opened": True, "target_nwb_opened_during_training_setup": False}
                    if fold_date == "19250101" else
                    {"schema": SCHEMA, "protocol": PROTOCOL, "fold_date": fold_date, "source_sessions": list(cache.source_sessions),
                    "target_sessions_not_opened": list(targets), "files": [{"session": name, "nwb_sha256": records[name].input_sha256} for name in cache.source_sessions],
                    "event_parser_sha256": event_v1.sha256_file(Path(event_v1.__file__)), "event_estimator_sha256": event_v1.sha256_file(Path(event_v2.__file__)),
                    "basis": basis.manifest(), "carrier_cache_sha256": cache.manifest["cache_sha256"],
                    "normalized_cache_sha256": event_v1.canonical_sha256({"cache": cache.manifest["cache_sha256"], "normalizer": normalizer.normalizer_sha256}),
                    "normalizer": normalizer.manifest, "normalizer_sha256": normalizer.normalizer_sha256,
                    "source_window_indices_sha256": dataset.window_indices_sha256, "batch_order_sha256": sampler.batch_order_sha256,
                    "calibration_schedule_sha256": sampler.schedule_sha256, "carrier_dim": SE5_DIM,
                    "calibration_n_trials": M4_BUDGET, "fixed_epochs": FIXED_EPOCHS,
                    "target_nwb_opened_during_training_setup": False, "target_session_optimizer_steps": 0,
                    "target_session_backward_steps": 0, "deployment_carrier_dense_velocity_opened": False}))
        if snapshot_receipt:
            _need(manifest["source_window_indices_sha256"] == dataset.window_indices_sha256 and manifest["batch_order_sha256"] == sampler.batch_order_sha256
                  and manifest["calibration_schedule_sha256"] == sampler.schedule_sha256 and manifest["carrier_cache_sha256"] == cache.manifest["cache_sha256"],
                  "dated H-SE5 snapshot/source dataset binding drift")
            manifest = dict(manifest)
            manifest["source_snapshot"] = {"receipt": str(Path(snapshot_receipt).resolve()),
                "receipt_sha256": snapshot.receipt_sha256, "snapshot": str(snapshot.snapshot_path),
                "snapshot_sha256": snapshot.snapshot_sha256, "training_authority": "immutable_source_snapshot"}
        self.records, self.event_sessions, self.basis, self.carrier_cache, self.normalizer = records, event_sessions, basis, cache, normalizer
        self.train_dataset, self.train_batch_sampler, self._manifest = dataset, sampler, manifest
        self._manifest_sha256, self._setup_done = event_v1.canonical_sha256(manifest), True

    def train_dataloader(self):
        if not self._setup_done: raise RuntimeError("call setup('fit') before dated H-SE5 train_dataloader")
        return DataLoader(self.train_dataset, batch_sampler=self.train_batch_sampler, num_workers=0, pin_memory=bool(self.hparams.pin_memory))

    def val_dataloader(self): return []
    def test_dataloader(self): raise RuntimeError("dated H-SE5 target evaluation is isolated")
    def predict_dataloader(self): raise RuntimeError("dated H-SE5 target evaluation is isolated")
    def pilot_manifest(self) -> dict[str, Any]:
        if not self._setup_done: raise RuntimeError("dated H-SE5 DataModule is not set up")
        return dict(self._manifest)
    @property
    def pilot_manifest_sha256(self) -> str:
        if not self._setup_done: raise RuntimeError("dated H-SE5 DataModule is not set up")
        return self._manifest_sha256


@dataclass(frozen=True)
class DatedSparseTargetSupport:
    session_name: str; trial_values: tuple[float, float, float, float]; fifth_trial: float; query_first_bin: int
    identity: np.ndarray; carriers: Mapping[str, np.ndarray]; carrier_sha256: Mapping[str, str]


class H1SparseEventStrictTargetDatasetDated(Dataset):
    INTERVENTIONS = ("full", "zero", "row", "label")
    def __init__(self, records: Mapping[str, Any], event_sessions: Mapping[str, event_v1.EventSession], basis: event_v2.EndpointBasisV2,
                 normalizer: DatedSparseScalarNormalizer, target_sessions: Sequence[str], intervention: str = "full") -> None:
        _need(intervention in self.INTERVENTIONS, f"unknown dated H-SE5 intervention {intervention}")
        self.target_sessions, self.records = tuple(target_sessions), {name: records[name] for name in target_sessions}
        self.event_sessions, self.basis, self.normalizer, self.intervention = {name: event_sessions[name] for name in target_sessions}, basis, normalizer, intervention
        self.support: dict[str, DatedSparseTargetSupport] = {}; self.window_indices: list[tuple[str, int]] = []
        for name in self.target_sessions:
            record, events = self.records[name], self.event_sessions[name]
            _need(tuple(record.trial_values) == tuple(events.trial_values), f"{name}: dated target trial order drift")
            trial_values, fifth = tuple(float(value) for value in record.trial_values[:M4_BUDGET]), float(record.trial_values[M4_BUDGET])
            bins = np.flatnonzero(record.eval_mask & np.isfinite(record.trial_num) & (record.trial_num == fifth)); _need(bins.size > 0, f"{name}: missing fifth trial")
            full_raw, _ = event_v2.fit_session_range(events, basis, start_index=0, budget=M4_BUDGET)
            label_raw, _ = event_v2.fit_session_range(events, basis, start_index=0, budget=M4_BUDGET, shuffled_labels=True)
            row_raw, _ = event_v2.row_shuffle(full_raw, session=name, budget=M4_BUDGET)
            full = normalizer.normalize(full_raw); carriers = {"full": full, "zero": np.zeros_like(full), "row": normalizer.normalize(row_raw), "label": normalizer.normalize(label_raw)}
            _need(all(not np.array_equal(full, carriers[key]) for key in ("zero", "row", "label")), f"{name}: dated H-SE5 control collapsed")
            boundary = int(bins[0]); self.support[name] = DatedSparseTargetSupport(name, trial_values, fifth, boundary, interpolate_identity(record, trial_values), carriers,
                {key: event_v1.array_sha256(np.asarray(value, np.float64)) for key, value in carriers.items()})
            self.window_indices.extend((name, start) for start in range(boundary, record.neural.shape[0] - WINDOW + 1) if record.eval_mask[start + WINDOW - 1])
        _need(bool(self.window_indices), "dated H-SE5 strict target has no post-support query windows")
        self.window_indices_sha256 = _window_manifest_hash(self.window_indices)

    def with_intervention(self, intervention: str) -> "H1SparseEventStrictTargetDatasetDated":
        _need(intervention in self.INTERVENTIONS, f"unknown dated H-SE5 intervention {intervention}")
        clone = object.__new__(type(self)); clone.target_sessions = self.target_sessions; clone.records = self.records; clone.event_sessions = self.event_sessions
        clone.basis = self.basis; clone.normalizer = self.normalizer; clone.intervention = intervention; clone.support = self.support
        clone.window_indices = self.window_indices; clone.window_indices_sha256 = self.window_indices_sha256; return clone
    def __len__(self) -> int: return len(self.window_indices)
    def __getitem__(self, index: int):
        name, start = self.window_indices[int(index)]; end = start + WINDOW; record, support = self.records[name], self.support[name]
        _need(start >= support.query_first_bin and record.eval_mask[end - 1], "dated H-SE5 query boundary violation")
        return record.neural[start:end], record.velocity[start:end], support.identity, name, np.asarray(support.carriers[self.intervention], np.float32)
    def support_and_carrier_hashes(self) -> dict[str, Any]:
        return {name: {"trial_values": list(item.trial_values), "fifth_trial": item.fifth_trial, "query_first_bin": item.query_first_bin,
                       "carrier_sha256": dict(item.carrier_sha256)} for name, item in self.support.items()}


def build_dated_sparse_target_dataset(*, data_dir: str | Path, source_module: H1SparseEventDatedDataModule) -> H1SparseEventStrictTargetDatasetDated:
    _need(source_module._setup_done, "dated H-SE5 source module must be set up before target access")
    fold_date = str(source_module.hparams.fold_date); target_names = lodo_target_sessions(fold_date); indexed = index_heldin_calib(data_dir)
    records = {name: load_record(indexed[name]) for name in target_names}
    sessions = {name: event_v1.load_event_session(indexed[name]) for name in target_names}
    return H1SparseEventStrictTargetDatasetDated(records, sessions, source_module.basis, source_module.normalizer, target_names)
