"""Source-only Phase-2 data binding for H1 CarrierID date-LODO.

This module is intentionally separate from both the legacy fold-0 CarrierID
DataModule and the CCE DataModule.  It consumes, rather than recreates, a
single immutable Phase-1 source bundle for one confirmatory outer date.  The
same binding is usable by the paired H-S (SPINT identity) and H-C (4-D
CarrierID) source-training arms.

No target dataset, target loader, validation loader, formal evaluator, or
checkpoint restore path is implemented here.  The only public NWBs opened are
the exact source sessions already named in the Phase-1 receipt.  Outer-date
files are indexed only as filename metadata by the shared Phase-1 loader.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler

from src.data.h1_carrierid_date_lodo_source import (
    CarrierCache,
    CarrierCacheEntry,
    CarrierIdDateLodoSourceError,
    SourceRmsNormalizer,
    SourceSchedule,
    build_source_window_index,
    load_source_records_with_target_filename_index,
    source_sessions_for_date,
    target_sessions_for_date,
    validate_source_bundle_manifest,
)
from src.data.h1_m4_eb_pilot import (
    EXPECTED_NEURONS,
    H1PilotRecord,
    MAX_TRIAL_LENGTH,
    SUPPORT_TRIALS,
    interpolate_trial_identity,
)
from src.h1_m4_cce_contract import (
    CONFIRMATORY_DATES,
    FIXED_EPOCHS,
    FIXED_SEED,
    NORMALIZER_FLOOR,
    WINDOW_SIZE,
    array_sha256,
    canonical_sha256,
    sha256_file,
)


PHASE1_PREFLIGHT_SCHEMA = "h1_carrierid_date_lodo_source_cpu_preflight_v1"
PHASE1_PREFLIGHT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_SOURCE_PREFLIGHT_NOT_LAUNCHED"
PHASE2_SOURCE_BINDING_SCHEMA = "h1_carrierid_date_lodo_phase2_source_binding_v1"


class CarrierIdDateLodoPhase2Error(ValueError):
    """Fail-closed violation of a Phase-2 source-only invariant."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise CarrierIdDateLodoPhase2Error(message)


def _immutable(path: Path) -> bool:
    return path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o444


def _read_immutable_json(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    candidate = Path(path).resolve()
    _need(_immutable(candidate), f"Phase-2 requires immutable mode-0444 JSON: {candidate}")
    try:
        body = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CarrierIdDateLodoPhase2Error(f"invalid immutable JSON: {candidate}") from error
    _need(isinstance(body, dict), f"immutable JSON is not an object: {candidate}")
    return candidate, body, sha256_file(candidate)


def _read_immutable_carriers(path: Path) -> np.ndarray:
    _need(_immutable(path), f"Phase-2 requires immutable carrier cache: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            _need(tuple(archive.files) == ("carriers",), f"unexpected carrier-cache arrays: {archive.files}")
            carriers = np.asarray(archive["carriers"], dtype=np.float64)
    except (OSError, ValueError) as error:
        raise CarrierIdDateLodoPhase2Error(f"invalid carrier-cache archive: {path}") from error
    _need(
        carriers.ndim == 3 and carriers.shape[1:] == (EXPECTED_NEURONS, 4) and carriers.shape[0] > 0,
        f"carrier-cache shape drift: {carriers.shape}",
    )
    _need(np.isfinite(carriers).all(), "carrier-cache contains nonfinite values")
    return carriers


def _read_immutable_schedule(path: Path) -> np.ndarray:
    _need(_immutable(path), f"Phase-2 requires immutable schedule: {path}")
    try:
        schedule = np.asarray(np.load(path, allow_pickle=False), dtype=np.int16)
    except (OSError, ValueError) as error:
        raise CarrierIdDateLodoPhase2Error(f"invalid schedule array: {path}") from error
    _need(schedule.ndim == 2 and schedule.shape[0] == FIXED_EPOCHS and schedule.shape[1] > 0,
          f"Phase-1 schedule shape drift: {schedule.shape}")
    return schedule


def _expected_normalizer_digest(body: Mapping[str, Any]) -> str:
    canonical = {
        "schema": body.get("schema"),
        "formula": body.get("formula"),
        "floor": body.get("floor"),
        "source_cache_sha256": body.get("source_cache_sha256"),
        "entries": body.get("entries"),
        "rows": body.get("rows"),
        "dims": body.get("dims"),
        "s_src": body.get("s_src"),
    }
    return canonical_sha256(canonical)


@dataclass(frozen=True)
class Phase2SourceBinding:
    """Fully verified, immutable source state shared by H-S and H-C."""

    outer_date: str
    preflight_path: Path
    preflight_sha256: str
    source_manifest_path: Path
    source_manifest_sha256: str
    records: Mapping[str, H1PilotRecord]
    cache: CarrierCache
    normalizer: SourceRmsNormalizer
    source_windows: tuple[tuple[str, int], ...]
    source_window_indices_sha256: str
    batch_order: np.ndarray
    batch_order_sha256: str
    calibration_schedule: np.ndarray
    calibration_schedule_sha256: str
    target_filename_index: Mapping[str, Any]

    @property
    def source_sessions(self) -> tuple[str, ...]:
        return tuple(self.cache.source_sessions)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": PHASE2_SOURCE_BINDING_SCHEMA,
            "outer_date": self.outer_date,
            "preflight_path": str(self.preflight_path),
            "preflight_sha256": self.preflight_sha256,
            "source_manifest_path": str(self.source_manifest_path),
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_sessions": list(self.source_sessions),
            "source_files": [
                {"session": name, "sha256": self.records[name].input_sha256}
                for name in self.source_sessions
            ],
            "source_cache_sha256": self.cache.manifest["cache_sha256"],
            "normalizer_sha256": self.normalizer.normalizer_sha256,
            "normalized_cache_sha256": self.cache.manifest["normalized_cache_sha256"],
            "source_window_indices_sha256": self.source_window_indices_sha256,
            "batch_order_sha256": self.batch_order_sha256,
            "calibration_schedule_sha256": self.calibration_schedule_sha256,
            "epochs": FIXED_EPOCHS,
            "batch_size": 32,
            "seed": FIXED_SEED,
            "target_recordings_opened": 0,
            "target_bytes_read": 0,
            "target_filenames_indexed_only": list(self.target_filename_index["target_filenames_indexed_only"]),
            "warm_start_forbidden": True,
        }


def _load_phase1_cache(source_manifest: Mapping[str, Any], directory: Path) -> tuple[CarrierCache, SourceRmsNormalizer]:
    """Read all Phase-1 cache/normalizer artifacts and verify their linkage."""

    cache_row = source_manifest.get("carrier_cache")
    normalizer_row = source_manifest.get("normalizer")
    normalized_row = source_manifest.get("normalized_cache")
    _need(isinstance(cache_row, Mapping) and isinstance(normalizer_row, Mapping) and isinstance(normalized_row, Mapping),
          "source manifest lacks Phase-1 cache/normalizer rows")
    cache_manifest_path, cache_manifest, cache_manifest_sha = _read_immutable_json(cache_row.get("manifest_path", ""))
    _need(cache_manifest_path.parent == directory, "carrier cache manifest escapes its date bundle")
    _need(cache_manifest_sha == cache_row.get("manifest_sha256"), "carrier-cache manifest SHA drift")
    core = dict(cache_manifest)
    claimed_cache_sha = str(core.pop("cache_sha256", ""))
    _need(canonical_sha256(core) == claimed_cache_sha, "carrier-cache canonical SHA drift")
    _need(claimed_cache_sha == cache_row.get("cache_sha256"), "source manifest/cache SHA disagreement")
    raw_path = directory / "source_m4_carriers.npz"
    raw = _read_immutable_carriers(raw_path)
    # Phase-1's raw-cache manifest predates file-SHA publication.  Its
    # canonical cache SHA binds every entry's tensor SHA instead; verify that
    # stronger per-entry linkage below rather than inventing a missing field.
    rows = cache_manifest.get("entries")
    _need(isinstance(rows, list) and len(rows) == raw.shape[0], "carrier-cache entry/count drift")
    entries: list[CarrierCacheEntry] = []
    for index, row in enumerate(rows):
        _need(isinstance(row, Mapping), "carrier-cache entry is not an object")
        values = tuple(float(value) for value in row.get("trial_values", ()))
        _need(len(values) == SUPPORT_TRIALS, "carrier-cache support must be M=4")
        entry = CarrierCacheEntry(
            str(row.get("session")), int(row.get("start_index")), values, raw[index], str(row.get("carrier_sha256")),
        )
        _need(entry.carrier_sha256 == __import__("hashlib").sha256(np.ascontiguousarray(entry.carrier).tobytes()).hexdigest(),
              "carrier-cache entry tensor SHA drift")
        entries.append(entry)
    source_sessions = tuple(source_manifest.get("source_sessions", ()))
    cache = CarrierCache(entries, {**cache_manifest, "manifest_file_sha256": cache_manifest_sha}, source_sessions)

    normalizer_path, normalizer_body, normalizer_file_sha = _read_immutable_json(normalizer_row.get("manifest_path", ""))
    _need(normalizer_path.parent == directory, "normalizer manifest escapes its date bundle")
    _need(normalizer_file_sha == normalizer_row.get("manifest_file_sha256"), "normalizer file SHA drift")
    _need(normalizer_body.get("normalizer_sha256") == normalizer_row.get("normalizer_sha256"), "normalizer SHA disagreement")
    _need(_expected_normalizer_digest(normalizer_body) == normalizer_body.get("normalizer_sha256"), "normalizer digest drift")
    normalizer = SourceRmsNormalizer(
        s_src=float(normalizer_body["s_src"]), source_cache_sha256=str(normalizer_body["source_cache_sha256"]),
        entries=int(normalizer_body["entries"]), rows=int(normalizer_body["rows"]), dims=int(normalizer_body["dims"]),
        normalizer_sha256=str(normalizer_body["normalizer_sha256"]),
    )
    _need(normalizer.source_cache_sha256 == claimed_cache_sha, "normalizer uses another source cache")
    _need(normalizer.entries == len(entries) and normalizer.rows == EXPECTED_NEURONS and normalizer.dims == 4,
          "normalizer shape/count drift")
    _need(math.isfinite(normalizer.s_src) and normalizer.s_src >= 0.0, "normalizer scalar is invalid")

    normalized_manifest_path, normalized_manifest, normalized_manifest_sha = _read_immutable_json(
        normalized_row.get("manifest_path", "")
    )
    _need(normalized_manifest_path.parent == directory, "normalized-cache manifest escapes its date bundle")
    _need(normalized_manifest_sha == normalized_row.get("manifest_sha256"), "normalized-cache manifest SHA drift")
    _need(normalized_manifest.get("normalizer_sha256") == normalizer.normalizer_sha256,
          "normalized cache uses a different source normalizer")
    _need(normalized_manifest.get("source_cache_sha256") == claimed_cache_sha,
          "normalized cache uses a different source carrier cache")
    normalized_path = directory / "normalized_source_m4_carriers.npz"
    normalized = _read_immutable_carriers(normalized_path)
    _need(sha256_file(normalized_path) == normalized_manifest.get("arrays_file_sha256"),
          "normalized carrier-cache file SHA drift")
    _need(normalized.shape == raw.shape and np.array_equal(normalized, normalizer.normalize(raw)),
          "normalized cache no longer equals Phase-1 source scalar transform")
    _need(array_sha256(normalized) == normalized_row.get("normalized_carriers_sha256") == normalized_manifest.get("normalized_carriers_sha256"),
          "normalized cache tensor SHA drift")
    cache.manifest["normalized_cache_sha256"] = str(normalized_row["normalized_carriers_sha256"])
    return cache, normalizer


def load_phase2_source_binding(*, data_dir: str | Path, phase1_preflight_path: str | Path, outer_date: str) -> Phase2SourceBinding:
    """Open exactly one Phase-1 source bundle and its declared source NWBs."""

    date = str(outer_date)
    _need(date in CONFIRMATORY_DATES, "Phase-2 outer date must be a confirmatory date, never fold0")
    preflight_path, preflight, preflight_sha = _read_immutable_json(phase1_preflight_path)
    _need(preflight.get("schema") == PHASE1_PREFLIGHT_SCHEMA and preflight.get("status") == PHASE1_PREFLIGHT_STATUS,
          "Phase-2 requires a passed H1 CarrierID date-LODO Phase-1 preflight")
    _need(tuple(preflight.get("confirmatory_dates", ())) == CONFIRMATORY_DATES,
          "Phase-1 receipt is not canonical five-date LODO")
    rows = preflight.get("date_bundles")
    _need(isinstance(rows, Mapping) and set(rows) == set(CONFIRMATORY_DATES), "Phase-1 receipt date rows drift")
    summary = rows[date]
    _need(isinstance(summary, Mapping) and summary.get("outer_date") == date, "date source summary mismatch")
    source_manifest_path, source_manifest, source_manifest_sha = _read_immutable_json(summary.get("source_manifest_path", ""))
    _need(source_manifest_sha == summary.get("source_manifest_sha256"), "Phase-1 source-manifest SHA drift")
    validate_source_bundle_manifest(source_manifest, outer_date=date)
    _need(source_manifest_path.parent.name == date, "source manifest date-directory mismatch")
    expected_sources, expected_targets = source_sessions_for_date(date), target_sessions_for_date(date)
    _need(tuple(source_manifest["source_sessions"]) == expected_sources, "Phase-1 source partition drift")
    target_index = source_manifest["target_filename_index"]
    _need(tuple(target_index["target_sessions_indexed"]) == expected_targets, "Phase-1 target filename partition drift")
    _need(target_index["target_recordings_opened"] == 0 and target_index["target_bytes_read"] == 0,
          "Phase-1 receipt records target access")
    cache, normalizer = _load_phase1_cache(source_manifest, source_manifest_path.parent)

    records, audit = load_source_records_with_target_filename_index(data_dir, date)
    _need(tuple(records) == expected_sources and tuple(audit.source_sessions_opened) == expected_sources,
          "Phase-2 source record loader partition/order drift")
    _need(tuple(audit.target_sessions_indexed) == expected_targets and audit.manifest()["target_recordings_opened"] == 0,
          "Phase-2 target access boundary drift")
    file_rows = source_manifest.get("source_files")
    _need(isinstance(file_rows, list) and len(file_rows) == len(expected_sources), "source file receipt drift")
    expected_hashes = {str(row["session"]): str(row["sha256"]) for row in file_rows}
    _need(set(expected_hashes) == set(expected_sources), "source file receipt sessions drift")
    _need(all(records[name].input_sha256 == expected_hashes[name] for name in expected_sources),
          "source NWB hash differs from immutable Phase-1 receipt")

    windows = build_source_window_index(records, cache)
    binding = source_manifest["arms"]["H-S"]["shared_source_binding"]
    _need(binding == source_manifest["arms"]["H-C"]["shared_source_binding"], "H-S/H-C source binding diverged")
    _need(windows.window_indices_sha256 == binding["source_window_indices_sha256"], "source window index drift")
    expected_schedule = SourceSchedule(windows, date)
    schedule_manifest_path, schedule_manifest, schedule_manifest_sha = _read_immutable_json(
        source_manifest["schedule"]["manifest_path"]
    )
    _need(schedule_manifest_path.parent == source_manifest_path.parent, "schedule manifest escapes date bundle")
    _need(schedule_manifest_sha == source_manifest["schedule"]["manifest_sha256"], "schedule manifest SHA drift")
    schedule_path = source_manifest_path.parent / "source_calibration_schedule.npy"
    stored_schedule = _read_immutable_schedule(schedule_path)
    # As with the raw carrier cache, Phase-1's schedule manifest binds the
    # immutable tensor by array SHA, not by a redundant byte-file SHA.
    _need(np.array_equal(stored_schedule, expected_schedule.schedule), "regenerated source schedule differs from immutable Phase-1 schedule")
    _need(array_sha256(stored_schedule) == binding["calibration_schedule_sha256"], "schedule tensor SHA drift")
    _need(expected_schedule.batch_order_sha256 == binding["batch_order_sha256"], "batch-order SHA drift")
    _need(np.all(np.isin(stored_schedule, np.asarray([entry.start_index for entry in cache.entries], dtype=np.int16))),
          "schedule contains a calibration start absent from raw Phase-1 cache")
    return Phase2SourceBinding(
        outer_date=date, preflight_path=preflight_path, preflight_sha256=preflight_sha,
        source_manifest_path=source_manifest_path, source_manifest_sha256=source_manifest_sha,
        records=records, cache=cache, normalizer=normalizer,
        source_windows=windows.window_indices, source_window_indices_sha256=windows.window_indices_sha256,
        batch_order=expected_schedule.flat_indices, batch_order_sha256=expected_schedule.batch_order_sha256,
        calibration_schedule=stored_schedule, calibration_schedule_sha256=array_sha256(stored_schedule),
        target_filename_index=audit.manifest(),
    )


class H1CarrierIdDateLodoSourceDataset(Dataset):
    """Source-only [neural,target,identity,session,normalized-carrier] records."""

    def __init__(self, binding: Phase2SourceBinding) -> None:
        self.binding = binding
        self.records = {name: binding.records[name] for name in binding.source_sessions}
        self.cache, self.normalizer = binding.cache, binding.normalizer
        self.window_indices = binding.source_windows
        self.neural_data: dict[str, np.ndarray] = {}
        self.target_data: dict[str, np.ndarray] = {}
        self._trial_identity: dict[tuple[str, float], np.ndarray] = {}
        prehistory = WINDOW_SIZE - 1
        for name in binding.source_sessions:
            record = self.records[name]
            self.neural_data[name] = np.pad(record.neural, ((prehistory, 0), (0, 0)), constant_values=0.0)
            self.target_data[name] = np.pad(record.velocity, ((prehistory, 0), (0, 0)), constant_values=0.0)
            for support_start in self.cache.starts_by_session[name]:
                for value in self.cache.get(name, support_start).trial_values:
                    self._trial_identity.setdefault((name, float(value)), interpolate_trial_identity(record, value))
        _need(len(self.window_indices) > 0, "source-only dataset has no valid windows")

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, request: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
        _need(isinstance(request, tuple) and len(request) == 2, "source samples need declared (window,start) schedule")
        index, calibration_start = int(request[0]), int(request[1])
        session, start = self.window_indices[index]
        entry = self.cache.get(session, calibration_start)
        expected_values = tuple(self.records[session].trial_values[calibration_start : calibration_start + SUPPORT_TRIALS])
        _need(entry.trial_values == expected_values, "Phase-1 cache and source identity trial values drift")
        identity = np.stack([self._trial_identity[(session, value)] for value in entry.trial_values], axis=0)
        return (
            self.neural_data[session][start : start + WINDOW_SIZE],
            self.target_data[session][start : start + WINDOW_SIZE],
            identity,
            session,
            self.normalizer.normalize(entry.carrier).astype(np.float32),
        )


class H1CarrierIdDateLodoSchedule(Sampler[list[tuple[int, int]]]):
    """Use the immutable Phase-1 M=4 starts with its verified batch order."""

    def __init__(self, dataset: H1CarrierIdDateLodoSourceDataset, binding: Phase2SourceBinding) -> None:
        self.dataset, self.binding, self._epoch = dataset, binding, 0
        _need(binding.batch_order.ndim == 1 and binding.batch_order.size % 32 == 0, "batch order is not complete batches")
        _need(binding.calibration_schedule.shape == (FIXED_EPOCHS, binding.batch_order.size),
              "immutable schedule/batch-order shape mismatch")
        self.batches = tuple(
            tuple(int(index) for index in binding.batch_order[offset : offset + 32])
            for offset in range(0, binding.batch_order.size, 32)
        )
        _need(all(len(batch) == 32 for batch in self.batches), "schedule contains incomplete source batch")
        _need(all(len({dataset.window_indices[index][0] for index in batch}) == 1 for batch in self.batches),
              "shared Phase-1 batch order mixes source sessions")

    def __len__(self) -> int:
        return len(self.batches)

    def reset_epoch(self) -> None:
        self._epoch = 0

    def __iter__(self):
        if self._epoch >= FIXED_EPOCHS:
            raise RuntimeError("fixed Phase-1 source schedule exhausted after epoch 49")
        starts = self.binding.calibration_schedule[self._epoch]
        offset = 0
        for batch in self.batches:
            selected = starts[offset : offset + len(batch)]
            offset += len(batch)
            yield [(index, int(start)) for index, start in zip(batch, selected)]
        self._epoch += 1


class H1CarrierIdDateLodoSourceDataModule(pl.LightningDataModule):
    """One date's read-only Phase-1 source binding, shared by H-S/H-C."""

    def __init__(
        self, *, task: str, data_dir: str, phase1_preflight_path: str, outer_date: str,
        batch_size: int = 32, window_size: int = WINDOW_SIZE, calibration_n_trials: int = SUPPORT_TRIALS,
        max_trial_length: int = MAX_TRIAL_LENGTH, seed: int = FIXED_SEED, fixed_epochs: int = FIXED_EPOCHS,
        num_workers: int = 0, pin_memory: bool = False,
    ) -> None:
        super().__init__()
        fixed = {
            "task": str(task).lower() == "h1", "outer_date": str(outer_date) in CONFIRMATORY_DATES,
            "batch_size": int(batch_size) == 32, "window_size": int(window_size) == WINDOW_SIZE,
            "calibration_n_trials": int(calibration_n_trials) == SUPPORT_TRIALS,
            "max_trial_length": int(max_trial_length) == MAX_TRIAL_LENGTH, "seed": int(seed) == FIXED_SEED,
            "fixed_epochs": int(fixed_epochs) == FIXED_EPOCHS, "num_workers": int(num_workers) == 0,
        }
        _need(all(fixed.values()), f"Phase-2 fixed source data contract violated: {fixed}")
        self.save_hyperparameters(logger=False)
        self._setup_done = False

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit"}:
            raise RuntimeError("Phase-2 source DataModule permits fit only; target evaluation is not implemented")
        if self._setup_done:
            return
        if self.trainer is not None and self.trainer.world_size != 1:
            raise CarrierIdDateLodoPhase2Error("Phase-2 fixes one device and one immutable source schedule")
        binding = load_phase2_source_binding(
            data_dir=self.hparams.data_dir, phase1_preflight_path=self.hparams.phase1_preflight_path,
            outer_date=self.hparams.outer_date,
        )
        dataset = H1CarrierIdDateLodoSourceDataset(binding)
        sampler = H1CarrierIdDateLodoSchedule(dataset, binding)
        self.binding, self.train_dataset, self.train_batch_sampler = binding, dataset, sampler
        self._setup_done = True

    def train_dataloader(self) -> DataLoader:
        if not self._setup_done:
            raise RuntimeError("call setup('fit') before Phase-2 train_dataloader")
        return DataLoader(self.train_dataset, batch_sampler=self.train_batch_sampler, num_workers=0,
                          pin_memory=bool(self.hparams.pin_memory))

    def val_dataloader(self) -> list[Any]:
        return []

    def test_dataloader(self):
        raise RuntimeError("Phase-2 target evaluation is intentionally not implemented")

    def predict_dataloader(self):
        raise RuntimeError("Phase-2 target prediction is intentionally not implemented")

    @property
    def phase1_manifest_sha256(self) -> str:
        if not self._setup_done:
            raise RuntimeError("Phase-2 source binding is not set up")
        return self.binding.source_manifest_sha256

    def phase2_source_manifest(self) -> dict[str, Any]:
        if not self._setup_done:
            raise RuntimeError("Phase-2 source binding is not set up")
        return self.binding.manifest()
