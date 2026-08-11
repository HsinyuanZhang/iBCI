"""Source-only five-date preparation for H1 CarrierID.

This is deliberately a *data/provenance* layer, not a model, trainer, or
target evaluator.  It reconstructs the frozen M=4 functional carrier for one
of the five confirmatory H1 dates, computes a source-RMS normalizer, and
persists one shared source schedule.  Future H-S and H-C training wrappers
must consume the same per-date manifest; they may not independently recreate
or tune a carrier cache.

The public outer-date files are indexed only to record their filenames.  Their
bytes are never read here: ``load_record`` is invoked solely for the exact
non-outer-date source session list.  The CCE date partition and frozen
estimator helpers are reused as data semantics only.  No CCE residual model is
imported or constructed by this module.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import random
import stat
import uuid
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.data.h1_m4_cce_date_lodo import (
    reconstruct_plan_for_date,
    source_sessions_for_date,
    target_sessions_for_date,
)
from src.data.h1_m4_eb_pilot import (
    EXPECTED_NEURONS,
    H1PilotRecord,
    array_sha256,
    carrier_sha256,
    fit_frozen_carrier,
    index_heldin_calib,
    load_record,
)
from src.h1_m4_cce_contract import (
    CONFIRMATORY_DATES,
    FIXED_EPOCHS,
    FIXED_SEED,
    NORMALIZER_FLOOR,
    NORMALIZER_FORMULA,
    SUPPORT_TRIALS,
    WINDOW_SIZE,
    assert_confirmatory_date,
    canonical_sha256,
    reject_nonpublic_heldin_scope,
    sha256_file,
)


SOURCE_PLAN_SCHEMA = "h1_carrierid_date_lodo_frozen_m4_plan_v1"
SOURCE_CACHE_SCHEMA = "h1_carrierid_date_lodo_source_carrier_cache_v1"
SOURCE_NORMALIZER_SCHEMA = "h1_carrierid_date_lodo_source_rms_normalizer_v1"
SOURCE_NORMALIZED_CACHE_SCHEMA = "h1_carrierid_date_lodo_normalized_source_carrier_cache_v1"
SOURCE_SCHEDULE_SCHEMA = "h1_carrierid_date_lodo_shared_source_schedule_v1"
SOURCE_MANIFEST_SCHEMA = "h1_carrierid_date_lodo_shared_source_manifest_v1"


class CarrierIdDateLodoSourceError(ValueError):
    """Fail-closed source-only CarrierID date-LODO violation."""


def _immutable(path: Path) -> bool:
    return path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o444


def _publish_bytes_once(path: Path, payload: bytes) -> str:
    """Atomically publish one immutable artifact; never overwrite or resume."""

    output = path.resolve()
    if output.exists():
        raise FileExistsError(f"CarrierID date-LODO refuses existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(f"CarrierID date-LODO artifact collision: {output}") from error
    finally:
        if temporary.exists():
            temporary.unlink()
    if not _immutable(output):
        raise CarrierIdDateLodoSourceError(f"artifact was not immutable mode 0444: {output}")
    return sha256_file(output)


def _publish_json_once(path: Path, body: Mapping[str, Any]) -> str:
    return _publish_bytes_once(
        path, (json.dumps(dict(body), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    )


def _publish_npy_once(path: Path, array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return _publish_bytes_once(path, buffer.getvalue())


def _publish_npz_once(path: Path, **arrays: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return _publish_bytes_once(path, buffer.getvalue())


@dataclass(frozen=True)
class SourceLoadAudit:
    """File-level evidence that target files were indexed but never opened."""

    outer_date: str
    source_sessions_opened: tuple[str, ...]
    target_sessions_indexed: tuple[str, ...]
    target_filenames: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "outer_date": self.outer_date,
            "source_recordings_opened": len(self.source_sessions_opened),
            "source_sessions_opened": list(self.source_sessions_opened),
            "target_recordings_opened": 0,
            "target_sessions_indexed": list(self.target_sessions_indexed),
            "target_filenames_indexed_only": list(self.target_filenames),
            "target_bytes_read": 0,
        }


def load_source_records_with_target_filename_index(
    data_dir: str | Path, outer_date: str, *, record_loader=load_record
) -> tuple[dict[str, H1PilotRecord], SourceLoadAudit]:
    """Load only non-outer source NWBs while retaining outer filenames.

    ``index_heldin_calib`` only globs filenames and validates the fixed public
    held-in directory.  The only byte-reading operation below is the source
    loop.  The injectable loader exists solely for no-data contract tests.
    """

    date = assert_confirmatory_date(outer_date)
    reject_nonpublic_heldin_scope(data_dir)
    paths = index_heldin_calib(data_dir)
    source = tuple(source_sessions_for_date(date))
    target = tuple(target_sessions_for_date(date))
    if set(source) & set(target) or tuple(sorted((*source, *target))) != tuple(sorted(paths)):
        raise CarrierIdDateLodoSourceError("date-LODO source/target partition is not exact")
    target_filenames = tuple(paths[name].name for name in target)
    records: dict[str, H1PilotRecord] = {}
    opened: list[str] = []
    for name in source:
        records[name] = record_loader(paths[name])
        opened.append(name)
    if tuple(records) != source or tuple(opened) != source:
        raise CarrierIdDateLodoSourceError("source loader order differs from date-LODO partition")
    if any(record.date == date for record in records.values()):
        raise CarrierIdDateLodoSourceError("outer-date target leaked into source record loader")
    return records, SourceLoadAudit(date, tuple(opened), target, target_filenames)


@dataclass(frozen=True)
class CarrierCacheEntry:
    session_name: str
    start_index: int
    trial_values: tuple[float, float, float, float]
    carrier: np.ndarray
    carrier_sha256: str


class CarrierCache:
    """Shared date-specific raw carrier cache, independent of either arm."""

    def __init__(self, entries: Iterable[CarrierCacheEntry], manifest: Mapping[str, Any], source_sessions: Sequence[str]):
        self.entries = tuple(entries)
        self.manifest = dict(manifest)
        self.source_sessions = tuple(source_sessions)
        self._by_key = {(item.session_name, item.start_index): item for item in self.entries}
        self.starts_by_session = {
            name: tuple(item.start_index for item in self.entries if item.session_name == name)
            for name in self.source_sessions
        }
        if not self.entries or len(self._by_key) != len(self.entries):
            raise CarrierIdDateLodoSourceError("source carrier cache has duplicate or no entries")
        if any(not self.starts_by_session[name] for name in self.source_sessions):
            raise CarrierIdDateLodoSourceError("source carrier cache omits a source session")

    def get(self, session_name: str, start_index: int) -> CarrierCacheEntry:
        try:
            return self._by_key[(str(session_name), int(start_index))]
        except KeyError as error:
            raise CarrierIdDateLodoSourceError(
                f"uncached source support {session_name}:{start_index}"
            ) from error


def _legal_contiguous_starts(record: H1PilotRecord) -> tuple[int, ...]:
    starts: list[int] = []
    for start in range(len(record.trial_values) - SUPPORT_TRIALS + 1):
        values = record.trial_values[start : start + SUPPORT_TRIALS]
        if all(record.blocks_for(value).rates.shape[0] >= 2 for value in values):
            for value in values:
                record.eval_trial_neural(value)
            starts.append(start)
    if not starts:
        raise CarrierIdDateLodoSourceError(f"{record.session_name}: no legal contiguous source M=4 support")
    return tuple(starts)


def _plan_manifest(plan: Any) -> dict[str, Any]:
    arrays = {"mean": plan.mean, "scale": plan.scale, "pcs": plan.pcs, "U": plan.U, "mu": plan.mu}
    return {
        "schema": SOURCE_PLAN_SCHEMA,
        "outer_date": plan.outer_date,
        "source_sessions": list(plan.source_sessions),
        "source_input_sha256": list(plan.source_input_sha256),
        "q": int(plan.q),
        "lambda": float(plan.ridge_lambda),
        "tau2": float(plan.tau2),
        "raw_plan_sha256": plan.raw_plan_sha256,
        "raw_receipt_sha256": plan.raw_receipt_sha256,
        "eb_receipt_sha256": plan.eb_receipt_sha256,
        "array_sha256": {name: array_sha256(value) for name, value in arrays.items()},
        "array_shape": {name: list(value.shape) for name, value in arrays.items()},
        "transform_sha256": plan.transform_sha256,
    }


def _persist_plan(plan: Any, directory: Path) -> dict[str, Any]:
    manifest = _plan_manifest(plan)
    arrays_sha = _publish_npz_once(
        directory / "frozen_m4_plan.npz",
        mean=plan.mean,
        scale=plan.scale,
        pcs=plan.pcs,
        q=np.asarray(plan.q, dtype=np.int64),
        **{"lambda": np.asarray(plan.ridge_lambda, dtype=np.float64)},
        U=plan.U,
        mu=plan.mu,
        tau2=np.asarray(plan.tau2, dtype=np.float64),
    )
    manifest_sha = _publish_json_once(directory / "frozen_m4_plan.manifest.json", manifest)
    return {**manifest, "arrays_file_sha256": arrays_sha, "manifest_file_sha256": manifest_sha}


def _build_carrier_cache(records: Mapping[str, H1PilotRecord], plan: Any, directory: Path) -> CarrierCache:
    rows: list[dict[str, Any]] = []
    entries: list[CarrierCacheEntry] = []
    values: list[np.ndarray] = []
    for session in plan.source_sessions:
        record = records[session]
        for start in _legal_contiguous_starts(record):
            trial_values = tuple(float(value) for value in record.trial_values[start : start + SUPPORT_TRIALS])
            carrier = np.asarray(fit_frozen_carrier(record, plan, trial_values)["carrier"], dtype=np.float64)
            digest = carrier_sha256(carrier)
            rows.append({
                "session": session,
                "start_index": start,
                "trial_values": list(trial_values),
                "carrier_sha256": digest,
            })
            entries.append(CarrierCacheEntry(session, start, trial_values, carrier, digest))
            values.append(carrier)
    stacked = np.stack(values, axis=0)
    if stacked.ndim != 3 or stacked.shape[1:] != (EXPECTED_NEURONS, 4) or not np.isfinite(stacked).all():
        raise CarrierIdDateLodoSourceError(f"source carrier cache shape/finite violation: {stacked.shape}")
    manifest = {
        "schema": SOURCE_CACHE_SCHEMA,
        "outer_date": plan.outer_date,
        "source_sessions": list(plan.source_sessions),
        "transform_sha256": plan.transform_sha256,
        "carrier_dtype": str(stacked.dtype),
        "carrier_shape": list(stacked.shape),
        "entries": rows,
    }
    manifest["cache_sha256"] = canonical_sha256(manifest)
    arrays_sha = _publish_npz_once(directory / "source_m4_carriers.npz", carriers=stacked)
    manifest_sha = _publish_json_once(directory / "source_m4_carriers.manifest.json", manifest)
    manifest = {**manifest, "arrays_file_sha256": arrays_sha, "manifest_file_sha256": manifest_sha}
    return CarrierCache(entries, manifest, plan.source_sessions)


@dataclass(frozen=True)
class SourceRmsNormalizer:
    s_src: float
    source_cache_sha256: str
    entries: int
    rows: int
    dims: int
    normalizer_sha256: str

    @property
    def denominator(self) -> float:
        return max(float(self.s_src), NORMALIZER_FLOOR)

    def normalize(self, carrier: np.ndarray) -> np.ndarray:
        value = np.asarray(carrier, dtype=np.float64)
        if value.ndim < 2 or value.shape[-1] != 4 or not np.isfinite(value).all():
            raise CarrierIdDateLodoSourceError(f"invalid raw carrier for normalizer: {value.shape}")
        return np.asarray(value / self.denominator, dtype=np.float64)

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "schema": SOURCE_NORMALIZER_SCHEMA,
            "formula": NORMALIZER_FORMULA,
            "floor": NORMALIZER_FLOOR,
            "s_src": self.s_src,
            "denominator": self.denominator,
            "source_cache_sha256": self.source_cache_sha256,
            "entries": self.entries,
            "rows": self.rows,
            "dims": self.dims,
            "normalizer_sha256": self.normalizer_sha256,
        }


def _fit_source_rms_normalizer(cache: CarrierCache) -> SourceRmsNormalizer:
    values = np.stack([entry.carrier for entry in cache.entries], axis=0).astype(np.float64)
    if values.ndim != 3 or values.shape[1:] != (EXPECTED_NEURONS, 4):
        raise CarrierIdDateLodoSourceError("normalizer received malformed shared source cache")
    scalar = float(np.sqrt(np.mean(np.square(values, dtype=np.float64), dtype=np.float64)))
    if not math.isfinite(scalar) or scalar < 0:
        raise CarrierIdDateLodoSourceError("source RMS is undefined")
    body = {
        "schema": SOURCE_NORMALIZER_SCHEMA,
        "formula": NORMALIZER_FORMULA,
        "floor": NORMALIZER_FLOOR,
        "source_cache_sha256": cache.manifest["cache_sha256"],
        "entries": int(values.shape[0]),
        "rows": int(values.shape[1]),
        "dims": int(values.shape[2]),
        "s_src": scalar,
    }
    return SourceRmsNormalizer(
        scalar,
        str(cache.manifest["cache_sha256"]),
        int(values.shape[0]),
        int(values.shape[1]),
        int(values.shape[2]),
        canonical_sha256(body),
    )


def _persist_normalized_cache(cache: CarrierCache, normalizer: SourceRmsNormalizer, directory: Path, outer_date: str) -> dict[str, Any]:
    raw = np.stack([entry.carrier for entry in cache.entries], axis=0).astype(np.float64)
    normalized = normalizer.normalize(raw)
    arrays_sha = _publish_npz_once(directory / "normalized_source_m4_carriers.npz", carriers=normalized)
    body = {
        "schema": SOURCE_NORMALIZED_CACHE_SCHEMA,
        "outer_date": outer_date,
        "formula": NORMALIZER_FORMULA,
        "normalizer_floor": NORMALIZER_FLOOR,
        "normalizer_sha256": normalizer.normalizer_sha256,
        "source_cache_sha256": cache.manifest["cache_sha256"],
        "raw_shape": list(raw.shape),
        "normalized_shape": list(normalized.shape),
        "raw_carriers_sha256": array_sha256(raw),
        "normalized_carriers_sha256": array_sha256(normalized),
        "arrays_file_sha256": arrays_sha,
    }
    body["manifest_sha256"] = canonical_sha256(body)
    return {**body, "manifest_file_sha256": _publish_json_once(directory / "normalized_source_m4_carriers.manifest.json", body)}


@dataclass(frozen=True)
class SourceWindowIndex:
    """Minimal, framework-free source window index used only for scheduling.

    This deliberately does not inherit the CCE ``Dataset`` or construct a
    DataLoader.  Its ordering is identical to the CCE source-window rule:
    after left-padding a ``WINDOW_SIZE - 1`` history, a valid window ending at
    bin ``t`` has start index ``t``.  Thus the source-only scheduler needs
    only the public record's existing eval mask.
    """

    cache: CarrierCache
    window_indices: tuple[tuple[str, int], ...]
    window_indices_sha256: str


def build_source_window_index(records: Mapping[str, H1PilotRecord], cache: CarrierCache) -> SourceWindowIndex:
    windows: list[tuple[str, int]] = []
    for session in cache.source_sessions:
        record = records[session]
        if record.neural.shape[0] != record.eval_mask.shape[0]:
            raise CarrierIdDateLodoSourceError(f"{session}: source neural/eval-mask length mismatch")
        windows.extend((session, int(index)) for index, valid in enumerate(record.eval_mask) if bool(valid))
    if not windows:
        raise CarrierIdDateLodoSourceError("source window index has no eval-valid source windows")
    frozen = tuple(windows)
    return SourceWindowIndex(cache=cache, window_indices=frozen, window_indices_sha256=_window_hash(frozen))


class SourceSchedule:
    """A date-seeded, shared 50-epoch source schedule for both H-S and H-C."""

    def __init__(self, window_index: SourceWindowIndex, outer_date: str):
        date = assert_confirmatory_date(outer_date)
        grouped: dict[str, list[int]] = {name: [] for name in window_index.cache.source_sessions}
        for index, (session, _start) in enumerate(window_index.window_indices):
            grouped[session].append(index)
        batches: list[list[int]] = []
        for session in window_index.cache.source_sessions:
            token = int.from_bytes(hashlib.sha256(f"{FIXED_SEED}|{date}|carrierid-date-lodo-batch|{session}".encode()).digest()[:8], "big")
            permutation = random.Random(token).sample(grouped[session], len(grouped[session]))
            batches.extend(
                permutation[offset : offset + 32]
                for offset in range(0, len(permutation), 32)
                if len(permutation[offset : offset + 32]) == 32
            )
        token = int.from_bytes(hashlib.sha256(f"{FIXED_SEED}|{date}|carrierid-date-lodo-batches".encode()).digest()[:8], "big")
        self.batches = tuple(tuple(batch) for batch in random.Random(token).sample(batches, len(batches)))
        self.flat_indices = np.asarray([index for batch in self.batches for index in batch], dtype=np.int64)
        if not self.batches or self.flat_indices.size != len(self.batches) * 32:
            raise CarrierIdDateLodoSourceError("source schedule has no complete batches")
        self.outer_date = date
        self.batch_order_sha256 = array_sha256(self.flat_indices)
        schedule = np.empty((FIXED_EPOCHS, len(self.flat_indices)), dtype=np.int16)
        session_vector = np.asarray([window_index.window_indices[int(index)][0] for index in self.flat_indices], dtype=object)
        for session in window_index.cache.source_sessions:
            positions = np.flatnonzero(session_vector == session)
            legal = np.asarray(window_index.cache.starts_by_session[session], dtype=np.int16)
            token = hashlib.sha256(f"{FIXED_SEED}|{date}|carrierid-date-lodo-m4|{session}".encode()).digest()
            draws = np.random.default_rng(int.from_bytes(token[:8], "big")).integers(
                0, len(legal), size=(FIXED_EPOCHS, len(positions))
            )
            schedule[:, positions] = legal[draws]
        self.schedule = schedule
        self.schedule_sha256 = array_sha256(schedule)
        self.source_window_indices_sha256 = window_index.window_indices_sha256


def _window_hash(indices: Sequence[tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for session, start in indices:
        digest.update(session.encode("ascii"))
        digest.update(int(start).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def _persist_schedule(schedule: SourceSchedule, cache: CarrierCache, directory: Path) -> dict[str, Any]:
    array_sha = _publish_npy_once(directory / "source_calibration_schedule.npy", schedule.schedule)
    body = {
        "schema": SOURCE_SCHEDULE_SCHEMA,
        "outer_date": schedule.outer_date,
        "seed": FIXED_SEED,
        "epochs": FIXED_EPOCHS,
        "batch_size": 32,
        "batches_per_epoch": len(schedule.batches),
        "scheduled_samples_per_epoch": int(schedule.flat_indices.size),
        "source_window_indices_sha256": schedule.source_window_indices_sha256,
        "batch_order_sha256": schedule.batch_order_sha256,
        "calibration_schedule_sha256": schedule.schedule_sha256,
        "source_cache_sha256": cache.manifest["cache_sha256"],
        "selection": "date-seeded dedicated RNG selecting one legal contiguous M4 support per scheduled source sample",
    }
    return {**body, "schedule_file_sha256": array_sha, "manifest_file_sha256": _publish_json_once(directory / "source_schedule.manifest.json", body)}


def prepare_date_source_bundle(
    *,
    data_dir: str | Path,
    outer_date: str,
    raw_receipt_path: str | Path,
    eb_receipt_path: str | Path,
    cache_root: str | Path,
) -> dict[str, Any]:
    """Create one fresh immutable source bundle shared by H-S and H-C.

    The caller must use a new cache root.  This no-resume property makes a
    partially prepared date obvious instead of silently mixing artifacts from
    two source builds.
    """

    date = assert_confirmatory_date(outer_date)
    root = Path(cache_root).resolve()
    directory = root / date
    if directory.exists():
        raise FileExistsError(f"CarrierID date-LODO cache directory already exists: {directory}")
    records, load_audit = load_source_records_with_target_filename_index(data_dir, date)
    plan = reconstruct_plan_for_date(records, date, raw_receipt_path, eb_receipt_path)
    directory.mkdir(parents=True, exist_ok=False)
    plan_manifest = _persist_plan(plan, directory)
    carrier_cache = _build_carrier_cache(records, plan, directory)
    normalizer = _fit_source_rms_normalizer(carrier_cache)
    normalizer_path = directory / "source_rms_normalizer.manifest.json"
    normalizer_file_sha = _publish_json_once(normalizer_path, normalizer.manifest)
    normalized_manifest = _persist_normalized_cache(carrier_cache, normalizer, directory, date)
    # This framework-free index derives the exact source windows.  It cannot
    # construct a DataLoader, Trainer, CCE residual network, or checkpoint.
    source_windows = build_source_window_index(records, carrier_cache)
    schedule = SourceSchedule(source_windows, date)
    schedule_manifest = _persist_schedule(schedule, carrier_cache, directory)
    source_files = [
        {
            "role": "source_heldin_calib",
            "session": name,
            "date": records[name].date,
            "sha256": records[name].input_sha256,
            "size_bytes": records[name].path.stat().st_size,
        }
        for name in plan.source_sessions
    ]
    shared = {
        "outer_date": date,
        "source_manifest_schema": SOURCE_MANIFEST_SCHEMA,
        "source_cache_sha256": carrier_cache.manifest["cache_sha256"],
        "normalizer_sha256": normalizer.normalizer_sha256,
        "normalized_cache_sha256": normalized_manifest["normalized_carriers_sha256"],
        "calibration_schedule_sha256": schedule.schedule_sha256,
        "batch_order_sha256": schedule.batch_order_sha256,
        "source_window_indices_sha256": schedule.source_window_indices_sha256,
    }
    arms = {
        arm: {
            "role": role,
            "training_wrapper_status": "PHASE2_NOT_IMPLEMENTED",
            "shared_source_binding": dict(shared),
        }
        for arm, role in (
            ("H-S", "original SPINT identity MLP; carrier input ignored by future wrapper"),
            ("H-C", "h32 CarrierID replacement; future wrapper consumes normalized 4-D carrier"),
        )
    }
    manifest = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "status": "PASS_SOURCE_ONLY_SHARED_BUNDLE_NOT_LAUNCHED",
        "outer_date": date,
        "source_sessions": list(plan.source_sessions),
        "source_session_count": len(plan.source_sessions),
        "target_filename_index": load_audit.manifest(),
        "source_files": source_files,
        "frozen_plan": {
            "manifest_path": str((directory / "frozen_m4_plan.manifest.json").resolve()),
            "manifest_sha256": plan_manifest["manifest_file_sha256"],
            "transform_sha256": plan.transform_sha256,
            "raw_receipt_sha256": plan.raw_receipt_sha256,
            "eb_receipt_sha256": plan.eb_receipt_sha256,
        },
        "carrier_cache": {
            "manifest_path": str((directory / "source_m4_carriers.manifest.json").resolve()),
            "manifest_sha256": carrier_cache.manifest["manifest_file_sha256"],
            "cache_sha256": carrier_cache.manifest["cache_sha256"],
        },
        "normalizer": {
            "manifest_path": str(normalizer_path.resolve()),
            "manifest_file_sha256": normalizer_file_sha,
            **normalizer.manifest,
        },
        "normalized_cache": {
            "manifest_path": str((directory / "normalized_source_m4_carriers.manifest.json").resolve()),
            "manifest_sha256": normalized_manifest["manifest_file_sha256"],
            "normalized_carriers_sha256": normalized_manifest["normalized_carriers_sha256"],
        },
        "schedule": {
            "manifest_path": str((directory / "source_schedule.manifest.json").resolve()),
            "manifest_sha256": schedule_manifest["manifest_file_sha256"],
            "batches_per_epoch": schedule_manifest["batches_per_epoch"],
            "scheduled_samples_per_epoch": schedule_manifest["scheduled_samples_per_epoch"],
            "calibration_schedule_sha256": schedule.schedule_sha256,
            "batch_order_sha256": schedule.batch_order_sha256,
        },
        "arms": arms,
        "source_only_scope": {
            "target_recordings_opened": 0,
            "target_bytes_read": 0,
            "minival_opened_or_enumerated": False,
            "formal_heldout_opened_or_enumerated": False,
            "evalai_opened_or_enumerated": False,
            "cuda_constructed_or_launched": False,
            "trainer_constructed_or_launched": False,
            "checkpoint_created_or_loaded": False,
            "cce_residual_model_imported_or_constructed": False,
        },
    }
    manifest_path = directory / "shared_source_manifest.json"
    manifest_sha = _publish_json_once(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path.resolve()), "manifest_sha256": manifest_sha}


def validate_source_bundle_manifest(manifest: Mapping[str, Any], *, outer_date: str) -> None:
    """Validate enough immutable fields for the no-data Phase-2 planner."""

    date = assert_confirmatory_date(outer_date)
    if manifest.get("schema") != SOURCE_MANIFEST_SCHEMA or manifest.get("status") != "PASS_SOURCE_ONLY_SHARED_BUNDLE_NOT_LAUNCHED":
        raise CarrierIdDateLodoSourceError("unexpected CarrierID date-LODO source manifest")
    if manifest.get("outer_date") != date:
        raise CarrierIdDateLodoSourceError("source manifest outer-date mismatch")
    source, target = tuple(source_sessions_for_date(date)), tuple(target_sessions_for_date(date))
    if tuple(manifest.get("source_sessions", ())) != source or int(manifest.get("source_session_count", -1)) != len(source):
        raise CarrierIdDateLodoSourceError("source manifest source session partition drift")
    target_index = manifest.get("target_filename_index")
    if not isinstance(target_index, Mapping):
        raise CarrierIdDateLodoSourceError("source manifest lacks target filename index")
    if tuple(target_index.get("target_sessions_indexed", ())) != target:
        raise CarrierIdDateLodoSourceError("source manifest target filename partition drift")
    if target_index.get("target_recordings_opened") != 0 or target_index.get("target_bytes_read") != 0:
        raise CarrierIdDateLodoSourceError("source manifest records target access")
    arms = manifest.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"H-S", "H-C"}:
        raise CarrierIdDateLodoSourceError("source manifest must bind exactly shared H-S/H-C arms")
    left, right = arms["H-S"].get("shared_source_binding"), arms["H-C"].get("shared_source_binding")
    if not isinstance(left, Mapping) or dict(left) != dict(right):
        raise CarrierIdDateLodoSourceError("H-S/H-C do not share source cache/normalizer/schedule")
    scope = manifest.get("source_only_scope")
    if not isinstance(scope, Mapping) or any(scope.get(field) is not False for field in (
        "minival_opened_or_enumerated", "formal_heldout_opened_or_enumerated", "evalai_opened_or_enumerated",
        "cuda_constructed_or_launched", "trainer_constructed_or_launched", "checkpoint_created_or_loaded",
        "cce_residual_model_imported_or_constructed",
    )):
        raise CarrierIdDateLodoSourceError("source manifest violates source-only scope")
