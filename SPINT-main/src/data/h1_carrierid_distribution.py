"""Fresh, matched source consumers for the H1 support/query carrier diagnostic.

This module is intentionally separate from the sealed H1 CarrierID paths.  It
constructs a *common* source schedule from contiguous eight-trial blocks.  For
each schedule the two consumers see the same neural query window, behaviour
target, four-trial identity, session, sample order, normalizer, architecture,
and initialization.  Their sole differing tensor is the 4-D carrier:

``D-S4``
    carrier fitted on trials ``t .. t+3`` (ordinary support);
``D-Q4``
    carrier fitted on ``t+4 .. t+7`` (a source analogue of deliberately using
    query-local labels at target time).

The latter is deliberately non-deployable and is not an upper bound.  This
module never imports or opens H1 target/minival/formal/EvalAI data.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler

from src.data.h1_m4_eb_pilot import (
    FOLD0_DATE,
    H1_M4_FOLD0_SOURCE,
    MAX_TRIAL_LENGTH,
    SUPPORT_TRIALS,
    WINDOW,
    FrozenEBPlan,
    H1PilotRecord,
    PilotDataError,
    fit_frozen_carrier,
    interpolate_identity,
    load_source_records,
    reconstruct_frozen_plan,
)
from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    array_sha256,
    canonical_json_bytes,
    canonical_sha256,
    sha256_file,
)


DISTRIBUTION_ARMS = ("s4", "q4")
SOURCE_ONLY_SCHEMA = "h1_carrierid_h32_fresh_distribution_source_manifest_v1"
PAIR_CACHE_SCHEMA = "h1_carrierid_h32_fresh_distribution_pair_cache_v1"
SCHEDULE_SCHEMA = "h1_carrierid_h32_fresh_distribution_schedule_v1"


@dataclass(frozen=True)
class EightTrialSchedule:
    """One eligible contiguous 8-trial source schedule.

    ``query_first_bin`` is the first bin of ``t+4``.  Every query window is
    selected with its *start* at or after this boundary, matching the strict
    target convention rather than allowing a history to cross support.
    """

    session_name: str
    start_index: int
    support_values: tuple[float, float, float, float]
    query_values: tuple[float, float, float, float]
    query_first_bin: int

    @property
    def all_values(self) -> tuple[float, ...]:
        return self.support_values + self.query_values

    def manifest(self) -> dict[str, Any]:
        return {
            "session_name": self.session_name,
            "start_index": self.start_index,
            "support_trial_values": list(self.support_values),
            "query_trial_values": list(self.query_values),
            "query_first_bin": self.query_first_bin,
        }


@dataclass(frozen=True)
class DistributionSample:
    """A declared source (schedule, strict-query-window) training sample."""

    schedule_index: int
    window_start: int


@dataclass(frozen=True)
class PairedCarrierNormalizer:
    """One scalar fitted on both arms over the common 8-trial schedule scope."""

    s_pair: float
    floor: float
    normalizer_sha256: str
    source_schedule_sha256: str
    source_entries: int

    @property
    def denominator(self) -> float:
        return max(float(self.s_pair), float(self.floor))

    # H1CarrierIdLitModule reads ``s_src`` when binding checkpoint metadata.
    # The property keeps that schema-compatible name while the protocol makes
    # the different paired scope explicit in the manifest.
    @property
    def s_src(self) -> float:
        return float(self.s_pair)

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "h1_carrierid_h32_fresh_distribution_paired_scalar_v1",
            "formula": "s_pair=sqrt(mean(concat(S4_raw,Q4_raw)**2)); C_norm=C_raw/max(s_pair,1e-12)",
            "floor": float(self.floor),
            "s_pair": float(self.s_pair),
            "denominator": self.denominator,
            "source_schedule_sha256": self.source_schedule_sha256,
            "source_entries": int(self.source_entries),
            "normalizer_sha256": self.normalizer_sha256,
        }

    def normalize(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim < 2 or array.shape[-1] != 4 or not np.isfinite(array).all():
            raise NormalizedV2ContractError("fresh distribution carrier must be finite [...,4]")
        result = array / self.denominator
        if not np.isfinite(result).all():
            raise NormalizedV2ContractError("fresh distribution carrier normalization is nonfinite")
        return np.asarray(result, dtype=np.float64)


def _first_query_bin(record: H1PilotRecord, trial_value: float) -> int:
    bins = np.flatnonzero(
        record.eval_mask & np.isfinite(record.trial_num) & (record.trial_num == float(trial_value))
    )
    if bins.size == 0:
        raise PilotDataError(f"{record.session_name}: query trial {trial_value} has no eval-valid bin")
    return int(bins[0])


def eligible_eight_trial_schedules(record: H1PilotRecord) -> tuple[EightTrialSchedule, ...]:
    """Return every legal continuous 8-trial S4/Q4 schedule for one source record."""

    schedules: list[EightTrialSchedule] = []
    width = 2 * SUPPORT_TRIALS
    for start in range(0, len(record.trial_values) - width + 1):
        values = tuple(float(value) for value in record.trial_values[start : start + width])
        if len(values) != width:
            continue
        trials = [record.blocks_for(value) for value in values]
        if any(trial.rates.shape[0] < 2 for trial in trials):
            continue
        # Identity is always first-block/S4 identity.  Its cubic interpolation
        # must therefore be constructible before the schedule is eligible.
        for value in values[:SUPPORT_TRIALS]:
            record.eval_trial_neural(value)
        boundary = _first_query_bin(record, values[SUPPORT_TRIALS])
        # Strict query windows must exist after the S4/Q4 boundary.
        exists = any(
            record.eval_mask[end - 1]
            for end in range(boundary + WINDOW, record.neural.shape[0] + 1)
        )
        if not exists:
            continue
        schedules.append(
            EightTrialSchedule(
                session_name=record.session_name,
                start_index=int(start),
                support_values=tuple(values[:SUPPORT_TRIALS]),  # type: ignore[arg-type]
                query_values=tuple(values[SUPPORT_TRIALS:]),  # type: ignore[arg-type]
                query_first_bin=boundary,
            )
        )
    if not schedules:
        raise PilotDataError(f"{record.session_name}: no legal contiguous 8-trial S4/Q4 schedules")
    return tuple(schedules)


def build_common_eight_trial_schedules(
    records: Mapping[str, H1PilotRecord],
) -> tuple[EightTrialSchedule, ...]:
    """Build the source-only common schedule in fixed source-session order."""

    if tuple(records) != tuple(H1_M4_FOLD0_SOURCE):
        # ``dict`` construction in the loader has canonical order.  Reject a
        # differently ordered mapping so cache/sample hashes cannot drift.
        raise PilotDataError("fresh distribution schedule requires canonical 11-source session order")
    output: list[EightTrialSchedule] = []
    for name in H1_M4_FOLD0_SOURCE:
        record = records[name]
        if record.date == FOLD0_DATE:
            raise PilotDataError("target date leaked into fresh distribution source schedule")
        output.extend(eligible_eight_trial_schedules(record))
    return tuple(output)


def schedule_sha256(schedules: Sequence[EightTrialSchedule]) -> str:
    return canonical_sha256([schedule.manifest() for schedule in schedules])


def _carrier_pair_for_schedule(
    record: H1PilotRecord,
    plan: FrozenEBPlan,
    schedule: EightTrialSchedule,
) -> tuple[np.ndarray, np.ndarray]:
    if schedule.session_name != record.session_name:
        raise PilotDataError("schedule/session mismatch")
    s4 = fit_frozen_carrier(record, plan, schedule.support_values)["carrier"]
    q4 = fit_frozen_carrier(record, plan, schedule.query_values)["carrier"]
    if s4.shape != q4.shape or s4.shape != (record.num_neurons, 4):
        raise PilotDataError("fresh distribution S4/Q4 carrier shape mismatch")
    if not np.isfinite(s4).all() or not np.isfinite(q4).all():
        raise PilotDataError("fresh distribution carrier is nonfinite")
    return np.asarray(s4, dtype=np.float64), np.asarray(q4, dtype=np.float64)


def build_paired_carriers(
    records: Mapping[str, H1PilotRecord], plan: FrozenEBPlan, schedules: Sequence[EightTrialSchedule]
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the two carrier tensors on exactly the same eligible schedules."""

    s4_values: list[np.ndarray] = []
    q4_values: list[np.ndarray] = []
    for schedule in schedules:
        s4, q4 = _carrier_pair_for_schedule(records[schedule.session_name], plan, schedule)
        s4_values.append(s4); q4_values.append(q4)
    if not s4_values:
        raise PilotDataError("fresh distribution carrier set is empty")
    s4_stack, q4_stack = np.stack(s4_values, axis=0), np.stack(q4_values, axis=0)
    if s4_stack.shape != q4_stack.shape or s4_stack.shape[1:] != (176, 4):
        raise PilotDataError(f"unexpected paired carrier shape {s4_stack.shape}/{q4_stack.shape}")
    return s4_stack, q4_stack


def fit_paired_normalizer(
    s4_raw: np.ndarray, q4_raw: np.ndarray, *, source_schedule_digest: str
) -> PairedCarrierNormalizer:
    """Fit a shared scalar from the paired S4/Q4 tensors, never one arm alone."""

    s4 = np.asarray(s4_raw, dtype=np.float64)
    q4 = np.asarray(q4_raw, dtype=np.float64)
    if s4.shape != q4.shape or s4.ndim != 3 or s4.shape[1:] != (176, 4):
        raise NormalizedV2ContractError("paired normalizer requires matching [E,176,4] S4/Q4 tensors")
    if not np.isfinite(s4).all() or not np.isfinite(q4).all():
        raise NormalizedV2ContractError("paired normalizer input is nonfinite")
    s_pair = float(np.sqrt(np.mean(np.square(np.concatenate((s4, q4), axis=0), dtype=np.float64))))
    body = {
        "schema": "h1_carrierid_h32_fresh_distribution_paired_scalar_v1",
        "source_schedule_sha256": str(source_schedule_digest),
        "source_entries": int(s4.shape[0]),
        "s_pair": s_pair,
        "floor": 1.0e-12,
    }
    return PairedCarrierNormalizer(
        s_pair=s_pair,
        floor=1.0e-12,
        normalizer_sha256=canonical_sha256(body),
        source_schedule_sha256=str(source_schedule_digest),
        source_entries=int(s4.shape[0]),
    )


def _strict_query_window_starts(record: H1PilotRecord, boundary: int, count: int) -> tuple[int, ...]:
    candidates = np.asarray(
        [
            start
            for start in range(int(boundary), record.neural.shape[0] - WINDOW + 1)
            if bool(record.eval_mask[start + WINDOW - 1])
        ],
        dtype=np.int64,
    )
    if candidates.size < count:
        raise PilotDataError(
            f"{record.session_name}: only {candidates.size} strict query windows after boundary, need {count}"
        )
    # Exactly count unique, evenly spread fixed windows.  This constrains the
    # new diagnostic's GPU cost while preserving the strict post-S4 boundary.
    positions = np.linspace(0, candidates.size - 1, num=count, dtype=np.int64)
    selected = tuple(int(value) for value in candidates[positions])
    if len(set(selected)) != count:
        raise PilotDataError("strict query selection unexpectedly duplicated a window")
    return selected


def build_common_samples(
    records: Mapping[str, H1PilotRecord], schedules: Sequence[EightTrialSchedule], *, windows_per_schedule: int
) -> tuple[DistributionSample, ...]:
    if windows_per_schedule != 32:
        raise PilotDataError("fresh distribution diagnostic fixes windows_per_schedule=32")
    output: list[DistributionSample] = []
    for index, schedule in enumerate(schedules):
        record = records[schedule.session_name]
        for window_start in _strict_query_window_starts(record, schedule.query_first_bin, windows_per_schedule):
            output.append(DistributionSample(index, window_start))
    if not output:
        raise PilotDataError("fresh distribution sample list is empty")
    return tuple(output)


def sample_sha256(samples: Sequence[DistributionSample], schedules: Sequence[EightTrialSchedule]) -> str:
    body = [
        {"session_name": schedules[sample.schedule_index].session_name,
         "schedule_index": int(sample.schedule_index), "window_start": int(sample.window_start)}
        for sample in samples
    ]
    return canonical_sha256(body)


def _write_bytes_once(path: Path, payload: bytes) -> str:
    resolved = path.resolve()
    if resolved.exists():
        if not resolved.is_file() or stat.S_IMODE(resolved.stat().st_mode) != 0o444:
            raise NormalizedV2ContractError(f"fresh distribution artifact must be immutable 0444: {resolved}")
        if resolved.read_bytes() != payload:
            raise NormalizedV2ContractError(f"fresh distribution artifact drift / overwrite refused: {resolved}")
    else:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(payload)
        resolved.chmod(0o444)
    return sha256_file(resolved)


def _write_npz_once(path: Path, **arrays: np.ndarray) -> str:
    """Write a named-array cache once, comparing array values not ZIP timestamps."""

    resolved = path.resolve()
    expected = {name: np.asarray(value) for name, value in arrays.items()}
    if resolved.exists():
        if not resolved.is_file() or stat.S_IMODE(resolved.stat().st_mode) != 0o444:
            raise NormalizedV2ContractError(f"fresh distribution artifact must be immutable 0444: {resolved}")
        with np.load(resolved, allow_pickle=False) as existing:
            if set(existing.files) != set(expected) or any(
                not np.array_equal(np.asarray(existing[name]), value) for name, value in expected.items()
            ):
                raise NormalizedV2ContractError(f"fresh distribution artifact drift / overwrite refused: {resolved}")
    else:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        np.savez(resolved, **expected)
        resolved.chmod(0o444)
    return sha256_file(resolved)


def persist_paired_distribution_cache(
    cache_dir: str | Path,
    *,
    schedules: Sequence[EightTrialSchedule],
    samples: Sequence[DistributionSample],
    s4_raw: np.ndarray,
    q4_raw: np.ndarray,
    normalizer: PairedCarrierNormalizer,
    source_files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Persist only additive, immutable D-S4/D-Q4 source artifacts."""

    directory = Path(cache_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    schedule_digest, samples_digest = schedule_sha256(schedules), sample_sha256(samples, schedules)
    arrays_sha = _write_npz_once(
        directory / "h1_fresh_distribution_pair_cache.npz",
        s4_raw=np.asarray(s4_raw, dtype=np.float64), q4_raw=np.asarray(q4_raw, dtype=np.float64),
        schedule_indices=np.asarray([sample.schedule_index for sample in samples], dtype=np.int32),
        window_starts=np.asarray([sample.window_start for sample in samples], dtype=np.int64),
    )
    manifest = {
        "schema": PAIR_CACHE_SCHEMA,
        "source_sessions": list(H1_M4_FOLD0_SOURCE),
        "source_files": [dict(value) for value in source_files],
        "source_schedule_sha256": schedule_digest,
        "common_query_samples_sha256": samples_digest,
        "schedule_count": len(schedules),
        "sample_count": len(samples),
        "windows_per_schedule": 32,
        "s4_raw_sha256": array_sha256(np.asarray(s4_raw, dtype=np.float64)),
        "q4_raw_sha256": array_sha256(np.asarray(q4_raw, dtype=np.float64)),
        "paired_cache_file_sha256": arrays_sha,
        "normalizer": normalizer.manifest,
        "arms": {
            "s4": "carrier fit on t..t+3; shared first-block identity",
            "q4": "carrier fit on t+4..t+7; source analogue of query-label leakage",
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_payload = canonical_json_bytes(manifest)
    manifest_sha = _write_bytes_once(directory / "h1_fresh_distribution_pair_cache.manifest.json", manifest_payload)
    return {**manifest, "manifest_file_sha256": manifest_sha}


class H1CarrierIdDistributionDataset(Dataset):
    """One arm of the matched, strict-query, eight-trial source dataset."""

    def __init__(
        self,
        records: Mapping[str, H1PilotRecord],
        schedules: Sequence[EightTrialSchedule],
        samples: Sequence[DistributionSample],
        s4_normalized: np.ndarray,
        q4_normalized: np.ndarray,
        arm: str,
    ) -> None:
        if arm not in DISTRIBUTION_ARMS:
            raise NormalizedV2ContractError(f"unknown fresh distribution arm {arm!r}")
        self.records = {name: records[name] for name in H1_M4_FOLD0_SOURCE}
        self.schedules = tuple(schedules)
        self.samples = tuple(samples)
        self.arm = arm
        self.s4_normalized = np.asarray(s4_normalized, dtype=np.float32)
        self.q4_normalized = np.asarray(q4_normalized, dtype=np.float32)
        if self.s4_normalized.shape != self.q4_normalized.shape or self.s4_normalized.shape != (len(schedules), 176, 4):
            raise NormalizedV2ContractError("fresh distribution normalized carrier cache shape mismatch")
        if not np.isfinite(self.s4_normalized).all() or not np.isfinite(self.q4_normalized).all():
            raise NormalizedV2ContractError("fresh distribution normalized carrier cache nonfinite")
        self._identity: dict[int, np.ndarray] = {}
        for index, schedule in enumerate(self.schedules):
            self._identity[index] = interpolate_identity(self.records[schedule.session_name], schedule.support_values)
        self.common_query_samples_sha256 = sample_sha256(self.samples, self.schedules)
        self.identity_schedule_sha256 = canonical_sha256(
            [{"session": item.session_name, "support": list(item.support_values)} for item in self.schedules]
        )
        self.effective_source_carriers_sha256 = array_sha256(
            self.s4_normalized if arm == "s4" else self.q4_normalized
        )

    def __len__(self) -> int:
        return len(self.samples)

    def sample_metadata(self, index: int) -> dict[str, Any]:
        sample = self.samples[int(index)]
        schedule = self.schedules[sample.schedule_index]
        return {**schedule.manifest(), "window_start": int(sample.window_start), "arm": self.arm}

    def __getitem__(self, index: int):
        sample = self.samples[int(index)]
        schedule = self.schedules[sample.schedule_index]
        record = self.records[schedule.session_name]
        start, end = int(sample.window_start), int(sample.window_start) + WINDOW
        if start < schedule.query_first_bin or end > record.neural.shape[0] or not record.eval_mask[end - 1]:
            raise PilotDataError("fresh distribution query window violates its common strict boundary")
        carrier = self.s4_normalized if self.arm == "s4" else self.q4_normalized
        return (
            np.asarray(record.neural[start:end], dtype=np.float32),
            np.asarray(record.velocity[start:end], dtype=np.float32),
            self._identity[sample.schedule_index],
            schedule.session_name,
            carrier[sample.schedule_index],
        )


class H1DistributionPairedBatchSampler(Sampler[list[int]]):
    """A deterministic shared batch order; no arm-specific resampling exists."""

    def __init__(self, dataset: H1CarrierIdDistributionDataset, *, batch_size: int, seed: int, max_epochs: int) -> None:
        if batch_size != 32 or seed != 42 or max_epochs != 50:
            raise PilotDataError("fresh distribution diagnostic fixes batch=32, seed=42, epochs=50")
        self.dataset, self.batch_size, self.seed, self.max_epochs = dataset, batch_size, seed, max_epochs
        grouped: dict[str, list[int]] = {name: [] for name in H1_M4_FOLD0_SOURCE}
        for index, sample in enumerate(dataset.samples):
            grouped[dataset.schedules[sample.schedule_index].session_name].append(index)
        batches: list[list[int]] = []
        for name in H1_M4_FOLD0_SOURCE:
            indexes = random.Random(seed).sample(grouped[name], len(grouped[name]))
            batches.extend(
                indexes[offset : offset + batch_size]
                for offset in range(0, len(indexes), batch_size)
                if len(indexes[offset : offset + batch_size]) == batch_size
            )
        self.batches = random.Random(seed).sample(batches, len(batches))
        self.flat_indices = np.asarray([item for batch in self.batches for item in batch], dtype=np.int64)
        self.batch_order_sha256 = array_sha256(self.flat_indices)
        self._epoch = 0

    def reset_epoch(self) -> None:
        self._epoch = 0

    def __iter__(self):
        if self._epoch >= self.max_epochs:
            raise RuntimeError("fresh distribution fixed 50-epoch batch order exhausted")
        self._epoch += 1
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


class H1CarrierIdDistributionDataModule(pl.LightningDataModule):
    """Source-only paired DataModule for D-S4/D-Q4 fresh consumer training."""

    def __init__(
        self,
        *,
        task: str,
        data_dir: str,
        raw_receipt_path: str,
        eb_receipt_path: str,
        cache_dir: str,
        carrier_distribution_arm: str,
        batch_size: int = 32,
        window_size: int = 700,
        calibration_n_trials: int = 4,
        max_trial_length: int = 1024,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        fixed_epochs: int = 50,
        windows_per_schedule: int = 32,
        **unused: Any,
    ) -> None:
        super().__init__()
        if task != "h1" or carrier_distribution_arm not in DISTRIBUTION_ARMS:
            raise NormalizedV2ContractError("fresh distribution DataModule requires H1 arm in {'s4','q4'}")
        if (batch_size, window_size, calibration_n_trials, max_trial_length, num_workers, seed, fixed_epochs, windows_per_schedule) != (
            32, WINDOW, 4, MAX_TRIAL_LENGTH, 0, 42, 50, 32
        ):
            raise NormalizedV2ContractError("fresh distribution fixed source protocol parameter drift")
        self.save_hyperparameters(logger=False)
        self.batch_size_per_device = 32
        self._setup_done = False

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit"}:
            raise RuntimeError("fresh distribution DataModule permits source fit only; target is fail-closed")
        if self._setup_done:
            return
        records = load_source_records(self.hparams.data_dir)
        if len(records) != 11 or tuple(records) != H1_M4_FOLD0_SOURCE:
            raise PilotDataError("fresh distribution must open exactly the fixed 11 source recordings")
        plan = reconstruct_frozen_plan(records, self.hparams.raw_receipt_path, self.hparams.eb_receipt_path)
        schedules = build_common_eight_trial_schedules(records)
        schedule_digest = schedule_sha256(schedules)
        s4_raw, q4_raw = build_paired_carriers(records, plan, schedules)
        normalizer = fit_paired_normalizer(s4_raw, q4_raw, source_schedule_digest=schedule_digest)
        samples = build_common_samples(records, schedules, windows_per_schedule=self.hparams.windows_per_schedule)
        source_files = [
            {"session": name, "date": records[name].date, "sha256": records[name].input_sha256,
             "size_bytes": records[name].path.stat().st_size}
            for name in H1_M4_FOLD0_SOURCE
        ]
        cache_manifest = persist_paired_distribution_cache(
            self.hparams.cache_dir, schedules=schedules, samples=samples, s4_raw=s4_raw, q4_raw=q4_raw,
            normalizer=normalizer, source_files=source_files,
        )
        dataset = H1CarrierIdDistributionDataset(
            records, schedules, samples, normalizer.normalize(s4_raw), normalizer.normalize(q4_raw),
            self.hparams.carrier_distribution_arm,
        )
        sampler = H1DistributionPairedBatchSampler(dataset, batch_size=32, seed=42, max_epochs=50)
        self.records, self.plan = records, plan
        self.schedules, self.samples, self.normalizer = schedules, samples, normalizer
        self.train_dataset, self.train_batch_sampler = dataset, sampler
        self._manifest = {
            "schema": SOURCE_ONLY_SCHEMA,
            "fold_date": FOLD0_DATE,
            "arm": self.hparams.carrier_distribution_arm,
            "source_sessions": list(H1_M4_FOLD0_SOURCE),
            "source_files": source_files,
            "target_sessions_not_opened": ["ses-19250101T111740", "ses-19250101T112404"],
            "target_nwb_opened_during_training_setup": False,
            "minival_or_heldout_enumerated": False,
            "formal_or_evalai_opened_or_enumerated": False,
            "source_schedule_sha256": schedule_digest,
            "common_query_samples_sha256": dataset.common_query_samples_sha256,
            "identity_schedule_sha256": dataset.identity_schedule_sha256,
            "batch_order_sha256": sampler.batch_order_sha256,
            "s4_effective_carriers_sha256": array_sha256(dataset.s4_normalized),
            "q4_effective_carriers_sha256": array_sha256(dataset.q4_normalized),
            "effective_source_carriers_sha256": dataset.effective_source_carriers_sha256,
            "carrier_cache_sha256": cache_manifest["manifest_sha256"],
            "normalized_cache_sha256": canonical_sha256({
                "s4": array_sha256(dataset.s4_normalized), "q4": array_sha256(dataset.q4_normalized),
                "normalizer": normalizer.normalizer_sha256,
            }),
            "normalizer_sha256": normalizer.normalizer_sha256,
            "normalizer": normalizer.manifest,
            "transform_sha256": plan.transform_sha256,
            "schedule_count": len(schedules),
            "sample_count": len(samples),
            "batches_per_epoch": len(sampler),
            "scheduled_samples_per_epoch": int(sampler.flat_indices.size),
            "epochs": 50,
            "windows_per_schedule": 32,
            "q4_label_scope": "t+4..t+7 source analogue of deliberate query-label leakage",
            "same_arm_inputs_except_carrier": True,
        }
        self._manifest_sha256 = canonical_sha256(self._manifest)
        self._setup_done = True

    def train_dataloader(self):
        if not self._setup_done:
            raise RuntimeError("call setup('fit') before train_dataloader")
        return DataLoader(self.train_dataset, batch_sampler=self.train_batch_sampler, num_workers=0,
                          pin_memory=bool(self.hparams.pin_memory))

    def val_dataloader(self):
        return []

    def test_dataloader(self):
        raise RuntimeError("fresh distribution target/formal loader is forbidden")

    def predict_dataloader(self):
        raise RuntimeError("fresh distribution target prediction is forbidden")

    def pilot_manifest(self) -> dict[str, Any]:
        if not self._setup_done:
            raise RuntimeError("call setup('fit') before requesting fresh distribution manifest")
        return dict(self._manifest)

    @property
    def pilot_manifest_sha256(self) -> str:
        if not self._setup_done:
            raise RuntimeError("call setup('fit') before requesting fresh distribution manifest SHA")
        return self._manifest_sha256
