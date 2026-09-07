"""Exposure-matched source consumers for the fresh H1 D-S4/D-Q4 diagnostic.

This is deliberately a new namespace.  It does not alter the sealed
``h1_carrierid_distribution`` v1 source path.  The original fresh diagnostic
used 72 legal eight-trial schedules but only 32 post-support query windows per
schedule (2,304 samples / 72 batches per epoch).  That made a poor comparison
to H-C's 115,520 source samples / 3,610 batches per epoch.

The D-S4e/D-Q4e path keeps the same 72 paired-eligible schedules and the same
two carrier fits, but deterministically selects 115,520 *unique* strict query
samples each epoch.  Selection is stratified over schedules, session-batch
aligned, and spreads samples through every schedule's legal post-support
window set.  Thus the two arms differ only in S4 versus Q4 carrier content,
not source exposure, sample order, initialization, or data coverage.

Only the eleven public fold-0 source recordings are imported here.  Target,
minival, formal-heldout, and EvalAI interfaces remain absent by construction.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Sampler

from src.data import h1_carrierid_distribution as dist
from src.data.h1_m4_eb_pilot import (
    FOLD0_DATE,
    H1_M4_FOLD0_SOURCE,
    MAX_TRIAL_LENGTH,
    SUPPORT_TRIALS,
    WINDOW,
    H1PilotRecord,
    PilotDataError,
    load_source_records,
    reconstruct_frozen_plan,
)
from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    array_sha256,
    canonical_json_bytes,
    canonical_sha256,
)


EXPOSURE_SCHEMA = "h1_carrierid_h32_fresh_distribution_exposure_source_manifest_v1"
EXPOSURE_CACHE_SCHEMA = "h1_carrierid_h32_fresh_distribution_exposure_pair_cache_v1"
EXPOSURE_SCHEDULE_COUNT = 72
EXPOSURE_SAMPLES_PER_EPOCH = 115_520
EXPOSURE_BATCH_SIZE = 32
EXPOSURE_BATCHES_PER_EPOCH = 3_610
EXPOSURE_EPOCHS = 50
EXPOSURE_ARMS = ("s4", "q4")


@dataclass(frozen=True)
class ExposureSamplePlan:
    """The common strict-query plan used by both exposure-matched arms."""

    samples: tuple[dist.DistributionSample, ...]
    candidate_counts: tuple[int, ...]
    selected_counts: tuple[int, ...]
    quota_sha256: str

    def manifest(self, schedules: Sequence[dist.EightTrialSchedule]) -> dict[str, Any]:
        if len(schedules) != len(self.candidate_counts) or len(schedules) != len(self.selected_counts):
            raise NormalizedV2ContractError("exposure sample plan/schedule cardinality drift")
        return {
            "target_samples_per_epoch": EXPOSURE_SAMPLES_PER_EPOCH,
            "target_batches_per_epoch": EXPOSURE_BATCHES_PER_EPOCH,
            "schedule_count": len(schedules),
            "candidate_windows_per_schedule_min": int(min(self.candidate_counts)),
            "candidate_windows_per_schedule_median": float(np.median(self.candidate_counts)),
            "candidate_windows_per_schedule_max": int(max(self.candidate_counts)),
            "selected_windows_per_schedule_min": int(min(self.selected_counts)),
            "selected_windows_per_schedule_median": float(np.median(self.selected_counts)),
            "selected_windows_per_schedule_max": int(max(self.selected_counts)),
            "selection": (
                "unique deterministic cyclic-even spacing over each schedule's full strict "
                "post-support candidate window set; not repeated fixed-32 windows"
            ),
            "quota_sha256": self.quota_sha256,
            "common_query_samples_sha256": dist.sample_sha256(self.samples, schedules),
        }


def strict_post_support_window_candidates(
    record: H1PilotRecord, boundary: int
) -> np.ndarray:
    """Every legal window start whose entire 700-bin history is post-support."""

    candidates = np.asarray(
        [
            start
            for start in range(int(boundary), record.neural.shape[0] - WINDOW + 1)
            if bool(record.eval_mask[start + WINDOW - 1])
        ],
        dtype=np.int64,
    )
    if candidates.size == 0:
        raise PilotDataError(f"{record.session_name}: no strict post-support query windows")
    return candidates


def _balanced_schedule_quotas(
    schedules: Sequence[dist.EightTrialSchedule], *, total: int = EXPOSURE_SAMPLES_PER_EPOCH
) -> tuple[int, ...]:
    """Return deterministic schedule quotas with each session batch-aligned.

    A naive equal split gives the desired global total but causes per-session
    tail dropping by the sampler.  We remove each session's modulo-32 residue,
    then restore the (necessarily multiple-of-32) global residue in cyclic
    32-sample chunks.  Every source window selected below is therefore seen;
    no arm-specific or hidden tail truncation is possible.
    """

    if len(schedules) != EXPOSURE_SCHEDULE_COUNT:
        raise PilotDataError(
            f"exposure diagnostic requires exactly {EXPOSURE_SCHEDULE_COUNT} paired schedules, "
            f"got {len(schedules)}"
        )
    if total != EXPOSURE_SAMPLES_PER_EPOCH or total % EXPOSURE_BATCH_SIZE:
        raise PilotDataError("exposure sample total must be the frozen 115520 divisible by 32")
    base, extra = divmod(total, len(schedules))
    quotas = [base + int(index < extra) for index in range(len(schedules))]
    by_session: dict[str, list[int]] = defaultdict(list)
    for index, schedule in enumerate(schedules):
        by_session[schedule.session_name].append(index)
    residue_total = 0
    for name in H1_M4_FOLD0_SOURCE:
        indexes = by_session.get(name, [])
        if not indexes:
            raise PilotDataError(f"exposure schedule omitted canonical source session {name}")
        residue = sum(quotas[index] for index in indexes) % EXPOSURE_BATCH_SIZE
        if residue:
            donor = indexes[-1]
            if quotas[donor] <= residue:
                raise PilotDataError("exposure quota session-balance correction underflow")
            quotas[donor] -= residue
            residue_total += residue
    if residue_total % EXPOSURE_BATCH_SIZE:
        raise PilotDataError("exposure quota correction must be a whole number of batches")
    for chunk in range(residue_total // EXPOSURE_BATCH_SIZE):
        # A +32 change leaves the recipient session batch-aligned.  Cyclic
        # allocation avoids placing every correction into one schedule.
        quotas[chunk % len(quotas)] += EXPOSURE_BATCH_SIZE
    if sum(quotas) != total or any(value <= 0 for value in quotas):
        raise PilotDataError("exposure quota total/positivity contract failed")
    for name, indexes in by_session.items():
        if sum(quotas[index] for index in indexes) % EXPOSURE_BATCH_SIZE:
            raise PilotDataError(f"exposure quota is not batch-aligned for {name}")
    return tuple(int(value) for value in quotas)


def _even_cyclic_selection(
    candidates: np.ndarray, *, count: int, schedule: dist.EightTrialSchedule
) -> tuple[int, ...]:
    """Pick `count` different candidates with deterministic full-range spacing."""

    if candidates.ndim != 1 or candidates.size < count:
        raise PilotDataError(
            f"{schedule.session_name} schedule {schedule.start_index}: "
            f"{candidates.size} strict candidates cannot supply {count} unique samples"
        )
    token = hashlib.sha256(canonical_json_bytes(schedule.manifest())).digest()
    offset = int.from_bytes(token[:8], "big") % int(candidates.size)
    positions = (np.floor(np.arange(count, dtype=np.float64) * candidates.size / count).astype(np.int64) + offset)
    positions %= candidates.size
    selected = tuple(int(value) for value in candidates[positions])
    if len(set(selected)) != count:
        raise PilotDataError("exposure cyclic query selector repeated a window")
    return selected


def build_exposure_matched_samples(
    records: Mapping[str, H1PilotRecord], schedules: Sequence[dist.EightTrialSchedule]
) -> ExposureSamplePlan:
    """Build the common 115,520-sample strict-window plan without repetition."""

    quotas = _balanced_schedule_quotas(schedules)
    output: list[dist.DistributionSample] = []
    candidate_counts: list[int] = []
    for index, schedule in enumerate(schedules):
        record = records[schedule.session_name]
        candidates = strict_post_support_window_candidates(record, schedule.query_first_bin)
        candidate_counts.append(int(candidates.size))
        selected = _even_cyclic_selection(candidates, count=quotas[index], schedule=schedule)
        output.extend(dist.DistributionSample(schedule_index=index, window_start=start) for start in selected)
    if len(output) != EXPOSURE_SAMPLES_PER_EPOCH:
        raise PilotDataError("exposure common sample count drift")
    tuples = {(item.schedule_index, item.window_start) for item in output}
    if len(tuples) != len(output):
        raise PilotDataError("exposure plan contains a duplicate schedule/window pair")
    quota_body = [
        {
            "schedule": schedule.manifest(),
            "strict_candidate_count": candidate_counts[index],
            "selected_count": quotas[index],
        }
        for index, schedule in enumerate(schedules)
    ]
    return ExposureSamplePlan(
        samples=tuple(output),
        candidate_counts=tuple(candidate_counts),
        selected_counts=tuple(quotas),
        quota_sha256=canonical_sha256(quota_body),
    )


def persist_exposure_paired_cache(
    cache_dir: str | Path,
    *,
    schedules: Sequence[dist.EightTrialSchedule],
    plan: ExposureSamplePlan,
    s4_raw: np.ndarray,
    q4_raw: np.ndarray,
    normalizer: dist.PairedCarrierNormalizer,
    source_files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Persist a separately named immutable cache for the exposure repair."""

    directory = Path(cache_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    schedule_digest = dist.schedule_sha256(schedules)
    arrays_sha = dist._write_npz_once(  # type: ignore[attr-defined]
        directory / "h1_fresh_distribution_exposure_pair_cache.npz",
        s4_raw=np.asarray(s4_raw, dtype=np.float64),
        q4_raw=np.asarray(q4_raw, dtype=np.float64),
        schedule_indices=np.asarray([sample.schedule_index for sample in plan.samples], dtype=np.int16),
        window_starts=np.asarray([sample.window_start for sample in plan.samples], dtype=np.int64),
        selected_counts=np.asarray(plan.selected_counts, dtype=np.int32),
        candidate_counts=np.asarray(plan.candidate_counts, dtype=np.int32),
    )
    manifest = {
        "schema": EXPOSURE_CACHE_SCHEMA,
        "source_sessions": list(H1_M4_FOLD0_SOURCE),
        "source_files": [dict(value) for value in source_files],
        "source_schedule_sha256": schedule_digest,
        "schedule_count": len(schedules),
        "s4_raw_sha256": array_sha256(np.asarray(s4_raw, dtype=np.float64)),
        "q4_raw_sha256": array_sha256(np.asarray(q4_raw, dtype=np.float64)),
        "paired_cache_file_sha256": arrays_sha,
        "normalizer": normalizer.manifest,
        "sample_plan": plan.manifest(schedules),
        "arms": {
            "s4": "carrier fit on t..t+3; shared first-block identity",
            "q4": "carrier fit on t+4..t+7; source analogue of query-local labels",
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_sha = dist._write_bytes_once(  # type: ignore[attr-defined]
        directory / "h1_fresh_distribution_exposure_pair_cache.manifest.json",
        canonical_json_bytes(manifest),
    )
    return {**manifest, "manifest_file_sha256": manifest_sha}


class H1DistributionExposurePairedBatchSampler(Sampler[list[int]]):
    """The one common 3,610-batch source order for S4e and Q4e."""

    def __init__(self, dataset: dist.H1CarrierIdDistributionDataset) -> None:
        self.dataset = dataset
        grouped: dict[str, list[int]] = {name: [] for name in H1_M4_FOLD0_SOURCE}
        for index, sample in enumerate(dataset.samples):
            grouped[dataset.schedules[sample.schedule_index].session_name].append(index)
        batches: list[list[int]] = []
        for name in H1_M4_FOLD0_SOURCE:
            indexes = grouped[name]
            if len(indexes) % EXPOSURE_BATCH_SIZE:
                raise PilotDataError(f"exposure sampler would tail-drop {name}")
            shuffled = __import__("random").Random(42).sample(indexes, len(indexes))
            batches.extend(
                shuffled[offset : offset + EXPOSURE_BATCH_SIZE]
                for offset in range(0, len(shuffled), EXPOSURE_BATCH_SIZE)
            )
        self.batches = __import__("random").Random(42).sample(batches, len(batches))
        self.flat_indices = np.asarray([item for batch in self.batches for item in batch], dtype=np.int64)
        self.batch_order_sha256 = array_sha256(self.flat_indices)
        self._epoch = 0
        if len(self.batches) != EXPOSURE_BATCHES_PER_EPOCH or self.flat_indices.size != EXPOSURE_SAMPLES_PER_EPOCH:
            raise PilotDataError("exposure sampler throughput differs from the sealed H-C schedule")

    def reset_epoch(self) -> None:
        self._epoch = 0

    def __iter__(self):
        if self._epoch >= EXPOSURE_EPOCHS:
            raise RuntimeError("exposure paired batch order exhausted after fixed epoch 50")
        self._epoch += 1
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


class H1CarrierIdDistributionExposureDataModule(pl.LightningDataModule):
    """Source-only D-S4e/D-Q4e DataModule with H-C-matched exposure."""

    def __init__(
        self,
        *,
        task: str,
        data_dir: str,
        raw_receipt_path: str,
        eb_receipt_path: str,
        cache_dir: str,
        carrier_distribution_arm: str,
        batch_size: int = EXPOSURE_BATCH_SIZE,
        window_size: int = WINDOW,
        calibration_n_trials: int = SUPPORT_TRIALS,
        max_trial_length: int = MAX_TRIAL_LENGTH,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        fixed_epochs: int = EXPOSURE_EPOCHS,
        samples_per_epoch: int = EXPOSURE_SAMPLES_PER_EPOCH,
        **unused: Any,
    ) -> None:
        super().__init__()
        if task != "h1" or carrier_distribution_arm not in EXPOSURE_ARMS:
            raise NormalizedV2ContractError("exposure DataModule requires H1 arm in {'s4','q4'}")
        values = (batch_size, window_size, calibration_n_trials, max_trial_length, num_workers, seed, fixed_epochs, samples_per_epoch)
        expected = (EXPOSURE_BATCH_SIZE, WINDOW, SUPPORT_TRIALS, MAX_TRIAL_LENGTH, 0, 42, EXPOSURE_EPOCHS, EXPOSURE_SAMPLES_PER_EPOCH)
        if values != expected:
            raise NormalizedV2ContractError("exposure matched source protocol parameter drift")
        self.save_hyperparameters(logger=False)
        self.batch_size_per_device = EXPOSURE_BATCH_SIZE
        self._setup_done = False

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit"}:
            raise RuntimeError("exposure distribution DataModule permits source fit only; target is fail-closed")
        if self._setup_done:
            return
        records = load_source_records(self.hparams.data_dir)
        if len(records) != len(H1_M4_FOLD0_SOURCE) or tuple(records) != tuple(H1_M4_FOLD0_SOURCE):
            raise PilotDataError("exposure repair must open exactly the fixed 11 source recordings")
        frozen_plan = reconstruct_frozen_plan(records, self.hparams.raw_receipt_path, self.hparams.eb_receipt_path)
        schedules = dist.build_common_eight_trial_schedules(records)
        if len(schedules) != EXPOSURE_SCHEDULE_COUNT:
            raise PilotDataError("exposure repair must retain exactly the v1 72 paired schedules")
        s4_raw, q4_raw = dist.build_paired_carriers(records, frozen_plan, schedules)
        normalizer = dist.fit_paired_normalizer(
            s4_raw, q4_raw, source_schedule_digest=dist.schedule_sha256(schedules)
        )
        sample_plan = build_exposure_matched_samples(records, schedules)
        source_files = [
            {"session": name, "date": records[name].date, "sha256": records[name].input_sha256,
             "size_bytes": records[name].path.stat().st_size}
            for name in H1_M4_FOLD0_SOURCE
        ]
        cache_manifest = persist_exposure_paired_cache(
            self.hparams.cache_dir,
            schedules=schedules,
            plan=sample_plan,
            s4_raw=s4_raw,
            q4_raw=q4_raw,
            normalizer=normalizer,
            source_files=source_files,
        )
        dataset = dist.H1CarrierIdDistributionDataset(
            records,
            schedules,
            sample_plan.samples,
            normalizer.normalize(s4_raw),
            normalizer.normalize(q4_raw),
            self.hparams.carrier_distribution_arm,
        )
        sampler = H1DistributionExposurePairedBatchSampler(dataset)
        exposure = sample_plan.manifest(schedules)
        self.records, self.plan = records, frozen_plan
        self.schedules, self.samples, self.normalizer = schedules, sample_plan.samples, normalizer
        self.train_dataset, self.train_batch_sampler = dataset, sampler
        self._manifest = {
            "schema": EXPOSURE_SCHEMA,
            "fold_date": FOLD0_DATE,
            "arm": self.hparams.carrier_distribution_arm,
            "source_sessions": list(H1_M4_FOLD0_SOURCE),
            # Parent checkpoint binding historically reads ``files``; retain
            # it and a self-describing alias so source identity is explicit.
            "files": source_files,
            "source_files": source_files,
            "target_sessions_not_opened": ["ses-19250101T111740", "ses-19250101T112404"],
            "target_nwb_opened_during_training_setup": False,
            "minival_or_heldout_enumerated": False,
            "formal_or_evalai_opened_or_enumerated": False,
            "source_schedule_sha256": dist.schedule_sha256(schedules),
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
            "transform_sha256": frozen_plan.transform_sha256,
            "schedule_count": len(schedules),
            "sample_count": len(sample_plan.samples),
            "batches_per_epoch": len(sampler),
            "scheduled_samples_per_epoch": int(sampler.flat_indices.size),
            "epochs": EXPOSURE_EPOCHS,
            "s4_carrier": "t..t+3 standard support fit",
            "q4_label_scope": "t+4..t+7 source query-local leakage diagnostic only",
            "same_arm_inputs_except_carrier": True,
            "exposure_matched_to_h_c": True,
            "exposure_sample_plan": exposure,
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
        raise RuntimeError("exposure distribution target/formal loader is forbidden")

    def predict_dataloader(self):
        raise RuntimeError("exposure distribution target prediction is forbidden")

    def pilot_manifest(self) -> dict[str, Any]:
        if not self._setup_done:
            raise RuntimeError("call setup('fit') before requesting exposure manifest")
        return dict(self._manifest)

    @property
    def pilot_manifest_sha256(self) -> str:
        if not self._setup_done:
            raise RuntimeError("call setup('fit') before requesting exposure manifest SHA")
        return self._manifest_sha256
