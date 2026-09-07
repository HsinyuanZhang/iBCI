"""Pure contract tests for the fresh H1 D-S4/D-Q4 source diagnostic."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data import h1_carrierid_distribution as dist
from src.data.h1_m4_eb_pilot import H1_M4_FOLD0_SOURCE, TrialBlocks
from src.h1_m4_eb_normalized_v2_contract import NormalizedV2ContractError, state_hash
from scripts import h1_carrierid_distribution_preflight as preflight


class _FakeRecord:
    """Small in-memory record sufficient for source-schedule contract tests."""

    def __init__(self, *, session_name: str = "ses-19250108T110520", bad_trial: int | None = None) -> None:
        self.session_name = session_name
        self.date = "19250108"
        self.trial_values = tuple(float(index) for index in range(12))
        bins_per_trial, neurons = 100, 176
        self.neural = np.zeros((len(self.trial_values) * bins_per_trial, neurons), dtype=np.float32)
        self.velocity = np.zeros((self.neural.shape[0], 7), dtype=np.float32)
        self.eval_mask = np.ones(self.neural.shape[0], dtype=bool)
        self.trial_num = np.repeat(np.asarray(self.trial_values), bins_per_trial)
        self.trials = tuple(
            TrialBlocks(
                trial_number=value,
                rates=np.ones((1 if bad_trial == int(value) else 2, neurons), dtype=np.float64),
                velocity=np.full((1 if bad_trial == int(value) else 2, 7), value, dtype=np.float64),
                block_indices=np.zeros((1 if bad_trial == int(value) else 2, 5), dtype=np.int64),
            )
            for value in self.trial_values
        )

    @property
    def num_neurons(self) -> int:
        return 176

    def blocks_for(self, value: float) -> TrialBlocks:
        return self.trials[int(value)]

    def eval_trial_neural(self, value: float) -> np.ndarray:
        return np.zeros((5, 176), dtype=np.float32)


def test_eligibility_requires_one_contiguous_eight_trial_block():
    record = _FakeRecord(bad_trial=10)
    schedules = dist.eligible_eight_trial_schedules(record)
    # Start 2 is also excluded because its strict t+4 query boundary leaves
    # fewer than one 700-bin query window in this compact synthetic record.
    assert [entry.start_index for entry in schedules] == [0, 1]
    assert all(entry.support_values == tuple(float(value) for value in range(entry.start_index, entry.start_index + 4))
               for entry in schedules)
    assert all(entry.query_values == tuple(float(value) for value in range(entry.start_index + 4, entry.start_index + 8))
               for entry in schedules)


def test_query_mutation_changes_q4_carrier_but_not_s4(monkeypatch):
    record = _FakeRecord()
    schedule = dist.eligible_eight_trial_schedules(record)[0]

    def synthetic_fit(value, _plan, trials):
        scalar = sum(value.blocks_for(trial).velocity.mean() for trial in trials)
        return {"carrier": np.full((176, 4), scalar, dtype=np.float64)}

    monkeypatch.setattr(dist, "fit_frozen_carrier", synthetic_fit)
    first_s4, first_q4 = dist._carrier_pair_for_schedule(record, object(), schedule)
    altered = _FakeRecord()
    altered.trials = tuple(
        replace(item, velocity=item.velocity + 19.0) if index == 4 else item
        for index, item in enumerate(altered.trials)
    )
    second_s4, second_q4 = dist._carrier_pair_for_schedule(altered, object(), schedule)
    assert np.array_equal(first_s4, second_s4)
    assert not np.array_equal(first_q4, second_q4)


def test_s4_q4_only_difference_is_carrier_and_window_order_is_common():
    schedules = tuple(
        dist.EightTrialSchedule(
            session_name=name, start_index=0, support_values=(0.0, 1.0, 2.0, 3.0),
            query_values=(4.0, 5.0, 6.0, 7.0), query_first_bin=400,
        )
        for name in H1_M4_FOLD0_SOURCE
    )
    samples = tuple(dist.DistributionSample(index, 400 + offset) for index in range(len(schedules)) for offset in range(64))
    # The sampler needs only these two fields; this keeps the test independent
    # of disk/NWB access while asserting identical orders for the two arms.
    first = SimpleNamespace(samples=samples, schedules=schedules)
    second = SimpleNamespace(samples=samples, schedules=schedules)
    sampler_s4 = dist.H1DistributionPairedBatchSampler(first, batch_size=32, seed=42, max_epochs=50)
    sampler_q4 = dist.H1DistributionPairedBatchSampler(second, batch_size=32, seed=42, max_epochs=50)
    assert sampler_s4.batch_order_sha256 == sampler_q4.batch_order_sha256
    assert sampler_s4.batches == sampler_q4.batches
    raw_s4 = np.zeros((len(schedules), 176, 4), dtype=np.float64)
    raw_q4 = np.ones_like(raw_s4)
    normalizer = dist.fit_paired_normalizer(raw_s4, raw_q4, source_schedule_digest=dist.schedule_sha256(schedules))
    assert normalizer.manifest["source_entries"] == len(schedules)
    assert dist.sample_sha256(samples, schedules) == dist.sample_sha256(samples, schedules)
    assert not np.array_equal(normalizer.normalize(raw_s4), normalizer.normalize(raw_q4))


def test_target_access_is_fail_closed_without_opening_any_recording(tmp_path):
    datamodule = dist.H1CarrierIdDistributionDataModule(
        task="h1", data_dir=str(tmp_path), raw_receipt_path=str(tmp_path / "raw.json"),
        eb_receipt_path=str(tmp_path / "eb.json"), cache_dir=str(tmp_path / "cache"),
        carrier_distribution_arm="s4",
    )
    with pytest.raises(RuntimeError, match="source fit only"):
        datamodule.setup("test")
    source = (Path(__file__).resolve().parents[1] / "src/data/h1_carrierid_distribution.py").read_text(encoding="utf-8")
    assert "load_target_records" not in source
    assert "H1M4EBStrictTargetDataset" not in source


def test_common_seed_produces_exact_same_initial_state():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    left = preflight._new("s4")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    right = preflight._new("q4")
    assert state_hash(left.state_dict()) == state_hash(right.state_dict())


def test_new_cache_writer_refuses_overwrite_or_drift(tmp_path):
    path = tmp_path / "new_distribution_only" / "artifact.bin"
    first = dist._write_bytes_once(path, b"fresh-content")
    assert path.stat().st_mode & 0o777 == 0o444
    assert first == dist._write_bytes_once(path, b"fresh-content")
    with pytest.raises(NormalizedV2ContractError, match="overwrite refused"):
        dist._write_bytes_once(path, b"other-content")
