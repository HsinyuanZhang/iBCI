from __future__ import annotations

import inspect
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.data.falcon_datamodule import FalconDataModule, SessionBatchSampler


@dataclass
class _Dataset:
    window_indices: list[tuple[str, int]]


def test_session_batch_sampler_uses_requested_seed_without_mixing_sessions():
    dataset = _Dataset(
        [(session, index) for session in ("s1", "s2") for index in range(12)]
    )

    batches_42_a = list(SessionBatchSampler(dataset, batch_size=3, shuffle=True, seed=42))
    batches_42_b = list(SessionBatchSampler(dataset, batch_size=3, shuffle=True, seed=42))
    batches_43 = list(SessionBatchSampler(dataset, batch_size=3, shuffle=True, seed=43))

    assert batches_42_a == batches_42_b
    assert batches_43 != batches_42_a
    for batch in batches_42_a + batches_43:
        sessions = {dataset.window_indices[index][0] for index in batch}
        assert len(batch) == 3
        assert len(sessions) == 1


def test_session_batch_sampler_balances_sessions_without_changing_epoch_length():
    dataset = _Dataset(
        [("long", index) for index in range(18)]
        + [("medium", index) for index in range(12)]
        + [("short", index) for index in range(6)]
    )
    unbalanced = SessionBatchSampler(dataset, batch_size=3, shuffle=True, seed=42)
    balanced = SessionBatchSampler(
        dataset,
        batch_size=3,
        shuffle=True,
        seed=42,
        balance_sessions=True,
    )

    balanced_batches = list(balanced)
    counts = Counter(dataset.window_indices[batch[0]][0] for batch in balanced_batches)

    assert len(balanced_batches) == len(unbalanced) == 12
    assert max(counts.values()) - min(counts.values()) <= 1
    for batch in balanced_batches:
        assert len({dataset.window_indices[index][0] for index in batch}) == 1


def test_session_batch_sampler_can_reshuffle_reproducibly_each_epoch():
    dataset = _Dataset(
        [(session, index) for session in ("s1", "s2") for index in range(12)]
    )
    sampler_a = SessionBatchSampler(
        dataset, batch_size=3, shuffle=True, seed=43, reshuffle_each_epoch=True
    )
    sampler_b = SessionBatchSampler(
        dataset, batch_size=3, shuffle=True, seed=43, reshuffle_each_epoch=True
    )

    epoch_0_a, epoch_1_a = list(sampler_a), list(sampler_a)
    epoch_0_b, epoch_1_b = list(sampler_b), list(sampler_b)

    assert epoch_0_a == epoch_0_b
    assert epoch_1_a == epoch_1_b
    assert epoch_0_a != epoch_1_a


def test_session_batch_sampler_supports_tempered_balance_strength():
    dataset = _Dataset(
        [("long", index) for index in range(18)]
        + [("medium", index) for index in range(12)]
        + [("short", index) for index in range(6)]
    )
    tempered = SessionBatchSampler(
        dataset,
        batch_size=3,
        shuffle=False,
        balance_sessions=0.5,
    )

    counts = Counter(
        dataset.window_indices[batch[0]][0] for batch in list(tempered)
    )
    assert counts == {"long": 5, "medium": 4, "short": 3}
    assert len(tempered) == 12


def _unbalanced_dataset() -> _Dataset:
    return _Dataset(
        [("long", index) for index in range(18)]
        + [("medium", index) for index in range(12)]
        + [("short", index) for index in range(6)]
    )


def _legacy_session_batches(dataset: _Dataset, batch_size: int, *, shuffle: bool, seed: int) -> list[list[int]]:
    """Independent reconstruction of the pre-balance remainder-dropping sampler."""
    session_to_indices: dict[str, list[int]] = {}
    for idx, (session_name, _) in enumerate(dataset.window_indices):
        session_to_indices.setdefault(session_name, []).append(idx)
    batched_indices: list[list[int]] = []
    for session_indices in session_to_indices.values():
        indices = list(session_indices)
        if shuffle:
            indices = random.Random(seed).sample(indices, len(indices))
        for i in range(0, len(indices), batch_size):
            batch = indices[i : i + batch_size]
            if len(batch) == batch_size:
                batched_indices.append(batch)
    if shuffle:
        batched_indices = random.Random(seed).sample(batched_indices, len(batched_indices))
    return batched_indices


def test_disabled_balance_is_bitwise_identical_to_legacy_remainder_dropping_sampler() -> None:
    dataset = _unbalanced_dataset()
    for shuffle, seed in ((False, 42), (True, 42), (True, 43)):
        legacy = _legacy_session_batches(dataset, 3, shuffle=shuffle, seed=seed)
        default = list(SessionBatchSampler(dataset, batch_size=3, shuffle=shuffle, seed=seed))
        explicit_false = list(
            SessionBatchSampler(
                dataset, batch_size=3, shuffle=shuffle, seed=seed, balance_sessions=False
            )
        )
        zero_int = list(
            SessionBatchSampler(
                dataset, batch_size=3, shuffle=shuffle, seed=seed, balance_sessions=0
            )
        )
        zero_float = list(
            SessionBatchSampler(
                dataset,
                batch_size=3,
                shuffle=shuffle,
                seed=seed,
                balance_sessions=0.0,
                window_budget_per_session=None,
            )
        )
        assert default == legacy
        assert explicit_false == legacy
        assert zero_int == legacy
        assert zero_float == legacy


def test_equal_session_interpolation_changes_counts_only_when_enabled() -> None:
    dataset = _unbalanced_dataset()
    disabled = SessionBatchSampler(dataset, batch_size=3, shuffle=False, seed=42)
    enabled = SessionBatchSampler(
        dataset, batch_size=3, shuffle=False, seed=42, balance_sessions=True
    )
    disabled_counts = Counter(
        dataset.window_indices[batch[0]][0] for batch in list(disabled)
    )
    enabled_counts = Counter(
        dataset.window_indices[batch[0]][0] for batch in list(enabled)
    )
    assert disabled.original_session_batch_counts == {"long": 6, "medium": 4, "short": 2}
    assert disabled.session_batch_counts == disabled.original_session_batch_counts
    assert disabled_counts == {"long": 6, "medium": 4, "short": 2}
    assert enabled_counts == {"long": 4, "medium": 4, "short": 4}
    assert list(disabled) != list(enabled)
    assert len(disabled) == len(enabled) == 12


def test_falcon_datamodule_exposes_existing_balance_lever_default_off_and_does_not_route_window_budget() -> None:
    signature = inspect.signature(FalconDataModule.__init__)
    assert signature.parameters["balance_session_batches"].default is False
    assert "window_budget_per_session" not in signature.parameters
    sampler_signature = inspect.signature(SessionBatchSampler.__init__)
    assert sampler_signature.parameters["balance_sessions"].default is False
    assert sampler_signature.parameters["window_budget_per_session"].default is None

    source_text = (
        Path(__file__).resolve().parents[1] / "src/data/falcon_datamodule.py"
    ).read_text(encoding="utf-8")
    train_block_start = source_text.index("self.train_batch_sampler = SessionBatchSampler(")
    val_line = "self.val_heldin_batch_sampler = SessionBatchSampler(self.val_heldin_dataset, self.batch_size_per_device, shuffle=False)"
    train_block = source_text[train_block_start : source_text.index(val_line)]
    assert "balance_sessions=self.hparams.balance_session_batches" in train_block
    assert "window_budget_per_session" not in train_block
    assert val_line in source_text
    assert "balance_sessions" not in source_text[source_text.index(val_line) : source_text.index(val_line) + 200]
