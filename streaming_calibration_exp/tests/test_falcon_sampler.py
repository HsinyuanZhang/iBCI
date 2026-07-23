from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

from src.data.falcon_datamodule import SessionBatchSampler


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
