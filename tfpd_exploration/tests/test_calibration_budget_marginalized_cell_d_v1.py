from __future__ import annotations

import random
from collections.abc import Iterator

import numpy as np
import torch

from src import calibration_budget_marginalized_cell_d_v1 as cbm


def test_each_session_gets_every_integer_budget_before_repeat() -> None:
    for epoch in (0, 1, 47):
        for session in range(27):
            values = [cbm.budget_for_batch(epoch=epoch, session_index=session, cycle=cycle) for cycle in range(27)]
            assert sorted(values) == list(cbm.BUDGETS)


def test_schedule_does_not_consume_host_rng() -> None:
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    _ = cbm.budget_schedule_digest()
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_source_cache_requires_all_budgets_and_preserves_rows() -> None:
    roster = tuple(f"s{index:02d}" for index in range(27))
    rows = {
        session: {budget: torch.full((5, 4), float(budget), dtype=torch.float32) for budget in cbm.BUDGETS}
        for session in roster
    }
    evidence = {session: {"unit_count": 5} for session in roster}
    cache = cbm.SourceBudgetSideCache(roster, rows, evidence)
    assert torch.equal(cache.side(session=roster[3], budget=10, device="cpu"), rows[roster[3]][10])
    assert cache.payload()["posterior_used"] is False


def test_dry_plan_is_performance_cell_and_target_free() -> None:
    plan = cbm.dry_plan()
    assert plan["budgets"] == list(range(4, 31))
    assert plan["epochs"] == 48
    assert plan["target_optimizer_steps"] == 0


def test_real_cell_d_train_step_jointly_uses_selected_prefix_and_one_dropout() -> None:
    from src import cell_d_equal_session_v1 as equal
    from src.tfpd_lane import pop_robust

    class OneBatchSchedule:
        def __len__(self) -> int:
            return 1

        def iter_batches(self) -> Iterator[equal.ScheduledBatch]:
            yield equal.ScheduledBatch(
                epoch=0, position=0, session="s00", cycle=0, cycle_batch=0,
                dataset_indices=tuple(range(32)),
            )

    torch.manual_seed(12)
    model = pop_robust.build_population_robustness_model(seed=42, cell="D").train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    sampler = equal._OneIteratorEpochBatchSampler(OneBatchSchedule())
    assert list(iter(sampler)) == [list(range(32))]  # populate the reviewed FIFO evidence
    units = 5
    ordinary = torch.randn(32, units, 4)
    batch = (
        torch.randn(32, 50, units), torch.randn(32, 50, 2),
        torch.randn(32, 30, 100, units), tuple("s00" for _ in range(32)), ordinary,
    )
    cache = {"s00": {budget: torch.randn(units, 4) for budget in cbm.BUDGETS}}
    cache["s00"][30] = ordinary[0].clone()
    runtime = {
        "torch": torch, "device": torch.device("cpu"), "epoch_iterator": iter((batch,)),
        "epoch_sampler": sampler, "model": model, "optimizer": optimizer,
        "pop_robust": pop_robust, "cbm_session_index": {"s00": 0},
        "cbm_side_device": cache,
    }
    flags = equal.LifecycleFlags(source_train_opened=True)
    backend = object.__new__(cbm.CBMPhysicalBackend)
    outcome = backend.train_step(
        runtime, global_step=0, expected_lr=1e-4, require_full_proof=False, flags=flags,
    )
    assert outcome.batch_evidence["calibration_budget"] == cbm.budget_for_batch(
        epoch=0, session_index=0, cycle=0,
    )
    assert outcome.batch_evidence["b3s_trials"] == outcome.batch_evidence["ordinary_ols_t4_trials"]
    assert flags.backward_calls == flags.update_calls == 1
    assert outcome.population_examples == 32
