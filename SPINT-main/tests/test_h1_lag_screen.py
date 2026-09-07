"""Synthetic no-data contracts for the lag screen (CPU-only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data import h1_lag_screen as ls


def test_lag_range_covers_pm10():
    assert ls.LAG_RANGE == tuple(range(-10, 11))
    assert len(ls.LAG_RANGE) == 21


def test_null_offsets_outside_lag_set_and_at_least_1s():
    for off in ls.NULL_OFFSETS:
        assert off >= 50  # at least 1 second at 20ms bins
        assert off not in ls.LAG_RANGE


def test_build_plan_requires_two_recordings_min():
    with pytest.raises(Exception):
        ls.build_plan({})


    # Minimal synthetic records with required attributes
    from types import SimpleNamespace
    rng = np.random.default_rng(42)
    records = {}
    for i in range(4):
        trials = []
        for t in range(7):
            n = 20
            trials.append(SimpleNamespace(
                trial_number=float(t + 1),
                rates=rng.uniform(1, 10, (n, 176)),
                velocity=rng.normal(size=(n, 7)),
                block_indices=np.tile(np.arange(t * 100, t * 100 + n)[:, None], (1, 5)),
            ))
        records[f"ses-r{i}"] = SimpleNamespace(
            session_name=f"ses-r{i}", date=f"1925010{i+1}",
            neural=rng.uniform(0, 1, (700, 176)),
            velocity=rng.normal(size=(700, 7)),
            eval_mask=np.ones(700, dtype=bool),
            trial_num=np.concatenate([np.full(100, float(t+1)) for t in range(7)]),
            trials=trials,
        )
    # This should NOT raise — 4 recordings is enough for the grid search
    plan = ls.build_plan(records)
    assert plan.q in ls.Q_GRID
    assert plan.ridge_lambda in ls.LAMBDA_GRID


def test_lagged_blocks_drops_trial_boundary_crossers():
    from types import SimpleNamespace
    # A single trial occupying bins 10..29
    trial = SimpleNamespace(
        trial_number=1.0,
        rates=np.ones((4, 176)),
        velocity=np.zeros((4, 7)),
        block_indices=np.array([[10,11,12,13,14],[15,16,17,18,19],[20,21,22,23,24],[25,26,27,28,29]]),
    )
    record = SimpleNamespace(
        session_name="ses-test", trials=(trial,),
        neural=np.ones((30, 176)),
        velocity=np.ones((30, 7)),
        trial_num=np.zeros(30),
        eval_mask=np.ones(30, dtype=bool),
    )
    record.trial_num[10:30] = 1.0

    # tau=0: all 4 blocks valid
    b0 = ls.build_lagged_blocks(record, 0)
    assert b0.rates.shape[0] == 4
    assert b0.n_dropped == 0

    # tau=5: blocks at indices 25-29 shift to 30-34, out of trial -> dropped
    b5 = ls.build_lagged_blocks(record, 5)
    assert b5.n_dropped > 0
    assert b5.rates.shape[0] < 4


def test_arm_p_r2_returns_finite_on_synthetic():
    rng = np.random.default_rng(7)
    n, q = 100, 4
    mean = rng.normal(size=176)
    scale = np.ones(176)
    pcs = rng.normal(size=(16, 176))
    plan = ls.LagScreenPlan(mean=mean, scale=scale, pcs=pcs, q=q, ridge_lambda=1.0, source_grid_r2=0.5)
    rates = rng.uniform(1, 10, (n, 176))
    z = plan.project(rates)
    design = np.column_stack((np.ones(n), z))
    true_beta = rng.normal(size=(q + 1, 7))
    velocity = design @ true_beta + rng.normal(scale=0.01, size=(n, 7))
    blocks = ls.LaggedBlocks(tau=0, rates=rates, velocity=velocity, n_total=n, n_dropped=0)
    r2 = ls.arm_p_r2(blocks, plan, 1.0)
    assert np.isfinite(r2) and r2 > 0.9


def test_row_cosine_returns_one_for_identical():
    from src.data.h1_pooling_screen import row_cosine
    rng = np.random.default_rng(5)
    rows = rng.normal(size=(176, 7))
    assert abs(row_cosine(rows, rows) - 1.0) < 1e-10


def test_receipt_refuses_overwrite(tmp_path: Path):
    import json
    p = tmp_path / "receipt.json"
    result = {"schema": ls.LAG_SCREEN_SCHEMA, "test": True}
    ls.write_receipt(p, result)
    assert p.is_file()
    with pytest.raises(FileExistsError):
        ls.write_receipt(p, result)
