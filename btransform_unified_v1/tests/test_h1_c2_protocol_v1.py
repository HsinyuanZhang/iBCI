"""C2-protocol selection and prefix-cycle contracts (CPU only)."""
from __future__ import annotations

import numpy as np
import pytest

from btransform_unified_v1.c2_protocol import (
    C2_CYCLE,
    C2_TIE_BREAK,
    HO_SELECTION_METRIC,
    grouped_session_metrics,
    pick_m7_start,
    prefix_schedule,
    select_epoch,
    starts_for_budget,
)


def test_prefix_schedule_covers_cycle_and_is_balanced() -> None:
    n_batches = 731
    for epoch0 in (0, 7, 31):
        row = prefix_schedule(epoch0, n_batches)
        assert len(row) == n_batches
        assert set(row) == set(C2_CYCLE)
        counts = {m: row.count(m) for m in C2_CYCLE}
        assert max(counts.values()) - min(counts.values()) <= 1
        assert prefix_schedule(epoch0, n_batches) == row


def test_prefix_schedule_epoch_phase_rotates() -> None:
    rows = [prefix_schedule(e, 8) for e in range(16)]
    assert any(rows[i] != rows[j] for i in range(16) for j in range(i + 1, 16))
    assert all(row[0] in C2_CYCLE for row in rows)


def test_select_epoch_tie_break_higher_mean() -> None:
    rows = [
        {"epoch_zero_based": 0, HO_SELECTION_METRIC: 0.10, "worst_session_r2": 0.05, "session_std_population": 0.01},
        {"epoch_zero_based": 1, HO_SELECTION_METRIC: 0.20, "worst_session_r2": 0.01, "session_std_population": 0.20},
        {"epoch_zero_based": 2, HO_SELECTION_METRIC: 0.15, "worst_session_r2": 0.14, "session_std_population": 0.01},
    ]
    pick = select_epoch(rows)
    assert pick["epoch_zero_based"] == 1


def test_select_epoch_tie_break_worst_then_std_then_earlier() -> None:
    rows = [
        {"epoch_zero_based": 10, HO_SELECTION_METRIC: 0.40, "worst_session_r2": 0.10, "session_std_population": 0.05},
        {"epoch_zero_based": 4, HO_SELECTION_METRIC: 0.40, "worst_session_r2": 0.20, "session_std_population": 0.08},
        {"epoch_zero_based": 7, HO_SELECTION_METRIC: 0.40, "worst_session_r2": 0.20, "session_std_population": 0.08},
        {"epoch_zero_based": 11, HO_SELECTION_METRIC: 0.40, "worst_session_r2": 0.20, "session_std_population": 0.02},
    ]
    pick = select_epoch(rows)
    assert pick["epoch_zero_based"] == 11
    rows[3]["session_std_population"] = 0.08
    pick = select_epoch(rows)
    assert pick["epoch_zero_based"] == 4


def test_select_epoch_requires_complete_rows() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        select_epoch([{"epoch_zero_based": 0, HO_SELECTION_METRIC: 0.1}])


def test_grouped_session_metrics_matches_c2_aggregation() -> None:
    rng = np.random.default_rng(0)
    predictions = {}
    targets = {}
    masks = {}
    mapping = (("ses-a", "S6_set_1"), ("ses-b", "S6_set_2"), ("ses-c", "S7_set_1"), ("ses-d", "S7_set_2"))
    for _session, key in mapping:
        n = 32
        tgt = rng.normal(size=(n, 7))
        predictions[key] = tgt + 0.1 * rng.normal(size=(n, 7))
        targets[key] = tgt
        masks[key] = np.ones(n, dtype=bool)
    metrics = grouped_session_metrics(predictions, targets, masks, mapping)
    assert set(metrics["per_session_r2"]) == {"S6", "S7"}
    assert abs(metrics["r2_mean"] - float(np.mean(list(metrics["per_session_r2"].values())))) < 1e-12
    assert abs(metrics["r2_std_population"] - float(np.std(list(metrics["per_session_r2"].values()), ddof=0))) < 1e-12
    assert metrics["worst_session_r2"] == min(metrics["per_session_r2"].values())
    assert C2_TIE_BREAK == ("higher mean", "higher worst-session", "lower population std", "earlier epoch")


def test_starts_for_budget_requires_full_m7_block() -> None:
    assert starts_for_budget([0, 1, 2, 8], n_trials=12, budget=7) == (0, 1, 2)
    assert starts_for_budget([0, 1, 8], n_trials=12, budget=4) == (0, 1, 8)
    assert starts_for_budget([0, 1, 8], n_trials=12, budget=3) == (0, 1, 8)


def test_pick_m7_start_is_deterministic_and_varies_by_step() -> None:
    starts = (0, 1, 2, 3, 4)
    a = pick_m7_start("ses-19250101T111740", epoch0=0, step=0, starts=starts)
    b = pick_m7_start("ses-19250101T111740", epoch0=0, step=0, starts=starts)
    c = pick_m7_start("ses-19250101T111740", epoch0=0, step=1, starts=starts)
    assert a == b
    assert a in starts
    assert c in starts
    assert a != c or pick_m7_start("ses-19250101T111740", epoch0=1, step=0, starts=starts) != a
