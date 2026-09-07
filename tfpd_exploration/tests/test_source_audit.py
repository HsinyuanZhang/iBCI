"""Gate-0 tests for the Z4 boundary pilot's source-audit split and T_pre selector.

Covers HANDOFF_CARRIER_ADMISSION_CURRICULUM_20260816.md section 5 / section 11:

- determinism: the split is a pure function of (roster, window grid, frozen
  namespace) — rebuilds bit-identically, and depends on no RNG or dict order;
- 5% proportion per strict-27 train session with the frozen floor/min rule;
- no within/external/formal session can enter the audit surface;
- audit windows are excluded from the pilot's gradient index set, and the
  script's session-grouped batch sampler never yields an audit position;
- byte binding is deterministic and window-covering;
- the selector is the sealed earliest-near-best rule and refuses partial,
  non-finite, or held-out-column inputs.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tfpd_lane import source_audit as sa


ROSTER = tuple(f"sub-C_ses-CO-train{i:02d}" for i in range(27))
WITHIN_DEV = tuple(f"sub-C_ses-CO-dev{i}" for i in range(6))


def _grid(roster, starts_per_session):
    return [
        (session, int(start))
        for session in roster
        for start in np.linspace(0, 100000, starts_per_session, dtype=np.int64)
    ]


def test_window_digest_is_namespace_frozen_and_start_exact():
    d1 = sa.window_digest("ses", 100)
    assert d1 == sa.window_digest("ses", 100)
    assert d1 != sa.window_digest("ses", 101)
    assert d1 != sa.window_digest("other", 100)
    assert d1 != sa.window_digest("ses", 100, namespace="tfpd_pilot_v2")
    assert len(d1) == 64


def test_audit_count_floor_five_percent_with_minimum_one():
    assert sa.audit_count(1000) == 50
    assert sa.audit_count(100) == 5
    assert sa.audit_count(39) == 1  # floor(1.95) = 1
    assert sa.audit_count(10) == 1  # floor(0.5) = 0 -> minimum one
    with pytest.raises(ValueError):
        sa.audit_count(0)
    with pytest.raises(ValueError):
        sa.audit_count(-5)


def test_split_deterministic_and_five_percent_per_session():
    grid = _grid(ROSTER, 997)
    plan_a = sa.build_audit_plan(grid, ROSTER)
    plan_b = sa.build_audit_plan(list(grid), list(ROSTER))
    assert plan_a.plan_sha256 == plan_b.plan_sha256
    assert plan_a.audit_keys == plan_b.audit_keys
    assert plan_a.audit_positions == plan_b.audit_positions

    # 5% per session: floor(0.05 * 997) = 49 windows out of 997
    for row in plan_a.per_session:
        assert row["n_windows"] == 997
        assert row["n_audit"] == 49
        assert row["audit_fraction_of_session"] == pytest.approx(0.049, abs=1e-3)
    assert len(plan_a.audit_keys) == 27 * 49

    # membership is a pure function of the (session, start) set: the selected
    # windows per session do not depend on RNG state or surrounding calls
    random.Random(7).random()  # burn RNG state; the split must not see it
    plan_c = sa.build_audit_plan(grid, ROSTER)
    assert plan_c.audit_keys == plan_a.audit_keys


def test_audit_membership_is_pure_hash_selection():
    # every selected window's digest ranks within its session's first k digests
    grid = _grid(ROSTER, 400)
    plan = sa.build_audit_plan(grid, ROSTER)
    k = sa.audit_count(400)
    for session in ROSTER:
        session_starts = [start for s, start in grid if s == session]
        ranked = sorted(session_starts, key=lambda s: (sa.window_digest(session, s), s))
        expected = set(ranked[:k])
        selected = {start for s, start in plan.audit_keys if s == session}
        assert selected == expected


def test_non_roster_session_is_refused_before_selection():
    grid = _grid(ROSTER, 100) + [("sub-C_ses-CO-dev0", 0)]
    with pytest.raises(ValueError, match="non-roster session"):
        sa.build_audit_plan(grid, ROSTER)
    # within-dev names as the roster are refused as audit sources outright
    with pytest.raises(ValueError):
        sa.build_audit_plan(_grid(WITHIN_DEV, 100), ROSTER)


def test_assert_no_forbidden_sessions():
    sa.assert_no_forbidden_sessions(ROSTER, WITHIN_DEV)
    with pytest.raises(ValueError, match="forbidden sessions"):
        sa.assert_no_forbidden_sessions(ROSTER + (WITHIN_DEV[0],), WITHIN_DEV)


def test_audit_windows_excluded_from_training_indices():
    grid = _grid(ROSTER, 200)
    plan = sa.build_audit_plan(grid, ROSTER)
    audit = set(plan.audit_positions)
    train = set(plan.train_positions)
    assert audit.isdisjoint(train)
    assert audit | train == set(range(len(grid)))
    assert len(audit) == 27 * sa.audit_count(200)
    # the audit keys point at exactly the audited grid rows
    for position, (session, start) in zip(plan.audit_positions, plan.audit_keys):
        assert grid[position] == (session, start)


def test_script_sampler_never_yields_audit_positions():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_z4_boundary_pilot_under_test",
        ROOT / "scripts" / "run_z4_boundary_pilot.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeDataset:
        def __init__(self, grid):
            self.window_indices = [(s, int(t)) for s, t in grid]

    grid = _grid(ROSTER[:4], 130)
    dataset = FakeDataset(grid)
    plan = sa.build_audit_plan(grid, ROSTER[:4])
    sampler = module.SourceTrainBatchSampler(
        dataset, plan.train_positions, batch_size=32, shuffle=True, seed=42
    )
    audit = set(plan.audit_positions)
    yielded = [idx for batch in sampler for idx in batch]
    assert yielded
    assert audit.isdisjoint(yielded)
    # every yielded index is a real non-audit grid position
    assert set(yielded) <= set(plan.train_positions)
    # session grouping + drop-last semantics (130 -> 4 full batches of 32)
    for batch in sampler:
        sessions = {dataset.window_indices[idx][0] for idx in batch}
        assert len(sessions) == 1
        assert len(batch) == 32
    # deterministic across constructions (fixed permutation, like the official cells)
    again = module.SourceTrainBatchSampler(
        dataset, plan.train_positions, batch_size=32, shuffle=True, seed=42
    )
    assert list(sampler) == list(again)


class _FakeRecord:
    def __init__(self, seed, n_units=6, n_bins=500):
        rng = np.random.default_rng(seed)
        self.neural = rng.integers(0, 3, size=(n_bins, n_units)).astype(np.float32)
        self.behavior = rng.normal(size=(n_bins, 2)).astype(np.float32)
        self.calib_trials = rng.normal(size=(30, 100, n_units)).astype(np.float32)
        self.side_features = np.zeros((n_units, 4), dtype=np.float32)


class _FakeDataset:
    def __init__(self, grid):
        self.window_indices = [(s, int(t)) for s, t in grid]
        names = sorted({s for s, _ in grid})
        self.sessions = {s: _FakeRecord(i) for i, s in enumerate(names)}


def test_byte_binding_deterministic_and_covering():
    grid = [
        (session, int(start))
        for session in ROSTER[:3]
        for start in range(0, 400, 50)
    ]
    plan = sa.build_audit_plan(grid, ROSTER[:3])
    dataset = _FakeDataset(grid)
    binding = sa.bind_audit_windows(plan, dataset, window_size=50)
    again = sa.bind_audit_windows(plan, dataset, window_size=50)
    assert binding["binding_sha256"] == again["binding_sha256"]
    assert len(binding["windows"]) == len(plan.audit_keys)
    for entry, (session, start) in zip(binding["windows"], plan.audit_keys):
        assert entry["session"] == session and entry["window_start"] == start
        assert entry["window_end"] - entry["window_start"] == 50
        assert entry["receptive_field_coverage_bins"] == [start, start + 50]
        assert len(entry["neural_sha256"]) == 64
        assert len(entry["behavior_label_sha256"]) == 64
    # byte sensitivity: touching one audited behavior row changes the binding
    first = plan.audit_keys[0]
    dataset.sessions[first[0]].behavior[first[1]] += 1.0
    drifted = sa.bind_audit_windows(plan, dataset, window_size=50)
    assert drifted["binding_sha256"] != binding["binding_sha256"]
    # a truncated window (start too close to the session end) is refused
    grid_bad = grid + [(ROSTER[3], 480)]
    plan_bad = sa.build_audit_plan(grid_bad, ROSTER[:4])
    ds_bad = _FakeDataset([(s, t) for s, t in grid_bad])
    ds_bad.sessions[ROSTER[3]].neural = ds_bad.sessions[ROSTER[3]].neural[:490]
    with pytest.raises(ValueError, match="does not cover"):
        sa.bind_audit_windows(plan_bad, ds_bad, window_size=50)


def test_select_t_pre_earliest_near_best():
    # gentle monotone slope (0.001/epoch): the earliest endpoint within 0.005 of
    # the maximum (epoch 10) wins, not the argmax itself (epoch 15)
    scores = {e: 0.10 + 0.001 * e for e in range(16)}
    out = sa.select_t_pre(scores)
    assert out["s_max"] == scores[15]
    assert out["selected_epoch"] == 10
    assert out["t_pre"] == 11
    assert out["e_t4"] == 37

    # steep monotone slope (0.01/epoch): only the argmax is within tolerance
    scores = {e: 0.10 + 0.01 * e for e in range(16)}
    out = sa.select_t_pre(scores)
    assert out["selected_epoch"] == 15
    assert out["t_pre"] == 16
    assert out["e_t4"] == 32

    # late max, but an early endpoint sits within the 0.005 tolerance
    scores = {e: 0.20 for e in range(16)}
    scores[14] = 0.204  # s_max = 0.204, threshold = 0.199 -> epoch 0 qualifies
    out = sa.select_t_pre(scores)
    assert out["selected_epoch"] == 0 and out["t_pre"] == 1 and out["e_t4"] == 47

    # an early endpoint just OUTSIDE the tolerance is skipped
    scores = {e: 0.20 for e in range(16)}
    scores[0] = 0.1948  # s_max - 0.005 = 0.199 > 0.1948
    scores[1] = 0.199
    out = sa.select_t_pre(scores)
    assert out["selected_epoch"] == 1 and out["t_pre"] == 2 and out["e_t4"] == 46

    # exact-tolerance boundary qualifies (>=): all other early endpoints sit
    # strictly below s_max - 0.005, epoch 3 equals it exactly
    scores = {e: 0.1949 for e in range(16)}
    scores[3] = 0.195  # exactly s_max - 0.005
    scores[15] = 0.200
    out = sa.select_t_pre(scores)
    assert out["selected_epoch"] == 3 and out["t_pre"] == 4 and out["e_t4"] == 44


def test_select_t_pre_refuses_incomplete_nonfinite_and_heldout_inputs():
    with pytest.raises(ValueError, match="candidate epochs"):
        sa.select_t_pre({e: 0.1 for e in range(15)})
    with pytest.raises(ValueError, match="candidate epochs"):
        sa.select_t_pre({e: 0.1 for e in range(17)})
    bad = {e: 0.1 for e in range(16)}
    bad[7] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        sa.select_t_pre(bad)
    # six held-out scores can never masquerade as the source column
    with pytest.raises(ValueError, match="candidate epochs"):
        sa.select_t_pre({e: 0.1 for e in range(6)})
