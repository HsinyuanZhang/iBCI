"""No-data/no-CUDA tests for the AJPF-C continual-law route."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tfpd_exploration.src.m2_ajpf_c_v1 import chunk_law, plan, score, source_stream


# ---------------------------------------------------------------------------
# chunk law
# ---------------------------------------------------------------------------


def _synthetic_neural(seed: int, bins: int = 1000, units: int = 8):
    rng = np.random.default_rng(seed)
    neural = (rng.random((bins, units)) < 0.1).astype(np.float32)
    neural[300:400] *= 4.0  # one high-rate epoch
    return neural


def test_chunk_law_tumbling_nonoverlap_and_causality():
    neural = _synthetic_neural(0)
    commits = chunk_law.chunk_commit_bins(neural, 0, energy_gated=False)
    assert commits == [99, 199, 299, 399, 499, 599, 699, 799, 999 - 1] or len(commits) == 10
    for a, b in zip(commits, commits[1:]):
        assert b - a == chunk_law.CHUNK_BINS
    counts = chunk_law.committed_before(commits, np.asarray([99, 100, 198, 199]))
    assert counts.tolist() == [0, 1, 1, 1]  # strictly-before: a commit AT t is invisible at t


def test_energy_gate_is_deterministic_and_causal():
    neural = _synthetic_neural(1, bins=2000)
    first = chunk_law.chunk_commit_bins(neural, 0, energy_gated=True)
    second = chunk_law.chunk_commit_bins(neural, 0, energy_gated=True)
    assert first == second
    assert first[0] == 99  # first candidate always commits
    assert len(first) <= 20


def test_seeded_pool_protects_support_and_evicts_fifo():
    seed_rows = [np.full((100, 4), float(i), dtype=np.float32) for i in range(30)]
    pool = chunk_law.SeededPool(seed_rows, support_count=4)
    for i in range(30):
        pool.commit(np.full((100, 4), 100.0 + i, dtype=np.float32))
    stack = pool.stack()
    assert stack.shape == (chunk_law.POOL_CAPACITY, 100, 4)
    assert np.allclose(stack[0], 0.0) and np.allclose(stack[1], 1.0)  # support kept
    assert np.allclose(stack[4], 104.0)  # non-support FIFO: rows 100..103 already evicted


def test_pool_rows_for_state_matches_incremental():
    neural = _synthetic_neural(2, bins=5000)
    commits = chunk_law.chunk_commit_bins(neural, 0, energy_gated=False)
    seed_rows = [np.full((100, 8), float(i), dtype=np.float32) for i in range(30)]
    for count in (0, 1, 5, 40):
        direct = chunk_law.pool_rows_for_state(neural, seed_rows, 4, commits, count)
        pool = chunk_law.SeededPool(seed_rows, 4)
        for commit_bin in commits[:count]:
            pool.commit(neural[commit_bin - 99 : commit_bin + 1])
        assert np.array_equal(direct, pool.stack())


# ---------------------------------------------------------------------------
# source stream grouping
# ---------------------------------------------------------------------------


class _FakeMaterial:
    def __init__(self, session, bins, seed=0):
        rng = np.random.default_rng(seed)
        self.session = session
        self.neural = (rng.random((bins, 6)) < 0.2).astype(np.float32)
        self.targets = rng.normal(size=(bins, 2)).astype(np.float32)
        self.seed_rows = [np.zeros((100, 6), dtype=np.float32) for _ in range(30)]
        self.support_count = 4
        self.normalized_side = np.zeros((6, 4), dtype=np.float32)
        self.selected_indices = [0, 1, 2, 3]
        self.commits = {
            "chunk100": chunk_law.chunk_commit_bins(self.neural, 0, energy_gated=False),
            "chunk100e": chunk_law.chunk_commit_bins(self.neural, 0, energy_gated=True),
        }


def test_coordinate_stream_groups_share_state_and_respect_caps():
    materials = [_FakeMaterial("ses-A", 30_000, seed=3), _FakeMaterial("ses-B", 26_000, seed=4)]
    for law in plan.LAWS:
        groups = source_stream.coordinate_stream(materials, law)
        total = sum(len(group) for group in groups)
        assert len(groups) <= plan.GROUP_BUDGET_PER_EPOCH
        assert total >= 0.4 * 55_804  # V2 relative exposure floor
        for group in groups:
            assert 1 <= len(group) <= plan.BATCH_SIZE
            sessions = {c.session for c in group}
            states = {c.state_index for c in group}
            assert len(sessions) == 1 and len(states) == 1
            starts = [c.window_start for c in group]
            assert starts == sorted(starts)
            for c in group:
                assert c.query_trial_index == c.state_index


def test_coordinate_stream_is_deterministic():
    materials = [_FakeMaterial("ses-A", 60_000, seed=5)]
    one = source_stream.coordinate_stream(materials, "chunk100")
    two = source_stream.coordinate_stream(materials, "chunk100")
    assert [(c.session, c.state_index, c.window_start) for g in one for c in g] == \
           [(c.session, c.state_index, c.window_start) for g in two for c in g]


# ---------------------------------------------------------------------------
# score gates and bootstrap (synthetic)
# ---------------------------------------------------------------------------


def test_bootstrap_ci_and_gate_tiers():
    contrast = score.bootstrap_ci(np.asarray([0.02, 0.01, 0.03, -0.001, 0.015, 0.012]))
    assert contrast["n"] == 6 and contrast["positive"] == 5
    assert contrast["ci_low"] <= contrast["mean"] <= contrast["ci_high"]
    per_session = {f"s{i}": v for i, v in enumerate([0.02, 0.012, -0.02, 0.001, 0.011, 0.013])}
    mean = float(np.mean(list(per_session.values())))
    assert plan.GATE_CANDIDATE_MEAN <= mean
    assert min(per_session.values()) < plan.GATE_PAPER_WORST  # worst session violates paper tier


def test_plan_reuse_literals_match_ajpf_plan():
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import plan as ajpf_plan
    assert plan.AJPF_PLAN_LR["encoder_alpha_lr"] == ajpf_plan.ENCODER_ALPHA_LR
    assert plan.AJPF_PLAN_LR["decoder_lr"] == ajpf_plan.DECODER_LR
    assert plan.BATCH_SIZE == ajpf_plan.BATCH_SIZE
    assert plan.EPOCHS == ajpf_plan.EPOCHS
    assert plan.SEED == ajpf_plan.SEED
    assert plan.TARGET_COORDINATE_BUDGET == 91_717


def test_workorder_and_probe_receipt_exist():
    assert (REPO_ROOT / plan.WORKORDER_RELATIVE).is_file()
    assert (REPO_ROOT / plan.PROBE_RECEIPT_RELATIVE).is_file()
