"""Tests for the independent H1 tag-free position-context verifier."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "sua_exploration/scripts/verify_h1_tagfree_position_context.py"


def load_verify_module():
    spec = importlib.util.spec_from_file_location("verify_h1_tagfree_position_context", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verify():
    return load_verify_module()


def test_ridge_solve_matches_closed_form_and_leaves_intercept_unpenalised(verify):
    rng = np.random.default_rng(0)
    n, rank, channels = 12, 4, 6
    z = rng.normal(size=(n, rank))
    y = rng.normal(size=(n, channels))
    design = np.column_stack((np.ones(n), z))
    penalty = np.diag([0.0, 1.0, 1.0, 1.0, 1.0]) * (n * verify.TARGET_RIDGE_LAMBDA)
    expected_coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    expected_carrier = np.column_stack((expected_coef[1:].T, expected_coef[0]))

    carrier = verify.solve_target_ridge(z, y)
    assert np.allclose(carrier, expected_carrier, atol=1.0e-12)

    penalty_matrix = np.diag([0.0, 1.0, 1.0, 1.0, 1.0]) * (n * verify.TARGET_RIDGE_LAMBDA)
    assert penalty_matrix[0, 0] == 0.0
    assert np.all(penalty_matrix.diagonal()[1:] == n * verify.TARGET_RIDGE_LAMBDA)


def test_r2_by_channel_hand_computed_and_zero_variance_is_nan(verify):
    observed = np.array([[1.0, 3.0], [3.0, 3.0], [5.0, 3.0]], dtype=np.float64)
    predicted = np.array([[1.5, 1.0], [2.5, 4.0], [4.5, 5.0]], dtype=np.float64)

    total = np.sum(np.square(observed - observed.mean(axis=0, keepdims=True)), axis=0)
    residual = np.sum(np.square(observed - predicted), axis=0)
    expected = np.array([1.0 - residual[0] / total[0], np.nan], dtype=np.float64)

    output = verify.r2_by_channel(observed, predicted)
    assert np.allclose(output[0], expected[0], atol=1.0e-12)
    assert np.isnan(output[1])


def test_pca_projection_matches_svd_with_sign_canonicalisation(verify):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 6))
    rank = 3
    _u, singular, right = np.linalg.svd(x, full_matrices=False)
    projection = right[:rank].T
    projection = verify.canonicalize_projection(projection)

    ours, energy = verify.fit_pca_projection(x, rank)
    ours = verify.canonicalize_projection(ours)
    assert np.allclose(ours, projection, atol=1.0e-12)
    expected_energy = np.square(singular) / np.square(singular).sum()
    assert np.allclose(energy, expected_energy, atol=1.0e-12)


def _synthetic_context_event(rng, tag: str):
    from sua_exploration.mc_maze.h1_sparse_event_endpoint import MovementEvent

    displacement = rng.normal(size=7)
    start = rng.normal(size=7)
    midpoint = start + 0.5 * displacement
    base = MovementEvent(
        row_id=0,
        tag=tag,
        trial_value=0.0,
        trial_index=0,
        start_time=0.0,
        stop_time=1.0,
        duration_seconds=1.0,
        eval_bins=10,
        displacement=displacement,
        log_rates=rng.normal(size=176),
    )
    from sua_exploration.mc_maze.h1_event_carrier_design_screen import ContextEvent

    return ContextEvent(base=base, start_state=start.astype(np.float64), midpoint_state=midpoint.astype(np.float64))


def test_tag_free_families_ignore_tag_permutation(verify):
    from sua_exploration.mc_maze.h1_event_carrier_design_screen import TAG_NAMES

    rng = np.random.default_rng(2)
    tags = list(TAG_NAMES)
    events = [_synthetic_context_event(rng, tag) for tag in tags]
    permuted_tags = list(reversed(tags))

    for family in ("delta", "poscontext", "startstop", "deltastart"):
        original = verify.build_raw_features(events, family)
        shuffled = verify.build_raw_features(events, family, tags=permuted_tags)
        assert np.array_equal(original, shuffled)


def test_fourteen_column_families_span_same_space(verify):
    rng = np.random.default_rng(3)
    events = [_synthetic_context_event(rng, "Reach") for _ in range(16)]
    matrices = [
        verify.build_raw_features(events, family)
        for family in ("poscontext", "startstop", "deltastart")
    ]
    ranks = [int(np.linalg.matrix_rank(matrix)) for matrix in matrices]
    assert ranks == [14, 14, 14]
    for left, right in ((0, 1), (0, 2), (1, 2)):
        stacked = np.hstack((matrices[left], matrices[right]))
        assert int(np.linalg.matrix_rank(stacked)) == 14


def test_leave_largest_absolute_out_mean(verify):
    rows = [
        ("a", 0.10),
        ("b", -0.30),
        ("c", 0.05),
        ("d", 0.20),
    ]
    summary = verify.paired_summary(rows)
    values = np.array([0.10, -0.30, 0.05, 0.20], dtype=np.float64)
    remove = int(np.argmax(np.abs(values)))
    kept = np.delete(values, remove)
    assert summary["removed_session"] == rows[remove][0]
    assert summary["leave_largest_absolute_out_mean"] == pytest.approx(float(np.mean(kept)))
    assert summary["mean"] == pytest.approx(float(np.mean(values)))
    assert summary["median"] == pytest.approx(float(np.median(values)))
    assert summary["positive"] == 3
