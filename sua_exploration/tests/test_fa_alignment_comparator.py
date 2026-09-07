"""Unit tests for the FA-alignment comparator (synthetic data only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sua_exploration.mc_maze import fa_alignment_comparator as core
from sua_exploration.mc_maze import source_pooled_ridge as spr


def _rng() -> np.random.Generator:
    return core.rng_from_material(core.SEED_MATERIAL)


def test_fa_recovers_known_low_rank_structure() -> None:
    rng = _rng()
    n_samples, n_channels, d = 2500, 18, 3
    loadings_true = rng.normal(size=(n_channels, d))
    latents = rng.normal(size=(n_samples, d))
    noise = 0.05 * rng.normal(size=(n_samples, n_channels))
    observations = latents @ loadings_true.T + noise
    fit = core.fit_factor_analysis(observations, d)
    reconstructed = fit.loadings @ fit.loadings.T + np.diag(fit.uniquenesses)
    empirical = np.cov(observations, rowvar=False)
    relative = float(np.linalg.norm(reconstructed - empirical) / np.linalg.norm(empirical))
    assert relative < 0.15
    true_basis = np.linalg.qr(loadings_true)[0]
    fit_basis = np.linalg.qr(fit.loadings)[0]
    overlap = np.linalg.svd(true_basis.T @ fit_basis, compute_uv=False)
    assert float(overlap.min()) > 0.85


def test_alignment_recovers_known_rotation() -> None:
    rng = _rng()
    n_samples, d = 4000, 3
    scales = np.asarray([4.0, 2.0, 0.7], dtype=np.float64)
    source = rng.normal(size=(n_samples, d)) * scales
    theta = 0.7
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    target = source @ rotation.T
    transform = core.estimate_latent_alignment(source, target, target_labels=None)
    aligned = transform.apply(target)
    np.testing.assert_allclose(np.cov(aligned, rowvar=False), np.cov(source, rowvar=False), rtol=0.05, atol=0.15)
    gram = np.abs(transform.matrix @ rotation)
    assignment = gram / np.maximum(gram.max(axis=0, keepdims=True), 1.0e-12)
    assert float(assignment.max(axis=0).min()) > 0.85


def test_zero_target_label_guard_fires() -> None:
    rng = _rng()
    source = rng.normal(size=(40, 3))
    target = rng.normal(size=(40, 3))
    labels = rng.normal(size=(40, 2))
    with pytest.raises(core.FaAlignmentError, match="zero target labels"):
        core.estimate_latent_alignment(source, target, target_labels=labels)
    with pytest.raises(core.FaAlignmentError, match="zero target labels"):
        core.estimate_channel_anchored_alignment(
            rng.normal(size=(8, 3)), rng.normal(size=(8, 3)), target_labels=labels
        )


def test_correspondence_detector_definable_and_undefinable() -> None:
    stable = [np.array([10, 11, 12], dtype=np.int64) for _ in range(4)]
    mapping, identical, count = core.detect_correspondence(stable, identifiers_are_positional_only=False)
    assert mapping is True
    assert identical is True
    assert count == 3
    positional = [np.array([0, 1, 2], dtype=np.int64) for _ in range(4)]
    mapping, identical, count = core.detect_correspondence(positional, identifiers_are_positional_only=True)
    assert mapping is False
    assert identical is True
    assert count == 0
    mismatched = [np.array([1, 2, 3], dtype=np.int64), np.array([1, 2, 4], dtype=np.int64)]
    mapping, identical, count = core.detect_correspondence(mismatched, identifiers_are_positional_only=False)
    assert mapping is False
    assert identical is False
    assert count == 0
    spr_mapping, spr_identical, spr_count = spr.detect_correspondence(
        mismatched, identifiers_are_positional_only=False
    )
    assert (mapping, identical, count) == (spr_mapping, spr_identical, spr_count)


def test_faa_no_align_is_identity() -> None:
    rng = _rng()
    d = 4
    latents = rng.normal(size=(25, d))
    labels = rng.normal(size=(25, 2))
    target = rng.normal(size=(12, d))
    _prediction, transform = core.run_faa_no_align(
        source_latents=latents, source_labels=labels, target_latents=target
    )
    assert transform.is_identity() is True
    np.testing.assert_array_equal(transform.matrix, np.eye(d))
    np.testing.assert_array_equal(transform.apply(target), np.asarray(target, dtype=np.float64))


def test_shared_d_mismatch_is_caught() -> None:
    with pytest.raises(core.FaAlignmentError, match="shared latent dimensionality d mismatch"):
        core.require_shared_d({"session_a": 8, "session_b": 16})
    assert core.require_shared_d({"session_a": 8, "session_b": 8}) == 8
    with pytest.raises(core.FaAlignmentError, match="shared latent dimensionality d mismatch"):
        core.resolve_shared_d_from_feasibility({"a": [4], "b": [8, 16]})


def test_r2_helpers_match_hand_computed_values() -> None:
    truth = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0], [4.0, 4.0]], dtype=np.float64)
    estimate = np.array([[1.1, 1.8], [2.2, 3.1], [2.7, 5.4], [3.8, 4.2]], dtype=np.float64)
    residual = np.square(truth - estimate).sum(axis=0)
    total = np.square(truth - truth.mean(axis=0, keepdims=True)).sum(axis=0)
    unweighted_manual = float((1.0 - residual / total).mean())
    vw_manual = 1.0 - float(residual.sum()) / float(total.sum())
    assert core.r2_unweighted(estimate, truth) == pytest.approx(unweighted_manual)
    assert core.r2_variance_weighted(estimate, truth) == pytest.approx(vw_manual)
    assert core.score_predictions(estimate, truth, dataset="rt") == pytest.approx(vw_manual)
    assert core.score_predictions(estimate, truth, dataset="subject_m") == pytest.approx(vw_manual)


def test_determinism_across_two_runs() -> None:
    rng = _rng()
    observations = rng.normal(size=(300, 10)) @ rng.normal(size=(10, 12))
    fit_a = core.fit_factor_analysis(observations, 4)
    fit_b = core.fit_factor_analysis(observations, 4)
    np.testing.assert_array_equal(fit_a.loadings, fit_b.loadings)
    np.testing.assert_array_equal(fit_a.uniquenesses, fit_b.uniquenesses)
    source = rng.normal(size=(80, 4)) * np.array([3.0, 2.0, 1.0, 0.5])
    angle = 0.4
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0, 0.0],
            [np.sin(angle), np.cos(angle), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    target = source @ rotation.T
    first = core.estimate_latent_alignment(source, target)
    second = core.estimate_latent_alignment(source, target)
    np.testing.assert_array_equal(first.matrix, second.matrix)
    np.testing.assert_array_equal(first.apply(target), second.apply(target))


def test_implementation_files_exist() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert (repo_root / "sua_exploration/mc_maze/fa_alignment_comparator.py").is_file()
    assert (repo_root / "sua_exploration/scripts/run_fa_alignment_comparator.py").is_file()
    assert (repo_root / "sua_exploration/tests/test_fa_alignment_comparator.py").is_file()
