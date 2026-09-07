"""Numerical contracts for behavior_manifold_v2 (offline synthetic fixtures)."""
from __future__ import annotations

import json
import os
import stat
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sua_exploration.behavior_autoencoder_v1.core import (
    AutoencoderSpec,
    array_sha256,
    fit_affine_ridge,
    fit_pca_manifold,
    predict_affine_ridge,
)
from sua_exploration.behavior_autoencoder_v1.m1_screen import (
    FittedManifold,
    score_one_session,
)
from sua_exploration.h1_m1_priority_v1.core import (
    DirectRidgeSession,
    covariance_spectrum,
    direct_ridge_loso,
    regression_metrics,
)

from sua_exploration.behavior_manifold_v2 import protocol as v2_protocol
from sua_exploration.behavior_manifold_v2.e1_oracle import oracle_direct_ridge, oracle_q8_ensemble
from sua_exploration.behavior_manifold_v2.e2_budget import q8_budget_ensemble
from sua_exploration.behavior_manifold_v2.e3_spectrum import experiment_e3
from sua_exploration.behavior_manifold_v2.e4_attribution import _pearson, _pooled_per_output
from sua_exploration.behavior_manifold_v2.e5_estimators import (
    _latent_ridge_weights,
    _shrunk_query_prediction,
    gcv_lambda_curves,
)
from sua_exploration.behavior_manifold_v2.protocol import (
    aligned_budget,
    budget_normalizers,
    direct_ridge_budget,
    full_session_arrays,
)


def _session(name: str, seed: int, *, rows: int = 1100, trials: int = 40, outputs: int = 6,
             noise: float = 0.05, inputs: int = 5) -> DirectRidgeSession:
    rng = np.random.default_rng(seed)
    starts = np.linspace(4, rows - 24, trials).astype(int)
    change = np.zeros(rows, dtype=bool)
    change[starts] = True
    neural = rng.normal(size=(rows, inputs)).astype(np.float32)
    latent_weight = rng.normal(size=(inputs, 3)) / np.sqrt(inputs)
    basis = rng.normal(size=(3, outputs)) / np.sqrt(3)
    target = (neural @ latent_weight @ basis + noise * rng.normal(size=(rows, outputs))).astype(np.float32)
    mask = np.ones(rows, dtype=bool)
    mask[: int(starts[0])] = False
    mask[:: 37] = False
    names = tuple(f"m{index}" for index in range(outputs))
    return DirectRidgeSession(name, neural, target, mask, change, names)


def _four_sessions(**kwargs) -> dict[str, DirectRidgeSession]:
    return {f"s{index}": _session(f"s{index}", index, **kwargs) for index in range(4)}


def test_aligned_budget_reproduces_frozen_m10_split_exactly() -> None:
    session = _session("s0", 0)
    for split in ("support", "query"):
        _, _, frozen_indices = session.aligned(0, split=split)
        _, _, budget_indices = aligned_budget(session, 10, split=split)
        assert np.array_equal(frozen_indices, budget_indices)
        assert array_sha256(frozen_indices) == array_sha256(budget_indices)


def test_aligned_budget_partitions_chronologically() -> None:
    session = _session("s0", 1)
    trial_id = session.trial_id
    for budget in (4, 10, 30):
        _, _, support_indices = aligned_budget(session, budget, split="support")
        _, _, query_indices = aligned_budget(session, budget, split="query")
        assert np.intersect1d(support_indices, query_indices).size == 0
        assert trial_id[support_indices].max() <= budget
        assert trial_id[query_indices].min() >= budget + 1
        legal = np.flatnonzero(np.asarray(session.eval_mask, bool) & (trial_id > 0))
        assert np.array_equal(np.sort(np.concatenate((support_indices, query_indices))), legal)
    with pytest.raises(ValueError, match="insufficient"):
        aligned_budget(session, 10_000, split="query")


def test_full_session_arrays_is_the_m10_union() -> None:
    session = _session("s0", 2)
    _, _, support = aligned_budget(session, 10, split="support")
    _, _, query = aligned_budget(session, 10, split="query")
    _, _, full = full_session_arrays(session)
    assert np.array_equal(np.sort(np.concatenate((support, query))), full)


def test_direct_ridge_budget_m10_reproduces_direct_ridge_loso_bit_exact() -> None:
    sessions = _four_sessions(noise=0.03)
    frozen = direct_ridge_loso(sessions, lag_grid=(0,), lambda_grid=(0.1, 0.3))
    budget = direct_ridge_budget(sessions, 10)
    for name in sorted(sessions):
        frozen_metrics = frozen["folds"][name]["target_metrics"]
        budget_metrics = budget["folds"][name]["target_metrics"]
        assert (
            frozen["folds"][name]["selected"]["ridge_lambda_per_sample"]
            == budget["folds"][name]["selected"]["ridge_lambda_per_sample"]
        )
        assert frozen_metrics["prediction_sha256"] == budget_metrics["prediction_sha256"]
        assert frozen_metrics["pooled_variance_weighted_r2"] == budget_metrics["pooled_variance_weighted_r2"]
    assert frozen["equal_session"] == budget["equal_session"]


def test_budget_normalizers_match_frozen_source_normalizer_at_m10() -> None:
    sessions = _four_sessions()
    names = sorted(sessions)
    for target in names:
        others = [sessions[name] for name in names if name != target]
        from sua_exploration.h1_m1_priority_v1.core import _source_normalizer

        frozen = _source_normalizer(others, 0)
        generalized = budget_normalizers(others, 10)
        for key in frozen:
            assert np.array_equal(frozen[key], generalized[key])


def test_gcv_loo_is_exact_and_selects_small_lambda_when_snr_is_high() -> None:
    rng = np.random.default_rng(5)
    n, d, q = 150, 6, 3
    x = rng.normal(size=(n, d))
    weight = rng.normal(size=(d, q))
    z = x @ weight + 0.05 * rng.normal(size=(n, q))
    result = gcv_lambda_curves(x, z, lambdas=(1.0e-3, 1.0e-1, 1.0e1))
    value = 1.0e-1
    weight, intercept = fit_affine_ridge(x, z, ridge_lambda=value)
    fitted = x @ weight + intercept
    brute = np.zeros_like(z)
    for index in range(n):
        mask = np.arange(n) != index
        w, b = fit_affine_ridge(x[mask], z[mask], ridge_lambda=value * n / (n - 1))
        brute[index] = x[index] @ w + b
    closed = z - (z - fitted) / (1.0 - _hat_diagonal(x, value))[:, None]
    assert np.abs(closed - brute).max() < 1.0e-8
    assert np.isclose(result["curves"][str(value)]["exact_loo_mean_squared_error"],
                      float(np.square(brute - z).mean()), atol=1.0e-10)
    assert result["selected_lambda_gcv"] == 1.0e-3
    # Heavy noise must push the selection toward stronger regularization.
    noisy = gcv_lambda_curves(x, z + 5.0 * rng.normal(size=z.shape), lambdas=(1.0e-3, 1.0e-1, 1.0e1))
    assert noisy["selected_lambda_gcv"] > 1.0e-3


def reference_truth(session: DirectRidgeSession) -> np.ndarray:
    return aligned_budget(session, 10, split="query")[1]


def _hat_diagonal(x: np.ndarray, value: float) -> np.ndarray:
    n, d = x.shape
    augmented = np.concatenate((x, np.ones((n, 1))), axis=1)
    penalty = np.eye(d + 1) * (value * n)
    penalty[-1, -1] = 0.0
    inverse = np.linalg.inv(augmented.T @ augmented + penalty)
    return np.einsum("ij,jk,ik->i", augmented, inverse, augmented)


def _pca_manifold(values: np.ndarray, latent_dim: int) -> FittedManifold:
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 1.0e-8)
    standardized = (values - mean) / scale
    pca = fit_pca_manifold(standardized, latent_dim)
    spec = AutoencoderSpec("pca", latent_dim)
    return FittedManifold(spec, mean, scale, pca, None, torch.device("cpu"), {"kind": "pca"})


def test_eb_shrinkage_recovers_source_regression_when_support_is_tiny() -> None:
    rng = np.random.default_rng(7)
    rows, inputs, latent_dim, outputs = 2400, 24, 3, 6
    inputs_weight = rng.normal(size=(inputs, latent_dim)) / np.sqrt(inputs)
    basis = rng.normal(size=(latent_dim, outputs)) / np.sqrt(latent_dim)

    def make(seed: int, noise: float) -> DirectRidgeSession:
        local = np.random.default_rng(seed)
        neural = local.normal(size=(rows, inputs))
        behavior = neural @ inputs_weight @ basis + noise * local.normal(size=(rows, outputs))
        starts = np.linspace(3, rows - 40, 40).astype(int)
        change = np.zeros(rows, dtype=bool)
        change[starts] = True
        mask = np.ones(rows, dtype=bool)
        mask[: int(starts[0])] = False
        return DirectRidgeSession(
            f"n{seed}", neural.astype(np.float32), behavior.astype(np.float32), mask, change,
            tuple(f"m{i}" for i in range(outputs)),
        )

    sessions = {f"s{index}": make(100 + index, 0.05) for index in range(4)}
    sessions["s0"] = make(100, 0.5)  # noisier target; its one-trial support is underdetermined
    all_behavior = np.concatenate(
        [np.asarray(session.target[np.asarray(session.eval_mask, bool) & (session.trial_id > 0)], dtype=np.float64)
         for session in sessions.values()]
    )
    manifold = _pca_manifold(all_behavior, latent_dim)
    x_mean = np.zeros(inputs)
    x_scale = np.ones(inputs)
    target = sessions["s0"]
    sources = [sessions[name] for name in ("s1", "s2", "s3")]
    tiny_support_x, tiny_support_y, _ = aligned_budget(target, 1, split="support")
    query_x, query_y, _ = aligned_budget(target, 10, split="query")
    plain_w, plain_b = _latent_ridge_weights(tiny_support_x, tiny_support_y, manifold, x_mean, x_scale, ridge_lambda=0.01)
    full_x = np.concatenate([full_session_arrays(source)[0] for source in sources])
    full_y = np.concatenate([full_session_arrays(source)[1] for source in sources])
    source_w, source_b = _latent_ridge_weights(full_x, full_y, manifold, x_mean, x_scale, ridge_lambda=0.01)
    gammas = np.linspace(0.0, 1.0, 11)
    scores = []
    for gamma in gammas:
        prediction, truth = _shrunk_query_prediction(
            target, manifold, x_mean, x_scale, budget=10,
            target_weight=plain_w, target_intercept=plain_b,
            source_weight=source_w, source_intercept=source_b, gamma=float(gamma),
        )
        assert np.array_equal(truth, query_y)
        scores.append(regression_metrics(truth, prediction)["pooled_variance_weighted_r2"])
    scores = np.asarray(scores)
    assert scores.max() > scores[-1] + 0.01  # some shrinkage strictly beats gamma = 1
    assert scores[int(np.argmax(scores))] == scores.max()
    # gamma = 1 must reproduce the unshrunk deployment route prediction bit-exactly.
    gamma_one, _ = _shrunk_query_prediction(
        target, manifold, x_mean, x_scale, budget=10,
        target_weight=plain_w, target_intercept=plain_b,
        source_weight=source_w, source_intercept=source_b, gamma=1.0,
    )
    plain = manifold.decode(predict_affine_ridge((query_x - x_mean) / x_scale, plain_w, plain_b))
    assert np.array_equal(gamma_one, plain)


def test_oracle_support_ridge_dominates_m10_support_ridge() -> None:
    sessions = _four_sessions(noise=0.05)
    names = sorted(sessions)
    target = sessions[names[0]]
    others = [sessions[name] for name in names[1:]]
    normalizer = budget_normalizers(others, 10)
    oracle_metrics, _, _ = oracle_direct_ridge(target, others, ridge_lambda=0.1)
    deployed_metrics, _, _ = v2_protocol.fit_score_budget_ridge(
        target, normalizer, budget=10, ridge_lambda=0.1
    )
    assert oracle_metrics["pooled_variance_weighted_r2"] > deployed_metrics["pooled_variance_weighted_r2"]
    assert oracle_metrics["query_indices_sha256"] == deployed_metrics["query_indices_sha256"]
    assert oracle_metrics["full_session_bins"] > deployed_metrics["support_bins"]


def test_oracle_q8_uses_full_session_latents() -> None:
    sessions = _four_sessions(noise=0.05)
    names = sorted(sessions)
    target = sessions[names[0]]
    all_behavior = np.concatenate(
        [np.asarray(session.target[np.asarray(session.eval_mask, bool) & (session.trial_id > 0)], dtype=np.float64)
         for session in sessions.values()]
    )
    manifold = _pca_manifold(all_behavior, 3)
    stub_target = SimpleNamespace(
        x_mean=np.zeros(target.neural.shape[1]),
        x_scale=np.ones(target.neural.shape[1]),
        seeds=[
            SimpleNamespace(manifold=manifold, ridge_lambda=0.3),
            SimpleNamespace(manifold=manifold, ridge_lambda=0.3),
        ],
    )
    metrics, truth, _ = oracle_q8_ensemble(target, stub_target, ridge_lambda=0.3)
    _, _, query_indices = aligned_budget(target, 10, split="query")
    assert metrics["query_indices_sha256"] == array_sha256(query_indices)
    assert metrics["full_session_bins"] == full_session_arrays(target)[0].shape[0]
    assert np.array_equal(truth, target.target[query_indices].astype(np.float64))
    deployed = score_one_session(
        target, manifold, x_mean=stub_target.x_mean, x_scale=stub_target.x_scale, ridge_lambda=0.3
    )[0]
    assert metrics["pooled_variance_weighted_r2"] > 0.5
    assert metrics["pooled_variance_weighted_r2"] > deployed["pooled_variance_weighted_r2"]


def test_q8_budget_ensemble_matches_score_one_session_mean_at_m10() -> None:
    sessions = _four_sessions(noise=0.05)
    names = sorted(sessions)
    target = sessions[names[0]]
    all_behavior = np.concatenate(
        [np.asarray(session.target[np.asarray(session.eval_mask, bool) & (session.trial_id > 0)], dtype=np.float64)
         for session in sessions.values()]
    )
    manifold = _pca_manifold(all_behavior, 3)
    stub_target = SimpleNamespace(
        x_mean=np.zeros(target.neural.shape[1]),
        x_scale=np.ones(target.neural.shape[1]),
        seeds=[
            SimpleNamespace(manifold=manifold, ridge_lambda=0.3, seed_index=0, state_sha256="0" * 64),
            SimpleNamespace(manifold=manifold, ridge_lambda=0.3, seed_index=1, state_sha256="1" * 64),
        ],
    )
    metrics, truth, prediction, seed_rows = q8_budget_ensemble(target, stub_target, budget=10)
    reference = np.mean(
        np.stack(
            [
                score_one_session(target, manifold, x_mean=stub_target.x_mean, x_scale=stub_target.x_scale,
                                  ridge_lambda=0.3)[2]
                for _ in range(2)
            ],
            axis=0,
        ),
        axis=0,
        dtype=np.float64,
    )
    assert np.array_equal(prediction, reference)
    assert metrics["prediction_sha256"] == array_sha256(reference)
    assert [row["seed_index"] for row in seed_rows] == [0, 1]
    assert np.array_equal(truth, reference_truth(target))


def test_participation_ratios_across_subsets() -> None:
    sessions = _four_sessions()
    result = experiment_e3(sessions=sessions)
    for name in sorted(sessions):
        for subset in ("full_session", "m10_support", "post_m10_query"):
            entry = result["per_session"][name][subset]
            assert 1.0 <= entry["participation_ratio"] <= 6.0
            assert abs(sum(entry["variance_fraction_desc"]) - 1.0) < 1.0e-9
            assert np.all(np.diff(entry["covariance_eigenvalues_desc"]) <= 1.0e-12)


def test_pooled_per_output_and_pearson_math() -> None:
    metrics_a = {"sse_per_output": [1.0, 2.0], "tss_per_output": [4.0, 4.0]}
    metrics_b = {"sse_per_output": [2.0, 2.0], "tss_per_output": [4.0, 8.0]}
    pooled = _pooled_per_output([metrics_a, metrics_b])
    assert np.allclose(pooled, [1.0 - 3.0 / 8.0, 1.0 - 4.0 / 12.0])
    exact = _pearson(np.asarray([1.0, 2.0, 3.0, 4.0]), np.asarray([2.0, 4.0, 6.0, 8.2]))
    assert exact["pearson_r"] > 0.999
    with pytest.raises(ValueError, match="degenerate"):
        _pearson(np.asarray([1.0, 2.0, 3.0]), np.asarray([5.0, 5.0, 5.0]))
    with pytest.raises(ValueError, match="degenerate"):
        _pearson(np.asarray([1.0, 1.0, 1.0]), np.asarray([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="at least three"):
        _pearson(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0]))


def test_pca8_projection_is_the_orthogonal_source_frozen_operator() -> None:
    from sua_exploration.behavior_autoencoder_v1.m1_screen import fit_source_manifold

    sessions = _four_sessions(noise=0.05)
    sources = {name: session for name, session in list(sorted(sessions.items()))[1:]}
    manifold = fit_source_manifold(sources, AutoencoderSpec("pca", 3), device=torch.device("cpu"), seed=0)
    assert manifold.fit_evidence["kind"] == "pca"
    rng = np.random.default_rng(11)
    raw = manifold.decode(manifold.encode(rng.normal(size=(64, 6)) * 2.0))
    # Projection is idempotent and leaves on-manifold points fixed.
    twice = manifold.decode(manifold.encode(raw))
    assert np.abs(twice - raw).max() < 1.0e-9
    components = manifold.pca.components
    assert components.shape == (3, 6)
    assert np.allclose(components @ components.T, np.eye(3), atol=1.0e-9)
    # Target session behaviour never entered the basis.
    assert manifold.fit_evidence["source_sessions"] == [name for name in sorted(sources)]


def test_supervised_mismatched_fit_is_source_frozen_and_flags_the_mismatch() -> None:
    from sua_exploration.behavior_manifold_v2.e7_supervised_control import (
        SupervisedMismatchedSpec,
        fit_supervised_mismatched_manifold,
    )

    sessions = {f"s{index}": _session(f"s{index}", index, rows=600, trials=20) for index in range(2)}
    spec = SupervisedMismatchedSpec(3, 8, 0.1, pretrain_epochs=1, fine_steps=3)
    manifold = fit_supervised_mismatched_manifold(
        sessions, spec, device=torch.device("cpu"), seed=3,
        query_batch_size=64, reconstruction_batch_size=128,
    )
    value = np.asarray(sessions["s0"].target[:16], dtype=np.float64)
    reconstruction = manifold.decode(manifold.encode(value))
    assert reconstruction.shape == value.shape and np.isfinite(reconstruction).all()
    assert manifold.fit_evidence["in_loop_ridge_support"] == "full_session"
    assert manifold.fit_evidence["deployment_matched_objective"] is False
    assert manifold.fit_evidence["source_sessions"] == ["s0", "s1"]
    assert manifold.fit_evidence["parameter_count"] == (6 * 8 + 8) + (8 * 3 + 3) + (3 * 8 + 8) + (8 * 6 + 6)


def test_nested_supervised_mismatched_excludes_target_and_deploys_via_m10() -> None:
    from sua_exploration.behavior_manifold_v2.e7_supervised_control import (
        SupervisedMismatchedSpec,
        nested_supervised_mismatched_loso,
    )

    sessions = _four_sessions(noise=0.05)
    specs = (SupervisedMismatchedSpec(3, 12, 0.1, pretrain_epochs=1, fine_steps=4),)
    result = nested_supervised_mismatched_loso(
        sessions, specs, device="cpu", deployment_lambda_grid=(0.1,), seed_offset=0,
    )
    assert result["target_used_for_selection"] is False
    assert result["protocol"]["target_backward_steps"] == 0
    for name, fold in result["folds"].items():
        assert name not in fold["source_sessions"]
        assert fold["target_excluded"] is True
        assert fold["final_fit"]["in_loop_ridge_support"] == "full_session"
        assert np.isfinite(fold["target_metrics"]["pooled_variance_weighted_r2"])
    assert len(result["folds"]) == 4
    # The real performance claim is made by the M1 run, not this 3-step fixture.


def test_receipt_writer_is_transactional_and_immutable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(v2_protocol, "RESULT_ROOT", tmp_path)
    digest = v2_protocol.write_receipt("probe.json", {"a": 1})
    path = tmp_path / "probe.json"
    sidecar = tmp_path / "probe.json.sha256"
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444
    assert sidecar.read_text(encoding="ascii") == f"{digest}  probe.json\n"
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}
    with pytest.raises(ValueError, match="refusing overwrite"):
        v2_protocol.write_receipt("probe.json", {"a": 2})


# --------------------------------------------------------------------------------------
# probe_alignment (aux-head gate): probe math on synthetic fixtures
# --------------------------------------------------------------------------------------
from sua_exploration.behavior_manifold_v2.probe_alignment import (  # noqa: E402
    GREENLIGHT_RATIO,
    KILL_RATIO,
    build_centered_query_gram,
    fit_probe_ridge,
    kernel_probe,
    nested_probe_cell,
    probe_verdict,
)


def test_probe_ridge_dual_matches_primal_when_n_exceeds_p() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(200, 12))
    z = rng.normal(size=(200, 5))
    for ridge_lambda in (0.1, 0.3, 1.0):
        dual_weight, dual_intercept = fit_probe_ridge(x, z, ridge_lambda=ridge_lambda)
        primal_weight, primal_intercept = fit_affine_ridge(x, z, ridge_lambda=ridge_lambda)
        assert np.allclose(dual_weight, primal_weight, atol=1.0e-9)
        assert np.allclose(dual_intercept, primal_intercept, atol=1.0e-9)


def test_probe_ridge_underdetermined_matches_primal_normal_equations() -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(size=(30, 48))
    z = rng.normal(size=(30, 4))
    ridge_lambda = 0.3
    weight, intercept = fit_probe_ridge(x, z, ridge_lambda=ridge_lambda)
    xc = x - x.mean(axis=0)
    zc = z - z.mean(axis=0)
    matrix = xc.T @ xc + ridge_lambda * x.shape[0] * np.eye(x.shape[1])
    reference = np.linalg.solve(matrix, xc.T @ zc)
    assert np.allclose(weight, reference, atol=1.0e-8)
    assert np.allclose(intercept, z.mean(axis=0) - x.mean(axis=0) @ reference, atol=1.0e-8)


def test_kernel_probe_prediction_equals_direct_ridge_prediction() -> None:
    rng = np.random.default_rng(13)
    support_x = rng.normal(size=(60, 7))
    query_x = rng.normal(size=(40, 7))
    support_z = rng.normal(size=(60, 3))
    gram = build_centered_query_gram(
        query_x, support_x, np.zeros(7), np.ones(7), chunk=16
    )
    reference_gram = (query_x - support_x.mean(axis=0)) @ (support_x - support_x.mean(axis=0)).T
    assert np.allclose(gram, reference_gram, atol=1.0e-10)
    for ridge_lambda in (0.1, 0.3):
        prediction, weight, intercept, _ = kernel_probe(
            support_x, support_z, gram, ridge_lambda=ridge_lambda
        )
        direct_weight, direct_intercept = fit_probe_ridge(support_x, support_z, ridge_lambda=ridge_lambda)
        assert np.allclose(weight, direct_weight, atol=1.0e-9)
        assert np.allclose(intercept, direct_intercept, atol=1.0e-9)
        assert np.allclose(
            prediction,
            predict_affine_ridge(query_x, direct_weight, direct_intercept),
            atol=1.0e-9,
        )


def test_probe_verdict_thresholds_are_preregistered_boundaries() -> None:
    raw = 0.6
    assert probe_verdict(0.9 * raw, raw)["verdict"] == "KILL_AUX_HEAD_CELL"
    assert probe_verdict(0.95 * raw, raw)["verdict"] == "KILL_AUX_HEAD_CELL"
    assert probe_verdict(0.75 * raw, raw)["verdict"] == "GREEN_LIGHT_AUX_HEAD_CELL"
    assert probe_verdict(0.5 * raw, raw)["verdict"] == "GREEN_LIGHT_AUX_HEAD_CELL"
    middle = probe_verdict(0.8 * raw, raw)
    assert middle["verdict"] == "AMBIGUOUS"
    assert middle["thresholds"] == {"kill_at_ratio_ge": KILL_RATIO, "greenlight_at_ratio_le": GREENLIGHT_RATIO}
    with pytest.raises(ValueError, match="invalid verdict"):
        probe_verdict(0.5, 0.0)


def _probe_fixture(
    seed: int,
    *,
    sessions: int = 3,
    support: int = 240,
    query: int = 800,
    features: int = 320,
    outputs: int = 6,
    latent: int = 4,
    noise: float = 0.02,
):
    rng = np.random.default_rng(seed)
    # Low-rank signal (8 informative directions) so the M10-support ridge is not
    # shrinkage-starved in the p >> n regime, mirroring the real penultimate
    # representation whose behaviour-relevant subspace is far below its width.
    generator = np.zeros((features, outputs))
    generator[:8] = rng.normal(size=(8, outputs)) / np.sqrt(8)
    warp = rng.normal(size=(outputs, latent)) / np.sqrt(outputs)
    support_features, query_features, raw, latent_linear, latent_warped = {}, {}, {}, {}, {}
    for index in range(sessions):
        name = f"s{index}"
        hs = rng.normal(size=(support, features))
        hq = rng.normal(size=(query, features))
        # Behaviour-relevant directions dominate the spectrum (as in the real
        # penultimate representation), so the M10-support ridge is not
        # shrinkage-starved in the p >> n regime.
        hs[:, :8] *= 8.0
        hq[:, :8] *= 8.0
        ys = hs @ generator + noise * rng.normal(size=(support, outputs))
        yq = hq @ generator + noise * rng.normal(size=(query, outputs))
        support_features[name] = hs
        query_features[name] = hq
        raw[name] = (ys, yq)
        latent_linear[name] = (
            ys @ warp + noise * rng.normal(size=(support, latent)),
            yq @ warp + noise * rng.normal(size=(query, latent)),
        )
        latent_warped[name] = (np.sin(3.0 * ys) @ warp, np.sin(3.0 * yq) @ warp)
    grams = {
        name: build_centered_query_gram(
            query_features[name], support_features[name], np.zeros(features), np.ones(features)
        )
        for name in support_features
    }
    return support_features, grams, raw, latent_linear, latent_warped


def test_nested_probe_reads_kill_when_latent_is_linearly_recoverable() -> None:
    support_features, grams, raw, latent_linear, _ = _probe_fixture(21)
    probed = "s1"
    raw_cell = nested_probe_cell(support_features, grams, raw, probed=probed)
    latent_cell = nested_probe_cell(support_features, grams, latent_linear, probed=probed)
    raw_r2 = raw_cell["selected_query_r2"]
    latent_r2 = latent_cell["selected_query_r2"]
    assert raw_r2 > 0.9
    reading = probe_verdict(latent_r2, raw_r2)
    assert reading["ratio"] >= KILL_RATIO
    assert reading["verdict"] == "KILL_AUX_HEAD_CELL"
    assert latent_cell["selected"]["ridge_lambda_per_sample"] in (0.1, 0.3)
    assert latent_cell["support_bins"] == support_features[probed].shape[0]
    assert latent_cell["query_bins"] == raw[probed][1].shape[0]
    # The dual path handled the p >> n regime (320 features, 240 support rows).
    assert support_features[probed].shape[1] > support_features[probed].shape[0]


def test_nested_probe_reads_green_light_when_latent_is_nonlinearly_warped() -> None:
    support_features, grams, raw, _, latent_warped = _probe_fixture(22)
    raw_cell = nested_probe_cell(support_features, grams, raw, probed="s1")
    warped_cell = nested_probe_cell(support_features, grams, latent_warped, probed="s1")
    raw_r2 = raw_cell["selected_query_r2"]
    warped_r2 = warped_cell["selected_query_r2"]
    reading = probe_verdict(warped_r2, raw_r2)
    assert reading["ratio"] < GREENLIGHT_RATIO
    assert reading["verdict"] == "GREEN_LIGHT_AUX_HEAD_CELL"
    # The raw control confirms the estimator itself still solves the linear task
    # on the same support rows: the deficit is specific to the nonlinear warp.
    assert raw_r2 > 0.9


# --------------------------------------------------------------------------------------
# e8_readout: deployment-legal penultimate-to-latent readout math pins
# --------------------------------------------------------------------------------------
from sua_exploration.behavior_manifold_v2.e8_readout import (  # noqa: E402
    deploy_latent_readout,
    e8_verdict,
    ensemble_mean,
    lane_split_positions,
    mix_predictions,
)


class _LinearManifold:
    """Deterministic linear stand-in: encode/decode are fixed matrices."""

    def __init__(self, encode_matrix: np.ndarray, decode_matrix: np.ndarray) -> None:
        self.encode_matrix = encode_matrix
        self.decode_matrix = decode_matrix

    def encode(self, raw: np.ndarray) -> np.ndarray:
        return np.asarray(raw, dtype=np.float64) @ self.encode_matrix

    def decode(self, latent: np.ndarray) -> np.ndarray:
        return np.asarray(latent, dtype=np.float64) @ self.decode_matrix


def test_e8_lane_split_positions_pin_the_m10_label_budget() -> None:
    session = _session("s0", 7)
    legal = np.flatnonzero(np.asarray(session.eval_mask, bool) & (session.trial_id > 0))
    support, query = lane_split_positions(session, legal)
    assert np.array_equal(legal[support], aligned_budget(session, 10, split="support")[2])
    assert np.array_equal(legal[query], aligned_budget(session, 10, split="query")[2])
    # Leakage guards: non-ascending/duplicated bins, pre-trial bins, and any support
    # selection that deviates from the lane M10 bins must all fail closed.
    with pytest.raises(ValueError, match="ascending"):
        lane_split_positions(session, np.concatenate((legal[:5], legal[:5])))
    with pytest.raises(ValueError, match="pre-first-trial"):
        lane_split_positions(session, np.concatenate((np.array([0], dtype=np.int64), legal[1:])))
    with pytest.raises(ValueError, match="lane split query"):
        lane_split_positions(
            session,
            np.concatenate((legal[legal < session.boundary], legal[legal >= session.boundary][1:])),
        )
    # Dropping one M10 support bin must break the exact support-budget equality.
    with pytest.raises(ValueError, match="lane split support"):
        lane_split_positions(session, legal[1:])


def test_e8_deploy_latent_readout_shapes_and_decode_math() -> None:
    rng = np.random.default_rng(31)
    encode_matrix = rng.normal(size=(16, 8))
    decode_matrix = rng.normal(size=(8, 16))
    manifold = _LinearManifold(encode_matrix, decode_matrix)
    support_x = rng.normal(size=(90, 25))
    query_x = rng.normal(size=(45, 25))
    support_y = rng.normal(size=(90, 16))
    gram = build_centered_query_gram(query_x, support_x, np.zeros(25), np.ones(25))
    support_z = manifold.encode(support_y)
    decoded, latent_prediction, weight, intercept = deploy_latent_readout(
        support_x, support_z, gram, manifold=manifold, ridge_lambda=0.3
    )
    # Decode-path shape pins: behaviour comes back at the lane's 16 outputs and the
    # latent prediction at the manifold's q width.
    assert decoded.shape == (45, 16)
    assert latent_prediction.shape == (45, 8)
    direct_weight, direct_intercept = fit_probe_ridge(support_x, support_z, ridge_lambda=0.3)
    assert np.allclose(weight, direct_weight, atol=1.0e-9)
    assert np.allclose(intercept, direct_intercept, atol=1.0e-9)
    assert np.allclose(
        decoded,
        predict_affine_ridge(query_x, direct_weight, direct_intercept) @ decode_matrix,
        atol=1.0e-9,
    )


def test_e8_ensemble_and_mix_conventions() -> None:
    rng = np.random.default_rng(32)
    predictions = [rng.normal(size=(20, 16)) for _ in range(3)]
    ensemble = ensemble_mean(predictions)
    assert ensemble.shape == (20, 16)
    assert np.allclose(ensemble, np.mean(np.stack(predictions), axis=0), atol=0.0)
    left, right = predictions[0], predictions[1]
    assert np.allclose(mix_predictions(left, right), 0.5 * left + 0.5 * right, atol=1.0e-12)
    assert np.allclose(mix_predictions(left, right, weight=0.25), 0.25 * left + 0.75 * right, atol=1.0e-12)
    with pytest.raises(ValueError, match="mix weight"):
        mix_predictions(left, right, weight=1.5)
    with pytest.raises(ValueError, match="mix shape"):
        mix_predictions(left, rng.normal(size=(21, 16)))


def test_e8_verdict_preregistered_bands() -> None:
    raw_head = 0.670196
    assert e8_verdict(raw_head + 0.01, raw_head)["verdict"] == "FREE_FULL_DECODER_IMPROVEMENT"
    assert e8_verdict(raw_head + 0.02, raw_head)["verdict"] == "FREE_FULL_DECODER_IMPROVEMENT"
    assert e8_verdict(raw_head, raw_head)["verdict"] == "READOUT_ADDS_NOTHING"
    assert e8_verdict(raw_head - 0.05, raw_head)["verdict"] == "READOUT_ADDS_NOTHING"
    pareto = e8_verdict(raw_head + 0.005, raw_head)
    assert pareto["verdict"] == "NEW_PARETO_ENTRY"
    assert pareto["delta_vs_raw_head"] == pytest.approx(0.005)
    # Band resolution: a value at/below the raw head is dominated on the same
    # surface and reads as "adds nothing" even when above the 0.434 reference;
    # the Pareto band covers values above the raw head but short of +0.01.
    mid = e8_verdict(0.5, raw_head)
    assert mid["verdict"] == "READOUT_ADDS_NOTHING"
    assert e8_verdict(raw_head - 0.2, raw_head)["verdict"] == "READOUT_ADDS_NOTHING"
