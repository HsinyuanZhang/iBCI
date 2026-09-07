from __future__ import annotations

import numpy as np
import pytest
import torch
import json

from sua_exploration.behavior_autoencoder_v1.core import (
    AutoencoderSpec,
    BehaviorAutoencoder,
    fit_affine_ridge,
    fit_pca_manifold,
    predict_affine_ridge,
    weighted_reconstruction_loss,
)
from sua_exploration.behavior_autoencoder_v1.m1_screen import nested_manifold_loso
from sua_exploration.behavior_autoencoder_v1.combine import combine_family_results
from sua_exploration.behavior_autoencoder_v1.calibration_aware import CalibrationAwareSpec, fit_calibration_aware_manifold
from sua_exploration.behavior_autoencoder_v1.predictive_pca import fit_predictive_pca
from sua_exploration.h1_m1_priority_v1.core import DirectRidgeSession


def test_pca_is_canonical_and_exact_on_rank_two_data() -> None:
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(100, 2))
    basis = np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]])
    values = latent @ basis
    manifold = fit_pca_manifold(values, 2)
    reconstructed = manifold.decode_numpy(manifold.encode_numpy(values))
    assert np.allclose(reconstructed, values, atol=1e-10)
    for row in manifold.components:
        assert row[np.argmax(np.abs(row))] >= 0.0


def test_affine_ridge_recovers_linear_map() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 4)); matrix = rng.normal(size=(4, 3)); bias = rng.normal(size=(3,))
    z = x @ matrix + bias
    weight, intercept = fit_affine_ridge(x, z, ridge_lambda=0.0)
    assert np.allclose(predict_affine_ridge(x, weight, intercept), z, atol=1e-10)


def test_mlp_is_strict_bottleneck_and_gradients_flow() -> None:
    model = BehaviorAutoencoder(16, 6, 24, "gelu")
    value = torch.randn(32, 16)
    latent = model.encode(value)
    prediction = model.decode(latent)
    assert latent.shape == (32, 6) and prediction.shape == value.shape
    loss = weighted_reconstruction_loss(prediction, value, source_scale=torch.arange(1, 17, dtype=torch.float32), raw_loss_fraction=0.5)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_specs_reject_non_bottleneck() -> None:
    with pytest.raises(ValueError, match="strict bottleneck"):
        AutoencoderSpec("mlp", 16, 32).validate(16)
    AutoencoderSpec("pca", 6).validate(16)


def _nested_session(name: str, seed: int) -> DirectRidgeSession:
    rng = np.random.default_rng(seed)
    rows = 620
    starts = np.arange(10, 610, 50)
    change = np.zeros(rows, dtype=bool); change[starts] = True
    neural = rng.normal(size=(rows, 5)).astype(np.float32)
    latent_weight = rng.normal(size=(5, 2))
    behavior_basis = np.asarray([[1.0, 0.2, -0.4, 0.7], [-0.3, 1.1, 0.6, -0.2]])
    target = (neural @ latent_weight @ behavior_basis + 0.01 * rng.normal(size=(rows, 4))).astype(np.float32)
    mask = np.ones(rows, dtype=bool); mask[: starts[0]] = False
    return DirectRidgeSession(name, neural, target, mask, change, ("a", "b", "c", "d"))


def test_nested_pca_excludes_outer_and_inner_sessions() -> None:
    sessions = {f"s{index}": _nested_session(f"s{index}", index) for index in range(4)}
    result = nested_manifold_loso(
        sessions, (AutoencoderSpec("pca", 2),), device="cpu", lambda_grid=(0.0, 1.0e-3),
    )
    assert result["target_support_or_query_used_for_candidate_selection"] is False
    assert result["equal_session"]["mean_r2"] > 0.99
    for name, fold in result["folds"].items():
        assert name not in fold["source_sessions"]
        assert fold["target_excluded_from_all_manifold_training_and_selection"] is True
        assert fold["candidate_count"] == 2


def test_combiner_uses_source_score_not_target_score(tmp_path) -> None:
    def family(kind: str, source_scores: tuple[float, float], target_scores: tuple[float, float]) -> dict:
        names = ["s0", "s1"]
        folds = {}
        for index, name in enumerate(names):
            folds[name] = {
                "selected": {"equal_source_session_mean_r2": source_scores[index], "spec_name": kind},
                "target_metrics": {"pooled_variance_weighted_r2": target_scores[index], "sse_per_output": [1.0], "tss_per_output": [10.0]},
                "baseline_target_metrics": {"pooled_variance_weighted_r2": 0.5},
                "final_manifold_fit": {}, "input_path": name, "input_sha256": "0" * 64,
            }
        return {
            "schema": "m1_m10_source_frozen_behavior_manifold_nested_loso_v1",
            "target_support_or_query_used_for_candidate_selection": False,
            "sessions": names, "output_names": ["a"], "lambda_grid_per_sample": [0.3],
            "directridge_equal_session": {"mean_r2": 0.5},
            "candidate_specs": [{"kind": kind, "latent_dim": 1, "hidden_dim": 0 if kind == "pca" else 2, "raw_loss_fraction": 0.5, "activation": "gelu"}],
            "folds": folds,
        }
    # Family A has deliberately worse target scores but better source scores;
    # the combiner must still choose A and never optimize on the target.
    a = tmp_path / "a.json"; b = tmp_path / "b.json"
    a.write_text(json.dumps(family("pca", (0.9, 0.9), (0.1, 0.1))))
    b.write_text(json.dumps(family("mlp", (0.8, 0.8), (0.9, 0.9))))
    result = combine_family_results((a, b))
    assert result["target_used_for_selection"] is False
    assert result["equal_session"]["mean_r2"] == pytest.approx(0.1)


def test_calibration_aware_fit_is_source_frozen_and_decodes() -> None:
    sessions = {f"s{index}": _nested_session(f"s{index}", index) for index in range(2)}
    manifold = fit_calibration_aware_manifold(
        sessions, CalibrationAwareSpec(2, 8, 0.1, pretrain_epochs=2, fine_steps=5),
        device=torch.device("cpu"), seed=42, query_batch_size=64, reconstruction_batch_size=64,
    )
    value = sessions["s0"].target[:20]
    reconstructed = manifold.decode(manifold.encode(value))
    assert reconstructed.shape == value.shape
    assert np.isfinite(reconstructed).all()
    assert manifold.fit_evidence["source_sessions"] == ["s0", "s1"]
    assert manifold.fit_evidence["closed_form_ridge_recomputed_differentiably"] is True


def test_predictive_pca_uses_predictions_not_query_labels() -> None:
    sessions = {f"s{index}": _nested_session(f"s{index}", index) for index in range(2)}
    manifold = fit_predictive_pca(sessions, 2)
    assert manifold.pca is not None
    assert manifold.fit_evidence["source_query_labels_used_to_fit_basis"] is False
    assert manifold.decode(manifold.encode(sessions["s0"].target[:10])).shape == (10, 4)
