"""CPU contracts for the selected-T4 confidence-FiLM Stage-0 arm."""
from __future__ import annotations

import pytest
import torch
from torch import nn
import numpy as np

from src.models.streaming_calibration_module import load_selected_t4_full_student_warmstart
from src.models.components.streaming_encoders import (
    CalibrationConfidenceFiLMEarlyPoolEncoder,
    SideFeatureEarlyPoolEncoder,
    build_encoder,
)
from mc_maze.unit_side_features import (
    CANONICAL_DIRECTIONS_RAD,
    permute_t4c_component,
    tuning_fit_confidence_descriptor,
)


def _inputs():
    torch.manual_seed(17)
    calib = torch.rand(2, 30, 100, 4)
    t4 = torch.randn(2, 4, 4)
    confidence = torch.randn(2, 4, 2)
    return calib, t4, torch.cat([t4, confidence], dim=-1)


def test_zero_init_confidence_film_is_exact_selected_t4_baseline():
    calib, t4, side = _inputs()
    baseline = SideFeatureEarlyPoolEncoder(100, 50, 64, side_dim=4)
    film = CalibrationConfidenceFiLMEarlyPoolEncoder(100, 50, 64, side_dim=6)
    film.load_t4_state_dict(baseline.state_dict())
    assert torch.equal(film.pre_pool[0].weight, baseline.pre_pool[0].weight)
    assert torch.equal(film.post_pool[0].weight, baseline.post_pool[0].weight)
    assert torch.count_nonzero(film.confidence_film.weight).item() == 0
    assert torch.count_nonzero(film.confidence_film.bias).item() == 0
    with torch.no_grad():
        expected = baseline.forward_batch(calib, side_features=t4)
        observed = film.forward_batch(calib, side_features=side)
    assert torch.equal(observed, expected)


def test_t4_confidence_context_changes_pooled_activity_modulation_only():
    calib, _t4, side = _inputs()
    film = CalibrationConfidenceFiLMEarlyPoolEncoder(100, 50, 16, side_dim=6)
    with torch.no_grad():
        # Move only the added FiLM head off its zero-init point.
        film.confidence_film.weight.fill_(0.05)
    first = film.forward_batch(calib, side_features=side)
    changed_side = side.clone()
    changed_side[..., 4:] += 2.0
    second = film.forward_batch(calib, side_features=changed_side)
    assert not torch.equal(first, second)


def test_additive_nofilm_control_is_parameter_matched_and_zero_init_equivalent():
    calib, t4, side = _inputs()
    baseline = SideFeatureEarlyPoolEncoder(100, 50, 16, side_dim=4)
    film = CalibrationConfidenceFiLMEarlyPoolEncoder(100, 50, 16, side_dim=6)
    additive = CalibrationConfidenceFiLMEarlyPoolEncoder(
        100, 50, 16, side_dim=6, additive_only=True
    )
    film.load_t4_state_dict(baseline.state_dict())
    additive.load_t4_state_dict(baseline.state_dict())
    assert sum(p.numel() for p in film.parameters()) == sum(p.numel() for p in additive.parameters())
    assert torch.equal(film.forward_batch(calib, side_features=side), baseline.forward_batch(calib, side_features=t4))
    assert torch.equal(additive.forward_batch(calib, side_features=side), baseline.forward_batch(calib, side_features=t4))


def test_residual_only_film_is_parameter_matched_and_masks_geometry_exactly():
    calib, _t4, side = _inputs()
    full = CalibrationConfidenceFiLMEarlyPoolEncoder(
        100, 50, 16, side_dim=6
    )
    residual_only = CalibrationConfidenceFiLMEarlyPoolEncoder(
        100,
        50,
        16,
        side_dim=6,
        confidence_mask=(True, False),
    )
    residual_only.load_state_dict(full.state_dict(), strict=True)
    with torch.no_grad():
        residual_only.confidence_film.weight.fill_(0.05)
    assert sum(p.numel() for p in full.parameters()) == sum(
        p.numel() for p in residual_only.parameters()
    )
    assert "confidence_mask" not in residual_only.state_dict()
    reference = residual_only.forward_batch(calib, side_features=side)
    changed_geometry = side.clone()
    changed_geometry[..., 5] += 100.0
    changed_residual = side.clone()
    changed_residual[..., 4] += 2.0
    assert torch.equal(
        residual_only.forward_batch(
            calib, side_features=changed_geometry
        ),
        reference,
    )
    assert not torch.equal(
        residual_only.forward_batch(
            calib, side_features=changed_residual
        ),
        reference,
    )


def test_residual_only_builder_variants_keep_six_wide_parameter_matched_context():
    residual = build_encoder(
        "B3SCFR",
        window_size=50,
        trial_length=100,
        hidden_dim=16,
        side_dim=6,
    )
    shuffled = build_encoder(
        "B3SCFRS",
        window_size=50,
        trial_length=100,
        hidden_dim=16,
        side_dim=6,
    )
    additive = build_encoder(
        "B3SCFRA",
        window_size=50,
        trial_length=100,
        hidden_dim=16,
        side_dim=6,
    )
    assert torch.equal(
        residual.confidence_mask, torch.tensor([1.0, 0.0])
    )
    assert torch.equal(shuffled.confidence_mask, residual.confidence_mask)
    assert torch.equal(additive.confidence_mask, residual.confidence_mask)
    assert additive.additive_only is True
    assert len(
        {
            sum(parameter.numel() for parameter in encoder.parameters())
            for encoder in (residual, shuffled, additive)
        }
    ) == 1


def test_b3scf_build_fails_without_t4_and_rejects_non_t4_warmstart():
    with pytest.raises(ValueError, match="side_features"):
        build_encoder("B3SCF", window_size=50, trial_length=100, hidden_dim=16, side_dim=0)
    film = CalibrationConfidenceFiLMEarlyPoolEncoder(100, 50, 16, side_dim=6)
    with pytest.raises(ValueError, match="exact ordinary B3S/T4"):
        film.load_t4_state_dict({"not_a_t4_weight": torch.zeros(1)})


def test_selected_t4_full_student_warmstart_keeps_complete_prediction_bitwise_equal(tmp_path):
    """Decoder and T4 substrate—not optimizer state—are shared by continuation arms."""
    calib, t4, side = _inputs()
    selected_decoder = nn.Linear(50, 3)
    selected_t4 = SideFeatureEarlyPoolEncoder(100, 50, 64, side_dim=4)
    state = {
        **{f"student.decoder.{key}": value for key, value in selected_decoder.state_dict().items()},
        **{f"student.id_encoder.{key}": value for key, value in selected_t4.state_dict().items()},
    }
    checkpoint = tmp_path / "selected_t4.ckpt"
    torch.save({"state_dict": state, "optimizer_states": [{"not_loaded": True}]}, checkpoint)
    continuation_decoder = nn.Linear(50, 3)
    continuation_t4 = SideFeatureEarlyPoolEncoder(100, 50, 64, side_dim=4)
    film_decoder = nn.Linear(50, 3)
    film = CalibrationConfidenceFiLMEarlyPoolEncoder(100, 50, 64, side_dim=6)
    load_selected_t4_full_student_warmstart(continuation_decoder, continuation_t4, checkpoint)
    load_selected_t4_full_student_warmstart(film_decoder, film, checkpoint)
    assert all(torch.equal(continuation_decoder.state_dict()[key], selected_decoder.state_dict()[key]) for key in selected_decoder.state_dict())
    assert all(torch.equal(film_decoder.state_dict()[key], selected_decoder.state_dict()[key]) for key in selected_decoder.state_dict())
    assert torch.count_nonzero(film.confidence_film.weight).item() == 0
    with torch.no_grad():
        selected_prediction = selected_decoder(selected_t4.forward_batch(calib, side_features=t4))
        continuation_prediction = continuation_decoder(continuation_t4.forward_batch(calib, side_features=t4))
        film_prediction = film_decoder(film.forward_batch(calib, side_features=side))
    assert torch.equal(continuation_prediction, selected_prediction)
    assert torch.equal(film_prediction, selected_prediction)


def test_fit_confidence_formula_and_component_controls_are_exact():
    # Direction 0 is intentionally over-represented: confidence must evaluate
    # residuals around the selected equal-per-direction-mean T4 fit, not around
    # a second trial-weighted least-squares fit.
    directions = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5], dtype=np.int64)
    theta = np.asarray([CANONICAL_DIRECTIONS_RAD[index] for index in directions])
    design = np.stack([np.ones_like(theta), np.cos(theta), np.sin(theta)], axis=1)
    rates = design @ np.array([3.0, 1.0, -0.5]) + np.array(
        [1.2, 0.8, 1.1, 0.9, -0.2, 0.15, -0.1, 0.05, -0.15]
    )
    present = sorted(set(directions))
    direction_theta = np.asarray([CANONICAL_DIRECTIONS_RAD[index] for index in present])
    direction_design = np.stack(
        [np.ones_like(direction_theta), np.cos(direction_theta), np.sin(direction_theta)],
        axis=1,
    )
    direction_means = np.asarray([rates[directions == index].mean() for index in present])
    selected_beta, *_ = np.linalg.lstsq(direction_design, direction_means, rcond=None)
    selected_t4 = np.asarray(
        [
            selected_beta[1],
            selected_beta[2],
            np.hypot(selected_beta[1], selected_beta[2]),
            selected_beta[0],
        ]
    )
    observed = tuning_fit_confidence_descriptor(
        rates, directions, selected_t4=selected_t4
    )
    assert np.array_equal(
        observed, tuning_fit_confidence_descriptor(rates, directions)
    )
    residual_variance = np.sum((rates - design @ selected_beta) ** 2) / (rates.size - 3)
    design_covariance = np.linalg.inv(design.T @ design)
    expected = np.array([
        np.log(residual_variance + 1e-8),
        0.5 * np.log(np.linalg.cond(design_covariance[1:3, 1:3])),
    ], dtype=np.float32)
    assert np.allclose(observed, expected)
    trial_weighted_beta, *_ = np.linalg.lstsq(design, rates, rcond=None)
    trial_weighted_variance = np.sum(
        (rates - design @ trial_weighted_beta) ** 2
    ) / (rates.size - 3)
    assert not np.isclose(residual_variance, trial_weighted_variance)
    features = np.arange(30, dtype=np.float32).reshape(5, 6)
    t4_shuffled = permute_t4c_component(features, component="t4", permutation_seed=42)
    confidence_shuffled = permute_t4c_component(features, component="confidence", permutation_seed=42)
    residual_shuffled = permute_t4c_component(
        features, component="residual", permutation_seed=42
    )
    assert np.array_equal(t4_shuffled[:, 4:], features[:, 4:])
    assert np.array_equal(confidence_shuffled[:, :4], features[:, :4])
    assert np.array_equal(residual_shuffled[:, :4], features[:, :4])
    assert np.array_equal(residual_shuffled[:, 5], features[:, 5])
    assert np.array_equal(
        np.sort(residual_shuffled[:, 4]), np.sort(features[:, 4])
    )
    assert not np.array_equal(residual_shuffled[:, 4], features[:, 4])
    assert np.array_equal(np.sort(t4_shuffled[:, :4], axis=0), np.sort(features[:, :4], axis=0))
    assert np.array_equal(np.sort(confidence_shuffled[:, 4:], axis=0), np.sort(features[:, 4:], axis=0))
    with pytest.raises(ValueError, match="rank=3"):
        tuning_fit_confidence_descriptor(np.ones(3), np.array([0, 0, 1]))


def test_fit_confidence_separates_unit_noise_scale_from_session_design_shape():
    balanced_directions = np.tile(np.arange(8, dtype=np.int64), 2)
    theta = np.asarray(
        [CANONICAL_DIRECTIONS_RAD[index] for index in balanced_directions]
    )
    design = np.stack([np.ones_like(theta), np.cos(theta), np.sin(theta)], axis=1)
    beta = np.array([4.0, 0.8, -0.3])
    noise = np.linspace(-0.4, 0.4, balanced_directions.size)
    rates = design @ beta + noise
    selected_t4 = np.array([beta[1], beta[2], np.hypot(beta[1], beta[2]), beta[0]])

    base = tuning_fit_confidence_descriptor(
        rates, balanced_directions, selected_t4=selected_t4
    )
    scaled = tuning_fit_confidence_descriptor(
        rates * 3.0,
        balanced_directions,
        selected_t4=selected_t4 * 3.0,
    )
    assert np.isclose(scaled[0] - base[0], 2.0 * np.log(3.0), atol=1e-5)
    assert scaled[1] == base[1]

    imbalanced_directions = np.array(
        [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7],
        dtype=np.int64,
    )
    imbalanced_theta = np.asarray(
        [CANONICAL_DIRECTIONS_RAD[index] for index in imbalanced_directions]
    )
    imbalanced_design = np.stack(
        [
            np.ones_like(imbalanced_theta),
            np.cos(imbalanced_theta),
            np.sin(imbalanced_theta),
        ],
        axis=1,
    )
    imbalanced_rates = imbalanced_design @ beta + noise
    imbalanced = tuning_fit_confidence_descriptor(
        imbalanced_rates,
        imbalanced_directions,
        selected_t4=selected_t4,
    )
    assert imbalanced[1] > base[1]
