"""No-data CPU contracts for the M2 hold-vs-reach FiLM probe."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
STREAMING_ROOT = REPO_ROOT / "streaming_calibration_exp"
for _path in (str(STREAMING_ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder

from tfpd_exploration.src.m2_hold_film_probe_v1 import core, plan
from tfpd_exploration.src.m2_hold_film_probe_v1.coeffs import (
    ContrastError,
    contrast_from_trial_rates,
    hold_reach_masks,
    shuffle_contrast,
)
from tfpd_exploration.src.m2_hold_film_probe_v1.encoder import HoldContrastFiLMEarlyPoolEncoder


def _synthetic_calib(*, seed: int = 3):
    rng = np.random.default_rng(seed)
    angles = np.full(40, np.nan, dtype=np.float64)
    angles[[3, 9, 11, 21, 31]] = [0.0, np.pi / 2, np.pi, -np.pi / 2, 0.4]
    lengths = np.full(40, 100.0, dtype=np.float64)
    hold = ~np.isfinite(angles[:30])
    reach = np.isfinite(angles[:30])
    rates = np.full((40, 96), 0.4, dtype=np.float64)
    rates[:30][hold] = 0.2 + 0.01 * np.arange(96)
    rates[:30][reach] = 0.8 + 0.02 * np.arange(96)
    rates[30:] = 9.0
    sums = rates * lengths[:, None]
    return sums, lengths, angles


def test_plan_pins_frozen_t4_and_forbids_the_wrong_levers() -> None:
    assert plan.CHECKPOINT_SHA256 == "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
    assert plan.SEALED_M30_EXTERNAL == pytest.approx(0.29521985196829853)
    assert plan.SIDE_DIM == 8
    assert plan.T4_DIM == 4
    assert plan.CONTRAST_DIM == 4
    assert plan.HIDDEN_DIM + plan.T4_DIM == 68


def test_hold_is_nonfinite_angle_and_ignores_trials_after_30() -> None:
    sums, lengths, angles = _synthetic_calib()
    hold, reach = hold_reach_masks(angles)
    assert int(hold.sum()) == 26
    assert int(reach.sum()) == 4
    assert 31 not in np.flatnonzero(reach)
    contrast = contrast_from_trial_rates(sums, lengths, angles)
    mutated_sums = sums.copy()
    mutated_angles = angles.copy()
    mutated_sums[30:] = 0.0
    mutated_angles[31] = 1.7
    assert np.array_equal(
        contrast,
        contrast_from_trial_rates(mutated_sums, lengths.copy(), mutated_angles),
    )
    with pytest.raises(ContrastError):
        contrast_from_trial_rates(sums[:10], lengths[:10], angles[:10])


def test_contrast_uses_relative_hold_reach_not_absolute_rate() -> None:
    sums, lengths, angles = _synthetic_calib()
    contrast = contrast_from_trial_rates(sums, lengths, angles)
    assert contrast.shape == (96, 4)
    assert np.isfinite(contrast).all()
    shifted = sums.copy()
    shifted[:30] += 50.0 * lengths[:30, None]
    shifted_contrast = contrast_from_trial_rates(shifted, lengths, angles)
    assert np.allclose(contrast[:, 0], shifted_contrast[:, 0], atol=1e-5)
    signature = inspect.signature(contrast_from_trial_rates)
    assert set(signature.parameters) == {"spike_sums", "lengths", "angles", "horizon"}


def test_shuffle_destroys_channel_alignment() -> None:
    sums, lengths, angles = _synthetic_calib()
    contrast = contrast_from_trial_rates(sums, lengths, angles)
    shuffled = shuffle_contrast(contrast, seed=42)
    assert shuffled.shape == contrast.shape
    assert not np.array_equal(shuffled, contrast)
    assert np.array_equal(np.sort(shuffled, axis=0), np.sort(contrast, axis=0))


def test_zero_init_film_is_bitwise_native_t4() -> None:
    torch.manual_seed(17)
    calib = torch.rand(2, 30, 100, 8)
    t4 = torch.randn(2, 8, 4)
    contrast = torch.randn(2, 8, 4)
    side = torch.cat([t4, contrast], dim=-1)
    baseline = SideFeatureEarlyPoolEncoder(100, 50, 64, side_dim=4)
    film = HoldContrastFiLMEarlyPoolEncoder(100, 50, 64, side_dim=8)
    film.load_t4_state_dict(baseline.state_dict())
    assert torch.equal(film.pre_pool[0].weight, baseline.pre_pool[0].weight)
    assert torch.equal(film.post_pool[0].weight, baseline.post_pool[0].weight)
    assert int(film.post_pool[0].in_features) == 68
    assert torch.count_nonzero(film.contrast_film.weight).item() == 0
    torch.manual_seed(plan.SEED)
    seeded_a = HoldContrastFiLMEarlyPoolEncoder(100, 50, 64, side_dim=8)
    torch.manual_seed(plan.SEED)
    seeded_b = HoldContrastFiLMEarlyPoolEncoder(100, 50, 64, side_dim=8)
    assert torch.equal(seeded_a.contrast_context[0].weight, seeded_b.contrast_context[0].weight)
    with torch.no_grad():
        expected = baseline.forward_batch(calib, side_features=t4)
        observed = film.forward_batch(calib, side_features=side)
    assert torch.equal(observed, expected)


def test_nonzero_film_reads_contrast_and_freeze_set_is_exact() -> None:
    torch.manual_seed(4)
    calib = torch.rand(1, 30, 100, 8)
    t4 = torch.randn(1, 8, 4)
    contrast = torch.randn(1, 8, 4)
    film = HoldContrastFiLMEarlyPoolEncoder(100, 50, 16, side_dim=8, film_rank=4)
    film.freeze_base_path()
    assert film.trainable_parameter_names() == (
        "contrast_context.0.weight",
        "contrast_context.0.bias",
        "contrast_film.weight",
        "contrast_film.bias",
    )
    assert not film.pre_pool[0].weight.requires_grad
    assert not film.post_pool[0].weight.requires_grad
    with torch.no_grad():
        film.contrast_film.weight.fill_(0.05)
    first = film.forward_batch(calib, side_features=torch.cat([t4, contrast], dim=-1))
    changed = contrast + 2.0
    second = film.forward_batch(calib, side_features=torch.cat([t4, changed], dim=-1))
    assert not torch.equal(first, second)
    shuffled = contrast[:, torch.randperm(8), :]
    third = film.forward_batch(calib, side_features=torch.cat([t4, shuffled], dim=-1))
    assert not torch.equal(first, third)


def _summary(mean: float, n: int = 6) -> dict[str, object]:
    values = {f"s{i}": float(mean) for i in range(n)}
    return core.summarize_sessions(values)


def _contrast(candidate_mean: float, reference_mean: float, *, n: int = 6, positive: int | None = None):
    if positive is None:
        candidate = {f"s{i}": candidate_mean for i in range(n)}
        reference = {f"s{i}": reference_mean for i in range(n)}
    else:
        candidate = {f"s{i}": reference_mean + 0.02 for i in range(positive)}
        candidate.update({f"s{i}": reference_mean - 0.001 for i in range(positive, n)})
        reference = {f"s{i}": reference_mean for i in range(n)}
    return core.paired_contrast(candidate, reference)


def test_verdict_kill_and_continue_law() -> None:
    sealed = plan.SEALED_M30_EXTERNAL
    p0 = _summary(sealed)
    continue_v = core.decide_verdict(
        p0_external=p0,
        p1_external=_summary(sealed + 0.02),
        p1_shuffle_external=p0,
        c2_external=p0,
        p1_vs_p0=_contrast(sealed + 0.02, sealed),
        shuffle_vs_p0=_contrast(sealed, sealed),
        c2_vs_p0=_contrast(sealed, sealed),
    )
    assert continue_v["code"] == "CONTINUE_LABELED_SIGNAL"
    assert continue_v["joint_retrain_warranted"] is True
    assert continue_v["official_champion_claim"] is False

    shuffle_kill = core.decide_verdict(
        p0_external=p0,
        p1_external=_summary(sealed + 0.02),
        p1_shuffle_external=_summary(sealed + 0.01),
        c2_external=p0,
        p1_vs_p0=_contrast(sealed + 0.02, sealed),
        shuffle_vs_p0=_contrast(sealed + 0.01, sealed),
        c2_vs_p0=_contrast(sealed, sealed),
    )
    assert shuffle_kill["code"] == "KILL_SHUFFLE_CAPACITY"
    assert shuffle_kill["joint_retrain_warranted"] is False

    no_signal = core.decide_verdict(
        p0_external=p0,
        p1_external=_summary(sealed - 0.01),
        p1_shuffle_external=p0,
        c2_external=p0,
        p1_vs_p0=_contrast(sealed - 0.01, sealed),
        shuffle_vs_p0=_contrast(sealed, sealed),
        c2_vs_p0=_contrast(sealed, sealed),
    )
    assert no_signal["code"] == "KILL_NO_SIGNAL"
    assert no_signal["joint_retrain_warranted"] is False

    c2_kill = core.decide_verdict(
        p0_external=p0,
        p1_external=_summary(sealed + 0.02),
        p1_shuffle_external=p0,
        c2_external=_summary(sealed + 0.01),
        p1_vs_p0=_contrast(sealed + 0.02, sealed),
        shuffle_vs_p0=_contrast(sealed, sealed),
        c2_vs_p0=_contrast(sealed + 0.01, sealed),
    )
    assert c2_kill["code"] == "KILL_C2_CAPACITY"
    assert c2_kill["joint_retrain_warranted"] is False

    weak_mean = core.decide_verdict(
        p0_external=p0,
        p1_external=_summary(sealed + 0.002),
        p1_shuffle_external=p0,
        c2_external=p0,
        p1_vs_p0=_contrast(sealed + 0.002, sealed),
        shuffle_vs_p0=_contrast(sealed, sealed),
        c2_vs_p0=_contrast(sealed, sealed),
    )
    assert weak_mean["code"] == "WEAK_LABELED_SIGNAL"
    assert weak_mean["joint_retrain_warranted"] is False

    weak_breadth = core.decide_verdict(
        p0_external=p0,
        p1_external=_summary(sealed + 0.02),
        p1_shuffle_external=p0,
        c2_external=p0,
        p1_vs_p0=_contrast(sealed + 0.02, sealed, positive=3),
        shuffle_vs_p0=_contrast(sealed, sealed),
        c2_vs_p0=_contrast(sealed, sealed),
    )
    assert weak_breadth["code"] == "WEAK_LABELED_SIGNAL"
    assert weak_breadth["p1_positive_sessions"] == 3
    assert weak_breadth["joint_retrain_warranted"] is False

    parity = core.decide_verdict(
        p0_external=_summary(0.21),
        p1_external=_summary(0.22),
        p1_shuffle_external=_summary(0.21),
        c2_external=_summary(0.21),
        p1_vs_p0=_contrast(0.22, 0.21),
        shuffle_vs_p0=_contrast(0.21, 0.21),
        c2_vs_p0=_contrast(0.21, 0.21),
    )
    assert parity["code"] == "FAIL_P0_PARITY"
