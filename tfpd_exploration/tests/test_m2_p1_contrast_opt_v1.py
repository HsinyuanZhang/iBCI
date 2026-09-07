"""CPU contracts for P1 contrast optimization / EvalAI-prep gates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tfpd_exploration.src.m2_hold_film_probe_v1 import core as probe_core
from tfpd_exploration.src.m2_p1_contrast_opt_v1 import core, plan


def test_cost_card_is_cached_identity_zero_online_overhead() -> None:
    card = core.cost_card()
    assert card["online_extra_parameters"] == 0
    assert card["online_extra_mac"] == 0
    assert card["film_parameters"] == 1224
    assert card["evalai_push_this_cell"] is False
    assert card["tta"] is False


def test_keeps_gain_is_80_percent_of_sealed_p1() -> None:
    assert core.keeps_gain(plan.SEALED_P1_EXTERNAL) is True
    assert core.keeps_gain(plan.SEALED_P0_EXTERNAL + 0.5 * plan.SEALED_P1_GAIN) is False
    assert abs(plan.SEALED_P1_GAIN - 0.01341006773350297) < 1e-12


def _summary(mean: float, n: int = 6) -> dict[str, object]:
    return probe_core.summarize_sessions({f"s{i}": float(mean) for i in range(n)})


def _delta(cand: float, ref: float) -> dict[str, object]:
    return probe_core.paired_contrast(
        {f"s{i}": cand for i in range(6)},
        {f"s{i}": ref for i in range(6)},
    )


def test_submit_prefers_means_when_it_keeps_full_p1_m33_gain() -> None:
    p0 = _summary(0.30)
    choice = core.choose_submit_candidate(
        p0_m33=p0,
        p1_m33=_summary(0.32),
        means_m33=_summary(0.318),
        p1_m33_shuffle=p0,
        means_m33_shuffle=p0,
        p1_m33_vs_p0=_delta(0.32, 0.30),
        means_m33_vs_p0=_delta(0.318, 0.30),
        means_shuffle_vs_p0=_delta(0.30, 0.30),
        p1_shuffle_vs_p0=_delta(0.30, 0.30),
    )
    assert choice["candidate"] == "means_m33_retrain"
    assert choice["export_licensed"] is True
    assert choice["evalai_push"] is False


def test_submit_keeps_full_p1_when_means_lose_the_gain() -> None:
    p0 = _summary(0.30)
    choice = core.choose_submit_candidate(
        p0_m33=p0,
        p1_m33=_summary(0.32),
        means_m33=_summary(0.306),
        p1_m33_shuffle=p0,
        means_m33_shuffle=p0,
        p1_m33_vs_p0=_delta(0.32, 0.30),
        means_m33_vs_p0=_delta(0.306, 0.30),
        means_shuffle_vs_p0=_delta(0.30, 0.30),
        p1_shuffle_vs_p0=_delta(0.30, 0.30),
    )
    assert choice["candidate"] == "p1_full_m33_transfer"
    assert choice["export_licensed"] is True


def test_submit_refuses_export_when_shuffle_also_rises() -> None:
    p0 = _summary(0.30)
    choice = core.choose_submit_candidate(
        p0_m33=p0,
        p1_m33=_summary(0.32),
        means_m33=_summary(0.32),
        p1_m33_shuffle=_summary(0.31),
        means_m33_shuffle=_summary(0.31),
        p1_m33_vs_p0=_delta(0.32, 0.30),
        means_m33_vs_p0=_delta(0.32, 0.30),
        means_shuffle_vs_p0=_delta(0.31, 0.30),
        p1_shuffle_vs_p0=_delta(0.31, 0.30),
    )
    assert choice["export_licensed"] is False
    assert choice["candidate"] is None


def test_contrast_mask_zeros_selected_columns() -> None:
    contrast = np.ones((96, 4), dtype=np.float32)
    mask = np.asarray(plan.MASKS["means"], dtype=np.float32)
    masked = contrast * mask[None, :]
    assert np.allclose(masked[:, :2], 1.0)
    assert np.allclose(masked[:, 2:], 0.0)
