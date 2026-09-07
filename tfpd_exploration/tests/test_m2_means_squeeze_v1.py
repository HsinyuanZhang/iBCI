"""CPU contracts for native-M33 mean-only FiLM squeeze / EvalAI-max select."""

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

from tfpd_exploration.src.m2_hold_film_probe_v1 import core as probe_core
from tfpd_exploration.src.m2_hold_film_probe_v1.encoder import HoldContrastFiLMEarlyPoolEncoder
from tfpd_exploration.src.m2_means_squeeze_v1 import core, plan


def _summary(mean: float, n: int = 6) -> dict[str, object]:
    return probe_core.summarize_sessions({f"s{i}": float(mean) for i in range(n)})


def _arm(
    name: str,
    mean: float,
    *,
    median: float | None = None,
    positive: int = 6,
    p0: float = 0.2991329172968798,
) -> dict[str, object]:
    sessions = {}
    for i in range(6):
        sessions[f"s{i}"] = p0 + 0.02 if i < positive else p0 - 0.001
    if positive == 6:
        sessions = {f"s{i}": float(mean) for i in range(6)}
    summary = probe_core.summarize_sessions(sessions)
    if median is not None and positive == 6:
        summary = {
            **summary,
            "equal_session_mean": float(mean),
            "equal_session_median": float(median),
        }
    p0_map = {f"s{i}": p0 for i in range(6)}
    return {
        "name": name,
        "summaries": {"external_official_query": summary},
        "vs_p0": probe_core.paired_contrast(
            summary["per_session_r2"] if positive != 6 else {f"s{i}": mean for i in range(6)},
            p0_map,
        ),
    }


def test_plan_pins_frozen_surface_and_evalai_max_policy() -> None:
    assert plan.CHECKPOINT_SHA256 == "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
    assert plan.P0_M33_EXTERNAL == pytest.approx(0.2991329172968798)
    assert plan.MEANS_M33_BASELINE == pytest.approx(0.3190522122162025)
    assert plan.OFFICIAL_T4_578221_EXTERNAL == pytest.approx(0.30324395)
    assert plan.M33_HORIZON == 33
    assert plan.T4_MODE == "native"
    assert plan.SHUFFLE_IS_REJECT is False
    assert plan.EVALAI_PUSH_THIS_CELL is True
    assert plan.WARM_START_T4_PLUS_CONTRAST == "probe_p0"
    assert plan.RESULT_ROOT_RELATIVE.endswith("m2_means_squeeze_v2")
    assert plan.OPT_V1_ROOT_RELATIVE.endswith("m2_p1_contrast_opt_v1")


def test_catalog_covers_underfit_4d_retrain_and_contrast_only() -> None:
    names = [row["name"] for row in plan.CONFIGS]
    assert names[0] == "means_ep6_lr1e4_s42"
    assert "full4d_ep6_lr1e4_s42" in names
    assert "means_contrast_only_ep12_lr1e4_s42" in names
    assert any(row["epochs"] >= 12 and row["mask"] == "means" for row in plan.CONFIGS)
    assert len(plan.CONFIGS) <= 12
    assert len(names) == len(set(names))
    replica = plan.CONFIGS[0]
    assert replica["mask"] == "means"
    assert replica["epochs"] == 6
    assert replica["learning_rate"] == 1.0e-4
    assert replica["seed"] == 42
    assert replica["film_input"] == "t4_plus_contrast"


def test_masks_are_native_contrast_columns() -> None:
    assert plan.MASKS["means"] == (1.0, 1.0, 0.0, 0.0)
    assert plan.MASKS["full4d"] == (1.0, 1.0, 1.0, 1.0)
    assert plan.MASKS["delta"] == (1.0, 0.0, 0.0, 0.0)
    assert plan.MASKS["logratio"] == (0.0, 1.0, 0.0, 0.0)
    assert plan.MASKS["means_reachstd"] == (1.0, 1.0, 0.0, 1.0)


def test_floor_rejects_weak_or_narrow_gains() -> None:
    p0 = _summary(plan.P0_M33_EXTERNAL)
    weak = _arm("weak", plan.P0_M33_EXTERNAL + 0.004)
    assert core.passes_floor(weak, p0) is False
    narrow = _arm("narrow", plan.P0_M33_EXTERNAL + 0.02, positive=3)
    assert core.passes_floor(narrow, p0) is False
    ok = _arm("ok", 0.319)
    assert core.passes_floor(ok, p0) is True


def test_select_picks_highest_local_ho_even_if_shuffle_rises() -> None:
    p0 = _summary(plan.P0_M33_EXTERNAL)
    means = _arm("means_ep6_lr1e4_s42", 0.31905, median=0.290)
    four = _arm("full4d_ep6_lr1e4_s42", 0.322, median=0.288)
    longer = _arm("means_ep18_lr1e4_s42", 0.324, median=0.291)
    choice = core.choose_submit_candidate(
        p0_external=p0,
        arms=[means, four, longer],
        shuffle_means={"means_ep18_lr1e4_s42": 0.314},
    )
    assert choice["candidate"] == "means_ep18_lr1e4_s42"
    assert choice["export_licensed"] is True
    assert choice["evalai_push"] is True
    assert choice["shuffle_reject"] is False
    assert choice["local_heldout_mean"] == pytest.approx(0.324)


def test_select_tie_breaks_median_then_simpler_mask() -> None:
    p0 = _summary(plan.P0_M33_EXTERNAL)
    a = _arm("full4d_ep12_lr1e4_s42", 0.321, median=0.280)
    b = _arm("means_ep12_lr1e4_s42", 0.321, median=0.292)
    choice = core.choose_submit_candidate(p0_external=p0, arms=[a, b])
    assert choice["candidate"] == "means_ep12_lr1e4_s42"


def test_select_falls_back_to_sealed_means_if_sweep_misses_floor() -> None:
    p0 = _summary(plan.P0_M33_EXTERNAL)
    weak = _arm("means_ep6_lr3e4_s42", plan.P0_M33_EXTERNAL + 0.001)
    choice = core.choose_submit_candidate(
        p0_external=p0,
        arms=[weak],
        sealed_means_m33=_arm("sealed_means_m33", plan.MEANS_M33_BASELINE, median=0.290),
    )
    assert choice["candidate"] == "sealed_means_m33"
    assert choice["export_licensed"] is True
    assert choice["fallback_to_sealed_means"] is True


def _film_modulation(film: HoldContrastFiLMEarlyPoolEncoder, t4: torch.Tensor, contrast: torch.Tensor) -> torch.Tensor:
    side = torch.cat([t4, contrast], dim=-1)
    film_in = contrast if film.film_input == "contrast_only" else side
    return film.contrast_film(film.contrast_context(film_in))


def test_contrast_only_film_ignores_t4_in_gamma_beta() -> None:
    torch.manual_seed(4)
    t4 = torch.randn(1, 8, 4)
    t4_shifted = t4 + 1.7
    contrast = torch.randn(1, 8, 4)
    film = HoldContrastFiLMEarlyPoolEncoder(
        100, 50, 16, side_dim=8, film_rank=4, film_input="contrast_only"
    )
    assert film.film_input == "contrast_only"
    assert film.contrast_context[0].in_features == 4
    with torch.no_grad():
        film.contrast_film.weight.fill_(0.05)
    assert torch.equal(_film_modulation(film, t4, contrast), _film_modulation(film, t4_shifted, contrast))
    assert not torch.equal(_film_modulation(film, t4, contrast), _film_modulation(film, t4, contrast + 1.0))


def test_default_film_still_zero_init_native_t4() -> None:
    torch.manual_seed(17)
    calib = torch.rand(2, 30, 100, 8)
    t4 = torch.randn(2, 8, 4)
    contrast = torch.randn(2, 8, 4)
    baseline = SideFeatureEarlyPoolEncoder(100, 50, 64, side_dim=4)
    film = HoldContrastFiLMEarlyPoolEncoder(100, 50, 64, side_dim=8)
    film.load_t4_state_dict(baseline.state_dict())
    with torch.no_grad():
        expected = baseline.forward_batch(calib, side_features=t4)
        observed = film.forward_batch(calib, side_features=torch.cat([t4, contrast], dim=-1))
    assert torch.equal(observed, expected)


def test_train_film_accepts_squeeze_overrides() -> None:
    from tfpd_exploration.src.m2_hold_film_probe_v1.physical import train_film

    signature = inspect.signature(train_film)
    for name in ("learning_rate", "epochs", "seed", "windows_per_session"):
        assert name in signature.parameters
