from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tfpd_exploration.h1_series_20260830.src.h1_calibration_profile_film_v1.core import (
    H1CalibrationProfileFiLMError,
    build_film,
    film_identity,
    profile_from_arrays,
    profile_from_support,
)
from tfpd_exploration.h1_series_20260830.src.h1_calibration_profile_film_v1.plan import (
    DATE_ORDER,
    FILM_PARAMETERS,
    PROFILE_MASK,
    decide_oof,
)
from tfpd_exploration.h1_series_20260830.src.h1_support_resampled_postpool_v1.core import (
    late_identity,
    native_identity,
)


def _toy_net():
    import torch

    return SimpleNamespace(
        carrier_pre_pool=torch.nn.Sequential(torch.nn.Linear(1024, 32), torch.nn.ReLU()),
        carrier_post_pool=torch.nn.Sequential(
            torch.nn.Linear(36, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 700),
        ),
        zero_carrier=False,
    )


def _trainable_toy_net():
    import torch

    class Transformer(torch.nn.Module):
        def forward(self, query, source):
            return query + source.mean(dim=1, keepdim=True), None

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.carrier_pre_pool = torch.nn.Sequential(torch.nn.Linear(1024, 32), torch.nn.ReLU())
            self.carrier_post_pool = torch.nn.Sequential(
                torch.nn.Linear(36, 32), torch.nn.ReLU(),
                torch.nn.Linear(32, 32), torch.nn.ReLU(),
                torch.nn.Linear(32, 700),
            )
            self.fc_in = torch.nn.Linear(700, 4)
            self.rep = torch.nn.Parameter(torch.randn(7, 700))
            self.transformer = Transformer()
            self.fc_out = torch.nn.Linear(4, 700)
            self.zero_carrier = False
            self.dynamic_dropout = False
            self.dropout_rate = 0.0

    return Net()


def test_profile_geometry_mask_and_finiteness():
    rng = np.random.default_rng(3)
    neural = rng.poisson(0.2, size=(240, 176)).astype(np.float32)
    velocity = rng.normal(size=(240, 7)).astype(np.float32)
    profile, evidence = profile_from_arrays(neural, velocity)
    assert profile.shape == (176, 4)
    assert np.isfinite(profile).all()
    assert np.count_nonzero(profile[:, 2:]) == 0
    assert evidence["profile_mask"] == list(PROFILE_MASK)
    assert evidence["low_state_bins"] >= 16
    assert evidence["high_state_bins"] >= 16
    assert evidence["raw_profile"].shape == (176, 4)


def test_profile_rejects_degenerate_speed():
    neural = np.ones((100, 176), dtype=np.float32)
    velocity = np.zeros((100, 7), dtype=np.float32)
    with pytest.raises(H1CalibrationProfileFiLMError, match="quartiles"):
        profile_from_arrays(neural, velocity)


def test_profile_reads_only_declared_support_trials():
    rng = np.random.default_rng(5)
    record = SimpleNamespace(
        eval_mask=np.ones(180, dtype=bool),
        trial_num=np.repeat(np.arange(1, 7, dtype=np.float64), 30),
        neural=rng.poisson(0.3, size=(180, 176)).astype(np.float32),
        velocity=rng.normal(size=(180, 7)).astype(np.float32),
    )
    first, first_evidence = profile_from_support(record, (1.0, 2.0, 3.0))
    record.neural[90:] += 1000.0
    record.velocity[90:] *= -200.0
    second, second_evidence = profile_from_support(record, (1.0, 2.0, 3.0))
    assert np.array_equal(first, second)
    assert first_evidence["support_trials"] == second_evidence["support_trials"] == [1.0, 2.0, 3.0]


def test_zero_film_is_bitwise_native_for_both_pooling_positions():
    import torch

    torch.manual_seed(7)
    net = _toy_net()
    film = build_film()
    activity = torch.randn(1, 3, 1024, 176)
    carrier = torch.randn(1, 176, 4)
    profile = torch.randn(1, 176, 4)
    early = film_identity(net, activity, carrier, profile, film, late=False)
    late = film_identity(net, activity, carrier, profile, film, late=True)
    assert torch.equal(early, native_identity(net, activity, carrier))
    assert torch.equal(late, late_identity(net, activity, carrier))
    assert sum(parameter.numel() for parameter in film.parameters()) == FILM_PARAMETERS


def test_same_film_operator_changes_both_pooling_paths_after_nonzero_head():
    import torch

    torch.manual_seed(11)
    net = _toy_net()
    film = build_film()
    with torch.no_grad():
        film[2].bias.fill_(0.05)
    activity = torch.randn(1, 3, 1024, 176)
    carrier = torch.randn(1, 176, 4)
    profile = torch.randn(1, 176, 4)
    early = film_identity(net, activity, carrier, profile, film, late=False)
    late = film_identity(net, activity, carrier, profile, film, late=True)
    assert not torch.equal(early, native_identity(net, activity, carrier))
    assert not torch.equal(late, late_identity(net, activity, carrier))


def _rows(early, late):
    rows = []
    for date, eg, lg in zip(DATE_ORDER, early, late, strict=True):
        rows.append({
            "outer_date": date,
            "scores": {
                "EP-ZERO": 0.40,
                "EP-FILM": 0.40 + eg,
                "LP-ZERO": 0.44,
                "LP-FILM": 0.44 + lg,
            },
        })
    return rows


def test_decision_keeps_pooling_and_film_as_separate_factors():
    both = decide_oof(_rows([0.01] * 5, [0.008] * 5))
    assert both["classification"] == "FILM_POOLING_ROBUST"
    assert both["selected_h1_product"] == "LP-FILM"
    early = decide_oof(_rows([0.01] * 5, [-0.002] * 5))
    assert early["classification"] == "FILM_EARLY_REPLICATION_ONLY"
    assert early["m2_to_h1_early_replication"] is True
    assert early["selected_h1_product"] == "LP-R3"
    late = decide_oof(_rows([-0.002] * 5, [0.01] * 5))
    assert late["classification"] == "FILM_LATE_SUBSTRATE_ONLY"
    assert late["m2_to_h1_early_replication"] is False
    assert late["selected_h1_product"] == "LP-FILM"


def test_paired_runner_updates_only_the_two_film_modules():
    import copy
    import torch
    from tfpd_exploration.h1_series_20260830.src.h1_calibration_profile_film_v1.evaluate import train_pair
    from src.h1_m4_cce_contract import state_hash

    torch.manual_seed(13)
    early = _trainable_toy_net()
    late = copy.deepcopy(early)
    before = (state_hash(early.state_dict()), state_hash(late.state_dict()))
    rng = np.random.default_rng(17)
    support = []
    for _ in range(2):
        support.append({
            "activity": rng.normal(size=(3, 1024, 176)).astype(np.float32),
            "carrier": rng.normal(size=(176, 4)).astype(np.float32),
            "profile": rng.normal(size=(176, 4)).astype(np.float32),
        })
    row = {
        "session": "synthetic",
        "train_endpoints": np.array([0], dtype=np.int64),
        "neural": rng.normal(size=(1, 176)).astype(np.float32),
        "target_stream": rng.normal(size=(1, 7)).astype(np.float32),
        "support_bank": support,
    }
    films, receipt = train_pair(early, late, [row], device="cpu")
    assert receipt["steps_per_arm"] == 12
    assert receipt["first_identity_bitwise_equal"] == {"EP-FILM": True, "LP-FILM": True}
    assert receipt["first_prediction_bitwise_equal"] == {"EP-FILM": True, "LP-FILM": True}
    assert before == (state_hash(early.state_dict()), state_hash(late.state_dict()))
    assert all(receipt["initial_film_state_sha256"][arm] != receipt["final_film_state_sha256"][arm]
               for arm in films)


def test_actual_lp_r3_predecessor_is_bound():
    from pathlib import Path
    from tfpd_exploration.h1_series_20260830.src.h1_calibration_profile_film_v1.evaluate import load_predecessor

    repo = Path(__file__).resolve().parents[2]
    folds, authority = load_predecessor(repo)
    assert tuple(folds) == DATE_ORDER
    assert authority["terminal_sha256"].startswith("c047bbf5")
    assert all(folds[date]["checkpoints"]["LP-R3"]["sha256"] for date in DATE_ORDER)
