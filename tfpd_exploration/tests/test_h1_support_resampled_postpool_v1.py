from __future__ import annotations

import pytest

from tfpd_exploration.h1_series_20260830.src.h1_support_resampled_postpool_v1.core import (
    freeze_decoder_train_identity,
    late_identity,
    native_identity,
    normalized_identity_distill,
    support_index,
)
from tfpd_exploration.h1_series_20260830.src.h1_support_resampled_postpool_v1.plan import (
    DATE_ORDER,
    decide_oof,
)
from tfpd_exploration.h1_series_20260830.src.h1_support_resampled_postpool_v1.evaluate import (
    _load_branch_checkpoint,
    _save_branch_checkpoint,
)


def _rows(fixed: list[float], random_values: list[float], srpd: list[float]):
    return [
        {
            "outer_date": date,
            "scores": {
                "FROZEN-C1": 0.3,
                "LP-F3": 0.3 + a,
                "LP-R3": 0.3 + b,
                "SRPD": 0.3 + c,
            },
        }
        for date, a, b, c in zip(DATE_ORDER, fixed, random_values, srpd)
    ]


def test_decision_prefers_srpd_when_primary_and_safety_pass() -> None:
    result = decide_oof(_rows([0.004] * 5, [0.006] * 5, [0.007] * 5))
    assert result["pass"] is True
    assert result["selected_arm"] == "SRPD"


def test_decision_falls_back_to_random_then_fixed() -> None:
    random_result = decide_oof(_rows([0.003] * 5, [0.006] * 5, [-0.02] * 5))
    assert random_result["selected_arm"] == "LP-R3"
    fixed_result = decide_oof(_rows([0.006] * 5, [-0.02] * 5, [-0.02] * 5))
    assert fixed_result["selected_arm"] == "LP-F3"


def test_decision_stops_on_one_date_outlier() -> None:
    values = [0.02, 0.01, 0.01, 0.01, -0.03]
    result = decide_oof(_rows(values, values, values))
    assert result["pass"] is False
    assert result["selected_arm"] is None


def test_carrier_extension_requires_effect_and_common_direction() -> None:
    result = decide_oof(_rows([0.0] * 5, [0.004] * 5, [0.004] * 5))
    assert result["carrier_extension_triggered"] is True
    assert "LP-R3_MINUS_LP-F3" in result["carrier_extension_triggered_contrasts"]


def test_support_schedule_is_anchored_and_shared() -> None:
    assert support_index(epoch=0, session="s", batch_ordinal=0, block_count=9, random_arm=False) == (0, "first_m3")
    assert support_index(epoch=0, session="s", batch_ordinal=1, block_count=9, random_arm=False) == (0, "first_m3")
    first = support_index(epoch=0, session="s", batch_ordinal=0, block_count=9, random_arm=True)
    random_choice = support_index(epoch=0, session="s", batch_ordinal=1, block_count=9, random_arm=True)
    assert first == (0, "first_m3")
    assert 1 <= random_choice[0] < 9 and random_choice[1] == "nonfirst_uniform"
    assert random_choice == support_index(epoch=0, session="s", batch_ordinal=1, block_count=9, random_arm=True)


def _toy():
    torch = pytest.importorskip("torch")

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.zero_carrier = False
            self.other = torch.nn.Linear(2, 2)
            self.carrier_pre_pool = torch.nn.Sequential(torch.nn.Linear(1024, 32), torch.nn.ReLU())
            self.carrier_post_pool = torch.nn.Sequential(
                torch.nn.Linear(36, 32), torch.nn.ReLU(),
                torch.nn.Linear(32, 32), torch.nn.ReLU(),
                torch.nn.Linear(32, 700),
            )

    return Toy()


def test_only_identity_branch_is_trainable() -> None:
    model = _toy()
    parameters = freeze_decoder_train_identity(model)
    assert sum(parameter.numel() for parameter in parameters) == 58_140
    assert model.other.weight.requires_grad is False
    assert model.carrier_pre_pool[0].weight.requires_grad is True
    assert model.training is False


def test_late_pool_is_exact_mean_after_post_mlp_and_differs_from_native() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(2)
    model = _toy()
    activity = torch.randn(1, 3, 1024, 176)
    carrier = torch.randn(1, 176, 4)
    late = late_identity(model, activity, carrier)
    encoded = model.carrier_pre_pool(activity.permute(0, 1, 3, 2))
    expected = model.carrier_post_pool(torch.cat((encoded, carrier[:, None].expand(-1, 3, -1, -1)), dim=-1)).mean(dim=1)
    native = native_identity(model, activity, carrier)
    assert torch.equal(late, expected)
    assert not torch.equal(late, native)


def test_normalized_distillation_is_finite_and_reaches_student() -> None:
    torch = pytest.importorskip("torch")
    student = torch.nn.Parameter(torch.randn(1, 4, 7))
    teacher = torch.randn(1, 4, 7)
    loss = normalized_identity_distill(student, teacher)
    loss.backward()
    assert torch.isfinite(loss)
    assert student.grad is not None and int(torch.count_nonzero(student.grad)) > 0


def test_branch_checkpoint_roundtrip_is_strict(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    base = _toy()
    trained = _toy()
    trained.load_state_dict(base.state_dict(), strict=True)
    with torch.no_grad():
        trained.carrier_post_pool[-1].bias.add_(0.125)
    path = tmp_path / "branch.pt"
    evidence = _save_branch_checkpoint(path, outer_date="19250108", arm="LP-F3", net=trained)
    restored = _load_branch_checkpoint(base, path, expected=evidence, device="cpu")
    for name, value in trained.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    assert path.stat().st_mode & 0o777 == 0o444
