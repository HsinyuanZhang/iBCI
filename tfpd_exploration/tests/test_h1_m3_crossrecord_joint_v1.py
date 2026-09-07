from __future__ import annotations

import random

import numpy as np
import pytest

from tfpd_exploration.h1_series_20260830.src.h1_m3_crossrecord_joint_v1.core import (
    capture_rng,
    joint_identity,
    native_and_post_identity,
    restore_rng,
    rng_equal,
    window_batch,
)
from tfpd_exploration.h1_series_20260830.src.h1_m3_crossrecord_joint_v1.plan import (
    DATE_ORDER,
    decide_oof,
)


def _rows(native_gain: list[float], joint_gain: list[float]):
    return [
        {
            "outer_date": date,
            "scores": {
                "FROZEN-C1": 0.3,
                "N3-XR12": 0.3 + n,
                "J3-XR12": 0.3 + j,
            },
        }
        for date, n, j in zip(DATE_ORDER, native_gain, joint_gain)
    ]


def test_decision_prefers_joint_only_when_final_and_incremental_gates_pass() -> None:
    result = decide_oof(_rows([0.006] * 5, [0.008] * 5))
    assert result["pass"] is True
    assert result["selected_arm"] == "J3-XR12"


def test_decision_falls_back_to_native() -> None:
    result = decide_oof(_rows([0.007] * 5, [0.006, 0.006, 0.006, -0.002, -0.002]))
    assert result["native_gate"]["pass"] is True
    assert result["joint_increment_gate"]["pass"] is False
    assert result["selected_arm"] == "N3-XR12"


def test_decision_stops_when_gain_is_outlier_driven() -> None:
    result = decide_oof(_rows([0.04, -0.003, -0.002, -0.001, -0.004], [0.04, -0.003, -0.002, -0.001, -0.004]))
    assert result["pass"] is False
    assert result["selected_arm"] is None


def test_window_batch_is_zero_left_padded_and_endpoint_inclusive() -> None:
    neural = np.arange(4 * 176, dtype=np.float32).reshape(4, 176)
    result = window_batch(neural, np.asarray([0, 3], dtype=np.int64))
    assert result.shape == (2, 700, 176)
    assert np.count_nonzero(result[0, :-1]) == 0
    assert np.array_equal(result[0, -1], neural[0])
    assert np.array_equal(result[1, -4:], neural)


def test_rng_snapshot_roundtrip_cpu() -> None:
    torch = pytest.importorskip("torch")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    before = capture_rng()
    _ = random.random(), np.random.rand(), torch.rand(3)
    restore_rng(before)
    assert rng_equal(before, capture_rng())


def test_joint_zero_is_value_exact_and_gradient_reaches_alpha() -> None:
    torch = pytest.importorskip("torch")
    native = torch.randn(1, 176, 700, dtype=torch.float32)
    post = native + torch.randn_like(native) * 0.1
    alpha = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    value = joint_identity(native, post, alpha)
    assert torch.equal(value, native)
    value.square().mean().backward()
    assert alpha.grad is not None and torch.isfinite(alpha.grad)
    assert int(torch.count_nonzero(alpha.grad)) == 1


def test_native_and_post_operator_shapes_and_difference() -> None:
    torch = pytest.importorskip("torch")

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.zero_carrier = False
            self.carrier_pre_pool = torch.nn.Sequential(torch.nn.Linear(1024, 32), torch.nn.ReLU())
            self.carrier_post_pool = torch.nn.Sequential(
                torch.nn.Linear(36, 32), torch.nn.ReLU(),
                torch.nn.Linear(32, 32), torch.nn.ReLU(),
                torch.nn.Linear(32, 700),
            )

    net = Toy()
    activity = torch.randn(1, 3, 1024, 176)
    carrier = torch.randn(1, 176, 4)
    native, post = native_and_post_identity(net, activity, carrier)
    assert native.shape == post.shape == (1, 176, 700)
    assert not torch.equal(native, post)

