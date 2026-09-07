from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
H1_SRC = ROOT / "tfpd_exploration" / "h1_series_20260830" / "src"
if str(H1_SRC) not in sys.path:
    sys.path.insert(0, str(H1_SRC))

from h1_postpool_profile_gate_v1.core import profile_identity
from h1_postpool_profile_gate_v1.plan import DATE_ORDER, PROFILE_LENGTH, decide_oof


def test_zero_profile_is_exact_static_object_and_nonzero_profile_has_gradient() -> None:
    torch = pytest.importorskip("torch")
    static = torch.linspace(-1.0, 1.0, 3 * PROFILE_LENGTH).reshape(3, PROFILE_LENGTH)
    post = static + 0.25
    zero = torch.zeros(PROFILE_LENGTH)
    assert profile_identity(static, post, zero) is static
    trainable_zero = torch.zeros(PROFILE_LENGTH, requires_grad=True)
    trainable_output = profile_identity(static, post, trainable_zero)
    assert torch.equal(trainable_output, static)
    trainable_output.square().mean().backward()
    assert trainable_zero.grad is not None
    assert torch.count_nonzero(trainable_zero.grad).item() > 0
    profile = torch.full((PROFILE_LENGTH,), 0.1, requires_grad=True)
    output = profile_identity(static, post, profile)
    output.square().mean().backward()
    assert profile.grad is not None
    assert torch.isfinite(profile.grad).all()
    assert torch.count_nonzero(profile.grad).item() > 0


def test_profile_is_shared_across_units_and_bounded_by_tanh() -> None:
    torch = pytest.importorskip("torch")
    static = torch.zeros((2, PROFILE_LENGTH))
    post = torch.ones((2, PROFILE_LENGTH))
    profile = torch.full((PROFILE_LENGTH,), 100.0)
    output = profile_identity(static, post, profile)
    assert torch.equal(output[0], output[1])
    assert float(output.max()) <= 1.0


def test_profile_oof_gate_remains_strict() -> None:
    rows = [
        {"outer_date": date, "delta_vs_static": delta}
        for date, delta in zip(DATE_ORDER, (0.011, 0.008, 0.006, 0.002, -0.001), strict=True)
    ]
    assert decide_oof(rows)["pass"] is True
    rows[1]["delta_vs_static"] = -0.012
    assert decide_oof(rows)["pass"] is False
