#!/usr/bin/env python3
"""B3S/T4 side-concat coverage, gradient, and integer exactness tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_EXP = _HERE.parent / "streaming_calibration_exp"
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from b3_fake_quant import B3QATScales
from b3_hw_golden import B3Shapes, B3Weights, forward_b3_layered
from b3_qat_encoder import wrap_early_pool_with_qat
from b3_quant_engine import (
    ABLATION_PRESETS,
    build_quant_engine_bundle,
    forward_quant_engine,
    identity_metrics,
)
from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder


def _np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32)


def _weights(encoder) -> B3Weights:
    return B3Weights(
        pre_w=_np(encoder.pre_linear.weight),
        pre_b=_np(encoder.pre_linear.bias),
        post0_w=_np(encoder.post_linears[0].weight),
        post0_b=_np(encoder.post_linears[0].bias),
        post1_w=_np(encoder.post_linears[1].weight),
        post1_b=_np(encoder.post_linears[1].bias),
        post2_w=_np(encoder.post_linears[2].weight),
        post2_b=_np(encoder.post_linears[2].bias),
    )


def test_t4_shared_scale_concat_is_bit_exact_and_trainable() -> None:
    torch.manual_seed(7)
    base = SideFeatureEarlyPoolEncoder(
        trial_length=8,
        window_size=4,
        hidden_dim=4,
        side_dim=4,
        num_post_layers=3,
    )
    scales = B3QATScales(
        input=0.035,
        pre_out=0.025,
        mean=0.02,
        post0_out=0.015,
        post1_out=0.012,
        E=0.02,
    )
    qat = wrap_early_pool_with_qat(base, scales, num_trials=3)
    calib = torch.rand(1, 3, 8, 5)
    side = torch.randn(1, 5, 4) * 0.25

    with torch.no_grad():
        stages_t = qat.forward_integer_with_stages(calib, side_features=side)
    assert stages_t["post0_input_q"].shape == (1, 5, 8)
    # The T4 tail is explicitly A8 at the same scale as pooled activity.
    expected_side_q = torch.clamp(
        torch.round(side / qat.shared_scales.s_mean.value()), -128, 127
    )
    assert torch.equal(stages_t["post0_input_q"][..., -4:], expected_side_q)

    weights = _weights(qat)
    shapes = B3Shapes(T=8, D=4, W=4, N=5, M=3)
    bundle = build_quant_engine_bundle(
        weights,
        shapes,
        qat.to_frozen_scales(["train-only"]),
        ABLATION_PRESETS["w8_a8_e8"],
    )
    stages_np = forward_quant_engine(
        _np(calib.squeeze(0)),
        bundle,
        side_features=_np(side.squeeze(0)),
    )
    metrics = identity_metrics(
        _np(stages_t["E_dequant"].squeeze(0)), stages_np["E_dequant"]
    )
    assert metrics["max_abs"] == 0.0, metrics
    assert stages_np["post0_input"].shape == (5, 8)
    assert stages_np["diagnostics"]["side"]["shared_post0_input_scale"] == (
        bundle.scales.mean_scale_i8
    )

    # FP golden must also exercise the real 8->4 post0 matrix, not bypass T4.
    fp_np = forward_b3_layered(
        _np(calib.squeeze(0)), weights, side_features=_np(side.squeeze(0))
    )
    with torch.no_grad():
        fp_t = qat.forward_fp32(calib, side_features=side)
    assert np.allclose(fp_np["E"], _np(fp_t.squeeze(0)), atol=1e-6, rtol=1e-6)

    qat.train()
    loss = qat.forward_batch(calib, side_features=side).pow(2).mean()
    loss.backward()
    gradients = {
        "pre": qat.pre_linear.weight.grad,
        "post0_activity": qat.post_linears[0].weight.grad[:, :4],
        "post0_t4": qat.post_linears[0].weight.grad[:, 4:],
        "post1": qat.post_linears[1].weight.grad,
        "post2": qat.post_linears[2].weight.grad,
    }
    for name, gradient in gradients.items():
        assert gradient is not None, f"missing gradient for {name}"
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient) > 0, name


def test_t4_side_features_are_required() -> None:
    base = SideFeatureEarlyPoolEncoder(
        trial_length=8,
        window_size=4,
        hidden_dim=4,
        side_dim=4,
        num_post_layers=3,
    )
    scales = B3QATScales(0.03, 0.03, 0.03, 0.03, 0.03, 0.03)
    qat = wrap_early_pool_with_qat(base, scales, num_trials=3)
    calib = torch.rand(1, 3, 8, 5)
    try:
        qat.forward_batch(calib)
    except ValueError as exc:
        assert "requires side_features" in str(exc)
    else:
        raise AssertionError("T4 QAT accepted a missing side-feature tensor")
