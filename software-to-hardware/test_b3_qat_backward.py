#!/usr/bin/env python3
"""Backward coverage + forward bit-exact checks for QAT integer STE."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_EXP = _HERE.parent / "streaming_calibration_exp"
if _EXP.is_dir() and str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from b3_fake_quant import B3QATScales
from b3_hw_golden import B3Shapes, B3Weights
from b3_qat_encoder import wrap_early_pool_with_qat
from b3_quant_engine import ABLATION_PRESETS, build_quant_engine_bundle, forward_quant_engine, identity_metrics
from src.models.components.streaming_encoders import EarlyPoolEncoder


def _build_encoder(device: torch.device):
    base = EarlyPoolEncoder(trial_length=100, window_size=50, hidden_dim=64)
    scales = B3QATScales(0.037, 0.045, 0.032, 0.010, 0.007, 0.037)
    return wrap_early_pool_with_qat(base, scales, num_trials=33).to(device)


def _random_calib(device: torch.device) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(0)
    x = torch.randn(33, 100, 96, generator=g).abs() * 0.5
    return x.unsqueeze(0).to(device)


def _export_weights(enc) -> B3Weights:
    def _np(t):
        return t.detach().cpu().numpy().astype(np.float32)

    return B3Weights(
        pre_w=_np(enc.pre_linear.weight),
        pre_b=_np(enc.pre_linear.bias),
        post0_w=_np(enc.post_linears[0].weight),
        post0_b=_np(enc.post_linears[0].bias),
        post1_w=_np(enc.post_linears[1].weight),
        post1_b=_np(enc.post_linears[1].bias),
        post2_w=_np(enc.post_linears[2].weight),
        post2_b=_np(enc.post_linears[2].bias),
    )


def _align_e(enc, calib_t: torch.Tensor, calib_np: np.ndarray) -> dict:
    with torch.no_grad():
        e_fake = enc.forward_batch(calib_t).squeeze(0).cpu().numpy()
    shapes = B3Shapes(T=calib_np.shape[1], D=64, W=50, N=calib_np.shape[2], M=calib_np.shape[0])
    bundle = build_quant_engine_bundle(
        _export_weights(enc), shapes, enc.to_frozen_scales([]), ABLATION_PRESETS["w8_a8_e8"],
    )
    e_int = forward_quant_engine(calib_np, bundle)["E_dequant"]
    return identity_metrics(e_fake, e_int)


def _layer_weights(enc):
    return {
        "pre_linear": enc.pre_linear.weight,
        "post0": enc.post_linears[0].weight,
        "post1": enc.post_linears[1].weight,
        "post2": enc.post_linears[2].weight,
    }


def test_backward_coverage_and_exact_forward() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = _build_encoder(device)
    calib_t = _random_calib(device)
    calib_np = calib_t.squeeze(0).detach().cpu().numpy()

    align0 = _align_e(enc, calib_t, calib_np)
    assert align0["max_abs"] == 0.0, align0

    enc.train()
    e = enc.forward_batch(calib_t)
    loss = e.pow(2).mean()
    loss.backward()

    stats = {}
    for key, w in _layer_weights(enc).items():
        g = w.grad
        assert g is not None, f"no grad for {key}"
        stats[key] = {
            "finite_frac": float(torch.isfinite(g).float().mean()),
            "nonzero_frac": float((g.abs() > 0).float().mean()),
        }
        assert stats[key]["finite_frac"] == 1.0
        assert stats[key]["nonzero_frac"] > 0.05, stats[key]

    before = {k: v.detach().clone() for k, v in _layer_weights(enc).items()}
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    opt.step()

    deltas = {k: float((w - before[k]).abs().max()) for k, w in _layer_weights(enc).items()}
    for key, d in deltas.items():
        assert d > 0.0, f"{key} unchanged after step (delta={d})"

    enc.eval()
    align1 = _align_e(enc, calib_t, calib_np)
    assert align1["max_abs"] == 0.0, align1

    print("grad stats:", stats)
    print("weight deltas:", deltas)


if __name__ == "__main__":
    test_backward_coverage_and_exact_forward()
    print("PASS")
