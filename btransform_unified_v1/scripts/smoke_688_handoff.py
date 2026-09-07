"""CPU synthetic wiring test, not an NWB/protocol audit or a training runner."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "btransform_unified_v1/src"), str(ROOT)]
from pack_688_handoff import load_state
from btransform_unified_v1 import BTransformerUnifiedDecoderIdentity, TaskBank
from btransform_unified_v1.bank import array_sha256
from streaming_calibration_exp.src.models.components.streaming_encoders import (
    BatchReferenceEncoder, _build_affine_stack,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    encoder = BatchReferenceEncoder(_build_affine_stack(100, 512, 3, 512),
                                    _build_affine_stack(512, 512, 3, 50), 50).eval()
    encoder.load_state_dict(load_state(args.bundle / "b0_s42_e11_encoder.npz"), strict=True)
    encoder.requires_grad_(False)
    with torch.no_grad():
        e = encoder.forward_batch(torch.rand(1, 30, 100, 7))[0].numpy()
    assert e.shape == (7, 50) and np.isfinite(e).all()
    e0, carrier = np.zeros((100, 50), np.float32), np.zeros((100, 4), np.float32)
    e0[:7] = e
    carrier[:7] = np.random.default_rng(42).normal(size=(7, 4))
    mask = np.arange(100) < 7
    bank = TaskBank("synthetic-only", e0, carrier, mask,
                    np.zeros((0, 50, 100), np.float32), np.zeros((0, 2), np.float32),
                    np.zeros(0, np.int64),
                    dict(shape=list(e0.shape), trial_count=30, estimator="synthetic-smoke",
                         array_sha256=array_sha256(e0), budget=30))
    geometry = dict(task="dandi688", window=50, prefix=0, units=100,
                    e0_dim=50, carrier_dim=4, out_dim=2, target_scale=5)
    model = BTransformerUnifiedDecoderIdentity(geometry, seed=42, identity_mode="proj_add",
                                               proj_dim=16, temporal_layers=4).eval()
    assert model.init_meta["fallback"] is False, model.init_meta
    x = torch.rand(2, 50, 100)
    x[:, :, 7:] = 0
    with torch.no_grad():
        pred = model(x, bank)
        noisy_pad = x.clone()
        noisy_pad[:, :, 7:] = 100
        torch.testing.assert_close(pred, model(noisy_pad, bank), atol=1e-6, rtol=1e-6)
        assert pred.shape == (2, 2) and torch.isfinite(pred).all()
    zero_e = np.zeros_like(e0)
    no_e0 = replace(bank, E0=zero_e,
                    calibration_meta=dict(bank.calibration_meta, array_sha256=array_sha256(zero_e)))
    model.zero_grad(set_to_none=True)
    model(x, no_e0).square().mean().backward()
    assert torch.count_nonzero(model.frontend.e0_proj.weight.grad) == 0
    model.zero_grad(set_to_none=True)
    model(x, replace(bank, carrier=np.zeros_like(carrier))).square().mean().backward()
    assert torch.count_nonzero(model.frontend.token_mlp[0].weight.grad[:, -4:]) == 0
    print(json.dumps(dict(status="PASS", encoder_shape=list(e.shape), output_shape=list(pred.shape),
                          padded_input_invariance="PASS", no_e0_data_gradient="ZERO",
                          no_c_data_gradient="ZERO", init_meta=model.init_meta,
                          nwb_opened=False, gpu_used=False), indent=2))


if __name__ == "__main__":
    main()
