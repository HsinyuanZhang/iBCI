"""CPU-only training dry-run. Must not be used to launch GPU Stage 1."""
from __future__ import annotations

import os

import numpy as np
import torch

from .constants import N_CHANNELS, N_FREQ, N_MS_BINS, N_SPEC_FRAMES, VALID_END, VALID_START
from .metric import normalize_signal
from .model import B1SpintSFCJ, build_six_arms
from .sfc import zero9


def _metric_surrogate(pred_raw: torch.Tensor, tgt_raw: torch.Tensor) -> torch.Tensor:
    # pred/tgt [B,158,880]; mask valid frames; per-trial min-max MSE
    pred = pred_raw[:, :, VALID_START:VALID_END]
    tgt = tgt_raw[:, :, VALID_START:VALID_END]
    losses = []
    for b in range(pred.size(0)):
        p = pred[b]
        t = tgt[b]
        p_n = (p - p.min()) / (p.max() - p.min() + 1e-12)
        t_n = (t - t.min()) / (t.max() - t.min() + 1e-12)
        losses.append(torch.mean((p_n - t_n) ** 2))
    return torch.stack(losses).mean()


def dry_run(*, d_model: int = 32, n_heads: int = 4, steps: int = 2, seed: int = 0) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        # caller should export CUDA_VISIBLE_DEVICES=""; we still force CPU
        pass
    device = torch.device("cpu")
    torch.manual_seed(seed)
    model = B1SpintSFCJ(d_model=d_model, n_heads=n_heads, fusion="jr1", carrier_kind="zero").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.0)
    losses = []
    for step in range(steps):
        x = torch.rand(1, N_MS_BINS, N_CHANNELS, device=device)
        calib = torch.rand(1, 3, N_MS_BINS, N_CHANNELS, device=device)
        carrier = torch.zeros(1, N_CHANNELS, 9, device=device)
        tgt = torch.rand(1, N_FREQ, N_SPEC_FRAMES, device=device) + 1.0
        pred = model(x, calib, carrier, return_raw=True)
        loss = _metric_surrogate(pred, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if not np.isfinite(losses[-1]):
            raise RuntimeError(f"non-finite dry-run loss at step {step}")
    return {
        "device": "cpu",
        "d_model": d_model,
        "n_heads": n_heads,
        "steps": steps,
        "losses": losses,
        "finite": True,
        "six_arm_factory_ok": list(build_six_arms(d_model=d_model, n_heads=n_heads, seed=seed).keys()),
    }


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(dry_run())
