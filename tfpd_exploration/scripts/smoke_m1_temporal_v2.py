#!/usr/bin/env python3
"""Real-data 100-update smoke. Does not write EvalAI slot files or start 12ep."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from tfpd_exploration.src.m1_temporal_decoder_quick_product_v1.config import (
    GRAD_CLIP,
    LR,
    N_UNITS,
    SEED,
    WEIGHT_DECAY,
)
from tfpd_exploration.src.m1_temporal_decoder_quick_product_v1.ema import DecoderEMA
from tfpd_exploration.src.m1_temporal_v2 import plan as v2_plan
from tfpd_exploration.src.m1_temporal_v2.bank import load_source_bank
from tfpd_exploration.src.m1_temporal_v2.data import build_source_only_datamodule, materialize_source_banks
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import (
    M1TemporalFlatDecoder,
    M1TemporalRouteDecoder,
    adamw_param_groups,
    prove_zero_gate_equals_flat,
    shared_parameter_max_abs_diff,
    whole_unit_dropout,
)

MICROBATCH = 8
SMOKE_STEPS = 100
DEST = v2_plan.RESULT_ROOT / "smoke_100"


def _session_name(session) -> str:
    text = session.decode("ascii") if isinstance(session, bytes) else str(session)
    return text if text.startswith("ses-") else str(session)


def _finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} is not finite")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    DEST.mkdir(parents=True, exist_ok=True)
    sealed = load_source_bank()
    data = build_source_only_datamodule(sealed)
    if getattr(data, "target_path", None) is not None or data.val_heldin_dataset is not None:
        raise RuntimeError("smoke opened outer query")
    banks = materialize_source_banks()
    for name, bank in banks.items():
        sealed_t = torch.as_tensor(sealed["normalized"][name], dtype=torch.float32)
        if not torch.equal(bank.T.cpu(), sealed_t):
            raise RuntimeError(f"{name} materialized T != sealed bank")
    flat = M1TemporalFlatDecoder(seed=SEED)
    route = M1TemporalRouteDecoder(seed=SEED, flat_template=flat)
    if shared_parameter_max_abs_diff(flat, route) != 0.0:
        raise RuntimeError("shared init drifted")
    first = next(iter(banks.values()))
    prove_zero_gate_equals_flat(flat, route, torch.randn(2, 32, N_UNITS), first)

    gpu_banks = {
        name: type(bank)(E0=bank.E0.to(device), T=bank.T.to(device), unit_mask=bank.unit_mask.to(device))
        for name, bank in banks.items()
    }
    loader = data.train_dataloader()
    model = M1TemporalFlatDecoder(seed=SEED).to(device)
    model.train()
    opt = torch.optim.AdamW(adamw_param_groups(model, WEIGHT_DECAY), lr=LR)
    ema = DecoderEMA(model)
    last_loss = None
    last_grad = None
    start = time.time()
    step = 0
    while step < SMOKE_STEPS:
        for batch_id, batch in enumerate(loader):
            neural, target, _calib, session, _carrier = batch[:5]
            name = _session_name(session[0] if not isinstance(session, str) else session)
            neural = neural.to(device, dtype=torch.float32)
            if neural.dim() == 4:
                neural = neural.squeeze(-1) if neural.size(-1) == 1 else neural.mean(-1)
            if neural.size(-1) != N_UNITS and neural.size(1) == N_UNITS:
                neural = neural.transpose(1, 2)
            target = target.to(device, dtype=torch.float32)
            if target.dim() == 3:
                target = target[:, -1, :]
            bank = gpu_banks[name]
            n = int(neural.size(0))
            step += 1
            keep = whole_unit_dropout(torch.ones(N_UNITS, dtype=torch.bool), p=0.10)
            keep = keep.reshape(-1).unsqueeze(0).expand(n, -1).contiguous().to(device)
            opt.zero_grad(set_to_none=True)
            running = 0.0
            for offset in range(0, n, MICROBATCH):
                sl = slice(offset, min(offset + MICROBATCH, n))
                chunk = int(neural[sl].size(0))
                pred = model.forward_last(neural[sl], bank, dropout_keep=keep[sl])
                loss = F.mse_loss(pred, target[sl]) * (chunk / float(n))
                _finite("loss", loss)
                loss.backward()
                running += float(loss.detach().item())
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if not (grad_norm == grad_norm):
                raise RuntimeError("grad norm is NaN")
            opt.step()
            ema.update_after_step(model)
            last_loss = running
            last_grad = grad_norm
            if step >= SMOKE_STEPS:
                break
    elapsed = time.time() - start
    ckpt = DEST / "smoke_100.pt"
    torch.save(
        {
            "schema": "m1_temporal_v2_smoke_ckpt",
            "arm": "flat",
            "global_step": step,
            "raw_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "ema": ema.checkpoint_state(),
            "optimizer": opt.state_dict(),
            "train_mse": last_loss,
        },
        ckpt,
    )
    restored = M1TemporalFlatDecoder(seed=SEED).to(device)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    restored.load_state_dict(payload["raw_state_dict"])
    restored.to(device)
    restored.train()
    opt2 = torch.optim.AdamW(adamw_param_groups(restored, WEIGHT_DECAY), lr=LR)
    opt2.load_state_dict(payload["optimizer"])
    neural, target, _calib, session, _carrier = next(iter(loader))[:5]
    name = _session_name(session[0] if not isinstance(session, str) else session)
    neural = neural.to(device, dtype=torch.float32)
    if neural.dim() == 4:
        neural = neural.squeeze(-1) if neural.size(-1) == 1 else neural.mean(-1)
    if neural.size(-1) != N_UNITS and neural.size(1) == N_UNITS:
        neural = neural.transpose(1, 2)
    target = target.to(device, dtype=torch.float32)
    if target.dim() == 3:
        target = target[:, -1, :]
    pred = restored.forward_last(neural[:MICROBATCH], gpu_banks[name])
    loss = F.mse_loss(pred, target[:MICROBATCH])
    _finite("resume_loss", loss)
    loss.backward()
    resume_grad = float(torch.nn.utils.clip_grad_norm_(restored.parameters(), GRAD_CLIP))
    opt2.step()
    receipt = {
        "status": "SMOKE_OK",
        "revision": v2_plan.REVISION,
        "carrier_name": v2_plan.CARRIER_NAME,
        "bank_npz_sha256": sealed["npz_sha256"],
        "steps": step,
        "last_loss": last_loss,
        "last_grad_norm": last_grad,
        "resume_loss": float(loss.detach().item()),
        "resume_grad_norm": resume_grad,
        "elapsed_s": elapsed,
        "seconds_per_update": elapsed / max(step, 1),
        "microbatch": MICROBATCH,
        "target_path_resolved": False,
        "val_heldin_dataset": False,
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    (DEST / "smoke_100.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
