"""LR, dropout domain, RAW step, checkpoint, disposable smoke/profile.

Formal 24-epoch launch lives in ``launch.py`` and is refused unless
``M2_SMALL_TRAIN=1``. This module is CPU-testable.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from tfpd_exploration.src.m2_dual_track_v1.contracts import SessionBank, make_stub_bank
from tfpd_exploration.src.m2_dual_track_v1.decoders import whole_unit_dropout
from tfpd_exploration.src.m2_dual_track_v1.training import adamw_param_groups as old_adamw_param_groups

from .config import (
    ADAM_BETAS,
    ADAM_EPS,
    CELL_S0,
    CELL_S1,
    SCHEDULE_COSINE,
    SCHEDULE_LEGACY,
    cell_lr_bounds,
    EMA_DECAY,
    GRAD_CLIP,
    LR_MAX,
    LR_MIN,
    SMALL,
    TOTAL_UPDATES,
    UPDATES_PER_EPOCH,
    WEIGHT_DECAY,
    require,
)
from .decoder import SmallTransformerDecoder
from .ema import DecoderEMA


def lr_at_update(
    cell: str,
    t: int,
    *,
    w: int = UPDATES_PER_EPOCH,
    total: int = TOTAL_UPDATES,
    lr_max: float | None = None,
    lr_min: float | None = None,
) -> float:
    """1-based optimizer update ``t``. Set this LR before ``optimizer.step``."""
    require(t >= 1, "t is 1-based")
    if lr_max is None or lr_min is None:
        lr_max, lr_min = cell_lr_bounds(cell)
    if 1 <= t <= w:
        return float(lr_max) * float(t) / float(w)
    epoch = (int(t) - 1) // int(w) + 1
    if cell in SCHEDULE_LEGACY:
        if 2 <= epoch <= 12:
            return float(lr_max)
        require(13 <= epoch <= 24, f"legacy epoch out of range: {epoch}")
        return float(lr_max) * (0.1 + 0.45 * (1.0 + math.cos(math.pi * (epoch - 13) / 11.0)))
    if cell in SCHEDULE_COSINE:
        p = float(t - w) / float(total - w)
        return float(lr_min) + 0.5 * (float(lr_max) - float(lr_min)) * (1.0 + math.cos(math.pi * p))
    raise ValueError(f"unknown cell {cell}")


def apply_cell_lr(optimizer: torch.optim.Optimizer, cell: str, t: int) -> float:
    lr = lr_at_update(cell, t)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def make_optimizer(named_parameters: Any) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        old_adamw_param_groups(named_parameters, weight_decay=WEIGHT_DECAY),
        lr=LR_MAX,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
    )


def unit_dropout_seed(seed: int, epoch: int, batch_id: int) -> int:
    payload = f"m2_small_unit_dropout|{int(seed)}|{int(epoch)}|{int(batch_id)}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**63)


def unit_dropout_mask(
    unit_mask: torch.Tensor,
    *,
    seed: int,
    epoch: int,
    batch_id: int,
    p: float = SMALL.unit_dropout,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(unit_dropout_seed(seed, epoch, batch_id))
    return whole_unit_dropout(unit_mask, p=p, generator=generator)


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(payload: Mapping[str, Any] | None) -> None:
    if not payload:
        return
    if payload.get("python") is not None:
        random.setstate(payload["python"])
    if payload.get("numpy") is not None:
        np.random.set_state(payload["numpy"])
    if payload.get("torch") is not None:
        torch.set_rng_state(payload["torch"])
    if payload.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda"])


@dataclass
class TrainState:
    cell: str
    global_step: int = 0
    epoch: int = 1
    batch_id: int = 0
    seed: int = 42
    manifest_digest: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def run_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    bank: SessionBank,
    *,
    cell: str,
    t: int,
    seed: int,
    epoch: int,
    batch_id: int,
    ema: DecoderEMA | None = None,
    state: TrainState | None = None,
) -> dict[str, Any]:
    apply_cell_lr(optimizer, cell, t)
    keep = unit_dropout_mask(bank.unit_mask, seed=seed, epoch=epoch, batch_id=batch_id)
    if keep.dim() == 1:
        keep = keep.unsqueeze(0).expand(x.size(0), -1)
    model.train()
    pred = model.forward_last(x, bank, keep, dropout_keep=keep)
    loss = nn.functional.mse_loss(pred, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(list(model.trainable_parameters().values()), GRAD_CLIP)
    optimizer.step()
    if ema is not None:
        ema.update_after_step(model)
    if state is not None:
        state.global_step = int(t)
        state.epoch = int(epoch)
        state.batch_id = int(batch_id)
    return {"loss": float(loss.detach().cpu()), "lr": float(optimizer.param_groups[0]["lr"])}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: DecoderEMA,
    state: TrainState,
) -> dict[str, Any]:
    return {
        "schema": "m2_b_small_stability_v1_ckpt",
        "cell": state.cell,
        "global_step": state.global_step,
        "epoch": state.epoch,
        "batch_id": state.batch_id,
        "seed": state.seed,
        "manifest_digest": state.manifest_digest,
        "raw_state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "ema": ema.checkpoint_state(),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng": capture_rng(),
        "lr": float(optimizer.param_groups[0]["lr"]),
        "ema_decay": EMA_DECAY,
    }


def load_checkpoint(
    payload: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: DecoderEMA,
    state: TrainState,
) -> None:
    model.load_state_dict(payload["raw_state_dict"])
    optimizer.load_state_dict(payload["optimizer"])
    ema.load_checkpoint_state(payload["ema"])
    state.cell = str(payload["cell"])
    state.global_step = int(payload["global_step"])
    state.epoch = int(payload["epoch"])
    state.batch_id = int(payload["batch_id"])
    state.seed = int(payload["seed"])
    state.manifest_digest = str(payload["manifest_digest"])
    restore_rng(payload.get("rng"))


def disposable_smoke_steps(*, n_steps: int = 2) -> dict[str, Any]:
    """Throwaway FP32 steps. Restores the caller RNG so formal state is unchanged."""
    formal = capture_rng()
    try:
        torch.manual_seed(0)
        np.random.seed(0)
        model = SmallTransformerDecoder(seed=0)
        opt = make_optimizer(model.trainable_parameters().items())
        bank = make_stub_bank(seed=0)
        x = torch.zeros(1, SMALL.window, SMALL.channels)
        y = torch.zeros(1, 2)
        for step in range(1, int(n_steps) + 1):
            run_optimizer_step(
                model,
                opt,
                x,
                y,
                bank,
                cell=CELL_S1,
                t=step,
                seed=0,
                epoch=1,
                batch_id=step,
                ema=None,
            )
    finally:
        restore_rng(formal)
    after = capture_rng()
    return {
        "n_steps": int(n_steps),
        "advanced_formal_rng": not torch.equal(after["torch"], formal["torch"]),
        "disposable": True,
        "precision": "fp32",
    }


def disposable_profile_hook(*, n_steps: int = 100, device: str = "cpu") -> dict[str, Any]:
    """Implemented 100-step disposable profile. Stage0 must not launch this on GPU."""
    require(str(device) == "cpu", "profile hook GPU launch is forbidden in Stage0")
    bank = make_stub_bank(seed=1)
    x = torch.randn(2, SMALL.window, SMALL.channels)
    y = torch.randn(2, 2)
    model = SmallTransformerDecoder(seed=1)
    opt = make_optimizer(model.trainable_parameters().items())
    ema = DecoderEMA(model, decay=EMA_DECAY)
    for step in range(1, int(n_steps) + 1):
        run_optimizer_step(
            model,
            opt,
            x,
            y,
            bank,
            cell=CELL_S1,
            t=step,
            seed=1,
            epoch=1,
            batch_id=step,
            ema=ema,
        )
    return {
        "n_steps": int(n_steps),
        "launched_on_gpu": False,
        "disposable": True,
        "device": "cpu",
        "implemented": True,
        "note": "100-step GPU profile is registered but not launched by Stage0",
    }
