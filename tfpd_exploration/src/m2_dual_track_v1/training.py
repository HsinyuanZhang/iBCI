"""Shared trainer skeleton for A/B. Does not start 12-epoch training itself.

Epoch = one full pass over every eligible source window; the tail batch is
kept.  Effective batch 32, grouped by session.  Checkpoint selection is
highest source-minival equal-session mean over epochs 1-12; ties within
1e-10 take the earlier epoch.  Ext-4 is not scored every epoch.

Preflight 20+100 timing uses a throwaway model/optimizer and must not leak
into formal epoch 1.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Iterable, Iterator, Mapping

import torch
from torch import nn

from . import data, plan

NOT_READY = "NOT_READY"


def resolve_arm_builder(arm: str) -> Callable[..., Any]:
    if arm == "A-QMEM":
        try:
            from tfpd_exploration.src.m2_dual_track_v1 import calibration_memory as module
        except ImportError as exc:
            raise plan.DualTrackError(f"{NOT_READY}: A-QMEM module missing ({exc})") from exc
        for name in ("build_candidate", "build_a_candidate", "AQMemCandidate"):
            if hasattr(module, name):
                return getattr(module, name)
        raise plan.DualTrackError(f"{NOT_READY}: calibration_memory has no candidate builder")
    if arm in {"B-MAMBA", "B-TRANSFORMER"}:
        try:
            from tfpd_exploration.src.m2_dual_track_v1 import decoders as module
        except ImportError as exc:
            raise plan.DualTrackError(f"{NOT_READY}: B decoder module missing ({exc})") from exc
        attr = "build_mamba" if arm == "B-MAMBA" else "build_transformer"
        for name in (attr, "build_candidate", "BCandidate"):
            if hasattr(module, name):
                return getattr(module, name)
        raise plan.DualTrackError(f"{NOT_READY}: decoders has no builder for {arm}")
    raise plan.DualTrackError(f"unknown arm {arm}")


def is_no_weight_decay(name: str, parameter: nn.Parameter, extra_ids: set[int] | None = None) -> bool:
    if extra_ids and id(parameter) in extra_ids:
        return True
    lowered = name.lower()
    if parameter.ndim <= 1:
        return True
    if name.endswith(".bias") or lowered.endswith("bias"):
        return True
    if "norm" in lowered or "ln" in lowered.split(".") or "bn" in lowered.split("."):
        return True
    if "dynamics" in lowered:
        return True
    return False


def adamw_param_groups(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    *,
    weight_decay: float,
    extra_no_decay: Iterable[nn.Parameter] = (),
) -> list[dict[str, Any]]:
    extra_ids = {id(parameter) for parameter in extra_no_decay}
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if is_no_weight_decay(name, parameter, extra_ids):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    plan.require(bool(groups), "no trainable parameters")
    return groups


def build_optimizer(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    *,
    lr: float,
    weight_decay: float,
    extra_no_decay: Iterable[nn.Parameter] = (),
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        adamw_param_groups(named_parameters, weight_decay=weight_decay, extra_no_decay=extra_no_decay),
        lr=lr,
        betas=plan.ADAM_BETAS,
        eps=plan.ADAM_EPS,
    )


def warmup_factor(step_index: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step_index + 1) / float(warmup_steps))


def apply_warmup_lr(optimizer: torch.optim.Optimizer, base_lr: float, step_index: int, warmup_steps: int) -> float:
    factor = warmup_factor(step_index, warmup_steps)
    for group in optimizer.param_groups:
        group["lr"] = base_lr * factor
    return base_lr * factor


def cosine_tail_factor(
    epoch: int,
    *,
    start_epoch: int = plan.EPOCHS + 1,
    end_epoch: int = plan.EPOCHS_EXTENDED,
    min_factor: float = 0.1,
) -> float:
    """Epoch-13-to-24 cosine down to 0.1× base LR. Unused unless max_epochs>12."""
    if int(epoch) < int(start_epoch):
        return 1.0
    if int(epoch) >= int(end_epoch):
        return float(min_factor)
    progress = (int(epoch) - int(start_epoch)) / float(int(end_epoch) - int(start_epoch))
    return float(min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress)))


def apply_b_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    *,
    global_step: int,
    warmup_steps: int,
    epoch: int,
    max_epochs: int,
) -> float:
    if int(max_epochs) > plan.EPOCHS and int(epoch) > plan.EPOCHS:
        lr = float(base_lr) * cosine_tail_factor(int(epoch))
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr
    return apply_warmup_lr(optimizer, base_lr, global_step, warmup_steps)


def arm_hparams(arm: str) -> dict[str, Any]:
    if arm == "A-QMEM":
        return {
            "lr": plan.A_LR,
            "weight_decay": plan.A_WEIGHT_DECAY,
            "grad_clip": plan.A_GRAD_CLIP,
            "warmup": "updates_frac",
            "warmup_frac": plan.A_WARMUP_FRAC,
            "precision": plan.A_PRECISION,
            "dropout": 0.0,
            "unit_dropout": 0.0,
            "inherited_decoder_eval": True,
        }
    if arm in {"B-MAMBA", "B-TRANSFORMER"}:
        return {
            "lr": plan.B_LR,
            "weight_decay": plan.B_WEIGHT_DECAY,
            "grad_clip": plan.B_GRAD_CLIP,
            "warmup": "epochs",
            "warmup_epochs": plan.B_WARMUP_EPOCHS,
            "precision": plan.B_PRECISION,
            "unit_dropout": plan.B_UNIT_DROPOUT,
            "fp32_smoke_before_bf16": True,
        }
    raise plan.DualTrackError(f"unknown arm {arm}")


def select_checkpoint(epoch_scores: Mapping[int, float], *, max_epoch: int = plan.EPOCHS) -> int:
    """Highest source-minival equal-session mean; tie within 1e-10 → earlier epoch."""
    plan.require(bool(epoch_scores), "no epoch scores")
    best_epoch: int | None = None
    best = -math.inf
    for epoch in sorted(int(key) for key in epoch_scores):
        if epoch < 1 or epoch > int(max_epoch):
            continue
        score = float(epoch_scores[epoch])
        if best_epoch is None or score > best + plan.TIE_EPS:
            best_epoch = epoch
            best = score
    plan.require(best_epoch is not None, "no eligible epoch in 1-12")
    return int(best_epoch)


def epoch_batches(
    banks: Mapping[str, Any],
    *,
    batch_size: int = plan.EFFECTIVE_BATCH,
    device: str | torch.device = "cpu",
    session_order: Iterable[str] | None = None,
) -> Iterator[Any]:
    """Original sequential path: calendar session order, time order inside session.

    Kept for the frozen first B-MAMBA record and for evaluation. New B training
    must use ``epoch_batches_shuffled`` / a shared manifest instead.
    """
    order = list(session_order or banks.keys())
    for session in order:
        yield from data.iter_session_batches(
            banks[session],
            batch_size=batch_size,
            device=device,
            target_space=plan.TRAINING_TARGET_SPACE,
            drop_last=False,
        )


def epoch_batches_shuffled(
    banks: Mapping[str, Any],
    manifest: Mapping[str, Any],
    epoch: int,
    *,
    device: str | torch.device = "cpu",
    target_space: str = plan.TRAINING_TARGET_SPACE,
) -> Iterator[Any]:
    from . import sampler

    yield from sampler.iter_manifest_batches(
        banks, manifest, epoch, device=device, target_space=target_space
    )


def count_updates(banks: Mapping[str, Any], *, batch_size: int = plan.EFFECTIVE_BATCH) -> int:
    total = 0
    for bank in banks.values():
        n_win = int(len(bank.eligible_starts))
        total += (n_win + batch_size - 1) // batch_size
    return total


def isolated_preflight(
    build_throwaway: Callable[[], tuple[Any, torch.optim.Optimizer]],
    step_fn: Callable[[Any, torch.optim.Optimizer], None],
    *,
    warmup_steps: int = 20,
    timed_steps: int = 100,
) -> dict[str, Any]:
    """20+100 timing on a throwaway instance. Must not be reused for epoch 1."""
    model, optimizer = build_throwaway()
    for _ in range(int(warmup_steps)):
        step_fn(model, optimizer)
    for _ in range(int(timed_steps)):
        step_fn(model, optimizer)
    del model, optimizer
    return {
        "warmup_steps": int(warmup_steps),
        "timed_steps": int(timed_steps),
        "leaked_into_formal_epoch1": False,
        "formal_epoch1_must_rebuild": [
            "model",
            "common_frontend",
            "optimizer",
            "scheduler",
            "sampler_rng",
            "dropout_rng",
            "recurrent_state",
        ],
    }


def formal_epoch1_reset_checklist() -> tuple[str, ...]:
    return (
        "rebuild_model",
        "rebuild_common_frontend",
        "rebuild_optimizer",
        "rebuild_scheduler",
        "reset_sampler_rng",
        "reset_dropout_rng",
        "reset_recurrent_state",
        "reuse_frozen_bank_ok",
    )


def train_arm(
    arm: str,
    seed: int = plan.SEED_PRIMARY,
    device: str = "cpu",
    max_epochs: int = plan.EPOCHS,
) -> int:
    """Plug-in entry. Raises NOT_READY if A/B are missing. Does not auto-launch."""
    del seed, device, max_epochs
    resolve_arm_builder(arm)
    if os.environ.get("DUAL_TRACK_LAUNCH") != "1":
        raise plan.DualTrackError(
            f"{NOT_READY}: trainer skeleton is ready for {arm} (AdamW, warmup, "
            "tail-keeping session batches, source-minival selection, isolated preflight) "
            "but the shared pipeline worker does not start 12-epoch training. "
            "Coordinator launches after Stage 0 + GPU lease with DUAL_TRACK_LAUNCH=1."
        )
    raise plan.DualTrackError(
        f"{NOT_READY}: {arm} builder imported but the formal 12-epoch loop is "
        "coordinator-owned after Stage 0. Ext-4 is not scored every epoch."
    )
