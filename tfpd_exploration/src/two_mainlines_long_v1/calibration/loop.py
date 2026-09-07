"""One consumer update. Explicit train(); P-FIX uses cache; P-CA recomputes D."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from . import constants as C
from . import optimizer as optmod
from .factory import FreshPArm
from .pfix_cache import carrier_for_step


def session_name(session) -> str:
    first = session[0] if not isinstance(session, (str, bytes, bytearray)) else session
    if isinstance(first, (bytes, bytearray)):
        return first.decode("ascii")
    return str(first)


def one_update(
    arm: FreshPArm,
    batch,
    *,
    cache: dict[str, dict[str, torch.Tensor]] | None,
    steps_per_warmup: int,
) -> dict[str, Any]:
    mode = arm.enter_train()
    if not mode["student_training"]:
        raise RuntimeError("student is not in train() during update")
    neural, target, calib, session, _unused = batch
    name = session_name(session)
    lr = optmod.warmup_lr(arm.global_step, steps_per_warmup, C.LR)
    optmod.set_lr(arm.optimizer, lr)
    carrier = carrier_for_step(arm, name, cache)
    if carrier.ndim == 2:
        carrier_b = carrier.unsqueeze(0)
    else:
        carrier_b = carrier
    if carrier_b.shape[0] == 1 and neural.shape[0] > 1:
        carrier_b = carrier_b.expand(neural.shape[0], -1, -1)
    neural_t = neural.to(arm.device)
    target_t = target.to(arm.device)
    calib_t = calib.to(arm.device)
    pred, _identity = arm.student(neural_t, calib_trials=calib_t, carrier=carrier_b)
    pred_last = pred[:, -1, :] if pred.ndim == 3 else pred
    target_last = target_t[:, -1, :] if target_t.ndim == 3 else target_t
    loss = F.mse_loss(pred_last, target_last)
    arm.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    params = [parameter for group in arm.optimizer.param_groups for parameter in group["params"]]
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, C.CLIP))
    arm.optimizer.step()
    arm.global_step += 1
    arm.sampler_index += 1
    return {
        "loss": float(loss.detach().cpu()),
        "lr": lr,
        "grad_norm": grad_norm,
        "session": name,
        "train_mode": mode,
        "global_step": arm.global_step,
    }
