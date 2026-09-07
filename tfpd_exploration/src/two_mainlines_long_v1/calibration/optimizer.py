"""AdamW groups, explicit RNG seeding, and warmup-then-constant LR."""
from __future__ import annotations

import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from . import constants as C


def seed_all(seed: int, *, device: torch.device | None = None) -> dict[str, object]:
    """Actually set Python / NumPy / Torch / CUDA RNGs. Writing a spec is not enough."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_seeded = False
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cuda_seeded = True
    elif torch.cuda.is_available() and str(torch.cuda.device_count()) != "0":
        torch.cuda.manual_seed_all(seed)
        cuda_seeded = True
    return {
        "seed": seed,
        "python": True,
        "numpy": True,
        "torch": True,
        "cuda": cuda_seeded,
    }


def is_no_weight_decay(name: str, parameter: nn.Parameter) -> bool:
    lowered = name.lower()
    if parameter.ndim <= 1:
        return True
    if name.endswith(".bias") or lowered.endswith("bias"):
        return True
    if "norm" in lowered or "ln" in lowered.split(".") or "bn" in lowered.split("."):
        return True
    return False


def named_trainable(modules: Iterable[tuple[str, nn.Module | nn.Parameter]]) -> list[tuple[str, nn.Parameter]]:
    named: list[tuple[str, nn.Parameter]] = []
    for prefix, module in modules:
        if isinstance(module, nn.Parameter):
            if module.requires_grad:
                named.append((prefix, module))
            continue
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                named.append((f"{prefix}.{name}" if prefix else name, parameter))
    return named


def adamw_param_groups(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    *,
    weight_decay: float = C.WEIGHT_DECAY,
) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if is_no_weight_decay(name, parameter):
            no_decay.append(parameter)
            no_decay_names.append(name)
        else:
            decay.append(parameter)
            decay_names.append(name)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay), "names": decay_names})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "names": no_decay_names})
    if not groups:
        raise RuntimeError("no trainable AdamW parameters")
    return groups


def build_adamw(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    *,
    lr: float = C.LR,
    weight_decay: float = C.WEIGHT_DECAY,
) -> torch.optim.AdamW:
    groups = adamw_param_groups(named_parameters, weight_decay=weight_decay)
    clean = [{"params": group["params"], "weight_decay": group["weight_decay"]} for group in groups]
    return torch.optim.AdamW(clean, lr=lr)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def warmup_lr(step: int, steps_per_warmup: int, base_lr: float = C.LR) -> float:
    if steps_per_warmup <= 0:
        return float(base_lr)
    progress = min(int(step) + 1, int(steps_per_warmup))
    return float(base_lr) * float(progress) / float(steps_per_warmup)


def record_train_mode(student, basis, optimizer: torch.optim.AdamW) -> dict[str, object]:
    decoder = student.decoder
    groups = []
    for group in optimizer.param_groups:
        groups.append(
            {
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "lr": float(group.get("lr", 0.0)),
                "n_params": len(list(group["params"])),
            }
        )
    return {
        "student_training": bool(student.training),
        "decoder_training": bool(decoder.training),
        "id_encoder_training": bool(student.id_encoder.training),
        "dynamic_dropout": bool(getattr(decoder, "dynamic_dropout", False)),
        "dropout_rate": float(getattr(decoder, "dropout_rate", 0.0)),
        "tf_drop_rate": float(getattr(decoder, "tf_drop_rate", 0.0)),
        "basis_requires_grad": bool(basis.raw_dictionary.requires_grad),
        "carrier_projection_requires_grad": bool(student.carrier_projection_weight.requires_grad),
        "adamw_groups": groups,
    }
