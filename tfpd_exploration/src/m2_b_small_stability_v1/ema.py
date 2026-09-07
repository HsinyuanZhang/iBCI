"""Decoder-only EMA shadow. Never trains; never aliases RAW storage."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import torch
from torch import nn

from .config import EMA_DECAY

T = TypeVar("T")


def _decoder_named(module: nn.Module) -> dict[str, nn.Parameter]:
    if hasattr(module, "trainable_parameters"):
        return dict(module.trainable_parameters())
    return {name: param for name, param in module.named_parameters() if param.requires_grad}


class DecoderEMA:
    """First successful update copies RAW; later updates use decay * EMA + (1-d) * RAW."""

    def __init__(self, module: nn.Module, decay: float = EMA_DECAY) -> None:
        self.decay = float(decay)
        self.n_updates = 0
        self.shadow: dict[str, torch.Tensor] = {
            name: torch.empty_like(param.detach(), dtype=torch.float32)
            for name, param in _decoder_named(module).items()
        }
        for tensor in self.shadow.values():
            tensor.requires_grad_(False)

    def update_after_step(self, module: nn.Module) -> None:
        named = _decoder_named(module)
        with torch.no_grad():
            if self.n_updates == 0:
                for name, param in named.items():
                    self.shadow[name] = param.detach().to(dtype=torch.float32).clone()
                    self.shadow[name].requires_grad_(False)
            else:
                decay = self.decay
                one_minus = 1.0 - decay
                for name, param in named.items():
                    raw = param.detach().to(dtype=torch.float32)
                    self.shadow[name].mul_(decay).add_(raw, alpha=one_minus)
        self.n_updates += 1

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "n_updates": self.n_updates,
            "shadow": {name: tensor.detach().clone() for name, tensor in self.shadow.items()},
        }

    def load_checkpoint_state(self, payload: dict[str, Any]) -> None:
        self.decay = float(payload["decay"])
        self.n_updates = int(payload["n_updates"])
        self.shadow = {name: tensor.detach().clone() for name, tensor in payload["shadow"].items()}
        for tensor in self.shadow.values():
            tensor.requires_grad_(False)

    def score_with_ema(self, module: nn.Module, fn: Callable[[nn.Module], T]) -> T:
        """Temporarily swap EMA into a copy of RAW, then restore RAW. Does not alias."""
        named = _decoder_named(module)
        raw_backup = {name: param.detach().clone() for name, param in named.items()}
        was_training = bool(module.training)
        try:
            with torch.no_grad():
                for name, param in named.items():
                    param.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))
            module.eval()
            return fn(module)
        finally:
            with torch.no_grad():
                for name, param in named.items():
                    param.copy_(raw_backup[name].to(device=param.device, dtype=param.dtype))
            module.train(was_training)
