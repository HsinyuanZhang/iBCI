"""Small, auditable operators for the late-pool branch-only screen."""
from __future__ import annotations

import hashlib
from typing import Any

from .plan import DISTILL_FLOOR, SEED, SUPPORT, TRIAL_LENGTH, UNITS, WINDOW


class H1SupportResampledError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise H1SupportResampledError(message)


def native_identity(net: Any, activity: Any, carrier: Any) -> Any:
    import torch

    require(isinstance(activity, torch.Tensor) and activity.ndim == 4, "activity must be a tensor [B,M,T,N]")
    require(tuple(activity.shape[1:]) == (SUPPORT, TRIAL_LENGTH, UNITS), "activity geometry drift")
    require(isinstance(carrier, torch.Tensor) and tuple(carrier.shape) == (activity.shape[0], UNITS, 4), "carrier geometry drift")
    encoded = net.carrier_pre_pool(activity.permute(0, 1, 3, 2))
    effective = torch.zeros_like(carrier) if net.zero_carrier else carrier.to(encoded)
    result = net.carrier_post_pool(torch.cat((encoded.mean(dim=1), effective), dim=-1))
    require(tuple(result.shape) == (activity.shape[0], UNITS, WINDOW), "native identity geometry drift")
    require(bool(torch.isfinite(result).all()), "native identity is nonfinite")
    return result


def late_identity(net: Any, activity: Any, carrier: Any) -> Any:
    import torch

    require(isinstance(activity, torch.Tensor) and activity.ndim == 4, "activity must be a tensor [B,M,T,N]")
    require(tuple(activity.shape[1:]) == (SUPPORT, TRIAL_LENGTH, UNITS), "activity geometry drift")
    require(isinstance(carrier, torch.Tensor) and tuple(carrier.shape) == (activity.shape[0], UNITS, 4), "carrier geometry drift")
    encoded = net.carrier_pre_pool(activity.permute(0, 1, 3, 2))
    effective = torch.zeros_like(carrier) if net.zero_carrier else carrier.to(encoded)
    expanded = effective[:, None].expand(-1, SUPPORT, -1, -1)
    result = net.carrier_post_pool(torch.cat((encoded, expanded), dim=-1)).mean(dim=1)
    require(tuple(result.shape) == (activity.shape[0], UNITS, WINDOW), "late identity geometry drift")
    require(bool(torch.isfinite(result).all()), "late identity is nonfinite")
    return result


def freeze_decoder_train_identity(net: Any) -> tuple[Any, ...]:
    """Freeze the C1 body and expose only the two identity MLPs to Adam."""

    net.eval()
    trainable: list[Any] = []
    for name, parameter in net.named_parameters():
        enabled = name.startswith("carrier_pre_pool.") or name.startswith("carrier_post_pool.")
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(parameter)
    expected = sum(parameter.numel() for parameter in net.carrier_pre_pool.parameters()) + sum(
        parameter.numel() for parameter in net.carrier_post_pool.parameters()
    )
    require(trainable and sum(parameter.numel() for parameter in trainable) == expected == 58_140,
            "identity-branch parameter surface drift")
    require(not net.training, "frozen decoder must remain in evaluation mode")
    return tuple(trainable)


def support_index(*, epoch: int, session: str, batch_ordinal: int, block_count: int, random_arm: bool) -> tuple[int, str]:
    """Return the frozen 50/50 first-M3/non-first-contiguous-M3 choice."""

    require(epoch >= 0 and batch_ordinal >= 0 and block_count >= 2, "support-bank geometry drift")
    if not random_arm or (epoch + batch_ordinal) % 2 == 0:
        return 0, "first_m3"
    token = hashlib.sha256(
        f"h1-srpd|{SEED}|{epoch}|{session}|{batch_ordinal}|{block_count}".encode("utf-8")
    ).digest()
    return 1 + int.from_bytes(token[:8], "big") % (block_count - 1), "nonfirst_uniform"


def normalized_identity_distill(student: Any, teacher: Any) -> Any:
    import torch

    require(isinstance(student, torch.Tensor) and isinstance(teacher, torch.Tensor), "distillation inputs must be tensors")
    require(student.shape == teacher.shape and student.ndim == 3, "distillation identity geometry drift")
    denominator = teacher.detach().square().mean() + DISTILL_FLOOR
    value = torch.nn.functional.mse_loss(student, teacher.detach()) / denominator
    require(bool(torch.isfinite(value)), "distillation loss is nonfinite")
    return value


__all__ = (
    "H1SupportResampledError", "freeze_decoder_train_identity", "late_identity",
    "native_identity", "normalized_identity_distill", "require", "support_index",
)

