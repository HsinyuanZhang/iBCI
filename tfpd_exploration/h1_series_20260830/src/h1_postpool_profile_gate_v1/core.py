"""Pure profile-gate operator."""
from __future__ import annotations

from typing import Any

from .plan import PROFILE_LENGTH


def profile_identity(static: Any, post: Any, profile: Any) -> Any:
    import torch

    if not (isinstance(static, torch.Tensor) and isinstance(post, torch.Tensor)
            and isinstance(profile, torch.Tensor)):
        raise TypeError("profile operator requires tensors")
    if tuple(static.shape) != tuple(post.shape) or tuple(static.shape)[-1] != PROFILE_LENGTH:
        raise ValueError("profile identity geometry drift")
    if tuple(profile.shape) != (PROFILE_LENGTH,):
        raise ValueError("profile vector geometry drift")
    if not bool(torch.isfinite(static).all() and torch.isfinite(post).all() and torch.isfinite(profile).all()):
        raise ValueError("profile identity input nonfinite")
    if bool(torch.count_nonzero(profile).item() == 0) and not profile.requires_grad:
        return static
    result = static + torch.tanh(profile).unsqueeze(0) * (post - static)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("profile identity result nonfinite")
    return result


__all__ = ("profile_identity",)
