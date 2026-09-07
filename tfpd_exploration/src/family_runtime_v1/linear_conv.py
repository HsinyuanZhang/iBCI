"""Algebraic k=5 local repair as five small linear projections.

This removes the Conv1d dispatch and four discarded suffix activations. It
preserves the finite-window zero boundary, weights, bias, and activation.
Different CPU kernels may round FP32 dot products differently.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def repaired_local_features_linear(local_conv, raw: torch.Tensor) -> torch.Tensor:
    """Return [B,5,N,C] for window indices (0,1,2,3,W-1)."""
    if raw.ndim != 3 or raw.shape[1] < 9:
        raise ValueError("linear repair requires [B,W,N] with W >= 9")
    conv = local_conv.conv
    if (int(local_conv.cfg.conv_kernel) != 5 or local_conv.left_pad != 4
            or conv.kernel_size != (5,) or conv.stride != (1,)
            or conv.dilation != (1,) or conv.padding != (0,)
            or conv.groups != 1 or conv.in_channels != 1
            or conv.padding_mode != "zeros"):
        raise ValueError("linear repair requires single-input causal k=5, stride=dilation=1")
    # The prefix has an independent left boundary; never borrow bins that
    # have fallen outside the current finite window.
    prefix = F.pad(raw[:, :4].transpose(1, 2), (4, 0)).unfold(-1, 5, 1)
    newest = raw[:, -5:].transpose(1, 2).unsqueeze(2)
    contexts = torch.cat((prefix, newest), dim=2)
    projected = F.linear(contexts, conv.weight[:, 0, :], conv.bias)
    return local_conv.act(projected).permute(0, 2, 1, 3).contiguous()
