"""Exact head-grouped value projection for lifted M2 slot attention."""
from __future__ import annotations

import torch


def grouped_value_projection(attended_tokens: torch.Tensor, value_weight: torch.Tensor) -> torch.Tensor:
    """Apply one value matrix per head without broadcasting it across batches.

    attended_tokens has shape [B,H,S,D] and value_weight has [H,D,HD].
    All B*S rows of a head are grouped into one bmm. The algebra and FP32
    precision are unchanged; kernel-dependent reduction order may round
    individual dot products differently and is checked by parity tests.
    """
    batch, heads, slots, width = attended_tokens.shape
    if value_weight.shape[:2] != (heads, width):
        raise ValueError("head/value projection geometry drift")
    grouped = attended_tokens.permute(1, 0, 2, 3).reshape(heads, batch * slots, width)
    projected = torch.bmm(grouped, value_weight)
    return projected.reshape(heads, batch, slots, value_weight.shape[2]).permute(1, 0, 2, 3)
