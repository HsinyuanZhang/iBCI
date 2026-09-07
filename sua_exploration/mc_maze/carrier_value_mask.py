"""Forward-only carrier-value masking primitives for the A2 B3S decoder.

The diagnostic removes complete unit tokens from cross-attention.  Its source
screen therefore uses both attention weight and the projected value norm;
attention mass alone is not evidence that a unit can or cannot affect output.
No production module or parameter is modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F


class CarrierValueMaskError(RuntimeError):
    pass


MaskMode = Literal["low_gain", "high_gain", "random"]


@dataclass(frozen=True)
class ValueWeightedImportance:
    attention_mass: torch.Tensor
    projected_value_norm: torch.Tensor
    value_weighted_mass: torch.Tensor


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CarrierValueMaskError(message)


def _validate_student(student: Any) -> None:
    _require(not bool(student.training), "carrier-value mask requires eval mode")
    _require(getattr(student, "decoder_mode", None) == "coupled", "mask requires coupled decoder")
    _require(getattr(student, "fixed_slot_router", None) is None, "mask forbids fixed-slot routing")
    encoder = getattr(student, "id_encoder", None)
    _require(encoder is not None and getattr(encoder, "side_dim", None) == 4,
             "mask requires B3S side_dim=4")
    decoder = getattr(student, "decoder", None)
    _require(decoder is not None and len(decoder.transformer.layers) == 1,
             "source mechanism probe is frozen to one cross-attention layer")


def mask_count(num_units: int, fraction: float) -> int:
    _require(num_units > 1, "mask requires at least two units")
    _require(0.0 < float(fraction) < 1.0, "mask fraction must be in (0,1)")
    return min(num_units - 1, max(1, int(num_units * float(fraction))))


def unit_mask(
    side_features: torch.Tensor,
    *,
    fraction: float,
    mode: MaskMode,
    seed: int | None = None,
) -> torch.Tensor:
    """Return a boolean ``[B,N]`` key-padding mask.

    T4 is ordered ``[a,c,m,b]``; ranking standardized ``m`` preserves the raw
    within-session ordering because the frozen source standard deviation is
    strictly positive.  Random control permutations are frozen on CPU and then
    copied to the execution device.
    """
    _require(side_features.ndim == 3 and side_features.shape[-1] == 4,
             "side_features must have shape [B,N,4]")
    batch, units, _ = side_features.shape
    count = mask_count(units, fraction)
    result = torch.zeros(batch, units, dtype=torch.bool, device=side_features.device)
    if mode in {"low_gain", "high_gain"}:
        gain = side_features[..., 2]
        order = torch.argsort(gain, dim=1, descending=(mode == "high_gain"), stable=True)
        result.scatter_(1, order[:, :count], True)
        return result
    _require(mode == "random", f"unknown mask mode: {mode}")
    _require(seed is not None, "random mask requires a frozen seed")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(units, generator=generator)[:count].to(side_features.device)
    result[:, order] = True
    return result


def decode_with_unit_mask(
    student: Any,
    neural: torch.Tensor,
    calib_trials: torch.Tensor,
    side_features: torch.Tensor,
    key_padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the unchanged coupled read-in with selected unit tokens excluded."""
    _validate_student(student)
    _require(neural.ndim == 3 and calib_trials.ndim == 4, "invalid neural/calibration rank")
    batch, _window, units = neural.shape
    _require(side_features.shape[:2] == (batch, units), "side-feature unit shape drift")
    _require(key_padding_mask.shape == (batch, units) and key_padding_mask.dtype == torch.bool,
             "key padding mask must be bool [B,N]")
    _require((~key_padding_mask).any(dim=1).all().item(), "mask cannot remove every unit")
    identity = student.compute_identity(calib_trials, side_features=side_features)
    tokens = student.decoder.fc_in(neural.permute(0, 2, 1) + identity)
    rep = student.decoder.fc_in(student.decoder.rep).to(tokens)
    transformed, _scores = student.decoder.transformer(
        rep.repeat(batch, 1, 1), tokens, key_padding_mask=key_padding_mask
    )
    return student.decoder.fc_out(transformed).permute(0, 2, 1)


def first_layer_value_weighted_importance(
    student: Any,
    neural: torch.Tensor,
    calib_trials: torch.Tensor,
    side_features: torch.Tensor,
) -> ValueWeightedImportance:
    """Measure per-unit attention mass times projected-value norm.

    This is a nonnegative routing proxy.  It is deliberately not called an
    exact output decomposition because output projection and downstream FFN
    mixing can create signed cancellation.
    """
    _validate_student(student)
    batch, _window, units = neural.shape
    identity = student.compute_identity(calib_trials, side_features=side_features)
    tokens = student.decoder.fc_in(neural.permute(0, 2, 1) + identity)
    query = student.decoder.fc_in(student.decoder.rep).to(tokens).repeat(batch, 1, 1)
    layer = student.decoder.transformer.layers[0]
    query_norm = layer.norm1(query)
    token_norm = layer.norm1(tokens)
    mha = layer.cross_attn
    _require(mha.in_proj_weight is not None and mha._qkv_same_embed_dim,
             "diagnostic requires fused same-width Q/K/V projection")
    _output, weights = mha(
        query=query_norm,
        key=token_norm,
        value=token_norm,
        need_weights=True,
        average_attn_weights=False,
    )
    _require(weights.ndim == 4 and weights.shape[-1] == units,
             "unexpected per-head attention shape")
    width = int(mha.embed_dim)
    value_weight = mha.in_proj_weight[2 * width : 3 * width]
    value_bias = None if mha.in_proj_bias is None else mha.in_proj_bias[2 * width : 3 * width]
    projected_value = F.linear(token_norm, value_weight, value_bias)
    heads = int(mha.num_heads)
    head_dim = width // heads
    value_by_head = projected_value.reshape(batch, units, heads, head_dim).permute(0, 2, 1, 3)
    value_norm = torch.linalg.vector_norm(value_by_head, dim=-1)
    attention_mass = weights.mean(dim=(1, 2))
    value_weighted = (weights * value_norm.unsqueeze(2)).mean(dim=(1, 2))
    return ValueWeightedImportance(
        attention_mass=attention_mass,
        projected_value_norm=value_norm.mean(dim=1),
        value_weighted_mass=value_weighted,
    )


def masked_share(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    _require(values.ndim == 2 and mask.shape == values.shape and mask.dtype == torch.bool,
             "masked-share inputs must be [B,N] with a boolean mask")
    _require(torch.isfinite(values).all().item() and (values >= 0).all().item(),
             "masked-share values must be finite and nonnegative")
    total = values.sum(dim=1)
    _require((total > 0).all().item(), "masked-share denominator must be positive")
    return values.masked_fill(~mask, 0).sum(dim=1) / total
