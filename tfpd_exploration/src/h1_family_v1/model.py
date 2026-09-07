"""CRST-B4 H1 common frontend models for the source-only preflight.

The FLAT initialization deliberately comes from H1 V2 rather than the older
``initialize_h1_flat`` helper: this keeps all non-scale choices byte-for-byte
on the V2 constructor path.  ROUTE then copies every shared tensor and creates
only its frozen-design routing tensors.
"""
from __future__ import annotations

import torch
from torch import nn

from tfpd_exploration.src.h1_optimized_v2.model import H1FullWindowControl
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_config import H1TemporalConfig, H1_TEMPORAL
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import (
    CausalTransformerStack,
    H1Bank,
    SharedSetFrontend,
    SlotUnitAttention,
    _reinit_param,
    copy_flat_into_route,
    initialize_route_only,
    prove_zero_gate_equals_flat,
    shared_parameter_max_abs_diff,
)


class UnscaledDotSlotUnitAttention(SlotUnitAttention):
    """Common set-v2 rule: remove only the default sqrt(head_dim) attenuation.

    Both FLAT's qk term and ROUTE's already-defined additive calibration bonus
    are multiplied once, before the unchanged mask and softmax.
    """

    logit_multiplier_name = "sqrt_head_dim"

    def forward(self, slots, tokens, unit_keep, e0=None, hc=None):  # type: ignore[override]
        batch_width, n_slots, dim = slots.shape
        n_units = tokens.size(1)
        query = self.q_proj(slots).view(batch_width, n_slots, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(tokens).view(batch_width, n_units, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(tokens).view(batch_width, n_units, self.n_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if self.routed:
            if e0 is None or hc is None:
                raise RuntimeError("ROUTE needs E0 and H-C")
            batch = unit_keep.size(0)
            width = batch_width // batch
            bonus = self.routing_bonus(e0, hc, batch)
            logits = logits + bonus.unsqueeze(1).expand(-1, width, -1, -1, -1).reshape(batch_width, self.n_heads, n_slots, n_units)
        # The one intentional v2 rule: (qk/sqrt(d) + legal route bonus) * sqrt(d).
        logits = logits * (self.head_dim ** 0.5)
        pad = (~unit_keep).unsqueeze(1).expand(-1, batch_width // unit_keep.size(0), -1).reshape(batch_width, n_units)
        weights = torch.softmax(logits.masked_fill(pad.unsqueeze(1).unsqueeze(2), float("-inf")), dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        hidden = torch.matmul(weights, value)
        hidden = hidden.transpose(1, 2).contiguous().view(batch_width, n_slots, dim)
        return self.out_proj(hidden)


def _replace_attention_with_unscaled(frontend: SharedSetFrontend) -> None:
    old = frontend.attn
    new = UnscaledDotSlotUnitAttention(frontend.cfg, routed=old.routed)
    new.load_state_dict(old.state_dict(), strict=True)
    frontend.attn = new


class H1V2RouteWindowControl(nn.Module):
    """V2 full-window reader with the sole FLAT -> ROUTE attention difference."""

    training_target_space = "runtime_scaled_velocity__target_is_20x_native"
    prediction_divisor = 20.0

    def __init__(self, cfg: H1TemporalConfig | None = None, seed: int = 42) -> None:
        super().__init__()
        self.cfg = cfg or H1_TEMPORAL
        self.activity_scale = 1.0
        self.frontend = SharedSetFrontend(self.cfg, routed=True)
        self.final_norm = nn.LayerNorm(self.cfg.temporal_width)
        self.readout = nn.Sequential(
            nn.Linear(self.cfg.temporal_width, self.cfg.readout_hidden),
            nn.GELU(),
            nn.Linear(self.cfg.readout_hidden, self.cfg.out_dim),
        )
        self.temporal = CausalTransformerStack(self.cfg)

    def encode_frontend(self, x: torch.Tensor, bank: H1Bank, dropout_keep: torch.Tensor | None = None) -> torch.Tensor:
        keep = bank.unit_mask if dropout_keep is None else dropout_keep
        if keep.ndim == 1:
            keep = keep.unsqueeze(0).expand(x.size(0), -1)
        return self.frontend(
            x,
            bank.E0.to(x),
            bank.T.to(x),
            keep.to(device=x.device, dtype=torch.bool),
            tile_size=None,
            use_checkpoint=False,
        )

    def forward_hidden(self, x: torch.Tensor, bank: H1Bank, dropout_keep: torch.Tensor | None = None) -> torch.Tensor:
        return self.temporal(self.encode_frontend(x, bank, dropout_keep))[:, -1]

    def forward_last(self, x: torch.Tensor, bank: H1Bank, dropout_keep: torch.Tensor | None = None) -> torch.Tensor:
        return self.readout(self.final_norm(self.forward_hidden(x, bank, dropout_keep)))


def make_v2_initialized_pair(*, seed: int = 42, cfg: H1TemporalConfig | None = None) -> tuple[H1FullWindowControl, H1V2RouteWindowControl]:
    """Make scale-1 FLAT/ROUTE with exact V2 shared initialization."""
    flat = H1FullWindowControl(cfg=cfg, seed=seed, activity_scale=1.0)
    route = H1V2RouteWindowControl(cfg=cfg, seed=seed)
    copy_flat_into_route(flat, route)
    initialize_route_only(route, seed)
    return flat, route


def make_v2_unscaled_dot_pair(*, seed: int = 42, cfg: H1TemporalConfig | None = None) -> tuple[H1FullWindowControl, H1V2RouteWindowControl]:
    """V2 seed-42 weights with exactly one new shared softmax-logit rule."""
    flat = H1FullWindowControl(cfg=cfg, seed=seed, activity_scale=1.0)
    _replace_attention_with_unscaled(flat.frontend)
    route = H1V2RouteWindowControl(cfg=cfg, seed=seed)
    _replace_attention_with_unscaled(route.frontend)
    copy_flat_into_route(flat, route)
    initialize_route_only(route, seed)
    return flat, route


def local_fc1_balance_factor(cfg: H1TemporalConfig | None = None) -> float:
    active = cfg or H1_TEMPORAL
    return float(((active.local_dim + active.e0_dim + active.hc_dim) / active.local_dim) ** 0.5)


def make_v2_unscaled_dot_localbalanced_pair(*, seed: int = 42, cfg: H1TemporalConfig | None = None) -> tuple[H1FullWindowControl, H1V2RouteWindowControl]:
    """Prospective init-only local FC1 balancing over an otherwise v2 pair."""
    flat, route = make_v2_unscaled_dot_pair(seed=seed, cfg=cfg)
    factor = local_fc1_balance_factor(flat.cfg)
    with torch.no_grad():
        # FC1 order is exactly [local16 | E0_700 | T_4].  Route receives the
        # same shared scaling; only its legal attention bonus remains unique.
        flat.frontend.token_mlp.fc1.weight[:, :flat.cfg.local_dim].mul_(factor)
        route.frontend.token_mlp.fc1.weight[:, :route.cfg.local_dim].mul_(factor)
    return flat, route


def initialization_receipt(flat: nn.Module, route: H1V2RouteWindowControl) -> dict[str, float | int]:
    route_names = [name for name, _ in route.named_parameters() if ".attn.route_" in name or name.endswith(".attn.q_cal") or name.endswith(".attn.g")]
    if not route_names:
        raise RuntimeError("ROUTE has no route-only parameters")
    if float(route.frontend.attn.g.abs().max().item()) != 0.0:
        raise RuntimeError("ROUTE gate is not initialized to zero")
    shared = shared_parameter_max_abs_diff(flat, route)
    if shared != 0.0:
        raise RuntimeError(f"shared FLAT/ROUTE initialization mismatch: {shared}")
    return {"shared_parameter_max_abs_diff": shared, "route_only_parameter_tensors": len(route_names)}


def zero_gate_parity(flat: H1FullWindowControl, route: H1V2RouteWindowControl, x: torch.Tensor, bank: H1Bank) -> dict[str, float]:
    # The helper is structurally duck-typed despite its historical annotations.
    return prove_zero_gate_equals_flat(flat, route, x, bank, atol=1e-5, rtol=1e-4)


def route_gate_gradient_l1(route: H1V2RouteWindowControl) -> float:
    grad = route.frontend.attn.g.grad
    if grad is None:
        return 0.0
    return float(grad.detach().abs().sum().item())
