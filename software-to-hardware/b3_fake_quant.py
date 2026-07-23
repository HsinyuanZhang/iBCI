"""Bit-exact integer ops + STE backward matching b3_quant_engine."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

INT8_MAX = 127
INT8_MIN = -128


@dataclass
class B3QATScales:
    input: float
    pre_out: float
    mean: float
    post0_out: float
    post1_out: float
    E: float


class SharedScale(nn.Module):
    """Single positive scale parameter shared across tensor edges (log parameterization)."""

    def __init__(self, init: float, *, learnable: bool = False, bounds_ratio: float = 4.0) -> None:
        super().__init__()
        init = max(float(init), 1e-8)
        log_init = torch.tensor(init).log().clamp_min(-20.0)
        self.register_buffer("init_log_scale", log_init.clone())
        log_bound = math.log(max(bounds_ratio, 1.01))
        self.register_buffer("log_min", (log_init - log_bound).clone())
        self.register_buffer("log_max", (log_init + log_bound).clone())
        if learnable:
            self.log_scale = nn.Parameter(log_init.clone())
        else:
            self.register_buffer("log_scale", log_init.clone())

    def value(self) -> torch.Tensor:
        log_s = self.log_scale
        if isinstance(log_s, nn.Parameter):
            log_s = log_s.clamp(self.log_min.item(), self.log_max.item())
        return log_s.exp().clamp_min(1e-8)

    def relative_to_init(self) -> float:
        return float((self.value().detach() / self.init_log_scale.exp().clamp_min(1e-8)).cpu())

    def forward(self) -> torch.Tensor:
        return self.value()


class B3SharedScales(nn.Module):
    """Six logical scales with layer-edge sharing."""

    def __init__(self, init: B3QATScales, *, learnable: bool = False) -> None:
        super().__init__()
        self.s_input = SharedScale(init.input, learnable=learnable)
        self.s_feat = SharedScale(init.pre_out, learnable=learnable)
        self.s_mean = SharedScale(init.mean, learnable=learnable)
        self.s_post0 = SharedScale(init.post0_out, learnable=learnable)
        self.s_post1 = SharedScale(init.post1_out, learnable=learnable)
        self.s_E = SharedScale(init.E, learnable=learnable)

    def as_dict(self) -> dict[str, float]:
        return {
            "input": float(self.s_input.value().detach().cpu()),
            "pre_out": float(self.s_feat.value().detach().cpu()),
            "mean": float(self.s_mean.value().detach().cpu()),
            "post0_out": float(self.s_post0.value().detach().cpu()),
            "post1_out": float(self.s_post1.value().detach().cpu()),
            "E": float(self.s_E.value().detach().cpu()),
        }

    def relative_to_init_dict(self) -> dict[str, float]:
        return {
            "input": self.s_input.relative_to_init(),
            "pre_out": self.s_feat.relative_to_init(),
            "mean": self.s_mean.relative_to_init(),
            "post0_out": self.s_post0.relative_to_init(),
            "post1_out": self.s_post1.relative_to_init(),
            "E": self.s_E.relative_to_init(),
        }

    def scale_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for mod in (self.s_input, self.s_feat, self.s_mean, self.s_post0, self.s_post1, self.s_E):
            if isinstance(mod.log_scale, nn.Parameter):
                params.append(mod.log_scale)
        return params


def ste_identity(exact: torch.Tensor, surrogate: torch.Tensor) -> torch.Tensor:
    return exact + (surrogate - surrogate.detach())


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return ste_identity(torch.round(x), x)


def ste_quantize_symmetric(x_fp: torch.Tensor, scale: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    """Exact round+clamp forward; STE through x/scale."""
    scale = scale.clamp_min(1e-8)
    x_scaled = x_fp / scale
    q_exact = torch.clamp(torch.round(x_scaled), qmin, qmax)
    return ste_identity(q_exact, x_scaled)


def quant_saturation_stats(
    x_fp: torch.Tensor,
    scale: torch.Tensor,
    qmin: int,
    qmax: int,
) -> Dict[str, float]:
    """Pre-round clip rate and dynamic range for a quantizer edge."""
    scale = scale.clamp_min(1e-8)
    pre = torch.round(x_fp / scale)
    clipped = torch.clamp(pre, qmin, qmax)
    sat = (pre != clipped).float()
    return {
        "saturation_rate": float(sat.mean().cpu()),
        "clip_rate": float(sat.mean().cpu()),
        "pre_clip_max": float(pre.max().cpu()),
        "pre_clip_min": float(pre.min().cpu()),
    }


def weight_scales_per_channel(w_fp: torch.Tensor) -> torch.Tensor:
    return w_fp.abs().amax(dim=1).clamp_min(1e-8) / float(INT8_MAX)


def quantize_weight_channel_ste(w_fp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    w_scale = weight_scales_per_channel(w_fp)
    w_q = ste_quantize_symmetric(w_fp, w_scale.unsqueeze(1), INT8_MIN, INT8_MAX)
    return w_q, w_scale


def _quantize_weight_exact(w_fp: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(
        torch.round(w_fp / w_scale.unsqueeze(1).clamp_min(1e-8)),
        INT8_MIN,
        INT8_MAX,
    )


def integer_shift_right(prod_i64: torch.Tensor, shift: int) -> torch.Tensor:
    """Arithmetic right shift with half-LSB rounding (matches b3_quant_engine)."""
    if shift <= 0:
        return prod_i64
    half = 1 << (shift - 1)
    return (prod_i64 + half) >> shift


def integer_requant_forward(
    acc: torch.Tensor,
    mult: torch.Tensor,
    shift: int,
    *,
    relu: bool,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    acc64 = acc.to(torch.int64)
    mult64 = mult.to(torch.int64)
    while mult64.ndim < acc64.ndim:
        mult64 = mult64.reshape(*([1] * (acc64.ndim - mult64.ndim)), *mult64.shape)
    prod = acc64 * mult64
    rounded = integer_shift_right(prod, shift)
    if relu:
        rounded = torch.clamp(rounded, min=0)
    return torch.clamp(rounded, qmin, qmax).to(torch.float32)


def _requant_surrogate(
    acc_sur: torch.Tensor,
    eff_scale: torch.Tensor,
    out_scale: torch.Tensor,
    *,
    relu: bool,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """Float accumulator × eff / out_scale once (no second division)."""
    ratio = eff_scale / out_scale.clamp_min(1e-8)
    while ratio.ndim < acc_sur.ndim:
        ratio = ratio.reshape(*([1] * (acc_sur.ndim - ratio.ndim)), *ratio.shape)
    y_sur = acc_sur * ratio
    if relu:
        y_sur = torch.clamp(y_sur, min=0.0)
    return torch.clamp(y_sur, float(qmin), float(qmax))


def integer_requant_ste(
    acc_exact: torch.Tensor,
    mult: torch.Tensor,
    shift: int,
    *,
    relu: bool,
    qmin: int,
    qmax: int,
    acc_sur: torch.Tensor,
    eff_scale: torch.Tensor,
    out_scale: torch.Tensor,
) -> torch.Tensor:
    q_exact = integer_requant_forward(acc_exact, mult, shift, relu=relu, qmin=qmin, qmax=qmax)
    q_sur = _requant_surrogate(acc_sur, eff_scale, out_scale, relu=relu, qmin=qmin, qmax=qmax)
    return ste_identity(q_exact, q_sur)


def _integer_mac_exact(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    bias_i32: torch.Tensor,
) -> torch.Tensor:
    lead = x_q.shape[:-1]
    out_f = w_q.shape[0]
    x_flat = torch.round(x_q).detach().reshape(-1, x_q.shape[-1])
    w_i = w_q.detach().reshape(out_f, -1)
    acc = torch.round(
        x_flat.to(torch.float64) @ w_i.to(torch.float64).t()
        + bias_i32.detach().to(torch.float64).unsqueeze(0)
    ).to(torch.int64).reshape(*lead, out_f)
    return acc


def _integer_mac_surrogate(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    bias_fp: torch.Tensor,
) -> torch.Tensor:
    lead = x_q.shape[:-1]
    out_f = w_q.shape[0]
    x_flat = x_q.reshape(-1, x_q.shape[-1])
    w_i = w_q.reshape(out_f, -1)
    return (x_flat @ w_i.t() + bias_fp.unsqueeze(0)).reshape(*lead, out_f)


def integer_linear_layer(
    x_q: torch.Tensor,
    w_fp: torch.Tensor,
    b_fp: torch.Tensor,
    in_scale: torch.Tensor,
    out_scale: torch.Tensor,
    *,
    relu: bool,
    requant_shift: int = 31,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """x_q integer-valued float [..., in]; returns (q_out integer-valued, acc exact int path)."""
    w_scale = weight_scales_per_channel(w_fp)
    eff = in_scale * w_scale
    bias_fp_sur = b_fp / eff.clamp_min(1e-8)

    w_q_exact = _quantize_weight_exact(w_fp, w_scale)
    bias_exact = torch.round(bias_fp_sur)
    w_q_sur, _ = quantize_weight_channel_ste(w_fp)

    acc_exact = _integer_mac_exact(x_q, w_q_exact, bias_exact)
    acc_sur = _integer_mac_surrogate(x_q, w_q_sur, bias_fp_sur)

    ratio = eff.detach() / out_scale.detach().clamp_min(1e-8)
    mult = torch.round(ratio * float(1 << requant_shift)).to(torch.int64)
    qmin, qmax = (0, INT8_MAX) if relu else (INT8_MIN, INT8_MAX)
    q = integer_requant_ste(
        acc_exact,
        mult,
        requant_shift,
        relu=relu,
        qmin=qmin,
        qmax=qmax,
        acc_sur=acc_sur,
        eff_scale=eff,
        out_scale=out_scale,
    )
    return q, acc_exact


def integer_mean_pool(
    sum_feat_q: torch.Tensor,
    *,
    reciprocal: int,
    recip_shift: int,
    s_feat: torch.Tensor,
    s_mean: torch.Tensor,
) -> torch.Tensor:
    sum_round = torch.round(sum_feat_q)
    sum_exact = sum_round.detach().to(torch.int64)
    prod_exact = sum_exact * int(reciprocal)
    mean_i_exact = integer_shift_right(prod_exact, recip_shift).to(torch.float32)
    mean_fp_exact = torch.clamp(mean_i_exact * s_feat.detach(), min=0.0)
    q_exact = torch.clamp(
        torch.round(mean_fp_exact / s_mean.detach().clamp_min(1e-8)),
        0,
        INT8_MAX,
    ).to(torch.float32)

    mean_fp_sur = torch.clamp(
        sum_feat_q * (float(reciprocal) / float(1 << recip_shift)) * s_feat,
        min=0.0,
    )
    q_sur = torch.clamp(mean_fp_sur / s_mean.clamp_min(1e-8), 0.0, float(INT8_MAX))
    return ste_identity(q_exact, q_sur)
