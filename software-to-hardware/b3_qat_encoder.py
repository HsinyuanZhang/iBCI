"""QAT EarlyPool encoder with bit-exact integer forward matching b3_quant_engine."""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from b3_fake_quant import (
    B3QATScales,
    B3SharedScales,
    integer_linear_layer,
    integer_mean_pool,
    quant_saturation_stats,
    ste_quantize_symmetric,
)
from b3_quant_engine import FrozenActivationScales

try:
    from src.models.components.streaming_encoders import (
        CalibrationEncoder,
        EarlyPoolEncoder,
        SideFeatureEarlyPoolEncoder,
    )
except ImportError:
    CalibrationEncoder = nn.Module  # type: ignore
    EarlyPoolEncoder = nn.Module  # type: ignore
    SideFeatureEarlyPoolEncoder = nn.Module  # type: ignore


def _trial_to_bnt(trial: torch.Tensor) -> torch.Tensor:
    if trial.dim() == 2:
        trial = trial.unsqueeze(0)
    if trial.dim() != 3:
        raise ValueError(f"Expected [B,T,N] or [T,N], got {tuple(trial.shape)}")
    return trial.permute(0, 2, 1)


class QATEarlyPoolEncoder(CalibrationEncoder):
    variant = "B3_QAT"

    def __init__(
        self,
        base: EarlyPoolEncoder,
        scales: B3QATScales,
        *,
        num_trials: int = 33,
        learnable_scales: bool = False,
        recip_shift: int = 20,
        quantize_weights: bool = True,
        quantize_activations: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(base, (EarlyPoolEncoder, SideFeatureEarlyPoolEncoder)):
            raise TypeError(
                "QATEarlyPoolEncoder requires EarlyPoolEncoder or "
                "SideFeatureEarlyPoolEncoder"
            )
        self.trial_length = base.trial_length
        self.window_size = base.window_size
        self.hidden_dim = base.hidden_dim
        self.side_dim = int(getattr(base, "side_dim", 0))
        self.electrode_embed_dim = int(getattr(base, "electrode_embed_dim", 0))
        if self.electrode_embed_dim:
            raise ValueError(
                "T4 INT8 supports numeric side_features only; absolute electrode "
                "embedding is outside this quantization contract"
            )
        self.pad_value = base.pad_value
        self.num_trials = int(num_trials)
        self.recip_shift = int(recip_shift)
        self.quantize_weights = bool(quantize_weights)
        self.quantize_activations = bool(quantize_activations)

        self.pre_linear = base.pre_pool[0]
        post_layers = [m for m in base.post_pool if isinstance(m, nn.Linear)]
        self.post_linears = nn.ModuleList(post_layers)
        self.shared_scales = B3SharedScales(scales, learnable=learnable_scales)
        recip = int(round((1 << self.recip_shift) / max(self.num_trials, 1)))
        self.register_buffer("reciprocal", torch.tensor(recip, dtype=torch.int64), persistent=False)

    def reset_stream(self, batch_size: int, num_neurons: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        return {
            "sum_feat": torch.zeros(batch_size, num_neurons, self.hidden_dim, device=device, dtype=dtype),
            "trial_count": 0,
        }

    def _quantize_input(self, x_fp: torch.Tensor) -> torch.Tensor:
        if not self.quantize_activations:
            return x_fp
        return ste_quantize_symmetric(x_fp, self.shared_scales.s_input.value(), -128, 127)

    def _pre_trial_feat_q(self, trial_fp: torch.Tensor) -> torch.Tensor:
        x_q = self._quantize_input(trial_fp)
        if not self.quantize_weights and not self.quantize_activations:
            return torch.relu(self.pre_linear(trial_fp))
        if not self.quantize_weights:
            h = torch.relu(self.pre_linear(trial_fp))
            return ste_quantize_symmetric(h, self.shared_scales.s_feat.value(), 0, 127)
        q, _ = integer_linear_layer(
            x_q,
            self.pre_linear.weight,
            self.pre_linear.bias,
            self.shared_scales.s_input.value(),
            self.shared_scales.s_feat.value(),
            relu=True,
        )
        return q

    def push_trial(self, state: Dict[str, Any], trial: torch.Tensor, trial_length: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, Any]:
        trial = _trial_to_bnt(trial)
        feat_q = self._pre_trial_feat_q(trial)
        state["sum_feat"] = state["sum_feat"] + feat_q
        state["trial_count"] += 1
        return state

    def _concat_side(
        self,
        mean_q: torch.Tensor,
        side_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.side_dim == 0:
            if side_features is not None and side_features.shape[-1] != 0:
                raise ValueError("Plain B3 QAT encoder does not accept side_features")
            return mean_q
        if side_features is None:
            raise ValueError(f"B3S QAT encoder requires side_features dim={self.side_dim}")
        if side_features.shape != (*mean_q.shape[:-1], self.side_dim):
            raise ValueError(
                "side_features shape must match pooled activity batch/unit axes: "
                f"expected {(*mean_q.shape[:-1], self.side_dim)}, "
                f"got {tuple(side_features.shape)}"
            )
        side_q = ste_quantize_symmetric(
            side_features,
            self.shared_scales.s_mean.value(),
            -128,
            127,
        )
        return torch.cat([mean_q, side_q], dim=-1)

    def forward_integer_with_stages(
        self,
        calib_fp: torch.Tensor,
        side_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Bit-exact integer path; calib_fp [B,M,T,N] or [M,T,N]."""
        if calib_fp.dim() == 3:
            calib_fp = calib_fp.unsqueeze(0)
        batch_size, m_trials, _, num_neurons = calib_fp.shape
        calib_q = self._quantize_input(calib_fp)
        state = self.reset_stream(batch_size, num_neurons, calib_fp.device, calib_fp.dtype)
        feat_all = []
        for m in range(m_trials):
            trial_q = calib_q[:, m].permute(0, 2, 1)
            if not self.quantize_weights and not self.quantize_activations:
                feat_q = torch.relu(self.pre_linear(calib_fp[:, m].permute(0, 2, 1)))
            elif not self.quantize_weights:
                feat_q = ste_quantize_symmetric(
                    torch.relu(self.pre_linear(calib_fp[:, m].permute(0, 2, 1))),
                    self.shared_scales.s_feat.value(), 0, 127,
                )
            else:
                feat_q, _ = integer_linear_layer(
                    trial_q,
                    self.pre_linear.weight,
                    self.pre_linear.bias,
                    self.shared_scales.s_input.value(),
                    self.shared_scales.s_feat.value(),
                    relu=True,
                )
            state["sum_feat"] = state["sum_feat"] + feat_q
            state["trial_count"] += 1
            feat_all.append(feat_q.clone())
        sum_feat = state["sum_feat"]
        mean_q = integer_mean_pool(
            sum_feat,
            reciprocal=int(self.reciprocal.item()),
            recip_shift=self.recip_shift,
            s_feat=self.shared_scales.s_feat.value(),
            s_mean=self.shared_scales.s_mean.value(),
        )
        post0_input_q = self._concat_side(mean_q, side_features)
        post0_q, post0_acc = integer_linear_layer(
            post0_input_q, self.post_linears[0].weight, self.post_linears[0].bias,
            self.shared_scales.s_mean.value(), self.shared_scales.s_post0.value(), relu=True,
        )
        post1_q, post1_acc = integer_linear_layer(
            post0_q, self.post_linears[1].weight, self.post_linears[1].bias,
            self.shared_scales.s_post0.value(), self.shared_scales.s_post1.value(), relu=True,
        )
        post2_q, post2_acc = integer_linear_layer(
            post1_q, self.post_linears[2].weight, self.post_linears[2].bias,
            self.shared_scales.s_post1.value(), self.shared_scales.s_E.value(), relu=False,
        )
        e_q = post2_q
        e_dequant = e_q * self.shared_scales.s_E.value()
        return {
            "calib_q": calib_q,
            "feat_q": feat_all[-1] if feat_all else sum_feat,
            "sum_feat": sum_feat,
            "mean_q": mean_q,
            "post0_input_q": post0_input_q,
            "post0_q": post0_q,
            "post1_q": post1_q,
            "post2_acc": post2_acc,
            "E_q": e_q,
            "E_dequant": e_dequant,
        }

    def finalize_identity(self, state: Dict[str, Any]) -> torch.Tensor:
        if state["trial_count"] == 0:
            raise ValueError("trial_count must be > 0")
        mean_q = integer_mean_pool(
            state["sum_feat"],
            reciprocal=int(self.reciprocal.item()),
            recip_shift=self.recip_shift,
            s_feat=self.shared_scales.s_feat.value(),
            s_mean=self.shared_scales.s_mean.value(),
        )
        mean_q = self._concat_side(mean_q, state.get("side_features"))
        h, _ = integer_linear_layer(
            mean_q, self.post_linears[0].weight, self.post_linears[0].bias,
            self.shared_scales.s_mean.value(), self.shared_scales.s_post0.value(), relu=True,
        )
        h, _ = integer_linear_layer(
            h, self.post_linears[1].weight, self.post_linears[1].bias,
            self.shared_scales.s_post0.value(), self.shared_scales.s_post1.value(), relu=True,
        )
        acc, post2_acc = integer_linear_layer(
            h, self.post_linears[2].weight, self.post_linears[2].bias,
            self.shared_scales.s_post1.value(), self.shared_scales.s_E.value(), relu=False,
        )
        return acc * self.shared_scales.s_E.value()

    def forward_batch(
        self,
        calib_trials: torch.Tensor,
        trial_lengths: Optional[torch.Tensor] = None,
        side_features: Optional[torch.Tensor] = None,
        electrode_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del trial_lengths
        if electrode_ids is not None:
            raise ValueError("T4 INT8 encoder does not consume electrode_ids")
        return self.forward_integer_with_stages(
            calib_trials, side_features=side_features
        )["E_dequant"]

    def forward_fp32_stages(
        self,
        calib_trials: torch.Tensor,
        side_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if calib_trials.dim() == 3:
            calib_trials = calib_trials.unsqueeze(0)
        batch_size = calib_trials.shape[0]
        state = self.reset_stream(batch_size, calib_trials.shape[-1], calib_trials.device, calib_trials.dtype)
        pre_relu_all = []
        for m in range(calib_trials.shape[1]):
            trial = _trial_to_bnt(calib_trials[:, m])
            feat = torch.relu(self.pre_linear(trial))
            pre_relu_all.append(feat)
            state["sum_feat"] = state["sum_feat"] + feat
            state["trial_count"] += 1
        mean_feat = state["sum_feat"] / float(state["trial_count"])
        pre_relu_cat = torch.cat(pre_relu_all, dim=0)
        h = mean_feat
        if self.side_dim:
            if side_features is None:
                raise ValueError(
                    f"B3S FP shadow requires side_features dim={self.side_dim}"
                )
            if side_features.shape != (*mean_feat.shape[:-1], self.side_dim):
                raise ValueError(
                    f"side_features shape mismatch: {tuple(side_features.shape)}"
                )
            h = torch.cat([h, side_features], dim=-1)
        post0 = self.post_linears[0](h)
        post0_relu = torch.relu(post0)
        post1 = self.post_linears[1](post0_relu)
        post1_relu = torch.relu(post1)
        e = self.post_linears[2](post1_relu)
        return {
            "input": calib_trials,
            "pre_relu": pre_relu_cat,
            "mean_feat": torch.relu(mean_feat),
            "post0_input": h,
            "post0_relu": post0_relu,
            "post1_relu": post1_relu,
            "E": e,
        }

    def forward_fp32(
        self,
        calib_trials: torch.Tensor,
        side_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_fp32_stages(
            calib_trials, side_features=side_features
        )["E"]

    @torch.no_grad()
    def compute_quant_diagnostics(
        self,
        calib_fp: torch.Tensor,
        side_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        s = self.shared_scales
        stages = self.forward_fp32_stages(
            calib_fp, side_features=side_features
        )
        diagnostics = {
            "scales_relative": s.relative_to_init_dict(),
            "scales_current": s.as_dict(),
            "input": quant_saturation_stats(stages["input"], s.s_input.value(), -128, 127),
            "pre_out": quant_saturation_stats(stages["pre_relu"], s.s_feat.value(), 0, 127),
            "mean": quant_saturation_stats(stages["mean_feat"], s.s_mean.value(), 0, 127),
            "post0_out": quant_saturation_stats(stages["post0_relu"], s.s_post0.value(), 0, 127),
            "post1_out": quant_saturation_stats(stages["post1_relu"], s.s_post1.value(), 0, 127),
            "E": quant_saturation_stats(stages["E"], s.s_E.value(), -128, 127),
        }
        if self.side_dim:
            assert side_features is not None
            diagnostics["side"] = quant_saturation_stats(
                side_features, s.s_mean.value(), -128, 127
            )
            diagnostics["side"]["shared_post0_input_scale"] = float(
                s.s_mean.value().detach().cpu()
            )
        return diagnostics

    def export_scales_dict(self) -> Dict[str, float]:
        return self.shared_scales.as_dict()

    def to_frozen_scales(self, source_sessions: Optional[List[str]] = None) -> FrozenActivationScales:
        d = self.export_scales_dict()
        s8 = d["input"]
        return FrozenActivationScales(
            input_scale_i8=d["input"],
            input_scale_i16=s8 * 127.0 / 32767.0,
            pre_out_scale_i8=d["pre_out"],
            pre_out_scale_i16=d["pre_out"] * 127.0 / 32767.0,
            mean_scale_i8=d["mean"],
            mean_scale_i16=d["mean"] * 127.0 / 32767.0,
            post0_out_scale_i8=d["post0_out"],
            post0_out_scale_i16=d["post0_out"] * 127.0 / 32767.0,
            post1_out_scale_i8=d["post1_out"],
            post1_out_scale_i16=d["post1_out"] * 127.0 / 32767.0,
            E_int8_scale=d["E"],
            E_int16_scale=d["E"] * 127.0 / 32767.0,
            source_sessions=list(source_sessions or []),
        )


def wrap_early_pool_with_qat(
    encoder: EarlyPoolEncoder,
    scales: B3QATScales,
    *,
    num_trials: int = 33,
    learnable_scales: bool = False,
    **kwargs,
) -> QATEarlyPoolEncoder:
    return QATEarlyPoolEncoder(encoder, scales, num_trials=num_trials, learnable_scales=learnable_scales, **kwargs)


def clone_anchor_encoder(base: EarlyPoolEncoder) -> EarlyPoolEncoder:
    cloned = copy.deepcopy(base)
    for p in cloned.parameters():
        p.requires_grad = False
    cloned.eval()
    return cloned
