"""Low-state, baseline-preserving T4 attention-logit residual for SPINT.

The selected coupled T4 path remains unchanged.  Calibration caches only a
rank-``r`` factor per unit.  At decode time a learned per-output-query factor
forms a bias ``B[c,n] = q[c] @ u[n]``.  The primary arm adds this bias to every
teacher attention head; the parameter-matched additive control aggregates the
same bias after attention and never changes attention selection.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .spint import SpintModel
from .streaming_encoders import CalibrationEncoder
from .streaming_spint import StreamingSpintModel


def _tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class T4LogitResidualState:
    """Calibration-static per-unit factors, shape ``[B,N,r]``."""

    unit_factors: torch.Tensor

    def __post_init__(self) -> None:
        if self.unit_factors.ndim != 3:
            raise ValueError("unit_factors must have shape [B,N,r]")

    @property
    def nbytes(self) -> int:
        return self.unit_factors.numel() * self.unit_factors.element_size()


class ZeroInitializedT4LogitResidual(nn.Module):
    """Factorized T4/query interaction with an exactly zero initial bias."""

    def __init__(self, *, t4_dim: int, rank: int, num_queries: int) -> None:
        super().__init__()
        if t4_dim <= 0 or rank <= 0 or num_queries <= 0:
            raise ValueError("t4_dim, rank, and num_queries must be positive")
        self.t4_dim = int(t4_dim)
        self.rank = int(rank)
        self.num_queries = int(num_queries)
        self.unit_projection = nn.Linear(self.t4_dim, self.rank, bias=False)
        self.activation = nn.Tanh()
        self.query_factors = nn.Parameter(torch.zeros(self.num_queries, self.rank))
        self._initial_sha256 = self.factor_sha256()

    def derive_unit_factors(self, t4: torch.Tensor) -> torch.Tensor:
        if t4.ndim != 3 or t4.shape[-1] != self.t4_dim:
            raise ValueError(f"T4 must have shape [B,N,{self.t4_dim}]")
        return self.activation(self.unit_projection(t4))

    def logit_bias(self, unit_factors: torch.Tensor) -> torch.Tensor:
        if unit_factors.ndim != 3 or unit_factors.shape[-1] != self.rank:
            raise ValueError(f"unit_factors must have shape [B,N,{self.rank}]")
        return torch.einsum("cr,bnr->bcn", self.query_factors, unit_factors)

    def forward(self, t4: torch.Tensor) -> torch.Tensor:
        return self.logit_bias(self.derive_unit_factors(t4))

    def factor_sha256(self) -> str:
        return _tensor_state_sha256(dict(self.state_dict()))

    @property
    def initialization_receipt(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "family": "zero_initialized_factorized_t4_attention_logit_residual",
            "t4_dim": self.t4_dim,
            "rank": self.rank,
            "num_queries": self.num_queries,
            "initial_factor_sha256": self._initial_sha256,
            "active_factor_sha256": self.factor_sha256(),
            "query_factors_zero_initialized": True,
            "bias_parameters": 0,
            "parameter_count": self.t4_dim * self.rank + self.num_queries * self.rank,
        }


def _teacher_multihead_attention_with_shared_logit_bias(
    attention: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the teacher's ``need_weights=True`` path, then add ``bias``.

    PyTorch changes from ``bmm`` to ``baddbmm`` whenever ``attn_mask`` is
    supplied.  At production width that creates small floating-point changes
    even for an all-zero mask.  Keeping the original ``bmm`` and adding the
    shared bias afterward makes a zero residual bitwise identical while
    retaining ordinary gradients and the trained non-zero mechanism.
    """
    if not attention.batch_first:
        raise ValueError("T4 logit residual requires batch_first teacher attention")
    if attention.bias_k is not None or attention.bias_v is not None or attention.add_zero_attn:
        raise ValueError("unsupported teacher attention bias/zero-attention option")
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query, key, and value must be batched rank-three tensors")
    batch_size, target_length, embed_dim = query.shape
    source_length = key.shape[1]
    if key.shape != value.shape or key.shape[0] != batch_size or key.shape[2] != embed_dim:
        raise ValueError("teacher key/value shapes do not match query")
    if bias.shape != (batch_size, target_length, source_length):
        raise ValueError("shared logit bias must have shape [B,C,N]")
    num_heads = attention.num_heads
    if embed_dim != attention.embed_dim or embed_dim % num_heads:
        raise ValueError("teacher attention embed/head dimensions are inconsistent")
    head_dim = embed_dim // num_heads

    # MultiheadAttention.forward transposes batch-first inputs before calling
    # multi_head_attention_forward.  The following operations match its
    # need_weights=True/no-mask branch exactly through the output projection.
    transposed_query = query.transpose(0, 1)
    transposed_key = key.transpose(0, 1)
    # Preserve ``key is value`` exactly as MultiheadAttention.forward does;
    # _in_projection_packed then uses the same combined K/V projection.
    transposed_value = transposed_key if key is value else value.transpose(0, 1)
    q, k, v = F._in_projection_packed(  # type: ignore[attr-defined]
        transposed_query,
        transposed_key,
        transposed_value,
        attention.in_proj_weight,
        attention.in_proj_bias,
    )
    q = q.view(target_length, batch_size * num_heads, head_dim).transpose(0, 1)
    k = k.view(source_length, batch_size * num_heads, head_dim).transpose(0, 1)
    v = v.view(source_length, batch_size * num_heads, head_dim).transpose(0, 1)
    q_scaled = q * math.sqrt(1.0 / float(head_dim))
    scores = torch.bmm(q_scaled, k.transpose(-2, -1))
    expanded_bias = (
        bias[:, None, :, :]
        .expand(batch_size, num_heads, target_length, source_length)
        .reshape(batch_size * num_heads, target_length, source_length)
    )
    scores = scores + expanded_bias
    weights = F.softmax(scores, dim=-1)
    if attention.dropout > 0.0:
        weights = F.dropout(weights, p=attention.dropout, training=attention.training)
    output = torch.bmm(weights, v)
    output = output.transpose(0, 1).contiguous().view(target_length * batch_size, embed_dim)
    output = F.linear(output, attention.out_proj.weight, attention.out_proj.bias)
    return output.view(target_length, batch_size, embed_dim).transpose(0, 1)


class CoupledT4LogitResidualStreamingSpint(StreamingSpintModel):
    """Selected coupled T4 plus a factorized residual interaction."""

    def __init__(
        self,
        *,
        decoder: SpintModel,
        id_encoder: CalibrationEncoder,
        residual_mode: Literal["aligned", "shuffled"],
        interaction_mode: Literal["attention_logit", "additive_control"],
        residual_rank: int = 8,
        residual_permutation_seed: int | None = None,
    ) -> None:
        if residual_mode not in {"aligned", "shuffled"}:
            raise ValueError("residual_mode must be 'aligned' or 'shuffled'")
        if interaction_mode not in {"attention_logit", "additive_control"}:
            raise ValueError("unsupported interaction_mode")
        if residual_mode == "shuffled" and residual_permutation_seed is None:
            raise ValueError("shuffled residual requires a permutation seed")
        if residual_mode == "aligned" and residual_permutation_seed is not None:
            raise ValueError("aligned residual forbids a permutation seed")
        if decoder.num_layers != 1:
            raise ValueError("T4 logit residual currently requires one decoder layer")
        super().__init__(
            decoder=decoder,
            id_encoder=id_encoder,
            fixed_slot_count=0,
            decoder_mode="coupled",
        )
        self.decoder_mode = "coupled_t4_logit_residual"
        self.residual_mode = residual_mode
        self.interaction_mode = interaction_mode
        self.residual_permutation_seed = residual_permutation_seed
        self.t4_logit_residual = ZeroInitializedT4LogitResidual(
            t4_dim=4,
            rank=residual_rank,
            num_queries=decoder.num_covariates,
        )
        self._backbone_frozen_for_residual_pilot = False

    def _residual_t4(self, aligned_t4: torch.Tensor) -> torch.Tensor:
        if aligned_t4.ndim != 3 or aligned_t4.shape[-1] != 4:
            raise ValueError("aligned T4 must have shape [B,N,4]")
        if self.residual_mode == "aligned":
            return aligned_t4
        assert self.residual_permutation_seed is not None
        order = np.random.RandomState(self.residual_permutation_seed).permutation(aligned_t4.shape[1])
        index = torch.as_tensor(order, device=aligned_t4.device)
        return aligned_t4.index_select(1, index)

    def derive_t4_logit_residual_state(self, aligned_t4: torch.Tensor) -> T4LogitResidualState:
        residual_t4 = self._residual_t4(aligned_t4)
        return T4LogitResidualState(self.t4_logit_residual.derive_unit_factors(residual_t4))

    def _expanded_state(
        self,
        state: T4LogitResidualState,
        *,
        batch_size: int,
        num_units: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(state, T4LogitResidualState):
            raise TypeError("state must be T4LogitResidualState")
        factors = state.unit_factors
        if factors.shape[0] == 1 and batch_size > 1:
            factors = factors.expand(batch_size, -1, -1)
        expected = (batch_size, num_units, self.t4_logit_residual.rank)
        if factors.shape != expected:
            raise ValueError(f"residual state shape {tuple(factors.shape)} does not match {expected}")
        return factors.to(reference)

    def decode_with_t4_logit_residual_state(
        self,
        neural: torch.Tensor,
        identity: torch.Tensor,
        state: T4LogitResidualState,
    ) -> torch.Tensor:
        if neural.ndim != 3 or neural.shape[1] != self.window_size:
            raise ValueError(f"neural must have shape [B,{self.window_size},N]")
        batch_size, _, num_units = neural.shape
        identity = self._expanded_identity(identity, batch_size, num_units)
        source = neural.permute(0, 2, 1) + identity
        source = self._apply_decoder_neuron_dropout(source)
        hidden_source = self.decoder.fc_in(source)
        query = self.decoder.fc_in(self.decoder.rep).to(hidden_source).repeat(batch_size, 1, 1)
        factors = self._expanded_state(
            state,
            batch_size=batch_size,
            num_units=num_units,
            reference=hidden_source,
        )
        bias = self.t4_logit_residual.logit_bias(factors)

        layer = self.decoder.transformer.layers[0]
        normalized_query = layer.norm1(query)
        normalized_source = layer.norm1(hidden_source)
        if self.interaction_mode == "attention_logit":
            attention_output = _teacher_multihead_attention_with_shared_logit_bias(
                layer.cross_attn,
                normalized_query,
                normalized_source,
                normalized_source,
                bias,
            )
        else:
            attention_output, _ = layer.cross_attn(
                normalized_query,
                normalized_source,
                normalized_source,
            )
            # Same factors and parameters, but no unit-relative attention
            # interaction.  A per-query scalar is added uniformly in hidden
            # space after attention.
            attention_output = attention_output + bias.mean(dim=-1, keepdim=True)
        output = query + layer.dropout(attention_output)
        output = output + layer.dropout(layer.ffn(layer.norm2(output)))
        return self.decoder.fc_out(output).permute(0, 2, 1)

    def decode_with_t4_logit_residual(
        self,
        neural: torch.Tensor,
        identity: torch.Tensor,
        aligned_t4: torch.Tensor,
    ) -> torch.Tensor:
        state = self.derive_t4_logit_residual_state(aligned_t4)
        return self.decode_with_t4_logit_residual_state(neural, identity, state)

    def forward(
        self,
        neural: torch.Tensor,
        calib_trials: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        side_features: Optional[torch.Tensor] = None,
        decoder_key_features: Optional[torch.Tensor] = None,
        electrode_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if decoder_key_features is not None:
            raise ValueError("logit-residual adapter owns its direct T4 routing")
        if side_features is None:
            raise ValueError("logit-residual adapter requires aligned T4 side features")
        neuron_gate = None
        if identity is None:
            if calib_trials is None:
                raise ValueError("calib_trials or identity is required")
            if hasattr(self.id_encoder, "forward_batch_with_gate"):
                identity, neuron_gate = self.id_encoder.forward_batch_with_gate(
                    calib_trials, side_features=side_features
                )
            else:
                identity = self.compute_identity(
                    calib_trials,
                    side_features=side_features,
                    electrode_ids=electrode_ids,
                )
        if neuron_gate is not None:
            raise ValueError("T4 logit residual does not support encoder neuron gates")
        behavior = self.decode_with_t4_logit_residual(neural, identity, side_features)
        return behavior, identity

    def freeze_backbone_for_residual_pilot(self) -> int:
        frozen = 0
        for module in (self.decoder, self.id_encoder):
            for parameter in module.parameters():
                if parameter.requires_grad:
                    parameter.requires_grad = False
                    frozen += parameter.numel()
            module.eval()
        for parameter in self.t4_logit_residual.parameters():
            parameter.requires_grad = True
        self._backbone_frozen_for_residual_pilot = True
        return frozen

    @property
    def logit_residual_receipt(self) -> dict[str, object]:
        receipt = self.t4_logit_residual.initialization_receipt
        receipt.update(
            {
                "residual_mode": self.residual_mode,
                "interaction_mode": self.interaction_mode,
                "residual_permutation_seed": self.residual_permutation_seed,
                "encoder_receives_aligned_t4": True,
                "teacher_coupled_activity_identity_readin_preserved": True,
                "teacher_query_key_value_and_output_projections_preserved": True,
                "teacher_head_count": self.decoder.transformer.layers[0].cross_attn.num_heads,
                "bias_shared_across_teacher_heads": True,
                "backbone_frozen_for_residual_pilot": self._backbone_frozen_for_residual_pilot,
            }
        )
        return receipt

    def residual_cost_receipt(self, *, batch_size: int = 1, num_units: int = 64) -> dict[str, object]:
        if batch_size <= 0 or num_units <= 0:
            raise ValueError("batch_size and num_units must be positive")
        coupled = super().decoder_cost_comparison_receipt(
            batch_size=batch_size, num_neurons=num_units
        )["coupled"]
        rank = self.t4_logit_residual.rank
        queries = self.decoder.num_covariates
        heads = self.decoder.transformer.layers[0].cross_attn.num_heads
        state_elements = num_units * rank
        return {
            "schema_version": 1,
            "reference_shape": {
                "batch_size": batch_size,
                "num_units": num_units,
                "num_queries": queries,
                "teacher_heads": heads,
                "residual_rank": rank,
                "t4_dim": 4,
            },
            "coupled_reference": coupled,
            "calibration_only_unit_factor_macs": num_units * 4 * rank,
            "online_increment": {
                "query_unit_factor_macs": batch_size * queries * num_units * rank,
                "shared_bias_additions_across_heads": batch_size * heads * queries * num_units,
                "additional_teacher_width_linear_macs": 0,
            },
            "persistent_additional_state": {
                "elements": state_elements,
                "bytes_fp32": state_elements * 4,
                "bytes_fp16": state_elements * 2,
                "bytes_int8_without_quantization_metadata": state_elements,
            },
            "trainable_parameter_count": 4 * rank + queries * rank,
            "state_and_compute_linear_in_num_units": True,
            "neuron_axis_quadratic_term": False,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if self._backbone_frozen_for_residual_pilot:
            self.decoder.eval()
            self.id_encoder.eval()
        return self
