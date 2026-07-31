"""Teacher-head-preserving decoupled K/V diagnostic.

This module is intentionally isolated from the active v1 and v2 runners.  It
preserves a teacher cross-attention layer's full Q/K/V/out projections,
per-head softmaxes, norms and FFN, while allowing the projected key to be
derived from a session-static identity tensor and cached independently of the
online activity value.

``exact`` in this module means exact preservation of the teacher attention
topology and tensors when key and value inputs are identical.  It does not
claim equivalence between ``fc_in(x + E)`` and the diagnostic's separate
``fc_in(E)`` key plus ``fc_in(x)`` value paths.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _tensor_state_sha256(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class HeadPreservingKVState:
    """The projected static key and no other calibration artifact."""

    projected_key: torch.Tensor

    def __post_init__(self) -> None:
        if self.projected_key.ndim != 3:
            raise ValueError(
                "projected_key must have shape [B,N,D], got "
                f"{tuple(self.projected_key.shape)}"
            )

    @property
    def nbytes(self) -> int:
        return self.projected_key.numel() * self.projected_key.element_size()


class TeacherHeadPreservingDecoupledCrossAttention(nn.Module):
    """Full-width multi-head attention with a calibration-time K cache."""

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        residual_dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        for name, value in {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if not 0.0 <= residual_dropout < 1.0:
            raise ValueError("residual_dropout must be in [0,1)")
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0,1)")

        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        self.dim_feedforward = int(dim_feedforward)
        self.attention_dropout_probability = float(attention_dropout)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(residual_dropout),
            nn.Linear(self.dim_feedforward, self.d_model),
        )
        self.dropout = nn.Dropout(residual_dropout)
        self._initialization_receipt: dict[str, Any] | None = None

    @classmethod
    def from_teacher(
        cls, teacher_layer: nn.Module
    ) -> "TeacherHeadPreservingDecoupledCrossAttention":
        """Construct and initialize from one SPINT ``CrossAttentionLayer``."""
        attention = getattr(teacher_layer, "cross_attn", None)
        norm1 = getattr(teacher_layer, "norm1", None)
        norm2 = getattr(teacher_layer, "norm2", None)
        ffn = getattr(teacher_layer, "ffn", None)
        residual_dropout = getattr(teacher_layer, "dropout", None)
        if not isinstance(attention, nn.MultiheadAttention):
            raise TypeError("teacher_layer.cross_attn must be MultiheadAttention")
        if not isinstance(norm1, nn.LayerNorm) or not isinstance(
            norm2, nn.LayerNorm
        ):
            raise TypeError("teacher layer must expose norm1 and norm2")
        if not isinstance(ffn, nn.Sequential) or len(ffn) != 4:
            raise TypeError("teacher layer must expose the four-module SPINT FFN")
        if not isinstance(ffn[0], nn.Linear) or not isinstance(
            ffn[3], nn.Linear
        ):
            raise TypeError("teacher FFN endpoints must be Linear modules")
        if not isinstance(residual_dropout, nn.Dropout):
            raise TypeError("teacher layer must expose residual dropout")

        module = cls(
            d_model=attention.embed_dim,
            nhead=attention.num_heads,
            dim_feedforward=ffn[0].out_features,
            residual_dropout=float(residual_dropout.p),
            attention_dropout=float(attention.dropout),
        )
        module.initialize_from_teacher(teacher_layer)
        return module

    @torch.no_grad()
    def initialize_from_teacher(
        self, teacher_layer: nn.Module
    ) -> dict[str, Any]:
        """Copy the complete teacher layer without SVD or head averaging."""
        attention = getattr(teacher_layer, "cross_attn", None)
        if not isinstance(attention, nn.MultiheadAttention):
            raise TypeError("teacher_layer.cross_attn must be MultiheadAttention")
        if attention.embed_dim != self.d_model:
            raise ValueError("teacher embed dimension does not match")
        if attention.num_heads != self.nhead:
            raise ValueError("teacher head count does not match")
        if not attention.batch_first:
            raise ValueError("teacher attention must use batch_first=True")
        if attention.in_proj_weight is None:
            raise ValueError("teacher must expose packed Q/K/V weights")
        if attention.kdim not in {None, self.d_model} or attention.vdim not in {
            None,
            self.d_model,
        }:
            raise ValueError("teacher must use a common Q/K/V embed dimension")
        if attention.bias_k is not None or attention.bias_v is not None:
            raise ValueError("teacher bias_k/bias_v tokens are unsupported")
        if attention.add_zero_attn:
            raise ValueError("teacher add_zero_attn is unsupported")
        if float(attention.dropout) != self.attention_dropout_probability:
            raise ValueError("teacher attention-dropout probability does not match")

        weight_q, weight_k, weight_v = attention.in_proj_weight.chunk(
            3, dim=0
        )
        if attention.in_proj_bias is None:
            bias_q = weight_q.new_zeros(self.d_model)
            bias_k = weight_k.new_zeros(self.d_model)
            bias_v = weight_v.new_zeros(self.d_model)
            teacher_has_in_bias = False
        else:
            bias_q, bias_k, bias_v = attention.in_proj_bias.chunk(3, dim=0)
            teacher_has_in_bias = True

        for target, weight, bias in (
            (self.q_proj, weight_q, bias_q),
            (self.k_proj, weight_k, bias_k),
            (self.v_proj, weight_v, bias_v),
        ):
            target.weight.copy_(weight)
            target.bias.copy_(bias)
        self.out_proj.load_state_dict(
            attention.out_proj.state_dict(), strict=True
        )
        self.norm1.load_state_dict(
            teacher_layer.norm1.state_dict(), strict=True
        )
        self.norm2.load_state_dict(
            teacher_layer.norm2.state_dict(), strict=True
        )
        self.ffn.load_state_dict(teacher_layer.ffn.state_dict(), strict=True)

        teacher_dropout = getattr(teacher_layer, "dropout", None)
        if not isinstance(teacher_dropout, nn.Dropout):
            raise TypeError("teacher layer must expose residual dropout")
        if float(teacher_dropout.p) != float(self.dropout.p):
            raise ValueError("teacher residual-dropout probability does not match")

        receipt = {
            "schema_version": 1,
            "strategy": "exact_teacher_head_projection_copy",
            "teacher_embed_dim": self.d_model,
            "teacher_head_count": self.nhead,
            "teacher_head_dim": self.head_dim,
            "student_softmax_head_count": self.nhead,
            "teacher_headwise_softmax_preserved": True,
            "low_rank_factorization_used": False,
            "head_averaging_used": False,
            "teacher_input_projection_bias_present": teacher_has_in_bias,
            "attention_dropout_probability": (
                self.attention_dropout_probability
            ),
            "residual_dropout_probability": float(self.dropout.p),
            "factor_sha256": self.factor_sha256(),
        }
        self._initialization_receipt = receipt
        return dict(receipt)

    @property
    def initialization_receipt(self) -> dict[str, Any]:
        if self._initialization_receipt is None:
            raise RuntimeError("module has not been initialized from a teacher")
        receipt = dict(self._initialization_receipt)
        receipt["initial_factor_sha256"] = receipt.pop("factor_sha256")
        receipt["active_factor_sha256"] = self.factor_sha256()
        return receipt

    def factor_sha256(self) -> str:
        return _tensor_state_sha256({
            "q_proj.weight": self.q_proj.weight,
            "q_proj.bias": self.q_proj.bias,
            "k_proj.weight": self.k_proj.weight,
            "k_proj.bias": self.k_proj.bias,
            "v_proj.weight": self.v_proj.weight,
            "v_proj.bias": self.v_proj.bias,
            "out_proj.weight": self.out_proj.weight,
            "out_proj.bias": self.out_proj.bias,
            "norm1.weight": self.norm1.weight,
            "norm1.bias": self.norm1.bias,
            "norm2.weight": self.norm2.weight,
            "norm2.bias": self.norm2.bias,
            **{
                f"ffn.{name}": tensor
                for name, tensor in self.ffn.state_dict().items()
            },
        })

    @staticmethod
    def _validate_hidden(
        tensor: torch.Tensor, name: str, width: int
    ) -> None:
        if tensor.ndim != 3 or tensor.shape[-1] != width:
            raise ValueError(
                f"{name} must have shape [B,tokens,{width}], got "
                f"{tuple(tensor.shape)}"
            )

    def derive_static_key(
        self, hidden_identity: torch.Tensor
    ) -> HeadPreservingKVState:
        """Calibration path: ``K = Wk(norm1(fc_in(E)))``."""
        self._validate_hidden(
            hidden_identity, "hidden_identity", self.d_model
        )
        return HeadPreservingKVState(
            self.k_proj(self.norm1(hidden_identity))
        )

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = tensor.shape
        return tensor.view(
            batch_size, token_count, self.nhead, self.head_dim
        ).transpose(1, 2)

    def forward_cached(
        self,
        query_hidden: torch.Tensor,
        state: HeadPreservingKVState,
        value_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Online path; returns output and unaveraged ``[B,H,C,N]`` weights."""
        if not isinstance(state, HeadPreservingKVState):
            raise TypeError("state must be HeadPreservingKVState")
        self._validate_hidden(query_hidden, "query_hidden", self.d_model)
        self._validate_hidden(value_hidden, "value_hidden", self.d_model)
        if query_hidden.shape[0] != value_hidden.shape[0]:
            raise ValueError("query and value batch sizes must match")
        projected_key = state.projected_key
        if projected_key.shape[-1] != self.d_model:
            raise ValueError("cached key width does not match d_model")
        if projected_key.shape[0] == 1 and query_hidden.shape[0] > 1:
            projected_key = projected_key.expand(
                query_hidden.shape[0], -1, -1
            )
        if projected_key.shape[0] != query_hidden.shape[0]:
            raise ValueError("cached key batch size does not match query")
        if projected_key.shape[1] != value_hidden.shape[1]:
            raise ValueError("cached key and value unit counts must match")
        if (
            query_hidden.device != value_hidden.device
            or query_hidden.device != projected_key.device
        ):
            raise ValueError("query, key and value must be on the same device")

        query = self._split_heads(
            self.q_proj(self.norm1(query_hidden))
        )
        key = self._split_heads(projected_key)
        value = self._split_heads(
            self.v_proj(self.norm1(value_hidden))
        )
        logits = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)
        attention = torch.softmax(logits, dim=-1)
        attention = F.dropout(
            attention,
            p=self.attention_dropout_probability,
            training=self.training,
        )
        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).contiguous().view(
            query_hidden.shape[0],
            query_hidden.shape[1],
            self.d_model,
        )
        cross_attention_output = self.out_proj(attended)
        output = query_hidden + self.dropout(cross_attention_output)
        output = output + self.dropout(self.ffn(self.norm2(output)))
        return output, attention

    def forward(
        self,
        query_hidden: torch.Tensor,
        key_hidden: torch.Tensor,
        value_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Non-cached reference path used by semantic-equivalence tests."""
        return self.forward_cached(
            query_hidden,
            self.derive_static_key(key_hidden),
            value_hidden,
        )

    def cache_receipt(
        self, state: HeadPreservingKVState
    ) -> dict[str, Any]:
        if not isinstance(state, HeadPreservingKVState):
            raise TypeError("state must be HeadPreservingKVState")
        batch_size, num_units, width = state.projected_key.shape
        if width != self.d_model:
            raise ValueError("state key width does not match d_model")
        return {
            "schema_version": 1,
            "persistent_tensors": [{
                "name": "projected_static_key",
                "shape": [batch_size, num_units, width],
                "bytes": state.nbytes,
            }],
            "cache_bytes": state.nbytes,
            "excludes": [
                "identity",
                "raw_calibration",
                "activity_values",
                "attention_scores",
            ],
        }

    def decoder_cost_receipt(
        self,
        *,
        batch_size: int,
        num_units: int,
        num_queries: int,
        window_size: int,
    ) -> dict[str, Any]:
        """Configured MAC/state receipt for the complete oracle decoder path."""
        for name, value in {
            "batch_size": batch_size,
            "num_units": num_units,
            "num_queries": num_queries,
            "window_size": window_size,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        b = batch_size
        n = num_units
        c = num_queries
        w = window_size
        d = self.d_model
        f = self.dim_feedforward

        source_readin = b * n * (w * d + d * d)
        query_readin = b * c * (w * d + d * d)
        query_projection = b * c * d * d
        value_projection = b * n * d * d
        qk_scores = b * c * n * d
        weighted_values = qk_scores
        output_projection = b * c * d * d
        ffn = b * c * 2 * d * f
        output_readout = b * c * d * w
        online_total = (
            source_readin
            + query_readin
            + query_projection
            + value_projection
            + qk_scores
            + weighted_values
            + output_projection
            + ffn
            + output_readout
        )
        key_projection = b * n * d * d
        calibration_total = source_readin + key_projection
        cache_elements = b * n * d
        return {
            "schema_version": 1,
            "reference_shape": {
                "batch_size": b,
                "num_units": n,
                "num_queries": c,
                "window_size": w,
                "model_dim": d,
                "head_count": self.nhead,
                "head_dim": self.head_dim,
                "feedforward_dim": f,
            },
            "online_macs_per_window": {
                "activity_readin": source_readin,
                "query_readin": query_readin,
                "query_projection": query_projection,
                "value_projection": value_projection,
                "qk_scores": qk_scores,
                "weighted_values": weighted_values,
                "attention_output_projection": output_projection,
                "ffn": ffn,
                "output_readout": output_readout,
                "total": online_total,
                "no_unit_quadratic_term": True,
            },
            "calibration_only_macs": {
                "identity_readin": source_readin,
                "static_key_projection": key_projection,
                "total": calibration_total,
            },
            "persistent_state": {
                "elements": cache_elements,
                "bytes_fp32": cache_elements * 4,
                "bytes_fp16": cache_elements * 2,
                "bytes_int8_without_quantization_metadata": cache_elements,
            },
            "counted_operations": (
                "Linear, attention matmul, FFN and readin/readout MACs"
            ),
            "excluded_operations": [
                "elementwise_add",
                "activation",
                "normalization",
                "softmax",
                "dropout",
                "reshape",
            ],
        }
