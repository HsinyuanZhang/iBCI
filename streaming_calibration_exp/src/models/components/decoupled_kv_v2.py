"""Representation-preserving decoupled K/V attention primitives.

This module is intentionally not wired into :mod:`streaming_spint` yet.  The
fresh 32/32 Stage-0 screen imports the legacy implementation from ``spint.py``;
keeping this follow-up in a separate module prevents an in-flight sequential
screen from changing implementation between arms.

The follow-up operates in the teacher's hidden space:

``h_E = teacher_readin(E)``, ``h_x = teacher_readin(x)``.

Only a projected static key is retained after calibration.  The low-rank
Q/K and V/output weights can be initialized from truncated SVDs of a teacher
``nn.MultiheadAttention`` layer.  This factorizes proxies for the sum of the
teacher heads' pre-softmax bilinear forms and for the linear value/output
composition.  It does not approximate the teacher's separate head-wise
softmax distributions, their head-specific value mixing, or the complete
multi-head forward function.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch
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


def _relative_frobenius_residual(singular_values: torch.Tensor, rank: int) -> float:
    total = singular_values.square().sum()
    if float(total) == 0.0:
        return 0.0
    discarded = singular_values[rank:].square().sum()
    return float(torch.sqrt(discarded / total))


def _retained_energy(singular_values: torch.Tensor, rank: int) -> float:
    total = singular_values.square().sum()
    if float(total) == 0.0:
        return 1.0
    return float(singular_values[:rank].square().sum() / total)


@dataclass(frozen=True)
class HiddenDecoupledKVState:
    """The complete persistent state for a static-key deployment path."""

    projected_key: torch.Tensor

    def __post_init__(self) -> None:
        if self.projected_key.ndim != 3:
            raise ValueError(
                "projected_key must have shape [B,N,Dk], got "
                f"{tuple(self.projected_key.shape)}"
            )

    @property
    def nbytes(self) -> int:
        key = self.projected_key
        return key.numel() * key.element_size()


class TeacherSVDDecoupledCrossAttention(nn.Module):
    """Single-head hidden-space K/V factorization with a cached static key.

    ``key_dim`` and ``value_dim`` are deliberately independent.  There is one
    attention head so the global rank-``key_dim`` Q/K SVD factorization is not
    silently converted into a block-diagonal multi-head approximation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        key_dim: int = 48,
        value_dim: int = 64,
        direct_feature_dim: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        for name, value in {
            "d_model": d_model,
            "key_dim": key_dim,
            "value_dim": value_dim,
            "direct_feature_dim": direct_feature_dim,
            "dim_feedforward": dim_feedforward,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if key_dim > d_model or value_dim > d_model:
            raise ValueError("key_dim and value_dim must not exceed d_model")

        self.d_model = int(d_model)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.direct_feature_dim = int(direct_feature_dim)
        self.dim_feedforward = int(dim_feedforward)

        self.norm1 = nn.LayerNorm(d_model)
        self.query_proj = nn.Linear(d_model, key_dim, bias=True)
        self.key_proj = nn.Linear(d_model, key_dim, bias=False)
        self.direct_key_proj = nn.Linear(direct_feature_dim, key_dim, bias=False)
        self.value_proj = nn.Linear(d_model, value_dim, bias=False)
        self.out_proj = nn.Linear(value_dim, d_model, bias=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.direct_key_proj.weight)

    @staticmethod
    def _validate_hidden(tensor: torch.Tensor, name: str, width: int) -> None:
        if tensor.ndim != 3 or tensor.shape[-1] != width:
            raise ValueError(
                f"{name} must have shape [B,tokens,{width}], got "
                f"{tuple(tensor.shape)}"
            )

    @torch.no_grad()
    def initialize_from_teacher(
        self,
        *,
        teacher_attn: nn.MultiheadAttention,
        teacher_norm1: nn.LayerNorm,
        teacher_norm2: nn.LayerNorm,
        teacher_ffn: nn.Sequential,
    ) -> dict[str, Any]:
        """Initialize the low-rank maps and copied base from a teacher layer.

        Q/K factorize a rank-limited proxy for the sum of teacher heads'
        pre-softmax bilinear forms, ``Wq.T @ Wk / sqrt(teacher_head_dim)``.
        V/output factorize the linear composition ``Wo @ Wv``. Neither proxy
        preserves head-wise softmax. The key-dependent affine logit induced by
        teacher ``bq`` is least-squares projected into the low-rank query bias;
        ``bk`` produces only a per-query constant and is softmax-invariant.
        The teacher value bias contribution ``Wo @ bv`` is folded into the
        output bias. SVD is performed in float64 on CPU, then copied to this
        module's dtype and device.
        """
        if teacher_attn.embed_dim != self.d_model:
            raise ValueError(
                f"teacher embed_dim={teacher_attn.embed_dim} does not match "
                f"d_model={self.d_model}"
            )
        if teacher_attn.kdim not in {None, self.d_model}:
            raise ValueError("teacher must use the common Q/K/V embed dimension")
        if teacher_attn.vdim not in {None, self.d_model}:
            raise ValueError("teacher must use the common Q/K/V embed dimension")
        if teacher_attn.in_proj_weight is None:
            raise ValueError("teacher must expose a packed in_proj_weight")
        if self.d_model % teacher_attn.num_heads != 0:
            raise ValueError("teacher embed_dim must be divisible by num_heads")
        if teacher_attn.bias_k is not None or teacher_attn.bias_v is not None:
            raise ValueError("teacher bias_k/bias_v tokens are unsupported")
        if teacher_attn.add_zero_attn:
            raise ValueError("teacher add_zero_attn is unsupported")

        teacher_head_dim = self.d_model // teacher_attn.num_heads
        packed = teacher_attn.in_proj_weight.detach().to(
            device="cpu", dtype=torch.float64
        )
        wq, wk, wv = packed.chunk(3, dim=0)
        wo = teacher_attn.out_proj.weight.detach().to(
            device="cpu", dtype=torch.float64
        )

        # New scores divide by sqrt(Dk).  Scale the SVD target so the resulting
        # factors approximate the teacher's aggregate /sqrt(head_dim) bilinear.
        qk_target = math.sqrt(self.key_dim / teacher_head_dim) * (wq.T @ wk)
        uq, sq, vhq = torch.linalg.svd(qk_target, full_matrices=False)
        sqrt_sq = torch.sqrt(sq[: self.key_dim])
        wq_low = sqrt_sq[:, None] * uq[:, : self.key_dim].T
        wk_low = sqrt_sq[:, None] * vhq[: self.key_dim, :]

        vo_target = wo @ wv
        uv, sv, vhv = torch.linalg.svd(vo_target, full_matrices=False)
        sqrt_sv = torch.sqrt(sv[: self.value_dim])
        wout_low = uv[:, : self.value_dim] * sqrt_sv[None, :]
        wv_low = sqrt_sv[:, None] * vhv[: self.value_dim, :]

        in_bias = teacher_attn.in_proj_bias
        if in_bias is None:
            bq = torch.zeros(self.d_model, dtype=torch.float64)
            bk = torch.zeros(self.d_model, dtype=torch.float64)
            bv = torch.zeros(self.d_model, dtype=torch.float64)
        else:
            bq, bk, bv = (
                part.detach().to(device="cpu", dtype=torch.float64)
                for part in in_bias.chunk(3, dim=0)
            )
        # In the aggregate teacher-logit proxy, bq creates the only
        # key-dependent affine term. Solve B.T @ a ~= sqrt(Dk/dh) Wk.T @ bq.
        query_bias_target = (
            math.sqrt(self.key_dim / teacher_head_dim) * (wk.T @ bq)
        )
        query_bias_low = torch.linalg.lstsq(
            wk_low.T, query_bias_target[:, None]
        ).solution[:, 0]
        query_bias_reconstruction = wk_low.T @ query_bias_low
        query_bias_target_norm = torch.linalg.vector_norm(query_bias_target)
        query_bias_residual = torch.linalg.vector_norm(
            query_bias_reconstruction - query_bias_target
        )
        query_bias_relative_residual = (
            0.0
            if float(query_bias_target_norm) == 0.0
            else float(query_bias_residual / query_bias_target_norm)
        )

        self.query_proj.weight.copy_(
            wq_low.to(self.query_proj.weight)
        )
        self.query_proj.bias.copy_(
            query_bias_low.to(self.query_proj.bias)
        )
        self.key_proj.weight.copy_(
            wk_low.to(self.key_proj.weight)
        )
        self.value_proj.weight.copy_(
            wv_low.to(self.value_proj.weight)
        )
        self.out_proj.weight.copy_(
            wout_low.to(self.out_proj.weight)
        )
        teacher_out_bias = (
            torch.zeros(self.d_model, dtype=torch.float64)
            if teacher_attn.out_proj.bias is None
            else teacher_attn.out_proj.bias.detach().to(
                device="cpu", dtype=torch.float64
            )
        )
        folded_out_bias = teacher_out_bias + wo @ bv
        self.out_proj.bias.copy_(
            folded_out_bias.to(self.out_proj.bias)
        )
        self.direct_key_proj.weight.zero_()
        self.norm1.load_state_dict(teacher_norm1.state_dict(), strict=True)
        self.norm2.load_state_dict(teacher_norm2.state_dict(), strict=True)
        self.ffn.load_state_dict(teacher_ffn.state_dict(), strict=True)

        bq_norm = float(torch.linalg.vector_norm(bq))
        bk_norm = float(torch.linalg.vector_norm(bk))
        bv_norm = float(torch.linalg.vector_norm(bv))
        factors = {
            "query_proj.weight": self.query_proj.weight,
            "query_proj.bias": self.query_proj.bias,
            "key_proj.weight": self.key_proj.weight,
            "value_proj.weight": self.value_proj.weight,
            "out_proj.weight": self.out_proj.weight,
            "out_proj.bias": self.out_proj.bias,
            "direct_key_proj.weight": self.direct_key_proj.weight,
        }
        return {
            "schema_version": 1,
            "strategy": "teacher_affine_proxy_global_bilinear_svd",
            "bias_policy": (
                "bq_lstsq_bk_softmax_invariant_bv_folded_into_output"
            ),
            "teacher_embed_dim": self.d_model,
            "teacher_head_count": teacher_attn.num_heads,
            "teacher_head_dim": teacher_head_dim,
            "teacher_softmax_head_count": teacher_attn.num_heads,
            "low_rank_attention_heads": 1,
            "student_softmax_head_count": 1,
            "key_rank": self.key_dim,
            "value_rank": self.value_dim,
            "qk_object": "sum_of_teacher_pre_softmax_bilinear_forms",
            "vo_object": "linear_value_output_composition_only",
            "teacher_headwise_softmax_preserved": False,
            "qk_relative_frobenius_residual": _relative_frobenius_residual(
                sq, self.key_dim
            ),
            "qk_retained_energy": _retained_energy(sq, self.key_dim),
            "vo_relative_frobenius_residual": _relative_frobenius_residual(
                sv, self.value_dim
            ),
            "vo_retained_energy": _retained_energy(sv, self.value_dim),
            "teacher_q_bias_l2": bq_norm,
            "teacher_k_bias_l2_softmax_invariant": bk_norm,
            "teacher_v_bias_l2_folded_into_output": bv_norm,
            "query_bias_relative_lstsq_residual": query_bias_relative_residual,
            "construction_residual_dtype": "float64",
            "teacher_out_proj_bias_copied": (
                teacher_attn.out_proj.bias is not None
            ),
            "teacher_value_bias_folded_into_output_eval_exact": True,
            "teacher_value_bias_fold_exactness": (
                "eval_only_attention_dropout_disabled"
            ),
            "teacher_attention_dropout_probability": float(
                teacher_attn.dropout
            ),
            "teacher_training_attention_dropout_preserved": False,
            "direct_key_branch_zero_initialized": bool(
                torch.count_nonzero(self.direct_key_proj.weight) == 0
            ),
            "exact_teacher_mha_equivalence": False,
            "factor_sha256": _tensor_state_sha256(factors),
        }

    def derive_static_key(
        self,
        hidden_identity: torch.Tensor,
        direct_features: torch.Tensor,
    ) -> HiddenDecoupledKVState:
        """Project and return the only persistent calibration state."""
        self._validate_hidden(
            hidden_identity, "hidden_identity", self.d_model
        )
        self._validate_hidden(
            direct_features, "direct_features", self.direct_feature_dim
        )
        if hidden_identity.shape[:2] != direct_features.shape[:2]:
            raise ValueError(
                "hidden_identity and direct_features must share [B,N]"
            )
        if hidden_identity.device != direct_features.device:
            raise ValueError(
                "hidden_identity and direct_features must share device"
            )
        if hidden_identity.dtype != direct_features.dtype:
            raise ValueError(
                "hidden_identity and direct_features must share dtype"
            )
        projected_key = (
            self.key_proj(self.norm1(hidden_identity))
            + self.direct_key_proj(direct_features)
        )
        return HiddenDecoupledKVState(projected_key)

    def _forward_projected(
        self,
        query: torch.Tensor,
        projected_key: torch.Tensor,
        hidden_activity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Common attention path after a static or dynamic key projection."""
        self._validate_hidden(query, "query", self.d_model)
        self._validate_hidden(
            projected_key, "projected_key", self.key_dim
        )
        self._validate_hidden(
            hidden_activity, "hidden_activity", self.d_model
        )
        if query.shape[0] != projected_key.shape[0]:
            raise ValueError("query and projected_key must share batch size")
        if hidden_activity.shape[:2] != projected_key.shape[:2]:
            raise ValueError(
                "hidden_activity and projected_key must share [B,N]"
            )
        if (
            query.device != projected_key.device
            or query.device != hidden_activity.device
        ):
            raise ValueError(
                "query, projected_key, and hidden_activity must share device"
            )
        if (
            query.dtype != projected_key.dtype
            or query.dtype != hidden_activity.dtype
        ):
            raise ValueError(
                "query, projected_key, and hidden_activity must share dtype"
            )

        q = self.query_proj(self.norm1(query))
        v = self.value_proj(self.norm1(hidden_activity))
        logits = torch.matmul(q, projected_key.transpose(-2, -1))
        attention = self.attn_dropout(
            torch.softmax(logits / math.sqrt(self.key_dim), dim=-1)
        )
        attended = torch.matmul(attention, v)
        x = query + self.dropout(self.out_proj(attended))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, attention

    def forward_cached(
        self,
        query: torch.Tensor,
        state: HiddenDecoupledKVState,
        hidden_activity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode with cached K and online hidden-space activity V."""
        if not isinstance(state, HiddenDecoupledKVState):
            raise TypeError("state must be HiddenDecoupledKVState")
        return self._forward_projected(
            query, state.projected_key, hidden_activity
        )

    def forward_dynamic_activity_key(
        self,
        query: torch.Tensor,
        hidden_activity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure ``K(h_x),V(h_x)`` control with no identity or static cache.

        The key projection runs once on every online forward.  The direct-T4
        projection is deliberately absent from this path.
        """
        self._validate_hidden(
            hidden_activity, "hidden_activity", self.d_model
        )
        projected_key = self.key_proj(self.norm1(hidden_activity))
        return self._forward_projected(
            query, projected_key, hidden_activity
        )

    def forward(
        self,
        query: torch.Tensor,
        hidden_identity: torch.Tensor,
        direct_features: torch.Tensor,
        hidden_activity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference non-cached path."""
        state = self.derive_static_key(hidden_identity, direct_features)
        return self.forward_cached(query, state, hidden_activity)

    def cache_receipt(self, state: HiddenDecoupledKVState) -> dict[str, Any]:
        if not isinstance(state, HiddenDecoupledKVState):
            raise TypeError("state must be HiddenDecoupledKVState")
        key = state.projected_key
        if key.shape[-1] != self.key_dim:
            raise ValueError("cached key width does not match module key_dim")
        return {
            "schema_version": 1,
            "persistent_tensors": [{
                "name": "projected_static_key",
                "shape": list(key.shape),
                "dtype": str(key.dtype),
                "bytes": state.nbytes,
            }],
            "cache_bytes": state.nbytes,
            "persistent_state_fields": ["projected_key"],
            "excludes": [
                "identity",
                "hidden_identity",
                "direct_key_features",
                "raw_calibration",
                "activity",
                "hidden_activity",
                "values",
                "attention_scores",
            ],
        }

    def cost_receipt(
        self,
        *,
        batch_size: int,
        num_units: int,
        num_queries: int,
        window_size: int,
        dynamic_activity_key: bool = False,
        element_size: int = 4,
    ) -> dict[str, Any]:
        """Return exact configured MAC and persistent-state accounting.

        The receipt includes the external two-layer teacher read-in for activity
        and queries.  Static-key calibration separately includes the identity
        read-in, hidden-key projection and direct-feature projection.
        """
        for name, value in {
            "batch_size": batch_size,
            "num_units": num_units,
            "num_queries": num_queries,
            "window_size": window_size,
            "element_size": element_size,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        b = batch_size
        n = num_units
        c = num_queries
        w = window_size
        d = self.d_model
        dk = self.key_dim
        dv = self.value_dim
        f = self.dim_feedforward

        source_readin = b * n * (w * d + d * d)
        query_readin = b * c * (w * d + d * d)
        query_projection = b * c * d * dk
        value_projection = b * n * d * dv
        qk_scores = b * c * n * dk
        weighted_values = b * c * n * dv
        output_projection = b * c * dv * d
        ffn = b * c * 2 * d * f
        output_readout = b * c * d * w
        dynamic_key_projection = b * n * d * dk if dynamic_activity_key else 0
        online_terms = {
            "activity_readin": source_readin,
            "query_readin": query_readin,
            "query_projection": query_projection,
            "value_projection": value_projection,
            "dynamic_key_projection": dynamic_key_projection,
            "qk_scores": qk_scores,
            "weighted_values": weighted_values,
            "attention_output_projection": output_projection,
            "ffn": ffn,
            "output_readout": output_readout,
        }

        identity_readin = 0 if dynamic_activity_key else source_readin
        static_hidden_key = 0 if dynamic_activity_key else b * n * d * dk
        static_direct_key = (
            0
            if dynamic_activity_key
            else b * n * self.direct_feature_dim * dk
        )
        calibration_terms = {
            "identity_readin": identity_readin,
            "hidden_key_projection": static_hidden_key,
            "direct_key_projection": static_direct_key,
        }
        persistent_width = 0 if dynamic_activity_key else dk
        return {
            "schema_version": 1,
            "reference_shape": {
                "batch_size": b,
                "num_units": n,
                "num_queries": c,
                "window_size": w,
                "model_dim": d,
                "feedforward_dim": f,
                "key_dim": dk,
                "value_dim": dv,
                "direct_feature_dim": self.direct_feature_dim,
            },
            "dynamic_activity_key": dynamic_activity_key,
            "key_source": (
                "hidden_activity"
                if dynamic_activity_key
                else "calibration_hidden_identity_plus_direct_feature"
            ),
            "online_macs_per_frame": {
                **online_terms,
                "total": sum(online_terms.values()),
                "no_unit_quadratic_term": True,
            },
            "calibration_only_macs": {
                **calibration_terms,
                "total": sum(calibration_terms.values()),
            },
            "persistent_state": {
                "projected_static_key_width": persistent_width,
                "bytes": b * n * persistent_width * element_size,
                "static_key_cache_applicable": not dynamic_activity_key,
                "cache_contract": (
                    "none" if dynamic_activity_key else "projected_static_key_only"
                ),
            },
            "counted_operations": "Linear, attention matmul, and FFN MACs",
            "excluded_operations": [
                "elementwise_add",
                "activation",
                "normalization",
                "softmax",
                "dropout",
            ],
        }
