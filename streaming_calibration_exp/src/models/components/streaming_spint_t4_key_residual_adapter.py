"""Baseline-preserving static T4 key residual for coupled SPINT.

The adapter keeps the successful coupled path intact:

``src = fc_in(activity + E)``

and adds a calibration-static, zero-initialized residual to the *normalized
key input only*. Query, value, teacher attention heads, output projection,
norms, FFN and readout are unchanged. At initialization the residual is exactly
zero, so eval-mode output is bitwise identical to the coupled substrate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn

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
class T4KeyResidualState:
    """Cached full-width residual added to the teacher key input."""

    hidden_key_residual: torch.Tensor

    def __post_init__(self) -> None:
        if self.hidden_key_residual.ndim != 3:
            raise ValueError(
                "hidden_key_residual must have shape [B,N,D]"
            )

    @property
    def nbytes(self) -> int:
        value = self.hidden_key_residual
        return value.numel() * value.element_size()


class ZeroInitializedT4KeyResidual(nn.Module):
    """Small low-rank T4 map with an exactly zero initial output."""

    def __init__(
        self,
        *,
        t4_dim: int,
        rank: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if t4_dim <= 0 or rank <= 0 or hidden_dim <= 0:
            raise ValueError(
                "t4_dim, rank and hidden_dim must be positive"
            )
        self.t4_dim = int(t4_dim)
        self.rank = int(rank)
        self.hidden_dim = int(hidden_dim)
        self.input_projection = nn.Linear(
            self.t4_dim, self.rank, bias=False
        )
        self.activation = nn.Tanh()
        self.output_projection = nn.Linear(
            self.rank, self.hidden_dim, bias=False
        )
        nn.init.zeros_(self.output_projection.weight)
        self._initial_sha256 = self.factor_sha256()

    def forward(self, t4: torch.Tensor) -> torch.Tensor:
        if t4.ndim != 3 or t4.shape[-1] != self.t4_dim:
            raise ValueError(
                f"T4 must have shape [B,N,{self.t4_dim}]"
            )
        return self.output_projection(
            self.activation(self.input_projection(t4))
        )

    def factor_sha256(self) -> str:
        return _tensor_state_sha256({
            name: tensor
            for name, tensor in self.state_dict().items()
        })

    @property
    def initialization_receipt(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "family": "zero_initialized_low_rank_t4_key_residual",
            "t4_dim": self.t4_dim,
            "rank": self.rank,
            "hidden_dim": self.hidden_dim,
            "initial_factor_sha256": self._initial_sha256,
            "active_factor_sha256": self.factor_sha256(),
            "output_projection_zero_initialized": True,
            "bias_parameters": 0,
        }

    def calibration_macs(self, num_units: int) -> int:
        if num_units <= 0:
            raise ValueError("num_units must be positive")
        return num_units * (
            self.t4_dim * self.rank
            + self.rank * self.hidden_dim
        )


class CoupledT4KeyResidualStreamingSpint(StreamingSpintModel):
    """Coupled SPINT with a cached direct-T4 key-only residual."""

    def __init__(
        self,
        *,
        decoder: SpintModel,
        id_encoder: CalibrationEncoder,
        residual_mode: Literal["aligned", "shuffled"],
        residual_rank: int = 8,
        residual_permutation_seed: int | None = None,
    ) -> None:
        if residual_mode not in {"aligned", "shuffled"}:
            raise ValueError(
                "residual_mode must be 'aligned' or 'shuffled'"
            )
        if (
            residual_mode == "shuffled"
            and residual_permutation_seed is None
        ):
            raise ValueError(
                "shuffled residual requires a permutation seed"
            )
        if (
            residual_mode == "aligned"
            and residual_permutation_seed is not None
        ):
            raise ValueError(
                "aligned residual forbids a permutation seed"
            )
        if decoder.num_layers != 1:
            raise ValueError(
                "T4 key residual currently requires one decoder layer"
            )
        super().__init__(
            decoder=decoder,
            id_encoder=id_encoder,
            fixed_slot_count=0,
            decoder_mode="coupled",
        )
        self.decoder_mode = "coupled_t4_key_residual"
        self.residual_mode = residual_mode
        self.residual_permutation_seed = residual_permutation_seed
        self.t4_key_residual = ZeroInitializedT4KeyResidual(
            t4_dim=4,
            rank=residual_rank,
            hidden_dim=decoder.model_dim,
        )
        self._backbone_frozen_for_residual_pilot = False

    def _residual_t4(self, aligned_t4: torch.Tensor) -> torch.Tensor:
        if (
            aligned_t4.ndim != 3
            or aligned_t4.shape[-1] != 4
        ):
            raise ValueError("aligned T4 must have shape [B,N,4]")
        if self.residual_mode == "aligned":
            return aligned_t4
        assert self.residual_permutation_seed is not None
        order = np.random.RandomState(
            self.residual_permutation_seed
        ).permutation(aligned_t4.shape[1])
        index = torch.as_tensor(order, device=aligned_t4.device)
        return aligned_t4.index_select(1, index)

    def derive_t4_key_residual_state(
        self, aligned_t4: torch.Tensor
    ) -> T4KeyResidualState:
        """Calibration-only T4→Δkey path."""
        return T4KeyResidualState(
            self.t4_key_residual(self._residual_t4(aligned_t4))
        )

    def _expanded_residual_state(
        self,
        state: T4KeyResidualState,
        *,
        batch_size: int,
        num_units: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(state, T4KeyResidualState):
            raise TypeError("state must be T4KeyResidualState")
        residual = state.hidden_key_residual
        if residual.shape[0] == 1 and batch_size > 1:
            residual = residual.expand(batch_size, -1, -1)
        if residual.shape != (
            batch_size,
            num_units,
            self.decoder.model_dim,
        ):
            raise ValueError(
                "residual state shape does not match online input"
            )
        return residual.to(reference)

    def decode_with_t4_key_residual_state(
        self,
        neural: torch.Tensor,
        identity: torch.Tensor,
        state: T4KeyResidualState,
    ) -> torch.Tensor:
        """Online coupled decode with a cached key-only residual."""
        if neural.ndim != 3 or neural.shape[1] != self.window_size:
            raise ValueError(
                f"neural must have shape [B,{self.window_size},N]"
            )
        batch_size, _, num_units = neural.shape
        identity = self._expanded_identity(
            identity, batch_size, num_units
        )
        source = neural.permute(0, 2, 1) + identity
        source = self._apply_decoder_neuron_dropout(source)
        hidden_source = self.decoder.fc_in(source)
        query = self.decoder.fc_in(self.decoder.rep).to(hidden_source)
        query = query.repeat(batch_size, 1, 1)
        residual = self._expanded_residual_state(
            state,
            batch_size=batch_size,
            num_units=num_units,
            reference=hidden_source,
        )

        layer = self.decoder.transformer.layers[0]
        normalized_query = layer.norm1(query)
        normalized_source = layer.norm1(hidden_source)
        attention_output, _ = layer.cross_attn(
            normalized_query,
            normalized_source + residual,
            normalized_source,
        )
        output = query + layer.dropout(attention_output)
        output = output + layer.dropout(
            layer.ffn(layer.norm2(output))
        )
        return self.decoder.fc_out(output).permute(0, 2, 1)

    def decode_with_t4_key_residual(
        self,
        neural: torch.Tensor,
        identity: torch.Tensor,
        aligned_t4: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable train/reference path."""
        state = self.derive_t4_key_residual_state(aligned_t4)
        return self.decode_with_t4_key_residual_state(
            neural, identity, state
        )

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
            raise ValueError(
                "key-residual adapter owns its direct T4 routing"
            )
        if side_features is None:
            raise ValueError(
                "key-residual adapter requires aligned T4 side features"
            )
        neuron_gate = None
        if identity is None:
            if calib_trials is None:
                raise ValueError(
                    "calib_trials or identity is required"
                )
            if hasattr(self.id_encoder, "forward_batch_with_gate"):
                identity, neuron_gate = (
                    self.id_encoder.forward_batch_with_gate(
                        calib_trials,
                        side_features=side_features,
                    )
                )
            else:
                identity = self.compute_identity(
                    calib_trials,
                    side_features=side_features,
                    electrode_ids=electrode_ids,
                )
        if neuron_gate is not None:
            raise ValueError(
                "T4 key residual does not support encoder neuron gates"
            )
        behavior = self.decode_with_t4_key_residual(
            neural, identity, side_features
        )
        return behavior, identity

    def freeze_backbone_for_residual_pilot(self) -> int:
        """Train only the new residual factors in the first causal screen."""
        frozen = 0
        for module in (self.decoder, self.id_encoder):
            for parameter in module.parameters():
                if parameter.requires_grad:
                    parameter.requires_grad = False
                    frozen += parameter.numel()
            module.eval()
        for parameter in self.t4_key_residual.parameters():
            parameter.requires_grad = True
        self._backbone_frozen_for_residual_pilot = True
        return frozen

    @property
    def key_residual_receipt(self) -> dict[str, object]:
        receipt = self.t4_key_residual.initialization_receipt
        receipt.update({
            "residual_mode": self.residual_mode,
            "residual_permutation_seed": (
                self.residual_permutation_seed
            ),
            "encoder_receives_aligned_t4": True,
            "teacher_coupled_activity_identity_readin_preserved": True,
            "teacher_query_path_preserved": True,
            "teacher_value_path_preserved": True,
            "teacher_head_count": (
                self.decoder.transformer.layers[0]
                .cross_attn.num_heads
            ),
            "backbone_frozen_for_residual_pilot": (
                self._backbone_frozen_for_residual_pilot
            ),
        })
        return receipt

    def residual_cost_receipt(
        self,
        *,
        batch_size: int = 1,
        num_units: int = 64,
    ) -> dict[str, object]:
        if batch_size <= 0 or num_units <= 0:
            raise ValueError(
                "batch_size and num_units must be positive"
            )
        coupled = super().decoder_cost_comparison_receipt(
            batch_size=batch_size,
            num_neurons=num_units,
        )["coupled"]
        calibration = self.t4_key_residual.calibration_macs(
            num_units
        )
        state_elements = (
            num_units * self.decoder.model_dim
        )
        return {
            "schema_version": 1,
            "reference_shape": {
                "batch_size": batch_size,
                "num_units": num_units,
                "window_size": self.window_size,
                "model_dim": self.decoder.model_dim,
                "residual_rank": self.t4_key_residual.rank,
                "t4_dim": 4,
            },
            "coupled_reference": coupled,
            "online_linear_attention_ffn_macs_per_window": (
                coupled["total"]
            ),
            "online_increment": {
                "elementwise_key_additions": (
                    batch_size * state_elements
                ),
                "additional_linear_macs": 0,
            },
            "calibration_only_residual_macs": calibration,
            "persistent_additional_state": {
                "elements": state_elements,
                "bytes_fp32": state_elements * 4,
                "bytes_fp16": state_elements * 2,
                "bytes_int8_without_quantization_metadata": (
                    state_elements
                ),
            },
            "identity_encoder_compute_included": False,
            "accuracy_candidate_not_efficiency_candidate": True,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if self._backbone_frozen_for_residual_pilot:
            self.decoder.eval()
            self.id_encoder.eval()
        return self
