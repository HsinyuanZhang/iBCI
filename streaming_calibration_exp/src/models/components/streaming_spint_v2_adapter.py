"""Isolated streaming adapter for the teacher-readin decoupled K/V follow-up.

The active Stage-0 screen does not import this module.  It exists so a
result-triggered v2 round can be wired into the training module with a small,
auditable selector change after every v1 arm has completed.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch

from .decoupled_kv_v2 import (
    HiddenDecoupledKVState,
    TeacherSVDDecoupledCrossAttention,
)
from .spint import SpintModel
from .streaming_encoders import CalibrationEncoder
from .streaming_spint import StreamingSpintModel


class TeacherReadinDecoupledStreamingSpint(StreamingSpintModel):
    """Streaming SPINT with hidden-space static K and online activity V."""

    def __init__(
        self,
        *,
        decoder: SpintModel,
        id_encoder: CalibrationEncoder,
        key_mode: Literal["e_t4", "e_ts4", "e_only", "x_only"],
        key_dim: int = 48,
        value_dim: int = 64,
        direct_feature_dim: int = 4,
    ) -> None:
        if key_mode not in {"e_t4", "e_ts4", "e_only", "x_only"}:
            raise ValueError(
                "key_mode must be one of {'e_t4','e_ts4','e_only','x_only'}"
            )
        if decoder.num_layers != 1:
            raise ValueError("teacher-readin decoupled v2 requires one layer")

        # Build only the unchanged coupled substrate.  The v2 module below is
        # independent of the v1 CachedDecoupledMultiLayerCrossAttention.
        super().__init__(
            decoder=decoder,
            id_encoder=id_encoder,
            fixed_slot_count=0,
            decoder_mode="coupled",
        )
        self.decoder_mode = "teacher_readin_decoupled_v2"
        self.decoupled_key_mode = key_mode
        self.decoupled_direct_feature_dim = int(direct_feature_dim)
        legacy = decoder.transformer.layers[0]
        self.decoupled_v2 = TeacherSVDDecoupledCrossAttention(
            d_model=decoder.model_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            direct_feature_dim=direct_feature_dim,
            dim_feedforward=legacy.ffn[0].out_features,
            dropout=decoder.tf_drop_rate,
        )
        self._v2_initialization_receipt = (
            self.decoupled_v2.initialize_from_teacher(
                teacher_attn=legacy.cross_attn,
                teacher_norm1=legacy.norm1,
                teacher_norm2=legacy.norm2,
                teacher_ffn=legacy.ffn,
            )
        )
        # These teacher-decoder modules are not on the v2 forward path. Freeze
        # them now so an end-to-end optimizer does not allocate state for
        # parameters that can never receive a task gradient.
        for unused in (
            self.decoder.transformer,
            self.decoder.fc_id_in,
            self.decoder.fc_id_out,
        ):
            for parameter in unused.parameters():
                parameter.requires_grad = False
        self._v2_initialization_receipt[
            "unused_legacy_decoder_parameters_frozen"
        ] = True
        self._v2_initialization_receipt["identity_used_by_decoder"] = (
            key_mode != "x_only"
        )
        self._v2_initialization_receipt[
            "x_only_identity_encoder_role"
        ] = "framework_metrics_only"

    @property
    def v2_initialization_receipt(self) -> dict[str, object]:
        return dict(self._v2_initialization_receipt)

    def _expanded_direct_features(
        self,
        direct_features: torch.Tensor | None,
        *,
        batch_size: int,
        num_units: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if self.decoupled_key_mode == "e_only":
            if direct_features is not None:
                raise ValueError(
                    "e_only forbids supplied direct key features"
                )
            return reference.new_zeros(
                batch_size, num_units, self.decoupled_direct_feature_dim
            )
        if self.decoupled_key_mode == "x_only":
            raise ValueError(
                "x_only forbids direct key features and static key derivation"
            )
        if direct_features is None:
            raise ValueError(
                f"{self.decoupled_key_mode} requires direct key features"
            )
        if (
            direct_features.ndim != 3
            or direct_features.shape[-1] != self.decoupled_direct_feature_dim
        ):
            raise ValueError(
                "direct key features must have shape "
                f"[B,N,{self.decoupled_direct_feature_dim}]"
            )
        if direct_features.shape[0] == 1 and batch_size > 1:
            direct_features = direct_features.expand(batch_size, -1, -1)
        if direct_features.shape[:2] != (batch_size, num_units):
            raise ValueError(
                "direct key feature batch/unit dimensions must match identity"
            )
        return direct_features.to(reference)

    def derive_decoupled_kv_state(
        self,
        identity: torch.Tensor,
        decoder_key_features: torch.Tensor | None = None,
    ) -> HiddenDecoupledKVState:
        """Calibration-time ``E→h_E→K`` path."""
        if self.decoupled_key_mode == "x_only":
            raise RuntimeError("x_only has no static calibration state")
        if identity.ndim != 3 or identity.shape[-1] != self.window_size:
            raise ValueError(
                f"identity must have shape [B,N,{self.window_size}]"
            )
        batch_size, num_units = identity.shape[:2]
        direct = self._expanded_direct_features(
            decoder_key_features,
            batch_size=batch_size,
            num_units=num_units,
            reference=identity,
        )
        hidden_identity = self.decoder.fc_in(identity)
        return self.decoupled_v2.derive_static_key(
            hidden_identity, direct
        )

    def _query_and_hidden_activity(
        self, neural: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if neural.ndim != 3 or neural.shape[1] != self.window_size:
            raise ValueError(
                f"neural must have shape [B,{self.window_size},N]"
            )
        activity = self._apply_decoder_neuron_dropout(
            neural.permute(0, 2, 1)
        )
        hidden_activity = self.decoder.fc_in(activity)
        query = self.decoder.fc_in(self.decoder.rep).to(hidden_activity)
        return query.repeat(neural.shape[0], 1, 1), hidden_activity

    def decode_with_decoupled_kv_state(
        self,
        neural: torch.Tensor,
        state: HiddenDecoupledKVState,
    ) -> torch.Tensor:
        """Online cached-static-key decode; never recomputes E or K."""
        if self.decoupled_key_mode == "x_only":
            raise RuntimeError("x_only has no static calibration state")
        if self._decoder_frozen:
            self.decoder.eval()
            self.decoupled_v2.eval()
        query, hidden_activity = self._query_and_hidden_activity(neural)
        if state.projected_key.shape[0] == 1 and query.shape[0] > 1:
            state = HiddenDecoupledKVState(
                state.projected_key.expand(query.shape[0], -1, -1)
            )
        transformer_output, _ = self.decoupled_v2.forward_cached(
            query, state, hidden_activity
        )
        return self.decoder.fc_out(transformer_output).permute(0, 2, 1)

    def decode_with_decoupled_identity(
        self,
        neural: torch.Tensor,
        identity: torch.Tensor,
        decoder_key_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Differentiable train/reference path for static and x-only arms."""
        if self.decoupled_key_mode == "x_only":
            if decoder_key_features is not None:
                raise ValueError("x_only forbids supplied direct key features")
            return self.decode_x_only(neural)
        batch_size = neural.shape[0]
        num_units = neural.shape[-1]
        identity = self._expanded_identity(
            identity, batch_size, num_units
        )
        if self._decoder_frozen:
            self.decoder.eval()
            self.decoupled_v2.eval()
        query, hidden_activity = self._query_and_hidden_activity(neural)
        direct = self._expanded_direct_features(
            decoder_key_features,
            batch_size=batch_size,
            num_units=num_units,
            reference=identity,
        )
        hidden_identity = self.decoder.fc_in(identity)
        state = self.decoupled_v2.derive_static_key(
            hidden_identity, direct
        )
        transformer_output, _ = self.decoupled_v2.forward_cached(
            query, state, hidden_activity
        )
        return self.decoder.fc_out(transformer_output).permute(0, 2, 1)

    def decode_x_only(self, neural: torch.Tensor) -> torch.Tensor:
        """Activity-only deployment path requiring no E or calibration state."""
        if self.decoupled_key_mode != "x_only":
            raise RuntimeError("decode_x_only requires key_mode='x_only'")
        if self._decoder_frozen:
            self.decoder.eval()
            self.decoupled_v2.eval()
        query, hidden_activity = self._query_and_hidden_activity(neural)
        transformer_output, _ = (
            self.decoupled_v2.forward_dynamic_activity_key(
                query, hidden_activity
            )
        )
        return self.decoder.fc_out(transformer_output).permute(0, 2, 1)

    def forward(
        self,
        neural: torch.Tensor,
        calib_trials: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        side_features: Optional[torch.Tensor] = None,
        decoder_key_features: Optional[torch.Tensor] = None,
        electrode_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Explicit v2 dispatch; x-only E is framework/metric-only."""
        neuron_gate = None
        if identity is None:
            if calib_trials is None:
                raise ValueError(
                    "Training forward requires calib_trials or identity; "
                    "use decode_x_only(neural) for calibration-free x-only deployment"
                )
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
            raise ValueError(
                "teacher-readin decoupled v2 does not support encoder neuron gates"
            )
        if self.decoupled_key_mode == "x_only":
            if decoder_key_features is not None:
                raise ValueError("x_only forbids supplied direct key features")
            behavior = self.decode_x_only(neural)
        else:
            behavior = self.decode_with_decoupled_identity(
                neural,
                identity,
                decoder_key_features=decoder_key_features,
            )
        return behavior, identity

    def v2_cost_receipt(
        self,
        *,
        batch_size: int = 1,
        num_units: int = 64,
    ) -> dict[str, object]:
        cost = self.decoupled_v2.cost_receipt(
            batch_size=batch_size,
            num_units=num_units,
            num_queries=self.decoder.num_covariates,
            window_size=self.window_size,
            dynamic_activity_key=self.decoupled_key_mode == "x_only",
        )
        cost["reference_dtype"] = "float32"
        cost["reference_element_size_bytes"] = 4
        cost["identity_encoder_compute_included"] = False
        cost["identity_encoder_computed_for_framework_metrics_only"] = (
            self.decoupled_key_mode == "x_only"
        )
        coupled = super().decoder_cost_comparison_receipt(
            batch_size=batch_size, num_neurons=num_units
        )["coupled"]
        assert isinstance(coupled, dict)
        online = cost["online_macs_per_frame"]
        assert isinstance(online, dict)
        cost["coupled_reference"] = coupled
        cost["online_mac_reduction_fraction_vs_coupled"] = (
            1.0 - int(online["total"]) / int(coupled["total"])
        )
        cost["persistent_state_nonincreasing_vs_E"] = (
            int(cost["persistent_state"]["bytes"])
            <= batch_size * num_units * self.window_size * 4
        )
        return cost

    def freeze_decoder(self) -> int:
        frozen = super().freeze_decoder()
        for parameter in self.decoupled_v2.parameters():
            if parameter.requires_grad:
                parameter.requires_grad = False
                frozen += parameter.numel()
        self.decoupled_v2.eval()
        return frozen

    def train(self, mode: bool = True):
        super().train(mode)
        if self._decoder_frozen:
            self.decoupled_v2.eval()
        return self
