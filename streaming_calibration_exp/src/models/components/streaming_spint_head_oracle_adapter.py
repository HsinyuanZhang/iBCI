"""Isolated streaming adapter for the teacher-head-preserving K/V oracle."""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch.nn.parameter import UninitializedParameter

from .head_preserving_decoupled_oracle import (
    HeadPreservingKVState,
    TeacherHeadPreservingDecoupledCrossAttention,
)
from .spint import SpintModel
from .streaming_encoders import CalibrationEncoder
from .streaming_spint import StreamingSpintModel


class TeacherHeadOracleStreamingSpint(StreamingSpintModel):
    """Full-teacher-head static K(E), online V(x) diagnostic."""

    def __init__(
        self,
        *,
        decoder: SpintModel,
        id_encoder: CalibrationEncoder,
        key_mode: Literal["e_t4", "e_ts4"],
        key_permutation_seed: int | None = None,
    ) -> None:
        if key_mode not in {"e_t4", "e_ts4"}:
            raise ValueError("key_mode must be 'e_t4' or 'e_ts4'")
        if key_mode == "e_ts4" and key_permutation_seed is None:
            raise ValueError("e_ts4 requires a key permutation seed")
        if key_mode == "e_t4" and key_permutation_seed is not None:
            raise ValueError("e_t4 forbids a key permutation seed")
        if decoder.num_layers != 1:
            raise ValueError("head-preserving oracle requires one decoder layer")

        super().__init__(
            decoder=decoder,
            id_encoder=id_encoder,
            fixed_slot_count=0,
            decoder_mode="coupled",
        )
        self.decoder_mode = "teacher_head_preserving_decoupled_oracle"
        self.oracle_key_mode = key_mode
        self.oracle_key_permutation_seed = key_permutation_seed
        legacy = decoder.transformer.layers[0]
        self.head_oracle = (
            TeacherHeadPreservingDecoupledCrossAttention.from_teacher(
                legacy
            )
        )

        # The copied oracle replaces these modules on the active task path.
        for unused in (
            self.decoder.transformer,
            self.decoder.fc_id_in,
            self.decoder.fc_id_out,
        ):
            for parameter in unused.parameters():
                parameter.requires_grad = False
            unused.eval()

    @property
    def oracle_initialization_receipt(self) -> dict[str, object]:
        receipt = dict(self.head_oracle.initialization_receipt)
        receipt.update({
            "key_mode": self.oracle_key_mode,
            "key_permutation_seed": self.oracle_key_permutation_seed,
            "identity_encoder_receives_aligned_t4": True,
            "legacy_transformer_active": False,
            "legacy_transformer_trainable": False,
            "optimizer_trainable_parameter_count": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
        })
        inactive = [
            parameter
            for module in (
                self.decoder.transformer,
                self.decoder.fc_id_in,
                self.decoder.fc_id_out,
            )
            for parameter in module.parameters()
        ]
        receipt["inactive_legacy_parameter_count"] = sum(
            parameter.numel()
            for parameter in inactive
            if not isinstance(parameter, UninitializedParameter)
        )
        return receipt

    def _identity_key_input(self, identity: torch.Tensor) -> torch.Tensor:
        if self.oracle_key_mode == "e_t4":
            return identity
        assert self.oracle_key_permutation_seed is not None
        order = np.random.RandomState(
            self.oracle_key_permutation_seed
        ).permutation(identity.shape[1])
        index = torch.as_tensor(order, device=identity.device)
        return identity.index_select(1, index)

    def derive_head_oracle_state(
        self, identity: torch.Tensor
    ) -> HeadPreservingKVState:
        """Calibration-only ``E→fc_in→norm1→Wk`` path."""
        if identity.ndim != 3 or identity.shape[-1] != self.window_size:
            raise ValueError(
                f"identity must have shape [B,N,{self.window_size}]"
            )
        key_identity = self._identity_key_input(identity)
        hidden_identity = self.decoder.fc_in(key_identity)
        return self.head_oracle.derive_static_key(hidden_identity)

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

    def decode_with_head_oracle_state(
        self,
        neural: torch.Tensor,
        state: HeadPreservingKVState,
    ) -> torch.Tensor:
        """Online cached-K path; E and Wk are absent from this graph."""
        if self._decoder_frozen:
            self.decoder.eval()
            self.head_oracle.eval()
        query, hidden_activity = self._query_and_hidden_activity(neural)
        output, _ = self.head_oracle.forward_cached(
            query, state, hidden_activity
        )
        return self.decoder.fc_out(output).permute(0, 2, 1)

    def decode_with_head_oracle_identity(
        self,
        neural: torch.Tensor,
        identity: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable training/reference path."""
        batch_size = neural.shape[0]
        num_units = neural.shape[-1]
        identity = self._expanded_identity(
            identity, batch_size, num_units
        )
        if self._decoder_frozen:
            self.decoder.eval()
            self.head_oracle.eval()
        query, hidden_activity = self._query_and_hidden_activity(neural)
        state = self.derive_head_oracle_state(identity)
        output, _ = self.head_oracle.forward_cached(
            query, state, hidden_activity
        )
        return self.decoder.fc_out(output).permute(0, 2, 1)

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
                "head oracle has no direct decoder T4 branch; "
                "TS4 changes only decoder-K E rows"
            )
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
            raise ValueError("head oracle does not support encoder neuron gates")
        behavior = self.decode_with_head_oracle_identity(
            neural, identity
        )
        return behavior, identity

    def oracle_cost_receipt(
        self,
        *,
        batch_size: int = 1,
        num_units: int = 64,
    ) -> dict[str, object]:
        receipt = self.head_oracle.decoder_cost_receipt(
            batch_size=batch_size,
            num_units=num_units,
            num_queries=self.decoder.num_covariates,
            window_size=self.window_size,
        )
        coupled = super().decoder_cost_comparison_receipt(
            batch_size=batch_size, num_neurons=num_units
        )["coupled"]
        assert isinstance(coupled, dict)
        online = receipt["online_macs_per_window"]
        assert isinstance(online, dict)
        receipt["coupled_reference"] = coupled
        receipt["online_mac_reduction_fraction_vs_coupled"] = (
            1.0 - int(online["total"]) / int(coupled["total"])
        )
        receipt["identity_encoder_compute_included"] = False
        return receipt

    def freeze_decoder(self) -> int:
        frozen = super().freeze_decoder()
        for parameter in self.head_oracle.parameters():
            if parameter.requires_grad:
                parameter.requires_grad = False
                frozen += parameter.numel()
        self.head_oracle.eval()
        return frozen

    def train(self, mode: bool = True):
        super().train(mode)
        self.decoder.transformer.eval()
        self.decoder.fc_id_in.eval()
        self.decoder.fc_id_out.eval()
        if self._decoder_frozen:
            self.head_oracle.eval()
        return self
