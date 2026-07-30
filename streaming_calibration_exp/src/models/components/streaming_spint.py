"""SPINT decoder wrapper with pluggable streaming calibration encoder."""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from src.models.components.spint import SpintModel
from src.models.components.streaming_encoders import CalibrationEncoder


class CalibrationFixedSlotRouter(nn.Module):
    """Map a variable-size NeuronID-conditioned set to fixed decoder tokens.

    Calibration derives session-specific routing and FiLM parameters from the
    per-unit identities.  The same state can then be reused to project every
    live neural window from that session into a fixed number of slot tokens.
    """

    def __init__(
        self,
        window_size: int,
        slot_count: int,
        router_dim: int,
        routing_mode: str,
        fusion: str,
        temperature: float,
    ) -> None:
        super().__init__()
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")
        if router_dim <= 0:
            raise ValueError("router_dim must be positive")
        if routing_mode not in {"soft", "top1"}:
            raise ValueError("routing_mode must be 'soft' or 'top1'")
        if fusion not in {"additive", "film"}:
            raise ValueError("fusion must be 'additive' or 'film'")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self.window_size = window_size
        self.slot_count = slot_count
        self.router_dim = router_dim
        self.routing_mode = routing_mode
        self.fusion = fusion
        self.temperature = temperature
        self.key_projection = nn.Linear(window_size, router_dim, bias=False)
        self.slot_queries = nn.Parameter(torch.empty(slot_count, router_dim))
        self.mass_projection = nn.Linear(1, window_size)
        self.gain_projection = nn.Linear(window_size, window_size)
        self.bias_projection = nn.Linear(window_size, window_size)
        nn.init.normal_(self.slot_queries, std=router_dim**-0.5)
        nn.init.zeros_(self.mass_projection.weight)
        nn.init.zeros_(self.mass_projection.bias)
        nn.init.zeros_(self.gain_projection.weight)
        nn.init.zeros_(self.gain_projection.bias)
        nn.init.zeros_(self.bias_projection.weight)
        nn.init.zeros_(self.bias_projection.bias)

    def _expand_identity(self, identity: torch.Tensor, num_neurons: int) -> torch.Tensor:
        if identity.ndim != 3:
            raise ValueError(f"Expected identity [B,N,W], got {tuple(identity.shape)}")
        if identity.shape[-1] != self.window_size:
            raise ValueError(
                f"Identity window {identity.shape[-1]} does not match router window {self.window_size}"
            )
        if identity.shape[1] == num_neurons:
            return identity
        if identity.shape[1] == 1:
            return identity.expand(-1, num_neurons, -1)
        raise ValueError(
            f"Identity neuron dimension {identity.shape[1]} does not match neural dimension {num_neurons}"
        )

    def derive_calibration_state(self, identity: torch.Tensor, num_neurons: int) -> dict[str, torch.Tensor]:
        """Derive a serializable session routing state from NeuronID outputs."""
        identity = self._expand_identity(identity, num_neurons)
        keys = self.key_projection(identity)
        logits = torch.einsum("bnd,kd->bnk", keys, self.slot_queries)
        logits = logits / (self.router_dim**0.5 * self.temperature)
        soft_assignment = torch.softmax(logits, dim=-1)
        if self.routing_mode == "top1":
            indices = soft_assignment.argmax(dim=-1)
            hard_assignment = torch.nn.functional.one_hot(
                indices, num_classes=self.slot_count
            ).to(dtype=soft_assignment.dtype)
            assignment = (
                hard_assignment + soft_assignment - soft_assignment.detach()
                if self.training
                else hard_assignment
            )
        else:
            assignment = soft_assignment

        slot_mass = assignment.sum(dim=1)
        normalized_assignment = assignment.transpose(1, 2) / slot_mass.unsqueeze(-1).clamp_min(1.0e-6)
        slot_identity = torch.einsum("bkn,bnw->bkw", normalized_assignment, identity)
        mass_feature = torch.log1p(slot_mass).unsqueeze(-1)
        conditioning = slot_identity + self.mass_projection(mass_feature)
        if self.fusion == "additive":
            gain = torch.ones_like(conditioning)
            bias = conditioning
        else:
            gain = 1.0 + 0.5 * torch.tanh(self.gain_projection(conditioning))
            bias = self.bias_projection(conditioning)
        return {
            "assignment": assignment,
            "normalized_assignment": normalized_assignment,
            "slot_mass": slot_mass,
            "slot_identity": slot_identity,
            "gain": gain,
            "bias": bias,
        }

    def project_neural(
        self,
        neural_tokens: torch.Tensor,
        calibration_state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Project ``[B,N,W]`` live spikes into fixed ``[B,K,W]`` slot tokens."""
        if neural_tokens.ndim != 3:
            raise ValueError(f"Expected neural tokens [B,N,W], got {tuple(neural_tokens.shape)}")
        if neural_tokens.shape[-1] != self.window_size:
            raise ValueError(
                f"Neural window {neural_tokens.shape[-1]} does not match router window {self.window_size}"
            )
        normalized_assignment = calibration_state["normalized_assignment"]
        state_batch_size = normalized_assignment.shape[0]
        neural_batch_size = neural_tokens.shape[0]
        if state_batch_size == 1 and neural_batch_size > 1:
            normalized_assignment = normalized_assignment.expand(neural_batch_size, -1, -1)
            gain = calibration_state["gain"].expand(neural_batch_size, -1, -1)
            bias = calibration_state["bias"].expand(neural_batch_size, -1, -1)
            neuron_gate = calibration_state.get("neuron_gate")
            if neuron_gate is not None:
                neuron_gate = neuron_gate.expand(neural_batch_size, -1, -1)
        elif state_batch_size == neural_batch_size:
            gain = calibration_state["gain"]
            bias = calibration_state["bias"]
            neuron_gate = calibration_state.get("neuron_gate")
        else:
            raise ValueError(
                "Calibration-state batch size must match neural batch size, unless a single "
                "session state is reused for multiple online windows"
            )
        if normalized_assignment.shape[-1] != neural_tokens.shape[1]:
            raise ValueError("Calibration-state unit count must match neural unit count")
        if neuron_gate is not None:
            expected_gate_shape = (*neural_tokens.shape[:2], 1)
            if neuron_gate.shape != expected_gate_shape:
                raise ValueError(
                    "Calibration-state neuron gate must have shape "
                    f"{expected_gate_shape}, got {tuple(neuron_gate.shape)}"
                )
            neural_tokens = neural_tokens * neuron_gate
        slot_neural = torch.einsum("bkn,bnw->bkw", normalized_assignment, neural_tokens)
        return gain * slot_neural + bias

    def forward(self, neural_tokens: torch.Tensor, identity: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        calibration_state = self.derive_calibration_state(identity, neural_tokens.shape[1])
        return self.project_neural(neural_tokens, calibration_state), calibration_state


class StreamingSpintModel(nn.Module):
    """Frozen decoder + trainable/streaming calibration encoder."""

    def __init__(
        self,
        decoder: SpintModel,
        id_encoder: CalibrationEncoder,
        fixed_slot_count: int = 0,
        fixed_slot_dim: int = 32,
        fixed_slot_mode: str = "soft",
        fixed_slot_fusion: str = "film",
        fixed_slot_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.id_encoder = id_encoder
        self.window_size = decoder.window_size
        self._decoder_frozen = False
        self.fixed_slot_router = (
            CalibrationFixedSlotRouter(
                window_size=decoder.window_size,
                slot_count=fixed_slot_count,
                router_dim=fixed_slot_dim,
                routing_mode=fixed_slot_mode,
                fusion=fixed_slot_fusion,
                temperature=fixed_slot_temperature,
            )
            if fixed_slot_count > 0
            else None
        )

    def compute_identity(
        self,
        calib_trials: torch.Tensor,
        side_features: Optional[torch.Tensor] = None,
        electrode_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.id_encoder.forward_batch(
            calib_trials, side_features=side_features, electrode_ids=electrode_ids
        )

    def decode_with_identity(
        self,
        neural: torch.Tensor,
        identity: torch.Tensor,
        neuron_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """neural: [B,W,N], identity: [B,N,W] -> behavior [B,W,C].

        Frozen decoder weights stay in eval() with requires_grad=False, but the
        forward path must remain differentiable w.r.t. identity E.
        """
        src = neural.permute(0, 2, 1)
        if neuron_gate is not None:
            src = src * neuron_gate
        if self.fixed_slot_router is None:
            src = src + identity
        else:
            src, _ = self.fixed_slot_router(src, identity)

        if self._decoder_frozen:
            self.decoder.eval()

        batch_size = src.size(0)
        num_neurons = src.size(1)
        dropout_mask = torch.ones(batch_size, num_neurons, device=src.device, dtype=src.dtype)
        if not self._decoder_frozen:
            if self.decoder.dynamic_dropout and self.training:
                import random

                p = random.uniform(self.decoder.dynamic_dropout_low, self.decoder.dynamic_dropout_high)
                dropout_mask = torch.nn.functional.dropout(dropout_mask, p=p, training=True)
            elif self.decoder.dropout_rate > 0.0 and self.training:
                dropout_mask = torch.nn.functional.dropout(
                    dropout_mask, p=self.decoder.dropout_rate, training=True
                )
        src = src * dropout_mask.unsqueeze(-1)

        src = self.decoder.fc_in(src)
        rep = self.decoder.fc_in(self.decoder.rep).to(src)
        transformer_output, _ = self.decoder.transformer(rep.repeat(src.size(0), 1, 1), src)
        output = self.decoder.fc_out(transformer_output)
        return output.permute(0, 2, 1)

    @torch.no_grad()
    def derive_fixed_slot_state(
        self,
        identity: torch.Tensor,
        num_neurons: int,
        neuron_gate: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Export the calibration-only state required by a fixed-slot deployment."""
        if self.fixed_slot_router is None:
            raise RuntimeError("derive_fixed_slot_state requires fixed_slot_count > 0")
        calibration_state = self.fixed_slot_router.derive_calibration_state(identity, num_neurons)
        if neuron_gate is not None:
            expected_gate_shape = (identity.shape[0], num_neurons, 1)
            if neuron_gate.shape != expected_gate_shape:
                raise ValueError(
                    f"Neuron gate must have shape {expected_gate_shape}, got {tuple(neuron_gate.shape)}"
                )
            calibration_state["neuron_gate"] = neuron_gate
        return calibration_state

    @torch.no_grad()
    def decode_with_fixed_slot_state(
        self,
        neural: torch.Tensor,
        calibration_state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Decode with a previously derived fixed-slot state without recomputing identities."""
        if self.fixed_slot_router is None:
            raise RuntimeError("decode_with_fixed_slot_state requires fixed_slot_count > 0")
        src = neural.permute(0, 2, 1)
        src = self.fixed_slot_router.project_neural(src, calibration_state)
        if self._decoder_frozen:
            self.decoder.eval()
        src = self.decoder.fc_in(src)
        rep = self.decoder.fc_in(self.decoder.rep).to(src)
        transformer_output, _ = self.decoder.transformer(rep.repeat(src.size(0), 1, 1), src)
        output = self.decoder.fc_out(transformer_output)
        return output.permute(0, 2, 1)

    def forward(
        self,
        neural: torch.Tensor,
        calib_trials: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        side_features: Optional[torch.Tensor] = None,
        electrode_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        neuron_gate = None
        if identity is None:
            if calib_trials is None:
                raise ValueError("Either calib_trials or identity must be provided")
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
        behavior = self.decode_with_identity(neural, identity, neuron_gate=neuron_gate)
        return behavior, identity

    def freeze_decoder(self) -> int:
        frozen = 0
        for param in self.decoder.parameters():
            param.requires_grad = False
            frozen += param.numel()
        self._decoder_frozen = True
        self.decoder.eval()
        return frozen

    def train(self, mode: bool = True):
        super().train(mode)
        if self._decoder_frozen:
            self.decoder.eval()
        return self

    def trainable_encoder_parameters(self):
        return (parameter for parameter in self.id_encoder.parameters() if parameter.requires_grad)
