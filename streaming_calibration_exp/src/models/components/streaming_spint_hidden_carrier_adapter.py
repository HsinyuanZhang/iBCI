"""Hidden-space carrier adapter for coupled streaming SPINT.

Default ``add_site='waveform'`` is ordinary W-add: ``h = fc_in(x + E)``.
``add_site='hidden'`` keeps the matched activity identity and adds a bias-free
map in decoder hidden space:

``h = fc_in(x + E^A) + P(carrier)``.

``P`` has no bias, so ``P(0)=0``. Zero initialization makes ``P`` identically
zero, so H-add matches W-add on the same activity identity. Decoder and
teacher tensors are untouched and remain strict-loadable; no new teacher is
introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import chain
from typing import Iterator, Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .spint import SpintModel
from .streaming_encoders import CalibrationEncoder
from .streaming_spint import StreamingSpintModel


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


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
class HiddenCarrierState:
    """Cached hidden-space add, shape ``[B,N,H]``."""

    hidden_add: torch.Tensor

    def __post_init__(self) -> None:
        if self.hidden_add.ndim != 3:
            raise ValueError("hidden_add must have shape [B,N,H]")

    @property
    def nbytes(self) -> int:
        value = self.hidden_add
        return value.numel() * value.element_size()


class ZeroInitializedHiddenCarrierMap(nn.Module):
    """Bias-free carrier→hidden map created without advancing torch RNG.

    ``nn.Linear`` is deliberately not used: its constructor samples a weight
    before callers can zero it.  A1 needs H-add construction to leave the
    common W-add RNG stream untouched, so the sole factor is registered
    directly as an exact-zero parameter.
    """

    def __init__(self, *, carrier_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if carrier_dim <= 0 or hidden_dim <= 0:
            raise ValueError("carrier_dim and hidden_dim must be positive")
        self.carrier_dim = int(carrier_dim)
        self.hidden_dim = int(hidden_dim)
        self.weight = nn.Parameter(
            torch.zeros(self.hidden_dim, self.carrier_dim, dtype=torch.float32)
        )
        # An inspectable non-parameter attribute makes the no-bias contract
        # explicit while keeping the state dict to exactly one tensor.
        self.bias: None = None
        self._initial_sha256 = self.factor_sha256()

    @property
    def projection(self) -> "ZeroInitializedHiddenCarrierMap":
        """Compatibility view for the original prototype's ``.projection``.

        This property does not register another module or parameter; the
        checkpoint key remains the canonical ``weight``.
        """

        return self

    @property
    def bias_parameters(self) -> int:
        return 0

    def forward(self, carrier: torch.Tensor) -> torch.Tensor:
        if carrier.ndim != 3 or carrier.shape[-1] != self.carrier_dim:
            raise ValueError(
                f"carrier must have shape [B,N,{self.carrier_dim}]"
            )
        return F.linear(carrier, self.weight, bias=None)

    def factor_sha256(self) -> str:
        return _tensor_state_sha256(dict(self.state_dict()))

    @property
    def initialization_receipt(self) -> dict[str, object]:
        zeros = self.weight.new_zeros(1, 1, self.carrier_dim)
        p_zero = self.forward(zeros)
        return {
            "schema_version": 1,
            "family": "zero_initialized_bias_free_hidden_carrier_map",
            "carrier_dim": self.carrier_dim,
            "hidden_dim": self.hidden_dim,
            "initial_factor_sha256": self._initial_sha256,
            "active_factor_sha256": self.factor_sha256(),
            "bias_parameters": self.bias_parameters,
            "p_of_zero_is_zero": bool(torch.equal(p_zero, torch.zeros_like(p_zero))),
            "parameter_count": self.carrier_dim * self.hidden_dim,
        }


class HiddenSpaceCarrierStreamingSpint(StreamingSpintModel):
    """A1-only student with an optional post-readin hidden carrier.

    The shared :class:`StreamingSpintModel` stays byte-for-byte unchanged.
    This subclass owns the small duplicated coupled decode segment required to
    insert ``P(T4)`` after ``decoder.fc_in`` and before the transformer.
    """

    def __init__(
        self,
        *,
        decoder: SpintModel,
        id_encoder: CalibrationEncoder,
        add_site: Literal["waveform", "hidden"] = "waveform",
        attachment_mode: Literal["aligned", "shuffled"] = "aligned",
        permutation_seed: int | None = None,
        carrier_dim: int = 4,
    ) -> None:
        if add_site not in {"waveform", "hidden"}:
            raise ValueError("add_site must be 'waveform' or 'hidden'")
        if attachment_mode not in {"aligned", "shuffled"}:
            raise ValueError("attachment_mode must be 'aligned' or 'shuffled'")
        if attachment_mode == "shuffled" and permutation_seed is None:
            raise ValueError("shuffled attachment requires a permutation seed")
        if attachment_mode == "aligned" and permutation_seed is not None:
            raise ValueError("aligned attachment forbids a permutation seed")
        super().__init__(
            decoder=decoder,
            id_encoder=id_encoder,
            fixed_slot_count=0,
            decoder_mode="coupled",
        )
        self.add_site = add_site
        self.attachment_mode = attachment_mode
        self.permutation_seed = permutation_seed
        self.carrier_dim = int(carrier_dim)
        self.hidden_carrier_map: ZeroInitializedHiddenCarrierMap | None = (
            ZeroInitializedHiddenCarrierMap(
                carrier_dim=self.carrier_dim,
                hidden_dim=decoder.model_dim,
            )
            if self.add_site == "hidden"
            else None
        )

    def _maybe_shuffle(self, carrier: torch.Tensor) -> torch.Tensor:
        if carrier.ndim != 3 or carrier.shape[-1] != self.carrier_dim:
            raise ValueError(
                f"carrier must have shape [B,N,{self.carrier_dim}]"
            )
        if self.attachment_mode == "aligned":
            return carrier
        assert self.permutation_seed is not None
        order = np.random.RandomState(self.permutation_seed).permutation(
            carrier.shape[1]
        )
        index = torch.as_tensor(order, device=carrier.device)
        return carrier.index_select(1, index)

    def set_attachment_control_for_evaluation(
        self,
        *,
        attachment_mode: Literal["aligned", "shuffled"],
        permutation_seed: int | None = None,
    ) -> None:
        """Switch only the T4-to-unit attachment on a loaded checkpoint.

        This is the TS4 same-checkpoint control.  It is deliberately refused
        in training mode so it cannot become an unrecorded source-training
        family.
        """

        if self.training:
            raise RuntimeError("A1 attachment controls are evaluation-only")
        if attachment_mode not in {"aligned", "shuffled"}:
            raise ValueError("attachment_mode must be 'aligned' or 'shuffled'")
        if attachment_mode == "aligned" and permutation_seed is not None:
            raise ValueError("aligned attachment forbids a permutation seed")
        if attachment_mode == "shuffled" and permutation_seed is None:
            raise ValueError("shuffled attachment requires a permutation seed")
        self.attachment_mode = attachment_mode
        self.permutation_seed = permutation_seed

    def derive_hidden_carrier_state(
        self, carrier: torch.Tensor
    ) -> HiddenCarrierState:
        if self.hidden_carrier_map is None:
            raise RuntimeError("hidden carrier state requires add_site='hidden'")
        return HiddenCarrierState(
            self.hidden_carrier_map(self._maybe_shuffle(carrier))
        )

    def _expanded_hidden_state(
        self,
        state: HiddenCarrierState,
        *,
        batch_size: int,
        num_units: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(state, HiddenCarrierState):
            raise TypeError("state must be HiddenCarrierState")
        hidden = state.hidden_add
        if hidden.shape[0] == 1 and batch_size > 1:
            hidden = hidden.expand(batch_size, -1, -1)
        expected = (batch_size, num_units, self.decoder.model_dim)
        if hidden.shape != expected:
            raise ValueError(
                f"hidden carrier state shape {tuple(hidden.shape)} "
                f"does not match {expected}"
            )
        return hidden.to(reference)

    def decode_with_hidden_carrier_state(
        self,
        neural: torch.Tensor,
        identity: torch.Tensor,
        state: HiddenCarrierState,
    ) -> torch.Tensor:
        if neural.ndim != 3 or neural.shape[1] != self.window_size:
            raise ValueError(
                f"neural must have shape [B,{self.window_size},N]"
            )
        batch_size, _, num_units = neural.shape
        identity = self._expanded_identity(identity, batch_size, num_units)
        hidden = self._expanded_hidden_state(
            state,
            batch_size=batch_size,
            num_units=num_units,
            reference=neural,
        )
        # Keep this logic local to A1.  Adding an optional argument to the
        # shared W-add implementation would invalidate B1/A2 source bindings.
        src = neural.permute(0, 2, 1)
        src = src + identity

        if self._decoder_frozen:
            self.decoder.eval()

        dropout_mask = torch.ones(
            batch_size,
            num_units,
            device=src.device,
            dtype=src.dtype,
        )
        if not self._decoder_frozen:
            if self.decoder.dynamic_dropout and self.training:
                import random

                probability = random.uniform(
                    self.decoder.dynamic_dropout_low,
                    self.decoder.dynamic_dropout_high,
                )
                dropout_mask = F.dropout(
                    dropout_mask,
                    p=probability,
                    training=True,
                )
            elif self.decoder.dropout_rate > 0.0 and self.training:
                dropout_mask = F.dropout(
                    dropout_mask,
                    p=self.decoder.dropout_rate,
                    training=True,
                )
        src = src * dropout_mask.unsqueeze(-1)
        src = self.decoder.fc_in(src)
        src = src + hidden
        rep = self.decoder.fc_in(self.decoder.rep).to(src)
        transformer_output, _ = self.decoder.transformer(
            rep.repeat(src.size(0), 1, 1), src
        )
        output = self.decoder.fc_out(transformer_output)
        return output.permute(0, 2, 1)

    def decode_with_hidden_carrier(
        self,
        neural: torch.Tensor,
        identity: torch.Tensor,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        state = self.derive_hidden_carrier_state(carrier)
        return self.decode_with_hidden_carrier_state(neural, identity, state)

    def support_query_provenance_receipt(
        self,
        neural: torch.Tensor,
        *,
        calib_trials: torch.Tensor,
        side_features: torch.Tensor,
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "query_neural_sha256": _tensor_sha256(neural),
            "support_calib_sha256": _tensor_sha256(calib_trials),
            "carrier_sha256": _tensor_sha256(side_features),
            "query_shape": tuple(neural.shape),
            "support_shape": tuple(calib_trials.shape),
            "carrier_shape": tuple(side_features.shape),
            "add_site": self.add_site,
            "attachment_mode": self.attachment_mode,
        }

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
            raise ValueError("hidden-space adapter owns its direct carrier routing")
        if side_features is None:
            raise ValueError(
                "hidden-space adapter requires aligned carrier side features"
            )
        if identity is None:
            if calib_trials is None:
                raise ValueError("calib_trials or identity is required")
            encoder_side = (
                torch.zeros_like(side_features)
                if self.add_site == "hidden"
                else self._maybe_shuffle(side_features)
            )
            identity = self.compute_identity(
                calib_trials,
                side_features=encoder_side,
                electrode_ids=electrode_ids,
            )
        if self.add_site == "waveform":
            behavior = self.decode_with_identity(neural, identity)
        else:
            behavior = self.decode_with_hidden_carrier(
                neural, identity, side_features
            )
        return behavior, identity

    def trainable_encoder_parameters(self) -> Iterator[nn.Parameter]:
        """Return the frozen-decoder A1 trainables, including ``P``.

        The generic Lightning module calls this method when the decoder is
        frozen.  Keeping the adapter factor here prevents the silent failure
        where ``P.requires_grad`` is true but it never enters the optimizer.
        """

        encoder_parameters = (
            parameter
            for parameter in self.id_encoder.parameters()
            if parameter.requires_grad
        )
        if self.hidden_carrier_map is None:
            return encoder_parameters
        adapter_parameters = (
            parameter
            for parameter in self.hidden_carrier_map.parameters()
            if parameter.requires_grad
        )
        return chain(encoder_parameters, adapter_parameters)

    @property
    def hidden_carrier_receipt(self) -> dict[str, object]:
        if self.hidden_carrier_map is None:
            raise RuntimeError("hidden carrier receipt requires add_site='hidden'")
        receipt = self.hidden_carrier_map.initialization_receipt
        receipt.update(
            {
                "add_site": self.add_site,
                "attachment_mode": self.attachment_mode,
                "permutation_seed": self.permutation_seed,
                "p_of_zero_is_zero": True,
                "bias_parameters": 0,
                "teacher_coupled_activity_identity_readin_preserved": True,
                "teacher_query_key_value_and_output_projections_preserved": True,
                "new_teacher_required": False,
                "decoder_strict_loadable": True,
            }
        )
        return receipt

    def hidden_carrier_cost_receipt(
        self,
        *,
        batch_size: int = 1,
        num_units: int = 64,
    ) -> dict[str, object]:
        if self.hidden_carrier_map is None:
            raise RuntimeError("hidden carrier cost requires add_site='hidden'")
        if batch_size <= 0 or num_units <= 0:
            raise ValueError("batch_size and num_units must be positive")
        coupled = super().decoder_cost_comparison_receipt(
            batch_size=batch_size, num_neurons=num_units
        )["coupled"]
        hidden_dim = self.decoder.model_dim
        state_elements = num_units * hidden_dim
        decoder_parameter_count = sum(
            parameter.numel() for parameter in self.decoder.parameters()
        )
        return {
            "schema_version": 1,
            "reference_shape": {
                "batch_size": batch_size,
                "num_units": num_units,
                "hidden_dim": hidden_dim,
                "carrier_dim": self.carrier_dim,
                "window_size": self.window_size,
            },
            "coupled_reference": coupled,
            "trainable_parameter_count": self.carrier_dim * hidden_dim,
            "decoder_parameter_count": decoder_parameter_count,
            "teacher_tensors_unmodified": True,
            "calibration_only_hidden_map_macs": (
                num_units * self.carrier_dim * hidden_dim
            ),
            "online_increment": {
                "additional_linear_macs": 0,
                "hidden_additions": batch_size * state_elements,
            },
            "persistent_additional_state": {
                "elements": state_elements,
                "bytes_fp32": state_elements * 4,
                "bytes_fp16": state_elements * 2,
            },
            "add_site": self.add_site,
            "w_add_unchanged_when_disabled": self.add_site == "waveform",
        }
