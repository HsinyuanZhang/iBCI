"""Fixed-width CarrierID consumer variants for source-date LODO only.

This module is deliberately additive to :mod:`h1_carrierid_spint`.  The
existing H-C implementation and checkpoints must remain byte-compatible.
``H1CarrierIdCiSpint`` admits exactly two *interface* widths while preserving
the 32-wide neural pre-pool and the 32-wide token passed to the SPINT decoder:

``1024 -> 32;  [32, carrier4] -> interface(32|64) -> 32 -> 700``.

It is not an H64 implementation.  In particular, the neural temporal path is
fixed at 32 and the downstream SPINT decoder is untouched.  All controls use
the same topology; ``zero_carrier`` is a literal model-bound C0 control, not
a data-loader shortcut.
"""
from __future__ import annotations

import random
from typing import Mapping

import torch
from torch import nn

from src.h1_m4_cce_contract import state_hash
from src.models.components.h1_carrierid_spint import (
    H1_CARRIERID_DIM,
    H1_CARRIERID_PRE_PARAMETERS,
    H1_CARRIERID_TRIAL_LENGTH,
    H1_CARRIERID_WINDOW_SIZE,
    H1_SPINT_ID_PARAMETERS,
    H1_SPINT_WHOLE_MODEL_PARAMETERS,
)
from src.models.components.spint import SpintModel


H1_CARRIERID_CI_HIDDEN_DIM = 32
H1_CARRIERID_CI_INTERFACE_DIMS = (32, 64)
H1_CARRIERID_CI32_PARAMETERS = 58_140
H1_CARRIERID_CI64_PARAMETERS = 60_348
H1_CARRIERID_CI32_WHOLE_MODEL_PARAMETERS = 10_947_836
H1_CARRIERID_CI64_WHOLE_MODEL_PARAMETERS = 10_950_044


class H1CarrierIdCiSpint(SpintModel):
    """CarrierID with a frozen choice of the carrier attachment width.

    ``carrier_interface_dim=32`` is the fresh matched CI32 arm.  ``64`` is
    CI64 and modifies only the first joint carrier/activity interface.  The
    initial carrier columns are literal zero for *every* arm, so Full, C0,
    LS, and RS start from the same function under a common seed.
    """

    def __init__(
        self,
        *,
        carrier_hidden_dim: int = H1_CARRIERID_CI_HIDDEN_DIM,
        carrier_interface_dim: int = 32,
        carrier_dim: int = H1_CARRIERID_DIM,
        carrier_trial_length: int = H1_CARRIERID_TRIAL_LENGTH,
        zero_carrier: bool = False,
        **kwargs,
    ) -> None:
        if int(carrier_hidden_dim) != H1_CARRIERID_CI_HIDDEN_DIM:
            raise ValueError("CI variants fix carrier_hidden_dim=32; H64 is prohibited in this module")
        if int(carrier_interface_dim) not in H1_CARRIERID_CI_INTERFACE_DIMS:
            raise ValueError("CI carrier_interface_dim must be exactly 32 or 64")
        if int(carrier_dim) != H1_CARRIERID_DIM:
            raise ValueError("CI variants require the normalized four-dimensional M=4 carrier")
        if int(carrier_trial_length) != H1_CARRIERID_TRIAL_LENGTH:
            raise ValueError("CI variants fix calibration trial length T=1024")
        if int(kwargs.get("window_size", -1)) != H1_CARRIERID_WINDOW_SIZE:
            raise ValueError("CI variants fix SPINT window_size=700")

        # Build SPINT first so shared decoder/backbone construction and RNG
        # consumption are identical across CI32/CI64 when the caller resets
        # the documented seed before each fresh construction.
        super().__init__(**kwargs)
        del self.fc_id_in
        del self.fc_id_out

        self.carrier_hidden_dim = H1_CARRIERID_CI_HIDDEN_DIM
        self.carrier_interface_dim = int(carrier_interface_dim)
        self.carrier_dim = H1_CARRIERID_DIM
        self.carrier_trial_length = H1_CARRIERID_TRIAL_LENGTH
        self.zero_carrier = bool(zero_carrier)
        self.carrier_pre_pool = nn.Sequential(
            nn.Linear(self.carrier_trial_length, self.carrier_hidden_dim),
            nn.ReLU(),
        )
        self.carrier_post_pool = nn.Sequential(
            nn.Linear(self.carrier_hidden_dim + self.carrier_dim, self.carrier_interface_dim),
            nn.ReLU(),
            nn.Linear(self.carrier_interface_dim, self.carrier_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.carrier_hidden_dim, self.window_size),
        )
        with torch.no_grad():
            self.carrier_post_pool[0].weight[:, self.carrier_hidden_dim :].zero_()

        expected = self.expected_carrier_parameter_count(self.carrier_interface_dim)
        if self.carrier_parameter_count() != expected:
            raise RuntimeError("CI CarrierID parameter accounting drifted")
        expected_whole = self.expected_whole_model_parameter_count(self.carrier_interface_dim)
        if sum(parameter.numel() for parameter in self.parameters()) != expected_whole:
            raise RuntimeError("CI CarrierID whole-model parameter accounting drifted")

    @staticmethod
    def expected_carrier_parameter_count(interface_dim: int) -> int:
        if int(interface_dim) == 32:
            return H1_CARRIERID_CI32_PARAMETERS
        if int(interface_dim) == 64:
            return H1_CARRIERID_CI64_PARAMETERS
        raise ValueError("interface_dim must be 32 or 64")

    @staticmethod
    def expected_whole_model_parameter_count(interface_dim: int) -> int:
        if int(interface_dim) == 32:
            return H1_CARRIERID_CI32_WHOLE_MODEL_PARAMETERS
        if int(interface_dim) == 64:
            return H1_CARRIERID_CI64_WHOLE_MODEL_PARAMETERS
        raise ValueError("interface_dim must be 32 or 64")

    def carrier_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.carrier_pre_pool.parameters()) + sum(
            parameter.numel() for parameter in self.carrier_post_pool.parameters()
        )

    def carrier_parameter_breakdown(self) -> dict[str, int]:
        return {
            "pre_pool": H1_CARRIERID_PRE_PARAMETERS,
            "post_pool": self.carrier_parameter_count() - H1_CARRIERID_PRE_PARAMETERS,
            "total": self.carrier_parameter_count(),
            "interface_dim": self.carrier_interface_dim,
            "original_spint_identity": H1_SPINT_ID_PARAMETERS,
            "original_spint_whole_model": H1_SPINT_WHOLE_MODEL_PARAMETERS,
            "carrierid_whole_model": sum(parameter.numel() for parameter in self.parameters()),
        }

    def shared_backbone_state_hash(self) -> str:
        """Hash all state except width-specific joint-interface tensors.

        This is the comparison domain required for CI32-vs-CI64 initialisation:
        the SPINT decoder and 1024-to-32 temporal pre-pool must be exactly the
        same, whereas ``carrier_post_pool`` is intentionally shape-different.
        """

        state: Mapping[str, torch.Tensor] = self.state_dict()
        shared = {name: tensor for name, tensor in state.items() if not name.startswith("carrier_post_pool.")}
        return state_hash(shared)

    def carrierid_identity_projection(
        self, calib_trialized_neural_features: torch.Tensor, carrier: torch.Tensor
    ) -> torch.Tensor:
        if calib_trialized_neural_features.ndim != 4:
            raise ValueError("CI calibration features must have shape [B,M,T,N]")
        batch_size, _trials, trial_length, num_neurons = calib_trialized_neural_features.shape
        if int(trial_length) != self.carrier_trial_length:
            raise ValueError(f"CI carrier requires T={self.carrier_trial_length}, got T={trial_length}")
        if carrier.ndim != 3 or tuple(carrier.shape) != (batch_size, num_neurons, self.carrier_dim):
            raise ValueError(f"CI carrier must be [B,N,{self.carrier_dim}] matching calibration features")
        temporal = calib_trialized_neural_features.permute(0, 1, 3, 2)
        pooled = self.carrier_pre_pool(temporal).mean(dim=1, keepdim=False)
        effective_carrier = torch.zeros_like(carrier) if self.zero_carrier else carrier
        joined = torch.cat((pooled, effective_carrier.to(dtype=pooled.dtype, device=pooled.device)), dim=-1)
        return self.carrier_post_pool(joined)

    def forward(self, src, calib_trialized_neural_features=None, carrier=None):
        if calib_trialized_neural_features is None or carrier is None:
            raise ValueError("CI CarrierID SPINT requires calibration identity and [B,N,4] carrier")
        src = src.permute(0, 2, 1)
        batch_size, num_neurons = src.size(0), src.size(1)
        identity = self.carrierid_identity_projection(calib_trialized_neural_features, carrier)
        src = src + identity
        dropout_mask = torch.ones(batch_size, num_neurons).to(src)
        if self.dynamic_dropout:
            p = random.uniform(self.dynamic_dropout_low, self.dynamic_dropout_high)
            dropout_mask = torch.nn.functional.dropout(dropout_mask, p=p, training=self.training)
        else:
            dropout_mask = torch.nn.functional.dropout(dropout_mask, p=self.dropout_rate, training=self.training)
        src = src * dropout_mask.unsqueeze(-1)
        src = self.fc_in(src)
        rep = self.fc_in(self.rep).to(src)
        output, _ = self.transformer(rep.repeat(batch_size, 1, 1), src)
        return self.fc_out(output).permute(0, 2, 1)
