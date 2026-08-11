"""Parameter-efficient H1 CarrierID replacement for SPINT's identity MLP.

``H1CarrierIdSpint`` deliberately builds the original :class:`SpintModel`
first, removes only ``fc_id_in`` and ``fc_id_out``, and then inserts a small
carrier-aware identity path.  Consequently the read-in, learned behavioural
queries, cross-attention stack, and output head retain the exact parameter
initialisation/topology of a matched H1 SPINT model.

The carrier is a *deployment forward input*, not an optimisation variable:
no target-session gradient, optimiser state, or learned session table is
required.  The h=32 implementation is intentionally fixed by contract; this
is a single pre-registered pilot rather than a hidden-width sweep.
"""
from __future__ import annotations

import random

import torch
from torch import nn

from src.models.components.spint import SpintModel


H1_CARRIERID_HIDDEN_DIM = 32
H1_CARRIERID_DIM = 4
H1_CARRIERID_TRIAL_LENGTH = 1024
H1_CARRIERID_WINDOW_SIZE = 700
H1_CARRIERID_PRE_PARAMETERS = 32_800
H1_CARRIERID_POST_PARAMETERS = 25_340
H1_CARRIERID_PARAMETERS = 58_140
H1_SPINT_ID_PARAMETERS = 5_965_500
# These are static model parameters.  The V2 additive-residual comparison had
# a further 2,800 parameters; CarrierID deliberately has no residual matrix.
H1_SPINT_WHOLE_MODEL_PARAMETERS = 16_855_196
H1_CARRIERID_WHOLE_MODEL_PARAMETERS = 10_947_836


class H1CarrierIdSpint(SpintModel):
    """SPINT with its large identity MLP replaced by the fixed h=32 CarrierID.

    The path is, per neural channel,

    ``[B,M,N,1024] -> Linear(1024,32)+ReLU -> mean_M``
    ``concat(normalized_carrier[B,N,4]) -> 36->32->32->700``.

    ``zero_carrier`` is a model-bound deployment intervention used only for
    the separately trained H-C0 arm.  It makes the carrier a literal zero at
    the model boundary while preserving an identical source DataModule and
    model parameterization.
    """

    def __init__(
        self,
        *,
        carrier_hidden_dim: int = H1_CARRIERID_HIDDEN_DIM,
        carrier_dim: int = H1_CARRIERID_DIM,
        carrier_trial_length: int = H1_CARRIERID_TRIAL_LENGTH,
        zero_carrier: bool = False,
        **kwargs,
    ) -> None:
        if int(carrier_hidden_dim) != H1_CARRIERID_HIDDEN_DIM:
            raise ValueError("H1 CarrierID pilot fixes carrier_hidden_dim=32; hidden-width sweeps are forbidden")
        if int(carrier_dim) != H1_CARRIERID_DIM:
            raise ValueError("H1 CarrierID requires a four-dimensional normalized EB carrier")
        if int(carrier_trial_length) != H1_CARRIERID_TRIAL_LENGTH:
            raise ValueError("H1 CarrierID pilot fixes calibration trial length T=1024")
        if int(kwargs.get("window_size", -1)) != H1_CARRIERID_WINDOW_SIZE:
            raise ValueError("H1 CarrierID pilot fixes SPINT window_size=700")

        # Preserve all original SPINT construction/RNG consumption through the
        # downstream modules, then erase precisely the two old identity MLPs.
        super().__init__(**kwargs)
        del self.fc_id_in
        del self.fc_id_out

        self.carrier_hidden_dim = H1_CARRIERID_HIDDEN_DIM
        self.carrier_dim = H1_CARRIERID_DIM
        self.carrier_trial_length = H1_CARRIERID_TRIAL_LENGTH
        self.zero_carrier = bool(zero_carrier)
        self.carrier_pre_pool = nn.Sequential(
            nn.Linear(self.carrier_trial_length, self.carrier_hidden_dim),
            nn.ReLU(),
        )
        self.carrier_post_pool = nn.Sequential(
            nn.Linear(self.carrier_hidden_dim + self.carrier_dim, self.carrier_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.carrier_hidden_dim, self.carrier_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.carrier_hidden_dim, self.window_size),
        )
        # At step zero Full and H-C0 are the same *function*.  This does not
        # freeze these weights: the full arm can learn to use a correct carrier
        # from source-side backpropagation, whereas H-C0 cannot see one.
        with torch.no_grad():
            self.carrier_post_pool[0].weight[:, self.carrier_hidden_dim :].zero_()

        if self.carrier_parameter_count() != H1_CARRIERID_PARAMETERS:
            raise RuntimeError("CarrierID parameter accounting drifted from the registered h=32 design")

    def carrier_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.carrier_pre_pool.parameters()) + sum(
            parameter.numel() for parameter in self.carrier_post_pool.parameters()
        )

    def carrier_parameter_breakdown(self) -> dict[str, int]:
        return {
            "pre_pool": sum(parameter.numel() for parameter in self.carrier_pre_pool.parameters()),
            "post_pool": sum(parameter.numel() for parameter in self.carrier_post_pool.parameters()),
            "total": self.carrier_parameter_count(),
            "original_spint_identity": H1_SPINT_ID_PARAMETERS,
            "original_spint_whole_model": H1_SPINT_WHOLE_MODEL_PARAMETERS,
            "carrierid_whole_model": sum(parameter.numel() for parameter in self.parameters()),
        }

    def carrierid_identity_projection(
        self,
        calib_trialized_neural_features: torch.Tensor,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        """Build one [B,N,700] identity token without mutating session state."""

        if calib_trialized_neural_features.ndim != 4:
            raise ValueError(
                "CarrierID calibration features must be [B,M,T,N], got "
                f"{tuple(calib_trialized_neural_features.shape)}"
            )
        batch_size, _trials, trial_length, num_neurons = calib_trialized_neural_features.shape
        if int(trial_length) != self.carrier_trial_length:
            raise ValueError(f"CarrierID requires T={self.carrier_trial_length}, got T={trial_length}")
        if carrier.ndim != 3 or tuple(carrier.shape) != (batch_size, num_neurons, self.carrier_dim):
            raise ValueError(
                f"CarrierID carrier must be [B,N,{self.carrier_dim}] matching calibration features, got {tuple(carrier.shape)}"
            )
        # B,M,T,N -> B,M,N,T; one shared linear map per channel/trial.
        temporal = calib_trialized_neural_features.permute(0, 1, 3, 2)
        pooled = self.carrier_pre_pool(temporal).mean(dim=1, keepdim=False)
        effective_carrier = torch.zeros_like(carrier) if self.zero_carrier else carrier
        joined = torch.cat((pooled, effective_carrier.to(dtype=pooled.dtype, device=pooled.device)), dim=-1)
        return self.carrier_post_pool(joined)

    def forward(self, src, calib_trialized_neural_features=None, carrier=None):
        if calib_trialized_neural_features is None or carrier is None:
            raise ValueError("H1 CarrierID SPINT requires calibration identity and [B,N,4] carrier")
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
