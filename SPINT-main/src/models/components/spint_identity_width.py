"""Architecture-preserving width-scaled SPINT activity identity encoder.

This module deliberately leaves the SPINT read-in, representation tokens,
cross-attention decoder, loss call-site, and M=4 activity input semantics
unchanged.  It changes only the hidden width used inside the original
``fc_id_in -> mean(trials) -> fc_id_out`` MLP.

For ``identity_width=1024`` the module has the same parameter names, shapes,
initialisation order, and forward computation as :class:`SpintModel`.  Smaller
widths retain all six affine stages and the two ReLU stages on each side of the
mean-over-trials operation; only their internal feature dimension changes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn

from src.models.components.spint import MultiLayerCrossAttention


REFERENCE_IDENTITY_WIDTH = 1024


class SpintIdentityWidthError(ValueError):
    """Raised when a width-scaled SPINT identity contract is violated."""


@dataclass(frozen=True)
class SpintIdentityWidthAccounting:
    """Static parameter and dense-MAC accounting for the identity path only."""

    identity_width: int
    parameters: int
    dense_macs_m4_n176: int


def identity_parameter_count(identity_width: int, *, trial_length: int = 1024, window_size: int = 700) -> int:
    """Return original-SPINT identity MLP parameters, including all biases.

    The preserved topology is ``T->h->h->h``, mean over the four calibration
    trials, then ``h->h->h->W``.  This is deliberately not the compact
    CarrierID topology.
    """

    h, t, w = int(identity_width), int(trial_length), int(window_size)
    if h <= 0 or t <= 0 or w <= 0:
        raise SpintIdentityWidthError("identity width, trial length, and window size must be positive")
    # fc_id_in: T*h+h + 2*(h*h+h); fc_id_out: 2*(h*h+h) + h*W+W.
    return (t * h + h) + 4 * (h * h + h) + (h * w + w)


def identity_dense_macs(
    identity_width: int,
    *,
    support_trials: int = 4,
    neurons: int = 176,
    trial_length: int = 1024,
    window_size: int = 700,
) -> int:
    """Return dense Linear weight multiply-adds for one M=4 session token.

    Biases, nonlinearities, mean pooling, dropout, and all non-identity SPINT
    decoder operations are intentionally excluded.  The convention matches the
    earlier H1 SPINT receipt, but here it applies to an activity-only width
    sweep.
    """

    h, m, n, t, w = (
        int(identity_width), int(support_trials), int(neurons), int(trial_length), int(window_size)
    )
    if min(h, m, n, t, w) <= 0:
        raise SpintIdentityWidthError("MAC dimensions must be positive")
    # Trial-side: M*N*(T*h + 2*h*h); post-pool side: N*(2*h*h + h*W).
    return m * n * (t * h + 2 * h * h) + n * (2 * h * h + h * w)


def identity_accounting(identity_width: int) -> SpintIdentityWidthAccounting:
    """Return fixed H1 M=4 accounting for one predeclared identity width."""

    h = int(identity_width)
    return SpintIdentityWidthAccounting(
        identity_width=h,
        parameters=identity_parameter_count(h),
        dense_macs_m4_n176=identity_dense_macs(h),
    )


class SpintIdentityWidthModel(nn.Module):
    """Original SPINT with width scaled only inside the activity identity MLP.

    ``identity_width=1024`` is intentionally a literal H-S reference.  To
    isolate the width intervention for compact arms, the non-identity SPINT
    backbone is initialized from the same RNG stream it would receive in that
    H-S reference.  This initialization control does not alter any runtime
    computation or optimizer semantics; it merely prevents changing the ID
    width from also shifting the decoder's random initial state.
    """

    def __init__(
        self,
        model_dim: int,
        num_covariates: int,
        window_size: int,
        *,
        identity_width: int,
        num_heads: int = 2,
        num_layers: int = 1,
        num_id_layers: int = 3,
        use_learnable_id: bool = True,
        learnable_id_type: str = "mlp",
        learnable_rep: bool = True,
        dropout_rate: float = 0.0,
        dynamic_dropout: bool = False,
        dynamic_dropout_low: float = 0.0,
        dynamic_dropout_high: float = 1.0,
        tf_drop_rate: float = 0.1,
        readin_layer_type: str = "mlp",
    ) -> None:
        super().__init__()
        if int(model_dim) != REFERENCE_IDENTITY_WIDTH:
            raise SpintIdentityWidthError("H1 width sweep fixes the downstream SPINT model_dim=1024")
        if int(identity_width) <= 0 or int(identity_width) > REFERENCE_IDENTITY_WIDTH:
            raise SpintIdentityWidthError("identity_width must be in [1,1024]")
        if int(num_id_layers) != 3:
            raise SpintIdentityWidthError("width sweep preserves exactly three ID layers on each side of pooling")
        if not use_learnable_id or str(learnable_id_type) != "mlp":
            raise SpintIdentityWidthError("width sweep requires the original learnable MLP identity path")
        if str(readin_layer_type) != "mlp":
            raise SpintIdentityWidthError("width sweep preserves the original MLP read-in")

        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.num_id_layers = int(num_id_layers)
        self.num_covariates = int(num_covariates)
        self.window_size = int(window_size)
        self.identity_width = int(identity_width)
        self.use_learnable_id = bool(use_learnable_id)
        self.learnable_id_type = str(learnable_id_type)
        self.learnable_rep = bool(learnable_rep)
        self.dropout_rate = float(dropout_rate)
        self.tf_drop_rate = float(tf_drop_rate)
        self.dynamic_dropout = bool(dynamic_dropout)
        self.dynamic_dropout_low = float(dynamic_dropout_low)
        self.dynamic_dropout_high = float(dynamic_dropout_high)
        self.readin_layer_type = str(readin_layer_type)

        # This ordering intentionally starts exactly as SpintModel.
        self.fc_in = nn.Sequential(
            nn.Linear(self.window_size, self.model_dim),
            nn.ReLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        state_after_readin = torch.get_rng_state()

        # The compact ID layers are initialized from the H-S identity stream,
        # then the outer RNG is advanced as though a width-1024 identity MLP
        # had been created.  The shared decoder is consequently bit-identical
        # across width arms at common seed, while the actual ID tensors are the
        # only tensors whose shapes/initial values differ.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(state_after_readin)
            self.fc_id_in, self.fc_id_out = self._make_identity_layers(self.identity_width)

        # Advance the real CPU RNG through the exact H-S (h=1024) initialized
        # non-lazy ID layers.  LazyLinear itself consumes no random values.
        for _ in range(2):
            nn.Linear(REFERENCE_IDENTITY_WIDTH, REFERENCE_IDENTITY_WIDTH)
        for _ in range(2):
            nn.Linear(REFERENCE_IDENTITY_WIDTH, REFERENCE_IDENTITY_WIDTH)
        nn.Linear(REFERENCE_IDENTITY_WIDTH, self.window_size)

        # This remaining construction is a literal copy/order of SpintModel.
        self.fc_out = nn.Linear(self.model_dim, self.window_size)
        if self.learnable_rep:
            self.rep = nn.Parameter(torch.randn(1, self.num_covariates, self.window_size))
        else:
            self.rep = (
                torch.arange(start=1, end=self.num_covariates + 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(1, 1, self.window_size)
                / self.num_covariates
            )
        self.transformer = MultiLayerCrossAttention(
            num_layers=self.num_layers,
            d_model=self.model_dim,
            nhead=self.num_heads,
            dropout=self.tf_drop_rate,
        )

    def _make_identity_layers(self, width: int) -> tuple[nn.Sequential, nn.Sequential]:
        in_layers: list[nn.Module] = [nn.LazyLinear(width)]
        for _ in range(self.num_id_layers - 1):
            in_layers.extend([nn.ReLU(), nn.Linear(width, width)])
        out_layers: list[nn.Module] = []
        for _ in range(self.num_id_layers - 1):
            out_layers.extend([nn.Linear(width, width), nn.ReLU()])
        out_layers.append(nn.Linear(width, self.window_size))
        return nn.Sequential(*in_layers), nn.Sequential(*out_layers)

    def identity_projection(self, calib_trialized_neural_features: torch.Tensor) -> torch.Tensor:
        """Build an activity-only per-unit token from exactly the supplied trials."""

        if calib_trialized_neural_features.ndim != 4:
            raise SpintIdentityWidthError(
                "SPINT identity must be [batch, support_trials, trial_length, neurons]"
            )
        identity = calib_trialized_neural_features.permute(0, 1, 3, 2)
        if identity.shape[-1] != 1024:
            raise SpintIdentityWidthError(
                f"SPINT width identity trial length must be 1024, got {identity.shape[-1]}"
            )
        identity = self.fc_id_in(identity)
        identity = torch.mean(identity, dim=1, keepdim=False)
        return self.fc_id_out(identity)

    def forward(self, src: torch.Tensor, calib_trialized_neural_features: torch.Tensor | None = None) -> torch.Tensor:
        if calib_trialized_neural_features is None:
            raise SpintIdentityWidthError("activity-only SPINT width model requires calibration neural activity")
        src = src.permute(0, 2, 1)
        batch_size, num_neurons = src.size(0), src.size(1)
        identity = self.identity_projection(calib_trialized_neural_features)
        if identity.shape != src.shape:
            raise SpintIdentityWidthError(
                f"identity/source shape mismatch: identity={tuple(identity.shape)} source={tuple(src.shape)}"
            )
        src = src + identity

        dropout_mask = torch.ones(batch_size, num_neurons).to(src)
        if self.dynamic_dropout:
            probability = random.uniform(self.dynamic_dropout_low, self.dynamic_dropout_high)
            dropout_mask = torch.nn.functional.dropout(dropout_mask, p=probability, training=self.training)
        else:
            dropout_mask = torch.nn.functional.dropout(dropout_mask, p=self.dropout_rate, training=self.training)
        src = src * dropout_mask.unsqueeze(-1)

        src = self.fc_in(src)
        rep = self.fc_in(self.rep).to(src)
        output, _ = self.transformer(rep.repeat(batch_size, 1, 1), src)
        return self.fc_out(output).permute(0, 2, 1)

    def identity_parameter_count(self) -> int:
        observed = sum(parameter.numel() for name, parameter in self.named_parameters() if name.startswith(("fc_id_in.", "fc_id_out.")))
        expected = identity_parameter_count(self.identity_width, trial_length=1024, window_size=self.window_size)
        if observed != expected:
            raise SpintIdentityWidthError(f"identity parameter accounting drift: {observed} != {expected}")
        return observed

    def identity_dense_macs_m4_n176(self) -> int:
        return identity_dense_macs(self.identity_width, support_trials=4, neurons=176, trial_length=1024, window_size=self.window_size)
