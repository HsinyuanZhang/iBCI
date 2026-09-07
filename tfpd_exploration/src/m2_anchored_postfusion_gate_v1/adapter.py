"""Torch-only APFG scalar adapter.

This module is intentionally separate from the inert import path.  A future
runtime must strict-load the selected-T4 POOLED student *before* calling
``freeze_and_install`` so state-dict keys remain the historical native keys.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from . import plan


class AdapterError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterError(message)


class AnchoredPostFusionGate(nn.Module):
    """A native B3S identity plus exactly one trainable scalar residual gate."""

    def __init__(self, native: nn.Module) -> None:
        super().__init__()
        _require(getattr(native, "variant", None) == "B3S", "APFG requires native B3S encoder")
        _require(int(getattr(native, "trial_length", -1)) == plan.TRIAL_LENGTH, "B3S trial length drift")
        _require(int(getattr(native, "window_size", -1)) == plan.WINDOW_BINS, "B3S window length drift")
        _require(int(getattr(native, "side_dim", -1)) == plan.SIDE_DIM, "B3S T4 side dimension drift")
        _require(int(getattr(native, "electrode_embed_dim", 0)) == 0,
                 "APFG V1 does not authorize electrode embeddings")
        self.native = native
        self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.variant = "B3S_APFG"
        self.trial_length = int(native.trial_length)
        self.window_size = int(native.window_size)
        self.hidden_dim = int(native.hidden_dim)
        self.side_dim = int(native.side_dim)
        self.electrode_embed_dim = 0
        # The enclosing student stays in eval permanently.  This explicit flag
        # selects a differentiable residual expression for source alpha steps
        # without toggling inherited dropout/BatchNorm into training mode.
        self._alpha_training_enabled = False

    def set_alpha_training(self, enabled: bool) -> None:
        self._alpha_training_enabled = bool(enabled)
        self.native.eval()

    def _postfusion_identity(self, calib_trials: torch.Tensor, side_features: torch.Tensor | None) -> torch.Tensor:
        _require(calib_trials.ndim == 4 and int(calib_trials.shape[2]) == self.trial_length,
                 "APFG calibration must be [B,M,100,N]")
        _require(side_features is not None and int(side_features.shape[-1]) == self.side_dim,
                 "APFG requires frozen four-dimensional T4 side feature")
        # Same arrival-order float32 summation used by the frozen post-fusion
        # probe; torch.mean is intentionally not an authority-equivalent form.
        per_trial = self.native.pre_pool(calib_trials.permute(0, 1, 3, 2))
        side = side_features.unsqueeze(1).expand(-1, int(per_trial.shape[1]), -1, -1)
        values = self.native.post_pool(torch.cat((per_trial, side), dim=-1))
        total = values[:, 0]
        for index in range(1, int(values.shape[1])):
            total = total + values[:, index]
        return total / int(values.shape[1])

    def forward_batch(self, calib_trials: torch.Tensor, trial_lengths: torch.Tensor | None = None,
                      side_features: torch.Tensor | None = None,
                      electrode_ids: torch.Tensor | None = None) -> torch.Tensor:
        _require(electrode_ids is None, "APFG V1 does not authorize electrode ids")
        native = self.native.forward_batch(calib_trials, trial_lengths=trial_lengths,
                                           side_features=side_features, electrode_ids=None)
        # In evaluation only, exact +0.0 is an *operational* anchor route.  In
        # training retain the residual expression so alpha has a first-step
        # gradient even from its prescribed zero initialization.
        if not self._alpha_training_enabled and exact_positive_zero(self.alpha):
            return native
        post = self._postfusion_identity(calib_trials, side_features)
        return native + torch.tanh(self.alpha) * (post - native)

    def reset_stream(self, *args: Any, **kwargs: Any) -> Any:
        return self.native.reset_stream(*args, **kwargs)

    def push_trial(self, *args: Any, **kwargs: Any) -> Any:
        return self.native.push_trial(*args, **kwargs)

    def finalize_identity(self, *args: Any, **kwargs: Any) -> Any:
        return self.native.finalize_identity(*args, **kwargs)


def freeze_and_install(student: Any) -> AnchoredPostFusionGate:
    """Install post-load APFG and make alpha the only trainable parameter."""
    _require(getattr(student, "_decoder_frozen", False), "APFG requires the sealed decoder already frozen")
    for parameter in student.parameters():
        parameter.requires_grad = False
        parameter.grad = None
    adapter = AnchoredPostFusionGate(student.id_encoder)
    student.id_encoder = adapter
    _require(tuple(name for name, parameter in student.named_parameters() if parameter.requires_grad) == ("id_encoder.alpha",),
             "APFG trainable topology is not exactly scalar alpha")
    return adapter


def alpha_only_adam(adapter: AnchoredPostFusionGate) -> torch.optim.Adam:
    trainable = tuple(parameter for parameter in adapter.parameters() if parameter.requires_grad)
    _require(trainable == (adapter.alpha,), "APFG optimizer would include inherited parameter")
    return torch.optim.Adam(
        trainable, lr=plan.ADAM_LR, weight_decay=plan.ADAM_WEIGHT_DECAY,
        betas=plan.ADAM_BETAS, eps=plan.ADAM_EPS, amsgrad=plan.ADAM_AMSGRAD,
    )


def parameter_evidence(adapter: AnchoredPostFusionGate) -> dict[str, object]:
    named = tuple((name, parameter) for name, parameter in adapter.named_parameters())
    trainable = tuple(name for name, parameter in named if parameter.requires_grad)
    _require(trainable == ("alpha",), "APFG inherited parameter became trainable")
    _require(adapter.alpha.shape == torch.Size([]) and adapter.alpha.dtype == torch.float32,
             "APFG alpha scalar/dtype drift")
    return {"trainable_parameter_names": list(trainable), "alpha_shape": [],
            "alpha_dtype": str(adapter.alpha.dtype),
            "alpha_exact_positive_zero": exact_positive_zero(adapter.alpha),
            "alpha_training_enabled": bool(adapter._alpha_training_enabled),
            "inherited_parameter_count": len(named) - 1}


def exact_positive_zero(value: torch.Tensor) -> bool:
    """Recognize only IEEE +0.0; ``torch.equal(-0,+0)`` is insufficient."""
    _require(value.shape == torch.Size([]), "APFG alpha must remain scalar")
    detached = value.detach()
    return bool(detached.item() == 0.0 and not torch.signbit(detached).item())


__all__ = ("AdapterError", "AnchoredPostFusionGate", "freeze_and_install", "alpha_only_adam", "parameter_evidence",
           "exact_positive_zero")
