"""Trainable post-fusion B3S identity variants.

Torch is intentionally imported only here, never by the package initializer or
the public dry CLI.  The adapter wraps a freshly constructed sealed B3S encoder
and retains its exact pre/post-pool parameter objects; each screen arm receives
an independent enclosing model/state, not shared mutable optimizer state.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from . import plan


class VariantError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VariantError(message)


class PostFusionIdentityAdapter(nn.Module):
    """Post-fusion mean or zero-initialized native/post-fusion residual adapter."""

    def __init__(self, native: nn.Module, arm: str) -> None:
        super().__init__()
        _require(arm in plan.ARMS, "unknown postfusion arm")
        _require(getattr(native, "variant", None) == "B3S", "postfusion route requires B3S native encoder")
        _require(int(getattr(native, "trial_length", -1)) == plan.TRIAL_LENGTH,
                 "B3S trial-length drift")
        _require(int(getattr(native, "window_size", -1)) == plan.WINDOW_BINS,
                 "B3S window-size drift")
        _require(int(getattr(native, "side_dim", -1)) == plan.SIDE_DIM, "B3S T4 side dimension drift")
        self.native = native
        self.arm = arm
        # Kept as ordinary attributes so model code that checks encoder geometry
        # sees the original sealed values.
        self.variant = "B3S"
        self.trial_length = int(native.trial_length)
        self.window_size = int(native.window_size)
        self.hidden_dim = int(native.hidden_dim)
        self.side_dim = int(native.side_dim)
        self.electrode_embed_dim = int(getattr(native, "electrode_embed_dim", 0))
        if arm == "PF-R1":
            self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32))
        elif arm == "PF-R50":
            self.alpha = nn.Parameter(torch.zeros((plan.WINDOW_BINS,), dtype=torch.float32))
        else:
            self.register_parameter("alpha", None)

    def _postfusion_identity(self, calib_trials: torch.Tensor, side_features: torch.Tensor | None,
                             electrode_ids: torch.Tensor | None) -> torch.Tensor:
        _require(calib_trials.ndim == 4 and int(calib_trials.shape[2]) == self.trial_length,
                 "calibration tensor must be [B,M,100,N]")
        _require(side_features is not None, "B3S postfusion requires T4 side features")
        _require(self.electrode_embed_dim == 0 and electrode_ids is None,
                 "V1 postfusion screen does not authorize electrode embeddings")
        # [B,M,T,N] -> [B,M,N,T] -> [B,M,N,H]
        per_trial_features = self.native.pre_pool(calib_trials.permute(0, 1, 3, 2))
        side = side_features.unsqueeze(1).expand(-1, per_trial_features.shape[1], -1, -1)
        values = self.native.post_pool(torch.cat((per_trial_features, side), dim=-1))
        # The frozen probe's authority is *arrival-order float32 addition*,
        # then one division.  ``torch.mean`` is mathematically identical but
        # may choose a different reduction tree, so it is not an acceptable
        # receipt-equivalent spelling here.
        total = values[:, 0]
        for index in range(1, int(values.shape[1])):
            total = total + values[:, index]
        return total / int(values.shape[1])

    def forward_batch(self, calib_trials: torch.Tensor, trial_lengths: torch.Tensor | None = None,
                      side_features: torch.Tensor | None = None,
                      electrode_ids: torch.Tensor | None = None) -> torch.Tensor:
        native = self.native.forward_batch(
            calib_trials, trial_lengths=trial_lengths, side_features=side_features,
            electrode_ids=electrode_ids,
        )
        postfusion = self._postfusion_identity(calib_trials, side_features, electrode_ids)
        if self.arm == "PF-MEAN":
            return postfusion
        assert self.alpha is not None
        # At alpha==0 this is bitwise native in the approved CPU constructibility
        # proof, while keeping alpha's first-step derivative equal to the branch
        # difference.  Do not branch on zero: that would sever alpha's gradient.
        return native + torch.tanh(self.alpha) * (postfusion - native)

    # These delegations only satisfy the base encoder interface during training
    # setup.  They are *not* a post-fusion continual scorer: that scorer must
    # explicitly use the probe's per-trial identity pool rather than claiming
    # native pre-fusion stream state has changed placement.
    def reset_stream(self, *args: Any, **kwargs: Any) -> Any:
        return self.native.reset_stream(*args, **kwargs)

    def push_trial(self, *args: Any, **kwargs: Any) -> Any:
        return self.native.push_trial(*args, **kwargs)

    def finalize_identity(self, *args: Any, **kwargs: Any) -> Any:
        return self.native.finalize_identity(*args, **kwargs)


def install_variant(student: Any, arm: str) -> PostFusionIdentityAdapter:
    """Replace only a fresh student's B3S encoder before optimizer creation."""
    _require(getattr(student, "_decoder_frozen", False), "postfusion route requires frozen decoder")
    adapter = PostFusionIdentityAdapter(student.id_encoder, arm)
    student.id_encoder = adapter
    return adapter


def added_parameter_count(adapter: PostFusionIdentityAdapter) -> int:
    if adapter.alpha is None:
        return 0
    return int(adapter.alpha.numel())


def parameter_topology(adapter: PostFusionIdentityAdapter) -> dict[str, object]:
    expected = plan.ARM_ADDED_PARAMETERS[adapter.arm]
    observed = added_parameter_count(adapter)
    _require(observed == expected, "postfusion alpha parameter topology drift")
    return {
        "arm": adapter.arm,
        "added_parameter_count": observed,
        "alpha_shape": None if adapter.alpha is None else list(adapter.alpha.shape),
        "alpha_exact_positive_zero": (adapter.alpha is None or bool(torch.equal(adapter.alpha.detach(), torch.zeros_like(adapter.alpha)))),
        "native_pre_pool_parameter_ids": [id(parameter) for parameter in adapter.native.pre_pool.parameters()],
        "native_post_pool_parameter_ids": [id(parameter) for parameter in adapter.native.post_pool.parameters()],
    }


__all__ = ("VariantError", "PostFusionIdentityAdapter", "install_variant", "added_parameter_count", "parameter_topology")
