"""RT L-D explicit carrier-to-live-activity gain.

This is deliberately a very small, decoder-side operator.  It does not alter
the identity encoder or the SPINT decoder: for every unit ``i`` it applies

``x'_{i,t} = x_{i,t} * (1 + g_t(carrier_i))``.

The bias-free, zero-initialised projection is part of the experimental
contract, rather than a convenience default.  It makes the gain arm an exact
null of the additive Full arm at step zero while leaving a 200-parameter
carrier-to-window map for learning.  ``g`` outputs the complete live window.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RtLdLiveGainState:
  """Calibration-derived per-unit, per-live-bin gain cache."""

  factor: torch.Tensor

  def __post_init__(self) -> None:
    if self.factor.ndim != 3:
      raise ValueError(
        "RT L-D gain state must have shape [B,N,W], got "
        f"{tuple(self.factor.shape)}"
      )

  @property
  def nbytes(self) -> int:
    return self.factor.numel() * self.factor.element_size()


class CarrierLiveActivityGain(nn.Module):
  """Bias-free carrier projection used only to modulate live activity."""

  def __init__(self, carrier_dim: int = 4, window_size: int = 50) -> None:
    super().__init__()
    if carrier_dim <= 0:
      raise ValueError("carrier_dim must be positive")
    if window_size <= 0:
      raise ValueError("window_size must be positive")
    self.carrier_dim = int(carrier_dim)
    self.window_size = int(window_size)
    # Do not instantiate nn.Linear and then zero it: its reset_parameters()
    # would advance the global CPU RNG after the otherwise common A0/G
    # backbone.  This is exactly the same bias-free affine map as
    # Linear(4,50,bias=False), expressed as an already-zero Parameter so it
    # consumes no RNG at construction.
    self.weight = nn.Parameter(torch.zeros(self.window_size, self.carrier_dim))

  def _validate_carrier(self, carrier: torch.Tensor) -> None:
    if carrier.ndim != 3 or carrier.shape[-1] != self.carrier_dim:
      raise ValueError(
        "RT L-D carrier must have shape [B,N,"
        f"{self.carrier_dim}], got {tuple(carrier.shape)}"
      )

  def derive_state(self, carrier: torch.Tensor) -> RtLdLiveGainState:
    self._validate_carrier(carrier)
    return RtLdLiveGainState(factor=1.0 + F.linear(carrier, self.weight, bias=None))

  def apply_state(self, live_activity: torch.Tensor, state: RtLdLiveGainState) -> torch.Tensor:
    if live_activity.ndim != 3:
      raise ValueError(
        "RT L-D live activity must have shape [B,N,W], got "
        f"{tuple(live_activity.shape)}"
      )
    if state.factor.shape != live_activity.shape:
      raise ValueError("RT L-D gain state must exactly match live activity [B,N,W]")
    return live_activity * state.factor

  def forward(self, live_activity: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
    return self.apply_state(live_activity, self.derive_state(carrier))

  def receipt(self, *, batch_size: int, num_units: int) -> dict[str, object]:
    if batch_size <= 0 or num_units <= 0:
      raise ValueError("batch_size and num_units must be positive")
    scalar_count = batch_size * num_units
    return {
      "schema_version": 1,
      "operator": "x_prime_i_t = x_i_t * (1 + g_t(carrier_i))",
      "carrier_dim": self.carrier_dim,
      "window_size": self.window_size,
      "bias": False,
      "zero_initialized": bool(torch.count_nonzero(self.weight) == 0),
      "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
      "parameter_names": ["weight"],
      "calibration_only_macs": {
        "carrier_projection": scalar_count * self.carrier_dim * self.window_size
      },
      # This is an entire W=50 live decode window.  Do not multiply by W a
      # second time: the factor tensor is already [B,N,W].
      "online_macs_per_decode_window": {
        "live_activity_gain": scalar_count * self.window_size
      },
      "persistent_state": {
        "fields": ["factor"],
        "shape": [batch_size, num_units, self.window_size],
        "bytes_fp32": scalar_count * self.window_size * 4,
        "excludes": ["carrier", "identity", "calibration_trials", "raw_activity"],
      },
      "excluded_operations": ["elementwise_add", "state read/write"],
    }
