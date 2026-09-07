"""Activity-path dropout: zero the pooled activity vector before side-feature concat.

Unlike neuron dropout (which drops individual neurons and degrades both paths),
activity-path dropout forces the identity encoder to rely on the carrier alone
with probability ``p`` by zeroing the trial-averaged pooled activity vector
``mean_feat`` while leaving side features untouched.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class ActivityPathDropoutStrategy(ABC):
  """Base class. Subclasses implement :meth:`sample_mask`."""

  def __init__(self, p: float) -> None:
    if not (0.0 <= p <= 1.0):
      raise ValueError(f"activity-path dropout p must be in [0, 1], got {p}")
    self.p = p

  @abstractmethod
  def sample_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
    """Return ``[B, 1]`` float tensor broadcastable over hidden dim."""
    raise NotImplementedError

  def extra_repr(self) -> str:
    return f"p={self.p}"

  def __repr__(self) -> str:
    return f"{type(self).__name__}({self.extra_repr()})"


class NoActivityPathDropout(ActivityPathDropoutStrategy):
  """Passthrough; mask is all ones."""

  def __init__(self) -> None:
    super().__init__(0.0)

  def sample_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.ones(batch_size, 1, device=device)


class BernoulliActivityPathDropout(ActivityPathDropoutStrategy):
  """Per-batch-element Bernoulli mask over the pooled activity vector."""

  def sample_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
    if self.p <= 0.0:
      return torch.ones(batch_size, 1, device=device)
    if self.p >= 1.0:
      return torch.zeros(batch_size, 1, device=device)
    return torch.bernoulli(
      torch.full((batch_size, 1), 1.0 - self.p, device=device)
    )


def build_activity_path_dropout(*, p: float = 0.0) -> ActivityPathDropoutStrategy:
  """Factory mirroring ``build_neuron_dropout``."""
  if p <= 0.0:
    return NoActivityPathDropout()
  return BernoulliActivityPathDropout(p=p)


def apply_activity_path_mask(pooled: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  """Broadcast-multiply a ``[B, 1]`` mask over ``[B, N, H]`` pooled activity."""
  if pooled.dim() != 3:
    raise ValueError(f"Expected pooled [B,N,H], got {tuple(pooled.shape)}")
  if mask.dim() != 2 or mask.shape[1] != 1:
    raise ValueError(f"Expected mask [B,1], got {tuple(mask.shape)}")
  if mask.shape[0] != pooled.shape[0]:
    raise ValueError(
      f"mask batch {mask.shape[0]} incompatible with pooled batch {pooled.shape[0]}"
    )
  return pooled * mask.unsqueeze(-1)
