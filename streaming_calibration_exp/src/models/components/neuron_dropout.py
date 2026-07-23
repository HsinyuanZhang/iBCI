"""Structured neuron dropout strategies for calibration encoder training.

All strategies return a binary mask of shape ``[B, N]`` where ``1`` means the
neuron survives and ``0`` means it is dropped (zeroed in both calibration trials
and the online neural window).

Drop ratios are intentionally capped at 0.3 — this simulates realistic chronic
electrode degradation (typically 10-30% unit loss over months), NOT extreme
scenarios. Larger drops would discard too much primary information.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch


# Hard cap on dropout probability. NEVER exceed this in any strategy.
# 30% simulates worst-case realistic chronic degradation.
MAX_NEURON_DROPOUT_P = 0.30


def _validate_p_range(p_low: float, p_high: float) -> None:
  if not (0.0 <= p_low <= p_high <= MAX_NEURON_DROPOUT_P):
    raise ValueError(
      f"Dropout range [{p_low}, {p_high}] invalid; require "
      f"0 <= p_low <= p_high <= {MAX_NEURON_DROPOUT_P}"
    )


class NeuronDropoutStrategy(ABC):
  """Base class. Subclasses implement :meth:`sample_mask`."""

  def __init__(self, p_low: float, p_high: float) -> None:
    _validate_p_range(p_low, p_high)
    self.p_low = p_low
    self.p_high = p_high

  @abstractmethod
  def sample_mask(self, batch_size: int, num_neurons: int, device: torch.device) -> torch.Tensor:
    """Return ``[B, N]`` float tensor with values in {0.0, 1.0}."""
    raise NotImplementedError

  def extra_repr(self) -> str:
    return f"p_low={self.p_low}, p_high={self.p_high}"

  def __repr__(self) -> str:
    return f"{type(self).__name__}({self.extra_repr()})"


class NoDropout(NeuronDropoutStrategy):
  """Passthrough; mask is all ones."""

  def __init__(self) -> None:
    super().__init__(0.0, 0.0)

  def sample_mask(self, batch_size: int, num_neurons: int, device: torch.device) -> torch.Tensor:
    return torch.ones(batch_size, num_neurons, device=device)


class UniformNeuronDropout(NeuronDropoutStrategy):
  """Each neuron independently dropped with probability ``p ~ U(p_low, p_high)``.

  A single ``p`` is sampled per forward pass (per batch) and applied to all
  samples in the batch. This mirrors SPINT's original ``dynamic_dropout``
  behaviour where one ``p`` is drawn per forward call.
  """

  def sample_mask(self, batch_size: int, num_neurons: int, device: torch.device) -> torch.Tensor:
    p = float(torch.empty(()).uniform_(self.p_low, self.p_high).item())
    mask = torch.ones(batch_size, num_neurons, device=device)
    if p > 0.0:
      mask = torch.nn.functional.dropout(mask, p=p, training=True)
    return mask


class BlockNeuronDropout(NeuronDropoutStrategy):
  """Drop contiguous blocks of ``block_size`` adjacent neurons.

  Models the failure of physically adjacent electrodes on a Utah array.
  Within each block the entire block is either kept or dropped (Bernoulli with
  probability ``p ~ U(p_low, p_high)``). ``p`` is the per-block drop rate; the
  expected fraction of neurons dropped is approximately ``p``.
  """

  def __init__(self, p_low: float, p_high: float, block_size: int = 4) -> None:
    super().__init__(p_low, p_high)
    if block_size < 1:
      raise ValueError(f"block_size must be >= 1, got {block_size}")
    self.block_size = block_size

  def extra_repr(self) -> str:
    return f"p_low={self.p_low}, p_high={self.p_high}, block_size={self.block_size}"

  def sample_mask(self, batch_size: int, num_neurons: int, device: torch.device) -> torch.Tensor:
    p = float(torch.empty(()).uniform_(self.p_low, self.p_high).item())
    bs = self.block_size
    num_blocks = (num_neurons + bs - 1) // bs
    # Block-level Bernoulli mask [B, num_blocks]
    block_keep = torch.bernoulli(
      torch.full((batch_size, num_blocks), 1.0 - p, device=device)
    )
    # Expand blocks to neurons [B, num_blocks, bs] -> [B, num_neurons]
    block_mask = block_keep.unsqueeze(-1).expand(batch_size, num_blocks, bs)
    mask = block_mask.reshape(batch_size, num_blocks * bs)[:, :num_neurons]
    return mask.contiguous()


class CurriculumDropout(NeuronDropoutStrategy):
  """Wrap a base strategy and ramp its effective ``p_high`` from 0 to target.

  During the warmup (first ``warmup_epochs`` epochs), the drop probability is
  linearly scaled from 0 to the base strategy's configured range. This avoids
  destabilising early training when the encoder has not yet learned useful
  representations.

  The ``current_epoch`` must be set externally via :meth:`set_epoch` (called
  from the Lightning module's ``on_train_epoch_start`` hook).
  """

  def __init__(
    self,
    base: NeuronDropoutStrategy,
    warmup_epochs: int = 10,
  ) -> None:
    super().__init__(base.p_low, base.p_high)
    if warmup_epochs < 0:
      raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
    self.base = base
    self.warmup_epochs = warmup_epochs
    self._current_epoch = 0

  def set_epoch(self, epoch: int) -> None:
    self._current_epoch = max(0, int(epoch))

  def _scale(self) -> float:
    if self.warmup_epochs <= 0:
      return 1.0
    return min(1.0, self._current_epoch / float(self.warmup_epochs))

  def extra_repr(self) -> str:
    return (
      f"base={type(self.base).__name__}, "
      f"p_low={self.p_low}, p_high={self.p_high}, "
      f"warmup_epochs={self.warmup_epochs}, epoch={self._current_epoch}"
    )

  def sample_mask(self, batch_size: int, num_neurons: int, device: torch.device) -> torch.Tensor:
    scale = self._scale()
    if scale <= 0.0:
      return torch.ones(batch_size, num_neurons, device=device)
    # Temporarily scale the base strategy's range
    orig_low, orig_high = self.base.p_low, self.base.p_high
    self.base.p_low = orig_low * scale
    self.base.p_high = orig_high * scale
    try:
      return self.base.sample_mask(batch_size, num_neurons, device)
    finally:
      self.base.p_low = orig_low
      self.base.p_high = orig_high


def build_neuron_dropout(
  mode: str = "none",
  p_low: float = 0.0,
  p_high: float = 0.3,
  block_size: int = 4,
  warmup_epochs: int = 10,
) -> NeuronDropoutStrategy:
  """Factory for dropout strategies. Mirrors the ``build_encoder`` pattern."""
  mode = mode.lower()
  if mode == "none":
    return NoDropout()
  if mode == "uniform":
    return UniformNeuronDropout(p_low=p_low, p_high=p_high)
  if mode == "block":
    return BlockNeuronDropout(p_low=p_low, p_high=p_high, block_size=block_size)
  if mode == "curriculum":
    base = UniformNeuronDropout(p_low=p_low, p_high=p_high)
    return CurriculumDropout(base=base, warmup_epochs=warmup_epochs)
  if mode == "curriculum_block":
    base = BlockNeuronDropout(p_low=p_low, p_high=p_high, block_size=block_size)
    return CurriculumDropout(base=base, warmup_epochs=warmup_epochs)
  raise ValueError(f"Unknown neuron_dropout mode: {mode}")


def apply_mask_to_calib(calib: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  """Broadcast-multiply a ``[B, N]`` mask over ``[B, M, T, N]`` calibration trials."""
  if calib.dim() != 4:
    raise ValueError(f"Expected calib [B,M,T,N], got {tuple(calib.shape)}")
  if mask.dim() != 2:
    raise ValueError(f"Expected mask [B,N], got {tuple(mask.shape)}")
  if mask.shape != calib.shape[:1] + calib.shape[-1:]:
    raise ValueError(
      f"mask shape {tuple(mask.shape)} incompatible with calib {tuple(calib.shape)}"
    )
  return calib * mask.unsqueeze(1).unsqueeze(2)


def apply_mask_to_neural(neural: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  """Broadcast-multiply a ``[B, N]`` mask over ``[B, W, N]`` online neural window."""
  if neural.dim() != 3:
    raise ValueError(f"Expected neural [B,W,N], got {tuple(neural.shape)}")
  if mask.dim() != 2:
    raise ValueError(f"Expected mask [B,N], got {tuple(mask.shape)}")
  if mask.shape != neural.shape[:1] + neural.shape[-1:]:
    raise ValueError(
      f"mask shape {tuple(mask.shape)} incompatible with neural {tuple(neural.shape)}"
    )
  return neural * mask.unsqueeze(1)


def masked_identity_mse(
  e_student: torch.Tensor,
  e_teacher: torch.Tensor,
  mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  """Identity MSE restricted to surviving neurons.

  Args:
    e_student: ``[B, N, W]`` student identity.
    e_teacher: ``[B, N, W]`` teacher identity (computed with all neurons).
    mask: ``[B, N]`` binary mask. If ``None`` or all-ones, behaves as the
      normalised identity MSE used elsewhere in the codebase.

  Returns:
    Scalar loss. The normalisation denom is still computed over the full
    teacher identity (so the scale is comparable to the unmasked metric).
  """
  if e_student.shape != e_teacher.shape:
    raise ValueError(
      f"e_student {tuple(e_student.shape)} != e_teacher {tuple(e_teacher.shape)}"
    )
  denom = (e_teacher ** 2).mean().clamp_min(1e-8)
  squared = (e_student - e_teacher) ** 2
  if mask is None:
    return squared.mean() / denom
  if mask.dim() != 2:
    raise ValueError(f"Expected mask [B,N], got {tuple(mask.shape)}")
  # Broadcast mask [B,N] -> [B,N,W]
  m = mask.unsqueeze(-1).to(e_student.dtype)
  # Average over surviving entries only; use the same denom scale for
  # comparability with the unmasked metric.
  surviving = m.sum().clamp_min(1.0)
  return (squared * m).sum() / surviving / denom
