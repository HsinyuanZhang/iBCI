"""Correspondence-breaking training augmentations (B4).

Applies deterministic unit permutation, subsetting, or pseudo-electrode
pooling consistently across query neural windows, calibration trials, and
side features.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch


CorrespondenceBreakingMode = Literal["none", "permute", "subset", "electrode_pool"]


@dataclass(frozen=True)
class CorrespondenceBreakingReceipt:
  mode: str
  requested_seed: int
  resolved_seed: int
  session_name: str
  permutation: Optional[np.ndarray] = None
  kept_indices: Optional[np.ndarray] = None
  electrode_channel_ids: Optional[np.ndarray] = None


def _derived_seed(*, seed: int, session_name: str, mode: str) -> int:
  payload = f"CorrespondenceBreaking-v1:{mode}:{session_name}:seed={seed}".encode("utf-8")
  return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def deterministic_unit_permutation(
  num_units: int,
  *,
  seed: int,
  session_name: str,
) -> np.ndarray:
  """Session-salted deterministic permutation over unit indices."""
  if num_units < 1:
    raise ValueError("permutation requires at least one unit")
  derived = _derived_seed(seed=seed, session_name=session_name, mode="permute")
  permutation = np.random.RandomState(derived).permutation(num_units)
  if num_units > 1 and np.array_equal(permutation, np.arange(num_units)):
    permutation = np.roll(permutation, 1)
  return permutation


def deterministic_unit_subset(
  num_units: int,
  *,
  keep_fraction: float,
  seed: int,
  session_name: str,
) -> np.ndarray:
  """Return sorted indices of units kept after deterministic subsetting."""
  if not (0.0 < keep_fraction <= 1.0):
    raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
  keep_count = max(1, int(round(num_units * keep_fraction)))
  derived = _derived_seed(seed=seed, session_name=session_name, mode="subset")
  order = np.random.RandomState(derived).permutation(num_units)
  kept = np.sort(order[:keep_count])
  return kept


def _permute_last_dim(tensor: torch.Tensor, permutation: np.ndarray) -> torch.Tensor:
  index = torch.as_tensor(permutation, device=tensor.device, dtype=torch.long)
  return tensor.index_select(-1, index)


def _permute_unit_dim(tensor: torch.Tensor, permutation: np.ndarray) -> torch.Tensor:
  """Permute the unit/channel axis for ``[B, N, ...]`` tensors."""
  index = torch.as_tensor(permutation, device=tensor.device, dtype=torch.long)
  return tensor.index_select(1, index)


def _subset_unit_dim(tensor: torch.Tensor, kept: np.ndarray) -> torch.Tensor:
  index = torch.as_tensor(kept, device=tensor.device, dtype=torch.long)
  return tensor.index_select(1, index)


def _pool_trial_rates_by_electrode(
  trial_rates: np.ndarray,
  electrode_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
  """Local copy of ``unit_side_features.pool_trial_rates_by_electrode``."""
  if trial_rates.ndim != 2:
    raise ValueError(f"Expected trial rates [units, trials], got {trial_rates.shape}")
  if electrode_ids.ndim != 1 or electrode_ids.shape[0] != trial_rates.shape[0]:
    raise ValueError(
      "electrode_ids must contain exactly one value per unit; got "
      f"{electrode_ids.shape} for {trial_rates.shape[0]} units"
    )
  channel_ids, inverse = np.unique(electrode_ids, return_inverse=True)
  pooled = np.zeros((channel_ids.size, trial_rates.shape[1]), dtype=np.float64)
  np.add.at(pooled, inverse, trial_rates)
  return pooled, channel_ids


def _pool_unit_axis_numpy(
  values: np.ndarray,
  electrode_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
  """Pool the last axis of ``values`` with electrode summation."""
  if values.ndim < 2:
    raise ValueError(f"expected at least 2 dims, got {values.shape}")
  leading = values.shape[:-1]
  units = values.shape[-1]
  flat = values.reshape(-1, units)
  pooled, channel_ids = _pool_trial_rates_by_electrode(flat.T, electrode_ids)
  return pooled.T.reshape(*leading, channel_ids.size), channel_ids


def _average_side_features_by_electrode(
  side_features: np.ndarray,
  electrode_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
  channel_ids, inverse = np.unique(electrode_ids, return_inverse=True)
  pooled = np.zeros((channel_ids.size, side_features.shape[-1]), dtype=side_features.dtype)
  counts = np.zeros(channel_ids.size, dtype=np.float64)
  np.add.at(pooled, inverse, side_features)
  np.add.at(counts, inverse, 1.0)
  counts = np.maximum(counts, 1.0)
  return pooled / counts[:, None], channel_ids


def apply_correspondence_breaking(
  neural: torch.Tensor,
  calib: torch.Tensor,
  side_features: Optional[torch.Tensor],
  electrode_ids: Optional[torch.Tensor],
  *,
  mode: CorrespondenceBreakingMode = "none",
  seed: int = 0,
  session_name: str = "",
  keep_fraction: float = 0.75,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], CorrespondenceBreakingReceipt]:
  """Apply augmentation consistently across all unit-aligned tensors."""
  receipt = CorrespondenceBreakingReceipt(
    mode=mode,
    requested_seed=int(seed),
    resolved_seed=_derived_seed(seed=seed, session_name=session_name, mode=mode),
    session_name=session_name,
  )
  if mode == "none":
    return neural, calib, side_features, electrode_ids, receipt

  num_units = neural.shape[-1]
  if calib.shape[-1] != num_units:
    raise ValueError("calib neuron dim must match neural")
  if side_features is not None and side_features.shape[-2] != num_units:
    raise ValueError("side_features unit dim must match neural")
  if electrode_ids is not None and electrode_ids.shape[-1] != num_units:
    raise ValueError("electrode_ids must have one id per unit")

  if mode == "permute":
    permutation = deterministic_unit_permutation(
      num_units, seed=seed, session_name=session_name
    )
    receipt = CorrespondenceBreakingReceipt(
      mode=mode,
      requested_seed=int(seed),
      resolved_seed=_derived_seed(seed=seed, session_name=session_name, mode=mode),
      session_name=session_name,
      permutation=permutation,
    )
    neural_out = _permute_last_dim(neural, permutation)
    calib_out = _permute_last_dim(calib, permutation)
    side_out = None if side_features is None else _permute_unit_dim(side_features, permutation)
    elec_out = None if electrode_ids is None else _permute_last_dim(electrode_ids, permutation)
    return neural_out, calib_out, side_out, elec_out, receipt

  if mode == "subset":
    kept = deterministic_unit_subset(
      num_units, keep_fraction=keep_fraction, seed=seed, session_name=session_name
    )
    receipt = CorrespondenceBreakingReceipt(
      mode=mode,
      requested_seed=int(seed),
      resolved_seed=_derived_seed(seed=seed, session_name=session_name, mode=mode),
      session_name=session_name,
      kept_indices=kept,
    )
    neural_out = _subset_last_dim(neural, kept)
    calib_out = _subset_last_dim(calib, kept)
    side_out = None if side_features is None else _subset_unit_dim(side_features, kept)
    elec_out = None if electrode_ids is None else _subset_last_dim(electrode_ids, kept)
    return neural_out, calib_out, side_out, elec_out, receipt

  if mode == "electrode_pool":
    if electrode_ids is None:
      raise ValueError("electrode_pool requires electrode_ids")
    elec_np = electrode_ids.detach().cpu().numpy()
    if elec_np.ndim == 2:
      if elec_np.shape[0] != neural.shape[0]:
        raise ValueError("electrode_ids batch must match neural batch")
      if not np.all(elec_np == elec_np[:1]):
        raise ValueError("electrode_pool requires identical electrode_ids across batch")
      elec_np = elec_np[0]
    neural_np = neural.detach().cpu().numpy()
    calib_np = calib.detach().cpu().numpy()
    pooled_neural, channel_ids = _pool_unit_axis_numpy(neural_np, elec_np)
    pooled_calib, _ = _pool_unit_axis_numpy(calib_np, elec_np)
    side_out = None
    if side_features is not None:
      side_np = side_features.detach().cpu().numpy()
      pooled_side_batches = []
      for batch_idx in range(side_np.shape[0]):
        pooled_side, _ = _average_side_features_by_electrode(side_np[batch_idx], elec_np)
        pooled_side_batches.append(pooled_side)
      side_out = torch.as_tensor(
        np.stack(pooled_side_batches, axis=0),
        device=side_features.device,
        dtype=side_features.dtype,
      )
    neural_out = torch.as_tensor(pooled_neural, device=neural.device, dtype=neural.dtype)
    calib_out = torch.as_tensor(pooled_calib, device=calib.device, dtype=calib.dtype)
    elec_out = torch.as_tensor(channel_ids, device=electrode_ids.device).view(1, -1)
    elec_out = elec_out.expand(neural.shape[0], -1)
    receipt = CorrespondenceBreakingReceipt(
      mode=mode,
      requested_seed=int(seed),
      resolved_seed=_derived_seed(seed=seed, session_name=session_name, mode=mode),
      session_name=session_name,
      electrode_channel_ids=channel_ids,
    )
    return neural_out, calib_out, side_out, elec_out, receipt

  raise ValueError(f"Unknown correspondence_breaking mode: {mode}")
