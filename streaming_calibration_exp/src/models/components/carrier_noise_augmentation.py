"""Estimator-noise augmentation for T4 carriers (B3).

Perturbs only the ``[a, c]`` block using the analytically known small-``M``
OLS covariance ``sigma^2 (X'X)^-1``.  Modulation depth ``m`` is always
recomputed as ``hypot(a, c)`` after perturbation and is never perturbed
independently.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

# Frozen center-out directions (matches unit_side_features.CANONICAL_DIRECTIONS_RAD).
_CANONICAL_DIRECTIONS_RAD: Tuple[float, ...] = tuple(
  -3.0 * math.pi / 4.0 + k * (math.pi / 4.0) for k in range(8)
)


def ac_block_covariance_from_design(
  direction_indices: np.ndarray,
  *,
  residual_variance: float,
) -> np.ndarray:
  """Return the 2x2 covariance of ``[a, c]`` from labelled trial-level rates."""
  directions = np.asarray(direction_indices, dtype=np.int64)
  valid = directions >= 0
  if int(valid.sum()) < 3:
    raise ValueError("carrier noise requires at least three valid labelled trials")
  theta = np.asarray(
    [_CANONICAL_DIRECTIONS_RAD[int(index)] for index in directions[valid]],
    dtype=np.float64,
  )
  design = np.stack([np.ones_like(theta), np.cos(theta), np.sin(theta)], axis=1)
  rank = int(np.linalg.matrix_rank(design))
  if rank != 3:
    raise ValueError(f"carrier noise requires rank-3 design; got rank={rank}")
  inv_xtx = np.linalg.inv(design.T @ design)
  return float(residual_variance) * inv_xtx[1:3, 1:3]


def ac_block_cholesky(
  covariance: np.ndarray,
  *,
  eps: float = 1.0e-12,
) -> np.ndarray:
  """Cholesky factor of the ``[a, c]`` covariance block."""
  cov = np.asarray(covariance, dtype=np.float64)
  if cov.shape != (2, 2):
    raise ValueError(f"expected [2,2] covariance, got {cov.shape}")
  jitter = eps * np.eye(2, dtype=np.float64)
  return np.linalg.cholesky(cov + jitter)


def recompute_t4_modulation(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
  """Return ``m = hypot(a, c)`` with the same shape as ``a``."""
  return torch.hypot(a, c)


def perturb_t4_carrier(
  carrier: torch.Tensor,
  *,
  cholesky_ac: torch.Tensor,
  scale: float,
  generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
  """Add Gaussian noise to ``[a, c]`` and recompute ``m``.

  Args:
    carrier: ``[..., 4]`` tensor in T4 order ``(a, c, m, b)``.
    cholesky_ac: lower-triangular Cholesky of the ``[a, c]`` covariance.
    scale: noise multiplier.  ``0`` returns ``carrier`` unchanged (bitwise).
    generator: optional RNG for reproducibility.
  """
  if scale <= 0.0:
    return carrier
  if carrier.shape[-1] != 4:
    raise ValueError(f"T4 carrier must have last dim 4, got {carrier.shape}")
  if cholesky_ac.shape[-2:] != (2, 2):
    raise ValueError(f"cholesky_ac must be [...,2,2], got {cholesky_ac.shape}")

  out = carrier.clone()
  a = out[..., 0]
  c = out[..., 1]
  noise = torch.randn(
    a.shape + (2,),
    device=carrier.device,
    dtype=carrier.dtype,
    generator=generator,
  )
  delta = torch.matmul(noise, cholesky_ac.to(device=carrier.device, dtype=carrier.dtype).transpose(-1, -2)) * float(scale)
  a_new = a + delta[..., 0]
  c_new = c + delta[..., 1]
  out[..., 0] = a_new
  out[..., 1] = c_new
  out[..., 2] = recompute_t4_modulation(a_new, c_new)
  return out


def perturb_t4_carrier_numpy(
  carrier: np.ndarray,
  *,
  cholesky_ac: np.ndarray,
  scale: float,
  seed: Optional[int] = None,
) -> np.ndarray:
  """NumPy helper for unit tests and offline covariance checks."""
  tensor = torch.as_tensor(carrier, dtype=torch.float64)
  chol = torch.as_tensor(cholesky_ac, dtype=torch.float64)
  generator = None
  if seed is not None:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
  result = perturb_t4_carrier(
    tensor,
    cholesky_ac=chol,
    scale=scale,
    generator=generator,
  )
  return result.detach().cpu().numpy()


def sample_perturbed_carrier_batch(
  carrier: torch.Tensor,
  cholesky_ac: torch.Tensor,
  *,
  scale: float,
  generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
  """Batch wrapper accepting per-unit Cholesky factors ``[B, N, 2, 2]``."""
  if scale <= 0.0:
    return carrier
  if cholesky_ac.shape[:-2] != carrier.shape[:-1]:
    raise ValueError(
      f"cholesky batch {cholesky_ac.shape[:-2]} != carrier batch {carrier.shape[:-1]}"
    )
  flat_carrier = carrier.reshape(-1, 4)
  flat_chol = cholesky_ac.reshape(-1, 2, 2)
  flat_out = perturb_t4_carrier(
    flat_carrier,
    cholesky_ac=flat_chol,
    scale=scale,
    generator=generator,
  )
  return flat_out.reshape(carrier.shape)
