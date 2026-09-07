"""R1 rotation fold and S1 SmoothQuant-style scale re-split (exact plan-time folds)."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol

import numpy as np

from .penalty import PenaltySpec


class MutablePlan(Protocol):
  scale: np.ndarray
  pcs: np.ndarray
  q: int


def hadamard_rotation(q: int) -> np.ndarray:
  """Sylvester Hadamard construction; exact orthogonal matrix for q a power of two."""
  if q < 1 or (q & (q - 1)) != 0:
    raise ValueError(f"Hadamard rotation requires q to be a power of two, got {q}")
  h = np.array([[1.0]], dtype=np.float64)
  while h.shape[0] < q:
    h = np.block([[h, h], [h, -h]])
  return h / np.sqrt(q)


def random_orthogonal(q: int, seed: int) -> np.ndarray:
  rng = np.random.default_rng(seed)
  mat = rng.standard_normal((q, q))
  q_mat, _ = np.linalg.qr(mat)
  if np.linalg.det(q_mat) < 0:
    q_mat[:, 0] *= -1.0
  return q_mat


def apply_rotation_fold(plan: Any, rotation: np.ndarray, *, penalty: PenaltySpec | None = None) -> Any:
  """R1: fold orthogonal rotation into frozen PCs; runtime carrier unchanged (isotropic penalty)."""
  if penalty is not None:
    penalty.require_isotropic_for_rotation()
  r = np.asarray(rotation, dtype=np.float64)
  q = plan.q
  if r.shape != (q, q):
    raise ValueError(f"rotation must be [{q}, {q}]")
  if not np.allclose(r.T @ r, np.eye(q), atol=1e-12):
    raise ValueError("rotation must be orthogonal")

  pcs = np.asarray(plan.pcs, dtype=np.float64).copy()
  pcs[:q] = r.T @ pcs[:q]
  return replace(plan, pcs=pcs)


def apply_smoothquant_resplit(plan: Any, target_scale: np.ndarray) -> Any:
  """S1: re-split activation/weight scale between ``scale`` and ``pcs`` (exact)."""
  s = np.asarray(target_scale, dtype=np.float64).reshape(-1)
  scale = np.asarray(plan.scale, dtype=np.float64)
  if s.shape != scale.shape:
    raise ValueError("target_scale must match plan.scale shape")
  if np.any(s <= 0):
    raise ValueError("target_scale entries must be positive")

  ratio = s / scale
  pcs = np.asarray(plan.pcs, dtype=np.float64).copy()
  pcs *= ratio[None, :]
  return replace(plan, scale=s, pcs=pcs)


def smoothquant_scales(
  activation_max: np.ndarray,
  weight_max: np.ndarray,
  alpha: float,
) -> np.ndarray:
  """SmoothQuant selection: s_j = max|X_j|^alpha / max|W_j|^(1-alpha)."""
  if not 0.0 <= alpha <= 1.0:
    raise ValueError("alpha must be in [0, 1]")
  x_max = np.asarray(activation_max, dtype=np.float64).reshape(-1)
  w_max = np.asarray(weight_max, dtype=np.float64).reshape(-1)
  if x_max.shape != w_max.shape:
    raise ValueError("activation_max and weight_max must have the same shape")
  eps = 1.0e-30
  return np.power(np.maximum(x_max, eps), alpha) / np.power(np.maximum(w_max, eps), 1.0 - alpha)
