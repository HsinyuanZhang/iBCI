"""W1: ridge penalty parameterizations and basis-change equivalence notes.

Equivalence (documented here, not evaluated):

Rescaling coordinates by diagonal ``S`` and applying isotropic penalty ``lambda`` in
the new basis is equivalent to diagonal ridge ``lambda * S^{-2}`` in the original
basis. A learned diagonal ridge is therefore a learned whitening transform.

R1 rotation invariance holds **only** when the penalty is isotropic in the basis
being rotated. Non-isotropic (diagonal) penalties must refuse rotation folds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

PenaltyMode = Literal["fixed_prior", "constant_per_sample"]


@dataclass(frozen=True)
class PenaltySpec:
  """Ridge penalty specification for the 17-parameter design (intercept + q PCs)."""

  mode: PenaltyMode
  ridge_lambda: float
  n_ref: float | None = None
  diagonal: np.ndarray | None = None

  @classmethod
  def fixed_prior(
    cls,
    ridge_lambda: float,
    *,
    diagonal: np.ndarray | None = None,
  ) -> PenaltySpec:
    return cls(mode="fixed_prior", ridge_lambda=float(ridge_lambda), diagonal=diagonal)

  @classmethod
  def constant_per_sample(
    cls,
    ridge_lambda: float,
    *,
    n_ref: float,
    diagonal: np.ndarray | None = None,
  ) -> PenaltySpec:
    return cls(
      mode="constant_per_sample",
      ridge_lambda=float(ridge_lambda),
      n_ref=float(n_ref),
      diagonal=diagonal,
    )

  @property
  def is_isotropic(self) -> bool:
    if self.diagonal is None:
      return True
    diag = np.asarray(self.diagonal, dtype=np.float64).reshape(-1)
    if diag.size == 0:
      return True
    return bool(np.allclose(diag, diag[0], rtol=0.0, atol=0.0))

  def require_isotropic_for_rotation(self) -> None:
    if not self.is_isotropic:
      raise ValueError(
        "rotation fold (R1) requires an isotropic ridge penalty in the rotated basis; "
        "diagonal penalty forfeits exact rotation invariance"
      )


def _penalty_vector(spec: PenaltySpec, design_dim: int = 17) -> np.ndarray:
  if spec.diagonal is None:
    vec = np.ones(design_dim - 1, dtype=np.float64)
  else:
    vec = np.asarray(spec.diagonal, dtype=np.float64).reshape(-1)
    if vec.shape != (design_dim - 1,):
      raise ValueError(f"diagonal penalty must have shape ({design_dim - 1},)")
  return vec


def build_regularizer(spec: PenaltySpec, n: int, design_dim: int = 17) -> np.ndarray:
  """Build the [design_dim, design_dim] ridge matrix (intercept unpenalized)."""
  vec = _penalty_vector(spec, design_dim)
  if spec.mode == "fixed_prior":
    scale = spec.ridge_lambda
  elif spec.mode == "constant_per_sample":
    if spec.n_ref is None or spec.n_ref <= 0:
      raise ValueError("constant_per_sample penalty requires positive n_ref")
    scale = spec.ridge_lambda / spec.n_ref
  else:
    raise ValueError(f"unknown penalty mode: {spec.mode}")

  reg = np.zeros((design_dim, design_dim), dtype=np.float64)
  reg[1:, 1:] = np.diag(scale * vec)
  return reg


def build_system_matrix(
  dtd: np.ndarray,
  n: int,
  spec: PenaltySpec,
) -> np.ndarray:
  """Assemble the normal-equation matrix under the chosen parameterization."""
  reg = build_regularizer(spec, n, design_dim=dtd.shape[0])
  if spec.mode == "fixed_prior":
    return dtd + reg
  if spec.mode == "constant_per_sample":
    if n <= 0:
      raise ValueError("n must be positive for constant_per_sample system")
    return dtd / float(n) + reg
  raise ValueError(f"unknown penalty mode: {spec.mode}")
