"""Differentiable M1 bank ridge. Matches NumPy fit_unit_ridge /n + slope penalty."""
from __future__ import annotations

import torch

from . import plan


class CarrierSolverError(RuntimeError):
    """Fail closed for the torch ridge contract."""


def solve_ridge(
    z: torch.Tensor,
    rates: torch.Tensor,
    ridge_lambda: float = plan.M1_RIDGE_LAMBDA,
) -> torch.Tensor:
    """Return [N, rank+1] carriers as [slopes..., intercept].

    System: (X^T X / n + Lambda) beta = X^T rates / n
    with X = [1, z] and Lambda = diag([0, λ, ..., λ]).
    """
    if z.ndim != 2 or rates.ndim != 2 or z.shape[0] != rates.shape[0] or z.shape[0] < 1:
        raise CarrierSolverError(f"ridge shapes z={tuple(z.shape)} rates={tuple(rates.shape)}")
    if not torch.isfinite(z).all() or not torch.isfinite(rates).all():
        raise CarrierSolverError("ridge input nonfinite")
    n = float(z.shape[0])
    ones = torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)
    design = torch.cat((ones, z), dim=1)
    gram = (design.T @ design) / n
    penalty = torch.zeros(design.shape[1], dtype=z.dtype, device=z.device)
    penalty[1:] = float(ridge_lambda)
    rhs = (design.T @ rates) / n
    try:
        beta = torch.linalg.solve(gram + torch.diag(penalty), rhs)
    except RuntimeError as error:
        raise CarrierSolverError(f"ridge solve failed: {error}") from error
    if not torch.isfinite(beta).all():
        raise CarrierSolverError("ridge nonfinite")
    weights = beta[1:].transpose(0, 1)
    intercepts = beta[0].unsqueeze(-1)
    return torch.cat((weights, intercepts), dim=-1)
