"""Named f_eta: source-frozen rSyn3 dictionary with row-L2 gauge + NNLS.

Zero perturbation is the existing NNLS estimator. Training uses the same
forward, with implicit differentiation on the NNLS active set.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3 as bank

from . import plan


class BasisError(RuntimeError):
    """Fail closed for the named behavior basis."""


NNLS_SUPPORT_EPS = plan.P_NNLS_SUPPORT_EPS


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise BasisError(f"{name} must be finite")


def _row_normalize(dictionary: torch.Tensor) -> torch.Tensor:
    nonnegative = torch.relu(dictionary)
    norms = torch.linalg.vector_norm(nonnegative, dim=1, keepdim=True)
    if bool((norms <= 0).any()):
        raise BasisError("NNMF dictionary row vanished")
    return nonnegative / norms


def _assert_dictionary_full_rank(dictionary: torch.Tensor) -> None:
    rank = int(torch.linalg.matrix_rank(dictionary.to(dtype=torch.float64), atol=1.0e-8, rtol=0.0))
    if rank < int(dictionary.shape[0]):
        raise BasisError("degenerate dictionary")


def nnls_with_active_set(dictionary: torch.Tensor, scaled: torch.Tensor) -> torch.Tensor:
    """NNLS forward via the bank, then torch LS on the support for autograd."""
    if dictionary.ndim != 2 or scaled.ndim != 2 or dictionary.shape[1] != scaled.shape[1]:
        raise BasisError(f"NNLS shapes D={tuple(dictionary.shape)} Y={tuple(scaled.shape)}")
    if int(dictionary.shape[0]) != int(plan.M1_RANK):
        raise BasisError("f_eta rank must stay 3")
    _assert_finite("dictionary", dictionary)
    _assert_finite("scaled_emg", scaled)
    _assert_dictionary_full_rank(dictionary)
    d_np = np.asarray(dictionary.detach().cpu().numpy(), dtype=np.float64)
    y_np = np.asarray(scaled.detach().cpu().numpy(), dtype=np.float64)
    z_np = bank.nnls_activations(y_np, d_np)
    z = scaled.new_zeros((scaled.shape[0], dictionary.shape[0]))
    eps = float(NNLS_SUPPORT_EPS)
    for index in range(scaled.shape[0]):
        support = z_np[index] > eps
        if not bool(np.any(support)):
            continue
        mask = torch.as_tensor(support, device=dictionary.device)
        subset = dictionary[mask]
        gram = subset @ subset.transpose(0, 1)
        rhs = subset @ scaled[index]
        try:
            coef = torch.linalg.solve(gram, rhs)
        except RuntimeError as error:
            raise BasisError(f"active-set solve failed at row {index}: {error}") from error
        z[index, mask] = coef
    return z


def nnls_contract_report(
    dictionary: torch.Tensor,
    scaled: torch.Tensor,
    z: torch.Tensor,
) -> dict[str, object]:
    """SciPy parity, nonnegativity, KKT residual. Never silently ridge-repairs NNLS."""
    d_np = np.asarray(dictionary.detach().cpu().numpy(), dtype=np.float64)
    y_np = np.asarray(scaled.detach().cpu().numpy(), dtype=np.float64)
    z_np = np.asarray(z.detach().cpu().numpy(), dtype=np.float64)
    z_scipy = bank.nnls_activations(y_np, d_np)
    eps = float(NNLS_SUPPORT_EPS)
    kkt_vals: list[float] = []
    empty = True
    for index in range(z_np.shape[0]):
        recon = z_np[index] @ d_np
        grad = d_np @ (recon - y_np[index])
        support = z_np[index] > eps
        if bool(np.any(support)):
            empty = False
            kkt_vals.append(float(np.max(np.abs(grad[support]))))
        inactive = np.logical_not(support)
        if bool(np.any(inactive)):
            kkt_vals.append(float(max(0.0, float(-np.min(grad[inactive])))))
    scipy_near = bool(np.any((z_scipy > 0.0) & (z_scipy <= 1.0e-8)))
    torch_near = bool(np.any((z_np > 0.0) & (z_np <= 1.0e-8)))
    input_tiny = bool(float(np.max(np.abs(y_np))) <= 1.0e-8) if y_np.size else True
    return {
        "support_eps": eps,
        "scipy_forward_max_abs": float(np.max(np.abs(z_np - z_scipy))) if z_np.size else 0.0,
        "nonnegative": bool(np.all(z_np >= -1.0e-12)),
        "kkt_residual": float(max(kkt_vals)) if kkt_vals else 0.0,
        "empty_active_set": bool(empty),
        "near_boundary_disclosed": bool(scipy_near or torch_near or input_tiny),
        "ridge_repair_used": False,
    }


class RowNormalizedNNMFBasis(nn.Module):
    """z = NNLS(relu(y) / scale, row_normalize(relu(D)))."""

    name = "row_normalized_nnmf_nnls_v1"

    def __init__(
        self,
        dictionary: torch.Tensor,
        scale: torch.Tensor,
        *,
        trainable: bool,
    ) -> None:
        super().__init__()
        if dictionary.ndim != 2 or tuple(dictionary.shape) != (plan.M1_RANK, plan.M1_EMG_DIM):
            raise BasisError(f"dictionary must be {(plan.M1_RANK, plan.M1_EMG_DIM)}, got {tuple(dictionary.shape)}")
        if scale.ndim != 1 or int(scale.shape[0]) != plan.M1_EMG_DIM:
            raise BasisError(f"scale must be ({plan.M1_EMG_DIM},)")
        self.raw_dictionary = nn.Parameter(dictionary.to(dtype=torch.float64), requires_grad=bool(trainable))
        self.register_buffer("scale", scale.to(dtype=torch.float64))

    def constrained_dictionary(self) -> torch.Tensor:
        return _row_normalize(self.raw_dictionary)

    def encode(self, y_support: torch.Tensor) -> torch.Tensor:
        return encode_from_raw(self.raw_dictionary, self.scale, y_support)


def encode_from_raw(
    raw_dictionary: torch.Tensor,
    scale: torch.Tensor,
    y_support: torch.Tensor,
) -> torch.Tensor:
    """Same encode law without wrapping the dictionary as a new Parameter."""
    if y_support.ndim != 2 or int(y_support.shape[1]) != plan.M1_EMG_DIM:
        raise BasisError(f"y must be (T,{plan.M1_EMG_DIM})")
    _assert_finite("emg", y_support)
    _assert_finite("scale", scale)
    _assert_finite("dictionary", raw_dictionary)
    if bool((scale < plan.M1_SCALE_FLOOR).any()):
        raise BasisError(f"scale below floor {plan.M1_SCALE_FLOOR}")
    rectified = torch.relu(y_support)
    scaled = rectified / scale
    return nnls_with_active_set(_row_normalize(raw_dictionary), scaled)
