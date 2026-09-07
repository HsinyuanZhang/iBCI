"""Behavior-manifold v2: five closed-form measurements completing the M1 route.

Reuses behavior_autoencoder_v1 loaders, frozen manifolds, and the governing
scorer by import.  Nothing here trains on a target session: every experiment is
either the frozen deployment rule refit at a different support budget or an
explicitly labelled oracle leakage diagnostic.
"""
from .protocol import (
    aligned_budget,
    budget_normalizers,
    direct_ridge_budget,
    fit_score_budget_ridge,
    full_session_arrays,
    write_receipt,
)

__all__ = [
    "aligned_budget",
    "budget_normalizers",
    "direct_ridge_budget",
    "fit_score_budget_ridge",
    "full_session_arrays",
    "write_receipt",
]
