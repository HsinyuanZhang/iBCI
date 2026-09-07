"""Affine miscalibration diagnostics (EXECUTION_GUIDANCE §8, queue item 5).

Zero-training, pure-numpy diagnostics over frozen last-bin prediction streams:

* per-session six-vector affine corrections ``[A00, A01, A10, A11, b0, b1]``
  fitted by closed-form least squares (target labels => oracle rows are
  leakage-labelled and never selection/deployment eligible);
* dispersion of the corrections across the sessions of a cell (covariance
  trace, leading eigenvalue, median distance to the mean, first-PC fraction);
* the deployable diagnostic ``r0_gain``: the mean of the OTHER sessions'
  corrections (leave-one-out, zero target-label fitting on the held session)
  applied unchanged to the held session.
"""
from .affine import (
    apply_affine,
    cross_surface_r0_rows,
    dispersion_stats,
    fit_affine,
    full_opportunity_rows,
    loo_r0_rows,
    r2_float64,
    six_vector_payload,
)

__all__ = [
    "apply_affine",
    "cross_surface_r0_rows",
    "dispersion_stats",
    "fit_affine",
    "full_opportunity_rows",
    "loo_r0_rows",
    "r2_float64",
    "six_vector_payload",
]
