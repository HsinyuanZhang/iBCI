"""M2 Stage-O oracle carrier headroom test (V1).

One inference-only, zero-training CPU replay on top of the sealed LOCAL-M2
governing run ``cdm_p1_m2_local_v1``: the missing Stage-O prerequisite that
DANDI had before its P1 program.

Cells
-----
* ``O0m``  -- the sealed F00m activity/frozen-T4 law verbatim (the baseline);
* ``O2m``  -- the SAME activity state plus the TRUE completed-trial direction
  fed to the support-anchored block refit under the Stage-O amendment law
  (always-commit + trust-region projection only, no three-factor rejection
  semantics, ``rho_M = 1.0``, block = 1 trial).

The true direction is diagnostic target-label leakage; nothing in this package
can select, promote or deploy anything.
"""

from .plan import (
    ALPHA_M_PRIMARY,
    ALPHA_M_SENSITIVITY,
    BUDGETS,
    HELD_OUT_SURFACE,
    M30,
    SELECTION_SURFACE,
    SURFACES,
    VERDICT_GO,
    VERDICT_NULL,
)

__all__ = [
    "ALPHA_M_PRIMARY",
    "ALPHA_M_SENSITIVITY",
    "BUDGETS",
    "HELD_OUT_SURFACE",
    "M30",
    "SELECTION_SURFACE",
    "SURFACES",
    "VERDICT_GO",
    "VERDICT_NULL",
]
