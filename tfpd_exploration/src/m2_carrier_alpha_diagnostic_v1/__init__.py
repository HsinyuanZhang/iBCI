"""M2 P1-carrier alpha_M post-hoc diagnostic (V1).

One inference-only, zero-training CPU replay of the sealed LOCAL-M2 governing
run ``cdm_p1_m2_local_v1``: the F01m carrier law at ``alpha_M in {0.125, 0.5}``
(plus ``alpha_M = 0`` as the reproduction anchor) on both surfaces, every
other hyperparameter bound verbatim to the governing selection, to test
whether the within-fold choice of ``alpha_M = 0`` matched held-out reality.

This run can never select a hyperparameter, promote a cell, or update any
model or checkpoint; every ``alpha_M > 0`` row is target-label-leaking
post-hoc evidence and is labelled as such.
"""

from .plan import (
    ALPHA_ALL,
    ALPHA_ANCHOR,
    ALPHA_SCAN,
    BUDGETS,
    HELD_OUT_SURFACE,
    SELECTION_SURFACE,
    SURFACES,
    VERDICT_MATCHED,
    VERDICT_MISMATCH,
)

__all__ = [
    "ALPHA_ALL",
    "ALPHA_ANCHOR",
    "ALPHA_SCAN",
    "BUDGETS",
    "HELD_OUT_SURFACE",
    "SELECTION_SURFACE",
    "SURFACES",
    "VERDICT_MATCHED",
    "VERDICT_MISMATCH",
]
