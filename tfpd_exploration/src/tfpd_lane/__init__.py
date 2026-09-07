"""TFPD lane package: post-bilinear consumer extension deliverables.

Deliverable A (HANDOFF_POST_BILINEAR_CONSUMER_EXTENSION_20260815.md §5.5):
zero-GPU pre-gate for the time-varying weighting premise -> `pregate`.

Deliverable B (§5.7): forward-only mechanism diagnostics over a frozen
Stage-1 checkpoint -> `mech_diag`.

This package adds files only; it never modifies `src/tfpd/` or the sealed
Stage-0/Stage-1 runners.
"""

from src.tfpd_lane import mech_diag, pregate, receipt

__all__ = ["mech_diag", "pregate", "receipt"]
