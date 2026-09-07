"""P2' oracle-policy decomposition matrix over the frozen Cell-D CDM evaluator.

Implements the reviewer-mandated matrix of DESIGN_LEARNED_GATE_SMOOTHED_
PSEUDOLABEL_20260828.md sections 10.11/10.12 (with the constraints of 10.2,
10.4, 10.7, 10.8).  Inference-level only: the sealed Cell-D SWA checkpoint is
never retrained, no target optimizer/backward runs, and the CDM state machine
is imported from the accepted V8/V2/activity-only implementations.
"""

from .plan import CELL, RESULT_ROOT_RELATIVE, SMOKE_RESULT_ROOT_RELATIVE

__all__ = ["CELL", "RESULT_ROOT_RELATIVE", "SMOKE_RESULT_ROOT_RELATIVE"]
