"""Learnable causal output filter — frozen-output route (P0-P4).

Implements the frozen-output route of
``docs/DESIGN_LEARNABLE_CAUSAL_OUTPUT_FILTER_20260829.md``: the F0-F4 causal
filter ladder with TRIAL_RESET semantics, source-only filter selection, the §8
adaptive oracles and the P0-P4 experiment sequence over cached frozen
prediction streams.  Inference-only; the sealed Cell-D checkpoint is never
retrained here, and no stage of this package runs a decoder forward.
"""

from .plan import RESULT_ROOT_RELATIVE, SCHEMA

__all__ = ["SCHEMA", "RESULT_ROOT_RELATIVE"]
