"""Isolated carrier quantization / numerics candidate stack.

Not imported by any active experiment path. See README.md.
"""

from .reference import (
    AccumulationState,
    accumulate,
    accumulate_blocks,
    fit_carrier_batch,
    solve_carrier,
)
from .folds import (
    apply_rotation_fold,
    apply_smoothquant_resplit,
    hadamard_rotation,
    random_orthogonal,
    smoothquant_scales,
)
from .penalty import PenaltySpec, build_regularizer, build_system_matrix
from .audit import run_audit, FOLD0_DATE_FORBIDDEN
from .fixedpoint import (
    allocate_bits_reliability_weighted,
    allocate_bits_uniform,
    simulate_quantized_carrier,
    ldl_solve,
)

__all__ = [
    "AccumulationState",
    "accumulate",
    "accumulate_blocks",
    "fit_carrier_batch",
    "solve_carrier",
    "apply_rotation_fold",
    "apply_smoothquant_resplit",
    "hadamard_rotation",
    "random_orthogonal",
    "smoothquant_scales",
    "PenaltySpec",
    "build_regularizer",
    "build_system_matrix",
    "run_audit",
    "FOLD0_DATE_FORBIDDEN",
    "allocate_bits_reliability_weighted",
    "allocate_bits_uniform",
    "simulate_quantized_carrier",
    "ldl_solve",
]
