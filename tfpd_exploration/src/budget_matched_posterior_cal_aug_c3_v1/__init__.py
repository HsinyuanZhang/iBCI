"""Angular-reliability successor to budget-matched posterior CAL-AUG C2."""

from .features import (
    C3Arm,
    BudgetMatchedReliabilityDataset,
    ReliabilityNormalizer,
    fit_source_reliability_normalizer,
    widen_cell_d_for_reliability,
)

__all__ = [
    "C3Arm",
    "BudgetMatchedReliabilityDataset",
    "ReliabilityNormalizer",
    "fit_source_reliability_normalizer",
    "widen_cell_d_for_reliability",
]
