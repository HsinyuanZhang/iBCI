"""Fresh M2 CRST-B4 FW-QueryAge16 family.

This package is intentionally additive.  It does not alter the frozen M2
CRST-B4 implementation or the reusable M1 current-query operator.
"""

from .model import (
    FAMILY_NAME_FLAT,
    FAMILY_NAME_ROUTE,
    M2QueryAgeFamilyDecoder,
    make_paired_queryage_decoders,
    shared_parameter_max_abs_diff,
)

__all__ = [
    "FAMILY_NAME_FLAT",
    "FAMILY_NAME_ROUTE",
    "M2QueryAgeFamilyDecoder",
    "make_paired_queryage_decoders",
    "shared_parameter_max_abs_diff",
]
