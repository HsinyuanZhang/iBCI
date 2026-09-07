"""H1 temporal + routing decoder (owned by H1 worker). Do not put M2 runtime here."""

from .h1_calibration import C2_CKPT_SHA256, C2_EPOCH15_PATH, FrozenC2Materializer, load_frozen_c2_materializer
from .h1_config import H1_TEMPORAL, H1TemporalConfig
from .h1_temporal import (
    H1TemporalFlatDecoder,
    H1TemporalRouteDecoder,
    adamw_param_groups,
    copy_flat_into_route,
    count_h1_decoder_parameters,
    initialize_h1_flat,
    initialize_route_only,
    prove_zero_gate_equals_flat,
)

__all__ = [
    "C2_CKPT_SHA256",
    "C2_EPOCH15_PATH",
    "FrozenC2Materializer",
    "H1_TEMPORAL",
    "H1TemporalConfig",
    "H1TemporalFlatDecoder",
    "H1TemporalRouteDecoder",
    "adamw_param_groups",
    "copy_flat_into_route",
    "count_h1_decoder_parameters",
    "initialize_h1_flat",
    "initialize_route_only",
    "load_frozen_c2_materializer",
    "prove_zero_gate_equals_flat",
]
