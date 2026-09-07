"""M1 small-B temporal Transformer. Same stack as H1/M2; W=100, out=16."""

from __future__ import annotations

from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import (
    H1Bank,
    H1TemporalFlatDecoder,
    H1TemporalRouteDecoder,
    adamw_param_groups,
    copy_flat_into_route,
    count_h1_decoder_parameters,
    initialize_h1_flat,
    initialize_route_only,
    prove_zero_gate_equals_flat,
    shared_parameter_max_abs_diff,
    whole_unit_dropout,
)

from .m1_config import M1_TEMPORAL, M1TemporalConfig

M1Bank = H1Bank


class M1TemporalFlatDecoder(H1TemporalFlatDecoder):
    name = "M1-TEMPORAL-TRF-FLAT"

    def __init__(self, seed: int = 42, cfg: M1TemporalConfig | None = None) -> None:
        H1TemporalFlatDecoder.__init__(self, seed=seed, cfg=cfg or M1_TEMPORAL)


class M1TemporalRouteDecoder(H1TemporalRouteDecoder):
    name = "M1-TEMPORAL-TRF-ROUTE"

    def __init__(
        self,
        seed: int = 42,
        cfg: M1TemporalConfig | None = None,
        *,
        flat_template: M1TemporalFlatDecoder | None = None,
    ) -> None:
        H1TemporalRouteDecoder.__init__(
            self,
            seed=seed,
            cfg=cfg or M1_TEMPORAL,
            flat_template=flat_template,
        )


def count_m1_decoder_parameters(module) -> dict[str, int]:
    return count_h1_decoder_parameters(module)


__all__ = [
    "M1Bank",
    "M1TemporalFlatDecoder",
    "M1TemporalRouteDecoder",
    "adamw_param_groups",
    "copy_flat_into_route",
    "count_m1_decoder_parameters",
    "initialize_h1_flat",
    "initialize_route_only",
    "prove_zero_gate_equals_flat",
    "shared_parameter_max_abs_diff",
    "whole_unit_dropout",
]
