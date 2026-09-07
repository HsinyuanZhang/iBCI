"""Bind the common full-window QueryTemporalStack into the M1 decoder shell."""
from __future__ import annotations
import torch
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import M1TemporalFlatDecoder, M1TemporalRouteDecoder
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2 import QueryTemporalStack
from . import plan

def _common_stack():
    return QueryTemporalStack(width=256,heads=8,layers=4,ffn=512,window=100,age_buckets=16,seed=plan.SEED)

def build(arm: str="flat"):
    if arm not in {"flat","route"}: raise ValueError(arm)
    # Construct both legacy shells first.  Route's inherited initializer copies
    # the compatible original full-window stack and frontend from this original
    # FLAT shell; replacing FLAT before that copy is an incompatible topology.
    original_flat=M1TemporalFlatDecoder(seed=plan.SEED)
    original_route=M1TemporalRouteDecoder(seed=plan.SEED,flat_template=original_flat)
    shell=original_flat if arm=="flat" else original_route
    old=shell.temporal; common=_common_stack()
    common.initialize_from_full_window(old)
    shell.temporal=common
    return shell

def assert_common_binding(model):
    if not isinstance(model.temporal, QueryTemporalStack):
        raise RuntimeError(f"optimized M1 bound non-common temporal module {type(model.temporal).__module__}")
