"""V3 alias of the frozen V2 capacity harness with an isolated output root."""
from __future__ import annotations
from tfpd_exploration.src.h1_optimized_v2 import capacity_probe as base
from .model import make_matched_pair
base.OUT=base.CACHE_ROOT/'capacity_probe_208_source_v3'
base.make_matched_pair=make_matched_pair
if __name__=='__main__': base.main()
