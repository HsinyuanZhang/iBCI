"""The only V6 variable is V4 query attention's fixed read-time recency prior."""
from __future__ import annotations
import torch
from tfpd_exploration.src.h1_optimized_v4.model import H1SignedQuery
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v4 import RecencyPriorQueryTemporalStack

FRONTEND_CONTRACT_VERSION=4;TEMPORAL_CONTRACT_VERSION=4
class H1RecencyPriorQuery(H1SignedQuery):
 def __init__(self):
  super().__init__();c=self.cfg;self.temporal=RecencyPriorQueryTemporalStack(width=c.temporal_width,heads=c.heads,layers=c.layers,ffn=c.ffn,window=c.window,age_buckets=16,seed=42)
def make_matched_pair(*,activity_scale=1.):
 if activity_scale!=1.:raise ValueError('V6 retains raw-count V4 frontend contract')
 from tfpd_exploration.src.h1_optimized_v4.model import make_matched_pair as v4
 full,_=v4(activity_scale=1.);query=H1RecencyPriorQuery()
 for key in ('frontend','final_norm','readout'):getattr(query,key).load_state_dict(getattr(full,key).state_dict(),strict=True)
 query.temporal.initialize_from_full_window(full.temporal)
 if int(full.frontend_contract_version)!=4 or int(query.frontend_contract_version)!=4 or int(query.temporal.temporal_contract_version)!=4:raise RuntimeError('V6 contract drift')
 return full,query
