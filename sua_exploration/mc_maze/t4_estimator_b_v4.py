"""Append-only v4 binding: v3 estimator mechanics plus reconciled ordinal provenance.

Historical Step-2A ``ordinals`` are selected rewarded-pool positions, whereas current
``trial_index`` is the raw NWB trial-table index.  v4 never conflates them: both are stored and
the selected-trial identity remains fail-closed on start/stop/direction equality.
"""
from mc_maze.t4_estimator_b_v3 import *  # noqa: F401,F403

ORDINAL_SEMANTICS_VERSION="selected_rewarded_pool_ordinal_plus_raw_nwb_trial_index_v1"
