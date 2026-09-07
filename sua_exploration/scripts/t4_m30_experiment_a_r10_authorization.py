"""Fixed-path r10 authorization binding.

The cryptographic checks and atomic single-use claim implementation live in the
reviewed r4 module.  This module only pins every mutable path to the r10 launch.
"""
from pathlib import Path

import t4_m30_experiment_a_r4_authorization as base

ROOT = base.ROOT
base.RECEIPT = ROOT / (
    "sua_exploration/results/"
    "t4_m30_experiment_a_descriptor_prelaunch_v3_r10_20260803/receipt.json"
)
base.AUTH = ROOT / (
    "sua_exploration/results/"
    "t4_m30_experiment_a_descriptor_prelaunch_v3_r10_20260803/"
    "root_gpu_authorization.json"
)
base.SIG = ROOT / (
    "sua_exploration/results/"
    "t4_m30_experiment_a_descriptor_prelaunch_v3_r10_20260803/"
    "root_gpu_authorization.sig"
)
base.CLAIM = ROOT / (
    "sua_exploration/results/sua_t4_m30_component_attribution_v10/"
    "authorization_nonce_claim.json"
)

claim = base.claim
require_claim = base.require_claim
RECEIPT = base.RECEIPT

