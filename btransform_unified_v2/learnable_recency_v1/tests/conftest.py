"""CPU-only test discipline for learnable recency (mirrors btransform_unified_v2/tests)."""

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
V1 = WORKSPACE / "btransform_unified_v1"
for path in (TESTS, PACKAGE / "src", ROOT / "src", V1 / "src", V1 / "scripts", WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
