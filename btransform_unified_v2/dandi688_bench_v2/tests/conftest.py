"""Run the standalone benchmark and its shared M2 code in one Python process."""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
for path in (WORKSPACE, ROOT, ROOT / "src", ROOT / "learnable_recency_v1" / "src",
             WORKSPACE / "btransform_unified_v1" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
torch.set_num_threads(2)
