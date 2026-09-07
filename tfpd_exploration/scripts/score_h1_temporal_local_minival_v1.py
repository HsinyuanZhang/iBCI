#!/usr/bin/env python3
"""Score H1 temporal EMA on held-in-minival. No hidden held-out eval."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.score_local import main


if __name__ == "__main__":
    main()
