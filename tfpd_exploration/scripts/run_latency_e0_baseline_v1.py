#!/usr/bin/env python3
"""Start latency E0 microprobe. CPU only. New result root."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfpd_exploration.src.two_mainlines_long_v1.latency_opt_v1.e0_baseline import main


if __name__ == "__main__":
    main()
