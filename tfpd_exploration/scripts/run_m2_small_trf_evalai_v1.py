#!/usr/bin/env python3
"""S1 small-Transformer EvalAI pipeline. D-worker owned."""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.pipeline import run_slot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=("replay", "export", "parity", "image", "register", "all"))
    parser.add_argument("--skip-register", action="store_true")
    args = parser.parse_args()
    run_slot("small", stage=args.stage, skip_register=args.skip_register)


if __name__ == "__main__":
    main()
