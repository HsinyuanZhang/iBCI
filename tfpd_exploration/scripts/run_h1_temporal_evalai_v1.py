#!/usr/bin/env python3
"""H1 temporal Transformer EvalAI pipeline. H1-worker owned. Snapshot E_H=5 EMA."""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tfpd_exploration.src.two_mainlines_long_v1.h1_runtime.pipeline import run_slot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("flat", "route"))
    parser.add_argument("--stage", default="all", choices=("export", "parity", "image", "register", "all"))
    parser.add_argument("--skip-register", action="store_true")
    args = parser.parse_args()
    report = run_slot(args.arm, stage=args.stage, skip_register=args.skip_register)
    print(report.get("register") or report.get("container") or report.get("parity") or report.get("payload"))


if __name__ == "__main__":
    main()
