#!/usr/bin/env python
"""Run P1–P5 gates. CPU by default; pass --cuda to use GPU1."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    os.environ["PYTHONNOUSERSITE"] = "1"
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        device = "cuda:0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
    from tfpd_exploration.src.two_mainlines_long_v1.calibration.gates import run_all_gates

    summary = run_all_gates(device=device)
    print(summary["all_pass"], {name: payload["passed"] for name, payload in summary["gates"].items()})
    return 0 if summary["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
