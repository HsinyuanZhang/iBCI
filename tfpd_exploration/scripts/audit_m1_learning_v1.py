#!/usr/bin/env python3
"""M1 real-learning check. Does not assume H1 /20. No EvalAI submit."""

from __future__ import annotations

import json
import os

import torch

from tfpd_exploration.src.m1_temporal_v2.learn_check import run_learn_check


def main() -> None:
    os.environ["PYTHONNOUSERSITE"] = "1"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    report = run_learn_check(torch.device("cuda:0"))
    print(json.dumps({
        "status": report["status"],
        "divisor": report["prediction_divisor"],
        "assumes_h1_div20": report["assumes_h1_div20"],
        "r2": report["raw_vs_y"]["r2"],
        "zero_r2": report["raw_vs_y"]["zero_r2"],
        "mean_r2": report["raw_vs_y"]["mean_r2"],
        "std_ratio": report["raw_vs_y"]["pred_std_over_target_std"],
        "input_delta": report["input_abs_delta_l2"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
