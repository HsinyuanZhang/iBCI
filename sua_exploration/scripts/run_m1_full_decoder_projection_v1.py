#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
ordered = (ROOT / "streaming_calibration_exp", ROOT, ROOT / "SPINT-main")
for value in ordered:
    while str(value) in sys.path: sys.path.remove(str(value))
for value in reversed(ordered): sys.path.insert(0, str(value))

from sua_exploration.behavior_autoencoder_v1.full_projection import evaluate_fold0_projection
from sua_exploration.h1_m1_priority_v1.execute import load_m1_sessions


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, action="append", required=True); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise SystemExit(f"refusing overwrite: {args.output}")
    result = evaluate_fold0_projection(load_m1_sessions(), args.input, device=args.device); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "baseline_r2": result["full_decoder_authority"]["baseline_metrics"]["pooled_variance_weighted_r2"], "projected_r2": result["projected_metrics"]["pooled_variance_weighted_r2"], "delta": result["delta_vs_unprojected_full_decoder"], "positive_outputs": result["positive_outputs"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
