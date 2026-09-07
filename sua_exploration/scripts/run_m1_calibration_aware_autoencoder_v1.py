#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "SPINT-main"):
    if str(value) not in sys.path: sys.path.insert(0, str(value))

from sua_exploration.behavior_autoencoder_v1.calibration_aware import CalibrationAwareSpec, nested_calibration_aware_loso
from sua_exploration.h1_m1_priority_v1.execute import load_m1_sessions


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--q", type=int, choices=(4, 8, 12), required=True); parser.add_argument("--hidden", type=int, choices=(64, 128), default=64); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--seed-offset", type=int, default=0); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise SystemExit(f"refusing overwrite: {args.output}")
    specs = tuple(CalibrationAwareSpec(args.q, args.hidden, weight) for weight in (0.03, 0.1, 0.3, 1.0))
    result = nested_calibration_aware_loso(load_m1_sessions(), specs, device=args.device, seed_offset=args.seed_offset)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "equal_session": result["equal_session"], "paired_delta_vs_directridge": result["paired_delta_vs_directridge"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
