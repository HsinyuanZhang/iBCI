#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from sua_exploration.behavior_autoencoder_v1.combine import combine_family_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise SystemExit(f"refusing overwrite: {args.output}")
    body = combine_family_results(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "equal_session": body["equal_session"], "paired_delta_vs_directridge": body["paired_delta_vs_directridge"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
