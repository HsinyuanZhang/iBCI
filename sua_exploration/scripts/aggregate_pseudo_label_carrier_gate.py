#!/usr/bin/env python3
"""Fail-closed aggregator for B8 pseudo-label carrier gate receipts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))

from mc_maze.pseudo_label_carrier_gate_replay import validate_receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    summary = validate_receipt(receipt)
    text = json.dumps(summary, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
