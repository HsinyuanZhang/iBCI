#!/usr/bin/env python3
"""Fail-closed aggregator for the B1 carrier-arm loss-mode sweep."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.b1_carrier_loss_mode_matrix import (
  IncompleteB1MatrixError,
  aggregate_b1_carrier_loss_mode_file,
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--matrix",
    type=Path,
    required=True,
    help="Path to gate2_revised_matrix.csv containing B1 carrier cells",
  )
  parser.add_argument("--fold", type=int, default=0)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  try:
    decision = aggregate_b1_carrier_loss_mode_file(
      str(args.matrix.resolve()), fold_id=args.fold, seed=args.seed
    )
  except IncompleteB1MatrixError as exc:
    print(json.dumps({"ready": False, "error": str(exc)}, indent=2))
    raise SystemExit(1) from exc

  print(json.dumps(asdict(decision), indent=2))


if __name__ == "__main__":
  main()
