#!/usr/bin/env python3
"""Read-only B1 scorer construction smoke for an existing development checkpoint.

The check opens only the supplied local checkpoint and its frozen teacher so
that the scorer's explicit ``setup('test')`` plus strict state restoration is
tested against a real Lightning payload.  It does not instantiate a data
module, open NWB, make a forward pass, train, score, or write a receipt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.score_b1_m2_factorial_epochs import _instantiate_and_strict_load_checkpoint
from src.metrics.b1_m2_factorial import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--working-dir", type=Path, default=PROJECT)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint is absent: {checkpoint}")
    model = _instantiate_and_strict_load_checkpoint(checkpoint, working_dir=args.working_dir.resolve())
    print(json.dumps({
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "working_dir": str(args.working_dir.resolve()),
        "model_class": type(model).__name__,
        "setup_test_called": True,
        "strict_state_dict_load": True,
        "unexpected_keys": [],
        "missing_keys": [],
        "nwb_opened": False,
        "model_forward_called": False,
        "training_started": False,
        "receipt_written": False,
    }, indent=2))


if __name__ == "__main__":
    main()
