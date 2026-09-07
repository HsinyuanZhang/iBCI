#!/usr/bin/env python3
"""Bind one completed B1 cell's artifact and Hydra checkpoint paths.

This is metadata-only.  It never launches training, imports Torch/Lightning,
opens NWB, scores a checkpoint, or writes inside a run directory.  A future
authorized executor must run it only after a completed cell has preserved the
fixed epoch-004--011 files.  The immutable binder prevents a separately
located Hydra log and artifact directory from being combined post hoc.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.metrics import b1_m2_factorial as core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future-launch-receipt", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--checkpoint-run-dir", required=True, type=Path)
    parser.add_argument("--execution-completion-receipt", type=Path, required=True,
                        help="successful completion receipt required for every B1 launch contract")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() or Path(f"{args.out}.sha256").exists():
        raise SystemExit(f"Refusing existing B1 post-training binding: {args.out}")
    digest = core.write_post_training_binding(
        args.out,
        future_launch_receipt=args.future_launch_receipt,
        artifact=args.artifact,
        checkpoint_run_dir=args.checkpoint_run_dir,
        interpreter=Path(sys.executable),
        script=PROJECT / "src/train.py",
        working_dir=PROJECT,
        execution_completion_receipt=args.execution_completion_receipt,
    )
    print(json.dumps({
        "receipt": str(args.out.resolve()),
        "sha256": digest,
        "metadata_only": True,
        "torch_imported": False,
        "nwb_opened": False,
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
