#!/usr/bin/env python3
"""PACD V1 bounded source-smoke entry point.

Default operation is deliberately inert: ``--dry-run`` emits the static
contract and exits before importing Torch, resolving a result root, querying a
GPU, opening data, or loading a checkpoint.  Real execution requires both an
explicit output directory and the exact acknowledgement flag.  That prevents
a documentation/read-only invocation from accidentally competing with an
active experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.paired_anchored_calibration_dropout_v1 import plan  # noqa: E402
from src.paired_anchored_calibration_dropout_v1 import smoke  # noqa: E402


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PACD V1 paired source smoke")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="default: static, no-Torch plan")
    mode.add_argument("--execute", action="store_true", help="run only after reviewer authorisation")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="fresh explicit output directory; required with --execute")
    parser.add_argument("--steps", type=int, default=plan.SMOKE_STEPS)
    parser.add_argument("--seed", type=int, default=plan.SEED)
    parser.add_argument("--train-batch-size", type=int, default=plan.TRAIN_BATCH_SIZE)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="fixed at zero for the isolated source smoke",
    )
    parser.add_argument(
        "--execution-authorized-by-root",
        action="store_true",
        help="explicit acknowledgement; no scientific or resource authorisation is inferred",
    )
    parser.add_argument(
        "--initial-state",
        type=Path,
        default=ROOT / "results/admission_arms_v1/canonical_initial_state.pt",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.steps <= 0:
        print("--steps must be positive", file=sys.stderr)
        return 2
    if args.train_batch_size <= 0:
        print("--train-batch-size must be positive", file=sys.stderr)
        return 2
    if args.num_workers != 0:
        print("PACD V1 fixes --num-workers=0", file=sys.stderr)
        return 2
    if not args.execute:
        print(json.dumps(smoke.dry_payload(), indent=2, sort_keys=True))
        return 0
    if args.out_dir is None:
        print("--execute requires an explicit fresh --out-dir", file=sys.stderr)
        return 2
    if not args.execution_authorized_by_root:
        print(
            "--execute requires --execution-authorized-by-root; this CLI never infers launch authority",
            file=sys.stderr,
        )
        return 2
    try:
        return smoke.execute(root=REPO, out_dir=args.out_dir, args=args)
    except BaseException as error:  # execute publishes an immutable failure when possible
        print(f"PACD smoke stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
