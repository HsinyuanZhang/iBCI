#!/usr/bin/env python3
"""Default-blocked runner for one future consumed-sub-C scorer-adapter parity run.

The only usable mode in this delivery is ``dry-run``.  It reads the immutable
parity-v2 prelaunch bundle and prints its no-execution disposition.  ``execute``
is an intentional hard fence before importing the Torch/C1 runtime helper, so
it cannot load a checkpoint/NWB or run a forward in the current delivery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))

from scripts.write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v2 import (  # noqa: E402
    DEFAULT_OUTPUT,
    StaticParityV2Error,
    verify_stored_prelaunch,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "execute"), default="dry-run")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--prelaunch-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--authorization", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        stored = verify_stored_prelaunch(args.prelaunch_dir, args.repo_root)
        if args.mode == "dry-run":
            print(json.dumps({"status": "PARITY_EXECUTION_NOT_AUTHORIZED", "prelaunch": stored, "checkpoint_nwb_forward_allowed": False}, indent=2, sort_keys=True))
            return 0
        # Do not import the helper on this path: the prohibition must happen
        # before a potential C1/Torch runtime import, checkpoint, or NWB touch.
        del args.output_root, args.authorization
        raise StaticParityV2Error(
            "PARITY_EXECUTION_NOT_AUTHORIZED: a future append-only authorization must bind this parity-v2 prelaunch and an independently reviewed adapter before any runtime import"
        )
    except StaticParityV2Error as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
