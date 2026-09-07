#!/usr/bin/env python3
"""Fail-closed runner for future independently-loaded data-adapter parity v3.

`dry-run` verifies only the immutable static prelaunch bundle. `execute`
raises before importing the v3 helper, so it cannot open a checkpoint/NWB or
perform a data load, forward, or metric update in this delivery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))

from scripts.write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v3 import (  # noqa: E402
    DEFAULT_OUTPUT,
    StaticParityV3Error,
    verify_stored_prelaunch,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "execute"), default="dry-run")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--prelaunch-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        stored = verify_stored_prelaunch(args.prelaunch_dir, args.repo_root)
        if args.mode == "dry-run":
            print(
                json.dumps(
                    {
                        "status": "PARITY_EXECUTION_NOT_AUTHORIZED",
                        "prelaunch": stored,
                        "checkpoint_nwb_forward_allowed": False,
                        "v2_disposition": "INSUFFICIENT_DATA_ADAPTER_PARITY_NON_AUTHORIZING",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        # Do not import the helper on this path. The fence is intentionally
        # before every runtime owner import, checkpoint/NWB access, or forward.
        del args.authorization, args.output_root
        raise StaticParityV3Error(
            "PARITY_EXECUTION_NOT_AUTHORIZED: v3 requires a later append-only "
            "one-time authorization and root review before concrete "
            "data-adapter parity may execute"
        )
    except StaticParityV3Error as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

