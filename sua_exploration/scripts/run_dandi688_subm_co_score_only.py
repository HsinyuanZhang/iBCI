#!/usr/bin/env python3
"""Strict, default-blocked score-only entrypoint for DANDI 000688 sub-M CO.

The default ``dry-run`` mode validates immutable metadata/source pins and
prints a plan.  It cannot load a checkpoint, import the scorer, open target
NWB content, compute a prediction/R2, or use a GPU.  The two executable modes
remain unavailable until a separate root-issued authorization record exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))

from mc_maze.subm_co_score_only import (  # noqa: E402
    AuthorizationError,
    ScoreOnlyContractError,
    aggregate_authorized,
    build_dry_run_plan,
    score_authorized,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("dry-run", "score", "aggregate"),
        default="dry-run",
        help="dry-run is the only non-authorized mode; score seals only; aggregate is separate.",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--nwb-asset-root",
        type=Path,
        help="Authorized score mode only: exact asset-ID NWB directory from the v2 preflight cache.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.mode == "dry-run":
            print(json.dumps(build_dry_run_plan(args.repo_root), indent=2, sort_keys=True))
            return 0
        if args.authorization is None or args.output_root is None:
            raise AuthorizationError("--authorization and --output-root are mandatory outside dry-run")
        if args.mode == "score":
            if args.nwb_asset_root is None:
                raise AuthorizationError("--nwb-asset-root is mandatory for authorized score mode")
            result = score_authorized(
                authorization_path=args.authorization,
                output_root=args.output_root,
                nwb_asset_root=args.nwb_asset_root,
                root=args.repo_root,
            )
        else:
            result = aggregate_authorized(
                authorization_path=args.authorization,
                output_root=args.output_root,
                root=args.repo_root,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (AuthorizationError, ScoreOnlyContractError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
