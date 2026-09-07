#!/usr/bin/env python3
"""Default-blocked v2 score-only entrypoint for DANDI 000688 sub-M CO.

``dry-run`` is the only usable mode in this delivery.  The two future action
modes require a separate, signature-verified root authorization, the stored
v2 prelaunch bundle, an explicit single execution device, and a root-pinned
scorer-adapter parity receipt from an already-consumed sub-C development
session.  No dry run opens a checkpoint/NWB, invokes a model, computes a real
score, or uses a GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))

from mc_maze.subm_co_score_only_v2 import (  # noqa: E402
    AuthorizationError,
    ScoreOnlyContractError,
    aggregate_authorized_v2,
    build_dry_run_plan_v2,
    score_authorized_v2,
)


DEFAULT_PRELAUNCH = ROOT / "sua_exploration/results/dandi_000688_subm_co_score_only_prelaunch_v2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "score", "aggregate"), default="dry-run")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--prelaunch-dir", type=Path, default=DEFAULT_PRELAUNCH)
    parser.add_argument(
        "--device",
        help="Exact authorized device for all 180 cells: cpu or cuda:<index>. Required outside dry-run.",
    )
    parser.add_argument(
        "--nwb-asset-root",
        type=Path,
        help="Future score mode only; not opened before authorization succeeds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.mode == "dry-run":
            print(json.dumps(build_dry_run_plan_v2(args.repo_root), indent=2, sort_keys=True))
            return 0
        if args.authorization is None or args.output_root is None or args.device is None:
            raise AuthorizationError("--authorization, --output-root, and --device are mandatory outside dry-run")
        if args.mode == "score":
            if args.nwb_asset_root is None:
                raise AuthorizationError("--nwb-asset-root is mandatory for future authorized score mode")
            result = score_authorized_v2(
                authorization_path=args.authorization,
                output_root=args.output_root,
                prelaunch_dir=args.prelaunch_dir,
                nwb_asset_root=args.nwb_asset_root,
                requested_device=args.device,
                root=args.repo_root,
            )
        else:
            result = aggregate_authorized_v2(
                authorization_path=args.authorization,
                output_root=args.output_root,
                prelaunch_dir=args.prelaunch_dir,
                requested_device=args.device,
                root=args.repo_root,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (AuthorizationError, ScoreOnlyContractError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
