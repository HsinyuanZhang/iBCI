#!/usr/bin/env python3
"""Dry-plan by default; reviewed dual-flag isolated Stage-P launcher."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


os.environ["PYTHONNOUSERSITE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_subject_m_stagep_paired_live_launcher as launcher  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--i-have-authorization", action="store_true")
    parser.add_argument("--internal-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--view", choices=launcher.PAIR_ORDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.execute != args.i_have_authorization:
        parser.error("real launch requires both --execute and --i-have-authorization")
    if args.internal_child and not args.execute:
        parser.error("internal child requires the same dual authorization flags")
    try:
        if not args.execute:
            sys.stdout.write(json.dumps(launcher.build_dry_plan(), sort_keys=True, indent=2) + "\n")
            return 0
        if os.environ.get(launcher.ROOT_ENV) != "1":
            raise launcher.StagePPairedLiveLauncherError(
                f"real launch requires {launcher.ROOT_ENV}=1")
        if args.internal_child:
            token = os.environ.get(launcher.INTERNAL_TOKEN_ENV, "")
            if args.view is None:
                raise launcher.StagePPairedLiveLauncherError("internal child view missing")
            return launcher.execute_internal_child(view=args.view, token=token)
        result = launcher.execute_authorized_parent()
        sys.stdout.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
        return 0
    except launcher.StagePPairedLiveLauncherError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

