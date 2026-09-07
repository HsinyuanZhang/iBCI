#!/usr/bin/env python3
"""Dry by default; reviewed dual-flag paired Stage-P v2 launcher."""
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

import track_b_v2_subject_m_stagep_paired_live_launcher_v2 as launcher  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--i-have-authorization", action="store_true")
    parser.add_argument("--internal-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--view", choices=launcher.PAIR_ORDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.execute != args.i_have_authorization:
        parser.error("v2 real launch requires both --execute and --i-have-authorization")
    if args.internal_child and not args.execute:
        parser.error("v2 internal child requires dual authorization")
    try:
        if not args.execute:
            sys.stdout.write(json.dumps(launcher.build_dry_plan(), sort_keys=True, indent=2) + "\n")
            return 0
        if os.environ.get(launcher.ROOT_ENV) != "1":
            raise launcher.StagePPairedLiveLauncherV2Error(
                f"v2 real launch requires {launcher.ROOT_ENV}=1")
        if args.internal_child:
            if args.view is None:
                raise launcher.StagePPairedLiveLauncherV2Error("v2 internal child view missing")
            return launcher.execute_internal_child(
                view=args.view, token=os.environ.get(launcher.INTERNAL_TOKEN_ENV, ""))
        result = launcher.execute_authorized_parent()
        sys.stdout.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
        return 0
    except launcher.StagePPairedLiveLauncherV2Error as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
