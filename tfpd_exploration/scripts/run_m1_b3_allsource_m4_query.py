#!/usr/bin/env python3
"""CPU M4-calib / remaining-query diagnostic for all-source B3 students.

Default surface: three later-day 10-trial public calib files, first 4 trials
for identity, remaining 6 as query. Wave-1 and wave-2 rSyn3 students.
Not EvalAI hidden held-out.

    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" \\
      python tfpd_exploration/scripts/run_m1_b3_allsource_m4_query.py --dry-run

    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" \\
      python tfpd_exploration/scripts/run_m1_b3_allsource_m4_query.py \\
        --execute --arm b3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument(
        "--arm",
        choices=("b3", "b3s_rsyn3", "b3s_rsyn3_freeze", "b3s_rsyn3_acyc"),
        default=None,
    )
    parser.add_argument(
        "--include-source-in-train",
        action="store_true",
        help="also score remaining query on the four in-train source files (slow, in-train)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if arguments.execute_gpu:
        parser.error("cannot mint a GPU capability")
    if not arguments.execute:
        from tfpd_exploration.src.m1_b3_allsource_v1.plan import dry_plan

        payload = dry_plan()
        payload["cli"] = "run_m1_b3_allsource_m4_query.py"
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0
    if arguments.arm is None:
        parser.error("--execute requires --arm")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SystemExit("execute requires PYTHONNOUSERSITE=1")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from tfpd_exploration.src.m1_b3_allsource_v1.m4_query import execute

    shas, terminal, failure = execute(
        root,
        arguments.arm,
        include_source_in_train=bool(arguments.include_source_in_train),
    )
    print(json.dumps(
        {"terminal_sha256": terminal, "failure_sha256": failure, "bodies": shas},
        sort_keys=True, separators=(",", ":"),
    ))
    return 0 if terminal and failure is None else 2


if __name__ == "__main__":
    sys.exit(main())
