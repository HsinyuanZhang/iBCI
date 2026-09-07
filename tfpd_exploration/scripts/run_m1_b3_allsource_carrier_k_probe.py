#!/usr/bin/env python3
"""CPU carrier-k probe for the frozen all-source b3s_rsyn3 student.

Tries chronological / tgt_loc D-opt / EMG-synergy D-opt at k=3,4,5,6,
plus chronological k=10. Encoder identity still uses all 10 calib neural
trials. Ranking R² is in-train source remaining-query after trial 10.
Not EvalAI hidden held-out.

    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" \\
      python tfpd_exploration/scripts/run_m1_b3_allsource_carrier_k_probe.py --dry-run

    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" \\
      python tfpd_exploration/scripts/run_m1_b3_allsource_carrier_k_probe.py --execute
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
        payload["cli"] = "run_m1_b3_allsource_carrier_k_probe.py"
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SystemExit("execute requires PYTHONNOUSERSITE=1")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from tfpd_exploration.src.m1_b3_allsource_v1.carrier_k_probe import execute

    shas, terminal, failure = execute(root)
    print(json.dumps(
        {"terminal_sha256": terminal, "failure_sha256": failure, "bodies": shas},
        sort_keys=True, separators=(",", ":"),
    ))
    return 0 if terminal and failure is None else 2


if __name__ == "__main__":
    sys.exit(main())
