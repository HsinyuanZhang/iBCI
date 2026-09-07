#!/usr/bin/env python3
"""Run the frozen CPU-only B1 ARTP held-in source screen."""
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "INERT_DRY_RUN",
                    "route": "B1_ARTP_V1",
                    "requires": "--execute",
                    "device": "cpu",
                    "evalai_push": False,
                },
                sort_keys=True,
            )
        )
        return
    from tfpd_exploration.src.b1_artp_v1 import plan
    from tfpd_exploration.src.b1_artp_v1.screen import run_source_screen
    from tfpd_exploration.src.b1_sfcj_v1.util import write_json

    target = plan.RESULT_ROOT / "source_screen.json"
    if target.exists():
        raise FileExistsError(target)
    receipt = run_source_screen()
    write_json(target, receipt)
    print(json.dumps({"status": receipt["status"], "path": str(target), "all_gates_pass": receipt["all_gates_pass"]}, sort_keys=True))


if __name__ == "__main__":
    main()
