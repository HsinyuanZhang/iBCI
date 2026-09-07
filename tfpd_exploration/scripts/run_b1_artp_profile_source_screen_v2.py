#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "INERT_DRY_RUN", "route": "B1_ARTP_PROFILE_V2", "device": "cpu"}, sort_keys=True))
        return
    from tfpd_exploration.src.b1_artp_v2 import plan
    from tfpd_exploration.src.b1_artp_v2.screen import run_source_screen
    from tfpd_exploration.src.b1_sfcj_v1.util import write_json

    target = plan.RESULT_ROOT / "source_screen.json"
    if target.exists():
        raise FileExistsError(target)
    receipt = run_source_screen()
    write_json(target, receipt)
    print(json.dumps({"status": receipt["status"], "path": str(target), "all_primary_gates_pass": receipt["all_primary_gates_pass"], "continual": receipt["continual_profile_decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
