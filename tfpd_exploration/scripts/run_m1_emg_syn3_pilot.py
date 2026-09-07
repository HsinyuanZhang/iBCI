#!/usr/bin/env python3
"""Public Stage-1 pilot CLI. Always inert without a separate GPU capability."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.execute:
        parser.error("Stage-1 GPU requires a separately issued capability")
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1.plan import dry_cli_payload
    payload = dry_cli_payload()
    payload["cli"] = "run_m1_emg_syn3_pilot.py"
    payload["stage1_arms"] = list(payload["stage1_arms"])
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
