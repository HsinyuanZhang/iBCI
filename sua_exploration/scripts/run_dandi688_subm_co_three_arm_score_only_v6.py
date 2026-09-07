#!/usr/bin/env python3
"""Blocked-only runner for external sub-M three-arm V6."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sua_exploration.mc_maze.subm_co_three_arm_score_only_v6 import V6BlockedError, refuse_blocked_execution  # noqa: E402
from sua_exploration.scripts.write_dandi688_subm_co_three_arm_score_only_prelaunch_v6 import DEFAULT_OUTPUT, StaticV6Error, load_stored_blocked_prelaunch  # noqa: E402


def dry_run_payload(prelaunch_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    stored = load_stored_blocked_prelaunch(prelaunch_dir)
    return {"status": stored["status"], "matrix": {"N": 15, "views": 2, "seeds": 3, "arms": 3, "cells": 270}, "missing_checkpoint_slots": stored["missing_checkpoint_slots"], "contract_sha256": stored["contract_sha256"], "verified_grant_created": False, "checkpoint_nwb_torch_gpu_external_r2_allowed": False, "external_capability_created": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--mode", choices=("dry-run", "score"), default="dry-run"); parser.add_argument("--prelaunch-dir", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args(argv)
    try:
        load_stored_blocked_prelaunch(args.prelaunch_dir)
        if args.mode == "dry-run":
            print(json.dumps(dry_run_payload(args.prelaunch_dir), indent=2, sort_keys=True)); return 0
        refuse_blocked_execution(); raise AssertionError("unreachable")
    except (StaticV6Error, V6BlockedError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
