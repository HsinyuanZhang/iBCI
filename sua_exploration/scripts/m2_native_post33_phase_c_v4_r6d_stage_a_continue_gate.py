#!/usr/bin/env python3
"""No-output fixed-policy Stage-A validator for the r6d live coordinator.

The process deliberately exports only an exit status.  It validates the
already-signed fixed-policy decision and never prints an endpoint value.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_openers_v4 import validate_stage_a_decision


EXIT_CONTINUE = 0
EXIT_VALIDATED_STOP = 10
EXIT_INVALID = 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--classify-signed", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        decision = validate_stage_a_decision(args.root, require_continue=False)
    except Exception:
        return EXIT_INVALID
    if decision.get("decision") == "continue_without_positive_claim":
        return EXIT_CONTINUE
    if decision.get("decision") == "seed42_severe_negative_futility_stop":
        return EXIT_VALIDATED_STOP
    return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
