#!/usr/bin/env python3
"""Static dry entry point for the additive PIRG score-v2 repair."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap() -> None:
    root = Path(__file__).resolve().parents[2]
    package_root = str(root / "tfpd_exploration")
    if package_root not in sys.path:
        sys.path.insert(0, package_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PIRG score-v2 scalar-digest repair dry plan only")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-reviewed", action="store_true")
    args = parser.parse_args(argv)
    _bootstrap()
    from src.posterior_identity_residual_gate_score_v2.score_v2 import dry_plan

    if args.execute or args.root_reviewed:
        raise SystemExit("PIRG score V2 remains dry: an in-process root-reviewed capability is required")
    print(json.dumps(dry_plan(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
