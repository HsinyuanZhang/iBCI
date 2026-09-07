#!/usr/bin/env python3
"""Public static entry point for PIRG; it deliberately cannot launch work."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap() -> None:
    root = Path(__file__).resolve().parents[2]
    package_root = root / "tfpd_exploration"
    rendered = str(package_root)
    if rendered not in sys.path:
        sys.path.insert(0, rendered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PIRG dry plan only")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-reviewed", action="store_true")
    args = parser.parse_args(argv)
    _bootstrap()
    from src.posterior_identity_residual_gate_v1.plan import dry_plan

    if args.execute or args.root_reviewed:
        # Keep this branch before importing any lifecycle/physical module.  A
        # CLI flag is never an execution capability and must not even begin a
        # source/device/output route.
        raise SystemExit("PIRG remains dry: an in-process root-reviewed capability is required")
    print(json.dumps(dry_plan(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
