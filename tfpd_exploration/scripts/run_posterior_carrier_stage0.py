#!/usr/bin/env python3
"""Static dry/preflight entry point for posterior-carrier Stage 0.

This script intentionally imports only the standard-library-only plan module.
It has no execution flag, no output creation capability, and no path to Torch,
data/cache/checkpoints, CUDA/GPU, remote hosts, source smoke, or scoring.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_dry_plan() -> dict:
    # Keep the public route static: no import of posterior_carrier_v1.core.
    package_parent = str(ROOT / "tfpd_exploration")
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
    from src.posterior_carrier_v1.plan import dry_plan

    return dry_plan()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Posterior-carrier Stage-0 dry/preflight plan")
    parser.add_argument("--preflight", action="store_true", help="render the no-data preflight contract")
    parser.add_argument("--execute", action="store_true", help="always rejected; no Phase-B route exists")
    args = parser.parse_args(argv)
    if args.execute:
        parser.error("execution is not implemented; root-reviewed Phase-B authority is required")
    payload = _load_dry_plan()
    payload["requested_mode"] = "preflight" if args.preflight else "dry"
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

