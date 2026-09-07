#!/usr/bin/env python3
"""Public static entry point for the future PIRG 3+3 score screen.

It intentionally imports only the standard-library plan on the zero-argument
path.  CLI flags are not a score capability and fail before any Torch, V3,
artifact, evaluation-input, or device module can be imported.
"""
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
    parser = argparse.ArgumentParser(description="PIRG quick-score dry plan only")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-reviewed", action="store_true")
    args = parser.parse_args(argv)
    _bootstrap()
    from src.posterior_identity_residual_gate_v1.plan import dry_plan

    if args.execute or args.root_reviewed:
        raise SystemExit("PIRG score remains dry: an in-process root-reviewed capability is required")
    payload = dry_plan()
    payload["route"] = "PIRG_QUICK_SCORE_3WITHIN_3EXTERNAL_M30_M10_M4"
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
