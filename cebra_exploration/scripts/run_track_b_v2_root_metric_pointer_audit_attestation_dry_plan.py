#!/usr/bin/env python3
"""Render the no-write root-audit attestation template for one metric pointer scope."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_contract as base  # noqa: E402
import track_b_v2_metric_pointer_authority as pointer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("subject_m", "rt"))
    parser.add_argument("--view", choices=("sua", "pseudo_mua"))
    args = parser.parse_args()
    if args.dataset == "subject_m" and args.view is None:
        parser.error("subject_m requires --view")
    if args.dataset == "rt" and args.view is not None:
        parser.error("rt has no --view")
    dry = pointer.build_metric_pointer_dry_plan(args.dataset, args.view)
    sys.stdout.buffer.write(base.canonical_json_bytes(
        pointer.build_root_metric_pointer_audit_attestation_dry_plan(dry_plan=dry)
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
