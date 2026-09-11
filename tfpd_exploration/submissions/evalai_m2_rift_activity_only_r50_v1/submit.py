#!/usr/bin/env python3
"""Guarded EvalAI push/register for M2 RIFT ACTIVITY_ONLY R50.

Delegates to the fixed-control helper. Default is read-only preflight.
Does nothing mutable unless --execute is passed with the immutable IDs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HELPER = REPO / "btransform_unified_v2/scripts/final_ablation_official_v1/submit_fixed_control.py"
DEFAULT_MANIFEST = HERE / "artifacts/fixed_control_manifest.json"
PROFILE = "m2_activity_only"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-image-id", default="")
    parser.add_argument("--confirm-payload-sha256", default="")
    args = parser.parse_args()
    cmd = [sys.executable, str(HELPER), "--profile", PROFILE, "--manifest", args.manifest]
    if args.execute:
        cmd.append("--execute")
    if args.confirm_image_id:
        cmd.extend(["--confirm-image-id", args.confirm_image_id])
    if args.confirm_payload_sha256:
        cmd.extend(["--confirm-payload-sha256", args.confirm_payload_sha256])
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
