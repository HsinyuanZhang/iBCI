#!/usr/bin/env python3
"""Read-only/source-only preflight for the additive original-SPINT B0 bridge.

This command deliberately has no receipt-writing option: root must review and
mint any immutable authorization separately.  It never resolves or opens a
sub-M NWB, and it never imports a model unless ``--verify-checkpoint-payload``
is requested for the archived source checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import subm_b0_external_score_bridge as bridge  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=bridge.RESULT_ROOT)
    parser.add_argument("--verify-checkpoint-payload", action="store_true")
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        print("FAIL_CLOSED: CUDA_VISIBLE_DEVICES must be empty", file=sys.stderr)
        return 2
    try:
        payload = bridge.build_preflight(output_root=args.output_root,
                                         verify_checkpoint_payload=args.verify_checkpoint_payload)
    except (bridge.B0BridgeError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
