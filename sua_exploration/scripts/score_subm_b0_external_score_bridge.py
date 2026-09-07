#!/usr/bin/env python3
"""CPU score entrypoint for a root-authorized original-SPINT B0 bridge cell.

The default ``--dry-run`` is source-only and opens no target.  ``--execute``
is deliberately unusable without an immutable root authorization pair whose
body binds the exact fresh preflight/source bundle.  There is no train/GPU
option and no checkpoint-selection input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import subm_b0_external_score_bridge as bridge  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=bridge.SEEDS, required=True)
    parser.add_argument("--output-root", type=Path, default=bridge.RESULT_ROOT)
    parser.add_argument("--preflight", type=Path,
                        help="root-reviewed immutable bridge preflight pair; required with --execute")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        print("FAIL_CLOSED: CUDA_VISIBLE_DEVICES must be empty", file=sys.stderr)
        return 2
    try:
        if not args.execute:
            preflight = bridge.build_preflight(output_root=args.output_root, verify_checkpoint_payload=False)
            print(json.dumps({"status": "B0_BRIDGE_DRY_RUN_NOT_AUTHORIZED_TO_OPEN_TARGET",
                              "preflight_sha256": hashlib.sha256(bridge.canonical_json_bytes(preflight)).hexdigest(),
                              "target_data_opened": False, "gpu_used": False, "score_emitted": False},
                             indent=2, sort_keys=True))
            return 0
        if args.authorization is None or args.preflight is None:
            raise bridge.B0BridgeError("--execute requires immutable --preflight and --authorization")
        preflight, _preflight_sha = bridge.load_immutable_preflight(args.preflight)
        preflight = {**preflight, "immutable_preflight_path": str(args.preflight.resolve())}
        payload = bridge.execute_cpu_score(args.seed, output_path=bridge.b0_score_path(args.seed, result_root=args.output_root),
                                           preflight=preflight, authorization_path=args.authorization)
    except (bridge.B0BridgeError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": payload["status"], "mean_r2": payload["mean_r2"], "seed": args.seed},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
