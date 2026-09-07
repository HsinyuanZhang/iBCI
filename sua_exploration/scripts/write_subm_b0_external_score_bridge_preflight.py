#!/usr/bin/env python3
"""Root-only publisher for an immutable, target-free B0 bridge preflight."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import subm_b0_external_score_bridge as bridge  # noqa: E402

MINT_ENV = "B0_BRIDGE_ROOT_PREFLIGHT_MINT"
MINT_VALUE = "I_AUTHORIZE_B0_EXTERNAL_SCORE_BRIDGE_PREFLIGHT_MINT"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=bridge.RESULT_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-checkpoint-payload", action="store_true")
    args = parser.parse_args(argv)
    if os.environ.get(MINT_ENV) != MINT_VALUE:
        print(f"FAIL_CLOSED: set {MINT_ENV}={MINT_VALUE}", file=sys.stderr)
        return 2
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        print("FAIL_CLOSED: CUDA_VISIBLE_DEVICES must be empty", file=sys.stderr)
        return 2
    try:
        payload = bridge.build_preflight(output_root=args.output_root,
                                         verify_checkpoint_payload=args.verify_checkpoint_payload)
        payload = {**payload, "immutable_preflight_path": str(args.out.resolve())}
        _body, sidecar, digest = bridge._write_pair_once(args.out, payload)
    except (bridge.B0BridgeError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(args.out.resolve()), "sidecar": str(sidecar), "sha256": digest,
                      "target_data_opened": False, "score_emitted": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
