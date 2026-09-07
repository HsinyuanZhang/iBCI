#!/usr/bin/env python3
"""Root-only publisher for the one immutable B0 bridge score authorization."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import subm_b0_external_score_bridge as bridge  # noqa: E402

MINT_ENV = "B0_BRIDGE_ROOT_SCORE_AUTHORIZATION"
MINT_VALUE = "I_AUTHORIZE_B0_EXTERNAL_SCORE_BRIDGE_CPU_SCORE"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.environ.get(MINT_ENV) != MINT_VALUE:
        print(f"FAIL_CLOSED: set {MINT_ENV}={MINT_VALUE}", file=sys.stderr)
        return 2
    try:
        preflight, digest = bridge.load_immutable_preflight(args.preflight)
        preflight = {**preflight, "immutable_preflight_path": str(args.preflight.resolve())}
        payload = {
            "schema": bridge.ROOT_AUTHORIZATION_SCHEMA, "root_authorized": True,
            "preflight_path": str(args.preflight.resolve()), "preflight_sha256": digest,
            "b0_source_audit_sha256": preflight["b0_source_audit_sha256"],
            "output_root": preflight["output_root"], "allowed_seeds": list(bridge.SEEDS),
            "cpu_only": True, "gpu_permitted": False, "target_updates_permitted": False,
            "target_epoch_selection_permitted": False,
            "scientific_scope": "fixed B0 external forward scores; T4/B0 and Z4/B0 system contrasts only, not carrier causal contrasts",
        }
        _body, sidecar, out_digest = bridge._write_pair_once(args.out, payload)
    except (bridge.B0BridgeError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(args.out.resolve()), "sidecar": str(sidecar), "sha256": out_digest,
                      "target_data_opened": False, "score_emitted": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
