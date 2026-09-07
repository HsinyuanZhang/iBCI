#!/usr/bin/env python3
"""Aggregate all three immutable B0 bridge score pairs without target access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import subm_b0_external_score_bridge as bridge  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=bridge.RESULT_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = bridge.aggregate_payload(b0_score_pairs={seed: bridge.b0_score_path(seed, result_root=args.result_root)
                                                           for seed in bridge.SEEDS})
        _body, sidecar, digest = bridge._write_pair_once(args.out, payload)
    except (bridge.B0BridgeError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(args.out.resolve()), "sidecar": str(sidecar), "sha256": digest,
                      "target_data_opened_by_aggregator": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
