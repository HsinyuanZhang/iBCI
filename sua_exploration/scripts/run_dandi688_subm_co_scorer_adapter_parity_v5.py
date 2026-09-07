#!/usr/bin/env python3
"""Fixed-prelaunch runner for future one-time v5 schema-bridge parity.

Dry-run reads and verifies only the immutable v5 prelaunch. Execute accepts no
caller policy, public key, checkpoint, NWB, or model argument: it validates the
detached authorization against policy reconstructed from the fixed 0444 bundle,
claims its fresh nonce, and only then can import the corrected v5 helper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.subm_co_scorer_adapter_parity_execution_v5 import (  # noqa: E402
    ParityV5AuthorizationError,
    ParityV5ExecutionError,
    execute_from_fixed_prelaunch,
)
from sua_exploration.scripts.write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v5 import (  # noqa: E402
    DEFAULT_OUTPUT,
    StaticParityV5Error,
    load_stored_prelaunch,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "execute"), default="dry-run")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--prelaunch-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--signature", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        stored = load_stored_prelaunch(args.prelaunch_dir, args.repo_root)
        if args.mode == "dry-run":
            print(
                json.dumps(
                    {
                        "status": "SCHEMA_BRIDGE_CPU_PARITY_EXECUTION_NOT_AUTHORIZED",
                        "prelaunch": {
                            key: stored[key]
                            for key in (
                                "draft_sha256",
                                "receipt_sha256",
                                "seal_sha256",
                                "execution_policy_sha256",
                                "status",
                            )
                        },
                        "checkpoint_nwb_npz_torch_import_allowed": False,
                        "external_subm_scoring_allowed": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.authorization is None or args.signature is None or args.output_root is None:
            raise ParityV5AuthorizationError(
                "execute requires detached --authorization/--signature and one --output-root"
            )
        result = execute_from_fixed_prelaunch(
            authorization_path=args.authorization,
            signature_path=args.signature,
            output_root=args.output_root,
            prelaunch_dir=args.prelaunch_dir,
            root=args.repo_root,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (StaticParityV5Error, ParityV5AuthorizationError, ParityV5ExecutionError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
