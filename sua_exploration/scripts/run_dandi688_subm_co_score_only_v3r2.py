#!/usr/bin/env python3
"""Fixed-prelaunch V3R2 runner for the future external sub-M matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.subm_co_score_only_v3 import (  # noqa: E402
    ScoreV3AuthorizationError,
)
from sua_exploration.mc_maze.subm_co_score_only_v3r2 import (  # noqa: E402
    ScoreV3R2ExecutionError,
    execute_from_fixed_prelaunch_v3r2,
)
from sua_exploration.scripts.write_dandi688_subm_co_score_only_prelaunch_v3r2 import (  # noqa: E402
    DEFAULT_OUTPUT,
    StaticScoreV3R2Error,
    load_stored_prelaunch,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "score"), default="dry-run")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--prelaunch-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--signature", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--external-nwb-root", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        stored = load_stored_prelaunch(args.prelaunch_dir, args.repo_root)
        if args.mode == "dry-run":
            print(
                json.dumps(
                    {
                        "status": "V3R2_READY_BUT_NOT_AUTHORIZED_PENDING_NATIVE_M2",
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
                        "deployment_budget": {
                            "activity_identity_trials": 30,
                            "t4_fit_pool_trials": 50,
                            "query_start": "strictly_after_rewarded_trial_50",
                        },
                        "frozen_matrix_cells": 180,
                        "claim": "shared_t4_vs_shared_ts4_only",
                        "native_m2_completion_required_before_signature": True,
                        "authorization_created": False,
                        "nonce_claimed": False,
                        "external_subm_data_access_allowed": False,
                        "checkpoint_nwb_normalizer_torch_import_allowed": False,
                        "model_forward_calls": 0,
                        "r2_computations": 0,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if (
            args.authorization is None
            or args.signature is None
            or args.output_root is None
            or args.external_nwb_root is None
        ):
            raise ScoreV3AuthorizationError(
                "score requires detached --authorization/--signature, --output-root, and --external-nwb-root"
            )
        result = execute_from_fixed_prelaunch_v3r2(
            authorization_path=args.authorization,
            signature_path=args.signature,
            output_root=args.output_root,
            external_nwb_root=args.external_nwb_root,
            prelaunch_dir=args.prelaunch_dir,
            root=args.repo_root,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (StaticScoreV3R2Error, ScoreV3AuthorizationError, ScoreV3R2ExecutionError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
