#!/usr/bin/env python3
"""Metadata-only runner for the blocked external sub-M three-arm V4 package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.subm_co_three_arm_score_only_v4 import (  # noqa: E402
    PreAuthorizationAudit,
    ThreeArmV4BlockedError,
    refuse_blocked_execution,
)
from sua_exploration.scripts.write_dandi688_subm_co_three_arm_score_only_prelaunch_v4 import (  # noqa: E402
    DEFAULT_OUTPUT,
    StaticThreeArmV4Error,
    load_stored_blocked_prelaunch,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "score"), default="dry-run")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--prelaunch-dir", type=Path, default=DEFAULT_OUTPUT)
    # These paths are deliberately never opened while the stored package is
    # blocked.  They define the later authorization-first CLI shape only.
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--signature", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--external-nwb-root", type=Path)
    return parser.parse_args(argv)


def dry_run_payload(prelaunch_dir: Path, root: Path = ROOT) -> dict[str, object]:
    stored = load_stored_blocked_prelaunch(prelaunch_dir, root)
    return {
        "status": stored["status"],
        "matrix": {
            "N": 15,
            "views": ["sua", "pseudo_mua"],
            "seeds": [42, 43, 44],
            "arms": ["shared_t4", "shared_zero4", "shared_ts4"],
            "cell_count": 270,
        },
        "missing_terminal_slots": stored["authority"]["missing_checkpoint_slots"],
        "prelaunch": {
            key: stored[key]
            for key in (
                "draft_sha256",
                "receipt_sha256",
                "blocked_seal_sha256",
                "executable_prelaunch_sealed",
            )
        },
        "authorization_or_signature_read": False,
        "checkpoint_nwb_normalizer_torch_import_allowed": False,
        "model_forward_or_r2_allowed": False,
        "gpu_allowed": False,
        "external_scoring_capability_created": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        # Stored blocked authority is verified before mode-specific behavior.
        load_stored_blocked_prelaunch(args.prelaunch_dir, args.repo_root)
        if args.mode == "dry-run":
            print(json.dumps(dry_run_payload(args.prelaunch_dir, args.repo_root), indent=2, sort_keys=True))
            return 0
        # Fail before even checking whether caller paths exist.  In particular,
        # no authorization/signature/NWB/checkpoint path is opened here.
        audit = PreAuthorizationAudit()
        refuse_blocked_execution(audit=audit)
        raise AssertionError("unreachable")
    except (StaticThreeArmV4Error, ThreeArmV4BlockedError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

