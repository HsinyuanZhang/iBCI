#!/usr/bin/env python3
"""Print a Track-B v2 no-data target/query plan; never discover or execute data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_contract as base  # noqa: E402
import track_b_v2_target_query_scaffold as target  # noqa: E402


def parse_authority_sha(value: str) -> tuple[str, str]:
    key, separator, digest = value.partition("=")
    if not separator or not key or not target._valid_sha(digest):
        raise argparse.ArgumentTypeError("--source-authority-sha must be NAME=64-lowercase-hex-SHA")
    return key, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("subject_m", "rt"))
    parser.add_argument("--view", choices=("sua", "pseudo_mua"))
    parser.add_argument("--outer-fold-id", required=True)
    parser.add_argument("--target-session-id", required=True)
    parser.add_argument("--pointer-proposal-json", required=True,
                        help="Existing root-audited NOT-MINTED proposal JSON; this script does not mint it.")
    parser.add_argument("--root-audit-attestation-sha256", required=True)
    parser.add_argument("--source-authority-sha", action="append", default=[], type=parse_authority_sha,
                        help="Repeat exact six times: NAME=SHA256")
    args = parser.parse_args()
    if args.dataset == "rt" and args.view is not None:
        parser.error("rt has no --view")
    if args.dataset == "subject_m" and args.view is None:
        parser.error("subject_m requires --view")
    proposal_path = Path(args.pointer_proposal_json)
    try:
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read pointer proposal JSON: {exc}")
    source_authorities = dict(args.source_authority_sha)
    try:
        payload = target.build_no_data_target_query_scaffold(
            dataset=args.dataset,
            view=args.view,
            outer_fold_id=args.outer_fold_id,
            target_session_id=args.target_session_id,
            source_authority_sha256s=source_authorities,
            pointer_proposal_payload=proposal,
            root_audit_attestation_sha256=args.root_audit_attestation_sha256,
        )
    except (base.TrackBV2ContractError, target.TrackBV2TargetQueryScaffoldError) as exc:
        parser.error(str(exc))
    sys.stdout.buffer.write(base.canonical_json_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
