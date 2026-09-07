#!/usr/bin/env python3
"""Run only Track-B v2's official-pointer-gated, no-target CPU adapter preflight."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_contract as base  # noqa: E402
import track_b_v2_live_contract as live  # noqa: E402
import track_b_v2_source_adapter as source  # noqa: E402
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
    parser.add_argument("--target-session-id", required=True, help="Opaque ID only; no target data are opened.")
    parser.add_argument("--official-pointer-body", required=True, help="Root-minted immutable pointer body.")
    parser.add_argument("--source-authority-sha", action="append", default=[], type=parse_authority_sha,
                        help="Repeat exact six times: NAME=SHA256")
    args = parser.parse_args()
    if args.dataset == "subject_m" and args.view is None:
        parser.error("subject_m requires --view")
    if args.dataset == "rt" and args.view is not None:
        parser.error("rt has no --view")
    pointer_body = Path(args.official_pointer_body)
    pair = live.ExplicitSealedReceiptPair(
        role="canonical_reference_body_pointer",
        body_path=pointer_body,
        sidecar_path=pointer_body.with_name(f"{pointer_body.name}.sha256"),
    )
    # RT has no single source roster.  Rebuild its explicit 15-fold topology
    # from the verified lineage body in memory; this reads no NWB/target data
    # and still leaves every fold's source materialization unexecuted.
    rt_plan = source.build_rt_15fold_source_authority_plan() if args.dataset == "rt" else None
    try:
        payload = target.build_source_only_target_adapter_preflight(
            dataset=args.dataset, view=args.view, outer_fold_id=args.outer_fold_id,
            target_session_id=args.target_session_id, source_authority_sha256s=dict(args.source_authority_sha),
            official_metric_pointer_pair=pair, rt_15fold_source_authority_plan=rt_plan,
        )
    except (base.TrackBV2ContractError, target.TrackBV2TargetQueryScaffoldError) as exc:
        parser.error(str(exc))
    sys.stdout.buffer.write(base.canonical_json_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
