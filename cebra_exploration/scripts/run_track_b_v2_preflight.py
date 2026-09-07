#!/usr/bin/env python3
"""No-data, no-GPU runner for the H1-excluded Track-B v2 contract.

This is intentionally *not* a CEBRA scorer.  It has no data-path option and
does not import CEBRA, the old comparator, an NWB loader, or a checkpoint.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any


# Set before importing any project module. The script has no CUDA API calls.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONNOUSERSITE"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC = REPO_ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_contract as contract


def _parse_session_count(value: str) -> tuple[str, int]:
    try:
        session_id, raw_count = value.split("=", 1)
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--source-unit-count expects SESSION_ID=POSITIVE_INTEGER") from exc
    if not session_id or session_id != session_id.strip() or count <= 0:
        raise argparse.ArgumentTypeError("--source-unit-count expects SESSION_ID=POSITIVE_INTEGER")
    return session_id, count


def _write_if_requested(*, output_path: Path | None, payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "status": payload["status"],
        "dataset": payload.get("dataset"),
        "view": payload.get("view"),
        "real_data_opened": payload.get("real_data_opened", False),
        "gpu_used": payload.get("gpu_used", False),
        "scoring_executed": payload.get("scoring_executed", False),
    }
    if output_path is not None:
        receipt = contract.write_immutable_receipt(output_path, payload)
        summary["receipt_path"] = receipt["body_path"]
        summary["receipt_sha256"] = receipt["body_sha256"]
        summary["receipt_sidecar_path"] = receipt["sidecar_path"]
        summary["receipt_sidecar_sha256"] = receipt["sidecar_sha256"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Track-B v2 H1-excluded no-data preflight")
    parser.add_argument("--dataset", required=True, choices=contract.ALLOWED_DATASETS)
    parser.add_argument("--view", choices=contract.SUBJECT_M_VIEWS, default=None)
    parser.add_argument(
        "--unified-unseen-serviceability",
        action="store_true",
        help="Emit only the structural UnifiedSolver unseen-session verdict; no CEBRA/data call.",
    )
    parser.add_argument(
        "--source-unit-count",
        action="append",
        type=_parse_session_count,
        default=[],
        metavar="SESSION_ID=N",
        help="Metadata-only source unit count; required with --unified-unseen-serviceability.",
    )
    parser.add_argument("--target-session-id", default=None)
    parser.add_argument("--target-unit-count", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional new O_EXCL/0444 JSON receipt path.")
    parser.add_argument(
        "--score",
        action="store_true",
        help="Always refused: this v2 runner has no data or scoring implementation.",
    )
    args = parser.parse_args()

    if args.score:
        contract.refuse_score_execution()
    dataset, view = contract.validate_scope(args.dataset, args.view)
    if args.unified_unseen_serviceability:
        contract.require(args.source_unit_count, "--unified-unseen-serviceability requires --source-unit-count")
        contract.require(args.target_session_id is not None, "--unified-unseen-serviceability requires --target-session-id")
        contract.require(args.target_unit_count is not None, "--unified-unseen-serviceability requires --target-unit-count")
        ids = [item[0] for item in args.source_unit_count]
        counts = [item[1] for item in args.source_unit_count]
        payload = contract.build_unified_unseen_serviceability_receipt(
            dataset=dataset,
            view=view,
            source_session_ids=ids,
            source_unit_counts=counts,
            target_session_id=args.target_session_id,
            target_unit_count=args.target_unit_count,
        )
    else:
        contract.require(not args.source_unit_count, "--source-unit-count is only valid with --unified-unseen-serviceability")
        contract.require(args.target_session_id is None and args.target_unit_count is None, "target metadata is only valid with --unified-unseen-serviceability")
        payload = contract.build_no_data_preflight(dataset=dataset, view=view)

    import json

    summary = _write_if_requested(output_path=args.output, payload=payload)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
