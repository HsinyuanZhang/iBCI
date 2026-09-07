#!/usr/bin/env python3
"""Dry-plan or root-publish the canonical RT 15-session local asset authority.

No input path is discovered.  A future authorised root invocation must provide
15 explicit JSON ``--asset-row`` values and the two execution flags.  The
current task permits only the default dry plan; it must not mint the pair.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


os.environ["PYTHONNOUSERSITE"] = "1"
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_contract as base  # noqa: E402
import track_b_v2_rt_local_asset_authority as authority  # noqa: E402


def _parse_row(value: str) -> dict:
    try:
        row = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("asset row must be a JSON object") from exc
    if not isinstance(row, dict):
        raise argparse.ArgumentTypeError("asset row must be a JSON object")
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-row", action="append", type=_parse_row, default=[],
                        help="Explicit JSON row with session_id, canonical_path, byte_size, sha256; repeat 15 times.")
    parser.add_argument("--output", type=Path, default=authority.CANONICAL_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--i-am-root-publisher", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.expanduser().absolute()
    try:
        authority.require(output == authority.CANONICAL_OUTPUT,
                          "alternate RT local asset authority output is forbidden")
        authority.require(args.execute == args.i_am_root_publisher,
                          "publishing requires both --execute and --i-am-root-publisher")
        if not args.execute:
            authority.require(not args.asset_row, "dry plan accepts no asset rows")
            sys.stdout.buffer.write(base.canonical_json_bytes({
                "schema": "track_b_v2_rt_local_asset_authority_root_publisher_dry_plan_v1",
                "status": "DRY_PLAN_ONLY__NO_RECEIPT_READ__NO_DATA__NO_WRITE",
                "canonical_output": str(authority.CANONICAL_OUTPUT),
                "required_parent_body_sha256": {
                    "rt_metric_pointer": authority.POINTER_SHA256,
                    "rt_full15_source_manifest": authority.MANIFEST_SHA256,
                    "rt_full15_source_aggregate": authority.AGGREGATE_SHA256,
                },
                "required_explicit_asset_row_count": 15,
                "required_asset_row_fields": ["session_id", "canonical_path", "byte_size", "sha256"],
                "directory_glob_scan_or_available_tree_inference_permitted": False,
                "data_file_opened": False,
                "data_file_statted": False,
                "data_file_rehashed": False,
                "target_or_formal_data_opened": False,
                "cebra_imported": False,
                "gpu_used": False,
                "receipt_minted": False,
            }))
            return 0
        authority.require(not os.path.lexists(output) and
                          not os.path.lexists(output.with_name(f"{output.name}.sha256")),
                          "canonical RT local asset authority output pair must be fresh")
        authority.require(len(args.asset_row) == 15,
                          "root publisher requires exactly 15 explicit --asset-row values")
        published = authority.publish_authority(
            root_asset_rows=args.asset_row, entrypoint=Path(__file__).absolute(),
        )
    except (authority.TrackBV2RTLocalAssetAuthorityError,
            authority.live.TrackBV2LiveContractError,
            authority.source.TrackBV2SourceAdapterError,
            authority.metric_pointer.TrackBV2MetricPointerAuthorityError) as exc:
        parser.error(str(exc))
    sys.stdout.buffer.write(base.canonical_json_bytes({
        "status": authority.STATUS, "published": published,
        "target_or_formal_data_opened": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
