#!/usr/bin/env python3
"""Verify a dated H-SE5 source snapshot; build only explicit temp bundles.

The official 19250108 authority is published by
``h1_hse5_lodo_date2_preflight.py`` from the same live source module that it
audits.  This utility intentionally cannot create that official path: a
standalone rebuild would introduce a second SVD/cache realization and break
the preflight-to-training binding.  Its optional build mode exists only for
isolated tests and requires both ``--temp-mode`` and an explicit expected
source-manifest SHA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
for directory in (REPOSITORY, ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from src.data.h1_sparse_event_endpoint_dated import H1SparseEventDatedDataModule
from src.data.h1_sparse_event_source_snapshot_dated import load_snapshot, write_snapshot


OFFICIAL_ROOT = ROOT / "pilot_artifacts/h1_hse5_lodo_19250108"
OFFICIAL_SNAPSHOT = OFFICIAL_ROOT / "H1_HSE5_LODO_19250108_SOURCE_v1.npz"
OFFICIAL_RECEIPT = OFFICIAL_ROOT / "H1_HSE5_LODO_19250108_SOURCE_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", type=Path, help="immutable receipt to verify without opening NWB files")
    parser.add_argument("--fold-date", default="19250108")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/000954")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--expected-manifest-sha256", default=None)
    parser.add_argument("--temp-mode", action="store_true", help="permit a non-official, write-once test snapshot build")
    args = parser.parse_args()
    if args.verify_only:
        if any(value is not None for value in (args.cache_dir, args.snapshot, args.receipt, args.expected_manifest_sha256)) or args.temp_mode:
            parser.error("--verify-only cannot be combined with build arguments")
        snapshot = load_snapshot(args.verify_only)
        result = {
            "status": "PASS", "fold_date": snapshot.basis.outer_date,
            "receipt": str(snapshot.receipt_path), "snapshot": str(snapshot.snapshot_path),
            "manifest_sha256": snapshot.manifest_sha256, "schedule_sha256": snapshot.schedule_sha256,
        }
    else:
        if not args.temp_mode:
            parser.error("official source authority is published only by h1_hse5_lodo_date2_preflight.py; standalone build requires --temp-mode")
        if not args.expected_manifest_sha256:
            parser.error("standalone temp build requires --expected-manifest-sha256")
        if any(value is None for value in (args.cache_dir, args.snapshot, args.receipt)):
            parser.error("standalone temp build requires --cache-dir, --snapshot, and --receipt")
        snapshot_path, receipt_path = args.snapshot.resolve(), args.receipt.resolve()
        if snapshot_path in {OFFICIAL_SNAPSHOT.resolve(), OFFICIAL_RECEIPT.resolve()} or receipt_path in {OFFICIAL_SNAPSHOT.resolve(), OFFICIAL_RECEIPT.resolve()}:
            parser.error("--temp-mode must not target an official H-SE5 authority path")
        if str(args.fold_date) == "19250101":
            parser.error("dated temp builder refuses the sealed fold-0 date")
        module = H1SparseEventDatedDataModule(
            task="h1", data_dir=str(args.data_dir.resolve()), cache_dir=str(args.cache_dir.resolve()), fold_date=str(args.fold_date),
        )
        module.setup("fit")
        result = write_snapshot(
            snapshot_path=snapshot_path, receipt_path=receipt_path, source_module=module,
            builder_path=Path(__file__), expected_manifest_sha256=args.expected_manifest_sha256,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
