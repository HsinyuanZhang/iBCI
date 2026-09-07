#!/usr/bin/env python3
"""Build or verify an immutable, hash-bound H-SE5 source snapshot.

Build mode is source-only: it constructs ``H1SparseEventDataModule`` and calls
exactly ``setup('fit')`` before writing the snapshot.  Verify mode reads only
the immutable snapshot and receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.h1_sparse_event_endpoint import H1SparseEventDataModule
from src.data.h1_sparse_event_source_snapshot import load_snapshot, write_snapshot


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _build(args: argparse.Namespace) -> dict[str, object]:
    _need(args.expected_manifest_sha256 is not None and len(args.expected_manifest_sha256) == 64,
          "build requires --expected-manifest-sha256 as a full 64-character SHA-256")
    _need(args.snapshot is not None and args.receipt is not None, "build requires --snapshot and --receipt")
    source = H1SparseEventDataModule(task="h1", data_dir=str(args.data_dir.resolve()), cache_dir=str(args.cache_dir.resolve()))
    # Contractually the sole DataModule entry point for this builder.
    source.setup("fit")
    return write_snapshot(snapshot_path=args.snapshot, receipt_path=args.receipt, source_module=source,
                          expected_manifest_sha256=args.expected_manifest_sha256, builder_path=Path(__file__))


def _verify(receipt: Path) -> dict[str, object]:
    snapshot = load_snapshot(receipt)
    return {
        "status": "PASS", "receipt": str(snapshot.receipt_path), "receipt_sha256": snapshot.receipt_sha256,
        "snapshot": str(snapshot.snapshot_path), "snapshot_sha256": snapshot.snapshot_sha256,
        "manifest_sha256": snapshot.manifest_sha256, "basis_sha256": snapshot.basis.basis_sha256,
        "normalizer_sha256": snapshot.normalizer.normalizer_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", type=Path, help="immutable snapshot receipt to load and verify")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/000954")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "pilot_artifacts/h1_sparse_event_endpoint/shared_source_cache")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    result = _verify(args.verify_only) if args.verify_only else _build(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
