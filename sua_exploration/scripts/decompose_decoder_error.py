#!/usr/bin/env python3
"""Paired decoder error-structure decomposition (forward-only, CPU-only).

Given a paired carrier/control window bundle — or optionally collected from two
matched sealed checkpoints on identical query windows — emit a deterministic JSON
receipt with pre-registered per-stratum deltas.

This scaffolding authorizes pytest on synthetic fixtures only.  It does not
authorize GPU runs, training, or opening sealed formal-test sessions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mc_maze.decoder_error_structure import (  # noqa: E402
    SCHEMA_VERSION,
    assert_sessions_allowed,
    canonical_json_bytes,
    decompose_records,
    records_from_bundle_rows,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_window_bundle(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("window bundle must be a JSON object")
    rows = payload.get("windows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("window bundle must contain a nonempty 'windows' list")
    identities = payload.get("query_window_identities")
    if identities is not None and len(identities) != len(rows):
        raise ValueError("query_window_identities length must match windows")
    return payload


def write_receipt(path: Path, body: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(body) + b"\n")
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-bundle",
        type=Path,
        required=True,
        help="JSON bundle with paired per-window predictions and metadata",
    )
    parser.add_argument(
        "--carrier-checkpoint",
        type=Path,
        required=True,
        help="Carrier-arm checkpoint used to produce pred_carrier (SHA-256 recorded)",
    )
    parser.add_argument(
        "--control-checkpoint",
        type=Path,
        required=True,
        help="Matched control checkpoint used to produce pred_control (SHA-256 recorded)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Deterministic JSON receipt path",
    )
    args = parser.parse_args(argv)

    bundle = load_window_bundle(args.window_bundle)
    session_names = sorted({str(row["session_name"]) for row in bundle["windows"]})
    assert_sessions_allowed(session_names)

    carrier_sha = sha256_file(args.carrier_checkpoint)
    control_sha = sha256_file(args.control_checkpoint)
    records = records_from_bundle_rows(bundle["windows"])
    receipt = decompose_records(
        records,
        carrier_checkpoint_sha256=carrier_sha,
        control_checkpoint_sha256=control_sha,
        query_window_identities=bundle.get("query_window_identities"),
    )
    receipt["window_bundle"] = str(args.window_bundle.resolve())
    receipt["window_bundle_sha256"] = sha256_file(args.window_bundle)
    receipt["schema_version"] = SCHEMA_VERSION
    receipt["sealed_test_sessions_opened"] = False
    receipt["device"] = "cpu"
    receipt_sha = write_receipt(args.output, receipt)
    print(json.dumps({"output": str(args.output), "receipt_sha256": receipt_sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
