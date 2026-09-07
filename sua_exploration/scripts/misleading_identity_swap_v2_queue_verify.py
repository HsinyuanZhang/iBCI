#!/usr/bin/env python3
"""Verified, read-only queue readiness check for swap-v2 terminal receipts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
for entry in (SUA_ROOT, REPO_ROOT / "streaming_calibration_exp"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from mc_maze import misleading_identity_swap_v2_core as core
from scripts.misleading_identity_swap_v2_preflight import (
    current_implementation_bindings, load_verified_official_preflight,
)


def verify_terminal(
    path: Path, *, cell: str, official: Mapping[str, Any], official_sha: str,
    live: Mapping[str, Any], require_canonical_path: bool = True,
) -> dict[str, Any]:
    body_exists = os.path.lexists(path)
    side = core.sidecar_path(path)
    side_exists = os.path.lexists(side)
    if not body_exists and not side_exists:
        return {"path": str(path), "ready": False, "status": "MISSING_NOT_LAUNCHED"}
    core.require(body_exists and side_exists, f"{cell}: partial terminal immutable pair")
    if require_canonical_path:
        expected = Path(official["cell_output_paths"][cell]) / "terminal_receipt.json"
        core.require(path.resolve() == expected.resolve(), f"{cell}: noncanonical terminal path")
    payload, digest = core.load_verified_immutable_json(path)
    core.require(payload.get("receipt_kind") == "misleading_identity_swap_v2_cell_terminal" and
                 payload.get("status") == "CELL_TRAINING_COMPLETE__DEVELOPMENT_NOT_OFFICIAL" and
                 payload.get("cell") == cell, f"{cell}: terminal status/kind drift")
    core.require(payload.get("official_preflight_path") == str(core.OFFICIAL_PREFLIGHT_PATH.resolve()) and
                 payload.get("official_preflight_sha256") == official_sha,
                 f"{cell}: terminal official preflight drift")
    core.require(payload.get("implementation_bindings") == live and
                 payload.get("implementation_bindings_sha256") == official["implementation_bindings_sha256"],
                 f"{cell}: terminal live closure drift")
    for key in ("matching_authority_path", "matching_authority_sha256", "source_lineage_path",
                "source_lineage_sha256", "initial_state_path", "initial_state_file_sha256",
                "initial_state_dict_sha256", "cell_output_root"):
        core.require(payload.get(key) == official.get(key), f"{cell}: terminal {key} drift")
    core.require(payload.get("cell_output_path") == official["cell_output_paths"][cell],
                 f"{cell}: terminal output path drift")
    return {"path": str(path), "sha256": digest, "ready": True, "status": payload["status"]}


def verify_queue(official_path: Path, lineage_path: Path) -> dict[str, Any]:
    official, official_sha = load_verified_official_preflight(official_path)
    lineage, lineage_sha = core.load_verified_immutable_json(lineage_path)
    core.require(lineage_path.resolve() == core.SOURCE_LINEAGE_PATH.resolve() and
                 lineage_path.resolve() == Path(official["source_lineage_path"]).resolve() and
                 lineage_sha == official["source_lineage_sha256"],
                 "queue source lineage differs from canonical official preflight")
    live = current_implementation_bindings()
    rows = {}
    for cell in core.CELLS:
        path = Path(official["cell_output_paths"][cell]) / "terminal_receipt.json"
        rows[cell] = verify_terminal(
            path, cell=cell, official=official, official_sha=official_sha, live=live
        )
    return {
        "status": "DRY_RUN__NO_TARGET_NO_GPU",
        "terminal_cells": rows,
        "official_preflight_path": str(core.OFFICIAL_PREFLIGHT_PATH.resolve()),
        "official_preflight_sha256": official_sha,
        "source_lineage_path": str(lineage_path.resolve()),
        "source_lineage_sha256": lineage_sha,
        "next": "independent review then explicit target-authority and 16-score authorization",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-preflight", type=Path, default=core.OFFICIAL_PREFLIGHT_PATH)
    parser.add_argument("--source-lineage", type=Path, default=core.SOURCE_LINEAGE_PATH)
    args = parser.parse_args()
    print(json.dumps(verify_queue(args.official_preflight, args.source_lineage), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
