#!/usr/bin/env python3
"""Verify or atomically claim the fixed AC4-RS4 v1 prelaunch receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / (
    "sua_exploration/results/t4_m30_ac4_rs4_prelaunch_v1_20260804/receipt.json"
)
CLAIM = ROOT / (
    "sua_exploration/results/sua_t4_m30_ac4_rs4_v1/authorization_claim.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict:
    if not RECEIPT.is_file():
        raise PermissionError("AC4-RS4 prelaunch receipt missing")
    receipt = json.loads(RECEIPT.read_text())
    if receipt.get("status") != "PASS" or receipt.get("gpu_launch_authorized") is not True:
        raise PermissionError("AC4-RS4 GPU launch is not authorized")
    if receipt.get("formal_test_opened") is not False:
        raise PermissionError("formal SUA scope violation")
    for relative, expected in receipt["source_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise PermissionError(f"source drift: {relative}")
    return receipt


def claim() -> dict:
    receipt = verify()
    CLAIM.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "receipt_sha256": sha256(RECEIPT),
        "authorization_id": receipt["authorization_id"],
    }
    try:
        descriptor = os.open(CLAIM, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise PermissionError("AC4-RS4 authorization already claimed") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    return receipt


def require_claim() -> dict:
    receipt = verify()
    if not CLAIM.is_file():
        raise PermissionError("AC4-RS4 authorization claim missing")
    claim_payload = json.loads(CLAIM.read_text())
    if claim_payload != {
        "receipt_sha256": sha256(RECEIPT),
        "authorization_id": receipt["authorization_id"],
    }:
        raise PermissionError("AC4-RS4 claim drift")
    return {
        "authorization_id": receipt["authorization_id"],
        "receipt_sha256": sha256(RECEIPT),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true")
    group.add_argument("--claim", action="store_true")
    group.add_argument("--require-claim", action="store_true")
    args = parser.parse_args()
    if args.claim:
        claim()
    elif args.require_claim:
        require_claim()
    else:
        verify()


if __name__ == "__main__":
    main()
