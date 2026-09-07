#!/usr/bin/env python3
"""Audit whether public H1 held-in minival files overlap held-in calibration.

This is a read-only data audit.  It opens only the public H1 held-in-calib and
held-in-minival trees, compares every loader-visible array, and writes one
immutable JSON receipt.  Formal held-out files are never enumerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from falcon_challenge.config import FalconTask
from falcon_challenge.dataloaders import load_nwb


SCHEMA = "h1_heldin_minival_prefix_audit_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_prefix(candidate: np.ndarray, source: np.ndarray) -> bool:
    length = int(candidate.shape[0])
    return length <= int(source.shape[0]) and bool(
        np.array_equal(candidate, source[:length], equal_nan=True)
    )


def session_id(path: Path) -> str:
    marker = "_ses-"
    if marker not in path.stem:
        raise ValueError(f"cannot parse H1 session id from {path}")
    return path.stem.split(marker, 1)[1]


def audit(data_dir: Path) -> dict[str, Any]:
    root = data_dir.resolve()
    if "held-out" in str(root).lower():
        raise ValueError("held-out paths are forbidden in the held-in prefix audit")

    calib_dir = root / "sub-HumanPitt-held-in-calib"
    minival_dir = root / "sub-HumanPitt-held-in-minival"
    if not calib_dir.is_dir() or not minival_dir.is_dir():
        raise FileNotFoundError("expected H1 held-in calib/minival directories")

    calib_files = sorted(calib_dir.glob("*.nwb"))
    minival_files = sorted(minival_dir.glob("*.nwb"))
    if len(calib_files) != 13 or len(minival_files) != 13:
        raise RuntimeError(
            f"expected exactly 13 H1 held-in pairs, got "
            f"{len(calib_files)} calib and {len(minival_files)} minival"
        )

    minival_by_session = {session_id(path): path for path in minival_files}
    if len(minival_by_session) != len(minival_files):
        raise RuntimeError("duplicate H1 minival session identifiers")

    rows: list[dict[str, Any]] = []
    for calib_path in calib_files:
        sid = session_id(calib_path)
        if sid not in minival_by_session:
            raise RuntimeError(f"missing minival file for {sid}")
        minival_path = minival_by_session[sid]

        calib_arrays = load_nwb(calib_path, FalconTask.h1)
        minival_arrays = load_nwb(minival_path, FalconTask.h1)
        names = ("neural", "behavior", "trial_change", "eval_mask")
        flags = {
            name: exact_prefix(minival, calib)
            for name, minival, calib in zip(
                names, minival_arrays, calib_arrays, strict=True
            )
        }
        rows.append(
            {
                "session": sid,
                "calib_path": str(calib_path.resolve()),
                "calib_sha256": sha256_file(calib_path),
                "calib_bins": int(calib_arrays[0].shape[0]),
                "minival_path": str(minival_path.resolve()),
                "minival_sha256": sha256_file(minival_path),
                "minival_bins": int(minival_arrays[0].shape[0]),
                "field_is_bit_exact_prefix": flags,
                "all_loader_fields_are_bit_exact_prefix": all(flags.values()),
            }
        )

    all_prefix = all(row["all_loader_fields_are_bit_exact_prefix"] for row in rows)
    return {
        "schema": SCHEMA,
        "status": (
            "CONFIRMED_13_OF_13_HELDIN_MINIVAL_ARE_CALIB_PREFIXES"
            if all_prefix
            else "NOT_ALL_HELDIN_MINIVAL_ARE_CALIB_PREFIXES"
        ),
        "task": "h1",
        "scope": "public_heldin_only",
        "formal_heldout_opened": False,
        "pair_count": len(rows),
        "all_pairs_all_loader_fields_bit_exact_prefix": all_prefix,
        "loader_fields": ["neural", "behavior", "trial_change", "eval_mask"],
        "interpretation": {
            "local_minival_r2": "implementation_regression_only",
            "independent_heldin_evidence": False,
            "heldout_evidence": False,
            "paper_evalai_private_split_reproduced": False,
        },
        "pairs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {output}")
    payload = audit(args.data_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(output, 0o444)
    print(json.dumps({"output": str(output), "status": payload["status"]}))


if __name__ == "__main__":
    main()
