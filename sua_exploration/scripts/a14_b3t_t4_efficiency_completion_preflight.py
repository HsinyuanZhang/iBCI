#!/usr/bin/env python3
"""CPU-only fail-closed preflight for A14 B3T+TS4 completion cell."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SUA = REPO_ROOT / "sua_exploration"
CONTRACT = SUA / "docs/A14_B3T_T4_EFFICIENCY_COMPLETION_CONTRACT_20260812.md"
SCREEN_ID = "sua_b3t_t4_efficiency_v1"
RESULT_ROOT = SUA / "results" / SCREEN_ID
MISSING_ARM = "b3t_ts4"
MISSING_SEED = 42
REQUIRED_EXISTING = ("t4_s42.json", "b3t_t4_s42.json")
AGGREGATOR = SUA / "scripts/aggregate_sua_b3t_t4_efficiency.py"
RUNNER = SUA / "scripts/run_sua_b3t_t4_one_cell.sh"
SEALED = {
    "sub-C_ses-CO-20151113",
    "sub-C_ses-CO-20151116",
    "sub-C_ses-CO-20151117",
    "sub-C_ses-CO-20151119",
    "sub-C_ses-CO-20151120",
    "sub-C_ses-CO-20151201",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args()

    blockers: list[dict[str, str]] = []
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "screen_id": SCREEN_ID,
        "missing_cell": f"{MISSING_ARM}_s{MISSING_SEED}",
        "contract_path": str(CONTRACT),
        "contract_sha256": sha256_file(CONTRACT) if CONTRACT.is_file() else None,
        "gpu_authorized": False,
        "creates_run_root": False,
    }

    if not CONTRACT.is_file():
        blockers.append(fail("MISSING_CONTRACT", str(CONTRACT)))
    for path, label in ((AGGREGATOR, "aggregator"), (RUNNER, "reference runner")):
        if not path.is_file():
            blockers.append(fail("MISSING_BINDING", f"{label}: {path}"))

    if not RESULT_ROOT.is_dir():
        blockers.append(fail("MISSING_PARTIAL_MATRIX", str(RESULT_ROOT)))
    else:
        for name in REQUIRED_EXISTING:
            path = RESULT_ROOT / name
            if not path.is_file():
                blockers.append(fail("MISSING_PREREQUISITE_CELL", name))
        missing_path = RESULT_ROOT / f"{MISSING_ARM}_s{MISSING_SEED}.json"
        if missing_path.exists():
            blockers.append(fail("MISSING_CELL_ALREADY_PRESENT", str(missing_path)))
        aggregate = RESULT_ROOT / "aggregate.json"
        if aggregate.exists():
            blockers.append(fail("AGGREGATE_ALREADY_EXISTS", str(aggregate)))

    # Sealed-session guard via existing receipt session lists
    for name in REQUIRED_EXISTING:
        path = RESULT_ROOT / name
        if path.is_file():
            artifact = json.loads(path.read_text(encoding="utf-8"))
            splits = artifact.get("session_splits") or {}
            for session in splits.get("val") or []:
                if session in SEALED:
                    blockers.append(fail("SEALED_SESSION_IN_RECEIPT", f"{name}: {session}"))

    if not blockers and AGGREGATOR.is_file():
        import subprocess

        proc = subprocess.run(
            [
                sys.executable,
                str(AGGREGATOR),
                "--result-dir",
                str(RESULT_ROOT),
                "--seeds",
                str(MISSING_SEED),
                "--aligned-only",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            blockers.append(
                fail(
                    "ALIGNED_FIRST_GATE_FAILED",
                    (proc.stderr or proc.stdout or "nonzero exit").strip(),
                )
            )
        else:
            receipt["aligned_first_gate"] = json.loads(proc.stdout)

    status = "READY_FOR_SEPARATE_GPU_REVIEW" if not blockers else "STOP_A14_PREFLIGHT_BLOCKERS"
    receipt.update(
        {
            "status": status,
            "implementation_blockers": blockers,
            "expected_new_gpu_cells": 1,
        }
    )
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        if args.receipt.exists():
            raise FileExistsError(f"refusing to overwrite receipt: {args.receipt}")
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text, encoding="utf-8")
    print(text)
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
