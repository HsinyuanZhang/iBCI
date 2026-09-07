#!/usr/bin/env python3
"""CPU-only fail-closed preflight for A10 no-backprop cost matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SUA = REPO_ROOT / "sua_exploration"
CONTRACT = SUA / "docs/A10_NO_BACKPROP_COST_CONTRACT_20260812.md"
SCREEN_ID = "a10_no_backprop_cost_v1"
SUPERSEDED_REASON = (
    "A10 v1 is superseded: gradient-free direction-label calibration and "
    "behavior-label weight adaptation do not have matched supervision, and "
    "the adaptation entrypoint is intentionally incomplete"
)
RESULT_ROOT = SUA / "results" / SCREEN_ID
MAINLINE_ROOT = SUA / "results/sua_spint_t4_mainline_fp32_v1"
MANIFEST = SUA / "configs/subc_co_27_6_strict_train_val_manifest.json"
TEACHER = SUA / "checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
EVAL_SCRIPT = SUA / "scripts/eval_adaptation_dandi688.py"
SEEDS = (42, 43, 44)
ARMS = ("gradient_free", "readout_probe", "full_finetune")
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
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()

    result_root = args.result_root.expanduser().resolve()

    blockers: list[dict[str, str]] = []
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "screen_id": SCREEN_ID,
        "contract_path": str(CONTRACT),
        "contract_sha256": sha256_file(CONTRACT) if CONTRACT.is_file() else None,
        "gpu_authorized": False,
        "creates_run_root": False,
    }

    if not CONTRACT.is_file():
        blockers.append(fail("MISSING_CONTRACT", str(CONTRACT)))
    for path, label in (
        (MANIFEST, "manifest"),
        (TEACHER, "teacher"),
        (EVAL_SCRIPT, "eval_adaptation"),
    ):
        if not path.is_file():
            blockers.append(fail("MISSING_BINDING", f"{label}: {path}"))

    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        splits = manifest.get("session_splits") or manifest
        for key in ("train", "val", "test"):
            hit = [s for s in (splits.get(key) or []) if s in SEALED]
            if hit:
                blockers.append(fail("SEALED_SESSION_IN_MANIFEST", f"{key}: {hit}"))

    for seed in SEEDS:
        mainline = MAINLINE_ROOT / f"t4_s{seed}.json"
        if not mainline.is_file():
            blockers.append(fail("MISSING_MAINLINE_SOURCE", str(mainline)))

    if result_root.exists():
        for arm in ARMS:
            for seed in SEEDS:
                out = result_root / f"{arm}_s{seed}.json"
                if out.exists():
                    blockers.append(fail("NONEMPTY_OUTPUT_ROOT", str(out)))
        agg = result_root / "aggregate.json"
        if agg.exists():
            blockers.append(fail("NONEMPTY_OUTPUT_ROOT", str(agg)))

    # A10 v1 is retained only as a historical design record.  Even a clean
    # filesystem/preflight must not turn it back into an executable contract.
    blockers.append(fail("SUPERSEDED_A10_V1", SUPERSEDED_REASON))
    status = "STOP_A10_V1_SUPERSEDED"
    receipt.update(
        {
            "status": status,
            "implementation_blockers": blockers,
            "expected_gpu_cells": len(ARMS) * len(SEEDS),
            "arms": list(ARMS),
            "seeds": list(SEEDS),
        }
    )
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        if args.receipt.exists():
            raise FileExistsError(f"refusing to overwrite receipt: {args.receipt}")
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text, encoding="utf-8")
    print(text)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
