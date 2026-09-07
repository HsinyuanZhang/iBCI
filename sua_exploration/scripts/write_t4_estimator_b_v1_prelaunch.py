#!/usr/bin/env python3
"""Write-once, no-data prelaunch receipt for Experiment B v1."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUA = ROOT / "sua_exploration"
sys.path.insert(0, str(SUA))

from scripts.audit_t4_estimator_b_v1 import build_dry_run_receipt, strict_json  # noqa: E402

MANIFEST = SUA / "configs/subc_co_27_6_strict_train_val_manifest.json"
OUT = SUA / "results/t4_estimator_b_v1_prelaunch/receipt.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"write-once receipt exists: {OUT}")
    receipt = build_dry_run_receipt(MANIFEST)
    receipt["writer_sha256"] = sha256(Path(__file__))
    receipt["write_once_output"] = str(OUT.resolve())
    # The namespace directory can exist from an interrupted receipt attempt; write-once is
    # enforced by the file collision above, not by directory creation.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(strict_json(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(OUT), "sha256": sha256(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
