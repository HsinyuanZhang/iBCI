#!/usr/bin/env python3
"""Verify that H-SE5 V2r2 changes only q4 projected-label accounting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ORIGINAL_SHA = "e4c12cad1e0678dec722bd622fd34a66eef1bf8205e4aac7193e8c968428f47f"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    path = args.receipt.resolve()
    corrected = json.loads(path.read_text(encoding="utf-8"))
    correction = corrected.pop("accounting_correction")
    if corrected.pop("schema") != "h1_sparse_event_endpoint_v2_source_audit_v2r2":
        raise ValueError("V2r2 schema mismatch")
    original_path = Path(correction["original_receipt"])
    if sha(original_path) != ORIGINAL_SHA or correction["original_receipt_sha256"] != ORIGINAL_SHA:
        raise ValueError("original V2 receipt mismatch")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    original.pop("schema")
    changed = 0
    for budget in ("M3", "M4"):
        for session, row in corrected["budgets"][budget]["sessions"].items():
            current = row["label_accounting"]
            old = original["budgets"][budget]["sessions"][session]["label_accounting"]
            events = row["support_events"]
            if current["projected_model_input_scalars"] != events * 4 or old["projected_model_input_scalars"] != events * 3:
                raise ValueError(f"{budget}/{session}: projected-label correction mismatch")
            current["projected_model_input_scalars"] = old["projected_model_input_scalars"]
            changed += 1
    if corrected != original:
        raise ValueError("V2r2 changed content beyond projected-label accounting")
    if changed != 26 or correction["scientific_metrics_or_gates_changed"] is not False:
        raise ValueError("V2r2 correction manifest mismatch")
    digest = sha(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").split()[0] != digest:
        raise ValueError("V2r2 sidecar mismatch")
    print(json.dumps({"status": "PASS", "receipt": str(path), "sha256": digest, "rows_verified": changed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
