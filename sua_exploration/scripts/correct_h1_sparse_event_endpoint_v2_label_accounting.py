#!/usr/bin/env python3
"""Publish a versioned correction of H-SE5 projected-label accounting only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "sua_exploration/results/h1_sparse_event_endpoint_v2/source_audit.json"
DEFAULT_OUTPUT = ROOT / "sua_exploration/results/h1_sparse_event_endpoint_v2/source_audit_v2r2.json"
EXPECTED_INPUT_SHA = "e4c12cad1e0678dec722bd622fd34a66eef1bf8205e4aac7193e8c968428f47f"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if sha(source) != EXPECTED_INPUT_SHA:
        raise ValueError("H-SE5 V2 accounting correction input SHA mismatch")
    body = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
    if body.get("schema") != "h1_sparse_event_endpoint_v2_source_audit_v1":
        raise ValueError("H-SE5 V2 accounting correction schema mismatch")
    corrected = 0
    for budget in ("M3", "M4"):
        for row in body["budgets"][budget]["sessions"].values():
            events = int(row["support_events"])
            accounting = row["label_accounting"]
            if accounting["acquisition_endpoint_position_scalars"] != events * 14:
                raise ValueError("acquisition endpoint accounting drift")
            accounting["projected_model_input_scalars"] = events * 4
            corrected += 1
    body["schema"] = "h1_sparse_event_endpoint_v2_source_audit_v2r2"
    body["accounting_correction"] = {
        "original_receipt": str(source),
        "original_receipt_sha256": EXPECTED_INPUT_SHA,
        "corrected_field": "budgets.M*.sessions.*.label_accounting.projected_model_input_scalars",
        "old_formula": "support_events * 3 inherited from V1",
        "new_formula": "support_events * 4 for H-SE5 q=4 model input",
        "corrected_session_budget_rows": corrected,
        "scientific_metrics_or_gates_changed": False,
        "correction_script": str(Path(__file__).resolve()),
        "correction_script_sha256": sha(Path(__file__).resolve()),
    }
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(body, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o444); os.link(temporary, output)
    finally:
        if temporary.exists(): temporary.unlink()
    digest = sha(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    fd = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w", encoding="ascii") as handle:
        handle.write(f"{digest}  {output.name}\n"); handle.flush(); os.fsync(handle.fileno())
    print(json.dumps({"output": str(output), "sha256": digest, "rows_corrected": corrected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
