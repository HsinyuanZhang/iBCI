#!/usr/bin/env python3
"""Run the frozen H1 Version-B phase/event CPU source-only gate.

No R2 metric, decoder, training configuration, target optimizer, GPU, held-out
file, minival file, formal label, or EvalAI API is imported by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import uuid
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.h1_phase_event_carrier_vb import (  # noqa: E402
    PROPOSAL_SCHEMA,
    SCHEMA,
    PhaseEventCarrierError,
    audit_sessions,
    load_all_allowed_sessions,
    load_frozen_proposal,
)


ART = ROOT / "pilot_artifacts/h1_phase_event_carrier_vb"
PROPOSAL = ART / "H1_PHASE_EVENT_CARRIER_VB_PROPOSAL_v1.json"
OUTPUT = ART / "H1_PHASE_EVENT_CARRIER_VB_SOURCE_GATE_v1.json"
DATA_ROOT = ROOT / "data/000954"
SOURCE = ROOT / "src/data/h1_phase_event_carrier_vb.py"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_immutable_json(path: str | Path, value: Mapping[str, Any]) -> str:
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"Version-B source gate refuses to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    if not output.is_file() or output.is_symlink() or stat.S_IMODE(output.stat().st_mode) != 0o444:
        raise PhaseEventCarrierError(f"source-gate receipt was not published immutable 0444: {output}")
    return hashlib.sha256(encoded).hexdigest()


def run(data_root: str | Path = DATA_ROOT) -> dict[str, Any]:
    proposal = load_frozen_proposal(PROPOSAL)
    sessions = load_all_allowed_sessions(data_root)
    result = audit_sessions(sessions)
    result["proposal"] = {
        "path": str(PROPOSAL),
        "sha256": sha256_file(PROPOSAL),
        "schema": proposal["schema"],
        "status": proposal["status"],
    }
    result["implementation"] = {
        "source_path": str(SOURCE),
        "source_sha256": sha256_file(SOURCE),
        "audit_path": str(Path(__file__).resolve()),
        "audit_sha256": sha256_file(Path(__file__).resolve()),
    }
    if result.get("schema") != SCHEMA or proposal.get("schema") != PROPOSAL_SCHEMA:
        raise PhaseEventCarrierError("source-gate/proposal schema binding failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    body = run(args.data_root)
    digest = write_immutable_json(args.output, body)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": digest, "status": body["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
