from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dandi688_bench_v2 import data, protocol
from dandi688_bench_v2.common import SCHEMA
from dandi688_bench_v2.final_access import (FA_CELLS, REQUIRED_CELLS, SUPPLEMENTAL_FULL_CELLS,
                                             FinalAccess, canonical_payload_sha256)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal(tmp_path: Path) -> tuple[Path, Path]:
    artifact = (tmp_path / "formal-artifact.json").resolve()
    artifact.write_text('{"formal": true}\n')
    selections = {}
    for cell in REQUIRED_CELLS:
        if cell in FA_CELLS:
            selections[cell] = {"status": "UNAVAILABLE", "reason": "formal FA rank gate", "grid_artifact": str(artifact)}
        else:
            selections[cell] = {"status": "SELECTED", "artifact_paths": [str(artifact)]}
    supplemental = {cell: {"status": "SELECTED", "artifact_paths": [str(artifact)]}
                    for cell in SUPPLEMENTAL_FULL_CELLS}
    payload = {"schema": "dandi688_v2_final_selection_seal", "status": "FROZEN_FORMAL_SELECTION",
               "model_schema": SCHEMA,
               "protocol": protocol.protocol_dict(), "artifacts": {str(artifact): _sha(artifact)},
               "required_cells": list(REQUIRED_CELLS), "cell_selections": selections,
               "supplemental_full_selections": supplemental,
               "encoders": {representation: {"path": str(artifact), "sha256": _sha(artifact)}
                            for representation in ("sua", "pmua")}}
    payload["sha256"] = canonical_payload_sha256(payload)
    seal = tmp_path / "selection_seal.json"; seal.write_text(json.dumps(payload, sort_keys=True))
    return seal, artifact


def test_valid_capability_authorizes_only_exact_final_roster_without_raw_open(tmp_path, monkeypatch):
    seal, _artifact = _seal(tmp_path)
    capability = FinalAccess.from_manifest(seal)
    monkeypatch.setattr(data, "NWBHDF5IO", lambda *a, **k: (_ for _ in ()).throw(AssertionError("raw NWB opened")))
    # A nonexistent root proves authorization completes before any raw NWB reader is invoked.
    with pytest.raises(FileNotFoundError):
        data.load_pair(protocol.FINAL_SESSIONS[0], purpose="final", final_access=capability,
                       raw_root=tmp_path / "absent")
    with pytest.raises(PermissionError):
        capability.authorize(protocol.TRAIN_SESSIONS[0])


def test_final_raw_is_denied_before_path_or_nwb_open_without_capability(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "NWBHDF5IO", lambda *a, **k: (_ for _ in ()).throw(AssertionError("raw NWB opened")))
    with pytest.raises(PermissionError):
        data.load_pair(protocol.FINAL_SESSIONS[0], purpose="final", raw_root=tmp_path / "absent")


@pytest.mark.parametrize("mutate", ["seal", "artifact", "smoke", "coverage", "old_film", "missing_supplement", "incomplete_seeds"])
def test_tampered_or_incomplete_seal_is_denied(tmp_path, mutate):
    seal, artifact = _seal(tmp_path)
    payload = json.loads(seal.read_text())
    if mutate == "seal":
        payload["status"] = "DRAFT"
        seal.write_text(json.dumps(payload))
    elif mutate == "artifact":
        artifact.write_text('{"formal": false}\n')
    elif mutate == "smoke":
        payload["cell_selections"]["full_sua"]["status"] = "SMOKE"
        payload["sha256"] = canonical_payload_sha256(payload)
        seal.write_text(json.dumps(payload))
    elif mutate == "coverage":
        payload["cell_selections"].pop("raw_set_coral_pmua")
        payload["sha256"] = canonical_payload_sha256(payload)
        seal.write_text(json.dumps(payload))
    elif mutate == "old_film":
        payload["model_schema"] = "dandi688_v2_2015_move_t4_learned"
        payload["sha256"] = canonical_payload_sha256(payload)
        seal.write_text(json.dumps(payload))
    elif mutate == "missing_supplement":
        payload["supplemental_full_selections"] = {}
        payload["sha256"] = canonical_payload_sha256(payload)
        seal.write_text(json.dumps(payload))
    else:
        payload["supplemental_full_selections"] = {"full_sua_s43": payload["cell_selections"]["full_sua"]}
        payload["sha256"] = canonical_payload_sha256(payload)
        seal.write_text(json.dumps(payload))
    with pytest.raises(PermissionError):
        FinalAccess.from_manifest(seal)


def test_capability_rereads_seal_and_artifact_before_access(tmp_path):
    seal, artifact = _seal(tmp_path)
    capability = FinalAccess.from_manifest(seal)
    artifact.write_text("tampered\n")
    with pytest.raises(PermissionError):
        capability.authorize(protocol.FINAL_SESSIONS[0])
