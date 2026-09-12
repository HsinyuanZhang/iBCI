"""Cryptographically checked capability for the one formal final-data access phase."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import protocol
from .common import SCHEMA

SEAL_SCHEMA = "dandi688_v2_final_selection_seal"
SEAL_STATUS = "FROZEN_FORMAL_SELECTION"
REQUIRED_CELLS = (
    "full_sua", "full_pmua", "activity_sua", "activity_pmua", "raw_set_sua", "raw_set_pmua",
    "wf_fss_sua", "wf_fss_pmua", "wf_zs_h0_pmua", "diag_z_wf_pmua", "coral_wf_pmua",
    "aligned_fa_wf_pmua", "aligned_fa_stable_wf_pmua", "raw_set_diag_z_pmua", "raw_set_coral_pmua",
)
SUPPLEMENTAL_FULL_CELLS = ("full_sua_s43", "full_pmua_s43", "full_sua_s44", "full_pmua_s44")
FINAL_SCORE_CELLS = (*REQUIRED_CELLS, *SUPPLEMENTAL_FULL_CELLS)
FA_CELLS = frozenset({"aligned_fa_wf_pmua", "aligned_fa_stable_wf_pmua"})


def canonical_payload_sha256(payload: dict[str, Any]) -> str:
    """Hash exactly the seal JSON excluding its self-hash field."""
    body = dict(payload)
    body.pop("sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_and_validate(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SEAL_SCHEMA or payload.get("status") != SEAL_STATUS:
        raise PermissionError("final selection seal is not frozen/formal")
    if payload.get("model_schema") != SCHEMA:
        raise PermissionError("final selection seal does not use the current B3S architecture")
    if payload.get("protocol") != protocol.protocol_dict():
        raise PermissionError("final selection seal protocol mismatch")
    if payload.get("sha256") != canonical_payload_sha256(payload):
        raise PermissionError("final selection seal self-hash mismatch")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise PermissionError("final selection seal has no artifacts")
    artifact_hashes: dict[str, str] = {}
    for raw_path, digest in artifacts.items():
        artifact_path = Path(raw_path)
        if not artifact_path.is_absolute() or not isinstance(digest, str) or len(digest) != 64:
            raise PermissionError("final selection artifact mapping must use absolute SHA-256 paths")
        if not artifact_path.is_file() or _sha256_file(artifact_path) != digest:
            raise PermissionError(f"final selection artifact hash mismatch: {artifact_path}")
        artifact_hashes[str(artifact_path)] = digest
    if tuple(payload.get("required_cells", ())) != REQUIRED_CELLS:
        raise PermissionError("final selection required-cell roster mismatch")
    selections = payload.get("cell_selections")
    if not isinstance(selections, dict) or set(selections) != set(REQUIRED_CELLS):
        raise PermissionError("final selection cell coverage mismatch")
    supplemental = payload.get("supplemental_full_selections")
    if not isinstance(supplemental, dict) or set(supplemental) != set(SUPPLEMENTAL_FULL_CELLS):
        raise PermissionError("supplemental Full selections must contain both representations and both seeds")
    if len((*REQUIRED_CELLS, *supplemental)) != len(FINAL_SCORE_CELLS):
        raise PermissionError("final selection must contain exactly nineteen scored cells")
    encoders = payload.get("encoders")
    if not isinstance(encoders, dict) or set(encoders) != {"sua", "pmua"}:
        raise PermissionError("final selection must seal both representation encoders")
    for representation, encoder in encoders.items():
        if (not isinstance(encoder, dict) or encoder.get("path") not in artifact_hashes
                or encoder.get("sha256") != artifact_hashes[encoder.get("path")]):
            raise PermissionError(f"final selection encoder artifact mismatch: {representation}")
    for cell in FINAL_SCORE_CELLS:
        selection = selections[cell] if cell in selections else supplemental[cell]
        if not isinstance(selection, dict):
            raise PermissionError(f"invalid final selection entry: {cell}")
        status = selection.get("status")
        if status == "SMOKE" or status not in {"SELECTED", "UNAVAILABLE"}:
            raise PermissionError(f"non-formal final selection status: {cell}={status!r}")
        if status == "SELECTED":
            linked = selection.get("artifact_paths")
            if not isinstance(linked, list) or not linked:
                raise PermissionError(f"selected cell lacks artifact paths: {cell}")
            if any(not isinstance(item, str) or item not in artifact_hashes for item in linked):
                raise PermissionError(f"selected cell links unsealed artifact: {cell}")
        else:
            if cell not in FA_CELLS or not isinstance(selection.get("reason"), str) or not selection["reason"]:
                raise PermissionError(f"only FA cells may be unavailable with a reason: {cell}")
            grid = selection.get("grid_artifact")
            if not isinstance(grid, str) or grid not in artifact_hashes:
                raise PermissionError(f"unavailable FA cell needs sealed grid artifact: {cell}")
    return payload, artifact_hashes


@dataclass(frozen=True)
class FinalAccess:
    """A revalidating capability; callers cannot enable final access with a boolean."""

    manifest_path: Path
    manifest_sha256: str
    artifact_hashes: tuple[tuple[str, str], ...]

    @classmethod
    def from_manifest(cls, path: Path) -> "FinalAccess":
        path = Path(path).resolve()
        _payload, artifacts = _load_and_validate(path)
        return cls(path, _sha256_file(path), tuple(sorted(artifacts.items())))

    def validate(self) -> None:
        """Re-read seal and every sealed artifact, catching post-creation tampering."""
        _payload, artifacts = _load_and_validate(self.manifest_path)
        if _sha256_file(self.manifest_path) != self.manifest_sha256:
            raise PermissionError("final selection seal changed after capability creation")
        if tuple(sorted(artifacts.items())) != self.artifact_hashes:
            raise PermissionError("final selection artifact set changed after capability creation")

    def authorize(self, session_id: str) -> None:
        self.validate()
        if session_id not in protocol.FINAL_SESSIONS:
            raise PermissionError("final capability only authorizes the frozen final roster")
