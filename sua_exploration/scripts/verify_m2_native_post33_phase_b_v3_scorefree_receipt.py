#!/usr/bin/env python3
"""Verify the Phase-B score-free receipt; this is not the missing C4 verifier."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "sua_exploration/results/m2_native_post33_phase_b_v3_scorefree_20260804/phase_b_scorefree_receipt.json"
SIDECAR = RECEIPT.with_suffix(".sha256")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if RECEIPT.is_symlink() or not RECEIPT.is_file():
        raise FileNotFoundError(RECEIPT)
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected_sha = SIDECAR.read_text(encoding="utf-8").split()[0]
    if sha256(RECEIPT) != expected_sha:
        raise RuntimeError("Phase-B receipt sidecar mismatch")
    if receipt.get("status") != "NO_GO_C4_FINAL_RUNNER_FINALIZERS_COST_RUNTIME_EVIDENCE_REQUIRED":
        raise ValueError("Phase-B status is not the frozen NO-GO state")
    for field in ("gpu_authorization_created", "accuracy_claim_authorized", "training_authorized"):
        if receipt.get(field) is not False:
            raise ValueError(f"{field} must be false")
    expected_scope = {
        "gpu_used": False,
        "training_started": False,
        "optimizer_steps": 0,
        "new_endpoint_r2_values_read": 0,
        "scorer_modules_imported": 0,
        "formal_sua_paths_resolved": 0,
        "formal_sua_files_opened": 0,
        "evalai_calls": 0,
    }
    if receipt.get("execution_scope") != expected_scope:
        raise ValueError("execution scope drift")
    for relative, metadata in receipt.get("source_map", {}).items():
        source = ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != metadata.get("size_bytes") or sha256(source) != metadata.get("sha256"):
            raise RuntimeError(f"Phase-B source drift: {relative}")
    for lock in receipt.get("upstream_locks", {}).values():
        upstream = Path(lock["path"])
        if upstream.stat().st_size != lock["size_bytes"] or sha256(upstream) != lock["sha256"]:
            raise RuntimeError(f"upstream receipt drift: {upstream}")
        upstream_receipt = json.loads(upstream.read_text(encoding="utf-8"))
        entries = upstream_receipt.get("source_map", {})
        if len(entries) != lock["source_map_entries"]:
            raise RuntimeError("upstream source-map cardinality drift")
        for relative, metadata in entries.items():
            if sha256(ROOT / relative) != metadata["sha256"]:
                raise RuntimeError(f"upstream frozen source drift: {relative}")
    print("PASS_PHASE_B_V3_SCORE_FREE_NO_GO")
    print(RECEIPT)
    print(expected_sha)


if __name__ == "__main__":
    main()

