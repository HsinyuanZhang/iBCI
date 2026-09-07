#!/usr/bin/env python3
"""Seal the CPU feasibility audit; this writer cannot launch a formal oracle."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "sua_exploration" / "results" / "m1_decoder_latent_alignment_oracle_v1"
PROTOCOL = ROOT / "sua_exploration" / "docs" / "M1_DECODER_LATENT_ALIGNMENT_ORACLE_PROTOCOL.md"
AUDIT = OUT / "feasibility.json"
SCRIPT = ROOT / "sua_exploration" / "scripts" / "audit_m1_decoder_latent_alignment_oracle.py"
TEST = ROOT / "sua_exploration" / "tests" / "test_m1_decoder_latent_alignment_oracle.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> Path:
    target = OUT / "protocol_receipt.json"
    if not OUT.is_dir() or not AUDIT.is_file():
        raise FileNotFoundError("feasibility artifact must exist before receipt sealing")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("decision") != "canonical_target_defined_cpu_feasibility_pass":
        raise ValueError("receipt cannot seal a non-passing canonical-target feasibility audit")
    receipt = {
        "schema_version": "m1_decoder_latent_alignment_oracle_receipt_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hashes": {
            "protocol": {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": sha256(PROTOCOL)},
            "feasibility": {"path": str(AUDIT.relative_to(ROOT)), "sha256": sha256(AUDIT)},
            "audit_script": {"path": str(SCRIPT.relative_to(ROOT)), "sha256": sha256(SCRIPT)},
            "audit_test": {"path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)},
        },
        "canonical_target_status": "defined_cpu_inference_feasibility_only",
        "formal_oracle_status": "blocked_pending_root_review",
        "not_executed": [
            "rank_or_lambda_fit", "future_rate_or_behavior_scoring", "selection_or_report_evaluation",
            "GPU_launch", "decoder_training", "EvalAI_access_or_submission",
        ],
    }
    with target.open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    (OUT / "protocol_receipt.sha256").write_text(f"{sha256(target)}  protocol_receipt.json\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(run())
