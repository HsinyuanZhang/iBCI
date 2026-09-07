#!/usr/bin/env python3
"""Seal v2 CPU evidence without permitting any formal-oracle action."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "sua_exploration" / "results" / "m1_decoder_latent_alignment_oracle_v2"
PROTOCOL = ROOT / "sua_exploration" / "docs" / "M1_DECODER_LATENT_ALIGNMENT_ORACLE_V2_PROTOCOL.md"
ADDENDUM = OUT / "identity_base_addendum.json"
SCRIPT = ROOT / "sua_exploration" / "scripts" / "audit_m1_decoder_latent_alignment_oracle_base_addendum.py"
TEST = ROOT / "sua_exploration" / "tests" / "test_m1_decoder_latent_alignment_oracle_base_addendum.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> Path:
    target = OUT / "protocol_receipt.json"
    if not OUT.is_dir() or not ADDENDUM.is_file():
        raise FileNotFoundError("v2 addendum must exist before receipt sealing")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    if addendum.get("decision") != "pass_non_circular_base_and_bit_exact_decoder":
        raise ValueError("cannot seal a non-passing base/decoder addendum")
    receipt = {
        "schema_version": "m1_decoder_latent_alignment_oracle_receipt_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hashes": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in {
            "protocol": PROTOCOL, "identity_base_addendum": ADDENDUM, "audit_script": SCRIPT, "audit_test": TEST,
        }.items()},
        "base_and_decoder_status": "non_circular_base_defined_and_f0_decoder_bit_exact_to_teacher",
        "formal_oracle_status": "blocked_pending_root_review",
        "not_executed": ["adapter_fit", "rank_or_lambda_selection", "R2_scoring", "selection_or_report_evaluation",
                         "GPU_launch", "decoder_training", "heldout_access", "EvalAI_access_or_submission"],
    }
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "protocol_receipt.sha256").write_text(f"{sha256(target)}  protocol_receipt.json\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(run())
