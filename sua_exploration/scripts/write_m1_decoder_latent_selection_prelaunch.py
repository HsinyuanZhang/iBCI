#!/usr/bin/env python3
"""Seal the CPU-only selection-stage implementation without loading any query data."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "sua_exploration" / "results" / "m1_decoder_latent_alignment_oracle_v2"
PROTOCOL = ROOT / "sua_exploration" / "docs" / "M1_DECODER_LATENT_ALIGNMENT_ORACLE_V2_PROTOCOL.md"
MODULE = ROOT / "sua_exploration" / "scripts" / "m1_decoder_latent_selection_stage.py"
RUNNER = ROOT / "sua_exploration" / "scripts" / "m1_decoder_latent_selection_runner.py"
TEST = ROOT / "sua_exploration" / "tests" / "test_m1_decoder_latent_selection_stage.py"
ADDENDUM = OUT / "identity_base_addendum.json"
V1_AUDIT = ROOT / "sua_exploration" / "results" / "m1_decoder_latent_alignment_oracle_v1" / "feasibility.json"
F0 = ROOT / "streaming_calibration_exp" / "outputs" / "streaming_calibration" / "m1_clean_selection_v1_f0_m1_f1_s42_20260801_192017" / "checkpoints" / "best.ckpt"
TEACHER = ROOT / "SPINT-main" / "logs" / "train" / "runs" / "2026-07-21-19-11-01" / "checkpoints" / "best_ckpt" / "epoch_019.ckpt"
HASHES = {F0: "1ec318f81cfaa9f6eb5e998a2b34135e2bde9941c48fb47bc46f47632f0d6cd8", TEACHER: "c81a2bbd860452e6186a9ecf55c0b747da61baef4fae3212f61521be68cc5ac2"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> Path:
    target = OUT / "selection_stage_prelaunch.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    if not ADDENDUM.is_file() or json.loads(ADDENDUM.read_text()).get("decision") != "pass_non_circular_base_and_bit_exact_decoder":
        raise ValueError("selection stage requires passing non-circular identity-base addendum")
    v1 = json.loads(V1_AUDIT.read_text(encoding="utf-8"))
    label_levels = {
        name: record.get("support_label_levels")
        for name, record in v1.get("canonical_target", {}).get("sessions", {}).items()
    }
    if {tuple(value) for value in label_levels.values()} != {(1, 2, 3, 4)} or len(label_levels) != 4:
        raise ValueError("M1 selection prelaunch requires all four obj_id levels in every audited M10 support")
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen checkpoint hash mismatch: {path}")
    text = PROTOCOL.read_text(encoding="utf-8")
    required = ["inner LOSO", "evaluated exactly once as a gate", "F_full(C)", "B_r=V[:,0:r]", "[210,end)"]
    if any(item not in text for item in required):
        raise ValueError("protocol does not contain the frozen selection-stage contract")
    receipt = {
        "schema_version": "m1_decoder_latent_selection_stage_prelaunch_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "implementation_and_hash_preflight_passed_selection_execution_requires_root_review",
        "frozen": {"outer_left_out": "ses-20120926", "support": [0, 10], "inner_and_outer_selection": [10, 210],
                   "sealed_report_not_opened": [210, None], "rank_grid": [1, 2, 3], "lambda_grid": [1e-4, 1e-2, 1.0, 100.0],
                   "arms": ["full", "rate_only", "label_shuffle", "rate_residualized_condition_only"]},
        "hashes": {key: {"path": str(value.relative_to(ROOT)), "sha256": sha256(value)} for key, value in {
            "protocol": PROTOCOL, "selection_module": MODULE, "selection_runner": RUNNER, "selection_tests": TEST,
            "base_addendum": ADDENDUM, "v1_support_label_audit": V1_AUDIT, "f0_checkpoint": F0, "teacher_checkpoint": TEACHER,
        }.items()},
        "support_label_audit": {"per_session_levels": label_levels,
                                "interpretation": "all four levels are required in this locked M1 run; feature masks remain fail-closed encoding for future incomplete-support settings, not an authorization to pool missing-level sessions."},
        "not_executed": ["M1_selection_data_loading", "inner_loso_scoring", "outer_selection_gate", "sealed_report_reading",
                         "adapter_training", "GPU_launch", "heldout_access", "EvalAI_access_or_submission"],
    }
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "selection_stage_prelaunch.sha256").write_text(f"{sha256(target)}  selection_stage_prelaunch.json\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(run())
