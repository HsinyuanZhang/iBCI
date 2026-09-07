#!/usr/bin/env python3
"""Write the Phase-B v3 score-free receipt exactly once."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "sua_exploration/results/m2_native_post33_phase_b_v3_scorefree_20260804"
OUT = OUT_DIR / "phase_b_scorefree_receipt.json"
SHA_OUT = OUT_DIR / "phase_b_scorefree_receipt.sha256"

ACTIVE_C1 = ROOT / "sua_exploration/results/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist/receipt.json"
PHASE_A_V2 = ROOT / "sua_exploration/results/m2_native_t4_spint_post33_confirm_v1_draft_prelaunch_v2_hash_hardened_20260804/draft_scorefree_receipt.json"
INDEPENDENT_REVIEW = ROOT / "sua_exploration/docs/M2_NATIVE_POST33_V2_INDEPENDENT_REVIEW_20260804.md"

SOURCE_FILES = (
    "SPINT-main/configs/callbacks/post33_source_only_v3.yaml",
    "SPINT-main/configs/data/falcon_m2_post33_confirm_v3.yaml",
    "SPINT-main/configs/experiment/m2_native_post33_confirm_v3_spint.yaml",
    "SPINT-main/configs/hydra/post33_cell_v3.yaml",
    "SPINT-main/configs/model/falcon_m2_post33_confirm_v3.yaml",
    "SPINT-main/src/callbacks/post33_source_selector_v3.py",
    "SPINT-main/src/models/falcon_post33_confirm_v3_module.py",
    "streaming_calibration_exp/configs/data/falcon_m2_post33_confirm_v3.yaml",
    "streaming_calibration_exp/configs/experiment/m2_native_post33_confirm_v3_t4.yaml",
    "streaming_calibration_exp/configs/hydra/post33_cell_v3.yaml",
    "streaming_calibration_exp/configs/model/streaming_b3s_t4_post33_paired_v3.yaml",
    "streaming_calibration_exp/src/data/falcon_post33_confirm_v3_datamodule.py",
    "streaming_calibration_exp/src/models/streaming_post33_paired_t4_v3_module.py",
    "streaming_calibration_exp/src/utils/post33_paired_teacher_v3.py",
    "sua_exploration/docs/M2_NATIVE_POST33_PHASE_B_V3_SCORE_FREE_STATUS_20260804.md",
    "sua_exploration/docs/M2_NATIVE_POST33_V2_INDEPENDENT_REVIEW_20260804.md",
    "sua_exploration/mc_maze/m2_native_post33_phase_b_v3.py",
    "sua_exploration/scripts/finalize_m2_native_post33_spint_v3_receipt.py",
    "sua_exploration/scripts/preflight_m2_native_post33_phase_b_v3_cell.py",
    "sua_exploration/scripts/verify_m2_native_post33_phase_b_v3_scorefree_receipt.py",
    "sua_exploration/scripts/write_m2_native_post33_phase_b_v3_scorefree_receipt.py",
    "sua_exploration/tests/test_m2_native_post33_phase_b_v3.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        cursor = 0
        while cursor < len(data):
            cursor += os.write(fd, data[cursor:])
        os.fsync(fd)
    finally:
        os.close(fd)


def source_map() -> dict[str, dict[str, Any]]:
    result = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing/non-canonical Phase-B source: {path}")
        result[relative] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    return result


def verify_upstream_lock(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    entries = receipt.get("source_map")
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"upstream receipt has no source_map: {path}")
    drift = []
    for relative, metadata in entries.items():
        source = ROOT / relative
        observed = sha256(source)
        if observed != metadata.get("sha256"):
            drift.append(relative)
    if drift:
        raise RuntimeError(f"upstream source-map drift: {drift}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "source_map_entries": len(entries),
        "source_map_entries_reverified_now": len(entries),
        "drift_count": 0,
    }


def build() -> dict[str, Any]:
    review_sha = sha256(INDEPENDENT_REVIEW)
    if review_sha != "5743d2f72a58d167f8b6473b09736f4514b81636c03fd41e683e0ef5376e4cfd":
        raise RuntimeError("independent review SHA drift")
    return {
        "schema": "m2_native_post33_phase_b_v3_scorefree_receipt",
        "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
        "phase_id": "PHASE_B_V3",
        "date": "2026-08-04",
        "status": "NO_GO_C4_FINAL_RUNNER_FINALIZERS_COST_RUNTIME_EVIDENCE_REQUIRED",
        "gpu_authorization_created": False,
        "accuracy_claim_authorized": False,
        "training_authorized": False,
        "independent_review": {
            "path": str(INDEPENDENT_REVIEW.resolve()),
            "sha256": review_sha,
        },
        "implemented": {
            "c1_source_metric": {
                "versioned_spint_module": True,
                "exact_source_sessions": 6,
                "per_source_total_gt_2": True,
                "per_source_finite": True,
                "outer_validation_total": 0,
                "equal_session_mean": True,
                "outer_only_test_without_empty_source_placeholders": True,
                "selector_epochs": [0, 34],
                "selector_policy": "max_finite_equal_session_mean_then_earlier_epoch",
            },
            "c2_paired_teacher": {
                "spint_completion_receipt_schema": "m2_post33_spint_completion_receipt_v3",
                "matching_fold_seed_required": True,
                "manual_teacher_path_forbidden": True,
                "legacy_default_fallback_present": False,
                "decoder_tensor_closure": "31/31 name_shape_dtype_num_bytes_sha256",
                "decoder_stages": ["pretrain", "posttrain", "reload", "prequery"],
                "requires_grad_required": 0,
                "optimizer_intersection_required": 0,
                "updated_tensors_required": 0,
            },
            "c3_ownership_paths": {
                "path_identity": ["protocol", "phase", "arm", "fold", "seed"],
                "ownership_write": "os.open O_CREAT|O_EXCL",
                "status_write_once": ["started", "completed", "failed"],
                "same_cell_concurrency_winners": 1,
                "different_cell_collision_count": 0,
            },
            "h1_hparams": {
                "public_hparams_overwritten": False,
                "dedicated_setup_without_base_setup_delegation": True,
                "public_internal_contracts_separate_immutable": True,
                "generic_split_consumer_fails_closed": True,
            },
        },
        "verification": {
            "phase_b_focused_tests": {"passed": 26, "failed": 0},
            "phase_a_plus_phase_b_tests": {"passed": 52, "failed": 0},
            "py_compile": "PASS",
            "actual_hydra_compose": {
                "spint_source_metric_module": "PASS",
                "t4_mandatory_paired_receipt": "PASS",
                "deterministic_cell_paths": "PASS",
                "legacy_epoch_034_in_v3_configs": False,
            },
            "runtime_public_hparams_constructor": "PASS",
        },
        "execution_scope": {
            "gpu_used": False,
            "training_started": False,
            "optimizer_steps": 0,
            "new_endpoint_r2_values_read": 0,
            "scorer_modules_imported": 0,
            "formal_sua_paths_resolved": 0,
            "formal_sua_files_opened": 0,
            "evalai_calls": 0,
        },
        "upstream_locks": {
            "active_c1_v3r2": verify_upstream_lock(ACTIVE_C1),
            "phase_a_v2": verify_upstream_lock(PHASE_A_V2),
        },
        "remaining_blockers": [
            "C4 final anti-tamper verifier with exact canonical result/checkpoint/status sets",
            "final matrix launcher with crash/interrupt failed-status finalization",
            "T4 result and matrix-level finalizers",
            "paired-arm parameter/MAC/calibration-state/runtime-state cost receipt",
            "production pretrain/posttrain/reload/prequery 31-of-31 decoder evidence",
            "timestamped secondary streaming artifacts must be suppressed or canonicalized",
        ],
        "source_map": source_map(),
    }


def main() -> None:
    payload = build()
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    write_exclusive(OUT, encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    write_exclusive(SHA_OUT, f"{digest}  {OUT.name}\n".encode("utf-8"))
    print(OUT)
    print(digest)


if __name__ == "__main__":
    main()

