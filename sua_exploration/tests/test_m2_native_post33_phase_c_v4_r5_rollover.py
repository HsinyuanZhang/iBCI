"""CPU-only r5 capability and expiry-boundary regression checks."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from sua_exploration.mc_maze import m2_native_post33_authorization_v4 as authorization
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    file_metadata,
    sha256_file,
    validate_shard_manifest,
)
from sua_exploration.mc_maze.m2_native_post33_program_v4 import validate_phase_c_program_receipt


ROOT = Path(__file__).resolve().parents[2]
R3 = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r3"
R5 = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r5"
RESULT_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r3"
PROGRAM = R5 / "program/phase_c_program_r5.json"
PORTABLE = R5 / "manifest/portable_r5.json"
SUPPLEMENT = R3 / "cost/cost_supplement_r3.json"


def test_r5_single_anchor_exact_29h_authorizations() -> None:
    assert authorization.MAX_VALIDITY_SECONDS == 30 * 3600
    assert sha256_file(authorization.PUBLIC_KEY) == authorization.PUBLIC_KEY_SHA256
    validate_phase_c_program_receipt(PROGRAM)
    proof = json.loads((R5 / "r5_rollover_semantic_equality.json").read_text(encoding="utf-8"))
    assert proof["authorization_policy"]["max_validity_seconds"] == 30 * 3600
    assert proof["authorization_policy"]["validity_hours"] == 29
    assert set(proof["program"]["only_fixed_trust_anchor_constant_source_delta"]) == {
        "sua_exploration/mc_maze/m2_native_post33_authorization_v4.py"
    }
    assert proof["forbidden_activity"] == {
        "gpu_used": False, "score_data_accessed": False, "endpoint_opened": False,
        "cell_opened": False, "result_root_created": False,
    }
    expected = {
        "gpu0": ("stage_a_execution_gpu0_r5.json", "shard_stage_a_gpu0_r5.json", [0, 2, 4, 6], "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"),
        "gpu1": ("stage_a_execution_gpu1_r5.json", "shard_stage_a_gpu1_r5.json", [1, 3, 5], "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"),
        "opening": ("stage_a_opening_r5.json", "shard_stage_a_opening_r5.json", list(range(7)), "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"),
    }
    nonces: set[str] = set()
    for name, (auth_name, shard_name, folds, gpu) in expected.items():
        auth_path, shard_path = R5 / "auth" / auth_name, R5 / "manifest" / shard_name
        shard = validate_shard_manifest(shard_path, portable_manifest_path=PORTABLE, cell_root=RESULT_ROOT)
        assert shard["host_id"] == "hw3090" and shard["gpu_id"] == gpu
        assert shard["fold_allowlist"] == folds and shard["seed_allowlist"] == [42]
        body = json.loads(auth_path.read_text(encoding="utf-8"))["authorization"]
        issued, expires = datetime.fromisoformat(body["issued_at"]), datetime.fromisoformat(body["expires_at"])
        assert expires - issued == timedelta(hours=29)
        assert body["public_key"] == file_metadata(authorization.PUBLIC_KEY)
        valid = authorization.verify_signed_authorization(
            auth_path, auth_path.with_suffix(".sig"), phase_c_program_receipt_path=PROGRAM,
            portable_manifest_path=PORTABLE, shard_manifest_path=shard_path,
            cost_supplement_path=SUPPLEMENT, cell_root=RESULT_ROOT, now=expires - timedelta(seconds=1),
        )
        assert valid["gpu_id"] == gpu
        with_raised = False
        try:
            authorization.verify_signed_authorization(
                auth_path, auth_path.with_suffix(".sig"), phase_c_program_receipt_path=PROGRAM,
                portable_manifest_path=PORTABLE, shard_manifest_path=shard_path,
                cost_supplement_path=SUPPLEMENT, cell_root=RESULT_ROOT, now=expires + timedelta(seconds=1),
            )
        except PermissionError:
            with_raised = True
        assert with_raised, name
        nonces.add(str(body["single_use_nonce"]))
    assert len(nonces) == 3
    assert not RESULT_ROOT.exists()
