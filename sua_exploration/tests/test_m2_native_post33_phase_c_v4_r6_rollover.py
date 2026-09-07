"""CPU-only audit of the r6b fresh-root Phase-C recovery capability bundle."""
from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path

import pytest

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
R6 = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6b"
R3_RESULT_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r3"
R6_RESULT_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r6b"
PROGRAM = R6 / "program/phase_c_program_r6b.json"
PORTABLE = R6 / "manifest/portable_r6b.json"
SUPPLEMENT = R3 / "cost/cost_supplement_r3.json"
PROOF = R6 / "r6b_fresh_root_rollover_semantic_equality.json"
LAUNCH = R6 / "launch/stage_a_commands_r6b.json"


def test_r6b_fresh_root_signed_stage_a_bundle() -> None:
    """Verify the recovery only changes the approved code/root/auth boundaries."""
    assert R3.is_dir() and R5.is_dir() and R6.is_dir()
    assert R3_RESULT_ROOT.is_dir()
    assert not R6_RESULT_ROOT.exists()
    assert authorization.MAX_VALIDITY_SECONDS == 30 * 3600
    assert sha256_file(authorization.PUBLIC_KEY) == authorization.PUBLIC_KEY_SHA256
    assert authorization.PUBLIC_KEY.name == "m2_native_post33_phase_c_v4_r6b_root_ed25519_public.pem"
    validate_phase_c_program_receipt(PROGRAM)
    proof = json.loads(PROOF.read_text(encoding="utf-8"))
    assert proof["r6b_fresh_result_root"] == str(R6_RESULT_ROOT.resolve())
    assert proof["r6b_fresh_result_root_exists_at_issuance"] is False
    assert proof["r3_failed_history"]["r3_result_root"] == str(R3_RESULT_ROOT.resolve())
    assert proof["r3_failed_history"]["r3_reuse_forbidden"] is True
    assert proof["r3_failed_history"]["score_or_endpoint_payload_opened"] is False
    aborted = proof["aborted_prewrite_r6_anchor"]
    assert aborted["status"] == "ABORTED_PREWRITE_BURNED_ANCHOR"
    assert aborted["public_key"]["canonical_path"].endswith(
        "m2_native_post33_phase_c_v4_r6_root_ed25519_public.pem"
    )
    assert aborted["r6_receipt_root_exists"] is False
    assert aborted["r6_cell_root_exists"] is False
    assert aborted["authorization_nonce_claimed"] is False
    assert aborted["gpu_used"] is False
    assert aborted["endpoint_opened"] is False
    assert aborted["score_data_accessed"] is False
    assert aborted["formal_data_accessed"] is False
    assert aborted["private_key_persisted"] is False
    assert aborted["r6_public_anchor_reused"] is False
    documents = proof["frozen_c1_document_and_postseal_addendum"]
    assert documents["restored_exactly"] is True
    assert documents["frozen_phase_a_document"]["size_bytes"] == documents["sealed_size_bytes"]
    assert documents["frozen_phase_a_document"]["sha256"] == documents["sealed_sha256"]
    assert documents["postseal_addendum"]["canonical_path"].endswith(
        "PHASE_C_ATTRIBUTION_POWER_ADDENDUM_20260805.md"
    )
    assert documents["addendum_is_outside_phase_c_runtime_source_closure"] is True
    legacy_z4 = proof["legacy_shared_z4_release_boundary"]
    assert legacy_z4["bound_phase_c_result_root"] == str(R3_RESULT_ROOT.resolve())
    assert legacy_z4["r6b_root_not_accepted"] is True
    assert legacy_z4["r6b_authorizations_do_not_authorize_shared_z4_release"] is True
    assert proof["forbidden_activity"] == {
        "gpu_used": False,
        "score_data_accessed": False,
        "formal_data_accessed": False,
        "endpoint_opened": False,
        "r3_or_r5_history_rewritten": False,
        "r6b_result_root_created": False,
    }
    assert set(proof["program"]["only_allowed_production_source_deltas"]) == {
        "SPINT-main/src/evaluate_post33_phase_c_v4.py",
        "streaming_calibration_exp/src/evaluate_post33_phase_c_v4.py",
        "sua_exploration/mc_maze/m2_native_post33_authorization_v4.py",
    }
    assert proof["program"]["source_closure_membership_equal"] is True
    assert proof["program"]["all_program_semantic_fields_equal"] is True
    assert proof["fixed_anchor_exact_source_diff"]["verifier_logic_changed"] is False
    assert proof["fixed_anchor_exact_source_diff"]["runtime_gate_changed"] is False
    assert proof["single_fixed_trust_anchor"]["public_key"] == file_metadata(authorization.PUBLIC_KEY)
    assert proof["regression_test"]["test_name"] == (
        "test_workers_restore_eval_mode_before_cached_deployment_benchmark"
    )
    for repair in proof["evaluator_eval_mode_repairs"].values():
        assert repair["trainer_test_position"] < repair["model_eval_position"] < repair["benchmark_position"]

    expected = {
        "gpu0": (
            "stage_a_execution_gpu0_r6b.json",
            "shard_stage_a_gpu0_r6b.json",
            [0, 2, 4, 6],
            "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9",
        ),
        "gpu1": (
            "stage_a_execution_gpu1_r6b.json",
            "shard_stage_a_gpu1_r6b.json",
            [1, 3, 5],
            "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86",
        ),
        "opening": (
            "stage_a_opening_r6b.json",
            "shard_stage_a_opening_r6b.json",
            list(range(7)),
            "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9",
        ),
    }
    nonces: set[str] = set()
    for name, (auth_name, shard_name, folds, gpu) in expected.items():
        auth_path, shard_path = R6 / "auth" / auth_name, R6 / "manifest" / shard_name
        shard = validate_shard_manifest(shard_path, portable_manifest_path=PORTABLE, cell_root=R6_RESULT_ROOT)
        assert shard["host_id"] == "hw3090" and shard["gpu_id"] == gpu
        assert shard["fold_allowlist"] == folds and shard["seed_allowlist"] == [42]
        body = json.loads(auth_path.read_text(encoding="utf-8"))["authorization"]
        issued, expires = datetime.fromisoformat(body["issued_at"]), datetime.fromisoformat(body["expires_at"])
        assert expires - issued == timedelta(hours=29)
        assert body["absolute_cell_root"] == str(R6_RESULT_ROOT.resolve())
        assert body["public_key"] == file_metadata(authorization.PUBLIC_KEY)
        valid = authorization.verify_signed_authorization(
            auth_path,
            auth_path.with_suffix(".sig"),
            phase_c_program_receipt_path=PROGRAM,
            portable_manifest_path=PORTABLE,
            shard_manifest_path=shard_path,
            cost_supplement_path=SUPPLEMENT,
            cell_root=R6_RESULT_ROOT,
            now=expires - timedelta(seconds=1),
        )
        assert valid["gpu_id"] == gpu
        with pytest.raises(PermissionError, match="expired/not-yet-valid/exceeds"):
            authorization.verify_signed_authorization(
                auth_path,
                auth_path.with_suffix(".sig"),
                phase_c_program_receipt_path=PROGRAM,
                portable_manifest_path=PORTABLE,
                shard_manifest_path=shard_path,
                cost_supplement_path=SUPPLEMENT,
                cell_root=R6_RESULT_ROOT,
                now=expires + timedelta(seconds=1),
            )
        nonces.add(str(body["single_use_nonce"]))
    assert len(nonces) == 3

    launch = json.loads(LAUNCH.read_text(encoding="utf-8"))
    assert launch["status"] == "PREPARED_NOT_EXECUTED"
    assert launch["cell_root_must_be_absent_until_command_execution"] == str(R6_RESULT_ROOT.resolve())
    assert launch["commands"]["gpu0"]["fold_allowlist"] == [0, 2, 4, 6]
    assert launch["commands"]["gpu1"]["fold_allowlist"] == [1, 3, 5]
    assert launch["commands"]["gpu0"]["cuda_visible_devices"] == expected["gpu0"][3]
    assert launch["commands"]["gpu1"]["cuda_visible_devices"] == expected["gpu1"][3]
    assert not R6_RESULT_ROOT.exists()
