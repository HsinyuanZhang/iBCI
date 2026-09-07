"""No-data held-graph and actual-c51-receipt tests for AOF-S V2."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v2 import binding, driver, plan
from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v2.profile import V2_RECOVERY_PROFILE
from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1.export_aofs_static_payload import (
    V1_TOP_LEVEL_RECEIPT_CODEC,
    V2_SIDE_EVIDENCE_RECEIPT_CODEC,
)


def _container_minival() -> dict[str, object]:
    return {
        "image_id": "sha256:" + "a" * 64, "image_tag": "synthetic-v2",
        "base_image_id": "sha256:b179efcba8e36202aec688b3104397919576e30d4344c309febcf692d4d7aaf8",
        "network_disabled": True, "pull": False, "container_removed": True,
        "command": ["--evaluation", "local", "--split", "m2", "--phase", "minival", "--batch-size", "7"],
        "stdout_sha256": "b" * 64, "prediction_path": "/output/prediction.pkl",
        "prediction_sha256": "c" * 64, "target_path": "/output/ground_truth.pkl", "target_sha256": "d" * 64,
    }


_V2_CANONICAL_ARTIFACT_EXISTS = os.path.lexists(ROOT / plan.ARTIFACT_ROOT_RELATIVE)


def test_exact_v1_pre_payload_failure_graph_is_held_fd_bound():
    witness = binding.validate_v1_failure_graph(ROOT)
    assert witness["bodies"] == plan.V1_FAILURE_BODIES
    assert witness["closure_sha256"] == plan.V1_FAILURE_CLOSURE_SHA256
    assert witness["failure_stage"] == "post_predecessor_pre_payload_artifact_docker"


def test_actual_c51_receipt_has_only_side_evidence_selection_indices():
    receipt = json.loads((ROOT / "sua_exploration/evalai_t4_m2_activity_budget/artifacts/t4_m2_seed42_ridge_m4_activity30_identity.receipt.json").read_text())
    assert len(receipt["session_records"]) == 13
    for session, row in receipt["session_records"].items():
        assert "selected_indices" not in row and "selected_indices_sha256" not in row, session
        assert V2_SIDE_EVIDENCE_RECEIPT_CODEC.selected(row) == row["side_evidence"]["selected_indices"], session
        assert V2_SIDE_EVIDENCE_RECEIPT_CODEC.selected_sha256(row) == row["side_evidence"]["selected_indices_sha256"], session
        with pytest.raises(Exception, match="selected-index path missing"):
            V1_TOP_LEVEL_RECEIPT_CODEC.selected(row)


def test_v2_profile_is_frozen_and_live_mint_is_fail_closed():
    assert V2_RECOVERY_PROFILE.artifact_root_relative == plan.ARTIFACT_ROOT_RELATIVE
    with pytest.raises(Exception):
        V2_RECOVERY_PROFILE.name = "forged"  # type: ignore[misc]
    assert not hasattr(driver, "issue_live_capability")


def test_v2_closure_is_explicit_and_current():
    mapping = driver.closure_map(ROOT)
    assert set(mapping) == set(plan.STATIC_CLOSURE_RELATIVES)
    assert len(driver.closure_sha256(mapping)) == 64


@pytest.mark.skipif(_V2_CANONICAL_ARTIFACT_EXISTS, reason="post-success/failure canonical V2 artifact makes fresh-root synthetic issuer inapplicable")
def test_v2_shared_production_lifecycle_success_uses_v2_schemas(tmp_path):
    root = tmp_path / "result"
    result = driver._execute_synthetic_for_test(
        repo_root=ROOT, result_root=root,
        build=lambda path: {"payload_sha256": "a" * 64, "session_records": {"s": {}}},
        validate=lambda path: {"host_minival": {"network": False}},
        docker=lambda: {"container_minival": _container_minival(), "network": "none", "pull": "false"},
    )
    assert result["status"] == "LOCAL_BUILD_V2_VALIDATED_NOT_SUBMITTED"
    terminal = json.loads((root / "terminal.json").read_text())
    assert terminal["schema"] == "m2_aof_scalar_static_package_v2_terminal"
    assert not (root / "failure.json").exists()
    assert set(path.name for path in root.iterdir()) == {
        name for body in ("attempt.json", "predecessor_authority.json", "input_authority.json", "build.json", "validation.json", "terminal.json")
        for name in (body, body + ".sha256")
    }


@pytest.mark.skipif(_V2_CANONICAL_ARTIFACT_EXISTS, reason="post-success/failure canonical V2 artifact makes fresh-root synthetic issuer inapplicable")
def test_v2_shared_production_lifecycle_failure_keeps_prefix(tmp_path):
    root = tmp_path / "result_failure"
    with pytest.raises(Exception, match="local build failed"):
        driver._execute_synthetic_for_test(
            repo_root=ROOT, result_root=root,
            build=lambda path: (_ for _ in ()).throw(ValueError("synthetic payload failure")),
            validate=lambda path: {}, docker=lambda: {"container_minival": _container_minival()},
        )
    failure = json.loads((root / "failure.json").read_text())
    assert failure["schema"] == "m2_aof_scalar_static_package_v2_failure"
    assert failure["published_prefix"] == ["attempt.json", "predecessor_authority.json"]
    assert not (root / "terminal.json").exists()


@pytest.mark.skipif(_V2_CANONICAL_ARTIFACT_EXISTS, reason="post-success/failure canonical V2 artifact makes fresh-root synthetic issuer inapplicable")
def test_v2_capability_rejects_parent_drift_and_reuse(tmp_path):
    root = tmp_path / "result_drift"
    mapping = driver.closure_map(ROOT)
    capability = driver._issue_capability(repo_root=ROOT, result_root=root, reviewed=mapping, digest=driver.closure_sha256(mapping))
    driver._consume(capability, ROOT)
    with pytest.raises(Exception, match="already consumed"):
        driver._consume(capability, ROOT)


@pytest.mark.skipif(_V2_CANONICAL_ARTIFACT_EXISTS, reason="post-success/failure canonical V2 artifact makes fresh-root synthetic issuer inapplicable")
def test_v2_capability_rejects_parent_replacement_before_attempt(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    root = parent / "result"
    mapping = driver.closure_map(ROOT)
    capability = driver._issue_capability(repo_root=ROOT, result_root=root, reviewed=mapping, digest=driver.closure_sha256(mapping))
    parent.rename(tmp_path / "replaced")
    parent.mkdir()
    with pytest.raises(Exception, match="result root/parent drift"):
        driver._consume(capability, ROOT)


def test_payload_sealing_requires_immutable_body_and_one_sidecar(tmp_path):
    from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1 import driver as lifecycle
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(b"sealed synthetic payload")
    payload.chmod(0o444)
    descriptor = lifecycle._seal_artifact(payload)
    assert descriptor["sha256"] == lifecycle._sha256(payload)
    with pytest.raises(Exception, match="sidecar already exists"):
        lifecycle._seal_artifact(payload)


@pytest.mark.skipif(_V2_CANONICAL_ARTIFACT_EXISTS, reason="post-success/failure canonical V2 artifact makes fresh-root synthetic issuer inapplicable")
def test_v2_profiled_lifecycle_rejects_missing_container_validation(tmp_path):
    root = tmp_path / "no_container"
    with pytest.raises(Exception, match="local build failed"):
        driver._execute_synthetic_for_test(
            repo_root=ROOT, result_root=root, build=lambda path: {"payload_sha256": "a" * 64},
            validate=lambda path: {}, docker=lambda: {"network": "none"},
        )
    assert not (root / "terminal.json").exists() and (root / "failure.json").is_file()
