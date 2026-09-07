"""No-target focused tests for the paired producer v2 successor."""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_subject_m_stagep_paired_real_producer as v1  # noqa: E402
import track_b_v2_subject_m_stagep_paired_real_producer_v2 as v2  # noqa: E402


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def test_authority_hash_domain_exactly_matches_source_adapter_and_not_raw_bytes() -> None:
    import numpy as np

    _, authority = v2.load_source_authority_bundle("sua")
    scaler = authority["source_behavior_normalizer"]
    for role in ("mean", "std"):
        value = np.ascontiguousarray(scaler[f"{role}_float32"], dtype=np.float32)
        assert v2._authority_array_sha256(value) == scaler[f"{role}_array_sha256"]
        assert v1._raw_array_sha(value) != scaler[f"{role}_array_sha256"]


def test_v1_failure_is_exact_immutable_and_v2_root_is_distinct() -> None:
    proof = v2.validate_immutable_v1_failure()
    assert proof["completion_body_sha256"] == v2.V1_FAILURE_COMPLETION_BODY_SHA256
    assert proof["v1_output_reuse_permitted"] is False
    assert v2.RESULT_ROOT != v2.V1_RESULT_ROOT
    assert "_v2" in v2.RESULT_ROOT.name


def test_topology_adds_attempt_lineage_and_stream_pairs_without_v1_paths() -> None:
    topology = v2._topology("sua")
    assert set(topology) >= {"target_access_attempt", "target_lineage", "child_stdout", "child_stderr"}
    assert all(str(v2.RESULT_ROOT) in str(path) for key, path in topology.items()
               if key not in {"scores", "caller_path_override_permitted", "successor_version"})
    assert str(v2.V1_RESULT_ROOT) not in str(topology)


def test_access_attempt_is_preopen_and_binds_source_preflight_authority() -> None:
    cap = v1.ViewExecutionCapability(
        view="sua", cell=v1.sealed_runtime.StagePCell.from_view("sua").as_dict(),
        admission_sha256=_sha("a"), official_preflight_body_sha256=_sha("o"),
        implementation_closure_sha256=_sha("i"), source_authority_set_sha256=_sha("s"),
        v9_preflight_sha256=v2.V9_PREFLIGHT_SHA256)
    payload = v2.build_target_access_attempt_payload(
        capability=cap, source_receipt_body_sha256=_sha("source-body"),
        source_payload_sha256=_sha("source-payload"), launcher_preflight_body_sha256=_sha("pf"),
        target_authority={"canonical": True}, caller_pid=123)
    assert payload["target_opened_before_publication"] is False
    assert payload["next_operation_may_open_target"] is True
    poisoned = copy.deepcopy(payload); poisoned["source_receipt_body_sha256"] = "bad"
    assert poisoned != payload


def test_public_plan_is_no_target_and_explicitly_supersedes_v1() -> None:
    plan = v2.build_no_target_review_plan()
    assert plan["status"] == v2.STATUS_REVIEW
    assert plan["target_opened"] is plan["formal_data_opened"] is plan["gpu_used"] is False
    assert plan["v1_failure_provenance"]["v1_output_reuse_permitted"] is False
    assert plan["normalizer_hash_domain_repair"]["normalizer_values_changed"] is False


def test_real_sua_target_only_held_fd_v9_regression() -> None:
    """Run explicitly after review; source rebuild + authorized dev target, no GPU/fit."""
    if os.environ.get("TRACK_B_RUN_REAL_SUA_TARGET_ONLY") != "1":
        pytest.skip("explicit real target-only integration gate not enabled")
    admissions = {view: v1.sealed_runtime.build_stagep_live_admission(view=view)
                  for view in v2.PAIR_ORDER}
    plan = v2.build_no_target_review_plan()
    cap = v2.bind_execution_capabilities(admissions=admissions, reviewed_plan=plan)["sua"]
    source = v2.materialize_and_verify_strict27_source(cap)
    original_topology = v2._topology
    with tempfile.TemporaryDirectory(prefix="trackb_v2_real_target_only_") as directory:
        def temporary_topology(view: str) -> dict:
            topology = dict(original_topology(view))
            topology["private_snapshot"] = str(Path(directory) / f"{view}_held_target.nwb")
            return topology
        v2._topology = temporary_topology
        try:
            target = v2.materialize_target_from_private_snapshot(cap, source)
            lineage = v2.build_prefit_target_lineage_payload(
                target=target, source_receipt_body_sha256=_sha("source-body"),
                target_access_attempt_body_sha256=_sha("attempt-body"))
        finally:
            v2._topology = original_topology
    assert lineage["valid_starts_int64_sha256"] == v2.V9_VALID_STARTS_SHA256
    assert lineage["ordered_target_behavior_float32_sha256"] == v2.V9_TARGET_SHA256
    assert len(lineage["ordered_prediction_endpoint_int64_sha256"]) == 64
    assert len(lineage["ordered_offset10_RF_int64_sha256"]) == 64
    assert lineage["every_RF_wholly_inside_held_suffix"] is True
    assert lineage["every_RF_support_disjoint"] is True
    assert lineage["query_row_count"] == v2.V9_QUERY_COUNT
    assert lineage["cebra_fit_started"] is lineage["gpu_fit_started"] is False
    assert lineage["formal_data_opened"] is False
