"""Synthetic-only tests for the pre-execution V9 three-arm core.

No test in this file opens external sub-M data, an NWB file, a checkpoint, a
prediction produced by a real model, or a formal R² result.  Arrays below are
tiny synthetic fixtures written below pytest's temporary directory.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from sua_exploration.mc_maze import subm_co_three_arm_score_only_v9 as v9


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "sua_exploration/mc_maze/subm_co_three_arm_score_only_v9.py"


def _contract(*, bootstrap_replicates: int = 11) -> v9.V9MatrixContract:
    return v9.V9MatrixContract(
        tuple(
            v9.V9Session(
                asset_id=f"asset_{index:02d}",
                session_id=f"session_{index:02d}",
                query_window_count=3,
            )
            for index in range(v9.EXPECTED_SESSION_COUNT)
        ),
        bootstrap_replicates=bootstrap_replicates,
    )


def _token(contract: v9.V9MatrixContract, root: Path, *, purpose: str = "initial") -> v9.V9SyntheticRunToken:
    return v9.prepare_synthetic_run_token(contract, root, purpose=purpose, token_id=f"synthetic-{purpose}")


def _evidence(key: v9.V9CellKey, targets: Any | None = None) -> dict[str, Any]:
    if targets is None:
        _prediction, targets = _arrays(key)
    return {
        "checkpoint_sha256": ("a" if key.arm == "shared_t4" else "b" if key.arm == "shared_zero4" else "c") * 64,
        "input_trace_sha256": f"{key.asset_id}_{key.view}_{key.arm}_{key.seed}",
        "query_behavior_trace_sha256": v9.query_behavior_trace_sha256(targets),
        "phase": "synthetic_artifact_only",
    }


def _arrays(key: v9.V9CellKey):
    import numpy as np

    # Behavior is a session-level held-out target.  It must not depend on the
    # training seed, view, or descriptor arm used to make its prediction.
    target = np.asarray(
        [[0.0, 0.5], [1.0, 1.5], [2.0, 4.0]],
        dtype=np.float32,
    )
    if key.arm == "shared_t4":
        prediction = target.copy()
    elif key.arm == "shared_zero4":
        prediction = np.zeros_like(target)
    else:
        prediction = np.ascontiguousarray(target * np.float32(0.45))
    return np.ascontiguousarray(prediction), np.ascontiguousarray(target)


def _write_all(writer: v9.V9ArtifactOnlyWriter, contract: v9.V9MatrixContract, *, skip: set[v9.V9CellKey] | None = None) -> None:
    skipped = skip or set()
    for key in contract.expected_keys:
        if key in skipped:
            continue
        prediction, target = _arrays(key)
        writer.write_cell_artifact(key, prediction, target, evidence=_evidence(key, target))


def _rewrite_immutable(path: Path, raw: bytes) -> None:
    """Test-only corruption helper for a pytest-owned synthetic output root."""

    path.chmod(0o644)
    try:
        path.write_bytes(raw)
    finally:
        path.chmod(0o444)


def _rewrite_committed_artifact_and_evidence(
    root: Path,
    key: v9.V9CellKey,
    *,
    targets: Any,
    trace: str,
) -> None:
    """Simulate a sophisticated test-only mutation that updates local pins."""

    import numpy as np

    artifact_path = root / key.artifact_relative_path
    with np.load(artifact_path, allow_pickle=False) as loaded:
        predictions = np.ascontiguousarray(loaded["predictions"])
    replacement_targets = np.ascontiguousarray(targets, dtype=np.float32)
    artifact_path.chmod(0o644)
    try:
        with artifact_path.open("wb") as handle:
            np.savez_compressed(handle, predictions=predictions, targets=replacement_targets)
    finally:
        artifact_path.chmod(0o444)
    commit_path = root / key.commit_relative_path
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["prediction_target_artifact"]["sha256"] = v9.sha256_file(artifact_path)
    commit["prediction_target_artifact"]["bytes"] = artifact_path.stat().st_size
    commit["evidence"]["query_behavior_trace_sha256"] = trace
    _rewrite_immutable(commit_path, v9.canonical_bytes(commit))


def test_contract_freezes_exact_270_cell_topology_and_30_50_post50_boundary() -> None:
    contract = _contract()
    assert len(contract.expected_keys) == 270
    assert len(set(contract.expected_keys)) == 270
    payload = contract.as_dict()
    assert payload["views"] == ["sua", "pseudo_mua"]
    assert payload["arms"] == ["shared_t4", "shared_zero4", "shared_ts4"]
    assert payload["seeds"] == [42, 43, 44]
    assert payload["chronology"] == {
        "activity_identity_trials": 30,
        "t4_fit_pool_trials": 50,
        "query_rule": "strictly_after_rewarded_trial_50",
        "chronological": True,
    }
    with pytest.raises(v9.V9Error, match="exactly 15"):
        v9.V9MatrixContract(contract.cohort[:-1])
    duplicate = list(contract.cohort)
    duplicate[-1] = duplicate[0]
    with pytest.raises(v9.V9Error, match="duplicate"):
        v9.V9MatrixContract(tuple(duplicate))


def test_synthetic_helper_targets_are_seed_view_and_arm_independent() -> None:
    contract = _contract()
    session = contract.cohort[0]
    reference: bytes | None = None
    for view in v9.VIEWS:
        for arm in v9.ARMS:
            for seed in v9.SEEDS:
                key = v9.V9CellKey(session.asset_id, session.session_id, view, arm, seed)
                _prediction, targets = _arrays(key)
                target_bytes = targets.tobytes(order="C")
                if reference is None:
                    reference = target_bytes
                assert target_bytes == reference
                assert _evidence(key, targets)["query_behavior_trace_sha256"] == v9.query_behavior_trace_sha256(targets)


def test_core_has_no_top_level_numpy_torch_or_external_data_owner_import() -> None:
    tree = ast.parse(CORE.read_text(encoding="utf-8"), filename=str(CORE))
    top_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_imports.append(node.module or "")
    forbidden = ("numpy", "torch", "pynwb", "multisession_datamodule", "unit_side_features", "datamodule")
    assert not [name for name in top_imports if any(fragment in name for fragment in forbidden)]
    source = CORE.read_text(encoding="utf-8")
    assert "load_dandi688_session" not in source
    assert "load_frozen_model" not in source
    assert "formal_external_execution_unavailable" in source


def test_authorization_first_boundary_is_provable_in_a_fresh_process() -> None:
    base = (
        "from sua_exploration.mc_maze.subm_co_three_arm_score_only_v9 "
        "import assert_authorization_first_import_boundary; "
        "assert_authorization_first_import_boundary(); print('PASS')"
    )
    clean = subprocess.run(
        [sys.executable, "-c", base], cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert clean.returncode == 0, clean.stderr
    assert clean.stdout.strip() == "PASS"
    contaminated = subprocess.run(
        [
            sys.executable,
            "-c",
            "import numpy; " + base.replace("assert_authorization_first_import_boundary(); print('PASS')", "assert_authorization_first_import_boundary()"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert contaminated.returncode != 0
    assert "runtime imported before ordering token" in contaminated.stderr


def test_writer_requires_synthetic_ordering_token_and_never_claims_formal_scope(tmp_path: Path) -> None:
    contract = _contract()
    root = tmp_path / "run"
    token = _token(contract, root)
    assert token.formal_external_scoring_permitted is False
    writer = v9.V9ArtifactOnlyWriter(root, contract, token)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["scope"] == v9.SYNTHETIC_SCOPE
    assert manifest["phase_a_metrics"] == "FORBIDDEN_UNTIL_EXACT_270_ARTIFACT_COMMITS"
    assert manifest["external_subm_opened"] is False
    assert manifest["checkpoint_opened"] is False
    assert manifest["model_forward_performed"] is False
    wrong = v9.prepare_synthetic_run_token(contract, tmp_path / "different")
    with pytest.raises(v9.V9AuthorizationOrderError, match="output-root"):
        v9.V9ArtifactOnlyWriter(root, contract, wrong)
    assert writer.manifest_sha256 == v9.sha256_file(root / "run_manifest.json")


def test_phase_a_writes_no_partial_r2_and_finalizer_refuses_incomplete_matrix(tmp_path: Path) -> None:
    contract = _contract()
    root = tmp_path / "partial"
    writer = v9.V9ArtifactOnlyWriter(root, contract, _token(contract, root))
    key = contract.expected_keys[0]
    prediction, target = _arrays(key)
    writer.write_cell_artifact(key, prediction, target, evidence=_evidence(key))
    commit_path = root / key.commit_relative_path
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    assert commit["status"] == "ARTIFACT_COMMITTED_NO_METRIC"
    assert "r2" not in commit
    assert "score" not in commit
    assert commit["metric_computation"] == "FORBIDDEN_IN_PHASE_A"
    assert commit["evidence"]["query_behavior_trace_sha256"] == v9.query_behavior_trace_sha256(target)
    state = writer.state()
    assert len(state.completed) == 1
    assert len(state.missing) == 269
    with pytest.raises(v9.V9IncompleteMatrixError, match="exact 270"):
        v9.finalize_synthetic_artifacts(root, contract)
    assert not (root / "aggregate/endpoint_aggregate.json").exists()
    with pytest.raises(v9.V9ContinuationError, match="already committed"):
        writer.write_cell_artifact(key, prediction, target, evidence=_evidence(key))
    with pytest.raises(v9.V9ArtifactError, match="metric/result"):
        writer.write_cell_artifact(
            contract.expected_keys[1],
            *_arrays(contract.expected_keys[1]),
            evidence={"r2": 0.99},
        )


def test_exact_270_artifact_finalizer_reopens_all_cells_and_separates_two_contrasts(tmp_path: Path) -> None:
    contract = _contract(bootstrap_replicates=13)
    root = tmp_path / "complete"
    writer = v9.V9ArtifactOnlyWriter(root, contract, _token(contract, root))
    _write_all(writer, contract)
    state = writer.state()
    assert state.complete
    assert len(state.completed) == 270
    result = v9.finalize_synthetic_artifacts(root, contract)
    aggregate = result["aggregate"]
    assert aggregate["status"] == "SYNTHETIC_FULL_270_FINALIZED_NOT_FORMAL"
    assert aggregate["verified_cell_count"] == 270
    assert aggregate["aggregate_recomputed_only_after_full_artifact_matrix"] is True
    assert len(aggregate["cells"]) == 270
    assert set(aggregate["comparisons"]) == {
        "shared_t4_minus_shared_zero4",
        "shared_t4_minus_shared_ts4",
    }
    for comparison in aggregate["comparisons"].values():
        assert set(comparison["views"]) == {"sua", "pseudo_mua"}
        assert comparison["cross_view_rescue_used"] is False
        for view in comparison["views"].values():
            assert view["positive_session_required"] == 12
            assert set(view["seed_means_r2"]) == {"42", "43", "44"}
    aggregate_path = root / "aggregate/endpoint_aggregate.json"
    assert result["aggregate_sha256"] == v9.sha256_file(aggregate_path)
    with pytest.raises(v9.V9ArtifactError, match="unknown artifact topology"):
        v9.scan_artifact_state(root, contract)
    with pytest.raises(v9.V9ArtifactError, match="overwrite"):
        v9.finalize_synthetic_artifacts(root, contract)


def test_missing_only_recovery_binds_exact_committed_map_and_cannot_recompute(tmp_path: Path) -> None:
    contract = _contract(bootstrap_replicates=7)
    root = tmp_path / "recover"
    initial = v9.V9ArtifactOnlyWriter(root, contract, _token(contract, root))
    first = contract.expected_keys[0]
    second = contract.expected_keys[1]
    for key in (first, second):
        prediction, target = _arrays(key)
        initial.write_cell_artifact(key, prediction, target, evidence=_evidence(key))
    binding = initial.state().recovery_binding()
    recovery_token = _token(contract, root, purpose="recovery")
    recovered = v9.V9ArtifactOnlyWriter.open_missing_only_recovery(root, contract, recovery_token, binding)
    with pytest.raises(v9.V9ContinuationError, match="already committed"):
        recovered.write_cell_artifact(first, *_arrays(first), evidence=_evidence(first))
    _write_all(recovered, contract, skip={first, second})
    assert recovered.state().complete
    assert v9.finalize_synthetic_artifacts(root, contract)["aggregate"]["verified_cell_count"] == 270


def test_recovery_rejects_map_drift_or_unknown_orphan_artifact(tmp_path: Path) -> None:
    contract = _contract()
    root = tmp_path / "bad-recovery"
    writer = v9.V9ArtifactOnlyWriter(root, contract, _token(contract, root))
    first, second = contract.expected_keys[:2]
    writer.write_cell_artifact(first, *_arrays(first), evidence=_evidence(first))
    stale = writer.state().recovery_binding()
    writer.write_cell_artifact(second, *_arrays(second), evidence=_evidence(second))
    with pytest.raises(v9.V9ContinuationError, match="binding differs"):
        v9.V9ArtifactOnlyWriter.open_missing_only_recovery(
            root, contract, _token(contract, root, purpose="recovery"), stale
        )
    extra = root / "unknown.txt"
    extra.write_text("not a V9 artifact", encoding="utf-8")
    extra.chmod(0o444)
    with pytest.raises(v9.V9ArtifactError, match="unknown artifact topology"):
        v9.scan_artifact_state(root, contract)


def test_reopener_rejects_orphan_commit_or_orphan_npz(tmp_path: Path) -> None:
    contract = _contract()
    root = tmp_path / "orphan"
    writer = v9.V9ArtifactOnlyWriter(root, contract, _token(contract, root))
    key = contract.expected_keys[0]
    writer.write_cell_artifact(key, *_arrays(key), evidence=_evidence(key))
    artifact = root / key.artifact_relative_path
    artifact.unlink()
    with pytest.raises(v9.V9ArtifactError, match="orphan artifact or orphan commit"):
        v9.scan_artifact_state(root, contract)


def test_scan_and_finalizer_reject_target_or_trace_tampering_within_paired_asset(tmp_path: Path) -> None:
    import numpy as np

    contract = _contract()
    session = contract.cohort[0]
    first = v9.V9CellKey(session.asset_id, session.session_id, "sua", "shared_t4", 42)
    peer = v9.V9CellKey(session.asset_id, session.session_id, "pseudo_mua", "shared_ts4", 44)

    target_root = tmp_path / "target-tamper"
    target_writer = v9.V9ArtifactOnlyWriter(target_root, contract, _token(contract, target_root))
    for key in (first, peer):
        prediction, targets = _arrays(key)
        target_writer.write_cell_artifact(key, prediction, targets, evidence=_evidence(key, targets))
    changed_targets = _arrays(peer)[1].copy()
    changed_targets[0, 0] += np.float32(0.25)
    _rewrite_committed_artifact_and_evidence(
        target_root,
        peer,
        targets=changed_targets,
        trace=v9.query_behavior_trace_sha256(changed_targets),
    )
    with pytest.raises(v9.V9ArtifactError, match="within one asset/session"):
        v9.scan_artifact_state(target_root, contract)
    with pytest.raises(v9.V9ArtifactError, match="within one asset/session"):
        v9.finalize_synthetic_artifacts(target_root, contract)

    trace_root = tmp_path / "trace-tamper"
    trace_writer = v9.V9ArtifactOnlyWriter(trace_root, contract, _token(contract, trace_root))
    prediction, targets = _arrays(first)
    trace_writer.write_cell_artifact(first, prediction, targets, evidence=_evidence(first, targets))
    trace_commit = trace_root / first.commit_relative_path
    trace_payload = json.loads(trace_commit.read_text(encoding="utf-8"))
    trace_payload["evidence"]["query_behavior_trace_sha256"] = "0" * 64
    _rewrite_immutable(trace_commit, v9.canonical_bytes(trace_payload))
    with pytest.raises(v9.V9ArtifactError, match="does not match ordered artifact targets"):
        v9.scan_artifact_state(trace_root, contract)
    with pytest.raises(v9.V9ArtifactError, match="does not match ordered artifact targets"):
        v9.finalize_synthetic_artifacts(trace_root, contract)


def test_direct_zero4_has_no_label_rate_or_normalizer_input_and_rejects_negative_zero() -> None:
    import numpy as np

    class Poison:
        def __array__(self, *args: Any, **kwargs: Any):
            raise AssertionError("zero4 descriptor read forbidden value")

        def __iter__(self):
            raise AssertionError("zero4 descriptor iterated forbidden value")

    record = {
        "n_units": 3,
        "neural": np.zeros((8, 3), dtype=np.float32),
        "target_direction": Poison(),
        "t4_trial_rates": Poison(),
        "side_mean": Poison(),
        "side_std": Poison(),
        "source_unit_count": 99,
    }
    updated = v9.attach_direct_zero4_to_synthetic_record(record)
    side = updated["side_features"]
    v9.require_direct_zero4(side)
    assert side.shape == (3, 4)
    receipt = updated["zero4_descriptor_receipt"]
    assert receipt == {
        "construction_input": ["channel_count"],
        "target_direction_label_reads_for_descriptor": 0,
        "t4_trial_rate_reads_for_descriptor": 0,
        "target_t4_rate_fit_calls": 0,
        "source_t4_normalizer_value_reads": 0,
        "source_t4_normalizer_arithmetic_performed": False,
        "raw_t4_constructed": False,
        "side_feature_loader_calls_for_zero4": 0,
        "side_normalizer_passed_to_zero4_constructor": False,
        "bitwise_positive_float32_zero": True,
        "side_shape": [3, 4],
    }
    with pytest.raises(v9.V9Zero4Error, match="bitwise"):
        v9.require_direct_zero4(np.full((3, 4), -0.0, dtype=np.float32))
    with pytest.raises(v9.V9Zero4Error, match="0 < channel_count < 100"):
        v9.direct_zero4_from_channel_count(0)
    with pytest.raises(v9.V9Zero4Error, match="0 < channel_count < 100"):
        v9.direct_zero4_from_channel_count(100)
    with pytest.raises(v9.V9Zero4Error, match="channel axis"):
        v9.attach_direct_zero4_to_synthetic_record({"n_units": 99, "neural": record["neural"]})


def test_synthetic_core_explicitly_refuses_formal_external_execution() -> None:
    with pytest.raises(v9.V9AuthorizationOrderError, match="no formal authorization"):
        v9.formal_external_execution_unavailable()
