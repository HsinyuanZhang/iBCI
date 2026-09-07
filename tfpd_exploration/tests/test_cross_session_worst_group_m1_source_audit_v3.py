"""No-data/no-CUDA gates for CS-WG M1 source-audit V3."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfpd_exploration.src.cross_session_worst_group_v1 import core  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import plan  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_audit_v2 as v2  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_audit_v3 as v3  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_physical as physical  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_reader  # noqa: E402


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_pair(directory: Path, name: str, payload: Mapping[str, object]) -> str:
    body = _json_bytes(dict(payload))
    digest = _sha(body)
    body_path = directory / name
    sidecar_path = directory / f"{name}.sha256"
    body_path.write_bytes(body)
    sidecar_path.write_bytes(f"{digest}  {name}\n".encode("ascii"))
    os.chmod(body_path, 0o444)
    os.chmod(sidecar_path, 0o444)
    return digest


def _stage_v3_closure(tmp_path: Path) -> Path:
    staged = tmp_path / "stage"
    closure = v3.implementation_closure(ROOT)
    for row in closure["paths"]:
        relative = str(row["path"])
        source = ROOT / relative
        destination = staged / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    (staged / "tfpd_exploration/results").mkdir(parents=True, exist_ok=True)
    return staged


def _write_synthetic_v1_v2_failure_chain(staged: Path) -> tuple[
    v2.V1FailedGraphExpectation, v3.V2FailedGraphExpectation, v3.HeldFailurePredecessorChain,
]:
    """Create only temporary shaped receipt graphs; never read a real result root."""
    v1_identity = v1.build_source_audit_identity(staged)
    v1_attempt = v1._source_audit_attempt_payload(v1_identity)
    v1_attempt_sha = _sha(_json_bytes(v1_attempt))
    v1_backend = {
        "schema": "cross_session_worst_group_m1_source_audit_physical_launch_v1",
        "provider": "StrictM1SourceProvider",
        "source_opened": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
    }
    v1_launch = v1._source_audit_launch_payload(v1_identity, v1_attempt_sha, v1_backend)
    v1_launch_sha = _sha(_json_bytes(v1_launch))
    v1_error = source_reader.SourceReaderError(
        "CS-WG native parser refuses to replace an existing non-streaming top-level src",
    )
    v1_failure = v1._source_audit_failure_payload(
        v1_identity,
        attempt_sha256=v1_attempt_sha,
        launch_sha256=v1_launch_sha,
        authority_sha256=None,
        progress=v1.LifecycleProgress(source_resolved_or_opened=False),
        error=v1_error,
    )
    v1_failure_sha = _sha(_json_bytes(v1_failure))
    v1_root = staged / v2.V1_FAILED_ROOT_RELATIVE
    v1_root.mkdir(parents=True)
    os.chmod(v1_root, 0o755)
    assert _write_pair(v1_root, "attempt.json", v1_attempt) == v1_attempt_sha
    assert _write_pair(v1_root, "launch.json", v1_launch) == v1_launch_sha
    assert _write_pair(v1_root, "failure.json", v1_failure) == v1_failure_sha
    v1_expectation = v2.V1FailedGraphExpectation(
        root_relative=v2.V1_FAILED_ROOT_RELATIVE,
        attempt_sha256=v1_attempt_sha,
        launch_sha256=v1_launch_sha,
        failure_sha256=v1_failure_sha,
        error_class=type(v1_error).__name__,
        error_sha256=_sha(repr(v1_error).encode("utf-8")),
    )
    held_v1 = v2._validate_held_v1_failed_audit_graph(staged, v1_expectation)

    v2_identity = v2.build_source_audit_v2_identity(staged)
    v2_attempt = v2._v2_attempt_payload(v2_identity, held_v1)
    v2_attempt_sha = _sha(_json_bytes(v2_attempt))
    v2_launch = v2._v2_launch_payload(v2_identity, held_v1, v2_attempt_sha, v1_backend)
    v2_launch_sha = _sha(_json_bytes(v2_launch))
    v2_error = physical.SourcePhysicalError("CS-WG common source stratum topology absent: synthetic")
    v2_failure = v2._v2_failure_payload(
        v2_identity,
        held_v1,
        attempt_sha256=v2_attempt_sha,
        launch_sha256=v2_launch_sha,
        authority_sha256=None,
        progress=v1.LifecycleProgress(source_resolved_or_opened=True),
        error=v2_error,
    )
    v2_failure_sha = _sha(_json_bytes(v2_failure))
    v2_root = staged / v3.V2_FAILED_ROOT_RELATIVE
    v2_root.mkdir(parents=True)
    os.chmod(v2_root, 0o755)
    assert _write_pair(v2_root, "attempt.json", v2_attempt) == v2_attempt_sha
    assert _write_pair(v2_root, "launch.json", v2_launch) == v2_launch_sha
    assert _write_pair(v2_root, "failure.json", v2_failure) == v2_failure_sha
    v2_expectation = v3.V2FailedGraphExpectation(
        root_relative=v3.V2_FAILED_ROOT_RELATIVE,
        attempt_sha256=v2_attempt_sha,
        launch_sha256=v2_launch_sha,
        failure_sha256=v2_failure_sha,
        error_class=type(v2_error).__name__,
        error_sha256=_sha(repr(v2_error).encode("utf-8")),
    )
    held_v2 = v3._validate_held_v2_failed_audit_graph(staged, v2_expectation, held_v1, v2_identity)
    return v1_expectation, v2_expectation, v3.HeldFailurePredecessorChain(held_v1, held_v2)


def _descriptor(session: str) -> physical.SourceFileDescriptor:
    ordinal = plan.HELD_IN_SOURCE_SESSIONS.index(session) + 1
    return physical.SourceFileDescriptor(
        session_id=session,
        relative_path=f"synthetic_heldin/{session}.nwb",
        sha256=(f"{ordinal:x}" * 64)[:64],
        byte_count=1_000_000 + ordinal,
    )


def _raw_final_outputs(seed_counts: Mapping[int, int]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for seed, count in sorted(seed_counts.items()):
        assert 0 <= seed < plan.M1_RAW_BEHAVIOR_OUTPUTS and count > 0
        for _ in range(count):
            row = np.zeros((plan.M1_RAW_BEHAVIOR_OUTPUTS,), dtype=np.float32)
            row[seed] = float(seed + 1)
            rows.append(row)
    return np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)


def _material(descriptor: physical.SourceFileDescriptor, seed_counts: Mapping[int, int]) -> physical.SourceSessionMaterial:
    raw = _raw_final_outputs(seed_counts)
    labels = core.SourceOnlyFinalBinLabels(descriptor.session_id, raw, np.ones((raw.shape[0],), dtype=np.bool_))
    calibration = np.full(
        plan.M1_CALIBRATION_SHAPE_PER_ROW,
        float(plan.HELD_IN_SOURCE_SESSIONS.index(descriptor.session_id) + 1),
        dtype=np.float32,
    )
    calibration.setflags(write=False)
    calibration_sha = core.array_digest(calibration)
    ordinal = plan.HELD_IN_SOURCE_SESSIONS.index(descriptor.session_id) + 1
    held = physical.held_source_identity_payload(
        descriptor,
        device=20,
        inode=1_000 + ordinal,
        byte_count=descriptor.byte_count,
        body_sha256=descriptor.sha256,
        mode=0o644,
        hard_link_count=1,
    )
    native = physical.native_session_evidence(
        descriptor,
        calibration_session=descriptor.session_id,
        calibration_sha256=calibration_sha,
        row_count=len(labels.valid_indices),
        ordered_query_identity_sha256=(f"{ordinal:x}" * 64)[:64],
        ordered_window_start_sha256="a" * 64,
        ordered_target_evalmask_sha256="b" * 64,
        held_before=held,
        held_after=held,
        reader_recipe_sha256="c" * 64,
        external_versions={
            "falcon_challenge": "synthetic", "lightning": "synthetic", "numpy": "synthetic",
            "scipy": "synthetic", "torch": "synthetic",
        },
    )
    rows: dict[int, physical.UnassignedSourceM1Row] = {}
    for index in labels.valid_indices.tolist():
        rows[int(index)] = physical.UnassignedSourceM1Row(
            session_id=descriptor.session_id,
            sample_index=int(index),
            sample_id=f"{descriptor.session_id}:synthetic:{index}",
            calibration_session=descriptor.session_id,
            calibration_sha256=calibration_sha,
            model_inputs={
                "x": np.full((plan.M1_WINDOW_SIZE, plan.M1_UNIT_COUNT), float(index), dtype=np.float32),
                "calib_trialized_neural_features": calibration.view(),
            },
            raw_final_target=labels.raw_final_outputs[index],
        )
    return physical.SourceSessionMaterial(
        descriptor=descriptor,
        labels=labels,
        rows_by_sample_index=rows,
        valid_source_windows=96,
        calibration_session=descriptor.session_id,
        calibration_sha256=calibration_sha,
        calibration_backing=calibration,
        native_evidence=native,
    )


def _materials_for_counts(seed_counts_by_session: Mapping[str, Mapping[int, int]]) -> tuple[
    v1.SourceRouteSpec, tuple[physical.SourceFileDescriptor, ...], Mapping[str, physical.SourceSessionMaterial],
]:
    spec = v1.source_audit_spec()
    sessions = spec.stage0_spec.source_sessions
    assert tuple(seed_counts_by_session) == sessions
    descriptors = tuple(_descriptor(session) for session in sessions)
    materials = {
        descriptor.session_id: _material(descriptor, seed_counts_by_session[descriptor.session_id])
        for descriptor in descriptors
    }
    return spec, descriptors, materials


def _equal_counts() -> dict[str, dict[int, int]]:
    spec = v1.source_audit_spec()
    return {session: {seed: 2 for seed in range(12)} for session in spec.stage0_spec.source_sessions}


def _unequal_min2_counts() -> dict[str, dict[int, int]]:
    sessions = v1.source_audit_spec().stage0_spec.source_sessions
    return {
        sessions[0]: {seed: 2 for seed in range(12)},
        sessions[1]: {seed: 2 for seed in range(13)},
        sessions[2]: {**{seed: 2 for seed in range(10)}, 10: 1, 13: 2},
    }


def _short_counts() -> dict[str, dict[int, int]]:
    return {
        session: {seed: 2 for seed in range(9)}
        for session in v1.source_audit_spec().stage0_spec.source_sessions
    }


def _prepared(seed_counts_by_session: Mapping[str, Mapping[int, int]]) -> v3.CommonStratumPreparedAudit:
    spec, descriptors, materials = _materials_for_counts(seed_counts_by_session)
    return v3.build_common_stratum_prepared_audit(
        physical_module=physical, spec=spec, descriptors=descriptors, materials=materials,
    )


class _SyntheticManifest:
    def __init__(self, descriptors: tuple[physical.SourceFileDescriptor, ...]) -> None:
        self.descriptors = descriptors
        self.calls: list[tuple[str, ...]] = []

    def resolve_exact_sources(self, spec: v1.SourceRouteSpec) -> tuple[physical.SourceFileDescriptor, ...]:
        self.calls.append(tuple(spec.stage0_spec.source_sessions))
        return self.descriptors


class _SyntheticReader:
    def __init__(
        self, *, root: Path, materials: Mapping[str, physical.SourceSessionMaterial], events: list[str],
    ) -> None:
        self.root = root
        self.materials = materials
        self.events = events

    def read_source_session(self, descriptor: physical.SourceFileDescriptor) -> physical.SourceSessionMaterial:
        assert (self.root / v3.V3_ROOT_RELATIVE / "attempt.json").is_file()
        self.events.append(f"read:{descriptor.session_id}")
        return self.materials[descriptor.session_id]


def _no_bootstrap(_root: Path) -> object:
    return object()


def _chain_loader(chain: v3.HeldFailurePredecessorChain):
    return lambda _root: chain


def _issue_test_capability(
    staged: Path, identity: v3.SourceAuditV3Identity, chain: v3.HeldFailurePredecessorChain,
) -> v3.SourceAuditV3Capability:
    return v3._issue_source_audit_v3_capability(
        staged,
        identity,
        review_seal=v3._V3_ROOT_REVIEW_SEAL,
        predecessor_loader=_chain_loader(chain),
        route_bootstrapper=_no_bootstrap,
    )


def test_v3_workorder_v2_closure_identity_metadata_and_static_module_binding(tmp_path: Path) -> None:
    assert v3.WORKORDER_SHA256 == "ebc536ff0a2e1c64e0a6e33b8b6ca0f744f79b08db2b7c56d45275481ea94d14"
    assert v3.V2_REPAIRED_CLOSURE_SHA256 == "ca77c59103830fcb38e6d51af49b355485b6b916a3b4402e26d4b20e21588fa0"
    assert v3.V2_IDENTITY_SHA256 == "e9565bce5d0f2e4e683b599f5e0e0e65a089eae4d0308afd7aa2d58170a6b581"
    assert v2.LIVE_V1_FAILED_EXPECTATION.payload()["pairs"] == [
        {"name": "attempt.json", "sha256": "5d0cd206644d29b4ac81139d9a7cbe4aaedc404935a1029597ef1a0f3f9a1e75"},
        {"name": "launch.json", "sha256": "895fb18d342a51a91e97e5a89ef13e53d662bd556fc2b0881ca1e7be5a70a99d"},
        {"name": "failure.json", "sha256": "f5ec355bd26d4bb13f48117e122a2f2b4c5eb3dac2e18cc9b357dfa10554436b"},
    ]
    assert v3.LIVE_V2_FAILED_EXPECTATION.payload()["pairs"] == [
        {"name": "attempt.json", "sha256": "163e73fc5c793b009e7ec86f4af7cc1841bec0bbf5865629c7013a3df468350b"},
        {"name": "launch.json", "sha256": "9af2bb8c400dad927b0d3dbf711d9d24242c0e4bfb55b8163a96e0a5e4ea4fcc"},
        {"name": "failure.json", "sha256": "945a7b5f843e14a6cb4008309c9df4ab510b5f60dbda4a33ba54e6dc563889cb"},
    ]
    closure = v3.implementation_closure(ROOT)
    assert closure["inherited_v2_closure_sha256"] == v3.V2_REPAIRED_CLOSURE_SHA256
    assert closure["sealed_metadata_manifest_sha256"] == v1.M1_METADATA_MANIFEST_SHA256
    assert all(hasattr(v3, name) for name in v3.__all__)
    paths = [row["path"] for row in closure["paths"]]
    for path in (
        v3.WORKORDER_RELATIVE,
        "tfpd_exploration/src/cross_session_worst_group_v1/source_audit_v3.py",
        "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_audit_v3.py",
        "tfpd_exploration/tests/test_cross_session_worst_group_m1_source_audit_v3.py",
    ):
        assert path in paths
    staged = _stage_v3_closure(tmp_path)
    identity = v3.build_source_audit_v3_identity(staged)
    v3.validate_source_audit_v3_identity_current(staged, identity)
    workorder = staged / v3.WORKORDER_RELATIVE
    workorder.write_text(workorder.read_text(encoding="utf-8") + "\nforged\n", encoding="utf-8")
    with pytest.raises(v3.SourceAuditV3Error, match="workorder|closure"):
        v3.validate_source_audit_v3_identity_current(staged, identity)


def test_v1_v2_held_failure_chain_enforces_exact_topology_semantics_and_links(tmp_path: Path) -> None:
    staged = _stage_v3_closure(tmp_path)
    v1_expectation, v2_expectation, chain = _write_synthetic_v1_v2_failure_chain(staged)
    assert chain.v2_graph.v1_predecessor_sha256 == chain.v1_graph.sha256
    assert chain.v2_graph.failure["progress"]["source_resolved_or_opened"] is True
    root = staged / v2_expectation.root_relative
    assert tuple(sorted(path.name for path in root.iterdir())) == (
        "attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256",
        "launch.json", "launch.json.sha256",
    )
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in root.iterdir())

    extra = root / "terminal.json"
    extra.write_text("{}", encoding="utf-8")
    os.chmod(extra, 0o444)
    with pytest.raises(v3.SourceAuditV3Error, match="topology"):
        v3._validate_held_v2_failed_audit_graph(staged, v2_expectation, chain.v1_graph, v2.build_source_audit_v2_identity(staged))
    os.chmod(extra, 0o644)
    extra.unlink()

    failure_path = root / "failure.json"
    body = json.loads(failure_path.read_text(encoding="utf-8"))
    body["error_class"] = "ForgedError"
    os.chmod(failure_path, 0o644)
    os.chmod(root / "failure.json.sha256", 0o644)
    forged_sha = _write_pair(root, "failure.json", body)
    forged = replace(v2_expectation, failure_sha256=forged_sha)
    with pytest.raises(v3.SourceAuditV3Error, match="failure source/model/CUDA|semantic"):
        v3._validate_held_v2_failed_audit_graph(staged, forged, chain.v1_graph, v2.build_source_audit_v2_identity(staged))

    # V1 is independently held; a V2 graph cannot merely repeat a hand-made
    # V1 binding payload with a different current V1 directory identity.
    v1_root = staged / v1_expectation.root_relative
    replacement = v1_root.with_name("v1_replaced")
    v1_root.rename(replacement)
    os.symlink(replacement.name, v1_root, target_is_directory=True)
    with pytest.raises(v3.SourceAuditV3Error, match="V1 predecessor"):
        v3.validate_v1_v2_failed_predecessor_chain(staged)


def test_exact_equal_common_strata_preserve_all_rows_and_deterministic_episodes() -> None:
    prepared = _prepared(_equal_counts())
    evidence = prepared.fallback
    assert evidence.constructible is True
    assert len(evidence.eligible_common_min2) == 12
    payload = evidence.payload()
    assert payload["raw_common_intersection"] == payload["eligible_common_min2"]
    for session, retention in payload["per_session_retention"].items():
        assert retention["retained_fraction"] == 1.0
        assert retention["dropped_noncommon_window_count"] == 0
        assert retention["dropped_sparse_common_window_count"] == 0
        assert retention["original_pool_row_sha256"] == retention["retained_pool_row_sha256"]
        assert session in prepared.inherited_prepared.pools
    for step_index in range(24):
        episode = prepared.episode(step_index)
        assert episode.quota.counts == plan.outer_fold_episode_quota(episode.session_ids, step_index=step_index).counts
        assert len(episode.all_rows) == 32
        for batch in episode.microbatches:
            assert len({row.sample_id for row in batch.rows}) == len(batch.rows)
            assert {row.stratum for row in batch.rows}.issubset(set(evidence.eligible_common_min2))
            values = [sum(row.stratum == item for row in batch.rows) for item in episode.represented_strata]
            assert max(values) - min(values) <= 1
            assert max(values) <= 2


def test_unequal_min2_common_prunes_original_pools_and_excludes_count_one_stratum() -> None:
    prepared = _prepared(_unequal_min2_counts())
    evidence = prepared.fallback
    payload = evidence.payload()
    assert len(evidence.raw_common) == 11
    assert len(evidence.eligible_common_min2) == 10
    assert payload["eligible_common_min2_count"] == 10
    assert payload["exact_identical_eligible_training_pool_set"] is True
    sparse_drops = [entry["dropped_sparse_common_window_count"] for entry in payload["per_session_retention"].values()]
    assert sorted(sparse_drops) == [1, 2, 2]
    assert all(entry["retained_eligible_window_count"] == 20
               for entry in payload["per_session_retention"].values())
    for pool in prepared.inherited_prepared.pools.values():
        assert pool.strata == evidence.eligible_common_min2
        assert all(row.stratum in evidence.eligible_common_min2 for row in pool.rows)

    reversed_counts = {
        session: tuple(reversed(table)) for session, table in evidence.original_counts.items()
    }
    with pytest.raises(v3.SourceAuditV3Error, match="count-table"):
        replace(evidence, original_counts=reversed_counts)


def test_fewer_than_ten_common_min2_strata_is_typed_pre_model_cuda_stop() -> None:
    spec, descriptors, materials = _materials_for_counts(_short_counts())
    with pytest.raises(v3.CommonStratumConstructibilityError) as caught:
        v3.build_common_stratum_prepared_audit(
            physical_module=physical, spec=spec, descriptors=descriptors, materials=materials,
        )
    failure = caught.value.payload()
    assert failure["stage"] == "common_stratum_constructibility"
    assert failure["reason"] == "eligible_common_min2_below_required_10"
    assert failure["eligible_common_min2_count"] == 9
    assert failure["model_constructed"] is False and failure["cuda_initialized"] is False
    assert failure["optimizer_steps_completed"] == 0
    assert failure["fallback_topology"]["constructible_step_zero_b32"] is False

    zero_eligible = {
        session: {seed: 1 for seed in range(9)}
        for session in v1.source_audit_spec().stage0_spec.source_sessions
    }
    spec, descriptors, materials = _materials_for_counts(zero_eligible)
    with pytest.raises(v3.CommonStratumConstructibilityError) as caught_zero:
        v3.build_common_stratum_prepared_audit(
            physical_module=physical, spec=spec, descriptors=descriptors, materials=materials,
        )
    assert caught_zero.value.payload()["eligible_common_min2_count"] == 0


def test_source_order_target_leakage_and_pool_substitution_fail_closed() -> None:
    spec, descriptors, materials = _materials_for_counts(_equal_counts())
    reversed_materials = {session: materials[session] for session in reversed(tuple(materials))}
    with pytest.raises(v3.SourceAuditV3Error, match="ordering|selection"):
        v3.build_common_stratum_prepared_audit(
            physical_module=physical, spec=spec, descriptors=descriptors, materials=reversed_materials,
        )
    target_descriptor = _descriptor(spec.stage0_spec.outer_target_session)
    target_material = _material(target_descriptor, {seed: 2 for seed in range(12)})
    leaked = dict(materials)
    leaked.pop(spec.stage0_spec.source_sessions[0])
    leaked[spec.stage0_spec.outer_target_session] = target_material
    with pytest.raises(v3.SourceAuditV3Error, match="selection|outer-target"):
        v3.build_common_stratum_prepared_audit(
            physical_module=physical, spec=spec, descriptors=descriptors, materials=leaked,
        )

    prepared = _prepared(_equal_counts())
    pools = dict(prepared.inherited_prepared.pools)
    swapped = dict(pools)
    sessions = tuple(swapped)
    swapped[sessions[1]] = pools[sessions[0]]
    with pytest.raises(v3.SourceAuditV3Error, match="pool"):
        v3.derive_common_stratum_fallback(swapped, source_sessions=sessions)


def test_fallback_receipt_reconstructs_original_assignments_and_step_zero_witness(tmp_path: Path) -> None:
    """A reported fallback cannot drift away from all original source labels."""
    staged = _stage_v3_closure(tmp_path)
    _v1_expectation, _v2_expectation, chain = _write_synthetic_v1_v2_failure_chain(staged)
    identity = v3.build_source_audit_v3_identity(staged)
    prepared = _prepared(_unequal_min2_counts())
    authority = v3._v3_source_authority_payload(identity, chain, prepared)
    v3._validate_v3_source_authority(authority, identity, chain, prepared)
    witness = authority["common_stratum_step_zero_evidence"]
    assert witness["quota"]["counts"] == [10, 11, 11]
    assert witness["row_count"] == 32
    assert witness["max_rows_per_represented_stratum_per_session"] == 2

    forged_assignment = json.loads(json.dumps(authority))
    first_session = prepared.fallback.source_sessions[0]
    forged_assignment["inherited_v1_source_authority"]["assigned_strata"][first_session]["strata"][0] = {
        "active_flag": True,
        "raw_output_norm_quantile": 3,
        "dominant_coordinate": 15,
    }
    forged_assignment["inherited_v1_source_authority_sha256"] = _sha(_json_bytes(
        forged_assignment["inherited_v1_source_authority"],
    ))
    with pytest.raises(v3.SourceAuditV3Error, match="count table|assignment"):
        v3._validate_v3_source_authority(forged_assignment, identity, chain, prepared)

    forged_witness = json.loads(json.dumps(authority))
    forged_witness["common_stratum_step_zero_evidence"]["row_count"] = 31
    with pytest.raises(v3.SourceAuditV3Error, match="fallback|step-zero"):
        v3._validate_v3_source_authority(forged_witness, identity, chain, prepared)

    events: list[str] = []
    with pytest.raises(v3.SourceAuditV3Error, match="predecessor"):
        v3._issue_source_audit_v3_capability(
            staged,
            identity,
            review_seal=v3._V3_ROOT_REVIEW_SEAL,
            predecessor_loader=lambda _root: (_ for _ in ()).throw(v3.SourceAuditV3Error("predecessor tamper")),
            route_bootstrapper=lambda _root: events.append("bootstrap"),
        )
    assert events == []
    assert not (staged / v3.V3_ROOT_RELATIVE).exists()


def test_physical_provider_and_v3_lifecycle_are_attempt_before_source_and_terminalize(tmp_path: Path) -> None:
    staged = _stage_v3_closure(tmp_path)
    _v1_expectation, _v2_expectation, chain = _write_synthetic_v1_v2_failure_chain(staged)
    identity = v3.build_source_audit_v3_identity(staged)
    capability = _issue_test_capability(staged, identity, chain)
    spec, descriptors, materials = _materials_for_counts(_unequal_min2_counts())
    assert spec == identity.inherited_v1_identity.spec
    events: list[str] = []
    provider = v3.CommonStratumSourceProvider(
        physical, _SyntheticManifest(descriptors), _SyntheticReader(root=staged, materials=materials, events=events),
    )
    backend = v3.PhysicalCommonStratumAuditBackend(provider)
    result = v3._execute_reviewed_source_audit_v3(
        staged, identity=identity, capability=capability, backend=backend,
        predecessor_loader=_chain_loader(chain), route_bootstrapper=_no_bootstrap,
    )
    assert result.terminal_sha256 is not None and result.failure_sha256 is None
    assert events == [f"read:{session}" for session in spec.stage0_spec.source_sessions]
    root = staged / v3.V3_ROOT_RELATIVE
    assert {path.name for path in root.iterdir()} == {
        "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
        "source_authority.json", "source_authority.json.sha256", "audit.json", "audit.json.sha256",
        "terminal.json", "terminal.json.sha256",
    }
    authority = json.loads((root / "source_authority.json").read_text(encoding="utf-8"))
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    assert authority["deterministic_common_stratum_fallback"]["eligible_common_min2_count"] == 10
    assert terminal["source_only"] is True and terminal["model_constructed"] is False
    assert terminal["cuda_initialized"] is False and terminal["optimizer_steps_completed"] == 0


def test_typed_common_stratum_failure_is_transactional_without_authority_or_terminal(tmp_path: Path) -> None:
    staged = _stage_v3_closure(tmp_path)
    _v1_expectation, _v2_expectation, chain = _write_synthetic_v1_v2_failure_chain(staged)
    identity = v3.build_source_audit_v3_identity(staged)
    capability = _issue_test_capability(staged, identity, chain)
    _spec, descriptors, materials = _materials_for_counts(_short_counts())
    provider = v3.CommonStratumSourceProvider(
        physical, _SyntheticManifest(descriptors), _SyntheticReader(root=staged, materials=materials, events=[]),
    )
    result = v3._execute_reviewed_source_audit_v3(
        staged, identity=identity, capability=capability,
        backend=v3.PhysicalCommonStratumAuditBackend(provider),
        predecessor_loader=_chain_loader(chain), route_bootstrapper=_no_bootstrap,
    )
    assert result.terminal_sha256 is None and result.failure_sha256 is not None
    root = staged / v3.V3_ROOT_RELATIVE
    assert {path.name for path in root.iterdir()} == {
        "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
        "failure.json", "failure.json.sha256",
    }
    failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
    assert failure["failure_stage"] == "common_stratum_constructibility"
    assert failure["typed_failure_topology"]["eligible_common_min2_count"] == 9
    assert failure["source_authority_sha256"] is None
    assert failure["progress"]["source_resolved_or_opened"] is True
    assert failure["progress"]["model_constructed"] is False
    assert failure["progress"]["cuda_initialized"] is False
    assert failure["progress"]["optimizer_steps_completed"] == 0


def test_clean_namespace_subprocess_and_static_cli_have_no_nwb_or_cuda(tmp_path: Path) -> None:
    environment = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONPATH": str(ROOT),
    }
    import_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from tfpd_exploration.src.cross_session_worst_group_v1 import source_audit_v3; "
            "print('torch' in sys.modules); print('src' in sys.modules)",
        ],
        cwd=tmp_path, env=environment, text=True, capture_output=True, check=True,
    )
    assert import_probe.stdout.splitlines() == ["False", "False"]
    bad = "\n".join((
        "import pathlib, sys",
        f"root = pathlib.Path({str(ROOT)!r})",
        "sys.path.insert(0, str(root / 'tfpd_exploration'))",
        "import src.cross_session_worst_group_v1.source_physical",
        "from tfpd_exploration.src.cross_session_worst_group_v1 import source_audit_v3 as v3",
        "try:",
        "    v3.bootstrap_reviewed_v3_route(root)",
        "except v3.SourceAuditV3Error as error:",
        "    print(type(error).__name__)",
        "    print(str(error))",
        "else:",
        "    raise SystemExit('bad namespace unexpectedly accepted')",
    ))
    result = subprocess.run([sys.executable, "-c", bad], cwd=tmp_path, env=environment,
                            text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == ["SourceAuditV3Error", "CS-WG V3 reviewed namespace bootstrap failed"]

    good = (
        "import pathlib, torch; "
        f"root=pathlib.Path({str(ROOT)!r}); "
        "from tfpd_exploration.src.cross_session_worst_group_v1 import source_audit_v3 as v3; "
        "value=v3.reach_deferred_parser_seam(root); "
        "source_root=root/'lexical_source_root_not_opened'; "
        "backend=v3.build_reviewed_v3_source_audit_backend(root=root,source_root=source_root); "
        "print(value['opens_nwb']); print(type(backend).__name__); "
        "print(len(backend.provider.read_events)); print(source_root.exists()); "
        "print(torch.cuda.is_initialized())"
    )
    result = subprocess.run([sys.executable, "-c", good], cwd=tmp_path, env=environment,
                            text=True, capture_output=True, check=True)
    opened, backend_name, read_count, source_exists, cuda = result.stdout.splitlines()
    assert opened == "False" and backend_name == "PhysicalCommonStratumAuditBackend"
    assert read_count == "0" and source_exists == "False" and cuda == "False"

    script = ROOT / "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_audit_v3.py"
    probe = subprocess.run(
        [sys.executable, "-c", f"import runpy,sys; runpy.run_path({str(script)!r}); print('torch' in sys.modules)"],
        cwd=tmp_path, env=environment, text=True, capture_output=True, check=True,
    )
    assert probe.stdout.strip() == "False"
    dry = subprocess.run([sys.executable, str(script), "--dry-run"], cwd=tmp_path, env=environment,
                         text=True, capture_output=True, check=True)
    payload = json.loads(dry.stdout)
    assert payload["current_gpu_smoke_capability_issuable"] is False
    assert payload["opens_source_or_target"] is False and payload["initializes_cuda"] is False
    denied = subprocess.run([sys.executable, str(script), "--execute"], cwd=tmp_path, env=environment,
                            text=True, capture_output=True)
    assert denied.returncode != 0 and "root-reviewed" in denied.stderr
