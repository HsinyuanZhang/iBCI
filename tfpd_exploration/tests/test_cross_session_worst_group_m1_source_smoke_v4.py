"""No-data/no-CUDA gates for CS-WG M1 V4 common-stratum source smoke."""
from __future__ import annotations

import hashlib
import inspect
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
from tfpd_exploration.src.cross_session_worst_group_v1 import source_audit_v3 as v3  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_physical as physical  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_smoke_physical_v4 as physical_v4  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_smoke_v4 as v4  # noqa: E402


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


def _stage_v4_closure(tmp_path: Path) -> Path:
    staged = tmp_path / "stage"
    closure = v4.implementation_closure(ROOT)
    for row in closure["paths"]:
        relative = str(row["path"])
        source = ROOT / relative
        target = staged / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (staged / "tfpd_exploration/results").mkdir(parents=True, exist_ok=True)
    return staged


def _device() -> v1.DeviceProfile:
    return v1.DeviceProfile(
        cuda_visible_devices="7",
        torch_device="cuda:0",
        uuid="GPU-synthetic-v4",
        pci_bus_id="00000000:07:00.0",
        name="Synthetic GPU",
        compute_capability=(9, 0),
        total_memory_bytes=12_345_678,
        torch_version="synthetic-torch",
        cuda_version="synthetic-cuda",
        cudnn_version=90000,
    )


def _environ() -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "7",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        **v1.THREAD_ENVIRONMENT,
    }


def _descriptor(session: str) -> physical.SourceFileDescriptor:
    ordinal = plan.HELD_IN_SOURCE_SESSIONS.index(session) + 1
    return physical.SourceFileDescriptor(
        session_id=session,
        relative_path=f"synthetic_heldin/{session}.nwb",
        sha256=(f"{ordinal:x}" * 64)[:64],
        byte_count=1_000_000 + ordinal,
    )


def _raw_final_outputs() -> np.ndarray:
    rows: list[np.ndarray] = []
    for coordinate in range(12):
        for _ in range(2):
            item = np.zeros((plan.M1_RAW_BEHAVIOR_OUTPUTS,), dtype=np.float32)
            item[coordinate] = float(coordinate + 1)
            rows.append(item)
    return np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)


def _material(descriptor: physical.SourceFileDescriptor) -> physical.SourceSessionMaterial:
    raw = _raw_final_outputs()
    labels = core.SourceOnlyFinalBinLabels(descriptor.session_id, raw, np.ones((raw.shape[0],), dtype=np.bool_))
    calibration = np.full(
        plan.M1_CALIBRATION_SHAPE_PER_ROW,
        float(plan.HELD_IN_SOURCE_SESSIONS.index(descriptor.session_id) + 1), dtype=np.float32,
    )
    calibration.setflags(write=False)
    calibration_sha = core.array_digest(calibration)
    ordinal = plan.HELD_IN_SOURCE_SESSIONS.index(descriptor.session_id) + 1
    held = physical.held_source_identity_payload(
        descriptor, device=20, inode=10_000 + ordinal, byte_count=descriptor.byte_count,
        body_sha256=descriptor.sha256, mode=0o644, hard_link_count=1,
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
        view = calibration.view()
        view.setflags(write=False)
        rows[int(index)] = physical.UnassignedSourceM1Row(
            session_id=descriptor.session_id,
            sample_index=int(index),
            sample_id=f"{descriptor.session_id}:synthetic:{index}",
            calibration_session=descriptor.session_id,
            calibration_sha256=calibration_sha,
            model_inputs={
                "x": np.full((plan.M1_WINDOW_SIZE, plan.M1_UNIT_COUNT), float(index), dtype=np.float32),
                "calib_trialized_neural_features": view,
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


@pytest.fixture()
def synthetic_audit_and_smoke_prepared() -> physical_v4.CachedCommonStratumSmokePrepared:
    spec = v1.source_audit_spec()
    descriptors = tuple(_descriptor(session) for session in spec.stage0_spec.source_sessions)
    materials = {item.session_id: physical_v4._cache_material(_material(item)) for item in descriptors}
    audit = v3.build_common_stratum_prepared_audit(
        physical_module=physical, spec=spec, descriptors=descriptors, materials=materials,
    )
    return physical_v4.rebind_v3_audit_prepared_to_smoke(physical_module=physical, audit_prepared=audit)


def _fake_graph() -> v4.HeldV3CompletedAuditGraph:
    expectation = v4.V3CompletedAuditGraphExpectation(
        attempt_sha256="1" * 64,
        launch_sha256="2" * 64,
        source_authority_sha256="3" * 64,
        audit_sha256="4" * 64,
        terminal_sha256="5" * 64,
    )
    return v4.HeldV3CompletedAuditGraph(
        expectation=expectation,
        root_identity=(44, 55),
        named_chain_identities=((".", 1, 2),),
        attempt={"synthetic": "attempt"},
        launch={"synthetic": "launch"},
        source_authority={"deterministic_common_stratum_fallback": {"synthetic": "fallback"}},
        audit={"synthetic": "audit"},
        terminal={"synthetic": "terminal"},
    )


def _valid_inner_smoke(identity: v1.SourceExecutionIdentity) -> dict[str, object]:
    name_digest = _sha(_json_bytes(["weight"]))
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_v1",
        "identity_sha256": identity.sha256,
        "optimizer_steps": 100,
        "total_windows_per_step": 32,
        "one_concatenated_forward_per_step": True,
        "calibration_shape_per_row": [10, 1024, 64],
        "model_parameter_count": 15_007_496,
        "model_output_shape": [32, 100, 16],
        "finite_objective": True,
        "finite_model": True,
        "finite_gradients": True,
        "gradient_coverage": {
            "trainable_parameter_count": 1,
            "trainable_parameter_names_sha256": name_digest,
            "observed_gradient_count": 1,
            "observed_gradient_names_sha256": name_digest,
            "missing_trainable_names": [],
            "excluded_trainable_names": [],
        },
        "finite_adam_state": True,
        "model_state_changed": True,
        "checkpoint_reload_strict": True,
        "best_checkpoint_reload_strict": True,
        "last_checkpoint_reload_strict": True,
        "session_objective_derivatives_nonnegative": True,
        "dynamic_dropout_preserved": True,
        "initial_model_state_sha256": "a" * 64,
        "final_model_state_sha256": "b" * 64,
        "best_checkpoint_state_sha256": "c" * 64,
        "rng": {
            "schema": "cross_session_worst_group_m1_rng_policy_v1",
            "seed": 42,
            "domains": ["python", "numpy", "torch_cpu", "torch_cuda_selected"],
            "scheduler_uses_host_rng": False,
            "model_initialization_and_dynamic_dropout_seeded": True,
            "pre_run_state_digest": "d" * 64,
            "post_run_state_restored": True,
        },
        "resources": {
            "elapsed_seconds": 1.0,
            "steps_per_second": 100.0,
            "samples_per_second": 3200.0,
            "cuda_current_allocated_bytes": 1,
            "cuda_peak_allocated_bytes": 2,
            "cuda_current_reserved_bytes": 3,
            "cuda_peak_reserved_bytes": 4,
        },
        "model_constructed": True,
        "cuda_initialized": True,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
        "_checkpoint_bodies": {"best_source_train_loss": b"best", "last": b"last"},
    }


class _MockBackend:
    def __init__(self, *, staged: Path, prepared: physical_v4.CachedCommonStratumSmokePrepared,
                 fallback_sha256: str, events: list[str]) -> None:
        self.staged = staged
        self.prepared = prepared
        self.fallback_sha256 = fallback_sha256
        self.events = events
        self._inner: dict[str, object] | None = None

    def launch_payload(self, _identity: v4.SourceSmokeV4Identity) -> Mapping[str, object]:
        self.events.append("launch")
        return {
            "schema": "cross_session_worst_group_m1_source_smoke_physical_launch_v4",
            "provider": "V4CachedCommonStratumSourceProvider",
            "runner": "TorchCSWGSmokeRunner",
            "inherited_v3_fallback": v3.COMMON_STRATUM_FALLBACK_MODE,
            "source_opened": False,
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps_completed": 0,
            "source_only": True,
        }

    def prepare_source(self, _identity: v4.SourceSmokeV4Identity) -> object:
        assert (self.staged / v4.V4_ROOT_RELATIVE / "attempt.json").is_file()
        self.events.append("prepare")
        return self.prepared

    def run_smoke(self, identity: v4.SourceSmokeV4Identity) -> Mapping[str, object]:
        self.events.append("run")
        self._inner = _valid_inner_smoke(identity.inherited_v1_smoke_identity)
        return dict(self._inner)

    def checkpoint_bodies(self) -> Mapping[str, bytes]:
        return {"best_source_train_loss": b"best", "last": b"last"}

    def progress(self) -> v1.LifecycleProgress:
        return v1.LifecycleProgress(
            source_resolved_or_opened=True, model_constructed=True, cuda_initialized=True,
            optimizer_steps_completed=100, source_authority_published=True,
        )

    def close(self) -> None:
        self.events.append("close")


def _patched_prepared_fallback(
    prepared: physical_v4.CachedCommonStratumSmokePrepared, fallback: Mapping[str, object], digest: str,
) -> Any:
    """Typed test wrapper only; production physical prep remains exact V3 literal."""
    class _SyntheticPrepared:
        inherited_smoke_prepared = prepared.inherited_smoke_prepared
        fallback_payload = dict(fallback)
        fallback_sha256 = digest
        step_zero_common_stratum_evidence = dict(prepared.step_zero_common_stratum_evidence)

        @staticmethod
        def digest_cache_payload() -> Mapping[str, object]:
            return prepared.digest_cache_payload()

        @staticmethod
        def inherited_v1_authority_fragment() -> Mapping[str, object]:
            fragment = dict(prepared.inherited_v1_authority_fragment())
            fragment["deterministic_common_stratum_fallback"] = dict(fallback)
            fragment["deterministic_common_stratum_fallback_sha256"] = digest
            return fragment

    return _SyntheticPrepared()


def _synthetic_v3_semantics(**values: object) -> None:
    attempt = values["attempt"]
    launch = values["launch"]
    authority = values["authority"]
    audit = values["audit"]
    terminal = values["terminal"]
    expectation = values["expectation"]
    assert isinstance(attempt, Mapping) and isinstance(launch, Mapping)
    assert isinstance(authority, Mapping) and isinstance(audit, Mapping) and isinstance(terminal, Mapping)
    assert isinstance(expectation, v4.V3CompletedAuditGraphExpectation)
    assert launch["attempt_sha256"] == expectation.attempt_sha256
    assert audit["source_authority_sha256"] == expectation.source_authority_sha256
    assert terminal["audit_sha256"] == expectation.audit_sha256


def _write_synthetic_v3_completed_graph(staged: Path) -> v4.HeldV3CompletedAuditGraph:
    identity = v3.build_source_audit_v3_identity(staged)
    root = staged / v4.V3_ROOT_RELATIVE
    root.mkdir(parents=True)
    os.chmod(root, 0o755)
    attempt = {"schema": "synthetic_v3_attempt", "identity": identity.payload()}
    attempt_sha = _write_pair(root, "attempt.json", attempt)
    launch = {"schema": "synthetic_v3_launch", "attempt_sha256": attempt_sha}
    launch_sha = _write_pair(root, "launch.json", launch)
    authority = {"schema": "synthetic_v3_authority"}
    authority_sha = _write_pair(root, "source_authority.json", authority)
    audit = {"schema": "synthetic_v3_audit", "source_authority_sha256": authority_sha}
    audit_sha = _write_pair(root, "audit.json", audit)
    terminal = {"schema": "synthetic_v3_terminal", "audit_sha256": audit_sha}
    terminal_sha = _write_pair(root, "terminal.json", terminal)
    expectation = v4.V3CompletedAuditGraphExpectation(
        attempt_sha256=attempt_sha, launch_sha256=launch_sha, source_authority_sha256=authority_sha,
        audit_sha256=audit_sha, terminal_sha256=terminal_sha,
    )
    return v4._validate_held_v3_completed_graph(
        staged, expectation, identity, semantic_validator=_synthetic_v3_semantics,
    )


def test_v4_workorder_closure_and_static_literals(tmp_path: Path) -> None:
    assert v4.WORKORDER_SHA256 == "c3ae0d1f5ea1f033419891f7a5217dc3cdc0011284b545d2ae5d150886b0fbe6"
    assert v4.LIVE_V3_COMPLETED_EXPECTATION.payload()["exact_leaf_count"] == 10
    assert v4.V3_CLOSURE_SHA256 == "d52168e567188b8ede816f4764cf829ecd920b2540c323fee14569ec7503fa1e"
    assert v4.V3_FALLBACK_SHA256 == "2e763763bf55276bfef175815ec7b6cb454393a87278513f86f8d99677618089"
    closure = v4.implementation_closure(ROOT)
    assert closure["accepted_v3_closure_sha256"] == v4.V3_CLOSURE_SHA256
    assert len(closure["paths"]) == len(set(row["path"] for row in closure["paths"]))
    for expected in (
        v4.WORKORDER_RELATIVE,
        "tfpd_exploration/src/cross_session_worst_group_v1/source_smoke_v4.py",
        "tfpd_exploration/src/cross_session_worst_group_v1/source_smoke_physical_v4.py",
        "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_smoke_v4.py",
        "tfpd_exploration/tests/test_cross_session_worst_group_m1_source_smoke_v4.py",
    ):
        assert expected in [row["path"] for row in closure["paths"]]
    staged = _stage_v4_closure(tmp_path)
    identity = v4.build_source_smoke_v4_identity(staged, device=_device())
    v4.validate_source_smoke_v4_identity_current(staged, identity)
    workorder = staged / v4.WORKORDER_RELATIVE
    workorder.write_text(workorder.read_text(encoding="utf-8") + "\nforged\n", encoding="utf-8")
    with pytest.raises(v4.SourceSmokeV4Error, match="workorder|closure"):
        v4.validate_source_smoke_v4_identity_current(staged, identity)


def test_held_v3_ten_leaf_loader_rejects_topology_sidecar_and_body_drift(tmp_path: Path) -> None:
    staged = _stage_v4_closure(tmp_path)
    held = _write_synthetic_v3_completed_graph(staged)
    assert held.expectation.payload()["exact_leaf_count"] == 10
    graph_root = staged / v4.V3_ROOT_RELATIVE
    assert len(tuple(graph_root.iterdir())) == 10
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in graph_root.iterdir())
    extra = graph_root / "failure.json"
    extra.write_text("{}", encoding="utf-8")
    os.chmod(extra, 0o444)
    with pytest.raises(v4.SourceSmokeV4Error, match="topology|failure|extra"):
        v4._validate_held_v3_completed_graph(
            staged, held.expectation, v3.build_source_audit_v3_identity(staged),
            semantic_validator=_synthetic_v3_semantics,
        )
    os.chmod(extra, 0o644)
    extra.unlink()
    body = graph_root / "terminal.json"
    sidecar = graph_root / "terminal.json.sha256"
    os.chmod(body, 0o644)
    os.chmod(sidecar, 0o644)
    body.write_text("{}", encoding="utf-8")
    sidecar.write_text(f"{_sha(b'{}')}  terminal.json\n", encoding="ascii")
    os.chmod(body, 0o444)
    os.chmod(sidecar, 0o444)
    with pytest.raises(v4.SourceSmokeV4Error, match="digest"):
        v4._validate_held_v3_completed_graph(
            staged, held.expectation, v3.build_source_audit_v3_identity(staged),
            semantic_validator=_synthetic_v3_semantics,
        )


def test_cached_rows_preserve_exact_digest_semantics_without_calibration_rehash(
    synthetic_audit_and_smoke_prepared: physical_v4.CachedCommonStratumSmokePrepared,
) -> None:
    prepared = synthetic_audit_and_smoke_prepared
    session = prepared.inherited_smoke_prepared.spec.stage0_spec.source_sessions[0]
    material = prepared.inherited_smoke_prepared.materials[session]
    cached_row = next(iter(material.rows_by_sample_index.values())).bind_stratum(
        prepared.audit_prepared.fallback.eligible_common_min2[0],
    )
    base_row = material.rows_by_sample_index[cached_row.sample_index].base.bind_stratum(cached_row.stratum)
    assert cached_row.input_digests() == base_row.input_digests()
    assert np.shares_memory(
        cached_row.model_inputs["calib_trialized_neural_features"], material.calibration_backing,
    )
    # The cached method is intentionally a literal mapping return.  It does
    # not call the generic array hasher (which would rehash the shared 2.5MiB
    # backing once for every source row).
    assert "array_digest" not in inspect.getsource(physical_v4.CachedSourceEpisodeRow.input_digests)
    cache = prepared.digest_cache_payload()
    assert cache["calibration_digest_recomputed_per_row"] is False
    assert cache["all_b32_rows_still_stack_per_row_calibration"] is True
    v4._validate_digest_cache_payload(
        cache,
        type("Identity", (), {"inherited_v1_smoke_identity": type("Inner", (), {
            "spec": prepared.inherited_smoke_prepared.spec,
        })()})(),
    )
    forged = dict(cache)
    forged["calibration_digest_recomputed_per_row"] = True
    with pytest.raises(v4.SourceSmokeV4Error, match="digest cache"):
        v4._validate_digest_cache_payload(
            forged,
            type("Identity", (), {"inherited_v1_smoke_identity": type("Inner", (), {
                "spec": prepared.inherited_smoke_prepared.spec,
            })()})(),
        )


def test_cached_rows_share_one_session_backing_until_exact_b32_stack(
    synthetic_audit_and_smoke_prepared: physical_v4.CachedCommonStratumSmokePrepared,
) -> None:
    """The cache removes repeated 2.5MiB hashes, never B32 materialization.

    This is intentionally a source-only CPU proof: the V3-rebound rows keep
    their single immutable session backing before the route-owned core stacks
    an exact B32 into a newly owned tensor for the one mixed-session forward.
    """
    prepared = synthetic_audit_and_smoke_prepared
    episode = prepared.inherited_smoke_prepared.episode(0)
    rows = episode.all_rows
    assert len(rows) == 32 and len({row.sample_id for row in rows}) == 32
    session = episode.session_ids[0]
    session_rows = [row for row in rows if row.session_id == session]
    backing = prepared.inherited_smoke_prepared.materials[session].calibration_backing
    assert all(np.shares_memory(row.model_inputs["calib_trialized_neural_features"], backing)
               for row in session_rows)
    # The frozen core, not the cache, owns the explicit per-row B32 stack.
    concatenated = core.concatenate_episode_for_one_forward(
        episode, compatibility=prepared.inherited_smoke_prepared.compatibility,
    )
    calibration = concatenated.model_inputs["calib_trialized_neural_features"]
    assert tuple(calibration.shape) == (32, 10, 1024, 64)
    assert not np.shares_memory(calibration.cpu().numpy(), backing)
    import torch
    assert torch.cuda.is_initialized() is False


def test_v3_common_fallback_and_step_zero_are_preserved_by_v4_rebind(
    synthetic_audit_and_smoke_prepared: physical_v4.CachedCommonStratumSmokePrepared,
) -> None:
    prepared = synthetic_audit_and_smoke_prepared
    assert prepared.inherited_smoke_prepared.spec == v1.source_smoke_spec()
    assert prepared.fallback_payload == prepared.audit_prepared.fallback.payload()
    assert prepared.step_zero_common_stratum_evidence == prepared.audit_prepared.step_zero_common_stratum_evidence()
    episode = prepared.inherited_smoke_prepared.episode(0)
    assert episode.quota.counts == (10, 11, 11)
    assert len(episode.all_rows) == 32
    assert len({row.sample_id for row in episode.all_rows}) == 32
    source = inspect.getsource(physical_v4.rebind_v3_audit_prepared_to_smoke)
    assert "v1.source_smoke_spec()" in source and "PreparedSourceFold" in source


def test_capability_requires_predecessor_before_fresh_root_or_execution(tmp_path: Path) -> None:
    staged = _stage_v4_closure(tmp_path)
    identity = v4.build_source_smoke_v4_identity(staged, device=_device())
    events: list[str] = []
    with pytest.raises(v4.SourceSmokeV4Error, match="predecessor"):
        v4._issue_root_reviewed_source_smoke_v4_capability(
            staged, identity, environ=_environ(), review_seal=v4._V4_ROOT_REVIEW_SEAL,
            predecessor_loader=lambda _root: (_ for _ in ()).throw(v4.SourceSmokeV4Error("predecessor missing")),
        )
    assert not (staged / v4.V4_ROOT_RELATIVE).exists()
    graph = _fake_graph()
    capability = v4._issue_root_reviewed_source_smoke_v4_capability(
        staged, identity, environ=_environ(), review_seal=v4._V4_ROOT_REVIEW_SEAL,
        predecessor_loader=lambda _root: events.append("predecessor") or graph,
    )
    assert capability.v3_completed_graph_sha256 == graph.sha256 and events == ["predecessor"]


def test_complete_mock_lifecycle_is_attempt_before_source_and_terminalizes(
    tmp_path: Path, synthetic_audit_and_smoke_prepared: physical_v4.CachedCommonStratumSmokePrepared,
) -> None:
    staged = _stage_v4_closure(tmp_path)
    identity = v4.build_source_smoke_v4_identity(staged, device=_device())
    graph = _fake_graph()
    capability = v4.SourceSmokeV4Capability(identity.sha256, graph.sha256, v4._V4_ROOT_REVIEW_SEAL)
    synthetic_fallback = {"synthetic": "fallback", "source_only": True}
    synthetic_fallback_sha = _sha(_json_bytes(synthetic_fallback))
    prepared = _patched_prepared_fallback(synthetic_audit_and_smoke_prepared, synthetic_fallback, synthetic_fallback_sha)
    events: list[str] = []
    backend = _MockBackend(staged=staged, prepared=prepared, fallback_sha256=synthetic_fallback_sha, events=events)
    result = v4._execute_reviewed_source_smoke_v4(
        staged, identity=identity, capability=capability, backend=backend, environ=_environ(),
        predecessor_loader=lambda _root: graph, accepted_fallback_sha256=synthetic_fallback_sha,
    )
    assert result.terminal_sha256 is not None and result.failure_sha256 is None
    assert events == ["launch", "prepare", "run", "close"]
    root = staged / v4.V4_ROOT_RELATIVE
    assert {path.name for path in root.iterdir()} == {
        "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
        "source_authority.json", "source_authority.json.sha256", "smoke.json", "smoke.json.sha256",
        "checkpoint_best_source_train_loss.pt", "checkpoint_best_source_train_loss.pt.sha256",
        "checkpoint_last.pt", "checkpoint_last.pt.sha256", "checkpoint_manifest.json", "checkpoint_manifest.json.sha256",
        "terminal.json", "terminal.json.sha256",
    }
    authority = json.loads((root / "source_authority.json").read_text(encoding="utf-8"))
    smoke = json.loads((root / "smoke.json").read_text(encoding="utf-8"))
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    assert authority["route_local_input_digest_cache"]["calibration_digest_recomputed_per_row"] is False
    assert smoke["optimizer_steps"] == 100 and smoke["one_concatenated_forward_per_step"] is True
    assert terminal["source_only"] is True and terminal["target_optimizer_backward_update"] == 0
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in root.iterdir())
    for body in sorted(path for path in root.iterdir() if not path.name.endswith(".sha256")):
        digest = _sha(body.read_bytes())
        assert (root / f"{body.name}.sha256").read_bytes() == f"{digest}  {body.name}\n".encode("ascii")


def test_tampered_v3_graph_or_cached_authority_fails_before_terminal(
    tmp_path: Path, synthetic_audit_and_smoke_prepared: physical_v4.CachedCommonStratumSmokePrepared,
) -> None:
    staged = _stage_v4_closure(tmp_path)
    identity = v4.build_source_smoke_v4_identity(staged, device=_device())
    graph = _fake_graph()
    capability = v4.SourceSmokeV4Capability(identity.sha256, graph.sha256, v4._V4_ROOT_REVIEW_SEAL)
    fallback = {"synthetic": "fallback", "source_only": True}
    fallback_sha = _sha(_json_bytes(fallback))
    prepared = _patched_prepared_fallback(synthetic_audit_and_smoke_prepared, fallback, fallback_sha)
    # A changed returned graph no longer matches the capability before reserve.
    swapped = _fake_graph()
    swapped = v4.HeldV3CompletedAuditGraph(
        expectation=swapped.expectation, root_identity=(99, 100), named_chain_identities=swapped.named_chain_identities,
        attempt=swapped.attempt, launch=swapped.launch, source_authority=swapped.source_authority,
        audit=swapped.audit, terminal=swapped.terminal,
    )
    with pytest.raises(v4.SourceSmokeV4Error, match="predecessor/capability"):
        v4._execute_reviewed_source_smoke_v4(
            staged, identity=identity, capability=capability,
            backend=_MockBackend(staged=staged, prepared=prepared, fallback_sha256=fallback_sha, events=[]),
            environ=_environ(), predecessor_loader=lambda _root: swapped, accepted_fallback_sha256=fallback_sha,
        )
    assert not (staged / v4.V4_ROOT_RELATIVE).exists()


def test_v3_completed_graph_is_revalidated_after_forwards_before_terminal(
    tmp_path: Path, synthetic_audit_and_smoke_prepared: physical_v4.CachedCommonStratumSmokePrepared,
) -> None:
    staged = _stage_v4_closure(tmp_path)
    identity = v4.build_source_smoke_v4_identity(staged, device=_device())
    graph = _fake_graph()
    changed = v4.HeldV3CompletedAuditGraph(
        expectation=graph.expectation,
        root_identity=(graph.root_identity[0], graph.root_identity[1] + 1),
        named_chain_identities=graph.named_chain_identities,
        attempt=graph.attempt,
        launch=graph.launch,
        source_authority=graph.source_authority,
        audit=graph.audit,
        terminal=graph.terminal,
    )
    capability = v4.SourceSmokeV4Capability(identity.sha256, graph.sha256, v4._V4_ROOT_REVIEW_SEAL)
    fallback = {"synthetic": "fallback", "source_only": True}
    fallback_sha = _sha(_json_bytes(fallback))
    prepared = _patched_prepared_fallback(synthetic_audit_and_smoke_prepared, fallback, fallback_sha)
    events: list[str] = []
    calls = 0

    def loader(_root: Path) -> v4.HeldV3CompletedAuditGraph:
        nonlocal calls
        calls += 1
        return graph if calls == 1 else changed

    result = v4._execute_reviewed_source_smoke_v4(
        staged,
        identity=identity,
        capability=capability,
        backend=_MockBackend(staged=staged, prepared=prepared, fallback_sha256=fallback_sha, events=events),
        environ=_environ(),
        predecessor_loader=loader,
        accepted_fallback_sha256=fallback_sha,
    )
    assert calls == 2 and result.terminal_sha256 is None and result.failure_sha256 is not None
    root = staged / v4.V4_ROOT_RELATIVE
    failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
    assert failure["terminal_published"] is False
    assert "accepted V3 predecessor drifted" in failure["error_class"] or failure["error_class"] == "SourceSmokeV4Error"
    assert not (root / "terminal.json").exists()
    assert events == ["launch", "prepare", "run", "close"]


def test_physical_v4_runner_seam_is_narrow_and_deferred() -> None:
    source = inspect.getsource(physical_v4.PhysicalCommonStratumSmokeV4Backend.run_smoke)
    assert "identity.inherited_v1_smoke_identity" in source
    assert "self.smoke_runner.run" in source
    assert "torch" not in source.lower()
    factory = inspect.getsource(physical_v4.build_reviewed_v4_source_smoke_backend)
    assert "build_reviewed_v3_source_audit_backend" in factory
    assert "TorchCSWGSmokeRunner" in factory


def test_public_cli_and_import_are_static_no_torch_no_cuda(tmp_path: Path) -> None:
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
    probe = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; from tfpd_exploration.src.cross_session_worst_group_v1 import source_smoke_v4; "
            "print('torch' in sys.modules); print('src' in sys.modules)",
        ],
        cwd=tmp_path, env=environment, text=True, capture_output=True, check=True,
    )
    assert probe.stdout.splitlines() == ["False", "False"]
    script = ROOT / "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_smoke_v4.py"
    dry = subprocess.run([sys.executable, str(script), "--dry-run"], cwd=tmp_path, env=environment,
                         text=True, capture_output=True, check=True)
    payload = json.loads(dry.stdout)
    assert payload["imports_torch"] is False and payload["creates_root_or_receipt"] is False
    assert payload["accepted_v3_completed_graph"]["exact_leaf_count"] == 10
    denied = subprocess.run([sys.executable, str(script), "--execute"], cwd=tmp_path, env=environment,
                            text=True, capture_output=True)
    assert denied.returncode != 0 and "root-reviewed" in denied.stderr
