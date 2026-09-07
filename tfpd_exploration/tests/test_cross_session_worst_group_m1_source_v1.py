"""Focused no-data/no-CUDA gates for CS-WG M1 Source Lifecycle V1."""
from __future__ import annotations

import gc
from dataclasses import dataclass, replace
import inspect
import os
from pathlib import Path
import random
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
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as lifecycle  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_physical as physical  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_reader as reader  # noqa: E402


def _descriptor(session: str) -> physical.SourceFileDescriptor:
    digest = (f"{plan.HELD_IN_SOURCE_SESSIONS.index(session) + 1:x}" * 64)[:64]
    return physical.SourceFileDescriptor(
        session_id=session,
        relative_path=f"m1_heldin/{session}.nwb",
        sha256=digest,
        byte_count=10_000 + plan.HELD_IN_SOURCE_SESSIONS.index(session),
    )


def _manifest() -> physical.FrozenM1SourceManifest:
    return physical.FrozenM1SourceManifest({session: _descriptor(session) for session in plan.HELD_IN_SOURCE_SESSIONS})


def _labels(session: str, *, dominant_coordinate: int = 0) -> core.SourceOnlyFinalBinLabels:
    # One common stratum with exactly eleven candidates is sufficient for the
    # step-0 11/11/10 episode and keeps this test fixture compact.
    raw = np.zeros((11, plan.M1_RAW_BEHAVIOR_OUTPUTS), dtype=np.float32)
    raw[:, dominant_coordinate] = 1.0
    return core.SourceOnlyFinalBinLabels(session, raw, np.ones((11,), dtype=np.bool_))


def _native_evidence(
    descriptor: physical.SourceFileDescriptor, *, calibration_sha256: str, row_count: int,
) -> dict[str, object]:
    held = physical.held_source_identity_payload(
        descriptor,
        device=1,
        inode=1000 + plan.HELD_IN_SOURCE_SESSIONS.index(descriptor.session_id),
        byte_count=descriptor.byte_count,
        body_sha256=descriptor.sha256,
        mode=0o644,
        hard_link_count=1,
    )
    prefix = f"{plan.HELD_IN_SOURCE_SESSIONS.index(descriptor.session_id) + 1:x}" * 64
    return physical.native_session_evidence(
        descriptor,
        calibration_session=descriptor.session_id,
        calibration_sha256=calibration_sha256,
        row_count=row_count,
        ordered_query_identity_sha256=prefix[:64],
        ordered_window_start_sha256=("a" * 64),
        ordered_target_evalmask_sha256=("b" * 64),
        held_before=held,
        held_after=held,
        reader_recipe_sha256=("c" * 64),
        external_versions={
            "falcon_challenge": "synthetic", "lightning": "synthetic", "numpy": "synthetic",
            "scipy": "synthetic", "torch": "synthetic",
        },
    )


def _material(descriptor: physical.SourceFileDescriptor, *, dominant_coordinate: int = 0) -> physical.SourceSessionMaterial:
    labels = _labels(descriptor.session_id, dominant_coordinate=dominant_coordinate)
    rows: dict[int, physical.UnassignedSourceM1Row] = {}
    calibration = np.full(
        plan.M1_CALIBRATION_SHAPE_PER_ROW,
        float(10_000 * (plan.HELD_IN_SOURCE_SESSIONS.index(descriptor.session_id) + 1)),
        dtype=np.float32,
    )
    calibration.setflags(write=False)
    calibration_sha256 = core.array_digest(calibration)
    for index in labels.valid_indices.tolist():
        # Native FalconDataset returns the same sealed chronological M10
        # calibration content for each query window of one session.  Every row
        # carries its own immutable header/view and session/digest binding;
        # the physical B32 route explicitly stacks those rows later.
        rows[index] = physical.UnassignedSourceM1Row(
            session_id=descriptor.session_id,
            sample_index=index,
            sample_id=f"{descriptor.session_id}:source:{index}",
            calibration_session=descriptor.session_id,
            calibration_sha256=calibration_sha256,
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
        valid_source_windows=64,
        calibration_session=descriptor.session_id,
        calibration_sha256=calibration_sha256,
        calibration_backing=calibration,
        native_evidence=_native_evidence(descriptor, calibration_sha256=calibration_sha256, row_count=len(rows)),
    )


class _SyntheticReader:
    def __init__(self, materials: Mapping[str, physical.SourceSessionMaterial]) -> None:
        self.materials = dict(materials)
        self.events: list[str] = []

    def read_source_session(self, descriptor: physical.SourceFileDescriptor) -> physical.SourceSessionMaterial:
        self.events.append(descriptor.session_id)
        return self.materials[descriptor.session_id]


@pytest.fixture(scope="module")
def smoke_prepared() -> tuple[physical.FrozenM1SourceManifest, physical.PreparedSourceFold]:
    manifest = _manifest()
    spec = lifecycle.source_smoke_spec()
    reader = _SyntheticReader({session: _material(manifest.files_by_session[session]) for session in spec.stage0_spec.source_sessions})
    provider = physical.StrictM1SourceProvider(manifest, reader)
    prepared = provider.prepare(spec)
    assert reader.events == list(spec.stage0_spec.source_sessions)
    return manifest, prepared


def _fake_device() -> lifecycle.DeviceProfile:
    return lifecycle.DeviceProfile(
        cuda_visible_devices="root-selected-single-device",
        torch_device="cuda:0",
        uuid="GPU-synthetic-reviewed-profile",
        pci_bus_id="00000000:ff:00.0",
        name="synthetic-only-no-cuda-profile",
        compute_capability=(8, 0),
        total_memory_bytes=1,
        torch_version="synthetic",
        cuda_version="synthetic",
        cudnn_version=1,
    )


def _device_environment(profile: lifecycle.DeviceProfile) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": profile.cuda_visible_devices,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        **lifecycle.THREAD_ENVIRONMENT,
    }


def _stage_lifecycle_closure(tmp_path: Path) -> Path:
    """Copy only closure-bound source leaves into an isolated temporary tree."""
    closure = lifecycle.implementation_closure(ROOT)
    for row in closure["paths"]:
        relative = str(row["path"])
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    (tmp_path / "tfpd_exploration/results").mkdir(parents=True, exist_ok=True)
    return tmp_path


class _LifecycleBackend:
    def __init__(
        self,
        *,
        prepared: physical.PreparedSourceFold,
        root: Path,
        fail_at: str | None = None,
    ) -> None:
        self.prepared = prepared
        self.root = root
        self.fail_at = fail_at
        self.calls: list[str] = []
        self._progress = lifecycle.LifecycleProgress()

    def _attempt_path(self) -> Path:
        return self.root / self.prepared.spec.root_relative / "attempt.json"

    def launch_payload(self, identity: lifecycle.SourceExecutionIdentity) -> Mapping[str, object]:
        assert identity.spec == self.prepared.spec
        assert self._attempt_path().is_file(), "attempt must precede backend/source/model work"
        self.calls.extend(("attempt_seen", "launch"))
        if self.fail_at == "launch":
            raise RuntimeError("synthetic launch failure")
        return {
            "schema": "synthetic_cswg_backend_launch",
            "source_opened": False,
            "model_constructed": False,
            "cuda_initialized": False,
        }

    def prepare_source(self, identity: lifecycle.SourceExecutionIdentity) -> Mapping[str, object]:
        assert identity.spec == self.prepared.spec
        self.calls.append("prepare_source")
        self._progress = lifecycle.LifecycleProgress(source_resolved_or_opened=True)
        if self.fail_at == "prepare":
            raise RuntimeError("synthetic source failure")
        return self.prepared.authority_fragment()

    def run_smoke(self, identity: lifecycle.SourceExecutionIdentity) -> Mapping[str, object]:
        assert identity.spec == self.prepared.spec
        self.calls.append("run_smoke")
        self._progress = lifecycle.LifecycleProgress(
            source_resolved_or_opened=True,
            model_constructed=True,
            cuda_initialized=False,
            source_authority_published=True,
        )
        if self.fail_at == "smoke":
            raise RuntimeError("synthetic smoke failure")
        return {
            "schema": "cross_session_worst_group_m1_source_smoke_v1",
            "identity_sha256": identity.sha256,
            "optimizer_steps": lifecycle.SMOKE_STEPS,
            "total_windows_per_step": plan.TOTAL_BATCH_SIZE,
            "one_concatenated_forward_per_step": True,
            "calibration_shape_per_row": list(plan.M1_CALIBRATION_SHAPE_PER_ROW),
            "model_parameter_count": plan.M1_LIVE_PARAMETERS_AFTER_LAZY1024,
            "model_output_shape": [plan.TOTAL_BATCH_SIZE, plan.M1_WINDOW_SIZE, plan.M1_RAW_BEHAVIOR_OUTPUTS],
            "finite_objective": True,
            "finite_model": True,
            "finite_gradients": True,
            "gradient_coverage": {
                "trainable_parameter_count": 2,
                "trainable_parameter_names_sha256": "4" * 64,
                "observed_gradient_count": 2,
                "observed_gradient_names_sha256": "4" * 64,
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
            "initial_model_state_sha256": "1" * 64,
            "final_model_state_sha256": "2" * 64,
            "best_checkpoint_state_sha256": "3" * 64,
            "rng": {
                "schema": "cross_session_worst_group_m1_rng_policy_v1",
                "seed": 42,
                "domains": ["python", "numpy", "torch_cpu", "torch_cuda_selected"],
                "scheduler_uses_host_rng": False,
                "model_initialization_and_dynamic_dropout_seeded": True,
                "pre_run_state_digest": "5" * 64,
                "post_run_state_restored": True,
            },
            "resources": {
                "elapsed_seconds": 1.0,
                "steps_per_second": 100.0,
                "samples_per_second": 3200.0,
                "cuda_current_allocated_bytes": 0,
                "cuda_peak_allocated_bytes": 0,
                "cuda_current_reserved_bytes": 0,
                "cuda_peak_reserved_bytes": 0,
            },
            "model_constructed": True,
            "cuda_initialized": False,
            "source_only": True,
            "target_optimizer_backward_update": 0,
            **lifecycle.FORBIDDEN_SURFACE_FLAGS,
            "_checkpoint_bodies": {
                "best_source_train_loss": b"synthetic-best-source-checkpoint",
                "last": b"synthetic-last-source-checkpoint",
            },
        }

    def progress(self) -> lifecycle.LifecycleProgress:
        return self._progress

    def close(self) -> None:
        self.calls.append("close")


class _AttemptCheckingReader(_SyntheticReader):
    def __init__(self, materials: Mapping[str, physical.SourceSessionMaterial], *, attempt_path: Path) -> None:
        super().__init__(materials)
        self.attempt_path = attempt_path

    def read_source_session(self, descriptor: physical.SourceFileDescriptor) -> physical.SourceSessionMaterial:
        assert self.attempt_path.is_file(), "durable audit attempt must precede source resolution"
        return super().read_source_session(descriptor)


def test_separate_cpu_source_audit_lifecycle_is_attempt_first_and_leaves_gpu_smoke_root_prospective(
    tmp_path: Path,
    smoke_prepared: tuple[physical.FrozenM1SourceManifest, physical.PreparedSourceFold],
) -> None:
    manifest, prepared = smoke_prepared
    staged, identity, capability = _temporary_audit_identity(tmp_path)
    assert identity.spec.root_relative == lifecycle.SOURCE_AUDIT_ROOT_RELATIVE
    assert identity.spec.root_relative != lifecycle.SOURCE_SMOKE_ROOT_RELATIVE
    audit_root = staged / identity.spec.root_relative
    source_reader = _AttemptCheckingReader(
        {session: prepared.materials[session] for session in identity.spec.stage0_spec.source_sessions},
        attempt_path=audit_root / "attempt.json",
    )
    backend = physical.PhysicalCSWGSourceAuditBackend(physical.StrictM1SourceProvider(manifest, source_reader))
    result = lifecycle.execute_reviewed_source_audit(
        staged,
        identity=identity,
        capability=capability,
        backend=backend,
    )
    assert result.terminal_sha256 is not None and result.failure_sha256 is None
    assert source_reader.events == list(identity.spec.stage0_spec.source_sessions)
    assert not (staged / lifecycle.SOURCE_SMOKE_ROOT_RELATIVE).exists()
    audit = json_load(audit_root / "audit.json")
    terminal = json_load(audit_root / "terminal.json")
    assert audit["model_constructed"] is False and audit["cuda_initialized"] is False
    assert audit["optimizer_steps_completed"] == 0
    assert terminal["model_constructed"] is False and terminal["cuda_initialized"] is False
    assert terminal["optimizer_steps_completed"] == 0


def test_source_audit_common_stratum_failure_is_typed_before_model_or_cuda(tmp_path: Path) -> None:
    manifest = _manifest()
    identity_root, identity, capability = _temporary_audit_identity(tmp_path)
    sessions = identity.spec.stage0_spec.source_sessions
    reader = _SyntheticReader({
        sessions[0]: _material(manifest.files_by_session[sessions[0]], dominant_coordinate=0),
        sessions[1]: _material(manifest.files_by_session[sessions[1]], dominant_coordinate=1),
        sessions[2]: _material(manifest.files_by_session[sessions[2]], dominant_coordinate=2),
    })
    backend = physical.PhysicalCSWGSourceAuditBackend(physical.StrictM1SourceProvider(manifest, reader))
    result = lifecycle.execute_reviewed_source_audit(
        identity_root,
        identity=identity,
        capability=capability,
        backend=backend,
    )
    assert result.failure_sha256 is not None and result.terminal_sha256 is None
    failure = json_load(identity_root / identity.spec.root_relative / "failure.json")
    assert failure["progress"]["source_resolved_or_opened"] is True
    assert failure["progress"]["model_constructed"] is False
    assert failure["progress"]["cuda_initialized"] is False
    assert failure["progress"]["optimizer_steps_completed"] == 0


def _temporary_identity(tmp_path: Path) -> tuple[Path, lifecycle.SourceExecutionIdentity, lifecycle.SourceExecutionCapability, dict[str, str]]:
    staged = _stage_lifecycle_closure(tmp_path)
    profile = _fake_device()
    environment = _device_environment(profile)
    identity = lifecycle.build_identity(staged, spec=lifecycle.source_smoke_spec(), device=profile)
    # The original GPU smoke identity is intentionally non-issuable until a
    # successor binds a real immutable source-audit graph.  Synthetic
    # transactional tests still need to exercise the lifecycle after a
    # trusted in-process capability boundary, so they use the private test
    # seal directly rather than pretending the public issuer is GO.
    capability = lifecycle.SourceExecutionCapability(identity.sha256, lifecycle._ROOT_REVIEW_SEAL)
    return staged, identity, capability, environment


def _temporary_audit_identity(
    tmp_path: Path,
) -> tuple[Path, lifecycle.SourceAuditIdentity, lifecycle.SourceAuditCapability]:
    staged = _stage_lifecycle_closure(tmp_path)
    identity = lifecycle.build_source_audit_identity(staged)
    capability = lifecycle.issue_root_reviewed_source_audit_capability(
        staged,
        identity,
        review_seal=lifecycle._ROOT_REVIEW_SEAL,
    )
    return staged, identity, capability


def test_exact_workorder_stage0_closure_and_static_lifecycle_closure() -> None:
    assert lifecycle.WORKORDER_SHA256 == "ca0c3b754c602ec490e4f5b74c5bf85a93764184e6f9a489a10ec4eed892e043"
    assert lifecycle.ACCEPTED_STAGE0_CLOSURE_SHA256 == "dd1fc152d4f7f900d6707bcea46136e1bde7cf781ca2f99dc12991296be7bb51"
    closure = lifecycle.implementation_closure(ROOT)
    assert closure["accepted_stage0_closure_sha256"] == lifecycle.ACCEPTED_STAGE0_CLOSURE_SHA256
    paths = [row["path"] for row in closure["paths"]]
    for required in (
        "tfpd_exploration/src/cross_session_worst_group_v1/source_lifecycle.py",
        "tfpd_exploration/src/cross_session_worst_group_v1/source_physical.py",
        "tfpd_exploration/src/cross_session_worst_group_v1/source_reader.py",
        "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_v1.py",
        "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_reader_audit.py",
        "tfpd_exploration/tests/test_cross_session_worst_group_m1_source_v1.py",
        "tfpd_exploration/tests/test_cross_session_worst_group_m1_source_reader.py",
    ):
        assert required in paths
    assert len(closure["closure_sha256"]) == 64


def test_all_four_fold_topologies_include_route_owned_fold3_without_target_source() -> None:
    for target in plan.HELD_IN_SOURCE_SESSIONS:
        cswg, erm = lifecycle.build_fold_route_specs(target)
        assert cswg.stage0_spec.outer_target_session == erm.stage0_spec.outer_target_session == target
        assert target not in cswg.stage0_spec.source_sessions
        assert tuple(cswg.stage0_spec.source_sessions) == tuple(
            item for item in plan.HELD_IN_SOURCE_SESSIONS if item != target
        )
        assert cswg.stage0_spec.lambda_ == 1.0
        assert erm.stage0_spec.lambda_ == 0.0
        assert cswg.stage0_spec.tau == erm.stage0_spec.tau == 0.01
    fold3, _erm = lifecycle.build_fold_route_specs("20120928")
    selected = _manifest().select_exact_sources(fold3)
    assert tuple(item.session_id for item in selected) == ("20120924", "20120926", "20120927")


def test_deferred_descriptor_provider_constructs_source_free_and_resolves_only_selected_sources_in_prepare(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    spec = lifecycle.source_smoke_spec()
    resolution_events: list[tuple[str, ...]] = []

    def loader(sessions: tuple[str, ...]) -> tuple[physical.SourceFileDescriptor, ...]:
        resolution_events.append(sessions)
        return tuple(manifest.files_by_session[session] for session in sessions)

    deferred = physical.DeferredSelectedSourceDescriptorProvider(loader)
    metadata_authority = lifecycle.load_m1_metadata_manifest_authority(ROOT)
    metadata_rows = {row["session_id"]: row for row in metadata_authority["source_rows"]}
    sealed_resolution_events: list[tuple[str, ...]] = []

    def sealed_loader(sessions: tuple[str, ...]) -> tuple[physical.SourceFileDescriptor, ...]:
        sealed_resolution_events.append(sessions)
        return tuple(
            physical.SourceFileDescriptor(
                session_id=session,
                relative_path=str(metadata_rows[session]["relative_path"]),
                sha256=str(metadata_rows[session]["sha256"]),
                # Byte count is intentionally a post-attempt source fact.
                byte_count=1,
            )
            for session in sessions
        )

    sealed = physical.SealedMetadataBoundSourceDescriptorProvider(
        sealed_loader,
        metadata_authority=metadata_authority,
    )
    # The route-owned factories build their own concrete descriptor loader
    # from the descriptor-read metadata authority.  A launcher cannot inject
    # an arbitrary source mapping; construction receives only a lexical source
    # root and does no NWB descriptor lookup, body hash, byte count, or target
    # resolution before the durable attempt.
    absent_root = tmp_path / "not-opened-before-attempt"
    route_backend = physical.build_route_owned_source_audit_backend(
        root=ROOT,
        source_root=absent_root,
    )
    assert resolution_events == []
    assert sealed_resolution_events == []
    assert isinstance(route_backend.provider.manifest, physical.SealedMetadataBoundSourceDescriptorProvider)
    assert isinstance(route_backend.provider.manifest.loader, reader.DeferredHeldM1DescriptorLoader)
    assert route_backend.provider.manifest.resolution_events == []
    assert not absent_root.exists()
    smoke_backend = physical.build_route_owned_physical_backend(
        root=ROOT,
        source_root=absent_root,
    )
    assert resolution_events == []
    assert sealed_resolution_events == []
    assert isinstance(smoke_backend.provider.manifest, physical.SealedMetadataBoundSourceDescriptorProvider)
    assert isinstance(smoke_backend.provider.manifest.loader, reader.DeferredHeldM1DescriptorLoader)
    assert "descriptor_provider" not in inspect.signature(physical.build_route_owned_source_audit_backend).parameters
    assert "descriptor_provider" not in inspect.signature(physical.build_route_owned_physical_backend).parameters

    sealed_descriptors = tuple(
        physical.SourceFileDescriptor(
            session_id=session,
            relative_path=str(metadata_rows[session]["relative_path"]),
            sha256=str(metadata_rows[session]["sha256"]),
            byte_count=1,
        )
        for session in spec.stage0_spec.source_sessions
    )
    sealed_materials = {
        descriptor.session_id: _material(descriptor, dominant_coordinate=0)
        for descriptor in sealed_descriptors
    }
    sealed_reader = _SyntheticReader(sealed_materials)
    sealed_prepared = physical.StrictM1SourceProvider(sealed, sealed_reader).prepare(spec)
    assert sealed_resolution_events == [spec.stage0_spec.source_sessions]
    assert sealed_reader.events == list(spec.stage0_spec.source_sessions)
    assert tuple(item.relative_path for item in sealed_prepared.descriptors) == tuple(
        str(metadata_rows[session]["relative_path"]) for session in spec.stage0_spec.source_sessions
    )

    materials = {
        session: _material(manifest.files_by_session[session], dominant_coordinate=0)
        for session in spec.stage0_spec.source_sessions
    }
    synthetic_reader = _SyntheticReader(materials)
    provider = physical.StrictM1SourceProvider(deferred, synthetic_reader)
    prepared = provider.prepare(spec)
    assert resolution_events == [spec.stage0_spec.source_sessions]
    assert synthetic_reader.events == list(spec.stage0_spec.source_sessions)
    assert spec.stage0_spec.outer_target_session not in resolution_events[0]
    assert tuple(item.session_id for item in prepared.descriptors) == spec.stage0_spec.source_sessions
    assert deferred.payload()["resolves_hashes_selected_descriptors_only_inside_prepare_source"] is True
    assert sealed.payload()["metadata_authority"] == metadata_authority


def test_exact_metadata_only_manifest_is_identity_bound_and_body_or_row_drift_fails_before_capability(
    tmp_path: Path,
) -> None:
    authority = lifecycle.load_m1_metadata_manifest_authority(ROOT)
    assert authority["body_sha256"] == lifecycle.M1_METADATA_MANIFEST_SHA256
    assert [row["session_id"] for row in authority["source_rows"]] == list(plan.HELD_IN_SOURCE_SESSIONS)
    assert all("byte_count" not in row for row in authority["source_rows"])
    staged = _stage_lifecycle_closure(tmp_path)
    identity = lifecycle.build_source_audit_identity(staged)
    assert identity.payload()["m1_metadata_manifest"] == lifecycle.m1_metadata_manifest_binding_payload()
    manifest_path = staged / lifecycle.M1_METADATA_MANIFEST_RELATIVE
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8").replace("held-in-calib", "forged", 1),
                             encoding="utf-8")
    with pytest.raises(lifecycle.SourceLifecycleError, match="body SHA"):
        lifecycle.load_m1_metadata_manifest_authority(staged)
    forged_authority = dict(authority)
    forged_authority["source_rows"] = list(authority["source_rows"])
    forged_authority["source_rows"][0] = {
        **forged_authority["source_rows"][0], "relative_path": "SPINT-main/data/forged.nwb",
    }
    with pytest.raises(physical.SourcePhysicalError, match="sealed metadata descriptor provider authority"):
        physical.SealedMetadataBoundSourceDescriptorProvider(lambda _sessions: (), forged_authority)


def test_fixed_lambda_tau_no_search_interface_and_fold_scoped_full_roots() -> None:
    cswg, erm = lifecycle.build_fold_route_specs("20120924")
    assert cswg.root_relative != erm.root_relative
    assert cswg.root_relative != lifecycle.SOURCE_SMOKE_ROOT_RELATIVE
    with pytest.raises(lifecycle.SourceLifecycleError, match="lambda/tau"):
        lifecycle.SourceRouteSpec(replace(cswg.stage0_spec, lambda_=0.5), "full", cswg.root_relative)
    with pytest.raises(lifecycle.SourceLifecycleError, match="matched ERM"):
        lifecycle.SourceRouteSpec(replace(erm.stage0_spec, tau=0.02), "full", erm.root_relative)


def test_fold_parametric_audit_factory_preserves_legacy_fold0_payload_byte_semantics() -> None:
    """New outer-fold audits cannot rewrite the accepted historical audit spec."""
    legacy = lifecycle.source_audit_spec()
    assert lifecycle.sha256_bytes(
        lifecycle._json_bytes(legacy.payload()),  # noqa: SLF001 - frozen payload regression
    ) == "11929265a28ead3ec97669d11f8aff955f11bc0f21236eb19775670de7e93356"
    assert legacy.root_relative == lifecycle.SOURCE_AUDIT_ROOT_RELATIVE
    assert legacy.stage0_spec.outer_target_session == lifecycle.SOURCE_SMOKE_OUTER_TARGET

    fold26 = lifecycle.source_audit_spec_for_outer_target("20120926")
    assert fold26.run_kind == "audit"
    assert fold26.root_relative == lifecycle.fold_audit_root_relative("20120926")
    assert fold26.stage0_spec.outer_target_session == "20120926"
    assert fold26.stage0_spec.source_sessions == ("20120924", "20120927", "20120928")
    assert fold26.root_relative != legacy.root_relative

    cswg26, _erm26 = plan.build_outer_fold_specs(
        outer_target_session="20120926",
        initialization_seed=lifecycle.SEED,
        lambda_=lifecycle.CSWG_LAMBDA,
        tau=lifecycle.CSWG_TAU,
    )
    with pytest.raises(lifecycle.SourceLifecycleError, match="exact CPU-only fold root/spec"):
        lifecycle.SourceRouteSpec(cswg26, "audit", lifecycle.SOURCE_AUDIT_ROOT_RELATIVE)
    with pytest.raises(lifecycle.SourceLifecycleError, match="non-legacy"):
        lifecycle.fold_audit_root_relative(lifecycle.SOURCE_SMOKE_OUTER_TARGET)
    with pytest.raises(lifecycle.SourceLifecycleError, match="legacy fold"):
        lifecycle.source_audit_spec_for_outer_target(lifecycle.SOURCE_SMOKE_OUTER_TARGET)


def test_source_path_admission_and_session_calibration_ownership_fail_closed() -> None:
    with pytest.raises(physical.SourcePhysicalError, match="surface/topology"):
        physical.SourceFileDescriptor("20120924", "test/20120924.nwb", "a" * 64, 1)
    descriptor = _descriptor("20120924")
    with pytest.raises(physical.SourcePhysicalError, match="physical-view"):
        physical.UnassignedSourceM1Row(
            session_id=descriptor.session_id,
            sample_index=0,
            sample_id="bad-length",
            calibration_session=descriptor.session_id,
            calibration_sha256="a" * 64,
            model_inputs={
                "x": np.zeros((100, 64), dtype=np.float32),
                "calib_trialized_neural_features": np.zeros((10, 100, 64), dtype=np.float32),
            },
            raw_final_target=np.zeros((16,), dtype=np.float32),
        )
    labels = _labels(descriptor.session_id)
    calibration = np.zeros(plan.M1_CALIBRATION_SHAPE_PER_ROW, dtype=np.float32)
    calibration.setflags(write=False)
    calibration_sha256 = core.array_digest(calibration)
    rows = {
        index: physical.UnassignedSourceM1Row(
            descriptor.session_id, index, f"r-{index}", descriptor.session_id, calibration_sha256,
            {"x": np.zeros((100, 64), dtype=np.float32), "calib_trialized_neural_features": calibration.view()},
            labels.raw_final_outputs[index],
        )
        for index in labels.valid_indices.tolist()
    }
    material = physical.SourceSessionMaterial(
        descriptor, labels, rows, 64, descriptor.session_id, calibration_sha256, calibration,
        _native_evidence(descriptor, calibration_sha256=calibration_sha256, row_count=len(rows)),
    )
    assert material.payload()["all_row_calibration_sha256_match_session"] is True
    wrong_rows = dict(rows)
    with pytest.raises(physical.SourcePhysicalError, match="physical-view"):
        physical.UnassignedSourceM1Row(
            descriptor.session_id, 0, "cross-session", "20120926", calibration_sha256,
            {"x": np.zeros((100, 64), dtype=np.float32), "calib_trialized_neural_features": calibration.view()},
            labels.raw_final_outputs[0],
        )


def test_source_authority_cannot_accept_backend_override_of_protected_source_fields(
    tmp_path: Path,
    smoke_prepared: tuple[physical.FrozenM1SourceManifest, physical.PreparedSourceFold],
) -> None:
    _manifest_value, prepared = smoke_prepared
    _staged, identity, _capability, _environment = _temporary_identity(tmp_path)
    forged = dict(prepared.authority_fragment())
    forged["source_only"] = False
    with pytest.raises(lifecycle.SourceLifecycleError, match="overwrite protected"):
        lifecycle.source_authority_payload(identity, forged)
    forged = dict(prepared.authority_fragment())
    forged["metadata_manifest"] = {"body_sha256": "0" * 64}
    with pytest.raises(lifecycle.SourceLifecycleError, match="overwrite protected"):
        lifecycle.source_authority_payload(identity, forged)


def test_prepare_is_exact_three_source_only_and_missing_common_stratum_stops_pre_model() -> None:
    manifest = _manifest()
    spec = lifecycle.source_smoke_spec()
    source_sessions = spec.stage0_spec.source_sessions
    materials = {
        source_sessions[0]: _material(manifest.files_by_session[source_sessions[0]], dominant_coordinate=0),
        source_sessions[1]: _material(manifest.files_by_session[source_sessions[1]], dominant_coordinate=1),
        source_sessions[2]: _material(manifest.files_by_session[source_sessions[2]], dominant_coordinate=2),
    }
    reader = _SyntheticReader(materials)
    provider = physical.StrictM1SourceProvider(manifest, reader)
    with pytest.raises(physical.SourcePhysicalError, match="common source stratum topology absent"):
        provider.prepare(spec)
    assert reader.events == list(source_sessions)
    assert spec.stage0_spec.outer_target_session not in reader.events


def test_prepared_source_authority_and_paired_erm_step_count(smoke_prepared: tuple[physical.FrozenM1SourceManifest, physical.PreparedSourceFold]) -> None:
    _manifest_value, prepared = smoke_prepared
    assert prepared.paired_steps_per_epoch == 2 * 3
    assert lifecycle.paired_epoch_step_count(prepared.spec, {
        session: material.valid_source_windows for session, material in prepared.materials.items()
    }) == prepared.paired_steps_per_epoch
    authority = prepared.authority_fragment()
    assert authority["source_manifest_file_sessions"] == list(prepared.spec.stage0_spec.source_sessions)
    assert authority["source_label_authority"]["source_sessions"] == list(prepared.spec.stage0_spec.source_sessions)
    assert authority["common_stratum_count"] == 1
    assert authority["concat_compatibility"]["sessions"] == list(prepared.spec.stage0_spec.source_sessions)
    assert authority["concat_compatibility"]["input_specs"] == [
        {"key": "calib_trialized_neural_features", "dtype": "float32", "per_row_shape": [10, 1024, 64]},
        {"key": "x", "dtype": "float32", "per_row_shape": [100, 64]},
    ]


class _OneForwardM1:
    def __init__(self) -> None:
        import torch

        self.scale = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.calls = 0

    def __call__(self, *, x, calib_trialized_neural_features):
        import torch

        self.calls += 1
        assert x.shape == (32, 100, 64)
        assert calib_trialized_neural_features.shape == (32, 10, 1024, 64)
        signal = x.mean(dim=(1, 2), keepdim=True) * self.scale
        return signal.expand(-1, 100, 16)


def test_one_concatenated_b32_forward_rotating_quotas_current_losses_and_no_rng_mutation(
    smoke_prepared: tuple[physical.FrozenM1SourceManifest, physical.PreparedSourceFold],
) -> None:
    import torch

    _manifest_value, prepared = smoke_prepared
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    episode0 = prepared.episode(0)
    episode1 = prepared.episode(1)
    assert episode0.quota.counts == (10, 11, 11)
    assert episode1.quota.counts == (11, 10, 11)
    assert episode0.digest == prepared.episode(0).digest
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    graph = _OneForwardM1()
    step = core.RouteOwnedMixedSessionTrainingStep(
        model=graph,
        compatibility=prepared.compatibility,
        run_spec=prepared.spec.stage0_spec,
    )
    objective = step.run(episode0, lambda_=1.0, tau=0.01)
    assert graph.calls == 1
    assert objective.loss.requires_grad
    weights = torch.autograd.grad(objective.loss, [entry.value for entry in objective.session_losses], retain_graph=True)
    assert all(float(weight.detach()) >= 0.0 for weight in weights)
    objective.loss.backward()
    assert graph.scale.grad is not None and torch.isfinite(graph.scale.grad)


def test_exact_cpu_m1_adapter_inference_equivalence_and_one_adam_step(smoke_prepared: tuple[physical.FrozenM1SourceManifest, physical.PreparedSourceFold]) -> None:
    import torch

    assert torch.cuda.is_initialized() is False
    _manifest_value, prepared = smoke_prepared
    row = prepared.materials[prepared.spec.stage0_spec.source_sessions[0]].rows_by_sample_index[0]
    model = physical.load_exact_m1_spint_model(ROOT)
    materialized = physical.materialize_exact_m1_model(model)
    assert materialized == {
        "output_shape": [1, 100, 16],
        "live_parameter_count": 15_007_496,
        "fc_id_in_weight_shape": [1024, 1024],
        "device": "cpu",
    }
    adapter = physical.ExactM1ForwardAdapter(model)
    x = torch.as_tensor(np.array(row.model_inputs["x"], copy=True)).unsqueeze(0)
    calibration = torch.as_tensor(np.array(row.model_inputs["calib_trialized_neural_features"], copy=True)).unsqueeze(0)
    target = torch.as_tensor(np.array(row.raw_final_target, copy=True)).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        baseline = model(x, calib_trialized_neural_features=calibration)
        through_adapter = adapter(x=x, calib_trialized_neural_features=calibration)
    assert adapter.forward_calls == 1
    assert torch.equal(baseline, through_adapter)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-5, weight_decay=0.0)
    before = tuple(parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad)
    output = adapter(x=x, calib_trialized_neural_features=calibration)
    loss = ((output[:, -1, :] - target) ** 2).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters() if parameter.requires_grad)
    optimizer.step()
    after = tuple(parameter.detach() for parameter in model.parameters() if parameter.requires_grad)
    assert any(not torch.equal(left, right) for left, right in zip(before, after, strict=True))
    assert torch.cuda.is_initialized() is False
    del model, optimizer, before, after
    gc.collect()


def test_strict_checkpoint_reload_rechecks_both_state_bytes_and_rejects_one_tensor_drift() -> None:
    import io

    import torch

    model = torch.nn.Linear(3, 2)
    best_expected = physical._state_digest(model)
    best_buffer = io.BytesIO()
    torch.save(model.state_dict(), best_buffer)
    with torch.no_grad():
        model.weight[0, 0] += 0.25
    last_expected = physical._state_digest(model)
    last_buffer = io.BytesIO()
    torch.save(model.state_dict(), last_buffer)

    def factory() -> torch.nn.Module:
        return torch.nn.Linear(3, 2)

    assert physical.strict_reload_checkpoint_bytes(
        best_buffer.getvalue(), expected_state_sha256=best_expected, model_factory=factory, device="cpu",
    ) == best_expected
    assert physical.strict_reload_checkpoint_bytes(
        last_buffer.getvalue(), expected_state_sha256=last_expected, model_factory=factory, device="cpu",
    ) == last_expected
    tampered = torch.load(io.BytesIO(last_buffer.getvalue()), map_location="cpu", weights_only=True)
    assert isinstance(tampered, Mapping)
    tampered["weight"] = tampered["weight"].clone()
    tampered["weight"][0, 0] += 1.0
    tampered_buffer = io.BytesIO()
    torch.save(tampered, tampered_buffer)
    with pytest.raises(physical.SourcePhysicalError, match="checkpoint reload state digest drift"):
        physical.strict_reload_checkpoint_bytes(
            tampered_buffer.getvalue(), expected_state_sha256=last_expected, model_factory=factory, device="cpu",
        )
    assert torch.cuda.is_initialized() is False


def test_dynamic_device_profile_has_no_fixed_ordinal_and_no_cuda_probe() -> None:
    import torch

    profile = _fake_device()
    lifecycle.validate_device_environment(profile, _device_environment(profile))
    wrong = _device_environment(profile)
    wrong["OMP_NUM_THREADS"] = "2"
    with pytest.raises(lifecycle.SourceLifecycleError, match="environment"):
        lifecycle.validate_device_environment(profile, wrong)
    assert torch.cuda.is_initialized() is False


class _FakeSelectedCudaRng:
    """CPU-only stand-in for the selected CUDA RNG state; it never initializes CUDA."""

    def __init__(self) -> None:
        import torch

        self.state = torch.arange(16, dtype=torch.uint8)

    def get_rng_state(self, device: int) -> object:
        assert device == 0
        return self.state.clone()

    def set_rng_state(self, state: object, *, device: int) -> None:
        assert device == 0
        self.state = state.clone()

    def manual_seed_all(self, seed: int) -> None:
        import torch

        self.state = torch.full((16,), seed % 256, dtype=torch.uint8)


class _FakeTorchRng:
    def __init__(self) -> None:
        import torch

        self._torch = torch
        self.cuda = _FakeSelectedCudaRng()

    def get_rng_state(self) -> object:
        return self._torch.get_rng_state()

    def set_rng_state(self, state: object) -> None:
        self._torch.set_rng_state(state)

    def manual_seed(self, seed: int) -> None:
        self._torch.manual_seed(seed)


def test_seed42_rng_snapshot_restores_host_domains_and_repeats_initialization_law() -> None:
    import torch

    fake = _FakeTorchRng()
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_cpu = torch.get_rng_state().clone()
    before_cuda = fake.cuda.get_rng_state(0)

    def seeded_probe() -> tuple[float, float, object]:
        snapshot = physical.ProcessRngSnapshot.capture_and_seed(fake, seed=42)
        values = (random.random(), float(np.random.random()), torch.rand((5,), dtype=torch.float32))
        assert snapshot.restore() is True
        return values

    first = seeded_probe()
    second = seeded_probe()
    assert first[0] == second[0] and first[1] == second[1] and torch.equal(first[2], second[2])
    assert random.getstate() == before_python
    assert np.array_equal(np.random.get_state()[1], before_numpy[1])
    assert torch.equal(torch.get_rng_state(), before_cpu)
    assert torch.equal(fake.cuda.get_rng_state(0), before_cuda)
    assert torch.cuda.is_initialized() is False


def test_resource_and_final_model_state_validators_reject_nonfinite_values_without_cuda() -> None:
    import torch

    valid = {
        "elapsed_seconds": 1.0,
        "steps_per_second": 1.0,
        "samples_per_second": 32.0,
        "cuda_current_allocated_bytes": 0,
        "cuda_peak_allocated_bytes": 0,
        "cuda_current_reserved_bytes": 0,
        "cuda_peak_reserved_bytes": 0,
    }
    lifecycle._validate_resources(valid)
    for field, value in (("elapsed_seconds", float("inf")), ("steps_per_second", float("nan"))):
        forged = dict(valid)
        forged[field] = value
        with pytest.raises(lifecycle.SourceLifecycleError, match="finite"):
            lifecycle._validate_resources(forged)
    forged_memory = dict(valid)
    forged_memory["cuda_current_allocated_bytes"] = -1
    with pytest.raises(lifecycle.SourceLifecycleError, match="finite"):
        lifecycle._validate_resources(forged_memory)
    model = torch.nn.Linear(2, 2)
    physical._state_digest(model)
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    with pytest.raises(physical.SourcePhysicalError, match="nonfinite"):
        physical._state_digest(model)
    adam_model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(adam_model.parameters(), lr=1.0e-3)
    loss = adam_model(torch.ones((1, 1), dtype=torch.float32)).sum()
    loss.backward()
    optimizer.step()
    assert physical.TorchCSWGSmokeRunner._finite_adam_state(optimizer) is True
    with torch.no_grad():
        next(iter(optimizer.state.values()))["exp_avg"].fill_(float("inf"))
    assert physical.TorchCSWGSmokeRunner._finite_adam_state(optimizer) is False
    assert torch.cuda.is_initialized() is False


def test_smoke_gradient_coverage_receipt_rejects_missing_or_mismatched_name_digest(
    tmp_path: Path,
    smoke_prepared: tuple[physical.FrozenM1SourceManifest, physical.PreparedSourceFold],
) -> None:
    _manifest_value, prepared = smoke_prepared
    staged, identity, _capability, _environment = _temporary_identity(tmp_path)
    payload = dict(_LifecycleBackend(prepared=prepared, root=staged).run_smoke(identity))
    payload.pop("_checkpoint_bodies")
    lifecycle._validate_smoke_result(payload, identity)
    forged = dict(payload)
    coverage = dict(payload["gradient_coverage"])
    coverage["observed_gradient_count"] = 1
    forged["gradient_coverage"] = coverage
    with pytest.raises(lifecycle.SourceLifecycleError, match="gradient coverage"):
        lifecycle._validate_smoke_result(forged, identity)
    coverage = dict(payload["gradient_coverage"])
    coverage["observed_gradient_names_sha256"] = "f" * 64
    forged["gradient_coverage"] = coverage
    with pytest.raises(lifecycle.SourceLifecycleError, match="gradient coverage"):
        lifecycle._validate_smoke_result(forged, identity)


def test_default_physical_factory_has_real_deferred_torch_runner_without_opening_source_or_cuda() -> None:
    import torch

    manifest = _manifest()
    backend = physical.build_default_physical_backend(
        root=ROOT,
        manifest=manifest,
        reader=_SyntheticReader({}),
    )
    assert isinstance(backend, physical.PhysicalCSWGSourceBackend)
    assert isinstance(backend.smoke_runner, physical.TorchCSWGSmokeRunner)
    assert backend.provider.read_events == []
    assert torch.cuda.is_initialized() is False


def test_route_owned_source_audit_factory_is_deferred_and_has_no_model_or_cuda_runner() -> None:
    import torch

    backend = physical.build_route_owned_source_audit_backend(
        root=ROOT,
        source_root=Path("/synthetic-not-opened"),
    )
    assert isinstance(backend, physical.PhysicalCSWGSourceAuditBackend)
    assert backend.provider.read_events == []
    assert isinstance(backend.provider.manifest, physical.SealedMetadataBoundSourceDescriptorProvider)
    assert isinstance(backend.provider.manifest.loader, reader.DeferredHeldM1DescriptorLoader)
    assert backend.provider.manifest.resolution_events == []
    assert torch.cuda.is_initialized() is False


class _FailingProgressRunner:
    def run(self, *, identity, prepared):  # type: ignore[no-untyped-def]
        raise RuntimeError("synthetic post-cuda step failure")

    def progress(self) -> lifecycle.LifecycleProgress:
        return lifecycle.LifecycleProgress(
            source_resolved_or_opened=True,
            model_constructed=True,
            cuda_initialized=True,
            optimizer_steps_completed=73,
            source_authority_published=True,
        )

    def close(self) -> None:
        return None


@dataclass
class _BoundaryFailingRunner:
    reported: lifecycle.LifecycleProgress

    def run(self, *, identity, prepared):  # type: ignore[no-untyped-def]
        raise RuntimeError("synthetic boundary failure")

    def progress(self) -> lifecycle.LifecycleProgress:
        return self.reported

    def close(self) -> None:
        return None


def test_physical_backend_preserves_runner_failure_progress_without_cuda_probe(
    smoke_prepared: tuple[physical.FrozenM1SourceManifest, physical.PreparedSourceFold],
) -> None:
    import torch

    manifest, prepared = smoke_prepared
    reader = _SyntheticReader({session: prepared.materials[session] for session in prepared.spec.stage0_spec.source_sessions})
    backend = physical.build_physical_backend(manifest=manifest, reader=reader, smoke_runner=_FailingProgressRunner())
    backend.prepare_source(_temporary_identity_for_spec(prepared.spec))
    with pytest.raises(RuntimeError, match="step failure"):
        backend.run_smoke(_temporary_identity_for_spec(prepared.spec))
    assert backend.progress().optimizer_steps_completed == 73
    assert backend.progress().cuda_initialized is True
    assert torch.cuda.is_initialized() is False


@pytest.mark.parametrize(
    "failure_stage,expected",
    [
        ("prepare", lifecycle.LifecycleProgress(source_resolved_or_opened=True)),
        ("cuda", lifecycle.LifecycleProgress(
            source_resolved_or_opened=True, cuda_initialized=True, source_authority_published=True,
        )),
        ("model", lifecycle.LifecycleProgress(
            source_resolved_or_opened=True, model_constructed=True, cuda_initialized=True,
            source_authority_published=True,
        )),
    ],
)
def test_failure_receipts_preserve_exact_source_cuda_and_model_boundaries(
    tmp_path: Path,
    smoke_prepared: tuple[physical.FrozenM1SourceManifest, physical.PreparedSourceFold],
    failure_stage: str,
    expected: lifecycle.LifecycleProgress,
) -> None:
    manifest, prepared = smoke_prepared
    staged, identity, capability, environment = _temporary_identity(tmp_path)
    if failure_stage == "prepare":
        backend: lifecycle.DeferredSourceBackend = _LifecycleBackend(
            prepared=prepared, root=staged, fail_at="prepare",
        )
    else:
        source_reader = _SyntheticReader({
            session: prepared.materials[session] for session in prepared.spec.stage0_spec.source_sessions
        })
        backend = physical.build_physical_backend(
            manifest=manifest,
            reader=source_reader,
            smoke_runner=_BoundaryFailingRunner(expected),
        )
    result = lifecycle.execute_reviewed_source_smoke(
        staged, identity=identity, capability=capability, backend=backend, environ=environment,
    )
    assert result.failure_sha256 is not None and result.terminal_sha256 is None
    progress = json_load(staged / identity.spec.root_relative / "failure.json")["progress"]
    assert progress == expected.payload()


def _temporary_identity_for_spec(spec: lifecycle.SourceRouteSpec) -> lifecycle.SourceExecutionIdentity:
    # Physical backend methods bind the spec; closure/device values are not
    # touched before the real lifecycle owns authorization and CUDA.
    return lifecycle.SourceExecutionIdentity(
        spec=spec,
        closure={"closure_sha256": "e" * 64},
        device=_fake_device(),
    )


def test_source_authority_cross_bindings_reject_forged_numeric_or_label_fields(
    tmp_path: Path,
    smoke_prepared: tuple[physical.FrozenM1SourceManifest, physical.PreparedSourceFold],
) -> None:
    _manifest_value, prepared = smoke_prepared
    staged, identity, _capability, _environment = _temporary_identity(tmp_path)
    authority = lifecycle.source_authority_payload(identity, prepared.authority_fragment())
    lifecycle._validate_source_authority(authority, identity)
    changed_windows = dict(authority)
    changed_windows["valid_source_windows"] = dict(authority["valid_source_windows"])
    changed_windows["valid_source_windows"][prepared.spec.stage0_spec.source_sessions[0]] += plan.TOTAL_BATCH_SIZE
    with pytest.raises(lifecycle.SourceLifecycleError, match="paired-step"):
        lifecycle._validate_source_authority(changed_windows, identity)
    changed_label = dict(authority)
    changed_label["source_session_materials"] = [dict(item) for item in authority["source_session_materials"]]
    changed_label["source_session_materials"][0]["source_label_digest"] = "f" * 64
    with pytest.raises(lifecycle.SourceLifecycleError, match="row/calibration"):
        lifecycle._validate_source_authority(changed_label, identity)
    assert staged.is_dir()


def test_immutable_artifact_pair_rejects_hardlink_and_sidecar_substitution(tmp_path: Path) -> None:
    (tmp_path / "tfpd_exploration/results").mkdir(parents=True)
    artifact = lifecycle.ImmutableArtifactRoot.reserve(tmp_path, lifecycle.source_smoke_spec())
    try:
        digest = artifact.publish_json("attempt.json", {"value": "fixed"})
        assert artifact.read_json_pair("attempt.json", expected_sha256=digest) == {"value": "fixed"}
        # A hard-link alias turns the held leaf into a mutable alias even when
        # the bytes/mode are unchanged, so descriptor reload must reject it.
        os.link(artifact.path / "attempt.json", tmp_path / "alias.json")
        with pytest.raises(lifecycle.SourceLifecycleError, match="leaf mode/type"):
            artifact.read_json_pair("attempt.json", expected_sha256=digest)
        os.unlink(tmp_path / "alias.json")
        sidecar = artifact.path / "attempt.json.sha256"
        os.chmod(sidecar, 0o644)
        sidecar.write_text("0" * 64 + "  attempt.json\n", encoding="ascii")
        os.chmod(sidecar, 0o444)
        with pytest.raises(lifecycle.SourceLifecycleError, match="sidecar"):
            artifact.read_json_pair("attempt.json", expected_sha256=digest)
    finally:
        artifact.close()


def test_final_revalidation_rejects_replaced_valid_pair_after_publication(
    tmp_path: Path,
    smoke_prepared: tuple[physical.FrozenM1SourceManifest, physical.PreparedSourceFold],
) -> None:
    _manifest_value, prepared = smoke_prepared
    staged, identity, _capability, _environment = _temporary_identity(tmp_path)
    artifact = lifecycle.ImmutableArtifactRoot.reserve(staged, identity.spec)
    try:
        backend = _LifecycleBackend(prepared=prepared, root=staged)
        attempt_sha = artifact.publish_json("attempt.json", lifecycle._attempt_payload(identity))
        launch_sha = artifact.publish_json(
            "launch.json", lifecycle._launch_payload(identity, attempt_sha, backend.launch_payload(identity)),
        )
        authority = lifecycle.source_authority_payload(identity, backend.prepare_source(identity))
        authority_sha = artifact.publish_json("source_authority.json", authority)
        smoke = dict(backend.run_smoke(identity))
        smoke["source_authority_sha256"] = authority_sha
        bodies = smoke.pop("_checkpoint_bodies")
        lifecycle._validate_smoke_result(smoke, identity)
        smoke_sha = artifact.publish_json("smoke.json", smoke)
        manifest, manifest_sha = lifecycle._publish_smoke_checkpoints(artifact, identity, bodies, smoke=smoke)
        assert manifest["checkpoints"]["best_source_train_loss"]["state_sha256"] == smoke[
            "best_checkpoint_state_sha256"
        ]
        assert manifest["checkpoints"]["last"]["state_sha256"] == smoke["final_model_state_sha256"]
        # Replace launch body and canonical sidecar together.  Self-consistency
        # alone would pass, but final revalidation must retain its publication
        # digest binding and reject before terminal.
        body_path = artifact.path / "launch.json"
        sidecar_path = artifact.path / "launch.json.sha256"
        replacement = lifecycle._json_bytes({"schema": "forged"})
        replacement_sha = lifecycle.sha256_bytes(replacement)
        os.chmod(body_path, 0o644)
        os.chmod(sidecar_path, 0o644)
        body_path.write_bytes(replacement)
        sidecar_path.write_text(f"{replacement_sha}  launch.json\n", encoding="ascii")
        os.chmod(body_path, 0o444)
        os.chmod(sidecar_path, 0o444)
        with pytest.raises(lifecycle.SourceLifecycleError, match="body digest"):
            lifecycle._revalidate_published_smoke_graph(
                artifact,
                identity,
                attempt_sha256=attempt_sha,
                launch_sha256=launch_sha,
                authority_sha256=authority_sha,
                smoke_sha256=smoke_sha,
                checkpoint_manifest_sha256=manifest_sha,
            )
    finally:
        artifact.close()


def test_lifecycle_success_is_attempt_before_source_and_terminal_pairs_are_immutable(
    tmp_path: Path,
    smoke_prepared: tuple[physical.FrozenM1SourceManifest, physical.PreparedSourceFold],
) -> None:
    _manifest_value, prepared = smoke_prepared
    staged, identity, capability, environment = _temporary_identity(tmp_path)
    backend = _LifecycleBackend(prepared=prepared, root=staged)
    result = lifecycle.execute_reviewed_source_smoke(
        staged, identity=identity, capability=capability, backend=backend, environ=environment,
    )
    assert result.terminal_sha256 is not None and result.failure_sha256 is None
    assert backend.calls[:4] == ["attempt_seen", "launch", "prepare_source", "run_smoke"]
    artifact_root = staged / identity.spec.root_relative
    expected = (
        "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
        "source_authority.json", "source_authority.json.sha256", "smoke.json", "smoke.json.sha256",
        "checkpoint_best_source_train_loss.pt", "checkpoint_best_source_train_loss.pt.sha256",
        "checkpoint_last.pt", "checkpoint_last.pt.sha256",
        "checkpoint_manifest.json", "checkpoint_manifest.json.sha256", "terminal.json", "terminal.json.sha256",
    )
    assert tuple(sorted(path.name for path in artifact_root.iterdir())) == tuple(sorted(expected))
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in artifact_root.iterdir())
    terminal = json_load(artifact_root / "terminal.json")
    assert terminal["attempt_sha256"] == result.attempt_sha256
    assert terminal["source_authority_sha256"] == result.source_authority_sha256
    assert terminal["launch_closure_sha256"] == terminal["final_closure_sha256"] == identity.closure["closure_sha256"]


def json_load(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("fail_at,has_authority", [("launch", False), ("smoke", True)])
def test_lifecycle_failure_receipts_are_honest_before_and_after_authority(
    tmp_path: Path,
    smoke_prepared: tuple[physical.FrozenM1SourceManifest, physical.PreparedSourceFold],
    fail_at: str,
    has_authority: bool,
) -> None:
    _manifest_value, prepared = smoke_prepared
    staged, identity, capability, environment = _temporary_identity(tmp_path)
    backend = _LifecycleBackend(prepared=prepared, root=staged, fail_at=fail_at)
    result = lifecycle.execute_reviewed_source_smoke(
        staged, identity=identity, capability=capability, backend=backend, environ=environment,
    )
    assert result.terminal_sha256 is None and result.failure_sha256 is not None
    artifact_root = staged / identity.spec.root_relative
    assert (artifact_root / "failure.json").is_file()
    failure = json_load(artifact_root / "failure.json")
    assert failure["attempt_sha256"] == result.attempt_sha256
    assert (failure["source_authority_sha256"] is not None) is has_authority
    assert failure["progress"]["source_resolved_or_opened"] is has_authority
    assert failure["progress"]["target_optimizer_backward_update"] == 0
    assert not (artifact_root / "terminal.json").exists()


def test_lifecycle_closure_root_and_sidecar_adversaries_fail_closed(tmp_path: Path) -> None:
    staged = _stage_lifecycle_closure(tmp_path)
    # A closure-bound byte change is caught before any root reservation.
    core_copy = staged / "tfpd_exploration/src/cross_session_worst_group_v1/core.py"
    core_copy.write_text(core_copy.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
    with pytest.raises(lifecycle.SourceLifecycleError, match="Stage-0 closure drift"):
        lifecycle.implementation_closure(staged)
    # A named fresh root cannot be represented by a symlink or pre-existing directory.
    root = tmp_path / "separate"
    (root / "tfpd_exploration/results").mkdir(parents=True)
    target = root / "target"
    target.mkdir()
    candidate = root / lifecycle.SOURCE_SMOKE_ROOT_RELATIVE
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.symlink_to(target, target_is_directory=True)
    with pytest.raises(lifecycle.SourceLifecycleError, match="already exists"):
        lifecycle.assert_prospective_root_fresh(root, lifecycle.source_smoke_spec())


def test_static_cli_is_stdlib_only_and_public_execution_flags_fail_closed(tmp_path: Path) -> None:
    script = ROOT / "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_v1.py"
    environment = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    probe = subprocess.run(
        [sys.executable, "-c", (
            "import runpy,sys; "
            f"runpy.run_path({str(script)!r}, run_name='not_main'); "
            "print('torch' in sys.modules)"
        )],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert probe.stdout.strip() == "False"
    dry = subprocess.run([sys.executable, str(script), "--dry-run"], cwd=tmp_path, env=environment,
                         text=True, capture_output=True, check=True)
    payload = json_load_from_text(dry.stdout)
    assert payload["opens_source_or_target"] is False and payload["initializes_cuda"] is False
    denied = subprocess.run([sys.executable, str(script), "--execute"], cwd=tmp_path, env=environment,
                            text=True, capture_output=True)
    assert denied.returncode != 0
    assert "root-reviewed" in denied.stderr


def test_current_smoke_capability_is_explicitly_blocked_pending_a_successor_bound_to_source_audit(
    tmp_path: Path,
) -> None:
    staged = _stage_lifecycle_closure(tmp_path)
    profile = _fake_device()
    environment = _device_environment(profile)
    identity = lifecycle.build_identity(staged, spec=lifecycle.source_smoke_spec(), device=profile)
    with pytest.raises(lifecycle.SourceLifecycleError, match="source-audit terminal/authority graph"):
        lifecycle.issue_root_reviewed_capability(
            staged,
            identity,
            environ=environment,
            review_seal=lifecycle._ROOT_REVIEW_SEAL,
        )
    payload = lifecycle.dry_plan(staged)
    assert payload["current_gpu_smoke_capability_issuable"] is False
    assert payload["future_gpu_smoke_requires_successor_identity"] is True
    assert "source-audit terminal" in payload["future_gpu_smoke_required_predecessor"]
    assert payload["source_descriptor_resolution"]["current_closure_bound_metadata_only_manifest_available"] is True
    assert payload["source_descriptor_resolution"]["route_owned_factory_requires"] == "route-owned DeferredHeldM1DescriptorLoader wrapped by sealed metadata-bound descriptor provider"
    assert "constructs no source descriptor" in payload["source_descriptor_resolution"]["construction"]
    assert payload["source_descriptor_resolution"]["metadata_manifest"] == lifecycle.m1_metadata_manifest_binding_payload()


def test_deferred_physical_module_imports_no_torch_or_cuda(tmp_path: Path) -> None:
    environment = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": str(ROOT),
    }
    probe = subprocess.run(
        [sys.executable, "-c", (
            "import sys; "
            "import tfpd_exploration.src.cross_session_worst_group_v1.source_physical; "
            "print('torch' in sys.modules)"
        )],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert probe.stdout.strip() == "False"


def json_load_from_text(value: str) -> dict[str, Any]:
    import json

    loaded = json.loads(value)
    assert isinstance(loaded, dict)
    return loaded
