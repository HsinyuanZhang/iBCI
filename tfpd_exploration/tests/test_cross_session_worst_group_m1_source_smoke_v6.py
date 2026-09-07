"""Focused no-data/no-CUDA tests for the CS-WG M1 V6 audit-spec successor."""
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
from tfpd_exploration.src.cross_session_worst_group_v1 import source_smoke_physical_v4 as v4_physical  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_smoke_v4 as v4  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_smoke_physical_v6 as physical_v6  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_smoke_v5 as v5  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_smoke_v6 as v6  # noqa: E402


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_pair(directory: Path, name: str, payload: Mapping[str, object]) -> str:
    body = _json_bytes(dict(payload))
    digest = _sha(body)
    (directory / name).write_bytes(body)
    (directory / f"{name}.sha256").write_bytes(f"{digest}  {name}\n".encode("ascii"))
    os.chmod(directory / name, 0o444)
    os.chmod(directory / f"{name}.sha256", 0o444)
    return digest


def _stage_v6_closure(tmp_path: Path) -> Path:
    staged = tmp_path / "stage"
    closure = v6.implementation_closure(ROOT)
    for row in closure["paths"]:
        relative = str(row["path"])
        target = staged / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    (staged / "tfpd_exploration/results").mkdir(parents=True, exist_ok=True)
    return staged


def _device() -> v1.DeviceProfile:
    return v1.DeviceProfile(
        cuda_visible_devices="7",
        torch_device="cuda:0",
        uuid="GPU-synthetic-v6",
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
            row = np.zeros((plan.M1_RAW_BEHAVIOR_OUTPUTS,), dtype=np.float32)
            row[coordinate] = float(coordinate + 1)
            rows.append(row)
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
def prepared() -> Any:
    audit_spec = v1.source_audit_spec()
    descriptors = tuple(_descriptor(session) for session in audit_spec.stage0_spec.source_sessions)
    materials = {item.session_id: v4_physical._cache_material(_material(item)) for item in descriptors}
    audit = v3.build_common_stratum_prepared_audit(
        physical_module=physical, spec=audit_spec, descriptors=descriptors, materials=materials,
    )
    return v4_physical.rebind_v3_audit_prepared_to_smoke(physical_module=physical, audit_prepared=audit)


def _historical_v3_identity_payload() -> dict[str, object]:
    inherited_v1 = {
        "schema": "cross_session_worst_group_m1_source_audit_identity_v1",
        "spec": v1.source_audit_spec().payload(),
    }
    return {
        "schema": "cross_session_worst_group_m1_source_audit_identity_v3",
        "inherited_v2_identity": {"inherited_v1_identity": inherited_v1},
        "closure": {"closure_sha256": v5.V3_CLOSURE_SHA256},
    }


def _fake_graphs(prepared_value: Any) -> tuple[v5.HeldHistoricalGraph, v5.HeldHistoricalGraph]:
    fallback, fallback_sha, step_zero = v6._prepared_fallback(prepared_value)
    v3_graph = v5.HeldHistoricalGraph(
        expectation=v6.V3_COMPLETED_EXPECTATION,
        root_identity=(101, 102), named_chain_identities=((".", 101, 102),),
        bodies={
            "attempt.json": {"identity": _historical_v3_identity_payload()},
            "launch.json": {"synthetic": "launch"},
            "source_authority.json": {
                "deterministic_common_stratum_fallback": fallback,
                "deterministic_common_stratum_fallback_sha256": fallback_sha,
                "common_stratum_step_zero_evidence": step_zero,
            },
            "audit.json": {"synthetic": "audit"},
            "terminal.json": {"synthetic": "terminal"},
        },
    )
    v5_graph = v5.HeldHistoricalGraph(
        expectation=v6.V5_FAILED_EXPECTATION,
        root_identity=(103, 104), named_chain_identities=((".", 103, 104),),
        bodies={
            "attempt.json": {"synthetic": "attempt"},
            "launch.json": {"synthetic": "launch"},
            "failure.json": {"synthetic": "failure"},
        },
    )
    return v3_graph, v5_graph


def _passed_derivative_evidence(sessions: tuple[str, str, str]) -> dict[str, object]:
    collector = v5.DerivativeEvidenceCollector(sessions)
    for step in range(v1.SMOKE_STEPS):
        collector.observe({
            "schema": "cross_session_worst_group_m1_derivative_observation_v1",
            "step_index": step,
            "session_ids": sessions,
            "session_loss_values_fp32": (1.0, 1.0, 1.0),
            "raw_autograd_weight_values_fp32": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            "tensor_count": 0,
            "graph_retained": False,
        })
    return collector.validate(expected_steps=v1.SMOKE_STEPS)


def _valid_inner_smoke(identity: v1.SourceExecutionIdentity) -> dict[str, object]:
    names_digest = _sha(_json_bytes(["weight"]))
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
            "trainable_parameter_names_sha256": names_digest,
            "observed_gradient_count": 1,
            "observed_gradient_names_sha256": names_digest,
            "missing_trainable_names": [],
            "excluded_trainable_names": [],
        },
        "finite_adam_state": True,
        "model_state_changed": True,
        "checkpoint_reload_strict": True,
        "best_checkpoint_reload_strict": True,
        "last_checkpoint_reload_strict": True,
        "session_objective_derivatives_nonnegative": False,
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
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


class _MockBackend:
    def __init__(self, *, staged: Path, prepared_value: Any, events: list[str]) -> None:
        self.staged = staged
        self.prepared_value = prepared_value
        self.events = events
        self._progress = v1.LifecycleProgress()

    def launch_payload(self, identity: v6.SourceSmokeV6Identity) -> Mapping[str, object]:
        del identity
        return {
            "schema": "cross_session_worst_group_m1_source_smoke_physical_launch_v6",
            "provider": "V6AuditSpecReboundCommonStratumSourceProvider",
            "runner": "TorchCSWGSmokeRunner",
            "derivative_observer": "V5DerivativeEvidenceCollector",
            "audit_spec_then_rebind_to_smoke": True,
            "source_opened": False,
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps_completed": 0,
            "source_only": True,
        }

    def prepare_source(self, identity: v6.SourceSmokeV6Identity) -> Any:
        attempt = self.staged / identity.spec.root_relative / "attempt.json"
        assert attempt.exists(), "attempt must precede prepare"
        self.events.append("prepare_after_attempt")
        self._progress = v1.LifecycleProgress(source_resolved_or_opened=True)
        return self.prepared_value

    def run_smoke(self, identity: v6.SourceSmokeV6Identity) -> Mapping[str, object]:
        self.events.append("run_100")
        self._progress = v1.LifecycleProgress(
            source_resolved_or_opened=True, source_authority_published=True,
            model_constructed=True, cuda_initialized=True, optimizer_steps_completed=100,
        )
        result = _valid_inner_smoke(identity.inherited_v1_smoke_identity)
        result["_checkpoint_bodies"] = {"best_source_train_loss": b"best", "last": b"last"}
        result["_v5_derivative_evidence"] = _passed_derivative_evidence(
            tuple(identity.inherited_v1_smoke_identity.spec.stage0_spec.source_sessions),  # type: ignore[arg-type]
        )
        return result

    def checkpoint_bodies(self) -> Mapping[str, bytes]:
        return {"best_source_train_loss": b"best", "last": b"last"}

    def progress(self) -> v1.LifecycleProgress:
        return self._progress

    def close(self) -> None:
        self.events.append("close")


def test_v6_workorder_closure_and_public_dry_cli_are_static(tmp_path: Path) -> None:
    assert v6.WORKORDER_SHA256 == "3baf485008655aa5dc9e6a24d2a8e1805b791e1c78e377b1eeb45b3eceea1a3c"
    closure = v6.implementation_closure(ROOT)
    assert closure["historical_v3_closure_sha256"] == v5.V3_CLOSURE_SHA256
    assert closure["historical_v5_failed_graph"]["exact_leaf_count"] == 6
    assert len(closure["paths"]) == len({item["path"] for item in closure["paths"]})
    assert {v6.WORKORDER_RELATIVE, *v6._RUNTIME_DEPENDENCY_PATHS, *v6._OWNED_PATHS}.issubset(
        {item["path"] for item in closure["paths"]},
    )
    staged = _stage_v6_closure(tmp_path)
    identity = v6.build_source_smoke_v6_identity(staged, device=_device())
    v6.validate_source_smoke_v6_identity_current(staged, identity)
    script = ROOT / "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_smoke_v6.py"
    command = [sys.executable, str(script), "--dry-run"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, env={
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "PATH": os.environ["PATH"],
    })
    payload = json.loads(completed.stdout)
    assert payload["phase"] == v6.PHASE and payload["imports_torch"] is False
    assert payload["workorder_sha256"] == v6.WORKORDER_SHA256
    probe = subprocess.run([
        sys.executable, "-c",
        "import importlib.util,sys; "
        f"s=importlib.util.spec_from_file_location('v6dry',{str(script)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.main(['--dry-run']); "
        "print('TORCH_PRESENT='+str('torch' in sys.modules))",
    ], check=True, capture_output=True, text=True, env={
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "PATH": os.environ["PATH"],
    })
    assert "TORCH_PRESENT=False" in probe.stdout
    rejected = subprocess.run(
        [sys.executable, str(script), "--execute"], capture_output=True, text=True,
        env={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "PATH": os.environ["PATH"]},
    )
    assert rejected.returncode != 0 and "root-reviewed capability" in rejected.stderr


def test_v6_closure_leaf_reader_uses_held_no_follow_fd_and_revalidates_named_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "closure"
    root.mkdir()
    regular = root / "regular.py"
    regular.write_bytes(b"exact regular closure bytes\n")
    assert v6._read_regular_no_follow(root, "regular.py") == _sha(regular.read_bytes())

    target = tmp_path / "target.py"
    target.write_bytes(b"must never be followed\n")
    linked = root / "linked.py"
    linked.symlink_to(target)
    with pytest.raises(v6.SourceSmokeV6Error, match="non-symlink|no-follow open"):
        v6._read_regular_no_follow(root, "linked.py")

    leaf = root / "replace.py"
    leaf.write_bytes(b"opened through the held descriptor\n")
    original_read = os.read
    replaced = False

    def replace_named_leaf_after_first_read(file_descriptor: int, count: int) -> bytes:
        nonlocal replaced
        chunk = original_read(file_descriptor, count)
        if chunk and not replaced:
            replaced = True
            prior = root / "replace-prior.py"
            leaf.rename(prior)
            replacement = root / "replace-new.py"
            replacement.write_bytes(b"replacement named leaf\n")
            os.replace(replacement, leaf)
        return chunk

    monkeypatch.setattr(v6.os, "read", replace_named_leaf_after_first_read)
    with pytest.raises(v6.SourceSmokeV6Error, match="named identity changed after read"):
        v6._read_regular_no_follow(root, "replace.py")
    assert replaced is True
    reader = inspect.getsource(v6._read_regular_no_follow)
    assert "os.open" in reader and "O_NOFOLLOW" in reader and "os.fstat" in reader and "os.read" in reader


def test_v6_route_import_is_clean_no_torch_or_cuda_initialization() -> None:
    probe = subprocess.run([
        sys.executable, "-c",
        "import sys; "
        "from tfpd_exploration.src.cross_session_worst_group_v1 import source_smoke_v6,source_smoke_physical_v6; "
        "print('TORCH_PRESENT='+str('torch' in sys.modules))",
    ], check=True, capture_output=True, text=True, env={
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "PATH": os.environ["PATH"],
        "PYTHONPATH": f"{ROOT / 'tfpd_exploration'}:{ROOT / 'tfpd_exploration/src'}",
    })
    assert probe.stdout.strip() == "TORCH_PRESENT=False"


def test_direct_smoke_spec_to_v3_builder_fails_but_audit_then_rebind_passes(prepared: Any, tmp_path: Path) -> None:
    audit_spec = v1.source_audit_spec()
    smoke_spec = v1.source_smoke_spec()
    descriptors = tuple(_descriptor(session) for session in audit_spec.stage0_spec.source_sessions)
    materials = {item.session_id: v4_physical._cache_material(_material(item)) for item in descriptors}
    with pytest.raises(v3.SourceAuditV3Error, match="source selection/outer-target ordering drift"):
        v3.build_common_stratum_prepared_audit(
            physical_module=physical, spec=smoke_spec, descriptors=descriptors, materials=materials,
        )
    audit = v3.build_common_stratum_prepared_audit(
        physical_module=physical, spec=audit_spec, descriptors=descriptors, materials=materials,
    )
    rebound = v4_physical.rebind_v3_audit_prepared_to_smoke(physical_module=physical, audit_prepared=audit)
    assert rebound.audit_prepared.spec == audit_spec
    assert rebound.inherited_smoke_prepared.spec == smoke_spec
    # The actual V6 provider must retain the same V4-approved operation order
    # without rebuilding V4/V3 current closures.
    provider_source = inspect.getsource(physical_v6.V6AuditSpecReboundCommonStratumSourceProvider.prepare)
    assert "accepted_v3_identity.inherited_v1_identity.spec" in provider_source
    assert "rebind_v3_audit_prepared_to_smoke" in provider_source
    assert "source_smoke_spec()" not in provider_source.split("audit_spec =", 1)[1].split("build_common", 1)[0]
    factory_source = inspect.getsource(physical_v6.build_reviewed_v6_source_smoke_backend)
    assert "_expected_v3_identity" not in factory_source and "implementation_closure" not in factory_source
    assert prepared.inherited_smoke_prepared.spec == smoke_spec


def test_v6_provider_passes_historical_audit_spec_then_rebinds_exact_smoke_spec(tmp_path: Path) -> None:
    staged = _stage_v6_closure(tmp_path)
    identity = v6.build_source_smoke_v6_identity(staged, device=_device())
    descriptors = tuple(_descriptor(session) for session in v1.source_audit_spec().stage0_spec.source_sessions)
    materials = {item.session_id: _material(item) for item in descriptors}

    class _Manifest:
        received: v1.SourceRouteSpec | None = None

        def resolve_exact_sources(self, spec: v1.SourceRouteSpec) -> tuple[physical.SourceFileDescriptor, ...]:
            self.received = spec
            return descriptors

    class _Reader:
        def read_source_session(self, descriptor: physical.SourceFileDescriptor) -> physical.SourceSessionMaterial:
            return materials[descriptor.session_id]

    manifest = _Manifest()
    provider = physical_v6.V6AuditSpecReboundCommonStratumSourceProvider(physical, manifest, _Reader())
    actual = provider.prepare(identity)
    assert manifest.received == v1.source_audit_spec()
    assert actual.audit_prepared.spec == v1.source_audit_spec()
    assert actual.inherited_smoke_prepared.spec == v1.source_smoke_spec()
    with pytest.raises(v6.SourceSmokeV6Error, match="audit-spec"):
        v6.HistoricalV1AuditIdentity(v1.source_smoke_spec())


def test_historical_v3_codec_requires_exact_audit_spec_and_never_rebuilds_current_v3() -> None:
    identity = _historical_v3_identity_payload()
    graph = v5.HeldHistoricalGraph(
        expectation=v6.V3_COMPLETED_EXPECTATION, root_identity=(1, 2), named_chain_identities=((".", 1, 2),),
        bodies={
            "attempt.json": {"identity": identity}, "launch.json": {}, "source_authority.json": {},
            "audit.json": {}, "terminal.json": {},
        },
    )
    # The production exact SHA check is intentionally separate from this codec
    # unit test; a synthetic body cannot hash to the immutable real V3 SHA.
    source = inspect.getsource(v6.historical_v3_accepted_identity)
    assert "build_source_audit_v3_identity" not in source and "implementation_closure" not in source
    assert v6._historical_v3_audit_spec_from_identity_payload(
        identity, expected_identity_sha256=_sha(_json_bytes(identity)), expected_closure_sha256=v5.V3_CLOSURE_SHA256,
    ) == v1.source_audit_spec()
    mutated = _historical_v3_identity_payload()
    inherited_v1 = mutated["inherited_v2_identity"]["inherited_v1_identity"]
    assert isinstance(inherited_v1, dict)
    inherited_v1["spec"] = v1.source_smoke_spec().payload()
    bad_graph = v5.HeldHistoricalGraph(
        expectation=v6.V3_COMPLETED_EXPECTATION, root_identity=(1, 2), named_chain_identities=((".", 1, 2),),
        bodies={
            "attempt.json": {"identity": mutated}, "launch.json": {}, "source_authority.json": {},
            "audit.json": {}, "terminal.json": {},
        },
    )
    assert graph.expectation == v6.V3_COMPLETED_EXPECTATION
    with pytest.raises(v6.SourceSmokeV6Error, match="audit-spec"):
        bad_identity = bad_graph.body("attempt.json")["identity"]
        assert isinstance(bad_identity, Mapping)
        v6._historical_v3_audit_spec_from_identity_payload(
            bad_identity, expected_identity_sha256=_sha(_json_bytes(dict(bad_identity))),
            expected_closure_sha256=v5.V3_CLOSURE_SHA256,
        )


def test_v6_is_independent_of_stale_current_v3_v4_closure_reconstruction() -> None:
    """The observer seam makes old closure rebuilds fail closed, by design."""
    with pytest.raises(v4.SourceSmokeV4Error, match="inherited V3 closure"):
        v4.implementation_closure(ROOT)
    assert v6.implementation_closure(ROOT)["historical_v3_closure_sha256"] == v5.V3_CLOSURE_SHA256


def test_held_pair_loader_rejects_extra_and_recomputed_body_substitution(tmp_path: Path) -> None:
    root = tmp_path / "stage"
    graph_root = root / "synthetic/failed"
    graph_root.mkdir(parents=True)
    os.chmod(graph_root, 0o755)
    pairs = tuple((name, _write_pair(graph_root, name, {"kind": name})) for name in ("attempt.json", "failure.json"))
    expectation = v5.HistoricalGraphExpectation("synthetic/failed", pairs, 4, "synthetic V6 held graph")
    held = v5._read_held_historical_graph(root, expectation, semantic_validator=lambda graph: graph.payload())
    assert held.expectation == expectation
    extra = graph_root / "extra.json"
    extra.write_text("{}", encoding="utf-8")
    os.chmod(extra, 0o444)
    with pytest.raises(v5.SourceSmokeV5Error, match="topology|extra"):
        v5._read_held_historical_graph(root, expectation, semantic_validator=lambda graph: graph.payload())
    os.chmod(extra, 0o644)
    extra.unlink()
    body = graph_root / "failure.json"
    sidecar = graph_root / "failure.json.sha256"
    os.chmod(body, 0o644)
    os.chmod(sidecar, 0o644)
    body.write_bytes(_json_bytes({"kind": "substituted"}))
    sidecar.write_bytes(f"{_sha(body.read_bytes())}  failure.json\n".encode("ascii"))
    os.chmod(body, 0o444)
    os.chmod(sidecar, 0o444)
    with pytest.raises(v5.SourceSmokeV5Error, match="digest"):
        v5._read_held_historical_graph(root, expectation, semantic_validator=lambda graph: graph.payload())


def test_v5_actual_shaped_failure_semantics_rejects_tamper_and_binds_literal_preimage() -> None:
    assert _sha(v6.V5_FAILURE_PREIMAGE.encode("utf-8")) == v6.V5_FAILURE_ERROR_SHA256
    identity = {
        "schema": "cross_session_worst_group_m1_source_smoke_identity_v5",
        "cell": v5.CELL,
        "phase": "m1_source_smoke_v5_granular_derivative_successor",
        "closure": {"closure_sha256": "a" * 64},
        "accepted_v3_closure_sha256_historical": v5.V3_CLOSURE_SHA256,
        "accepted_v3_identity_sha256_historical": v5.V3_IDENTITY_SHA256,
        "accepted_v3_completed_graph": v5.V3_COMPLETED_EXPECTATION.payload(),
        "failed_v4_graph": v5.V4_FAILED_EXPECTATION.payload(),
    }
    flags = {"source_only": True, "target_optimizer_backward_update": 0, **v1.FORBIDDEN_SURFACE_FLAGS}
    good = v5.HeldHistoricalGraph(
        expectation=v6.V5_FAILED_EXPECTATION, root_identity=(3, 4), named_chain_identities=((".", 3, 4),),
        bodies={
            "attempt.json": {
                "schema": "cross_session_worst_group_m1_source_smoke_attempt_v5", "cell": v5.CELL,
                "status": "ATTEMPT_RESERVED", "identity": identity, "source_resolved_or_opened": False,
                "model_constructed": False, "cuda_initialized": False, "optimizer_steps_completed": 0, **flags,
            },
            "launch.json": {
                "schema": "cross_session_worst_group_m1_source_smoke_launch_v5", "cell": v5.CELL,
                "status": "LAUNCHED", "identity": identity, "attempt_sha256": v6.V5_ATTEMPT_SHA256,
                "source_resolved_or_opened": False, "model_constructed": False, "cuda_initialized": False,
                "optimizer_steps_completed": 0, **flags,
            },
            "failure.json": {
                "schema": "cross_session_worst_group_m1_source_smoke_failure_v5", "cell": v5.CELL,
                "status": "FAILED", "identity": identity, "attempt_sha256": v6.V5_ATTEMPT_SHA256,
                "launch_sha256": v6.V5_LAUNCH_SHA256, "source_authority_sha256": None,
                "failure_stage": "physical_or_lifecycle", "failed_predicate": "backend_or_lifecycle_exception",
                "error_class": v6.V5_FAILURE_CLASS, "error_sha256": v6.V5_FAILURE_ERROR_SHA256,
                "terminal_published": False,
                "progress": v1.LifecycleProgress(source_resolved_or_opened=True).payload(), **flags,
            },
        },
    )
    v6._validate_historical_v5_graph(good)
    bad_bodies = dict(good.bodies)
    bad_failure = dict(good.body("failure.json"))
    bad_failure["error_sha256"] = "b" * 64
    bad_bodies["failure.json"] = bad_failure
    graph = v5.HeldHistoricalGraph(
        expectation=v6.V5_FAILED_EXPECTATION, root_identity=(3, 4), named_chain_identities=((".", 3, 4),),
        bodies=bad_bodies,
    )
    with pytest.raises(v6.SourceSmokeV6Error, match="failure semantics"):
        v6._validate_historical_v5_graph(graph)


def test_complete_mock_lifecycle_has_attempt_before_prepare_and_v5_gate(prepared: Any, tmp_path: Path) -> None:
    staged = _stage_v6_closure(tmp_path)
    identity = v6.build_source_smoke_v6_identity(staged, device=_device())
    v3_graph, v5_graph = _fake_graphs(prepared)
    capability = v6.SourceSmokeV6Capability(identity.sha256, v3_graph.sha256, v5_graph.sha256, v6._V6_ROOT_REVIEW_SEAL)
    events: list[str] = []
    predecessor_calls: list[str] = []
    backend = _MockBackend(staged=staged, prepared_value=prepared, events=events)
    _fallback, fallback_sha, _step_zero = v6._prepared_fallback(prepared)
    result = v6._execute_reviewed_source_smoke_v6(
        staged, identity=identity, capability=capability, backend=backend, environ=_environ(),
        predecessor_loader=lambda _root: (predecessor_calls.append("held-read") or (v3_graph, v5_graph)),
        typed_v3_identity_validator=lambda current, _graph: (
            current.accepted_v3_identity.inherited_v1_identity.spec == v1.source_audit_spec()
            or (_ for _ in ()).throw(v6.SourceSmokeV6Error("synthetic typed V3 mismatch")),
        ),
        accepted_fallback_sha256=fallback_sha,
    )
    assert result.terminal_sha256 is not None and result.failure_sha256 is None
    assert events == ["prepare_after_attempt", "run_100", "close"]
    assert predecessor_calls == ["held-read", "held-read"]
    artifact = staged / v6.V6_ROOT_RELATIVE
    expected = {
        "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
        "source_authority.json", "source_authority.json.sha256", "smoke.json", "smoke.json.sha256",
        "checkpoint_best_source_train_loss.pt", "checkpoint_best_source_train_loss.pt.sha256",
        "checkpoint_last.pt", "checkpoint_last.pt.sha256", "checkpoint_manifest.json", "checkpoint_manifest.json.sha256",
        "terminal.json", "terminal.json.sha256",
    }
    assert {path.name for path in artifact.iterdir()} == expected
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in artifact.iterdir())
    authority = json.loads((artifact / "source_authority.json").read_text(encoding="utf-8"))
    assert authority["audit_spec_to_smoke_rebind"]["v3_audit_builder_received_audit_spec"] is True
    smoke = json.loads((artifact / "smoke.json").read_text(encoding="utf-8"))
    assert smoke["v5_derivative_numeric_gate_retained_exactly"] is True


def test_collision_rejects_before_backend_prepare(prepared: Any, tmp_path: Path) -> None:
    staged = _stage_v6_closure(tmp_path)
    identity = v6.build_source_smoke_v6_identity(staged, device=_device())
    v3_graph, v5_graph = _fake_graphs(prepared)
    capability = v6.SourceSmokeV6Capability(identity.sha256, v3_graph.sha256, v5_graph.sha256, v6._V6_ROOT_REVIEW_SEAL)
    collision = staged / v6.V6_ROOT_RELATIVE
    collision.mkdir(parents=True)
    events: list[str] = []
    with pytest.raises(v6.SourceSmokeV6Error, match="already exists"):
        v6._execute_reviewed_source_smoke_v6(
            staged, identity=identity, capability=capability,
            backend=_MockBackend(staged=staged, prepared_value=prepared, events=events), environ=_environ(),
            predecessor_loader=lambda _root: (v3_graph, v5_graph),
            typed_v3_identity_validator=lambda _identity, _graph: None,
        )
    assert events == []


def test_capability_predecessor_failure_occurs_before_fresh_v6_root(tmp_path: Path) -> None:
    staged = _stage_v6_closure(tmp_path)
    identity = v6.build_source_smoke_v6_identity(staged, device=_device())
    with pytest.raises(v6.SourceSmokeV6Error, match="predecessor absent"):
        v6._issue_root_reviewed_source_smoke_v6_capability(
            staged, identity, environ=_environ(), review_seal=v6._V6_ROOT_REVIEW_SEAL,
            predecessor_loader=lambda _root: (_ for _ in ()).throw(v6.SourceSmokeV6Error("predecessor absent")),
        )
    assert not (staged / v6.V6_ROOT_RELATIVE).exists()


def test_v6_physical_backend_keeps_v5_observer_and_does_not_copy_optimizer_loop() -> None:
    source = inspect.getsource(physical_v6.PhysicalCommonStratumSmokeV6Backend.run_smoke)
    assert "self.smoke_runner.run" in source and "optimizer.step" not in source and "torch.optim" not in source
    provider = inspect.getsource(physical_v6.V6AuditSpecReboundCommonStratumSourceProvider.prepare)
    assert "audit_spec" in provider and "build_common_stratum_prepared_audit" in provider
    assert "rebind_v3_audit_prepared_to_smoke" in provider
    factory = inspect.getsource(physical_v6.build_reviewed_v6_source_smoke_backend)
    assert "DerivativeEvidenceCollector" in factory and "bootstrap_reviewed_v1_route" in factory
    import torch
    assert torch.cuda.is_initialized() is False
