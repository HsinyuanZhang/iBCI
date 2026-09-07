"""No-data/no-CUDA gates for CS-WG M1 source-audit V2."""
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
    sidecar = directory / f"{name}.sha256"
    body_path.write_bytes(body)
    sidecar.write_bytes(f"{digest}  {name}\n".encode("ascii"))
    os.chmod(body_path, 0o444)
    os.chmod(sidecar, 0o444)
    return digest


def _stage_v2_closure(tmp_path: Path) -> Path:
    staged = tmp_path / "stage"
    closure = v2.implementation_closure(ROOT)
    for row in closure["paths"]:
        relative = str(row["path"])
        source = ROOT / relative
        destination = staged / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    (staged / "tfpd_exploration/results").mkdir(parents=True, exist_ok=True)
    return staged


def _write_synthetic_v1_failure_graph(staged: Path) -> v2.V1FailedGraphExpectation:
    """Build a live-shaped V1 failure graph with test-local body digests."""
    inherited = v1.build_source_audit_identity(staged)
    attempt = v1._source_audit_attempt_payload(inherited)
    attempt_sha = _sha(_json_bytes(attempt))
    backend = {
        "schema": "cross_session_worst_group_m1_source_audit_physical_launch_v1",
        "provider": "StrictM1SourceProvider",
        "source_opened": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
    }
    launch = v1._source_audit_launch_payload(inherited, attempt_sha, backend)
    launch_sha = _sha(_json_bytes(launch))
    error = source_reader.SourceReaderError(
        "CS-WG native parser refuses to replace an existing non-streaming top-level src",
    )
    failure = v1._source_audit_failure_payload(
        inherited,
        attempt_sha256=attempt_sha,
        launch_sha256=launch_sha,
        authority_sha256=None,
        progress=v1.LifecycleProgress(source_resolved_or_opened=False),
        error=error,
    )
    failure_sha = _sha(_json_bytes(failure))
    root = staged / v2.V1_FAILED_ROOT_RELATIVE
    root.mkdir(parents=True)
    os.chmod(root, 0o755)
    assert _write_pair(root, "attempt.json", attempt) == attempt_sha
    assert _write_pair(root, "launch.json", launch) == launch_sha
    assert _write_pair(root, "failure.json", failure) == failure_sha
    return v2.V1FailedGraphExpectation(
        root_relative=v2.V1_FAILED_ROOT_RELATIVE,
        attempt_sha256=attempt_sha,
        launch_sha256=launch_sha,
        failure_sha256=failure_sha,
        error_class="SourceReaderError",
        error_sha256=_sha(repr(error).encode("utf-8")),
    )


def _test_predecessor_loader(expectation: v2.V1FailedGraphExpectation):
    return lambda root: v2._validate_held_v1_failed_audit_graph(root, expectation)


def _descriptor(session: str) -> physical.SourceFileDescriptor:
    position = plan.HELD_IN_SOURCE_SESSIONS.index(session) + 1
    return physical.SourceFileDescriptor(
        session_id=session,
        relative_path=f"synthetic_heldin/{session}.nwb",
        sha256=(f"{position:x}" * 64)[:64],
        byte_count=10_000 + position,
    )


def _material(descriptor: physical.SourceFileDescriptor) -> physical.SourceSessionMaterial:
    raw = np.zeros((11, plan.M1_RAW_BEHAVIOR_OUTPUTS), dtype=np.float32)
    raw[:, 0] = 1.0
    labels = core.SourceOnlyFinalBinLabels(descriptor.session_id, raw, np.ones((11,), dtype=np.bool_))
    calibration = np.full(
        plan.M1_CALIBRATION_SHAPE_PER_ROW,
        float(plan.HELD_IN_SOURCE_SESSIONS.index(descriptor.session_id) + 1),
        dtype=np.float32,
    )
    calibration.setflags(write=False)
    calibration_sha = core.array_digest(calibration)
    held = physical.held_source_identity_payload(
        descriptor,
        device=1,
        inode=100 + plan.HELD_IN_SOURCE_SESSIONS.index(descriptor.session_id),
        byte_count=descriptor.byte_count,
        body_sha256=descriptor.sha256,
        mode=0o644,
        hard_link_count=1,
    )
    position = plan.HELD_IN_SOURCE_SESSIONS.index(descriptor.session_id) + 1
    prefix = (f"{position:x}" * 64)[:64]
    evidence = physical.native_session_evidence(
        descriptor,
        calibration_session=descriptor.session_id,
        calibration_sha256=calibration_sha,
        row_count=11,
        ordered_query_identity_sha256=prefix,
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
        valid_source_windows=64,
        calibration_session=descriptor.session_id,
        calibration_sha256=calibration_sha,
        calibration_backing=calibration,
        native_evidence=evidence,
    )


def _prepared_fold() -> physical.PreparedSourceFold:
    spec = v1.source_audit_spec()
    descriptors = tuple(_descriptor(session) for session in spec.stage0_spec.source_sessions)
    materials = {item.session_id: _material(item) for item in descriptors}
    return physical.prepare_source_fold(spec, descriptors=descriptors, materials=materials)


class _V2AuditBackend:
    def __init__(self, *, root: Path, prepared: physical.PreparedSourceFold, fail_at: str | None = None) -> None:
        self.root = root
        self.prepared = prepared
        self.fail_at = fail_at
        self.calls: list[str] = []
        self._progress = v1.LifecycleProgress()

    def launch_payload(self, identity: v1.SourceAuditIdentity) -> Mapping[str, object]:
        assert identity.spec == self.prepared.spec
        assert (self.root / v2.V2_ROOT_RELATIVE / "attempt.json").is_file()
        self.calls.extend(("attempt_seen", "launch"))
        if self.fail_at == "launch":
            raise RuntimeError("synthetic launch failure")
        return {
            "schema": "cross_session_worst_group_m1_source_audit_physical_launch_v1",
            "provider": "StrictM1SourceProvider",
            "source_opened": False,
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps_completed": 0,
            "source_only": True,
        }

    def prepare_source(self, identity: v1.SourceAuditIdentity) -> Mapping[str, object]:
        assert identity.spec == self.prepared.spec
        self.calls.append("prepare_source")
        self._progress = v1.LifecycleProgress(source_resolved_or_opened=True)
        if self.fail_at == "prepare":
            raise RuntimeError("synthetic prepare failure")
        return self.prepared.authority_fragment()

    def run_source_audit(self, identity: v1.SourceAuditIdentity) -> Mapping[str, object]:
        assert identity.spec == self.prepared.spec
        self.calls.append("run_source_audit")
        self._progress = v1.LifecycleProgress(source_resolved_or_opened=True, source_authority_published=True)
        if self.fail_at == "audit":
            raise RuntimeError("synthetic audit failure")
        return {
            "schema": "cross_session_worst_group_m1_source_audit_v1",
            "identity_sha256": identity.sha256,
            "valid_source_windows": {
                session: self.prepared.materials[session].valid_source_windows
                for session in identity.spec.stage0_spec.source_sessions
            },
            "common_strata": [item.payload() for item in self.prepared.common_strata],
            "step_zero_calibration_ownership": self.prepared.step_zero_calibration_ownership(),
            "paired_cswg_and_matched_erm_steps_per_epoch": self.prepared.paired_steps_per_epoch,
            "constructible_step_zero_b32": True,
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps_completed": 0,
            "source_only": True,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }

    def progress(self) -> v1.LifecycleProgress:
        return self._progress

    def close(self) -> None:
        self.calls.append("close")


def _no_bootstrap(_root: Path) -> object:
    return object()


def _issue_test_capability(
    staged: Path, identity: v2.SourceAuditV2Identity, expectation: v2.V1FailedGraphExpectation,
) -> v2.SourceAuditV2Capability:
    return v2._issue_source_audit_v2_capability(
        staged,
        identity,
        review_seal=v2._V2_ROOT_REVIEW_SEAL,
        predecessor_loader=_test_predecessor_loader(expectation),
        route_bootstrapper=_no_bootstrap,
    )


def test_v2_workorder_inherited_v1_closure_and_metadata_binding() -> None:
    assert v2.WORKORDER_SHA256 == "dc439c5a5692e20e5e6cb3f37ee5fcb7137a35b0cde5f5aaedc90379c3e49607"
    assert v2.V1_REPAIRED_CLOSURE_SHA256 == "b5d4fdf5505fd53f160d56160daa642e36e8077171f654d9c5cdff6ee8fb7a28"
    assert v2.V1_METADATA_MANIFEST_SHA256 == v1.M1_METADATA_MANIFEST_SHA256
    closure = v2.implementation_closure(ROOT)
    assert closure["inherited_v1_closure_sha256"] == v2.V1_REPAIRED_CLOSURE_SHA256
    assert closure["sealed_metadata_manifest_sha256"] == v2.V1_METADATA_MANIFEST_SHA256
    paths = [row["path"] for row in closure["paths"]]
    for path in (
        v2.WORKORDER_RELATIVE,
        "tfpd_exploration/src/cross_session_worst_group_v1/source_audit_v2.py",
        "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_audit_v2.py",
        "tfpd_exploration/tests/test_cross_session_worst_group_m1_source_audit_v2.py",
    ):
        assert path in paths
    # The public additive export surface must remain importable without
    # reaching the deferred source reader or touching CUDA.
    assert all(hasattr(v2, name) for name in v2.__all__)


def test_v2_current_closure_drift_fails_closed(tmp_path: Path) -> None:
    staged = _stage_v2_closure(tmp_path)
    identity = v2.build_source_audit_v2_identity(staged)
    v2.validate_source_audit_v2_identity_current(staged, identity)
    staged_workorder = staged / v2.WORKORDER_RELATIVE
    staged_workorder.write_text(staged_workorder.read_text(encoding="utf-8") + "\nforged\n", encoding="utf-8")
    with pytest.raises(v2.SourceAuditV2Error, match="workorder|closure"):
        v2.validate_source_audit_v2_identity_current(staged, identity)


def test_held_v1_predecessor_graph_exact_topology_semantics_and_adversaries(tmp_path: Path) -> None:
    staged = _stage_v2_closure(tmp_path)
    expectation = _write_synthetic_v1_failure_graph(staged)
    graph = v2._validate_held_v1_failed_audit_graph(staged, expectation)
    assert graph.payload()["expectation"]["exact_leaf_count"] == 6
    assert graph.attempt["source_resolved_or_opened"] is False
    assert graph.failure["source_authority_sha256"] is None
    root = staged / expectation.root_relative
    assert tuple(sorted(path.name for path in root.iterdir())) == (
        "attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256",
        "launch.json", "launch.json.sha256",
    )
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in root.iterdir())

    extra = root / "terminal.json"
    extra.write_text("{}", encoding="utf-8")
    os.chmod(extra, 0o444)
    with pytest.raises(v2.SourceAuditV2Error, match="topology"):
        v2._validate_held_v1_failed_audit_graph(staged, expectation)
    os.chmod(extra, 0o644)
    extra.unlink()

    failure = root / "failure.json"
    os.chmod(failure, 0o644)
    failure.write_bytes(_json_bytes({"schema": "forged"}))
    os.chmod(failure, 0o444)
    with pytest.raises(v2.SourceAuditV2Error, match="digest"):
        v2._validate_held_v1_failed_audit_graph(staged, expectation)


def test_predecessor_semantic_and_named_root_substitution_fail_before_v2_capability_or_root(tmp_path: Path) -> None:
    staged = _stage_v2_closure(tmp_path)
    expectation = _write_synthetic_v1_failure_graph(staged)
    identity = v2.build_source_audit_v2_identity(staged)
    root = staged / expectation.root_relative
    failure_path = root / "failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    failure["error_class"] = "ForgedError"
    os.chmod(failure_path, 0o644)
    os.chmod(root / "failure.json.sha256", 0o644)
    replacement_sha = _write_pair(root, "failure.json", failure)
    forged = replace(expectation, failure_sha256=replacement_sha)
    with pytest.raises(v2.SourceAuditV2Error, match="semantic"):
        bootstrap_calls: list[str] = []
        v2._issue_source_audit_v2_capability(
            staged, identity, review_seal=v2._V2_ROOT_REVIEW_SEAL,
            predecessor_loader=_test_predecessor_loader(forged),
            route_bootstrapper=lambda _root: bootstrap_calls.append("bootstrap"),
        )
    assert bootstrap_calls == []
    assert not (staged / v2.V2_ROOT_RELATIVE).exists()

    # Restore an exact graph and demonstrate that an aliased predecessor root
    # cannot survive the held no-follow directory walk.
    shutil.rmtree(root)
    expectation = _write_synthetic_v1_failure_graph(staged)
    root = staged / expectation.root_relative
    replacement = root.with_name("v1_replaced")
    root.rename(replacement)
    os.symlink(replacement.name, root, target_is_directory=True)
    with pytest.raises(v2.SourceAuditV2Error, match="unsafe|no-follow|directory"):
        v2._validate_held_v1_failed_audit_graph(staged, expectation)
    assert not (staged / v2.V2_ROOT_RELATIVE).exists()


def test_v2_fresh_root_collision_rejects_after_held_predecessor_without_reusing_v1(tmp_path: Path) -> None:
    staged = _stage_v2_closure(tmp_path)
    expectation = _write_synthetic_v1_failure_graph(staged)
    identity = v2.build_source_audit_v2_identity(staged)
    collision = staged / v2.V2_ROOT_RELATIVE
    collision.mkdir(parents=True)
    events: list[str] = []

    def predecessor_loader(root: Path) -> v2.HeldV1FailedAuditGraph:
        events.append("predecessor")
        return v2._validate_held_v1_failed_audit_graph(root, expectation)

    def bootstrapper(_root: Path) -> object:
        events.append("bootstrap")
        return object()

    with pytest.raises(v2.SourceAuditV2Error, match="already exists"):
        v2._issue_source_audit_v2_capability(
            staged, identity, review_seal=v2._V2_ROOT_REVIEW_SEAL,
            predecessor_loader=predecessor_loader, route_bootstrapper=bootstrapper,
        )
    assert events == ["predecessor", "bootstrap"]
    assert tuple(sorted(path.name for path in (staged / expectation.root_relative).iterdir())) == (
        "attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256",
        "launch.json", "launch.json.sha256",
    )
    assert tuple(collision.iterdir()) == ()


def test_v2_lifecycle_is_attempt_before_source_and_terminalizes_a_fresh_root(tmp_path: Path) -> None:
    staged = _stage_v2_closure(tmp_path)
    expectation = _write_synthetic_v1_failure_graph(staged)
    identity = v2.build_source_audit_v2_identity(staged)
    capability = _issue_test_capability(staged, identity, expectation)
    backend = _V2AuditBackend(root=staged, prepared=_prepared_fold())
    result = v2._execute_reviewed_source_audit_v2(
        staged, identity=identity, capability=capability, backend=backend,
        predecessor_loader=_test_predecessor_loader(expectation), route_bootstrapper=_no_bootstrap,
    )
    assert result.terminal_sha256 is not None and result.failure_sha256 is None
    assert backend.calls == ["attempt_seen", "launch", "prepare_source", "run_source_audit", "close"]
    root = staged / v2.V2_ROOT_RELATIVE
    expected = {
        "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
        "source_authority.json", "source_authority.json.sha256", "audit.json", "audit.json.sha256",
        "terminal.json", "terminal.json.sha256",
    }
    assert {path.name for path in root.iterdir()} == expected
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in root.iterdir())
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["v1_failed_predecessor_sha256"] == capability.predecessor_binding_sha256
    assert terminal["source_only"] is True and terminal["model_constructed"] is False
    assert terminal["cuda_initialized"] is False and terminal["optimizer_steps_completed"] == 0


def test_v2_failure_after_attempt_is_honest_and_v1_public_loader_rejects_synthetic_graph(tmp_path: Path) -> None:
    staged = _stage_v2_closure(tmp_path)
    expectation = _write_synthetic_v1_failure_graph(staged)
    identity = v2.build_source_audit_v2_identity(staged)
    capability = _issue_test_capability(staged, identity, expectation)
    backend = _V2AuditBackend(root=staged, prepared=_prepared_fold(), fail_at="prepare")
    result = v2._execute_reviewed_source_audit_v2(
        staged, identity=identity, capability=capability, backend=backend,
        predecessor_loader=_test_predecessor_loader(expectation), route_bootstrapper=_no_bootstrap,
    )
    assert result.terminal_sha256 is None and result.failure_sha256 is not None
    failure = json.loads((staged / v2.V2_ROOT_RELATIVE / "failure.json").read_text(encoding="utf-8"))
    assert failure["source_authority_sha256"] is None
    assert failure["progress"]["source_resolved_or_opened"] is True
    assert failure["progress"]["model_constructed"] is False
    assert failure["progress"]["cuda_initialized"] is False
    with pytest.raises(v2.SourceAuditV2Error, match="digest"):
        v2.validate_v1_failed_audit_graph(staged)


def test_v2_clean_subprocess_reproduces_bad_namespace_and_reaches_deferred_parser_seam_without_cuda(tmp_path: Path) -> None:
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
    bad = "\n".join((
        "import pathlib, sys",
        f"root = pathlib.Path({str(ROOT)!r})",
        "sys.path.insert(0, str(root / 'tfpd_exploration'))",
        "import src.cross_session_worst_group_v1.source_physical",
        "from tfpd_exploration.src.cross_session_worst_group_v1 import source_audit_v2 as v2",
        "try:",
        "    v2.bootstrap_reviewed_v1_route(root)",
        "except v2.SourceAuditV2Error as error:",
        "    print(type(error).__name__)",
        "    print(str(error))",
        "else:",
        "    raise SystemExit('bad namespace unexpectedly accepted')",
    ))
    result = subprocess.run([sys.executable, "-c", bad], cwd=tmp_path, env=environment,
                            text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == ["SourceAuditV2Error", "CS-WG V2 refuses an existing non-streaming top-level src"]

    good = (
        "import pathlib, torch; "
        f"root=pathlib.Path({str(ROOT)!r}); "
        "from tfpd_exploration.src.cross_session_worst_group_v1 import source_audit_v2 as v2; "
        "value=v2.reach_deferred_parser_seam(root); "
        "source_root=root/'lexical_source_root_not_opened'; "
        "backend=v2.build_reviewed_v2_source_audit_backend(root=root,source_root=source_root); "
        "print(value['route_module']); print(value['parser_module_path']); "
        "print(value['opens_nwb']); print(type(backend).__name__); "
        "print(len(backend.provider.read_events)); print(source_root.exists()); "
        "print(torch.cuda.is_initialized())"
    )
    result = subprocess.run([sys.executable, "-c", good], cwd=tmp_path, env=environment,
                            text=True, capture_output=True, check=True)
    route, parser_path, opened, backend_name, read_count, source_exists, cuda = result.stdout.splitlines()
    assert route == "tfpd_exploration.src.cross_session_worst_group_v1.source_physical"
    assert parser_path.endswith("/streaming_calibration_exp/src/data/falcon_datamodule.py")
    assert opened == "False" and backend_name == "PhysicalCSWGSourceAuditBackend"
    assert read_count == "0" and source_exists == "False" and cuda == "False"


def test_static_v2_cli_does_not_import_torch_or_issue_execution(tmp_path: Path) -> None:
    script = ROOT / "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_audit_v2.py"
    environment = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
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
    assert denied.returncode != 0
    assert "root-reviewed" in denied.stderr
