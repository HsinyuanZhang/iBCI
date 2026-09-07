"""Focused synthetic/no-CUDA tests for fold-20120924 MATCHED_ERM full v1."""
from __future__ import annotations

from dataclasses import dataclass
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

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfpd_exploration.src.cross_session_worst_group_full_v1 import full_train as shared_full  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_matched_erm_full_v1 import full_train as erm  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_matched_erm_full_v1 import physical as erm_physical  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import plan  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_physical as runner  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_smoke_v6 as v6  # noqa: E402


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_pair(directory: Path, name: str, body: bytes) -> str:
    digest = _sha(body)
    (directory / name).write_bytes(body)
    (directory / f"{name}.sha256").write_bytes(f"{digest}  {name}\n".encode("ascii"))
    os.chmod(directory / name, 0o444)
    os.chmod(directory / f"{name}.sha256", 0o444)
    return digest


def _write_json_pair(directory: Path, name: str, value: Mapping[str, object]) -> str:
    return _write_pair(directory, name, _json_bytes(dict(value)))


def _flags() -> dict[str, object]:
    return {
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _device() -> v1.DeviceProfile:
    return v1.DeviceProfile(
        cuda_visible_devices="synthetic-erm-device",
        torch_device="cuda:0",
        uuid="GPU-synthetic-erm", pci_bus_id="00000000:08:00.0", name="Synthetic ERM GPU",
        compute_capability=(9, 0), total_memory_bytes=12_345_678,
        torch_version="synthetic-torch", cuda_version="synthetic-cuda", cudnn_version=90000,
    )


def _environ() -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "synthetic-erm-device", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        **v1.THREAD_ENVIRONMENT,
    }


def _stage_closure(tmp_path: Path) -> Path:
    staged = tmp_path / "stage"
    closure = erm.implementation_closure(ROOT)
    for row in closure["paths"]:
        relative = str(row["path"])
        destination = staged / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    (staged / "tfpd_exploration/results").mkdir(parents=True, exist_ok=True)
    (staged / erm.MATCHED_ERM_FULL_ROOT_RELATIVE).parent.mkdir(parents=True, exist_ok=True)
    return staged


def _fake_v6_graph(staged: Path) -> tuple[shared_full.AcceptedV6SmokeGraph, shared_full.AcceptedV6GraphExpectation]:
    relative = "tfpd_exploration/results/synthetic_matched_erm_accepted_v6"
    directory = staged / relative
    directory.mkdir(parents=True)
    identity = {
        "schema": "cross_session_worst_group_m1_source_smoke_identity_v6",
        "cell": shared_full.CELL,
        "phase": "m1_source_smoke_v6_audit_spec_successor",
        "accepted_v3_identity": {
            "schema": "cross_session_worst_group_m1_historical_v3_audit_identity_v6",
            "inherited_v1_audit_spec": v1.source_audit_spec().payload(),
            "historical_v3_identity_sha256": v6.V3_IDENTITY_SHA256,
            "historical_v3_closure_sha256": v6.V3_CLOSURE_SHA256,
            "source_only": True,
        },
        "accepted_v3_completed_graph": v6.V3_COMPLETED_EXPECTATION.payload(),
        "failed_v5_graph": v6.V5_FAILED_EXPECTATION.payload(),
    }
    identity_sha = _sha(_json_bytes(identity))
    closure_sha = "3" * 64
    best_sha = _write_pair(directory, "checkpoint_best_source_train_loss.pt", b"synthetic-erm-v6-best")
    last_sha = _write_pair(directory, "checkpoint_last.pt", b"synthetic-erm-v6-last")
    attempt_sha = _write_json_pair(directory, "attempt.json", {"identity": identity})
    launch_sha = _write_json_pair(directory, "launch.json", {"identity": identity})
    authority_sha = _write_json_pair(directory, "source_authority.json", {"identity": identity})
    smoke_sha = _write_json_pair(directory, "smoke.json", {
        "identity_sha256": identity_sha, "optimizer_steps": v1.SMOKE_STEPS, **_flags(),
    })
    manifest_sha = _write_json_pair(directory, "checkpoint_manifest.json", {
        "schema": "cross_session_worst_group_m1_source_smoke_checkpoint_manifest_v6",
        "inherited_v1_checkpoint_manifest": {
            "checkpoints": {
                "best_source_train_loss": {
                    "filename": "checkpoint_best_source_train_loss.pt", "sha256": best_sha,
                },
                "last": {"filename": "checkpoint_last.pt", "sha256": last_sha},
            },
        },
    })
    terminal_sha = _write_json_pair(directory, "terminal.json", {
        "schema": "cross_session_worst_group_m1_source_smoke_terminal_v6",
        "status": "PASS_SOURCE_SMOKE_COMMON_STRATUM_CONSTRUCTIBLE_AUDIT_SPEC_REBOUND",
        "identity": identity,
        "launch_closure_sha256": closure_sha, "final_closure_sha256": closure_sha,
        "attempt_sha256": attempt_sha, "launch_sha256": launch_sha,
        "source_authority_sha256": authority_sha, "smoke_sha256": smoke_sha,
        "checkpoint_manifest_sha256": manifest_sha, **_flags(),
    })
    expectation = shared_full.AcceptedV6GraphExpectation(
        root_relative=relative, terminal_sha256=terminal_sha,
        identity_sha256=identity_sha, closure_sha256=closure_sha,
    )
    return shared_full.load_accepted_v6_smoke_graph(staged, expectation=expectation), expectation


@dataclass
class _FakePrepared:
    sessions: tuple[str, str, str]
    steps_per_epoch: int
    graph_sha256: str
    audit_spec: v1.SourceRouteSpec

    def authority_fragment(self) -> dict[str, object]:
        windows = {session: 64 for session in self.sessions}
        expected = v1.paired_epoch_step_count(
            v1.build_fold_route_specs(v1.SOURCE_SMOKE_OUTER_TARGET)[1], windows,
        )
        assert expected == self.steps_per_epoch
        return {
            "valid_source_windows": windows,
            "paired_cswg_and_matched_erm_steps_per_epoch": expected,
            "accepted_v6_historical_audit_spec": self.audit_spec.payload(),
            "accepted_v6_historical_audit_spec_sha256": self.audit_spec.sha256,
            "accepted_v6_smoke_graph_sha256": self.graph_sha256,
        }


@dataclass
class _FakeERMFullBackend:
    root: Path
    graph_sha256: str
    events: list[str]
    fail_after_authority: bool = False
    _bodies: Mapping[str, bytes] | None = None
    _progress: v1.LifecycleProgress = v1.LifecycleProgress()
    closed: bool = False

    def launch_payload(self, identity: shared_full.FullTrainingIdentity) -> Mapping[str, object]:
        self.events.append("launch")
        return {
            "schema": "cross_session_worst_group_m1_full_training_physical_launch_v1",
            "provider": "V6BoundFullCommonStratumSourceProvider",
            "runner": "TorchCSWGFullTrainingRunner",
            "derivative_observer": "MatchedERMDerivativeEvidenceCollector",
            "accepted_v6_smoke_graph_sha256": self.graph_sha256,
            "source_opened": False, "model_constructed": False, "cuda_initialized": False,
            "optimizer_steps_completed": 0, "swa_enabled": False, "swa_artifact_forbidden": True,
            **_flags(),
        }

    def prepare_source(self, identity: shared_full.FullTrainingIdentity) -> object:
        assert (self.root / identity.spec.root_relative / "attempt.json").is_file()
        self.events.append("prepare_after_attempt")
        self._progress = v1.LifecycleProgress(source_resolved_or_opened=True)
        return _FakePrepared(
            tuple(identity.inherited_v1_full_identity.spec.stage0_spec.source_sessions), 6,
            self.graph_sha256, v1.source_audit_spec(),
        )

    def run_full(self, identity: shared_full.FullTrainingIdentity, epoch_observer: Any) -> Mapping[str, object]:
        self.events.append("run_full")
        if self.fail_after_authority:
            raise RuntimeError("synthetic matched-ERM post-authority failure")
        assert identity.inherited_v1_full_identity.spec.stage0_spec.system == "MATCHED_ERM"
        assert runner.TorchCSWGTrainingRunner.objective_args_from_identity(identity.inherited_v1_full_identity) == (0.0, 0.01)
        total = 20 * 6
        collector = erm_physical.MatchedERMDerivativeEvidenceCollector(
            tuple(identity.inherited_v1_full_identity.spec.stage0_spec.source_sessions),
        )
        for index in range(total):
            collector.observe({
                "schema": "cross_session_worst_group_m1_derivative_observation_v1",
                "step_index": index,
                "session_ids": tuple(identity.inherited_v1_full_identity.spec.stage0_spec.source_sessions),
                "session_loss_values_fp32": (1.0, 2.0, 3.0),
                "raw_autograd_weight_values_fp32": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
                "tensor_count": 0, "graph_retained": False,
            })
        names_digest = _sha(_json_bytes(["weight"]))
        epochs: list[dict[str, object]] = []
        for index in range(20):
            loss = 1.0 if index == 4 else 2.0
            state = "b" * 64 if index == 4 else ("c" * 64 if index == 19 else f"{index % 10:x}" * 64)
            row = {
                "schema": "cross_session_worst_group_m1_source_training_epoch_v1",
                "epoch_index": index, "steps_in_epoch": 6,
                "global_optimizer_steps_completed": (index + 1) * 6,
                "source_train_loss": loss,
                "source_train_loss_aggregation": "arithmetic_mean_of_complete_source_objective_per_step",
                "episode_index_scope": "epoch_local", "one_concatenated_forward_per_step": True,
                "finite_objective": True, "finite_model": True, "finite_adam_state": True,
                "model_state_sha256": state, **_flags(),
            }
            epochs.append(row)
            epoch_observer(row)
        self._bodies = {"best_source_train_loss": b"erm-best", "last": b"erm-last"}
        self._progress = v1.LifecycleProgress(
            source_resolved_or_opened=True, model_constructed=True, cuda_initialized=True,
            optimizer_steps_completed=total, source_authority_published=True,
        )
        return {
            "schema": "cross_session_worst_group_m1_source_full_training_physical_v1",
            "identity_sha256": identity.inherited_v1_full_identity.sha256,
            "training_plan": runner.SourceTrainingPlan.full(steps_per_epoch=6).payload(),
            "epochs": epochs, "optimizer_steps": total, "total_windows_per_step": plan.TOTAL_BATCH_SIZE,
            "one_concatenated_forward_per_step": True,
            "calibration_shape_per_row": list(plan.M1_CALIBRATION_SHAPE_PER_ROW),
            "model_parameter_count": plan.M1_LIVE_PARAMETERS_AFTER_LAZY1024,
            "model_output_shape": [plan.TOTAL_BATCH_SIZE, plan.M1_WINDOW_SIZE, plan.M1_RAW_BEHAVIOR_OUTPUTS],
            "finite_objective": True, "finite_model": True, "finite_gradients": True,
            "gradient_coverage": {
                "trainable_parameter_count": 1, "trainable_parameter_names_sha256": names_digest,
                "observed_gradient_count": 1, "observed_gradient_names_sha256": names_digest,
                "missing_trainable_names": [], "excluded_trainable_names": [],
            },
            "finite_adam_state": True, "model_state_changed": True,
            "checkpoint_reload_strict": True, "best_checkpoint_reload_strict": True,
            "last_checkpoint_reload_strict": True, "session_objective_derivatives_nonnegative": True,
            "dynamic_dropout_preserved": True,
            "initial_model_state_sha256": "a" * 64, "final_model_state_sha256": "c" * 64,
            "best_checkpoint_state_sha256": "b" * 64,
            "best_source_train_loss": 1.0, "best_source_train_loss_epoch_index": 4,
            "swa_enabled": False, "swa_artifact_forbidden": True,
            "rng": {"seed": v1.SEED, "post_run_state_restored": True},
            "runtime_environment": {"tf32_matmul_after": False, "tf32_cudnn_after": False, "amp": False, "compile": False},
            "resources": {
                "elapsed_seconds": 1.0, "steps_per_second": float(total), "samples_per_second": float(total * 32),
                "cuda_current_allocated_bytes": 1, "cuda_peak_allocated_bytes": 2,
                "cuda_current_reserved_bytes": 1, "cuda_peak_reserved_bytes": 2,
            },
            "derivative_numeric_evidence": collector.validate(expected_steps=total),
            "model_constructed": True, "cuda_initialized": True, **_flags(),
            "_checkpoint_bodies": dict(self._bodies),
        }

    def checkpoint_bodies(self) -> Mapping[str, bytes]:
        assert self._bodies is not None
        return dict(self._bodies)

    def progress(self) -> v1.LifecycleProgress:
        return self._progress

    def close(self) -> None:
        self.closed = True
        self.events.append("close")


def _identity_graph_capability(staged: Path) -> tuple[shared_full.FullTrainingIdentity, shared_full.AcceptedV6SmokeGraph, shared_full.FullTrainingCapability]:
    graph, expectation = _fake_v6_graph(staged)
    identity = erm.build_matched_erm_full_identity(staged, device=_device(), v6_expectation=expectation)
    capability = shared_full._issue_root_reviewed_full_capability(
        staged, identity=identity, environ=_environ(), graph_loader=lambda _root: graph,
    )
    return identity, graph, capability


def test_matched_spec_is_exact_same_fold_and_historical_cswg_default_is_preserved() -> None:
    historical = shared_full.full_training_spec()
    assert historical.sha256 == "bc45b521bfddefa28a32b126532406e04e06de7314d543f462c54574045d5c35"
    assert historical.payload()["full_system"] == "CS_WG"
    assert historical.root_relative == shared_full.FULL_ROOT_RELATIVE
    spec = erm.matched_erm_full_training_spec()
    assert spec.payload()["full_system"] == "MATCHED_ERM"
    assert spec.root_relative == erm.MATCHED_ERM_FULL_ROOT_RELATIVE
    assert spec.inherited_v1_full_spec.stage0_spec.lambda_ == 0.0
    assert spec.inherited_v1_full_spec.stage0_spec.source_sessions == historical.inherited_v1_full_spec.stage0_spec.source_sessions
    assert spec.inherited_v1_full_spec.stage0_spec.outer_target_session == historical.inherited_v1_full_spec.stage0_spec.outer_target_session
    assert spec.payload()["swa_enabled"] is False and spec.payload()["swa_artifact_forbidden"] is True


def test_shared_runner_derives_only_typed_objective_args_and_keeps_one_loop() -> None:
    cswg, erm_spec = v1.build_fold_route_specs(v1.SOURCE_SMOKE_OUTER_TARGET)
    closure = {"closure_sha256": "a" * 64}
    cswg_identity = v1.SourceExecutionIdentity(cswg, closure, _device())
    erm_identity = v1.SourceExecutionIdentity(erm_spec, closure, _device())
    assert runner.TorchCSWGTrainingRunner.objective_args_from_identity(cswg_identity) == (1.0, 0.01)
    assert runner.TorchCSWGTrainingRunner.objective_args_from_identity(erm_identity) == (0.0, 0.01)
    source = inspect.getsource(runner.TorchCSWGTrainingRunner.run_training)
    assert source.count("for global_step_index in range(execution_plan.total_steps)") == 1
    assert "objective_args_from_identity(identity)" in source
    assert "optimizer = torch.optim.Adam" in source
    assert "lambda_=lifecycle.CSWG_LAMBDA" not in source


def test_erm_derivative_collector_requires_uniform_mean_reference() -> None:
    collector = erm_physical.MatchedERMDerivativeEvidenceCollector(("20120926", "20120927", "20120928"))
    payload = {
        "schema": "cross_session_worst_group_m1_derivative_observation_v1", "step_index": 0,
        "session_ids": ("20120926", "20120927", "20120928"),
        "session_loss_values_fp32": (0.5, 2.0, 3.0),
        "raw_autograd_weight_values_fp32": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        "tensor_count": 0, "graph_retained": False,
    }
    collector.observe(payload)
    evidence = collector.validate(expected_steps=1)
    assert evidence["objective_system"] == "MATCHED_ERM"
    assert evidence["uniform_reference_weight"] == 1.0 / 3.0
    bad = dict(payload)
    bad["raw_autograd_weight_values_fp32"] = (0.9, 0.05, 0.05)
    with pytest.raises(erm_physical.MatchedERMFullPhysicalError, match="uniform-reference"):
        erm_physical.MatchedERMDerivativeEvidenceCollector(("20120926", "20120927", "20120928")).observe(bad)


def test_physical_factory_is_thin_v6_provider_composition_without_optimizer_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = _stage_closure(tmp_path)
    graph, _expectation = _fake_v6_graph(staged)

    class _Provider:
        manifest = object()
        reader = object()

    _Provider.__name__ = "StrictM1SourceProvider"

    class _Inherited:
        provider = _Provider()

    class _Runner:
        def __init__(self, root: Path, *, derivative_observer: Any) -> None:
            self.root = root
            self.derivative_observer = derivative_observer

    class _Module:
        TorchCSWGFullTrainingRunner = _Runner

        @staticmethod
        def build_route_owned_source_audit_backend(*, root: Path, source_root: Path) -> _Inherited:
            assert root == staged and source_root == Path("/synthetic/source")
            return _Inherited()

    monkeypatch.setattr(erm_physical.v2, "bootstrap_reviewed_v1_route", lambda root: _Module)
    backend = erm_physical.build_reviewed_matched_erm_full_backend(
        root=staged, source_root=Path("/synthetic/source"), accepted_v6_graph=graph,
    )
    assert backend.derivative_observer_label == "MatchedERMDerivativeEvidenceCollector"
    assert isinstance(backend.derivative_collector, erm_physical.MatchedERMDerivativeEvidenceCollector)
    physical_source = inspect.getsource(erm_physical)
    assert "torch.optim.Adam" not in physical_source
    assert "V6BoundFullCommonStratumSourceProvider" in physical_source


def test_full_temp_lifecycle_is_20_epoch_attempt_before_prepare_and_no_swa(tmp_path: Path) -> None:
    staged = _stage_closure(tmp_path)
    identity, graph, capability = _identity_graph_capability(staged)
    events: list[str] = []
    backend = _FakeERMFullBackend(staged, graph.sha256, events)
    result = shared_full._execute_reviewed_full_training(
        staged, identity=identity, capability=capability, backend=backend,
        environ=_environ(), graph_loader=lambda _root: graph,
    )
    assert result.terminal_sha256 is not None and result.failure_sha256 is None
    assert events[:3] == ["launch", "prepare_after_attempt", "run_full"]
    assert events[-1] == "close" and backend.closed
    root = staged / identity.spec.root_relative
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert terminal["identity"]["spec"]["full_system"] == "MATCHED_ERM"
    assert terminal["source_only"] is True and terminal["target_optimizer_backward_update"] == 0
    assert tuple(sorted(manifest["checkpoints"])) == ("best_source_train_loss", "last")
    assert manifest["best_epoch_index"] == 4 and manifest["last_epoch_index"] == 19
    assert len(result.epoch_sha256) == 20
    assert not any("swa" in path.name.lower() for path in root.iterdir())
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in root.iterdir())


def test_v6_predecessor_capability_and_failure_order_fail_closed(tmp_path: Path) -> None:
    staged = _stage_closure(tmp_path)
    graph, expectation = _fake_v6_graph(staged)
    identity = erm.build_matched_erm_full_identity(staged, device=_device(), v6_expectation=expectation)
    bad_env = dict(_environ())
    bad_env["CUDA_VISIBLE_DEVICES"] = "wrong"
    with pytest.raises(shared_full.CSWGFullTrainError, match="device environment"):
        shared_full._issue_root_reviewed_full_capability(
            staged, identity=identity, environ=bad_env, graph_loader=lambda _root: graph,
        )
    prospective = staged / identity.spec.root_relative
    prospective.parent.mkdir(parents=True, exist_ok=True)
    prospective.mkdir()
    with pytest.raises(shared_full.CSWGFullTrainError, match="already exists"):
        shared_full._issue_root_reviewed_full_capability(
            staged, identity=identity, environ=_environ(), graph_loader=lambda _root: graph,
        )


def test_extension_closure_and_public_dry_cli_are_inert() -> None:
    first = erm.implementation_closure(ROOT)
    second = erm.implementation_closure(ROOT)
    assert first == second
    assert tuple(first["successor_extension_paths"]) == (
        "tfpd_exploration/docs/WORKORDER_CS_WG_M1_MATCHED_ERM_FULL_V1_20260827.md",
        "tfpd_exploration/src/cross_session_worst_group_matched_erm_full_v1/__init__.py",
        "tfpd_exploration/src/cross_session_worst_group_matched_erm_full_v1/full_train.py",
        "tfpd_exploration/src/cross_session_worst_group_matched_erm_full_v1/physical.py",
        "tfpd_exploration/scripts/run_cross_session_worst_group_m1_matched_erm_full_v1.py",
        "tfpd_exploration/tests/test_cross_session_worst_group_m1_matched_erm_full_v1.py",
    )
    script = ROOT / "tfpd_exploration/scripts/run_cross_session_worst_group_m1_matched_erm_full_v1.py"
    env = {
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    }
    complete = subprocess.run([sys.executable, str(script), "--dry-run"], text=True, capture_output=True,
                              check=True, env={**os.environ, **env})
    payload = json.loads(complete.stdout)
    assert payload["fixed_same_fold"]["lambda"] == 0.0
    assert payload["imports_torch"] is False and payload["initializes_cuda"] is False
    rejected = subprocess.run([sys.executable, str(script), "--execute"], text=True, capture_output=True,
                              env={**os.environ, **env})
    assert rejected.returncode != 0 and "root-reviewed capability" in rejected.stderr
    probe = subprocess.run(
        [sys.executable, "-c", (
            "import runpy,sys; "
            f"sys.argv=[{str(script)!r}, '--dry-run']; "
            "\ntry:\n runpy.run_path(sys.argv[0], run_name='__main__')\n"
            "except SystemExit:\n pass\n"
            "print('TORCH_PRESENT=' + str('torch' in sys.modules))"
        )], text=True, capture_output=True, check=True, env={**os.environ, **env},
    )
    assert probe.stdout.rstrip().endswith("TORCH_PRESENT=False")
