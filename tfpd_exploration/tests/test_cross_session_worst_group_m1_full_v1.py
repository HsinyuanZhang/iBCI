"""Focused no-data/no-CUDA tests for the V6-bound CS-WG full successor."""
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

from tfpd_exploration.src.cross_session_worst_group_full_v1 import full_train as full  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_full_v1 import physical as full_physical  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import plan  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_physical as shared  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_smoke_v5 as v5  # noqa: E402
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
        cuda_visible_devices="synthetic-visible-device",
        torch_device="cuda:0",
        uuid="GPU-synthetic-full",
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
        "CUDA_VISIBLE_DEVICES": "synthetic-visible-device",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        **v1.THREAD_ENVIRONMENT,
    }


def _stage_full_closure(tmp_path: Path) -> Path:
    """Stage code metadata only, never a source/result artifact body."""
    staged = tmp_path / "stage"
    closure = full.implementation_closure(ROOT)
    for row in closure["paths"]:
        relative = str(row["path"])
        target = staged / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    (staged / "tfpd_exploration/results").mkdir(parents=True, exist_ok=True)
    # The shared immutable-artifact primitive intentionally requires a
    # pre-existing named parent and creates only the fresh terminal directory.
    # This is a synthetic test parent, never the canonical prospective root.
    (staged / full.FULL_ROOT_RELATIVE).parent.mkdir(parents=True, exist_ok=True)
    return staged


def _fake_v6_graph(staged: Path) -> tuple[full.AcceptedV6SmokeGraph, full.AcceptedV6GraphExpectation]:
    """Create a complete synthetic 16-leaf V6 terminal graph for descriptor tests."""
    relative = "tfpd_exploration/results/synthetic_accepted_v6"
    directory = staged / relative
    directory.mkdir(parents=True)
    identity = {
        "schema": "cross_session_worst_group_m1_source_smoke_identity_v6",
        "cell": full.CELL,
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
    closure_sha = "c" * 64
    best_sha = _write_pair(directory, "checkpoint_best_source_train_loss.pt", b"synthetic-v6-best")
    last_sha = _write_pair(directory, "checkpoint_last.pt", b"synthetic-v6-last")
    attempt_sha = _write_json_pair(directory, "attempt.json", {"identity": identity})
    launch_sha = _write_json_pair(directory, "launch.json", {"identity": identity})
    authority_sha = _write_json_pair(directory, "source_authority.json", {"identity": identity})
    smoke_sha = _write_json_pair(directory, "smoke.json", {
        "identity_sha256": identity_sha,
        "optimizer_steps": v1.SMOKE_STEPS,
        **_flags(),
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
    terminal = {
        "schema": "cross_session_worst_group_m1_source_smoke_terminal_v6",
        "status": "PASS_SOURCE_SMOKE_COMMON_STRATUM_CONSTRUCTIBLE_AUDIT_SPEC_REBOUND",
        "identity": identity,
        "launch_closure_sha256": closure_sha,
        "final_closure_sha256": closure_sha,
        "attempt_sha256": attempt_sha,
        "launch_sha256": launch_sha,
        "source_authority_sha256": authority_sha,
        "smoke_sha256": smoke_sha,
        "checkpoint_manifest_sha256": manifest_sha,
        **_flags(),
    }
    terminal_sha = _write_json_pair(directory, "terminal.json", terminal)
    expectation = full.AcceptedV6GraphExpectation(
        root_relative=relative,
        terminal_sha256=terminal_sha,
        identity_sha256=identity_sha,
        closure_sha256=closure_sha,
    )
    return full.load_accepted_v6_smoke_graph(staged, expectation=expectation), expectation


def _derivative_evidence(sessions: tuple[str, str, str], expected_steps: int) -> dict[str, object]:
    collector = full_physical.FullDerivativeEvidenceCollector(sessions)
    for step in range(expected_steps):
        collector.observe({
            "schema": "cross_session_worst_group_m1_derivative_observation_v1",
            "step_index": step,
            "session_ids": sessions,
            "session_loss_values_fp32": (1.0, 1.0, 1.0),
            "raw_autograd_weight_values_fp32": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            "tensor_count": 0,
            "graph_retained": False,
        })
    return collector.validate(expected_steps=expected_steps)


@dataclass
class _FakePrepared:
    sessions: tuple[str, str, str]
    steps_per_epoch: int
    audit_spec: v1.SourceRouteSpec = v1.source_audit_spec()
    accepted_v6_graph_sha256: str = ""

    def authority_fragment(self) -> dict[str, object]:
        # The value must be cardinality-consistent under the actual V1
        # floor-per-session law, not merely sum to B32s by accident.  The
        # focused lifecycle uses six steps: two complete B32 units/session.
        assert self.steps_per_epoch % len(self.sessions) == 0
        windows = {
            session: (self.steps_per_epoch // len(self.sessions)) * plan.TOTAL_BATCH_SIZE
            for session in self.sessions
        }
        expected = v1.paired_epoch_step_count(
            full.full_training_spec().inherited_v1_full_spec, windows,
        )
        assert expected == self.steps_per_epoch
        return {
            "valid_source_windows": windows,
            "paired_cswg_and_matched_erm_steps_per_epoch": expected,
            "accepted_v6_historical_audit_spec": self.audit_spec.payload(),
            "accepted_v6_historical_audit_spec_sha256": self.audit_spec.sha256,
            "accepted_v6_smoke_graph_sha256": self.accepted_v6_graph_sha256,
            "synthetic_source_only": True,
        }


@dataclass
class _FakeFullBackend:
    root: Path
    steps_per_epoch: int
    events: list[str]
    _bodies: Mapping[str, bytes] | None = None
    _progress: v1.LifecycleProgress = v1.LifecycleProgress()
    closed: bool = False
    inject_swa_claim: bool = False
    fail_after_authority: bool = False
    accepted_v6_graph_sha256: str = ""

    def launch_payload(self, identity: full.FullTrainingIdentity) -> Mapping[str, object]:
        self.events.append("launch")
        return {
            "schema": "cross_session_worst_group_m1_full_training_physical_launch_v1",
            "provider": "V6BoundFullCommonStratumSourceProvider",
            "runner": "TorchCSWGFullTrainingRunner",
            "accepted_v6_smoke_graph_sha256": self.accepted_v6_graph_sha256,
            "source_opened": False,
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps_completed": 0,
            "swa_enabled": False,
            "swa_artifact_forbidden": True,
            **_flags(),
        }

    def prepare_source(self, identity: full.FullTrainingIdentity) -> object:
        attempt = self.root / identity.spec.root_relative / "attempt.json"
        assert attempt.is_file(), "attempt must be durable before source preparation"
        self.events.append("prepare_after_attempt")
        self._progress = v1.LifecycleProgress(source_resolved_or_opened=True)
        return _FakePrepared(
            tuple(identity.inherited_v1_full_identity.spec.stage0_spec.source_sessions), self.steps_per_epoch,
            accepted_v6_graph_sha256=self.accepted_v6_graph_sha256,
        )

    def run_full(self, identity: full.FullTrainingIdentity, epoch_observer: Any) -> Mapping[str, object]:
        self.events.append("run_full")
        if self.fail_after_authority:
            raise RuntimeError("synthetic full after authority failure")
        sessions = tuple(identity.inherited_v1_full_identity.spec.stage0_spec.source_sessions)
        total = plan.M1_EPOCH_BUDGET * self.steps_per_epoch
        names_digest = _sha(_json_bytes(["weight"]))
        loss_values = [2.0] * plan.M1_EPOCH_BUDGET
        loss_values[4] = 1.0
        loss_values[5] = 1.0  # strict first-minimum selection is epoch 4.
        epochs: list[dict[str, object]] = []
        for index, loss in enumerate(loss_values):
            state = "b" * 64 if index == 4 else ("c" * 64 if index == 19 else f"{index % 10:x}" * 64)
            row = {
                "schema": "cross_session_worst_group_m1_source_training_epoch_v1",
                "epoch_index": index,
                "steps_in_epoch": self.steps_per_epoch,
                "global_optimizer_steps_completed": (index + 1) * self.steps_per_epoch,
                "source_train_loss": loss,
                "source_train_loss_aggregation": "arithmetic_mean_of_complete_source_objective_per_step",
                "episode_index_scope": "epoch_local",
                "one_concatenated_forward_per_step": True,
                "finite_objective": True,
                "finite_model": True,
                "finite_adam_state": True,
                "model_state_sha256": state,
                **_flags(),
            }
            epochs.append(row)
            epoch_observer(row)
        self._bodies = {"best_source_train_loss": b"full-best", "last": b"full-last"}
        self._progress = v1.LifecycleProgress(
            source_resolved_or_opened=True, model_constructed=True,
            cuda_initialized=True, optimizer_steps_completed=total, source_authority_published=True,
        )
        result: dict[str, object] = {
            "schema": "cross_session_worst_group_m1_source_full_training_physical_v1",
            "identity_sha256": identity.inherited_v1_full_identity.sha256,
            "training_plan": shared.SourceTrainingPlan.full(steps_per_epoch=self.steps_per_epoch).payload(),
            "epochs": epochs,
            "optimizer_steps": total,
            "total_windows_per_step": plan.TOTAL_BATCH_SIZE,
            "one_concatenated_forward_per_step": True,
            "calibration_shape_per_row": list(plan.M1_CALIBRATION_SHAPE_PER_ROW),
            "model_parameter_count": plan.M1_LIVE_PARAMETERS_AFTER_LAZY1024,
            "model_output_shape": [plan.TOTAL_BATCH_SIZE, plan.M1_WINDOW_SIZE, plan.M1_RAW_BEHAVIOR_OUTPUTS],
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
            "session_objective_derivatives_nonnegative": True,
            "dynamic_dropout_preserved": True,
            "initial_model_state_sha256": "a" * 64,
            "final_model_state_sha256": "c" * 64,
            "best_checkpoint_state_sha256": "b" * 64,
            "best_source_train_loss": 1.0,
            "best_source_train_loss_epoch_index": 4,
            "swa_enabled": False,
            "swa_artifact_forbidden": True,
            "rng": {"seed": v1.SEED, "post_run_state_restored": True},
            "runtime_environment": {
                "tf32_matmul_after": False, "tf32_cudnn_after": False, "amp": False, "compile": False,
            },
            "resources": {
                "elapsed_seconds": 1.0, "steps_per_second": float(total), "samples_per_second": float(total * 32),
                "cuda_current_allocated_bytes": 1, "cuda_peak_allocated_bytes": 2,
                "cuda_current_reserved_bytes": 1, "cuda_peak_reserved_bytes": 2,
            },
            "derivative_numeric_evidence": _derivative_evidence(sessions, total),
            "model_constructed": True,
            "cuda_initialized": True,
            **_flags(),
            "_checkpoint_bodies": dict(self._bodies),
        }
        if self.inject_swa_claim:
            result["swa.pt"] = "forbidden"
        return result

    def checkpoint_bodies(self) -> Mapping[str, bytes]:
        assert self._bodies is not None
        return dict(self._bodies)

    def progress(self) -> v1.LifecycleProgress:
        return self._progress

    def close(self) -> None:
        self.closed = True
        self.events.append("close")


def _identity_and_capability(staged: Path) -> tuple[full.FullTrainingIdentity, full.AcceptedV6SmokeGraph, full.FullTrainingCapability]:
    graph, expectation = _fake_v6_graph(staged)
    identity = full.build_full_training_identity(staged, device=_device(), v6_expectation=expectation)
    cap = full._issue_root_reviewed_full_capability(
        staged, identity=identity, environ=_environ(), graph_loader=lambda _root: graph,
    )
    return identity, graph, cap


def test_full_plan_is_exact_20_epochs_best_last_and_no_swa() -> None:
    spec = full.full_training_spec()
    payload = spec.payload()
    assert payload["epoch_count"] == 20
    assert payload["optimizer"] == "Adam"
    assert payload["adam_lr"] == 1.0e-5
    assert payload["adam_weight_decay"] == 0.0
    assert payload["scheduler"] == "None"
    assert payload["checkpoint_monitor"] == "source_train_loss_epoch_mean"
    assert payload["swa_enabled"] is False
    assert payload["swa_artifact_forbidden"] is True
    names = full._expected_success_names(20, terminal=True)
    assert all("swa" not in name.lower() for name in names)
    with pytest.raises(full.CSWGFullTrainError, match="SWA"):
        full._reject_swa_claim({"swa.pt": "forbidden", "swa_enabled": False, "swa_artifact_forbidden": True})
    with pytest.raises(full.CSWGFullTrainError, match="SWA"):
        full._reject_swa_claim({"swa_manifest": "forbidden", "swa_enabled": False, "swa_artifact_forbidden": True})
    with pytest.raises(shared.SourcePhysicalError, match="SWA"):
        shared.SourceTrainingPlan(
            run_kind="full", epoch_count=20, steps_per_epoch=1,
            reset_episode_index_each_epoch=True,
            checkpoint_selection="epoch_mean_source_train_loss_first_minimum", swa_enabled=True,
        )
    with pytest.raises((AttributeError, TypeError)):
        shared.SourceTrainingPlan.full(steps_per_epoch=6).epoch_count = 1  # type: ignore[misc]


def test_one_shared_optimizer_loop_and_smoke_facade_uses_exact_historical_plan() -> None:
    core_source = inspect.getsource(shared.TorchCSWGTrainingRunner.run_training)
    smoke_source = inspect.getsource(shared.TorchCSWGSmokeRunner.run)
    full_source = inspect.getsource(shared.TorchCSWGFullTrainingRunner.run_full)
    assert core_source.count("for global_step_index in range(execution_plan.total_steps)") == 1
    assert "optimizer = torch.optim.Adam" in core_source
    assert core_source.count("strict_reload_checkpoint_bytes(") == 2
    assert "run_training(" in smoke_source and "torch.optim.Adam" not in smoke_source
    assert "run_training(" in full_source and "torch.optim.Adam" not in full_source

    class ProbeSmoke(shared.TorchCSWGSmokeRunner):
        def run_training(self, **kwargs: Any) -> Mapping[str, object]:
            return {"plan": kwargs["execution_plan"].payload(), "epoch_observer": kwargs.get("epoch_observer")}

    class ProbeFull(shared.TorchCSWGFullTrainingRunner):
        def run_training(self, **kwargs: Any) -> Mapping[str, object]:
            return {"plan": kwargs["execution_plan"].payload(), "epoch_observer": kwargs["epoch_observer"]}

    smoke = ProbeSmoke(ROOT).run(identity=object(), prepared=object())  # type: ignore[arg-type]
    assert smoke["plan"] == shared.SourceTrainingPlan.smoke().payload()
    assert smoke["epoch_observer"] is None
    prepared = type("Prepared", (), {"paired_steps_per_epoch": 7})()
    observer = lambda _row: None
    full_result = ProbeFull(ROOT).run_full(identity=object(), prepared=prepared, epoch_observer=observer)  # type: ignore[arg-type]
    assert full_result["plan"] == shared.SourceTrainingPlan.full(steps_per_epoch=7).payload()
    assert full_result["epoch_observer"] is observer


def test_held_v6_loader_requires_exact_16_leaf_terminal_graph_and_rejects_tamper(tmp_path: Path) -> None:
    staged = _stage_full_closure(tmp_path)
    graph, expectation = _fake_v6_graph(staged)
    assert graph.expectation == expectation
    assert len(graph.body_sha256) == 8
    assert graph.terminal["status"].startswith("PASS_SOURCE_SMOKE")
    assert full.historical_v6_audit_spec(graph) == v1.source_audit_spec()
    leaf = staged / expectation.root_relative / "smoke.json"
    os.chmod(leaf, 0o644)
    with pytest.raises(full.CSWGFullTrainError, match="mode/identity"):
        full.load_accepted_v6_smoke_graph(staged, expectation=expectation)
    os.chmod(leaf, 0o444)
    (staged / expectation.root_relative / "failure.json").write_text("{}", encoding="utf-8")
    with pytest.raises(full.CSWGFullTrainError, match="topology/failure/extra"):
        full.load_accepted_v6_smoke_graph(staged, expectation=expectation)


def test_full_temp_lifecycle_is_attempt_before_prepare_terminalized_and_has_no_swa(tmp_path: Path) -> None:
    staged = _stage_full_closure(tmp_path)
    identity, graph, capability = _identity_and_capability(staged)
    events: list[str] = []
    backend = _FakeFullBackend(staged, steps_per_epoch=6, events=events)
    backend.accepted_v6_graph_sha256 = graph.sha256
    result = full._execute_reviewed_full_training(
        staged, identity=identity, capability=capability, backend=backend, environ=_environ(),
        graph_loader=lambda _root: graph,
    )
    assert result.terminal_sha256 is not None and result.failure_sha256 is None
    assert events[:3] == ["launch", "prepare_after_attempt", "run_full"]
    assert events[-1] == "close" and backend.closed is True
    assert tuple(result.epoch_sha256) == tuple(f"epoch_{index:02d}.json" for index in range(20))
    root = staged / identity.spec.root_relative
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "PASS_SOURCE_FULL_FIXED_20_EPOCH_NO_SWA"
    assert terminal["swa_enabled"] is False and terminal["swa_artifact_forbidden"] is True
    assert tuple(sorted(manifest["checkpoints"])) == ("best_source_train_loss", "last")
    assert manifest["best_epoch_index"] == 4
    assert terminal["checkpoint_best_source_train_loss_sha256"] == manifest["checkpoints"]["best_source_train_loss"]["sha256"]
    assert terminal["checkpoint_last_sha256"] == manifest["checkpoints"]["last"]["sha256"]
    assert terminal["checkpoint_best_source_train_loss_state_sha256"] == manifest["checkpoints"]["best_source_train_loss"]["state_sha256"]
    assert terminal["checkpoint_last_state_sha256"] == manifest["checkpoints"]["last"]["state_sha256"]
    assert not any("swa" in path.name.lower() for path in root.iterdir())
    assert len(tuple(root.iterdir())) == len(full._expected_success_names(20, terminal=True))
    for path in root.iterdir():
        assert stat.S_ISREG(path.stat().st_mode) and stat.S_IMODE(path.stat().st_mode) == 0o444


def test_full_failure_after_authority_is_honest_and_never_publishes_terminal(tmp_path: Path) -> None:
    staged = _stage_full_closure(tmp_path)
    identity, graph, capability = _identity_and_capability(staged)
    backend = _FakeFullBackend(staged, steps_per_epoch=6, events=[], fail_after_authority=True)
    backend.accepted_v6_graph_sha256 = graph.sha256
    result = full._execute_reviewed_full_training(
        staged, identity=identity, capability=capability, backend=backend, environ=_environ(),
        graph_loader=lambda _root: graph,
    )
    assert result.failure_sha256 is not None and result.terminal_sha256 is None
    root = staged / identity.spec.root_relative
    failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
    assert failure["source_authority_sha256"] == result.source_authority_sha256
    assert failure["progress"]["source_resolved_or_opened"] is True
    assert failure["terminal_published"] is False
    assert not (root / "terminal.json").exists()


def test_swa_claim_and_derivative_or_checkpoint_drift_fail_before_terminal(tmp_path: Path) -> None:
    staged = _stage_full_closure(tmp_path)
    identity, graph, capability = _identity_and_capability(staged)
    backend = _FakeFullBackend(staged, steps_per_epoch=6, events=[], inject_swa_claim=True)
    backend.accepted_v6_graph_sha256 = graph.sha256
    result = full._execute_reviewed_full_training(
        staged, identity=identity, capability=capability, backend=backend, environ=_environ(),
        graph_loader=lambda _root: graph,
    )
    assert result.failure_sha256 is not None and result.terminal_sha256 is None

    collector = full_physical.FullDerivativeEvidenceCollector(
        tuple(identity.inherited_v1_full_identity.spec.stage0_spec.source_sessions),
    )
    with pytest.raises(full_physical.CSWGFullPhysicalError, match="scalar domain"):
        collector.observe({
            "schema": "cross_session_worst_group_m1_derivative_observation_v1",
            "step_index": 0,
            "session_ids": tuple(identity.inherited_v1_full_identity.spec.stage0_spec.source_sessions),
            "session_loss_values_fp32": (0.0, 20.0, 20.0),
            "raw_autograd_weight_values_fp32": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            "tensor_count": 0,
            "graph_retained": False,
        })


def test_capability_requires_v6_graph_before_fresh_root_and_rejects_environment_drift(tmp_path: Path) -> None:
    staged = _stage_full_closure(tmp_path)
    graph, expectation = _fake_v6_graph(staged)
    identity = full.build_full_training_identity(staged, device=_device(), v6_expectation=expectation)
    bad_env = dict(_environ())
    bad_env["CUDA_VISIBLE_DEVICES"] = "wrong"
    with pytest.raises(full.CSWGFullTrainError, match="device environment"):
        full._issue_root_reviewed_full_capability(
            staged, identity=identity, environ=bad_env, graph_loader=lambda _root: graph,
        )
    prospective = staged / identity.spec.root_relative
    prospective.parent.mkdir(parents=True, exist_ok=True)
    prospective.mkdir()
    with pytest.raises(full.CSWGFullTrainError, match="already exists"):
        full._issue_root_reviewed_full_capability(
            staged, identity=identity, environ=_environ(), graph_loader=lambda _root: graph,
        )


def test_backend_and_source_authority_cannot_substitute_a_different_v6_graph(tmp_path: Path) -> None:
    staged = _stage_full_closure(tmp_path)
    identity, graph, _capability = _identity_and_capability(staged)
    backend = _FakeFullBackend(staged, steps_per_epoch=6, events=[], accepted_v6_graph_sha256="0" * 64)
    with pytest.raises(full.CSWGFullTrainError, match="physical launch schema"):
        full._full_launch_payload(identity, graph, "a" * 64, backend.launch_payload(identity))
    with pytest.raises(full.CSWGFullTrainError, match="source authority"):
        full._full_source_authority_payload(
            identity, graph,
            _FakePrepared(
                tuple(identity.inherited_v1_full_identity.spec.stage0_spec.source_sessions), 6,
                accepted_v6_graph_sha256="0" * 64,
            ),
        )


def test_public_dry_cli_is_stdlib_only_and_execution_is_rejected() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_cross_session_worst_group_m1_full_v1.py"
    env = {
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    }
    completed = subprocess.run([sys.executable, str(script), "--dry-run"], check=True, text=True,
                               capture_output=True, env={**os.environ, **env})
    payload = json.loads(completed.stdout)
    assert payload["swa_enabled"] is False and payload["swa_artifact_forbidden"] is True
    assert payload["imports_torch"] is False and payload["initializes_cuda"] is False
    rejected = subprocess.run([sys.executable, str(script), "--execute"], text=True,
                              capture_output=True, env={**os.environ, **env})
    assert rejected.returncode != 0
    assert "root-reviewed capability" in rejected.stderr
    # Check module state in the isolated public-CLI process rather than the
    # pytest worker, whose prior V1 fixtures are permitted to import Torch.
    probe = subprocess.run(
        [sys.executable, "-c", (
            "import runpy,sys; "
            f"sys.argv=[{str(script)!r}, '--dry-run']; "
            "\ntry:\n runpy.run_path(sys.argv[0], run_name='__main__')\n"
            "except SystemExit:\n pass\n"
            "print('TORCH_PRESENT=' + str('torch' in sys.modules))"
        )],
        check=True, text=True, capture_output=True, env={**os.environ, **env},
    )
    assert probe.stdout.rstrip().endswith("TORCH_PRESENT=False")
    qualified = subprocess.run(
        [sys.executable, "-c", (
            "import sys; "
            "import tfpd_exploration.src.cross_session_worst_group_full_v1.full_train as route; "
            "print(route.__name__); "
            "print('TOPLEVEL_V1=' + str('cross_session_worst_group_v1' in sys.modules)); "
            "print('TORCH_PRESENT=' + str('torch' in sys.modules))"
        )],
        check=True, text=True, capture_output=True,
        env={**os.environ, **env, "PYTHONPATH": str(ROOT)},
    )
    assert qualified.stdout.splitlines() == [
        "tfpd_exploration.src.cross_session_worst_group_full_v1.full_train",
        "TOPLEVEL_V1=False", "TORCH_PRESENT=False",
    ]


def test_current_full_closure_is_explicit_and_historical_v6_closure_is_never_rebuilt() -> None:
    first = full.implementation_closure(ROOT)
    second = full.implementation_closure(ROOT)
    assert first == second
    assert first["historical_v6_closure_sha256"] == full.V6_CLOSURE_SHA256
    paths = [row["path"] for row in first["paths"]]
    assert len(paths) == len(set(paths))
    source = inspect.getsource(full.implementation_closure)
    assert "v6.implementation_closure" not in source
    assert "tfpd_exploration/src/cross_session_worst_group_v1/source_smoke_v6.py" in paths
