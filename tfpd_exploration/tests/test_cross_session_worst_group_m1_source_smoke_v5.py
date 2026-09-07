"""Focused no-data/no-CUDA tests for CS-WG M1 V5 derivative successor."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping

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
from tfpd_exploration.src.cross_session_worst_group_v1 import source_smoke_v4 as v4  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_smoke_physical_v4 as v4_physical  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_smoke_physical_v5 as physical_v5  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_smoke_v5 as v5  # noqa: E402


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


def _stage_v5_closure(tmp_path: Path) -> Path:
    staged = tmp_path / "stage"
    closure = v5.implementation_closure(ROOT)
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
        uuid="GPU-synthetic-v5",
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
    spec = v1.source_audit_spec()
    descriptors = tuple(_descriptor(session) for session in spec.stage0_spec.source_sessions)
    materials = {item.session_id: v4_physical._cache_material(_material(item)) for item in descriptors}
    audit = v3.build_common_stratum_prepared_audit(
        physical_module=physical, spec=spec, descriptors=descriptors, materials=materials,
    )
    return v4_physical.rebind_v3_audit_prepared_to_smoke(physical_module=physical, audit_prepared=audit)


def _fake_graphs(prepared_value: Any) -> tuple[v5.HeldHistoricalGraph, v5.HeldHistoricalGraph, str]:
    fallback, fallback_sha, _step_zero = v5._prepared_fallback(prepared_value)
    v3_graph = v5.HeldHistoricalGraph(
        expectation=v5.V3_COMPLETED_EXPECTATION,
        root_identity=(101, 102),
        named_chain_identities=((".", 101, 102),),
        bodies={
            "attempt.json": {"synthetic": "attempt"},
            "launch.json": {"synthetic": "launch"},
            "source_authority.json": {
                "deterministic_common_stratum_fallback": fallback,
                "deterministic_common_stratum_fallback_sha256": fallback_sha,
                "common_stratum_step_zero_evidence": dict(prepared_value.step_zero_common_stratum_evidence),
            },
            "audit.json": {"synthetic": "audit"},
            "terminal.json": {"synthetic": "terminal"},
        },
    )
    v4_graph = v5.HeldHistoricalGraph(
        expectation=v5.V4_FAILED_EXPECTATION,
        root_identity=(103, 104),
        named_chain_identities=((".", 103, 104),),
        bodies={
            "attempt.json": {"synthetic": "attempt"},
            "launch.json": {"synthetic": "launch"},
            "source_authority.json": {"synthetic": "authority"},
            "failure.json": {"synthetic": "failure"},
        },
    )
    return v3_graph, v4_graph, fallback_sha


def _valid_inner_smoke(identity: v1.SourceExecutionIdentity, *, derivative_nonnegative: bool = False) -> dict[str, object]:
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
        "session_objective_derivatives_nonnegative": derivative_nonnegative,
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


def _passed_derivative_evidence(sessions: tuple[str, str, str]) -> dict[str, object]:
    collector = v5.DerivativeEvidenceCollector(sessions)
    # Exact CPU FP32 centered-objective derivative for [0, 1, 1].
    raw = (-4.76837158203125e-7, 0.5000002384185791, 0.5000002384185791)
    for step in range(v1.SMOKE_STEPS):
        collector.observe({
            "schema": "cross_session_worst_group_m1_derivative_observation_v1",
            "step_index": step,
            "session_ids": sessions,
            "session_loss_values_fp32": (0.0, 1.0, 1.0),
            "raw_autograd_weight_values_fp32": raw,
            "tensor_count": 0,
            "graph_retained": False,
        })
    return collector.validate(expected_steps=v1.SMOKE_STEPS)


class _MockBackend:
    def __init__(self, *, staged: Path, prepared_value: Any, derivative_evidence: Mapping[str, object],
                 inner_mutation: Callable[[dict[str, object]], None] | None = None,
                 run_error: BaseException | None = None, progress_steps: int = 100,
                 events: list[str] | None = None) -> None:
        self.staged = staged
        self.prepared_value = prepared_value
        self.derivative_evidence = dict(derivative_evidence)
        self.inner_mutation = inner_mutation
        self.run_error = run_error
        self.progress_steps = progress_steps
        self.events = [] if events is None else events

    def launch_payload(self, identity: v5.SourceSmokeV5Identity) -> Mapping[str, object]:
        root = self.staged / identity.spec.root_relative
        assert (root / "attempt.json").exists()
        self.events.append("attempt_before_launch")
        return {
            "schema": "cross_session_worst_group_m1_source_smoke_physical_launch_v5",
            "provider": "V5CachedCommonStratumSourceProvider",
            "runner": "TorchCSWGSmokeRunner",
            "derivative_observer": "V5DerivativeEvidenceCollector",
            "source_opened": False,
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps_completed": 0,
            "source_only": True,
        }

    def prepare_source(self, identity: v5.SourceSmokeV5Identity) -> Any:
        self.events.append("prepare_after_attempt")
        return self.prepared_value

    def run_smoke(self, identity: v5.SourceSmokeV5Identity) -> Mapping[str, object]:
        self.events.append("run_100")
        if self.run_error is not None:
            raise self.run_error
        result = _valid_inner_smoke(identity.inherited_v1_smoke_identity)
        if self.inner_mutation is not None:
            self.inner_mutation(result)
        result["_v5_derivative_evidence"] = dict(self.derivative_evidence)
        return result

    def checkpoint_bodies(self) -> Mapping[str, bytes]:
        return {"best_source_train_loss": b"best", "last": b"last"}

    def progress(self) -> v1.LifecycleProgress:
        return v1.LifecycleProgress(
            source_resolved_or_opened=True,
            model_constructed=True,
            cuda_initialized=True,
            optimizer_steps_completed=self.progress_steps,
            source_authority_published=True,
        )

    def close(self) -> None:
        self.events.append("close")


def test_v5_workorder_closure_literals_and_static_dry_plan(tmp_path: Path) -> None:
    assert v5.WORKORDER_SHA256 == "65dec719e8931270ae1d9b9bdc11c550a8f68198b6457e327ac90431ae1a85d4"
    assert v5.V3_COMPLETED_EXPECTATION.payload()["exact_leaf_count"] == 10
    assert v5.V4_FAILED_EXPECTATION.payload()["exact_leaf_count"] == 8
    closure = v5.implementation_closure(ROOT)
    assert closure["historical_v3_closure_sha256"] == v5.V3_CLOSURE_SHA256
    assert closure["historical_v4_closure_sha256"] == v5.V4_HISTORICAL_CLOSURE_SHA256
    assert len(closure["paths"]) == len({row["path"] for row in closure["paths"]})
    assert {v5.WORKORDER_RELATIVE, *v5._RUNTIME_DEPENDENCY_PATHS, *v5._OWNED_PATHS}.issubset(
        {row["path"] for row in closure["paths"]},
    )
    staged = _stage_v5_closure(tmp_path)
    identity = v5.build_source_smoke_v5_identity(staged, device=_device())
    v5.validate_source_smoke_v5_identity_current(staged, identity)
    workorder = staged / v5.WORKORDER_RELATIVE
    workorder.write_text(workorder.read_text(encoding="utf-8") + "\nforged\n", encoding="utf-8")
    with pytest.raises(v5.SourceSmokeV5Error, match="workorder|closure"):
        v5.validate_source_smoke_v5_identity_current(staged, identity)


def test_held_graph_loader_rejects_extra_sidecar_and_body_substitution(tmp_path: Path) -> None:
    staged = tmp_path / "stage"
    graph_root = staged / "synthetic/failed"
    graph_root.mkdir(parents=True)
    os.chmod(graph_root, 0o755)
    expectation_rows = []
    for name, payload in (("attempt.json", {"kind": "attempt"}), ("failure.json", {"kind": "failure"})):
        expectation_rows.append((name, _write_pair(graph_root, name, payload)))
    expectation = v5.HistoricalGraphExpectation("synthetic/failed", tuple(expectation_rows), 4, "synthetic held")
    held = v5._read_held_historical_graph(staged, expectation, semantic_validator=lambda graph: graph.payload())
    assert held.expectation == expectation and held.payload()["expectation"]["exact_leaf_count"] == 4
    extra = graph_root / "extra.json"
    extra.write_text("{}", encoding="utf-8")
    os.chmod(extra, 0o444)
    with pytest.raises(v5.SourceSmokeV5Error, match="topology|extra"):
        v5._read_held_historical_graph(staged, expectation, semantic_validator=lambda graph: graph.payload())
    os.chmod(extra, 0o644)
    extra.unlink()
    body = graph_root / "failure.json"
    sidecar = graph_root / "failure.json.sha256"
    os.chmod(body, 0o644)
    os.chmod(sidecar, 0o644)
    body.write_bytes(_json_bytes({"kind": "forged"}))
    sidecar.write_bytes(f"{_sha(body.read_bytes())}  failure.json\n".encode("ascii"))
    os.chmod(body, 0o444)
    os.chmod(sidecar, 0o444)
    with pytest.raises(v5.SourceSmokeV5Error, match="digest"):
        v5._read_held_historical_graph(staged, expectation, semantic_validator=lambda graph: graph.payload())


def test_exact_historical_graph_semantic_validators_reject_malformed_actual_topology() -> None:
    """Exact SHA topology is insufficient without the producer receipt schema."""
    v3_graph = v5.HeldHistoricalGraph(
        expectation=v5.V3_COMPLETED_EXPECTATION,
        root_identity=(1, 2), named_chain_identities=((".", 1, 2),),
        bodies={name: {} for name, _digest in v5.V3_COMPLETED_EXPECTATION.pairs()},
    )
    with pytest.raises(v5.SourceSmokeV5Error, match="identity"):
        v5._validate_historical_v3_graph(v3_graph)
    v4_graph = v5.HeldHistoricalGraph(
        expectation=v5.V4_FAILED_EXPECTATION,
        root_identity=(3, 4), named_chain_identities=((".", 3, 4),),
        bodies={name: {} for name, _digest in v5.V4_FAILED_EXPECTATION.pairs()},
    )
    with pytest.raises(v5.SourceSmokeV5Error, match="identity"):
        v5._validate_historical_v4_graph(v4_graph)


def test_v3_actual_audit_identity_digest_schema_rejects_missing_or_substituted_nested_identity() -> None:
    """V3 audit has only an identity digest; its sibling receipts carry the object."""
    identity = {"schema": "synthetic_v3_identity", "closure": {"closure_sha256": "a" * 64}}
    digest = v5._sha(_json_bytes(identity))
    audit = {"identity_sha256": digest}
    nested = {label: {"identity": identity} for label in ("launch", "authority", "terminal")}
    v5._validate_v3_identity_propagation(
        identity=identity, audit=audit, nested_identity_receipts=nested, expected_identity_sha256=digest,
    )
    with pytest.raises(v5.SourceSmokeV5Error, match="nested identity"):
        v5._validate_v3_identity_propagation(
            identity=identity, audit=audit,
            nested_identity_receipts={**nested, "launch": {}}, expected_identity_sha256=digest,
        )
    with pytest.raises(v5.SourceSmokeV5Error, match="nested identity"):
        v5._validate_v3_identity_propagation(
            identity=identity, audit=audit,
            nested_identity_receipts={**nested, "terminal": {"identity": {"schema": "substituted"}}},
            expected_identity_sha256=digest,
        )
    with pytest.raises(v5.SourceSmokeV5Error, match="audit identity"):
        v5._validate_v3_identity_propagation(
            identity=identity, audit={"identity": identity, "identity_sha256": digest},
            nested_identity_receipts=nested, expected_identity_sha256=digest,
        )


def test_v1_default_observer_is_noop_and_v5_observer_is_compact_cpu_only() -> None:
    class _Poison:
        pass

    default = physical.TorchCSWGSmokeRunner(ROOT)
    default_state = dict(vars(default))
    default._observe_session_objective_derivatives(step_index=0, session_losses=(_Poison(),) * 3, derivatives=(_Poison(),) * 3)
    assert default.derivative_observer is None and dict(vars(default)) == default_state
    import torch

    class _Loss:
        def __init__(self, session_id: str, value: float) -> None:
            self.session_id = session_id
            self.value = torch.tensor(value, dtype=torch.float32, requires_grad=True)

    observed: list[Mapping[str, object]] = []
    runner = physical.TorchCSWGSmokeRunner(ROOT, derivative_observer=observed.append)
    losses = tuple(_Loss(session, value) for session, value in zip(("a", "b", "c"), (0.0, 1.0, 1.0), strict=True))
    derivatives = torch.autograd.grad(sum(loss.value for loss in losses), [loss.value for loss in losses])
    runner._observe_session_objective_derivatives(step_index=0, session_losses=losses, derivatives=derivatives)
    assert len(observed) == 1
    payload = observed[0]
    assert payload["tensor_count"] == 0 and payload["graph_retained"] is False
    assert payload["session_ids"] == ("a", "b", "c")
    assert all(isinstance(value, float) for value in payload["session_loss_values_fp32"])
    with pytest.raises(RuntimeError, match="observer"):
        physical.TorchCSWGSmokeRunner(ROOT, derivative_observer=lambda _payload: (_ for _ in ()).throw(RuntimeError("observer")))._observe_session_objective_derivatives(
            step_index=0, session_losses=losses, derivatives=derivatives,
        )
    source = inspect.getsource(physical.TorchCSWGSmokeRunner.run)
    assert source.index("torch.autograd.grad") < source.index("_observe_session_objective_derivatives") < source.index("optimizer.zero_grad")
    assert torch.cuda.is_initialized() is False


def test_fp32_tiny_negative_passes_fixed_gate_and_material_or_wrong_values_fail() -> None:
    sessions = ("20120926", "20120927", "20120928")
    evidence = _passed_derivative_evidence(sessions)
    assert evidence["raw_global_min"] < 0.0
    assert evidence["raw_global_min"] >= v5.DERIVATIVE_MIN_TOLERANCE
    assert evidence["max_reference_abs_error"] <= v5.DERIVATIVE_REFERENCE_ABS_TOLERANCE
    bad_domain = v5.DerivativeEvidenceCollector(sessions)
    for index in range(100):
        bad_domain.observe({
            "schema": "cross_session_worst_group_m1_derivative_observation_v1", "step_index": index,
            "session_ids": sessions, "session_loss_values_fp32": (0.0, 20.0, 20.0),
            "raw_autograd_weight_values_fp32": (9.695689186628442e-6, 0.4999951422214508, 0.4999951422214508),
            "tensor_count": 0, "graph_retained": False,
        })
    with pytest.raises(v5.V5ValidationError, match="loss domain") as domain_error:
        bad_domain.validate(expected_steps=100)
    assert domain_error.value.failed_predicate == "derivative_numeric_gate"
    bad_reference = v5.DerivativeEvidenceCollector(sessions)
    for index in range(100):
        bad_reference.observe({
            "schema": "cross_session_worst_group_m1_derivative_observation_v1", "step_index": index,
            "session_ids": sessions, "session_loss_values_fp32": (0.0, 1.0, 1.0),
            "raw_autograd_weight_values_fp32": (-2.1e-5, 0.5000105, 0.5000105),
            "tensor_count": 0, "graph_retained": False,
        })
    with pytest.raises(v5.V5ValidationError, match="raw/reference") as reference_error:
        bad_reference.validate(expected_steps=100)
    assert reference_error.value.observed_summary["raw_min"] < v5.DERIVATIVE_MIN_TOLERANCE
    wrong_sum = v5.DerivativeEvidenceCollector(sessions)
    for index in range(100):
        wrong_sum.observe({
            "schema": "cross_session_worst_group_m1_derivative_observation_v1", "step_index": index,
            "session_ids": sessions, "session_loss_values_fp32": (0.0, 1.0, 1.0),
            "raw_autograd_weight_values_fp32": (0.1, 0.1, 0.1),
            "tensor_count": 0, "graph_retained": False,
        })
    with pytest.raises(v5.V5ValidationError, match="raw/reference"):
        wrong_sum.validate(expected_steps=100)


def test_cpu_formula_reproduces_the_pre_registered_fp32_cancellation_case() -> None:
    import torch

    values = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32, requires_grad=True)
    mean = values.mean()
    objective = mean + 0.01 * (torch.logsumexp((values - mean) / 0.01, dim=0) - math.log(3))
    raw = torch.autograd.grad(objective, values)[0]
    assert float(raw.min()) < 0.0
    evidence = _passed_derivative_evidence(("20120926", "20120927", "20120928"))
    assert evidence["all_rows_pass"] is True
    assert torch.cuda.is_initialized() is False


def test_granular_validator_preserves_actual_culprit_categories(tmp_path: Path, prepared: Any) -> None:
    staged = _stage_v5_closure(tmp_path)
    identity = v5.build_source_smoke_v5_identity(staged, device=_device())
    v3_graph, v4_graph, _fallback_sha = _fake_graphs(prepared)
    authority = v5._v5_source_authority_payload(
        identity, v3_graph, v4_graph, prepared, accepted_fallback_sha256=_fallback_sha,
    )
    valid = _valid_inner_smoke(identity.inherited_v1_smoke_identity)
    evidence = _passed_derivative_evidence(tuple(identity.inherited_v1_smoke_identity.spec.stage0_spec.source_sessions))
    valid_payload = v5._v5_smoke_payload(identity, v3_graph, v4_graph, "1" * 64, authority, valid, evidence)
    assert valid_payload["non_derivative_v1_predicates_validated"] is True
    coverage = _valid_inner_smoke(identity.inherited_v1_smoke_identity)
    coverage["gradient_coverage"] = {"trainable_parameter_count": 1}
    with pytest.raises(v5.V5ValidationError) as error:
        v5._v5_smoke_payload(identity, v3_graph, v4_graph, "1" * 64, authority, coverage, evidence)
    assert error.value.failed_predicate == "gradient_coverage"
    resources = _valid_inner_smoke(identity.inherited_v1_smoke_identity)
    resources["resources"] = {**resources["resources"], "elapsed_seconds": float("inf")}
    with pytest.raises(v5.V5ValidationError) as error:
        v5._v5_smoke_payload(identity, v3_graph, v4_graph, "1" * 64, authority, resources, evidence)
    assert error.value.failed_predicate == "resources"
    rng = _valid_inner_smoke(identity.inherited_v1_smoke_identity)
    rng["rng"] = {**rng["rng"], "post_run_state_restored": False}
    with pytest.raises(v5.V5ValidationError) as error:
        v5._v5_smoke_payload(identity, v3_graph, v4_graph, "1" * 64, authority, rng, evidence)
    assert error.value.failed_predicate == "rng"


def test_complete_mock_100_step_lifecycle_terminalizes_after_attempt(tmp_path: Path, prepared: Any) -> None:
    staged = _stage_v5_closure(tmp_path)
    identity = v5.build_source_smoke_v5_identity(staged, device=_device())
    v3_graph, v4_graph, fallback_sha = _fake_graphs(prepared)
    capability = v5.SourceSmokeV5Capability(identity.sha256, v3_graph.sha256, v4_graph.sha256, v5._V5_ROOT_REVIEW_SEAL)
    evidence = _passed_derivative_evidence(tuple(identity.inherited_v1_smoke_identity.spec.stage0_spec.source_sessions))
    events: list[str] = []
    backend = _MockBackend(staged=staged, prepared_value=prepared, derivative_evidence=evidence, events=events)
    result = v5._execute_reviewed_source_smoke_v5(
        staged, identity=identity, capability=capability, backend=backend, environ=_environ(),
        predecessor_loader=lambda _root: (v3_graph, v4_graph), accepted_fallback_sha256=fallback_sha,
    )
    assert result.terminal_sha256 is not None and result.failure_sha256 is None
    assert events == ["attempt_before_launch", "prepare_after_attempt", "run_100", "close"]
    root = staged / v5.V5_ROOT_RELATIVE
    expected = {
        "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
        "source_authority.json", "source_authority.json.sha256", "smoke.json", "smoke.json.sha256",
        "checkpoint_best_source_train_loss.pt", "checkpoint_best_source_train_loss.pt.sha256",
        "checkpoint_last.pt", "checkpoint_last.pt.sha256", "checkpoint_manifest.json", "checkpoint_manifest.json.sha256",
        "terminal.json", "terminal.json.sha256",
    }
    assert {path.name for path in root.iterdir()} == expected
    smoke = json.loads((root / "smoke.json").read_text(encoding="utf-8"))
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    assert smoke["derivative_numeric_evidence"]["steps_observed"] == 100
    assert smoke["inherited_v1_smoke"]["session_objective_derivatives_nonnegative"] is False
    assert terminal["target_optimizer_backward_update"] == 0
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in root.iterdir())


def test_granular_failure_keeps_completed_step_progress_and_no_terminal(tmp_path: Path, prepared: Any) -> None:
    staged = _stage_v5_closure(tmp_path)
    identity = v5.build_source_smoke_v5_identity(staged, device=_device())
    v3_graph, v4_graph, fallback_sha = _fake_graphs(prepared)
    capability = v5.SourceSmokeV5Capability(identity.sha256, v3_graph.sha256, v4_graph.sha256, v5._V5_ROOT_REVIEW_SEAL)
    evidence = _passed_derivative_evidence(tuple(identity.inherited_v1_smoke_identity.spec.stage0_spec.source_sessions))
    backend = _MockBackend(
        staged=staged, prepared_value=prepared, derivative_evidence=evidence,
        inner_mutation=lambda result: result.__setitem__("gradient_coverage", {"trainable_parameter_count": 1}),
    )
    result = v5._execute_reviewed_source_smoke_v5(
        staged, identity=identity, capability=capability, backend=backend, environ=_environ(),
        predecessor_loader=lambda _root: (v3_graph, v4_graph), accepted_fallback_sha256=fallback_sha,
    )
    assert result.failure_sha256 is not None and result.terminal_sha256 is None
    failure = json.loads((staged / v5.V5_ROOT_RELATIVE / "failure.json").read_text(encoding="utf-8"))
    assert failure["failure_stage"] == "v5_smoke_validation"
    assert failure["failed_predicate"] == "gradient_coverage"
    assert failure["progress"]["optimizer_steps_completed"] == 100
    assert not (staged / v5.V5_ROOT_RELATIVE / "terminal.json").exists()


def test_derivative_observer_failure_receipts_the_last_completed_boundary(tmp_path: Path, prepared: Any) -> None:
    staged = _stage_v5_closure(tmp_path)
    identity = v5.build_source_smoke_v5_identity(staged, device=_device())
    v3_graph, v4_graph, fallback_sha = _fake_graphs(prepared)
    capability = v5.SourceSmokeV5Capability(identity.sha256, v3_graph.sha256, v4_graph.sha256, v5._V5_ROOT_REVIEW_SEAL)
    backend = _MockBackend(
        staged=staged,
        prepared_value=prepared,
        derivative_evidence=_passed_derivative_evidence(
            tuple(identity.inherited_v1_smoke_identity.spec.stage0_spec.source_sessions),
        ),
        run_error=v5.DerivativeObservationError("synthetic observer rejected a detached scalar", completed_steps=73),
        progress_steps=73,
    )
    result = v5._execute_reviewed_source_smoke_v5(
        staged, identity=identity, capability=capability, backend=backend, environ=_environ(),
        predecessor_loader=lambda _root: (v3_graph, v4_graph), accepted_fallback_sha256=fallback_sha,
    )
    assert result.failure_sha256 is not None and result.terminal_sha256 is None
    failure = json.loads((staged / v5.V5_ROOT_RELATIVE / "failure.json").read_text(encoding="utf-8"))
    assert failure["failure_stage"] == "derivative_observer"
    assert failure["failed_predicate"] == "derivative_observer_exception"
    assert failure["observed_summary"] == {"completed_optimizer_steps": 73}
    assert failure["progress"]["optimizer_steps_completed"] == 73
    assert not (staged / v5.V5_ROOT_RELATIVE / "terminal.json").exists()


def test_capability_requires_both_predecessors_before_fresh_root(tmp_path: Path, prepared: Any) -> None:
    staged = _stage_v5_closure(tmp_path)
    identity = v5.build_source_smoke_v5_identity(staged, device=_device())
    with pytest.raises(v5.SourceSmokeV5Error, match="predecessor"):
        v5._issue_root_reviewed_source_smoke_v5_capability(
            staged, identity, environ=_environ(), review_seal=v5._V5_ROOT_REVIEW_SEAL,
            predecessor_loader=lambda _root: (_ for _ in ()).throw(v5.SourceSmokeV5Error("predecessor absent")),
        )
    assert not (staged / v5.V5_ROOT_RELATIVE).exists()
    v3_graph, v4_graph, _fallback_sha = _fake_graphs(prepared)
    capability = v5._issue_root_reviewed_source_smoke_v5_capability(
        staged, identity, environ=_environ(), review_seal=v5._V5_ROOT_REVIEW_SEAL,
        predecessor_loader=lambda _root: (v3_graph, v4_graph),
    )
    assert capability.v3_completed_graph_sha256 == v3_graph.sha256
    forged_v4 = v5.HeldHistoricalGraph(
        expectation=v4_graph.expectation, root_identity=(999, 1000), named_chain_identities=v4_graph.named_chain_identities,
        bodies=v4_graph.bodies,
    )
    with pytest.raises(v5.SourceSmokeV5Error, match="predecessor/capability"):
        v5._execute_reviewed_source_smoke_v5(
            staged, identity=identity, capability=capability,
            backend=_MockBackend(staged=staged, prepared_value=prepared, derivative_evidence=_passed_derivative_evidence(
                tuple(identity.inherited_v1_smoke_identity.spec.stage0_spec.source_sessions),
            )),
            environ=_environ(), predecessor_loader=lambda _root: (v3_graph, forged_v4),
            accepted_fallback_sha256=_fallback_sha,
        )
    assert not (staged / v5.V5_ROOT_RELATIVE).exists()


def test_physical_v5_composition_reuses_v4_cache_and_exact_v1_runner_without_loop_copy() -> None:
    source = inspect.getsource(physical_v5.PhysicalCommonStratumSmokeV5Backend.run_smoke)
    assert "self.smoke_runner.run" in source and "torch.optim" not in source and "optimizer.step" not in source
    factory = inspect.getsource(physical_v5.build_reviewed_v5_source_smoke_backend)
    assert "bootstrap_reviewed_v1_route" in factory and "TorchCSWGSmokeRunner" in factory
    provider = inspect.getsource(physical_v5.V5CachedCommonStratumSourceProvider.prepare)
    assert "v4_physical._cache_material" in provider and "build_common_stratum_prepared_audit" in provider


def test_physical_factory_uses_successor_closure_not_frozen_v2_v3_v4_current_closure_builders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, prepared: Any,
) -> None:
    """A V5 physical factory may compose historical helpers but never rebuild their closures.

    The isolated traps are test-only.  The production factory has no fallback
    or module mutation path; it uses V5's closure plus the held historical
    receipt graphs.
    """
    calls: list[str] = []

    def _forbidden(label: str) -> Callable[..., object]:
        def raise_if_called(*_args: object, **_kwargs: object) -> object:
            calls.append(label)
            raise AssertionError(f"unexpected frozen {label} current-closure rebuild")
        return raise_if_called

    monkeypatch.setattr(v2, "implementation_closure", _forbidden("v2"))
    monkeypatch.setattr(v3, "implementation_closure", _forbidden("v3"))
    monkeypatch.setattr(v4, "implementation_closure", _forbidden("v4"))
    v3_graph, v4_graph, _fallback_sha = _fake_graphs(prepared)
    backend = physical_v5.build_reviewed_v5_source_smoke_backend(
        root=ROOT,
        source_root=tmp_path / "never-opened-source-root",
        expected_v3_graph=v3_graph,
        expected_v4_graph=v4_graph,
    )
    assert isinstance(backend, physical_v5.PhysicalCommonStratumSmokeV5Backend)
    assert calls == []
    source = inspect.getsource(physical_v5.build_reviewed_v5_source_smoke_backend)
    assert all(token not in source for token in (
        "build_source_audit_v3_identity", "build_source_smoke_v4_identity", "implementation_closure",
    ))


def test_public_cli_is_static_and_no_cuda(tmp_path: Path) -> None:
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
        [sys.executable, "-c", "import sys; from tfpd_exploration.src.cross_session_worst_group_v1 import source_smoke_v5; print('torch' in sys.modules); print('src' in sys.modules)"],
        cwd=tmp_path, env=environment, text=True, capture_output=True, check=True,
    )
    assert probe.stdout.splitlines() == ["False", "False"]
    script = ROOT / "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_smoke_v5.py"
    dry = subprocess.run([sys.executable, str(script), "--dry-run"], cwd=tmp_path, env=environment,
                         text=True, capture_output=True, check=True)
    payload = json.loads(dry.stdout)
    assert payload["imports_torch"] is False and payload["creates_root_or_receipt"] is False
    denied = subprocess.run([sys.executable, str(script), "--execute"], cwd=tmp_path, env=environment,
                            text=True, capture_output=True)
    assert denied.returncode != 0 and "root-reviewed" in denied.stderr
