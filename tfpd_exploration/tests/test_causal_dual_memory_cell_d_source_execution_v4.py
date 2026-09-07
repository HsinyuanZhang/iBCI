"""Focused no-data/no-CUDA tests for CDM-D Source Execution V4."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import pytest


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration/src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.causal_dual_memory_cell_d_v1 import source_execute as v1  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_physical as physical_v1  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_physical_v2 as physical_v2  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_physical_v3 as physical_v3  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_physical_v4 as physical_v4  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_v2 as v2  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_v3 as v3  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_v4 as v4  # noqa: E402


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _profile() -> dict[str, object]:
    return dict(v1.COMPATIBLE_DEVICE_PROFILES["gpu1"])


def _roster() -> tuple[str, ...]:
    return (v1.SOURCE_SMOKE_SESSION, *(f"sub-C_ses-CO-{index:08d}" for index in range(26)))


def _write_pair(directory: Path, name: str, payload: Mapping[str, object]) -> str:
    body = v4._json_bytes(dict(payload))
    digest = _sha(body)
    (directory / name).write_bytes(body)
    (directory / f"{name}.sha256").write_bytes(f"{digest}  {name}\n".encode("ascii"))
    os.chmod(directory / name, 0o444)
    os.chmod(directory / f"{name}.sha256", 0o444)
    return digest


def _write_v3_predecessor(
    root: Path,
    *,
    mutate: str | None = None,
) -> v4.V3FailedPredecessorExpectation:
    """Create an exact schema-valid synthetic V3 failure graph under ``root``."""

    relative = "tfpd_exploration/results/synthetic_v3_failure"
    directory = root / relative
    directory.mkdir(parents=True)
    identity = {"closure": {"closure_sha256": "a" * 64}, "synthetic": True}
    identity_sha = _sha(v4._json_bytes(identity))
    attempt = {
        "schema": "causal_dual_memory_cell_d_source_execution_attempt_v3",
        "status": "ATTEMPT_RESERVED",
        "identity": identity,
        "source_only": True,
        "source_resolved_or_opened": False,
        "checkpoint_opened": False,
        "cuda_initialized": False,
    }
    attempt_sha = _write_pair(directory, "attempt.json", attempt)
    launch = {
        "schema": "causal_dual_memory_cell_d_source_execution_launch_v3",
        "status": "LAUNCHED",
        "identity": identity,
        "attempt_sha256": attempt_sha,
        "launch_closure_sha256": "a" * 64,
    }
    launch_sha = _write_pair(directory, "launch.json", launch)
    flags: dict[str, object] = {
        "stage": "prepare",
        "source_resolved": True,
        "source_opened": False,
        "checkpoint_opened": False,
        "cuda_initialized": False,
        "model_forward_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "normalizer_refit": False,
        "within_opened": False,
        "external_opened": False,
        "formal_opened": False,
        "target_opened": False,
        "oom_retry_attempted": False,
    }
    if mutate == "source_opened":
        flags["source_opened"] = True
    failure = {
        "schema": "causal_dual_memory_cell_d_source_execution_failure_v3",
        "status": "FAILED",
        "identity": identity if mutate != "identity" else {"closure": {"closure_sha256": "b" * 64}},
        "attempt_sha256": attempt_sha,
        "launch_sha256": launch_sha,
        "source_authority_sha256": None,
        "stage": "prepare",
        "error_class": "SourceExecutionError",
        "error_sha256": "e" * 64,
        "flags": flags,
        "terminal_published": False,
    }
    failure_sha = _write_pair(directory, "failure.json", failure)
    expectation = v4.V3FailedPredecessorExpectation(
        root_relative=relative,
        attempt_sha256=attempt_sha,
        launch_sha256=launch_sha,
        failure_sha256=failure_sha,
        identity_sha256=identity_sha,
        v3_closure_sha256="a" * 64,
        error_sha256="e" * 64,
    )
    if mutate == "extra":
        (directory / "extra.json").write_text("{}", encoding="utf-8")
        os.chmod(directory / "extra.json", 0o444)
    if mutate == "sidecar":
        os.chmod(directory / "failure.json.sha256", 0o644)
        (directory / "failure.json.sha256").write_text(f"{failure_sha}  forged.json\n", encoding="ascii")
        os.chmod(directory / "failure.json.sha256", 0o444)
    if mutate == "mode":
        os.chmod(directory / "launch.json", 0o644)
    return expectation


def _stage_current_closure(tmp_path: Path) -> None:
    """Copy only explicit closure leaves into a disposable no-data tree."""

    for relative in (*v4._V4_INHERITED_PATHS, *v4._V4_OWNED_PATHS):
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        os.chmod(target, 0o644)
    (tmp_path / "tfpd_exploration/results").mkdir(parents=True, exist_ok=True)


def _identity(stage: Path, expectation: v4.V3FailedPredecessorExpectation) -> v4.SourceExecutionV4Identity:
    return v4.SourceExecutionV4Identity(
        spec=v4.SourceExecutionV4Spec(
            "source_smoke", "tfpd_exploration/results/v4_synthetic_smoke", v1.SOURCE_SMOKE_SESSION, 10, (10, 11),
        ),
        closure=v4.execution_closure_payload(stage, predecessor=expectation),
        strict_train_roster=_roster(),
        fixed_assets={asset.label: asset.payload() for asset in v1.FIXED_ASSETS},
        normalizers=v1.SEALED_NORMALIZERS,
        selected_device=_profile(),
        v3_failed_predecessor=expectation,
    )


def test_workorder_explicit_current_closure_and_v3_semantic_spec() -> None:
    closure = v4.execution_closure_payload(ROOT)
    paths = [row["path"] for row in closure["paths"]]
    assert v4.WORKORDER_SHA256 == "b56ca00f932358655e1038490f39e9c304f5e08f9ff6874d1f68f62515a8b879"
    assert len(paths) == len(set(paths)) == len(v4._V4_INHERITED_PATHS) + len(v4._V4_OWNED_PATHS)
    assert paths[-5:] == list(v4._V4_OWNED_PATHS)
    assert "tfpd_exploration/src/tfpd_lane/pop_robust.py" in paths
    assert "tfpd_exploration/src/tfpd_lane/arm_common.py" in paths
    assert v4.SOURCE_SMOKE_SPEC.payload()["smoke"] == v3.SOURCE_SMOKE_SPEC.payload()["smoke"]
    assert v4.SOURCE_GATE_SPEC.payload()["fail_fast_budget_order"] == [30, 10, 4]
    assert "glob" not in inspect.getsource(v4.execution_closure_payload).lower()


def test_default_v1_loader_remains_historical_and_current_v3_core_fails_closed() -> None:
    """The injected successor does not relax the V1/V2 default closure path."""

    with pytest.raises(v1.SourceExecutionError, match="accepted Stage-0 dependency byte drift"):
        v1.execution_closure_payload(ROOT)
    with pytest.raises(v1.SourceExecutionError, match="accepted Stage-0 dependency byte drift"):
        physical_v1._load_closure_bound_module(
            ROOT,
            relative="tfpd_exploration/src/tfpd_lane/arm_common.py",
            module_name="v4_default_historical_reject",
        )
    assert "v4_default_historical_reject" not in sys.modules


def test_injected_loader_rehashes_exact_bytes_before_private_execution(tmp_path: Path) -> None:
    relative = "private/runtime_helper.py"
    helper = tmp_path / relative
    helper.parent.mkdir(parents=True)
    body = b"EXECUTED_VALUE = 17\n"
    helper.write_bytes(body)
    builder_identity = physical_v1.execute.execution_closure_payload

    def builder(root: Path) -> dict[str, object]:
        assert root == tmp_path
        return {"paths": [{"path": relative, "sha256": _sha(body)}]}

    name = "cdmd_v4_test_private_helper"
    try:
        module = physical_v1._load_closure_bound_module(
            tmp_path, relative=relative, module_name=name, closure_builder=builder,
        )
        assert module.EXECUTED_VALUE == 17
        assert physical_v1.execute.execution_closure_payload is builder_identity
    finally:
        sys.modules.pop(name, None)
    helper.write_bytes(b"EXECUTED_VALUE = 99\n")
    with pytest.raises(physical_v1.SourceExecutionPhysicalError, match="bytes drift"):
        physical_v1._load_closure_bound_module(
            tmp_path, relative=relative, module_name=name, closure_builder=builder,
        )
    assert name not in sys.modules


@pytest.mark.parametrize("kind", ("missing", "duplicate", "symlink", "parent_swap"))
def test_loader_rejects_bad_closure_or_unsafe_path_before_execution(tmp_path: Path, kind: str) -> None:
    relative = "runtime/helper.py"
    helper = tmp_path / relative
    helper.parent.mkdir(parents=True)
    body = b"EXECUTED = True\n"
    helper.write_bytes(body)
    name = f"cdmd_v4_bad_{kind}"

    def builder(_root: Path) -> dict[str, object]:
        if kind == "missing":
            return {"paths": []}
        if kind == "duplicate":
            return {"paths": [{"path": relative, "sha256": _sha(body)}] * 2}
        if kind == "parent_swap":
            replacement = tmp_path / "replacement"
            replacement.mkdir(exist_ok=True)
            (replacement / "helper.py").write_bytes(b"EXECUTED = 'forged'\n")
            os.replace(helper.parent, tmp_path / "old_runtime")
            os.replace(replacement, helper.parent)
        return {"paths": [{"path": relative, "sha256": _sha(body)}]}

    if kind == "symlink":
        helper.unlink()
        target = tmp_path / "target.py"
        target.write_bytes(body)
        helper.symlink_to(target)
    with pytest.raises(physical_v1.SourceExecutionPhysicalError):
        physical_v1._load_closure_bound_module(
            tmp_path, relative=relative, module_name=name, closure_builder=builder,
        )
    assert name not in sys.modules
    source = inspect.getsource(physical_v1._load_closure_bound_module)
    assert "after_named" in source and "after_held" in source and "O_NOFOLLOW" in source


def test_v4_provider_and_executor_are_the_only_runtime_closure_injection_seams(tmp_path: Path) -> None:
    source_cap = v1._issue_root_reviewed_source_data_capability(
        canonical_root=tmp_path / "strict_source", strict_train_roster=_roster(), seal=v1._ROOT_REVIEW_SEAL,
    )
    backend = physical_v4.build_reviewed_physical_backend(
        root=ROOT, source_data=source_cap, selected_device=_profile(),
    )
    assert type(backend.provider) is physical_v4.V4ThetaStrict27SessionProvider
    assert type(backend.executor) is physical_v4.V4IndependentActivityCellDFourGroupExecutor
    assert isinstance(backend.provider, physical_v2.V2ThetaStrict27SessionProvider)
    assert isinstance(backend.executor, physical_v3.IndependentActivityCellDFourGroupExecutor)
    provider_source = inspect.getsource(physical_v4.V4ThetaStrict27SessionProvider._runtime_modules)
    executor_source = inspect.getsource(physical_v4.V4IndependentActivityCellDFourGroupExecutor._runtime_modules)
    assert "closure_builder=v4.execution_closure_payload" in provider_source
    assert "closure_builder=v4.execution_closure_payload" in executor_source
    assert "monkeypatch" not in inspect.getsource(physical_v4).lower()
    assert not (tmp_path / "strict_source").exists()


@pytest.mark.parametrize("mutate", (None, "extra", "sidecar", "mode", "identity", "source_opened"))
def test_v3_failed_predecessor_is_held_descriptor_exact_and_semantic(
    tmp_path: Path, mutate: str | None,
) -> None:
    expectation = _write_v3_predecessor(tmp_path, mutate=mutate)
    if mutate is None:
        evidence = v4.validate_v3_failed_predecessor(tmp_path, expectation)
        assert evidence["predecessor"] == expectation.payload()
        return
    with pytest.raises(v4.SourceExecutionV4Error):
        v4.validate_v3_failed_predecessor(tmp_path, expectation)


def test_v4_capability_rechecks_held_predecessor_current_closure_environment_and_fresh_root(tmp_path: Path) -> None:
    _stage_current_closure(tmp_path)
    expectation = _write_v3_predecessor(tmp_path)
    identity = _identity(tmp_path, expectation)
    environment = {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    capability = v4._issue_root_reviewed_capability(
        tmp_path, identity, source_data_root=None, seal=v4._ROOT_REVIEW_SEAL, environ=environment,
    )
    assert capability.identity_sha256 == identity.sha256
    assert not (tmp_path / identity.spec.root_relative).exists()
    with pytest.raises(v4.SourceExecutionV4Error, match="CUDA_VISIBLE_DEVICES"):
        v4._issue_root_reviewed_capability(
            tmp_path, identity, source_data_root=None, seal=v4._ROOT_REVIEW_SEAL,
            environ={"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
        )
    assert not (tmp_path / identity.spec.root_relative).exists()


class _FailingPrePrepareBackend:
    def __init__(self) -> None:
        self.events: list[str] = []

    def preflight(self, *, root, identity, flags):
        self.events.append("preflight")
        return {"source_resolved_or_opened": False, "checkpoint_opened": False, "cuda_initialized": False}

    def prepare(self, *, root, identity, flags):
        self.events.append("prepare")
        assert (Path(root) / identity.spec.root_relative / "attempt.json").is_file()
        assert (Path(root) / identity.spec.root_relative / "launch.json").is_file()
        raise RuntimeError("synthetic V4 prepare failure")

    def close(self, runtime):
        self.events.append("close")


def _runtime_attestation() -> dict[str, object]:
    return {
        **_profile(),
        "visible_devices": 1,
        "attested": True,
        "torch_cuda_matmul_allow_tf32": False,
        "torch_cudnn_allow_tf32": False,
    }


def _sealed_swa_proof() -> dict[str, object]:
    return {
        "schema": v1.SEALED_SWA_LOAD_PROOF_SCHEMA,
        "sealed_terminal_sha256": v1.SEALED_CELL_D_TERMINAL_SHA256,
        "sealed_swa_sha256": v1.SEALED_CELL_D_SWA_SHA256,
        "fresh_strict_load": True,
        "recomputed_state_dict_sha256": "c" * 64,
        "initialized_trainable_parameters": v1.SEALED_CELL_D_INITIALIZED_TRAINABLE_PARAMETERS,
        "uninitialized_lazy_keys": list(v1.SEALED_CELL_D_UNINITIALIZED_LAZY_KEYS),
        "model_eval": True,
        "no_grad": True,
        "finite_forward": True,
        "repeated_fixed_forward_bitwise_equal": True,
        "model_state_unchanged": True,
        "dynamic_dropout_calls": 0,
    }


def _theta_proof() -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_theta_raw_proof_v2",
        "session": v1.SOURCE_SMOKE_SESSION,
        "hash_law": v2.THETA_RAW_HASH_LAW,
        "hash_law_expression": v2.THETA_RAW_HASH_LAW_TEXT,
        "raw_t4_shape": [7, 4],
        "raw_t4_dtype": "float32",
        "raw_t4_all_finite": True,
        "raw_t4_sha256": "1" * 64,
        "sealed_raw_t4_sha256": "1" * 64,
        "theta_float64_sha256": "2" * 64,
        "sealed_theta_float64_sha256": "2" * 64,
        "theta_atan2_float64_bitwise_equal": True,
        "valid_mask_sha256": "3" * 64,
        "sealed_valid_mask_sha256": "3" * 64,
        "validity_raw_m_gt_modulation_eps_bitwise_equal": True,
        "known_invalid_unit_count": 0,
        "canonical_unit_order_exact": True,
        "canonical_unit_order_sha256": "4" * 64,
    }


def _committing_trace(
    *,
    trial_id: str,
    state_before: str,
    state_after: str,
    activity_before: str,
    activity_after: str,
    carrier_before: str,
    carrier_after: str,
    carrier_committed: bool,
) -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "budget": 10,
        "schema": "causal_dual_memory_independent_activity_outcome_v3",
        "activity_transition_committed": True,
        "activity_fifo_changed": True,
        "carrier_transition_committed": carrier_committed,
        "carrier_rejection_reason_or_null": None if carrier_committed else "movement_too_short",
        "activity_rejection_reason_or_null": None,
        "state_before_sha256": state_before * 64,
        "state_after_sha256": state_after * 64,
        "activity_before_sha256": activity_before * 64,
        "activity_after_sha256": activity_after * 64,
        "carrier_before_sha256": carrier_before * 64,
        "carrier_after_sha256": carrier_after * 64,
    }


class _SuccessfulSmokeBackend:
    """Complete synthetic lifecycle backend; it opens neither data nor CUDA."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.runtime = object()

    def preflight(self, *, root, identity, flags):
        self.events.append("preflight")
        return {"source_resolved_or_opened": False, "checkpoint_opened": False, "cuda_initialized": False}

    def prepare(self, *, root, identity, flags):
        self.events.append("prepare")
        flags.source_resolved = flags.source_opened = flags.checkpoint_opened = flags.cuda_initialized = True
        return self.runtime

    def source_authority(self, runtime, *, identity, flags):
        self.events.append("source_authority")
        proof = _theta_proof()
        topology = {
            "session_id": v1.SOURCE_SMOKE_SESSION,
            "source_descriptor_sha256": "a" * 64,
            "raw_t4_channel_order_sha256": proof["canonical_unit_order_sha256"],
            "theta_valid_mask_sha256": proof["valid_mask_sha256"],
            "total_unit_count": 7,
            "valid_unit_count": 7,
            "invalid_unit_count": 0,
            "valid_mask_sha256": proof["valid_mask_sha256"],
            "invalid_assignment_is_minus_one": True,
            "theta_authority_binds_only_validity_not_budget_groups": True,
        }
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_authority_v1",
            "cell": v1.CELL,
            "identity_sha256": identity.sha256,
            "strict_train_roster": list(identity.strict_train_roster),
            "strict_train_roster_sha256": v1.roster_sha256(identity.strict_train_roster),
            "normalizers": identity.normalizers.payload(),
            "fixed_assets": {asset.label: asset.payload() for asset in v1.FIXED_ASSETS},
            "runtime_environment": _runtime_attestation(),
            "sealed_swa_load_proof": _sealed_swa_proof(),
            "physical_session_count": 1,
            "strict_manifest_bound_without_nonphysical_resolution": True,
            "sessions": [topology],
            "v2_theta_raw_proofs": [proof],
            "source_only": True,
            "access": {
                "source_opened": True, "within_opened": False, "external_opened": False,
                "formal_opened": False, "target_opened": False, "optimizer_steps": 0,
                "backward_calls": 0, "parameter_updates": 0, "normalizer_refit": False,
            },
        }

    def run_smoke(self, runtime, *, identity, flags):
        self.events.append("smoke")
        flags.model_forward_calls += 8
        trace = [
            _committing_trace(
                trial_id="smoke-10", state_before="1", state_after="2",
                activity_before="3", activity_after="4", carrier_before="5", carrier_after="5",
                carrier_committed=False,
            ),
            _committing_trace(
                trial_id="smoke-11", state_before="2", state_after="6",
                activity_before="4", activity_after="7", carrier_before="5", carrier_after="8",
                carrier_committed=True,
            ),
        ]
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_smoke_v3",
            "session": v1.SOURCE_SMOKE_SESSION,
            "budget": 10,
            "support_positions": list(range(10)),
            "audit_positions": [10, 11],
            "support_trial_ids": [f"support-{index}" for index in range(10)],
            "audit_trial_ids": ["smoke-10", "smoke-11"],
            "group_count": 4,
            "all_four_groups_finalized": True,
            "b8_threshold_applied": False,
            "required_activity_transition_count": 2,
            "activity_fifo_capacity": 20,
            "activity_transition_committed_count": 2,
            "carrier_transition_committed_count": 1,
            "carrier_transition_rejected_count": 1,
            "budget_initial_carrier_recipe": "fixed_ridge_by_trial",
            "budget_initial_carrier_support_rows": 10,
            "raw_m30_t4_used_as_initializer": False,
            "initial_support_rate_domain": v1.INITIAL_SUPPORT_RATE_DOMAIN,
            "online_update_rate_domain": v1.ONLINE_UPDATE_RATE_DOMAIN,
            "initial_support_rates_sha256": "4" * 64,
            "initial_support_exposure_seconds_sha256": "5" * 64,
            "budget_initial_carrier_parity": {"mode": "fixed_ridge_by_trial"},
            "budget_initial_carrier_sha256": "6" * 64,
            "budget_groups_sha256": "7" * 64,
            "budget_group_assignment_sha256": "8" * 64,
            "budget_group_valid_mask_sha256": "9" * 64,
            "v3_independent_activity_transitions": trace,
            "source_only": True,
        }

    def revalidate(self, *, root, identity, flags):
        self.events.append("revalidate")

    def resources(self, runtime, *, flags):
        self.events.append("resources")
        return {
            "endpoint_chunks_per_s": 2.0, "trials_per_s": 1.0, "wall_seconds": 1.0,
            "endpoint_chunks_completed": 2, "completed_trials": 1, "rss_bytes": 1,
            "current_cuda_allocated_bytes": 4, "current_cuda_reserved_bytes": 8,
            "peak_cuda_allocated_bytes": 4, "peak_cuda_reserved_bytes": 8,
            "selected_device": _profile(),
        }

    def close(self, runtime):
        self.events.append("close")


def test_v4_failure_lifecycle_is_attempt_then_launch_then_honest_failure(tmp_path: Path) -> None:
    _stage_current_closure(tmp_path)
    expectation = _write_v3_predecessor(tmp_path)
    identity = _identity(tmp_path, expectation)
    capability = v4._issue_root_reviewed_capability(
        tmp_path, identity, source_data_root=None, seal=v4._ROOT_REVIEW_SEAL,
        environ={"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
    )
    backend = _FailingPrePrepareBackend()
    with pytest.raises(RuntimeError, match="synthetic V4 prepare failure"):
        v4.execute_authorized(
            tmp_path, identity=identity, capability=capability, backend=backend,
            environ={"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
        )
    output = tmp_path / identity.spec.root_relative
    assert {item.name for item in output.iterdir()} == {
        "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
        "failure.json", "failure.json.sha256",
    }
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["stage"] == "prepare"
    assert failure["source_authority_sha256"] is None
    assert failure["flags"]["source_opened"] is False
    assert failure["v4_binding"]["v3_failed_predecessor"] == expectation.payload()
    assert backend.events == ["preflight", "prepare", "close"]


def test_v4_complete_success_lifecycle_revalidates_live_held_root_not_prospective_absence(tmp_path: Path) -> None:
    _stage_current_closure(tmp_path)
    expectation = _write_v3_predecessor(tmp_path)
    identity = _identity(tmp_path, expectation)
    environment = {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    capability = v4._issue_root_reviewed_capability(
        tmp_path, identity, source_data_root=None, seal=v4._ROOT_REVIEW_SEAL, environ=environment,
    )
    backend = _SuccessfulSmokeBackend()
    result = v4.execute_authorized(
        tmp_path, identity=identity, capability=capability, backend=backend, environ=environment,
    )
    assert result["status"] == "SMOKE_COMPLETED"
    assert backend.events == ["preflight", "prepare", "source_authority", "smoke", "revalidate", "resources", "close"]
    output = tmp_path / identity.spec.root_relative
    expected_pairs = {"attempt.json", "launch.json", "source_authority.json", "smoke.json", "terminal.json"}
    assert {item.name for item in output.iterdir()} == expected_pairs | {f"{name}.sha256" for name in expected_pairs}
    assert not (output / "failure.json").exists()
    pair_sha = {}
    for name in expected_pairs:
        body = (output / name).read_bytes()
        digest = _sha(body)
        pair_sha[name] = digest
        assert (output / f"{name}.sha256").read_bytes() == f"{digest}  {name}\n".encode("ascii")
    terminal_body = (output / "terminal.json").read_bytes()
    terminal = json.loads(terminal_body)
    assert _sha(terminal_body) == result["terminal_sha256"]
    assert pair_sha["terminal.json"] == result["terminal_sha256"]
    assert terminal["attempt_sha256"] == pair_sha["attempt.json"]
    assert terminal["launch_sha256"] == pair_sha["launch.json"]
    assert terminal["source_authority_sha256"] == pair_sha["source_authority.json"]
    assert terminal["evidence_sha256s"] == {"smoke.json": pair_sha["smoke.json"]}
    assert terminal["launch_closure_sha256"] == terminal["final_closure_sha256"] == identity.closure["closure_sha256"]
    assert terminal["v4_binding"]["v3_failed_predecessor"] == expectation.payload()


def test_v4_live_revalidation_rejects_named_root_substitution_after_reserve(tmp_path: Path) -> None:
    _stage_current_closure(tmp_path)
    expectation = _write_v3_predecessor(tmp_path)
    identity = _identity(tmp_path, expectation)
    environment = {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    capability = v4._issue_root_reviewed_capability(
        tmp_path, identity, source_data_root=None, seal=v4._ROOT_REVIEW_SEAL, environ=environment,
    )
    artifact = v1.reserve_artifact_root(tmp_path, spec=identity.spec, roster=identity.strict_train_roster)
    try:
        attempt_sha = artifact.publish_json_pair("attempt.json", {"synthetic": True})
        original = artifact.directory
        os.rename(original, tmp_path / "held-original-root")
        original.mkdir()
        with pytest.raises(v4.SourceExecutionV4Error, match="identity drift"):
            v4._validate_live_final_revalidation(
                tmp_path, identity, capability, artifact, environment,
                expected_published_sha256s={"attempt.json": attempt_sha},
            )
    finally:
        artifact.close()


def test_v4_live_revalidation_rejects_replaced_self_consistent_pair_after_reserve(tmp_path: Path) -> None:
    """The publication-returned digest, not merely the sidecar, binds live bytes."""

    _stage_current_closure(tmp_path)
    expectation = _write_v3_predecessor(tmp_path)
    identity = _identity(tmp_path, expectation)
    environment = {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    capability = v4._issue_root_reviewed_capability(
        tmp_path, identity, source_data_root=None, seal=v4._ROOT_REVIEW_SEAL, environ=environment,
    )
    artifact = v1.reserve_artifact_root(tmp_path, spec=identity.spec, roster=identity.strict_train_roster)
    try:
        attempt_sha = artifact.publish_json_pair("attempt.json", {"synthetic": True})
        forged_body = v4._json_bytes({"synthetic": "replaced-but-self-consistent"})
        forged_digest = _sha(forged_body)
        replacement_body = tmp_path / "replacement-attempt.json"
        replacement_sidecar = tmp_path / "replacement-attempt.json.sha256"
        replacement_body.write_bytes(forged_body)
        replacement_sidecar.write_bytes(f"{forged_digest}  attempt.json\n".encode("ascii"))
        os.chmod(replacement_body, 0o444)
        os.chmod(replacement_sidecar, 0o444)
        os.replace(replacement_body, artifact.directory / "attempt.json")
        os.replace(replacement_sidecar, artifact.directory / "attempt.json.sha256")
        with pytest.raises(v4.SourceExecutionV4Error, match="body SHA drift"):
            v4._validate_live_final_revalidation(
                tmp_path, identity, capability, artifact, environment,
                expected_published_sha256s={"attempt.json": attempt_sha},
            )
    finally:
        artifact.close()


def test_v4_smoke_and_v3_trace_contract_remain_exactly_reachable() -> None:
    assert v4.SOURCE_SMOKE_SPEC.smoke_budget == 10
    assert v4.SOURCE_SMOKE_SPEC.smoke_audit_positions == (10, 11)
    trace = [
        {
            "trial_id": "trial-10", "budget": 10,
            "schema": "causal_dual_memory_independent_activity_outcome_v3",
            "activity_transition_committed": True, "activity_fifo_changed": True,
            "carrier_transition_committed": False, "carrier_rejection_reason_or_null": "movement_too_short",
            "activity_rejection_reason_or_null": None,
            "state_before_sha256": "1" * 64, "state_after_sha256": "2" * 64,
            "activity_before_sha256": "3" * 64, "activity_after_sha256": "4" * 64,
            "carrier_before_sha256": "5" * 64, "carrier_after_sha256": "5" * 64,
        },
        {
            "trial_id": "trial-11", "budget": 10,
            "schema": "causal_dual_memory_independent_activity_outcome_v3",
            "activity_transition_committed": True, "activity_fifo_changed": True,
            "carrier_transition_committed": True, "carrier_rejection_reason_or_null": None,
            "activity_rejection_reason_or_null": None,
            "state_before_sha256": "2" * 64, "state_after_sha256": "6" * 64,
            "activity_before_sha256": "4" * 64, "activity_after_sha256": "7" * 64,
            "carrier_before_sha256": "5" * 64, "carrier_after_sha256": "8" * 64,
        },
    ]
    v3._validate_transition_trace(
        trace, budget=10, expected_trial_ids=("trial-10", "trial-11"), require_offline_m30=False, smoke=True,
    )
    broken = [dict(row) for row in trace]
    broken[1]["state_before_sha256"] = "f" * 64
    with pytest.raises(v3.SourceExecutionV3Error, match="digest chain"):
        v3._validate_transition_trace(
            broken, budget=10, expected_trial_ids=("trial-10", "trial-11"), require_offline_m30=False, smoke=True,
        )


def test_static_cli_is_torch_free_dry_and_public_execution_fails_closed() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate_v4.py"
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    dry = subprocess.run(
        [sys.executable, "-I", str(script), "--dry-run"], cwd=ROOT, text=True,
        capture_output=True, check=True, env=environment,
    )
    payload = json.loads(dry.stdout)
    assert payload["workorder_sha256"] == v4.WORKORDER_SHA256
    assert payload["opens_source"] is payload["loads_checkpoint"] is payload["initializes_cuda"] is False
    blocked = subprocess.run(
        [sys.executable, "-I", str(script), "--execute", "--source-smoke"], cwd=ROOT, text=True,
        capture_output=True, env=environment,
    )
    assert blocked.returncode != 0 and "root-reviewed in-process V4 capability" in blocked.stderr
    probe = (
        "import pathlib,sys; root=pathlib.Path.cwd()/'tfpd_exploration'; "
        "sys.path.insert(0,str(root)); import src.causal_dual_memory_cell_d_v1.source_execute_v4; "
        "print('torch' in sys.modules)"
    )
    imported = subprocess.run(
        [sys.executable, "-I", "-c", probe], cwd=ROOT, text=True,
        capture_output=True, check=True, env=environment,
    )
    assert imported.stdout.strip() == "False"
    assert "import torch" not in script.read_text(encoding="utf-8")


def test_owned_source_contains_no_global_loader_or_package_monkeypatch() -> None:
    source = inspect.getsource(physical_v4)
    assert "setattr(" not in source
    assert "sys.modules" not in source
    assert "monkeypatch" not in source
    assert physical_v1.ConcreteStrict27SessionProvider._runtime_modules is not physical_v4.V4ThetaStrict27SessionProvider._runtime_modules
    assert physical_v1.ConcreteCellDFourGroupExecutor._runtime_modules is not physical_v4.V4IndependentActivityCellDFourGroupExecutor._runtime_modules
