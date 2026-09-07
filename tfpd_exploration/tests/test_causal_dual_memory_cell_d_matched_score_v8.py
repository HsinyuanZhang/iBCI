"""Focused no-data/no-CUDA tests for the CDM-D V8 helper successor."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from dataclasses import replace
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.causal_dual_memory_cell_d_score_v1 import physical as v1physical
from src.causal_dual_memory_cell_d_score_v1 import plan as v1plan
from src.causal_dual_memory_cell_d_score_v1 import score as v1score
from src.causal_dual_memory_cell_d_score_v5 import physical as v5physical
from src.causal_dual_memory_cell_d_score_v7 import physical as v7physical
from src.causal_dual_memory_cell_d_score_v7 import score as v7score
from src.causal_dual_memory_cell_d_score_v8 import physical, plan, score


ROOT = Path(__file__).resolve().parents[2]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identity() -> plan.ScoreIdentity:
    rows = {path: _sha(f"v8:{path}") for path in plan.IMPLEMENTATION_PATHS}
    rows[plan.WORKORDER_RELATIVE] = plan.WORKORDER_SHA256
    return plan.ScoreIdentity(plan.ImplementationClosure(rows).payload())


def _v7_test_support() -> object:
    """Load only synthetic V7 fixtures; it never resolves a live artifact."""
    location = ROOT / "tfpd_exploration/tests/test_causal_dual_memory_cell_d_matched_score_v7.py"
    specification = importlib.util.spec_from_file_location("_v8_v7_synthetic_support", location)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_pair(directory: Path, name: str, payload: object) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    (directory / name).write_bytes(body)
    (directory / f"{name}.sha256").write_bytes(f"{digest}  {name}\n".encode("ascii"))
    os.chmod(directory / name, 0o444)
    os.chmod(directory / f"{name}.sha256", 0o444)
    return digest


def _synthetic_v7_helper_failure(tmp_path: Path) -> plan.V7PhysicalHelperFailureContract:
    """Create an actual-shaped held V7 three-pair graph, not schema mocks."""
    support = _v7_test_support()
    identity = support._identity()
    binding = support._source_gate_binding(prefix="v8-helper-history")
    authority, fixed = support._input_authority()
    authority_root = tmp_path / plan.V7_PHYSICAL_HELPER_FAILURE.authority_root_relative
    failed_root = tmp_path / plan.V7_PHYSICAL_HELPER_FAILURE.score_root_relative
    authority_root.mkdir(parents=True)
    failed_root.mkdir(parents=True)
    os.chmod(failed_root, 0o755)
    preflight = v7score.build_target_free_preflight(
        root=tmp_path, identity=identity, source_gate=binding, fixed_authority=fixed,
    )
    pre_sha = _write_pair(authority_root, "official_preflight.json", preflight)
    authorization = v7score.build_root_authorization(official_preflight_sha256=pre_sha, preflight=preflight)
    auth_sha = _write_pair(authority_root, "root_authorization.json", authorization)
    attempt = v7score._attempt_payload(identity, pre_sha, auth_sha)
    attempt_sha = _write_pair(failed_root, "attempt.json", attempt)
    input_payload = authority.payload(identity=identity)
    input_sha = _write_pair(failed_root, "input_authority.json", input_payload)
    failure = v7score._failure_payload(
        identity, attempt_sha, input_sha, "budget_m30",
        AttributeError("source_execute_physical_v5._variable_prefix_array_digest"),
        {
            "within_assets_opened": True, "external_assets_opened": True,
            "checkpoint_opened": True, "cuda_initialized": True,
            "full_system_forward_count": 4238, "group_forward_count": 0,
        },
    )
    failure_sha = _write_pair(failed_root, "failure.json", failure)
    info = failed_root.stat()
    return plan.V7PhysicalHelperFailureContract(
        authority_root_relative=plan.V7_PHYSICAL_HELPER_FAILURE.authority_root_relative,
        preflight_sha256=pre_sha, authorization_sha256=auth_sha,
        identity_sha256=score._digest(identity.payload()),
        implementation_closure_sha256=identity.payload()["closure"]["closure_sha256"],
        score_root_relative=plan.V7_PHYSICAL_HELPER_FAILURE.score_root_relative,
        score_directory_device=int(info.st_dev), score_directory_inode=int(info.st_ino),
        score_directory_mode=stat.S_IMODE(info.st_mode),
        attempt_sha256=attempt_sha, input_authority_sha256=input_sha, failure_sha256=failure_sha,
        launch_log_sha256=_sha("synthetic-v7-helper-launch"), failure_stage="budget_m30",
        failure_class="AttributeError", failure_error_sha256=str(failure["error_sha256"]),
    )


def _actual_helper_runtime() -> tuple[physical.V8ReviewedCDMScoreRuntime, object, object]:
    profile = dict(v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])
    runtime = physical.V8ReviewedCDMScoreRuntime(root=ROOT, selected_device_profile=profile)
    wrapper = importlib.import_module("src.causal_dual_memory_cell_d_v1.source_execute_physical_v5")
    helper = importlib.import_module("src.causal_dual_memory_cell_d_v1.source_execute_physical")
    runtime._v5_runtime_sha256_by_path = {
        runtime._V1_HELPER_RELATIVE: plan._read_regular_no_follow(ROOT / runtime._V1_HELPER_RELATIVE),
    }
    return runtime, wrapper, helper


def test_v1_default_helper_hook_preserves_historical_single_module_law() -> None:
    profile = dict(v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])
    runtime = v1physical.ReviewedCDMScoreRuntime(root=ROOT, selected_device_profile=profile)
    module = object()
    assert runtime._source_physical_helper_module(module) is module


def test_real_v5_wrapper_has_no_helper_reexports_and_v8_authenticates_exact_v1_dependency() -> None:
    runtime, wrapper, helper = _actual_helper_runtime()
    assert "_variable_prefix_array_digest" not in vars(wrapper)
    assert "ConcreteCellDFourGroupExecutor" not in vars(wrapper)
    assert "_normalized_active_t4" not in vars(wrapper)
    assert "_torch_variable_prefix_forward" not in vars(wrapper)
    assert vars(wrapper)["v1_physical"] is helper
    selected = runtime._source_physical_helper_module(wrapper)
    assert selected is helper
    assert callable(vars(selected)["_variable_prefix_array_digest"])
    executor_type = vars(selected)["ConcreteCellDFourGroupExecutor"]
    assert callable(vars(executor_type)["_normalized_active_t4"])
    assert callable(vars(executor_type)["_torch_variable_prefix_forward"])
    assert getattr(runtime, "state", None) is None


@pytest.mark.parametrize("fault", ("object", "name", "path", "sha", "reexport", "class"))
def test_v8_helper_authentication_rejects_forgery_before_prepare(fault: str, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, wrapper, helper = _actual_helper_runtime()
    if fault == "object":
        forged = SimpleNamespace(__name__=wrapper.__name__, v1_physical=object())
        candidate = forged
    elif fault == "name":
        candidate = SimpleNamespace(__name__="forged.wrapper", v1_physical=helper)
    elif fault == "path":
        monkeypatch.setattr(helper, "__file__", str(ROOT / "forged.py"))
        candidate = wrapper
    elif fault == "sha":
        runtime._v5_runtime_sha256_by_path[runtime._V1_HELPER_RELATIVE] = _sha("wrong")
        candidate = wrapper
    elif fault == "reexport":
        candidate = SimpleNamespace(
            __name__=wrapper.__name__, v1_physical=helper,
            _variable_prefix_array_digest=vars(helper)["_variable_prefix_array_digest"],
        )
    else:
        original = vars(helper)["ConcreteCellDFourGroupExecutor"]
        monkeypatch.setattr(helper, "ConcreteCellDFourGroupExecutor", type("Forged", (), {}))
        candidate = wrapper
        assert original is not vars(helper)["ConcreteCellDFourGroupExecutor"]
    with pytest.raises(physical.PhysicalV8ScoreError):
        runtime._source_physical_helper_module(candidate)
    assert runtime.state is None


def test_v8_dispatches_digest_normalization_and_variable_prefix_to_helper_while_v5_executor_role_is_distinct() -> None:
    calls: list[tuple[str, object]] = []

    class HelperExecutor:
        @staticmethod
        def _normalized_active_t4(runtime_state: object, raw: object) -> object:
            calls.append(("normalized", (runtime_state, raw)))
            return SimpleNamespace(detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: "normalized")))

        @staticmethod
        def _torch_variable_prefix_forward(runtime_state: object, **kwargs: object) -> tuple[str, dict[str, object]]:
            calls.append(("forward", (runtime_state, kwargs)))
            return "prediction", {"forward_chunk_count": 1}

    def digest(value: object) -> str:
        calls.append(("digest", value))
        return "a" * 64

    helper = SimpleNamespace(
        _variable_prefix_array_digest=digest,
        ConcreteCellDFourGroupExecutor=HelperExecutor,
    )
    wrapper = importlib.import_module("src.causal_dual_memory_cell_d_v1.source_execute_physical_v5")
    real_executor = vars(wrapper)["V5OneShotFinalizedRowExecutor"](root=ROOT)
    activity = np.zeros((4, 100, 3), dtype=np.float32)
    state = SimpleNamespace(
        source_physical=wrapper, source_physical_helpers=helper,
        executor=real_executor, executor_state="executor-state",
        forward_chunks=0, closed=False,
    )
    runtime = physical.V8ReviewedCDMScoreRuntime(
        root=ROOT, selected_device_profile=dict(v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"]),
    )
    runtime.state = state
    groups = SimpleNamespace(held_mask=lambda group: f"mask-{group}")
    memory = SimpleNamespace(
        read_prediction_inputs=lambda: SimpleNamespace(activity_trials=activity, active_t4="raw-t4"),
        state=SimpleNamespace(
            carrier=SimpleNamespace(config=SimpleNamespace(support_budget_m=4), groups=groups),
            activity=SimpleNamespace(query_count=0),
        ),
    )
    trial = SimpleNamespace(neural_windows="neural", velocity_validity="validity")
    predictions, evidence = runtime._group_predictions(trial=trial, memory=memory)
    assert predictions == ("prediction",) * 4
    assert len(evidence) == 4 and state.forward_chunks == 8
    assert [name for name, _value in calls] == ["normalized", "digest", "forward", "forward", "forward", "forward"]
    assert all(payload[1]["expected_prefix_activity_sha256"] == "a" * 64 for name, payload in calls if name == "forward")
    assert type(state.executor) is vars(wrapper)["V5OneShotFinalizedRowExecutor"]
    assert type(state.executor) is not HelperExecutor


def test_v8_reuses_v5_one_shot_executor_and_not_v1_executor() -> None:
    runtime, wrapper, helper = _actual_helper_runtime()
    executor = runtime._build_source_executor(wrapper)
    assert type(executor) is vars(wrapper)["V5OneShotFinalizedRowExecutor"]
    assert type(executor) is not vars(helper)["ConcreteCellDFourGroupExecutor"]
    assert isinstance(runtime, v7physical.V7ReviewedCDMScoreRuntime)
    assert isinstance(runtime, v5physical.V5ReviewedCDMScoreRuntime)


def test_v8_helper_selection_is_before_checkpoint_and_v5_executor_construction_in_real_prepare_order() -> None:
    runtime, wrapper, helper = _actual_helper_runtime()
    assert runtime._source_physical_helper_module(wrapper) is helper
    assert type(runtime._build_source_executor(wrapper)) is vars(wrapper)["V5OneShotFinalizedRowExecutor"]
    prepare_source = inspect.getsource(v1physical.ReviewedCDMScoreRuntime.prepare)
    helper_line = prepare_source.index("source_physical_helpers = self._source_physical_helper_module(source_physical)")
    checkpoint_line = prepare_source.index("self._checkpoint_opened = True")
    sealed_line = prepare_source.index("substrate.load_sealed_cell_d_material")
    flags_line = prepare_source.index("flags = self._build_runtime_flags(source_execute)")
    executor_line = prepare_source.index("executor = self._build_source_executor(source_physical)")
    assert helper_line < checkpoint_line < sealed_line < flags_line < executor_line


def test_v8_identity_and_closure_bind_exact_v7_failure_literal() -> None:
    identity = _identity()
    payload = identity.payload()
    assert payload["v7_physical_helper_failure"] == plan.V7_PHYSICAL_HELPER_FAILURE.payload()
    assert payload["score_spec"]["matrix_cell_count"] == 12
    assert plan.score_matrix() == tuple(
        (budget, surface, system)
        for budget in plan.BUDGETS for surface in plan.SURFACES for system in plan.SYSTEMS
    )
    forged = dict(payload["v7_physical_helper_failure"])
    forged["failure_sha256"] = _sha("forged")
    assert forged != plan.V7_PHYSICAL_HELPER_FAILURE.payload()
    closure = plan.implementation_closure(ROOT).payload()
    assert plan.validate_implementation_closure(closure) == closure
    assert "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical.py" in plan.IMPLEMENTATION_PATHS
    assert "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v5.py" in plan.IMPLEMENTATION_PATHS
    rows = {path: _sha(f"closure-drift:{path}") for path in plan.IMPLEMENTATION_PATHS}
    rows[plan.WORKORDER_RELATIVE] = plan.WORKORDER_SHA256
    rows["tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical.py"] = _sha("forged-helper")
    assert plan.ImplementationClosure(rows).payload() != closure


def test_v8_held_v7_three_pair_reader_accepts_actual_shaped_synthetic_graph() -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        contract = _synthetic_v7_helper_failure(root)
        assert score._read_v7_physical_helper_failure(
            root, contract=contract, require_literal_contract=False,
        ) == contract.payload()


@pytest.mark.parametrize("mutation, expected", (
    ("missing", "topology"),
    ("extra", "topology"),
    ("mode", "mode"),
    ("recomputed_body", "body SHA"),
    ("sidecar", "canonical sidecar"),
    ("symlink", "descriptor"),
    ("inode", "identity"),
))
def test_v8_held_v7_three_pair_reader_rejects_leaf_or_identity_tamper(
    tmp_path: Path, mutation: str, expected: str,
) -> None:
    contract = _synthetic_v7_helper_failure(tmp_path)
    failed = tmp_path / contract.score_root_relative
    if mutation == "missing":
        os.chmod(failed / "attempt.json.sha256", 0o644)
        (failed / "attempt.json.sha256").unlink()
    elif mutation == "extra":
        (failed / "extra.json").write_bytes(b"{}")
    elif mutation == "mode":
        os.chmod(failed / "failure.json", 0o644)
    elif mutation == "recomputed_body":
        body = failed / "failure.json"
        sidecar = failed / "failure.json.sha256"
        os.chmod(body, 0o644)
        os.chmod(sidecar, 0o644)
        replacement = json.dumps({"replacement": True}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        body.write_bytes(replacement)
        sidecar.write_bytes(f"{hashlib.sha256(replacement).hexdigest()}  failure.json\n".encode("ascii"))
        os.chmod(body, 0o444)
        os.chmod(sidecar, 0o444)
    elif mutation == "sidecar":
        sidecar = failed / "input_authority.json.sha256"
        os.chmod(sidecar, 0o644)
        body = (failed / "input_authority.json").read_bytes()
        sidecar.write_bytes(f"{hashlib.sha256(body).hexdigest()}  failure.json\n".encode("ascii"))
        os.chmod(sidecar, 0o444)
    elif mutation == "symlink":
        body = failed / "failure.json"
        os.chmod(body, 0o644)
        body.unlink()
        os.symlink("attempt.json", body)
    elif mutation == "inode":
        failed.rename(failed.with_name("old-v7-failed"))
        failed.mkdir()
    else:  # pragma: no cover - literal parametrization above
        raise AssertionError(mutation)
    with pytest.raises(score.V8ScoreError, match=expected):
        score._read_v7_physical_helper_failure(tmp_path, contract=contract, require_literal_contract=False)


def test_v8_static_cli_is_no_torch_dry_and_public_flags_fail_closed() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_matched_score_v8.py"
    clean = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": ""}
    result = subprocess.run([sys.executable, str(script)], text=True, capture_output=True, env=clean, check=False)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["workorder_sha256"] == plan.WORKORDER_SHA256
    assert payload["no_torch_import"] is True and payload["no_data_access"] is True and payload["no_result_write"] is True
    code = (
        "import importlib.util,sys;"
        f"s=importlib.util.spec_from_file_location('candidate',{str(script)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.main([]);"
        "print('TORCH='+str('torch' in sys.modules))"
    )
    static = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, env=clean, check=False)
    assert static.returncode == 0 and "TORCH=False" in static.stdout
    denied = subprocess.run(
        [sys.executable, str(script), "--execute", "--root-reviewed"], text=True, capture_output=True, env=clean, check=False,
    )
    assert denied.returncode != 0 and "opaque in-process root capability" in denied.stderr


def test_v8_public_predecessor_validator_requires_literal_contract_identity(tmp_path: Path) -> None:
    forged = plan.V7PhysicalHelperFailureContract(
        **{**plan.V7_PHYSICAL_HELPER_FAILURE.__dict__, "failure_sha256": _sha("forged")},
    )
    with pytest.raises(score.V8ScoreError):
        score._read_v7_physical_helper_failure(tmp_path, contract=forged, require_literal_contract=True)


def test_v8_complete_twelve_cell_lifecycle_attempt_before_prepare_and_honest_failure(tmp_path: Path) -> None:
    """Exercise V8's new receipt codecs through the shared lifecycle only."""
    support = _v7_test_support()
    identity = _identity()
    binding = support._source_gate_binding(prefix="v8-lifecycle")
    authority, fixed = support._input_authority()
    preflight = score.build_target_free_preflight(
        root=tmp_path, identity=identity, source_gate=binding, fixed_authority=fixed,
    )
    pre_sha = score._digest(preflight)
    authorization = score.build_root_authorization(official_preflight_sha256=pre_sha, preflight=preflight)
    auth_sha = score._digest(authorization)
    capability = v1score.issue_execution_capability(
        durable_preflight_sha256=pre_sha, durable_authorization_sha256=auth_sha, identity=identity,
        root_capability=v1score._issue_root_publication_capability(),
    )
    hooks = replace(
        score.V8_LIFECYCLE_HOOKS,
        implementation_closure=lambda _root: identity.payload()["closure"],
        validate_source_gate=lambda _root: binding,
        validate_reserved_score_artifact=lambda _root, _artifact, _identity: None,
    )
    events: list[str] = []
    artifact = support._Artifact(events)
    result = v1score.run_profiled_score_lifecycle(
        tmp_path, identity=identity, capability=capability,
        backend=support._Runtime(authority, events), artifact=artifact,
        official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
        preflight=preflight, authorization=authorization, hooks=hooks,
    )
    assert events[:4] == ["preflight", "publish:attempt.json", "prepare", "materialize"]
    assert result["verdict"] == "ADVANCE_SHORT_BUDGET"
    assert {"attempt.json", "input_authority.json", "score.json", "terminal.json"} <= set(artifact.items)
    assert "failure.json" not in artifact.items
    score_body = json.loads(artifact.items["score.json"])
    assert score_body["schema"] == "causal_dual_memory_cell_d_matched_score_v8"
    assert score_body["v7_physical_helper_failure"] == plan.V7_PHYSICAL_HELPER_FAILURE.payload()
    assert score_body["budget_execution_order"] == [30, 10, 4]
    assert len(score_body["cell_execution_order"]) == 12

    failed_events: list[str] = []
    failed = support._Artifact(failed_events)
    with pytest.raises(RuntimeError, match="post-attempt"):
        v1score.run_profiled_score_lifecycle(
            tmp_path, identity=identity, capability=capability,
            backend=support._Runtime(authority, failed_events, fail_prepare=True), artifact=failed,
            official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
            preflight=preflight, authorization=authorization, hooks=hooks,
        )
    assert failed_events[:3] == ["preflight", "publish:attempt.json", "prepare"]
    assert "failure.json" in failed.items and "terminal.json" not in failed.items
    failure = json.loads(failed.items["failure.json"])
    assert failure["schema"] == "causal_dual_memory_cell_d_score_failure_v8"
    assert failure["stage"] == "prepare" and failure["checkpoint_opened"] is False
