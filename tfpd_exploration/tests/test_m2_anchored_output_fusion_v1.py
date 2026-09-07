from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tfpd_exploration.src.m2_anchored_output_fusion_v1 import core, driver, plan
from tfpd_exploration.src.m2_anchored_output_fusion_v1.runner import StaticAOFRunner
from tfpd_exploration.src.m2_anchored_output_fusion_v1.runner import _official_receipt, _official_native_score, _official_cpu_identities, _official_array_digest


def _pairs(count: int, beta: float = 0.25):
    rng = np.random.default_rng(3)
    result = {}
    for index in range(count):
        native = rng.normal(size=(5 + index, 2))
        delta = rng.normal(size=(5 + index, 2))
        result[f"s{index}"] = (native, native + delta, native + beta * delta)
    return result


def test_closed_form_recovers_known_equal_session_beta_and_zero_is_native():
    pairs = _pairs(5)
    fit = core.beta_equal_session(pairs)
    assert abs(fit["beta"] - 0.25) < 1e-12
    native, post, _ = pairs["s0"]
    assert core.fuse(native, post, 0.0) is native
    assert np.array_equal(core.fuse(native, post, 0.0), native)
    assert core.fuse(native, post, -0.0) is not native


def test_gate_and_all_seven_refit_conditions_are_explicit():
    validation = _pairs(2, beta=0.1)
    passed = core.gate(validation, 0.1)
    assert passed["passed"]
    assert passed["positive_sessions"] == 2
    assert passed["worst_delta"] >= 0.001
    assert len(core.beta_equal_session(_pairs(7))["per_session"]) == 7
    with pytest.raises(core.AOFError):
        core.beta_equal_session({"bad": (np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)))})
    with pytest.raises(core.AOFError, match="lexical"):
        core.gate(validation, 0.1, expected_sessions=("s1", "s0"))


def test_runner_lexical_5_2_order_and_all_seven_refit_are_conditional(monkeypatch):
    runner = StaticAOFRunner("unused", {})
    names = StaticAOFRunner.FIT + StaticAOFRunner.VALIDATION
    runner.static = {name: object() for name in names}
    raw = {name: values for name, values in zip(names, _pairs(7, beta=0.1).values())}
    opened = []

    def pairs(session):
        opened.append(session)
        native, post, target = raw[session]
        return native, post, target, {"zero_prediction_exact": True, "zero_branch": "direct_native"}

    monkeypatch.setattr(runner, "pairs", pairs)
    result = runner.run()
    assert result["source_split"] == {"fit": list(StaticAOFRunner.FIT), "validation": list(StaticAOFRunner.VALIDATION)}
    assert result["validation"]["passed"] is True
    assert result["fit"]["objective"] == "equal_session_mse_surrogate_not_direct_r2"
    assert result["fit"]["behavior_scaling_factor"] == 5.0
    assert result["all7_refit"] is not None and len(result["all7_refit"]["per_session"]) == 7
    assert opened == list(names)

    monkeypatch.setattr(core, "gate", lambda *_, **__: {"passed": False, "rows": {name: {} for name in StaticAOFRunner.VALIDATION}})
    failed = runner.run()
    assert failed["all7_refit"] is None


def test_closure_rejects_missing_or_duplicate_leaf(monkeypatch, tmp_path):
    monkeypatch.setattr(plan, "CLOSURE", ("missing.py",))
    with pytest.raises(ValueError, match="missing/symlink"):
        plan.closure(tmp_path)
    leaf = tmp_path / "leaf.py"
    leaf.write_text("x")
    monkeypatch.setattr(plan, "CLOSURE", ("leaf.py", "leaf.py"))
    with pytest.raises(ValueError, match="duplicate"):
        plan.closure(tmp_path)


def test_exact_official_act30_receipt_is_read_only_code_authority():
    payload = _official_receipt(Path("/home/xinyuan/Work_host/SPINT"))
    native_rows = _official_native_score(Path("/home/xinyuan/Work_host/SPINT"), payload)
    cpu_identities = _official_cpu_identities(Path("/home/xinyuan/Work_host/SPINT"), payload)
    assert payload["schema_version"] == "m2_dopt4_act30_official_payload_receipt_v1"
    assert payload["status"] == "EXPORTED_NOT_SUBMITTED"
    assert payload["session_count"] == 13
    for session in StaticAOFRunner.FIT + StaticAOFRunner.VALIDATION:
        row = payload["session_records"][session]
        assert row["activity_shape"] == [30, 100, 96]
        assert len(row["activity_sha256"]) == len(row["side_sha256"]) == len(row["identity_sha256"]) == 64
        assert native_rows[session]["cell"] == "ridge_activity30_m4"
        assert native_rows[session]["surface"] == "within_post30"
        assert _official_array_digest(cpu_identities[row["dataset_tag"]]) == row["identity_sha256"]


class _FakeRunner:
    def __init__(self, root, launch, *, fail=False):
        self.root = root
        self.launch = launch
        self.fail = fail

    def prepare(self):
        if self.fail:
            raise RuntimeError("synthetic source prepare failure")
        return {
            "pit_materializations": 1,
            "sessions": [f"s{i}" for i in range(7)],
            "checkpoint_sha256": plan.CHECKPOINT_SHA256,
            "student_state_sha256": plan.STUDENT_STATE_SHA256,
            "model_state_before_sha256": plan.STUDENT_STATE_SHA256,
            "model_state_after_sha256": plan.STUDENT_STATE_SHA256,
            "model_state_unchanged": True,
            "parameter_updates": 0,
            "target_parameter_updates": 0,
        }

    def run(self):
        evidence = {
            f"s{i}": {
                "raw_m30_hz_audit_sha256": "a" * 64,
                "normalized_side_sha256": "b" * 64,
                "ordered_first30_activity_sha256": "c" * 64,
                "ordered_support_indices_sha256": "d" * 64,
                "model_state_before_sha256": plan.STUDENT_STATE_SHA256,
                "model_state_after_sha256": plan.STUDENT_STATE_SHA256,
                "zero_prediction_exact": True,
                "parameter_updates": 0,
                "target_parameter_updates": 0,
            }
            for i in range(7)
        }
        return {
            "fit": {"beta": 0.1},
            "validation": {"passed": True, "mean_delta": 0.01, "positive_sessions": 2, "worst_delta": 0.002},
            "session_evidence": evidence,
            "source_split": {"fit": [f"s{i}" for i in range(5)], "validation": ["s5", "s6"]},
            "all7_refit": {"beta": 0.2},
            "source_target_access": True,
            "hidden_external_evalai_target_access": False,
        }


def _set_env(monkeypatch):
    for key, value in plan.ENV.items():
        monkeypatch.setenv(key, value)


def _patch_static_admission(monkeypatch):
    def _file_sha(path):
        return plan.WORKORDER_SHA256 if str(path).endswith(plan.WORKORDER_RELATIVE) else plan.DESIGN_SHA256
    monkeypatch.setattr(plan, "file_sha", _file_sha)
    monkeypatch.setattr(plan, "closure", lambda _: {"aof.py": "0" * 64})


def _make_result_parent(root):
    (root / Path(plan.ROOT_RELATIVE).parent).mkdir(parents=True)


def test_no_cuda_lifecycle_success_has_prefix_and_terminal(monkeypatch, tmp_path):
    _set_env(monkeypatch)
    _patch_static_admission(monkeypatch)
    _make_result_parent(tmp_path)
    terminal, failure = driver.execute(
        tmp_path,
        _gpu_attestor=lambda: {"cuda_visible_devices": "0", "logical_device": 0, "uuid": plan.GPU_UUID},
        _runner_factory=_FakeRunner,
    )
    assert failure is None and isinstance(terminal, str) and len(terminal) == 64
    out = tmp_path / plan.ROOT_RELATIVE
    names = {p.name for p in out.iterdir()}
    assert "terminal.json" in names and "failure.json" not in names
    for body in out.glob("*.json"):
        assert (body.stat().st_mode & 0o777) == 0o444
        assert (body.with_name(body.name + ".sha256")).is_file()
    payload = __import__("json").loads((out / "terminal.json").read_text())
    assert payload["source_target_indices_opened"] is True
    assert payload["hidden_external_evalai_target_access"] is False


def test_no_cuda_lifecycle_failure_preserves_honest_attempt_prefix(monkeypatch, tmp_path):
    _set_env(monkeypatch)
    _patch_static_admission(monkeypatch)
    _make_result_parent(tmp_path)

    def factory(root, launch):
        return _FakeRunner(root, launch, fail=True)

    terminal, failure = driver.execute(
        tmp_path,
        _gpu_attestor=lambda: {"cuda_visible_devices": "0", "logical_device": 0, "uuid": plan.GPU_UUID},
        _runner_factory=factory,
    )
    assert terminal is None and isinstance(failure, str)
    out = tmp_path / plan.ROOT_RELATIVE
    assert (out / "attempt.json").is_file() and (out / "launch.json").is_file()
    assert (out / "failure.json").is_file() and not (out / "terminal.json").exists()
    payload = __import__("json").loads((out / "failure.json").read_text())
    assert payload["progress"]["source_prepare_attempted"] is True
    assert payload["progress"]["source_opened"] is False
    assert payload["progress"]["stage"] == "source_prepare"
