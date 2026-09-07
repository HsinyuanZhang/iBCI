from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tfpd_exploration.src.m2_anchored_output_matrix_fusion_v1 import binding, core, driver, plan, runner
from tfpd_exploration.src.m2_anchored_output_fusion_v1.runner import _digest as _aof_digest


def _pairs() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(42)
    matrix = np.array([[0.18, -0.11], [0.07, 0.15]], dtype=np.float64)
    out = {}
    for index, session in enumerate(plan.SESSIONS):
        native = rng.normal(size=(16 + index, 2))
        delta = rng.normal(size=(16 + index, 2))
        out[session] = (native, native + delta, native + delta @ matrix)
    return out


def test_direct_matrix_fit_and_zero_anchor_are_exact():
    pairs = _pairs()
    fit = core.fit_matrix(pairs, expected_sessions=plan.SESSIONS)
    assert np.allclose(fit["matrix"], [[.18, -.11], [.07, .15]], atol=1e-13)
    native, post, _ = pairs[plan.SESSIONS[0]]
    zero = core.zero_fuse(native, post, np.zeros((2, 2), dtype=np.float64))
    assert zero is native and np.array_equal(zero, native)
    negative_zero = np.zeros((2, 2), dtype=np.float64)
    negative_zero[0, 0] = -0.0
    nonsentinel = core.zero_fuse(native, post, negative_zero)
    assert nonsentinel is not native and np.array_equal(nonsentinel, native)
    with pytest.raises(core.MatrixFusionError, match="Cholesky"):
        core.fit_matrix({session: (p[0], p[0], p[2]) for session, p in pairs.items()}, expected_sessions=plan.SESSIONS)


def test_seven_fold_oof_is_lexical_and_matrix_beats_scalar_on_matrix_data():
    output = core.leave_one_session_out(_pairs())
    assert output["fold_order"] == list(plan.SESSIONS)
    assert output["passed"] is True
    assert output["positive_sessions"] == 7
    assert output["mean_matrix_minus_scalar"] >= .003
    with pytest.raises(core.MatrixFusionError, match="lexical"):
        core.leave_one_session_out(dict(reversed(list(_pairs().items()))))


def test_fold_api_is_explicit_train_six_and_rejects_held_leakage_or_bad_spectrum():
    pairs = _pairs()
    held = plan.SESSIONS[2]
    training = {session: pairs[session] for session in plan.SESSIONS if session != held}
    fold = core.fit_fold(training, held_session=held, train_sessions_exactly_six=tuple(training))
    assert fold["held_session"] == held
    assert fold["train_sessions"] == list(training)
    assert fold["matrix_fit"]["lambda_max"] >= 1e-12
    assert fold["matrix_fit"]["eigenvalue_ratio"] >= 1e-8
    assert fold["matrix_fit"]["relative_solve_residual"] <= 1e-12
    leaked = dict(training); leaked[held] = pairs[held]
    with pytest.raises(core.MatrixFusionError, match="leaked"):
        core.fit_fold(leaked, held_session=held, train_sessions_exactly_six=tuple(training))
    with pytest.raises(core.MatrixFusionError, match="leakage or order"):
        core.fit_fold(training, held_session=held, train_sessions_exactly_six=tuple(reversed(tuple(training))))
    collinear = {session: (p[0], p[0] + np.repeat((p[1] - p[0])[:, :1], 2, axis=1), p[2])
                 for session, p in training.items()}
    with pytest.raises(core.MatrixFusionError, match="eigenvalue|Cholesky"):
        core.fit_fold(collinear, held_session=held, train_sessions_exactly_six=tuple(training))


def test_actual_aof_v1_predecessor_is_exact_held_graph_read_only():
    root = Path("/home/xinyuan/Work_host/SPINT")
    witness = binding.validate_aof_v1_graph(root)
    assert witness["bodies"] == plan.AOF_V1_BODIES
    assert witness["closure_sha256"] == plan.AOF_V1_CLOSURE_SHA256
    assert tuple(witness["paired_evidence"]) == plan.SESSIONS


def test_real_docs_sha_is_checked_without_monkeypatch():
    root = Path("/home/xinyuan/Work_host/SPINT")
    assert plan.file_sha256(root / plan.DESIGN_RELATIVE) == plan.DESIGN_SHA256
    assert plan.file_sha256(root / plan.WORKORDER_RELATIVE) == plan.WORKORDER_SHA256


def test_materializer_rehashes_returned_arrays_not_only_receipt_strings():
    native = np.arange(12, dtype=np.float32).reshape(6, 2)
    post, target = native + 1.0, native + 0.5
    receipt = {
        "native_prediction_sha256": _aof_digest(native), "post_prediction_sha256": _aof_digest(post),
        "target_sha256": _aof_digest(target), "starts_sha256": "a" * 64,
        "native_identity_sha256": "b" * 64, "post_identity_sha256": "c" * 64,
        "ordered_first30_activity_sha256": "d" * 64, "ordered_support_indices_sha256": "e" * 64,
        "normalized_t4_sha256": "f" * 64, "raw_t4_sha256": "0" * 64,
        "model_state_before_sha256": "1" * 64, "model_state_after_sha256": "1" * 64,
        "windows": 6, "official_cpu_to_gpu_bridge": {"bridge": True}, "zero_prediction_exact": True,
        "parameter_updates": 0, "target_parameter_updates": 0,
    }
    predecessor = {"paired_evidence": {session: dict(receipt) for session in plan.SESSIONS}}
    materializer = runner.MatrixSourceMaterializer(Path("unused"), {}, predecessor)

    class FakeAOF:
        def pairs(self, session):
            return native, post, target, dict(receipt)
    materializer._aof = FakeAOF()
    pairs, _ = materializer.materialize_once()
    assert tuple(pairs) == plan.SESSIONS
    bad = dict(receipt); bad["target_sha256"] = "9" * 64
    predecessor_bad = {"paired_evidence": {session: dict(bad) for session in plan.SESSIONS}}
    materializer = runner.MatrixSourceMaterializer(Path("unused"), {}, predecessor_bad)
    materializer._aof = FakeAOF()
    with pytest.raises(runner.RunnerError, match="authority drift"):
        materializer.materialize_once()


def test_predecessor_rejects_extra_leaf(tmp_path):
    source = Path("/home/xinyuan/Work_host/SPINT") / plan.AOF_V1_ROOT_RELATIVE
    destination = tmp_path / plan.AOF_V1_ROOT_RELATIVE
    destination.mkdir(parents=True)
    for child in source.iterdir():
        target = destination / child.name
        target.write_bytes(child.read_bytes())
        target.chmod(0o444)
    (destination / "extra.json").write_text("{}")
    with pytest.raises(binding.BindingError, match="topology/extra"):
        binding.validate_aof_v1_graph(tmp_path)


class _SyntheticMaterializer:
    def __init__(self, root, launch, predecessor, *, fail=False, extra=False):
        self.root, self.fail, self.extra = Path(root), fail, extra

    def prepare(self):
        if self.fail:
            raise RuntimeError("synthetic prepare")
        return {"pit_materializations": 1, "optimizer_constructed": False,
                "parameter_updates": 0, "target_parameter_updates": 0}

    def materialize_once(self):
        if self.extra:
            (self.root / plan.ROOT_RELATIVE / "injected-extra.txt").write_text("adversarial")
        return _pairs(), {session: {"marker": session} for session in plan.SESSIONS}


def _set_env(monkeypatch):
    for key, value in plan.ENV.items():
        monkeypatch.setenv(key, value)


def _patch_static_admission(monkeypatch):
    monkeypatch.setattr(plan, "file_sha256", lambda path: plan.DESIGN_SHA256 if str(path).endswith(plan.DESIGN_RELATIVE) else plan.WORKORDER_SHA256)
    monkeypatch.setattr(plan, "closure", lambda _: {"route.py": "0" * 64})
    monkeypatch.setattr(binding, "validate_aof_v1_graph", lambda _: {"paired_evidence": {}, "bodies": plan.AOF_V1_BODIES})


def _reviewed() -> str:
    return plan.sha256_bytes(plan.json_bytes({"route.py": "0" * 64}))


def _parent(tmp_path):
    (tmp_path / Path(plan.ROOT_RELATIVE).parent).mkdir(parents=True)


def test_attempt_first_synthetic_success_and_failure_xor(monkeypatch, tmp_path):
    _set_env(monkeypatch); _patch_static_admission(monkeypatch); _parent(tmp_path)
    terminal, failure = driver.execute(tmp_path, reviewed_closure_sha256=_reviewed(),
                                       _gpu_attestor=lambda: {"uuid": plan.GPU_UUID, "logical_device": 0},
                                       _runner_factory=_SyntheticMaterializer)
    assert isinstance(terminal, str) and failure is None
    root = tmp_path / plan.ROOT_RELATIVE
    terminal_body = json.loads((root / "terminal.json").read_text())
    assert (root / "attempt.json").is_file() and (root / "predecessor_authority.json").is_file()
    assert (root / "oof.json").is_file() and not (root / "failure.json").exists()
    assert terminal_body["all7_refit_performed"] is True
    assert terminal_body["source_target_access_authorized"] is True
    assert terminal_body["source_target_access_attempted"] is True
    assert terminal_body["source_target_access"] is True
    assert terminal_body["final_predecessor"]["bodies"] == plan.AOF_V1_BODIES
    assert terminal_body["reviewed_closure_sha256"] == _reviewed()
    source_body = json.loads((root / "source_authority.json").read_text())
    oof_body = json.loads((root / "oof.json").read_text())
    for body in (source_body, oof_body):
        assert body["source_target_access_authorized"] is True
        assert body["source_target_access"] is True
        assert body["hidden_external_evalai_target_access"] is False

    # A distinct fresh parent exercises source failure while preserving only
    # the immutable attempt/predecessor/launch prefix.
    other = tmp_path / "other"; _parent(other)
    terminal, failure = driver.execute(other, reviewed_closure_sha256=_reviewed(),
                                       _gpu_attestor=lambda: {"uuid": plan.GPU_UUID, "logical_device": 0},
                                       _runner_factory=lambda *args: _SyntheticMaterializer(*args, fail=True))
    assert terminal is None and isinstance(failure, str)
    out = other / plan.ROOT_RELATIVE
    assert (out / "failure.json").is_file() and not (out / "terminal.json").exists()
    failure_body = json.loads((out / "failure.json").read_text())
    assert failure_body["progress"]["source_prepare_attempted"] is True
    assert failure_body["progress"]["source_opened"] is False
    assert failure_body["source_target_access_authorized"] is True
    assert failure_body["source_target_access_attempted"] is True
    assert failure_body["source_target_access"] is True
    assert failure_body["final_predecessor"]["bodies"] == plan.AOF_V1_BODIES


def test_closure_rejects_missing_leaf(monkeypatch, tmp_path):
    monkeypatch.setattr(plan, "CLOSURE", ("missing.py",))
    with pytest.raises(ValueError, match="missing/symlink"):
        plan.closure(tmp_path)


def test_reviewed_closure_required_before_reservation_and_extra_cannot_terminal(monkeypatch, tmp_path):
    _set_env(monkeypatch); _patch_static_admission(monkeypatch); _parent(tmp_path)
    with pytest.raises(TypeError, match="reviewed_closure_sha256"):
        driver.execute(tmp_path)
    assert not (tmp_path / plan.ROOT_RELATIVE).exists()
    with pytest.raises(RuntimeError, match="reviewed closure"):
        driver.execute(tmp_path, reviewed_closure_sha256="f" * 64,
                       _gpu_attestor=lambda: {"uuid": plan.GPU_UUID, "logical_device": 0},
                       _runner_factory=_SyntheticMaterializer)
    assert not (tmp_path / plan.ROOT_RELATIVE).exists()
    with pytest.raises(Exception, match="topology"):
        driver.execute(tmp_path, reviewed_closure_sha256=_reviewed(),
                       _gpu_attestor=lambda: {"uuid": plan.GPU_UUID, "logical_device": 0},
                       _runner_factory=lambda *args: _SyntheticMaterializer(*args, extra=True))
    root = tmp_path / plan.ROOT_RELATIVE
    assert not (root / "terminal.json").exists()
