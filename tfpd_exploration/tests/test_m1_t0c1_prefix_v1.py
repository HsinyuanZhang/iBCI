"""Synthetic/no-data/no-CUDA tests for the M1 matched T0/C1 prefix pair V1."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Mapping

import pytest

from tfpd_exploration.src.m1_t0c1_prefix_v1 import driver, hook, plan, receipts, schedule


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "tfpd_exploration/scripts/run_m1_t0c1_prefix_v1.py"


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


# ---------------------------------------------------------------------------
# Schedule and operator laws
# ---------------------------------------------------------------------------


def test_cycle_law_is_frozen_and_quarter_floors() -> None:
    assert plan.CYCLE == (10, 5, 2)
    assert plan.CYCLE_LAW["quarter_disclosure"].startswith("10/4 = 2.5 floored")
    assert schedule.sequence(7) == [10, 5, 2, 10, 5, 2, 10]
    assert schedule.effective_prefix_length(10, 10) == 10
    assert schedule.effective_prefix_length(5, 10) == 5
    assert schedule.effective_prefix_length(2, 10) == 2
    with pytest.raises(schedule.ScheduleError):
        schedule.validate_cycle((30, 10, 4))
    with pytest.raises(schedule.ScheduleError):
        schedule.m_at(-1)


class _TrainingModule:
    def __init__(self) -> None:
        self.training = True


def _kwargs(calibration):
    return {"calib_trialized_neural_features": calibration}


def _calibration(m: int) -> object:
    class _Tensor:
        def __init__(self, m_value: int) -> None:
            self.shape = (32, m_value, 1024, 64)

        def __getitem__(self, item):
            assert item == (slice(None), slice(None, None if _Visible.value is None else _Visible.value, None))
            return _Visible.value

    _Visible.value = m
    return _Tensor(m)


class _Visible:
    value: int | None = None


def test_operator_training_gated_and_t0_identity() -> None:
    import torch

    operator = hook.M1CalPrefixOperator("t0", record_steps=4)
    module = _TrainingModule()
    calibration = torch.ones(32, 10, 4, 4)
    returned = operator.forward_pre_hook(module, (), dict(_kwargs(calibration)))
    assert returned is None  # T0: kwargs unchanged, operator disabled
    assert operator.training_invocations == 1 and operator.effective_prefixes == [10]
    module.training = False
    assert operator.forward_pre_hook(module, (), dict(_kwargs(calibration))) is None
    assert operator.eval_invocations == 1 and operator.training_invocations == 1


def test_operator_c1_slices_cycle_and_rejects_short_blocks() -> None:
    import torch

    operator = hook.M1CalPrefixOperator("c1", record_steps=6)
    module = _TrainingModule()
    seen: list[int] = []
    for _ in range(6):
        calibration = torch.ones(32, 10, 4, 4)
        caller_kwargs = dict(_kwargs(calibration))
        args, kwargs_out = operator.forward_pre_hook(module, (), caller_kwargs)
        assert kwargs_out is not None and kwargs_out is not caller_kwargs
        assert caller_kwargs["calib_trialized_neural_features"] is calibration  # caller dict untouched
        seen.append(int(kwargs_out["calib_trialized_neural_features"].shape[1]))
    assert seen == [10, 5, 2, 10, 5, 2]
    assert operator.effective_prefixes == [10, 5, 2, 10, 5, 2]
    assert operator.snapshot()["recorded_prefix_sequence"] == [10, 5, 2, 10, 5, 2]
    with pytest.raises(hook.HookError):
        operator.forward_pre_hook(module, (), {})
    short = torch.ones(32, 9, 4, 4)
    with pytest.raises(hook.HookError):
        operator.forward_pre_hook(module, (), _kwargs(short))


def test_rng_stream_recorder_digests_state() -> None:
    recorder = hook.TrainingStreamRecorder()
    random.seed(3)
    before = random.getstate()
    random.random()
    after = random.getstate()
    recorder.record_step(0, before_state=before, after_state=after,
                         sample_ids=["20120924:native_window_start:5"], loss=0.5)
    payload = recorder.payload(1)
    assert payload["recorded_steps"] == 1
    assert payload["rows"][0]["rng_state_before_sha256"] != payload["rows"][0]["rng_state_after_sha256"]
    from tfpd_exploration.src.m1_t0c1_prefix_v1.schedule import ScheduleError

    with pytest.raises(ScheduleError):
        recorder.record_step(1, before_state=(1,), after_state=(1,), sample_ids=[], loss=0.0)


# ---------------------------------------------------------------------------
# Plan, dropout proof, receipts
# ---------------------------------------------------------------------------


def test_dropout_proof_and_phase1_motivation_are_bound() -> None:
    proof = plan.DROPOUT_PROOF
    assert proof["byte_identical"] is True
    assert proof["block_sha256_both_trees"] == "eacb0402448636cd0e757e9d1a28065209a5decacdbe68fe73722f634f858884"
    assert proof["m1_line_block"]["lines"] == "449-455"
    assert proof["spint_original_block"]["lines"] == "131-137"
    assert proof["override_applied"] is False and proof["verified_before_launch"] is True
    motivation = plan.PHASE1_MOTIVATION
    assert motivation["phase1_verdict"] == "HEADROOM_PRESENT"
    assert motivation["phase1_gap"] == pytest.approx(-0.12488512198130286)
    assert "headroom-capture" in motivation["note"]
    assert plan.OPERATOR_RESOLUTION["decision"].startswith("12 h hard timeout PER ARM")
    assert plan.DERIVATIVE_SCAN_OMISSION["both_arms_identical_loop"] is True


def test_stage_names_cover_expected_leaves() -> None:
    smoke = receipts.smoke_stage_names()
    assert len(smoke) == 12 and "equality.json.sha256" in smoke
    arm = receipts.arm_stage_names()
    assert len(arm) == (9 + plan.EPOCHS) * 2
    assert "checkpoint_best_source_train_loss.pt.sha256" in arm
    phase3 = receipts.phase3_stage_names()
    assert len(phase3) == (2 + 2 * 2 * len(plan.SCORE_ORDER) + 2) * 2 == 40


def test_run_stage_lifecycle_is_attempt_first_and_failure_honest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "tfpd_exploration/results/m1_t0c1_prefix_v1").mkdir(parents=True)
    events: list[str] = []
    from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1

    original = v1.ImmutableArtifactRoot.publish_json

    def observed(self: object, name: str, payload: Mapping[str, object]) -> str:
        events.append(name)
        return original(self, name, payload)

    monkeypatch.setattr(v1.ImmutableArtifactRoot, "publish_json", observed)
    closure = plan.implementation_closure(ROOT)
    attempt = receipts.stage_attempt_payload("smoke", plan.PairSpec().sha256, closure)

    def bodies(artifact: v1.ImmutableArtifactRoot) -> dict[str, str]:
        events.append("bodies")
        return {
            "t0_smoke.json": artifact.publish_json("t0_smoke.json", {"ok": True}),
            "c1_smoke.json": artifact.publish_json("c1_smoke.json", {"ok": True}),
            "equality.json": artifact.publish_json("equality.json", {"ok": True}),
        }

    shas, terminal_sha, failure_sha = receipts.run_stage(
        tmp_path, relative="tfpd_exploration/results/m1_t0c1_prefix_v1/smoke",
        attempt_payload=attempt, launch_builder=lambda: {"device": "cpu"},
        body_publisher=bodies, terminal_builder=lambda s: {"status": "COMPLETE"},
        expected_terminal_names=receipts.smoke_stage_names, progress=lambda: {"p": 1},
    )
    assert terminal_sha is not None and failure_sha is None
    assert events[:3] == ["attempt.json", "launch.json", "bodies"]
    assert events[-1] == "terminal.json"

    # failure path
    def failing_bodies(_artifact: v1.ImmutableArtifactRoot) -> dict[str, str]:
        raise RuntimeError("synthetic stage failure")

    attempt2 = receipts.stage_attempt_payload("t0", plan.PairSpec().sha256, closure)
    shas2, terminal2, failure2 = receipts.run_stage(
        tmp_path, relative="tfpd_exploration/results/m1_t0c1_prefix_v1/t0",
        attempt_payload=attempt2, launch_builder=lambda: {"device": "cpu"},
        body_publisher=failing_bodies, terminal_builder=lambda s: {},
        expected_terminal_names=receipts.arm_stage_names, progress=lambda: {"p": 2},
    )
    assert failure2 is not None and terminal2 is None
    failure = json.loads((tmp_path / "tfpd_exploration/results/m1_t0c1_prefix_v1/t0/failure.json").read_text())
    assert failure["error_class"] == "RuntimeError" and failure["progress"] == {"p": 2}
    with pytest.raises(receipts.ReceiptError):
        receipts.run_stage(
            tmp_path, relative="tfpd_exploration/results/m1_t0c1_prefix_v1/t0",
            attempt_payload=attempt2, launch_builder=lambda: {}, body_publisher=bodies,
            terminal_builder=lambda s: {}, expected_terminal_names=receipts.arm_stage_names,
        )


def test_smoke_equality_validator_enforces_every_channel() -> None:
    base = {
        "initial_model_state_sha256": "a" * 64,
        "stream_records": {
            "rng_stream_digest": "b" * 64, "batch_stream_digest": "c" * 64,
            "rows": [{"rng_state_after_sha256": "d" * 64}],
        },
        "operator_snapshot": {
            "effective_prefixes": [10, 10],
            "recorded_prefix_sequence": [10, 10],
            "records": [],
        },
        "eval_dropout_inactive_proof": {"bit_identical": True},
        "optimizer_steps": plan.SMOKE_STEPS,
    }
    c1 = json.loads(_json(base))
    c1["operator_snapshot"]["effective_prefixes"] = [10, 5]
    c1["operator_snapshot"]["recorded_prefix_sequence"] = [10, 5]
    equality = driver._validate_smoke_equality(base, c1)
    assert equality["dropout_p_stream_identical"] is True
    drifted = json.loads(_json(c1))
    drifted["stream_records"] = dict(drifted["stream_records"], rng_stream_digest="0" * 64)
    with pytest.raises(driver.DriverError):
        driver._validate_smoke_equality(base, drifted)


def test_phase3_table_aggregates_all_cells() -> None:
    cells = {}
    for arm in plan.ARMS:
        for deployment in plan.PHASE3_DEPLOYMENTS:
            for session_id in plan.SCORE_ORDER:
                cells[f"{arm}_{deployment}_{session_id}"] = {"governing_r2": 0.5}
    table = __import__(
        "tfpd_exploration.src.m1_t0c1_prefix_v1.phase3", fromlist=["build_table"],
    ).build_table(cells)
    assert len(table["rows"]) == 8
    assert all(row["equal_session_mean"] == pytest.approx(0.5) for row in table["rows"])
    assert all(delta["c1_minus_t0"] == pytest.approx(0.0)
               for delta in table["deltas"] if "c1_minus_t0" in delta)
    del cells["t0_static_m10_20120924"]
    with pytest.raises(Exception):
        __import__(
            "tfpd_exploration.src.m1_t0c1_prefix_v1.phase3", fromlist=["build_table"],
        ).build_table(cells)


def test_epoch_loss_coverage_invariant_never_weakened() -> None:
    """Regression for the t0_attempt1 failure: the recorder must cover every
    step; the epoch-mean coverage expectation itself stays exact."""
    import random

    from tfpd_exploration.src.m1_t0c1_prefix_v1 import trainer as trainer_module

    stream = hook.TrainingStreamRecorder()
    random.seed(11)
    for step in range(2 * 6):
        before = random.getstate()
        random.random()
        stream.record_step(step, before_state=before, after_state=random.getstate(),
                           sample_ids=[f"20120926:native_window_start:{step}"], loss=0.5 + step)
    assert trainer_module.epoch_mean_loss(stream, 5, 6) == pytest.approx(0.5 + 2.5)
    assert trainer_module.epoch_mean_loss(stream, 11, 6) == pytest.approx(0.5 + 8.5)
    partial = hook.TrainingStreamRecorder()
    partial.record_step(0, before_state=random.getstate(), after_state=random.getstate(),
                        sample_ids=["a"], loss=1.0)  # a recorder that stopped early
    with pytest.raises(trainer_module.TrainerError, match="epoch loss row coverage drift"):
        trainer_module.epoch_mean_loss(partial, 5, 6)
    # the published stream head stays limited while rows cover everything
    payload = stream.payload(3)
    assert payload["recorded_steps"] == 3 and len(stream.rows) == 12


def test_sealed_smoke_binding_fails_closed_on_drift(tmp_path: Path) -> None:
    smoke_root = tmp_path / plan.SMOKE_ROOT_RELATIVE
    smoke_root.mkdir(parents=True)
    terminal = {
        "schema": "m1_t0c1_smoke_terminal_v1",
        "status": plan.SEALED_SMOKE_STATUS,
    }
    body = _json(terminal)
    digest = _sha(body)
    (smoke_root / "terminal.json").write_bytes(body)
    (smoke_root / "terminal.json.sha256").write_bytes(f"{digest}  terminal.json\n".encode("ascii"))
    with pytest.raises(driver.DriverError, match="sealed smoke predecessor terminal digest drift"):
        driver._verify_sealed_smoke(tmp_path)  # wrong digest (not the sealed literal)



    closure = plan.implementation_closure(ROOT)
    assert plan.validate_current_closure(ROOT, closure) == closure
    with pytest.raises(plan.M1T0C1PlanError):
        plan.validate_current_closure(ROOT, {**closure, "closure_sha256": "0" * 64})
    process = subprocess.run(
        [sys.executable, str(CLI), "--dry-run"], text=True, capture_output=True, check=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    payload = json.loads(process.stdout)
    assert payload["public_execution_authorized"] is False
    assert payload["dropout_proof_bound_before_launch"] is True
    assert subprocess.run(
        [sys.executable, str(CLI), "--execute"], text=True, capture_output=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""},
    ).returncode != 0
    probe = (
        "import sys, json;\n"
        f"sys.path.insert(0, {str(ROOT)!r});\n"
        "from tfpd_exploration.src.m1_t0c1_prefix_v1 import phase3, plan;\n"
        "assert callable(phase3.open_session_dataset);\n"
        "assert callable(phase3.score_static) and callable(phase3.score_cdm_fifo);\n"
        "assert plan.safe_relative('SPINT-main/data/x.nwb') == 'SPINT-main/data/x.nwb';\n"
        "rejected = False\n"
        "try:\n"
        "    plan.safe_relative('/absolute/x.nwb')\n"
        "except plan.M1T0C1PlanError:\n"
        "    rejected = True\n"
        "assert rejected\n"
        "print(json.dumps({'torch': 'torch' in sys.modules,"
        " 'arm_terminals': sorted(plan.SEALED_ARM_TERMINAL_SHA256)}))"
    )
    process = subprocess.run(
        [sys.executable, "-c", probe], text=True, capture_output=True, check=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    payload = json.loads(process.stdout)
    assert payload["torch"] is False
    assert payload["arm_terminals"] == ["c1", "t0"]
