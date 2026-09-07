"""Synthetic/no-data/no-CUDA tests for the PIT-M2 matched pair V1."""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from tfpd_exploration.src.pit_m2_v1 import hook, phase3, plan, receipts, schedule, smoke, trainer


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "tfpd_exploration/scripts/run_pit_m2_v1.py"


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


# ---------------------------------------------------------------------------
# Plan literals and the frozen cycle law
# ---------------------------------------------------------------------------

def test_cycle_law_is_frozen_and_quarter_floors() -> None:
    assert plan.CYCLE == (10, 5, 2)
    assert plan.CYCLE_LAW["terms"] == {"full": 10, "half": 5, "quarter_floored": 2}
    assert plan.CYCLE_LAW["quarter_disclosure"].startswith("10/4 = 2.5 floored")
    assert plan.CYCLE_LAW["training_gated"] is True and plan.CYCLE_LAW["eval_untouched"] is True
    assert schedule.sequence(7) == [10, 5, 2, 10, 5, 2, 10]
    assert schedule.sequence(40) == [10, 5, 2] * 13 + [10]
    assert schedule.effective_prefix_length(10, 33) == 10
    assert schedule.effective_prefix_length(5, 33) == 5
    assert schedule.effective_prefix_length(2, 33) == 2
    with pytest.raises(schedule.ScheduleError):
        schedule.validate_cycle((30, 10, 4))
    with pytest.raises(schedule.ScheduleError):
        schedule.validate_cycle((10, 5, 2, 1))
    with pytest.raises(schedule.ScheduleError):
        schedule.m_at(-1)


def test_single_budget_m10_deployment_binding() -> None:
    assert plan.DEPLOYMENT_BUDGET_M == 10
    assert plan.TRAINING_CALIBRATION_N_TRIALS == 33
    assert max(plan.CYCLE) == plan.DEPLOYMENT_BUDGET_M
    assert "single-budget M10" in plan.CYCLE_LAW["basis"]
    assert plan.STATIC_LAW["cell"].startswith("ridge_static_m10")
    assert plan.FIFO_LAW["cell"].startswith("m10_activity_only")


def test_sealed_anchors_are_self_consistent() -> None:
    for anchor in (plan.F00M_ANCHOR, plan.F01M_ANCHOR):
        values = list(anchor["per_session_r2"].values())
        assert anchor["session_count"] == len(values) == 6
        mean = sum(values) / len(values)
        assert abs(mean - anchor["equal_session_mean"]) < 1e-12
        assert anchor["checkpoint_sha256"] == plan.SEALED_CHECKPOINT_SHA256
        assert anchor["rerun_by_this_lane"] is False
    assert plan.FACTORIAL_CELLS == ("F00m", "F10m", "F01m", "F11m")
    assert set(plan.SEALED_CELLS) == {"F00m", "F01m"}
    assert set(plan.FRESH_CELLS) == {"F10m", "F11m"}
    assert plan.GATES["primary"]["driving_cell"] == "F10m"
    assert plan.GATES["primary"]["reference_cell"] == "F00m"
    assert plan.GATES["secondary_stacking"]["driving_cell"] == "F11m"
    assert plan.GATES["secondary_stacking"]["role"].startswith("stacking-reading SECONDARY")
    assert "NO promotion" in plan.GATES["secondary_stacking"]["role"]
    assert plan.GATES["primary"]["breadth_min"] / plan.GATES["primary"]["breadth_denominator"] \
        == pytest.approx(2 / 3)
    assert plan.GATE_BOUNDARY_EPSILON == 1.0e-12


def test_sealed_trainer_binding_points_at_the_25d7bc72_recipe() -> None:
    binding = plan.TRAINER_BINDING
    assert binding["launch_command"]["path"].endswith("run_m2_spint_t4_mainline.sh")
    assert "experiment=b3s_t4_m2_loso_internal" in binding["launch_command"]["what"]
    assert "data.calibration_n_trials=33" in binding["launch_command"]["what"]
    assert "trainer.max_epochs=12" in binding["launch_command"]["what"]
    assert binding["checkpoint_lineage"]["sealed_checkpoint_sha256"] == plan.SEALED_CHECKPOINT_SHA256
    for relative in plan.IMMUTABLE_ANCHOR_RELATIVE:
        assert (ROOT / relative).is_file(), relative
    for relative in plan.OWNED_PATHS:
        assert (ROOT / relative).is_file(), relative


def test_dropout_proof_records_the_carried_teacher_stream_law() -> None:
    finding = plan.DROPOUT_PROOF["m2_recipe_finding"]
    assert finding["sealed_freeze_decoder"] is True
    assert finding["student_dropout_branch_reachable"] is False
    assert finding["student_dropout_p_draws_per_training_step"] == 0
    assert finding["teacher_dropout_p_draws_per_training_step"] == 1
    assert finding["teacher_draw_is_inert_in_eval"] is True
    assert finding["teacher_draw_is_arm_independent"] is True
    assert plan.DROPOUT_PROOF["spint_original_block"]["sha256_at_design"] == \
        plan.DROPOUT_PROOF["streaming_teacher_block"]["sha256_at_design"]
    assert plan.DROPOUT_PROOF["override_applied"] is False
    assert plan.DROPOUT_PROOF["verified_before_launch"] is True


# ---------------------------------------------------------------------------
# The operator: gating, RNG-freedom, T0m identity
# ---------------------------------------------------------------------------

class _TrainingModule:
    def __init__(self, training: bool = True) -> None:
        self.training = training


def _kwargs(calibration):
    return {"calib_trials": calibration}


def test_operator_training_gated_and_t0m_identity() -> None:
    import torch

    operator = hook.PitM2PrefixOperator("t0m", record_steps=4)
    module = _TrainingModule()
    calibration = torch.ones(32, 33, 100, 96)
    returned = operator.forward_pre_hook(module, (), dict(_kwargs(calibration)))
    assert returned is None  # T0m: kwargs unchanged, operator disabled
    assert operator.training_invocations == 1
    assert operator.effective_prefixes == [33]
    module.training = False
    assert operator.forward_pre_hook(module, (), dict(_kwargs(calibration))) is None
    assert operator.eval_invocations == 1 and operator.training_invocations == 1


def test_operator_c1m_slices_cycle_and_rejects_drift() -> None:
    import torch

    operator = hook.PitM2PrefixOperator("c1m", record_steps=6)
    module = _TrainingModule()
    seen: list[int] = []
    for _ in range(6):
        calibration = torch.ones(32, 33, 100, 96)
        caller_kwargs = dict(_kwargs(calibration))
        args, kwargs_out = operator.forward_pre_hook(module, (), caller_kwargs)
        assert kwargs_out is not None and kwargs_out is not caller_kwargs
        assert caller_kwargs["calib_trials"] is calibration  # caller dict untouched
        seen.append(int(kwargs_out["calib_trials"].shape[1]))
    assert seen == [10, 5, 2, 10, 5, 2]
    assert operator.effective_prefixes == [10, 5, 2, 10, 5, 2]
    assert operator.snapshot()["recorded_prefix_sequence"] == [10, 5, 2, 10, 5, 2]
    with pytest.raises(hook.HookError):
        operator.forward_pre_hook(module, (), {})  # calib_trials missing
    wrong_block = torch.ones(32, 10, 100, 96)
    with pytest.raises(hook.HookError):
        operator.forward_pre_hook(module, (), _kwargs(wrong_block))  # native 33 only


def test_recording_window_is_bounded_and_counter_is_not() -> None:
    import torch

    operator = hook.PitM2PrefixOperator("c1m", record_steps=3)
    module = _TrainingModule()
    for _ in range(5):
        operator.forward_pre_hook(module, (), _kwargs(torch.ones(2, 33, 100, 96)))
    assert operator.training_invocations == 5
    assert len(operator.records) == 3
    assert operator.effective_prefixes == [10, 5, 2, 10, 5]


def test_operator_consumes_no_python_numpy_or_torch_rng(monkeypatch) -> None:
    import numpy as np
    import torch

    operator = hook.PitM2PrefixOperator("c1m", record_steps=4)
    module = _TrainingModule()
    calibration = torch.ones(32, 33, 100, 96)

    calls = {"python": 0, "numpy": 0, "torch": 0}

    def counting_python(*args, **kwargs):
        calls["python"] += 1
        return 0.5

    def counting_numpy(*args, **kwargs):
        calls["numpy"] += 1
        return np.zeros(())

    def counting_torch(*args, **kwargs):
        calls["torch"] += 1
        return torch.zeros(1)

    monkeypatch.setattr(random, "uniform", counting_python, raising=False)
    monkeypatch.setattr(random, "random", counting_python, raising=False)
    monkeypatch.setattr(random, "randrange", counting_python, raising=False)
    monkeypatch.setattr(random, "shuffle", counting_python, raising=False)
    for name in ("uniform", "normal", "rand", "randint", "choice", "shuffle"):
        monkeypatch.setattr(np.random, name, counting_numpy, raising=False)
    for name in ("rand", "randn", "randint", "randperm"):
        monkeypatch.setattr(torch, name, counting_torch, raising=False)

    for _ in range(4):
        operator.forward_pre_hook(module, (), _kwargs(calibration))
        module.training = False
        operator.forward_pre_hook(module, (), _kwargs(calibration))
        module.training = True
    assert calls == {"python": 0, "numpy": 0, "torch": 0}


def test_counting_rng_probe_counts_and_restores() -> None:
    import numpy as np
    import torch

    with hook.CountingRngProbe() as probe:
        assert random.random() >= 0.0
        assert np.random.rand() >= 0.0
        assert torch.rand(1).numel() == 1
        assert probe.snapshot() == {"python": 1, "numpy": 1, "torch": 1}
    assert callable(random.random)
    assert random.random() >= 0.0  # restored, still functional


def test_attach_and_detach_on_a_synthetic_module() -> None:
    import torch

    class _Model(torch.nn.Module):
        def forward(self, neural, calib_trials=None):  # pragma: no cover - thin
            return neural, calib_trials

    operator = hook.PitM2PrefixOperator("c1m", record_steps=0)
    model = _Model()
    operator.attach(model)
    assert operator._handle is not None
    with pytest.raises(hook.HookError):
        operator.attach(model)
    operator.detach()
    assert operator._handle is None
    operator.detach()  # idempotent


# ---------------------------------------------------------------------------
# Digest helpers
# ---------------------------------------------------------------------------

def test_stream_recorder_digests_state() -> None:
    recorder = hook.TrainingStreamRecorder()
    random.seed(3)
    before = random.getstate()
    random.random()
    after = random.getstate()
    import torch

    torch_state = torch.get_rng_state()
    recorder.record_step(0, before_python=before, after_python=after,
                         before_torch=torch_state, after_torch=torch_state,
                         sample_ids=["ses-x@window_start:5"], t4_bytes_digest="ab" * 32,
                         loss=0.5)
    payload = recorder.payload(1)
    assert payload["recorded_steps"] == 1
    assert payload["rng_stream_digest"] == schedule.sequence_digest(
        [payload["rows"][0]["rng_state_after_sha256"]])
    assert payload["t4_bytes_stream_digest"] == schedule.sequence_digest(["ab" * 32])
    with pytest.raises(hook.HookError):
        recorder.record_step(-1, before_python=before, after_python=after,
                             before_torch=torch_state, after_torch=torch_state,
                             sample_ids=["a"], t4_bytes_digest="ab" * 32, loss=0.5)


def test_digest_helpers_deterministic_and_order_sensitive() -> None:
    import numpy as np
    import torch

    array = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert schedule.tensor_digest(array) == schedule.tensor_digest(array.copy())
    assert schedule.tensor_digest(array) != schedule.tensor_digest(array * 2)
    assert schedule.tensor_digest(torch.from_numpy(array)) == schedule.tensor_digest(array)
    assert schedule.batch_digest(["a:1", "b:2"]) != schedule.batch_digest(["b:2", "a:1"])
    assert schedule.sequence_digest([10, 5, 2]) == schedule.sequence_digest((10, 5, 2))
    assert schedule.int_list_digest([[1, 2], [3]]) == schedule.int_list_digest([[1, 2], [3]])
    assert schedule.int_list_digest([[1], [2]]) != schedule.int_list_digest([[2], [1]])
    random.seed(11)
    state = random.getstate()
    assert schedule.rng_state_digest(state) == schedule.rng_state_digest(random.getstate())
    assert schedule.torch_rng_state_digest(torch.get_rng_state()) == \
        schedule.torch_rng_state_digest(torch.get_rng_state())
    with pytest.raises(schedule.ScheduleError):
        schedule.tensor_digest(np.asarray([float("nan")]))


# ---------------------------------------------------------------------------
# Gate arithmetic and the 1e-12 boundary band
# ---------------------------------------------------------------------------

def test_gate_exact_boundary_passes() -> None:
    result = phase3.evaluate_gate(delta_mean=0.01, positive_sessions=4,
                                  session_count=6, floor=0.01, breadth_min=4)
    assert result["exact_delta_pass"] is True
    assert result["within_epsilon_band_of_boundary"] is False
    assert result["verdict"] == "PASSED"


def test_gate_miss_inside_epsilon_band_is_disclosed_not_flipped() -> None:
    delta = 0.01 - 5.0e-13
    result = phase3.evaluate_gate(delta_mean=delta, positive_sessions=4,
                                  session_count=6, floor=0.01, breadth_min=4)
    assert result["exact_delta_pass"] is False
    assert result["within_epsilon_band_of_boundary"] is True
    assert result["verdict"] == "NOT_MET__WITHIN_EPSILON_BAND_OF_BOUNDARY"


def test_gate_miss_outside_band_fails_plainly() -> None:
    delta = 0.01 - 1.0e-9
    result = phase3.evaluate_gate(delta_mean=delta, positive_sessions=6,
                                  session_count=6, floor=0.01, breadth_min=4)
    assert result["within_epsilon_band_of_boundary"] is False
    assert result["verdict"] == "NOT_MET"


def test_gate_breadth_failure_is_disclosed_separately() -> None:
    result = phase3.evaluate_gate(delta_mean=0.05, positive_sessions=3,
                                  session_count=6, floor=0.01, breadth_min=4)
    assert result["delta_verdict"] == "PASSED"
    assert result["breadth_pass"] is False
    assert result["verdict"] == "NOT_MET__BREADTH"


def _synthetic_table(fresh_static_mean: float, fresh_fifo_mean: float,
                     positive_static: int, positive_fifo: int) -> dict[str, object]:
    sessions = sorted(plan.F00M_ANCHOR["per_session_r2"])
    static_values = {
        name: plan.F00M_ANCHOR["per_session_r2"][name]
        + (0.1 if index < positive_static else -0.05) + fresh_static_mean
        for index, name in enumerate(sessions)
    }
    fifo_values = {
        name: plan.F01M_ANCHOR["per_session_r2"][name]
        + (0.1 if index < positive_fifo else -0.05) + fresh_fifo_mean
        for index, name in enumerate(sessions)
    }
    anchors = {
        "F00m": {"per_session_r2": plan.F00M_ANCHOR["per_session_r2"],
                 "equal_session_mean": plan.F00M_ANCHOR["equal_session_mean"],
                 "rerun_by_this_lane": False},
        "F01m": {"per_session_r2": plan.F01M_ANCHOR["per_session_r2"],
                 "equal_session_mean": plan.F01M_ANCHOR["equal_session_mean"],
                 "rerun_by_this_lane": False},
    }
    fresh = {
        "c1m_static_external_official_query": {
            "per_session_r2": static_values,
            "equal_session_mean": sum(static_values.values()) / 6},
        "c1m_fifo_external_post30_local": {
            "per_session_r2": fifo_values,
            "equal_session_mean": sum(fifo_values.values()) / 6},
        "c1m_static_within_post30": {
            "per_session_r2": {name: 0.5 for name in plan.WITHIN_SESSION_NAMES},
            "equal_session_mean": 0.5},
        "c1m_fifo_within_post30": {
            "per_session_r2": {name: 0.5 for name in plan.WITHIN_SESSION_NAMES},
            "equal_session_mean": 0.5},
        "t0m_static_external_official_query": {
            "per_session_r2": static_values,
            "equal_session_mean": sum(static_values.values()) / 6},
        "t0m_fifo_external_post30_local": {
            "per_session_r2": fifo_values,
            "equal_session_mean": sum(fifo_values.values()) / 6},
    }
    return phase3.build_table(anchors, fresh)


def test_build_table_and_gates_roles() -> None:
    table = _synthetic_table(0.02, 0.05, positive_static=5, positive_fifo=6)
    assert set(table["cells"]) == {"F00m", "F01m", "F10m", "F11m"}
    assert table["cells"]["F00m"]["rerun_by_this_lane"] is False
    assert table["cells"]["F01m"]["rerun_by_this_lane"] is False
    assert table["cells"]["F10m"]["surface"] == "external_official_query"
    assert table["cells"]["F11m"]["surface"] == "external_post30_local"
    assert sorted(table["cells"]["F10m"]["per_session_r2"]) == sorted(plan.EXTERNAL_SESSION_NAMES)
    assert "c1m_static_within_post30" in table["reported_not_gated"]
    assert "t0m_static_external_official_query" in table["reported_not_gated"]
    gates = phase3.evaluate_gates(table)
    assert gates["primary"]["verdict"] == "PASSED"
    assert gates["primary"]["positive_sessions"] == 5
    assert gates["primary"]["final_disposition"] == plan.GATES["primary"]["disposition_passed"]
    assert gates["secondary_stacking"]["verdict"] == "PASSED"
    assert gates["secondary_stacking"]["promotion_attached"] is False
    assert gates["secondary_stacking"]["disposition"] == "REPORTED_NO_PROMOTION"
    failing = phase3.evaluate_gates(_synthetic_table(-0.2, -0.2, 0, 0))
    assert failing["primary"]["verdict"] == "NOT_MET"
    assert failing["primary"]["final_disposition"] == plan.GATES["primary"]["disposition_failed"]
    with pytest.raises(phase3.Phase3Error):
        _synthetic_table_bad_roster()


def test_build_table_rejects_the_attempt3_clobbered_summary_shape() -> None:
    """Regression: the attempt-3 failure fed build_table a path-keyed summary
    whose external entry had been clobbered by the within roster.  build_table
    must fail closed on that shape instead of producing a vague mismatch."""
    anchors = {
        "F00m": {"per_session_r2": plan.F00M_ANCHOR["per_session_r2"],
                 "equal_session_mean": 0.2, "rerun_by_this_lane": False},
        "F01m": {"per_session_r2": plan.F01M_ANCHOR["per_session_r2"],
                 "equal_session_mean": 0.2, "rerun_by_this_lane": False},
    }
    # the old (path-only) key shape is rejected outright
    with pytest.raises(phase3.Phase3Error, match="external c1m cells"):
        phase3.build_table(anchors, {"c1m_static": {"per_session_r2": {}},
                                     "c1m_fifo": {"per_session_r2": {}}})
    # an external-keyed entry carrying the WITHIN roster is rejected fail-closed
    with pytest.raises(phase3.Phase3Error, match="not the external roster"):
        phase3.build_table(anchors, {
            "c1m_static_external_official_query": {
                "per_session_r2": {name: 0.5 for name in plan.WITHIN_SESSION_NAMES}},
            "c1m_fifo_external_post30_local": {
                "per_session_r2": {name: 0.5 for name in plan.EXTERNAL_SESSION_NAMES}},
        })


def test_implementation_closure_digests_anchors_and_owned_bytes() -> None:
    closure = plan.implementation_closure(ROOT)
    assert closure["immutable_anchors_sha256"][plan.SEALED_CHECKPOINT_RELATIVE] \
        == plan.SEALED_CHECKPOINT_SHA256
    assert closure["immutable_anchors_sha256"][plan.SEALED_RESOLVED_CONFIG_RELATIVE] \
        == plan.SEALED_RESOLVED_CONFIG_SHA256
    assert len(closure["live_source_sha256_at_attempt"]) == len(plan.LIVE_SOURCE_RELATIVE)
    assert set(closure["owned_sha256"]) == set(plan.OWNED_PATHS)
    rebuilt = plan.implementation_closure(ROOT)
    assert rebuilt["closure_sha256"] == closure["closure_sha256"]


def _synthetic_table_bad_roster() -> None:
    anchors = {
        "F00m": {"per_session_r2": {"only-one": 0.1}, "equal_session_mean": 0.1},
        "F01m": {"per_session_r2": plan.F01M_ANCHOR["per_session_r2"],
                 "equal_session_mean": 0.2},
    }
    phase3.build_table(anchors, {
        "c1m_static": {"per_session_r2": plan.F00M_ANCHOR["per_session_r2"]},
        "c1m_fifo": {"per_session_r2": plan.F01M_ANCHOR["per_session_r2"]},
    })


def test_sealed_anchor_loading_verifies_sha_and_roster() -> None:
    anchors = phase3.load_sealed_anchors(ROOT)
    assert anchors["F00m"]["source_sha256"] == plan.F00M_ANCHOR["sha256"]
    assert anchors["F01m"]["source_sha256"] == plan.F01M_ANCHOR["sha256"]
    assert anchors["F00m"]["rerun_by_this_lane"] is False
    assert anchors["F01m"]["rerun_by_this_lane"] is False
    assert abs(anchors["F00m"]["equal_session_mean"]
               - plan.F00M_ANCHOR["equal_session_mean"]) < 1e-15
    assert abs(anchors["F01m"]["equal_session_mean"]
               - plan.F01M_ANCHOR["equal_session_mean"]) < 1e-15


def test_phase3_scorer_reuse_symbols_resolve() -> None:
    """The GPU phase3 stage reuses sealed-screen helpers by name; fail closed
    here (no data, no CUDA) if any of those attributes drifts away."""
    from tfpd_exploration.src.causal_dual_memory_cell_d_v1 import core as cdm_core
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import physical as cdm_physical
    from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import plan as cdm_plan
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import core as screen_core
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import physical as screen_physical
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core as shared_core

    for name in ("_score_session",):
        assert callable(getattr(screen_physical, name))
    cell = screen_core.CellSpec("ridge_static_m10", plan.DEPLOYMENT_BUDGET_M,
                                plan.DEPLOYMENT_BUDGET_M)
    assert cell.name == "ridge_static_m10"
    for name in ("_native_trial_views", "_finite_m10_indices", "_raw_fixed_ridge_m30",
                 "_query_trial_rows", "_build_memory", "_score_system"):
        assert callable(getattr(cdm_physical, name))
    assert cdm_plan.ACTIVITY_STACK_LIMIT == 30
    assert cdm_plan.SYSTEM_ACTIVITY == "activity_only"
    assert callable(cdm_core.B3SInterpolatedSpikeCountTrial)
    assert callable(cdm_core.channel_order_digest)
    assert callable(shared_core.array_sha256)


# ---------------------------------------------------------------------------
# Train-stage refusal law
# ---------------------------------------------------------------------------

def _gpu_text(rows: list[tuple[str, str, str, str]]) -> str:
    return "\n".join(",".join(row) for row in rows) + "\n"


IDLE_ROWS = [("0", "GPU-aaa", "5", "0"), ("1", "GPU-bbb", "10", "0")]
BUSY_ROWS = [("0", "GPU-aaa", "5", "0"), ("1", "GPU-bbb", "2354", "41")]


def test_train_stage_refuses_without_authorization() -> None:
    with pytest.raises(trainer.TrainerError, match="gpu-authorized"):
        trainer.require_train_authorization(gpu_authorized=False,
                                            cmdline_runner=lambda argv: _gpu_text(IDLE_ROWS))


def test_train_stage_refuses_when_a_gpu_is_busy() -> None:
    with pytest.raises(trainer.TrainerError, match="busy"):
        trainer.require_train_authorization(gpu_authorized=True,
                                            cmdline_runner=lambda argv: _gpu_text(BUSY_ROWS))


def test_train_stage_authorization_passes_only_when_both_idle() -> None:
    receipt = trainer.require_train_authorization(
        gpu_authorized=True, cmdline_runner=lambda argv: _gpu_text(IDLE_ROWS))
    assert receipt["law"] == "both_gpus_idle_v1"
    assert receipt["both_gpus_idle"] is True
    assert len(receipt["gpu_rows"]) == 2


# ---------------------------------------------------------------------------
# Target-GPU-idle guard (operator amendment 2026-09-02)
# ---------------------------------------------------------------------------

APPS_IDLE = ""
APPS_ON_GPU1 = "GPU-bbb,783126,/python3.10,708\n"


def _runner(gpu_rows: list[tuple[str, str, str, str]], apps_text: str):
    def run(argv: list[str]) -> str:
        if "--query-gpu=index" in " ".join(argv):
            return _gpu_text(gpu_rows)
        return apps_text
    return run


def test_target_idle_guard_relaxes_a_busy_other_card() -> None:
    # GPU 0 busy (mainline), target GPU 1 idle: the amended law PASSES.
    rows = [("0", "GPU-aaa", "1167", "22"), ("1", "GPU-bbb", "23", "0")]
    receipt = trainer.require_train_authorization(
        gpu_authorized=True, gpu_index=1, cmdline_runner=_runner(rows, APPS_IDLE))
    assert receipt["law"] == "target_gpu_idle_v2_20260902"
    assert receipt["target_compute_apps"] == []
    assert len(receipt["other_cards_recorded"]) == 1
    assert receipt["other_cards_recorded"][0]["index"] == "0"


def test_target_idle_guard_refuses_a_busy_target() -> None:
    with pytest.raises(trainer.TrainerError, match="target GPU 1 is busy"):
        trainer.require_train_authorization(
            gpu_authorized=True, gpu_index=1,
            cmdline_runner=_runner([("0", "GPU-aaa", "5", "0"),
                                    ("1", "GPU-bbb", "2354", "41")], APPS_IDLE))


def test_target_idle_guard_refuses_compute_apps_on_the_target() -> None:
    with pytest.raises(trainer.TrainerError, match="compute apps"):
        trainer.require_train_authorization(
            gpu_authorized=True, gpu_index=1, cmdline_runner=_runner(IDLE_ROWS, APPS_ON_GPU1))
    # apps on the OTHER card do not refuse the target run
    receipt = trainer.require_train_authorization(
        gpu_authorized=True, gpu_index=0,
        cmdline_runner=_runner(IDLE_ROWS, APPS_ON_GPU1))
    assert receipt["gpu_index"] == 0


def test_target_idle_guard_refuses_an_absent_target() -> None:
    with pytest.raises(trainer.TrainerError, match="absent from the query"):
        trainer.assert_target_gpu_idle(7, cmdline_runner=_runner(IDLE_ROWS, APPS_IDLE))


def test_query_compute_apps_parses_and_fails_closed() -> None:
    apps = trainer.query_compute_apps(cmdline_runner=lambda argv: APPS_ON_GPU1)
    assert apps[0]["gpu_uuid"] == "GPU-bbb" and apps[0]["pid"] == "783126"
    assert trainer.query_compute_apps(cmdline_runner=lambda argv: "") == []
    with pytest.raises(trainer.TrainerError):
        trainer.query_compute_apps(cmdline_runner=lambda argv: "too,few,fields\n")


def _load_cli_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_pit_m2_v1_cli", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_arm_flag_runs_one_arm_and_serial_default_runs_both(monkeypatch) -> None:
    cli_module = _load_cli_module()
    from tfpd_exploration.src.pit_m2_v1 import driver as driver_module
    from tfpd_exploration.src.pit_m2_v1 import trainer as trainer_module

    monkeypatch.setattr(trainer_module, "require_train_authorization",
                        lambda gpu_authorized, gpu_index=None, cmdline_runner=None: {
                            "gpu_authorized": True, "law": "stub"})
    recorded: list[str] = []
    monkeypatch.setattr(driver_module, "execute_arm",
                        lambda root, *, arm, gpu_index, gpu_authorized:
                        recorded.append(arm) or (f"terminal-{arm}", None))
    arguments = cli_module.build_parser().parse_args(
        ["--stage", "train", "--execute", "--gpu-authorized", "--gpu-index", "1",
         "--arm", "c1m"])
    assert arguments.arm == "c1m"
    exit_code = cli_module.main(
        ["--stage", "train", "--execute", "--gpu-authorized", "--gpu-index", "1",
         "--arm", "c1m"])
    assert exit_code == 0 and recorded == ["c1m"]
    exit_code = cli_module.main(
        ["--stage", "train", "--execute", "--gpu-authorized", "--gpu-index", "1"])
    assert exit_code == 0 and recorded == ["c1m", "t0m", "c1m"]


def test_query_gpu_rows_parses_and_fails_closed() -> None:
    rows = trainer.query_gpu_rows(cmdline_runner=lambda argv: _gpu_text(IDLE_ROWS))
    assert rows[0]["uuid"] == "GPU-aaa" and rows[1]["memory_used_mib"] == "10"
    with pytest.raises(trainer.TrainerError):
        trainer.query_gpu_rows(cmdline_runner=lambda argv: "garbage,line\n")
    with pytest.raises(trainer.TrainerError):
        trainer.query_gpu_rows(cmdline_runner=lambda argv: "")


def test_cli_attempt_is_inert_and_train_refuses() -> None:
    environment = {"PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT)}
    completed = subprocess.run(
        [sys.executable, str(CLI), "--stage", "attempt"],
        capture_output=True, text=True, env={**__import__("os").environ, **environment},
        timeout=120,
    )
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["stages"] == ["attempt", "smoke", "train", "phase3"]
    assert payload["imports_torch"] is False and payload["creates_root_or_receipt"] is False
    assert payload["sealed_trainer_binding"]["checkpoint_sha256"] == plan.SEALED_CHECKPOINT_SHA256

    refused = subprocess.run(
        [sys.executable, str(CLI), "--stage", "train", "--execute"],
        capture_output=True, text=True, env={**__import__("os").environ, **environment},
        timeout=120,
    )
    assert refused.returncode == 2
    assert "REFUSED" in refused.stdout


# ---------------------------------------------------------------------------
# Smoke equality logic on synthetic bodies
# ---------------------------------------------------------------------------

def _synthetic_arm_body(arm: str) -> dict[str, object]:
    stream = {
        "rng_stream_digest": "r" * 64,
        "torch_rng_stream_digest": "t" * 64,
        "batch_stream_digest": "b" * 64,
        "t4_bytes_stream_digest": "4" * 64,
        "rows": [{"step": index, "rng_state_after_sha256": f"{index:064d}"}
                 for index in range(plan.SMOKE_STEPS)],
    }
    operator = {
        "recorded_prefix_sequence": schedule.sequence(plan.SMOKE_STEPS)
        if arm == "c1m" else [33] * plan.SMOKE_STEPS,
        "effective_prefixes": schedule.sequence(plan.SMOKE_STEPS)
        if arm == "c1m" else [33] * plan.SMOKE_STEPS,
        "records": [{"visible_slice_sha256": "v" * 64}],
    }
    return {
        "arm": arm,
        "optimizer_steps": plan.SMOKE_STEPS,
        "stream_records": stream,
        "operator_snapshot": operator,
        "rng_draw_counts_during_steps": {"python": plan.SMOKE_STEPS, "numpy": 0, "torch": 0},
        "eval_dropout_inactive_proof": {"bit_identical": True},
        "teacher_eval_dropout_inert_proof": {"bit_identical": True,
                                             "distinct_p_draws_observed": True},
        "cuda_initialized": False,
        "authority": {
            "initial_student_state_sha256": "s" * 64,
            "initial_teacher_state_sha256": "T" * 64,
            "rng_digests": {"after_seed_everything": ["a", "b"]},
            "sampler_batch_stream_sha256": "S" * 64,
            "normalization_sha256": plan.NORMALIZATION_SHA256,
            "heldout_dataset_built": False,
        },
    }


def test_validate_smoke_equality_passes_on_matched_synthetic_pair() -> None:
    equality = smoke.validate_smoke_equality(_synthetic_arm_body("t0m"),
                                             _synthetic_arm_body("c1m"))
    assert all(value is True for value in equality["checks"].values()
               if not isinstance(value, list))
    assert equality["c1m_recorded_prefix_sequence"] == schedule.sequence(plan.SMOKE_STEPS)
    assert equality["dropout_p_draws_per_arm"] == plan.SMOKE_STEPS
    assert equality["target_path_resolved"] is False


def test_validate_smoke_equality_fails_on_any_drift() -> None:
    body = _synthetic_arm_body("c1m")
    body["authority"]["initial_student_state_sha256"] = "x" * 64
    with pytest.raises(smoke.SmokeError, match="initial_student_state_sha256"):
        smoke.validate_smoke_equality(_synthetic_arm_body("t0m"), body)
    rng_body = _synthetic_arm_body("c1m")
    rng_body["rng_draw_counts_during_steps"]["python"] = 7
    with pytest.raises(smoke.SmokeError, match="dropout_p_draw_counter_one_per_step_both_arms"):
        smoke.validate_smoke_equality(_synthetic_arm_body("t0m"), rng_body)
    aux_body = _synthetic_arm_body("c1m")
    aux_body["rng_draw_counts_during_steps"]["torch"] = 3
    with pytest.raises(smoke.SmokeError, match="dropout_p_aux_counts_equal_across_arms"):
        smoke.validate_smoke_equality(_synthetic_arm_body("t0m"), aux_body)
    teacher_body = _synthetic_arm_body("c1m")
    teacher_body["teacher_eval_dropout_inert_proof"]["bit_identical"] = False
    with pytest.raises(smoke.SmokeError, match="teacher_eval_dropout_inert_both_arms"):
        smoke.validate_smoke_equality(_synthetic_arm_body("t0m"), teacher_body)
    cycle_body = _synthetic_arm_body("c1m")
    cycle_body["operator_snapshot"]["recorded_prefix_sequence"] = [10] * plan.SMOKE_STEPS
    with pytest.raises(smoke.SmokeError, match="c1m_prefix_sequence_is_cycle"):
        smoke.validate_smoke_equality(_synthetic_arm_body("t0m"), cycle_body)
    cuda_body = _synthetic_arm_body("c1m")
    cuda_body["cuda_initialized"] = True
    with pytest.raises(smoke.SmokeError, match="cuda_never_initialized"):
        smoke.validate_smoke_equality(_synthetic_arm_body("t0m"), cuda_body)


# ---------------------------------------------------------------------------
# Regressions for the 2026-09-02 attempt-1 failures (phase3 duplicate leaf;
# stale validation metric tracker)
# ---------------------------------------------------------------------------

def test_frozen_rosters_match_the_sealed_anchors() -> None:
    assert plan.SURFACE_SESSION_ROSTERS["external_official_query"] \
        == plan.SURFACE_SESSION_ROSTERS["external_post30_local"]
    assert sorted(plan.EXTERNAL_SESSION_NAMES) == sorted(plan.F00M_ANCHOR["per_session_r2"])
    assert sorted(plan.EXTERNAL_SESSION_NAMES) == sorted(plan.F01M_ANCHOR["per_session_r2"])
    assert len(plan.WITHIN_SESSION_NAMES) == plan.WITHIN_SESSION_COUNT == 7
    assert len(plan.EXTERNAL_SESSION_NAMES) == plan.EXTERNAL_SESSION_COUNT == 6


def test_phase3_stage_names_are_unique_per_session() -> None:
    names = receipts.phase3_stage_names()
    # 2 arms x 2 paths x (6 external + 7 within) sessions = 52 score leaves.
    score_names = [name for name in names
                   if name.startswith("score_") and name.endswith(".json")]
    assert len(score_names) == 2 * 2 * (6 + 7) == 52
    assert len(set(names)) == len(names)  # the attempt-1 leaf collision cannot recur
    expected_leaf = "score_t0m_static_external_official_query_ses-2020-10-30-Run1.json"
    assert expected_leaf in names
    # the OLD (colliding) surface-level leaf name must NOT be part of the stage
    assert "score_t0m_static_external_official_query.json" not in names


def test_two_session_publish_into_one_root_does_not_collide(tmp_path) -> None:
    """The exact attempt-1 failure mode, replayed on a scratch root: publishing
    two DIFFERENT per-session rows must both succeed (and the old shared-leaf
    scheme must still raise FileExistsError and be cleaned up)."""
    from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1

    from tfpd_exploration.src.pit_m2_v1 import receipts as lane_receipts

    (tmp_path / "regression_phase3").mkdir(parents=True, exist_ok=True)
    spec = lane_receipts.StageSpec("regression_phase3/smoke")
    artifact = v1.ImmutableArtifactRoot.reserve(tmp_path, spec)  # type: ignore[arg-type]
    try:
        sessions = plan.EXTERNAL_SESSION_NAMES[:2]
        for index, session in enumerate(sessions):
            name = f"score_c1m_static_external_official_query_{session}.json"
            artifact.publish_json(name, {"session": session, "r2": 0.1 * index})
        with pytest.raises(Exception):
            shared = "score_c1m_static_external_official_query.json"
            artifact.publish_json(shared, {"session": sessions[0], "r2": 0.5})
            artifact.publish_json(shared, {"session": sessions[1], "r2": 0.6})  # O_EXCL
    finally:
        artifact.close()
    root = tmp_path / "regression_phase3" / "smoke"
    published = sorted(item.name for item in root.iterdir())
    for session in sessions:
        assert f"score_c1m_static_external_official_query_{session}.json" in published
    assert not (root / "score_c1m_static_external_official_query.json").exists()


class _FakeR2Metric:
    def __init__(self, total: int, value: float) -> None:
        self.total = total
        self._value = value

    def compute(self) -> float:
        return self._value


class _FakeValModule:
    def __init__(self, metrics: dict[str, _FakeR2Metric]) -> None:
        self.val_heldin_r2 = metrics


def test_epoch_val_metric_computes_fresh_and_skips_short_sessions() -> None:
    module = _FakeValModule({
        "a": _FakeR2Metric(total=100, value=0.5),
        "b": _FakeR2Metric(total=100, value=0.7),
        "short": _FakeR2Metric(total=2, value=99.0),  # skipped exactly as the module does
    })
    assert trainer.epoch_val_metric(module) == pytest.approx(0.6)
    assert trainer.epoch_val_metric(_FakeValModule({"short": _FakeR2Metric(1, 1.0)})) is None
    assert trainer.epoch_val_metric(_FakeValModule({})) is None


def test_attempt1_failure_disclosure_is_bound_into_attempt_payloads() -> None:
    assert "one leaf per (arm,path,surface,session)" in \
        plan.PHASE3_ATTEMPT1_DISCLOSURE["phase3_duplicate_leaf_publish"]
    assert "previous epoch's metric" in \
        plan.PHASE3_ATTEMPT1_DISCLOSURE["arm_stale_val_tracker"]
    payload = receipts.stage_attempt_payload("phase3", plan.PairSpec().sha256, {"closure_sha256": "0" * 64})
    assert payload["attempt_history_disclosure"] == dict(plan.PHASE3_ATTEMPT1_DISCLOSURE)
