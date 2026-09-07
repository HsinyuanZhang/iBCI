"""CAL-AUG V1 review-gate tests (no data, no CUDA).

Work order section 7 / guidance section 13: focused tests must be green before
any GPU process.  Covered here:

* the pinned sealed boundary (every ``plan.PINNED_SHA256`` literal matches the
  live bytes, the work order included);
* cycle arithmetic, counter semantics and every digest helper;
* hook gating: training-only slice, eval untouched, T0 identity, counter
  advances only on training-mode invocations, per-step recording window;
* stream preservation: the operator consumes no Python/NumPy/Torch RNG
  (monkeypatched counters stay at zero) and never mutates the caller's kwargs;
* the encoder ``push_trial`` probe (trial_count == M evidence) restores itself;
* mechanism gate boundary semantics (0.01 passes, 0.01 - 1e-13 fails; 18/27 vs
  17/27; the M30 safety clause) with the 1e-12 epsilon as a band, never a flip;
* deployment gate arithmetic incl. the "other budget >= 0" and M30/within
  safety clauses, and the combined disposition incl.
  ``MECHANISM_POSITIVE__DEPLOYMENT_INCONCLUSIVE``;
* the timeout guard raising and publishing an atomic ``CELL_FAILED`` receipt
  (stage, steps completed, ``terminal_published=false``);
* attempt-before-data ordering, both as the guard and in the runner source.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))
# ROOT last: tfpd_exploration/src must win `import src` over the
# streaming tree; the streaming components are either file-path loaded
# by the sealed arm runner or merged via src.__path__ inside tfpd_lane.
sys.path.insert(0, str(ROOT))

from src.cal_aug_v1 import deployment, hook, mechanism, plan, receipts, schedule  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def receipt_module():
    spec = importlib.util.spec_from_file_location(
        "tfpd_lane_receipt_under_test", ROOT / "src/tfpd_lane/receipt.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# pinned sealed boundary
# ---------------------------------------------------------------------------


def test_every_pinned_sha_literal_matches_the_live_bytes():
    table = receipts.verify_pinned_files(REPO)
    assert set(table) == set(plan.PINNED_SHA256)
    for relative, digest in table.items():
        assert digest == plan.PINNED_SHA256[relative]
        assert digest == _sha(REPO / relative)


def test_pinned_boundary_fails_closed_on_a_tampered_copy(tmp_path):
    victim = tmp_path / "run_admission_arm.py"
    victim.write_text("print('tampered')\n")
    bogus = dict(plan.PINNED_SHA256)
    bogus["tfpd_exploration/scripts/run_admission_arm.py"] = _sha(victim)
    from src.cal_aug_v1 import plan as plan_module

    original = dict(plan_module.PINNED_SHA256)
    plan_module.PINNED_SHA256 = bogus
    try:
        # the real file's bytes no longer match the tampered literal
        with pytest.raises(receipts.ReceiptError):
            receipts.verify_pinned_files(REPO)
    finally:
        plan_module.PINNED_SHA256 = original


def test_sealed_predecessors_sidecar_verified():
    table = receipts.verify_sealed_predecessors(REPO)
    assert set(table) == set(plan.SEALED_PREDECESSOR_FILES)
    for relative, digest in table.items():
        assert digest == plan.SEALED_PREDECESSOR_SHA256[relative]


# ---------------------------------------------------------------------------
# cycle arithmetic and digests
# ---------------------------------------------------------------------------


def test_parse_cycle_and_validation():
    assert schedule.parse_cycle("30,10,4") == (30, 10, 4)
    assert schedule.parse_cycle(" 30, 10 ,4 ") == (30, 10, 4)
    assert schedule.parse_cycle("7") == (7,)
    for bad in ("", "30,,4", "0,10", "31", "-3", "3.5", "x,y", "30;10"):
        with pytest.raises(schedule.ScheduleError):
            schedule.parse_cycle(bad)


def test_cycle_arithmetic_is_pure_index_math():
    cycle = (30, 10, 4)
    assert schedule.m_at(0, cycle) == 30
    assert schedule.m_at(1, cycle) == 10
    assert schedule.m_at(2, cycle) == 4
    assert schedule.m_at(3, cycle) == 30
    assert schedule.m_at(1_628_399, cycle) == cycle[1_628_399 % 3]
    assert schedule.prefix_sequence(7, cycle) == [30, 10, 4, 30, 10, 4, 30]
    assert schedule.effective_prefix_length(10, 30) == 10
    assert schedule.effective_prefix_length(30, 30) == 30
    with pytest.raises(schedule.ScheduleError):
        schedule.m_at(-1, cycle)


def test_digest_helpers_deterministic_and_order_sensitive():
    assert schedule.cycle_digest((30, 10, 4)) == schedule.cycle_digest((30, 10, 4))
    assert schedule.cycle_digest((30, 10, 4)) != schedule.cycle_digest((10, 30, 4))
    assert schedule.sequence_digest([30, 10, 4]) != schedule.sequence_digest([10, 30, 4])
    assert schedule.sequence_digest([30, 10, 4]) != schedule.sequence_digest([30, 10])
    # full-precision float stream: order and exact bits matter
    a = schedule.float_stream_digest([0.1, 0.2])
    assert a == schedule.float_stream_digest([0.1, 0.2])
    assert a != schedule.float_stream_digest([0.2, 0.1])
    order = schedule.batch_order_digest([[1, 2], [3]], ["sA", "sA", "sB"])
    assert order["n_batches"] == 2
    assert order["combined_sha256"] == schedule.batch_order_digest([[1, 2], [3]], ["sA", "sA", "sB"])[
        "combined_sha256"
    ]
    assert order["combined_sha256"] != schedule.batch_order_digest([[3], [1, 2]], ["sB", "sA", "sA"])[
        "combined_sha256"
    ]
    assert schedule.mapping_digest({"a": 1}) == schedule.mapping_digest({"a": 1})


def test_tensor_digest_matches_the_lane_law_and_normalizes_signed_zero():
    spec = importlib.util.spec_from_file_location(
        "tfpd_lane_arm_common_for_digest", ROOT / "src/tfpd_lane/arm_common.py"
    )
    arm_common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arm_common)

    tensor = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    assert schedule.tensor_digest(tensor) == arm_common.tensor_sha256(tensor)
    assert schedule.tensor_digest(tensor[:, :2]) == arm_common.tensor_sha256(tensor[:, :2])
    minus_zero = torch.full((2, 2), -0.0)
    plus_zero = torch.full((2, 2), 0.0)
    assert schedule.tensor_digest(minus_zero) == schedule.tensor_digest(plus_zero)
    assert schedule.tensor_digest(torch.zeros(0)) != schedule.tensor_digest(torch.zeros(1))


# ---------------------------------------------------------------------------
# the prefix operator
# ---------------------------------------------------------------------------


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.received = []

    def forward(self, neural, calib_trials=None, side_features=None):
        self.received.append(calib_trials)
        return self.scale * neural.sum()


def _batch(b=2, trials=30, units=5):
    return (
        torch.rand(b, 50, units),
        torch.rand(b, trials, 100, units),
        torch.randn(b, units, 4),
    )


def test_c1_training_mode_slices_the_scheduled_prefix():
    model = ToyModel()
    operator = hook.CalPrefixOperator("c1", (30, 10, 4), record_steps=10)
    operator.attach(model)
    neural, calib, side = _batch()
    calib_snapshot = calib.clone()
    model.train()
    for expected_m in (30, 10, 4, 30):
        model(neural, calib_trials=calib, side_features=side)
        assert tuple(model.received[-1].shape) == (2, expected_m, 100, 5)
        assert model.received[-1] is not calib
        assert torch.equal(model.received[-1], calib[:, :expected_m])
    operator.detach()
    assert operator.training_invocations == 4
    assert [r["scheduled_m"] for r in operator.records] == [30, 10, 4, 30]
    assert [r["effective_m"] for r in operator.records] == [30, 10, 4, 30]
    # the M=30 slice digest equals the full-block digest (chronological prefix)
    assert operator.records[0]["visible_slice_sha256"] == operator.records[0]["full_block_sha256"]
    assert operator.records[1]["visible_slice_sha256"] != operator.records[1]["full_block_sha256"]
    # the operator never mutated the caller's calibration block
    assert torch.equal(calib, calib_snapshot)


def test_eval_mode_is_untouched_and_counter_frozen():
    model = ToyModel()
    operator = hook.CalPrefixOperator("c1", (30, 10, 4), record_steps=10)
    operator.attach(model)
    neural, calib, side = _batch()
    model.eval()
    for _ in range(3):
        model(neural, calib_trials=calib, side_features=side)
        assert model.received[-1] is calib  # exact tensor identity, no slice
    assert operator.training_invocations == 0
    assert operator.eval_invocations == 3
    assert operator.records == []
    # then training: counter resumes from the same arithmetic domain
    model.train()
    model(neural, calib_trials=calib, side_features=side)
    assert operator.training_invocations == 1
    assert operator.records[0]["scheduled_m"] == 30
    operator.detach()


def test_t0_identity_operator_disabled_but_counter_runs():
    model = ToyModel()
    operator = hook.CalPrefixOperator("t0", (30, 10, 4), record_steps=10)
    operator.attach(model)
    neural, calib, side = _batch()
    model.train()
    for _ in range(4):
        model(neural, calib_trials=calib, side_features=side)
        assert model.received[-1] is calib  # identity: same tensor object
    assert operator.training_invocations == 4
    assert [r["scheduled_m"] for r in operator.records] == [30, 10, 4, 30]
    assert [r["effective_m"] for r in operator.records] == [30, 30, 30, 30]
    assert all(r["operator_applied"] is False for r in operator.records)
    assert all(
        r["visible_slice_sha256"] == r["full_block_sha256"] for r in operator.records
    )
    operator.detach()


def test_recording_window_is_bounded_and_counter_is_not():
    model = ToyModel()
    operator = hook.CalPrefixOperator("c1", (30, 10, 4), record_steps=2)
    operator.attach(model)
    neural, calib, side = _batch()
    model.train()
    for _ in range(5):
        model(neural, calib_trials=calib, side_features=side)
    assert operator.training_invocations == 5
    assert len(operator.records) == 2
    assert operator.snapshot()["training_invocations"] == 5
    assert operator.snapshot()["n_recorded"] == 2
    operator.detach()


def test_operator_consumes_no_python_numpy_or_torch_rng(monkeypatch):
    calls = {"python": 0, "numpy": 0, "torch": 0}

    def counting_python(*args, **kwargs):
        calls["python"] += 1
        return 0.5

    def counting_numpy(*args, **kwargs):
        calls["numpy"] += 1
        return 0.5

    def counting_torch(*args, **kwargs):
        calls["torch"] += 1
        return torch.ones(1)

    import numpy as np
    import random

    # fixtures must be built BEFORE the RNG monkeypatch: _batch itself uses
    # torch.rand, and the patched stubs return 1-element tensors
    prepared = {arm: (ToyModel(), *_batch()) for arm in ("t0", "c1")}
    monkeypatch.setattr(random, "uniform", counting_python, raising=False)
    monkeypatch.setattr(random, "random", counting_python, raising=False)
    monkeypatch.setattr(random, "randrange", counting_python, raising=False)
    for name in ("uniform", "normal", "rand", "randint", "choice", "shuffle"):
        monkeypatch.setattr(np.random, name, counting_numpy, raising=False)
    for name in ("rand", "randn", "randint", "randperm", "normal", "poisson", "multinomial"):
        monkeypatch.setattr(torch, name, counting_torch, raising=False)

    for arm in ("t0", "c1"):
        model, neural, calib, side = prepared[arm]
        operator = hook.CalPrefixOperator(arm, (30, 10, 4), record_steps=8)
        operator.attach(model)
        model.train()
        for _ in range(6):
            model(neural, calib_trials=calib, side_features=side)
        model.eval()
        model(neural, calib_trials=calib, side_features=side)
        operator.detach()
    assert calls == {"python": 0, "numpy": 0, "torch": 0}


def test_c1_requires_calib_trials_in_training_kwargs():
    model = ToyModel()
    operator = hook.CalPrefixOperator("c1", (30, 10, 4))
    operator.attach(model)
    model.train()
    with pytest.raises(hook.HookError):
        model(torch.rand(2, 50, 5))
    operator.detach()


def test_read_only_neural_digest_hook_is_transparent():
    model = ToyModel()
    store: list = []
    handle = model.register_forward_pre_hook(
        hook.make_read_only_neural_digest_hook(store, limit=2), with_kwargs=True
    )
    neural, calib, side = _batch()
    model.train()
    model(neural, calib_trials=calib, side_features=side)
    assert model.received[-1].shape[1] == 30  # kwargs untouched by the recorder
    model(neural, calib_trials=calib, side_features=side)
    model(neural, calib_trials=calib, side_features=side)
    handle.remove()
    assert len(store) == 2  # limit respected
    assert all(entry["neural_sha256"] == schedule.tensor_digest(neural) for entry in store)
    assert all(entry["training"] for entry in store)


class ToyEncoder:
    def __init__(self):
        self.trials_seen = []
        self.batch_calls = 0

    def push_trial(self, state, trial):
        self.trials_seen.append(trial)
        return state

    def forward_batch(self, calib_trials, side_features=None):
        self.batch_calls += 1
        for index in range(calib_trials.shape[1]):
            self.push_trial({}, calib_trials[:, index])
        return {}


def test_encoder_trial_probe_counts_and_restores():
    encoder = ToyEncoder()
    with hook.EncoderTrialProbe(encoder) as probe:
        for _ in range(7):
            encoder.push_trial({}, torch.zeros(3))
        assert probe.count == 7
        assert probe.mark_forward_boundary() == 7
        encoder.push_trial({}, torch.zeros(3))
        assert probe.mark_forward_boundary() == 1
    assert "push_trial" not in encoder.__dict__
    assert encoder.push_trial({}, torch.zeros(3)) == {}  # original bound method restored


# ---------------------------------------------------------------------------
# mechanism gate boundary semantics
# ---------------------------------------------------------------------------


def _registration(**overrides):
    inputs = {
        "recovery_m10": 0.02, "recovery_m4": 0.02,
        "positive_sessions_m10": 20, "positive_sessions_m4": 20,
        "c1_minus_t0_m30_mean": 0.0,
    }
    inputs.update(overrides)
    return mechanism.registration_gate(**inputs)


def test_mechanism_gate_boundary_exact_ge():
    passing = _registration(recovery_m10=0.01, recovery_m4=-1.0)
    assert passing["recovery_margin_m10"]["meets_margin"] is True
    assert passing["passed"] is True
    assert passing["disposition"] == plan.MECHANISM_REGISTERED

    below = _registration(recovery_m10=0.01 - 1e-13, recovery_m4=-1.0)
    assert below["recovery_margin_m10"]["meets_margin"] is False
    assert below["recovery_margin_m10"]["within_epsilon_band_of_boundary"] is True
    assert below["passed"] is False
    assert below["disposition"] == plan.MECHANISM_NULL

    clearly_below = _registration(recovery_m10=0.01 - 1e-9, recovery_m4=-1.0)
    assert clearly_below["recovery_margin_m10"]["within_epsilon_band_of_boundary"] is False


def test_mechanism_gate_breadth_boundary_18_of_27():
    assert _registration(positive_sessions_m10=18)["passed"] is True
    assert _registration(positive_sessions_m10=17, recovery_m4=-1.0)["passed"] is False
    assert _registration(positive_sessions_m10=17, recovery_m4=-1.0)["breadth"]["m10_breadth_passed"] is False
    # the M4 lead can carry the gate on its own
    assert _registration(
        recovery_m10=-1.0, recovery_m4=0.05, positive_sessions_m4=19
    )["passed"] is True
    assert _registration(
        recovery_m10=-1.0, recovery_m4=0.05, positive_sessions_m4=17
    )["passed"] is False


def test_mechanism_gate_m30_safety_clause():
    assert _registration(c1_minus_t0_m30_mean=-0.01)["passed"] is True
    failed = _registration(c1_minus_t0_m30_mean=-0.01 - 1e-13)
    assert failed["passed"] is False
    assert failed["m30_safety_c1_minus_t0"]["within_epsilon_band_of_boundary"] is True
    assert _registration(c1_minus_t0_m30_mean=-0.02)["passed"] is False


def test_prefix_degradation_and_recovery_arithmetic():
    assert mechanism.prefix_degradation(0.30, 0.40) == pytest.approx(-0.10)
    r2 = {
        "c1": {"s1": 0.35, "s2": 0.25}, "c1_30": {"s1": 0.40, "s2": 0.30},
        "t0": {"s1": 0.30, "s2": 0.20}, "t0_30": {"s1": 0.40, "s2": 0.30},
    }
    row = mechanism.per_session_recovery(r2["c1"], r2["c1_30"], r2["t0"], r2["t0_30"])
    assert row["per_session"]["s1"] == pytest.approx(0.05)
    assert row["per_session"]["s2"] == pytest.approx(0.05)
    assert row["equal_session_mean_recovery"] == pytest.approx(0.05)
    assert row["positive_sessions"] == 2
    with pytest.raises(mechanism.MechanismError):
        mechanism.per_session_recovery(
            {"s1": 0.1}, {"s1": 0.2, "s3": 0.0}, {"s1": 0.3}, {"s1": 0.4}
        )
    assert mechanism.equal_session_mean([0.2, 0.4]) == pytest.approx(0.3)
    with pytest.raises(mechanism.MechanismError):
        mechanism.equal_session_mean([])


# ---------------------------------------------------------------------------
# deployment gate arithmetic
# ---------------------------------------------------------------------------


def _ext(mean, pos=15, lb=0.01):
    return {"mean": mean, "n_positive": pos, "bootstrap_lb": lb}


def _within(m4=0.0, m10=0.0, m30=0.0):
    return {"4": m4, "10": m10, "30": m30}


def test_lower_gate_passes_on_a_qualifying_lead_budget():
    gate = deployment.lower_continuation_gate(
        external_m4=_ext(0.02, 11, 0.005),
        external_m10=_ext(0.001, 8),
        external_m30_mean=-0.01,
        within_means_by_budget=_within(),
    )
    assert gate["lead_m4"] is True and gate["lead_m10"] is False
    assert gate["other_low_budget_clause"]["passed"] is True  # 0.001 >= 0
    assert gate["external_m30_safety"]["meets_margin"] is True
    assert gate["passed"] is True
    assert gate["disposition"] == plan.LOWER_CONTINUATION_PASSED


def test_lower_gate_boundary_is_exact():
    at_margin = deployment.lower_continuation_gate(
        external_m4=_ext(0.015, 10, 0.0), external_m10=_ext(0.0),
        external_m30_mean=0.0, within_means_by_budget=_within(),
    )
    assert at_margin["passed"] is True  # exactly +0.015, 10/15, LB exactly 0
    below = deployment.lower_continuation_gate(
        external_m4=_ext(0.015 - 1e-13, 10, 0.0), external_m10=_ext(0.0),
        external_m30_mean=0.0, within_means_by_budget=_within(),
    )
    assert below["passed"] is False
    lb_negative = deployment.lower_continuation_gate(
        external_m4=_ext(0.02, 11, -1e-9), external_m10=_ext(0.0),
        external_m30_mean=0.0, within_means_by_budget=_within(),
    )
    assert lb_negative["passed"] is False


def test_lower_gate_other_budget_must_not_regress():
    gate = deployment.lower_continuation_gate(
        external_m4=_ext(0.02, 11, 0.005),
        external_m10=_ext(-1e-9, 8),
        external_m30_mean=0.0,
        within_means_by_budget=_within(),
    )
    assert gate["lead_m4"] is True
    assert gate["other_low_budget_clause"]["passed"] is False
    assert gate["passed"] is False


def test_lower_gate_m30_and_within_safety_clauses():
    m30 = deployment.lower_continuation_gate(
        external_m4=_ext(0.02, 11), external_m10=_ext(0.0),
        external_m30_mean=-0.02, within_means_by_budget=_within(),
    )
    assert m30["passed"] is True  # exactly at the -0.02 boundary
    m30_breach = deployment.lower_continuation_gate(
        external_m4=_ext(0.02, 11), external_m10=_ext(0.0),
        external_m30_mean=-0.02 - 1e-13, within_means_by_budget=_within(),
    )
    assert m30_breach["passed"] is False
    within_breach = deployment.lower_continuation_gate(
        external_m4=_ext(0.02, 11), external_m10=_ext(0.0),
        external_m30_mean=0.0, within_means_by_budget=_within(m10=-0.02 - 1e-13),
    )
    assert within_breach["passed"] is False
    assert within_breach["within_safety_by_budget"]["10"]["meets_margin"] is False
    # every reported budget must be present (never average budgets)
    with pytest.raises(deployment.DeploymentError):
        deployment.lower_continuation_gate(
            external_m4=_ext(0.02, 11), external_m10=_ext(0.0),
            external_m30_mean=0.0, within_means_by_budget={"4": 0.0, "10": 0.0},
        )


def test_primary_gate_needs_0p03_and_all_safety():
    weak_lead = deployment.primary_claim_gate(
        external_m4=_ext(0.02, 11, 0.01), external_m10=_ext(0.001),
        external_m30_mean=0.0, within_means_by_budget=_within(),
    )
    assert weak_lead["passed"] is False  # +0.02 clears lower, not primary
    strong = deployment.primary_claim_gate(
        external_m4=_ext(0.03, 10, 0.01), external_m10=_ext(0.0),
        external_m30_mean=-0.019, within_means_by_budget=_within(m4=-0.019),
    )
    assert strong["passed"] is True
    assert strong["disposition"] == plan.PRIMARY_CLAIM_PASSED
    at_boundary = deployment.primary_claim_gate(
        external_m4=_ext(0.03 - 1e-13, 10, 0.01), external_m10=_ext(0.0),
        external_m30_mean=0.0, within_means_by_budget=_within(),
    )
    assert at_boundary["passed"] is False


def test_combined_disposition_including_mechanism_positive_inconclusive():
    lower = {"passed": False}
    primary = {"passed": False}
    combined = deployment.combined_disposition(
        mechanism_gate={"passed": True}, lower_gate=lower, primary_gate=primary
    )
    assert combined["status"] == plan.MECHANISM_POSITIVE_DEPLOYMENT_INCONCLUSIVE
    assert "stops C1 expansion" in combined["stop_rule"]
    null = deployment.combined_disposition(
        mechanism_gate={"passed": False}, lower_gate=lower, primary_gate=primary
    )
    assert null["status"] == plan.LOWER_CONTINUATION_FAILED
    advanced = deployment.combined_disposition(
        mechanism_gate={"passed": True}, lower_gate={"passed": True}, primary_gate=primary
    )
    assert advanced["status"] == plan.LOWER_CONTINUATION_PASSED


def test_gate_inputs_from_summary_shape():
    summaries = {
        f"{surface}:m{budget}": {
            "equal_session_mean_delta": 0.01 * (1 if surface == "external" else -1),
            "positive_sessions": 9,
            "bootstrap_lb": 0.0,
            "n_sessions": 15 if surface == "external" else 6,
        }
        for surface in ("within", "external")
        for budget in plan.BUDGETS
    }
    gate_inputs = deployment.gate_inputs_from_summary(summaries)
    assert gate_inputs["external_m4"]["mean"] == pytest.approx(0.01)
    assert gate_inputs["external_m4"]["n_positive"] == 9
    assert gate_inputs["within_means_by_budget"] == {"4": -0.01, "10": -0.01, "30": -0.01}


# ---------------------------------------------------------------------------
# timeout guard + CELL_FAILED receipt
# ---------------------------------------------------------------------------


def test_deadline_guard_raises_with_stage_and_steps():
    guard = receipts.DeadlineGuard(timeout_seconds=0.0)
    guard.stage = "training_epoch_3"
    guard.steps_completed = 101_976
    with pytest.raises(receipts.CalAugTimeout) as info:
        guard.check("training_epoch_3", 101_976)
    assert info.value.stage == "training_epoch_3"
    assert info.value.steps_completed == 101_976
    assert info.value.limit_s == 0.0
    fresh = receipts.DeadlineGuard(timeout_seconds=3600.0)
    fresh.check("training_start", 0)  # must not raise


def test_deadline_loader_yields_and_fires_the_guard():
    guard = receipts.DeadlineGuard(timeout_seconds=0.0)
    seen = []
    loader = receipts.DeadlineLoader([{"b": 1}, {"b": 2}], guard, on_step=seen.append)
    with pytest.raises(receipts.CalAugTimeout):
        for _ in loader:
            pass
    generous = receipts.DeadlineGuard(timeout_seconds=60.0)
    loader = receipts.DeadlineLoader([{"b": 1}, {"b": 2}], generous, on_step=seen.append)
    assert list(loader) == [{"b": 1}, {"b": 2}]
    assert seen == [1, 2]
    assert generous.steps_completed == 2
    loader.steps_offset = 100
    assert list(loader) == [{"b": 1}, {"b": 2}]
    assert seen == [1, 2, 1, 2]
    assert generous.steps_completed == 102


def test_publish_failure_writes_atomic_cell_failed(tmp_path, receipt_module):
    sha = receipts.publish_failure(
        tmp_path, receipt_module,
        {
            "schema": plan.SCHEMA + "_cell",
            "arm": "c1",
            "stage": "training_batch_33924",
            "steps_completed": 1_628_400,
            "timeout": True,
            "failure": {"kind": "CalAugTimeout", "detail": "wall clock"},
        },
    )
    path = tmp_path / "terminal.json"
    body = json.loads(path.read_text())
    assert body["status"] == receipts.CELL_FAILED
    assert body["terminal_published"] is False
    assert body["stage"] == "training_batch_33924"
    assert body["steps_completed"] == 1_628_400
    assert sha == _sha(path)
    assert (path.with_name("terminal.json.sha256")).read_text().startswith(sha)
    assert (path.stat().st_mode & 0o777) == 0o444
    # immutable: the second failure publish can never overwrite the receipt
    again = receipts.publish_failure(tmp_path, receipt_module, {"schema": "again"})
    assert again is None
    assert json.loads(path.read_text())["steps_completed"] == 1_628_400


def test_probe_payload_projections():
    payload = receipts.probe_payload(
        arm="c1", measured_steps=100, warmup_steps=20, seconds=1.0
    )
    assert payload["steps_per_second"] == pytest.approx(100.0)
    assert payload["projected_seconds_per_arm"] == pytest.approx(plan.TOTAL_OPTIMIZER_STEPS / 100.0)
    assert payload["projected_pair_gpu_hours"] == pytest.approx(
        2 * plan.TOTAL_OPTIMIZER_STEPS / 100.0 / 3600.0
    )
    # 100 steps/s -> ~9.05 pair GPU-hours: inside the 12 h planning ceiling
    assert payload["projected_pair_gpu_hours"] < plan.PLANNING_CEILING_GPU_HOURS
    assert payload["within_planning_ceiling"] is True
    slow = receipts.probe_payload(arm="t0", measured_steps=100, warmup_steps=20, seconds=2.0)
    assert slow["steps_per_second"] == pytest.approx(50.0)
    assert slow["projected_pair_gpu_hours"] > plan.PLANNING_CEILING_GPU_HOURS
    assert slow["within_planning_ceiling"] is False
    with pytest.raises(receipts.ReceiptError):
        receipts.probe_payload(arm="t0", measured_steps=0, warmup_steps=20, seconds=1.0)


# ---------------------------------------------------------------------------
# attempt-before-data ordering
# ---------------------------------------------------------------------------


def test_require_attempt_gates_data_access(tmp_path):
    with pytest.raises(receipts.ReceiptError):
        receipts.require_attempt(tmp_path, "data_materialization")
    receipts.publish_attempt(
        tmp_path, _receipt_writer(),
        {"schema": "x", "status": "ATTEMPT_PUBLISHED"},
    )
    receipts.require_attempt(tmp_path, "data_materialization")  # now authorized
    guard = receipts.attempt_guard(tmp_path)
    guard("build_datamodule")


def _receipt_writer():
    spec = importlib.util.spec_from_file_location(
        "tfpd_lane_receipt_for_attempt", ROOT / "src/tfpd_lane/receipt.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_sources_publish_attempt_before_any_data_access():
    for name in ("run_cal_aug_cell_v1.py", "run_cal_aug_smoke_v1.py",
                 "run_cal_aug_mechanism_v1.py", "run_cal_aug_deployment_v1.py"):
        source = (ROOT / "scripts" / name).read_text()
        attempt_pos = source.index("receipts.publish_attempt(")
        data_pos = source.index("arm_runner.build_datamodule") if "build_datamodule" in source \
            else source.index('receipts.require_attempt(')
        assert attempt_pos < data_pos, f"{name}: attempt must precede data access"
        assert "receipts.verify_pinned_files" in source
        assert source.index("receipts.verify_pinned_files") < attempt_pos


def test_trainer_cli_rejects_bad_arm_without_touching_data():
    import os
    import subprocess

    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_cal_aug_cell_v1.py"), "--arm", "x9"],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300,
    )
    assert proc.returncode == 2  # argparse reject, no receipts, no data


def test_trainer_cli_rejects_existing_root(tmp_path):
    import os
    import subprocess

    (tmp_path / "t0_operator_disabled").mkdir(parents=True)
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/run_cal_aug_cell_v1.py"),
            "--arm", "t0", "--device", "cpu", "--out-root", str(tmp_path),
        ],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300,
    )
    assert proc.returncode == 2


def test_deployment_driver_enforces_the_cpu_environment(tmp_path):
    import os
    import subprocess

    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="0")  # wrong on purpose
    env.pop("SUBC_DATA_ROOT", None)
    proc = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/run_cal_aug_deployment_v1.py"),
            "--out-root", str(tmp_path),
        ],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300,
    )
    assert proc.returncode == 3
    assert "deployment environment drift" in proc.stderr


def test_mechanism_driver_requires_arm_terminal_receipt(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "cal_aug_mechanism_driver_under_test", ROOT / "scripts/run_cal_aug_mechanism_v1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arm_dir = tmp_path / "t0_operator_disabled"
    arm_dir.mkdir()
    with pytest.raises(SystemExit):
        module._require_arm_terminal(arm_dir)
    (arm_dir / "terminal.json").write_text(json.dumps({"status": "CAL_AUG_CELL_FAILED"}))
    with pytest.raises(SystemExit):
        module._require_arm_terminal(arm_dir)
    (arm_dir / "terminal.json").write_text(json.dumps({"status": "CAL_AUG_CELL_TERMINAL"}))
    assert module._require_arm_terminal(arm_dir)["status"] == "CAL_AUG_CELL_TERMINAL"


# ---------------------------------------------------------------------------
# plan-level invariants
# ---------------------------------------------------------------------------


def test_plan_freezes_the_declared_budget_and_cycle():
    assert plan.DEFAULT_CYCLE == (30, 10, 4)
    assert plan.EPOCHS * plan.STEPS_PER_EPOCH == plan.TOTAL_OPTIMIZER_STEPS == 1_628_400
    assert plan.READOUT_PREFIXES == (30, 10, 4)
    assert plan.MECHANISM_POSITIVE_SOURCE_SESSIONS_REQUIRED == 18
    assert plan.MECHANISM_RECOVERY_MARGIN_R2 == 0.01
    assert plan.MECHANISM_M30_SAFETY_MARGIN_R2 == -0.01
    assert plan.LOWER_CONTINUATION_DELTA_R2 == 0.015
    assert plan.PRIMARY_DELTA_R2 == 0.03
    assert plan.EXTERNAL_POSITIVE_REQUIRED == 10
    assert plan.EXTERNAL_M30_SAFETY_MARGIN_R2 == -0.02
    assert plan.WITHIN_SAFETY_MARGIN_R2 == -0.02
    assert plan.GATE_BOUNDARY_EPSILON == 1e-12
    assert plan.BOUND_GPU_UUID.startswith("GPU-")
    assert plan.HARD_TIMEOUT_SECONDS == 8 * 3600
    for relative in plan.OWNED_PATHS:
        assert (REPO / relative).is_file(), relative


def test_gpu_binding_skipped_on_cpu():
    binding = receipts.gpu_binding("cpu", require_uuid=None)
    assert binding["mode"].startswith("cpu")
