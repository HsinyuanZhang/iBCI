"""Synthetic/no-data/no-CUDA tests for the M1 matched T0/C1 50-epoch prefix pair."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest

from tfpd_exploration.src.m1_t0c1_prefix_v1_50ep import lr_schedule, phase3, plan, probe, receipts


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "tfpd_exploration/scripts/run_m1_t0c1_prefix_v1_50ep.py"


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


# ---------------------------------------------------------------------------
# LR law
# ---------------------------------------------------------------------------


def test_lr_boundary_values() -> None:
    assert lr_schedule.lr_at(0) == 1e-5
    assert lr_schedule.lr_at(9902) == pytest.approx(1e-4, abs=1e-18, rel=0.0)
    assert lr_schedule.lr_at(247549) == pytest.approx(1e-6, rel=1e-9, abs=1e-12)


def test_lr_warmup_strictly_increasing_and_cosine_strictly_decreasing() -> None:
    previous = lr_schedule.lr_at(0)
    for step in range(1, 9902):
        current = lr_schedule.lr_at(step)
        assert current > previous
        previous = current
    previous = lr_schedule.lr_at(9902)
    for step in range(9903, 247550):
        current = lr_schedule.lr_at(step)
        assert current < previous
        previous = current


def test_lr_out_of_range_raises() -> None:
    with pytest.raises(lr_schedule.LRScheduleError):
        lr_schedule.lr_at(-1)
    with pytest.raises(lr_schedule.LRScheduleError):
        lr_schedule.lr_at(247550)
    with pytest.raises(lr_schedule.LRScheduleError):
        lr_schedule.lr_at(True)  # type: ignore[arg-type]  # bool is not int


def test_lr_at_is_independent_of_arm() -> None:
    first = [lr_schedule.lr_at(step) for step in (0, 9902, 247549)]
    second = [lr_schedule.lr_at(step) for step in (0, 9902, 247549)]
    assert first == second


def test_apply_sets_every_param_group() -> None:
    class _FakeOptimizer:
        param_groups = [{"lr": 0.0, "name": "a"}, {"lr": 9.9, "name": "b"}]

    optimizer = _FakeOptimizer()
    applied = lr_schedule.apply(optimizer, 9902)
    assert applied == pytest.approx(1e-4, abs=1e-18, rel=0.0)
    assert optimizer.param_groups[0]["lr"] == applied
    assert optimizer.param_groups[1]["lr"] == applied


def test_lr_at_consumes_no_rng() -> None:
    random.seed(12345)
    before = random.getstate()
    for step in range(100):
        lr_schedule.lr_at(step)
    assert random.getstate() == before


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def test_plan_frozen_literals() -> None:
    assert plan.EPOCHS == 50
    assert plan.TOTAL_OPTIMIZER_STEPS == 247550
    assert plan.CHECKPOINT_EPOCH_INDICES == (9, 19, 29, 39, 49)
    assert plan.CYCLE == (10, 5, 2)
    assert plan.DROPOUT_PROOF["block_sha256_both_trees"] == (
        "eacb0402448636cd0e757e9d1a28065209a5decacdbe68fe73722f634f858884"
    )
    assert plan.PREDECESSOR_20EP_ARM_TERMINAL_SHA256["t0"] == (
        "4aea40a317047beec059505231bb9996190c4eb2235392d08a3c948cf7ab9aa9"
    )
    assert plan.PREDECESSOR_20EP_ARM_TERMINAL_SHA256["c1"] == (
        "cbef49e8e356103c56a9ffa673a268de29e1c5564fc5957c26474b1cf065da99"
    )
    assert plan.PREDECESSOR_20EP_PHASE3_TERMINAL_SHA256 == (
        "821818b1ef9068aed87b437af3a8363cf2226e7778776dd9c6c10746bf22f096"
    )
    assert plan.HARD_TIMEOUT_SECONDS_PER_ARM == 43200
    assert plan.LINEAR_PROJECTION_50EP_HOURS == {"t0": 6.29, "c1": 5.43}
    assert plan.POST_MEMO_SECONDS_PER_STEP["t0"] == 0.08577056604618638
    assert plan.POST_MEMO_SECONDS_PER_STEP["c1"] == 0.07313887172792367
    assert plan.PRECEDENT_20EP_FULL_RUN_SECONDS_PER_STEP == {"t0": 0.10018, "c1": 0.08771}
    assert plan.DROPOUT_PROOF["hash_convention"]["trailing_newline"] is False
    assert plan.DROPOUT_PROOF["hash_convention"]["byte_count"] == 508
    assert plan.DROPOUT_PROOF["hash_convention"]["without_trailing_newline_sha256"] == (
        "eacb0402448636cd0e757e9d1a28065209a5decacdbe68fe73722f634f858884"
    )
    assert plan.RESULT_ROOT_RELATIVE == "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep"
    assert plan.SMOKE_ROOT_RELATIVE == "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/smoke"
    assert plan.ARM_ROOT_RELATIVE["t0"] == "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/t0"
    assert plan.ARM_ROOT_RELATIVE["c1"] == "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/c1"
    assert plan.PROBE_ROOT_RELATIVE == "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/probe"
    assert plan.PHASE3_ROOT_RELATIVE == "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/phase3_table"
    assert plan.HELDIN_TRAINING_SESSIONS == ("20120926", "20120927", "20120928")
    assert plan.LAUNCHER_RELATIVE in plan.OWNED_PATHS
    assert plan.LAUNCHER_RELATIVE == "tfpd_exploration/scripts/launch_m1_t0c1_prefix_v1_50ep.py"
    assert plan.ARM_CONCURRENCY == "concurrent_pair_on_gpu1"
    assert plan.MAX_GPU1_USED_MEMORY_MIB_BEFORE_START == 8192
    assert plan.CONCURRENT_ARM_STAGES == ("t0", "c1")
    assert plan.EXCLUSIVE_CARD_STAGES == ("smoke", "probe", "phase3")
    assert plan.ARM_CONCURRENCY_LAW["runtime_abort_heuristic_unchanged"] is True
    assert plan.ARM_CONCURRENCY_LAW["runtime_abort_estimate_seconds"] == 40000
    assert plan.ARM_CONCURRENCY_LAW["max_used_memory_mib_before_start"] == 8192
    assert "NOT comparable to the sealed 20-epoch reference timings" in (
        plan.ARM_CONCURRENCY_TIMING_DISCLOSURE
    )
    assert "cudnn.benchmark is off" in plan.ARM_CONCURRENCY_TIMING_DISCLOSURE
    assert plan.PairSpec50().payload()["arm_concurrency_law"] == plan.ARM_CONCURRENCY_LAW
    assert plan.VAL_HELDOUT_ACCESS_LAW["law_sha256"] == (
        "24205e652628aaf428e4d5bc27cad9c6386589c719db02613e9e3870c7fd341a"
    )
    first = plan.PairSpec50().sha256
    second = plan.PairSpec50().sha256
    assert first == second
    assert len(first) == 64


def test_plan_import_does_not_import_torch() -> None:
    process = subprocess.run(
        [sys.executable, "-c",
         "import sys\n"
         f"sys.path.insert(0, {str(ROOT)!r})\n"
         "from tfpd_exploration.src.m1_t0c1_prefix_v1_50ep import plan\n"
         "assert 'torch' not in sys.modules\n"
         "payload = plan.dry_plan()\n"
         "assert payload['imports_torch'] is False\n"
         "assert payload['initializes_cuda'] is False\n"
         "assert payload['creates_root_or_receipt'] is False\n"
         "print('ok')\n"],
        text=True, capture_output=True, check=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "CUDA_VISIBLE_DEVICES": ""},
    )
    assert process.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------


def test_arm_stage_names_cover_fifty_epochs_and_five_checkpoints() -> None:
    names = receipts.arm_stage_names()
    for index in range(50):
        assert f"epoch_{index:02d}.json" in names
        assert f"epoch_{index:02d}.json.sha256" in names
    assert "epoch_50.json" not in names
    assert "epoch_50.json.sha256" not in names
    for epoch_index in plan.CHECKPOINT_EPOCH_INDICES:
        filename = plan.epoch_checkpoint_filename(epoch_index)
        assert filename in names
        assert f"{filename}.sha256" in names
    assert "checkpoint_best_source_train_loss.pt" in names
    assert "checkpoint_best_source_train_loss.pt.sha256" in names


def test_probe_stage_names_cover_seventy_cells() -> None:
    names = receipts.probe_stage_names()
    score_bodies = [name for name in names if name.startswith("score_") and not name.endswith(".sha256")]
    assert len(score_bodies) == 70
    for arm in plan.ARMS:
        for epoch_index in plan.CHECKPOINT_EPOCH_INDICES:
            for session_id in plan.PROBE_SESSIONS:
                assert f"score_{arm}_epoch_{epoch_index:02d}_{session_id}.json" in names
                assert f"score_{arm}_epoch_{epoch_index:02d}_{session_id}.json.sha256" in names
    assert "20121004" in "".join(score_bodies)
    assert "20120924" in "".join(score_bodies)
    assert "opened_sessions.json" in names
    assert "opened_sessions.json.sha256" in names


def test_run_stage_lifecycle_is_attempt_first_and_failure_honest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep").mkdir(parents=True)
    events: list[str] = []
    from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1

    original = v1.ImmutableArtifactRoot.publish_json

    def observed(self: object, name: str, payload: Mapping[str, object]) -> str:
        events.append(name)
        return original(self, name, payload)

    monkeypatch.setattr(v1.ImmutableArtifactRoot, "publish_json", observed)
    closure = plan.implementation_closure(ROOT)
    attempt = receipts.stage_attempt_payload("smoke", plan.PairSpec50().sha256, closure)

    def bodies(artifact: v1.ImmutableArtifactRoot) -> dict[str, str]:
        events.append("bodies")
        return {
            "t0_smoke.json": artifact.publish_json("t0_smoke.json", {"ok": True}),
            "c1_smoke.json": artifact.publish_json("c1_smoke.json", {"ok": True}),
            "equality.json": artifact.publish_json("equality.json", {"ok": True}),
        }

    shas, terminal_sha, failure_sha = receipts.run_stage(
        tmp_path, relative="tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/smoke",
        attempt_payload=attempt, launch_builder=lambda: {"device": "cpu"},
        body_publisher=bodies, terminal_builder=lambda s: {"status": "COMPLETE"},
        expected_terminal_names=receipts.smoke_stage_names, progress=lambda: {"p": 1},
    )
    assert terminal_sha is not None and failure_sha is None
    assert events[:3] == ["attempt.json", "launch.json", "bodies"]
    assert events[-1] == "terminal.json"

    def failing_bodies(_artifact: v1.ImmutableArtifactRoot) -> dict[str, str]:
        raise RuntimeError("synthetic stage failure")

    attempt2 = receipts.stage_attempt_payload("t0", plan.PairSpec50().sha256, closure)
    shas2, terminal2, failure2 = receipts.run_stage(
        tmp_path, relative="tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/t0",
        attempt_payload=attempt2, launch_builder=lambda: {"device": "cpu"},
        body_publisher=failing_bodies, terminal_builder=lambda s: {},
        expected_terminal_names=receipts.arm_stage_names, progress=lambda: {"p": 2},
    )
    assert failure2 is not None and terminal2 is None
    failure_root = tmp_path / "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/t0"
    assert failure_root.is_dir()
    failure = json.loads((failure_root / "failure.json").read_text())
    assert failure["error_class"] == "RuntimeError" and failure["progress"] == {"p": 2}
    assert failure["terminal_published"] is False
    with pytest.raises(receipts.ReceiptError):
        receipts.run_stage(
            tmp_path, relative="tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/t0",
            attempt_payload=attempt2, launch_builder=lambda: {}, body_publisher=bodies,
            terminal_builder=lambda s: {}, expected_terminal_names=receipts.arm_stage_names,
        )


# ---------------------------------------------------------------------------
# Probe verdict and selection
# ---------------------------------------------------------------------------


def test_overfitting_verdict_synthetic_curves() -> None:
    monotone = {9: 0.1, 19: 0.2, 29: 0.3, 39: 0.4, 49: 0.5}
    absent = probe.overfitting_verdict(monotone, surface="val_heldout")
    assert absent["verdict"] == "OVERFITTING_ABSENT_WITHIN_50EP"
    assert absent["argmax_epoch_index"] == 49
    assert absent["descriptive_not_inferential"] is True
    assert absent["surface"] == "val_heldout"

    interior = {9: 0.1, 19: 0.4, 29: 0.3, 39: 0.2, 49: 0.2}
    present = probe.overfitting_verdict(interior, surface="val_heldout")
    assert present["verdict"] == "OVERFITTING_PRESENT"
    assert present["argmax_epoch_index"] == 19
    assert present["drop_from_argmax_to_epoch50"] == pytest.approx(0.2)

    # drop == 0.005 exactly (0.005 - 0.0) must be INCONCLUSIVE (strict >).
    at_threshold = {9: 0.0, 19: 0.005, 29: 0.0, 39: 0.0, 49: 0.0}
    inconclusive = probe.overfitting_verdict(at_threshold, surface="val_heldout")
    assert inconclusive["drop_from_argmax_to_epoch50"] == 0.005
    assert inconclusive["verdict"] == "OVERFITTING_INCONCLUSIVE"

    just_above = {9: 0.0, 19: math.nextafter(0.005, 1.0), 29: 0.0, 39: 0.0, 49: 0.0}
    present_above = probe.overfitting_verdict(just_above, surface="val_heldout")
    assert present_above["verdict"] == "OVERFITTING_PRESENT"

    just_below = {9: 0.0, 19: math.nextafter(0.005, 0.0), 29: 0.0, 39: 0.0, 49: 0.0}
    below = probe.overfitting_verdict(just_below, surface="val_heldout")
    assert below["verdict"] == "OVERFITTING_INCONCLUSIVE"

    fold_absent = probe.overfitting_verdict(monotone, surface="test_fold")
    assert fold_absent["surface"] == "test_fold"
    assert fold_absent["descriptive_not_inferential"] is True
    with pytest.raises(probe.ProbeError):
        probe.overfitting_verdict(monotone, surface="train_fit")


def _three_session_spread(mean: float, delta: float) -> dict[str, float]:
    return {
        "20121004": mean - delta,
        "20121017": mean,
        "20121024": mean + delta,
    }


def test_overfitting_verdict_drop_within_session_spread() -> None:
    # drop = 0.01 > 0.005 → PRESENT; large across-session SD at argmax → within spread.
    means_within = {9: 0.10, 19: 0.40, 29: 0.30, 39: 0.20, 49: 0.39}
    per_within = {epoch: _three_session_spread(value, 0.20) for epoch, value in means_within.items()}
    within = probe.overfitting_verdict(
        means_within, surface="val_heldout", per_session_by_epoch=per_within,
    )
    assert within["verdict"] == "OVERFITTING_PRESENT"
    assert within["drop_within_session_spread"] is True
    assert within["descriptive_not_inferential"] is True
    assert within["spread_reference_epoch_index"] == 19
    assert set(within["per_session_by_epoch"]["19"]) == {"20121004", "20121017", "20121024"}

    # drop = 0.20 > 0.005 → PRESENT; tiny SD at argmax → not within spread.
    means_outside = {9: 0.10, 19: 0.40, 29: 0.30, 39: 0.20, 49: 0.20}
    per_outside = {epoch: _three_session_spread(value, 0.001) for epoch, value in means_outside.items()}
    outside = probe.overfitting_verdict(
        means_outside, surface="val_heldout", per_session_by_epoch=per_outside,
    )
    assert outside["verdict"] == "OVERFITTING_PRESENT"
    assert outside["drop_within_session_spread"] is False


def test_select_epoch_argmax_tiebreak_and_fail_closed() -> None:
    curve = {9: 0.1, 19: 0.5, 29: 0.3, 39: 0.2, 49: 0.4}
    sessions = plan.VAL_HELDOUT_SESSIONS
    assert probe.select_epoch(curve, surface="val_heldout", sessions=sessions) == 19
    assert probe.select_epoch(
        {9: 0.5, 19: 0.5, 29: 0.1, 39: 0.1, 49: 0.1},
        surface="val_heldout", sessions=sessions,
    ) == 9
    with pytest.raises(TypeError):
        probe.select_epoch(curve)  # surface is required
    with pytest.raises(TypeError):
        probe.select_epoch(curve, surface="val_heldout")  # sessions is mandatory
    with pytest.raises(probe.ProbeError, match="sessions is mandatory"):
        probe.select_epoch(curve, surface="val_heldout", sessions=None)  # type: ignore[arg-type]
    with pytest.raises(probe.ProbeError):
        probe.select_epoch(curve, surface="train_fit", sessions=sessions)
    with pytest.raises(probe.ProbeError):
        probe.select_epoch(curve, surface="test_fold", sessions=sessions)
    with pytest.raises(probe.ProbeError):
        probe.select_epoch(curve, surface="heldin_training", sessions=sessions)
    with pytest.raises(probe.ProbeError):
        probe.select_epoch({9: 0.1, 19: 0.2, 29: 0.3, 39: 0.4},
                           surface="val_heldout", sessions=sessions)
    with pytest.raises(probe.ProbeError):
        probe.select_epoch({10: 0.1, 20: 0.2, 30: 0.3, 40: 0.4, 50: 0.5},
                           surface="val_heldout", sessions=sessions)
    with pytest.raises(probe.ProbeError):
        probe.select_epoch(
            curve, surface="val_heldout",
            sessions=("20121004", "20121017", "20120924"),
        )


def test_observed_val_heldout_digest_is_persisted() -> None:
    frozen = dict(plan.VAL_HELDOUT_BODY_SHA256)
    for session_id, digest in frozen.items():
        pair = probe.val_heldout_digest_receipt(session_id, digest)
        assert pair["body_sha256"] == digest
        assert pair["observed_body_sha256"] == digest
        assert "observed_body_sha256" in pair
    opened = {
        session_id: {"body_sha256": digest, "observed_body_sha256": digest}
        for session_id, digest in frozen.items()
    }
    table = probe.val_heldout_opened_digest_table(opened)
    assert set(table) == set(plan.VAL_HELDOUT_SESSIONS)
    for session_id, digest in frozen.items():
        assert table[session_id]["body_sha256"] == digest
        assert table[session_id]["observed_body_sha256"] == digest
    with pytest.raises(probe.ProbeError):
        probe.val_heldout_digest_receipt("20121004", "0" * 64)
    with pytest.raises(probe.ProbeError):
        probe.val_heldout_opened_digest_table({
            **opened,
            "20121004": {"body_sha256": frozen["20121004"], "observed_body_sha256": "0" * 64},
        })


def test_val_heldout_access_law_fail_closed(tmp_path: Path) -> None:
    legal = plan.val_heldout_relative_path("20121004")
    assert plan.assert_evaluation_only_path(legal) == legal
    with pytest.raises(plan.M1T0C150EpPlanError, match="forbidden token"):
        plan.assert_evaluation_only_path(
            "SPINT-main/data/000941/sub-MonkeyL-held-out-calib/"
            "sub-MonkeyL-held-out-minival_ses-20121004_behavior+ecephys.nwb",
        )
    with pytest.raises(plan.M1T0C150EpPlanError, match="forbidden token"):
        plan.assert_evaluation_only_path(
            "SPINT-main/data/000941/sub-MonkeyL-held-out-calib/"
            "sub-MonkeyL-test_ses-20121004_behavior+ecephys.nwb",
        )
    dummy = tmp_path / "dummy.nwb"
    dummy.write_bytes(b"not-the-sealed-val-heldout-body")
    with pytest.raises(plan.M1T0C150EpPlanError, match="digest mismatch"):
        plan.assert_val_heldout_body_digest("20121004", dummy.read_bytes())
    with pytest.raises(plan.M1T0C150EpPlanError, match="20120924 reached the selection input"):
        plan.assert_selection_sessions(("20121004", "20121017", "20120924"))
    assert plan.assert_selection_sessions(plan.VAL_HELDOUT_SESSIONS) == plan.VAL_HELDOUT_SESSIONS
    law = plan.VAL_HELDOUT_ACCESS_LAW
    assert law["training_use"] is False
    assert law["gradient_updates"] == 0
    assert law["optimizer_steps"] == 0
    assert law["labels_used_for"] == "metric only"
    assert law["schema"] == "m1_t0c1_50ep_val_heldout_access_law_v1"


def test_runtime_abort_heuristic_boundary() -> None:
    under = 1000.0 + 49.0 * 795.0
    over = 1000.0 + 49.0 * 796.0
    assert under < 40000 < over
    assert plan.assert_runtime_abort_heuristic(1000.0, 795.0) == pytest.approx(under)
    with pytest.raises(plan.M1T0C150EpPlanError, match="exceeds 40000"):
        plan.assert_runtime_abort_heuristic(1000.0, 796.0)
    # Exact 40000 does not exceed the bound.
    assert plan.assert_runtime_abort_heuristic(40000.0, 0.0) == 40000.0
    import inspect
    abort_src = inspect.getsource(plan.assert_runtime_abort_heuristic)
    estimate_src = inspect.getsource(plan.runtime_abort_estimate_seconds)
    assert "estimate <= RUNTIME_ABORT_ESTIMATE_SECONDS" in abort_src
    assert "49.0" in estimate_src
    trainer_src = (
        ROOT / "tfpd_exploration/src/m1_t0c1_prefix_v1_50ep/trainer.py"
    ).read_text(encoding="utf-8")
    assert "epoch_1_seconds = float(elapsed) - epoch_0_seconds" in trainer_src
    assert "plan.assert_runtime_abort_heuristic(epoch_0_seconds, epoch_1_seconds)" in trainer_src
    assert plan.RUNTIME_ABORT_ESTIMATE_SECONDS == 40_000


def test_probe_surface_constants() -> None:
    assert plan.VAL_HELDOUT_SESSIONS == ("20121004", "20121017", "20121024")
    assert plan.VAL_HELDOUT_BODY_SHA256 == {
        "20121004": "782c1fd090facfb3c50b6a85da8209ca3aea71f66bd4edc13d61fe4a55a3429d",
        "20121017": "8dd22c67500445ec1e0c11980475badbba9e960a05db16b78aaaad5502b3c652",
        "20121024": "bbeb6c7d66e2c2e9b76506021a6cf8800bc19021b6d1fd03c4eb6de71490bafb",
    }
    assert "20120924" not in plan.VAL_HELDOUT_SESSIONS
    assert plan.SELECTION_SURFACE == "val_heldout"
    assert plan.TEST_FOLD_SESSIONS == ("20120924",)
    assert plan.PROBE_SESSIONS == (
        "20120924", "20120926", "20120927", "20120928",
        "20121004", "20121017", "20121024",
    )
    assert len(plan.PROBE_SESSIONS) == 7


def test_spint_resource_parity_block() -> None:
    released = plan.SPINT_RESOURCE_PARITY["spint_m1_released"]
    assert released["max_epochs"] == 50
    assert released["lr"] == 1e-4
    deficit = plan.SPINT_RESOURCE_PARITY["disclosed_deficits"]["total_optimizer_steps"]
    assert deficit["this_lane"] == 247550
    assert deficit["spint_approx"] == 333350
    assert plan.SPINT_RESOURCE_PARITY["this_lane"]["epochs"] == 50
    assert plan.SPINT_RESOURCE_PARITY["this_lane"]["total_optimizer_steps"] == 247550


# ---------------------------------------------------------------------------
# Phase 3 reference delta
# ---------------------------------------------------------------------------


def test_reference_delta_against_spec_5_1_and_cross_recipe_flag() -> None:
    cells: dict[str, dict[str, object]] = {}
    for arm in plan.ARMS:
        for deployment in plan.PHASE3_DEPLOYMENTS:
            for session_id in plan.SCORE_ORDER:
                cells[f"{arm}_{deployment}_{session_id}"] = {"governing_r2": 0.5}
    table = phase3.build_table(cells)
    payload = phase3.reference_delta(table)
    assert payload["cross_recipe"] is True
    assert len(payload["deltas"]) == 8
    by_key = {(item["arm"], item["deployment"], item["surface"]): item
              for item in payload["deltas"]}
    for arm in plan.ARMS:
        for deployment in plan.PHASE3_DEPLOYMENTS:
            for surface in plan.PHASE3_SURFACES:
                item = by_key[(arm, deployment, surface)]
                reference = plan.REFERENCE_20EP_EQUAL_SESSION_MEAN[arm][deployment][surface]
                assert item["cross_recipe"] is True
                assert item["peak_lr"] == plan.LR_WARMUP_END
                assert item["schedule"] == plan.LR_SCHEDULE_KIND
                assert item["this_lane_equal_session_mean"] == pytest.approx(0.5)
                assert item["reference_20ep_equal_session_mean"] == reference
                assert item["delta_this_minus_20ep"] == pytest.approx(0.5 - reference)
                assert "constant lr=1e-5" in item["note"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_dry_run_and_execute_rejected() -> None:
    process = subprocess.run(
        [sys.executable, str(CLI), "--dry-run"], text=True, capture_output=True, check=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "CUDA_VISIBLE_DEVICES": ""},
    )
    payload = json.loads(process.stdout)
    assert payload["public_execution_authorized"] is False
    assert payload["budget"]["epochs"] == 50
    assert payload["budget"]["total_optimizer_steps"] == 247550
    assert payload["hard_timeout_seconds_per_arm"] == 43200
    assert payload["gpu"] == "CUDA_VISIBLE_DEVICES=1"
    assert payload["imports_torch"] is False
    rejected = subprocess.run(
        [sys.executable, str(CLI), "--execute"], text=True, capture_output=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    assert rejected.returncode != 0


def test_smoke_equality_validator_enforces_lr_channel() -> None:
    from tfpd_exploration.src.m1_t0c1_prefix_v1_50ep import driver

    expected_lr = [lr_schedule.lr_at(index) for index in range(plan.SMOKE_STEPS)]
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
        "lr_per_step": list(expected_lr),
        "lr_stream_digest": lr_schedule.sequence_digest(expected_lr),
    }
    c1 = json.loads(_json(base))
    c1["operator_snapshot"]["effective_prefixes"] = [10, 5]
    c1["operator_snapshot"]["recorded_prefix_sequence"] = [10, 5]
    c1["lr_per_step"] = list(expected_lr)
    c1["lr_stream_digest"] = lr_schedule.sequence_digest(expected_lr)
    equality = driver._validate_smoke_equality(base, c1)
    assert equality["lr_sequence_identical"] is True
    drifted = json.loads(_json(c1))
    drifted["lr_per_step"] = list(expected_lr)
    drifted["lr_per_step"][0] = 0.0
    with pytest.raises(driver.DriverError):
        driver._validate_smoke_equality(base, drifted)


def test_probe_and_phase3_bind_gpu1_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect

    from tfpd_exploration.src.m1_t0c1_prefix_v1_50ep import driver

    assert "require_gpu1_profile" in inspect.getsource(driver.execute_probe)
    assert "require_gpu1_profile" in inspect.getsource(driver.execute_phase3)
    assert "live_device_profile" in inspect.getsource(driver.require_gpu1_profile)

    class _Gpu1:
        def payload(self) -> dict[str, object]:
            return {
                "uuid": plan.DEVICE_IDENTITY_LAW["required_uuid"],
                "pci_bus_id": plan.DEVICE_IDENTITY_LAW["required_pci_bus_id"],
                "name": "NVIDIA GeForce RTX 3090",
            }

    monkeypatch.setattr(driver.trainer, "live_device_profile", lambda torch_module: _Gpu1())
    payload = driver.require_gpu1_profile()
    assert payload["uuid"] == plan.DEVICE_IDENTITY_LAW["required_uuid"]
    assert payload["pci_bus_id"] == plan.DEVICE_IDENTITY_LAW["required_pci_bus_id"]

    class _Gpu0:
        def payload(self) -> dict[str, object]:
            return {
                "uuid": plan.DEVICE_IDENTITY_LAW["refused_uuid_gpu0"],
                "pci_bus_id": "00000000:01:00.0",
            }

    monkeypatch.setattr(driver.trainer, "live_device_profile", lambda torch_module: _Gpu0())
    with pytest.raises(driver.DriverError):
        driver.require_gpu1_profile()


# ---------------------------------------------------------------------------
# GPU-1 arm concurrency allowlist (CPU-only: inject nvidia-smi and /proc)
# ---------------------------------------------------------------------------


def _load_launcher():
    import importlib.util

    path = ROOT / "tfpd_exploration/scripts/launch_m1_t0c1_prefix_v1_50ep.py"
    spec = importlib.util.spec_from_file_location(
        "m1_t0c1_prefix_v1_50ep_launcher", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _launcher_argv(stage: str) -> list[str]:
    launcher = (
        ROOT / "tfpd_exploration/scripts/launch_m1_t0c1_prefix_v1_50ep.py"
    ).resolve()
    return [sys.executable, str(launcher), "--stage", stage, "--i-am-root-reviewer"]


def test_gpu1_concurrency_allowlist_and_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    launch = _load_launcher()
    launcher_path = launch.LAUNCHER_PATH
    assert launcher_path == (
        ROOT / "tfpd_exploration/scripts/launch_m1_t0c1_prefix_v1_50ep.py"
    ).resolve()

    def boom_subprocess(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("concurrency tests must not shell out")

    monkeypatch.setattr(subprocess, "run", boom_subprocess)

    def cmdlines(mapping: dict[int, list[str]]):
        def reader(pid: int) -> list[str]:
            if pid not in mapping:
                raise FileNotFoundError(f"/proc/{pid}/cmdline")
            return mapping[pid]
        return reader

    admitted = launch.evaluate_gpu1_concurrency(
        "t0",
        compute_apps=[(4242, 1201)],
        used_memory_mib=1300,
        cmdline_of=cmdlines({4242: _launcher_argv("c1")}),
        launcher=launcher_path,
    )
    assert admitted["concurrency_mode"] == "sibling_arm_allowed"
    assert admitted["sibling_arm_compute_apps"] == [
        {"pid": 4242, "stage": "c1", "used_memory_mib": 1201},
    ]

    equals_form = launch.evaluate_gpu1_concurrency(
        "c1",
        compute_apps=[(7, 1201)],
        used_memory_mib=1300,
        cmdline_of=cmdlines({
            7: [sys.executable, str(launcher_path), "--stage=t0", "--i-am-root-reviewer"],
        }),
        launcher=launcher_path,
    )
    assert equals_form["sibling_arm_compute_apps"][0]["stage"] == "t0"

    alone = launch.evaluate_gpu1_concurrency(
        "t0", compute_apps=[], used_memory_mib=12,
        cmdline_of=cmdlines({}), launcher=launcher_path,
    )
    assert alone["concurrency_mode"] == "sibling_arm_allowed"
    assert alone["sibling_arm_compute_apps"] == []

    at_limit = launch.evaluate_gpu1_concurrency(
        "t0", compute_apps=[], used_memory_mib=8192,
        cmdline_of=cmdlines({}), launcher=launcher_path,
    )
    assert at_limit["concurrency_mode"] == "sibling_arm_allowed"

    with pytest.raises(launch.PreflightError, match="foreign compute process"):
        launch.evaluate_gpu1_concurrency(
            "t0",
            compute_apps=[(99, 100)],
            used_memory_mib=200,
            cmdline_of=cmdlines({99: [sys.executable, "/tmp/other_train.py", "--stage", "c1"]}),
            launcher=launcher_path,
        )

    with pytest.raises(launch.PreflightError, match="refusing double-launch"):
        launch.evaluate_gpu1_concurrency(
            "t0",
            compute_apps=[(11, 1201)],
            used_memory_mib=1300,
            cmdline_of=cmdlines({11: _launcher_argv("t0")}),
            launcher=launcher_path,
        )

    with pytest.raises(launch.PreflightError, match="cannot be resolved to a cmdline"):
        launch.evaluate_gpu1_concurrency(
            "t0",
            compute_apps=[(404, 100)],
            used_memory_mib=200,
            cmdline_of=cmdlines({}),
            launcher=launcher_path,
        )

    with pytest.raises(launch.PreflightError, match="empty cmdline"):
        launch.evaluate_gpu1_concurrency(
            "c1",
            compute_apps=[(5, 100)],
            used_memory_mib=200,
            cmdline_of=lambda pid: [],
            launcher=launcher_path,
        )

    with pytest.raises(launch.PreflightError, match="exceeds 8192 MiB"):
        launch.evaluate_gpu1_concurrency(
            "t0", compute_apps=[], used_memory_mib=8193,
            cmdline_of=cmdlines({}), launcher=launcher_path,
        )

    for exclusive in ("smoke", "probe", "phase3"):
        empty = launch.evaluate_gpu1_concurrency(
            exclusive, compute_apps=[], used_memory_mib=12,
            cmdline_of=cmdlines({}), launcher=launcher_path,
        )
        assert empty["concurrency_mode"] == "exclusive_card"
        assert empty["sibling_arm_compute_apps"] == []
        with pytest.raises(launch.PreflightError, match="requires an exclusive card"):
            launch.evaluate_gpu1_concurrency(
                exclusive,
                compute_apps=[(4242, 1201)],
                used_memory_mib=1300,
                cmdline_of=cmdlines({4242: _launcher_argv("t0")}),
                launcher=launcher_path,
            )

    with pytest.raises(launch.PreflightError, match="more than one sibling"):
        launch.evaluate_gpu1_concurrency(
            "t0",
            compute_apps=[(1, 100), (2, 100)],
            used_memory_mib=400,
            cmdline_of=cmdlines({1: _launcher_argv("c1"), 2: _launcher_argv("c1")}),
            launcher=launcher_path,
        )


def test_arm_concurrency_is_bound_on_t0_c1_receipts() -> None:
    closure = plan.implementation_closure(ROOT)
    spec_sha = plan.PairSpec50().sha256
    t0 = receipts.stage_attempt_payload("t0", spec_sha, closure, concurrent_sibling_stage="c1")
    assert t0["arm_concurrency"] == "concurrent_pair_on_gpu1"
    assert t0["concurrent_sibling_stage"] == "c1"
    assert t0["arm_concurrency_timing_disclosure"] == plan.ARM_CONCURRENCY_TIMING_DISCLOSURE
    assert "NOT comparable to the sealed 20-epoch reference timings" in t0[
        "arm_concurrency_timing_disclosure"
    ]
    alone = receipts.stage_attempt_payload("c1", spec_sha, closure)
    assert alone["arm_concurrency"] == "concurrent_pair_on_gpu1"
    assert alone["concurrent_sibling_stage"] is None
    smoke = receipts.stage_attempt_payload("smoke", spec_sha, closure)
    assert "arm_concurrency" not in smoke
    with pytest.raises(receipts.ReceiptError50, match="only t0/c1"):
        receipts.stage_attempt_payload(
            "smoke", spec_sha, closure, concurrent_sibling_stage="t0",
        )
    with pytest.raises(receipts.ReceiptError50, match="must be 'c1'"):
        receipts.arm_concurrency_receipt("t0", "t0")
    from tfpd_exploration.src.m1_t0c1_prefix_v1_50ep import driver
    launch = driver._launch_payload("t0", "cuda:0", concurrent_sibling_stage="c1")
    assert launch["arm_concurrency"] == "concurrent_pair_on_gpu1"
    assert launch["concurrent_sibling_stage"] == "c1"
    probe_launch = driver._launch_payload("probe", "cuda:0")
    assert "arm_concurrency" not in probe_launch
    import inspect
    arm_src = inspect.getsource(driver.execute_arm)
    assert "concurrent_sibling_stage" in arm_src
    assert "arm_concurrency_receipt" in arm_src


def test_check_gpu1_preflight_keys_are_injected_not_shelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = _load_launcher()

    def fake_smi(*args: str) -> str:
        assert "--id" in args and args[args.index("--id") + 1] == "1"
        joined = " ".join(args)
        if "query-gpu=" in joined:
            return (
                f"{launch.GPU1_UUID}, 00000000:03:00.0, "
                f"{launch.EXPECTED_DEVICE_NAME}, 12 MiB\n"
            )
        if "query-compute-apps=" in joined:
            return ""
        raise AssertionError(f"unexpected nvidia-smi args: {args}")

    monkeypatch.setattr(launch, "_nvidia_smi", fake_smi)
    monkeypatch.setattr(
        launch, "read_proc_cmdline",
        lambda pid: (_ for _ in ()).throw(AssertionError("no /proc in this test")),
    )
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("subprocess.run must not be called")),
    )
    payload = launch.check_gpu1("t0")
    assert payload["uuid"] == launch.GPU1_UUID
    assert payload["uuid"] != launch.GPU0_UUID_REFUSED
    assert payload["pci_bus_id"] == "00000000:03:00.0"
    assert payload["name"] == launch.EXPECTED_DEVICE_NAME
    assert payload["foreign_compute_apps"] == []
    assert payload["concurrency_mode"] == "sibling_arm_allowed"
    assert payload["sibling_arm_compute_apps"] == []
    exclusive = launch.check_gpu1("smoke")
    assert exclusive["concurrency_mode"] == "exclusive_card"

    def occupied(*args: str) -> str:
        assert args[args.index("--id") + 1] == "1"
        joined = " ".join(args)
        if "query-gpu=" in joined:
            return (
                f"{launch.GPU1_UUID}, 00000000:03:00.0, "
                f"{launch.EXPECTED_DEVICE_NAME}, 1300 MiB\n"
            )
        return "4242, 1201 MiB\n"

    monkeypatch.setattr(launch, "_nvidia_smi", occupied)
    monkeypatch.setattr(launch, "read_proc_cmdline", lambda pid: _launcher_argv("c1"))
    sibling = launch.check_gpu1("t0")
    assert sibling["concurrency_mode"] == "sibling_arm_allowed"
    assert sibling["sibling_arm_compute_apps"] == [
        {"pid": 4242, "stage": "c1", "used_memory_mib": 1201},
    ]
    with pytest.raises(launch.PreflightError, match="requires an exclusive card"):
        launch.check_gpu1("probe")

    def gpu0(*args: str) -> str:
        assert args[args.index("--id") + 1] == "1"
        return (
            f"{launch.GPU0_UUID_REFUSED}, 00000000:01:00.0, "
            f"{launch.EXPECTED_DEVICE_NAME}, 12 MiB\n"
        )

    monkeypatch.setattr(launch, "_nvidia_smi", gpu0)
    with pytest.raises(launch.PreflightError, match="uuid drift|physical GPU 0"):
        launch.check_gpu1("t0")

