"""Synthetic/no-data/no-CUDA tests for the M1 held-in vs held-out fold gap V1."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import pytest

from tfpd_exploration.src.m1_heldin_heldout_gap_v1 import plan, score
from tfpd_exploration.src.cross_session_worst_group_fold20120924_score_v1 import plan as cswg_plan


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "tfpd_exploration/scripts/run_m1_heldin_heldout_gap_v1.py"


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_pair(directory: Path, name: str, body: bytes) -> str:
    path = directory / name
    path.write_bytes(body)
    digest = _sha(body)
    (directory / f"{name}.sha256").write_bytes(f"{digest}  {name}\n".encode("ascii"))
    os.chmod(path, 0o444)
    os.chmod(directory / f"{name}.sha256", 0o444)
    return digest


def _replace_pair(directory: Path, name: str, body: bytes) -> str:
    os.chmod(directory / name, 0o644)
    os.chmod(directory / f"{name}.sha256", 0o644)
    return _write_pair(directory, name, body)


# ---------------------------------------------------------------------------
# Synthetic frozen producer (CS-WG, fold 20120924) 56-leaf graph
# ---------------------------------------------------------------------------


def _producer_expectation(tmp_path: Path):
    directory = tmp_path / plan.PRODUCER_ROOT_RELATIVE
    directory.mkdir(parents=True)
    checkpoint = b"synthetic-cswg-fold20120924-best-checkpoint"
    best_state = "d" * 64
    identity = {
        "schema": "cross_session_worst_group_m1_full_training_identity_v1",
        "spec": {
            "inherited_v1_full_spec": {
                "stage0_run_spec": {
                    "system": "CS_WG", "lambda": 1.0, "tau": 0.01,
                    "outer_target_session": plan.ANCHOR_SESSION,
                    "source_sessions": list(plan.HELDIN_TRAINING_SESSIONS),
                },
            },
            "swa_enabled": False,
            "swa_artifact_forbidden": True,
        },
    }
    identity_sha = _sha(_json(identity))
    body_sha: dict[str, str] = {}
    for name in ("attempt.json", "launch.json", "source_authority.json"):
        body_sha[name] = _write_pair(directory, name, _json({"name": name, "system": "CS_WG"}))
    for epoch in range(20):
        name = f"epoch_{epoch:02d}.json"
        body_sha[name] = _write_pair(directory, name, _json({"name": name, "epoch_index": epoch}))
    training = {
        "schema": "cross_session_worst_group_m1_full_training_result_v1",
        "identity_sha256": identity_sha,
        "swa_enabled": False, "swa_artifact_forbidden": True, "source_only": True,
    }
    body_sha["training.json"] = _write_pair(directory, "training.json", _json(training))
    body_sha["checkpoint_best_source_train_loss.pt"] = _write_pair(
        directory, "checkpoint_best_source_train_loss.pt", checkpoint,
    )
    body_sha["checkpoint_last.pt"] = _write_pair(directory, "checkpoint_last.pt", checkpoint)
    manifest = {
        "schema": "cross_session_worst_group_m1_full_training_checkpoint_manifest_v1",
        "checkpoints": {
            "best_source_train_loss": {
                "filename": "checkpoint_best_source_train_loss.pt",
                "sha256": body_sha["checkpoint_best_source_train_loss.pt"],
                "state_sha256": best_state, "strict_reload": True,
            },
            "last": {
                "filename": "checkpoint_last.pt", "sha256": body_sha["checkpoint_last.pt"],
                "state_sha256": best_state, "strict_reload": True,
            },
        },
        "best_epoch_index": 19, "last_epoch_index": 19,
        "swa_enabled": False, "swa_artifact_forbidden": True,
    }
    body_sha["checkpoint_manifest.json"] = _write_pair(directory, "checkpoint_manifest.json", _json(manifest))
    terminal = {
        "schema": "cross_session_worst_group_m1_full_training_terminal_v1",
        "status": "PASS_SOURCE_FULL_FIXED_20_EPOCH_NO_SWA",
        "identity": identity,
        "attempt_sha256": body_sha["attempt.json"],
        "launch_sha256": body_sha["launch.json"],
        "source_authority_sha256": body_sha["source_authority.json"],
        "training_sha256": body_sha["training.json"],
        "epoch_sha256": {f"epoch_{epoch:02d}.json": body_sha[f"epoch_{epoch:02d}.json"] for epoch in range(20)},
        "checkpoint_manifest_sha256": body_sha["checkpoint_manifest.json"],
        "checkpoint_best_source_train_loss_sha256": body_sha["checkpoint_best_source_train_loss.pt"],
        "checkpoint_last_sha256": body_sha["checkpoint_last.pt"],
        "checkpoint_best_source_train_loss_state_sha256": best_state,
        "checkpoint_last_state_sha256": best_state,
        "swa_enabled": False, "swa_artifact_forbidden": True,
        "source_only": True, "target_optimizer_backward_update": 0,
    }
    body_sha["terminal.json"] = _write_pair(directory, "terminal.json", _json(terminal))
    expectation = cswg_plan.CompletedFullExpectation(
        root_relative=plan.PRODUCER_ROOT_RELATIVE,
        terminal_sha256=body_sha["terminal.json"],
        training_sha256=body_sha["training.json"],
        identity_sha256=identity_sha,
        checkpoint_manifest_sha256=body_sha["checkpoint_manifest.json"],
        best_checkpoint_sha256=body_sha["checkpoint_best_source_train_loss.pt"],
        best_checkpoint_state_sha256=best_state,
        last_checkpoint_sha256=body_sha["checkpoint_last.pt"],
        last_checkpoint_state_sha256=best_state,
        best_epoch_index=19, last_epoch_index=19,
        expected_system="CS_WG",
        extra_body_sha256={
            "attempt.json": body_sha["attempt.json"],
            "launch.json": body_sha["launch.json"],
            "source_authority.json": body_sha["source_authority.json"],
        },
    )
    return expectation, score.load_producer_graph(tmp_path, expectation=expectation)


# ---------------------------------------------------------------------------
# Synthetic frozen anchor receipt graph
# ---------------------------------------------------------------------------


def _anchor_literals(*, input_authority_sha256: str, score_sha256: str, terminal_sha256: str,
                     attempt_sha256: str = "a" * 64, launch_sha256: str = "b" * 64,
                     ) -> plan.AnchorLiterals:
    return plan.AnchorLiterals(
        attempt_sha256=attempt_sha256,
        launch_sha256=launch_sha256,
        input_authority_sha256=input_authority_sha256,
        score_sha256=score_sha256,
        terminal_sha256=terminal_sha256,
        reader_recipe_sha256="6" * 64,
        ordered_window_start_sha256="3" * 64,
        ordered_query_identity_sha256="4" * 64,
        ordered_target_evalmask_sha256="5" * 64,
        calibration_sha256="2" * 64,
        target_descriptor_sha256="1" * 64,
    )


def _anchor_receipt(tmp_path: Path, literals: plan.AnchorLiterals | None = None):
    directory = tmp_path / plan.ANCHOR_ROOT_RELATIVE
    directory.mkdir(parents=True, exist_ok=True)
    provisional = literals or _anchor_literals(
        input_authority_sha256="c" * 64, score_sha256="d" * 64, terminal_sha256="e" * 64,
    )
    input_authority = {
        "schema": "cross_session_worst_group_m1_fold20120924_heldin_input_authority_v1",
        "n_windows": provisional.n_windows,
        "model_input_shape": [100, 64],
        "calibration_shape_per_row": [10, 1024, 64],
        "ordered_window_start_sha256": provisional.ordered_window_start_sha256,
        "target_sha256": provisional.target_sha256,
        "calibration_sha256": provisional.calibration_sha256,
        "target_descriptor": {
            "session_id": plan.ANCHOR_SESSION,
            "relative_path": "SPINT-main/data/000941/synthetic_ses-20120924.nwb",
            "sha256": provisional.target_descriptor_sha256,
            "byte_count": provisional.target_descriptor_byte_count,
            "role": "m1_heldin_source_nwb", "source_only": True,
            "forbidden_surfaces_absent": True,
        },
        "target_reader_native_evidence": {
            "session_id": plan.ANCHOR_SESSION,
            "calibration_session": plan.ANCHOR_SESSION,
            "calibration_sha256": provisional.calibration_sha256,
            "all_row_calibration_sha256_match_session": True,
            "no_cross_session_calibration_substitution": True,
            "ordered_query_identity_sha256": provisional.ordered_query_identity_sha256,
            "ordered_window_start_sha256": provisional.ordered_window_start_sha256,
            "ordered_target_evalmask_sha256": provisional.ordered_target_evalmask_sha256,
            "reader_recipe_sha256": provisional.reader_recipe_sha256,
        },
    }
    input_sha = _write_pair(directory, "input_authority.json", _json(input_authority))
    score_body = {
        "schema": "cross_session_worst_group_m1_fold20120924_heldin_score_payload_v1",
        "input_authority": input_authority,
        "target_session": plan.ANCHOR_SESSION,
        "n_windows": provisional.n_windows,
        "full_system_forward_count": provisional.forward_batches,
        "governing_r2": provisional.governing_r2,
        "prediction_sha256": provisional.prediction_sha256,
        "target_sha256": provisional.target_sha256,
    }
    score_sha = _write_pair(directory, "score.json", _json(score_body))
    attempt_sha = _write_pair(directory, "attempt.json", _json({"status": "ATTEMPT_RESERVED"}))
    launch_sha = _write_pair(directory, "launch.json", _json({"status": "LAUNCHED"}))
    terminal_sha = _write_pair(directory, "terminal.json", _json({
        "status": plan.ANCHOR_TERMINAL_STATUS, "input_authority_sha256": input_sha,
        "score_sha256": score_sha, "governing_r2": provisional.governing_r2,
    }))
    if literals is None:
        literals = _anchor_literals(
            input_authority_sha256=input_sha, score_sha256=score_sha, terminal_sha256=terminal_sha,
            attempt_sha256=attempt_sha, launch_sha256=launch_sha,
        )
    return score.load_anchor_evidence(tmp_path, anchor=literals), literals


def _prepared_fragment(anchor: score.AnchorEvidence, session_id: str) -> dict[str, object]:
    if session_id == plan.ANCHOR_SESSION:
        fragment = dict(anchor.same_input_fragment())
    else:
        fragment = {
            "n_windows": 50_000,
            "model_input_shape": [100, 64],
            "calibration_shape_per_row": [10, 1024, 64],
            "ordered_window_start_sha256": "7" * 64,
            "target_sha256": "8" * 64,
            "calibration_sha256": "9" * 64,
            "target_descriptor": {
                "session_id": session_id,
                "relative_path": f"SPINT-main/data/000941/synthetic_ses-{session_id}.nwb",
                "sha256": "0" * 64, "byte_count": 70_000_000,
                "role": "m1_heldin_source_nwb", "source_only": True,
                "forbidden_surfaces_absent": True,
            },
            "target_reader_native_evidence": {
                "session_id": session_id, "calibration_session": session_id,
                "calibration_sha256": "9" * 64,
                "all_row_calibration_sha256_match_session": True,
                "no_cross_session_calibration_substitution": True,
                "ordered_query_identity_sha256": "a" * 64,
                "ordered_window_start_sha256": "7" * 64,
                "ordered_target_evalmask_sha256": "b" * 64,
                "reader_recipe_sha256": "c" * 64,
            },
        }
    fragment["input_record_sha256"] = "d" * 64
    return fragment


class _FakeBackend:
    def __init__(self, events: list[str], anchor: score.AnchorEvidence, *, fail_at: str | None = None,
                 session_values: Mapping[str, float] | None = None) -> None:
        self.events = events
        self.anchor = anchor
        self.fail_at = fail_at
        self.source_root = Path("/synthetic/m1-heldin")
        self.closed = False
        self.values = dict(session_values or {})

    def launch_payload(self, identity, graph, anchor) -> Mapping[str, object]:
        self.events.append("backend:launch")
        if self.fail_at == "launch":
            raise RuntimeError("synthetic launch failure")
        return {"schema": "m1_heldin_heldout_gap_physical_launch_v1", "device": "cpu"}

    def prepare_session(self, identity, graph, session_id: str) -> Mapping[str, object]:
        self.events.append(f"backend:prepare:{session_id}")
        if self.fail_at == f"prepare:{session_id}":
            raise RuntimeError(f"synthetic prepare failure: {session_id}")
        return _prepared_fragment(self.anchor, session_id)

    def score_session(self, identity, graph, session_id: str, input_authority) -> Mapping[str, object]:
        self.events.append(f"backend:score:{session_id}")
        if self.fail_at == f"score:{session_id}":
            raise RuntimeError(f"synthetic score failure: {session_id}")
        value = self.values.get(session_id, 0.5)
        return score._session_score_payload(
            identity, graph, self.anchor, session_id, input_authority,
            forward_count=plan.ANCHOR_FORWARD_BATCHES if session_id == plan.ANCHOR_SESSION else 391,
            prediction_sha256=plan.ANCHOR_PREDICTION_SHA256 if session_id == plan.ANCHOR_SESSION else "e" * 64,
            target_sha256=str(input_authority["target_sha256"]),
            governing_r2=value,
            model_state_before_sha256=plan.ANCHOR_MODEL_STATE_SHA256,
            model_state_after_sha256=plan.ANCHOR_MODEL_STATE_SHA256,
        )

    def progress(self) -> score.GapProgress:
        return score.GapProgress(
            sessions_resolved_or_opened=tuple(plan.SCORE_ORDER[:2]),
            checkpoint_opened=True, model_constructed=True,
            full_system_forward_batches=820, input_authorities_published=2,
            session_scores_published=2,
        )

    def close(self) -> None:
        self.closed = True


def _identity(anchor_literals: plan.AnchorLiterals) -> plan.GapIdentity:
    return plan.GapIdentity(plan.GapSpec(), plan.implementation_closure(ROOT), anchor_literals)


def _lifecycle_root(tmp_path: Path) -> Path:
    (tmp_path / "tfpd_exploration/results").mkdir(parents=True, exist_ok=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_anchor_literals_are_the_frozen_receipt_values() -> None:
    payload = plan.DEFAULT_ANCHOR.payload()
    assert payload["input_authority_sha256"] == plan.ANCHOR_INPUT_AUTHORITY_SHA256
    assert payload["score_sha256"] == plan.ANCHOR_SCORE_SHA256
    assert payload["governing_r2"] == pytest.approx(0.5679166316986084)
    assert payload["n_windows"] == 54_849 and payload["forward_batches"] == 429
    with pytest.raises(plan.M1GapPlanError):
        plan.AnchorLiterals(input_authority_sha256="z" * 64)


def test_arm_roster_and_fold_bookkeeping_are_preregistered() -> None:
    assert plan.HELDOUT_FOLD_SESSIONS == ("20120924",)
    assert plan.HELDIN_TRAINING_SESSIONS == ("20120926", "20120927", "20120928")
    assert plan.SCORE_ORDER[0] == plan.ANCHOR_SESSION
    assert score.session_arm("20120924") == "heldout_fold"
    assert all(score.session_arm(s) == "heldin_training" for s in plan.HELDIN_TRAINING_SESSIONS)
    assert plan.FOLD_TARGET_OF_SESSION == {
        "20120924": "fold0", "20120926": "fold1", "20120927": "fold2", "20120928": None,
    }
    assert plan.VERDICT_RULE["bootstrap"]["seed"] == 42
    assert plan.VERDICT_RULE["bootstrap"]["draws"] == 10_000
    assert plan.OFFICIAL_HELDOUT_BLOCKED["silently_substituted"] is False


def test_producer_graph_binds_and_adversaries_fail_closed(tmp_path: Path) -> None:
    expectation, graph = _producer_expectation(tmp_path)
    assert len(graph.body_sha256) == 28
    assert graph.expectation.root_relative == plan.PRODUCER_ROOT_RELATIVE
    directory = tmp_path / plan.PRODUCER_ROOT_RELATIVE
    _replace_pair(directory, "source_authority.json", _json({"forged": True}))
    with pytest.raises(score.M1GapScoreError):
        score.load_producer_graph(tmp_path, expectation=expectation)


@pytest.mark.parametrize("mutation", ("rehash_native", "wrong_r2", "extra_leaf"))
def test_anchor_receipt_adversaries_fail_closed(tmp_path: Path, mutation: str) -> None:
    anchor, literals = _anchor_receipt(tmp_path)
    assert anchor.score["governing_r2"] == literals.governing_r2
    directory = tmp_path / plan.ANCHOR_ROOT_RELATIVE
    if mutation == "extra_leaf":
        (directory / "failure.json").write_text("{}")
        with pytest.raises(score.M1GapScoreError):
            score.load_anchor_evidence(tmp_path, anchor=literals)
        return
    inputs = json.loads((directory / "input_authority.json").read_text())
    if mutation == "rehash_native":
        native = dict(inputs["target_reader_native_evidence"])
        native["reader_recipe_sha256"] = "0" * 64
        inputs["target_reader_native_evidence"] = native
    else:
        inputs["n_windows"] = 54_850
    input_sha = _replace_pair(directory, "input_authority.json", _json(inputs))
    scored = json.loads((directory / "score.json").read_text())
    scored["input_authority"] = inputs
    if mutation == "wrong_r2":
        scored["governing_r2"] = 0.5
    score_sha = _replace_pair(directory, "score.json", _json(scored))
    terminal = json.loads((directory / "terminal.json").read_text())
    terminal["input_authority_sha256"] = input_sha
    terminal["score_sha256"] = score_sha
    _replace_pair(directory, "terminal.json", _json(terminal))
    rehashed = plan.AnchorLiterals(
        input_authority_sha256=input_sha, score_sha256=score_sha,
        terminal_sha256=literals.terminal_sha256,
    )
    with pytest.raises(score.M1GapScoreError):
        score.load_anchor_evidence(tmp_path, anchor=rehashed)


def test_session_input_authority_enforces_anchor_reproduction(tmp_path: Path) -> None:
    _expectation, graph = _producer_expectation(tmp_path)
    anchor, _literals = _anchor_receipt(tmp_path)
    identity = _identity(anchor.anchor)
    inputs = score._session_input_authority_payload(
        identity, graph, anchor, plan.ANCHOR_SESSION, _prepared_fragment(anchor, plan.ANCHOR_SESSION),
    )
    assert inputs["arm"] == "heldout_fold" and inputs["anchor_session_reproduction"] is True
    drifted = _prepared_fragment(anchor, plan.ANCHOR_SESSION)
    drifted["n_windows"] = 54_850
    with pytest.raises(score.M1GapScoreError, match="anchor session reproduction drift"):
        score._session_input_authority_payload(identity, graph, anchor, plan.ANCHOR_SESSION, drifted)
    foreign = _prepared_fragment(anchor, "20120926")
    foreign["target_descriptor"] = dict(foreign["target_descriptor"], session_id="20120927")
    with pytest.raises(score.M1GapScoreError):
        score._session_input_authority_payload(identity, graph, anchor, "20120926", foreign)


def test_session_score_enforces_exact_anchor_reproduction(tmp_path: Path) -> None:
    _expectation, graph = _producer_expectation(tmp_path)
    anchor, _literals = _anchor_receipt(tmp_path)
    identity = _identity(anchor.anchor)
    inputs = score._session_input_authority_payload(
        identity, graph, anchor, plan.ANCHOR_SESSION, _prepared_fragment(anchor, plan.ANCHOR_SESSION),
    )
    payload = score._session_score_payload(
        identity, graph, anchor, plan.ANCHOR_SESSION, inputs,
        forward_count=plan.ANCHOR_FORWARD_BATCHES,
        prediction_sha256=plan.ANCHOR_PREDICTION_SHA256,
        target_sha256=plan.ANCHOR_TARGET_SHA256,
        governing_r2=plan.ANCHOR_GOVERNING_R2,
        model_state_before_sha256=plan.ANCHOR_MODEL_STATE_SHA256,
        model_state_after_sha256=plan.ANCHOR_MODEL_STATE_SHA256,
    )
    assert payload["anchor_reproduced_exactly"] is True
    with pytest.raises(score.M1GapScoreError, match="anchor session exact reproduction drift"):
        score._session_score_payload(
            identity, graph, anchor, plan.ANCHOR_SESSION, inputs,
            forward_count=plan.ANCHOR_FORWARD_BATCHES,
            prediction_sha256=plan.ANCHOR_PREDICTION_SHA256,
            target_sha256=plan.ANCHOR_TARGET_SHA256,
            governing_r2=0.5679,
            model_state_before_sha256=plan.ANCHOR_MODEL_STATE_SHA256,
            model_state_after_sha256=plan.ANCHOR_MODEL_STATE_SHA256,
        )


def test_arm_summary_paired_bootstrap_and_verdict_branches() -> None:
    heldin = {"20120926": 0.60, "20120927": 0.62, "20120928": 0.58}
    heldout = {"20120924": 0.5679166316986084}
    summary = score.arm_summary(heldin, sessions=plan.HELDIN_TRAINING_SESSIONS)
    assert summary["n"] == 3
    assert summary["equal_session_mean"] == pytest.approx(0.6)
    assert summary["equal_session_sd"] == pytest.approx(0.016329931618554518)
    out = score.arm_summary(heldout, sessions=plan.HELDOUT_FOLD_SESSIONS)
    assert out["equal_session_sd"] == 0.0 and out["n"] == 1
    bootstrap = score.paired_bootstrap(heldin, heldout)
    assert bootstrap["gap_point"] == pytest.approx(0.5679166316986084 - 0.6)
    assert bootstrap["seed"] == 42 and bootstrap["draws"] == 10_000
    assert bootstrap["heldout_arm_degenerate"] is True
    repeat = score.paired_bootstrap(heldin, heldout)
    assert repeat["gap_ci95"] == bootstrap["gap_ci95"]
    with pytest.raises(score.M1GapScoreError):
        score.paired_bootstrap(heldin, heldout, seed=43)
    tight = score.apply_preregistered_verdict(gap=0.01, ci_low=-0.02, ci_high=0.02)
    assert tight["verdict"] == "NO_HEADROOM" and tight["direction"] is None
    worse = score.apply_preregistered_verdict(gap=-0.05, ci_low=-0.08, ci_high=-0.02)
    assert worse["verdict"] == "HEADROOM_PRESENT" and worse["direction"] == "HELD_OUT_WORSE"
    better = score.apply_preregistered_verdict(gap=0.05, ci_low=0.02, ci_high=0.08)
    assert better["verdict"] == "HEADROOM_PRESENT" and better["direction"] == "HELD_OUT_BETTER"
    wide = score.apply_preregistered_verdict(gap=0.01, ci_low=-0.05, ci_high=0.05)
    assert wide["verdict"] == "INDETERMINATE"


def test_paired_table_binds_all_sessions_and_pre_registered_verdict(tmp_path: Path) -> None:
    _expectation, graph = _producer_expectation(tmp_path)
    anchor, _literals = _anchor_receipt(tmp_path)
    identity = _identity(anchor.anchor)
    values = {"20120924": plan.ANCHOR_GOVERNING_R2, "20120926": 0.60,
              "20120927": 0.62, "20120928": 0.58}
    session_scores = {}
    for session_id in plan.SCORE_ORDER:
        inputs = score._session_input_authority_payload(
            identity, graph, anchor, session_id, _prepared_fragment(anchor, session_id),
        )
        session_scores[session_id] = score._session_score_payload(
            identity, graph, anchor, session_id, inputs,
            forward_count=plan.ANCHOR_FORWARD_BATCHES if session_id == plan.ANCHOR_SESSION else 391,
            prediction_sha256=plan.ANCHOR_PREDICTION_SHA256 if session_id == plan.ANCHOR_SESSION else "e" * 64,
            target_sha256=str(inputs["target_sha256"]),
            governing_r2=values[session_id],
            model_state_before_sha256=plan.ANCHOR_MODEL_STATE_SHA256,
            model_state_after_sha256=plan.ANCHOR_MODEL_STATE_SHA256,
        )
        session_scores[session_id]["input_authority"] = inputs
    table = score.build_paired_table(identity, graph, anchor, session_scores)
    assert [row["session_id"] for row in table["sessions"]] == list(plan.SCORE_ORDER)
    assert table["heldin_training_arm"]["equal_session_mean"] == pytest.approx(0.6)
    assert table["heldout_fold_arm"]["equal_session_mean"] == pytest.approx(plan.ANCHOR_GOVERNING_R2)
    assert table["gap"]["value"] == pytest.approx(plan.ANCHOR_GOVERNING_R2 - 0.6)
    assert table["verdict"]["verdict"] in {"NO_HEADROOM", "HEADROOM_PRESENT", "INDETERMINATE"}
    assert table["verdict"]["decided_before_any_scoring"] is True
    assert table["official_heldout_surface_blocked"]["silently_substituted"] is False
    del session_scores["20120928"]
    with pytest.raises(score.M1GapScoreError):
        score.build_paired_table(identity, graph, anchor, session_scores)


def test_lifecycle_attempt_precedes_backend_and_terminal_pins_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    expectation, graph = _producer_expectation(tmp_path)
    anchor, _literals = _anchor_receipt(tmp_path)
    identity = _identity(anchor.anchor)
    root = _lifecycle_root(tmp_path)
    capability = score.GapCapability(
        identity.sha256, graph.sha256, score._sha(score._json_bytes(anchor.payload())),
        "/synthetic/m1-heldin", score._GAP_REVIEW_SEAL,
    )
    events: list[str] = []
    values = {"20120924": plan.ANCHOR_GOVERNING_R2, "20120926": 0.60,
              "20120927": 0.62, "20120928": 0.58}
    backend = _FakeBackend(events, anchor, session_values=values)
    from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1

    original = v1.ImmutableArtifactRoot.publish_json

    def observed_publish(self: object, name: str, payload: Mapping[str, object]) -> str:
        events.append(f"publish:{name}")
        return original(self, name, payload)

    monkeypatch.setattr(score, "validate_identity_current", lambda _root, observed: observed == identity)
    monkeypatch.setattr(v1.ImmutableArtifactRoot, "publish_json", observed_publish)
    result = score.execute_reviewed_gap_score(
        root, identity=identity, capability=capability, backend=backend,
        graph_loader=lambda _root: graph, anchor_loader=lambda _root: anchor,
    )
    assert result.terminal_sha256 is not None and result.failure_sha256 is None
    assert events.index("publish:attempt.json") < events.index("backend:launch")
    assert events.index("publish:input_authority_20120924.json") < events.index("backend:score:20120924")
    assert events.index("publish:score_20120928.json") < events.index("publish:paired_table.json")
    assert events.index("publish:paired_table.json") < events.index("publish:terminal.json")
    terminal = json.loads((root / plan.RESULT_ROOT_RELATIVE / "terminal.json").read_text())
    assert terminal["status"] == "COMPLETE_DESCRIPTIVE_HELDIN_HELDOUT_GAP"
    assert terminal["anchor_reproduced_exactly"] is True
    assert terminal["target_optimizer_backward_update"] == 0
    assert set(terminal["session_score_sha256"]) == set(plan.SCORE_ORDER)
    table = json.loads((root / plan.RESULT_ROOT_RELATIVE / "paired_table.json").read_text())
    assert table["heldin_training_arm"]["equal_session_mean"] == pytest.approx(0.6)
    assert backend.closed is True
    names = sorted(os.listdir(root / plan.RESULT_ROOT_RELATIVE))
    assert names == sorted(score._success_names(terminal=True))


def test_failure_is_honest_with_progress_and_zero_target_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    expectation, graph = _producer_expectation(tmp_path)
    anchor, _literals = _anchor_receipt(tmp_path)
    identity = _identity(anchor.anchor)
    root = _lifecycle_root(tmp_path)
    capability = score.GapCapability(
        identity.sha256, graph.sha256, score._sha(score._json_bytes(anchor.payload())),
        "/synthetic/m1-heldin", score._GAP_REVIEW_SEAL,
    )
    monkeypatch.setattr(score, "validate_identity_current", lambda _root, observed: observed == identity)
    result = score.execute_reviewed_gap_score(
        root, identity=identity, capability=capability,
        backend=_FakeBackend([], anchor, fail_at="score:20120926"),
        graph_loader=lambda _root: graph, anchor_loader=lambda _root: anchor,
    )
    assert result.failure_sha256 is not None and result.terminal_sha256 is None
    failure = json.loads((root / plan.RESULT_ROOT_RELATIVE / "failure.json").read_text())
    assert failure["progress"]["target_optimizer_steps"] == 0
    assert failure["progress"]["sessions_resolved_or_opened"] == ["20120924", "20120926"]
    assert failure["terminal_published"] is False
    assert "paired_table_sha256" not in failure or failure["paired_table_sha256"] is None


def test_capability_or_source_root_substitution_fails_before_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    expectation, graph = _producer_expectation(tmp_path)
    anchor, _literals = _anchor_receipt(tmp_path)
    identity = _identity(anchor.anchor)
    root = _lifecycle_root(tmp_path)
    capability = score.GapCapability(
        identity.sha256, graph.sha256, score._sha(score._json_bytes(anchor.payload())),
        "/synthetic/canonical", score._GAP_REVIEW_SEAL,
    )
    monkeypatch.setattr(score, "validate_identity_current", lambda _root, observed: observed == identity)
    with pytest.raises(score.M1GapScoreError, match="source-root"):
        score.execute_reviewed_gap_score(
            root, identity=identity, capability=capability, backend=_FakeBackend([], anchor),
            graph_loader=lambda _root: graph, anchor_loader=lambda _root: anchor,
        )
    assert not (root / plan.RESULT_ROOT_RELATIVE).exists()


def test_closure_dry_cli_and_module_import_are_inert() -> None:
    closure = plan.implementation_closure(ROOT)
    assert len(closure["paths"]) >= 30
    assert plan.validate_current_closure(ROOT, closure) == closure
    with pytest.raises(plan.M1GapPlanError):
        plan.validate_current_closure(ROOT, {**closure, "closure_sha256": "0" * 64})
    process = subprocess.run(
        [sys.executable, str(CLI), "--dry-run"], text=True, capture_output=True, check=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    payload = json.loads(process.stdout)
    assert payload["opens_nwb_or_checkpoint"] is False
    assert payload["imports_torch"] is False
    assert payload["public_execution_authorized"] is False
    assert payload["anchor_reproduction_required_exactly"] is True
    assert subprocess.run(
        [sys.executable, str(CLI), "--execute"], text=True, capture_output=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""},
    ).returncode != 0
    probe = (
        "import sys, json;"
        f"sys.path.insert(0, {str(ROOT)!r});"
        "from tfpd_exploration.src.m1_heldin_heldout_gap_v1 import plan, score;"
        "print(json.dumps({'torch': 'torch' in sys.modules, 'numpy': 'numpy' in sys.modules,"
        " 'cuda': bool(sys.modules.get('torch.cuda'))}))"
    )
    inert = subprocess.run(
        [sys.executable, "-c", probe], text=True, capture_output=True, check=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    assert json.loads(inert.stdout) == {"torch": False, "numpy": False, "cuda": False}
