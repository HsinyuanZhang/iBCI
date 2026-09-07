"""Synthetic/no-CUDA tests for terminal-pinned matched-ERM held-in score V1."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import pytest

from tfpd_exploration.src.cross_session_worst_group_fold20120924_matched_erm_score_v1 import plan, score
from tfpd_exploration.src.cross_session_worst_group_fold20120924_score_v1 import plan as cswg_plan
from tfpd_exploration.src.cross_session_worst_group_fold20120924_score_v1 import score as shared_score
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "tfpd_exploration/scripts/run_cross_session_worst_group_fold20120924_matched_erm_heldin_score_v1.py"


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


def _fixture_binding_and_graph(tmp_path: Path) -> tuple[plan.MatchedERMFullBinding, score.MatchedERMCompletedFullGraph]:
    directory = tmp_path / plan.MATCHED_ERM_FULL_ROOT_RELATIVE
    directory.mkdir(parents=True)
    best_checkpoint = b"synthetic-matched-erm-best-checkpoint"
    last_checkpoint = b"synthetic-matched-erm-last-checkpoint"
    best_state = "b" * 64
    last_state = "c" * 64
    identity = {
        "schema": "cross_session_worst_group_m1_full_training_identity_v1",
        "spec": {
            "inherited_v1_full_spec": {
                "stage0_run_spec": {
                    "system": "MATCHED_ERM", "lambda": 0.0, "tau": 0.01,
                    "outer_target_session": plan.TARGET_SESSION,
                    "source_sessions": list(plan.SOURCE_SESSIONS),
                },
            },
            "swa_enabled": False,
            "swa_artifact_forbidden": True,
        },
    }
    identity_sha = _sha(_json(identity))
    body_sha: dict[str, str] = {}
    for name in ("attempt.json", "launch.json", "source_authority.json"):
        body_sha[name] = _write_pair(directory, name, _json({"name": name, "system": "MATCHED_ERM"}))
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
        directory, "checkpoint_best_source_train_loss.pt", best_checkpoint,
    )
    body_sha["checkpoint_last.pt"] = _write_pair(directory, "checkpoint_last.pt", last_checkpoint)
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
                "state_sha256": last_state, "strict_reload": True,
            },
        },
        "best_epoch_index": 11, "last_epoch_index": 19,
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
        "checkpoint_last_state_sha256": last_state,
        "swa_enabled": False, "swa_artifact_forbidden": True,
        "source_only": True, "target_optimizer_backward_update": 0,
    }
    body_sha["terminal.json"] = _write_pair(directory, "terminal.json", _json(terminal))
    binding = plan.MatchedERMFullBinding(
        attempt_sha256=body_sha["attempt.json"], launch_sha256=body_sha["launch.json"],
        source_authority_sha256=body_sha["source_authority.json"], training_sha256=body_sha["training.json"],
        identity_sha256=identity_sha, checkpoint_manifest_sha256=body_sha["checkpoint_manifest.json"],
        best_checkpoint_sha256=body_sha["checkpoint_best_source_train_loss.pt"],
        best_checkpoint_state_sha256=best_state, last_checkpoint_sha256=body_sha["checkpoint_last.pt"],
        last_checkpoint_state_sha256=last_state, terminal_sha256=body_sha["terminal.json"],
        best_epoch_index=11, last_epoch_index=19,
    )
    graph = score.load_completed_matched_erm_full_graph(tmp_path, binding=binding)
    return binding, graph


def _fixture_comparator(tmp_path: Path) -> plan.SameInputAnchor:
    """Create an actual-shaped immutable CS-WG receipt graph under temp root."""
    directory = tmp_path / plan.CSWG_COMPARATOR_ROOT_RELATIVE
    directory.mkdir(parents=True, exist_ok=True)
    values = {
        "target_descriptor_sha256": "d" * 64,
        "calibration_sha256": "e" * 64,
        "ordered_window_start_sha256": "f" * 64,
        "ordered_query_identity_sha256": "a" * 64,
        "ordered_target_evalmask_sha256": "b" * 64,
        "target_sha256": "c" * 64,
        "reader_recipe_sha256": "9" * 64,
    }
    input_authority = {
        "schema": "cross_session_worst_group_m1_fold20120924_heldin_input_authority_v1",
        "n_windows": 54_849,
        "model_input_shape": [100, 64],
        "calibration_shape_per_row": [10, 1024, 64],
        "input_record_sha256": "8" * 64,
        "ordered_window_start_sha256": values["ordered_window_start_sha256"],
        "target_sha256": values["target_sha256"],
        "calibration_sha256": values["calibration_sha256"],
        "target_descriptor": {
            "sha256": values["target_descriptor_sha256"], "byte_count": 73_077_382,
        },
        "target_reader_native_evidence": {
            "ordered_query_identity_sha256": values["ordered_query_identity_sha256"],
            "ordered_target_evalmask_sha256": values["ordered_target_evalmask_sha256"],
            "reader_recipe_sha256": values["reader_recipe_sha256"],
        },
    }
    input_sha = _write_pair(directory, "input_authority.json", _json(input_authority))
    score_body = {
        "schema": "cross_session_worst_group_m1_fold20120924_heldin_score_payload_v1",
        "input_authority": input_authority,
        "n_windows": 54_849,
        "target_sha256": values["target_sha256"],
        "governing_r2": 0.5,
    }
    score_sha = _write_pair(directory, "score.json", _json(score_body))
    _write_pair(directory, "attempt.json", _json({"status": "ATTEMPT_RESERVED"}))
    _write_pair(directory, "launch.json", _json({"status": "LAUNCHED"}))
    _write_pair(directory, "terminal.json", _json({
        "status": "COMPLETE_DESCRIPTIVE_HELDIN_R2", "input_authority_sha256": input_sha,
        "score_sha256": score_sha,
    }))
    return plan.SameInputAnchor(
        comparator_input_authority_sha256=input_sha, comparator_score_sha256=score_sha,
        n_windows=54_849, target_descriptor_sha256=values["target_descriptor_sha256"],
        target_descriptor_byte_count=73_077_382, calibration_sha256=values["calibration_sha256"],
        ordered_window_start_sha256=values["ordered_window_start_sha256"],
        ordered_query_identity_sha256=values["ordered_query_identity_sha256"],
        ordered_target_evalmask_sha256=values["ordered_target_evalmask_sha256"],
        target_sha256=values["target_sha256"], reader_recipe_sha256=values["reader_recipe_sha256"],
    )


def _prepared_fragment(anchor: plan.SameInputAnchor = plan.DEFAULT_SAME_INPUT_ANCHOR) -> dict[str, object]:
    return {
        "n_windows": anchor.n_windows,
        "model_input_shape": [100, 64],
        "calibration_shape_per_row": [10, 1024, 64],
        "input_record_sha256": "1" * 64,
        "ordered_window_start_sha256": anchor.ordered_window_start_sha256,
        "target_sha256": anchor.target_sha256,
        "calibration_sha256": anchor.calibration_sha256,
        "target_descriptor": {
            "session_id": plan.TARGET_SESSION, "relative_path": "sub-C/synthetic.nwb",
            "sha256": anchor.target_descriptor_sha256, "byte_count": anchor.target_descriptor_byte_count,
        },
        "target_reader_native_evidence": {
            "ordered_query_identity_sha256": anchor.ordered_query_identity_sha256,
            "ordered_target_evalmask_sha256": anchor.ordered_target_evalmask_sha256,
            "reader_recipe_sha256": anchor.reader_recipe_sha256,
        },
    }


class _FakeBackend:
    def __init__(self, events: list[str], *, fail_at: str | None = None) -> None:
        self.events = events
        self.fail_at = fail_at
        self.source_root = Path("/synthetic/m1-heldin")
        self.closed = False

    def launch_payload(self, identity: plan.MatchedERMScoreIdentity,
                       graph: score.MatchedERMCompletedFullGraph) -> Mapping[str, object]:
        self.events.append("backend:launch")
        if self.fail_at == "launch":
            raise RuntimeError("synthetic launch failure")
        return score.MatchedERMPhysicalCodec().launch_payload(identity, graph, device="cpu")

    def prepare_inputs(self, identity: plan.MatchedERMScoreIdentity,
                       graph: score.MatchedERMCompletedFullGraph) -> Mapping[str, object]:
        self.events.append("backend:prepare")
        if self.fail_at == "prepare":
            raise RuntimeError("synthetic prepare failure")
        return _prepared_fragment(identity.same_input_anchor)

    def score(self, identity: plan.MatchedERMScoreIdentity, graph: score.MatchedERMCompletedFullGraph,
              input_authority: Mapping[str, object]) -> Mapping[str, object]:
        self.events.append("backend:score")
        if self.fail_at == "score":
            raise RuntimeError("synthetic score failure")
        return score.MatchedERMPhysicalCodec().score_payload(
            identity, graph, input_authority, forward_count=1, n_windows=plan.EXPECTED_N_WINDOWS,
            prediction_sha256="5" * 64, target_sha256=identity.same_input_anchor.target_sha256,
            governing_r2=0.25, model_state_before_sha256="6" * 64, model_state_after_sha256="6" * 64,
        )

    def progress(self) -> shared_score.ScoreProgress:
        return shared_score.ScoreProgress(
            target_resolved_or_opened=self.fail_at in {"prepare", "score"},
            checkpoint_opened=self.fail_at == "score", model_constructed=False,
        )

    def close(self) -> None:
        self.closed = True


def _identity(
    binding: plan.MatchedERMFullBinding,
    anchor: plan.SameInputAnchor = plan.DEFAULT_SAME_INPUT_ANCHOR,
) -> plan.MatchedERMScoreIdentity:
    return plan.MatchedERMScoreIdentity(
        plan.MatchedERMScoreSpec(), plan.implementation_closure(ROOT), binding, anchor,
    )


def _lifecycle_root(tmp_path: Path) -> Path:
    (tmp_path / "tfpd_exploration/results").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_binding_has_no_default_future_literals_and_default_v1_payload_is_unchanged() -> None:
    with pytest.raises(TypeError):
        plan.MatchedERMFullBinding()  # type: ignore[call-arg]
    historical = cswg_plan.DEFAULT_COMPLETED_FULL_EXPECTATION.payload()
    assert historical["last_checkpoint_must_equal_selected_best"] is True
    assert historical["best_epoch_index"] == historical["last_epoch_index"] == 19
    assert "extra_body_sha256" not in historical and "expected_system" not in historical


def test_exact_matched_erm_56_leaf_graph_binds_attempt_authority_and_distinct_last(tmp_path: Path) -> None:
    binding, graph = _fixture_binding_and_graph(tmp_path)
    assert len(graph.shared_graph.body_sha256) == 28
    assert graph.shared_graph.body_sha256["attempt.json"] == binding.attempt_sha256
    assert graph.shared_graph.body_sha256["source_authority.json"] == binding.source_authority_sha256
    assert graph.shared_graph.body_sha256["checkpoint_best_source_train_loss.pt"] != graph.shared_graph.body_sha256["checkpoint_last.pt"]
    assert graph.shared_graph.manifest["best_epoch_index"] == 11
    assert graph.shared_graph.manifest["last_epoch_index"] == 19


@pytest.mark.parametrize("mutation", ("source_authority", "lambda", "best_state", "extra"))
def test_producer_graph_adversaries_fail_closed(tmp_path: Path, mutation: str) -> None:
    binding, _graph = _fixture_binding_and_graph(tmp_path)
    directory = tmp_path / plan.MATCHED_ERM_FULL_ROOT_RELATIVE
    if mutation == "source_authority":
        _replace_pair(directory, "source_authority.json", _json({"forged": True}))
    elif mutation == "lambda":
        terminal = json.loads((directory / "terminal.json").read_text())
        terminal["identity"]["spec"]["inherited_v1_full_spec"]["stage0_run_spec"]["lambda"] = 1.0
        _replace_pair(directory, "terminal.json", _json(terminal))
    elif mutation == "best_state":
        manifest = json.loads((directory / "checkpoint_manifest.json").read_text())
        manifest["checkpoints"]["best_source_train_loss"]["state_sha256"] = "0" * 64
        _replace_pair(directory, "checkpoint_manifest.json", _json(manifest))
    else:
        (directory / "failure.json").write_text("{}")
    with pytest.raises(score.MatchedERMHeldInScoreError):
        score.load_completed_matched_erm_full_graph(tmp_path, binding=binding)


def test_same_input_anchor_compares_underlying_evidence_not_whole_input_sha(tmp_path: Path) -> None:
    binding, _graph = _fixture_binding_and_graph(tmp_path)
    anchor = _fixture_comparator(tmp_path)
    graph = score.load_matched_erm_score_graph(tmp_path, binding=binding, anchor=anchor)
    identity = _identity(binding, anchor)
    payload = score._input_authority_payload(identity, graph, _prepared_fragment(anchor))
    assert payload["input_record_sha256"] == "1" * 64
    assert payload["same_input_anchor"]["comparator_input_authority_sha256"] \
        == anchor.comparator_input_authority_sha256
    assert payload["cswg_comparator_evidence_sha256"] == graph.comparator_evidence.sha256
    wrong = _prepared_fragment(anchor)
    wrong["target_reader_native_evidence"] = dict(wrong["target_reader_native_evidence"])
    wrong["target_reader_native_evidence"]["reader_recipe_sha256"] = "0" * 64
    with pytest.raises(score.MatchedERMHeldInScoreError, match="same-input"):
        score._input_authority_payload(identity, graph, wrong)


def test_comparator_descriptor_reconstruction_rejects_rehashed_field_drift(tmp_path: Path) -> None:
    """A valid replacement pair cannot turn the CS-WG evidence into an anchor."""
    anchor = _fixture_comparator(tmp_path)
    directory = tmp_path / plan.CSWG_COMPARATOR_ROOT_RELATIVE
    inputs = json.loads((directory / "input_authority.json").read_text())
    native = dict(inputs["target_reader_native_evidence"])
    native["reader_recipe_sha256"] = "0" * 64
    inputs["target_reader_native_evidence"] = native
    input_sha = _replace_pair(directory, "input_authority.json", _json(inputs))
    scored = json.loads((directory / "score.json").read_text())
    scored["input_authority"] = inputs
    score_sha = _replace_pair(directory, "score.json", _json(scored))
    terminal = json.loads((directory / "terminal.json").read_text())
    terminal["input_authority_sha256"] = input_sha
    terminal["score_sha256"] = score_sha
    _replace_pair(directory, "terminal.json", _json(terminal))
    rehashed_anchor = replace(
        anchor,
        comparator_input_authority_sha256=input_sha,
        comparator_score_sha256=score_sha,
    )
    with pytest.raises(score.MatchedERMHeldInScoreError, match="same-input"):
        score.load_cswg_same_input_comparator_evidence(tmp_path, anchor=rehashed_anchor)


def test_temp_lifecycle_is_attempt_before_target_and_terminal_pins_comparator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, _graph = _fixture_binding_and_graph(tmp_path)
    anchor = _fixture_comparator(tmp_path)
    graph = score.load_matched_erm_score_graph(tmp_path, binding=binding, anchor=anchor)
    root = _lifecycle_root(tmp_path)
    identity = _identity(binding, anchor)
    capability = score.MatchedERMScoreCapability(
        identity.sha256, graph.sha256, "/synthetic/m1-heldin", score._MATCHED_ERM_SCORE_REVIEW_SEAL,
    )
    events: list[str] = []
    backend = _FakeBackend(events)
    original = v1.ImmutableArtifactRoot.publish_json

    def observed_publish(self: object, name: str, payload: Mapping[str, object]) -> str:
        events.append(f"publish:{name}")
        return original(self, name, payload)

    monkeypatch.setattr(score, "validate_identity_current", lambda _root, observed: observed == identity)
    monkeypatch.setattr(v1.ImmutableArtifactRoot, "publish_json", observed_publish)
    result = score.execute_reviewed_matched_erm_heldin_score(
        root, identity=identity, capability=capability, backend=backend,
    )
    assert result.terminal_sha256 is not None and result.failure_sha256 is None
    assert events.index("publish:attempt.json") < events.index("backend:launch") < events.index("backend:prepare")
    assert events.index("publish:input_authority.json") < events.index("backend:score")
    terminal = json.loads((root / identity.spec.root_relative / "terminal.json").read_text())
    assert terminal["n_windows"] == 54_849
    assert terminal["comparator_cswg_score_sha256"] == anchor.comparator_score_sha256
    assert terminal["cswg_comparator_evidence_sha256"] == graph.comparator_evidence.sha256
    assert terminal["target_optimizer_backward_update"] == 0
    assert backend.closed is True


def test_failure_remains_honest_and_target_updates_are_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binding, _graph = _fixture_binding_and_graph(tmp_path)
    anchor = _fixture_comparator(tmp_path)
    graph = score.load_matched_erm_score_graph(tmp_path, binding=binding, anchor=anchor)
    root = _lifecycle_root(tmp_path)
    identity = _identity(binding, anchor)
    capability = score.MatchedERMScoreCapability(
        identity.sha256, graph.sha256, "/synthetic/m1-heldin", score._MATCHED_ERM_SCORE_REVIEW_SEAL,
    )
    monkeypatch.setattr(score, "validate_identity_current", lambda _root, observed: observed == identity)
    result = score.execute_reviewed_matched_erm_heldin_score(
        root, identity=identity, capability=capability, backend=_FakeBackend([], fail_at="prepare"),
    )
    assert result.failure_sha256 is not None and result.terminal_sha256 is None
    failure = json.loads((root / identity.spec.root_relative / "failure.json").read_text())
    assert failure["progress"]["target_resolved_or_opened"] is True
    assert failure["progress"]["target_optimizer_steps"] == 0
    assert failure["comparator_cswg_input_authority_sha256"] == anchor.comparator_input_authority_sha256
    assert failure["comparator_cswg_score_sha256"] == anchor.comparator_score_sha256
    assert failure["cswg_comparator_evidence_sha256"] == graph.comparator_evidence.sha256
    assert failure["terminal_published"] is False


def test_capability_or_source_root_substitution_fails_before_reserve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binding, _graph = _fixture_binding_and_graph(tmp_path)
    anchor = _fixture_comparator(tmp_path)
    graph = score.load_matched_erm_score_graph(tmp_path, binding=binding, anchor=anchor)
    root = _lifecycle_root(tmp_path)
    identity = _identity(binding, anchor)
    capability = score.MatchedERMScoreCapability(
        identity.sha256, graph.sha256, "/synthetic/canonical", score._MATCHED_ERM_SCORE_REVIEW_SEAL,
    )
    monkeypatch.setattr(score, "validate_identity_current", lambda _root, observed: observed == identity)
    with pytest.raises(score.MatchedERMHeldInScoreError, match="source-root"):
        score.execute_reviewed_matched_erm_heldin_score(
            root, identity=identity, capability=capability, backend=_FakeBackend([]),
        )
    assert not (root / identity.spec.root_relative).exists()


def test_codec_uses_wrapped_shared_checkpoint_graph_without_physical_or_torch(tmp_path: Path) -> None:
    binding, _graph = _fixture_binding_and_graph(tmp_path)
    anchor = _fixture_comparator(tmp_path)
    graph = score.load_matched_erm_score_graph(tmp_path, binding=binding, anchor=anchor)
    identity = _identity(binding, anchor)
    codec = score.MatchedERMPhysicalCodec()
    codec.validate_identity_graph(identity, graph)
    assert codec.shared_checkpoint_graph(graph) is graph.shared_graph
    payload = codec.input_authority_payload(identity, graph, _prepared_fragment(anchor))
    assert payload["completed_matched_erm_full_graph_sha256"] == graph.sha256
    with pytest.raises(score.MatchedERMHeldInScoreError):
        codec.validate_identity_graph(object(), object())


def test_closure_dry_cli_and_module_import_are_inert() -> None:
    closure = plan.implementation_closure(ROOT)
    assert len(closure["paths"]) >= 30
    assert plan.validate_current_closure(ROOT, closure) == closure
    with pytest.raises(plan.MatchedERMHeldInScorePlanError):
        plan.validate_current_closure(ROOT, {**closure, "closure_sha256": "0" * 64})
    process = subprocess.run(
        [sys.executable, str(CLI), "--dry-run"], text=True, capture_output=True, check=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    payload = json.loads(process.stdout)
    assert payload["opens_future_result_or_nwb"] is False
    assert payload["imports_torch"] is False
    assert payload["public_execution_authorized"] is False
    assert subprocess.run(
        [sys.executable, str(CLI), "--execute"], text=True, capture_output=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""},
    ).returncode != 0
