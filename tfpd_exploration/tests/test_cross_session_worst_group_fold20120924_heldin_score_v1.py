"""Synthetic/no-CUDA checks for CS-WG fold-20120924 held-in score V1."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import pytest

from tfpd_exploration.src.cross_session_worst_group_fold20120924_score_v1 import plan, score
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1


REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "tfpd_exploration/scripts/run_cross_session_worst_group_fold20120924_heldin_score_v1.py"


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


def _fixture_full_graph(tmp_path: Path) -> tuple[plan.CompletedFullExpectation, score.CompletedFullGraph]:
    directory = tmp_path / "producer"
    directory.mkdir()
    checkpoint = b"synthetic-best-checkpoint-body"
    checkpoint_sha = _sha(checkpoint)
    state_sha = "a" * 64
    identity = {
        "schema": "cross_session_worst_group_m1_full_training_identity_v1",
        "spec": {
            "inherited_v1_full_spec": {
                "stage0_run_spec": {
                    "system": "CS_WG",
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
        body_sha[name] = _write_pair(directory, name, _json({"name": name}))
    for index in range(20):
        name = f"epoch_{index:02d}.json"
        body_sha[name] = _write_pair(directory, name, _json({"name": name, "epoch_index": index}))
    training = {
        "schema": "cross_session_worst_group_m1_full_training_result_v1",
        "identity_sha256": identity_sha,
        "swa_enabled": False,
        "swa_artifact_forbidden": True,
        "source_only": True,
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
                "filename": "checkpoint_best_source_train_loss.pt", "sha256": checkpoint_sha,
                "state_sha256": state_sha, "strict_reload": True,
            },
            "last": {
                "filename": "checkpoint_last.pt", "sha256": checkpoint_sha,
                "state_sha256": state_sha, "strict_reload": True,
            },
        },
        "best_epoch_index": 19,
        "last_epoch_index": 19,
        "swa_enabled": False,
        "swa_artifact_forbidden": True,
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
        "epoch_sha256": {f"epoch_{index:02d}.json": body_sha[f"epoch_{index:02d}.json"] for index in range(20)},
        "checkpoint_manifest_sha256": body_sha["checkpoint_manifest.json"],
        "checkpoint_best_source_train_loss_sha256": checkpoint_sha,
        "checkpoint_last_sha256": checkpoint_sha,
        "checkpoint_best_source_train_loss_state_sha256": state_sha,
        "checkpoint_last_state_sha256": state_sha,
        "swa_enabled": False,
        "swa_artifact_forbidden": True,
        "source_only": True,
        "target_optimizer_backward_update": 0,
    }
    terminal_sha = _write_pair(directory, "terminal.json", _json(terminal))
    expectation = plan.CompletedFullExpectation(
        root_relative="producer", terminal_sha256=terminal_sha,
        training_sha256=body_sha["training.json"], identity_sha256=identity_sha,
        checkpoint_manifest_sha256=body_sha["checkpoint_manifest.json"],
        best_checkpoint_sha256=checkpoint_sha, best_checkpoint_state_sha256=state_sha,
    )
    graph = score.load_completed_full_graph(tmp_path, expectation=expectation)
    return expectation, graph


class _FakeBackend:
    def __init__(self, events: list[str], *, fail_at: str | None = None) -> None:
        self.events = events
        self.fail_at = fail_at
        self.closed = False
        self.source_root = Path("/synthetic/m1-heldin")

    def launch_payload(self, identity: plan.ScoreIdentity, graph: score.CompletedFullGraph) -> Mapping[str, object]:
        self.events.append("backend:launch")
        if self.fail_at == "launch":
            raise RuntimeError("synthetic launch failure")
        return {"schema": "synthetic-score-backend", "opens_target": False}

    def prepare_inputs(self, identity: plan.ScoreIdentity, graph: score.CompletedFullGraph) -> Mapping[str, object]:
        self.events.append("backend:prepare")
        if self.fail_at == "prepare":
            raise RuntimeError("synthetic prepare failure")
        return {
            "n_windows": 7,
            "model_input_shape": [100, 64],
            "calibration_shape_per_row": [10, 1024, 64],
            "input_record_sha256": "1" * 64,
            "ordered_window_start_sha256": "2" * 64,
            "target_sha256": "3" * 64,
            "calibration_sha256": "4" * 64,
        }

    def score(self, identity: plan.ScoreIdentity, graph: score.CompletedFullGraph,
              input_authority: Mapping[str, object]) -> Mapping[str, object]:
        self.events.append("backend:score")
        if self.fail_at == "score":
            raise RuntimeError("synthetic score failure")
        input_sha = _sha(_json(dict(input_authority)))
        return {
            "schema": "cross_session_worst_group_m1_fold20120924_heldin_score_payload_v1",
            "identity_sha256": identity.sha256,
            "completed_full_graph_sha256": graph.sha256,
            "input_authority_sha256": input_sha,
            "target_session": plan.TARGET_SESSION,
            "selected_checkpoint_role": "best_source_train_loss",
            "metric": plan.METRIC_LABEL,
            "last_bin_only": True,
            "eval_mode": True,
            "no_grad": True,
            "dynamic_dropout_disabled": True,
            "target_metric_only": True,
            "target_labels_used_only_for_metric": True,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
            "full_system_forward_count": 1,
            "n_windows": 7,
            "governing_r2": 0.25,
            "prediction_sha256": "5" * 64,
            "target_sha256": input_authority["target_sha256"],
            "model_state_before_sha256": "6" * 64,
            "model_state_after_sha256": "6" * 64,
        }

    def progress(self) -> score.ScoreProgress:
        return score.ScoreProgress(
            target_resolved_or_opened=self.fail_at in {"prepare", "score"},
            checkpoint_opened=self.fail_at == "score", model_constructed=False,
            full_system_forward_count=0,
        )

    def close(self) -> None:
        self.closed = True


def _identity() -> plan.ScoreIdentity:
    return plan.build_identity(REPO_ROOT)


def _lifecycle_root(tmp_path: Path) -> Path:
    (tmp_path / "tfpd_exploration/results").mkdir(parents=True)
    return tmp_path


def test_completed_full_56_leaf_graph_reloads_and_selects_best_role(tmp_path: Path) -> None:
    expectation, graph = _fixture_full_graph(tmp_path)
    assert graph.expectation == expectation
    assert len(graph.body_sha256) == 28
    assert graph.body_sha256["checkpoint_best_source_train_loss.pt"] == expectation.best_checkpoint_sha256
    assert graph.body_sha256["checkpoint_last.pt"] == expectation.best_checkpoint_sha256
    assert graph.manifest["checkpoints"]["best_source_train_loss"]["filename"] == "checkpoint_best_source_train_loss.pt"
    assert score.read_selected_checkpoint_bytes(tmp_path, graph) == b"synthetic-best-checkpoint-body"


@pytest.mark.parametrize("mutation", ("extra", "sidecar", "mode", "identity", "role"))
def test_completed_full_graph_tampering_fails_closed(tmp_path: Path, mutation: str) -> None:
    expectation, _graph = _fixture_full_graph(tmp_path)
    directory = tmp_path / "producer"
    if mutation == "extra":
        (directory / "failure.json").write_text("{}")
    elif mutation == "sidecar":
        os.chmod(directory / "terminal.json.sha256", 0o644)
        (directory / "terminal.json.sha256").write_text("forged\n")
        os.chmod(directory / "terminal.json.sha256", 0o444)
    elif mutation == "mode":
        os.chmod(directory / "checkpoint_last.pt", 0o644)
    elif mutation == "identity":
        # Replacing a body/sidecar pair cannot evade the literal terminal SHA.
        terminal = json.loads((directory / "terminal.json").read_text())
        terminal["identity"]["spec"]["inherited_v1_full_spec"]["stage0_run_spec"]["outer_target_session"] = "20120926"
        _replace_pair(directory, "terminal.json", _json(terminal))
    else:
        manifest = json.loads((directory / "checkpoint_manifest.json").read_text())
        manifest["checkpoints"]["best_source_train_loss"]["filename"] = "checkpoint_last.pt"
        _replace_pair(directory, "checkpoint_manifest.json", _json(manifest))
    with pytest.raises(score.HeldInScoreError):
        score.load_completed_full_graph(tmp_path, expectation=expectation)


def test_lifecycle_publishes_attempt_before_backend_and_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expectation, graph = _fixture_full_graph(tmp_path)
    root = _lifecycle_root(tmp_path)
    identity = plan.ScoreIdentity(plan.ScoreSpec(), plan.implementation_closure(REPO_ROOT), expectation)
    capability = score.ScoreCapability(identity.sha256, graph.sha256, "/synthetic/m1-heldin", score._SCORE_REVIEW_SEAL)
    events: list[str] = []
    original = v1.ImmutableArtifactRoot.publish_json

    def observed_publish(self: v1.ImmutableArtifactRoot, name: str, payload: Mapping[str, object]) -> str:
        events.append(f"publish:{name}")
        return original(self, name, payload)

    # Lifecycle publication is tested in a temporary artifact namespace.  The
    # production call intentionally uses one repository root for current-byte
    # closure validation and its canonical result-relative root; the separate
    # closure test above covers that real code-root law without creating any
    # workspace result root.  The mock lifecycle therefore injects only the
    # revalidation function, not a physical/data path.
    monkeypatch.setattr(score, "validate_identity_current", lambda _root, observed: observed == identity)
    monkeypatch.setattr(v1.ImmutableArtifactRoot, "publish_json", observed_publish)
    backend = _FakeBackend(events)
    result = score.execute_reviewed_heldin_score(root, identity=identity, capability=capability,
                                                  backend=backend, graph_loader=lambda _root: graph)
    assert result.terminal_sha256 is not None and result.failure_sha256 is None
    assert events.index("publish:attempt.json") < events.index("backend:launch") < events.index("backend:prepare")
    assert events.index("publish:input_authority.json") < events.index("backend:score")
    assert backend.closed is True
    leaves = sorted((root / identity.spec.root_relative).iterdir())
    assert len(leaves) == 10
    assert not any(path.name == "failure.json" for path in leaves)


def test_lifecycle_failure_is_honest_and_does_not_publish_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expectation, graph = _fixture_full_graph(tmp_path)
    root = _lifecycle_root(tmp_path)
    identity = plan.ScoreIdentity(plan.ScoreSpec(), plan.implementation_closure(REPO_ROOT), expectation)
    capability = score.ScoreCapability(identity.sha256, graph.sha256, "/synthetic/m1-heldin", score._SCORE_REVIEW_SEAL)
    backend = _FakeBackend([], fail_at="prepare")
    monkeypatch.setattr(score, "validate_identity_current", lambda _root, observed: observed == identity)
    result = score.execute_reviewed_heldin_score(root, identity=identity, capability=capability,
                                                  backend=backend, graph_loader=lambda _root: graph)
    assert result.failure_sha256 is not None and result.terminal_sha256 is None
    failure = json.loads((root / identity.spec.root_relative / "failure.json").read_text())
    assert failure["progress"]["target_resolved_or_opened"] is True
    assert failure["progress"]["target_optimizer_steps"] == 0
    assert failure["terminal_published"] is False
    assert not (root / identity.spec.root_relative / "terminal.json").exists()


def test_capability_rejects_target_source_root_substitution_before_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    expectation, graph = _fixture_full_graph(tmp_path)
    root = _lifecycle_root(tmp_path)
    identity = plan.ScoreIdentity(plan.ScoreSpec(), plan.implementation_closure(REPO_ROOT), expectation)
    capability = score.ScoreCapability(identity.sha256, graph.sha256, "/synthetic/canonical", score._SCORE_REVIEW_SEAL)
    backend = _FakeBackend([])
    monkeypatch.setattr(score, "validate_identity_current", lambda _root, observed: observed == identity)
    with pytest.raises(score.HeldInScoreError, match="source-root"):
        score.execute_reviewed_heldin_score(root, identity=identity, capability=capability,
                                            backend=backend, graph_loader=lambda _root: graph)
    assert not (root / identity.spec.root_relative).exists()


def test_identity_closure_and_dry_cli_are_inert() -> None:
    closure = plan.implementation_closure(REPO_ROOT)
    assert len(closure["paths"]) >= 20
    plan.validate_current_closure(REPO_ROOT, closure)
    with pytest.raises(plan.HeldInScorePlanError):
        plan.validate_current_closure(REPO_ROOT, {**closure, "closure_sha256": "0" * 64})
    process = subprocess.run(
        [sys.executable, str(CLI), "--dry-run"], capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    payload = json.loads(process.stdout)
    assert payload["opens_nwb_or_checkpoint"] is False
    assert payload["imports_torch"] is False
    assert payload["public_execution_authorized"] is False
    assert subprocess.run(
        [sys.executable, str(CLI), "--execute"], capture_output=True, text=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""},
    ).returncode != 0


def test_module_import_is_torch_free_and_metric_matches_governing_cpu_reference() -> None:
    import torch
    from torchmetrics.regression import R2Score
    from tfpd_exploration.src.cross_session_worst_group_fold20120924_score_v1 import physical

    prediction = torch.tensor([[0.1, 0.4], [0.2, 0.1], [0.5, 0.2], [0.9, 0.7]], dtype=torch.float32)
    target = torch.tensor([[0.0, 0.5], [0.3, 0.2], [0.6, 0.3], [0.8, 0.8]], dtype=torch.float32)
    # Expand the deterministic two-output fixture to the exact M1 16-output metric surface.
    prediction16 = prediction.repeat(1, 8)
    target16 = target.repeat(1, 8)
    observed = physical.variance_weighted_last_bin_r2(prediction16, target16)
    reference = float(R2Score(multioutput="variance_weighted")(prediction16, target16))
    assert observed == reference
    with pytest.raises(physical.HeldInScorePhysicalError, match="shape"):
        physical.variance_weighted_last_bin_r2(prediction16.unsqueeze(1), target16.unsqueeze(1))
    assert torch.cuda.is_initialized() is False


def test_physical_source_is_lazy_and_has_no_top_level_torch_import() -> None:
    source = (REPO_ROOT / "tfpd_exploration/src/cross_session_worst_group_fold20120924_score_v1/physical.py").read_text()
    top_level = source.split("def variance_weighted_last_bin_r2", 1)[0]
    assert "import torch" not in top_level
    assert "from torch" not in top_level
