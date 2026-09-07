"""No-data/no-CUDA tests for the seed-43 Phase-E replication addendum."""
from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.tfsr_b3st4_ddrop_seed43_v1 import score_43 as route


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _body_sha(value: object) -> str:
    return route._sha(route._json_bytes(value))


def _training(*, cell: str = route.CELL, build: str = route.BUILD_LABEL) -> route.Seed43TrainingEvidence:
    return route.Seed43TrainingEvidence(
        terminal_sha256=SHA_A,
        swa_sha256=SHA_B,
        swa_state_digest=SHA_C,
        checkpoint_sha256={"44": SHA_A, "45": SHA_B, "46": SHA_C, "47": SHA_D},
        launch_closure={"closure_sha256": SHA_A},
        lineage={"seed": 43},
        build_disclosure={"build": build},
        training_peak_memory_bytes=17,
        cell=cell,
    )


def _upstream(
    *,
    verdict: str = "CLEAR_GO",
    input_payload: dict[str, object] | None = None,
    score_payload: dict[str, object] | None = None,
    terminal_payload: dict[str, object] | None = None,
) -> route.UpstreamSeed42Evidence:
    payload = input_payload or {
        "schema": "tfsr_phase_e_input_authority_v1",
        "cell": "TFSR_B3ST4_DDROP_SEED42",
        "records": [],
        "source_normalizer_body_sha256": SHA_A,
        "t4_normalizer_semantic_sha256": SHA_B,
        "behavior_normalizer_semantic_sha256": SHA_C,
        "strict_manifest_sha256": SHA_D,
        "raw_to_normalized_exact": True,
        "no_cache_readonly_adapter": True,
    }
    score = score_payload or {"cell_d": {"within": {}, "external": {}}, "a2_contextual": {"pooled_per_session": {}}}
    terminal = terminal_payload or {"identity": {}}
    return route.UpstreamSeed42Evidence(
        input_authority_sha256=_body_sha(payload),
        score_sha256=_body_sha(score),
        terminal_sha256=_body_sha(terminal),
        verdict=verdict,
        input_payload=payload,
        score_payload=score,
        terminal_payload=terminal,
    )


def _authorization(training: route.Seed43TrainingEvidence, upstream: route.UpstreamSeed42Evidence) -> route.Seed43Authorization:
    closure = {"paths": [], "sha256_by_path": {}, "closure_sha256": SHA_D}
    return route.Seed43Authorization(
        preflight_sha256=SHA_A,
        root_authorization_sha256=SHA_B,
        preflight={}, root_authorization={}, closure=closure,
    )


def _identity(training: route.Seed43TrainingEvidence, upstream: route.UpstreamSeed42Evidence) -> tuple[route.Seed43ScoreIdentity, route.ExecutionCapability]:
    auth = _authorization(training, upstream)
    identity = route.Seed43ScoreIdentity(
        training=training,
        upstream=upstream,
        closure=auth.closure,
        authorization=auth.payload(),
    )
    return identity, route._issue_execution_capability(auth)


def test_seed43_evidence_rejects_seed42_and_wrong_build() -> None:
    with pytest.raises(ValueError, match="seed42"):
        _training(cell="TFSR_B3ST4_DDROP_SEED42")
    with pytest.raises(ValueError, match="build"):
        _training(build="v1_eager")
    with pytest.raises(ValueError, match="checkpoint"):
        route.Seed43TrainingEvidence(
            terminal_sha256=SHA_A, swa_sha256=SHA_B, swa_state_digest=SHA_C,
            checkpoint_sha256={"44": SHA_A}, launch_closure={}, lineage={},
            build_disclosure={"build": route.BUILD_LABEL}, training_peak_memory_bytes=0,
        )


def test_seed43_native_terminal_chain_rejects_failed_or_foreign_chain_before_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the seed43-only loader with fake receipts, never Tensor/NWB/CUDA."""

    class Spec:
        epochs = 48
        steps_per_epoch = 33_925
        throughput_probe_steps = 100
        checkpoint_epochs = (44, 45, 46, 47)
        total_optimizer_steps = 1_628_400

        @staticmethod
        def payload() -> dict[str, object]:
            return {"seed": 43, "epochs": 48}

    terminal = {
        "schema": "tfsr_b3st4_ddrop_seed43_train_terminal_v1",
        "status": "TRAINING_COMPLETE",
        "cell": route.CELL,
        "run_spec": Spec.payload(),
        "epochs": 48,
        "steps_per_epoch": 33_925,
        "total_optimizer_steps": 1_628_400,
        "attempt_sha256": SHA_A,
        "launch_sha256": SHA_B,
        "throughput_sha256": SHA_C,
        "epoch_receipt_sha256": [SHA_A] * 48,
        "checkpoint_sha256": {"44": SHA_A, "45": SHA_B, "46": SHA_C, "47": SHA_D},
        "swa_sha256": SHA_A,
        "swa_state_digest": SHA_B,
        "launch_closure": {"x": SHA_A},
        "final_closure": {"x": SHA_A},
        "lineage": {"seed": 43},
        "build_disclosure": {"build": route.BUILD_LABEL},
        "boundaries": {
            "source_only": True, "target_or_formal_opened": False,
            "scientific_result": False, "score": False, "capture_diagnostics": False,
        },
    }

    expected_pair_digests = {
        "terminal.json": None,
        "checkpoint-44.pt": SHA_A,
        "checkpoint-45.pt": SHA_B,
        "checkpoint-46.pt": SHA_C,
        "checkpoint-47.pt": SHA_D,
        "swa.pt": SHA_A,
    }

    class Artifact:
        def reload_pair(self, name: str, digest: str | None = None) -> bytes:
            if name not in expected_pair_digests:
                raise AssertionError(name)
            expected = expected_pair_digests[name]
            if expected is not None and digest != expected:
                raise RuntimeError(f"digest drift for {name}")
            if name == "terminal.json":
                return route._json_bytes(terminal)
            return b"synthetic"

        def reload_json(self, name: str, _digest: str) -> dict[str, object]:
            if name == "attempt.json":
                return {}
            if name == "launch.json":
                return {}
            if name.startswith("throughput"):
                return {}
            if name.startswith("epoch-"):
                return {"resources": {"peak_allocated_bytes": 3}}
            raise AssertionError(name)

        @staticmethod
        def has_name(_name: str) -> bool:
            return False

    calls: list[str] = []

    class Backend:
        def __init__(self, _root: Path) -> None:
            calls.append("backend")

        def validate_checkpoint(self, body: bytes, epoch: int, global_step: int, spec: Spec, **kwargs: object) -> dict[str, object]:
            assert body == b"synthetic" and epoch in Spec.checkpoint_epochs and global_step == (epoch + 1) * Spec.steps_per_epoch
            binding = kwargs["expected_binding"]
            if binding != {
                "cell": route.CELL,
                "run_spec": Spec.payload(),
                "launch_sha256": SHA_B,
                "launch_closure": {"x": SHA_A},
                "lineage": {"seed": 43},
            }:
                raise RuntimeError("checkpoint binding drift")
            return {}

        @staticmethod
        def validate_swa(body: bytes, spec: Spec, **kwargs: object) -> dict[str, object]:
            assert body == b"synthetic" and spec is Spec
            return {"state_digest": SHA_B}

    def validate_terminal_receipt(value: dict[str, object], *_: object) -> None:
        calls.append("terminal")
        if value.get("lineage") != {"seed": 43}:
            raise RuntimeError("lineage drift")

    fake_train = SimpleNamespace(
        CELL=route.CELL,
        PUBLIC_SPEC=Spec,
        production_identity=lambda root: {"identity": True},
        validate_terminal_receipt=validate_terminal_receipt,
        validate_attempt_receipt=lambda *args: calls.append("attempt"),
        validate_launch_receipt=lambda *args: calls.append("launch"),
        validate_throughput_receipt=lambda *args: calls.append("throughput"),
        validate_epoch_receipt=lambda *args: calls.append("epoch"),
        TrainingBackend43=Backend,
    )
    fake_contract = SimpleNamespace(
        verify_frozen_route=lambda root: calls.append("frozen"),
        verify_throughput_v2_receipt=lambda root: {"conclusion_fastest_kind": "jit_scripted_step"},
        validate_build_disclosure=lambda value: dict(value),
        BUILD_DISCLOSURE={"build": route.BUILD_LABEL},
    )
    monkeypatch.setattr(route, "_seed43_train", lambda: fake_train)
    monkeypatch.setattr(route, "_seed43_contract", lambda: fake_contract)
    monkeypatch.setattr(route, "_attach_readonly_seed43_artifact", lambda root, train: Artifact())
    evidence = route.validate_seed43_training_terminal(Path.cwd())
    assert evidence.cell == route.CELL
    assert calls[0:2] == ["frozen", "terminal"] or calls[0:2] == ["frozen", "backend"]
    terminal["status"] = "TRAINING_FAILED"
    with pytest.raises(route.FailClosedError, match="invalid before score input"):
        route.validate_seed43_training_terminal(Path.cwd())
    terminal["status"] = "TRAINING_COMPLETE"
    terminal["launch_closure"] = {"x": SHA_D}
    with pytest.raises(route.FailClosedError, match="invalid before score input"):
        route.validate_seed43_training_terminal(Path.cwd())
    terminal["launch_closure"] = {"x": SHA_A}
    terminal["lineage"] = {"seed": 42}
    with pytest.raises(route.FailClosedError, match="invalid before score input"):
        route.validate_seed43_training_terminal(Path.cwd())
    terminal["lineage"] = {"seed": 43}
    terminal["checkpoint_sha256"] = {"44": SHA_D, "45": SHA_B, "46": SHA_C, "47": SHA_D}
    with pytest.raises(route.FailClosedError, match="invalid before score input"):
        route.validate_seed43_training_terminal(Path.cwd())
    terminal["checkpoint_sha256"] = {"44": SHA_A, "45": SHA_B, "46": SHA_C, "47": SHA_D}
    terminal["swa_sha256"] = SHA_D
    with pytest.raises(route.FailClosedError, match="invalid before score input"):
        route.validate_seed43_training_terminal(Path.cwd())
    assert "backend" in calls  # validation happens before any physical score adapter/model exists.


def test_seed43_artifact_attachment_binds_frozen_train42_identity_and_rejects_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seed43 module must use the exact closure-bound generic root guard."""
    from src.tfsr_b3st4_ddrop_seed43_v1 import train_43

    root = tmp_path / "repository"
    directory = root / route.SEED43_TRAIN_ROOT_RELATIVE
    directory.parent.mkdir(parents=True)
    directory.mkdir()
    artifact = route._attach_readonly_seed43_artifact(root, train_43)
    assert isinstance(artifact, train_43.ArtifactRoot)
    assert route.FROZEN_TRAIN42_RELATIVE in route._addendum_closure_paths(SimpleNamespace(PHASE_E_CLOSURE=()))
    digest = artifact.publish_bytes("epoch-01.json", b'{"metadata_only": true}\n')
    assert artifact.reload_pair("epoch-01.json", digest) == b'{"metadata_only": true}\n'

    # An arbitrary ``train42`` look-alike cannot become an identity-helper
    # fallback, even if the supplied seed43 module otherwise has the expected
    # public artifact aliases.
    with monkeypatch.context() as scoped:
        scoped.setattr(train_43, "train42", SimpleNamespace())
        with pytest.raises(route.FailClosedError, match="artifact root identity drift"):
            route._attach_readonly_seed43_artifact(root, train_43)

    # A canonical-name symlink is rejected before an ArtifactRoot exists.
    aliased_root = tmp_path / "symlink-repository"
    aliased_directory = aliased_root / route.SEED43_TRAIN_ROOT_RELATIVE
    aliased_directory.parent.mkdir(parents=True)
    backing = tmp_path / "backing-training-root"
    backing.mkdir()
    os.symlink(backing, aliased_directory, target_is_directory=True)
    with pytest.raises(route.FailClosedError, match="absent or aliased"):
        route._attach_readonly_seed43_artifact(aliased_root, train_43)

    # The returned capability retains the exact named inode.  Replacing the
    # directory after attachment cannot redirect a metadata reload.
    displaced = directory.with_name("displaced-training-root")
    directory.rename(displaced)
    directory.mkdir()
    with pytest.raises(RuntimeError, match="identity drift"):
        artifact.reload_pair("epoch-01.json", digest)


def test_live_seed43_epoch01_metadata_attaches_and_validates_with_production_identity() -> None:
    """Read only the immutable epoch-01 receipt; never resolve data or tensors."""
    from src.tfsr_b3st4_ddrop_seed43_v1 import train_43

    root = Path(__file__).resolve().parents[2]
    live_root = root / route.SEED43_TRAIN_ROOT_RELATIVE
    assert live_root.is_dir() and not live_root.is_symlink()
    artifact = route._attach_readonly_seed43_artifact(root, train_43)
    epoch = artifact.reload_json("epoch-01.json")
    identity = train_43.production_identity(root)
    train_43.validate_epoch_receipt(
        epoch,
        1,
        2 * train_43.PUBLIC_SPEC.steps_per_epoch,
        train_43.PUBLIC_SPEC,
        identity,
    )


def test_score_cli_live_bootstrap_repairs_parent_only_lane_import_for_real_metadata() -> None:
    """The live CLI, not ambient PYTHONPATH, must expose ``tfpd_lane``.

    This isolated interpreter starts in the former parent-only environment.
    It first proves the historical schedule validation failure, then invokes
    the actual score CLI bootstrap and validates the immutable epoch-01
    metadata.  It never opens an NWB, deserializes a checkpoint, or exposes a
    CUDA device.
    """
    root = Path(__file__).resolve().parents[2]
    cli = root / "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed43_score.py"
    parent = root / "tfpd_exploration"
    lane = root / "tfpd_exploration/src"
    probe = r'''
import importlib.util
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
parent = str(root / "tfpd_exploration")
lane = str(root / "tfpd_exploration/src")
# Recreate the former score-CLI import environment: ``src`` resolves but the
# top-level lane package does not.  A clean subprocess prevents an earlier
# test's sys.modules cache from masking the defect.
sys.path[:] = [parent] + [entry for entry in sys.path if entry not in {parent, lane}]
from src.tfsr_b3st4_ddrop_seed43_v1 import train_43
try:
    train_43.lr_for_step(train_43.PUBLIC_SPEC.steps_per_epoch)
except ModuleNotFoundError as error:
    assert error.name == "tfpd_lane", error
else:
    raise AssertionError("parent-only environment unexpectedly resolved tfpd_lane")

cli_path = root / "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed43_score.py"
spec = importlib.util.spec_from_file_location("seed43_score_cli_probe", cli_path)
assert spec is not None and spec.loader is not None
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
cli._bootstrap_route_import_paths(live=True)
assert sys.path[:2] == [lane, parent]
assert train_43.lr_for_step(train_43.PUBLIC_SPEC.steps_per_epoch) > 0.0

from src.tfsr_b3st4_ddrop_seed43_v1 import score_43 as route
artifact = route._attach_readonly_seed43_artifact(root, train_43)
epoch = artifact.reload_json("epoch-01.json")
identity = train_43.production_identity(root)
train_43.validate_epoch_receipt(
    epoch, 1, 2 * train_43.PUBLIC_SPEC.steps_per_epoch,
    train_43.PUBLIC_SPEC, identity,
)
print("PARENT_ONLY_FAIL_THEN_CLI_BOOTSTRAP_METADATA_PASS")
'''
    env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": str(parent),
    }
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(root)], cwd=root, env=env,
        text=True, capture_output=True, check=True,
    )
    assert completed.stdout.strip() == "PARENT_ONLY_FAIL_THEN_CLI_BOOTSTRAP_METADATA_PASS"


def test_addendum_closure_binds_live_epoch_schedule_authority() -> None:
    """The closure must bind the direct ``train_43.lr_for_step`` import."""
    relative = "tfpd_exploration/src/tfpd_lane/arm_common.py"
    root = Path(__file__).resolve().parents[2]
    closure = route.addendum_closure(root)
    assert relative in closure["paths"]
    assert closure["sha256_by_path"][relative] == route._sha((root / relative).read_bytes())
    assert route.validate_addendum_closure(closure) == closure


def test_upstream_seed42_validator_is_mandatory_and_uses_frozen_validator() -> None:
    calls: list[str] = []

    class Identity:
        def __init__(self, **kwargs: object) -> None:
            calls.append("identity")
            self.kwargs = kwargs

    fake_s42 = SimpleNamespace(
        ScoreIdentity=Identity,
        validate_input_authority_payload=lambda payload, identity: calls.append("input"),
        validate_score_payload=lambda payload, **kwargs: calls.append("score"),
        validate_terminal_payload=lambda payload, identity, score, **kwargs: calls.append("terminal"),
    )
    terminal = {
        "identity": {
            "fixed_authorities": {"x": {}}, "training_terminal_sha256": SHA_A,
            "training_swa_sha256": SHA_B, "training_swa_state_digest": SHA_C,
            "launch_closure": {}, "phase_e_authorization": {},
        },
        "verdict": "HOLD",
    }
    input_payload: dict[str, object] = {}
    score_payload: dict[str, object] = {}
    upstream = route.validate_upstream_seed42_payloads(
        fake_s42, input_payload=input_payload, input_sha256=_body_sha(input_payload),
        score_payload=score_payload, score_sha256=_body_sha(score_payload),
        terminal_payload=terminal, terminal_sha256=_body_sha(terminal),
    )
    assert upstream.verdict == "HOLD"
    assert calls == ["identity", "input", "score", "terminal"]
    for field, valid_sha in (
        ("input", _body_sha(input_payload)),
        ("score", _body_sha(score_payload)),
        ("terminal", _body_sha(terminal)),
    ):
        kwargs = {
            "input_sha256": _body_sha(input_payload),
            "score_sha256": _body_sha(score_payload),
            "terminal_sha256": _body_sha(terminal),
        }
        kwargs[f"{field}_sha256"] = SHA_A if valid_sha != SHA_A else SHA_B
        with pytest.raises(route.FailClosedError, match="descriptor payload/body SHA"):
            route.validate_upstream_seed42_payloads(
                fake_s42, input_payload=input_payload, score_payload=score_payload,
                terminal_payload=terminal, **kwargs,
            )
    terminal["verdict"] = "UNKNOWN"
    with pytest.raises(route.FailClosedError, match="verdict"):
        route.validate_upstream_seed42_payloads(
            fake_s42, input_payload=input_payload, input_sha256=_body_sha(input_payload),
            score_payload=score_payload, score_sha256=_body_sha(score_payload),
            terminal_payload=terminal, terminal_sha256=_body_sha(terminal),
        )


def test_descriptor_to_typed_evidence_rejects_each_forged_body_sha_before_schema_reuse() -> None:
    """A 64-hex string is not evidence unless it hashes the exact body."""
    upstream = _upstream()
    for field in ("input_authority_sha256", "score_sha256", "terminal_sha256"):
        values = {
            "input_authority_sha256": upstream.input_authority_sha256,
            "score_sha256": upstream.score_sha256,
            "terminal_sha256": upstream.terminal_sha256,
        }
        values[field] = SHA_A if values[field] != SHA_A else SHA_B
        with pytest.raises(ValueError, match="body SHA"):
            route.UpstreamSeed42Evidence(
                **values, verdict=upstream.verdict, input_payload=upstream.input_payload,
                score_payload=upstream.score_payload, terminal_payload=upstream.terminal_payload,
            )
    score = {"cell": route.CELL, "status": "REPLICATION_SCORE_COMPLETE", "build": route.BUILD_LABEL}
    terminal = {"cell": route.CELL, "status": "REPLICATION_SCORE_COMPLETE"}
    with pytest.raises(ValueError, match="body SHA"):
        route.Seed43CompletedScoreEvidence(
            score_sha256=SHA_A, terminal_sha256=_body_sha(terminal), score_payload=score, terminal_payload=terminal,
        )

    # The explicit validator repeats the same check before it can call an
    # expensive/reused schema validator or reach a report path.
    identity, _capability = _identity(_training(), upstream)
    with pytest.raises(route.FailClosedError, match="descriptor payload/body SHA"):
        route.validate_seed43_completed_score_payloads(
            score_payload=score, score_sha256=SHA_A, terminal_payload=terminal,
            terminal_sha256=_body_sha(terminal), identity=identity,
        )


def test_parity_manifest_requires_every_scored_forward_and_rejects_inequality() -> None:
    flags = route.Score43Flags()
    manifest = route.AcceleratedParityManifest()
    for surface in ("within", "external"):
        for mode in ("aligned", "zero", "wrong_pair"):
            flags.record_forward("tfsr", surface, mode)
            manifest.record(surface=surface, mode=mode, eager_bytes=b"same", accelerated_bytes=b"same", shape=(1, 50, 2))
    payload = manifest.payload(flags)
    assert route.validate_parity_manifest(payload, flags) == payload
    with pytest.raises(route.FailClosedError, match="inequality"):
        manifest.record(surface="within", mode="aligned", eager_bytes=b"eager", accelerated_bytes=b"accelerated", shape=(1, 50, 2))
    missing = dict(payload)
    missing["events"] = missing["events"][:-1]
    missing["events_sha256"] = route._sha(route._json_bytes(missing["events"]))
    with pytest.raises(route.FailClosedError, match="misses|counter"):
        route.validate_parity_manifest(missing, flags)


def test_upstream_hold_stop_is_non_governing_and_cannot_be_rewritten() -> None:
    assert route._replication_role("CLEAR_GO") == "CONFIRMATORY_REPLICATION"
    assert route._replication_role("HOLD") == "NON_GOVERNING_POST_AUTHORIZATION_REPLICATION"
    assert route._replication_role("STOP") == "NON_GOVERNING_POST_AUTHORIZATION_REPLICATION"
    with pytest.raises(route.FailClosedError):
        route._replication_role("CLEAR_GO_REWRITTEN")


def test_two_seed_summary_is_receipt_bound_descriptive_and_rejects_build_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tfsr_b3st4_ddrop_v1 import score as s42

    within = tuple(f"within-{index}" for index in range(6))
    external = tuple(f"external-{index:02d}" for index in range(15))

    def aligned_map(value: float) -> dict[str, dict[str, object]]:
        return {
            surface: {
                "aligned": _synthetic_mode_evidence(
                    s42, system="tfsr", surface=surface, mode="aligned",
                    sessions=(within if surface == "within" else external), governing_r2=value,
                    input_authority_sha256=SHA_A,
                ).payload(),
            }
            for surface in ("within", "external")
        }

    upstream_score = {"tfsr": aligned_map(0.20)}
    upstream_terminal = {
        "identity": {
            "fixed_authorities": {}, "training_terminal_sha256": SHA_A,
            "training_swa_sha256": SHA_B, "training_swa_state_digest": SHA_C,
            "launch_closure": {}, "phase_e_authorization": {},
        },
        "verdict": "CLEAR_GO",
    }
    upstream = route.UpstreamSeed42Evidence(
        input_authority_sha256=_body_sha(_upstream().input_payload), score_sha256=_body_sha(upstream_score),
        terminal_sha256=_body_sha(upstream_terminal), verdict="CLEAR_GO", input_payload=_upstream().input_payload,
        score_payload=upstream_score, terminal_payload=upstream_terminal,
    )
    identity, _capability = _identity(_training(), upstream)
    seed43_score = {
        "cell": route.CELL, "status": "REPLICATION_SCORE_COMPLETE", "build": route.BUILD_LABEL,
        "input_replay_sha256": SHA_D, "tfsr": aligned_map(0.30),
    }
    seed43_terminal = {"cell": route.CELL, "status": "REPLICATION_SCORE_COMPLETE"}
    seed43 = route.Seed43CompletedScoreEvidence(
        score_sha256=_body_sha(seed43_score), terminal_sha256=_body_sha(seed43_terminal), score_payload=seed43_score,
        terminal_payload=seed43_terminal,
    )
    class FrozenValidatorProxy:
        ScoreIdentity = type("Identity", (), {"__init__": lambda self, **kwargs: None})

        @staticmethod
        def validate_input_authority_payload(*_args: object, **_kwargs: object) -> None:
            return None

        @staticmethod
        def validate_score_payload(*_args: object, **_kwargs: object) -> None:
            return None

        @staticmethod
        def validate_terminal_payload(*_args: object, **_kwargs: object) -> None:
            return None

        @staticmethod
        def mode_evidence_from_payload(value: object) -> Any:
            return s42.mode_evidence_from_payload(value)

    # Retain both new body-digest checks at the actual descriptor-to-typed
    # boundary.  Only the frozen schemas are minimized here because this test
    # intentionally supplies a compact, no-data score fixture.
    monkeypatch.setattr(route, "_seed42_score", lambda: FrozenValidatorProxy())
    monkeypatch.setattr(route, "validate_score_payload", lambda value, **_kwargs: dict(value))
    monkeypatch.setattr(route, "validate_terminal_payload", lambda value, **_kwargs: dict(value))
    summary = route.two_seed_descriptive_summary(upstream=upstream, seed43=seed43, identity=identity)
    assert summary["seeds"] == [42, 43]
    assert summary["individual_receipts"]["seed42"] == {
        "build_label": route.SEED42_BUILD_LABEL,
        "score_sha256": _body_sha(upstream_score), "terminal_sha256": _body_sha(upstream_terminal),
    }
    assert summary["individual_receipts"]["seed43"] == {
        "seed": 43, "build_label": route.BUILD_LABEL,
        "score_sha256": _body_sha(seed43_score), "terminal_sha256": _body_sha(seed43_terminal),
    }
    external_means = summary["aligned_equal_session_means"]["external"]
    assert external_means["seed42_aligned_equal_session_mean"] == pytest.approx(0.2)
    assert external_means["seed43_aligned_equal_session_mean"] == pytest.approx(0.3)
    assert external_means["two_seed_descriptive_equal_weight_mean"] == pytest.approx(0.25)
    assert summary["description_only"] is True
    assert summary["three_seed_superiority_claim"] is False
    bad_score = dict(seed43_score)
    bad_score["build"] = "hidden-or-mismatched"
    with pytest.raises(ValueError, match="build"):
        route.Seed43CompletedScoreEvidence(
            score_sha256=_body_sha(bad_score), terminal_sha256=_body_sha(seed43_terminal), score_payload=bad_score,
            terminal_payload=seed43_terminal,
        )


def test_seed43_roots_are_distinct_from_seed42_without_reservation() -> None:
    from src.tfsr_b3st4_ddrop_v1 import score as s42

    assert route.SEED43_AUTHORITY_ROOT_RELATIVE != s42.AUTHORITY_ROOT_RELATIVE
    assert route.SEED43_SCORE_ROOT_RELATIVE != s42.SCORE_ROOT_RELATIVE
    assert route.SEED43_AUTHORITY_ROOT_RELATIVE != route.SEED43_SCORE_ROOT_RELATIVE
    route._require_seed43_output_names()


def test_authorization_rebuilds_current_closure_before_any_backend_or_asset_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale-but-self-consistent durable closure must fail pre-data."""
    training, upstream = _training(), _upstream()
    durable = {"paths": ["old.py"], "sha256_by_path": {"old.py": SHA_A}, "closure_sha256": SHA_B}
    preflight = {"addendum_closure": durable}
    authorization = {"addendum_closure": durable}
    reads: list[str] = []

    def read_pair(_root: Path, name: str) -> tuple[dict[str, object], str]:
        reads.append(name)
        return (preflight if name == "official_preflight.json" else authorization), (SHA_A if name == "official_preflight.json" else SHA_B)

    monkeypatch.setattr(route, "_read_seed43_authority_pair", read_pair)
    monkeypatch.setattr(route, "validate_target_free_preflight", lambda value, **_: dict(value))
    monkeypatch.setattr(route, "validate_root_authorization", lambda value, **_: dict(value))
    monkeypatch.setattr(
        route, "addendum_closure",
        lambda _root: {"paths": ["current.py"], "sha256_by_path": {"current.py": SHA_C}, "closure_sha256": SHA_D},
    )
    with pytest.raises(route.FailClosedError, match="current implementation before backend"):
        route.verify_seed43_authorization(Path.cwd(), training=training, upstream=upstream)
    assert reads == ["official_preflight.json", "root_authorization.json"]


def test_physical_backend_is_a_subclass_without_global_mutation_or_cell_d() -> None:
    from src.tfsr_b3st4_ddrop_v1 import score as s42

    fixed = {item.name: object() for item in s42.FIXED_AUTHORITIES}
    upstream_training = s42.TrainingEvidence(
        terminal_sha256=SHA_A, swa_sha256=SHA_B, swa_state_digest=SHA_C,
        closure={}, checkpoint_sha256={"44": SHA_A, "45": SHA_B, "46": SHA_C, "47": SHA_D},
        training_peak_memory_bytes=0,
    )
    upstream_authorization = s42.PhaseEAuthorization(
        preflight_sha256=SHA_A, root_authorization_sha256=SHA_B,
        preflight={}, root_authorization={}, expected_closure={},
    )
    frozen_device = dict(s42.FROZEN_SCORE_DEVICE)
    backend = route.build_seed43_physical_backend(
        root=Path.cwd(), fixed_authorities=fixed, upstream_training=upstream_training,
        upstream_authorization=upstream_authorization, seed43_training=_training(),
    )
    assert isinstance(backend, s42.PhysicalMatchedScoreBackend)
    # The addendum inherits the audited single-model scoring core verbatim.
    # It does not wrap/copy score_tfsr or _score_system_surface merely to hide
    # a seed-42 model binding; only the seed43 SWA loader/forward is replaced.
    assert type(backend).score_tfsr is s42.PhysicalMatchedScoreBackend.score_tfsr
    assert type(backend)._score_system_surface is s42.PhysicalMatchedScoreBackend._score_system_surface
    assert backend._cell_d is None and backend._tfsr is None
    assert dict(s42.FROZEN_SCORE_DEVICE) == frozen_device
    with pytest.raises(route.FailClosedError, match="never rerun Cell-D"):
        backend.score_cell_d()


def test_actual_dynamic_subclass_forward_enforces_eager_accelerated_parity_before_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the actual nested override, not only the manifest helper."""
    import torch

    class ParentBackend:
        def __init__(self, **_: object) -> None:
            self._runtime: dict[str, object] = {}
            self._cell_d = None
            self._tfsr = None
            self._state_at_load: dict[str, str] = {}

        def _load_runtime(self) -> dict[str, object]:
            return self._runtime

        @staticmethod
        def _all_gradients_none(_model: object, _torch: object) -> bool:
            return True

        @staticmethod
        def _assert_tfsr_eval_no_mask(_model: object, _torch: object) -> None:
            return None

        def _forward_tfsr(self, _model: object, neural: object, _calib: object, _capability: object, *,
                          surface: str, mode: str, flags: route.Score43Flags,
                          measure_latency: bool = False) -> object:
            assert measure_latency is False
            flags.record_forward("tfsr", surface, mode)
            return neural

    fake_s42 = SimpleNamespace(
        PhysicalMatchedScoreBackend=ParentBackend,
        _physical_model_state_digest=lambda *_: SHA_A,
        FROZEN_SCORE_DEVICE={},
    )
    monkeypatch.setattr(route, "_seed42_score", lambda: fake_s42)
    backend = route.build_seed43_physical_backend(
        root=Path.cwd(), fixed_authorities={}, upstream_training=object(), upstream_authorization=object(),
        seed43_training=_training(),
    )
    backend._runtime = {"torch": torch}
    backend._scripted_step = object()
    backend._accelerated_eval = lambda _torch, _model, _step, neural, _calib, _capability: neural.clone()
    flags = route.Score43Flags()
    neural = torch.zeros((2, 50, 2), dtype=torch.float32)
    output = backend._forward_tfsr(
        object(), neural, object(), object(), surface="within", mode="aligned", flags=flags,
    )
    assert torch.equal(output, neural)
    assert len(backend._parity_manifest.events) == 1
    assert flags.forward_calls["within"]["aligned"] == 1

    unequal = route.build_seed43_physical_backend(
        root=Path.cwd(), fixed_authorities={}, upstream_training=object(), upstream_authorization=object(),
        seed43_training=_training(),
    )
    unequal._runtime = {"torch": torch}
    unequal._scripted_step = object()
    unequal._accelerated_eval = lambda _torch, _model, _step, neural, _calib, _capability: neural + 1.0
    with pytest.raises(route.FailClosedError, match="inequality before metric"):
        unequal._forward_tfsr(
            object(), neural, object(), object(), surface="within", mode="aligned", flags=route.Score43Flags(),
        )
    assert unequal._parity_manifest.events == []


def test_input_replay_requires_exact_upstream_records_targets_masks_and_normalizers() -> None:
    upstream = _upstream()
    flags = route.Score43Flags()
    observed = dict(route._upstream_input_evidence_payload(upstream))
    evidence = SimpleNamespace(payload=lambda: observed)
    replay = route._validate_input_replay(evidence, upstream=upstream, flags=flags, s42=object())
    assert route.validate_input_replay_payload(replay, upstream=upstream) == replay
    observed["records"] = [{"target_last_bin_sha256": SHA_D}]
    with pytest.raises(route.FailClosedError, match="differs"):
        route._validate_input_replay(evidence, upstream=upstream, flags=flags, s42=object())


def test_public_execution_capability_fails_before_any_validator_or_target_route() -> None:
    calls: list[str] = []

    def unexpected(_: Path) -> Any:
        calls.append("called")
        raise AssertionError("must fail before target/training/authority resolution")

    with pytest.raises(route.FailClosedError, match="capability"):
        route.execute_authorized(Path.cwd(), capability=None, seed43_training_validator=unexpected, upstream_loader=unexpected)
    assert calls == []


def test_failure_publication_is_terminal_exclusive_without_data_or_cuda(tmp_path: Path) -> None:
    from src.tfsr_b3st4_ddrop_v1 import score as s42

    training = _training()
    upstream = _upstream()
    identity, capability = _identity(training, upstream)
    artifact = s42.reserve_artifact_root(tmp_path, "replication-attempt", route.SCORE_TOPOLOGY)

    class FailingBackend:
        def resolve_inputs(self, **_: object) -> object:
            raise route.FailClosedError("synthetic source adapter failure")

        def score_tfsr(self, **_: object) -> object:
            raise AssertionError("unreachable")

        def build_parity_manifest(self, **_: object) -> dict[str, object]:
            raise AssertionError("unreachable")

        def reverify_after_forwards(self, **_: object) -> None:
            raise AssertionError("unreachable")

        def resource_disclosure(self) -> dict[str, object]:
            raise AssertionError("unreachable")

        def close(self) -> None:
            return None

    with pytest.raises(route.FailClosedError, match="synthetic source"):
        route.run_addendum_lifecycle(
            artifact=artifact, identity=identity, execution_capability=capability, upstream=upstream,
            within_roster=(), external_sessions=(), within_assets_factory=lambda: (), external_assets_factory=lambda: (),
            backend=FailingBackend(), final_reverify=lambda: identity.closure,
        )
    assert artifact.has_name("attempt.json")
    assert artifact.has_name("failure.json")
    assert not artifact.has_name("terminal.json")
    failure_body = artifact.reload_pair("failure.json")
    route.validate_failure_payload(
        artifact.reload_json("failure.json", route._sha(failure_body)), identity=identity,
        attempt_sha256=route._sha(artifact.reload_pair("attempt.json")), input_replay_sha256=None,
    )


def test_failure_after_durable_input_replay_binds_identity_attempt_and_replay(tmp_path: Path) -> None:
    """A later failure cannot erase or ambiguously describe its input replay."""
    from src.tfsr_b3st4_ddrop_v1 import score as s42

    identity, _capability = _identity(_training(), _upstream())
    artifact = s42.reserve_artifact_root(tmp_path, "failure-after-replay", route.SCORE_TOPOLOGY)
    attempt_sha = artifact.publish_json("attempt.json", route._attempt_payload(identity))
    replay_sha = artifact.publish_json("input_replay.json", {"synthetic": True})
    flags = route.Score43Flags(stage="after_input_replay")
    route._publish_failure(
        artifact, flags, identity=identity, attempt_sha256=attempt_sha, input_replay_sha256=replay_sha,
    )
    failure = artifact.reload_json("failure.json")
    route.validate_failure_payload(
        failure, identity=identity, attempt_sha256=attempt_sha, input_replay_sha256=replay_sha,
    )
    assert failure["identity"] == identity.payload()
    assert failure["attempt_sha256"] == attempt_sha
    assert failure["input_replay_sha256"] == replay_sha
    tampered = dict(failure)
    tampered["input_replay_sha256"] = None
    with pytest.raises(route.FailClosedError, match="schema drift"):
        route.validate_failure_payload(
            tampered, identity=identity, attempt_sha256=attempt_sha, input_replay_sha256=replay_sha,
        )


def _synthetic_mode_evidence(
    s42: Any,
    *,
    system: str,
    surface: str,
    mode: str,
    sessions: tuple[str, ...],
    governing_r2: float,
    input_authority_sha256: str,
) -> Any:
    rows = tuple(
        s42.SessionScore(
            session=session, n_windows=3, governing_r2=governing_r2,
            full_window_r2=governing_r2, output_sha256=SHA_A,
            input_authority_sha256=input_authority_sha256,
        )
        for session in sessions
    )
    control: dict[str, object] | None = None
    if system == "tfsr":
        control = {"mode": mode, "post_normalization": True, "b3s_recomputed": True}
        if mode == "zero":
            control["exact_zeros_like"] = True
        if mode == "wrong_pair":
            control.update({"permutation_is_derangement": True, "permutation_sha256": SHA_B})
    return s42.ModeEvidence(
        system=system, surface=surface, mode=mode, sessions=rows,
        state_before_sha256=SHA_C, state_after_sha256=SHA_C,
        eval_mode=True, gradients_none=True, finite_output=True, output_shape=(1, 50, 2),
        repeat_bitwise_equal=(True if system == "tfsr" and mode == "aligned" else None),
        eval_no_mask=(True if system == "tfsr" else None),
        capture_diagnostics=(False if system == "tfsr" else None),
        b3s_recomputed=(True if system == "tfsr" else None),
        t4_control=control, latency_ms=0.0, peak_memory_bytes=0,
    )


def test_complete_synthetic_lifecycle_publishes_six_tfsr_cells_with_replay_and_atomic_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The addendum lifecycle runs no Cell-D call even on its success path."""
    from src.tfsr_b3st4_ddrop_v1 import score as real_s42

    within = tuple(f"within-{index}" for index in range(6))
    external = tuple(f"external-{index:02d}" for index in range(15))
    input_sha = SHA_D

    def tfsr_map(value: float) -> dict[str, dict[str, object]]:
        return {
            surface: {
                mode: _synthetic_mode_evidence(
                    real_s42, system="tfsr", surface=surface, mode=mode,
                    sessions=(within if surface == "within" else external), governing_r2=value,
                    input_authority_sha256=input_sha,
                ).payload()
                for mode in ("aligned", "zero", "wrong_pair")
            }
            for surface in ("within", "external")
        }

    upstream_score = {
        "cell_d": {
            surface: _synthetic_mode_evidence(
                real_s42, system="cell_d", surface=surface, mode="aligned",
                sessions=(within if surface == "within" else external), governing_r2=0.10,
                input_authority_sha256=input_sha,
            ).payload()
            for surface in ("within", "external")
        },
        "tfsr": tfsr_map(0.20),
        "a2_contextual": {
            "pooled_per_session": {
                "within": {session: 0.05 for session in within},
                "external": {session: 0.05 for session in external},
            },
        },
    }
    upstream_terminal = {"identity": {}}
    upstream_input = _upstream().input_payload
    upstream = route.UpstreamSeed42Evidence(
        input_authority_sha256=_body_sha(upstream_input), score_sha256=_body_sha(upstream_score),
        terminal_sha256=_body_sha(upstream_terminal), verdict="CLEAR_GO", input_payload=upstream_input,
        score_payload=upstream_score, terminal_payload=upstream_terminal,
    )
    training = _training()
    identity, capability = _identity(training, upstream)

    class S42Proxy:
        PUBLIC_SPEC = real_s42.PUBLIC_SPEC
        FROZEN_SCORE_DEVICE = real_s42.FROZEN_SCORE_DEVICE

        def __getattr__(self, name: str) -> Any:
            return getattr(real_s42, name)

        @staticmethod
        def validate_input_authority_evidence(_evidence: object, **_: object) -> None:
            # The input replay equality itself is tested below.  Constructing
            # a real 21-session input authority would require parser data,
            # which this lifecycle unit test must never access.
            return None

    proxy = S42Proxy()
    monkeypatch.setattr(route, "_seed42_score", lambda: proxy)
    monkeypatch.setattr(route, "validate_addendum_closure", lambda value, **_: dict(value))
    artifact = real_s42.reserve_artifact_root(tmp_path, "complete-addendum", route.SCORE_TOPOLOGY)

    class SuccessfulBackend:
        def __init__(self) -> None:
            self.manifest = route.AcceleratedParityManifest()
            self.closed = False
            self.cell_d_calls = 0

        def resolve_inputs(self, *, flags: route.Score43Flags, **_: object) -> object:
            flags.within_opened = True
            flags.external_opened = True
            return SimpleNamespace(payload=lambda: route._upstream_input_evidence_payload(upstream))

        def score_tfsr(self, *, surface: str, mode: str, input_authority_sha256: str,
                       flags: route.Score43Flags) -> Any:
            flags.record_forward("tfsr", surface, mode)
            self.manifest.record(
                surface=surface, mode=mode, eager_bytes=b"eager", accelerated_bytes=b"eager",
                shape=(1, 50, 2),
            )
            return _synthetic_mode_evidence(
                real_s42, system="tfsr", surface=surface, mode=mode,
                sessions=(within if surface == "within" else external), governing_r2=0.20,
                input_authority_sha256=input_authority_sha256,
            )

        def build_parity_manifest(self, *, flags: route.Score43Flags) -> dict[str, object]:
            return self.manifest.payload(flags)

        @staticmethod
        def reverify_after_forwards(*, flags: route.Score43Flags) -> None:
            assert flags.backward_calls == flags.optimizer_calls == 0

        @staticmethod
        def resource_disclosure() -> dict[str, object]:
            return {
                "schema": "tfsr_seed43_phase_e_replication_resources_v1",
                "runtime": {**dict(real_s42.FROZEN_SCORE_DEVICE), "torchmetrics_version": "1.5.1"},
                "seed43_training_peak_memory_bytes": 0,
                "score_peak_memory_bytes": 0,
                "aligned_eager_latency_ms": 0.0,
                "tfsr_parameters": 1,
                "build": route.BUILD_LABEL,
                "cell_d_replayed_upstream_only": True,
                "cell_d_forward_calls": 0,
                "forward_only": True,
                "no_grad": True,
            }

        def close(self) -> None:
            self.closed = True

    backend = SuccessfulBackend()
    terminal = route.run_addendum_lifecycle(
        artifact=artifact, identity=identity, execution_capability=capability, upstream=upstream,
        within_roster=within, external_sessions=external,
        within_assets_factory=lambda: tuple(SimpleNamespace(session=item) for item in within),
        external_assets_factory=lambda: tuple(SimpleNamespace(session=item) for item in external),
        backend=backend, final_reverify=lambda: identity.closure,
    )
    score_body = artifact.reload_pair("score.json")
    terminal_body = artifact.reload_pair("terminal.json")
    score = artifact.reload_json("score.json", route._sha(score_body))
    actual_input_sha = score["input_replay_sha256"]
    assert isinstance(actual_input_sha, str)
    route.validate_score_payload(score, identity=identity, input_replay_sha256=actual_input_sha)
    route.validate_terminal_payload(
        terminal, identity=identity, score_payload=score, expected_score_sha256=route._sha(score_body),
        expected_input_replay_sha256=actual_input_sha,
    )
    completed = route.validate_seed43_completed_score_payloads(
        score_payload=score, score_sha256=route._sha(score_body), terminal_payload=terminal,
        terminal_sha256=route._sha(terminal_body), identity=identity,
    )
    assert completed.binding_payload()["build_label"] == route.BUILD_LABEL
    with pytest.raises(route.FailClosedError, match="descriptor payload/body SHA"):
        route.validate_seed43_completed_score_payloads(
            score_payload=score, score_sha256=SHA_A, terminal_payload=terminal,
            terminal_sha256=route._sha(terminal_body), identity=identity,
        )
    assert route._sha(terminal_body) == route._sha(route._json_bytes(terminal))
    assert not artifact.has_name("failure.json")
    assert backend.closed is True and backend.cell_d_calls == 0
    assert len(score["accelerated_eager_parity"]["events"]) == 6
    assert score["boundaries"]["tfsr_forward_calls"] == {
        "within": {"aligned": 1, "zero": 1, "wrong_pair": 1},
        "external": {"aligned": 1, "zero": 1, "wrong_pair": 1},
    }


def test_static_cli_is_torch_free_and_public_flags_fail_before_authority(tmp_path: Path) -> None:
    cli = Path("tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed43_score.py").resolve()
    guard = r'''
import builtins, runpy, sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("torch import forbidden on static CLI path")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
path = sys.argv[1]
sys.argv = [path]
runpy.run_path(path, run_name="__main__")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
'''
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(
        [sys.executable, "-c", guard, str(cli)], cwd=Path.cwd(), env=env,
        text=True, capture_output=True, check=True,
    )
    assert "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_SCORE" in completed.stdout
    public = subprocess.run(
        [sys.executable, str(cli), "--execute", "--i-have-seed43-phase-e-replication-authorization"],
        cwd=Path.cwd(), env=env, text=True, capture_output=True,
    )
    assert public.returncode != 0
    assert "capability required before authority/data/model resolution" in public.stderr
