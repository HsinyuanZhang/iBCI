"""Focused no-data/no-CUDA tests for the deferred Posterior Carrier full route."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Mapping

import pytest
import torch

from src.posterior_carrier_v1 import (
    full_train,
    phase_b,
    phase_b_v2,
    phase_b_v3,
    source_adapter_v2,
)


ROOT = Path(__file__).resolve().parents[2]
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _source_root_payload() -> dict[str, object]:
    return {
        "schema": "posterior_carrier_strict27_source_data_root_v1",
        "source_data_root": phase_b.CANONICAL_SOURCE_DATA_ROOT,
        "directory_device": 1,
        "directory_inode": 2,
        "nwb_copied_into_stage": False,
        "nwb_symlink_or_bind_mount_authorized": False,
    }


def _identity(
    *,
    strict_source_metadata_sha256: str = SHA_A,
    roster: list[str] | None = None,
) -> phase_b_v3.RunIdentityV3:
    roster = [f"source-{index:02d}" for index in range(27)] if roster is None else list(roster)
    source = {
        "schema": "posterior_carrier_target_free_source_identity_v1",
        "roster": roster,
        "roster_sha256": phase_b.sha256_bytes(phase_b.canonical_json_bytes(roster)),
        "strict_source_metadata_sha256": strict_source_metadata_sha256,
        "manifest_sha256": SHA_A,
        "ordinary_raw_t4_semantic_sha256": SHA_A,
        "behavior_normalizer_semantic_sha256": SHA_A,
        "source_lineage_sha256": SHA_A,
        "source_data_root": _source_root_payload(),
        "source_only": True,
        "target_opened": False,
        "within_opened": False,
        "external_opened": False,
        "formal_opened": False,
        "h1_opened": False,
    }
    v3 = phase_b_v3.phase_b_v3_closure(ROOT)
    assert v3["closure_sha256"] == full_train.ACCEPTED_PHASE_B_V3_CLOSURE_SHA256
    v2 = phase_b_v3._v2_closure_from_v3(v3)
    return phase_b_v3.RunIdentityV3(
        base_identity=phase_b_v2.RunIdentityV2(
            base_identity=phase_b.RunIdentity(
                source_authority=source,
                closure=phase_b_v2.base_v1_closure_from_v2(v2),
                remote_device=dict(phase_b.REMOTE_TORCH_AUTHORITY),
            ),
            closure=v2,
        ),
        closure=v3,
    )


def _lineage(smoke_identity: phase_b_v3.RunIdentityV3 | None = None) -> dict[str, object]:
    smoke_identity = _identity(strict_source_metadata_sha256=SHA_A) if smoke_identity is None else smoke_identity
    return {
        "root_relative": phase_b_v3.SOURCE_SMOKE_V3_ROOT_RELATIVE,
        "attempt_sha256": SHA_A,
        "launch_sha256": SHA_A,
        "source_authority_sha256": SHA_A,
        "step100_sha256": SHA_A,
        "terminal_sha256": SHA_A,
        "step100_status": "SOURCE_SMOKE_V3_100_STEPS_COMPLETE",
        "terminal_status": "SOURCE_SMOKE_V3_COMPLETE__NON_AUTHORITATIVE",
        "phase_b_v3_closure_sha256": full_train.ACCEPTED_PHASE_B_V3_CLOSURE_SHA256,
        "completed_smoke_identity_sha256": full_train._digest(
            full_train._json(smoke_identity.payload())
        ),
    }


def _full_identity() -> full_train.FullRunIdentity:
    completed_smoke = _identity(strict_source_metadata_sha256=SHA_A)
    return full_train.FullRunIdentity(
        # A freshly staged full route has new descriptor identities even when
        # all source authority bytes/science semantics are the same as the
        # immutable completed smoke stage.
        phase_b_v3_identity=_identity(strict_source_metadata_sha256=SHA_B),
        completed_smoke_v3_identity=completed_smoke,
        closure=full_train.full_training_closure(ROOT),
        source_smoke_lineage=_lineage(completed_smoke),
    )


def _source_authority(
    identity: full_train.FullRunIdentity,
    launch_sha256: str,
    spec: full_train.FullTrainingSpec,
) -> dict[str, object]:
    roster = list(identity.phase_b_v3_identity.base_identity.base_identity.source_authority["roster"])
    schedule = full_train._epoch_schedule_payload(roster=roster, epoch=0)
    complete = full_train.core.build_budget_schedule(epochs=48, session_count=27)
    return {
        "schema": "posterior_carrier_full_source_authority_v1",
        "cell": full_train.CELL,
        "phase": full_train.FULL_PHASE,
        "spec": spec.payload(),
        "identity": identity.payload(),
        "full_launch_sha256": launch_sha256,
        "phase_b_v3_source_authority": {"synthetic": "v3-authority"},
        "phase_b_v3_source_authority_sha256": full_train._digest(
            full_train._json({"synthetic": "v3-authority"})
        ),
        "schedule": {
            "roster": roster,
            "budgets": [4, 10, 30],
            "formula": "budgets[(epoch + session_index) % 3]",
            "epochs": 48,
            "session_count": 27,
            "epochs_per_budget_per_session": 16,
            "schedule_sha256": full_train.core.budget_schedule_digest(complete),
        },
        "remote_torch_authority": dict(phase_b.REMOTE_TORCH_AUTHORITY),
        "boundaries": full_train.source_only_boundaries(),
        "status": "STRICT27_POSTERIOR_SOURCE_AUTHORITY_READY",
    }


def _tiny_spec() -> full_train.FullTrainingSpec:
    return full_train.FullTrainingSpec(
        epochs=2, batch_size=32, steps_per_epoch=2,
        checkpoint_epochs=(0, 1), throughput_steps=1, public=False,
    )


def _valid_v3_smoke_summary() -> dict[str, object]:
    cache = {
        "posterior_fit_calls": 81,
        "posterior_inverse_calls": 81,
        "deterministic_mean_view_builds": 81,
        "epoch_sampled_view_builds": 27,
        "device_epoch_view_builds": 27,
        "normalized_view_builds": 108,
        "batch_loop_requests": 100,
        "batch_loop_inverse_calls": 0,
        "source_sessions": 27,
        "scheduled_session_epochs": 27,
    }
    return phase_b.SmokeStepSummary(
        loss_first=1.0, loss_last=0.5, loss_min=0.5, loss_max=1.0,
        nonincreasing_transitions=99, losses_count=100,
        critical_gradients={key: True for key in full_train.CRITICAL_GRADIENT_KEYS},
        finite_model=True, finite_adam=True, model_state_sha256=SHA_A,
        optimizer_state_sha256=SHA_B, posterior_prepare_seconds=0.0,
        optimizer_core_seconds=1.0, optimizer_steps_per_second=100.0,
        peak_allocated_bytes=4, peak_reserved_bytes=5,
        current_allocated_bytes=2, current_reserved_bytes=3,
        posterior_cache=cache, dropout=dict(full_train.DROP_OUT_CONTRACT),
        remote_device=dict(phase_b.REMOTE_TORCH_AUTHORITY),
    ).payload()


def _reserve_temporary_artifact(spec: full_train.FullTrainingSpec) -> tuple[tempfile.TemporaryDirectory[str], full_train.ArtifactRoot]:
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    return temporary, full_train.reserve_full_train_root(root, spec=spec, relative="full-output")


def test_public_spec_and_closure_bind_the_accepted_v3_smoke_code_exactly() -> None:
    assert full_train.PUBLIC_SPEC.total_steps == 1_628_400
    assert full_train.PUBLIC_SPEC.checkpoint_epochs == (44, 45, 46, 47)
    assert full_train.FULL_CLOSURE_PATHS[:len(phase_b_v3.PHASE_B_V3_CLOSURE_PATHS)] == phase_b_v3.PHASE_B_V3_CLOSURE_PATHS
    closure = full_train.full_training_closure(ROOT)
    assert closure["sha256_by_path"]["tfpd_exploration/src/posterior_carrier_v1/phase_b_v3.py"] == (
        phase_b_v3.phase_b_v3_closure(ROOT)["sha256_by_path"]["tfpd_exploration/src/posterior_carrier_v1/phase_b_v3.py"]
    )
    forged = copy.deepcopy(closure)
    forged["sha256_by_path"]["tfpd_exploration/src/posterior_carrier_v1/phase_b_v3.py"] = SHA_A
    body = {"paths": forged["paths"], "sha256_by_path": forged["sha256_by_path"]}
    forged["closure_sha256"] = full_train._digest(full_train._json(body))
    with pytest.raises(full_train.FullTrainError, match="accepted v3 closure"):
        full_train.validate_full_training_closure(forged)


def test_cross_stage_full_identity_preserves_smoke_evidence_without_falsely_rebuilding_it() -> None:
    smoke_identity = _identity(strict_source_metadata_sha256=SHA_A)
    full_stage_identity = _identity(strict_source_metadata_sha256=SHA_B)
    identity = full_train.FullRunIdentity(
        phase_b_v3_identity=full_stage_identity,
        completed_smoke_v3_identity=smoke_identity,
        closure=full_train.full_training_closure(ROOT),
        source_smoke_lineage=_lineage(smoke_identity),
    )
    payload = identity.payload()
    binding = payload["smoke_to_full_source_identity_binding"]
    assert payload["completed_smoke_v3_identity"] == smoke_identity.payload()
    assert payload["fresh_full_stage_v3_identity"] == full_stage_identity.payload()
    assert binding["completed_smoke_strict_source_metadata_sha256"] == SHA_A
    assert binding["fresh_full_stage_strict_source_metadata_sha256"] == SHA_B
    assert binding["stage_bound_metadata_must_differ"] is True
    assert binding["stage_bound_difference_reason"].startswith("fresh_full_stage_descriptor_identities")
    assert binding["completed_smoke_v3_identity_sha256"] == _lineage(smoke_identity)["completed_smoke_identity_sha256"]
    assert binding["fresh_full_stage_v3_identity_sha256"] != binding["completed_smoke_v3_identity_sha256"]
    assert binding["accepted_phase_b_v3_closure_sha256"] == full_train.ACCEPTED_PHASE_B_V3_CLOSURE_SHA256
    assert set(binding["source_authority_asset_sha256s"]) == set(phase_b.SOURCE_AUTHORITY_ASSET_PATHS)

    # Reusing the fresh full-stage identity as smoke evidence would conceal
    # the staging migration instead of validating immutable smoke bytes.
    same_stage = full_train.FullRunIdentity(
        phase_b_v3_identity=full_stage_identity,
        completed_smoke_v3_identity=full_stage_identity,
        closure=identity.closure,
        source_smoke_lineage=_lineage(full_stage_identity),
    )
    with pytest.raises(full_train.FullTrainError, match="must not share descriptor-bound"):
        same_stage.payload()

    different_roster = [f"source-{index:02d}" for index in range(27)]
    different_roster[-1] = "other-source-26"
    source_drift = full_train.FullRunIdentity(
        phase_b_v3_identity=_identity(
            strict_source_metadata_sha256=SHA_C, roster=different_roster,
        ),
        completed_smoke_v3_identity=smoke_identity,
        closure=identity.closure,
        source_smoke_lineage=_lineage(smoke_identity),
    )
    with pytest.raises(full_train.FullTrainError, match="stable source identity drift"):
        source_drift.payload()

    # A hand-constructed typed identity cannot swap in a different smoke
    # identity while retaining receipt hashes from the completed smoke graph.
    lineage_swap = full_train.FullRunIdentity(
        phase_b_v3_identity=full_stage_identity,
        completed_smoke_v3_identity=_identity(strict_source_metadata_sha256=SHA_C),
        closure=identity.closure,
        source_smoke_lineage=_lineage(smoke_identity),
    )
    with pytest.raises(full_train.FullTrainError, match="lineage/identity digest drift"):
        lineage_swap.payload()


def test_root_capability_binds_the_exact_completed_v3_step_and_terminal_lineage() -> None:
    identity = _full_identity()
    capability = full_train._issue_root_review_capability_for_audited_full_route(identity=identity)
    capability.validate(identity=identity)
    forged_lineage = copy.deepcopy(identity.source_smoke_lineage)
    forged_lineage["step100_sha256"] = SHA_C
    forged = full_train.FullRunIdentity(
        phase_b_v3_identity=identity.phase_b_v3_identity,
        completed_smoke_v3_identity=identity.completed_smoke_v3_identity,
        closure=identity.closure,
        source_smoke_lineage=forged_lineage,
    )
    with pytest.raises(full_train.FullTrainError, match="immutable smoke-v3 lineage drift"):
        capability.validate(identity=forged)


def test_mock_full_lifecycle_is_epoch_bound_and_publishes_only_after_complete_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _full_identity()
    spec = _tiny_spec()
    validated_v3: list[Mapping[str, object]] = []

    def validate_v3(value: Mapping[str, object], **_kwargs: object) -> dict[str, object]:
        validated_v3.append(dict(value))
        return dict(value)

    monkeypatch.setattr(full_train.phase_b_v3, "validate_source_authority_v3", validate_v3)
    backend = full_train.DeterministicMockBackend(source_authority_factory=_source_authority)
    temporary, artifact = _reserve_temporary_artifact(spec)
    try:
        terminal = full_train.run_full_training_lifecycle(
            backend=backend, artifact=artifact, identity_factory=lambda: identity, spec=spec,
        )
        assert terminal["status"] == "FULL_TRAINING_COMPLETE__SOURCE_ONLY__AWAITING_SEPARATE_SCORER"
        assert backend.closed is True
        assert backend.expensive_proof_count == spec.epochs
        assert backend.proof_requests == [False, True, False, True]
        assert len(validated_v3) >= 2  # source authority write + immutable reload.
        assert artifact.has_name("terminal.json") is True
        assert artifact.has_name("failure.json") is False
        for name in ("attempt.json", "launch.json", "source_authority.json", "throughput1.json", "epoch-00.json", "epoch-01.json", "checkpoint-00.pt", "checkpoint-01.pt", "swa_final4.pt", "terminal.json"):
            assert (artifact.directory / name).exists()
            assert (artifact.directory / f"{name}.sha256").exists()
            assert (artifact.directory / name).stat().st_mode & 0o777 == 0o444
    finally:
        temporary.cleanup()


def test_mock_failure_after_an_optimizer_step_is_honest_and_never_publishes_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _full_identity()
    spec = _tiny_spec()
    monkeypatch.setattr(full_train.phase_b_v3, "validate_source_authority_v3", lambda value, **_kwargs: dict(value))
    backend = full_train.DeterministicMockBackend(source_authority_factory=_source_authority, failure="after_step")
    temporary, artifact = _reserve_temporary_artifact(spec)
    try:
        with pytest.raises(full_train.FullTrainingExecutionError):
            full_train.run_full_training_lifecycle(
                backend=backend, artifact=artifact, identity_factory=lambda: identity, spec=spec,
            )
        failure = artifact.reload_json("failure.json")
        assert failure["stage"] == "epoch"
        assert failure["progress"]["optimizer_steps_completed"] == 2
        assert failure["progress"]["source_opened"] is True
        assert failure["progress"]["remote_initialized"] is True
        assert artifact.has_name("terminal.json") is False
        assert backend.closed is True
    finally:
        temporary.cleanup()


def _write_pair(directory: Path, name: str, value: Mapping[str, object]) -> str:
    body = full_train._json(dict(value))
    digest = full_train._digest(body)
    (directory / name).write_bytes(body)
    (directory / f"{name}.sha256").write_bytes(f"{digest}  {name}\n".encode("ascii"))
    os.chmod(directory / name, 0o444)
    os.chmod(directory / f"{name}.sha256", 0o444)
    return digest


def test_full_route_cannot_consume_live_or_partial_v3_smoke_root() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / phase_b_v3.SOURCE_SMOKE_V3_ROOT_RELATIVE).mkdir(parents=True)
        with pytest.raises(full_train.FullTrainError, match="missing or inaccessible"):
            full_train.load_completed_source_smoke_lineage_v3(root)


def test_completed_v3_smoke_loader_requires_step100_and_terminal_graph_before_full_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    monkeypatch.setattr(full_train.phase_b_v3, "validate_source_authority_v3", lambda value, **_kwargs: dict(value))
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        directory = root / phase_b_v3.SOURCE_SMOKE_V3_ROOT_RELATIVE
        directory.mkdir(parents=True)
        attempt = {
            "schema": "posterior_carrier_source_smoke_attempt_v3", "cell": full_train.CELL,
            "phase": phase_b_v3.PHASE_B_V3, "spec": phase_b.SMOKE_SPEC.payload(),
            "identity": identity.payload(), "v2_failed_predecessor": identity.predecessor.payload(),
            "boundaries": identity.payload()["boundaries"], "status": "ATTEMPT_STARTED_SOURCE_ONLY_V3",
        }
        attempt_sha = _write_pair(directory, "attempt.json", attempt)
        launch = {
            "schema": "posterior_carrier_source_smoke_launch_v3", "cell": full_train.CELL,
            "phase": phase_b_v3.PHASE_B_V3, "spec": phase_b.SMOKE_SPEC.payload(),
            "identity": identity.payload(), "attempt_sha256": attempt_sha,
            "v2_failed_predecessor": identity.predecessor.payload(), "launch_closure": identity.closure,
            "status": "SOURCE_SMOKE_V3_LAUNCHED",
        }
        launch_sha = _write_pair(directory, "launch.json", launch)
        authority_sha = _write_pair(directory, "source_authority.json", {"synthetic": "v3-authority"})
        step = {
            "schema": "posterior_carrier_source_smoke_step100_v3", "cell": full_train.CELL,
            "phase": phase_b_v3.PHASE_B_V3, "spec": phase_b.SMOKE_SPEC.payload(),
            "identity": identity.payload(), "launch_sha256": launch_sha,
            "source_authority_sha256": authority_sha, "summary": _valid_v3_smoke_summary(),
            "v2_failed_predecessor": identity.predecessor.payload(),
            "boundaries": identity.payload()["boundaries"], "status": "SOURCE_SMOKE_V3_100_STEPS_COMPLETE",
        }
        step_sha = _write_pair(directory, "step100.json", step)
        terminal = {
            "schema": "posterior_carrier_source_smoke_terminal_v3", "cell": full_train.CELL,
            "phase": phase_b_v3.PHASE_B_V3, "spec": phase_b.SMOKE_SPEC.payload(),
            "identity": identity.payload(), "attempt_sha256": attempt_sha, "launch_sha256": launch_sha,
            "source_authority_sha256": authority_sha, "step100_sha256": step_sha,
            "v2_failed_predecessor": identity.predecessor.payload(),
            "launch_closure": identity.closure, "final_closure": identity.closure,
            "boundaries": identity.payload()["boundaries"],
            "status": "SOURCE_SMOKE_V3_COMPLETE__NON_AUTHORITATIVE",
        }
        terminal_sha = _write_pair(directory, "terminal.json", terminal)
        completed = full_train.load_completed_source_smoke_lineage_v3(root)
        assert completed.smoke_identity.payload() == identity.payload()
        assert completed.lineage["terminal_sha256"] == terminal_sha
        assert completed.lineage["step100_sha256"] == step_sha
        # A fresh full-stage descriptor read has a different strict metadata
        # body SHA.  It must still consume the exact immutable smoke receipt,
        # rather than revalidate that receipt against the fresh-stage identity.
        migrated = full_train.FullRunIdentity(
            phase_b_v3_identity=_identity(strict_source_metadata_sha256=SHA_B),
            completed_smoke_v3_identity=completed.smoke_identity,
            closure=full_train.full_training_closure(ROOT),
            source_smoke_lineage=completed.lineage,
        )
        assert migrated.payload()["smoke_to_full_source_identity_binding"]["completed_smoke_strict_source_metadata_sha256"] == SHA_A
        assert migrated.payload()["smoke_to_full_source_identity_binding"]["fresh_full_stage_strict_source_metadata_sha256"] == SHA_B
        # Exercise the reviewed builder itself on a fresh full-stage fixture:
        # it must validate receipt graph A, then bind fresh source identity B.
        fresh_full_stage = _identity(strict_source_metadata_sha256=SHA_B)
        fresh_full_closure = full_train.full_training_closure(ROOT)
        monkeypatch.setattr(full_train.phase_b_v3, "phase_b_v3_closure", lambda _root: fresh_full_stage.closure)
        monkeypatch.setattr(full_train, "full_training_closure", lambda _root: fresh_full_closure)
        rebuilt = full_train.build_reviewed_full_identity(
            root, phase_b_v3_identity=fresh_full_stage,
        )
        assert rebuilt.completed_smoke_v3_identity.payload() == identity.payload()
        assert rebuilt.phase_b_v3_identity.payload() == fresh_full_stage.payload()
        (directory / "failure.json").write_bytes(b"unexpected\n")
        os.chmod(directory / "failure.json", 0o444)
        with pytest.raises(full_train.FullTrainError, match="coexist"):
            full_train.load_completed_source_smoke_lineage_v3(root)
        os.unlink(directory / "failure.json")
        bad = json.loads((directory / "terminal.json").read_bytes())
        os.chmod(directory / "terminal.json", 0o600)
        os.chmod(directory / "terminal.json.sha256", 0o600)
        bad["step100_sha256"] = SHA_A
        _write_pair(directory, "terminal.json", bad)
        with pytest.raises(full_train.FullTrainError, match="terminal graph"):
            full_train.load_completed_source_smoke_lineage_v3(root)


def test_future_full_staging_plan_is_explicit_and_copies_only_completed_v3_smoke_receipts() -> None:
    identity = _full_identity()

    class _SourceCapability:
        def payload(self) -> Mapping[str, object]:
            return _source_root_payload()

    plan = full_train.full_remote_staging_plan(identity=identity, source_data=_SourceCapability())
    paths = [item["relative_path"] for item in plan["stage_files"]]
    assert plan["accepted_phase_b_v3_closure_sha256"] == full_train.ACCEPTED_PHASE_B_V3_CLOSURE_SHA256
    assert plan["nwb_assets_in_stage"] is False
    assert full_train.FULL_MODULE_RELATIVE in paths
    assert f"{phase_b_v3.SOURCE_SMOKE_V3_ROOT_RELATIVE}/step100.json" in paths
    assert f"{phase_b_v3.SOURCE_SMOKE_V3_ROOT_RELATIVE}/terminal.json" in paths
    assert not any("*" in item or "glob" in item for item in paths)
    forged = copy.deepcopy(identity.source_smoke_lineage)
    forged["terminal_status"] = "SOURCE_SMOKE_V3_FAILED_HONESTLY"
    bad = full_train.FullRunIdentity(
        phase_b_v3_identity=identity.phase_b_v3_identity,
        completed_smoke_v3_identity=identity.completed_smoke_v3_identity,
        closure=identity.closure,
        source_smoke_lineage=forged,
    )
    with pytest.raises(full_train.FullTrainError, match="terminal status"):
        full_train.full_remote_staging_plan(identity=bad, source_data=_SourceCapability())


def test_deferred_physical_backend_selects_only_the_v2_source_adapter_after_v3_tf32_enforcement_before_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _full_identity()

    class _SourceCapability:
        def payload(self) -> Mapping[str, object]:
            return _source_root_payload()

        def validate(self) -> Path:
            return Path("/synthetic/external-source")

    calls = {"tf32_enforcement": 0, "v2_adapter": 0}

    def fail_v2_adapter(*_args: object, **kwargs: object) -> object:
        calls["v2_adapter"] += 1
        kwargs["on_source_opened"]()
        raise source_adapter_v2.SourceAdapterV2Error("synthetic source-only stop")

    monkeypatch.setattr(full_train.phase_b, "validate_stage_source_separation", lambda **_kwargs: None)
    monkeypatch.setattr(full_train, "full_training_closure", lambda _root: identity.closure)
    monkeypatch.setattr(full_train.phase_b_v3, "phase_b_v3_closure", lambda _root: identity.phase_b_v3_identity.closure)
    policy = phase_b_v3.TF32Enforcement(
        pre_matmul_allow_tf32=False,
        pre_cudnn_allow_tf32=False,
        pre_amp_enabled=False,
        post_matmul_allow_tf32=False,
        post_cudnn_allow_tf32=False,
        post_amp_enabled=False,
    )
    def enforce_before_source_adapter(_torch: object) -> phase_b_v3.TF32Enforcement:
        calls["tf32_enforcement"] += 1
        assert calls["v2_adapter"] == 0
        return policy

    monkeypatch.setattr(full_train.phase_b_v3.TF32Enforcement, "enforce", enforce_before_source_adapter)
    monkeypatch.setattr(source_adapter_v2, "build_physical_source_adapter_v2", fail_v2_adapter)
    backend = full_train.RemotePosteriorFullTrainingBackend(
        Path("/synthetic/stage"), source_data=_SourceCapability(), num_workers=4,
    )
    with pytest.raises(full_train.FullTrainingExecutionError) as caught:
        backend.prepare(full_train.PUBLIC_SPEC, identity)
    assert calls == {"tf32_enforcement": 1, "v2_adapter": 1}
    assert caught.value.stage == "prepare"
    assert caught.value.progress.source_opened is True
    assert caught.value.progress.remote_initialized is False


def test_swa_state_mean_is_tensor_exact_and_preserves_lazy_topology() -> None:
    from torch.nn.parameter import UninitializedParameter

    first = {"weight": torch.tensor([1.0, 3.0]), "lazy": UninitializedParameter()}
    second = {"weight": torch.tensor([3.0, 5.0]), "lazy": UninitializedParameter()}
    mean = full_train._arithmetic_swa_state([first, second], torch_module=torch)
    assert torch.equal(mean["weight"], torch.tensor([2.0, 4.0]))
    assert isinstance(mean["lazy"], UninitializedParameter)
    assert full_train._lazy_safe_state_equal(mean, mean, torch_module=torch)
    changed = {"weight": torch.tensor([2.0, 4.1]), "lazy": UninitializedParameter()}
    assert not full_train._lazy_safe_state_equal(mean, changed, torch_module=torch)


def test_static_public_cli_never_imports_torch_and_execution_flags_fail_closed() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_posterior_carrier_full_train.py"
    environment = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": ""}
    code = (
        "import importlib.util,json,sys; "
        f"spec=importlib.util.spec_from_file_location('pc_full', {str(script)!r}); "
        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);"
        "payload=module._load_plan();print(json.dumps({'torch_loaded':'torch' in sys.modules,'status':payload['status']}))"
    )
    completed = subprocess.run([sys.executable, "-c", code], env=environment, check=True, text=True, capture_output=True)
    assert json.loads(completed.stdout) == {
        "torch_loaded": False,
        "status": "DRY_FAIL_CLOSED_PENDING_ROOT_REVIEW_AND_COMPLETED_SMOKE_V3",
    }
    rejected = subprocess.run(
        [sys.executable, str(script), "--execute-remote-full"], env=environment, text=True, capture_output=True,
    )
    assert rejected.returncode != 0
    assert "root-reviewed capability" in rejected.stderr
