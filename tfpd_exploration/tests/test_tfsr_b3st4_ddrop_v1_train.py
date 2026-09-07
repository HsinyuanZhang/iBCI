"""Phase-D lifecycle tests.

Every lifecycle execution below uses ``DeterministicMockBackend``.  It opens no
NWB and cannot initialize CUDA; the one explicit CPU-only Adam regression
exercises the repaired scalar-state boundary proof without invoking a model
builder, source adapter, or CUDA runtime.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tfpd_exploration"))
from src.tfsr_b3st4_ddrop_v1 import train


MOCK_SPEC = train.RunSpec(epochs=2, batch_size=2, steps_per_epoch=2, checkpoint_epochs=(0, 1), throughput_probe_steps=2)


def _env(extra: str = "") -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": ":".join(item for item in (extra, str(ROOT / "tfpd_exploration"), str(ROOT / "tfpd_exploration/src")) if item),
    })
    return value


def _artifact(tmp_path: Path, spec: train.RunSpec = MOCK_SPEC) -> train.ArtifactRoot:
    return train.reserve_artifact_root(tmp_path, "artifacts", spec.topology)


def _run_mock(tmp_path: Path, *, failure: str | None = None, spec: train.RunSpec = MOCK_SPEC):
    backend = train.DeterministicMockBackend(failure=failure)
    artifact = _artifact(tmp_path, spec)
    return artifact, backend, lambda: train.run_lifecycle(
        spec=spec, backend=backend, artifact=artifact, identity_factory=train.mock_identity,
    )


def _rewrite_pair(artifact: train.ArtifactRoot, name: str, body: bytes) -> None:
    """Controlled post-publication corruption for reloader/validator tests."""
    path = artifact.directory / name
    sidecar = artifact.directory / f"{name}.sha256"
    os.chmod(path, 0o644)
    path.write_bytes(body)
    os.chmod(path, 0o444)
    digest = train._sha(body)
    os.chmod(sidecar, 0o644)
    sidecar.write_bytes(f"{digest}  {name}\n".encode())
    os.chmod(sidecar, 0o444)


def _preterminal_hashes(artifact: train.ArtifactRoot, spec: train.RunSpec) -> dict[str, str]:
    names = set(spec.topology) - {"terminal.json", "failure.json"}
    return {name: train._sha(artifact.reload_pair(name)) for name in names}


def test_public_static_plan_binds_phase_c_and_keeps_canonical_phase_d_absent():
    plan = train.training_plan(ROOT)
    assert plan["status"] == "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_LAUNCH"
    assert plan["phase_c_acceptance"]["body_sha256"] == train.PHASE_C_RECEIPT_SHA
    assert plan["training"]["total_optimizer_steps"] == 48 * 33_925
    assert plan["training"]["checkpoint_epochs"] == [44, 45, 46, 47]
    assert not (ROOT / train.TRAIN_ROOT_RELATIVE).exists()


def test_public_flags_reject_before_torch_and_default_cli_is_static(tmp_path: Path):
    (tmp_path / "torch.py").write_text("raise RuntimeError('TORCH_IMPORTED')\n")
    script = ROOT / "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed42_train.py"
    dry = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=_env(str(tmp_path)), capture_output=True, text=True)
    assert dry.returncode == 0
    assert json.loads(dry.stdout)["authorization"] == "none"
    for arguments in (("--execute",), ("--i-have-48epoch-authorization",), ("--unknown",), ("--execute", "--execute")):
        done = subprocess.run([sys.executable, str(script), *arguments], cwd=ROOT, env=_env(str(tmp_path)), capture_output=True, text=True)
        assert done.returncode != 0
        assert "TORCH_IMPORTED" not in done.stdout + done.stderr


def test_public_lr_uses_exact_live_arm_common_and_mock_spec_is_deterministic():
    sys.path.insert(0, str(ROOT / "tfpd_exploration/src"))
    from tfpd_lane.arm_common import lr_at_step
    probes = [0, train.WARMUP_STEPS - 1, train.WARMUP_STEPS, train.TOTAL_STEPS - 1]
    probes.extend(((index * 15_485_863) % train.TOTAL_STEPS) for index in range(1, 101))
    assert all(train.lr_for_step(step) == lr_at_step(step, 48, 33_925) for step in probes)
    assert [train.lr_for_spec(MOCK_SPEC, step) for step in range(4)] == pytest.approx([1e-5, 3.25e-5, 5.5e-5, 7.75e-5])
    with pytest.raises(ValueError):
        train.lr_for_step(train.TOTAL_STEPS)


def test_runspec_rejects_budget_and_checkpoint_drift():
    with pytest.raises(ValueError):
        train.RunSpec(0, 2, 2, (0,), 1)
    with pytest.raises(ValueError):
        train.RunSpec(2, 2, 2, (1, 0), 1)
    with pytest.raises(ValueError):
        train.RunSpec(2, 2, 2, (2,), 1)
    with pytest.raises(ValueError):
        train.RunSpec(2, 2, 2, (0,), 5)
    with pytest.raises(ValueError):
        train.RunSpec(2, 2, 2, (0,), 1, seed=43)


def test_artifact_root_reservation_collision_and_transactional_rollback(tmp_path: Path, monkeypatch):
    existing = tmp_path / "artifacts"
    existing.mkdir()
    with pytest.raises(RuntimeError):
        train.reserve_artifact_root(tmp_path, "artifacts", MOCK_SPEC.topology)
    existing.rmdir()
    artifact = _artifact(tmp_path)
    calls = {"count": 0}
    original = train._write_full

    def fail_second(fd: int, data: bytes) -> None:
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("synthetic sidecar write failure")
        original(fd, data)

    monkeypatch.setattr(train, "_write_full", fail_second)
    with pytest.raises(OSError):
        artifact.publish_bytes("launch.json", b"{}\n")
    assert artifact.directory.exists()
    assert not (artifact.directory / "launch.json").exists()
    assert not (artifact.directory / "launch.json.sha256").exists()


def test_artifact_parent_dirfd_named_root_identity_rejects_replacement(tmp_path: Path):
    artifact = _artifact(tmp_path)
    moved = tmp_path / "moved"
    artifact.directory.rename(moved)
    artifact.directory.mkdir()
    with pytest.raises(RuntimeError, match="identity"):
        artifact.publish_bytes("attempt.json", b"{}\n")


def test_artifact_reloader_rejects_sidecar_and_mode_drift(tmp_path: Path):
    artifact = _artifact(tmp_path)
    digest = artifact.publish_bytes("attempt.json", b"{}\n")
    assert artifact.reload_pair("attempt.json", digest) == b"{}\n"
    sidecar = artifact.directory / "attempt.json.sha256"
    os.chmod(sidecar, 0o644)
    sidecar.write_text("bad\n")
    os.chmod(sidecar, 0o444)
    with pytest.raises(RuntimeError):
        artifact.reload_pair("attempt.json")
    os.chmod(artifact.directory / "attempt.json", 0o644)
    with pytest.raises(RuntimeError):
        artifact.reload_pair("attempt.json")


def test_mock_lifecycle_publishes_and_same_fd_reloads_every_required_artifact(tmp_path: Path):
    artifact, backend, run = _run_mock(tmp_path)
    terminal = run()
    assert terminal["status"] == "TRAINING_COMPLETE"
    assert backend.closed is True
    expected = set(MOCK_SPEC.topology) - {"failure.json"}
    assert {path.name for path in artifact.directory.iterdir() if not path.name.endswith(".sha256")} == expected
    for name in expected:
        assert stat_mode(artifact.directory / name) == 0o444
        assert stat_mode(artifact.directory / f"{name}.sha256") == 0o444
        artifact.reload_pair(name)
    train.validate_terminal_receipt(artifact.reload_json("terminal.json"), MOCK_SPEC, train.mock_identity())
    assert terminal["swa_evaluation_proof"]["repeat_bitwise_equal"] is True
    assert terminal["swa_evaluation_proof"]["eval_no_mask"] is True
    assert terminal["swa_evaluation_proof"]["state_unchanged"] is True
    assert backend.proof_requests == [False, True, False, True]
    assert backend.expensive_proof_count == 2


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_mock_epoch_receipts_record_all_required_dropout_lrs_gradients_resources_and_progress(tmp_path: Path):
    artifact, _, run = _run_mock(tmp_path)
    run()
    first = artifact.reload_json("epoch-00.json")
    second = artifact.reload_json("epoch-01.json")
    train.validate_epoch_receipt(first, 0, 2, MOCK_SPEC)
    train.validate_epoch_receipt(second, 1, 4, MOCK_SPEC)
    assert set(first["dropout"]) == {"p_min", "p_max", "p_mean", "p_q25", "p_q50", "p_q75", "kept", "dropped", "kept_fraction", "dropped_fraction", "all_zero_examples", "population_examples", "all_zero_fraction", "max_gain"}
    assert first["dropout"]["p_min"] == 0.25
    assert first["dropout"]["p_max"] == 0.75
    assert first["lr"]["first"] == first["lr"]["expected_first"]
    assert first["progress"] == {"epoch": 0, "completed_epochs": 1, "epochs": 2, "global_step": 2, "total_optimizer_steps": 4}
    assert all(first["critical_gradients"].values())
    assert first["finite"] == {"model": True, "optimizer": True}
    assert first["boundaries"]["target_or_formal_opened"] is False


def test_mock_lifecycle_uses_two_epochs_two_steps_both_checkpoints_and_threshold_two(tmp_path: Path):
    artifact, _, run = _run_mock(tmp_path)
    terminal = run()
    assert (artifact.directory / "throughput2.json").exists()
    assert (artifact.directory / "checkpoint-00.pt").exists()
    assert (artifact.directory / "checkpoint-01.pt").exists()
    assert terminal["total_optimizer_steps"] == 4
    assert terminal["checkpoint_sha256"].keys() == {"0", "1"}
    assert terminal["artifact_validation"].keys() >= {"throughput2.json", "checkpoint-00.pt", "checkpoint-01.pt"}


@pytest.mark.parametrize(
    ("failure", "source_opened", "gpu_initialized", "steps"),
    (("before_gpu", False, False, 0), ("during_adapter", True, True, 0), ("after_step", True, True, 1)),
)
def test_failure_envelopes_are_honest_and_never_publish_terminal(tmp_path: Path, failure: str, source_opened: bool, gpu_initialized: bool, steps: int):
    artifact, backend, run = _run_mock(tmp_path, failure=failure)
    with pytest.raises(RuntimeError, match="synthetic"):
        run()
    failure_receipt = artifact.reload_json("failure.json")
    train.validate_failure_receipt(failure_receipt)
    assert failure_receipt["source_opened"] is source_opened
    assert failure_receipt["gpu_initialized"] is gpu_initialized
    assert failure_receipt["optimizer_steps_completed"] == steps
    assert failure_receipt["terminal_published"] is False
    assert not (artifact.directory / "terminal.json").exists()
    assert backend.closed is True


def test_failure_before_gpu_has_no_launch_or_epoch_artifacts(tmp_path: Path):
    artifact, _, run = _run_mock(tmp_path, failure="before_gpu")
    with pytest.raises(RuntimeError):
        run()
    bodies = {path.name for path in artifact.directory.iterdir() if not path.name.endswith(".sha256")}
    assert bodies == {"attempt.json", "failure.json"}


def test_epoch_corruption_is_caught_after_pair_reload(tmp_path: Path):
    artifact, _, run = _run_mock(tmp_path)
    run()
    value = dict(artifact.reload_json("epoch-00.json"))
    value["epoch"] = 9
    _rewrite_pair(artifact, "epoch-00.json", train._json(value))
    with pytest.raises(RuntimeError, match="accounting"):
        train.validate_epoch_receipt(artifact.reload_json("epoch-00.json"), 0, 2, MOCK_SPEC)


def test_checkpoint_corruption_is_caught_after_pair_reload(tmp_path: Path):
    artifact, backend, run = _run_mock(tmp_path)
    run()
    value = json.loads(artifact.reload_pair("checkpoint-00.pt"))
    value["global_step"] = 999
    _rewrite_pair(artifact, "checkpoint-00.pt", train._json(value))
    with pytest.raises(RuntimeError, match="checkpoint"):
        backend.validate_checkpoint(artifact.reload_pair("checkpoint-00.pt"), 0, 2, MOCK_SPEC)


@pytest.mark.parametrize("binding_field", ("launch_sha256", "launch_closure"))
def test_terminal_finalizer_rejects_checkpoint_launch_binding_tampering(tmp_path: Path, binding_field: str):
    artifact, backend, run = _run_mock(tmp_path)
    run()
    value = json.loads(artifact.reload_pair("checkpoint-00.pt"))
    value["binding"] = dict(value["binding"])
    if binding_field == "launch_sha256":
        value["binding"][binding_field] = "f" * 64
    else:
        value["binding"][binding_field] = {"stage0": {"closure_sha256": "f" * 64}}
    # The body and its sidecar are recomputed consistently.  Thus a simple
    # SHA/mode check still passes; only exact terminal binding validation can
    # reject this adversarial but syntactically valid checkpoint.
    _rewrite_pair(artifact, "checkpoint-00.pt", train._json(value))
    hashes = _preterminal_hashes(artifact, MOCK_SPEC)
    expected_binding = {
        "cell": train.CELL, "run_spec": MOCK_SPEC.payload(), "launch_sha256": hashes["launch.json"],
        "launch_closure": train.mock_identity().closures,
    }
    with pytest.raises(RuntimeError, match="exact launch binding"):
        backend.build_swa(
            None,
            {epoch: artifact.reload_pair(f"checkpoint-{epoch:02d}.pt") for epoch in MOCK_SPEC.checkpoint_epochs},
            MOCK_SPEC, expected_binding=expected_binding,
        )
    with pytest.raises(RuntimeError, match="exact launch binding"):
        train._validate_all_preterminal_artifacts(
            artifact, MOCK_SPEC, train.mock_identity(), backend, hashes,
        )


def test_swa_corruption_is_caught_after_pair_reload(tmp_path: Path):
    artifact, backend, run = _run_mock(tmp_path)
    run()
    value = json.loads(artifact.reload_pair("swa.pt"))
    value["evaluation_proof"]["eval_no_mask"] = False
    _rewrite_pair(artifact, "swa.pt", train._json(value))
    with pytest.raises(RuntimeError, match="SWA"):
        backend.validate_swa(artifact.reload_pair("swa.pt"), MOCK_SPEC)


def test_terminal_corruption_is_caught_after_pair_reload(tmp_path: Path):
    artifact, _, run = _run_mock(tmp_path)
    run()
    value = dict(artifact.reload_json("terminal.json"))
    value["status"] = "TAMPERED"
    _rewrite_pair(artifact, "terminal.json", train._json(value))
    with pytest.raises(RuntimeError, match="terminal"):
        train.validate_terminal_receipt(artifact.reload_json("terminal.json"), MOCK_SPEC, train.mock_identity())


def test_attempt_validator_rejects_each_boundary_and_topology_drift():
    identity = train.mock_identity()
    value = train._attempt_payload(MOCK_SPEC, identity)
    train.validate_attempt_receipt(value, MOCK_SPEC, identity)
    for key, changed in (("cell", "bad"), ("source_opened", True), ("gpu_initialized", True), ("target_or_formal_opened", True), ("topology", [])):
        tampered = dict(value)
        tampered[key] = changed
        with pytest.raises(RuntimeError):
            train.validate_attempt_receipt(tampered, MOCK_SPEC, identity)


def test_launch_validator_rejects_identity_optimizer_and_boundary_drift():
    identity = train.mock_identity()
    value = train._launch_payload(MOCK_SPEC, identity)
    train.validate_launch_receipt(value, MOCK_SPEC, identity)
    for key, changed in (("device", {}), ("optimizer", {}), ("source_authorities", {}), ("boundaries", {}), ("run_spec", {})):
        tampered = dict(value)
        tampered[key] = changed
        with pytest.raises(RuntimeError):
            train.validate_launch_receipt(tampered, MOCK_SPEC, identity)


def test_throughput_validator_rejects_threshold_numeric_resource_and_boundary_drift():
    identity = train.mock_identity()
    flags = train.LifecycleFlags(source_opened=True, gpu_initialized=True,
                                 predecessor=identity.predecessor,
                                 launch_closure=identity.closures)
    value = train._throughput_payload(MOCK_SPEC, 1.0, {"rss_bytes": 1, "peak_allocated_bytes": 2, "peak_reserved_bytes": 3}, flags)
    train.validate_throughput_receipt(value, MOCK_SPEC)
    for key, changed in (("threshold_steps", 1), ("elapsed_seconds", 0.0), ("steps_per_second", float("nan")), ("resources", {}), ("boundaries", {})):
        tampered = dict(value)
        tampered[key] = changed
        with pytest.raises(RuntimeError):
            train.validate_throughput_receipt(tampered, MOCK_SPEC)


def test_terminal_validator_rejects_closure_sha_checkpoint_and_proof_drift(tmp_path: Path):
    artifact, _, run = _run_mock(tmp_path)
    value = dict(run())
    identity = train.mock_identity()
    train.validate_terminal_receipt(value, MOCK_SPEC, identity)
    for mutate in (
        lambda item: item.__setitem__("total_optimizer_steps", 3),
        lambda item: item.__setitem__("final_closure", {"bad": True}),
        lambda item: item["checkpoint_sha256"].__setitem__("0", "bad"),
        lambda item: item["swa_evaluation_proof"].__setitem__("repeat_bitwise_equal", False),
    ):
        tampered = json.loads(json.dumps(value))
        mutate(tampered)
        with pytest.raises(RuntimeError):
            train.validate_terminal_receipt(tampered, MOCK_SPEC, identity)


def test_mock_backend_cannot_silently_open_target_or_formal_domain(tmp_path: Path):
    class BadBoundary(train.DeterministicMockBackend):
        def prepare(self, spec, identity, flags):
            runtime = super().prepare(spec, identity, flags)
            flags.target_or_formal_opened = True
            return runtime

    backend = BadBoundary()
    artifact = _artifact(tmp_path)
    with pytest.raises(RuntimeError, match="target/formal"):
        train.run_lifecycle(spec=MOCK_SPEC, backend=backend, artifact=artifact, identity_factory=train.mock_identity)
    receipt = artifact.reload_json("failure.json")
    # Failure records are deliberately honest rather than source-only success
    # certificates; only the terminal/epoch paths require this field to be
    # false.
    assert receipt["target_or_formal_opened"] is True
    train.validate_failure_receipt(receipt)


def test_public_topology_is_exact_and_test_topology_is_not_a_cli_override():
    assert train.PUBLIC_SPEC.topology == train.TOPOLOGY
    assert train.PUBLIC_SPEC.topology[2] == "throughput100.json"
    assert MOCK_SPEC.topology[2] == "throughput2.json"
    assert train.PUBLIC_SPEC.payload()["epochs"] == 48
    assert MOCK_SPEC.payload()["epochs"] == 2


def test_full_state_proof_is_requested_once_per_epoch_and_never_in_public_first_100_steps(tmp_path: Path):
    # This directly audits the public control predicate: step index 99 means
    # optimizer step 100, which remains far from public epoch boundary 33,925.
    assert not any(train.requires_epoch_proof(train.PUBLIC_SPEC, step) for step in range(100))
    assert train.requires_epoch_proof(train.PUBLIC_SPEC, train.PUBLIC_SPEC.steps_per_epoch - 1) is True
    assert train.requires_epoch_proof(train.PUBLIC_SPEC, train.PUBLIC_SPEC.steps_per_epoch) is False

    # Exercise the same injected lifecycle around a 101-step mock epoch: its
    # threshold-100 receipt is written before the sole full-state proof at
    # step 101, proving the bounded scheduling behavior rather than a comment.
    spec = train.RunSpec(epochs=1, batch_size=2, steps_per_epoch=101, checkpoint_epochs=(0,), throughput_probe_steps=100)
    backend = train.DeterministicMockBackend()
    artifact = _artifact(tmp_path, spec)
    train.run_lifecycle(spec=spec, backend=backend, artifact=artifact, identity_factory=train.mock_identity)
    assert backend.proof_requests[:100] == [False] * 100
    assert backend.proof_requests[100:] == [True]
    assert backend.expensive_proof_count == 1
    assert (artifact.directory / "throughput100.json").exists()


def test_nonboundary_steps_cannot_fabricate_full_state_or_gradient_evidence():
    backend = train.DeterministicMockBackend()
    ordinary = backend.train_step({"step": 0}, 1e-5, 0, require_epoch_proof=False)
    train._validate_step_outcome(ordinary, 1e-5, require_epoch_proof=False)
    forged = replace(ordinary, critical_gradients={name: True for name in train._CRITICAL_GRADIENT_GROUPS})
    with pytest.raises(RuntimeError, match="fabricated"):
        train._validate_step_outcome(forged, 1e-5, require_epoch_proof=False)
    boundary = backend.train_step({"step": 0}, 1e-5, 1, require_epoch_proof=True)
    train._validate_step_outcome(boundary, 1e-5, require_epoch_proof=True)


def test_artifact_capability_rejects_every_name_outside_its_fixed_topology(tmp_path: Path):
    artifact = _artifact(tmp_path)
    for name in ("unexpected.json", "../attempt.json", "nested/attempt.json", "", "terminal.json.bak"):
        with pytest.raises(RuntimeError):
            artifact.publish_bytes(name, b"{}\n")


@pytest.mark.parametrize("key,value", (("terminal_published", True), ("optimizer_steps_completed", -1), ("traceback_sha256", "bad")))
def test_failure_validator_rejects_terminal_or_accounting_drift(key: str, value: object):
    flags = train.LifecycleFlags(stage="epoch", source_opened=True, gpu_initialized=True,
                                 optimizer_steps_completed=1,
                                 predecessor=train.mock_identity().predecessor,
                                 identity_closure=train.mock_identity().closures)
    receipt = train._failure_payload(flags)
    receipt[key] = value
    with pytest.raises(RuntimeError):
        train.validate_failure_receipt(receipt)


@pytest.mark.parametrize("key,value", (("p_q50", 1.1), ("kept_fraction", 0.0), ("state", {})))
def test_epoch_validator_rejects_quantile_fraction_and_state_drift(tmp_path: Path, key: str, value: object):
    artifact, _, run = _run_mock(tmp_path)
    run()
    receipt = dict(artifact.reload_json("epoch-00.json"))
    if key in receipt["dropout"]:
        receipt["dropout"] = dict(receipt["dropout"])
        receipt["dropout"][key] = value
    else:
        receipt[key] = value
    with pytest.raises(RuntimeError):
        train.validate_epoch_receipt(receipt, 0, 2, MOCK_SPEC)


def test_terminal_finalizer_detects_launch_final_identity_drift_and_emits_no_terminal(tmp_path: Path):
    artifact = _artifact(tmp_path)
    backend = train.DeterministicMockBackend()
    calls = {"count": 0}

    def identity_factory():
        calls["count"] += 1
        identity = train.mock_identity()
        if calls["count"] == 2:
                return train.RunIdentity(identity.phase_c_acceptance, identity.source_authorities,
                                     {"stage0": {"closure_sha256": "c" * 64}, "phase_c": {"closure_sha256": "d" * 64}, "phase_d": {"closure_sha256": "f" * 64}},
                                     identity.device, identity.predecessor)
        return identity

    with pytest.raises(RuntimeError, match="identity drift"):
        train.run_lifecycle(spec=MOCK_SPEC, backend=backend, artifact=artifact, identity_factory=identity_factory)
    failure = artifact.reload_json("failure.json")
    assert failure["stage"] == "terminal"
    assert not (artifact.directory / "terminal.json").exists()


def _legacy_nonscalar_tensor_digest(state, torch) -> str:
    """The exact v1 byte path, legal only for tensors with at least one dim."""
    digest = __import__("hashlib").sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def test_tensor_digest_accepts_scalar_empty_and_preserves_all_nonscalar_v1_bytes():
    import torch

    assert torch.cuda.is_initialized() is False
    tensors = {
        "scalar_float": torch.tensor(1.25, dtype=torch.float32),
        "empty_float": torch.empty((0,), dtype=torch.float32),
        "vector_float": torch.tensor([1.0, -2.0], dtype=torch.float32),
        "matrix_float": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "noncontiguous_float": torch.arange(6, dtype=torch.float32).reshape(2, 3).transpose(0, 1),
        "vector_int": torch.tensor([1, 2, 3], dtype=torch.int64),
        "matrix_bool": torch.tensor([[True, False], [False, True]], dtype=torch.bool),
    }
    for name, tensor in tensors.items():
        observed = train._tensor_digest({name: tensor}, torch)
        assert len(observed) == 64
        assert observed == train._tensor_digest({name: tensor.clone()}, torch)
        if tensor.ndim > 0:
            assert observed == _legacy_nonscalar_tensor_digest({name: tensor}, torch)
    assert torch.cuda.is_initialized() is False


def test_real_cpu_adam_scalar_step_passes_the_exact_epoch_boundary_proof():
    import torch

    assert torch.cuda.is_initialized() is False
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = model(torch.tensor([[0.5, -1.0, 2.0]], dtype=torch.float32)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer_state = optimizer.state_dict()["state"]
    entries = tuple(optimizer_state.values())
    assert entries and all(item["step"].ndim == 0 for item in entries)
    assert all(item["exp_avg"].ndim > 0 and item["exp_avg_sq"].ndim > 0 for item in entries)
    expected_gradients = {name: True for name in train._CRITICAL_GRADIENT_GROUPS}
    proof = train._epoch_boundary_proof(model, optimizer, torch, lambda _model, _torch: expected_gradients)
    gradients, finite_model, finite_optimizer, model_digest, optimizer_digest = proof
    assert gradients == expected_gradients
    assert finite_model is True and finite_optimizer is True
    assert len(model_digest) == 64 and len(optimizer_digest) == 64
    assert optimizer_digest == train._optimizer_digest(optimizer, torch)
    assert torch.cuda.is_initialized() is False


def _write_fake_predecessor(root: Path, *, relative: str = "predecessor-v1") -> dict[str, str]:
    directory = root / relative
    directory.mkdir()
    failure = {
        "schema": "tfsr_b3st4_ddrop_train_failure_v2", "cell": train.CELL,
        "stage": "optimizer_step", "source_opened": True, "gpu_initialized": True,
        "optimizer_steps_completed": 33_924, "target_or_formal_opened": False,
        "terminal_published": False, "traceback_sha256": "a" * 64,
    }
    bodies = {
        "attempt.json": b'{"synthetic":"attempt"}\n',
        "launch.json": b'{"synthetic":"launch"}\n',
        "throughput100.json": b'{"synthetic":"throughput"}\n',
        "failure.json": train._json(failure),
    }
    hashes: dict[str, str] = {}
    for name, body in bodies.items():
        path = directory / name
        path.write_bytes(body)
        digest = train._sha(body)
        hashes[name] = digest
        (directory / f"{name}.sha256").write_bytes(f"{digest}  {name}\n".encode("ascii"))
        os.chmod(path, 0o444)
        os.chmod(directory / f"{name}.sha256", 0o444)
    return hashes


def test_real_v1_predecessor_is_exact_and_v2_root_is_still_absent():
    lineage = train.verify_predecessor_failure(ROOT)
    assert lineage == train._public_predecessor_lineage()
    predecessor = ROOT / train.PREDECESSOR_TRAIN_ROOT_RELATIVE
    assert {item.name for item in predecessor.iterdir()} == train.PREDECESSOR_EXPECTED_LEAVES
    assert not (ROOT / train.TRAIN_ROOT_RELATIVE).exists()


@pytest.mark.parametrize("fault", ("missing", "symlink", "mutable", "malformed", "wrong_sha"))
def test_predecessor_evidence_adversaries_fail_closed(tmp_path: Path, fault: str):
    hashes = _write_fake_predecessor(tmp_path)
    relative = "predecessor-v1"
    directory = tmp_path / relative
    if fault == "missing":
        (directory / "launch.json.sha256").unlink()
    elif fault == "symlink":
        target = directory / "attempt.json"
        os.chmod(target, 0o644)
        target.unlink()
        target.symlink_to("failure.json")
    elif fault == "mutable":
        os.chmod(directory / "failure.json", 0o644)
    elif fault == "malformed":
        body = train._json({"not": "a v1 failure"})
        _rewrite_pair(type("SyntheticArtifact", (), {"directory": directory})(), "failure.json", body)
        hashes["failure.json"] = train._sha(body)
    elif fault == "wrong_sha":
        hashes["failure.json"] = "f" * 64
    with pytest.raises(RuntimeError):
        train._verify_predecessor_failure_at(tmp_path, relative, hashes)


def test_v1_copy_cannot_be_used_as_v2_output_and_freshness_stops_before_backend(tmp_path: Path, monkeypatch):
    # A byte-for-byte copied v1 root is still an occupied v2 root, never a
    # resumable successor.  The output gate trips before backend construction.
    copied = tmp_path / train.TRAIN_ROOT_RELATIVE
    copied.parent.mkdir(parents=True)
    shutil.copytree(ROOT / train.PREDECESSOR_TRAIN_ROOT_RELATIVE, copied)
    touched = {"backend": False}

    class ForbiddenBackend:
        def __init__(self, _root):
            touched["backend"] = True

    monkeypatch.setattr(train, "TorchTrainingBackend", ForbiddenBackend)
    with pytest.raises(RuntimeError, match="fresh canonical"):
        train.execute_training(tmp_path)
    assert touched["backend"] is False


def test_identity_or_predecessor_failure_cannot_mint_a_partial_v2_root(tmp_path: Path, monkeypatch):
    def fail_identity(_root):
        raise RuntimeError("synthetic immutable predecessor failure")

    (tmp_path / Path(train.TRAIN_ROOT_RELATIVE).parent).mkdir(parents=True)
    monkeypatch.setattr(train, "production_identity", fail_identity)
    with pytest.raises(RuntimeError, match="predecessor"):
        train.execute_training(tmp_path)
    assert not (tmp_path / train.TRAIN_ROOT_RELATIVE).exists()


def test_mock_success_and_failure_receipts_bind_the_same_predecessor_and_launch_closure(tmp_path: Path):
    identity = train.mock_identity()
    artifact, _, run = _run_mock(tmp_path)
    run()
    for name in ("attempt.json", "launch.json", "throughput2.json", "epoch-00.json", "epoch-01.json", "terminal.json"):
        value = artifact.reload_json(name)
        assert value["predecessor"] == identity.predecessor
        if name == "launch.json":
            assert value["closures"] == identity.closures
        elif name not in {"attempt.json", "terminal.json"}:
            assert value["launch_closure"] == identity.closures
    terminal = artifact.reload_json("terminal.json")
    assert terminal["launch_closure"] == terminal["final_closure"] == identity.closures
    for name in ("checkpoint-00.pt", "checkpoint-01.pt", "swa.pt"):
        value = json.loads(artifact.reload_pair(name))
        assert value["binding"]["predecessor"] == identity.predecessor
        assert value["binding"]["launch_closure"] == identity.closures

    failure_root = tmp_path / "failure"
    failure_root.mkdir()
    failure_artifact, _, failure_run = _run_mock(failure_root, failure="after_step")
    with pytest.raises(RuntimeError):
        failure_run()
    failure = failure_artifact.reload_json("failure.json")
    assert failure["predecessor"] == identity.predecessor
    assert failure["closures"] == identity.closures
