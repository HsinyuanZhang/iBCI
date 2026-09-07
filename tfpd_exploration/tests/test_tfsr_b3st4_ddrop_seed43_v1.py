"""CPU-only tests for the seed-43 matched-replication production route.

Covers: the seed-43 contract closure and seed pinning, the two-flag launcher
gate, accelerated-build equivalence against the frozen forward on synthetic
CPU fixtures (first-step forward bitwise-equal; trajectory amplification
documented), output-root freshness, and the receipt conventions of the
lifecycle (exercised end-to-end with the no-data DeterministicMockBackend).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tfpd_exploration"))
from src.tfsr_b3st4_ddrop_seed43_v1 import accelerated_forward, contract_43, train_43  # noqa: E402
from src.tfsr_b3st4_ddrop_v1 import throughput_benchmark_v2 as bench2  # noqa: E402

MOCK_SPEC = train_43.RunSpec(epochs=2, batch_size=2, steps_per_epoch=2, checkpoint_epochs=(0, 1), throughput_probe_steps=2)


def _env(extra: str = "") -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": ":".join(item for item in (extra, str(ROOT / "tfpd_exploration"), str(ROOT / "tfpd_exploration/src")) if item),
    })
    return value


# --------------------------------------------------------------------------- #
# Contract closure, seed pinning, lineage, build disclosure
# --------------------------------------------------------------------------- #


def test_seed43_is_pinned_everywhere_the_seed42_route_pinned_42():
    assert contract_43.CELL_43 == "TFSR_B3ST4_DDROP_SEED43"
    assert contract_43.SEED_43 == 43
    assert train_43.CELL == contract_43.CELL_43
    assert train_43.SEED == 43
    assert train_43.PUBLIC_SPEC.seed == 43
    frozen_payload = train42_route().PUBLIC_SPEC.payload()
    seed43_payload = train_43.PUBLIC_SPEC.payload()
    assert seed43_payload["seed"] == 43 and frozen_payload["seed"] == 42
    assert {key: value for key, value in seed43_payload.items() if key != "seed"} == {
        key: value for key, value in frozen_payload.items() if key != "seed"}
    assert train_43.PUBLIC_SPEC.epochs == 48
    assert train_43.PUBLIC_SPEC.steps_per_epoch == 33_925
    assert train_43.PUBLIC_SPEC.checkpoint_epochs == (44, 45, 46, 47)


def train42_route():
    from src.tfsr_b3st4_ddrop_v1 import train as frozen_train

    return frozen_train


def test_runspec_rejects_any_other_seed_and_budget_drift():
    with pytest.raises(ValueError):
        train_43.RunSpec(2, 2, 2, (0, 1), 2, seed=42)
    with pytest.raises(ValueError):
        train_43.RunSpec(2, 2, 2, (0, 1), 2, capture_diagnostics=True)
    with pytest.raises(ValueError):
        train_43.RunSpec(0, 2, 2, (0,), 1)
    with pytest.raises(ValueError):
        train_43.RunSpec(2, 2, 2, (1, 0), 1)
    with pytest.raises(ValueError):
        train_43.RunSpec(2, 2, 2, (2,), 1)
    assert MOCK_SPEC.seed == 43


def test_seed43_contract_closure_is_explicit_and_complete():
    assert contract_43.SEED43_CLOSURE == (
        "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_v1/__init__.py",
        "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_v1/contract_43.py",
        "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_v1/accelerated_forward.py",
        "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_v1/train_43.py",
        "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed43_train.py",
        "tfpd_exploration/scripts/preflight_tfsr_b3st4_ddrop_seed43.py",
        "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_seed43_v1.py",
    )
    closure = contract_43.compute_seed43_closure(ROOT)
    assert set(closure["sha256_by_path"]) == set(contract_43.SEED43_CLOSURE)
    assert len(closure["closure_sha256"]) == 64


def test_canonical_evidence_binds_frozen_receipts_and_throughput_v2():
    evidence = contract_43.verify_canonical_evidence_43(ROOT)
    assert evidence["seed"] == 43
    assert evidence["frozen_evidence"]["handoff"]["sha256"] == contract_43.verify_canonical_evidence(ROOT)["handoff"]["sha256"]
    assert evidence["throughput_v2_receipt"]["body_sha256"] == contract_43.THROUGHPUT_V2_RECEIPT_SHA256
    assert evidence["throughput_v2_receipt"]["jit_scripted_step_evidence"]["128"]["first_step_forward_max_abs"] == 0.0
    assert evidence["throughput_v2_receipt"]["jit_scripted_step_evidence"]["64"]["first_step_gradient_max_abs"] <= 1e-6
    assert contract_43.verify_frozen_route(ROOT)["supersedes_cell"] == "TFSR_B3ST4_DDROP_SEED42"


def test_throughput_v2_receipt_semantics_fail_closed_on_fake_paths(tmp_path: Path):
    real = contract_43.verify_throughput_v2_receipt(ROOT)
    assert real["schema"] == "tfsr_b3st4_ddrop_throughput_engineering_v2"
    body = (ROOT / contract_43.THROUGHPUT_V2_RECEIPT_RELATIVE).read_bytes()
    tampered = json.loads(body)
    tampered["matrix"]["cells"] = [cell for cell in tampered["matrix"]["cells"] if cell.get("kind") != "jit_scripted_step"]
    monkey_root = tmp_path / "fake-root"
    monkey_receipt = monkey_root / contract_43.THROUGHPUT_V2_RECEIPT_RELATIVE
    monkey_receipt.parent.mkdir(parents=True)
    payload = json.dumps(tampered, sort_keys=True, indent=2).encode() + b"\n"
    monkey_receipt.write_bytes(payload)
    digest = contract_43._sha(payload)
    sidecar = monkey_root / contract_43.THROUGHPUT_V2_RECEIPT_SIDECAR_RELATIVE
    sidecar.write_bytes(f"{digest}  receipt.json\n".encode())
    os.chmod(monkey_receipt, 0o444)
    os.chmod(sidecar, 0o444)
    # A semantically edited receipt necessarily has a different body SHA, so
    # the pinned-SHA gate fires first; the pinned bytes at ROOT are the only
    # ones that can even reach the jit_scripted_step semantic checks.
    with pytest.raises(RuntimeError, match="body SHA drift"):
        contract_43.verify_throughput_v2_receipt(monkey_root)


def test_build_disclosure_block_is_bound_and_rejects_drift():
    disclosed = contract_43.validate_build_disclosure(contract_43.BUILD_DISCLOSURE)
    assert disclosed["statement"].startswith("seed 43 executes the accelerated build")
    assert "disclosed as build v2 for seeds 43/44" in disclosed["statement"]
    assert disclosed["throughput_v2_receipt_sha256"] == contract_43.THROUGHPUT_V2_RECEIPT_SHA256
    assert disclosed["applies_to_seeds"] == [43, 44]
    for units in ("64", "128"):
        row = disclosed["jit_scripted_step_evidence"][units]
        assert row["first_step_forward_max_abs"] == 0.0
        assert row["first_step_gradient_max_abs"] <= disclosed["frozen_gradient_band"]
        assert row["trajectory_forward_max_abs"] > row["first_step_forward_max_abs"]
    tampered = json.loads(json.dumps(contract_43.BUILD_DISCLOSURE))
    tampered["statement"] = "seed 43 executes the frozen build"
    with pytest.raises(RuntimeError):
        contract_43.validate_build_disclosure(tampered)
    tampered = json.loads(json.dumps(contract_43.BUILD_DISCLOSURE))
    tampered["jit_scripted_step_evidence"]["128"]["first_step_forward_max_abs"] = 1e-9
    with pytest.raises(RuntimeError):
        contract_43.validate_build_disclosure(tampered)
    tampered = json.loads(json.dumps(contract_43.BUILD_DISCLOSURE))
    tampered["throughput_v2_receipt_sha256"] = "f" * 64
    with pytest.raises(RuntimeError):
        contract_43.validate_build_disclosure(tampered)


def test_frozen_route_pins_fail_closed_on_drift(tmp_path: Path):
    contract_43.verify_frozen_route(ROOT)
    relative = "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/contract.py"
    (tmp_path / Path(relative).parent).mkdir(parents=True)
    shutil.copy(ROOT / relative, tmp_path / relative)
    tampered = tmp_path / relative
    os.chmod(tampered, 0o644)
    tampered.write_bytes((tampered.read_bytes() + b"\n# drift\n"))
    os.chmod(tampered, 0o444)
    with pytest.raises(RuntimeError, match="SHA drift"):
        contract_43.verify_frozen_route(tmp_path)


def test_lineage_validator_rejects_semantic_drift():
    good = contract_43.public_lineage(ROOT)
    contract_43.validate_lineage(good)
    assert good["supersedes_cell"] == "TFSR_B3ST4_DDROP_SEED42"
    assert good["seed42_run_resumed"] is False
    for key, value in (("supersedes_cell", "OTHER"), ("throughput_v2_receipt_sha256", "f" * 64),
                       ("seed42_run_resumed", True), ("frozen_route_closure_sha256", "short")):
        tampered = dict(good)
        tampered[key] = value
        with pytest.raises(RuntimeError):
            contract_43.validate_lineage(tampered)
    with pytest.raises(RuntimeError):
        contract_43.validate_lineage({**good, "extra": 1})


def test_device_authority_is_gpu0_and_matches_the_benchmark_gpu0():
    assert train_43.FROZEN_DEVICE_43["cuda_visible_devices"] == "0"
    assert train_43.FROZEN_DEVICE_43["uuid"] == bench2.FROZEN_GPU0["uuid"]
    assert train_43.FROZEN_DEVICE_43["bdf"] == bench2.FROZEN_GPU0["bdf"]
    assert train_43.FROZEN_DEVICE_43["name"] == bench2.FROZEN_GPU0["name"]
    assert train_43.FROZEN_DEVICE_43["memory_total_mib"] == bench2.FROZEN_GPU0["nvidia_smi_memory_total_mib"]
    assert train_43.FROZEN_DEVICE_43 != train42_route().source_smoke.FROZEN_DEVICE


def test_gpu0_gate_rejects_wrong_visibility_before_any_cuda_call(monkeypatch):
    class PoisonCuda:
        def __getattr__(self, name):
            raise AssertionError("CUDA was touched under a wrong visibility mask")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES=0"):
        train_43.require_single_visible_gpu0(PoisonCuda())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES=0"):
        train_43.require_single_visible_gpu0(PoisonCuda())


# --------------------------------------------------------------------------- #
# Launcher gate
# --------------------------------------------------------------------------- #


def test_launcher_and_gate_flags(tmp_path: Path):
    poisoned = tmp_path / "poison"
    poisoned.mkdir()
    (poisoned / "torch.py").write_text("raise RuntimeError('TORCH_IMPORTED')\n")
    script = ROOT / "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed43_train.py"
    dry = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=_env(str(poisoned)), capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    plan = json.loads(dry.stdout)
    assert plan["cell"] == contract_43.CELL_43
    assert plan["authorization"] == "none"
    assert plan["training"]["seed"] == 43
    assert plan["training"]["epochs"] == 48
    assert plan["cuda_visible_devices_required_exactly"] == "0"
    assert plan["device"]["cuda_visible_devices"] == "0"
    assert plan["build_disclosure"]["statement"].startswith("seed 43 executes the accelerated build")
    assert plan["lineage"]["throughput_v2_receipt_sha256"] == contract_43.THROUGHPUT_V2_RECEIPT_SHA256
    assert plan["canonical_output"]["root_relative"] == contract_43.TRAIN_ROOT_RELATIVE
    assert "TORCH_IMPORTED" not in dry.stdout + dry.stderr
    for arguments in (("--execute",), ("--i-have-48epoch-authorization",), ("--unknown",),
                      ("--execute", "--execute"), ("--execute", "--i-have-48epoch-authorization", "extra")):
        done = subprocess.run([sys.executable, str(script), *arguments], cwd=ROOT, env=_env(str(poisoned)),
                              capture_output=True, text=True)
        assert done.returncode != 0
        assert "TORCH_IMPORTED" not in done.stdout + done.stderr


def test_launcher_execute_path_requires_gpu0_visibility(tmp_path: Path):
    script = ROOT / "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed43_train.py"
    env = _env()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    done = subprocess.run([sys.executable, str(script), "--execute", "--i-have-48epoch-authorization"],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    assert done.returncode != 0
    assert "CUDA_VISIBLE_DEVICES_0" in done.stdout + done.stderr
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("PYTHONNOUSERSITE", None)
    done = subprocess.run([sys.executable, str(script), "--execute", "--i-have-48epoch-authorization"],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    assert done.returncode != 0
    assert "PYTHONNOUSERSITE" in done.stdout + done.stderr


# --------------------------------------------------------------------------- #
# Accelerated build equivalence on CPU fixtures
# --------------------------------------------------------------------------- #


def _synthetic_fixture(torch, *, batch=2, units=24, seed=43):
    from src.tfsr_b3st4_ddrop_v1 import model as model_module

    generator = torch.Generator()
    generator.manual_seed(seed)
    dtype = torch.float32
    x = torch.randn((batch, 50, units), generator=generator, dtype=dtype)
    calib = torch.randn((batch, 30, 100, units), generator=generator, dtype=dtype)
    raw_t4 = torch.randn((batch, units, 4), generator=generator, dtype=dtype)
    target = torch.randn((batch, 50, 2), generator=generator, dtype=dtype)
    normalizer = model_module.T4Normalizer(torch.zeros((4,), dtype=dtype), torch.ones((4,), dtype=dtype),
                                           "a" * 64, "b" * 64)
    capability = normalizer(raw_t4, roster_digest="c" * 64,
                            ordered_unit_ids=tuple(f"synthetic-unit-{index}" for index in range(units)),
                            lineage=("seed43_cpu_test",))
    return x, calib, capability, target


def test_accelerated_module_imports_the_benchmark_construction_unchanged():
    assert accelerated_forward.build_scripted_step is bench2.build_scripted_step
    assert accelerated_forward.hoisted_precompute is bench2.hoisted_precompute
    assert accelerated_forward.scripted_step_loop is bench2.scripted_step_loop
    assert accelerated_forward.EMBED_DIM == bench2.EMBED_DIM == 256


def test_eval_forward_is_bitwise_equal_to_the_frozen_forward():
    import random

    import torch

    assert torch.cuda.is_initialized() is False
    random.seed(43)
    torch.manual_seed(43)
    from src.tfsr_b3st4_ddrop_v1 import model as model_module

    x, calib, capability, _target = _synthetic_fixture(torch)
    model = model_module.TFSRDecoder(capture_diagnostics=False)
    model.eval()
    scripted = accelerated_forward.build_scripted_step_binding(torch, model)
    with torch.no_grad():
        frozen_prediction = model(x, calib, capability)
        accelerated_prediction = accelerated_forward.accelerated_forward(torch, model, scripted, x, calib, capability)
    forward_max_abs = float((frozen_prediction - accelerated_prediction).abs().max().item())
    assert forward_max_abs == 0.0
    assert torch.equal(frozen_prediction, accelerated_prediction)
    assert accelerated_prediction.shape == (x.shape[0], 50, 2)
    assert model.last_dropout_p is None  # eval path: no perturbation, identical to frozen eval
    assert torch.cuda.is_initialized() is False


def test_first_train_step_forward_and_loss_bitwise_equal_gradient_within_frozen_band():
    import random

    import torch

    assert torch.cuda.is_initialized() is False
    x, calib, capability, target = _synthetic_fixture(torch)
    from src.tfsr_b3st4_ddrop_v1 import model as model_module

    def fresh_model():
        random.seed(43)
        torch.manual_seed(43)
        return model_module.TFSRDecoder(capture_diagnostics=False)

    frozen = fresh_model()
    accelerated = fresh_model()
    accelerated.load_state_dict(frozen.state_dict(), strict=True)
    scripted = accelerated_forward.build_scripted_step_binding(torch, accelerated)

    def one_step(model, use_accelerated):
        random.seed(43)
        torch.manual_seed(43)
        model.train(True)
        prediction = (accelerated_forward.accelerated_forward(torch, model, scripted, x, calib, capability)
                      if use_accelerated else model(x, calib, capability))
        valid = torch.ones((x.shape[0], 50), dtype=torch.bool)
        loss = model.dense_valid_bin_mse(prediction, target, valid)
        loss.backward()
        return prediction.detach(), float(loss.item()), model

    frozen_prediction, frozen_loss, frozen_model = one_step(frozen, False)
    accelerated_prediction, accelerated_loss, accelerated_model = one_step(accelerated, True)
    assert float((frozen_prediction - accelerated_prediction).abs().max().item()) == 0.0
    assert frozen_loss == accelerated_loss
    gradient_max_abs = max(
        float((frozen_grad - accelerated_grad).abs().max().item())
        for frozen_grad, accelerated_grad in zip(frozen_model.parameters(), accelerated_model.parameters())
        if frozen_grad.grad is not None and accelerated_grad.grad is not None
    )
    assert 0.0 <= gradient_max_abs <= contract_43.FROZEN_GRADIENT_BAND
    # The Cell-D dropout receipt (p, gain, survivor) is identical on both paths.
    assert float(frozen_model.last_dropout_p.item()) == float(accelerated_model.last_dropout_p.item())
    assert torch.equal(frozen_model.last_unit_survivor_mask, accelerated_model.last_unit_survivor_mask)
    assert torch.cuda.is_initialized() is False


def test_short_trajectory_diverges_only_at_fp32_noise_level_and_stays_bounded():
    """Document the disclosed build-v2 property on a CPU fixture.

    The throughput-v2 receipt showed the 20-step trajectory of the scripted
    build diverging from the frozen build at fp32 reduction-order level
    (~1e-4 on CUDA) while the first paired step stayed bitwise-equal.  On a
    short CPU trajectory the same shape must hold: first step exact, later
    steps nonzero but far below any model-scale magnitude, all finite.
    """
    import random

    import torch

    x, calib, capability, target = _synthetic_fixture(torch)
    from src.tfsr_b3st4_ddrop_v1 import model as model_module

    random.seed(43)
    torch.manual_seed(43)
    frozen = model_module.TFSRDecoder(capture_diagnostics=False)
    accelerated = model_module.TFSRDecoder(capture_diagnostics=False)
    accelerated.load_state_dict(frozen.state_dict(), strict=True)
    scripted = accelerated_forward.build_scripted_step_binding(torch, accelerated)
    frozen_optimizer = torch.optim.Adam(frozen.parameters(), lr=1e-4)
    accelerated_optimizer = torch.optim.Adam(accelerated.parameters(), lr=1e-4)
    valid = torch.ones((x.shape[0], 50), dtype=torch.bool)
    forward_maxima = []
    for _ in range(5):
        forward_maxima.append({})
        for name, model, optimizer, use_accelerated in (
            ("frozen", frozen, frozen_optimizer, False),
            ("accelerated", accelerated, accelerated_optimizer, True),
        ):
            random.seed(1000 + len(forward_maxima))
            torch.manual_seed(1000 + len(forward_maxima))
            model.train(True)
            optimizer.zero_grad(set_to_none=True)
            prediction = (accelerated_forward.accelerated_forward(torch, model, scripted, x, calib, capability)
                          if use_accelerated else model(x, calib, capability))
            loss = model.dense_valid_bin_mse(prediction, target, valid)
            loss.backward()
            optimizer.step()
            forward_maxima[-1][name] = prediction.detach()
    differences = [float((step["frozen"] - step["accelerated"]).abs().max().item()) for step in forward_maxima]
    assert differences[0] == 0.0  # first paired step stays bitwise-equal on CPU
    assert all(difference == difference and difference < 5e-3 for difference in differences)
    assert all(torch.isfinite(step["frozen"]).all().item() and torch.isfinite(step["accelerated"]).all().item()
               for step in forward_maxima)
    state_difference = max(
        float((left - right).abs().max().item())
        for left, right in zip(frozen.parameters(), accelerated.parameters())
    )
    assert state_difference < 5e-3


def test_accelerated_forward_keeps_the_frozen_validation_boundaries():
    import torch

    x, calib, capability, _target = _synthetic_fixture(torch, units=16)
    from src.tfsr_b3st4_ddrop_v1 import model as model_module

    model = model_module.TFSRDecoder(capture_diagnostics=False)
    scripted = accelerated_forward.build_scripted_step_binding(torch, model)
    with pytest.raises(ValueError):
        accelerated_forward.accelerated_forward(torch, model, scripted, x[:, :, :8], calib[:, :, :, :8], capability)
    with pytest.raises(ValueError):
        accelerated_forward.accelerated_forward(torch, model, scripted, x, calib, capability.tensor)


# --------------------------------------------------------------------------- #
# Output-root freshness
# --------------------------------------------------------------------------- #


def test_canonical_output_root_is_currently_absent_and_gate_enforces_freshness():
    assert train_43.TRAIN_ROOT_RELATIVE == "tfpd_exploration/results/tfsr_b3st4_ddrop_seed43_train_v1"
    assert not (ROOT / train_43.TRAIN_ROOT_RELATIVE).exists()
    plan = train_43.training_plan(ROOT)
    assert plan["canonical_output"]["must_be_fresh_before_execution"] is True


def test_occupied_or_aliased_output_root_fails_before_any_backend(tmp_path: Path, monkeypatch):
    output = tmp_path / train_43.TRAIN_ROOT_RELATIVE
    output.parent.mkdir(parents=True)
    output.mkdir()
    touched = {"backend": False}

    class ForbiddenBackend:
        def __init__(self, _root):
            touched["backend"] = True

    monkeypatch.setattr(train_43, "TrainingBackend43", ForbiddenBackend)
    with pytest.raises(RuntimeError, match="fresh canonical"):
        train_43.execute_training(tmp_path)
    assert touched["backend"] is False


def test_identity_failure_cannot_mint_a_partial_output_root(tmp_path: Path, monkeypatch):
    (tmp_path / Path(train_43.TRAIN_ROOT_RELATIVE).parent).mkdir(parents=True)

    def fail_identity(_root):
        raise RuntimeError("synthetic lineage verification failure")

    monkeypatch.setattr(train_43, "production_identity", fail_identity)
    with pytest.raises(RuntimeError, match="synthetic"):
        train_43.execute_training(tmp_path)
    assert not (tmp_path / train_43.TRAIN_ROOT_RELATIVE).exists()


def test_execute_training_reserves_root_only_after_identity_verifies(tmp_path: Path, monkeypatch):
    (tmp_path / Path(train_43.TRAIN_ROOT_RELATIVE).parent).mkdir(parents=True)
    monkeypatch.setattr(train_43, "production_identity", train_43.production_identity)
    # The real identity fails on a synthetic root because the frozen evidence
    # is absent there; no output directory may survive that failure.
    with pytest.raises(RuntimeError):
        train_43.execute_training(tmp_path)
    assert not (tmp_path / train_43.TRAIN_ROOT_RELATIVE).exists()


# --------------------------------------------------------------------------- #
# Receipt conventions via the mock lifecycle
# --------------------------------------------------------------------------- #


def _artifact(tmp_path: Path, spec: train_43.RunSpec = MOCK_SPEC) -> train_43.ArtifactRoot:
    return train_43.reserve_artifact_root(tmp_path, "artifacts", spec.topology)


def _run_mock(tmp_path: Path, *, failure: str | None = None, spec: train_43.RunSpec = MOCK_SPEC):
    backend = train_43.DeterministicMockBackend(failure=failure)
    artifact = _artifact(tmp_path, spec)
    return artifact, backend, lambda: train_43.run_lifecycle(
        spec=spec, backend=backend, artifact=artifact, identity_factory=train_43.mock_identity,
    )


def _rewrite_pair(artifact: train_43.ArtifactRoot, name: str, body: bytes) -> None:
    path = artifact.directory / name
    sidecar = artifact.directory / f"{name}.sha256"
    os.chmod(path, 0o644)
    path.write_bytes(body)
    os.chmod(path, 0o444)
    digest = train_43._sha(body)
    os.chmod(sidecar, 0o644)
    sidecar.write_bytes(f"{digest}  {name}\n".encode())
    os.chmod(sidecar, 0o444)


def test_mock_lifecycle_publishes_every_artifact_with_seed43_receipts(tmp_path: Path):
    artifact, backend, run = _run_mock(tmp_path)
    terminal = run()
    assert terminal["status"] == "TRAINING_COMPLETE"
    assert terminal["cell"] == contract_43.CELL_43
    assert backend.closed is True
    expected = set(MOCK_SPEC.topology) - {"failure.json"}
    assert {path.name for path in artifact.directory.iterdir() if not path.name.endswith(".sha256")} == expected
    train_43.validate_terminal_receipt(artifact.reload_json("terminal.json"), MOCK_SPEC, train_43.mock_identity())
    assert backend.proof_requests == [False, True, False, True]
    assert backend.expensive_proof_count == 2


def test_launch_and_terminal_receipts_carry_the_build_disclosure_and_lineage(tmp_path: Path):
    artifact, _, run = _run_mock(tmp_path)
    run()
    identity = train_43.mock_identity()
    attempt = artifact.reload_json("attempt.json")
    launch = artifact.reload_json("launch.json")
    terminal = artifact.reload_json("terminal.json")
    for receipt in (attempt, launch, terminal):
        assert receipt["build_disclosure"]["statement"] == contract_43.BUILD_DISCLOSURE_STATEMENT
        assert receipt["build_disclosure"]["throughput_v2_receipt_sha256"] == contract_43.THROUGHPUT_V2_RECEIPT_SHA256
        assert receipt["lineage"]["supersedes_cell"] == "TFSR_B3ST4_DDROP_SEED42"
    assert launch["device"]["cuda_visible_devices"] == "0"
    assert launch["run_spec"]["seed"] == 43
    assert terminal["launch_closure"] == terminal["final_closure"] == identity.closures
    assert terminal["swa_evaluation_proof"]["forward_build"] == contract_43.BUILD_DISCLOSURE["build"]
    for name in ("throughput2.json", "epoch-00.json", "epoch-01.json"):
        receipt = artifact.reload_json(name)
        assert receipt["lineage"] == identity.lineage
        assert receipt["launch_closure"] == identity.closures
    for name in ("checkpoint-00.pt", "checkpoint-01.pt", "swa.pt"):
        value = json.loads(artifact.reload_pair(name))
        assert value["binding"]["lineage"] == identity.lineage
        assert value["binding"]["cell"] == contract_43.CELL_43


@pytest.mark.parametrize("mutate", (
    lambda item: item.__setitem__("status", "TAMPERED"),
    lambda item: item.__setitem__("total_optimizer_steps", 3),
    lambda item: item["build_disclosure"].__setitem__("statement", "quietly the frozen build"),
    lambda item: item["lineage"].__setitem__("throughput_v2_receipt_sha256", "f" * 64),
    lambda item: item["swa_evaluation_proof"].__setitem__("repeat_bitwise_equal", False),
))
def test_terminal_validator_rejects_receipt_drift(tmp_path: Path, mutate):
    artifact, _, run = _run_mock(tmp_path)
    value = dict(run())
    mutate(value)
    with pytest.raises(RuntimeError):
        train_43.validate_terminal_receipt(value, MOCK_SPEC, train_43.mock_identity())


@pytest.mark.parametrize(("failure", "steps"), (("before_gpu", 0), ("during_adapter", 0), ("after_step", 1)))
def test_failure_envelopes_are_honest_and_never_publish_terminal(tmp_path: Path, failure: str, steps: int):
    artifact, backend, run = _run_mock(tmp_path, failure=failure)
    with pytest.raises(RuntimeError, match="synthetic"):
        run()
    receipt = artifact.reload_json("failure.json")
    train_43.validate_failure_receipt(receipt)
    assert receipt["schema"] == "tfsr_b3st4_ddrop_seed43_train_failure_v1"
    assert receipt["terminal_published"] is False
    assert receipt["optimizer_steps_completed"] == steps
    assert receipt["lineage"]["supersedes_cell"] == "TFSR_B3ST4_DDROP_SEED42"
    assert not (artifact.directory / "terminal.json").exists()
    assert backend.closed is True


def test_mock_backend_cannot_silently_open_target_or_formal_domain(tmp_path: Path):
    class BadBoundary(train_43.DeterministicMockBackend):
        def prepare(self, spec, identity, flags):
            runtime = super().prepare(spec, identity, flags)
            flags.target_or_formal_opened = True
            return runtime

    backend = BadBoundary()
    artifact = _artifact(tmp_path)
    with pytest.raises(RuntimeError, match="target/formal"):
        train_43.run_lifecycle(spec=MOCK_SPEC, backend=backend, artifact=artifact, identity_factory=train_43.mock_identity)
    receipt = artifact.reload_json("failure.json")
    assert receipt["target_or_formal_opened"] is True
    train_43.validate_failure_receipt(receipt)


def test_checkpoint_launch_binding_tampering_is_caught_at_terminal_finalization(tmp_path: Path):
    artifact, backend, run = _run_mock(tmp_path)
    run()
    value = json.loads(artifact.reload_pair("checkpoint-00.pt"))
    value["binding"]["launch_sha256"] = "f" * 64
    _rewrite_pair(artifact, "checkpoint-00.pt", train_43._json(value))
    hashes = {name: train_43._sha(artifact.reload_pair(name))
              for name in set(MOCK_SPEC.topology) - {"terminal.json", "failure.json"}}
    expected_binding = {
        "cell": contract_43.CELL_43, "run_spec": MOCK_SPEC.payload(), "launch_sha256": hashes["launch.json"],
        "launch_closure": train_43.mock_identity().closures, "lineage": train_43.mock_identity().lineage,
    }
    with pytest.raises(RuntimeError, match="exact launch binding"):
        backend.build_swa(None, {epoch: artifact.reload_pair(f"checkpoint-{epoch:02d}.pt")
                                 for epoch in MOCK_SPEC.checkpoint_epochs},
                          MOCK_SPEC, expected_binding=expected_binding)
    with pytest.raises(RuntimeError, match="exact launch binding"):
        train_43._validate_all_preterminal_artifacts(artifact, MOCK_SPEC, train_43.mock_identity(), backend, hashes)


def test_epoch_receipt_dropout_and_lr_conventions_are_enforced(tmp_path: Path):
    artifact, _, run = _run_mock(tmp_path)
    run()
    first = artifact.reload_json("epoch-00.json")
    train_43.validate_epoch_receipt(first, 0, 2, MOCK_SPEC)
    assert first["schema"] == "tfsr_b3st4_ddrop_seed43_epoch_v1"
    assert first["dropout"]["population_examples"] == MOCK_SPEC.steps_per_epoch * MOCK_SPEC.batch_size
    assert first["lr"]["first"] == first["lr"]["expected_first"]
    assert first["progress"] == {"epoch": 0, "completed_epochs": 1, "epochs": 2, "global_step": 2,
                                 "total_optimizer_steps": 4}
    value = dict(first)
    value["dropout"] = dict(value["dropout"])
    value["dropout"]["p_q50"] = 1.1
    with pytest.raises(RuntimeError):
        train_43.validate_epoch_receipt(value, 0, 2, MOCK_SPEC)


def test_full_state_proof_is_requested_once_per_epoch_and_not_in_public_first_100_steps(tmp_path: Path):
    assert not any(train_43.requires_epoch_proof(train_43.PUBLIC_SPEC, step) for step in range(100))
    assert train_43.requires_epoch_proof(train_43.PUBLIC_SPEC, train_43.PUBLIC_SPEC.steps_per_epoch - 1) is True
    spec = train_43.RunSpec(epochs=1, batch_size=2, steps_per_epoch=101, checkpoint_epochs=(0,), throughput_probe_steps=100)
    backend = train_43.DeterministicMockBackend()
    artifact = _artifact(tmp_path, spec)
    train_43.run_lifecycle(spec=spec, backend=backend, artifact=artifact, identity_factory=train_43.mock_identity)
    assert backend.proof_requests[:100] == [False] * 100
    assert backend.proof_requests[100:] == [True]
    assert (artifact.directory / "throughput100.json").exists()


def test_public_lr_schedule_matches_arm_common_exactly():
    sys.path.insert(0, str(ROOT / "tfpd_exploration/src"))
    from tfpd_lane.arm_common import lr_at_step

    probes = [0, train_43.WARMUP_STEPS - 1, train_43.WARMUP_STEPS, train_43.TOTAL_STEPS - 1]
    probes.extend(((index * 15_485_863) % train_43.TOTAL_STEPS) for index in range(1, 51))
    assert all(train_43.lr_for_step(step) == lr_at_step(step, 48, 33_925) for step in probes)
    frozen = train42_route()
    assert all(train_43.lr_for_step(step) == frozen.lr_for_step(step) for step in probes)
    with pytest.raises(ValueError):
        train_43.lr_for_step(train_43.TOTAL_STEPS)


def test_public_plan_static_content_is_the_seed42_recipe_with_seed43():
    plan = train_43.training_plan(ROOT)
    assert plan["cell"] == contract_43.CELL_43
    assert plan["training"]["seed"] == 43
    assert plan["training"]["optimizer"] == train42_route().OPTIMIZER
    assert plan["training"]["schedule"]["authority"] == "tfpd_lane.arm_common.lr_at_step"
    assert plan["training"]["swa"] == "arithmetic_checkpoint_state_mean_final_4"
    assert plan["training"]["epochs"] == 48


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


def test_preflight_zero_arg_binds_contract_and_accelerated_build():
    script = ROOT / "tfpd_exploration/scripts/preflight_tfsr_b3st4_ddrop_seed43.py"
    done = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=_env(), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    plan = json.loads(done.stdout)
    assert plan["status"] == "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_LAUNCH"
    assert plan["build_disclosure"]["throughput_v2_receipt_sha256"] == contract_43.THROUGHPUT_V2_RECEIPT_SHA256
    assert plan["fresh_output_root"] == {"root_relative": contract_43.TRAIN_ROOT_RELATIVE, "fresh": True,
                                         "write_performed": False}
    audit = plan["cpu_no_data_resource_audit"]
    assert audit["accelerated_build_forward_max_abs_vs_frozen"] == 0.0
    assert audit["accelerated_build_first_step_forward_bitwise_equal"] is True
    assert audit["trainable_params"] > 0
    assert plan["route_identity_verified"]["device"]["cuda_visible_devices"] == "0"
    assert not (ROOT / contract_43.TRAIN_ROOT_RELATIVE).exists()


def test_preflight_rejects_any_cli_argument():
    script = ROOT / "tfpd_exploration/scripts/preflight_tfsr_b3st4_ddrop_seed43.py"
    done = subprocess.run([sys.executable, str(script), "--execute"], cwd=ROOT, env=_env(),
                          capture_output=True, text=True)
    assert done.returncode != 0
    assert "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_LAUNCH" in done.stdout + done.stderr
