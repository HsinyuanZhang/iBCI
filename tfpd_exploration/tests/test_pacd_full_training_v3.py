"""No-data/no-CUDA contract tests for PACD full V3."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.paired_anchored_calibration_dropout_full_v1 import runner
from src.paired_anchored_calibration_dropout_full_v1 import smoke as shared
from src.paired_anchored_calibration_dropout_full_v3 import plan, predecessor, smoke


def _leaf(path: Path, body: bytes) -> str:
    path.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    os.chmod(path, 0o444)
    side = path.with_name(path.name + ".sha256")
    side.write_text(f"{digest}  {path.name}\n")
    os.chmod(side, 0o444)
    return digest


def _v2_failure_graph(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    """Synthetic bytes obey V3's semantic graph, with locally rebound SHAs."""
    out = tmp_path / "v2_failure"
    out.mkdir()
    closure = "74b9495597fa8ca1b2a6731d78c2fb10108e47e2fc4cc00540b34d68c377763c"
    attempt = {"schema": "pacd_matched_full_training_v2_attempt", "arm": "p0",
               "source_closure": {"closure_sha256": closure}}
    attempt_sha = _leaf(out / "attempt.json", (json.dumps(attempt) + "\n").encode())
    authority = {"target_access": False, "attempt_sha256": attempt_sha}
    authority_sha = _leaf(out / "source_authority.json", (json.dumps(authority) + "\n").encode())
    launch = {"schema": "pacd_matched_full_training_v2_launch", "target_access": False,
              "attempt": {"sha256": attempt_sha}, "source_authority": {"sha256": authority_sha}}
    launch_sha = _leaf(out / "launch.json", (json.dumps(launch) + "\n").encode())
    failure = {
        "schema": "pacd_matched_full_training_v2_failure", "status": "CELL_FAILED",
        "failure": {"kind": "PACDError", "detail": "encoder gradient is zero"},
        "progress": {"epochs_published": 0, "checkpoints_published": 0, "swa_published": False},
        "target_access": False, "source_closure": {"launch": {"closure_sha256": closure},
                                                        "final": {"closure_sha256": closure}},
        "predecessor": predecessor._V1_PREDECESSOR,
    }
    failure_sha = _leaf(out / "failure.json", (json.dumps(failure) + "\n").encode())
    shas = {"attempt.json": attempt_sha, "launch.json": launch_sha,
            "source_authority.json": authority_sha, "failure.json": failure_sha}
    monkeypatch.setattr(plan, "V2_FAILURE_RELATIVE", "v2_failure")
    monkeypatch.setattr(plan, "V2_FAILURE_SHAS", shas)
    return out, shas


def test_v3_held_fd_predecessor_accepts_and_rejects_semantic_drift(tmp_path, monkeypatch):
    out, shas = _v2_failure_graph(tmp_path, monkeypatch)
    accepted = predecessor.validate_v2_failure(tmp_path, "v2_failure")
    assert accepted["failure_sha256"] == shas["failure.json"]
    assert accepted["topology"] == list(predecessor.LEAVES)
    # Rebind the literal only after changing the body, so this proves a
    # semantic (not merely byte-hash) fail-closed check.
    failure = json.loads((out / "failure.json").read_text())
    failure["failure"]["detail"] = "different failure"
    os.chmod(out / "failure.json", 0o644)
    os.chmod(out / "failure.json.sha256", 0o644)
    replacement = _leaf(out / "failure.json", (json.dumps(failure) + "\n").encode())
    monkeypatch.setattr(plan, "V2_FAILURE_SHAS", {**shas, "failure.json": replacement})
    with pytest.raises(predecessor.PredecessorError, match="failure cause"):
        predecessor.validate_v2_failure(tmp_path, "v2_failure")


def test_v3_held_fd_predecessor_rejects_extra_and_mode(tmp_path, monkeypatch):
    out, _ = _v2_failure_graph(tmp_path, monkeypatch)
    (out / "extra").write_text("x")
    with pytest.raises(predecessor.PredecessorError, match="topology"):
        predecessor.validate_v2_failure(tmp_path, "v2_failure")
    (out / "extra").unlink()
    os.chmod(out / "launch.json", 0o644)
    with pytest.raises(predecessor.PredecessorError, match="invalid mode/type"):
        predecessor.validate_v2_failure(tmp_path, "v2_failure")


def test_v3_profile_is_frozen_and_cli_is_inert():
    assert smoke.V3_EXECUTION_PROFILE.identity == "full-v3"
    assert smoke.V3_EXECUTION_PROFILE.plan.ZERO_ENCODER_POLICY == "accept_if_paired_unit_mask_empty"
    with pytest.raises(Exception):
        smoke.V3_EXECUTION_PROFILE.identity = "mutated"
    result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/run_pacd_full_training_v3.py")],
                            capture_output=True, text=True)
    assert result.returncode == 0 and plan.CELL in result.stdout
    result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/run_pacd_full_training_v3.py"), "--execute"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    probe = subprocess.run([
        sys.executable, "-S", "-c",
        "import runpy,sys; runpy.run_path(sys.argv[1],run_name='v3_dry_probe'); "
        "assert 'torch' not in sys.modules and 'torch.cuda' not in sys.modules",
        str(ROOT / "scripts/run_pacd_full_training_v3.py"),
    ], capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr


class _Tensor:
    def to(self, device):
        return self


class _Cuda:
    def synchronize(self, device):
        pass


class _Torch:
    def __init__(self):
        self.cuda = _Cuda()


class _Opt:
    param_groups = [{}]


class _ArmCommon:
    @staticmethod
    def lr_at_step(step, epochs, steps):
        return 1e-3


def _epoch_evidence(*, zero_encoder: bool) -> dict:
    return {
        "loss_anchor": 1., "loss_short": 1., "loss_combined": 1.,
        "anchor_encoder_grad_norm": 0.1, "anchor_decoder_grad_norm": 0.1,
        "short_encoder_grad_norm": 0.1, "short_decoder_grad_norm": 0.1,
        "combined_encoder_grad_norm": 0. if zero_encoder else 0.2,
        "combined_decoder_grad_norm": 0.3,
        "combined_encoder_zero_accepted": zero_encoder,
        "combined_encoder_zero_reason": "all_units_dropped_valid_zero" if zero_encoder else None,
        "paired_unit_mask_empty": zero_encoder,
        "dropout": {"p": 0.2}, "rng_before_sha256": "a", "rng_after_pair_sha256": "b",
        "calibration_anchor_sha256": "c", "calibration_short_sha256": "d",
        "calibration_full_sha256": "e", "calibration_full_after_sha256": "e",
        "rng_short_transition_equal": True, "rng_pair_transition_equal": True,
        "prediction_pair_equal": True, "identity_pair_equal": True, "valid_bins": 3,
        "optimizer_steps": 1, "parameter_finiteness": {"materialized": 29, "skipped_uninitialized_lazy": 2},
        "short_encoder_grad_norm": 0.1, "short_decoder_grad_norm": 0.1,
        "encoder_branch_gradient_cosine": 0.5, "decoder_branch_gradient_cosine": 0.5,
    }


def test_v3_epoch_requires_positive_coverage_but_records_valid_zero(tmp_path, monkeypatch):
    del tmp_path
    monkeypatch.setattr(runner.plan, "STEPS_PER_EPOCH", 2)
    monkeypatch.setattr(runner.plan, "SENTINELS", (0, 1))
    rows = iter([_epoch_evidence(zero_encoder=True), _epoch_evidence(zero_encoder=False)])
    monkeypatch.setattr(runner, "paired_train_step", lambda **kwargs: next(rows))
    batch = (_Tensor(), _Tensor(), _Tensor(), ["s"], _Tensor())
    output = runner.run_epoch(model=object(), optimizer=_Opt(), loader=[batch, batch],
                              arm={"name": "p0", "short_m": 30}, device="cpu", torch=_Torch(),
                              arm_common=_ArmCommon(), encoder_parameters=[], decoder_parameters=[],
                              epoch=0, pad_value=-1., steps_per_epoch=2,
                              zero_encoder_policy="accept_if_paired_unit_mask_empty")
    assert output["gradient_coverage"] == {
        "zero_encoder_steps": 1, "accepted_zero_encoder_steps": 1,
        "zero_encoder_reason": "all_units_dropped_valid_zero", "zero_decoder_steps": 0,
        "positive_encoder_steps": 1, "positive_decoder_steps": 2,
    }
    rows = iter([_epoch_evidence(zero_encoder=True), _epoch_evidence(zero_encoder=True)])
    monkeypatch.setattr(runner, "paired_train_step", lambda **kwargs: next(rows))
    with pytest.raises(runner.FullInvariantError, match="all-epoch zero encoder-gradient coverage"):
        runner.run_epoch(model=object(), optimizer=_Opt(), loader=[batch, batch],
                         arm={"name": "p0", "short_m": 30}, device="cpu", torch=_Torch(),
                         arm_common=_ArmCommon(), encoder_parameters=[], decoder_parameters=[],
                         epoch=0, pad_value=-1., steps_per_epoch=2,
                         zero_encoder_policy="accept_if_paired_unit_mask_empty")


def _synthetic_plan() -> SimpleNamespace:
    return SimpleNamespace(
        CELL="PACD_MATCHED_FULL_TRAINING_V3", SCHEMA="pacd_matched_full_training_v3",
        ARMS={"p0": {"short_m": 30, "root": "v3out"}}, BOUND_PATTERNS=(),
        REVIEW_EVIDENCE_PATHS=(), EPOCHS=4, STEPS_PER_EPOCH=1, TOTAL_STEPS=4,
        SEED=42, BATCH_SIZE=32, FINAL_EPOCHS=(0, 1, 2, 3), SENTINELS=(0,),
        EXPECTED_INITIAL_STATE_SHA="state", EXPECTED_BEHAVIOR_NORMALIZER_SHA="behavior",
        EXPECTED_SIDE_NORMALIZER_SHA="side", V2_THROUGHPUT_STEPS_PER_SECOND={"p0": 1.},
        MECHANICAL_PROJECTED_HOURS={"p0": 1.},
        ZERO_ENCODER_POLICY="accept_if_paired_unit_mask_empty",
    )


def test_v3_wrapper_shared_execute_synthetic_success_and_failure(tmp_path, monkeypatch):
    """The V3 wrapper reaches the one shared lifecycle; no copied loop exists."""
    receipt = shared.v1.load_stdlib_receipt_module(REPO)
    gpu, pred, closure = {"physical_index": "0"}, {"v2_failure": "exact"}, {"closure_sha256": "v3closure", "files": {}}
    profile = shared.ExecutionProfile(identity="full-v3-test", plan=_synthetic_plan(), predecessor_validator=lambda root: pred)
    monkeypatch.setattr(shared.v1, "preflight_gpu0_idle", lambda: gpu)
    monkeypatch.setattr(shared.v1, "recheck_gpu0_after_attempt", lambda value: gpu)
    monkeypatch.setattr(shared.v1, "load_stdlib_receipt_module", lambda root: receipt)
    monkeypatch.setattr(shared.v1, "exact_source_closure", lambda root, receipt, patterns: closure)
    monkeypatch.setattr(shared.v1, "verify_expected_sealed_files", lambda root, receipt: {})
    monkeypatch.setattr(shared.v1, "_assert_cuda_binding_after_attempt", lambda torch_value: "cpu")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ps = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(1.)) for _ in range(29)])

        def forward(self, neural, *, calib_trials, side_features):
            return neural.mean(-1, keepdim=True).repeat(1, 1, 2), side_features.mean(-1)

    initial = tmp_path / "initial.pt"
    torch.save({"state_dict": Model().state_dict(), "state_sha256": "state"}, initial)
    (tmp_path / "initial.pt.sha256").write_text(f"{hashlib.sha256(initial.read_bytes()).hexdigest()}  initial.pt\n")
    manifest = tmp_path / "manifest"
    manifest.write_text("m")

    class Dataset:
        sessions = ["s"]
        window_indices = [("s", 0)]

    class DM:
        train_dataset = Dataset()
        session_splits = {"train": [f"s{i}" for i in range(27)]}
        session_files = {"val": [], "test": []}
        _behavior_stats = (torch.tensor(0.), torch.tensor(1.))
        _side_feature_stats = (torch.tensor(0.), torch.tensor(1.))

    class A2:
        MANIFEST_PATH = manifest
        EXPECTED_MANIFEST_SHA256 = hashlib.sha256(b"m").hexdigest()

        @staticmethod
        def normalizer_value_sha256(left, right):
            return "behavior" if left is DM._behavior_stats[0] else "side"

    class AR:
        PAD_VALUE = -1.
        WINDOW_SIZE = 50
        build_datamodule = staticmethod(lambda args: (DM(), A2()))

        @staticmethod
        def seal_file(path):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            os.chmod(path, 0o444)
            side = Path(str(path) + ".sha256")
            side.write_text(f"{digest}  {path.name}\n")
            os.chmod(side, 0o444)

    class AC:
        ADAM_CONSTRUCTOR = {"lr": 1e-3, "betas": [.9, .999], "eps": 1e-8,
                            "weight_decay": 0., "amsgrad": False}
        param_groups_by_branch = staticmethod(lambda model: (list(model.ps), list(model.ps)))
        state_sha256 = staticmethod(lambda model: "state")
        optimizer_sha256 = staticmethod(lambda optimizer: "optimizer")
        t4_authority_fingerprint = staticmethod(lambda sessions: "t4")
        w_side_block = staticmethod(lambda model: torch.zeros(1))

    class PR:
        build_population_robustness_model = staticmethod(lambda **kwargs: Model())

    class MS:
        @staticmethod
        def build_swa_final_four(paths, out):
            torch.save(torch.load(paths[-1], weights_only=False), out)
            return {"inherited": True}

    class Cuda:
        reset_peak_memory_stats = staticmethod(lambda device: None)
        synchronize = staticmethod(lambda device: None)
        memory_allocated = staticmethod(lambda device: 0)
        max_memory_allocated = staticmethod(lambda device: 0)
        max_memory_reserved = staticmethod(lambda device: 0)

    class Factory:
        identity = "v3-synthetic"
        production = False

        def after_attempt(self, root, gpu_value):
            torch_view = type("TorchView", (), {
                "cuda": Cuda(), "backends": torch.backends, "load": staticmethod(torch.load),
                "save": staticmethod(torch.save), "optim": torch.optim,
                "count_nonzero": staticmethod(torch.count_nonzero), "no_grad": staticmethod(torch.no_grad),
            })()
            return type("Runtime", (), {"torch": torch_view,
                "pl": type("PL", (), {"seed_everything": staticmethod(lambda *args, **kwargs: None)})(),
                "device": "cpu", "stack": {"receipt": receipt, "arm_runner": AR,
                "pop_robust": PR, "arm_common": AC, "matched_scorer": MS}})()

        def sampler(self, dataset, **kwargs):
            return type("Sampler", (), {"batched_indices": [[0]], "__len__": lambda self: 1})()

        def loader(self, dataset, sampler):
            return []

    factory = Factory()
    received_policies = []

    def fake_epoch(**kwargs):
        received_policies.append(kwargs["zero_encoder_policy"])
        if kwargs.get("probe_sink"):
            kwargs["probe_sink"]((torch.randn(2, 50, 5), torch.randn(2, 30, 100, 5), torch.randn(2, 5, 4)))
        return {"epoch": kwargs["epoch"], "optimizer_steps": 1, "gradient_coverage": {"positive_encoder_steps": 1}}

    monkeypatch.setattr(shared, "run_epoch", fake_epoch)
    monkeypatch.setattr(shared, "_swa_runtime_proof", lambda **kwargs: {
        "state_before_sha256": "state", "state_after_sha256": "state", "output_finite": True,
    })
    monkeypatch.setattr(shared.v1, "_finite_optimizer_state", lambda torch_value, optimizer: True)
    capability = shared._issue_root_capability_after_preflight(
        root=tmp_path, arm="p0", runtime_factory=factory, profile=profile
    )
    args = SimpleNamespace(seed=42, train_batch_size=32, num_workers=0, initial_state=initial)
    assert smoke.execute(root=tmp_path, arm="p0", args=args, capability=capability,
                         runtime_factory=factory, profile=profile) == 0
    out = tmp_path / "v3out"
    terminal = json.loads((out / "terminal.json").read_text())
    assert terminal["schema"] == "pacd_matched_full_training_v3_terminal"
    assert len(terminal["epochs"]) == 4 and len(terminal["checkpoints"]) == 4
    assert received_policies == ["accept_if_paired_unit_mask_empty"] * 4
    with pytest.raises(RuntimeError, match="already consumed"):
        smoke.execute(root=tmp_path, arm="p0", args=args, capability=capability,
                      runtime_factory=factory, profile=profile)

    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    failure_capability = shared._issue_root_capability_after_preflight(
        root=failed_root, arm="p0", runtime_factory=factory, profile=profile
    )
    monkeypatch.setattr(shared.v1, "recheck_gpu0_after_attempt",
                        lambda value: (_ for _ in ()).throw(RuntimeError("V3 boundary")))
    with pytest.raises(RuntimeError, match="V3 boundary"):
        smoke.execute(root=failed_root, arm="p0", args=args, capability=failure_capability,
                      runtime_factory=factory, profile=profile)
    failed_out = failed_root / "v3out"
    assert (failed_out / "attempt.json").is_file() and (failed_out / "failure.json").is_file()
    assert not (failed_out / "terminal.json").exists()
