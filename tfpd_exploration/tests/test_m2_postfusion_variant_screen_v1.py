"""Focused no-data / no-CUDA constructibility tests for the PF screen."""
from __future__ import annotations

import copy
import os
import random
import subprocess
import sys
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from tfpd_exploration.src.m2_postfusion_variant_screen_v1 import controller, plan
from tfpd_exploration.src.m2_postfusion_variant_screen_v1 import variants
from tfpd_exploration.src.m2_postfusion_variant_screen_v1 import lifecycle, selection
from tfpd_exploration.src.m2_postfusion_variant_screen_v1 import runner as screen_runner
from tfpd_exploration.src.m2_postfusion_variant_screen_v1 import driver
from tfpd_exploration.src.m2_postfusion_probe_v1 import memory as probe_memory
from streaming_calibration_exp.src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder


REPO_ROOT = Path(__file__).resolve().parents[2]


def _native() -> SideFeatureEarlyPoolEncoder:
    torch.manual_seed(19)
    return SideFeatureEarlyPoolEncoder(
        trial_length=plan.TRIAL_LENGTH, window_size=plan.WINDOW_BINS,
        hidden_dim=64, side_dim=plan.SIDE_DIM, num_post_layers=3,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(23)
    calib = torch.randn(3, 30, plan.TRIAL_LENGTH, 7, dtype=torch.float32)
    side = torch.randn(3, 7, plan.SIDE_DIM, dtype=torch.float32)
    neural = torch.randn(3, plan.WINDOW_BINS, 7, dtype=torch.float32)
    return calib, side, neural


def _postfusion_probe_reference(native, calib, side) -> torch.Tensor:
    values = []
    for index in range(calib.shape[1]):
        identity = probe_memory.per_trial_identity(
            native, torch, calib[0, index].detach().numpy(), side[0:1],
            channels=calib.shape[-1], device=torch.device("cpu"), dtype=torch.float32,
        )
        values.append(identity)
    reference0 = probe_memory.sequential_mean(values)
    # The other two batch rows use the same frozen-probe arithmetic; doing them
    # independently avoids treating batch dimension as a pool dimension.
    remaining = []
    for batch in range(1, calib.shape[0]):
        per_trial = [probe_memory.per_trial_identity(
            native, torch, calib[batch, index].detach().numpy(), side[batch:batch + 1],
            channels=calib.shape[-1], device=torch.device("cpu"), dtype=torch.float32,
        ) for index in range(calib.shape[1])]
        remaining.append(probe_memory.sequential_mean(per_trial))
    return torch.cat([reference0, *remaining], dim=0)


def test_static_authority_and_cycle_are_exact_without_cuda() -> None:
    assert plan.validate_static_authority(REPO_ROOT)["workorder_sha256"] == plan.WORKORDER_SHA256
    assert plan.validate_pool_cycle() == (30, 10, 4)
    assert not torch.cuda.is_initialized()
    with pytest.raises(plan.PlanError):
        plan.validate_pool_cycle((30, 4, 10))


def test_explicit_execution_closure_covers_direct_runtime_import_trace() -> None:
    """No-glob route closure contains every reviewed direct runtime leaf.

    This is deliberately a source-only import trace: importing the actual PIT
    runtime would pull Torch and its source/data construction stack, which is
    outside the no-CUDA constructibility boundary.
    """
    required = {
        "tfpd_exploration/src/m2_postfusion_variant_screen_v1/plan.py",
        "tfpd_exploration/src/m2_postfusion_variant_screen_v1/controller.py",
        "tfpd_exploration/src/m2_postfusion_variant_screen_v1/variants.py",
        "tfpd_exploration/src/m2_postfusion_variant_screen_v1/selection.py",
        "tfpd_exploration/src/m2_postfusion_variant_screen_v1/runner.py",
        "tfpd_exploration/src/m2_postfusion_variant_screen_v1/driver.py",
        "tfpd_exploration/src/m2_postfusion_variant_screen_v1/lifecycle.py",
        "tfpd_exploration/src/pit_m2_v1/plan.py",
        "tfpd_exploration/src/pit_m2_v1/trainer.py",
        "tfpd_exploration/src/pit_m2_v1/hook.py",
        "tfpd_exploration/src/pit_m2_v1/schedule.py",
        "tfpd_exploration/src/m2_postfusion_probe_v1/memory.py",
        "tfpd_exploration/src/cross_session_worst_group_v1/source_lifecycle.py",
        "streaming_calibration_exp/src/models/streaming_calibration_module.py",
        "streaming_calibration_exp/src/models/falcon_module.py",
        "streaming_calibration_exp/src/models/components/streaming_spint.py",
        "streaming_calibration_exp/src/models/components/streaming_encoders.py",
        "streaming_calibration_exp/src/data/falcon_datamodule.py",
        "streaming_calibration_exp/src/models/components/neuron_dropout.py",
        "streaming_calibration_exp/src/models/components/carrier_noise_augmentation.py",
        "streaming_calibration_exp/src/models/components/correspondence_breaking.py",
        "streaming_calibration_exp/src/models/components/spint.py",
        "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
        "streaming_calibration_exp/third_party/catalyst/distributed_sampler.py",
        "streaming_calibration_exp/third_party/falcon_challenge/filtering.py",
    }
    assert required <= set(plan.STATIC_CLOSURE_RELATIVES)
    assert all((REPO_ROOT / relative).is_file() for relative in plan.STATIC_CLOSURE_RELATIVES)


def test_pf_mean_matches_frozen_postfusion_probe_arithmetic() -> None:
    native = _native()
    calib, side, _ = _inputs()
    adapter = variants.PostFusionIdentityAdapter(native, "PF-MEAN")
    observed = adapter.forward_batch(calib, side_features=side)
    reference = _postfusion_probe_reference(native, calib, side)
    assert torch.equal(observed, reference)
    native_identity = native.forward_batch(calib, side_features=side)
    assert not torch.equal(observed, native_identity)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("arm", ("PF-R1", "PF-R50"))
def test_residual_arms_are_bitwise_native_at_exact_zero_and_gradient_connected(arm: str) -> None:
    native = _native()
    calib, side, _ = _inputs()
    adapter = variants.PostFusionIdentityAdapter(native, arm)
    expected = native.forward_batch(calib, side_features=side)
    observed = adapter.forward_batch(calib, side_features=side)
    assert torch.equal(observed, expected)
    assert adapter.alpha is not None
    assert torch.equal(adapter.alpha.detach(), torch.zeros_like(adapter.alpha))
    # A fixed non-degenerate target makes the residual's first derivative test
    # a real graph check, not a hand-written alpha update.
    target = torch.linspace(-0.7, 0.9, observed.numel(), dtype=torch.float32).reshape_as(observed)
    loss = torch.mean((observed - target) ** 2)
    loss.backward()
    assert adapter.alpha.grad is not None
    assert bool(torch.isfinite(adapter.alpha.grad).all())
    assert float(adapter.alpha.grad.abs().sum()) > 0.0
    assert any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
               for parameter in adapter.native.pre_pool.parameters())
    assert any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
               for parameter in adapter.native.post_pool.parameters())


def test_parameter_topology_and_independent_arm_state() -> None:
    native = _native()
    adapters = {arm: variants.PostFusionIdentityAdapter(copy.deepcopy(native), arm) for arm in plan.ARMS}
    for arm, adapter in adapters.items():
        topology = variants.parameter_topology(adapter)
        assert topology["added_parameter_count"] == plan.ARM_ADDED_PARAMETERS[arm]
    assert adapters["PF-MEAN"].native.pre_pool[0].weight is not adapters["PF-R1"].native.pre_pool[0].weight
    with torch.no_grad():
        assert adapters["PF-R1"].alpha is not None
        adapters["PF-R1"].alpha.add_(0.25)
    assert adapters["PF-R50"].alpha is not None
    assert torch.equal(adapters["PF-R50"].alpha, torch.zeros_like(adapters["PF-R50"].alpha))


def test_adapter_install_before_device_transfer_keeps_alpha_native_and_input_colocated() -> None:
    native = _native()
    adapter = variants.PostFusionIdentityAdapter(native, "PF-R50").to(torch.device("cpu"))
    calib, side, _ = _inputs()
    output = adapter.forward_batch(calib, side_features=side)
    assert adapter.alpha is not None
    assert adapter.alpha.device == next(adapter.native.parameters()).device == calib.device == side.device == output.device


def test_shared_controller_records_identical_chronological_selection_and_no_draw() -> None:
    controller_a = controller.ChronologicalPoolController()
    observed = [controller_a.decide(step, available_members=33) for step in range(6)]
    assert [item.member_count for item in observed] == [30, 10, 4, 30, 10, 4]
    assert all(item.member_indices == tuple(range(item.member_count)) for item in observed)
    assert all(item.generator_state_before_sha256 == item.generator_state_after_sha256 for item in observed)
    with pytest.raises(controller.ControllerError):
        controller_a.decide(0, available_members=29)


def test_shared_batch_evidence_uses_one_batch_and_one_t4_for_all_arms() -> None:
    torch.manual_seed(31)
    batch = (
        torch.randn(2, 50, 7), torch.randn(2, 50, 2), torch.randn(2, 33, 100, 7),
        ["source-a", "source-a"], torch.randn(2, 7, 4),
    )
    decision = controller.ChronologicalPoolController().decide(0, available_members=33)
    evidence = controller.shared_batch_evidence(
        batch=batch, decision=decision, python_rng_sha256="a" * 64,
        numpy_rng_sha256="b" * 64, torch_rng_sha256="c" * 64,
    )
    assert evidence["batch_object_id"] == id(batch)
    assert evidence["pool"]["member_indices"] == list(range(30))
    assert evidence["t4_sha256"] == controller.tensor_digest(batch[4])
    assert evidence["calibration_selected_sha256"] == controller.tensor_digest(batch[2][:, :30])


def test_arm_order_permutation_preserves_per_arm_update_under_replayed_rng() -> None:
    calib, side, _ = _inputs()
    target = torch.ones(3, 7, 50)

    def run(order: tuple[str, ...]) -> dict[str, dict[str, torch.Tensor]]:
        torch.manual_seed(101)
        initial = _native().state_dict()
        results = {}
        for arm in order:
            native = _native()
            native.load_state_dict(initial, strict=True)
            adapter = variants.PostFusionIdentityAdapter(native, arm)
            optimizer = torch.optim.Adam(adapter.parameters(), lr=plan.ADAM_LR)
            # Each independent arm gets the same pre-forward state, as the live
            # single-process runner restores per-arm state around its turn.
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            optimizer.zero_grad(set_to_none=True)
            loss = ((adapter.forward_batch(calib[:, :30], side_features=side) - target) ** 2).mean()
            loss.backward(); optimizer.step()
            results[arm] = {name: value.detach().clone() for name, value in adapter.state_dict().items()}
        return results

    forward = run(plan.ARMS)
    reverse = run(tuple(reversed(plan.ARMS)))
    for arm in plan.ARMS:
        assert forward[arm].keys() == reverse[arm].keys()
        assert all(torch.equal(forward[arm][key], reverse[arm][key]) for key in forward[arm])


def test_runner_steps_one_resident_batch_through_three_independent_real_adapters() -> None:
    class Student(torch.nn.Module):
        def __init__(self, adapter):
            super().__init__()
            self.id_encoder = adapter
            self.decoder = torch.nn.Linear(1, 1)
            for parameter in self.decoder.parameters():
                parameter.requires_grad_(False)

    class Module(torch.nn.Module):
        def __init__(self, adapter):
            super().__init__()
            self.student = Student(adapter)

        def model_step(self, batch):
            _neural, target, calib, sessions, side = batch
            identity = self.student.id_encoder.forward_batch(calib, side_features=side)
            return {"loss": ((identity - target) ** 2).mean(), "session_name": sessions}

    coordinated = screen_runner.PostFusionVariantScreenRunner(repo_root=REPO_ROOT, device="cpu")
    for arm in plan.ARMS:
        adapter = variants.PostFusionIdentityAdapter(_native(), arm)
        module = Module(adapter)
        coordinated.models[arm] = module
        coordinated.adapters[arm] = adapter
        coordinated.optimizers[arm] = torch.optim.Adam(adapter.parameters(), lr=plan.ADAM_LR)
    coordinated._canonical_rng = screen_runner._capture_rng(torch.device("cpu"))
    torch.manual_seed(41)
    batch = (torch.randn(2, 50, 5), torch.randn(2, 5, 50), torch.randn(2, 33, 100, 5),
             ["source-a", "source-a"], torch.randn(2, 5, 4))
    record = coordinated._step_group(batch, detailed=True)
    assert record["paired_rng_exact"] is True
    assert record["shared"]["batch_object_id"] == id(batch)
    assert record["shared"]["pool"]["member_count"] == 30
    assert all(record["arms"][arm]["state_before_sha256"] != record["arms"][arm]["state_after_sha256"]
               for arm in plan.ARMS)
    assert all(coordinated.adapters[arm].alpha is None or float(coordinated.adapters[arm].alpha.abs().sum()) > 0.0
               for arm in plan.ARMS)


def test_live_smoke_boundary_stops_at_12_and_same_iterator_can_resume() -> None:
    class Student(torch.nn.Module):
        def __init__(self, adapter):
            super().__init__(); self.id_encoder = adapter; self.decoder = torch.nn.Linear(1, 1)
            for parameter in self.decoder.parameters(): parameter.requires_grad_(False)

    class Module(torch.nn.Module):
        def __init__(self, adapter): super().__init__(); self.student = Student(adapter)
        def model_step(self, batch):
            _neural, target, calib, sessions, side = batch
            return {"loss": ((self.student.id_encoder.forward_batch(calib, side_features=side) - target) ** 2).mean(),
                    "session_name": sessions}

    def build() -> tuple[screen_runner.PostFusionVariantScreenRunner, list[tuple]]:
        value = screen_runner.PostFusionVariantScreenRunner(repo_root=REPO_ROOT, device="cpu")
        for arm in plan.ARMS:
            adapter = variants.PostFusionIdentityAdapter(_native(), arm)
            module = Module(adapter)
            value.models[arm] = module; value.adapters[arm] = adapter
            value.optimizers[arm] = torch.optim.Adam(adapter.parameters(), lr=plan.ADAM_LR)
        value._canonical_rng = screen_runner._capture_rng(torch.device("cpu"))
        value.source_authority = {"source_heldin_in_sample_monitor": {"monitor_groups": ["mon-a"]}}
        value.source_heldin_in_sample_monitor_metric = lambda arm: {"per_session_r2": {"mon-a": 0.1}, "equal_session_mean": 0.1}  # type: ignore[method-assign]
        batch = (torch.randn(1, 50, 3), torch.randn(1, 3, 50), torch.randn(1, 33, 100, 3),
                 ["source-a"], torch.randn(1, 3, 4))
        return value, [batch for _ in range(15)]

    failing, loader = build(); failing.datamodule = object(); failing.train_loader = loader
    with pytest.raises(screen_runner.RunnerError):
        failing.run_fixed_screen(smoke_gate=lambda records: (_ for _ in ()).throw(screen_runner.RunnerError("gate")))
    assert failing.global_step == 12

    resumed, loader = build(); resumed.train_loader = loader
    iterator = iter(loader)
    records = [resumed._step_group(next(iterator), detailed=True) for _ in range(12)]
    resumed._validate_smoke_records(records)
    epoch = resumed._finish_epoch(1, iterator, completed_steps=12, detailed_records=records)
    assert epoch["step_count"] == 15
    assert resumed.global_step == 15

    expired, loader = build(); expired.train_loader = loader
    with pytest.raises(screen_runner.RunnerError, match="180-minute hard cap"):
        expired._finish_epoch(1, iter(loader), deadline=__import__("time").monotonic() - 1.0)
    assert expired.global_step == 0


def test_no_cuda_is_initialized_by_constructibility_suite() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1")
    assert not torch.cuda.is_initialized()


def test_source_monitor_selection_is_explicitly_in_sample_equal_mean_and_fixed12_tiebroken() -> None:
    train_groups = tuple(f"source-{index}" for index in range(7))
    train_windows = tuple((train_groups[index % 7], index) for index in range(plan.SOURCE_TRAIN_WINDOW_COUNT))
    monitor_windows = train_windows[:plan.SOURCE_MONITOR_WINDOW_COUNT]
    monitor = selection.validate_source_heldin_in_sample_monitor(
        train_groups, train_groups, train_windows, monitor_windows,
    )
    assert monitor["monitor_name"] == "source_heldin_in_sample_monitor"
    assert monitor["monitor_windows_contained_in_train"] == plan.SOURCE_MONITOR_WINDOW_COUNT
    with pytest.raises(selection.SelectionError):
        selection.validate_source_heldin_in_sample_monitor(
            train_groups, train_groups, train_windows, train_windows[:1010],
        )
    curves = {
        arm: [{"epoch": epoch, "source_heldin_in_sample_monitor_mean": 0.4 + 0.01 * epoch}
              for epoch in range(1, 13)]
        for arm in plan.ARMS
    }
    curves["PF-R1"][-1]["source_heldin_in_sample_monitor_mean"] = curves["PF-MEAN"][-1]["source_heldin_in_sample_monitor_mean"] + 0.001
    curves["PF-R50"][-1]["source_heldin_in_sample_monitor_mean"] = curves["PF-MEAN"][-1]["source_heldin_in_sample_monitor_mean"] + 0.0015
    winner = selection.choose_screen_winner(curves)
    assert winner["winner"] == "PF-MEAN"  # all are inside the 0.002 simple tie band
    checkpoint = selection.choose_screen_checkpoint(curves["PF-MEAN"])
    assert checkpoint["epoch"] == 12
    with pytest.raises(selection.SelectionError):
        selection.choose_screen_winner({**curves, "PF-MEAN": curves["PF-MEAN"][:-1]})


def test_attempt_first_terminal_xor_failure_lifecycle(tmp_path: Path) -> None:
    relative = "tfpd_exploration/results/m2_postfusion_variant_screen_v1/screen"
    terminal, failure = lifecycle.execute_stage(
        repo_root=tmp_path, root_relative=relative, attempt={"status": "attempt"},
        launch=lambda: {"status": "launch"},
        bodies=lambda artifact: {"body.json": artifact.publish_json("body.json", {"ok": True})},
        terminal=lambda shas, attempt_sha, launch_sha: {"body_sha256": shas["body.json"],
                                                        "attempt_seen": attempt_sha, "launch_seen": launch_sha},
        progress=lambda: {"stage": "synthetic"},
    )
    assert terminal is not None and failure is None
    root = tmp_path / relative
    assert (root / "terminal.json").exists() and not (root / "failure.json").exists()
    for name in ("attempt.json", "launch.json", "body.json", "terminal.json"):
        assert (root / name).stat().st_mode & 0o777 == 0o444
        assert (root / f"{name}.sha256").read_text(encoding="ascii").endswith(f"  {name}\n")


def test_failure_lifecycle_is_exclusive_before_terminal(tmp_path: Path) -> None:
    relative = "tfpd_exploration/results/m2_postfusion_variant_screen_v1/screen"
    terminal, failure = lifecycle.execute_stage(
        repo_root=tmp_path, root_relative=relative, attempt={"status": "attempt"},
        launch=lambda: {"status": "launch"},
        bodies=lambda artifact: (_ for _ in ()).throw(RuntimeError("synthetic body failure")),
        terminal=lambda *_: {}, progress=lambda: {"before": "body"},
    )
    assert terminal is None and failure is not None
    root = tmp_path / relative
    assert (root / "failure.json").exists() and not (root / "terminal.json").exists()


def test_public_cli_is_inert_and_imports_no_torch_under_python_s(tmp_path: Path) -> None:
    code = (
        "import importlib.util,sys; "
        "p=sys.argv[1]; "
        "s=importlib.util.spec_from_file_location('pf_cli',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
        "r=m.main(['--dry-run']); assert r==0; assert 'torch' not in sys.modules"
    )
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO_ROOT)})
    completed = subprocess.run(
        [sys.executable, "-S", "-c", code,
         str(REPO_ROOT / "tfpd_exploration/scripts/run_m2_postfusion_variant_screen_v1.py")],
        env=environment, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "INERT_PLAN_ONLY__NO_CAPABILITY" in completed.stdout


def test_driver_publishes_all_preregistered_source_best_binary_checkpoints(tmp_path: Path) -> None:
    # Build only the explicit closure tree under a temporary repo.  This never
    # copies a dataset, checkpoint, result receipt, or CUDA/runtime asset.
    for relative in plan.STATIC_CLOSURE_RELATIVES:
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    class FakeRunner:
        def __init__(self, *, repo_root, device):
            assert Path(repo_root) == tmp_path and device == "cuda:0"

        def prepare(self):
            return {"source_only": True, "loader_constructions": 1}

        def run_fixed_screen(self):
            return {
                "source_authority": {"source_only": True}, "screen_epochs": [],
                "winner": {"winner": "PF-MEAN"},
                "selected_checkpoint": {"epoch": 3, "source_heldin_in_sample_monitor_mean": 0.2},
                "source_best_checkpoints": {
                    arm: {"epoch": 3, "student_state_sha256": (letter * 64)}
                    for arm, letter in zip(plan.ARMS, ("a", "b", "c"))
                },
                "endpoint_slope_limitations": {
                    "PF-MEAN": {"epoch11_to_epoch12_monitor_slope": 0.0, "possible_horizon_limitation": False},
                    "PF-R1": {"epoch11_to_epoch12_monitor_slope": 0.01, "possible_horizon_limitation": True},
                    "PF-R50": {"epoch11_to_epoch12_monitor_slope": -0.02, "possible_horizon_limitation": False},
                },
                "runtime": {"total_wall_seconds": 1.25, "hard_cap_seconds": 10800,
                            "within_hard_cap": True, "shared_optimizer_step_groups": 36,
                            "shared_step_groups_per_second": 28.8},
                "matched_prefusion_control_trained": False,
                "_source_best_checkpoint_bodies": {
                    "PF-MEAN": b"synthetic-mean-state", "PF-R1": b"synthetic-r1-state",
                    "PF-R50": b"synthetic-r50-state",
                },
            }

    capability = driver._mint_test_capability(tmp_path)
    terminal, failure = driver.execute_screen(
        capability=capability, runner_factory=FakeRunner,
        _test_launch_validator=lambda: {"cuda_visible_devices": "0", "visible_device_count": 1, "torch_device": "cuda:0"},
    )
    assert terminal is not None and failure is None
    root = tmp_path / plan.SCREEN_ROOT_RELATIVE
    assert (root / "source_best_pf_mean.pt").read_bytes() == b"synthetic-mean-state"
    assert (root / "source_best_pf_r1.pt").read_bytes() == b"synthetic-r1-state"
    assert (root / "source_best_pf_r50.pt").read_bytes() == b"synthetic-r50-state"
    screen = __import__("json").loads((root / "screen.json").read_text(encoding="utf-8"))
    assert screen["source_best_checkpoints"]["PF-MEAN"]["sha256"] == __import__("hashlib").sha256(b"synthetic-mean-state").hexdigest()
    assert screen["endpoint_slope_limitations"]["PF-R1"]["possible_horizon_limitation"] is True
    assert set(screen["endpoint_slope_limitations"]) == set(plan.ARMS)
    terminal_body = __import__("json").loads((root / "terminal.json").read_text(encoding="utf-8"))
    assert terminal_body["source_best_checkpoint_sha256"]["PF-MEAN"] == screen["source_best_checkpoints"]["PF-MEAN"]["sha256"]
    assert terminal_body["runtime"] == screen["runtime"]
    assert not (root / "failure.json").exists()
    with pytest.raises(driver.DriverError):
        driver.execute_screen(capability=capability, runner_factory=FakeRunner,
                              _test_launch_validator=lambda: {})
