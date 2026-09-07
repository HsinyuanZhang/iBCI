#!/usr/bin/env python3
"""Synthetic real-CUDA B=1 smoke for the Phase-C v5 device preparation.

This script deliberately uses no project datamodule, checkpoint, calibration
labels, behavior target, R² object, result root, or endpoint payload.  It
exercises the exact lifecycle that failed in r6d:

1. a minimal Lightning module creates a *plain* cached identity on CUDA during
   ``Trainer.test``;
2. Lightning 2.4 tears down by moving registered module state to CPU and
   restores the original training flag;
3. v5 explicitly repairs the deployment net/cache/input state;
4. the unchanged v4 profiler executes its real 5-warmup + 20-timed B=1 loop
   through a callable that checks the actual tensor supplied by the profiler.

It is a device-lifecycle smoke only, never a formal evaluation command.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import torch
from lightning.pytorch import LightningModule, Trainer
from torch import nn
from torch.utils.data import DataLoader

from sua_exploration.mc_maze.m2_native_post33_deployment_profiler_v4 import DeploymentProfilerV4
from sua_exploration.mc_maze.m2_native_post33_deployment_device_prep_v5 import (
    prepare_cached_online_b1_after_trainer_test_v5,
    tensor_sha256_v5,
)


class _ToyDeploymentNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.readout = nn.Linear(96, 2, bias=False)
        self.register_buffer("gain", torch.ones(1, 1, 96))

    def forward(self, neural: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return self.readout((neural + identity.permute(0, 2, 1)) * self.gain)


class _SyntheticPostTestModule(LightningModule):
    """Only the attributes consumed by the production cached B=1 path."""

    def __init__(self, arm: str) -> None:
        super().__init__()
        self.arm = arm
        if arm == "spint":
            self.net = _ToyDeploymentNet()
            self._cached_identity = None
        elif arm == "t4":
            self.student = _ToyDeploymentNet()
            self._cached_outer_identity = None
        else:  # pragma: no cover - fixed caller set
            raise ValueError(arm)
        self.test_forward_calls = 0

    def on_test_start(self) -> None:
        # This occurs after the strategy has placed registered module state on
        # CUDA.  The cache is intentionally not a registered buffer/parameter.
        cache = torch.arange(96 * 50, device=self.device, dtype=torch.float32).reshape(1, 96, 50)
        if self.arm == "spint":
            self._cached_identity = cache
        else:
            self._cached_outer_identity = cache

    def cached_online_forward(self, neural: torch.Tensor) -> torch.Tensor:
        if self.arm == "spint":
            assert self._cached_identity is not None
            return self.net(neural, self._cached_identity)
        assert self._cached_outer_identity is not None
        return self.student(neural, self._cached_outer_identity)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        output = self.cached_online_forward(batch)
        if output.shape != (1, 50, 2):
            raise RuntimeError(f"synthetic test output shape drift: {tuple(output.shape)}")
        self.test_forward_calls += 1


def _run_arm(arm: str) -> dict[str, object]:
    model = _SyntheticPostTestModule(arm).train()
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
        use_distributed_sampler=False,
    )
    # A single synthetic B=1 neural window, not a project dataset or endpoint.
    dataloader = DataLoader(torch.zeros(1, 50, 96), batch_size=1)
    trainer.test(model=model, dataloaders=dataloader, verbose=False)

    deployment_module = model.net if arm == "spint" else model.student
    cache_attribute = "_cached_identity" if arm == "spint" else "_cached_outer_identity"
    cache = getattr(model, cache_attribute)
    if not isinstance(cache, torch.Tensor):
        raise RuntimeError("synthetic plain cache was not created during test")
    parameter_devices = {parameter.device.type for parameter in deployment_module.parameters()}
    if parameter_devices != {"cpu"}:
        raise RuntimeError(f"Trainer.test did not reproduce CPU module teardown: {parameter_devices}")
    if cache.device.type != "cuda":
        raise RuntimeError(f"Trainer.test did not preserve plain CUDA cache: {cache.device}")
    if model.training is not True:
        raise RuntimeError("Trainer.test did not restore the pre-test training flag")

    expected_cache_sha256 = tensor_sha256_v5(cache)
    expected_cache_bytes = cache.numel() * cache.element_size()
    neural_cpu = torch.full((1, 50, 96), 0.25, dtype=torch.float32)
    target = torch.device("cuda", torch.cuda.current_device())
    prepared = prepare_cached_online_b1_after_trainer_test_v5(
        model, arm=arm, neural_cpu=neural_cpu,
        target_device=target,
        expected_cache_sha256=expected_cache_sha256,
        expected_cache_bytes=expected_cache_bytes,
    )
    # The production profiler owns the same CPU->CUDA transfer that r6d used.
    # The bound function checks the *actual* tensor passed on every repeat.
    profiler = DeploymentProfilerV4()
    profiler.benchmark_online_b1(
        prepared.bind_checked_forward(model.cached_online_forward), neural_cpu
    )
    prepared.assert_ready()  # full cache-byte hash after all 25 forwards
    benchmark = profiler.online_b1_microbenchmark
    if not isinstance(benchmark, dict) or len(benchmark["timing_ns"]["samples"]) != 20:
        raise RuntimeError("real CUDA B=1 profiler did not complete 5+20 calls")
    return {
        "arm": arm,
        "trainer_test_reproduced": {
            "registered_parameter_device_after_test": "cpu",
            "plain_cache_device_after_test": "cuda",
            "training_flag_after_test": True,
        },
        "prepared": {
            "target_device": prepared.evidence.target_device,
            "cache_bytes": prepared.evidence.cache_bytes,
            "parameter_count": prepared.evidence.parameter_count,
            "buffer_count": prepared.evidence.buffer_count,
            "score_or_metric_accessed": prepared.evidence.score_or_metric_accessed,
            "chronology_or_label_mutation_performed": (
                prepared.evidence.chronology_or_label_mutation_performed
            ),
        },
        "profiler": {
            "warmup_repeats": benchmark["warmup_repeats"],
            "timed_repeats": benchmark["timed_repeats"],
            "input_residency": benchmark["input_residency"],
            "behavior_target_read": benchmark["behavior_target_read"],
        },
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("SKIP: real CUDA smoke requires an available CUDA device")
    torch.manual_seed(0)
    results = [_run_arm("spint"), _run_arm("t4")]
    print(json.dumps({
        "schema": "m2_post33_phase_c_v5_synthetic_cuda_b1_smoke_v1",
        "formal_endpoint_or_r2_accessed": False,
        "results": results,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
