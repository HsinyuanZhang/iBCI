"""Focused score-free tests for the Phase-C v5 post-test device boundary."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from sua_exploration.mc_maze.m2_native_post33_deployment_device_prep_v5 import (
    DeploymentDevicePreparationError,
    assert_cached_deployment_ready_v5,
    prepare_cached_online_b1_after_trainer_test_v5,
    tensor_sha256_v5,
)


class _ToyDeploymentNet(nn.Module):
    """Small cache-consuming decoder with both parameters and a buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.readout = nn.Linear(96, 2, bias=False)
        self.register_buffer("gain", torch.ones(1, 1, 96))

    def forward(self, neural: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return self.readout((neural + identity.permute(0, 2, 1)) * self.gain)


class _ToyOwner(nn.Module):
    def __init__(self, arm: str) -> None:
        super().__init__()
        if arm == "spint":
            self.net = _ToyDeploymentNet()
            self._cached_identity = torch.arange(96 * 50, dtype=torch.float32).reshape(1, 96, 50)
        elif arm == "t4":
            self.student = _ToyDeploymentNet()
            self._cached_outer_identity = torch.arange(96 * 50, dtype=torch.float32).reshape(1, 96, 50)
        else:  # pragma: no cover - fixture guard
            raise ValueError(arm)
        # Sentinels model information whose order/values must remain untouched
        # by a deployment-device operation.
        self.trial_order_sentinel = (0, 1, 2, 33)
        self.target_label_sentinel = ("not-read", "not-mutated")

    def cached_online_forward(self, neural: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "net"):
            return self.net(neural, self._cached_identity)
        return self.student(neural, self._cached_outer_identity)


@pytest.mark.parametrize("arm,cache_attribute,module_attribute", [
    ("spint", "_cached_identity", "net"),
    ("t4", "_cached_outer_identity", "student"),
])
def test_cpu_preparation_is_symmetric_and_preserves_cache_bytes(
    arm: str, cache_attribute: str, module_attribute: str,
) -> None:
    owner = _ToyOwner(arm).train()
    neural_cpu = torch.zeros(1, 50, 96)
    cache = getattr(owner, cache_attribute)
    expected_hash = tensor_sha256_v5(cache)
    expected_bytes = cache.numel() * cache.element_size()
    trial_order_before = owner.trial_order_sentinel
    labels_before = owner.target_label_sentinel

    prepared = prepare_cached_online_b1_after_trainer_test_v5(
        owner,
        arm=arm,  # type: ignore[arg-type]
        neural_cpu=neural_cpu,
        target_device="cpu",
        expected_cache_sha256=expected_hash,
        expected_cache_bytes=expected_bytes,
        allow_non_cuda_for_test=True,
    )

    assert prepared.evidence.arm == arm
    assert prepared.evidence.deployment_module_attribute == module_attribute
    assert prepared.evidence.cached_identity_attribute == cache_attribute
    assert prepared.evidence.cache_sha256 == expected_hash
    assert prepared.evidence.cache_bytes == expected_bytes
    assert prepared.evidence.pre_cache_device == "cpu"
    assert prepared.evidence.pre_input_device == "cpu"
    assert prepared.evidence.score_or_metric_accessed is False
    assert prepared.evidence.chronology_or_label_mutation_performed is False
    assert owner.training is False
    assert getattr(owner, module_attribute).training is False
    assert getattr(owner, cache_attribute).device.type == "cpu"
    assert tensor_sha256_v5(getattr(owner, cache_attribute)) == expected_hash
    assert owner.trial_order_sentinel == trial_order_before
    assert owner.target_label_sentinel == labels_before
    output = prepared.checked_forward(owner.cached_online_forward)
    assert output.shape == (1, 50, 2)


@pytest.mark.parametrize("arm,cache_attribute", [
    ("spint", "_cached_identity"),
    ("t4", "_cached_outer_identity"),
])
def test_checked_forward_rejects_mode_drift_after_preparation(arm: str, cache_attribute: str) -> None:
    owner = _ToyOwner(arm)
    cache = getattr(owner, cache_attribute)
    prepared = prepare_cached_online_b1_after_trainer_test_v5(
        owner,
        arm=arm,  # type: ignore[arg-type]
        neural_cpu=torch.zeros(1, 50, 96),
        target_device="cpu",
        expected_cache_sha256=tensor_sha256_v5(cache),
        expected_cache_bytes=cache.numel() * cache.element_size(),
        allow_non_cuda_for_test=True,
    )
    owner.train()
    with pytest.raises(DeploymentDevicePreparationError, match="eval mode"):
        prepared.checked_forward(owner.cached_online_forward)


def test_ready_assertion_rejects_mixed_input_device_before_forward() -> None:
    owner = _ToyOwner("spint")
    cache = owner._cached_identity
    expected_hash = tensor_sha256_v5(cache)
    expected_bytes = cache.numel() * cache.element_size()
    owner.eval()
    with pytest.raises(DeploymentDevicePreparationError, match="input device"):
        assert_cached_deployment_ready_v5(
            owner,
            arm="spint",
            neural=torch.empty((1, 50, 96), device="meta"),
            target_device="cpu",
            expected_cache_sha256=expected_hash,
            expected_cache_bytes=expected_bytes,
        )


def test_production_path_rejects_non_cuda_target() -> None:
    owner = _ToyOwner("spint")
    with pytest.raises(DeploymentDevicePreparationError, match="must be CUDA"):
        prepare_cached_online_b1_after_trainer_test_v5(
            owner,
            arm="spint",
            neural_cpu=torch.zeros(1, 50, 96),
            target_device="cpu",
        )
