"""Fail-closed post-``Trainer.test`` deployment preparation for Phase-C v5.

This is deliberately a small, score-free boundary.  Lightning 2.4 tears a
test strategy down by calling ``lightning_module.cpu()``.  Registered module
parameters and buffers therefore move, while the Phase-C cached identity is a
plain tensor attribute and does not.  A cached online B=1 forward immediately
after ``Trainer.test`` must prepare both sides explicitly.

The module contains no data loading, labels, metrics, checkpoints, chronology,
or endpoint code.  It is intended to be called by the *new* r7 evaluator only
after the historical test loop has finished and before its B=1 deployment
microbenchmark begins.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Literal

import torch
from torch import nn


ARM = Literal["spint", "t4"]
PREPARATION_SCHEMA = "m2_post33_phase_c_deployment_device_preparation_v5"


class DeploymentDevicePreparationError(RuntimeError):
    """Raised before a mixed-device cached deployment forward can execute."""


@dataclass(frozen=True)
class _ArmSpec:
    deployment_module_attribute: str
    cached_identity_attribute: str


_ARM_SPECS: dict[ARM, _ArmSpec] = {
    "spint": _ArmSpec("net", "_cached_identity"),
    "t4": _ArmSpec("student", "_cached_outer_identity"),
}


@dataclass(frozen=True)
class DeploymentDeviceEvidenceV5:
    """Score-free evidence returned by one explicit preparation operation."""

    schema: str
    arm: ARM
    target_device: str
    deployment_module_attribute: str
    cached_identity_attribute: str
    cache_sha256: str
    cache_bytes: int
    cache_shape: tuple[int, ...]
    cache_dtype: str
    input_shape: tuple[int, ...]
    input_dtype: str
    pre_parameter_devices: tuple[str, ...]
    pre_buffer_devices: tuple[str, ...]
    pre_cache_device: str
    pre_input_device: str
    parameter_count: int
    buffer_count: int
    cache_requires_grad: bool
    chronology_or_label_mutation_performed: bool
    score_or_metric_accessed: bool


@dataclass
class PreparedCachedDeploymentV5:
    """Prepared B=1 neural input plus a checked-forward boundary."""

    owner: nn.Module
    arm: ARM
    deployment_module: nn.Module
    cache_attribute: str
    neural: torch.Tensor
    expected_cache_sha256: str
    expected_cache_bytes: int
    target_device: torch.device
    evidence: DeploymentDeviceEvidenceV5

    def assert_ready_for(self, neural: torch.Tensor, *, verify_cache_hash: bool = True) -> None:
        """Assert one concrete execution input still shares the target device.

        The cached-byte hash is checked at preparation and after the benchmark.
        A bound benchmark callable only checks device/shape/mode/byte-count on
        every repeat: hashing through a CUDA-to-CPU copy inside every timed
        iteration would corrupt the latency measurement it is meant to guard.
        """
        assert_cached_deployment_ready_v5(
            self.owner,
            arm=self.arm,
            neural=neural,
            target_device=self.target_device,
            expected_cache_sha256=self.expected_cache_sha256,
            expected_cache_bytes=self.expected_cache_bytes,
            verify_cache_hash=verify_cache_hash,
        )

    def assert_ready(self) -> None:
        """Assert the prepared input and canonical cache are intact."""
        self.assert_ready_for(self.neural, verify_cache_hash=True)

    def checked_forward(self, forward: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Run one forward with pre- and post-device/cache assertions."""
        self.assert_ready()
        output = forward(self.neural)
        self.assert_ready()
        return output

    def bind_checked_forward(
        self, forward: Callable[[torch.Tensor], torch.Tensor]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Bind checks to the profiler's actual CUDA input on every invocation.

        ``DeploymentProfilerV4.benchmark_online_b1`` owns the CPU-to-CUDA
        transfer.  This wrapper therefore accepts the tensor it actually
        passes, instead of silently assuming ``self.neural`` was used.
        """
        def checked(neural: torch.Tensor) -> torch.Tensor:
            self.assert_ready_for(neural, verify_cache_hash=False)
            output = forward(neural)
            self.assert_ready_for(neural, verify_cache_hash=False)
            return output

        return checked


def tensor_sha256_v5(tensor: torch.Tensor) -> str:
    """Hash canonical tensor bytes without changing dtype, shape, or values."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("cached identity must be a torch.Tensor")
    if tensor.device.type == "meta":
        raise DeploymentDevicePreparationError("meta cached tensors cannot prove byte preservation")
    canonical = tensor.detach().cpu().contiguous()
    return hashlib.sha256(canonical.numpy().tobytes(order="C")).hexdigest()


def _canonical_device(device: torch.device | str) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and requested.index is None:
        if not torch.cuda.is_available():
            raise DeploymentDevicePreparationError("CUDA target requested but CUDA is unavailable")
        return torch.device("cuda", torch.cuda.current_device())
    return requested


def _arm_spec(arm: ARM) -> _ArmSpec:
    try:
        return _ARM_SPECS[arm]
    except KeyError as exc:
        raise ValueError(f"unsupported Phase-C deployment arm: {arm!r}") from exc


def _require_owner_and_state(owner: nn.Module, arm: ARM) -> tuple[_ArmSpec, nn.Module, torch.Tensor]:
    if not isinstance(owner, nn.Module):
        raise TypeError("deployment owner must be an nn.Module")
    spec = _arm_spec(arm)
    deployment_module = getattr(owner, spec.deployment_module_attribute, None)
    if not isinstance(deployment_module, nn.Module):
        raise DeploymentDevicePreparationError(
            f"{arm} deployment module {spec.deployment_module_attribute!r} is unavailable"
        )
    # The cache is deliberately a plain attribute in v4.  A registration change
    # would alter the lifecycle being repaired, so fail closed rather than
    # silently treating a different model state as equivalent.
    if spec.cached_identity_attribute in owner._buffers:
        raise DeploymentDevicePreparationError("cached identity unexpectedly became a registered buffer")
    if spec.cached_identity_attribute in owner._parameters:
        raise DeploymentDevicePreparationError("cached identity unexpectedly became a registered parameter")
    cache = getattr(owner, spec.cached_identity_attribute, None)
    if not isinstance(cache, torch.Tensor):
        raise DeploymentDevicePreparationError(
            f"{arm} cached identity {spec.cached_identity_attribute!r} is unavailable"
        )
    if cache.requires_grad:
        raise DeploymentDevicePreparationError("cached deployment identity must be detached")
    return spec, deployment_module, cache


def _module_device_lists(module: nn.Module) -> tuple[tuple[str, ...], tuple[str, ...]]:
    parameters = tuple(sorted({str(parameter.device) for parameter in module.parameters(recurse=True)}))
    buffers = tuple(sorted({str(buffer.device) for buffer in module.buffers(recurse=True)}))
    return parameters, buffers


def _assert_module_on_device(module: nn.Module, target_device: torch.device) -> tuple[int, int]:
    bad_parameters = [
        name for name, parameter in module.named_parameters(recurse=True)
        if parameter.device != target_device
    ]
    bad_buffers = [
        name for name, buffer in module.named_buffers(recurse=True)
        if buffer.device != target_device
    ]
    if bad_parameters or bad_buffers:
        raise DeploymentDevicePreparationError(
            "deployment module has mixed device state: "
            f"parameter_fields={bad_parameters}, buffer_fields={bad_buffers}, target={target_device}"
        )
    return (
        sum(1 for _ in module.parameters(recurse=True)),
        sum(1 for _ in module.buffers(recurse=True)),
    )


def assert_cached_deployment_ready_v5(
    owner: nn.Module,
    *,
    arm: ARM,
    neural: torch.Tensor,
    target_device: torch.device | str,
    expected_cache_sha256: str,
    expected_cache_bytes: int,
    verify_cache_hash: bool = True,
) -> None:
    """Fail before a forward if module/cache/input/mode no longer match."""
    target = _canonical_device(target_device)
    spec, deployment_module, cache = _require_owner_and_state(owner, arm)
    if not isinstance(neural, torch.Tensor):
        raise TypeError("cached deployment neural input must be a torch.Tensor")
    if tuple(neural.shape) != (1, 50, 96):
        raise DeploymentDevicePreparationError(
            f"cached deployment B=1 input must be [1,50,96], got {tuple(neural.shape)}"
        )
    _assert_module_on_device(deployment_module, target)
    if cache.device != target:
        raise DeploymentDevicePreparationError(
            f"cached identity device {cache.device} != deployment device {target}"
        )
    if neural.device != target:
        raise DeploymentDevicePreparationError(
            f"B=1 input device {neural.device} != deployment device {target}"
        )
    if owner.training or deployment_module.training:
        raise DeploymentDevicePreparationError("cached deployment must run in eval mode after Trainer.test")
    cache_bytes = int(cache.numel() * cache.element_size())
    if cache_bytes != expected_cache_bytes:
        raise DeploymentDevicePreparationError("cached identity byte count changed during device preparation")
    if verify_cache_hash and tensor_sha256_v5(cache) != expected_cache_sha256:
        raise DeploymentDevicePreparationError("cached identity bytes changed during device preparation")


def prepare_cached_online_b1_after_trainer_test_v5(
    owner: nn.Module,
    *,
    arm: ARM,
    neural_cpu: torch.Tensor,
    target_device: torch.device | str = "cuda",
    expected_cache_sha256: str | None = None,
    expected_cache_bytes: int | None = None,
    allow_non_cuda_for_test: bool = False,
) -> PreparedCachedDeploymentV5:
    """Atomically prepare the cached online B=1 path after ``Trainer.test``.

    The canonical cache hash/byte count is captured *before* moving anything;
    the exact same values are checked after moving the plain cache tensor.  No
    support/descriptor labels, trial order, target, metric, or score is read.
    ``allow_non_cuda_for_test`` exists only so unit tests can exercise the
    invariant on CPU-only hosts; production r7 must keep it false.
    """
    target = _canonical_device(target_device)
    if target.type != "cuda" and not allow_non_cuda_for_test:
        raise DeploymentDevicePreparationError("production cached deployment target must be CUDA")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise DeploymentDevicePreparationError("CUDA target requested but CUDA is unavailable")
    if not isinstance(neural_cpu, torch.Tensor):
        raise TypeError("B=1 neural input must be a torch.Tensor")
    if neural_cpu.device.type != "cpu":
        raise DeploymentDevicePreparationError("post-test B=1 source input must be CPU-resident")
    if tuple(neural_cpu.shape) != (1, 50, 96):
        raise DeploymentDevicePreparationError(
            f"cached deployment B=1 source input must be [1,50,96], got {tuple(neural_cpu.shape)}"
        )
    spec, deployment_module, cache = _require_owner_and_state(owner, arm)
    pre_parameter_devices, pre_buffer_devices = _module_device_lists(deployment_module)
    observed_hash = tensor_sha256_v5(cache)
    observed_bytes = int(cache.numel() * cache.element_size())
    if expected_cache_sha256 is not None and observed_hash != expected_cache_sha256:
        raise DeploymentDevicePreparationError("cached identity hash differs from its sealed canonical value")
    if expected_cache_bytes is not None and observed_bytes != expected_cache_bytes:
        raise DeploymentDevicePreparationError("cached identity byte count differs from its sealed value")

    # The three participants must be moved explicitly.  Calling only
    # ``owner.eval()`` addresses neither the CPU teardown nor the plain tensor.
    deployment_module.to(target)
    owner.eval()
    deployment_module.eval()
    setattr(owner, spec.cached_identity_attribute, cache.to(target))
    neural = neural_cpu.to(target)

    prepared = PreparedCachedDeploymentV5(
        owner=owner,
        arm=arm,
        deployment_module=deployment_module,
        cache_attribute=spec.cached_identity_attribute,
        neural=neural,
        expected_cache_sha256=observed_hash,
        expected_cache_bytes=observed_bytes,
        target_device=target,
        evidence=DeploymentDeviceEvidenceV5(
            schema=PREPARATION_SCHEMA,
            arm=arm,
            target_device=str(target),
            deployment_module_attribute=spec.deployment_module_attribute,
            cached_identity_attribute=spec.cached_identity_attribute,
            cache_sha256=observed_hash,
            cache_bytes=observed_bytes,
            cache_shape=tuple(cache.shape),
            cache_dtype=str(cache.dtype),
            input_shape=tuple(neural_cpu.shape),
            input_dtype=str(neural_cpu.dtype),
            pre_parameter_devices=pre_parameter_devices,
            pre_buffer_devices=pre_buffer_devices,
            pre_cache_device=str(cache.device),
            pre_input_device=str(neural_cpu.device),
            parameter_count=sum(1 for _ in deployment_module.parameters(recurse=True)),
            buffer_count=sum(1 for _ in deployment_module.buffers(recurse=True)),
            cache_requires_grad=bool(cache.requires_grad),
            chronology_or_label_mutation_performed=False,
            score_or_metric_accessed=False,
        ),
    )
    prepared.assert_ready()
    return prepared
