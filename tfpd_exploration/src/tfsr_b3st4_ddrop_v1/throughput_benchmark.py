"""Fail-closed, engineering-only TF-SR throughput equivalence benchmark.

This module deliberately has no top-level Torch, dataset, checkpoint, or CUDA
import.  The public CLI loads it as a synthetic package so a zero-argument
invocation can only describe the sealed plan.  The physical GPU route is kept
behind a root-review capability that this work order does not issue.

Nothing here is a training result: inputs are deterministic synthetic tensors,
the only admissible historical read is the sealed Phase-D v2 throughput JSON,
and the sole possible output is an explicitly engineering-only receipt.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import resource
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


CELL = "TFSR_B3ST4_DDROP_THROUGHPUT_ENGINEERING_V1"
PHASE = "TFSR_THROUGHPUT_EQUIVALENCE_20260819"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_TFSR_THROUGHPUT_EQUIVALENCE_20260819.md"
WORKORDER_SHA256 = "335ba42c717d150ca3fd28ff8445de9857e7eac93fc4fedd0b46e83606061b31"
MODEL_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/model.py"
MODEL_SHA256 = "3d4a3a8d4e2a68e9933e84274f2a62308a6eeb5a6978671b53e72af442148fc4"
NATIVE_ACTIVITY_RELATIVE = "tfpd_exploration/src/tfpd/bilinear_readin.py"
NATIVE_ACTIVITY_SHA256 = "2576e91baa2ecbae7d9b5734d2aad6618563fe5cd8dbb5cf0f15358caf244ac0"
SOURCE_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/throughput_benchmark.py"
CLI_RELATIVE = "tfpd_exploration/scripts/benchmark_tfsr_b3st4_ddrop_throughput.py"
TEST_RELATIVE = "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_v1_throughput_benchmark.py"
LIVE_THROUGHPUT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_seed42_train_v2/throughput100.json"
LIVE_THROUGHPUT_SHA256 = "60b12c8f79d58ee50af4dd60255de3bf77a69ced4c7ebafe758710225b2da419"
LIVE_THROUGHPUT_SIDECAR_RELATIVE = LIVE_THROUGHPUT_RELATIVE + ".sha256"
OUTPUT_ROOT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_throughput_engineering_v1"
OUTPUT_TOPOLOGY = ("receipt.json",)

# This is deliberately an explicit, ordered, non-globbed closure.  The live
# receipt sidecar is included because the body hash alone is not a durable
# immutable-pair proof.
BENCHMARK_CLOSURE = (
    WORKORDER_RELATIVE,
    SOURCE_RELATIVE,
    CLI_RELATIVE,
    TEST_RELATIVE,
    MODEL_RELATIVE,
    NATIVE_ACTIVITY_RELATIVE,
    LIVE_THROUGHPUT_RELATIVE,
    LIVE_THROUGHPUT_SIDECAR_RELATIVE,
)

FROZEN_BASELINE = {
    "steps_per_second": 5.910276663093795,
    "elapsed_seconds": 16.91968171717599,
    "projected_48epoch_seconds": 275503.17740077665,
    "threshold_steps": 100,
    "epochs": 48,
    "steps_per_epoch": 33_925,
}
FROZEN_GPU0 = {
    "cuda_visible_devices": "0",
    "internal_device": "cuda:0",
    "uuid": "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9",
    "bdf": "00000000:01:00.0",
    "name": "NVIDIA GeForce RTX 3090",
    # These are distinct authorities.  NVML's nominal board capacity and
    # Torch's allocatable-device property intentionally do not agree after a
    # shared-MiB conversion on this host.
    "nvidia_smi_memory_total_mib": 24576,
    "torch_total_memory_bytes": 25_435_111_424,
}
FROZEN_ADAM = {
    "cls": "Adam",
    "lr": 1e-5,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
    "gradient_clipping": None,
}
ENGINEERING_PROJECTION_LABEL = "ENGINEERING_ESTIMATE_NOT_A_TRAINING_RESULT"
PUBLIC_FLAGS = frozenset(("--execute", "--i-have-root-throughput-authorization"))

# The production-contract and deferred-receipt mathematical paths use the
# same graph.  Separate CUDA backward executions nevertheless have bounded
# FP32 reduction-order variation on the reviewed RTX 3090 runtime.  These are
# not tunable parameters: they are the frozen pre-timing contract from the
# root diagnostic.  Exact forward/loss equality remains required because both
# are observed before backward; state/gradient comparisons are numerical.
MATHEMATICAL_FP32_TOLERANCES = {
    "forward_max_abs": 0.0,
    "loss_abs": 0.0,
    "gradient_max_abs": 1e-6,
    "model_state_max_abs": 2e-5,
    "optimizer_state_max_abs": 1e-7,
}
MATHEMATICAL_TIMING_GATE = "MATHEMATICAL_TIMING_GATE"
DIAGNOSTIC_NON_AUTHORIZING = "DIAGNOSTIC_NON_AUTHORIZING"
COMPILE_UNAVAILABLE = "COMPILE_UNAVAILABLE"
COMPILE_FAILURE_STAGES = frozenset((
    "variant_construction", "warmup", "measurement", "post_measurement_audit",
))


class FailClosedError(RuntimeError):
    """A provenance, isolation, or equivalence violation; never a score."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exception_class_name(error: BaseException) -> str:
    """Return an unambiguous, receipt-safe exact exception class name."""
    cls = type(error)
    return f"{cls.__module__}.{cls.__qualname__}"


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")).encode("utf-8") + b"\n"


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _require_sha(value: object, label: str) -> str:
    if not _is_sha(value):
        raise FailClosedError(f"{label} must be an exact lowercase SHA-256")
    return str(value)


def _finite(value: object, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FailClosedError("expected finite numeric value")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0) or (nonnegative and number < 0.0):
        raise FailClosedError("numeric finiteness/sign drift")
    return number


def _safe_relative(relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise FailClosedError("canonical relative path is required")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.name in {"", ".", ".."}:
        raise FailClosedError("unsafe relative path")
    return path


def _read_fd_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1 << 20)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _identity(info: os.stat_result) -> tuple[int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size)


def _canonical_regular_bytes(path: Path, *, expected_mode: int | None = None) -> bytes:
    """Read one canonical regular file through a stable no-follow descriptor."""
    absolute = path.absolute()
    try:
        if path.resolve(strict=True) != absolute:
            raise FailClosedError(f"path alias or symlink forbidden: {path}")
        before = os.lstat(absolute)
    except OSError as error:
        raise FailClosedError(f"required file absent or unreadable: {path}") from error
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise FailClosedError("required input is not a canonical regular file")
    if expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode:
        raise FailClosedError("required input mode drift")
    try:
        descriptor = os.open(absolute, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise FailClosedError("cannot open required input without following links") from error
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode) or stat.S_ISLNK(opened.st_mode)
                or _identity(opened) != _identity(before)):
            raise FailClosedError("required input changed between lstat and open")
        if expected_mode is not None and stat.S_IMODE(opened.st_mode) != expected_mode:
            raise FailClosedError("opened input mode drift")
        body = _read_fd_all(descriptor)
    finally:
        os.close(descriptor)
    try:
        after = os.lstat(absolute)
    except OSError as error:
        raise FailClosedError("required input vanished during read") from error
    if (not stat.S_ISREG(after.st_mode) or stat.S_ISLNK(after.st_mode)
            or _identity(after) != _identity(before) or len(body) != before.st_size):
        raise FailClosedError("required input changed during read")
    if expected_mode is not None and stat.S_IMODE(after.st_mode) != expected_mode:
        raise FailClosedError("post-read input mode drift")
    return body


def _closure_digest(hashes: Mapping[str, str]) -> str:
    return _sha(json.dumps({"files": dict(hashes)}, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def validate_closure_payload(value: Mapping[str, Any]) -> dict[str, object]:
    expected = {"paths", "sha256_by_path", "closure_sha256"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FailClosedError("benchmark closure schema drift")
    paths, hashes = value.get("paths"), value.get("sha256_by_path")
    if paths != list(BENCHMARK_CLOSURE) or not isinstance(hashes, Mapping) or set(hashes) != set(BENCHMARK_CLOSURE):
        raise FailClosedError("benchmark closure path map drift")
    if any(not _is_sha(hashes[path]) for path in BENCHMARK_CLOSURE):
        raise FailClosedError("benchmark closure file digest drift")
    if value.get("closure_sha256") != _closure_digest({path: str(hashes[path]) for path in BENCHMARK_CLOSURE}):
        raise FailClosedError("benchmark closure aggregate drift")
    if hashes[WORKORDER_RELATIVE] != WORKORDER_SHA256:
        raise FailClosedError("throughput work-order closure drift")
    if hashes[MODEL_RELATIVE] != MODEL_SHA256:
        raise FailClosedError("frozen TF-SR model closure drift")
    if hashes[NATIVE_ACTIVITY_RELATIVE] != NATIVE_ACTIVITY_SHA256:
        raise FailClosedError("native causal activity dependency closure drift")
    if hashes[LIVE_THROUGHPUT_RELATIVE] != LIVE_THROUGHPUT_SHA256:
        raise FailClosedError("live Phase-D throughput closure drift")
    return {
        "paths": list(BENCHMARK_CLOSURE),
        "sha256_by_path": {path: str(hashes[path]) for path in BENCHMARK_CLOSURE},
        "closure_sha256": str(value["closure_sha256"]),
    }


def compute_benchmark_closure(root: Path) -> dict[str, object]:
    """Descriptor-safely hash the exact benchmark/runtime closure.

    This is intentionally reachable only from a reviewed physical route.  The
    zero-argument plan exposes its declared paths and fixed hashes without
    reading a running result, source dataset, or GPU.
    """
    hashes: dict[str, str] = {}
    for relative in BENCHMARK_CLOSURE:
        _safe_relative(relative)
        mode = 0o444 if relative in {LIVE_THROUGHPUT_RELATIVE, LIVE_THROUGHPUT_SIDECAR_RELATIVE} else None
        hashes[relative] = _sha(_canonical_regular_bytes(root.absolute() / relative, expected_mode=mode))
    return validate_closure_payload({
        "paths": list(BENCHMARK_CLOSURE),
        "sha256_by_path": hashes,
        "closure_sha256": _closure_digest(hashes),
    })


def load_live_baseline_evidence(root: Path) -> dict[str, object]:
    """Load the sole permitted historical runtime input: immutable throughput100."""
    path = root.absolute() / LIVE_THROUGHPUT_RELATIVE
    body = _canonical_regular_bytes(path, expected_mode=0o444)
    if _sha(body) != LIVE_THROUGHPUT_SHA256:
        raise FailClosedError("live throughput body SHA drift")
    sidecar = _canonical_regular_bytes(Path(str(path) + ".sha256"), expected_mode=0o444)
    expected_sidecar = f"{LIVE_THROUGHPUT_SHA256}  {path.name}\n".encode("ascii")
    if sidecar != expected_sidecar:
        raise FailClosedError("live throughput sidecar drift")
    try:
        payload = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise FailClosedError("live throughput JSON malformed") from error
    if not isinstance(payload, Mapping):
        raise FailClosedError("live throughput JSON root drift")
    expected = {
        "schema", "cell", "threshold_steps", "elapsed_seconds", "steps_per_second",
        "eta_seconds_estimate_only", "predecessor", "launch_closure", "resources", "boundaries",
    }
    if (set(payload) != expected or payload.get("schema") != "tfsr_b3st4_ddrop_throughput_v2"
            or payload.get("cell") != "TFSR_B3ST4_DDROP_SEED42"
            or payload.get("threshold_steps") != FROZEN_BASELINE["threshold_steps"]
            or _finite(payload.get("steps_per_second"), positive=True) != FROZEN_BASELINE["steps_per_second"]
            or _finite(payload.get("elapsed_seconds"), positive=True) != FROZEN_BASELINE["elapsed_seconds"]):
        raise FailClosedError("live throughput fixed evidence semantic drift")
    return {
        "relative_path": LIVE_THROUGHPUT_RELATIVE,
        "body_sha256": LIVE_THROUGHPUT_SHA256,
        "steps_per_second": FROZEN_BASELINE["steps_per_second"],
        "elapsed_seconds": FROZEN_BASELINE["elapsed_seconds"],
        "projected_48epoch_seconds": FROZEN_BASELINE["projected_48epoch_seconds"],
    }


@dataclass(frozen=True)
class BenchmarkSpec:
    """Frozen, engineering-only synthetic workload and timing budget."""

    seed: int = 42
    window: int = 50
    calibration_bins: int = 30
    calibration_width: int = 100
    t4_width: int = 4
    output_channels: int = 2
    units: int = 128
    warmup_steps: int = 5
    measured_steps: int = 20
    dtype: str = "float32"

    def __post_init__(self) -> None:
        fixed = {
            "seed": self.seed, "window": self.window, "calibration_bins": self.calibration_bins,
            "calibration_width": self.calibration_width, "t4_width": self.t4_width,
            "output_channels": self.output_channels, "units": self.units, "dtype": self.dtype,
        }
        expected = {
            "seed": 42, "window": 50, "calibration_bins": 30, "calibration_width": 100,
            "t4_width": 4, "output_channels": 2, "units": 128, "dtype": "float32",
        }
        if fixed != expected:
            raise ValueError("benchmark workload topology is frozen")
        if (type(self.warmup_steps) is not int or self.warmup_steps < 1
                or type(self.measured_steps) is not int or self.measured_steps < 20):
            raise ValueError("benchmark requires >=1 warmup and >=20 measured steps")

    def payload(self) -> dict[str, object]:
        return {
            "seed": self.seed, "window": self.window, "calibration": [self.calibration_bins, self.calibration_width],
            "t4_width": self.t4_width, "output_channels": self.output_channels, "units": self.units,
            "warmup_steps": self.warmup_steps, "measured_steps": self.measured_steps, "dtype": self.dtype,
        }


PUBLIC_SPEC = BenchmarkSpec()
EAGER_BATCHES = (32, 64, 128)
TRAINING_STEPS_48_EPOCHS = FROZEN_BASELINE["epochs"] * FROZEN_BASELINE["steps_per_epoch"]


def planned_matrix() -> list[dict[str, object]]:
    """Return the exact ordered matrix without resolving Torch or a device."""
    return [
        {"label": "production_contract_eager_b32", "kind": "production_contract", "execution": "eager", "batch_size": 32},
        {"label": "mathematical_eager_b32", "kind": "mathematical", "execution": "eager", "batch_size": 32},
        {"label": "validation_hoisted_eager_b32", "kind": "validation_hoisted", "execution": "eager", "batch_size": 32},
        {"label": "production_contract_eager_b64", "kind": "production_contract", "execution": "eager", "batch_size": 64},
        {"label": "production_contract_eager_b128", "kind": "production_contract", "execution": "eager", "batch_size": 128},
        {"label": "mathematical_compile_b32", "kind": "mathematical", "execution": "compiled", "batch_size": 32},
        {"label": "mathematical_compile_fastest_eager", "kind": "mathematical", "execution": "compiled", "batch_size": "fastest_non_oom_eager"},
        {"label": "grucell_equivalent_eager_b32", "kind": "grucell_equivalent", "execution": "candidate", "batch_size": 32},
    ]


def dry_plan() -> dict[str, object]:
    """Pure static plan: no Torch import, filesystem read, CUDA call, or write."""
    return {
        "cell": CELL,
        "phase": PHASE,
        "status": "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_BENCHMARK",
        "authorization": "ROOT_AUDIT_REQUIRED",
        "execution_flags_required_together": sorted(PUBLIC_FLAGS),
        "benchmark_spec": PUBLIC_SPEC.payload(),
        "planned_matrix": planned_matrix(),
        "closure": {
            "paths": list(BENCHMARK_CLOSURE), "model_sha256": MODEL_SHA256,
            "native_activity_sha256": NATIVE_ACTIVITY_SHA256,
            "live_throughput_sha256": LIVE_THROUGHPUT_SHA256, "workorder_sha256": WORKORDER_SHA256,
            "non_globbed": True,
        },
        "canonical_output": {
            "root_relative": OUTPUT_ROOT_RELATIVE, "topology": list(OUTPUT_TOPOLOGY),
            "must_be_fresh_before_execution": True,
        },
        "boundaries": {
            "purpose": "ENGINEERING_ONLY", "scientific_result": False, "data_opened": False,
            "checkpoint_opened": False, "target_or_formal": False, "authorizes_training": False,
        },
    }


@dataclass(frozen=True)
class SyntheticBatch:
    """A fully synthetic workload whose T4 field must be a typed capability."""

    x: Any
    calib: Any
    normalized_t4: Any
    target: Any
    valid: Any


def _synthetic_sha(label: str) -> str:
    return _sha(("TF-SR throughput synthetic:" + label).encode("utf-8"))


def validate_synthetic_batch(
    batch: SyntheticBatch,
    *,
    torch: Any,
    model_module: Any,
    spec: BenchmarkSpec = PUBLIC_SPEC,
    batch_size: int,
    device: Any,
) -> None:
    """Prove the benchmark never passes a bare T4 tensor to TF-SR."""
    if not isinstance(batch, SyntheticBatch):
        raise FailClosedError("benchmark requires SyntheticBatch")
    capability_type = model_module.NormalizedT4Batch
    if not isinstance(batch.normalized_t4, capability_type):
        raise FailClosedError("benchmark requires typed NormalizedT4Batch, never bare T4")
    batch.normalized_t4.validate()
    expected_x = (batch_size, spec.window, spec.units)
    expected_calib = (batch_size, spec.calibration_bins, spec.calibration_width, spec.units)
    expected_target = (batch_size, spec.window, spec.output_channels)
    if (tuple(batch.x.shape) != expected_x or tuple(batch.calib.shape) != expected_calib
            or tuple(batch.target.shape) != expected_target or tuple(batch.valid.shape) != expected_target[:2]
            or tuple(batch.normalized_t4.tensor.shape) != (batch_size, spec.units, spec.t4_width)):
        raise FailClosedError("synthetic workload shape drift")
    tensors = (batch.x, batch.calib, batch.normalized_t4.tensor, batch.target)
    if any(not tensor.is_floating_point() or tensor.device != device or not bool(torch.isfinite(tensor).all().item())
           for tensor in tensors):
        raise FailClosedError("synthetic workload finite/device drift")
    if batch.valid.dtype != torch.bool or batch.valid.device != device or not bool(batch.valid.all().item()):
        raise FailClosedError("synthetic valid-bin mask drift")


def make_synthetic_batch(
    *,
    torch: Any,
    model_module: Any,
    batch_size: int,
    device: Any,
    spec: BenchmarkSpec = PUBLIC_SPEC,
) -> SyntheticBatch:
    """Make deterministic finite synthetic tensors and one typed T4 capability."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("synthetic batch size must be a positive exact integer")
    generator = torch.Generator(device=device)
    generator.manual_seed(spec.seed)
    dtype = torch.float32
    x = torch.randn((batch_size, spec.window, spec.units), generator=generator, device=device, dtype=dtype)
    calib = torch.randn((batch_size, spec.calibration_bins, spec.calibration_width, spec.units),
                        generator=generator, device=device, dtype=dtype)
    raw_t4 = torch.randn((batch_size, spec.units, spec.t4_width), generator=generator, device=device, dtype=dtype)
    target = torch.randn((batch_size, spec.window, spec.output_channels), generator=generator, device=device, dtype=dtype)
    valid = torch.ones((batch_size, spec.window), device=device, dtype=torch.bool)
    normalizer = model_module.T4Normalizer(
        torch.zeros((spec.t4_width,), device=device, dtype=dtype),
        torch.ones((spec.t4_width,), device=device, dtype=dtype),
        _synthetic_sha("raw-t4"), _synthetic_sha("normalized-t4"),
    )
    ids = tuple(f"synthetic-unit-{index:03d}" for index in range(spec.units))
    normalized_t4 = normalizer(
        raw_t4,
        roster_digest=_synthetic_sha("roster"),
        ordered_unit_ids=ids,
        lineage=("synthetic-throughput-only", "seed-42", "no-dataset"),
    )
    value = SyntheticBatch(x=x, calib=calib, normalized_t4=normalized_t4, target=target, valid=valid)
    validate_synthetic_batch(value, torch=torch, model_module=model_module, spec=spec, batch_size=batch_size, device=device)
    return value


# Explicit original-GRU -> benchmark-local GRUCell state names.  This map is
# included in the receipt so reviewers can see that no tensor was omitted.
GRU_TENSOR_COPY_MAP = {
    "gru.weight_ih_l0": "gru.cell.weight_ih",
    "gru.weight_hh_l0": "gru.cell.weight_hh",
    "gru.bias_ih_l0": "gru.cell.bias_ih",
    "gru.bias_hh_l0": "gru.cell.bias_hh",
}
_CANDIDATE_TO_BASELINE_GRU = {candidate: baseline for baseline, candidate in GRU_TENSOR_COPY_MAP.items()}


def build_grucell_equivalent(torch: Any, baseline_gru: Any) -> Any:
    """Return a benchmark-local one-layer GRUCell sequence implementation."""
    if (getattr(baseline_gru, "num_layers", None) != 1 or getattr(baseline_gru, "bidirectional", None) is not False
            or getattr(baseline_gru, "batch_first", None) is not True or getattr(baseline_gru, "dropout", None) != 0.0):
        raise FailClosedError("frozen GRU topology is not the expected one-layer batch-first GRU")

    class _GRUCellSequence(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cell = torch.nn.GRUCell(baseline_gru.input_size, baseline_gru.hidden_size, bias=True)

        def forward(self, input_tensor: Any, hidden: Any | None = None) -> tuple[Any, Any]:
            if input_tensor.ndim != 3 or input_tensor.shape[-1] != baseline_gru.input_size:
                raise ValueError("GRUCell equivalent input shape drift")
            if hidden is None:
                current = input_tensor.new_zeros((input_tensor.shape[0], baseline_gru.hidden_size))
            else:
                if tuple(hidden.shape) != (1, input_tensor.shape[0], baseline_gru.hidden_size):
                    raise ValueError("GRUCell equivalent hidden shape drift")
                current = hidden[0]
            outputs: list[Any] = []
            for index in range(input_tensor.shape[1]):
                current = self.cell(input_tensor[:, index], current)
                outputs.append(current)
            return torch.stack(outputs, dim=1), current.unsqueeze(0)

    replacement = _GRUCellSequence().to(device=baseline_gru.weight_ih_l0.device, dtype=baseline_gru.weight_ih_l0.dtype)
    source = dict(baseline_gru.named_parameters(recurse=False))
    destination = dict(replacement.named_parameters())
    expected_source = {name.removeprefix("gru.") for name in GRU_TENSOR_COPY_MAP}
    expected_destination = {name.removeprefix("gru.") for name in GRU_TENSOR_COPY_MAP.values()}
    if set(source) != expected_source or set(destination) != expected_destination:
        raise FailClosedError("GRU tensor copy-map topology drift")
    with torch.no_grad():
        for baseline_name, candidate_name in GRU_TENSOR_COPY_MAP.items():
            source_name, destination_name = baseline_name.removeprefix("gru."), candidate_name.removeprefix("gru.")
            if source[source_name].shape != destination[destination_name].shape:
                raise FailClosedError("GRU tensor copy-map shape drift")
            destination[destination_name].copy_(source[source_name])
    return replacement


def build_grucell_candidate(torch: Any, baseline_model: Any) -> Any:
    """Copy the whole model, then replace only GRU with explicitly copied cell."""
    candidate = copy.deepcopy(baseline_model)
    candidate.gru = build_grucell_equivalent(torch, baseline_model.gru)
    return candidate


def _tensor_digest(torch: Any, state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        if not torch.is_tensor(tensor):
            raise FailClosedError("state digest accepts tensors only")
        material = tensor.detach().cpu().contiguous().reshape(-1)
        digest.update(str(key).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(material.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _canonical_model_state(torch: Any, model: Any, *, candidate: bool) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, tensor in model.state_dict().items():
        canonical = _CANDIDATE_TO_BASELINE_GRU.get(name, name) if candidate else name
        if canonical in state:
            raise FailClosedError("candidate state canonical-name collision")
        state[canonical] = tensor
    return state


def model_state_digest(torch: Any, model: Any, *, candidate: bool = False) -> str:
    return _tensor_digest(torch, _canonical_model_state(torch, model, candidate=candidate))


def _digest_value(torch: Any, value: Any, digest: Any) -> None:
    if torch.is_tensor(value):
        material = value.detach().cpu().contiguous().reshape(-1)
        digest.update(b"tensor")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(material.view(torch.uint8).numpy().tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=str):
            digest.update(str(key).encode("utf-8"))
            _digest_value(torch, value[key], digest)
    elif isinstance(value, (tuple, list)):
        digest.update(b"sequence")
        for item in value:
            _digest_value(torch, item, digest)
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    else:
        raise FailClosedError("unsupported optimizer-state value")


def optimizer_state_digest(torch: Any, optimizer: Any) -> str:
    digest = hashlib.sha256()
    _digest_value(torch, optimizer.state_dict(), digest)
    return digest.hexdigest()


def _all_finite_parameters(torch: Any, model: Any) -> bool:
    return all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters())


def _all_finite_gradients(torch: Any, model: Any) -> bool:
    return all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
               for parameter in model.parameters() if parameter.requires_grad)


def _gradient_snapshot(model: Any) -> dict[str, Any]:
    """Detach gradients before the required post-step ``zero_grad`` call."""
    result: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise FailClosedError("benchmark gradient snapshot missing parameter gradient")
            result[name] = parameter.grad.detach().clone()
    return result


def _all_finite_optimizer(torch: Any, optimizer: Any) -> bool:
    def walk(value: Any) -> bool:
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all().item())
        if isinstance(value, Mapping):
            return all(walk(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return all(walk(item) for item in value)
        return True
    return walk(optimizer.state_dict())


def _max_tensor_difference(torch: Any, left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    if set(left) != set(right):
        raise FailClosedError("equivalence state key drift")
    maximum = 0.0
    for key in sorted(left):
        lhs, rhs = left[key], right[key]
        if not torch.is_tensor(lhs) or not torch.is_tensor(rhs) or lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
            raise FailClosedError("equivalence tensor topology drift")
        maximum = max(maximum, float((lhs.detach() - rhs.detach()).abs().max().item()))
    return maximum


def _flatten_optimizer_tensors(torch: Any, value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if torch.is_tensor(value):
        result[prefix] = value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            result.update(_flatten_optimizer_tensors(torch, value[key], prefix + "/" + str(key)))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            result.update(_flatten_optimizer_tensors(torch, item, prefix + f"/{index}"))
    return result


def optimizer_state_max_difference(torch: Any, left: Any, right: Any) -> float:
    return _max_tensor_difference(torch, _flatten_optimizer_tensors(torch, left), _flatten_optimizer_tensors(torch, right))


def _write_full(descriptor: int, body: bytes) -> None:
    view = memoryview(body)
    while view:
        count = os.write(descriptor, view)
        if count <= 0:
            raise OSError("short benchmark receipt write")
        view = view[count:]


def _canonical_directory_identity(path: Path) -> tuple[int, int]:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise FailClosedError("benchmark output directory is absent") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise FailClosedError("benchmark output directory is not canonical")
    return (info.st_dev, info.st_ino)


@dataclass(frozen=True)
class OutputArtifactRoot:
    directory: Path
    identity: tuple[int, int]
    parent: Path
    parent_identity: tuple[int, int]

    def _assert_identity(self) -> None:
        if _canonical_directory_identity(self.directory) != self.identity:
            raise FailClosedError("benchmark output root identity drift")
        try:
            descriptor = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError as error:
            raise FailClosedError("cannot open benchmark output parent") from error
        try:
            info = os.fstat(descriptor)
            named = os.stat(self.directory.name, dir_fd=descriptor, follow_symlinks=False)
            if ((info.st_dev, info.st_ino) != self.parent_identity or not stat.S_ISDIR(named.st_mode)
                    or stat.S_ISLNK(named.st_mode) or (named.st_dev, named.st_ino) != self.identity):
                raise FailClosedError("benchmark output parent/named-root identity drift")
        finally:
            os.close(descriptor)

    def publish_receipt(self, payload: Mapping[str, Any]) -> str:
        """Publish the only allowed engineering output as an O_EXCL transaction."""
        validate_engineering_receipt(payload)
        body = _json_bytes(payload)
        digest = _sha(body)
        self._assert_identity()
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        created: list[tuple[str, int, int]] = []
        try:
            for leaf in ("receipt.json", "receipt.json.sha256"):
                try:
                    os.stat(leaf, dir_fd=descriptor, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                raise FailClosedError("benchmark receipt output collision")
            payloads = (
                ("receipt.json", body),
                ("receipt.json.sha256", f"{digest}  receipt.json\n".encode("ascii")),
            )
            for leaf, content in payloads:
                fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=descriptor)
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode):
                        raise FailClosedError("receipt O_EXCL type drift")
                    created.append((leaf, info.st_dev, info.st_ino))
                    _write_full(fd, content)
                    os.fchmod(fd, 0o444)
                    os.fsync(fd)
                finally:
                    os.close(fd)
            os.fsync(descriptor)
            # Same-FD reload validates both content and immutable modes before
            # this method can report publication success.
            reloaded_body = self._read_pair("receipt.json", digest)
            reloaded = json.loads(reloaded_body)
            if not isinstance(reloaded, Mapping):
                raise FailClosedError("published receipt JSON root drift")
            validate_engineering_receipt(reloaded)
            self._assert_identity()
            parent_fd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            return digest
        except BaseException:
            for leaf, device, inode in reversed(created):
                try:
                    info = os.stat(leaf, dir_fd=descriptor, follow_symlinks=False)
                    if (info.st_dev, info.st_ino) == (device, inode):
                        os.unlink(leaf, dir_fd=descriptor)
                except OSError:
                    pass
            try:
                os.fsync(descriptor)
            except OSError:
                pass
            raise
        finally:
            os.close(descriptor)

    def _read_pair(self, name: str, expected_sha256: str | None = None) -> bytes:
        self._assert_identity()
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            def read(leaf: str) -> bytes:
                try:
                    fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
                except OSError as error:
                    raise FailClosedError("published benchmark receipt leaf missing") from error
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
                        raise FailClosedError("published benchmark receipt mode/type drift")
                    return _read_fd_all(fd)
                finally:
                    os.close(fd)
            body = read(name)
            digest = _sha(body)
            if expected_sha256 is not None and digest != expected_sha256:
                raise FailClosedError("published benchmark receipt SHA drift")
            if read(name + ".sha256") != f"{digest}  {name}\n".encode("ascii"):
                raise FailClosedError("published benchmark receipt sidecar drift")
            return body
        finally:
            os.close(descriptor)


def reserve_output_root(parent: Path, name: str) -> OutputArtifactRoot:
    """Reserve a fresh named output directory; used only after reviewed timing."""
    if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
        raise ValueError("benchmark output name must be one safe component")
    parent = parent.absolute()
    parent_identity = _canonical_directory_identity(parent)
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != parent_identity:
            raise FailClosedError("benchmark output parent identity drift")
        try:
            os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FailClosedError("fresh canonical benchmark output root required")
        os.mkdir(name, 0o755, dir_fd=descriptor)
        os.fsync(descriptor)
        info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise FailClosedError("reserved benchmark output root invalid")
        identity = (info.st_dev, info.st_ino)
    finally:
        os.close(descriptor)
    return OutputArtifactRoot(parent / name, identity, parent, parent_identity)


def canonical_output_parent(root: Path) -> tuple[Path, str]:
    relative = _safe_relative(OUTPUT_ROOT_RELATIVE)
    path = root.absolute() / relative
    return path.parent, path.name


def require_canonical_output_fresh(root: Path) -> None:
    parent, name = canonical_output_parent(root)
    # Reading a parent directory is not a target/data/checkpoint access.
    parent_identity = _canonical_directory_identity(parent)
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) != parent_identity:
            raise FailClosedError("benchmark output parent identity drift")
        try:
            os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise FailClosedError("canonical benchmark output root must be fresh")
    finally:
        os.close(descriptor)


def _runtime_seed(torch: Any, seed: int) -> None:
    """Reset only runtime RNGs so each variant sees the same dropout stream."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _validate_exact_adam(optimizer: Any) -> None:
    if optimizer.__class__.__name__ != FROZEN_ADAM["cls"]:
        raise FailClosedError("benchmark optimizer class drift")
    groups = optimizer.param_groups
    if len(groups) != 1:
        raise FailClosedError("benchmark optimizer group count drift")
    group = groups[0]
    if (group.get("lr") != FROZEN_ADAM["lr"] or tuple(group.get("betas", ())) != tuple(FROZEN_ADAM["betas"])
            or group.get("eps") != FROZEN_ADAM["eps"] or group.get("weight_decay") != FROZEN_ADAM["weight_decay"]
            or group.get("amsgrad") is not FROZEN_ADAM["amsgrad"] or group.get("maximize") is not False
            or group.get("foreach") is not None or group.get("capturable") is not False
            or group.get("differentiable") is not False or group.get("fused") is not None):
        raise FailClosedError("frozen Adam configuration drift")


def _new_frozen_adam(torch: Any, model: Any) -> Any:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FROZEN_ADAM["lr"],
        betas=tuple(FROZEN_ADAM["betas"]),
        eps=FROZEN_ADAM["eps"],
        weight_decay=FROZEN_ADAM["weight_decay"],
        amsgrad=FROZEN_ADAM["amsgrad"],
    )
    _validate_exact_adam(optimizer)
    return optimizer


def _production_contract_step(
    torch: Any,
    model_module: Any,
    model: Any,
    optimizer: Any,
    batch: SyntheticBatch,
    *,
    capture_gradients: bool = False,
) -> dict[str, Any]:
    """Copy the live ordinary-step contract, including synchronizing receipt reads.

    This intentionally retains the loss finite `.item()` check plus every
    loss/dropout/survivor/gain conversion performed by
    ``TorchTrainingBackend.train_step`` outside its epoch-boundary proof.
    """
    optimizer.zero_grad(set_to_none=True)
    model.train(True)
    prediction = model(batch.x, batch.calib, batch.normalized_t4)
    loss = model_module.TFSRDecoder.dense_valid_bin_mse(prediction, batch.target, batch.valid)
    if not torch.isfinite(loss).item():
        raise FailClosedError("production-contract synthetic loss is nonfinite")
    loss.backward()
    gradients_finite = _all_finite_gradients(torch, model) if capture_gradients else None
    gradients = _gradient_snapshot(model) if capture_gradients else None
    optimizer.step()
    gain, survivor, dropout_p = model.last_unit_gain_mask, model.last_unit_survivor_mask, model.last_dropout_p
    if gain is None or survivor is None or dropout_p is None:
        raise FailClosedError("production-contract dropout receipt extraction missing")
    # Exact ordinary production receipt conversions: these are specifically
    # what the mathematical timing cell is allowed to defer, and nothing else.
    kept = int(survivor.sum().item())
    receipt = {
        "loss": float(loss.item()),
        "dropout_p": float(dropout_p.item()),
        "kept": kept,
        "dropped": int(survivor.numel() - kept),
        "all_zero_examples": int((survivor.sum(dim=1) == 0).sum().item()),
        "population_examples": int(survivor.shape[0]),
        "max_gain": float(gain.max().item()),
    }
    optimizer.zero_grad(set_to_none=True)
    return {
        "prediction": prediction.detach(), "loss_tensor": loss.detach(), "receipt": receipt,
        "finite": {"forward": None, "loss": True,
                   "gradient": gradients_finite}, "gradient_state": gradients,
    }


@dataclass(frozen=True)
class _DeferredReceipt:
    loss: Any
    dropout_p: Any
    survivor: Any
    gain: Any

    def materialize(self) -> dict[str, object]:
        """Perform the production receipt conversions only after the timer ends."""
        return {
            "loss": float(self.loss.item()),
            "dropout_p": float(self.dropout_p.item()),
            "kept": int(self.survivor.sum().item()),
            "dropped": int(self.survivor.numel() - self.survivor.sum().item()),
            "all_zero_examples": int((self.survivor.sum(dim=1) == 0).sum().item()),
            "population_examples": int(self.survivor.shape[0]),
            "max_gain": float(self.gain.max().item()),
        }


def _mathematical_step(
    torch: Any,
    model_module: Any,
    model: Any,
    optimizer: Any,
    batch: SyntheticBatch,
    *,
    capture_gradients: bool = False,
) -> dict[str, Any]:
    """Same training math as production, with only receipt conversion deferred.

    In particular, the live loss-finiteness guard stays in this cell.  It is a
    training-contract validation, not a post-step receipt field; removing it
    would make this cell a second, mixed validation-hoisting treatment rather
    than the narrow ``deferred receipt`` comparison specified by the work
    order.
    """
    optimizer.zero_grad(set_to_none=True)
    model.train(True)
    prediction = model(batch.x, batch.calib, batch.normalized_t4)
    loss = model_module.TFSRDecoder.dense_valid_bin_mse(prediction, batch.target, batch.valid)
    if not torch.isfinite(loss).item():
        raise FailClosedError("mathematical synthetic loss is nonfinite")
    loss.backward()
    gradients_finite = _all_finite_gradients(torch, model) if capture_gradients else None
    gradients = _gradient_snapshot(model) if capture_gradients else None
    optimizer.step()
    gain, survivor, dropout_p = model.last_unit_gain_mask, model.last_unit_survivor_mask, model.last_dropout_p
    if gain is None or survivor is None or dropout_p is None:
        raise FailClosedError("mathematical-step dropout tensors missing")
    deferred = _DeferredReceipt(loss.detach(), dropout_p.detach(), survivor.detach(), gain.detach())
    optimizer.zero_grad(set_to_none=True)
    return {
        "prediction": prediction.detach(), "loss_tensor": loss.detach(), "deferred_receipt": deferred,
        "finite_loss_tensor": True, "gradient_finite": gradients_finite,
        "gradient_state": gradients,
    }


def _validate_hoisted_inputs(torch: Any, model_module: Any, model: Any, batch: SyntheticBatch) -> None:
    """Pay frozen typed-input/loss validation once outside the timed loop."""
    model._validate(batch.x, batch.calib, batch.normalized_t4)
    # B3S.forward contains its own capability/calibration checks; exercise it
    # once without timing so the hoisted route does not silently admit a
    # malformed typed input.
    identity = model.b3s(batch.calib, batch.normalized_t4)
    if tuple(identity.shape) != (batch.x.shape[0], batch.x.shape[2], 50):
        raise FailClosedError("validation-hoisted B3S shape drift")
    model_module.TFSRDecoder.dense_valid_bin_mse(torch.zeros_like(batch.target), batch.target, batch.valid)


def _validation_hoisted_forward(torch: Any, model: Any, batch: SyntheticBatch) -> Any:
    """Benchmark-local call-through of the frozen submodules after validation.

    No module is monkeypatched or modified.  The exact B3S layers, activity
    encoder, token MLP, Cell-D dropout, attention, GRU, norms, FFN, and head
    are called in the same order as ``TFSRDecoder.forward``; only repeated
    Python validation/finite scans are absent from the timed loop.
    """
    identity_features = model.b3s.pre_pool(batch.calib.permute(0, 1, 3, 2)).mean(dim=1)
    identity = model.b3s.post_pool(torch.cat((identity_features, batch.normalized_t4.tensor), dim=-1))
    activity = model.activity_encoder(batch.x)
    pre_mask_tokens = model.unit_mlp(
        torch.cat((activity, identity[:, None].expand(-1, 50, -1, -1)), dim=-1)
    )
    tokens = model._mask(pre_mask_tokens)
    survivor = model.last_unit_survivor_mask
    if survivor is None:
        raise FailClosedError("validation-hoisted Cell-D survivor mask missing")
    mass, _activity_mass = model._population_mass(activity, survivor)
    hidden = batch.x.new_zeros(1, batch.x.shape[0], 256)
    outputs: list[Any] = []
    for time_index in range(50):
        queries = model.query_base[None] + model.state_query(hidden[0]).view(batch.x.shape[0], 2, 256)
        read, _ = model.attn(queries, tokens[:, time_index], tokens[:, time_index], need_weights=False)
        read = model.norm1(queries + read)
        read = model.norm2(read + model.ffn(read))
        step = torch.cat((read.flatten(1), mass[:, time_index]), dim=1).unsqueeze(1)
        _ignored, hidden = model.gru(step, hidden)
        outputs.append(model.head(hidden[0]))
    # Match the frozen forward's capture_diagnostics=False postconditions.
    model.last_pre_mask_tokens = None
    model.last_post_mask_tokens = None
    model.last_activity_features = None
    model.last_activity_mass = None
    model.last_mass = None
    model.last_readin = None
    model.last_prediction = None
    model.last_states = None
    return torch.stack(outputs, dim=1)


def _validation_hoisted_dense_mse(torch: Any, prediction: Any, batch: SyntheticBatch) -> Any:
    """Exact dense valid-bin MSE arithmetic after its contract was prechecked."""
    per_bin_mse = (prediction - batch.target).square().mean(dim=-1)
    valid_count = batch.valid.sum()
    return (per_bin_mse * batch.valid.to(dtype=prediction.dtype)).sum() / valid_count


def _validation_hoisted_step(
    torch: Any,
    model_module: Any,
    model: Any,
    optimizer: Any,
    batch: SyntheticBatch,
    *,
    capture_gradients: bool = False,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    model.train(True)
    prediction = _validation_hoisted_forward(torch, model, batch)
    loss = _validation_hoisted_dense_mse(torch, prediction, batch)
    loss.backward()
    gradients_finite = _all_finite_gradients(torch, model) if capture_gradients else None
    gradients = _gradient_snapshot(model) if capture_gradients else None
    optimizer.step()
    gain, survivor, dropout_p = model.last_unit_gain_mask, model.last_unit_survivor_mask, model.last_dropout_p
    if gain is None or survivor is None or dropout_p is None:
        raise FailClosedError("validation-hoisted dropout tensors missing")
    deferred = _DeferredReceipt(loss.detach(), dropout_p.detach(), survivor.detach(), gain.detach())
    optimizer.zero_grad(set_to_none=True)
    return {
        "prediction": prediction.detach(), "loss_tensor": loss.detach(), "deferred_receipt": deferred,
        "finite_loss_tensor": None, "gradient_finite": gradients_finite,
        "gradient_state": gradients,
    }


def _materialize_step_observation(torch: Any, observation: Mapping[str, Any]) -> dict[str, object]:
    """Turn post-step tensors into receipt scalars only outside a timing region."""
    if "receipt" in observation:
        receipt = observation["receipt"]
        if not isinstance(receipt, Mapping):
            raise FailClosedError("production receipt extraction type drift")
        finite = observation.get("finite")
        if not isinstance(finite, Mapping) or set(finite) != {"forward", "loss", "gradient"}:
            raise FailClosedError("production finite evidence drift")
        forward = bool(torch.isfinite(observation["prediction"]).all().item()) if finite["forward"] is None else bool(finite["forward"])
        if finite["gradient"] is None:
            return {"receipt": dict(receipt), "finite": {"forward": forward, "loss": bool(finite["loss"]), "gradient": None}}
        return {"receipt": dict(receipt), "finite": {"forward": forward, "loss": bool(finite["loss"]), "gradient": bool(finite["gradient"])}}
    deferred = observation.get("deferred_receipt")
    finite_loss = observation.get("finite_loss_tensor")
    gradient = observation.get("gradient_finite")
    prediction = observation.get("prediction")
    loss_tensor = observation.get("loss_tensor")
    if (not isinstance(deferred, _DeferredReceipt) or finite_loss not in {None, True}
            or not torch.is_tensor(prediction) or not torch.is_tensor(loss_tensor)):
        raise FailClosedError("deferred receipt evidence drift")
    finite = {
        "forward": bool(torch.isfinite(prediction).all().item()),
        # The mathematical route already paid this exact live guard in its
        # timed body; validation-hoisted deliberately evaluates it only here.
        "loss": bool(finite_loss) if finite_loss is True else bool(torch.isfinite(loss_tensor).item()),
        "gradient": None if gradient is None else gradient is True,
    }
    return {"receipt": deferred.materialize(), "finite": finite}


def _snapshot_cpu_state(torch: Any, model: Any) -> dict[str, Any]:
    return {name: tensor.detach().cpu().contiguous().clone() for name, tensor in model.state_dict().items()}


def _fresh_model_from_snapshot(torch: Any, model_module: Any, snapshot: Mapping[str, Any], device: Any) -> Any:
    model = model_module.TFSRDecoder(capture_diagnostics=False).to(device)
    model.load_state_dict({name: tensor.to(device=device) for name, tensor in snapshot.items()}, strict=True)
    if model.capture_diagnostics is not False:
        raise FailClosedError("benchmark model diagnostics configuration drift")
    return model


def _max_gradient_difference(torch: Any, left: Mapping[str, Any], right: Mapping[str, Any], *, candidate_right: bool = False) -> float:
    right_params = {
        (_CANDIDATE_TO_BASELINE_GRU.get(name, name) if candidate_right else name): value
        for name, value in right.items()
    }
    if set(left) != set(right_params):
        raise FailClosedError("equivalence gradient parameter topology drift")
    maximum = 0.0
    for name in sorted(left):
        lhs, rhs = left[name], right_params[name]
        if not torch.is_tensor(lhs) or not torch.is_tensor(rhs) or lhs.shape != rhs.shape:
            raise FailClosedError("equivalence missing gradient")
        maximum = max(maximum, float((lhs - rhs).abs().max().item()))
    return maximum


def _comparison_report(
    torch: Any,
    *,
    baseline_model: Any,
    candidate_model: Any,
    baseline_optimizer: Any,
    candidate_optimizer: Any,
    baseline_observation: Mapping[str, Any],
    candidate_observation: Mapping[str, Any],
    candidate_is_grucell: bool,
    comparison_role: str,
    label: str,
) -> dict[str, object]:
    prediction_left, prediction_right = baseline_observation["prediction"], candidate_observation["prediction"]
    loss_left, loss_right = baseline_observation["loss_tensor"], candidate_observation["loss_tensor"]
    if not all(torch.is_tensor(item) for item in (prediction_left, prediction_right, loss_left, loss_right)):
        raise FailClosedError("equivalence forward/loss tensors missing")
    forward_difference = float((prediction_left - prediction_right).abs().max().item())
    loss_difference = float((loss_left - loss_right).abs().item())
    baseline_gradients, candidate_gradients = baseline_observation.get("gradient_state"), candidate_observation.get("gradient_state")
    if not isinstance(baseline_gradients, Mapping) or not isinstance(candidate_gradients, Mapping):
        raise FailClosedError("equivalence gradient snapshots missing")
    gradient_difference = _max_gradient_difference(torch, baseline_gradients, candidate_gradients,
                                                    candidate_right=candidate_is_grucell)
    model_difference = _max_tensor_difference(
        torch,
        _canonical_model_state(torch, baseline_model, candidate=False),
        _canonical_model_state(torch, candidate_model, candidate=candidate_is_grucell),
    )
    optimizer_difference = optimizer_state_max_difference(
        torch, baseline_optimizer.state_dict(), candidate_optimizer.state_dict(),
    )
    finite = {
        "baseline_model": _all_finite_parameters(torch, baseline_model),
        "candidate_model": _all_finite_parameters(torch, candidate_model),
        "baseline_optimizer": _all_finite_optimizer(torch, baseline_optimizer),
        "candidate_optimizer": _all_finite_optimizer(torch, candidate_optimizer),
    }
    if not all(finite.values()):
        raise FailClosedError("equivalence nonfinite post-step state")
    differences = {
        "forward_max_abs": forward_difference,
        "loss_abs": loss_difference,
        "gradient_max_abs": gradient_difference,
        "model_state_max_abs": model_difference,
        "optimizer_state_max_abs": optimizer_difference,
    }
    exact_match = all(value == 0.0 for value in differences.values())
    within_tolerance = all(
        differences[key] <= MATHEMATICAL_FP32_TOLERANCES[key]
        for key in MATHEMATICAL_FP32_TOLERANCES
    )
    return {
        "label": label,
        "same_initial_model_state": True,
        "same_initial_fresh_adam_state": True,
        "differences": differences,
        "all_finite": finite,
        "exact_match": exact_match,
        "fp32_tolerance_map": dict(MATHEMATICAL_FP32_TOLERANCES),
        "within_tolerance": within_tolerance,
        "comparison_role": comparison_role,
        "gru_tensor_copy_map": dict(GRU_TENSOR_COPY_MAP) if candidate_is_grucell else None,
    }


CELL_SCHEMA = "tfsr_b3st4_ddrop_throughput_cell_v1"
_CELL_KEYS = {
    "schema", "label", "kind", "execution", "batch_size", "units", "dtype", "warmup_steps",
    "measured_steps", "status", "cuda_oom", "oom_error_sha256", "typed_normalized_t4",
    "compile_exception_class", "compile_exception_repr_sha256", "compile_failure_stage",
    "post_error_cuda_cleanup_reset",
    "initial_model_state_sha256", "initial_optimizer_state_sha256", "state_before_measure_sha256",
    "optimizer_before_measure_sha256", "fresh_model_from_reference", "fresh_optimizer_state",
    "reused_warmed_state", "finite", "post_measurement_finite", "timing", "resources", "runtime",
    "equivalence",
}


def _validate_runtime(value: object) -> dict[str, object]:
    expected = {"gpu", "torch_version", "cuda_version", "cudnn_version", "cpu_threads"}
    if not isinstance(value, Mapping) or set(value) != expected or value.get("gpu") != FROZEN_GPU0:
        raise FailClosedError("benchmark runtime GPU contract drift")
    if not all(isinstance(value.get(name), (str, type(None))) for name in ("torch_version", "cuda_version", "cudnn_version")):
        raise FailClosedError("benchmark runtime version schema drift")
    threads = value.get("cpu_threads")
    if not isinstance(threads, Mapping) or threads != {"intraop": 1, "interop": 1}:
        raise FailClosedError("benchmark CPU thread-pool contract drift")
    return {
        "gpu": dict(FROZEN_GPU0), "torch_version": value["torch_version"],
        "cuda_version": value["cuda_version"], "cudnn_version": value["cudnn_version"],
        "cpu_threads": {"intraop": 1, "interop": 1},
    }


def _validate_equivalence(
    value: object,
    *,
    require_mathematical_timing_gate: bool,
    require_gru_map: bool,
) -> dict[str, object]:
    """Recompute comparison facts and enforce the one admissible timing gate.

    ``exact_match`` is reporting evidence, never a substitute for its raw
    differences.  Only the mathematical deferred-receipt cell may use the
    frozen FP32 tolerance contract to proceed to timing.  Hoisted and GRUCell
    comparisons retain the same raw evidence but are diagnostic-only even if
    they happen to fall inside that numerical band.
    """
    expected = {
        "label", "same_initial_model_state", "same_initial_fresh_adam_state", "differences", "all_finite",
        "exact_match", "fp32_tolerance_map", "within_tolerance", "comparison_role", "gru_tensor_copy_map",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FailClosedError("benchmark equivalence schema drift")
    if value.get("same_initial_model_state") is not True or value.get("same_initial_fresh_adam_state") is not True:
        raise FailClosedError("benchmark equivalence initial-state drift")
    differences = value.get("differences")
    difference_keys = {"forward_max_abs", "loss_abs", "gradient_max_abs", "model_state_max_abs", "optimizer_state_max_abs"}
    if not isinstance(differences, Mapping) or set(differences) != difference_keys:
        raise FailClosedError("benchmark equivalence difference schema drift")
    checked_differences = {key: _finite(differences[key], nonnegative=True) for key in difference_keys}
    finite = value.get("all_finite")
    if not isinstance(finite, Mapping) or set(finite) != {
        "baseline_model", "candidate_model", "baseline_optimizer", "candidate_optimizer",
    } or any(item is not True for item in finite.values()):
        raise FailClosedError("benchmark equivalence finite proof drift")
    if type(value.get("exact_match")) is not bool:
        raise FailClosedError("benchmark equivalence exact-match type drift")
    actual_exact = all(item == 0.0 for item in checked_differences.values())
    if value["exact_match"] is not actual_exact:
        raise FailClosedError("benchmark equivalence exact-match drift")
    tolerance_map = value.get("fp32_tolerance_map")
    if not isinstance(tolerance_map, Mapping) or set(tolerance_map) != set(MATHEMATICAL_FP32_TOLERANCES):
        raise FailClosedError("benchmark equivalence FP32 tolerance-map schema drift")
    # Require the receipt to preserve the literal float map rather than a
    # semantically similar widened/narrowed integer-or-float variant.
    if any(type(tolerance_map[key]) is not float for key in MATHEMATICAL_FP32_TOLERANCES) or dict(tolerance_map) != MATHEMATICAL_FP32_TOLERANCES:
        raise FailClosedError("benchmark equivalence FP32 tolerance-map drift")
    actual_within_tolerance = all(
        checked_differences[key] <= MATHEMATICAL_FP32_TOLERANCES[key]
        for key in MATHEMATICAL_FP32_TOLERANCES
    )
    if type(value.get("within_tolerance")) is not bool or value["within_tolerance"] is not actual_within_tolerance:
        raise FailClosedError("benchmark equivalence within-tolerance drift")
    expected_role = MATHEMATICAL_TIMING_GATE if require_mathematical_timing_gate else DIAGNOSTIC_NON_AUTHORIZING
    if value.get("comparison_role") != expected_role:
        raise FailClosedError("benchmark equivalence comparison-role drift")
    if require_mathematical_timing_gate and not actual_within_tolerance:
        raise FailClosedError("benchmark mathematical FP32 tolerance gate drift")
    mapping = value.get("gru_tensor_copy_map")
    if require_gru_map:
        if mapping != GRU_TENSOR_COPY_MAP:
            raise FailClosedError("benchmark GRU tensor copy-map drift")
    elif mapping is not None:
        raise FailClosedError("non-GRU equivalence carries a GRU copy map")
    if not isinstance(value.get("label"), str) or not value["label"]:
        raise FailClosedError("benchmark equivalence label drift")
    return {
        "label": str(value["label"]), "same_initial_model_state": True,
        "same_initial_fresh_adam_state": True, "differences": checked_differences,
        "all_finite": {key: True for key in finite}, "exact_match": actual_exact,
        "fp32_tolerance_map": dict(MATHEMATICAL_FP32_TOLERANCES),
        "within_tolerance": actual_within_tolerance, "comparison_role": expected_role,
        "gru_tensor_copy_map": dict(GRU_TENSOR_COPY_MAP) if require_gru_map else None,
    }


def validate_cell_payload(
    value: Mapping[str, Any],
    *,
    label: str,
    kind: str,
    execution: str,
    batch_size: int,
    reference_model_sha256: str,
    reference_optimizer_sha256: str,
) -> dict[str, object]:
    """Validate a measurement record before it enters any engineering receipt."""
    if (not isinstance(value, Mapping) or set(value) != _CELL_KEYS or value.get("schema") != CELL_SCHEMA
            or value.get("label") != label or value.get("kind") != kind or value.get("execution") != execution
            or value.get("batch_size") != batch_size or value.get("units") != PUBLIC_SPEC.units
            or value.get("dtype") != PUBLIC_SPEC.dtype or value.get("warmup_steps") != PUBLIC_SPEC.warmup_steps
            or value.get("measured_steps") != PUBLIC_SPEC.measured_steps or value.get("typed_normalized_t4") is not True
            or value.get("initial_model_state_sha256") != reference_model_sha256
            or value.get("initial_optimizer_state_sha256") != reference_optimizer_sha256
            or value.get("fresh_model_from_reference") is not True or value.get("fresh_optimizer_state") is not True
            or value.get("reused_warmed_state") is not False):
        raise FailClosedError("benchmark cell identity/freshness drift")
    _require_sha(value.get("state_before_measure_sha256"), "benchmark pre-measure model state")
    _require_sha(value.get("optimizer_before_measure_sha256"), "benchmark pre-measure optimizer state")
    _validate_runtime(value.get("runtime"))
    status, is_oom = value.get("status"), value.get("cuda_oom")
    if status not in {"MEASURED", "CUDA_OOM", COMPILE_UNAVAILABLE} or is_oom is not (status == "CUDA_OOM"):
        raise FailClosedError("benchmark cell OOM status drift")
    oom_sha = value.get("oom_error_sha256")
    compile_exception_class = value.get("compile_exception_class")
    compile_exception_repr_sha = value.get("compile_exception_repr_sha256")
    compile_failure_stage = value.get("compile_failure_stage")
    post_error_cleanup = value.get("post_error_cuda_cleanup_reset")
    if status == "CUDA_OOM":
        _require_sha(oom_sha, "CUDA OOM error")
        if (compile_exception_class is not None or compile_exception_repr_sha is not None
                or compile_failure_stage is not None or post_error_cleanup is not False):
            raise FailClosedError("CUDA OOM cell has fabricated compile-unavailable evidence")
        if value.get("timing") is not None or value.get("finite") is not None or value.get("post_measurement_finite") is not None:
            raise FailClosedError("CUDA OOM cell has fabricated measurement evidence")
        if value.get("equivalence") is not None:
            raise FailClosedError("CUDA OOM cell has fabricated equivalence evidence")
        resources = value.get("resources")
        if not isinstance(resources, Mapping) or set(resources) != {
            "peak_allocated_bytes", "peak_reserved_bytes", "rss_bytes", "oom_cleanup_reset",
        }:
            raise FailClosedError("CUDA OOM resource schema drift")
        if resources.get("oom_cleanup_reset") is not True:
            raise FailClosedError("CUDA OOM cleanup/reset proof drift")
        for key in ("peak_allocated_bytes", "peak_reserved_bytes", "rss_bytes"):
            if type(resources.get(key)) is not int or resources[key] < 0:
                raise FailClosedError("CUDA OOM resource value drift")
        return dict(value)
    if status == COMPILE_UNAVAILABLE:
        if execution != "compiled" or kind != "mathematical" or oom_sha is not None:
            raise FailClosedError("compile-unavailable cell scope/OOM drift")
        if (not isinstance(compile_exception_class, str) or not compile_exception_class
                or compile_exception_class != compile_exception_class.strip() or "." not in compile_exception_class):
            raise FailClosedError("compile-unavailable exception-class drift")
        _require_sha(compile_exception_repr_sha, "compile-unavailable exception repr")
        if compile_failure_stage not in COMPILE_FAILURE_STAGES:
            raise FailClosedError("compile-unavailable failure-stage drift")
        if post_error_cleanup is not True:
            raise FailClosedError("compile-unavailable CUDA cleanup/reset proof drift")
        if (value.get("timing") is not None or value.get("finite") is not None
                or value.get("post_measurement_finite") is not None or value.get("equivalence") is not None):
            raise FailClosedError("compile-unavailable cell has fabricated measurement evidence")
        resources = value.get("resources")
        if not isinstance(resources, Mapping) or set(resources) != {
            "peak_allocated_bytes", "peak_reserved_bytes", "rss_bytes", "oom_cleanup_reset",
        }:
            raise FailClosedError("compile-unavailable resource schema drift")
        if resources.get("oom_cleanup_reset") is not False:
            raise FailClosedError("compile-unavailable cell falsely reports OOM cleanup")
        for key in ("peak_allocated_bytes", "peak_reserved_bytes", "rss_bytes"):
            if type(resources.get(key)) is not int or resources[key] < 0:
                raise FailClosedError("compile-unavailable resource value drift")
        return dict(value)
    if oom_sha is not None:
        raise FailClosedError("measured benchmark cell has OOM error SHA")
    if (compile_exception_class is not None or compile_exception_repr_sha is not None
            or compile_failure_stage is not None or post_error_cleanup is not False):
        raise FailClosedError("measured benchmark cell has fabricated compile-unavailable evidence")
    finite = value.get("finite")
    if not isinstance(finite, Mapping) or finite != {"forward": True, "loss": True, "gradient": True}:
        raise FailClosedError("benchmark forward/loss/gradient finite drift")
    post = value.get("post_measurement_finite")
    if not isinstance(post, Mapping) or post != {"model": True, "optimizer": True}:
        raise FailClosedError("benchmark post-measure finite drift")
    timing = value.get("timing")
    expected_timing = {
        "warmup_wall_seconds", "measured_total_wall_seconds", "median_step_wall_seconds",
        "steps_per_second", "samples_per_second", "projected_48epoch_seconds", "projection_label",
    }
    if not isinstance(timing, Mapping) or set(timing) != expected_timing:
        raise FailClosedError("benchmark timing schema drift")
    for key in expected_timing - {"projection_label"}:
        _finite(timing.get(key), positive=True)
    if timing.get("projection_label") != ENGINEERING_PROJECTION_LABEL:
        raise FailClosedError("benchmark projection label drift")
    expected_projection = TRAINING_STEPS_48_EPOCHS / float(timing["steps_per_second"])
    if not math.isclose(float(timing["projected_48epoch_seconds"]), expected_projection, rel_tol=0.0, abs_tol=1e-9):
        raise FailClosedError("benchmark projected engineering wall drift")
    resources = value.get("resources")
    if not isinstance(resources, Mapping) or set(resources) != {"peak_allocated_bytes", "peak_reserved_bytes", "rss_bytes", "oom_cleanup_reset"}:
        raise FailClosedError("benchmark resource schema drift")
    if resources.get("oom_cleanup_reset") is not False:
        raise FailClosedError("measured cell falsely reports OOM cleanup")
    for key in ("peak_allocated_bytes", "peak_reserved_bytes", "rss_bytes"):
        if type(resources.get(key)) is not int or resources[key] < 0:
            raise FailClosedError("benchmark resource value drift")
    equivalence = value.get("equivalence")
    if kind == "mathematical" and execution == "eager":
        _validate_equivalence(
            equivalence, require_mathematical_timing_gate=True, require_gru_map=False,
        )
    elif kind == "validation_hoisted":
        _validate_equivalence(
            equivalence, require_mathematical_timing_gate=False, require_gru_map=False,
        )
    elif kind == "grucell_equivalent":
        _validate_equivalence(
            equivalence, require_mathematical_timing_gate=False, require_gru_map=True,
        )
    elif equivalence is not None:
        raise FailClosedError("baseline/compiled cell must not fabricate unrelated equivalence")
    return dict(value)


class MatrixBackend(Protocol):
    """Injected no-data backend for matrix scheduling and CPU adversarial tests."""

    @property
    def reference_model_state_sha256(self) -> str: ...

    @property
    def reference_optimizer_state_sha256(self) -> str: ...

    def compare_before_timing(self, *, kind: str, batch_size: int) -> Mapping[str, Any]: ...

    def measure(
        self,
        *,
        label: str,
        kind: str,
        execution: str,
        batch_size: int,
        equivalence: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]: ...

    def cleanup_after_oom(self, *, label: str, batch_size: int) -> None: ...


def _measure_cell(
    backend: MatrixBackend,
    *,
    label: str,
    kind: str,
    execution: str,
    batch_size: int,
    equivalence: Mapping[str, Any] | None,
) -> dict[str, object]:
    value = dict(backend.measure(
        label=label, kind=kind, execution=execution, batch_size=batch_size, equivalence=equivalence,
    ))
    validated = validate_cell_payload(
        value, label=label, kind=kind, execution=execution, batch_size=batch_size,
        reference_model_sha256=backend.reference_model_state_sha256,
        reference_optimizer_sha256=backend.reference_optimizer_state_sha256,
    )
    if validated["status"] == "CUDA_OOM":
        backend.cleanup_after_oom(label=label, batch_size=batch_size)
    return validated


def run_benchmark_matrix(backend: MatrixBackend) -> dict[str, object]:
    """Schedule the frozen ordered engineering matrix without any fallback.

    The backend receives one call per actual attempt.  In particular, an OOM
    never causes a hidden retry or an implicit smaller batch; only the next
    *predeclared* matrix action may proceed.
    """
    _require_sha(backend.reference_model_state_sha256, "reference model state")
    _require_sha(backend.reference_optimizer_state_sha256, "reference optimizer state")
    cells: list[dict[str, object]] = []

    production32 = _measure_cell(
        backend, label="production_contract_eager_b32", kind="production_contract", execution="eager",
        batch_size=32, equivalence=None,
    )
    cells.append(production32)
    if production32["status"] == "CUDA_OOM":
        # There is no viable base state against which mathematical equality
        # can be proven.  Keeping the root empty is more honest than emitting
        # an incomplete engineering comparison.
        raise FailClosedError("batch-32 production-contract baseline OOM; matrix cannot establish equivalence")

    mathematical_equivalence = backend.compare_before_timing(kind="mathematical", batch_size=32)
    # This is the only comparison that can open a timing interval for the
    # deferred-receipt graph.  Bitwise exactness is preserved in the receipt,
    # but the frozen FP32 contract is the gate because independent CUDA
    # backward executions have a reviewed reduction-order band.
    _validate_equivalence(
        mathematical_equivalence, require_mathematical_timing_gate=True, require_gru_map=False,
    )
    cells.append(_measure_cell(
        backend, label="mathematical_eager_b32", kind="mathematical", execution="eager",
        batch_size=32, equivalence=mathematical_equivalence,
    ))

    hoisted_equivalence = backend.compare_before_timing(kind="validation_hoisted", batch_size=32)
    _validate_equivalence(
        hoisted_equivalence, require_mathematical_timing_gate=False, require_gru_map=False,
    )
    cells.append(_measure_cell(
        backend, label="validation_hoisted_eager_b32", kind="validation_hoisted", execution="eager",
        batch_size=32, equivalence=hoisted_equivalence,
    ))

    eager_successes = [production32]
    stopped_after_oom: int | None = None
    for batch_size in (64, 128):
        cell = _measure_cell(
            backend, label=f"production_contract_eager_b{batch_size}", kind="production_contract", execution="eager",
            batch_size=batch_size, equivalence=None,
        )
        cells.append(cell)
        if cell["status"] == "CUDA_OOM":
            stopped_after_oom = batch_size
            break
        eager_successes.append(cell)
    fastest = max(
        eager_successes,
        key=lambda item: (float(item["timing"]["samples_per_second"]), int(item["batch_size"])),
    )
    fastest_batch = int(fastest["batch_size"])

    compiled32 = _measure_cell(
        backend, label="mathematical_compile_b32", kind="mathematical", execution="compiled",
        batch_size=32, equivalence=None,
    )
    cells.append(compiled32)
    if compiled32["status"] == "CUDA_OOM":
        # No second compiled attempt is admissible after the declared batch-32
        # OOM — including the degenerate case where the fastest eager batch is
        # also 32.  Retrying that same condition would turn a recorded OOM
        # into an unreported retry rather than an engineering observation.
        compiled_fastest: dict[str, object] | None = None
        compiled_fastest_skipped = "compiled_batch32_oom"
    elif compiled32["status"] == COMPILE_UNAVAILABLE:
        # The installed compiler implementation is optional.  Record its
        # exact bounded failure once at batch 32, never retry it at another
        # batch, and continue with the independent GRUCell diagnostic.
        compiled_fastest = None
        compiled_fastest_skipped = "compiled_batch32_implementation_unavailable"
    else:
        compiled_fastest = _measure_cell(
            backend, label=f"mathematical_compile_fastest_eager_b{fastest_batch}", kind="mathematical",
            execution="compiled", batch_size=fastest_batch, equivalence=None,
        )
        cells.append(compiled_fastest)
        compiled_fastest_skipped = None

    gru_equivalence = backend.compare_before_timing(kind="grucell_equivalent", batch_size=32)
    _validate_equivalence(
        gru_equivalence, require_mathematical_timing_gate=False, require_gru_map=True,
    )
    cells.append(_measure_cell(
        backend, label="grucell_equivalent_eager_b32", kind="grucell_equivalent", execution="candidate",
        batch_size=32, equivalence=gru_equivalence,
    ))
    return {
        "cells": cells,
        "fastest_non_oom_eager_batch": fastest_batch,
        "eager_larger_batch_stopped_after_oom": stopped_after_oom,
        "compiled_fastest_eager_skipped": compiled_fastest_skipped,
        "compiled_fastest_eager_attempted": compiled_fastest is not None,
    }


def _query_gpu0_from_nvidia_smi() -> Mapping[str, object]:
    """Query physical GPU0 directly; no visible-device remapping is trusted."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi", "--id=0", "--query-gpu=uuid,pci.bus_id,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FailClosedError("cannot query physical GPU0 through nvidia-smi") from error
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise FailClosedError("physical GPU0 query cardinality drift")
    fields = [item.strip() for item in rows[0].split(",")]
    if len(fields) != 4:
        raise FailClosedError("physical GPU0 query schema drift")
    try:
        memory = int(fields[3])
    except ValueError as error:
        raise FailClosedError("physical GPU0 memory field drift") from error
    # This is an NVML/nvidia-smi nominal-capacity authority only.  Do not
    # compare or convert it against Torch's device-property byte authority.
    return {
        "uuid": fields[0], "bdf": fields[1], "name": fields[2],
        "nvidia_smi_memory_total_mib": memory,
    }


def require_exact_gpu0(torch: Any, *, query_gpu0: Callable[[], Mapping[str, object]] = _query_gpu0_from_nvidia_smi) -> dict[str, object]:
    """Accept exactly physical GPU0 exposed as the sole logical CUDA device."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        # Deliberately before a torch.cuda call: no accidental fallback or
        # device discovery is permissible under a different visibility mask.
        raise FailClosedError("throughput benchmark requires CUDA_VISIBLE_DEVICES=0 exactly")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise FailClosedError("throughput benchmark requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    if device.type != "cuda" or device.index != 0:
        raise FailClosedError("throughput benchmark logical device drift")
    external = dict(query_gpu0())
    expected_nvidia_smi = {
        "uuid": FROZEN_GPU0["uuid"], "bdf": FROZEN_GPU0["bdf"],
        "name": FROZEN_GPU0["name"],
        "nvidia_smi_memory_total_mib": FROZEN_GPU0["nvidia_smi_memory_total_mib"],
    }
    if external != expected_nvidia_smi:
        raise FailClosedError("throughput benchmark nvidia-smi GPU0 identity drift")
    props = torch.cuda.get_device_properties(0)
    if getattr(props, "name", None) != FROZEN_GPU0["name"]:
        raise FailClosedError("throughput benchmark Torch GPU0 name drift")
    torch_total_memory = getattr(props, "total_memory", None)
    if type(torch_total_memory) is not int or torch_total_memory != FROZEN_GPU0["torch_total_memory_bytes"]:
        raise FailClosedError("throughput benchmark Torch GPU0 total-memory authority drift")
    cudnn = torch.backends.cudnn.version()
    return {
        "gpu": dict(FROZEN_GPU0), "torch_version": str(torch.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn_version": None if cudnn is None else str(cudnn),
        "cpu_threads": {"intraop": 1, "interop": 1},
    }


@dataclass
class _VariantRuntime:
    model: Any
    executable: Any
    optimizer: Any
    batch: SyntheticBatch
    candidate: bool
    initial_model_state_sha256: str
    initial_optimizer_state_sha256: str


class PhysicalBenchmarkBackend:
    """Lazy, no-data CUDA backend reachable only after independent review."""

    def __init__(self, root: Path) -> None:
        self.root = root.absolute()
        self.torch: Any | None = None
        self.model_module: Any | None = None
        self.device: Any | None = None
        self.runtime: Mapping[str, object] | None = None
        self.snapshot: Mapping[str, Any] | None = None
        self._reference_model_state_sha256: str | None = None
        self._reference_optimizer_state_sha256: str | None = None
        self._oom_cleanup_calls: list[tuple[str, int]] = []

    @property
    def reference_model_state_sha256(self) -> str:
        if self._reference_model_state_sha256 is None:
            raise FailClosedError("physical benchmark backend has not prepared reference model")
        return self._reference_model_state_sha256

    @property
    def reference_optimizer_state_sha256(self) -> str:
        if self._reference_optimizer_state_sha256 is None:
            raise FailClosedError("physical benchmark backend has not prepared reference optimizer")
        return self._reference_optimizer_state_sha256

    def prepare(self) -> None:
        """Initialize GPU0 only after closure/baseline/fresh-output gates passed."""
        if self.torch is not None:
            raise FailClosedError("physical benchmark backend prepared twice")
        import torch
        from . import model as model_module
        # Exact one-thread setup is part of the measured engineering contract.
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            raise FailClosedError("cannot establish one-thread benchmark interop pool") from error
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise FailClosedError("benchmark CPU thread-pool setting drift")
        runtime = require_exact_gpu0(torch)
        _runtime_seed(torch, PUBLIC_SPEC.seed)
        device = torch.device("cuda:0")
        template = model_module.TFSRDecoder(capture_diagnostics=False).to(device)
        snapshot = _snapshot_cpu_state(torch, template)
        reference_model = model_state_digest(torch, template)
        optimizer = _new_frozen_adam(torch, template)
        reference_optimizer = optimizer_state_digest(torch, optimizer)
        self.torch, self.model_module, self.device, self.runtime = torch, model_module, device, runtime
        self.snapshot = snapshot
        self._reference_model_state_sha256 = reference_model
        self._reference_optimizer_state_sha256 = reference_optimizer
        # The reference instance is a setup-only seed source.  Retaining it
        # (or every completed variant) would inflate later peak-memory cells
        # and could create a false OOM unrelated to the fresh variant under
        # test.  The immutable CPU snapshot/digests are the only lineage
        # state deliberately retained across cells.
        del template, optimizer
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

    def _require_prepared(self) -> tuple[Any, Any, Any, Mapping[str, Any], Mapping[str, Any]]:
        if (self.torch is None or self.model_module is None or self.device is None or self.runtime is None
                or self.snapshot is None):
            raise FailClosedError("physical benchmark backend not prepared")
        return self.torch, self.model_module, self.device, self.runtime, self.snapshot

    def _fresh_variant(self, *, batch_size: int, execution: str, candidate: bool) -> _VariantRuntime:
        torch, model_module, device, _runtime, snapshot = self._require_prepared()
        if candidate and (execution != "candidate" or batch_size != 32):
            raise FailClosedError("GRUCell candidate is frozen to batch 32 only")
        if not candidate and execution not in {"eager", "compiled"}:
            raise FailClosedError("baseline variant execution label drift")
        base = _fresh_model_from_snapshot(torch, model_module, snapshot, device)
        model = build_grucell_candidate(torch, base) if candidate else base
        initial_model = model_state_digest(torch, model, candidate=candidate)
        if initial_model != self.reference_model_state_sha256:
            raise FailClosedError("benchmark variant did not start from exact reference model state")
        optimizer = _new_frozen_adam(torch, model)
        initial_optimizer = optimizer_state_digest(torch, optimizer)
        if initial_optimizer != self.reference_optimizer_state_sha256:
            raise FailClosedError("benchmark variant did not start from fresh exact Adam state")
        executable = torch.compile(model) if execution == "compiled" else model
        batch = make_synthetic_batch(torch=torch, model_module=model_module, batch_size=batch_size, device=device)
        return _VariantRuntime(
            model=model, executable=executable, optimizer=optimizer, batch=batch, candidate=candidate,
            initial_model_state_sha256=initial_model, initial_optimizer_state_sha256=initial_optimizer,
        )

    def _one_step(self, variant: _VariantRuntime, kind: str, *, capture_gradients: bool = False) -> Mapping[str, Any]:
        torch, model_module, _device, _runtime, _snapshot = self._require_prepared()
        if kind == "production_contract":
            return _production_contract_step(
                torch, model_module, variant.executable, variant.optimizer, variant.batch,
                capture_gradients=capture_gradients,
            )
        if kind == "mathematical":
            return _mathematical_step(
                torch, model_module, variant.executable, variant.optimizer, variant.batch,
                capture_gradients=capture_gradients,
            )
        if kind == "validation_hoisted":
            return _validation_hoisted_step(
                torch, model_module, variant.executable, variant.optimizer, variant.batch,
                capture_gradients=capture_gradients,
            )
        if kind == "grucell_equivalent":
            return _production_contract_step(
                torch, model_module, variant.executable, variant.optimizer, variant.batch,
                capture_gradients=capture_gradients,
            )
        raise FailClosedError("unknown benchmark step kind")

    def _comparison_variant_pair(self, *, kind: str, batch_size: int) -> tuple[_VariantRuntime, _VariantRuntime]:
        if kind not in {"mathematical", "validation_hoisted", "grucell_equivalent"}:
            raise FailClosedError("unsupported benchmark comparison kind")
        baseline = self._fresh_variant(batch_size=batch_size, execution="eager", candidate=False)
        candidate = self._fresh_variant(
            batch_size=batch_size, execution="candidate" if kind == "grucell_equivalent" else "eager",
            candidate=(kind == "grucell_equivalent"),
        )
        if kind == "validation_hoisted":
            torch, model_module, _device, _runtime, _snapshot = self._require_prepared()
            _validate_hoisted_inputs(torch, model_module, candidate.model, candidate.batch)
        return baseline, candidate

    def compare_before_timing(self, *, kind: str, batch_size: int) -> Mapping[str, Any]:
        """Run one untimed paired step to make equivalence visible first."""
        torch, _model_module, _device, _runtime, _snapshot = self._require_prepared()
        baseline, candidate = self._comparison_variant_pair(kind=kind, batch_size=batch_size)
        _runtime_seed(torch, PUBLIC_SPEC.seed)
        baseline_observation = _production_contract_step(
            torch, self.model_module, baseline.executable, baseline.optimizer, baseline.batch, capture_gradients=True,
        )
        _runtime_seed(torch, PUBLIC_SPEC.seed)
        candidate_observation = self._one_step(candidate, kind, capture_gradients=True)
        return _comparison_report(
            torch,
            baseline_model=baseline.model,
            candidate_model=candidate.model,
            baseline_optimizer=baseline.optimizer,
            candidate_optimizer=candidate.optimizer,
            baseline_observation=baseline_observation,
            candidate_observation=candidate_observation,
            candidate_is_grucell=(kind == "grucell_equivalent"),
            comparison_role=(
                MATHEMATICAL_TIMING_GATE if kind == "mathematical" else DIAGNOSTIC_NON_AUTHORIZING
            ),
            label=f"production_contract_vs_{kind}_b{batch_size}",
        )

    def _resources(self, *, oom_cleanup_reset: bool) -> dict[str, object]:
        torch, _model_module, _device, _runtime, _snapshot = self._require_prepared()
        return {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "oom_cleanup_reset": oom_cleanup_reset,
        }

    def _clean_cuda_after_recorded_error(self) -> None:
        """Release allocator state before emitting an allowed error record."""
        torch, _model_module, _device, _runtime, _snapshot = self._require_prepared()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

    def _clean_cuda_oom(self) -> None:
        self._clean_cuda_after_recorded_error()

    def measure(
        self,
        *,
        label: str,
        kind: str,
        execution: str,
        batch_size: int,
        equivalence: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        """Time exactly one fresh variant, never a previous warm model/Adam."""
        torch, model_module, _device, runtime, _snapshot = self._require_prepared()
        candidate = kind == "grucell_equivalent"
        variant: _VariantRuntime | None = None
        failure_stage = "variant_construction"
        try:
            variant = self._fresh_variant(batch_size=batch_size, execution=execution, candidate=candidate)
            if kind == "validation_hoisted":
                _validate_hoisted_inputs(torch, model_module, variant.model, variant.batch)
            failure_stage = "warmup"
            _runtime_seed(torch, PUBLIC_SPEC.seed)
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize(0)
            warmup_start = time.perf_counter()
            for _ in range(PUBLIC_SPEC.warmup_steps):
                observation = self._one_step(variant, kind)
                _materialize_step_observation(torch, observation)
            torch.cuda.synchronize(0)
            warmup_seconds = max(time.perf_counter() - warmup_start, 1e-12)
            failure_stage = "measurement"
            state_before_measure = model_state_digest(torch, variant.model, candidate=candidate)
            optimizer_before_measure = optimizer_state_digest(torch, variant.optimizer)
            durations: list[float] = []
            finite = {"forward": True, "loss": True, "gradient": True}
            for _ in range(PUBLIC_SPEC.measured_steps):
                torch.cuda.synchronize(0)
                start = time.perf_counter()
                observation = self._one_step(variant, kind)
                torch.cuda.synchronize(0)
                durations.append(max(time.perf_counter() - start, 1e-12))
                # Required receipt conversions occur outside the timed interval
                # for mathematical/hoisted cells; production already paid them.
                materialized = _materialize_step_observation(torch, observation)
                finite = {
                    key: bool(finite[key] and materialized["finite"][key] is not False)
                    for key in finite
                }
            failure_stage = "post_measurement_audit"
            # This proof describes the exact end state of the timed region.
            # The separate gradient audit below advances a fresh step solely
            # to obtain an unambiguous gradient-finiteness observation; it
            # must not silently redefine the measured end state.
            post = {"model": _all_finite_parameters(torch, variant.model),
                    "optimizer": _all_finite_optimizer(torch, variant.optimizer)}
            # Gradient finiteness is audited outside the timing interval on an
            # additional explicit step.  Timed mathematical/hoisted cells are
            # thereby not contaminated by a full gradient scan that the live
            # ordinary production step never performs.
            audit = self._one_step(variant, kind, capture_gradients=True)
            audited = _materialize_step_observation(torch, audit)
            finite = {
                "forward": bool(finite["forward"] and audited["finite"]["forward"]),
                "loss": bool(finite["loss"] and audited["finite"]["loss"]),
                "gradient": audited["finite"]["gradient"] is True,
            }
            total = sum(durations)
            median = sorted(durations)[len(durations) // 2] if len(durations) % 2 else (
                sorted(durations)[len(durations) // 2 - 1] + sorted(durations)[len(durations) // 2]
            ) / 2.0
            steps_per_second = PUBLIC_SPEC.measured_steps / total
            if not all(finite.values()) or not all(post.values()):
                raise FailClosedError("benchmark measured nonfinite state")
            return {
                "schema": CELL_SCHEMA, "label": label, "kind": kind, "execution": execution,
                "batch_size": batch_size, "units": PUBLIC_SPEC.units, "dtype": PUBLIC_SPEC.dtype,
                "warmup_steps": PUBLIC_SPEC.warmup_steps, "measured_steps": PUBLIC_SPEC.measured_steps,
                "status": "MEASURED", "cuda_oom": False, "oom_error_sha256": None,
                "compile_exception_class": None, "compile_exception_repr_sha256": None,
                "compile_failure_stage": None, "post_error_cuda_cleanup_reset": False,
                "typed_normalized_t4": True,
                "initial_model_state_sha256": variant.initial_model_state_sha256,
                "initial_optimizer_state_sha256": variant.initial_optimizer_state_sha256,
                "state_before_measure_sha256": state_before_measure,
                "optimizer_before_measure_sha256": optimizer_before_measure,
                "fresh_model_from_reference": True, "fresh_optimizer_state": True, "reused_warmed_state": False,
                "finite": finite, "post_measurement_finite": post,
                "timing": {
                    "warmup_wall_seconds": warmup_seconds, "measured_total_wall_seconds": total,
                    "median_step_wall_seconds": median, "steps_per_second": steps_per_second,
                    "samples_per_second": steps_per_second * batch_size,
                    "projected_48epoch_seconds": TRAINING_STEPS_48_EPOCHS / steps_per_second,
                    "projection_label": ENGINEERING_PROJECTION_LABEL,
                },
                "resources": self._resources(oom_cleanup_reset=False),
                "runtime": dict(runtime), "equivalence": None if equivalence is None else dict(equivalence),
            }
        except torch.OutOfMemoryError as error:
            self._clean_cuda_oom()
            return {
                "schema": CELL_SCHEMA, "label": label, "kind": kind, "execution": execution,
                "batch_size": batch_size, "units": PUBLIC_SPEC.units, "dtype": PUBLIC_SPEC.dtype,
                "warmup_steps": PUBLIC_SPEC.warmup_steps, "measured_steps": PUBLIC_SPEC.measured_steps,
                "status": "CUDA_OOM", "cuda_oom": True, "oom_error_sha256": _sha(repr(error).encode("utf-8")),
                "compile_exception_class": None, "compile_exception_repr_sha256": None,
                "compile_failure_stage": None, "post_error_cuda_cleanup_reset": False,
                "typed_normalized_t4": True,
                "initial_model_state_sha256": self.reference_model_state_sha256 if variant is None else variant.initial_model_state_sha256,
                "initial_optimizer_state_sha256": self.reference_optimizer_state_sha256 if variant is None else variant.initial_optimizer_state_sha256,
                "state_before_measure_sha256": self.reference_model_state_sha256 if variant is None else model_state_digest(torch, variant.model, candidate=candidate),
                "optimizer_before_measure_sha256": self.reference_optimizer_state_sha256 if variant is None else optimizer_state_digest(torch, variant.optimizer),
                "fresh_model_from_reference": True, "fresh_optimizer_state": True, "reused_warmed_state": False,
                "finite": None, "post_measurement_finite": None, "timing": None,
                "resources": self._resources(oom_cleanup_reset=True), "runtime": dict(runtime), "equivalence": None,
            }
        except FailClosedError:
            # Provenance, numerical, and receipt-contract violations are not
            # compiler availability observations.  They must keep the route
            # fail-closed even when the attempted executable was compiled.
            raise
        except Exception as error:
            # A compiler/import/runtime fault is an admissible engineering
            # observation only for an explicitly compiled cell.  The same
            # exception from every eager or candidate graph must escape and
            # keep the canonical output root empty.
            if execution != "compiled":
                raise
            self._clean_cuda_after_recorded_error()
            return {
                "schema": CELL_SCHEMA, "label": label, "kind": kind, "execution": execution,
                "batch_size": batch_size, "units": PUBLIC_SPEC.units, "dtype": PUBLIC_SPEC.dtype,
                "warmup_steps": PUBLIC_SPEC.warmup_steps, "measured_steps": PUBLIC_SPEC.measured_steps,
                "status": COMPILE_UNAVAILABLE, "cuda_oom": False, "oom_error_sha256": None,
                "compile_exception_class": _exception_class_name(error),
                "compile_exception_repr_sha256": _sha(repr(error).encode("utf-8")),
                "compile_failure_stage": failure_stage, "post_error_cuda_cleanup_reset": True,
                "typed_normalized_t4": True,
                "initial_model_state_sha256": self.reference_model_state_sha256 if variant is None else variant.initial_model_state_sha256,
                "initial_optimizer_state_sha256": self.reference_optimizer_state_sha256 if variant is None else variant.initial_optimizer_state_sha256,
                "state_before_measure_sha256": self.reference_model_state_sha256 if variant is None else model_state_digest(torch, variant.model, candidate=candidate),
                "optimizer_before_measure_sha256": self.reference_optimizer_state_sha256 if variant is None else optimizer_state_digest(torch, variant.optimizer),
                "fresh_model_from_reference": True, "fresh_optimizer_state": True, "reused_warmed_state": False,
                "finite": None, "post_measurement_finite": None, "timing": None,
                "resources": self._resources(oom_cleanup_reset=False), "runtime": dict(runtime), "equivalence": None,
            }

    def cleanup_after_oom(self, *, label: str, batch_size: int) -> None:
        # Core calls this once per OOM cell; no retry and no alternative batch
        # are constructed here.  Reset again as an explicit postcondition.
        self._oom_cleanup_calls.append((label, batch_size))
        self._clean_cuda_oom()


def _validate_matrix_payload(value: object) -> dict[str, object]:
    """Reconstruct the one legal attempted matrix from recorded statuses.

    This is deliberately stricter than checking membership of a few labels.
    The OOM stop rules change which later labels are legal, so a forged
    receipt cannot claim a stop while retaining a forbidden retry/large-batch
    measurement elsewhere in the list.
    """
    expected = {
        "cells", "fastest_non_oom_eager_batch", "eager_larger_batch_stopped_after_oom",
        "compiled_fastest_eager_skipped", "compiled_fastest_eager_attempted",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FailClosedError("benchmark matrix schema drift")
    cells = value.get("cells")
    if not isinstance(cells, list) or len(cells) < 6:
        raise FailClosedError("benchmark matrix has too few attempted cells")
    if not isinstance(cells[0], Mapping):
        raise FailClosedError("benchmark first-cell schema drift")
    reference_model = cells[0].get("initial_model_state_sha256")
    reference_optimizer = cells[0].get("initial_optimizer_state_sha256")
    _require_sha(reference_model, "matrix reference model state")
    _require_sha(reference_optimizer, "matrix reference optimizer state")

    cursor = 0
    checked_cells: list[dict[str, object]] = []

    def take(label: str, kind: str, execution: str, batch_size: int) -> dict[str, object]:
        nonlocal cursor
        if cursor >= len(cells):
            raise FailClosedError("benchmark attempted matrix is truncated")
        item = cells[cursor]
        if not isinstance(item, Mapping) or item.get("label") != label:
            raise FailClosedError("benchmark attempted matrix order drift")
        cursor += 1
        checked = validate_cell_payload(
            item, label=label, kind=kind, execution=execution, batch_size=batch_size,
            reference_model_sha256=str(reference_model), reference_optimizer_sha256=str(reference_optimizer),
        )
        checked_cells.append(checked)
        return checked

    production32 = take("production_contract_eager_b32", "production_contract", "eager", 32)
    if production32["status"] != "MEASURED":
        # The scheduler itself refuses this condition because no exact
        # mathematical baseline could then be established.
        raise FailClosedError("benchmark production batch-32 baseline must be measured")
    take("mathematical_eager_b32", "mathematical", "eager", 32)
    take("validation_hoisted_eager_b32", "validation_hoisted", "eager", 32)

    eager_successes = [production32]
    production64 = take("production_contract_eager_b64", "production_contract", "eager", 64)
    if production64["status"] == "CUDA_OOM":
        expected_eager_stop: int | None = 64
    else:
        eager_successes.append(production64)
        production128 = take("production_contract_eager_b128", "production_contract", "eager", 128)
        if production128["status"] == "CUDA_OOM":
            expected_eager_stop = 128
        else:
            eager_successes.append(production128)
            expected_eager_stop = None

    fastest = value.get("fastest_non_oom_eager_batch")
    if type(fastest) is not int or fastest not in {32, 64, 128}:
        raise FailClosedError("benchmark fastest eager batch type drift")
    expected_fastest = max(eager_successes, key=lambda item: (float(item["timing"]["samples_per_second"]), int(item["batch_size"])))
    if fastest != expected_fastest["batch_size"]:
        raise FailClosedError("benchmark fastest eager selection drift")
    if value.get("eager_larger_batch_stopped_after_oom") != expected_eager_stop:
        raise FailClosedError("benchmark eager OOM stop marker/status drift")

    compiled32 = take("mathematical_compile_b32", "mathematical", "compiled", 32)
    compiled_skipped = value.get("compiled_fastest_eager_skipped")
    if compiled32["status"] == "CUDA_OOM":
        if compiled_skipped != "compiled_batch32_oom" or value.get("compiled_fastest_eager_attempted") is not False:
            raise FailClosedError("benchmark compiled-fastest OOM marker/status drift")
    elif compiled32["status"] == COMPILE_UNAVAILABLE:
        if (compiled_skipped != "compiled_batch32_implementation_unavailable"
                or value.get("compiled_fastest_eager_attempted") is not False):
            raise FailClosedError("benchmark compiled-fastest implementation-unavailable marker/status drift")
    else:
        if compiled_skipped is not None or value.get("compiled_fastest_eager_attempted") is not True:
            raise FailClosedError("benchmark compiled-fastest attempt marker/status drift")
        take(f"mathematical_compile_fastest_eager_b{fastest}", "mathematical", "compiled", fastest)

    take("grucell_equivalent_eager_b32", "grucell_equivalent", "candidate", 32)
    if cursor != len(cells):
        raise FailClosedError("benchmark attempted matrix has forbidden extra/reordered cells")
    return {
        "cells": checked_cells, "fastest_non_oom_eager_batch": fastest,
        "eager_larger_batch_stopped_after_oom": expected_eager_stop,
        "compiled_fastest_eager_skipped": compiled_skipped,
        "compiled_fastest_eager_attempted": value["compiled_fastest_eager_attempted"],
    }


def build_engineering_receipt(
    *,
    baseline: Mapping[str, object],
    matrix: Mapping[str, object],
    launch_closure: Mapping[str, object],
    final_closure: Mapping[str, object],
    runtime: Mapping[str, object],
) -> dict[str, object]:
    """Build, but do not publish, a non-scientific engineering receipt."""
    launch = validate_closure_payload(launch_closure)
    final = validate_closure_payload(final_closure)
    checked_matrix = _validate_matrix_payload(matrix)
    checked_runtime = _validate_runtime(runtime)
    expected_baseline = {
        "relative_path": LIVE_THROUGHPUT_RELATIVE, "body_sha256": LIVE_THROUGHPUT_SHA256,
        "steps_per_second": FROZEN_BASELINE["steps_per_second"], "elapsed_seconds": FROZEN_BASELINE["elapsed_seconds"],
        "projected_48epoch_seconds": FROZEN_BASELINE["projected_48epoch_seconds"],
    }
    if dict(baseline) != expected_baseline:
        raise FailClosedError("engineering receipt fixed baseline evidence drift")
    return {
        "schema": "tfsr_b3st4_ddrop_throughput_engineering_v1",
        "status": "ENGINEERING_BENCHMARK_COMPLETE",
        "cell": CELL,
        "phase": PHASE,
        "purpose": "ENGINEERING_ONLY",
        "scientific_result": False,
        "data_opened": False,
        "checkpoint_opened": False,
        "target_or_formal": False,
        "authorizes_training": False,
        "benchmark_spec": PUBLIC_SPEC.payload(),
        "fixed_evidence": {
            "workorder": {"relative_path": WORKORDER_RELATIVE, "sha256": WORKORDER_SHA256},
            "model": {"relative_path": MODEL_RELATIVE, "sha256": MODEL_SHA256},
            "native_causal_activity": {
                "relative_path": NATIVE_ACTIVITY_RELATIVE,
                "sha256": NATIVE_ACTIVITY_SHA256,
            },
            "live_phase_d_v2_throughput": dict(expected_baseline),
            "frozen_adam": dict(FROZEN_ADAM),
        },
        "runtime": checked_runtime,
        "matrix": checked_matrix,
        "launch_closure": launch,
        "final_closure": final,
        "launch_final_closure_equal": launch == final,
        "canonical_output": {"root_relative": OUTPUT_ROOT_RELATIVE, "topology": list(OUTPUT_TOPOLOGY)},
    }


def validate_engineering_receipt(value: Mapping[str, Any]) -> dict[str, object]:
    expected = {
        "schema", "status", "cell", "phase", "purpose", "scientific_result", "data_opened",
        "checkpoint_opened", "target_or_formal", "authorizes_training", "benchmark_spec", "fixed_evidence",
        "runtime", "matrix", "launch_closure", "final_closure", "launch_final_closure_equal", "canonical_output",
    }
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_b3st4_ddrop_throughput_engineering_v1"
            or value.get("status") != "ENGINEERING_BENCHMARK_COMPLETE" or value.get("cell") != CELL
            or value.get("phase") != PHASE or value.get("purpose") != "ENGINEERING_ONLY"
            or value.get("scientific_result") is not False or value.get("data_opened") is not False
            or value.get("checkpoint_opened") is not False or value.get("target_or_formal") is not False
            or value.get("authorizes_training") is not False or value.get("benchmark_spec") != PUBLIC_SPEC.payload()
            or value.get("canonical_output") != {"root_relative": OUTPUT_ROOT_RELATIVE, "topology": list(OUTPUT_TOPOLOGY)}):
        raise FailClosedError("engineering receipt boundary/schema drift")
    fixed = value.get("fixed_evidence")
    expected_baseline = {
        "relative_path": LIVE_THROUGHPUT_RELATIVE, "body_sha256": LIVE_THROUGHPUT_SHA256,
        "steps_per_second": FROZEN_BASELINE["steps_per_second"], "elapsed_seconds": FROZEN_BASELINE["elapsed_seconds"],
        "projected_48epoch_seconds": FROZEN_BASELINE["projected_48epoch_seconds"],
    }
    if (not isinstance(fixed, Mapping) or set(fixed) != {
                "workorder", "model", "native_causal_activity", "live_phase_d_v2_throughput", "frozen_adam",
            }
            or fixed.get("workorder") != {"relative_path": WORKORDER_RELATIVE, "sha256": WORKORDER_SHA256}
            or fixed.get("model") != {"relative_path": MODEL_RELATIVE, "sha256": MODEL_SHA256}
            or fixed.get("native_causal_activity") != {
                "relative_path": NATIVE_ACTIVITY_RELATIVE,
                "sha256": NATIVE_ACTIVITY_SHA256,
            }
            or fixed.get("live_phase_d_v2_throughput") != expected_baseline
            or fixed.get("frozen_adam") != FROZEN_ADAM):
        raise FailClosedError("engineering receipt fixed evidence drift")
    runtime = _validate_runtime(value.get("runtime"))
    matrix = _validate_matrix_payload(value.get("matrix"))
    launch, final = validate_closure_payload(value.get("launch_closure")), validate_closure_payload(value.get("final_closure"))
    if launch != final or value.get("launch_final_closure_equal") is not True:
        raise FailClosedError("engineering receipt launch/final closure drift")
    return {
        "runtime": runtime, "matrix": matrix, "launch_closure": launch, "final_closure": final,
    }


def publish_after_stable_final_closure(
    *,
    root: Path,
    receipt: Mapping[str, Any],
    launch_closure: Mapping[str, object],
    final_closure: Mapping[str, object],
) -> str:
    """Refuse output reservation if any reviewed source closure has drifted."""
    launch, final = validate_closure_payload(launch_closure), validate_closure_payload(final_closure)
    if launch != final:
        raise FailClosedError("launch/final closure drift prevents benchmark receipt publication")
    validate_engineering_receipt(receipt)
    if receipt.get("launch_closure") != launch or receipt.get("final_closure") != final:
        raise FailClosedError("receipt closure does not bind stable launch/final closure")
    require_canonical_output_fresh(root)
    parent, name = canonical_output_parent(root)
    artifact = reserve_output_root(parent, name)
    return artifact.publish_receipt(receipt)


class RootReviewCapability:
    """Opaque audit token; this work order deliberately exposes no issuer."""

    __slots__ = ("_seal",)

    def __init__(self, seal: object) -> None:
        if seal is not _ROOT_REVIEW_SEAL:
            raise TypeError("root review capability may only be issued after independent audit")
        self._seal = seal


_ROOT_REVIEW_SEAL = object()


def _issue_root_review_capability_for_audited_route() -> RootReviewCapability:
    """Private bridge for a future root-approved invocation, never the CLI."""
    return RootReviewCapability(_ROOT_REVIEW_SEAL)


def _require_root_review_capability(capability: object) -> None:
    if not isinstance(capability, RootReviewCapability) or capability._seal is not _ROOT_REVIEW_SEAL:
        raise FailClosedError("ROOT_AUDIT_REQUIRED_NO_GPU_BENCHMARK")


def run_reviewed_gpu_benchmark(root: Path, *, capability: RootReviewCapability) -> Mapping[str, Any]:
    """The only physical route; not callable from the current public CLI."""
    _require_root_review_capability(capability)
    require_canonical_output_fresh(root)
    launch_closure = compute_benchmark_closure(root)
    baseline = load_live_baseline_evidence(root)
    backend = PhysicalBenchmarkBackend(root)
    backend.prepare()
    matrix = run_benchmark_matrix(backend)
    final_closure = compute_benchmark_closure(root)
    receipt = build_engineering_receipt(
        baseline=baseline, matrix=matrix, launch_closure=launch_closure, final_closure=final_closure,
        runtime=backend.runtime if backend.runtime is not None else {},
    )
    digest = publish_after_stable_final_closure(
        root=root, receipt=receipt, launch_closure=launch_closure, final_closure=final_closure,
    )
    return {"receipt": receipt, "receipt_sha256": digest}


def reject_unreviewed_public_execution() -> None:
    """Current two-flag CLI terminal: root audit has not authorized GPU work."""
    raise FailClosedError("ROOT_AUDIT_REQUIRED_NO_GPU_BENCHMARK")
