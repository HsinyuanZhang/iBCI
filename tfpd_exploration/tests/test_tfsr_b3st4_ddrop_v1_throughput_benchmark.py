"""Synthetic CPU/adversarial tests for the engineering-only throughput audit."""
from __future__ import annotations

import copy
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tfpd_exploration"))
from src.tfsr_b3st4_ddrop_v1 import throughput_benchmark as bench


def _sha(label: str) -> str:
    return bench._sha(label.encode("ascii"))


def _closure(*, salt: str = "") -> dict[str, object]:
    hashes = {path: _sha("closure:" + salt + path) for path in bench.BENCHMARK_CLOSURE}
    hashes[bench.WORKORDER_RELATIVE] = bench.WORKORDER_SHA256
    hashes[bench.MODEL_RELATIVE] = bench.MODEL_SHA256
    hashes[bench.NATIVE_ACTIVITY_RELATIVE] = bench.NATIVE_ACTIVITY_SHA256
    hashes[bench.LIVE_THROUGHPUT_RELATIVE] = bench.LIVE_THROUGHPUT_SHA256
    return {
        "paths": list(bench.BENCHMARK_CLOSURE),
        "sha256_by_path": hashes,
        "closure_sha256": bench._closure_digest(hashes),
    }


def _runtime() -> dict[str, object]:
    return {
        "gpu": dict(bench.FROZEN_GPU0), "torch_version": "synthetic-torch",
        "cuda_version": "synthetic-cuda", "cudnn_version": "synthetic-cudnn",
        "cpu_threads": {"intraop": 1, "interop": 1},
    }


def _equivalence(
    kind: str,
    *,
    exact: bool,
    gru: bool = False,
    differences: Mapping[str, float] | None = None,
) -> dict[str, object]:
    raw_differences = {
        "forward_max_abs": 0.0 if exact else 1e-6,
        "loss_abs": 0.0 if exact else 2e-6,
        "gradient_max_abs": 0.0 if exact else 3e-6,
        "model_state_max_abs": 0.0 if exact else 4e-6,
        "optimizer_state_max_abs": 0.0 if exact else 5e-6,
    }
    if differences is not None:
        raw_differences = dict(differences)
    actual_exact = all(value == 0.0 for value in raw_differences.values())
    within_tolerance = all(
        raw_differences[key] <= bench.MATHEMATICAL_FP32_TOLERANCES[key]
        for key in bench.MATHEMATICAL_FP32_TOLERANCES
    )
    return {
        "label": "production_contract_vs_" + kind + "_b32",
        "same_initial_model_state": True,
        "same_initial_fresh_adam_state": True,
        "differences": raw_differences,
        "all_finite": {
            "baseline_model": True, "candidate_model": True,
            "baseline_optimizer": True, "candidate_optimizer": True,
        },
        "exact_match": actual_exact,
        "fp32_tolerance_map": dict(bench.MATHEMATICAL_FP32_TOLERANCES),
        "within_tolerance": within_tolerance,
        "comparison_role": (
            bench.MATHEMATICAL_TIMING_GATE if kind == "mathematical" else bench.DIAGNOSTIC_NON_AUTHORIZING
        ),
        "gru_tensor_copy_map": dict(bench.GRU_TENSOR_COPY_MAP) if gru else None,
    }


def _mathematical_at_fp32_threshold() -> dict[str, object]:
    """A valid non-bitwise mathematical comparison exactly on all bounds."""
    return _equivalence(
        "mathematical",
        exact=False,
        differences=dict(bench.MATHEMATICAL_FP32_TOLERANCES),
    )


class _FakeBackend:
    """No-Torch matrix backend with auditable fresh-state/OOM behavior."""

    reference_model_state_sha256 = _sha("reference-model")
    reference_optimizer_state_sha256 = _sha("reference-adam")

    def __init__(
        self,
        *,
        oom_batches: set[int] | None = None,
        oom_labels: set[str] | None = None,
        compile_unavailable_labels: set[str] | None = None,
        bad_fresh: bool = False,
        mathematical_equivalence: Mapping[str, Any] | None = None,
    ) -> None:
        self.oom_batches = set() if oom_batches is None else set(oom_batches)
        self.oom_labels = set() if oom_labels is None else set(oom_labels)
        self.compile_unavailable_labels = (
            set() if compile_unavailable_labels is None else set(compile_unavailable_labels)
        )
        self.bad_fresh = bad_fresh
        self.mathematical_equivalence = (
            None if mathematical_equivalence is None else copy.deepcopy(dict(mathematical_equivalence))
        )
        self.calls: list[tuple[str, int]] = []
        self.cleanup_calls: list[tuple[str, int]] = []
        self.comparisons: list[str] = []

    def compare_before_timing(self, *, kind: str, batch_size: int) -> Mapping[str, Any]:
        self.comparisons.append(kind)
        if kind == "mathematical" and self.mathematical_equivalence is not None:
            return copy.deepcopy(self.mathematical_equivalence)
        return _equivalence(kind, exact=(kind == "mathematical"), gru=(kind == "grucell_equivalent"))

    def measure(
        self,
        *,
        label: str,
        kind: str,
        execution: str,
        batch_size: int,
        equivalence: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        self.calls.append((label, batch_size))
        initial_model = _sha("wrong-model") if self.bad_fresh else self.reference_model_state_sha256
        common = {
            "schema": bench.CELL_SCHEMA, "label": label, "kind": kind, "execution": execution,
            "batch_size": batch_size, "units": 128, "dtype": "float32",
            "warmup_steps": 5, "measured_steps": 20, "typed_normalized_t4": True,
            "initial_model_state_sha256": initial_model,
            "initial_optimizer_state_sha256": self.reference_optimizer_state_sha256,
            "state_before_measure_sha256": _sha("after-warmup:" + label),
            "optimizer_before_measure_sha256": _sha("after-warmup-adam:" + label),
            "fresh_model_from_reference": True, "fresh_optimizer_state": True, "reused_warmed_state": False,
            "compile_exception_class": None, "compile_exception_repr_sha256": None,
            "compile_failure_stage": None, "post_error_cuda_cleanup_reset": False,
            "runtime": _runtime(), "equivalence": None if equivalence is None else dict(equivalence),
        }
        if batch_size in self.oom_batches or label in self.oom_labels:
            return {
                **common, "status": "CUDA_OOM", "cuda_oom": True, "oom_error_sha256": _sha("oom:" + label),
                "finite": None, "post_measurement_finite": None, "timing": None,
                "resources": {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0, "rss_bytes": 1,
                              "oom_cleanup_reset": True},
                "equivalence": None,
            }
        if label in self.compile_unavailable_labels:
            return {
                **common, "status": bench.COMPILE_UNAVAILABLE, "cuda_oom": False, "oom_error_sha256": None,
                "compile_exception_class": "builtins.RuntimeError",
                "compile_exception_repr_sha256": _sha("compile-unavailable:" + label),
                "compile_failure_stage": "variant_construction", "post_error_cuda_cleanup_reset": True,
                "finite": None, "post_measurement_finite": None, "timing": None,
                "resources": {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0, "rss_bytes": 1,
                              "oom_cleanup_reset": False},
                "equivalence": None,
            }
        rate = 10.0
        return {
            **common, "status": "MEASURED", "cuda_oom": False, "oom_error_sha256": None,
            "finite": {"forward": True, "loss": True, "gradient": True},
            "post_measurement_finite": {"model": True, "optimizer": True},
            "timing": {
                "warmup_wall_seconds": 0.1, "measured_total_wall_seconds": 2.0,
                "median_step_wall_seconds": 0.1, "steps_per_second": rate,
                "samples_per_second": rate * batch_size,
                "projected_48epoch_seconds": bench.TRAINING_STEPS_48_EPOCHS / rate,
                "projection_label": bench.ENGINEERING_PROJECTION_LABEL,
            },
            "resources": {"peak_allocated_bytes": 2, "peak_reserved_bytes": 3, "rss_bytes": 4,
                          "oom_cleanup_reset": False},
        }

    def cleanup_after_oom(self, *, label: str, batch_size: int) -> None:
        self.cleanup_calls.append((label, batch_size))


def _matrix(
    *,
    oom_batches: set[int] | None = None,
    oom_labels: set[str] | None = None,
    compile_unavailable_labels: set[str] | None = None,
) -> dict[str, object]:
    return bench.run_benchmark_matrix(
        _FakeBackend(
            oom_batches=oom_batches,
            oom_labels=oom_labels,
            compile_unavailable_labels=compile_unavailable_labels,
        )
    )


def _baseline() -> dict[str, object]:
    return {
        "relative_path": bench.LIVE_THROUGHPUT_RELATIVE,
        "body_sha256": bench.LIVE_THROUGHPUT_SHA256,
        "steps_per_second": bench.FROZEN_BASELINE["steps_per_second"],
        "elapsed_seconds": bench.FROZEN_BASELINE["elapsed_seconds"],
        "projected_48epoch_seconds": bench.FROZEN_BASELINE["projected_48epoch_seconds"],
    }


def _receipt() -> dict[str, object]:
    closure = _closure()
    return bench.build_engineering_receipt(
        baseline=_baseline(), matrix=_matrix(), launch_closure=closure, final_closure=closure, runtime=_runtime(),
    )


def _guarded_cli_env(tmp_path: Path) -> dict[str, str]:
    guard = tmp_path / "sitecustomize.py"
    guard.write_text(
        "import builtins\n"
        "_real = builtins.__import__\n"
        "def _guard(name, *args, **kwargs):\n"
        "    if name == 'torch' or name.startswith('torch.'):\n"
        "        raise RuntimeError('TORCH_IMPORT_FORBIDDEN')\n"
        "    return _real(name, *args, **kwargs)\n"
        "builtins.__import__ = _guard\n"
    )
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(tmp_path), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    })
    return env


def test_zero_argument_cli_is_static_dry_and_never_imports_torch(tmp_path: Path):
    script = ROOT / "tfpd_exploration/scripts/benchmark_tfsr_b3st4_ddrop_throughput.py"
    completed = subprocess.run([sys.executable, str(script)], text=True, capture_output=True, env=_guarded_cli_env(tmp_path))
    assert completed.returncode == 0, completed.stderr
    assert "TORCH_IMPORT_FORBIDDEN" not in completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["status"] == "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_BENCHMARK"
    assert plan["authorization"] == "ROOT_AUDIT_REQUIRED"
    assert plan["closure"]["non_globbed"] is True
    assert plan["closure"]["native_activity_sha256"] == bench.NATIVE_ACTIVITY_SHA256


def test_partial_flags_reject_before_torch_and_paired_flags_stop_at_root_audit(tmp_path: Path):
    script = ROOT / "tfpd_exploration/scripts/benchmark_tfsr_b3st4_ddrop_throughput.py"
    env = _guarded_cli_env(tmp_path)
    partial = subprocess.run([sys.executable, str(script), "--execute"], text=True, capture_output=True, env=env)
    assert partial.returncode != 0
    assert "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_BENCHMARK" in partial.stderr
    assert "TORCH_IMPORT_FORBIDDEN" not in partial.stderr
    paired = subprocess.run(
        [sys.executable, str(script), "--execute", "--i-have-root-throughput-authorization"],
        text=True, capture_output=True, env=env,
    )
    assert paired.returncode != 0
    assert "ROOT_AUDIT_REQUIRED_NO_GPU_BENCHMARK" in paired.stderr
    assert "TORCH_IMPORT_FORBIDDEN" not in paired.stderr


def test_explicit_non_globbed_closure_binds_only_allowed_files_model_and_receipt():
    assert bench.BENCHMARK_CLOSURE == (
        bench.WORKORDER_RELATIVE, bench.SOURCE_RELATIVE, bench.CLI_RELATIVE, bench.TEST_RELATIVE,
        bench.MODEL_RELATIVE, bench.NATIVE_ACTIVITY_RELATIVE,
        bench.LIVE_THROUGHPUT_RELATIVE, bench.LIVE_THROUGHPUT_SIDECAR_RELATIVE,
    )
    assert all("*" not in path and "?" not in path for path in bench.BENCHMARK_CLOSURE)
    assert bench.validate_closure_payload(_closure())["closure_sha256"] == _closure()["closure_sha256"]
    forged = _closure()
    forged["sha256_by_path"][bench.MODEL_RELATIVE] = _sha("wrong-model")
    forged["closure_sha256"] = bench._closure_digest(forged["sha256_by_path"])
    with pytest.raises(bench.FailClosedError, match="model closure"):
        bench.validate_closure_payload(forged)
    forged_native = _closure()
    forged_native["sha256_by_path"][bench.NATIVE_ACTIVITY_RELATIVE] = _sha("wrong-native-activity")
    forged_native["closure_sha256"] = bench._closure_digest(forged_native["sha256_by_path"])
    with pytest.raises(bench.FailClosedError, match="native causal activity"):
        bench.validate_closure_payload(forged_native)


class _FakeCuda:
    def __init__(self, *, torch_total_memory_bytes: int | None = None) -> None:
        self.calls: list[str] = []
        self.torch_total_memory_bytes = (
            bench.FROZEN_GPU0["torch_total_memory_bytes"]
            if torch_total_memory_bytes is None else torch_total_memory_bytes
        )

    def is_available(self) -> bool:
        self.calls.append("is_available")
        return True

    def device_count(self) -> int:
        self.calls.append("device_count")
        return 1

    def get_device_properties(self, index: int) -> Any:
        self.calls.append("properties")
        assert index == 0
        return SimpleNamespace(name=bench.FROZEN_GPU0["name"], total_memory=self.torch_total_memory_bytes)


class _FakeTorch:
    def __init__(self, *, torch_total_memory_bytes: int | None = None) -> None:
        self.cuda = _FakeCuda(torch_total_memory_bytes=torch_total_memory_bytes)
        self.__version__ = "fake"
        self.version = SimpleNamespace(cuda="fake-cuda")
        self.backends = SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 123))

    @staticmethod
    def device(name: str) -> Any:
        assert name == "cuda:0"
        return SimpleNamespace(type="cuda", index=0)


def _nominal_gpu0() -> dict[str, object]:
    return {
        "uuid": bench.FROZEN_GPU0["uuid"], "bdf": bench.FROZEN_GPU0["bdf"],
        "name": bench.FROZEN_GPU0["name"],
        "nvidia_smi_memory_total_mib": bench.FROZEN_GPU0["nvidia_smi_memory_total_mib"],
    }


def test_gpu0_contract_keeps_nvidia_nominal_and_torch_byte_authorities_separate(monkeypatch: pytest.MonkeyPatch):
    torch = _FakeTorch()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(bench.FailClosedError, match="CUDA_VISIBLE_DEVICES=0"):
        bench.require_exact_gpu0(torch, query_gpu0=_nominal_gpu0)
    assert torch.cuda.calls == []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    result = bench.require_exact_gpu0(torch, query_gpu0=_nominal_gpu0)
    assert result["gpu"] == bench.FROZEN_GPU0
    assert result["cpu_threads"] == {"intraop": 1, "interop": 1}

    wrong_nominal = _nominal_gpu0()
    wrong_nominal["nvidia_smi_memory_total_mib"] = int(wrong_nominal["nvidia_smi_memory_total_mib"]) - 1
    with pytest.raises(bench.FailClosedError, match="nvidia-smi GPU0 identity"):
        bench.require_exact_gpu0(_FakeTorch(), query_gpu0=lambda: wrong_nominal)

    wrong_torch_bytes = _FakeTorch(torch_total_memory_bytes=bench.FROZEN_GPU0["torch_total_memory_bytes"] - 1)
    with pytest.raises(bench.FailClosedError, match="total-memory authority"):
        bench.require_exact_gpu0(wrong_torch_bytes, query_gpu0=_nominal_gpu0)


def test_synthetic_input_requires_typed_normalized_t4_capability_cpu_only():
    import torch
    from src.tfsr_b3st4_ddrop_v1 import model

    batch = bench.make_synthetic_batch(
        torch=torch, model_module=model, batch_size=1, device=torch.device("cpu"), spec=bench.PUBLIC_SPEC,
    )
    bench.validate_synthetic_batch(
        batch, torch=torch, model_module=model, batch_size=1, device=torch.device("cpu"), spec=bench.PUBLIC_SPEC,
    )
    bare = bench.SyntheticBatch(
        x=batch.x, calib=batch.calib, normalized_t4=batch.normalized_t4.tensor, target=batch.target, valid=batch.valid,
    )
    with pytest.raises(bench.FailClosedError, match="typed NormalizedT4Batch"):
        bench.validate_synthetic_batch(
            bare, torch=torch, model_module=model, batch_size=1, device=torch.device("cpu"), spec=bench.PUBLIC_SPEC,
        )


def test_validation_hoisted_callthrough_matches_frozen_cpu_forward_and_dense_loss_without_cuda_init():
    """Exercise the real frozen submodules without entering CPU autograd.

    The installed CUDA-linked Torch build attempts device-driver discovery
    from its autograd engine even for CPU tensors.  A no-grad forward/loss
    proof still verifies the intended call-through topology and exact dense
    arithmetic while keeping this required focused test genuinely no-CUDA.
    """
    import torch
    from src.tfsr_b3st4_ddrop_v1 import model as model_module

    decoder = model_module.TFSRDecoder(capture_diagnostics=False)
    decoder.eval()
    batch = bench.make_synthetic_batch(
        torch=torch, model_module=model_module, batch_size=1, device=torch.device("cpu"), spec=bench.PUBLIC_SPEC,
    )
    with torch.no_grad():
        frozen_prediction = decoder(batch.x, batch.calib, batch.normalized_t4)
        hoisted_prediction = bench._validation_hoisted_forward(torch, decoder, batch)
        frozen_loss = model_module.TFSRDecoder.dense_valid_bin_mse(
            frozen_prediction, batch.target, batch.valid,
        )
        hoisted_loss = bench._validation_hoisted_dense_mse(torch, hoisted_prediction, batch)
    assert torch.equal(frozen_prediction, hoisted_prediction)
    assert torch.equal(frozen_loss, hoisted_loss)
    assert torch.cuda.is_initialized() is False


def test_grucell_copy_map_copies_every_gru_tensor_and_matches_cpu_forward_without_cuda_init():
    import torch

    # ``torch.manual_seed`` fans out to CUDA generators in this build.  A
    # dedicated CPU generator keeps this focused proof genuinely no-CUDA.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    baseline = torch.nn.GRU(515, 256, batch_first=True)
    candidate = bench.build_grucell_equivalent(torch, baseline)
    assert set(dict(candidate.named_parameters())) == {name.removeprefix("gru.") for name in bench.GRU_TENSOR_COPY_MAP.values()}
    x = torch.randn((1, 2, 515), generator=generator)
    # This environment's CUDA-linked Torch build attempts driver discovery
    # from the CPU autograd engine itself.  Keep focused tests genuinely
    # no-CUDA: the required gradient-difference receipt is schema-validated
    # below, while this concrete test proves all four copied tensors preserve
    # the actual CPU forward graph.
    with torch.no_grad():
        out_a, _ = baseline(x)
        out_b, _ = candidate(x)
    assert torch.allclose(out_a, out_b, rtol=1e-5, atol=1e-6)
    for baseline_name, candidate_name in bench.GRU_TENSOR_COPY_MAP.items():
        left = dict(baseline.named_parameters())[baseline_name.removeprefix("gru.")]
        right = dict(candidate.named_parameters())[candidate_name.removeprefix("gru.")]
        assert torch.equal(left, right)
    assert torch.cuda.is_initialized() is False


def test_matrix_enforces_fresh_state_exact_math_equivalence_and_stops_larger_eager_batches_after_oom():
    backend = _FakeBackend(oom_batches={64})
    matrix = bench.run_benchmark_matrix(backend)
    labels = [item[0] for item in backend.calls]
    assert "production_contract_eager_b64" in labels
    assert "production_contract_eager_b128" not in labels
    assert backend.cleanup_calls == [("production_contract_eager_b64", 64)]
    assert backend.comparisons == ["mathematical", "validation_hoisted", "grucell_equivalent"]
    assert matrix["eager_larger_batch_stopped_after_oom"] == 64
    assert matrix["fastest_non_oom_eager_batch"] == 32
    with pytest.raises(bench.FailClosedError, match="freshness"):
        bench.run_benchmark_matrix(_FakeBackend(bad_fresh=True))


def test_compiled_batch32_oom_is_recorded_once_and_never_retried_as_fastest_batch():
    backend = _FakeBackend(oom_labels={"mathematical_compile_b32"})
    matrix = bench.run_benchmark_matrix(backend)
    labels = [label for label, _batch in backend.calls]
    assert labels.count("mathematical_compile_b32") == 1
    assert "mathematical_compile_fastest_eager_b128" not in labels
    assert "mathematical_compile_fastest_eager_b32" not in labels
    assert matrix["compiled_fastest_eager_skipped"] == "compiled_batch32_oom"
    assert matrix["compiled_fastest_eager_attempted"] is False
    assert backend.cleanup_calls.count(("mathematical_compile_b32", 32)) == 1


def test_compiled_batch32_unavailable_skips_fastest_once_and_still_runs_gru():
    backend = _FakeBackend(compile_unavailable_labels={"mathematical_compile_b32"})
    matrix = bench.run_benchmark_matrix(backend)
    labels = [label for label, _batch in backend.calls]
    assert "mathematical_compile_b32" in labels
    assert not any(label.startswith("mathematical_compile_fastest_eager") for label in labels)
    assert labels[-1] == "grucell_equivalent_eager_b32"
    assert matrix["compiled_fastest_eager_skipped"] == "compiled_batch32_implementation_unavailable"
    assert matrix["compiled_fastest_eager_attempted"] is False
    compiled = next(item for item in matrix["cells"] if item["label"] == "mathematical_compile_b32")
    assert compiled["status"] == bench.COMPILE_UNAVAILABLE
    assert compiled["compile_exception_class"] == "builtins.RuntimeError"
    assert compiled["post_error_cuda_cleanup_reset"] is True
    assert bench._validate_matrix_payload(matrix)["compiled_fastest_eager_skipped"] == (
        "compiled_batch32_implementation_unavailable"
    )


def test_compiled_fastest_may_be_unavailable_after_measured_b32_and_still_finishes_gru():
    matrix = _matrix(compile_unavailable_labels={"mathematical_compile_fastest_eager_b128"})
    labels = [item["label"] for item in matrix["cells"]]
    assert labels[-2:] == ["mathematical_compile_fastest_eager_b128", "grucell_equivalent_eager_b32"]
    fastest = matrix["cells"][-2]
    assert fastest["status"] == bench.COMPILE_UNAVAILABLE
    assert matrix["compiled_fastest_eager_attempted"] is True
    assert matrix["compiled_fastest_eager_skipped"] is None
    assert bench._validate_matrix_payload(matrix)["cells"][-2]["status"] == bench.COMPILE_UNAVAILABLE


def test_compile_unavailable_matrix_markers_order_scope_and_cleanup_are_not_forgeable():
    unavailable = _matrix(compile_unavailable_labels={"mathematical_compile_b32"})
    normal = _matrix()
    fastest = next(item for item in normal["cells"] if item["label"] == "mathematical_compile_fastest_eager_b128")

    forged_fastest = copy.deepcopy(unavailable)
    forged_fastest["cells"].insert(-1, copy.deepcopy(fastest))
    with pytest.raises(bench.FailClosedError, match="attempted matrix order"):
        bench._validate_matrix_payload(forged_fastest)

    forged_marker = copy.deepcopy(unavailable)
    forged_marker["compiled_fastest_eager_skipped"] = "compiled_batch32_oom"
    with pytest.raises(bench.FailClosedError, match="implementation-unavailable marker/status"):
        bench._validate_matrix_payload(forged_marker)

    forged_cleanup = copy.deepcopy(unavailable)
    compiled = next(item for item in forged_cleanup["cells"] if item["label"] == "mathematical_compile_b32")
    compiled["post_error_cuda_cleanup_reset"] = False
    with pytest.raises(bench.FailClosedError, match="CUDA cleanup/reset proof"):
        bench._validate_matrix_payload(forged_cleanup)

    forged_stage = copy.deepcopy(unavailable)
    compiled = next(item for item in forged_stage["cells"] if item["label"] == "mathematical_compile_b32")
    compiled["compile_failure_stage"] = "made_up_stage"
    with pytest.raises(bench.FailClosedError, match="failure-stage"):
        bench._validate_matrix_payload(forged_stage)

    forged_exception = copy.deepcopy(unavailable)
    compiled = next(item for item in forged_exception["cells"] if item["label"] == "mathematical_compile_b32")
    compiled["compile_exception_repr_sha256"] = "not-a-sha"
    with pytest.raises(bench.FailClosedError, match="exception repr"):
        bench._validate_matrix_payload(forged_exception)

    with pytest.raises(bench.FailClosedError, match="compile-unavailable cell scope"):
        bench.run_benchmark_matrix(_FakeBackend(compile_unavailable_labels={"production_contract_eager_b32"}))

    receipt = bench.build_engineering_receipt(
        baseline=_baseline(), matrix=unavailable, launch_closure=_closure(), final_closure=_closure(), runtime=_runtime(),
    )
    assert bench.validate_engineering_receipt(receipt)["matrix"]["compiled_fastest_eager_skipped"] == (
        "compiled_batch32_implementation_unavailable"
    )


@pytest.mark.parametrize("stage", sorted(bench.COMPILE_FAILURE_STAGES))
def test_compile_unavailable_receipt_schema_allows_only_the_four_named_failure_stages(stage: str):
    matrix = _matrix(compile_unavailable_labels={"mathematical_compile_b32"})
    record = next(item for item in matrix["cells"] if item["label"] == "mathematical_compile_b32")
    record["compile_failure_stage"] = stage
    checked = bench.validate_cell_payload(
        record,
        label="mathematical_compile_b32", kind="mathematical", execution="compiled", batch_size=32,
        reference_model_sha256=_FakeBackend.reference_model_state_sha256,
        reference_optimizer_sha256=_FakeBackend.reference_optimizer_state_sha256,
    )
    assert checked["compile_failure_stage"] == stage


class _ExceptionRouteCuda:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def empty_cache(self) -> None:
        self.calls.append("empty_cache")

    def reset_peak_memory_stats(self, index: int) -> None:
        assert index == 0
        self.calls.append("reset_peak_memory_stats")

    def max_memory_allocated(self, index: int) -> int:
        assert index == 0
        self.calls.append("max_memory_allocated")
        return 0

    def max_memory_reserved(self, index: int) -> int:
        assert index == 0
        self.calls.append("max_memory_reserved")
        return 0


class _ExceptionRouteTorch:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self) -> None:
        self.cuda = _ExceptionRouteCuda()


def _physical_backend_for_synthetic_exception_route() -> tuple[Any, _ExceptionRouteTorch]:
    torch = _ExceptionRouteTorch()
    backend = bench.PhysicalBenchmarkBackend(ROOT)
    backend.torch = torch
    backend.model_module = object()
    backend.device = object()
    backend.runtime = _runtime()
    backend.snapshot = {}
    backend._reference_model_state_sha256 = _sha("physical-exception-reference-model")
    backend._reference_optimizer_state_sha256 = _sha("physical-exception-reference-adam")
    return backend, torch


def test_only_compiled_non_oom_exception_becomes_compile_unavailable_record(monkeypatch: pytest.MonkeyPatch):
    backend, torch = _physical_backend_for_synthetic_exception_route()

    def raise_compiler_error(**_kwargs: Any) -> Any:
        raise RuntimeError("synthetic compiler implementation unavailable")

    monkeypatch.setattr(backend, "_fresh_variant", raise_compiler_error)
    record = backend.measure(
        label="mathematical_compile_b32", kind="mathematical", execution="compiled", batch_size=32,
        equivalence=None,
    )
    assert record["status"] == bench.COMPILE_UNAVAILABLE
    assert record["compile_exception_class"] == "builtins.RuntimeError"
    assert record["compile_exception_repr_sha256"] == bench._sha(
        repr(RuntimeError("synthetic compiler implementation unavailable")).encode("utf-8")
    )
    assert record["compile_failure_stage"] == "variant_construction"
    assert record["post_error_cuda_cleanup_reset"] is True
    assert record["timing"] is None and record["equivalence"] is None
    assert torch.cuda.calls == [
        "empty_cache", "reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved",
    ]
    checked = bench.validate_cell_payload(
        record,
        label="mathematical_compile_b32", kind="mathematical", execution="compiled", batch_size=32,
        reference_model_sha256=backend.reference_model_state_sha256,
        reference_optimizer_sha256=backend.reference_optimizer_state_sha256,
    )
    assert checked["status"] == bench.COMPILE_UNAVAILABLE


def test_compiled_fail_closed_error_never_becomes_compile_unavailable_record(monkeypatch: pytest.MonkeyPatch):
    backend, torch = _physical_backend_for_synthetic_exception_route()

    def raise_contract_error(**_kwargs: Any) -> Any:
        raise bench.FailClosedError("synthetic compiled numerical/provenance failure")

    monkeypatch.setattr(backend, "_fresh_variant", raise_contract_error)
    with pytest.raises(bench.FailClosedError, match="synthetic compiled numerical/provenance failure"):
        backend.measure(
            label="mathematical_compile_b32", kind="mathematical", execution="compiled", batch_size=32,
            equivalence=None,
        )
    # A route-owned contract failure neither receives allocator cleanup as an
    # availability record nor supplies a synthetic receipt payload.
    assert torch.cuda.calls == []


@pytest.mark.parametrize(
    ("label", "kind", "execution"),
    (
        ("production_contract_eager_b32", "production_contract", "eager"),
        ("mathematical_eager_b32", "mathematical", "eager"),
        ("validation_hoisted_eager_b32", "validation_hoisted", "eager"),
        ("grucell_equivalent_eager_b32", "grucell_equivalent", "candidate"),
    ),
)
def test_noncompiled_non_oom_exceptions_never_become_receipt_records(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    kind: str,
    execution: str,
):
    backend, torch = _physical_backend_for_synthetic_exception_route()

    def raise_compiler_error(**_kwargs: Any) -> Any:
        raise RuntimeError("synthetic noncompiled route failure")

    monkeypatch.setattr(backend, "_fresh_variant", raise_compiler_error)
    with pytest.raises(RuntimeError, match="synthetic noncompiled route failure"):
        backend.measure(
            label=label, kind=kind, execution=execution, batch_size=32,
            equivalence=None,
        )
    # No cleanup record is emitted and no exception is silently transformed on
    # any noncompiled production, mathematical, hoisted, or candidate path.
    assert torch.cuda.calls == []


def test_matrix_receipt_reconstructs_exact_attempt_order_and_oom_stops():
    normal = _matrix()
    assert bench._validate_matrix_payload(normal)["eager_larger_batch_stopped_after_oom"] is None

    # A real batch-64 OOM stops the larger eager cell.  Injecting a genuine,
    # otherwise valid b128 receipt cannot turn that forbidden retry into a
    # valid matrix simply by retaining the stop marker.
    b64_oom = _matrix(oom_batches={64})
    b128 = next(item for item in normal["cells"] if item["label"] == "production_contract_eager_b128")
    b64_oom["cells"].insert(4, copy.deepcopy(b128))
    with pytest.raises(bench.FailClosedError, match="attempted matrix order"):
        bench._validate_matrix_payload(b64_oom)

    # Conversely, a measured b64 has to be followed by the declared b128
    # attempt; omission cannot masquerade as an unrecorded stop.
    missing_b128 = copy.deepcopy(normal)
    del missing_b128["cells"][4]
    with pytest.raises(bench.FailClosedError, match="attempted matrix order"):
        bench._validate_matrix_payload(missing_b128)

    bad_eager_marker = _matrix(oom_batches={64})
    bad_eager_marker["eager_larger_batch_stopped_after_oom"] = None
    with pytest.raises(bench.FailClosedError, match="OOM stop marker/status"):
        bench._validate_matrix_payload(bad_eager_marker)

    b128_oom = _matrix(oom_batches={128})
    assert bench._validate_matrix_payload(b128_oom)["eager_larger_batch_stopped_after_oom"] == 128
    b128_oom["eager_larger_batch_stopped_after_oom"] = 64
    with pytest.raises(bench.FailClosedError, match="OOM stop marker/status"):
        bench._validate_matrix_payload(b128_oom)

    # Once compiled b32 OOMs, its fastest-batch sibling is forbidden even if
    # a valid result from another attempted matrix is spliced in.
    compiled_oom = _matrix(oom_labels={"mathematical_compile_b32"})
    compiled_fastest = next(
        item for item in normal["cells"] if item["label"] == "mathematical_compile_fastest_eager_b128"
    )
    compiled_oom["cells"].insert(-1, copy.deepcopy(compiled_fastest))
    with pytest.raises(bench.FailClosedError, match="attempted matrix order"):
        bench._validate_matrix_payload(compiled_oom)

    bad_compiled_marker = _matrix(oom_labels={"mathematical_compile_b32"})
    bad_compiled_marker["compiled_fastest_eager_skipped"] = None
    with pytest.raises(bench.FailClosedError, match="compiled-fastest OOM marker/status"):
        bench._validate_matrix_payload(bad_compiled_marker)

    # A non-OOM compiled b32 must be followed immediately by a compiled cell
    # at the independently reconstructed fastest eager batch, then GRU last.
    delayed_compiled_fastest = copy.deepcopy(normal)
    delayed_compiled_fastest["cells"].append(delayed_compiled_fastest["cells"].pop(6))
    with pytest.raises(bench.FailClosedError, match="attempted matrix order"):
        bench._validate_matrix_payload(delayed_compiled_fastest)


def test_math_equivalence_uses_frozen_fp32_gate_and_hoisted_gru_are_diagnostic_only():
    backend = _FakeBackend()
    math = backend.compare_before_timing(kind="mathematical", batch_size=32)
    checked_exact = bench._validate_equivalence(
        math, require_mathematical_timing_gate=True, require_gru_map=False,
    )
    assert checked_exact["exact_match"] is True
    assert checked_exact["within_tolerance"] is True
    assert checked_exact["comparison_role"] == bench.MATHEMATICAL_TIMING_GATE

    at_threshold = _mathematical_at_fp32_threshold()
    checked_threshold = bench._validate_equivalence(
        at_threshold, require_mathematical_timing_gate=True, require_gru_map=False,
    )
    assert checked_threshold["exact_match"] is False
    assert checked_threshold["within_tolerance"] is True
    assert checked_threshold["fp32_tolerance_map"] == bench.MATHEMATICAL_FP32_TOLERANCES

    forged = copy.deepcopy(math)
    forged["differences"]["forward_max_abs"] = 0.01
    forged["exact_match"] = True
    with pytest.raises(bench.FailClosedError, match="exact-match"):
        bench._validate_equivalence(
            forged, require_mathematical_timing_gate=True, require_gru_map=False,
        )

    hoisted = bench._validate_equivalence(
        backend.compare_before_timing(kind="validation_hoisted", batch_size=32),
        require_mathematical_timing_gate=False, require_gru_map=False,
    )
    assert hoisted["gru_tensor_copy_map"] is None
    assert hoisted["comparison_role"] == bench.DIAGNOSTIC_NON_AUTHORIZING
    assert hoisted["within_tolerance"] is False
    gru = bench._validate_equivalence(
        backend.compare_before_timing(kind="grucell_equivalent", batch_size=32),
        require_mathematical_timing_gate=False, require_gru_map=True,
    )
    assert gru["gru_tensor_copy_map"] == bench.GRU_TENSOR_COPY_MAP
    assert gru["comparison_role"] == bench.DIAGNOSTIC_NON_AUTHORIZING
    assert set(gru["differences"]) == {
        "forward_max_abs", "loss_abs", "gradient_max_abs", "model_state_max_abs", "optimizer_state_max_abs",
    }


@pytest.mark.parametrize("difference_key", sorted(bench.MATHEMATICAL_FP32_TOLERANCES))
def test_math_fp32_tolerance_boundary_and_any_single_excess_fail(difference_key: str):
    at_threshold = _mathematical_at_fp32_threshold()
    assert bench._validate_equivalence(
        at_threshold, require_mathematical_timing_gate=True, require_gru_map=False,
    )["within_tolerance"] is True

    above_threshold = copy.deepcopy(at_threshold)
    threshold = float(bench.MATHEMATICAL_FP32_TOLERANCES[difference_key])
    above_threshold["differences"][difference_key] = threshold + 1e-12
    above_threshold["exact_match"] = False
    above_threshold["within_tolerance"] = False
    with pytest.raises(bench.FailClosedError, match="tolerance gate"):
        bench._validate_equivalence(
            above_threshold, require_mathematical_timing_gate=True, require_gru_map=False,
        )


def test_math_timing_cannot_begin_without_valid_within_tolerance_evidence():
    rejected = _mathematical_at_fp32_threshold()
    rejected["differences"]["gradient_max_abs"] = 1.0e-6 + 1.0e-12
    rejected["exact_match"] = False
    rejected["within_tolerance"] = False
    backend = _FakeBackend(mathematical_equivalence=rejected)
    with pytest.raises(bench.FailClosedError, match="tolerance gate"):
        bench.run_benchmark_matrix(backend)
    # The production baseline can be measured, but the mathematical cell's
    # timer has not been opened after its comparison failed.
    assert backend.calls == [("production_contract_eager_b32", 32)]


def test_equivalence_rejects_forged_tolerance_exact_and_within_flags():
    valid = _mathematical_at_fp32_threshold()

    forged_within = copy.deepcopy(valid)
    forged_within["within_tolerance"] = False
    with pytest.raises(bench.FailClosedError, match="within-tolerance"):
        bench._validate_equivalence(
            forged_within, require_mathematical_timing_gate=True, require_gru_map=False,
        )

    forged_exact = copy.deepcopy(valid)
    forged_exact["exact_match"] = True
    with pytest.raises(bench.FailClosedError, match="exact-match"):
        bench._validate_equivalence(
            forged_exact, require_mathematical_timing_gate=True, require_gru_map=False,
        )

    forged_tolerance = copy.deepcopy(valid)
    forged_tolerance["fp32_tolerance_map"]["gradient_max_abs"] = 2.0e-6
    with pytest.raises(bench.FailClosedError, match="tolerance-map drift"):
        bench._validate_equivalence(
            forged_tolerance, require_mathematical_timing_gate=True, require_gru_map=False,
        )


def test_receipt_boundaries_closure_drift_and_transactional_0444_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    receipt = _receipt()
    checked = bench.validate_engineering_receipt(receipt)
    assert checked["matrix"]["cells"][0]["label"] == "production_contract_eager_b32"
    parent = tmp_path / "tfpd_exploration" / "results"
    parent.mkdir(parents=True)
    root = tmp_path
    launch, drifted = _closure(), _closure(salt="drift")
    with pytest.raises(bench.FailClosedError, match="closure drift prevents"):
        bench.publish_after_stable_final_closure(
            root=root, receipt=receipt, launch_closure=launch, final_closure=drifted,
        )
    assert not (parent / "tfsr_b3st4_ddrop_throughput_engineering_v1").exists()

    artifact = bench.reserve_output_root(tmp_path, "transaction")
    digest = artifact.publish_receipt(receipt)
    assert artifact._read_pair("receipt.json", digest) == bench._json_bytes(receipt)
    assert stat.S_IMODE((artifact.directory / "receipt.json").stat().st_mode) == 0o444
    assert stat.S_IMODE((artifact.directory / "receipt.json.sha256").stat().st_mode) == 0o444
    with pytest.raises(bench.FailClosedError, match="collision"):
        artifact.publish_receipt(receipt)

    rollback = bench.reserve_output_root(tmp_path, "rollback")
    original_write = bench._write_full
    calls = {"count": 0}

    def fail_second_write(descriptor: int, body: bytes) -> None:
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("synthetic sidecar write failure")
        original_write(descriptor, body)

    monkeypatch.setattr(bench, "_write_full", fail_second_write)
    with pytest.raises(OSError, match="sidecar"):
        rollback.publish_receipt(receipt)
    assert not (rollback.directory / "receipt.json").exists()
    assert not (rollback.directory / "receipt.json.sha256").exists()


def test_receipt_is_engineering_only_and_canonical_root_is_currently_fresh():
    receipt = _receipt()
    assert receipt["purpose"] == "ENGINEERING_ONLY"
    assert receipt["scientific_result"] is False
    assert receipt["data_opened"] is False
    assert receipt["checkpoint_opened"] is False
    assert receipt["target_or_formal"] is False
    assert receipt["authorizes_training"] is False
    assert not (ROOT / bench.OUTPUT_ROOT_RELATIVE).exists()
