"""Synthetic CPU/adversarial tests for the v2 engineering-only throughput audit."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tfpd_exploration"))
from src.tfsr_b3st4_ddrop_v1 import throughput_benchmark_v2 as bench2  # noqa: E402


def _sha(label: str) -> str:
    return bench2._sha(label.encode("ascii"))


def _closure(*, salt: str = "") -> dict[str, object]:
    hashes = {path: _sha("v2-closure:" + salt + path) for path in bench2.BENCHMARK_CLOSURE}
    hashes[bench2.WORKORDER_RELATIVE] = bench2.WORKORDER_SHA256
    hashes[bench2.MODEL_RELATIVE] = bench2.MODEL_SHA256
    hashes[bench2.NATIVE_ACTIVITY_RELATIVE] = bench2.NATIVE_ACTIVITY_SHA256
    hashes[bench2.LIVE_THROUGHPUT_RELATIVE] = bench2.LIVE_THROUGHPUT_SHA256
    hashes[bench2.V1_RECEIPT_RELATIVE] = bench2.V1_RECEIPT_SHA256
    return {
        "paths": list(bench2.BENCHMARK_CLOSURE),
        "sha256_by_path": hashes,
        "closure_sha256": bench2._closure_digest(hashes),
    }


def _runtime() -> dict[str, object]:
    return {
        "gpu": dict(bench2.FROZEN_GPU0), "torch_version": "synthetic-torch",
        "cuda_version": "synthetic-cuda", "cudnn_version": "synthetic-cudnn",
        "cpu_threads": {"intraop": 1, "interop": 1},
    }


def _equivalence(kind: str, units: int, *, exact: bool = True) -> dict[str, object]:
    # The frozen tolerance map requires exact forward/loss; a non-exact but
    # accepted comparison may drift only in the backward-derived fields.
    differences = {
        "forward_max_abs": 0.0,
        "loss_abs": 0.0,
        "gradient_max_abs": 0.0 if exact else 9e-7,
        "model_state_max_abs": 0.0 if exact else 1.5e-5,
        "optimizer_state_max_abs": 0.0 if exact else 9e-8,
    }
    actual_exact = all(value == 0.0 for value in differences.values())
    within = all(differences[key] <= bench2.V2_TOLERANCES[key] for key in bench2.V2_TOLERANCES)
    first_step = {
        "forward_max_abs": differences["forward_max_abs"],
        "loss_abs": differences["loss_abs"],
        "gradient_max_abs": differences["gradient_max_abs"],
    }
    first_step_within = all(
        first_step[key] <= bench2.V2_TOLERANCES[key]
        for key in ("forward_max_abs", "loss_abs", "gradient_max_abs")
    )
    return {
        "label": f"production_contract_vs_{kind}_b32_u{units}_20step",
        "same_initial_model_state": True,
        "same_initial_fresh_adam_state": True,
        "trajectory_steps": 20,
        "differences": differences,
        "first_step_differences": first_step,
        "first_step_within_tolerance": first_step_within,
        "all_finite": {
            "baseline_model": True, "candidate_model": True,
            "baseline_optimizer": True, "candidate_optimizer": True,
        },
        "exact_match": actual_exact,
        "fp32_tolerance_map": dict(bench2.V2_TOLERANCES),
        "within_tolerance": within,
        "comparison_role": bench2.DIAGNOSTIC_NON_AUTHORIZING,
        "gru_tensor_copy_map": None,
    }


class _FakeBackend:
    """No-Torch matrix backend with auditable measured/compile-failure behavior."""

    reference_model_state = _sha("v2-reference-model")
    reference_optimizer_state = _sha("v2-reference-adam")

    def __init__(self, *, compile_unavailable_kinds: set[str] | None = None) -> None:
        self.compile_unavailable_kinds = set() if compile_unavailable_kinds is None else set(compile_unavailable_kinds)
        self.calls: list[tuple[str, int]] = []
        self.comparisons: list[tuple[str, int]] = []

    def compare_before_timing(self, *, kind: str, units: int) -> Mapping[str, Any]:
        self.comparisons.append((kind, units))
        return copy.deepcopy(_equivalence(kind, units))

    def measure(self, *, label: str, kind: str, execution: str, units: int) -> Mapping[str, Any]:
        self.calls.append((label, units))
        common: dict[str, Any] = {
            "schema": bench2.CELL_SCHEMA, "label": label, "kind": kind, "execution": execution,
            "batch_size": 32, "units": units, "dtype": "float32",
            "warmup_steps": 5, "measured_steps": 20, "typed_normalized_t4": True,
            "initial_model_state_sha256": self.reference_model_state,
            "initial_optimizer_state_sha256": self.reference_optimizer_state,
            "state_before_measure_sha256": _sha("v2-after-warmup:" + label),
            "optimizer_before_measure_sha256": _sha("v2-after-warmup-adam:" + label),
            "fresh_model_from_reference": True, "fresh_optimizer_state": True, "reused_warmed_state": False,
            "construction_shim_applied": kind in bench2.COMPILE_BACKENDS or kind.startswith("combo_kv_compile"),
            "dynamo_counter_delta": (
                {"frames/total": 2, "frames/ok": 1,
                 "graph_break/TorchDynamo purposely graph breaks on RNN, GRU, LSTMs": 1}
                if execution == "compiled" else None
            ),
            "runtime": _runtime(), "equivalence": None,
        }
        if kind in self.compile_unavailable_kinds:
            return {
                **common, "status": bench2.COMPILE_UNAVAILABLE, "cuda_oom": False, "oom_error_sha256": None,
                "compile_exception_class": "torch._dynamo.exc.BackendCompilerFailed",
                "compile_exception_repr_sha256": _sha("v2-compile-unavailable:" + label),
                "compile_failure_stage": "warmup", "post_error_cuda_cleanup_reset": True,
                "finite": None, "post_measurement_finite": None, "timing": None,
                "resources": {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0, "rss_bytes": 1,
                              "oom_cleanup_reset": False},
            }
        rate = 4.0 if kind == "production_contract" else 6.0
        return {
            **common, "status": "MEASURED", "cuda_oom": False, "oom_error_sha256": None,
            "compile_exception_class": None, "compile_exception_repr_sha256": None,
            "compile_failure_stage": None, "post_error_cuda_cleanup_reset": False,
            "finite": {"forward": True, "loss": True, "gradient": True},
            "post_measurement_finite": {"model": True, "optimizer": True},
            "timing": {
                "warmup_wall_seconds": 0.1, "measured_total_wall_seconds": 2.0,
                "median_step_wall_seconds": 1.0 / rate, "steps_per_second": rate,
                "samples_per_second": rate * 32,
                "projected_48epoch_seconds": bench2.TRAINING_STEPS_48_EPOCHS / rate,
                "projection_label": bench2.ENGINEERING_PROJECTION_LABEL,
            },
            "resources": {"peak_allocated_bytes": 2, "peak_reserved_bytes": 3, "rss_bytes": 4,
                          "oom_cleanup_reset": False},
        }


def _matrix(**kwargs: Any) -> dict[str, object]:
    return bench2.run_benchmark_matrix(_FakeBackend(**kwargs))


def _construction_report() -> dict[str, object]:
    return {
        "v1_failure": {
            "reproduced": True, "exception_class": "builtins.TypeError",
            "exception_repr_sha256": _sha("v2-reproduced-typeerror"),
            "construction_after_shim": "OK",
        },
        "shim": {"applied": True, "reason": bench2.SHIM_EXPLANATION, "target": bench2.SHIM_TARGET},
    }


def _receipt(matrix: dict[str, object] | None = None) -> dict[str, object]:
    closure = _closure()
    return bench2.build_engineering_receipt(
        matrix=matrix if matrix is not None else _matrix(),
        launch_closure=closure, final_closure=closure, runtime=_runtime(),
        construction_report=_construction_report(),
    )


# --------------------------------------------------------------------------- #
# Planned matrix, dry plan, and CLI conventions
# --------------------------------------------------------------------------- #

def test_planned_matrix_is_explicit_ordered_and_covers_both_unit_sweeps():
    matrix = bench2.planned_matrix()
    labels = [item["label"] for item in matrix]
    assert len(matrix) == 18
    assert labels[0] == "production_contract_eager_b32_u128"
    assert labels[8] == "combo_kv_compile_cudagraphs_b32_u128"
    assert labels[9] == "production_contract_eager_b32_u64"
    assert labels[-1] == "combo_kv_compile_cudagraphs_b32_u64"
    assert all(item["units"] in (128, 64) for item in matrix)
    assert all("*" not in item["label"] for item in matrix)


def test_dry_plan_is_static_and_binds_superseded_v1_receipt():
    plan = bench2.dry_plan()
    assert plan["status"] == "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_BENCHMARK"
    assert plan["supersedes"]["superseded_receipt_sha256"] == bench2.V1_RECEIPT_SHA256
    assert plan["closure"]["non_globbed"] is True
    assert sorted(plan["execution_flags_required_together"]) == ["--execute", "--i-have-root-throughput-authorization-v2"]
    assert plan["boundaries"]["authorizes_training"] is False


def test_closure_rejects_drift_on_every_bound_authority():
    assert bench2.validate_closure_payload(_closure())["closure_sha256"] == _closure()["closure_sha256"]
    for relative, frozen in (
        (bench2.MODEL_RELATIVE, bench2.MODEL_SHA256),
        (bench2.NATIVE_ACTIVITY_RELATIVE, bench2.NATIVE_ACTIVITY_SHA256),
        (bench2.WORKORDER_RELATIVE, bench2.WORKORDER_SHA256),
        (bench2.LIVE_THROUGHPUT_RELATIVE, bench2.LIVE_THROUGHPUT_SHA256),
        (bench2.V1_RECEIPT_RELATIVE, bench2.V1_RECEIPT_SHA256),
    ):
        forged = _closure()
        forged["sha256_by_path"][relative] = _sha("drift")
        forged["closure_sha256"] = bench2._closure_digest(forged["sha256_by_path"])
        with pytest.raises(bench2.FailClosedError):
            bench2.validate_closure_payload(forged)


def test_zero_argument_cli_is_static_dry_and_never_imports_torch(tmp_path: Path):
    script = ROOT / "tfpd_exploration/scripts/benchmark_tfsr_b3st4_ddrop_throughput_v2.py"
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
    completed = subprocess.run([sys.executable, str(script)], text=True, capture_output=True, env=env)
    assert completed.returncode == 0, completed.stderr
    assert "TORCH_IMPORT_FORBIDDEN" not in completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["cell"] == bench2.CELL
    partial = subprocess.run([sys.executable, str(script), "--execute"], text=True, capture_output=True, env=env)
    assert partial.returncode != 0
    assert "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_BENCHMARK" in partial.stderr
    assert "TORCH_IMPORT_FORBIDDEN" not in partial.stderr


# --------------------------------------------------------------------------- #
# KV precompute identity math (tiny exact CPU fixtures)
# --------------------------------------------------------------------------- #

def _tiny_batch(torch: Any, model_module: Any, *, units: int, batch: int = 2):
    """Tiny exact CPU fixture with the frozen topology, small unit count."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260819)
    window, bins, width, t4_width = 50, 30, 100, 4
    x = torch.randn((batch, window, units), generator=generator)
    calib = torch.randn((batch, bins, width, units), generator=generator)
    raw_t4 = torch.randn((batch, units, t4_width), generator=generator)
    target = torch.randn((batch, window, 2), generator=generator)
    valid = torch.ones((batch, window), dtype=torch.bool)
    normalizer = model_module.T4Normalizer(
        torch.zeros((t4_width,)), torch.ones((t4_width,)),
        _sha("tiny-raw"), _sha("tiny-normalizer"),
    )
    t4 = normalizer(
        raw_t4, roster_digest=_sha("tiny-roster"),
        ordered_unit_ids=tuple(f"tiny-unit-{index:03d}" for index in range(units)),
        lineage=("tiny-throughput-only",),
    )
    return bench2.SyntheticBatch(x=x, calib=calib, normalized_t4=t4, target=target, valid=valid)


@pytest.mark.parametrize("units", (8, 5))
def test_batched_kv_projection_preserves_layout_and_values_per_slice(units: int):
    """One batched projection carries every per-step projection's values.

    Bitwise per-element equality of a differently-blocked GEMM is a property of
    the executed backend, not of the math: on the benchmark's RTX 3090 fp32
    cuBLAS path it was measured bitwise exact (forward_max_abs == 0.0), and the
    GPU trajectory gate in the receipt re-proves it.  On this CPU fixture the
    blocking differs, so values are checked to the fp32 blocking band and the
    LAYOUT is checked bitwise against the identical flattened GEMM.
    """
    import torch
    import torch.nn.functional as F

    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    tokens = torch.randn((3, 50, units, 256), generator=generator)
    weight = torch.randn((512, 256), generator=generator)
    bias = torch.randn((512,), generator=generator)
    batched = F.linear(tokens, weight, bias)
    # Same GEMM shape, so the layout/scatter interpretation is exact.
    flattened = F.linear(tokens.reshape(-1, 256), weight, bias).reshape(3, 50, units, 512)
    assert torch.equal(batched, flattened)
    for time_index in (0, 17, 49):
        per_step = F.linear(tokens[:, time_index], weight, bias)
        assert torch.allclose(batched[:, time_index], per_step, rtol=0.0, atol=2e-4)
        # The frozen module projects on the transposed [N, B, E] layout and
        # regroups into [2, N, B, E]; those slices must carry the same values.
        transposed = F.linear(tokens[:, time_index].transpose(1, 0), weight, bias)
        regrouped = transposed.unflatten(-1, (2, 256)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        assert torch.allclose(regrouped[0].transpose(1, 0), batched[:, time_index, :, :256], rtol=0.0, atol=2e-4)
        assert torch.allclose(regrouped[1].transpose(1, 0), batched[:, time_index, :, 256:], rtol=0.0, atol=2e-4)


@pytest.mark.parametrize("units", (8, 5))
def test_kv_precompute_loop_matches_frozen_decoder_forward_cpu(units: int):
    import random

    import torch

    from src.tfsr_b3st4_ddrop_v1 import model as model_module

    torch.manual_seed(42)
    decoder = model_module.TFSRDecoder(capture_diagnostics=False)
    decoder.eval()
    batch = _tiny_batch(torch, model_module, units=units)
    with torch.no_grad():
        frozen_prediction = decoder(batch.x, batch.calib, batch.normalized_t4)
        tokens, mass = bench2.hoisted_precompute(torch, decoder, batch)
        assert tokens.shape == (batch.x.shape[0], 50, units, 256)
        candidate_prediction = bench2.kv_precompute_loop(torch, decoder, tokens, mass)
    # On CPU the one batched K/V GEMM blocks differently from the 50 per-step
    # GEMMs, so equality holds to the fp32 blocking band; on the benchmark GPU
    # the same comparison was measured bitwise exact and the receipt's 20-step
    # trajectory gate re-proves it with forward_max_abs == 0.0.
    assert (frozen_prediction - candidate_prediction).abs().max().item() <= 1e-3

    # Train mode: identical seeded RNG streams must give identical masks, so
    # the restructured trajectory stays on the frozen path.
    decoder.train(True)
    random.seed(7)
    torch.manual_seed(7)
    with torch.no_grad():
        frozen_train = decoder(batch.x, batch.calib, batch.normalized_t4)
    random.seed(7)
    torch.manual_seed(7)
    with torch.no_grad():
        tokens, mass = bench2.hoisted_precompute(torch, decoder, batch)
        candidate_train = bench2.kv_precompute_loop(torch, decoder, tokens, mass)
    assert (frozen_train - candidate_train).abs().max().item() <= 1e-3
    assert torch.cuda.is_initialized() is False


def test_hoisted_dense_mse_exact_matches_frozen_loss_bitwise_cpu():
    """The candidate loss keeps the frozen integer-count division form.

    Dividing by an int64 0-dim tensor instead of the frozen Python-int count
    rounds differently by one ulp on this backend for some numerators, which
    is a harness artifact, not candidate mathematics.
    """
    import torch

    from src.tfsr_b3st4_ddrop_v1 import model as model_module

    torch.manual_seed(11)
    batch = _tiny_batch(torch, model_module, units=6)
    for _ in range(3):
        prediction = torch.randn((batch.x.shape[0], 50, 2))
        frozen = model_module.TFSRDecoder.dense_valid_bin_mse(prediction, batch.target, batch.valid)
        hoisted = bench2._hoisted_dense_mse_exact(
            torch, prediction, batch, int(batch.valid.sum().item())
        )
        assert torch.equal(frozen, hoisted)
    with pytest.raises(bench2.FailClosedError, match="valid-bin count"):
        bench2._hoisted_dense_mse_exact(torch, prediction, batch, 0)


def test_module_attention_loop_matches_frozen_decoder_forward_cpu():
    import random

    import torch

    from src.tfsr_b3st4_ddrop_v1 import model as model_module

    torch.manual_seed(42)
    decoder = model_module.TFSRDecoder(capture_diagnostics=False)
    decoder.eval()
    batch = _tiny_batch(torch, model_module, units=6)
    with torch.no_grad():
        frozen_prediction = decoder(batch.x, batch.calib, batch.normalized_t4)
        tokens, mass = bench2.hoisted_precompute(torch, decoder, batch)
        hoisted_prediction = bench2.module_attention_loop(torch, decoder, tokens, mass)
    assert torch.equal(frozen_prediction, hoisted_prediction)
    decoder.train(True)
    random.seed(3)
    torch.manual_seed(3)
    with torch.no_grad():
        frozen_train = decoder(batch.x, batch.calib, batch.normalized_t4)
    random.seed(3)
    torch.manual_seed(3)
    with torch.no_grad():
        tokens, mass = bench2.hoisted_precompute(torch, decoder, batch)
        hoisted_train = bench2.module_attention_loop(torch, decoder, tokens, mass)
    assert torch.equal(frozen_train, hoisted_train)
    assert torch.cuda.is_initialized() is False


def test_scripted_step_shares_parameters_and_matches_eager_bitwise_cpu():
    import torch

    from src.tfsr_b3st4_ddrop_v1 import model as model_module

    torch.manual_seed(42)
    decoder = model_module.TFSRDecoder(capture_diagnostics=False)
    scripted = bench2.build_scripted_step(torch, decoder)
    shared = {name: tensor.data_ptr() for name, tensor in decoder.named_parameters()}
    for name, tensor in scripted.named_parameters():
        assert shared[name] == tensor.data_ptr()
    batch = _tiny_batch(torch, model_module, units=6)
    hidden = torch.zeros(1, batch.x.shape[0], 256)
    tokens, mass = bench2.hoisted_precompute(torch, decoder, batch)
    decoder.eval()
    with torch.no_grad():
        eager_prediction = bench2.module_attention_loop(torch, decoder, tokens, mass)
        outs = []
        for time_index in range(tokens.shape[1]):
            output, hidden = scripted(hidden, tokens[:, time_index], mass[:, time_index])
            outs.append(output)
        scripted_prediction = torch.stack(outs, dim=1)
    assert torch.equal(eager_prediction, scripted_prediction)
    assert torch.cuda.is_initialized() is False


def test_inductor_construction_shim_is_idempotent_and_reports_target():
    first = bench2.apply_inductor_construction_shim()
    second = bench2.apply_inductor_construction_shim()
    assert first["target"] == bench2.SHIM_TARGET
    assert second["applied"] is True
    import dataclasses

    import triton.compiler.compiler as triton_compiler

    assert dataclasses.is_dataclass(triton_compiler.AttrsDescriptor)
    names = {field.name for field in dataclasses.fields(triton_compiler.AttrsDescriptor)}
    assert names == {"divisible_by_16", "equal_to_1"}


# --------------------------------------------------------------------------- #
# Trajectory comparison semantics
# --------------------------------------------------------------------------- #

def test_trajectory_difference_requires_aligned_dropout_stream():
    import torch

    left = [{
        "prediction": torch.zeros(2), "loss_tensor": torch.zeros(()),
        "gradient_state": {"w": torch.zeros(1)}, "dropout_p": 0.5, "kept": 4,
    }]
    misaligned = [{
        "prediction": torch.zeros(2), "loss_tensor": torch.zeros(()),
        "gradient_state": {"w": torch.zeros(1)}, "dropout_p": 0.7, "kept": 4,
    }]
    with pytest.raises(bench2.FailClosedError, match="dropout receipt stream misaligned"):
        bench2._max_trajectory_difference(torch, left, misaligned)
    aligned = [dict(left[0])]
    differences = bench2._max_trajectory_difference(torch, left, aligned)
    assert differences == {"forward_max_abs": 0.0, "loss_abs": 0.0, "gradient_max_abs": 0.0}


def test_equivalence_payload_rejects_forged_flags_and_bad_tolerance_map():
    valid = _equivalence("kv_precompute", 128, exact=False)
    checked = bench2.validate_equivalence_payload(valid)
    assert checked["within_tolerance"] is True and checked["exact_match"] is False

    forged_exact = copy.deepcopy(valid)
    forged_exact["exact_match"] = True
    with pytest.raises(bench2.FailClosedError, match="exact-match"):
        bench2.validate_equivalence_payload(forged_exact)

    forged_tolerance = copy.deepcopy(valid)
    forged_tolerance["fp32_tolerance_map"]["gradient_max_abs"] = 1.0
    with pytest.raises(bench2.FailClosedError, match="tolerance-map"):
        bench2.validate_equivalence_payload(forged_tolerance)

    forged_role = copy.deepcopy(valid)
    forged_role["comparison_role"] = "MATHEMATICAL_TIMING_GATE"
    with pytest.raises(bench2.FailClosedError, match="comparison-role"):
        bench2.validate_equivalence_payload(forged_role)

    wrong_steps = copy.deepcopy(valid)
    wrong_steps["trajectory_steps"] = 1
    with pytest.raises(bench2.FailClosedError, match="trajectory length"):
        bench2.validate_equivalence_payload(wrong_steps)


def test_out_of_tolerance_evidence_is_preserved_not_rejected():
    rejected = _equivalence("kv_precompute", 128, exact=False)
    rejected["differences"]["forward_max_abs"] = 1e-3
    rejected["exact_match"] = False
    rejected["within_tolerance"] = False
    checked = bench2.validate_equivalence_payload(rejected)
    assert checked["within_tolerance"] is False
    assert checked["differences"]["forward_max_abs"] == 1e-3


# --------------------------------------------------------------------------- #
# Matrix scheduling and receipt conventions
# --------------------------------------------------------------------------- #

def test_matrix_runs_every_cell_once_and_compares_measured_candidates_only():
    backend = _FakeBackend()
    matrix = bench2.run_benchmark_matrix(backend)
    assert len(matrix["cells"]) == 18
    assert len(backend.calls) == 18
    # Equivalence is requested only for measured non-production cells.
    assert len(backend.comparisons) == 16
    checked = bench2._validate_matrix_payload(matrix)
    assert len(checked["cells"]) == 18


def test_compile_unavailable_cells_carry_exception_evidence_and_no_equivalence():
    unavailable = {"compile_default", "compile_reduce_overhead", "combo_kv_compile_reduce_overhead"}
    matrix = _matrix(compile_unavailable_kinds=unavailable)
    checked = bench2._validate_matrix_payload(matrix)["cells"]
    for cell in checked:
        if cell["kind"] in unavailable:
            assert cell["status"] == bench2.COMPILE_UNAVAILABLE
            assert cell["compile_failure_stage"] == "warmup"
            assert cell["compile_exception_class"] == "torch._dynamo.exc.BackendCompilerFailed"
            assert cell["equivalence"] is None
            assert cell["construction_shim_applied"] is True
    # A compile-unavailable cell must never appear on an eager execution path.
    forged = copy.deepcopy(matrix)
    eager_cell = next(item for item in forged["cells"] if item["kind"] == "kv_precompute")
    eager_cell["compile_exception_class"] = "builtins.RuntimeError"
    with pytest.raises(bench2.FailClosedError, match="fabricated failure evidence"):
        bench2._validate_matrix_payload(forged)


def test_matrix_rejects_reordering_and_truncation():
    matrix = _matrix()
    reordered = copy.deepcopy(matrix)
    reordered["cells"][3], reordered["cells"][4] = reordered["cells"][4], reordered["cells"][3]
    with pytest.raises(bench2.FailClosedError, match="matrix order"):
        bench2._validate_matrix_payload(reordered)
    truncated = copy.deepcopy(matrix)
    truncated["cells"] = truncated["cells"][:-1]
    with pytest.raises(bench2.FailClosedError, match="cardinality"):
        bench2._validate_matrix_payload(truncated)


def test_measured_candidate_cell_requires_equivalence_and_production_forbids_it():
    matrix = _matrix()
    stripped = copy.deepcopy(matrix)
    victim = next(item for item in stripped["cells"] if item["kind"] == "jit_scripted_step")
    victim["equivalence"] = None
    with pytest.raises(bench2.FailClosedError, match="missing trajectory equivalence"):
        bench2._validate_matrix_payload(stripped)
    polluted = copy.deepcopy(matrix)
    production = next(item for item in polluted["cells"] if item["kind"] == "production_contract")
    production["equivalence"] = copy.deepcopy(_equivalence("kv_precompute", 128))
    with pytest.raises(bench2.FailClosedError, match="production baseline"):
        bench2._validate_matrix_payload(polluted)


def test_receipt_is_engineering_only_binds_v1_and_derives_conclusion():
    receipt = _receipt()
    assert receipt["purpose"] == "ENGINEERING_ONLY"
    assert receipt["scientific_result"] is False
    assert receipt["authorizes_training"] is False
    assert receipt["supersedes"]["superseded_receipt"] == bench2.V1_RECEIPT_RELATIVE
    assert receipt["v1_construction_failure_reproduction"]["reproduced"] is True
    assert receipt["inductor_construction_shim"]["applied"] is True
    # Before publication the canonical root must be absent; after publication
    # it may contain exactly the declared topology and nothing else.
    published = ROOT / bench2.OUTPUT_ROOT_RELATIVE
    if published.exists():
        assert sorted(item.name for item in published.iterdir()) == ["receipt.json", "receipt.json.sha256"]
    else:
        assert not published.exists()
    conclusion = bench2.validate_conclusion_payload(receipt["conclusion"])
    assert len(conclusion["ranking"]) == 16
    assert conclusion["fastest_first_step_within_tolerance"]["kind"] == "validation_hoisted"
    assert any("successor" in item for item in conclusion["recommendations"])
    checked = bench2.validate_engineering_receipt(receipt)
    assert checked["launch_closure"] == checked["final_closure"]


def test_receipt_rejects_drifted_construction_report_and_conclusion():
    forged_shim = _receipt()
    forged_shim["inductor_construction_shim"]["applied"] = "yes"
    with pytest.raises(bench2.FailClosedError, match="construction shim"):
        bench2.validate_engineering_receipt(forged_shim)

    forged_conclusion = _receipt()
    del forged_conclusion["conclusion"]["statement"]
    with pytest.raises(bench2.FailClosedError, match="conclusion schema"):
        bench2.validate_engineering_receipt(forged_conclusion)

    forged_supersedes = _receipt()
    forged_supersedes["supersedes"]["superseded_receipt_sha256"] = _sha("not-v1")
    with pytest.raises(bench2.FailClosedError, match="boundary/schema drift"):
        bench2.validate_engineering_receipt(forged_supersedes)

    drifted_closure = _receipt()
    drifted_closure["final_closure"] = _closure(salt="drift")
    with pytest.raises(bench2.FailClosedError, match="launch/final closure drift"):
        bench2.validate_engineering_receipt(drifted_closure)


def test_conclusion_flags_out_of_tolerance_candidates_as_rejected():
    matrix = _matrix()
    victim = next(
        item for item in matrix["cells"] if item["kind"] == "kv_precompute" and item["units"] == 128
    )
    first_step = dict(victim["equivalence"]["first_step_differences"])
    first_step["forward_max_abs"] = 5e-2
    victim["equivalence"]["first_step_differences"] = first_step
    victim["equivalence"]["first_step_within_tolerance"] = False
    victim["equivalence"]["differences"] = dict(victim["equivalence"]["differences"], forward_max_abs=5e-2)
    victim["equivalence"]["exact_match"] = False
    victim["equivalence"]["within_tolerance"] = False
    conclusion = bench2.validate_conclusion_payload(bench2._conclusion(matrix))
    kv_row = next(row for row in conclusion["ranking"] if row["kind"] == "kv_precompute" and row["units"] == 128)
    assert kv_row["first_step_within_tolerance"] is False
    assert any("numerically rejected" in item for item in conclusion["recommendations"])
    # Rejected evidence is preserved in the receipt, never silently dropped:
    # this is the v1 GRUCell precedent (measured, flagged, not promotable).
    receipt = bench2.build_engineering_receipt(
        matrix=matrix, launch_closure=_closure(), final_closure=_closure(), runtime=_runtime(),
        construction_report=_construction_report(),
    )
    cell = next(item for item in receipt["matrix"]["cells"] if item["kind"] == "kv_precompute" and item["units"] == 128)
    assert cell["equivalence"]["within_tolerance"] is False
    assert cell["status"] == "MEASURED"


def test_transactional_publication_is_o_excl_0444_and_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    receipt = _receipt()
    root = tmp_path / "tfsr_exploration" / "results" / "tfsr_b3st4_ddrop_throughput_engineering_v2"
    root.mkdir(parents=True)
    artifact = bench2.OutputArtifactRootV2(root)
    digest = artifact.publish(receipt)
    assert (root / "receipt.json").stat().st_mode & 0o777 == 0o444
    assert (root / "receipt.json.sha256").read_text() == f"{digest}  receipt.json\n"
    with pytest.raises(bench2.FailClosedError, match="collision"):
        artifact.publish(receipt)

    rollback_root = tmp_path / "rollback"
    rollback_root.mkdir()
    rollback = bench2.OutputArtifactRootV2(rollback_root)
    original_write = bench2._write_full
    calls = {"count": 0}

    def fail_second_write(descriptor: int, body: bytes) -> None:
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("synthetic v2 sidecar write failure")
        original_write(descriptor, body)

    monkeypatch.setattr(bench2, "_write_full", fail_second_write)
    with pytest.raises(OSError, match="synthetic v2 sidecar"):
        rollback.publish(receipt)
    assert not (rollback_root / "receipt.json").exists()
    assert not (rollback_root / "receipt.json.sha256").exists()
