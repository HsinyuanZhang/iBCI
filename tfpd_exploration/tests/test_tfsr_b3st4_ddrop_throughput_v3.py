"""Synthetic CPU/adversarial tests for the v3 engineering-only throughput audit."""
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
from src.tfsr_b3st4_ddrop_v1 import throughput_benchmark_v3 as bench3  # noqa: E402


def _sha(label: str) -> str:
    return bench3._sha(label.encode("ascii"))


def _closure(*, salt: str = "") -> dict[str, object]:
    hashes = {path: _sha("v3-closure:" + salt + path) for path in bench3.BENCHMARK_CLOSURE}
    hashes[bench3.WORKORDER_RELATIVE] = bench3.WORKORDER_SHA256
    hashes[bench3.MODEL_RELATIVE] = bench3.MODEL_SHA256
    hashes[bench3.NATIVE_ACTIVITY_RELATIVE] = bench3.NATIVE_ACTIVITY_SHA256
    hashes[bench3.LIVE_THROUGHPUT_RELATIVE] = bench3.LIVE_THROUGHPUT_SHA256
    hashes[bench3.V1_RECEIPT_RELATIVE] = bench3.V1_RECEIPT_SHA256
    hashes[bench3.V2_RECEIPT_RELATIVE] = bench3.V2_RECEIPT_SHA256
    return {
        "paths": list(bench3.BENCHMARK_CLOSURE),
        "sha256_by_path": hashes,
        "closure_sha256": bench3._closure_digest(hashes),
    }


def _runtime(triton: str | None = "3.2.0") -> dict[str, object]:
    return {
        "gpu": dict(bench3.FROZEN_GPU0), "torch_version": "synthetic-torch",
        "cuda_version": "synthetic-cuda", "cudnn_version": "synthetic-cudnn",
        "triton_version": triton, "cpu_threads": {"intraop": 1, "interop": 1},
    }


def _equivalence(kind: str, units: int, *, exact: bool = True) -> dict[str, object]:
    differences = {
        "forward_max_abs": 0.0,
        "loss_abs": 0.0,
        "gradient_max_abs": 0.0 if exact else 9e-7,
        "model_state_max_abs": 0.0 if exact else 1.5e-5,
        "optimizer_state_max_abs": 0.0 if exact else 9e-8,
    }
    first_step = {key: differences[key] for key in ("forward_max_abs", "loss_abs", "gradient_max_abs")}
    return {
        "label": f"production_contract_vs_{kind}_b32_u{units}_20step",
        "same_initial_model_state": True,
        "same_initial_fresh_adam_state": True,
        "trajectory_steps": 20,
        "differences": differences,
        "first_step_differences": first_step,
        "first_step_within_tolerance": all(
            first_step[key] <= bench3.V3_TOLERANCES[key] for key in first_step
        ),
        "all_finite": {
            "baseline_model": True, "candidate_model": True,
            "baseline_optimizer": True, "candidate_optimizer": True,
        },
        "exact_match": all(value == 0.0 for value in differences.values()),
        "fp32_tolerance_map": dict(bench3.V3_TOLERANCES),
        "within_tolerance": all(
            differences[key] <= bench3.V3_TOLERANCES[key] for key in bench3.V3_TOLERANCES
        ),
        "comparison_role": bench3.DIAGNOSTIC_NON_AUTHORIZING,
        "gru_tensor_copy_map": None,
    }


class _FakeBackend:
    """No-GPU matrix backend with auditable measured/unavailable behavior."""

    reference_model_state = _sha("v3-reference-model")
    reference_optimizer_state = _sha("v3-reference-adam")

    def __init__(self, *, unavailable: set[str] | None = None) -> None:
        self.unavailable = set() if unavailable is None else set(unavailable)
        self.calls: list[str] = []
        self.comparisons: list[str] = []
        import torch
        self.torch = torch

    def compare_before_timing(self, *, kind: str, units: int) -> Mapping[str, Any]:
        self.comparisons.append(f"{kind}@{units}")
        return copy.deepcopy(_equivalence(kind, units))

    def _clean_cuda_after_recorded_error(self) -> None:  # pragma: no cover - CPU fake
        return None

    def measure(self, *, label: str, kind: str, execution: str, environment: str, units: int) -> Mapping[str, Any]:
        self.calls.append(label)
        details = None
        if kind == "cuda_graphed_core":
            details = {
                "graph_capture": "torch.cuda.graphs.make_graphed_callables",
                "num_warmup_iters": 3, "scope": "50-step recurrent core",
                "attention": "F.multi_head_attention_forward with the frozen weights",
                "gru": "torch._VF.gru with the frozen tensors in one flat buffer",
            }
        elif kind in bench3.VENV_KINDS:
            details = {
                "environment": "triton31_venv", "triton_version": bench3.TRITON_PIN,
                "compiler": "inductor default" if kind == "venv_compile_default" else "inductor reduce-overhead",
                "construction_shim_applied": False,
            }
        common: dict[str, Any] = {
            "schema": bench3.CELL_SCHEMA, "label": label, "kind": kind, "execution": execution,
            "environment": environment, "batch_size": 32, "units": units, "dtype": "float32",
            "warmup_steps": 5, "measured_steps": 20, "typed_normalized_t4": True,
            "initial_model_state_sha256": self.reference_model_state,
            "initial_optimizer_state_sha256": self.reference_optimizer_state,
            "state_before_measure_sha256": _sha("v3-after-warmup:" + label),
            "optimizer_before_measure_sha256": _sha("v3-after-warmup-adam:" + label),
            "fresh_model_from_reference": True, "fresh_optimizer_state": True, "reused_warmed_state": False,
            "candidate_details": details, "runtime": _runtime(), "equivalence": None,
        }
        if label in self.unavailable:
            return {
                **common, "status": bench3.IMPLEMENTATION_UNAVAILABLE, "cuda_oom": False,
                "oom_error_sha256": None,
                "implementation_exception_class": "torch._dynamo.exc.BackendCompilerFailed",
                "implementation_exception_repr_sha256": _sha("v3-unavailable:" + label),
                "implementation_failure_stage": "warmup", "post_error_cuda_cleanup_reset": True,
                "finite": None, "post_measurement_finite": None, "timing": None,
                "resources": {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0, "rss_bytes": 1,
                              "oom_cleanup_reset": False},
            }
        rate = 4.0 if kind == "production_contract" else 6.0
        return {
            **common, "status": "MEASURED", "cuda_oom": False, "oom_error_sha256": None,
            "implementation_exception_class": None, "implementation_exception_repr_sha256": None,
            "implementation_failure_stage": None, "post_error_cuda_cleanup_reset": False,
            "finite": {"forward": True, "loss": True, "gradient": True},
            "post_measurement_finite": {"model": True, "optimizer": True},
            "timing": {
                "warmup_wall_seconds": 0.1, "measured_total_wall_seconds": 2.0,
                "median_step_wall_seconds": 1.0 / rate, "steps_per_second": rate,
                "samples_per_second": rate * 32,
                "projected_48epoch_seconds": bench3.TRAINING_STEPS_48_EPOCHS / rate,
                "projection_label": bench3.ENGINEERING_PROJECTION_LABEL,
            },
            "resources": {"peak_allocated_bytes": 2, "peak_reserved_bytes": 3, "rss_bytes": 4,
                          "oom_cleanup_reset": False},
        }


def _cells(**kwargs: Any) -> list[dict[str, object]]:
    """Merge base and venv fake-backend cells in the exact planned order."""
    backend = _FakeBackend(**kwargs)
    base = bench3.run_base_matrix(backend)
    venv = bench3.run_venv_matrix(backend)
    by_label = {cell["label"]: cell for cell in list(base) + list(venv)}
    return [dict(by_label[str(plan["label"])]) for plan in bench3.planned_matrix()]


def _venv_report() -> dict[str, object]:
    return {
        "venv_python": "/tmp/tfsr_triton_venv/bin/python", "created": True, "reused": False,
        "triton_pin": bench3.TRITON_PIN, "torch_version": "2.5.1.post303",
        "triton_version": bench3.TRITON_PIN, "failure": None,
    }


def _receipt(cells: list[dict[str, object]] | None = None) -> dict[str, object]:
    closure = _closure()
    return bench3.build_engineering_receipt(
        cells=cells if cells is not None else _cells(),
        launch_closure=closure, final_closure=closure, runtime=_runtime(),
        venv_report=_venv_report(),
    )


# --------------------------------------------------------------------------- #
# Planned matrix, dry plan, and CLI conventions
# --------------------------------------------------------------------------- #

def test_planned_matrix_is_explicit_ordered_and_covers_both_unit_sweeps():
    matrix = bench3.planned_matrix()
    labels = [item["label"] for item in matrix]
    assert len(matrix) == 10
    assert labels[0] == "production_contract_b32_u128"
    assert labels[2] == "cuda_graphed_core_b32_u128"
    assert labels[4] == "venv_compile_reduce_overhead_b32_u128"
    assert labels[5] == "production_contract_b32_u64"
    assert labels[-1] == "venv_compile_reduce_overhead_b32_u64"
    assert all(item["units"] in (128, 64) for item in matrix)
    assert all("*" not in item["label"] for item in matrix)


def test_dry_plan_is_static_and_binds_superseded_v2_receipt():
    plan = bench3.dry_plan()
    assert plan["status"] == "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_BENCHMARK"
    assert plan["supersedes"]["superseded_receipt_sha256"] == bench3.V2_RECEIPT_SHA256
    assert plan["closure"]["non_globbed"] is True
    assert sorted(plan["execution_flags_required_together"]) == [
        "--execute", "--i-have-root-throughput-authorization-v3",
    ]
    assert plan["boundaries"]["authorizes_training"] is False
    assert plan["levers"]["venv_compile"]["triton_pin"] == bench3.TRITON_PIN


def test_closure_rejects_drift_on_every_bound_authority():
    assert bench3.validate_closure_payload(_closure())["closure_sha256"] == _closure()["closure_sha256"]
    for relative, frozen in (
        (bench3.MODEL_RELATIVE, bench3.MODEL_SHA256),
        (bench3.NATIVE_ACTIVITY_RELATIVE, bench3.NATIVE_ACTIVITY_SHA256),
        (bench3.WORKORDER_RELATIVE, bench3.WORKORDER_SHA256),
        (bench3.LIVE_THROUGHPUT_RELATIVE, bench3.LIVE_THROUGHPUT_SHA256),
        (bench3.V1_RECEIPT_RELATIVE, bench3.V1_RECEIPT_SHA256),
        (bench3.V2_RECEIPT_RELATIVE, bench3.V2_RECEIPT_SHA256),
    ):
        forged = _closure()
        forged["sha256_by_path"][relative] = _sha("drift")
        forged["closure_sha256"] = bench3._closure_digest(forged["sha256_by_path"])
        with pytest.raises(bench3.FailClosedError):
            bench3.validate_closure_payload(forged)


def test_zero_argument_cli_is_static_dry_and_never_imports_torch(tmp_path: Path):
    script = ROOT / "tfpd_exploration/scripts/benchmark_tfsr_b3st4_ddrop_throughput_v3.py"
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
    assert plan["cell"] == bench3.CELL
    partial = subprocess.run([sys.executable, str(script), "--execute"], text=True, capture_output=True, env=env)
    assert partial.returncode != 0
    assert "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_BENCHMARK" in partial.stderr
    assert "TORCH_IMPORT_FORBIDDEN" not in partial.stderr


# --------------------------------------------------------------------------- #
# The graphed-core math laws (tiny exact CPU fixtures)
# --------------------------------------------------------------------------- #

def test_mha_function_is_the_module_path_bitwise_cpu():
    """F.multi_head_attention_forward with the frozen weights is the module op."""
    import torch

    from src.tfsr_b3st4_ddrop_v1 import model as model_module

    torch.manual_seed(3)
    decoder = model_module.TFSRDecoder(capture_diagnostics=False)
    decoder.eval()
    units, batch = 6, 2
    tokens = torch.randn(batch, units, 256)
    queries = torch.randn(batch, 2, 256)
    with torch.no_grad():
        module_read, _ = decoder.attn(queries, tokens, tokens, need_weights=False)
        core = bench3.build_core_function(torch)
        import torch.nn.functional as F
        fn_read, _ = F.multi_head_attention_forward(
            queries.transpose(1, 0), tokens.transpose(1, 0), tokens.transpose(1, 0),
            256, 4, decoder.attn.in_proj_weight, decoder.attn.in_proj_bias,
            None, None, False, 0.0, decoder.attn.out_proj.weight, decoder.attn.out_proj.bias,
            training=False, need_weights=False,
        )
        fn_read = fn_read.transpose(1, 0)
    assert torch.equal(module_read, fn_read)
    assert torch.cuda.is_initialized() is False


def test_gru_flat_buffer_roundtrip_and_gradient_split():
    """The flat GRU buffer carries the four frozen tensors and splits grads."""
    import torch

    from src.tfsr_b3st4_ddrop_v1 import model as model_module

    torch.manual_seed(5)
    decoder = model_module.TFSRDecoder(capture_diagnostics=False)
    flat = bench3.gru_flat_buffer(torch, decoder.gru)
    assert flat.shape == (3 * 256 * 515 + 3 * 256 * 256 + 2 * 3 * 256,)
    w_ih = flat[:3 * 256 * 515].view(3 * 256, 515)
    w_hh = flat[3 * 256 * 515:3 * 256 * 515 + 3 * 256 * 256].view(3 * 256, 256)
    b_ih = flat[3 * 256 * 515 + 3 * 256 * 256:3 * 256 * 515 + 3 * 256 * 256 + 3 * 256]
    b_hh = flat[3 * 256 * 515 + 3 * 256 * 256 + 3 * 256:]
    assert torch.equal(w_ih, decoder.gru.weight_ih_l0)
    assert torch.equal(w_hh, decoder.gru.weight_hh_l0)
    assert torch.equal(b_ih, decoder.gru.bias_ih_l0)
    assert torch.equal(b_hh, decoder.gru.bias_hh_l0)
    flat.sum().double().backward()
    for name in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
        tensor = getattr(decoder.gru, name)
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all().item()
    assert torch.cuda.is_initialized() is False


@pytest.mark.parametrize("units", (4, 3))
def test_core_function_matches_frozen_module_loop_forward_cpu(units: int):
    """The graphed core's math equals the frozen module loop (train mode too)."""
    import random

    import torch

    from src.tfsr_b3st4_ddrop_v1 import model as model_module
    from src.tfsr_b3st4_ddrop_v1 import throughput_benchmark_v2 as bench2

    torch.manual_seed(42)
    decoder = model_module.TFSRDecoder(capture_diagnostics=False).train(True)
    batch, window = 2, 5
    generator = torch.Generator().manual_seed(11)
    tokens = (torch.randn((batch, window, units, 256), generator=generator) * 0.5).requires_grad_(True)
    mass = (torch.randn((batch, window, 3), generator=generator) * 0.1).requires_grad_(True)
    core = bench3.build_core_function(torch)
    flat = bench3.gru_flat_buffer(torch, decoder.gru)
    candidate = core(tokens, mass, *bench3.core_parameter_tensors(decoder), flat)
    reference = bench2.module_attention_loop(torch, decoder, tokens, mass)
    # Same op sequence on CPU: equal within the fp32 GEMM blocking band.
    assert (candidate - reference).abs().max().item() <= 2e-5

    # The returned graph-input surface differentiates every core parameter.
    params = list(bench3.core_parameter_tensors(decoder)) + [
        decoder.gru.weight_ih_l0, decoder.gru.weight_hh_l0,
        decoder.gru.bias_ih_l0, decoder.gru.bias_hh_l0,
    ]
    flat2 = bench3.gru_flat_buffer(torch, decoder.gru)
    out = core(tokens, mass, *bench3.core_parameter_tensors(decoder), flat2)
    loss = (out * torch.randn_like(out)).sum()
    grads = torch.autograd.grad(loss, [tokens, mass] + params, allow_unused=False)
    assert all(grad is not None and torch.isfinite(grad).all().item() for grad in grads)
    assert torch.cuda.is_initialized() is False


def test_core_function_rejects_wrong_shapes():
    import torch

    torch.manual_seed(0)
    core = bench3.build_core_function(torch)
    tokens = torch.randn(2, 4, 3, 128)  # wrong embed width
    mass = torch.randn(2, 4, 3)
    # 18 core parameters follow the two tensor inputs (the flat GRU buffer is last).
    dummy = [torch.randn(2, 2)] * 18
    with pytest.raises(ValueError):
        core(tokens, mass, *dummy)
    with pytest.raises(ValueError):
        core(torch.randn(2, 4, 3, 256), torch.randn(2, 5, 3), *dummy)


# --------------------------------------------------------------------------- #
# Cell schema, matrix, and unavailable semantics
# --------------------------------------------------------------------------- #

def test_matrix_validates_order_and_requires_equivalence_for_measured_candidates():
    cells = _cells()
    checked = bench3._validate_matrix_cells(cells)
    assert len(checked) == 10
    stripped = copy.deepcopy(cells)
    victim = next(item for item in stripped if item["kind"] == "cuda_graphed_core")
    victim["equivalence"] = None
    with pytest.raises(bench3.FailClosedError, match="missing trajectory equivalence"):
        bench3._validate_matrix_cells(stripped)
    polluted = copy.deepcopy(cells)
    production = next(item for item in polluted if item["kind"] == "production_contract")
    production["equivalence"] = copy.deepcopy(_equivalence("cuda_graphed_core", 128))
    with pytest.raises(bench3.FailClosedError, match="production baseline"):
        bench3._validate_matrix_cells(polluted)


def test_matrix_rejects_reordering_and_truncation():
    cells = _cells()
    reordered = copy.deepcopy(cells)
    reordered[1], reordered[2] = reordered[2], reordered[1]
    with pytest.raises(bench3.FailClosedError, match="matrix order"):
        bench3._validate_matrix_cells(reordered)
    truncated = copy.deepcopy(cells)
    truncated = truncated[:-1]
    with pytest.raises(bench3.FailClosedError, match="cardinality"):
        bench3._validate_matrix_cells(truncated)


def test_unavailable_cells_carry_exception_evidence_and_no_equivalence():
    unavailable = {"venv_compile_default_b32_u128", "cuda_graphed_core_b32_u64"}
    cells = _cells(unavailable=unavailable)
    checked = bench3._validate_matrix_cells(cells)
    for cell in checked:
        if cell["label"] in unavailable:
            assert cell["status"] == bench3.IMPLEMENTATION_UNAVAILABLE
            assert cell["implementation_failure_stage"] == "warmup"
            assert cell["equivalence"] is None
            assert cell["timing"] is None
            assert cell["fresh_model_from_reference"] is True
    # An unavailable cell on an eager path with a fabricated exception is
    # still admissible evidence here, but a MEASURED cell is not.
    forged = copy.deepcopy(cells)
    victim = next(item for item in forged if item["kind"] == "jit_scripted_step")
    victim["implementation_exception_class"] = "builtins.RuntimeError"
    with pytest.raises(bench3.FailClosedError, match="fabricated failure evidence"):
        bench3._validate_matrix_cells(forged)


def test_never_constructed_unavailable_cell_allows_null_lineage_but_not_half_lineage():
    cells = _cells()
    victim = next(item for item in cells if item["kind"] == "venv_compile_default")
    never = copy.deepcopy(victim)
    never.update({
        "status": bench3.IMPLEMENTATION_UNAVAILABLE, "cuda_oom": False, "oom_error_sha256": None,
        "implementation_exception_class": "venv.VenvCellProcessFailure",
        "implementation_exception_repr_sha256": _sha("venv-failure"),
        "implementation_failure_stage": "variant_construction",
        "post_error_cuda_cleanup_reset": True,
        "initial_model_state_sha256": None, "initial_optimizer_state_sha256": None,
        "fresh_model_from_reference": False, "fresh_optimizer_state": False,
        "finite": None, "post_measurement_finite": None, "timing": None, "equivalence": None,
        "runtime": None,
        "state_before_measure_sha256": _sha("marker"),
        "optimizer_before_measure_sha256": _sha("marker-adam"),
    })
    checked = bench3.validate_cell_payload(
        never, label=str(never["label"]), kind="venv_compile_default", execution="compiled",
        environment="triton31_venv", units=int(never["units"]),
    )
    assert checked["status"] == bench3.IMPLEMENTATION_UNAVAILABLE
    half = copy.deepcopy(never)
    half["runtime"] = _runtime()  # runtime present without constructed lineage
    with pytest.raises(bench3.FailClosedError, match="lineage/runtime"):
        bench3.validate_cell_payload(
            half, label=str(half["label"]), kind="venv_compile_default", execution="compiled",
            environment="triton31_venv", units=int(half["units"]),
        )


def test_measured_venv_cell_requires_venv_environment_evidence():
    cells = _cells()
    victim = next(item for item in cells if item["kind"] == "venv_compile_default")
    stripped = copy.deepcopy(victim)
    stripped["candidate_details"] = None
    with pytest.raises(bench3.FailClosedError, match="venv cell environment"):
        bench3.validate_cell_payload(
            stripped, label=str(stripped["label"]), kind="venv_compile_default",
            execution="compiled", environment="triton31_venv", units=int(stripped["units"]),
            reference_model_sha256=_FakeBackend.reference_model_state,
            reference_optimizer_sha256=_FakeBackend.reference_optimizer_state,
        )


def test_runtime_contract_requires_triton_field():
    good = _runtime("3.1.0")
    assert bench3._validate_runtime(good)["triton_version"] == "3.1.0"
    with pytest.raises(bench3.FailClosedError):
        bench3._validate_runtime({key: value for key, value in _runtime().items() if key != "triton_version"})


# --------------------------------------------------------------------------- #
# Venv report and receipt conventions
# --------------------------------------------------------------------------- #

def test_venv_report_schema_rejects_drift():
    checked = bench3._validate_venv_report(_venv_report())
    assert checked["triton_pin"] == bench3.TRITON_PIN
    forged = _venv_report()
    forged["triton_pin"] = "3.2.0"
    with pytest.raises(bench3.FailClosedError):
        bench3._validate_venv_report(forged)
    missing = _venv_report()
    del missing["failure"]
    with pytest.raises(bench3.FailClosedError):
        bench3._validate_venv_report(missing)


def test_receipt_is_engineering_only_binds_v2_and_derives_conclusion():
    receipt = _receipt()
    assert receipt["purpose"] == "ENGINEERING_ONLY"
    assert receipt["scientific_result"] is False
    assert receipt["authorizes_training"] is False
    assert receipt["supersedes"]["superseded_receipt"] == bench3.V2_RECEIPT_RELATIVE
    assert receipt["triton_venv"]["triton_version"] == bench3.TRITON_PIN
    published = ROOT / bench3.OUTPUT_ROOT_RELATIVE
    if published.exists():
        assert sorted(item.name for item in published.iterdir()) == ["receipt.json", "receipt.json.sha256"]
    else:
        assert not published.exists()
    conclusion = bench3.validate_conclusion_payload(receipt["conclusion"])
    assert len(conclusion["ranking"]) == 8
    assert conclusion["fastest_first_step_within_tolerance"]["kind"] != "production_contract"
    assert any("successor" in item for item in conclusion["recommendations"])
    checked = bench3.validate_engineering_receipt(receipt)
    assert checked["launch_closure"] == checked["final_closure"]


def test_receipt_rejects_drifted_venv_conclusion_and_closure():
    forged_venv = _receipt()
    forged_venv["triton_venv"]["created"] = "yes"
    with pytest.raises(bench3.FailClosedError, match="venv report flag"):
        bench3.validate_engineering_receipt(forged_venv)

    forged_conclusion = _receipt()
    del forged_conclusion["conclusion"]["statement"]
    with pytest.raises(bench3.FailClosedError, match="conclusion schema"):
        bench3.validate_engineering_receipt(forged_conclusion)

    drifted = _receipt()
    drifted["final_closure"] = _closure(salt="drift")
    with pytest.raises(bench3.FailClosedError, match="launch/final closure drift"):
        bench3.validate_engineering_receipt(drifted)


def test_receipt_refuses_measured_venv_cells_without_verified_venv():
    cells = _cells()
    broken_venv = _venv_report()
    broken_venv["failure"] = {"ok": False, "error": "pip unreachable"}
    closure = _closure()
    with pytest.raises(bench3.FailClosedError, match="verified triton-pinned venv"):
        bench3.build_engineering_receipt(
            cells=cells, launch_closure=closure, final_closure=closure,
            runtime=_runtime(), venv_report=broken_venv,
        )


def test_conclusion_flags_unavailable_and_rejected_cells():
    cells = _cells(unavailable={"cuda_graphed_core_b32_u128"})
    victim = next(item for item in cells if item["kind"] == "venv_compile_default" and item["units"] == 64)
    first_step = dict(victim["equivalence"]["first_step_differences"])
    first_step["forward_max_abs"] = 5e-2
    victim["equivalence"]["first_step_differences"] = first_step
    victim["equivalence"]["first_step_within_tolerance"] = False
    victim["equivalence"]["differences"] = dict(victim["equivalence"]["differences"], forward_max_abs=5e-2)
    victim["equivalence"]["exact_match"] = False
    victim["equivalence"]["within_tolerance"] = False
    conclusion = bench3.validate_conclusion_payload(bench3._conclusion(cells, _venv_report()))
    assert any("numerically rejected" in item for item in conclusion["recommendations"])
    assert any("implementation-unavailable" in item for item in conclusion["recommendations"])
    unavailable_row = next(
        row for row in conclusion["ranking"]
        if row["kind"] == "cuda_graphed_core" and row["units"] == 128
    )
    assert unavailable_row["measured"] is False
    assert unavailable_row["median_step_wall_seconds"] is None


def test_transactional_publication_is_o_excl_0444_and_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    receipt = _receipt()
    root = tmp_path / "tfsr_exploration" / "results" / "tfsr_b3st4_ddrop_throughput_engineering_v3"
    root.mkdir(parents=True)
    artifact = bench3.OutputArtifactRootV3(root)
    digest = artifact.publish(receipt)
    assert (root / "receipt.json").stat().st_mode & 0o777 == 0o444
    assert (root / "receipt.json.sha256").read_text() == f"{digest}  receipt.json\n"
    with pytest.raises(bench3.FailClosedError, match="collision"):
        artifact.publish(receipt)

    rollback_root = tmp_path / "rollback"
    rollback_root.mkdir()
    rollback = bench3.OutputArtifactRootV3(rollback_root)
    original_write = bench3._write_full
    calls = {"count": 0}

    def fail_second_write(descriptor: int, body: bytes) -> None:
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("synthetic v3 sidecar write failure")
        original_write(descriptor, body)

    monkeypatch.setattr(bench3, "_write_full", fail_second_write)
    with pytest.raises(OSError, match="synthetic v3 sidecar"):
        rollback.publish(receipt)
    assert not (rollback_root / "receipt.json").exists()
    assert not (rollback_root / "receipt.json.sha256").exists()
