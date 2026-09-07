"""Fail-closed, engineering-only TF-SR throughput equivalence benchmark, v3.

This is the v3 continuation of the 2026-08-19 throughput work order.  v2
measured the pathology this module attacks: the production step is backward
CPU-dispatch bound (forward ~14 ms GPU-bound; backward ~213 ms of autograd
per-node dispatch on this torch 2.5.1.post303 build; the MultiheadAttention
module backward alone ~203 ms of dispatch against ~25 ms of kernels), compile
is dead on the installed torch 2.5.1/triton 3.2.0 pair, kv_precompute made
things worse, and ``jit_scripted_step`` (+50-56%) was the only working lever.

v3 measures two conventional levers against a freshly re-timed production
baseline and the re-timed v2 winner:

1. ``cuda_graphed_core`` -- ``torch.cuda.graphs.make_graphed_callables`` over
   the 50-step recurrent core (queries -> attention -> norm/FFN -> GRU ->
   head) with the Cell-D dropout mask applied OUTSIDE the graph (the masked
   tokens and the mass channels are graph inputs copied into static buffers
   per step).  The core function reproduces the frozen math exactly:
   ``F.multi_head_attention_forward`` with the frozen weights (measured
   bitwise 0.0 against the ``nn.MultiheadAttention`` module call on this GPU)
   and ``torch._VF.gru`` with the frozen GRU tensors passed through one flat
   buffer (the exact cuDNN op sequence ``nn.GRU.forward`` uses, avoiding the
   per-call cuDNN weight-compaction the separate-tuple form triggers).  The
   core parameters are explicit function inputs, so the captured backward
   returns their gradients and autograd accumulates into ``.grad`` normally.
2. ``venv_compile_default`` / ``venv_compile_reduce_overhead`` -- the v2
   compile cells re-run inside a THROWAWAY ``--system-site-packages`` venv
   that shadows only the triton package (triton 3.1.0, the version this torch
   build's inductor expects), reusing the base torch installation.  Nothing
   in the shared environment is modified.  The venv runs as a subprocess on
   GPU0 under the same protocol; if the arrangement fails, the receipt
   records exactly why.

Everything is engineering evidence only: synthetic tensors, GPU0, no dataset,
no checkpoint, no scientific score, no authorization to change the live run.
This module has no top-level Torch import; the public CLI loads it as a
synthetic package so a zero-argument invocation stays data-free.
"""
from __future__ import annotations

import json
import math
import os
import resource
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .throughput_benchmark import (
    DIAGNOSTIC_NON_AUTHORIZING,
    ENGINEERING_PROJECTION_LABEL,
    FailClosedError,
    FROZEN_ADAM,
    FROZEN_BASELINE,
    FROZEN_GPU0,
    LIVE_THROUGHPUT_RELATIVE,
    LIVE_THROUGHPUT_SHA256,
    LIVE_THROUGHPUT_SIDECAR_RELATIVE,
    MATHEMATICAL_FP32_TOLERANCES,
    MODEL_RELATIVE,
    MODEL_SHA256,
    NATIVE_ACTIVITY_RELATIVE,
    NATIVE_ACTIVITY_SHA256,
    TRAINING_STEPS_48_EPOCHS,
    WORKORDER_RELATIVE,
    WORKORDER_SHA256,
    _all_finite_gradients,
    _all_finite_optimizer,
    _all_finite_parameters,
    _canonical_directory_identity,
    _canonical_regular_bytes,
    _closure_digest,
    _exception_class_name,
    _finite,
    _fresh_model_from_snapshot,
    _gradient_snapshot,
    _is_sha,
    _json_bytes,
    _materialize_step_observation,
    _new_frozen_adam,
    _production_contract_step,
    _require_sha,
    _runtime_seed,
    _safe_relative,
    _sha,
    _snapshot_cpu_state,
    _validate_hoisted_inputs,
    _write_full,
    make_synthetic_batch,
    model_state_digest,
    optimizer_state_digest,
    optimizer_state_max_difference,
    require_exact_gpu0,
    validate_synthetic_batch,
)
from .throughput_benchmark import SyntheticBatch
from .throughput_benchmark_v2 import (
    UNIT_ORDER,
    UNIT_SPECS,
    V1_RECEIPT_RELATIVE,
    V1_RECEIPT_SHA256,
    build_scripted_step,
    build_trajectory_equivalence,
    hoisted_precompute,
    scripted_step_loop,
    validate_equivalence_payload,
)
from .throughput_benchmark_v2 import (
    _hoisted_dense_mse_exact,
    _max_tensor_difference,
    _max_trajectory_difference,
    _trajectory_run,
)


CELL = "TFSR_B3ST4_DDROP_THROUGHPUT_ENGINEERING_V3"
PHASE = "TFSR_THROUGHPUT_EQUIVALENCE_20260819_V3"
SOURCE_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/throughput_benchmark_v3.py"
CLI_RELATIVE = "tfpd_exploration/scripts/benchmark_tfsr_b3st4_ddrop_throughput_v3.py"
TEST_RELATIVE = "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_throughput_v3.py"
V2_SOURCE_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/throughput_benchmark_v2.py"
V2_CLI_RELATIVE = "tfpd_exploration/scripts/benchmark_tfsr_b3st4_ddrop_throughput_v2.py"
V2_TEST_RELATIVE = "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_throughput_v2.py"
V2_RECEIPT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_throughput_engineering_v2/receipt.json"
V2_RECEIPT_SHA256 = "3438b9a5c264cb497315c6e2650c46f2892a8fea3c97d3af433d7e22f10b6f57"
V1_SOURCE_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/throughput_benchmark.py"
V1_CLI_RELATIVE = "tfpd_exploration/scripts/benchmark_tfsr_b3st4_ddrop_throughput.py"
V1_TEST_RELATIVE = "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_v1_throughput_benchmark.py"
OUTPUT_ROOT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_throughput_engineering_v3"
OUTPUT_TOPOLOGY = ("receipt.json",)
PUBLIC_FLAGS = frozenset(("--execute", "--i-have-root-throughput-authorization-v3"))
VENV_FLAGS = frozenset(("--execute", "--venv-cells-only", "--i-have-root-throughput-authorization-v3"))

# The throwaway triton-pinned venv used only by the venv compile cells.  It is
# created from the SAME interpreter that runs the benchmark with
# ``--system-site-packages`` (reusing the installed torch 2.5.1.post303) and
# shadows only the triton package.  The shared environment is never modified.
TRITON_PIN = "3.1.0"
VENV_DIRECTORY = Path("/tmp/tfsr_triton_venv")

# v2 same-session medians, carried as cross-session context only.  v3 re-times
# production and jit_scripted_step in its own session; every v3 speedup is
# computed against the v3 production cell, never against these numbers.
V2_MEASURED_MEDIANS = {
    "production_contract": {128: 0.24695335747674108, 64: 0.14422188024036586},
    "jit_scripted_step": {128: 0.15797596611082554, 64: 0.09636688558384776},
}

BENCHMARK_CLOSURE = (
    WORKORDER_RELATIVE,
    SOURCE_RELATIVE,
    CLI_RELATIVE,
    TEST_RELATIVE,
    MODEL_RELATIVE,
    NATIVE_ACTIVITY_RELATIVE,
    LIVE_THROUGHPUT_RELATIVE,
    LIVE_THROUGHPUT_SIDECAR_RELATIVE,
    V1_SOURCE_RELATIVE,
    V1_CLI_RELATIVE,
    V1_TEST_RELATIVE,
    V1_RECEIPT_RELATIVE,
    V2_SOURCE_RELATIVE,
    V2_CLI_RELATIVE,
    V2_TEST_RELATIVE,
    V2_RECEIPT_RELATIVE,
)

SUPERSEDES = {
    "relation": "v3 continuation of the same engineering-only throughput audit",
    "superseded_receipt": V2_RECEIPT_RELATIVE,
    "superseded_receipt_sha256": V2_RECEIPT_SHA256,
    "note": (
        "v2 measured the per-node backward dispatch pathology (~213 ms of autograd "
        "dispatch per step on this torch build), the inductor/triton AttrsDescriptor "
        "construction failure, and found jit_scripted_step the only working lever.  v3 "
        "adds CUDA-graph capture of the recurrent core (make_graphed_callables) and the "
        "triton-pinned throwaway-venv unlock of torch.compile; no v2 measurement is "
        "retimed or replaced."
    ),
}

V3_TOLERANCES = dict(MATHEMATICAL_FP32_TOLERANCES)
BATCH_SIZE = 32
BASE_KINDS = ("production_contract", "jit_scripted_step", "cuda_graphed_core")
VENV_KINDS = ("venv_compile_default", "venv_compile_reduce_overhead")
KIND_EXECUTION = {
    "production_contract": "eager",
    "jit_scripted_step": "eager",
    "cuda_graphed_core": "eager",
    "venv_compile_default": "compiled",
    "venv_compile_reduce_overhead": "compiled",
}
KIND_ENVIRONMENT = {kind: "base" for kind in BASE_KINDS}
KIND_ENVIRONMENT.update({kind: "triton31_venv" for kind in VENV_KINDS})
IMPLEMENTATION_UNAVAILABLE = "IMPLEMENTATION_UNAVAILABLE"
IMPLEMENTATION_FAILURE_STAGES = frozenset((
    "variant_construction", "warmup", "measurement", "post_measurement_audit",
))
EMBED_DIM = 256


def planned_matrix() -> list[dict[str, object]]:
    """Return the exact ordered v3 matrix without resolving Torch or a device."""
    matrix: list[dict[str, object]] = []
    for units in UNIT_ORDER:
        suffix = f"_u{units}"
        for kind in BASE_KINDS + VENV_KINDS:
            matrix.append({
                "label": f"{kind}_b32{suffix}",
                "kind": kind,
                "execution": KIND_EXECUTION[kind],
                "environment": KIND_ENVIRONMENT[kind],
                "units": units,
            })
    return matrix


def dry_plan() -> dict[str, object]:
    """Pure static plan: no Torch import, filesystem read, CUDA call, or write."""
    return {
        "cell": CELL,
        "phase": PHASE,
        "status": "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_BENCHMARK",
        "authorization": "ROOT_AUDIT_REQUIRED",
        "execution_flags_required_together": sorted(PUBLIC_FLAGS),
        "supersedes": dict(SUPERSEDES),
        "planned_matrix": planned_matrix(),
        "fp32_tolerance_map": dict(V3_TOLERANCES),
        "levers": {
            "cuda_graphed_core": {
                "api": "torch.cuda.graphs.make_graphed_callables",
                "scope": "the 50-step recurrent core; Cell-D mask and mass assembly stay outside the graph",
                "attention": "F.multi_head_attention_forward with the frozen weights (module-exact math)",
                "gru": "torch._VF.gru with the frozen tensors passed through one flat buffer",
                "parameters": "explicit function inputs so the captured backward returns their gradients",
            },
            "venv_compile": {
                "arrangement": "throwaway --system-site-packages venv shadowing only triton",
                "triton_pin": TRITON_PIN,
                "venv_directory": str(VENV_DIRECTORY),
                "kinds": list(VENV_KINDS),
                "isolation": "the shared environment is never modified; the venv subprocess runs the same GPU0 protocol",
            },
        },
        "closure": {
            "paths": list(BENCHMARK_CLOSURE),
            "model_sha256": MODEL_SHA256,
            "native_activity_sha256": NATIVE_ACTIVITY_SHA256,
            "live_throughput_sha256": LIVE_THROUGHPUT_SHA256,
            "workorder_sha256": WORKORDER_SHA256,
            "v1_receipt_sha256": V1_RECEIPT_SHA256,
            "v2_receipt_sha256": V2_RECEIPT_SHA256,
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


# --------------------------------------------------------------------------- #
# Closure handling (descriptor-safe, identical conventions to v1/v2)
# --------------------------------------------------------------------------- #

def validate_closure_payload(value: Mapping[str, Any]) -> dict[str, object]:
    expected = {"paths", "sha256_by_path", "closure_sha256"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FailClosedError("v3 benchmark closure schema drift")
    paths, hashes = value.get("paths"), value.get("sha256_by_path")
    if paths != list(BENCHMARK_CLOSURE) or not isinstance(hashes, Mapping) or set(hashes) != set(BENCHMARK_CLOSURE):
        raise FailClosedError("v3 benchmark closure path map drift")
    if any(not _is_sha(hashes[path]) for path in BENCHMARK_CLOSURE):
        raise FailClosedError("v3 benchmark closure file digest drift")
    if value.get("closure_sha256") != _closure_digest({path: str(hashes[path]) for path in BENCHMARK_CLOSURE}):
        raise FailClosedError("v3 benchmark closure aggregate drift")
    if hashes[WORKORDER_RELATIVE] != WORKORDER_SHA256:
        raise FailClosedError("throughput work-order closure drift")
    if hashes[MODEL_RELATIVE] != MODEL_SHA256:
        raise FailClosedError("frozen TF-SR model closure drift")
    if hashes[NATIVE_ACTIVITY_RELATIVE] != NATIVE_ACTIVITY_SHA256:
        raise FailClosedError("native causal activity dependency closure drift")
    if hashes[LIVE_THROUGHPUT_RELATIVE] != LIVE_THROUGHPUT_SHA256:
        raise FailClosedError("live Phase-D throughput closure drift")
    if hashes[V1_RECEIPT_RELATIVE] != V1_RECEIPT_SHA256:
        raise FailClosedError("superseded v1 engineering receipt closure drift")
    if hashes[V2_RECEIPT_RELATIVE] != V2_RECEIPT_SHA256:
        raise FailClosedError("superseded v2 engineering receipt closure drift")
    return {
        "paths": list(BENCHMARK_CLOSURE),
        "sha256_by_path": {path: str(hashes[path]) for path in BENCHMARK_CLOSURE},
        "closure_sha256": str(value["closure_sha256"]),
    }


def compute_benchmark_closure(root: Path) -> dict[str, object]:
    """Descriptor-safely hash the exact v3 benchmark/runtime closure."""
    hashes: dict[str, str] = {}
    sealed_0444 = {
        LIVE_THROUGHPUT_RELATIVE, LIVE_THROUGHPUT_SIDECAR_RELATIVE,
        V1_RECEIPT_RELATIVE, V2_RECEIPT_RELATIVE,
    }
    for relative in BENCHMARK_CLOSURE:
        _safe_relative(relative)
        mode = 0o444 if relative in sealed_0444 else None
        hashes[relative] = _sha(_canonical_regular_bytes(root.absolute() / relative, expected_mode=mode))
    return validate_closure_payload({
        "paths": list(BENCHMARK_CLOSURE),
        "sha256_by_path": hashes,
        "closure_sha256": _closure_digest(hashes),
    })


# --------------------------------------------------------------------------- #
# Lever 1: the graphed recurrent core
# --------------------------------------------------------------------------- #

GRU_IH_ROWS, GRU_HH_ROWS, GRU_BIAS = 3 * EMBED_DIM, 3 * EMBED_DIM, 3 * EMBED_DIM
_GRU_FLAT_IH = GRU_IH_ROWS * 515
_GRU_FLAT_HH = GRU_HH_ROWS * EMBED_DIM


def gru_flat_buffer(torch: Any, gru: Any) -> Any:
    """One autograd-tracked flat buffer carrying the four frozen GRU tensors.

    Passing the GRU tensors as separate graph inputs makes cuDNN re-compact
    them on every call; one flat buffer sliced inside the graph keeps the
    single-contiguous-chunk layout ``nn.GRU.flatten_parameters`` produces, and
    the CatBackward splits the returned gradient back into the four
    parameters inside normal autograd.
    """
    return torch.cat((
        gru.weight_ih_l0.flatten(), gru.weight_hh_l0.flatten(),
        gru.bias_ih_l0.flatten(), gru.bias_hh_l0.flatten(),
    ))


def core_parameter_tensors(model: Any) -> tuple[Any, ...]:
    """The recurrent-core parameters in the graphed-function input order."""
    return (
        model.query_base,
        model.state_query.weight, model.state_query.bias,
        model.attn.in_proj_weight, model.attn.in_proj_bias,
        model.attn.out_proj.weight, model.attn.out_proj.bias,
        model.norm1.weight, model.norm1.bias, model.norm2.weight, model.norm2.bias,
        model.ffn[0].weight, model.ffn[0].bias, model.ffn[2].weight, model.ffn[2].bias,
        model.head.weight, model.head.bias,
    )


def build_core_function(torch: Any):
    """The module-exact 50-step core as a pure function of tensor arguments.

    ``F.multi_head_attention_forward`` with the frozen weights is the exact
    function ``nn.MultiheadAttention.forward`` calls (verified bitwise 0.0 on
    this GPU), and ``torch._VF.gru`` with the flat-buffer slices is the exact
    cuDNN op sequence ``nn.GRU.forward`` calls.  Every core parameter is an
    explicit input so the captured backward returns its gradient.
    """
    import torch.nn.functional as F

    def core(tokens, mass, query_base, sq_w, sq_b, in_w, in_b, ow, ob,
             n1w, n1b, n2w, n2b, f1w, f1b, f2w, f2b, hw, hb, gru_flat):
        batch, window, units, embed = tokens.shape
        if embed != EMBED_DIM:
            raise ValueError("graphed core requires the frozen 256-wide token axis")
        if tuple(mass.shape) != (batch, window, 3):
            raise ValueError("graphed core mass channel drift")
        w_ih = gru_flat[:_GRU_FLAT_IH].view(GRU_IH_ROWS, 515)
        w_hh = gru_flat[_GRU_FLAT_IH:_GRU_FLAT_IH + _GRU_FLAT_HH].view(GRU_HH_ROWS, EMBED_DIM)
        b_ih = gru_flat[_GRU_FLAT_IH + _GRU_FLAT_HH:_GRU_FLAT_IH + _GRU_FLAT_HH + GRU_BIAS]
        b_hh = gru_flat[_GRU_FLAT_IH + _GRU_FLAT_HH + GRU_BIAS:]
        hidden = tokens.new_zeros(1, batch, EMBED_DIM)
        outputs = []
        for time_index in range(window):
            queries = query_base[None] + F.linear(hidden[0], sq_w, sq_b).view(batch, 2, EMBED_DIM)
            read, _ = F.multi_head_attention_forward(
                queries.transpose(1, 0), tokens[:, time_index].transpose(1, 0),
                tokens[:, time_index].transpose(1, 0), EMBED_DIM, 4, in_w, in_b,
                None, None, False, 0.0, ow, ob, training=True, need_weights=False,
            )
            read = read.transpose(1, 0)
            read = F.layer_norm(queries + read, (EMBED_DIM,), n1w, n1b, 1e-5)
            activated = F.relu(F.linear(read, f1w, f1b))
            read = F.layer_norm(read + F.linear(activated, f2w, f2b), (EMBED_DIM,), n2w, n2b, 1e-5)
            step = torch.cat((read.flatten(1), mass[:, time_index]), dim=1).unsqueeze(1)
            _out, hidden = torch._VF.gru(
                step, hidden, (w_ih, w_hh, b_ih, b_hh), True, 1, 0.0, True, False, True,
            )
            outputs.append(F.linear(hidden[0], hw, hb))
        return torch.stack(outputs, dim=1)

    return core


def build_graphed_core(torch: Any, model: Any, *, units: int, num_warmup_iters: int = 3) -> Any:
    """Capture the module-exact recurrent core as forward+backward CUDA graphs.

    The sample args bind the static input buffers to the REAL workload shapes
    (batch 32, window 50, this unit count); a dedicated generator seeds them
    so capture never touches the runtime dropout stream.
    """
    core = build_core_function(torch)
    device = model.query_base.device
    dtype = model.query_base.dtype
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    sample = (
        (torch.randn((BATCH_SIZE, UNIT_SPECS[units].window, units, EMBED_DIM), generator=generator, device=device, dtype=dtype) * 0.5).requires_grad_(True),
        (torch.randn((BATCH_SIZE, UNIT_SPECS[units].window, 3), generator=generator, device=device, dtype=dtype) * 0.1).requires_grad_(True),
    )
    sample = sample + tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in core_parameter_tensors(model)
    )
    sample = sample + (gru_flat_buffer(torch, model.gru).detach().clone().requires_grad_(True),)
    # Each construction gets its own pool; the caller guarantees the previous
    # graphed callable and every tensor it allocated are already destroyed
    # (see _release_graphed_memory), so successive captures do not accumulate.
    return torch.cuda.graphs.make_graphed_callables(
        core, sample, num_warmup_iters=num_warmup_iters,
    )


def graphed_core_loop(torch: Any, model: Any, graphed_core: Any, tokens: Any, mass: Any) -> Any:
    """One graphed call over the recurrent core with the fresh masked tokens."""
    return graphed_core(tokens, mass, *core_parameter_tensors(model), gru_flat_buffer(torch, model.gru))


# --------------------------------------------------------------------------- #
# Lever 2: the throwaway triton-pinned venv
# --------------------------------------------------------------------------- #

def _venv_python() -> Path:
    return VENV_DIRECTORY / "bin" / "python"


def probe_triton_venv(python: Path) -> Mapping[str, object] | None:
    """Verify the venv resolves the base torch with the pinned triton."""
    if not python.is_file():
        return None
    code = (
        "import torch, triton, dataclasses\n"
        "from triton.compiler.compiler import AttrsDescriptor\n"
        f"assert triton.__version__ == '{TRITON_PIN}', triton.__version__\n"
        "assert torch.__version__ == '2.5.1.post303', torch.__version__\n"
        "assert dataclasses.is_dataclass(AttrsDescriptor)\n"
        "import torch._inductor  # noqa: F401  (construction probe, no GPU)\n"
        "print(torch.__version__, triton.__version__)\n"
    )
    try:
        completed = subprocess.run(
            [str(python), "-c", code], capture_output=True, text=True, timeout=180,
            env={**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""},
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"ok": False, "error": f"{_exception_class_name(error)}: {str(error)[:200]}"}
    if completed.returncode != 0:
        return {"ok": False, "error": completed.stderr.strip()[-400:]}
    torch_version, triton_version = completed.stdout.strip().split()
    return {"ok": True, "torch_version": torch_version, "triton_version": triton_version}


def ensure_triton_venv() -> dict[str, object]:
    """Create (or reuse) the throwaway venv shadowing only the triton package.

    The venv is built from the SAME interpreter running the benchmark with
    ``--system-site-packages``, so it reuses the installed torch wheels and
    shadows only triton.  Nothing under the shared environment is written.
    """
    python = _venv_python()
    probe = probe_triton_venv(python)
    if probe is not None and probe.get("ok") is True:
        return {
            "venv_python": str(python), "created": False, "reused": True,
            "triton_pin": TRITON_PIN, "torch_version": probe["torch_version"],
            "triton_version": probe["triton_version"], "failure": None,
        }
    failure: dict[str, object] | None = None if probe is None else dict(probe)
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(VENV_DIRECTORY)],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip()[-400:] or "venv creation failed")
        completed = subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", "--disable-pip-version-check",
             "-q", f"triton=={TRITON_PIN}"],
            capture_output=True, text=True, timeout=900,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip()[-400:] or "pip install failed")
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        message = f"{_exception_class_name(error)}: {str(error)[:400]}"
        failure = {"ok": False, "error": message}
        return {
            "venv_python": str(python), "created": False, "reused": False,
            "triton_pin": TRITON_PIN, "torch_version": None, "triton_version": None,
            "failure": failure,
        }
    probe = probe_triton_venv(python)
    if probe is None or probe.get("ok") is not True:
        return {
            "venv_python": str(python), "created": True, "reused": False,
            "triton_pin": TRITON_PIN, "torch_version": None, "triton_version": None,
            "failure": probe if probe is not None else {"ok": False, "error": "probe absent"},
        }
    return {
        "venv_python": str(python), "created": True, "reused": False,
        "triton_pin": TRITON_PIN, "torch_version": probe["torch_version"],
        "triton_version": probe["triton_version"], "failure": None,
    }


def venv_cells_available(venv_report: Mapping[str, Any]) -> bool:
    return venv_report.get("failure") is None and venv_report.get("triton_version") == TRITON_PIN


def run_venv_cell_process(root: Path, venv_report: Mapping[str, Any]) -> dict[str, object]:
    """Run the venv compile cells in the throwaway venv and parse their JSON."""
    command = [str(venv_report["venv_python"]), str(root.absolute() / CLI_RELATIVE),
               "--execute", "--venv-cells-only", "--i-have-root-throughput-authorization-v3"]
    environment = {
        **os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=2400, cwd=str(root.absolute()), env=environment,
        )
    except subprocess.TimeoutExpired as error:
        return {"ok": False, "error": f"{_exception_class_name(error)}: venv cell process exceeded 2400s"}
    if completed.returncode != 0:
        return {"ok": False, "error": completed.stderr.strip()[-2000:] or "venv process failed"}
    marker = "__V3_VENV_CELLS_JSON__"
    try:
        payload = json.loads(completed.stdout.split(marker, 1)[1].strip())
    except (IndexError, json.JSONDecodeError) as error:
        return {"ok": False, "error": f"venv stdout marker parse failure: {error}"}
    return {"ok": True, "payload": payload}


# --------------------------------------------------------------------------- #
# Candidate step (base environment)
# --------------------------------------------------------------------------- #

def _graphed_candidate_step(
    torch: Any,
    model: Any,
    optimizer: Any,
    batch: SyntheticBatch,
    *,
    graphed_core: Any,
    valid_count: int,
    capture_gradients: bool = False,
) -> dict[str, Any]:
    """One CUDA-graphed training step; Cell-D mask stays outside the graph.

    The graphed callable reuses one static output buffer, so the returned
    prediction is cloned for any later comparison before the next replay.
    """
    optimizer.zero_grad(set_to_none=True)
    model.train(True)
    tokens, mass = hoisted_precompute(torch, model, batch)
    prediction = graphed_core_loop(torch, model, graphed_core, tokens, mass)
    stored_prediction = prediction.detach().clone()
    loss = _hoisted_dense_mse_exact(torch, prediction, batch, valid_count)
    stored_loss = loss.detach().clone()
    loss.backward()
    gradients_finite = _all_finite_gradients(torch, model) if capture_gradients else None
    gradients = _gradient_snapshot(model) if capture_gradients else None
    optimizer.step()
    gain, survivor, dropout_p = model.last_unit_gain_mask, model.last_unit_survivor_mask, model.last_dropout_p
    if gain is None or survivor is None or dropout_p is None:
        raise FailClosedError("v3 graphed candidate dropout tensors missing")
    from .throughput_benchmark import _DeferredReceipt
    deferred = _DeferredReceipt(loss.detach(), dropout_p.detach(), survivor.detach(), gain.detach())
    optimizer.zero_grad(set_to_none=True)
    return {
        "prediction": stored_prediction, "loss_tensor": stored_loss, "deferred_receipt": deferred,
        "finite_loss_tensor": None, "gradient_finite": gradients_finite,
        "gradient_state": gradients, "dropout_p": float(dropout_p.detach().item()),
        "kept": int(survivor.detach().sum().item()),
    }


def _scripted_candidate_step(
    torch: Any,
    model: Any,
    optimizer: Any,
    batch: SyntheticBatch,
    *,
    scripted_step: Any,
    valid_count: int,
    capture_gradients: bool = False,
) -> dict[str, Any]:
    """The v2 jit_scripted_step candidate, re-timed in the v3 session."""
    optimizer.zero_grad(set_to_none=True)
    model.train(True)
    tokens, mass = hoisted_precompute(torch, model, batch)
    prediction = scripted_step_loop(torch, scripted_step, tokens, mass)
    loss = _hoisted_dense_mse_exact(torch, prediction, batch, valid_count)
    loss.backward()
    gradients_finite = _all_finite_gradients(torch, model) if capture_gradients else None
    gradients = _gradient_snapshot(model) if capture_gradients else None
    optimizer.step()
    gain, survivor, dropout_p = model.last_unit_gain_mask, model.last_unit_survivor_mask, model.last_dropout_p
    if gain is None or survivor is None or dropout_p is None:
        raise FailClosedError("v3 scripted candidate dropout tensors missing")
    from .throughput_benchmark import _DeferredReceipt
    deferred = _DeferredReceipt(loss.detach(), dropout_p.detach(), survivor.detach(), gain.detach())
    optimizer.zero_grad(set_to_none=True)
    return {
        "prediction": prediction.detach(), "loss_tensor": loss.detach(), "deferred_receipt": deferred,
        "finite_loss_tensor": None, "gradient_finite": gradients_finite,
        "gradient_state": gradients, "dropout_p": float(dropout_p.detach().item()),
        "kept": int(survivor.detach().sum().item()),
    }


def _venv_compiled_loop(torch: Any, model: Any, kind: str) -> Any:
    """Compile the module-exact hoisted loop with the declared compiler mode."""
    from .throughput_benchmark_v2 import module_attention_loop

    if kind == "venv_compile_default":
        def loop(tokens: Any, mass: Any) -> Any:
            return module_attention_loop(torch, model, tokens, mass)
        return torch.compile(loop)
    if kind == "venv_compile_reduce_overhead":
        def loop(tokens: Any, mass: Any) -> Any:
            return module_attention_loop(torch, model, tokens, mass)
        return torch.compile(loop, mode="reduce-overhead")
    raise FailClosedError("unknown v3 venv compile kind")


def _venv_candidate_step(
    torch: Any,
    model: Any,
    optimizer: Any,
    batch: SyntheticBatch,
    *,
    kind: str,
    compiled_loop: Any,
    valid_count: int,
    capture_gradients: bool = False,
) -> dict[str, Any]:
    """One compiled training step inside the triton-pinned venv."""
    optimizer.zero_grad(set_to_none=True)
    model.train(True)
    tokens, mass = hoisted_precompute(torch, model, batch)
    prediction = compiled_loop(tokens, mass)
    loss = _hoisted_dense_mse_exact(torch, prediction, batch, valid_count)
    loss.backward()
    gradients_finite = _all_finite_gradients(torch, model) if capture_gradients else None
    gradients = _gradient_snapshot(model) if capture_gradients else None
    optimizer.step()
    gain, survivor, dropout_p = model.last_unit_gain_mask, model.last_unit_survivor_mask, model.last_dropout_p
    if gain is None or survivor is None or dropout_p is None:
        raise FailClosedError("v3 venv candidate dropout tensors missing")
    from .throughput_benchmark import _DeferredReceipt
    deferred = _DeferredReceipt(loss.detach(), dropout_p.detach(), survivor.detach(), gain.detach())
    optimizer.zero_grad(set_to_none=True)
    return {
        "prediction": prediction.detach().clone(), "loss_tensor": loss.detach(), "deferred_receipt": deferred,
        "finite_loss_tensor": None, "gradient_finite": gradients_finite,
        "gradient_state": gradients, "dropout_p": float(dropout_p.detach().item()),
        "kept": int(survivor.detach().sum().item()),
    }


# --------------------------------------------------------------------------- #
# Receipt cell schema
# --------------------------------------------------------------------------- #

CELL_SCHEMA = "tfsr_b3st4_ddrop_throughput_cell_v3"
_CELL_KEYS = {
    "schema", "label", "kind", "execution", "environment", "batch_size", "units", "dtype",
    "warmup_steps", "measured_steps", "status", "cuda_oom", "oom_error_sha256",
    "typed_normalized_t4", "implementation_exception_class",
    "implementation_exception_repr_sha256", "implementation_failure_stage",
    "post_error_cuda_cleanup_reset", "initial_model_state_sha256",
    "initial_optimizer_state_sha256", "state_before_measure_sha256",
    "optimizer_before_measure_sha256", "fresh_model_from_reference",
    "fresh_optimizer_state", "reused_warmed_state", "finite",
    "post_measurement_finite", "timing", "resources", "runtime", "equivalence",
    "candidate_details",
}


def _validate_runtime(value: object) -> dict[str, object]:
    expected = {"gpu", "torch_version", "cuda_version", "cudnn_version", "cpu_threads", "triton_version"}
    if (not isinstance(value, Mapping) or set(value) != expected or value.get("gpu") != FROZEN_GPU0
            or not all(isinstance(value.get(name), (str, type(None)))
                       for name in ("torch_version", "cuda_version", "cudnn_version", "triton_version"))):
        raise FailClosedError("v3 benchmark runtime GPU contract drift")
    threads = value.get("cpu_threads")
    if not isinstance(threads, Mapping) or threads != {"intraop": 1, "interop": 1}:
        raise FailClosedError("v3 benchmark CPU thread-pool contract drift")
    return {
        "gpu": dict(FROZEN_GPU0), "torch_version": value["torch_version"],
        "cuda_version": value["cuda_version"], "cudnn_version": value["cudnn_version"],
        "triton_version": value["triton_version"], "cpu_threads": {"intraop": 1, "interop": 1},
    }


def validate_cell_payload(
    value: Mapping[str, Any],
    *,
    label: str,
    kind: str,
    execution: str,
    environment: str,
    units: int,
    reference_model_sha256: str | None = None,
    reference_optimizer_sha256: str | None = None,
) -> dict[str, object]:
    if (not isinstance(value, Mapping) or set(value) != _CELL_KEYS or value.get("schema") != CELL_SCHEMA
            or value.get("label") != label or value.get("kind") != kind
            or value.get("execution") != execution or value.get("environment") != environment
            or value.get("batch_size") != BATCH_SIZE or value.get("units") != units
            or value.get("dtype") != "float32"
            or value.get("warmup_steps") != UNIT_SPECS[units].warmup_steps
            or value.get("measured_steps") != UNIT_SPECS[units].measured_steps
            or value.get("typed_normalized_t4") is not True
            or value.get("reused_warmed_state") is not False
            or (value.get("candidate_details") is not None
                and not isinstance(value.get("candidate_details"), Mapping))):
        raise FailClosedError("v3 benchmark cell identity/freshness drift")
    status, is_oom = value.get("status"), value.get("cuda_oom")
    if status not in {"MEASURED", "CUDA_OOM", IMPLEMENTATION_UNAVAILABLE} or is_oom is not (status == "CUDA_OOM"):
        raise FailClosedError("v3 benchmark cell OOM status drift")
    oom_sha = value.get("oom_error_sha256")
    exception_class = value.get("implementation_exception_class")
    exception_repr_sha = value.get("implementation_exception_repr_sha256")
    failure_stage = value.get("implementation_failure_stage")
    post_error_cleanup = value.get("post_error_cuda_cleanup_reset")
    if status in {"CUDA_OOM", IMPLEMENTATION_UNAVAILABLE}:
        # A failed cell never carries measurement or equivalence evidence.  A
        # cell whose variant was never constructed (e.g. the venv process
        # itself failed) honestly carries null lineage and a null runtime;
        # a cell that failed mid-flight carries its real lineage/runtime.
        if (value.get("timing") is not None or value.get("finite") is not None
                or value.get("post_measurement_finite") is not None or value.get("equivalence") is not None):
            raise FailClosedError("v3 failed cell has fabricated measurement evidence")
        resources = value.get("resources")
        if not isinstance(resources, Mapping) or set(resources) != {
            "peak_allocated_bytes", "peak_reserved_bytes", "rss_bytes", "oom_cleanup_reset",
        }:
            raise FailClosedError("v3 failed-cell resource schema drift")
        for key in ("peak_allocated_bytes", "peak_reserved_bytes", "rss_bytes"):
            if type(resources.get(key)) is not int or resources[key] < 0:
                raise FailClosedError("v3 failed-cell resource value drift")
        constructed = value.get("fresh_model_from_reference") is True and value.get("fresh_optimizer_state") is True
        if constructed:
            if reference_model_sha256 is None or value.get("initial_model_state_sha256") != reference_model_sha256 \
                    or value.get("initial_optimizer_state_sha256") != reference_optimizer_sha256:
                raise FailClosedError("v3 failed constructed-cell lineage drift")
            _require_sha(value.get("state_before_measure_sha256"), "v3 pre-measure model state")
            _require_sha(value.get("optimizer_before_measure_sha256"), "v3 pre-measure optimizer state")
            _validate_runtime(value.get("runtime"))
        else:
            if (value.get("initial_model_state_sha256") is not None
                    or value.get("initial_optimizer_state_sha256") is not None
                    or value.get("fresh_model_from_reference") is not False
                    or value.get("fresh_optimizer_state") is not False
                    or value.get("runtime") is not None):
                raise FailClosedError("v3 never-constructed cell lineage/runtime drift")
            _require_sha(value.get("state_before_measure_sha256"), "v3 unavailable marker model state")
            _require_sha(value.get("optimizer_before_measure_sha256"), "v3 unavailable marker optimizer state")
        if status == "CUDA_OOM":
            _require_sha(oom_sha, "v3 CUDA OOM error")
            if (exception_class is not None or exception_repr_sha is not None or failure_stage is not None
                    or post_error_cleanup is not False or resources.get("oom_cleanup_reset") is not True):
                raise FailClosedError("v3 CUDA OOM cell evidence drift")
            return dict(value)
        if oom_sha is not None:
            raise FailClosedError("v3 implementation-unavailable cell OOM drift")
        if (not isinstance(exception_class, str) or not exception_class
                or exception_class != exception_class.strip() or "." not in exception_class):
            raise FailClosedError("v3 implementation-unavailable exception-class drift")
        _require_sha(exception_repr_sha, "v3 implementation-unavailable exception repr")
        if failure_stage not in IMPLEMENTATION_FAILURE_STAGES:
            raise FailClosedError("v3 implementation-unavailable failure-stage drift")
        if post_error_cleanup is not True or resources.get("oom_cleanup_reset") is not False:
            raise FailClosedError("v3 implementation-unavailable CUDA cleanup proof drift")
        return dict(value)
    if (value.get("initial_model_state_sha256") != reference_model_sha256
            or value.get("initial_optimizer_state_sha256") != reference_optimizer_sha256
            or value.get("fresh_model_from_reference") is not True or value.get("fresh_optimizer_state") is not True):
        raise FailClosedError("v3 measured cell identity/freshness drift")
    _require_sha(value.get("state_before_measure_sha256"), "v3 pre-measure model state")
    _require_sha(value.get("optimizer_before_measure_sha256"), "v3 pre-measure optimizer state")
    _validate_runtime(value.get("runtime"))
    if kind == "cuda_graphed_core":
        details = value.get("candidate_details") or {}
        if details.get("graph_capture") != "torch.cuda.graphs.make_graphed_callables":
            raise FailClosedError("v3 graphed cell capture evidence drift")
    if kind in VENV_KINDS:
        details = value.get("candidate_details") or {}
        if details.get("environment") != "triton31_venv" or details.get("triton_version") != TRITON_PIN:
            raise FailClosedError("v3 venv cell environment evidence drift")
    if oom_sha is not None or exception_class is not None or exception_repr_sha is not None or failure_stage is not None:
        raise FailClosedError("v3 measured cell has fabricated failure evidence")
    if post_error_cleanup is not False:
        raise FailClosedError("v3 measured cell cleanup drift")
    finite = value.get("finite")
    if not isinstance(finite, Mapping) or finite != {"forward": True, "loss": True, "gradient": True}:
        raise FailClosedError("v3 forward/loss/gradient finite drift")
    post = value.get("post_measurement_finite")
    if not isinstance(post, Mapping) or post != {"model": True, "optimizer": True}:
        raise FailClosedError("v3 post-measure finite drift")
    timing = value.get("timing")
    expected_timing = {
        "warmup_wall_seconds", "measured_total_wall_seconds", "median_step_wall_seconds",
        "steps_per_second", "samples_per_second", "projected_48epoch_seconds", "projection_label",
    }
    if not isinstance(timing, Mapping) or set(timing) != expected_timing:
        raise FailClosedError("v3 timing schema drift")
    for key in expected_timing - {"projection_label"}:
        _finite(timing.get(key), positive=True)
    if timing.get("projection_label") != ENGINEERING_PROJECTION_LABEL:
        raise FailClosedError("v3 projection label drift")
    expected_projection = TRAINING_STEPS_48_EPOCHS / float(timing["steps_per_second"])
    if not math.isclose(float(timing["projected_48epoch_seconds"]), expected_projection, rel_tol=0.0, abs_tol=1e-9):
        raise FailClosedError("v3 projected engineering wall drift")
    resources = value.get("resources")
    if not isinstance(resources, Mapping) or set(resources) != {
        "peak_allocated_bytes", "peak_reserved_bytes", "rss_bytes", "oom_cleanup_reset",
    } or resources.get("oom_cleanup_reset") is not False:
        raise FailClosedError("v3 measured resource schema drift")
    for key in ("peak_allocated_bytes", "peak_reserved_bytes", "rss_bytes"):
        if type(resources.get(key)) is not int or resources[key] < 0:
            raise FailClosedError("v3 resource value drift")
    equivalence = value.get("equivalence")
    if kind == "production_contract":
        if equivalence is not None:
            raise FailClosedError("v3 production baseline must not carry equivalence")
    else:
        if equivalence is None:
            raise FailClosedError("v3 measured candidate cell is missing trajectory equivalence")
        validate_equivalence_payload(equivalence)
    return dict(value)


# --------------------------------------------------------------------------- #
# Physical backend
# --------------------------------------------------------------------------- #

@dataclass
class _VariantRuntime:
    model: Any
    optimizer: Any
    batch: SyntheticBatch
    valid_count: int
    graphed_core: Any
    scripted_step: Any
    compiled_loop: Any
    initial_model_state_sha256: str
    initial_optimizer_state_sha256: str


class PhysicalBenchmarkBackendV3:
    """Lazy, no-data CUDA backend for the v3 matrix (GPU0 only)."""

    def __init__(self, root: Path, *, kinds: tuple[str, ...] | None = None) -> None:
        self.root = root.absolute()
        self.kinds = tuple(BASE_KINDS + VENV_KINDS) if kinds is None else tuple(kinds)
        self.torch: Any | None = None
        self.model_module: Any | None = None
        self.device: Any | None = None
        self.runtime: Mapping[str, object] | None = None
        self.snapshot: Mapping[str, Any] | None = None
        self.reference_model_state_sha256: str | None = None
        self.reference_optimizer_state_sha256: str | None = None
        self.environment = "base"

    def prepare(self) -> None:
        if self.torch is not None:
            raise FailClosedError("v3 physical benchmark backend prepared twice")
        import torch
        from . import model as model_module
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            raise FailClosedError("cannot establish one-thread v3 benchmark interop pool") from error
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise FailClosedError("v3 benchmark CPU thread-pool setting drift")
        runtime = require_exact_gpu0(torch)
        runtime = {**dict(runtime), "triton_version": _installed_triton_version()}
        _runtime_seed(torch, UNIT_SPECS[128].seed)
        device = torch.device("cuda:0")
        template = model_module.TFSRDecoder(capture_diagnostics=False).to(device)
        snapshot = _snapshot_cpu_state(torch, template)
        reference_model = model_state_digest(torch, template)
        optimizer = _new_frozen_adam(torch, template)
        reference_optimizer = optimizer_state_digest(torch, optimizer)
        self.torch, self.model_module, self.device, self.runtime = torch, model_module, device, runtime
        self.snapshot = snapshot
        self.reference_model_state_sha256 = reference_model
        self.reference_optimizer_state_sha256 = reference_optimizer
        del template, optimizer
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

    @property
    def reference_model_state(self) -> str:
        if self.reference_model_state_sha256 is None:
            raise FailClosedError("v3 backend reference model state is not prepared")
        return self.reference_model_state_sha256

    @property
    def reference_optimizer_state(self) -> str:
        if self.reference_optimizer_state_sha256 is None:
            raise FailClosedError("v3 backend reference optimizer state is not prepared")
        return self.reference_optimizer_state_sha256

    def _require_prepared(self) -> tuple[Any, Any, Any, Mapping[str, Any], Mapping[str, Any]]:
        if (self.torch is None or self.model_module is None or self.device is None or self.runtime is None
                or self.snapshot is None or self.reference_model_state_sha256 is None
                or self.reference_optimizer_state_sha256 is None):
            raise FailClosedError("v3 physical benchmark backend not prepared")
        return (self.torch, self.model_module, self.device, self.runtime, self.snapshot)

    def _fresh_variant(self, *, kind: str, units: int) -> _VariantRuntime:
        torch, model_module, device, _runtime, snapshot = self._require_prepared()
        model = _fresh_model_from_snapshot(torch, model_module, snapshot, device)
        initial_model = model_state_digest(torch, model)
        if initial_model != self.reference_model_state:
            raise FailClosedError("v3 variant did not start from exact reference model state")
        optimizer = _new_frozen_adam(torch, model)
        initial_optimizer = optimizer_state_digest(torch, optimizer)
        if initial_optimizer != self.reference_optimizer_state:
            raise FailClosedError("v3 variant did not start from fresh exact Adam state")
        batch = make_synthetic_batch(
            torch=torch, model_module=model_module, batch_size=BATCH_SIZE, device=device,
            spec=UNIT_SPECS[units],
        )
        validate_synthetic_batch(
            batch, torch=torch, model_module=model_module, batch_size=BATCH_SIZE, device=device,
            spec=UNIT_SPECS[units],
        )
        graphed_core = build_graphed_core(torch, model, units=units) if kind == "cuda_graphed_core" else None
        scripted_step = build_scripted_step(torch, model) if kind == "jit_scripted_step" else None
        compiled_loop = _venv_compiled_loop(torch, model, kind) if kind in VENV_KINDS else None
        return _VariantRuntime(
            model=model, optimizer=optimizer, batch=batch,
            valid_count=int(batch.valid.sum().item()),
            graphed_core=graphed_core, scripted_step=scripted_step, compiled_loop=compiled_loop,
            initial_model_state_sha256=initial_model, initial_optimizer_state_sha256=initial_optimizer,
        )

    def _candidate_details(self, kind: str) -> dict[str, object] | None:
        if kind == "cuda_graphed_core":
            return {
                "graph_capture": "torch.cuda.graphs.make_graphed_callables",
                "num_warmup_iters": 3,
                "scope": "50-step recurrent core; Cell-D mask outside the graph",
                "attention": "F.multi_head_attention_forward with the frozen weights",
                "gru": "torch._VF.gru with the frozen tensors in one flat buffer",
            }
        if kind in VENV_KINDS:
            return {
                "environment": "triton31_venv",
                "triton_version": TRITON_PIN,
                "compiler": "inductor default" if kind == "venv_compile_default" else "inductor reduce-overhead",
                "construction_shim_applied": False,
            }
        return None

    def _bound_step(self, variant: _VariantRuntime, kind: str):
        torch = self.torch
        if torch is None:
            raise FailClosedError("v3 backend not prepared")

        if kind == "production_contract":
            model_module = self.model_module

            def step(*, capture_gradients: bool = False):
                return _production_contract_step(
                    torch, model_module, variant.model, variant.optimizer, variant.batch,
                    capture_gradients=capture_gradients,
                )
            return step
        if kind == "jit_scripted_step":
            def step(*, capture_gradients: bool = False):
                return _scripted_candidate_step(
                    torch, variant.model, variant.optimizer, variant.batch,
                    scripted_step=variant.scripted_step, valid_count=variant.valid_count,
                    capture_gradients=capture_gradients,
                )
            return step
        if kind == "cuda_graphed_core":
            def step(*, capture_gradients: bool = False):
                return _graphed_candidate_step(
                    torch, variant.model, variant.optimizer, variant.batch,
                    graphed_core=variant.graphed_core, valid_count=variant.valid_count,
                    capture_gradients=capture_gradients,
                )
            return step
        if kind in VENV_KINDS:
            def step(*, capture_gradients: bool = False):
                return _venv_candidate_step(
                    torch, variant.model, variant.optimizer, variant.batch, kind=kind,
                    compiled_loop=variant.compiled_loop, valid_count=variant.valid_count,
                    capture_gradients=capture_gradients,
                )
            return step
        raise FailClosedError("unknown v3 candidate step kind")

    def compare_before_timing(self, *, kind: str, units: int) -> Mapping[str, Any]:
        """Untimed paired 20-step trajectories from the same reference state."""
        torch, model_module, _device, _runtime, _snapshot = self._require_prepared()
        if kind == "production_contract":
            raise FailClosedError("v3 comparison kind cannot be the production baseline")
        if kind == "cuda_graphed_core":
            return self._graphed_trajectory_equivalence(units=units)
        baseline = self._fresh_variant(kind="production_contract", units=units)
        candidate = self._fresh_variant(kind=kind, units=units)
        _validate_hoisted_inputs(torch, model_module, candidate.model, candidate.batch)
        equivalence = build_trajectory_equivalence(
            torch, kind=kind, units=units, seed=UNIT_SPECS[units].seed,
            steps=UNIT_SPECS[units].measured_steps,
            baseline_model=baseline.model, baseline_optimizer=baseline.optimizer,
            baseline_step=self._bound_step(baseline, "production_contract"),
            candidate_model=candidate.model, candidate_optimizer=candidate.optimizer,
            candidate_step=self._bound_step(candidate, kind),
        )
        del baseline, candidate
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        return equivalence

    def _release_graphed_memory(self) -> None:
        """Destroy any dead graph pools before the next graphed capture."""
        import gc

        torch, _model_module, _device, _runtime, _snapshot = self._require_prepared()
        gc.collect()
        torch.cuda.empty_cache()

    def _graphed_trajectory_equivalence(self, *, units: int) -> Mapping[str, Any]:
        """20-step graphed comparison, candidate first, graphs torn down early.

        The captured backward keeps the whole unrolled core alive in the graph
        pool, so the graphed candidate trajectory runs FIRST and its CUDA
        graphs are destroyed before the eager production baseline trajectory
        needs the memory.  The comparison itself is v2's exact convention.
        """
        import gc

        torch, model_module, _device, _runtime, _snapshot = self._require_prepared()
        spec = UNIT_SPECS[units]
        self._release_graphed_memory()
        candidate = self._fresh_variant(kind="cuda_graphed_core", units=units)
        _validate_hoisted_inputs(torch, model_module, candidate.model, candidate.batch)
        candidate_run = _trajectory_run(
            torch, self._bound_step(candidate, "cuda_graphed_core"), steps=spec.measured_steps, seed=spec.seed,
        )
        candidate_optimizer_finite = _all_finite_optimizer(torch, candidate.optimizer)
        candidate.graphed_core = None
        self._release_graphed_memory()
        baseline = self._fresh_variant(kind="production_contract", units=units)
        baseline_run = _trajectory_run(
            torch, self._bound_step(baseline, "production_contract"), steps=spec.measured_steps, seed=spec.seed,
        )
        differences = _max_trajectory_difference(torch, baseline_run, candidate_run)
        first_step = _max_trajectory_difference(torch, baseline_run[:1], candidate_run[:1])
        differences["model_state_max_abs"] = _max_tensor_difference(
            torch, dict(baseline.model.state_dict()), dict(candidate.model.state_dict()),
        )
        differences["optimizer_state_max_abs"] = optimizer_state_max_difference(
            torch, baseline.optimizer.state_dict(), candidate.optimizer.state_dict(),
        )
        finite = {
            "baseline_model": _all_finite_parameters(torch, baseline.model),
            "candidate_model": _all_finite_parameters(torch, candidate.model),
            "baseline_optimizer": _all_finite_optimizer(torch, baseline.optimizer),
            "candidate_optimizer": candidate_optimizer_finite,
        }
        if not all(finite.values()):
            raise FailClosedError("v3 graphed trajectory equivalence nonfinite post-step state")
        exact_match = all(value == 0.0 for value in differences.values())
        within_tolerance = all(differences[key] <= V3_TOLERANCES[key] for key in V3_TOLERANCES)
        first_step_within = all(
            first_step[key] <= V3_TOLERANCES[key]
            for key in ("forward_max_abs", "loss_abs", "gradient_max_abs")
        )
        del baseline, candidate
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        return {
            "label": f"production_contract_vs_cuda_graphed_core_b32_u{units}_20step",
            "same_initial_model_state": True,
            "same_initial_fresh_adam_state": True,
            "trajectory_steps": spec.measured_steps,
            "differences": differences,
            "first_step_differences": first_step,
            "first_step_within_tolerance": first_step_within,
            "all_finite": finite,
            "exact_match": exact_match,
            "fp32_tolerance_map": dict(V3_TOLERANCES),
            "within_tolerance": within_tolerance,
            "comparison_role": DIAGNOSTIC_NON_AUTHORIZING,
            "gru_tensor_copy_map": None,
        }

    def _resources(self, *, oom_cleanup_reset: bool) -> dict[str, object]:
        torch, _model_module, _device, _runtime, _snapshot = self._require_prepared()
        return {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "oom_cleanup_reset": oom_cleanup_reset,
        }

    def _clean_cuda_after_recorded_error(self) -> None:
        torch, _model_module, _device, _runtime, _snapshot = self._require_prepared()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

    def measure(
        self,
        *,
        label: str,
        kind: str,
        execution: str,
        environment: str,
        units: int,
    ) -> Mapping[str, Any]:
        torch, model_module, _device, runtime, _snapshot = self._require_prepared()
        spec = UNIT_SPECS[units]
        variant: _VariantRuntime | None = None
        failure_stage = "variant_construction"

        def common(status: str, **extra: Any) -> dict[str, object]:
            return {
                "schema": CELL_SCHEMA, "label": label, "kind": kind, "execution": execution,
                "environment": environment, "batch_size": BATCH_SIZE, "units": units,
                "dtype": spec.dtype, "warmup_steps": spec.warmup_steps,
                "measured_steps": spec.measured_steps, "status": status,
                "typed_normalized_t4": True,
                "initial_model_state_sha256": (
                    self.reference_model_state if variant is None else variant.initial_model_state_sha256
                ),
                "initial_optimizer_state_sha256": (
                    self.reference_optimizer_state if variant is None else variant.initial_optimizer_state_sha256
                ),
                "state_before_measure_sha256": (
                    self.reference_model_state if variant is None else model_state_digest(torch, variant.model)
                ),
                "optimizer_before_measure_sha256": (
                    self.reference_optimizer_state if variant is None else optimizer_state_digest(torch, variant.optimizer)
                ),
                "fresh_model_from_reference": True, "fresh_optimizer_state": True,
                "reused_warmed_state": False,
                "candidate_details": self._candidate_details(kind),
                "runtime": dict(runtime), "equivalence": None,
                **extra,
            }

        try:
            variant = self._fresh_variant(kind=kind, units=units)
            step = self._bound_step(variant, kind)
            if kind != "production_contract":
                _validate_hoisted_inputs(torch, model_module, variant.model, variant.batch)
            failure_stage = "warmup"
            _runtime_seed(torch, spec.seed)
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize(0)
            warmup_start = time.perf_counter()
            for _ in range(spec.warmup_steps):
                observation = step()
                _materialize_step_observation(torch, observation)
            torch.cuda.synchronize(0)
            warmup_seconds = max(time.perf_counter() - warmup_start, 1e-12)
            failure_stage = "measurement"
            state_before_measure = model_state_digest(torch, variant.model)
            optimizer_before_measure = optimizer_state_digest(torch, variant.optimizer)
            durations: list[float] = []
            finite = {"forward": True, "loss": True, "gradient": True}
            for _ in range(spec.measured_steps):
                torch.cuda.synchronize(0)
                start = time.perf_counter()
                observation = step()
                torch.cuda.synchronize(0)
                durations.append(max(time.perf_counter() - start, 1e-12))
                materialized = _materialize_step_observation(torch, observation)
                finite = {key: bool(finite[key] and materialized["finite"][key] is not False) for key in finite}
            failure_stage = "post_measurement_audit"
            post = {
                "model": _all_finite_parameters(torch, variant.model),
                "optimizer": _all_finite_optimizer(torch, variant.optimizer),
            }
            audit = step(capture_gradients=True)
            audited = _materialize_step_observation(torch, audit)
            finite = {
                "forward": bool(finite["forward"] and audited["finite"]["forward"]),
                "loss": bool(finite["loss"] and audited["finite"]["loss"]),
                "gradient": audited["finite"]["gradient"] is True,
            }
            total = sum(durations)
            ordered = sorted(durations)
            median = ordered[len(ordered) // 2] if len(ordered) % 2 else (
                ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]
            ) / 2.0
            steps_per_second = spec.measured_steps / total
            if not all(finite.values()) or not all(post.values()):
                raise FailClosedError("v3 benchmark measured nonfinite state")
            return common(
                "MEASURED", cuda_oom=False, oom_error_sha256=None,
                implementation_exception_class=None, implementation_exception_repr_sha256=None,
                implementation_failure_stage=None, post_error_cuda_cleanup_reset=False,
                finite=finite, post_measurement_finite=post,
                timing={
                    "warmup_wall_seconds": warmup_seconds, "measured_total_wall_seconds": total,
                    "median_step_wall_seconds": median, "steps_per_second": steps_per_second,
                    "samples_per_second": steps_per_second * BATCH_SIZE,
                    "projected_48epoch_seconds": TRAINING_STEPS_48_EPOCHS / steps_per_second,
                    "projection_label": ENGINEERING_PROJECTION_LABEL,
                },
                resources=self._resources(oom_cleanup_reset=False),
            )
        except torch.OutOfMemoryError as error:
            self._clean_cuda_after_recorded_error()
            return common(
                "CUDA_OOM", cuda_oom=True, oom_error_sha256=_sha(repr(error).encode("utf-8")),
                implementation_exception_class=None, implementation_exception_repr_sha256=None,
                implementation_failure_stage=None, post_error_cuda_cleanup_reset=False,
                finite=None, post_measurement_finite=None, timing=None,
                resources=self._resources(oom_cleanup_reset=True),
            )
        except FailClosedError:
            raise
        except Exception as error:
            if execution != "compiled" and kind != "cuda_graphed_core":
                raise
            self._clean_cuda_after_recorded_error()
            return common(
                IMPLEMENTATION_UNAVAILABLE, cuda_oom=False, oom_error_sha256=None,
                implementation_exception_class=_exception_class_name(error),
                implementation_exception_repr_sha256=_sha(repr(error).encode("utf-8")),
                implementation_failure_stage=failure_stage, post_error_cuda_cleanup_reset=True,
                finite=None, post_measurement_finite=None, timing=None,
                resources=self._resources(oom_cleanup_reset=False),
            )


def _installed_triton_version() -> str | None:
    try:
        import triton  # noqa: PLC0415
    except ImportError:
        return None
    return str(triton.__version__)


class MatrixBackend(Protocol):
    """Injected no-data backend for matrix scheduling and CPU adversarial tests."""

    reference_model_state: str
    reference_optimizer_state: str

    def compare_before_timing(self, *, kind: str, units: int) -> Mapping[str, Any]: ...

    def measure(
        self,
        *,
        label: str,
        kind: str,
        execution: str,
        environment: str,
        units: int,
    ) -> Mapping[str, Any]: ...


def _measure_cell(
    backend: Any,
    *,
    label: str,
    kind: str,
    execution: str,
    environment: str,
    units: int,
) -> dict[str, object]:
    raw = dict(backend.measure(
        label=label, kind=kind, execution=execution, environment=environment, units=units,
    ))
    if raw["status"] == "MEASURED" and kind != "production_contract":
        try:
            equivalence = validate_equivalence_payload(backend.compare_before_timing(kind=kind, units=units))
            raw["equivalence"] = dict(equivalence)
        except backend.torch.OutOfMemoryError as error:  # type: ignore[attr-defined]
            # A cell whose timing fit but whose PAIRED equivalence trajectory
            # does not fit alongside the capture's graph pool is recorded as
            # implementation-unavailable with the exact OOM evidence; a timed
            # cell without equivalence evidence must not enter the receipt.
            backend._clean_cuda_after_recorded_error()
            raw.update({
                "status": IMPLEMENTATION_UNAVAILABLE,
                "implementation_exception_class": "torch.OutOfMemoryError",
                "implementation_exception_repr_sha256": _sha(repr(error).encode("utf-8")),
                "implementation_failure_stage": "post_measurement_audit",
                "post_error_cuda_cleanup_reset": True,
                "finite": None, "post_measurement_finite": None, "timing": None,
                "equivalence": None,
                "resources": raw["resources"] | {"oom_cleanup_reset": False},
            })
    return validate_cell_payload(
        raw, label=label, kind=kind, execution=execution, environment=environment, units=units,
        reference_model_sha256=backend.reference_model_state,
        reference_optimizer_sha256=backend.reference_optimizer_state,
    )


def run_base_matrix(backend: MatrixBackend) -> list[dict[str, object]]:
    """Schedule the base-environment cells of the frozen ordered v3 matrix."""
    _require_sha(backend.reference_model_state, "v3 reference model state")
    _require_sha(backend.reference_optimizer_state, "v3 reference optimizer state")
    cells: list[dict[str, object]] = []
    for planned in planned_matrix():
        if planned["environment"] != "base":
            continue
        cell = _measure_cell(
            backend, label=str(planned["label"]), kind=str(planned["kind"]),
            execution=str(planned["execution"]), environment="base", units=int(planned["units"]),
        )
        if planned["kind"] == "production_contract" and cell["status"] != "MEASURED":
            raise FailClosedError("v3 production baseline must be measured to establish equivalence")
        cells.append(cell)
    return cells


def run_venv_matrix(backend: MatrixBackend) -> list[dict[str, object]]:
    """Schedule the triton-venv cells (used inside the venv subprocess)."""
    _require_sha(backend.reference_model_state, "v3 reference model state")
    _require_sha(backend.reference_optimizer_state, "v3 reference optimizer state")
    cells: list[dict[str, object]] = []
    for planned in planned_matrix():
        if planned["environment"] != "triton31_venv":
            continue
        cells.append(_measure_cell(
            backend, label=str(planned["label"]), kind=str(planned["kind"]),
            execution=str(planned["execution"]), environment="triton31_venv",
            units=int(planned["units"]),
        ))
    return cells


def _validate_matrix_cells(cells: object) -> list[dict[str, object]]:
    """Validate the recorded cells against the exact planned v3 matrix."""
    if not isinstance(cells, list):
        raise FailClosedError("v3 benchmark matrix cells schema drift")
    planned = planned_matrix()
    if len(cells) != len(planned):
        raise FailClosedError("v3 benchmark matrix cardinality drift")
    first = cells[0]
    if not isinstance(first, Mapping):
        raise FailClosedError("v3 benchmark first-cell schema drift")
    reference_model = str(first.get("initial_model_state_sha256"))
    reference_optimizer = str(first.get("initial_optimizer_state_sha256"))
    _require_sha(reference_model, "v3 matrix reference model state")
    _require_sha(reference_optimizer, "v3 matrix reference optimizer state")
    checked: list[dict[str, object]] = []
    for cursor, plan in enumerate(planned):
        item = cells[cursor]
        if not isinstance(item, Mapping) or item.get("label") != plan["label"]:
            raise FailClosedError("v3 benchmark attempted matrix order drift")
        checked.append(validate_cell_payload(
            item, label=str(plan["label"]), kind=str(plan["kind"]), execution=str(plan["execution"]),
            environment=str(plan["environment"]), units=int(plan["units"]),
            reference_model_sha256=reference_model, reference_optimizer_sha256=reference_optimizer,
        ))
    return checked


# --------------------------------------------------------------------------- #
# Conclusion and receipt
# --------------------------------------------------------------------------- #

def _conclusion(cells: list[dict[str, object]], venv_report: Mapping[str, Any]) -> dict[str, object]:
    """Derive an honest engineering ranking from the measured cells.

    Acceptance uses the FIRST-STEP comparison against the frozen map (the
    direct identity proof).  The 20-step trajectory is reported verbatim with
    the amplification note.  Speedups are always within-session, against the
    v3 production cell of the same unit count.  The venv cells were measured
    in a separate process (same GPU, same protocol); their comparison with
    base-environment cells carries that caveat explicitly.
    """
    rows: list[dict[str, object]] = []
    for units in UNIT_ORDER:
        production = next(
            (item for item in cells
             if item["kind"] == "production_contract" and item["units"] == units and item["status"] == "MEASURED"),
            None,
        )
        if production is None:
            raise FailClosedError("v3 conclusion requires a measured production baseline per unit count")
        production_median = float(production["timing"]["median_step_wall_seconds"])
        for item in cells:
            if item["units"] != units or item["kind"] == "production_contract":
                continue
            if item["status"] != "MEASURED":
                rows.append({
                    "units": units, "kind": item["kind"], "label": item["label"],
                    "environment": item["environment"],
                    "median_step_wall_seconds": None, "speedup_vs_production": None,
                    "first_step_forward_max_abs": None, "first_step_within_tolerance": None,
                    "trajectory_forward_max_abs": None, "trajectory_within_tolerance": None,
                    "measured": False,
                    "unavailable_reason": item.get("implementation_exception_class"),
                })
                continue
            equivalence = item["equivalence"] or {}
            differences = equivalence.get("differences") or {}
            first_step = equivalence.get("first_step_differences") or {}
            rows.append({
                "units": units, "kind": item["kind"], "label": item["label"],
                "environment": item["environment"],
                "median_step_wall_seconds": float(item["timing"]["median_step_wall_seconds"]),
                "speedup_vs_production": production_median / float(item["timing"]["median_step_wall_seconds"]),
                "first_step_forward_max_abs": first_step.get("forward_max_abs"),
                "first_step_within_tolerance": equivalence.get("first_step_within_tolerance"),
                "trajectory_forward_max_abs": differences.get("forward_max_abs"),
                "trajectory_within_tolerance": equivalence.get("within_tolerance"),
                "measured": True, "unavailable_reason": None,
            })
    accepted = [row for row in rows if row["measured"] and row["first_step_within_tolerance"] is True]
    fastest_accepted = min(accepted, key=lambda row: float(row["median_step_wall_seconds"])) if accepted else None
    recommendations: list[str] = []
    if fastest_accepted is not None and float(fastest_accepted["speedup_vs_production"]) > 1.05:
        recommendations.append(
            f"request a production successor review for {fastest_accepted['kind']} at units={fastest_accepted['units']} "
            f"(+{100.0 * (float(fastest_accepted['speedup_vs_production']) - 1.0):.1f}% vs the v3 production-contract "
            "eager cell; first paired step exact/in-band under the frozen FP32 map)"
        )
    rejected = [row for row in rows if row["measured"] and row["first_step_within_tolerance"] is False]
    for row in rejected:
        recommendations.append(
            f"{row['kind']} at units={row['units']} is numerically rejected: its FIRST paired step is outside the "
            "frozen tolerance map, so the implementation is not the same mathematics; do not promote it"
        )
    amplified = [row for row in accepted if row["trajectory_within_tolerance"] is False]
    if amplified:
        recommendations.append(
            "cells whose first step is exact/in-band but whose 20-step trajectory exceeds the single-step map "
            f"({', '.join(sorted({str(row['kind']) for row in amplified}))}) show recurrent fp32 amplification of "
            "backward reduction-order differences, not different math; a production successor review must define a "
            "trajectory-scale tolerance before promotion"
        )
    unavailable = [row for row in rows if not row["measured"]]
    if unavailable:
        labels = ", ".join(sorted("{}@u{}".format(row["kind"], row["units"]) for row in unavailable))
        recommendations.append(
            f"implementation-unavailable cells ({labels}) carry their exact "
            "exception class/SHA and are environment evidence, not candidate rejections"
        )
    if any(row["environment"] == "triton31_venv" and row["measured"] for row in rows):
        recommendations.append(
            "venv compile cells ran in a separate triton-3.1.0 process on the same GPU0 under the same protocol; "
            "their wall times are comparable engineering evidence, but a production successor request must decide "
            "whether the throwaway-venv arrangement is an acceptable deployment environment"
        )
    if venv_report.get("failure") is not None:
        recommendations.append(
            "the triton-pinned venv could not be established/reused; the recorded failure is the environment verdict "
            "for the compile lever on this host"
        )
    return {
        "ranking": sorted(rows, key=lambda row: (int(row["units"]), float(row["median_step_wall_seconds"] or 1e18))),
        "fastest_first_step_within_tolerance": fastest_accepted,
        "recommendations": recommendations,
        "statement": (
            "Engineering throughput evidence only. No cell here authorizes a training successor by itself; "
            "each recommendation is a request for a separate reviewed successor decision."
        ),
    }


def validate_conclusion_payload(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "ranking", "fastest_first_step_within_tolerance", "recommendations", "statement",
    }:
        raise FailClosedError("v3 conclusion schema drift")
    ranking = value.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise FailClosedError("v3 conclusion ranking drift")
    for row in ranking:
        if not isinstance(row, Mapping) or set(row) != {
            "units", "kind", "label", "environment", "median_step_wall_seconds",
            "speedup_vs_production", "first_step_forward_max_abs", "first_step_within_tolerance",
            "trajectory_forward_max_abs", "trajectory_within_tolerance", "measured", "unavailable_reason",
        }:
            raise FailClosedError("v3 conclusion ranking row drift")
        if row["measured"] is not (row["median_step_wall_seconds"] is not None):
            raise FailClosedError("v3 conclusion measured/timing drift")
    if not isinstance(value.get("recommendations"), list) or not isinstance(value.get("statement"), str):
        raise FailClosedError("v3 conclusion text drift")
    fastest = value.get("fastest_first_step_within_tolerance")
    if fastest is not None and not isinstance(fastest, Mapping):
        raise FailClosedError("v3 conclusion fastest-row drift")
    return dict(value)


def _validate_venv_report(value: object) -> dict[str, object]:
    expected = {"venv_python", "created", "reused", "triton_pin", "torch_version", "triton_version", "failure"}
    if not isinstance(value, Mapping) or set(value) != expected or value.get("triton_pin") != TRITON_PIN:
        raise FailClosedError("v3 venv report schema drift")
    if type(value.get("created")) is not bool or type(value.get("reused")) is not bool:
        raise FailClosedError("v3 venv report flag drift")
    if value.get("failure") is not None and not isinstance(value.get("failure"), Mapping):
        raise FailClosedError("v3 venv failure evidence drift")
    return dict(value)


def build_engineering_receipt(
    *,
    cells: list[dict[str, object]],
    launch_closure: Mapping[str, Any],
    final_closure: Mapping[str, Any],
    runtime: Mapping[str, Any],
    venv_report: Mapping[str, Any],
) -> dict[str, object]:
    launch = validate_closure_payload(launch_closure)
    final = validate_closure_payload(final_closure)
    checked_cells = _validate_matrix_cells(cells)
    checked_runtime = _validate_runtime(runtime)
    checked_venv = _validate_venv_report(venv_report)
    if any(cell["environment"] == "triton31_venv" and cell["status"] == "MEASURED" for cell in checked_cells):
        if checked_venv.get("failure") is not None or checked_venv.get("triton_version") != TRITON_PIN:
            raise FailClosedError("v3 measured venv cell without a verified triton-pinned venv")
    return {
        "schema": "tfsr_b3st4_ddrop_throughput_engineering_v3",
        "status": "ENGINEERING_BENCHMARK_COMPLETE",
        "cell": CELL,
        "phase": PHASE,
        "purpose": "ENGINEERING_ONLY",
        "scientific_result": False,
        "data_opened": False,
        "checkpoint_opened": False,
        "target_or_formal": False,
        "authorizes_training": False,
        "supersedes": dict(SUPERSEDES),
        "benchmark_spec": {str(units): UNIT_SPECS[units].payload() for units in UNIT_ORDER},
        "fixed_evidence": {
            "workorder": {"relative_path": WORKORDER_RELATIVE, "sha256": WORKORDER_SHA256},
            "model": {"relative_path": MODEL_RELATIVE, "sha256": MODEL_SHA256},
            "native_causal_activity": {"relative_path": NATIVE_ACTIVITY_RELATIVE, "sha256": NATIVE_ACTIVITY_SHA256},
            "live_phase_d_v2_throughput": {
                "relative_path": LIVE_THROUGHPUT_RELATIVE,
                "body_sha256": LIVE_THROUGHPUT_SHA256,
                "steps_per_second": FROZEN_BASELINE["steps_per_second"],
                "elapsed_seconds": FROZEN_BASELINE["elapsed_seconds"],
                "projected_48epoch_seconds": FROZEN_BASELINE["projected_48epoch_seconds"],
            },
            "frozen_adam": dict(FROZEN_ADAM),
            "v2_measured_medians_context_only": {kind: dict(sizes) for kind, sizes in V2_MEASURED_MEDIANS.items()},
        },
        "runtime": checked_runtime,
        "triton_venv": checked_venv,
        "matrix": {"cells": checked_cells},
        "conclusion": _conclusion(checked_cells, checked_venv),
        "launch_closure": launch,
        "final_closure": final,
        "launch_final_closure_equal": launch == final,
        "canonical_output": {"root_relative": OUTPUT_ROOT_RELATIVE, "topology": list(OUTPUT_TOPOLOGY)},
    }


def validate_engineering_receipt(value: Mapping[str, Any]) -> dict[str, object]:
    expected = {
        "schema", "status", "cell", "phase", "purpose", "scientific_result", "data_opened",
        "checkpoint_opened", "target_or_formal", "authorizes_training", "supersedes", "benchmark_spec",
        "fixed_evidence", "runtime", "triton_venv", "matrix", "conclusion", "launch_closure",
        "final_closure", "launch_final_closure_equal", "canonical_output",
    }
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_b3st4_ddrop_throughput_engineering_v3"
            or value.get("status") != "ENGINEERING_BENCHMARK_COMPLETE" or value.get("cell") != CELL
            or value.get("phase") != PHASE or value.get("purpose") != "ENGINEERING_ONLY"
            or value.get("scientific_result") is not False or value.get("data_opened") is not False
            or value.get("checkpoint_opened") is not False or value.get("target_or_formal") is not False
            or value.get("authorizes_training") is not False
            or value.get("supersedes") != SUPERSEDES
            or value.get("benchmark_spec") != {str(units): UNIT_SPECS[units].payload() for units in UNIT_ORDER}
            or value.get("canonical_output") != {"root_relative": OUTPUT_ROOT_RELATIVE, "topology": list(OUTPUT_TOPOLOGY)}):
        raise FailClosedError("v3 engineering receipt boundary/schema drift")
    fixed = value.get("fixed_evidence")
    if (not isinstance(fixed, Mapping) or set(fixed) != {
                "workorder", "model", "native_causal_activity", "live_phase_d_v2_throughput",
                "frozen_adam", "v2_measured_medians_context_only",
            }
            or fixed.get("workorder") != {"relative_path": WORKORDER_RELATIVE, "sha256": WORKORDER_SHA256}
            or fixed.get("model") != {"relative_path": MODEL_RELATIVE, "sha256": MODEL_SHA256}
            or fixed.get("native_causal_activity") != {
                "relative_path": NATIVE_ACTIVITY_RELATIVE, "sha256": NATIVE_ACTIVITY_SHA256,
            }
            or fixed.get("frozen_adam") != FROZEN_ADAM
            or fixed.get("v2_measured_medians_context_only") != V2_MEASURED_MEDIANS):
        raise FailClosedError("v3 engineering receipt fixed evidence drift")
    _validate_runtime(value.get("runtime"))
    _validate_venv_report(value.get("triton_venv"))
    _validate_matrix_cells((value.get("matrix") or {}).get("cells"))
    validate_conclusion_payload(value.get("conclusion"))
    launch = validate_closure_payload(value.get("launch_closure"))
    final = validate_closure_payload(value.get("final_closure"))
    if launch != final or value.get("launch_final_closure_equal") is not True:
        raise FailClosedError("v3 engineering receipt launch/final closure drift")
    return {"launch_closure": launch, "final_closure": final}


class OutputArtifactRootV3:
    """Transactional O_EXCL + fsync + 0444 publisher for the single v3 receipt."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.absolute()

    def publish(self, payload: Mapping[str, Any]) -> str:
        validate_engineering_receipt(payload)
        body = _json_bytes(payload)
        digest = _sha(body)
        identity = _canonical_directory_identity(self.directory)
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        created: list[tuple[str, int, int]] = []
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != identity:
                raise FailClosedError("v3 output root identity drift")
            for leaf in ("receipt.json", "receipt.json.sha256"):
                try:
                    os.stat(leaf, dir_fd=descriptor, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                raise FailClosedError("v3 benchmark receipt output collision")
            payloads = (
                ("receipt.json", body),
                ("receipt.json.sha256", f"{digest}  receipt.json\n".encode("ascii")),
            )
            for leaf, content in payloads:
                fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=descriptor)
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode):
                        raise FailClosedError("v3 receipt O_EXCL type drift")
                    created.append((leaf, info.st_dev, info.st_ino))
                    _write_full(fd, content)
                    os.fchmod(fd, 0o444)
                    os.fsync(fd)
                finally:
                    os.close(fd)
            os.fsync(descriptor)
            self._verify_pair(descriptor, digest)
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

    def _verify_pair(self, descriptor: int, digest: str) -> None:
        body = self._read_leaf(descriptor, "receipt.json")
        if _sha(body) != digest:
            raise FailClosedError("published v3 receipt SHA drift")
        if self._read_leaf(descriptor, "receipt.json.sha256") != f"{digest}  receipt.json\n".encode("ascii"):
            raise FailClosedError("published v3 receipt sidecar drift")

    @staticmethod
    def _read_leaf(descriptor: int, leaf: str) -> bytes:
        fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
                raise FailClosedError("published v3 receipt leaf mode/type drift")
            chunks: list[bytes] = []
            while True:
                block = os.read(fd, 1 << 20)
                if not block:
                    return b"".join(chunks)
                chunks.append(block)
        finally:
            os.close(fd)


def require_canonical_output_fresh(root: Path) -> None:
    relative = _safe_relative(OUTPUT_ROOT_RELATIVE)
    path = root.absolute() / relative
    parent_identity = _canonical_directory_identity(path.parent)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) != parent_identity:
            raise FailClosedError("v3 output parent identity drift")
        try:
            os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise FailClosedError("canonical v3 output root must be fresh")
    finally:
        os.close(descriptor)


def run_venv_cells_only(root: Path) -> dict[str, object]:
    """The venv-subprocess route: measure only the venv cells, print JSON.

    The venv resolves the SAME installed torch (2.5.1.post303) with the
    pinned triton; the inductor construction shim from v2 is deliberately NOT
    applied because this triton ships the dataclass AttrsDescriptor the
    torch build expects.
    """
    backend = PhysicalBenchmarkBackendV3(root, kinds=VENV_KINDS)
    backend.environment = "triton31_venv"
    backend.prepare()
    cells = run_venv_matrix(backend)
    return {
        "cells": cells,
        "runtime": dict(backend.runtime) if backend.runtime is not None else {},
        "triton_version": _installed_triton_version(),
    }


def run_reviewed_gpu_benchmark_v3(root: Path) -> Mapping[str, Any]:
    """The only physical route: venv cells, GPU0 base matrix, one receipt."""
    require_canonical_output_fresh(root)
    launch_closure = compute_benchmark_closure(root)
    # The base backend establishes the reference model/Adam digests first; the
    # venv subprocess must reproduce exactly these digests from the same seed.
    backend = PhysicalBenchmarkBackendV3(root, kinds=BASE_KINDS)
    backend.prepare()
    venv_report = ensure_triton_venv()
    venv_cells: list[dict[str, object]] = []
    venv_failure: dict[str, object] | None = None
    if venv_cells_available(venv_report):
        outcome = run_venv_cell_process(root, venv_report)
        if outcome.get("ok") is True:
            payload = outcome["payload"]
            venv_cells = list(payload["cells"])
            for cell in venv_cells:
                validate_cell_payload(
                    cell, label=str(cell["label"]), kind=str(cell["kind"]),
                    execution=str(cell["execution"]), environment="triton31_venv",
                    units=int(cell["units"]),
                    reference_model_sha256=backend.reference_model_state,
                    reference_optimizer_sha256=backend.reference_optimizer_state,
                )
        else:
            venv_failure = {"ok": False, "error": str(outcome.get("error"))[:800]}
    else:
        venv_failure = dict(venv_report["failure"]) if venv_report.get("failure") is not None else {
            "ok": False, "error": "venv unavailable without a recorded failure",
        }
    if venv_failure is not None:
        for planned in planned_matrix():
            if planned["environment"] != "triton31_venv":
                continue
            venv_cells.append({
                "schema": CELL_SCHEMA, "label": str(planned["label"]), "kind": str(planned["kind"]),
                "execution": str(planned["execution"]), "environment": "triton31_venv",
                "batch_size": BATCH_SIZE, "units": int(planned["units"]), "dtype": "float32",
                "warmup_steps": UNIT_SPECS[int(planned["units"])].warmup_steps,
                "measured_steps": UNIT_SPECS[int(planned["units"])].measured_steps,
                "status": IMPLEMENTATION_UNAVAILABLE, "cuda_oom": False, "oom_error_sha256": None,
                "typed_normalized_t4": True,
                "implementation_exception_class": "venv.VenvCellProcessFailure",
                "implementation_exception_repr_sha256": _sha(str(sorted(venv_failure.items())).encode("utf-8")),
                "implementation_failure_stage": "variant_construction",
                "post_error_cuda_cleanup_reset": True,
                "initial_model_state_sha256": None, "initial_optimizer_state_sha256": None,
                "state_before_measure_sha256": _sha(("v3-venv-unavailable:" + str(planned["label"])).encode("utf-8")),
                "optimizer_before_measure_sha256": _sha(("v3-venv-unavailable-adam:" + str(planned["label"])).encode("utf-8")),
                "fresh_model_from_reference": False, "fresh_optimizer_state": False,
                "reused_warmed_state": False, "finite": None, "post_measurement_finite": None,
                "timing": None,
                "resources": {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0, "rss_bytes": 0,
                              "oom_cleanup_reset": False},
                "runtime": None, "equivalence": None, "candidate_details": None,
            })
    base_cells = run_base_matrix(backend)
    cells: list[dict[str, object]] = []
    for planned in planned_matrix():
        for cell in list(base_cells) + list(venv_cells):
            if cell["label"] == planned["label"]:
                cells.append(dict(cell))
                break
        else:
            raise FailClosedError(f"v3 matrix cell missing after measurement: {planned['label']}")
    final_closure = compute_benchmark_closure(root)
    if backend.runtime is None:
        raise FailClosedError("v3 backend missing runtime evidence")
    receipt = build_engineering_receipt(
        cells=cells, launch_closure=launch_closure, final_closure=final_closure,
        runtime=backend.runtime, venv_report=venv_report,
    )
    validate_engineering_receipt(receipt)
    output = root.absolute() / _safe_relative(OUTPUT_ROOT_RELATIVE)
    output.mkdir(parents=True, mode=0o755, exist_ok=False)
    digest = OutputArtifactRootV3(output).publish(receipt)
    return {"receipt": receipt, "receipt_sha256": digest}
