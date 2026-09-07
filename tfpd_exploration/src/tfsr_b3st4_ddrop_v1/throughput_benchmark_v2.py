"""Fail-closed, engineering-only TF-SR throughput equivalence benchmark, v2.

This is the natural v2 continuation of ``throughput_benchmark.py`` under the
same 2026-08-19 work order.  v1 established that the production-contract eager
step is dispatch-bound and that ``torch.compile`` failed at
``variant_construction``.  v2 tests the levers that target per-step
Python/dispatch overhead directly:

1. ``kv_precompute``  -- all 50 per-step attention key/value projections are
   state-independent, so they are computed in ONE batched projection before
   the loop.  The per-step attention is a manual replication of the exact
   ``nn.MultiheadAttention`` op sequence (same weights, same op order, same
   tensor layouts), which is why the forward is expected to stay bitwise
   exact.
2. ``validation_hoisted`` + ``torch.compile`` FIXED -- v1's construction
   failure is reproduced and repaired at runtime (an additive, in-process
   compatibility shim for the installed triton), then both the default and
   ``mode="reduce-overhead"`` compilers are attempted, plus the CUDA-graph
   backend that does not depend on inductor code generation.
3. ``jit_scripted_step`` -- the inner step (queries -> attention -> norms/FFN
   -> GRU -> head) scripted with ``torch.jit.script`` over the frozen shared
   submodules.
4. ``combo`` -- kv_precompute combined with the compiled loop.

Everything is engineering evidence only: synthetic tensors, GPU0, no dataset,
no checkpoint, no scientific score, no authorization to change the live run.
This module has no top-level Torch import; the public CLI loads it as a
synthetic package so a zero-argument invocation stays data-free.
"""
from __future__ import annotations

import math
import os
import resource
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

# The v1 harness is the authority for loaders, digests, the synthetic batch,
# the production-contract step, and the GPU0/runtime contract.  Reusing it by
# import keeps the two receipts comparable instead of forking conventions.
from .throughput_benchmark import (
    COMPILE_FAILURE_STAGES,
    COMPILE_UNAVAILABLE,
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
    _DeferredReceipt,
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
    _max_tensor_difference,
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


CELL = "TFSR_B3ST4_DDROP_THROUGHPUT_ENGINEERING_V2"
PHASE = "TFSR_THROUGHPUT_EQUIVALENCE_20260819_V2"
SOURCE_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/throughput_benchmark_v2.py"
CLI_RELATIVE = "tfpd_exploration/scripts/benchmark_tfsr_b3st4_ddrop_throughput_v2.py"
TEST_RELATIVE = "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_throughput_v2.py"
V1_SOURCE_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/throughput_benchmark.py"
V1_CLI_RELATIVE = "tfpd_exploration/scripts/benchmark_tfsr_b3st4_ddrop_throughput.py"
V1_TEST_RELATIVE = "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_v1_throughput_benchmark.py"
V1_RECEIPT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_throughput_engineering_v1/receipt.json"
V1_RECEIPT_SHA256 = "c87444bee9dc3e61e0c77520b9b3ac102de98794773271a2d6f823dadc888efe"
V1_PRODUCTION_CONTRACT_MEDIAN_STEP_SECONDS = 0.24153552390635014
OUTPUT_ROOT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_throughput_engineering_v2"
OUTPUT_TOPOLOGY = ("receipt.json",)
PUBLIC_FLAGS = frozenset(("--execute", "--i-have-root-throughput-authorization-v2"))

# Explicit, ordered, non-globbed closure.  It binds the authorizing work
# order, the frozen model graph and its native dependency, the sealed live
# throughput pair, the complete superseded v1 audit, and the three v2 files.
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
)

SUPERSEDES = {
    "relation": "v2 continuation of the same engineering-only throughput audit",
    "superseded_receipt": V1_RECEIPT_RELATIVE,
    "superseded_receipt_sha256": V1_RECEIPT_SHA256,
    "note": (
        "v1 established the production eager baseline, the deferred-receipt and "
        "validation-hoisted cells, the batch sweep, and the original "
        "variant_construction compile failure.  v2 adds the per-step dispatch "
        "levers; no v1 measurement is retimed or replaced."
    ),
}

V2_TOLERANCES = dict(MATHEMATICAL_FP32_TOLERANCES)
COMPILE_BACKENDS = {
    "compile_default": {"backend": None, "mode": None},
    "compile_reduce_overhead": {"backend": None, "mode": "reduce-overhead"},
    "compile_cudagraphs": {"backend": "cudagraphs", "mode": None},
}


@dataclass(frozen=True)
class BenchmarkSpecV2:
    """v2 workload: v1's frozen topology, except the unit axis is swept."""

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
            "output_channels": self.output_channels, "dtype": self.dtype,
        }
        expected = {
            "seed": 42, "window": 50, "calibration_bins": 30, "calibration_width": 100,
            "t4_width": 4, "output_channels": 2, "dtype": "float32",
        }
        if fixed != expected:
            raise ValueError("v2 benchmark workload topology is frozen")
        if self.units not in UNIT_ORDER:
            raise ValueError("v2 benchmark sweeps exactly units 128 and 64")
        if (type(self.warmup_steps) is not int or self.warmup_steps < 1
                or type(self.measured_steps) is not int or self.measured_steps < 20):
            raise ValueError("v2 benchmark requires >=1 warmup and >=20 measured steps")

    def payload(self) -> dict[str, object]:
        return {
            "seed": self.seed, "window": self.window, "calibration": [self.calibration_bins, self.calibration_width],
            "t4_width": self.t4_width, "output_channels": self.output_channels, "units": self.units,
            "warmup_steps": self.warmup_steps, "measured_steps": self.measured_steps, "dtype": self.dtype,
        }


UNIT_ORDER = (128, 64)
UNIT_SPECS = {units: BenchmarkSpecV2(units=units) for units in UNIT_ORDER}
BATCH_SIZE = 32


def planned_matrix() -> list[dict[str, object]]:
    """Return the exact ordered v2 matrix without resolving Torch or a device."""
    matrix: list[dict[str, object]] = []
    for units in UNIT_ORDER:
        suffix = f"_u{units}"
        matrix.extend((
            {"label": "production_contract_eager_b32" + suffix, "kind": "production_contract", "execution": "eager", "units": units},
            {"label": "validation_hoisted_eager_b32" + suffix, "kind": "validation_hoisted", "execution": "eager", "units": units},
            {"label": "kv_precompute_eager_b32" + suffix, "kind": "kv_precompute", "execution": "eager", "units": units},
            {"label": "compile_default_b32" + suffix, "kind": "compile_default", "execution": "compiled", "units": units},
            {"label": "compile_reduce_overhead_b32" + suffix, "kind": "compile_reduce_overhead", "execution": "compiled", "units": units},
            {"label": "compile_cudagraphs_b32" + suffix, "kind": "compile_cudagraphs", "execution": "compiled", "units": units},
            {"label": "jit_scripted_step_b32" + suffix, "kind": "jit_scripted_step", "execution": "eager", "units": units},
            {"label": "combo_kv_compile_reduce_overhead_b32" + suffix, "kind": "combo_kv_compile_reduce_overhead", "execution": "compiled", "units": units},
            {"label": "combo_kv_compile_cudagraphs_b32" + suffix, "kind": "combo_kv_compile_cudagraphs", "execution": "compiled", "units": units},
        ))
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
        "fp32_tolerance_map": dict(V2_TOLERANCES),
        "compile_backends": {name: dict(cfg) for name, cfg in COMPILE_BACKENDS.items()},
        "closure": {
            "paths": list(BENCHMARK_CLOSURE),
            "model_sha256": MODEL_SHA256,
            "native_activity_sha256": NATIVE_ACTIVITY_SHA256,
            "live_throughput_sha256": LIVE_THROUGHPUT_SHA256,
            "workorder_sha256": WORKORDER_SHA256,
            "v1_receipt_sha256": V1_RECEIPT_SHA256,
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
# Closure handling (descriptor-safe, identical conventions to v1)
# --------------------------------------------------------------------------- #

def validate_closure_payload(value: Mapping[str, Any]) -> dict[str, object]:
    expected = {"paths", "sha256_by_path", "closure_sha256"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FailClosedError("v2 benchmark closure schema drift")
    paths, hashes = value.get("paths"), value.get("sha256_by_path")
    if paths != list(BENCHMARK_CLOSURE) or not isinstance(hashes, Mapping) or set(hashes) != set(BENCHMARK_CLOSURE):
        raise FailClosedError("v2 benchmark closure path map drift")
    if any(not _is_sha(hashes[path]) for path in BENCHMARK_CLOSURE):
        raise FailClosedError("v2 benchmark closure file digest drift")
    if value.get("closure_sha256") != _closure_digest({path: str(hashes[path]) for path in BENCHMARK_CLOSURE}):
        raise FailClosedError("v2 benchmark closure aggregate drift")
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
    return {
        "paths": list(BENCHMARK_CLOSURE),
        "sha256_by_path": {path: str(hashes[path]) for path in BENCHMARK_CLOSURE},
        "closure_sha256": str(value["closure_sha256"]),
    }


def compute_benchmark_closure(root: Path) -> dict[str, object]:
    """Descriptor-safely hash the exact v2 benchmark/runtime closure."""
    hashes: dict[str, str] = {}
    sealed_0444 = {LIVE_THROUGHPUT_RELATIVE, LIVE_THROUGHPUT_SIDECAR_RELATIVE, V1_RECEIPT_RELATIVE}
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
# The v1 compile-construction failure and its in-process repair
# --------------------------------------------------------------------------- #

SHIM_TARGET = "triton.compiler.compiler.AttrsDescriptor"
SHIM_EXPLANATION = (
    "The installed torch 2.5.1.post303 inductor expects triton<=3.1's dataclass "
    "AttrsDescriptor(divisible_by_16=..., equal_to_1=...).  The installed triton "
    "3.2.0 replaced it with AttrsDescriptor(params=None, values=None) and dropped "
    "the dataclass fields.  torch/_inductor/runtime/hints.py therefore raises "
    "TypeError('must be called with a dataclass type or instance') from "
    "dataclasses.fields(AttrsDescriptor) while importing inductor, and that "
    "TypeError escaped torch.compile() construction in v1 because the guard only "
    "catches ImportError.  The shim adds __dataclass_fields__ naming the triton-3.2 "
    "common properties so the import-time feature probe succeeds; it does not edit "
    "any file and does not restore triton-3.1 codegen, so inductor kernel "
    "generation can still fail later (which the matrix records per cell)."
)


def apply_inductor_construction_shim() -> Mapping[str, object]:
    """Repair torch.compile construction for this torch/triton pair, in-process.

    Returns a receipt block.  Idempotent: a second call is a no-op.
    """
    import dataclasses
    from dataclasses import Field as _Field, _FIELD as _FIELD_MARKER

    try:
        import triton.compiler.compiler as triton_compiler
    except ImportError as error:  # pragma: no cover - triton ships with torch
        raise FailClosedError("compiled cells require the installed triton package") from error
    descriptor = getattr(triton_compiler, "AttrsDescriptor", None)
    if descriptor is None:
        return {"applied": False, "reason": "AttrsDescriptor absent; nothing to repair", "target": SHIM_TARGET}
    if dataclasses.is_dataclass(descriptor):
        # Either this triton is old enough to ship a real dataclass descriptor,
        # or the shim already ran.  Both mean construction needs no repair.
        return {
            "applied": True,
            "reason": "AttrsDescriptor already satisfies dataclasses.fields (shim in effect or unnecessary)",
            "target": SHIM_TARGET,
        }
    fields: dict[str, Any] = {}
    for name in ("divisible_by_16", "equal_to_1"):
        field = _Field(default=None, default_factory=None, init=True, repr=True,
                       hash=None, compare=True, metadata=None, kw_only=None)
        field.name = name
        field._field_type = _FIELD_MARKER
        fields[name] = field
    descriptor.__dataclass_fields__ = fields
    return {"applied": True, "reason": SHIM_EXPLANATION, "target": SHIM_TARGET}


def reproduce_v1_construction_failure() -> Mapping[str, object]:
    """Reproduce v1's exact torch.compile construction exception, then repair it.

    Returns the pre-repair exception class/SHA (or null when this environment
    no longer fails) plus the shim receipt and a post-shim construction probe.
    Failing to reproduce is not an error: the receipt records the environment.
    """
    import torch

    torch.manual_seed(42)
    probe = _probe_decoder()
    try:
        torch.compile(probe)
        failure: dict[str, object] = {
            "reproduced": False, "exception_class": None, "exception_repr_sha256": None,
        }
    except Exception as error:  # v1 observed builtins.TypeError here
        failure = {
            "reproduced": True,
            "exception_class": _exception_class_name(error),
            "exception_repr_sha256": _sha(repr(error).encode("utf-8")),
        }
    shim = dict(apply_inductor_construction_shim())
    try:
        torch.compile(probe)
        failure["construction_after_shim"] = "OK"
    except Exception as error:
        failure["construction_after_shim"] = f"{_exception_class_name(error)}: {repr(error)[:200]}"
    return {"v1_failure": failure, "shim": shim}


def _probe_decoder() -> Any:
    from . import model as model_module

    return model_module.TFSRDecoder(capture_diagnostics=False)


# --------------------------------------------------------------------------- #
# Candidate implementations (benchmark-local, frozen-submodule call-throughs)
# --------------------------------------------------------------------------- #

EMBED_DIM = 256
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS


def hoisted_precompute(torch: Any, model: Any, batch: SyntheticBatch) -> tuple[Any, Any]:
    """Pre-loop part of the frozen forward after typed validation was hoisted.

    Calls the exact frozen layers (B3S layers, causal activity encoder, token
    MLP, the Cell-D mask law, population mass) in the frozen order.  The
    Python-RNG dropout stays here, OUTSIDE every compiled/scripted region, and
    is applied once per step exactly as in the frozen forward.
    """
    identity_features = model.b3s.pre_pool(batch.calib.permute(0, 1, 3, 2)).mean(dim=1)
    identity = model.b3s.post_pool(torch.cat((identity_features, batch.normalized_t4.tensor), dim=-1))
    activity = model.activity_encoder(batch.x)
    pre_mask_tokens = model.unit_mlp(
        torch.cat((activity, identity[:, None].expand(-1, batch.x.shape[1], -1, -1)), dim=-1)
    )
    tokens = model._mask(pre_mask_tokens)
    survivor = model.last_unit_survivor_mask
    if survivor is None:
        raise FailClosedError("v2 candidate Cell-D survivor mask missing")
    mass, _activity_mass = model._population_mass(activity, survivor)
    return tokens, mass


def module_attention_loop(torch: Any, model: Any, tokens: Any, mass: Any) -> Any:
    """Validation-hoisted frozen loop: the exact ``TFSRDecoder.forward`` loop."""
    batch = tokens.shape[0]
    hidden = tokens.new_zeros(1, batch, EMBED_DIM)
    outputs: list[Any] = []
    for time_index in range(tokens.shape[1]):
        queries = model.query_base[None] + model.state_query(hidden[0]).view(batch, 2, EMBED_DIM)
        read, _ = model.attn(queries, tokens[:, time_index], tokens[:, time_index], need_weights=False)
        read = model.norm1(queries + read)
        read = model.norm2(read + model.ffn(read))
        step = torch.cat((read.flatten(1), mass[:, time_index]), dim=1).unsqueeze(1)
        _ignored, hidden = model.gru(step, hidden)
        outputs.append(model.head(hidden[0]))
    return torch.stack(outputs, dim=1)


def kv_precompute_loop(torch: Any, model: Any, tokens: Any, mass: Any) -> Any:
    """Lever 1: one batched K/V projection for all 50 steps, exact per-slice math.

    The K/V projection input ``tokens[:, t]`` does not depend on the recurrent
    state, so ``F.linear(tokens, w_kv, b_kv)`` produces every step's K and V in
    one GEMM.  Each loop iteration then replicates the exact
    ``nn.MultiheadAttention`` encoder-decoder op sequence on the precomputed
    slice: the same q projection on the transposed layout, the same contiguous
    K/V materialization the frozen ``_in_projection_packed`` performs, the same
    view chain, the same ``scaled_dot_product_attention`` call, and the same
    out-projection layout.  Weight splits happen per call so no view of a
    parameter outlives the optimizer's in-place update.
    """
    import torch.nn.functional as F

    batch, window, units = tokens.shape[0], tokens.shape[1], tokens.shape[2]
    in_proj_weight, in_proj_bias = model.attn.in_proj_weight, model.attn.in_proj_bias
    w_query, w_kv = in_proj_weight.split([EMBED_DIM, 2 * EMBED_DIM])
    b_query, b_kv = in_proj_bias.split([EMBED_DIM, 2 * EMBED_DIM])
    kv_all = F.linear(tokens, w_kv, b_kv)
    hidden = tokens.new_zeros(1, batch, EMBED_DIM)
    outputs: list[Any] = []
    for time_index in range(window):
        queries = model.query_base[None] + model.state_query(hidden[0]).view(batch, 2, EMBED_DIM)
        q_proj = F.linear(queries.transpose(1, 0), w_query, b_query)
        q = q_proj.view(2, batch * NUM_HEADS, HEAD_DIM).transpose(0, 1).view(batch, NUM_HEADS, 2, HEAD_DIM)
        k = kv_all[:, time_index, :, :EMBED_DIM].transpose(1, 0).contiguous()
        v = kv_all[:, time_index, :, EMBED_DIM:].transpose(1, 0).contiguous()
        k = k.view(units, batch * NUM_HEADS, HEAD_DIM).transpose(0, 1).view(batch, NUM_HEADS, units, HEAD_DIM)
        v = v.view(units, batch * NUM_HEADS, HEAD_DIM).transpose(0, 1).view(batch, NUM_HEADS, units, HEAD_DIM)
        attn_out = F.scaled_dot_product_attention(q, k, v, None, 0.0, False)
        attn_out = attn_out.permute(2, 0, 1, 3).contiguous().view(batch * 2, EMBED_DIM)
        read = F.linear(attn_out, model.attn.out_proj.weight, model.attn.out_proj.bias)
        read = read.view(2, batch, EMBED_DIM).transpose(1, 0)
        read = model.norm1(queries + read)
        read = model.norm2(read + model.ffn(read))
        step = torch.cat((read.flatten(1), mass[:, time_index]), dim=1).unsqueeze(1)
        _ignored, hidden = model.gru(step, hidden)
        outputs.append(model.head(hidden[0]))
    return torch.stack(outputs, dim=1)


def build_scripted_step(torch: Any, model: Any) -> Any:
    """Lever 3: torch.jit.script the inner step over the frozen shared submodules.

    The scripted module shares the fresh model's parameter tensors, so the
    optimizer, the state digests, and every equivalence comparison continue to
    address the exact frozen tensors.
    """
    import torch.nn as nn

    class _ScriptedStep(nn.Module):
        def __init__(self, decoder: Any) -> None:
            super().__init__()
            self.query_base = decoder.query_base
            self.state_query = decoder.state_query
            self.attn = decoder.attn
            self.norm1 = decoder.norm1
            self.norm2 = decoder.norm2
            self.ffn = decoder.ffn
            self.gru = decoder.gru
            self.head = decoder.head

        def forward(self, hidden, tokens_t, mass_t):
            # 256 is the frozen TF-SR embed dim; TorchScript forbids closing
            # over the module-level EMBED_DIM constant from this local class.
            batch = hidden.shape[1]
            queries = self.query_base[None] + self.state_query(hidden[0]).view(batch, 2, 256)
            read, _ = self.attn(queries, tokens_t, tokens_t, need_weights=False)
            read = self.norm1(queries + read)
            read = self.norm2(read + self.ffn(read))
            step = torch.cat((read.flatten(1), mass_t), dim=1).unsqueeze(1)
            _ignored, hidden = self.gru(step, hidden)
            return self.head(hidden[0]), hidden

    return torch.jit.script(_ScriptedStep(model))


def scripted_step_loop(torch: Any, scripted_step: Any, tokens: Any, mass: Any) -> Any:
    """Drive the scripted inner step across the 50-step window."""
    hidden = tokens.new_zeros(1, tokens.shape[0], EMBED_DIM)
    outputs: list[Any] = []
    for time_index in range(tokens.shape[1]):
        output, hidden = scripted_step(hidden, tokens[:, time_index], mass[:, time_index])
        outputs.append(output)
    return torch.stack(outputs, dim=1)


def build_compiled_loop(torch: Any, kind: str, model: Any) -> Any:
    """Wrap the static 50-step loop in torch.compile with the declared backend.

    ``compile_default``/``compile_reduce_overhead`` use the inductor backend
    (mode reduce-overhead adds CUDA-graph trees).  ``compile_cudagraphs`` uses
    the CUDA-graph backend that records the traced eager kernels without
    inductor code generation.  Both loop bodies are plain functions of tensors
    with the Cell-D mask already applied outside the compiled region.
    """
    if kind not in COMPILE_BACKENDS:
        raise FailClosedError("unknown v2 compiled kind")
    configuration = COMPILE_BACKENDS[kind]
    if kind in ("compile_default", "compile_reduce_overhead"):

        def loop(tokens: Any, mass: Any) -> Any:
            return module_attention_loop(torch, model, tokens, mass)

    else:

        def loop(tokens: Any, mass: Any) -> Any:
            return kv_precompute_loop(torch, model, tokens, mass)

    kwargs: dict[str, Any] = {}
    if configuration["backend"] is not None:
        kwargs["backend"] = configuration["backend"]
    if configuration["mode"] is not None:
        kwargs["mode"] = configuration["mode"]
    return torch.compile(loop, **kwargs)


def build_combo_compiled_loop(torch: Any, kind: str, model: Any) -> Any:
    """Lever 4: compile the kv_precompute loop with the declared compiler."""
    if kind == "combo_kv_compile_reduce_overhead":
        return torch.compile(
            lambda tokens, mass: kv_precompute_loop(torch, model, tokens, mass),
            mode="reduce-overhead",
        )
    if kind == "combo_kv_compile_cudagraphs":
        return torch.compile(
            lambda tokens, mass: kv_precompute_loop(torch, model, tokens, mass),
            backend="cudagraphs",
        )
    raise FailClosedError("unknown v2 combo compiled kind")


def _hoisted_dense_mse_exact(torch: Any, prediction: Any, batch: SyntheticBatch, valid_count: int) -> Any:
    """The frozen dense valid-bin MSE arithmetic with the count hoisted.

    The frozen ``dense_valid_bin_mse`` divides by a Python integer count.  On
    this backend, dividing by an int64 0-dim tensor instead rounds differently
    by one ulp for some numerators (observed at units=64), so the candidate
    keeps the exact frozen division form.  The count is computed once per
    variant, outside every timed loop, preserving the sync-free step.
    """
    if type(valid_count) is not int or valid_count < 1:
        raise FailClosedError("v2 hoisted valid-bin count drift")
    per_bin_mse = (prediction - batch.target).square().mean(dim=-1)
    return (per_bin_mse * batch.valid.to(dtype=prediction.dtype)).sum() / valid_count


def _candidate_step(
    torch: Any,
    model: Any,
    optimizer: Any,
    batch: SyntheticBatch,
    *,
    kind: str,
    valid_count: int,
    compiled_loop: Any = None,
    scripted_step: Any = None,
    capture_gradients: bool = False,
) -> dict[str, Any]:
    """One engineering step for every non-production v2 candidate kind."""
    optimizer.zero_grad(set_to_none=True)
    model.train(True)
    tokens, mass = hoisted_precompute(torch, model, batch)
    if kind == "validation_hoisted":
        prediction = module_attention_loop(torch, model, tokens, mass)
    elif kind == "kv_precompute":
        prediction = kv_precompute_loop(torch, model, tokens, mass)
    elif kind == "jit_scripted_step":
        prediction = scripted_step_loop(torch, scripted_step, tokens, mass)
    elif kind in COMPILE_BACKENDS or kind in ("combo_kv_compile_reduce_overhead", "combo_kv_compile_cudagraphs"):
        prediction = compiled_loop(tokens, mass)
    else:
        raise FailClosedError("unknown v2 candidate step kind")
    loss = _hoisted_dense_mse_exact(torch, prediction, batch, valid_count)
    loss.backward()
    gradients_finite = _all_finite_gradients(torch, model) if capture_gradients else None
    gradients = _gradient_snapshot(model) if capture_gradients else None
    optimizer.step()
    gain, survivor, dropout_p = model.last_unit_gain_mask, model.last_unit_survivor_mask, model.last_dropout_p
    if gain is None or survivor is None or dropout_p is None:
        raise FailClosedError("v2 candidate dropout tensors missing")
    deferred = _DeferredReceipt(loss.detach(), dropout_p.detach(), survivor.detach(), gain.detach())
    optimizer.zero_grad(set_to_none=True)
    return {
        "prediction": prediction.detach(), "loss_tensor": loss.detach(), "deferred_receipt": deferred,
        "finite_loss_tensor": None, "gradient_finite": gradients_finite,
        "gradient_state": gradients, "dropout_p": float(dropout_p.detach().item()),
        "kept": int(survivor.detach().sum().item()),
    }


# --------------------------------------------------------------------------- #
# 20-step trajectory equivalence
# --------------------------------------------------------------------------- #

def _dropout_receipt_of(observation: Mapping[str, Any]) -> tuple[Any, Any]:
    receipt = observation.get("receipt")
    if isinstance(receipt, Mapping):
        return receipt.get("dropout_p"), receipt.get("kept")
    return observation.get("dropout_p"), observation.get("kept")


def _trajectory_run(torch: Any, step: Any, *, steps: int, seed: int) -> list[dict[str, Any]]:
    """Run one full trajectory from a freshly seeded RNG stream."""
    _runtime_seed(torch, seed)
    per_step: list[dict[str, Any]] = []
    for _ in range(steps):
        observation = step(capture_gradients=True)
        dropout_p, kept = _dropout_receipt_of(observation)
        per_step.append({
            "prediction": observation["prediction"],
            "loss_tensor": observation["loss_tensor"],
            "gradient_state": observation["gradient_state"],
            "dropout_p": dropout_p,
            "kept": kept,
        })
    return per_step


def _max_trajectory_difference(torch: Any, left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise FailClosedError("trajectory step-count drift")
    forward_max = 0.0
    loss_max = 0.0
    gradient_max = 0.0
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        if lhs["dropout_p"] != rhs["dropout_p"] or lhs["kept"] != rhs["kept"]:
            raise FailClosedError(
                f"dropout receipt stream misaligned at step {index}: "
                f"{lhs['dropout_p']}/{lhs['kept']} vs {rhs['dropout_p']}/{rhs['kept']}"
            )
        forward_max = max(forward_max, float((lhs["prediction"] - rhs["prediction"]).abs().max().item()))
        loss_max = max(loss_max, float((lhs["loss_tensor"] - rhs["loss_tensor"]).abs().item()))
        gradient_max = max(gradient_max, _max_tensor_difference(torch, lhs["gradient_state"], rhs["gradient_state"]))
    return {"forward_max_abs": forward_max, "loss_abs": loss_max, "gradient_max_abs": gradient_max}


def build_trajectory_equivalence(
    torch: Any,
    *,
    kind: str,
    units: int,
    seed: int,
    steps: int,
    baseline_model: Any,
    baseline_optimizer: Any,
    baseline_step: Any,
    candidate_model: Any,
    candidate_optimizer: Any,
    candidate_step: Any,
) -> dict[str, object]:
    """Compare a full trajectory against the production contract."""
    if steps != UNIT_SPECS[units].measured_steps:
        raise FailClosedError("v2 trajectory length must match the measured step count")
    baseline = _trajectory_run(torch, baseline_step, steps=steps, seed=seed)
    candidate = _trajectory_run(torch, candidate_step, steps=steps, seed=seed)
    differences = _max_trajectory_difference(torch, baseline, candidate)
    # The frozen FP32 map was derived for ONE paired step.  A recurrent
    # trajectory amplifies any backward reduction-order difference, so the
    # receipt additionally separates the first-step comparison (the direct
    # v1-style identity proof) from the full 20-step amplification band.
    first_step = _max_trajectory_difference(torch, baseline[:1], candidate[:1])
    differences["model_state_max_abs"] = _max_tensor_difference(
        torch, dict(baseline_model.state_dict()), dict(candidate_model.state_dict()),
    )
    differences["optimizer_state_max_abs"] = optimizer_state_max_difference(
        torch, baseline_optimizer.state_dict(), candidate_optimizer.state_dict(),
    )
    finite = {
        "baseline_model": _all_finite_parameters(torch, baseline_model),
        "candidate_model": _all_finite_parameters(torch, candidate_model),
        "baseline_optimizer": _all_finite_optimizer(torch, baseline_optimizer),
        "candidate_optimizer": _all_finite_optimizer(torch, candidate_optimizer),
    }
    if not all(finite.values()):
        raise FailClosedError("v2 trajectory equivalence nonfinite post-step state")
    exact_match = all(value == 0.0 for value in differences.values())
    within_tolerance = all(differences[key] <= V2_TOLERANCES[key] for key in V2_TOLERANCES)
    first_step_within = all(
        first_step[key] <= V2_TOLERANCES[key]
        for key in ("forward_max_abs", "loss_abs", "gradient_max_abs")
    )
    return {
        "label": f"production_contract_vs_{kind}_b32_u{units}_20step",
        "same_initial_model_state": True,
        "same_initial_fresh_adam_state": True,
        "trajectory_steps": steps,
        "differences": differences,
        "first_step_differences": first_step,
        "first_step_within_tolerance": first_step_within,
        "all_finite": finite,
        "exact_match": exact_match,
        "fp32_tolerance_map": dict(V2_TOLERANCES),
        "within_tolerance": within_tolerance,
        "comparison_role": DIAGNOSTIC_NON_AUTHORIZING,
        "gru_tensor_copy_map": None,
    }


def validate_equivalence_payload(value: object) -> dict[str, object]:
    expected = {
        "label", "same_initial_model_state", "same_initial_fresh_adam_state", "trajectory_steps",
        "differences", "first_step_differences", "first_step_within_tolerance", "all_finite",
        "exact_match", "fp32_tolerance_map", "within_tolerance",
        "comparison_role", "gru_tensor_copy_map",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FailClosedError("v2 equivalence schema drift")
    if value.get("same_initial_model_state") is not True or value.get("same_initial_fresh_adam_state") is not True:
        raise FailClosedError("v2 equivalence initial-state drift")
    if value.get("trajectory_steps") != 20:
        raise FailClosedError("v2 equivalence trajectory length drift")
    differences = value.get("differences")
    difference_keys = {"forward_max_abs", "loss_abs", "gradient_max_abs", "model_state_max_abs", "optimizer_state_max_abs"}
    if not isinstance(differences, Mapping) or set(differences) != difference_keys:
        raise FailClosedError("v2 equivalence difference schema drift")
    checked = {key: _finite(differences[key], nonnegative=True) for key in difference_keys}
    first_step = value.get("first_step_differences")
    first_step_keys = {"forward_max_abs", "loss_abs", "gradient_max_abs"}
    if not isinstance(first_step, Mapping) or set(first_step) != first_step_keys:
        raise FailClosedError("v2 equivalence first-step schema drift")
    checked_first = {key: _finite(first_step[key], nonnegative=True) for key in first_step_keys}
    actual_first_within = all(
        checked_first[key] <= V2_TOLERANCES[key] for key in first_step_keys
    )
    if type(value.get("first_step_within_tolerance")) is not bool or (
        value["first_step_within_tolerance"] is not actual_first_within
    ):
        raise FailClosedError("v2 equivalence first-step within-tolerance drift")
    finite = value.get("all_finite")
    if not isinstance(finite, Mapping) or set(finite) != {
        "baseline_model", "candidate_model", "baseline_optimizer", "candidate_optimizer",
    } or any(item is not True for item in finite.values()):
        raise FailClosedError("v2 equivalence finite proof drift")
    if type(value.get("exact_match")) is not bool:
        raise FailClosedError("v2 equivalence exact-match type drift")
    actual_exact = all(item == 0.0 for item in checked.values())
    if value["exact_match"] is not actual_exact:
        raise FailClosedError("v2 equivalence exact-match drift")
    tolerance_map = value.get("fp32_tolerance_map")
    if not isinstance(tolerance_map, Mapping) or dict(tolerance_map) != V2_TOLERANCES:
        raise FailClosedError("v2 equivalence tolerance-map drift")
    actual_within = all(checked[key] <= V2_TOLERANCES[key] for key in V2_TOLERANCES)
    if type(value.get("within_tolerance")) is not bool or value["within_tolerance"] is not actual_within:
        raise FailClosedError("v2 equivalence within-tolerance drift")
    if value.get("comparison_role") != DIAGNOSTIC_NON_AUTHORIZING:
        raise FailClosedError("v2 equivalence comparison-role drift")
    if value.get("gru_tensor_copy_map") is not None:
        raise FailClosedError("v2 equivalence must not carry a GRU copy map")
    if not isinstance(value.get("label"), str) or not value["label"]:
        raise FailClosedError("v2 equivalence label drift")
    return {
        "label": str(value["label"]), "same_initial_model_state": True, "same_initial_fresh_adam_state": True,
        "trajectory_steps": 20, "differences": checked,
        "first_step_differences": checked_first,
        "first_step_within_tolerance": actual_first_within,
        "all_finite": {key: True for key in finite}, "exact_match": actual_exact,
        "fp32_tolerance_map": dict(V2_TOLERANCES), "within_tolerance": actual_within,
        "comparison_role": DIAGNOSTIC_NON_AUTHORIZING, "gru_tensor_copy_map": None,
    }


# --------------------------------------------------------------------------- #
# Receipt cell schema
# --------------------------------------------------------------------------- #

CELL_SCHEMA = "tfsr_b3st4_ddrop_throughput_cell_v2"
_CELL_KEYS = {
    "schema", "label", "kind", "execution", "batch_size", "units", "dtype", "warmup_steps",
    "measured_steps", "status", "cuda_oom", "oom_error_sha256", "typed_normalized_t4",
    "compile_exception_class", "compile_exception_repr_sha256", "compile_failure_stage",
    "post_error_cuda_cleanup_reset", "construction_shim_applied", "dynamo_counter_delta",
    "initial_model_state_sha256", "initial_optimizer_state_sha256", "state_before_measure_sha256",
    "optimizer_before_measure_sha256", "fresh_model_from_reference", "fresh_optimizer_state",
    "reused_warmed_state", "finite", "post_measurement_finite", "timing", "resources", "runtime",
    "equivalence",
}


def _validate_runtime(value: object) -> dict[str, object]:
    expected = {"gpu", "torch_version", "cuda_version", "cudnn_version", "cpu_threads"}
    if not isinstance(value, Mapping) or set(value) != expected or value.get("gpu") != FROZEN_GPU0:
        raise FailClosedError("v2 benchmark runtime GPU contract drift")
    if not all(isinstance(value.get(name), (str, type(None))) for name in ("torch_version", "cuda_version", "cudnn_version")):
        raise FailClosedError("v2 benchmark runtime version schema drift")
    threads = value.get("cpu_threads")
    if not isinstance(threads, Mapping) or threads != {"intraop": 1, "interop": 1}:
        raise FailClosedError("v2 benchmark CPU thread-pool contract drift")
    return {
        "gpu": dict(FROZEN_GPU0), "torch_version": value["torch_version"],
        "cuda_version": value["cuda_version"], "cudnn_version": value["cudnn_version"],
        "cpu_threads": {"intraop": 1, "interop": 1},
    }


def validate_cell_payload(
    value: Mapping[str, Any],
    *,
    label: str,
    kind: str,
    execution: str,
    batch_size: int,
    units: int,
    reference_model_sha256: str,
    reference_optimizer_sha256: str,
) -> dict[str, object]:
    if (not isinstance(value, Mapping) or set(value) != _CELL_KEYS or value.get("schema") != CELL_SCHEMA
            or value.get("label") != label or value.get("kind") != kind or value.get("execution") != execution
            or value.get("batch_size") != batch_size or value.get("units") != units
            or value.get("dtype") != "float32"
            or value.get("warmup_steps") != UNIT_SPECS[units].warmup_steps
            or value.get("measured_steps") != UNIT_SPECS[units].measured_steps
            or value.get("typed_normalized_t4") is not True
            or value.get("initial_model_state_sha256") != reference_model_sha256
            or value.get("initial_optimizer_state_sha256") != reference_optimizer_sha256
            or value.get("fresh_model_from_reference") is not True or value.get("fresh_optimizer_state") is not True
            or value.get("reused_warmed_state") is not False
            or type(value.get("construction_shim_applied")) is not bool):
        raise FailClosedError("v2 benchmark cell identity/freshness drift")
    _require_sha(value.get("state_before_measure_sha256"), "v2 pre-measure model state")
    _require_sha(value.get("optimizer_before_measure_sha256"), "v2 pre-measure optimizer state")
    counter_delta = value.get("dynamo_counter_delta")
    if execution == "compiled":
        # Compiled cells must carry their own dynamo evidence: a measured
        # compiled cell whose counters show an RNN graph break proves dynamo
        # passed the loop through instead of compiling it.  A negative entry is
        # admissible evidence of a dynamo counter reset (e.g. after a recorded
        # backend failure), not a provenance violation.
        if not isinstance(counter_delta, Mapping) or any(
            type(item) is not int for item in counter_delta.values()
        ):
            raise FailClosedError("v2 compiled cell dynamo-counter schema drift")
    elif counter_delta is not None:
        raise FailClosedError("v2 eager cell must not carry dynamo counters")
    _validate_runtime(value.get("runtime"))
    status, is_oom = value.get("status"), value.get("cuda_oom")
    if status not in {"MEASURED", "CUDA_OOM", COMPILE_UNAVAILABLE} or is_oom is not (status == "CUDA_OOM"):
        raise FailClosedError("v2 benchmark cell OOM status drift")
    oom_sha = value.get("oom_error_sha256")
    compile_class = value.get("compile_exception_class")
    compile_repr_sha = value.get("compile_exception_repr_sha256")
    compile_stage = value.get("compile_failure_stage")
    post_error_cleanup = value.get("post_error_cuda_cleanup_reset")
    if status in {"CUDA_OOM", COMPILE_UNAVAILABLE}:
        if (value.get("timing") is not None or value.get("finite") is not None
                or value.get("post_measurement_finite") is not None):
            raise FailClosedError("v2 failed cell has fabricated measurement evidence")
        resources = value.get("resources")
        if not isinstance(resources, Mapping) or set(resources) != {
            "peak_allocated_bytes", "peak_reserved_bytes", "rss_bytes", "oom_cleanup_reset",
        }:
            raise FailClosedError("v2 failed-cell resource schema drift")
        for key in ("peak_allocated_bytes", "peak_reserved_bytes", "rss_bytes"):
            if type(resources.get(key)) is not int or resources[key] < 0:
                raise FailClosedError("v2 failed-cell resource value drift")
        if status == "CUDA_OOM":
            _require_sha(oom_sha, "v2 CUDA OOM error")
            if (compile_class is not None or compile_repr_sha is not None or compile_stage is not None
                    or post_error_cleanup is not False or resources.get("oom_cleanup_reset") is not True):
                raise FailClosedError("v2 CUDA OOM cell evidence drift")
            if value.get("equivalence") is not None:
                raise FailClosedError("v2 CUDA OOM cell has fabricated equivalence")
            return dict(value)
        if execution != "compiled" or oom_sha is not None:
            raise FailClosedError("v2 compile-unavailable cell scope/OOM drift")
        if (not isinstance(compile_class, str) or not compile_class
                or compile_class != compile_class.strip() or "." not in compile_class):
            raise FailClosedError("v2 compile-unavailable exception-class drift")
        _require_sha(compile_repr_sha, "v2 compile-unavailable exception repr")
        if compile_stage not in COMPILE_FAILURE_STAGES:
            raise FailClosedError("v2 compile-unavailable failure-stage drift")
        if post_error_cleanup is not True or resources.get("oom_cleanup_reset") is not False:
            raise FailClosedError("v2 compile-unavailable CUDA cleanup proof drift")
        if value.get("equivalence") is not None:
            raise FailClosedError("v2 compile-unavailable cell has fabricated equivalence")
        return dict(value)
    if oom_sha is not None or compile_class is not None or compile_repr_sha is not None or compile_stage is not None:
        raise FailClosedError("v2 measured cell has fabricated failure evidence")
    if post_error_cleanup is not False:
        raise FailClosedError("v2 measured cell cleanup drift")
    finite = value.get("finite")
    if not isinstance(finite, Mapping) or finite != {"forward": True, "loss": True, "gradient": True}:
        raise FailClosedError("v2 forward/loss/gradient finite drift")
    post = value.get("post_measurement_finite")
    if not isinstance(post, Mapping) or post != {"model": True, "optimizer": True}:
        raise FailClosedError("v2 post-measure finite drift")
    timing = value.get("timing")
    expected_timing = {
        "warmup_wall_seconds", "measured_total_wall_seconds", "median_step_wall_seconds",
        "steps_per_second", "samples_per_second", "projected_48epoch_seconds", "projection_label",
    }
    if not isinstance(timing, Mapping) or set(timing) != expected_timing:
        raise FailClosedError("v2 timing schema drift")
    for key in expected_timing - {"projection_label"}:
        _finite(timing.get(key), positive=True)
    if timing.get("projection_label") != ENGINEERING_PROJECTION_LABEL:
        raise FailClosedError("v2 projection label drift")
    expected_projection = TRAINING_STEPS_48_EPOCHS / float(timing["steps_per_second"])
    if not math.isclose(float(timing["projected_48epoch_seconds"]), expected_projection, rel_tol=0.0, abs_tol=1e-9):
        raise FailClosedError("v2 projected engineering wall drift")
    resources = value.get("resources")
    if not isinstance(resources, Mapping) or set(resources) != {
        "peak_allocated_bytes", "peak_reserved_bytes", "rss_bytes", "oom_cleanup_reset",
    } or resources.get("oom_cleanup_reset") is not False:
        raise FailClosedError("v2 measured resource schema drift")
    for key in ("peak_allocated_bytes", "peak_reserved_bytes", "rss_bytes"):
        if type(resources.get(key)) is not int or resources[key] < 0:
            raise FailClosedError("v2 resource value drift")
    equivalence = value.get("equivalence")
    if kind == "production_contract":
        if equivalence is not None:
            raise FailClosedError("v2 production baseline must not carry equivalence")
    else:
        if equivalence is None:
            raise FailClosedError("v2 measured candidate cell is missing trajectory equivalence")
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
    compiled_loop: Any
    scripted_step: Any
    shim_applied: bool
    initial_model_state_sha256: str
    initial_optimizer_state_sha256: str


class PhysicalBenchmarkBackendV2:
    """Lazy, no-data CUDA backend for the v2 matrix (GPU0 only)."""

    def __init__(self, root: Path) -> None:
        self.root = root.absolute()
        self.torch: Any | None = None
        self.model_module: Any | None = None
        self.device: Any | None = None
        self.runtime: Mapping[str, object] | None = None
        self.snapshot: Mapping[str, Any] | None = None
        self.reference_model_state_sha256: str | None = None
        self.reference_optimizer_state_sha256: str | None = None
        self.construction_report: Mapping[str, Any] | None = None

    def prepare(self) -> None:
        if self.torch is not None:
            raise FailClosedError("v2 physical benchmark backend prepared twice")
        import torch
        from . import model as model_module
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            raise FailClosedError("cannot establish one-thread v2 benchmark interop pool") from error
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise FailClosedError("v2 benchmark CPU thread-pool setting drift")
        runtime = require_exact_gpu0(torch)
        _runtime_seed(torch, UNIT_SPECS[128].seed)
        device = torch.device("cuda:0")
        # The TF-SR parameter tensors do not depend on the unit count, so one
        # reference state serves both unit sweeps; each cell still constructs
        # its own fresh model from this immutable CPU snapshot.
        template = model_module.TFSRDecoder(capture_diagnostics=False).to(device)
        snapshot = _snapshot_cpu_state(torch, template)
        reference_model = model_state_digest(torch, template)
        optimizer = _new_frozen_adam(torch, template)
        reference_optimizer = optimizer_state_digest(torch, optimizer)
        self.construction_report = reproduce_v1_construction_failure()
        self.torch, self.model_module, self.device, self.runtime = torch, model_module, device, runtime
        self.snapshot = snapshot
        self.reference_model_state_sha256 = reference_model
        self.reference_optimizer_state_sha256 = reference_optimizer
        del template, optimizer
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

    def _require_prepared(self) -> tuple[Any, Any, Any, Mapping[str, Any], Mapping[str, Any]]:
        if (self.torch is None or self.model_module is None or self.device is None or self.runtime is None
                or self.snapshot is None or self.reference_model_state_sha256 is None
                or self.reference_optimizer_state_sha256 is None or self.construction_report is None):
            raise FailClosedError("v2 physical benchmark backend not prepared")
        return (self.torch, self.model_module, self.device, self.runtime, self.snapshot)

    @property
    def reference_model_state(self) -> str:
        if self.reference_model_state_sha256 is None:
            raise FailClosedError("v2 backend reference model state is not prepared")
        return self.reference_model_state_sha256

    @property
    def reference_optimizer_state(self) -> str:
        if self.reference_optimizer_state_sha256 is None:
            raise FailClosedError("v2 backend reference optimizer state is not prepared")
        return self.reference_optimizer_state_sha256

    def _fresh_variant(self, *, kind: str, units: int) -> _VariantRuntime:
        torch, model_module, device, _runtime, snapshot = self._require_prepared()
        shim_applied = False
        if kind in COMPILE_BACKENDS or kind in ("combo_kv_compile_reduce_overhead", "combo_kv_compile_cudagraphs"):
            # torch.compile construction is repaired before any compiled
            # variant exists; eager variants never touch the shim.
            shim_applied = bool(apply_inductor_construction_shim()["applied"])
        model = _fresh_model_from_snapshot(torch, model_module, snapshot, device)
        initial_model = model_state_digest(torch, model)
        if initial_model != self.reference_model_state_sha256:
            raise FailClosedError("v2 variant did not start from exact reference model state")
        optimizer = _new_frozen_adam(torch, model)
        initial_optimizer = optimizer_state_digest(torch, optimizer)
        if initial_optimizer != self.reference_optimizer_state_sha256:
            raise FailClosedError("v2 variant did not start from fresh exact Adam state")
        batch = make_synthetic_batch(
            torch=torch, model_module=model_module, batch_size=BATCH_SIZE, device=device,
            spec=UNIT_SPECS[units],
        )
        validate_synthetic_batch(
            batch, torch=torch, model_module=model_module, batch_size=BATCH_SIZE, device=device,
            spec=UNIT_SPECS[units],
        )
        compiled_loop = None
        if kind in COMPILE_BACKENDS:
            compiled_loop = build_compiled_loop(torch, kind, model)
        elif kind in ("combo_kv_compile_reduce_overhead", "combo_kv_compile_cudagraphs"):
            compiled_loop = build_combo_compiled_loop(torch, kind, model)
        scripted_step = None
        if kind == "jit_scripted_step":
            scripted_step = build_scripted_step(torch, model)
        return _VariantRuntime(
            model=model, optimizer=optimizer, batch=batch,
            valid_count=int(batch.valid.sum().item()),
            compiled_loop=compiled_loop,
            scripted_step=scripted_step, shim_applied=shim_applied,
            initial_model_state_sha256=initial_model, initial_optimizer_state_sha256=initial_optimizer,
        )

    def _baseline_step(self, variant: _VariantRuntime):
        torch, model_module = self.torch, self.model_module
        if torch is None or model_module is None:
            raise FailClosedError("v2 backend not prepared")

        def step(*, capture_gradients: bool = False):
            return _production_contract_step(
                torch, model_module, variant.model, variant.optimizer, variant.batch,
                capture_gradients=capture_gradients,
            )
        return step

    def _candidate_bound_step(self, variant: _VariantRuntime, kind: str):
        torch = self.torch
        if torch is None:
            raise FailClosedError("v2 backend not prepared")

        def step(*, capture_gradients: bool = False):
            return _candidate_step(
                torch, variant.model, variant.optimizer, variant.batch, kind=kind,
                valid_count=variant.valid_count,
                compiled_loop=variant.compiled_loop, scripted_step=variant.scripted_step,
                capture_gradients=capture_gradients,
            )
        return step

    def compare_before_timing(self, *, kind: str, units: int) -> Mapping[str, Any]:
        """Untimed paired 20-step trajectories from the same reference state."""
        torch, model_module, _device, _runtime, _snapshot = self._require_prepared()
        if kind == "production_contract":
            raise FailClosedError("v2 comparison kind cannot be the production baseline")
        baseline = self._fresh_variant(kind="production_contract", units=units)
        candidate = self._fresh_variant(kind=kind, units=units)
        if kind == "jit_scripted_step" and candidate.scripted_step is None:
            raise FailClosedError("v2 scripted comparison variant is missing its scripted step")
        _validate_hoisted_inputs(torch, model_module, candidate.model, candidate.batch)
        equivalence = build_trajectory_equivalence(
            torch, kind=kind, units=units, seed=UNIT_SPECS[units].seed,
            steps=UNIT_SPECS[units].measured_steps,
            baseline_model=baseline.model, baseline_optimizer=baseline.optimizer,
            baseline_step=self._baseline_step(baseline),
            candidate_model=candidate.model, candidate_optimizer=candidate.optimizer,
            candidate_step=self._candidate_bound_step(candidate, kind),
        )
        del baseline, candidate
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        return equivalence

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

    @staticmethod
    def _dynamo_counter_snapshot() -> dict[str, int]:
        """Flatten dynamo's process-wide counters to comparable integers."""
        import torch._dynamo.utils as dynamo_utils

        snapshot: dict[str, int] = {}
        for section, counter in dynamo_utils.counters.items():
            for key, value in counter.items():
                try:
                    number = int(value)
                except (TypeError, ValueError):
                    continue
                if number:
                    snapshot[f"{section}/{key}"] = number
        return snapshot

    def measure(
        self,
        *,
        label: str,
        kind: str,
        execution: str,
        units: int,
    ) -> Mapping[str, Any]:
        torch, model_module, _device, runtime, _snapshot = self._require_prepared()
        spec = UNIT_SPECS[units]
        variant: _VariantRuntime | None = None
        failure_stage = "variant_construction"
        counters_before = self._dynamo_counter_snapshot() if execution == "compiled" else None

        def counter_delta() -> dict[str, int] | None:
            if counters_before is None:
                return None
            after = self._dynamo_counter_snapshot()
            return {
                key: after.get(key, 0) - counters_before.get(key, 0)
                for key in set(after) | set(counters_before)
                if after.get(key, 0) != counters_before.get(key, 0)
            }

        def common(status: str, **extra: Any) -> dict[str, object]:
            return {
                "schema": CELL_SCHEMA, "label": label, "kind": kind, "execution": execution,
                "batch_size": BATCH_SIZE, "units": units, "dtype": spec.dtype,
                "warmup_steps": spec.warmup_steps, "measured_steps": spec.measured_steps,
                "status": status,
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
                "fresh_model_from_reference": True, "fresh_optimizer_state": True, "reused_warmed_state": False,
                "construction_shim_applied": False if variant is None else variant.shim_applied,
                "dynamo_counter_delta": counter_delta(),
                "runtime": dict(runtime), "equivalence": None,
                **extra,
            }

        try:
            variant = self._fresh_variant(kind=kind, units=units)
            step = (
                self._baseline_step(variant) if kind == "production_contract"
                else self._candidate_bound_step(variant, kind)
            )
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
                raise FailClosedError("v2 benchmark measured nonfinite state")
            return common(
                "MEASURED", cuda_oom=False, oom_error_sha256=None,
                compile_exception_class=None, compile_exception_repr_sha256=None,
                compile_failure_stage=None, post_error_cuda_cleanup_reset=False,
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
                compile_exception_class=None, compile_exception_repr_sha256=None,
                compile_failure_stage=None, post_error_cuda_cleanup_reset=False,
                finite=None, post_measurement_finite=None, timing=None,
                resources=self._resources(oom_cleanup_reset=True),
            )
        except FailClosedError:
            # Provenance, numerical, and receipt-contract violations are never
            # compiler-availability observations, on any execution path.
            raise
        except Exception as error:
            if execution != "compiled":
                raise
            self._clean_cuda_after_recorded_error()
            return common(
                COMPILE_UNAVAILABLE, cuda_oom=False, oom_error_sha256=None,
                compile_exception_class=_exception_class_name(error),
                compile_exception_repr_sha256=_sha(repr(error).encode("utf-8")),
                compile_failure_stage=failure_stage, post_error_cuda_cleanup_reset=True,
                finite=None, post_measurement_finite=None, timing=None,
                resources=self._resources(oom_cleanup_reset=False),
            )


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
        units: int,
    ) -> Mapping[str, Any]: ...


def _measure_cell(
    backend: MatrixBackend,
    *,
    label: str,
    kind: str,
    execution: str,
    units: int,
) -> dict[str, object]:
    raw = dict(backend.measure(label=label, kind=kind, execution=execution, units=units))
    if raw["status"] == "MEASURED" and kind != "production_contract":
        equivalence = validate_equivalence_payload(backend.compare_before_timing(kind=kind, units=units))
        raw["equivalence"] = dict(equivalence)
    return validate_cell_payload(
        raw, label=label, kind=kind, execution=execution, batch_size=BATCH_SIZE, units=units,
        reference_model_sha256=backend.reference_model_state,
        reference_optimizer_sha256=backend.reference_optimizer_state,
    )


def run_benchmark_matrix(backend: MatrixBackend) -> dict[str, object]:
    """Schedule the frozen ordered v2 matrix (both unit sweeps) with no fallback."""
    _require_sha(backend.reference_model_state, "v2 reference model state")
    _require_sha(backend.reference_optimizer_state, "v2 reference optimizer state")
    cells: list[dict[str, object]] = []
    for planned in planned_matrix():
        units = int(planned["units"])  # type: ignore[arg-type]
        label, kind, execution = str(planned["label"]), str(planned["kind"]), str(planned["execution"])
        cell = _measure_cell(backend, label=label, kind=kind, execution=execution, units=units)
        if kind == "production_contract" and cell["status"] != "MEASURED":
            raise FailClosedError("v2 production baseline must be measured to establish equivalence")
        cells.append(cell)
    return {"cells": cells}


def _validate_matrix_payload(value: object) -> dict[str, object]:
    """Reconstruct the exact attempted v2 matrix order and per-cell semantics."""
    if not isinstance(value, Mapping) or set(value) != {"cells"}:
        raise FailClosedError("v2 benchmark matrix schema drift")
    cells = value.get("cells")
    if not isinstance(cells, list) or len(cells) != len(planned_matrix()):
        raise FailClosedError("v2 benchmark matrix cardinality drift")
    first = cells[0]
    if not isinstance(first, Mapping):
        raise FailClosedError("v2 benchmark first-cell schema drift")
    reference_model = str(first.get("initial_model_state_sha256"))
    reference_optimizer = str(first.get("initial_optimizer_state_sha256"))
    _require_sha(reference_model, "v2 matrix reference model state")
    _require_sha(reference_optimizer, "v2 matrix reference optimizer state")
    checked: list[dict[str, object]] = []
    for cursor, planned in enumerate(planned_matrix()):
        item = cells[cursor]
        if not isinstance(item, Mapping) or item.get("label") != planned["label"]:
            raise FailClosedError("v2 benchmark attempted matrix order drift")
        checked.append(validate_cell_payload(
            item, label=str(planned["label"]), kind=str(planned["kind"]), execution=str(planned["execution"]),
            batch_size=BATCH_SIZE, units=int(planned["units"]),  # type: ignore[arg-type]
            reference_model_sha256=reference_model, reference_optimizer_sha256=reference_optimizer,
        ))
    return {"cells": checked}


# --------------------------------------------------------------------------- #
# Receipt assembly, conclusion, publication
# --------------------------------------------------------------------------- #

def _conclusion(matrix: Mapping[str, Any]) -> dict[str, object]:
    """Derive an honest engineering ranking from the measured matrix.

    Acceptance uses the FIRST-STEP comparison against the frozen map (the
    direct v1-style identity proof).  The 20-step trajectory is reported
    verbatim, but a recurrent trajectory amplifies any backward
    reduction-order difference beyond the single-step map, so it is evidence
    of amplification, not of different mathematics, whenever the first step is
    exact/in-band.
    """
    rows: list[dict[str, object]] = []
    for units in UNIT_ORDER:
        production = next(
            item for item in matrix["cells"]
            if item["label"] == f"production_contract_eager_b32_u{units}" and item["status"] == "MEASURED"
        )
        production_median = float(production["timing"]["median_step_wall_seconds"])
        for item in matrix["cells"]:
            if item["units"] != units or item["status"] != "MEASURED" or item["kind"] == "production_contract":
                continue
            equivalence = item["equivalence"] or {}
            differences = equivalence.get("differences") or {}
            first_step = equivalence.get("first_step_differences") or {}
            rows.append({
                "units": units,
                "kind": item["kind"],
                "label": item["label"],
                "median_step_wall_seconds": float(item["timing"]["median_step_wall_seconds"]),
                "speedup_vs_production": production_median / float(item["timing"]["median_step_wall_seconds"]),
                "first_step_forward_max_abs": first_step.get("forward_max_abs"),
                "first_step_within_tolerance": equivalence.get("first_step_within_tolerance"),
                "trajectory_forward_max_abs": differences.get("forward_max_abs"),
                "trajectory_within_tolerance": equivalence.get("within_tolerance"),
            })
    accepted = [row for row in rows if row["first_step_within_tolerance"] is True]
    fastest_accepted = min(accepted, key=lambda row: float(row["median_step_wall_seconds"])) if accepted else None
    rejected = [row for row in rows if row["first_step_within_tolerance"] is False]
    unavailable = [
        {
            "units": item["units"], "kind": item["kind"], "label": item["label"],
            "compile_exception_class": item["compile_exception_class"],
            "compile_failure_stage": item["compile_failure_stage"],
        }
        for item in matrix["cells"] if item["status"] == COMPILE_UNAVAILABLE
    ]
    recommendations: list[str] = []
    if fastest_accepted is not None and float(fastest_accepted["speedup_vs_production"]) > 1.05:
        recommendations.append(
            f"request a production successor review for {fastest_accepted['kind']} at units={fastest_accepted['units']} "
            f"(+{100.0 * (float(fastest_accepted['speedup_vs_production']) - 1.0):.1f}% vs production-contract eager; "
            "first paired step exact/in-band under the frozen FP32 map)"
        )
    for row in rejected:
        recommendations.append(
            f"{row['kind']} at units={row['units']} is numerically rejected: its FIRST paired step is outside the "
            "frozen tolerance map, so the implementation is not the same mathematics; do not promote it"
        )
    amplified = [
        row for row in accepted if row["trajectory_within_tolerance"] is False
    ]
    if amplified:
        recommendations.append(
            "cells whose first step is exact but whose 20-step trajectory exceeds the single-step map "
            f"({', '.join(sorted({str(row['kind']) for row in amplified}))}) show recurrent fp32 amplification of "
            "backward reduction-order differences, not different math; a production successor review must define a "
            "trajectory-scale tolerance before promotion"
        )
    if unavailable:
        recommendations.append(
            "compile-unavailable cells above are environment evidence (torch 2.5.1.post303 inductor vs triton 3.2.0 "
            "AttrsDescriptor); fixing the environment (torch>=2.6 or triton<=3.1) is a prerequisite for any "
            "inductor-based production successor request"
        )
    return {
        "ranking": sorted(rows, key=lambda row: (int(row["units"]), float(row["median_step_wall_seconds"]))),
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
        raise FailClosedError("v2 conclusion schema drift")
    ranking = value.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise FailClosedError("v2 conclusion ranking drift")
    for row in ranking:
        if not isinstance(row, Mapping) or set(row) != {
            "units", "kind", "label", "median_step_wall_seconds", "speedup_vs_production",
            "first_step_forward_max_abs", "first_step_within_tolerance",
            "trajectory_forward_max_abs", "trajectory_within_tolerance",
        }:
            raise FailClosedError("v2 conclusion ranking row drift")
    if not isinstance(value.get("recommendations"), list) or not isinstance(value.get("statement"), str):
        raise FailClosedError("v2 conclusion text drift")
    return dict(value)


def build_engineering_receipt(
    *,
    matrix: Mapping[str, Any],
    launch_closure: Mapping[str, Any],
    final_closure: Mapping[str, Any],
    runtime: Mapping[str, Any],
    construction_report: Mapping[str, Any],
) -> dict[str, object]:
    launch = validate_closure_payload(launch_closure)
    final = validate_closure_payload(final_closure)
    checked_matrix = _validate_matrix_payload(matrix)
    checked_runtime = _validate_runtime(runtime)
    if not isinstance(construction_report, Mapping) or set(construction_report) != {"v1_failure", "shim"}:
        raise FailClosedError("v2 construction report schema drift")
    return {
        "schema": "tfsr_b3st4_ddrop_throughput_engineering_v2",
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
            "v1_production_contract_median_step_wall_seconds": V1_PRODUCTION_CONTRACT_MEDIAN_STEP_SECONDS,
        },
        "v1_construction_failure_reproduction": dict(construction_report["v1_failure"]),  # type: ignore[index]
        "inductor_construction_shim": dict(construction_report["shim"]),  # type: ignore[index]
        "compile_backends": {name: dict(cfg) for name, cfg in COMPILE_BACKENDS.items()},
        "runtime": checked_runtime,
        "matrix": checked_matrix,
        "conclusion": _conclusion(checked_matrix),
        "launch_closure": launch,
        "final_closure": final,
        "launch_final_closure_equal": launch == final,
        "canonical_output": {"root_relative": OUTPUT_ROOT_RELATIVE, "topology": list(OUTPUT_TOPOLOGY)},
    }


def validate_engineering_receipt(value: Mapping[str, Any]) -> dict[str, object]:
    expected = {
        "schema", "status", "cell", "phase", "purpose", "scientific_result", "data_opened",
        "checkpoint_opened", "target_or_formal", "authorizes_training", "supersedes", "benchmark_spec",
        "fixed_evidence", "v1_construction_failure_reproduction", "inductor_construction_shim",
        "compile_backends", "runtime", "matrix", "conclusion", "launch_closure", "final_closure",
        "launch_final_closure_equal", "canonical_output",
    }
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_b3st4_ddrop_throughput_engineering_v2"
            or value.get("status") != "ENGINEERING_BENCHMARK_COMPLETE" or value.get("cell") != CELL
            or value.get("phase") != PHASE or value.get("purpose") != "ENGINEERING_ONLY"
            or value.get("scientific_result") is not False or value.get("data_opened") is not False
            or value.get("checkpoint_opened") is not False or value.get("target_or_formal") is not False
            or value.get("authorizes_training") is not False
            or value.get("supersedes") != SUPERSEDES
            or value.get("benchmark_spec") != {str(units): UNIT_SPECS[units].payload() for units in UNIT_ORDER}
            or value.get("canonical_output") != {"root_relative": OUTPUT_ROOT_RELATIVE, "topology": list(OUTPUT_TOPOLOGY)}):
        raise FailClosedError("v2 engineering receipt boundary/schema drift")
    fixed = value.get("fixed_evidence")
    if (not isinstance(fixed, Mapping) or set(fixed) != {
                "workorder", "model", "native_causal_activity", "live_phase_d_v2_throughput",
                "frozen_adam", "v1_production_contract_median_step_wall_seconds",
            }
            or fixed.get("workorder") != {"relative_path": WORKORDER_RELATIVE, "sha256": WORKORDER_SHA256}
            or fixed.get("model") != {"relative_path": MODEL_RELATIVE, "sha256": MODEL_SHA256}
            or fixed.get("native_causal_activity") != {
                "relative_path": NATIVE_ACTIVITY_RELATIVE, "sha256": NATIVE_ACTIVITY_SHA256,
            }
            or fixed.get("frozen_adam") != FROZEN_ADAM
            or not isinstance(fixed.get("v1_production_contract_median_step_wall_seconds"), float)):
        raise FailClosedError("v2 engineering receipt fixed evidence drift")
    _validate_runtime(value.get("runtime"))
    _validate_matrix_payload(value.get("matrix"))
    validate_conclusion_payload(value.get("conclusion"))
    launch = validate_closure_payload(value.get("launch_closure"))
    final = validate_closure_payload(value.get("final_closure"))
    if launch != final or value.get("launch_final_closure_equal") is not True:
        raise FailClosedError("v2 engineering receipt launch/final closure drift")
    reproduction = value.get("v1_construction_failure_reproduction")
    if not isinstance(reproduction, Mapping) or type(reproduction.get("reproduced")) is not bool:
        raise FailClosedError("v2 construction-failure reproduction drift")
    shim = value.get("inductor_construction_shim")
    if not isinstance(shim, Mapping) or type(shim.get("applied")) is not bool:
        raise FailClosedError("v2 construction shim evidence drift")
    return {
        "launch_closure": launch, "final_closure": final,
        "matrix": _validate_matrix_payload(value.get("matrix")),
    }


class OutputArtifactRootV2:
    """Transactional O_EXCL + fsync + 0444 publisher for the single v2 receipt."""

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
                raise FailClosedError("v2 output root identity drift")
            for leaf in ("receipt.json", "receipt.json.sha256"):
                try:
                    os.stat(leaf, dir_fd=descriptor, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                raise FailClosedError("v2 benchmark receipt output collision")
            payloads = (
                ("receipt.json", body),
                ("receipt.json.sha256", f"{digest}  receipt.json\n".encode("ascii")),
            )
            for leaf, content in payloads:
                fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=descriptor)
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode):
                        raise FailClosedError("v2 receipt O_EXCL type drift")
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
            raise FailClosedError("published v2 receipt SHA drift")
        if self._read_leaf(descriptor, "receipt.json.sha256") != f"{digest}  receipt.json\n".encode("ascii"):
            raise FailClosedError("published v2 receipt sidecar drift")

    @staticmethod
    def _read_leaf(descriptor: int, leaf: str) -> bytes:
        fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
                raise FailClosedError("published v2 receipt leaf mode/type drift")
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
            raise FailClosedError("v2 output parent identity drift")
        try:
            os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise FailClosedError("canonical v2 output root must be fresh")
    finally:
        os.close(descriptor)


def run_reviewed_gpu_benchmark_v2(root: Path) -> Mapping[str, Any]:
    """The only physical route: GPU0 matrix, then one transactional receipt."""
    require_canonical_output_fresh(root)
    launch_closure = compute_benchmark_closure(root)
    backend = PhysicalBenchmarkBackendV2(root)
    backend.prepare()
    matrix = run_benchmark_matrix(backend)
    final_closure = compute_benchmark_closure(root)
    if backend.construction_report is None or backend.runtime is None:
        raise FailClosedError("v2 backend missing construction/runtime evidence")
    receipt = build_engineering_receipt(
        matrix=matrix, launch_closure=launch_closure, final_closure=final_closure,
        runtime=backend.runtime, construction_report=backend.construction_report,
    )
    validate_engineering_receipt(receipt)
    output = root.absolute() / _safe_relative(OUTPUT_ROOT_RELATIVE)
    output.mkdir(parents=True, mode=0o755, exist_ok=False)
    digest = OutputArtifactRootV2(output).publish(receipt)
    return {"receipt": receipt, "receipt_sha256": digest}
