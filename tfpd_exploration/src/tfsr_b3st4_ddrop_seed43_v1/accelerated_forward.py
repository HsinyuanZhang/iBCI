"""Accelerated seed-43 forward: frozen TF-SR topology, jit-scripted recurrent core.

The topology is the frozen seed-42 ``TFSRDecoder`` imported, never copied.  The
50-step recurrent core is executed through the jit-scripted inner-step
construction from the frozen throughput-v2 engineering benchmark
(:func:`src.tfsr_b3st4_ddrop_v1.throughput_benchmark_v2.build_scripted_step`).
That builder is imported directly so this route provably executes the audited
construction; the route's tests pin the imported symbols' identity against the
benchmark module, which fails if anyone forks them.

Numerical contract (throughput-v2 receipt, disclosed in ``contract_43``):

* first paired training step: forward and loss bitwise-equal to the frozen
  eager forward (``forward_max_abs == 0.0``), gradients within the frozen
  ``1e-6`` band;
* 20-step trajectories diverge from the frozen build at fp32 reduction-order
  noise level -- the same property the purely-hoisted (mathematically
  identical) validation cell shows.  This is the disclosed build v2 property,
  not a trajectory-equivalence claim.

Like the benchmark modules, this file imports no torch at module level; the
authorized backend passes the runtime in.
"""

from __future__ import annotations

from typing import Any, NamedTuple

# Imported, never duplicated: these are the exact frozen v2 constructions the
# sealed throughput-v2 receipt measured and the exact frozen pre-loop stages
# (B3S identity, causal activity encoder, fused token MLP, Cell-D whole-unit
# mask law, population-mass channels) in the frozen order.
from src.tfsr_b3st4_ddrop_v1.throughput_benchmark_v2 import (
    EMBED_DIM,
    build_scripted_step,
    hoisted_precompute,
    scripted_step_loop,
)

__all__ = [
    "ACCELERATED_BUILD",
    "EMBED_DIM",
    "AcceleratedBatch",
    "accelerated_forward",
    "build_scripted_step",
    "build_scripted_step_binding",
    "hoisted_precompute",
    "scripted_step_loop",
]

ACCELERATED_BUILD = "v2_accelerated_jit_scripted_step"


class AcceleratedBatch(NamedTuple):
    """The three workload fields ``hoisted_precompute`` reads."""

    x: Any
    calib: Any
    normalized_t4: Any


def build_scripted_step_binding(torch: Any, model: Any) -> Any:
    """Bind the benchmark's jit-scripted inner step to the model's own tensors.

    ``build_scripted_step`` shares the fresh model's parameter tensors, so the
    optimizer, the epoch-boundary state digests, and every equivalence
    comparison continue to address the exact frozen tensors.
    """
    return build_scripted_step(torch, model)


def accelerated_forward(
    torch: Any,
    model: Any,
    scripted_step: Any,
    x: Any,
    calib: Any,
    normalized_t4: Any,
) -> Any:
    """Execute the exact frozen forward with the jit-scripted recurrent core.

    Typed validation is the frozen ``TFSRDecoder._validate`` (capability
    checks, shape/alignment/finiteness, shared device/dtype).  In train mode
    the Python-RNG dropout draw happens inside ``model._mask`` exactly once
    per call and sets ``last_dropout_p``/``last_unit_gain_mask``/
    ``last_unit_survivor_mask`` identically to the frozen forward.
    """
    model._validate(x, calib, normalized_t4)
    batch = AcceleratedBatch(x=x, calib=calib, normalized_t4=normalized_t4)
    tokens, mass = hoisted_precompute(torch, model, batch)
    return scripted_step_loop(torch, scripted_step, tokens, mass)
