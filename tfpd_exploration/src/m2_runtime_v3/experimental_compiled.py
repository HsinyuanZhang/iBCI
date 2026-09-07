"""Local-only exact torch.compile experiment for the M2 temporal tail.

This deliberately compiles neither bank handling, frontend caching nor public
host input.  It contains the full first three temporal blocks and the exact
last-row-Q final block, with fp32 operations unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .runtime import RuntimeV3Decoder


PAYLOAD = Path("tfpd_exploration/submissions/evalai_m2_small_trf_e_opt_v1/artifacts/m2_small_trf_s1_ema_e19.pkl")
TAGS = ["Run1_20201019", "Run2_20201019", "Run1_20201020", "Run2_20201020", "Run1_20201027", "Run2_20201027", "Run1_20201028"]


class ExactTemporalTail(nn.Module):
    """Static-shape [B=7,W=50,D] exact temporal path; no KV persistence."""
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.temporal = model.temporal
        self.final_norm = model.final_norm
        self.readout = model.readout

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = z + self.temporal.pe[: z.size(1)].unsqueeze(0).to(dtype=z.dtype)
        blocks = self.temporal.blocks
        for block in blocks[:-1]:
            hidden = block(hidden)
        block = blocks[-1]
        normed = block.norm1(hidden)
        attn = block.attn
        batch, width, dim = normed.shape
        query = F.linear(normed[:, -1:], attn.qkv.weight[:dim], attn.qkv.bias[:dim])
        query = query.view(batch, 1, attn.n_heads, attn.head_dim).transpose(1, 2)
        key_value = F.linear(normed, attn.qkv.weight[dim:], attn.qkv.bias[dim:])
        key_value = key_value.view(batch, width, 2, attn.n_heads, attn.head_dim)
        key, value = key_value.unbind(dim=2)
        out = F.scaled_dot_product_attention(query, key.transpose(1, 2), value.transpose(1, 2), dropout_p=0.0, is_causal=False)
        out = attn.proj(out.transpose(1, 2).contiguous().view(batch, 1, dim))
        last = hidden[:, -1:] + out
        last = last + block.ffn(block.norm2(last))
        return self.readout(self.final_norm(last))[:, 0, :]


def _stat(values: list[float]) -> dict[str, float]:
    return {"mean_ms": float(statistics.mean(values)), "p95_ms": float(np.percentile(values, 95)), "max_ms": float(max(values))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=128)
    parser.add_argument("--preflight", action="store_true", help="compile + one parity call only; no timing sweep")
    parser.add_argument("--public", action="store_true", help="measure complete public predict surface")
    args = parser.parse_args()
    if args.calls != 128:
        raise ValueError("bounded experimental protocol fixes calls=128")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    decoder = RuntimeV3Decoder(PAYLOAD, batch_size=7); decoder.reset(TAGS)
    engine = decoder._engine
    rows = np.random.default_rng(660).poisson(0.4, size=(160, 7, 96)).astype(np.float32)
    for row in rows[:32]:
        engine.advance(torch.from_numpy(row).unsqueeze(1))
    z = engine.frontend.detach().clone()
    eager = ExactTemporalTail(decoder.model).eval()
    with torch.inference_mode():
        reference = eager(z)
    cold_start = time.perf_counter()
    try:
        compiled = torch.compile(eager, fullgraph=True, dynamic=False)
        with torch.inference_mode():
            compiled_out = compiled(z)
    except Exception as exc:
        result = {"schema": "m2_runtime_v3_compile_probe_v1", "status": "compile_failed", "exception": repr(exc), "cold_compile_seconds": time.perf_counter()-cold_start, "cpu_affinity": sorted(os.sched_getaffinity(0))}
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n"); print(json.dumps(result)); return
    err = float(torch.max(torch.abs(reference-compiled_out)).item())
    if args.preflight:
        result = {"schema":"m2_runtime_v3_compile_probe_v1","status":"compiled_parity_pass" if err <= 1.0e-5 else "compiled_parity_fail","payload":str(PAYLOAD),"cpu_affinity":sorted(os.sched_getaffinity(0)),"torch":torch.__version__,"triton":__import__('triton').__version__,"shape":list(z.shape),"exact_max_abs":err,"cold_compile_seconds":time.perf_counter()-cold_start,"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"scope":"preflight only; no timed calls"}
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n"); print(json.dumps(result)); return
    if args.public:
        # Reset both streams after cold construction.  Only the exact temporal
        # tail is substituted; guards/frontend/host conversion stay public.
        eager_decoder = RuntimeV3Decoder(PAYLOAD, batch_size=7); eager_decoder.reset(TAGS)
        decoder.reset(TAGS); decoder._engine._last = compiled
        parity_rows = np.random.default_rng(661).poisson(0.4, size=(64, 7, 96)).astype(np.float32)
        actual_err = 0.0
        for row in parity_rows:
            actual_err = max(actual_err, float(np.max(np.abs(eager_decoder.predict(row) - decoder.predict(row)))))
        benchmark_rows = np.random.default_rng(662).poisson(0.4, size=(160, 7, 96)).astype(np.float32)
        eager_decoder.reset(TAGS); decoder.reset(TAGS); decoder._engine._last = compiled
        for row in benchmark_rows[:32]: eager_decoder.predict(row); decoder.predict(row)
        eager_ms, compiled_ms = [], []
        for row in benchmark_rows[32:]:
            t=time.perf_counter_ns(); eager_decoder.predict(row); eager_ms.append((time.perf_counter_ns()-t)/1e6)
            t=time.perf_counter_ns(); decoder.predict(row); compiled_ms.append((time.perf_counter_ns()-t)/1e6)
        result = {"schema":"m2_runtime_v3_compile_public_b7_v1","status":"completed","payload":str(PAYLOAD),"cpu_affinity":sorted(os.sched_getaffinity(0)),"torch":torch.__version__,"triton":__import__('triton').__version__,"threads":{"torch":torch.get_num_threads(),"omp":os.environ.get("OMP_NUM_THREADS"),"mkl":os.environ.get("MKL_NUM_THREADS"),"openblas":os.environ.get("OPENBLAS_NUM_THREADS")},"cold_compile_seconds":time.perf_counter()-cold_start,"cold_process_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"parity_raw_stream_calls":64,"public_prediction_max_abs":actual_err,"timed_calls":128,"warmup_calls":32,"eager_public_predict":_stat(eager_ms),"compiled_public_predict":_stat(compiled_ms),"scope":"complete RuntimeV3Decoder.predict host float32 bin through NumPy output; compilation only substitutes exact temporal tail"}
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n"); print(json.dumps(result)); return
    warm_eager, warm_compiled = [], []
    with torch.inference_mode():
        for _ in range(args.calls):
            t=time.perf_counter_ns(); eager(z); warm_eager.append((time.perf_counter_ns()-t)/1e6)
        for _ in range(args.calls):
            t=time.perf_counter_ns(); compiled(z); warm_compiled.append((time.perf_counter_ns()-t)/1e6)
    result = {"schema":"m2_runtime_v3_compile_probe_v1","status":"completed","payload":str(PAYLOAD),"cpu_affinity":sorted(os.sched_getaffinity(0)),"torch":torch.__version__,"triton":__import__('triton').__version__,"shape":list(z.shape),"exact_max_abs":err,"cold_compile_seconds":time.perf_counter()-cold_start,"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"calls":args.calls,"eager_temporal_tail":_stat(warm_eager),"compiled_temporal_tail":_stat(warm_compiled),"scope":"temporal tail only; frontend/banks/public host are excluded"}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n"); print(json.dumps(result))


if __name__ == "__main__": main()
