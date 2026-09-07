"""Actual public-return-path CPU timings for H1/M1 current-query decoders.

Run each replay in a fresh process under the coordinator's single-flight CPU
lease. Every measured arm requires a trained state unless
--initialization-only is explicit; unmeasured arms need no checkpoint.
No R2 or learned-performance claim is derived from a timing stream.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .streaming import CurrentQueryStream, ExactFullWindowStream
from .validate_stream import state_sha


class FullWindowAPI:
    """Matched observation/return boundary, without any frontend/memory reuse."""
    def __init__(self, model, bank, *, task, session_id, unit_ids):
        self.model, self.bank, self.task = model, bank, task
        self.window, self.n = model.cfg.window, len(unit_ids)
        self.divisor = 20. if task == "h1" else 1.
        self.reset()

    def reset(self, *, history=None):
        p = next(self.model.parameters())
        self.raw = torch.zeros(1, self.window, self.n, device=p.device, dtype=p.dtype)
        if history is not None:
            source = torch.as_tensor(history, device=p.device, dtype=p.dtype)
            count = min(self.window, source.shape[1])
            if count:
                self.raw[:, -count:] = source[:, -count:]

    @torch.no_grad()
    def predict(self, observation):
        p = next(self.model.parameters())
        x = torch.as_tensor(observation, device=p.device, dtype=p.dtype)
        if x.shape != (1, self.n) or not bool(torch.isfinite(x).all()):
            raise ValueError("finite [1,N] observation required")
        self.raw = torch.cat((self.raw[:, 1:], x[:, None]), dim=1)
        result = self.model.forward_last(self.raw, self.bank) / self.divisor
        if not bool(torch.isfinite(result).all()):
            raise ValueError("nonfinite prediction")
        return result.detach().cpu().numpy().astype(np.float32, copy=True)


def _quantiles(values):
    x = np.asarray(values, dtype=np.float64)
    return {"p50_ms": float(np.percentile(x, 50)), "p95_ms": float(np.percentile(x, 95)),
            "p99_ms": float(np.percentile(x, 99)), "max_ms": float(x.max()), "mean_ms": float(x.mean())}


def _inputs(task):
    if task == "h1":
        from tfpd_exploration.src.h1_optimized_v2.cache import CACHE
        from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
        cache = torch.load(CACHE, map_location="cpu", weights_only=False)
        name = sorted(cache["train"])[0]; row = cache["train"][name]
        return row["neural"], H1Bank(**row["bank"]), name, tuple(range(176)), {
            "source_nwb_sha256": row["sha256"], "source": "H1 real source-training continuous bins",
            "input_contract": "raw binned neural counts; fixed scalar32 inside model", "padding_stripped": 0,
        }
    from tfpd_exploration.src.m1_optimized_v2 import plan
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import M1Bank
    path = plan.RESULT_ROOT / "m1_optimized_v2_source_runtime_cache.npz"
    receipt = json.loads(path.with_suffix(".receipt.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["npz_sha256"]:
        raise RuntimeError("source runtime cache hash mismatch")
    source = np.load(path, allow_pickle=False); name = plan.SOURCE_SESSIONS[0]
    bank = M1Bank(torch.from_numpy(source[f"bank_e0/{name}"]), torch.from_numpy(source[f"bank_t/{name}"]),
                  torch.from_numpy(source[f"bank_unit_mask/{name}"]))
    provenance = np.load(plan.RESULT_ROOT / "rSyn3-refit-v1.source-only.provenance-supplement.npz", allow_pickle=False)
    ids = tuple(provenance[f"nwb_unit_ids_in_rate_column_order/{name}"].tolist())
    x = source[f"raw_neural/{name}"]
    if not np.all(x[:99] == 0):
        raise ValueError("M1 dataset startup padding contract changed")
    return x[99:], bank, name, ids, {"source": "M1 real source-training continuous bins",
        "cache_sha256": receipt["npz_sha256"], "carrier_sha256": receipt["carrier_npz_sha256"],
        "input_contract": "raw binned neural counts; no smoothing", "padding_stripped": 99}


def _required_arms(modes):
    return {"full" if mode in {"full", "full_exact"} else "query" for mode in modes}


def _validate_state_arguments(full_state, query_state, initialization_only, modes):
    required = _required_arms(modes)
    supplied = {"full": full_state, "query": query_state}
    if not initialization_only and any(supplied[arm] is None for arm in required):
        raise ValueError("audited standalone model state required for each measured arm")
    if initialization_only and (full_state is not None or query_state is not None):
        raise ValueError("do not mix initialization and trained artifact timing")
    return required


def _models(task, full_state, query_state, initialization_only, h1_revision="v2_activity32", modes=("full", "query_cached")):
    required = _validate_state_arguments(full_state, query_state, initialization_only, modes)
    if task == "h1":
        if h1_revision == "v5_logage":
            from tfpd_exploration.src.h1_optimized_v5.model import make_matched_pair
        elif h1_revision == "v4_signed":
            from tfpd_exploration.src.h1_optimized_v4.model import make_matched_pair
        elif h1_revision == "v3_centered":
            from tfpd_exploration.src.h1_optimized_v3.model import make_matched_pair
        else:
            from tfpd_exploration.src.h1_optimized_v2.model import make_matched_pair
        full, query = make_matched_pair(activity_scale=1. if h1_revision in {"v4_signed", "v5_logage"} else 32.)
    else:
        from tfpd_exploration.src.m1_optimized_v2.model import build
        from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import M1TemporalFlatDecoder
        full, query = M1TemporalFlatDecoder(seed=42), build("flat")
    provenance = {}
    for name, model, path in (("full", full, full_state), ("query", query, query_state)):
        if name not in required:
            continue
        if path is not None:
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False), strict=True)
            provenance[name] = {"file": str(path), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        else:
            provenance[name] = {"weights": "matched seed42 initialization; not a trained result"}
        model.eval()
        provenance[name]["state_sha256"] = state_sha(model)
    return full, query, provenance


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    x, bank, name, roster, input_receipt = _inputs(args.task)
    x = np.ascontiguousarray(x, dtype=np.float32)
    full, query, weights = _models(args.task, args.full_state, args.query_state, args.initialization_only, args.h1_revision, args.modes)
    if args.task == "h1" and args.h1_revision in {"v4_signed", "v5_logage"}:
        input_receipt["input_contract"] = "raw binned neural counts, scale1; signed carrier-conditioned mixing frontend"
    if min(args.calls, args.warmup_calls, args.threads) < 1:
        raise ValueError("positive calls, warmup_calls and threads required")
    if len(x) < full.cfg.window + args.warmup_calls + args.calls:
        raise ValueError("continuous source too short; no cyclic timing replay enabled")
    records = []
    for mode in args.modes:
        model = full if mode in {"full", "full_exact"} else query
        cls = CurrentQueryStream if mode == "query_cached" else ExactFullWindowStream if mode == "full_exact" else FullWindowAPI
        t0 = time.perf_counter_ns()
        engine = cls(model, bank, task=args.task, session_id=name, unit_ids=roster)
        construct_ms = (time.perf_counter_ns() - t0) / 1e6
        startup = []
        for step in range(8):
            t0 = time.perf_counter_ns(); result = engine.predict(x[step:step+1]); startup.append((time.perf_counter_ns()-t0)/1e6)
            if not np.isfinite(result).all():
                raise ValueError("nonfinite startup result")
        t0 = time.perf_counter_ns(); engine.reset(history=x[None, :model.cfg.window]); reset_ms = (time.perf_counter_ns()-t0)/1e6
        # Bulk-loading the actual complete raw history is equivalent to arriving
        # at this window after startup. Warmup uses genuine public API calls.
        for step in range(model.cfg.window, model.cfg.window+args.warmup_calls):
            engine.predict(x[step:step+1])
        times = []
        for step in range(model.cfg.window+args.warmup_calls, model.cfg.window+args.warmup_calls+args.calls):
            t0 = time.perf_counter_ns(); result = engine.predict(x[step:step+1]); times.append((time.perf_counter_ns()-t0)/1e6)
            if result.dtype != np.float32 or result.shape[0] != 1 or not np.isfinite(result).all():
                raise ValueError("public output contract failed")
        row = {"mode": mode, "calls": args.calls, "batch": 1, "public_predict_wall": _quantiles(times),
               "construct_including_empty_window_cache_ms": construct_ms, "startup_first8_ms": startup,
               "reset_full_actual_history_ms": reset_ms, "full_history_warmup_calls": args.warmup_calls,
               "cache_state_bytes": getattr(engine, "state_bytes", None)}
        records.append(row); print(json.dumps(row, sort_keys=True), flush=True)
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.lower().startswith("model name")), "unknown")
    code_paths = [Path(__file__), Path(__file__).with_name("core.py"), Path(__file__).with_name("streaming.py")]
    from tfpd_exploration.src.two_mainlines_long_v1.latency_opt_v2 import FrontendWindowCache
    code_paths.append(Path(inspect.getfile(FrontendWindowCache)))
    for measured in (full, query):
        for module in (measured, measured.frontend, measured.temporal):
            code_paths.append(Path(inspect.getfile(type(module))))
    report = {"schema": "shared_current_query_public_api_cpu_benchmark_v2", "task": args.task,
              "operator_revision": args.h1_revision if args.task == "h1" else "m1_optimized_v2",
              "initialization_only": args.initialization_only, "session": name, "inputs": input_receipt,
              "weights": weights, "operator_code_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
              "measurement": "one [1,N] observation through buffer/frontend/temporal/readout/unit conversion/host float32 return",
              "not_an_r2_evaluation": True, "data_stream": "continuous actual source bins, no cyclic replay",
              "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
              "affinity": sorted(os.sched_getaffinity(0)), "cpu": cpu, "platform": platform.platform(), "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--task", choices=("h1", "m1"), required=True)
    p.add_argument("--h1-revision", choices=("v2_activity32", "v3_centered", "v4_signed", "v5_logage"), default="v2_activity32")
    p.add_argument("--full-state", type=Path); p.add_argument("--query-state", type=Path)
    p.add_argument("--initialization-only", action="store_true"); p.add_argument("--calls", type=int, default=2048)
    p.add_argument("--warmup-calls", type=int, default=128)
    p.add_argument("--threads", type=int, default=2); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--modes", nargs="+", choices=("full", "full_exact", "query_uncached", "query_cached"), default=("full", "full_exact", "query_uncached", "query_cached"))
    main(p.parse_args())
