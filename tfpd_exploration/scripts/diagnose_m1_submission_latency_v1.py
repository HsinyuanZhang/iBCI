"""Read-only payload/runtime comparison; never edits the submission package.

This is a local timing/parity diagnostic, not a Falcon container acceptance or
R2 evaluation. B4 uses three real source sessions and an independent replay of
the first session in lane four; no outer-session observations are opened.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from falcon_challenge.config import FalconConfig, FalconTask
from tfpd_exploration.src.m1_optimized_v2.model import build
from tfpd_exploration.src.m1_optimized_v2 import plan
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2.streaming import CurrentQueryStream
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import M1Bank


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    return {f"p{q}_ms": float(np.percentile(values, q)) for q in (50, 95, 99)} | {
        "mean_ms": float(np.mean(values)), "max_ms": float(np.max(values))}


class DiagnosticHeterogeneousStream(CurrentQueryStream):
    """Explicit experimental batch-bank adapter, NOT production API acceptance.

    Each row has its own bank/mask. Common unit column layout is 64 entries.
    The underlying arithmetic already broadcasts per-row banks; dedicated
    production lifecycle/session-mutation tests are outside this diagnostic.
    """
    def __init__(self, model, banks, tags):
        self.row_tags = tuple(tags)
        bank = M1Bank(torch.stack([b.E0 for b in banks]), torch.stack([b.T for b in banks]),
                      torch.stack([b.unit_mask for b in banks]))
        super().__init__(model, bank, task="m1", session_id=tuple(tags), unit_ids=tuple(range(64)),
                         batch_size=len(tags))


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    package = Path("tfpd_exploration/submissions/evalai_m1_runtime_v3_selected_t_v1")
    runtime = package / "m1_trf_falcon_decoder.py"
    payload_path = package / "artifacts/m1_optimized_v2_t_ema_e6.pkl"
    initial_hashes = {str(p): digest(p) for p in (runtime, payload_path, package / "decode.py")}
    spec = importlib.util.spec_from_file_location("diagnostic_m1_submitted_runtime", runtime)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    payload = mod.load_payload(payload_path)
    receipt_path = package / "artifacts/payload.receipt.json"
    if not receipt_path.exists():
        receipt_path = package / "artifacts/payload.receipt.inherited.json"
    receipt = json.loads(receipt_path.read_text())
    assert digest(payload_path) == receipt["payload_sha256"]
    model = build("flat").eval()
    model.load_state_dict({k: torch.as_tensor(v, dtype=torch.float32) for k, v in payload["state_dict"].items()}, strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    cache_path = plan.RESULT_ROOT / "m1_optimized_v2_source_runtime_cache.npz"
    cache_receipt = json.loads(cache_path.with_suffix(".receipt.json").read_text())
    assert digest(cache_path) == cache_receipt["npz_sha256"]
    cache = np.load(cache_path, allow_pickle=False)
    config = FalconConfig(task=FalconTask.m1)
    source_names = [*plan.SOURCE_SESSIONS, plan.SOURCE_SESSIONS[0]]
    names = source_names[:args.batch]
    tags = [name.removeprefix("ses-") for name in names]
    # Use genuine full tag stems, so the submitted reset path runs unchanged.
    stems = [Path(receipt["banks"]["records"][name]["file"]).stem for name in names]
    n_steps = args.warmup + args.calls
    observations = np.stack([cache[f"raw_neural/{name}"][99:99+n_steps] for name in names], axis=1)
    assert observations.shape == (n_steps, args.batch, 64)
    banks = [M1Bank(torch.as_tensor(payload["bank_by_dataset_tag"][tag]["E0"], dtype=torch.float32),
                    torch.as_tensor(payload["bank_by_dataset_tag"][tag]["T"], dtype=torch.float32),
                    torch.as_tensor(payload["bank_by_dataset_tag"][tag]["unit_mask"], dtype=torch.bool)) for tag in tags]
    rows, predictions = [], {}
    for mode in ("submitted_serial_full100", "existing_cache_serial", "diagnostic_heterogeneous_cache"):
        start = time.perf_counter_ns()
        if mode == "submitted_serial_full100":
            engine = mod.M1TemporalFalconDecoder(config, str(payload_path), batch_size=args.batch)
            engine.reset(stems)
            invoke = engine.predict
        elif mode == "existing_cache_serial":
            engines = [CurrentQueryStream(model, bank, task="m1", session_id=tag, unit_ids=tuple(range(64)))
                       for bank, tag in zip(banks, tags)]
            def invoke(x):
                return np.concatenate([e.predict(x[i:i+1]) for i, e in enumerate(engines)], axis=0)
        else:
            engine = DiagnosticHeterogeneousStream(model, banks, tags)
            invoke = engine.predict
        construction_ms = (time.perf_counter_ns() - start) / 1e6
        durations, outputs, startup = [], [], []
        for i, x in enumerate(observations):
            start = time.perf_counter_ns()
            result = invoke(x)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            assert result.shape == (args.batch, 16) and result.dtype == np.float32 and np.isfinite(result).all()
            outputs.append(result.copy())
            if i < 8:
                startup.append(elapsed)
            if i >= args.warmup:
                durations.append(elapsed)
        predictions[mode] = np.stack(outputs)
        row = {"mode": mode, "batch": args.batch, "calls": args.calls, "public_predict_whole_batch": quantiles(durations),
               "constructor_reset_ms": construction_ms, "startup_first8_ms": startup}
        rows.append(row)
        print(json.dumps(row), flush=True)
    oracle = predictions["submitted_serial_full100"]
    parity = {}
    for mode, values in predictions.items():
        error = float(np.max(np.abs(values-oracle)))
        parity[mode] = {"max_native_abs_error": error, "all_startup_and_rollover_calls": n_steps,
                        "atol_1e_5_pass": bool(error <= 1e-5)}
        assert error <= 1e-5, (mode, error)
    assert initial_hashes == {p: digest(p) for p in initial_hashes}, "user package changed during diagnostic"
    result = {"schema": "m1_submitted_runtime_local_latency_diagnostic_v1", "not_official_latency": True,
              "not_container_acceptance": True, "not_r2_evaluation": True, "threads": args.threads,
              "torch": torch.__version__, "affinity": sorted(os.sched_getaffinity(0)), "calls": args.calls,
              "warmup_calls": args.warmup, "batch": args.batch, "session_rows": names,
              "fourth_lane": "independent replay of source 20120926, not a fourth distinct bank",
              "inputs": "continuous source neural bins; original 99-bin storage padding removed; no cyclic replay",
              "source_cache_sha256": cache_receipt["npz_sha256"], "package_hashes": initial_hashes,
              "script_sha256": digest(__file__), "parity": parity, "records": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, choices=(1, 4), default=4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--calls", type=int, default=128)
    p.add_argument("--warmup", type=int, default=128)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
