"""Disposable two-GPU CRST-B4 paired-recipe resource smoke.

This is not a formal trainer.  It keeps FLAT and ROUTE in separate processes,
uses only source sampler batches, and writes receipts but never a learned
checkpoint, minival score, selection, or formal result root.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tfpd_exploration.src.h1_optimized_v2.cache import ROOT as H1_ROOT, build_or_load, validate_authority
from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
from tfpd_exploration.src.h1_optimized_v4.paired_train import batches, collate
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank

from .model import initialization_receipt, make_v2_unscaled_dot_localbalanced_pair, route_gate_gradient_l1, zero_gate_parity

OUT = H1_ROOT / "family_v1" / "crst_b4_localbalanced_splitarm_recipe_smoke_v1"
SEED = 42
MICRO, EFFECTIVE, LR, EMA_DECAY, DROP_P = 8, 32, 1e-4, .9995, .10
PAIRS, WARM_PAIRS, INFERENCE_WINDOWS, INFERENCE_BATCH = 8, 2, 96, 8
EPOCH_UPDATES, EPOCHS = 731, 12
MAX_WALL_SECONDS = 300


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_sha(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _shared_state_sha(flat: torch.nn.Module, route: torch.nn.Module) -> str:
    left, right = flat.state_dict(), route.state_dict()
    common = {name: left[name] for name in left.keys() & right.keys() if torch.equal(left[name], right[name])}
    if not common:
        raise RuntimeError("matched pair has no byte-identical shared state")
    return _state_sha(common)


def dropout_keep(n: int, epoch: int, batch_index: int, bank_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    seed = int.from_bytes(hashlib.sha256(f"{SEED}|keep|{epoch}|{batch_index}".encode()).digest()[:8], "little")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_keep = torch.rand((n, bank_mask.numel()), generator=generator) >= DROP_P
    keep = random_keep & bank_mask.detach().cpu().to(torch.bool).view(1, -1)
    if not bool(keep.any(dim=1).all()):
        raise RuntimeError("empty kept-unit row")
    return keep.to(device)


def set_epoch1_warmup(optimizer: torch.optim.Optimizer, step: int) -> float:
    """The candidate formal law: linear per-step warmup over epoch 1."""
    value = LR * min(1.0, step / EPOCH_UPDATES)
    for group in optimizer.param_groups:
        group["lr"] = value
    return value


def process_memory_mib() -> float | None:
    info = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    for line in info.stdout.splitlines():
        pid, memory = (item.strip() for item in line.split(",", 1))
        if int(pid) == os.getpid():
            return float(memory)
    return None


def _worker(arm: str, physical_gpu: int) -> None:
    out = OUT / "workers" / f"{arm}.json"
    ready = OUT / "barrier" / f"{arm}.ready"
    start = OUT / "barrier" / "START"
    # Cache integrity may deserialize minival arrays, but this worker only
    # indexes cache['train']; it creates no minival iterator or scorer.
    cache = build_or_load()
    recorded = json.loads((H1_ROOT / "source_cache_authority.json").read_text())
    validate_authority(cache, recorded)
    source_batches = batches(cache, 1)[:PAIRS]
    if len(source_batches) != PAIRS or any(len(starts) != EFFECTIVE for _, starts in source_batches):
        raise RuntimeError("first eight full-source sampler batches must be B=32")
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    device = torch.device("cuda:0")
    flat, route = make_v2_unscaled_dot_localbalanced_pair(seed=SEED)
    init = initialization_receipt(flat, route)
    shared_sha = _shared_state_sha(flat, route)
    selected = {"flat": flat, "route": route}[arm].to(device)
    if arm == "flat":
        del route
    else:
        del flat
    optimizer = torch.optim.AdamW(groups(selected), lr=LR)
    ema = DecoderEMA(selected, decay=EMA_DECAY)
    torch.cuda.reset_peak_memory_stats(device)

    first_name, first_starts = source_batches[0]
    first = cache["train"][first_name]
    x0, y0 = collate(first, first_starts, device)
    bank0 = H1Bank(*[first["bank"][key].to(device) for key in ("E0", "T", "unit_mask")])
    # g=0 checks happen while the discarded counterpart still exists in CPU
    # memory; only the selected arm goes to GPU.
    if arm == "route":
        flat_cpu, route_cpu = make_v2_unscaled_dot_localbalanced_pair(seed=SEED)
        parity = zero_gate_parity(flat_cpu, route_cpu, x0[:MICRO].cpu(), H1Bank(bank0.E0.cpu(), bank0.T.cpu(), bank0.unit_mask.cpu()))
        route_cpu.train(); route_cpu.zero_grad(set_to_none=True)
        keep_cpu = dropout_keep(MICRO, 1, 0, bank0.unit_mask.cpu(), torch.device("cpu"))
        F.mse_loss(route_cpu.forward_last(x0[:MICRO].cpu(), H1Bank(bank0.E0.cpu(), bank0.T.cpu(), bank0.unit_mask.cpu()), dropout_keep=keep_cpu), y0[:MICRO].cpu()).backward()
        gate_grad = route_gate_gradient_l1(route_cpu)
        del flat_cpu, route_cpu
    else:
        # Equivalent matched-law certificate constructed from the same fresh
        # seed in this independent process; ROUTE performs the active gradient.
        flat_cpu, route_cpu = make_v2_unscaled_dot_localbalanced_pair(seed=SEED)
        parity = zero_gate_parity(flat_cpu, route_cpu, x0[:MICRO].cpu(), H1Bank(bank0.E0.cpu(), bank0.T.cpu(), bank0.unit_mask.cpu()))
        gate_grad = None
        del flat_cpu, route_cpu
    if parity["max_abs_diff"] != 0.0:
        raise RuntimeError("g0 pair parity failed")
    if arm == "route" and (gate_grad is None or gate_grad <= 0.0):
        raise RuntimeError("g0 ROUTE gate gradient failed")

    ready.write_text(json.dumps({"arm": arm, "physical_gpu": physical_gpu, "shared_state_sha256": shared_sha}) + "\n")
    deadline = time.monotonic() + MAX_WALL_SECONDS
    while not start.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("coordinator did not release ready barrier")
        time.sleep(.02)
    losses, pair_seconds, lrs = [], [], []
    for batch_index, (name, starts) in enumerate(source_batches):
        row = cache["train"][name]
        x, y = collate(row, starts, device)
        bank = H1Bank(*[row["bank"][key].to(device) for key in ("E0", "T", "unit_mask")])
        keep = dropout_keep(len(x), 1, batch_index, bank.unit_mask, device)
        if bool((keep & ~bank.unit_mask.to(torch.bool).view(1, -1)).any()):
            raise RuntimeError("masked unit revived")
        torch.cuda.synchronize(device); started = time.perf_counter()
        selected.train(); optimizer.zero_grad(set_to_none=True)
        lr = set_epoch1_warmup(optimizer, batch_index + 1)
        parts = []
        for offset in range(0, EFFECTIVE, MICRO):
            loss = F.mse_loss(selected.forward_last(x[offset:offset + MICRO], bank, dropout_keep=keep[offset:offset + MICRO]), y[offset:offset + MICRO]) * (MICRO / EFFECTIVE)
            loss.backward(); parts.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(selected.parameters(), 1.0); optimizer.step(); ema.update_after_step(selected)
        torch.cuda.synchronize(device); elapsed = time.perf_counter() - started
        losses.append(float(np.mean(parts))); lrs.append(lr)
        if batch_index >= WARM_PAIRS:
            pair_seconds.append(elapsed)

    # Source-only B=8 EMA forwards over exactly 96 source windows.  This is a
    # forward-cost proxy, not a minival score or a reportable metric.
    source96 = source_batches[:INFERENCE_WINDOWS // EFFECTIVE]
    source_inference_seconds = []
    with torch.inference_mode():
        for name, starts in source96:
            row = cache["train"][name]
            x, _ = collate(row, starts, device)
            bank = H1Bank(*[row["bank"][key].to(device) for key in ("E0", "T", "unit_mask")])
            for offset in range(0, EFFECTIVE, INFERENCE_BATCH):
                torch.cuda.synchronize(device); began = time.perf_counter()
                prediction = ema.score_with_ema(selected, lambda candidate: candidate.forward_last(x[offset:offset + INFERENCE_BATCH], bank))
                torch.cuda.synchronize(device); source_inference_seconds.append(time.perf_counter() - began)
                if not bool(torch.isfinite(prediction).all()):
                    raise RuntimeError("non-finite source-only EMA forward")

    # In-memory only: source state is not saved to a retained checkpoint.
    torch.cuda.synchronize(device); ckpt_start = time.perf_counter()
    payload = {"model": selected.state_dict(), "optimizer": optimizer.state_dict(), "ema": ema.checkpoint_state(), "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(), "cuda": torch.cuda.get_rng_state_all()}}
    buffer = io.BytesIO(); torch.save(payload, buffer); torch.cuda.synchronize(device)
    checkpoint_seconds, checkpoint_bytes = time.perf_counter() - ckpt_start, len(buffer.getvalue())
    receipt = {
        "schema": "h1_crst_b4_localbalanced_splitarm_disposable_worker_v1", "status": "COMPLETE_DISPOSABLE_SOURCE_ONLY", "arm": arm, "physical_gpu": physical_gpu,
        "formal_training_launched": False, "formal_output_root_created": False, "retained_learned_candidate": False, "source_only": True,
        "cache_open_disclosure": "build_or_load plus validate_authority may deserialize cached minival arrays for integrity; worker used cache['train'] only and invoked no minival iterator/scoring/selection",
        "shared_state_sha256": shared_sha, "initialization": init, "g0_parity": parity, "route_gate_gradient_l1_at_g0": gate_grad,
        "recipe": {"seed": SEED, "microbatch": MICRO, "effective_batch": EFFECTIVE, "lr": LR, "warmup": "epoch1 per-step LR=1e-4*min(1, step/731)", "dropout": "stateless paired random p=.10 AND bank.unit_mask", "ema": EMA_DECAY, "first_full_source_batches": PAIRS},
        "sampler_batch_sessions": [name for name, _ in source_batches], "batch_sizes": [len(starts) for _, starts in source_batches], "losses": losses, "warmup_lrs": lrs,
        "timing": {"warm_pairs": WARM_PAIRS, "timed_pairs": len(pair_seconds), "seconds": pair_seconds, "mean_seconds": float(np.mean(pair_seconds)), "projected_train_only_seconds": float(np.mean(pair_seconds)) * EPOCH_UPDATES * EPOCHS, "source96_ema_b8_seconds": source_inference_seconds, "source96_ema_b8_total_seconds": float(sum(source_inference_seconds)), "selection_forward_proxy_seconds_per_2908": float(sum(source_inference_seconds)) * (2908 / INFERENCE_WINDOWS), "checkpoint_in_memory_serialization_seconds": checkpoint_seconds, "checkpoint_in_memory_bytes": checkpoint_bytes},
        "memory": {"peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20, "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20, "process_memory_mib": process_memory_mib()},
        "operator_sha256": {"familyformal_split_smoke.py": sha(Path(__file__)), "model.py": sha(Path(__file__).with_name("model.py"))},
    }
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _coordinator() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing nonempty disposable smoke root {OUT}")
    (OUT / "workers").mkdir(parents=True); (OUT / "barrier").mkdir()
    spec = {"flat": 0, "route": 1}
    processes = {}
    for arm, gpu in spec.items():
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1"}
        processes[arm] = subprocess.Popen(["taskset", "-c", "4-7", sys.executable, "-m", "tfpd_exploration.src.h1_family_v1.familyformal_split_smoke", "--worker", arm, "--physical-gpu", str(gpu)], cwd=Path.cwd(), env=env)
    deadline = time.monotonic() + MAX_WALL_SECONDS
    try:
        while not all((OUT / "barrier" / f"{arm}.ready").exists() for arm in spec):
            if any(process.poll() is not None for process in processes.values()):
                raise RuntimeError("worker failed before ready barrier")
            if time.monotonic() > deadline:
                raise TimeoutError("ready barrier exceeded five-minute hard bound")
            time.sleep(.05)
        ready = {arm: json.loads((OUT / "barrier" / f"{arm}.ready").read_text()) for arm in spec}
        if ready["flat"]["shared_state_sha256"] != ready["route"]["shared_state_sha256"]:
            raise RuntimeError("split processes constructed unequal shared initial state")
        (OUT / "barrier" / "START").write_text(json.dumps({"released_at_monotonic": time.monotonic()}) + "\n")
        for process in processes.values():
            remaining = max(1, int(deadline - time.monotonic()))
            process.wait(timeout=remaining)
        if any(process.returncode != 0 for process in processes.values()):
            raise RuntimeError("worker exited nonzero")
        workers = {arm: json.loads((OUT / "workers" / f"{arm}.json").read_text()) for arm in spec}
        if workers["flat"]["shared_state_sha256"] != workers["route"]["shared_state_sha256"]:
            raise RuntimeError("worker shared state receipt mismatch")
        report = {"schema": "h1_crst_b4_localbalanced_splitarm_disposable_smoke_v1", "status": "COMPLETE_DISPOSABLE_SOURCE_ONLY", "formal_training_launched": False, "formal_output_root_created": False, "retained_learned_candidate": False, "workers": {arm: str(OUT / "workers" / f"{arm}.json") for arm in spec}, "parallel_provisional": {"train_only_wall_seconds_max_arm": max(workers[a]["timing"]["projected_train_only_seconds"] for a in spec), "selection_forward_proxy_seconds_sum_arms": sum(workers[a]["timing"]["selection_forward_proxy_seconds_per_2908"] for a in spec), "checkpoint_serialization_seconds_sum_arms": sum(workers[a]["timing"]["checkpoint_in_memory_serialization_seconds"] for a in spec), "note": "parallel training wall uses max per-arm extrapolation; forward proxy and in-memory checkpoint serialization are reported separately, not a complete validation/export wall estimate."}, "old_smoke_disclosure": "The prior sequential disposable smoke used constant LR despite describing warmup; it was resource-only. This split smoke uses the frozen epoch1 per-step warmup law."}
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except BaseException:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--worker", choices=("flat", "route")); parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args()
    if args.worker: _worker(args.worker, args.physical_gpu)
    else: _coordinator()
