#!/usr/bin/env python3
"""Bounded source-only B32 CUDA/CPU feasibility profile for DANDI688 v2.

This is deliberately separate from ``run.py``: it exercises the formal
training step without creating a selection-eligible checkpoint or changing
the frozen training recipe.  It never loads final data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parents[1]
for entry in (WORKSPACE, PACKAGE.parent, PACKAGE.parent / "src",
              PACKAGE.parent / "learnable_recency_v1" / "src",
              WORKSPACE / "btransform_unified_v1" / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np
import torch
from torch import nn

from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
from btransform_unified_v1.schedule import warmup_cosine_lr
from learnable_recency_v1.wrap import apply_group_lrs

from dandi688_bench_v2.common import SCHEMA, RECIPE, PairedSampler, atomic_json, fit_source_stats, fresh_directory, load_records, source_hashes
from dandi688_bench_v2.data import padded_windows
from dandi688_bench_v2.model import DandiRiftDecoder
from dandi688_bench_v2.training import calibration_inputs


ARMS = ("full_pretrain", "full_frozen", "activity", "raw_set")


def _finite_grads(model: torch.nn.Module) -> bool:
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters())


def _one_arm(records, stats: dict, representation: str, arm_name: str, device: torch.device,
             steps: int, frozen_encoder=None) -> tuple[dict, object | None]:
    arm = "full" if arm_name.startswith("full_") else arm_name
    if arm_name == "full_frozen":
        if frozen_encoder is None:
            raise ValueError("full_frozen requires the preceding bounded full_pretrain encoder")
        for parameter in frozen_encoder.parameters():
            parameter.requires_grad_(False)
    model = DandiRiftDecoder(arm, seed=42, encoder=(frozen_encoder if arm_name == "full_frozen" else None),
                             freeze_encoder=arm_name == "full_frozen").to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.optimizer_param_groups(lr=RECIPE["lr_peak"],
                                                                 weight_decay=RECIPE["weight_decay"]),
                                  lr=RECIPE["lr_peak"], betas=tuple(RECIPE["betas"]), eps=RECIPE["eps"])
    ema = DecoderEMA(model, decay=RECIPE["ema_decay"])
    calibration = {r.session_id: calibration_inputs(r, arm, stats, device) for r in records}
    y_mean = torch.tensor(stats["velocity_mean"], device=device, dtype=torch.float32)
    y_std = torch.tensor(stats["velocity_std"], device=device, dtype=torch.float32)
    sampler = PairedSampler(records, 42, batch=RECIPE["batch"], updates_per_segment=steps)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    losses, elapsed, finite = [], [], True
    for batch_index, (record, endpoints) in enumerate(sampler.segment()):
        x, timevalid, unitmask = padded_windows(record, endpoints)
        tensor_x = torch.from_numpy(x).to(device)
        base_mask = torch.from_numpy(unitmask).to(device).expand(len(endpoints), -1)
        keep_generator = torch.Generator(device="cpu").manual_seed(unit_dropout_seed(42, 1, batch_index))
        keep = whole_unit_dropout(base_mask, p=RECIPE["whole_unit_dropout"], generator=keep_generator)
        target = (torch.from_numpy(record.velocity[endpoints]).to(device) - y_mean) / y_std
        step = batch_index + 1
        apply_group_lrs(optimizer, warmup_cosine_lr(step, total_steps=RECIPE["segments"] * RECIPE["updates_per_segment"],
                                                     warmup_steps=RECIPE["warmup_updates"], peak=RECIPE["lr_peak"],
                                                     min_factor=RECIPE["lr_min_factor"]))
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with amp:
            prediction = model(tensor_x, **calibration[record.session_id], unit_mask=base_mask,
                               dropout_keep=keep, input_valid_mask=torch.from_numpy(timevalid).to(device))
            loss = nn.functional.mse_loss(prediction.float(), target.float())
        finite = finite and bool(torch.isfinite(loss))
        loss.backward()
        finite = finite and _finite_grads(model)
        nn.utils.clip_grad_norm_(model.parameters(), RECIPE["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        ema.update_after_step(model)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed.append(time.perf_counter() - started)
        losses.append(float(loss.detach().cpu()))
    # Exclude the first allocation/JIT-adjacent step from throughput when possible.
    timed = elapsed[1:] if len(elapsed) > 1 else elapsed
    row = {"arm": arm_name, "representation": representation, "steps": steps, "batch": RECIPE["batch"],
            "context_bins": RECIPE["context_bins"], "max_units": 100,
            "precision": "bfloat16_autocast" if device.type == "cuda" else "fp32",
            "finite_loss_and_gradients": finite, "losses": losses,
            "step_seconds": elapsed, "median_step_seconds": float(np.median(timed)),
            "steps_per_second": float(len(timed) / sum(timed)),
            "max_allocated_mb": (float(torch.cuda.max_memory_allocated(device)) / 2**20
                                 if device.type == "cuda" else None),
            "parameters_total": sum(p.numel() for p in model.parameters()),
            "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "sampler": sampler.receipt()}
    # Keep only the profile-only pretrained encoder long enough to verify the
    # Full-frozen path immediately afterwards.
    return row, (model.encoder if arm_name == "full_pretrain" else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=3, choices=range(1, 9))
    parser.add_argument("--representations", nargs="+", default=("sua", "pmua"), choices=("sua", "pmua"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    dest = fresh_directory(args.dest)
    started = time.time()
    rows = []
    for representation in args.representations:
        records = load_records(args.cache, representation, "train")
        stats = fit_source_stats(records)
        pretrained_encoder = None
        for arm in ARMS:
            row, exported_encoder = _one_arm(records, stats, representation, arm, device, args.steps,
                                             frozen_encoder=pretrained_encoder)
            rows.append(row)
            if exported_encoder is not None:
                pretrained_encoder = exported_encoder
            print(json.dumps({k: row[k] for k in ("representation", "arm", "steps_per_second", "max_allocated_mb", "finite_loss_and_gradients")}), flush=True)
    receipt = {"schema": SCHEMA + "_gpu_feasibility_profile", "status": "PROFILE_ONLY_INELIGIBLE_FOR_SELECTION",
               "source_only": True, "final_sessions_opened": 0, "device": str(device), "recipe": RECIPE,
               "steps_per_arm": args.steps, "rows": rows, "elapsed_seconds": time.time() - started,
               "code_hashes": source_hashes()}
    atomic_json(dest / "receipt.json", receipt)
    print(json.dumps({"receipt": str(dest / "receipt.json"), "rows": len(rows), "final_sessions_opened": 0}), flush=True)


if __name__ == "__main__":
    main()
