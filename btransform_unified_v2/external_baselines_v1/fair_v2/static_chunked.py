#!/usr/bin/env python3
"""Exact finite-receptive-field chunked inference for frozen static RIFT controls.

Unlike ``static_controls.predict``, this evaluates each causal native bin once.
A chunk carries the complete finite left receptive field, then discards its
prefix scores.  It is mathematically equivalent to the historical fixed
context windows and exists as a separate runner so the reference path remains
frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from static_controls import CKPTS, EXPECTED, fit, model_for, sha_arr, sha_file, stream_parity, transform
from data import load_task
from metrics import prediction_report

CONTEXT = {"m1": 100, "m2": 50, "h1": 300}


def receptive_field(model: torch.nn.Module) -> int:
    """Native rows that can affect one final score, including local conv."""
    conv = model.frontend.local_conv.conv
    if tuple(conv.kernel_size) != (5,) or tuple(conv.stride) != (1,) or tuple(conv.dilation) != (1,):
        raise RuntimeError(f"unexpected frontend local convolution: {conv}")
    windows = tuple(int(x) for x in model.temporal_config.windows)
    if len(windows) != int(model.temporal_config.layers) or any(x < 1 for x in windows):
        raise RuntimeError(f"invalid temporal windows: {windows}")
    return int(conv.kernel_size[0]) + sum(x - 1 for x in windows)


def score_scale(task: str) -> float:
    if task == "m2":
        from tfpd_exploration.src.m2_dual_track_v1 import plan
        return float(plan.BEHAVIOR_SCALE)
    if task == "h1":
        from btransform_unified_v1 import h1_config
        return float(h1_config.TARGET_MULTIPLIER)
    return 1.0


def full_input(item: dict, transformed_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pad = int(item["pad"])
    x = np.zeros_like(np.asarray(item["X"]), dtype=np.float32)
    if len(x) != pad + len(transformed_raw):
        raise RuntimeError("raw/padded input length drift")
    x[pad:] = transformed_raw
    valid = np.zeros(len(x), dtype=bool)
    valid[pad:] = True
    return x, valid


def chunked_native_scores(model: torch.nn.Module, x: np.ndarray, valid: np.ndarray, *, chunk_rows: int) -> np.ndarray:
    """Return one native score per row using finite-left-prefix chunks."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    r = receptive_field(model)
    out: list[np.ndarray] = []
    with torch.inference_mode():
        for begin in range(0, len(x), chunk_rows):
            end = min(len(x), begin + chunk_rows)
            left = max(0, begin - (r - 1))
            z = torch.from_numpy(np.ascontiguousarray(x[left:end])).unsqueeze(0)
            mask = torch.from_numpy(np.ascontiguousarray(valid[left:end])).unsqueeze(0)
            score = model.forward_scores(z, input_valid_mask=mask)[0, begin-left:].cpu().numpy()
            if len(score) != end - begin:
                raise RuntimeError("chunk did not return its complete suffix")
            out.append(score)
    return np.concatenate(out, axis=0)


def offline_native_at(model: torch.nn.Module, x: np.ndarray, valid: np.ndarray, endpoints: np.ndarray) -> np.ndarray:
    """Reference fixed-window forward scores for selected endpoint positions."""
    r = receptive_field(model)
    endpoints = np.asarray(endpoints, dtype=np.int64)
    if np.any(endpoints < r - 1) or np.any(endpoints >= len(x)):
        raise ValueError("reference endpoints outside a complete causal context")
    windows = np.stack([x[t-r+1:t+1] for t in endpoints])
    masks = np.stack([valid[t-r+1:t+1] for t in endpoints])
    with torch.inference_mode():
        return model(torch.from_numpy(windows), input_valid_mask=torch.from_numpy(masks)).cpu().numpy()


def parity_endpoints(item: dict, length: int, r: int, chunk_rows: int) -> np.ndarray:
    starts = np.asarray(item["starts"], dtype=np.int64)
    endpoints = starts + r - 1
    if len(endpoints) == 0 or endpoints.min() < r - 1 or endpoints.max() >= length:
        raise RuntimeError("evaluation starts do not map to complete native endpoints")
    candidates = [int(endpoints[0]), int(endpoints[-1]), int(endpoints[len(endpoints)//2])]
    for boundary in range(chunk_rows, length, chunk_rows):
        possible = endpoints[endpoints >= boundary]
        if len(possible):
            candidates.extend([int(possible[0]), int(possible[-1])])
    return np.asarray(sorted(set(candidates)), dtype=np.int64)


def run(task: str, dest: Path, *, chunk_rows: int, threads: int) -> dict:
    if dest.exists():
        raise FileExistsError(f"destination must be fresh: {dest}")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    data = load_task(task, include_evaluation=True)
    model, meta = model_for(task)
    r = receptive_field(model)
    if r != CONTEXT[task] or r != int(meta["context_bins"]):
        raise RuntimeError(f"receptive field {r} does not equal fixed context {CONTEXT[task]}")
    dest.mkdir(parents=True)
    arms, parity = {}, {}
    for arm in ("identity", "diag_z", "coral"):
        predictions, arm_parity = {}, {}
        for session, item in data["evaluation"].items():
            mapping = fit(data["train"], item, arm)
            raw = transform(np.asarray(item["X"])[int(item["pad"]):], mapping)
            x, valid = full_input(item, raw)
            native = chunked_native_scores(model, x, valid, chunk_rows=chunk_rows)
            endpoint = np.asarray(item["starts"], dtype=np.int64) + r - 1
            prediction = (native[endpoint] / score_scale(task)).astype(np.float32, copy=False)
            np.save(dest / f"{arm}_{session}_pred.npy", prediction)
            predictions[session] = prediction
            check = parity_endpoints(item, len(x), r, chunk_rows)
            reference = offline_native_at(model, x, valid, check)
            error = float(np.max(np.abs(native[check] - reference)))
            if error > 2e-5:
                raise RuntimeError(f"{arm}/{session}: chunk/reference max error {error}")
            arm_parity[session] = {
                "n_endpoints": int(len(check)), "endpoints": check.tolist(),
                "native_score_max_abs_error": error,
            }
            if arm == "identity":
                arm_parity[session]["stream_real_row_max_abs_error"] = stream_parity(task, model, item, raw)
        report = prediction_report(task, data["evaluation"], predictions)
        for session, row in report["per_session"].items():
            row["support_n"] = len(data["evaluation"][session]["support"])
            row["support_provenance"] = data["evaluation"][session]["support_provenance"]
        arms[arm] = report
        parity[arm] = arm_parity
    code = Path(__file__).read_bytes()
    receipt = {
        "schema": "fair_v2_static_rift_chunked_controls_v1",
        "task": task,
        "checkpoint": str(CKPTS[task]), "checkpoint_sha256": EXPECTED[task],
        "checkpoint_verified_sha256": sha_file(CKPTS[task]), "view": "EMA frozen", "device": "cpu",
        "inference_code_sha256": hashlib.sha256(code).hexdigest(),
        "reference_runner": "fair_v2/static_controls.py fixed-window predict",
        "receptive_field": r, "temporal_windows": list(model.temporal_config.windows),
        "chunk_rows": chunk_rows, "torch_threads": threads,
        "transform_contract": "pooled ALL held-in raw support; target raw support only; no Y; input prior to frontend.local_conv; literal zero padded rows",
        "coral": {"shrinkage": 0.1, "ridge": 0.001, "formula": "(x-mu_target) Ct^-1/2 Cs^1/2 + mu_source"},
        "parity": parity, "arms": arms,
    }
    (dest / "report.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    shutil.copy2(Path(__file__), dest / "inference_static_chunked_snapshot.py")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("m1", "m2", "h1"), required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.task, args.dest, chunk_rows=args.chunk_rows, threads=args.threads), indent=2, default=str))


if __name__ == "__main__":
    main()
