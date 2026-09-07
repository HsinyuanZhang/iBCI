"""Small, fixed CPU diagnostic for an immutable paired-training checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .cache import ROOT as CACHE_ROOT
from .cache import build_or_load, validate_authority
from .c2_reference import _operator_code_authority
from .model import make_matched_pair
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank


W = 700


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _r2(p: np.ndarray, y: np.ndarray) -> float:
    p, y = p.astype(np.float64), y.astype(np.float64)
    return float(1.0 - np.square(p - y).sum() / np.square(y - y.mean(axis=0, keepdims=True)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--count", type=int, default=12)
    args = ap.parse_args()
    if not 1 <= args.count <= 15:
        raise ValueError("diagnostic count must be in [1,15]")
    root = CACHE_ROOT / args.attempt
    out = root / f"raw_epoch_{args.epoch:03d}_fixed_train_diagnostic.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic: {out}")
    cache = build_or_load()
    recorded = json.loads((CACHE_ROOT / "source_cache_authority.json").read_text())
    validate_authority(cache, recorded)
    # Predeclared, source-train-only sample: lexicographic sessions, then each
    # session's chronological first endpoints, round-robin until count.
    picks: list[tuple[str, int]] = []
    names = sorted(cache["train"])
    offset = 0
    while len(picks) < args.count:
        for name in names:
            starts = cache["train"][name]["query_starts"]
            if offset < len(starts) and len(picks) < args.count:
                picks.append((name, int(starts[offset])))
        offset += 1
    results = {"schema": "h1_optimized_v2_raw_checkpoint_fixed_train_diagnostic_v1", "attempt": args.attempt, "epoch": args.epoch, "device": "cpu", "subset_rule": "lexicographic source-train sessions, chronological endpoints, round-robin", "items": [{"session": n, "start": s, "end": s + W - 1} for n, s in picks], "input_authority": recorded, "operator_code_sha256": _operator_code_authority(), "arms": {}}
    for arm in ("full", "t"):
        model = make_matched_pair(activity_scale=32.0)[0 if arm == "full" else 1].cpu().eval()
        ckpt = root / f"{arm}_epoch_{args.epoch:03d}.pt"
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        pred, perturbed, target = [], [], []
        with torch.inference_mode():
            for session, start in picks:
                row = cache["train"][session]
                x = torch.as_tensor(row["neural"][start : start + W], dtype=torch.float32)[None]
                bank = H1Bank(row["bank"]["E0"], row["bank"]["T"], row["bank"]["unit_mask"])
                pred.append((model.forward_last(x, bank)[0] / 20.0).numpy())
                # Fixed input perturbation: adding 0.1 to every observed bin.
                perturbed.append((model.forward_last(x + 0.1, bank)[0] / 20.0).numpy())
                target.append(np.asarray(row["velocity"][start + W - 1], dtype=np.float32))
        p, q, y = np.asarray(pred), np.asarray(perturbed), np.asarray(target)
        zero = np.zeros_like(y)
        mean = np.broadcast_to(y.mean(axis=0, keepdims=True), y.shape)
        results["arms"][arm] = {"checkpoint": str(ckpt), "checkpoint_sha256": _sha(ckpt), "prediction_std": float(p.std()), "target_std": float(y.std()), "prediction_mean": float(p.mean()), "target_mean": float(y.mean()), "mean_abs_response_to_plus_0p1_input": float(np.abs(q - p).mean()), "native_r2_subset": _r2(p, y), "zero_native_r2_subset": _r2(zero, y), "train_subset_mean_native_r2": _r2(mean, y)}
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results["arms"], sort_keys=True))


if __name__ == "__main__":
    main()
