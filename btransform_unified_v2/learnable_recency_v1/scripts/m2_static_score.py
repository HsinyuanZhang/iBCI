#!/usr/bin/env python3
"""Score a static M2 run on EXT6.

Default scores the last-epoch EMA as a sidecar.  ``--pick`` scans all 24 EMA
checkpoints with the same earliest-max equal_session_mean rule as FULL.

The EXT6 reader is query-only: it opens X_store, target_store,
eligible_starts, and mapping metadata.  No support/calibration identity files
are allowed on this path.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
WS = ROOT.parent
V1 = WS / "btransform_unified_v1"
for p in (PKG / "src", ROOT, ROOT / "src", V1 / "src", WS, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from learnable_recency_v1.config import config_from_run_meta
from learnable_recency_v1.static_model import StaticLearnableRiftDecoder
import m2_static_train as train
from scripts.rift_v1 import m2_ext6_epoch_pick as frozen

RESULTS = PKG / "results"
SCHEMA = "m2_static_ext6_final_ema_score_v1"
PICK_SCHEMA = "m2_static_ext6_earliest_max_score_v1"
SELECTION_RULE = "earliest maximum finite unweighted equal_session_mean"


def read(p: Path) -> dict[str, Any]:
    x = json.loads(p.read_text())
    if not isinstance(x, dict):
        raise RuntimeError(f"object required: {p}")
    return x


def atom(p: Path, x: Mapping[str, Any]):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(x, indent=2, sort_keys=True, default=str) + "\n")
    q.replace(p)


def query_surface(cache: Path) -> dict[str, dict[str, Any]]:
    """Only concrete EXT6 query inputs; intentionally never verifies a support root."""
    receipt_path = cache / "official_heldout_query_banks.json"
    receipt = read(receipt_path)
    if (
        receipt.get("schema") != "m2_small_s1_visible_ext6_official_heldout_query_v1"
        or receipt.get("hidden_or_test_opened") is not False
    ):
        raise RuntimeError("EXT6 query receipt drift")
    declared = receipt.get("sessions")
    if not isinstance(declared, Mapping) or set(declared) != set(frozen.SIX):
        raise RuntimeError("EXT6 query receipt session roster drift")
    out = {}
    for session in frozen.SIX:
        d = cache / session
        x = d / "X_store.npy"
        y = d / "target_store.npy"
        s = d / "eligible_starts.npy"
        m = d / "mapping.json"
        if not all(p.is_file() for p in (x, y, s, m)):
            raise FileNotFoundError(d)
        xa = np.load(x, mmap_mode="r")
        ya = np.load(y, mmap_mode="r")
        sa = np.load(s, mmap_mode="r")
        mapping = read(m)
        expected = declared[session]
        if (
            xa.ndim != 2
            or xa.shape[1] != 96
            or sa.ndim != 1
            or len(sa) == 0
            or ya.shape != (len(sa), 2)
            or np.any(np.diff(sa) < 0)
            or int(sa[0]) < 0
            or int(sa[-1]) + train.CONTEXT > len(xa)
            or int(mapping.get("query_pad_bins", -1)) != 49
            or int(expected.get("query_start_trial", -1)) != 0
            or int(expected.get("window_count", -1)) != len(sa)
        ):
            raise RuntimeError(f"{session}: EXT6 query metadata/shape drift")
        out[session] = {
            "X": xa,
            "Y": ya,
            "starts": sa,
            "pad": 49,
            "dir": d,
            "hashes": {
                n: train.sha(d / n)
                for n in (
                    "X_store.npy",
                    "target_store.npy",
                    "eligible_starts.npy",
                    "mapping.json",
                )
            },
        }
    return out


def run(args):
    started = time.monotonic()
    run_dir = args.run_dir.resolve()
    meta = read(run_dir / "run_meta.json")
    is_smoke = meta.get("status") == "SMOKE"
    receipt = read(
        run_dir / ("smoke_receipt.json" if is_smoke else "train_receipt.json")
    )
    if (
        meta.get("schema") != train.SCHEMA
        or receipt.get("status") != "COMPLETED"
        or (is_smoke and not args.allow_smoke)
        or (not is_smoke and meta.get("status") != "FORMAL")
    ):
        raise RuntimeError(
            "completed formal static train run required (or explicit --allow-smoke)"
        )
    if is_smoke and (args.max_batches is None or args.max_batches < 1):
        raise RuntimeError("smoke score requires a positive --max-batches limit")
    if not is_smoke and args.allow_smoke:
        raise RuntimeError("--allow-smoke is valid only for a smoke run")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("--max-batches must be positive")
    ckpt = run_dir / ("epoch_001.pt" if is_smoke else "epoch_024.pt")
    state = torch.load(ckpt, map_location=args.device, weights_only=False)
    if state.get("schema") != train.CKPT_SCHEMA or (
        not is_smoke and (int(state.get("epoch", -1)) != 24 or state.get("smoke"))
    ):
        raise RuntimeError("final formal static checkpoint required")
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    cfg = config_from_run_meta(meta, "m2")
    model = StaticLearnableRiftDecoder("m2", cfg, context_bins=50, seed=42).to(device)
    model.temporal.set_attention_backend("local")
    model.load_state_dict(state["raw_state_dict"], strict=True)
    shadow = state.get("ema", {}).get("shadow")
    named = dict(model.named_parameters())
    if not isinstance(shadow, Mapping) or set(shadow) != set(named):
        raise RuntimeError("EMA state does not match static model")
    with torch.no_grad():
        for n, p in named.items():
            p.copy_(shadow[n].to(p.device, p.dtype))
    surface = query_surface(args.query_cache.resolve())
    report = train.score_surface(model, surface, device, limit=args.max_batches)
    out = {
        "schema": SCHEMA,
        "status": "SMOKE_COMPLETED" if is_smoke else "COMPLETED",
        "formal_claim": False if is_smoke else True,
        "partial": bool(args.max_batches is not None),
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": train.sha(ckpt),
        "view": "EMA",
        "selection": {
            "epoch": int(state["epoch"]),
            "rule": (
                "smoke checkpoint; no held-out tuning"
                if is_smoke
                else "prespecified final; no held-out tuning"
            ),
        },
        "query_only": True,
        "query_data_hashes": {s: v["hashes"] for s, v in surface.items()},
        "metrics": report,
        "official_test_used": False,
        "runtime_seconds": time.monotonic() - started,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    if is_smoke and (out["status"] != "SMOKE_COMPLETED" or not out["partial"]):
        raise RuntimeError("smoke score receipt must be marked partial")
    atom(args.dest.resolve() / "score_receipt.json", out)
    return out


def apply_ema(model, state, device):
    model.load_state_dict(state["raw_state_dict"], strict=True)
    shadow = state.get("ema", {}).get("shadow")
    named = dict(model.named_parameters())
    if not isinstance(shadow, Mapping) or set(shadow) != set(named):
        raise RuntimeError("EMA state does not match static model")
    with torch.no_grad():
        for n, p in named.items():
            p.copy_(shadow[n].to(p.device, p.dtype))
    model.to(device)
    model.eval()


def pick(args):
    started = time.monotonic()
    run_dir = args.run_dir.resolve()
    meta = read(run_dir / "run_meta.json")
    receipt = read(run_dir / "train_receipt.json")
    if (
        meta.get("schema") != train.SCHEMA
        or receipt.get("status") != "COMPLETED"
        or meta.get("status") != "FORMAL"
    ):
        raise RuntimeError("completed formal static train run required")
    if args.allow_smoke or args.max_batches is not None:
        raise RuntimeError("pick is formal full-surface only")
    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    cfg = config_from_run_meta(meta, "m2")
    model = StaticLearnableRiftDecoder("m2", cfg, context_bins=50, seed=42).to(device)
    model.temporal.set_attention_backend("local")
    surface = query_surface(args.query_cache.resolve())
    progress_path = dest / "pick_progress.json"
    progress = read(progress_path) if progress_path.is_file() else {"completed": {}}
    curve = progress.get("completed", {})
    for epoch in range(1, train.EPOCHS + 1):
        if str(epoch) in curve:
            continue
        ckpt = run_dir / f"epoch_{epoch:03d}.pt"
        state = torch.load(ckpt, map_location=args.device, weights_only=False)
        if state.get("schema") != train.CKPT_SCHEMA or int(state.get("epoch", -1)) != epoch:
            raise RuntimeError(f"static checkpoint drift at epoch {epoch}")
        apply_ema(model, state, device)
        report = train.score_surface(model, surface, device)
        curve[str(epoch)] = {**report, "checkpoint_sha256": train.sha(ckpt)}
        atom(progress_path, {"completed": curve})
    values = {epoch: float(curve[str(epoch)]["equal_session_mean"]) for epoch in range(1, train.EPOCHS + 1)}
    best = max(range(1, train.EPOCHS + 1), key=lambda epoch: (values[epoch], -epoch))
    out = {
        "schema": PICK_SCHEMA,
        "status": "COMPLETED",
        "formal_claim": True,
        "partial": False,
        "run_dir": str(run_dir),
        "view": "EMA",
        "selection": {
            "epoch": best,
            "equal_session_mean": values[best],
            "rule": SELECTION_RULE,
        },
        "ema_by_epoch": curve,
        "query_only": True,
        "query_data_hashes": {s: v["hashes"] for s, v in surface.items()},
        "official_test_used": False,
        "runtime_seconds": time.monotonic() - started,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    atom(dest / "score_receipt.json", out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--dest", type=Path)
    p.add_argument("--query-cache", type=Path, default=frozen.QUERY_CACHE)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--max-batches", type=int)
    p.add_argument("--allow-smoke", action="store_true")
    p.add_argument("--pick", action="store_true")
    a = p.parse_args()
    if a.max_batches is not None and a.max_batches < 1:
        p.error("--max-batches must be positive")
    if a.pick and a.dest is None:
        a.dest = RESULTS / "selection_m2_static_ext6_earliest_max_s42"
    elif a.dest is None:
        a.dest = RESULTS / "selection_m2_static_final_ema_ext6_s42"
    print(json.dumps(pick(a) if a.pick else run(a), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
