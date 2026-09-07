"""H1 §6 identity-usage control at L=100. Waits for a free GPU unless --cpu-smoke.

Arms (same 8-slot CausalPE4, same windows, only identity usage changes):
  a_concat700  — cache E0 [176,700] concat (current family usage)
  b_joined36   — C2 pre-pool 32 + H-C 4 concat (CAL-3b)
  c_add_tail   — SPINT-like: add E0[:, -100:] onto the 100-bin spike window
  d_e0_zero    — E0 zeros, carrier only
  e_permute    — E0 columns reversed (structure-break)

Train: cache train query_starts, last 100 bins, 20y MSE, 12 ep cosine.
Score: minival all eval_mask ends, native /20. No EvalAI. New result root only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path("/home/xinyuan/Work_host/SPINT/btransform_unified_v1/src")))

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/xinyuan/Work_host/SPINT")
CACHE = ROOT / "tfpd_exploration/results/decoder_validation_v2/20260905_190000/h1/source_cache.pt"
OUT = ROOT / "btransform_unified_v1/results/h1_sec6_fivearm_l100_v1"
ARMS = ("a_concat700", "b_joined36", "c_add_tail", "d_e0_zero", "e_permute")
WINDOW = 100
UNITS = 176
OUT_DIM = 7
SCALE = 20.0


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, np.float64)
    target = np.asarray(target, np.float64)
    denom = np.square(target - target.mean(0)).sum()
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(1.0 - np.square(pred - target).sum() / denom)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_free(min_free_mib: int = 18000) -> str | None:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return None
    for line in out.strip().splitlines():
        index, used, total = [p.strip() for p in line.split(",")]
        if int(total) - int(used) >= min_free_mib:
            return index
    return None


def _windows(neural: np.ndarray, ends: np.ndarray) -> np.ndarray:
    hist = np.zeros((len(ends), WINDOW, UNITS), dtype=np.float32)
    for i, end in enumerate(ends):
        start = int(end) - WINDOW + 1
        if start >= 0:
            hist[i] = neural[start : int(end) + 1]
        else:
            hist[i, -int(end) - 1 :] = neural[: int(end) + 1]
    return hist


def _split_arrays(cache: dict, split: str, use_query_starts: bool) -> dict[str, dict]:
    rows = {}
    for session, row in cache[split].items():
        neural = np.ascontiguousarray(row["neural"], dtype=np.float32)
        vel = np.ascontiguousarray(row["velocity"], dtype=np.float32)
        if use_query_starts:
            starts = np.asarray(row["query_starts"], dtype=np.int64)
            ends = starts + 699
            ends = ends[(ends >= 0) & (ends < len(neural))]
            mask = np.asarray(row["eval_mask"], dtype=np.bool_)
            ends = ends[mask[ends]] if mask.size else ends
        else:
            ends = np.flatnonzero(np.asarray(row["eval_mask"], dtype=np.bool_)).astype(np.int64)
        rows[session] = {
            "X": _windows(neural, ends),
            "y": vel[ends],
            "E0": np.ascontiguousarray(row["bank"]["E0"].cpu().numpy() if hasattr(row["bank"]["E0"], "cpu") else row["bank"]["E0"], dtype=np.float32),
            "T": np.ascontiguousarray(row["bank"]["T"].cpu().numpy() if hasattr(row["bank"]["T"], "cpu") else row["bank"]["T"], dtype=np.float32),
            "mask": np.ascontiguousarray(row["bank"]["unit_mask"].cpu().numpy() if hasattr(row["bank"]["unit_mask"], "cpu") else row["bank"]["unit_mask"], dtype=np.bool_),
        }
    return rows


def _joined36(e0: np.ndarray, carrier: np.ndarray) -> np.ndarray:
    # Honest proxy only if we cannot rebuild C2 pre-pool: take first 32 of E0 + T.
    # Marked as PROXY in the receipt. Real 36-d is filled when activity is supplied.
    return np.concatenate([e0[:, :32], carrier], axis=1).astype(np.float32)


def _make_bank(arm: str, row: dict, device: torch.device):
    from btransform_unified_v1.bank import TaskBank, array_sha256

    e0 = row["E0"]
    carrier = row["T"]
    if arm == "a_concat700":
        ident = e0
    elif arm == "b_joined36":
        ident = _joined36(e0, carrier)
    elif arm == "c_add_tail":
        ident = e0  # still stored; add happens on X
    elif arm == "d_e0_zero":
        ident = np.zeros_like(e0)
    elif arm == "e_permute":
        ident = e0[:, ::-1].copy()
    else:
        raise ValueError(arm)
    e0_dim = 36 if arm == "b_joined36" else 700
    if ident.shape[1] != e0_dim:
        ident = ident[:, :e0_dim]
    return TaskBank(
        session_id="h1",
        E0=ident,
        carrier=carrier,
        unit_mask=row["mask"],
        X_store=row["X"][:1],
        target_store=row["y"][:1],
        window_ids=np.zeros(1, dtype=np.int64),
        calibration_meta={
            "shape": tuple(ident.shape),
            "trial_count": 3,
            "estimator": arm,
            "array_sha256": array_sha256(ident),
            "budget": 3,
        },
    )


def _model(arm: str, device: torch.device):
    from btransform_unified_v1.model import BTransformerUnifiedDecoder

    e0_dim = 36 if arm == "b_joined36" else 700
    return BTransformerUnifiedDecoder({
        "task": "h1-l100",
        "window": WINDOW,
        "prefix": 0,
        "units": UNITS,
        "e0_dim": e0_dim,
        "carrier_dim": 4,
        "out_dim": OUT_DIM,
    }, seed=42).to(device)


def _prepare_x(arm: str, x: np.ndarray, e0: np.ndarray) -> np.ndarray:
    if arm != "c_add_tail":
        return x
    return x + e0[:, -WINDOW:].T[None]  # broadcast [1,100,176] += [100,176] via [176,100].T


def train_arm(arm: str, train: dict, val: dict, device: torch.device, epochs: int, smoke: bool) -> dict:
    from btransform_unified_v1.ema import DecoderEMA
    from btransform_unified_v1.schedule import warmup_cosine_lr

    model = _model(arm, device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    ema = DecoderEMA(model, decay=0.9995)
    sessions = list(train)
    rng = np.random.default_rng(42)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = sessions.copy()
        rng.shuffle(order)
        losses = []
        step = 0
        total_steps = max(1, sum(len(train[s]["X"]) for s in sessions) // 32)
        warmup = max(1, total_steps)
        for session in order:
            row = train[session]
            bank = _make_bank(arm, row, device)
            X = _prepare_x(arm, row["X"], row["E0"])
            y = row["y"] * SCALE
            idx = rng.permutation(len(X))
            if smoke:
                idx = idx[:64]
            for offset in range(0, len(idx), 32):
                take = idx[offset : offset + 32]
                xb = torch.from_numpy(X[take]).to(device)
                yb = torch.from_numpy(y[take]).to(device)
                pred = model(xb, bank)
                loss = F.mse_loss(pred, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                for group in opt.param_groups:
                    group["lr"] = warmup_cosine_lr(step, max(total_steps * epochs, 2), warmup)
                opt.step()
                ema.update_after_step(model)
                losses.append(float(loss.detach().cpu()))
                step += 1
                if smoke and step >= 4:
                    break
            if smoke and step >= 4:
                break
        scores = _score_arm(arm, model, ema, val, device, smoke=smoke)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **scores})
        print(json.dumps({"arm": arm, **history[-1]}), flush=True)
        if smoke:
            break
    return {"arm": arm, "history": history, "final": history[-1]}


@torch.no_grad()
def _score_arm(arm, model, ema, val, device, *, smoke: bool) -> dict:
    raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if ema.n_updates > 0:
        ema.apply_to(model)
    eval_model = model
    eval_model.eval()
    preds, ys, names = [], [], []
    for session, row in val.items():
        bank = _make_bank(arm, row, device)
        X = _prepare_x(arm, row["X"], row["E0"])
        y = row["y"]
        take = np.arange(min(len(X), 64 if smoke else len(X)))
        for offset in range(0, len(take), 32):
            sl = take[offset : offset + 32]
            xb = torch.from_numpy(X[sl]).to(device)
            out = eval_model(xb, bank).cpu().numpy() / SCALE
            preds.append(out)
            ys.append(y[sl])
            names.extend([session] * len(sl))
    try:
        pred = np.concatenate(preds)
        y = np.concatenate(ys)
        names = np.asarray(names)
        per = {s: _r2(pred[names == s], y[names == s]) for s in sorted(set(names))}
        return {
            "pooled_r2": _r2(pred, y),
            "equal_session_mean_r2": float(np.mean(list(per.values()))),
            "per_session_r2": per,
            "n": int(len(y)),
        }
    finally:
        model.load_state_dict(raw, strict=True)
        model.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--wait-gpu", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    if not args.cpu_smoke:
        uuid = gpu_free()
        while args.wait_gpu and uuid is None:
            print("waiting for a free GPU (>=18 GiB)", flush=True)
            time.sleep(60)
            uuid = gpu_free()
        if uuid is None:
            raise RuntimeError("no free GPU; rerun with --wait-gpu or --cpu-smoke")
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
        device = torch.device("cuda:0")
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    train = _split_arrays(cache, "train", use_query_starts=True)
    val = _split_arrays(cache, "minival", use_query_starts=False)
    report = {
        "schema": "h1_sec6_fivearm_l100_v1",
        "window": WINDOW,
        "b_joined36_note": "PROXY: first 32 of fused E0 + T, not C2 pre-pool 36-d. Rebuild when trialized activity is wired.",
        "cache_sha256": _sha(CACHE),
        "device": str(device),
        "smoke": args.cpu_smoke,
        "arms": {},
    }
    (OUT / "input_authority.json").write_text(json.dumps({"status": "RUNNING", **report}, indent=2) + "\n")
    for arm in ARMS:
        report["arms"][arm] = train_arm(arm, train, val, device, args.epochs, args.cpu_smoke)
        (OUT / "live.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    report["status"] = "COMPLETE_SMOKE" if args.cpu_smoke else "COMPLETE_DEV_ONLY"
    (OUT / "receipt.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: v["final"] for k, v in report["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
