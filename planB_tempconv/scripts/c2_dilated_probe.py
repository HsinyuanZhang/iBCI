#!/usr/bin/env python3
"""§3 Step 3: dilated temporal-conv ablation (seeded).

Step 1 (capacity probe, c1) showed widening the channel-mix FC does NOT help
(hidden 32->64->128: sat R2 0.168 -> 0.152 -> 0.137). That gate skipped the
*width* ladder, but it did not directly test adding **temporal-conv depth** /
growing the receptive field -- a different axis. This script closes that gap:

  d1       dilations=[1]      RF=9   (current baseline, ~9 frames / 180 ms)
  d3_plain dilations=[1,1,1]  RF=25  (more conv layers, no dilation growth)
  d3_dil   dilations=[1,2,4]  RF=57  (dilated stack; covers the full 50-frame window)

All at hidden=32, k=9, causal padding, ref_raw regime, CORAL-diag cross-day.
If d3_dil barely beats d1 (delta < noise floor ~0.04), temporal RF is also not
the bottleneck -> reinforces "go to FiLM (§4)".

Plan evidence going in: ridge N_HIST 7->50 only gained +0.006, so a longer
linear window barely helps; expected payoff here is small. Params: d1=4130,
d3=6050 (dilation is free; only the extra conv layers cost).
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _env
from _env import (held_in, held_out_sessions, continuous_folds, eval_r2,
                  N_FOLDS, N_CH, N_OUT, RESULTS, FIGURES, fit_coral_diag)
from _env import resolve_session_key, load_session, HELD_OUT_SESSION_KEYS
from lib_planb import held_in_stats, make_train_windows
from models.tcn_student import DepthwiseTemporalConvStudent

W = 50
K = 9
EPOCHS = 50
BS = 2048
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BIN_S = 0.02
BUDGETS = [200, 400, 800, 1600, 3200, 10 ** 9]
CONFIGS = {
    "d1":       ([1], False),
    "d3_plain": ([1, 1, 1], False),
    "d3_dil":   ([1, 2, 4], False),
    "d3_res":   ([1, 2, 4], True),     # dilated stack + per-layer residual
}


def rf_of(dilations, k=K):
    return 1 + (k - 1) * sum(dilations)


def train_tcn_seeded(smooth_hi, vel_hi, dilations, residual, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    T = len(vel_hi)
    x_t = torch.tensor(np.ascontiguousarray(smooth_hi, dtype=np.float32), device=DEVICE)
    vel_t = torch.tensor(np.ascontiguousarray(vel_hi, dtype=np.float32), device=DEVICE)
    idx = torch.arange(W - 1, T, device=DEVICE)
    tcn = DepthwiseTemporalConvStudent(N_CH, W, K, hidden=32, n_out=N_OUT,
                                       dilations=dilations, residual=residual).to(DEVICE)
    opt = torch.optim.Adam(tcn.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.MSELoss()
    for ep in range(EPOCHS):
        tcn.train()
        g = torch.Generator(device=DEVICE).manual_seed(seed * 7919 + ep)
        perm = idx[torch.randperm(len(idx), generator=g, device=DEVICE)]
        for s in range(0, len(perm), BS):
            b = perm[s:s + BS]
            loss = lossf(tcn(make_train_windows(x_t, b, W)), vel_t[b])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(tcn.parameters(), 1.0)
            opt.step()
    # in-sample (held-in) R2: underfit (~0) -> dead units / opt failure; high -> overfit
    tcn.eval()
    with torch.no_grad():
        pred_hi = torch.cat([tcn(make_train_windows(x_t, idx[s:s + BS], W))
                             for s in range(0, len(idx), BS)]).cpu().numpy()
    train_r2 = eval_r2(vel_hi[idx.cpu().numpy()], pred_hi)
    return tcn, train_r2


@torch.no_grad()
def predict_r2(tcn, x_seg_np, y_seg_np):
    from lib_planb import predict_frozen_segment
    pred, yt = predict_frozen_segment(tcn, x_seg_np, y_seg_np, W, device=DEVICE)
    if len(yt) == 0:
        return float("nan")
    return eval_r2(yt, pred)


def subsample_abs(mask, n_frames, rng):
    idx = np.flatnonzero(mask)
    if n_frames >= len(idx):
        return mask
    pick = rng.choice(idx, size=n_frames, replace=False)
    out = np.zeros_like(mask); out[pick] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 sessions")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = list(range(1, args.seeds + 1))

    smooth_hi, vel_hi, _ = held_in()
    smooth_hi = np.asarray(smooth_hi, np.float32)
    vel_hi = np.asarray(vel_hi, np.float32)
    mu, sigma, cov = held_in_stats(smooth_hi)
    vmu, vsd = vel_hi.mean(0), vel_hi.std(0) + 1e-6
    vel_hi = (vel_hi - vmu) / vsd

    hout = held_out_sessions()
    keys = HELD_OUT_SESSION_KEYS[:2] if args.quick else HELD_OUT_SESSION_KEYS

    rows = []
    t0 = time.time()
    done = 0
    n_combo = len(CONFIGS) * len(seeds)
    summary = {}
    for name, (dil, res) in CONFIGS.items():
        sat_by_seed = []
        for seed in seeds:
            tcn, train_r2 = train_tcn_seeded(smooth_hi, vel_hi, dil, res, seed)
            nparam = sum(p.numel() for p in tcn.parameters())
            acc = {b: [] for b in BUDGETS}
            for key in keys:
                stem = resolve_session_key(hout, key)
                smooth, vel, _ = load_session(hout[stem])
                smooth = np.asarray(smooth, np.float32)
                vel = (np.asarray(vel, np.float32) - vmu) / vsd
                folds = continuous_folds(len(vel), N_FOLDS)
                for b in BUDGETS:
                    rng = np.random.default_rng(hash(key) % 2 ** 31)
                    r2s = []
                    for calib_m, test_m in folds:
                        calib_m = subsample_abs(calib_m, b, rng)
                        a_coral, b_coral = fit_coral_diag(smooth[calib_m], mu, sigma)
                        x_seg = smooth[test_m] * a_coral + b_coral
                        y_seg = vel[test_m]
                        r2 = predict_r2(tcn, x_seg, y_seg)
                        if r2 == r2:
                            r2s.append(r2)
                    m = float(np.mean(r2s)) if r2s else float("nan")
                    acc[b].append(m)
                    rows.append({"config": name, "rf": rf_of(dil), "seed": seed,
                                 "nparam": nparam, "session": key,
                                 "budget_frames": b,
                                 "calib_seconds": round(min(b, folds[0][0].sum()) * BIN_S, 1),
                                 "ref_raw_r2": m})
            sat = float(np.mean(acc[BUDGETS[-1]])) if acc[BUDGETS[-1]] else float("nan")
            sat_by_seed.append(sat)
            done += 1
            print(f"[{done}/{n_combo}] {name:8s} RF={rf_of(dil):2d} seed={seed} "
                  f"params={nparam:6d} trainR2={train_r2:.3f} ref_raw sat={sat:.3f}  "
                  f"({time.time()-t0:.0f}s)")
        arr = np.array(sat_by_seed)
        summary[name] = arr
        print(f"  -> {name:8s}: mean={arr.mean():.3f} std={arr.std():.3f} "
              f"[{', '.join(f'{x:.3f}' for x in arr)}]")

    out_csv = RESULTS / "c2_dilated_probe.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {out_csv}")

    print("\n=== Dilated probe summary (ref_raw saturated, mean over seeds) ===")
    for name in CONFIGS:
        arr = summary[name]
        print(f"  {name:8s} RF={rf_of(CONFIGS[name][0]):2d}: sat R2 = {arr.mean():.3f} +/- {arr.std():.3f}")
    base = summary["d1"].mean()
    print("\nDeltas vs d1 (RF=9):")
    for name in ("d3_plain", "d3_dil", "d3_res"):
        d = summary[name].mean() - base
        print(f"  d1 -> {name}: delta = {d:+.4f}")
    noise = summary["d1"].std()
    print(f"\nNoise floor (d1, {len(seeds)} seeds): std = {noise:.4f}")


if __name__ == "__main__":
    main()
