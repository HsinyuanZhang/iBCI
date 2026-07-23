#!/usr/bin/env python3
"""§3 Step 1: capacity-elasticity probe (seeded).

For each (seed, hidden) pair, train a causal TCN on RAW held-in (ref_raw regime,
the meaningful one) and eval all 6 held-out sessions x budgets with CORAL-diag.

Two purposes, served in one run:
  1. NOISE FLOOR: hidden=32 across N seeds -> run-to-run std of saturated R2.
     Needed because the §3 capacity gate uses a ~0.01 delta threshold; unseeded
     b3 training showed a ~0.015 swing, which could swamp the signal.
  2. CAPACITY GATE: hidden 32 -> 64 -> 128 at MATCHED seed. If the matched-seed
     delta is < ~0.01, capacity is NOT binding -> skip the width ladder, go to
     FiLM (§4).

ref_raw only (skips own_zscore to halve eval time; the gate is on ref_raw sat).
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
CEILINGS = {"ridge-CORAL": 0.114, "sym-pad TCN (1 run)": 0.20,
            "SPINT GF-FSU": 0.26, "LOO oracle": 0.268}


def train_tcn_seeded(smooth_hi, vel_hi, hidden, seed):
    """Train a causal TCN on raw held-in with a fixed seed -> reproducible."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    T = len(vel_hi)
    x_t = torch.tensor(np.ascontiguousarray(smooth_hi, dtype=np.float32), device=DEVICE)
    vel_t = torch.tensor(np.ascontiguousarray(vel_hi, dtype=np.float32), device=DEVICE)
    idx = torch.arange(W - 1, T, device=DEVICE)
    tcn = DepthwiseTemporalConvStudent(N_CH, W, K, hidden=hidden, n_out=N_OUT).to(DEVICE)
    opt = torch.optim.Adam(tcn.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.MSELoss()
    for ep in range(EPOCHS):
        tcn.train()
        g = torch.Generator(device=DEVICE).manual_seed(seed * 7919 + ep)
        perm = idx[torch.randperm(len(idx), generator=g, device=DEVICE)]
        tot = 0.0
        for s in range(0, len(perm), BS):
            b = perm[s:s + BS]
            loss = lossf(tcn(make_train_windows(x_t, b, W)), vel_t[b])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(tcn.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
    return tcn


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
    out = np.zeros_like(mask)
    out[pick] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 sessions")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hiddens", type=str, default="32,64,128")
    args = ap.parse_args()
    hiddens = [int(h) for h in args.hiddens.split(",")]
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
    n_combo = len(seeds) * len(hiddens)
    done = 0
    for hidden in hiddens:
        sat_by_seed = []
        for seed in seeds:
            tcn = train_tcn_seeded(smooth_hi, vel_hi, hidden, seed)
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
                        if r2 == r2:  # filter nan (nan != nan)
                            r2s.append(r2)
                    m = float(np.mean(r2s)) if r2s else float("nan")
                    acc[b].append(m)
                    rows.append({"seed": seed, "hidden": hidden, "nparam": nparam,
                                 "session": key, "budget_frames": b,
                                 "calib_seconds": round(min(b, folds[0][0].sum()) * BIN_S, 1),
                                 "ref_raw_r2": m})
            sat = float(np.mean(acc[BUDGETS[-1]])) if acc[BUDGETS[-1]] else float("nan")
            sat_by_seed.append(sat)
            done += 1
            print(f"[{done}/{n_combo}] hidden={hidden:3d} seed={seed} params={nparam:6d} "
                  f"ref_raw sat={sat:.3f}  ({time.time()-t0:.0f}s)")

        arr = np.array(sat_by_seed)
        print(f"  -> hidden={hidden:3d}: mean={arr.mean():.3f} std={arr.std():.3f} "
              f"[{', '.join(f'{x:.3f}' for x in arr)}]")

    out_csv = RESULTS / "c1_capacity_probe.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {out_csv}")

    # --- summary: per-hidden mean over seeds, and noise floor at hidden=32 ---
    print("\n=== Capacity probe summary (ref_raw saturated, mean over seeds) ===")
    summary = {}
    for hidden in hiddens:
        sats = [float(r["ref_raw_r2"]) for r in rows
                if r["hidden"] == hidden and r["budget_frames"] == 10 ** 9]
        # group by seed to get per-seed session-mean, then stats over seeds
        per_seed = []
        for seed in seeds:
            v = [float(r["ref_raw_r2"]) for r in rows
                 if r["hidden"] == hidden and r["seed"] == seed and r["budget_frames"] == 10 ** 9]
            per_seed.append(float(np.mean(v)))
        arr = np.array(per_seed)
        summary[hidden] = arr
        print(f"  hidden={hidden:3d} params~{rows[[r['hidden'] for r in rows].index(hidden)]['nparam']:6d}: "
              f"sat R2 = {arr.mean():.3f} +/- {arr.std():.3f}")

    if 32 in summary:
        noise = summary[32].std()
        print(f"\nNoise floor (hidden=32, across {len(seeds)} seeds): std = {noise:.4f}")
        print(f"Gate threshold ~0.01 {'EXCEEDS' if noise > 0.01 else 'is below'} noise.")
    print("\nMatched-seed deltas (seed-1):")
    base = summary[hiddens[0]]
    for h in hiddens[1:]:
        # use mean over seeds for delta
        print(f"  hidden {hiddens[0]}->{h}: delta = {summary[h].mean()-base.mean():+.4f}")
    gate_pass = (summary[hiddens[-1]].mean() - base.mean()) < 0.01
    print(f"\nGATE: capacity binding? {'NO (delta<0.01 -> skip width ladder, go to FiLM)' if gate_pass else 'YES (continue width ladder)'}")


if __name__ == "__main__":
    main()
