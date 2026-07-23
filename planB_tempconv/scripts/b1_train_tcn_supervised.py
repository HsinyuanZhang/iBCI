#!/usr/bin/env python3
"""B1 (gate experiment): supervised upper bound of the temporal-conv student on M2.

Trains on held-in with 5-fold continuous CV; compares:
  - linear      : flatten (96*W) -> 2   == the ridge/Wiener student structure
  - tcn_kK      : depthwise temporal conv (kernel K) -> last frame -> mix -> 2
Reports within-session mean R2 (variance_weighted) + param counts.

Milestone: depthwise TCN within-R2 >= linear, with fewer params -> Plan B worth pursuing.
If TCN cannot beat linear, M2 temporal structure is limited -> fall back / stop.

Stats/windowing (unified, v2):
  - z-score uses channel_stats (floor) -- identical basis to B3/allframe, NOT the
    raw mean/std used previously.
  - window-end indices come from lib_planb.valid_window_indices(fold_mask, W), so a
    test-fold window can no longer reach back into the calib fold (fold-boundary
    leakage fixed) and windows never cross an internal mask gap.
  - window gather uses lib_planb.make_train_windows (single implementation).

Run:  <m2r_venv>/bin/python scripts/b1_train_tcn_supervised.py [--quick]
Out:  outputs/results/b1_supervised.csv
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _env
from _env import held_in, continuous_folds, eval_r2, N_FOLDS, N_CH, N_OUT, RESULTS
from lib.rest_align import channel_stats
from lib_planb import valid_window_indices, make_train_windows
from models.tcn_student import DepthwiseTemporalConvStudent, LinearWienerStudent

W = 50                         # temporal window (SPINT-M2 uses 50); gives conv room
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_eval(model, smooth_t, vel_t, calib_idx, test_idx, epochs, bs, lr, verbose=False):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.MSELoss()
    n = len(calib_idx)
    for ep in range(epochs):
        model.train()
        perm = calib_idx[torch.randperm(n, device=DEVICE)]
        tot = 0.0
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            x = make_train_windows(smooth_t, b, W)
            y = vel_t[b]
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"    ep{ep:02d} train_mse={tot/n:.4f}")
    # eval
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for s in range(0, len(test_idx), 8192):
            b = test_idx[s:s + 8192]
            preds.append(model(make_train_windows(smooth_t, b, W)).cpu().numpy())
            ys.append(vel_t[b].cpu().numpy())
    return eval_r2(np.concatenate(ys), np.concatenate(preds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 folds, fewer epochs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    smooth, vel = held_in()[:2]
    smooth = np.asarray(smooth, np.float32); vel = np.asarray(vel, np.float32)
    # held-in z-score with channel_stats (floor) -- unified basis with B3/allframe
    mu, sd = channel_stats(smooth)
    smooth = (smooth - mu) / sd
    # standardize velocity targets for stable MSE optimization; R2 is invariant to this
    vmu, vsd = vel.mean(0), vel.std(0) + 1e-6
    vel = (vel - vmu) / vsd
    T = len(vel)
    smooth_t = torch.tensor(smooth, device=DEVICE)
    vel_t = torch.tensor(vel, device=DEVICE)
    print(f"held-in T={T}  N_CH={N_CH}  W={W}  device={DEVICE}")

    def builders():
        yield "linear", lambda: LinearWienerStudent(N_CH, W, N_OUT)
        for K in (5, 9, 15):
            yield f"tcn_k{K}", (lambda K=K: DepthwiseTemporalConvStudent(N_CH, W, K, hidden=32, n_out=N_OUT))

    folds = continuous_folds(T, N_FOLDS)
    if args.quick:
        folds = folds[:2]; args.epochs = min(args.epochs, 8)

    rows = []
    for name, build in builders():
        nparam = build().n_params() if hasattr(build(), "n_params") else sum(p.numel() for p in build().parameters())
        scores = []
        t0 = time.time()
        for fi, (calib_m, test_m) in enumerate(folds):
            # fold + gap safe: window fully inside the fold's own contiguous runs
            ci = torch.tensor(valid_window_indices(calib_m, W), device=DEVICE)
            ti = torch.tensor(valid_window_indices(test_m, W), device=DEVICE)
            torch.manual_seed(fi)
            r2 = train_eval(build(), smooth_t, vel_t, ci, ti, args.epochs, args.bs, args.lr,
                            verbose=(fi == 0))
            scores.append(r2)
        m = float(np.mean(scores)); sd_ = float(np.std(scores))
        dt = time.time() - t0
        print(f"{name:10s} params={nparam:6,}  R2={m:.4f} ± {sd_:.4f}  folds={[f'{s:.3f}' for s in scores]}  {dt:.0f}s")
        rows.append({"model": name, "params": nparam, "r2_mean": m, "r2_std": sd_,
                     "folds": ";".join(f"{s:.4f}" for s in scores)})

    with open(RESULTS / "b1_supervised.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved {RESULTS / 'b1_supervised.csv'}")
    lin = next(r for r in rows if r["model"] == "linear")["r2_mean"]
    best_tcn = max(r["r2_mean"] for r in rows if r["model"].startswith("tcn"))
    verdict = "PASS: TCN >= linear -> proceed to B3" if best_tcn >= lin else "FAIL: TCN < linear -> reconsider"
    print(f"\nlinear={lin:.4f}  best_tcn={best_tcn:.4f}  ->  {verdict}")


if __name__ == "__main__":
    main()
