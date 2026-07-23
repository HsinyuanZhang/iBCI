#!/usr/bin/env python3
"""§8: Content-dependent channel gating — SE block & GLU activation (seeded).

Motivation. All prior levers were STATIC (width c1, depth c2) or SESSION-LEVEL
(FiLM c3). SPINT's edge is content-dependent per-frame channel routing (cross-
attention). This probe tests two cheap, quantization-friendly mechanisms that add
that missing "dynamic" axis:

  SETCN : Squeeze-Excitation channel attention. Pools temporal conv output over
          time -> bottleneck (C->C/r->C) -> sigmoid gate -> per-channel rescale.
          Gate changes every frame based on neural activity. ~2.4k extra params.

  GLU   : Gated Linear Unit in the mix. Linear(C,2h) -> split -> a*sigmoid(g).
          Replaces fixed ReLU with a learned content-dependent gate. ~3.1k extra.

Control: seed-matched plain TCN (DepthwiseTemporalConvStudent hidden=32) for
isolated delta at identical data/seeds/eval. ref_raw, CORAL-diag, 6 sessions.
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
from lib_planb import held_in_stats, make_train_windows, predict_frozen_segment
from models.tcn_student import (DepthwiseTemporalConvStudent, SETCNStudent,
                                GLUTCNStudent)

W = 50
K = 9
EPOCHS = 50
BS = 2048
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BIN_S = 0.02
BUDGETS = [200, 400, 800, 1600, 3200, 10 ** 9]
CONFIGS = {
    "plain": lambda: DepthwiseTemporalConvStudent(N_CH, W, K, hidden=32, n_out=N_OUT),
    "se":    lambda: SETCNStudent(N_CH, W, K, hidden=32, n_out=N_OUT),
    "glu":   lambda: GLUTCNStudent(N_CH, W, K, hidden=32, n_out=N_OUT),
}


def train_model(model, smooth_hi, vel_hi, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    T = len(vel_hi)
    x_t = torch.tensor(np.ascontiguousarray(smooth_hi, dtype=np.float32), device=DEVICE)
    vel_t = torch.tensor(np.ascontiguousarray(vel_hi, dtype=np.float32), device=DEVICE)
    idx = torch.arange(W - 1, T, device=DEVICE)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.MSELoss()
    for ep in range(EPOCHS):
        model.train()
        g = torch.Generator(device=DEVICE).manual_seed(seed * 7919 + ep)
        perm = idx[torch.randperm(len(idx), generator=g, device=DEVICE)]
        for s in range(0, len(perm), BS):
            b = perm[s:s + BS]
            loss = lossf(model(make_train_windows(x_t, b, W)), vel_t[b])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    # in-sample R2 for diagnostics
    model.eval()
    with torch.no_grad():
        pred_hi = torch.cat([model(make_train_windows(x_t, idx[s:s + BS], W))
                             for s in range(0, len(idx), BS)]).cpu().numpy()
    train_r2 = eval_r2(vel_hi[idx.cpu().numpy()], pred_hi)
    return model, train_r2


@torch.no_grad()
def predict_r2(model, x_seg_np, y_seg_np):
    pred, yt = predict_frozen_segment(model, x_seg_np, y_seg_np, W, device=DEVICE)
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
    for name, factory in CONFIGS.items():
        sat_by_seed = []
        for seed in seeds:
            model, train_r2 = train_model(factory(), smooth_hi, vel_hi, seed)
            nparam = sum(p.numel() for p in model.parameters())
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
                        r2 = predict_r2(model, x_seg, y_seg)
                        if r2 == r2:
                            r2s.append(r2)
                    m = float(np.mean(r2s)) if r2s else float("nan")
                    acc[b].append(m)
                    rows.append({"config": name, "seed": seed, "nparam": nparam,
                                 "session": key, "budget_frames": b,
                                 "calib_seconds": round(min(b, folds[0][0].sum()) * BIN_S, 1),
                                 "ref_raw_r2": m})
            sat = float(np.mean(acc[BUDGETS[-1]])) if acc[BUDGETS[-1]] else float("nan")
            sat_by_seed.append(sat)
            done += 1
            print(f"[{done}/{n_combo}] {name:6s} seed={seed} params={nparam:6d} "
                  f"trainR2={train_r2:.3f} ref_raw sat={sat:.3f}  ({time.time()-t0:.0f}s)")
        arr = np.array(sat_by_seed)
        summary[name] = arr
        print(f"  -> {name:6s}: mean={arr.mean():.3f} std={arr.std():.3f} "
              f"[{', '.join(f'{x:.3f}' for x in arr)}]")

    out_csv = RESULTS / "c4_se_glu_probe.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {out_csv}")

    print("\n=== SE/GLU probe summary (ref_raw saturated, mean over seeds) ===")
    for name in CONFIGS:
        arr = summary[name]
        print(f"  {name:6s}: sat R2 = {arr.mean():.3f} +/- {arr.std():.3f}")
    base = summary["plain"].mean()
    print("\nDeltas vs plain:")
    for name in ("se", "glu"):
        d = summary[name].mean() - base
        print(f"  plain -> {name}: delta = {d:+.4f}")
    noise = summary["plain"].std()
    print(f"\nNoise floor (plain, {len(seeds)} seeds): std = {noise:.4f}")


if __name__ == "__main__":
    main()
