#!/usr/bin/env python3
"""§4: FiLM conditioning from per-session CORAL calib stats (seeded).

Hypothesis (Plan §2/§4): the missing 0.06 R2 vs SPINT is session *conditioning*,
not FFN width (Step 1 disproved width) nor temporal RF (c2 tests that). SPINT
buys adaptation with cross-attention; the cheap, on-chip analogue is FiLM:
feed the CORAL-diag calib vector [a,b] (2*96 numbers, computed once per session
off-model) through a tiny generator -> per-channel gamma,beta that affine-modulate
the conv output before the pointwise mix.

TRAINING PROBLEM. The held-in "session" has CORAL(a,b)=(1,0) against itself, so a
naive train on held-in gives the generator zero conditioning variance to learn
from. Fix: *augment* by applying a random residual per-channel affine warp to the
held-in windows and conditioning the generator on the SAME warp. This simulates
the cross-session CORAL spread the model will see at test:

    ea ~ N(0, aug_a), eb ~ N(0, aug_b)          # per channel
    a_cor = 1 + ea, b_cor = eb
    x_aligned = a_cor * x_raw + b_cor            # held-in-raw warped
    cond = [a_cor, b_cor]                         # what the generator sees
    loss = mse(FilmTCN(x_aligned, cond), vel)

aug_a, aug_b are measured from the EMPIRICAL per-session CORAL spread across the
6 held-out sessions, so the training conditioning distribution matches test.

CONTROL. A seed-matched PLAIN TCN (DepthwiseTemporalConvStudent hidden=32, no
FiLM, no aug) is trained and evaluated identically, so the FiLM delta is isolated
at matched seeds (same data, same eval folds/budgets).

ref_raw regime, CORAL-diag cross-day, same 6 sessions / folds / budgets as c1.
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
from models.tcn_student import FilmTCNStudent, DepthwiseTemporalConvStudent

W = 50
K = 9
EPOCHS = 50
BS = 2048
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BIN_S = 0.02
BUDGETS = [200, 400, 800, 1600, 3200, 10 ** 9]


def measure_coral_spread(hout, keys, mu_ref, sigma_ref):
    """Pooled per-channel std of CORAL (a,b) across held-out sessions.

    Returns (aug_a, aug_b): std of (a-1) and of b, pooled over sessions x channels.
    Uses a large calib fold per session (saturated regime) for stable estimates.
    """
    a_all, b_all = [], []
    for key in keys:
        stem = resolve_session_key(hout, key)
        smooth, _, _ = load_session(hout[stem])
        smooth = np.asarray(smooth, np.float32)
        folds = continuous_folds(len(smooth), N_FOLDS)
        calib_mask = folds[0][0]                     # first fold calib (large)
        a, b = fit_coral_diag(smooth[calib_mask], mu_ref, sigma_ref)
        a_all.append(a); b_all.append(b)
    a_all = np.stack(a_all)                          # (n_sess, n_ch)
    b_all = np.stack(b_all)
    aug_a = float(np.std(a_all - 1.0))              # pooled std of residual
    aug_b = float(np.std(b_all))
    return aug_a, aug_b


def train_plain_seeded(smooth_hi, vel_hi, hidden, seed):
    torch.manual_seed(seed); np.random.seed(seed)
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
        for s in range(0, len(perm), BS):
            b = perm[s:s + BS]
            loss = lossf(tcn(make_train_windows(x_t, b, W)), vel_t[b])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(tcn.parameters(), 1.0)
            opt.step()
    return tcn


def train_film_seeded(smooth_hi, vel_hi, hidden, seed, aug_a, aug_b):
    """Train FilmTCN on held-in RAW with random residual-warp augmentation."""
    torch.manual_seed(seed); np.random.seed(seed)
    T = len(vel_hi)
    x_t = torch.tensor(np.ascontiguousarray(smooth_hi, dtype=np.float32), device=DEVICE)
    vel_t = torch.tensor(np.ascontiguousarray(vel_hi, dtype=np.float32), device=DEVICE)
    idx = torch.arange(W - 1, T, device=DEVICE)
    model = FilmTCNStudent(N_CH, W, K, hidden=hidden, n_out=N_OUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.MSELoss()
    for ep in range(EPOCHS):
        model.train()
        g = torch.Generator(device=DEVICE).manual_seed(seed * 7919 + ep)
        perm = idx[torch.randperm(len(idx), generator=g, device=DEVICE)]
        for s in range(0, len(perm), BS):
            b = perm[s:s + BS]
            xw = make_train_windows(x_t, b, W)              # (B,C,W) held-in RAW
            ea = torch.randn(N_CH, device=DEVICE) * aug_a
            eb = torch.randn(N_CH, device=DEVICE) * aug_b
            a_cor = (1.0 + ea).view(1, N_CH, 1)
            b_cor = eb.view(1, N_CH, 1)
            x_aligned = a_cor * xw + b_cor
            cond = torch.cat([a_cor.view(1, N_CH), b_cor.view(1, N_CH)], dim=1)  # (1, 2C)
            cond = cond.expand(len(b), -1)
            loss = lossf(model(x_aligned, cond), vel_t[b])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


@torch.no_grad()
def predict_r2_plain(tcn, x_seg_np, y_seg_np):
    pred, yt = predict_frozen_segment(tcn, x_seg_np, y_seg_np, W, device=DEVICE)
    if len(yt) == 0:
        return float("nan")
    return eval_r2(yt, pred)


@torch.no_grad()
def predict_r2_film(model, x_seg_np, y_seg_np, a_cor, b_cor):
    """Frozen FilmTCN over one contiguous aligned segment; cond = real session [a,b]."""
    from lib_planb import window_segment
    T = x_seg_np.shape[0]
    if T < W:
        return float("nan")
    X, y_w = window_segment(x_seg_np, y_seg_np, W)
    model.eval()
    dev = next(model.parameters()).device
    cond_np = np.concatenate([a_cor, b_cor]).astype(np.float32)   # (2C,)
    cond_b = np.broadcast_to(cond_np, (8192, 2 * N_CH))
    preds = []
    for i in range(0, len(X), 8192):
        xb = torch.from_numpy(X[i:i + 8192]).to(dev)
        n = len(xb)
        cb = torch.from_numpy(cond_b[:n]).to(dev).clone()
        preds.append(model(xb, cb).cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    if len(y_w) == 0:
        return float("nan")
    return eval_r2(y_w, pred)


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
    ap.add_argument("--hidden", type=int, default=32)
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

    aug_a, aug_b = measure_coral_spread(hout, keys, mu, sigma)
    print(f"Empirical CORAL spread: aug_a={aug_a:.4f}  aug_b={aug_b:.4f}")

    rows = []
    t0 = time.time()
    summary = {}
    for seed in seeds:
        # --- plain TCN control (seed-matched) ---
        tcn = train_plain_seeded(smooth_hi, vel_hi, args.hidden, seed)
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
                    r2 = predict_r2_plain(tcn, x_seg, vel[test_m])
                    if r2 == r2:
                        r2s.append(r2)
                m = float(np.mean(r2s)) if r2s else float("nan")
                acc[b].append(m)
                rows.append({"config": "plain", "seed": seed, "session": key,
                             "budget_frames": b, "ref_raw_r2": m})
        sat = float(np.mean(acc[BUDGETS[-1]])) if acc[BUDGETS[-1]] else float("nan")
        summary.setdefault("plain", []).append(sat)
        print(f"[plain]    seed={seed} ref_raw sat={sat:.3f}  ({time.time()-t0:.0f}s)")

        # --- FiLM TCN (augmented) ---
        model = train_film_seeded(smooth_hi, vel_hi, args.hidden, seed, aug_a, aug_b)
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
                    r2 = predict_r2_film(model, x_seg, vel[test_m], a_coral, b_coral)
                    if r2 == r2:
                        r2s.append(r2)
                m = float(np.mean(r2s)) if r2s else float("nan")
                acc[b].append(m)
                rows.append({"config": "film", "seed": seed, "session": key,
                             "budget_frames": b, "ref_raw_r2": m})
        sat = float(np.mean(acc[BUDGETS[-1]])) if acc[BUDGETS[-1]] else float("nan")
        summary.setdefault("film", []).append(sat)
        print(f"[film]     seed={seed} ref_raw sat={sat:.3f}  ({time.time()-t0:.0f}s)")

    out_csv = RESULTS / "c3_film_probe.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "seed", "session",
                                          "budget_frames", "ref_raw_r2"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {out_csv}")

    print("\n=== FiLM probe summary (ref_raw saturated, mean over seeds) ===")
    for name in ("plain", "film"):
        arr = np.array(summary[name])
        print(f"  {name:6s}: sat R2 = {arr.mean():.3f} +/- {arr.std():.3f}  "
              f"[{', '.join(f'{x:.3f}' for x in arr)}]")
    delta = np.mean(summary["film"]) - np.mean(summary["plain"])
    print(f"\nFiLM vs plain delta: {delta:+.4f}")


if __name__ == "__main__":
    main()
