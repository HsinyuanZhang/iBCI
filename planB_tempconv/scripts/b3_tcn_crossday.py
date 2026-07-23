#!/usr/bin/env python3
"""B3: cross-day few-shot curve -- frozen TCN + unsupervised CORAL-diag.

Two alignment regimes, each a frozen k9 TCN trained on held-in:

  own_zscore (legacy/collapsed): train on z-scored held-in; test transforms the
    held-out segment by CORAL-diag (today->held-in-ref) THEN post-zscore with the
    held-in stats. Because the held-in z-score basis == the CORAL reference, the
    reference cancels exactly, leaving (x-mu_today)/sigma_today -- i.e. per-session
    own-zscore. This is the regime the ridge baseline (15_ridge_fewshot_curve.py
    coral_diag arm) also evaluates under, so TCN-own_zscore vs ridge=0.114 is a
    like-for-like decoder comparison under the SAME transform.

  ref_raw (new/meaningful): train on RAW held-in (no z-score); test transforms the
    held-out segment by CORAL-diag(today -> held-in-raw-ref) with NO post-zscore.
    CORAL(held-in -> held-in-raw-ref) = identity, so train/test representations
    match, and the held-in per-channel scale genuinely shapes the warp. This is
    the only diagonal design where the held-in reference is not a no-op. The
    scientific question: does a meaningful reference raise the ceiling over 0.149?

Windowing: all windows go through lib_planb (gap-safe contiguous-run indexing +
predict_frozen_segment), replacing the previous inline unfold.

Mirrors 15_ridge_fewshot_curve.py (same sessions, folds, budgets) so curves are
directly comparable to ridge-CORAL=0.114 and SPINT=0.26.
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
from lib_planb import (held_in_stats, make_train_windows, valid_window_indices,
                       predict_frozen_segment)
from models.tcn_student import DepthwiseTemporalConvStudent

W = 50
K = 9
HIDDEN = 32
EPOCHS = 50
BS = 2048
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BIN_S = 0.02
BUDGETS = [200, 400, 800, 1600, 3200, 10 ** 9]
REGIMES = ["own_zscore", "ref_raw"]
CEILINGS = {"static (no adapt)": 0.049, "ridge-CORAL": 0.114,
            "SPINT GF-FSU": 0.26, "LOO oracle": 0.268}


def train_tcn(smooth_hi_repr, vel_hi, tag):
    """Train a frozen k9 TCN on the given held-in representation."""
    T = len(vel_hi)
    x_t = torch.tensor(np.ascontiguousarray(smooth_hi_repr, dtype=np.float32), device=DEVICE)
    vel_t = torch.tensor(np.ascontiguousarray(vel_hi, dtype=np.float32), device=DEVICE)
    idx = torch.arange(W - 1, T, device=DEVICE)
    tcn = DepthwiseTemporalConvStudent(N_CH, W, K, hidden=HIDDEN, n_out=N_OUT).to(DEVICE)
    opt = torch.optim.Adam(tcn.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.MSELoss()
    for ep in range(EPOCHS):
        tcn.train()
        perm = idx[torch.randperm(len(idx), device=DEVICE)]
        tot = 0.0
        for s in range(0, len(perm), BS):
            b = perm[s:s + BS]
            loss = lossf(tcn(make_train_windows(x_t, b, W)), vel_t[b])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(tcn.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"  [{tag}] train ep{ep:02d} mse={tot / len(perm):.4f}")
    return tcn


@torch.no_grad()
def predict_r2(tcn, x_seg_np, y_seg_np):
    """Slide frozen TCN over one contiguous transformed segment -> R2 (variance_weighted)."""
    pred, yt = predict_frozen_segment(tcn, x_seg_np, y_seg_np, W, device=DEVICE)
    if len(yt) == 0:
        return float("nan")
    return eval_r2(yt, pred)


def transform_segment_own_zscore(smooth, test_mask, a_coral, b_coral, mu_z, sigma_z):
    """CORAL-diag -> post-zscore  ==  per-session own-zscore (reference cancels)."""
    x = smooth[test_mask] * a_coral + b_coral
    return (x - mu_z) / sigma_z


def transform_segment_ref_raw(smooth, test_mask, a_coral, b_coral):
    """CORAL-diag(today -> held-in-raw-ref), NO post-zscore. Reference is meaningful."""
    return smooth[test_mask] * a_coral + b_coral


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
    args = ap.parse_args()

    smooth_hi, vel_hi, _ = held_in()
    smooth_hi = np.asarray(smooth_hi, np.float32)
    vel_hi = np.asarray(vel_hi, np.float32)
    mu, sigma, cov = held_in_stats(smooth_hi)
    # standardize velocity targets once (R2 is invariant; both regimes share targets)
    vmu, vsd = vel_hi.mean(0), vel_hi.std(0) + 1e-6
    vel_hi = (vel_hi - vmu) / vsd
    vel_test_scale = (vsd, vmu)  # to invert predictions back for eval? eval_r2 invariant; keep for reference

    print("Training two frozen TCN (k9): own_zscore on z-scored held-in, ref_raw on raw held-in...")
    tcn_z = train_tcn((smooth_hi - mu) / sigma, vel_hi, "own_zscore")
    tcn_r = train_tcn(smooth_hi, vel_hi, "ref_raw")
    nparam = sum(p.numel() for p in tcn_z.parameters())
    print(f"TCN k9 params={nparam}  W={W}  device={DEVICE}")
    torch.save({"state_dict_z": tcn_z.state_dict(), "state_dict_r": tcn_r.state_dict(),
                "mu": mu, "sigma": sigma, "cov": cov,
                "W": W, "K": K, "nparam": nparam},
               RESULTS / "b3_frozen_tcn.pt")

    hout = held_out_sessions()
    keys = HELD_OUT_SESSION_KEYS[:2] if args.quick else HELD_OUT_SESSION_KEYS
    rows = []
    acc = {reg: {b: [] for b in BUDGETS} for reg in REGIMES}
    for key in keys:
        stem = resolve_session_key(hout, key)
        smooth, vel, _ = load_session(hout[stem])
        smooth = np.asarray(smooth, np.float32)
        vel = np.asarray(vel, np.float32)
        vel = (vel - vmu) / vsd
        folds = continuous_folds(len(vel), N_FOLDS)
        for b in BUDGETS:
            rng = np.random.default_rng(hash(key) % 2 ** 31)
            for reg in REGIMES:
                r2s = []
                for calib_m, test_m in folds:
                    calib_m = subsample_abs(calib_m, b, rng)
                    a_coral, b_coral = fit_coral_diag(smooth[calib_m], mu, sigma)
                    if reg == "own_zscore":
                        x_seg = transform_segment_own_zscore(smooth, test_m, a_coral, b_coral, mu, sigma)
                        tcn = tcn_z
                    else:
                        x_seg = transform_segment_ref_raw(smooth, test_m, a_coral, b_coral)
                        tcn = tcn_r
                    y_seg = vel[test_m]
                    r2 = predict_r2(tcn, x_seg, y_seg)
                    if r2 == r2:
                        r2s.append(r2)
                m = float(np.mean(r2s)) if r2s else float("nan")
                kept = int(min(b, folds[0][0].sum()))
                row = {"regime": reg, "session": key, "budget_frames": b, "kept_frames": kept,
                       "calib_seconds": round(kept * BIN_S, 1), "tcn_coral_r2": m}
                rows.append(row)
                acc[reg][b].append(m)
                print(f"  {reg:10s} {key:24s} budget={b:>8} ({kept * BIN_S:5.1f}s) tcn_coral={m:.3f}")

    out_csv = RESULTS / "b3_tcn_fewshot_curve.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {out_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = {"own_zscore": "C0", "ref_raw": "C3"}
        for reg in REGIMES:
            xs = [float(np.mean([r["kept_frames"] for r in rows
                                 if r["regime"] == reg and r["budget_frames"] == b])) * BIN_S
                  for b in BUDGETS]
            ys = [float(np.mean(acc[reg][b])) for b in BUDGETS]
            ax.plot(xs, ys, marker="o", color=colors[reg],
                    label=f"TCN k9 + CORAL ({reg})")
        for name, val in CEILINGS.items():
            ax.axhline(val, ls="--", lw=1, alpha=0.6)
            ax.text(ax.get_xlim()[1], val, f" {name}={val}", va="center", fontsize=7)
        ax.set_xscale("log"); ax.set_xlabel("calib data (seconds, log)")
        ax.set_ylabel("held-out mean R2"); ax.legend(fontsize=8, loc="lower right")
        ax.set_title("Plan B: frozen TCN + CORAL-diag cross-day few-shot (M2), 2 regimes")
        fig.tight_layout()
        fig.savefig(FIGURES / "b3_tcn_fewshot_curve.png", dpi=150, bbox_inches="tight")
        print(f"Saved {FIGURES / 'b3_tcn_fewshot_curve.png'}")
    except Exception as e:
        print(f"plot skipped: {e}")

    print("\n=== mean-over-sessions R2 ===")
    for reg in REGIMES:
        for b in BUDGETS:
            m = float(np.mean(acc[reg][b])) if acc[reg][b] else float("nan")
            print(f"  {reg:10s} {b:>8} frames ({b * BIN_S:6.0f}s):  tcn_coral={m:.3f}")
    sat_z = float(np.mean(acc["own_zscore"][BUDGETS[-1]])) if acc["own_zscore"][BUDGETS[-1]] else float("nan")
    sat_r = float(np.mean(acc["ref_raw"][BUDGETS[-1]])) if acc["ref_raw"][BUDGETS[-1]] else float("nan")
    print(f"\nown_zscore saturated={sat_z:.3f}  ref_raw saturated={sat_r:.3f}  ridge-CORAL=0.114")
    print(f"GATE own_zscore vs ridge: {'PASS' if sat_z > 0.114 else 'FAIL'} (TCN must beat ridge+CORAL)")
    print(f"GATE ref_raw vs own_zscore: {'RAISED' if sat_r > sat_z else 'NOT raised'} "
          f"(does meaningful reference raise the TCN ceiling?)")


if __name__ == "__main__":
    main()
