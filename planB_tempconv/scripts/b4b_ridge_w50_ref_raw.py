#!/usr/bin/env python3
"""B4b: Window-length-matched ridge control for B3 TCN claim.

B3 showed TCN(W=50, ref_raw) ≈ 0.20 vs ridge(N_HIST=7, ref_raw) ≈ 0.11.
But the temporal windows differ: 1000 ms vs 140 ms. This script runs ridge
with N_HIST=50 (same 1000 ms window, 96×50 = 4800 flat features) on ref_raw.

  if ridge(N_HIST=50, ref_raw) ≈ 0.20 → gap is window length, conv adds little
  if ridge(N_HIST=50, ref_raw) << 0.20 → conv structure genuinely contributes

Mirrors b4_ridge_ref_raw.py byte-for-byte except N_HIST=50.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _env  # noqa: bootstraps m2-research on sys.path
from _env import (held_in, held_out_sessions, continuous_folds, eval_r2,
                  N_FOLDS, N_CH, N_OUT, RESULTS, FIGURES,
                  fit_coral_diag, apply_coral_diag)
from _env import resolve_session_key, load_session, HELD_OUT_SESSION_KEYS
from lib.rest_align import allframe_stats, channel_stats, apply_fixed_zscore
from lib.decoding_utils import generate_lagged_matrix
from sklearn.linear_model import RidgeCV

BIN_S = 0.02
N_HIST = 50
BUDGETS = [200, 400, 800, 1600, 3200, 10 ** 9]
REGIMES = ["own_zscore", "ref_raw"]
CEILINGS = {"static (no adapt)": 0.049, "ridge-CORAL N_HIST=7": 0.114,
            "TCN W=50 ref_raw": 0.20, "SPINT GF-FSU": 0.26, "LOO oracle": 0.268}


def subsample_abs(mask, n_frames, rng):
    idx = np.flatnonzero(mask)
    if n_frames >= len(idx):
        return mask
    pick = rng.choice(idx, size=n_frames, replace=False)
    out = np.zeros_like(mask)
    out[pick] = True
    return out


def fit_ridge_frozen(X, y):
    rcv = RidgeCV(alphas=np.logspace(-5, 5, 20), scoring="r2")
    rcv.fit(X, y)
    return rcv.coef_, rcv.intercept_, rcv.alpha_


def ridge_predict(W, b, X):
    return X @ W.T + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 sessions")
    args = ap.parse_args()

    smooth_hi, vel_hi, _ = held_in()
    smooth_hi = np.asarray(smooth_hi, np.float32)
    vel_hi = np.asarray(vel_hi, np.float32)

    mu_z, sigma_z = channel_stats(smooth_hi)
    smooth_z = apply_fixed_zscore(smooth_hi, mu_z, sigma_z)

    print(f"Building lagged matrices (N_HIST={N_HIST}, {N_CH*N_HIST} features)...")
    X_z = generate_lagged_matrix(smooth_z, N_HIST)
    X_r = generate_lagged_matrix(smooth_hi, N_HIST)
    y_hi = vel_hi[N_HIST:]
    print(f"  X shape: {X_z.shape}  y shape: {y_hi.shape}")

    mu_ref_raw, sigma_ref_raw, _, _ = allframe_stats(smooth_hi)

    print(f"Training two frozen ridge (N_HIST={N_HIST})...")
    W_z, b_z, alpha_z = fit_ridge_frozen(X_z, y_hi)
    W_r, b_r, alpha_r = fit_ridge_frozen(X_r, y_hi)
    print(f"  ridge_z (own_zscore): alpha={alpha_z:.4g}  W={W_z.shape}")
    print(f"  ridge_r (ref_raw):    alpha={alpha_r:.4g}  W={W_r.shape}")

    hout = held_out_sessions()
    keys = HELD_OUT_SESSION_KEYS[:2] if args.quick else HELD_OUT_SESSION_KEYS
    rows = []
    acc = {reg: {b: [] for b in BUDGETS} for reg in REGIMES}

    for key in keys:
        stem = resolve_session_key(hout, key)
        smooth, vel, _ = load_session(hout[stem])
        smooth = np.asarray(smooth, np.float32)
        vel = np.asarray(vel, np.float32)
        folds = continuous_folds(len(vel), N_FOLDS)

        for b in BUDGETS:
            rng = np.random.default_rng(hash(key) % 2 ** 31)
            for reg in REGIMES:
                r2s = []
                for calib_m, test_m in folds:
                    calib_m = subsample_abs(calib_m, b, rng)
                    if reg == "own_zscore":
                        a_c, b_c = fit_coral_diag(smooth[calib_m], mu_ref_raw, sigma_ref_raw)
                        x_test = apply_coral_diag(smooth[test_m], a_c, b_c)
                        x_test = apply_fixed_zscore(x_test, mu_z, sigma_z)
                        W, bias = W_z, b_z
                    else:
                        a_c, b_c = fit_coral_diag(smooth[calib_m], mu_ref_raw, sigma_ref_raw)
                        x_test = apply_coral_diag(smooth[test_m], a_c, b_c)
                        W, bias = W_r, b_r
                    X_test = generate_lagged_matrix(x_test, N_HIST)
                    y_test = vel[test_m][N_HIST:]
                    if len(y_test) < 5:
                        continue
                    pred = ridge_predict(W, bias, X_test)
                    r2 = eval_r2(y_test, pred)
                    if r2 == r2:
                        r2s.append(r2)
                m = float(np.mean(r2s)) if r2s else float("nan")
                kept = int(min(b, folds[0][0].sum()))
                rows.append({"regime": reg, "session": key, "budget_frames": b,
                             "kept_frames": kept,
                             "calib_seconds": round(kept * BIN_S, 1),
                             "ridge_r2": m})
                acc[reg][b].append(m)
                print(f"  {reg:10s} {key:24s} budget={b:>8} ({kept * BIN_S:5.1f}s) ridge={m:.3f}")

    out_csv = RESULTS / "b4b_ridge_w50_ref_raw.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
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
            ax.plot(xs, ys, marker="s", color=colors[reg],
                    label=f"ridge N_HIST={N_HIST} ({reg})")
        for name, val in CEILINGS.items():
            ax.axhline(val, ls="--", lw=1, alpha=0.6)
            ax.text(ax.get_xlim()[1], val, f" {name}={val}", va="center", fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("calib data (seconds, log)")
        ax.set_ylabel("held-out mean R2")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title(f"B4b: ridge N_HIST={N_HIST} + CORAL-diag (window-matched vs TCN W=50)")
        fig.tight_layout()
        fig.savefig(FIGURES / "b4b_ridge_w50_ref_raw.png", dpi=150, bbox_inches="tight")
        print(f"Saved {FIGURES / 'b4b_ridge_w50_ref_raw.png'}")
    except Exception as e:
        print(f"plot skipped: {e}")

    print("\n=== mean-over-sessions R2 (ridge N_HIST=50) ===")
    for reg in REGIMES:
        for b in BUDGETS:
            m = float(np.mean(acc[reg][b])) if acc[reg][b] else float("nan")
            print(f"  {reg:10s} {b:>8} frames ({b * BIN_S:6.0f}s):  ridge={m:.3f}")

    sat_z = float(np.mean(acc["own_zscore"][BUDGETS[-1]])) if acc["own_zscore"][BUDGETS[-1]] else float("nan")
    sat_r = float(np.mean(acc["ref_raw"][BUDGETS[-1]])) if acc["ref_raw"][BUDGETS[-1]] else float("nan")

    print(f"\n{'=' * 60}")
    print(f"ridge(N_HIST=7,  ref_raw) saturated = ~0.11   (from b4)")
    print(f"ridge(N_HIST=50, ref_raw) saturated = {sat_r:.3f}   (THIS experiment)")
    print(f"ridge(N_HIST=50, own_zsc) saturated = {sat_z:.3f}")
    print(f"TCN(W=50,        ref_raw) saturated = ~0.20   (from b3)")
    print(f"{'=' * 60}")

    if sat_r >= 0.17:
        print(f"\nVERDICT: ridge(W=50, ref_raw)={sat_r:.3f} ≈ TCN(W=50, ref_raw)=0.20")
        print(f"  → window length explains the gap; conv structure adds little")
        print(f"  → B3 'TCN > ridge' claim NOT supported at matched window")
    else:
        gap = 0.20 - sat_r
        print(f"\nVERDICT: ridge(W=50, ref_raw)={sat_r:.3f} << TCN(W=50, ref_raw)=0.20  (gap={gap:.3f})")
        print(f"  → conv structure genuinely contributes beyond window length")
        print(f"  → B3 claim holds: TCN temporal conv > flat ridge at same window")


if __name__ == "__main__":
    main()
