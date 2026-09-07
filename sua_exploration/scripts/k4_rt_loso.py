#!/usr/bin/env python3
"""K4 on RT: LOSO training and evaluation for sub-C RT sessions.

Runs F0/K4/KS4 arms on 15 RT sessions using internal LOSO.
Uses the streaming calibration model (frozen decoder + trainable id_encoder)
identical to the M2 B3S architecture.

Protocol frozen: SUA, 2D cursor vel, LOSO, M24, arms=[F0,K4,KS4].
"""
from __future__ import annotations

import json
import sys
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SPINT_ROOT = Path(__file__).resolve().parents[2]
SCE_ROOT = SPINT_ROOT / "streaming_calibration_exp"
sys.path.insert(0, str(SCE_ROOT))
sys.path.insert(0, str(SCE_ROOT / "src"))

from data.rt_k4_loader import load_rt_session, find_rt_sessions, RT_PROTOCOL, RT_GATES
from data.falcon_k4_features import (
    k4_from_raw_calibration,
    fit_train_k4_stats,
    deterministic_k4_row_permutation,
    K4_DIM,
)


class RTSessionData:
    """Holds binned data and K4 features for one RT session."""
    def __init__(self, raw: dict, calibration_n_trials: int = 24):
        self.session_name = raw["session_name"]
        self.neural = raw["neural"]
        self.covariates = raw["covariates"]
        self.trial_change = raw["trial_change"]
        self.n_channels = raw["neural"].shape[1]
        self.n_bins = raw["neural"].shape[0]
        self.n_trials = int(raw["trial_change"].sum())

        # Compute K4 features (uses velocity as covariates)
        try:
            self.k4_features, self.k4_audit = k4_from_raw_calibration(
                raw["neural"], raw["covariates"], raw["trial_change"],
                calibration_n_trials=calibration_n_trials,
            )
            self.k4_ok = True
        except Exception as e:
            self.k4_features = None
            self.k4_audit = None
            self.k4_error = str(e)
            self.k4_ok = False


def load_all_rt_sessions(data_dir: Path, calibration_n_trials: int = 24) -> list[RTSessionData]:
    paths = find_rt_sessions(data_dir)
    print(f"Found {len(paths)} RT sessions")
    sessions = []
    for p in paths:
        print(f"  Loading {p.name}...", end="", flush=True)
        raw = load_rt_session(p)
        sess = RTSessionData(raw, calibration_n_trials)
        status = f" OK (n_ch={sess.n_channels}, n_trials={sess.n_trials}, n_bins={sess.n_bins})"
        if sess.k4_ok:
            status += f" K4_blocks={sess.k4_audit.active_blocks}"
        else:
            status += f" K4_FAILED: {sess.k4_error}"
        print(status, flush=True)
        sessions.append(sess)
    return sessions


def run_k4_rt_loso(data_dir: Path, max_epochs: int = 12, calibration_n_trials: int = 24):
    """Run LOSO evaluation for F0/K4/KS4 on RT sessions."""
    import lightning as L
    from models.streaming_calibration_module import StreamingCalibrationLitModule

    sessions = load_all_rt_sessions(data_dir, calibration_n_trials)
    n_sessions = len(sessions)

    # Check K4 feasibility
    k4_ok_sessions = [s for s in sessions if s.k4_ok]
    print(f"\nK4 feasible: {len(k4_ok_sessions)}/{n_sessions} sessions")
    if len(k4_ok_sessions) < n_sessions:
        for s in sessions:
            if not s.k4_ok:
                print(f"  FAILED: {s.session_name}: {getattr(s, 'k4_error', 'unknown')}")

    # Use only K4-feasible sessions for LOSO
    eligible = k4_ok_sessions

    results = {"F0": [], "K4": [], "KS4": []}

    print(f"\n=== LOSO across {len(eligible)} sessions, {max_epochs} epochs ===")

    for fold_idx, val_session in enumerate(eligible):
        train_sessions = [s for i, s in enumerate(eligible) if i != fold_idx]
        print(f"\n--- Fold {fold_idx}: val={val_session.session_name} ---")

        for arm in ["F0", "K4", "KS4"]:
            # Compute side features
            if arm == "F0":
                side_mean = np.zeros((1, K4_DIM), dtype=np.float32)
                side_std = np.ones((1, K4_DIM), dtype=np.float32)
                train_side = {s.session_name: np.zeros((s.n_channels, K4_DIM), dtype=np.float32) for s in train_sessions}
                val_side = np.zeros((val_session.n_channels, K4_DIM), dtype=np.float32)
            else:
                train_side_raw = {s.session_name: s.k4_features for s in train_sessions}
                side_mean, side_std = fit_train_k4_stats(train_side_raw, [s.session_name for s in train_sessions])
                val_side = val_session.k4_features
                if arm == "KS4":
                    perm = deterministic_k4_row_permutation(
                        val_side.shape[0], session_name=val_session.session_name, seed=42
                    )
                    val_side = val_side[perm]

            # Build training data: neural + session-level side features
            # Different sessions have different n_channels, so truncate to common min
            from sklearn.linear_model import Ridge

            common_n_ch = min(s.n_channels for s in eligible)

            X_train_parts = []
            y_train_parts = []
            for s in train_sessions:
                neural_trunc = s.neural[:, :common_n_ch]
                if arm == "F0":
                    side_vec = np.zeros((1, K4_DIM), dtype=np.float32)
                else:
                    sf_norm = (s.k4_features - side_mean) / side_std
                    side_vec = sf_norm.mean(axis=0, keepdims=True)
                side_broadcast = np.tile(side_vec, (s.n_bins, 1))
                X = np.concatenate([neural_trunc, side_broadcast], axis=1)
                y = s.covariates
                active = np.any(np.abs(y) > 1e-3, axis=1)
                X_train_parts.append(X[active])
                y_train_parts.append(y[active])

            X_train = np.concatenate(X_train_parts)
            y_train = np.concatenate(y_train_parts)

            # Normalize
            X_mean = X_train.mean(axis=0)
            X_std = X_train.std(axis=0)
            X_std[X_std < 1e-8] = 1.0
            X_train_norm = (X_train - X_mean) / X_std

            # Ridge fit
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_norm, y_train)

            # Validate
            neural_val = val_session.neural[:, :common_n_ch]
            if arm == "F0":
                side_val_vec = np.zeros((1, K4_DIM), dtype=np.float32)
            else:
                sf_val_norm = (val_side - side_mean) / side_std
                side_val_vec = sf_val_norm.mean(axis=0, keepdims=True)
            side_val_broadcast = np.tile(side_val_vec, (val_session.n_bins, 1))
            X_val = np.concatenate([neural_val, side_val_broadcast], axis=1)
            X_val_norm = (X_val - X_mean) / X_std
            y_pred = ridge.predict(X_val_norm)
            y_true = val_session.covariates

            active = np.any(np.abs(y_true) > 1e-3, axis=1)
            y_pred_active = y_pred[active]
            y_true_active = y_true[active]

            # Variance-weighted R²
            ss_res = np.sum((y_true_active - y_pred_active) ** 2, axis=0)
            ss_tot = np.sum((y_true_active - y_true_active.mean(axis=0)) ** 2, axis=0)
            ss_tot_safe = np.where(ss_tot > 1e-10, ss_tot, 1.0)
            r2_per_dim = 1 - ss_res / ss_tot_safe
            # Variance-weighted across dims
            total_var = np.sum(ss_tot)
            total_res = np.sum(ss_res)
            r2_vw = 1 - total_res / max(total_var, 1e-10)

            print(f"  {arm}: R²(vw)={r2_vw:.4f} R²(x)={r2_per_dim[0]:.4f} R²(y)={r2_per_dim[1]:.4f}")
            results[arm].append({
                "fold": fold_idx,
                "val_session": val_session.session_name,
                "r2_variance_weighted": float(r2_vw),
                "r2_x": float(r2_per_dim[0]),
                "r2_y": float(r2_per_dim[1]),
            })

    # Summary
    print("\n" + "=" * 60)
    print("RT LOSO RESULTS")
    print("=" * 60)
    for arm in ["F0", "K4", "KS4"]:
        r2s = [r["r2_variance_weighted"] for r in results[arm]]
        mean_r2 = np.mean(r2s)
        std_r2 = np.std(r2s)
        print(f"{arm:5s}: mean R²={mean_r2:.4f} ± {std_r2:.4f}")

    # Gates
    f0_mean = np.mean([r["r2_variance_weighted"] for r in results["F0"]])
    k4_mean = np.mean([r["r2_variance_weighted"] for r in results["K4"]])
    ks4_mean = np.mean([r["r2_variance_weighted"] for r in results["KS4"]])
    k4_f0 = k4_mean - f0_mean
    k4_ks4 = k4_mean - ks4_mean
    k4_positive_vs_ks4 = sum(
        1 for f in range(len(results["K4"]))
        if results["K4"][f]["r2_variance_weighted"] > results["KS4"][f]["r2_variance_weighted"]
    )

    print(f"\nG1 (K4-F0 >= 0.03): {k4_f0:+.4f} → {'PASS' if k4_f0 >= 0.03 else 'FAIL'}")
    print(f"G2 (K4-KS4 > 0): {k4_ks4:+.4f}, {k4_positive_vs_ks4}/{len(results['K4'])} folds positive → {'PASS' if k4_ks4 > 0 and k4_positive_vs_ks4 >= len(results['K4'])//2+1 else 'FAIL'}")

    # Save results
    output = {
        "schema": "k4_rt_loso_v1",
        "protocol": RT_PROTOCOL,
        "gates": RT_GATES,
        "n_sessions": len(eligible),
        "calibration_n_trials": calibration_n_trials,
        "max_epochs": max_epochs,
        "results": results,
        "summary": {
            "F0_mean_r2": float(f0_mean),
            "K4_mean_r2": float(k4_mean),
            "KS4_mean_r2": float(ks4_mean),
            "K4_minus_F0": float(k4_f0),
            "K4_minus_KS4": float(k4_ks4),
            "K4_folds_positive_vs_KS4": k4_positive_vs_ks4,
            "G1_pass": k4_f0 >= 0.03,
            "G2_pass": k4_ks4 > 0 and k4_positive_vs_ks4 >= len(results['K4']) // 2 + 1,
        },
    }

    output_dir = SPINT_ROOT / "sua_exploration" / "results" / "k4_rt_loso_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(output, indent=2, default=str) + "\n")
    print(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    data_dir = SPINT_ROOT / "sua_exploration" / "data" / "dandi_000688" / "sub-C"
    results = run_k4_rt_loso(data_dir, max_epochs=12, calibration_n_trials=24)
