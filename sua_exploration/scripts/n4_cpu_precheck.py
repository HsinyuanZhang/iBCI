#!/usr/bin/env python3
"""N4 CPU pre-check (A-阶段0): behavior-free carrier descriptor audit.

Computes N4 features on 7 held-in M2 sessions and checks:
1. Per-dim 12/12 split-half across-channel Pearson
2. Per-dim residual R² against T4
3. mean_rate vs T4 baseline_rate per-session Pearson
4. Cross-session channel identification top-1

Gate: at least 2 dims with split-half r >= 0.5 in majority (>=4/7) of sessions.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

SPINT_ROOT = Path(__file__).resolve().parents[2]
SCE_ROOT = SPINT_ROOT / "streaming_calibration_exp"
CARRIER_PERF_SRC = SPINT_ROOT / "sua_exploration" / "carrier_perf_program" / "src"

for p in [str(SCE_ROOT), str(SCE_ROOT / "src"), str(CARRIER_PERF_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data.falcon_n4_features import (
    N4_FEATURE_NAMES as _N4_FEATURE_NAMES,
    N4_CALIBRATION_TRIALS as _N4_CALIBRATION_TRIALS,
    n4_from_raw_calibration,
)
N4_FEATURE_NAMES = _N4_FEATURE_NAMES
N4_CALIBRATION_TRIALS = _N4_CALIBRATION_TRIALS
from carrier_perf.k4_component_decomposition import (
    CALIBRATION_N_TRIALS,
    SPLIT_HALF_N_TRIALS,
    RESIDUAL_R2_EXPLAINED_THRESHOLD,
    SPLIT_HALF_RELIABILITY_THRESHOLD,
    T4_FEATURE_NAMES,
    heldin_m2_paths,
    sha256_file,
    _load_m2_session,
    _pearson,
    residual_r_squared,
    channel_matching_accuracy,
)

SCHEMA = "n4_neural_only_carrier_cpu_precheck_v1"


def n4_from_trial_window(
    neural: np.ndarray,
    trial_change: np.ndarray,
    *,
    trial_start: int,
    n_trials: int,
    source: str,
) -> np.ndarray:
    """N4 on a chronological trial window (no behavior filtering)."""
    neural = np.asarray(neural, dtype=np.float64)
    trial_change = np.asarray(trial_change, dtype=bool)
    starts = np.flatnonzero(trial_change)
    if len(starts) < trial_start + n_trials:
        raise ValueError(
            f"N4 trial window [{trial_start}:{trial_start + n_trials}) unavailable for {source}"
        )

    # Extract the sub-array for this trial window
    t_start_idx = int(starts[trial_start])
    if trial_start + n_trials < len(starts):
        t_end_idx = int(starts[trial_start + n_trials])
    else:
        t_end_idx = len(trial_change)

    sub_neural = neural[t_start_idx:t_end_idx]
    sub_trial_change = trial_change[t_start_idx:t_end_idx].copy()
    sub_trial_change[0] = True

    features, _ = n4_from_raw_calibration(
        sub_neural,
        sub_trial_change,
        calibration_n_trials=n_trials,
        degeneracy_policy="fill_median",
    )
    return features


def audit_n4_session(session: dict) -> dict:
    """Compute N4 features and audit metrics for one session."""
    nwb_path = Path(session["nwb_path"])
    # Load raw neural arrays the same way _load_m2_session does
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    neural, covariates, trial_change, eval_mask = load_nwb(nwb_path, FalconTask.m2)
    neural = np.asarray(neural, dtype=np.float64)
    trial_change = np.asarray(trial_change, dtype=bool)

    # N4 full calibration
    n4_full, n4_audit = n4_from_raw_calibration(
        neural, trial_change,
        calibration_n_trials=CALIBRATION_N_TRIALS,
        degeneracy_policy="fill_median",
    )

    # N4 split-half
    n4_h1 = n4_from_trial_window(
        neural, trial_change,
        trial_start=0, n_trials=SPLIT_HALF_N_TRIALS,
        source=f"{nwb_path.name}[0:{SPLIT_HALF_N_TRIALS}]",
    )
    n4_h2 = n4_from_trial_window(
        neural, trial_change,
        trial_start=SPLIT_HALF_N_TRIALS, n_trials=SPLIT_HALF_N_TRIALS,
        source=f"{nwb_path.name}[{SPLIT_HALF_N_TRIALS}:{CALIBRATION_N_TRIALS}]",
    )

    t4 = np.asarray(session["t4_full"], dtype=np.float64)

    # Split-half reliability per dim
    sh_r = []
    for d in range(4):
        r = _pearson(n4_h1[:, d], n4_h2[:, d])
        sh_r.append(r)

    # Residual R² of each N4 dim against T4 (4 dims)
    resid = []
    for d in range(4):
        r = residual_r_squared(n4_full[:, d], t4)
        resid.append(r)

    # mean_rate vs T4 baseline_rate correlation
    mr_vs_br = _pearson(n4_full[:, 0], t4[:, 3])

    return {
        "session": session["session"],
        "nwb_sha256": session["nwb_sha256"],
        "n_channels": n4_full.shape[0],
        "n4_audit": n4_audit.as_dict(),
        "n4_features": n4_full.tolist(),
        "n4_split_half": {"first": n4_h1.tolist(), "second": n4_h2.tolist()},
        "split_half_reliability": {N4_FEATURE_NAMES[i]: sh_r[i] for i in range(4)},
        "residual_r_squared_vs_t4": {N4_FEATURE_NAMES[i]: resid[i] for i in range(4)},
        "mean_rate_vs_t4_baseline_rate_pearson": mr_vs_br,
    }


def check_gate(session_records: list[dict]) -> dict:
    """Check the pre-declared majority gate: >=2 dims with split-half r>=0.5 in >=4/7 sessions."""
    n_sessions = len(session_records)
    majority_threshold = n_sessions // 2 + 1  # 4/7

    per_dim_pass = {}
    for dim_name in N4_FEATURE_NAMES:
        pass_count = sum(
            1 for r in session_records
            if r["split_half_reliability"][dim_name] is not None
            and math.isfinite(r["split_half_reliability"][dim_name])
            and r["split_half_reliability"][dim_name] >= SPLIT_HALF_RELIABILITY_THRESHOLD
        )
        per_dim_pass[dim_name] = {
            "n_sessions_passing": pass_count,
            "majority_pass": pass_count >= majority_threshold,
            "per_session_r": [r["split_half_reliability"][dim_name] for r in session_records],
        }

    dims_passing_majority = sum(1 for v in per_dim_pass.values() if v["majority_pass"])
    gate_passed = dims_passing_majority >= 2

    return {
        "gate_name": "n4_cpu_precheck_majority_split_half",
        "threshold_r": SPLIT_HALF_RELIABILITY_THRESHOLD,
        "majority_rule": f">={majority_threshold}/{n_sessions} sessions",
        "min_dims_passing": 2,
        "dims_passing_majority": dims_passing_majority,
        "per_dim": per_dim_pass,
        "gate_passed": gate_passed,
    }


def cross_session_identification(session_records: list[dict]) -> dict:
    """Channel matching using N4 descriptors."""
    names = [r["session"] for r in session_records]
    n4_arrays = [np.asarray(r["n4_features"], dtype=np.float64) for r in session_records]

    top1_scores = []
    chance_levels = []
    for i in range(len(n4_arrays)):
        for j in range(len(n4_arrays)):
            if i == j:
                continue
            row = channel_matching_accuracy(n4_arrays[i], n4_arrays[j])
            top1_scores.append(row["top1_accuracy"])
            chance_levels.append(row["chance_top1_accuracy"])

    return {
        "n_pairs": len(top1_scores),
        "top1_accuracy_mean": float(np.mean(top1_scores)) if top1_scores else float("nan"),
        "top1_accuracy_std": float(np.std(top1_scores, ddof=1)) if len(top1_scores) >= 2 else float("nan"),
        "chance_top1_accuracy": float(np.mean(chance_levels)) if chance_levels else float("nan"),
    }


def main():
    data_dir = SPINT_ROOT / "SPINT-main" / "data" / "000953"
    if not data_dir.exists():
        print(f"ERROR: data dir {data_dir} not found", file=sys.stderr)
        sys.exit(1)

    paths = heldin_m2_paths(data_dir)
    print(f"Found {len(paths)} held-in M2 sessions", flush=True)

    # Load sessions using existing k4_component_decomposition loader (gives us T4 too)
    loaded = [_load_m2_session(p) for p in paths]
    print(f"Loaded {len(loaded)} sessions with T4/K4 features", flush=True)

    # Compute N4 audit per session
    n4_records = []
    for i, sess in enumerate(loaded):
        print(f"  N4 [{i+1}/{len(loaded)}] {sess['session']}...", end="", flush=True)
        rec = audit_n4_session(sess)
        n4_records.append(rec)
        print(f" OK (n_blocks={rec['n4_audit']['n_blocks']}, degenerate={rec['n4_audit']['n_degenerate_channels']})", flush=True)

    # Run gate check
    gate = check_gate(n4_records)
    ident = cross_session_identification(n4_records)

    # Build residual R² summary
    resid_summary = {}
    for dim_name in N4_FEATURE_NAMES:
        vals = [r["residual_r_squared_vs_t4"][dim_name] for r in n4_records]
        finite = [v for v in vals if v is not None and math.isfinite(v)]
        resid_summary[dim_name] = {
            "mean": float(np.mean(finite)) if finite else float("nan"),
            "per_session": vals,
            "verdict": "explained_by_t4" if (finite and np.mean(finite) < RESIDUAL_R2_EXPLAINED_THRESHOLD) else "novel",
        }

    mr_br_vals = [r["mean_rate_vs_t4_baseline_rate_pearson"] for r in n4_records]
    finite_mr_br = [v for v in mr_br_vals if v is not None and math.isfinite(v)]

    audit = {
        "schema": SCHEMA,
        "status": "completed_cpu_only",
        "no_gpu": True,
        "no_heldout_opened": True,
        "calibration_n_trials": CALIBRATION_N_TRIALS,
        "split_half_n_trials": SPLIT_HALF_N_TRIALS,
        "n_sessions": len(n4_records),
        "n4_feature_names": list(N4_FEATURE_NAMES),
        "input_nwb_sha256": {r["session"]: r["nwb_sha256"] for r in n4_records},
        "gate": gate,
        "residual_r2_vs_t4_summary": resid_summary,
        "mean_rate_vs_t4_baseline_rate": {
            "per_session_pearson": mr_br_vals,
            "mean": float(np.mean(finite_mr_br)) if finite_mr_br else float("nan"),
        },
        "cross_session_identification": ident,
        "sessions": [{k: v for k, v in r.items() if k != "n4_features" and k != "n4_split_half"} for r in n4_records],
    }

    # Write audit
    output_dir = SPINT_ROOT / "sua_exploration" / "results" / "n4_cpu_precheck_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "audit.json"

    def _strict(obj):
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, np.ndarray):
            return _strict(obj.tolist())
        if isinstance(obj, dict):
            return {str(k): _strict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_strict(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return _strict(float(obj))
        return obj

    output_path.write_text(
        json.dumps(_strict(audit), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Print summary
    print("\n" + "=" * 60)
    print("N4 CPU PRE-CHECK SUMMARY")
    print("=" * 60)
    print(f"Sessions: {len(n4_records)}")
    print(f"\nSplit-half reliability (r >= {SPLIT_HALF_RELIABILITY_THRESHOLD}):")
    for dim_name in N4_FEATURE_NAMES:
        gd = gate["per_dim"][dim_name]
        rs = [f"{v:.3f}" for v in gd["per_session_r"] if v is not None and math.isfinite(v)]
        print(f"  {dim_name:25s}: {gd['n_sessions_passing']}/{len(n4_records)} pass  r=[{', '.join(rs)}]  {'PASS' if gd['majority_pass'] else 'FAIL'}")

    print(f"\nDims passing majority gate: {gate['dims_passing_majority']} (need >= 2)")
    print(f"GATE: {'PASSED' if gate['gate_passed'] else 'FAILED'}")

    print(f"\nResidual R² vs T4 (below {RESIDUAL_R2_EXPLAINED_THRESHOLD} = explained by T4):")
    for dim_name in N4_FEATURE_NAMES:
        rs = resid_summary[dim_name]
        print(f"  {dim_name:25s}: mean={rs['mean']:.3f}  [{rs['verdict']}]")

    print(f"\nmean_rate vs T4 baseline_rate Pearson: mean={audit['mean_rate_vs_t4_baseline_rate']['mean']:.3f}")
    print(f"\nCross-session channel identification:")
    print(f"  top-1 accuracy: {ident['top1_accuracy_mean']:.4f} (chance={ident['chance_top1_accuracy']:.4f})")

    print(f"\nAudit written to: {output_path}")
    return audit


if __name__ == "__main__":
    main()
