#!/usr/bin/env python3
"""Priority A2b: same-target label-density dose response.

Fixed: target (dense 2-D velocity), features (50*N), query, standardization, lambda, solver.
Variable: K = number of labeled windows per trial (nested selection).

K ∈ {1, 2, 4, 8, 16, all} with mask seeds 42/43/44.
Equal-trial weighting. Correct weighted ridge with unpenalized intercept.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import sys
import json
import hashlib
import time
import argparse
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "sua_exploration"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sua_exploration.mc_maze import subm_v9_f0_pv_ridge as numerical
from sua_exploration.mc_maze.subm_co_score_only_v2 import recompute_torchmetrics_r2_cpu
from sua_exploration.scripts.run_subm_v9_f0_pv_ridge_controls import (
    load_v9_inputs, load_mean_std, load_v9_target,
    _build_view_base, _nwb_path_and_pin, _runtime_owners,
)
from sua_exploration.mc_maze.multisession_datamodule import _compute_valid_starts

# Reuse the correct weighted ridge from A2a
from sua_exploration.scripts.run_priority_a2_weighting_control import (
    fit_ridge_weighted_correct, predict_ridge_correct, predict_query_batched,
    sha256_array, sha256_file,
)

V9_RUN_ROOT = REPO_ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_v9_formal_20260805"
NWB_ROOT = REPO_ROOT / "sua_exploration/data/dandi_000688"
OUTPUT_DIR = REPO_ROOT / "sua_exploration/results/trial_level_ridge_v1"
VIEWS = ("sua", "pseudo_mua")
BUDGETS = (15, 30, 50)
HISTORY_BINS = numerical.HISTORY_BINS
EXPECTED_SESSIONS = 15
KS = (1, 2, 4, 8, 16, -1)  # -1 means all
MASK_SEEDS = (42,)  # Reduced from (42,43,44) for runtime; seed variance reported from A2a context


def build_nested_masks(trial_of_window, n_windows, trial_indices, K_values, mask_seeds):
    """Build nested per-trial row-selection masks.

    For each trial, shuffle its window indices with a fixed seed, then select first K.
    Masks are nested: K=1 subset K=2 subset K=4 subset ...
    """
    masks = {}
    K_sorted = sorted(k for k in K_values if k > 0)
    for seed in mask_seeds:
        prev_mask = np.zeros(n_windows, dtype=bool)
        for K in K_sorted:
            current_mask = prev_mask.copy()
            for ti in trial_indices:
                trial_rows = np.where(trial_of_window == ti)[0]
                if len(trial_rows) == 0:
                    continue
                local_seed = (seed * 100003 + ti) % (2**32)
                local_rng = np.random.Generator(np.random.PCG64(local_seed))
                perm = local_rng.permutation(len(trial_rows))
                n_select = min(K, len(trial_rows))
                selected_indices = trial_rows[perm[:n_select]]
                current_mask[selected_indices] = True
            # Verify nesting: every True in prev must be True in current
            assert np.all(~prev_mask | current_mask), \
                f"mask not nested at K={K}, seed={seed}"
            prev_mask = current_mask
            masks[(seed, K)] = current_mask

    # K=-1 (all) is always all-True regardless of seed
    for seed in mask_seeds:
        masks[(seed, -1)] = np.ones(n_windows, dtype=bool)

    return masks


def run_batch(session_start, session_end):
    """Run A2b for sessions [session_start, session_end) and write batch JSON."""
    t0 = time.monotonic()
    print(f"[{time.strftime('%H:%M:%S')}] Loading V9 inputs...", file=sys.stderr)
    v9 = load_v9_inputs(V9_RUN_ROOT, repo_root=REPO_ROOT)
    owners = _runtime_owners(REPO_ROOT)
    behavior_stats = {v: load_mean_std(v9.behavior_normalizers[v][0], label=f"{v} behavior") for v in VIEWS}
    cohort = v9.cohort
    assert len(cohort) == EXPECTED_SESSIONS
    cohort_assets = [r.asset_id for r in cohort]

    session_end = min(session_end, len(cohort))

    # results[(view, budget)][K] = {seed: [r2_per_session_in_batch]}
    results = {}
    mask_hashes = {}
    batch_session_assets = []

    for si in range(session_start, session_end):
        row = cohort[si]
        batch_session_assets.append(row.asset_id)
        nwb_path = _nwb_path_and_pin(NWB_ROOT, row)
        print(f"[{time.strftime('%H:%M:%S')}] Session {si}/{session_end}: {row.asset_id}", file=sys.stderr)
        for view in VIEWS:
            mean, std = behavior_stats[view]
            record, _rebuilt, builder_trials, _bridge = _build_view_base(
                repo_root=REPO_ROOT, nwb_path=nwb_path, view=view, mean=mean, std=std, owners=owners,
            )
            v9_target, v9_target_sha, _ = load_v9_target(v9, row, view)

            # Precompute query features once per view (same for all budgets and K values)
            query_features = numerical.raw_window_features(
                record.neural, np.ascontiguousarray(record.valid_starts, dtype=np.int64),
            )

            for budget in BUDGETS:
                key = (view, budget)
                if key not in results:
                    results[key] = {K: {seed: [] for seed in MASK_SEEDS} for K in KS}
                if key not in mask_hashes:
                    mask_hashes[key] = {}

                support = list(builder_trials[:budget])
                cal_starts = np.ascontiguousarray(
                    _compute_valid_starts(support, HISTORY_BINS), dtype=np.int64,
                )
                cal_features = numerical.raw_window_features(record.neural, cal_starts)
                dense_Y = numerical.targets_at_window_end(record.behavior, cal_starts)

                trial_bounds = [(int(t["start"]), int(t["stop"])) for t in support]
                bounds_arr = np.array(trial_bounds, dtype=np.int64)
                trial_of_window = np.empty(len(cal_starts), dtype=np.int64)
                for wi, ws in enumerate(cal_starts):
                    ti = np.searchsorted(bounds_arr[:, 1], ws, side="right")
                    trial_of_window[wi] = min(ti, len(support) - 1)

                trial_indices = sorted(np.unique(trial_of_window).tolist())
                masks = build_nested_masks(
                    trial_of_window, len(cal_starts), trial_indices, KS, MASK_SEEDS,
                )

                # Record mask hashes (first session in batch only)
                if si == session_start:
                    for seed in MASK_SEEDS:
                        for K in KS:
                            mh_key = f"seed{seed}_K{K if K > 0 else 'all'}"
                            mask_hashes[key][mh_key] = sha256_array(masks[(seed, K)].astype(np.int8))

                for seed in MASK_SEEDS:
                    for K in KS:
                        mask = masks[(seed, K)]
                        sel_features = cal_features[mask]
                        sel_targets = dense_Y[mask]
                        sel_trial_of_window = trial_of_window[mask]

                        trial_counts = np.bincount(sel_trial_of_window, minlength=len(support)).astype(np.float64)
                        weights = np.ones(mask.sum(), dtype=np.float64)
                        for wi in range(mask.sum()):
                            weights[wi] = 1.0 / trial_counts[sel_trial_of_window[wi]]
                        weights *= mask.sum() / weights.sum()

                        readout, intercept = fit_ridge_weighted_correct(
                            sel_features, sel_targets, weights,
                            normalized_lambda=numerical.RIDGE_NORMALIZED_LAMBDA,
                        )
                        pred = predict_ridge_correct(query_features, readout, intercept)
                        r2 = float(recompute_torchmetrics_r2_cpu(pred, v9_target))
                        results[key][K][seed].append(r2)

            del record, _rebuilt

    elapsed = time.monotonic() - t0

    # Serialize results keyed by asset_id for combining
    batch_data = {
        "session_start": session_start,
        "session_end": session_end,
        "session_assets": batch_session_assets,
        "results": {},
        "mask_hashes": {f"{k[0]}_M{k[1]}": v for k, v in mask_hashes.items()},
        "elapsed_seconds": elapsed,
    }
    for key, k_dict in results.items():
        vk = f"{key[0]}_M{key[1]}"
        batch_data["results"][vk] = {}
        for K in KS:
            K_label = K if K > 0 else "all"
            batch_data["results"][vk][f"K{K_label}"] = {}
            for seed in MASK_SEEDS:
                batch_data["results"][vk][f"K{K_label}"][str(seed)] = {
                    batch_session_assets[i]: results[key][K][seed][i]
                    for i in range(len(batch_session_assets))
                }

    batch_dir = OUTPUT_DIR / "a2b_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_path = batch_dir / f"batch_{session_start:02d}_{session_end:02d}.json"
    batch_bytes = (json.dumps(batch_data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    batch_path.write_bytes(batch_bytes)
    os.chmod(batch_path, 0o444)
    print(f"[{time.strftime('%H:%M:%S')}] Batch {session_start}-{session_end} written: {batch_path}", file=sys.stderr)
    print(f"[{time.strftime('%H:%M:%S')}] Elapsed: {elapsed:.1f}s", file=sys.stderr)


def combine_batches():
    """Read all batch files and assemble the final receipt."""
    t0 = time.monotonic()
    batch_dir = OUTPUT_DIR / "a2b_batches"
    batch_files = sorted(batch_dir.glob("batch_*.json"))
    if not batch_files:
        print("No batch files found!", file=sys.stderr)
        sys.exit(1)

    print(f"Combining {len(batch_files)} batch files...", file=sys.stderr)

    # Load V9 for cohort info
    v9 = load_v9_inputs(V9_RUN_ROOT, repo_root=REPO_ROOT)
    cohort = v9.cohort
    cohort_assets = [r.asset_id for r in cohort]

    # T4 sealed references
    budget_agg = json.loads((REPO_ROOT / "sua_exploration/results/dandi_000688_subm_v9_t4_label_budget_v1_full/aggregate/endpoint_aggregate_torchmetrics151.json").read_text())
    v9_agg = json.loads((V9_RUN_ROOT / "aggregate/endpoint_aggregate_torchmetrics151.json").read_text())
    t4_ref = {}
    for view in VIEWS:
        t4_ref[view] = {}
        for budget in (15, 30):
            t4_ref[view][budget] = float(np.mean([c["r2"] for c in budget_agg["cells"] if c["view"] == view and c["budget"] == budget]))
        t4_ref[view][50] = float(np.mean([c["r2"] for c in v9_agg["cells"] if c["view"] == view and c["arm"] == "shared_t4"]))

    # Merge batch results: collect per-session r2 values
    merged = {}  # merged[vk][K_label][seed][asset_id] = r2
    mask_hashes_all = {}
    all_session_assets = []
    for bf in batch_files:
        bdata = json.loads(bf.read_text())
        all_session_assets.extend(bdata["session_assets"])
        for vk, k_dict in bdata["results"].items():
            if vk not in merged:
                merged[vk] = {}
            for K_label, seed_dict in k_dict.items():
                if K_label not in merged[vk]:
                    merged[vk][K_label] = {}
                for seed, session_r2 in seed_dict.items():
                    if seed not in merged[vk][K_label]:
                        merged[vk][K_label][seed] = {}
                    merged[vk][K_label][seed].update(session_r2)
        for mk, mv in bdata["mask_hashes"].items():
            if mk not in mask_hashes_all:
                mask_hashes_all[mk] = mv

    assert len(all_session_assets) == EXPECTED_SESSIONS, \
        f"Expected {EXPECTED_SESSIONS} sessions, got {len(all_session_assets)}"

    # Assemble analysis
    all_results = {}
    for view in VIEWS:
        for budget in BUDGETS:
            vk = f"{view}_M{budget}"
            r = {}

            for K in KS:
                K_label = K if K > 0 else "all"
                seed_means = []
                for seed in MASK_SEEDS:
                    arr = np.array([merged[vk][f"K{K_label}"][str(seed)][a] for a in cohort_assets])
                    seed_means.append(float(arr.mean()))
                r[f"K{K_label}"] = {
                    "mean_r2_across_seeds": float(np.mean(seed_means)),
                    "per_seed_mean_r2": {str(s): sm for s, sm in zip(MASK_SEEDS, seed_means)},
                    "mask_seed_variance": float(np.var(seed_means)),
                    "per_session_per_seed": {
                        str(s): {a: merged[vk][f"K{K_label}"][str(s)][a] for a in cohort_assets}
                        for s in MASK_SEEDS
                    },
                }

            # Delta density: all - K1
            all_arr = np.array([merged[vk]["Kall"]["42"][a] for a in cohort_assets])
            k1_arr = np.array([merged[vk]["K1"]["42"][a] for a in cohort_assets])
            delta_density = all_arr - k1_arr
            r["delta_all_minus_K1"] = {
                "seed": 42,
                "mean": float(delta_density.mean()),
                "median": float(np.median(delta_density)),
                "positive_sessions": int(np.sum(delta_density > 0)),
                "per_session": {cohort_assets[i]: float(delta_density[i]) for i in range(EXPECTED_SESSIONS)},
            }

            # Dose-response slope: R² vs log2(K) from K1 to K16
            K_pos = sorted(k for k in KS if k > 0)
            k_values_log2 = [np.log2(k) for k in K_pos]
            r2_at_k = [np.mean([merged[vk][f"K{k}"][str(s)][a] for s in MASK_SEEDS for a in cohort_assets]) for k in K_pos]
            if len(k_values_log2) >= 2:
                slope = (r2_at_k[-1] - r2_at_k[0]) / (k_values_log2[-1] - k_values_log2[0])
                r["dose_response_slope_log2"] = float(slope)
            r["t4_reference"] = t4_ref[view][budget]
            all_results[vk] = r

    # Print summary
    print("\n" + "=" * 120, file=sys.stderr)
    print("PRIORITY A2b: SAME-TARGET LABEL-DENSITY DOSE RESPONSE", file=sys.stderr)
    print("=" * 120, file=sys.stderr)
    k_labels = [k if k > 0 else "all" for k in KS]
    print(f"\n{'B/V':<20} " + " ".join(f"K={kl:<10}" for kl in k_labels) + f" {'T4':<10} {'Δ(all-K1)':<10}", file=sys.stderr)
    for view in VIEWS:
        for budget in BUDGETS:
            vk = f"{view}_M{budget}"
            r = all_results[vk]
            vals = " ".join(f"{r[f'K{kl}']['mean_r2_across_seeds']:<10.4f}" for kl in k_labels)
            delta = r["delta_all_minus_K1"]["mean"]
            print(f"M={budget} {view:<14} {vals} {r['t4_reference']:<10.4f} {delta:<+10.4f}", file=sys.stderr)

    elapsed = time.monotonic() - t0
    receipt = {
        "schema": "priority_a2b_same_target_density_v1",
        "status": "COMPLETED_CPU_ONLY",
        "date": "2026-08-11",
        "definition": (
            "Same target (dense 2-D velocity), same features (50*N), same query, "
            "same standardization, same lambda=1, same solver. "
            "Only K varies: number of labeled windows per trial, nested selection. "
            "K ∈ {1,2,4,8,16,all}, mask seed 42, equal-trial weighting."
        ),
        "cohort": [{"asset_id": r.asset_id, "session_id": r.session_id} for r in cohort],
        "results": all_results,
        "mask_hashes": mask_hashes_all,
        "mask_nesting_verified": True,
        "batch_files": [bf.name for bf in sorted(batch_dir.glob("batch_*.json"))],
        "runner_sha256": sha256_file(Path(__file__)),
        "v9_manifest_sha256": v9.manifest_sha256,
        "non_interference": {
            "gpu_used": False,
            "thread_caps": "OMP/MKL/OPENBLAS=4, nice -n 10",
            "output_directory": str(OUTPUT_DIR),
        },
        "elapsed_combine_seconds": elapsed,
    }
    receipt_bytes = (json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    receipt_path = OUTPUT_DIR / "priority_a2_same_target_density_receipt.json"
    receipt_path.write_bytes(receipt_bytes)
    os.chmod(receipt_path, 0o444)
    print(f"\nReceipt: {receipt_path}", file=sys.stderr)
    print(f"SHA-256: {hashlib.sha256(receipt_bytes).hexdigest()}", file=sys.stderr)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Priority A2b: same-target label-density dose response")
    parser.add_argument("--session-start", type=int, default=0, help="Start session index (inclusive)")
    parser.add_argument("--session-end", type=int, default=15, help="End session index (exclusive)")
    parser.add_argument("--combine", action="store_true", help="Combine batch files and write final receipt")
    args = parser.parse_args()

    if args.combine:
        combine_batches()
    else:
        run_batch(args.session_start, args.session_end)
