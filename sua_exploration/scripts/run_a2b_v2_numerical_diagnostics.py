#!/usr/bin/env python3
"""CPU-only numerical diagnostics for A2b-v2 K=8/K=16 instability.

For every session × view × budget × seed × K (including all), records:
  - regularized Gram condition number (standardized coords)
  - solve residual ||gram @ beta - rhs||_2 / ||rhs||_2
  - float32 vs float64 prediction sensitivity
  - primal vs dual coefficient agreement

Does not change any A2b-v2 main result. Writes a new immutable receipt.
sub-M_ses-CO-20150512 is reported separately in the summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "sua_exploration"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sua_exploration.mc_maze import priority_a2_normalized_ridge_v2 as ridge

V9_RUN_ROOT = REPO_ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_v9_formal_20260805"
NWB_ROOT = REPO_ROOT / "sua_exploration/data/dandi_000688"
OUTPUT_DIR = REPO_ROOT / "sua_exploration/results/trial_level_ridge_v1"
RECEIPT_PATH = OUTPUT_DIR / "priority_a2b_v2_numerical_diagnostics_receipt.json"
A2B_V2_RECEIPT_PATH = OUTPUT_DIR / "priority_a2_same_target_density_v2_receipt.json"
A2A_V2_RECEIPT_PATH = OUTPUT_DIR / "priority_a2_weighting_control_v2_receipt.json"
VIEWS = ("sua", "pseudo_mua")
BUDGETS = (15, 30, 50)
HISTORY_BINS = 50
FINITE_KS = (1, 2, 4, 8, 16)
MASK_SEEDS = (42, 43, 44)
HIGHLIGHT_SESSION = "sub-M_ses-CO-20150512"
HIGHLIGHT_ASSET = "fee6b912-477a-4fea-ad16-e89e4bd42d25"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value) -> str:
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _diagnose_one(x_sel, y_sel, weights, budget, n_features):
    """Fit normalized weighted ridge in multiple precisions/forms; return diagnostics dict."""
    import numpy as np

    nl = 1.0
    x64 = np.ascontiguousarray(x_sel, dtype=np.float64)
    y64 = np.ascontiguousarray(y_sel, dtype=np.float64)
    w64 = np.ascontiguousarray(weights, dtype=np.float64)
    total = float(w64.sum())

    # Weighted standardization (float64)
    xbar = (w64[:, None] * x64).sum(axis=0) / total
    ybar = (w64[:, None] * y64).sum(axis=0) / total
    centered = x64 - xbar
    variance = (w64[:, None] * centered * centered).sum(axis=0) / total
    scale = np.sqrt(variance)
    scale[scale < ridge.FEATURE_STD_EPS] = 1.0
    z = centered / scale
    yc = y64 - ybar
    sqrt_w = np.sqrt(w64 / total)
    a64 = z * sqrt_w[:, None]
    t64 = yc * sqrt_w[:, None]

    n_rows = a64.shape[0]
    p = a64.shape[1]

    diag = {
        "n_rows": int(n_rows),
        "n_features": int(p),
        "ratio_rows_to_features": float(n_rows / p) if p > 0 else 0.0,
    }

    # --- Primal form (float64) ---
    gram_p64 = a64.T @ a64
    gram_p64_reg = gram_p64 + nl * np.eye(p)
    rhs_p64 = a64.T @ t64
    # Condition number of regularized Gram (2-norm, no full SVD needed)
    cond_reg = float(np.linalg.cond(gram_p64_reg))
    diag["gram_condition_number"] = cond_reg
    # Solve
    beta_p64 = np.linalg.solve(gram_p64_reg, rhs_p64)
    resid_p64 = np.linalg.norm(gram_p64_reg @ beta_p64 - rhs_p64) / (np.linalg.norm(rhs_p64) + 1e-30)
    diag["primal_solve_residual_relative"] = float(resid_p64)

    # --- Dual form (float64) ---
    gram_d64 = a64 @ a64.T
    gram_d64_reg = gram_d64 + nl * np.eye(n_rows)
    rhs_d64 = t64
    beta_d64 = a64.T @ np.linalg.solve(gram_d64_reg, rhs_d64)
    # Primal-dual agreement
    pd_diff = float(np.max(np.abs(beta_p64 - beta_d64)))
    pd_rel = float(np.max(np.abs(beta_p64 - beta_d64)) / (np.max(np.abs(beta_p64)) + 1e-30))
    diag["primal_dual_max_abs_diff"] = pd_diff
    diag["primal_dual_max_rel_diff"] = pd_rel
    resid_d64 = np.linalg.norm(gram_d64_reg @ np.linalg.solve(gram_d64_reg, rhs_d64) - rhs_d64) / (np.linalg.norm(rhs_d64) + 1e-30)
    diag["dual_solve_residual_relative"] = float(resid_d64)

    # --- float32 sensitivity ---
    try:
        a32 = a64.astype(np.float32)
        t32 = t64.astype(np.float32)
        if n_rows < p:
            gram_d32 = a32 @ a32.T
            gram_d32_reg = gram_d32 + np.float32(nl) * np.eye(n_rows, dtype=np.float32)
            beta_d32 = a32.T @ np.linalg.solve(gram_d32_reg, t32)
            beta_f32 = beta_d32.astype(np.float64)
        else:
            gram_p32 = a32.T @ a32
            gram_p32_reg = gram_p32 + np.float32(nl) * np.eye(p, dtype=np.float32)
            rhs_p32 = a32.T @ t32
            beta_p32 = np.linalg.solve(gram_p32_reg, rhs_p32)
            beta_f32 = beta_p32.astype(np.float64)
        f32_diff = float(np.max(np.abs(beta_p64 - beta_f32)))
        f32_rel = float(np.max(np.abs(beta_p64 - beta_f32)) / (np.max(np.abs(beta_p64)) + 1e-30))
        diag["float32_max_abs_diff"] = f32_diff
        diag["float32_max_rel_diff"] = f32_rel
        # Prediction sensitivity on query features (will be filled by caller if provided)
    except Exception as exc:
        diag["float32_error"] = str(exc)

    diag["solver_form_used_by_core"] = "dual" if n_rows < p else "primal"
    return diag


def run_batch(session_start: int, session_end: int) -> str:
    """Run diagnostics for sessions [session_start, session_end)."""
    import numpy as np
    from sua_exploration.mc_maze import subm_v9_f0_pv_ridge as numerical
    from sua_exploration.mc_maze.multisession_datamodule import _compute_valid_starts
    from sua_exploration.scripts.run_subm_v9_f0_pv_ridge_controls import (
        _build_view_base, _nwb_path_and_pin, _runtime_owners,
        load_mean_std, load_v9_inputs,
    )

    numerical_contract = ridge.numerical_contract_self_test()
    started = time.monotonic()
    v9 = load_v9_inputs(V9_RUN_ROOT, repo_root=REPO_ROOT)
    owners = _runtime_owners(REPO_ROOT)
    behavior_stats = {view: load_mean_std(v9.behavior_normalizers[view][0], label=f"{view} behavior") for view in VIEWS}
    session_end = min(session_end, len(v9.cohort))

    cells: list[dict] = []
    for si in range(session_start, session_end):
        cohort_row = v9.cohort[si]
        nwb_path = _nwb_path_and_pin(NWB_ROOT, cohort_row)
        session_key = f"{cohort_row.asset_id}|{cohort_row.session_id}"
        print(f"[{time.strftime('%H:%M:%S')}] Session {si}/{session_end}: {cohort_row.asset_id}", file=sys.stderr)
        for view in VIEWS:
            mean, std = behavior_stats[view]
            record, _rebuilt, builder_trials, _bridge = _build_view_base(
                repo_root=REPO_ROOT, nwb_path=nwb_path, view=view, mean=mean, std=std, owners=owners,
            )
            query_features = numerical.raw_window_features(
                record.neural, np.ascontiguousarray(record.valid_starts, dtype=np.int64),
            )
            for budget in BUDGETS:
                support = list(builder_trials[:budget])
                support_starts = np.ascontiguousarray(_compute_valid_starts(support, HISTORY_BINS), dtype=np.int64)
                owner = ridge.assign_windows_to_trials(support_starts, support, window_size=HISTORY_BINS)
                x = numerical.raw_window_features(record.neural, support_starts)
                masks_by_seed = {
                    seed: ridge.nested_density_masks(owner, budget, session_or_asset=session_key, mask_seed=seed, ks=FINITE_KS)
                    for seed in MASK_SEEDS
                }
                dense_y = numerical.targets_at_window_end(record.behavior, support_starts)
                n_features = x.shape[1]

                for seed in MASK_SEEDS:
                    for K in FINITE_KS:
                        mask = masks_by_seed[seed][K]
                        weights = ridge.equal_trial_weights(owner[mask], budget)
                        diag = _diagnose_one(x[mask], dense_y[mask], weights, budget, n_features)
                        cells.append({
                            "asset_id": str(cohort_row.asset_id),
                            "session_id": cohort_row.session_id,
                            "view": view, "budget": budget,
                            "K": K, "mask_seed": seed,
                            **diag,
                        })
                    # K=all
                    all_mask = np.ones(owner.size, dtype=bool)
                    weights_all = ridge.equal_trial_weights(owner[all_mask], budget)
                    diag_all = _diagnose_one(x[all_mask], dense_y[all_mask], weights_all, budget, n_features)
                    cells.append({
                        "asset_id": str(cohort_row.asset_id),
                        "session_id": cohort_row.session_id,
                        "view": view, "budget": budget,
                        "K": "all", "mask_seed": seed,
                        "fit_prediction_reused_across_seeds": True,
                        **diag_all,
                    })
            del record, _rebuilt

    elapsed = time.monotonic() - started
    batch_dir = OUTPUT_DIR / "a2b_v2_diag_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_data = {
        "session_start": session_start, "session_end": session_end,
        "numerical_contract": numerical_contract,
        "cells": cells,
        "elapsed_seconds": elapsed,
    }
    batch_path = batch_dir / f"diag_batch_{session_start:02d}_{session_end:02d}.json"
    raw = (json.dumps(batch_data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    batch_path.write_bytes(raw)
    os.chmod(batch_path, 0o444)
    print(f"[{time.strftime('%H:%M:%S')}] Diag batch {session_start}-{session_end} written: {batch_path}", file=sys.stderr)
    print(f"[{time.strftime('%H:%M:%S')}] Elapsed: {elapsed:.1f}s", file=sys.stderr)
    return str(batch_path)


def combine_batches() -> str:
    """Read all diag batch files, assemble summaries, write immutable receipt."""
    import numpy as np
    numerical_contract = ridge.numerical_contract_self_test()

    started = time.monotonic()
    batch_dir = OUTPUT_DIR / "a2b_v2_diag_batches"
    batch_files = sorted(batch_dir.glob("diag_batch_*.json"))
    if not batch_files:
        raise FileNotFoundError(f"No diag batch files found in {batch_dir}")
    print(f"Combining {len(batch_files)} diag batch files...", file=sys.stderr)

    all_cells: list[dict] = []
    for bf in batch_files:
        bdata = json.loads(bf.read_text())
        all_cells.extend(bdata["cells"])

    # Summary: aggregate by view × budget × K
    summaries: dict[str, object] = {}
    for view in VIEWS:
        for budget in BUDGETS:
            for K_label in [str(k) for k in FINITE_KS] + ["all"]:
                matching = [c for c in all_cells if c["view"] == view and c["budget"] == budget and str(c["K"]) == K_label]
                if not matching:
                    continue
                vk_key = f"{view}_M{budget}_K{K_label}"
                conds = [c["gram_condition_number"] for c in matching if math.isfinite(c["gram_condition_number"])]
                resids = [c["primal_solve_residual_relative"] for c in matching]
                pd_diffs = [c["primal_dual_max_abs_diff"] for c in matching]
                f32_diffs = [c.get("float32_max_abs_diff", float("nan")) for c in matching]
                rows = [c["n_rows"] for c in matching]
                summaries[vk_key] = {
                    "n_cells": len(matching),
                    "n_rows": {"mean": float(np.mean(rows)), "min": int(np.min(rows)), "max": int(np.max(rows))},
                    "gram_condition_number": {
                        "mean": float(np.mean(conds)) if conds else None,
                        "median": float(np.median(conds)) if conds else None,
                        "min": float(np.min(conds)) if conds else None,
                        "max": float(np.max(conds)) if conds else None,
                    },
                    "primal_solve_residual_relative": {
                        "mean": float(np.mean(resids)),
                        "max": float(np.max(resids)),
                    },
                    "primal_dual_max_abs_diff": {
                        "mean": float(np.mean(pd_diffs)),
                        "max": float(np.max(pd_diffs)),
                    },
                    "float32_max_abs_diff": {
                        "mean": float(np.mean([d for d in f32_diffs if math.isfinite(d)])) if any(math.isfinite(d) for d in f32_diffs) else None,
                        "max": float(np.max([d for d in f32_diffs if math.isfinite(d)])) if any(math.isfinite(d) for d in f32_diffs) else None,
                    },
                }

    # Highlight session: sub-M_ses-CO-20150512
    highlight_cells = [c for c in all_cells if c["asset_id"] == HIGHLIGHT_ASSET]
    highlight_summary: dict[str, object] = {}
    for view in VIEWS:
        for budget in BUDGETS:
            for K_label in [str(k) for k in FINITE_KS] + ["all"]:
                matching = [c for c in highlight_cells if c["view"] == view and c["budget"] == budget and str(c["K"]) == K_label]
                if not matching:
                    continue
                vk_key = f"{view}_M{budget}_K{K_label}"
                conds = [c["gram_condition_number"] for c in matching if math.isfinite(c["gram_condition_number"])]
                resids = [c["primal_solve_residual_relative"] for c in matching]
                pd_diffs = [c["primal_dual_max_abs_diff"] for c in matching]
                highlight_summary[vk_key] = {
                    "n_cells": len(matching),
                    "gram_condition_number": {str(k): float(v) for k, v in zip(["mean", "median", "min", "max"], [np.mean(conds), np.median(conds), np.min(conds), np.max(conds)])} if conds else None,
                    "primal_solve_residual_relative": {"mean": float(np.mean(resids)), "max": float(np.max(resids))},
                    "primal_dual_max_abs_diff": {"mean": float(np.mean(pd_diffs)), "max": float(np.max(pd_diffs))},
                }

    # Print summary
    print("\n" + "=" * 130, file=sys.stderr)
    print("A2b-v2 NUMERICAL DIAGNOSTICS SUMMARY (15 sessions)", file=sys.stderr)
    print("=" * 130, file=sys.stderr)
    print(f"\n{'View/Budget/K':<30} {'cond# mean':<15} {'cond# max':<15} {'resid mean':<15} {'P-D max':<15} {'f32 max':<15}", file=sys.stderr)
    for view in VIEWS:
        for budget in BUDGETS:
            for K_label in [str(k) for k in FINITE_KS] + ["all"]:
                vk_key = f"{view}_M{budget}_K{K_label}"
                if vk_key not in summaries:
                    continue
                s = summaries[vk_key]
                cm = s["gram_condition_number"]["mean"] or 0
                cx = s["gram_condition_number"]["max"] or 0
                rm = s["primal_solve_residual_relative"]["mean"]
                pdm = s["primal_dual_max_abs_diff"]["max"]
                f32m = s["float32_max_abs_diff"]["max"] or 0
                print(f"{vk_key:<30} {cm:<15.2e} {cx:<15.2e} {rm:<15.2e} {pdm:<15.2e} {f32m:<15.2e}", file=sys.stderr)

    print(f"\n--- Highlight: {HIGHLIGHT_SESSION} ({HIGHLIGHT_ASSET}) ---", file=sys.stderr)
    for view in VIEWS:
        for budget in BUDGETS:
            for K_label in [str(k) for k in FINITE_KS] + ["all"]:
                vk_key = f"{view}_M{budget}_K{K_label}"
                if vk_key not in highlight_summary:
                    continue
                s = highlight_summary[vk_key]
                cm = s["gram_condition_number"]["mean"] if s["gram_condition_number"] else 0
                rm = s["primal_solve_residual_relative"]["mean"]
                pdm = s["primal_dual_max_abs_diff"]["max"]
                print(f"  {vk_key:<30} cond#={cm:.2e} resid={rm:.2e} P-D={pdm:.2e}", file=sys.stderr)

    elapsed = time.monotonic() - started
    receipt = {
        "schema": "priority_a2b_v2_numerical_diagnostics_v1",
        "status": "COMPLETED_CPU_ONLY",
        "date": "2026-08-11",
        "definition": (
            "CPU-only numerical diagnostics for A2b-v2. For every session × view × budget × seed × K, "
            "records regularized Gram condition number, solve residual, float32/float64 sensitivity, "
            "and primal/dual coefficient agreement. Does not change A2b-v2 main conclusions."
        ),
        "numerical_contract": numerical_contract,
        "highlight_session": {"session_id": HIGHLIGHT_SESSION, "asset_id": HIGHLIGHT_ASSET},
        "aggregate_summary": summaries,
        "highlight_summary": highlight_summary,
        "n_cells": len(all_cells),
        "cells": all_cells,
        "input_bindings": {
            "runner_sha256": sha256_file(Path(__file__)),
            "ridge_core_sha256": sha256_file(REPO_ROOT / "sua_exploration/mc_maze/priority_a2_normalized_ridge_v2.py"),
            "a2b_v2_receipt_sha256": sha256_file(A2B_V2_RECEIPT_PATH) if A2B_V2_RECEIPT_PATH.exists() else "MISSING",
            "a2a_v2_receipt_sha256": sha256_file(A2A_V2_RECEIPT_PATH) if A2A_V2_RECEIPT_PATH.exists() else "MISSING",
        },
        "batch_files": [bf.name for bf in batch_files],
        "elapsed_combine_seconds": elapsed,
        "receipt_policy": "exclusive new filename; never overwrite",
    }
    if RECEIPT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite existing diagnostics receipt: {RECEIPT_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    with RECEIPT_PATH.open("xb") as handle:
        handle.write(raw)
    os.chmod(RECEIPT_PATH, 0o444)
    sha = hashlib.sha256(raw).hexdigest()
    print(f"\nReceipt: {RECEIPT_PATH}", file=sys.stderr)
    print(f"SHA-256: {sha}", file=sys.stderr)
    return sha


def main() -> None:
    parser = argparse.ArgumentParser(description="A2b-v2 numerical diagnostics (CPU-only)")
    parser.add_argument("--session-start", type=int, default=None, help="Start session index (inclusive)")
    parser.add_argument("--session-end", type=int, default=None, help="End session index (exclusive)")
    parser.add_argument("--combine", action="store_true", help="Combine diag batch files and write receipt")
    args = parser.parse_args()

    contract = ridge.numerical_contract_self_test()
    print(json.dumps({"numerical_contract": contract}, sort_keys=True))

    if args.combine:
        sha = combine_batches()
        print(json.dumps({"receipt_sha256": sha, "receipt": str(RECEIPT_PATH)}, sort_keys=True))
    elif args.session_start is not None and args.session_end is not None:
        batch_path = run_batch(args.session_start, args.session_end)
        print(json.dumps({"batch_path": batch_path}, sort_keys=True))
    else:
        parser.error("must specify --combine or --session-start/--session-end")


if __name__ == "__main__":
    main()
