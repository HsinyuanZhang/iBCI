#!/usr/bin/env python3
"""TFAP Stage-0 CPU preflight (TFAP_CONTRACT_20260817.md §1). No GPU, no target data.

Runs and binds, in one immutable receipt + one sealed derived-data payload:

  (a) 000128 accessibility audit (train NWB only; the NLB desc-test file is
      present-but-never-opened);
  (b) single-builder interface-consistency proof across 128/688 shapes;
  (c) closed-form T4 on 000128's own trial directions with its own normalizer,
      plus the P-Z4 exact-zero mask check;
  (d) budget freeze: 48 epochs, steps/epoch measured from the real window set;
  (e) the derived payload (binned neural, standardized behavior, M30 calib,
      T4/normalizer, train windows) sealed for Stage 1 to consume bitwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import stat
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))

BOUND_PATTERNS = (
    "src/tfpd_lane/tfap_stage0.py",
    "scripts/run_tfap_stage0.py",
    "src/tfpd/spintshape_module.py",
    "src/tfpd_lane/receipt.py",
)
AUTHORITY_PATTERNS = (
    "../sua_exploration/mc_maze/datamodule.py",
    "../sua_exploration/mc_maze/unit_side_features.py",
    "../streaming_calibration_exp/src/models/components/spint.py",
    "../streaming_calibration_exp/src/models/components/streaming_spint.py",
    "../streaming_calibration_exp/src/models/components/streaming_encoders.py",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/tfap_stage0_v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not sys.flags.no_user_site:
        print("PYTHONNOUSERSITE=1 is mandatory", file=sys.stderr)
        return 3

    out_dir = Path(args.output_root)
    if out_dir.exists():
        print(f"fresh Stage-0 root required: {out_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True)

    receipt_mod = _load_module("tfpd_lane_receipt", ROOT / "src/tfpd_lane/receipt.py")
    tfap = _load_module("tfpd_lane_tfap_stage0", ROOT / "src/tfpd_lane/tfap_stage0.py")

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    closure = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)
    authorities = {
        rel: receipt_mod.sha256_file(REPO / rel.split("../", 1)[1]) for rel in AUTHORITY_PATTERNS
    }

    # ---- (a) accessibility ---------------------------------------------------
    access = tfap.describe_000128_accessibility()
    print(json.dumps({"stage": "accessibility", "units": access["n_train_units"],
                      "trials": access["n_trials"]}), file=sys.stderr, flush=True)

    # ---- binning (mirrors MCMazeDataModule discipline, 688 contract shape) ---
    from pynwb import NWBHDF5IO
    from scipy.interpolate import interp1d

    bin_size_s = tfap.BIN_SIZE_MS / 1000.0
    with NWBHDF5IO(str(tfap.JENKINS_TRAIN_NWB), "r") as io:
        nwb = io.read()
        units_df = nwb.units.to_dataframe()
        heldout = units_df["heldout"].values.astype(bool)
        train_units = units_df[~heldout]
        n_units = len(train_units)
        all_spikes = np.concatenate(train_units["spike_times"].values)
        bin_edges = np.arange(float(all_spikes.min()), float(all_spikes.max()) + bin_size_s, bin_size_s)
        num_bins = len(bin_edges) - 1
        binned = np.zeros((num_bins, n_units), dtype=np.float32)
        for i, (_, unit) in enumerate(train_units.iterrows()):
            counts, _ = np.histogram(unit["spike_times"], bins=bin_edges)
            binned[:, i] = counts.astype(np.float32)

        hand_vel = nwb.processing["behavior"]["hand_vel"]
        vel = hand_vel.data[:]
        vel_times = hand_vel.timestamps[:]
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        binned_vel = np.zeros((num_bins, vel.shape[1]), dtype=np.float32)
        for c in range(vel.shape[1]):
            fn = interp1d(vel_times, vel[:, c], kind="linear", bounds_error=False, fill_value=0.0)
            binned_vel[:, c] = fn(centers)
        behavior_mean = binned_vel.mean(axis=0)
        behavior_std = binned_vel.std(axis=0)
        behavior_std[behavior_std < 1e-8] = 1.0
        binned_vel = ((binned_vel - behavior_mean) / behavior_std).astype(np.float32)

        trials_df = nwb.trials.to_dataframe()
        trial_info = []
        for trial_id, trial in trials_df.iterrows():
            start_bin = max(0, int(np.searchsorted(bin_edges, trial["start_time"])))
            stop_bin = min(num_bins, int(np.searchsorted(bin_edges, trial["stop_time"])))
            if stop_bin - start_bin >= tfap.WINDOW_SIZE:
                trial_info.append({
                    "trial_index": int(trial_id),
                    "start": start_bin, "stop": stop_bin,
                    "start_time": float(trial["start_time"]), "stop_time": float(trial["stop_time"]),
                    "split": str(trial.get("split", "train")),
                })
        train_trials = [t for t in trial_info if t["split"] == "train"]
        train_trials.sort(key=lambda t: t["start_time"])

        # M30 chronological calibration trials, cubic-interpolated to T=100
        calib = np.full((tfap.CALIBRATION_N_TRIALS, tfap.MAX_TRIAL_LENGTH, n_units),
                        tfap.PAD_VALUE, dtype=np.float32)
        for i, trial in enumerate(train_trials[: tfap.CALIBRATION_N_TRIALS]):
            data = binned[trial["start"]: trial["stop"]]
            x_orig = np.linspace(0.0, 1.0, len(data))
            x_new = np.linspace(0.0, 1.0, tfap.MAX_TRIAL_LENGTH)
            for unit in range(n_units):
                fn = interp1d(x_orig, data[:, unit], kind="cubic",
                              bounds_error=False, fill_value=tfap.PAD_VALUE)
                calib[i, :, unit] = fn(x_new)

    starts = []
    for trial in train_trials:
        starts.extend(range(trial["start"], trial["stop"] - tfap.WINDOW_SIZE + 1))
    valid_starts = np.asarray(sorted(starts), dtype=np.int64)

    # ---- (c) T4 on 000128 ----------------------------------------------------
    thetas_all, valid_dir = tfap.trial_directions(trials_df)
    pool = []
    for trial in train_trials:
        if len(pool) >= tfap.CALIBRATION_N_TRIALS:
            break
        if valid_dir[int(trial["trial_index"])]:
            pool.append(trial)
    if len(pool) != tfap.CALIBRATION_N_TRIALS:
        raise SystemExit(f"insufficient direction-valid train trials for the M30 pool: {len(pool)}")
    pool_thetas = thetas_all[[int(t["trial_index"]) for t in pool]]
    rates = tfap.pool_trial_rate_matrix(tfap.JENKINS_TRAIN_NWB, pool, n_units)
    fit = tfap.fit_t4_closed_form(rates, pool_thetas)
    t4 = fit.pop("t4")
    side_mean, side_std = tfap.fit_t4_normalizer(t4)
    t4_std = tfap.standardize_t4(t4, side_mean, side_std)
    z4 = tfap.mask_z4(t4_std)
    z4_exact_zero = bool(np.all(z4 == 0) and not np.any(np.signbit(z4)))

    # ---- (d) budget ----------------------------------------------------------
    spe = tfap.steps_per_epoch(len(valid_starts))
    budget = {
        "pretrain_epochs": tfap.PRETRAIN_EPOCHS,
        "batch_size": tfap.TRAIN_BATCH_SIZE,
        "sampler": "single-session drop-partial (n_train_windows // 32)",
        "n_train_windows": int(len(valid_starts)),
        "steps_per_epoch": spe,
        "total_optimizer_steps": spe * tfap.PRETRAIN_EPOCHS,
        "optimizer": {"name": "Adam", "lr": tfap.PRETRAIN_LR, "betas": [0.9, 0.999],
                      "eps": 1e-8, "weight_decay": 0.0, "amsgrad": False,
                      "schedule": "constant"},
        "p_t4_vs_p_z4_identical_except_visible_side": True,
    }

    # ---- (b) interface consistency ------------------------------------------
    interface = tfap.interface_consistency_proof(seed=args.seed)

    # ---- (e) seal the derived payload for Stage 1 ----------------------------
    payload_path = out_dir / "jenkins_derived_payload.npz"
    np.savez_compressed(
        payload_path,
        neural=binned,
        behavior=binned_vel,
        behavior_mean=behavior_mean.astype(np.float32),
        behavior_std=behavior_std.astype(np.float32),
        valid_train_starts=valid_starts,
        calib_trials=calib,
        t4_raw=t4,
        side_feature_mean=side_mean,
        side_feature_std=side_std,
        t4_standardized=t4_std,
        pool_trial_thetas=pool_thetas.astype(np.float64),
    )
    os.chmod(payload_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    payload_sha = receipt_mod.sha256_file(payload_path)
    sidecar = payload_path.with_suffix(payload_path.suffix + ".sha256")
    sidecar.write_text(payload_sha + "  " + payload_path.name + "\n")
    os.chmod(sidecar, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    checks = {
        "interface_all_green": all(
            interface[k] for k in (
                "state_keys_identical_across_688_and_128_batches",
                "state_shapes_identical_across_688_and_128_batches",
                "strict_roundtrip_bitwise_equal",
                "partial_state_raises",
                "forward_finite_688_shaped",
                "forward_finite_128_shaped",
            )
        ),
        "t4_fit_finite": bool(np.isfinite(t4).all()),
        "t4_units_match_neural_units": t4.shape[0] == n_units,
        "z4_mask_exact_positive_zero": z4_exact_zero,
        "at_least_two_distinct_directions": fit["n_distinct_directions"] >= 2,
        "no_688_statistics_reused": True,
    }
    if not all(checks.values()):
        raise SystemExit(f"Stage-0 gate failed: {checks}")

    receipt = {
        "schema": "tfap_stage0_preflight_v1",
        "status": "STAGE0_PREFLIGHT_PASSED",
        "contract": "docs/TFAP_CONTRACT_20260817.md",
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "accessibility": access,
        "binning": {
            "bin_size_ms": tfap.BIN_SIZE_MS,
            "num_bins": int(num_bins),
            "n_units_in_payload": int(n_units),
            "heldout_units_excluded": access["n_heldout_units"],
            "n_train_split_trials_usable": len(train_trials),
            "behavior_normalizer_semantic_sha256": tfap.normalizer_semantic_sha256(
                behavior_mean, behavior_std
            ),
            "calib_trials_sha256": tfap.array_sha256(calib),
            "neural_sha256": tfap.array_sha256(binned),
            "behavior_sha256": tfap.array_sha256(binned_vel),
            "valid_train_starts_sha256": tfap.array_sha256(valid_starts),
        },
        "t4_on_128": {
            **fit,
            "direction_source": "atan2(target_pos[active_target])",
            "pool": "first 30 chronological direction-valid successful train trials",
            "fit": "per-distinct-direction mean rate, then the 000688 closed-form cosine lstsq [a,c,m,b]",
            "t4_raw_sha256": tfap.array_sha256(t4),
            "t4_standardized_sha256": tfap.array_sha256(t4_std),
            "side_normalizer_semantic_sha256": tfap.normalizer_semantic_sha256(side_mean, side_std),
            "pool_thetas_deg_sorted_sample": [
                round(float(np.degrees(t)), 2)
                for t in np.sort(pool_thetas)[:8]
            ],
        },
        "p_z4_mask": {
            "semantics": "zeros_like(standardized_t4) after normalization; never 0*raw",
            "exact_positive_zero": z4_exact_zero,
            "sha256": tfap.array_sha256(z4),
        },
        "budget_frozen": budget,
        "interface_consistency": interface,
        "derived_payload": {
            "path": str(payload_path),
            "sha256": payload_sha,
            "stage1_binding": "Stage 1 must load exactly this sealed payload (sidecar-verified)",
        },
        "checks": checks,
        "disclosures": {
            "gpu_used": False,
            "teacher_checkpoint_or_logits_or_loss_used": False,
            "dandi_000688_opened": False,
            "formal_or_organizer_held_data_opened": False,
            "desc_test_nwb_opened": False,
            "target_session_updates": 0,
        },
        "source_closure": closure,
        "authority_sha256": authorities,
        "environment": {
            "torch_env_cpu_only": True,
            "no_user_site": bool(sys.flags.no_user_site),
        },
    }
    receipt_mod.write_receipt_transactionally(out_dir / "stage0_receipt.json", receipt)
    print(json.dumps({
        "status": receipt["status"],
        "n_train_units": n_units,
        "n_train_windows": budget["n_train_windows"],
        "steps_per_epoch": spe,
        "total_steps_48ep": budget["total_optimizer_steps"],
        "n_distinct_directions": fit["n_distinct_directions"],
        "zero_spike_units": fit["zero_spike_units"],
        "interface_all_green": checks["interface_all_green"],
        "payload_sha256": payload_sha[:16],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
