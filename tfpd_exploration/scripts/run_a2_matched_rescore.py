#!/usr/bin/env python3
"""Read-only matched rescoring of the sealed A2 references (user directive).

Scores the A2 teacher-initialized sources (T4 and Z4 arms, seeds 42/43/44) on
the SAME two surfaces and with the SAME matched metric family as arm A /
P-T4 / P-Z4 (per-session variance-weighted R2, equal weight per session, M30
chronological calibration, source-only normalizer injection on external),
WITHOUT touching any A2 file:

- A2 checkpoint authority: the per-cell sealed run directories resolved from
  `mc_maze.a2_matched_subject_shift_v2_core` (checkpoint roots bound by the A2
  cell launch receipts); the scored artifact per cell is the FINAL epoch
  checkpoint (epoch_011.ckpt, one-based epoch 12) — the single deployable
  endpoint, chosen because arm A is also a single deployable SWA endpoint.
  A2's own sealed estimand (the epochs-5..12 window mean) is CITED alongside
  as lineage cross-check, not rescored.
- A2 deployment prediction semantics are honored exactly: the student
  StreamingSpintModel output is taken at the LAST timestep of each 50-bin
  window and divided by behavior_scaling_factor (5.0)
  (`_slice_last_timestep`/`decode_last_behavior` in the A2 lineage).  For a
  matched query granularity, arm A is rescored at the SAME last-bin
  granularity (its full-window numbers from Gate-3/4 remain the deployable
  reference and are cited).
- Model reconstruction is authoritative: `StreamingCalibrationLitModule.
  load_from_checkpoint` (teacher checkpoint opened READ-ONLY for the sealed
  initialization lineage, then the trained state overwrites it).

Outputs (0444 receipt): per-session R2 for every A2 cell and arm A on both
surfaces, per-seed paired deltas (A2_T4 - armA_lastbin) with the full §10
statistics (mean/median/n-positive/min-max/all deltas/bootstrap CI/sign
pattern), the 3-seed pooled paired statistics, the A2 Z4 arm means, and
cross-checks against the sealed A2 per-epoch-12 per-session values.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "sua_exploration" / "scripts"))

PAD_VALUE = -1.0
AUTH_VALUE = "I_AUTHORIZE_TFPD_DEVELOPMENT_TARGET_ACCESS"
BEHAVIOR_SCALING_FACTOR = 5.0
EVAL_BATCH_SIZE = 128
BOUND_PATTERNS = (
    "scripts/run_a2_matched_rescore.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "src/tfpd_lane/arm_common.py",
)
A2_FINAL_CKPT = "epoch_011.ckpt"  # one-based epoch 12 = final of the 12-epoch A2 training


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_surfaces(args, a2, side_group: str):
    from mc_maze.multisession_datamodule import (
        Dandi688MultiSessionDataModule,
        fit_behavior_stats,
    )
    from mc_maze.unit_side_features import base_feature_group, fit_side_feature_stats

    within = Dandi688MultiSessionDataModule(
        data_dir=str(a2.SUBC_DATA_ROOT), task="CO", split_counts=(27, 6, 6), batch_size=32,
        window_size=50, calibration_n_trials=30, max_trial_length=100, bin_size_ms=20,
        num_workers=args.num_workers, random_calibration=False, seed=42,
        max_units_exclusive=100, cache_dir=str(a2.SOURCE_CACHE_ROOT), signal_view="sua",
        side_feature_group=side_group, side_feature_pool_size=30,
        train_val_manifest_path=str(a2.MANIFEST_PATH),
    )
    within.setup("validate")
    ext = Dandi688MultiSessionDataModule(
        data_dir=str(a2.SUBM_DATA_ROOT), task="CO", split_counts=(0, 15, 0), batch_size=32,
        window_size=50, calibration_n_trials=30, max_trial_length=100, bin_size_ms=20,
        num_workers=args.num_workers, random_calibration=False, seed=42,
        max_units_exclusive=100, cache_dir=str(ROOT / "cache/subm_external_v1"),
        signal_view="sua", side_feature_group=side_group, side_feature_pool_size=30,
    )
    train_paths, _v, _n = a2.active_source_session_paths()
    mean, std = fit_behavior_stats(train_paths, 20, cache_dir=a2.SOURCE_CACHE_ROOT)
    semantic = a2.normalizer_value_sha256(mean, std)
    if not semantic.startswith("f062506c"):
        raise SystemExit("source behavior normalizer semantic SHA drift")
    ext._behavior_stats = (mean, std)
    # stats are fitted on the BASE group (t4 for the z4-masked arm); only the
    # per-session loader applies the z4 mask after normalization
    side_mean, side_std = fit_side_feature_stats(
        train_paths, feature_group=base_feature_group(side_group), pool_size=30,
        cache_dir=a2.SOURCE_CACHE_ROOT, signal_view="sua",
    )
    ext._side_feature_stats = (side_mean, side_std)
    ext.setup("validate")

    def starts(dataset):
        table = {}
        for _pos, (session, start) in enumerate(dataset.window_indices):
            table.setdefault(session, []).append(int(start))
        return table

    return (within.val_dataset, starts(within.val_dataset)), (ext.val_dataset, starts(ext.val_dataset))


def score_last_bin(forward, dataset, starts_by_session, device, scorer,
                   output_scale: float = 1.0):
    """Deployment-native granularity: last timestep of each window.

    `forward(neural, calib, side)` returns the RAW decoder output [B, W, 2].
    `output_scale` applies the system's deployment scaling exactly once:
    BEHAVIOR_SCALING_FACTOR (5.0) for the A2 lineage (trained on scaled
    targets), 1.0 for arm A (trained on standardized behavior directly).
    Targets: last bin of each window's standardized behavior.
    """
    per_session = []
    with torch.no_grad():
        if torch.is_grad_enabled():
            raise SystemExit("scoring must run with autograd disabled")
        for session in sorted(starts_by_session):
            record = dataset.sessions[session]
            starts = starts_by_session[session]
            calib = torch.from_numpy(record.calib_trials.copy()).float().unsqueeze(0).to(device)
            side = torch.from_numpy(record.side_features.copy()).float().unsqueeze(0).to(device)
            preds, tgts = [], []
            for i in range(0, len(starts), EVAL_BATCH_SIZE):
                chunk = starts[i : i + EVAL_BATCH_SIZE]
                neural = (
                    torch.from_numpy(np.stack([record.neural[s : s + 50] for s in chunk]))
                    .float()
                    .to(device)
                )
                behavior = (
                    torch.from_numpy(np.stack([record.behavior[s : s + 50] for s in chunk]))
                    .float()
                    .to(device)
                )
                out = forward(neural, calib.expand(neural.shape[0], -1, -1, -1),
                              side.expand(neural.shape[0], -1, -1))
                raw = out[0] if isinstance(out, tuple) else out  # (behavior, identity)
                pred = raw[:, -1, :] / output_scale
                target = behavior[:, -1, :]
                valid = (behavior[:, -1, :] != PAD_VALUE).all(dim=-1)
                preds.append(pred[valid].cpu())
                tgts.append(target[valid].cpu())
            per_session.append(
                {"session": session, "r2": scorer(torch.cat(preds), torch.cat(tgts)),
                 "n_windows": len(starts)}
            )
    return {
        "per_session": per_session,
        "mean_r2": float(np.mean([row["r2"] for row in per_session])),
        "n_sessions": len(per_session),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/a2_matched_rescore_v1")
    parser.add_argument("--authorize-target", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not sys.flags.no_user_site:
        print("PYTHONNOUSERSITE=1 is mandatory", file=sys.stderr)
        return 3
    authorized = args.authorize_target == AUTH_VALUE
    if not args.dry_run and not authorized:
        print("external scoring requires --authorize-target " + AUTH_VALUE, file=sys.stderr)
        return 3

    receipt_mod = _load_module("tfpd_lane_receipt", ROOT / "src/tfpd_lane/receipt.py")
    arm_common = _load_module("tfpd_lane_arm_common", ROOT / "src/tfpd_lane/arm_common.py")
    matched_scorer = _load_module(
        "tfpd_lane_matched_scorer", ROOT / "src/tfpd_lane/matched_scorer.py"
    )
    import mc_maze.a2_matched_subject_shift_v2_core as a2

    plan = {
        "schema": "tfpd_a2_matched_rescore_v1",
        "seeds": args.seeds,
        "arms": ["t4", "z4"],
        "surfaces": ["within_dev_6", "external_sub_M_15"],
        "a2_endpoint": f"final checkpoint {A2_FINAL_CKPT} (one-based epoch 12)",
        "granularity": "last-timestep-of-window; A2 output divided by behavior_scaling_factor 5.0",
        "authorized": authorized,
    }
    if args.dry_run:
        print(json.dumps({**plan, "status": "DRY_RUN__NO_NWB_OPENED"}, indent=1))
        return 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA unavailable ({args.device})", file=sys.stderr)
        return 3
    out_dir = Path(args.output_root)
    if out_dir.exists():
        print(f"fresh output root required: {out_dir}", file=sys.stderr)
        return 2
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_dir.mkdir(parents=True)
    closure = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)

    # teacher authority (read-only) before any model load
    if arm_common.sha256_file(a2.TEACHER_PATH) != a2.EXPECTED_TEACHER_SHA256:
        raise SystemExit("A2 teacher checkpoint SHA drift against the sealed expectation")
    from select_gradient_free_protocol_dandi688 import load_frozen_model

    # ---- checkpoint authority (read-only) ------------------------------------
    cells = {}
    for arm in ("t4", "z4"):
        for seed in args.seeds:
            run_dir = a2.source_run_dir(f"source_{arm}", seed)
            ckpt = run_dir / "epoch_ckpts" / A2_FINAL_CKPT
            if not ckpt.is_file():
                raise SystemExit(f"A2 checkpoint missing: {ckpt}")
            cells[(arm, seed)] = {
                "path": str(ckpt),
                "sha256": arm_common.sha256_file(ckpt),
                "run_dir": str(run_dir),
            }

    results = {}
    integrity = {}
    surfaces_by_group = {}
    for side_group in ("t4", "z4"):
        surfaces_by_group[side_group] = build_surfaces(args, a2, side_group)

    for (arm, seed), info in sorted(cells.items()):
        lit = load_frozen_model(
            Path(info["path"]), Path(a2.TEACHER_PATH), "B3S", torch.device("cpu"),
            identity_mode="calibrated",
        )
        student = lit.student
        if student is None:
            raise SystemExit("A2 student model absent after load")
        student.to(device).eval()
        state_before = arm_common.state_sha256(student)

        def a2_forward(neural, calib, side, _s=student):
            return _s(neural, calib_trials=calib, side_features=side)

        (within_ds, within_starts), (ext_ds, ext_starts) = surfaces_by_group[arm]
        name = f"A2_{arm}_s{seed}"
        results[name] = {
            "within": score_last_bin(a2_forward, within_ds, within_starts, device,
                                     matched_scorer.session_r2,
                                     output_scale=BEHAVIOR_SCALING_FACTOR),
            "external": score_last_bin(a2_forward, ext_ds, ext_starts, device,
                                       matched_scorer.session_r2,
                                       output_scale=BEHAVIOR_SCALING_FACTOR),
        }
        state_after = arm_common.state_sha256(student)
        integrity[name] = {
            **info,
            "state_unchanged_during_scoring": state_before == state_after,
            "grads_all_none_after": all(
                p.grad is None for p in student.parameters()
                if not isinstance(p, torch.nn.parameter.UninitializedParameter)
            ),
        }
        if state_before != state_after:
            raise SystemExit(f"A2 state mutated during scoring: {name}")
        print(json.dumps({
            "model": name,
            "within_native": round(results[name]["within"]["mean_r2"], 6),
            "external_native": round(results[name]["external"]["mean_r2"], 6),
        }), flush=True)
        del lit, student
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- arm A at the matched last-bin granularity ---------------------------
    spintshape = _load_module("tfpd_spintshape_module", ROOT / "src/tfpd/spintshape_module.py")
    armA_path = ROOT / "results/admission_arms_v1/armA_direct_t4_48/swa_final4.pt"
    armA_sidecar = Path(str(armA_path) + ".sha256")
    armA_sha = arm_common.sha256_file(armA_path)
    if armA_sha != armA_sidecar.read_text().split()[0]:
        raise SystemExit("arm A SWA SHA mismatch")
    state = torch.load(armA_path, map_location="cpu", weights_only=False)["state_dict"]
    state = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in state.items()}
    armA = spintshape.build_spintshape_model(seed=42)
    armA.load_state_dict(state, strict=True)
    armA.to(device).eval()
    (within_ds, within_starts), (ext_ds, ext_starts) = surfaces_by_group["t4"]

    def arma_forward(neural, calib, side, _m=armA):
        return _m(neural, calib_trials=calib, side_features=side)

    armA_lastbin = {
        "within": score_last_bin(arma_forward, within_ds, within_starts, device,
                                 matched_scorer.session_r2),
        "external": score_last_bin(arma_forward, ext_ds, ext_starts, device,
                                   matched_scorer.session_r2),
    }
    results["armA_direct_t4_48_swa_lastbin"] = armA_lastbin
    integrity["armA_direct_t4_48_swa_lastbin"] = {
        "path": str(armA_path), "sha256": armA_sha,
        "note": "arm A rescored at the last-bin granularity to match A2's deployment query semantics; its full-window deployable numbers stay sealed in Gate-3/4",
    }
    print(json.dumps({
        "model": "armA_lastbin",
        "within_native": round(armA_lastbin["within"]["mean_r2"], 6),
        "external_native": round(armA_lastbin["external"]["mean_r2"], 6),
    }), flush=True)
    del armA

    # ---- sealed cross-check (A2's own epoch-12 per-session values) -----------
    sealed_crosscheck = {}
    for arm in ("t4", "z4"):
        for seed in args.seeds:
            for domain, surface in (
                ("within_subject", "within"), ("external_subject_M", "external")
            ):
                p = (
                    Path(a2.RESULT_ROOT)
                    / f"{domain}_source_{arm}_s{seed}.json"
                )
                if not p.is_file():
                    continue
                sealed = json.loads(p.read_text())
                mine = {row["session"]: row["r2"] for row in results[f"A2_{arm}_s{seed}"][surface]["per_session"]}
                sealed12 = (
                    sealed["per_epoch"]["12"].get("per_session_r2")
                    if "12" in sealed.get("per_epoch", {}) else None
                )
                if sealed12:
                    diffs = [abs(mine[s] - sealed12[s]) for s in sealed12 if s in mine]
                    sealed_crosscheck[f"A2_{arm}_s{seed}_{surface}"] = {
                        "sealed_receipt": str(p),
                        "sealed_receipt_sha256": arm_common.sha256_file(p),
                        "n_sessions_compared": len(diffs),
                        "max_abs_per_session_diff": max(diffs) if diffs else None,
                    }

    # ---- paired contrasts vs arm A (per seed + pooled) -----------------------
    def paired(numerator_rows, denominator_rows, sessions):
        deltas = [numerator_rows[s] - denominator_rows[s] for s in sessions]
        stats = matched_scorer.paired_session_stats(deltas, seed=42, n_boot=10000)
        return stats

    def table(model, surface):
        return {row["session"]: row["r2"] for row in results[model][surface]["per_session"]}

    within_sessions = sorted(within_starts)
    ext_sessions = sorted(ext_starts)
    contrasts = {}
    for seed in args.seeds:
        for surface, sessions in (("within", within_sessions), ("external", ext_sessions)):
            contrasts[f"A2_t4_s{seed}_minus_armA_{surface}"] = paired(
                table(f"A2_t4_s{seed}", surface),
                table("armA_direct_t4_48_swa_lastbin", surface),
                sessions,
            )
    # 3-seed pooled: mean of per-session scores across seeds, then paired
    pooled = {}
    for arm in ("t4", "z4"):
        for surface, sessions in (("within", within_sessions), ("external", ext_sessions)):
            per_seed = [table(f"A2_{arm}_s{seed}", surface) for seed in args.seeds]
            pooled[f"A2_{arm}_pooled_{surface}"] = {
                s: float(np.mean([t[s] for t in per_seed])) for s in sessions
            }
    for surface, sessions in (("within", within_sessions), ("external", ext_sessions)):
        contrasts[f"A2_t4_pooled_minus_armA_{surface}"] = paired(
            pooled[f"A2_t4_pooled_{surface}"],
            table("armA_direct_t4_48_swa_lastbin", surface),
            sessions,
        )

    closure_final = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)
    receipt = {
        "schema": "tfpd_a2_matched_rescore_v1",
        "status": "A2_MATCHED_RESCORE_COMPLETE__READ_ONLY",
        "supersedes": {
            "void_receipt": "results/a2_matched_rescore_v1/a2_rescore_receipt.json",
            "reason": (
                "the superseded run omitted the A2 deployment output scaling "
                "(division by behavior_scaling_factor 5.0), producing meaningless "
                "negative scores for every A2 cell; arm A numbers were unaffected"
            ),
        },
        "nature": (
            "sealed-artifact read-only rescoring under the TFPD matched scorer; "
            "NO A2 file was modified, no retraining, no adaptation"
        ),
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plan": plan,
        "scored_artifacts": integrity,
        "results": results,
        "pooled_per_session": pooled,
        "contrasts_a2_minus_armA": contrasts,
        "sealed_crosscheck": sealed_crosscheck,
        "semantics": {
            "metric": "src/tfpd_lane/matched_scorer.session_r2 (variance_weighted), equal weight per session",
            "query": "all val/external windows; prediction and target taken at the LAST timestep of each 50-bin window",
            "a2_scaling": "student output / 5.0 (behavior_scaling_factor), per the A2 deployment convention",
            "external_normalizers": "strict-27 source-only behavior + side stats injected (semantic SHA f062506c… verified)",
            "armA_reference_full_window": {
                "within": 0.516273, "external": 0.16105,
                "note": "arm A deployable full-window numbers (Gate-3/4 sealed); last-bin here is the matched-granularity sensitivity",
            },
        },
        "disclosures": {
            "a2_files_modified": False,
            "teacher_checkpoint_opened_read_only": True,
            "target_updates_gradients_optimizer_steps_during_scoring": 0,
            "checkpoint_selection_performed": False,
            "formal_or_organizer_held_data_opened": False,
            "normalizer_refit_on_target": False,
        },
        "source_closure": {
            "launch": closure, "final": closure_final,
            "launch_final_closure_equal": closure["closure_sha256"] == closure_final["closure_sha256"],
        },
        "environment": {
            "device": str(device), "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "no_user_site": bool(sys.flags.no_user_site),
        },
    }
    receipt_mod.write_receipt_transactionally(out_dir / "a2_rescore_receipt.json", receipt)
    summary = {
        "a2_t4_means": {f"s{seed}": {
            "within": round(results[f"A2_t4_s{seed}"]["within"]["mean_r2"], 6),
            "external": round(results[f"A2_t4_s{seed}"]["external"]["mean_r2"], 6),
        } for seed in args.seeds},
        "a2_z4_means": {f"s{seed}": {
            "within": round(results[f"A2_z4_s{seed}"]["within"]["mean_r2"], 6),
            "external": round(results[f"A2_z4_s{seed}"]["external"]["mean_r2"], 6),
        } for seed in args.seeds},
        "armA_lastbin": {
            "within": round(armA_lastbin["within"]["mean_r2"], 6),
            "external": round(armA_lastbin["external"]["mean_r2"], 6),
        },
        "pooled_contrasts": {
            k: {"mean": round(v["mean"], 6), "n_pos": v["n_positive"]}
            for k, v in contrasts.items() if "pooled" in k
        },
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
