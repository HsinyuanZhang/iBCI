#!/usr/bin/env python3
"""Gate-3 within-development screen for the three admission arms (§11 Gate 3).

Scores the three immutable arm SWA artifacts (A/B/C, Gate 2) plus the current
12-epoch spintshape SWA baseline on the six within-dev sessions with the one
matched scorer (per-session variance-weighted R2, equal weight per session),
then applies the frozen curriculum gate and classification table:

- B - A >= +0.03 mean AND median > 0 AND >= 4/6 sessions positive, AND
- B - C >= +0.03 mean AND median > 0 AND >= 4/6 sessions positive
  -> curriculum passes the seed-42 within screen.

Every contrast carries the full §10 paired-session statistics (mean, median,
n positive, min/max, all deltas, fixed-seed bootstrap 95% interval, exact sign
pattern) plus a two-sided exact sign-test p-value as a descriptive extra.
The A2 teacher-initialized reference (T4 R2 0.5750) is CITED ONLY: mean-level
difference against the best new arm, not a session-paired claim.

All four models are scored by ONE code path (the pilot's proven score_track:
canonical normalized-T4 visible side, eval mode, no gradients, no state
mutation).  The baseline is re-scored here for internal pairing consistency
and cross-checked against its sealed receipt value 0.47272146762392814.

Receipt (0444): results/gate3_within_screen_v1/within_screen_receipt.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))

import torch

BOUND_PATTERNS = (
    "scripts/run_gate3_within_screen.py",
    "scripts/run_z4_boundary_pilot.py",
    "src/tfpd_lane/arm_common.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "src/tfpd/spintshape_module.py",
)

MODELS = {
    "armA_direct_t4_48_swa": ROOT / "results/admission_arms_v1/armA_direct_t4_48/swa_final4.pt",
    "armB_z4_pretrain_then_t4_swa": ROOT / "results/admission_arms_v1/armB_z4_pretrain_then_t4/swa_final4.pt",
    "armC_direct_t4_exposure_matched_swa": ROOT / "results/admission_arms_v1/armC_direct_t4_exposure_matched/swa_final4.pt",
    "baseline_spintshape12_swa": ROOT / "results/stage1_source_cells_v1/spintshape_t4/swa_final4.pt",
}
BASELINE_SEALED_MEAN = 0.47272146762392814  # within_val_swa_final4_eval.json (0.4727 receipt)
BASELINE_SEALED_PER_SESSION = {
    "sub-C_ses-CO-20151103": 0.3428,
    "sub-C_ses-CO-20151104": 0.3887,
    "sub-C_ses-CO-20151106": 0.3007,
    "sub-C_ses-CO-20151109": 0.5712,
    "sub-C_ses-CO-20151110": 0.6275,
    "sub-C_ses-CO-20151112": 0.6055,
}
A2_T4_CITED = 0.5750
GATE_THRESHOLD = 0.03
CONTRASTS = (
    ("B_minus_A", "armB_z4_pretrain_then_t4_swa", "armA_direct_t4_48_swa",
     "curriculum under equal 48-epoch budget (gate input)"),
    ("B_minus_C", "armB_z4_pretrain_then_t4_swa", "armC_direct_t4_exposure_matched_swa",
     "Z4-pretraining value at matched T4 exposure (gate input)"),
    ("A_minus_baseline12", "armA_direct_t4_48_swa", "baseline_spintshape12_swa",
     "long-schedule effect vs current 12-epoch SWA"),
    ("B_minus_baseline12", "armB_z4_pretrain_then_t4_swa", "baseline_spintshape12_swa",
     "descriptive"),
    ("C_minus_baseline12", "armC_direct_t4_exposure_matched_swa", "baseline_spintshape12_swa",
     "descriptive"),
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sign_test_pvalue(deltas) -> float:
    """Two-sided exact sign-test p-value (zeros excluded)."""
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = max(pos, neg)
    p_one = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * p_one)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "results/gate3_within_screen_v1"
    )
    args = parser.parse_args()

    if not sys.flags.no_user_site:
        print("PYTHONNOUSERSITE=1 is mandatory", file=sys.stderr)
        return 3
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA unavailable ({args.device})", file=sys.stderr)
        return 3

    out_dir = Path(args.output_root)
    if out_dir.exists():
        print(f"fresh output root required: {out_dir}", file=sys.stderr)
        return 2

    receipt_mod = _load_module("tfpd_lane_receipt", ROOT / "src/tfpd_lane/receipt.py")
    arm_common = _load_module("tfpd_lane_arm_common", ROOT / "src/tfpd_lane/arm_common.py")
    pilot = _load_module(
        "tfpd_z4_pilot_module", ROOT / "scripts/run_z4_boundary_pilot.py"
    )
    matched_scorer = _load_module(
        "tfpd_lane_matched_scorer", ROOT / "src/tfpd_lane/matched_scorer.py"
    )
    spintshape = _load_module(
        "tfpd_spintshape_module", ROOT / "src/tfpd/spintshape_module.py"
    )

    import lightning.pytorch as pl

    import mc_maze.a2_matched_subject_shift_v2_core as a2
    from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_dir.mkdir(parents=True)

    closure = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)

    # ---- within-dev surface (the only development data this screen touches) --
    dm = Dandi688MultiSessionDataModule(
        data_dir=str(a2.SUBC_DATA_ROOT),
        task="CO",
        split_counts=(27, 6, 6),
        batch_size=32,
        window_size=50,
        calibration_n_trials=30,
        max_trial_length=100,
        bin_size_ms=20,
        num_workers=args.num_workers,
        random_calibration=False,
        seed=args.seed,
        max_units_exclusive=100,
        cache_dir=str(a2.SOURCE_CACHE_ROOT),
        signal_view="sua",
        side_feature_group="t4",
        side_feature_pool_size=30,
        train_val_manifest_path=str(a2.MANIFEST_PATH),
    )
    dm.setup("validate")
    val_dataset = dm.val_dataset
    within_roster = tuple(dm.session_splits["val"])
    if len(within_roster) != 6:
        raise SystemExit("within-dev roster drift (expected 6 sessions)")
    positions = list(range(len(val_dataset.window_indices)))

    pl.seed_everything(args.seed, workers=True)
    scores = {}
    integrity = {}
    for name, path in MODELS.items():
        path = Path(path)
        sidecar = Path(str(path) + ".sha256")
        if not path.is_file():
            raise SystemExit(f"SWA artifact missing: {path}")
        artifact_sha = arm_common.sha256_file(path)
        sidecar_verified = False
        if sidecar.is_file():
            expected = sidecar.read_text().split()[0]
            if artifact_sha != expected:
                raise SystemExit(f"SWA SHA mismatch against sidecar: {path}")
            sidecar_verified = True
        elif name != "baseline_spintshape12_swa":
            raise SystemExit(f"SWA sidecar missing: {path}")
        # the legacy 12-epoch baseline predates the sidecar convention; bind it
        # by its sealed evaluation receipt instead (read-only, never rescored
        # before this screen)
        if name == "baseline_spintshape12_swa":
            sealed_eval = ROOT / "results/stage1_source_cells_v1/spintshape_t4/within_val_swa_final4_eval.json"
            if not sealed_eval.is_file():
                raise SystemExit(f"baseline sealed eval receipt missing: {sealed_eval}")
            sealed_sha = arm_common.sha256_file(sealed_eval)
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        # legacy Lightning-built SWA keys carry the 'model.' prefix; strip it
        state = {
            (k[len("model."):] if k.startswith("model.") else k): v
            for k, v in state.items()
        }
        model = spintshape.build_spintshape_model(seed=args.seed)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        with torch.no_grad():
            probe = pilot.probe_scoring_forward_mode(
                model, val_dataset, positions, device, args.eval_batch_size, 50
            )
        forward_mode = "identity_cached" if probe["identity_cached_path_accepted"] else "full"
        result = pilot.score_track(
            model, val_dataset, positions, device, args.eval_batch_size, 50,
            cap_per_session=None, forward_mode=forward_mode,
        )
        scores[name] = result
        integrity[name] = {
            "path": str(path),
            "sha256": artifact_sha,
            "sidecar_verified": sidecar_verified,
            "strict_reload": True,
            "forward_mode": forward_mode,
            "identity_probe_max_abs_diff": probe["max_abs_diff"],
            "mean_r2": result["mean_r2"],
            "n_windows_scored": result["n_windows_scored"],
        }
        if name == "baseline_spintshape12_swa":
            integrity[name]["sealed_eval_receipt_sha256"] = sealed_sha
            integrity[name]["binding_note"] = (
                "legacy artifact predates the sidecar convention; bound by sha256 computed "
                "at screen time plus its sealed within_val_swa_final4_eval.json receipt"
            )
        print(json.dumps({"model": name, "mean_r2": round(result["mean_r2"], 6)}), flush=True)
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    per_session = {
        name: {row["session"]: row["r2"] for row in result["per_session"]}
        for name, result in scores.items()
    }
    sessions = sorted(within_roster)

    baseline_cross_check = {
        "sealed_receipt_mean": BASELINE_SEALED_MEAN,
        "rescored_mean": scores["baseline_spintshape12_swa"]["mean_r2"],
        "abs_diff": abs(scores["baseline_spintshape12_swa"]["mean_r2"] - BASELINE_SEALED_MEAN),
        "max_per_session_abs_diff_vs_sealed_receipt": max(
            abs(per_session["baseline_spintshape12_swa"][s] - BASELINE_SEALED_PER_SESSION[s])
            for s in sessions
        ),
        "note": "sealed receipt rounded per-session values to 4 decimals",
    }

    contrasts = {}
    for label, numerator, denominator, role in CONTRASTS:
        deltas = [
            per_session[numerator][s] - per_session[denominator][s] for s in sessions
        ]
        stats = matched_scorer.paired_session_stats(deltas, seed=42, n_boot=10000)
        stats.update(
            {
                "contrast": f"{numerator} - {denominator}",
                "role": role,
                "sign_test_p_two_sided_exact": sign_test_pvalue(deltas),
                "gate_threshold_mean": GATE_THRESHOLD,
                "mean_ge_threshold": stats["mean"] >= GATE_THRESHOLD,
                "median_positive": stats["median"] > 0,
                "n_positive_ge_4_of_6": stats["n_positive"] >= 4,
            }
        )
        contrasts[label] = stats

    gate_ba = contrasts["B_minus_A"]
    gate_bc = contrasts["B_minus_C"]
    curriculum_pass = (
        gate_ba["mean_ge_threshold"] and gate_ba["median_positive"] and gate_ba["n_positive_ge_4_of_6"]
        and gate_bc["mean_ge_threshold"] and gate_bc["median_positive"] and gate_bc["n_positive_ge_4_of_6"]
    )
    means = {name: result["mean_r2"] for name, result in scores.items()}
    best_new = max(("armA_direct_t4_48_swa", "armB_z4_pretrain_then_t4_swa",
                    "armC_direct_t4_exposure_matched_swa"), key=lambda n: means[n])

    a_baseline_gain = contrasts["A_minus_baseline12"]["mean"]
    if curriculum_pass:
        classification = "curriculum passes seed-42 within screen"
    elif means["armB_z4_pretrain_then_t4_swa"] > means["armA_direct_t4_48_swa"] and (
        gate_ba["mean_ge_threshold"] or gate_bc["mean_ge_threshold"]
    ):
        classification = (
            "B beats one comparator only: useful equal-budget recipe or matched-exposure "
            "value, but the full curriculum gate is not met"
        )
    elif a_baseline_gain >= 0.01:
        classification = "A improves while B does not clear the gate: adopt long direct-T4 for performance"
    elif max(contrasts["B_minus_A"]["mean"], contrasts["B_minus_C"]["mean"],
             a_baseline_gain) < 0.01:
        classification = "no new arm improves by +0.01: close training-budget route"
    else:
        classification = "partial gains below the +0.03 gate: retain best arm, no main training-method claim"

    receipt = {
        "schema": "tfpd_gate3_within_screen_v1",
        "status": "GATE3_WITHIN_SCREEN_COMPLETE",
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate": "HANDOFF_CARRIER_ADMISSION_CURRICULUM_20260816.md section 11 Gate 3 (within-development screen)",
        "scored_artifacts": integrity,
        "within_surface": {
            "sessions": list(within_roster),
            "n_windows": len(positions),
            "side_features": "canonical normalized T4 (deployment inference graph)",
            "scorer": "src/tfpd_lane/matched_scorer.session_r2 (torchmetrics R2Score, variance_weighted), equal weight per session",
        },
        "baseline_cross_check": baseline_cross_check,
        "model_means": means,
        "best_new_arm": best_new,
        "contrasts": contrasts,
        "curriculum_gate": {
            "rule": "B-A and B-C each: mean >= +0.03 AND median > 0 AND >= 4/6 sessions positive",
            "B_minus_A_pass": bool(
                gate_ba["mean_ge_threshold"] and gate_ba["median_positive"] and gate_ba["n_positive_ge_4_of_6"]
            ),
            "B_minus_C_pass": bool(
                gate_bc["mean_ge_threshold"] and gate_bc["median_positive"] and gate_bc["n_positive_ge_4_of_6"]
            ),
            "curriculum_gate_pass": bool(curriculum_pass),
            "classification": classification,
        },
        "a2_reference_cited_only": {
            "a2_t4_r2": A2_T4_CITED,
            "best_new_arm": best_new,
            "best_new_mean_r2": means[best_new],
            "best_new_minus_a2": means[best_new] - A2_T4_CITED,
            "note": (
                "A2 is a separately matched sealed reference under its own checkpoint "
                "authority and its own fixed scorer lineage; cited at mean level only, "
                "not session-paired, per the handoff's metric-parity caveat"
            ),
        },
        "disclosures": {
            "target_updates_gradients_optimizer_steps_during_scoring": 0,
            "checkpoint_selection_performed": False,
            "swa_artifacts_read_only": True,
            "formal_or_organizer_held_data_opened": False,
            "external_sub_m_opened": False,
            "within_dev_role": "Gate-3 screen (authorized); never touched during fitting or T_pre selection",
            "teacher_checkpoint_logits_or_loss_used": False,
        },
        "source_closure": closure,
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "no_user_site": bool(sys.flags.no_user_site),
        },
    }
    receipt_mod.write_receipt_transactionally(out_dir / "within_screen_receipt.json", receipt)
    print(
        json.dumps(
            {
                "means": {k: round(v, 6) for k, v in means.items()},
                "B_minus_A": {"mean": round(gate_ba["mean"], 5), "n_pos": gate_ba["n_positive"]},
                "B_minus_C": {"mean": round(gate_bc["mean"], 5), "n_pos": gate_bc["n_positive"]},
                "A_minus_baseline12": {"mean": round(contrasts["A_minus_baseline12"]["mean"], 5)},
                "curriculum_gate_pass": bool(curriculum_pass),
                "classification": classification,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
