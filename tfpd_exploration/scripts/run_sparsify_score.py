#!/usr/bin/env python3
"""One-shot dual-granularity scoring + frozen decision rules for R / S2.

WORKORDER_BEAT_A2_SPARSIFICATION_20260818.md §7/§9: scores the terminal R and
S2 SWAs once, read-only, on the frozen within-6 / external-15 development
sets under BOTH granularities (GOVERNING last-bin equal-session; DIAGNOSTIC
full-window equal-session and window-weighted), with per-session n_windows,
2014/2015 date blocks and `sub-M_ses-CO-20141203` reported separately, paired
session deltas + sign counts + descriptive bootstrap intervals versus the
SEALED arm A and D references, and the frozen verdicts:

- R_recovery = (R_ext - ArmA_ext) / (D_ext - ArmA_ext) at governing
  granularity, with the ≥0.75 / ≤0.50 / inconclusive bands;
- S2-over-D three-condition gate (external mean ≥ +0.03, ≥ 10/15 positive,
  within mean ≥ -0.03);
- A2 development screen: BOTH absolute governing means must meet the sealed
  pooled bars, LOADED from the SHA-verified _r1 receipt (no hardcoded values).

No confidence interval or secondary aggregation may rescue a failed governing
mean; full-window results are labelled diagnostics throughout.
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

PAD_VALUE = -1.0
AUTH_VALUE = "I_AUTHORIZE_TFPD_DEVELOPMENT_TARGET_ACCESS"
BOUND_PATTERNS = (
    "scripts/run_sparsify_score.py",
    "scripts/run_sparsify_cell.py",
    "scripts/run_a2_matched_rescore.py",
    "scripts/run_z4_boundary_pilot.py",
    "src/tfpd_lane/sparsification.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "src/tfpd_lane/arm_common.py",
)
A2_R1_RECEIPT = ROOT / "results/a2_matched_rescore_v1_r1/a2_rescore_receipt.json"
A2_R1_RECEIPT_SHA = "0ae0f74c6b5d606599d1108378836c8cbb7d1e5df05a453aa83f682255734d4e"
VOID_ROOT = ROOT / "results/a2_matched_rescore_v1"  # rejected, never opened
CANONICAL_INITIAL_STATE = ROOT / "results/admission_arms_v1/canonical_initial_state.pt"
THETA_AUTHORITY_FILE = ROOT / "results/sparsification_theta_authority_v1/theta_authority.pt"


def load_governing_references(sha256_file):
    """A2 bars and ArmA governing means FROM the sealed _r1 receipt only.

    Verifies the receipt body SHA against the frozen value; the void root is
    present-but-rejected and never opened.  No hardcoded bars remain.
    """
    if not A2_R1_RECEIPT.is_file():
        raise SystemExit("A2 _r1 receipt missing")
    body_sha = sha256_file(A2_R1_RECEIPT)
    if body_sha != A2_R1_RECEIPT_SHA:
        raise SystemExit(f"A2 _r1 receipt SHA mismatch: {body_sha}")
    payload = json.loads(A2_R1_RECEIPT.read_text())
    ext = payload["pooled_per_session"]["A2_t4_pooled_external"]
    win = payload["pooled_per_session"]["A2_t4_pooled_within"]
    arm_a = payload["results"]["armA_direct_t4_48_swa_lastbin"]
    return {
        "a2_bars": {"external": float(sum(ext.values()) / len(ext)),
                    "within": float(sum(win.values()) / len(win))},
        "armA_external": float(arm_a["external"]["mean_r2"]),
        "armA_within": float(arm_a["within"]["mean_r2"]),
        "receipt_sha256": body_sha,
        "void_root_opened": False,
    }
MODELS = {
    "R_swa": ROOT / "results/sparsification_v1/cellR_elementwise/swa_final4.pt",
    "S2_swa": ROOT / "results/sparsification_v1/cellS2_carrier_sector/swa_final4.pt",
    "D_swa": ROOT / "results/pop_robust_v1/cellD_2heads_dynamic_dropout/swa_final4.pt",
    "armA_swa": ROOT / "results/admission_arms_v1/armA_direct_t4_48/swa_final4.pt",
}
SEPARATE_SESSION = "sub-M_ses-CO-20141203"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_cell_terminal(cell_name, arm_common, canonical_payload, theta_payload):
    """Full §9 validation of one cell's terminal receipt + sealed artifacts."""
    import stat as _stat

    receipt_path = ROOT / f"results/sparsification_v1/{cell_name}/terminal_receipt.json"
    sidecar = Path(str(receipt_path) + ".sha256")
    problems = []
    if not receipt_path.is_file() or not sidecar.is_file():
        raise SystemExit(f"terminal receipt or sidecar missing: {receipt_path}")
    if receipt_path.is_symlink() or sidecar.is_symlink():
        problems.append("symlinked receipt/sidecar")
    body_sha = arm_common.sha256_file(receipt_path)
    if body_sha != sidecar.read_text().split()[0]:
        problems.append("sidecar SHA mismatch")
    receipt_mode = _stat.S_IMODE(receipt_path.stat().st_mode)
    if receipt_mode != 0o444:
        problems.append(f"receipt mode {oct(receipt_mode)} != 0444")
    payload = json.loads(receipt_path.read_text())
    if payload["status"] != "CELL_TERMINAL":
        problems.append(f"status {payload['status']}")
    if not payload["source_closure"].get("launch_final_closure_equal"):
        problems.append("launch/final closure inequality")
    initial = payload.get("initial_state", {})
    if initial.get("state_dict_sha256") != canonical_payload["state_sha256"]:
        problems.append("terminal-bound initial-state state SHA != canonical")
    if initial.get("artifact_sha256") != arm_common.sha256_file(CANONICAL_INITIAL_STATE):
        problems.append("terminal-bound initial-state artifact SHA drift")
    theta_bound = payload.get("integrity", {}).get("theta_authority", {})
    if theta_bound.get("sha256") != arm_common.sha256_file(THETA_AUTHORITY_FILE):
        problems.append("terminal-bound theta-authority SHA drift")
    if theta_bound.get("authority_sha256") != theta_payload["authority_sha256"]:
        problems.append("terminal-bound theta authority_sha256 drift")
    normalizer = payload.get("data_contract", {}).get(
        "behavior_normalizer_semantic_sha256", ""
    )
    if not str(normalizer).startswith("f062506c"):
        problems.append("terminal-bound normalizer semantic SHA drift")
    if problems:
        raise SystemExit(f"cell {cell_name} terminal validation failed: {problems}")
    return {
        "terminal_receipt": str(receipt_path),
        "terminal_receipt_sha256": body_sha,
        "mode_0444": True,
        "non_symlink": True,
        "launch_final_closure_equal": True,
        "initial_state_reconciled": True,
        "theta_authority_reconciled": True,
        "normalizer_reconciled_f062506c": True,
        "swa_sha256": payload["swa"]["sha256"],
        "p_sequence_sha256": payload["p_sequence_sha256"],
        "epochs_run": payload["epochs_run"],
    }


def window_weighted_mean(per_session_rows):
    """Full-window DIAGNOSTIC aggregation weighted by per-session n_windows."""
    total_windows = sum(row.get("n_windows_scored", 0) for row in per_session_rows)
    if total_windows == 0:
        return None
    return float(
        sum(row["r2"] * row.get("n_windows_scored", 0) for row in per_session_rows)
        / total_windows
    )


def date_block(session: str) -> str:
    return "2014" if "-2014" in session else ("2015" if "-2015" in session else "other")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/sparsification_score_v1")
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
    if args.dry_run:
        print(json.dumps({"schema": "tfpd_sparsification_score_v1",
                          "status": "DRY_RUN__NO_NWB_OPENED"}, indent=1))
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

    receipt_mod = _load_module("tfpd_lane_receipt", ROOT / "src/tfpd_lane/receipt.py")
    arm_common = _load_module("tfpd_lane_arm_common", ROOT / "src/tfpd_lane/arm_common.py")
    matched_scorer = _load_module(
        "tfpd_lane_matched_scorer", ROOT / "src/tfpd_lane/matched_scorer.py"
    )
    rescorer = _load_module("tfpd_a2_rescorer", ROOT / "scripts/run_a2_matched_rescore.py")
    pilot = _load_module("tfpd_z4_pilot", ROOT / "scripts/run_z4_boundary_pilot.py")
    sparsification = _load_module(
        "tfpd_lane_sparsification", ROOT / "src/tfpd_lane/sparsification.py"
    )
    closure = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)

    # ---- full terminal-receipt validation before scoring opens -------------
    references = load_governing_references(arm_common.sha256_file)
    A2_BARS = references["a2_bars"]
    ARM_A_EXTERNAL = references["armA_external"]
    ARM_A_WITHIN = references["armA_within"]
    canonical_payload = torch.load(
        ROOT / "results/admission_arms_v1/canonical_initial_state.pt",
        map_location="cpu", weights_only=False,
    )
    theta_payload = torch.load(
        ROOT / "results/sparsification_theta_authority_v1/theta_authority.pt",
        map_location="cpu", weights_only=False,
    )
    provenance = {}
    for cell, name in (("R", "cellR_elementwise"), ("S2", "cellS2_carrier_sector")):
        provenance[cell] = validate_cell_terminal(
            name, arm_common, canonical_payload, theta_payload
        )
    if provenance["R"]["p_sequence_sha256"] != provenance["S2"]["p_sequence_sha256"]:
        raise SystemExit("R/S2 p-stream SHA inequality")

    import mc_maze.a2_matched_subject_shift_v2_core as a2

    surfaces = rescorer.build_surfaces(args, a2, "t4")
    (within_ds, within_starts), (ext_ds, ext_starts) = surfaces

    results = {}
    integrity = {}
    for name, path in MODELS.items():
        path = Path(path)
        sidecar = Path(str(path) + ".sha256")
        if not path.is_file() or not sidecar.is_file():
            raise SystemExit(f"SWA or sidecar missing: {path}")
        sha = arm_common.sha256_file(path)
        if sha != sidecar.read_text().split()[0]:
            raise SystemExit(f"SWA SHA mismatch: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        state = {(k[len("model."):] if k.startswith("model.") else k): v
                 for k, v in state.items()}
        if name in ("R_swa", "S2_swa"):
            cell = "R" if name == "R_swa" else "S2"
            model = sparsification.build_sparsified_model(seed=42, cell=cell)
        else:
            spintshape = _load_module("tfpd_spintshape_module", ROOT / "src/tfpd/spintshape_module.py")
            model = spintshape.build_spintshape_model(seed=42)
        model.load_state_dict(state, strict=True)
        state_before = arm_common.state_sha256(model)
        model.to(device).eval()

        def forward(neural, calib, side, _m=model):
            return _m(neural, calib_trials=calib, side_features=side)

        governing = {
            surface: rescorer.score_last_bin(
                forward, ds, starts, device, matched_scorer.session_r2, output_scale=1.0
            )
            for surface, (ds, starts) in (
                ("within", (within_ds, within_starts)),
                ("external", (ext_ds, ext_starts)),
            )
        }
        diagnostic = {
            surface: pilot.score_track(
                model, ds, list(range(len(ds.window_indices))) if False else all_positions(ds, starts),
                device, 128, 50, cap_per_session=None, forward_mode="identity_cached",
            )
            for surface, (ds, starts) in (
                ("within", (within_ds, within_starts)),
                ("external", (ext_ds, ext_starts)),
            )
        }
        state_after = arm_common.state_sha256(model)
        results[name] = {"governing_last_bin": governing, "diagnostic_full_window": diagnostic}
        integrity[name] = {
            "path": str(path), "sha256": sha,
            "state_unchanged_during_scoring": state_before == state_after,
            "grads_all_none_after": all(
                p.grad is None for p in model.parameters()
                if not isinstance(p, torch.nn.parameter.UninitializedParameter)
            ),
        }
        if state_before != state_after:
            raise SystemExit(f"scoring mutated state: {name}")
        print(json.dumps({
            "model": name,
            "within_gov": round(governing["within"]["mean_r2"], 6),
            "external_gov": round(governing["external"]["mean_r2"], 6),
            "within_full": round(diagnostic["within"]["mean_r2"], 6),
            "external_full": round(diagnostic["external"]["mean_r2"], 6),
        }), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def per_session(model, surface, granularity):
        return {
            row["session"]: row["r2"]
            for row in results[model][granularity][surface]["per_session"]
        }

    def paired(model_a, model_b, surface, granularity="governing_last_bin"):
        table_a = per_session(model_a, surface, granularity)
        table_b = per_session(model_b, surface, granularity)
        sessions = sorted(table_a)
        deltas = [table_a[s] - table_b[s] for s in sessions]
        stats = matched_scorer.paired_session_stats(deltas, seed=42, n_boot=10000)
        stats["contrast"] = f"{model_a} - {model_b} ({surface}, {granularity})"
        return stats

    # (5) the FULL §7 contrast matrix: every pair the decision rules consume,
    # in BOTH granularities (governing drives verdicts; diagnostic reported)
    contrast_pairs = (
        ("R_swa", "armA_swa"), ("R_swa", "D_swa"),
        ("S2_swa", "D_swa"), ("S2_swa", "armA_swa"),
    )
    contrasts = {}
    for granularity in ("governing_last_bin", "diagnostic_full_window"):
        for model_a, model_b in contrast_pairs:
            for surface in ("within", "external"):
                contrasts[f"{model_a}_minus_{model_b}_{surface}_{granularity}"] = paired(
                    model_a, model_b, surface, granularity
                )

    # (1) window-weighted full-window diagnostic means, clearly non-governing
    window_weighted = {
        model: {
            surface: window_weighted_mean(
                results[model]["diagnostic_full_window"][surface]["per_session"]
            )
            for surface in ("within", "external")
        }
        for model in MODELS
    }

    r_ext = results["R_swa"]["governing_last_bin"]["external"]["mean_r2"]
    d_ext = results["D_swa"]["governing_last_bin"]["external"]["mean_r2"]
    d_gain = d_ext - ARM_A_EXTERNAL
    r_recovery = (r_ext - ARM_A_EXTERNAL) / d_gain if d_gain != 0 else None
    if r_recovery is None:
        r_verdict = "undefined (D gain is zero)"
    elif r_recovery >= 0.75:
        r_verdict = "generic elementwise regularization explains most of D; no population-sparsification mechanism claim"
    elif r_recovery <= 0.50:
        r_verdict = "whole-unit structure is materially load-bearing"
    else:
        r_verdict = "mechanism result inconclusive (0.50 < R_recovery < 0.75)"

    s2_ext = contrasts["S2_swa_minus_D_swa_external_governing_last_bin"]
    s2_within = contrasts["S2_swa_minus_D_swa_within_governing_last_bin"]
    s2_gate = {
        "external_mean_ge_plus_0.03": s2_ext["mean"] >= 0.03,
        "external_positive_ge_10_of_15": s2_ext["n_positive"] >= 10,
        "within_mean_ge_minus_0.03": s2_within["mean"] >= -0.03,
    }
    s2_pass = all(s2_gate.values())
    if s2_pass:
        s2_verdict = "S2 clears the S2-over-D gate; conditional S_perm is triggered"
    elif abs(s2_ext["mean"]) < 0.01 and s2_gate["within_mean_ge_minus_0.03"]:
        s2_verdict = "S2 and D practically equivalent: cardinality invariance is sufficient"
    elif s2_ext["mean"] <= -0.01:
        s2_verdict = "sector-gap training recipe rejected"
    else:
        s2_verdict = "positive delta below +0.03: promising, mechanism not established"

    a2_screen = {}
    for cell, model in (("R", "R_swa"), ("S2", "S2_swa")):
        ext_mean = results[model]["governing_last_bin"]["external"]["mean_r2"]
        within_mean = results[model]["governing_last_bin"]["within"]["mean_r2"]
        a2_screen[cell] = {
            "external_mean": ext_mean, "external_meets_bar": ext_mean >= A2_BARS["external"],
            "within_mean": within_mean, "within_meets_bar": within_mean >= A2_BARS["within"],
            "development_screen_pass": bool(
                ext_mean >= A2_BARS["external"] and within_mean >= A2_BARS["within"]
            ),
            "note": "seed-42 development screen only; no superiority claim",
        }

    # (2) date blocks + the separately-reported session, external GOVERNING:
    # per-block mean, paired block delta vs arm A, sign/count, window weighting
    blocks = {}
    block_stats = {}
    armA_table = per_session("armA_swa", "external", "governing_last_bin")
    for model in MODELS:
        table = per_session(model, "external", "governing_last_bin")
        for session, value in table.items():
            if session == SEPARATE_SESSION:
                blocks.setdefault(model, {}).setdefault("separate_20141203", {})[session] = value
            else:
                blocks.setdefault(model, {}).setdefault(date_block(session), {})[session] = value
        n_windows = {
            row["session"]: row.get("n_windows_scored", 0)
            for row in results[model]["governing_last_bin"]["external"]["per_session"]
        }
        for block, entries in blocks[model].items():
            values = list(entries.values())
            sessions = sorted(entries)
            deltas = [entries[s2] - armA_table[s2] for s2 in sessions]
            block_weight = sum(n_windows.get(s2, 0) for s2 in sessions) / max(
                sum(n_windows.values()), 1
            )
            block_stats.setdefault(model, {})[block] = {
                "n_sessions": len(sessions),
                "block_mean": float(np.mean(values)),
                "paired_delta_vs_armA_mean": float(np.mean(deltas)),
                "paired_delta_vs_armA_n_positive": int(sum(d > 0 for d in deltas)),
                "paired_delta_vs_armA_n_total": len(deltas),
                "window_share_of_external": float(block_weight),
                "sessions": sessions,
            }

    closure_final = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)
    receipt = {
        "schema": "tfpd_sparsification_score_v1",
        "status": "SPARSIFICATION_SCORED",
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cell_provenance": provenance,
        "governing_references": references,
        "window_weighted_full_window_diagnostic": {
            "note": "DIAGNOSTIC ONLY, never governing; equal-session last-bin remains the governing pair",
            "means": window_weighted,
        },
        "external_date_block_statistics": {
            "note": "governing last-bin external; separate_20141203 excluded from year blocks",
            "blocks": block_stats,
        },
        "scored_artifacts": integrity,
        "results": results,
        "n_windows": {
            surface: {
                row["session"]: row.get("n_windows")
                for row in results["R_swa"]["governing_last_bin"][surface]["per_session"]
            }
            for surface in ("within", "external")
        },
        "contrasts": contrasts,
        "r_interpretation": {
            "D_gain_governing": d_gain,
            "R_recovery": r_recovery,
            "bands": ">=0.75 generic regularization; <=0.50 whole-unit load-bearing; between inconclusive",
            "verdict": r_verdict,
        },
        "s2_over_D_gate": {
            "conditions": s2_gate,
            "pass": s2_pass,
            "verdict": s2_verdict,
        },
        "a2_development_screen": {
            "bars": A2_BARS,
            "cells": a2_screen,
            "note": "both absolute governing means must meet the bars; seeds 43/44 mandatory before any claim",
        },
        "external_date_blocks_and_separate_session": blocks,
        "granularity_labels": {
            "governing_last_bin": "GOVERNING: last timestep, variance-weighted, equal session weight",
            "diagnostic_full_window": "DIAGNOSTIC ONLY: cannot rescue a failed governing mean",
        },
        "disclosures": {
            "target_updates_gradients_optimizer_steps": 0,
            "checkpoint_selection_performed": False,
            "formal_or_organizer_held_data_opened": False,
            "normalizer_refit_on_target": False,
            "sealed_files_modified": False,
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
    receipt_mod.write_receipt_transactionally(out_dir / "sparsification_score_receipt.json", receipt)
    print(json.dumps({
        "R_gov": {"within": round(results["R_swa"]["governing_last_bin"]["within"]["mean_r2"], 6),
                  "external": round(r_ext, 6)},
        "S2_gov": {"within": round(results["S2_swa"]["governing_last_bin"]["within"]["mean_r2"], 6),
                   "external": round(results["S2_swa"]["governing_last_bin"]["external"]["mean_r2"], 6)},
        "D_gov_external": round(d_ext, 6),
        "R_recovery": round(r_recovery, 4) if r_recovery is not None else None,
        "S2_minus_D_external_mean": round(s2_ext["mean"], 6),
        "S2_gate_pass": s2_pass,
        "r_verdict": r_verdict,
        "s2_verdict": s2_verdict,
        "a2_screen": {k: v["development_screen_pass"] for k, v in a2_screen.items()},
    }, indent=1))
    return 0


def all_positions(dataset, starts_by_session):
    positions = []
    index = {s: 0 for s in starts_by_session}
    for position, (session, _start) in enumerate(dataset.window_indices):
        if session in starts_by_session:
            positions.append(position)
    return positions


if __name__ == "__main__":
    raise SystemExit(main())
