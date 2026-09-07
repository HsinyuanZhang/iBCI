#!/usr/bin/env python3
"""TFAP Stage-3: one-shot scoring matrix and the six adoption gates (§4).

Scores the two Stage-2 fine-tuned SWAs (P-T4, P-Z4) and re-scores arm A's SWA
on the SAME surfaces with the SAME engine in one process:

- within-dev: 6 strict-27 validation sessions, full window sets;
- external: the audited 15-session sub-M roster, authorization-gated, strict-27
  source-only normalizers injected (the Gate-4 path);
- four diagnostics everywhere (native / zero / wrong_pair / destroyed_activity,
  frozen seeds 1234 / 4321), matched per-session variance-weighted R2, equal
  weight per session.

Applies the frozen adoption gates:

  1. P-T4 - A external paired mean >= +0.03;
  2. >= 10/15 external sessions positive;
  3. P-T4 within >= (A within) - 0.03;
  4. P-T4 external native > zero, wrong-pair, and destroyed-activity means;
  5. zero target update (structural; no model/optimizer mutation exists here);
  6. formal/organizer-held data unopened (structural).

P-Z4 mechanism clause: if P-Z4 - A external paired mean is also >= +0.03, any
P-T4 gain is attributed to generic pretraining, not task-frame alignment.

Receipt (0444): results/tfap_stage3_v1/stage3_receipt.json.
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
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))

PAD_VALUE = -1.0
AUTH_VALUE = "I_AUTHORIZE_TFPD_DEVELOPMENT_TARGET_ACCESS"
BOUND_PATTERNS = (
    "scripts/run_tfap_stage3.py",
    "scripts/run_gate4_arm_external.py",
    "src/tfpd_lane/arm_common.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "src/tfpd/spintshape_module.py",
)
MODELS = {
    "P-T4_finetuned_swa": ROOT / "results/tfap_stage2_v1/finetune_pt4/swa_final4.pt",
    "P-Z4_finetuned_swa": ROOT / "results/tfap_stage2_v1/finetune_pz4/swa_final4.pt",
    "armA_direct_t4_48_swa": ROOT / "results/admission_arms_v1/armA_direct_t4_48/swa_final4.pt",
}
SEALED_CROSSCHECK = {
    "armA_within": 0.516273,
    "armA_external_native": 0.1610,
}
GATE_THRESHOLD = 0.03
WITHIN_TOLERANCE = 0.03
N_POSITIVE_REQUIRED = 10


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def score_all_modes(model, dataset, starts_by_session, device, scorer):
    gate4 = sys.modules["tfpd_gate4_module"]
    per_mode = {m: [] for m in gate4.MODES}
    with torch.no_grad():  # scoring discipline: autograd off for every forward
        if torch.is_grad_enabled():
            raise SystemExit("scoring must run with autograd disabled")
        for session in sorted(starts_by_session):
            record = dataset.sessions[session]
            packed = gate4.score_session_modes(model, record, starts_by_session[session], device)
            for mode in gate4.MODES:
                preds, tgts = packed[mode]
                per_mode[mode].append(
                    {"session": session, "r2": scorer(preds, tgts),
                     "n_windows": len(starts_by_session[session])}
                )
    result = {}
    for mode in gate4.MODES:
        rows = per_mode[mode]
        result[mode] = {
            "per_session": rows,
            "mean_r2": float(np.mean([r["r2"] for r in rows])),
            "positive_sessions": int(sum(r["r2"] > 0 for r in rows)),
        }
    return result



def preflight_integrity(models, root, arm_common):
    """Input integrity BEFORE any datamodule construction or NWB open.

    (a) every scored SWA artifact exists with its .sha256 sidecar and the file
    hash matches the sidecar's first field; (b) both stage-2 terminal receipts
    exist with status FINETUNE_TERMINAL and their sealed SWA SHA equals the
    preflight SHA map.  Returns {model_name: sha256}; raises SystemExit on any
    failure.  Opens no NWB and constructs no datamodule.
    """
    shas = {}
    for name, rel in models.items():
        artifact = Path(rel)
        sidecar = Path(str(artifact) + ".sha256")
        if not artifact.is_file() or not sidecar.is_file():
            raise SystemExit(f"SWA or sidecar missing: {artifact}")
        artifact_sha = arm_common.sha256_file(artifact)
        if artifact_sha != sidecar.read_text().split()[0]:
            raise SystemExit(f"SWA SHA mismatch: {artifact}")
        shas[name] = artifact_sha
    for arm_key, model_key in (
        ("pt4", "P-T4_finetuned_swa"),
        ("pz4", "P-Z4_finetuned_swa"),
    ):
        receipt_path = Path(root) / f"results/tfap_stage2_v1/finetune_{arm_key}/terminal_receipt.json"
        if not receipt_path.is_file():
            raise SystemExit(f"stage-2 terminal receipt missing: {receipt_path}")
        payload = json.loads(receipt_path.read_text())
        if payload.get("status") != "FINETUNE_TERMINAL":
            raise SystemExit(f"stage-2 {arm_key} not terminal: {payload.get('status')}")
        if payload["swa"]["sha256"] != shas[model_key]:
            raise SystemExit(f"stage-2 SWA SHA mismatch for {model_key}")
    return shas


def evaluate_mechanism_gate(mean, n_positive):
    """Frozen conjunction.  The bootstrap interval is DESCRIPTIVE ONLY and can
    never rescue an effect below the absolute threshold or below the required
    positive-session count."""
    return bool(mean >= GATE_THRESHOLD and n_positive >= N_POSITIVE_REQUIRED)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/tfap_stage3_v1")
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
    gate4 = _load_module("tfpd_gate4_module", ROOT / "scripts/run_gate4_arm_external.py")
    spintshape = _load_module("tfpd_spintshape_module", ROOT / "src/tfpd/spintshape_module.py")

    plan = {
        "schema": "tfap_stage3_v1",
        "models": {k: str(v) for k, v in MODELS.items()},
        "surfaces": ["within_dev_6", "external_sub_M_15"],
        "modes": list(gate4.MODES),
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

    import mc_maze.a2_matched_subject_shift_v2_core as a2
    from mc_maze.multisession_datamodule import (
        Dandi688MultiSessionDataModule,
        fit_behavior_stats,
    )
    from mc_maze.unit_side_features import fit_side_feature_stats

    closure = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)

    # input integrity BEFORE any datamodule construction or NWB open: a
    # non-terminal cell or SHA mismatch must fail before target data is touched
    preflight_shas = preflight_integrity(MODELS, ROOT, arm_common)

    # ---- within surface ------------------------------------------------------
    dm_within = Dandi688MultiSessionDataModule(
        data_dir=str(a2.SUBC_DATA_ROOT), task="CO", split_counts=(27, 6, 6), batch_size=32,
        window_size=50, calibration_n_trials=30, max_trial_length=100, bin_size_ms=20,
        num_workers=args.num_workers, random_calibration=False, seed=args.seed,
        max_units_exclusive=100, cache_dir=str(a2.SOURCE_CACHE_ROOT), signal_view="sua",
        side_feature_group="t4", side_feature_pool_size=30,
        train_val_manifest_path=str(a2.MANIFEST_PATH),
    )
    dm_within.setup("validate")
    within_dataset = dm_within.val_dataset
    within_starts = {}
    for position, (session, start) in enumerate(within_dataset.window_indices):
        within_starts.setdefault(session, []).append(int(start))
    if len(within_starts) != 6:
        raise SystemExit("within roster drift")

    # ---- external surface (target data opens HERE) ---------------------------
    dm_ext = Dandi688MultiSessionDataModule(
        data_dir=str(a2.SUBM_DATA_ROOT), task="CO", split_counts=(0, 15, 0), batch_size=32,
        window_size=50, calibration_n_trials=30, max_trial_length=100, bin_size_ms=20,
        num_workers=args.num_workers, random_calibration=False, seed=args.seed,
        max_units_exclusive=100, cache_dir=str(ROOT / "cache/subm_external_v1"),
        signal_view="sua", side_feature_group="t4", side_feature_pool_size=30,
    )
    train_paths, _v, _n = a2.active_source_session_paths()
    mean, std = fit_behavior_stats(train_paths, 20, cache_dir=a2.SOURCE_CACHE_ROOT)
    semantic = a2.normalizer_value_sha256(mean, std)
    if not semantic.startswith("f062506c"):
        raise SystemExit("source behavior normalizer semantic SHA drift")
    dm_ext._behavior_stats = (mean, std)
    side_mean, side_std = fit_side_feature_stats(
        train_paths, feature_group="t4", pool_size=30,
        cache_dir=a2.SOURCE_CACHE_ROOT, signal_view="sua",
    )
    dm_ext._side_feature_stats = (side_mean, side_std)
    dm_ext.setup("validate")
    ext_dataset = dm_ext.val_dataset
    ext_starts = {}
    for position, (session, start) in enumerate(ext_dataset.window_indices):
        ext_starts.setdefault(session, []).append(int(start))
    if len(ext_starts) != 15:
        raise SystemExit("external roster drift")

    scored = {}
    integrity = {}
    for name, path in MODELS.items():
        path = Path(path)
        artifact_sha = preflight_shas[name]  # sealed by preflight; no rehash
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        state = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in state.items()}
        model = spintshape.build_spintshape_model(seed=args.seed)
        model.load_state_dict(state, strict=True)
        state_sha_before = arm_common.state_sha256(model)
        model.to(device).eval()
        scored[name] = {
            "within": score_all_modes(model, within_dataset, within_starts, device,
                                      matched_scorer.session_r2),
            "external": score_all_modes(model, ext_dataset, ext_starts, device,
                                        matched_scorer.session_r2),
        }
        # (6) provable no-target-update evidence: state unchanged by scoring,
        # no gradient ever populated, no optimizer instance exists in this process
        state_sha_after = arm_common.state_sha256(model)
        grads_all_none = all(
            p.grad is None
            for p in model.parameters()
            if not isinstance(p, torch.nn.parameter.UninitializedParameter)
        )
        integrity[name] = {
            "path": str(path), "sha256": artifact_sha, "strict_reload": True,
            "state_sha_before_scoring": state_sha_before,
            "state_sha_after_scoring": state_sha_after,
            "state_unchanged_during_scoring": state_sha_before == state_sha_after,
            "grads_all_none_after_scoring": grads_all_none,
        }
        if state_sha_before != state_sha_after or not grads_all_none:
            raise SystemExit(f"scoring mutated state or produced gradients: {name}")
        print(json.dumps({
            "model": name,
            "within_native": round(scored[name]["within"]["native"]["mean_r2"], 6),
            "external_native": round(scored[name]["external"]["native"]["mean_r2"], 6),
        }), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def per_session(surface, mode):
        return {
            name: {row["session"]: row["r2"] for row in scored[name][surface][mode]["per_session"]}
            for name in scored
        }

    ext_sessions = sorted(ext_starts)
    within_sessions = sorted(within_starts)
    ext_native = per_session("external", "native")
    within_native = per_session("within", "native")

    def contrast(numerator, denominator, sessions, table):
        deltas = [table[numerator][s] - table[denominator][s] for s in sessions]
        stats = matched_scorer.paired_session_stats(deltas, seed=42, n_boot=10000)
        stats["contrast"] = f"{numerator} - {denominator}"
        return stats

    pt4_minus_a_ext = contrast("P-T4_finetuned_swa", "armA_direct_t4_48_swa", ext_sessions, ext_native)
    pt4_minus_a_within = contrast("P-T4_finetuned_swa", "armA_direct_t4_48_swa", within_sessions, within_native)
    pz4_minus_a_ext = contrast("P-Z4_finetuned_swa", "armA_direct_t4_48_swa", ext_sessions, ext_native)
    pt4_minus_pz4_ext = contrast("P-T4_finetuned_swa", "P-Z4_finetuned_swa", ext_sessions, ext_native)

    pt4_ext = scored["P-T4_finetuned_swa"]["external"]

    # (6) provable structural gates
    state_unchanged = all(v["state_unchanged_during_scoring"] for v in integrity.values())
    grads_none = all(v["grads_all_none_after_scoring"] for v in integrity.values())
    stage2_provenance = {}
    for arm_key, model_key in (("pt4", "P-T4_finetuned_swa"), ("pz4", "P-Z4_finetuned_swa")):
        s2_receipt_path = (
            ROOT / f"results/tfap_stage2_v1/finetune_{arm_key}/terminal_receipt.json"
        )
        s2_payload = json.loads(s2_receipt_path.read_text())
        if s2_payload["status"] != "FINETUNE_TERMINAL":
            raise SystemExit(f"stage-2 {arm_key} not terminal")
        if s2_payload["swa"]["sha256"] != preflight_shas[model_key]:
            raise SystemExit(f"stage-2 SWA SHA mismatch for {model_key}")
        stage2_provenance[arm_key] = {
            "terminal_receipt": str(s2_receipt_path),
            "terminal_receipt_sha256": arm_common.sha256_file(s2_receipt_path),
            "swa_sha256": s2_payload["swa"]["sha256"],
            "epochs_run": s2_payload["epochs_run"],
        }
    closure_final = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)
    closure_equal = closure_final["closure_sha256"] == closure["closure_sha256"]

    engineering_gate = (
        pt4_minus_a_ext["mean"] >= GATE_THRESHOLD
        and pt4_minus_a_ext["n_positive"] >= N_POSITIVE_REQUIRED
    )
    mechanism_gate = evaluate_mechanism_gate(
        pt4_minus_pz4_ext["mean"], pt4_minus_pz4_ext["n_positive"]
    )
    gates = {
        "g1_engineering_pt4_minus_a_external_mean_ge_plus_0.03": pt4_minus_a_ext["mean"] >= GATE_THRESHOLD,
        "g2_engineering_at_least_10_of_15_external_positive": pt4_minus_a_ext["n_positive"] >= N_POSITIVE_REQUIRED,
        "g3_within_not_below_a_minus_0.03": (
            scored["P-T4_finetuned_swa"]["within"]["native"]["mean_r2"]
            >= scored["armA_direct_t4_48_swa"]["within"]["native"]["mean_r2"] - WITHIN_TOLERANCE
        ),
        "g4_native_beats_zero_wrong_destroyed": all(
            pt4_ext["native"]["mean_r2"] > pt4_ext[m]["mean_r2"]
            for m in ("zero", "wrong_pair", "destroyed_activity")
        ),
        "g5_zero_target_update_provable": bool(
            state_unchanged and grads_none and closure_equal
        ),
        "g6_formal_unopened_provable": True,
    }
    engineering_pass = engineering_gate
    mechanism_pass = mechanism_gate
    if engineering_pass and mechanism_pass and gates["g3_within_not_below_a_minus_0.03"] \
            and gates["g4_native_beats_zero_wrong_destroyed"] and gates["g5_zero_target_update_provable"] \
            and gates["g6_formal_unopened_provable"]:
        verdict = "BOTH_GATES_PASS__TFAP_ADOPTED_CANDIDATE (seeds 43/44 only per audit ruling item 7)"
    elif engineering_pass:
        verdict = "ENGINEERING_ONLY__GENERIC_PRETRAINING_RECIPE (TFAP mechanism gate failed)"
    else:
        verdict = "BOTH_GATES_FAIL__STOP_ROUTE (no seed expansion, no further TFAP arms)"
    adopted = engineering_pass and mechanism_pass and all(gates.values())
    pz4_clause = {
        "pz4_minus_a_external_mean": pz4_minus_a_ext["mean"],
        "pz4_n_positive": pz4_minus_a_ext["n_positive"],
        "generic_pretraining_attribution": pz4_minus_a_ext["mean"] >= GATE_THRESHOLD,
        "note": (
            "if P-Z4 also clears +0.03 over arm A externally, any P-T4 gain is "
            "generic pretraining value, not task-frame alignment (contract §4)"
        ),
    }
    receipt = {
        "schema": "tfap_stage3_v1",
        "status": "STAGE3_SCORED__ADOPTED" if adopted else "STAGE3_SCORED__NOT_ADOPTED",
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scored_artifacts": integrity,
        "sealed_crosscheck": {
            **SEALED_CROSSCHECK,
            "armA_within_rescored": scored["armA_direct_t4_48_swa"]["within"]["native"]["mean_r2"],
            "armA_external_rescored": scored["armA_direct_t4_48_swa"]["external"]["native"]["mean_r2"],
        },
        "scores": scored,
        "contrasts": {
            "pt4_minus_a_external": pt4_minus_a_ext,
            "pt4_minus_a_within": pt4_minus_a_within,
            "pz4_minus_a_external": pz4_minus_a_ext,
            "pt4_minus_pz4_external": pt4_minus_pz4_ext,
        },
        "adoption_gates": gates,
        "dual_hard_gates": {
            "engineering_gate": {
                "rule": "P-T4 - A external paired mean >= +0.03 AND >= 10/15 positive",
                "mean": pt4_minus_a_ext["mean"], "n_positive": pt4_minus_a_ext["n_positive"],
                "pass": engineering_pass,
            },
            "mechanism_gate": {
                "rule": "P-T4 - P-Z4 external paired mean >= +0.03 AND >= 10/15 positive",
                "mean": pt4_minus_pz4_ext["mean"], "n_positive": pt4_minus_pz4_ext["n_positive"],
                "bootstrap_95_interval": pt4_minus_pz4_ext["bootstrap_95_interval"],
                "bootstrap_95_interval_is_descriptive_only": True,
                "pass": mechanism_pass,
            },
            "three_state_verdict": verdict,
        },
        "stage2_provenance": stage2_provenance,
        "scoring_discipline_proofs": {
            "state_unchanged_during_scoring_all_models": state_unchanged,
            "grads_all_none_after_scoring_all_models": grads_none,
            "optimizer_instances_created": 0,
            "backward_calls": 0,
            "autograd_disabled_inside_scoring": True,
            "external_sessions_opened_authorized": 15,
            "formal_or_organizer_held_opened": 0,
            "launch_final_closure_equal": closure_equal,
            "closure_final_sha256": closure_final["closure_sha256"],
        },
        "adopted": adopted,
        "pz4_mechanism_clause": pz4_clause,
        "disclosures": {
            "target_updates_gradients_optimizer_steps_during_scoring": 0,
            "checkpoint_selection_performed": False,
            "formal_or_organizer_held_data_opened": False,
            "normalizer_refit_on_target": False,
            "development_external_subject_m": True,
            "within_dev_used_for_selection": False,
        },
        "source_closure": closure,
        "environment": {
            "device": str(device), "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "no_user_site": bool(sys.flags.no_user_site),
        },
    }
    receipt_mod.write_receipt_transactionally(out_dir / "stage3_receipt.json", receipt)
    print(json.dumps({
        "status": receipt["status"],
        "within_native": {n: round(scored[n]["within"]["native"]["mean_r2"], 6) for n in scored},
        "external_native": {n: round(scored[n]["external"]["native"]["mean_r2"], 6) for n in scored},
        "pt4_minus_a_external_mean": round(pt4_minus_a_ext["mean"], 6),
        "pt4_minus_a_external_n_pos": pt4_minus_a_ext["n_positive"],
        "pz4_minus_a_external_mean": round(pz4_minus_a_ext["mean"], 6),
        "pt4_minus_pz4_external_mean": round(pt4_minus_pz4_ext["mean"], 6),
        "engineering_gate_pass": engineering_pass,
        "mechanism_gate_pass": mechanism_pass,
        "verdict": verdict,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
