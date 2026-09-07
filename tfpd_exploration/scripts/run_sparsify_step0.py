#!/usr/bin/env python3
"""BEAT_A2_SPARSIFICATION Step 0 (workorder §3): zero-GPU-training, read-only.

(3.1) Rescore the sealed D and DH SWAs at the GOVERNING granularity — the
exact A2 `_r1` scorer convention: `matched_scorer.session_r2`, variance
weighted, equal session weight, LAST bin of each 50-bin window, strict-27
source-only normalizers (semantic SHA f062506c…) — on the within 6 and
external 15 surfaces.  Diagnostic only: it may guide a later S64 proposal and
cannot change S2's frozen 2-head base.

(3.2) Behavior-scaling parity audit, recorded, changing nothing: the A2
teacher lineage trains on `behavior_scaling_factor=5.0`-scaled targets and
divides by five at deployment; Arm A / D / DH train on standardized behavior
directly (no scaling factor exists anywhere in their lineage).  R and S2 stay
exact Arm A/D convention replicas; the mismatch is disclosed as an
A2-vs-teacher-free training-parameterization confound.

Authority discipline: binds the four §2 immutable bodies/sidecars, REJECTS the
void `results/a2_matched_rescore_v1/` root (present on disk, never opened),
and verifies the four governing numbers reproduce bit-exactly from the sealed
_r1 receipt before scoring.

Receipt (0444): results/sparsification_step0_v1/step0_receipt.json.
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

AUTH_VALUE = "I_AUTHORIZE_TFPD_DEVELOPMENT_TARGET_ACCESS"
BOUND_PATTERNS = (
    "scripts/run_sparsify_step0.py",
    "scripts/run_a2_matched_rescore.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "src/tfpd_lane/arm_common.py",
)
MODELS = {
    "D_swa": ROOT / "results/pop_robust_v1/cellD_2heads_dynamic_dropout/swa_final4.pt",
    "DH_swa": ROOT / "results/pop_robust_v1/cellDH_64heads_dynamic_dropout/swa_final4.pt",
}
GOVERNING = {
    "a2_pooled_external": 0.3460880616472827,
    "a2_pooled_within": 0.5776186750994788,
    "armA_external": 0.2603564786414305,
    "armA_within": 0.5538396437962850,
}
VOID_ROOT = ROOT / "results/a2_matched_rescore_v1"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/sparsification_step0_v1")
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
        print(json.dumps({"schema": "tfpd_sparsification_step0_v1",
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
    rescorer = _load_module(
        "tfpd_a2_rescorer", ROOT / "scripts/run_a2_matched_rescore.py"
    )
    closure = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)

    # ---- authority bindings (§2) + void-root rejection -----------------------
    authorities = {
        "a2_r1_receipt": {
            "path": "results/a2_matched_rescore_v1_r1/a2_rescore_receipt.json",
            "expected_sha256": "0ae0f74c6b5d606599d1108378836c8cbb7d1e5df05a453aa83f682255734d4e",
        },
        "armA_swa": {
            "path": "results/admission_arms_v1/armA_direct_t4_48/swa_final4.pt",
            "expected_sha256": "920eb4c9827dbe66f98e110d9e4880d6f2ea4180fe9acc8ac30cacf3073f6ad0",
        },
        "DDH_fullwindow_receipt": {
            "path": "results/pop_robust_v1/matched_score_receipt.json",
            "expected_sha256": "91bf81f9bc33c32bb1e6a2472d465a6e3c8eb50fd36c3b249dfea4af27652aaf",
        },
    }
    for name, entry in authorities.items():
        actual = arm_common.sha256_file(ROOT / entry["path"])
        if actual != entry["expected_sha256"]:
            raise SystemExit(f"authority SHA mismatch for {name}")
        entry["verified"] = True
    void_rejected = {
        "void_root": str(VOID_ROOT),
        "present_on_disk": VOID_ROOT.is_file(),
        "opened": False,
        "policy": "rejected per §2; all A2/ArmA numbers come from the _r1 receipt only",
    }

    a2_r1 = json.loads((ROOT / authorities["a2_r1_receipt"]["path"]).read_text())
    pooled_within = float(np.mean(list(a2_r1["pooled_per_session"]["A2_t4_pooled_within"].values())))
    pooled_external = float(np.mean(list(a2_r1["pooled_per_session"]["A2_t4_pooled_external"].values())))
    governing_reproduced = {
        "a2_pooled_external": pooled_external == GOVERNING["a2_pooled_external"],
        "a2_pooled_within": pooled_within == GOVERNING["a2_pooled_within"],
        "armA_external": (
            a2_r1["results"]["armA_direct_t4_48_swa_lastbin"]["external"]["mean_r2"]
            == GOVERNING["armA_external"]
        ),
        "armA_within": (
            a2_r1["results"]["armA_direct_t4_48_swa_lastbin"]["within"]["mean_r2"]
            == GOVERNING["armA_within"]
        ),
    }
    if not all(governing_reproduced.values()):
        raise SystemExit(f"governing numbers failed to reproduce bit-exactly: {governing_reproduced}")

    # ---- (3.2) behavior-scaling parity audit (recorded, no change) ----------
    scaling_parity = {
        "A2_teacher_lineage": {
            "behavior_scaling_factor": 5.0,
            "predict_scaled_behavior": True,
            "deployment": "output divided by 5.0 (decode_last_behavior / _slice_last_timestep)",
            "evidence": "streaming_calibration_module hparams + the _r1 rescorer's output_scale=5.0 division",
        },
        "armA_D_DH_lineage": {
            "behavior_scaling_factor": None,
            "predict_scaled_behavior": False,
            "deployment": "model output IS the standardized behavior; no scaling factor exists in the spintshape/arm-runner/pop-robust lineage",
            "evidence": "src/tfpd/spintshape_module.py loss on raw model output vs standardized behavior; scripts/run_admission_arm.py & run_pop_robust_cell.py contain no scaling factor",
            "scoring_convention_here": "output_scale=1.0 in the last-bin scorer",
        },
        "R_S2_policy": "exact Arm A/D convention replicas (unscaled); the mismatch is a disclosed A2-vs-teacher-free parameterization confound, not a sparsification effect",
    }

    # ---- (3.1) D/DH last-bin rescore ----------------------------------------
    import mc_maze.a2_matched_subject_shift_v2_core as a2

    surfaces = rescorer.build_surfaces(args, a2, "t4")
    spintshape = _load_module("tfpd_spintshape_module", ROOT / "src/tfpd/spintshape_module.py")

    results = {}
    integrity = {}
    for name, path in MODELS.items():
        path = Path(path)
        sidecar = Path(str(path) + ".sha256")
        if not path.is_file() or not sidecar.is_file():
            raise SystemExit(f"cell SWA or sidecar missing: {path}")
        sha = arm_common.sha256_file(path)
        if sha != sidecar.read_text().split()[0]:
            raise SystemExit(f"cell SWA SHA mismatch: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        state = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in state.items()}
        heads = {"D_swa": 2, "DH_swa": 64}[name]
        model = spintshape.build_spintshape_model(seed=42)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()

        def forward(neural, calib, side, _m=model):
            return _m(neural, calib_trials=calib, side_features=side)

        (within_ds, within_starts), (ext_ds, ext_starts) = surfaces
        results[name] = {
            "within": rescorer.score_last_bin(forward, within_ds, within_starts, device,
                                              matched_scorer.session_r2, output_scale=1.0),
            "external": rescorer.score_last_bin(forward, ext_ds, ext_starts, device,
                                                matched_scorer.session_r2, output_scale=1.0),
        }
        integrity[name] = {"path": str(path), "sha256": sha, "num_heads": heads}
        print(json.dumps({
            "model": name, "heads": heads,
            "within_lastbin": round(results[name]["within"]["mean_r2"], 6),
            "external_lastbin": round(results[name]["external"]["mean_r2"], 6),
        }), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # paired vs governing arm A per session (A2 _r1 values reused read-only)
    armA_per_session = {
        surface: {
            row["session"]: row["r2"]
            for row in a2_r1["results"]["armA_direct_t4_48_swa_lastbin"][surface]["per_session"]
        }
        for surface in ("within", "external")
    }
    contrasts = {}
    for name in MODELS:
        for surface in ("within", "external"):
            mine = {row["session"]: row["r2"] for row in results[name][surface]["per_session"]}
            deltas = [mine[s] - armA_per_session[surface][s] for s in sorted(armA_per_session[surface])]
            stats = matched_scorer.paired_session_stats(deltas, seed=42, n_boot=10000)
            stats["contrast"] = f"{name} - armA_lastbin ({surface})"
            contrasts[f"{name}_minus_armA_{surface}"] = stats

    d_ext = results["D_swa"]["external"]["mean_r2"]
    receipt = {
        "schema": "tfpd_sparsification_step0_v1",
        "status": "STEP0_COMPLETE__READ_ONLY",
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authorities": authorities,
        "void_root_rejected": void_rejected,
        "governing_numbers_reproduced_bitexactly": governing_reproduced,
        "governing_bars": GOVERNING,
        "behavior_scaling_parity_audit": scaling_parity,
        "scoring_convention": {
            "governing": "last-bin, variance-weighted, equal session weight, source-only normalizers (f062506c…)",
            "output_scale": "1.0 for D/DH (unscaled lineage)",
            "full_window_note": "the sealed full-window D/DH receipt (91bf81f9…) remains the diagnostic; this last-bin pair is governing",
        },
        "scored_artifacts": integrity,
        "results": results,
        "contrasts_vs_governing_armA": contrasts,
        "d_external_gain_over_armA": d_ext - GOVERNING["armA_external"],
        "disclosures": {
            "sealed_files_modified": False,
            "void_root_opened": False,
            "target_updates_gradients_optimizer_steps": 0,
            "formal_or_organizer_held_data_opened": False,
            "s2_base_changed_by_this_diagnostic": False,
        },
        "source_closure": closure,
        "environment": {
            "device": str(device), "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "no_user_site": bool(sys.flags.no_user_site),
        },
    }
    receipt_mod.write_receipt_transactionally(out_dir / "step0_receipt.json", receipt)
    print(json.dumps({
        "D_lastbin": {"within": round(results["D_swa"]["within"]["mean_r2"], 6),
                      "external": round(d_ext, 6)},
        "DH_lastbin": {"within": round(results["DH_swa"]["within"]["mean_r2"], 6),
                       "external": round(results["DH_swa"]["external"]["mean_r2"], 6)},
        "D_external_gain_over_armA": round(d_ext - GOVERNING["armA_external"], 6),
        "R_recovery_denominator_note": "R_recovery uses THIS last-bin D gain as the governing denominator",
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
