#!/usr/bin/env python3
"""Gate-4 external (subject-M) scoring for the admission arms — AUTHORIZATION-GATED.

Same external path and authorization discipline as the Stage-1 `extswa`
evaluators (`run_stage1_external_eval.py` / `run_stage1_spintshape_eval.py
--external --swa`):

- the audited 15-session external sub-M roster through the same
  Dandi688MultiSessionDataModule family (M30 chronological support, query
  strictly after trial 30, units < 100, sua);
- house rule enforced: target NWB data opens ONLY after
  --authorize-target I_AUTHORIZED... (frozen value below) and never before;
- the external surface is scored under the frozen strict-27 SOURCE-ONLY
  behavior and side-feature normalizers (injected + semantic-SHA verified);
  refitting statistics on target sessions is forbidden;
- one matched scorer everywhere: per-session variance-weighted R2, equal
  weight per session;
- modes: native + zero / wrong_pair / destroyed_activity diagnostics with the
  Stage-1 frozen seeds (1234 / 4321).

Differences from the Stage-1 spintshape extswa runner (disclosed in every
receipt): the scored artifact is the Gate-2 admission-arm final-four SWA
(`results/admission_arms_v1/arm{A,C}_*/swa_final4.pt`, sidecar-verified), the
schema is `tfpd_gate4_arm_external`, and the forward engine hoists the
per-session identity computation out of the batch loop (mathematically
identical; probe max-abs-diff recorded — 0.0 on this stack), which is the same
engine the Gate-3 within screen used.

Receipt (0444): results/gate4_arm_external_v1/arm{X}__external_subject_M.json.
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
WRONG_PAIR_SEED = 1234
DESTROY_SEED = 4321
EVAL_BATCH_SIZE = 128
ARM_PATHS = {
    "A": ROOT / "results/admission_arms_v1/armA_direct_t4_48/swa_final4.pt",
    "C": ROOT / "results/admission_arms_v1/armC_direct_t4_exposure_matched/swa_final4.pt",
}
MODES = ("native", "zero", "wrong_pair", "destroyed_activity")
BOUND_PATTERNS = (
    "scripts/run_gate4_arm_external.py",
    "scripts/run_z4_boundary_pilot.py",
    "src/tfpd_lane/arm_common.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "src/tfpd/spintshape_module.py",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def destroy_activity(x: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    batch, length, num_units = x.shape
    lags = torch.randint(1, length, (num_units,), generator=generator)
    out = x.clone()
    for unit in range(num_units):
        out[:, :, unit] = torch.roll(x[:, :, unit], shifts=int(lags[unit]), dims=1)
    return out


def score_session_modes(model, record, starts, device) -> dict:
    """All four modes for one external session with per-mode cached identity."""
    from torch.nn.parameter import UninitializedParameter  # noqa: F401

    calib = torch.from_numpy(record.calib_trials.copy()).float().unsqueeze(0).to(device)
    side = torch.from_numpy(record.side_features.copy()).float().unsqueeze(0).to(device)
    zero_side = torch.zeros_like(side)
    generator = torch.Generator().manual_seed(WRONG_PAIR_SEED)
    perm = torch.randperm(side.shape[1], generator=generator)
    wrong_side = side[:, perm]
    identities = {
        "native": model.compute_identity(calib, side_features=side),
        "zero": model.compute_identity(calib, side_features=zero_side),
        "wrong_pair": model.compute_identity(calib, side_features=wrong_side),
        # destroyed_activity keeps the native calibration/side identity and
        # destroys only the query-window time structure (Stage-1 semantics)
        "destroyed_activity": None,
    }
    identities["destroyed_activity"] = identities["native"]
    out = {}
    with torch.no_grad():
        for mode in MODES:
            identity = identities[mode]
            preds, tgts = [], []
            for i in range(0, len(starts), EVAL_BATCH_SIZE):
                chunk = starts[i : i + EVAL_BATCH_SIZE]
                neural = (
                    torch.from_numpy(
                        np.stack([record.neural[s : s + 50] for s in chunk])
                    )
                    .float()
                    .to(device)
                )
                behavior = (
                    torch.from_numpy(
                        np.stack([record.behavior[s : s + 50] for s in chunk])
                    )
                    .float()
                    .to(device)
                )
                if mode == "destroyed_activity":
                    neural = destroy_activity(neural, DESTROY_SEED)
                prediction = model.decode_with_identity(neural, identity)
                valid = (behavior != PAD_VALUE).all(dim=-1)
                preds.append(prediction[valid].cpu())
                tgts.append(behavior[valid].cpu())
            out[mode] = (torch.cat(preds), torch.cat(tgts))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["A", "C"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/gate4_arm_external_v1")
    parser.add_argument("--authorize-target", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not sys.flags.no_user_site:
        print("PYTHONNOUSERSITE=1 is mandatory", file=sys.stderr)
        return 3
    authorized = args.authorize_target == AUTH_VALUE
    if not args.dry_run and not authorized:
        print("target data access requires --authorize-target " + AUTH_VALUE, file=sys.stderr)
        return 3

    receipt_mod = _load_module("tfpd_lane_receipt", ROOT / "src/tfpd_lane/receipt.py")
    arm_common = _load_module("tfpd_lane_arm_common", ROOT / "src/tfpd_lane/arm_common.py")
    matched_scorer = _load_module(
        "tfpd_lane_matched_scorer", ROOT / "src/tfpd_lane/matched_scorer.py"
    )
    spintshape = _load_module("tfpd_spintshape_module", ROOT / "src/tfpd/spintshape_module.py")

    import mc_maze.a2_matched_subject_shift_v2_core as a2

    plan = {
        "schema": "tfpd_gate4_arm_external",
        "arm": args.arm,
        "artifact": str(ARM_PATHS[args.arm]),
        "data_root": str(a2.SUBM_DATA_ROOT),
        "n_sessions": 15,
        "modes": list(MODES),
        "normalizer": "strict-27 source-only behavior + side-feature stats (injected, semantic-SHA verified)",
        "authorized": authorized,
    }
    if args.dry_run:
        print(json.dumps({**plan, "status": "DRY_RUN__NO_NWB_OPENED"}, indent=1))
        return 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA unavailable ({args.device})", file=sys.stderr)
        return 3

    # ---- artifact integrity -------------------------------------------------
    path = Path(ARM_PATHS[args.arm])
    sidecar = Path(str(path) + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        print(f"arm SWA artifact or sidecar missing: {path}", file=sys.stderr)
        return 3
    artifact_sha = arm_common.sha256_file(path)
    if artifact_sha != sidecar.read_text().split()[0]:
        print("arm SWA SHA mismatch against sidecar", file=sys.stderr)
        return 3
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    state = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in state.items()}
    model = spintshape.build_spintshape_model(seed=42)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    # ---- target data opens HERE and nowhere earlier in this process ---------
    from mc_maze.multisession_datamodule import (
        Dandi688MultiSessionDataModule,
        fit_behavior_stats,
    )
    from mc_maze.unit_side_features import fit_side_feature_stats

    dm = Dandi688MultiSessionDataModule(
        data_dir=str(a2.SUBM_DATA_ROOT), task="CO", split_counts=(0, 15, 0), batch_size=32,
        window_size=50, calibration_n_trials=30, max_trial_length=100, bin_size_ms=20,
        num_workers=2, random_calibration=False, seed=42, max_units_exclusive=100,
        cache_dir=str(ROOT / "cache/subm_external_v1"), signal_view="sua",
        side_feature_group="t4", side_feature_pool_size=30,
    )
    train_paths, _val_paths, _names = a2.active_source_session_paths()
    mean, std = fit_behavior_stats(train_paths, 20, cache_dir=a2.SOURCE_CACHE_ROOT)
    semantic = a2.normalizer_value_sha256(mean, std)
    if not semantic.startswith("f062506c"):
        print(f"source normalizer semantic SHA drift: {semantic}", file=sys.stderr)
        return 3
    dm._behavior_stats = (mean, std)
    side_mean, side_std = fit_side_feature_stats(
        train_paths, feature_group="t4", pool_size=30,
        cache_dir=a2.SOURCE_CACHE_ROOT, signal_view="sua",
    )
    dm._side_feature_stats = (side_mean, side_std)
    side_semantic = a2.normalizer_value_sha256(side_mean, side_std)
    dm.setup("validate")
    val_dataset = dm.val_dataset
    roster = tuple(dm.session_splits["val"])
    if len(roster) != 15:
        print(f"external roster drift: {len(roster)} sessions", file=sys.stderr)
        return 3

    starts_by_session = {}
    for position, (session, start) in enumerate(val_dataset.window_indices):
        starts_by_session.setdefault(session, []).append(int(start))

    # forward-engine probe (identity-cached vs full forward) on the first batch
    probe_session = sorted(starts_by_session)[0]
    probe_record = val_dataset.sessions[probe_session]
    probe_starts = starts_by_session[probe_session][:8]
    calib = torch.from_numpy(probe_record.calib_trials.copy()).float().unsqueeze(0).to(device)
    side = torch.from_numpy(probe_record.side_features.copy()).float().unsqueeze(0).to(device)
    neural = (
        torch.from_numpy(np.stack([probe_record.neural[s : s + 50] for s in probe_starts]))
        .float()
        .to(device)
    )
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=side)
        fast = model.decode_with_identity(neural, identity)
        full, _ = model(
            neural, calib_trials=calib.expand(neural.shape[0], -1, -1, -1),
            side_features=side.expand(neural.shape[0], -1, -1),
        )
        probe_diff = float((fast - full).abs().max().item())

    per_mode_sessions = {m: [] for m in MODES}
    for session in sorted(starts_by_session):
        record = val_dataset.sessions[session]
        starts = starts_by_session[session]
        packed = score_session_modes(model, record, starts, device)
        for mode in MODES:
            preds, tgts = packed[mode]
            per_mode_sessions[mode].append(
                {"session": session, "r2": matched_scorer.session_r2(preds, tgts),
                 "n_windows": len(starts)}
            )
        print(json.dumps({"session": session, "native_r2": round(per_mode_sessions["native"][-1]["r2"], 4)}), flush=True)

    modes_result = {}
    for mode in MODES:
        rows = per_mode_sessions[mode]
        modes_result[mode] = {
            "per_session": rows,
            "mean_r2": float(np.mean([r["r2"] for r in rows])),
            "positive_sessions": int(sum(r["r2"] > 0 for r in rows)),
        }

    closure = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)
    payload = {
        **plan,
        "status": "DEVELOPMENT_EXTERNAL_SCORE_COMPLETE__NOT_OFFICIAL",
        "artifact_sha256": artifact_sha,
        "side_feature_semantic_sha256": side_semantic,
        "behavior_normalizer_semantic_sha256": semantic,
        "n_windows_total": int(sum(len(v) for v in starts_by_session.values())),
        "forward_engine": {
            "mode": "identity_cached_decode",
            "probe_max_abs_diff_vs_full_forward": probe_diff,
            "note": (
                "same engine as the Gate-3 within screen; the identity is constant per "
                "(session, mode), the probe bounds any numerical deviation from the "
                "Stage-1 extswa per-batch full forward"
            ),
        },
        "wrong_pair_seed": WRONG_PAIR_SEED,
        "destroyed_activity_seed": DESTROY_SEED,
        "modes": modes_result,
        "scorer": "src/tfpd_lane/matched_scorer.session_r2 (variance_weighted), equal weight per session",
        "disclosures": {
            "target_updates_gradients_optimizer_steps_during_scoring": 0,
            "checkpoint_selection_performed": False,
            "formal_or_organizer_held_data_opened": False,
            "normalizer_refit_on_target": False,
            "development_external_subject_m": True,
        },
        "source_closure": closure,
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "no_user_site": bool(sys.flags.no_user_site),
        },
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"arm{args.arm}__external_subject_M.json"
    if (out_dir / name).exists():
        print(f"refusing to overwrite existing receipt {out_dir / name}", file=sys.stderr)
        return 2
    receipt_mod.write_receipt_transactionally(out_dir / name, payload)
    print(json.dumps({m: round(r["mean_r2"], 4) for m, r in modes_result.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
