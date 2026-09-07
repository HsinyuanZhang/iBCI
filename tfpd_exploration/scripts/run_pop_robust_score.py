"""Matched scoring for population-robustness cells D and DH (within + external).

Loads each cell's predeclared SWA artifact, scores the six within-dev sessions
and the fifteen external sub-M sessions with the single matched scorer, runs
the four same-checkpoint diagnostics, and emits paired deltas versus sealed
Arm A with the contract's three-condition adoption gate.  Authorization-gated
for the external surface; strict-27 source normalizer injected and SHA-checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))
sys.path.insert(0, str(ROOT))

from src.tfpd_lane.matched_scorer import paired_session_stats  # after path setup

PAD_VALUE = -1.0
AUTH_VALUE = "I_AUTHORIZE_TFPD_DEVELOPMENT_TARGET_ACCESS"
BOUND = ("src/tfpd_lane/pop_robust.py", "scripts/run_pop_robust_score.py")
CELLS = {"D": "cellD_2heads_dynamic_dropout", "DH": "cellDH_64heads_dynamic_dropout"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_immutable(path: Path, payload: dict) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing receipt {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(sha256_file(path) + "  " + path.name + "\n")
    os.chmod(sidecar, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    residual = ((predictions - targets) ** 2).sum().item()
    total = ((targets - targets.mean(dim=0, keepdim=True)) ** 2).sum().item()
    return 1.0 - residual / max(total, 1e-12)


def destroy_activity(x: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    batch, length, num_units = x.shape
    lags = torch.randint(1, length, (num_units,), generator=generator)
    out = x.clone()
    for unit in range(num_units):
        out[:, :, unit] = torch.roll(x[:, :, unit], shifts=int(lags[unit]), dims=1)
    return out


def build_loaders(external: bool):
    from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule
    import mc_maze.a2_matched_subject_shift_v2_core as a2

    if external:
        dm = Dandi688MultiSessionDataModule(
            data_dir=str(a2.SUBM_DATA_ROOT), task="CO", split_counts=(0, 15, 0), batch_size=32,
            window_size=50, calibration_n_trials=30, max_trial_length=100, bin_size_ms=20,
            num_workers=2, random_calibration=False, seed=42, max_units_exclusive=100,
            cache_dir=str(ROOT / "cache/subm_external_v1"), signal_view="sua",
            side_feature_group="t4", side_feature_pool_size=30,
        )
        from mc_maze.multisession_datamodule import fit_behavior_stats
        from mc_maze.unit_side_features import fit_side_feature_stats

        train_paths, _v, _n = a2.active_source_session_paths()
        mean, std = fit_behavior_stats(train_paths, 20, cache_dir=a2.SOURCE_CACHE_ROOT)
        if not a2.normalizer_value_sha256(mean, std).startswith("f062506c"):
            raise SystemExit("source normalizer semantic SHA drift")
        dm._behavior_stats = (mean, std)
        side_mean, side_std = fit_side_feature_stats(
            train_paths, feature_group="t4", pool_size=30,
            cache_dir=a2.SOURCE_CACHE_ROOT, signal_view="sua",
        )
        dm._side_feature_stats = (side_mean, side_std)
    else:
        dm = Dandi688MultiSessionDataModule(
            data_dir=str(a2.SUBC_DATA_ROOT), task="CO", split_counts=(27, 6, 6), batch_size=32,
            window_size=50, calibration_n_trials=30, max_trial_length=100, bin_size_ms=20,
            num_workers=2, random_calibration=False, seed=42, max_units_exclusive=100,
            cache_dir=str(a2.SOURCE_CACHE_ROOT), signal_view="sua",
            side_feature_group="t4", side_feature_pool_size=30,
            train_val_manifest_path=str(a2.MANIFEST_PATH),
        )
    dm.setup("validate")
    return dm.val_dataloader()


def evaluate_model(model, loaders, mode: str, device) -> dict:
    preds, targets = {}, {}
    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                neural, behavior, calib, session, side = batch[:5]
                neural = neural.to(device); behavior = behavior.to(device)
                calib = calib.to(device); side = side.to(device)
                x, c = neural, side
                if mode == "zero":
                    c = torch.zeros_like(side)
                elif mode == "wrong_pair":
                    g = torch.Generator().manual_seed(1234)
                    c = side[:, torch.randperm(side.shape[1], generator=g)]
                elif mode == "destroyed_activity":
                    x = destroy_activity(neural, seed=4321)
                prediction, _ = model(x, calib_trials=calib, side_features=c)
                valid = (behavior != PAD_VALUE).all(dim=-1)
                name = session if isinstance(session, str) else session[0]
                preds.setdefault(name, []).append(prediction[valid].cpu())
                targets.setdefault(name, []).append(behavior[valid].cpu())
    per_session = [
        {"session": n, "r2": r2_score(torch.cat(p), torch.cat(t))}
        for n, p, t in zip(preds, preds.values(), targets.values())
    ]
    return {"per_session": per_session, "mean_r2": float(np.mean([s["r2"] for s in per_session]))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--authorize-target", default="")
    args = parser.parse_args()

    if not sys.flags.no_user_site:
        raise SystemExit("PYTHONNOUSERSITE=1 is mandatory")
    if args.authorize_target != AUTH_VALUE:
        raise SystemExit("external scoring requires --authorize-target " + AUTH_VALUE)

    sys.path.insert(0, str(REPO / "sua_exploration"))
    from src.tfpd_lane.pop_robust import build_population_robustness_model

    torch.serialization.add_safe_globals([torch.nn.parameter.UninitializedParameter])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Artifact integrity: SWA sha vs terminal receipt, before any data opens.
    swa_paths, integrity = {}, {}
    for cell, dirname in CELLS.items():
        cell_dir = ROOT / "results/pop_robust_v1" / dirname
        terminal = json.loads((cell_dir / "terminal_receipt.json").read_text())
        if terminal.get("status") != "CELL_TERMINAL":
            raise SystemExit(f"{cell} not terminal")
        swa = cell_dir / "swa_final4.pt"
        sha = sha256_file(swa)
        if terminal.get("swa_sha256", terminal.get("swa", {}).get("sha256", sha)) != sha:
            raise SystemExit(f"{cell} SWA SHA mismatch vs terminal receipt")
        swa_paths[cell], integrity[cell] = swa, {"sha256": sha, "heads": terminal.get("heads")}

    within_loaders = build_loaders(external=False)
    external_loaders = build_loaders(external=True)

    results = {}
    models_to_score = dict(swa_paths)
    # Arm A is scored live in the same pass so the pairing is same-engine,
    # same-loaders, no hand-copied numbers.
    armA_swa = ROOT / "results/admission_arms_v1/armA_direct_t4_48/swa_final4.pt"
    if not armA_swa.is_file():
        raise SystemExit("Arm A SWA artifact missing")
    models_to_score["armA"] = armA_swa
    for name, swa in models_to_score.items():
        if name == "armA":
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "sm_spintshape", ROOT / "src" / "tfpd" / "spintshape_module.py")
            mod = importlib.util.module_from_spec(spec)
            saved = list(sys.path)
            sys.path.insert(0, str(REPO / "streaming_calibration_exp"))
            for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
                del sys.modules[k]
            try:
                spec.loader.exec_module(mod)
            finally:
                sys.path[:] = saved
            model = mod.build_spintshape_model(seed=42)
        else:
            model = build_population_robustness_model(seed=42, cell=name)
        state = torch.load(swa, map_location="cpu", weights_only=False)["state_dict"]
        # LitModule "model." prefixes are stripped; raw StreamingSpintModel keys
        # (the matched_scorer-built Arm A SWA) load as-is.
        cleaned = {}
        for k, v in state.items():
            cleaned[k[len("model."):] if k.startswith("model.") else k] = v
        model.load_state_dict(cleaned)
        model.to(device).eval()
        results[name] = {
            "within": {m: evaluate_model(model, within_loaders, m, device) for m in
                       ["native", "zero", "wrong_pair", "destroyed_activity"]},
            "external": {m: evaluate_model(model, external_loaders, m, device) for m in
                         ["native", "zero", "wrong_pair", "destroyed_activity"]},
        }
        del model

    arm_a = {
        "external_native": results["armA"]["external"]["native"]["mean_r2"],
        "within_native": results["armA"]["within"]["native"]["mean_r2"],
        "external_per_session": [s["r2"] for s in results["armA"]["external"]["native"]["per_session"]],
        "within_per_session": [s["r2"] for s in results["armA"]["within"]["native"]["per_session"]],
    }

    def paired(cell: str) -> dict:
        ext_deltas = [
            s["r2"] - a for s, a in zip(
                results[cell]["external"]["native"]["per_session"],
                arm_a["external_per_session"],
            )
        ]
        win_deltas = [
            s["r2"] - a for s, a in zip(
                results[cell]["within"]["native"]["per_session"],
                arm_a["within_per_session"],
            )
        ]
        ext_stats = paired_session_stats(ext_deltas)
        win_stats = paired_session_stats(win_deltas)
        gate = (
            ext_stats["mean"] >= 0.03
            and ext_stats["n_positive"] >= 10
            and win_stats["mean"] >= -0.03
        )
        return {"external_delta": ext_stats, "within_delta": win_stats,
                "performance_candidate_gate": gate}

    out = {
        "schema": "pop_robust_matched_score_v1",
        "status": "SCORED",
        "cells": results,
        "integrity": integrity,
        "source_closure": {rel: sha256_file(ROOT / rel) for rel in BOUND},
        "arm_a_reference": arm_a,
        "paired_vs_arm_a": {c: paired(c) for c in CELLS},
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_immutable(ROOT / "results/pop_robust_v1/matched_score_receipt.json", out)
    for c in CELLS:
        print(c, "within", round(results[c]["within"]["native"]["mean_r2"], 4),
              "external", round(results[c]["external"]["native"]["mean_r2"], 4),
              "gate", out["paired_vs_arm_a"][c]["performance_candidate_gate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
