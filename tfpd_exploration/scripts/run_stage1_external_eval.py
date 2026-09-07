"""External (subject-M) evaluator for TFPD Stage-1 cells — AUTHORIZATION-GATED.

House rule: opening sub-M target NWB data requires an explicit, per-invocation
authorization.  This script refuses to touch target data unless
--authorize-target I_AUTHORIZE_TFPD_DEVELOPMENT_TARGET_ACCESS is passed AND
matches the frozen value.  --dry-run constructs the full plan (roster, model,
checkpoints, modes) without opening any NWB.

Scoring surface: the audited 15-session external sub-M roster through the same
Dandi688MultiSessionDataModule family pointed at the sub-M data root, M30
chronological support, query strictly after trial 30, per-session pooled R²,
epoch window 5-12.  Models: bilinear / population_vector / large
(spintshape has its own streaming path and is out of scope here).
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
sys.path.insert(0, str(ROOT))

PAD_VALUE = -1.0
AUTH_VALUE = "I_AUTHORIZE_TFPD_DEVELOPMENT_TARGET_ACCESS"
MODELS = {
    "bl": ("bilinear", "src/tfpd/bilinear_readin.py"),
    "pv": ("population_vector", "src/tfpd/population_vector.py"),
    "large": ("task_frame_set_attention", "src/tfpd_large.py"),
}


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


from src.tfpd_lane.matched_scorer import session_r2 as r2_score  # single matched implementation


def destroy_activity(x: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    batch, length, num_units = x.shape
    lags = torch.randint(1, length, (num_units,), generator=generator)
    out = x.clone()
    for unit in range(num_units):
        out[:, :, unit] = torch.roll(x[:, :, unit], shifts=int(lags[unit]), dims=1)
    return out


def evaluate_mode(model, loaders, mode: str) -> dict:
    device = next(model.parameters()).device
    preds, targets = {}, {}
    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                neural, behavior, _calib, session, side = batch[:5]
                neural, behavior, side = neural.to(device), behavior.to(device), side.to(device)
                x, c = neural, side
                if mode == "zero":
                    c = torch.zeros_like(side)
                elif mode == "wrong_pair":
                    g = torch.Generator().manual_seed(1234)
                    c = side[:, torch.randperm(side.shape[1], generator=g)]
                elif mode == "destroyed_activity":
                    x = destroy_activity(neural, seed=4321)
                prediction = model(x, c)
                valid = (behavior != PAD_VALUE).all(dim=-1)
                # Row-mask selection preserves [n, 2]; an element-wise mask would flatten
                # both velocity dims into one 1-D stream and turn the per-output mean
                # into a grand mean (the optimism bug flagged in the 11:30 ledger).
                mask = valid
                name = session if isinstance(session, str) else session[0]
                preds.setdefault(name, []).append(prediction[mask].cpu())
                targets.setdefault(name, []).append(behavior[mask].cpu())
    per_session = [
        {"session": n, "r2": r2_score(torch.cat(p), torch.cat(t))}
        for n, p, t in zip(preds, preds.values(), targets.values())
    ]
    return {
        "per_session": per_session,
        "mean_r2": float(np.mean([s["r2"] for s in per_session])),
        "positive_sessions": int(sum(s["r2"] > 0 for s in per_session)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, help="e.g. large_t4, bl_t4, pv_z4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/stage1_external_eval_v1")
    parser.add_argument("--authorize-target", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule
    import mc_maze.a2_matched_subject_shift_v2_core as a2

    if not sys.flags.no_user_site:
        raise SystemExit("PYTHONNOUSERSITE=1 is mandatory")

    prefix, carrier_group = args.cell.split("_")
    if prefix not in MODELS:
        raise SystemExit(f"unknown model prefix {prefix}")
    authorized = args.authorize_target == AUTH_VALUE
    if not args.dry_run and not authorized:
        raise SystemExit("target data access requires --authorize-target " + AUTH_VALUE)

    cell_root_candidates = [ROOT / "results/stage1_source_cells_v1_r1", ROOT / "results/stage1_source_cells_v1"]

    def epoch_of(p: Path) -> int:
        return int(p.name.split("=", 1)[1].split("-", 1)[0])

    cell_dir = None
    window = []
    for root in cell_root_candidates:
        candidate = root / args.cell
        if not (candidate / "terminal_receipt.json").is_file():
            continue
        found = [c for c in sorted((candidate / "epoch_ckpts").glob("*.ckpt")) if 5 <= epoch_of(c) + 1 <= 12]
        if len(found) == 8:
            cell_dir, window = candidate, found
            break
        # Incomplete cells (e.g. the superseded pv_z4 in _v1) are skipped in
        # favour of a complete retrain root.
    if cell_dir is None:
        raise SystemExit(f"no complete terminal cell found for {args.cell}")

    plan = {
        "schema": "tfpd_stage1_external_eval_v1",
        "cell": args.cell,
        "data_root": str(a2.SUBM_DATA_ROOT),
        "epoch_window_one_based": [5, 12],
        "modes": ["native"] + (["zero", "wrong_pair", "destroyed_activity"] if carrier_group == "t4" else []),
        "authorized": authorized,
    }
    if args.dry_run:
        print(json.dumps({**plan, "status": "DRY_RUN__NO_NWB_OPENED"}, indent=1))
        return 0

    # Target data opens HERE and nowhere earlier in this process.
    dm = Dandi688MultiSessionDataModule(
        data_dir=str(a2.SUBM_DATA_ROOT), task="CO", split_counts=(0, 15, 0), batch_size=32,
        window_size=50, calibration_n_trials=30, max_trial_length=100, bin_size_ms=20,
        num_workers=2, random_calibration=False, seed=42, max_units_exclusive=100,
        cache_dir=str(ROOT / "cache/subm_external_v1"), signal_view="sua",
        side_feature_group=carrier_group, side_feature_pool_size=30,
    )
    # House rule: the external surface is scored under the frozen strict-27
    # SOURCE-ONLY behavior normalizer; refitting stats on target sessions is
    # forbidden (A2/B0 contract).  Inject it and verify the semantic SHA.
    from mc_maze.multisession_datamodule import fit_behavior_stats

    train_paths, _val_paths, _names = a2.active_source_session_paths()
    mean, std = fit_behavior_stats(train_paths, 20, cache_dir=a2.SOURCE_CACHE_ROOT)
    semantic = a2.normalizer_value_sha256(mean, std)
    if not semantic.startswith("f062506c"):
        raise SystemExit(f"source normalizer semantic SHA drift: {semantic}")
    dm._behavior_stats = (mean, std)
    # Side-feature z-scoring is likewise train-only (strict-27 source); inject
    # the source-fit statistics instead of letting the module refit on target.
    if carrier_group != "none":
        from mc_maze.unit_side_features import fit_side_feature_stats

        if carrier_group == "t4":
            side_mean, side_std = fit_side_feature_stats(
                train_paths, feature_group=carrier_group, pool_size=30,
                cache_dir=a2.SOURCE_CACHE_ROOT, signal_view="sua",
            )
        else:
            import numpy as _np
            side_mean, side_std = _np.zeros(4, _np.float32), _np.ones(4, _np.float32)
        dm._side_feature_stats = (side_mean, side_std)
    dm.setup("validate")
    loaders = dm.val_dataloader()

    if prefix == "large":
        from src.tfpd_large import TaskFrameSetAttentionDecoder as Model
    elif prefix == "bl":
        from src.tfpd.bilinear_readin import BilinearTaskFrameDecoder as Model
    else:
        from src.tfpd.population_vector import LearnedPopulationVectorDecoder as Model

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    modes = plan["modes"]
    results = {m: {"per_epoch": [], "mean_r2": None} for m in modes}
    for ckpt_path in window:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = Model()
        stripped = {k[len("model."):]: v for k, v in state["state_dict"].items()}
        model.load_state_dict(stripped)
        model.to(device).eval()
        for mode in modes:
            score = evaluate_mode(model, loaders, mode)
            score["epoch"] = ckpt_path.name
            results[mode]["per_epoch"].append(score)
        del model
    for mode in modes:
        results[mode]["mean_r2"] = float(np.mean([e["mean_r2"] for e in results[mode]["per_epoch"]]))

    out_path = args.output_root / f"{args.cell}__external_subject_M.json"
    payload = {
        **plan,
        "status": "DEVELOPMENT_EXTERNAL_SCORE_COMPLETE__NOT_OFFICIAL",
        "checkpoints": [{"file": c.name, "sha256": sha256_file(c)} for c in window],
        "modes": results,
        "device": str(device),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_immutable(out_path, payload)
    print(json.dumps({m: round(r["mean_r2"], 4) for m, r in results.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
