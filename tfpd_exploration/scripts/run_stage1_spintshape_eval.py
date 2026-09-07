"""Within-validation four-diagnostic evaluation for the SPINT-shape comparator cells.

The streaming model consumes calib_trials [B,30,100,N] (the memory-heavy
tensor), so unlike the TFPD evaluators this script iterates the val loaders
lazily per (checkpoint, mode) instead of materializing trimmed batches.
Diagnostics: native / zero side features / wrong-pair side features /
destroyed activity (frozen circular-shift seed).  Pooled per-session R²,
epoch window 5-12.
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
sys.path.insert(0, str(ROOT))  # src.tfpd_lane must resolve to THIS package

PAD_VALUE = -1.0
BOUND = ("src/tfpd/spintshape_module.py", "scripts/run_stage1_spintshape_eval.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_immutable(path: Path, payload: dict) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing receipt {path}")
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


def _load_spintshape_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tfpd_spintshape_module", ROOT / "src" / "tfpd" / "spintshape_module.py"
    )
    module = importlib.util.module_from_spec(spec)
    # The module imports streaming_calibration_exp's `src.models.*`; with ROOT
    # first on sys.path (for src.tfpd_lane), streaming must temporarily take
    # precedence during exec, then be restored.
    saved_path = list(sys.path)
    # Purge cached 'src*' entries: line 52 imports src.tfpd_lane under the
    # tfpd root, and a cached tfpd `src` would shadow streaming's regardless
    # of path order.
    saved_modules = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for k in list(saved_modules):
        del sys.modules[k]
    sys.path.insert(0, str(REPO / "streaming_calibration_exp"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            if k not in saved_modules:
                del sys.modules[k]
        sys.modules.update(saved_modules)
    return module


def evaluate_mode(model, loaders, mode: str) -> dict:
    device = next(model.parameters()).device
    preds, targets = {}, {}
    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                neural, behavior, calib, session, side = batch[:5]
                neural = neural.to(device)
                behavior = behavior.to(device)
                calib = calib.to(device)
                side = side.to(device)
                x, c, cb = neural, side, calib
                if mode == "zero":
                    c = torch.zeros_like(side)
                elif mode == "wrong_pair":
                    g = torch.Generator().manual_seed(1234)
                    c = side[:, torch.randperm(side.shape[1], generator=g)]
                elif mode == "destroyed_activity":
                    x = destroy_activity(neural, seed=4321)
                prediction, _identity = model(x, calib_trials=cb, side_features=c)
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
    return {"per_session": per_session, "mean_r2": float(np.mean([s["r2"] for s in per_session]))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["spintshape_t4", "spintshape_z4"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/stage1_source_cells_v1")
    parser.add_argument("--external", action="store_true", help="score the sub-M external surface")
    parser.add_argument("--swa", action="store_true", help="score the predeclared final-four SWA artifact")
    parser.add_argument("--authorize-target", default="")
    args = parser.parse_args()

    from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule
    import mc_maze.a2_matched_subject_shift_v2_core as a2

    if not sys.flags.no_user_site:
        raise SystemExit("PYTHONNOUSERSITE=1 is mandatory")
    if args.external and args.authorize_target != "I_AUTHORIZE_TFPD_DEVELOPMENT_TARGET_ACCESS":
        raise SystemExit("external scoring requires --authorize-target I_AUTHORIZE_TFPD_DEVELOPMENT_TARGET_ACCESS")

    carrier_group = args.arm.split("_")[1]
    cell_dir = args.output_root / args.arm
    terminal_path = cell_dir / "terminal_receipt.json"
    if not terminal_path.is_file():
        raise SystemExit(f"cell not terminal: {cell_dir}")

    def epoch_of(p: Path) -> int:
        return int(p.name.split("=", 1)[1].split("-", 1)[0])

    if args.swa:
        swa_path = cell_dir / "swa_final4.pt"
        if not swa_path.is_file():
            raise SystemExit(f"SWA artifact missing: {swa_path}")
        window = [swa_path]
    else:
        window = [c for c in sorted((cell_dir / "epoch_ckpts").glob("*.ckpt")) if 5 <= epoch_of(c) + 1 <= 12]
        if len(window) != 8:
            raise SystemExit(f"expected 8 window checkpoints, found {len(window)}")

    if args.external:
        dm = Dandi688MultiSessionDataModule(
            data_dir=str(a2.SUBM_DATA_ROOT), task="CO", split_counts=(0, 15, 0), batch_size=32,
            window_size=50, calibration_n_trials=30, max_trial_length=100, bin_size_ms=20,
            num_workers=2, random_calibration=False, seed=42, max_units_exclusive=100,
            cache_dir=str(ROOT / "cache/subm_external_v1"), signal_view="sua",
            side_feature_group=carrier_group, side_feature_pool_size=30,
        )
        from mc_maze.multisession_datamodule import fit_behavior_stats
        from mc_maze.unit_side_features import fit_side_feature_stats

        train_paths, _val_paths, _names = a2.active_source_session_paths()
        mean, std = fit_behavior_stats(train_paths, 20, cache_dir=a2.SOURCE_CACHE_ROOT)
        semantic = a2.normalizer_value_sha256(mean, std)
        if not semantic.startswith("f062506c"):
            raise SystemExit(f"source normalizer semantic SHA drift: {semantic}")
        dm._behavior_stats = (mean, std)
        if carrier_group == "t4":
            side_mean, side_std = fit_side_feature_stats(
                train_paths, feature_group=carrier_group, pool_size=30,
                cache_dir=a2.SOURCE_CACHE_ROOT, signal_view="sua",
            )
        else:
            import numpy as _np
            side_mean, side_std = _np.zeros(4, _np.float32), _np.ones(4, _np.float32)
        dm._side_feature_stats = (side_mean, side_std)
    else:
        dm = Dandi688MultiSessionDataModule(
            data_dir=str(a2.SUBC_DATA_ROOT), task="CO", split_counts=(27, 6, 6), batch_size=32,
            window_size=50, calibration_n_trials=30, max_trial_length=100, bin_size_ms=20,
            num_workers=2, random_calibration=False, seed=42, max_units_exclusive=100,
            cache_dir=str(a2.SOURCE_CACHE_ROOT), signal_view="sua",
            side_feature_group=carrier_group, side_feature_pool_size=30,
            train_val_manifest_path=str(a2.MANIFEST_PATH),
        )
    dm.setup("validate")
    loaders = dm.val_dataloader()

    spintshape = _load_spintshape_module()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Our own checkpoints contain torch.nn.parameter.UninitializedParameter (the
    # LazyLinear additive port that the coupled streaming path never
    # materializes); allowlist it for weights-only loading.
    torch.serialization.add_safe_globals([torch.nn.parameter.UninitializedParameter])
    modes = ["native"] + (["zero", "wrong_pair", "destroyed_activity"] if carrier_group == "t4" else [])
    results = {m: {"per_epoch": [], "mean_r2": None} for m in modes}
    for ckpt_path in window:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = spintshape.build_spintshape_model(seed=42)
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

    if args.external:
        out_path = ROOT / "results/stage1_external_eval_v1" / f"{args.arm}__external_subject_M{'__swa' if args.swa else ''}.json"
        schema = "tfpd_stage1_spintshape_external_epoch_window_v1"
        status = "DEVELOPMENT_EXTERNAL_SCORE_COMPLETE__NOT_OFFICIAL"
    elif args.swa:
        out_path = cell_dir / "within_val_swa_final4_eval.json"
        schema = "tfpd_stage1_spintshape_within_val_swa_v1"
        status = "COMPLETE"
    else:
        out_path = cell_dir / "within_val_epoch_window_eval.json"
        schema = "tfpd_stage1_spintshape_within_val_epoch_window_v1"
        status = "COMPLETE"
    payload = {
        "schema": schema,
        "status": status,
        "arm": args.arm,
        "epoch_window_one_based": [5, 12],
        "checkpoints": [{"file": c.name, "sha256": sha256_file(c)} for c in window],
        "terminal_receipt_sha256": sha256_file(terminal_path),
        "source_closure": {rel: sha256_file(ROOT / rel) for rel in BOUND},
        "modes": results,
        "device": str(device),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_immutable(out_path, payload)
    print(json.dumps({m: round(r["mean_r2"], 4) for m, r in results.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
