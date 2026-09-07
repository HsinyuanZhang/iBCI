"""Within-validation four-diagnostic evaluation for the LARGE TFPD cells.

Same discipline as run_stage1_val_eval.py (trimmed batches, pooled per-session
R², frozen diagnostic seeds, epoch window 5-12) but builds the
TaskFrameSetAttentionDecoder and binds only the large-rung files.
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
BOUND = ("src/tfpd_large.py", "scripts/run_stage1_large_eval.py")


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


def evaluate_mode(model, batches, mode: str) -> dict:
    device = next(model.parameters()).device
    preds, targets = {}, {}
    with torch.no_grad():
        for neural, behavior, side, session in batches:
            neural, behavior, side = neural.to(device), behavior.to(device), side.to(device)
            carrier, x = side, neural
            if mode == "zero":
                carrier = torch.zeros_like(side)
            elif mode == "wrong_pair":
                g = torch.Generator().manual_seed(1234)
                carrier = side[:, torch.randperm(side.shape[1], generator=g)]
            elif mode == "destroyed_activity":
                x = destroy_activity(neural, seed=4321)
            prediction = model(x, carrier)
            valid = (behavior != PAD_VALUE).all(dim=-1)
            mask = valid  # row mask keeps [n, 2]; element-wise would flatten dims
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
    parser.add_argument("--arm", required=True, choices=["large_t4", "large_z4"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/stage1_source_cells_v1")
    args = parser.parse_args()

    from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule
    import mc_maze.a2_matched_subject_shift_v2_core as a2
    from src.tfpd_large import TaskFrameSetAttentionDecoder

    if not sys.flags.no_user_site:
        raise SystemExit("PYTHONNOUSERSITE=1 is mandatory")

    carrier_group = args.arm.split("_")[1]
    cell_dir = args.output_root / args.arm
    terminal_path = cell_dir / "terminal_receipt.json"
    if not terminal_path.is_file():
        raise SystemExit(f"cell not terminal: {cell_dir}")

    def epoch_of(p: Path) -> int:
        return int(p.name.split("=", 1)[1].split("-", 1)[0])

    window = [c for c in sorted((cell_dir / "epoch_ckpts").glob("*.ckpt")) if 5 <= epoch_of(c) + 1 <= 12]
    if len(window) != 8:
        raise SystemExit(f"expected 8 window checkpoints, found {len(window)}")

    dm = Dandi688MultiSessionDataModule(
        data_dir=str(a2.SUBC_DATA_ROOT), task="CO", split_counts=(27, 6, 6), batch_size=32,
        window_size=50, calibration_n_trials=30, max_trial_length=100, bin_size_ms=20,
        num_workers=4, random_calibration=False, seed=42, max_units_exclusive=100,
        cache_dir=str(a2.SOURCE_CACHE_ROOT), signal_view="sua",
        side_feature_group=carrier_group, side_feature_pool_size=30,
        train_val_manifest_path=str(a2.MANIFEST_PATH),
    )
    dm.setup("validate")
    batches = []
    for loader in dm.val_dataloader():
        for batch in loader:
            neural, behavior, _calib, session, side = batch[:5]
            batches.append((neural, behavior, side, session))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    modes = ["native"] + (["zero", "wrong_pair", "destroyed_activity"] if carrier_group == "t4" else [])
    results = {m: {"per_epoch": [], "mean_r2": None} for m in modes}
    for ckpt_path in window:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = TaskFrameSetAttentionDecoder()
        stripped = {k[len("model."):]: v for k, v in state["state_dict"].items()}
        model.load_state_dict(stripped)
        model.to(device).eval()
        for mode in modes:
            score = evaluate_mode(model, batches, mode)
            score["epoch"] = ckpt_path.name
            results[mode]["per_epoch"].append(score)
        del model
    for mode in modes:
        results[mode]["mean_r2"] = float(np.mean([e["mean_r2"] for e in results[mode]["per_epoch"]]))

    out_path = cell_dir / "within_val_epoch_window_eval.json"
    payload = {
        "schema": "tfpd_stage1_large_within_val_epoch_window_v1",
        "status": "COMPLETE",
        "arm": args.arm,
        "epoch_window_one_based": [5, 12],
        "checkpoints": [{"file": c.name, "sha256": sha256_file(c)} for c in window],
        "terminal_receipt_sha256": sha256_file(terminal_path),
        "source_closure": {
            rel: sha256_file(ROOT / rel) for rel in BOUND
        },
        "modes": results,
        "device": str(device),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_immutable(out_path, payload)
    print(json.dumps({m: round(r["mean_r2"], 4) for m, r in results.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
