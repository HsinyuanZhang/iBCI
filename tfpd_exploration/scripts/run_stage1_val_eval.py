"""Stage-1 within-validation evaluator for TFPD cells.

Scores a completed source cell on the 6 strict-27 validation sessions (source-side
development data; no external target access) over the frozen epoch window 5-12
(zero-based epoch_004..epoch_011), and runs the four same-checkpoint mechanism
diagnostics on the T4 arm: aligned / exact-zero / wrong-pair / activity-destroyed.

Writes an immutable receipt.  CPU or a single visible GPU.
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

MODEL_BY_PREFIX = {"bl": "bilinear", "pv": "population_vector"}
PAD_VALUE = -1.0


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
    """Circularly shift each unit's series by a random lag (frozen seed),
    preserving each unit's marginal count sequence while destroying its
    alignment with behaviour."""
    generator = torch.Generator().manual_seed(seed)
    batch, length, num_units = x.shape
    lags = torch.randint(1, length, (num_units,), generator=generator)
    out = x.clone()
    for unit in range(num_units):
        out[:, :, unit] = torch.roll(x[:, :, unit], shifts=int(lags[unit]), dims=1)
    return out


def evaluate_mode(model, batches, mode: str) -> dict:
    """Pooled per-session R²: predictions are accumulated across all batches of a
    session and scored once.  Per-batch R² averaging is unstable — batches whose
    behaviour is near-constant (inter-trial rest) have vanishing target variance
    and explode the mean (observed -1e5-scale artifacts in the first attempt)."""
    device = next(model.parameters()).device
    session_preds: dict[str, list[torch.Tensor]] = {}
    session_targets: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for neural, behavior, side, session in batches:
            neural = neural.to(device)
            behavior = behavior.to(device)
            side = side.to(device)
            carrier = side
            x = neural
            if mode == "zero":
                carrier = torch.zeros_like(side)
            elif mode == "wrong_pair":
                generator = torch.Generator().manual_seed(1234)
                perm = torch.randperm(side.shape[1], generator=generator)
                carrier = side[:, perm]
            elif mode == "destroyed_activity":
                x = destroy_activity(neural, seed=4321)
            prediction = model(x, carrier)
            valid = (behavior != PAD_VALUE).all(dim=-1)
            mask = valid  # row mask keeps [n, 2]; element-wise would flatten dims
            name = session if isinstance(session, str) else session[0]
            session_preds.setdefault(name, []).append(prediction[mask].cpu())
            session_targets.setdefault(name, []).append(behavior[mask].cpu())
            del prediction
    per_session = [
        {
            "session": name,
            "r2": r2_score(torch.cat(preds), torch.cat(targets)),
            "n_bins": int(sum(t.numel() for t in targets) // targets[0].shape[-1]),
        }
        for name, preds, targets in zip(
            session_preds, session_preds.values(), session_targets.values()
        )
    ]
    return {
        "per_session": per_session,
        "mean_r2": float(np.mean([s["r2"] for s in per_session])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["bl_t4", "bl_z4", "pv_t4", "pv_z4"])
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/stage1_source_cells_v1")
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule
    import mc_maze.a2_matched_subject_shift_v2_core as a2
    from src.tfpd.stage1_module import build_stage1_model

    if not sys.flags.no_user_site:
        raise SystemExit("PYTHONNOUSERSITE=1 is mandatory")

    model_prefix, carrier_group = args.arm.split("_")
    cell_dir = args.output_root / args.arm
    terminal_path = cell_dir / "terminal_receipt.json"
    if not terminal_path.is_file():
        raise SystemExit(f"cell not terminal: {cell_dir}")

    ckpts = sorted((cell_dir / "epoch_ckpts").glob("*.ckpt"))

    def epoch_of(path: Path) -> int:
        return int(path.name.split("=", 1)[1].split("-", 1)[0])

    window = [c for c in ckpts if 5 <= epoch_of(c) + 1 <= 12]
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
    loaders = dm.val_dataloader()
    # Trim each batch to exactly the tensors the evaluator consumes.  The calib
    # tensor (32x30x100xN) is ~100x larger than everything else combined and is
    # never used here; materializing it for every window OOM-killed the first
    # evaluation attempt (exit 137).
    batches = []
    for loader in loaders:
        for batch in loader:
            if len(batch) == 6:
                neural, behavior, _calib, session, side, _electrode = batch
            else:
                neural, behavior, _calib, session, side = batch
            batches.append((neural, behavior, side, session))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    modes = ["native"]
    if carrier_group == "t4":
        modes += ["zero", "wrong_pair", "destroyed_activity"]
    results = {mode: {"per_epoch": [], "mean_r2": None} for mode in modes}
    for ckpt_path in window:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = build_stage1_model(MODEL_BY_PREFIX[model_prefix], seed=42)
        # Lightning stores the LitModule's parameters under the "model." prefix.
        stripped = {
            key[len("model."):]: value for key, value in state["state_dict"].items()
        }
        model.load_state_dict(stripped)
        model.to(device).eval()
        for mode in modes:
            score = evaluate_mode(model, batches, mode)
            score["epoch"] = ckpt_path.name
            results[mode]["per_epoch"].append(score)
    for mode in modes:
        results[mode]["mean_r2"] = float(np.mean([e["mean_r2"] for e in results[mode]["per_epoch"]]))

    out_path = args.out_path or (cell_dir / "within_val_epoch_window_eval.json")
    payload = {
        "schema": "tfpd_stage1_within_val_epoch_window_v1",
        "status": "COMPLETE",
        "arm": args.arm,
        "epoch_window_one_based": [5, 12],
        "checkpoints": [
            {"file": c.name, "sha256": sha256_file(c)} for c in window
        ],
        "terminal_receipt_sha256": sha256_file(terminal_path),
        "modes": results,
        "device": str(device),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_immutable(out_path, payload)
    print(json.dumps({m: round(r["mean_r2"], 4) for m, r in results.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
