#!/usr/bin/env python3
"""Copy best.ckpt into clean-selection training artifacts (train-only runs skip test export)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


SCREEN = "m1_clean_selection_v1"
GROUPS = ("f0", "t4", "ts4")
CELLS = {"fold1_seed42": (1, 42), "fold1_seed43": (1, 43), "fold2_seed42": (2, 42)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def best_checkpoint_from_run(run_dir: Path) -> tuple[Path, float]:
    best_path = None
    best_score = None
    for ckpt in sorted((run_dir / "checkpoints/best_ckpt").glob("epoch_*.ckpt")):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        for value in state.get("callbacks", {}).values():
            if not isinstance(value, dict):
                continue
            if value.get("monitor") != "val_heldin/r2_mean":
                continue
            path = value.get("best_model_path")
            score = value.get("best_model_score")
            if path and score is not None:
                candidate = Path(path)
                if candidate.is_file():
                    score_f = float(score.item() if hasattr(score, "item") else score)
                    if best_score is None or score_f >= best_score:
                        best_score = score_f
                        best_path = candidate
    if best_path is None:
        raise ValueError(f"no best checkpoint found under {run_dir}")
    return best_path, float(best_score)


def find_run_dir(sce: Path, run_id: str, fold: int, seed: int) -> Path:
    matches = sorted(sce.glob(f"logs/train/runs/*_rid-{run_id}_f{fold}_s{seed}"))
    if not matches:
        raise ValueError(f"missing lightning run for {run_id} f{fold} s{seed}")
    return matches[-1]


def find_artifact(sce: Path, run_id: str, fold: int, seed: int) -> Path:
    matches = sorted(sce.glob(f"outputs/streaming_calibration/{run_id}_f{fold}_s{seed}_*"))
    if len(matches) != 1:
        raise ValueError(f"expected one artifact for {run_id} f{fold} s{seed}, found {len(matches)}")
    return matches[0]


def finalize_one(sce: Path, group: str, cell: str) -> dict:
    fold, seed = CELLS[cell]
    run_id = f"{SCREEN}_{group}_m1"
    artifact = find_artifact(sce, run_id, fold, seed)
    run_dir = find_run_dir(sce, run_id, fold, seed)
    best_src, best_score = best_checkpoint_from_run(run_dir)
    last_src = run_dir / "checkpoints/best_ckpt/last.ckpt"
    if not last_src.is_file():
        raise ValueError(f"missing last.ckpt: {last_src}")
    ckpt_dir = artifact / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dst = ckpt_dir / "best.ckpt"
    last_dst = ckpt_dir / "last.ckpt"
    best_dst.write_bytes(best_src.read_bytes())
    if not last_dst.is_file():
        last_dst.write_bytes(last_src.read_bytes())
    manifest = {
        "source_best_checkpoint_path": str(best_src.resolve()),
        "source_best_checkpoint_sha256": sha256(best_src),
        "selected_by_metric": "val_heldin/r2_mean",
        "selected_metric_value": best_score,
        "artifact_best_checkpoint_path": str(best_dst.resolve()),
        "artifact_best_checkpoint_sha256": sha256(best_dst),
        "artifact_last_checkpoint_path": str(last_dst.resolve()),
        "artifact_last_checkpoint_sha256": sha256(last_dst),
        "selection_window": "[10, 210)",
    }
    (artifact / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "group": group,
        "cell": cell,
        "artifact": str(artifact),
        "best_score": best_score,
        "best_sha256": manifest["artifact_best_checkpoint_sha256"],
        "last_sha256": manifest["artifact_last_checkpoint_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    sce = args.root.resolve() / "streaming_calibration_exp"
    results = []
    for group in GROUPS:
        for cell in CELLS:
            results.append(finalize_one(sce, group, cell))
    out = args.root.resolve() / f"sua_exploration/results/{SCREEN}/training_finalize.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(out)
    print(sha256(out))


if __name__ == "__main__":
    main()
