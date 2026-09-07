"""Stage-1 source-cell runner for TFPD arms.

Reuses the audited strict-27/M30 datamodule configuration read-only (identical
to the sealed swap-v2 successor cells) and trains one TFPD arm from standard
initialization, task-only.  Writes immutable launch/terminal receipts with a
source-file closure, mirroring the Stage-0 receipt discipline.

Cells: --arm {bl_t4, bl_z4, pv_t4, pv_z4}   (model × carrier group)
       --device cuda:0 | cpu
       --dry-run limits batches and epochs for the CPU shape smoke.

Usage:
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 python scripts/run_stage1_source_cell.py \
        --arm bl_t4 --device cuda:0 --output-root results/stage1_source_cells_v1
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

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
# Order matters: streaming_calibration_exp also owns a `src` package, so the
# package root must precede it on sys.path.
sys.path.insert(0, str(REPO / "sua_exploration"))  # mc_maze.* read-only reuse
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))
sys.path.insert(0, str(ROOT))  # src.tfpd resolves here, first

BOUND_PATTERNS = (
    "src/tfpd/*.py",
    "scripts/run_stage1_source_cell.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_closure() -> dict:
    files = {}
    for pattern in BOUND_PATTERNS:
        for path in sorted(ROOT.glob(pattern)):
            files[str(path.relative_to(ROOT))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    closure = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    return {"files": files, "closure_sha256": closure}


def write_immutable(path: Path, payload: dict) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing receipt {path}")
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(sha256_file(path) + "  " + path.name + "\n")
    os.chmod(sidecar, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


MODEL_BY_PREFIX = {"bl": "bilinear", "pv": "population_vector"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["bl_t4", "bl_z4", "pv_t4", "pv_z4"])
    parser.add_argument("--device", default="cuda:0", choices=["cuda:0", "cuda:1", "cpu"])
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/stage1_source_cells_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint

    from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule
    from src.tfpd.stage1_module import TFPDSourceLitModule

    if not sys.flags.no_user_site:
        raise SystemExit("PYTHONNOUSERSITE=1 is mandatory")
    if args.device != "cpu" and args.dry_run:
        raise SystemExit("dry-run must use --device cpu")

    model_prefix, carrier_group = args.arm.split("_")
    model_name = MODEL_BY_PREFIX[model_prefix]
    out_dir = args.output_root / args.arm
    if out_dir.exists():
        raise SystemExit(f"fresh cell output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)

    import mc_maze.a2_matched_subject_shift_v2_core as a2

    closure = source_closure()
    launch = {
        "schema": "tfpd_stage1_source_cell_v1",
        "status": "DRY_RUN" if args.dry_run else "LAUNCHED",
        "arm": args.arm,
        "model_name": model_name,
        "carrier_group": carrier_group,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "device": args.device,
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "source_closure": closure,
        "datamodule_contract": {
            "data_dir": str(a2.SUBC_DATA_ROOT),
            "split_counts": [27, 6, 6],
            "window_size": 50,
            "calibration_n_trials": 30,
            "random_calibration": False,
            "max_units_exclusive": 100,
            "signal_view": "sua",
            "side_feature_group": carrier_group,
            "side_feature_pool_size": 30,
            "train_val_manifest_path": str(a2.MANIFEST_PATH),
            "expected_manifest_sha256": a2.EXPECTED_MANIFEST_SHA256,
        },
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_immutable(out_dir / "launch_receipt.json", launch)

    pl.seed_everything(args.seed, workers=True)
    dm = Dandi688MultiSessionDataModule(
        data_dir=str(a2.SUBC_DATA_ROOT), task="CO", split_counts=(27, 6, 6), batch_size=32,
        window_size=50, calibration_n_trials=30, max_trial_length=100, bin_size_ms=20,
        num_workers=4, random_calibration=False, seed=args.seed, max_units_exclusive=100,
        cache_dir=str(a2.SOURCE_CACHE_ROOT), signal_view="sua",
        side_feature_group=carrier_group, side_feature_pool_size=30,
        train_val_manifest_path=str(a2.MANIFEST_PATH),
    )
    dm.setup("fit")

    module = TFPDSourceLitModule(model_name=model_name, arm=carrier_group, lr=1e-4, seed=args.seed)

    limit = {"limit_train_batches": 3, "limit_val_batches": 2, "max_epochs": 1} if args.dry_run else {}
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(out_dir / "epoch_ckpts"), every_n_epochs=1, save_top_k=-1
    )
    trainer = pl.Trainer(
        accelerator="gpu" if args.device.startswith("cuda") else "cpu",
        devices=[int(args.device.split(":")[1])] if args.device.startswith("cuda") else 1,
        max_epochs=limit.get("max_epochs", args.max_epochs),
        limit_train_batches=limit.get("limit_train_batches"),
        limit_val_batches=limit.get("limit_val_batches"),
        callbacks=[checkpoint_cb],
        logger=False,
        enable_progress_bar=False,
        num_sanity_val_steps=2,
    )
    trainer.fit(module, datamodule=dm)

    final_closure = source_closure()
    terminal = {
        **launch,
        "status": "DRY_RUN_COMPLETE" if args.dry_run else "SOURCE_CELL_TERMINAL",
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "final_closure": final_closure,
        "launch_final_closure_equal": final_closure["closure_sha256"] == closure["closure_sha256"],
        "epochs_run": trainer.current_epoch + 1,
        # The val metric is reset at each epoch end, so read the last logged
        # value instead of recomputing (torchmetrics raises on empty state).
        "final_val_r2": (
            float(trainer.callback_metrics.get("val/r2_mean"))
            if not args.dry_run and "val/r2_mean" in trainer.callback_metrics
            else None
        ),
    }
    write_immutable(out_dir / "terminal_receipt.json", terminal)
    print(json.dumps({"arm": args.arm, "status": terminal["status"], "epochs": terminal["epochs_run"],
                      "val_r2": terminal["final_val_r2"], "closure_equal": terminal["launch_final_closure_equal"]}, indent=1))
    return 0 if terminal["launch_final_closure_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
