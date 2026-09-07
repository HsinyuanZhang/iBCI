"""Stage-1 source-cell runner for the large TFPD rung (TaskFrameSetAttentionDecoder).

Mirrors run_stage1_source_cell.py's contract but binds only files outside the
running cell closures: this script + src/tfpd_large.py.  Cells: large_t4,
large_z4.
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
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(ROOT))

BOUND_PATTERNS = ("src/tfpd_large.py", "scripts/run_stage1_large_cell.py")
PAD_VALUE = -1.0


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


def build_lit_module():
    import lightning.pytorch as pl
    import torch
    from torchmetrics.regression import R2Score
    from src.tfpd_large import TaskFrameSetAttentionDecoder

    class LargeTFPDLitModule(pl.LightningModule):
        def __init__(self, arm: str, lr: float = 1e-4, seed: int = 42) -> None:
            super().__init__()
            self.save_hyperparameters(logger=False)
            torch.manual_seed(seed)
            self.model = TaskFrameSetAttentionDecoder()
            self.val_r2 = R2Score(multioutput="variance_weighted")

        def _shared_step(self, batch):
            neural, behavior, _calib, _session, side, *_ = batch
            if side.shape[-1] != 4:
                raise ValueError(f"expected 4-wide side features, got {side.shape[-1]}")
            prediction = self.model(neural, side)
            valid = (behavior != PAD_VALUE).all(dim=-1)
            if not valid.any():
                return None, prediction, behavior, valid
            diff2 = ((prediction - behavior) ** 2).sum(dim=-1)
            loss = (diff2 * valid).sum() / (valid.sum() * behavior.shape[-1])
            return loss, prediction, behavior, valid

        def training_step(self, batch, batch_idx):
            loss, _, _, _ = self._shared_step(batch)
            if loss is None:
                return None
            self.log("train/loss", loss, on_step=True)
            return loss

        def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
            # Predeclared for the 5M attention rung: first-step grad norm ~122
            # without clipping lands the model in a dead plateau.
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        def validation_step(self, batch, batch_idx, dataloader_idx=0):
            loss, prediction, behavior, valid = self._shared_step(batch)
            if loss is None:
                return None
            self.log("val/loss", loss, prog_bar=True)
            mask = valid.unsqueeze(-1).expand_as(prediction)
            self.val_r2.update(prediction[mask], behavior[mask])
            return loss

        def on_validation_epoch_end(self):
            self.log("val/r2_mean", self.val_r2.compute(), prog_bar=True)
            self.val_r2.reset()

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    return LargeTFPDLitModule


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["large_t4", "large_z4"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/stage1_source_cells_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=12)
    args = parser.parse_args()

    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint

    from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule

    if not sys.flags.no_user_site:
        raise SystemExit("PYTHONNOUSERSITE=1 is mandatory")

    carrier_group = args.arm.split("_")[1]
    out_dir = args.output_root / args.arm
    if out_dir.exists():
        raise SystemExit(f"fresh cell output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)

    import mc_maze.a2_matched_subject_shift_v2_core as a2

    closure = source_closure()
    launch = {
        "schema": "tfpd_stage1_large_cell_v1",
        "status": "LAUNCHED",
        "arm": args.arm,
        "model_name": "TaskFrameSetAttentionDecoder_5M",
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
    module = build_lit_module()(arm=carrier_group, lr=1e-4, seed=args.seed)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(out_dir / "epoch_ckpts"), every_n_epochs=1, save_top_k=-1
    )
    trainer = pl.Trainer(
        accelerator="gpu" if args.device.startswith("cuda") else "cpu",
        devices=[int(args.device.split(":")[1])] if args.device.startswith("cuda") else 1,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb],
        logger=False,
        enable_progress_bar=False,
        num_sanity_val_steps=2,
    )
    trainer.fit(module, datamodule=dm)

    final_closure = source_closure()
    ckpts = sorted((out_dir / "epoch_ckpts").glob("*.ckpt"))
    terminal = {
        **launch,
        "status": "SOURCE_CELL_TERMINAL" if len(ckpts) == args.max_epochs else "SOURCE_CELL_INCOMPLETE",
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "final_closure": final_closure,
        "launch_final_closure_equal": final_closure["closure_sha256"] == closure["closure_sha256"],
        "epochs_run": len(ckpts),
        "final_val_r2": (
            float(trainer.callback_metrics.get("val/r2_mean"))
            if "val/r2_mean" in trainer.callback_metrics
            else None
        ),
    }
    write_immutable(out_dir / "terminal_receipt.json", terminal)
    print(json.dumps({"arm": args.arm, "status": terminal["status"], "epochs": terminal["epochs_run"],
                      "val_r2": terminal["final_val_r2"]}, indent=1))
    return 0 if terminal["launch_final_closure_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
