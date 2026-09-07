#!/usr/bin/env python3
"""Run the exact production Stage-P sanity path on source/dev data only.

This is a successor-regression gate, not a result producer.  It exercises the
real `MisleadingIdentitySwapV2LitModule` through `lightning.Trainer.fit` for
all four predeclared source cells, including Lightning's pre-fit validation
sanity loop and one epoch-zero train/validation boundary.  It creates no
checkpoint or receipt, uses CPU only, and never resolves target or formal
sub-C data.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
SCE_ROOT = REPO_ROOT / "streaming_calibration_exp"
for entry in (SUA_ROOT, SCE_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from mc_maze import a2_matched_subject_shift_v2_core as a2
from mc_maze import misleading_identity_swap_v2_core as core
from scripts import misleading_identity_swap_v2_train_cell as train_cell


def _factors(cell: str) -> tuple[str, str]:
    core.require(cell in core.CELLS, "source sanity cell drift")
    identity, carrier = cell.split("_", 1)
    return identity, carrier


def _datamodule(*, carrier: str):
    from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule

    return Dandi688MultiSessionDataModule(
        data_dir=str(a2.SUBC_DATA_ROOT), task="CO", split_counts=(27, 6, 6),
        batch_size=32, window_size=50, calibration_n_trials=30,
        max_trial_length=100, bin_size_ms=20, num_workers=0,
        random_calibration=False, seed=42, max_units_exclusive=100,
        cache_dir=str(a2.SOURCE_CACHE_ROOT), signal_view="sua",
        side_feature_group=carrier, side_feature_pool_size=30,
        train_val_manifest_path=str(a2.MANIFEST_PATH),
    )


def run_one_cell(*, cell: str, authority: Path, initial_state: Path) -> dict[str, Any]:
    """Fit one batch through epoch zero and validate source/dev only."""
    import lightning.pytorch as pl
    from mc_maze.misleading_identity_swap_v2_trainer import MisleadingIdentitySwapV2LitModule
    from scripts.train_variant_dandi688 import configure_multisession_metrics

    identity, carrier = _factors(cell)
    pl.seed_everything(42, workers=True)
    dm = _datamodule(carrier=carrier)
    dm.setup("fit")
    core.require(tuple(dm.session_splits["train"]) == tuple(a2.load_strict_manifest()["train"]),
                 f"{cell}: strict27 source roster drift")
    core.require(tuple(dm.session_splits["test"]) == tuple(a2.load_strict_manifest()["test"]),
                 f"{cell}: formal roster name drift")
    model = train_cell._model(authority, identity)
    core.require(isinstance(model, MisleadingIdentitySwapV2LitModule),
                 f"{cell}: production wrapper substitution")
    model.setup("fit")
    initial, initial_sha = train_cell._load_initial(initial_state)
    core.require(initial.get("matching_authority_sha256") == core.load_verified_authority(authority).sha256,
                 f"{cell}: initial/source authority drift")
    model.load_state_dict(initial["state_dict"], strict=True)
    core.require(train_cell._state_sha(model.state_dict()) == initial.get("state_dict_sha256"),
                 f"{cell}: strict shared initial state drift")
    configure_multisession_metrics(model, dm)
    with tempfile.TemporaryDirectory(prefix=f"swap_v2_{cell}_source_sanity_") as scratch:
        trainer = pl.Trainer(
            max_epochs=1, accelerator="cpu", devices=1,
            logger=False, enable_checkpointing=False, enable_model_summary=False,
            deterministic=True, num_sanity_val_steps=1,
            limit_train_batches=1, limit_val_batches=1,
            default_root_dir=scratch, enable_progress_bar=False,
        )
        trainer.fit(model, datamodule=dm)
        core.require(trainer.global_step >= 1, f"{cell}: did not reach an epoch-zero train batch")
        receipt = model.intervention_receipt()
    trace = receipt.get("last_swap_trace")
    core.require(isinstance(trace, dict) and trace.get("phase") == "eval_clean" and
                 trace.get("swap_applied") is False,
                 f"{cell}: source sanity/validation did not finish in clean eval")
    return {
        "cell": cell,
        "source_train_session_count": len(dm.session_splits["train"]),
        "development_validation_session_count": len(dm.session_splits["val"]),
        "formal_test_session_names_only": list(dm.session_splits["test"]),
        "formal_subc_test_nwb_opened": False,
        "target_nwb_opened": False,
        "target_authority_opened": False,
        "gpu_used": False,
        "epoch_zero_train_batch_completed": True,
        "lightning_sanity_validation_completed_clean": True,
        "last_validation_trace": trace,
        "initial_state_file_sha256": initial_sha,
    }


def execute() -> dict[str, Any]:
    core.require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in {"", "-1"},
                 "source sanity regression must run CPU-only with CUDA hidden")
    authority = core.SOURCE_AUTHORITY_PATH
    initial = core.INITIAL_STATE_PATH
    core.load_verified_authority(authority)
    train_cell._load_initial(initial)
    rows = [run_one_cell(cell=cell, authority=authority, initial_state=initial) for cell in core.CELLS]
    return {
        "status": "SOURCE_ONLY_PRODUCTION_TRAINER_FIT_SANITY_PASS__NOT_A_RECEIPT",
        "cells": rows,
        "cell_order": list(core.CELLS),
        "target_nwb_opened": False,
        "formal_subc_test_nwb_opened": False,
        "checkpoint_or_result_published": False,
        "gpu_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-source-only", action="store_true")
    args = parser.parse_args()
    if not args.execute_source_only:
        print(json.dumps({
            "status": "DRY_RUN__SOURCE_ONLY_PRODUCTION_TRAINER_FIT_SANITY_REQUIRED",
            "cells": list(core.CELLS), "target_nwb_opened": False,
            "formal_subc_test_nwb_opened": False, "gpu_used": False,
        }, indent=2, sort_keys=True))
        return 0
    print(json.dumps(execute(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
