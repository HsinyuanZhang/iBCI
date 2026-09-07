#!/usr/bin/env python3
"""CPU-only launch receipt for the Stage-R RT B2/LatePool references.

This preflight never opens an NWB, initializes a teacher checkpoint, touches
CUDA, or launches a Trainer.  It composes the exact B2 clean-nested config,
asserts the split/selection/deployment contract, and records encoder cost for
the intended first fold.  It prepares R-S (D1024) and R-S128 (D128) without
authorizing either GPU run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ENTRY = PROJECT_ROOT / "src" / "train.py"
CONFIG_NAME = "rt_clean_nested_loso_b2_stage_r"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compose(*, hidden_dim: int, fold: int, seed: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(TRAIN_ENTRY),
        f"experiment={CONFIG_NAME}",
        f"data.loso_fold={fold}",
        f"data.outer_loso_fold={fold}",
        f"model.id_hidden_dim={hidden_dim}",
        f"seed={seed}",
        "test=false",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "--cfg",
        "job",
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            "Stage-R B2 Hydra composition failed:\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    cfg = OmegaConf.create(completed.stdout)
    values = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(values, dict):
        raise ValueError("Hydra did not emit a mapping config")
    # ``--cfg job`` correctly preserves Hydra-runtime bookkeeping
    # interpolations (which cannot be resolved outside a Hydra job), while
    # retaining local data aliases such as ``${.calibration_n_trials}``.
    # Resolve only the latter through the live DictConfig before auditing.
    data_values = values.get("data")
    if not isinstance(data_values, dict):
        raise ValueError("Hydra config has no data mapping")
    data_values["calibration_n_trials"] = int(cfg.data.calibration_n_trials)
    data_values["query_start_trial"] = int(cfg.data.query_start_trial)
    data_values["outer_loso_fold"] = int(cfg.data.outer_loso_fold)
    return values


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _encoder_cost(hidden_dim: int) -> dict[str, int | str | bool]:
    # Import after config composition to keep a malformed config from looking
    # like a valid launch merely because its Python classes can be imported.
    from src.models.components.streaming_encoders import LatePoolEncoder

    encoder = LatePoolEncoder(trial_length=100, window_size=50, id_hidden_dim=hidden_dim)
    profile = encoder.cost_profile(num_neurons=96, trial_length=100, num_trials=24)
    return {
        "variant": str(profile.variant),
        "id_hidden_dim": int(hidden_dim),
        "parameter_count": int(profile.parameter_count),
        "weight_bytes": int(profile.weight_bytes),
        "trial_buffer_bytes": int(profile.trial_buffer_bytes),
        "support_state_bytes": int(profile.support_state_bytes),
        "peak_live_state_bytes": int(profile.peak_live_state_bytes),
        "mac_per_trial": int(profile.mac_per_trial),
        "mac_per_session": int(profile.mac_per_session),
        "requires_cubic_interpolation": bool(profile.requires_cubic_interpolation),
        "requires_general_multiplier": bool(profile.requires_general_multiplier),
    }


def _first_cell_command(*, run_root: Path, hidden_dim: int, fold: int, seed: int) -> list[str]:
    run_id = f"rt_clean_nested_loso_m24_b2_d{hidden_dim}"
    fit_dir = run_root / f"b2_d{hidden_dim}" / f"fold_{fold:02d}" / f"seed_{seed}" / "fit"
    return [
        sys.executable,
        str(TRAIN_ENTRY),
        f"experiment={CONFIG_NAME}",
        f"run_id={run_id}",
        f"data.loso_fold={fold}",
        f"data.outer_loso_fold={fold}",
        f"model.id_hidden_dim={hidden_dim}",
        f"seed={seed}",
        "test=false",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        f"hydra.run.dir={fit_dir}",
        f"paths.log_dir={run_root / '_hydra_logs'}",
        f"paths.artifact_dir={run_root / '_artifacts'}",
    ]


def _audit_config(cfg: dict[str, Any], *, hidden_dim: int, fold: int, seed: int) -> None:
    data = cfg.get("data")
    model = cfg.get("model")
    trainer = cfg.get("trainer")
    callbacks = cfg.get("callbacks")
    _require(isinstance(data, dict) and isinstance(model, dict), "B2 Stage-R config lacks data/model mapping")
    _require(data.get("_target_") == "src.data.rt_nested_loso_datamodule.RtNestedLossoDataModule", "Stage-R must use clean RT nested data module")
    _require(data.get("side_feature_group") == "none", "B2 Stage-R must provide no carrier/side feature")
    _require(int(data.get("loso_fold", -1)) == fold and int(data.get("outer_loso_fold", -1)) == fold, "Stage-R outer-fold binding failed")
    _require(int(data.get("calibration_n_trials", -1)) == 24 and int(data.get("query_start_trial", -1)) == 24, "Stage-R must use chronological M24/q24")
    _require(bool(cfg.get("train")) and not bool(cfg.get("test")), "Stage-R fit must be train=true/test=false")
    _require(model.get("variant") == "B2" and int(model.get("id_hidden_dim", -1)) == hidden_dim, "Stage-R B2 width binding failed")
    _require(model.get("freeze_decoder") is False and model.get("loss_mode") == "task_only", "Stage-R must match R-C joint decoder/task-only schedule")
    _require(isinstance(trainer, dict) and int(trainer.get("max_epochs", -1)) == 35, "Stage-R must use R-C 35 epochs")
    _require(int(cfg.get("seed", -1)) == seed, "Stage-R seed binding failed")
    _require(isinstance(callbacks, dict) and "rt_nested_selection_receipt" in callbacks, "Stage-R needs a fit selection receipt")
    receipt = callbacks["rt_nested_selection_receipt"]
    _require(isinstance(receipt, dict) and receipt.get("monitor") == "val_heldin/r2_mean", "Stage-R must select only on inner validation R2")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-d128", action="store_true")
    args = parser.parse_args()
    if not 0 <= int(args.fold) < 15:
        raise ValueError("--fold must be in [0,14]")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Stage-R B2 preflight: {output}")
    widths = [1024] + ([128] if args.include_d128 else [])
    arms: dict[str, Any] = {}
    for width in widths:
        cfg = _compose(hidden_dim=width, fold=int(args.fold), seed=int(args.seed))
        _audit_config(cfg, hidden_dim=width, fold=int(args.fold), seed=int(args.seed))
        command = _first_cell_command(
            run_root=args.run_root.resolve(), hidden_dim=width, fold=int(args.fold), seed=int(args.seed)
        )
        arms[f"R-S{'128' if width == 128 else ''}"] = {
            "id_hidden_dim": width,
            "cost": _encoder_cost(width),
            "first_cell_train_command": command,
            "formal_heldout": False,
            "outer_target_opener": "src/rt_clean_nested_loso_eval.py after passing selection receipt only",
        }
    payload = {
        "schema": "rt_stage_r_b2_cpu_preflight_v1",
        "status": "READY_NOT_LAUNCHED",
        "purpose": "Stage-R B2/LatePool clean nested-LOSO first-cell preparation only",
        "nwb_opened": False,
        "cuda_touched": False,
        "trainer_launched": False,
        "formal_heldout_opened": False,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "config_path": str((PROJECT_ROOT / "configs" / "experiment" / f"{CONFIG_NAME}.yaml").resolve()),
        "config_sha256": _sha256(PROJECT_ROOT / "configs" / "experiment" / f"{CONFIG_NAME}.yaml"),
        "data_module_path": str((PROJECT_ROOT / "src" / "data" / "rt_nested_loso_datamodule.py").resolve()),
        "data_module_sha256": _sha256(PROJECT_ROOT / "src" / "data" / "rt_nested_loso_datamodule.py"),
        "outer_evaluator_path": str((PROJECT_ROOT / "src" / "rt_clean_nested_loso_eval.py").resolve()),
        "outer_evaluator_sha256": _sha256(PROJECT_ROOT / "src" / "rt_clean_nested_loso_eval.py"),
        "arms": arms,
        "comparability": {
            "same_clean_nested_loso_partition": True,
            "same_chronological_support_trials": 24,
            "same_query_start_trial": 24,
            "same_inner_selection_metric": "val_heldin/r2_mean",
            "same_epoch_budget": 35,
            "same_joint_decoder_training": True,
            "only_intended_mechanism_change": "B2 LatePool identity encoder replaces B3S carrier encoder",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
