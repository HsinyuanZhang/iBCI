#!/usr/bin/env python3
"""CPU-only constructibility receipt for remote-compatible RT Stage-R B2.

The original D128 command used ``side_feature_group=none``.  The 5070Ti
``RtNestedLossoDataModule`` deliberately accepts only named arms, so that run
terminated at DataModule construction before a Trainer, NWB file, teacher
checkpoint, CUDA context, or outer target was touched.  This additive preflight
uses the module's documented ``zero4`` arm instead.  ``zero4`` is an exact
all-zero width-matched tensor and B2 has ``side_dim=0``; no carrier reaches the
LatePool encoder.

Unlike the historical compositional preflight, this script actually creates
the DataModule and Lightning module.  It intentionally does *not* call
``setup()``, construct a Trainer, fetch a batch, initialize the teacher, or
open a dataset/checkpoint.  Its output is a launch receipt, never a GPU launch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "rt_clean_nested_loso_b2_stage_r_zero4"
TRAIN_ENTRY = PROJECT_ROOT / "src" / "train.py"
ORIGINAL_FAILED_LOG_RELATIVE = Path(
    "outputs/rt_stage_r_b2_remote/launch_logs/d128_f0_s42.log"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _compose(*, hidden_dim: int, fold: int, seed: int):
    """Compose an exact job config without running the training entry point."""

    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                f"experiment={CONFIG_NAME}",
                f"data.loso_fold={fold}",
                f"data.outer_loso_fold={fold}",
                f"model.id_hidden_dim={hidden_dim}",
                f"seed={seed}",
                "test=false",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
            ],
        )
    return cfg


def _encoder_cost(hidden_dim: int) -> dict[str, int | str | bool]:
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
    run_id = f"rt_clean_nested_loso_m24_b2_d{hidden_dim}_zero4"
    fit_dir = run_root / f"b2_d{hidden_dim}_zero4" / f"fold_{fold:02d}" / f"seed_{seed}" / "fit"
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


def _audit_config(cfg: Any, *, hidden_dim: int, fold: int, seed: int) -> None:
    _require(
        cfg.data._target_ == "src.data.rt_nested_loso_datamodule.RtNestedLossoDataModule",
        "Stage-R must use the clean RT nested data module",
    )
    _require(
        str(cfg.data.side_feature_group).lower() == "zero4",
        "remote-compatible B2 alias must use documented zero4 arm",
    )
    _require(
        int(cfg.data.loso_fold) == fold and int(cfg.data.outer_loso_fold) == fold,
        "outer-fold binding failed",
    )
    _require(
        int(cfg.data.calibration_n_trials) == 24 and int(cfg.data.query_start_trial) == 24,
        "Stage-R must use chronological M24/q24",
    )
    _require(bool(cfg.train) and not bool(cfg.test), "fit must be train=true/test=false")
    _require(
        str(cfg.model.variant) == "B2" and int(cfg.model.id_hidden_dim) == hidden_dim,
        "B2 width binding failed",
    )
    # ``side_dim`` is a constructor default rather than an explicit B2 YAML
    # key in this worktree, so read it with a default instead of struct-style
    # attribute lookup.  The instantiated module below independently verifies
    # the effective value.
    _require(int(cfg.model.get("side_dim", 0)) == 0, "B2 no-carrier alias requires side_dim=0")
    _require(
        cfg.model.freeze_decoder is False and cfg.model.loss_mode == "task_only",
        "Stage-R must match R-C joint-decoder/task-only schedule",
    )
    _require(int(cfg.trainer.max_epochs) == 35, "Stage-R must use 35 epochs")
    _require(int(cfg.seed) == seed, "seed binding failed")
    _require(
        "rt_nested_selection_receipt" in cfg.callbacks
        and cfg.callbacks.rt_nested_selection_receipt.monitor == "val_heldin/r2_mean",
        "Stage-R needs an inner-validation selection receipt",
    )


def _construct_without_io(cfg: Any) -> dict[str, Any]:
    """Instantiate exactly the objects reached before train() calls setup()."""

    cuda_before = bool(torch.cuda.is_initialized())
    datamodule = instantiate(cfg.data)
    _require(not bool(datamodule._setup_complete), "preflight must not call DataModule.setup")
    _require(not bool(datamodule.outer_target_loaded), "target cannot be opened in preflight")
    _require(not bool(datamodule.outer_target_query_labels_read), "target labels cannot be read in preflight")
    _require(str(datamodule._feature_group) == "zero4", "DataModule did not bind zero4 alias")

    model = instantiate(cfg.model)
    _require(model.student is None and model.teacher is None, "model setup/I-O occurred during construction")
    _require(int(model._side_dim) == 0, "B2 model unexpectedly consumes side features")
    cuda_after = bool(torch.cuda.is_initialized())
    _require(cuda_after == cuda_before, "preflight unexpectedly initialized CUDA")
    return {
        "datamodule_class": f"{type(datamodule).__module__}.{type(datamodule).__qualname__}",
        "datamodule_setup_called": bool(datamodule._setup_complete),
        "datamodule_feature_group": str(datamodule._feature_group),
        "outer_target_loaded": bool(datamodule.outer_target_loaded),
        "outer_target_query_labels_read": bool(datamodule.outer_target_query_labels_read),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_variant": str(model._variant),
        "model_side_dim": int(model._side_dim),
        "model_teacher_initialized": model.teacher is not None,
        "model_student_initialized": model.student is not None,
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": cuda_after,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-d128", action="store_true")
    args = parser.parse_args()
    _require(0 <= int(args.fold) < 15, "--fold must be in [0,14]")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite B2 zero4 preflight: {output}")
    widths = [1024] + ([128] if args.include_d128 else [])
    arms: dict[str, Any] = {}
    for width in widths:
        cfg = _compose(hidden_dim=width, fold=int(args.fold), seed=int(args.seed))
        _audit_config(cfg, hidden_dim=width, fold=int(args.fold), seed=int(args.seed))
        construction = _construct_without_io(cfg)
        arms[f"R-S{'128' if width == 128 else ''}"] = {
            "id_hidden_dim": width,
            "canonical_mechanism": "B2 LatePool with no side carrier",
            "implementation_arm": "zero4",
            "zero4_semantics": "exact [N,4] zeros; no descriptor fit and no target/velocity label use",
            "b2_side_consumption": "side_dim=0; LatePool encoder ignores the zero4 tensor",
            "cost": _encoder_cost(width),
            "object_construction": construction,
            "first_cell_train_command": _first_cell_command(
                run_root=args.run_root.resolve(), hidden_dim=width, fold=int(args.fold), seed=int(args.seed)
            ),
            "formal_heldout": False,
            "outer_target_opener": "src/rt_clean_nested_loso_eval.py after passing selection receipt only",
        }
    failed_log = PROJECT_ROOT / ORIGINAL_FAILED_LOG_RELATIVE
    _require(failed_log.is_file(), f"preserved failed D128 log is missing: {failed_log}")
    payload = {
        "schema": "rt_stage_r_b2_zero4_constructibility_preflight_v2",
        "status": "READY_NOT_LAUNCHED",
        "purpose": "remote-compatible Stage-R B2/LatePool first-cell preparation only",
        "plumbing_incident": {
            "original_config": "rt_clean_nested_loso_b2_stage_r",
            "original_arm": "none",
            "failure_stage": "DataModule __init__ before Trainer.fit",
            "failure_reason": "remote RtNestedLossoDataModule does not support side_feature_group='none'",
            "result_interpretation": "NO_RESULT: no trainer, source NWB, target NWB, or CUDA context was reached",
            "preserved_log": str(failed_log.resolve()),
            "preserved_log_sha256": _sha256(failed_log),
        },
        "nwb_opened": False,
        "cuda_touched": False,
        "trainer_launched": False,
        "formal_heldout_opened": False,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "config_path": str((PROJECT_ROOT / "configs" / "experiment" / f"{CONFIG_NAME}.yaml").resolve()),
        "config_sha256": _sha256(PROJECT_ROOT / "configs" / "experiment" / f"{CONFIG_NAME}.yaml"),
        "preflight_script_path": str(Path(__file__).resolve()),
        "preflight_script_sha256": _sha256(Path(__file__).resolve()),
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
            "no_descriptor_or_behavioral_label_used_by_b2_side_path": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
