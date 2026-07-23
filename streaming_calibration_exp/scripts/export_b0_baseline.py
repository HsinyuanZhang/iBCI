"""Export matched B0 teacher baseline metrics to an immutable artifact run."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import lightning as L
import rootutils
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.run_artifacts import (
    METRICS_PER_SESSION_FIELDS,
    METRICS_SUMMARY_FIELDS,
    build_test_metric_rows,
    checkpoint_sha256,
    parse_test_metrics,
    write_checkpoint_manifest,
    write_environment,
    write_git_state,
    write_metrics_table,
    write_resolved_config,
    write_source_manifest,
    write_split_manifest,
)
from src.models.components.streaming_encoders import EncoderCostProfile
from src.models.falcon_module import FalconLitModule


B0_FULL_SUPPORT_CACHE_BYTES = 33 * 100 * 96 * 4
B0_MAC_PER_TRIAL = 96 * (100 * 512 + 2 * 512 * 512)
B0_MAC_PER_SESSION = 1_875_935_232


def _b0_profile(id_params: int) -> EncoderCostProfile:
    return EncoderCostProfile(
        parameter_count=id_params,
        weight_bytes=id_params * 4,
        trial_buffer_bytes=B0_FULL_SUPPORT_CACHE_BYTES,
        support_state_bytes=0,
        peak_live_state_bytes=B0_FULL_SUPPORT_CACHE_BYTES,
        mac_per_trial=B0_MAC_PER_TRIAL,
        mac_per_session=B0_MAC_PER_SESSION,
        requires_cubic_interpolation=True,
        requires_general_multiplier=True,
        requires_divider=True,
        variant="B0",
    )


def _write_effective_eval_config(run_dir: Path, cfg, teacher_ckpt: Path) -> None:
    payload = {
        "model_target": "src.models.falcon_module.FalconLitModule",
        "variant": "B0",
        "teacher_checkpoint_path": str(teacher_ckpt.resolve()),
        "teacher_checkpoint_sha256": checkpoint_sha256(teacher_ckpt),
        "seed": int(cfg.seed),
        "calibration_trials": int(cfg.data.calibration_n_trials),
        "data_preprocessing": OmegaConf.to_container(cfg.data, resolve=True),
        "include_heldout_in_test": bool(cfg.data.include_heldout_in_test),
        "trainer": OmegaConf.to_container(cfg.trainer, resolve=True),
    }
    (run_dir / "effective_eval_config.json").write_text(json.dumps(payload, indent=2) + "\n")


def export_b0(teacher_ckpt: Path, run_dir: Path, seed: int = 42) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(config_name="eval_b0.yaml", overrides=[f"seed={seed}", f"run_id={run_dir.name}"])

    write_resolved_config(run_dir, cfg)
    _write_effective_eval_config(run_dir, cfg, teacher_ckpt)
    write_environment(run_dir)
    write_git_state(run_dir, Path("."))
    write_source_manifest(run_dir, Path("."))
    write_split_manifest(
        run_dir,
        {
            "validation_protocol": "minival",
            "fold_id": None,
            "train_sessions": [],
            "validation_sessions": [],
            "heldout_evaluated_in_fit": False,
            "heldout_evaluated_in_test": True,
        },
    )

    L.seed_everything(seed, workers=True)
    datamodule = instantiate(cfg.data)
    model = FalconLitModule.load_from_checkpoint(str(teacher_ckpt), weights_only=False)
    model.eval()

    trainer = instantiate(cfg.trainer)
    trainer.test(model=model, datamodule=datamodule, weights_only=False)

    module = model.net
    id_params = sum(p.numel() for name, p in module.named_parameters() if "fc_id" in name)
    profile = _b0_profile(id_params)
    parsed = parse_test_metrics(trainer.callback_metrics)
    summary_rows, per_session_rows = build_test_metric_rows(
        run_id=run_dir.name,
        variant="B0",
        seed=seed,
        calibration_trials=int(cfg.data.calibration_n_trials),
        parsed=parsed,
        profile=profile,
        baseline={"heldin": {}, "heldout": {}},
        validation_protocol="minival",
        fold_id=None,
    )
    write_metrics_table(run_dir, summary_rows, "metrics_summary.csv", METRICS_SUMMARY_FIELDS)
    write_metrics_table(run_dir, per_session_rows, "metrics_per_session.csv", METRICS_PER_SESSION_FIELDS)
    write_checkpoint_manifest(
        run_dir,
        teacher_ckpt,
        selected_by_metric="teacher_epoch_034",
        selected_metric_value=None,
        copy_checkpoint=True,
    )

    hardware = {
        "variant": "B0",
        "parameter_count": id_params,
        "weight_bytes": id_params * 4,
        "trial_buffer_bytes": B0_FULL_SUPPORT_CACHE_BYTES,
        "support_state_bytes": 0,
        "full_support_cache_bytes": B0_FULL_SUPPORT_CACHE_BYTES,
        "peak_live_state_bytes": B0_FULL_SUPPORT_CACHE_BYTES,
        "mac_per_trial": B0_MAC_PER_TRIAL,
        "mac_per_session": B0_MAC_PER_SESSION,
        "cost_source": "exact_source_formula",
        "note": "B0 stores the full M*T*N support cache in trial_buffer_bytes; support_state_bytes=0.",
    }
    (run_dir / "hardware_cost.json").write_text(json.dumps(hardware, indent=2) + "\n")

    gate0 = {
        "teacher_checkpoint": str(teacher_ckpt.resolve()),
        "teacher_sha256": checkpoint_sha256(teacher_ckpt),
        "id_encoder_parameters": id_params,
        "total_parameters": sum(p.numel() for p in module.parameters()),
        "baseline_metrics_dir": str(run_dir.resolve()),
        "metrics_per_session_csv": str((run_dir / "metrics_per_session.csv").resolve()),
        "test_heldin_r2_mean": parsed.heldin_mean_r2,
        "test_heldout_r2_mean": parsed.heldout_mean_r2,
    }
    (run_dir / "gate0_baseline.json").write_text(json.dumps(gate0, indent=2) + "\n")
    return gate0


def promote_to_canonical(run_dir: Path, canonical_dir: Path) -> None:
    canonical_dir.parent.mkdir(parents=True, exist_ok=True)
    if canonical_dir.exists():
        shutil.rmtree(canonical_dir)
    shutil.copytree(run_dir, canonical_dir)
    pointer = {
        "canonical_dir": str(canonical_dir.resolve()),
        "source_run_dir": str(run_dir.resolve()),
        "promoted_at_utc": datetime.utcnow().isoformat() + "Z",
        "metrics_per_session_csv": str((canonical_dir / "metrics_per_session.csv").resolve()),
    }
    (canonical_dir.parent / "b0_baseline_pointer.json").write_text(json.dumps(pointer, indent=2) + "\n")
    legacy_gate0 = canonical_dir.parent / "gate0_baseline.json"
    legacy_gate0.write_text((canonical_dir / "gate0_baseline.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher-ckpt",
        type=Path,
        default=Path("../SPINT-main/logs/train/runs/2026-07-07-16-05-16/checkpoints/best_ckpt/epoch_034.ckpt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Immutable run directory. Defaults to outputs/streaming_calibration/b0_baseline_runs/<run_id>",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--promote-to-canonical",
        action="store_true",
        help="Copy this run to outputs/streaming_calibration/b0_baseline after export.",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"b0_s{args.seed}_{stamp}"
    run_dir = args.output_dir or Path("outputs/streaming_calibration/b0_baseline_runs") / run_id

    gate0 = export_b0(args.teacher_ckpt.resolve(), run_dir, seed=args.seed)
    if args.promote_to_canonical:
        promote_to_canonical(run_dir, Path("outputs/streaming_calibration/b0_baseline"))
    print(json.dumps(gate0, indent=2))
    print(f"Immutable B0 run: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
