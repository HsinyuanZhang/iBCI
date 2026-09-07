#!/usr/bin/env python3
"""Fail-closed CPU-only single-cell executor for 0f M2 domain/query audit.

This is intentionally not a general evaluation command.  Its sole permitted
job is one frozen F0/T4 M2 fold-1/seed-42 M24 audit cell.  The production
FalconDataModule continues to reserve held-in query exclusion; this script
creates an isolated held-in query dataset after ordinary test-only setup and
does not modify the production path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
MUA_ROOT = ROOT / "streaming_calibration_exp"
if str(MUA_ROOT) not in sys.path:
    sys.path.insert(0, str(MUA_ROOT))

import hydra  # noqa: E402
import lightning as L  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.falcon_datamodule import FalconDataModule, FalconDataset, SessionBatchSampler  # noqa: E402
from src.models.streaming_calibration_module import StreamingCalibrationLitModule  # noqa: E402


RECEIPT = ROOT / "sua_exploration/results/m2_m24_domain_query_sign_audit_v1/protocol_receipt_v2.json"
SOURCE = ROOT / "sua_exploration/results/m2_m24_disjoint_source_v1/aggregate_internal.json"
EXPECTED_RECEIPT_SHA = "fb45c8cdab688919eddb577438c28f6ebdba5e6f9d964d183af58ded62bd9356"
EXPECTED_SOURCE_SHA = "75f335b5671fa4a4d310245c2098593c9f23fcf2e86675eb133cc1a0bd5f040a"
FROZEN_TEACHER = ROOT / "SPINT-main/logs/train/runs/2026-07-07-16-05-16/checkpoints/best_ckpt/epoch_034.ckpt"
EXPECTED_TEACHER_SHA = "fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec"
CONFIG_EXPERIMENT = {"f0": "b3_native_mua_f0_m2_m24_loso_internal", "t4": "b3s_t4_m2_m24_loso_internal"}
AUDIT_MODEL_CONFIG = {"f0": "streaming_b3_m2_m24_source_exact", "t4": "streaming_b3s_t4_m2_m24_source_exact"}
SOURCE_RESOLVED_CONFIG = {
    "f0": ROOT / "streaming_calibration_exp/outputs/streaming_calibration/m2_m24_disjoint_source_v1_f0_m2_f1_s42_20260731_220717/resolved_config.yaml",
    "t4": ROOT / "streaming_calibration_exp/outputs/streaming_calibration/m2_m24_disjoint_source_v1_t4_m2_f1_s42_20260731_220722/resolved_config.yaml",
}
EXPECTED_SOURCE_RESOLVED_CONFIG_SHA = {
    "f0": "5f6627b808c565a981f359579890a10bc703c6eeee96a20fb2cd5583515fd101",
    "t4": "52cc96253d495bb0e7ef8a78afbb49bed05097a5b813cb6f6c590c1ea5c51ad5",
}
BEST = {
    "f0": (ROOT / "streaming_calibration_exp/outputs/streaming_calibration/m2_m24_disjoint_source_v1_f0_m2_f1_s42_20260731_220717/checkpoints/best.ckpt", "da64728868d25476bc728ad00587f06e69298612c39652aafff37666819d6ab5"),
    "t4": (ROOT / "streaming_calibration_exp/outputs/streaming_calibration/m2_m24_disjoint_source_v1_t4_m2_f1_s42_20260731_220722/checkpoints/best.ckpt", "b9be07d02afc92504dca9e1846bfef1f9e9753d5645f1929efd0abf9de93d04f"),
}
EPOCH9 = {
    "f0": (ROOT / "streaming_calibration_exp/logs/train/runs/2026-07-31-22-07-17-816214_rid-m2_m24_disjoint_source_v1_f0_m2_f1_s42/checkpoints/periodic_ckpt/epoch_009.ckpt", "dc7b45bd9f4266c3116ab5df04c3bfcc7f541c71204d9a9ca0e1e0b09f2817c5"),
    "t4": (ROOT / "streaming_calibration_exp/logs/train/runs/2026-07-31-22-07-22-237428_rid-m2_m24_disjoint_source_v1_t4_m2_f1_s42/checkpoints/periodic_ckpt/epoch_009.ckpt", "5d093371581897cefa64e4057440cd5d8bcacf3106bfec55113e9c238672d2c6"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail_closed_preflight(args: argparse.Namespace) -> tuple[Path, str]:
    if args.arm not in CONFIG_EXPERIMENT:
        raise ValueError("arm must be exactly f0 or t4")
    if args.domain not in {"heldin", "heldout"}:
        raise ValueError("domain must be heldin or heldout")
    if args.query_start not in {0, 24}:
        raise ValueError("query_start must be exactly 0 or 24")
    if args.checkpoint_policy not in {"best", "epoch9"}:
        raise ValueError("checkpoint_policy must be best or epoch9")
    if args.fold != 1 or args.seed != 42 or args.calibration_trials != 24 or args.window_size != 50:
        raise ValueError("only frozen M2/fold1/seed42/M24/window50 audit is permitted")
    if args.accelerator != "cpu" or args.devices != 1:
        raise ValueError("this audit permits CPU and one device only")
    if torch.cuda.is_initialized() or torch.cuda.is_available():
        # CUDA availability is intentionally rejected, not merely unused: this executor
        # must never become a path that silently grabs a 3090.
        raise RuntimeError("CUDA must be unavailable/uninitialized for this CPU-only audit")
    if sha256(RECEIPT) != EXPECTED_RECEIPT_SHA or sha256(SOURCE) != EXPECTED_SOURCE_SHA:
        raise RuntimeError("protocol receipt or source aggregate SHA drift")
    if not FROZEN_TEACHER.is_file() or sha256(FROZEN_TEACHER) != EXPECTED_TEACHER_SHA:
        raise RuntimeError("frozen source teacher checkpoint missing or SHA drift")
    source_config = SOURCE_RESOLVED_CONFIG[args.arm]
    if not source_config.is_file() or sha256(source_config) != EXPECTED_SOURCE_RESOLVED_CONFIG_SHA[args.arm]:
        raise RuntimeError("frozen source resolved-config missing or SHA drift")
    table = BEST if args.checkpoint_policy == "best" else EPOCH9
    checkpoint, expected_sha = table[args.arm]
    if not checkpoint.is_file() or sha256(checkpoint) != expected_sha:
        raise RuntimeError("frozen checkpoint missing or SHA drift")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {args.out}")
    return checkpoint, expected_sha


def assert_full_source_model_mapping(cfg: Any, arm: str) -> dict[str, Any]:
    """Return the complete source-exact mapping or fail on any key/value drift."""
    composed_model = OmegaConf.to_container(cfg.model, resolve=True)
    source_model = OmegaConf.to_container(OmegaConf.load(SOURCE_RESOLVED_CONFIG[arm]).model, resolve=False)
    assert isinstance(composed_model, dict) and isinstance(source_model, dict)
    # Resolve the four source-config interpolations against the frozen audit
    # contract; every other key/value must remain verbatim source-equivalent.
    source_model.update({
        "teacher_ckpt_path": str(FROZEN_TEACHER.resolve()), "window_size": int(cfg.data.window_size),
        "trial_length": int(cfg.data.max_trial_length), "pad_value": float(cfg.data.pad_value),
    })
    if composed_model != source_model:
        only_composed = sorted(set(composed_model).difference(source_model))
        only_source = sorted(set(source_model).difference(composed_model))
        changed = sorted(k for k in set(composed_model).intersection(source_model) if composed_model[k] != source_model[k])
        raise RuntimeError(f"full source model mapping drift; only_composed={only_composed}; only_source={only_source}; changed={changed}")
    return composed_model


def compose_datamodule(args: argparse.Namespace) -> tuple[FalconDataModule, Any]:
    config_dir = str(MUA_ROOT / "configs")
    # query_start is kept at 0 for standard DM construction.  Held-in q24 is
    # rebuilt in make_heldin_query_loader below, after the source-only train
    # normalization has been created by the untouched production setup.
    overrides = [
        f"experiment={CONFIG_EXPERIMENT[args.arm]}", f"model={AUDIT_MODEL_CONFIG[args.arm]}", "data.loso_fold=1", "seed=42",
        f"data.data_dir={ROOT / 'SPINT-main/data/000953'}", f"model.teacher_ckpt_path={FROZEN_TEACHER}",
        "train=false", "test=true", "data.calibration_n_trials=24", "data.random_calibration=false",
        "data.include_heldout_in_fit=false", f"data.include_heldout_in_test={'true' if args.domain == 'heldout' else 'false'}",
        "data.query_start_trial=0", "data.allow_empty_heldout_query=false",
        "data.num_workers=0", "data.pin_memory=false",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train.yaml", overrides=overrides)
    dm = hydra.utils.instantiate(cfg.data)
    if not isinstance(dm, FalconDataModule):
        raise TypeError("expected FalconDataModule")
    composed_model = assert_full_source_model_mapping(cfg, args.arm)
    expected_side_group = "none" if args.arm == "f0" else "t4"
    composed_mapping = {
        "source_exact_model_config": AUDIT_MODEL_CONFIG[args.arm], "source_resolved_config": str(SOURCE_RESOLVED_CONFIG[args.arm].resolve()),
        "source_resolved_config_sha256": EXPECTED_SOURCE_RESOLVED_CONFIG_SHA[args.arm], "model": composed_model,
        "side_feature_group": str(cfg.data.side_feature_group), "calibration_n_trials": int(cfg.data.calibration_n_trials),
        "window_size": int(cfg.data.window_size), "validation_protocol": str(cfg.data.validation_protocol),
        "loso_fold": int(cfg.data.loso_fold), "include_heldout_in_fit": bool(cfg.data.include_heldout_in_fit),
    }
    if composed_mapping["side_feature_group"] != expected_side_group or (composed_mapping["calibration_n_trials"], composed_mapping["window_size"], composed_mapping["validation_protocol"], composed_mapping["loso_fold"], composed_mapping["include_heldout_in_fit"]) != (24, 50, "loso", 1, False):
        raise RuntimeError("source-exact model passed but audit data contract drifted")
    dm.setup(stage="test")
    dm._audit_composed_mapping = composed_mapping
    return dm, cfg


def _common_dataset_kwargs(dm: FalconDataModule) -> dict[str, Any]:
    h = dm.hparams
    kwargs: dict[str, Any] = {
        "window_size": int(h.window_size), "calibration_n_trials": int(h.calibration_n_trials),
        "random_calibration": False, "smooth_calibration": bool(h.smooth_calibration),
        "max_trial_length": int(h.max_trial_length), "use_calib_intertrials": bool(h.use_calib_intertrials),
        "trial_feature_type": h.trial_feature_type, "remove_still_times": bool(h.remove_still_times),
        "remove_calib_still_times": bool(h.remove_calib_still_times),
        "use_calib_active_segments": bool(h.use_calib_active_segments), "calib_n_active_segments": int(h.calib_n_active_segments),
        "interpolate_trials": bool(h.interpolate_trials), "interpolate_trials_kind": h.interpolate_trials_kind,
        "pad_value": float(h.pad_value), "side_feature_group": str(h.side_feature_group),
        "side_feature_shuffle_seed": int(h.side_feature_shuffle_seed),
    }
    normalization = getattr(dm, "native_t4_normalization", None)
    if normalization is not None:
        kwargs["side_feature_mean"] = normalization["mean"]
        kwargs["side_feature_std"] = normalization["std"]
    return kwargs


def make_heldin_query_loader(dm: FalconDataModule, query_start: int) -> tuple[DataLoader, dict[str, Any]]:
    names = list(dm.val_heldin_session_names)
    query = dm._subset_sessions(dm.val_heldin_sessions, names)
    calibration = dm._subset_sessions(dm.train_calib_heldin_sessions, names)
    dataset = FalconDataset(
        sessions_dict=query, calib_sessions_dict=calibration, split="audit_heldin_query_only",
        query_start_trial=query_start, allow_empty_query_sessions=False, **_common_dataset_kwargs(dm),
    )
    sampler = SessionBatchSampler(dataset, dm.batch_size_per_device, shuffle=False)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=False), dataset.query_window_audit


def make_heldout_loader(dm: FalconDataModule, query_start: int) -> tuple[DataLoader, dict[str, Any]]:
    # Production held-out construction is reused. Rebuild only because q0 and
    # q24 are separately materialized audit cells, never selected by score.
    if query_start == 0:
        if dm.val_heldout_dataset is None:
            raise RuntimeError("held-out dataset was not built")
        dataset = dm.val_heldout_dataset
    else:
        h = dm.hparams
        # dm holds prepared held-out sessions and source-only normalization.
        dataset = FalconDataset(
            sessions_dict=dm.val_calib_heldout_sessions, calib_sessions_dict=dm.val_calib_heldout_sessions,
            split="audit_heldout_future_query", query_start_trial=query_start,
            allow_empty_query_sessions=False, **_common_dataset_kwargs(dm),
        )
    sampler = SessionBatchSampler(dataset, dm.batch_size_per_device, shuffle=False)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=False), dataset.query_window_audit


def source_heldin_anchor_loader(dm: FalconDataModule) -> DataLoader:
    """Return only the standard source-heldin loader for production idx-0 semantics."""
    loaders = dm.test_dataloader()
    if not isinstance(loaders, list) or len(loaders) != 2:
        raise RuntimeError("held-out audit requires the production [heldin, heldout] test-loader topology")
    return loaders[0]


def query_eligibility_audit(dm: FalconDataModule, args: argparse.Namespace) -> dict[str, Any]:
    """Audit eligibility without a model forward pass or any score output."""
    if args.domain == "heldin":
        sessions = list(dm.val_heldin_session_names)
        rows = {}
        for session in sessions:
            query_trial_boundaries = int(dm.val_heldin_sessions[session]["trial_change"].sum())
            calibration_trial_boundaries = int(dm.train_calib_heldin_sessions[session]["trial_change"].sum())
            eligible = args.query_start == 0 or query_trial_boundaries > args.query_start
            rows[session] = {
                "query_file_trial_boundaries": query_trial_boundaries,
                "calibration_file_trial_boundaries": calibration_trial_boundaries,
                "support_trials": args.query_start,
                "status": "eligible" if eligible else "structural_ineligible_zero_future_query",
                "reason": None if eligible else "held-in minival query file has no trial boundary after the fixed 24-trial support boundary",
            }
        return {"domain": "heldin", "rows": rows, "metric_route": {"target_loader_index": 0, "target_metric_prefix": "test_heldin_"}}
    target_loader, window_audit = make_heldout_loader(dm, args.query_start)
    anchor = source_heldin_anchor_loader(dm)
    return {
        "domain": "heldout", "rows": window_audit,
        "metric_route": {
            "anchor_loader_index": 0, "anchor_metric_prefix": "test_heldin_", "anchor_role": "unscored_ignored_no_selection",
            "target_loader_index": 1, "target_metric_prefix": "test_heldout_", "target_loader_batches": len(target_loader),
            "anchor_loader_batches": len(anchor),
        },
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint, checkpoint_sha = fail_closed_preflight(args)
    dm, cfg = compose_datamodule(args)
    eligibility = query_eligibility_audit(dm, args)
    if args.domain == "heldin" and args.query_start == 24:
        raise RuntimeError(f"structural_ineligible_zero_future_query: {eligibility}")
    if args.domain == "heldin":
        loader, window_audit = make_heldin_query_loader(dm, args.query_start)
        loaders: DataLoader | list[DataLoader] = loader
        target_prefix = "test_heldin_"
    else:
        loader, window_audit = make_heldout_loader(dm, args.query_start)
        anchor_loader = source_heldin_anchor_loader(dm)
        loaders = [anchor_loader, loader]
        target_prefix = "test_heldout_"
    if args.query_start == 24:
        for session, audit in window_audit.items():
            if not audit["full_window_disjoint"] or audit["eligible_windows"] <= 0:
                raise RuntimeError(f"q24 full-window disjointness failure for {session}: {audit}")
            if audit["minimum_window_start_padded_bin"] != audit["raw_query_start_bin"] + 49:
                raise RuntimeError(f"q24 minimum-start audit mismatch for {session}: {audit}")
    # Match the project test-only runner: materialize the module first, let
    # Lightning call setup(), then restore the frozen checkpoint. Direct
    # load_from_checkpoint is invalid here because setup() creates teacher and
    # student submodules required by this checkpoint's state dictionary.
    model = hydra.utils.instantiate(cfg.model)
    if not isinstance(model, StreamingCalibrationLitModule):
        raise TypeError("expected StreamingCalibrationLitModule")
    trainer = L.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False)
    trainer.test(model=model, dataloaders=loaders, ckpt_path=str(checkpoint), verbose=False)
    metrics = {name: float(value.detach().cpu()) for name, value in trainer.callback_metrics.items() if hasattr(value, "detach")}
    session_r2 = {}
    for session in window_audit:
        key = f"{target_prefix}{session}/r2"
        if key not in metrics:
            raise RuntimeError(f"missing per-session R2 metric {key}")
        session_r2[session] = metrics[key]
    anchor_metrics_ignored = None
    if args.domain == "heldout":
        anchor_session_r2 = {}
        for session in dm.val_heldin_session_names:
            key = f"test_heldin_{session}/r2"
            if key not in metrics:
                raise RuntimeError(f"missing source-heldin anchor metric {key}")
            anchor_session_r2[session] = metrics[key]
        anchor_metrics_ignored = {
            "metric_prefix": "test_heldin_", "session_r2": anchor_session_r2,
            "role": "ignored_provenance_only_not_used_for_selection_or_aggregation",
        }
    return {
        "schema_version": 1, "workflow": "0f_m2_m24_domain_query_sign_audit",
        "scope": "CPU-only, frozen-checkpoint, test-only; no backward/optimizer/checkpoint selection",
        "cell": {"arm": args.arm, "domain": args.domain, "query_start_trial": args.query_start,
                 "checkpoint_policy": args.checkpoint_policy, "task": "m2", "fold": 1, "seed": 42,
                 "calibration_trials": 24, "window_size": 50},
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha},
        "receipt": {"path": str(RECEIPT.resolve()), "sha256": EXPECTED_RECEIPT_SHA},
        "source_internal_aggregate": {"path": str(SOURCE.resolve()), "sha256": EXPECTED_SOURCE_SHA},
        "query_start_role": "diagnostic_resubstitution_not_future_query" if args.query_start == 0 else "chronological_future_query",
        "session_r2": session_r2, "equal_session_mean_r2": sum(session_r2.values()) / len(session_r2),
        "query_window_audit": window_audit, "cuda_available_at_preflight": False,
        "composed_model_mapping": dm._audit_composed_mapping,
        "metric_route": query_eligibility_audit(dm, args)["metric_route"],
        "anchor_metrics_ignored": anchor_metrics_ignored,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("f0", "t4"))
    parser.add_argument("--domain", required=True, choices=("heldin", "heldout"))
    parser.add_argument("--query-start", required=True, type=int, choices=(0, 24))
    parser.add_argument("--checkpoint-policy", required=True, choices=("best", "epoch9"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-trials", type=int, default=24)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="preflight plus trial-boundary eligibility only; no model forward pass")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        checkpoint, checkpoint_sha = fail_closed_preflight(args)
        dm, _ = compose_datamodule(args)
        result = {
            "schema_version": 1, "workflow": "0f_m2_m24_domain_query_sign_audit_dry_run",
            "dry_run": True, "no_model_forward": True, "no_backward": True, "no_optimizer": True,
            "cell": {"arm": args.arm, "domain": args.domain, "query_start_trial": args.query_start,
                     "checkpoint_policy": args.checkpoint_policy, "task": "m2", "fold": 1, "seed": 42,
                     "calibration_trials": 24, "window_size": 50},
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha},
            "receipt": {"path": str(RECEIPT.resolve()), "sha256": EXPECTED_RECEIPT_SHA},
            "eligibility": query_eligibility_audit(dm, args), "cuda_available_at_preflight": False,
            "composed_model_mapping": dm._audit_composed_mapping,
        }
    else:
        result = evaluate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # preflight establishes out does not exist; parent is created only after it.
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
