#!/usr/bin/env python3
"""One-shot outer evaluator for the audited RT AFC4 XLSv2 arm.

All fit identity, inner-only selection, source-normalizer scope, immutable
support-audit, and source-session permutation receipts are validated before
the outer target NWB is opened.  The target pass is inference-only and the
target support permutation must match the immutable CPU audit exactly before
any query window is scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

# The production supervisor invokes this file by absolute path.  In that mode
# Python places ``src/`` rather than the project root on ``sys.path``, so the
# package import below would fail before any data are opened.  Keep the CLI
# valid both as a script and as ``python -m src...``; this is import bootstrap
# only and does not alter the evaluator or its target boundary.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import rt_clean_nested_loso_eval as base
from src.data.afc4_xls_v2 import audited_session_expected_sha256
from src.data.afc4_xls_v2_adapter import (
    AUDIT_SHA256,
    load_immutable_xls_v2_audit,
)
from src.data.falcon_datamodule import SessionBatchSampler
from src.data.rt_nested_loso_datamodule import (
    RtNestedLossoDataModule,
    build_outer_target_dataset,
    nested_loso_partition,
)


ARM = "afc4_xls_v2"
SCHEMA = "rt_clean_nested_loso_xls_v2_outer_eval_v1"
STATUS = "PASS_ONE_SHOT_XLS_V2_OUTER_TARGET_NO_BACKPROP"


class XlsV2OuterEvalError(ValueError):
    """An XLSv2-specific outer-evaluation invariant failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise XlsV2OuterEvalError(message)


def validate_xls_v2_fit_manifest(
    manifest: Mapping[str, Any],
    *,
    support_audit_path: str | Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate the source-only XLSv2 receipt without opening an NWB."""

    _need(manifest.get("validation_protocol") == "nested_loso", "XLSv2 requires nested_loso")
    _need(manifest.get("requested_side_feature_group") == ARM, "XLSv2 manifest arm drift")
    arm = manifest.get("arm")
    _need(isinstance(arm, Mapping) and arm.get("canonical_arm") == ARM, "XLSv2 canonical arm drift")
    nested = manifest.get("nested_selection")
    _need(isinstance(nested, Mapping) and nested.get("clean") is True, "XLSv2 nested selection is not clean")
    for key, value in (
        ("outer_target_loaded_during_fit", False),
        ("outer_target_query_labels_read_during_fit", False),
        ("inner_validation_only_for_checkpoint_selection", True),
        ("checkpoint_metric", "val_heldin/r2_mean"),
        ("checkpoint_metric_scope", "inner_validation_session_only"),
    ):
        _need(nested.get(key) == value, f"XLSv2 nested selection drift: {key}")
    calibration, query = manifest.get("calibration"), manifest.get("query")
    _need(isinstance(calibration, Mapping) and isinstance(query, Mapping), "XLSv2 M24/query receipt absent")
    _need(
        calibration.get("budget_trials") == 24
        and calibration.get("trial_index_range") == [0, 24]
        and calibration.get("target_calibration_optimizer_steps") == 0,
        "XLSv2 must use chronological M24 with zero target optimizer steps",
    )
    _need(
        query.get("query_start_trial") == 24
        and query.get("window_size_bins") == 50
        and query.get("full_window_after_support_required") is True,
        "XLSv2 query must be q24/window50/full-window-disjoint",
    )
    normalizer = manifest.get("source_only_normalizer")
    inner_train = list(manifest.get("inner_train_sessions", []))
    inner_validation = manifest.get("inner_validation_session")
    outer_target = manifest.get("target_session")
    _need(len(inner_train) == 13 and isinstance(inner_validation, str), "XLSv2 requires 13 inner train plus one inner validation")
    _need(isinstance(normalizer, Mapping), "XLSv2 source-only normalizer absent")
    _need(normalizer.get("fit_scope") == "inner_train_sessions_only", "XLSv2 normalizer scope drift")
    _need(normalizer.get("feature_group") == ARM, "XLSv2 normalizer feature group drift")
    _need(list(normalizer.get("fit_sessions", [])) == inner_train, "XLSv2 normalizer fit sessions drift")
    _need(normalizer.get("excluded_inner_validation_session") == inner_validation, "XLSv2 normalizer included inner validation")
    _need(normalizer.get("excluded_outer_target_session") == outer_target, "XLSv2 normalizer included outer target")

    receipt, digest = load_immutable_xls_v2_audit(support_audit_path)
    _need(digest == AUDIT_SHA256, "XLSv2 immutable support-audit SHA drift")
    binding = manifest.get("xls_v2_support_audit")
    _need(isinstance(binding, Mapping), "XLSv2 support-audit binding absent")
    _need(binding.get("sha256") == AUDIT_SHA256, "XLSv2 manifest audit SHA drift")
    _need(binding.get("mode_required") == "0444", "XLSv2 manifest audit mode drift")
    _need(Path(str(binding.get("path", ""))).resolve() == Path(support_audit_path).resolve(), "XLSv2 manifest audit path drift")
    _need(binding.get("query_labels_available_to_generator") is False, "XLSv2 generator saw query labels")
    _need(binding.get("common_inverse_or_alignment_map") is False, "XLSv2 manifest permits an alignment map")
    per_session = binding.get("per_session_permutation_sha256")
    _need(isinstance(per_session, Mapping), "XLSv2 per-session permutation map absent")
    fit_sessions = inner_train + [str(inner_validation)]
    _need(set(per_session) == set(fit_sessions), "XLSv2 permutation map must cover exactly 14 fit sessions")

    source_audits = manifest.get("source_k4_calibration_audit")
    inner_audits = manifest.get("inner_validation_k4_calibration_audit")
    _need(isinstance(source_audits, Mapping) and isinstance(inner_audits, Mapping), "XLSv2 K4 audit maps absent")
    combined = {**source_audits, **inner_audits}
    _need(set(source_audits) == set(inner_train), "XLSv2 source audit scope is not inner-train-only")
    _need(set(inner_audits) == {inner_validation}, "XLSv2 inner-validation audit scope drift")
    for session_name in fit_sessions:
        row = combined.get(session_name)
        _need(isinstance(row, Mapping), f"XLSv2 calibration audit absent for {session_name}")
        actual = row.get("label_permutation_sha256")
        expected = audited_session_expected_sha256(receipt, session_name=session_name)
        _need(actual == expected == per_session.get(session_name), f"XLSv2 permutation SHA mismatch for {session_name}")
        _need(row.get("xls_v2_support_audit_sha256") == AUDIT_SHA256, f"XLSv2 audit SHA missing for {session_name}")
        _need(row.get("query_labels_available_to_generator") is False, f"XLSv2 query-label scope drift for {session_name}")
        _need(row.get("common_inverse_or_alignment_map") is False, f"XLSv2 alignment-map drift for {session_name}")
    return receipt, {str(key): str(value) for key, value in per_session.items()}


def _normalizer(manifest: Mapping[str, Any], split: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    normalizer = manifest["source_only_normalizer"]
    _need(list(normalizer.get("fit_sessions", [])) == list(split["inner_train_sessions"]), "normalizer/split drift")
    mean = np.asarray(normalizer.get("mean"), dtype=np.float32)
    std = np.asarray(normalizer.get("std"), dtype=np.float32)
    _need(mean.shape == (4,) and std.shape == (4,), "XLSv2 normalizer must be four-dimensional")
    _need(np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std > 0.0), "invalid XLSv2 normalizer")
    return mean, std


def evaluate_outer_target(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    split_manifest_path: str | Path,
    selection_receipt_path: str | Path,
    output_path: str | Path | None = None,
    outer_loso_fold: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Validate, open one target once, and score without updating state."""

    cfg = base._resolved_config(config_path)
    manifest = base._json_load(split_manifest_path)
    audit_path = Path(str(cfg.data.get("xls_v2_support_audit_path", ""))).resolve()
    audit_receipt, source_permutation_sha = validate_xls_v2_fit_manifest(
        manifest,
        support_audit_path=audit_path,
    )
    base._validate_selection_receipt(
        receipt_path=selection_receipt_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        split_manifest_path=split_manifest_path,
        cfg=cfg,
        manifest=manifest,
    )
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(f"Refusing to overwrite XLSv2 outer receipt: {output_path}")

    sessions = list(manifest.get("session_names", []))
    fold = int(manifest.get("outer_loso_fold"))
    _need(bool(sessions), "XLSv2 manifest session list absent")
    if outer_loso_fold is not None:
        _need(int(outer_loso_fold) == fold, "CLI outer fold disagrees with XLSv2 manifest")
    split = nested_loso_partition(sessions, fold, expected_session_count=len(sessions))
    _need(split.outer_target_session == manifest.get("target_session"), "XLSv2 outer target session drift")
    _need(list(split.inner_train_sessions) == list(manifest.get("inner_train_sessions", [])), "XLSv2 inner train split drift")
    _need(split.inner_validation_session == manifest.get("inner_validation_session"), "XLSv2 inner validation drift")
    _need(
        str(cfg.data.get("_target_", ""))
        == "src.data.rt_nested_loso_datamodule.RtNestedLossoDataModule",
        "XLSv2 evaluator requires the clean RT nested DataModule",
    )
    _need(str(cfg.data.side_feature_group).lower() == ARM, "XLSv2 evaluator config arm drift")
    _need(int(cfg.seed) == 42, "XLSv2 evaluator is frozen to seed 42")
    mean, std = _normalizer(manifest, split.as_dict())

    model = hydra.utils.instantiate(cfg.model)
    setup = getattr(model, "setup", None)
    if callable(setup):
        setup("fit")
    base._restore_selected_checkpoint(model, checkpoint_path)
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _need(not model.training, "XLSv2 outer model failed to enter eval mode")
    state_before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    state_before_sha = base._state_digest(state_before)

    target_dataset, target_split, target_path = build_outer_target_dataset(
        data_dir=Path(str(cfg.data.data_dir)),
        outer_loso_fold=fold,
        side_feature_group=ARM,
        side_feature_shuffle_seed=int(cfg.data.side_feature_shuffle_seed),
        calibration_n_trials=int(cfg.data.calibration_n_trials),
        query_start_trial=int(cfg.data.query_start_trial),
        window_size=int(cfg.data.window_size),
        max_trial_length=int(cfg.data.max_trial_length),
        interpolate_trials=bool(cfg.data.interpolate_trials),
        interpolate_trials_kind=str(cfg.data.interpolate_trials_kind),
        pad_value=float(cfg.data.pad_value),
        expected_session_count=int(cfg.data.expected_session_count),
        side_feature_mean=mean,
        side_feature_std=std,
        xls_v2_support_audit_path=audit_path,
    )
    _need(target_split == split, "XLSv2 target partition changed at open boundary")
    target_audit = target_dataset.k4_audits.get(split.outer_target_session)
    _need(isinstance(target_audit, Mapping), "XLSv2 target support audit absent")
    target_permutation_sha = target_audit.get("label_permutation_sha256")
    expected_target_sha = audited_session_expected_sha256(audit_receipt, session_name=split.outer_target_session)
    _need(target_permutation_sha == expected_target_sha, "XLSv2 target support permutation SHA mismatch")
    _need(target_audit.get("xls_v2_support_audit_sha256") == AUDIT_SHA256, "XLSv2 target audit SHA drift")
    _need(target_audit.get("query_labels_available_to_generator") is False, "XLSv2 target generator saw query labels")

    sampler = SessionBatchSampler(target_dataset, int(cfg.data.batch_size), shuffle=False)
    _need(bool(sampler.session_batch_counts), "XLSv2 outer target has no post-M24 query windows")
    loader = DataLoader(target_dataset, batch_sampler=sampler, num_workers=0, pin_memory=False)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model.model_step(base._move_batch(batch, torch_device))
            predictions.append(output["behavior_pred"].detach().cpu().numpy().reshape(-1, 2))
            targets.append(output["behavior_target"].detach().cpu().numpy().reshape(-1, 2))
    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    score = base._r2_variance_weighted(prediction, target)
    state_after = model.state_dict()
    state_after_sha = base._state_digest(state_after)
    _need(state_before_sha == state_after_sha, "XLSv2 outer pass changed model state")
    _need(
        all(torch.equal(state_before[key], value.detach().cpu()) for key, value in state_after.items()),
        "XLSv2 outer pass changed a model tensor",
    )

    checkpoint = Path(checkpoint_path).resolve()
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "arm": ARM,
        "run_id": str(cfg.run_id),
        "seed": int(cfg.seed),
        "outer_loso_fold": fold,
        "outer_target_session": split.outer_target_session,
        "outer_target_path": str(target_path.resolve()),
        "inner_train_sessions": list(split.inner_train_sessions),
        "inner_validation_session": split.inner_validation_session,
        "query_start_trial": 24,
        "window_size": 50,
        "query_windows_evaluated": int(prediction.shape[0]),
        "r2_variance_weighted": score,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "selection_receipt_path": str(Path(selection_receipt_path).resolve()),
        "config_path": str(Path(config_path).resolve()),
        "fit_split_manifest": str(Path(split_manifest_path).resolve()),
        "normalizer_fit_scope": "inner_train_sessions_only",
        "xls_v2_support_audit_path": str(audit_path),
        "xls_v2_support_audit_sha256": AUDIT_SHA256,
        "xls_v2_source_permutation_sha256": source_permutation_sha,
        "xls_v2_target_permutation_sha256": target_permutation_sha,
        "target_support_calibration_velocity_used": True,
        "target_support_calibration_labels_used": True,
        "target_query_labels_used_for_scoring_only": True,
        "target_query_labels_used_for_calibration": False,
        "target_query_labels_used_for_normalization": False,
        "target_query_labels_used_for_checkpoint_selection": False,
        "target_backpropagation": False,
        "optimizer_present": False,
        "model_training_mode": False,
        "model_state_sha256_before": state_before_sha,
        "model_state_sha256_after": state_after_sha,
        "model_state_unchanged": True,
        "query_labels_available_to_xls_v2_generator": False,
        "common_inverse_or_alignment_map": False,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite XLSv2 outer receipt: {destination}")
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(evaluate_outer_target(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split_manifest_path=args.split_manifest,
        selection_receipt_path=args.selection_receipt,
        output_path=args.output,
        outer_loso_fold=args.outer_fold,
        device=args.device,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
