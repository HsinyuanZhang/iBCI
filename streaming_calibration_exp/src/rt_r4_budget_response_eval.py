#!/usr/bin/env python3
"""One-shot outer evaluator for RT R4's fixed-M24/common-q24 contract.

The generic clean RT evaluator is intentionally frozen to one M24 carrier
budget.  R4 instead keeps the neural/activity calibration tensor at M24 and
re-fits only the analytic AFC4 carrier from a chronological M6 or M12 target
support prefix.  This module reuses the existing clean selection-receipt,
checkpoint, R2, and state-identity validators while providing the one missing
R4-specific target builder.

No Trainer or optimizer is constructed.  The selected model is frozen, the
target is evaluated under ``torch.no_grad()``, and the receipt is written once
and made immutable only after the before/after state SHA-256 values match.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.falcon_datamodule import FalconDataset, SessionBatchSampler  # noqa: E402
import src.data.rt_nested_loso_datamodule as nested  # noqa: E402
from src.data.rt_r4_budget_response_datamodule import (  # noqa: E402
    COMMON_QUERY_START,
    FIXED_ACTIVITY_BUDGET,
    PRIMARY_CARRIER_BUDGETS,
    R4_ARMS,
    fit_r4_prefix_descriptor,
)
from src.rt_clean_nested_loso_eval import (  # noqa: E402
    _file_sha256,
    _json_load,
    _move_batch,
    _normalizer_from_manifest,
    _r2_variance_weighted,
    _resolved_config,
    _restore_selected_checkpoint,
    _state_digest,
    _validate_selection_receipt,
)


R4_OUTER_SCHEMA = "rt_r4_budget_response_outer_eval_v1"
R4_OUTER_STATUS = "PASS_R4_ONE_SHOT_OUTER_TARGET_NO_BACKPROP"
R4_DATAMODULE_TARGET = (
    "src.data.rt_r4_budget_response_datamodule."
    "RtR4BudgetResponseNestedLossoDataModule"
)


class RtR4OuterEvalError(ValueError):
    """A fail-closed R4 target-side or receipt invariant failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RtR4OuterEvalError(message)


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def query_window_identity_sha256(dataset: FalconDataset) -> str:
    """Hash ordered query identities, without hashing query behavior values."""

    rows = [[str(name), int(start)] for name, start in dataset.window_indices]
    _need(bool(rows), "R4 outer target has no q24 query-window identities")
    return _canonical_sha256(
        {
            "query_start_trial": int(dataset.query_start_trial),
            "window_size": int(dataset.window_size),
            "ordered_session_and_padded_start": rows,
        }
    )


def _normalizer_sha256(
    *, mean: np.ndarray, std: np.ndarray, manifest: Mapping[str, Any]
) -> str:
    normalizer = manifest.get("source_only_normalizer")
    _need(isinstance(normalizer, Mapping), "R4 source-only normalizer is absent")
    return _canonical_sha256(
        {
            "fit_scope": normalizer.get("fit_scope"),
            "fit_sessions": list(normalizer.get("fit_sessions", [])),
            "excluded_inner_validation_session": normalizer.get(
                "excluded_inner_validation_session"
            ),
            "excluded_outer_target_session": normalizer.get(
                "excluded_outer_target_session"
            ),
            "activity_calibration_trials": normalizer.get(
                "activity_calibration_trials"
            ),
            "carrier_calibration_trials": normalizer.get(
                "carrier_calibration_trials"
            ),
            "mean_float32": np.asarray(mean, dtype=np.float32).tolist(),
            "std_float32": np.asarray(std, dtype=np.float32).tolist(),
        }
    )


def _validate_r4_manifest(
    manifest: Mapping[str, Any], *, cfg: Any
) -> tuple[int, str]:
    r4 = manifest.get("rt_r4_budget_response")
    _need(isinstance(r4, Mapping), "R4 evaluator requires the R4 split contract")
    _need(
        r4.get("schema") == "rt_r4_budget_response_common_q24_split_v1",
        "R4 split schema drift",
    )
    budget = int(cfg.data.side_feature_calibration_n_trials)
    arm = str(cfg.data.side_feature_group).lower()
    _need(budget in PRIMARY_CARRIER_BUDGETS, "Primary R4 evaluator permits M6/M12 only")
    _need(arm in R4_ARMS, f"R4 evaluator permits only {R4_ARMS}")
    _need(
        str(cfg.data.get("_target_", "")) == R4_DATAMODULE_TARGET,
        "R4 evaluator refused a non-R4 DataModule",
    )
    _need(
        int(cfg.data.calibration_n_trials) == FIXED_ACTIVITY_BUDGET
        and int(cfg.data.query_start_trial) == COMMON_QUERY_START,
        "R4 evaluator requires activity M24 and query q24",
    )
    _need(
        int(r4.get("activity_calibration_trials", -1)) == FIXED_ACTIVITY_BUDGET
        and int(r4.get("carrier_calibration_trials", -1)) == budget
        and int(r4.get("common_query_start_trial", -1)) == COMMON_QUERY_START,
        "R4 manifest/config budget identity mismatch",
    )
    _need(
        r4.get("activity_trial_index_range") == [0, FIXED_ACTIVITY_BUDGET]
        and r4.get("carrier_trial_index_range") == [0, budget]
        and r4.get("unused_for_carrier_and_query_trial_range")
        == [budget, COMMON_QUERY_START],
        "R4 split ranges drifted",
    )
    _need(
        r4.get("only_carrier_fit_prefix_varies") is True
        and r4.get("neural_activity_tensor_budget_varies") is False
        and r4.get("outer_target_loaded_during_fit") is False,
        "R4 split no longer proves isolated carrier-prefix variation",
    )
    normalizer = manifest.get("source_only_normalizer")
    _need(isinstance(normalizer, Mapping), "R4 manifest lacks source-only normalizer")
    _need(
        int(normalizer.get("activity_calibration_trials", -1))
        == FIXED_ACTIVITY_BUDGET
        and int(normalizer.get("carrier_calibration_trials", -1)) == budget
        and int(normalizer.get("common_query_start_trial", -1))
        == COMMON_QUERY_START,
        "R4 normalizer budget binding drifted",
    )
    return budget, arm


def build_r4_outer_target_dataset(
    *,
    data_dir: str | Path,
    outer_loso_fold: int,
    side_feature_group: str,
    side_feature_shuffle_seed: int,
    carrier_calibration_n_trials: int,
    activity_calibration_n_trials: int = FIXED_ACTIVITY_BUDGET,
    query_start_trial: int = COMMON_QUERY_START,
    window_size: int = 50,
    max_trial_length: int = 100,
    interpolate_trials: bool = True,
    interpolate_trials_kind: str = "cubic",
    pad_value: float = -1.0,
    expected_session_count: int = nested.RT_EXPECTED_SESSION_COUNT,
    side_feature_mean: np.ndarray | None = None,
    side_feature_std: np.ndarray | None = None,
) -> tuple[FalconDataset, nested.NestedRtSplit, Path, dict[str, Any]]:
    """Open one target, keep M24 activity/q24, and replace only its AFC4 prefix."""

    budget = int(carrier_calibration_n_trials)
    group = str(side_feature_group).lower()
    _need(budget in PRIMARY_CARRIER_BUDGETS, "R4 outer builder permits M6/M12 only")
    _need(group in R4_ARMS, f"R4 outer builder permits only {R4_ARMS}")
    _need(
        int(activity_calibration_n_trials) == FIXED_ACTIVITY_BUDGET,
        "R4 outer activity must remain M24",
    )
    _need(int(query_start_trial) == COMMON_QUERY_START, "R4 outer query must remain q24")
    _need(
        side_feature_mean is not None and side_feature_std is not None,
        "R4 outer builder requires the checkpoint-bound inner-train normalizer",
    )
    mean = np.asarray(side_feature_mean, dtype=np.float32)
    std = np.asarray(side_feature_std, dtype=np.float32)
    _need(
        mean.shape == (4,) and std.shape == (4,)
        and np.isfinite(mean).all() and np.isfinite(std).all()
        and np.all(std > 0.0),
        "R4 outer builder received an invalid source-only normalizer",
    )

    indexed = nested.index_rt_session_paths(
        data_dir, expected_session_count=int(expected_session_count)
    )
    split = nested.nested_loso_partition(
        tuple(indexed), int(outer_loso_fold), expected_session_count=int(expected_session_count)
    )
    target_path = indexed[split.outer_target_session]
    raw = nested.load_rt_session(target_path)
    _need(
        str(raw.get("session_name")) == split.outer_target_session,
        "R4 outer target name disagrees with indexed path",
    )
    hparams = {
        "window_size": int(window_size),
        "calibration_n_trials": FIXED_ACTIVITY_BUDGET,
        "max_trial_length": int(max_trial_length),
        "interpolate_trials": bool(interpolate_trials),
        "interpolate_trials_kind": str(interpolate_trials_kind),
        "pad_value": float(pad_value),
        "side_feature_shuffle_seed": int(side_feature_shuffle_seed),
        "xls_v2_support_audit_path": None,
    }
    dataset = FalconDataset(
        sessions_dict=OrderedDict([(split.outer_target_session, raw)]),
        calib_sessions_dict=OrderedDict([(split.outer_target_session, raw)]),
        split="rt_r4_outer_target_one_shot",
        side_feature_mean=mean,
        side_feature_std=std,
        **nested._dataset_kwargs(hparams, group, COMMON_QUERY_START),
    )

    name = split.outer_target_session
    _need(int(dataset.calib_n_trials[name]) == FIXED_ACTIVITY_BUDGET,
          "R4 target activity tensor is not M24")
    m24_raw = np.asarray(dataset.k4_raw_features[name], dtype=np.float32).copy()
    prefix_raw, prefix_audit = fit_r4_prefix_descriptor(
        dataset.k4_raw_calibration[name], carrier_budget_trials=budget
    )
    dataset.k4_raw_features[name] = prefix_raw
    dataset.k4_audits[name] = prefix_audit
    dataset._side_feature_cache.clear()
    normalized_masked = dataset._native_k4_side_features(
        name, 0, FIXED_ACTIVITY_BUDGET
    ).copy()
    _need(
        prefix_audit.get("calibration_trials") == budget,
        "R4 target prefix audit does not bind the requested budget",
    )
    if group == "afc4_mb4":
        _need(
            np.array_equal(normalized_masked[:, :2], np.zeros_like(normalized_masked[:, :2])),
            "R4 MB4 target mask did not zero normalized direction components",
        )
    audit = {
        "schema": "rt_r4_outer_target_prefix_fit_v1",
        "status": "PASS_TARGET_ACTIVITY_M24_CARRIER_PREFIX_QUERY_Q24",
        "outer_target_session": name,
        "activity_calibration_trials": FIXED_ACTIVITY_BUDGET,
        "carrier_calibration_trials": budget,
        "query_start_trial": COMMON_QUERY_START,
        "unused_for_carrier_and_query_trial_range": [budget, COMMON_QUERY_START],
        "raw_prefix_descriptor_sha256": _array_sha256(prefix_raw),
        "raw_m24_descriptor_sha256_diagnostic_only": _array_sha256(m24_raw),
        "raw_prefix_is_not_m24": bool(not np.array_equal(prefix_raw, m24_raw)),
        "normalized_masked_descriptor_sha256": _array_sha256(normalized_masked),
        "prefix_fit": prefix_audit,
        "query_window_identity_sha256": query_window_identity_sha256(dataset),
        "query_windows": int(len(dataset.window_indices)),
    }
    return dataset, split, target_path, audit


def _write_immutable_json(path: Path, body: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite R4 outer receipt: {path}")
    payload = json.dumps(dict(body), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    path.chmod(0o444)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_r4_outer_target(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    split_manifest_path: str | Path,
    selection_receipt_path: str | Path,
    output_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    outer_loso_fold: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate one selected R4 checkpoint once, without optimizer or backward."""

    cfg = _resolved_config(config_path)
    _need(data_dir is None, "--data-dir override is disabled")
    manifest = _json_load(split_manifest_path)
    _need(manifest.get("validation_protocol") == "nested_loso",
          "R4 outer evaluator requires nested LOSO")
    nested_selection = manifest.get("nested_selection")
    _need(isinstance(nested_selection, Mapping) and nested_selection.get("clean") is True,
          "R4 evaluator refused a non-clean split")
    _need(
        nested_selection.get("outer_target_loaded_during_fit") is False
        and nested_selection.get("outer_target_query_labels_read_during_fit") is False
        and nested_selection.get("inner_validation_only_for_checkpoint_selection") is True
        and nested_selection.get("checkpoint_metric") == "val_heldin/r2_mean"
        and nested_selection.get("checkpoint_metric_scope") == "inner_validation_session_only",
        "R4 split does not prove clean inner-only checkpoint selection",
    )
    budget, arm = _validate_r4_manifest(manifest, cfg=cfg)

    # Selection/checkpoint/config/split identities are validated before the
    # only target opener below.
    _validate_selection_receipt(
        receipt_path=selection_receipt_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        split_manifest_path=split_manifest_path,
        cfg=cfg,
        manifest=manifest,
    )
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(f"Refusing to overwrite R4 outer receipt: {output_path}")

    all_sessions = list(manifest.get("session_names", manifest.get("all_sessions", [])))
    _need(bool(all_sessions), "R4 split manifest has no deterministic session list")
    fold = int(manifest.get("outer_loso_fold", manifest.get("loso_fold")))
    _need(outer_loso_fold is None or int(outer_loso_fold) == fold,
          "CLI outer fold disagrees with R4 fit manifest")
    split = nested.nested_loso_partition(
        all_sessions, fold, expected_session_count=len(all_sessions)
    )
    _need(split.outer_target_session == manifest.get("target_session"),
          "R4 target session disagrees with deterministic partition")
    _need(list(split.inner_train_sessions) == list(manifest.get("inner_train_sessions", [])),
          "R4 inner-train list disagrees with deterministic partition")
    _need(split.inner_validation_session == manifest.get("inner_validation_session"),
          "R4 inner-validation session disagrees with deterministic partition")
    mean, std = _normalizer_from_manifest(
        manifest, split=split.as_dict(), feature_group=arm
    )
    _need(mean is not None and std is not None, "R4 evaluator requires carrier normalization")
    normalizer_sha = _normalizer_sha256(mean=mean, std=std, manifest=manifest)

    model = hydra.utils.instantiate(cfg.model)
    setup = getattr(model, "setup", None)
    if callable(setup):
        setup("fit")
    _restore_selected_checkpoint(model, checkpoint_path)
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _need(not model.training, "R4 evaluator failed to enter eval mode")
    state_before = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    state_before_sha = _state_digest(state_before)

    target_dataset, target_split, target_path, target_audit = build_r4_outer_target_dataset(
        data_dir=Path(str(cfg.data.data_dir)),
        outer_loso_fold=fold,
        side_feature_group=arm,
        side_feature_shuffle_seed=int(cfg.data.side_feature_shuffle_seed),
        carrier_calibration_n_trials=budget,
        activity_calibration_n_trials=int(cfg.data.calibration_n_trials),
        query_start_trial=int(cfg.data.query_start_trial),
        window_size=int(cfg.data.window_size),
        max_trial_length=int(cfg.data.max_trial_length),
        interpolate_trials=bool(cfg.data.interpolate_trials),
        interpolate_trials_kind=str(cfg.data.interpolate_trials_kind),
        pad_value=float(cfg.data.pad_value),
        expected_session_count=int(cfg.data.expected_session_count),
        side_feature_mean=mean,
        side_feature_std=std,
    )
    _need(target_split == split, "R4 target partition changed after fit selection")
    sampler = SessionBatchSampler(target_dataset, int(cfg.data.batch_size), shuffle=False)
    _need(bool(sampler.session_batch_counts), "R4 outer target has no eligible q24 windows")
    loader = DataLoader(
        target_dataset, batch_sampler=sampler, num_workers=0, pin_memory=False
    )

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            out = model.model_step(_move_batch(batch, torch_device))
            predictions.append(out["behavior_pred"].detach().cpu().numpy().reshape(-1, 2))
            targets.append(out["behavior_target"].detach().cpu().numpy().reshape(-1, 2))
    prediction_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    score = _r2_variance_weighted(prediction_array, target_array)
    state_after = model.state_dict()
    state_after_sha = _state_digest(state_after)
    if state_before_sha != state_after_sha:
        changed = [
            key for key, value in state_after.items()
            if not torch.equal(state_before[key], value.detach().cpu())
        ]
        raise RuntimeError(f"R4 outer evaluation changed model state: {changed[:8]}")

    checkpoint = Path(checkpoint_path).resolve()
    result: dict[str, Any] = {
        "schema": R4_OUTER_SCHEMA,
        "status": R4_OUTER_STATUS,
        "run_id": str(cfg.run_id),
        "seed": int(cfg.seed),
        "arm": arm,
        "outer_loso_fold": fold,
        "outer_target_session": target_split.outer_target_session,
        "outer_target_path": str(target_path.resolve()),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "selection_receipt_path": str(Path(selection_receipt_path).resolve()),
        "selection_receipt_sha256": _file_sha256(selection_receipt_path),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": _file_sha256(config_path),
        "fit_split_manifest": str(Path(split_manifest_path).resolve()),
        "fit_split_manifest_sha256": _file_sha256(split_manifest_path),
        "activity_calibration_trials": FIXED_ACTIVITY_BUDGET,
        "carrier_calibration_trials": budget,
        "query_start_trial": COMMON_QUERY_START,
        "window_size": int(cfg.data.window_size),
        "query_windows_evaluated": int(prediction_array.shape[0]),
        "query_window_identity_sha256": target_audit["query_window_identity_sha256"],
        "r2_variance_weighted": score,
        "target_carrier_prefix_fit": target_audit,
        "source_only_normalizer_sha256": normalizer_sha,
        "normalizer_fit_scope": "inner_train_sessions_only",
        "normalizer_fit_sessions": list(split.inner_train_sessions),
        "target_support_calibration_velocity_used": True,
        "target_support_trials_used_for_carrier": [0, budget],
        "target_support_trials_used_for_activity": [0, FIXED_ACTIVITY_BUDGET],
        "target_query_labels_used_for_scoring_only": True,
        "target_query_labels_used_for_calibration": False,
        "target_query_labels_used_for_normalization": False,
        "target_query_labels_used_for_checkpoint_selection": False,
        "target_backpropagation": False,
        "optimizer_present": False,
        "trainer_present": False,
        "model_training_mode": False,
        "model_state_sha256_before": state_before_sha,
        "model_state_sha256_after": state_after_sha,
        "model_state_unchanged": True,
    }
    if output_path is not None:
        _write_immutable_json(Path(output_path), result)
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
    result = evaluate_r4_outer_target(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split_manifest_path=args.split_manifest,
        selection_receipt_path=args.selection_receipt,
        output_path=args.output,
        outer_loso_fold=args.outer_fold,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

