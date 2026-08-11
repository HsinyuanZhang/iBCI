#!/usr/bin/env python3
"""One-shot outer-target evaluator for the clean RT nested-LOSO path.

This worker is intentionally separate from ``src/train.py``.  A clean fit
uses :class:`src.data.rt_nested_loso_datamodule.RtNestedLossoDataModule`, whose
``test_dataloader`` is fail-closed and whose outer target NWB is never opened.
After Lightning has selected a checkpoint using the *inner* source validation
session, this script opens exactly one outer target and evaluates it once.

The evaluator never calls ``backward`` or an optimizer, forces the model into
evaluation mode, freezes parameters, and verifies that the model state is
byte-for-byte unchanged by the target pass.  It uses the normalizer serialized
in the fit split manifest; no target statistics are fitted here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.falcon_datamodule import SessionBatchSampler  # noqa: E402
from src.data.rt_nested_loso_datamodule import (  # noqa: E402
    build_outer_target_dataset,
    nested_loso_partition,
)


def _json_load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _resolved_config(path: str | Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    try:
        OmegaConf.resolve(cfg)
    except Exception as error:
        # ``src.train`` serializes Hydra's runtime interpolations verbatim in
        # ``resolved_config.yaml`` (notably ``${hydra:runtime.output_dir}``).
        # The standalone evaluator is intentionally outside a Hydra job, so
        # replace only those bookkeeping paths with the config's parent and
        # resolve the substantive data/model interpolations normally.  Any
        # other unresolved interpolation still fails closed below.
        if "paths" not in cfg or "hydra" not in str(error).lower():
            raise
        run_dir = str(Path(path).resolve().parent)
        cfg.paths.output_dir = run_dir
        cfg.paths.work_dir = run_dir
        OmegaConf.resolve(cfg)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"Resolved config is not a mapping: {path}")
    return cfg


def _normalizer_from_manifest(
    manifest: Mapping[str, Any],
    *,
    split: Mapping[str, Any],
    feature_group: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read and validate the inner-train-only carrier normalizer."""

    normalizer = manifest.get("source_only_normalizer")
    inner_train = list(split["inner_train_sessions"])
    if feature_group in {
        "k4",
        "ks4",
        "k4ls",
        "afc4_vel",
        "afc4_rs",
        "afc4_ls",
        "afc4_mb4",
        "afc4_b4",
        "afc4_w4",
        "rt_sparse_endpoint_t4d",
    }:
        if not isinstance(normalizer, Mapping):
            raise ValueError("AFC4 outer evaluation requires source_only_normalizer in the fit manifest")
        if normalizer.get("fit_scope") != "inner_train_sessions_only":
            raise ValueError("Outer evaluator refused a normalizer not fit on inner train only")
        if list(normalizer.get("fit_sessions", [])) != inner_train:
            raise ValueError("Outer evaluator normalizer fit_sessions do not equal inner_train_sessions")
        if normalizer.get("excluded_outer_target_session") != split["outer_target_session"]:
            raise ValueError("Outer evaluator normalizer does not exclude the outer target")
        mean = np.asarray(normalizer.get("mean"), dtype=np.float32)
        std = np.asarray(normalizer.get("std"), dtype=np.float32)
        if mean.shape != (4,) or std.shape != (4,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("Outer evaluator found an invalid carrier normalizer")
        if np.any(std <= 0.0):
            raise ValueError("Outer evaluator found non-positive carrier normalizer std")
        if feature_group == "rt_sparse_endpoint_t4d" and (not np.array_equal(mean[2:], np.zeros(2, dtype=np.float32)) or not np.array_equal(std[2:], np.ones(2, dtype=np.float32))):
            raise ValueError("T4d normalizer must preserve its exact zero pad")
        return mean, std
    if normalizer not in (None, {}):
        raise ValueError("Zero4 outer evaluator unexpectedly received a carrier normalizer")
    return None, None


def _move_batch(batch: Sequence[Any], device: torch.device) -> tuple[Any, ...]:
    moved: list[Any] = []
    for value in batch:
        moved.append(value.to(device) if isinstance(value, torch.Tensor) else value)
    return tuple(moved)


def _r2_variance_weighted(predictions: np.ndarray, targets: np.ndarray) -> float:
    pred = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[0] < 3:
        raise ValueError(f"Outer RT R2 inputs must match [samples,2], got {pred.shape}/{target.shape}")
    residual = np.square(target - pred).sum(axis=0)
    total = np.square(target - target.mean(axis=0, keepdims=True)).sum(axis=0)
    denominator = float(total.sum())
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Outer RT target has zero or non-finite variance")
    score = 1.0 - float(residual.sum()) / denominator
    if not np.isfinite(score):
        raise ValueError("Outer RT R2 is non-finite")
    return float(score)


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Model state entry is not a tensor: {key}")
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_selection_receipt(
    *,
    receipt_path: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    split_manifest_path: str | Path,
    cfg: DictConfig,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate fit-end checkpoint identity before opening outer target data."""

    receipt = _json_load(receipt_path)
    if receipt.get("schema") != "rt_clean_nested_loso_selection_receipt_v1":
        raise ValueError("Outer evaluator requires a clean RT fit selection receipt")
    if receipt.get("status") != "PASS_FIT_INNER_SELECTION_ONLY":
        raise ValueError("Outer evaluator refused a non-passing RT selection receipt")
    if bool(cfg.get("test", False)):
        raise ValueError("Clean RT fit config must set test=false; outer evaluation is a separate one-shot phase")
    if receipt.get("selected_by_metric") != "val_heldin/r2_mean":
        raise ValueError("RT selection receipt monitor is not val_heldin/r2_mean")
    if receipt.get("selected_metric_scope") != "inner_validation_session_only":
        raise ValueError("RT selection receipt does not bind checkpoint selection to inner validation")
    if Path(str(receipt.get("selection_receipt_path", ""))).resolve() != Path(receipt_path).resolve():
        raise ValueError("CLI selection receipt path does not match its self-identity")
    checkpoint = Path(checkpoint_path).resolve()
    recorded_checkpoint = Path(str(receipt.get("best_model_path", ""))).resolve()
    if recorded_checkpoint != checkpoint:
        raise ValueError("CLI checkpoint does not equal the fit receipt's best_model_path")
    if not checkpoint.is_file() or _file_sha256(checkpoint) != receipt.get("best_model_sha256"):
        raise ValueError("Selected checkpoint bytes do not match the fit selection receipt")
    config = Path(config_path).resolve()
    recorded_config = Path(str(receipt.get("config_path", ""))).resolve()
    if recorded_config != config:
        raise ValueError("CLI config does not equal the fit receipt's config_path")
    if not config.is_file() or _file_sha256(config) != receipt.get("config_sha256"):
        raise ValueError("Evaluator config bytes do not match the fit selection receipt")
    split_manifest = Path(split_manifest_path).resolve()
    if not split_manifest.is_file() or _file_sha256(split_manifest) != receipt.get("split_manifest_sha256"):
        raise ValueError("Fit split manifest bytes do not match the fit selection receipt")
    recorded_split_manifest = Path(str(receipt.get("split_manifest_path", ""))).resolve()
    if recorded_split_manifest != split_manifest:
        raise ValueError("CLI split manifest path does not equal the fit receipt's recorded path")

    cfg_run_id = str(cfg.get("run_id", ""))
    if str(receipt.get("run_id", "")) != cfg_run_id:
        raise ValueError("RT selection receipt run_id disagrees with evaluator config")
    arm = str(cfg.data.side_feature_group).lower()
    if receipt.get("arm") != arm or manifest.get("requested_side_feature_group") != arm:
        raise ValueError("RT selection receipt/config/manifest arm identity mismatch")
    fold = int(cfg.data.outer_loso_fold)
    if int(receipt.get("outer_loso_fold", -1)) != fold:
        raise ValueError("RT selection receipt fold disagrees with evaluator config")
    if int(receipt.get("seed", -1)) != int(cfg.seed):
        raise ValueError("RT selection receipt seed disagrees with evaluator config")
    run_dir = checkpoint.parents[2] if len(checkpoint.parents) > 2 else checkpoint.parent
    if Path(str(receipt.get("run_dir", ""))).resolve() != run_dir.resolve():
        raise ValueError("RT selection receipt run directory disagrees with checkpoint path")
    if receipt.get("formal_heldout_opened") is not False:
        raise ValueError("RT selection receipt unexpectedly opened formal heldout")
    if receipt.get("outer_target_loaded_during_fit") is not False:
        raise ValueError("RT selection receipt does not prove target loader exclusion")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Selected checkpoint payload is not a mapping")
    if int(payload.get("epoch", -1)) != int(receipt.get("selected_epoch", -2)):
        raise ValueError("Selected checkpoint epoch disagrees with fit receipt")
    if int(payload.get("global_step", -1)) != int(receipt.get("selected_global_step", -2)):
        raise ValueError("Selected checkpoint global_step disagrees with fit receipt")
    selected_value = receipt.get("selected_metric_value")
    if not isinstance(selected_value, (float, int)) or not np.isfinite(float(selected_value)):
        raise ValueError("Fit selection receipt has no finite selected metric value")
    callbacks = payload.get("callbacks", {})
    if not isinstance(callbacks, Mapping):
        raise ValueError("Selected checkpoint lacks callback provenance")
    callback_matches = []
    for state in callbacks.values():
        if not isinstance(state, Mapping) or state.get("monitor") != "val_heldin/r2_mean":
            continue
        best_path = str(state.get("best_model_path", ""))
        score = state.get("best_model_score")
        score_value = None if score is None else float(score.item() if hasattr(score, "item") else score)
        if Path(best_path).resolve() == checkpoint and score_value is not None:
            callback_matches.append(score_value)
    if not callback_matches or not any(abs(value - float(selected_value)) <= 1.0e-7 for value in callback_matches):
        raise ValueError("Selected checkpoint callback provenance does not match fit receipt")
    return receipt


def _restore_selected_checkpoint(model: torch.nn.Module, checkpoint: str | Path) -> dict[str, Any]:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Selected RT checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("Selected RT checkpoint is not a Lightning state_dict checkpoint")
    incompatible = model.load_state_dict(payload["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Selected RT checkpoint state mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return payload


def evaluate_outer_target(
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
    """Evaluate one selected checkpoint on one outer target, exactly once."""

    cfg = _resolved_config(config_path)
    if data_dir is not None:
        raise ValueError("--data-dir override is disabled; evaluator data path is bound to the fit config")
    manifest = _json_load(split_manifest_path)
    if manifest.get("validation_protocol") != "nested_loso":
        raise ValueError("Outer evaluator requires a clean nested_loso fit manifest")
    nested = manifest.get("nested_selection")
    if not isinstance(nested, Mapping) or not nested.get("clean"):
        raise ValueError("Outer evaluator refused a non-clean nested fit manifest")
    if nested.get("outer_target_loaded_during_fit") is not False:
        raise ValueError("Fit manifest does not prove outer-target loader exclusion")
    if nested.get("outer_target_query_labels_read_during_fit") is not False:
        raise ValueError("Fit manifest does not prove outer-target query-label exclusion")
    if nested.get("inner_validation_only_for_checkpoint_selection") is not True:
        raise ValueError("Fit manifest does not prove inner-only checkpoint selection")
    if nested.get("checkpoint_metric") != "val_heldin/r2_mean":
        raise ValueError("Outer evaluator requires the declared inner validation checkpoint metric")
    if nested.get("checkpoint_metric_scope") != "inner_validation_session_only":
        raise ValueError("Outer evaluator refused an unspecified checkpoint metric scope")

    # Validate checkpoint and all fit identities before any target NWB is
    # opened.  This is deliberately before build_outer_target_dataset below.
    _validate_selection_receipt(
        receipt_path=selection_receipt_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        split_manifest_path=split_manifest_path,
        cfg=cfg,
        manifest=manifest,
    )
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(f"Refusing to overwrite outer-evaluation receipt: {output_path}")

    all_sessions = list(manifest.get("session_names", manifest.get("all_sessions", [])))
    if not all_sessions:
        all_sessions = list(manifest.get("source_sessions", [])) + [str(manifest["target_session"])]
    fold = int(manifest.get("outer_loso_fold", manifest.get("loso_fold")))
    if outer_loso_fold is not None and int(outer_loso_fold) != fold:
        raise ValueError("CLI outer_loso_fold disagrees with fit manifest")
    split = nested_loso_partition(all_sessions, fold, expected_session_count=len(all_sessions))
    if split.outer_target_session != manifest.get("target_session"):
        raise ValueError("Fit manifest target session disagrees with deterministic partition")
    if list(split.inner_train_sessions) != list(manifest.get("inner_train_sessions", [])):
        raise ValueError("Fit manifest inner-train session list disagrees with deterministic partition")
    if split.inner_validation_session != manifest.get("inner_validation_session"):
        raise ValueError("Fit manifest inner-validation session disagrees with deterministic partition")

    model_cfg = cfg.model
    standard_datamodule = "src.data.rt_nested_loso_datamodule.RtNestedLossoDataModule"
    rt_ld_datamodule = "src.data.rt_ld_datamodule.RtLdNestedLossoDataModule"
    configured_data_target = str(cfg.data.get("_target_", ""))
    is_rt_ld = configured_data_target == rt_ld_datamodule
    if configured_data_target not in {standard_datamodule, rt_ld_datamodule}:
        raise ValueError(
            "Outer evaluator requires the clean RT data module or its isolated "
            f"L-D adapter, got {configured_data_target!r}"
        )
    feature_group = str(cfg.data.side_feature_group).lower()
    mean, std = _normalizer_from_manifest(manifest, split=split.as_dict(), feature_group=feature_group)
    resolved_data_dir = Path(str(cfg.data.data_dir))

    # Instantiate and initialize the same model class, then restore the exact
    # selected checkpoint before opening target data.  No Trainer is used: this
    # worker has no optimizer, callback, or test/validation lifecycle that
    # could update model metrics or decoder state.
    model = hydra.utils.instantiate(model_cfg)
    setup = getattr(model, "setup", None)
    if callable(setup):
        setup("fit")
    _restore_selected_checkpoint(model, checkpoint_path)
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training:
        raise RuntimeError("Outer evaluator failed to enter eval mode")
    state_before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    state_before_digest = _state_digest(state_before)

    # Build the target dataset only after all fit-manifest and checkpoint
    # contracts have passed.  The normalizer is copied from 13 train sessions;
    # no transform is fit from the target payload.
    target_dataset, target_split, target_path = build_outer_target_dataset(
        data_dir=resolved_data_dir,
        outer_loso_fold=fold,
        side_feature_group=feature_group,
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
    )
    if is_rt_ld:
        # Build ordinary aligned Full first, then append only the isolated
        # consumer carrier.  XLSv2 consequently cannot reach the Full identity
        # encoder through the evaluator's input contract either.
        from src.data.rt_ld_adapter import RtLdDualCarrierDataset
        gain_source = str(cfg.data.get("rt_ld_gain_source", "")).lower()
        if gain_source not in {"full", "xls_v2"}:
            raise ValueError("RT L-D outer evaluator requires gain source full or xls_v2")
        rt_ld_manifest = manifest.get("rt_ld")
        if not isinstance(rt_ld_manifest, Mapping):
            raise ValueError("RT L-D outer evaluator requires the fit's dual-carrier manifest")
        if rt_ld_manifest.get("identity_carrier") != "aligned_full_afc4":
            raise ValueError("RT L-D outer evaluator refused a non-Full identity carrier")
        expected_gain = "aligned_full_afc4" if gain_source == "full" else "strong_xls_v2"
        if rt_ld_manifest.get("gain_carrier") != expected_gain:
            raise ValueError("RT L-D outer evaluator gain source disagrees with fit manifest")
        if rt_ld_manifest.get("identity_never_receives_xls_v2") is not True:
            raise ValueError("RT L-D fit manifest does not isolate XLSv2 from identity")
        target_dataset = RtLdDualCarrierDataset(
            target_dataset,
            gain_source=gain_source,
            xls_v2_support_audit_path=str(cfg.data.get("xls_v2_support_audit_path", "")),
        )
    if target_split != split:
        raise RuntimeError("Outer target dataset partition changed between fit receipt and evaluator")
    state_after_target_carrier = model.state_dict()
    state_after_target_carrier_digest = _state_digest(state_after_target_carrier)
    if state_before_digest != state_after_target_carrier_digest:
        raise RuntimeError("Outer target carrier construction changed model state")
    target_sampler = SessionBatchSampler(target_dataset, int(cfg.data.batch_size), shuffle=False)
    if not target_sampler.session_batch_counts:
        raise RuntimeError("Outer target has no eligible post-M24 query windows")
    target_loader = DataLoader(
        target_dataset,
        batch_sampler=target_sampler,
        num_workers=0,
        pin_memory=False,
    )

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in target_loader:
            out = model.model_step(_move_batch(batch, torch_device))
            predictions.append(out["behavior_pred"].detach().cpu().numpy().reshape(-1, 2))
            targets.append(out["behavior_target"].detach().cpu().numpy().reshape(-1, 2))
    prediction_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    score = _r2_variance_weighted(prediction_array, target_array)
    state_after = model.state_dict()
    state_after_digest = _state_digest(state_after)
    if state_before_digest != state_after_digest:
        changed = [
            key
            for key, value in state_after.items()
            if not torch.equal(state_before[key], value.detach().cpu())
        ]
        raise RuntimeError(f"Outer target evaluation changed model state: {changed[:8]}")
    matched_query_window_identity = getattr(target_dataset, "query_window_audit", None)
    if not isinstance(matched_query_window_identity, Mapping):
        raise RuntimeError("Outer target dataset lacks the required matched query-window audit")

    checkpoint = Path(checkpoint_path).resolve()
    carrier_semantics = {
        "rt_sparse_endpoint_t4d": {"dense_velocity_carrier": False, "endpoint_direction_support": True, "support_label_used": True},
        "zero4": {"dense_velocity_carrier": False, "endpoint_direction_support": False, "support_label_used": False},
        "afc4_vel": {"dense_velocity_carrier": True, "endpoint_direction_support": False, "support_label_used": True},
    }.get(feature_group, {"dense_velocity_carrier": feature_group != "none", "endpoint_direction_support": False, "support_label_used": feature_group != "none"})
    result: dict[str, Any] = {
        "schema": "rt_clean_nested_loso_outer_eval_v1",
        "status": "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "selection_receipt_path": str(Path(selection_receipt_path).resolve()),
        "config_path": str(Path(config_path).resolve()),
        "fit_split_manifest": str(Path(split_manifest_path).resolve()),
        "data_dir": str(resolved_data_dir.resolve()),
        "run_id": str(cfg.run_id),
        "seed": int(cfg.seed),
        "arm": feature_group,
        "outer_loso_fold": fold,
        "outer_target_session": target_split.outer_target_session,
        "outer_target_path": str(target_path.resolve()),
        "inner_train_sessions": list(target_split.inner_train_sessions),
        "inner_validation_session": target_split.inner_validation_session,
        "query_start_trial": int(cfg.data.query_start_trial),
        "window_size": int(cfg.data.window_size),
        "query_windows_evaluated": int(prediction_array.shape[0]),
        "r2_variance_weighted": score,
        "target_support_calibration_velocity_used": carrier_semantics["dense_velocity_carrier"],
        "target_support_calibration_labels_used": carrier_semantics["support_label_used"],
        "carrier_semantics": carrier_semantics,
        "target_query_labels_used_for_scoring_only": True,
        "target_query_labels_used_for_calibration": False,
        "target_query_labels_used_for_normalization": False,
        "target_query_labels_used_for_checkpoint_selection": False,
        "target_backpropagation": False,
        "optimizer_present": False,
        "model_training_mode": False,
        "model_state_sha256_before": state_before_digest,
        "model_state_sha256_after_target_carrier": state_after_target_carrier_digest,
        "model_state_sha256_after": state_after_digest,
        "model_state_unchanged": True,
        "model_state_three_point_unchanged": True,
        "matched_query_window_identity": matched_query_window_identity,
        "normalizer_fit_scope": manifest.get("source_only_normalizer", {}).get("fit_scope")
        if isinstance(manifest.get("source_only_normalizer"), Mapping)
        else "none_zero4",
    }
    if feature_group == "rt_sparse_endpoint_t4d":
        result["outer_t4d_access_audit"] = target_dataset.t4d_access_audits
    if is_rt_ld:
        result["rt_ld"] = {
            "identity_carrier": "aligned_full_afc4",
            "gain_carrier": "aligned_full_afc4" if str(cfg.data.rt_ld_gain_source).lower() == "full" else "strong_xls_v2",
            "gain_input_only": True,
            "identity_never_receives_xls_v2": True,
            "xls_v2_support_audit_sha256": manifest["rt_ld"].get("xls_v2_support_audit_sha256"),
        }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite outer-evaluation receipt: {destination}")
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="resolved config.yaml from the clean fit")
    parser.add_argument("--checkpoint", type=Path, required=True, help="selected inner-validation checkpoint")
    parser.add_argument("--split-manifest", type=Path, required=True, help="fit artifact split_manifest.json")
    parser.add_argument("--selection-receipt", type=Path, required=True, help="fit-end selection receipt")
    parser.add_argument("--output", type=Path, required=True, help="new outer-evaluation receipt path")
    parser.add_argument("--outer-fold", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="evaluation device; cpu is the safe default")
    args = parser.parse_args()
    result = evaluate_outer_target(
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
