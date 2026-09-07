#!/usr/bin/env python3
"""Strict fold-0 post-hoc headroom audit for the normalized H1 M=4 carrier.

This is deliberately *not* a new H1 experiment or a model-selection rule.
It evaluates the already-seen public held-in-calibration fold-0 records once
more, with the two immutable epoch-49 V2 checkpoints, and forms a fixed
half-and-half output ensemble:

    blend = base + 0.5 * (joint - base)

The coefficient is a protocol constant, not a fitted hyperparameter.  The
only optional optimisation reported here is an explicitly labelled oracle
diagnostic computed after opening this already-seen fold; it never controls a
pass/fail decision and cannot authorize target, minival, formal, or EvalAI
use.  No training, checkpoint write, or checkpoint mutation is permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.h1_m4_eb_normalized_v2_evaluate import (
    _instantiate_model,
    _validate_resolved_config,
)
from src.data.h1_m4_eb_normalized_v2 import (
    H1M4EBNormalizedV2DataModule,
    H1M4EBNormalizedV2StrictTargetDataset,
)
from src.data.h1_m4_eb_pilot import (
    H1_M4_FOLD0_TARGET,
    load_target_records,
    validate_target_receipt_binding,
)
from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    array_sha256,
    assert_immutable_receipt,
    assert_state_immutable,
    canonical_sha256,
    immutable_mode_0444,
    load_and_validate_terminal_checkpoint,
    reject_target_or_heldout_scope,
    sha256_file,
    state_hash,
    validate_paired_checkpoint_bindings,
)


AUDIT_SCHEMA = "h1_m4_carrier_complementary_ensemble_fold0_posthoc_headroom_audit_v1"
AUDIT_STATUS = "COMPLETE_H1_M4_CARRIER_COMPLEMENTARY_ENSEMBLE_POSTHOC_HEADROOM_AUDIT"
FIXED_ALPHA = 0.5
INTERVENTIONS = ("full", "zero", "row", "label")


class CarrierComplementaryAuditError(NormalizedV2ContractError):
    """Fail-closed violation of the fixed post-hoc ensemble protocol."""


def require_fixed_alpha(alpha: float = FIXED_ALPHA) -> float:
    """Return exactly 0.5, rejecting all attempted post-hoc tuning."""

    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise CarrierComplementaryAuditError("ensemble alpha must be the fixed literal 0.5") from exc
    if not math.isfinite(value) or value != FIXED_ALPHA:
        raise CarrierComplementaryAuditError(
            f"post-hoc ensemble alpha is frozen at {FIXED_ALPHA}; got {alpha!r}"
        )
    return FIXED_ALPHA


def fixed_half_blend(base_prediction: np.ndarray, joint_prediction: np.ndarray, *, alpha: float = FIXED_ALPHA) -> np.ndarray:
    """Form the fixed blend without allowing an alpha search."""

    weight = require_fixed_alpha(alpha)
    base = np.asarray(base_prediction, dtype=np.float64)
    joint = np.asarray(joint_prediction, dtype=np.float64)
    if base.shape != joint.shape or base.ndim != 2 or base.shape[0] == 0:
        raise CarrierComplementaryAuditError(
            f"base/joint prediction shape mismatch or empty arrays: {base.shape} vs {joint.shape}"
        )
    if not np.isfinite(base).all() or not np.isfinite(joint).all():
        raise CarrierComplementaryAuditError("base/joint predictions must be finite")
    return np.asarray(base + weight * (joint - base), dtype=np.float64)


def _r2_from_arrays(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 2 or truth.shape[0] == 0:
        raise CarrierComplementaryAuditError(
            f"R2 requires matching nonempty [windows,outputs] arrays, got {truth.shape} vs {estimate.shape}"
        )
    if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise CarrierComplementaryAuditError("R2 receives a nonfinite target or prediction")
    sse = float(np.square(truth - estimate).sum())
    tss = float(np.square(truth - truth.mean(axis=0, keepdims=True)).sum())
    if not math.isfinite(sse) or not math.isfinite(tss) or tss <= 0.0:
        raise CarrierComplementaryAuditError("R2 is undefined because the target variance is nonpositive")
    return {"r2": float(1.0 - sse / tss), "sse": sse, "tss": tss}


def r2_summary(target: np.ndarray, prediction: np.ndarray, sessions: Sequence[str]) -> dict[str, Any]:
    """Report global-pool and independently centered session-VW R2 values."""

    truth = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    names = tuple(str(name) for name in sessions)
    if truth.shape != estimate.shape or truth.ndim != 2 or len(names) != truth.shape[0]:
        raise CarrierComplementaryAuditError("target/prediction/session order mismatch in R2 summary")
    if set(names) != set(H1_M4_FOLD0_TARGET):
        raise CarrierComplementaryAuditError(
            f"R2 summary must contain precisely the two fold-0 target recordings, got {sorted(set(names))}"
        )
    pooled = _r2_from_arrays(truth, estimate)
    per_session: dict[str, dict[str, float | int]] = {}
    total_sse = 0.0
    total_tss = 0.0
    for name in H1_M4_FOLD0_TARGET:
        index = np.asarray([item == name for item in names], dtype=bool)
        if not bool(index.any()):
            raise CarrierComplementaryAuditError(f"missing fold-0 target recording {name}")
        item = _r2_from_arrays(truth[index], estimate[index])
        per_session[name] = {"samples": int(index.sum()), **item}
        total_sse += item["sse"]
        total_tss += item["tss"]
    if total_tss <= 0.0 or not math.isfinite(total_sse) or not math.isfinite(total_tss):
        raise CarrierComplementaryAuditError("per-session variance-weighted R2 is undefined")
    return {
        "pooled": pooled,
        "per_session_variance_weighted": {
            "r2": float(1.0 - total_sse / total_tss),
            "sse": float(total_sse),
            "tss": float(total_tss),
            "weighting": "sum session SSE / sum independently centered session TSS",
        },
        "per_session": per_session,
    }


def _metric_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    return {
        "pooled_r2": float(left["pooled"]["r2"] - right["pooled"]["r2"]),
        "per_session_variance_weighted_r2": float(
            left["per_session_variance_weighted"]["r2"] - right["per_session_variance_weighted"]["r2"]
        ),
    }


def oracle_alpha_diagnostic(base_prediction: np.ndarray, joint_prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Post-hoc, never-selecting least-squares alpha diagnostic for one arm."""

    base = np.asarray(base_prediction, dtype=np.float64)
    joint = np.asarray(joint_prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if base.shape != joint.shape or base.shape != truth.shape:
        raise CarrierComplementaryAuditError("oracle alpha arrays have incompatible shapes")
    direction = joint - base
    denominator = float(np.square(direction).sum())
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise CarrierComplementaryAuditError("oracle alpha is undefined for an identically equal base/joint pair")
    alpha = float(np.sum(direction * (truth - base)) / denominator)
    prediction = base + alpha * direction
    return {
        "posthoc_diagnostic_only": True,
        "must_not_control_pass_fail_or_candidate_selection": True,
        "unconstrained_least_squares_alpha": alpha,
        "pooled": _r2_from_arrays(truth, prediction),
        "prediction_sha256": array_sha256(prediction),
    }


def assert_audit_scope_safe(*paths: str | Path) -> None:
    """Fail closed on every supplied artifact path carrying forbidden scope words."""

    for path in paths:
        reject_target_or_heldout_scope(Path(path).resolve())


def _write_atomic_immutable_json(path: str | Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    """Write one 0444 receipt by atomic link, refusing any replacement race."""

    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite post-hoc audit artifact {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, output)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite post-hoc audit artifact {output}") from None
        finally:
            temporary.unlink(missing_ok=True)
        directory_fd = os.open(str(output.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if not immutable_mode_0444(output):
        raise CarrierComplementaryAuditError(f"atomic receipt mode was not exactly 0444: {output}")
    return output, hashlib.sha256(encoded).hexdigest()


def _predict(model, dataset: H1M4EBNormalizedV2StrictTargetDataset, device: torch.device) -> dict[str, Any]:
    """Run one frozen model/view and retain exact query-window order."""

    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sessions: list[str] = []
    batch_sizes: list[int] = []
    before = state_hash(model.state_dict())
    with torch.no_grad():
        for neural, target, identity, session_name, carrier in loader:
            output = model(
                neural.to(device=device, dtype=torch.float32),
                calib_trialized_neural_features=identity.to(device=device, dtype=torch.float32),
                carrier=carrier.to(device=device, dtype=torch.float32),
            )
            if model.hparams.decode_last_timestep_only:
                output = output[:, -1:, :]
                target = target[:, -1:, :]
            if model.hparams.predict_scaled_behavior:
                output = output / model.hparams.behavior_scaling_factor
            predictions.append(np.asarray(output[:, -1, :].detach().cpu().numpy(), dtype=np.float64))
            targets.append(np.asarray(target[:, -1, :].detach().cpu().numpy(), dtype=np.float64))
            sessions.extend(str(name) for name in session_name)
            batch_sizes.append(int(output.shape[0]))
    after = state_hash(model.state_dict())
    assert_state_immutable(before, after, f"carrier-complementary/{model.pilot_arm}/{dataset.intervention}")
    if not batch_sizes or sum(batch_sizes) != len(dataset):
        raise CarrierComplementaryAuditError("audit prediction omitted query windows")
    if batch_sizes[-1] != (len(dataset) % 32 or 32):
        raise CarrierComplementaryAuditError("audit prediction discarded the final remainder batch")
    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    names = tuple(sessions)
    if prediction.shape != target.shape or len(names) != prediction.shape[0]:
        raise CarrierComplementaryAuditError("audit prediction concatenation shape/order drift")
    counts = {name: names.count(name) for name in H1_M4_FOLD0_TARGET}
    if set(names) != set(H1_M4_FOLD0_TARGET) or sum(counts.values()) != len(dataset):
        raise CarrierComplementaryAuditError("audit prediction omitted or added a fold-0 target recording")
    return {
        "prediction": prediction,
        "target": target,
        "sessions": names,
        "samples": int(len(dataset)),
        "session_samples": counts,
        "batches": int(len(batch_sizes)),
        "last_batch_size": int(batch_sizes[-1]),
        "prediction_sha256": array_sha256(prediction),
        "target_sha256": array_sha256(target),
        "session_order_sha256": canonical_sha256(list(names)),
        "model_state_sha256_before": before,
        "model_state_sha256_after": after,
        "state_immutable": before == after,
        "query_window_indices_sha256": dataset.window_indices_sha256,
    }


def _assert_aligned(reference: Mapping[str, Any], candidate: Mapping[str, Any], name: str) -> None:
    if reference["samples"] != candidate["samples"]:
        raise CarrierComplementaryAuditError(f"{name} changed target sample count")
    if reference["sessions"] != candidate["sessions"]:
        raise CarrierComplementaryAuditError(f"{name} changed target query session order")
    if not np.array_equal(reference["target"], candidate["target"]):
        raise CarrierComplementaryAuditError(f"{name} changed target values")
    if reference["query_window_indices_sha256"] != candidate["query_window_indices_sha256"]:
        raise CarrierComplementaryAuditError(f"{name} changed query windows")


def _read_reference_terminal_receipt(path: str | Path) -> dict[str, Any]:
    receipt = assert_immutable_receipt(path)
    if receipt.get("schema") != "h1_m4_eb_normalized_v2_fold0_terminal_gate_v1":
        raise CarrierComplementaryAuditError("reference terminal receipt schema is not normalized V2 fold-0")
    if receipt.get("fold_date") != "19250101":
        raise CarrierComplementaryAuditError("reference terminal receipt fold date drift")
    scope = receipt.get("data_scope", {})
    if not isinstance(scope, Mapping) or any(bool(scope.get(key)) for key in ("minival_opened", "heldout_opened", "evalai_opened", "formal_heldout_opened")):
        raise CarrierComplementaryAuditError("reference terminal receipt exceeded permitted held-in scope")
    return receipt


def _validate_reference_terminal_binding(
    reference: Mapping[str, Any],
    paired: Mapping[str, str],
    *,
    base_checkpoint_sha256: str,
    joint_checkpoint_sha256: str,
    base_config_sha256: str,
    joint_config_sha256: str,
) -> None:
    """Prove this post-hoc view is of the exact terminal V2 checkpoint pair."""

    checkpoints = reference.get("checkpoints")
    target = reference.get("target")
    normalizer = reference.get("normalizer")
    if not isinstance(checkpoints, Mapping) or not isinstance(target, Mapping) or not isinstance(normalizer, Mapping):
        raise CarrierComplementaryAuditError("reference terminal receipt has malformed bindings")
    for arm, checkpoint_sha, config_sha in (
        ("base", base_checkpoint_sha256, base_config_sha256),
        ("joint", joint_checkpoint_sha256, joint_config_sha256),
    ):
        row = checkpoints.get(arm)
        if not isinstance(row, Mapping) or row.get("sha256") != checkpoint_sha or row.get("config_sha256") != config_sha:
            raise CarrierComplementaryAuditError(f"reference terminal receipt does not bind the current {arm} artifact")
    if reference.get("source_manifest_sha256") != paired["source_manifest_sha256"]:
        raise CarrierComplementaryAuditError("reference terminal source manifest differs from current checkpoint pair")
    if normalizer.get("normalizer_sha256") != paired["normalizer_sha256"]:
        raise CarrierComplementaryAuditError("reference terminal normalizer differs from current checkpoint pair")
    if tuple(target.get("sessions", ())) != H1_M4_FOLD0_TARGET:
        raise CarrierComplementaryAuditError("reference terminal receipt target sessions are not the fold-0 pair")
    for key in ("source_cache_sha256", "normalized_cache_sha256"):
        if checkpoints.get("paired", {}).get(key) != paired[key]:
            raise CarrierComplementaryAuditError(f"reference terminal paired binding differs at {key}")


def run_posthoc_headroom_audit(
    *,
    data_dir: str | Path,
    raw_receipt_path: str | Path,
    eb_receipt_path: str | Path,
    shared_cache_dir: str | Path,
    base_checkpoint_path: str | Path,
    joint_checkpoint_path: str | Path,
    base_config_path: str | Path,
    joint_config_path: str | Path,
    reference_terminal_receipt_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run the one fixed, already-seen-fold post-hoc ensemble audit."""

    if device not in {"cuda", "cpu"}:
        raise ValueError("carrier-complementary audit device must be cuda or cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("carrier-complementary audit requested cuda but CUDA is unavailable")
    require_fixed_alpha()
    paths = (
        data_dir,
        raw_receipt_path,
        eb_receipt_path,
        shared_cache_dir,
        base_checkpoint_path,
        joint_checkpoint_path,
        base_config_path,
        joint_config_path,
        reference_terminal_receipt_path,
        output_path,
    )
    assert_audit_scope_safe(*paths)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite post-hoc audit artifact {output}")
    device_value = torch.device(device)
    base_config_path = Path(base_config_path).resolve()
    joint_config_path = Path(joint_config_path).resolve()

    # Both config/checkpoint pairs and the earlier terminal receipt bind before
    # any NWB file is opened.  Source setup runs before the two target records.
    base_config = _validate_resolved_config(base_config_path, "base")
    joint_config = _validate_resolved_config(joint_config_path, "joint")
    base_checkpoint, base_meta = load_and_validate_terminal_checkpoint(
        base_checkpoint_path, base_config_path, expected_arm="base"
    )
    joint_checkpoint, joint_meta = load_and_validate_terminal_checkpoint(
        joint_checkpoint_path, joint_config_path, expected_arm="joint"
    )
    paired = validate_paired_checkpoint_bindings(base_meta, joint_meta)
    reference_terminal = _read_reference_terminal_receipt(reference_terminal_receipt_path)
    base_checkpoint_sha256 = sha256_file(base_checkpoint_path)
    joint_checkpoint_sha256 = sha256_file(joint_checkpoint_path)
    base_config_sha256 = sha256_file(base_config_path)
    joint_config_sha256 = sha256_file(joint_config_path)
    _validate_reference_terminal_binding(
        reference_terminal,
        paired,
        base_checkpoint_sha256=base_checkpoint_sha256,
        joint_checkpoint_sha256=joint_checkpoint_sha256,
        base_config_sha256=base_config_sha256,
        joint_config_sha256=joint_config_sha256,
    )

    source_module = H1M4EBNormalizedV2DataModule(
        task="h1",
        data_dir=str(Path(data_dir).resolve()),
        raw_receipt_path=str(Path(raw_receipt_path).resolve()),
        eb_receipt_path=str(Path(eb_receipt_path).resolve()),
        cache_dir=str(Path(shared_cache_dir).resolve()),
    )
    source_module.setup("fit")
    source_manifest = source_module.pilot_manifest()
    if source_module.pilot_manifest_sha256 != paired["source_manifest_sha256"]:
        raise CarrierComplementaryAuditError("runtime source manifest differs from both terminal checkpoints")
    if source_module.normalizer.normalizer_sha256 != paired["normalizer_sha256"]:
        raise CarrierComplementaryAuditError("runtime normalizer differs from both terminal checkpoints")
    if source_manifest["carrier_cache_sha256"] != paired["source_cache_sha256"]:
        raise CarrierComplementaryAuditError("runtime raw source carrier cache differs from both terminal checkpoints")
    if source_manifest["normalized_cache_sha256"] != paired["normalized_cache_sha256"]:
        raise CarrierComplementaryAuditError("runtime normalized source cache differs from both terminal checkpoints")
    if source_manifest.get("target_nwb_opened_during_training_setup") is not False:
        raise CarrierComplementaryAuditError("source setup opened a target NWB")
    if source_manifest.get("minival_or_heldout_enumerated") is not False:
        raise CarrierComplementaryAuditError("source setup enumerated forbidden evaluation scope")

    base_model = _instantiate_model(base_config, base_checkpoint, device_value)
    joint_model = _instantiate_model(joint_config, joint_checkpoint, device_value)
    del base_checkpoint, joint_checkpoint

    # This is intentionally the first target data access and uses only the two
    # public held-in-calibration fold-0 records.
    target_records = load_target_records(data_dir)
    validate_target_receipt_binding(target_records, source_module.plan, raw_receipt_path, eb_receipt_path)
    full = H1M4EBNormalizedV2StrictTargetDataset(
        target_records, source_module.plan, source_module.normalizer, "full"
    )
    variants = {name: full.with_intervention(name) for name in INTERVENTIONS}
    support_hashes = full.support_and_carrier_hashes()
    for name, dataset in variants.items():
        if dataset.window_indices_sha256 != full.window_indices_sha256:
            raise CarrierComplementaryAuditError(f"{name} intervention changed query window order")
        if dataset.support_and_carrier_hashes() != support_hashes:
            raise CarrierComplementaryAuditError(f"{name} intervention changed support or carrier binding")

    base = _predict(base_model, variants["full"], device_value)
    joint = {name: _predict(joint_model, variants[name], device_value) for name in INTERVENTIONS}
    for name, item in joint.items():
        _assert_aligned(base, item, f"joint/{name}")

    target = np.asarray(base["target"], dtype=np.float64)
    sessions = tuple(base["sessions"])
    metrics_base = r2_summary(target, base["prediction"], sessions)
    metrics_joint = {name: r2_summary(target, item["prediction"], sessions) for name, item in joint.items()}
    blend_predictions = {
        name: fixed_half_blend(base["prediction"], item["prediction"]) for name, item in joint.items()
    }
    metrics_blend = {name: r2_summary(target, prediction, sessions) for name, prediction in blend_predictions.items()}
    fixed_full = metrics_blend["full"]
    deltas = {
        "full_blend_minus_base": _metric_delta(fixed_full, metrics_base),
        "full_blend_minus_zero_blend": _metric_delta(fixed_full, metrics_blend["zero"]),
        "full_blend_minus_row_blend": _metric_delta(fixed_full, metrics_blend["row"]),
        "full_blend_minus_label_blend": _metric_delta(fixed_full, metrics_blend["label"]),
    }
    oracle = oracle_alpha_diagnostic(base["prediction"], joint["full"]["prediction"], target)

    receipt = {
        "schema": AUDIT_SCHEMA,
        "status": AUDIT_STATUS,
        "claim_status": (
            "post-hoc headroom diagnostic on an already-seen public held-in fold; "
            "not confirmatory, not held-out, not formal, not an authorization to claim a carrier gain"
        ),
        "training_or_checkpoint_update_performed": False,
        "fold_date": "19250101",
        "evaluation_device": str(device_value),
        "fixed_blend": {
            "formula": "base + 0.5 * (joint - base)",
            "alpha": FIXED_ALPHA,
            "alpha_optimized": False,
            "alpha_cli_override_available": False,
        },
        "checkpoint_binding_completed_before_target_open": True,
        "checkpoints": {
            "base": {
                "path": str(Path(base_checkpoint_path).resolve()),
                "sha256": base_checkpoint_sha256,
                "config_path": str(base_config_path),
                "config_sha256": base_config_sha256,
                "metadata": base_meta,
            },
            "joint": {
                "path": str(Path(joint_checkpoint_path).resolve()),
                "sha256": joint_checkpoint_sha256,
                "config_path": str(joint_config_path),
                "config_sha256": joint_config_sha256,
                "metadata": joint_meta,
            },
            "paired": paired,
        },
        "reference_seen_fold_terminal_receipt": {
            "path": str(Path(reference_terminal_receipt_path).resolve()),
            "sha256": sha256_file(reference_terminal_receipt_path),
            "status": reference_terminal.get("status"),
            "schema": reference_terminal.get("schema"),
        },
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_module.pilot_manifest_sha256,
        "normalizer": source_module.normalizer.manifest,
        "target": {
            "sessions": list(H1_M4_FOLD0_TARGET),
            "files": {name: target_records[name].input_sha256 for name in H1_M4_FOLD0_TARGET},
            "normalized_support_and_carrier_hashes": support_hashes,
            "strict_query_window_indices_sha256": full.window_indices_sha256,
            "all_query_histories_start_at_or_after_fifth_trial": True,
            "remainder_preserved": True,
            "samples": int(base["samples"]),
            "session_samples": base["session_samples"],
            "target_values_sha256": base["target_sha256"],
            "session_order_sha256": base["session_order_sha256"],
        },
        "prediction_provenance": {
            "base": {key: value for key, value in base.items() if key not in {"prediction", "target", "sessions"}},
            "joint": {
                name: {key: value for key, value in item.items() if key not in {"prediction", "target", "sessions"}}
                for name, item in joint.items()
            },
            "fixed_blend_prediction_sha256": {name: array_sha256(value) for name, value in blend_predictions.items()},
        },
        "metrics": {
            "definition": {
                "pooled": "1 - sum squared error / globally centered target sum of squares across both recordings",
                "per_session_variance_weighted": "1 - sum session squared error / sum independently centered session target sum of squares",
            },
            "base": metrics_base,
            "joint": metrics_joint,
            "fixed_half_blend": metrics_blend,
            "fixed_half_blend_deltas": deltas,
            "oracle_full_alpha_posthoc_diagnostic": oracle,
        },
        "data_scope": {
            "opened": "13 public held-in-calib NWBs only (11 source, then 2 already-seen fold-0 target)",
            "minival_opened": False,
            "heldout_opened": False,
            "evalai_opened": False,
            "formal_heldout_opened": False,
            "formal_submission_created": False,
        },
    }
    written, digest = _write_atomic_immutable_json(output, receipt)
    receipt["receipt_path"] = str(written)
    receipt["receipt_sha256"] = digest
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data/000954")
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--eb-receipt", type=Path, required=True)
    parser.add_argument("--shared-cache-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--joint-config", type=Path, required=True)
    parser.add_argument("--reference-terminal-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    result = run_posthoc_headroom_audit(
        data_dir=args.data_dir,
        raw_receipt_path=args.raw_receipt,
        eb_receipt_path=args.eb_receipt,
        shared_cache_dir=args.shared_cache_dir,
        base_checkpoint_path=args.base_checkpoint,
        joint_checkpoint_path=args.joint_checkpoint,
        base_config_path=args.base_config,
        joint_config_path=args.joint_config,
        reference_terminal_receipt_path=args.reference_terminal_receipt,
        output_path=args.output,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "receipt": result["receipt_path"],
                "sha256": result["receipt_sha256"],
                "fixed_alpha": result["fixed_blend"]["alpha"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
