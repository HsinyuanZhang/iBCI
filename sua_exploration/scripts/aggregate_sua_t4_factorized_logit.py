#!/usr/bin/env python3
"""Fail-closed aggregation for the selected-T4 attention-logit residual screen.

This is deliberately an *auditor*, not a selector.  It accepts exactly the frozen
seed-42, 12-epoch validation round described in
``SUA_T4_FACTORIZED_LOGIT_RESIDUAL_PROTOCOL.md`` and refuses incomplete or
provenance-drifting input.  It never opens DANDI data and never reads a formal-test
NWB.  A numerical gate failure is written as ``gate: fail``; a provenance failure
raises before either output artifact is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "sua_exploration/results/sua_t4_factorized_logit_residual_v1"
BASELINE = ROOT / "sua_exploration/results/sua_t4_confidence_film_v1/t4_continuation_m50_s42.json"
ANCHOR = ROOT / "sua_exploration/checkpoints/sua_t4_confidence_film_v1_t4m50_dandi688_co_s42/epoch_ckpts/epoch_011.ckpt"
TEACHER = ROOT / "sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
MANIFEST = ROOT / "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
PROTOCOL = ROOT / "sua_exploration/docs/SUA_T4_FACTORIZED_LOGIT_RESIDUAL_PROTOCOL.md"
PREFLIGHT = RESULT_DIR / "preflight.json"

ANCHOR_SHA = "cf533e7cd97801d53985383b37f3fa1ff72fb6c385eaeb8c81273c3fa128273d"
TEACHER_SHA = "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d"
MANIFEST_SHA = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
EPOCHS = tuple(range(5, 13))
EXPECTED_TRAINABLE = (
    "t4_logit_residual.query_factors",
    "t4_logit_residual.unit_projection.weight",
)
EXTRA_STATE_KEYS = tuple(f"student.{name}" for name in EXPECTED_TRAINABLE)
ARM_SPEC = {
    "aligned_logit": ("aligned", "attention_logit", None),
    "shuffled_logit": ("shuffled", "attention_logit", 42),
    "additive_control": ("aligned", "additive_control", None),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fail(message: str) -> None:
    raise ValueError(message)


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric") from exc
    if not math.isfinite(result):
        _fail(f"{name} is not finite")
    return result


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and torch.equal(a, b)


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Match the residual component's canonical tensor-state receipt."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"missing required artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON: {path}") from exc
    if not isinstance(payload, dict):
        _fail(f"JSON root must be an object: {path}")
    return payload


@dataclass(frozen=True)
class ScoredResult:
    arm: str
    source_path: Path
    result: dict[str, Any]
    metadata_path: Path
    metadata: dict[str, Any]
    sessions: tuple[str, ...]
    score: float
    session_means: dict[str, float]
    checkpoint_receipts: dict[str, Any]


def rederive_result_scores(result: Mapping[str, Any], *, label: str) -> tuple[tuple[str, ...], float, dict[str, float]]:
    """Recompute all score quantities from epoch/session leaves; trust no summary."""
    if list(result.get("epoch_list", ())) != list(EPOCHS):
        _fail(f"{label}: epoch_list must be exactly {list(EPOCHS)}")
    protocol = result.get("protocol")
    if not isinstance(protocol, Mapping) or list(protocol.get("epoch_window", ())) != list(EPOCHS):
        _fail(f"{label}: protocol epoch window drift")
    per_epoch = result.get("per_epoch")
    if not isinstance(per_epoch, Mapping) or set(per_epoch) != {str(x) for x in EPOCHS}:
        _fail(f"{label}: per_epoch must contain exactly the frozen epochs")
    sessions: tuple[str, ...] | None = None
    epoch_means: list[float] = []
    values_by_session: dict[str, list[float]] = {}
    for epoch in EPOCHS:
        row = per_epoch[str(epoch)]
        if not isinstance(row, Mapping) or not isinstance(row.get("per_session_r2"), Mapping):
            _fail(f"{label}: malformed per_epoch[{epoch}]")
        row_sessions = tuple(sorted(row["per_session_r2"]))
        if sessions is None:
            sessions = row_sessions
            if len(sessions) != 6:
                _fail(f"{label}: must report exactly six validation sessions")
            values_by_session = {s: [] for s in sessions}
        elif row_sessions != sessions:
            _fail(f"{label}: validation session set changes by epoch")
        values = [_finite_float(row["per_session_r2"][s], f"{label} epoch {epoch} {s}") for s in sessions]
        recomputed_mean = float(np.mean(values))
        stored_mean = _finite_float(row.get("mean_r2"), f"{label} epoch {epoch} mean_r2")
        if not math.isclose(recomputed_mean, stored_mean, rel_tol=0.0, abs_tol=1e-10):
            _fail(f"{label}: epoch {epoch} mean_r2 does not equal the six-session mean")
        stored_summary = result.get("per_epoch_mean_r2", {})
        if not isinstance(stored_summary, Mapping) or not math.isclose(
            stored_mean, _finite_float(stored_summary.get(str(epoch)), f"{label} per_epoch_mean_r2[{epoch}]"), rel_tol=0.0, abs_tol=1e-10
        ):
            _fail(f"{label}: per_epoch_mean_r2 drift at epoch {epoch}")
        epoch_means.append(recomputed_mean)
        for session, value in zip(sessions, values):
            values_by_session[session].append(value)
    assert sessions is not None
    score = float(np.mean(epoch_means))
    stored_score = _finite_float(result.get("variant_score"), f"{label} variant_score")
    if not math.isclose(score, stored_score, rel_tol=0.0, abs_tol=1e-10):
        _fail(f"{label}: variant_score is not the rederived epoch-window mean")
    return sessions, score, {s: float(np.mean(v)) for s, v in values_by_session.items()}


def _validate_common_result(result: Mapping[str, Any], *, label: str, expected_sessions: tuple[str, ...] | None) -> None:
    required = {
        "schema_version": 1,
        "generated_by": "eval_epoch_window_generic_dandi688.py",
        "purpose": "epoch_window_deterministic_checkpoint_selection",
        "seed": 42, "variant": "B3S", "task": "CO", "signal_view": "sua",
        "split_counts": [27, 6, 6], "max_units_exclusive": 100,
        "no_test_files_evaluated": True, "uses_backward_gradients": False,
        "uses_behavior_labels_for_weight_updates": False,
        "calibration_features_use_behavior_labels": True,
        "calibration_trial_selection_uses_behavior_labels": False,
        "calibration_feature_label_scope": "chronological_rewarded_trials[0:50]",
        "checkpoint_selection_rule": "pre_declared_fixed_epoch_window_no_argmax",
        "teacher_ckpt_sha256": TEACHER_SHA,
        "train_val_manifest_sha256": MANIFEST_SHA,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            _fail(f"{label}: {key}={result.get(key)!r}, expected {expected!r}")
    protocol = result.get("protocol")
    if not isinstance(protocol, Mapping):
        _fail(f"{label}: missing protocol receipt")
    protocol_expected = {
        "total_epochs": 12, "burn_in_epochs": 4, "selection_mode": "first",
        "calibration_n": 30, "train_activity_calibration_n": 30,
        "evaluation_forward_calibration_n": 30, "label_feature_calibration_n": 50,
        "pool_size": 50,
    }
    for key, expected in protocol_expected.items():
        if protocol.get(key) != expected:
            _fail(f"{label}: protocol.{key} drift")
    sessions, _, _ = rederive_result_scores(result, label=label)
    if expected_sessions is not None and sessions != expected_sessions:
        _fail(f"{label}: result does not use exactly the baseline six validation sessions")
    split = result.get("session_splits")
    if not isinstance(split, Mapping) or tuple(sorted(split.get("val", ()))) != sessions:
        _fail(f"{label}: session_splits.val receipt drift")


def _metadata_for(result: Mapping[str, Any], *, label: str) -> tuple[Path, dict[str, Any]]:
    path = Path(str(result.get("run_metadata_path", "")))
    if not path.is_file():
        _fail(f"{label}: run_metadata_path missing")
    if sha256_file(path) != result.get("run_metadata_sha256"):
        _fail(f"{label}: run_metadata SHA-256 drift")
    metadata = _load_json(path)
    if metadata.get("status") != "completed":
        _fail(f"{label}: run metadata status must be completed")
    if metadata.get("held_out_test_evaluated") is not False:
        _fail(f"{label}: held-out/formal test receipt is not false")
    # These three independent seals make a validation artifact invalid if formal
    # NWBs were put into a dataloader even if no final metric was written.
    session_files = metadata.get("session_files")
    if not isinstance(session_files, Mapping) or session_files.get("test") != []:
        _fail(f"{label}: session_files.test must be an empty list")
    fit_contract = metadata.get("trainer_fit_validation_loader_contract")
    if not isinstance(fit_contract, Mapping) or fit_contract.get("formal_test_sessions_loaded_during_fit") is not False:
        _fail(f"{label}: fit-loader formal-test isolation receipt drift")
    held_out_protocol = metadata.get("held_out_evaluation_protocol")
    expected_seal = {
        "name": "frozen_gradient_free_streaming_calibration",
        "held_out_test_evaluated": False,
        "backward_gradients_on_held_out_sessions": False,
        "held_out_behavior_labels_used_for_updates": False,
        "calibration_input": "held-out session spikes only",
        "model_weights": "best checkpoint must be frozen during held-out evaluation",
        "formal_test_entrypoint": "scripts/eval_adaptation_dandi688.py",
    }
    if not isinstance(held_out_protocol, Mapping):
        _fail(f"{label}: missing held-out evaluation isolation seal")
    for key, expected in expected_seal.items():
        if held_out_protocol.get(key) != expected:
            _fail(f"{label}: held-out evaluation isolation seal drift at {key}")
    if metadata.get("teacher_sha256") != TEACHER_SHA:
        _fail(f"{label}: teacher SHA in metadata drift")
    if metadata.get("train_val_manifest_sha256") != MANIFEST_SHA:
        _fail(f"{label}: manifest SHA in metadata drift")
    if metadata.get("session_splits") != result.get("session_splits"):
        _fail(f"{label}: metadata/result split receipt disagree")
    result_run_dir = Path(str(result.get("run_dir", "")))
    metadata_run_dir = Path(str(metadata.get("output_dir", "")))
    if not result_run_dir.is_absolute() or not metadata_run_dir.is_absolute() or result_run_dir != metadata_run_dir:
        _fail(f"{label}: result run_dir must exactly equal metadata.output_dir")
    training = metadata.get("training")
    if not isinstance(training, Mapping):
        _fail(f"{label}: missing training metadata")
    for key, expected in {"max_epochs": 12, "no_early_stopping": True, "checkpoint_every_epoch": True,
                          "calibration_n_trials": 30, "learning_rate": 1e-4, "loss_mode": "task_only",
                          "identity_mode": "calibrated", "deterministic": True}.items():
        if training.get(key) != expected:
            _fail(f"{label}: training.{key} drift")
    side = metadata.get("side_features")
    if not isinstance(side, Mapping) or side.get("group") != "t4" or side.get("pool_size") != 50 or side.get("side_dim") != 4:
        _fail(f"{label}: ordinary aligned T4 substrate receipt drift")
    if metadata.get("decoder_architecture", {}).get("mode") != "coupled":
        _fail(f"{label}: coupled decoder receipt drift")
    return path, metadata


def _validate_preflight(preflight: Mapping[str, Any]) -> None:
    if preflight.get("gate") != "pass" or preflight.get("no_dataset_opened") is not True or preflight.get("no_formal_test_opened") is not True:
        _fail("preflight is not a strict no-data/no-formal pass")
    if preflight.get("anchor", {}).get("sha256") != ANCHOR_SHA or preflight.get("teacher", {}).get("sha256") != TEACHER_SHA:
        _fail("preflight anchor/teacher receipt drift")
    if preflight.get("protocol", {}).get("sha256") != sha256_file(PROTOCOL):
        _fail("preflight protocol SHA drift")
    if preflight.get("trainable_parameter_count") != {arm: 48 for arm in ARM_SPEC}:
        _fail("preflight trainable count receipt drift")
    for arm in ARM_SPEC:
        if tuple(preflight.get("optimizer_trainable_names", {}).get(arm, ())) != EXPECTED_TRAINABLE:
            _fail(f"preflight optimizer whitelist drift for {arm}")
        checks = preflight.get("zero_and_cache_checks", {}).get(arm, {})
        if checks.get("inherited_coupled_bit_equal") is not True or checks.get("cached_on_the_fly_bit_equal") is not True or checks.get("zero_bias_nonzero_count") != 0:
            _fail(f"preflight exactness receipt drift for {arm}")
    cost = preflight.get("cost_receipt_n64", {})
    if cost.get("trainable_parameter_count") != 48 or cost.get("neuron_axis_quadratic_term") is not False:
        _fail("preflight cost receipt drift")
    if cost.get("persistent_additional_state", {}).get("bytes_fp32") != 2048:
        _fail("preflight persistent state receipt drift")
    if cost.get("calibration_only_unit_factor_macs") != 2048 or cost.get("online_increment", {}).get("query_unit_factor_macs") != 1024:
        _fail("preflight MAC receipt drift")


def _validate_arm_metadata(metadata: Mapping[str, Any], *, arm: str, preflight: Mapping[str, Any]) -> None:
    mode, interaction, permutation_seed = ARM_SPEC[arm]
    if metadata.get("encoder_warmstart_sha256") != ANCHOR_SHA:
        _fail(f"{arm}: selected anchor SHA drift in run metadata")
    receipt = metadata.get("t4_logit_residual")
    if not isinstance(receipt, Mapping):
        _fail(f"{arm}: missing logit-residual run receipt")
    expected = {"enabled": True, "residual_mode": mode, "interaction_mode": interaction,
                "rank": 8, "residual_permutation_seed": permutation_seed,
                "selected_t4_substrate_frozen": True,
                "optimizer_trainable_parameter_count": 48}
    for key, value in expected.items():
        if receipt.get(key) != value:
            _fail(f"{arm}: t4_logit_residual.{key} drift")
    if tuple(receipt.get("optimizer_trainable_parameter_names", ())) != EXPECTED_TRAINABLE:
        _fail(f"{arm}: residual optimizer whitelist drift")
    init = receipt.get("initialization_receipt", {})
    for key, value in {"rank": 8, "parameter_count": 48, "residual_mode": mode,
                       "interaction_mode": interaction, "residual_permutation_seed": permutation_seed,
                       "query_factors_zero_initialized": True, "backbone_frozen_for_residual_pilot": True,
                       "teacher_head_count": 64, "bias_shared_across_teacher_heads": True}.items():
        if init.get(key) != value:
            _fail(f"{arm}: initialization receipt {key} drift")
    if receipt.get("cost_receipt_reference_n64") != preflight.get("cost_receipt_n64"):
        _fail(f"{arm}: exact cost receipt differs from preflight")


def verify_checkpoint(
    checkpoint_path: Path,
    *, expected_sha: str,
    anchor_state: Mapping[str, torch.Tensor],
    arm: str,
    epoch: int,
) -> dict[str, Any]:
    """Require exactly two new factor tensors; all anchor tensors bit-match."""
    if not checkpoint_path.is_file():
        _fail(f"{arm} epoch {epoch}: checkpoint missing")
    if sha256_file(checkpoint_path) != expected_sha:
        _fail(f"{arm} epoch {epoch}: checkpoint SHA-256 drift")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        _fail(f"{arm} epoch {epoch}: checkpoint missing state_dict")
    state_keys = set(state)
    expected_keys = set(anchor_state) | set(EXTRA_STATE_KEYS)
    if state_keys != expected_keys:
        extra = sorted(state_keys - set(anchor_state))
        missing = sorted(set(anchor_state) - state_keys)
        _fail(f"{arm} epoch {epoch}: expected only two extra student factor tensors; extra={extra}, missing={missing}")
    for name, anchor_tensor in anchor_state.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or not _same_bytes(value, anchor_tensor):
            _fail(f"{arm} epoch {epoch}: inherited anchor tensor drifted: {name}")
    factor_state = {
        "query_factors": state["student.t4_logit_residual.query_factors"],
        "unit_projection.weight": state["student.t4_logit_residual.unit_projection.weight"],
    }
    if tuple(factor_state["query_factors"].shape) != (2, 8) or tuple(factor_state["unit_projection.weight"].shape) != (8, 4):
        _fail(f"{arm} epoch {epoch}: factor shape drift")
    receipt = payload.get("t4_logit_residual_receipt")
    mode, interaction, permutation_seed = ARM_SPEC[arm]
    expected = {"module": "T4LogitResidualLitModule", "residual_mode": mode,
                "interaction_mode": interaction, "residual_rank": 8,
                "residual_permutation_seed": permutation_seed,
                "teacher_checkpoint_sha256": TEACHER_SHA, "selected_t4_anchor_sha256": ANCHOR_SHA,
                "backbone_frozen": True, "cached_state": "per-unit rank factor only"}
    if not isinstance(receipt, Mapping):
        _fail(f"{arm} epoch {epoch}: missing checkpoint logit-residual receipt")
    for key, value in expected.items():
        if receipt.get(key) != value:
            _fail(f"{arm} epoch {epoch}: checkpoint receipt {key} drift")
    factor_sha = tensor_state_sha256(factor_state)
    if receipt.get("active_factor_sha256") != factor_sha:
        _fail(f"{arm} epoch {epoch}: active factor SHA receipt drift")
    return {
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": expected_sha,
        "factor_sha256": factor_sha,
        "query_factors_l2": float(torch.linalg.vector_norm(factor_state["query_factors"]).item()),
        "unit_projection_l2": float(torch.linalg.vector_norm(factor_state["unit_projection.weight"]).item()),
        "query_factors_nonzero": int(torch.count_nonzero(factor_state["query_factors"]).item()),
        "unit_projection_nonzero": int(torch.count_nonzero(factor_state["unit_projection.weight"]).item()),
        "initial_factor_sha256": receipt.get("initial_factor_sha256"),
    }


def _bootstrap_ci(delta: np.ndarray, *, seed: int = 42, repeats: int = 20000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(repeats, len(delta)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _sign_flip_exact_two_sided(delta: np.ndarray) -> float | None:
    """Exact all-sign-flip p-value on observed session magnitudes (descriptive).

    Unlike a Wilcoxon implementation this does not need a rank/tie convention:
    enumerate the 2^n possible signs of the observed nonzero magnitudes and compare
    their absolute mean with the observed absolute mean.  It is intentionally not
    presented as a population inferential test for six related sessions.
    """
    magnitudes = np.abs(np.asarray(delta, dtype=float))
    magnitudes = magnitudes[magnitudes > 0.0]
    if len(magnitudes) == 0:
        return None
    observed = abs(float(np.mean(delta)))
    signs = np.array(
        [[1.0 if mask & (1 << i) else -1.0 for i in range(len(magnitudes))]
         for mask in range(1 << len(magnitudes))],
        dtype=float,
    )
    null_abs_means = np.abs((signs * magnitudes).mean(axis=1))
    return float(np.mean(null_abs_means >= observed - 1e-12))


def paired_comparison(aligned: ScoredResult, control: ScoredResult, *, control_name: str) -> dict[str, Any]:
    if aligned.sessions != control.sessions:
        _fail(f"aligned/control session mismatch: {control_name}")
    delta = np.asarray([aligned.session_means[s] - control.session_means[s] for s in aligned.sessions], dtype=float)
    mean_delta = float(aligned.score - control.score)
    if not math.isclose(mean_delta, float(delta.mean()), rel_tol=0.0, abs_tol=1e-10):
        _fail(f"paired score/session delta inconsistency versus {control_name}")
    return {
        "control": control_name,
        "mean_delta_r2": mean_delta,
        "per_session_delta_r2": {s: float(v) for s, v in zip(aligned.sessions, delta)},
        "all_six_sessions_positive": bool(np.all(delta > 0.0)),
        "sessions_positive": int(np.sum(delta > 0.0)),
        "bootstrap_95pct_ci_descriptive": _bootstrap_ci(delta),
        "sign_flip_exact_two_sided_p_descriptive": _sign_flip_exact_two_sided(delta),
        "practical_delta_at_least_0p03": bool(mean_delta >= 0.03),
    }


def _load_scored_result(path: Path, *, arm: str, expected_sessions: tuple[str, ...] | None,
                        anchor_state: Mapping[str, torch.Tensor] | None, preflight: Mapping[str, Any] | None) -> ScoredResult:
    result = _load_json(path)
    _validate_common_result(result, label=arm, expected_sessions=expected_sessions)
    metadata_path, metadata = _metadata_for(result, label=arm)
    sessions, score, session_means = rederive_result_scores(result, label=arm)
    for epoch in EPOCHS:
        row = result["per_epoch"][str(epoch)]
        expected_checkpoint = Path(str(result["run_dir"])) / "epoch_ckpts" / f"epoch_{epoch - 1:03d}.ckpt"
        if Path(str(row.get("checkpoint_path", ""))) != expected_checkpoint:
            _fail(f"{arm}: epoch {epoch} checkpoint path is not exactly the frozen run_dir epoch path")
    checkpoint_receipts: dict[str, Any] = {}
    if arm != "t4_continuation":
        assert anchor_state is not None and preflight is not None
        _validate_arm_metadata(metadata, arm=arm, preflight=preflight)
        for epoch in EPOCHS:
            row = result["per_epoch"][str(epoch)]
            checkpoint_receipts[str(epoch)] = verify_checkpoint(
                Path(row["checkpoint_path"]), expected_sha=str(row["checkpoint_sha256"]),
                anchor_state=anchor_state, arm=arm, epoch=epoch,
            )
    return ScoredResult(arm, path, result, metadata_path, metadata, sessions, score, session_means, checkpoint_receipts)


def aggregate(result_dir: Path = RESULT_DIR, baseline_path: Path = BASELINE) -> dict[str, Any]:
    """Audit a complete round and return an output-ready immutable receipt."""
    for path, expected in ((ANCHOR, ANCHOR_SHA), (TEACHER, TEACHER_SHA), (MANIFEST, MANIFEST_SHA), (PROTOCOL, None)):
        if not path.is_file():
            _fail(f"missing frozen input: {path}")
        if expected is not None and sha256_file(path) != expected:
            _fail(f"frozen input SHA drift: {path}")
    preflight_path = result_dir / "preflight.json"
    preflight = _load_json(preflight_path)
    _validate_preflight(preflight)
    baseline = _load_scored_result(baseline_path, arm="t4_continuation", expected_sessions=None,
                                   anchor_state=None, preflight=None)
    anchor_payload = torch.load(ANCHOR, map_location="cpu", weights_only=False)
    anchor_state = anchor_payload.get("state_dict")
    if not isinstance(anchor_state, Mapping):
        _fail("selected anchor has no state_dict")
    arms: dict[str, ScoredResult] = {"t4_continuation": baseline}
    for arm in ARM_SPEC:
        arms[arm] = _load_scored_result(
            result_dir / f"{arm}_s42.json", arm=arm, expected_sessions=baseline.sessions,
            anchor_state=anchor_state, preflight=preflight,
        )
    comparisons = {
        key: paired_comparison(arms["aligned_logit"], arms[key], control_name=key)
        for key in ("t4_continuation", "shuffled_logit", "additive_control")
    }
    gate_pass = all(x["practical_delta_at_least_0p03"] and x["all_six_sessions_positive"] for x in comparisons.values())
    disposition = (
        "pass_seed42_validation_only_authorizes_separately_reviewed_three_seed_confirmation"
        if gate_pass else
        "fail_stop_no_rank_seed_confidence_formal_or_int8_follow_up_authorized"
    )
    return {
        "schema_version": "sua_t4_factorized_logit_aggregate_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "frozen_seed42_validation_only_causal_gate",
        "formal_test_opened_by_aggregator": False,
        "frozen_inputs": {
            "protocol": {"path": str(PROTOCOL), "sha256": sha256_file(PROTOCOL)},
            "preflight": {"path": str(preflight_path), "sha256": sha256_file(preflight_path)},
            "selected_anchor": {"path": str(ANCHOR), "sha256": ANCHOR_SHA},
            "teacher": {"path": str(TEACHER), "sha256": TEACHER_SHA},
            "manifest": {"path": str(MANIFEST), "sha256": MANIFEST_SHA},
        },
        "epoch_window": list(EPOCHS),
        "validation_sessions": list(baseline.sessions),
        "arms": {
            name: {
                "result_path": str(item.source_path), "result_sha256": sha256_file(item.source_path),
                "run_metadata_path": str(item.metadata_path), "run_metadata_sha256": sha256_file(item.metadata_path),
                "rederived_variant_score": item.score, "rederived_session_mean_r2": item.session_means,
                "checkpoint_factor_receipts": item.checkpoint_receipts,
            }
            for name, item in arms.items()
        },
        "comparisons_aligned_minus_control": comparisons,
        "primary_gate": {
            "threshold_mean_delta_r2": 0.03,
            "required_sessions_positive": 6,
            "pass": gate_pass,
            "disposition": disposition,
            "inferential_statistics_are_descriptive_only": True,
        },
    }


def render_report(receipt: Mapping[str, Any]) -> str:
    gate = receipt["primary_gate"]
    lines = [
        "# SUA selected-T4 factorized attention-logit residual: final causal screen",
        "",
        f"**Gate:** {'PASS' if gate['pass'] else 'FAIL'}  ",
        f"**Disposition:** `{gate['disposition']}`  ",
        "**Scope:** seed-42 validation only; no formal SUA session was opened by this aggregation.",
        "",
        "The score in every arm was rederived as the unweighted mean of epochs 5–12, and each session value was rederived as its mean across the same eight checkpoints. The numerical gate requires `aligned − control >= +0.03 R²` and positive paired deltas in all six validation sessions for every listed control. Bootstrap intervals and exact all-sign-flip values are descriptive only.",
        "",
        "| Arm | Re-derived validation R² |",
        "|---|---:|",
    ]
    for arm, data in receipt["arms"].items():
        lines.append(f"| {arm} | {data['rederived_variant_score']:.6f} |")
    lines += ["", "| Aligned comparison | Mean ΔR² | Positive sessions | Bootstrap 95% CI (descriptive) | Exact sign-flip p (descriptive) | Pass |", "|---|---:|---:|---|---:|---|"]
    for key, data in receipt["comparisons_aligned_minus_control"].items():
        ci = data["bootstrap_95pct_ci_descriptive"]
        p = data["sign_flip_exact_two_sided_p_descriptive"]
        passed = data["practical_delta_at_least_0p03"] and data["all_six_sessions_positive"]
        lines.append(f"| aligned − {key} | {data['mean_delta_r2']:+.6f} | {data['sessions_positive']}/6 | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {p if p is not None else 'undefined'} | {'yes' if passed else 'no'} |")
    lines += ["", "## Integrity receipts", "", "- Every scored residual checkpoint had exactly two extra student tensors: `t4_logit_residual.query_factors` and `t4_logit_residual.unit_projection.weight`.", "- Every other checkpoint tensor was compared bitwise against the selected T4 anchor; factor SHA-256 and norms are in `aggregate.json`.", "- The audit verified the strict 27/6/6 manifest, T4=50/activity=30/evaluation-start=50 receipts, rank 8, exactly 48 optimized parameters, frozen cost receipt, and no-formal-test receipts."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--out", type=Path, default=None, help="Default: <result-dir>/aggregate.json")
    parser.add_argument("--report", type=Path, default=None, help="Default: <result-dir>/report.md")
    args = parser.parse_args()
    out = args.out or args.result_dir / "aggregate.json"
    report = args.report or args.result_dir / "report.md"
    # This precondition intentionally precedes any result loading: never replace a receipt.
    for path, label in ((out, "aggregate"), (report, "report")):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite {label}: {path}")
    receipt = aggregate(args.result_dir, args.baseline)
    out.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL retains the non-overwrite property if another finalizer races us.
    for path, text in ((out, json.dumps(receipt, indent=2, sort_keys=True) + "\n"), (report, render_report(receipt))):
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as exc:
            raise SystemExit(f"Refusing to overwrite concurrently-created artifact: {path}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(f"Wrote immutable aggregate: {out}")
    print(f"Wrote immutable report: {report}")
    print(f"Primary gate: {'PASS' if receipt['primary_gate']['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
