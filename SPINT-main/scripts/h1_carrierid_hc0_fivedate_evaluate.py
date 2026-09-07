#!/usr/bin/env python3
"""Evaluate and aggregate the five-date separately trained H-C0 control.

Each date reuses the exact strict target dataset already reported for H-S/H-C.
The new checkpoint has the same compact H-C topology but `zero_carrier=true`.
No target optimization, backward pass, checkpoint selection, or formal hidden
evaluation is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h1_carrierid_date_lodo_phase2_terminal_evaluate import _evaluate
from src.data import h1_carrierid_date_lodo_target as target_module
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256, sha256_file


ARTIFACT_ROOT = ROOT / "pilot_artifacts/h1_carrierid_date_lodo_hc0_fivedate"
REFERENCE_ROOT = ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase2/terminal_evaluations"
PAIR_PREFLIGHT_SCHEMA = "h1_carrierid_date_lodo_phase2_pair_cpu_preflight_v1"
PAIR_PREFLIGHT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_PHASE2_PAIR_SOURCE_ONLY_NOT_LAUNCHED"
DATE_RECEIPT_SCHEMA = "h1_carrierid_hc0_date_lodo_evaluation_v1"
DATE_RECEIPT_STATUS = "PASS_HC0_STRICT_DEVELOPMENT_EVALUATION"
REFERENCE_SCHEMA = "h1_carrierid_date_lodo_phase2_terminal_evaluation_v1"

# The source-only H-C0 runs use a fixed, date-specific number of complete
# source batches per epoch.  Keep this schedule explicit: a checkpoint can
# otherwise claim epoch 49 while silently containing a truncated or padded
# training loop.  These values are derived from the frozen Phase-2 source
# schedules (global_step == batches_per_epoch * 50 for the observed terminal
# checkpoints) and are part of the five-date evaluation contract.
FIXED_EPOCH_ZERO_BASED = 49
FIXED_EPOCHS_COMPLETED = 50
EXPECTED_SOURCE_BATCHES_PER_EPOCH = {
    "19250108": 3356,
    "19250113": 3422,
    "19250115": 3454,
    "19250119": 3404,
    "19250120": 3419,
}
EXPECTED_LIGHTNING_VERSION = "2.4.0"


def _expected_source_batches(date: str) -> int:
    try:
        return int(EXPECTED_SOURCE_BATCHES_PER_EPOCH[date])
    except KeyError as error:
        raise ValueError(f"no fixed source-batch schedule for H-C0 date {date}") from error


def _expected_global_step(date: str) -> int:
    return _expected_source_batches(date) * FIXED_EPOCHS_COMPLETED


def _validate_progress_counter(value: Any, *, label: str, expected: int) -> None:
    _need(type(value) is int and value == expected, f"{label} terminal progress drift")


def _validate_terminal_progress(payload: Mapping[str, Any], *, date: str) -> None:
    """Fail closed on Lightning's terminal epoch/batch/optimizer loop state."""

    batches = _expected_source_batches(date)
    global_step = _expected_global_step(date)
    loops = payload.get("loops")
    _need(isinstance(loops, Mapping), "H-C0 checkpoint loops are missing")
    fit = loops.get("fit_loop")
    _need(isinstance(fit, Mapping), "H-C0 fit loop state is missing")

    epoch_progress = fit.get("epoch_progress")
    _need(isinstance(epoch_progress, Mapping), "H-C0 epoch progress is missing")
    epoch_total = epoch_progress.get("total")
    _need(isinstance(epoch_total, Mapping), "H-C0 total epoch progress is missing")
    for key, expected in {
        "ready": FIXED_EPOCHS_COMPLETED,
        "completed": FIXED_EPOCH_ZERO_BASED,
        "started": FIXED_EPOCHS_COMPLETED,
        "processed": FIXED_EPOCHS_COMPLETED,
    }.items():
        _validate_progress_counter(epoch_total.get(key), label=f"fit_loop.epoch_progress.total.{key}", expected=expected)
    epoch_current = epoch_progress.get("current")
    _need(isinstance(epoch_current, Mapping), "H-C0 current epoch progress is missing")
    for key, expected in {
        "ready": FIXED_EPOCHS_COMPLETED,
        "completed": FIXED_EPOCH_ZERO_BASED,
        "started": FIXED_EPOCHS_COMPLETED,
        "processed": FIXED_EPOCHS_COMPLETED,
    }.items():
        _validate_progress_counter(epoch_current.get(key), label=f"fit_loop.epoch_progress.current.{key}", expected=expected)

    epoch_state = fit.get("epoch_loop.state_dict")
    _need(isinstance(epoch_state, Mapping), "H-C0 epoch-loop state is missing")
    _validate_progress_counter(
        epoch_state.get("_batches_that_stepped"),
        label="fit_loop.epoch_loop.state_dict._batches_that_stepped",
        expected=global_step,
    )
    batch_progress = fit.get("epoch_loop.batch_progress")
    _need(isinstance(batch_progress, Mapping), "H-C0 batch progress is missing")
    for scope_name, scope in (("total", global_step), ("current", batches)):
        counters = batch_progress.get(scope_name)
        _need(isinstance(counters, Mapping), f"H-C0 {scope_name} batch progress is missing")
        for key in ("ready", "completed", "started", "processed"):
            _validate_progress_counter(counters.get(key), label=f"fit_loop.batch_progress.{scope_name}.{key}", expected=scope)
    _need(batch_progress.get("is_last_batch") is True, "H-C0 terminal batch is not marked last")

    automatic = fit.get("epoch_loop.automatic_optimization.optim_progress")
    _need(isinstance(automatic, Mapping), "H-C0 optimizer progress is missing")
    optimizer = automatic.get("optimizer")
    _need(isinstance(optimizer, Mapping), "H-C0 optimizer progress is missing")
    step = optimizer.get("step")
    _need(isinstance(step, Mapping), "H-C0 optimizer step progress is missing")
    for scope_name, scope in (("total", global_step), ("current", batches)):
        counters = step.get(scope_name)
        _need(isinstance(counters, Mapping), f"H-C0 optimizer {scope_name} step progress is missing")
        for key in ("ready", "completed"):
            _validate_progress_counter(counters.get(key), label=f"optimizer.step.{scope_name}.{key}", expected=scope)
    zero_grad = optimizer.get("zero_grad")
    _need(isinstance(zero_grad, Mapping), "H-C0 optimizer zero_grad progress is missing")
    for scope_name, scope in (("total", global_step), ("current", batches)):
        counters = zero_grad.get(scope_name)
        _need(isinstance(counters, Mapping), f"H-C0 optimizer {scope_name} zero_grad progress is missing")
        for key in ("ready", "completed", "started"):
            _validate_progress_counter(counters.get(key), label=f"optimizer.zero_grad.{scope_name}.{key}", expected=scope)

    scheduler_progress = fit.get("epoch_loop.scheduler_progress")
    _need(isinstance(scheduler_progress, Mapping), "H-C0 scheduler progress is missing")
    for scope_name in ("total", "current"):
        counters = scheduler_progress.get(scope_name)
        _need(isinstance(counters, Mapping), f"H-C0 scheduler {scope_name} progress is missing")
        for key in ("ready", "completed"):
            _validate_progress_counter(counters.get(key), label=f"scheduler.{scope_name}.{key}", expected=0)
    _need(payload.get("lr_schedulers") == [], "H-C0 checkpoint unexpectedly contains a scheduler")


def _validate_finite_state(value: Any, *, label: str) -> None:
    """Recursively verify optimizer/state tensors are finite on CPU."""

    if torch.is_tensor(value):
        _need(bool(torch.isfinite(value).all()), f"{label} contains nonfinite tensor values")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite_state(child, label=f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite_state(child, label=f"{label}[{index}]")


def _validate_terminal_checkpoint_payload(
    payload: Any, *, date: str, config_path: Path,
) -> dict[str, Any]:
    """Validate a fixed e49 H-C0 checkpoint before model or target access."""

    _need(isinstance(payload, Mapping), "H-C0 checkpoint must be a mapping")
    _need(type(payload.get("epoch")) is int and payload.get("epoch") == FIXED_EPOCH_ZERO_BASED,
          "H-C0 checkpoint top-level epoch is not fixed e49")
    _need(type(payload.get("global_step")) is int and payload.get("global_step") == _expected_global_step(date),
          f"H-C0 checkpoint global_step is not the fixed {_expected_global_step(date)} source steps")
    _need(payload.get("pytorch-lightning_version") == EXPECTED_LIGHTNING_VERSION,
          "H-C0 checkpoint PyTorch Lightning version drift")
    optimizer_states = payload.get("optimizer_states")
    _need(isinstance(optimizer_states, list) and len(optimizer_states) == 1,
          "H-C0 checkpoint optimizer state count drift")
    _validate_finite_state(optimizer_states, label="H-C0 optimizer_states")
    _validate_terminal_progress(payload, date=date)
    state_dict = payload.get("state_dict")
    _need(isinstance(state_dict, Mapping) and bool(state_dict), "H-C0 checkpoint state_dict is empty")
    for name, tensor in state_dict.items():
        _need(torch.is_tensor(tensor), f"H-C0 state_dict entry is not a tensor: {name}")
        _need(bool(torch.isfinite(tensor).all()), f"H-C0 state_dict entry is nonfinite: {name}")
    metadata = payload.get("h1_carrierid_date_lodo_phase2")
    _need(isinstance(metadata, dict) and metadata.get("checkpoint_epoch_zero_based") == FIXED_EPOCH_ZERO_BASED,
          "H-C0 checkpoint lacks fixed e49 source-only metadata")
    _need(metadata.get("epochs_completed") == FIXED_EPOCHS_COMPLETED,
          "H-C0 checkpoint metadata does not record 50 completed epochs")
    _need(not config_path.is_symlink(), f"H-C0 resolved config is symlinked: {config_path}")
    _need(metadata.get("config_sha256") == sha256_file(config_path),
          "H-C0 checkpoint/config SHA drift")
    return metadata


def _validate_resolved_hc0_config(config: Any, *, date: str, run_dir: Path) -> None:
    """Validate the resolved train lifecycle before opening a checkpoint/model."""

    _need(config.train is True and config.test is False, "H-C0 config train/test lifecycle drift")
    _need(config.ckpt_path in (None, "", "null"), "H-C0 config permits a warm-start checkpoint")
    trainer = config.trainer
    _need(int(trainer.min_epochs) == FIXED_EPOCHS_COMPLETED and int(trainer.max_epochs) == FIXED_EPOCHS_COMPLETED,
          "H-C0 trainer is not fixed to exactly 50 epochs")
    _need(int(trainer.limit_val_batches) == 0 and int(trainer.num_sanity_val_steps) == 0,
          "H-C0 trainer enables validation/sanity batches")
    data = config.data
    _need(int(data.fixed_epochs) == FIXED_EPOCHS_COMPLETED, "H-C0 DataModule fixed_epochs drift")
    phase2 = config.phase2
    _need(str(phase2.outer_date) == date, "H-C0 phase2 outer_date drift")
    callbacks = config.callbacks
    _need(isinstance(callbacks, Mapping), "H-C0 callbacks are malformed")
    _need(set(callbacks) == {"fixed_epoch50"}, "H-C0 config contains an unexpected/early-stopping callback")
    fixed = callbacks.get("fixed_epoch50")
    _need(isinstance(fixed, Mapping), "H-C0 fixed_epoch50 callback is missing")
    expected_dir = str((run_dir / "checkpoints/fixed_epoch50").resolve())
    # Hydra's ``${hydra:runtime.output_dir}`` resolver is unavailable when a
    # saved .hydra/config.yaml is loaded outside the original Hydra process;
    # inspect the raw interpolation token rather than resolving it here.
    dir_node = fixed._get_node("dirpath") if hasattr(fixed, "_get_node") else None
    dir_value = dir_node._value() if dir_node is not None and hasattr(dir_node, "_value") else fixed.get("dirpath")
    _need(
        str(dir_value) in {"${paths.output_dir}/checkpoints/fixed_epoch50", expected_dir},
        "H-C0 fixed_epoch50 callback path drift",
    )
    for key, expected in {
        "monitor": None, "every_n_epochs": FIXED_EPOCHS_COMPLETED,
        "save_last": False, "save_top_k": -1, "filename": "epoch_{epoch:03d}",
        "auto_insert_metric_name": False,
    }.items():
        _need(fixed.get(key) == expected, f"H-C0 fixed_epoch50 callback {key} drift")


def _reference_path(date: str) -> Path:
    return REFERENCE_ROOT / f"H1_CARRIERID_DATE_LODO_PHASE2_{date}_HS_HC_TERMINAL_EVALUATION_v1.json"


def _run_dir(date: str) -> Path:
    return ARTIFACT_ROOT / "gpu_runs" / date


def _output_path(date: str) -> Path:
    return ARTIFACT_ROOT / "evaluations" / f"H1_CARRIERID_HC0_{date}_EVALUATION_v1.json"


def _canonical_json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_pair_preflight(config: Any, *, date: str, reference_source_manifest_sha256: str) -> tuple[Path, dict[str, Any]]:
    """Validate the date-bound pair receipt named by the resolved run config.

    H-C0 source training already binds the immutable Phase-1 source bundle;
    this check is deliberately receipt-only and does not alter model inputs or
    target-window construction.  It prevents a resolved config from silently
    pointing at another date/pair before evaluation proceeds.
    """

    try:
        pair_value = config.phase2.pair_preflight_path
    except (AttributeError, KeyError) as error:
        raise ValueError("H-C0 resolved config lacks phase2.pair_preflight_path") from error
    pair_candidate = Path(str(pair_value))
    if pair_candidate.is_symlink():
        raise FileNotFoundError(f"H-C0 pair preflight is symlinked: {pair_candidate}")
    pair_path = pair_candidate.resolve()
    if not pair_path.is_file():
        raise FileNotFoundError(f"H-C0 pair preflight is missing: {pair_path}")
    if stat.S_IMODE(pair_path.stat().st_mode) != 0o444:
        raise ValueError(f"H-C0 pair preflight is not immutable mode-0444: {pair_path}")
    try:
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"H-C0 pair preflight JSON is unreadable: {pair_path}") from error
    if not isinstance(pair, dict):
        raise ValueError("H-C0 pair preflight must be a JSON object")
    if pair.get("schema") != PAIR_PREFLIGHT_SCHEMA or pair.get("status") != PAIR_PREFLIGHT_STATUS:
        raise ValueError("H-C0 pair preflight schema/status drift")
    if str(pair.get("outer_date", "")) != date:
        raise ValueError("H-C0 pair preflight/date drift")
    source = pair.get("source_binding")
    if not isinstance(source, dict) or str(source.get("outer_date", "")) != date:
        raise ValueError("H-C0 pair preflight source binding/date drift")
    if pair.get("source_binding_sha256") != canonical_sha256(source):
        raise ValueError("H-C0 pair preflight source-binding SHA drift")
    if source.get("target_recordings_opened") != 0 or source.get("target_bytes_read") != 0:
        raise ValueError("H-C0 pair preflight source binding records target access")
    if source.get("warm_start_forbidden") is not True:
        raise ValueError("H-C0 pair preflight permits a warm start")
    fresh_hc = pair.get("fresh_models", {}).get("h_c") if isinstance(pair.get("fresh_models"), dict) else None
    if not isinstance(fresh_hc, dict) or fresh_hc.get("component") != "H1CarrierIdSpint":
        raise ValueError("H-C0 pair preflight lacks the corresponding H-C fresh model")
    if fresh_hc.get("carrier_columns_literal_zero_at_init") is not True:
        raise ValueError("H-C0 pair preflight H-C initialization is not literal-zero")
    if source.get("source_manifest_sha256") != reference_source_manifest_sha256:
        raise ValueError("H-C0 pair preflight source manifest differs from the paired H-S/H-C evaluation")
    return pair_path, pair


def _write_immutable_json(path: Path, value: Any) -> str:
    """Atomically publish one mode-0444 JSON receipt and return its file SHA."""

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, output)
        return sha256_file(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: Any, *, label: str) -> float:
    _need(not isinstance(value, bool) and isinstance(value, (int, float)), f"{label} must be numeric")
    result = float(value)
    _need(math.isfinite(result), f"{label} must be finite")
    return result


def _positive_int(value: Any, *, label: str) -> int:
    _need(not isinstance(value, bool) and isinstance(value, int) and value > 0, f"{label} must be a positive integer")
    return int(value)


def _read_immutable_json_object(path: Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    _need(not candidate.is_symlink(), f"{label} is symlinked: {candidate}")
    _need(candidate.is_file(), f"{label} is missing: {candidate}")
    _need(stat.S_IMODE(candidate.stat().st_mode) == 0o444, f"{label} is not immutable mode-0444: {candidate}")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} JSON is unreadable: {candidate}") from error
    _need(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _validate_metric_block(value: Any, *, label: str) -> dict[str, Any]:
    _need(isinstance(value, Mapping), f"{label} must be an object")
    pooled_r2 = _finite_number(value.get("pooled_r2"), label=f"{label}.pooled_r2")
    samples = _positive_int(value.get("samples"), label=f"{label}.samples")
    _positive_int(value.get("batches"), label=f"{label}.batches")
    _positive_int(value.get("last_batch_size"), label=f"{label}.last_batch_size")
    _need(value.get("r2_accumulator_dtype") == "float64", f"{label} accumulator precision drift")
    _need(value.get("state_immutable") is True, f"{label} model state was not immutable")
    before = value.get("state_sha256_before")
    after = value.get("state_sha256_after")
    _need(_is_sha256(before) and before == after, f"{label} model-state SHA drift")
    _need(_is_sha256(value.get("query_window_indices_sha256")), f"{label} query-window SHA is invalid")

    per_session = value.get("per_session")
    _need(isinstance(per_session, Mapping) and per_session, f"{label}.per_session must be a non-empty object")
    normalized_sessions: dict[str, dict[str, Any]] = {}
    for session, session_value in per_session.items():
        _need(isinstance(session, str) and session, f"{label} has an invalid session key")
        _need(isinstance(session_value, Mapping), f"{label}.per_session.{session} must be an object")
        normalized_sessions[session] = {
            "r2": _finite_number(session_value.get("r2"), label=f"{label}.per_session.{session}.r2"),
            "samples": _positive_int(
                session_value.get("samples"), label=f"{label}.per_session.{session}.samples",
            ),
        }
    _need(
        sum(session["samples"] for session in normalized_sessions.values()) == samples,
        f"{label} per-session samples do not sum to samples",
    )
    return {
        "pooled_r2": pooled_r2,
        "samples": samples,
        "query_window_indices_sha256": str(value["query_window_indices_sha256"]),
        "per_session": normalized_sessions,
    }


def _validate_bound_file_row(value: Any, *, expected_path: Path, label: str) -> None:
    _need(isinstance(value, Mapping), f"{label} binding must be an object")
    claimed_path = Path(str(value.get("path", ""))).resolve()
    _need(claimed_path == expected_path.resolve(), f"{label} path/date binding drift")
    _need(_is_sha256(value.get("sha256")), f"{label} SHA is invalid")


def _validate_date_receipt(path: Path, *, expected_date: str) -> dict[str, Any]:
    """Validate one immutable date receipt without opening data, configs, or checkpoints."""

    _need(expected_date in CONFIRMATORY_DATES, f"unknown confirmatory date: {expected_date}")
    expected_output = _output_path(expected_date).resolve()
    _need(Path(path).resolve() == expected_output, "H-C0 receipt output path/date binding drift")
    receipt = _read_immutable_json_object(path, label="H-C0 date receipt")
    _need(receipt.get("schema") == DATE_RECEIPT_SCHEMA, "H-C0 date receipt schema drift")
    _need(receipt.get("status") == DATE_RECEIPT_STATUS, "H-C0 date receipt status drift")
    _need(receipt.get("outer_date") == expected_date, "H-C0 date receipt outer_date drift")

    claimed_content_sha = receipt.get("canonical_content_sha256")
    _need(_is_sha256(claimed_content_sha), "H-C0 date receipt canonical SHA is invalid")
    unhashed = dict(receipt)
    unhashed.pop("canonical_content_sha256", None)
    _need(_canonical_json_sha(unhashed) == claimed_content_sha, "H-C0 date receipt canonical SHA drift")

    scope = receipt.get("scope")
    _need(isinstance(scope, Mapping), "H-C0 date receipt scope must be an object")
    _need(scope.get("formal_heldout_opened") is False, "H-C0 receipt reports formal held-out access")
    _need(scope.get("evalai_opened") is False, "H-C0 receipt reports EvalAI access")
    for field in ("target_optimizer_steps", "target_backward_steps"):
        _need(type(scope.get(field)) is int and scope[field] == 0, f"H-C0 receipt reports nonzero {field}")

    run_dir = _run_dir(expected_date)
    _validate_bound_file_row(
        receipt.get("checkpoint"),
        expected_path=run_dir / "checkpoints/fixed_epoch50/epoch_049.ckpt",
        label="H-C0 checkpoint",
    )
    _validate_bound_file_row(
        receipt.get("config"), expected_path=run_dir / ".hydra/config.yaml", label="H-C0 config",
    )

    reference_row = receipt.get("reference")
    reference_path = _reference_path(expected_date)
    _validate_bound_file_row(reference_row, expected_path=reference_path, label="H-C reference")
    reference = _read_immutable_json_object(reference_path, label="H-C reference receipt")
    _need(sha256_file(reference_path) == reference_row["sha256"], "H-C reference receipt file SHA drift")
    _need(reference.get("schema") == REFERENCE_SCHEMA, "H-C reference receipt schema drift")
    _need(
        reference.get("status") == f"PASS_H1_CARRIERID_DATE_LODO_PHASE2_{expected_date}_HS_HC_EVALUATED",
        "H-C reference receipt status/date drift",
    )
    _need(reference.get("outer_date") == expected_date, "H-C reference receipt outer_date drift")

    pair_row = receipt.get("pair_preflight")
    _need(isinstance(pair_row, Mapping), "H-C0 pair-preflight binding must be an object")
    pair_path = Path(str(pair_row.get("path", "")))
    _need(_is_sha256(pair_row.get("sha256")), "H-C0 pair-preflight SHA is invalid")
    pair = _read_immutable_json_object(pair_path, label="H-C0 pair-preflight receipt")
    _need(sha256_file(pair_path) == pair_row["sha256"], "H-C0 pair-preflight file SHA drift")
    _need(pair.get("schema") == PAIR_PREFLIGHT_SCHEMA, "H-C0 pair-preflight schema drift")
    _need(pair.get("status") == PAIR_PREFLIGHT_STATUS, "H-C0 pair-preflight status drift")
    _need(pair.get("outer_date") == expected_date, "H-C0 pair-preflight outer_date drift")

    metrics = receipt.get("metrics")
    _need(isinstance(metrics, Mapping), "H-C0 date receipt metrics must be an object")
    blocks = {
        name: _validate_metric_block(metrics.get(name), label=f"metrics.{name}")
        for name in ("h_c0", "h_c_reference", "h_s_reference")
    }
    _need(metrics.get("h_c_reference") == reference.get("metrics", {}).get("h_c"), "H-C reference metrics drift")
    _need(metrics.get("h_s_reference") == reference.get("metrics", {}).get("h_s"), "H-S reference metrics drift")
    hc0 = blocks["h_c0"]
    hc = blocks["h_c_reference"]
    hs = blocks["h_s_reference"]
    session_keys = set(hc0["per_session"])
    _need(set(hc["per_session"]) == session_keys, "H-C/H-C0 per-session keys differ")
    _need(set(hs["per_session"]) == session_keys, "H-S/H-C0 per-session keys differ")
    _need(hc["samples"] == hc0["samples"] == hs["samples"], "H-S/H-C/H-C0 sample totals differ")
    _need(
        hc["query_window_indices_sha256"] == hc0["query_window_indices_sha256"]
        == hs["query_window_indices_sha256"],
        "H-S/H-C/H-C0 query windows differ",
    )
    for session in session_keys:
        _need(
            hc["per_session"][session]["samples"] == hc0["per_session"][session]["samples"]
            == hs["per_session"][session]["samples"],
            f"H-S/H-C/H-C0 samples differ for {session}",
        )

    h_c_minus_h_c0 = _finite_number(metrics.get("h_c_minus_h_c0"), label="metrics.h_c_minus_h_c0")
    h_c0_minus_h_s = _finite_number(metrics.get("h_c0_minus_h_s"), label="metrics.h_c0_minus_h_s")
    _need(
        math.isclose(h_c_minus_h_c0, hc["pooled_r2"] - hc0["pooled_r2"], rel_tol=0.0, abs_tol=1e-12),
        "H-C minus H-C0 pooled contrast drift",
    )
    _need(
        math.isclose(h_c0_minus_h_s, hc0["pooled_r2"] - hs["pooled_r2"], rel_tol=0.0, abs_tol=1e-12),
        "H-C0 minus H-S pooled contrast drift",
    )
    return receipt


def evaluate_date(date: str, *, device_name: str) -> dict[str, Any]:
    if date not in CONFIRMATORY_DATES:
        raise ValueError(f"unknown confirmatory date: {date}")
    output = _output_path(date)
    if output.exists() or output.is_symlink():
        return _validate_date_receipt(output, expected_date=date)
    reference_path = _reference_path(date)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    run_dir = _run_dir(date)
    config_path = run_dir / ".hydra/config.yaml"
    checkpoint_path = run_dir / "checkpoints/fixed_epoch50/epoch_049.ckpt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"H-C0 terminal artifacts incomplete for {date}: {run_dir}")

    config = OmegaConf.load(config_path)
    _validate_resolved_hc0_config(config, date=date, run_dir=run_dir)
    if config.protocol_id != f"h1_carrierid_date_lodo_hc0_{date}_source_only_v1":
        raise ValueError("H-C0 protocol/date drift")
    if config.model.net._target_ != "src.models.components.h1_carrierid_spint.H1CarrierIdSpint":
        raise ValueError("H-C0 consumer topology drift")
    if config.model.net.zero_carrier is not True:
        raise ValueError("H-C0 config does not enforce a literal zero carrier")
    pair_path, _pair = _validate_pair_preflight(
        config, date=date, reference_source_manifest_sha256=str(reference["source_manifest_sha256"]),
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = _validate_terminal_checkpoint_payload(payload, date=date, config_path=config_path)
    if metadata.get("arm") != "H-C" or str(metadata.get("outer_date", "")) != date:
        raise ValueError("H-C0 checkpoint arm/date drift")
    if metadata.get("phase2_source_binding_sha256") != _pair["source_binding_sha256"]:
        raise ValueError("H-C0 checkpoint source binding differs from its pair preflight")
    if metadata.get("phase1_source_manifest_sha256") != reference["source_manifest_sha256"]:
        raise ValueError("H-C0 checkpoint source manifest differs from the paired H-S/H-C evaluation")
    if metadata.get("selected_by") != "fixed_terminal_epoch_no_validation_or_target_selection":
        raise ValueError("H-C0 checkpoint selection is not fixed-terminal")
    if metadata.get("checkpoint_warm_start") is not False:
        raise ValueError("H-C0 checkpoint records a warm start")
    if metadata.get("target_optimizer_steps") != 0 or metadata.get("target_backward_steps") != 0:
        raise ValueError("H-C0 checkpoint reports target updates")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # The metadata stores only the digest.  The reference target manifest
    # contains the exact source manifest through its paired preflight receipt;
    # the date-scoped Phase-1 bundle is the canonical local reconstruction.
    source_manifest_path = (
        ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase1/source_bundles_v1" / date / "shared_source_manifest.json"
    )
    if sha256_file(source_manifest_path) != reference["source_manifest_sha256"]:
        raise ValueError("H-C0 source manifest differs from the prior H-S/H-C evaluation")
    plan, normalizer, _ = target_module.load_target_dependencies(source_manifest_path, outer_date=date)
    records = target_module.load_outer_date_target_records(ROOT / "data/000954", outer_date=date)
    dataset = target_module.H1CarrierIdDateLodoStrictTargetDataset(records, plan, normalizer, outer_date=date)
    metrics = _evaluate(model, dataset, device, tuple(records))
    reference_hc = reference["metrics"]["h_c"]
    reference_hs = reference["metrics"]["h_s"]
    if metrics["query_window_indices_sha256"] != reference_hc["query_window_indices_sha256"]:
        raise ValueError("H-C0 did not use the exact prior H-S/H-C query windows")
    if {name: records[name].input_sha256 for name in records} != reference["target"]["files"]:
        raise ValueError("target recording bytes changed since the prior paired evaluation")

    result = {
        "schema": DATE_RECEIPT_SCHEMA,
        "status": DATE_RECEIPT_STATUS,
        "outer_date": date,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()), "sha256": sha256_file(checkpoint_path),
            "metadata": metadata,
        },
        "config": {"path": str(config_path.resolve()), "sha256": sha256_file(config_path)},
        "pair_preflight": {"path": str(pair_path), "sha256": sha256_file(pair_path)},
        "reference": {"path": str(reference_path.resolve()), "sha256": sha256_file(reference_path)},
        "metrics": {
            "h_c0": metrics,
            "h_c_reference": reference_hc,
            "h_s_reference": reference_hs,
            "h_c_minus_h_c0": float(reference_hc["pooled_r2"] - metrics["pooled_r2"]),
            "h_c0_minus_h_s": float(metrics["pooled_r2"] - reference_hs["pooled_r2"]),
        },
        "scope": {
            "development_outer_date_replay": True,
            "formal_heldout_opened": False,
            "evalai_opened": False,
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
        },
    }
    result["canonical_content_sha256"] = _canonical_json_sha(result)
    _write_immutable_json(output, result)
    return _validate_date_receipt(output, expected_date=date)


def aggregate() -> dict[str, Any]:
    dates = tuple(CONFIRMATORY_DATES)
    _need(len(set(dates)) == len(dates), "confirmatory date contract contains duplicates")
    _need(dates == tuple(sorted(dates)), "confirmatory date contract is out of chronological order")
    rows: list[dict[str, Any]] = []
    for date in CONFIRMATORY_DATES:
        path = _output_path(date)
        rows.append(_validate_date_receipt(path, expected_date=date))
    _need([row["outer_date"] for row in rows] == list(dates), "H-C0 date receipts are duplicated or out of order")

    deltas = np.asarray([row["metrics"]["h_c_minus_h_c0"] for row in rows], dtype=np.float64)
    hc0_vs_hs = np.asarray([row["metrics"]["h_c0_minus_h_s"] for row in rows], dtype=np.float64)
    date_samples = np.asarray([row["metrics"]["h_c0"]["samples"] for row in rows], dtype=np.float64)
    recording_rows: list[dict[str, Any]] = []
    for row in rows:
        metrics = row["metrics"]
        for session, hc0_session in metrics["h_c0"]["per_session"].items():
            hc_session = metrics["h_c_reference"]["per_session"][session]
            hs_session = metrics["h_s_reference"]["per_session"][session]
            recording_rows.append({
                "outer_date": row["outer_date"],
                "session": session,
                "samples": int(hc0_session["samples"]),
                "h_c_minus_h_c0": float(hc_session["r2"] - hc0_session["r2"]),
                "h_c0_minus_h_s": float(hc0_session["r2"] - hs_session["r2"]),
            })
    recording_hc_hc0 = np.asarray(
        [recording["h_c_minus_h_c0"] for recording in recording_rows], dtype=np.float64,
    )
    recording_hc0_hs = np.asarray(
        [recording["h_c0_minus_h_s"] for recording in recording_rows], dtype=np.float64,
    )
    # Use one fixed resample-index matrix for both contrasts. This keeps the
    # two date-level intervals paired: every replicate resamples the same
    # outer dates for H-C minus H-C0 and H-C0 minus H-S.
    rng = np.random.RandomState(20260809)
    bootstrap_indices = rng.randint(0, len(deltas), size=(100_000, len(deltas)))
    sampled = deltas[bootstrap_indices].mean(axis=1)
    sampled_hc0_vs_hs = hc0_vs_hs[bootstrap_indices].mean(axis=1)
    result = {
        "schema": "h1_carrierid_hc0_fivedate_aggregate_v1",
        "status": "PASS_HC0_FIVEDATE_CAUSAL_DECOMPOSITION",
        "dates": list(CONFIRMATORY_DATES),
        "per_date_h_c_minus_h_c0": deltas.tolist(),
        "per_date_h_c0_minus_h_s": hc0_vs_hs.tolist(),
        "mean_date_h_c_minus_h_c0": float(deltas.mean()),
        "median_date_h_c_minus_h_c0": float(np.median(deltas)),
        "positive_dates_h_c_minus_h_c0": int(np.count_nonzero(deltas > 0.0)),
        "bootstrap_date_95ci_h_c_minus_h_c0": [float(value) for value in np.quantile(sampled, [0.025, 0.975])],
        "mean_date_h_c0_minus_h_s": float(hc0_vs_hs.mean()),
        "median_date_h_c0_minus_h_s": float(np.median(hc0_vs_hs)),
        "positive_dates_h_c0_minus_h_s": int(np.count_nonzero(hc0_vs_hs > 0.0)),
        "bootstrap_date_95ci_h_c0_minus_h_s": [
            float(value) for value in np.quantile(sampled_hc0_vs_hs, [0.025, 0.975])
        ],
        "sample_count_weighted_mean_date_h_c_minus_h_c0": float(np.average(deltas, weights=date_samples)),
        "sample_count_weighted_mean_date_h_c0_minus_h_s": float(np.average(hc0_vs_hs, weights=date_samples)),
        "equal_recording": {
            "descriptive_only": True,
            "recordings": len(recording_rows),
            "mean_h_c_minus_h_c0": float(recording_hc_hc0.mean()),
            "positive_h_c_minus_h_c0": int(np.count_nonzero(recording_hc_hc0 > 0.0)),
            "negative_h_c_minus_h_c0": int(np.count_nonzero(recording_hc_hc0 < 0.0)),
            "zero_h_c_minus_h_c0": int(np.count_nonzero(recording_hc_hc0 == 0.0)),
            "mean_h_c0_minus_h_s": float(recording_hc0_hs.mean()),
            "positive_h_c0_minus_h_s": int(np.count_nonzero(recording_hc0_hs > 0.0)),
            "negative_h_c0_minus_h_s": int(np.count_nonzero(recording_hc0_hs < 0.0)),
            "zero_h_c0_minus_h_s": int(np.count_nonzero(recording_hc0_hs == 0.0)),
        },
        "per_recording": recording_rows,
        "statistics_limit": (
            "Outer date is the primary inference unit; equal-recording statistics are descriptive nested reports "
            "and do not constitute an independent significance analysis. The sample-count-weighted mean-date "
            "contrasts are not pooled-prediction or global SSE/TSS R2 contrasts because date-specific TSS differs."
        ),
        "causal_read": (
            "H-C minus separately trained topology-matched H-C0 isolates the contribution of carrier content "
            "within the compact consumer family; H-C minus H-S remains the operational comparison."
        ),
        "evaluation_receipts": [
            {"path": str(_output_path(date).resolve()), "sha256": sha256_file(_output_path(date))}
            for date in dates
        ],
    }
    result["canonical_content_sha256"] = _canonical_json_sha(result)
    output = ARTIFACT_ROOT / "H1_CARRIERID_HC0_FIVEDATE_AGGREGATE_v1.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    _write_immutable_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=CONFIRMATORY_DATES)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if bool(args.date) == bool(args.aggregate):
        raise SystemExit("choose exactly one of --date or --aggregate")
    result = aggregate() if args.aggregate else evaluate_date(str(args.date), device_name=args.device)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
