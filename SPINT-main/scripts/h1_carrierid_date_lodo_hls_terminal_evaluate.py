#!/usr/bin/env python3
"""Explicit one-shot H-C versus h=32 H-LS outer-date evaluator.

All five H-LS source terminals and the completed original H-S/H-C aggregate
must be sealed before this module imports a target loader.  Evaluation is
forward-only and uses the same M=4 support, identity tensor, and strict query
windows for H-C and H-LS; only the carrier refit differs.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h1_carrierid_date_lodo_hls_fivedate_contract import (
    DATES, EVALUATOR_PREFLIGHT_SCHEMA, EVALUATOR_PREFLIGHT_STATUS,
    SOURCE_PREFLIGHT_SCHEMA, SOURCE_PREFLIGHT_STATUS, SOURCE_TERMINAL_SCHEMA,
    SOURCE_TERMINAL_STATUS, UPSTREAM_AGGREGATE_SCHEMA, UPSTREAM_AGGREGATE_STATUS,
    read_immutable_json, sha256_file, write_immutable_json,
)
from scripts.h1_carrierid_date_lodo_hls_fivedate_terminal_checker import CHECKER_SCHEMA, CHECKER_STATUS
from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    EARLY_PREFLIGHT_SCHEMA, EARLY_PREFLIGHT_STATUS, EARLY_TERMINAL_SCHEMA,
    EARLY_TERMINAL_STATUS, POST_BINDER_SCHEMA, POST_BINDER_STATUS,
)
from scripts.h1_carrierid_date_lodo_hls_target_evaluator_closure import (
    CLOSURE_FILES, CLOSURE_SCHEMA, CLOSURE_STATUS,
)
from src.h1_m4_cce_contract import state_hash


EVALUATION_SCHEMA = "h1_carrierid_date_lodo_hls_terminal_evaluation_v1"
EVALUATION_DIR = ROOT / "pilot_artifacts/h1_carrierid_date_lodo_hls/terminal_evaluations"
HC_REPRODUCTION_R2_ABS_TOLERANCE = 1.0e-7
HC_REPRODUCTION_R2_REL_TOLERANCE = 0.0


class HlsTerminalEvaluationError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise HlsTerminalEvaluationError(message)


def _canonical_output(date: str) -> Path:
    return (EVALUATION_DIR / f"H1_CARRIERID_DATE_LODO_HLS_{date}_HC_HLS_TERMINAL_EVALUATION_v1.json").resolve()


def _one_shot_slot(date: str, output: str | Path) -> Path:
    requested, canonical = Path(output).resolve(), _canonical_output(date)
    _need(requested == canonical, f"H-LS evaluation output must be canonical: {canonical}")
    _need(not requested.exists() and not requested.is_symlink() and not os.path.lexists(str(requested)),
          "H-LS evaluator refuses an existing canonical receipt")
    if EVALUATION_DIR.is_dir():
        for candidate in EVALUATION_DIR.glob("*.json"):
            try:
                body = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            _need(not (isinstance(body, Mapping) and body.get("schema") == EVALUATION_SCHEMA
                       and body.get("outer_date") == date),
                  "H-LS evaluator refuses a repeat outer-date opening")
    return canonical


def _read_readiness(path: Path) -> tuple[Path, dict[str, Any], str, dict[str, Any]]:
    closure_path, closure, closure_sha = read_immutable_json(
        path, schema=CLOSURE_SCHEMA, status=CLOSURE_STATUS,
    )
    _need(tuple(closure.get("fixed_grid", ())) == DATES,
          "H-LS evaluator closure does not bind the fixed five-date grid")
    _need(all(closure.get("code_sha256", {}).get(relative) == sha256_file(ROOT / relative)
              for relative in CLOSURE_FILES),
          "H-LS evaluator code closure changed before target opening")
    ready_row = closure.get("evaluator_preflight")
    _need(isinstance(ready_row, Mapping), "H-LS evaluator closure lacks readiness receipt")
    ready_path, ready, ready_sha = read_immutable_json(
        ready_row.get("path", ""), schema=EVALUATOR_PREFLIGHT_SCHEMA, status=EVALUATOR_PREFLIGHT_STATUS,
    )
    _need(ready_sha == ready_row.get("sha256"),
          "H-LS evaluator readiness changed after actual code closure")
    row = ready.get("source_terminal_aggregate")
    _need(isinstance(row, Mapping), "H-LS evaluator preflight lacks source-terminal aggregate")
    terminal_path, terminal, terminal_sha = read_immutable_json(
        row.get("path", ""), schema=CHECKER_SCHEMA, status=CHECKER_STATUS,
    )
    _need(terminal_sha == row.get("sha256") and tuple(terminal.get("fixed_grid", ())) == DATES,
          "H-LS source-terminal aggregate changed after evaluator preflight")
    _need(closure.get("source_terminal_aggregate", {}).get("path") == str(terminal_path)
          and closure.get("source_terminal_aggregate", {}).get("sha256") == terminal_sha,
          "H-LS evaluator closure binds another source-terminal aggregate")
    return closure_path, closure, closure_sha, terminal


def _read_upstream(path: Path, terminal: Mapping[str, Any]) -> tuple[Path, dict[str, Any], str]:
    upstream_path, upstream, upstream_sha = read_immutable_json(
        path, schema=UPSTREAM_AGGREGATE_SCHEMA, status=UPSTREAM_AGGREGATE_STATUS,
    )
    bound = terminal.get("upstream_aggregate")
    _need(isinstance(bound, Mapping) and bound.get("path") == str(upstream_path)
          and bound.get("sha256") == upstream_sha,
          "H-LS terminals bind a different completed H-S/H-C aggregate")
    _need(tuple(upstream.get("required_outer_dates", ())) == DATES
          and upstream.get("all_five_date_receipts_present_and_validated") is True,
          "H-LS evaluator requires the complete five-date upstream aggregate")
    return upstream_path, upstream, upstream_sha


def _read_hls_terminal(terminal: Mapping[str, Any], date: str) -> tuple[Path, dict[str, Any], str]:
    row = terminal.get("terminals", {}).get(date)
    _need(isinstance(row, Mapping), f"H-LS terminal aggregate lacks date {date}")
    if row.get("terminal_schema") == EARLY_TERMINAL_SCHEMA:
        _need(terminal.get("route") == "H1-HLS-EARLY-V2-POST-UPSTREAM"
              and row.get("terminal_status") == EARLY_TERMINAL_STATUS,
              "early-v2 H-LS terminal lacks explicit post-upstream route metadata")
        binder_row = terminal.get("post_upstream_binder")
        _need(isinstance(binder_row, Mapping), "early-v2 H-LS terminal aggregate lacks post-upstream binder")
        binder_path, binder, binder_sha = read_immutable_json(
            binder_row.get("path", ""), schema=POST_BINDER_SCHEMA, status=POST_BINDER_STATUS,
        )
        _need(binder_sha == binder_row.get("sha256")
              and binder.get("all_five_dates_compatible") is True
              and binder.get("dates", {}).get(date, {}).get("early_terminal", {}).get("path") == row.get("path")
              and binder.get("dates", {}).get(date, {}).get("early_terminal", {}).get("sha256") == row.get("sha256"),
              "early-v2 H-LS terminal is not authorized by the complete post-upstream binder")
        _need(binder.get("code_sha256", {}).get("post_upstream_binder")
              == sha256_file(ROOT / "scripts/h1_carrierid_date_lodo_hls_early_v2_post_upstream_binder.py"),
              "early-v2 post-upstream binder code changed before target opening")
        path, body, digest = read_immutable_json(
            row.get("path", ""), schema=EARLY_TERMINAL_SCHEMA, status=EARLY_TERMINAL_STATUS,
        )
        _need(digest == row.get("sha256") and body.get("outer_date") == date
              and body.get("source_binding_sha256") == row.get("source_binding_sha256")
              and body.get("base_source_binding_sha256") == row.get("base_source_binding_sha256"),
              "early-v2 H-LS terminal changed after post-upstream checker")
        return path, body, digest
    path, body, digest = read_immutable_json(
        row.get("path", ""), schema=SOURCE_TERMINAL_SCHEMA, status=SOURCE_TERMINAL_STATUS,
    )
    _need(digest == row.get("sha256") and body.get("outer_date") == date
          and body.get("source_binding_sha256") == row.get("source_binding_sha256"),
          "H-LS individual terminal changed after five-date checker")
    return path, body, digest


def _read_hc_evaluation(upstream: Mapping[str, Any], date: str) -> tuple[Path, dict[str, Any], str]:
    row = upstream.get("per_date", {}).get(date, {}).get("receipt")
    _need(isinstance(row, Mapping), f"upstream aggregate lacks original H-C evaluation row for {date}")
    path = Path(str(row.get("path", ""))).resolve()
    _need(path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444
          and sha256_file(path) == row.get("sha256"),
          "original H-C evaluation receipt changed after aggregate")
    body = json.loads(path.read_text(encoding="utf-8"))
    _need(isinstance(body, dict)
          and body.get("schema") == "h1_carrierid_date_lodo_phase2_terminal_evaluation_v1"
          and body.get("status") == f"PASS_H1_CARRIERID_DATE_LODO_PHASE2_{date}_HS_HC_EVALUATED"
          and body.get("outer_date") == date,
          "original H-C evaluation receipt schema/status/date drift")
    return path, body, sha256_file(path)


def _checkpoint_row(row: Mapping[str, Any], *, arm: str, date: str) -> Mapping[str, Any]:
    checkpoint = Path(str(row.get("path", ""))).resolve()
    config = Path(str(row.get("config_path", ""))).resolve()
    _need(checkpoint.is_file() and config.is_file()
          and row.get("sha256") == sha256_file(checkpoint)
          and row.get("config_sha256") == sha256_file(config),
          f"{arm} checkpoint/config changed before H-LS target opening")
    meta = row.get("metadata")
    _need(isinstance(meta, Mapping) and meta.get("arm") == arm and meta.get("outer_date") == date
          and meta.get("checkpoint_epoch_zero_based") == 49 and meta.get("epochs_completed") == 50
          and meta.get("target_optimizer_steps") == 0 and meta.get("target_backward_steps") == 0
          and meta.get("checkpoint_warm_start") is False,
          f"{arm} terminal metadata is not source-only fixed-e49 evidence")
    return row


def _instantiate(row: Mapping[str, Any], device: torch.device):
    config = OmegaConf.load(Path(str(row["config_path"])))
    payload = torch.load(Path(str(row["path"])), map_location="cpu", weights_only=False)
    _need(isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping),
          "H-C/H-LS terminal checkpoint is malformed")
    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device); model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _r2(truth: np.ndarray, estimate: np.ndarray) -> float:
    truth, estimate = np.asarray(truth, dtype=np.float64), np.asarray(estimate, dtype=np.float64)
    sse = float(np.square(truth - estimate).sum())
    tss = float(np.square(truth - truth.mean(axis=0, keepdims=True)).sum())
    _need(np.isfinite(sse) and np.isfinite(tss) and tss > 0.0, "H-LS strict R2 is undefined")
    return float(1.0 - sse / tss)


def _validate_original_hc_reproduction(
    original_evaluation: Mapping[str, Any], observed: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the newly evaluated H-C reproduces its sealed result.

    Structural quantities are exact.  Standard SSE/TSS R2 values are
    accumulated in float64 and compared with zero relative tolerance and the
    single predeclared absolute tolerance above.  The returned audit is
    written into the paired receipt only after every comparison passes.
    """

    original_metrics = original_evaluation.get("metrics")
    _need(isinstance(original_metrics, Mapping),
          "original sealed H-C evaluation lacks metrics mapping")
    expected = original_metrics.get("h_c")
    _need(isinstance(expected, Mapping),
          "original sealed H-C evaluation lacks metrics.h_c mapping")
    _need(isinstance(observed, Mapping), "new H-C evaluation metrics are malformed")
    _need(expected.get("r2_accumulator_dtype") == observed.get("r2_accumulator_dtype") == "float64",
          "original/new H-C R2 accumulator is not float64")

    exact_fields = ("query_window_indices_sha256", "samples", "batches", "last_batch_size")
    exact_checks: dict[str, Any] = {}
    for field in exact_fields:
        _need(field in expected and field in observed,
              f"original/new H-C metric lacks exact reproduction field: {field}")
        _need(expected[field] == observed[field],
              f"new H-C does not reproduce original {field}")
        exact_checks[field] = {"original": expected[field], "recomputed": observed[field], "equal": True}

    expected_pooled, observed_pooled = expected.get("pooled_r2"), observed.get("pooled_r2")
    _need(isinstance(expected_pooled, (int, float)) and math.isfinite(float(expected_pooled))
          and isinstance(observed_pooled, (int, float)) and math.isfinite(float(observed_pooled)),
          "original/new H-C pooled standard R2 is missing or nonfinite")
    pooled_difference = abs(float(observed_pooled) - float(expected_pooled))
    _need(math.isclose(float(observed_pooled), float(expected_pooled),
                       rel_tol=HC_REPRODUCTION_R2_REL_TOLERANCE,
                       abs_tol=HC_REPRODUCTION_R2_ABS_TOLERANCE),
          "new H-C pooled standard R2 does not reproduce original sealed H-C")

    expected_sessions, observed_sessions = expected.get("per_session"), observed.get("per_session")
    _need(isinstance(expected_sessions, Mapping) and isinstance(observed_sessions, Mapping),
          "original/new H-C per-recording metrics mapping is missing")
    _need(tuple(expected_sessions) == tuple(observed_sessions),
          "new H-C per-recording names/order do not reproduce original sealed H-C")
    recording_checks: dict[str, Any] = {}
    for session in expected_sessions:
        expected_row, observed_row = expected_sessions[session], observed_sessions[session]
        _need(isinstance(expected_row, Mapping) and isinstance(observed_row, Mapping),
              f"original/new H-C recording metric row is malformed: {session}")
        _need(isinstance(expected_row.get("samples"), int)
              and expected_row.get("samples") == observed_row.get("samples"),
              f"new H-C recording sample count does not reproduce original: {session}")
        expected_r2, observed_r2 = expected_row.get("r2"), observed_row.get("r2")
        _need(isinstance(expected_r2, (int, float)) and math.isfinite(float(expected_r2))
              and isinstance(observed_r2, (int, float)) and math.isfinite(float(observed_r2)),
              f"original/new H-C recording R2 is missing or nonfinite: {session}")
        difference = abs(float(observed_r2) - float(expected_r2))
        _need(math.isclose(float(observed_r2), float(expected_r2),
                           rel_tol=HC_REPRODUCTION_R2_REL_TOLERANCE,
                           abs_tol=HC_REPRODUCTION_R2_ABS_TOLERANCE),
              f"new H-C recording R2 does not reproduce original: {session}")
        recording_checks[str(session)] = {
            "samples": {"original": int(expected_row["samples"]),
                        "recomputed": int(observed_row["samples"]), "equal": True},
            "r2": {"original": float(expected_r2), "recomputed": float(observed_r2),
                   "absolute_difference": difference, "within_tolerance": True},
        }
    return {
        "passed": True,
        "required_before_hls_paired_result_publication": True,
        "metric_definition": "standard pooled/per-recording SSE-over-TSS R2 accumulated in float64",
        "tolerance": {"r2_absolute": HC_REPRODUCTION_R2_ABS_TOLERANCE,
                      "r2_relative": HC_REPRODUCTION_R2_REL_TOLERANCE},
        "exact_checks": exact_checks,
        "pooled_r2": {"original": float(expected_pooled), "recomputed": float(observed_pooled),
                      "absolute_difference": pooled_difference, "within_tolerance": True},
        "per_recording_names": list(expected_sessions),
        "per_recording_names_and_order_equal": True,
        "per_recording": recording_checks,
    }


def _validate_execution_route(terminal: Mapping[str, Any], *, early_route: bool,
                              device: str, execution_owner: str) -> str:
    """Bind early-v2 replay to the original 5070 Ti execution owner."""

    if not early_route:
        return "cpu" if device == "cpu" else str(torch.cuda.get_device_name(torch.device(device)))
    required_execution = terminal.get("required_target_execution")
    _need(isinstance(required_execution, Mapping)
          and required_execution.get("owner") == "remote5070ti_original_hc_replay"
          and required_execution.get("device_type") == "cuda"
          and required_execution.get("device_name_must_contain") == "5070 Ti",
          "early-v2 terminal checker lost the original-5070 H-C replay requirement")
    _need(device == "cuda" and execution_owner == "remote5070ti_original_hc_replay",
          "early-v2 target evaluation must run under the declared original 5070 Ti owner")
    device_name = str(torch.cuda.get_device_name(torch.device(device)))
    _need("5070 Ti" in device_name,
          "early-v2 target evaluation device is not the original 5070 Ti class")
    return device_name


def _evaluate(model: Any, dataset: Any, device: torch.device, sessions: tuple[str, ...]) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0)
    before = state_hash(model.state_dict())
    predictions: list[np.ndarray] = []; targets: list[np.ndarray] = []
    names: list[str] = []; batch_sizes: list[int] = []
    with torch.no_grad():
        for neural, target, identity, session, carrier in loader:
            output = model(neural.to(device=device, dtype=torch.float32),
                           calib_trialized_neural_features=identity.to(device=device, dtype=torch.float32),
                           carrier=carrier.to(device=device, dtype=torch.float32))
            if bool(model.hparams.decode_last_timestep_only):
                output, target = output[:, -1:, :], target[:, -1:, :]
            if bool(model.hparams.predict_scaled_behavior):
                output = output / model.hparams.behavior_scaling_factor
            predictions.append(output[:, -1, :].detach().cpu().numpy())
            targets.append(target[:, -1, :].detach().cpu().numpy())
            names.extend(str(value) for value in session); batch_sizes.append(int(output.shape[0]))
    after = state_hash(model.state_dict())
    _need(before == after, "H-C/H-LS evaluator mutated frozen model state")
    _need(batch_sizes and sum(batch_sizes) == len(dataset)
          and batch_sizes[-1] == (len(dataset) % 32 or 32),
          "H-C/H-LS evaluator dropped a target remainder")
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    _need(tuple(sorted(set(names))) == tuple(sorted(sessions)), "H-C/H-LS evaluator omitted a target session")
    per_session = {}
    for name in sessions:
        mask = np.asarray([value == name for value in names], dtype=bool)
        _need(mask.any(), f"H-C/H-LS evaluator omitted target session {name}")
        per_session[name] = {"samples": int(mask.sum()), "r2": _r2(target[mask], prediction[mask])}
    return {
        "pooled_r2": _r2(target, prediction), "per_session": per_session,
        "samples": len(dataset), "batches": len(batch_sizes), "last_batch_size": batch_sizes[-1],
        "r2_accumulator_dtype": "float64", "state_sha256_before": before,
        "state_sha256_after": after, "state_immutable": True,
        "query_window_indices_sha256": dataset.window_indices_sha256,
    }


def evaluate(*, evaluator_closure: Path, upstream_aggregate: Path, outer_date: str,
             data_dir: Path, output: Path, device: str, execution_owner: str = "") -> dict[str, Any]:
    _need(str(outer_date) in DATES, "H-LS evaluator date is outside the fixed grid")
    _need(device in {"cpu", "cuda"} and (device != "cuda" or torch.cuda.is_available()),
          "requested H-LS evaluator device is unavailable")
    output_path = _one_shot_slot(str(outer_date), output)
    closure_path, _closure, closure_sha, terminal = _read_readiness(evaluator_closure)
    upstream_path, upstream, upstream_sha = _read_upstream(upstream_aggregate, terminal)
    hls_terminal_path, hls_terminal, hls_terminal_sha = _read_hls_terminal(terminal, str(outer_date))
    hc_eval_path, hc_eval, hc_eval_sha = _read_hc_evaluation(upstream, str(outer_date))
    hc = _checkpoint_row(hc_eval["checkpoints"]["H-C"], arm="H-C", date=str(outer_date))
    hls = _checkpoint_row(hls_terminal["checkpoint"], arm="H-LS", date=str(outer_date))
    early_route = hls_terminal.get("schema") == EARLY_TERMINAL_SCHEMA
    source_preflight_path, source_preflight, source_preflight_sha = read_immutable_json(
        hls_terminal["source_preflight"]["path"],
        schema=EARLY_PREFLIGHT_SCHEMA if early_route else SOURCE_PREFLIGHT_SCHEMA,
        status=EARLY_PREFLIGHT_STATUS if early_route else SOURCE_PREFLIGHT_STATUS,
    )
    _need(source_preflight_sha == hls_terminal["source_preflight"]["sha256"],
          "H-LS source preflight changed after terminal audit")
    source = source_preflight.get("source_binding")
    _need(isinstance(source, Mapping)
          and source.get("source_manifest_sha256") == hc["metadata"].get("phase1_source_manifest_sha256")
          and source_preflight.get("source_binding_sha256") == hls["metadata"].get("hls_source_binding_sha256"),
          "H-C and H-LS do not share the exact Phase-1 source estimator binding")
    device_name = _validate_execution_route(
        terminal, early_route=early_route, device=device, execution_owner=execution_owner,
    )

    # Target modules are imported only after all immutable receipts,
    # checkpoints, configs, and source bindings above have passed.
    target_module = importlib.import_module("src.data.h1_carrierid_date_lodo_target")
    hls_target_module = importlib.import_module("src.data.h1_carrierid_date_lodo_hls_target")
    plan, normalizer, _manifest = target_module.load_target_dependencies(
        source["source_manifest_path"], outer_date=str(outer_date),
    )
    evaluation_device = torch.device(device)
    hc_model, hls_model = _instantiate(hc, evaluation_device), _instantiate(hls, evaluation_device)
    records = target_module.load_outer_date_target_records(data_dir, outer_date=str(outer_date))
    hc_dataset = target_module.H1CarrierIdDateLodoStrictTargetDataset(
        records, plan, normalizer, outer_date=str(outer_date),
    )
    hls_dataset = hls_target_module.H1CarrierIdDateLodoHlsStrictTargetDataset(
        records, plan, normalizer, outer_date=str(outer_date),
    )
    _need(hc_dataset.window_indices == hls_dataset.window_indices
          and hc_dataset.window_indices_sha256 == hls_dataset.window_indices_sha256,
          "H-C/H-LS strict target query windows differ")
    _need(all(np.array_equal(hc_dataset.support[name].identity, hls_dataset.support[name].identity)
              and hc_dataset.support[name].support_sha256 == hls_dataset.support[name].support_sha256
              and not np.array_equal(hc_dataset.support[name].normalized_carrier,
                                     hls_dataset.support[name].normalized_carrier)
              for name in records),
          "H-C/H-LS target views differ outside the carrier or failed to change it")
    sessions = tuple(records)
    hc_metrics = _evaluate(hc_model, hc_dataset, evaluation_device, sessions)
    hls_metrics = _evaluate(hls_model, hls_dataset, evaluation_device, sessions)
    hc_reproduction = _validate_original_hc_reproduction(hc_eval, hc_metrics)
    delta = float(hc_metrics["pooled_r2"] - hls_metrics["pooled_r2"])
    body = {
        "schema": EVALUATION_SCHEMA,
        "status": f"PASS_H1_CARRIERID_DATE_LODO_HLS_{outer_date}_HC_HLS_EVALUATED",
        "outer_date": str(outer_date), "device": str(evaluation_device),
        "execution_owner": execution_owner or "v1_unspecified_owner",
        "execution_device_name": device_name,
        "evaluator_closure": {"path": str(closure_path), "sha256": closure_sha},
        "upstream_aggregate": {"path": str(upstream_path), "sha256": upstream_sha},
        "hls_source_terminal": {"path": str(hls_terminal_path), "sha256": hls_terminal_sha},
        "original_hc_evaluation": {"path": str(hc_eval_path), "sha256": hc_eval_sha},
        "original_hc_reproduction_check": hc_reproduction,
        "source_preflight": {"path": str(source_preflight_path), "sha256": source_preflight_sha},
        "checkpoints": {"H-C": dict(hc), "H-LS": dict(hls)},
        "target": {
            "sessions": list(sessions), "files": {name: records[name].input_sha256 for name in sessions},
            "h_c_dataset": hc_dataset.manifest(), "h_ls_dataset": hls_dataset.manifest(),
            "shared_query_window_indices_sha256": hc_dataset.window_indices_sha256,
            "same_support_identity_and_query_windows": True,
        },
        "metrics": {"h_c": hc_metrics, "h_ls": hls_metrics, "h_c_minus_h_ls": delta},
        "deployment_updates": {"optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged": True},
        "one_shot": {"canonical_output_path": str(output_path),
                     "same_date_prior_hls_terminal_evaluation_receipts": 0},
        "scope": {"formal_heldout_opened": False, "minival_opened": False, "evalai_opened": False},
    }
    written, digest = write_immutable_json(output_path, body)
    return {"status": body["status"], "receipt_path": str(written), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-closure", required=True, type=Path)
    parser.add_argument("--upstream-aggregate", required=True, type=Path)
    parser.add_argument("--outer-date", required=True, choices=DATES)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--execution-owner", default="")
    parser.add_argument("--execute-target-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.execute_target_evaluation:
        raise SystemExit("refusing target access: pass --execute-target-evaluation after the five-date source gate")
    print(json.dumps(evaluate(evaluator_closure=args.evaluator_closure,
                             upstream_aggregate=args.upstream_aggregate, outer_date=args.outer_date,
                             data_dir=args.data_dir, output=args.output, device=args.device,
                             execution_owner=args.execution_owner), sort_keys=True))


if __name__ == "__main__":
    main()
