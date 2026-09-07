#!/usr/bin/env python3
"""Explicit one-shot held-source-date evaluator for all six H1-EST4 arms.

No target dependency is imported and no outer-date recording is opened until
the immutable six-arm source checker, checkpoint/config hashes, and evaluator
code closure all validate.  Evaluation is forward-only and hashes every model
state before and after.  The default CLI refuses target access unless
``--execute-target-evaluation`` is supplied explicitly.
"""
from __future__ import annotations

import argparse
import importlib
import json
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

from scripts.h1_carrierid_date_lodo_est4_terminal_checker import (
    CHECKER_SCHEMA,
    CHECKER_STATUS,
    CLOSURE_FILES,
    EST4_ARMS,
)
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256, sha256_file, state_hash, write_immutable_json


EVALUATION_SCHEMA = "h1_carrierid_date_lodo_est4_six_arm_terminal_evaluation_v1"
EVALUATION_DIR = ROOT / "pilot_artifacts" / "h1_carrierid_date_lodo_est4" / "terminal_evaluations"


class Est4TerminalEvaluationError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Est4TerminalEvaluationError(message)


def evaluation_status(date: str) -> str:
    return f"PASS_H1_CARRIERID_DATE_LODO_EST4_{date}_SIX_ARM_EVALUATED"


def canonical_output(date: str) -> Path:
    _need(date in CONFIRMATORY_DATES, "EST4 evaluator date is not canonical")
    return (EVALUATION_DIR / f"H1_CARRIERID_DATE_LODO_EST4_{date}_SIX_ARM_TERMINAL_EVALUATION_v1.json").resolve()


def _one_shot_slot(date: str, output: str | Path) -> Path:
    requested, canonical = Path(output).resolve(), canonical_output(date)
    _need(requested == canonical, f"EST4 evaluation output must be canonical: {canonical}")
    _need(not requested.exists() and not requested.is_symlink() and not os.path.lexists(str(requested)),
          "EST4 evaluator refuses an existing canonical receipt")
    if EVALUATION_DIR.is_dir():
        for candidate in EVALUATION_DIR.glob("*.json"):
            try:
                body = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            _need(not (isinstance(body, Mapping) and body.get("schema") == EVALUATION_SCHEMA
                       and body.get("outer_date") == date),
                  "EST4 evaluator refuses repeat held-source-date opening")
    return canonical


def _read_checker(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    candidate = Path(path).resolve()
    _need(candidate.is_file() and stat.S_IMODE(candidate.stat().st_mode) == 0o444,
          "EST4 evaluator requires immutable mode-0444 source checker")
    body = json.loads(candidate.read_text(encoding="utf-8"))
    _need(isinstance(body, dict) and body.get("schema") == CHECKER_SCHEMA
          and body.get("status") == CHECKER_STATUS,
          "EST4 source checker schema/status drift")
    for relative in CLOSURE_FILES:
        _need(body.get("code_sha256", {}).get(relative) == sha256_file(ROOT / relative),
              f"EST4 evaluator code closure drift at {relative}")
    return candidate, body, sha256_file(candidate)


def _checkpoint_rows(checker: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = checker.get("checkpoints")
    _need(isinstance(rows, Mapping) and set(rows) == set(EST4_ARMS),
          "EST4 checker does not bind exactly six arms")
    for arm in EST4_ARMS:
        row = rows[arm]
        _need(isinstance(row, Mapping), f"EST4 checker row malformed: {arm}")
        checkpoint = Path(str(row.get("path", ""))).resolve()
        config = checkpoint.parent.parent.parent / ".hydra" / "config.yaml"
        _need(checkpoint.is_file() and config.is_file()
              and row.get("sha256") == sha256_file(checkpoint)
              and row.get("metadata", {}).get("config_sha256") == sha256_file(config),
              f"EST4 checkpoint/config changed after source checker: {arm}")
        metadata = row.get("metadata")
        _need(isinstance(metadata, Mapping) and metadata.get("arm") == arm
              and metadata.get("checkpoint_epoch_zero_based") == 49
              and metadata.get("epochs_completed") == 50
              and metadata.get("target_optimizer_steps") == 0
              and metadata.get("target_backward_steps") == 0
              and metadata.get("checkpoint_warm_start") is False
              and metadata.get("target_evaluator_status") == "IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED",
              f"EST4 source checkpoint is not deployable fixed-e49 evidence: {arm}")
    return {arm: rows[arm] for arm in EST4_ARMS}


def _read_est4_preflight(checker: Mapping[str, Any], date: str) -> tuple[Path, dict[str, Any], str]:
    row = checker.get("est4_preflight")
    _need(isinstance(row, Mapping), "EST4 checker lacks source preflight binding")
    path = Path(str(row.get("path", ""))).resolve()
    _need(path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o444
          and row.get("sha256") == sha256_file(path),
          "EST4 source preflight changed after source checker")
    body = json.loads(path.read_text(encoding="utf-8"))
    _need(body.get("schema") == "h1_carrierid_date_lodo_est4_cpu_preflight_v1"
          and body.get("status") == "PASS_H1_CARRIERID_DATE_LODO_EST4_SOURCE_ONLY_NOT_LAUNCHED"
          and body.get("outer_date") == date,
          "EST4 source preflight schema/status/date drift")
    source = body.get("source_binding")
    _need(isinstance(source, Mapping) and checker.get("source_binding_sha256") == canonical_sha256(source)
          and body.get("source_binding_sha256") == canonical_sha256(source),
          "EST4 checker/source preflight binding drift")
    return path, body, sha256_file(path)


def _instantiate(row: Mapping[str, Any], device: torch.device):
    checkpoint = Path(str(row["path"])).resolve()
    config_path = checkpoint.parent.parent.parent / ".hydra" / "config.yaml"
    config = OmegaConf.load(config_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _need(isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping),
          "EST4 terminal checkpoint is malformed")
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
    _need(np.isfinite(sse) and np.isfinite(tss) and tss > 0.0, "EST4 strict R2 is undefined")
    return float(1.0 - sse / tss)


def _evaluate(model: Any, dataset: Any, device: torch.device, sessions: tuple[str, ...]) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0)
    before = state_hash(model.state_dict())
    predictions: list[np.ndarray] = []; targets: list[np.ndarray] = []
    names: list[str] = []; batch_sizes: list[int] = []
    with torch.no_grad():
        for neural, target, identity, session, carrier, rates, labels, mask, permutation in loader:
            output = model(
                neural.to(device=device, dtype=torch.float32),
                calib_trialized_neural_features=identity.to(device=device, dtype=torch.float32),
                carrier=carrier.to(device=device, dtype=torch.float32),
                calibration_rates=rates.to(device=device, dtype=torch.float32),
                calibration_labels=labels.to(device=device, dtype=torch.float32),
                calibration_mask=mask.to(device=device, dtype=torch.bool),
                row_permutation=permutation.to(device=device, dtype=torch.long),
            )
            if bool(model.hparams.decode_last_timestep_only):
                output, target = output[:, -1:, :], target[:, -1:, :]
            if bool(model.hparams.predict_scaled_behavior):
                output = output / model.hparams.behavior_scaling_factor
            predictions.append(output[:, -1, :].detach().cpu().numpy())
            targets.append(target[:, -1, :].detach().cpu().numpy())
            names.extend(str(value) for value in session); batch_sizes.append(int(output.shape[0]))
    after = state_hash(model.state_dict())
    _need(before == after, "EST4 evaluator mutated frozen model state")
    _need(batch_sizes and sum(batch_sizes) == len(dataset)
          and batch_sizes[-1] == (len(dataset) % 32 or 32),
          "EST4 evaluator dropped a batch or final remainder")
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    _need(tuple(sorted(set(names))) == tuple(sorted(sessions)),
          "EST4 evaluator omitted or added a held-source-date session")
    per_session: dict[str, Any] = {}
    for name in sessions:
        selected = np.asarray([value == name for value in names], dtype=bool)
        _need(selected.any(), f"EST4 evaluator omitted session {name}")
        per_session[name] = {"samples": int(selected.sum()), "r2": _r2(target[selected], prediction[selected])}
    return {
        "pooled_r2": _r2(target, prediction), "per_session": per_session,
        "samples": len(dataset), "batches": len(batch_sizes), "last_batch_size": batch_sizes[-1],
        "r2_accumulator_dtype": "float64", "state_sha256_before": before,
        "state_sha256_after": after, "state_immutable": True,
        "query_window_indices_sha256": dataset.window_indices_sha256,
    }


def evaluate(
    *, terminal_checker: str | Path, data_dir: str | Path, output: str | Path, device: str,
) -> dict[str, Any]:
    """Open one held source date only after the complete source gate passes."""

    _need(device in {"cpu", "cuda"} and (device != "cuda" or torch.cuda.is_available()),
          "EST4 evaluator device is unavailable")
    checker_path, checker, checker_sha = _read_checker(terminal_checker)
    date = str(checker.get("outer_date", ""))
    _need(date in CONFIRMATORY_DATES, "EST4 checker outer date is not canonical")
    output_path = _one_shot_slot(date, output)
    rows = _checkpoint_rows(checker)
    preflight_path, preflight, preflight_sha = _read_est4_preflight(checker, date)
    source = preflight["source_binding"]
    # All immutable code/checkpoint/config checks above precede the first
    # import of a target data route and the first outer-date byte access.
    target_base = importlib.import_module("src.data.h1_carrierid_date_lodo_target")
    target_est4 = importlib.import_module("src.data.h1_carrierid_date_lodo_est4_target")
    plan, normalizer, _manifest = target_base.load_target_dependencies(
        source["source_manifest_path"], outer_date=date,
    )
    evaluation_device = torch.device(device)
    models = {arm: _instantiate(rows[arm], evaluation_device) for arm in EST4_ARMS}
    records = target_base.load_outer_date_target_records(data_dir, outer_date=date)
    datasets = {
        arm: target_est4.H1CarrierIdDateLodoEst4StrictTargetDataset(
            records, plan, normalizer, outer_date=date, est4_arm=arm,
        )
        for arm in EST4_ARMS
    }
    query_hashes = {dataset.window_indices_sha256 for dataset in datasets.values()}
    _need(len(query_hashes) == 1, "EST4 six arms do not share exact strict query windows")
    sessions = tuple(records)
    metrics = {arm: _evaluate(models[arm], datasets[arm], evaluation_device, sessions) for arm in EST4_ARMS}
    _need(len({row["query_window_indices_sha256"] for row in metrics.values()}) == 1,
          "EST4 metric views disagree on strict query windows")
    body = {
        "schema": EVALUATION_SCHEMA, "status": evaluation_status(date),
        "outer_date": date, "device": str(evaluation_device),
        "terminal_checker": {"path": str(checker_path), "sha256": checker_sha},
        "est4_preflight": {"path": str(preflight_path), "sha256": preflight_sha},
        "checkpoints": {
            arm: {"path": rows[arm]["path"], "sha256": rows[arm]["sha256"],
                  "metadata": rows[arm]["metadata"]}
            for arm in EST4_ARMS
        },
        "target": {
            "sessions": list(sessions), "files": {name: records[name].input_sha256 for name in sessions},
            "strict_datasets": {arm: datasets[arm].manifest() for arm in EST4_ARMS},
            "shared_query_window_indices_sha256": next(iter(query_hashes)),
            "all_query_histories_start_at_or_after_fifth_trial": True,
        },
        "metrics": metrics,
        "deployment_updates": {"optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged": True},
        "one_shot": {"canonical_output_path": str(output_path),
                     "same_date_prior_terminal_evaluation_receipts": 0},
        "scope": {"opened": "declared public held-in-calib outer-date recordings only after the six-arm source gate",
                  "formal_heldout_opened": False, "minival_opened": False, "evalai_opened": False},
    }
    written, digest = write_immutable_json(output_path, body)
    return {"status": evaluation_status(date), "receipt_path": str(written), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-checker", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--execute-target-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.execute_target_evaluation:
        raise SystemExit(
            "refusing target access: pass --execute-target-evaluation only after reviewing the six-arm source checker"
        )
    print(json.dumps(evaluate(terminal_checker=args.terminal_checker, data_dir=args.data_dir,
                              output=args.output, device=args.device), sort_keys=True))


if __name__ == "__main__":
    main()
