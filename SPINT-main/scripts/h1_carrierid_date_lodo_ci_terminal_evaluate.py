#!/usr/bin/env python3
"""Explicit one-shot strict held-source-date evaluator for five H1 CI arms.

No target recording is opened until the immutable five-arm checker, every
checkpoint/config byte hash, and the code closure in that checker validate.
The evaluation path is forward-only: it freezes every parameter, performs no
optimiser/backward operation, and hashes model state before and after each arm.
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

from scripts.h1_carrierid_date_lodo_ci_terminal_checker import (
    CHECKER_SCHEMA, CHECKER_STATUS, CI_ARMS, CLOSURE_FILES,
)
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256, sha256_file, state_hash, write_immutable_json


EVALUATION_SCHEMA = "h1_carrierid_date_lodo_ci_five_arm_terminal_evaluation_v1"
EVALUATION_DIR = ROOT / "pilot_artifacts" / "h1_carrierid_date_lodo_ci" / "terminal_evaluations"


class CiTerminalEvaluationError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise CiTerminalEvaluationError(message)


def _status(date: str) -> str:
    return f"PASS_H1_CARRIERID_DATE_LODO_CI_{date}_FIVE_ARM_EVALUATED"


def _output_for_date(date: str) -> Path:
    return (EVALUATION_DIR / f"H1_CARRIERID_DATE_LODO_CI_{date}_FIVE_ARM_TERMINAL_EVALUATION_v1.json").resolve()


def _read_checker(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    candidate = Path(path).resolve()
    _need(candidate.is_file() and stat.S_IMODE(candidate.stat().st_mode) == 0o444,
          "CI evaluator requires immutable mode-0444 terminal checker")
    body = json.loads(candidate.read_text(encoding="utf-8"))
    _need(isinstance(body, dict) and body.get("schema") == CHECKER_SCHEMA and body.get("status") == CHECKER_STATUS,
          "CI terminal checker schema/status drift")
    for relative in CLOSURE_FILES:
        _need(body.get("code_sha256", {}).get(relative) == sha256_file(ROOT / relative),
              f"CI evaluator code closure drift at {relative}")
    return candidate, body, sha256_file(candidate)


def _one_shot_slot(date: str, output: str | Path) -> Path:
    requested, canonical = Path(output).resolve(), _output_for_date(date)
    _need(requested == canonical, f"CI evaluation output must be canonical per-date receipt: {canonical}")
    _need(not requested.exists() and not requested.is_symlink() and not os.path.lexists(str(requested)),
          "CI evaluator refuses existing canonical receipt")
    if EVALUATION_DIR.is_dir():
        for candidate in EVALUATION_DIR.glob("*.json"):
            try:
                body = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            _need(not (isinstance(body, Mapping) and body.get("schema") == EVALUATION_SCHEMA
                       and body.get("outer_date") == date), "CI evaluator refuses repeat held-source-date opening")
    return canonical


def _rows(checker: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = checker.get("checkpoints")
    _need(isinstance(rows, Mapping) and set(rows) == set(CI_ARMS), "CI checker does not bind exactly five arms")
    for arm in CI_ARMS:
        row = rows[arm]
        _need(isinstance(row, Mapping), f"CI checker row malformed: {arm}")
        checkpoint, config = Path(str(row.get("checkpoint_path", ""))).resolve(), Path(str(row.get("config_path", ""))).resolve()
        _need(checkpoint.is_file() and config.is_file()
              and row.get("checkpoint_sha256") == sha256_file(checkpoint)
              and row.get("config_sha256") == sha256_file(config),
              f"CI checkpoint/config changed after terminal checker: {arm}")
        meta = row.get("metadata")
        _need(isinstance(meta, Mapping) and meta.get("arm") == arm
              and meta.get("checkpoint_epoch_zero_based") == 49
              and meta.get("target_optimizer_steps") == 0 and meta.get("target_backward_steps") == 0
              and meta.get("checkpoint_warm_start") is False,
              f"CI source-only terminal metadata is not deployable: {arm}")
    return {arm: rows[arm] for arm in CI_ARMS}


def _instantiate(row: Mapping[str, Any], device: torch.device):
    config = OmegaConf.load(Path(str(row["config_path"])))
    payload = torch.load(Path(str(row["checkpoint_path"])), map_location="cpu", weights_only=False)
    _need(isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping), "CI checkpoint is malformed")
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
    _need(np.isfinite(sse) and np.isfinite(tss) and tss > 0, "CI strict R2 is undefined")
    return float(1.0 - sse / tss)


def _evaluate(model: Any, dataset: Any, device: torch.device, sessions: tuple[str, ...]) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0)
    before = state_hash(model.state_dict())
    predictions: list[np.ndarray] = []; targets: list[np.ndarray] = []; names: list[str] = []; batch_sizes: list[int] = []
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
    _need(before == after, "CI evaluator mutated frozen model state")
    _need(batch_sizes and sum(batch_sizes) == len(dataset) and batch_sizes[-1] == (len(dataset) % 32 or 32),
          "CI evaluator dropped a batch/remainder")
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    _need(tuple(sorted(set(names))) == tuple(sorted(sessions)), "CI evaluator omitted/added target session")
    per_session = {}
    for name in sessions:
        mask = np.asarray([value == name for value in names], dtype=bool)
        _need(mask.any(), f"CI evaluator omitted target session: {name}")
        per_session[name] = {"samples": int(mask.sum()), "r2": _r2(target[mask], prediction[mask])}
    return {"pooled_r2": _r2(target, prediction), "per_session": per_session, "samples": len(dataset),
            "batches": len(batch_sizes), "last_batch_size": batch_sizes[-1], "r2_accumulator_dtype": "float64",
            "state_sha256_before": before, "state_sha256_after": after, "state_immutable": True,
            "query_window_indices_sha256": dataset.window_indices_sha256}


def evaluate(*, terminal_checker: str | Path, data_dir: str | Path, output: str | Path, device: str) -> dict[str, Any]:
    """One strict forward-only evaluation after every source-only gate is closed."""

    _need(device in {"cpu", "cuda"} and (device != "cuda" or torch.cuda.is_available()), "CI evaluator device unavailable")
    checker_path, checker, checker_sha = _read_checker(terminal_checker)
    date = str(checker.get("outer_date", ""))
    _need(date in CONFIRMATORY_DATES, "CI checker outer date is not canonical")
    output_path = _one_shot_slot(date, output)
    rows = _rows(checker)
    source = checker.get("ci_preflight")
    _need(isinstance(source, Mapping), "CI checker lacks CI preflight receipt")
    # All receipt/config/checkpoint/model construction checks above occur
    # before this import retrieves target dependencies or opens a target NWB.
    target_module = importlib.import_module("src.data.h1_carrierid_date_lodo_target")
    ci_target_module = importlib.import_module("src.data.h1_carrierid_date_lodo_ci_target")
    ci_preflight_path = Path(str(source.get("path", ""))).resolve()
    _need(ci_preflight_path.is_file() and stat.S_IMODE(ci_preflight_path.stat().st_mode) == 0o444
          and source.get("sha256") == sha256_file(ci_preflight_path),
          "CI preflight receipt changed after terminal checker")
    preflight = json.loads(ci_preflight_path.read_text(encoding="utf-8"))
    _need(preflight.get("schema") == "h1_carrierid_date_lodo_ci_cpu_preflight_v1"
          and preflight.get("status") == "PASS_H1_CARRIERID_DATE_LODO_CI_SOURCE_ONLY_NOT_LAUNCHED"
          and preflight.get("outer_date") == date,
          "CI preflight schema/status/date drift before target opening")
    source_binding = preflight.get("source_binding")
    _need(isinstance(source_binding, Mapping) and checker.get("source_binding_sha256") == canonical_sha256(source_binding),
          "CI checker/preflight source binding drift")
    plan, normalizer, _manifest = target_module.load_target_dependencies(
        source_binding["source_manifest_path"], outer_date=date,
    )
    evaluation_device = torch.device(device)
    models = {arm: _instantiate(rows[arm], evaluation_device) for arm in CI_ARMS}
    records = target_module.load_outer_date_target_records(data_dir, outer_date=date)
    datasets = {
        arm: ci_target_module.H1CarrierIdDateLodoCiStrictTargetDataset(
            records, plan, normalizer, outer_date=date, carrier_intervention=arm.split("-", 1)[1].lower(),
        ) for arm in CI_ARMS
    }
    query_hashes = {dataset.window_indices_sha256 for dataset in datasets.values()}
    _need(len(query_hashes) == 1, "CI five arms do not share exact strict post-support target windows")
    _need(datasets["CI64-C0"].manifest()["c0_is_model_boundary_literal_zero"] is True,
          "CI C0 target control is not model-bound")
    for arm in ("CI64-LS", "CI64-RS"):
        _need(datasets[arm].manifest()["ls_rs_use_same_frozen_transform_definition_as_source"] is True,
              f"CI {arm} target carrier transform does not bind source definition")
    sessions = tuple(records)
    metrics = {arm: _evaluate(models[arm], datasets[arm], evaluation_device, sessions) for arm in CI_ARMS}
    _need(len({metric["query_window_indices_sha256"] for metric in metrics.values()}) == 1,
          "CI metric views disagree on target query windows")
    body = {
        "schema": EVALUATION_SCHEMA, "status": _status(date), "outer_date": date, "device": str(evaluation_device),
        "terminal_checker": {"path": str(checker_path), "sha256": checker_sha},
        "checkpoints": {arm: {"path": rows[arm]["checkpoint_path"], "sha256": rows[arm]["checkpoint_sha256"],
                               "config_path": rows[arm]["config_path"], "config_sha256": rows[arm]["config_sha256"],
                               "metadata": rows[arm]["metadata"]} for arm in CI_ARMS},
        "target": {"sessions": list(sessions), "files": {name: records[name].input_sha256 for name in sessions},
                   "strict_datasets": {arm: datasets[arm].manifest() for arm in CI_ARMS},
                   "shared_query_window_indices_sha256": next(iter(query_hashes)),
                   "all_query_histories_start_at_or_after_fifth_trial": True},
        "metrics": metrics,
        "deployment_updates": {"optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged": True},
        "one_shot": {"canonical_output_path": str(output_path), "same_date_prior_terminal_evaluation_receipts": 0},
        "scope": {"opened": "declared public held-in-calib outer-date recordings only after all five source gates",
                  "formal_heldout_opened": False, "minival_opened": False, "evalai_opened": False},
    }
    written, digest = write_immutable_json(output_path, body)
    return {"status": _status(date), "receipt_path": str(written), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-checker", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--execute-target-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.execute_target_evaluation:
        raise SystemExit("refusing target access: pass --execute-target-evaluation only after reviewing five-arm checker")
    print(json.dumps(evaluate(terminal_checker=args.terminal_checker, data_dir=args.data_dir,
                         output=args.output, device=args.device), sort_keys=True))


if __name__ == "__main__":
    main()
