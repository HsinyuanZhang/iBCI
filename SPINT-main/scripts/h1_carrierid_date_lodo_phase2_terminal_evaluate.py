#!/usr/bin/env python3
"""Explicit one-shot outer-date evaluator for the H1 date-LODO H-S/H-C pair.

This program is intentionally unusable without ``--execute-target-evaluation``.
Before its first target-recording open it verifies the immutable pre-open
receipt, every code hash in that receipt, both terminal checkpoint/config byte
identities, and the source-only M=4 plan/normalizer artifacts.  It does not
train, calibrate by gradient, or update model state.
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

from scripts.h1_carrierid_date_lodo_phase2_terminal_preflight import (
    CLOSURE_FILES,
    PREFLIGHT_SCHEMA,
    PREFLIGHT_STATUS,
)
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, sha256_file, state_hash, write_immutable_json


EVALUATION_SCHEMA = "h1_carrierid_date_lodo_phase2_terminal_evaluation_v1"
EVALUATION_ARTIFACT_DIR = ROOT / "pilot_artifacts" / "h1_carrierid_date_lodo_phase2" / "terminal_evaluations"


def _evaluation_status(outer_date: str) -> str:
    return f"PASS_H1_CARRIERID_DATE_LODO_PHASE2_{outer_date}_HS_HC_EVALUATED"


def _canonical_evaluation_output(outer_date: str) -> Path:
    """Return the only legal, date-scoped terminal-evaluation receipt path."""

    _need(outer_date in CONFIRMATORY_DATES, "terminal evaluator output date is not confirmatory")
    return (EVALUATION_ARTIFACT_DIR /
            f"H1_CARRIERID_DATE_LODO_PHASE2_{outer_date}_HS_HC_TERMINAL_EVALUATION_v1.json").resolve()


def _same_date_evaluation_receipts(outer_date: str) -> tuple[Path, ...]:
    """Find existing terminal-evaluation receipts without touching data files.

    Evaluator receipts are deliberately confined to ``pilot_artifacts``.  The
    scan is receipt-only, excludes ``data/``, and catches an older evaluator's
    alternative filename as well as the canonical name.
    """

    artifact_root = EVALUATION_ARTIFACT_DIR.parent
    if not artifact_root.is_dir():
        return ()
    matches: list[Path] = []
    for candidate in artifact_root.rglob("*.json"):
        try:
            text = candidate.read_text(encoding="utf-8")
            body = json.loads(text)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(body, Mapping) and body.get("schema") == EVALUATION_SCHEMA and body.get("outer_date") == outer_date:
            matches.append(candidate.resolve())
    return tuple(sorted(matches))


def _require_one_shot_evaluation_slot(*, outer_date: str, output_path: str | Path) -> Path:
    """Fail closed before target access if this date has ever been evaluated.

    The output argument is intentionally not a destination choice.  Binding it
    to a canonical per-date receipt prevents a second target opening by merely
    changing a filename.  The receipt scan additionally protects against a
    pre-canonical evaluation receipt recorded under an old alternate name.
    """

    requested = Path(output_path).resolve()
    canonical = _canonical_evaluation_output(outer_date)
    _need(requested == canonical,
          f"terminal evaluation output must be the canonical per-date receipt: {canonical}")
    _need(_is_within(requested, ROOT),
          "terminal evaluation receipt must stay in the isolated-stage root")
    _need(not requested.exists() and not requested.is_symlink() and not os.path.lexists(str(requested)),
          "refusing existing canonical terminal-evaluation receipt")
    existing = _same_date_evaluation_receipts(outer_date)
    _need(not existing,
          f"refusing repeat target evaluation for {outer_date}; existing terminal-evaluation receipt(s): {list(existing)}")
    return canonical


class DateLodoEvaluatorError(ValueError):
    """A target opening/evaluation precondition was not satisfied."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise DateLodoEvaluatorError(message)


def _read_preflight(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    resolved = Path(path).resolve()
    _need(resolved.is_file() and stat.S_IMODE(resolved.stat().st_mode) == 0o444,
          "terminal evaluator requires immutable mode-0444 preflight")
    body = json.loads(resolved.read_text(encoding="utf-8"))
    _need(isinstance(body, dict) and body.get("schema") == PREFLIGHT_SCHEMA and body.get("status") == PREFLIGHT_STATUS,
          "terminal evaluator preflight schema/status drift")
    for relative in CLOSURE_FILES:
        _need(body.get("code_sha256", {}).get(relative) == sha256_file(ROOT / relative),
              f"terminal evaluator source closure drift at {relative}")
    return resolved, body, sha256_file(resolved)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _target_data_root_from_preflight(preflight: Mapping[str, Any]) -> Path:
    """Recover the pre-open-bound canonical public data root without scanning it."""

    runtime = preflight.get("runtime")
    _need(isinstance(runtime, Mapping), "terminal preflight lacks runtime path binding")
    stage_root = Path(str(runtime.get("isolated_stage_root", ""))).resolve()
    canonical_repository_root = Path(str(runtime.get("canonical_data_repository_root", ""))).resolve()
    target_data_root = Path(str(runtime.get("target_data_root", ""))).resolve()
    _need(stage_root == ROOT.resolve(),
          "terminal evaluator must run from the exact isolated-stage code root bound by preflight")
    _need(canonical_repository_root.name == "SPINT-main"
          and target_data_root.is_dir()
          and target_data_root == canonical_repository_root / "data" / "000954",
          "terminal preflight canonical target-data path is malformed or unavailable")
    source = preflight.get("source_binding")
    _need(isinstance(source, Mapping), "terminal preflight lacks source binding")
    source_manifest = Path(str(source.get("source_manifest_path", ""))).resolve()
    _need(_is_within(source_manifest, canonical_repository_root / "pilot_artifacts")
          and source_manifest.is_file()
          and sha256_file(source_manifest) == source.get("source_manifest_sha256"),
          "terminal preflight source bundle no longer matches the canonical target-data repository")
    return target_data_root


def _checkpoint_config_rows(preflight: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = preflight.get("checkpoints")
    _need(isinstance(rows, Mapping) and set(rows) == {"H-S", "H-C"}, "preflight lacks exact H-S/H-C rows")
    for arm, row in rows.items():
        _need(isinstance(row, Mapping), f"malformed {arm} preflight row")
        checkpoint, config = Path(str(row.get("checkpoint_path", ""))).resolve(), Path(str(row.get("config_path", ""))).resolve()
        _need(checkpoint.is_file() and config.is_file(), f"{arm} terminal checkpoint/config no longer exists")
        _need(row.get("checkpoint_sha256") == sha256_file(checkpoint) and row.get("config_sha256") == sha256_file(config),
              f"{arm} checkpoint/config changed after pre-open receipt")
        meta = row.get("metadata")
        _need(isinstance(meta, Mapping) and meta.get("arm") == arm and meta.get("checkpoint_epoch_zero_based") == 49
              and meta.get("target_optimizer_steps") == 0 and meta.get("target_backward_steps") == 0
              and meta.get("checkpoint_warm_start") is False, f"{arm} checkpoint metadata is not deployable source-only evidence")
    return {arm: rows[arm] for arm in ("H-S", "H-C")}


def _instantiate(row: Mapping[str, Any], device: torch.device):
    config = OmegaConf.load(Path(str(row["config_path"])))
    payload = torch.load(Path(str(row["checkpoint_path"])), map_location="cpu", weights_only=False)
    _need(isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping), "terminal checkpoint is malformed")
    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device); model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _r2(truth: np.ndarray, estimate: np.ndarray) -> float:
    actual, predicted = np.asarray(truth, dtype=np.float64), np.asarray(estimate, dtype=np.float64)
    sse = float(np.square(actual - predicted).sum())
    centered = actual - actual.mean(axis=0, keepdims=True)
    tss = float(np.square(centered).sum())
    _need(np.isfinite(sse) and np.isfinite(tss) and tss > 0.0, "strict outer-date R2 is undefined")
    return float(1.0 - sse / tss)


def _evaluate(model: Any, dataset: Any, device: torch.device, sessions: tuple[str, ...]) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0)
    before = state_hash(model.state_dict())
    prediction_rows: list[np.ndarray] = []; target_rows: list[np.ndarray] = []; names: list[str] = []; batch_sizes: list[int] = []
    with torch.no_grad():
        for neural, target, identity, session, carrier in loader:
            output = model(neural.to(device=device, dtype=torch.float32),
                           calib_trialized_neural_features=identity.to(device=device, dtype=torch.float32),
                           carrier=carrier.to(device=device, dtype=torch.float32))
            if bool(model.hparams.decode_last_timestep_only):
                output, target = output[:, -1:, :], target[:, -1:, :]
            if bool(model.hparams.predict_scaled_behavior):
                output = output / model.hparams.behavior_scaling_factor
            prediction_rows.append(output[:, -1, :].detach().cpu().numpy())
            target_rows.append(target[:, -1, :].detach().cpu().numpy())
            names.extend(str(item) for item in session); batch_sizes.append(int(output.shape[0]))
    after = state_hash(model.state_dict())
    _need(before == after, "terminal evaluator mutated a frozen model")
    _need(bool(batch_sizes) and sum(batch_sizes) == len(dataset) and batch_sizes[-1] == (len(dataset) % 32 or 32),
          "target evaluator dropped a batch or final remainder")
    prediction, target = np.concatenate(prediction_rows, axis=0), np.concatenate(target_rows, axis=0)
    _need(tuple(sorted(set(names))) == tuple(sorted(sessions)), "target evaluator omitted/added an outer-date session")
    per_session = {}
    for name in sessions:
        mask = np.asarray([item == name for item in names], dtype=bool)
        _need(bool(mask.any()), f"outer-date session omitted: {name}")
        per_session[name] = {"samples": int(mask.sum()), "r2": _r2(target[mask], prediction[mask])}
    return {"pooled_r2": _r2(target, prediction), "per_session": per_session, "samples": len(dataset),
            "batches": len(batch_sizes), "last_batch_size": batch_sizes[-1], "r2_accumulator_dtype": "float64",
            "state_sha256_before": before, "state_sha256_after": after, "state_immutable": True,
            "query_window_indices_sha256": dataset.window_indices_sha256}


def evaluate(*, preflight_path: str | Path, output_path: str | Path, device: str) -> dict[str, Any]:
    """Perform one target evaluation only after all receipt/checkpoint checks pass."""

    _need(device in {"cpu", "cuda"}, "device must be cpu or cuda")
    _need(device != "cuda" or torch.cuda.is_available(), "requested CUDA evaluator is unavailable")
    preflight_path, preflight, preflight_sha = _read_preflight(preflight_path)
    outer_date = str(preflight.get("outer_date", ""))
    _need(outer_date in CONFIRMATORY_DATES, "terminal evaluator preflight outer_date is not confirmatory")
    output_path = _require_one_shot_evaluation_slot(outer_date=outer_date, output_path=output_path)
    _need(_is_within(preflight_path, ROOT) and _is_within(output_path, ROOT),
          "terminal preflight and evaluation receipt must stay in the isolated-stage root")
    target_data_root = _target_data_root_from_preflight(preflight)
    rows = _checkpoint_config_rows(preflight)
    for arm, row in rows.items():
        _need(_is_within(Path(str(row["checkpoint_path"])), ROOT)
              and _is_within(Path(str(row["config_path"])), ROOT),
              f"{arm} terminal checkpoint/config is outside the isolated-stage root")
    source = preflight.get("source_binding", {})
    _need(isinstance(source, Mapping) and source.get("sha256") == rows["H-S"]["metadata"].get("phase2_source_binding_sha256")
          and source.get("sha256") == rows["H-C"]["metadata"].get("phase2_source_binding_sha256"),
          "terminal checkpoints do not bind the preflight source schedule")
    # Still no target byte access: reconstruct the immutable estimator and source RMS normalizer from Phase-1 artifacts.
    target_module = importlib.import_module("src.data.h1_carrierid_date_lodo_target")
    plan, normalizer, source_manifest = target_module.load_target_dependencies(
        source["source_manifest_path"], outer_date=outer_date,
    )
    evaluation_device = torch.device(device)
    hs_model, hc_model = _instantiate(rows["H-S"], evaluation_device), _instantiate(rows["H-C"], evaluation_device)
    # The first target data operation is deliberately below every immutable binding above.
    records = target_module.load_outer_date_target_records(target_data_root, outer_date=outer_date)
    dataset = target_module.H1CarrierIdDateLodoStrictTargetDataset(records, plan, normalizer, outer_date=outer_date)
    sessions = tuple(records)
    hs_metrics, hc_metrics = _evaluate(hs_model, dataset, evaluation_device, sessions), _evaluate(hc_model, dataset, evaluation_device, sessions)
    _need(hs_metrics["query_window_indices_sha256"] == hc_metrics["query_window_indices_sha256"]
          == dataset.window_indices_sha256, "H-S/H-C target datasets do not share exact strict support/query windows")
    body = {
        "schema": EVALUATION_SCHEMA, "status": _evaluation_status(outer_date), "outer_date": outer_date, "device": str(evaluation_device),
        "preflight": {"path": str(preflight_path), "sha256": preflight_sha}, "source_manifest_sha256": sha256_file(Path(source["source_manifest_path"])),
        "checkpoints": {arm: {"path": row["checkpoint_path"], "sha256": row["checkpoint_sha256"],
                               "config_path": row["config_path"], "config_sha256": row["config_sha256"], "metadata": row["metadata"]}
                        for arm, row in rows.items()},
        "target": {"sessions": list(sessions), "files": {name: records[name].input_sha256 for name in sessions},
                   "strict_dataset": dataset.manifest(), "all_query_histories_start_at_or_after_fifth_trial": True},
        "metrics": {"h_s": hs_metrics, "h_c": hc_metrics, "h_c_minus_h_s": float(hc_metrics["pooled_r2"] - hs_metrics["pooled_r2"])},
        "deployment_updates": {"optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged": True},
        "one_shot": {"canonical_output_path": str(output_path),
                     "same_date_prior_terminal_evaluation_receipts": 0},
        "scope": {"opened": "only the declared public held-in-calib outer-date recordings after pair gate", "formal_heldout_opened": False,
                  "minival_opened": False, "evalai_opened": False},
    }
    written, digest = write_immutable_json(output_path, body)
    return {"status": _evaluation_status(outer_date), "receipt_path": str(written), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--execute-target-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.execute_target_evaluation:
        raise SystemExit("refusing target access: pass --execute-target-evaluation only after reviewing the immutable pair gate")
    print(json.dumps(evaluate(preflight_path=args.preflight, output_path=args.output, device=args.device), sort_keys=True))


if __name__ == "__main__":
    main()
