#!/usr/bin/env python3
"""One strict target evaluation per date for all fold-0-eligible compact arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from src.data.h1_spint_width_date_lodo import (
    H1SpintWidthDateLodoSourceDataModule,
    H1SpintWidthDateLodoStrictTargetDataset,
    load_spint_width_date_target_records,
)
from src.h1_m4_cce_contract import sha256_file, state_hash, write_immutable_json
from src.models.h1_spint_width_date_lodo_module import DATE_CHECKPOINT_SCHEMA
from src.models.h1_spint_width_module import WIDTH_ARMS


PREFLIGHT_SCHEMA = "h1_spint_identity_width_date_lodo_cpu_preflight_v1"
EVALUATION_SCHEMA = "h1_spint_identity_width_date_lodo_terminal_evaluation_v1"
MARGIN = 0.03
FOLLOWUP_DATES = ("19250108", "19250113", "19250115", "19250119")


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_immutable(path: str | Path, schema: str) -> tuple[Path, dict[str, Any], str]:
    candidate = Path(path).resolve()
    _need(candidate.is_file() and stat.S_IMODE(candidate.stat().st_mode) == 0o444, f"immutable receipt missing/mutable: {candidate}")
    body = json.loads(candidate.read_text(encoding="utf-8"))
    _need(isinstance(body, dict) and body.get("schema") == schema, f"receipt schema drift: {candidate}")
    return candidate, body, sha256_file(candidate)


def _r2(truth: np.ndarray, estimate: np.ndarray) -> float:
    y, yhat = np.asarray(truth, dtype=np.float64), np.asarray(estimate, dtype=np.float64)
    sse = float(np.square(y - yhat).sum())
    tss = float(np.square(y - y.mean(axis=0, keepdims=True)).sum())
    _need(np.isfinite(sse) and np.isfinite(tss) and tss > 0.0, "date R2 undefined")
    return float(1.0 - sse / tss)


def _evaluate(model: Any, dataset: H1SpintWidthDateLodoStrictTargetDataset, device: torch.device, sessions: tuple[str, ...]) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0)
    before = state_hash(model.state_dict())
    estimates: list[np.ndarray] = []; truths: list[np.ndarray] = []; names: list[str] = []; sizes: list[int] = []
    with torch.no_grad():
        for batch in loader:
            _need(len(batch) == 4, "strict compact target batch has carrier field")
            neural, target, identity, session = batch
            output = model(neural.to(device=device, dtype=torch.float32),
                           calib_trialized_neural_features=identity.to(device=device, dtype=torch.float32))
            if bool(model.hparams.decode_last_timestep_only):
                output, target = output[:, -1:, :], target[:, -1:, :]
            if bool(model.hparams.predict_scaled_behavior):
                output = output / model.hparams.behavior_scaling_factor
            estimates.append(output[:, -1, :].detach().cpu().numpy()); truths.append(target[:, -1, :].detach().cpu().numpy())
            names.extend(str(item) for item in session); sizes.append(int(output.shape[0]))
    after = state_hash(model.state_dict())
    _need(before == after, "target evaluation mutated compact model state")
    _need(bool(sizes) and sum(sizes) == len(dataset) and sizes[-1] == (len(dataset) % 32 or 32), "target evaluation dropped samples")
    estimate, truth = np.concatenate(estimates), np.concatenate(truths)
    _need(tuple(sorted(set(names))) == tuple(sorted(sessions)), "target sessions omitted/added")
    per_session = {}
    for name in sessions:
        mask = np.asarray([item == name for item in names], dtype=bool)
        _need(bool(mask.any()), f"target session missing: {name}")
        per_session[name] = {"samples": int(mask.sum()), "r2": _r2(truth[mask], estimate[mask])}
    return {"pooled_r2": _r2(truth, estimate), "per_session": per_session, "samples": len(dataset), "batches": len(sizes),
            "last_batch_size": sizes[-1], "r2_accumulator_dtype": "float64", "state_immutable": True,
            "state_sha256_before": before, "state_sha256_after": after, "query_window_indices_sha256": dataset.window_indices_sha256}


def _config(path: Path, *, arm: str, date: str, fold0: Path) -> Any:
    candidate = path.resolve()
    _need(candidate.is_file() and not candidate.is_symlink(), f"{arm}: resolved config missing/symlinked")
    cfg = OmegaConf.load(candidate)
    # Hydra serializes this all-digit date override as a YAML integer in the
    # run-local resolved config.  The protocol's canonical date identifier is
    # still the eight-character string used for paths, receipts, and metadata;
    # compare the serialized config against its exact numeric representation
    # so this representation detail cannot mask actual date drift.
    expected = {"seed": 42, "train": True, "test": False, "ckpt_path": None, "width_date.arm": arm,
                "width_date.identity_width": WIDTH_ARMS[arm], "width_date.outer_date": int(date),
                "width_date.fold0_terminal_receipt": str(fold0), "width_date.noninferiority_margin_r2": MARGIN,
                "data.calibration_n_trials": 4, "data.window_size": 700, "data.batch_size": 32, "data.fixed_epochs": 50,
                "trainer.max_epochs": 50, "trainer.min_epochs": 50, "trainer.limit_val_batches": 0,
                "trainer.num_sanity_val_steps": 0, "model.net.identity_width": WIDTH_ARMS[arm],
                "model.net.model_dim": 1024, "model.net.num_id_layers": 3,
                "model.net._target_": "src.models.components.spint_identity_width.SpintIdentityWidthModel",
                "data._target_": "src.data.h1_spint_width_date_lodo.H1SpintWidthDateLodoSourceDataModule"}
    for dotted, value in expected.items():
        _need(OmegaConf.select(cfg, dotted) == value, f"{arm}: config drift at {dotted}")
    _need("carrier" not in OmegaConf.to_yaml(cfg.model, resolve=False).lower(), f"{arm}: config model mentions carrier")
    return cfg


def _checkpoint(path: Path, config_path: Path, *, arm: str, date: str, expected_source: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = path.resolve()
    _need(candidate.is_file() and not candidate.is_symlink(), f"{arm}: terminal checkpoint missing/symlinked")
    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    _need(isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping) and payload.get("epoch") == 49,
          f"{arm}: malformed/non-e49 compact checkpoint")
    meta = payload.get("h1_spint_identity_width_date_lodo")
    _need(isinstance(meta, Mapping), f"{arm}: date compact metadata absent")
    expected = {"schema": DATE_CHECKPOINT_SCHEMA, "width_arm": arm, "identity_width": WIDTH_ARMS[arm], "outer_date": date,
                "fresh_seed": 42, "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
                "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection", "carrier_input": "absent",
                "target_optimizer_steps": 0, "target_backward_steps": 0, "checkpoint_warm_start": False,
                "phase1_source_manifest_sha256": expected_source["phase1_source_manifest_sha256"],
                "phase2_source_binding_sha256": expected_source["phase2_source_binding_sha256"]}
    for key, value in expected.items():
        _need(meta.get(key) == value, f"{arm}: checkpoint metadata drift at {key}")
    _need(meta.get("config_sha256") == sha256_file(config_path), f"{arm}: checkpoint/config SHA mismatch")
    _need(meta.get("identity_parameters") == 4 * WIDTH_ARMS[arm] ** 2 + 1729 * WIDTH_ARMS[arm] + 700,
          f"{arm}: parameter arithmetic drift")
    _need(payload.get("global_step", 0) > 0, f"{arm}: checkpoint has no source optimizer steps")
    return dict(payload), dict(meta)


def evaluate(*, data_dir: str | Path, phase1_preflight: str | Path, date_preflight: str | Path, fold0_receipt: str | Path,
             date: str, compact: Mapping[str, tuple[str | Path, str | Path]], output: str | Path, device: str) -> dict[str, Any]:
    _need(date in FOLLOWUP_DATES and device in {"cpu", "cuda"} and (device != "cuda" or torch.cuda.is_available()), "date/device invalid")
    output_path = Path(output).resolve()
    _need(not output_path.exists() and not output_path.is_symlink(), "refusing repeat/overwrite target evaluation output")
    fold_path, _fold, fold_sha = _read_immutable(fold0_receipt, "h1_spint_identity_width_fold0_terminal_evaluation_v1")
    pre_path, pre, pre_sha = _read_immutable(date_preflight, PREFLIGHT_SCHEMA)
    _need(pre.get("status") == "PASS_H1_SPINT_IDENTITY_WIDTH_DATE_LODO_SOURCE_ONLY_CPU_PREFLIGHT", "date source preflight not passed")
    gate = pre.get("fold0_gate", {})
    _need(gate.get("path") == str(fold_path) and gate.get("sha256") == fold_sha, "date preflight bound another fold0 receipt")
    eligible = tuple(gate.get("eligible_compact_arms", ()))
    _need(tuple(compact) == eligible and eligible, "evaluator requires exactly every fold0-eligible compact arm in canonical gate order")
    date_binding = pre.get("date_bindings", {}).get(date)
    _need(isinstance(date_binding, Mapping), "date source binding absent from preflight")
    source = date_binding.get("source_manifest", {})
    reference = date_binding.get("reused_hs_reference", {})
    _need(isinstance(source, Mapping) and isinstance(reference, Mapping), "date source/reference binding malformed")
    # Bind all compact configs/checkpoints, reuse audit, and source schedule
    # before the first outer-date target recording operation.
    configs: dict[str, Any] = {}; payloads: dict[str, dict[str, Any]] = {}; metadata: dict[str, dict[str, Any]] = {}
    for arm in eligible:
        checkpoint, config = compact[arm]
        cfg_path = Path(config).resolve(); cfg = _config(cfg_path, arm=arm, date=date, fold0=fold_path)
        payload, meta = _checkpoint(Path(checkpoint), cfg_path, arm=arm, date=date, expected_source=source)
        configs[arm], payloads[arm], metadata[arm] = cfg, payload, meta
    runtime_source = H1SpintWidthDateLodoSourceDataModule(task="h1", data_dir=str(Path(data_dir).resolve()),
                                                           phase1_preflight_path=str(Path(phase1_preflight).resolve()), outer_date=date)
    runtime_source.setup("fit")
    _need(runtime_source.phase1_manifest_sha256 == source["phase1_source_manifest_sha256"] == reference["source_manifest_sha256"],
          "runtime compact source and reused H-S reference source manifests differ")
    _need(runtime_source.source_manifest()["phase2_source_binding_sha256"] == source["phase2_source_binding_sha256"]
          == reference["source_binding_sha256"], "runtime compact source and reused H-S M4 schedule differ")
    eval_device = torch.device(device)
    models = {}
    for arm in eligible:
        model = hydra.utils.instantiate(configs[arm].model)
        model.load_state_dict(payloads[arm]["state_dict"], strict=True)
        model.to(eval_device); model.eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        models[arm] = model
    del payloads
    # First target operation: all source/checkpoint/reference identity bindings are now complete.
    records = load_spint_width_date_target_records(str(Path(data_dir).resolve()), outer_date=date)
    target = H1SpintWidthDateLodoStrictTargetDataset(records, outer_date=date)
    target_manifest = target.manifest()
    _need(target.window_indices_sha256 == reference["target_query_window_indices_sha256"], "compact and reused H-S strict query hashes differ")
    reference_support = reference["target_support"]
    _need(isinstance(reference_support, Mapping) and set(target_manifest["support"]) == set(reference_support),
          "compact and reused H-S target support session sets differ")
    # The older paired H-S receipt also records an H-C-only normalized-carrier
    # digest.  Compare every activity/support/query field it shares with this
    # activity-only route, without importing or reconstructing that carrier.
    for session, current in target_manifest["support"].items():
        prior = reference_support[session]
        _need(isinstance(prior, Mapping) and all(current.get(key) == prior.get(key) for key in
              ("support_trials", "fifth_trial", "query_first_bin", "support_sha256", "identity_sha256")),
              f"{session}: compact and reused H-S activity support bindings differ")
    _need({name: records[name].input_sha256 for name in records} == reference["target_files"], "target file hash drift from H-S reference")
    metrics = {arm: _evaluate(models[arm], target, eval_device, tuple(records)) for arm in eligible}
    hs_r2 = float(reference["h_s_pooled_r2"])
    deltas = {arm: float(metrics[arm]["pooled_r2"] - hs_r2) for arm in eligible}
    body = {
        "schema": EVALUATION_SCHEMA, "status": f"PASS_H1_SPINT_IDENTITY_WIDTH_DATE_LODO_{date}_COMPACT_EVALUATED",
        "outer_date": date, "device": str(eval_device), "preflight": {"path": str(pre_path), "sha256": pre_sha},
        "fold0_gate": {"path": str(fold_path), "sha256": fold_sha, "eligible_compact_arms": list(eligible), "margin_r2": MARGIN},
        "checkpoint_binding_completed_before_target_open": True,
        "reused_hs_reference": reference,
        "compact_checkpoints": {arm: {"path": str(Path(compact[arm][0]).resolve()), "sha256": sha256_file(compact[arm][0]),
                                       "config_path": str(Path(compact[arm][1]).resolve()), "config_sha256": sha256_file(compact[arm][1]),
                                       "metadata": metadata[arm]} for arm in eligible},
        "source_manifest": runtime_source.source_manifest(), "source_manifest_sha256": runtime_source.source_manifest_sha256,
        "target": {"sessions": list(records), "files": {name: records[name].input_sha256 for name in records},
                   "strict_dataset": target_manifest, "carrier_input": "absent", "all_query_histories_start_at_or_after_fifth_trial": True,
                   "remainder_preserved": True},
        "metrics": {"reused_hs_pooled_r2": hs_r2, "compact": metrics, "delta_r2_vs_reused_hs": deltas},
        "interpretation_boundary": "This matched compact-versus-large H-S comparison measures identity-encoder capacity redundancy or limited extra capacity headroom only. It does not compare activity-derived identity with no identity.",
        "scope": {"opened": "sealed source records for pre-open validation, then declared public held-in-calib target records exactly once for this date",
                  "target_optimizer_steps": 0, "target_backward_steps": 0, "minival_opened": False, "formal_or_organizer_data_opened": False},
    }
    written, digest = write_immutable_json(output_path, body)
    body["receipt_path"], body["receipt_sha256"] = str(written), digest
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/000954")
    parser.add_argument("--phase1-preflight", type=Path, default=ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase1/H1_CARRIERID_DATE_LODO_SOURCE_PREFLIGHT_v1.json")
    parser.add_argument("--date-preflight", required=True, type=Path)
    parser.add_argument("--fold0-receipt", required=True, type=Path)
    parser.add_argument("--date", choices=FOLLOWUP_DATES, required=True)
    parser.add_argument("--compact", action="append", nargs=3, metavar=("ARM", "CHECKPOINT", "CONFIG"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    compact = {arm: (checkpoint, config) for arm, checkpoint, config in args.compact}
    result = evaluate(data_dir=args.data_dir, phase1_preflight=args.phase1_preflight, date_preflight=args.date_preflight,
                      fold0_receipt=args.fold0_receipt, date=args.date, compact=compact, output=args.output, device=args.device)
    print(json.dumps({"status": result["status"], "receipt_sha256": result["receipt_sha256"], "deltas": result["metrics"]["delta_r2_vs_reused_hs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
