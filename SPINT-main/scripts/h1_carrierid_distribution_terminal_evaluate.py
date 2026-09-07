#!/usr/bin/env python3
"""One-shot paired target evaluator for H1 fresh D-S4/D-Q4 retraining.

This is a leakage diagnostic, not a deployment evaluator.  Target recordings
remain unopened until this program has bound an immutable terminal-preflight
receipt, a passing source-only terminal checkpoint-pair receipt, both exact
checkpoint/config byte identities, and a reconstructed source-only normalizer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import hydra
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.h1_carrierid_distribution import H1CarrierIdDistributionDataModule
from src.data.h1_carrierid_distribution_target import H1CarrierIdDistributionStrictTargetDataset
from src.data.h1_m4_eb_pilot import H1_M4_FOLD0_TARGET, load_target_records, validate_target_receipt_binding
from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    assert_immutable_receipt,
    sha256_file,
    write_immutable_json,
)
from scripts.h1_carrierid_distribution_terminal_checker import (
    CHECKER_SCHEMA,
    CHECKER_STATUS,
    TERMINAL_PREFLIGHT_SCHEMA,
    TERMINAL_PREFLIGHT_STATUS,
    _verify_source_closure,
)
from scripts.h1_carrierid_evaluate import _evaluate


TERMINAL_SCHEMA = "h1_carrierid_h32_fresh_distribution_leakage_terminal_eval_v1"
TERMINAL_STATUS = "PASS_H1_CARRIERID_H32_FRESH_D_S4_D_Q4_LEAKAGE_DIAGNOSTIC_EVALUATED"


def _load_config(path: Path, arm: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    cfg = OmegaConf.load(path)
    if (
        str(cfg.pilot.arm) != arm
        or str(cfg.data.carrier_distribution_arm) != arm
        or str(cfg.model._target_) != "src.models.h1_carrierid_distribution_module.H1CarrierIdDistributionLitModule"
        or bool(cfg.pilot.leakage_diagnostic_only) is not True
    ):
        raise NormalizedV2ContractError(f"D-{arm.upper()} resolved config does not bind leakage diagnostic arm")
    return cfg


def _instantiate(cfg: Any, checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise NormalizedV2ContractError("D-S4/D-Q4 terminal evaluator received invalid checkpoint")
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training:
        raise RuntimeError("D-S4/D-Q4 terminal evaluator failed to enter eval mode")
    return model


def _rebuild_s4_source(cfg: Any) -> H1CarrierIdDistributionDataModule:
    data = cfg.data
    source = H1CarrierIdDistributionDataModule(
        task=str(data.task), data_dir=str(data.data_dir), raw_receipt_path=str(data.raw_receipt_path),
        eb_receipt_path=str(data.eb_receipt_path), cache_dir=str(data.cache_dir),
        carrier_distribution_arm="s4", batch_size=int(data.batch_size), window_size=int(data.window_size),
        calibration_n_trials=int(data.calibration_n_trials), max_trial_length=int(data.max_trial_length),
        num_workers=int(data.num_workers), pin_memory=False, seed=int(data.seed), fixed_epochs=int(data.fixed_epochs),
        windows_per_schedule=int(data.windows_per_schedule),
    )
    source.setup("fit")
    return source


def _require_checker_binding(
    checker: Mapping[str, Any], *, s4_checkpoint: Path, s4_config: Path, q4_checkpoint: Path, q4_config: Path
) -> None:
    if checker.get("schema") != CHECKER_SCHEMA or checker.get("status") != CHECKER_STATUS:
        raise NormalizedV2ContractError("D-S4/D-Q4 evaluator requires passing source-only terminal pair checker")
    arms = checker.get("checkpoints", {})
    expected = {
        "d_s4": (s4_checkpoint, s4_config),
        "d_q4": (q4_checkpoint, q4_config),
    }
    for arm, (checkpoint, config) in expected.items():
        recorded = arms.get(arm, {})
        if Path(str(recorded.get("path", ""))).resolve() != checkpoint.resolve():
            raise NormalizedV2ContractError(f"terminal checker {arm} checkpoint path mismatch")
        if Path(str(recorded.get("config_path", ""))).resolve() != config.resolve():
            raise NormalizedV2ContractError(f"terminal checker {arm} config path mismatch")
        if recorded.get("sha256") != sha256_file(checkpoint) or recorded.get("config_sha256") != sha256_file(config):
            raise NormalizedV2ContractError(f"terminal checker {arm} checkpoint/config byte drift")
    gate = checker.get("target_gate", {})
    if gate.get("both_fixed_epoch49_checkpoints_validated_before_target_open") is not True:
        raise NormalizedV2ContractError("terminal checker did not authorize the post-checkpoint target boundary")


def evaluate_terminal(
    *,
    s4_checkpoint_path: str | Path,
    s4_config_path: str | Path,
    q4_checkpoint_path: str | Path,
    q4_config_path: str | Path,
    terminal_checker_path: str | Path,
    terminal_preflight_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Open the two fold-0 targets exactly after all source-only bindings pass."""

    if device not in {"cpu", "cuda"}:
        raise ValueError("D-S4/D-Q4 terminal evaluator device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("D-S4/D-Q4 terminal evaluator requested unavailable CUDA")
    preflight = assert_immutable_receipt(terminal_preflight_path, TERMINAL_PREFLIGHT_STATUS)
    if preflight.get("schema") != TERMINAL_PREFLIGHT_SCHEMA or preflight.get("launch", {}).get("authorized") is not False:
        raise NormalizedV2ContractError("D-S4/D-Q4 evaluator requires immutable no-target terminal preflight")
    source_closure = _verify_source_closure(preflight.get("source_sha256", {}))
    s4_checkpoint, s4_config = Path(s4_checkpoint_path).resolve(), Path(s4_config_path).resolve()
    q4_checkpoint, q4_config = Path(q4_checkpoint_path).resolve(), Path(q4_config_path).resolve()
    checker = assert_immutable_receipt(terminal_checker_path, CHECKER_STATUS)
    _require_checker_binding(checker, s4_checkpoint=s4_checkpoint, s4_config=s4_config,
                             q4_checkpoint=q4_checkpoint, q4_config=q4_config)
    if Path(output_path).exists():
        raise FileExistsError("refusing to overwrite D-S4/D-Q4 one-shot terminal receipt")
    s4_cfg, q4_cfg = _load_config(s4_config, "s4"), _load_config(q4_config, "q4")
    # Still source-only: rebind the exact paired normalizer and frozen plan.
    source = _rebuild_s4_source(s4_cfg)
    runtime = checker.get("runtime_source", {}).get("d_s4", {})
    if (
        source.pilot_manifest_sha256 != runtime.get("source_manifest_sha256")
        or source.normalizer.manifest != runtime.get("normalizer")
        or len(source.records) != 11
    ):
        raise NormalizedV2ContractError("D-S4/D-Q4 target evaluator source reconstruction drift")
    evaluation_device = torch.device(device)
    s4_model = _instantiate(s4_cfg, s4_checkpoint, evaluation_device)
    q4_model = _instantiate(q4_cfg, q4_checkpoint, evaluation_device)

    # First and only target data access in this execution path.
    target = load_target_records(s4_cfg.data.data_dir)
    validate_target_receipt_binding(target, source.plan, s4_cfg.data.raw_receipt_path, s4_cfg.data.eb_receipt_path)
    s4_dataset = H1CarrierIdDistributionStrictTargetDataset(target, source.plan, source.normalizer, "s4")
    q4_dataset = H1CarrierIdDistributionStrictTargetDataset(target, source.plan, source.normalizer, "q4")
    if (
        s4_dataset.window_indices_sha256 != q4_dataset.window_indices_sha256
        or s4_dataset.support_and_carrier_hashes() != q4_dataset.support_and_carrier_hashes()
    ):
        raise NormalizedV2ContractError("D-S4/D-Q4 target identity/window contract differs")
    s4_metrics = _evaluate(s4_model, s4_dataset, evaluation_device, "D-S4/support")
    q4_metrics = _evaluate(q4_model, q4_dataset, evaluation_device, "D-Q4/query-local-leakage")
    for metrics in (s4_metrics, q4_metrics):
        if metrics.get("r2_accumulator_dtype") != "float64" or metrics.get("state_immutable") is not True:
            raise NormalizedV2ContractError("D-S4/D-Q4 target metric/state contract failed")
    delta = {
        "pooled_r2": float(q4_metrics["pooled_r2"] - s4_metrics["pooled_r2"]),
        "per_session_r2": {
            name: float(q4_metrics["per_session"][name]["r2"] - s4_metrics["per_session"][name]["r2"])
            for name in H1_M4_FOLD0_TARGET
        },
    }
    receipt = {
        "schema": TERMINAL_SCHEMA, "status": TERMINAL_STATUS,
        "claim_boundary": "LEAKAGE_DIAGNOSTIC_ONLY_NOT_FOR_SELECTION_OR_PAPER_MAIN_RESULT",
        "checkpoint_binding_completed_before_target_open": True,
        "terminal_preflight": {"path": str(Path(terminal_preflight_path).resolve()), "sha256": sha256_file(terminal_preflight_path)},
        "terminal_checkpoint_checker": {"path": str(Path(terminal_checker_path).resolve()), "sha256": sha256_file(terminal_checker_path)},
        "source_closure": source_closure,
        "checkpoints": {
            "d_s4": {"path": str(s4_checkpoint), "sha256": sha256_file(s4_checkpoint), "config_sha256": sha256_file(s4_config)},
            "d_q4": {"path": str(q4_checkpoint), "sha256": sha256_file(q4_checkpoint), "config_sha256": sha256_file(q4_config)},
        },
        "source": {"manifest_sha256": source.pilot_manifest_sha256, "normalizer": source.normalizer.manifest,
                   "source_recordings_opened_before_target": len(source.records)},
        "target": {
            "sessions": list(H1_M4_FOLD0_TARGET),
            "files": {name: target[name].input_sha256 for name in H1_M4_FOLD0_TARGET},
            "strict_query_window_indices_sha256": s4_dataset.window_indices_sha256,
            "support_and_carrier_hashes": s4_dataset.support_and_carrier_hashes(),
            "same_identity_and_query_windows": True,
            "d_s4_carrier": "first four support trials t..t+3",
            "d_q4_carrier": "next four query-local trials t+4..t+7; deliberate labelled leakage diagnostic",
        },
        "metrics": {"d_s4": s4_metrics, "d_q4": q4_metrics, "d_q4_minus_d_s4": delta},
        "target_updates": {"optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged": True},
        "data_scope": {"opened": "11 source then exactly 2 public held-in-calib fold-0 target NWBs",
                       "minival_opened": False, "formal_heldout_opened": False, "evalai_opened": False},
    }
    path, digest = write_immutable_json(output_path, receipt)
    return {"status": TERMINAL_STATUS, "receipt_path": str(path), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s4-checkpoint", required=True, type=Path)
    parser.add_argument("--s4-config", required=True, type=Path)
    parser.add_argument("--q4-checkpoint", required=True, type=Path)
    parser.add_argument("--q4-config", required=True, type=Path)
    parser.add_argument("--terminal-checker", required=True, type=Path)
    parser.add_argument("--terminal-preflight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    print(json.dumps(evaluate_terminal(
        s4_checkpoint_path=args.s4_checkpoint, s4_config_path=args.s4_config,
        q4_checkpoint_path=args.q4_checkpoint, q4_config_path=args.q4_config,
        terminal_checker_path=args.terminal_checker, terminal_preflight_path=args.terminal_preflight,
        output_path=args.output, device=args.device,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
