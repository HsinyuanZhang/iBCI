#!/usr/bin/env python3
"""No-target terminal checker for one five-arm H1 CI source-date LODO run.

The checker opens only the immutable aggregate/preflight receipts, the five
source-trained terminal checkpoints, and their resolved Hydra configs.  It
does not import target data, construct a Trainer, initialise CUDA, or create a
model.  Its immutable output is the sole admissible input to the later strict
outer-date evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256, sha256_file, write_immutable_json
from scripts.h1_carrierid_date_lodo_ci_launch_receipt import (
    LAUNCH_RECEIPT_SCHEMA,
    LAUNCH_RECEIPT_STATUS,
    ROUTE as CI_ROUTE,
)


CI_ARMS = ("CI32-FULL", "CI64-FULL", "CI64-C0", "CI64-LS", "CI64-RS")
CI_PREFLIGHT_SCHEMA = "h1_carrierid_date_lodo_ci_cpu_preflight_v1"
CI_PREFLIGHT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_CI_SOURCE_ONLY_NOT_LAUNCHED"
AGGREGATE_SCHEMA = "h1_carrierid_date_lodo_five_date_heldout_aggregate_v1"
AGGREGATE_STATUS = "PASS_H1_CARRIERID_DATE_LODO_FIVE_DATE_SOURCE_DATE_SCREEN_COMPLETE_NO_ROUTE_SELECTED"
CHECKPOINT_SCHEMA = "h1_carrierid_date_lodo_ci_terminal_checkpoint_v1"
CHECKER_SCHEMA = "h1_carrierid_date_lodo_ci_five_arm_terminal_check_v1"
CHECKER_STATUS = "PASS_H1_CARRIERID_DATE_LODO_CI_FIVE_ARM_SOURCE_E49_CHECKPOINTS_NO_TARGET"
CLOSURE_FILES = (
    "src/data/h1_carrierid_date_lodo_ci_target.py",
    "src/data/h1_carrierid_date_lodo_target.py",
    "src/data/h1_carrierid_date_lodo_ci.py",
    "src/data/h1_carrierid_date_lodo_source.py",
    "src/data/h1_m4_eb_pilot.py",
    "src/models/components/h1_carrierid_ci_spint.py",
    "src/models/h1_carrierid_date_lodo_ci_module.py",
    "src/models/falcon_module.py",
    "src/models/components/spint.py",
    "src/h1_m4_cce_contract.py",
    "scripts/h1_carrierid_date_lodo_ci_terminal_checker.py",
    "scripts/h1_carrierid_date_lodo_ci_terminal_evaluate.py",
)


class CiTerminalCheckError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise CiTerminalCheckError(message)


def _sha(value: Any, label: str) -> str:
    _need(isinstance(value, str) and len(value) == 64 and all(item in "0123456789abcdef" for item in value),
          f"{label} must be a lowercase SHA-256")
    return value


def _immutable_json(path: str | Path, *, schema: str, status: str) -> tuple[Path, dict[str, Any], str]:
    candidate = Path(path).resolve()
    _need(candidate.is_file() and stat.S_IMODE(candidate.stat().st_mode) == 0o444,
          f"immutable mode-0444 receipt required: {candidate}")
    body = json.loads(candidate.read_text(encoding="utf-8"))
    _need(isinstance(body, dict) and body.get("schema") == schema and body.get("status") == status,
          f"receipt schema/status drift: {candidate}")
    return candidate, body, sha256_file(candidate)


def _validate_aggregate(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    aggregate_path, aggregate, digest = _immutable_json(path, schema=AGGREGATE_SCHEMA, status=AGGREGATE_STATUS)
    _need(tuple(aggregate.get("required_outer_dates", ())) == CONFIRMATORY_DATES
          and aggregate.get("all_five_date_receipts_present_and_validated") is True,
          "CI terminal checker requires completed canonical five-date aggregate")
    route = aggregate.get("route_prerequisite")
    _need(isinstance(route, Mapping) and route.get("status") == "source/date screen complete"
          and route.get("automatic_route_selection") == "FORBIDDEN",
          "CI terminal checker aggregate must be complete but cannot auto-select route")
    return aggregate_path, aggregate, digest


def _config_for_checkpoint(checkpoint: Path) -> Path:
    config = checkpoint.parent.parent.parent / ".hydra" / "config.yaml"
    _need(config.is_file(), f"resolved Hydra config missing next to checkpoint: {config}")
    return config


def _finite_state(state: Mapping[str, Any]) -> None:
    _need(bool(state), "checkpoint state_dict is empty")
    for name, tensor in state.items():
        _need(isinstance(tensor, torch.Tensor), f"checkpoint state is not tensor: {name}")
        if torch.is_floating_point(tensor) or torch.is_complex(tensor):
            _need(bool(torch.isfinite(tensor).all().item()), f"checkpoint state is nonfinite: {name}")


def _arm_spec(arm: str) -> tuple[int, str, bool]:
    _need(arm in CI_ARMS, f"unsupported CI arm: {arm}")
    return (32 if arm == "CI32-FULL" else 64, arm.split("-", 1)[1].lower(), arm == "CI64-C0")


def _check_config(
    config_path: Path, *, arm: str, outer_date: str, ci_preflight: Path, aggregate: Path,
) -> str:
    width, intervention, zero_carrier = _arm_spec(arm)
    cfg = OmegaConf.load(config_path)
    arm_token = arm.lower().replace("-", "_")
    fixed = (
        cfg.get("protocol_id") == f"h1_carrierid_date_lodo_{arm_token}_{outer_date}_source_only_v1",
        cfg.get("train") is True, cfg.get("test") is False, cfg.get("ckpt_path") is None, int(cfg.get("seed")) == 42,
        str(cfg.phase_ci.outer_date) == outer_date, str(cfg.phase_ci.arm) == arm,
        str(cfg.phase_ci.carrier_intervention) == intervention,
        Path(str(cfg.phase_ci.ci_preflight_path)).resolve() == ci_preflight,
        Path(str(cfg.phase_ci.five_date_aggregate_path)).resolve() == aggregate,
        str(cfg.data._target_) == "src.data.h1_carrierid_date_lodo_ci.H1CarrierIdDateLodoCiDataModule",
        str(cfg.data.ci_arm) == arm and str(cfg.data.carrier_intervention) == intervention,
        Path(str(cfg.data.ci_preflight_path)).resolve() == ci_preflight,
        Path(str(cfg.data.five_date_aggregate_path)).resolve() == aggregate,
        str(cfg.model._target_) == "src.models.h1_carrierid_date_lodo_ci_module.H1CarrierIdDateLodoCiLitModule",
        str(cfg.model.arm) == arm and str(cfg.model.outer_date) == outer_date and int(cfg.model.fixed_seed) == 42,
        Path(str(cfg.model.ci_preflight_path)).resolve() == ci_preflight,
        Path(str(cfg.model.five_date_aggregate_path)).resolve() == aggregate,
        str(cfg.model.net._target_) == "src.models.components.h1_carrierid_ci_spint.H1CarrierIdCiSpint",
        int(cfg.model.net.carrier_hidden_dim) == 32 and int(cfg.model.net.carrier_interface_dim) == width,
        bool(cfg.model.net.zero_carrier) is zero_carrier,
        int(cfg.trainer.max_epochs) == int(cfg.trainer.min_epochs) == 50,
        str(cfg.trainer.accelerator) == "gpu" and str(cfg.trainer.devices) == "1"
        and str(cfg.trainer.precision) == "32-true",
        int(cfg.trainer.limit_val_batches) == 0 and int(cfg.trainer.num_sanity_val_steps) == 0,
    )
    _need(all(fixed), f"resolved CI config violates fixed {arm} source-only contract")
    terminal = cfg.callbacks.get("fixed_epoch50")
    _need(terminal is not None and terminal.get("monitor") is None and int(terminal.get("every_n_epochs", -1)) == 50
          and int(terminal.get("save_top_k", 0)) == -1 and terminal.get("save_last") is False,
          "CI resolved config lost the fixed e49 callback")
    return sha256_file(config_path)


def _check_checkpoint(
    *, checkpoint_path: Path, arm: str, outer_date: str, ci_preflight: Mapping[str, Any],
    ci_preflight_path: Path, ci_preflight_sha: str, aggregate_path: Path, aggregate_sha: str,
) -> dict[str, Any]:
    _need(checkpoint_path.is_file(), f"CI checkpoint missing: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _need(isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping),
          "CI checkpoint is not a Lightning state_dict")
    _finite_state(payload["state_dict"])
    _need(int(payload.get("epoch", -1)) == 49 and isinstance(payload.get("global_step"), int)
          and int(payload["global_step"]) > 0, "CI checkpoint is not terminal fixed e49")
    config_path = _config_for_checkpoint(checkpoint_path)
    config_sha = _check_config(config_path, arm=arm, outer_date=outer_date,
                               ci_preflight=ci_preflight_path, aggregate=aggregate_path)
    meta = payload.get("h1_carrierid_date_lodo_ci")
    _need(isinstance(meta, Mapping), "CI checkpoint lacks terminal metadata")
    fresh = ci_preflight.get("fresh_models", {}).get(arm)
    _need(isinstance(fresh, Mapping), f"CI preflight lacks fresh {arm} model")
    source = ci_preflight.get("source_binding")
    _need(isinstance(source, Mapping), "CI preflight lacks source binding")
    required = {
        "schema": CHECKPOINT_SCHEMA, "arm": arm, "outer_date": outer_date, "fresh_seed": 42,
        "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
        "component_initial_state_sha256": fresh.get("initial_state_sha256"),
        "shared_backbone_initial_state_sha256": fresh.get("shared_backbone_initial_state_sha256"),
        "phase2_base_source_binding_sha256": canonical_sha256(source),
        "phase1_source_manifest_sha256": source.get("source_manifest_sha256"),
        "phase1_preflight_sha256": source.get("preflight_sha256"),
        "ci_preflight_sha256": ci_preflight_sha, "five_date_aggregate_sha256": aggregate_sha,
        "config_sha256": config_sha, "target_optimizer_steps": 0, "target_backward_steps": 0,
        "checkpoint_warm_start": False,
    }
    for key, value in required.items():
        _need(meta.get(key) == value, f"{arm} CI terminal metadata drift at {key}")
    for key in ("initial_state_sha256", "component_initial_state_sha256", "shared_backbone_initial_state_sha256",
                "ci_source_binding_sha256", "phase2_base_source_binding_sha256", "phase1_source_manifest_sha256",
                "phase1_preflight_sha256", "ci_preflight_sha256", "five_date_aggregate_sha256", "config_sha256"):
        _sha(meta.get(key), f"{arm} metadata.{key}")
    return {
        "arm": arm, "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_path": str(config_path), "config_sha256": config_sha, "metadata": dict(meta),
        "checkpoint_epoch_zero_based": 49, "global_step": int(payload["global_step"]),
        "state_dict_tensor_count": len(payload["state_dict"]), "state_dict_finite": True,
    }


def _validate_prepared_launch(
    *, launch_receipt_path: str | Path, ci_preflight_path: Path, ci_preflight_sha: str,
    aggregate_path: Path, aggregate_sha: str, outer_date: str,
) -> tuple[Path, str]:
    path, launch, digest = _immutable_json(
        launch_receipt_path, schema=LAUNCH_RECEIPT_SCHEMA, status=LAUNCH_RECEIPT_STATUS,
    )
    contract = launch.get("training_contract")
    aggregate = launch.get("five_date_aggregate")
    row = launch.get("ci_source_preflights", {}).get(outer_date)
    _need(launch.get("route") == CI_ROUTE and launch.get("explicit_operator_route") == CI_ROUTE
          and launch.get("not_a_gpu_launcher") is True and launch.get("launch_authorized") is False
          and launch.get("proposed_arms_per_date") == list(CI_ARMS)
          and isinstance(contract, Mapping) and contract.get("fresh_seed") == 42
          and contract.get("epochs") == 50 and contract.get("fixed_terminal_epoch_zero_based") == 49
          and contract.get("warm_start_forbidden") is True and contract.get("H64") == "PROHIBITED",
          "CI terminal checker requires the exact prepared five-arm receipt")
    _need(isinstance(aggregate, Mapping)
          and Path(str(aggregate.get("path", ""))).resolve() == aggregate_path.resolve()
          and aggregate.get("sha256") == aggregate_sha,
          "CI prepared receipt binds another five-date aggregate")
    _need(isinstance(row, Mapping)
          and Path(str(row.get("path", ""))).resolve() == ci_preflight_path.resolve()
          and row.get("sha256") == ci_preflight_sha,
          "CI prepared receipt binds another date preflight")
    return path, digest


def check_five(
    *, checkpoints: Mapping[str, str | Path], ci_preflight_path: str | Path,
    five_date_aggregate_path: str | Path, launch_receipt_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate five fresh source-only e49 checkpoints before target opening."""

    _need(tuple(checkpoints) == CI_ARMS, "CI checker requires exactly ordered CI32-full and four CI64 control arms")
    aggregate_path, _aggregate, aggregate_sha = _validate_aggregate(five_date_aggregate_path)
    preflight_path, preflight, preflight_sha = _immutable_json(
        ci_preflight_path, schema=CI_PREFLIGHT_SCHEMA, status=CI_PREFLIGHT_STATUS,
    )
    outer_date = str(preflight.get("outer_date", ""))
    _need(outer_date in CONFIRMATORY_DATES, "CI preflight outer date is not canonical")
    source = preflight.get("source_binding")
    _need(isinstance(source, Mapping) and source.get("outer_date") == outer_date
          and source.get("target_recordings_opened") == 0 and source.get("target_bytes_read") == 0,
          "CI preflight source boundary drift")
    launch_path, launch_sha = _validate_prepared_launch(
        launch_receipt_path=launch_receipt_path, ci_preflight_path=preflight_path,
        ci_preflight_sha=preflight_sha, aggregate_path=aggregate_path,
        aggregate_sha=aggregate_sha, outer_date=outer_date,
    )
    rows = {
        arm: _check_checkpoint(checkpoint_path=Path(checkpoints[arm]).resolve(), arm=arm, outer_date=outer_date,
                               ci_preflight=preflight, ci_preflight_path=preflight_path,
                               ci_preflight_sha=preflight_sha, aggregate_path=aggregate_path, aggregate_sha=aggregate_sha)
        for arm in CI_ARMS
    }
    ci64_initial = {rows[arm]["metadata"]["initial_state_sha256"] for arm in CI_ARMS if arm.startswith("CI64-")}
    shared_backbone = {rows[arm]["metadata"]["shared_backbone_initial_state_sha256"] for arm in CI_ARMS}
    common = ("phase2_base_source_binding_sha256", "phase1_source_manifest_sha256", "phase1_preflight_sha256",
              "ci_preflight_sha256", "five_date_aggregate_sha256")
    _need(len(ci64_initial) == 1, "CI64 Full/C0/LS/RS did not share exact fresh initial state")
    _need(len(shared_backbone) == 1, "CI32/CI64 did not share initial decoder-plus-prepool backbone")
    _need(all(len({rows[arm]["metadata"][field] for arm in CI_ARMS}) == 1 for field in common),
          "CI five arms do not share source/receipt provenance")
    body = {
        "schema": CHECKER_SCHEMA, "status": CHECKER_STATUS, "outer_date": outer_date,
        "ci_preflight": {"path": str(preflight_path), "sha256": preflight_sha},
        "prepared_launch_receipt": {"path": str(launch_path), "sha256": launch_sha,
                                    "consumed_by_terminal_checker": True},
        "five_date_aggregate": {"path": str(aggregate_path), "sha256": aggregate_sha,
                                "source_date_screen_complete": True, "automatic_route_selection": "FORBIDDEN"},
        "source_binding_sha256": canonical_sha256(source), "checkpoints": rows,
        "initialization_checks": {"ci64_full_c0_ls_rs_initial_state_equal": True,
                                  "ci32_ci64_shared_backbone_initial_state_equal": True},
        "equal_provenance_verified_fields": list(common),
        "code_sha256": {relative: sha256_file(ROOT / relative) for relative in CLOSURE_FILES},
        "scope": {"target_recordings_opened": 0, "target_bytes_read": 0,
                  "cuda_constructed_or_launched": False, "trainer_constructed_or_launched": False},
        "target_gate": "CLOSED: only a separate explicit strict evaluator may open the held source-date after this receipt.",
    }
    written, digest = write_immutable_json(output_path, body)
    return {"status": CHECKER_STATUS, "receipt_path": str(written), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci-preflight", required=True, type=Path)
    parser.add_argument("--five-date-aggregate", required=True, type=Path)
    parser.add_argument("--launch-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    for arm in CI_ARMS:
        parser.add_argument(f"--{arm.lower()}", required=True, type=Path)
    args = parser.parse_args()
    checkpoints = {arm: getattr(args, arm.lower().replace("-", "_")) for arm in CI_ARMS}
    print(json.dumps(check_five(checkpoints=checkpoints, ci_preflight_path=args.ci_preflight,
                                five_date_aggregate_path=args.five_date_aggregate,
                                launch_receipt_path=args.launch_receipt, output_path=args.output),
                     sort_keys=True))


if __name__ == "__main__":
    main()
