#!/usr/bin/env python3
"""No-target source-checkpoint closure for the six H1-EST4 arms.

This checker reads only named source checkpoints/configs and immutable
receipts.  It deliberately has no evaluator, data loader, trainer, CUDA, or
subprocess path.  A passing receipt is source-training provenance, never an
opened target result or an automatic route decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import uuid
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h1_carrierid_date_lodo_five_date_aggregate import AGGREGATE_SCHEMA, AGGREGATE_STATUS, DATES, ROUTE_PREREQUISITE_STATUS
from scripts.h1_carrierid_date_lodo_est4_launch_receipt import (
    LAUNCH_RECEIPT_SCHEMA,
    LAUNCH_RECEIPT_STATUS,
    ROUTE as EST4_ROUTE,
)
from src.data.h1_carrierid_date_lodo_est4 import EST4_ARMS, EST4_PREFLIGHT_SCHEMA, EST4_PREFLIGHT_STATUS
from src.models.h1_carrierid_date_lodo_est4_module import EST4_CHECKPOINT_SCHEMA


CHECKER_SCHEMA = "h1_carrierid_date_lodo_est4_six_arm_source_terminal_checker_v1"
CHECKER_STATUS = "PASS_H1_CARRIERID_DATE_LODO_EST4_SIX_ARM_SOURCE_ONLY_CHECKPOINT_CLOSURE"
CLOSURE_FILES = (
    "src/data/h1_carrierid_date_lodo_est4_target.py",
    "src/data/h1_carrierid_date_lodo_target.py",
    "src/data/h1_carrierid_date_lodo_est4.py",
    "src/data/h1_carrierid_date_lodo_source.py",
    "src/data/h1_m4_eb_pilot.py",
    "src/models/components/h1_carrierid_est4_spint.py",
    "src/models/components/h1_carrierid_spint.py",
    "src/models/h1_carrierid_date_lodo_est4_module.py",
    "src/models/falcon_module.py",
    "src/models/components/spint.py",
    "src/h1_m4_cce_contract.py",
    "scripts/h1_carrierid_date_lodo_est4_terminal_checker.py",
    "scripts/h1_carrierid_date_lodo_est4_terminal_evaluate.py",
)


class Est4TerminalCheckError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Est4TerminalCheckError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable_json(path: Path, *, schema: str, status: str) -> tuple[Path, dict[str, Any], str]:
    candidate = path.resolve()
    _need(candidate.is_file() and stat.S_IMODE(candidate.stat().st_mode) == 0o444,
          f"immutable mode-0444 receipt required: {candidate}")
    body = json.loads(candidate.read_text(encoding="utf-8"))
    _need(isinstance(body, dict) and body.get("schema") == schema and body.get("status") == status,
          f"receipt schema/status drift: {candidate}")
    return candidate, body, _sha(candidate)


def _validate_gate(*, preflight_path: Path, aggregate_path: Path, outer_date: str) -> tuple[dict[str, Any], str, str]:
    _aggregate_path, aggregate, aggregate_sha = _immutable_json(aggregate_path, schema=AGGREGATE_SCHEMA, status=AGGREGATE_STATUS)
    route = aggregate.get("route_prerequisite")
    _need(tuple(aggregate.get("required_outer_dates", ())) == DATES
          and aggregate.get("all_five_date_receipts_present_and_validated") is True
          and isinstance(route, Mapping) and route.get("status") == ROUTE_PREREQUISITE_STATUS
          and route.get("automatic_route_selection") == "FORBIDDEN", "EST4 requires completed non-selecting 5/5 aggregate")
    _preflight_path, preflight, preflight_sha = _immutable_json(preflight_path, schema=EST4_PREFLIGHT_SCHEMA, status=EST4_PREFLIGHT_STATUS)
    _need(preflight.get("outer_date") == outer_date and preflight.get("source_controls", {}).get("all_arms") == list(EST4_ARMS),
          "EST4 source preflight date/arm drift")
    return preflight, preflight_sha, aggregate_sha


def _validate_prepared_launch(
    *, launch_receipt_path: Path, preflight_path: Path, preflight_sha: str,
    aggregate_path: Path, aggregate_sha: str, outer_date: str,
) -> tuple[Path, str]:
    path, launch, digest = _immutable_json(
        launch_receipt_path, schema=LAUNCH_RECEIPT_SCHEMA, status=LAUNCH_RECEIPT_STATUS,
    )
    contract = launch.get("training_contract")
    aggregate = launch.get("five_date_aggregate")
    row = launch.get("est4_source_preflights", {}).get(outer_date)
    _need(launch.get("route") == EST4_ROUTE and launch.get("explicit_operator_route") == EST4_ROUTE
          and launch.get("not_a_gpu_launcher") is True and launch.get("launch_authorized") is False
          and launch.get("proposed_arms_per_date") == list(EST4_ARMS)
          and isinstance(contract, Mapping) and contract.get("fresh_seed") == 42
          and contract.get("epochs") == 50 and contract.get("fixed_terminal_epoch_zero_based") == 49
          and contract.get("warm_start_forbidden") is True,
          "EST4 terminal checker requires the exact prepared six-arm receipt")
    _need(isinstance(aggregate, Mapping)
          and Path(str(aggregate.get("path", ""))).resolve() == aggregate_path.resolve()
          and aggregate.get("sha256") == aggregate_sha,
          "EST4 prepared receipt binds another five-date aggregate")
    _need(isinstance(row, Mapping)
          and Path(str(row.get("path", ""))).resolve() == preflight_path.resolve()
          and row.get("sha256") == preflight_sha,
          "EST4 prepared receipt binds another date preflight")
    return path, digest


def _checkpoint(path: Path, *, arm: str, outer_date: str, preflight: Mapping[str, Any],
                preflight_sha: str, aggregate_sha: str) -> dict[str, Any]:
    candidate = path.resolve()
    _need(candidate.is_file(), f"EST4 checkpoint missing: {candidate}")
    checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
    _need(isinstance(checkpoint, dict) and int(checkpoint.get("epoch", -1)) == 49 and int(checkpoint.get("global_step", 0)) > 0,
          f"EST4 {arm} is not fixed terminal e49")
    metadata = checkpoint.get("h1_carrierid_date_lodo_est4")
    _need(isinstance(metadata, Mapping), f"EST4 {arm} checkpoint metadata missing")
    required = {
        "schema": EST4_CHECKPOINT_SCHEMA, "arm": arm, "outer_date": outer_date, "fresh_seed": 42,
        "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
        "target_optimizer_steps": 0, "target_backward_steps": 0, "checkpoint_warm_start": False,
        "target_evaluator_status": "IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED",
        "est4_preflight_sha256": preflight_sha, "five_date_aggregate_sha256": aggregate_sha,
    }
    for key, value in required.items():
        _need(metadata.get(key) == value, f"EST4 {arm} checkpoint metadata drift: {key}")
    expected_mode = "baseline" if arm.startswith("B-") else "learned"
    _need(metadata.get("estimator_mode") == expected_mode, f"EST4 {arm} estimator mode drift")
    expected_added = 0 if expected_mode == "baseline" else 2_850
    _need(metadata.get("estimator_added_learned_parameters") == expected_added
          and metadata.get("expected_estimator_added_learned_parameters") == expected_added,
          f"EST4 {arm} parameter accounting drift")
    for key in ("initial_state_sha256", "component_initial_state_sha256", "shared_backbone_initial_state_sha256",
                "est4_source_binding_sha256", "phase2_base_source_binding_sha256", "phase1_source_manifest_sha256",
                "phase1_preflight_sha256", "config_sha256", "frozen_plan_sha256"):
        value = metadata.get(key)
        _need(isinstance(value, str) and len(value) == 64, f"EST4 {arm} lacks SHA: {key}")
    config = candidate.parent.parent.parent / ".hydra/config.yaml"
    _need(config.is_file() and _sha(config) == metadata.get("config_sha256"), f"EST4 {arm} saved config SHA mismatch")
    text = config.read_text(encoding="utf-8")
    for fragment in (
        "train: true", "test: false", "ckpt_path: null", "seed: 42",
        "H1CarrierIdDateLodoEst4DataModule", "H1CarrierIdDateLodoEst4LitModule", f"arm: {arm}",
        "max_epochs: 50", "min_epochs: 50",
    ):
        _need(fragment in text, f"EST4 {arm} saved config lacks {fragment!r}")
    frozen = Path(str(metadata.get("frozen_plan_path", ""))).resolve()
    _need(frozen.is_file() and _sha(frozen) == metadata.get("frozen_plan_sha256"), f"EST4 {arm} frozen source plan drift")
    return {"path": str(candidate), "sha256": _sha(candidate), "metadata": dict(metadata)}


def check_six(*, checkpoints: Mapping[str, Path], est4_preflight_path: Path, five_date_aggregate_path: Path,
              launch_receipt_path: Path, output_path: Path) -> dict[str, Any]:
    _need(tuple(checkpoints) == EST4_ARMS, "EST4 checker requires each declared six-arm checkpoint exactly once")
    _need(not output_path.exists() and not os.path.lexists(str(output_path)), "refusing to overwrite EST4 terminal receipt")
    preflight, preflight_sha, aggregate_sha = _validate_gate(
        preflight_path=est4_preflight_path, aggregate_path=five_date_aggregate_path,
        outer_date=str(_immutable_json(est4_preflight_path, schema=EST4_PREFLIGHT_SCHEMA, status=EST4_PREFLIGHT_STATUS)[1]["outer_date"]),
    )
    outer_date = str(preflight["outer_date"])
    launch_path, launch_sha = _validate_prepared_launch(
        launch_receipt_path=launch_receipt_path, preflight_path=est4_preflight_path,
        preflight_sha=preflight_sha, aggregate_path=five_date_aggregate_path,
        aggregate_sha=aggregate_sha, outer_date=outer_date,
    )
    rows = {arm: _checkpoint(path, arm=arm, outer_date=outer_date, preflight=preflight,
                             preflight_sha=preflight_sha, aggregate_sha=aggregate_sha)
            for arm, path in checkpoints.items()}
    metas = {arm: rows[arm]["metadata"] for arm in EST4_ARMS}
    _need(len({metas[arm]["initial_state_sha256"] for arm in ("B-C", "B-LS")}) == 1,
          "B-C/B-LS initial state mismatch")
    _need(len({metas[arm]["initial_state_sha256"] for arm in ("L-C", "L-C0", "L-LS", "L-RS")}) == 1,
          "learned EST4 control initial state mismatch")
    shared_fields = ("shared_backbone_initial_state_sha256", "phase2_base_source_binding_sha256",
                     "phase1_source_manifest_sha256", "phase1_preflight_sha256", "frozen_plan_sha256")
    for field in shared_fields:
        _need(len({metas[arm][field] for arm in EST4_ARMS}) == 1, f"EST4 six-arm matched H-S/H-C provenance mismatch: {field}")
    payload = {
        "schema": CHECKER_SCHEMA, "status": CHECKER_STATUS, "outer_date": outer_date,
        "checkpoints": rows,
        "est4_preflight": {"path": str(est4_preflight_path.resolve()), "sha256": preflight_sha},
        "prepared_launch_receipt": {"path": str(launch_path), "sha256": launch_sha,
                                    "consumed_by_terminal_checker": True},
        "five_date_aggregate": {"path": str(five_date_aggregate_path.resolve()), "sha256": aggregate_sha,
                                "source_date_screen_complete": True,
                                "automatic_route_selection": "FORBIDDEN"},
        "source_binding_sha256": preflight.get("source_binding_sha256"),
        "receipt_binding": {"est4_preflight_sha256": preflight_sha, "five_date_aggregate_sha256": aggregate_sha},
        "initialization_checks": {"b_c_b_ls_initial_state_equal": True, "learned_controls_initial_state_equal": True,
                                  "all_arms_shared_hs_hc_backbone_and_source_equal": True},
        "code_sha256": {relative: _sha(ROOT / relative) for relative in CLOSURE_FILES},
        "scope": {"nwb_opened": False, "target_recordings_opened": 0, "target_bytes_read": 0,
                  "trainer_constructed_or_launched": False, "gpu_constructed_or_launched": False,
                  "target_optimizer_steps": 0, "target_backward_steps": 0,
                  "target_evaluator": "IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED"},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temp = output_path.parent / f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temp, 0o444); os.link(temp, output_path)
    finally:
        if temp.exists(): temp.unlink()
    _need(stat.S_IMODE(output_path.stat().st_mode) == 0o444, "EST4 terminal receipt lost immutable mode")
    return {"status": CHECKER_STATUS, "receipt_path": str(output_path.resolve()), "receipt_sha256": _sha(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", metavar="ARM=PATH", required=True)
    parser.add_argument("--est4-preflight", type=Path, required=True)
    parser.add_argument("--five-date-aggregate", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: dict[str, Path] = {}
    for item in args.checkpoint:
        arm, separator, path = item.partition("=")
        if not separator or arm in rows:
            raise SystemExit("each --checkpoint must be one unique ARM=PATH")
        rows[arm] = Path(path)
    print(json.dumps(check_six(checkpoints=rows, est4_preflight_path=args.est4_preflight,
                               five_date_aggregate_path=args.five_date_aggregate,
                               launch_receipt_path=args.launch_receipt,
                               output_path=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
