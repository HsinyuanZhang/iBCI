#!/usr/bin/env python3
"""Append-only H1-CI64-SLODO execution authority supplemental.

The original CI launch receipt remains immutable because it binds the completed
H-S/H-C five-date source/date screen.  This narrow supplemental adds the
separately sealed H-C/H-C0 causal-decomposition aggregate as an execution
prerequisite.  Neither input performs route selection, and this program never
opens recordings, checkpoints, or CUDA.
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
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h1_carrierid_date_lodo_ci_launch_receipt import (
    LAUNCH_RECEIPT_SCHEMA,
    LAUNCH_RECEIPT_STATUS,
    ROUTE,
    _read_immutable_json,
)


AUTHORITY_SCHEMA = "h1_carrierid_date_lodo_ci_execution_authority_supplement_v1"
AUTHORITY_STATUS = "PASS_H1_CARRIERID_DATE_LODO_CI64_SLODO_EXECUTION_AUTHORIZED"
HC0_SCHEMA = "h1_carrierid_hc0_fivedate_aggregate_v1"
HC0_STATUS = "PASS_HC0_FIVEDATE_CAUSAL_DECOMPOSITION"
HC0_SHA256 = "f90b41ff169262359499b14c65920569d63d5d71c04e9c672294c3d9f593aa6c"


class CiExecutionAuthorityError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise CiExecutionAuthorityError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_hc0(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    candidate = Path(path).resolve()
    _need(candidate.is_file() and not candidate.is_symlink()
          and stat.S_IMODE(candidate.stat().st_mode) == 0o444,
          f"immutable mode-0444 H-C/H-C0 aggregate required: {candidate}")
    digest = _sha(candidate)
    _need(digest == HC0_SHA256, "H-C/H-C0 aggregate SHA-256 differs from frozen f90b41ff...593aa6c")
    body = json.loads(candidate.read_text(encoding="utf-8"))
    _need(isinstance(body, dict) and body.get("schema") == HC0_SCHEMA and body.get("status") == HC0_STATUS,
          "H-C/H-C0 aggregate schema/status drift")
    _need(body.get("dates") == ["19250108", "19250113", "19250115", "19250119", "19250120"],
          "H-C/H-C0 aggregate canonical date order drift")
    return candidate, body, digest


def validate(authority_path: str | Path) -> tuple[Path, dict[str, Any], str]:
    candidate = Path(authority_path).resolve()
    _need(candidate.is_file() and not candidate.is_symlink()
          and stat.S_IMODE(candidate.stat().st_mode) == 0o444,
          f"immutable mode-0444 CI execution authority required: {candidate}")
    body = json.loads(candidate.read_text(encoding="utf-8"))
    _need(isinstance(body, dict) and body.get("schema") == AUTHORITY_SCHEMA
          and body.get("status") == AUTHORITY_STATUS, "CI execution authority schema/status drift")
    _need(body.get("route") == ROUTE and body.get("automatic_route_selection") == "FORBIDDEN",
          "CI execution authority route-selection boundary drift")
    _need(body.get("scope") == {"nwb_opened": False, "checkpoint_opened": False,
                                 "trainer_constructed_or_launched": False,
                                 "cuda_constructed_or_launched": False,
                                 "target_data_opened": False},
          "CI execution authority scope drift")
    return candidate, body, _sha(candidate)


def prepare(*, launch_receipt: Path, hc0_aggregate: Path, output: Path) -> dict[str, Any]:
    _need(not output.exists() and not output.is_symlink() and not os.path.lexists(str(output)),
          f"refusing to overwrite CI execution authority: {output}")
    launch_path, launch, launch_sha = _read_immutable_json(
        launch_receipt, schema=LAUNCH_RECEIPT_SCHEMA, status=LAUNCH_RECEIPT_STATUS,
    )
    _need(launch.get("route") == ROUTE and launch.get("explicit_operator_route") == ROUTE,
          "prepared CI launch receipt lost explicit H1-CI64-SLODO route")
    _need(launch.get("launch_authorized") is False and launch.get("not_a_gpu_launcher") is True,
          "prepared CI launch receipt semantics drift")
    hc0_path, _hc0, hc0_sha = _read_hc0(hc0_aggregate)
    payload = {
        "schema": AUTHORITY_SCHEMA,
        "status": AUTHORITY_STATUS,
        "route": ROUTE,
        "automatic_route_selection": "FORBIDDEN",
        "prepared_ci_launch_receipt": {"path": str(launch_path), "sha256": launch_sha,
                                         "schema": LAUNCH_RECEIPT_SCHEMA,
                                         "status": LAUNCH_RECEIPT_STATUS},
        "hc_hc0_hs_final_attribution_aggregate": {
            "path": str(hc0_path), "sha256": hc0_sha, "required_sha256": HC0_SHA256,
            "mode": "0444", "schema": HC0_SCHEMA, "status": HC0_STATUS,
            "role": "execution prerequisite only; does not select or automatically launch CI64",
        },
        "frozen_ci_protocol": {
            "all_arms_per_date": ["CI32-FULL", "CI64-FULL", "CI64-C0", "CI64-LS", "CI64-RS"],
            "fresh_seed": 42, "epochs": 50, "fixed_terminal_epoch_zero_based": 49,
            "target_optimizer_steps": 0, "target_backward_steps": 0,
            "ci_historical_aggregate_gate": "UNCHANGED",
            "post_ci_H64_practical_gate": "CI64-FULL minus CI32-FULL mean >= +0.03; not evaluated or selected here",
        },
        "scope": {"nwb_opened": False, "checkpoint_opened": False,
                  "trainer_constructed_or_launched": False,
                  "cuda_constructed_or_launched": False, "target_data_opened": False},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    _need(stat.S_IMODE(output.stat().st_mode) == 0o444, "CI execution authority lost immutable mode")
    return {"status": AUTHORITY_STATUS, "authority_path": str(output.resolve()),
            "authority_sha256": _sha(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-receipt", type=Path)
    parser.add_argument("--hc0-aggregate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        _need(args.launch_receipt is None and args.hc0_aggregate is None and args.output is None,
              "--verify cannot be combined with preparation arguments")
        path, _body, digest = validate(args.verify)
        print(json.dumps({"status": AUTHORITY_STATUS, "authority_path": str(path),
                          "authority_sha256": digest}, sort_keys=True))
        return
    _need(args.launch_receipt is not None and args.hc0_aggregate is not None and args.output is not None,
          "--launch-receipt, --hc0-aggregate, and --output are required")
    print(json.dumps(prepare(launch_receipt=args.launch_receipt, hc0_aggregate=args.hc0_aggregate,
                             output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
