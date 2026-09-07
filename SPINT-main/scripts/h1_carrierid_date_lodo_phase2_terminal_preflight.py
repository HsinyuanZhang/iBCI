#!/usr/bin/env python3
"""Prepare a strict H1 date-LODO evaluator without opening target recordings.

The preflight is deliberately source/receipt/config/checkpoint-only.  It may
be run as soon as both H-S and H-C have passed the paired e49 checker.  A
separate evaluator has to consume this immutable result before it can call the
isolated outer-date target loader.
"""
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

from scripts.h1_carrierid_date_lodo_phase2_terminal_checker import PAIR_SCHEMA, PAIR_STATUS
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256, sha256_file, write_immutable_json


PREFLIGHT_SCHEMA = "h1_carrierid_date_lodo_phase2_terminal_evaluator_preflight_v1"
PREFLIGHT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_PHASE2_EVALUATOR_PREOPEN_CLOSURE_NO_TARGET"
CLOSURE_FILES = (
    # Target view and every project-local module whose drift can change a
    # future terminal prediction.  This is intentionally wider than the
    # thin terminal wrapper: a checkpoint alone does not freeze its runtime.
    "src/data/h1_carrierid_date_lodo_target.py",
    "src/data/h1_carrierid_date_lodo_source.py",
    "src/data/h1_m4_eb_pilot.py",
    "src/h1_m4_cce_contract.py",
    "src/models/h1_carrierid_date_lodo_phase2_module.py",
    "src/models/falcon_module.py",
    "src/models/components/spint.py",
    "src/models/components/h1_carrierid_spint.py",
    # The pair terminal checker now accepts a date-bound H-S runtime-init
    # receipt.  Freeze both its GPU source-only producer and the future-date
    # queue that orders it before H-S terminal checking, so a later pre-open
    # receipt captures the complete source-to-terminal control path.
    "scripts/h1_carrierid_date_lodo_phase2_preflight.py",
    "scripts/h1_carrierid_date_lodo_future_remote_queue.py",
    "scripts/h1_carrierid_date_lodo_phase2_terminal_checker.py",
    "scripts/h1_carrierid_date_lodo_phase2_terminal_preflight.py",
    "scripts/h1_carrierid_date_lodo_phase2_terminal_evaluate.py",
    "tests/test_h1_carrierid_date_lodo_phase2_terminal_checker_contract.py",
    "tests/test_h1_carrierid_date_lodo_phase2_contract.py",
    "tests/test_h1_carrierid_date_lodo_future_remote_queue.py",
    "tests/test_h1_carrierid_date_lodo_phase2_terminal_evaluator_contract.py",
)


class DateLodoEvaluatorPreflightError(ValueError):
    """A required pre-open terminal binding did not hold."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise DateLodoEvaluatorPreflightError(message)


def _is_within(path: Path, parent: Path) -> bool:
    """Return whether resolved ``path`` is contained by resolved ``parent``."""

    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _canonical_target_data_root(path: str | Path) -> tuple[Path, Path]:
    """Bind the one public target-data directory without reading an NWB.

    Phase-2 source training runs in an isolated checkout on the remote host,
    whereas the public H1 files and immutable Phase-1 bundle live under the
    canonical checkout.  The evaluator must therefore not infer the target
    data path from its code/checkpoint checkout.
    """

    target_data_root = Path(path).resolve()
    canonical_repository_root = target_data_root.parent.parent
    _need(target_data_root.is_dir()
          and target_data_root == canonical_repository_root / "data" / "000954"
          and canonical_repository_root.name == "SPINT-main",
          "target data root must be the canonical SPINT-main/data/000954 directory")
    return target_data_root, canonical_repository_root


def _read_immutable(path: str | Path, *, schema: str, status: str) -> tuple[Path, dict[str, Any], str]:
    resolved = Path(path).resolve()
    _need(resolved.is_file() and stat.S_IMODE(resolved.stat().st_mode) == 0o444,
          f"receipt must be immutable mode 0444: {resolved}")
    body = json.loads(resolved.read_text(encoding="utf-8"))
    _need(isinstance(body, dict) and body.get("schema") == schema and body.get("status") == status,
          f"receipt schema/status drift: {resolved}")
    return resolved, body, sha256_file(resolved)


def _require_terminal_pair(checker: Mapping[str, Any], pair_preflight: Path, pair_sha: str) -> dict[str, Any]:
    _need(checker.get("pair_preflight") == {"path": str(pair_preflight), "sha256": pair_sha},
          "terminal pair checker is bound to another pair preflight")
    arms = {"H-S": checker.get("h_s"), "H-C": checker.get("h_c")}
    for arm, row in arms.items():
        _need(isinstance(row, Mapping), f"terminal pair checker lacks {arm} result")
        checkpoint, config = Path(str(row.get("checkpoint_path", ""))).resolve(), Path(str(row.get("config_path", ""))).resolve()
        _need(checkpoint.is_file() and config.is_file(), f"{arm} checkpoint/config missing since terminal check")
        _need(row.get("checkpoint_sha256") == sha256_file(checkpoint) and row.get("config_sha256") == sha256_file(config),
              f"{arm} checkpoint/config byte drift after terminal check")
        meta = row.get("metadata")
        _need(isinstance(meta, Mapping) and meta.get("arm") == arm and meta.get("checkpoint_epoch_zero_based") == 49
              and meta.get("target_optimizer_steps") == 0 and meta.get("target_backward_steps") == 0,
              f"{arm} terminal metadata cannot authorize target boundary")
    fields = tuple(checker.get("equal_schedule_verified_fields", ()))
    _need(fields == ("phase2_source_binding_sha256", "phase1_source_manifest_sha256", "phase1_preflight_sha256"),
          "terminal pair checker did not verify the complete common source binding")
    return {arm: dict(row) for arm, row in arms.items()}


def run(*, pair_terminal_checker: str | Path, pair_preflight: str | Path,
        target_data_root: str | Path, output: str | Path) -> dict[str, Any]:
    if Path(output).exists():
        raise FileExistsError(f"refusing to overwrite evaluator preflight: {output}")
    checker_path, checker, checker_sha = _read_immutable(pair_terminal_checker, schema=PAIR_SCHEMA, status=PAIR_STATUS)
    pair_path = Path(pair_preflight).resolve()
    _need(_is_within(checker_path, ROOT) and _is_within(pair_path, ROOT)
          and _is_within(Path(output).resolve(), ROOT),
          "terminal checker, pair preflight, and evaluator preflight output must stay in the isolated-stage root")
    _need(pair_path.is_file() and stat.S_IMODE(pair_path.stat().st_mode) == 0o444, "pair preflight must be immutable 0444")
    pair = json.loads(pair_path.read_text(encoding="utf-8")); pair_sha = sha256_file(pair_path)
    _need(pair.get("schema") == "h1_carrierid_date_lodo_phase2_pair_cpu_preflight_v1"
          and pair.get("status") == "PASS_H1_CARRIERID_DATE_LODO_PHASE2_PAIR_SOURCE_ONLY_NOT_LAUNCHED",
          "pair preflight is not passed source-only evidence")
    arms = _require_terminal_pair(checker, pair_path, pair_sha)
    for arm, row in arms.items():
        _need(_is_within(Path(str(row["checkpoint_path"])), ROOT)
              and _is_within(Path(str(row["config_path"])), ROOT),
              f"{arm} terminal checkpoint/config is outside the isolated-stage root")
    source = pair.get("source_binding")
    outer_date = str(pair.get("outer_date", ""))
    _need(outer_date in CONFIRMATORY_DATES, "pair preflight outer_date is not confirmatory")
    _need(isinstance(source, Mapping) and checker.get("source_binding_sha256") == pair.get("source_binding_sha256")
          and checker.get("outer_date") == outer_date and source.get("outer_date") == outer_date
          and source.get("target_recordings_opened") == 0
          and source.get("target_bytes_read") == 0
          and pair.get("source_binding_sha256") == canonical_sha256(source),
          "pair/source binding cannot authorize outer-date evaluation")
    source_manifest = Path(str(source.get("source_manifest_path", ""))).resolve()
    _need(source_manifest.is_file() and stat.S_IMODE(source_manifest.stat().st_mode) == 0o444
          and sha256_file(source_manifest) == source.get("source_manifest_sha256"),
          "declared immutable source manifest drift")
    target_data_root, canonical_repository_root = _canonical_target_data_root(target_data_root)
    _need(_is_within(source_manifest, canonical_repository_root / "pilot_artifacts"),
          "source manifest is not under the canonical repository bound to target data")
    code_sha = {relative: sha256_file(ROOT / relative) for relative in CLOSURE_FILES}
    receipt = {
        "schema": PREFLIGHT_SCHEMA, "status": PREFLIGHT_STATUS,
        "mode": "preopen_static_closure_only_no_data_loader_no_cuda_no_trainer_no_target",
        "outer_date": outer_date, "pair_terminal_checker": {"path": str(checker_path), "sha256": checker_sha},
        "pair_preflight": {"path": str(pair_path), "sha256": pair_sha},
        "source_binding": {"sha256": pair["source_binding_sha256"], "source_manifest_path": str(source_manifest),
                           "source_manifest_sha256": source["source_manifest_sha256"]},
        "runtime": {"isolated_stage_root": str(ROOT.resolve()),
                    "canonical_data_repository_root": str(canonical_repository_root),
                    "target_data_root": str(target_data_root)},
        "checkpoints": arms, "code_sha256": code_sha,
        "scope": {"target_recordings_opened": 0, "target_bytes_read": 0, "cuda_constructed_or_launched": False,
                  "trainer_constructed_or_launched": False, "checkpoint_created_or_loaded": False,
                  "target_data_directory_metadata_checked_only": True},
        "execution_gate": "A later explicit --execute-target-evaluation call may use this preflight once; no target evaluator was run here.",
    }
    path, digest = write_immutable_json(output, receipt)
    return {"status": PREFLIGHT_STATUS, "receipt_path": str(path), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-terminal-checker", type=Path, required=True)
    parser.add_argument("--pair-preflight", type=Path, required=True)
    parser.add_argument("--target-data-root", type=Path, required=True,
                        help="canonical SPINT-main/data/000954; its recordings are not opened by preflight")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(pair_terminal_checker=args.pair_terminal_checker, pair_preflight=args.pair_preflight,
                         target_data_root=args.target_data_root, output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
