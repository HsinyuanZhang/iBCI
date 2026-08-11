#!/usr/bin/env python3
"""Prepare source-only RT Stage-R B2/D1024 fits in descending fold order.

This intentionally small harness occupies an otherwise idle GPU with the
target-free portion of the already-selected Stage-R comparison.  It uses the
same constructibility preflight, B2/D1024 train command, and post-fit
selection/config/split identity audit as the live Stage-R supervisor, but it
does *not* construct or invoke an outer evaluator.  It also never writes the
normal supervisor's ``cell_terminal.json`` or ``supervisor_summary.json``.

Each completed source fit receives its own immutable ``source_ready.json``.
That receipt is a capability for a later, separately reviewed one-shot target
evaluation; it is not an accuracy result.  A pre-existing terminal receipt,
source-ready receipt, partial cell, or any unexpected target-evaluation
artifact is a fail-closed collision.  The harness never resumes or repairs a
cell in place.

The default invocation prepares fold 14 on GPU 0.  Any subsequent descending
subrange (for example ``13 12 11 10``) must be explicit, which avoids
accidentally colliding with the live positive fold-3 supervisor on GPU 1.
This module only prepares fits; omitting ``--execute`` prints a non-mutating
plan.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import stat
import sys
from typing import Any, Iterable, Mapping


SUA_ROOT = Path(__file__).resolve().parents[1]
ROOT = SUA_ROOT.parent
if str(SUA_ROOT) not in sys.path:
    sys.path.insert(0, str(SUA_ROOT))

from scripts import run_rt_stage_r_b2_d1024_folds03_14_supervisor as stage_r


PROGRAM_ID = "rt_stage_r_b2_d1024_descending_source_precompute_v1"
SOURCE_READY_SCHEMA = "rt_stage_r_b2_d1024_source_ready_v1"
# A default of only fold 14 prevents an accidental broad launch into cells a
# concurrent supervisor may already own.  Longer descending subranges require
# an explicit ``--folds 13 12 ...`` declaration after fold 14 is sealed.
DEFAULT_FOLDS = (14,)
DEFAULT_GPU = 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _paths(run_root: Path, fold: int) -> dict[str, Path]:
    """Use the canonical Stage-R cell layout plus one additive receipt."""

    paths = dict(stage_r._paths(run_root, fold))
    paths["log"] = paths["cell"] / "source_precompute.log"
    paths["source_ready"] = paths["cell"] / "source_ready.json"
    return paths


def _validate_descending_folds(folds: Iterable[int]) -> tuple[int, ...]:
    """Permit only a contiguous decreasing subrange of the Stage-R scope."""

    values = tuple(int(value) for value in folds)
    if not values:
        raise ValueError("at least one fold is required")
    invalid = [value for value in values if value not in stage_r.ALLOWED_FOLDS]
    if invalid:
        raise ValueError(f"only Stage-R folds 3..14 are allowed; got {invalid}")
    if len(set(values)) != len(values):
        raise ValueError("descending source-precompute folds must be unique")
    expected = tuple(range(values[0], values[0] - len(values), -1))
    if values != expected:
        raise ValueError(
            "source-precompute folds must be a contiguous descending sequence "
            "(for example 14 13 12)"
        )
    return values


def _require_source_program(python: Path) -> None:
    """Require only components needed for a target-free source fit."""

    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(f"Python executable missing: {python}")
    for path in (stage_r.TRAIN, stage_r.PREFLIGHT):
        if not path.is_file():
            raise FileNotFoundError(f"required RT source-fit component missing: {path}")


def _cell_collision(paths: Mapping[str, Path]) -> str | None:
    """Classify any existing cell state as a non-resumable collision."""

    cell = paths["cell"]
    if not cell.exists():
        return None
    if paths["terminal"].exists():
        return "existing terminal receipt"
    if paths["source_ready"].exists():
        return "existing source-ready receipt"
    if paths["outer"].exists():
        return "unexpected existing target-evaluation artifact"
    return "existing partial cell"


def _require_fresh_cell(paths: Mapping[str, Path]) -> None:
    collision = _cell_collision(paths)
    if collision is not None:
        raise FileExistsError(f"refusing source-precompute cell collision ({collision}): {paths['cell']}")


def _assert_no_target_artifacts(paths: Mapping[str, Path]) -> None:
    """The target-facing artifact must be absent before and after source work."""

    if paths["outer"].exists():
        raise RuntimeError(f"source-only harness found a target-evaluation artifact: {paths['outer']}")
    if paths["terminal"].exists():
        raise RuntimeError(f"source-only harness found a terminal artifact: {paths['terminal']}")


def _source_ready_payload(*, paths: Mapping[str, Path], fold: int,
                          selection: Mapping[str, Any], checkpoint: Path) -> dict[str, Any]:
    """Bind all post-fit source artifacts without assigning a target metric."""

    for key in ("selected_epoch", "selected_global_step"):
        value = selection.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"source-ready receipt requires finite nonnegative integer {key}")
    artifacts = {
        "constructibility_preflight": paths["preflight"],
        "selection_receipt": paths["selection"],
        "resolved_config": paths["config"],
        "split_manifest": paths["split"],
        "selected_checkpoint": checkpoint,
    }
    if any(not path.is_file() for path in artifacts.values()):
        missing = [name for name, path in artifacts.items() if not path.is_file()]
        raise FileNotFoundError(f"cannot emit source-ready receipt; missing {missing}")
    return {
        "schema": SOURCE_READY_SCHEMA,
        "status": "PASS_SOURCE_ONLY_FIT_READY",
        "program_id": PROGRAM_ID,
        "created_utc": _now(),
        "fold": fold,
        "seed": stage_r.SEED,
        "arm": stage_r.ARM,
        "canonical_mechanism": "B2 LatePool D1024; zero4 is loader-only ignored input",
        "source_fit": {
            "selection_metric": selection["selected_by_metric"],
            "selection_metric_scope": selection["selected_metric_scope"],
            "checkpoint_epoch": selection["selected_epoch"],
            "checkpoint_global_step": selection["selected_global_step"],
        },
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": stage_r._sha256(path)}
            for name, path in artifacts.items()
        },
        "target_access": {
            "outer_evaluator_constructed": False,
            "outer_evaluator_run": False,
            "outer_target_loaded_during_fit": False,
            "outer_target_query_labels_read_during_fit": False,
            "formal_heldout_opened": False,
        },
        "writer_scope": {
            "cell_terminal_written": False,
            "supervisor_summary_written": False,
            "resume_or_overwrite_permitted": False,
        },
        "harness_sha256": stage_r._sha256(Path(__file__).resolve()),
    }


def _validate_source_ready(path: Path, *, fold: int) -> dict[str, Any]:
    """Read-only integrity checker used by tests and later audited consumers."""

    if not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise ValueError(f"source-ready receipt must be immutable mode 0444: {path}")
    receipt = stage_r._json(path)
    required = {
        "schema": SOURCE_READY_SCHEMA,
        "status": "PASS_SOURCE_ONLY_FIT_READY",
        "program_id": PROGRAM_ID,
        "fold": fold,
        "seed": stage_r.SEED,
        "arm": stage_r.ARM,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("source-ready receipt identity mismatch")
    access = receipt.get("target_access")
    writer = receipt.get("writer_scope")
    if not isinstance(access, Mapping) or not isinstance(writer, Mapping):
        raise ValueError("source-ready receipt lacks access/write scope")
    expected_access = {
        "outer_evaluator_constructed": False,
        "outer_evaluator_run": False,
        "outer_target_loaded_during_fit": False,
        "outer_target_query_labels_read_during_fit": False,
        "formal_heldout_opened": False,
    }
    expected_writer = {
        "cell_terminal_written": False,
        "supervisor_summary_written": False,
        "resume_or_overwrite_permitted": False,
    }
    if any(access.get(key) != value for key, value in expected_access.items()):
        raise ValueError("source-ready receipt records target access")
    if any(writer.get(key) != value for key, value in expected_writer.items()):
        raise ValueError("source-ready receipt records an illegal writer scope")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("source-ready receipt lacks source artifacts")
    for name in ("constructibility_preflight", "selection_receipt", "resolved_config", "split_manifest", "selected_checkpoint"):
        item = artifacts.get(name)
        if not isinstance(item, Mapping):
            raise ValueError(f"source-ready receipt lacks {name}")
        candidate = Path(str(item.get("path", "")))
        if not candidate.is_file() or stage_r._sha256(candidate) != item.get("sha256"):
            raise ValueError(f"source-ready artifact drift: {name}")
    return receipt


def plan(*, run_root: Path, folds: Iterable[int], python: Path, gpu: int) -> dict[str, Any]:
    """Print executable source-fit commands only; create no artifact."""

    _require_source_program(python)
    selected = _validate_descending_folds(folds)
    if gpu < 0:
        raise ValueError("--gpu must be nonnegative")
    root = run_root.resolve()
    cells: dict[str, Any] = {}
    for fold in selected:
        paths = _paths(root, fold)
        _require_fresh_cell(paths)
        cells[f"fold_{fold:02d}"] = {
            "preflight": shlex.join(stage_r.preflight_command(
                python=python, paths=paths, run_root=root, fold=fold
            )),
            "source_fit": shlex.join(stage_r.train_command(
                python=python, paths=paths, run_root=root, fold=fold
            )),
            "source_ready_receipt": str(paths["source_ready"]),
        }
    return {
        "schema": "rt_stage_r_b2_d1024_descending_source_precompute_plan_v1",
        "mode": "plan_only_no_execution",
        "program_id": PROGRAM_ID,
        "folds": list(selected),
        "gpu_for_source_fit_only": gpu,
        "execution_order": "strict_descending_no_resume_v1",
        "target_access": "absent_from_this_harness",
        "cells": cells,
    }


def prepare_fold(*, run_root: Path, fold: int, python: Path, gpu: int) -> Path:
    """Run exactly one target-free preflight+fit transaction and seal readiness."""

    _require_source_program(python)
    _validate_descending_folds([fold])
    if gpu < 0:
        raise ValueError("--gpu must be nonnegative")
    paths = _paths(run_root.resolve(), fold)
    _require_fresh_cell(paths)
    cpu_env = dict(os.environ)
    cpu_env["CUDA_VISIBLE_DEVICES"] = ""
    gpu_env = dict(os.environ)
    gpu_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gpu_env["PYTHONUNBUFFERED"] = "1"

    code = stage_r._run(
        stage_r.preflight_command(python=python, paths=paths, run_root=run_root.resolve(), fold=fold),
        cwd=stage_r.STREAMING_ROOT,
        env=cpu_env,
        log_path=paths["log"],
    )
    if code != 0:
        raise RuntimeError(f"RT source-only preflight failed for fold {fold} with exit={code}")
    stage_r._validate_preflight(paths["preflight"], fold=fold)
    _assert_no_target_artifacts(paths)

    code = stage_r._run(
        stage_r.train_command(python=python, paths=paths, run_root=run_root.resolve(), fold=fold),
        cwd=stage_r.STREAMING_ROOT,
        env=gpu_env,
        log_path=paths["log"],
    )
    if code != 0:
        raise RuntimeError(f"RT source-only fit failed for fold {fold} with exit={code}")
    selection = stage_r._validate_fit_identity(paths, fold=fold)
    checkpoint = stage_r._selected_checkpoint(paths["selection"])
    _assert_no_target_artifacts(paths)

    payload = _source_ready_payload(paths=paths, fold=fold, selection=selection, checkpoint=checkpoint)
    stage_r._write_once(paths["source_ready"], payload)
    _validate_source_ready(paths["source_ready"], fold=fold)
    return paths["source_ready"]


def execute(*, run_root: Path, folds: Iterable[int], python: Path, gpu: int) -> dict[str, str]:
    """Prepare folds serially; a single collision/failure stops all later cells."""

    selected = _validate_descending_folds(folds)
    results: dict[str, str] = {}
    for fold in selected:
        receipt = prepare_fold(run_root=run_root, fold=fold, python=python, gpu=gpu)
        results[f"fold_{fold:02d}"] = str(receipt)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(DEFAULT_FOLDS))
    parser.add_argument("--python", type=Path, default=stage_r.DEFAULT_PYTHON)
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    parser.add_argument("--execute", action="store_true", help="required to launch any source fit")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(plan(
            run_root=args.run_root, folds=args.folds, python=args.python, gpu=args.gpu
        ), indent=2, sort_keys=True))
        return 0
    results = execute(run_root=args.run_root, folds=args.folds, python=args.python, gpu=args.gpu)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
