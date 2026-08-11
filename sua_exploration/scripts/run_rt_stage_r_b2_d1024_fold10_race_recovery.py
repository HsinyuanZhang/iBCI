#!/usr/bin/env python3
"""One-shot, isolated recovery for the RT Stage-R B2/D1024 fold-10 race.

The original supervisor wrote an immutable ``STOP_PARTIAL_FIT_WITHOUT_SELECTION``
receipt while its descendant fit was still running.  It is unsafe to resume
that directory: a checkpoint could have been written before the selection
callback and the previous terminal makes its provenance ambiguous.

This tool never modifies that historical cell.  Its only execution mode is an
explicit, serial fresh transaction in ``fold10_race_recovery_v1``:

1. retain and hash the immutable old failed receipt;
2. create a new root which must not have existed before launch;
3. run the existing target-free CPU preflight;
4. start one fresh M24/seed-42/B2-D1024 source fit with ``ckpt_path=null``;
5. validate the selection/config/split binding before one CPU outer eval; and
6. write an immutable recovery terminal that binds every source hash.

Omitting ``--execute-recovery`` only emits this plan.  No GPU, data, or outer
evaluation is created in plan mode.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Mapping

# ``python scripts/…py`` puts ``scripts/`` rather than the project root on
# sys.path.  Keep the executable entry point as usable as the test import.
SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PROJECT_ROOT))

from scripts import run_rt_stage_r_b2_d1024_folds03_14_supervisor as supervisor
from scripts import rt_stage_r_b2_d1024_fold10_race_recovery_checker as checker


FOLD = checker.FOLD
SEED = checker.SEED
ARM = checker.ARM
DEFAULT_PYTHON = supervisor.DEFAULT_PYTHON
PROGRAM_ID = "rt_stage_r_b2_d1024_fold10_race_recovery_v1"
LAUNCH_SCHEMA = "rt_stage_r_b2_d1024_fold10_race_recovery_launch_v1"
TERMINAL_SCHEMA = checker.RECOVERY_TERMINAL_SCHEMA
SUCCESS_STATUS = checker.RECOVERY_TERMINAL_STATUS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Reuse the supervisor's atomic 0444 receipt discipline."""

    supervisor._write_once(path, payload)


def _file_binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def fresh_train_command(*, python: Path, paths: Mapping[str, Path], run_root: Path) -> list[str]:
    """Exact frozen fit with an explicit prohibition on checkpoint reuse."""

    command = supervisor.train_command(
        python=python, paths=paths, run_root=run_root, fold=FOLD,
    )
    if any(part.startswith("ckpt_path=") for part in command):
        raise RuntimeError("base Stage-R command unexpectedly selected a checkpoint")
    return [*command, "ckpt_path=null"]


def _command_sha256(command: list[str]) -> str:
    return hashlib.sha256(shlex.join(command).encode("utf-8")).hexdigest()


def _require_fresh_recovery_root(recovery_root: Path) -> None:
    """A partial recovery is also non-resumable; do not create a second race."""

    if recovery_root.exists() or recovery_root.is_symlink():
        raise FileExistsError(
            "fold-10 recovery root already exists; refusing resume/reuse: "
            f"{recovery_root}"
        )


def _assert_launch_isolation(gpu: int) -> None:
    """Refuse a retry while its GPU or exact fold still has a writer.

    The first race was possible because two schedulers could observe the same
    cell at different stages.  A fresh directory alone does not prevent two
    fit processes from competing for the device or the fold.  Check both the
    requested GPU process table and CPU-side command lines immediately before
    the recovery root is created.
    """

    try:
        gpu_table = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            text=True,
        )
        uuid_by_index = {
            int(line.split(",", 1)[0].strip()): line.split(",", 1)[1].strip()
            for line in gpu_table.splitlines() if "," in line
        }
        if gpu not in uuid_by_index:
            raise RuntimeError(f"requested recovery GPU {gpu} is not observable")
        app_table = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
            text=True,
        )
        ps_table = subprocess.check_output(["ps", "-eo", "args="], text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot prove fold-10 recovery launch isolation") from error

    gpu_pids = []
    for line in app_table.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[1] == uuid_by_index[gpu] and fields[0].isdigit():
            gpu_pids.append(int(fields[0]))
    if gpu_pids:
        raise RuntimeError(f"requested recovery GPU {gpu} still has compute owners: {gpu_pids}")

    fold_token = f"data.loso_fold={FOLD}"
    collisions = [
        line for line in ps_table.splitlines()
        if "rt_clean_nested" in line.lower() and fold_token in line
    ]
    if collisions:
        raise RuntimeError(f"active exact-fold RT writer blocks fold-10 recovery: {collisions}")


def _validate_fresh_fit_lineage(paths: Mapping[str, Path]) -> None:
    """Audit the saved fresh fit after the normal source-only identity audit.

    The command-line override is necessary but not sufficient evidence: this
    closes the loop on the actual saved Hydra configuration and rejects a
    selected checkpoint outside the newly created ``fit`` directory.
    """

    config = paths["config"]
    text = config.read_text(encoding="utf-8")
    if re.search(r"(?m)^ckpt_path:\s*null\s*$", text) is None:
        raise ValueError("fresh recovery saved config does not bind ckpt_path: null")
    selection = supervisor._json(paths["selection"])
    selected = selection.get("best_model_path")
    if not isinstance(selected, str) or not selected:
        raise ValueError("fresh recovery selection does not name a checkpoint")
    checkpoint = Path(selected).resolve()
    try:
        checkpoint.relative_to(paths["fit"].resolve())
    except ValueError as error:
        raise ValueError("fresh recovery selected checkpoint escapes its new fit directory") from error


def plan(*, recovery_root: Path, python: Path, gpu: int,
         old_terminal_path: Path = checker.old_failure_terminal()) -> dict[str, Any]:
    """Return the fully bound recovery plan without creating a directory."""

    supervisor._require_program(python)
    if gpu < 0:
        raise ValueError("--gpu must be nonnegative")
    recovery_root = recovery_root.resolve()
    _require_fresh_recovery_root(recovery_root)
    old_failure = checker.validate_old_failure(terminal_path=old_terminal_path)
    paths = checker.recovery_paths(recovery_root)
    train = fresh_train_command(python=python, paths=paths, run_root=recovery_root)
    preflight = supervisor.preflight_command(
        python=python, paths=paths, run_root=recovery_root, fold=FOLD,
    )
    return {
        "schema": LAUNCH_SCHEMA,
        "mode": "plan_only_no_execution",
        "program_id": PROGRAM_ID,
        "old_failure_preserved": old_failure,
        "recovery_root": str(recovery_root),
        "freshness_contract": {
            "root_must_not_exist_before_execute": True,
            "old_cell_reused": False,
            "warmstart_forbidden": True,
            "configured_ckpt_path": None,
            "fit_attempts": 1,
        },
        "frozen_cell": {
            "fold": FOLD,
            "seed": SEED,
            "arm": ARM,
            "calibration_trials": supervisor.M,
            "query_start_trial": supervisor.M,
            "id_hidden_dim": supervisor.HIDDEN_DIM,
            "max_epochs": supervisor.EPOCHS,
            "test": False,
        },
        "commands": {
            "target_free_preflight_cpu": shlex.join(preflight),
            "fresh_source_fit_gpu_only_if_execute": shlex.join(train),
            "outer_eval_cpu_after_selection_config_split_audit": (
                "constructed only after fresh fit passes the existing target-free audit"
            ),
        },
        "formal_heldout_opened": False,
    }


def _launch_payload(*, recovery_root: Path, old_failure: Mapping[str, Any],
                    preflight: list[str], train: list[str], gpu: int) -> dict[str, Any]:
    return {
        "schema": LAUNCH_SCHEMA,
        "status": "EXECUTION_STARTED_FRESH_ROOT",
        "program_id": PROGRAM_ID,
        "created_utc": _now(),
        "old_failure_preserved": dict(old_failure),
        "recovery_root": str(recovery_root),
        "fold": FOLD,
        "seed": SEED,
        "arm": ARM,
        "freshness": {
            "old_cell_reused": False,
            "warmstart_forbidden": True,
            "configured_ckpt_path": None,
            "recovery_fit_attempts": 1,
        },
        "commands": {
            "preflight": {"argv": preflight, "sha256": _command_sha256(preflight)},
            "fresh_train": {"argv": train, "sha256": _command_sha256(train), "gpu": gpu},
        },
        "formal_heldout_opened": False,
    }


def _terminal_payload(*, status: str, recovery_root: Path, paths: Mapping[str, Path],
                      old_failure: Mapping[str, Any], stage: str,
                      code: int | None = None, error: BaseException | None = None,
                      train_command: list[str] | None = None) -> dict[str, Any]:
    files = {
        name: _file_binding(paths[name])
        for name in ("preflight", "selection", "config", "split", "outer")
        if paths[name].is_file()
    }
    if paths["selection"].is_file():
        selection = supervisor._json(paths["selection"])
        selected = selection.get("best_model_path")
        if isinstance(selected, str) and selected:
            checkpoint = Path(selected).resolve()
            if checkpoint.is_file():
                files["selected_checkpoint"] = _file_binding(checkpoint)
    return {
        "schema": TERMINAL_SCHEMA,
        "status": status,
        "program_id": PROGRAM_ID,
        "created_utc": _now(),
        "fold": FOLD,
        "seed": SEED,
        "arm": ARM,
        "stage": stage,
        "exit_code": code,
        "error": None if error is None else repr(error),
        "preserved_old_failure": dict(old_failure),
        "freshness": {
            "recovery_root": str(recovery_root),
            "cell": str(paths["cell"]),
            "old_cell_reused": False,
            "warmstart_forbidden": True,
            "configured_ckpt_path": None,
            "recovery_fit_attempts": 1,
        },
        "files": files,
        "train_command_sha256": None if train_command is None else _command_sha256(train_command),
        "formal_heldout_opened": False,
        "failure_preserved_no_overwrite": True,
    }


def execute(*, recovery_root: Path, python: Path, gpu: int,
            old_terminal_path: Path = checker.old_failure_terminal()) -> dict[str, Any]:
    """Run exactly one serial fresh retry.  This is the only GPU-capable path."""

    supervisor._require_program(python)
    if gpu < 0:
        raise ValueError("--gpu must be nonnegative")
    recovery_root = recovery_root.resolve()
    _require_fresh_recovery_root(recovery_root)
    old_failure = checker.validate_old_failure(terminal_path=old_terminal_path)
    _assert_launch_isolation(gpu)
    paths = checker.recovery_paths(recovery_root)
    preflight = supervisor.preflight_command(
        python=python, paths=paths, run_root=recovery_root, fold=FOLD,
    )
    train = fresh_train_command(python=python, paths=paths, run_root=recovery_root)
    _write_once(paths["launch"], _launch_payload(
        recovery_root=recovery_root, old_failure=old_failure, preflight=preflight,
        train=train, gpu=gpu,
    ))

    cpu_env = dict(os.environ)
    cpu_env["CUDA_VISIBLE_DEVICES"] = ""
    gpu_env = dict(os.environ)
    gpu_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gpu_env["PYTHONUNBUFFERED"] = "1"
    try:
        code = supervisor._run(preflight, cwd=supervisor.STREAMING_ROOT, env=cpu_env, log_path=paths["log"])
        if code:
            _write_once(paths["terminal"], _terminal_payload(
                status="PREFLIGHT_FAILED", recovery_root=recovery_root, paths=paths,
                old_failure=old_failure, stage="preflight", code=code, train_command=train,
            ))
            return {"status": "preflight_failed", "terminal": str(paths["terminal"])}
        supervisor._validate_preflight(paths["preflight"], fold=FOLD)
        code = supervisor._run(train, cwd=supervisor.STREAMING_ROOT, env=gpu_env, log_path=paths["log"])
        if code:
            _write_once(paths["terminal"], _terminal_payload(
                status="FIT_FAILED", recovery_root=recovery_root, paths=paths,
                old_failure=old_failure, stage="fit", code=code, train_command=train,
            ))
            return {"status": "fit_failed", "terminal": str(paths["terminal"])}
        supervisor._validate_fit_identity(paths, fold=FOLD)
        _validate_fresh_fit_lineage(paths)
        if paths["outer"].exists():
            raise RuntimeError("fresh recovery unexpectedly found an outer receipt before evaluation")
        evaluate = supervisor.eval_command(python=python, paths=paths, fold=FOLD)
        code = supervisor._run(evaluate, cwd=supervisor.STREAMING_ROOT, env=cpu_env, log_path=paths["log"])
        if code:
            _write_once(paths["terminal"], _terminal_payload(
                status="OUTER_EVAL_FAILED", recovery_root=recovery_root, paths=paths,
                old_failure=old_failure, stage="outer_eval", code=code, train_command=train,
            ))
            return {"status": "outer_eval_failed", "terminal": str(paths["terminal"])}
        supervisor._validate_outer(paths, fold=FOLD)
        _write_once(paths["terminal"], _terminal_payload(
            status=SUCCESS_STATUS, recovery_root=recovery_root, paths=paths,
            old_failure=old_failure, stage="complete", train_command=train,
        ))
        checker.validate_recovery_terminal(
            recovery_root=recovery_root, old_terminal_path=old_terminal_path,
        )
        return {"status": "passed", "terminal": str(paths["terminal"])}
    except BaseException as error:
        if not paths["terminal"].exists():
            _write_once(paths["terminal"], _terminal_payload(
                status="RECOVERY_EXCEPTION", recovery_root=recovery_root, paths=paths,
                old_failure=old_failure, stage="exception", error=error, train_command=train,
            ))
        return {"status": "recovery_exception", "terminal": str(paths["terminal"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", type=Path, default=checker.DEFAULT_RECOVERY_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--old-terminal", type=Path, default=checker.old_failure_terminal())
    parser.add_argument("--execute-recovery", action="store_true", help="required to run preflight, fit, or outer evaluation")
    args = parser.parse_args()
    if not args.execute_recovery:
        print(json.dumps(plan(
            recovery_root=args.recovery_root, python=args.python, gpu=args.gpu,
            old_terminal_path=args.old_terminal,
        ), indent=2, sort_keys=True))
        return 0
    result = execute(
        recovery_root=args.recovery_root, python=args.python, gpu=args.gpu,
        old_terminal_path=args.old_terminal,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
