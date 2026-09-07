#!/usr/bin/env python3
"""Fail-closed recovery worker for the interrupted RT MB4 GPU1 lane.

This deliberately *does not* edit or invoke the original MB4 worker.  The
original ``gpu1_after_asc`` worker has already produced one valid fold-0
cell, then exited before fold 1 created a directory because the exact-fold
R-LS writer gate was occupied.  This isolated surface can reuse precisely
that completed cell and execute only the missing folds 1--7 after the GPU0
descending controls session has gone away.

It preserves the original 15-fold matrix, config, seed, M=24, normalizer,
and one-shot outer evaluation contract.  It never chooses an order from a
score.  A recovery plan is receipt-only; ``--recover-gpu1`` is the only mode
that can construct a Trainer or CUDA process, and it must be started
explicitly from the already-created ``rt_mb4_gpu1_after_controls`` tmux
session.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import stat
import sys
import time
from typing import Any, Mapping


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
BASE_SCRIPT = PROJECT / "scripts/rt_mb4_matched_continuation.py"
ARM = "afc4_mb4"
SEED = 42
RECOVERY_FOLDS = tuple(range(1, 8))
RECOVERED_FOLDS = (0,) + RECOVERY_FOLDS
GPU = 1
CONTROLS_SESSION = "rt_controls_split_gpu0_after_desc"
RECOVERY_SESSION = "rt_mb4_gpu1_after_controls"
RECOVERY_SCHEMA = "rt_mb4_gpu1_isolated_recovery_plan_v1"
RECOVERY_STATUS = "PASS_RT_MB4_GPU1_RECOVERY_PLAN_NO_GPU_NO_NWB_NO_TRAINER"
DEFAULT_WORK_ROOT = PROJECT / "outputs/rt_mb4_matched_clean_nested_v1"
DEFAULT_CONTINUATION_PLAN = WORKSPACE / "sua_exploration/results/rt_mb4_matched_clean_nested_v1/RT_MB4_MATCHED_CONTINUATION_PLAN_v4.json"
DEFAULT_PREDECESSOR = WORKSPACE / "sua_exploration/results/rt_mb4_matched_clean_nested_v1/predecessor_terminals/rt_controls_asc_clean_terminal_v1.json"
DEFAULT_R_C_AGGREGATE = WORKSPACE / "sua_exploration/results/k4_rt_loso_v1/rt_seed42_clean_nested_aggregate.json"
DEFAULT_R_C_SEAL = WORKSPACE / "sua_exploration/results/k4_rt_loso_v1/rt_seed42_clean_nested_seal.marker"
DEFAULT_PLAN_OUTPUT = WORKSPACE / "sua_exploration/results/rt_mb4_matched_clean_nested_v1/RT_MB4_GPU1_RECOVERY_PLAN_v1.json"


class Mb4Gpu1RecoveryError(RuntimeError):
    """An isolated GPU1 recovery invariant failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Mb4Gpu1RecoveryError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _immutable(path: Path, label: str) -> None:
    _need(path.is_file(), f"{label} is missing: {path}")
    _need(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{label} must be mode 0444: {path}")


def _read_immutable_json(path: Path, label: str) -> tuple[Path, dict[str, Any], str]:
    candidate = path.resolve()
    _immutable(candidate, label)
    try:
        body = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise Mb4Gpu1RecoveryError(f"{label} is invalid JSON: {candidate}") from error
    _need(isinstance(body, dict), f"{label} must be a JSON object: {candidate}")
    return candidate, body, _sha256(candidate)


def _write_immutable(path: Path, body: Mapping[str, Any]) -> str:
    _need(not path.exists(), f"refusing to overwrite immutable recovery evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(body), indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)
    _immutable(path, "new recovery evidence")
    return _sha256(path)


def _base() -> Any:
    """Load the existing MB4 validator/worker without importing data/model code."""

    _need(BASE_SCRIPT.is_file(), f"base MB4 continuation is missing: {BASE_SCRIPT}")
    spec = importlib.util.spec_from_file_location("rt_mb4_continuation_for_gpu1_recovery", BASE_SCRIPT)
    _need(spec is not None and spec.loader is not None, "cannot load base MB4 continuation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plan_partition(plan: Mapping[str, Any], *, work_root: Path) -> Mapping[str, Any]:
    base = _base()
    _need(plan.get("schema") == "rt_mb4_matched_continuation_plan_v4" and
          plan.get("status") == "PLAN_ONLY_NO_GPU_NO_NWB_NO_TRAINER",
          "base MB4 continuation plan schema/status drift")
    _need(Path(str(plan.get("work_root", ""))).resolve() == work_root.resolve(),
          "base MB4 continuation work-root drift")
    partition = plan.get("partitions", {}).get("gpu1_after_asc") if isinstance(plan.get("partitions"), Mapping) else None
    _need(isinstance(partition, Mapping), "base plan lacks gpu1_after_asc partition")
    _need(partition.get("gpu") == GPU and tuple(partition.get("folds", ())) == tuple(range(8)),
          "base plan GPU1 fold matrix drift")
    cells = partition.get("cells")
    _need(isinstance(cells, list) and [row.get("fold") for row in cells if isinstance(row, Mapping)] == list(range(8)),
          "base plan GPU1 cells drift")
    for row in cells:
        _need(isinstance(row, Mapping), "base plan GPU1 cell is malformed")
        fold = int(row["fold"])
        expected = base.train_command(root=work_root, fold=fold, gpu=GPU)
        _need(row.get("train_command") == expected, f"base plan GPU1 fold {fold} train command drift")
    return partition


def _validate_common(*, work_root: Path, continuation_plan: Path, predecessor_terminal: Path,
                     aggregate_path: Path, seal_path: Path) -> tuple[Any, Any, dict[str, Any], str]:
    """Validate the sealed comparator, predecessor and original v4 plan."""

    base = _base()
    plan_path, plan, plan_sha = _read_immutable_json(continuation_plan, "base MB4 continuation plan")
    _plan_partition(plan, work_root=work_root)
    reference = base.load_rc_reference(aggregate_path=aggregate_path, seal_path=seal_path)
    bound = plan.get("r_c_reference")
    _need(isinstance(bound, Mapping) and bound.get("aggregate_sha256") == reference.aggregate_sha256 and
          bound.get("seal_sha256") == reference.seal_sha256 and
          bound.get("query_start_trial") == base.QUERY_START and bound.get("window_size") == base.WINDOW_SIZE,
          "base plan sealed Full R-C binding drift")
    predecessor = base.validate_predecessor_terminal(predecessor_terminal, expected_partition="asc")
    predecessor_ref = predecessor.get("r_c_reference")
    _need(isinstance(predecessor_ref, Mapping) and predecessor_ref.get("aggregate_sha256") == reference.aggregate_sha256 and
          predecessor_ref.get("seal_sha256") == reference.seal_sha256,
          "GPU1 predecessor does not bind the sealed Full R-C comparator")
    root = base._ensure_isolated_root(work_root)
    marker = root / "MB4_CONTINUATION_ROOT_v1.json"
    _need(marker.is_file(), f"MB4 recovery root lacks the original marker: {root}")
    _need(base._json(marker) == {"schema": "rt_mb4_continuation_root_v1", "arm": ARM, "seed": SEED},
          "MB4 recovery root marker drift")
    return base, reference, plan, plan_sha


def _cell_state(*, base: Any, root: Path, fold: int, reference: Any) -> dict[str, Any]:
    """Classify only a fully validated PASS bundle as skippable.

    ``validate_cell`` intentionally treats a FAILED terminal, a missing member,
    a mismatched config/split/checkpoint, and a bad outer receipt as an error.
    The recovery surface maps all of those to a hard failure rather than
    silently calling them "already done".
    """

    paths = base.cell_paths(root, fold)
    required = (paths.outer, paths.selection, paths.split, paths.config, paths.terminal)
    present = [path.exists() for path in required]
    if not any(present):
        _need(not paths.cell.exists(), f"MB4 recovery fold {fold} has a partial cell directory with no complete receipt bundle")
        return {"state": "missing", "fold": fold, "cell": str(paths.cell)}
    _need(all(present), f"MB4 recovery fold {fold} is partial and cannot be skipped or overwritten")
    try:
        row = base.validate_cell(root=root, fold=fold, reference=reference, require_import=False)
    except base.Mb4ContinuationError as error:
        raise Mb4Gpu1RecoveryError(f"MB4 recovery fold {fold} is failed/incomparable and cannot be skipped: {error}") from error
    _need(row.get("state") == "complete", f"MB4 recovery fold {fold} is not a complete PASS bundle")
    return dict(row)


def _validate_fold_inventory(*, base: Any, root: Path, reference: Any) -> dict[str, Any]:
    fold0 = _cell_state(base=base, root=root, fold=0, reference=reference)
    _need(fold0.get("state") == "complete", "MB4 recovery requires verified fold-0 PASS; it may not be rerun")
    rows = [_cell_state(base=base, root=root, fold=fold, reference=reference) for fold in RECOVERY_FOLDS]
    return {
        "fold0_verified_pass": fold0,
        "recovery_folds": rows,
        "skippable_verified_pass_folds": [row["fold"] for row in rows if row["state"] == "complete"],
        "fresh_missing_folds": [row["fold"] for row in rows if row["state"] == "missing"],
    }


def build_recovery_plan(*, work_root: Path, continuation_plan: Path, predecessor_terminal: Path,
                        aggregate_path: Path, seal_path: Path) -> dict[str, Any]:
    """Create a receipt-only frozen plan; no nwB, Trainer, CUDA or tmux call."""

    base, reference, _plan, continuation_sha = _validate_common(
        work_root=work_root, continuation_plan=continuation_plan, predecessor_terminal=predecessor_terminal,
        aggregate_path=aggregate_path, seal_path=seal_path,
    )
    inventory = _validate_fold_inventory(base=base, root=work_root.resolve(), reference=reference)
    fold0 = inventory["fold0_verified_pass"]
    return {
        "schema": RECOVERY_SCHEMA,
        "status": RECOVERY_STATUS,
        "mode": "receipt_only_no_nwb_no_trainer_no_cuda_no_tmux_mutation",
        "arm": ARM,
        "seed": SEED,
        "gpu": GPU,
        "work_root": str(work_root.resolve()),
        "base_continuation": {
            "path": str(continuation_plan.resolve()), "sha256": continuation_sha,
            "schema": "rt_mb4_matched_continuation_plan_v4",
        },
        "sealed_full_r_c": {
            "aggregate_path": str(reference.aggregate_path), "aggregate_sha256": reference.aggregate_sha256,
            "seal_path": str(reference.seal_path), "seal_sha256": reference.seal_sha256,
        },
        "verified_asc_predecessor": {
            "path": str(predecessor_terminal.resolve()), "sha256": _sha256(predecessor_terminal.resolve()),
            "schema": base.PREDECESSOR_SCHEMA, "partition": "asc",
        },
        "verified_fold0": {
            "fold": 0, "status": "PASS", "mb4_r2": fold0["mb4_r2"], "full_r2": fold0["full_r2"],
            "full_minus_mb4": fold0["full_minus_mb4"], "target_session": fold0["target_session"],
            "query_windows_evaluated": fold0["query_windows_evaluated"],
            "rule": "retained as an already complete matched cell; its signed delta is never a scheduling or stopping input",
        },
        "recovery": {
            "all_matrix_folds": list(range(15)), "recovered_gpu1_fold_set": list(RECOVERED_FOLDS),
            "fresh_execution_order": list(RECOVERY_FOLDS),
            "wait_for_tmux_session_to_disappear": CONTROLS_SESSION,
            "tmux_worker_session": RECOVERY_SESSION,
            "skip_rule": "only a complete receipt bundle that passes the original MB4 validator may be skipped; missing is fresh; partial, FAILED, invalid, or incomparable is fatal",
            "protocol": "same existing Hydra config, seed42, chronological M24, source-only normalizer, inner-only checkpoint choice, and one-shot outer evaluator",
            "stop_rule": "execute every currently missing fold 1..7; never inspect a signed effect to stop or reorder",
        },
        "preflight_inventory": inventory,
        "code_sha256": {"recovery": _sha256(Path(__file__).resolve()), "base_mb4": _sha256(BASE_SCRIPT)},
    }


def _validate_recovery_plan(path: Path, *, work_root: Path, continuation_plan: Path, predecessor_terminal: Path,
                            aggregate_path: Path, seal_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    candidate, plan, digest = _read_immutable_json(path, "MB4 GPU1 recovery plan")
    _need(plan.get("schema") == RECOVERY_SCHEMA and plan.get("status") == RECOVERY_STATUS,
          "MB4 GPU1 recovery plan schema/status drift")
    _need(plan.get("work_root") == str(work_root.resolve()) and plan.get("arm") == ARM and
          plan.get("seed") == SEED and plan.get("gpu") == GPU, "MB4 GPU1 recovery plan root/arm/seed/GPU drift")
    code = plan.get("code_sha256")
    _need(isinstance(code, Mapping) and code.get("recovery") == _sha256(Path(__file__).resolve()) and
          code.get("base_mb4") == _sha256(BASE_SCRIPT), "MB4 GPU1 recovery code closure drift")
    base, reference, _base_plan, base_plan_sha = _validate_common(
        work_root=work_root, continuation_plan=continuation_plan, predecessor_terminal=predecessor_terminal,
        aggregate_path=aggregate_path, seal_path=seal_path,
    )
    bound = plan.get("base_continuation")
    _need(isinstance(bound, Mapping) and bound.get("path") == str(continuation_plan.resolve()) and
          bound.get("sha256") == base_plan_sha, "MB4 GPU1 recovery plan base continuation binding drift")
    predecessor = plan.get("verified_asc_predecessor")
    _need(isinstance(predecessor, Mapping) and predecessor.get("path") == str(predecessor_terminal.resolve()) and
          predecessor.get("sha256") == _sha256(predecessor_terminal.resolve()),
          "MB4 GPU1 recovery plan predecessor binding drift")
    sealed = plan.get("sealed_full_r_c")
    _need(isinstance(sealed, Mapping) and sealed.get("aggregate_sha256") == reference.aggregate_sha256 and
          sealed.get("seal_sha256") == reference.seal_sha256, "MB4 GPU1 recovery sealed Full R-C binding drift")
    recovery = plan.get("recovery")
    _need(isinstance(recovery, Mapping) and tuple(recovery.get("recovered_gpu1_fold_set", ())) == RECOVERED_FOLDS and
          tuple(recovery.get("fresh_execution_order", ())) == RECOVERY_FOLDS and
          recovery.get("wait_for_tmux_session_to_disappear") == CONTROLS_SESSION and
          recovery.get("tmux_worker_session") == RECOVERY_SESSION,
          "MB4 GPU1 recovery matrix/session plan drift")
    return base, reference, plan


def _controls_session_exists(name: str = CONTROLS_SESSION) -> bool:
    """Return only a clean tmux present/absent state; malformed tmux fails closed."""

    try:
        completed = __import__("subprocess").run(
            ["tmux", "has-session", "-t", name], stdout=__import__("subprocess").DEVNULL,
            stderr=__import__("subprocess").DEVNULL, check=False,
        )
    except OSError as error:
        raise Mb4Gpu1RecoveryError("cannot inspect required GPU0 controls tmux session") from error
    _need(completed.returncode in (0, 1), f"tmux controls-session probe failed with exit={completed.returncode}")
    return completed.returncode == 0


def _wait_for_gpu0_controls_to_exit(*, poll_seconds: float) -> None:
    _need(poll_seconds > 0.0, "controls wait poll interval must be positive")
    while _controls_session_exists(CONTROLS_SESSION):
        time.sleep(poll_seconds)


def _lock_path(root: Path) -> Path:
    return root / "MB4_GPU1_RECOVERY_ACTIVE_v1.lock"


def _acquire_lock(root: Path, plan_path: Path) -> Path:
    lock = _lock_path(root)
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"schema": "rt_mb4_gpu1_recovery_active_lock_v1", "pid": os.getpid(),
                                     "recovery_plan": str(plan_path.resolve()), "plan_sha256": _sha256(plan_path.resolve())},
                                    sort_keys=True) + "\n")
    except FileExistsError as error:
        raise Mb4Gpu1RecoveryError(f"MB4 GPU1 recovery duplicate launch blocked by active/stale lock: {lock}") from error
    return lock


def _run_missing_fold(*, base: Any, root: Path, fold: int, reference: Any) -> dict[str, Any]:
    paths = base.cell_paths(root, fold)
    _need(not paths.cell.exists(), f"MB4 recovery fold {fold} unexpectedly exists after missing preflight")
    gpu_commands, all_rt_commands = base.active_compute_commands_on_gpu(GPU)
    # This check intentionally stays before mkdir: a same-fold active LS/RS
    # writer must leave no partial MB4 directory, exactly as the original
    # worker did for the interrupted fold 1.
    base.ensure_launch_safe(gpu_commands=gpu_commands, all_rt_commands=all_rt_commands, fold=fold, gpu=GPU)
    created = False
    try:
        paths.cell.mkdir(parents=True, exist_ok=False)
        created = True
        env = dict(os.environ)
        base._run_logged(base.train_command(root=root, fold=fold, gpu=GPU), cwd=base.PROJECT, env=env, log=paths.log)
        selection = base._json(paths.selection)
        base._validate_selection(selection, fold=fold)
        checkpoint = Path(str(selection.get("best_model_path", "")))
        _need(checkpoint.is_file(), f"MB4 recovery fold {fold} selected checkpoint is absent")
        base._run_logged(base.eval_command(root=root, fold=fold, checkpoint=checkpoint), cwd=base.PROJECT, env=env, log=paths.log)
        base.validate_cell(root=root, fold=fold, reference=reference, require_import=False, require_terminal=False)
        base._write_terminal(paths.terminal, {"status": "PASS", "arm": ARM, "fold": fold, "seed": SEED})
        return _cell_state(base=base, root=root, fold=fold, reference=reference)
    except Exception as error:
        if created and paths.cell.exists() and not paths.terminal.exists():
            base._write_terminal(paths.terminal, {"status": "FAILED", "arm": ARM, "fold": fold, "seed": SEED,
                                                "error": repr(error)})
        raise


def recover_gpu1(*, recovery_plan: Path, work_root: Path, continuation_plan: Path, predecessor_terminal: Path,
                 aggregate_path: Path, seal_path: Path, poll_seconds: float = 15.0) -> dict[str, Any]:
    """Run only the unexecuted GPU1 folds after GPU0's descending controls end."""

    _need(os.environ.get("CUDA_VISIBLE_DEVICES") == "1",
          "GPU1 recovery requires exactly CUDA_VISIBLE_DEVICES=1")
    base, reference, _plan = _validate_recovery_plan(
        recovery_plan, work_root=work_root, continuation_plan=continuation_plan,
        predecessor_terminal=predecessor_terminal, aggregate_path=aggregate_path, seal_path=seal_path,
    )
    inventory = _validate_fold_inventory(base=base, root=work_root.resolve(), reference=reference)
    lock = _acquire_lock(work_root.resolve(), recovery_plan)
    try:
        # The original abort was at the exact fold-1 global writer guard.
        # Waiting for this named descending lane makes the 1--7 order itself
        # static and avoids using outcomes to choose a nonstandard schedule.
        _wait_for_gpu0_controls_to_exit(poll_seconds=poll_seconds)
        completed, skipped = [], [0]
        for fold in RECOVERY_FOLDS:
            current = _cell_state(base=base, root=work_root.resolve(), fold=fold, reference=reference)
            if current["state"] == "complete":
                skipped.append(fold)
                continue
            _need(current["state"] == "missing", f"MB4 recovery fold {fold} changed to an unsafe state")
            _run_missing_fold(base=base, root=work_root.resolve(), fold=fold, reference=reference)
            completed.append(fold)
        final = _validate_fold_inventory(base=base, root=work_root.resolve(), reference=reference)
        _need(final["fresh_missing_folds"] == [], "GPU1 recovery finished with missing folds")
        return {"status": "PASS_RT_MB4_GPU1_ISOLATED_RECOVERY", "completed_folds": completed,
                "verified_skipped_folds": skipped, "fold0": inventory["fold0_verified_pass"],
                "final_inventory": final, "result_selection": "FORBIDDEN"}
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def tmux_send_command(*, recovery_plan: Path, work_root: Path, continuation_plan: Path, predecessor_terminal: Path,
                      aggregate_path: Path, seal_path: Path) -> str:
    """Return, but never execute, the command for the existing exact worker tmux."""

    python = shlex.quote(sys.executable)
    payload = " ".join((
        "cd", shlex.quote(str(PROJECT)), "&&", "exec", "env", "CUDA_VISIBLE_DEVICES=1", python,
        "scripts/rt_mb4_gpu1_recovery.py", "--recover-gpu1", "--recovery-plan", shlex.quote(str(recovery_plan.resolve())),
        "--work-root", shlex.quote(str(work_root.resolve())), "--continuation-plan", shlex.quote(str(continuation_plan.resolve())),
        "--predecessor-terminal", shlex.quote(str(predecessor_terminal.resolve())),
        "--rc-aggregate", shlex.quote(str(aggregate_path.resolve())), "--rc-seal", shlex.quote(str(seal_path.resolve())),
    ))
    return "tmux has-session -t {session} 2>/dev/null && tmux send-keys -t {session} {payload} C-m".format(
        session=shlex.quote(RECOVERY_SESSION), payload=shlex.quote(payload),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--recover-gpu1", action="store_true")
    modes.add_argument("--tmux-send-command", action="store_true")
    parser.add_argument("--recovery-plan", type=Path, default=DEFAULT_PLAN_OUTPUT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--continuation-plan", type=Path, default=DEFAULT_CONTINUATION_PLAN)
    parser.add_argument("--predecessor-terminal", type=Path, default=DEFAULT_PREDECESSOR)
    parser.add_argument("--rc-aggregate", type=Path, default=DEFAULT_R_C_AGGREGATE)
    parser.add_argument("--rc-seal", type=Path, default=DEFAULT_R_C_SEAL)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.recover_gpu1:
        result = recover_gpu1(recovery_plan=args.recovery_plan, work_root=args.work_root,
                              continuation_plan=args.continuation_plan, predecessor_terminal=args.predecessor_terminal,
                              aggregate_path=args.rc_aggregate, seal_path=args.rc_seal, poll_seconds=args.poll_seconds)
    elif args.tmux_send_command:
        result = {"tmux_send_command": tmux_send_command(
            recovery_plan=args.recovery_plan, work_root=args.work_root, continuation_plan=args.continuation_plan,
            predecessor_terminal=args.predecessor_terminal, aggregate_path=args.rc_aggregate, seal_path=args.rc_seal,
        ), "executed": False}
    else:
        body = build_recovery_plan(work_root=args.work_root, continuation_plan=args.continuation_plan,
                                   predecessor_terminal=args.predecessor_terminal, aggregate_path=args.rc_aggregate,
                                   seal_path=args.rc_seal)
        if args.plan:
            digest = _write_immutable(args.recovery_plan, body)
            result = {"status": body["status"], "recovery_plan": str(args.recovery_plan), "sha256": digest}
        else:
            result = body
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
