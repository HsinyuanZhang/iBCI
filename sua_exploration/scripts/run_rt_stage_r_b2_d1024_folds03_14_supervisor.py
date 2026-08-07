#!/usr/bin/env python3
"""Clean-RT supervisor for Stage-R B2/D1024 folds 3--14.

This is an additive execution harness, not a new RT model or endpoint.  It
uses the existing Stage-R zero4 constructibility preflight, the existing
inner-source selection receipt callback, and the existing one-shot outer
evaluator.  Folds 1--2 have their own queue and are deliberately rejected.

Each fold is an independent transaction in a fresh directory:

1. CPU-only constructibility preflight, with no NWB, Trainer, or CUDA use;
2. GPU source fit (`test=false`) with inner-validation-only checkpoint choice;
3. CPU one-shot outer-session forward evaluation after receipt validation.

With ``failure_policy=continue`` the supervisor uses a deliberately limited
one-level pipeline: while the single CPU outer evaluator for fold *N* runs,
the main thread can produce and audit the source fit for fold *N + 1* on the
single assigned GPU.  It harvests (and terminalizes) fold *N* before opening
the outer evaluator for fold *N + 1*.  Thus there is never more than one
target evaluator or one source fit, and no evaluator can be constructed
before that fold's selection/config/split audit succeeds.  ``stop`` remains
strictly serial so a failure cannot be followed by speculative source work.

The supervisor has no formal-heldout endpoint.  It writes an immutable cell
terminal receipt for either success or failure, then by default continues to
the next independent fold.  It never overwrites a cell, checkpoint, outer
evaluation receipt, terminal receipt, or program summary.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import traceback
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
STREAMING_ROOT = ROOT / "streaming_calibration_exp"
TRAIN = STREAMING_ROOT / "src" / "train.py"
PREFLIGHT = STREAMING_ROOT / "scripts" / "preflight_rt_stage_r_b2_zero4.py"
OUTER_EVALUATOR = STREAMING_ROOT / "src" / "rt_clean_nested_loso_eval.py"
DEFAULT_PYTHON = Path("/home/xinyuan/miniconda3/envs/spint/bin/python3.10")
PROGRAM_ID = "rt_stage_r_b2_d1024_folds03_14_pipeline_v2"
TERMINAL_SCHEMA = "rt_stage_r_b2_d1024_fold_terminal_v2"
SUMMARY_SCHEMA = "rt_stage_r_b2_d1024_folds03_14_supervisor_summary_v2"
ARM = "zero4"
SEED = 42
M = 24
HIDDEN_DIM = 1024
EPOCHS = 35
ALLOWED_FOLDS = frozenset(range(3, 15))


@dataclass(frozen=True)
class PreparedOuter:
    """A source-fitted fold that passed every target-free gate."""

    fold: int
    paths: Mapping[str, Path]


@dataclass(frozen=True)
class PendingOuter:
    """Exactly one CPU target evaluator launched after a legal source fit."""

    fold: int
    paths: Mapping[str, Path]
    future: Future[int]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_once(path: Path, payload: Mapping[str, Any], *, mode: int = 0o444) -> None:
    """Atomically create an immutable receipt; collision means do not resume."""

    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite RT Stage-R receipt: {path}") from None
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(body)
    path.chmod(mode)


def _cell_root(run_root: Path, fold: int) -> Path:
    return run_root / "cells" / "b2_d1024_zero4" / f"fold_{fold:02d}" / "seed_42"


def _paths(run_root: Path, fold: int) -> dict[str, Path]:
    cell = _cell_root(run_root, fold)
    fit = cell / "fit"
    return {
        "cell": cell,
        "preflight": cell / "constructibility_preflight.json",
        "log": cell / "supervisor.log",
        "fit": fit,
        "selection": fit / "rt_nested_selection_receipt.json",
        "config": fit / ".hydra" / "config.yaml",
        "split": fit / "split_manifest.json",
        "outer": cell / "outer_target_eval.json",
        "terminal": cell / "cell_terminal.json",
    }


def _validate_folds(folds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(value) for value in folds))
    if not values:
        raise ValueError("at least one fold is required")
    invalid = [value for value in values if value not in ALLOWED_FOLDS]
    if invalid:
        raise ValueError(f"only Stage-R expansion folds 3..14 are allowed; got {invalid}")
    return values


def _require_program(python: Path) -> None:
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(f"Python executable missing: {python}")
    for path in (TRAIN, PREFLIGHT, OUTER_EVALUATOR):
        if not path.is_file():
            raise FileNotFoundError(f"required existing RT component missing: {path}")


def preflight_command(*, python: Path, paths: Mapping[str, Path], run_root: Path, fold: int) -> list[str]:
    return [
        str(python), str(PREFLIGHT), "--output", str(paths["preflight"]), "--run-root", str(run_root),
        "--fold", str(fold), "--seed", str(SEED),
    ]


def train_command(*, python: Path, paths: Mapping[str, Path], run_root: Path, fold: int) -> list[str]:
    """Exact B2/D1024 source-fit command; unlike generic RT runner, B2-specific."""

    return [
        str(python), str(TRAIN), "experiment=rt_clean_nested_loso_b2_stage_r_zero4",
        f"run_id=rt_clean_nested_loso_m24_b2_d{HIDDEN_DIM}_zero4",
        f"data.loso_fold={fold}", f"data.outer_loso_fold={fold}",
        f"model.id_hidden_dim={HIDDEN_DIM}", f"seed={SEED}", "test=false",
        "trainer.accelerator=gpu", "trainer.devices=1", f"hydra.run.dir={paths['fit']}",
        f"paths.log_dir={run_root / '_hydra_logs'}", f"paths.artifact_dir={run_root / '_artifacts'}",
    ]


def eval_command(*, python: Path, paths: Mapping[str, Path], fold: int) -> list[str]:
    """Existing one-shot evaluator, deliberately forced to CPU after GPU fit."""

    return [
        str(python), str(OUTER_EVALUATOR), "--config", str(paths["config"]),
        "--checkpoint", str(_selected_checkpoint(paths["selection"])),
        "--split-manifest", str(paths["split"]), "--selection-receipt", str(paths["selection"]),
        "--output", str(paths["outer"]), "--outer-fold", str(fold), "--device", "cpu",
    ]


def _run(command: list[str], *, cwd: Path, env: Mapping[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("$ " + shlex.join(command) + "\n")
        stream.flush()
        completed = subprocess.run(command, cwd=cwd, env=dict(env), stdout=stream, stderr=subprocess.STDOUT)
        stream.write(f"[exit={completed.returncode}]\n")
        stream.flush()
    return int(completed.returncode)


def _selected_checkpoint(selection_path: Path) -> Path:
    selection = _validate_selection(selection_path, fold=None)
    checkpoint = Path(str(selection["best_model_path"])).resolve()
    if not checkpoint.is_file() or _sha256(checkpoint) != selection["best_model_sha256"]:
        raise ValueError("selected RT checkpoint missing or differs from inner-selection receipt")
    return checkpoint


def _validate_preflight(path: Path, *, fold: int) -> dict[str, Any]:
    payload = _json(path)
    if payload.get("schema") != "rt_stage_r_b2_zero4_constructibility_preflight_v2":
        raise ValueError("unexpected Stage-R preflight schema")
    if payload.get("status") != "READY_NOT_LAUNCHED" or int(payload.get("fold", -1)) != fold:
        raise ValueError("Stage-R preflight did not bind this fold")
    if payload.get("nwb_opened") is not False or payload.get("cuda_touched") is not False:
        raise ValueError("Stage-R preflight unexpectedly opened data or CUDA")
    arm = payload.get("arms", {}).get("R-S")
    if not isinstance(arm, Mapping) or int(arm.get("id_hidden_dim", -1)) != HIDDEN_DIM:
        raise ValueError("Stage-R preflight did not produce the D1024 arm")
    cost = arm.get("cost", {})
    if not isinstance(cost, Mapping) or str(cost.get("variant")) != "B2":
        raise ValueError("Stage-R preflight cost does not bind B2")
    construction = arm.get("object_construction", {})
    required = {
        "datamodule_setup_called": False,
        "outer_target_loaded": False,
        "outer_target_query_labels_read": False,
        "model_teacher_initialized": False,
        "model_student_initialized": False,
    }
    if not isinstance(construction, Mapping) or any(construction.get(k) != v for k, v in required.items()):
        raise ValueError("Stage-R preflight object construction leakage")
    return payload


def _validate_selection(path: Path, *, fold: int | None) -> dict[str, Any]:
    receipt = _json(path)
    required = {
        "schema": "rt_clean_nested_loso_selection_receipt_v1",
        "status": "PASS_FIT_INNER_SELECTION_ONLY",
        "arm": ARM,
        "seed": SEED,
        "selected_by_metric": "val_heldin/r2_mean",
        "selected_metric_scope": "inner_validation_session_only",
        "formal_heldout_opened": False,
        "outer_target_loaded_during_fit": False,
        "outer_target_query_labels_read_during_fit": False,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("RT selection receipt violates clean inner-only contract")
    if fold is not None and int(receipt.get("outer_loso_fold", -1)) != fold:
        raise ValueError("RT selection receipt fold mismatch")
    for field in ("best_model_path", "best_model_sha256", "config_path", "config_sha256", "split_manifest_path", "split_manifest_sha256"):
        if not receipt.get(field):
            raise ValueError(f"RT selection receipt lacks {field}")
    return receipt


def _validate_fit_identity(paths: Mapping[str, Path], *, fold: int) -> dict[str, Any]:
    selection = _validate_selection(paths["selection"], fold=fold)
    for path, digest, name in (
        (paths["config"], selection["config_sha256"], "config"),
        (paths["split"], selection["split_manifest_sha256"], "split manifest"),
    ):
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"RT selected {name} differs from its receipt")
    if Path(str(selection["config_path"])).resolve() != paths["config"].resolve():
        raise ValueError("RT selection config path mismatch")
    if Path(str(selection["split_manifest_path"])).resolve() != paths["split"].resolve():
        raise ValueError("RT selection split path mismatch")
    config_text = paths["config"].read_text(encoding="utf-8")
    for fragment in (
        "variant: B2", "id_hidden_dim: 1024", "side_feature_group: zero4",
        "calibration_n_trials: 24", "query_start_trial: 24", "freeze_decoder: false", "loss_mode: task_only",
    ):
        if fragment not in config_text:
            raise ValueError(f"RT D1024 config lacks fixed contract fragment {fragment!r}")
    return selection


def _validate_outer(paths: Mapping[str, Path], *, fold: int) -> dict[str, Any]:
    outer = _json(paths["outer"])
    required = {
        "schema": "rt_clean_nested_loso_outer_eval_v1",
        "status": "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP",
        "arm": ARM,
        "outer_loso_fold": fold,
        "seed": SEED,
        "query_start_trial": M,
        "target_backpropagation": False,
        "optimizer_present": False,
        "model_training_mode": False,
        "model_state_unchanged": True,
        "target_query_labels_used_for_calibration": False,
        "target_query_labels_used_for_normalization": False,
        "target_query_labels_used_for_checkpoint_selection": False,
        "target_query_labels_used_for_scoring_only": True,
    }
    if any(outer.get(key) != value for key, value in required.items()):
        raise ValueError("RT outer receipt violates one-shot forward-only contract")
    if outer.get("model_state_sha256_before") != outer.get("model_state_sha256_after"):
        raise ValueError("RT outer evaluator changed model state")
    if not isinstance(outer.get("r2_variance_weighted"), (float, int)):
        raise ValueError("RT outer receipt lacks scalar R2")
    return outer


def _terminal_payload(*, status: str, fold: int, paths: Mapping[str, Path], stage: str,
                      code: int | None = None, error: BaseException | None = None) -> dict[str, Any]:
    files = {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
             if name in {"preflight", "selection", "config", "split", "outer"} and path.is_file()}
    return {
        "schema": TERMINAL_SCHEMA,
        "status": status,
        "program_id": PROGRAM_ID,
        "arm": ARM,
        "canonical_mechanism": "B2 LatePool with no side carrier; zero4 is loader-only ignored input",
        "fold": fold,
        "seed": SEED,
        "stage": stage,
        "exit_code": code,
        "error": None if error is None else repr(error),
        "created_utc": _now(),
        "supervisor_sha256": _sha256(Path(__file__).resolve()),
        "files": files,
        "formal_heldout_opened": False,
        "failure_preserved_no_overwrite": True,
    }


def plan(*, run_root: Path, folds: Iterable[int], python: Path, gpu: int) -> dict[str, Any]:
    _require_program(python)
    selected = _validate_folds(folds)
    if gpu < 0:
        raise ValueError("--gpu must be nonnegative")
    cells = {}
    for fold in selected:
        paths = _paths(run_root.resolve(), fold)
        if any(path.exists() for path in paths.values()):
            raise FileExistsError(f"refusing non-fresh Stage-R plan cell fold {fold}: {paths['cell']}")
        cells[f"fold_{fold:02d}"] = {
            "preflight": shlex.join(preflight_command(python=python, paths=paths, run_root=run_root.resolve(), fold=fold)),
            "fit": shlex.join(train_command(python=python, paths=paths, run_root=run_root.resolve(), fold=fold)),
            "outer_eval_cpu_after_selected_fit": (
                "constructed only after target-free selection/config/split validation; "
                "in continue mode it may overlap the next fold's single GPU source fit"
            ),
            "output_root": str(paths["cell"]),
        }
    return {
        "schema": "rt_stage_r_b2_d1024_folds03_14_pipeline_plan_v2",
        "mode": "plan_only_no_execution",
        "folds": list(selected),
        "gpu_for_future_fit_only": gpu,
        "execution_mode_if_continue": "continue_one_level_source_fit_cpu_outer_pipeline_v2",
        "execution_mode_if_stop": "stop_strict_serial_no_speculative_source_fit_v2",
        "overlap_contract": {
            "permitted_overlap": "CPU outer evaluation N with GPU source fit N+1 only",
            "max_concurrent_source_fits": 1,
            "max_concurrent_cpu_outer_evaluators": 1,
            "outer_eval_requires_validated_target_free_selection_config_split": True,
            "outer_eval_N_plus_1_starts_only_after_outer_eval_N_terminalized": True,
            "final_pending_outer_is_explicitly_harvested": True,
        },
        "formal_heldout_endpoint": "absent",
        "cells": cells,
    }


def run_fold(*, run_root: Path, fold: int, python: Path, gpu: int) -> str:
    """Run one complete fold serially.

    This is intentionally retained for ``failure_policy=stop``.  In
    particular, it does not start a later source fit until this fold's outer
    evaluation is terminalized.
    """
    paths = _paths(run_root, fold)
    if paths["terminal"].exists():
        return "skipped_existing_terminal"
    if paths["cell"].exists() and not paths["preflight"].exists():
        _write_once(paths["terminal"], _terminal_payload(
            status="STOP_PARTIAL_CELL_WITHOUT_PREFLIGHT", fold=fold, paths=paths, stage="preflight"
        ))
        return "stopped_partial"
    cpu_env = dict(os.environ)
    cpu_env["CUDA_VISIBLE_DEVICES"] = ""
    gpu_env = dict(os.environ)
    gpu_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gpu_env["PYTHONUNBUFFERED"] = "1"
    try:
        if not paths["preflight"].exists():
            code = _run(preflight_command(python=python, paths=paths, run_root=run_root, fold=fold),
                        cwd=STREAMING_ROOT, env=cpu_env, log_path=paths["log"])
            if code:
                _write_once(paths["terminal"], _terminal_payload(
                    status="PREFLIGHT_FAILED", fold=fold, paths=paths, stage="preflight", code=code
                ))
                return "preflight_failed"
        _validate_preflight(paths["preflight"], fold=fold)
        if paths["fit"].exists() and not paths["selection"].exists():
            _write_once(paths["terminal"], _terminal_payload(
                status="STOP_PARTIAL_FIT_WITHOUT_SELECTION", fold=fold, paths=paths, stage="fit"
            ))
            return "stopped_partial"
        if not paths["selection"].exists():
            code = _run(train_command(python=python, paths=paths, run_root=run_root, fold=fold),
                        cwd=STREAMING_ROOT, env=gpu_env, log_path=paths["log"])
            if code:
                _write_once(paths["terminal"], _terminal_payload(
                    status="FIT_FAILED", fold=fold, paths=paths, stage="fit", code=code
                ))
                return "fit_failed"
        _validate_fit_identity(paths, fold=fold)
        if paths["outer"].exists():
            _write_once(paths["terminal"], _terminal_payload(
                status="STOP_EXISTING_OUTER_RECEIPT_REQUIRES_MANUAL_AUDIT", fold=fold, paths=paths, stage="outer_eval"
            ))
            return "stopped_existing_outer"
        code = _run(eval_command(python=python, paths=paths, fold=fold), cwd=STREAMING_ROOT,
                    env=cpu_env, log_path=paths["log"])
        if code:
            _write_once(paths["terminal"], _terminal_payload(
                status="OUTER_EVAL_FAILED", fold=fold, paths=paths, stage="outer_eval", code=code
            ))
            return "outer_eval_failed"
        _validate_outer(paths, fold=fold)
        _write_once(paths["terminal"], _terminal_payload(
            status="PASS_CPU_ONE_SHOT_OUTER_EVAL", fold=fold, paths=paths, stage="complete"
        ))
        return "passed"
    except BaseException as error:
        if not paths["terminal"].exists():
            _write_once(paths["terminal"], _terminal_payload(
                status="SUPERVISOR_EXCEPTION", fold=fold, paths=paths, stage="exception", error=error
            ))
        return "supervisor_exception"


def _prepare_source_fit(*, run_root: Path, fold: int, python: Path, gpu: int,
                        cpu_env: Mapping[str, str], gpu_env: Mapping[str, str]) -> tuple[str, PreparedOuter | None]:
    """Produce a fold's selected source fit without opening its outer target.

    A ``PreparedOuter`` is an explicit capability: it is returned only after
    the target-free preflight, source fit, selection receipt, config, and split
    have all passed validation.  Pipeline code must not call ``eval_command``
    for any other outcome.
    """

    paths = _paths(run_root, fold)
    if paths["terminal"].exists():
        return "skipped_existing_terminal", None
    if paths["cell"].exists() and not paths["preflight"].exists():
        _write_once(paths["terminal"], _terminal_payload(
            status="STOP_PARTIAL_CELL_WITHOUT_PREFLIGHT", fold=fold, paths=paths, stage="preflight"
        ))
        return "stopped_partial", None
    try:
        if not paths["preflight"].exists():
            code = _run(preflight_command(python=python, paths=paths, run_root=run_root, fold=fold),
                        cwd=STREAMING_ROOT, env=cpu_env, log_path=paths["log"])
            if code:
                _write_once(paths["terminal"], _terminal_payload(
                    status="PREFLIGHT_FAILED", fold=fold, paths=paths, stage="preflight", code=code
                ))
                return "preflight_failed", None
        _validate_preflight(paths["preflight"], fold=fold)
        if paths["fit"].exists() and not paths["selection"].exists():
            _write_once(paths["terminal"], _terminal_payload(
                status="STOP_PARTIAL_FIT_WITHOUT_SELECTION", fold=fold, paths=paths, stage="fit"
            ))
            return "stopped_partial", None
        if not paths["selection"].exists():
            code = _run(train_command(python=python, paths=paths, run_root=run_root, fold=fold),
                        cwd=STREAMING_ROOT, env=gpu_env, log_path=paths["log"])
            if code:
                _write_once(paths["terminal"], _terminal_payload(
                    status="FIT_FAILED", fold=fold, paths=paths, stage="fit", code=code
                ))
                return "fit_failed", None
        _validate_fit_identity(paths, fold=fold)
        if paths["outer"].exists():
            _write_once(paths["terminal"], _terminal_payload(
                status="STOP_EXISTING_OUTER_RECEIPT_REQUIRES_MANUAL_AUDIT", fold=fold, paths=paths, stage="outer_eval"
            ))
            return "stopped_existing_outer", None
        return "source_ready", PreparedOuter(fold=fold, paths=paths)
    except BaseException as error:
        if not paths["terminal"].exists():
            _write_once(paths["terminal"], _terminal_payload(
                status="SUPERVISOR_EXCEPTION", fold=fold, paths=paths, stage="source_audit", error=error
            ))
        return "supervisor_exception", None


def _launch_outer_eval(*, executor: ThreadPoolExecutor, prepared: PreparedOuter,
                       python: Path, cpu_env: Mapping[str, str]) -> PendingOuter:
    """Launch exactly one legal CPU target evaluator in the single-worker pool."""

    command = eval_command(python=python, paths=prepared.paths, fold=prepared.fold)
    future = executor.submit(
        _run, command, cwd=STREAMING_ROOT, env=cpu_env, log_path=prepared.paths["log"]
    )
    return PendingOuter(fold=prepared.fold, paths=prepared.paths, future=future)


def _harvest_outer(pending: PendingOuter) -> str:
    """Wait for a launched evaluator and emit its immutable terminal receipt."""

    try:
        code = int(pending.future.result())
        if code:
            _write_once(pending.paths["terminal"], _terminal_payload(
                status="OUTER_EVAL_FAILED", fold=pending.fold, paths=pending.paths,
                stage="outer_eval", code=code
            ))
            return "outer_eval_failed"
        _validate_outer(pending.paths, fold=pending.fold)
        _write_once(pending.paths["terminal"], _terminal_payload(
            status="PASS_CPU_ONE_SHOT_OUTER_EVAL", fold=pending.fold, paths=pending.paths, stage="complete"
        ))
        return "passed"
    except BaseException as error:
        if not pending.paths["terminal"].exists():
            _write_once(pending.paths["terminal"], _terminal_payload(
                status="SUPERVISOR_EXCEPTION", fold=pending.fold, paths=pending.paths,
                stage="outer_eval", error=error
            ))
        return "supervisor_exception"


def _write_summary(*, run_root: Path, outcomes: Mapping[str, str], failure_policy: str,
                   execution_mode: str) -> None:
    _write_once(run_root / "supervisor_summary.json", {
        "schema": SUMMARY_SCHEMA,
        "program_id": PROGRAM_ID,
        "created_utc": _now(),
        "outcomes": dict(outcomes),
        "failure_policy": failure_policy,
        "execution_mode": execution_mode,
        "pipeline_contract": {
            "max_concurrent_source_fits": 1,
            "max_concurrent_cpu_outer_evaluators": 1,
            "target_evaluator_requires_validated_selection_config_split": True,
            "final_pending_outer_is_harvested": True,
        },
        "formal_heldout_opened": False,
        "supervisor_sha256": _sha256(Path(__file__).resolve()),
    })


def _execute_continue_pipeline(*, run_root: Path, selected: tuple[int, ...], python: Path,
                               gpu: int) -> dict[str, str]:
    """Run the safe one-level GPU-source/CPU-outer pipeline.

    The previous outer is deliberately harvested only *after* the current
    source fit and audit complete.  Therefore CPU evaluation N overlaps GPU
    source work N+1, but evaluation N+1 cannot start before terminal receipt N
    exists.  This is a resource bound and a target-access bound, not merely a
    performance optimization.
    """

    outcomes: dict[str, str] = {}
    cpu_env = dict(os.environ)
    cpu_env["CUDA_VISIBLE_DEVICES"] = ""
    gpu_env = dict(os.environ)
    gpu_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gpu_env["PYTHONUNBUFFERED"] = "1"
    pending: PendingOuter | None = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rt-stage-r-cpu-outer") as executor:
        for fold in selected:
            source_outcome, prepared = _prepare_source_fit(
                run_root=run_root, fold=fold, python=python, gpu=gpu, cpu_env=cpu_env, gpu_env=gpu_env
            )

            # Source work for this fold may overlap the preceding legal outer
            # evaluator.  It must finish before we terminalize and replace it.
            if pending is not None:
                outcomes[f"fold_{pending.fold:02d}"] = _harvest_outer(pending)
                pending = None

            if prepared is None:
                outcomes[f"fold_{fold:02d}"] = source_outcome
                continue

            try:
                pending = _launch_outer_eval(
                    executor=executor, prepared=prepared, python=python, cpu_env=cpu_env
                )
            except BaseException as error:
                if not prepared.paths["terminal"].exists():
                    _write_once(prepared.paths["terminal"], _terminal_payload(
                        status="SUPERVISOR_EXCEPTION", fold=fold, paths=prepared.paths,
                        stage="outer_launch", error=error
                    ))
                outcomes[f"fold_{fold:02d}"] = "supervisor_exception"

        # A final CPU target evaluation cannot be left to executor shutdown:
        # harvest it explicitly so its terminal receipt is visible and audited.
        if pending is not None:
            outcomes[f"fold_{pending.fold:02d}"] = _harvest_outer(pending)
    _write_summary(
        run_root=run_root, outcomes=outcomes, failure_policy="continue",
        execution_mode="continue_one_level_source_fit_cpu_outer_pipeline_v2",
    )
    return outcomes


def execute(*, run_root: Path, folds: Iterable[int], python: Path, gpu: int, failure_policy: str) -> dict[str, str]:
    _require_program(python)
    selected = _validate_folds(folds)
    if failure_policy not in {"continue", "stop"}:
        raise ValueError("failure_policy must be continue or stop")
    run_root = run_root.resolve()
    if failure_policy == "continue":
        return _execute_continue_pipeline(run_root=run_root, selected=selected, python=python, gpu=gpu)

    # `stop` is deliberately serial: a failed source or outer stage must
    # prevent the next source fit from being started speculatively.
    outcomes: dict[str, str] = {}
    for fold in selected:
        outcome = run_fold(run_root=run_root, fold=fold, python=python, gpu=gpu)
        outcomes[f"fold_{fold:02d}"] = outcome
        if outcome != "passed":
            break
    _write_summary(
        run_root=run_root, outcomes=outcomes, failure_policy="stop",
        execution_mode="stop_strict_serial_no_speculative_source_fit_v2",
    )
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(3, 15)))
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--failure-policy", choices=("continue", "stop"), default="continue")
    parser.add_argument("--execute", action="store_true", help="required to run any fit/evaluation; omitted is plan-only")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(plan(run_root=args.run_root, folds=args.folds, python=args.python, gpu=args.gpu), indent=2, sort_keys=True))
        return 0
    outcomes = execute(run_root=args.run_root, folds=args.folds, python=args.python, gpu=args.gpu,
                       failure_policy=args.failure_policy)
    print(json.dumps(outcomes, indent=2, sort_keys=True))
    return 0 if all(value == "passed" for value in outcomes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
