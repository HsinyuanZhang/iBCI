#!/usr/bin/env python3
"""Run missing RT clean nested-LOSO RS/LS control cells, and nothing else.

This is a deliberately narrow supervisor for the two remaining RT mechanism
controls:

* ``afc4_rs``: complete carrier-row shuffle;
* ``afc4_ls``: segment-preserving velocity-label-association shuffle.

The sealed Full/B4/Zero result is never rewritten, fold 0 is excluded (its
remote control receipts already exist), and this program has no option for
W4, M6, M12, a formal FALCON held-out evaluation, or any other task.  A cell
is a separate two-phase transaction:

1. ``src/train.py`` fits with ``test=false`` and emits an inner-validation
   checkpoint selection receipt;
2. ``src/rt_clean_nested_loso_eval.py`` validates that receipt/checkpoint/
   config/split binding, then makes one outer-target forward-only evaluation.

All new run directories and copied receipts live under
``sua_exploration/results/k4_rt_loso_v1``.  Successful cells are immutable;
the supervisor skips an already-valid cell, records a failed cell without
terminating its GPU shard, and never overwrites either state.

Example (one arm on one physical GPU)::

    /home/xinyuan/miniconda3/envs/spint/bin/python \
      sua_exploration/scripts/run_rt_clean_nested_missing_supervisor.py \
      --execute --arm afc4_rs --folds 1-10 --gpu 0

The script intentionally starts one cell at a time, so two independent tmux
workers can reserve GPU0 for RS and GPU1 for LS without overlapping a device.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
STREAMING_ROOT = ROOT / "streaming_calibration_exp"
TRAIN = STREAMING_ROOT / "src" / "train.py"
OUTER_EVALUATOR = STREAMING_ROOT / "src" / "rt_clean_nested_loso_eval.py"
DEFAULT_PYTHON = Path("/home/xinyuan/miniconda3/envs/spint/bin/python")
RESULT_ROOT = ROOT / "sua_exploration" / "results" / "k4_rt_loso_v1"
PROGRAM_ID = "rt_clean_nested_rs_ls_local3090_v1"
ALLOWED_ARMS = frozenset({"afc4_rs", "afc4_ls"})
ALLOWED_FOLDS = frozenset(range(1, 15))  # fold 0 has sealed remote receipts
SEED = 42
M = 24


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json_exclusive(path: Path, payload: Mapping[str, Any], *, mode: int = 0o644) -> None:
    """Create a receipt once; a collision means another supervisor owns it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except FileExistsError:
        raise FileExistsError(f"Refusing to overwrite supervisor receipt: {path}") from None
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(serialized)


def _cell_root(program_root: Path, arm: str, fold: int) -> Path:
    return program_root / "cells" / arm / f"fold_{fold:02d}"


def _fit_dir(program_root: Path, arm: str, fold: int) -> Path:
    return _cell_root(program_root, arm, fold) / "fit"


def _completion_path(program_root: Path, arm: str, fold: int) -> Path:
    return _cell_root(program_root, arm, fold) / "cell_completion.json"


def _failure_path(program_root: Path, arm: str, fold: int) -> Path:
    return _cell_root(program_root, arm, fold) / "cell_failure.json"


def _validate_arguments(arm: str, folds: Iterable[int], gpu: int, python_bin: Path) -> tuple[int, ...]:
    if arm not in ALLOWED_ARMS:
        raise ValueError(f"Only RS/LS controls are allowed, got {arm!r}")
    normalized = tuple(dict.fromkeys(int(fold) for fold in folds))
    if not normalized:
        raise ValueError("At least one fold is required")
    invalid = [fold for fold in normalized if fold not in ALLOWED_FOLDS]
    if invalid:
        raise ValueError(
            f"Only missing folds 1..14 are legal (fold 0 is sealed remotely); got {invalid}"
        )
    if gpu < 0:
        raise ValueError("--gpu must be non-negative")
    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        raise FileNotFoundError(f"SPINT Python is not executable: {python_bin}")
    for required in (TRAIN, OUTER_EVALUATOR):
        if not required.is_file():
            raise FileNotFoundError(f"Required RT program is missing: {required}")
    return normalized


def _expect(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {value!r}")


def _validate_completed_cell(program_root: Path, arm: str, fold: int) -> dict[str, Any]:
    """Validate all final-state evidence without opening the target data again."""

    cell = _cell_root(program_root, arm, fold)
    fit = _fit_dir(program_root, arm, fold)
    completion_path = _completion_path(program_root, arm, fold)
    selection_path = fit / "rt_nested_selection_receipt.json"
    split_path = fit / "split_manifest.json"
    config_path = fit / ".hydra" / "config.yaml"
    outer_path = fit / "outer_target_eval.json"
    needed = (completion_path, selection_path, split_path, config_path, outer_path)
    if not all(path.is_file() for path in needed):
        missing = [str(path) for path in needed if not path.is_file()]
        raise FileNotFoundError(f"Cell evidence is incomplete: {missing}")

    selection = _json(selection_path)
    outer = _json(outer_path)
    completion = _json(completion_path)
    _expect(selection.get("schema"), "rt_clean_nested_loso_selection_receipt_v1", "selection schema")
    _expect(selection.get("status"), "PASS_FIT_INNER_SELECTION_ONLY", "selection status")
    _expect(selection.get("arm"), arm, "selection arm")
    _expect(selection.get("outer_loso_fold"), fold, "selection fold")
    _expect(selection.get("seed"), SEED, "selection seed")
    _expect(selection.get("formal_heldout_opened"), False, "selection formal-heldout flag")
    _expect(selection.get("outer_target_loaded_during_fit"), False, "selection target-loaded flag")
    _expect(
        selection.get("outer_target_query_labels_read_during_fit"),
        False,
        "selection target-query-label flag",
    )
    checkpoint = Path(str(selection.get("best_model_path", "")))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint missing: {checkpoint}")
    if _sha256(checkpoint) != selection.get("best_model_sha256"):
        raise ValueError("Selected checkpoint SHA-256 does not match selection receipt")
    for label, path, recorded in (
        ("config", config_path, selection.get("config_sha256")),
        ("split manifest", split_path, selection.get("split_manifest_sha256")),
    ):
        if _sha256(path) != recorded:
            raise ValueError(f"{label} SHA-256 does not match selection receipt")
    if Path(str(selection.get("config_path", ""))).resolve() != config_path.resolve():
        raise ValueError("Selection receipt config path does not bind this cell config")
    if Path(str(selection.get("split_manifest_path", ""))).resolve() != split_path.resolve():
        raise ValueError("Selection receipt split path does not bind this cell manifest")

    _expect(outer.get("schema"), "rt_clean_nested_loso_outer_eval_v1", "outer schema")
    _expect(outer.get("status"), "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP", "outer status")
    _expect(outer.get("arm"), arm, "outer arm")
    _expect(outer.get("outer_loso_fold"), fold, "outer fold")
    _expect(outer.get("seed"), SEED, "outer seed")
    _expect(outer.get("query_start_trial"), M, "outer calibration M")
    _expect(outer.get("target_backpropagation"), False, "outer target backprop")
    _expect(outer.get("optimizer_present"), False, "outer optimizer")
    _expect(outer.get("model_training_mode"), False, "outer model mode")
    _expect(outer.get("model_state_unchanged"), True, "outer model-state flag")
    _expect(
        outer.get("model_state_sha256_before"),
        outer.get("model_state_sha256_after"),
        "outer model-state SHA",
    )
    _expect(outer.get("target_query_labels_used_for_calibration"), False, "outer target calibration labels")
    _expect(outer.get("target_query_labels_used_for_normalization"), False, "outer target normalizer labels")
    _expect(
        outer.get("target_query_labels_used_for_checkpoint_selection"),
        False,
        "outer target checkpoint labels",
    )
    _expect(outer.get("target_query_labels_used_for_scoring_only"), True, "outer target score labels")
    _expect(Path(str(outer.get("checkpoint_path", ""))).resolve(), checkpoint.resolve(), "outer checkpoint path")
    _expect(outer.get("checkpoint_sha256"), selection.get("best_model_sha256"), "outer checkpoint SHA")
    _expect(
        Path(str(outer.get("selection_receipt_path", ""))).resolve(),
        selection_path.resolve(),
        "outer selection path",
    )
    _expect(Path(str(outer.get("config_path", ""))).resolve(), config_path.resolve(), "outer config path")
    _expect(Path(str(outer.get("fit_split_manifest", ""))).resolve(), split_path.resolve(), "outer split path")
    if not isinstance(outer.get("r2_variance_weighted"), (float, int)):
        raise ValueError("Outer RT evaluation has no finite scalar R²")
    if not isinstance(outer.get("query_windows_evaluated"), int) or outer["query_windows_evaluated"] <= 0:
        raise ValueError("Outer RT evaluation did not score positive query-window count")

    _expect(completion.get("schema"), "rt_clean_nested_missing_cell_completion_v1", "completion schema")
    _expect(completion.get("status"), "PASS_COMPLETED_AND_REVALIDATED", "completion status")
    _expect(completion.get("arm"), arm, "completion arm")
    _expect(completion.get("fold"), fold, "completion fold")
    _expect(completion.get("seed"), SEED, "completion seed")
    files = completion.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("Cell completion receipt lacks file hashes")
    expected_hashes = {
        "selection_receipt_sha256": _sha256(selection_path),
        "split_manifest_sha256": _sha256(split_path),
        "config_sha256": _sha256(config_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "outer_target_eval_sha256": _sha256(outer_path),
    }
    for key, expected in expected_hashes.items():
        _expect(files.get(key), expected, f"completion {key}")
    return {
        "status": "already_valid",
        "arm": arm,
        "fold": fold,
        "r2_variance_weighted": float(outer["r2_variance_weighted"]),
        "outer_target_session": outer.get("outer_target_session"),
        "cell_root": str(cell.resolve()),
    }


def _make_command(
    *, python_bin: Path, arm: str, fold: int, fit_dir: Path, artifact_dir: Path
) -> tuple[list[str], list[str]]:
    run_id = f"rt_clean_nested_loso_m24_{arm}_{PROGRAM_ID}"
    train = [
        str(python_bin),
        "-u",
        str(TRAIN),
        "experiment=rt_clean_nested_loso_m24",
        f"run_id={run_id}",
        f"data.loso_fold={fold}",
        f"data.outer_loso_fold={fold}",
        f"data.side_feature_group={arm}",
        f"seed={SEED}",
        "test=false",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        f"hydra.run.dir={fit_dir.resolve()}",
        f"paths.artifact_dir={artifact_dir.resolve()}",
    ]
    # The actual checkpoint filename is bound by the selection receipt only
    # after fit.  Keep an explicit insertion point rather than globbing a
    # checkpoint directory (which could choose a non-selected top-k file).
    evaluate_prefix = [
        str(python_bin),
        "-u",
        str(OUTER_EVALUATOR),
        "--config",
        str((fit_dir / ".hydra" / "config.yaml").resolve()),
        "--checkpoint",
    ]
    evaluate_suffix = [
        "--split-manifest",
        str((fit_dir / "split_manifest.json").resolve()),
        "--selection-receipt",
        str((fit_dir / "rt_nested_selection_receipt.json").resolve()),
        "--output",
        str((fit_dir / "outer_target_eval.json").resolve()),
        "--outer-fold",
        str(fold),
        "--device",
        "cuda",
    ]
    return train, evaluate_prefix + ["__CHECKPOINT_FROM_SELECTION_RECEIPT__"] + evaluate_suffix


def _relevant_failure_exists(program_root: Path, arm: str, fold: int) -> bool:
    return _failure_path(program_root, arm, fold).is_file()


def _immutable(path: Path) -> None:
    if path.is_file():
        path.chmod(0o444)


def _run_cell(
    *,
    program_root: Path,
    python_bin: Path,
    arm: str,
    fold: int,
    gpu: int,
    retry_failed: bool,
) -> dict[str, Any]:
    cell = _cell_root(program_root, arm, fold)
    completion = _completion_path(program_root, arm, fold)
    failure = _failure_path(program_root, arm, fold)
    try:
        return _validate_completed_cell(program_root, arm, fold)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        if completion.exists():
            raise RuntimeError(f"Existing completion receipt is invalid; refusing to overwrite: {completion}")
    if failure.exists() and not retry_failed:
        return {
            "status": "prior_failure_skipped",
            "arm": arm,
            "fold": fold,
            "failure_receipt": str(failure.resolve()),
        }

    fit = _fit_dir(program_root, arm, fold)
    artifact_dir = cell / "fit_artifacts"
    log_path = cell / "worker.log"
    train_command, evaluate_prefix = _make_command(
        python_bin=python_bin, arm=arm, fold=fold, fit_dir=fit, artifact_dir=artifact_dir
    )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["MPLCONFIGDIR"] = str(Path("/tmp") / f"{PROGRAM_ID}_{arm}_f{fold}_gpu{gpu}")
    environment["PYTHONNOUSERSITE"] = "1"
    environment["HYDRA_FULL_ERROR"] = "1"
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    try:
        claim = cell / "cell_claim.json"
        claim.parent.mkdir(parents=True, exist_ok=True)
        _write_json_exclusive(
            claim,
            {
                "schema": "rt_clean_nested_missing_cell_claim_v1",
                "program": PROGRAM_ID,
                "arm": arm,
                "fold": fold,
                "seed": SEED,
                "M": M,
                "gpu": gpu,
                "started_at": _utc_now(),
                "pid": os.getpid(),
                "formal_heldout_permitted": False,
            },
        )
        with log_path.open("x", encoding="utf-8") as log:
            log.write(f"[{_utc_now()}] start arm={arm} fold={fold} seed={SEED} M={M} gpu={gpu}\n")
            log.write("train command: " + json.dumps(train_command) + "\n")
            log.flush()
            train = subprocess.run(
                train_command,
                cwd=STREAMING_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if train.returncode != 0:
                raise RuntimeError(f"fit returned {train.returncode}")
            selection_path = fit / "rt_nested_selection_receipt.json"
            selection = _json(selection_path)
            checkpoint = Path(str(selection.get("best_model_path", "")))
            if not checkpoint.is_file():
                raise FileNotFoundError(f"fit did not produce selected checkpoint: {checkpoint}")
            evaluate = [
                str(checkpoint.resolve()) if part == "__CHECKPOINT_FROM_SELECTION_RECEIPT__" else part
                for part in evaluate_prefix
            ]
            log.write("outer-eval command: " + json.dumps(evaluate) + "\n")
            log.flush()
            outer = subprocess.run(
                evaluate,
                cwd=STREAMING_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if outer.returncode != 0:
                raise RuntimeError(f"outer evaluator returned {outer.returncode}")
            log.write(f"[{_utc_now()}] train_rc={train.returncode} eval_rc={outer.returncode}\n")
        # Write the completion receipt only after re-reading every binding.
        selection_path = fit / "rt_nested_selection_receipt.json"
        split_path = fit / "split_manifest.json"
        config_path = fit / ".hydra" / "config.yaml"
        outer_path = fit / "outer_target_eval.json"
        selection = _json(selection_path)
        checkpoint = Path(str(selection["best_model_path"])).resolve()
        _write_json_exclusive(
            completion,
            {
                "schema": "rt_clean_nested_missing_cell_completion_v1",
                "status": "PASS_COMPLETED_AND_REVALIDATED",
                "program": PROGRAM_ID,
                "arm": arm,
                "fold": fold,
                "seed": SEED,
                "M": M,
                "gpu": gpu,
                "completed_at": _utc_now(),
                "formal_heldout_opened": False,
                "fit_test_flag": False,
                "outer_target_eval_is_one_shot": True,
                "validation": {
                    "checkpoint_config_split_selection_bound": True,
                    "target_backpropagation": False,
                    "optimizer_present": False,
                    "model_state_unchanged": True,
                },
                "files": {
                    "selection_receipt": str(selection_path.resolve()),
                    "selection_receipt_sha256": _sha256(selection_path),
                    "split_manifest": str(split_path.resolve()),
                    "split_manifest_sha256": _sha256(split_path),
                    "config": str(config_path.resolve()),
                    "config_sha256": _sha256(config_path),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256(checkpoint),
                    "outer_target_eval": str(outer_path.resolve()),
                    "outer_target_eval_sha256": _sha256(outer_path),
                },
            },
        )
        # Verify the just-written receipt before freezing the final evidence.
        result = _validate_completed_cell(program_root, arm, fold)
        for evidence in (selection_path, split_path, config_path, checkpoint, outer_path, completion):
            _immutable(evidence)
        return {**result, "status": "completed"}
    except BaseException as error:  # log each failure, then permit later folds
        error_payload = {
            "schema": "rt_clean_nested_missing_cell_failure_v1",
            "status": "FAILED_CELL_CONTINUING_OTHER_FOLDS",
            "program": PROGRAM_ID,
            "arm": arm,
            "fold": fold,
            "seed": SEED,
            "M": M,
            "gpu": gpu,
            "failed_at": _utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "log_path": str(log_path.resolve()),
            "formal_heldout_opened": False,
            "traceback": traceback.format_exc(),
        }
        if failure.exists():
            error_payload["failure_receipt_write"] = "preexisting_failure_preserved"
        else:
            _write_json_exclusive(failure, error_payload)
        return {
            "status": "failed",
            "arm": arm,
            "fold": fold,
            "error": f"{type(error).__name__}: {error}",
            "failure_receipt": str(failure.resolve()),
        }


def _manifest(program_root: Path, *, arm: str, folds: tuple[int, ...], gpu: int, python_bin: Path) -> None:
    path = program_root / "launch_manifest.json"
    payload = {
        "schema": "rt_clean_nested_missing_supervisor_manifest_v1",
        "program": PROGRAM_ID,
        "status": "RUNNING_OR_COMPLETED",
        "created_at": _utc_now(),
        "task": "RT",
        "validation_protocol": "development_clean_nested_outer_LOSO",
        "arms_permitted": sorted(ALLOWED_ARMS),
        "arm": arm,
        "folds": list(folds),
        "seed": SEED,
        "calibration_trials": M,
        "gpu": gpu,
        "python": str(python_bin.resolve()),
        "fit_command_policy": {
            "experiment": "rt_clean_nested_loso_m24",
            "test": False,
            "selection_metric": "val_heldin/r2_mean",
            "outer_target_loaded_during_fit": False,
            "formal_heldout_opened": False,
        },
        "outer_eval_policy": {
            "entry": str(OUTER_EVALUATOR.resolve()),
            "one_shot": True,
            "target_backpropagation": False,
            "optimizer_present": False,
            "state_digest_must_be_unchanged": True,
        },
    }
    if path.exists():
        existing = _json(path)
        stable = {key: existing.get(key) for key in ("program", "task", "validation_protocol", "seed", "calibration_trials")}
        expected = {key: payload.get(key) for key in stable}
        if stable != expected:
            raise RuntimeError("Existing RT missing-control manifest binds a different program")
        return
    _write_json_exclusive(path, payload)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="required to touch a GPU")
    parser.add_argument("--arm", choices=sorted(ALLOWED_ARMS), required=True)
    parser.add_argument("--folds", required=True, help="comma/range list, e.g. 1-10 or 1,3,5")
    parser.add_argument("--gpu", type=int, required=True, help="physical CUDA GPU index")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--program-root", type=Path, default=RESULT_ROOT / PROGRAM_ID)
    parser.add_argument("--retry-failed", action="store_true", help="explicitly retry a prior failed cell")
    args = parser.parse_args(list(argv) if argv is not None else None)

    folds: list[int] = []
    for item in str(args.folds).split(","):
        if not item:
            continue
        if "-" in item:
            start, end = (int(value) for value in item.split("-", 1))
            if end < start:
                raise ValueError(f"Descending fold range is invalid: {item}")
            folds.extend(range(start, end + 1))
        else:
            folds.append(int(item))
    normalized_folds = _validate_arguments(args.arm, folds, args.gpu, args.python)
    program_root = args.program_root.resolve()
    dry_run = {
        "schema": "rt_clean_nested_missing_supervisor_dry_run_v1",
        "program": PROGRAM_ID,
        "task": "RT",
        "arm": args.arm,
        "folds": list(normalized_folds),
        "seed": SEED,
        "M": M,
        "gpu": args.gpu,
        "python": str(args.python.resolve()),
        "program_root": str(program_root),
        "formal_heldout_opened": False,
        "formal_heldout_permitted": False,
        "allowed_arms": sorted(ALLOWED_ARMS),
        "allowed_folds": sorted(ALLOWED_FOLDS),
    }
    if not args.execute:
        print(json.dumps(dry_run, indent=2, sort_keys=True))
        return 0

    program_root.mkdir(parents=True, exist_ok=True)
    _manifest(program_root, arm=args.arm, folds=normalized_folds, gpu=args.gpu, python_bin=args.python)
    results = []
    for fold in normalized_folds:
        result = _run_cell(
            program_root=program_root,
            python_bin=args.python,
            arm=args.arm,
            fold=fold,
            gpu=args.gpu,
            retry_failed=args.retry_failed,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    summary = {
        "schema": "rt_clean_nested_missing_supervisor_shard_summary_v1",
        "program": PROGRAM_ID,
        "completed_at": _utc_now(),
        "arm": args.arm,
        "folds": list(normalized_folds),
        "gpu": args.gpu,
        "seed": SEED,
        "M": M,
        "formal_heldout_opened": False,
        "counts": {
            "completed": sum(row.get("status") == "completed" for row in results),
            "already_valid": sum(row.get("status") == "already_valid" for row in results),
            "prior_failure_skipped": sum(row.get("status") == "prior_failure_skipped" for row in results),
            "failed": sum(row.get("status") == "failed" for row in results),
        },
        "cells": results,
    }
    summary_path = program_root / "shards" / f"{args.arm}_f{normalized_folds[0]:02d}_f{normalized_folds[-1]:02d}_gpu{args.gpu}.json"
    _write_json_exclusive(summary_path, summary)
    return 0 if summary["counts"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
