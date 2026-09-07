#!/usr/bin/env python3
"""Run only missing clean RT nested-LOSO cells in explicit cell directories.

This is a deployment supervisor, not a new estimator.  It uses the existing
``run_rt_clean_nested_loso.py`` command construction for the inner-only fit
and the existing one-shot evaluator for the outer target.  Each cell has an
independent Hydra run directory so selection receipts, checkpoints, and outer
evaluation receipts can never be commingled across arm/fold combinations.

The script deliberately has no option to run a formal held-out endpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import subprocess
import sys
import traceback
from typing import Any

# Running this file as ``python scripts/...`` puts ``scripts/`` on
# ``sys.path``; importing it in a smoke test puts the project root there.
# Support both entry modes without relying on package installation.
try:
    from scripts.run_rt_clean_nested_loso import _eval_command, _train_command
except ModuleNotFoundError:  # pragma: no cover - exercised by direct script mode
    from run_rt_clean_nested_loso import _eval_command, _train_command


SUPPORTED_ARMS = ("afc4_rs", "afc4_ls")
PASS_STATUS = "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP"


def _json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _completed_outer_receipt(path: Path, *, arm: str, fold: int, seed: int) -> bool:
    value = _json_object(path)
    return bool(
        value
        and value.get("schema") == "rt_clean_nested_loso_outer_eval_v1"
        and value.get("status") == PASS_STATUS
        and value.get("arm") == arm
        and int(value.get("outer_loso_fold", -1)) == fold
        and int(value.get("seed", -1)) == seed
    )


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    """Write a cell terminal state without replacing a prior record."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite supervisor terminal state: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _selected_checkpoint(receipt: Path) -> Path:
    value = _json_object(receipt)
    if not value or value.get("status") != "PASS_FIT_INNER_SELECTION_ONLY":
        raise RuntimeError(f"No passing clean selection receipt at {receipt}")
    checkpoint = Path(str(value.get("best_model_path", ""))).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selection receipt points to missing checkpoint: {checkpoint}")
    return checkpoint


def _command_text(command: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(item) for item in command)


def _run(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("$ " + _command_text(command) + "\n")
        stream.flush()
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        stream.write(f"[exit={completed.returncode}]\n")
        stream.flush()
    return int(completed.returncode)


def run_cell(*, project: Path, run_root: Path, arm: str, fold: int, seed: int, gpu: int) -> str:
    cell = run_root / arm / f"fold_{fold:02d}" / f"seed_{seed}"
    fit_dir = cell / "fit"
    receipt = fit_dir / "rt_nested_selection_receipt.json"
    config = fit_dir / ".hydra" / "config.yaml"
    split_manifest = fit_dir / "split_manifest.json"
    outer = cell / "outer_target_eval.json"
    terminal = cell / "cell_terminal.json"
    log = cell / "supervisor.log"

    if _completed_outer_receipt(outer, arm=arm, fold=fold, seed=seed):
        print(f"SKIP complete {arm} fold={fold} seed={seed}: {outer}", flush=True)
        return "skipped_complete"
    if outer.exists():
        raise RuntimeError(f"Outer receipt exists but is not a matching valid completion: {outer}")
    if terminal.exists():
        raise RuntimeError(f"Terminal state already exists; inspect rather than overwrite: {terminal}")
    if fit_dir.exists() and not receipt.exists():
        raise RuntimeError(f"Partial fit directory without selection receipt: {fit_dir}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")
    train_args = argparse.Namespace(fold=fold, arm=arm, seed=seed, accelerator="gpu", devices=1)
    if not receipt.exists():
        command = _train_command(train_args)
        command.extend(
            [
                f"hydra.run.dir={fit_dir}",
                f"paths.root_dir={project}",
                f"paths.log_dir={run_root / '_hydra_logs'}",
                f"paths.artifact_dir={run_root / '_artifacts'}",
            ]
        )
        print(f"START fit {arm} fold={fold} seed={seed} dir={fit_dir}", flush=True)
        code = _run(command, cwd=project, env=env, log_path=log)
        if code:
            _write_new_json(
                terminal,
                {"status": "FIT_FAILED", "arm": arm, "fold": fold, "seed": seed, "exit_code": code},
            )
            return "fit_failed"
    checkpoint = _selected_checkpoint(receipt)
    if not config.is_file() or not split_manifest.is_file():
        raise FileNotFoundError("Passing selection receipt lacks config or split manifest")
    eval_args = argparse.Namespace(
        config=config,
        checkpoint=checkpoint,
        split_manifest=split_manifest,
        selection_receipt=receipt,
        output=outer,
        outer_fold=fold,
        device="cuda",
    )
    command = _eval_command(eval_args)
    print(f"START outer-eval {arm} fold={fold} seed={seed} checkpoint={checkpoint.name}", flush=True)
    code = _run(command, cwd=project, env=env, log_path=log)
    if code:
        _write_new_json(
            terminal,
            {"status": "OUTER_EVAL_FAILED", "arm": arm, "fold": fold, "seed": seed, "exit_code": code},
        )
        return "outer_eval_failed"
    if not _completed_outer_receipt(outer, arm=arm, fold=fold, seed=seed):
        _write_new_json(
            terminal,
            {"status": "OUTER_RECEIPT_INVALID", "arm": arm, "fold": fold, "seed": seed},
        )
        return "outer_receipt_invalid"
    _write_new_json(
        terminal,
        {"status": "PASS", "arm": arm, "fold": fold, "seed": seed, "outer_receipt": str(outer)},
    )
    print(f"PASS {arm} fold={fold} seed={seed}: {outer}", flush=True)
    return "passed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--arms", choices=SUPPORTED_ARMS, nargs="+", default=list(SUPPORTED_ARMS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if not (project / "src" / "train.py").is_file():
        raise FileNotFoundError(f"Project root is incomplete: {project}")
    folds = tuple(args.folds)
    if len(set(folds)) != len(folds) or any(fold < 0 or fold >= 15 for fold in folds):
        raise ValueError("--folds must be unique values in [0,14]")
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, str] = {}
    for arm in args.arms:
        for fold in folds:
            key = f"{arm}:fold{fold}:seed{args.seed}"
            try:
                summary[key] = run_cell(
                    project=project,
                    run_root=run_root,
                    arm=arm,
                    fold=fold,
                    seed=int(args.seed),
                    gpu=int(args.gpu),
                )
            except Exception as error:  # Continue to the next independent cell.
                cell = run_root / arm / f"fold_{fold:02d}" / f"seed_{args.seed}"
                terminal = cell / "cell_terminal.json"
                if not terminal.exists():
                    _write_new_json(
                        terminal,
                        {
                            "status": "SUPERVISOR_EXCEPTION",
                            "arm": arm,
                            "fold": fold,
                            "seed": int(args.seed),
                            "error": repr(error),
                            "traceback": traceback.format_exc(),
                        },
                    )
                summary[key] = "supervisor_exception"
                print(f"FAILED {key}: {error!r}", flush=True)
    summary_path = run_root / "supervisor_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {summary_path}")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if all(status in {"passed", "skipped_complete"} for status in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
