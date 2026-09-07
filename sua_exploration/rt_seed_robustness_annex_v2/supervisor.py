"""Two-lane RT annex v2.1 wave supervisor.

The supervisor owns one paired ``(seed, fold)`` wave.  It has deterministic
cell directories, runs Full and MB4 with identical source/model overrides,
then consumes only the selection receipt emitted by that run to locate the
checkpoint, config, and split manifest.  It never chooses by mtime, score, or
an ad-hoc glob.  The default mode is a static dry-run; execution requires the
same explicit environment gate as :mod:`launcher`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import uuid
from typing import Any, Mapping, Sequence

from . import launcher
from . import spec


class SupervisorError(RuntimeError):
    """Raised when a wave cannot be completed without guessing provenance."""


GPU_LANES = {spec.FULL_ARM: "0", spec.MB4_ARM: "1"}


def _write_exclusive(path: Path, payload: Mapping[str, Any], *, readonly: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink() or os.path.lexists(str(path)):
        raise SupervisorError(f"refusing to overwrite supervisor receipt: {path}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise SupervisorError(f"refusing to overwrite supervisor receipt: {path}") from error
    finally:
        if temporary.exists():
            temporary.unlink()
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise SupervisorError(f"supervisor receipt is not immutable: {path}")


def _seal_receipt(path: Path, *, label: str) -> None:
    """Seal package-owned callback/finalizer output after it is complete."""
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise SupervisorError(f"{label} is not a regular non-symlink receipt")
    os.chmod(path, 0o444)


def _cell_dirs(artifact_root: str | Path, cell: launcher.Cell) -> dict[str, Path]:
    cell_dir = launcher._cell_dir(artifact_root, cell)
    fit_dir = launcher._fit_dir(artifact_root, cell)
    return {
        "cell_dir": cell_dir,
        "fit_dir": fit_dir,
        "source_initial": cell_dir / "source_initial_state.json",
        "selection": fit_dir / "rt_nested_selection_receipt.json",
        "split_manifest": fit_dir / "split_manifest.json",
        "resolved_config": fit_dir / ".hydra" / "config.yaml",
        "outer_eval": cell_dir / "outer_target_eval.json",
        "cell_receipt": cell_dir / "cell_receipt.json",
    }


def build_wave_plan(
    *,
    seed: int,
    fold: int,
    artifact_root: str | Path,
    python_executable: str = str(launcher.DEFAULT_PYTHON),
    snapshot: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic two-lane plan without reading data or running code."""

    root = launcher.validate_artifact_root(artifact_root, require_fresh=True)
    interpreter = launcher.interpreter_metadata(str(python_executable))
    snapshot = dict(snapshot or launcher.implementation_snapshot())
    cells: list[dict[str, Any]] = []
    for arm in (spec.FULL_ARM, spec.MB4_ARM):
        cell = launcher.Cell(seed=seed, fold=fold, arm=arm)
        paths = _cell_dirs(root, cell)
        command = launcher.build_train_command(
            cell,
            artifact_root=root,
            accelerator="gpu",
            devices=1,
            python_executable=str(python_executable),
            interpreter=interpreter,
            snapshot=snapshot,
        )
        lane = GPU_LANES[arm]
        cells.append(
            {
                "seed": cell.seed,
                "fold": cell.fold,
                "arm": cell.arm,
                "run_id": cell.run_id,
                "source_fit_command": command,
                "gpu_lane": lane,
                "cuda_visible_devices": lane,
                "source_fit_command_shell": launcher.render_execution_wrapper(
                    command, cuda_visible_devices=lane
                ),
                "execution_cwd": str(launcher.STREAMING_ROOT.resolve()),
                "source_initial_receipt": str(paths["source_initial"]),
                "fit_dir": str(paths["fit_dir"]),
                "selection_receipt": str(paths["selection"]),
                "split_manifest": str(paths["split_manifest"]),
                "resolved_config": str(paths["resolved_config"]),
                "outer_eval_receipt": str(paths["outer_eval"]),
                "cell_receipt": str(paths["cell_receipt"]),
                "checkpoint_source": "selection_receipt.best_model_path_only",
                "score_based_path_selection": False,
                "mtime_or_glob_path_selection": False,
            }
        )
    return {
        "schema": "rt_seed_robustness_annex_v2_1_wave_plan_v1",
        "status": "DRY_RUN_PREPARED_NOT_AUTHORIZED",
        "development_only": True,
        "gpu_authorized": False,
        "gpu_launched": False,
        "training_started": False,
        "nwb_read": False,
        "formal_heldout_opened": False,
        "seed": int(seed),
        "fold": int(fold),
        "arms": [spec.FULL_ARM, spec.MB4_ARM],
        "paired_wave": True,
        "artifact_root": str(root),
        "artifact_root_absolute": True,
        "artifact_root_fresh_absent_at_plan_build": True,
        "interpreter": interpreter,
        "implementation_snapshot": snapshot,
        "implementation_snapshot_sha256": launcher.implementation_snapshot_sha256(snapshot),
        "execution_environment": {
            "cwd": str(launcher.STREAMING_ROOT.resolve()),
            "pythonpath": launcher.execution_environment()["PYTHONPATH"],
            "python_no_user_site": launcher.execution_environment()["PYTHONNOUSERSITE"],
        },
        "failure_policy": {
            "any_source_failure": "stop_wave_before_target_eval",
            "pair_hash_or_accounting_mismatch": "stop_wave_no_aggregate",
            "selection_or_checkpoint_provenance_failure": "stop_wave_no_target_eval",
            "target_eval_or_finalize_failure": "stop_wave_no_aggregate",
            "no_mtime_or_score_path_guessing": True,
            "execution_requires_cli_flag": "--execute",
            "execution_enable_environment": f"{launcher.EXECUTION_ENABLE_ENV}=1",
            "gpu_authorization_environment": (
                f"{launcher.EXECUTION_AUTHORIZATION_ENV}="
                f"{launcher.EXECUTION_AUTHORIZATION_VALUE}"
            ),
        },
        "gpu_lanes": {
            "independent": True,
            "mapping": dict(GPU_LANES),
            "isolation": "one subprocess per CUDA_VISIBLE_DEVICES lane; trainer.devices=1",
        },
        "cells": cells,
    }


def _run_subprocess(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    launcher.require_execution_gate()
    return subprocess.run(
        list(command),
        cwd=launcher.STREAMING_ROOT,
        env=launcher.execution_environment(cuda_visible_devices=""),
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_selection_receipt(cell: launcher.Cell, paths: Mapping[str, Path]) -> dict[str, Any]:
    if not paths["selection"].is_file():
        raise SupervisorError(f"missing selection receipt from this wave: {paths['selection']}")
    selection = launcher._json_load(paths["selection"])
    if selection.get("schema") != "rt_clean_nested_loso_selection_receipt_v1":
        raise SupervisorError("selection receipt schema mismatch")
    if selection.get("status") != "PASS_FIT_INNER_SELECTION_ONLY":
        raise SupervisorError("selection receipt did not pass inner-only gate")
    if selection.get("arm") != cell.arm or int(selection.get("seed", -1)) != cell.seed:
        raise SupervisorError("selection receipt arm/seed mismatch")
    if int(selection.get("outer_loso_fold", -1)) != cell.fold:
        raise SupervisorError("selection receipt fold mismatch")
    if selection.get("run_id") != cell.run_id:
        raise SupervisorError("selection receipt run_id mismatch")
    if selection.get("selected_by_metric") != "val_heldin/r2_mean":
        raise SupervisorError("selection receipt metric mismatch")
    if selection.get("selected_metric_scope") != "inner_validation_session_only":
        raise SupervisorError("selection receipt metric is not inner-validation only")
    for field in (
        "formal_heldout_opened",
        "outer_target_loaded_during_fit",
        "outer_target_query_labels_read_during_fit",
    ):
        if selection.get(field) is not False:
            raise SupervisorError(f"selection receipt target guard failed: {field}")
    expected_paths = {
        "selection_receipt_path": paths["selection"].resolve(),
        "run_dir": paths["fit_dir"].resolve(),
        "split_manifest_path": paths["split_manifest"].resolve(),
        "config_path": paths["resolved_config"].resolve(),
    }
    for field, expected in expected_paths.items():
        recorded = Path(str(selection.get(field, ""))).resolve()
        if recorded != expected:
            raise SupervisorError(f"selection {field} is not the deterministic wave path")
    if launcher.sha256_file(paths["split_manifest"]) != selection.get("split_manifest_sha256"):
        raise SupervisorError("selection split-manifest hash mismatch")
    if launcher.sha256_file(paths["resolved_config"]) != selection.get("config_sha256"):
        raise SupervisorError("selection resolved-config hash mismatch")
    selected_epoch = int(selection.get("selected_epoch", -1))
    if selected_epoch < 0:
        raise SupervisorError("selection receipt has invalid selected_epoch")
    checkpoint = Path(str(selection.get("best_model_path", ""))).resolve()
    expected_checkpoint = (
        paths["fit_dir"].resolve()
        / "checkpoints"
        / "best_ckpt"
        / f"epoch_{selected_epoch:03d}.ckpt"
    )
    if checkpoint != expected_checkpoint or not checkpoint.is_file():
        raise SupervisorError(
            "selection checkpoint is not the deterministic selected-epoch path"
        )
    if re.fullmatch(r"epoch_[0-9]{3}\.ckpt", checkpoint.name) is None:
        raise SupervisorError("selection checkpoint filename is not deterministic")
    if not launcher.sha256_file(checkpoint) == selection.get("best_model_sha256"):
        raise SupervisorError("selected checkpoint hash does not match selection receipt")
    return selection


def run_wave(plan: Mapping[str, Any], *, receipt_path: str | Path) -> dict[str, Any]:
    """Execute one wave only after explicit authorization; stop on any error."""

    # This gate is deliberately before validation that could create a receipt,
    # and most importantly before the first Popen.
    launcher.require_execution_gate()
    if plan.get("status") != "DRY_RUN_PREPARED_NOT_AUTHORIZED":
        raise SupervisorError("wave plan schema/status is not a static plan")
    if plan.get("gpu_authorized") is not False or plan.get("gpu_launched") is not False:
        raise SupervisorError("wave plan must be the sealed non-authorizing static form")
    if plan.get("interpreter", {}).get("path") != str(launcher.DEFAULT_PYTHON):
        raise SupervisorError("wave plan does not use the pinned interpreter")
    artifact_root = launcher.validate_artifact_root(
        str(plan.get("artifact_root", "")), require_fresh=True
    )
    receipt = Path(receipt_path).resolve(strict=False)
    if artifact_root not in receipt.parents:
        raise SupervisorError(
            "authorized wave receipt must be written inside its external artifact root"
        )
    expected_snapshot = plan.get("implementation_snapshot")
    if not isinstance(expected_snapshot, Mapping):
        raise SupervisorError("wave plan lacks an implementation snapshot")
    if plan.get("implementation_snapshot_sha256") != launcher.implementation_snapshot_sha256(expected_snapshot):
        raise SupervisorError("wave plan implementation snapshot digest mismatch")
    launcher.assert_implementation_snapshot(expected_snapshot)
    cells = list(plan.get("cells", []))
    if len(cells) != 2 or {row.get("arm") for row in cells} != set(launcher.ALLOWED_ARMS):
        raise SupervisorError("wave plan is not exactly Full+MB4")
    for row in cells:
        cell = launcher.Cell(
            seed=int(row.get("seed", -1)),
            fold=int(row.get("fold", -1)),
            arm=str(row.get("arm", "")),
        )
        if cell.seed != int(plan.get("seed", -1)) or cell.fold != int(plan.get("fold", -1)):
            raise SupervisorError("wave cell does not match plan seed/fold")
        expected_paths = _cell_dirs(artifact_root, cell)
        path_fields = {
            "source_initial_receipt": "source_initial",
            "fit_dir": "fit_dir",
            "selection_receipt": "selection",
            "split_manifest": "split_manifest",
            "resolved_config": "resolved_config",
            "outer_eval_receipt": "outer_eval",
            "cell_receipt": "cell_receipt",
        }
        for row_field, path_field in path_fields.items():
            if Path(str(row.get(row_field, ""))).resolve() != expected_paths[path_field].resolve():
                raise SupervisorError(f"wave cell path drift: {row_field}")
        expected_command = launcher.build_train_command(
            cell,
            artifact_root=artifact_root,
            accelerator="gpu",
            devices=1,
            python_executable=str(launcher.DEFAULT_PYTHON),
            interpreter=plan["interpreter"],
            snapshot=expected_snapshot,
        )
        if list(row.get("source_fit_command", [])) != expected_command:
            raise SupervisorError("wave source-fit command drift")
        if row.get("checkpoint_source") != "selection_receipt.best_model_path_only":
            raise SupervisorError("wave checkpoint source is not selection-receipt-only")
        if row.get("score_based_path_selection") is not False or row.get("mtime_or_glob_path_selection") is not False:
            raise SupervisorError("wave plan permits score/mtime/glob path selection")
    subprocess_logs: list[dict[str, Any]] = []
    try:
        # The two lanes are started as one wave, but each output directory is
        # deterministic via hydra.run.dir in its command.  No mtime scan is
        # used after the processes return.
        processes: list[tuple[Mapping[str, Any], subprocess.Popen[str]]] = []
        for row in cells:
            command = [str(value) for value in row["source_fit_command"]]
            lane = str(row.get("cuda_visible_devices", ""))
            if lane != GPU_LANES.get(str(row.get("arm"))):
                raise SupervisorError("wave cell GPU lane mapping mismatch")
            process = subprocess.Popen(
                command,
                cwd=launcher.STREAMING_ROOT,
                env=launcher.execution_environment(cuda_visible_devices=lane),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append((row, process))
        for row, process in processes:
            stdout, stderr = process.communicate()
            subprocess_logs.append(
                {
                    "phase": "source_fit",
                    "arm": row["arm"],
                    "returncode": int(process.returncode),
                    "stdout_tail": stdout[-2000:],
                    "stderr_tail": stderr[-2000:],
                }
            )
            if process.returncode != 0:
                for _, other in processes:
                    if other.poll() is None:
                        other.terminate()
                raise SupervisorError(f"source fit failed for {row['arm']}")

        selection_rows: dict[str, dict[str, Any]] = {}
        for row in cells:
            cell = launcher.Cell(seed=int(row["seed"]), fold=int(row["fold"]), arm=str(row["arm"]))
            # Use the paths bound by the plan, not a newly inferred root.
            paths = {
                "fit_dir": Path(row["fit_dir"]),
                "selection": Path(row["selection_receipt"]),
                "split_manifest": Path(row["split_manifest"]),
                "resolved_config": Path(row["resolved_config"]),
                "source_initial": Path(row["source_initial_receipt"]),
                "outer_eval": Path(row["outer_eval_receipt"]),
                "cell_receipt": Path(row["cell_receipt"]),
            }
            selection_rows[row["arm"]] = _assert_selection_receipt(cell, paths)

        # Pair validation consumes only the source receipts generated by this
        # wave.  It cannot use a score or a prior checkpoint.
        full_row = next(row for row in cells if row["arm"] == spec.FULL_ARM)
        mb4_row = next(row for row in cells if row["arm"] == spec.MB4_ARM)
        pair_output = Path(full_row["fit_dir"]).parents[1] / f"s{plan['seed']}_f{plan['fold']}_paired_initial_state.json"
        launcher.pair_initial_state_receipts(
            cell_seed=int(plan["seed"]),
            cell_fold=int(plan["fold"]),
            full_receipt=full_row["source_initial_receipt"],
            mb4_receipt=mb4_row["source_initial_receipt"],
            output=pair_output,
        )

        # Target evaluation is driven by the exact checkpoint/config/split
        # paths recorded in each selection receipt.
        for row in cells:
            cell = launcher.Cell(seed=int(row["seed"]), fold=int(row["fold"]), arm=str(row["arm"]))
            selection = selection_rows[row["arm"]]
            eval_command = launcher.build_eval_command(
                cell,
                config=selection["config_path"],
                checkpoint=selection["best_model_path"],
                split_manifest=selection["split_manifest_path"],
                selection_receipt=row["selection_receipt"],
                output=row["outer_eval_receipt"],
                device="cpu",
                python_executable=str(plan["interpreter"]["path"]),
                interpreter=plan["interpreter"],
            )
            completed = _run_subprocess(eval_command)
            subprocess_logs.append(
                {
                    "phase": "outer_eval",
                    "arm": row["arm"],
                    "returncode": int(completed.returncode),
                    "stdout_tail": completed.stdout[-2000:],
                    "stderr_tail": completed.stderr[-2000:],
                }
            )
            if completed.returncode != 0:
                raise SupervisorError(f"outer evaluation failed for {row['arm']}")
            paths = {
                "fit_dir": Path(row["fit_dir"]),
                "selection": Path(row["selection_receipt"]),
                "split_manifest": Path(row["split_manifest"]),
                "resolved_config": Path(row["resolved_config"]),
                "source_initial": Path(row["source_initial_receipt"]),
                "outer_eval": Path(row["outer_eval_receipt"]),
                "cell_receipt": Path(row["cell_receipt"]),
            }
            launcher.build_cell_receipt(
                cell,
                source_initial_receipt=paths["source_initial"],
                outer_eval_receipt=paths["outer_eval"],
                selection_receipt=paths["selection"],
                split_manifest=paths["split_manifest"],
                paired_initial_receipt=pair_output,
                output=paths["cell_receipt"],
            )
            # The callback and shared finalizer retain their own writer
            # contracts; sealing happens only after this wave has consumed
            # their completed outputs, so no shared training behavior changes.
            for label, receipt_path in {
                "source-initial": paths["source_initial"],
                "selection": paths["selection"],
                "split": paths["split_manifest"],
                "outer-target": paths["outer_eval"],
                "cell": paths["cell_receipt"],
            }.items():
                _seal_receipt(receipt_path, label=label)
        _seal_receipt(pair_output, label="paired-initial-state")
    except Exception as error:
        failure = {
            "schema": "rt_seed_robustness_annex_v2_1_wave_receipt_v1",
            "status": "STOP_WAVE_FAILURE_NO_AGGREGATE",
            "development_only": True,
            "gpu_authorized": True,
            "gpu_launched": bool(subprocess_logs),
            "formal_heldout_opened": False,
            "seed": int(plan["seed"]),
            "fold": int(plan["fold"]),
            "error": f"{type(error).__name__}: {error}",
            "subprocess_logs": subprocess_logs,
        }
        _write_exclusive(receipt, failure)
        raise
    success = {
        "schema": "rt_seed_robustness_annex_v2_1_wave_receipt_v1",
        "status": "PASS_WAVE_COMPLETE_EXPLICITLY_AUTHORIZED",
        "development_only": True,
        "gpu_authorized": True,
        "gpu_launched": True,
        "formal_heldout_opened": False,
        "seed": int(plan["seed"]),
        "fold": int(plan["fold"]),
        "arms": [spec.FULL_ARM, spec.MB4_ARM],
        "subprocess_logs": subprocess_logs,
        "aggregate_allowed": True,
        "artifact_root": str(artifact_root),
    }
    _write_exclusive(receipt, success)
    return success


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = build_wave_plan(seed=args.seed, fold=args.fold, artifact_root=args.artifact_root)
    if not args.execute:
        output = args.output.resolve(strict=False)
        artifact_root = Path(plan["artifact_root"])
        if output == artifact_root or artifact_root in output.parents:
            raise SupervisorError(
                "dry-run receipt must remain outside the fresh artifact root"
            )
        _write_exclusive(args.output, plan)
        print(json.dumps({"status": plan["status"], "output": str(args.output)}, sort_keys=True))
        return 0
    result = run_wave(plan, receipt_path=args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
