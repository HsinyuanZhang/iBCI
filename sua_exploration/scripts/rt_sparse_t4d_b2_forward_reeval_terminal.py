#!/usr/bin/env python3
"""Terminal launcher plan and finalizer for uniform RT B2-D1024 re-evaluation.

The only allowed scoring operation is inference with the pre-existing selected
B2 checkpoint. This module does not retrain, select checkpoints, create an
optimizer, or overwrite either an old receipt or a new forward-only receipt.
``prepare`` is intentionally print-only. ``finalize`` refuses results unless
the Stage-2 45/45 terminal verifier has already passed.
"""
from __future__ import annotations

import argparse
import math as _math
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "sua_exploration"
COMPANION_PATH = PROJECT / "scripts/rt_sparse_t4d_vs_b2_d1024_companion.py"
STAGE2_PATH = PROJECT / "scripts/rt_sparse_endpoint_stage2_terminal_verify.py"
EVALUATOR = ROOT / "streaming_calibration_exp/src/rt_clean_nested_loso_eval.py"
FOLD0_IMPORT = ROOT / "sua_exploration/results/rt_sparse_t4d_b2_forward_reeval_v1/imported_fold0_legacy_v1/FOLD0_IMPORT_RECEIPT_v1.json"
FOLD0_REMOTE_ROOT = "/home/xinyuan/Work_host/SPINT/rt_sparse_t4d_b2_forward_reeval_v1"
FOLD0_HOST = "xinyuan@100.103.97.12"
EVALUATOR_CWD = ROOT / "streaming_calibration_exp"
REMOTE_EVALUATOR_CWD = "/home/xinyuan/Work_host/SPINT/rt_sparse_endpoint_stage2_5070_v1_20260810/streaming_calibration_exp"
FOLDS = tuple(range(15))


class TerminalError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    _need(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _immutable_json(path: Path) -> dict[str, Any]:
    _need(path.is_file(), f"missing immutable evidence: {path}")
    _need(stat.S_IMODE(path.stat().st_mode) == 0o444, f"evidence must be mode 0444: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _need(isinstance(value, dict), f"evidence root must be object: {path}")
    return value


def _stage_records(stage2, artifact_root: Path, mappings: Sequence[tuple[Path, Path]]):
    terminal = stage2.verify_terminal_bundle(artifact_root, path_mappings=mappings)
    manifest = stage2._load_json(artifact_root / stage2.MANIFEST_NAME)
    manifest_sha = stage2._sha256(artifact_root / stage2.MANIFEST_NAME)
    resolver = stage2.PathResolver(mappings)
    nwb_root = ROOT / "sua_exploration/data/dandi_000688/sub-C"
    rows: dict[int, dict[str, Any]] = {}
    for cell in stage2.validate_manifest_schema(manifest):
        if cell["arm"] != "rt_sparse_endpoint_t4d":
            continue
        closure = stage2._load_json(artifact_root / "cells" / f"{cell['output_key']}.json")
        checked = stage2.validate_cell_closure(cell, closure, matrix_manifest_sha256=manifest_sha, resolver=resolver)
        actual = stage2.replay_evaluated_query_identity(checked, nwb_root=nwb_root)
        checked["actual_query_digests"] = tuple(actual[key] for key in (
            "ordered_window_start_sha256", "ordered_target_covariate_evalmask_sha256", "ordered_query_identity_sha256"))
        rows[int(cell["fold"])] = checked
    _need(tuple(sorted(rows)) == FOLDS, "Stage2 terminal has not verified 15 T4d cells")
    return terminal, manifest, rows


def _check_fold0_import() -> dict[str, Any]:
    receipt = _immutable_json(FOLD0_IMPORT)
    _need(receipt.get("status") == "PASS_SHA_VERIFIED_NONDESTRUCTIVE_IMPORT", "fold0 import receipt did not pass")
    files = receipt.get("files")
    _need(isinstance(files, Mapping), "fold0 import files missing")
    for name in ("checkpoint", "selection", "split", "config", "teacher_metadata"):
        row = files.get(name)
        _need(isinstance(row, Mapping), f"fold0 import missing {name}")
        path = Path(str(row.get("path", "")))
        _need(path.is_file() and _sha(path) == row.get("sha256"), f"fold0 imported {name} SHA drift")
    return receipt


def _bound_local_nwb_sha(manifest: Mapping[str, Any], outer: Mapping[str, Any]) -> tuple[Path, str]:
    """Map a receipt's target basename onto the Stage-2 local allowlist."""
    root = ROOT / "sua_exploration/data/dandi_000688/sub-C"
    target_name = Path(str(outer.get("outer_target_path", ""))).name
    _need(bool(target_name), "new forward receipt has no target NWB basename")
    matches = [row for row in manifest["nwb_allowlist"] if Path(str(row.get("path", ""))).name == target_name]
    _need(len(matches) == 1, f"target NWB basename is absent/ambiguous in Stage2 allowlist: {target_name}")
    local = root / target_name
    expected = str(matches[0].get("sha256", ""))
    _need(local.is_file() and _sha(local) == expected, "local Stage2-allowlisted target NWB SHA mismatch")
    return root, expected


def prepare(stage_artifact_root: Path, *, output_root: Path, mappings: Sequence[tuple[Path, Path]] = (), device: str = "cuda", allow_existing_root: bool = False) -> dict[str, Any]:
    """Build a no-side-effect plan only after the full Stage-2 terminal PASS."""
    stage2 = _module(STAGE2_PATH, "rt_b2_stage2_terminal")
    companion = _module(COMPANION_PATH, "rt_b2_companion_terminal")
    terminal, manifest, stage_rows = _stage_records(stage2, stage_artifact_root, mappings)
    _need(allow_existing_root or not output_root.exists(), f"forward-only output root already exists: {output_root}")
    fold0_import = _check_fold0_import()
    _need(_sha(EVALUATOR) == manifest["surfaces"]["outer_evaluator"]["sha256"], "current evaluator SHA differs from Stage2-bound evaluator")
    legacy = companion._legacy_b2_rows()
    teacher_sha = str(terminal["workspace_bindings"]["teacher_sha256"])
    nwb = {Path(str(row["path"])).name.removeprefix("sub-C_").removesuffix("_behavior+ecephys.nwb"): row for row in manifest["nwb_allowlist"]}
    plans = []
    for fold in FOLDS:
        evidence = companion._validate_b2_self(fold, legacy[fold], stage_teacher_sha=teacher_sha, nwb_rows=nwb)
        selection = companion._json(evidence["files"]["selection"])
        checkpoint = Path(str(selection["best_model_path"]))
        config, split, selection_path = (evidence["files"][key] for key in ("config", "split", "selection"))
        local_output = output_root / "outer" / f"f{fold:02d}_b2_d1024_zero4_forward_only.json"
        base = ["/home/xinyuan/miniconda3/envs/spint/bin/python", str(EVALUATOR), "--config", str(config), "--checkpoint", str(checkpoint), "--split-manifest", str(split), "--selection-receipt", str(selection_path), "--output", str(local_output), "--outer-fold", str(fold), "--device", device]
        if fold == 0:
            # Do not rewrite absolute paths within old selection/callback bytes.
            imported = fold0_import["files"]
            remote_output = f"{FOLD0_REMOTE_ROOT}/outer/f00_b2_d1024_zero4_forward_only.json"
            remote = ["/home/xinyuan/miniconda3/envs/spint/bin/python", REMOTE_EVALUATOR_CWD + "/src/rt_clean_nested_loso_eval.py", "--config", str(imported["config"]["remote_path"]), "--checkpoint", str(imported["checkpoint"]["remote_path"]), "--split-manifest", str(imported["split"]["remote_path"]), "--selection-receipt", str(imported["selection"]["remote_path"]), "--output", remote_output, "--outer-fold", "0", "--device", device]
            plans.append({"fold": fold, "host": FOLD0_HOST, "cwd": REMOTE_EVALUATOR_CWD, "environment": {"CUDA_VISIBLE_DEVICES": "set by operator"}, "remote_command": remote, "remote_output": remote_output, "local_import_destination": str(local_output), "checkpoint_sha256": selection["best_model_sha256"]})
        else:
            _need(checkpoint.is_file() and _sha(checkpoint) == selection["best_model_sha256"], f"fold {fold}: sealed local checkpoint unavailable")
            plans.append({"fold": fold, "host": "local", "cwd": str(EVALUATOR_CWD), "environment": {"CUDA_VISIBLE_DEVICES": "set by operator"}, "command": base, "output": str(local_output), "checkpoint_sha256": selection["best_model_sha256"]})
    return {"schema": "rt_sparse_t4d_b2_forward_reeval_terminal_plan_v1", "status": "PASS_STAGE2_TERMINAL_15FOLD_FORWARD_ONLY_PREPARED_NOT_LAUNCHED", "stage2_terminal": terminal, "stage2_manifest_sha256": _sha(stage_artifact_root / stage2.MANIFEST_NAME), "output_root_must_not_exist": str(output_root), "device": device, "plans": plans, "requirements": ["run every fold under the stated cwd; legacy configs use paths.root_dir='.'", "run each plan exactly once", "do not overwrite historical or new outputs", "import fold0 remote result with SHA verification before finalize", "finalize checks state invariance and exact actual query digest equality"], "non_interference": {"gpu_started": False, "torch_imported": False, "nwb_opened": False, "training_started": False, "optimizer_constructed": False, "artifact_written": False}}


def _remote_source_sha(manifest: Mapping[str, Any]) -> dict[str, str]:
    paths = {
        "outer_evaluator": REMOTE_EVALUATOR_CWD + "/src/rt_clean_nested_loso_eval.py",
        "datamodule": REMOTE_EVALUATOR_CWD + "/src/data/rt_nested_loso_datamodule.py",
        "falcon_dataset": REMOTE_EVALUATOR_CWD + "/src/data/falcon_datamodule.py",
    }
    command = "sha256sum " + " ".join(shlex.quote(path) for path in paths.values())
    done = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", FOLD0_HOST, command], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    _need(done.returncode == 0, f"fold0 remote source SHA command failed: {done.stderr.strip()}")
    parsed = {row.split(maxsplit=1)[1].strip(): row.split(maxsplit=1)[0] for row in done.stdout.splitlines() if len(row.split(maxsplit=1)) == 2}
    result = {name: parsed.get(path, "") for name, path in paths.items()}
    for name, value in result.items():
        _need(value == manifest["surfaces"][name]["sha256"], f"fold0 remote {name} SHA differs from Stage2-bound evaluator")
    return result


def _seal(path: Path, payload: Mapping[str, Any]) -> None:
    _need(not path.exists(), f"refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".rt-b2-launch-", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_launch_binding(launch: Mapping[str, Any], *, plan: Mapping[str, Any], manifest: Mapping[str, Any],
                             fold: int, output: Path, stage_actual: tuple[str, str, str]) -> None:
    """Validate that a sealed receipt describes the recomputed exact-once plan."""
    _need(launch.get("status") == "PASS_EXACT_ONCE_FORWARD_ONLY" and launch.get("fold") == fold, f"fold {fold}: launch receipt failed")
    _need(launch.get("command") == plan.get("command", plan.get("remote_command")), f"fold {fold}: launch command differs from recomputed plan")
    _need(launch.get("cwd") == plan.get("cwd"), f"fold {fold}: launch cwd differs from recomputed plan")
    _need(launch.get("output", {}).get("path") == str(output) and launch.get("output", {}).get("sha256") == _sha(output), f"fold {fold}: launch/output binding mismatch")
    _need(tuple(launch.get("pre_execution_actual_query_digests", ())) == stage_actual == tuple(launch.get("stage2_actual_query_digests", ())), f"fold {fold}: launch did not bind exact evaluated query identity")
    execution = launch.get("execution")
    _need(isinstance(execution, Mapping) and execution.get("exit_code") == 0, f"fold {fold}: launch exit code is not zero")
    sources = execution.get("source_sha256")
    _need(isinstance(sources, Mapping), f"fold {fold}: launch lacks execution source SHA")
    for name in ("outer_evaluator", "datamodule", "falcon_dataset"):
        _need(sources.get(name) == manifest["surfaces"][name]["sha256"], f"fold {fold}: execution {name} SHA differs from Stage2 manifest")


def execute(stage_artifact_root: Path, *, output_root: Path, fold: int, mappings: Sequence[tuple[Path, Path]] = (), device: str = "cuda") -> dict[str, Any]:
    """Run exactly one sealed checkpoint once, then seal its execution receipt."""
    _need(fold in FOLDS, "fold must be in 0..14")
    stage2 = _module(STAGE2_PATH, "rt_b2_stage2_execute")
    companion = _module(COMPANION_PATH, "rt_b2_companion_execute")
    terminal, manifest, stage_rows = _stage_records(stage2, stage_artifact_root, mappings)
    plan = prepare(stage_artifact_root, output_root=output_root, mappings=mappings, device=device, allow_existing_root=True)
    item = next(item for item in plan["plans"] if item["fold"] == fold)
    output = output_root / "outer" / f"f{fold:02d}_b2_d1024_zero4_forward_only.json"
    launch_path = output_root / "launch" / f"f{fold:02d}_b2_d1024_zero4_launch_v1.json"
    _need(not output.exists() and not launch_path.exists(), f"fold {fold}: exact-once output or launch receipt already exists")
    legacy = companion._legacy_b2_rows()
    old = companion._json(companion._bound_legacy_file(legacy[fold], "outer"))
    evidence = {"fold": fold, "outer": old, "split": companion._json(companion._bound_legacy_file(legacy[fold], "split")), "config": companion._yaml(companion._bound_legacy_file(legacy[fold], "config"))}
    nwb_root, nwb_sha = _bound_local_nwb_sha(manifest, old)
    actual = companion._reconstruct_b2_query_identity(evidence, nwb_root=nwb_root, expected_target_nwb_sha256=nwb_sha)
    _need(actual == stage_rows[fold]["actual_query_digests"], f"fold {fold}: pre-execution actual query digest differs from Stage2")
    # Preserve an operator-pinned device set when supplied; an absent variable
    # deliberately remains absent so ``--device cuda`` can see the selected
    # host default rather than being silently changed to zero visible GPUs.
    env = ({"CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"]}
           if "CUDA_VISIBLE_DEVICES" in os.environ else {})
    if fold == 0:
        source_sha = _remote_source_sha(manifest)
        remote_env = ("env CUDA_VISIBLE_DEVICES=" + shlex.quote(env["CUDA_VISIBLE_DEVICES"]) + " " if "CUDA_VISIBLE_DEVICES" in env else "")
        remote = "cd " + shlex.quote(item["cwd"]) + " && printf '__REMOTE_PID=%s\\n' \"$$\" && exec " + remote_env + shlex.join(item["remote_command"])
        process = subprocess.Popen(["ssh", "-o", "BatchMode=yes", FOLD0_HOST, remote], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        stdout, _ = process.communicate()
        _need(process.returncode == 0, f"fold0 remote forward-only evaluator failed: {stdout[-2000:]}")
        remote_output = str(item["remote_output"])
        remote_sha_cmd = subprocess.run(["ssh", "-o", "BatchMode=yes", FOLD0_HOST, "sha256sum " + shlex.quote(remote_output)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        _need(remote_sha_cmd.returncode == 0, f"fold0 remote output SHA failed: {remote_sha_cmd.stderr.strip()}")
        remote_sha = remote_sha_cmd.stdout.split()[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        incoming = output.with_suffix(".incoming")
        _need(not incoming.exists(), f"fold0 incoming path already exists: {incoming}")
        copied = subprocess.run(["scp", f"{FOLD0_HOST}:{remote_output}", str(incoming)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        _need(copied.returncode == 0 and _sha(incoming) == remote_sha, "fold0 output transfer SHA mismatch")
        os.replace(incoming, output)
        execution = {"host": FOLD0_HOST, "exit_code": process.returncode, "remote_pid_line": next((line for line in stdout.splitlines() if line.startswith("__REMOTE_PID=")), None), "local_ssh_pid": process.pid, "remote_output": remote_output, "remote_output_sha256": remote_sha, "stdout_tail": stdout[-2000:], "source_sha256": source_sha}
    else:
        source_sha = {name: _sha(EVALUATOR.parent / relative) if name != "outer_evaluator" else _sha(EVALUATOR) for name, relative in {"outer_evaluator": Path("rt_clean_nested_loso_eval.py"), "datamodule": Path("data/rt_nested_loso_datamodule.py"), "falcon_dataset": Path("data/falcon_datamodule.py")}.items()}
        for name, value in source_sha.items():
            _need(value == manifest["surfaces"][name]["sha256"], f"local {name} SHA differs from Stage2-bound evaluator")
        process = subprocess.Popen(item["command"], cwd=item["cwd"], env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        stdout, _ = process.communicate()
        _need(process.returncode == 0, f"fold {fold} forward-only evaluator failed: {stdout[-2000:]}")
        execution = {"host": "local", "exit_code": process.returncode, "pid": process.pid, "stdout_tail": stdout[-2000:], "source_sha256": source_sha}
    _need(output.is_file(), f"fold {fold}: evaluator returned success without output")
    receipt = json.loads(output.read_text(encoding="utf-8"))
    audit = receipt.get("matched_query_window_identity", {}).get(stage_rows[fold]["session"], {})
    all_eligible = tuple(audit.get(key) for key in ("ordered_window_start_sha256", "ordered_target_covariate_evalmask_sha256", "ordered_query_identity_sha256"))
    _need(all_eligible == stage_rows[fold]["query_digests"], f"fold {fold}: output all-eligible query digest mismatch")
    _need(int(receipt.get("query_windows_evaluated", -1)) == int(old["query_windows_evaluated"]), f"fold {fold}: output count drift")
    launch = {"schema": "rt_sparse_t4d_b2_forward_reeval_launch_v1", "status": "PASS_EXACT_ONCE_FORWARD_ONLY", "fold": fold, "command": item.get("command", item.get("remote_command")), "cwd": item["cwd"], "environment": env, "stage2_manifest_sha256": _sha(stage_artifact_root / stage2.MANIFEST_NAME), "stage2_actual_query_digests": list(stage_rows[fold]["actual_query_digests"]), "pre_execution_actual_query_digests": list(actual), "output": {"path": str(output), "sha256": _sha(output)}, "execution": execution, "non_interference": {"training_started": False, "optimizer_constructed": False, "backward_called": False}}
    _seal(launch_path, launch)
    return {"status": launch["status"], "fold": fold, "output": str(output), "launch_receipt": str(launch_path), "launch_receipt_sha256": _sha(launch_path)}


def _summary(deltas: list[float], labels: Sequence[str] | None = None) -> dict[str, Any]:
    _need(bool(deltas), "empty paired delta list")
    names = list(labels) if labels is not None else [str(index) for index in range(len(deltas))]
    _need(len(names) == len(deltas), "summary labels/deltas length mismatch")
    removed = max(range(len(deltas)), key=lambda index: (abs(deltas[index]), -index))
    kept = deltas[:removed] + deltas[removed + 1:]
    positive, negative = sum(x > 0 for x in deltas), sum(x < 0 for x in deltas)
    nonzero = positive + negative
    tail = min(positive, negative)
    sign_p = (2.0 * sum(_math.comb(nonzero, count) for count in range(tail + 1)) / (2 ** nonzero)) if nonzero else 1.0
    return {"n": len(deltas), "mean": sum(deltas) / len(deltas), "median": sorted(deltas)[len(deltas)//2], "positive": positive, "negative": negative, "zero": sum(x == 0 for x in deltas), "leave_largest_absolute_out_mean": (sum(kept) / len(kept)) if kept else None, "removed": {"index": removed, "label": names[removed], "delta": deltas[removed]}, "two_sided_exact_sign_test_p": min(1.0, sign_p)}


def finalize(stage_artifact_root: Path, *, output_root: Path, mappings: Sequence[tuple[Path, Path]] = ()) -> dict[str, Any]:
    stage2 = _module(STAGE2_PATH, "rt_b2_stage2_final")
    companion = _module(COMPANION_PATH, "rt_b2_companion_final")
    terminal, manifest, stage_rows = _stage_records(stage2, stage_artifact_root, mappings)
    recomputed_plan = prepare(stage_artifact_root, output_root=output_root, mappings=mappings, allow_existing_root=True)
    plans = {int(item["fold"]): item for item in recomputed_plan["plans"]}
    _need(_sha(EVALUATOR) == manifest["surfaces"]["outer_evaluator"]["sha256"], "evaluator SHA differs from Stage2-bound evaluator")
    legacy = companion._legacy_b2_rows()
    result_rows, deltas = [], []
    for fold in FOLDS:
        path = output_root / "outer" / f"f{fold:02d}_b2_d1024_zero4_forward_only.json"
        launch_path = output_root / "launch" / f"f{fold:02d}_b2_d1024_zero4_launch_v1.json"
        _need(path.is_file(), f"missing new forward-only receipt: {path}")
        launch = _immutable_json(launch_path)
        _need(launch.get("stage2_manifest_sha256") == _sha(stage_artifact_root / stage2.MANIFEST_NAME), f"fold {fold}: launch manifest binding mismatch")
        _validate_launch_binding(launch, plan=plans[fold], manifest=manifest, fold=fold, output=path, stage_actual=stage_rows[fold]["actual_query_digests"])
        new = json.loads(path.read_text(encoding="utf-8"))
        _need(isinstance(new, Mapping) and new.get("status") == "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP", f"fold {fold}: forward receipt failed")
        old = companion._json(companion._bound_legacy_file(legacy[fold], "outer"))
        stage = stage_rows[fold]
        _need(new.get("arm") == "zero4" and new.get("outer_loso_fold") == fold, f"fold {fold}: new receipt identity mismatch")
        _need(new.get("checkpoint_sha256") == old.get("checkpoint_sha256"), f"fold {fold}: checkpoint SHA differs from sealed B2")
        _need(new.get("outer_target_session") == stage["session"], f"fold {fold}: target mismatch")
        _need(new.get("model_state_unchanged") is True and new.get("model_state_three_point_unchanged") is True and new.get("model_state_sha256_before") == new.get("model_state_sha256_after"), f"fold {fold}: model state changed")
        _need(new.get("target_backpropagation") is False and new.get("optimizer_present") is False and new.get("model_training_mode") is False, f"fold {fold}: non-forward-only state")
        full = stage["query_digests"]
        audit = new.get("matched_query_window_identity", {}).get(stage["session"], {})
        observed = tuple(audit.get(key) for key in ("ordered_window_start_sha256", "ordered_target_covariate_evalmask_sha256", "ordered_query_identity_sha256"))
        _need(observed == full, f"fold {fold}: all-eligible query digest mismatch")
        # The replay is explicitly bound to the newly executed receipt. The
        # old outer receipt is used only to prove checkpoint provenance.
        evidence = {"fold": fold, "outer": new, "split": companion._json(companion._bound_legacy_file(legacy[fold], "split")), "config": companion._yaml(companion._bound_legacy_file(legacy[fold], "config"))}
        nwb_root, nwb_sha = _bound_local_nwb_sha(manifest, new)
        actual = companion._reconstruct_b2_query_identity(evidence, nwb_root=nwb_root, expected_target_nwb_sha256=nwb_sha)
        _need(actual == stage["actual_query_digests"], f"fold {fold}: actual evaluated query digest mismatch")
        _need(int(new.get("query_windows_evaluated", -1)) == int(old.get("query_windows_evaluated", -2)), f"fold {fold}: forward evaluated-count drift")
        score = float(new.get("r2_variance_weighted"))
        _need(math.isfinite(score), f"fold {fold}: nonfinite R2")
        delta = float(stage["r2"]) - score
        deltas.append(delta)
        result_rows.append({"fold": fold, "session": stage["session"], "t4d_r2": stage["r2"], "b2_forward_only_r2": score, "t4d_minus_b2": delta, "receipt": {"path": str(path), "sha256": _sha(path)}, "launch_receipt": {"path": str(launch_path), "sha256": _sha(launch_path)}})
    final_path = output_root / "RT_T4D_VS_B2_D1024_FORWARD_ONLY_15FOLD_FINAL_v1.json"
    _need(not final_path.exists(), f"refusing to overwrite finalizer receipt: {final_path}")
    labels = [str(row["session"]) for row in result_rows]
    full = _summary(deltas, labels)
    prospective = [deltas[index] for index in range(4, 15)]
    prospective_summary = _summary(prospective, labels[4:15])
    prospective_gate = bool(prospective_summary["mean"] > 0.0 and prospective_summary["median"] > 0.0 and prospective_summary["positive"] > len(prospective) / 2.0)
    report = {"schema": "rt_t4d_vs_b2_d1024_forward_only_15fold_final_v1", "status": "PASS_15FOLD_EXACT_STAGE2_QUERY_FORWARD_ONLY", "rows": result_rows, "t4d_minus_b2_full15": full, "t4d_minus_b2_prospective_folds4_14": {"folds": list(range(4, 15)), "statistics": prospective_summary, "gate_mean_positive_median_positive_sign_majority": prospective_gate}, "stage2_terminal": terminal, "non_interference": {"training_started": False, "optimizer_constructed": False, "backward_called": False}}
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".rt-b2-final-", dir=final_path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o444)
        os.replace(temporary, final_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"status": report["status"], "final_receipt": str(final_path), "final_receipt_sha256": _sha(final_path), "full15": full, "prospective_folds4_14": report["t4d_minus_b2_prospective_folds4_14"]}


def _parse_map(value: str) -> tuple[Path, Path]:
    _need("=" in value, "--path-map requires REMOTE=LOCAL")
    left, right = value.split("=", 1)
    return Path(left), Path(right)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("prepare", "execute", "finalize"):
        command = sub.add_parser(mode)
        command.add_argument("--stage-artifact-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--path-map", action="append", default=[])
        if mode in {"prepare", "execute"}:
            command.add_argument("--device", default="cuda")
        if mode == "execute":
            command.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    mappings = tuple(_parse_map(item) for item in args.path_map)
    if args.mode == "prepare":
        result = prepare(args.stage_artifact_root, output_root=args.output_root, mappings=mappings, device=args.device)
    elif args.mode == "execute":
        result = execute(args.stage_artifact_root, output_root=args.output_root, mappings=mappings, device=args.device, fold=args.fold)
    else:
        result = finalize(args.stage_artifact_root, output_root=args.output_root, mappings=mappings)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except TerminalError as error:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(error)}, indent=2), file=sys.stderr)
        raise SystemExit(2) from error
