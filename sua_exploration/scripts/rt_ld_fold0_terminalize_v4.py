#!/usr/bin/env python3
"""Compose sealed RT L-D validators into the one-fold terminal decision.

This module is inert unless asked to prepare or validate a completed run.  It
does not train, evaluate, watch, or acquire a GPU.  The fit and outer-scope
contracts intentionally remain owned by the sealed v8 handoff implementation;
this terminalizer only adds the terminal receipt's root, binding, pairing,
delta, and atomic-output contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
V8_RUNNER = ROOT / "sua_exploration/scripts/rt_ld_device_handoff_v8.py"
V8_PLAN = ROOT / "sua_exploration/results/rt_ld_device_handoff_v8/RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v8.json"
V8_ROOT_REVIEW = ROOT / "sua_exploration/results/rt_ld_device_handoff_v8/RT_LD_DEVICE_HANDOFF_ROOT_REVIEW_v8.json"
# This is the immutable v2 -> v3 supersession receipt: it records that v3 was
# the immediate predecessor, without making the v4 supersession marker a
# self-referential plan dependency.
V3_SUPERSESSION = ROOT / "sua_exploration/results/rt_ld_fold0_terminalize_v2/RT_LD_FOLD0_TERMINALIZE_V2_SUPERSEDED_BY_V3.json"
V3_PLAN = ROOT / "sua_exploration/results/rt_ld_fold0_terminalize_v3/RT_LD_FOLD0_TERMINALIZE_PREPARED_PLAN_v3.json"
RESULT_DIR = ROOT / "sua_exploration/results/rt_ld_fold0_terminalize_v4"
PLAN = RESULT_DIR / "RT_LD_FOLD0_TERMINALIZE_PREPARED_PLAN_v4.json"
TEST = ROOT / "sua_exploration/tests/test_rt_ld_fold0_terminalize_v4.py"
GATE = 0.03


def _load_v8() -> Any:
    spec = importlib.util.spec_from_file_location("rt_ld_v8_sealed", V8_RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V8 = _load_v8()
ARM_SPECS: Mapping[str, Mapping[str, str]] = V8.ARM_SPECS
ARM_BY_DIRECTORY = {spec["directory"]: experiment for experiment, spec in ARM_SPECS.items()}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _immutable(path: Path) -> None:
    if not path.is_file() or path.is_symlink() or (path.stat().st_mode & 0o777) != 0o444:
        raise ValueError(f"immutable binding required: {path}")


def prepare() -> None:
    """Seal the non-executing v4 terminalizer plan exactly once."""
    if RESULT_DIR.exists():
        raise FileExistsError("v4 terminalizer plan is append-only")
    for path in (V8_PLAN, V8_ROOT_REVIEW, V3_SUPERSESSION, V3_PLAN):
        _immutable(path)
    bindings = {
        str(path.relative_to(ROOT)): _sha(path)
        for path in (V8_RUNNER, V8_PLAN, V8_ROOT_REVIEW, V3_SUPERSESSION, V3_PLAN)
    }
    body = {
        "schema": "rt_ld_fold0_terminalize_prepared_plan_v4",
        "status": "PREPARED_CPU_ONLY_NOT_TERMINAL",
        "script_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": _sha(Path(__file__).resolve()),
        "test_path": str(TEST.relative_to(ROOT)),
        "test_sha256": _sha(TEST),
        "bindings": bindings,
        "validator_composition": {
            "fit": "rt_ld_device_handoff_v8.V4._validate_fit_artifacts",
            "outer_scope": "rt_ld_device_handoff_v8._outer_scope_v8",
        },
        "single_outer_fold_seed_one_session_only": True,
        "frozen_gate": GATE,
        "gpu_launched": False,
        "formal_heldout_opened": False,
        "fail_closed": True,
    }
    RESULT_DIR.mkdir(parents=True)
    PLAN.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(PLAN, 0o444)


def _check_plan() -> dict[str, Any]:
    plan = _json(PLAN)
    if plan.get("schema") != "rt_ld_fold0_terminalize_prepared_plan_v4":
        raise ValueError("unexpected terminalizer plan schema")
    if plan.get("status") != "PREPARED_CPU_ONLY_NOT_TERMINAL":
        raise ValueError("terminalizer plan is not inert")
    if plan.get("script_sha256") != _sha(Path(__file__).resolve()):
        raise ValueError("terminalizer script hash drift")
    if plan.get("test_sha256") != _sha(TEST):
        raise ValueError("terminalizer test hash drift")
    if plan.get("gpu_launched") is not False or plan.get("formal_heldout_opened") is not False:
        raise ValueError("terminalizer plan inertness drift")
    bindings = plan.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("terminalizer binding map missing")
    expected = {str(path.relative_to(ROOT)) for path in (V8_RUNNER, V8_PLAN, V8_ROOT_REVIEW, V3_SUPERSESSION, V3_PLAN)}
    if set(bindings) != expected:
        raise ValueError("terminalizer binding path set drift")
    for relative, digest in bindings.items():
        if not isinstance(digest, str) or _sha(ROOT / relative) != digest:
            raise ValueError(f"terminalizer binding hash drift: {relative}")
    return plan


def _exact_path(receipt: Mapping[str, Any], key: str, expected: Path) -> None:
    value = receipt.get(key)
    if not isinstance(value, str) or Path(value).resolve() != expected.resolve():
        raise ValueError(f"exact receipt path mismatch: {key}")


def _join(outer: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: outer.get(key)
        for key in (
            "outer_target_session",
            "outer_target_path",
            "query_start_trial",
            "window_size",
            "query_windows_evaluated",
            "data_dir",
            "inner_train_sessions",
            "inner_validation_session",
        )
    }
    for key in ("outer_target_session", "outer_target_path", "data_dir", "inner_validation_session"):
        if not isinstance(result[key], str) or not result[key]:
            raise ValueError(f"missing paired join field: {key}")
    for key in ("query_start_trial", "window_size", "query_windows_evaluated"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"invalid paired join field: {key}")
    if not isinstance(result["inner_train_sessions"], list) or not result["inner_train_sessions"] or not all(
        isinstance(session, str) and session for session in result["inner_train_sessions"]
    ):
        raise ValueError("invalid paired inner-train sessions")
    return result


def _arm(run_root: Path, directory: str, experiment: str) -> dict[str, Any]:
    arm = run_root / directory
    if arm.is_symlink() or not arm.is_dir() or arm.resolve().parent != run_root:
        raise ValueError("arm directory escapes exact run root")

    # These two calls are deliberately direct composition.  Do not reproduce
    # v8's fit/outer scope checks here: a future sealed correction must flow
    # through this terminal decision unchanged.
    checkpoint = Path(V8.V4._validate_fit_artifacts(arm, experiment)).resolve()
    V8._outer_scope_v8(arm, experiment)

    selection_path = arm / "rt_nested_selection_receipt.json"
    split_path = arm / "split_manifest.json"
    config_path = arm / ".hydra/config.yaml"
    outer_path = arm / "rt_ld_outer_eval.json"
    selection = _json(selection_path)
    outer = _json(outer_path)
    if checkpoint != Path(str(selection.get("best_model_path", ""))).resolve():
        raise ValueError("sealed checkpoint differs from selected checkpoint")
    if selection.get("best_model_sha256") != _sha(checkpoint):
        raise ValueError("selected checkpoint hash drift")
    _exact_path(selection, "selection_receipt_path", selection_path)
    _exact_path(selection, "config_path", config_path)
    _exact_path(selection, "split_manifest_path", split_path)
    _exact_path(selection, "run_dir", arm)
    if selection.get("config_sha256") != _sha(config_path) or selection.get("split_manifest_sha256") != _sha(split_path):
        raise ValueError("selection config/split hash drift")
    _exact_path(outer, "checkpoint_path", checkpoint)
    _exact_path(outer, "selection_receipt_path", selection_path)
    _exact_path(outer, "config_path", config_path)
    _exact_path(outer, "fit_split_manifest", split_path)
    if outer.get("checkpoint_sha256") != _sha(checkpoint):
        raise ValueError("outer checkpoint hash drift")

    score = outer.get("r2_variance_weighted")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("finite outer R² required")
    return {
        "r2_variance_weighted": float(score),
        "join": _join(outer),
        "paths": {
            "config": str(config_path.resolve()),
            "selection": str(selection_path.resolve()),
            "split": str(split_path.resolve()),
            "outer_eval": str(outer_path.resolve()),
            "checkpoint": str(checkpoint),
        },
        "sha256": {
            "config": _sha(config_path),
            "selection": _sha(selection_path),
            "split": _sha(split_path),
            "outer_eval": _sha(outer_path),
            "checkpoint": _sha(checkpoint),
        },
    }


def terminalize(run_root: Path) -> dict[str, Any]:
    """Validate exactly the three completed arms and return a receipt object."""
    plan = _check_plan()
    run_root = Path(run_root).resolve()
    if not run_root.is_dir() or run_root.is_symlink():
        raise ValueError("real, non-symlink run root required")
    children = list(run_root.iterdir())
    if set(child.name for child in children) != set(ARM_BY_DIRECTORY) or any(child.is_symlink() for child in children):
        raise ValueError("exact non-symlink run-root children required")
    rows = {directory: _arm(run_root, directory, experiment) for directory, experiment in ARM_BY_DIRECTORY.items()}
    base_join = rows["01_a0"]["join"]
    if any(row["join"] != base_join for row in rows.values()):
        raise ValueError("paired outer scope/join mismatch")
    g_full_minus_a0 = rows["02_g_full"]["r2_variance_weighted"] - rows["01_a0"]["r2_variance_weighted"]
    g_full_minus_g_xls = rows["02_g_full"]["r2_variance_weighted"] - rows["03_g_xls"]["r2_variance_weighted"]
    passed = g_full_minus_a0 >= GATE and g_full_minus_g_xls >= GATE
    session = base_join["outer_target_session"]
    return {
        "schema": "rt_ld_fold0_terminal_receipt_v4",
        "status": "PASS_BOTH_GATES" if passed else "STOP_GATE_FAILED",
        "prepared_plan_path": str(PLAN.resolve()),
        "prepared_plan_sha256": _sha(PLAN),
        "upstream_bindings": plan["bindings"],
        "single_outer_fold_seed_one_session_only": True,
        "formal_heldout_opened": False,
        "deeper_film_attention_forbidden": not passed,
        "arms": rows,
        "g_full_minus_a0": {
            "pooled_delta": g_full_minus_a0,
            "mean": g_full_minus_a0,
            "median": g_full_minus_a0,
            "per_session_delta": {session: g_full_minus_a0},
            "positive_count": int(g_full_minus_a0 > 0),
            "session_count": 1,
        },
        "g_full_minus_g_xls": {
            "pooled_delta": g_full_minus_g_xls,
            "mean": g_full_minus_g_xls,
            "median": g_full_minus_g_xls,
            "per_session_delta": {session: g_full_minus_g_xls},
            "positive_count": int(g_full_minus_g_xls > 0),
            "session_count": 1,
        },
    }


def write(run_root: Path, output: Path) -> dict[str, Any]:
    """Atomically create one immutable terminal receipt; never overwrite."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("terminal receipt output already exists or is a symlink")
    receipt = terminalize(run_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".rt-ld-terminal-", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        os.chmod(output, 0o444)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    if args.run_root is None or (args.validate == bool(args.write)):
        parser.error("use --prepare, or exactly one of --validate/--write with --run-root")
    receipt = terminalize(args.run_root) if args.validate else write(args.run_root, args.write)
    print(json.dumps({"status": receipt["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
