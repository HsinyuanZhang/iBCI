#!/usr/bin/env python3
"""One-shot remote-5070 orchestrator for the imported early-v2 H-LS replay.

Default execution is a local receipt-only dry run.  The explicit action runs
the already-isolated binder, checker, readiness, closure, five atomic paired
evaluations, and receipt-only aggregate in their fixed order.  It does not
duplicate target logic: each paired evaluator computes H-C then H-LS and only
publishes/is accepted when the original-H-C reproduction check passes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    DATES,
    EARLY_LAUNCH_SCHEMA,
    EARLY_LAUNCH_STATUS,
    EARLY_TERMINAL_SCHEMA,
    EARLY_TERMINAL_STATUS,
    POST_BINDER_SCHEMA,
    POST_BINDER_STATUS,
    read_immutable_json,
    sha256_file,
    write_immutable_json,
)
from scripts.h1_carrierid_date_lodo_hls_early_v2_remote_transfer import (
    CODE_FILES as TRANSFER_CODE_FILES,
    REMOTE,
    TRANSFER_SCHEMA,
    TRANSFER_STATUS,
)
from scripts.h1_carrierid_date_lodo_hls_fivedate_aggregate import (
    AGGREGATE_SCHEMA,
    AGGREGATE_STATUS,
    EVALUATION_SCHEMA,
)
from scripts.h1_carrierid_date_lodo_hls_fivedate_contract import (
    EVALUATOR_PREFLIGHT_SCHEMA,
    EVALUATOR_PREFLIGHT_STATUS,
    UPSTREAM_AGGREGATE_SCHEMA,
    UPSTREAM_AGGREGATE_STATUS,
)
from scripts.h1_carrierid_date_lodo_hls_fivedate_terminal_checker import (
    CHECKER_SCHEMA,
    CHECKER_STATUS,
)
from scripts.h1_carrierid_date_lodo_hls_target_evaluator_closure import (
    CLOSURE_SCHEMA,
    CLOSURE_STATUS,
)
REPLAY_SCHEMA = "h1_carrierid_date_lodo_hls_early_v2_remote5070_replay_v1"
REPLAY_CLAIM_STATUS = "CLAIMED_REMOTE5070_FIXED_FIVEDATE_REPLAY_NOT_YET_COMPLETE"
REPLAY_PASS_STATUS = "PASS_H1_HLS_EARLY_V2_REMOTE5070_FIXED_FIVEDATE_REPLAY_AGGREGATED"
REPLAY_FAIL_STATUS = "FAILED_H1_HLS_EARLY_V2_REMOTE5070_REPLAY_FORENSIC_STATE_PRESERVED"
REPLAY_ROOT = ROOT / "pilot_artifacts/h1_carrierid_date_lodo_hls_early_v2/remote5070_replay"
EVALUATION_DIR = ROOT / "pilot_artifacts/h1_carrierid_date_lodo_hls/terminal_evaluations"
AGGREGATE_OUTPUT = (
    ROOT / "pilot_artifacts/h1_carrierid_date_lodo_hls/"
    "H1_CARRIERID_DATE_LODO_HLS_FIVEDATE_TERMINAL_AGGREGATE_v1.json"
)
DEFAULT_DATA_DIR = ROOT / "data/000954"

CHAIN_FILES = (
    "scripts/h1_carrierid_date_lodo_hls_early_v2_post_upstream_binder.py",
    "scripts/h1_carrierid_date_lodo_hls_early_v2_terminal_checker.py",
    "scripts/h1_carrierid_date_lodo_hls_fivedate_target_evaluator_preflight.py",
    "scripts/h1_carrierid_date_lodo_hls_target_evaluator_closure.py",
    "scripts/h1_carrierid_date_lodo_hls_terminal_evaluate.py",
    "scripts/h1_carrierid_date_lodo_hls_fivedate_aggregate.py",
    "scripts/h1_carrierid_date_lodo_hls_early_v2_remote5070_replay.py",
)


class Remote5070ReplayError(RuntimeError):
    """A one-shot remote replay invariant failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Remote5070ReplayError(message)


def _inside(path: str | Path, repo_root: Path, label: str) -> Path:
    raw = Path(path)
    _need(not raw.is_symlink(), f"{label} cannot be a symlink: {raw}")
    candidate = raw.resolve()
    try:
        candidate.relative_to(repo_root)
    except ValueError as error:
        raise Remote5070ReplayError(f"{label} is outside the fixed repo: {candidate}") from error
    return candidate


def _same_file(row: Mapping[str, Any], *, repo_root: Path, label: str) -> Path:
    path = _inside(str(row.get("path", "")), repo_root, label)
    _need(path.is_file() and not path.is_symlink(), f"missing closure file: {label}: {path}")
    _need(path.stat().st_size == row.get("size") and sha256_file(path) == row.get("sha256")
          and stat.S_IMODE(path.stat().st_mode) == row.get("mode"),
          f"transfer closure file changed: {label}: {path}")
    return path


def _validate_transfer_receipt(path: Path, *, repo_root: Path) -> tuple[Path, dict[str, Any], str]:
    receipt_path, receipt, receipt_sha = read_immutable_json(
        path, schema=TRANSFER_SCHEMA, status=TRANSFER_STATUS,
    )
    _need(receipt.get("remote") == REMOTE
          and receipt.get("identical_absolute_repo_root") == str(repo_root),
          "transfer receipt does not authorize this remote absolute repo")
    scope = receipt.get("scope")
    _need(isinstance(scope, Mapping) and scope.get("binder_called") is False
          and scope.get("checker_called") is False and scope.get("evaluator_called") is False
          and scope.get("target_opened") == 0 and scope.get("gpu_started") is False,
          "transfer receipt exceeded import-only scope")
    verification = receipt.get("verification")
    _need(isinstance(verification, Mapping)
          and verification.get("preflight_sha256_and_size") is True
          and verification.get("postflight_sha256_and_size") is True
          and verification.get("existing_mismatch_overwritten") is False
          and verification.get("rsync_ignore_existing") is True,
          "transfer receipt lacks no-overwrite pre/post verification")
    files = receipt.get("files")
    manifest = files.get("manifest") if isinstance(files, Mapping) else None
    _need(isinstance(manifest, list) and files.get("total") == len(manifest),
          "transfer receipt has an incomplete closure manifest")
    _need(isinstance(files.get("transferred_missing"), int)
          and isinstance(files.get("already_identical"), int)
          and files["transferred_missing"] + files["already_identical"] == len(manifest),
          "transfer receipt file accounting is incomplete")
    by_relative: dict[str, Mapping[str, Any]] = {}
    for row in manifest:
        _need(isinstance(row, Mapping), "transfer closure row is malformed")
        resolved = _same_file(row, repo_root=repo_root, label=str(row.get("label", "transfer file")))
        relative = str(resolved.relative_to(repo_root))
        _need(row.get("relative_path") == relative and relative not in by_relative,
              f"transfer closure relative-path identity is malformed/duplicated: {relative}")
        by_relative[relative] = row
    for row in receipt.get("remote_prerequisites", ()):
        _need(isinstance(row, Mapping), "remote prerequisite row is malformed")
        _same_file(row, repo_root=repo_root, label=str(row.get("label", "remote prerequisite")))
    missing = [relative for relative in TRANSFER_CODE_FILES if relative not in by_relative]
    _need(not missing, f"transfer receipt does not bind the complete replay code closure: {missing}")
    closure_rows: list[tuple[str, Any]] = [("early-v2 launch receipt", receipt.get("launch_receipt"))]
    dates = receipt.get("dates")
    if isinstance(dates, Mapping):
        closure_rows.extend(
            (f"{date} source terminal", dates.get(date, {}).get("terminal")) for date in DATES
        )
    for label, row in closure_rows:
        _need(isinstance(row, Mapping), f"transfer receipt lacks {label}")
        bound_path = _inside(str(row.get("path", "")), repo_root, label)
        manifest_row = by_relative.get(str(bound_path.relative_to(repo_root)))
        _need(isinstance(manifest_row, Mapping) and row.get("sha256") == manifest_row.get("sha256"),
              f"transfer receipt {label} is not bound by the verified closure manifest")
    return receipt_path, receipt, receipt_sha


def _validate_upstream(
    receipt: Mapping[str, Any], *, repo_root: Path,
) -> tuple[Path, dict[str, Any], str]:
    row = receipt.get("remote_upstream_aggregate")
    _need(isinstance(row, Mapping), "transfer receipt lacks remote upstream aggregate")
    path, body, digest = read_immutable_json(
        row.get("path", ""), schema=UPSTREAM_AGGREGATE_SCHEMA, status=UPSTREAM_AGGREGATE_STATUS,
    )
    _inside(path, repo_root, "remote upstream aggregate")
    route = body.get("route_prerequisite")
    _need(digest == row.get("sha256") and row.get("five_date_complete") is True
          and tuple(body.get("required_outer_dates", ())) == DATES
          and body.get("all_five_date_receipts_present_and_validated") is True
          and isinstance(route, Mapping) and route.get("status") == "source/date screen complete"
          and route.get("automatic_route_selection") == "FORBIDDEN",
          "remote H-S/H-C upstream is incomplete or changed after transfer")
    return path, body, digest


def _canonical_evaluation(date: str, evaluation_dir: Path) -> Path:
    return evaluation_dir / f"H1_CARRIERID_DATE_LODO_HLS_{date}_HC_HLS_TERMINAL_EVALUATION_v1.json"


def build_plan(
    *, transfer_receipt: Path, data_dir: Path = DEFAULT_DATA_DIR,
    repo_root: Path = ROOT, replay_root: Path | None = None,
    evaluation_dir: Path | None = None, aggregate_output: Path | None = None,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Construct an exact target-closed plan; no subprocess or target path is opened."""

    repo_root = Path(repo_root).resolve()
    replay_root = _inside(replay_root or (repo_root / REPLAY_ROOT.relative_to(ROOT)), repo_root, "replay root")
    evaluation_dir = _inside(evaluation_dir or (repo_root / EVALUATION_DIR.relative_to(ROOT)),
                             repo_root, "evaluation directory")
    aggregate_output = _inside(aggregate_output or (repo_root / AGGREGATE_OUTPUT.relative_to(ROOT)),
                               repo_root, "aggregate output")
    data_dir = _inside(data_dir, repo_root, "declared H1 data directory")
    transfer_path, transfer, transfer_sha = _validate_transfer_receipt(transfer_receipt, repo_root=repo_root)
    upstream_path, _upstream, upstream_sha = _validate_upstream(transfer, repo_root=repo_root)
    launch_row = transfer.get("launch_receipt")
    _need(isinstance(launch_row, Mapping), "transfer receipt lacks early-v2 launch receipt")
    launch_path, launch, launch_sha = read_immutable_json(
        launch_row.get("path", ""), schema=EARLY_LAUNCH_SCHEMA, status=EARLY_LAUNCH_STATUS,
    )
    _need(launch_sha == launch_row.get("sha256") and tuple(launch.get("fixed_grid", ())) == DATES,
          "early-v2 launch receipt changed after remote import")
    date_rows = transfer.get("dates")
    _need(isinstance(date_rows, Mapping) and tuple(date_rows) == DATES,
          "transfer receipt lacks the ordered five-date source terminal grid")
    terminals: dict[str, Path] = {}
    for date in DATES:
        row = date_rows.get(date, {}).get("terminal")
        _need(isinstance(row, Mapping), f"transfer receipt lacks source terminal: {date}")
        terminal_path, terminal, terminal_sha = read_immutable_json(
            row.get("path", ""), schema=EARLY_TERMINAL_SCHEMA, status=EARLY_TERMINAL_STATUS,
        )
        _need(terminal_sha == row.get("sha256") and terminal.get("outer_date") == date
              and terminal.get("arm") == "H-LS",
              f"imported H-LS terminal changed: {date}")
        terminals[date] = terminal_path

    outputs = {
        "binder": replay_root / "H1_HLS_EARLY_V2_POST_UPSTREAM_BINDER_v1.json",
        "checker": replay_root / "H1_HLS_EARLY_V2_FIVEDATE_SOURCE_TERMINAL_AGGREGATE_v1.json",
        "readiness": replay_root / "H1_HLS_EARLY_V2_TARGET_EVALUATOR_PREFLIGHT_v1.json",
        "closure": replay_root / "H1_HLS_EARLY_V2_TARGET_EVALUATOR_CLOSURE_v1.json",
        "aggregate": aggregate_output,
        "claim": replay_root / "H1_HLS_EARLY_V2_REMOTE5070_REPLAY_v1.claim.json",
        "log": replay_root / "H1_HLS_EARLY_V2_REMOTE5070_REPLAY_v1.log",
        "terminal": replay_root / "H1_HLS_EARLY_V2_REMOTE5070_REPLAY_v1.terminal.json",
    }
    evaluations = {date: _canonical_evaluation(date, evaluation_dir) for date in DATES}
    for label, path in (*outputs.items(), *((f"evaluation {date}", path) for date, path in evaluations.items())):
        _need(not os.path.lexists(str(path)), f"one-shot replay output already exists; refusing rerun: {label}: {path}")

    commands: list[dict[str, Any]] = []
    commands.append({"stage": "post_upstream_binder", "command": [
        python_executable, str(repo_root / CHAIN_FILES[0]), "--upstream-aggregate", str(upstream_path),
        "--early-launch-receipt", str(launch_path),
        *[item for date in DATES for item in ("--early-terminal", f"{date}={terminals[date]}")],
        "--output", str(outputs["binder"]),
    ]})
    commands.append({"stage": "terminal_checker", "command": [
        python_executable, str(repo_root / CHAIN_FILES[1]),
        "--post-upstream-binder", str(outputs["binder"]), "--output", str(outputs["checker"]),
    ]})
    commands.append({"stage": "target_evaluator_preflight", "command": [
        python_executable, str(repo_root / CHAIN_FILES[2]),
        "--source-terminal-aggregate", str(outputs["checker"]), "--output", str(outputs["readiness"]),
    ]})
    commands.append({"stage": "target_evaluator_closure", "command": [
        python_executable, str(repo_root / CHAIN_FILES[3]),
        "--evaluator-preflight", str(outputs["readiness"]), "--output", str(outputs["closure"]),
    ]})
    for date in DATES:
        commands.append({"stage": f"paired_evaluation_{date}", "outer_date": date, "target": True,
                         "command": [
            python_executable, str(repo_root / CHAIN_FILES[4]),
            "--evaluator-closure", str(outputs["closure"]),
            "--upstream-aggregate", str(upstream_path), "--outer-date", date,
            "--data-dir", str(data_dir), "--output", str(evaluations[date]),
            "--device", "cuda", "--execution-owner", "remote5070ti_original_hc_replay",
            "--execute-target-evaluation",
        ]})
    commands.append({"stage": "receipt_only_aggregate", "command": [
        python_executable, str(repo_root / CHAIN_FILES[5]),
        "--evaluation-dir", str(evaluation_dir), "--output", str(outputs["aggregate"]),
    ]})
    return {
        "schema": REPLAY_SCHEMA, "status": "DRY_RUN_REMOTE5070_REPLAY_NOT_EXECUTED",
        "repo_root": str(repo_root), "transfer_receipt": {"path": str(transfer_path), "sha256": transfer_sha},
        "upstream_aggregate": {"path": str(upstream_path), "sha256": upstream_sha},
        "launch_receipt": {"path": str(launch_path), "sha256": launch_sha},
        "source_terminals": {date: str(terminals[date]) for date in DATES},
        "data_dir_declared_not_opened": str(data_dir), "outputs": {key: str(value) for key, value in outputs.items()},
        "evaluations": {date: str(evaluations[date]) for date in DATES}, "commands": commands,
        "execution_contract": {
            "fixed_date_order": list(DATES), "run_all_dates_regardless_of_signed_delta": True,
            "no_intermediate_route_selection": True, "canonical_one_shot_no_rerun": True,
            "device_name_must_contain": "5070 Ti", "execution_owner": "remote5070ti_original_hc_replay",
            "atomic_compute_order": "H-C forward; H-LS forward; original-H-C reproduction validation",
            "acceptance_order": "paired receipt accepted/published only after reproduction PASS",
            "does_not_claim_reproduction_validation_precedes_hls_compute": True,
        },
        "scope": {"subprocess_started": False, "target_opened": 0, "gpu_queried": False},
    }


def _append_log(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), "event": event, **fields}, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def _run_streamed(
    command: Sequence[str], *, log: Path, cwd: Path,
    env: Mapping[str, str] | None = None,
) -> int:
    rendered = [str(value) for value in command]
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "COMMAND_START", "command": rendered}, sort_keys=True) + "\n")
        handle.flush()
        result = subprocess.run(rendered, cwd=cwd, env=None if env is None else dict(env),
                                text=True, stdout=handle, stderr=subprocess.STDOUT, check=False)
        handle.write(json.dumps({"event": "COMMAND_EXIT", "returncode": result.returncode}, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    return int(result.returncode)


def _gpu_name(device: str) -> str:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(device), "--query-gpu=name", "--format=csv,noheader"],
        text=True, capture_output=True, check=False,
    )
    _need(result.returncode == 0, f"cannot inspect replay GPU: {result.stderr.strip()}")
    name = result.stdout.strip()
    _need("5070 Ti" in name, f"replay GPU is not a 5070 Ti: {name!r}")
    return name


def _gpu_compute_owners(device: str) -> list[dict[str, Any]]:
    """Return every compute process on the selected physical GPU."""

    result = subprocess.run(
        ["nvidia-smi", "-i", str(device), "--query-compute-apps=pid,process_name",
         "--format=csv,noheader"],
        text=True, capture_output=True, check=False,
    )
    _need(result.returncode == 0,
          f"cannot inspect replay GPU compute owners: {result.stderr.strip()}")
    owners: list[dict[str, Any]] = []
    for fields in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if not fields or not any(value.strip() for value in fields):
            continue
        _need(len(fields) == 2 and fields[0].strip().isdigit(),
              f"malformed nvidia-smi compute-app row: {fields!r}")
        owners.append({"pid": int(fields[0].strip()), "process_name": fields[1].strip()})
    return owners


def _active_upstream_producers() -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in {os.getpid(), os.getppid()}:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        lowered = command.lower()
        producer = any(marker in lowered for marker in (
            "h1_carrierid_date_lodo_phase2", "h1_date_lodo_future",
            "h1_carrierid_date_lodo_hs_", "h1_carrierid_date_lodo_hc_",
        ))
        writer = any(marker in lowered for marker in ("train.py", "copy-import-all-and-finalize", "source-only e49"))
        if producer and writer:
            active.append({"pid": int(entry.name), "command": command})
    return active


def _check_receipt(path: Path, *, schema: str, status: str) -> tuple[Path, dict[str, Any], str]:
    return read_immutable_json(path, schema=schema, status=status)


def _check_chain_stage(
    *, stage: str, outputs: Mapping[str, Path], plan: Mapping[str, Any],
    observed: dict[str, dict[str, str]],
) -> None:
    """Validate each receipt's immediate inputs, not only schema/status."""

    if stage == "post_upstream_binder":
        path, body, digest = _check_receipt(
            outputs["binder"], schema=POST_BINDER_SCHEMA, status=POST_BINDER_STATUS,
        )
        _need(tuple(body.get("fixed_grid", ())) == DATES
              and body.get("all_five_dates_compatible") is True
              and body.get("upstream_aggregate") == plan["upstream_aggregate"]
              and body.get("early_launch_receipt") == plan["launch_receipt"],
              "post-upstream binder lost the frozen five-date/upstream/launch binding")
        observed["binder"] = {"path": str(path), "sha256": digest}
        return
    if stage == "terminal_checker":
        path, body, digest = _check_receipt(
            outputs["checker"], schema=CHECKER_SCHEMA, status=CHECKER_STATUS,
        )
        required = body.get("required_target_execution")
        _need(tuple(body.get("fixed_grid", ())) == DATES
              and body.get("post_upstream_binder", {}).get("path") == observed["binder"]["path"]
              and body.get("post_upstream_binder", {}).get("sha256") == observed["binder"]["sha256"]
              and body.get("upstream_aggregate") == plan["upstream_aggregate"]
              and isinstance(required, Mapping) and required.get("owner") ==
              "remote5070ti_original_hc_replay"
              and required.get("device_type") == "cuda"
              and required.get("device_name_must_contain") == "5070 Ti",
              "terminal checker lost the binder or remote-5070 execution binding")
        observed["checker"] = {"path": str(path), "sha256": digest}
        return
    if stage == "target_evaluator_preflight":
        path, body, digest = _check_receipt(
            outputs["readiness"], schema=EVALUATOR_PREFLIGHT_SCHEMA,
            status=EVALUATOR_PREFLIGHT_STATUS,
        )
        _need(body.get("source_terminal_aggregate") == observed["checker"]
              and body.get("not_a_target_evaluator") is True,
              "target readiness lost its source-terminal checker binding")
        observed["readiness"] = {"path": str(path), "sha256": digest}
        return
    if stage == "target_evaluator_closure":
        path, body, digest = _check_receipt(
            outputs["closure"], schema=CLOSURE_SCHEMA, status=CLOSURE_STATUS,
        )
        _need(tuple(body.get("fixed_grid", ())) == DATES
              and body.get("comparison") == "H-C minus H-LS"
              and body.get("evaluator_preflight") == observed["readiness"]
              and body.get("source_terminal_aggregate") == observed["checker"],
              "target evaluator closure lost the frozen readiness/checker binding")
        observed["closure"] = {"path": str(path), "sha256": digest}
        return
    raise Remote5070ReplayError(f"unexpected non-target chain stage: {stage}")


def _check_evaluation(path: Path, *, date: str, plan: Mapping[str, Any], gpu_name: str) -> dict[str, Any]:
    receipt_path, body, digest = _check_receipt(
        path, schema=EVALUATION_SCHEMA,
        status=f"PASS_H1_CARRIERID_DATE_LODO_HLS_{date}_HC_HLS_EVALUATED",
    )
    reproduction = body.get("original_hc_reproduction_check")
    metrics = body.get("metrics")
    delta = metrics.get("h_c_minus_h_ls") if isinstance(metrics, Mapping) else None
    _need(body.get("outer_date") == date and body.get("device") == "cuda"
          and body.get("execution_owner") == "remote5070ti_original_hc_replay"
          and "5070 Ti" in str(body.get("execution_device_name", ""))
          and body.get("one_shot", {}).get("canonical_output_path") == str(receipt_path)
          and body.get("one_shot", {}).get("same_date_prior_hls_terminal_evaluation_receipts") == 0,
          f"paired evaluation is not canonical remote5070 one-shot evidence: {date}")
    _need(isinstance(reproduction, Mapping) and reproduction.get("passed") is True
          and reproduction.get("required_before_hls_paired_result_publication") is True,
          f"original H-C reproduction did not pass before paired receipt acceptance: {date}")
    _need(isinstance(delta, (int, float)) and math.isfinite(float(delta)),
          f"paired signed delta is missing/nonfinite: {date}")
    _need(body.get("deployment_updates") == {
        "optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged": True,
    }, f"paired evaluator records deployment updates: {date}")
    target_scope = {
        "formal_heldout_opened": False,
        "minival_opened": False,
        "evalai_opened": False,
    }
    _need(body.get("scope") == target_scope,
          f"paired evaluator opened or failed to prove closure of a protected endpoint: {date}")
    _need(body.get("evaluator_closure", {}).get("path") == plan["outputs"]["closure"]
          and body.get("upstream_aggregate") == plan["upstream_aggregate"],
          f"paired evaluator receipt closure/upstream drift: {date}")
    _need("5070 Ti" in gpu_name, "postcheck lost the admitted 5070 Ti device")
    return {"path": str(receipt_path), "sha256": digest, "signed_h_c_minus_h_ls": float(delta),
            "target_scope": target_scope,
            "original_h_c_reproduction_passed_before_publication_and_acceptance": True,
            "does_not_claim_reproduction_validation_precedes_hls_compute": True}


def execute(plan: Mapping[str, Any], *, gpu_device: str = "0") -> dict[str, Any]:
    outputs = {key: Path(value) for key, value in plan["outputs"].items()}
    for label, path in (*outputs.items(), *((date, Path(path)) for date, path in plan["evaluations"].items())):
        _need(not os.path.lexists(str(path)), f"one-shot output appeared after dry run: {label}: {path}")
    _need(not _active_upstream_producers(), "remote H-S/H-C producer/finalizer is still active")
    gpu_name = _gpu_name(str(gpu_device))
    compute_owners = _gpu_compute_owners(str(gpu_device))
    _need(not compute_owners,
          f"remote 5070 Ti already has compute owners before replay claim: {compute_owners}")
    transfer_path, transfer, transfer_sha = _validate_transfer_receipt(
        Path(plan["transfer_receipt"]["path"]), repo_root=Path(plan["repo_root"]),
    )
    _need(transfer_sha == plan["transfer_receipt"]["sha256"], "transfer receipt changed after dry run")
    upstream_path, _upstream, upstream_sha = _validate_upstream(
        transfer, repo_root=Path(plan["repo_root"]),
    )
    _need({"path": str(upstream_path), "sha256": upstream_sha} == plan["upstream_aggregate"],
          "upstream aggregate changed after dry run")
    claim = {
        "schema": REPLAY_SCHEMA, "status": REPLAY_CLAIM_STATUS,
        "transfer_receipt": {"path": str(transfer_path), "sha256": transfer_sha},
        "upstream_aggregate": plan["upstream_aggregate"], "fixed_date_order": list(DATES),
        "gpu_device": str(gpu_device), "gpu_name": gpu_name,
        "scope": {"target_opened_before_claim": 0, "cleanup_on_failure": False,
                  "gpu_compute_owners_before_claim": compute_owners},
    }
    claim_path, claim_sha = write_immutable_json(outputs["claim"], claim)
    accepted: dict[str, Any] = {}
    observed: dict[str, dict[str, str]] = {}
    try:
        for row in plan["commands"]:
            stage, command = str(row["stage"]), list(row["command"])
            environment = None
            if row.get("target") is True:
                _need(not _active_upstream_producers(),
                      f"remote H-S/H-C producer/finalizer appeared before target stage: {stage}")
                compute_owners = _gpu_compute_owners(str(gpu_device))
                _need(not compute_owners,
                      f"remote 5070 Ti has compute owners before target stage {stage}: {compute_owners}")
                environment = dict(os.environ); environment["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
            returncode = _run_streamed(
                command, log=outputs["log"], cwd=Path(plan["repo_root"]), env=environment,
            )
            _need(returncode == 0, f"remote5070 replay stage failed: {stage}: rc={returncode}")
            if stage in {
                "post_upstream_binder", "terminal_checker",
                "target_evaluator_preflight", "target_evaluator_closure",
            }:
                _check_chain_stage(stage=stage, outputs=outputs, plan=plan, observed=observed)
            elif stage.startswith("paired_evaluation_"):
                date = str(row["outer_date"])
                accepted[date] = _check_evaluation(
                    Path(plan["evaluations"][date]), date=date, plan=plan, gpu_name=gpu_name,
                )
                _append_log(outputs["log"], "PAIRED_DATE_ACCEPTED_WITHOUT_SIGN_GATE",
                            outer_date=date, signed_delta=accepted[date]["signed_h_c_minus_h_ls"])
            elif stage == "receipt_only_aggregate":
                _need(tuple(accepted) == DATES,
                      "aggregate cannot run before all five reproduction-gated paired receipts")
                _path, aggregate, _sha = _check_receipt(
                    outputs["aggregate"], schema=AGGREGATE_SCHEMA, status=AGGREGATE_STATUS,
                )
                summary = aggregate.get("summary")
                per_date = aggregate.get("per_date")
                _need(tuple(aggregate.get("required_outer_dates", ())) == DATES
                      and aggregate.get("all_five_dates_reported") is True
                      and isinstance(summary, Mapping) and summary.get("n_outer_dates") == 5
                      and isinstance(per_date, Mapping) and tuple(per_date) == DATES
                      and all(per_date[date].get("receipt") == {
                          "path": accepted[date]["path"], "sha256": accepted[date]["sha256"],
                      } and math.isclose(
                          float(per_date[date].get("h_c_minus_h_ls")),
                          accepted[date]["signed_h_c_minus_h_ls"], rel_tol=0.0, abs_tol=1e-12,
                      ) for date in DATES)
                      and sum(int(summary.get(key, -99)) for key in (
                          "positive_date_count", "negative_date_count", "zero_date_count")) == 5,
                      "final aggregate lost the complete signed five-date grid")
        _need(tuple(accepted) == DATES, "remote replay did not accept all five fixed dates")
        final_transfer_path, final_transfer, final_transfer_sha = _validate_transfer_receipt(
            Path(plan["transfer_receipt"]["path"]), repo_root=Path(plan["repo_root"]),
        )
        final_upstream_path, _final_upstream, final_upstream_sha = _validate_upstream(
            final_transfer, repo_root=Path(plan["repo_root"]),
        )
        _need({"path": str(final_transfer_path), "sha256": final_transfer_sha}
              == plan["transfer_receipt"]
              and {"path": str(final_upstream_path), "sha256": final_upstream_sha}
              == plan["upstream_aggregate"],
              "transfer/upstream closure changed during the five-date target sequence")
        terminal = {
            "schema": REPLAY_SCHEMA, "status": REPLAY_PASS_STATUS,
            "claim": {"path": str(claim_path), "sha256": claim_sha},
            "transfer_receipt": plan["transfer_receipt"], "upstream_aggregate": plan["upstream_aggregate"],
            "gpu": {"physical_device": str(gpu_device), "name": gpu_name},
            "fixed_date_order": list(DATES), "accepted_paired_evaluations": accepted,
            "aggregate": {"path": str(outputs["aggregate"]), "sha256": sha256_file(outputs["aggregate"])},
            "result_policy": {"signed_deltas_preserved": True, "intermediate_sign_gate": False,
                              "route_selection": False, "all_five_dates_required": True},
            "atomic_evaluator_semantics": {
                "compute_order": "H-C forward; H-LS forward; original-H-C reproduction validation",
                "publication_and_acceptance_gate": "original_hc_reproduction_check PASS",
                "does_not_claim_reproduction_validation_precedes_hls_compute": True,
            },
            "scope": {"optimizer_steps": 0, "backward_steps": 0,
                      "cleanup_performed": False, "formal_test_opened": False,
                      "all_accepted_evaluation_target_scopes_verified": True},
        }
        terminal_path, terminal_sha = write_immutable_json(outputs["terminal"], terminal)
        return {"status": REPLAY_PASS_STATUS, "terminal_path": str(terminal_path),
                "terminal_sha256": terminal_sha, "aggregate": terminal["aggregate"]}
    except BaseException as error:
        _append_log(outputs["log"], "REPLAY_FAILED", error_type=type(error).__name__, error=str(error))
        if not os.path.lexists(str(outputs["terminal"])):
            write_immutable_json(outputs["terminal"], {
                "schema": REPLAY_SCHEMA, "status": REPLAY_FAIL_STATUS,
                "claim": {"path": str(claim_path), "sha256": claim_sha},
                "accepted_paired_evaluations": accepted,
                "error": {"type": type(error).__name__, "message": str(error)},
                "preservation": {"existing_canonical_receipts_deleted": False,
                                 "claim_deleted": False, "log_deleted": False,
                                 "cleanup_performed": False},
            })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-receipt", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--gpu-device", default="0")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--execute-remote5070-replay", action="store_true")
    args = parser.parse_args()
    plan = build_plan(transfer_receipt=args.transfer_receipt, data_dir=args.data_dir,
                      python_executable=args.python_executable)
    if not args.execute_remote5070_replay:
        print(json.dumps(plan, sort_keys=True))
        return
    print(json.dumps(execute(plan, gpu_device=args.gpu_device), sort_keys=True))


if __name__ == "__main__":
    main()
