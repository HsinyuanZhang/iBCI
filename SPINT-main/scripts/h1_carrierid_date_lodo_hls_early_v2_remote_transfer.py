#!/usr/bin/env python3
"""Verified, no-overwrite transfer of early-v2 H-LS source evidence to 5070 Ti.

The default path is local-only and prints a dry-run manifest.  Explicit
execution first proves that the remote five-date H-S/H-C producer has sealed
its immutable aggregate, then transfers only missing files with rsync
``--ignore-existing``.  Every existing or imported file must match the local
SHA-256 and size.  The tool never invokes the binder, checker, evaluator, a
target loader, or a GPU process.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    DATES,
    EARLY_LAUNCH_SCHEMA,
    EARLY_LAUNCH_STATUS,
    EARLY_PREFLIGHT_SCHEMA,
    EARLY_PREFLIGHT_STATUS,
    EARLY_TERMINAL_SCHEMA,
    EARLY_TERMINAL_STATUS,
    read_immutable_json,
    sha256_file,
    write_immutable_json,
)
from scripts.h1_carrierid_date_lodo_hls_fivedate_contract import (
    UPSTREAM_AGGREGATE_DEFAULT,
    UPSTREAM_AGGREGATE_SCHEMA,
    UPSTREAM_AGGREGATE_STATUS,
)
from scripts.h1_carrierid_date_lodo_hls_target_evaluator_closure import CLOSURE_FILES


REMOTE = "xinyuan@100.103.97.12"
TRANSFER_SCHEMA = "h1_carrierid_date_lodo_hls_early_v2_remote5070_transfer_v1"
TRANSFER_STATUS = "PASS_H1_HLS_EARLY_V2_REMOTE5070_SOURCE_CLOSURE_IMPORTED_NO_TARGET"
WRITER_CLAIM_SCHEMA = "h1_carrierid_date_lodo_hls_early_v2_writer_claim_v1"
WRITER_CLAIM_STATUS = "CLAIMED_BEFORE_SOURCE_PROCESS_START"

# CLOSURE_FILES are hashed by the future evaluator closure.  The extra files
# are needed to produce the remote binder/checker/readiness/closure receipts.
CODE_FILES = tuple(dict.fromkeys((*CLOSURE_FILES, *(
    "scripts/h1_carrierid_date_lodo_hls_fivedate_contract.py",
    "scripts/h1_carrierid_date_lodo_hls_fivedate_terminal_checker.py",
    "scripts/h1_carrierid_date_lodo_hls_fivedate_target_evaluator_preflight.py",
    "scripts/h1_carrierid_date_lodo_hls_target_evaluator_closure.py",
    "scripts/h1_carrierid_date_lodo_hls_fivedate_aggregate.py",
    "scripts/h1_carrierid_date_lodo_hls_early_v2_remote5070_replay.py",
    "scripts/h1_carrierid_date_lodo_hls_early_v2_remote_transfer.py",
    "src/h1_m4_cce_contract.py",
))))


class RemoteTransferError(RuntimeError):
    """A local closure, remote main-chain, or import invariant failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RemoteTransferError(message)


def _inside_repo(path: str | Path, repo_root: Path, label: str) -> Path:
    raw = Path(path)
    _need(not raw.is_symlink(), f"{label} cannot be a symlink: {raw}")
    candidate = raw.resolve()
    try:
        candidate.relative_to(repo_root)
    except ValueError as error:
        raise RemoteTransferError(f"{label} is outside the identical absolute repo root: {candidate}") from error
    _need("\n" not in str(candidate) and "\x00" not in str(candidate), f"unsafe path in {label}")
    return candidate


def _local_file(path: str | Path, *, repo_root: Path, label: str,
                expected_sha256: str | None = None, immutable: bool = False) -> dict[str, Any]:
    candidate = _inside_repo(path, repo_root, label)
    _need(candidate.is_file() and not candidate.is_symlink(), f"missing/non-regular {label}: {candidate}")
    mode = stat.S_IMODE(candidate.stat().st_mode)
    if immutable:
        _need(mode == 0o444, f"{label} must be immutable mode 0444: {candidate}")
    digest = sha256_file(candidate)
    if expected_sha256 is not None:
        _need(digest == expected_sha256, f"{label} SHA-256 differs from its receipt: {candidate}")
    return {"path": str(candidate), "relative_path": str(candidate.relative_to(repo_root)),
            "sha256": digest, "size": candidate.stat().st_size, "mode": mode, "label": label}


def _add_unique(rows: dict[str, dict[str, Any]], row: Mapping[str, Any]) -> None:
    path = str(row["path"])
    if path in rows:
        _need(rows[path]["sha256"] == row["sha256"] and rows[path]["size"] == row["size"],
              f"same closure path has conflicting local identity: {path}")
        return
    rows[path] = dict(row)


def _receipt_dependency(row: Mapping[str, Any], *, repo_root: Path, label: str) -> dict[str, Any]:
    _need(isinstance(row.get("path"), str) and isinstance(row.get("sha256"), str),
          f"{label} lacks path/SHA binding")
    return _local_file(row["path"], repo_root=repo_root, label=label,
                       expected_sha256=row["sha256"], immutable=True)


def build_plan(
    *, launch_receipt: Path, source_terminals: Mapping[str, Path], output: Path,
    upstream_aggregate: Path = UPSTREAM_AGGREGATE_DEFAULT, repo_root: Path = ROOT,
    code_files: Sequence[str] = CODE_FILES,
) -> dict[str, Any]:
    """Build the complete local manifest without SSH, rsync, data, or CUDA."""

    repo_root = Path(repo_root).resolve()
    _need(repo_root.is_dir() and not repo_root.is_symlink(), f"local repo root is invalid: {repo_root}")
    output = _inside_repo(output, repo_root, "transfer receipt output")
    _need(not os.path.lexists(str(output)), f"transfer receipt already exists: {output}")
    _need(not Path(launch_receipt).is_symlink(), "early-v2 launch receipt cannot be a symlink")
    launch_path, launch, launch_sha = read_immutable_json(
        launch_receipt, schema=EARLY_LAUNCH_SCHEMA, status=EARLY_LAUNCH_STATUS,
    )
    _inside_repo(launch_path, repo_root, "early-v2 launch receipt")
    _need(tuple(launch.get("fixed_grid", ())) == DATES and tuple(source_terminals) == DATES,
          "remote transfer requires the exact ordered five-date early-v2 terminal grid")

    transfer: dict[str, dict[str, Any]] = {}
    prerequisites: dict[str, dict[str, Any]] = {}
    _add_unique(transfer, _local_file(launch_path, repo_root=repo_root,
                                     label="early-v2 launch receipt", expected_sha256=launch_sha,
                                     immutable=True))
    date_rows: dict[str, Any] = {}
    for date in DATES:
        _need(not Path(source_terminals[date]).is_symlink(),
              f"early-v2 source terminal cannot be a symlink: {date}")
        terminal_path, terminal, terminal_sha = read_immutable_json(
            source_terminals[date], schema=EARLY_TERMINAL_SCHEMA, status=EARLY_TERMINAL_STATUS,
        )
        _inside_repo(terminal_path, repo_root, f"{date} source terminal")
        _need(terminal.get("outer_date") == date and terminal.get("arm") == "H-LS",
              f"early-v2 source terminal arm/date drift: {date}")
        scope, updates = terminal.get("scope"), terminal.get("deployment_updates")
        _need(isinstance(scope, Mapping) and scope.get("target_recordings_opened") == 0
              and scope.get("target_bytes_read") == 0
              and isinstance(updates, Mapping) and updates.get("target_optimizer_steps") == 0
              and updates.get("target_backward_steps") == 0
              and updates.get("target_model_state_updated") is False,
              f"source terminal is not target-closed: {date}")
        launch_row = launch.get("source_preflights", {}).get(date)
        _need(isinstance(launch_row, Mapping), f"launch receipt lacks source preflight: {date}")
        preflight_path, preflight, preflight_sha = read_immutable_json(
            launch_row.get("path", ""), schema=EARLY_PREFLIGHT_SCHEMA, status=EARLY_PREFLIGHT_STATUS,
        )
        _need(preflight_sha == launch_row.get("sha256") and preflight.get("outer_date") == date
              and terminal.get("source_preflight") == {"path": str(preflight_path), "sha256": preflight_sha},
              f"terminal/launch/source-preflight binding drift: {date}")
        claim_row = terminal.get("writer_claim")
        claim_path, claim, claim_sha = read_immutable_json(
            claim_row.get("path", "") if isinstance(claim_row, Mapping) else "",
            schema=WRITER_CLAIM_SCHEMA, status=WRITER_CLAIM_STATUS,
        )
        _need(claim_sha == claim_row.get("sha256") and claim.get("outer_date") == date,
              f"writer claim/terminal binding drift: {date}")
        checkpoint = terminal.get("checkpoint")
        _need(isinstance(checkpoint, Mapping), f"source terminal lacks checkpoint row: {date}")
        _need(isinstance(checkpoint.get("sha256"), str)
              and isinstance(checkpoint.get("config_sha256"), str),
              f"source terminal lacks checkpoint/config SHA binding: {date}")
        checkpoint_row = _local_file(
            checkpoint.get("path", ""), repo_root=repo_root, label=f"{date} H-LS e49 checkpoint",
            expected_sha256=checkpoint.get("sha256"),
        )
        config_row = _local_file(
            checkpoint.get("config_path", ""), repo_root=repo_root, label=f"{date} resolved config",
            expected_sha256=checkpoint.get("config_sha256"),
        )
        for item in (
            _local_file(terminal_path, repo_root=repo_root, label=f"{date} source terminal",
                        expected_sha256=terminal_sha, immutable=True),
            _local_file(preflight_path, repo_root=repo_root, label=f"{date} source preflight",
                        expected_sha256=preflight_sha, immutable=True),
            _local_file(claim_path, repo_root=repo_root, label=f"{date} writer claim",
                        expected_sha256=claim_sha, immutable=True), checkpoint_row, config_row,
        ):
            _add_unique(transfer, item)

        # These are producer/base artifacts.  They must already exist and be
        # identical remotely; this tool never transfers them.
        for label, dependency in (
            (f"{date} matched pair preflight", preflight.get("matched_h_c_pair_preflight")),
            (f"{date} Phase-1 preflight", preflight.get("phase1_preflight")),
            (f"{date} sealed null audit", preflight.get("null_strength_audit")),
            (f"{date} waiting plan", preflight.get("waiting_plan")),
        ):
            _need(isinstance(dependency, Mapping), f"source preflight lacks {label}")
            _add_unique(prerequisites, _receipt_dependency(dependency, repo_root=repo_root, label=label))
        source = preflight.get("source_binding")
        _need(isinstance(source, Mapping) and isinstance(source.get("source_manifest_path"), str)
              and isinstance(source.get("source_manifest_sha256"), str),
              f"source preflight lacks source manifest binding: {date}")
        source_manifest_row = _local_file(
            source["source_manifest_path"], repo_root=repo_root,
            label=f"{date} Phase-1 source manifest", expected_sha256=source["source_manifest_sha256"],
            immutable=True,
        )
        _add_unique(prerequisites, source_manifest_row)
        try:
            source_manifest = json.loads(Path(source_manifest_row["path"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RemoteTransferError(f"invalid Phase-1 source manifest: {date}") from error
        frozen_plan, normalizer = source_manifest.get("frozen_plan"), source_manifest.get("normalizer")
        _need(isinstance(frozen_plan, Mapping) and isinstance(normalizer, Mapping),
              f"Phase-1 source manifest lacks target replay dependencies: {date}")
        plan_manifest = _local_file(
            frozen_plan.get("manifest_path", ""), repo_root=repo_root,
            label=f"{date} frozen plan manifest", expected_sha256=frozen_plan.get("manifest_sha256"),
            immutable=True,
        )
        _add_unique(prerequisites, plan_manifest)
        _add_unique(prerequisites, _local_file(
            Path(plan_manifest["path"]).with_name("frozen_m4_plan.npz"), repo_root=repo_root,
            label=f"{date} frozen plan arrays", immutable=True,
        ))
        _add_unique(prerequisites, _local_file(
            normalizer.get("manifest_path", ""), repo_root=repo_root,
            label=f"{date} source normalizer manifest",
            expected_sha256=normalizer.get("manifest_file_sha256"), immutable=True,
        ))
        date_rows[date] = {
            "terminal": {"path": str(terminal_path), "sha256": terminal_sha},
            "checkpoint": checkpoint_row, "config": config_row,
            "source_preflight": {"path": str(preflight_path), "sha256": preflight_sha},
            "writer_claim": {"path": str(claim_path), "sha256": claim_sha},
        }

    for relative in code_files:
        _need(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
              f"code closure entry is not repo-relative: {relative}")
        _add_unique(transfer, _local_file(repo_root / relative, repo_root=repo_root,
                                         label=f"code closure:{relative}"))
    return {
        "schema": TRANSFER_SCHEMA, "status": "DRY_RUN_LOCAL_CLOSURE_READY_NOT_TRANSFERRED",
        "remote": REMOTE, "local_repo_root": str(repo_root), "remote_repo_root": str(repo_root),
        "remote_upstream_gate": {
            "path": str(_inside_repo(upstream_aggregate, repo_root, "remote upstream aggregate path")),
            "schema": UPSTREAM_AGGREGATE_SCHEMA, "status": UPSTREAM_AGGREGATE_STATUS,
            "required_dates": list(DATES),
        },
        "launch_receipt": {"path": str(launch_path), "sha256": launch_sha},
        "dates": date_rows, "transfer_files": list(transfer.values()),
        "remote_existing_prerequisites": list(prerequisites.values()),
        "transfer_receipt_output": str(output),
        "policy": {"rsync_only_missing": True, "refuse_existing_mismatch": True,
                   "pre_and_post_sha256_and_size": True, "remote_main_chain_must_be_complete": True,
                   "binder_called": False, "checker_called": False, "evaluator_called": False,
                   "target_opened": 0, "gpu_started": False},
    }


REMOTE_INSPECT_CODE = r'''import hashlib,json,os,stat,sys
def sha(p):
 d=hashlib.sha256()
 with open(p,"rb") as f:
  while True:
   b=f.read(1048576)
   if not b: break
   d.update(b)
 return d.hexdigest()
def state(p):
 if not os.path.lexists(p): return {"state":"MISSING"}
 if os.path.islink(p): return {"state":"INVALID","reason":"SYMLINK"}
 if not os.path.isfile(p): return {"state":"INVALID","reason":"NOT_REGULAR"}
 s=os.stat(p)
 return {"state":"FILE","size":s.st_size,"sha256":sha(p),"mode":stat.S_IMODE(s.st_mode)}
q=json.load(sys.stdin); root=q["repo_root"]; result={"repo_root":os.path.realpath(root),"files":{}}
active=[]
for entry in os.listdir("/proc"):
 if not entry.isdigit() or int(entry) in (os.getpid(),os.getppid()): continue
 try: cmd=open("/proc/"+entry+"/cmdline","rb").read().replace(b"\0",b" ").decode("utf-8","replace")
 except OSError: continue
 low=cmd.lower()
 producer=any(x in low for x in ("h1_carrierid_date_lodo_phase2","h1_date_lodo_future","h1_carrierid_date_lodo_hs_","h1_carrierid_date_lodo_hc_"))
 writer=any(x in low for x in ("train.py","copy-import-all-and-finalize","source-only e49"))
 if producer and writer: active.append({"pid":int(entry),"command":cmd})
result["active_main_chain_producers"]=active
gate_path=q["upstream"]["path"]; gs=state(gate_path); gate={"ready":False,"path":gate_path,"file":gs}
try:
 if os.path.realpath(root)!=root: raise ValueError("REMOTE_REPO_ROOT_MISMATCH")
 if gs.get("state")!="FILE" or gs.get("mode")!=0o444: raise ValueError("UPSTREAM_MISSING_OR_MUTABLE")
 b=json.load(open(gate_path));
 if b.get("schema")!=q["upstream"]["schema"] or b.get("status")!=q["upstream"]["status"]: raise ValueError("UPSTREAM_SCHEMA_STATUS")
 if b.get("required_outer_dates")!=q["dates"] or b.get("all_five_date_receipts_present_and_validated") is not True: raise ValueError("UPSTREAM_INCOMPLETE")
 route=b.get("route_prerequisite",{})
 if route.get("status")!="source/date screen complete" or route.get("automatic_route_selection")!="FORBIDDEN": raise ValueError("UPSTREAM_ROUTE")
 originals={}
 for date in q["dates"]:
  row=b.get("per_date",{}).get(date,{}).get("receipt",{}); p=row.get("path",""); s=state(p)
  if s.get("state")!="FILE" or s.get("mode")!=0o444 or s.get("sha256")!=row.get("sha256"): raise ValueError("ORIGINAL_RECEIPT_"+date)
  ob=json.load(open(p)); hc=ob.get("checkpoints",{}).get("H-C",{}); cp=state(hc.get("path","")); cf=state(hc.get("config_path",""))
  if ob.get("schema")!="h1_carrierid_date_lodo_phase2_terminal_evaluation_v1" or ob.get("status")!="PASS_H1_CARRIERID_DATE_LODO_PHASE2_"+date+"_HS_HC_EVALUATED" or ob.get("outer_date")!=date: raise ValueError("ORIGINAL_SCHEMA_STATUS_"+date)
  if cp.get("state")!="FILE" or cf.get("state")!="FILE" or not isinstance(hc.get("sha256"),str) or not isinstance(hc.get("config_sha256"),str) or cp.get("sha256")!=hc.get("sha256") or cf.get("sha256")!=hc.get("config_sha256"): raise ValueError("HC_CLOSURE_"+date)
  originals[date]={"receipt":s,"checkpoint":cp,"config":cf}
 gate={"ready":True,"path":gate_path,"sha256":gs["sha256"],"size":gs["size"],"original_hc":originals}
except Exception as e: gate["reason"]=str(e)
result["upstream_gate"]=gate
for p in q["paths"]: result["files"][p]=state(p)
print(json.dumps(result,sort_keys=True))'''


def _remote_python_command(code: str) -> str:
    """Quote one Python program for OpenSSH's remote-shell command string."""

    return f"python3 -c {shlex.quote(code)}"


def _remote_inspect(*, remote: str, plan: Mapping[str, Any], paths: Iterable[str]) -> dict[str, Any]:
    request = {"repo_root": plan["remote_repo_root"], "upstream": plan["remote_upstream_gate"],
               "dates": list(DATES), "paths": list(paths)}
    # OpenSSH concatenates arguments after the host and passes the result to a
    # remote shell.  Supplying python3/-c/code as separate local argv entries
    # loses quoting for spaces/newlines.  One explicitly quoted command is the
    # actual remote protocol boundary.
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               remote, _remote_python_command(REMOTE_INSPECT_CODE)]
    result = subprocess.run(command, input=json.dumps(request), text=True,
                            capture_output=True, check=False)
    _need(result.returncode == 0,
          f"remote read-only inspection failed: rc={result.returncode}: {result.stderr.strip()}")
    try:
        body = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RemoteTransferError("remote inspection returned invalid JSON") from error
    _need(isinstance(body, Mapping) and body.get("repo_root") == plan["remote_repo_root"],
          "remote repository is not at the identical absolute path")
    return dict(body)


def _matches(state: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    return (state.get("state") == "FILE" and state.get("sha256") == row.get("sha256")
            and state.get("size") == row.get("size") and state.get("mode") == row.get("mode"))


def _verify_gate(inspection: Mapping[str, Any], expected_sha: str | None = None) -> str:
    active = inspection.get("active_main_chain_producers")
    _need(isinstance(active, list) and not active,
          f"remote H-S/H-C producer/finalizer is still active; refusing all writes: {active}")
    gate = inspection.get("upstream_gate")
    _need(isinstance(gate, Mapping) and gate.get("ready") is True,
          f"remote five-date main chain is not complete: {gate.get('reason') if isinstance(gate, Mapping) else gate}")
    digest = str(gate.get("sha256", ""))
    _need(len(digest) == 64, "remote upstream gate has no SHA-256")
    if expected_sha is not None:
        _need(digest == expected_sha, "remote upstream aggregate changed during transfer")
    return digest


def _rsync_missing(*, remote: str, paths: Sequence[str]) -> None:
    if not paths:
        return
    # Absolute-path imports use ``remote:/`` as the destination so that the
    # transferred path remains byte-for-byte identical.  rsync otherwise
    # attempts to preserve the mtime of each parent directory (including
    # ``/home``), which is not writable for a normal account and makes a
    # successful no-overwrite import return code 23.  Preserve regular-file
    # metadata but omit directory mtimes; the receipt verifies file size/SHA/
    # mode after the copy.
    command = ["rsync", "-rlpt", "--omit-dir-times", "--relative", "--ignore-existing", "--protect-args",
               "--", *paths, f"{remote}:/"]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    _need(result.returncode == 0,
          f"no-overwrite rsync failed: rc={result.returncode}: {result.stderr.strip()}")


def _local_still_matches(row: Mapping[str, Any]) -> None:
    path = Path(str(row["path"]))
    _need(path.is_file() and not path.is_symlink() and path.stat().st_size == row["size"]
          and sha256_file(path) == row["sha256"],
          f"local transfer file changed after dry-run manifest: {path}")


def probe_remote_read_only(plan: Mapping[str, Any], *, remote: str = REMOTE) -> dict[str, Any]:
    """Run the complete remote admission check without writing or invoking rsync."""

    _need(remote == REMOTE and plan.get("remote") == REMOTE,
          f"transfer destination is fixed to {REMOTE}")
    transfer_rows = {str(row["path"]): row for row in plan["transfer_files"]}
    prerequisite_rows = {str(row["path"]): row for row in plan["remote_existing_prerequisites"]}
    output = str(plan["transfer_receipt_output"])
    inspection = _remote_inspect(
        remote=remote, plan=plan, paths=[*transfer_rows, *prerequisite_rows, output],
    )
    gate_sha = _verify_gate(inspection)
    states = inspection.get("files", {})
    _need(isinstance(states, Mapping), "remote inspection lacks file states")
    _need(states.get(output, {}).get("state") == "MISSING",
          "remote transfer receipt path already exists")
    for path, row in prerequisite_rows.items():
        _need(_matches(states.get(path, {}), row),
              f"remote producer/base prerequisite is missing or differs; refusing transfer: {path}")
    missing = 0
    identical = 0
    for path, row in transfer_rows.items():
        state = states.get(path, {})
        if state.get("state") == "MISSING":
            missing += 1
        else:
            _need(_matches(state, row), f"remote existing file differs; refusing overwrite: {path}")
            identical += 1
    return {"status": "PASS_REMOTE_READ_ONLY_PROBE_NO_TRANSFER",
            "remote_upstream_sha256": gate_sha, "missing_transfer_files": missing,
            "already_identical": identical, "remote_writes": 0, "rsync_invoked": False}


def execute(plan: Mapping[str, Any], *, remote: str = REMOTE) -> dict[str, Any]:
    _need(remote == REMOTE and plan.get("remote") == REMOTE,
          f"transfer destination is fixed to {REMOTE}")
    transfer_rows = {str(row["path"]): row for row in plan["transfer_files"]}
    prerequisite_rows = {str(row["path"]): row for row in plan["remote_existing_prerequisites"]}
    output = str(plan["transfer_receipt_output"])
    _need(not os.path.lexists(output), f"transfer receipt appeared after dry-run: {output}")
    for row in transfer_rows.values():
        _local_still_matches(row)
    inspect_paths = [*transfer_rows, *prerequisite_rows, output]
    before = _remote_inspect(remote=remote, plan=plan, paths=inspect_paths)
    gate_sha = _verify_gate(before)
    states = before.get("files", {})
    _need(isinstance(states, Mapping), "remote inspection lacks file states")
    _need(states.get(output, {}).get("state") == "MISSING",
          "remote transfer receipt path already exists")
    missing: list[str] = []
    for path, row in prerequisite_rows.items():
        _need(_matches(states.get(path, {}), row),
              f"remote producer/base prerequisite is missing or differs; refusing transfer: {path}")
    for path, row in transfer_rows.items():
        state = states.get(path, {})
        if state.get("state") == "MISSING":
            missing.append(path)
        else:
            _need(_matches(state, row), f"remote existing file differs; refusing overwrite: {path}")
    _rsync_missing(remote=remote, paths=missing)

    after = _remote_inspect(remote=remote, plan=plan, paths=inspect_paths)
    _verify_gate(after, gate_sha)
    after_states = after.get("files", {})
    for path, row in {**prerequisite_rows, **transfer_rows}.items():
        _need(_matches(after_states.get(path, {}), row), f"remote post-transfer verification failed: {path}")
    _need(after_states.get(output, {}).get("state") == "MISSING",
          "remote transfer receipt path raced with another handoff")

    receipt = {
        "schema": TRANSFER_SCHEMA, "status": TRANSFER_STATUS,
        "remote": remote, "identical_absolute_repo_root": plan["remote_repo_root"],
        "remote_upstream_aggregate": {"path": plan["remote_upstream_gate"]["path"],
                                      "sha256": gate_sha, "five_date_complete": True},
        "launch_receipt": plan["launch_receipt"], "dates": plan["dates"],
        "files": {"total": len(transfer_rows), "transferred_missing": len(missing),
                  "already_identical": len(transfer_rows) - len(missing),
                  "manifest": list(transfer_rows.values())},
        "remote_prerequisites": list(prerequisite_rows.values()),
        "verification": {"preflight_sha256_and_size": True, "postflight_sha256_and_size": True,
                         "existing_mismatch_overwritten": False, "rsync_ignore_existing": True},
        "scope": {"binder_called": False, "checker_called": False, "evaluator_called": False,
                  "target_opened": 0, "gpu_started": False,
                  "remote_active_producer_paths_written": False},
        "code_sha256": {"remote_transfer": sha256_file(Path(__file__).resolve())},
    }
    receipt_path, receipt_sha = write_immutable_json(output, receipt)
    receipt_row = {"path": str(receipt_path), "sha256": receipt_sha,
                   "size": receipt_path.stat().st_size, "mode": 0o444}
    _rsync_missing(remote=remote, paths=[str(receipt_path)])
    final = _remote_inspect(remote=remote, plan=plan, paths=[str(receipt_path)])
    _verify_gate(final, gate_sha)
    _need(_matches(final.get("files", {}).get(str(receipt_path), {}), receipt_row),
          "remote transfer receipt import verification failed")
    return {"status": TRANSFER_STATUS, "receipt_path": str(receipt_path),
            "receipt_sha256": receipt_sha, "transferred_missing": len(missing),
            "already_identical": len(transfer_rows) - len(missing)}


def _parse_date_paths(values: Sequence[str]) -> dict[str, Path]:
    rows: dict[str, Path] = {}
    for value in values:
        date, separator, path = value.partition("=")
        if not separator or date in rows:
            raise SystemExit("--source-terminal requires unique DATE=PATH values")
        rows[date] = Path(path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-receipt", required=True, type=Path)
    parser.add_argument("--source-terminal", action="append", required=True, metavar="DATE=PATH")
    parser.add_argument("--upstream-aggregate", type=Path, default=UPSTREAM_AGGREGATE_DEFAULT,
                        help="same absolute path inspected remotely; this local file is not opened")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--remote", default=REMOTE, choices=(REMOTE,))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--probe-remote-read-only", action="store_true")
    action.add_argument("--execute-transfer", action="store_true")
    args = parser.parse_args()
    plan = build_plan(launch_receipt=args.launch_receipt,
                      source_terminals=_parse_date_paths(args.source_terminal),
                      output=args.output, upstream_aggregate=args.upstream_aggregate)
    if args.probe_remote_read_only:
        print(json.dumps(probe_remote_read_only(plan, remote=args.remote), sort_keys=True))
        return
    if not args.execute_transfer:
        print(json.dumps(plan, sort_keys=True))
        return
    print(json.dumps(execute(plan, remote=args.remote), sort_keys=True))


if __name__ == "__main__":
    main()
