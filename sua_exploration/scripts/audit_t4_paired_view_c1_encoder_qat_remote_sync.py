#!/usr/bin/env python3
"""Read-only, fail-closed C1 encoder-only QAT seed-44 remote-sync audit.

The only authority is the immutable local QAT v2 prelaunch receipt and its
recursively bound local inputs.  Remote SHA-256 values are observations to be
compared with that authority, never new authority.  This program does not use
rsync/scp, write to the remote host, start tmux/GPU work, or access any formal
test NWB path.  It sends a read-only Python probe over SSH which hashes only
the explicitly manifest-listed 27 source and six development NWB files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
QAT_RECEIPT_REL = "sua_exploration/results/t4_paired_view_c1_encoder_qat_prelaunch_v2_20260805/receipt.json"
QAT_RECEIPT_SHA256 = "ec986e775fa3ca500247442d404df832fb0b1f460d9bb342e7cc1b717720bcdb"
PTQ_DATA_MANIFEST_REL = (
    "sua_exploration/results/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist/"
    "c1_train_val_33_manifest.json"
)
REMOTE = "xinyuan@100.103.97.12"
REMOTE_ROOT = "/home/xinyuan/Work_host/SPINT"
REMOTE_PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
EXPECTED_GPU_UUID = "GPU-a1ede67a-629a-916b-ab17-634135b5147d"
DEFAULT_OUT = (
    ROOT
    / "sua_exploration/results/t4_paired_view_c1_encoder_qat_remote_sync_audit_v1_20260805/"
    "audit_v3.json"
)


class AuditError(RuntimeError):
    """A local authority or scope failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot parse {path}: {exc}") from exc
    require(isinstance(result, dict), f"JSON root is not an object: {path}")
    return result


def resolve_receipt_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        try:
            relative = candidate.relative_to(ROOT)
        except ValueError as exc:
            raise AuditError(f"receipt path is outside expected workspace root: {candidate}") from exc
        candidate = ROOT / relative
    else:
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError as exc:
        raise AuditError(f"resolved path escapes workspace root: {candidate}") from exc
    return candidate


def checked_row(row: dict[str, Any], *, label: str, require_bytes: bool = False) -> tuple[Path, dict[str, Any]]:
    require(isinstance(row, dict), f"{label}: expected a receipt row")
    path = resolve_receipt_path(str(row.get("path", "")))
    expected_sha = row.get("sha256")
    require(isinstance(expected_sha, str) and len(expected_sha) == 64, f"{label}: invalid expected SHA")
    require(path.is_file(), f"{label}: local file missing: {path}")
    observed_sha = sha256_file(path)
    require(observed_sha == expected_sha, f"{label}: local SHA mismatch for {path}")
    if require_bytes or "bytes" in row:
        require(isinstance(row.get("bytes"), int), f"{label}: expected bytes missing")
        require(path.stat().st_size == row["bytes"], f"{label}: local byte mismatch for {path}")
    return path, {
        "path": str(path.relative_to(ROOT)),
        "sha256": observed_sha,
        "bytes": path.stat().st_size,
        "authority": label,
    }


def add_expected(
    entries: dict[str, dict[str, Any]],
    *,
    path: Path,
    sha256: str,
    bytes_count: int | None,
    label: str,
    require_local_file: bool = True,
) -> None:
    if require_local_file:
        require(path.is_file(), f"authority file disappeared: {path}")
    relative = str(path.relative_to(ROOT))
    existing = entries.get(relative)
    candidate = {"path": relative, "sha256": sha256, "bytes": bytes_count, "authority_labels": [label]}
    if existing is None:
        entries[relative] = candidate
        return
    require(existing["sha256"] == sha256, f"conflicting authority SHA for {relative}")
    if existing["bytes"] is not None and bytes_count is not None:
        require(existing["bytes"] == bytes_count, f"conflicting authority bytes for {relative}")
    if existing["bytes"] is None and bytes_count is not None:
        existing["bytes"] = bytes_count
    existing["authority_labels"].append(label)


def validate_strict_manifest(payload: dict[str, Any]) -> dict[str, list[str]]:
    splits = payload.get("session_splits")
    require(isinstance(splits, dict), "strict manifest missing session_splits")
    expected_counts = {"train": 27, "val": 6, "test": 6}
    result: dict[str, list[str]] = {}
    for split, count in expected_counts.items():
        values = splits.get(split)
        require(isinstance(values, list) and len(values) == count, f"strict manifest {split} count drift")
        require(all(isinstance(value, str) for value in values), f"strict manifest {split} names invalid")
        require(len(set(values)) == len(values), f"strict manifest {split} duplicate names")
        result[split] = list(values)
    require(not (set(result["train"]) & set(result["val"])), "train/dev overlap")
    require(not ((set(result["train"]) | set(result["val"])) & set(result["test"])), "opened/formal overlap")
    return result


def validate_ptq_prelaunch(payload: dict[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "status": "authorized_for_c1_shared_encoder_ptq",
        "scope": "C1_shared_B3S_T4_identity_encoder_W8A8_INT32_plus_FP32_decoder",
        "formal_test_files_opened": False,
        "c2_authorized": False,
        "ordinary_t4_int8_selection_receipt_reused": False,
    }
    failed = [key for key, value in expected.items() if payload.get(key) != value]
    require(not failed, f"PTQ prelaunch identity drift: {failed}")
    strict = payload.get("strict_manifest")
    require(isinstance(strict, dict), "PTQ prelaunch strict manifest missing")
    require(strict.get("counts") == [27, 6, 6], "PTQ strict manifest split counts drift")
    require(strict.get("formal_paths_resolved") is False, "PTQ prelaunch resolved formal paths")
    require(isinstance(payload.get("source_code"), dict) and payload["source_code"], "PTQ prelaunch source-code ledger missing")
    require(isinstance(payload.get("teacher"), dict), "PTQ prelaunch teacher row missing")


def local_authority() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Hash every declared authority locally and build the exact remote file set."""
    qat_path = ROOT / QAT_RECEIPT_REL
    require(qat_path.is_file(), f"QAT receipt missing: {qat_path}")
    qat_sha = sha256_file(qat_path)
    require(qat_sha == QAT_RECEIPT_SHA256, "authoritative QAT receipt SHA drift")
    qat = read_json(qat_path)
    require(qat.get("schema_version") == 1, "QAT receipt schema drift")
    require(qat.get("status") == "authorized_uniform_three_seed_c1_encoder_qat", "QAT receipt status drift")
    require(qat.get("formal_test_files_opened") is False, "QAT receipt opened formal files")
    require(qat.get("c2_authorized") is False, "QAT receipt authorizes C2")

    expected: dict[str, dict[str, Any]] = {}
    add_expected(expected, path=qat_path, sha256=qat_sha, bytes_count=qat_path.stat().st_size, label="qat_receipt")

    qat_source = qat.get("source_code")
    require(isinstance(qat_source, dict) and qat_source, "QAT source-code ledger missing")
    local_qat_source: list[dict[str, Any]] = []
    for name, row in sorted(qat_source.items()):
        path, checked = checked_row(row, label=f"qat_source:{name}", require_bytes=True)
        local_qat_source.append(checked)
        add_expected(expected, path=path, sha256=checked["sha256"], bytes_count=checked["bytes"], label=f"qat_source:{name}")

    ptq_path, ptq_checked = checked_row(qat["ptq_prelaunch"], label="qat.ptq_prelaunch", require_bytes=True)
    aggregate_path, aggregate_checked = checked_row(qat["ptq_aggregate"], label="qat.ptq_aggregate", require_bytes=True)
    add_expected(expected, path=ptq_path, sha256=ptq_checked["sha256"], bytes_count=ptq_checked["bytes"], label="ptq_prelaunch")
    add_expected(expected, path=aggregate_path, sha256=aggregate_checked["sha256"], bytes_count=aggregate_checked["bytes"], label="ptq_aggregate")

    aggregate = read_json(aggregate_path)
    expected_aggregate = {
        "schema_version": 1,
        "scope": "C1 shared B3S/T4 identity encoder W8A8 INT32 + FP32 decoder",
        "method": "uniform_three_seed_ptq",
        "seeds": [42, 43, 44],
        "views": ["sua", "pseudo_mua"],
        "prelaunch_sha256": ptq_checked["sha256"],
        "decoder_quantized_in_this_program": False,
        "decoder_precision": "FP32",
        "formal_test_files_opened": False,
        "ptq_pass": False,
        "next_step": "trigger_uniform_three_seed_encoder_qat",
        "qat_triggered_by_this_aggregate": True,
        "mixed_ptq_qat_seed_aggregate_forbidden": True,
    }
    require(not [key for key, value in expected_aggregate.items() if aggregate.get(key) != value], "PTQ aggregate trigger drift")

    seed_inputs = qat.get("seed_inputs")
    require(isinstance(seed_inputs, dict) and set(seed_inputs) == {"42", "43", "44"}, "QAT seed inputs drift")
    seed44 = seed_inputs["44"]
    report_path, report_checked = checked_row(seed44["ptq_report"], label="seed44.ptq_report", require_bytes=True)
    package_path, package_checked = checked_row(seed44["ptq_package"], label="seed44.ptq_package", require_bytes=True)
    checkpoint_path, checkpoint_checked = checked_row(seed44["checkpoint"], label="seed44.source_checkpoint")
    metadata_path, metadata_checked = checked_row(seed44["run_metadata"], label="seed44.run_metadata")
    for checked in (report_checked, package_checked, checkpoint_checked, metadata_checked):
        add_expected(expected, path=resolve_receipt_path(checked["path"]), sha256=checked["sha256"], bytes_count=checked["bytes"], label=checked["authority"])

    report = read_json(report_path)
    require(report.get("seed") == 44 and report.get("method") == "ptq", "seed44 PTQ report identity drift")
    require(report.get("formal_test_files_opened") is False, "seed44 PTQ report opened formal files")
    require(report.get("prelaunch_sha256") == ptq_checked["sha256"], "seed44 PTQ report prelaunch drift")
    package = report.get("integer_package")
    require(isinstance(package, dict), "seed44 PTQ report package missing")
    require(package.get("sha256") == package_checked["sha256"], "seed44 PTQ package report drift")
    require(report.get("checkpoint_sha256") == checkpoint_checked["sha256"], "seed44 checkpoint report drift")
    require(report.get("run_metadata_sha256") == metadata_checked["sha256"], "seed44 metadata report drift")

    metadata = read_json(metadata_path)
    require(metadata.get("seed") == 44, "seed44 run metadata seed drift")
    require(metadata.get("formal_sua_files_opened") is False, "seed44 metadata opened formal SUA")
    require(metadata.get("held_out_test_evaluated") is False, "seed44 metadata evaluated held-out data")

    ptq = read_json(ptq_path)
    validate_ptq_prelaunch(ptq)
    strict_path, strict_checked = checked_row(ptq["strict_manifest"], label="ptq.strict_manifest", require_bytes=True)
    teacher_path, teacher_checked = checked_row(ptq["teacher"], label="ptq.teacher", require_bytes=True)
    add_expected(expected, path=strict_path, sha256=strict_checked["sha256"], bytes_count=strict_checked["bytes"], label="ptq.strict_manifest")
    add_expected(expected, path=teacher_path, sha256=teacher_checked["sha256"], bytes_count=teacher_checked["bytes"], label="ptq.teacher")
    strict_splits = validate_strict_manifest(read_json(strict_path))
    require(ptq["strict_manifest"].get("source_sessions") == strict_splits["train"], "PTQ source-session sequence drift")
    require(ptq["strict_manifest"].get("development_sessions") == strict_splits["val"], "PTQ dev-session sequence drift")
    require(ptq["strict_manifest"].get("sealed_formal_session_names") == strict_splits["test"], "PTQ sealed formal names drift")

    local_ptq_source: list[dict[str, Any]] = []
    for name, row in sorted(ptq["source_code"].items()):
        path, checked = checked_row(row, label=f"ptq_source:{name}", require_bytes=True)
        local_ptq_source.append(checked)
        add_expected(expected, path=path, sha256=checked["sha256"], bytes_count=checked["bytes"], label=f"ptq_source:{name}")

    # The C1 run metadata, which QAT binds through seed44, binds this portable
    # 33-file inventory.  It intentionally has no formal NWB paths or hashes.
    require(metadata.get("train_val_manifest_sha256") == strict_checked["sha256"], "seed44 strict-manifest binding drift")
    require(metadata.get("teacher_sha256") == teacher_checked["sha256"], "seed44 teacher binding drift")
    data_manifest_path = ROOT / PTQ_DATA_MANIFEST_REL
    require(data_manifest_path.is_file(), "C1 33-file data manifest missing")
    data_manifest_sha = sha256_file(data_manifest_path)
    require(metadata.get("data_manifest_sha256") == data_manifest_sha, "seed44 33-file manifest binding drift")
    data_manifest = read_json(data_manifest_path)
    require(data_manifest.get("schema_version") == 1, "C1 data manifest schema drift")
    require(data_manifest.get("purpose") == "C1 exact portable train+validation data inventory: 27 source + 6 reused-development NWB files; formal test files are neither listed as paths nor hashed", "C1 data manifest purpose drift")
    require(data_manifest.get("file_count") == 33, "C1 data manifest file count drift")
    require(data_manifest.get("formal_test_paths_resolved") is False, "C1 data manifest resolved formal paths")
    require(data_manifest.get("formal_test_file_paths") == [], "C1 data manifest contains formal paths")
    require(data_manifest.get("formal_test_file_hashes") == [], "C1 data manifest contains formal hashes")
    require(data_manifest.get("source_manifest_sha256") == strict_checked["sha256"], "C1 data manifest strict binding drift")
    inventory = data_manifest.get("file_inventory")
    require(isinstance(inventory, dict) and set(inventory) == {"train", "val"}, "C1 data inventory split drift")
    require(data_manifest.get("session_splits", {}).get("train") == strict_splits["train"], "C1 data train order drift")
    require(data_manifest.get("session_splits", {}).get("val") == strict_splits["val"], "C1 data dev order drift")
    data_rows: list[dict[str, Any]] = []
    for split, expected_sessions in (("train", strict_splits["train"]), ("val", strict_splits["val"])):
        rows = inventory.get(split)
        require(isinstance(rows, list) and len(rows) == len(expected_sessions), f"C1 data {split} row count drift")
        require([row.get("session") for row in rows] == expected_sessions, f"C1 data {split} order drift")
        for row in rows:
            filename = row.get("filename")
            expected_name = f"{row['session']}_behavior+ecephys.nwb"
            require(filename == expected_name, f"C1 data filename drift: {row.get('session')}")
            require(isinstance(row.get("sha256"), str) and len(row["sha256"]) == 64, "C1 data SHA missing")
            require(isinstance(row.get("bytes"), int) and row["bytes"] > 0, "C1 data byte count missing")
            data_rows.append({"path": f"sua_exploration/data/dandi_000688/sub-C/{filename}", "sha256": row["sha256"], "bytes": row["bytes"], "split": split, "session": row["session"]})
    require(len(data_rows) == 33, "C1 data row total drift")
    add_expected(expected, path=data_manifest_path, sha256=data_manifest_sha, bytes_count=data_manifest_path.stat().st_size, label="seed44_bound_33_file_data_manifest")
    for row in data_rows:
        # This is the only NWB path set sent to remote; no formal path is constructed.
        add_expected(
            expected,
            path=ROOT / row["path"],
            sha256=row["sha256"],
            bytes_count=row["bytes"],
            label=f"{row['split']}_nwb:{row['session']}",
            require_local_file=False,
        )

    authority = {
        "qat_receipt": {"path": QAT_RECEIPT_REL, "sha256": qat_sha, "bytes": qat_path.stat().st_size},
        "qat_source_code": local_qat_source,
        "ptq_prelaunch": ptq_checked,
        "ptq_aggregate": aggregate_checked,
        "seed44": {"ptq_report": report_checked, "ptq_package": package_checked, "source_checkpoint": checkpoint_checked, "run_metadata": metadata_checked},
        "ptq_recursive": {"strict_manifest": strict_checked, "teacher": teacher_checked, "source_code": local_ptq_source, "data_manifest": {"path": str(data_manifest_path.relative_to(ROOT)), "sha256": data_manifest_sha, "bytes": data_manifest_path.stat().st_size}, "source_nwb_count": 27, "development_nwb_count": 6, "formal_session_names_sealed_only": strict_splits["test"], "formal_nwb_paths_constructed": False, "formal_nwb_files_accessed": False},
    }
    return authority, expected


REMOTE_PROBE = r'''
import hashlib, json, os, pathlib, subprocess, sys, traceback
REQUEST = __REQUEST_JSON__
ROOT = pathlib.Path(REQUEST["root"]).resolve()
result = {"probe_kind": "read_only_remote_sync_observation", "remote_root": str(ROOT), "formal_nwb_paths_constructed": False, "formal_nwb_files_accessed": False, "artifacts": [], "environment": {}}
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
for expected in REQUEST["files"]:
    rel = expected["path"]
    candidate = (ROOT / rel).resolve()
    row = {"path": rel, "authority_labels": expected["authority_labels"], "expected_sha256": expected["sha256"], "expected_bytes": expected["bytes"]}
    try:
        candidate.relative_to(ROOT)
    except ValueError:
        row.update({"state": "mismatch", "reason": "path_escapes_remote_root"})
        result["artifacts"].append(row)
        continue
    try:
        before = candidate.stat()
    except FileNotFoundError:
        row.update({"state": "missing"})
        result["artifacts"].append(row)
        continue
    except OSError as exc:
        row.update({"state": "mismatch", "reason": f"stat_error:{type(exc).__name__}:{exc}"})
        result["artifacts"].append(row)
        continue
    row["observed_bytes_before"] = before.st_size
    if not candidate.is_file():
        row.update({"state": "mismatch", "reason": "not_regular_file"})
        result["artifacts"].append(row)
        continue
    if expected["bytes"] is not None and before.st_size != expected["bytes"]:
        row.update({"state": "mismatch", "reason": "byte_count", "observed_bytes_after": before.st_size})
        result["artifacts"].append(row)
        continue
    try:
        observed_sha = digest(candidate)
        after = candidate.stat()
    except OSError as exc:
        row.update({"state": "mismatch", "reason": f"hash_error:{type(exc).__name__}:{exc}"})
        result["artifacts"].append(row)
        continue
    row.update({"observed_sha256": observed_sha, "observed_bytes_after": after.st_size})
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        row.update({"state": "unstable", "reason": "file_changed_while_hashing"})
    elif observed_sha != expected["sha256"]:
        row.update({"state": "mismatch", "reason": "sha256"})
    else:
        row.update({"state": "present"})
    result["artifacts"].append(row)
try:
    os.chdir(ROOT)
    result["environment"]["spint_python"] = {"executable": sys.executable, "cwd": os.getcwd(), "python_version": sys.version}
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        result["environment"]["torch_cuda"] = {"torch_version": torch.__version__, "torch_cuda_build": torch.version.cuda, "cuda_available": cuda_available, "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0}
    except Exception as exc:
        result["environment"]["torch_cuda"] = {"state": "unavailable", "error": f"{type(exc).__name__}:{exc}"}
    smi = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    result["environment"]["nvidia_smi"] = {"returncode": smi.returncode, "stdout": smi.stdout.strip().splitlines(), "stderr": smi.stderr.strip()}
    tmux = subprocess.run(["tmux", "list-sessions"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    result["environment"]["tmux"] = {"returncode": tmux.returncode, "stdout": tmux.stdout.strip().splitlines(), "stderr": tmux.stderr.strip(), "read_only_query": True}
except Exception as exc:
    result["environment"]["probe_error"] = f"{type(exc).__name__}:{exc}"
print(json.dumps(result, sort_keys=True))
'''


def probe_remote(expected: dict[str, dict[str, Any]], timeout_seconds: float) -> dict[str, Any]:
    request = {"root": REMOTE_ROOT, "files": [expected[key] for key in sorted(expected)]}
    # The probe embeds a Python literal, not a JSON string: it receives no
    # untrusted remote input and performs no remote writes.
    code = REMOTE_PROBE.replace("__REQUEST_JSON__", repr(request))
    command = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        REMOTE, REMOTE_PYTHON, "-",
    ]
    try:
        completed = subprocess.run(command, input=code, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_seconds, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"state": "unreachable", "error": f"{type(exc).__name__}: {exc}", "artifacts": [], "environment": {}}
    if completed.returncode != 0:
        return {"state": "unreachable", "returncode": completed.returncode, "stderr": completed.stderr.strip(), "artifacts": [], "environment": {}}
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {"state": "unreachable", "error": f"invalid remote JSON: {exc}", "stdout": completed.stdout[-2000:], "stderr": completed.stderr.strip(), "artifacts": [], "environment": {}}
    require(isinstance(payload, dict), "remote probe JSON root is not an object")
    return payload


def write_once_atomic(path: Path, body: bytes) -> None:
    path = path.resolve()
    require(not os.path.lexists(path), f"refusing to overwrite audit receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_receipt(authority: dict[str, Any], expected: dict[str, dict[str, Any]], remote: dict[str, Any]) -> dict[str, Any]:
    artifacts = remote.get("artifacts") if isinstance(remote.get("artifacts"), list) else []
    states = {"present": 0, "missing": 0, "mismatch": 0, "unstable": 0}
    for row in artifacts:
        if isinstance(row, dict) and row.get("state") in states:
            states[row["state"]] += 1
    environment = remote.get("environment") if isinstance(remote.get("environment"), dict) else {}
    torch_cuda = environment.get("torch_cuda") if isinstance(environment.get("torch_cuda"), dict) else {}
    smi = environment.get("nvidia_smi") if isinstance(environment.get("nvidia_smi"), dict) else {}
    gpu_uuid_present = any(EXPECTED_GPU_UUID in line for line in smi.get("stdout", []) if isinstance(line, str))
    environment_ok = (
        remote.get("remote_root") == REMOTE_ROOT
        and environment.get("spint_python", {}).get("executable") == REMOTE_PYTHON
        and environment.get("spint_python", {}).get("cwd") == REMOTE_ROOT
        and torch_cuda.get("cuda_available") is True
        and gpu_uuid_present
    )
    all_files_present = len(artifacts) == len(expected) and states["present"] == len(expected)
    closed = all_files_present and environment_ok and remote.get("formal_nwb_files_accessed") is False
    return {
        "schema_version": 1,
        "receipt_kind": "c1_encoder_only_qat_seed44_exact_remote_sync_closure_audit",
        "status": "REMOTE_SYNC_CLOSED" if closed else "REMOTE_SYNC_NOT_CLOSED",
        "authority_not_remote_recomputed_hashes": True,
        "remote_write_operations": 0,
        "rsync_or_scp_used": False,
        "gpu_work_started": False,
        "tmux_started": False,
        "formal_nwb_paths_constructed": False,
        "formal_nwb_files_accessed": False,
        "authority": authority,
        "expected_remote_file_count": len(expected),
        "expected_remote_source_nwb_count": 27,
        "expected_remote_development_nwb_count": 6,
        "sealed_formal_session_names_only": authority["ptq_recursive"]["formal_session_names_sealed_only"],
        "remote_observation": remote,
        "summary": {
            "artifact_states": states,
            "all_expected_files_present_and_exact": all_files_present,
            "environment_ok": environment_ok,
            "expected_gpu_uuid": EXPECTED_GPU_UUID,
            "expected_remote_python": REMOTE_PYTHON,
            "observed_expected_gpu_uuid": gpu_uuid_present,
            "tmux_checked_read_only": isinstance(environment.get("tmux"), dict),
            "remote_sync_closed_rule": "only exact authority-bound file matches plus CUDA-visible expected GPU produce REMOTE_SYNC_CLOSED",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--verify-local-only", action="store_true", help="validate local authority only; no SSH probe or receipt write")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.timeout_seconds > 0, "timeout must be positive")
    authority, expected = local_authority()
    if args.verify_local_only:
        print(json.dumps({"authority": authority, "expected_remote_file_count": len(expected)}, indent=2, sort_keys=True))
        return
    output = args.out.resolve()
    require(not os.path.lexists(output), f"refusing to overwrite audit receipt: {output}")
    remote = probe_remote(expected, args.timeout_seconds)
    receipt = build_receipt(authority, expected, remote)
    write_once_atomic(output, canonical_bytes(receipt))
    print(output)


if __name__ == "__main__":
    try:
        main()
    except AuditError as exc:
        raise SystemExit(f"AUDIT_FAIL_CLOSED: {exc}") from exc
