#!/usr/bin/env python3
"""Read-only v2 remote-sync closure audit with the pinned SPINT Conda Python.

This is intentionally a new entrypoint.  It preserves the v1 script and the
v1/audit_v3 receipt as an append-only ``ENVIRONMENT_PROBE_WRONG_INTERPRETER``
incident: audit_v3 correctly established the 60-file inventory result, but
used the host's /usr/bin/python3 instead of the designated SPINT interpreter.

The local QAT receipt and recursively bound local files remain the only
authority.  Remote digests are observations, compared to that authority only.
This script writes only its new local receipt; over SSH it performs stat/read
hashing of the sealed 60-item list, imports torch in the required environment,
and issues read-only nvidia-smi and tmux list-sessions queries.  It never
writes remotely, starts tmux/GPU work, uses rsync/scp, or constructs/accesses a
formal-session NWB path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LEGACY_SCRIPT_REL = "sua_exploration/scripts/audit_t4_paired_view_c1_encoder_qat_remote_sync.py"
LEGACY_SCRIPT_SHA256 = "1883b26b4a8c7061565a8d83ae992eeb809268b67e05316e491ffa200d224ea9"
V3_RECEIPT_REL = "sua_exploration/results/t4_paired_view_c1_encoder_qat_remote_sync_audit_v1_20260805/audit_v3.json"
V3_RECEIPT_SHA256 = "96586a54d40454172e691fca9e8e7faab81e5ea1d1094b1c83ae9fe23fcf9506"
REQUIRED_REMOTE_PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
DEFAULT_OUT = ROOT / "sua_exploration/results/t4_paired_view_c1_encoder_qat_remote_sync_audit_v2_20260805/receipt.json"


class AuditError(RuntimeError):
    """A local authority, legacy-incident, or interpreter-policy failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot parse {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def load_legacy_authority_module() -> ModuleType:
    """Load precisely the preserved v1 code after pinning its content hash."""
    path = ROOT / LEGACY_SCRIPT_REL
    require(path.is_file(), f"preserved v1 script missing: {path}")
    require(sha256_file(path) == LEGACY_SCRIPT_SHA256, "preserved v1 script SHA drift")
    spec = importlib.util.spec_from_file_location("c1_qat_remote_sync_v1_preserved", path)
    require(spec is not None and spec.loader is not None, "cannot load preserved v1 authority code")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_v3_incident() -> dict[str, Any]:
    """Bind the v2 receipt to v3 without changing that immutable incident."""
    path = ROOT / V3_RECEIPT_REL
    require(path.is_file(), f"v3 incident receipt missing: {path}")
    require(sha256_file(path) == V3_RECEIPT_SHA256, "v3 incident receipt SHA drift")
    v3 = read_json(path)
    require(v3.get("status") == "REMOTE_SYNC_NOT_CLOSED", "v3 incident status drift")
    require(v3.get("expected_remote_file_count") == 60, "v3 expected file count drift")
    require(v3.get("expected_remote_source_nwb_count") == 27, "v3 source NWB count drift")
    require(v3.get("expected_remote_development_nwb_count") == 6, "v3 development NWB count drift")
    summary = v3.get("summary")
    require(isinstance(summary, dict), "v3 summary missing")
    expected_states = {"present": 44, "missing": 16, "mismatch": 0, "unstable": 0}
    require(summary.get("artifact_states") == expected_states, "v3 60-file inventory outcome drift")
    require(summary.get("all_expected_files_present_and_exact") is False, "v3 file inventory unexpectedly closed")
    remote = v3.get("remote_observation")
    require(isinstance(remote, dict), "v3 remote observation missing")
    environment = remote.get("environment")
    require(isinstance(environment, dict), "v3 environment observation missing")
    python = environment.get("spint_python")
    require(isinstance(python, dict) and python.get("executable") == "/usr/bin/python3", "v3 wrong-interpreter incident evidence drift")
    torch = environment.get("torch_cuda")
    require(isinstance(torch, dict) and torch.get("state") == "unavailable", "v3 torch incident evidence drift")
    require(remote.get("formal_nwb_paths_constructed") is False, "v3 constructed formal NWB paths")
    require(remote.get("formal_nwb_files_accessed") is False, "v3 accessed formal NWB files")
    return {
        "v3_receipt": {"path": V3_RECEIPT_REL, "sha256": V3_RECEIPT_SHA256, "bytes": path.stat().st_size},
        "classification": "ENVIRONMENT_PROBE_WRONG_INTERPRETER",
        "wrong_remote_python": "/usr/bin/python3",
        "required_remote_python": REQUIRED_REMOTE_PYTHON,
        "superseded_scope": "environment disposition only",
        "preserved_60_file_authority_inventory_outcome": {
            "expected_remote_file_count": 60,
            "source_nwb_count": 27,
            "development_nwb_count": 6,
            "artifact_states": expected_states,
            "all_expected_files_present_and_exact": False,
        },
    }


def probe_remote(module: ModuleType, expected: dict[str, dict[str, Any]], timeout_seconds: float) -> dict[str, Any]:
    """Run the legacy read-only probe body in the one authorized interpreter."""
    request = {"root": module.REMOTE_ROOT, "files": [expected[key] for key in sorted(expected)]}
    code = module.REMOTE_PROBE.replace("__REQUEST_JSON__", repr(request))
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        module.REMOTE,
        REQUIRED_REMOTE_PYTHON,
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            input=code,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"state": "unreachable", "error": f"{type(exc).__name__}: {exc}", "artifacts": [], "environment": {}}
    if completed.returncode != 0:
        return {"state": "unreachable", "returncode": completed.returncode, "stderr": completed.stderr.strip(), "artifacts": [], "environment": {}}
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {
            "state": "unreachable",
            "error": f"invalid remote JSON: {exc}",
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr.strip(),
            "artifacts": [],
            "environment": {},
        }
    require(isinstance(payload, dict), "remote probe JSON root is not an object")
    payload["remote_python_requested"] = REQUIRED_REMOTE_PYTHON
    return payload


def build_receipt(module: ModuleType, authority: dict[str, Any], expected: dict[str, dict[str, Any]], remote: dict[str, Any], incident: dict[str, Any]) -> dict[str, Any]:
    """Keep v1 closure criteria and add the pinned-interpreter condition."""
    receipt = module.build_receipt(authority, expected, remote)
    environment = remote.get("environment") if isinstance(remote.get("environment"), dict) else {}
    python = environment.get("spint_python") if isinstance(environment.get("spint_python"), dict) else {}
    interpreter_ok = python.get("executable") == REQUIRED_REMOTE_PYTHON and remote.get("remote_python_requested") == REQUIRED_REMOTE_PYTHON
    summary = receipt["summary"]
    summary["environment_interpreter_required"] = REQUIRED_REMOTE_PYTHON
    summary["environment_interpreter_ok"] = interpreter_ok
    summary["environment_ok"] = bool(summary["environment_ok"] and interpreter_ok)
    closed = bool(summary["all_expected_files_present_and_exact"] and summary["environment_ok"] and remote.get("formal_nwb_files_accessed") is False)
    receipt["status"] = "REMOTE_SYNC_CLOSED" if closed else "REMOTE_SYNC_NOT_CLOSED"
    receipt["receipt_kind"] = "c1_encoder_only_qat_seed44_exact_remote_sync_closure_audit_v2_pinned_conda_python"
    receipt["interpreter_policy"] = {
        "remote_python": REQUIRED_REMOTE_PYTHON,
        "default_is_pinned_and_immutable": True,
        "forbidden_remote_python": "/usr/bin/python3",
        "remote_writes": 0,
        "gpu_or_tmux_work_started": False,
    }
    receipt["supersedes"] = incident
    receipt["summary"]["closure_rule"] = "only all 60 exact authority-bound files plus the pinned CUDA-visible SPINT interpreter on the expected GPU produce REMOTE_SYNC_CLOSED"
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--remote-python",
        default=REQUIRED_REMOTE_PYTHON,
        help="immutable policy value; only the designated SPINT Conda Python is accepted",
    )
    parser.add_argument("--verify-local-only", action="store_true", help="validate local authority and incident lineage only; do not use SSH or write a receipt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.timeout_seconds > 0, "timeout must be positive")
    require(args.remote_python == REQUIRED_REMOTE_PYTHON, f"forbidden remote Python: {args.remote_python}; required: {REQUIRED_REMOTE_PYTHON}")
    module = load_legacy_authority_module()
    incident = validate_v3_incident()
    authority, expected = module.local_authority()
    require(len(expected) == 60, "local authority expected remote file count drift")
    if args.verify_local_only:
        print(json.dumps({"expected_remote_file_count": len(expected), "incident": incident, "remote_python": REQUIRED_REMOTE_PYTHON}, indent=2, sort_keys=True))
        return
    output = args.out.resolve()
    require(not os.path.lexists(output), f"refusing to overwrite audit receipt: {output}")
    remote = probe_remote(module, expected, args.timeout_seconds)
    receipt = build_receipt(module, authority, expected, remote, incident)
    module.write_once_atomic(output, module.canonical_bytes(receipt))
    print(output)


if __name__ == "__main__":
    try:
        main()
    except (AuditError, RuntimeError) as exc:
        raise SystemExit(f"AUDIT_FAIL_CLOSED: {exc}") from exc
