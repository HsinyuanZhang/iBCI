#!/usr/bin/env python3
"""Copy a completed fold-2 state from its accidental path without overwriting.

The fold-2 runner was launched through a remote shell where ``$USER`` was
expanded inside the ``--state`` argument.  This utility preserves that
original immutable state, imports an identical byte copy only when the
canonical results path is absent, and records the path-recovery provenance in
an append-only receipt.  It never retrains a cell and never replaces an
existing state or receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


SCHEMA = "m1_compact_b3s_f2_s42_state_path_recovery_v1"
STATUS = "PASS_M1_COMPACT_B3S_F2_S42_STATE_PATH_RECOVERED"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"missing/symlinked path: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def immutable(path: Path, body: Mapping[str, Any]) -> str:
    need(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(body)
    value["canonical_content_sha256"] = canonical(value)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, 0o444)
        os.link(temp, path)
        temp.unlink()
    finally:
        if temp.exists():
            temp.unlink()
    return sha(path)


def recover(original: Path, canonical_path: Path, receipt: Path, runner_argv: list[str]) -> dict[str, Any]:
    original = original.resolve()
    canonical_path = canonical_path.resolve()
    receipt = receipt.resolve()
    need(original.is_file() and not original.is_symlink(), f"original state missing: {original}")
    need(original.stat().st_mode & 0o777 == 0o444, "original state must already be immutable")
    need(not canonical_path.exists() and not canonical_path.is_symlink(), f"canonical state already exists: {canonical_path}")
    need(not receipt.exists() and not receipt.is_symlink(), f"recovery receipt already exists: {receipt}")
    state = json.loads(original.read_text(encoding="utf-8"))
    need(state.get("schema") == "m1_compact_b3s_f2_s42_execution_v2", "state schema drift")
    need(
        state.get("canonical_content_sha256")
        == canonical({k: v for k, v in state.items() if k != "canonical_content_sha256"}),
        "original state canonical hash drift",
    )
    original_sha = sha(original)

    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{canonical_path.name}.", dir=str(canonical_path.parent))
    temp = Path(name)
    try:
        with original.open("rb") as source, os.fdopen(fd, "wb") as target:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        os.chmod(temp, 0o444)
        # Hard-link installation fails if another writer creates the canonical
        # path, so this operation cannot silently overwrite a state.
        os.link(temp, canonical_path)
        temp.unlink()
    finally:
        if temp.exists():
            temp.unlink()
    canonical_sha = sha(canonical_path)
    need(canonical_sha == original_sha, "canonical copy SHA differs from original")
    need(canonical_path.stat().st_mode & 0o777 == 0o444, "canonical state is not immutable")

    terminals = state.get("terminal_checkpoints", {})
    body = {
        "schema": SCHEMA,
        "status": STATUS,
        "method": {
            "original_preserved": True,
            "copy_only_if_canonical_absent": True,
            "copy_strategy": "copy_to_temp_fsync_then_hard_link; no overwrite",
            "retrained": False,
        },
        "original_state": {
            "path": str(original),
            "sha256": original_sha,
            "mode": oct(original.stat().st_mode & 0o777),
        },
        "canonical_state": {
            "path": str(canonical_path),
            "sha256": canonical_sha,
            "mode": oct(canonical_path.stat().st_mode & 0o777),
            "byte_identical_to_original": canonical_sha == original_sha,
        },
        "runner": {
            "argv": runner_argv,
            "state_argument_after_remote_shell_expansion": str(original),
            "shell_expansion": {
                "token": "$USER",
                "expanded_value": "xinyuan",
                "cause": "remote single-quoted command was interpreted by the remote shell before tmux command creation",
            },
        },
        "pair_binding": {
            "receipt_sha256": state.get("receipt_sha256"),
            "proposal_sha256": state.get("proposal_sha256"),
            "fixed_order": state.get("fixed_order"),
            "run_ids": {
                key: str(spec.get("path", "")).split("/runs/")[-1].split("/checkpoints")[0]
                for key, spec in terminals.items()
            },
            "terminal_checkpoint_sha256": {
                key: spec.get("sha256") for key, spec in terminals.items()
            },
            "resolved_config_sha256": {
                key: spec.get("resolved_config", {}).get("sha256") for key, spec in terminals.items()
            },
        },
        "evaluator": {
            "input_path": str(canonical_path),
            "input_sha256": canonical_sha,
            "must_use_canonical_copy": True,
            "target_opened_only_by_cpu_evaluator_after_pair": True,
        },
    }
    receipt_sha = immutable(receipt, body)
    return {"status": STATUS, "original_sha256": original_sha, "canonical_sha256": canonical_sha, "receipt_sha256": receipt_sha}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--runner-argv-json", required=True)
    args = parser.parse_args()
    argv = json.loads(args.runner_argv_json)
    need(isinstance(argv, list) and all(isinstance(item, str) for item in argv), "runner argv must be a JSON string list")
    print(json.dumps(recover(args.original, args.canonical, args.receipt, argv), sort_keys=True))


if __name__ == "__main__":
    main()
