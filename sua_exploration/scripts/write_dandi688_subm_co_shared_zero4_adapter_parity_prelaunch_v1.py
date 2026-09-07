#!/usr/bin/env python3
"""Write/verify the append-only consumed-sub-C zero4 adapter parity package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "sua_exploration/results/dandi_000688_subm_co_shared_zero4_adapter_parity_prelaunch_v1"
)
SOURCE_PATHS = (
    "sua_exploration/docs/DANDI_000688_SUBM_CO_SHARED_ZERO4_ADAPTER_PARITY_PROTOCOL_V1.md",
    "sua_exploration/mc_maze/subm_co_shared_zero4_adapter_parity_v1.py",
    "sua_exploration/scripts/run_dandi688_subm_co_shared_zero4_adapter_parity_v1.py",
    "sua_exploration/scripts/write_dandi688_subm_co_shared_zero4_adapter_parity_prelaunch_v1.py",
    "sua_exploration/tests/test_dandi688_subm_co_shared_zero4_adapter_parity_v1.py",
)
CONSUMED_SUBC = {
    "path": "sua_exploration/data/dandi_000688/sub-C/sub-C_ses-CO-20151103_behavior+ecephys.nwb",
    "session": "sub-C_ses-CO-20151103",
    "sha256": "7770b3c4ae13fde65276af4d67bcd157e878e7fa4b75d858ba92d90309e14de7",
    "bytes": 62145872,
}
DEPENDENCY_SOURCE_PINS = {
    "sua_exploration/mc_maze/paired_view_c1_shared_zero4.py":
        "ee754cb279fae1a72e88b51b1f11f94b22380e953dde972ef36eabcaf4295f88",
    "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v5.py":
        "563f095bdfab38e5df4cd6087e9dac27f2fb58a81190e0b9ad920215fba7adb4",
    "sua_exploration/mc_maze/multisession_datamodule.py":
        "674fb4c235ba8f9393a6d1614f1f6f4260177ed9751e88acb4c05c4396d81e2d",
    "sua_exploration/mc_maze/datamodule.py":
        "0c93359991c32e81b552e00169fa1f31a5b71d345782c9bf81b136bd5c708506",
    "sua_exploration/scripts/eval_adaptation_dandi688.py":
        "e452d19d738316a2ff54074585b22e526bb1cc9275bfcfe5d33aa1becc5ccc30",
    "sua_exploration/scripts/dandi688_gradient_free_protocol.py":
        "a0d1b331c0548a967bbd08804231c2045a360e8947f869dea2d8c251e4d70e68",
}


class StaticZero4ParityError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StaticZero4ParityError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value), sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _regular_under(root: Path, relative: str, label: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise StaticZero4ParityError(f"{label} escapes repository: {relative}") from exc
    require(candidate.is_file() and not candidate.is_symlink(), f"missing/unsafe {label}: {relative}")
    return candidate


def _current_source_map(root: Path) -> dict[str, str]:
    return {
        relative: sha256_file(_regular_under(root, relative, "V1 source"))
        for relative in SOURCE_PATHS
    }


def _dependency_source_map(root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in DEPENDENCY_SOURCE_PINS.items():
        digest = sha256_file(_regular_under(root, relative, "dependency source"))
        require(digest == expected, f"dependency source SHA drift: {relative}")
        observed[relative] = digest
    return observed


def static_authority(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    fixture = _regular_under(root, str(CONSUMED_SUBC["path"]), "consumed sub-C fixture")
    require(fixture.stat().st_size == CONSUMED_SUBC["bytes"], "consumed sub-C byte-size drift")
    require(sha256_file(fixture) == CONSUMED_SUBC["sha256"], "consumed sub-C SHA drift")
    return {
        "source_map": _current_source_map(root),
        "dependency_source_pins": _dependency_source_map(root),
        "fixed_fixture": dict(CONSUMED_SUBC),
        "matrix": {
            "views": ["sua", "pseudo_mua"],
            "label_variants": ["original", "label_shuffle", "label_drop"],
            "activity_first_n": 30,
            "t4_comparator_pool_and_query_boundary": 50,
        },
        "permitted_operation": "consumed_subc_shared_zero4_input_parity_only",
        "forbidden_operations": [
            "external_subm_access",
            "external_subm_scoring",
            "checkpoint_open",
            "model_forward",
            "r2_computation",
            "gpu_use",
            "external_scoring_authorization_or_signature_creation",
        ],
    }


def build_draft(root: Path = ROOT) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "kind": "dandi_000688_consumed_subc_shared_zero4_adapter_input_parity_prelaunch_v1",
        "status": "STATIC_CONSUMED_SUBC_ZERO4_INPUT_PARITY_READY_NO_EXTERNAL_CAPABILITY",
        "append_only": True,
        "authority": static_authority(root),
        "operations_by_prelaunch": {
            "consumed_subc_hashed": True,
            "consumed_subc_nwb_loaded": False,
            "external_subm_accessed": False,
            "external_subm_scored": False,
            "checkpoint_files_opened": 0,
            "model_forward_calls": 0,
            "r2_computations": 0,
            "gpu_used": False,
        },
        "external_scoring_capability_created": False,
    }


def build_receipt(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    draft = build_draft(root)
    draft_sha = hashlib.sha256(canonical_bytes(draft)).hexdigest()
    receipt = {
        "schema_version": "1",
        "kind": "dandi_000688_consumed_subc_shared_zero4_adapter_input_parity_prelaunch_receipt_v1",
        "status": draft["status"],
        "append_only": True,
        "draft": {"path": "parity_prelaunch.json", "sha256": draft_sha},
        "operations": dict(draft["operations_by_prelaunch"]),
        "scope_boundary": {
            "only_fixed_consumed_subc_fixture": True,
            "no_caller_data_path": True,
            "external_scoring_capability_created": False,
        },
    }
    return draft, receipt


def _write_immutable(path: Path, value: Mapping[str, Any]) -> str:
    raw = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    require(stat.S_IMODE(path.stat().st_mode) == 0o444, "immutable mode failed")
    return hashlib.sha256(raw).hexdigest()


def write_prelaunch(output_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    require(not output_dir.exists(), "zero4 parity prelaunch root already exists")
    draft, receipt = build_receipt(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    draft_path = output_dir / "parity_prelaunch.json"
    receipt_path = output_dir / "receipt.json"
    seal_path = output_dir / "seal.json"
    draft_sha = _write_immutable(draft_path, draft)
    require(receipt["draft"]["sha256"] == draft_sha, "draft/receipt binding drift")
    receipt_sha = _write_immutable(receipt_path, receipt)
    seal = {
        "schema_version": "1",
        "kind": "dandi_000688_consumed_subc_shared_zero4_adapter_input_parity_prelaunch_seal_v1",
        "status": receipt["status"],
        "append_only": True,
        "artifacts": [
            {
                "path": draft_path.name,
                "sha256": draft_sha,
                "bytes": draft_path.stat().st_size,
                "mode": "0444",
            },
            {
                "path": receipt_path.name,
                "sha256": receipt_sha,
                "bytes": receipt_path.stat().st_size,
                "mode": "0444",
            },
        ],
        "external_scoring_capability_created": False,
    }
    seal_sha = _write_immutable(seal_path, seal)
    return {
        "output_dir": str(output_dir),
        "draft_sha256": draft_sha,
        "receipt_sha256": receipt_sha,
        "seal_sha256": seal_sha,
        "status": receipt["status"],
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing/unsafe {label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticZero4ParityError(f"cannot parse {label}") from exc
    require(isinstance(value, dict), f"{label} is not an object")
    return value


def load_stored_prelaunch(prelaunch_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    prelaunch_dir = prelaunch_dir.resolve()
    draft_path = prelaunch_dir / "parity_prelaunch.json"
    receipt_path = prelaunch_dir / "receipt.json"
    seal_path = prelaunch_dir / "seal.json"
    for path in (draft_path, receipt_path, seal_path):
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444,
            f"stored zero4 parity artifact unsafe: {path.name}",
        )
    draft = _read_json(draft_path, "zero4 parity draft")
    receipt = _read_json(receipt_path, "zero4 parity receipt")
    seal = _read_json(seal_path, "zero4 parity seal")
    draft_sha = sha256_file(draft_path)
    receipt_sha = sha256_file(receipt_path)
    seal_sha = sha256_file(seal_path)
    require(draft.get("authority") == static_authority(root), "stored parity authority drift")
    require(
        receipt.get("draft") == {"path": draft_path.name, "sha256": draft_sha},
        "stored draft/receipt binding drift",
    )
    require(
        seal.get("artifacts")
        == [
            {
                "path": draft_path.name,
                "sha256": draft_sha,
                "bytes": draft_path.stat().st_size,
                "mode": "0444",
            },
            {
                "path": receipt_path.name,
                "sha256": receipt_sha,
                "bytes": receipt_path.stat().st_size,
                "mode": "0444",
            },
        ],
        "stored seal artifact binding drift",
    )
    require(
        draft.get("external_scoring_capability_created") is False
        and seal.get("external_scoring_capability_created") is False,
        "stored parity package gained external scoring capability",
    )
    return {
        "draft_sha256": draft_sha,
        "receipt_sha256": receipt_sha,
        "seal_sha256": seal_sha,
        "status": receipt.get("status"),
        "authority": draft["authority"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = (
            static_authority(args.repo_root)
            if args.verify_only
            else write_prelaunch(args.output_dir, args.repo_root)
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except StaticZero4ParityError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

