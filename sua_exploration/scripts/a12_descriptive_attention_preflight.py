#!/usr/bin/env python3
"""CPU-only metadata preflight for the A12 descriptive attention audit.

This entrypoint is intentionally incapable of opening an NWB, importing
Torch, deserializing a checkpoint, or running a decoder forward.  It binds
only the canonical SUA M30 T4/Z4 artifact pair:

* a SHA-qualified *historical* T4 M30 reference; and
* the v10 Z4 component-attribution arm.

That is a same-unit, same-seed development comparison, but it is not claimed
to be a freshly matched common training run.  The emitted receipt preserves
that lineage asymmetry and remains non-authorizing for any forward probe.

By default this command performs a full no-write dry run, including all 48
epoch-checkpoint SHA-256 checks.  ``--write-operational-receipt`` is an
explicit, write-once action and can publish only to the one canonical path.
"""
from __future__ import annotations

# These assignments deliberately precede every project import.  The module
# below is standard-library-only, and this script never imports a data/model
# helper; the explicit runtime assertion further prevents accidental drift.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
if str(SUA_ROOT) not in sys.path:
    sys.path.insert(0, str(SUA_ROOT))

from mc_maze import a12_descriptive_attention_audit as core  # noqa: E402


CANONICAL_RECEIPT_PATH = (REPO_ROOT / core.CANONICAL_PREFLIGHT_RELATIVE_PATH).resolve()
_FORBIDDEN_IMPORT_PREFIXES = ("torch", "pynwb", "lightning", "src.models")


def _is_forbidden_module(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in _FORBIDDEN_IMPORT_PREFIXES)


def assert_metadata_only_runtime() -> None:
    """Fail before audit work if this process has crossed a model/data boundary."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise core.A12AuditError("A12 metadata preflight requires CUDA_VISIBLE_DEVICES=''" )
    forbidden = sorted(name for name in sys.modules if _is_forbidden_module(name))
    if forbidden:
        raise core.A12AuditError(
            "A12 metadata preflight forbids model/data imports: " + ", ".join(forbidden)
        )


def canonical_receipt_path() -> Path:
    """The only location at which an operational A12 preflight may be published."""

    return CANONICAL_RECEIPT_PATH


def assert_operational_preflight_payload(
    payload: Mapping[str, Any],
    *,
    layout: core.AuditLayout,
) -> None:
    """Reject dry, non-canonical, or partially verified payloads before writing."""

    if not layout.canonical:
        raise core.A12AuditError("operational A12 receipt requires the canonical artifact layout")
    if payload.get("status") != core.PREFLIGHT_PASS_STATUS:
        raise core.A12AuditError("operational A12 receipt requires a passed metadata preflight")
    if payload.get("checkpoint_bytes_verified") is not True:
        raise core.A12AuditError("operational A12 receipt requires all 48 checkpoint SHA checks")
    core.validate_metadata_preflight_receipt(payload, layout=layout)


def write_operational_receipt(
    payload: Mapping[str, Any],
    *,
    layout: core.AuditLayout,
    output_path: Path | None = None,
) -> dict[str, str]:
    """Write a verified payload only to the fixed operational receipt path."""

    assert_operational_preflight_payload(payload, layout=layout)
    requested = canonical_receipt_path() if output_path is None else output_path.expanduser().resolve()
    if requested != canonical_receipt_path():
        raise core.A12AuditError(
            "operational A12 preflight must use its one canonical output path: "
            f"{canonical_receipt_path()}"
        )
    return core.write_immutable_json(requested, payload, label="A12 official metadata preflight")


def run_metadata_preflight() -> tuple[dict[str, Any], core.AuditLayout]:
    """Build the full canonical report without writing or crossing a data/model boundary."""

    assert_metadata_only_runtime()
    layout = core.canonical_layout()
    payload = core.build_metadata_preflight(
        layout=layout,
        verify_checkpoint_bytes=True,
        include_implementation_bindings=True,
    )
    assert_operational_preflight_payload(payload, layout=layout)
    return payload, layout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the canonical receipt inputs and print a no-write summary (the default).",
    )
    mode.add_argument(
        "--write-operational-receipt",
        action="store_true",
        help="Publish the passed receipt to its sole canonical write-once path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload, layout = run_metadata_preflight()
        if args.write_operational_receipt:
            artifact = write_operational_receipt(payload, layout=layout)
            print(
                json.dumps(
                    {
                        "mode": "write_operational_receipt",
                        "receipt": artifact,
                        "status": payload["status"],
                        "lineage": payload["canonical_scope"]["historical_t4_reference_qualification"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(
                json.dumps(
                    {
                        "mode": "dry_run",
                        "written": False,
                        "would_write_only_to": str(canonical_receipt_path()),
                        "status": payload["status"],
                        "checkpoint_sha_checks": 48,
                        "checkpoint_bytes_verified": payload["checkpoint_bytes_verified"],
                        "cpu_forward_batch_contract": payload["cpu_forward_batch_contract"],
                        "execution_scope": payload["execution_scope"],
                        "lineage": payload["canonical_scope"]["historical_t4_reference_qualification"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    except (core.A12AuditError, FileExistsError, OSError, ValueError) as exc:
        print(f"FAIL_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
