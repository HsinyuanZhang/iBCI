"""Run only the exact 11-record H1 fold-0 source PCF8 audit and print JSON."""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPINT = ROOT / "SPINT-main"
sys.path.insert(0, str(SPINT))

from src.data.h1_m4_eb_pilot import H1_M4_FOLD0_SOURCE, load_source_records  # noqa: E402
from pcf8_precursor import audit_source_records, canonical_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="publish immutable JSON once; omit to print")
    args = parser.parse_args()
    records = load_source_records(args.data_root)
    rendered = canonical_json(audit_source_records(records, H1_M4_FOLD0_SOURCE))
    if args.output is None:
        print(rendered, end="")
        return
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"PCF8 audit refuses to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
