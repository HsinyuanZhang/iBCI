"""Run immutable, source-only PCF8 ridge100 v2 audit."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SPINT-main"))
from src.data.h1_m4_eb_pilot import H1_M4_FOLD0_SOURCE, load_source_records  # noqa: E402
from pcf8_ridge100_v2 import audit_ridge100_source_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"PCF8 ridge100 v2 refuses to overwrite {output}")
    rendered = json.dumps(audit_ridge100_source_records(load_source_records(args.data_root), H1_M4_FOLD0_SOURCE), indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o444); os.link(temporary, output)
    finally:
        if temporary.exists(): temporary.unlink()


if __name__ == "__main__": main()
