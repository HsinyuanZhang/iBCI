#!/usr/bin/env python3
"""Fail-closed validator for the literal clean-teacher completion log record."""
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = b"`Trainer.fit` stopped: `max_epochs=35` reached."
REQUIRED_PROGRESS = b"Epoch 34: 100%"


def validate_completion_log(path: Path) -> None:
    raw = path.read_bytes()
    occurrences = raw.count(MARKER)
    if occurrences != 1:
        raise ValueError(f"completion marker must occur exactly once, found {occurrences}")
    position = raw.index(MARKER)
    if raw[position + len(MARKER) : position + len(MARKER) + 1] != b"\n":
        raise ValueError("completion marker is not immediately followed by a newline")
    previous_boundary = max(raw.rfind(b"\r", 0, position), raw.rfind(b"\n", 0, position))
    progress_segment = raw[previous_boundary + 1 : position]
    if REQUIRED_PROGRESS not in progress_segment:
        raise ValueError("completion marker is not attached to the final Epoch 34 progress record")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    validate_completion_log(args.log)


if __name__ == "__main__":
    main()
