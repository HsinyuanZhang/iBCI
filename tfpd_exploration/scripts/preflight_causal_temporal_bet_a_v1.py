#!/usr/bin/env python3
"""Print the no-write Bet A design contract; this is not an execution preflight."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# This must be set before the route import below.  A dry plan is non-writing even when the caller
# has not supplied PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True


_FORBIDDEN_PREFIXES = (
    "--execute",
    "--launch",
    "--mint",
    "--output",
    "--output-path",
    "--data",
    "--data-path",
    "--device",
    "--checkpoint",
    "--architecture",
    "--gpu",
)


def _reject_cli_arguments(arguments: list[str]) -> None:
    """Reject operational intent before importing the design contract or touching any data."""

    if not arguments:
        return
    first = arguments[0]
    if any(first == prefix or first.startswith(f"{prefix}=") for prefix in _FORBIDDEN_PREFIXES):
        raise SystemExit(f"refusing operational option {first!r}: NO_IMPLEMENTATION_NO_DATA_NO_CUDA_NO_WRITE_NO_LAUNCH")
    raise SystemExit(
        f"this dry-plan CLI accepts no options; refusing {first!r}: "
        "NO_IMPLEMENTATION_NO_DATA_NO_CUDA_NO_WRITE_NO_LAUNCH"
    )


def main(arguments: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    _reject_cli_arguments(arguments)
    # The sole import happens after the rejection gate.  It is a stdlib-only design descriptor.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from causal_temporal_bet_a_v1.contract import dry_plan

    print(json.dumps(dry_plan(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
