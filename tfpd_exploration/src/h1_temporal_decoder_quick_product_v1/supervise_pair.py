"""Disabled. Dual-GPU FLAT/ROUTE is the schedule; do not pkill or pin GPU0."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "supervise_pair is disabled. Run FLAT on GPU0 and ROUTE on GPU1 as separate processes."
    )


if __name__ == "__main__":
    main()
