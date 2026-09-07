"""Inert AOF-S V4 local-build entrypoint; it cannot issue a capability."""
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="AOF-S V4 local build is READY_REQUIRES_OPAQUE_CAPABILITY.")
    parser.add_argument("--dry-run", action="store_true", help="print inert admission state")
    parser.parse_args()
    print("READY_REQUIRES_OPAQUE_CAPABILITY; no capability is minted and no Docker/data/result action is performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
