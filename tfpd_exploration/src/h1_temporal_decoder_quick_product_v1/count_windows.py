"""Seal the stride-4 held-in window manifest and print update counts."""

from __future__ import annotations

import json

from .data import build_window_manifest


def main() -> None:
    pack = build_window_manifest()
    print(json.dumps(pack["manifest"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
