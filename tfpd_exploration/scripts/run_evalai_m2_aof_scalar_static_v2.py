#!/usr/bin/env python3
"""Inert AOF-S V2 recovery CLI; it cannot mint or execute a capability."""
from __future__ import annotations

import json


def main() -> None:
    print(json.dumps({
        "status": "INERT_V2_RECOVERY_REQUIRES_EXTERNAL_CLOSURE_REVIEW",
        "network_submission": False,
        "receipt_codec": "side_evidence.selected_indices+selected_indices_sha256",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
