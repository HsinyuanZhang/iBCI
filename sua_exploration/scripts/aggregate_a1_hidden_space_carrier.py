#!/usr/bin/env python3
"""Immutable aggregate for the minimal A1 H/T4 development pilot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUA = REPO / "sua_exploration"
if str(SUA) not in sys.path:
    sys.path.insert(0, str(SUA))

from a1_hidden_carrier.artifacts import (  # noqa: E402
    load_verified_immutable_json,
    write_immutable_json,
)
from a1_hidden_carrier.contract import aggregate_pilot  # noqa: E402
from a1_hidden_carrier.evidence import load_verified_preflight  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--fresh-score", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    preflight, preflight_sha = load_verified_preflight(args.preflight)
    fresh, fresh_sha = load_verified_immutable_json(args.fresh_score, label="A1 H/T4 score receipt")
    payload = aggregate_pilot(
        a2_anchor=preflight["a2_reuse_evidence"],
        fresh_score_receipt=fresh,
        fresh_score_receipt_sha256=fresh_sha,
        preflight_sha256=preflight_sha,
    )
    body, sidecar, digest = write_immutable_json(args.out, payload)
    print(json.dumps({
        "aggregate": str(body), "sidecar": str(sidecar), "sha256": digest,
        "passes": payload["passes"], "classification": payload["classification"],
        "mean_primary_interaction": payload["overall"]["mean_primary_interaction"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
