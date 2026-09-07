#!/usr/bin/env python3
"""Derive the H1 EP-FiLM/no-readout payload from the immutable 581866 payload."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "SPINT-main/local_data/h1_epfilm_evalai_v1/decoder.pt"
OUTPUT = Path(__file__).resolve().parent / "artifacts/decoder.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUTPUT}")
    source_sha = sha256(SOURCE)
    if source_sha != "df71cb9329a87b5073242044b2933391ced7bf866d1f481e994f71dbd97ba2d7":
        raise RuntimeError("581866 source payload drift")
    payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    eye = np.eye(7, dtype=np.float64)
    zero = np.zeros(7, dtype=np.float64)
    one = np.ones(7, dtype=np.float64)
    for row in payload["sessions"].values():
        row["readout"] = {
            "family": "MAT7",
            "ridge": 0.0,
            "p_mean": zero.copy(),
            "p_scale": one.copy(),
            "y_mean": zero.copy(),
            "y_scale": one.copy(),
            "weight": eye.copy(),
            "intercept": zero.copy(),
        }
        row["readout_calibration_trials"] = list(row["calibration_trials"])
    payload["readout_selection_sha256"] = "IDENTITY_READOUT_NO_FIT_V1"
    payload["readout_family"] = "MAT7"
    payload["readout_ridge"] = 0.0
    payload["derivation"] = {
        "source_payload_sha256": source_sha,
        "change": "replace every MAT7 map with exact 7-D identity map",
        "hidden_test_opened": False,
        "optimizer_steps": 0,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUTPUT)
    receipt = {
        "schema": "h1_epfilm_no_readout_v1_payload",
        "source_payload_sha256": source_sha,
        "payload_sha256": sha256(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "sessions": len(payload["sessions"]),
        "readout": "exact_identity",
        "hidden_test_opened": False,
    }
    (OUTPUT.parent / "payload.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
