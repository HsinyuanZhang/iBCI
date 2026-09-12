#!/usr/bin/env python3
"""Offline structural verifier for generated external GF packages."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_one(path: Path) -> dict:
    required = ("Dockerfile", "README.md", "falcon_decoder.py", "decode.py", "payload.npz", "manifest.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != "external_gf_falcon_payload_v1":
        raise ValueError(f"{path}: wrong manifest schema")
    payload = np.load(path / "payload.npz", allow_pickle=False)
    forbidden = [key for key in payload.files if any(word in key.lower() for word in ("label", "target_y", "query", "nwb", "pickle"))]
    if forbidden:
        raise ValueError(f"{path}: forbidden payload fields {forbidden}")
    for key in ("wf_mean", "wf_scale", "wf_coef"):
        if key not in payload.files or not np.isfinite(payload[key]).all():
            raise ValueError(f"{path}: invalid {key}")
    return {"package": str(path), "payload_sha256": sha(path / "payload.npz"), "manifest_sha256": sha(path / "manifest.json"), "arrays": len(payload.files)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([verify_one(path) for path in args.paths], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
