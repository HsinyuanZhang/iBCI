"""Write a provenance-only receipt after verifying frozen runtime artifacts.

This does not construct a model, read NWB/NPZ data, fit maps, or benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .runtime import FrozenM3Readout


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--plain-ema-state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    loaded = FrozenM3Readout.load(
        args.receipt, checkpoint_path=args.checkpoint, plain_ema_state_path=args.plain_ema_state,
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    payload = {
        "schema": "h1_m3_runtime_v1_binding_verification_v1",
        "scope": "provenance-only runtime binding verification; no model construction, data read, fitting, or timing",
        "source_receipt": str(loaded.receipt_path),
        "source_receipt_sha256": _sha(loaded.receipt_path),
        "maps": str(loaded.maps_path), "maps_sha256": _sha(loaded.maps_path),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": loaded.checkpoint_sha256,
        "plain_ema_state": str(args.plain_ema_state.resolve()),
        "plain_ema_state_sha256": loaded.plain_ema_state_sha256,
        "map_family": "MAT7", "fit_dtype": "float64", "output_dtype": "float32",
        "session_count": len(loaded.maps), "sessions": sorted(loaded.maps),
        "code_sha256": {name: _sha(source / name) for name in ("runtime.py", "write_binding_receipt.py")},
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(output), "sha256": _sha(output), "sessions": len(loaded.maps)}, sort_keys=True))


if __name__ == "__main__":
    main()
