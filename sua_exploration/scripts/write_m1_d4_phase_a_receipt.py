#!/usr/bin/env python3
"""Write an immutable CPU-only D4 Phase-A protocol receipt; never launches work."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "sua_exploration" / "results" / "m1_d4_pilot_v1"
INPUTS = {
    "m1_d4_semantic_audit": ROOT / "sua_exploration" / "results" / "m1_d4_semantics_v2" / "audit.json",
    "m1_d4_minimal_gpu_pilot_protocol": ROOT / "sua_exploration" / "docs" / "M1_D4_MINIMAL_GPU_PILOT_PROTOCOL.md",
    "m1_clean_selection_protocol_receipt": ROOT / "sua_exploration" / "results" / "m1_clean_selection_v1" / "protocol_receipt.json",
    "d4_feature_helper": ROOT / "streaming_calibration_exp" / "src" / "data" / "falcon_d4_features.py",
    "d4_feature_tests": ROOT / "streaming_calibration_exp" / "tests" / "test_falcon_d4_features.py",
}
EXPECTED = {
    "m1_d4_semantic_audit": "5d56e893ce7f9695766fd2d01a8786abf4be62aca0532eb3916105d2b034d160",
    "m1_d4_minimal_gpu_pilot_protocol": "d67134214f4517422eb908b6b37724aed5dc2f8f404b3edb1aaa9bd0a3c3de10",
    "m1_clean_selection_protocol_receipt": "7f4e9cf0d353ce80451a5afa62c1adbfbe8f6f88588a4e46392810dfaed8884c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_receipt() -> dict[str, Any]:
    inputs: dict[str, dict[str, str]] = {}
    for name, path in INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Phase-A receipt requires {name}: {path}")
        digest = sha256(path)
        expected = EXPECTED.get(name)
        if expected is not None and digest != expected:
            raise ValueError(f"Phase-A receipt hash mismatch for {name}: {digest} != {expected}")
        inputs[name] = {"path": str(path.relative_to(ROOT)), "sha256": digest}
    return {
        "schema_version": "m1_d4_phase_a_protocol_receipt_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_and_input_hashes": {
            name: record for name, record in inputs.items()
            if name in {
                "m1_d4_semantic_audit", "m1_d4_minimal_gpu_pilot_protocol",
                "m1_clean_selection_protocol_receipt",
            }
        },
        "feature_and_test_hashes": {
            name: record for name, record in inputs.items()
            if name in {"d4_feature_helper", "d4_feature_tests"}
        },
        "launch_blocked": {
            "status": "blocked",
            "reason": "m1_clean_selection_v1 training and sealed-report workflow must complete before a separately reviewed integration/launch phase",
        },
    }


def run(out: Path = DEFAULT_OUT) -> Path:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing Phase-A result directory: {out}")
    out.mkdir(parents=True, exist_ok=False)
    try:
        receipt = build_receipt()
        written = out / "protocol_receipt.json"
        with written.open("w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        (out / "protocol_receipt.sha256").write_text(f"{sha256(written)}  protocol_receipt.json\n", encoding="utf-8")
        return written
    except Exception:
        for child in out.iterdir():
            child.unlink()
        out.rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    print(run(parser.parse_args().out))


if __name__ == "__main__":
    main()
