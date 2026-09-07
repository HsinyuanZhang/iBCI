#!/usr/bin/env python3
"""Bind the CPU source-fit equivalence audit to the immutable v2 receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F1_S42_STAGED_RECEIPT_v2.json"
EQUIV = ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F1_S42_SOURCE_FIT_EQUIVALENCE_v1.json"
OUT = ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F1_S42_V2_EQUIVALENCE_BINDING_v1.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing/symlinked receipt: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("canonical_content_sha256") != canonical({k: v for k, v in value.items() if k != "canonical_content_sha256"}):
        raise RuntimeError(f"canonical drift: {path}")
    return value


def build() -> dict[str, Any]:
    parent = read(PARENT)
    equiv = read(EQUIV)
    if parent.get("schema") != "m1_compact_b3s_f1_s42_staged_v2":
        raise RuntimeError("parent is not v2 staged receipt")
    if equiv.get("status") != "PASS_M1_COMPACT_B3S_F1_S42_SOURCE_FIT_EQUIVALENT":
        raise RuntimeError("source-fit equivalence did not pass")
    return {
        "schema": "m1_compact_b3s_f1_s42_v2_equivalence_binding_v1",
        "status": "PASS_M1_COMPACT_B3S_F1_S42_V2_EQUIVALENCE_BOUND",
        "parent_v2_receipt": {"path": str(PARENT.resolve()), "sha256": sha(PARENT)},
        "equivalence_receipt": {"path": str(EQUIV.resolve()), "sha256": sha(EQUIV)},
        "scope": {"task": "m1", "fold": 1, "seed": 42, "arm": "b3s_zero4", "source_only_fit": True, "target_values_used": False},
        "equivalence": {"train_dataset_equal": bool(equiv["train_dataset"]["equal"]), "sampler_equal": bool(equiv["sampler"]["equal"]), "zero4_side_exact": all(item["side_exact_zero"] for item in equiv["representative_samples"].values()), "target_nwb_opened": bool(equiv["source_only_guard"]["target_nwb_opened"])},
    }


def write_once(body: dict[str, Any], output: Path = OUT) -> dict[str, Any]:
    if output.exists():
        existing = read(output)
        if canonical(body) != existing["canonical_content_sha256"]:
            raise RuntimeError("existing binding differs; refusing overwrite")
        return existing
    value = dict(body)
    value["canonical_content_sha256"] = canonical(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=str(output.parent))
    temporary = Path(name)
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.chmod(0o444)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    body = write_once(build(), args.output.resolve())
    print(json.dumps({"path": str(args.output.resolve()), "sha256": sha(args.output.resolve()), "status": body["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
