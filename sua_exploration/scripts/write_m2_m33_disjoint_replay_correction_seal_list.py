#!/usr/bin/env python3
"""Write an immutable explicit list of the nine sealed M33 correction arms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCREEN = "m2_m33_disjoint_replay_correction_v1"
PROTOCOL_SHA = "da4eb05db43401f2e314351e8882c6606cc4d020aed3ed45e860d52db9e59df1"
GROUPS = {"f0", "t4", "ts4"}
CELLS = {"fold1_seed42", "fold1_seed43", "fold2_seed42"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", action="append", required=True, metavar="GROUP/CELL=ARTIFACT_DIR")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    entries: dict[tuple[str, str], Path] = {}
    for raw in args.entry:
        try:
            key, raw_path = raw.split("=", 1)
            group, cell = key.split("/", 1)
        except ValueError as exc:
            raise ValueError(f"invalid --entry {raw!r}; expected GROUP/CELL=ARTIFACT_DIR") from exc
        if group not in GROUPS or cell not in CELLS or (group, cell) in entries:
            raise ValueError(f"invalid or duplicate explicit seal entry {raw!r}")
        entries[(group, cell)] = Path(raw_path).resolve()
    expected = {(group, cell) for group in GROUPS for cell in CELLS}
    if set(entries) != expected:
        raise ValueError(f"seal list must contain exactly nine arms; missing={sorted(expected-set(entries))}")
    sealed = {group: {} for group in sorted(GROUPS)}
    for (group, cell), artifact in sorted(entries.items()):
        provenance_path = artifact / "m33_disjoint_replay_correction_provenance.json"
        if not provenance_path.is_file():
            raise ValueError(f"{group}/{cell}: explicit artifact lacks a correction seal")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("protocol", {}).get("sha256") != PROTOCOL_SHA:
            raise ValueError(f"{group}/{cell}: protocol SHA mismatch")
        if provenance.get("group") != group or provenance.get("cell", {}).get("name") != cell:
            raise ValueError(f"{group}/{cell}: provenance identity mismatch")
        sealed[group][cell] = {
            "path": str(artifact),
            "provenance_sha256": sha256(provenance_path),
        }
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite explicit seal list: {out}")
    payload = {
        "schema_version": 1,
        "purpose": "M2_M33_correction_explicit_sealed_artifact_list",
        "protocol_sha256": PROTOCOL_SHA,
        "sealed_artifacts": sealed,
        "selection_rule": "Every artifact is supplied as an explicit command-line path; no glob or directory discovery is used. Only the listed provenance SHA values are admissible to the aggregate.",
        "historical_M33_scores_not_read": True,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    print(sha256(out))


if __name__ == "__main__":
    main()
