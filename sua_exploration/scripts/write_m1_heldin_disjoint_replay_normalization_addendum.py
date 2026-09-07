#!/usr/bin/env python3
"""Freeze train-only T4/TS4 normalization receipts for the M1 held-in replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import yaml


GROUPS = ("t4", "ts4")
CELLS = ("fold1_seed42", "fold1_seed43", "fold2_seed42")
CELL_META = {"fold1_seed42": (1, 42), "fold1_seed43": (1, 43), "fold2_seed42": (2, 42)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_train_normalization(root: Path, protocol: dict, group: str, cell: str) -> dict[str, Any]:
    sys.path.insert(0, str(root / "streaming_calibration_exp"))
    from src.data.falcon_datamodule import FalconDataModule

    frozen = protocol["frozen_source_arms"][group][cell]
    source_data = dict(frozen["source_data"])
    source_data.pop("_target_", None)
    source_data["data_dir"] = str(root / "SPINT-main/data/000941") + "/"
    source_data["heldin_query_start_trial"] = 0
    source_data["query_start_trial"] = 0
    source_data["num_workers"] = 0
    fold, seed = CELL_META[cell]
    source_data["loso_fold"] = fold
    source_data["sampler_seed"] = seed
    source_data["side_feature_shuffle_seed"] = seed
    dm = FalconDataModule(**source_data)
    trainer = MagicMock()
    trainer.world_size = 1
    dm.trainer = trainer
    dm.setup(stage="fit")
    norm = getattr(dm, "native_t4_normalization", None)
    if not isinstance(norm, dict):
        raise ValueError(f"{group}/{cell}: failed to compute train-only T4 normalization")
    return {
        "feature_group": norm["feature_group"],
        "mean": [float(x) for x in norm["mean"]],
        "std": [float(x) for x in norm["std"]],
        "train_sessions": list(norm["train_sessions"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha = sha256(protocol_path)
    norms: dict[str, dict[str, Any]] = {group: {} for group in GROUPS}
    for group in GROUPS:
        for cell in CELLS:
            frozen = protocol["frozen_source_arms"][group][cell]
            norms[group][cell] = {
                "source_checkpoint": frozen["checkpoint"],
                "source_resolved_config_sha256": frozen["source_resolved_config_sha256"],
                "native_t4_normalization": compute_train_normalization(root, protocol, group, cell),
            }
    out = args.out or root / "sua_exploration/results/m1_heldin_disjoint_replay_v1/normalization_parity_addendum_v1.json"
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite normalization addendum: {out}")
    payload = {
        "data_access_disclosure": "This script reads only protocol JSON and source resolved configs, then runs score-free datamodule setup to recompute train-only T4/TS4 normalization. It does not open metrics CSVs, aggregates, scores, or predictions.",
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha},
        "norms": norms,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    print(sha256(out))


if __name__ == "__main__":
    main()
