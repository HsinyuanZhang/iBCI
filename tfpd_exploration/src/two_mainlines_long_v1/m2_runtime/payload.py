"""Export Transformer weights + official M33 banks into an EvalAI pickle."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from falcon_challenge.config import FalconConfig, FalconTask

from . import PAYLOAD_SCHEMA
from . import constants as C
from .banks import bank_receipt, collect_official_banks
from .load_weights import export_state_numpy, load_pick


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def build_payload(kind: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    model, meta = load_pick(kind, device="cpu")
    banks = collect_official_banks()
    config = FalconConfig(task=FalconTask.m2)
    payload = {
        "schema_version": PAYLOAD_SCHEMA,
        "task": config.task,
        "kind": kind,
        "state_dict": export_state_numpy(model),
        "bank_by_dataset_tag": {
            tag: {
                "E0": np.ascontiguousarray(row["E0"], dtype=np.float32),
                "T": np.ascontiguousarray(row["T"], dtype=np.float32),
                "unit_mask": np.ascontiguousarray(row["unit_mask"], dtype=np.bool_),
            }
            for tag, row in banks.items()
        },
        "window_size": C.WINDOW,
        "behavior_scaling_factor": C.BEHAVIOR_SCALE,
        "smooth_observations": False,
        "metadata": {
            **meta,
            "label_budget": 33,
            "activity_budget": 33,
            "online_kv_cache": False,
            "old_spint_decoder": False,
            "tta": False,
            "official_tags": list(C.OFFICIAL_TAGS),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    digest = sha256_file(output)
    require(digest not in C.KNOWN_PAYLOAD_SHA256, f"payload digest collides with an existing submission: {digest}")
    receipt = {
        "schema": "m2_trf_payload_receipt_v1",
        "kind": kind,
        "payload_path": str(output),
        "payload_sha256": digest,
        "bytes": output.stat().st_size,
        "weight_sha256": meta["weight_sha256"],
        "banks": bank_receipt(banks),
        "dedup_vs_known_payloads": True,
        "uses_old_spint_decoder": False,
    }
    (output.parent / "payload.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt
