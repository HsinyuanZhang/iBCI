"""Export EMA weights + materialized H1 banks into a new EvalAI pickle.

Does not edit the sealed C2/581920 payload. The C2 checkpoint is used only
to materialize E0/H-C.
"""

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
from .load_weights import export_state_numpy, load_ema


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
    require(kind in {"flat", "route"}, f"unknown kind {kind}")
    model, meta = load_ema(kind, device="cpu")
    banks = collect_official_banks()
    config = FalconConfig(task=FalconTask.h1)
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
        "pe_max_len": C.PE_MAX_LEN,
        "behavior_scaling_factor": C.BEHAVIOR_SCALE,
        "metadata": {
            **meta,
            "calibration_ckpt_sha256": C.C2_CKPT_SHA256,
            "calibration_only": True,
            "online_kv_cache": False,
            "old_c2_decoder": False,
            "c2_identity_swap": False,
            "tta": False,
            "official_tags": list(C.OFFICIAL_TAGS),
            "disclosure": "known-source development; snapshot epoch 5 of intended 12; not clean LODO",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    digest = sha256_file(output)
    require(digest not in C.KNOWN_PAYLOAD_SHA256, f"payload digest collides with an existing submission: {digest}")
    require(digest != C.C2_PAYLOAD_SHA256, "new payload must not be the C2/581920 bytes")
    receipt = {
        "schema": "h1_temporal_trf_payload_receipt_v1",
        "kind": kind,
        "payload_path": str(output),
        "payload_sha256": digest,
        "bytes": output.stat().st_size,
        "weight_sha256": meta["weight_sha256"],
        "epoch": C.EPOCH,
        "epochs_target": C.EPOCHS_TARGET,
        "view": C.VIEW,
        "banks": bank_receipt(banks),
        "uses_old_c2_decoder": False,
        "c2_identity_swap": False,
        "c2_ckpt_sha256_calibration_only": C.C2_CKPT_SHA256,
    }
    (output.parent / "payload.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt
