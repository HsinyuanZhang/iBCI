"""Replay frozen S1/S2 picks on local ext-4 using the training decoders."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import torch

from tfpd_exploration.src.m2_dual_track_v1 import champion, contracts, data, plan

from . import constants as C
from .load_weights import load_pick


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def replay_ext4(kind: str, *, device: str = "cpu") -> dict[str, Any]:
    expected_mean = C.S1_EXPECTED_MEAN if kind == "small" else C.S2_EXPECTED_MEAN
    expected_per = C.S1_EXPECTED_PER_SESSION if kind == "small" else C.S2_EXPECTED_PER_SESSION
    model, meta = load_pick(kind, device=device)
    banks = {
        session: data.load_session_bank("ext4", session, device=device)
        for session in plan.EXT4_SESSIONS
    }
    per_session: dict[str, float] = {}
    rows: dict[str, Any] = {}
    model.eval()
    for session, bank in banks.items():
        preds: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for batch in data.iter_session_batches(
            bank,
            batch_size=plan.EFFECTIVE_BATCH,
            device=device,
            target_space=plan.SCORING_TARGET_SPACE,
        ):
            with torch.inference_mode():
                raw = model.forward_last(batch.X, batch.bank, batch.unit_mask)
            preds.append(np.ascontiguousarray(raw.detach().cpu().numpy() / plan.BEHAVIOR_SCALE, dtype=np.float32))
            targets.append(batch.last_target.detach().cpu().numpy())
        target = np.concatenate(targets, axis=0)
        prediction = np.concatenate(preds, axis=0)
        r2 = contracts.variance_weighted_r2(target, prediction)
        per_session[session] = float(r2)
        rows[session] = {
            "r2": float(r2),
            "window_count": int(target.shape[0]),
            "prediction_digest": champion.array_sha256(prediction),
            "target_digest": champion.array_sha256(target),
        }
        require(
            abs(float(r2) - expected_per[session]) < 1.0e-6,
            f"{kind} {session} r2 {r2} != {expected_per[session]}",
        )
    summary = contracts.summarize_sessions(per_session)
    mean = float(summary["equal_session_mean"])
    require(abs(mean - expected_mean) < 1.0e-6, f"{kind} mean {mean} != {expected_mean}")
    receipt = {
        "kind": kind,
        "match": True,
        "equal_session_mean": mean,
        "expected_equal_session_mean": expected_mean,
        "per_session": rows,
        "expected_per_session": expected_per,
        "weight_sha256": meta["weight_sha256"],
        "ckpt": meta["ckpt"],
        "view": meta["view"],
        "epoch": meta["epoch"],
        "replay_sha256": _sha256_text(json.dumps({"mean": mean, "per": per_session, "w": meta["weight_sha256"]}, sort_keys=True)),
        **meta,
    }
    if kind == "large":
        receipt["worst_session"] = C.S2_WORST_SESSION
        receipt["worst_session_r2"] = C.S2_WORST_R2
        receipt["disclosure"] = "architecture probe"
    dest = C.SLOT_ROOT / f"m2_{kind}_ext4_replay.json"
    dest.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt
