"""Shuffled session-pure batch manifest.

Keeps batches inside one session, but does not train sessions in calendar
order or walk overlapping windows in time order.  Window rows and targets
share one index permutation.  Sampler RNG is independent of model seeds.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from . import contracts, plan

SAMPLER_SCHEMA = "m2_dual_track_v1_shuffled_session_pure_v1"
SAMPLER_SEED = 20260905
SAMPLER_DOMAIN = "within_session_window_perm_then_global_batch_shuffle"


def _digest(payload: object) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def session_lengths(banks: Mapping[str, Any]) -> dict[str, int]:
    return {session: int(len(banks[session].eligible_starts)) for session in plan.HELDIN_SESSIONS}


def session_lengths_from_cache(cache_root: Path, surface: str = "source_train") -> dict[str, int]:
    lengths: dict[str, int] = {}
    for session in plan.HELDIN_SESSIONS:
        path = Path(cache_root) / surface / session / "eligible_starts.npy"
        plan.require(path.is_file(), f"missing eligible_starts: {path}")
        lengths[session] = int(np.load(path).shape[0])
    return lengths


def build_shuffled_manifest(
    lengths: Mapping[str, int],
    *,
    batch_size: int = plan.EFFECTIVE_BATCH,
    epochs: int = plan.EPOCHS,
    sampler_seed: int = SAMPLER_SEED,
) -> dict[str, Any]:
    plan.require(batch_size > 0, "batch_size")
    rng = np.random.default_rng(int(sampler_seed))
    epochs_out: dict[str, list[dict[str, Any]]] = {}
    for epoch in range(1, int(epochs) + 1):
        batches: list[dict[str, Any]] = []
        for session in plan.HELDIN_SESSIONS:
            n_win = int(lengths[session])
            perm = rng.permutation(n_win).astype(np.int64)
            for offset in range(0, n_win, batch_size):
                idx = perm[offset : offset + batch_size].tolist()
                batches.append({"session": session, "indices": idx})
        order = rng.permutation(len(batches))
        epochs_out[str(epoch)] = [batches[int(i)] for i in order]
    manifest = {
        "schema": SAMPLER_SCHEMA,
        "domain": SAMPLER_DOMAIN,
        "sampler_seed": int(sampler_seed),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "sessions": list(plan.HELDIN_SESSIONS),
        "lengths": {key: int(lengths[key]) for key in plan.HELDIN_SESSIONS},
        "session_pure": True,
        "drop_last": False,
        "calendar_session_order": False,
        "within_session_time_order": False,
        "shared_window_target_perm": True,
        "batches": epochs_out,
    }
    manifest["digest"] = _digest({k: v for k, v in manifest.items() if k != "digest"})
    return manifest


def extend_shuffled_manifest(parent: Mapping[str, Any], *, epochs: int) -> dict[str, Any]:
    """Grow a frozen 12-epoch manifest to ``epochs`` without rewriting the parent.

    The same sampler seed is replayed so epochs 1-12 match byte-for-batch.
    The new whole-file digest must differ; the parent digest is stored separately.
    """
    plan.require(int(parent["epochs"]) == plan.EPOCHS, "parent must be the frozen 12-epoch manifest")
    plan.require(int(epochs) > plan.EPOCHS, "extension epochs")
    grown = build_shuffled_manifest(
        parent["lengths"],
        batch_size=int(parent["batch_size"]),
        epochs=int(epochs),
        sampler_seed=int(parent["sampler_seed"]),
    )
    for epoch in range(1, plan.EPOCHS + 1):
        plan.require(
            grown["batches"][str(epoch)] == parent["batches"][str(epoch)],
            f"extended manifest drifted at epoch {epoch}",
        )
    grown["parent_12_digest"] = parent["digest"]
    grown["parent_12_prefix_equal"] = True
    grown["parent_12_epochs"] = plan.EPOCHS
    grown["digest"] = _digest({k: v for k, v in grown.items() if k != "digest"})
    plan.require(grown["digest"] != parent["digest"], "extended digest must not equal parent digest")
    return grown


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        plan.require(existing.get("digest") == manifest["digest"], "sampler manifest digest drift")
        return
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    plan.require(manifest.get("schema") == SAMPLER_SCHEMA, "sampler schema")
    expect = _digest({k: v for k, v in manifest.items() if k != "digest"})
    plan.require(manifest.get("digest") == expect, "sampler digest mismatch")
    return manifest


def batch_from_indices(
    bank: contracts.SessionBank,
    indices: list[int] | np.ndarray,
    *,
    device: str | torch.device,
    target_space: str,
) -> contracts.Batch:
    device = torch.device(device)
    idx = np.asarray(indices, dtype=np.int64)
    starts = np.asarray(bank.eligible_starts, dtype=np.int64)
    plan.require(idx.ndim == 1 and idx.size > 0, "empty batch indices")
    plan.require(np.all((idx >= 0) & (idx < starts.size)), "index out of range")
    store = np.asarray(bank.X_store)
    if store.ndim == 3:
        windows = np.ascontiguousarray(store[idx], dtype=np.float32)
    elif store.ndim == 2:
        windows = np.stack(
            [np.asarray(store[int(starts[i]) : int(starts[i]) + plan.WINDOW], dtype=np.float32) for i in idx],
            axis=0,
        )
    else:
        raise plan.DualTrackError(f"X_store ndim {store.ndim} is not 2 or 3")
    native = np.ascontiguousarray(np.asarray(bank.target_store)[idx], dtype=np.float32)
    if target_space == plan.TRAINING_TARGET_SPACE:
        last = native * np.float32(plan.BEHAVIOR_SCALE)
    elif target_space == plan.SCORING_TARGET_SPACE:
        last = native
    else:
        raise plan.DualTrackError(f"unknown target space {target_space}")
    return contracts.Batch(
        session_id=bank.session_id,
        X=torch.from_numpy(windows).to(device),
        last_target=torch.from_numpy(np.array(last, dtype=np.float32, copy=True)).to(device),
        bank=bank,
        window_ids=tuple(int(starts[i]) for i in idx),
        unit_mask=bank.unit_mask.to(device),
    )


def iter_manifest_batches(
    banks: Mapping[str, contracts.SessionBank],
    manifest: Mapping[str, Any],
    epoch: int,
    *,
    device: str | torch.device,
    target_space: str = plan.TRAINING_TARGET_SPACE,
) -> Iterator[contracts.Batch]:
    specs = manifest["batches"][str(int(epoch))]
    for spec in specs:
        session = spec["session"]
        yield batch_from_indices(
            banks[session],
            spec["indices"],
            device=device,
            target_space=target_space,
        )


def manifest_session_sequence(manifest: Mapping[str, Any], epoch: int) -> list[str]:
    return [spec["session"] for spec in manifest["batches"][str(int(epoch))]]
