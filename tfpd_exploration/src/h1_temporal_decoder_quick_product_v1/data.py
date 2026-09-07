"""Held-in H1 windows: first3 calibration, query W700 entirely after those trials."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_calibration import load_frozen_c2_materializer
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_config import H1_TEMPORAL
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank

from .config import C2_PAYLOAD, H1_DATA_DIR, HELDIN_SESSIONS, REPO_ROOT, RESULT_ROOT, SEED, STRIDE, WINDOW

_SPINT_MAIN = str(REPO_ROOT / "SPINT-main")
if _SPINT_MAIN not in sys.path:
    sys.path.insert(0, _SPINT_MAIN)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_split(split: str) -> dict[str, Path]:
    root = H1_DATA_DIR / f"sub-HumanPitt-{split}"
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*.nwb")):
        for session in HELDIN_SESSIONS:
            if session in path.name:
                found[session] = path
                break
    missing = [s for s in HELDIN_SESSIONS if s not in found]
    if missing:
        raise FileNotFoundError(f"{split} missing {missing}")
    return found


def _load_nwb(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb
    from pynwb import NWBHDF5IO

    neural, velocity, _change, eval_mask = load_nwb(path, FalconTask.h1)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as handle:
        nwb = handle.read()
        trial_num = np.asarray(nwb.acquisition["TrialNum"].data[:], dtype=np.float64)
    neural = np.asarray(neural, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
    return neural, velocity, mask, trial_num


def _ordered_trials(trial_num: np.ndarray, mask: np.ndarray) -> tuple[float, ...]:
    ordered = trial_num[mask & np.isfinite(trial_num)]
    values: list[float] = []
    for value in ordered.tolist():
        if not values or float(value) != values[-1]:
            values.append(float(value))
    if not values:
        raise RuntimeError("no eval-valid trials")
    return tuple(values)


def _first3_end_index(trial_num: np.ndarray, mask: np.ndarray, first3: tuple[float, ...]) -> int:
    legal = mask & np.isfinite(trial_num) & np.isin(trial_num, np.asarray(first3, dtype=np.float64))
    indices = np.flatnonzero(legal)
    if indices.size == 0:
        raise RuntimeError("first3 trials have no eval bins")
    return int(indices[-1])


def _query_starts(mask: np.ndarray, first3_end: int, stride: int) -> list[int]:
    """Windows whose entire W=700 history starts after first3. Last bin eval-valid."""
    starts: list[int] = []
    earliest = first3_end + 1
    last_bin_min = earliest + WINDOW - 1
    for last in range(last_bin_min, mask.shape[0], stride):
        if mask[last]:
            starts.append(last - WINDOW + 1)
    return starts


@dataclass
class SessionArrays:
    session: str
    neural: np.ndarray
    velocity: np.ndarray
    eval_mask: np.ndarray
    trial_num: np.ndarray
    first3: tuple[float, ...]
    first3_end: int
    query_starts: list[int]
    path: str
    sha256: str


def load_session_arrays(
    path: Path,
    session: str,
    stride: int = STRIDE,
    *,
    skip_first3: bool,
) -> SessionArrays:
    neural, velocity, mask, trial_num = _load_nwb(path)
    if neural.shape[1] != 176 or velocity.shape[1] != 7:
        raise RuntimeError(f"{session} shape drift {neural.shape} {velocity.shape}")
    trials = _ordered_trials(trial_num, mask)
    if skip_first3:
        if len(trials) < 4:
            raise RuntimeError(f"{session}: held-in-calib needs first3 plus a query trial")
        first3 = trials[:3]
        first3_end = _first3_end_index(trial_num, mask, first3)
        starts = _query_starts(mask, first3_end, stride)
    else:
        # Minival is a query-only file; calibration stays on the matching calib first3.
        first3 = ()
        first3_end = -1
        starts = _query_starts(mask, first3_end, stride)
    return SessionArrays(
        session=session,
        neural=neural,
        velocity=velocity,
        eval_mask=mask,
        trial_num=trial_num,
        first3=first3,
        first3_end=first3_end,
        query_starts=starts,
        path=str(path),
        sha256=_sha256_file(path),
    )


def _payload_session_map() -> dict[str, dict[str, Any]]:
    payload = torch.load(C2_PAYLOAD, map_location="cpu", weights_only=False)
    from falcon_challenge.config import FalconConfig, FalconTask

    cfg = FalconConfig(task=FalconTask.h1)
    by_session: dict[str, dict[str, Any]] = {}
    for key, row in payload["sessions"].items():
        name = str(row["session"])
        if not name.startswith("ses-"):
            name = f"ses-{name}" if not name.startswith("ses") else name
        # payload stores session without 'ses-' sometimes; keep raw and hashed.
        by_session[str(row["session"])] = row
        by_session[key] = row
        stem = f"sub-HumanPitt-held-in-calib_{row['session']}"
        by_session[cfg.hash_dataset(stem)] = row
    return by_session


def materialize_banks(sessions: dict[str, SessionArrays]) -> dict[str, H1Bank]:
    materializer = load_frozen_c2_materializer()
    payload = torch.load(C2_PAYLOAD, map_location="cpu", weights_only=False)
    rows = payload["sessions"]
    # Index payload rows by the session token they carry.
    by_token: dict[str, Any] = {}
    for row in rows.values():
        token = str(row["session"])
        by_token[token] = row
        by_token[token.replace("ses-", "")] = row
        if not token.startswith("ses-"):
            by_token[f"ses-{token}"] = row
    banks: dict[str, H1Bank] = {}
    for session, arrays in sessions.items():
        row = by_token.get(session) or by_token.get(session.replace("ses-", ""))
        if row is None:
            raise KeyError(f"no legal M3 cache for {session}")
        activity = torch.as_tensor(np.asarray(row["identity"], dtype=np.float32))
        carrier = torch.as_tensor(np.asarray(row["carrier"], dtype=np.float32))
        if activity.ndim == 3:
            activity = activity.unsqueeze(0)
        if carrier.ndim == 2:
            carrier = carrier.unsqueeze(0)
        e0, hc = materializer.materialize_bank(activity, carrier)
        banks[session] = H1Bank(
            E0=e0,
            T=hc,
            unit_mask=torch.ones(H1_TEMPORAL.n_units, dtype=torch.bool),
        )
    return banks


def build_window_manifest() -> dict[str, Any]:
    calib_paths = _index_split("held-in-calib")
    mini_paths = _index_split("held-in-minival")
    train_sessions = {
        name: load_session_arrays(calib_paths[name], name, skip_first3=True) for name in HELDIN_SESSIONS
    }
    mini_sessions = {
        name: load_session_arrays(mini_paths[name], name, skip_first3=False) for name in HELDIN_SESSIONS
    }
    train_count = sum(len(s.query_starts) for s in train_sessions.values())
    mini_count = sum(len(s.query_starts) for s in mini_sessions.values())
    updates = (train_count // 32)
    body = {
        "schema": "h1_temporal_window_manifest_v1",
        "seed": SEED,
        "stride": STRIDE,
        "window": WINDOW,
        "calibration": "canonical_first3",
        "query_history_rule": "entire_W700_starts_after_first3_end",
        "source_selection": "held-in-minival_same_rule",
        "disclosure": "known-source development, not clean LODO",
        "train_windows": train_count,
        "minival_windows": mini_count,
        "full_batches_per_epoch": updates,
        "tail_windows": train_count - updates * 32,
        "per_session_train": {
            name: {
                "n_windows": len(sess.query_starts),
                "first3": list(sess.first3),
                "first3_end": sess.first3_end,
                "n_bins": int(sess.neural.shape[0]),
                "sha256": sess.sha256,
            }
            for name, sess in train_sessions.items()
        },
        "per_session_minival": {
            name: {
                "n_windows": len(sess.query_starts),
                "first3": list(sess.first3),
                "first3_end": sess.first3_end,
                "n_bins": int(sess.neural.shape[0]),
                "sha256": sess.sha256,
            }
            for name, sess in mini_sessions.items()
        },
    }
    dest = RESULT_ROOT / "window_manifest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "manifest": body,
        "train_sessions": train_sessions,
        "mini_sessions": mini_sessions,
        "path": dest,
    }


def shuffled_batches(
    sessions: dict[str, SessionArrays],
    *,
    seed: int,
    epoch: int,
    batch_size: int = 32,
) -> list[list[tuple[str, int]]]:
    """Session-pure shuffled batches. Tail batch keeps its true length."""
    batches: list[list[tuple[str, int]]] = []
    for name in HELDIN_SESSIONS:
        starts = list(sessions[name].query_starts)
        token = hashlib.sha256(f"h1_temporal_batch|{seed}|{epoch}|{name}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(token[:8], "little") % (2**63))
        order = rng.permutation(len(starts))
        items = [(name, starts[int(i)]) for i in order]
        for offset in range(0, len(items), batch_size):
            chunk = items[offset : offset + batch_size]
            if chunk:
                batches.append(chunk)
    token = hashlib.sha256(f"h1_temporal_batch_order|{seed}|{epoch}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(token[:8], "little") % (2**63))
    order = rng.permutation(len(batches))
    return [batches[int(i)] for i in order]


def collate_batch(
    sessions: dict[str, SessionArrays],
    items: list[tuple[str, int]],
) -> tuple[str, torch.Tensor, torch.Tensor, list[int]]:
    session = items[0][0]
    if any(name != session for name, _ in items):
        raise RuntimeError("batch is not session-pure")
    rec = sessions[session]
    xs = []
    ys = []
    starts = []
    for _, start in items:
        xs.append(rec.neural[start : start + WINDOW])
        ys.append(rec.velocity[start + WINDOW - 1])
        starts.append(start)
    return (
        session,
        torch.as_tensor(np.stack(xs), dtype=torch.float32),
        torch.as_tensor(np.stack(ys), dtype=torch.float32),
        starts,
    )
