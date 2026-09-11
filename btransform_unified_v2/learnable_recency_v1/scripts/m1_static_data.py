"""Raw, static M1 data plane shared by the static training and scoring runner.

This module deliberately has no identity, calibration, target-support, E0, T,
or encoder dependency.  It reproduces the existing M1 RIFT query coordinate
law directly from NWB arrays: raw 20-ms neural bins, 99 explicit zero bins on
the left, and targets at every eligible query endpoint.
"""
from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent.parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DATA_ROOT = WORKSPACE_ROOT / "SPINT-main" / "data" / "000941"

SOURCE_SESSIONS = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
HELDOUT_SESSIONS = ("20121004", "20121017", "20121024")
CONTEXT = 100
PAD = CONTEXT - 1
RAW_UNITS = 64
TARGETS = 16
BATCH = 32
SEED = 42

EXPECTED_SOURCE_WINDOWS = {
    "ses-20120924": 54849,
    "ses-20120926": 54476,
    "ses-20120927": 49228,
    "ses-20120928": 54783,
}
EXPECTED_SOURCE_STARTS_SHA256 = {
    "ses-20120924": "0bc7c1f9d9f593fb8736e7d5c9c5ca6d5feda6ad88897f10687d303a8890b213",
    "ses-20120926": "2d8899d849a0a19c50defdcd8ebbca2c3f064a34df5fe33bebe37faf4831e3ce",
    "ses-20120927": "f8cfdd613e35b99fe784af3f2faa29651ed9ddffc6a88023c51217da2523d0f1",
    "ses-20120928": "9a6172513b4887caff151146d495d84da848f01722279cad164628bc917a9410",
}
EXPECTED_SOURCE_EVAL_MASK_SHA256 = {
    "ses-20120924": "a138e7f183383f4e3730e3abb5c6d0b8b8830176c821f0b137ee631f9133e1ea",
    "ses-20120926": "182d60ba55895c7d3cf73a683207ebcccac0e5cab17afcd28468f2477325f6db",
    "ses-20120927": "52e09bd49698bf4c2fc01eb68603d49ec19dbd7407fe89001cc307ba18c1c060",
    "ses-20120928": "b9825464bd116b85386e4a09c207e3a069d4efed67549dd8d4df5daa25449aa6",
}
EXPECTED_SAMPLER_SHA256 = "afce94aeaf359734faa06f115a2ee5b62011a9b1df5a99b11790d53e4b5df796"
EXPECTED_HELDOUT = {
    "20121004": {
        "window_count": 1305,
        "starts_sha256": "3a6ec694179836d7ff01397651834a91e7f70ca3a6192156b9735a8336168008",
        "target_sha256": "b3279f50f49f65d6501ed370356b2bdac5cadffd8989f45cc28c7a2db5678139",
    },
    "20121017": {
        "window_count": 1295,
        "starts_sha256": "b77072b898db5b7f95464fe26e563a4a93b0c54ab4dc8661e6c03ff60d954b4b",
        "target_sha256": "086c75182eea50642096799d9683c74bb365b0b9c3653a3bf0f4096775d45274",
    },
    "20121024": {
        "window_count": 1281,
        "starts_sha256": "67e207bf2425dae50d7ab9f28d7f148f09f48d8abe22d7c0abf6749a3e4a67a4",
        "target_sha256": "43a15b3acfa4e698a2d6f64d33ddf2358ee9230fed7b73c418b197cb3c9c0ded",
    },
}


def _array_sha(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_path(session: str) -> Path:
    if session not in SOURCE_SESSIONS:
        raise ValueError(f"unknown M1 source session: {session!r}")
    return DATA_ROOT / "sub-MonkeyL-held-in-calib" / (
        f"sub-MonkeyL-held-in-calib_ses-{session.removeprefix('ses-')}_behavior+ecephys.nwb"
    )


def _heldout_path(session: str) -> Path:
    if session not in HELDOUT_SESSIONS:
        raise ValueError(f"unknown M1 held-out session: {session!r}")
    return DATA_ROOT / "sub-MonkeyL-held-out-calib" / (
        f"sub-MonkeyL-held-out-calib_ses-{session}_behavior+ecephys.nwb"
    )


def _nwb_loader() -> Callable[
    [Path], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    for root in (WORKSPACE_ROOT / "SPINT-main", WORKSPACE_ROOT / "streaming_calibration_exp"):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    return lambda path, _task=FalconTask.m1: load_nwb(path, _task)


def padded_windows(raw_neural: np.ndarray, starts: np.ndarray, *, pad: int = PAD) -> np.ndarray:
    """Materialize [window,100,64] raw contexts with an explicit left pad."""
    neural = np.asarray(raw_neural, dtype=np.float32)
    query_starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    if neural.ndim != 2 or neural.shape[1] != RAW_UNITS:
        raise ValueError(f"M1 raw neural must be [T,{RAW_UNITS}], got {neural.shape}")
    if np.any(query_starts < 0) or np.any(query_starts >= neural.shape[0]):
        raise ValueError("M1 query start lies outside raw timeline")
    padded = np.pad(neural, ((pad, 0), (0, 0)), mode="constant", constant_values=0.0)
    offsets = np.arange(CONTEXT, dtype=np.int64)
    return np.ascontiguousarray(padded[query_starts[:, None] + offsets], dtype=np.float32)


def _load_item(
    session: str,
    path: Path,
    loader: Callable[[Path], tuple[Any, Any, Any, Any]],
) -> dict[str, Any]:
    neural, targets, _trial_change, eval_mask = loader(path)
    raw = np.asarray(neural, dtype=np.float32)
    y_all = np.asarray(targets, dtype=np.float32)
    mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
    if raw.ndim != 2 or raw.shape[1] != RAW_UNITS:
        raise RuntimeError(f"M1 unit-column drift for {session}: {raw.shape}")
    if y_all.shape != (raw.shape[0], TARGETS) or mask.shape != (raw.shape[0],):
        raise RuntimeError(f"M1 raw array topology drift for {session}")
    starts = np.flatnonzero(mask).astype(np.int64, copy=False)
    x = np.pad(raw, ((PAD, 0), (0, 0)), mode="constant", constant_values=0.0)
    y = np.ascontiguousarray(y_all[starts], dtype=np.float32)
    return {
        "X": np.ascontiguousarray(x, dtype=np.float32),
        "Y": y,
        "starts": np.ascontiguousarray(starts, dtype=np.int64),
        "unit_mask": np.ones(RAW_UNITS, dtype=bool),
        "pad": PAD,
        "file": str(path),
        "file_sha256": _file_sha(path),
        "eval_mask_sha256": _array_sha(np.pad(mask, (PAD, 0), constant_values=False)),
    }


def _build_batches(items: Mapping[str, Mapping[str, Any]]) -> list[tuple[str, np.ndarray]]:
    """Exact SessionBatchSampler order: local shuffle, then batch shuffle."""
    flat: list[tuple[str, np.ndarray]] = []
    for session in SOURCE_SESSIONS:
        n_windows = len(items[session]["starts"])
        local = random.Random(SEED).sample(range(n_windows), n_windows)
        for offset in range(0, len(local), BATCH):
            chunk = local[offset:offset + BATCH]
            if len(chunk) == BATCH:
                flat.append((session, np.asarray(chunk, dtype=np.int64)))
    return random.Random(SEED).sample(flat, len(flat))


def sampler_digest(
    batches: Iterable[tuple[str, np.ndarray]],
    items: Mapping[str, Mapping[str, Any]],
) -> str:
    """Hash batches exactly as legacy ``m1_projadd.sampler_digest`` does."""
    offsets: dict[str, int] = {}
    cursor = 0
    for session in SOURCE_SESSIONS:
        offsets[session] = cursor
        cursor += len(items[session]["starts"])
    digest = hashlib.sha256()
    for batch_index, (session, local) in enumerate(batches):
        indices = np.ascontiguousarray(np.asarray(local, dtype=np.int64).reshape(-1) + offsets[session])
        digest.update(np.asarray([batch_index, indices.size], dtype=np.int64).tobytes())
        digest.update(indices.tobytes())
    return digest.hexdigest()


def _contract(
    items: Mapping[str, Mapping[str, Any]],
    sessions: tuple[str, ...],
    batches: list[tuple[str, np.ndarray]] | None,
) -> dict[str, Any]:
    rows = {}
    for session in sessions:
        item = items[session]
        rows[session] = {
            "window_count": int(len(item["starts"])),
            "starts_sha256": _array_sha(item["starts"]),
            "target_sha256": _array_sha(item["Y"]),
            "eval_mask_sha256": item["eval_mask_sha256"],
            "data_file": item["file"],
            "data_file_sha256": item["file_sha256"],
        }
    contract: dict[str, Any] = {
        "schema": "m1_static_raw_data_v1",
        "sessions": list(sessions),
        "total_windows": int(sum(row["window_count"] for row in rows.values())),
        "per_session": rows,
        "raw_input": {
            "dtype": "float32",
            "columns": RAW_UNITS,
            "target_columns": TARGETS,
            "context": CONTEXT,
            "pad": PAD,
            "normalization": "none",
            "target_scaling": "none",
        },
        "unit_mask": "all_true_fixed_64_raw_columns",
        "calibration": "not read or used",
        "identity": "not read or used",
    }
    if batches is not None:
        dropped_tail_windows = {
            session: int(len(items[session]["starts"]) % BATCH)
            for session in SOURCE_SESSIONS
        }
        contract["sampler"] = {
            "batch": BATCH,
            "seed": SEED,
            "shuffle": True,
            "balance": False,
            "reshuffle_each_epoch": False,
        }
        contract["updates_per_epoch"] = len(batches)
        contract["used_windows_per_epoch"] = len(batches) * BATCH
        contract["dropped_tail_windows"] = int(sum(dropped_tail_windows.values()))
        contract["dropped_tail_windows_by_session"] = dropped_tail_windows
        contract["sampler_batch_sha256"] = sampler_digest(batches, items)
    return contract


def load_source() -> dict[str, Any]:
    """Load four raw source sessions and the fixed pure-shuffle batch plan."""
    loader = _nwb_loader()
    items = {
        session: _load_item(session, _source_path(session), loader)
        for session in SOURCE_SESSIONS
    }
    batches = _build_batches(items)
    contract = _contract(items, SOURCE_SESSIONS, batches)
    if (
        contract["total_windows"] != 213336
        or contract["updates_per_epoch"] != 6665
        or contract["used_windows_per_epoch"] != 213280
        or contract["dropped_tail_windows"] != 56
    ):
        raise RuntimeError("M1 source inventory drift")
    for session in SOURCE_SESSIONS:
        row = contract["per_session"][session]
        if (
            row["window_count"] != EXPECTED_SOURCE_WINDOWS[session]
            or row["starts_sha256"] != EXPECTED_SOURCE_STARTS_SHA256[session]
        ):
            raise RuntimeError(f"M1 source query drift: {session}")
        if row["eval_mask_sha256"] != EXPECTED_SOURCE_EVAL_MASK_SHA256[session]:
            raise RuntimeError(f"M1 source eval-mask drift: {session}")
    if contract["sampler_batch_sha256"] != EXPECTED_SAMPLER_SHA256:
        raise RuntimeError("M1 source sampler drift")
    return {
        "sessions": list(SOURCE_SESSIONS),
        "items": items,
        "batches": batches,
        "contract": contract,
    }


def load_heldout() -> dict[str, Any]:
    """Load the three local held-out raw query faces without support material."""
    loader = _nwb_loader()
    items = {
        session: _load_item(session, _heldout_path(session), loader)
        for session in HELDOUT_SESSIONS
    }
    contract = _contract(items, HELDOUT_SESSIONS, None)
    if contract["total_windows"] != 3881:
        raise RuntimeError("M1 held-out inventory drift")
    for session in HELDOUT_SESSIONS:
        actual = contract["per_session"][session]
        expected = EXPECTED_HELDOUT[session]
        if any(actual[key] != expected[key] for key in expected):
            raise RuntimeError(f"M1 held-out query drift: {session}")
    return {
        "sessions": list(HELDOUT_SESSIONS),
        "items": items,
        "contract": contract,
    }


def batch_tensors(
    item: Mapping[str, Any],
    indices: np.ndarray,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw [B,100,64], endpoint [B,16], and explicit coordinate validity."""
    rows = np.asarray(indices, dtype=np.int64).reshape(-1)
    starts = np.asarray(item["starts"], dtype=np.int64)[rows]
    x_all = np.asarray(item["X"], dtype=np.float32)
    offsets = np.arange(CONTEXT, dtype=np.int64)
    x = np.ascontiguousarray(x_all[starts[:, None] + offsets], dtype=np.float32)
    y = np.ascontiguousarray(np.asarray(item["Y"], dtype=np.float32)[rows], dtype=np.float32)
    valid = starts[:, None] + offsets >= PAD
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
        torch.from_numpy(np.ascontiguousarray(valid, dtype=bool)).to(device),
    )
