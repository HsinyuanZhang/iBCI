"""SLOT-AUDIT V1 cache — npz entries under a cumulative sha256-bound manifest.

Law inherited from ``src/learnable_output_filter_v1/streams.py`` (the second
bit-exact reproduction of the sealed static decode): every cache file is
written atomically, digest-bound into a cumulative ``manifest.json`` whose own
body SHA is pinned by a ``.sha256`` sidecar, and re-verified (file SHA vs the
manifest entry) on every load.  The FORMAT is this package's own (work order
section 3): one npz per (surface, session, budget) holding the FULL
``[W, 50, 2]`` prediction tensor plus the governing last-bin targets, window
starts, window validity and trial-boundary heads, and one npz per (surface,
session) holding the behavior-bin target source ``[n_bins, 2]``.

No decoder forward ever happens below this layer — bytes in, bytes out.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import plan

MANIFEST_SCHEMA = "slot_audit_v1_cache_v1"


class SlotAuditCacheError(RuntimeError):
    """Raised on any cache boundary, digest, or shape violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SlotAuditCacheError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """The lane's array digest (dtype + shape + byte SHA)."""
    payload = {
        "dtype": str(np.ascontiguousarray(array).dtype),
        "shape": list(np.ascontiguousarray(array).shape),
        "bytes_sha256": hashlib.sha256(
            np.ascontiguousarray(array).tobytes()
        ).hexdigest(),
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# manifest (cumulative, digest-bound)
# ---------------------------------------------------------------------------


def manifest_path(cache_root: Path) -> Path:
    return Path(cache_root) / "manifest.json"


def load_manifest(cache_root: Path) -> dict[str, Any]:
    path = manifest_path(cache_root)
    _require(path.exists(), f"slot audit cache manifest missing: {path}")
    body = json.loads(path.read_text(encoding="utf-8"))
    _require(body.get("schema") == MANIFEST_SCHEMA, "slot audit cache manifest schema drift")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    _require(sidecar.exists(), "slot audit cache manifest sidecar missing")
    _require(
        sidecar.read_text(encoding="ascii").strip() == f"{digest}  {path.name}",
        "slot audit cache manifest sidecar drift",
    )
    return body


def _update_manifest(
    cache_root: Path, mutate, expected_schema: str = MANIFEST_SCHEMA
) -> None:
    path = manifest_path(cache_root)
    if path.exists():
        body = json.loads(path.read_text(encoding="utf-8"))
    else:
        body = {"schema": expected_schema, "entries": {}}
    body = mutate(body)
    text = json.dumps(body, sort_keys=True, indent=2) + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")


def _add_entry(cache_root: Path, key: str, entry: dict[str, Any]) -> None:
    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        _require(key not in body["entries"], f"slot audit cache entry already present: {key}")
        body["entries"][key] = entry
        return body

    _update_manifest(cache_root, mutate)


def _write_npz(cache_root: Path, relative: str, **arrays: np.ndarray) -> Path:
    path = Path(cache_root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)
    return path


# ---------------------------------------------------------------------------
# entries
# ---------------------------------------------------------------------------


def prediction_key(surface: str, session: str, budget: int) -> str:
    return f"predictions:{surface}:{session}:m{int(budget)}"


def behavior_key(surface: str, session: str) -> str:
    return f"behavior:{surface}:{session}"


def prediction_relative(surface: str, session: str, budget: int) -> str:
    return f"predictions/{surface}/{session}/m{int(budget)}.npz"


def behavior_relative(surface: str, session: str) -> str:
    return f"behavior/{surface}/{session}.npz"


def cache_prediction(
    cache_root: Path, *, surface: str, session: str, budget: int,
    full_predictions: np.ndarray, targets: np.ndarray, starts: np.ndarray,
    valid: np.ndarray, trial_heads: np.ndarray,
) -> dict[str, Any]:
    """Write one (surface, session, budget) prediction entry + manifest row."""
    full = np.ascontiguousarray(full_predictions, dtype=np.float32)
    tgt = np.ascontiguousarray(targets, dtype=np.float32)
    st = np.ascontiguousarray(starts, dtype=np.int64)
    va = np.ascontiguousarray(valid, dtype=bool)
    hd = np.ascontiguousarray(trial_heads, dtype=bool)
    _require(
        full.ndim == 3 and full.shape[1] == plan.WINDOW_BINS and full.shape[2] == 2,
        f"prediction tensor shape drift: {full.shape}",
    )
    _require(
        tgt.shape == (full.shape[0], 2) and st.shape == (full.shape[0],)
        and va.shape == (full.shape[0],) and hd.shape == (full.shape[0],),
        "prediction entry axis drift",
    )
    relative = prediction_relative(surface, session, budget)
    path = _write_npz(
        cache_root, relative, full_predictions=full, targets=tgt, starts=st,
        valid=va, trial_heads=hd,
    )
    entry = {
        "kind": "predictions", "surface": surface, "session": session,
        "budget": int(budget), "relative": relative, "sha256": _sha256_file(path),
        "n_windows": int(full.shape[0]), "full_prediction_sha256": array_sha256(full),
        "full_prediction_bytes_sha256": hashlib.sha256(full.tobytes()).hexdigest(),
        "last_bin_prediction_sha256": array_sha256(
            np.ascontiguousarray(full[:, plan.GOVERNING_BIN, :])
        ),
        "last_bin_prediction_bytes_sha256": hashlib.sha256(
            np.ascontiguousarray(full[:, plan.GOVERNING_BIN, :]).tobytes()
        ).hexdigest(),
    }
    _add_entry(cache_root, prediction_key(surface, session, budget), entry)
    return entry


def cache_behavior(
    cache_root: Path, *, surface: str, session: str, behavior: np.ndarray,
    bin_valid: np.ndarray,
) -> dict[str, Any]:
    """Write one (surface, session) behavior-bin target entry + manifest row."""
    beh = np.ascontiguousarray(behavior, dtype=np.float32)
    bv = np.ascontiguousarray(bin_valid, dtype=bool)
    _require(
        beh.ndim == 2 and beh.shape[1] == 2 and bv.shape == (beh.shape[0],),
        "behavior entry shape drift",
    )
    relative = behavior_relative(surface, session)
    path = _write_npz(cache_root, relative, behavior=beh, bin_valid=bv)
    entry = {
        "kind": "behavior", "surface": surface, "session": session,
        "relative": relative, "sha256": _sha256_file(path),
        "n_bins": int(beh.shape[0]), "behavior_sha256": array_sha256(beh),
        "bin_valid_sha256": array_sha256(bv),
    }
    _add_entry(cache_root, behavior_key(surface, session), entry)
    return entry


def _load_entry(cache_root: Path, key: str) -> tuple[dict[str, Any], Path]:
    manifest = load_manifest(cache_root)
    _require(key in manifest["entries"], f"slot audit cache entry missing: {key}")
    entry = manifest["entries"][key]
    path = Path(cache_root) / str(entry["relative"])
    _require(
        _sha256_file(path) == entry["sha256"],
        f"slot audit cache digest drift: {key}",
    )
    return entry, path


def load_prediction(
    cache_root: Path, surface: str, session: str, budget: int,
) -> dict[str, np.ndarray]:
    entry, path = _load_entry(cache_root, prediction_key(surface, session, budget))
    with np.load(path) as payload:
        full = np.ascontiguousarray(payload["full_predictions"], dtype=np.float32)
        targets = np.ascontiguousarray(payload["targets"], dtype=np.float32)
        starts = np.ascontiguousarray(payload["starts"], dtype=np.int64)
        valid = np.ascontiguousarray(payload["valid"], dtype=bool)
        trial_heads = np.ascontiguousarray(payload["trial_heads"], dtype=bool)
    _require(
        full.shape == (int(entry["n_windows"]), plan.WINDOW_BINS, 2)
        and targets.shape == (int(entry["n_windows"]), 2)
        and starts.shape == (int(entry["n_windows"]),)
        and valid.shape == (int(entry["n_windows"]),)
        and trial_heads.shape == (int(entry["n_windows"]),),
        f"cached prediction shape drift: {surface} M{budget} {session}",
    )
    _require(
        array_sha256(full) == entry["full_prediction_sha256"],
        f"cached full-prediction array digest drift: {surface} M{budget} {session}",
    )
    return {
        "full_predictions": full, "targets": targets, "starts": starts,
        "valid": valid, "trial_heads": trial_heads,
    }


def load_behavior(cache_root: Path, surface: str, session: str) -> dict[str, np.ndarray]:
    entry, path = _load_entry(cache_root, behavior_key(surface, session))
    with np.load(path) as payload:
        behavior = np.ascontiguousarray(payload["behavior"], dtype=np.float32)
        bin_valid = np.ascontiguousarray(payload["bin_valid"], dtype=bool)
    _require(
        behavior.shape == (int(entry["n_bins"]), 2)
        and bin_valid.shape == (int(entry["n_bins"]),),
        f"cached behavior shape drift: {surface} {session}",
    )
    _require(
        array_sha256(behavior) == entry["behavior_sha256"],
        f"cached behavior array digest drift: {surface} {session}",
    )
    return {"behavior": behavior, "bin_valid": bin_valid}


def manifest_digest(cache_root: Path) -> dict[str, Any]:
    manifest = load_manifest(cache_root)
    path = manifest_path(cache_root)
    return {
        "schema": manifest["schema"],
        "entries": len(manifest["entries"]),
        "entry_keys": sorted(manifest["entries"]),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
