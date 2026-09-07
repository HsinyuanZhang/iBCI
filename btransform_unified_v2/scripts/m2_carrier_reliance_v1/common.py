"""Score-free, public-ext4 load contract for M2 direct MOVE--T4 reliance."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

PERMUTATION_SEEDS = (101, 102, 103)
EXT4 = (
    "ses-2020-10-30-Run1", "ses-2020-10-30-Run2",
    "ses-2020-11-18-Run1", "ses-2020-11-19-Run1",
)


def array_sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def array_identity(a: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a)
    return {"sha256": array_sha256(a), "shape": list(a.shape), "dtype": str(a.dtype), "nbytes": int(a.nbytes)}


def _inventory_sha(rows: list[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for key, values in rows:
        digest.update(key.encode()); digest.update(np.asarray(values, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_tag(session: str) -> str:
    match = re.fullmatch(r"ses-(\d{4})-(\d{2})-(\d{2})-Run([12])", session)
    if not match:
        raise ValueError(f"not an M2 session: {session}")
    year, month, day, run = match.groups()
    return f"Run{run}_{year}{month}{day}"


def _packed_module(path: Path):
    spec = importlib.util.spec_from_file_location("m2_carrier_packed", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import sealed decoder {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # required for dataclasses under a file import
    spec.loader.exec_module(module)
    return module


def load_surface(*, max_endpoints_per_date_group: int = 2048) -> dict[str, Any]:
    """Read only the already-built, legal ext4 dev-calibration cache and sealed banks."""
    root = Path(__file__).resolve().parents[2]
    workspace = root.parent
    submission = workspace / "tfpd_exploration/submissions/evalai_m2_small_concat_ort_v1"
    payload_path = submission / "artifacts/m2_small_trf_s42_ema_e08_ext6.pkl"
    cache = workspace / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache/ext4"
    normalizer = workspace / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache/move_t4_normalizer.json"
    packed = _packed_module(submission / "trf_falcon_decoder.py")
    payload = packed.load_payload(payload_path)
    if payload.get("schema_version") != "m2_trf_falcon_payload_v1" or payload.get("kind") != "small":
        raise RuntimeError("not the sealed M2 SMALL concat payload")
    if int(payload.get("window_size", -1)) != 50 or float(payload.get("behavior_scaling_factor", -1)) != 5.0:
        raise RuntimeError("unexpected sealed M2 geometry/scale")
    records = []
    for session in EXT4:
        tag = dataset_tag(session)
        row = payload["bank_by_dataset_tag"].get(tag)
        if row is None:
            raise RuntimeError(f"sealed payload has no bank for {tag}")
        source = cache / session
        neural = np.load(source / "X_store.npy", mmap_mode="r")
        targets = np.load(source / "target_store.npy", mmap_mode="r")
        window_starts_padded = np.load(source / "eligible_starts.npy").astype(np.int64)
        mapping = json.loads((source / "mapping.json").read_text())
        window = int(payload["window_size"])
        pad = int(mapping["query_pad_bins"])
        unit_mask = np.asarray(row["unit_mask"], dtype=bool)
        if neural.ndim != 2 or neural.shape[1] != 96 or len(targets) != len(window_starts_padded):
            raise RuntimeError(f"bad legal ext4 cache shape for {session}")
        if window != 50 or pad != window - 1:
            raise RuntimeError(f"unexpected M2 padded-window geometry for {session}")
        window_ends_padded = window_starts_padded + (window - 1)
        if (window_starts_padded < 0).any() or (window_ends_padded >= len(neural)).any():
            raise RuntimeError(f"M2 window coordinates exceed cached neural timeline for {session}")
        if np.asarray(row["E0"]).shape != (96, 50) or np.asarray(row["T"]).shape != (96, 4) or not unit_mask.all():
            raise RuntimeError(f"unexpected sealed bank for {tag}")
        records.append({"session": session, "tag": tag, "group": session[4:14], "neural": neural,
                        "targets": targets, "window_starts_padded": window_starts_padded,
                        "window_ends_padded": window_ends_padded,
                        "window_starts_raw_unpadded": window_starts_padded - pad,
                        "window_ends_raw_unpadded": window_ends_padded - pad,
                        "query_pad_bins": pad, "window_size": window,
                        "E0": np.asarray(row["E0"], np.float32),
                        "carrier": np.asarray(row["T"], np.float32), "unit_mask": unit_mask})
    for group in sorted({r["group"] for r in records}):
        members = [r for r in records if r["group"] == group]
        total = sum(len(r["window_starts_padded"]) for r in members)
        chosen = np.unique(np.linspace(0, total - 1, min(max_endpoints_per_date_group, total), dtype=np.int64))
        cursor = 0
        for row in members:
            local = chosen[(chosen >= cursor) & (chosen < cursor + len(row["window_starts_padded"]))] - cursor
            row["selected_ordinals"] = np.ascontiguousarray(local, dtype=np.int64)
            for coord in ("window_starts_padded", "window_ends_padded", "window_starts_raw_unpadded", "window_ends_raw_unpadded"):
                row[f"selected_{coord}"] = row[coord][local]
            cursor += len(row["window_starts_padded"])
    permutations: dict[str, dict[str, list[int]]] = {}
    for group in sorted({r["group"] for r in records}):
        members = [r for r in records if r["group"] == group]
        valid = np.flatnonzero(members[0]["unit_mask"]).astype(np.int64)
        if any(not np.array_equal(valid, np.flatnonzero(r["unit_mask"])) for r in members[1:]):
            raise RuntimeError(f"M2 {group} unit masks differ")
        permutations[group] = {str(seed): np.random.default_rng(seed).permutation(valid).tolist() for seed in PERMUTATION_SEEDS}
    return {"records": records, "groups": sorted({r["group"] for r in records}),
            "max_endpoints_per_date_group": max_endpoints_per_date_group,
            "query_inventory_sha256": _inventory_sha([(r["session"], r["selected_ordinals"]) for r in records]),
            "full_valid_inventory_sha256": _inventory_sha([(r["session"], np.arange(len(r["window_starts_padded"]), dtype=np.int64)) for r in records]),
            "permutation_seeds": PERMUTATION_SEEDS, "permutations": permutations,
            "payload_path": str(payload_path.resolve()), "payload_sha256": _file_sha(payload_path),
            "normalizer_path": str(normalizer.resolve()), "normalizer_sha256": _file_sha(normalizer),
            "normalizer": json.loads(normalizer.read_text()), "submission": str(submission.resolve())}


def clone_bank_for_arm(record: dict[str, Any], arm: str, seed: int | None = None) -> dict[str, np.ndarray]:
    e0 = np.array(record["E0"], copy=True); carrier = np.array(record["carrier"], copy=True)
    mask = np.array(record["unit_mask"], copy=True); valid = np.flatnonzero(mask)
    if arm == "normal":
        pass
    elif arm == "zero":
        carrier[valid] = 0.0
    elif arm == "shuffle":
        if seed not in PERMUTATION_SEEDS:
            raise ValueError("shuffle seed must be 101/102/103")
        carrier[valid] = carrier[np.random.default_rng(seed).permutation(valid)]
    else:
        raise ValueError("arm must be normal, zero, or shuffle")
    return {"E0": e0, "carrier": carrier, "unit_mask": mask}


def paired_date_bootstrap(values: dict[str, float], *, draws: int = 2000, seed: int = 20260907) -> dict[str, float]:
    keys = sorted(values)
    if len(keys) != 3:
        raise ValueError("M2 ext4 protocol requires exactly three calendar-date groups")
    raw = np.asarray([values[k] for k in keys], dtype=np.float64)
    rng = np.random.default_rng(seed); samples = raw[rng.integers(0, len(raw), size=(draws, len(raw)))].mean(1)
    return {"equal_date_group_mean": float(raw.mean()), "bootstrap_p025": float(np.quantile(samples, .025)),
            "bootstrap_p975": float(np.quantile(samples, .975)), "draws": draws, "seed": seed}


def write_protocol_artifacts(surface: dict[str, Any], dest: Path) -> dict[str, Path]:
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    records = []
    for row in surface["records"]:
        arrays = {name: array_identity(row[name]) for name in ("neural", "targets", "window_starts_padded", "window_ends_padded", "window_starts_raw_unpadded", "window_ends_raw_unpadded", "selected_ordinals", "selected_window_starts_padded", "selected_window_ends_padded", "selected_window_starts_raw_unpadded", "selected_window_ends_raw_unpadded", "E0", "carrier", "unit_mask")}
        records.append({"session": row["session"], "tag": row["tag"], "date_group": row["group"], "arrays": arrays,
                        "valid_window_count": int(len(row["window_starts_padded"])), "selected_window_count": int(len(row["selected_ordinals"])),
                        "coordinate_contract": "selected ordinal indexes target_store directly; padded end = padded start + 49; raw unpadded coordinate = padded coordinate - 49"})
    perms = {group: {seed: array_identity(np.asarray(mapping, np.int64)) for seed, mapping in row.items()}
             for group, row in surface["permutations"].items()}
    manifest = {"schema": "m2_move_t4_concat_load_surface_v1", "status": "SCORE_FREE_PROTOCOL_FROZEN", "score_free": True,
                "official_test_opened": False, "public_data_only": True, "surface": "local visible ext4 held-out-calib only; Nov24 excluded",
                "identity": "sealed M2 SMALL concat BT-EORT (not proj_add, not old SPINT runtime)",
                "payload": {"path": surface["payload_path"], "sha256": surface["payload_sha256"], "window": 50, "channels": 96, "scale": 5.0},
                "direct_intervention": "MOVE-T4 direct SessionBank.T only; E0/weights/neural/unit-mask/normalizer unchanged. E0 may retain T4-derived information.",
                "arms": ["REAL", "T4_ZERO", "T4_SHUF101", "T4_SHUF102", "T4_SHUF103"],
                "zero_semantics": "numeric zero of the source-seven-session standardized MOVE-T4 tensor; raw-space zero/mean is not asserted",
                "normalizer": {"path": surface["normalizer_path"], "sha256": surface["normalizer_sha256"], "detail": surface["normalizer"]},
                "endpoint_selection": {"method": "uniform linspace across concatenated eligible-window ordinals within calendar date group; targets bind by selected ordinal", "max_per_date_group": surface["max_endpoints_per_date_group"], "selected_inventory_sha256": surface["query_inventory_sha256"], "full_valid_inventory_sha256": surface["full_valid_inventory_sha256"]},
                "groups": surface["groups"], "records": records, "permutations": perms}
    provenance = {"schema": "m2_move_t4_concat_provenance_v1", "load_surface": "common.load_surface(max_endpoints_per_date_group=2048)", "score_free": True,
                  "official_test_opened": False, "payload_sha256": surface["payload_sha256"], "query_inventory_sha256": surface["query_inventory_sha256"],
                  "grouping": "equal calendar-date groups: 2020-10-30 (two runs), 2020-11-18, 2020-11-19; paired bootstrap uses three date deltas"}
    paths = {"manifest": dest / "manifest.json", "provenance": dest / "provenance.json"}
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    paths["provenance"].write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return paths
