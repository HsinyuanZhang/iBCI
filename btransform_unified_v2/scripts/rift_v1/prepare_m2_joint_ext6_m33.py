#!/usr/bin/env python3
"""Build the sealed six-session raw-M33 input for the M2 joint B/D picker.

The only public-record reader is the same official held-out-calib builder used
to create the ext6 query cache.  This program writes a fresh directory; it
never changes the dual-track cache or the query cache, and never scores a
decoder or opens EvalAI.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT.parent
for path in (ROOT, ROOT / "src", WS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tfpd_exploration.src.m2_dual_track_v1 import champion, data, plan
from tfpd_exploration.src.m2_hold_film_probe_v1.encoder import HoldContrastFiLMEarlyPoolEncoder

QUERY = WS / "tfpd_exploration/results/m2_small_s1_visible_ext6_epoch_pick_v1/official_heldout_query"
CACHE = WS / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache"
SIX = tuple(plan.EXTERNAL_SESSIONS)
EXT4 = set(plan.EXT4_SESSIONS)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arr_sha(value: np.ndarray) -> str:
    return champion.array_sha256(np.asarray(value))


def atomic(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def builder_module():
    source = WS / "tfpd_exploration/scripts/run_m2_small_s1_visible_ext6_epoch_pick_v1.py"
    spec = importlib.util.spec_from_file_location("m2_ext6_public_builder", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load official ext6 query builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.SIX) != SIX:
        raise RuntimeError("official ext6 builder six-session contract drift")
    return module, source


def source_binding(builder_source: Path) -> dict[str, str]:
    import tfpd_exploration.src.m2_hold_film_probe_v1.encoder as encoder
    feature_source = WS / "streaming_calibration_exp/src/data/falcon_t4_features.py"
    normalizer = CACHE / "move_t4_normalizer.json"
    paths = (Path(__file__), builder_source, Path(data.__file__), Path(champion.__file__), Path(plan.__file__),
             Path(encoder.__file__), feature_source, normalizer)
    return {str(path.resolve()): sha(path) for path in paths}


def frozen_encoder_binding() -> dict[str, Any]:
    files = {
        "film_states": WS / plan.FILM_STATES_RELATIVE,
        "selected_empty_head": WS / plan.SELECTED_HEAD_RELATIVE,
    }
    return {"identity_provider": "HoldContrastFiLMEarlyPoolEncoder(100,50,64,side8,rank8,num_post3,t4_plus_contrast)+canonical_p0_empty_head",
            "selected_head_state_sha256_expected": plan.SELECTED_HEAD_STATE_SHA256,
            "files": {name: {"path": str(path), "sha256": sha(path)} for name, path in files.items()}}


def build_joint_production_encoder() -> HoldContrastFiLMEarlyPoolEncoder:
    """Exact encoder construction used by `m2_joint_train.py`, without its absent legacy checkpoint."""
    encoder = HoldContrastFiLMEarlyPoolEncoder(100, 50, 64, side_dim=8, film_rank=8,
                                               num_post_layers=3, film_input="t4_plus_contrast")
    meta = champion.overlay_canonical_p0_and_empty_head(encoder)
    if meta["head_state_sha256"] != plan.SELECTED_HEAD_STATE_SHA256:
        raise RuntimeError("joint canonical p0/EMPTY head checksum drift")
    return encoder.eval()


def write_readme(dest: Path) -> None:
    (dest / "README.md").write_text(
        "# M2 joint ext6 raw-M33 input\n\n"
        "This directory is an immutable, fresh six-session input for joint B/D epoch selection. "
        "Each session contains the first 33 public held-out-calib neural trials as `calib_activity.npy`, "
        "plus independently rebuilt native MOVE-T4 `T.npy` and frozen-native-encoder `e0_u.pt`. "
        "`manifest.json` seals raw NWB paths and hashes, source/normalizer/checkpoint hashes, trial IDs, "
        "geometry, and byte-equality checks against the query cache. No decoder was scored.\n",
        encoding="utf-8",
    )


def run(dest: Path) -> dict[str, Any]:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE=1 required")
    dest = dest.resolve()
    if dest.exists():
        raise FileExistsError(f"fresh destination required: {dest}")
    if not (QUERY / "official_heldout_query_banks.json").is_file():
        raise FileNotFoundError(QUERY)
    builder, builder_source = builder_module()
    nwb_paths = builder._heldout_calib_nwb()
    if set(nwb_paths) != set(SIX):
        raise RuntimeError("official builder did not return exactly six public records")
    dataset, access = builder._build_official_heldout_dataset()
    if any(data.classify_nwb_role(path) == "hidden_or_test" for path in access.opened):
        raise RuntimeError("forbidden hidden/test record opened")
    # FileAccessLog records resolved strings while the builder returns Paths.
    # The Falcon builder invokes its tracked load path twice for each one
    # public record (data and trial metadata); that is legal only when every
    # one of the exact six allowlisted paths occurs exactly twice.
    expected_opens = {str(Path(path).resolve()) for path in nwb_paths.values()}
    actual_counts = Counter(str(Path(path).resolve()) for path in access.opened)
    if set(actual_counts) != expected_opens or set(actual_counts.values()) != {2}:
        raise RuntimeError(f"actual source access differs from official six-record double-read contract: {dict(actual_counts)}")
    normalizer_path = CACHE / "move_t4_normalizer.json"
    normalizer = json.loads(normalizer_path.read_text(encoding="utf-8"))
    mean = np.asarray(normalizer["mean"], dtype=np.float32)
    std = np.asarray(normalizer["std"], dtype=np.float32)
    encoder = build_joint_production_encoder()
    dest.mkdir(parents=True)
    write_readme(dest)
    rows: dict[str, Any] = {}
    for session in SIX:
        bundle = data._calib_bundle(dataset, session)
        activity = np.ascontiguousarray(bundle["activity"], dtype=np.float32)
        if activity.shape != (33, 100, 96):
            raise RuntimeError(f"{session}: raw M33 shape drift {activity.shape}")
        support_ids = list(range(33))
        move_t4 = champion.fit_move_t4(bundle["calib_neural"], bundle["calib_trial_change"], bundle["angles"],
                                        session=session, mean=mean, std=std)
        side = champion.empty_contrast_side(move_t4)
        e0, frozen_u = champion.native_e0_and_u(encoder, torch.from_numpy(activity), side)
        e0_np = np.ascontiguousarray(e0.numpy(), dtype=np.float32)
        t_np = np.ascontiguousarray(move_t4, dtype=np.float32)
        query_dir = QUERY / session
        query_e0 = torch.load(query_dir / "e0_u.pt", map_location="cpu", weights_only=False)["E0"].numpy()
        query_t = np.load(query_dir / "T.npy")
        e0_equal = np.array_equal(e0_np, query_e0)
        t_equal = np.array_equal(t_np, query_t)
        if not e0_equal or not t_equal:
            raise RuntimeError(f"{session}: reconstructed E0/T differs from sealed query cache")
        session_dest = dest / session
        session_dest.mkdir()
        np.save(session_dest / "calib_activity.npy", activity)
        np.save(session_dest / "T.npy", t_np)
        torch.save({"E0": torch.from_numpy(e0_np), "frozen_u": frozen_u.cpu()}, session_dest / "e0_u.pt")
        ext4 = CACHE / "ext4" / session
        ext4_checks: dict[str, bool] | None = None
        if session in EXT4:
            ext4_checks = {
                "calib_activity_byte_equal": np.array_equal(activity, np.load(ext4 / "calib_activity.npy")),
                "T_byte_equal": np.array_equal(t_np, np.load(ext4 / "T.npy")),
            }
            if not all(ext4_checks.values()):
                raise RuntimeError(f"{session}: rebuilt M33 differs from training ext4 cache")
        rows[session] = {
            "raw_nwb": str(nwb_paths[session]), "raw_nwb_sha256": sha(nwb_paths[session]),
            "support_trial_ids": support_ids, "shape": list(activity.shape),
            "calib_activity_sha256": arr_sha(activity), "T_sha256": arr_sha(t_np), "E0_sha256": arr_sha(e0_np),
            "frozen_u_sha256": arr_sha(frozen_u.numpy()), "query_cache": {
                "E0_byte_equal": e0_equal, "T_byte_equal": t_equal,
                "E0_sha256": arr_sha(np.asarray(query_e0)), "T_sha256": arr_sha(np.asarray(query_t)),
            }, "training_ext4": ext4_checks,
        }
    if set(rows) != set(SIX):
        raise RuntimeError("incomplete M33 artifact")
    manifest = {
        "schema": "m2_joint_ext6_raw_m33_v1", "status": "COMPLETED", "dest": str(dest),
        "sessions": rows, "six_sessions": list(SIX), "source_sha256": source_binding(builder_source),
        "normalizer": {"path": str(normalizer_path), "sha256": sha(normalizer_path), "mean": normalizer["mean"], "std": normalizer["std"]},
        "frozen_native_encoder": frozen_encoder_binding(), "query_cache": str(QUERY),
        "query_cache_receipt_sha256": sha(QUERY / "official_heldout_query_banks.json"),
        "opened_public_calib_nwbs": [str(path) for path in access.opened],
        "opened_public_calib_nwb_counts": dict(sorted(actual_counts.items())),
        "public_record_read_contract": "exactly six allowlisted resolved paths; each Falcon data+trial-metadata load records twice",
        "hidden_or_test_opened": False,
        "evalai_opened": False, "official_test_used": False, "decoder_scored": False,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic(dest / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="prepare sealed raw M33 for M2 joint ext6")
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
