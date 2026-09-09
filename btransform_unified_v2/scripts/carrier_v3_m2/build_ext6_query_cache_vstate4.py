#!/usr/bin/env python3
"""Rebuild the six-session official held-out query banks on the vstate4 path.

The X/target/starts/mapping query assets are copied byte-identically from the
sealed official ext6 query cache; only the identity banks (T.npy from the
vstate4 signed-state estimator, e0_u.pt from the remelt encoder conditioned on
that T) are rebuilt.  Statistics (rms, column normalizer) are reused from the
training-time ``carrier_normalizer.json`` that was fit on the seven held-in
sessions only; no ext6 session participates in any statistic.  Byte-equality
of the rebuilt banks against ``run_vstate4/cache/ext4`` for the four shared
sessions proves the same deterministic pipeline and encoder as training.

Reads only the six public held-out-calib NWBs through the official builder.
Never opens hidden/test records, never contacts EvalAI, never mutates the
official query cache or the dual-track cache.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT.parent
for p in (ROOT / "src", WS / "btransform_unified_v1" / "src", WS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from btransform_unified_v2 import carrier_profile_v3 as v3
from tfpd_exploration.src.m2_dual_track_v1 import champion, data as old_data, plan as old_plan

SIX = (
    "ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1", "ses-2020-11-24-Run1", "ses-2020-11-24-Run2",
)
EXT4_SHARED = ("ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1", "ses-2020-11-19-Run1")
OFFICIAL_QUERY = WS / "tfpd_exploration/results/m2_small_s1_visible_ext6_epoch_pick_v1/official_heldout_query"
RUN_CACHE = ROOT / "results/carrier_v3_m2/run_vstate4/cache"
QUERY_ASSETS = ("X_store.npy", "target_store.npy", "eligible_starts.npy", "mapping.json")
RECEIPT_SCHEMA = "m2_vstate4_ext6_official_heldout_query_v2"
DISCLOSURE = (
    "consumes dense velocity labels of calibration trials (Falcon-allowed support labels); "
    "distinct from the 688 sparse-label discipline. "
    "Plan doc: btransform_unified_v2/docs/PLAN_CARRIER_ITERATION_M2_688_20260909.md section 3.2: "
    "\u62ab\u9732\uff1a\u6d88\u8d39\u6821\u51c6 trial \u7684 dense \u901f\u5ea6\u6807\u7b7e\uff08FALCON \u5141\u8bb8\u7684 support \u6807\u7b7e\uff09\uff1b"
    "\u8fd9\u4e0e\u65e7\u6587\u6863\u4e2d 688 \u7684\"sparse-label\"\u7eaa\u5f8b\u4e0d\u662f\u540c\u4e00\u4e2a\u95ee\u9898\u3002"
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=RUN_CACHE / "ext6")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE=1 required")
    # MKL GEMM reduction order is thread-count dependent; bit-exact reproduction
    # of the training-time E0 requires a pinned thread count.
    torch.set_num_threads(int(args.torch_threads))
    dest = args.dest.resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"destination must be empty or absent: {dest}")

    variant_builder = _load_module(
        "m2_variant_cache_builder", Path(__file__).resolve().parent / "build_m2_variant_cache.py"
    )
    official_builder = _load_module(
        "m2_ext6_public_builder",
        WS / "tfpd_exploration/scripts/run_m2_small_s1_visible_ext6_epoch_pick_v1.py",
    )
    if tuple(official_builder.SIX) != SIX:
        raise RuntimeError("official ext6 builder six-session contract drift")

    normalizer_path = RUN_CACHE / "carrier_normalizer.json"
    normalizer = json.loads(normalizer_path.read_text())
    if (normalizer.get("variant") != "vstate4" or float(normalizer.get("n0", -1)) != 10.0
            or normalizer.get("estimator") != "vstate4"
            or sorted(normalizer.get("train_sessions", [])) != sorted(old_plan.HELDIN_SESSIONS)):
        raise RuntimeError("training carrier_normalizer.json is not the vstate4 held-in fit")
    mean = np.asarray(normalizer["mean"], dtype=np.float64)
    std = np.asarray(normalizer["std"], dtype=np.float64)
    rms = np.asarray(normalizer["rms"], dtype=np.float64)

    print("building official held-out dataset (six public calib NWBs)", flush=True)
    nwb_paths = official_builder._heldout_calib_nwb()
    if set(nwb_paths) != set(SIX):
        raise RuntimeError("official builder did not return exactly six public records")
    dataset, access = official_builder._build_official_heldout_dataset()
    if any(old_data.classify_nwb_role(path) == "hidden_or_test" for path in access.opened):
        raise RuntimeError("forbidden hidden/test record opened")

    print("loading remelt encoder (same loader as training cache build)", flush=True)
    encoder = variant_builder._load_e0_encoder(device="cpu")

    official_receipt = json.loads((OFFICIAL_QUERY / "official_heldout_query_banks.json").read_text())
    rows: dict[str, Any] = {}
    dest.mkdir(parents=True, exist_ok=True)
    for session in SIX:
        bundle = old_data._calib_bundle(dataset, session)
        cov = variant_builder._covariates(dataset, session)
        neural = bundle["calib_neural"]
        if cov.shape[0] != neural.shape[0]:
            raise RuntimeError(f"{session}: covariates {cov.shape} vs calib_neural {neural.shape}")
        raw, note = variant_builder._vstate_raw({**bundle, "calib_covariates": np.ascontiguousarray(cov)}, rms)
        t4 = v3.apply_column_normalizer(raw, mean, std)
        t4 = np.ascontiguousarray(t4, dtype=np.float32)

        src = OFFICIAL_QUERY / session
        mapping = json.loads((src / "mapping.json").read_text())
        if (mapping.get("query_is_padded_timeline") is not True or int(mapping.get("query_pad_bins", -1)) != 49
                or tuple(mapping.get("support_trial_ids", ())) != tuple(range(33))):
            raise RuntimeError(f"{session}: official query mapping contract drift")
        official_t = np.load(src / "T.npy")
        rel = float(np.linalg.norm(t4.astype(np.float64) - official_t.astype(np.float64))
                    / max(float(np.linalg.norm(official_t.astype(np.float64))), 1e-6))
        if rel < 1e-3:
            raise RuntimeError(f"{session}: rebuilt vstate4 T unexpectedly identical to official MOVE-T4 T")

        trials = torch.from_numpy(np.array(bundle["activity"], dtype=np.float32, copy=True))
        side = champion.empty_contrast_side(t4)
        e0, u = champion.native_e0_and_u(encoder, trials, side)

        session_dest = dest / session
        session_dest.mkdir(parents=True)
        for name in QUERY_ASSETS:
            shutil.copy2(src / name, session_dest / name)
        np.save(session_dest / "T.npy", t4)
        torch.save({"E0": e0.cpu(), "frozen_u": u.cpu()}, session_dest / "e0_u.pt")
        provenance = {
            **champion.cache_key_parts(),
            "inherited_preprocessing": champion.cache_key_parts()["preprocessing"],
            "preprocessing": "vstate4_signed_state_m33_all_trials_blocks100ms_empty_contrast_fp32",
            "estimator": "vstate4", "variant": "vstate4",
            "session_id": session, "surface": "official_heldout_query_vstate4",
            "unit_roster": "m2_96_contiguous",
            "support_ids": mapping["support_trial_ids"],
            "e0_path": "push_trial/finalize_identity",
            "e0_not_batched_mean_gemm": True,
            "e0_sha256": champion.array_sha256(e0.numpy()),
            "u_sha256": champion.array_sha256(u.numpy()),
            "t4_sha256": champion.array_sha256(t4),
            "eligible_start_sha256": old_data._digest_starts(np.load(session_dest / "eligible_starts.npy")),
            "official_move_t4_rel_frobenius": rel,
            "champion": "REF",
            "stats_provenance": {
                "carrier_normalizer": str(normalizer_path),
                "carrier_normalizer_sha256": sha(normalizer_path),
                "fit_sessions": normalizer["train_sessions"],
                "ext6_sessions_in_stats": False,
            },
            "query_assets_copied_from": str(src),
            "raw_nwb": str(nwb_paths[session]),
            "raw_nwb_sha256": sha(Path(nwb_paths[session])),
            **note,
        }
        _write_json(session_dest / "provenance.json", provenance)

        checks: dict[str, Any] = {"official_query_t_rel_frobenius": rel}
        if session in EXT4_SHARED:
            ext4 = RUN_CACHE / "ext4" / session
            t_equal = np.array_equal(t4, np.load(ext4 / "T.npy"))
            ext4_e0 = torch.load(ext4 / "e0_u.pt", map_location="cpu", weights_only=False)
            e0_equal = torch.equal(e0.cpu(), ext4_e0["E0"].cpu())
            u_equal = torch.equal(u.cpu(), ext4_e0["frozen_u"].cpu())
            if not (t_equal and e0_equal and u_equal):
                raise RuntimeError(
                    f"{session}: rebuilt bank differs from training ext4 cache "
                    f"(T={t_equal} E0={e0_equal} u={u_equal} "
                    f"e0_maxdiff={float((e0.cpu() - ext4_e0['E0'].cpu()).abs().max()):.3e})"
                )
            checks["training_ext4_T_byte_equal"] = bool(t_equal)
            checks["training_ext4_E0_byte_equal"] = bool(e0_equal)
            checks["training_ext4_u_byte_equal"] = bool(u_equal)

        official_payload = torch.load(src / "e0_u.pt", map_location="cpu", weights_only=False)
        rows[session] = {
            "window_count": int(len(np.load(session_dest / "eligible_starts.npy"))),
            "query_start_trial": 0,
            "files": {name: sha(session_dest / name) for name in (*QUERY_ASSETS, "T.npy", "e0_u.pt", "provenance.json")},
            "official_query_file_sha256": {name: sha(src / name) for name in (*QUERY_ASSETS, "T.npy", "e0_u.pt")},
            "query_assets_byte_equal_official": {
                name: sha(session_dest / name) == sha(src / name) for name in QUERY_ASSETS
            },
            "e0_official_move_t4_rel": float(np.linalg.norm(
                e0.numpy().astype(np.float64) - official_payload["E0"].numpy().astype(np.float64)
            ) / max(float(np.linalg.norm(official_payload["E0"].numpy().astype(np.float64))), 1e-6)),
            **checks,
        }
        print(f"  vstate4 bank {session} windows={rows[session]['window_count']} rel_t4={rel:.4f} checks={checks}", flush=True)
        del official_payload

    if set(rows) != set(SIX):
        raise RuntimeError("incomplete ext6 rebuild")
    receipt = {
        "schema": RECEIPT_SCHEMA, "status": "COMPLETED", "dest": str(dest),
        "six_sessions": list(SIX), "query_start_trial": 0,
        "identity": {
            "estimator": "vstate4_signed_state_m33_all_trials_blocks100ms_empty_contrast_fp32",
            "e0_encoder": "champion remelt encoder conditioned on vstate4 T (empty-contrast side)",
            "stats": "reused training-time carrier_normalizer.json (seven held-in sessions only)",
            "carrier_normalizer_sha256": sha(normalizer_path),
        },
        "disclosure": DISCLOSURE,
        "official_query_cache": str(OFFICIAL_QUERY),
        "official_query_cache_receipt_sha256": sha(OFFICIAL_QUERY / "official_heldout_query_banks.json"),
        "sessions": rows,
        "opened_public_calib_nwbs": [str(path) for path in access.opened],
        "torch_num_threads": int(args.torch_threads),
        "hidden_or_test_opened": False, "evalai_opened": False,
        "mutated_official_query_cache": False, "mutated_dual_track_cache": False,
        "mutated_training_cache": False,
        "implementation_sha256": {
            "carrier_profile_v3": sha(Path(v3.__file__)),
            "build_m2_variant_cache": sha(Path(__file__).resolve().parent / "build_m2_variant_cache.py"),
            "official_builder": sha(WS / "tfpd_exploration/scripts/run_m2_small_s1_visible_ext6_epoch_pick_v1.py"),
            "this_script": sha(Path(__file__)),
        },
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(dest / "ext6_query_receipt.json", receipt)
    os.chmod(dest / "ext6_query_receipt.json", 0o444)
    print(json.dumps({"status": "EXT6_VSTATE4_BANKS_BUILT", "dest": str(dest),
                      "receipt_sha256": sha(dest / "ext6_query_receipt.json")}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
