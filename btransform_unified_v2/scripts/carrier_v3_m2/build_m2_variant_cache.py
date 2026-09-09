#!/usr/bin/env python3
"""Build an M2 dual-track cache variant: new T.npy + remelted e0_u.pt, rest symlinked."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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

BIN_SECONDS = 0.02
MOVE_DURATION_S = (old_plan.MOVE_T4_STOP_BIN - old_plan.MOVE_T4_START_BIN) * BIN_SECONDS
BLOCK_BINS = 5
BLOCK_SECONDS = BLOCK_BINS * BIN_SECONDS
N0_U1 = 1.0
N0_VSTATE = 10.0
LINK_NAMES = (
    "X_store.npy", "target_store.npy", "eligible_starts.npy",
    "calib_activity.npy", "mapping.json", "extra.json",
)
SRC_CACHE = WS / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache"
VARIANTS = ("u1", "vstate4")
PREPROCESSING = {
    "u1": "u1_poisson_n0=1_harmonic_m33_bins[5,30)_empty_contrast_fp32",
    "vstate4": "vstate4_signed_state_m33_all_trials_blocks100ms_empty_contrast_fp32",
}


def _load_e0_encoder(device: str = "cpu"):
    """Remelt encoder is p0 + EMPTY head. Lightning champion is optional."""
    ckpt = old_plan.repo_root() / old_plan.CHAMPION_CKPT_RELATIVE
    if ckpt.is_file():
        try:
            return champion.load_frozen_champion(device=device).student.id_encoder
        except Exception as exc:
            print(f"lightning champion load failed ({exc}); falling back to p0+empty FiLM", flush=True)
    else:
        print("champion ckpt absent; loading remelt encoder from p0+empty FiLM", flush=True)
    from tfpd_exploration.src.m2_hold_film_probe_v1.encoder import (
        HoldContrastFiLMEarlyPoolEncoder,
    )

    encoder = HoldContrastFiLMEarlyPoolEncoder(
        trial_length=old_plan.CALIB_TRIAL_LENGTH,
        window_size=old_plan.WINDOW,
        hidden_dim=old_plan.HIDDEN_DIM,
        side_dim=champion.SIDE_DIM,
        film_rank=champion.FILM_RANK,
        num_post_layers=3,
        film_input="t4_plus_contrast",
    )
    meta = champion.overlay_canonical_p0_and_empty_head(encoder)
    encoder.eval()
    encoder.to(torch.device(device))
    print(f"p0+empty remelt encoder ready keys={meta.get('p0_keys')}", flush=True)
    return encoder


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _covariates(dataset: Any, session: str) -> np.ndarray:
    if hasattr(dataset, "calib_covariates") and session in getattr(dataset, "calib_covariates"):
        return np.asarray(dataset.calib_covariates[session], dtype=np.float64)
    raise RuntimeError(f"{session}: dataset has no calib_covariates")


def _collect_session_inputs() -> dict[str, dict[str, np.ndarray]]:
    source_log = old_data.FileAccessLog(role="source")
    ext_log = old_data.FileAccessLog(role="ext4")
    data_module = old_data.construct_source_datamodule(access=source_log)
    train_ds = data_module.train_dataset
    out: dict[str, dict[str, np.ndarray]] = {}
    for session in old_plan.HELDIN_SESSIONS:
        bundle = old_data._calib_bundle(train_ds, session)
        cov = _covariates(train_ds, session)
        neural = bundle["calib_neural"]
        if cov.shape[0] != neural.shape[0]:
            raise RuntimeError(
                f"{session}: calib_covariates {cov.shape} vs calib_neural {neural.shape}"
            )
        out[session] = {**bundle, "calib_covariates": np.ascontiguousarray(cov)}
    del data_module, train_ds
    ext_ds = old_data._build_ext4_dataset(ext_log)
    for session in old_plan.EXT4_SESSIONS:
        bundle = old_data._calib_bundle(ext_ds, session)
        cov = _covariates(ext_ds, session)
        neural = bundle["calib_neural"]
        if cov.shape[0] != neural.shape[0]:
            raise RuntimeError(
                f"{session}: calib_covariates {cov.shape} vs calib_neural {neural.shape}"
            )
        out[session] = {**bundle, "calib_covariates": np.ascontiguousarray(cov)}
    del ext_ds
    return out


def _u1_raw(bundle: dict[str, np.ndarray], session: str) -> tuple[np.ndarray, dict[str, Any]]:
    sums, lengths, angles = champion.move_t4_trial_sums(
        bundle["calib_neural"], bundle["calib_trial_change"], bundle["angles"], session=session
    )
    usable = np.isfinite(angles)
    if int(usable.sum()) < 3:
        raise RuntimeError(f"{session}: fewer than 3 directional MOVE trials")
    rates = np.asarray(sums[usable], dtype=np.float64) / np.asarray(lengths[usable], dtype=np.float64)[:, None]
    z, rate_mean, noise_rate = v3.poisson_standardize(rates, MOVE_DURATION_S)
    dirs = np.asarray([v3.nearest_canonical_index(float(a)) for a in angles[usable]], dtype=np.int64)
    response = v3.conditional_response(z, v3.one_hot_directions(dirs), N0_U1)
    a, c, m = v3.harmonic_readout(response)
    raw = v3.stack_t4(a, c, m, rate_mean / noise_rate)
    champion.ensure_streaming_on_path()
    from src.data.falcon_t4_features import t4_from_trial_sums
    legacy = t4_from_trial_sums(sums, lengths, angles, source=f"{session}:u1-compare")
    max_abs = float(np.max(np.abs(raw.astype(np.float64) - legacy.astype(np.float64))))
    return raw, {"n_directional": int(usable.sum()), "legacy_max_abs": max_abs, "n_dirs": int(len(set(dirs.tolist())))}


def _blocks(bundle: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    neural = np.asarray(bundle["calib_neural"], dtype=np.float64)
    cov = np.asarray(bundle["calib_covariates"], dtype=np.float64)
    if cov.ndim != 2 or cov.shape[1] < 2:
        raise RuntimeError(f"expected velocity covariates [T,>=2], got {cov.shape}")
    cov = cov[:, :2]
    starts = np.flatnonzero(np.asarray(bundle["calib_trial_change"], dtype=bool))
    ends = np.r_[starts[1:], neural.shape[0]]
    if starts.size < old_plan.SUPPORT_HORIZON:
        raise RuntimeError(f"need M33 trial starts, got {starts.size}")
    rates: list[np.ndarray] = []
    vels: list[np.ndarray] = []
    for trial in range(old_plan.SUPPORT_HORIZON):
        chunk_n = neural[int(starts[trial]):int(ends[trial])]
        chunk_v = cov[int(starts[trial]):int(ends[trial])]
        n_blocks = len(chunk_n) // BLOCK_BINS
        for i in range(n_blocks):
            sl = slice(i * BLOCK_BINS, (i + 1) * BLOCK_BINS)
            rates.append(chunk_n[sl].sum(axis=0) / BLOCK_SECONDS)
            vels.append(chunk_v[sl].mean(axis=0))
    if not rates:
        raise RuntimeError("no 100 ms blocks")
    return np.asarray(rates, dtype=np.float64), np.asarray(vels, dtype=np.float64)


def _vstate_raw(bundle: dict[str, np.ndarray], rms: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    rates, vels = _blocks(bundle)
    z, _, _ = v3.poisson_standardize(rates, BLOCK_SECONDS)
    weights = v3.signed_state_weights(vels, rms)
    response = v3.conditional_response(z, weights, N0_VSTATE)
    a = response[:, 0] - response[:, 1]
    c = response[:, 2] - response[:, 3]
    raw = v3.stack_t4(a, c, np.hypot(a, c), response.mean(axis=1))
    return raw, {"n_blocks": int(len(rates))}


def _symlink_session(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in LINK_NAMES:
        target = src / name
        if not target.exists():
            raise RuntimeError(f"missing source cache file {target}")
        link = dest / name
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(target.resolve(), link)


def _write_identity(dest: Path, t4: np.ndarray, encoder, session: str, surface: str, extra: dict[str, Any]) -> float:
    mapping = json.loads((dest / "mapping.json").read_text())
    activity = old_data.read_memmap(dest / "calib_activity.npy")
    trials = torch.from_numpy(np.array(activity, dtype=np.float32, copy=True))
    side = champion.empty_contrast_side(t4)
    e0, u = champion.native_e0_and_u(encoder, trials, side)
    native_t = np.load(SRC_CACHE / surface / session / "T.npy")
    rel = float(np.linalg.norm(t4 - native_t) / max(float(np.linalg.norm(native_t)), 1e-6))
    inherited = champion.cache_key_parts()
    variant = str(extra.get("variant") or "")
    provenance = {
        **inherited,
        "inherited_preprocessing": inherited.get("preprocessing"),
        "preprocessing": PREPROCESSING.get(variant, inherited.get("preprocessing")),
        "estimator": variant,
        "session_id": session,
        "surface": surface,
        "unit_roster": "m2_96_contiguous",
        "support_ids": mapping["support_trial_ids"],
        "e0_path": "push_trial/finalize_identity",
        "e0_not_batched_mean_gemm": True,
        "e0_sha256": champion.array_sha256(e0.numpy()),
        "u_sha256": champion.array_sha256(u.numpy()),
        "t4_sha256": champion.array_sha256(np.asarray(t4, dtype=np.float32)),
        "eligible_start_sha256": old_data._digest_starts(np.load(dest / "eligible_starts.npy")),
        "native_t4_rel_frobenius": rel,
        **extra,
    }
    provenance["preprocessing"] = PREPROCESSING.get(str(provenance.get("variant") or variant), provenance["preprocessing"])
    torch.save({"E0": e0.cpu(), "frozen_u": u.cpu()}, dest / "e0_u.pt")
    _write_json(dest / "provenance.json", provenance)
    return rel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), required=True)
    parser.add_argument("--dest", type=Path, help="run root (required unless --variant all)")
    parser.add_argument("--dest-root", type=Path,
                        default=ROOT / "results" / "carrier_v3_m2")
    args = parser.parse_args()
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    dests = {}
    if args.variant == "all":
        for name in variants:
            dests[name] = (args.dest_root / f"run_{name}").resolve()
    else:
        if args.dest is None:
            raise ValueError("--dest is required unless --variant all")
        dests[args.variant] = args.dest.resolve()
    for dest_root in dests.values():
        if dest_root.exists() and any(dest_root.iterdir()):
            raise FileExistsError(f"destination must be empty: {dest_root}")

    print("collecting M2 calib arrays", flush=True)
    bundles = _collect_session_inputs()
    print("loading M2 remelt encoder for E0", flush=True)
    encoder = _load_e0_encoder(device="cpu")
    held_vel = [_blocks(bundles[s])[1] for s in old_plan.HELDIN_SESSIONS]
    rms = np.maximum(np.sqrt(np.mean(np.square(np.concatenate(held_vel, axis=0)), axis=0)), v3.STD_FLOOR)

    for variant in variants:
        dest_root = dests[variant]
        cache_dest = dest_root / "cache"
        raw: dict[str, np.ndarray] = {}
        notes: dict[str, Any] = {}
        for session, bundle in bundles.items():
            if variant == "u1":
                raw[session], notes[session] = _u1_raw(bundle, session)
            else:
                raw[session], notes[session] = _vstate_raw(bundle, rms)
            print(f"  {variant} raw {session} {notes[session]}", flush=True)
        mean, std = v3.fit_column_normalizer([raw[s] for s in old_plan.HELDIN_SESSIONS])
        distances: dict[str, float] = {}
        surfaces = (
            ("source_train", old_plan.HELDIN_SESSIONS),
            ("source_minival", old_plan.HELDIN_SESSIONS),
            ("ext4", old_plan.EXT4_SESSIONS),
        )
        for surface, sessions in surfaces:
            for session in sessions:
                src = SRC_CACHE / surface / session
                dest = cache_dest / surface / session
                _symlink_session(src, dest)
                t4 = v3.apply_column_normalizer(raw[session], mean, std)
                np.save(dest / "T.npy", t4)
                distances[f"{surface}/{session}"] = _write_identity(
                    dest, t4, encoder, session, surface, {"variant": variant, "champion": "REF"}
                )
        normalizer = {
            "variant": variant,
            "train_sessions": list(old_plan.HELDIN_SESSIONS),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "n0": N0_U1 if variant == "u1" else N0_VSTATE,
            "estimator": variant,
            "preprocessing": PREPROCESSING[variant],
            "inherited_preprocessing": "move_t4_bins[5,30)_empty_contrast_fp32",
            "rms": rms.tolist(),
            "source_cache": str(SRC_CACHE),
            "source_cache_sha256": _sha_file(SRC_CACHE / "move_t4_normalizer.json") if (SRC_CACHE / "move_t4_normalizer.json").is_file() else None,
            "implementation_sha256": _sha_file(Path(v3.__file__)),
            "builder_sha256": _sha_file(Path(__file__)),
            "session_notes": notes,
            "e0_native_rel_frobenius": distances,
            "built_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(cache_dest / "carrier_normalizer.json", normalizer)
        print(f"wrote {dest_root}", flush=True)


if __name__ == "__main__":
    main()
