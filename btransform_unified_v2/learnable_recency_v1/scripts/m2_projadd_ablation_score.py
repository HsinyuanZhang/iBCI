#!/usr/bin/env python3
"""EXT6 EMA selector for a completed M2 proj_add identity-ablation run.

Mirrors m2_projadd_learnable_score.py and applies the SAME identity transform
to the EXT6 query pair banks that training used:

  activity_only: carrier zeroed, E0 untouched.
  norm_only:     E0 = z broadcast (z-stats are READ FROM run_meta — never
                 recomputed from held-out data); per-session support rates are
                 pooled from the frozen six-session raw-M33 directory whose
                 calib_activity is the same M33 support set behind the query
                 banks' E0 (byte-verified against the query cache and the
                 training ext4 cache by its sealed manifest).  For the four
                 ext4 sessions the recomputed rate/E0 digests must match the
                 training run_meta bit for bit.
  activity_only_empty_side: E0 is RECOMPUTED here with the same frozen
                 champion encoder and side=zeros(96,8) from the sealed raw-M33
                 support set (never read from any cached e0_u.pt).  The
                 encoder identity recorded in the training run_meta must be
                 reproduced bit for bit, and for the four ext4 sessions the
                 recomputed E0 digests must equal the training run's digests
                 bit for bit — any mismatch fails closed.

``--identity full`` runs are rejected: score them with
m2_projadd_learnable_score.py.  Imports the frozen picker and the ablation
trainer; does not edit either.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
WORKSPACE, V1 = ROOT.parent, ROOT.parent / "btransform_unified_v1"
for path in (PKG / "src", ROOT, ROOT / "src", V1 / "src", V1 / "scripts", WORKSPACE, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from btransform_unified_v1.bank import array_sha256
from learnable_recency_v1.config import config_from_run_meta
from learnable_recency_v1.wrap import LearnableRiftDecoder
from scripts.rift_v1 import m2_ext6_epoch_pick as frozen

import m2_projadd_ablation_train as ablation
import m2_projadd_learnable_score as template

RESULTS = PKG / "results"
PROJ_DIM = 16
SELECTION_SCHEMA = "m2_rift_projadd_ablation_ext6_epoch_pick_v1_selection"
EXT6_RAW_M33 = ROOT / "results/rift_v1/m2_joint_ext6_raw_m33_v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atom(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def npy_array_sha256(path: Path) -> str:
    """champion.array_sha256-compatible digest of a .npy file's array content."""
    array = np.ascontiguousarray(np.load(path, mmap_mode="r"))
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(np.asarray(array).tobytes(order="C"))
    return digest.hexdigest()


def verify_ext6_m33_root(root: Path) -> dict[str, str]:
    """Verify the sealed raw-M33 manifest; return per-session calib_activity digests."""
    manifest_path = root / "manifest.json"
    manifest = read(manifest_path)
    if manifest.get("schema") != "m2_joint_ext6_raw_m33_v1" or manifest.get("status") != "COMPLETED":
        raise RuntimeError("ext6 raw-M33 manifest schema/status drift")
    sessions = manifest.get("sessions")
    if not isinstance(sessions, Mapping) or set(sessions) != set(frozen.SIX):
        raise RuntimeError("ext6 raw-M33 manifest does not cover exactly the six query sessions")
    digests: dict[str, str] = {}
    for session in frozen.SIX:
        path = root / session / "calib_activity.npy"
        expected = str(sessions[session].get("calib_activity_sha256"))
        if npy_array_sha256(path) != expected:
            raise RuntimeError(f"{session}: ext6 raw-M33 calib_activity drift vs sealed manifest")
        digests[session] = expected
    return digests


def norm_stats_from_meta(meta: Mapping[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Recover mu/sigma exactly as recorded by the ablation trainer (no HO data touched)."""
    block = ((meta.get("identity_ablation") or {}).get("norm_only")) or {}
    mu = np.asarray(block.get("mu_src", []), dtype=np.float64)
    sigma = np.asarray(block.get("sigma_src", []), dtype=np.float64)
    if mu.shape != (96,) or sigma.shape != (96,):
        raise RuntimeError("run_meta norm_only mu/sigma arrays are missing/malformed")
    if float(block.get("sigma_eps", -1.0)) != ablation.SIGMA_EPS:
        raise RuntimeError("run_meta norm_only sigma_eps drift")
    if ablation.float64_sha256(mu) != str(block.get("mu_sha256")) or ablation.float64_sha256(sigma) != str(block.get("sigma_sha256")):
        raise RuntimeError("run_meta norm_only mu/sigma digest drift")
    return block, mu, sigma


def ext6_support_rates(root: Path, norm_block: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Pool per-session support rates from the sealed raw-M33 directory.

    The four sessions that also exist on the training ext4 surface must
    reproduce the training-run rate digests bit for bit.
    """
    recorded = (norm_block.get("rate_sha256") or {}).get("ext4") or {}
    rates: dict[str, np.ndarray] = {}
    for session in frozen.SIX:
        activity = np.load(root / session / "calib_activity.npy", mmap_mode="r")
        if tuple(int(v) for v in activity.shape) != ablation.SUPPORT_SHAPE:
            raise RuntimeError(f"{session}: ext6 raw-M33 shape {activity.shape} != {ablation.SUPPORT_SHAPE}")
        rate = ablation.pooled_rate_from_activity(np.asarray(activity))
        if session in recorded and ablation.float64_sha256(rate) != str(recorded[session]):
            raise RuntimeError(f"{session}: score-time support rate differs from the training run rate")
        rates[session] = rate
    return rates


def empty_side_block_from_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Recover the training run's empty_side record and validate its semantics."""
    block = ((meta.get("identity_ablation") or {}).get("empty_side")) or {}
    if str(block.get("side")) != ablation.EMPTY_SIDE_SEMANTICS or int(block.get("side_dim", -1)) != ablation.SIDE_DIM:
        raise RuntimeError("run_meta empty_side side semantics drift (expected zeros(N,8))")
    encoder = block.get("encoder")
    if not isinstance(encoder, Mapping) or "film_states" not in encoder or "selected_empty_head" not in encoder:
        raise RuntimeError("run_meta empty_side encoder identity missing/malformed")
    return dict(block)


def transform_query_banks(
    banks: Mapping[str, Any],
    identity: str,
    meta: Mapping[str, Any],
    raw_m33_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the training identity to the frozen query-pair TaskBanks (fresh arrays)."""
    if identity not in ablation.ABLATION_ARMS:
        raise RuntimeError(f"unsupported ablation identity for scoring: {identity!r}")
    mu = sigma = None
    rates: dict[str, np.ndarray] | None = None
    encoder: Any = None
    empty_block: dict[str, Any] | None = None
    if identity == "norm_only":
        norm_block, mu, sigma = norm_stats_from_meta(meta)
        rates = ext6_support_rates(raw_m33_root, norm_block)
    elif identity == ablation.EMPTY_SIDE_IDENTITY:
        empty_block = empty_side_block_from_meta(meta)
        encoder, fresh_encoder_record = ablation.frozen_empty_side_encoder(torch.device("cpu"))
        if dict(empty_block["encoder"]) != fresh_encoder_record:
            raise RuntimeError("score-time empty-side encoder identity differs from the training run (fail-closed)")
        verify_ext6_m33_root(raw_m33_root)
    out: dict[str, Any] = {}
    fresh_e0: dict[str, np.ndarray] = {}
    for session, bank in banks.items():
        rate = None if rates is None else rates[session]
        e0 = None
        if encoder is not None:
            e0 = ablation.empty_side_e0(encoder, raw_m33_root / session / "calib_activity.npy")
            fresh_e0[session] = e0
        out[session] = ablation.transform_bank(bank, identity, rate=rate, mu=mu, sigma=sigma, e0=e0)
    # Cross-check the four ext4-overlap sessions against the training digests.
    if identity == "norm_only":
        recorded_e0 = ((meta.get("identity_ablation") or {}).get("transformed_e0_sha256") or {}).get("ext4") or {}
        for session, digest in recorded_e0.items():
            if session in out and array_sha256(out[session].E0) != str(digest):
                raise RuntimeError(f"{session}: score-time transformed E0 differs from the training run")
    if identity == ablation.EMPTY_SIDE_IDENTITY:
        recorded_e0 = ((meta.get("identity_ablation") or {}).get("transformed_e0_sha256") or {}).get("ext4") or {}
        recorded_support = (empty_block.get("e0_sha256") or {}).get("ext4") or {}
        for session, digest in recorded_e0.items():
            if session not in out:
                continue
            if array_sha256(out[session].E0) != str(digest):
                raise RuntimeError(f"{session}: score-time empty-side E0 differs from the training run (fail-closed)")
            if session in recorded_support and ablation.float64_sha256(fresh_e0[session]) != str(recorded_support[session]):
                raise RuntimeError(f"{session}: score-time empty-side support E0 digest differs from the training run (fail-closed)")
    record = {
        "identity": identity,
        "transformed_e0_sha256": {s: array_sha256(b.E0) for s, b in out.items()},
        "transformed_carrier_sha256": {s: array_sha256(b.carrier) for s, b in out.items()},
    }
    if identity == "norm_only":
        record["mu_sha256"] = ablation.float64_sha256(mu)
        record["sigma_sha256"] = ablation.float64_sha256(sigma)
        record["rate_sha256"] = {s: ablation.float64_sha256(rates[s]) for s in out}
    if identity == ablation.EMPTY_SIDE_IDENTITY:
        record["side"] = ablation.EMPTY_SIDE_SEMANTICS
        record["side_dim"] = ablation.SIDE_DIM
        record["encoder_head_state_sha256"] = empty_block["encoder"]["head_state_sha256"]
        record["encoder_film_states_sha256"] = empty_block["encoder"]["film_states"]["sha256"]
        record["e0_sha256"] = {s: ablation.float64_sha256(e0) for s, e0 in fresh_e0.items()}
        record["ext4_e0_matches_training"] = {
            s: array_sha256(out[s].E0) == str(digest)
            for s, digest in (((meta.get("identity_ablation") or {}).get("transformed_e0_sha256") or {}).get("ext4") or {}).items()
            if s in out
        }
    return out, record


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    run_dir, dest, cache = args.run_dir.resolve(), args.dest.resolve(), args.query_cache.resolve()
    meta, receipt = read(run_dir / "run_meta.json"), read(run_dir / "train_receipt.json")
    if meta.get("schema") != ablation.SCHEMA or receipt.get("schema") != ablation.TRAIN_RECEIPT_SCHEMA:
        raise RuntimeError("score requires a completed formal proj_add identity-ablation M2 run")
    if meta.get("status") != "FORMAL" or receipt.get("status") != "COMPLETED":
        raise RuntimeError("score requires a completed formal proj_add identity-ablation M2 run")
    identity = str(meta.get("identity"))
    if identity == "full":
        raise RuntimeError("full-identity runs are scored by m2_projadd_learnable_score.py")
    if identity not in ablation.ABLATION_ARMS:
        raise RuntimeError(f"run_meta identity {identity!r} is not an ablation arm")
    if str(receipt.get("identity")) != identity:
        raise RuntimeError("train receipt identity differs from run_meta")
    template.assert_projadd_reference(meta)
    tier = meta.get("tier") or args.tier
    if dest.exists() and not args.resume:
        raise FileExistsError("ablation selection destination must be fresh unless --resume")
    dest.mkdir(parents=True, exist_ok=True)
    duals, banks = zip(*(frozen.load_query_pair(session, cache) for session in frozen.SIX))
    dual_map, bank_map_raw = dict(zip(frozen.SIX, duals)), dict(zip(frozen.SIX, banks))
    bank_map, transform_record = transform_query_banks(bank_map_raw, identity, meta, args.ext6_m33_root.resolve())
    for session, raw in bank_map_raw.items():
        if raw.E0 is bank_map[session].E0 or raw.carrier is bank_map[session].carrier:
            raise RuntimeError(f"{session}: query bank transform must not share identity arrays with the frozen bank")
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    recency_cfg = config_from_run_meta(meta, "m2")
    model = LearnableRiftDecoder("m2", recency_cfg, context_bins=50, seed=42, proj_dim=PROJ_DIM).to(device)
    model.temporal.set_attention_backend("local")
    if tuple(model.temporal_config.windows) != (13, 12, 12, 12):
        raise RuntimeError(f"ablation proj_add R50 D4 windows drift: {tuple(model.temporal_config.windows)}")
    progress_path = dest / "score_progress.json"
    progress = read(progress_path) if progress_path.exists() and args.resume else {"completed": {}}
    for epoch in frozen.EPOCHS:
        if str(epoch) in progress["completed"]:
            continue
        path = run_dir / f"epoch_{epoch:03d}.pt"
        state = torch.load(path, map_location=device, weights_only=False)
        if state.get("schema") != ablation.CHECKPOINT_SCHEMA or str(state.get("identity")) != identity:
            raise RuntimeError(f"epoch {epoch}: checkpoint is not from this ablation arm")
        model.load_state_dict(state["raw_state_dict"], strict=True)
        shadow = state.get("ema", {}).get("shadow")
        named = dict(model.named_parameters())
        if not isinstance(shadow, Mapping) or set(shadow) != set(named):
            raise RuntimeError(f"epoch {epoch}: EMA shadow does not match learnable parameters")
        with torch.no_grad():
            for name, value in named.items():
                value.copy_(shadow[name].to(value.device, value.dtype))
        report = frozen.score(model, dual_map, bank_map, device)
        frozen.validate_complete_report(report, frozen.query_asset_hashes(cache)["sessions"])
        progress["completed"][str(epoch)] = {**report, "checkpoint_sha256": sha(path)}
        atom(progress_path, progress)
    frozen.validate_complete_curve(progress["completed"], frozen.query_asset_hashes(cache)["sessions"])
    values = {epoch: float(progress["completed"][str(epoch)]["equal_session_mean"]) for epoch in frozen.EPOCHS}
    best = max(frozen.EPOCHS, key=lambda epoch: (values[epoch], -epoch))
    receipt_out = {
        "schema": SELECTION_SCHEMA,
        "status": "COMPLETED",
        "tier": tier,
        "identity": identity,
        "identity_transform": transform_record,
        "identity_interface": "proj_add",
        "proj_dim": PROJ_DIM,
        "view": frozen.VIEW,
        "ext6_m33_root": str(args.ext6_m33_root.resolve()),
        "ext6_m33_manifest_sha256": sha(args.ext6_m33_root.resolve() / "manifest.json"),
        "selection": {
            "rule": "earliest maximum finite unweighted equal_session_mean",
            "epoch": best,
            "equal_session_mean": values[best],
        },
        "ema_by_epoch": progress["completed"],
        "runtime_seconds": time.monotonic() - started,
        "official_test_used": False,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    atom(dest / "score_receipt.json", receipt_out)
    return receipt_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--tier", choices=("learned_slope", "fox_gate", "cable", "fixed"), default="learned_slope")
    parser.add_argument("--query-cache", type=Path, default=frozen.QUERY_CACHE)
    parser.add_argument("--ext6-m33-root", type=Path, default=EXT6_RAW_M33)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.dest is None:
        args.dest = RESULTS / f"selection_{args.run_dir.name}_ext6"
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
