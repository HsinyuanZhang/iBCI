#!/usr/bin/env python3
"""EXT6 EMA selector for a completed M2 proj_add (P16) learnable-recency run. Imports frozen pick/score; does not edit them."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
WORKSPACE, V1 = ROOT.parent, ROOT.parent / "btransform_unified_v1"
for path in (PKG / "src", ROOT, ROOT / "src", V1 / "src", V1 / "scripts", WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from learnable_recency_v1.config import config_from_run_meta
from learnable_recency_v1.p8 import maybe_truncate_p8
from learnable_recency_v1.wrap import LearnableRiftDecoder
from scripts.rift_v1 import m2_ext6_epoch_pick as frozen

RESULTS = PKG / "results"
PROJ_DIM = 16
SELECTION_SCHEMA = "m2_rift_projadd_learnable_ext6_epoch_pick_v1_selection"
REFERENCE = ROOT / "results/rift_v1/m2_r50_recency_s42_formal_v1"


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


def assert_projadd_reference(meta: Mapping[str, Any]) -> int:
    """The scored run must be a proj_add learnable run bound to the frozen proj_add recency reference.

    The run's own proj_dim is a free knob (recorded in run_meta); the frozen
    reference itself stays P16. Returns the run's proj_dim.
    """
    if meta.get("identity_interface") != "proj_add":
        raise RuntimeError("score requires a proj_add learnable run")
    try:
        run_proj_dim = int(meta.get("proj_dim", 16))
    except (TypeError, ValueError) as error:
        raise RuntimeError("run_meta proj_dim must be an integer") from error
    if run_proj_dim <= 0:
        raise RuntimeError("run_meta proj_dim must be positive")
    paired = meta.get("paired_recency_reference")
    if not isinstance(paired, Mapping) or paired.get("run") != str(REFERENCE):
        raise RuntimeError("learnable run is not bound to the frozen proj_add recency reference")
    reference = read(REFERENCE / "run_meta.json")
    if (reference.get("identity_interface") != "proj_add" or int(reference.get("proj_dim", -1)) != PROJ_DIM
            or reference.get("status") != "FORMAL" or reference.get("cell") != "M2-RIFT-R50-D4-P16-RECENCY-V1"):
        raise RuntimeError("frozen proj_add recency reference interface drift")
    return run_proj_dim


def rebuild_model(meta: Mapping[str, Any], recency_cfg, device: torch.device) -> LearnableRiftDecoder:
    """Rebuild the learnable decoder for scoring from the run's recorded seed/proj_dim."""
    proj_dim = int(meta.get("proj_dim", 16))
    build_dim = 16 if proj_dim == 8 else proj_dim
    model = LearnableRiftDecoder(
        "m2", recency_cfg, context_bins=50, seed=int(meta.get("seed", 42)), proj_dim=build_dim
    ).to(device)
    maybe_truncate_p8(model, proj_dim)
    model.temporal.set_attention_backend("local")
    if tuple(model.temporal_config.windows) != (13, 12, 12, 12):
        raise RuntimeError(f"learnable proj_add R50 D4 windows drift: {tuple(model.temporal_config.windows)}")
    return model


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    run_dir, dest, cache = args.run_dir.resolve(), args.dest.resolve(), args.query_cache.resolve()
    meta, receipt = read(run_dir / "run_meta.json"), read(run_dir / "train_receipt.json")
    if meta.get("status") != "FORMAL" or receipt.get("status") != "COMPLETED":
        raise RuntimeError("score requires a completed formal learnable proj_add M2 run")
    assert_projadd_reference(meta)
    tier = meta.get("tier") or args.tier
    if dest.exists() and not args.resume:
        raise FileExistsError("learnable selection destination must be fresh unless --resume")
    dest.mkdir(parents=True, exist_ok=True)
    duals, banks = zip(*(frozen.load_query_pair(session, cache) for session in frozen.SIX))
    dual_map, bank_map = dict(zip(frozen.SIX, duals)), dict(zip(frozen.SIX, banks))
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    recency_cfg = config_from_run_meta(meta, "m2")
    model = rebuild_model(meta, recency_cfg, device)
    progress_path = dest / "score_progress.json"
    progress = read(progress_path) if progress_path.exists() and args.resume else {"completed": {}}
    for epoch in frozen.EPOCHS:
        if str(epoch) in progress["completed"]:
            continue
        path = run_dir / f"epoch_{epoch:03d}.pt"
        state = torch.load(path, map_location=device, weights_only=False)
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
        "identity_interface": "proj_add",
        "proj_dim": int(model.proj_dim),
        "view": frozen.VIEW,
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.dest is None:
        args.dest = RESULTS / f"selection_m2_projadd_{args.tier}_ext6_s42"
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
