#!/usr/bin/env python3
"""EXT6 epoch-pick for the frozen proj_add (P16) recency reference run.

The reference's own score_receipt is an ext4 selection; this re-scores its 24
checkpoints on the same frozen EXT6 query cache used by the learnable arms so
all proj_add numbers share one face. Read-only over the reference run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent

import sys

for path in (PKG / "src", ROOT / "src", ROOT, ROOT.parent, ROOT.parent / "btransform_unified_v1/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from btransform_unified_v2.model import RiftDecoder
from scripts.rift_v1 import m2_ext6_epoch_pick as frozen

PROJ_DIM = 16
RESULTS = PKG / "results"


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    run_dir, dest, cache = args.run_dir.resolve(), args.dest.resolve(), args.query_cache.resolve()
    meta = json.loads((run_dir / "run_meta.json").read_text())
    needed = {"schema": "m2_rift_train_v1", "status": "FORMAL", "variant": "recency",
              "identity_interface": "proj_add", "proj_dim": PROJ_DIM, "seed": 42,
              "context_bins": 50, "epochs": 24, "updates_per_epoch": 3165}
    if any(meta.get(key) != value for key, value in needed.items()):
        raise RuntimeError("reference run is not the frozen proj_add P16 recency formal run")
    if dest.exists():
        raise FileExistsError("reference EXT6 destination must be fresh")
    dest.mkdir(parents=True, exist_ok=True)
    duals, banks = zip(*(frozen.load_query_pair(session, cache) for session in frozen.SIX))
    dual_map, bank_map = dict(zip(frozen.SIX, duals)), dict(zip(frozen.SIX, banks))
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    model = RiftDecoder("m2", context_bins=50, bias_mode="recency", seed=42, proj_dim=PROJ_DIM).to(device)
    model.temporal.set_attention_backend("local")
    progress_path = dest / "score_progress.json"
    progress: dict[str, Any] = {"completed": {}}
    for epoch in frozen.EPOCHS:
        path = run_dir / f"epoch_{epoch:03d}.pt"
        state = torch.load(path, map_location=device, weights_only=False)
        if state.get("schema") != "m2_rift_epoch_checkpoint_v1" or int(state.get("proj_dim", -1)) != PROJ_DIM:
            raise RuntimeError(f"epoch {epoch}: not a proj_add P16 reference checkpoint")
        model.load_state_dict(state["raw_state_dict"], strict=True)
        shadow = state.get("ema", {}).get("shadow")
        named = dict(model.named_parameters())
        if not isinstance(shadow, Mapping) or set(shadow) != set(named):
            raise RuntimeError(f"epoch {epoch}: EMA shadow does not match reference parameters")
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
    receipt = {
        "schema": "m2_rift_projadd_reference_ext6_epoch_pick_v1",
        "status": "COMPLETED",
        "tier": "reference_fixed_recency",
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
    atom(dest / "score_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "results/rift_v1/m2_r50_recency_s42_formal_v1")
    parser.add_argument("--dest", type=Path,
                        default=RESULTS / "selection_m2_projadd_reference_ext6_s42")
    parser.add_argument("--query-cache", type=Path, default=frozen.QUERY_CACHE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
