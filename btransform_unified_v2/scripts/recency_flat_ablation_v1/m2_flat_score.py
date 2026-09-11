#!/usr/bin/env python3
"""Isolated complete EXT6 EMA selector for the M2 R50 concat-flat run."""
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

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE, V1 = ROOT.parent, ROOT.parent / "btransform_unified_v1"
for path in (ROOT, ROOT / "src", V1 / "src", V1 / "scripts", WORKSPACE, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import m2_flat_train as train
from scripts.rift_v1 import m2_ext6_epoch_pick as frozen

SCHEMA = "m2_rift_concat_flat_ext6_epoch_pick_v1"
OUT_ROOT = ROOT / "results/recency_flat_ablation_v1"
DEFAULT_RUN = OUT_ROOT / "formal_m2_concat_flat_s42"
DEFAULT_DEST = OUT_ROOT / "selection_m2_concat_flat_ext6_s42"
REFERENCE_EXT6 = ROOT / "results/rift_v1/m2_r50_concat_s42_ext6_pick_v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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


def check_slopes(model: torch.nn.Module, label: str) -> None:
    slopes = model.temporal.recency_slopes
    if slopes.dtype != torch.float32 or tuple(slopes.shape) != (8,) or torch.count_nonzero(slopes).item() != 0:
        raise RuntimeError(f"{label}: flat slope buffer is not eight exact FP32 zeros")


def expected_query_assets(cache: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    original_manifest = read(REFERENCE_EXT6 / "manifest.json")
    assets = frozen.query_asset_hashes(cache)
    if assets != original_manifest.get("query_assets"):
        raise RuntimeError("flat selector query assets do not exactly equal frozen original EXT6 manifest")
    oracle = frozen.sealed_bt_oracle_binding(cache, assets)
    if oracle != original_manifest.get("sealed_bt_oracle"):
        raise RuntimeError("flat selector sealed query oracle does not equal frozen original EXT6 manifest")
    return assets, original_manifest


def validate_run(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    meta, receipt = read(run / "run_meta.json"), read(run / "train_receipt.json")
    required_meta = {"schema": train.SCHEMA, "status": "FORMAL", "cell": train.CELL, "seed": train.SEED,
                     "bias_mode": "flat", "context_bins": train.CONTEXT, "layer_windows": [13, 12, 12, 12],
                     "attention_backend": "local", "identity_interface": "concat", "epochs": train.EPOCHS,
                     "batch": train.BATCH, "updates_per_epoch": 3165, "flat_slopes": [0.0] * 8}
    if any(meta.get(key) != value for key, value in required_meta.items()):
        raise RuntimeError("flat run metadata contract drift")
    required_receipt = {"schema": "m2_rift_concat_flat_train_receipt_v1", "status": "COMPLETED", "cell": train.CELL,
                        "epochs": train.EPOCHS, "global_step": train.EPOCHS * 3165, "flat_slopes_zero": True}
    if any(receipt.get(key) != value for key, value in required_receipt.items()):
        raise RuntimeError("flat train receipt contract drift")
    if meta.get("source_hashes") != train.source_hashes() or receipt.get("source_hashes") != meta["source_hashes"]:
        raise RuntimeError("flat training source binding drift")
    if meta.get("paired_recency_reference") != train.reference_binding() or receipt.get("reference_binding") != meta["paired_recency_reference"]:
        raise RuntimeError("flat paired recency binding drift")
    if receipt.get("frozen_cache_hashes") != meta.get("frozen_cache_hashes"):
        raise RuntimeError("flat cache receipt binding drift")
    return meta, receipt


def validate_checkpoint(state: Mapping[str, Any], meta: Mapping[str, Any], epoch: int) -> None:
    required = {"schema": train.CHECKPOINT_SCHEMA, "cell": train.CELL, "epoch": epoch, "global_step": epoch * 3165,
                "seed": train.SEED, "context_bins": train.CONTEXT, "bias_mode": "flat", "attention_backend": "local",
                "identity_interface": "concat", "identity_e0_dim": 50, "proj_dim": None, "epochs": train.EPOCHS,
                "smoke": False, "flat_slopes": [0.0] * 8}
    if any(state.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"epoch {epoch}: flat checkpoint contract drift")
    for key in ("source_hashes", "frozen_cache_hashes", "reference_binding", "initialization_pairing", "manifest_digest"):
        expected = meta.get("paired_recency_reference") if key == "reference_binding" else meta.get(key)
        if state.get(key) != expected:
            raise RuntimeError(f"epoch {epoch}: checkpoint {key} binding drift")
    slopes = state.get("raw_state_dict", {}).get("temporal.recency_slopes")
    if not isinstance(slopes, torch.Tensor) or slopes.dtype != torch.float32 or tuple(slopes.shape) != (8,) or torch.count_nonzero(slopes).item() != 0:
        raise RuntimeError(f"epoch {epoch}: serialized flat slopes drift")
    if state.get("ema_updates") != epoch * 3165 or state.get("ema", {}).get("n_updates") != epoch * 3165:
        raise RuntimeError(f"epoch {epoch}: EMA update count drift")
    shadow = state.get("ema", {}).get("shadow")
    if not isinstance(shadow, Mapping) or not shadow or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item() for value in shadow.values()):
        raise RuntimeError(f"epoch {epoch}: EMA shadow is absent or nonfinite")


def manifest_for(run: Path, meta: Mapping[str, Any], receipt: Mapping[str, Any], cache: Path) -> dict[str, Any]:
    assets, original = expected_query_assets(cache)
    checkpoints = {str(epoch): {"path": str(run / f"epoch_{epoch:03d}.pt"), "sha256": sha(run / f"epoch_{epoch:03d}.pt")} for epoch in frozen.EPOCHS}
    return {"schema": SCHEMA + "_manifest", "run": str(run), "run_meta_sha256": sha(run / "run_meta.json"),
            "train_receipt_sha256": sha(run / "train_receipt.json"), "cell": train.CELL, "run_schema": train.SCHEMA,
            "seed": train.SEED, "bias_mode": "flat", "view": frozen.VIEW, "epochs": list(frozen.EPOCHS),
            "checkpoint_bytes": checkpoints, "query_cache": str(cache), "query_assets": assets,
            "sealed_bt_oracle": frozen.sealed_bt_oracle_binding(cache, assets),
            "original_ext6_manifest": str(REFERENCE_EXT6 / "manifest.json"), "original_ext6_manifest_sha256": sha(REFERENCE_EXT6 / "manifest.json"),
            "original_ext6_query_assets_sha256": canonical(original["query_assets"]), "paired_recency_reference": meta["paired_recency_reference"],
            "training_source_hashes": meta["source_hashes"], "selector_source_hashes": {str(Path(__file__).resolve()): sha(Path(__file__).resolve()), str(Path(frozen.__file__).resolve()): sha(Path(frozen.__file__).resolve()), str(Path(train.__file__).resolve()): sha(Path(train.__file__).resolve())},
            "official_test_used": False, "evalai_opened": False, "created_utc": datetime.now(timezone.utc).isoformat()}


def verify_manifest(manifest: Mapping[str, Any], run: Path, meta: Mapping[str, Any], receipt: Mapping[str, Any], cache: Path) -> None:
    fresh = manifest_for(run, meta, receipt, cache)
    if any(manifest.get(key) != value for key, value in fresh.items() if key != "created_utc"):
        raise RuntimeError("flat selection manifest drift")


def load_ema(model: torch.nn.Module, state: Mapping[str, Any]) -> None:
    model.load_state_dict(state["raw_state_dict"]); check_slopes(model, "post-raw-load")
    shadow = state.get("ema", {}).get("shadow")
    named = dict(model.named_parameters())
    if not isinstance(shadow, Mapping) or set(shadow) != set(named) or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item() for value in shadow.values()):
        raise RuntimeError("flat EMA parameter binding drift")
    with torch.no_grad():
        for name, value in named.items():
            value.copy_(shadow[name].to(value.device, value.dtype))
    check_slopes(model, "post-EMA-load")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic(); run_dir, dest, cache = args.run_dir.resolve(), args.dest.resolve(), args.query_cache.resolve()
    meta, receipt = validate_run(run_dir)
    candidate = manifest_for(run_dir, meta, receipt, cache)
    if dest.exists() and not args.resume:
        raise FileExistsError("flat selection destination must be fresh unless --resume is explicit")
    dest.mkdir(parents=True, exist_ok=True); manifest_path = dest / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise RuntimeError("existing flat manifest requires --resume")
        manifest = read(manifest_path); verify_manifest(manifest, run_dir, meta, receipt, cache)
    else:
        manifest = candidate; atom(manifest_path, manifest)
    duals, banks = zip(*(frozen.load_query_pair(session, cache) for session in frozen.SIX))
    dual_map, bank_map = dict(zip(frozen.SIX, duals)), dict(zip(frozen.SIX, banks))
    device = torch.device(args.device); torch.set_num_threads(args.cpu_threads)
    model = train.flat_decoder(device); check_slopes(model, "selector-init")
    progress_path = dest / "score_progress.json"
    if progress_path.exists():
        if not args.resume:
            raise RuntimeError("existing score progress requires --resume")
        progress = read(progress_path)
        if progress.get("manifest_sha256") != canonical(manifest):
            raise RuntimeError("flat score progress belongs to another manifest")
    else:
        progress = {"schema": SCHEMA + "_progress", "manifest_sha256": canonical(manifest), "completed": {}}
    for epoch in frozen.EPOCHS:
        item = manifest["checkpoint_bytes"][str(epoch)]; checkpoint_path = Path(item["path"])
        if sha(checkpoint_path) != item["sha256"]:
            raise RuntimeError(f"epoch {epoch}: checkpoint byte binding drift")
        prior = progress["completed"].get(str(epoch))
        if prior is not None:
            if prior.get("checkpoint_sha256") != item["sha256"]:
                raise RuntimeError(f"epoch {epoch}: scored checkpoint SHA drift")
            frozen.validate_complete_report(prior, manifest["query_assets"]["sessions"]); continue
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        validate_checkpoint(state, meta, epoch); load_ema(model, state)
        report = frozen.score(model, dual_map, bank_map, device)
        frozen.validate_complete_report(report, manifest["query_assets"]["sessions"])
        progress["completed"][str(epoch)] = {**report, "checkpoint_sha256": item["sha256"], "flat_slopes_zero": True}
        atom(progress_path, progress)
    frozen.validate_complete_curve(progress["completed"], manifest["query_assets"]["sessions"])
    values = {epoch: float(progress["completed"][str(epoch)]["equal_session_mean"]) for epoch in frozen.EPOCHS}
    best = max(frozen.EPOCHS, key=lambda epoch: (values[epoch], -epoch))
    verify_manifest(manifest, run_dir, meta, receipt, cache)
    selected_state = torch.load(Path(manifest["checkpoint_bytes"][str(best)]["path"]), map_location="cpu", weights_only=False)
    validate_checkpoint(selected_state, meta, best)
    package = train.flat_decoder(torch.device("cpu")); load_ema(package, selected_state)
    ema_only = {name: value.detach().cpu().clone() for name, value in package.state_dict().items()}
    check_slopes(package, "selected-EMA-package")
    package_path = dest / "selected_ema.pt"
    if package_path.exists():
        existing = torch.load(package_path, map_location="cpu", weights_only=True)
        if set(existing) != set(ema_only) or any(not torch.equal(existing[name], ema_only[name]) for name in ema_only):
            raise RuntimeError("existing flat selected EMA package differs")
    else:
        torch.save(ema_only, package_path)
    receipt_out = {"schema": SCHEMA + "_selection", "status": "COMPLETED", "manifest_sha256": canonical(manifest), "view": frozen.VIEW,
                   "selection": {"rule": "earliest maximum finite unweighted equal_session_mean", "epoch": best, "equal_session_mean": values[best]},
                   "selected": progress["completed"][str(best)], "ema_by_epoch": progress["completed"],
                   "selected_ema_state": {"path": str(package_path), "sha256": sha(package_path), "raw_state_serialized": False, "flat_slopes_zero": True},
                   "runtime_seconds": time.monotonic() - started, "device": str(device), "cpu_threads": args.cpu_threads,
                   "flat_slopes_zero": True, "query_assets_exactly_match_original_ext6": True, "official_test_used": False, "evalai_opened": False,
                   "utc": datetime.now(timezone.utc).isoformat()}
    atom(dest / "score_receipt.json", receipt_out); atom(dest / "selected_ema_receipt.json", receipt_out)
    return receipt_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN); parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--query-cache", type=Path, default=frozen.QUERY_CACHE); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
