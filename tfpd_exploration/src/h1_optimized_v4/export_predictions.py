"""Read-only native-unit prediction export for the sealed V4 formal run.

Exports the selected EMA and epoch-12 EMA as ordinary strict-loadable model
state dictionaries plus score-replay NPZ files.  It deliberately refuses to
run until selection is frozen and refuses to overwrite an export directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tfpd_exploration.src.h1_optimized_v2.cache import ROOT, build_or_load, validate_authority
from tfpd_exploration.src.h1_optimized_v2.score import CHUNK, _windows
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from .model import make_matched_pair


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def source_hashes() -> dict[str, str]:
    """Hash every implementation unit actually used by V4 export/replay."""
    here = Path(__file__).resolve()
    root = here.parents[1]
    files = {
        "v4_model": here.with_name("model.py"),
        "v4_paired_train": here.with_name("paired_train.py"),
        "v4_export": here,
        "v2_cache": root / "h1_optimized_v2/cache.py",
        "v2_score": root / "h1_optimized_v2/score.py",
        "ema": root / "h1_temporal_decoder_quick_product_v1/ema.py",
        "h1_temporal_root": root / "two_mainlines_long_v1/decoder/h1_temporal.py",
        "query_core": root / "two_mainlines_long_v1/current_query_v2/core.py",
        "query_streaming": root / "two_mainlines_long_v1/current_query_v2/streaming.py",
    }
    if any(not path.is_file() for path in files.values()):
        raise FileNotFoundError("required operator source is missing")
    return {name: sha(path) for name, path in files.items()}


def bank_mask_hashes(cache: dict) -> dict[str, dict[str, str]]:
    """Supplement legacy cache authority with exact per-session mask binding."""
    result = {}
    for split in ("train", "minival"):
        result[split] = {}
        for name, row in cache[split].items():
            mask = row["bank"]["unit_mask"].detach().cpu().contiguous().numpy()
            h = hashlib.sha256(); h.update(str(mask.dtype).encode()); h.update(str(tuple(mask.shape)).encode()); h.update(mask.tobytes())
            result[split][name] = h.hexdigest()
    return result


def assert_ema_contract(model: torch.nn.Module, ema_payload: dict) -> None:
    expected = {name for name, param in model.named_parameters() if param.requires_grad}
    actual = set(ema_payload["shadow"])
    if actual != expected:
        raise RuntimeError(f"EMA shadow/trainable-key mismatch: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
    if not hasattr(model, "frontend_contract_version") or int(model.frontend_contract_version.item()) != 4:
        raise RuntimeError("strict export requires frontend_contract_version == 4")


def native_predictions(model: torch.nn.Module, cache: dict, device: torch.device, mode: str) -> dict[str, np.ndarray]:
    parts = {"prediction_native_velocity": [], "target_native_velocity": [], "session_id": [], "bin_timestep": []}
    model.eval()
    with torch.inference_mode():
        for name, row in cache["minival"].items():
            ends = row["query_starts"] + 699 if mode == "selection" else np.flatnonzero(row["eval_mask"])
            bank = H1Bank(*[row["bank"][key].to(device) for key in ("E0", "T", "unit_mask")])
            pred = []
            for offset in range(0, len(ends), CHUNK):
                x = torch.as_tensor(_windows(row["neural"], ends[offset:offset + CHUNK]), device=device)
                pred.append((model.forward_last(x, bank) / 20.0).cpu().numpy().astype(np.float32, copy=False))
            parts["prediction_native_velocity"].append(np.concatenate(pred))
            parts["target_native_velocity"].append(np.asarray(row["velocity"][ends], dtype=np.float32))
            parts["session_id"].append(np.full(len(ends), name, dtype="U32"))
            parts["bin_timestep"].append(np.asarray(ends, dtype=np.int64))
    return {key: np.concatenate(value) for key, value in parts.items()}


def assert_ids(cache: dict, arrays: dict[str, np.ndarray], mode: str) -> None:
    expected_s, expected_t = [], []
    for name, row in cache["minival"].items():
        ends = row["query_starts"] + 699 if mode == "selection" else np.flatnonzero(row["eval_mask"])
        expected_s.append(np.full(len(ends), name, dtype="U32")); expected_t.append(np.asarray(ends, dtype=np.int64))
    s, t = np.concatenate(expected_s), np.concatenate(expected_t)
    if not np.array_equal(s, arrays["session_id"]) or not np.array_equal(t, arrays["bin_timestep"]):
        raise RuntimeError("export IDs differ from the frozen source cache")
    pairs = np.char.add(np.char.add(s, ":"), t.astype(str))
    if len(s) != (2908 if mode == "selection" else 20325) or len(np.unique(s)) != 13 or len(np.unique(pairs)) != len(pairs):
        raise RuntimeError("invalid export cardinality, session coverage, or duplicate source IDs")


def r2(prediction: np.ndarray, target: np.ndarray) -> float:
    p, y = prediction.astype(np.float64), target.astype(np.float64)
    return float(1.0 - np.square(p - y).sum() / np.square(y - y.mean(0, keepdims=True)).sum())


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--attempt", default="paired_v4_signed_12ep_v1"); parser.add_argument("--device", default="cpu")
    args = parser.parse_args(); root = ROOT / args.attempt; freeze_file = root / "selection_freeze.json"
    if not freeze_file.is_file() or not (root / "final.json").is_file():
        raise FileNotFoundError("a completed run and immutable selection_freeze.json are required")
    output = root / "independent_score_export"
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    cache = build_or_load(); authority = json.loads((ROOT / "source_cache_authority.json").read_text()); validate_authority(cache, authority)
    final_file = root / "final.json"
    freeze = json.loads(freeze_file.read_text()); device = torch.device(args.device); output.mkdir()
    manifest = {"schema": "h1_v4_independent_native_score_export_v2", "attempt": args.attempt,
                "committed_input_bindings": {"input_authority.json": sha(root / "input_authority.json"), "selection_freeze.json": sha(freeze_file), "final.json": sha(final_file), "source_cache_authority.json": sha(ROOT / "source_cache_authority.json")},
                "input_authority": authority, "operator_source_sha256": source_hashes(), "per_session_bank_unit_mask_sha256": bank_mask_hashes(cache),
                "write_semantics": "trainer checkpoint/report writes are treated as committed completed files; exporter makes no atomic-write claim",
                "surfaces": {"selection": {"n_bins": 2908}, "complete": {"n_bins": 20325}}, "arms": {}}
    pair = make_matched_pair()
    for index, arm in enumerate(("full", "t")):
        arm_manifest = {}; model = pair[index].to(device)
        paths = {"selected": Path(freeze["selected"][arm]["checkpoint"]), "epoch12": root / f"{arm}_epoch_012.pt"}
        for label, checkpoint in paths.items():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False); model.load_state_dict(payload["model"], strict=True); assert_ema_contract(model, payload["ema"])
            ema = DecoderEMA(model, decay=float(payload["ema"]["decay"])); ema.load_checkpoint_state(payload["ema"])
            state_file = output / f"{arm}_{label}_plain_ema_model_state.pt"; metadata_file = output / f"{arm}_{label}_ema_metadata.pt"; record = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint), "surfaces": {}}
            def export(view: torch.nn.Module) -> None:
                torch.save({key: value.detach().cpu().clone() for key, value in view.state_dict().items()}, state_file)
                for mode in ("selection", "complete"):
                    arrays = native_predictions(view, cache, device, mode); assert_ids(cache, arrays, mode)
                    archive = output / f"{arm}_{label}_{mode}_native.npz"; np.savez_compressed(archive, **arrays)
                    record["surfaces"][mode] = {"npz": str(archive), "sha256": sha(archive), "n_bins": int(len(arrays["bin_timestep"])), "session_count": 13, "r2_concat_float64": r2(arrays["prediction_native_velocity"], arrays["target_native_velocity"])}
            ema.score_with_ema(model, export)
            torch.save({"schema": "decoder_ema_metadata_v1", "arm": arm, "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint), "ema": ema.checkpoint_state()}, metadata_file)
            record.update({"plain_ema_model_state": str(state_file), "plain_ema_model_state_sha256": sha(state_file), "ema_metadata": str(metadata_file), "ema_metadata_sha256": sha(metadata_file), "ema_shadow_key_count": len(payload["ema"]["shadow"]), "frontend_contract_version": int(model.frontend_contract_version.item())})
            arm_manifest[label] = record
        manifest["arms"][arm] = arm_manifest
    manifest_file = output / "manifest.json"; manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n"); print(json.dumps({"manifest": str(manifest_file), "sha256": sha(manifest_file)}))


if __name__ == "__main__": main()
