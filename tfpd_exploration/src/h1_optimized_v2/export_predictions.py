"""Export sealed native-unit H1 predictions for independent score replay.

This is a post-training reader: it never invokes backward/optimizer code and
never changes an attempt's checkpoints, selection freeze, or progress file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .cache import ROOT as CACHE_ROOT
from .cache import build_or_load, validate_authority
from .c2_reference import _operator_code_authority
from .model import make_matched_pair
from .paired_train import W
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank


CHUNK = 16


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _windows(array: np.ndarray, ends: np.ndarray) -> np.ndarray:
    out = np.zeros((len(ends), W, array.shape[1]), dtype=np.float32)
    for index, end in enumerate(ends):
        take = min(int(end) + 1, W)
        out[index, -take:] = array[int(end) + 1 - take : int(end) + 1]
    return out


def _native_predictions(model: torch.nn.Module, cache: dict, *, device: torch.device, mode: str) -> dict[str, np.ndarray]:
    prediction: list[np.ndarray] = []
    target: list[np.ndarray] = []
    session_id: list[np.ndarray] = []
    bin_timestep: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for session, row in cache["minival"].items():
            ends = row["query_starts"] + W - 1 if mode == "selection" else np.flatnonzero(row["eval_mask"])
            bank = H1Bank(
                row["bank"]["E0"].to(device), row["bank"]["T"].to(device), row["bank"]["unit_mask"].to(device)
            )
            pieces = []
            for offset in range(0, len(ends), CHUNK):
                x = torch.as_tensor(_windows(row["neural"], ends[offset : offset + CHUNK]), device=device)
                pieces.append((model.forward_last(x, bank) / 20.0).cpu().numpy().astype(np.float32, copy=False))
            prediction.append(np.concatenate(pieces))
            target.append(np.asarray(row["velocity"][ends], dtype=np.float32))
            session_id.append(np.full(len(ends), session, dtype="U32"))
            bin_timestep.append(np.asarray(ends, dtype=np.int64))
    return {
        "prediction_native_velocity": np.concatenate(prediction),
        "target_native_velocity": np.concatenate(target),
        "session_id": np.concatenate(session_id),
        "bin_timestep": np.concatenate(bin_timestep),
    }


def _assert_exact_source_ids(cache: dict, arrays: dict[str, np.ndarray], mode: str) -> None:
    """Prove exported identifiers are the cache's exact scored endpoints."""
    expected_session, expected_timestep = [], []
    for session, row in cache["minival"].items():
        ends = row["query_starts"] + W - 1 if mode == "selection" else np.flatnonzero(row["eval_mask"])
        expected_session.append(np.full(len(ends), session, dtype="U32"))
        expected_timestep.append(np.asarray(ends, dtype=np.int64))
    actual_session, actual_timestep = arrays["session_id"], arrays["bin_timestep"]
    if not np.array_equal(actual_session, np.concatenate(expected_session)) or not np.array_equal(actual_timestep, np.concatenate(expected_timestep)):
        raise RuntimeError(f"{mode} exported session/bin identifiers diverged from source cache")
    pairs = np.char.add(np.char.add(actual_session, ":"), actual_timestep.astype(str))
    if len(np.unique(pairs)) != len(pairs):
        raise RuntimeError(f"{mode} exported duplicate session/bin identifiers")
    if len(np.unique(actual_session)) != 13:
        raise RuntimeError(f"{mode} export requires exactly 13 sessions, got {len(np.unique(actual_session))}")


def _r2_float64(prediction: np.ndarray, target: np.ndarray) -> float:
    p, y = prediction.astype(np.float64), target.astype(np.float64)
    return float(1.0 - np.square(p - y).sum() / np.square(y - y.mean(axis=0, keepdims=True)).sum())


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prediction export: {path}")
    np.savez_compressed(path, **arrays)
    return _sha(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = CACHE_ROOT / args.attempt
    freeze_path = root / "selection_freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError(f"final selection freeze is required: {freeze_path}")
    freeze = json.loads(freeze_path.read_text())
    cache = build_or_load()
    recorded = json.loads((CACHE_ROOT / "source_cache_authority.json").read_text())
    validate_authority(cache, recorded)
    device = torch.device(args.device)
    output = root / "independent_score_export"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite export root: {output}")
    output.mkdir()
    manifest: dict = {
        "schema": "h1_optimized_v2_independent_native_score_export_v1",
        "attempt": args.attempt,
        "device": str(device),
        "input_authority": recorded,
        "exporter_code_sha256": _sha(Path(__file__)),
        "operator_code_sha256": _operator_code_authority(),
        "surfaces": {
            "selection": {"definition": "frozen minival W700 stride-4 endpoints", "expected_bins": 2908},
            "complete": {"definition": "all held-in-minival eval_mask bins", "expected_bins": 20325},
        },
        "arms": {},
    }
    for arm in ("full", "t"):
        model = make_matched_pair(activity_scale=32.0)[0 if arm == "full" else 1].to(device)
        choices = {
            "selected": Path(freeze["selected"][arm]["checkpoint"]),
            "epoch12": root / f"{arm}_epoch_012.pt",
        }
        arm_record: dict = {}
        for label, checkpoint_path in choices.items():
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"], strict=True)
            ema = DecoderEMA(model, decay=float(payload["ema"]["decay"]))
            ema.load_checkpoint_state(payload["ema"])
            state_path = output / f"{arm}_{label}_ema_metadata.pt"
            torch.save({"schema": "decoder_ema_metadata_v1", "arm": arm, "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha(checkpoint_path), "ema": ema.checkpoint_state()}, state_path)
            plain_state_path = output / f"{arm}_{label}_plain_ema_model_state.pt"
            exported: dict = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha(checkpoint_path), "ema_metadata": str(state_path), "ema_metadata_sha256": _sha(state_path), "plain_ema_model_state": str(plain_state_path), "surfaces": {}}
            def run(view: torch.nn.Module) -> None:
                # This is intentionally a plain model state_dict—not EMA shadow
                # metadata—so an independent latency bench can strict-load it.
                torch.save({k: v.detach().cpu().clone() for k, v in view.state_dict().items()}, plain_state_path)
                exported["plain_ema_model_state_sha256"] = _sha(plain_state_path)
                for surface in ("selection", "complete"):
                    arrays = _native_predictions(view, cache, device=device, mode=surface)
                    _assert_exact_source_ids(cache, arrays, surface)
                    if len(arrays["prediction_native_velocity"]) != manifest["surfaces"][surface]["expected_bins"]:
                        raise RuntimeError(f"{arm}/{label}/{surface} bin-count drift")
                    npz = output / f"{arm}_{label}_{surface}_native.npz"
                    exported["surfaces"][surface] = {
                        "npz": str(npz), "sha256": _save_npz(npz, arrays),
                        "n_bins": int(len(arrays["bin_timestep"])),
                        "session_count": int(len(np.unique(arrays["session_id"]))),
                        "r2_concat_float64": _r2_float64(arrays["prediction_native_velocity"], arrays["target_native_velocity"]),
                    }
            ema.score_with_ema(model, run)
            arm_record[label] = exported
        manifest["arms"][arm] = arm_record
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "sha256": _sha(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
