#!/usr/bin/env python3
"""Score the frozen Original SPINT M1 payload on the BT HO-calib pick face.

This diagnostic intentionally has no training, GPU, EvalAI, or hidden-test
path.  It uses the exact direct reader used by the recent M1 BT epoch picks:
the three visible held-out-calib NWBs, chronological M10 support, full local
query rows, W=100, and native behaviour scale.  Before any score is written,
the payload's stored M10 tensors must byte-match the direct reader.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BT1 = ROOT / "btransform_unified_v1"
SPINT = ROOT / "SPINT-main"
STREAMING = ROOT / "streaming_calibration_exp"
PAYLOAD = ROOT / "sua_exploration/evalai_m1_threeway/artifacts/spint_m1_epoch19.pkl"
DEFAULT_OUT = ROOT / "btransform_unified_v2/results/diagnostics_v1/m1_original_same_surface_v1"
SESSIONS = ("20121004", "20121017", "20121024")
EXPECTED_COUNTS = {"20121004": 1305, "20121017": 1295, "20121024": 1281}
WINDOW = 100
CALIB_TRIALS = 10


def _prepend(path: Path) -> None:
    text = str(path)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)


# The direct reader imports ``src``.  It must resolve to streaming_calibration_exp,
# which is the reader lineage used by m1_projadd_depth2_series.
for _path in (ROOT, SPINT, BT1 / "src", STREAMING):
    _prepend(_path)

import torch  # noqa: E402
from falcon_challenge.config import FalconConfig, FalconTask  # noqa: E402
from third_party.falcon_challenge.spint_decoder import SpintDecoder  # noqa: E402


class _CPUUnpickler(pickle.Unpickler):
    """Original package loader, retaining the old scorer's CPU map location."""

    def find_class(self, module: str, name: str):  # type: ignore[override]
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: torch.load(
                io.BytesIO(value), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_payload() -> dict[str, Any]:
    with PAYLOAD.open("rb") as handle:
        payload = _CPUUnpickler(handle).load()
    required = {
        "decoder", "task", "window_size", "behavior_scaling_factor",
        "calib_trial_features", "smooth_calibration",
    }
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"Original payload missing keys: {sorted(missing)}")
    if str(payload["task"]) != str(FalconTask.m1):
        raise RuntimeError(f"Payload task drift: {payload['task']!r}")
    if int(payload["window_size"]) != WINDOW:
        raise RuntimeError(f"Payload W={payload['window_size']} != {WINDOW}")
    if float(payload["behavior_scaling_factor"]) != 1.0:
        raise RuntimeError("This diagnostic is native-scale only; payload scaling is not 1")
    if bool(payload["smooth_calibration"]):
        raise RuntimeError("Payload smoothing differs from the BT same-surface reader")
    return payload


def _open_surface() -> dict[str, dict[str, Any]]:
    # Import only after STREAMING has won the ``src`` package resolution.
    scripts = str(BT1 / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import m1_projadd_depth2_series as pick  # noqa: PLC0415

    opened: dict[str, dict[str, Any]] = {}
    for session in SESSIONS:
        row = pick.open_heldout_calib_session(session)
        if int(row["n_windows"]) != EXPECTED_COUNTS[session]:
            raise RuntimeError(f"{session}: n_windows drift {row['n_windows']}")
        opened[session] = row
    return opened


def _payload_tag(config: FalconConfig, session: str) -> str:
    # FALCON hashes an NWB filename stem, not a bare session identifier.
    stem = f"sub-MonkeyL-held-out-calib_ses-{session}_behavior+ecephys"
    tag = config.hash_dataset(stem)
    if tag != session:
        raise RuntimeError(f"unexpected FALCON dataset tag {tag!r} for {session}")
    return tag


def _query_arrays(dataset: Any, session: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = [i for i, (name, _start) in enumerate(dataset.window_indices) if name == session]
    starts = np.asarray([dataset.window_indices[i][1] for i in ids], dtype=np.int64)
    if len(ids) != EXPECTED_COUNTS[session]:
        raise RuntimeError(f"{session}: query-ID count drift {len(ids)}")
    neural = dataset.neural_data[session]
    behavior = dataset.covariate_data[session]
    windows = np.stack([neural[start : start + WINDOW] for start in starts]).astype(np.float32)
    targets = behavior[starts + WINDOW - 1].astype(np.float32, copy=True)
    if windows.shape != (EXPECTED_COUNTS[session], WINDOW, 64):
        raise RuntimeError(f"{session}: window shape drift {windows.shape}")
    if targets.shape != (EXPECTED_COUNTS[session], 16):
        raise RuntimeError(f"{session}: target shape drift {targets.shape}")
    # Verify these arrays are exactly what FalconDataset exposes to the BT scorer.
    for idx in (0, len(ids) // 2, len(ids) - 1):
        x, y, _calib, observed_session = dataset[ids[idx]]
        if observed_session != session or not np.array_equal(x, windows[idx]) or not np.array_equal(y[-1], targets[idx]):
            raise RuntimeError(f"{session}: direct window parity failed at local index {idx}")
    return starts, windows, targets


def _r2(predictions: np.ndarray, targets: np.ndarray) -> float:
    denom = np.square(targets - targets.mean(axis=0, keepdims=True)).sum()
    if not np.isfinite(denom) or denom <= 0:
        raise RuntimeError("invalid R2 denominator")
    return float(1.0 - np.square(predictions - targets).sum() / denom)


def _direct_predict(model: torch.nn.Module, windows: np.ndarray, calib: np.ndarray, batch: int) -> np.ndarray:
    pieces: list[np.ndarray] = []
    support = torch.as_tensor(calib, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        for left in range(0, len(windows), batch):
            x = torch.as_tensor(windows[left : left + batch], dtype=torch.float32)
            support_batch = support.expand(len(x), -1, -1, -1)
            output = model(x, calib_trialized_neural_features=support_batch)
            pieces.append(output[:, -1, :].cpu().numpy().astype(np.float32, copy=True))
    return np.concatenate(pieces, axis=0)


def _streaming_sample_parity(
    config: FalconConfig, session: str, windows: np.ndarray, starts: np.ndarray,
    direct_samples: np.ndarray,
) -> dict[str, Any]:
    """Compare the old online scorer with direct W=100 inference at real query IDs."""
    legacy = SpintDecoder(config, str(PAYLOAD), batch_size=1)
    stem = Path(f"sub-MonkeyL-held-out-calib_ses-{session}_behavior+ecephys.nwb")
    legacy.reset([stem])
    # Run the actual online state machine over each selected full W=100 window.
    # Its output at the final bin must equal the direct model's exact same window.
    sample_indices = (0, len(starts) // 2, len(starts) - 1)
    errors: list[float] = []
    for sample_pos, index in enumerate(sample_indices):
        legacy.reset([stem])
        prediction = None
        for row in windows[index]:
            prediction = legacy.predict(row[None, :])[0]
        assert prediction is not None
        errors.append(float(np.max(np.abs(prediction.astype(np.float32) - direct_samples[sample_pos]))))
    maximum = max(errors)
    if maximum > 2e-6:
        raise RuntimeError(f"{session}: old-scorer/direct parity exceeds tolerance: {maximum}")
    return {
        "query_local_indices": list(sample_indices),
        "window_starts": [int(starts[i]) for i in sample_indices],
        "max_abs_errors": errors,
        "max_abs_error": maximum,
        "tolerance": 2e-6,
    }


def audit(out: Path, batch: int) -> dict[str, Any]:
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only diagnostic refused because CUDA is visible")
    torch.set_num_threads(2)
    payload = _load_payload()
    opened = _open_surface()
    config = FalconConfig(task=FalconTask.m1)
    model = payload["decoder"].cpu().eval()
    rows: dict[str, Any] = {}
    for session in SESSIONS:
        tag = _payload_tag(config, session)
        direct_calib = np.ascontiguousarray(opened[session]["calib10"], dtype=np.float32)
        payload_calib = np.ascontiguousarray(payload["calib_trial_features"][tag], dtype=np.float32)
        if payload_calib.shape != (CALIB_TRIALS, 1024, 64):
            raise RuntimeError(f"{session}: payload M10 shape drift {payload_calib.shape}")
        if not np.array_equal(payload_calib, direct_calib):
            raise RuntimeError(f"{session}: Original payload M10 differs from same-surface reader")
        starts, windows, targets = _query_arrays(opened[session]["dataset"], session)
        # Audit only needs real samples.  Full direct inference is deliberately
        # deferred to ``score`` so the preflight does not consume the same CPU
        # work twice.
        sample_indices = (0, len(starts) // 2, len(starts) - 1)
        direct_samples = _direct_predict(
            model, windows[list(sample_indices)], payload_calib, min(batch, 3)
        )
        direct = _direct_predict(model, windows[:3], payload_calib, min(batch, 3))
        parity = _streaming_sample_parity(
            config, session, windows, starts, direct_samples
        )
        rows[session] = {
            "dataset_tag": tag,
            "n_windows": int(len(starts)),
            "body_sha256": str(opened[session]["body_sha256"]),
            "m10_sha256_direct": _sha_array(direct_calib),
            "m10_sha256_payload": _sha_array(payload_calib),
            "m10_byte_equal": True,
            "unit_axis": "N=64; direct reader array equals payload M10 byte-for-byte",
            "query_window_shape": list(windows.shape),
            "query_target_shape": list(targets.shape),
            "query_window_sha256": _sha_array(windows),
            "query_target_sha256": _sha_array(targets),
            "query_start_sha256": _sha_array(starts),
            "native_behavior_scaling_factor": float(payload["behavior_scaling_factor"]),
            "three_window_direct_prediction_sha256": _sha_array(direct),
            "streaming_direct_parity": parity,
        }
    report = {
        "schema": "m1_original_same_surface_audit_v1",
        "utc": _utc(),
        "payload": str(PAYLOAD),
        "payload_sha256": _sha_file(PAYLOAD),
        "old_scorer": "SPINT-main/third_party/falcon_challenge/spint_decoder.py::SpintDecoder",
        "reader": "btransform_unified_v1/scripts/m1_projadd_depth2_series.py::open_heldout_calib_session",
        "surface": "visible held-out-calib trio; M10; query_start_trial=0; full local query; no test",
        "cpu": {"torch_num_threads": 2, "cuda_visible": False},
        "sessions": rows,
    }
    _json(out / "audit.json", report)
    return report


def score(out: Path, batch: int) -> dict[str, Any]:
    audit_report = audit(out, batch)
    payload = _load_payload()
    opened = _open_surface()
    model = payload["decoder"].cpu().eval()
    results: dict[str, Any] = {}
    for session in SESSIONS:
        partial = out / f"partial_{session}.npz"
        starts, windows, targets = _query_arrays(opened[session]["dataset"], session)
        if partial.exists():
            with np.load(partial, allow_pickle=False) as saved:
                saved_starts = saved["window_starts"]
                predictions = saved["predictions"]
                saved_targets = saved["targets"]
            if not np.array_equal(saved_starts, starts) or not np.array_equal(saved_targets, targets):
                raise RuntimeError(f"{session}: incompatible recovery partial")
        else:
            calib = np.ascontiguousarray(payload["calib_trial_features"][session], dtype=np.float32)
            predictions = _direct_predict(model, windows, calib, batch)
            np.savez_compressed(partial, window_starts=starts, predictions=predictions, targets=targets)
        results[session] = {
            "n_windows": int(len(starts)),
            "r2": _r2(predictions, targets),
            "window_starts_sha256": _sha_array(starts),
            "predictions_sha256": _sha_array(predictions),
            "targets_sha256": _sha_array(targets),
            "partial_rawpred": str(partial),
        }
        _json(out / "progress.json", {"utc": _utc(), "completed": results})
    equal_mean = float(np.mean([results[s]["r2"] for s in SESSIONS]))
    report = {
        "schema": "m1_original_same_surface_score_v1",
        "utc": _utc(),
        "audit_sha256": _sha_file(out / "audit.json"),
        "surface": audit_report["surface"],
        "metric": "per-session R2 = 1 - sum((prediction-target)^2)/sum((target-mean(target,axis=0))^2); equal mean across sessions",
        "sessions": results,
        "equal_session_mean_r2": equal_mean,
        "not_official_test": True,
        "no_training": True,
        "no_gpu": True,
    }
    _json(out / "score.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "score"), default="score")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if args.batch < 1 or args.batch > 16:
        raise SystemExit("--batch must be in [1,16] for the two-core CPU contract")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise SystemExit("Refused: set CUDA_VISIBLE_DEVICES='' or '-1' for this CPU-only diagnostic")
    args.out.mkdir(parents=True, exist_ok=True)
    value = audit(args.out, args.batch) if args.stage == "audit" else score(args.out, args.batch)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
