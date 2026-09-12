#!/usr/bin/env python3
"""Host and container full held-out streaming replay for fair_v2 linear packs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import hashlib

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def replay_session(decoder, item: dict, basename: str, *, context: int) -> tuple[np.ndarray, int, int]:
    raw = np.asarray(item["X"], dtype=np.float32)
    starts = np.asarray(item["starts"], dtype=np.int64)
    pad = int(item["pad"])
    endpoints = starts + int(context) - 1
    decoder.reset([Path(basename)])
    outputs = []
    for bin_index in range(pad, len(raw)):
        outputs.append(decoder.predict(raw[bin_index : bin_index + 1]))
    stacked = np.stack(outputs, axis=0)
    prediction = stacked[endpoints - pad, 0]
    return np.ascontiguousarray(prediction, dtype=np.float32), int(len(raw) - pad), len(outputs)


def host_audit(payload_dir: Path) -> dict:
    PACK = Path(__file__).resolve().parent
    if str(PACK.parent.parent) not in sys.path:
        sys.path.insert(0, str(PACK.parent.parent))
    if str(PACK) not in sys.path:
        sys.path.insert(0, str(PACK))
    from falcon_challenge.config import FalconConfig, FalconTask
    from common import RESULTS
    from falcon_decoder import FairV2LinearFalconDecoder
    from fair_v2 import data

    manifest = json.loads((payload_dir / "manifest.json").read_text())
    task = str(manifest["task"])
    method = str(manifest["method"])
    config = FalconConfig(task=getattr(FalconTask, task))
    decoder = FairV2LinearFalconDecoder(
        config,
        str(payload_dir / "payload.npz"),
        manifest_path=str(payload_dir / "manifest.json"),
        batch_size=1,
    )
    loaded = data.load_task(task, include_evaluation=True)
    heldout = loaded["evaluation"]
    context = int(loaded["metadata"]["context"])
    started = time.monotonic()
    rows = {}
    total_bins = total_predictions = total_scored = 0
    by_source = {value["source_session"]: (tag, value) for tag, value in manifest["sessions"].items()}
    for session, item in heldout.items():
        tag, record = by_source[session]
        actual, bins, appended = replay_session(decoder, item, record["raw_basename"], context=context)
        expected = np.load(RESULTS / f"{task}_v2" / f"pred_{method}_{session}.npy")
        if actual.shape != expected.shape:
            raise RuntimeError(f"{session}: shape {actual.shape} != {expected.shape}")
        error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
        rows[session] = {
            "tag": tag,
            "raw_bins_streamed": bins,
            "raw_predictions_appended": appended,
            "scored_rows": int(len(actual)),
            "max_abs_error": error,
            "tolerance": 0.0,
            "pass": error == 0.0,
        }
        total_bins += bins
        total_predictions += appended
        total_scored += len(actual)
    failed = {k: v for k, v in rows.items() if not v["pass"]}
    if failed:
        raise RuntimeError("host streaming replay mismatch: " + json.dumps(failed, sort_keys=True))
    result = {
        "schema": "fair_v2_linear_streaming_audit_v1",
        "task": task,
        "method": method,
        "payload_sha256": sha256_file(payload_dir / "payload.npz"),
        "manifest_sha256": sha256_file(payload_dir / "manifest.json"),
        "stream_contract": {
            "reset_per_session": 1,
            "reset_per_trial": 0,
            "bin_ms": int(config.bin_size_ms),
            "batch_size": 1,
            "padding": "reset supplies literal zero filter and history",
        },
        "total_raw_bins_streamed": total_bins,
        "total_scored_rows": total_scored,
        "total_raw_predictions_appended": total_predictions,
        "max_abs_error": max(v["max_abs_error"] for v in rows.values()),
        "tolerance": 0.0,
        "elapsed_seconds": time.monotonic() - started,
        "sessions": rows,
    }
    (payload_dir / "streaming_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _inside(input_path: Path, metadata_path: Path, output_path: Path) -> None:
    sys.path.insert(0, "/workspace")
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_decoder import FairV2LinearFalconDecoder

    meta = json.loads(metadata_path.read_text())
    config = FalconConfig(task=getattr(FalconTask, meta["task"]))
    decoder = FairV2LinearFalconDecoder(
        config, "/data/payload.npz", manifest_path="/data/manifest.json", batch_size=1
    )
    started = time.monotonic()
    rows = {}
    raw_total = appended_total = scored_total = 0
    with np.load(input_path, allow_pickle=False) as arrays:
        for row in meta["sessions"]:
            raw = np.asarray(arrays[row["raw_key"]], dtype=np.float32)
            endpoints = np.asarray(arrays[row["endpoints_key"]], dtype=np.int64)
            expected = np.asarray(arrays[row["expected_key"]], dtype=np.float32)
            decoder.reset([Path(row["raw_basename"])])
            returned = [decoder.predict(raw[i : i + 1]) for i in range(len(raw))]
            actual = np.stack(returned, axis=0)[endpoints, 0]
            error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
            rows[row["session"]] = {
                "tag": row["tag"],
                "raw_bins_streamed": int(len(raw)),
                "raw_predictions_appended": int(len(returned)),
                "scored_rows": int(len(actual)),
                "max_abs_error": error,
                "tolerance": 0.0,
                "pass": error == 0.0,
            }
            raw_total += len(raw)
            appended_total += len(returned)
            scored_total += len(actual)
    failed = {k: v for k, v in rows.items() if not v["pass"]}
    if failed:
        raise RuntimeError("container streaming parity mismatch: " + json.dumps(failed, sort_keys=True))
    output_path.write_text(
        json.dumps(
            {
                "numpy_version": np.__version__,
                "runtime_decoder_sha256": sha256_file(Path("/workspace/falcon_decoder.py")),
                "elapsed_seconds": time.monotonic() - started,
                "total_raw_bins_streamed": raw_total,
                "total_raw_predictions_appended": appended_total,
                "total_scored_rows": scored_total,
                "max_abs_error": max(x["max_abs_error"] for x in rows.values()),
                "sessions": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _host_prepare(payload_dir: Path, temp: Path) -> tuple[Path, Path, dict]:
    PACK = Path(__file__).resolve().parent
    if str(PACK.parent.parent) not in sys.path:
        sys.path.insert(0, str(PACK.parent.parent))
    if str(PACK) not in sys.path:
        sys.path.insert(0, str(PACK))
    from common import RESULTS
    from fair_v2 import data

    manifest = json.loads((payload_dir / "manifest.json").read_text())
    task, method = manifest["task"], manifest["method"]
    loaded = data.load_task(task, include_evaluation=True)
    arrays: dict[str, np.ndarray] = {}
    sessions = []
    by_source = {value["source_session"]: (tag, value) for tag, value in manifest["sessions"].items()}
    for index, (session, item) in enumerate(loaded["evaluation"].items()):
        tag, record = by_source[session]
        raw = np.asarray(item["X"], dtype=np.float32)
        pad = int(item["pad"])
        endpoint = np.asarray(item["starts"], dtype=np.int64) + int(loaded["metadata"]["context"]) - 1 - pad
        stream = np.ascontiguousarray(raw[pad:])
        expected = np.load(RESULTS / f"{task}_v2" / f"pred_{method}_{session}.npy")
        prefix = f"s{index}"
        arrays[prefix + "_raw"] = stream
        arrays[prefix + "_endpoints"] = endpoint
        arrays[prefix + "_expected"] = np.asarray(expected, dtype=np.float32)
        sessions.append(
            {
                "session": session,
                "tag": tag,
                "raw_basename": record["raw_basename"],
                "raw_key": prefix + "_raw",
                "endpoints_key": prefix + "_endpoints",
                "expected_key": prefix + "_expected",
            }
        )
    inputs = temp / "inputs"
    inputs.mkdir()
    input_path, metadata_path = inputs / "query_streams.npz", inputs / "metadata.json"
    np.savez_compressed(input_path, **arrays)
    metadata_path.write_text(json.dumps({"task": task, "method": method, "sessions": sessions}, indent=2, sort_keys=True) + "\n")
    return input_path, metadata_path, manifest


def container_audit(payload_dir: Path) -> dict:
    PACK = Path(__file__).resolve().parent
    if str(PACK) not in sys.path:
        sys.path.insert(0, str(PACK))
    with tempfile.TemporaryDirectory(prefix="fair-v2-linear-container-") as tmp:
        temp = Path(tmp)
        input_path, metadata_path, manifest = _host_prepare(payload_dir, temp)
        from common import image_tag

        image = image_tag(manifest["task"], manifest["method"])
        parsed = json.loads(
            subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{json .}}"], text=True)
        )
        image_info = {
            "image": image,
            "image_id": parsed["Id"],
            "labels": parsed.get("Config", {}).get("Labels", {}),
        }
        output_dir = temp / "output"
        output_dir.mkdir()
        output_path = output_dir / "container_result.json"
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "-e",
            "CUDA_VISIBLE_DEVICES=",
            "-v",
            f"{input_path}:/audit/query_streams.npz:ro",
            "-v",
            f"{metadata_path}:/audit/metadata.json:ro",
            "-v",
            f"{Path(__file__).resolve()}:/audit/replay_linear.py:ro",
            "-v",
            f"{output_dir}:/result:rw",
            image,
            "python",
            "/audit/replay_linear.py",
            "--inside",
            "/audit/query_streams.npz",
            "/audit/metadata.json",
            "/result/container_result.json",
        ]
        subprocess.run(command, check=True, text=True)
        inside = json.loads(output_path.read_text())
    result = {
        "schema": "fair_v2_linear_container_streaming_audit_v1",
        "task": manifest["task"],
        "method": manifest["method"],
        "payload_sha256": sha256_file(payload_dir / "payload.npz"),
        "manifest_sha256": sha256_file(payload_dir / "manifest.json"),
        "image": image_info,
        "container_numpy_version": inside["numpy_version"],
        "runtime_decoder_sha256": inside["runtime_decoder_sha256"],
        "tolerance": 0.0,
        **{
            key: inside[key]
            for key in (
                "elapsed_seconds",
                "total_raw_bins_streamed",
                "total_raw_predictions_appended",
                "total_scored_rows",
                "max_abs_error",
                "sessions",
            )
        },
    }
    (payload_dir / "container_replay_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload_dir", nargs="?", type=Path)
    parser.add_argument("--inside", nargs=3, metavar=("INPUT_NPZ", "METADATA_JSON", "OUTPUT_JSON"))
    parser.add_argument("--container", action="store_true")
    args = parser.parse_args()
    if args.inside:
        _inside(*(Path(x) for x in args.inside))
        return
    if args.payload_dir is None:
        parser.error("payload_dir is required outside a container")
    if args.container:
        print(json.dumps(container_audit(args.payload_dir), indent=2, sort_keys=True))
        return
    print(json.dumps(host_audit(args.payload_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
