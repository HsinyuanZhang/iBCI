#!/usr/bin/env python3
"""Parity-audit each packaged CPU image against saved held-out predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
TOLERANCE = 2e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _inside(input_path: Path, metadata_path: Path, output_path: Path) -> None:
    """Run inside the final image, importing only image-resident decoder code."""
    sys.path.insert(0, "/workspace")
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_decoder import ExternalGFFalconDecoder

    meta = json.loads(metadata_path.read_text())
    config = FalconConfig(task=getattr(FalconTask, meta["task"]))
    decoder = ExternalGFFalconDecoder(config, "/data/payload.npz", manifest_path="/data/manifest.json", batch_size=1)
    started = time.monotonic()
    rows = {}
    raw_total = appended_total = scored_total = 0
    with np.load(input_path, allow_pickle=False) as arrays:
        for row in meta["sessions"]:
            raw = np.asarray(arrays[row["raw_key"]], dtype=np.float32)
            endpoints = np.asarray(arrays[row["endpoints_key"]], dtype=np.int64)
            expected = np.asarray(arrays[row["expected_key"]], dtype=np.float32)
            decoder.reset([Path(row["raw_basename"])])
            # Match FalconEvaluator: retain every returned prediction, then
            # stack the complete stream.  This catches returned-view aliasing.
            returned = []
            for i in range(len(raw)):
                returned.append(decoder.predict(raw[i:i + 1]))
            stacked = np.stack(returned, axis=0)
            actual = stacked[endpoints, 0]
            if actual.shape != expected.shape:
                raise RuntimeError(f"{row['session']}: shape mismatch {actual.shape}/{expected.shape}")
            error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
            rows[row["session"]] = {"tag": row["tag"], "raw_bins_streamed": int(len(raw)),
                                    "raw_predictions_appended": int(len(returned)), "scored_rows": int(len(actual)),
                                    "max_abs_error": error, "tolerance": TOLERANCE, "pass": error <= TOLERANCE}
            raw_total += len(raw); appended_total += len(returned); scored_total += len(actual)
    failed = {k: v for k, v in rows.items() if not v["pass"]}
    if failed:
        raise RuntimeError("container streaming parity mismatch: " + json.dumps(failed, sort_keys=True))
    output_path.write_text(json.dumps({"numpy_version": np.__version__, "runtime_decoder_sha256": sha256(Path("/workspace/falcon_decoder.py")),
        "elapsed_seconds": time.monotonic() - started, "total_raw_bins_streamed": raw_total,
        "total_raw_predictions_appended": appended_total, "total_scored_rows": scored_total,
        "max_abs_error": max(x["max_abs_error"] for x in rows.values()), "sessions": rows}, indent=2, sort_keys=True) + "\n")


def _host_prepare(payload_dir: Path, temp: Path) -> tuple[Path, Path, dict]:
    # Reuse the exporter's receipt-backed M2 loader.  It opens query X only for
    # held-out M2 and never attempts to reconstruct deleted target activity.
    sys.path.insert(0, str(Path(__file__).parent))
    from export_payloads import HERE as EXPORT_ROOT, load_export_data
    manifest = json.loads((payload_dir / "manifest.json").read_text())
    task, training_method = manifest["task"], manifest["training_method"]
    loaded = load_export_data(task)
    arrays: dict[str, np.ndarray] = {}
    sessions = []
    by_source = {value["source_session"]: (tag, value) for tag, value in manifest["sessions"].items()}
    for index, (session, item) in enumerate(loaded["evaluation"].items()):
        tag, record = by_source[session]
        raw = np.asarray(item["X"], dtype=np.float32)
        pad = int(item["pad"])
        endpoint = np.asarray(item["starts"], dtype=np.int64) + int(loaded["metadata"]["context"]) - 1 - pad
        stream = np.ascontiguousarray(raw[pad:])
        if np.any(endpoint < 0) or np.any(endpoint >= len(stream)):
            raise ValueError(f"{session}: invalid endpoint after pad")
        expected = np.load(EXPORT_ROOT / "results" / f"{task}_full_v1" / f"pred_{training_method}_{session}.npy")
        if len(endpoint) != len(expected):
            raise ValueError(f"{session}: endpoint/prediction row mismatch")
        prefix = f"s{index}"
        arrays[prefix + "_raw"] = stream
        arrays[prefix + "_endpoints"] = endpoint
        arrays[prefix + "_expected"] = np.asarray(expected, dtype=np.float32)
        sessions.append({"session": session, "tag": tag, "raw_basename": record["raw_basename"],
                         "raw_key": prefix + "_raw", "endpoints_key": prefix + "_endpoints", "expected_key": prefix + "_expected"})
    inputs = temp / "inputs"
    inputs.mkdir()
    input_path, metadata_path = inputs / "query_streams.npz", inputs / "metadata.json"
    np.savez_compressed(input_path, **arrays)
    metadata = {"task": task, "training_method": training_method, "sessions": sessions}
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return input_path, metadata_path, manifest


def _image_info(image: str) -> dict:
    template = "{{json .}}"
    value = subprocess.check_output(["docker", "image", "inspect", image, "--format", template], text=True)
    parsed = json.loads(value)
    return {"image": image, "image_id": parsed["Id"], "repo_digests": parsed.get("RepoDigests", []),
            "labels": parsed.get("Config", {}).get("Labels", {})}


def audit(payload_dir: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="external-gf-container-replay-") as tmp:
        temp = Path(tmp)
        input_path, metadata_path, manifest = _host_prepare(payload_dir, temp)
        image = f"external-gf-{manifest['task']}-{manifest['method']}-v1:cpu"
        image_info = _image_info(image)
        host_script = Path(__file__).resolve()
        output_dir = temp / "output"
        output_dir.mkdir()
        output_path = output_dir / "container_result.json"
        command = ["docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
                   "-v", f"{input_path}:/audit/query_streams.npz:ro", "-v", f"{metadata_path}:/audit/metadata.json:ro",
                   "-v", f"{host_script}:/audit/replay_containers.py:ro", "-v", f"{output_dir}:/result:rw",
                   image, "python", "/audit/replay_containers.py", "--inside", "/audit/query_streams.npz", "/audit/metadata.json", "/result/container_result.json"]
        subprocess.run(command, check=True, text=True)
        inside = json.loads(output_path.read_text())
    result = {"schema": "external_gf_falcon_container_streaming_audit_v1", "task": manifest["task"],
              "method": manifest["method"], "training_method": manifest["training_method"], "payload_sha256": sha256(payload_dir / "payload.npz"),
              "manifest_sha256": sha256(payload_dir / "manifest.json"), "container_replay_script_sha256": sha256(Path(__file__)),
              "image": image_info, "container_numpy_version": inside["numpy_version"],
              "runtime_decoder_sha256": inside["runtime_decoder_sha256"], "stream_contract": {"retention": "append_then_stack", "reset_per_session": 1, "reset_per_trial": 0, "batch_size": 1},
              **{key: inside[key] for key in ("elapsed_seconds", "total_raw_bins_streamed", "total_raw_predictions_appended", "total_scored_rows", "max_abs_error", "sessions")},
              "tolerance": TOLERANCE}
    (payload_dir / "container_replay_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload_dir", nargs="?", type=Path)
    parser.add_argument("--inside", nargs=3, metavar=("INPUT_NPZ", "METADATA_JSON", "OUTPUT_JSON"))
    args = parser.parse_args()
    if args.inside:
        _inside(*(Path(x) for x in args.inside)); return
    if args.payload_dir is None:
        parser.error("payload_dir is required outside a container")
    print(json.dumps(audit(args.payload_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
