#!/usr/bin/env python3
"""Finalize the H1 EP-FILM packaging receipt from the offline validation logs."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_calibration_profile_film_v3"
PAYLOAD_RELATIVE = "SPINT-main/local_data/h1_epfilm_evalai_v1/decoder.pt"
ARTIFACT_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/artifacts/h1_epfilm_evalai_v1"
PARITY_TIGHT = 1.0e-4
PARITY_LOOSE = 1.0e-3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(path: Path, value: dict) -> str:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "SPINT-main"))
    sys.path.insert(0, "/tmp/ibci-h1/deployment-v1/SPINT-main")
    from src.h1_m4_cce_contract import write_immutable_json

    _, digest = write_immutable_json(path, value)
    side = path.with_name(path.name + ".sha256")
    with side.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {path.name}\n")
    os.chmod(side, 0o444)
    return digest


def parse_minival_metrics(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Returning result from phase: minival: (\[.*\])", text)
    if not match:
        raise RuntimeError(f"minival metrics missing from {log_path}")
    payload = ast.literal_eval(match.group(1))
    return dict(payload[0]["minival_split_h1"])


def parse_smoke(log_path: Path) -> dict:
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{") and "PASS_H1_EP_FILM_CONTAINER_SMOKE" in line:
            body = json.loads(line)
            body.pop("prediction", None)
            return body
    raise RuntimeError(f"smoke result missing from {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-log", type=Path, required=True)
    parser.add_argument("--container-log", type=Path, required=True)
    parser.add_argument("--smoke-log", type=Path, required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    payload_path = REPO_ROOT / PAYLOAD_RELATIVE
    payload_sha = _sha256(payload_path)
    side = (ARTIFACT_ROOT / "calibration_authority.json.sha256").read_text(encoding="ascii").split("  ")[0]
    authority_sha = _sha256(ARTIFACT_ROOT / "calibration_authority.json")
    if side != authority_sha:
        raise RuntimeError("calibration authority sidecar drift")
    authority = json.loads((ARTIFACT_ROOT / "calibration_authority.json").read_text(encoding="utf-8"))
    if authority.get("payload_sha256") != payload_sha:
        raise RuntimeError("payload sha drift vs calibration authority")

    image_json = subprocess.check_output(
        ["docker", "inspect", args.image_tag], text=True)
    image = json.loads(image_json)[0]
    labels = image.get("Config", {}).get("Labels", {}) or {}
    if image.get("Id") != args.image_id:
        raise RuntimeError("image id drift")
    if labels.get("ibci.h1.package.sha256") != payload_sha:
        raise RuntimeError("image payload label drift")

    host = parse_minival_metrics(args.host_log)
    container = parse_minival_metrics(args.container_log)
    smoke = parse_smoke(args.smoke_log)
    parity = {
        key: {"host": host[key], "container": container[key],
              "abs_delta": abs(float(host[key]) - float(container[key]))}
        for key in sorted(set(host) & set(container))
    }
    r2_parity = max(row["abs_delta"] for key, row in parity.items() if "R2" in key)
    latency_delta = parity.get("Normalized Latency", {}).get("abs_delta")
    status = (
        "PASS_PARITY_TIGHT" if r2_parity <= PARITY_TIGHT
        else "PASS_PARITY_LOOSE_TORCH_VERSION_NOTE" if r2_parity <= PARITY_LOOSE
        else "FAIL_PARITY"
    )
    reference_rows = [
        row.get("deployment_reference") for row in authority["sessions"]
        if row.get("deployment_reference")
    ]
    builder_reference = {
        "per_session_readout_r2_mean": sum(row["readout_ep_film_r2"] for row in reference_rows) / len(reference_rows),
        "per_session_raw_ep_film_r2_mean": sum(row["raw_ep_film_r2"] for row in reference_rows) / len(reference_rows),
        "aggregation_note": (
            "builder reference averages 13 per-set R2; the evaluator averages 6 per-session "
            "R2 over concatenated sets, so the two means differ by aggregation only"
        ),
    }
    receipt = {
        "schema": "h1_calibration_profile_film_v3_packaging",
        "status": f"READY_{status}" if status.startswith("PASS") else status,
        "payload": {
            "relative": PAYLOAD_RELATIVE,
            "sha256": payload_sha,
            "film_state_sha256": labels.get("ibci.h1.film_state.sha256"),
            "checkpoint_sha256": labels.get("ibci.h1.checkpoint.sha256"),
            "calibration_authority_sha256": authority_sha,
            "readout_selection_sha256": authority.get("readout_selection_sha256"),
        },
        "image": {
            "tag": args.image_tag,
            "id": args.image_id,
            "size_bytes": image.get("Size"),
            "labels": labels,
            "offline_network": "none",
        },
        "offline_validation": {
            "container_smoke": smoke,
            "host_minival_metrics": host,
            "container_minival_metrics": container,
            "parity": parity,
            "r2_parity_max_abs_delta": r2_parity,
            "latency_parity_abs_delta": latency_delta,
            "parity_threshold_tight": PARITY_TIGHT,
            "parity_threshold_loose": PARITY_LOOSE,
            "builder_deployment_reference": builder_reference,
            "container_torch": "2.5.1.post303 (conda)",
            "host_torch": "2.3.1+cu121",
        },
        "method": {
            "name": "H1 EP-FILM cached identity (all-source, sensitivity)",
            "is_test_time_adaptive": False,
            "is_held_out_zero_shot": False,
            "is_pretrained": False,
            "official_calibration_trials": 3,
        },
        "push": {"authorized": True, "executed": False, "note": "push state lives in artifacts/h1_epfilm_evalai_v1/evalai_push_state.json"},
        "hidden_test_opened": False,
        "evalai_opened": False,
    }
    if not args.execute:
        print(json.dumps({"status": receipt["status"], "r2_parity_max_abs_delta": r2_parity,
                          "parity": parity}, indent=2, sort_keys=True))
        return 0
    if status == "FAIL_PARITY":
        raise RuntimeError(f"R2 parity exceeded the loose threshold: {r2_parity}")
    digest = _publish(RESULT_ROOT / "packaging_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "packaging_receipt_sha256": digest,
                      "r2_parity_max_abs_delta": r2_parity}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
