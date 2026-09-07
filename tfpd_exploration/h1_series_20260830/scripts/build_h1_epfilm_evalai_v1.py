#!/usr/bin/env python3
"""Build the H1 EP-FILM cached-identity deployment payload from the V3 receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_SRC = REPO_ROOT / "tfpd_exploration/h1_series_20260830/src"
LEGACY_DEFAULT = Path("/tmp/ibci-h1/deployment-v1")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _install_paths(legacy_root: Path) -> None:
    legacy_spint = (legacy_root / "SPINT-main").resolve()
    for value in (str(legacy_spint), str(LOCAL_SRC)):
        if value not in sys.path:
            sys.path.insert(0, value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(path: Path, value: dict) -> str:
    from src.h1_m4_cce_contract import write_immutable_json

    _, digest = write_immutable_json(path, value)
    side = path.with_name(path.name + ".sha256")
    with side.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {path.name}\n")
    os.chmod(side, 0o444)
    return digest


def _preflight(legacy_root: Path) -> dict[str, object]:
    from h1_calibration_profile_film_v3.plan import LEGACY_HEAD, LEGACY_PLAN_RELATIVE, LEGACY_PLAN_SHA256

    for variable in THREAD_VARIABLES:
        _require(os.environ.get(variable) == "1", f"CPU thread discipline requires {variable}=1")
    gpu: dict[str, object] = {"used": False}
    if os.environ.get("H1_EP_FILM_DEVICE", "cuda:0") == "cuda:0":
        _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "CUDA_VISIBLE_DEVICES must be exactly 0")
        gpu_line = subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-gpu=uuid,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True,
        ).strip()
        gpu_uuid, free_mib, utilization = (part.strip() for part in gpu_line.split(","))
        compute_apps = subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"], text=True,
        ).strip()
        _require(compute_apps == "", f"GPU0 has foreign compute processes: {compute_apps!r}")
        _require(int(free_mib) >= 4096 and int(utilization) <= 10, "GPU0 not idle for the light build pass")
        gpu = {"used": True, "uuid": gpu_uuid, "free_mib": int(free_mib), "utilization": int(utilization),
               "foreign_compute_processes": compute_apps}
    legacy = Path(legacy_root).resolve()
    _require(legacy.is_dir(), f"legacy checkout missing: {legacy}")
    _require(subprocess.check_output(["git", "-C", str(legacy), "rev-parse", "HEAD"], text=True).strip()
             == LEGACY_HEAD, "legacy checkout HEAD drift")
    _require(_sha256(legacy / LEGACY_PLAN_RELATIVE) == LEGACY_PLAN_SHA256, "legacy sealed plan.json drift")

    from h1_calibration_profile_film_v3.package import ARTIFACT_ROOT_RELATIVE, PAYLOAD_RELATIVE

    payload_path = REPO_ROOT / PAYLOAD_RELATIVE
    artifact_root = REPO_ROOT / ARTIFACT_ROOT_RELATIVE
    _require(not payload_path.exists(), "payload already built; one-shot discipline")
    _require(not (artifact_root / "calibration_authority.json").exists(), "calibration authority already built")

    import numpy
    import torch

    return {
        "device": os.environ.get("H1_EP_FILM_DEVICE", "cuda:0"),
        "gpu0": gpu,
        "cpu_thread_env": {variable: os.environ[variable] for variable in THREAD_VARIABLES},
        "legacy_head": LEGACY_HEAD,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": numpy.__version__,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--legacy-root", type=Path, default=LEGACY_DEFAULT)
    args = parser.parse_args()
    legacy_root = args.legacy_root.resolve()
    _install_paths(legacy_root)

    from h1_calibration_profile_film_v3.package import ARTIFACT_ROOT_RELATIVE, PAYLOAD_RELATIVE

    if not args.execute:
        print(json.dumps({
            "status": "INERT_USE_EXECUTE",
            "payload": str(REPO_ROOT / PAYLOAD_RELATIVE),
            "artifact_root": str(REPO_ROOT / ARTIFACT_ROOT_RELATIVE),
            "legacy_root": str(legacy_root),
        }, sort_keys=True))
        return 0

    authority = _preflight(legacy_root)
    artifact_root = REPO_ROOT / ARTIFACT_ROOT_RELATIVE
    artifact_root.mkdir(parents=True, exist_ok=True)
    attempt_sha = _publish(artifact_root / "build_attempt.json", {
        "schema": "h1_epfilm_evalai_v1_build_attempt",
        "status": "ATTEMPT_PAYLOAD_BUILD",
        **authority,
        "hidden_test_opened": False,
        "evalai_opened": False,
        "gpu1_touched": False,
    })
    try:
        from h1_calibration_profile_film_v3.package import build_payload

        summary = build_payload(REPO_ROOT, legacy_root, device=str(authority["device"]))
        summary["attempt_sha256"] = attempt_sha
        summary["preflight"] = authority
        print(json.dumps(summary, sort_keys=True))
        return 0
    except BaseException as error:
        _publish(artifact_root / "build_failure.json", {
            "schema": "h1_epfilm_evalai_v1_build_failure",
            "status": "FAIL_NO_RETRY",
            "attempt_sha256": attempt_sha,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "hidden_test_opened": False,
            "evalai_opened": False,
            "gpu1_touched": False,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
