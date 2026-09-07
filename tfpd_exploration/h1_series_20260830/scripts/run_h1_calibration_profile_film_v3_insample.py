#!/usr/bin/env python3
"""Run the V3 all-source EP-FILM local in-sample evaluation (EP-ZERO control) on GPU0."""
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
TIMER = "spint-lpr3-evalai-20260904.timer"
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


def _preflight(legacy_root: Path, result_root: Path) -> dict[str, object]:
    from h1_calibration_profile_film_v1.plan import EXPECTED_GPU0_UUID
    from h1_calibration_profile_film_v3.plan import LEGACY_HEAD, LEGACY_PLAN_RELATIVE, LEGACY_PLAN_SHA256

    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "CUDA_VISIBLE_DEVICES must be exactly 0")
    for variable in THREAD_VARIABLES:
        _require(os.environ.get(variable) == "1", f"CPU thread discipline requires {variable}=1")
    gpu_line = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    gpu_uuid, free_mib, utilization = (part.strip() for part in gpu_line.split(","))
    _require(gpu_uuid == EXPECTED_GPU0_UUID, "physical GPU0 UUID drift")
    _require(int(free_mib) >= 4096, "GPU0 has less than 4 GiB free")
    _require(int(utilization) <= 10, "GPU0 is not idle enough for a fresh launch")
    compute_apps = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"], text=True,
    ).strip()
    _require(compute_apps == "", f"GPU0 already has foreign compute processes: {compute_apps!r}")
    legacy = Path(legacy_root).resolve()
    _require(legacy.is_dir() and _git_head(legacy) == LEGACY_HEAD, "legacy checkout HEAD drift")
    _require(_sha256(legacy / LEGACY_PLAN_RELATIVE) == LEGACY_PLAN_SHA256, "legacy sealed plan.json drift")

    _require(result_root.is_dir(), "V3 result root missing (training must run first)")
    for name in ("attempt.json", "training_all_source.json", "terminal.json", "checkpoint_all_source_ep-film.pt"):
        body = result_root / name
        side = result_root / f"{name}.sha256"
        _require(body.is_file() and side.is_file(), f"sealed V3 receipt missing: {name}")
        _require(side.read_text(encoding="ascii") == f"{_sha256(body)}  {name}\n", f"V3 sidecar drift: {name}")
    _require(not (result_root / "local_insample_eval.json").exists(),
             "in-sample evaluation receipt already exists; no retry")
    timer_state = subprocess.check_output(
        ["systemctl", "--user", "show", TIMER, "--property=ActiveState", "--value"], text=True,
    ).strip()
    import numpy
    import torch

    return {
        "gpu0_uuid": gpu_uuid,
        "gpu0_free_mib_at_preflight": int(free_mib),
        "gpu0_utilization_at_preflight": int(utilization),
        "gpu0_foreign_compute_processes_at_preflight": compute_apps,
        "cpu_thread_env": {variable: os.environ[variable] for variable in THREAD_VARIABLES},
        "legacy_head": LEGACY_HEAD,
        "scheduled_lp_r3_timer_state": timer_state,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "numpy": numpy.__version__,
        },
    }


def _git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--legacy-root", type=Path, default=LEGACY_DEFAULT)
    args = parser.parse_args()
    legacy_root = args.legacy_root.resolve()
    _install_paths(legacy_root)

    from h1_calibration_profile_film_v3.plan import RESULT_ROOT_RELATIVE, SCHEMA

    result_root = REPO_ROOT / RESULT_ROOT_RELATIVE
    if not args.execute:
        print(json.dumps({
            "schema": f"{SCHEMA}_insample_dry",
            "status": "INERT_USE_EXECUTE",
            "result_root": str(result_root),
            "legacy_root": str(legacy_root),
        }, sort_keys=True))
        return 0

    authority = _preflight(legacy_root, result_root)
    try:
        from h1_calibration_profile_film_v3.insample import run_insample

        summary = run_insample(REPO_ROOT, legacy_root, device="cuda:0", preflight=authority)
        print(json.dumps(summary, sort_keys=True))
        return 0
    except BaseException as error:
        _publish(result_root / "local_insample_eval_failure.json", {
            "schema": f"{SCHEMA}_local_insample_eval_failure",
            "status": "FAIL_NO_RETRY",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "formal_heldout_opened": False,
            "evalai_opened": False,
            "gpu1_touched": False,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
