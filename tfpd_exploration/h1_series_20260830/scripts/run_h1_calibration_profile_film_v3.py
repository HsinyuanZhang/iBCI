#!/usr/bin/env python3
"""Run the H1 EP-FILM all-source V3 training (one arm + EP-ZERO anchor) on GPU0."""
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


def _git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _preflight(legacy_root: Path) -> dict[str, object]:
    from h1_calibration_profile_film_v1.plan import EXPECTED_GPU0_UUID
    from h1_calibration_profile_film_v3.plan import (
        ALL_SOURCE_CKPT_SHA256,
        ALL_SOURCE_IMPORT_RECEIPT_SHA256,
        ALL_SOURCE_ROOT_RELATIVE,
        IMPLEMENTATION_RELATIVES,
        LEGACY_CLOSURE_RELATIVES,
        LEGACY_HEAD,
        LEGACY_PLAN_RELATIVE,
        LEGACY_PLAN_SHA256,
        RESULT_ROOT_RELATIVE,
        WORKORDER_RELATIVE,
    )

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
        ["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    _require(compute_apps == "", f"GPU0 already has foreign compute processes: {compute_apps!r}")

    legacy = Path(legacy_root).resolve()
    _require(legacy.is_dir(), f"legacy checkout missing: {legacy}")
    _require(_git_head(legacy) == LEGACY_HEAD, "legacy checkout HEAD drift")
    _require(_sha256(legacy / LEGACY_PLAN_RELATIVE) == LEGACY_PLAN_SHA256, "legacy sealed plan.json drift")
    legacy_closure = {
        relative: _sha256(legacy / relative) for relative in LEGACY_CLOSURE_RELATIVES
    }
    all_source_root = REPO_ROOT / ALL_SOURCE_ROOT_RELATIVE
    _require(_sha256(all_source_root / "epoch_049.ckpt") == ALL_SOURCE_CKPT_SHA256,
             "all-source C1 checkpoint SHA drift")
    _require(_sha256(all_source_root / "import_receipt.json") == ALL_SOURCE_IMPORT_RECEIPT_SHA256,
             "all-source import receipt SHA drift")

    workorder = REPO_ROOT / WORKORDER_RELATIVE
    _require(workorder.is_file(), "V3 workorder missing")
    result_root = REPO_ROOT / RESULT_ROOT_RELATIVE
    _require(result_root.parent.is_dir() and not result_root.exists(),
             "V3 canonical result root unavailable or already exists")
    timer_state = subprocess.check_output(
        ["systemctl", "--user", "show", TIMER, "--property=ActiveState", "--value"], text=True,
    ).strip()

    import numpy
    import scipy
    import torch

    try:
        import falcon_challenge

        falcon_version = getattr(falcon_challenge, "__version__", "unknown")
        falcon_path = str(Path(falcon_challenge.__file__).resolve())
    except Exception:  # pragma: no cover - preflight must not crash on metadata
        falcon_version, falcon_path = "unknown", "unknown"
    from h1_calibration_profile_film_v3.evaluate import verify_v2_root

    v2_authority = verify_v2_root(REPO_ROOT)
    v2_authority.pop("source_rows_by_lodo_date", None)
    implementation = {relative: _sha256(REPO_ROOT / relative) for relative in IMPLEMENTATION_RELATIVES}
    return {
        "gpu0_uuid": gpu_uuid,
        "gpu0_free_mib_at_preflight": int(free_mib),
        "gpu0_utilization_at_preflight": int(utilization),
        "gpu0_foreign_compute_processes_at_preflight": compute_apps,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cpu_thread_env": {variable: os.environ[variable] for variable in THREAD_VARIABLES},
        "legacy_root": str(legacy),
        "legacy_head": LEGACY_HEAD,
        "legacy_sealed_plan_sha256": LEGACY_PLAN_SHA256,
        "legacy_closure_sha256": legacy_closure,
        "all_source_checkpoint_sha256": ALL_SOURCE_CKPT_SHA256,
        "all_source_import_receipt_sha256": ALL_SOURCE_IMPORT_RECEIPT_SHA256,
        "v2_authority": v2_authority,
        "workorder_sha256": _sha256(workorder),
        "implementation_sha256": implementation,
        "scheduled_lp_r3_timer_state": timer_state,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "numpy": numpy.__version__,
            "numpy_path": str(Path(numpy.__file__).resolve()),
            "scipy": scipy.__version__,
            "falcon_challenge": falcon_version,
            "falcon_challenge_path": falcon_path,
        },
    }


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
            "schema": f"{SCHEMA}_dry",
            "status": "INERT_USE_EXECUTE",
            "result_root": str(result_root),
            "legacy_root": str(legacy_root),
        }, sort_keys=True))
        return 0

    authority = _preflight(legacy_root)
    result_root.mkdir(parents=False, exist_ok=False)
    attempt_sha = _publish(result_root / "attempt.json", {
        "schema": f"{SCHEMA}_attempt",
        "status": "ATTEMPT_GPU0_ALL_SOURCE_EP_FILM_ONLY_V3",
        **authority,
        "formal_heldout_opened": False,
        "evalai_opened": False,
        "gpu1_touched": False,
    })
    try:
        from h1_calibration_profile_film_v3.evaluate import run

        summary = run(REPO_ROOT, legacy_root, device="cuda:0", receipt_root=result_root)
        terminal = {
            "schema": f"{SCHEMA}_terminal",
            "status": summary["status"],
            "attempt_sha256": attempt_sha,
            "training_receipt_sha256": summary["receipt_sha256"],
            "checkpoints": summary["checkpoints"],
            "training_summary": summary["training_summary"],
            "ep_zero_anchor": summary["ep_zero_anchor"],
            "plan_authority_summary": summary["plan_authority_summary"],
            "wall_time_seconds": summary["wall_time_seconds"],
            "started_at_utc": summary["started_at_utc"],
            "finished_at_utc": summary["finished_at_utc"],
            **authority,
            "formal_heldout_opened": False,
            "evalai_opened": False,
            "gpu1_touched": False,
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "target_model_updates": 0,
        }
        terminal_sha = _publish(result_root / "terminal.json", terminal)
        print(json.dumps({
            "status": terminal["status"],
            "terminal_sha256": terminal_sha,
            "checkpoint": summary["checkpoints"]["EP-FILM"],
            "ep_zero_anchor": summary["ep_zero_anchor"],
        }, sort_keys=True))
        return 0
    except BaseException as error:
        _publish(result_root / "failure.json", {
            "schema": f"{SCHEMA}_failure",
            "status": "FAIL_NO_RETRY",
            "attempt_sha256": attempt_sha,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "published_files": sorted(path.name for path in result_root.iterdir()),
            "formal_heldout_opened": False,
            "evalai_opened": False,
            "gpu1_touched": False,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
