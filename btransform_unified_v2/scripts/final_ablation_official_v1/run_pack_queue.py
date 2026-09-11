#!/usr/bin/env python3
"""Fresh-only, serial CPU pack queue for the five formal fixed controls.

This controller invokes ``pack_fixed_controls.py`` after each profile's
required formal stages complete.  It never imports submission code or performs a
network, registry, or EvalAI action.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/final_ablation_official_v1"
FORMAL_QUEUE = OUT / "root_formal_queue.json"
QUEUE = OUT / "root_pack_queue.json"
INPUTS = OUT / "root_pack_inputs_v1.json"
LOCK = OUT / "root_pack_queue.lock"
STOP = OUT / "STOP_BEFORE_NEXT_PACK"
HERE = Path(__file__).resolve().parent
PACKER = HERE / "pack_fixed_controls.py"
PYTHON = Path("/home/xinyuan/miniconda3/envs/spint/bin/python")
PROFILES = ("m2_activity_only", "m2_none", "h1_activity_only", "h1_none", "m1_none")
DESTS = {
    "m2_activity_only": ROOT.parent / "tfpd_exploration/submissions/evalai_m2_rift_activity_only_r50_v1",
    "m2_none": ROOT.parent / "tfpd_exploration/submissions/evalai_m2_rift_none_r50_v1",
    "h1_activity_only": ROOT.parent / "tfpd_exploration/submissions/evalai_h1_rift_activity_only_r300_v1",
    "h1_none": ROOT.parent / "tfpd_exploration/submissions/evalai_h1_rift_none_r300_v1",
    "m1_none": ROOT.parent / "tfpd_exploration/submissions/evalai_m1_rift_none_r100_v1",
}
TEMPLATES = (
    ROOT.parent / "tfpd_exploration/submissions/evalai_m2_rift_r50_ablation_v1",
    ROOT.parent / "tfpd_exploration/submissions/evalai_h1_rift_r300_ablation_v1",
    ROOT.parent / "tfpd_exploration/submissions/evalai_m1_rift_none_r100_v1",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"object JSON required: {path}")
    return value


def write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def input_files() -> dict[str, str]:
    files = {Path(__file__).resolve(), PACKER}
    for template in TEMPLATES:
        require(template.is_dir(), f"template missing: {template}")
        files.update(template.glob("*.py"))
        files.update(p for p in (template / "Dockerfile", template / ".dockerignore") if p.is_file())
    return {str(path): sha(path) for path in sorted(files)}


def formal_rows(formal: dict[str, Any], profile: str, *, completed: bool) -> list[dict[str, Any]]:
    rows = [x for x in formal.get("jobs", []) if isinstance(x, dict) and x.get("name") == profile]
    expected = ["train", "score"] if profile.startswith(("m1_", "m2_")) else ["train_and_score"]
    require([x.get("stage") for x in rows] == expected, f"{profile}: formal stage inventory drift")
    if completed:
        require(all(x.get("status") == "COMPLETED" and x.get("returncode") == 0 for x in rows), f"{profile}: formal stages incomplete")
    return rows


def prepare() -> None:
    require(not QUEUE.exists() and not INPUTS.exists(), "fresh pack queue/input files required")
    formal = read(FORMAL_QUEUE)
    require(formal.get("status") not in ("FAILED", "STOPPED", "CANCELLED"), "formal queue already failed/stopped")
    for profile in PROFILES:
        formal_rows(formal, profile, completed=False)
        if profile != "m1_none":
            require(not DESTS[profile].exists(), f"{profile}: package destination must be fresh")
    pins = input_files()
    write(INPUTS, {"schema": "root_final_ablation_pack_inputs_v1", "utc": now(), "file_sha256": pins,
                   "file_count": len(pins), "formal_input_path": formal["freeze_path"],
                   "formal_input_sha256": formal["freeze_sha256"], "external_actions": False})
    jobs = [{"profile": profile, "dest": str(DESTS[profile]),
             "argv": [str(PYTHON), str(PACKER), "--profile", profile, "--dest", str(DESTS[profile])],
             "log": str(OUT / f"pack_{profile}_attempt01.log"), "status": "PENDING"} for profile in PROFILES]
    require(all(not Path(job["log"]).exists() for job in jobs), "pack logs must be fresh")
    write(QUEUE, {"schema": "root_final_ablation_pack_queue_v1", "status": "PREPARED", "created_utc": now(),
                  "input_path": str(INPUTS), "input_sha256": sha(INPUTS), "formal_queue": str(FORMAL_QUEUE),
                  "formal_input_path": formal["freeze_path"], "formal_input_sha256": formal["freeze_sha256"],
                  "profiles": list(PROFILES), "jobs": jobs,
                  "official_submission_execution": "NOT_PART_OF_THIS_CONTROLLER", "external_actions": False})
    print(json.dumps({"status": "PREPARED", "jobs": len(jobs), "input_files": len(pins)}), flush=True)


def validate_inputs(queue: dict[str, Any], profile: str) -> list[dict[str, Any]]:
    require(sha(INPUTS) == queue["input_sha256"], "pack input receipt changed")
    pins = read(INPUTS)
    require(pins.get("formal_input_path") == queue["formal_input_path"], "formal input path drift")
    require(pins.get("formal_input_sha256") == queue["formal_input_sha256"], "formal input SHA drift")
    for name, digest in pins["file_sha256"].items():
        require(sha(Path(name)) == digest, f"pack input changed: {name}")
    # Reuse the training queue's authoritative formal-freeze validation.
    sys.path.insert(0, str(HERE))
    import run_queue  # pylint: disable=import-outside-toplevel
    formal = read(FORMAL_QUEUE)
    require(formal.get("status") not in ("FAILED", "STOPPED", "CANCELLED"), "formal queue failed/stopped")
    require(formal.get("freeze_path") == queue["formal_input_path"], "formal freeze path drift")
    require(formal.get("freeze_sha256") == queue["formal_input_sha256"], "formal freeze SHA drift")
    run_queue.validate_inputs(formal)
    return formal_rows(formal, profile, completed=True)


def verify_manifest(job: dict[str, Any]) -> None:
    profile = str(job["profile"]); dest = Path(str(job["dest"]))
    manifest_path = dest / "artifacts/fixed_control_manifest.json"
    container_path = dest / "artifacts/root_container_verify.json"
    manifest, container = read(manifest_path), read(container_path)
    require(manifest.get("status") == "LOCAL_PACKAGE_READY_NOT_SUBMITTED", f"{profile}: manifest status drift")
    require(manifest.get("profile") == profile and manifest.get("formal") is True, f"{profile}: manifest formal/profile drift")
    require(manifest.get("formal_queue") == str(FORMAL_QUEUE), f"{profile}: manifest formal queue drift")
    payload = Path(str(manifest.get("payload_path", "")))
    require(payload.is_file() and str(payload).startswith(str(dest) + os.sep), f"{profile}: manifest payload outside package")
    require(manifest.get("payload_sha256") == sha(payload), f"{profile}: manifest payload SHA drift")
    build_path = Path(str(manifest.get("build_receipt_path", "")))
    host_path = Path(str(manifest.get("host_receipt_path", "")))
    require(build_path.is_file() and host_path.is_file(), f"{profile}: missing build/host receipt")
    require(manifest.get("build_receipt_sha256") == sha(build_path), f"{profile}: build receipt SHA drift")
    require(manifest.get("host_receipt_sha256") == sha(host_path), f"{profile}: host receipt SHA drift")
    require(manifest.get("container_receipt_path") == str(container_path), f"{profile}: container receipt path drift")
    require(manifest.get("container_receipt_sha256") == sha(container_path), f"{profile}: container receipt SHA drift")
    require(container.get("status") == "PASSED" and container.get("profile") == profile, f"{profile}: container verification failed")
    require(container.get("payload_sha256") == manifest.get("payload_sha256"), f"{profile}: container/payload SHA drift")
    require(container.get("image_id") == manifest.get("image_id") and isinstance(container.get("image_id"), str), f"{profile}: image ID drift")
    require(container.get("bytes_exact") is True and container.get("smoke_exit_code") == 0, f"{profile}: container bytes/smoke verification failed")
    require(container.get("external_actions") is False and manifest.get("external_actions") is False, f"{profile}: external action marker drift")


def run() -> None:
    with LOCK.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = read(QUEUE)
        require(queue.get("status") == "PREPARED", "pack queue is fresh-only; inspect prior execution")
        queue.update(status="RUNNING", pid=os.getpid(), started_utc=now())
        write(QUEUE, queue)
        environment = os.environ.copy()
        environment.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
        try:
            for job in queue["jobs"]:
                while True:
                    require(not STOP.exists(), "requested stop before next pack")
                    formal = read(FORMAL_QUEUE)
                    require(formal.get("status") not in ("FAILED", "STOPPED", "CANCELLED"), "formal queue failed/stopped")
                    rows = formal_rows(formal, str(job["profile"]), completed=False)
                    require(not any(row.get("status") in ("FAILED", "STOPPED", "CANCELLED") for row in rows), f"{job['profile']}: required formal row failed/stopped")
                    if all(row.get("status") == "COMPLETED" and row.get("returncode") == 0 for row in rows):
                        validate_inputs(queue, str(job["profile"]))
                        break
                    queue.update(status="WAITING_FOR_FORMAL_STAGE", heartbeat_utc=now())
                    write(QUEUE, queue)
                    time.sleep(30)
                queue.update(status="RUNNING", heartbeat_utc=now())
                write(QUEUE, queue)
                log_path = Path(str(job["log"])); require(not log_path.exists(), f"pack log exists: {log_path}")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("x", encoding="utf-8") as log:
                    child = subprocess.Popen(job["argv"], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    job.update(status="RUNNING", child_pid=child.pid, started_utc=now())
                    write(QUEUE, queue)
                    print(json.dumps({"event": "PACK_STARTED", "profile": job["profile"], "pid": child.pid}), flush=True)
                    while child.poll() is None:
                        queue["heartbeat_utc"] = now(); write(QUEUE, queue); time.sleep(15)
                job.update(returncode=child.returncode, finished_utc=now())
                require(child.returncode == 0, f"pack failed: {job['profile']}; see {log_path}")
                verify_manifest(job)
                job["status"] = "COMPLETED"; write(QUEUE, queue)
                print(json.dumps({"event": "PACK_COMPLETED", "profile": job["profile"]}), flush=True)
            queue.update(status="COMPLETED", finished_utc=now()); write(QUEUE, queue)
        except BaseException as error:
            queue.update(status="FAILED", failure=str(error), failed_utc=now())
            for job in queue["jobs"]:
                if job.get("status") == "RUNNING": job["status"] = "FAILED"
            write(QUEUE, queue)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run"), required=True)
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else run()
