#!/usr/bin/env python3
"""Detached, receipt-backed launcher for the approved DANDI formal GPU campaign.

``--start`` is deliberately explicit.  A launched representation worker runs
the preregistered encoder/decoder seed roster on one GPU and persists an
exit-code receipt even after the invoking terminal disconnects.  It neither
times out nor retries a task automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
PYTHON = Path("/home/xinyuan/miniconda3/envs/spint/bin/python")
GPU_EXECUTION_POLICY = PACKAGE / "results" / "GPU_EXECUTION_POLICY.json"


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def plan(root: Path, cache: Path, representation: str, gpu: int) -> list[list[str]]:
    base = [str(PYTHON), str(PACKAGE / "run.py")]
    common = ["--cache", str(cache), "--representation", representation, "--seed", "42", "--device", "cuda"]
    pretrain = root / f"pretrain_{representation}_s42"
    return [base + ["pretrain", *common, "--dest", str(pretrain)],
            base + ["train", *common, "--arm", "full", "--encoder", str(pretrain / "encoder.pt"), "--dest", str(root / f"train_{representation}_full_s42")],
            base + ["train", *common, "--arm", "activity", "--dest", str(root / f"train_{representation}_activity_s42")],
            base + ["train", *common, "--arm", "raw_set", "--dest", str(root / f"train_{representation}_raw_set_s42")],
            base + ["train", "--cache", str(cache), "--representation", representation, "--seed", "43", "--device", "cuda",
                    "--arm", "full", "--encoder", str(pretrain / "encoder.pt"), "--dest", str(root / f"train_{representation}_full_s43")],
            base + ["train", "--cache", str(cache), "--representation", representation, "--seed", "44", "--device", "cuda",
                    "--arm", "full", "--encoder", str(pretrain / "encoder.pt"), "--dest", str(root / f"train_{representation}_full_s44")]]


def worker(args: argparse.Namespace) -> int:
    root, log = Path(args.root).resolve(), Path(args.log).resolve()
    commands = plan(root, Path(args.cache).resolve(), args.representation, args.gpu)
    receipt_path = root / f"worker_{args.representation}.json"
    started = time.time()
    receipt = {"schema": "dandi688_v2_campaign_worker", "representation": args.representation, "gpu": args.gpu,
               "pid": os.getpid(), "started_unix": started, "commands": commands, "status": "RUNNING"}
    atomic_json(receipt_path, receipt)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONNOUSERSITE": "1"}
    env.pop("PYTHONPATH", None)
    with log.open("a", buffering=1) as handle:
        for index, command in enumerate(commands, 1):
            print(json.dumps({"event": "command_start", "index": index, "command": command}), file=handle, flush=True)
            result = subprocess.run(command, cwd=PACKAGE.parents[1], env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                receipt.update({"status": "FAILED", "failed_index": index, "exit_code": result.returncode,
                                "ended_unix": time.time()})
                atomic_json(receipt_path, receipt)
                return result.returncode
    receipt.update({"status": "SUCCEEDED", "exit_code": 0, "ended_unix": time.time()})
    atomic_json(receipt_path, receipt)
    return 0


def start(args: argparse.Namespace) -> None:
    root, cache = Path(args.root).resolve(), Path(args.cache).resolve()
    # This gate deliberately precedes all directory creation and process work.
    # A user pause cannot be bypassed with another launcher option.
    if GPU_EXECUTION_POLICY.exists():
        policy = json.loads(GPU_EXECUTION_POLICY.read_text())
        if policy.get("enabled") is False:
            raise RuntimeError("GPU campaign start is disabled by user execution policy")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"campaign root must be fresh: {root}")
    if not cache.is_dir():
        raise FileNotFoundError(cache)
    root.mkdir(parents=True, exist_ok=False)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for representation, gpu in (("sua", args.gpu_sua), ("pmua", args.gpu_pmua)):
        log = root / f"{representation}.stdout.log"
        command = [str(PYTHON), str(Path(__file__).resolve()), "_worker", "--root", str(root), "--cache", str(cache),
                   "--representation", representation, "--gpu", str(gpu), "--log", str(log)]
        with log.open("a", buffering=1) as handle:
            process = subprocess.Popen(command, cwd=PACKAGE.parents[1], stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        atomic_json(root / f"launch_{representation}.json", {"schema": "dandi688_v2_campaign_launch", "representation": representation,
                    "gpu": gpu, "pid": process.pid, "started_utc": stamp, "stdout_log": str(log), "worker_command": command,
                    "planned_commands": plan(root, cache, representation, gpu), "automatic_retry": False, "automatic_timeout": False})
        print(json.dumps({"representation": representation, "pid": process.pid, "log": str(log)}))


def status(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    for representation in ("sua", "pmua"):
        launch = root / f"launch_{representation}.json"
        worker_receipt = root / f"worker_{representation}.json"
        if not launch.exists():
            print(json.dumps({"representation": representation, "status": "NOT_LAUNCHED"}))
            continue
        info = json.loads(launch.read_text())
        receipt = json.loads(worker_receipt.read_text()) if worker_receipt.exists() else None
        running = False
        try:
            os.kill(int(info["pid"]), 0); running = True
        except ProcessLookupError:
            pass
        print(json.dumps({"representation": representation, "pid": info["pid"], "pid_running": running,
                          "worker_receipt": receipt, "stdout_log": info["stdout_log"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "start", "status"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True, type=Path)
        p.add_argument("--cache", type=Path, default=PACKAGE / "results" / "prepared_2015_m33_v2")
        p.add_argument("--gpu-sua", type=int, default=0); p.add_argument("--gpu-pmua", type=int, default=1)
    p = sub.add_parser("_worker")
    p.add_argument("--root", required=True); p.add_argument("--cache", required=True); p.add_argument("--representation", choices=("sua", "pmua"), required=True)
    p.add_argument("--gpu", type=int, required=True); p.add_argument("--log", required=True)
    args = parser.parse_args()
    if args.command == "_worker":
        raise SystemExit(worker(args))
    if args.command == "plan":
        for representation, gpu in (("sua", args.gpu_sua), ("pmua", args.gpu_pmua)):
            print(json.dumps({"representation": representation, "gpu": gpu, "commands": plan(args.root.resolve(), args.cache.resolve(), representation, gpu)}, indent=2))
    elif args.command == "start": start(args)
    else: status(args)


if __name__ == "__main__": main()
