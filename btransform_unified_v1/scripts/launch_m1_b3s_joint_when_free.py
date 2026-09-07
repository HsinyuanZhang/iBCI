#!/usr/bin/env python3
"""Start M1 B3S+BT P16 D2 train only on a demonstrably idle GPU.

Does not kill or share with RIFT / other compute. Two idle samples 12s apart.
Prefer GPU1. After train: score + ext6 pick e18:24. Does not pack or submit.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/xinyuan/Work_host/SPINT")
SCR = ROOT / "btransform_unified_v1/scripts/m1_b3s_joint_series.py"
PY = Path("/home/xinyuan/miniconda3/envs/spint/bin/python")
IDLE_MEM = 600
IDLE_UTIL = 5
SAMPLE_S = 12.0
POLL_S = 30.0
ENV_FLAG = "BTRANSFORM_M1_B3S_JOINT_TRAIN"


def _gpus() -> list[dict]:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip().splitlines()
    pids: dict[str, list[int]] = {}
    for line in apps:
        if not line.strip() or line.strip().startswith("No running"):
            continue
        uuid, pid = [p.strip() for p in line.split(",", 1)]
        pids.setdefault(uuid, []).append(int(pid))
    out = []
    for line in rows:
        idx, uuid, mem, util = [p.strip() for p in line.split(",")]
        out.append(
            {
                "index": idx,
                "uuid": uuid,
                "mem": int(mem),
                "util": int(util),
                "pids": tuple(pids.get(uuid, ())),
            }
        )
    return out


def _idle(gpu: dict) -> bool:
    return not gpu["pids"] and gpu["mem"] < IDLE_MEM and gpu["util"] <= IDLE_UTIL


def _two_sample() -> dict | None:
    first = {g["uuid"]: g for g in _gpus() if _idle(g)}
    time.sleep(SAMPLE_S)
    second = {g["uuid"]: g for g in _gpus() if _idle(g)}
    shared = first.keys() & second.keys()
    if not shared:
        return None
    return sorted((second[u] for u in shared), key=lambda g: (g["index"] != "1", int(g["index"])))[0]


def _run(stage: str, root: Path, dest: Path, gpu_index: str, log: Path) -> int:
    env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "4",
        "CUDA_VISIBLE_DEVICES": gpu_index,
        ENV_FLAG: "1",
    }
    cmd = [
        str(PY),
        str(SCR),
        "--stage",
        stage,
        "--face",
        "fullsession",
        "--depth",
        "2",
        "--cuda-visible",
        gpu_index,
        "--root",
        str(root),
        "--dest",
        str(dest),
    ]
    if stage == "pick":
        cmd.extend(["--pick-epochs", "18:24"])
    with log.open("a") as handle:
        handle.write(f"\n# {datetime.now(timezone.utc).isoformat()} stage={stage} gpu={gpu_index}\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: launch_m1_b3s_joint_when_free.py <root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    dest = root / "depth2"
    dest.mkdir(parents=True, exist_ok=True)
    state_path = root / "launch_state.json"
    log = dest / "launch.log"

    if (dest / "epoch_pick.json").exists():
        print("pick already sealed; refuse relaunch", flush=True)
        return 0
    if not (root / "session_inventory.json").exists():
        print("REFUSED: run --stage probe first (CPU)", file=sys.stderr)
        return 2

    print(f"[wait] root={root} polling for idle GPU", flush=True)
    while True:
        gpu = _two_sample()
        state = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "waiting": gpu is None,
            "gpu": gpu,
        }
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        if gpu is None:
            time.sleep(POLL_S)
            continue
        print(f"[launch] idle GPU{gpu['index']} {gpu['uuid']}", flush=True)
        if not (dest / "train_receipt.json").exists():
            code = _run("train", root, dest, gpu["index"], log)
            if code != 0:
                print(f"train exit {code}", file=sys.stderr)
                return code
        if not (dest / "score_receipt.json").exists():
            code = _run("score", root, dest, gpu["index"], log)
            if code != 0:
                print(f"score exit {code}", file=sys.stderr)
                return code
        code = _run("pick", root, dest, gpu["index"], log)
        print(f"pick exit {code}", flush=True)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
