#!/usr/bin/env python3
"""Root-owned two-GPU integration queue for the fixed 33-cell protocol.

This orchestrates existing numerical entrypoints without changing their recipe.
The first M1/M2 training processes are adopted by PID; all remaining work is
launched here. A failure stops new admissions while another live fit finishes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cross_session_v1"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
LOCK = threading.RLock()
FOLD_LOCKS: dict[str, threading.Lock] = {}
FAILED = threading.Event()
PROGRAM: dict = {}
SEAL: dict = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def update(cell, **fields):
    with LOCK:
        cell.update(fields, updated_at=now())
        PROGRAM["updated_at"] = now()
        atomic(OUT / "program.json", PROGRAM)


def check_seal():
    for path, expected in SEAL["files"].items():
        if sha(path) != expected:
            raise RuntimeError(f"sealed source changed: {path}")
    if shutil.disk_usage(ROOT).free < 3 * 1024**3:
        raise RuntimeError("less than 3 GiB free disk before numerical stage")


def run_dir(cell):
    dataset, fold = cell["dataset"], cell["fold"]
    group = "m2_ext4" if dataset == "m2" else ("m1_loso_" + fold.removeprefix("ses-") if dataset == "m1" else "h1_lodo_" + fold)
    return OUT / group / cell["arm"] / "s42"


def date(cell):
    f = cell["fold"]
    return f"{f[:4]}-{f[4:6]}-{f[6:]}"


def env(gpu):
    e = os.environ.copy()
    e.update(PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", CUDA_VISIBLE_DEVICES=str(gpu))
    e["PYTHONPATH"] = ":".join(map(str, (ROOT, ROOT / "src", ROOT.parent / "btransform_unified_v1/src", ROOT.parent)))
    return e


def invoke(cell, gpu, label, args):
    check_seal()
    logdir = OUT / "queue_logs"
    logdir.mkdir(exist_ok=True)
    logfile = logdir / f'{cell["dataset"]}_{cell["fold"]}_{cell["arm"]}_{label}.log'
    if logfile.exists():
        raise FileExistsError(f"refusing to overwrite stage log {logfile}")
    command = [PYTHON, *map(str, args)]
    with logfile.open("w") as stream:
        proc = subprocess.Popen(command, cwd=ROOT, env=env(gpu), stdout=stream, stderr=subprocess.STDOUT)
        update(cell, stage=label, pid=proc.pid, gpu=gpu, run=str(run_dir(cell)), log=str(logfile), command=command)
        print(json.dumps({"event": "launch", "time": now(), "pid": proc.pid, "gpu": gpu, "label": label, "run": str(run_dir(cell))}), flush=True)
        code = proc.wait()
    if code:
        raise RuntimeError(f"{label} exited {code}; see {logfile}")


def args_for(cell, stage):
    ds, arm, run = cell["dataset"], cell["arm"], run_dir(cell)
    common = ["--dest", run, "--arm", arm, "--seed", "42", "--device", "cuda:0"]
    if ds == "m1":
        script = "m1_score.py" if stage == "score" else "m1_train.py"
        return [f"scripts/cross_session_v1/{script}", *common, "--target", cell["fold"], *([] if stage == "score" else ["--stage", stage])]
    if ds == "m2":
        script = "m2_score.py" if stage == "score" else "m2_train.py"
        return [f"scripts/cross_session_v1/{script}", *common, "--cpu-threads", "2", *([] if stage == "score" else ["--stage", stage, "--encoder-init", "random"])]
    return ["scripts/cross_session_v1/h1_train.py", *common, "--prepared", OUT / "h1_prepared_v2", "--fold", date(cell), "--stage", stage, "--cpu-threads", "2"]


def ensure_preflight(cell, gpu):
    ds = cell["dataset"]
    if ds == "m2":
        proof = json.loads((OUT / "m2_preflight_s42_attempt4.json").read_text())
        required = {"schema": "cross_session_m2_preflight_v1", "status": "COMPLETED", "cell": "M2-CROSS-SESSION-R50-D4-ZBD-M33-V1", "seed": 42, "rosters_disjoint": True}
        if any(proof.get(k) != v for k, v in required.items()):
            raise RuntimeError("M2 global preflight binding mismatch")
        if not proof.get("actual_intervention_checks") or not all(proof["actual_intervention_checks"].values()):
            raise RuntimeError("M2 intervention preflight incomplete")
        for arm in ("Z_NONE", "B_ACTIVITY_ONLY", "D_JOINT"):
            checks = proof.get("gradient_checks", {}).get(arm, {})
            if checks.get("finite_output") is not True or (arm != "Z_NONE" and checks.get("encoder_grad") is not True):
                raise RuntimeError("M2 gradient preflight incomplete")
        return
    key = ds + "/" + cell["fold"]
    with LOCK:
        fold_lock = FOLD_LOCKS.setdefault(key, threading.Lock())
    with fold_lock:
        canonical_cell = dict(cell, arm="Z_NONE")
        canonical_run = run_dir(canonical_cell)
        canonical = canonical_run / "preflight.json"
        # H1's first receipt predates a verified encoder dedup optimization.
        # A dedicated current-code receipt is kept outside all formal runs.
        if ds == "h1":
            canonical = OUT / "fold_preflights" / ("h1_" + cell["fold"]) / "preflight.json"
        if not canonical.exists():
            command = args_for(canonical_cell, "preflight")
            if ds == "h1":
                manifest = OUT / "h1_prepared_v2" / date(cell) / "source/manifest.json"
                if not manifest.is_file():
                    raise FileNotFoundError(manifest)
                command[command.index("--dest") + 1] = canonical.parent
            invoke(cell, gpu, "fold_preflight", command)
        proof = json.loads(canonical.read_text())
        expected_fold = cell["fold"] if ds == "m1" else date(cell)
        if proof.get("fold") != expected_fold or proof.get("source_only") is not True:
            raise RuntimeError("canonical fold preflight binding mismatch")
        if ds == "m1" and proof.get("status") != "PASSED":
            raise RuntimeError("M1 canonical preflight did not pass")
        if ds == "h1" and not all(proof.get("finite_backward", {}).get(a) is True for a in ("Z_NONE", "B_ACTIVITY_ONLY", "D_JOINT")):
            raise RuntimeError("H1 canonical preflight did not pass all arms")
        destination = run_dir(cell) / "preflight.json"
        if destination != canonical:
            if destination.exists():
                # Preserve the old pre-dedup H1 receipt explicitly.
                if ds == "h1" and destination.read_bytes() != canonical.read_bytes():
                    prior = OUT / "fold_preflights" / ("h1_" + cell["fold"]) / ("prior_" + cell["arm"] + ".json")
                    if not prior.exists():
                        shutil.copyfile(destination, prior)
                elif ds == "m1":
                    return
            reused = copy.deepcopy(proof)
            reused["arm"] = cell["arm"]
            reused["reused_all_arm_preflight"] = {"path": str(canonical), "sha256": sha(canonical), "reason": "same source fold/seed; original numerical preflight covers every Z/B/D arm"}
            atomic(destination, reused)


def target_prepare(cell, gpu):
    if cell["dataset"] != "h1":
        return
    key = "h1_target/" + cell["fold"]
    with LOCK:
        fold_lock = FOLD_LOCKS.setdefault(key, threading.Lock())
    with fold_lock:
        manifest = OUT / "h1_prepared_v2" / date(cell) / "target/manifest.json"
        if not manifest.is_file():
            invoke(cell, gpu, "target_prepare", ["scripts/cross_session_v1/h1_prepare.py", "--dest", OUT / "h1_prepared_v2", "--fold", date(cell), "--surface", "target"])


def complete_cell(cell, gpu, adopted_pid=None):
    if adopted_pid is not None:
        while Path(f"/proc/{adopted_pid}").exists():
            time.sleep(10)
        if not (run_dir(cell) / "train_receipt.json").is_file():
            raise RuntimeError(f"adopted training pid {adopted_pid} ended without a completed receipt")
    else:
        ensure_preflight(cell, gpu)
        update(cell, status="TRAINING")
        invoke(cell, gpu, "train", args_for(cell, "train"))
    update(cell, status="SCORING")
    target_prepare(cell, gpu)
    invoke(cell, gpu, "score", args_for(cell, "score"))
    receipt = run_dir(cell) / ("target_score.json" if cell["dataset"] == "h1" else "score_receipt.json")
    if not receipt.is_file():
        raise FileNotFoundError(receipt)
    update(cell, status="COMPLETED", score_receipt=str(receipt), score_sha256=sha(receipt), pid=None, completed_at=now())
    print(json.dumps({"event": "cell_completed", "time": now(), "run": str(run_dir(cell)), "gpu": gpu}), flush=True)


def take():
    with LOCK:
        if FAILED.is_set():
            return None
        pending = [x for x in PROGRAM["cells"] if x["status"] == "PENDING"]
        if not pending:
            return None
        cell = min(pending, key=lambda c: c["queue_priority"])
        update(cell, status="PREPARING")
        return cell


def worker(gpu, adopted_pid, initial):
    cell = initial
    try:
        complete_cell(cell, gpu, adopted_pid)
        while (cell := take()) is not None:
            complete_cell(cell, gpu)
    except Exception as exc:
        FAILED.set()
        if cell is not None:
            update(cell, status="FAILED", error=f"{type(exc).__name__}: {exc}")
        print(json.dumps({"event": "failure", "time": now(), "gpu": gpu, "error": str(exc)}), flush=True)


def main():
    global PROGRAM, SEAL
    p = argparse.ArgumentParser()
    p.add_argument("--adopt-m2-pid", required=True, type=int)
    p.add_argument("--adopt-m1-pid", required=True, type=int)
    a = p.parse_args()
    PROGRAM = json.loads((OUT / "program.json").read_text())
    SEAL = json.loads((OUT / "execution_source_seal.json").read_text())
    check_seal()
    initial = {x["dataset"]: x for x in PROGRAM["cells"] if x["status"] == "TRAINING"}
    if set(initial) != {"m1", "m2"}:
        raise RuntimeError("initial adoption requires exactly M1/M2 active fits")
    # Complete a full three-arm target fold per task early, then distribute the
    # rest dynamically; no scientific endpoint is used for queue priority.
    for i, c in enumerate(PROGRAM["cells"]):
        early = 0 if c["dataset"] == "m2" else (1 if c["dataset"] == "h1" and c["fold"] == "19250101" else (2 if c["dataset"] == "m1" and c["fold"] == "ses-20120924" else 3))
        c["queue_priority"] = early * 100 + i
    PROGRAM.update(execution_seal=str(OUT / "execution_source_seal.json"), queue_pid=os.getpid(), queue_started_at=now())
    atomic(OUT / "program.json", PROGRAM)
    threads = [threading.Thread(target=worker, args=(0, a.adopt_m2_pid, initial["m2"])), threading.Thread(target=worker, args=(1, a.adopt_m1_pid, initial["m1"]))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    PROGRAM["status"] = "QUEUE_FAILED" if FAILED.is_set() else "PRIMARY_GRID_COMPLETED_AWAITING_FIGURE"
    atomic(OUT / "program.json", PROGRAM)
    if FAILED.is_set():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
