"""Root-operated scheduler for six fresh temporal fits and three reused M2 fits."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from protocol import ARMS, DATASETS, OUT, ROOT, atomic_json, protocol, sha

HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def environment(gpu):
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE="1", PYTHONWARNINGS="ignore", CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", MPLBACKEND="Agg")
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=OUT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume-score-fix-handoff", action="store_true")
    args = parser.parse_args()
    dest = args.dest.resolve()
    if read(dest / "protocol.json") != protocol():
        raise RuntimeError("protocol differs from fixed temporal authority")
    ledger_path = dest / "program.json"
    if ledger_path.exists() and not args.resume_score_fix_handoff:
        raise FileExistsError("fresh scheduler refuses existing ledger; inspect existing process before any recovery")
    logs = dest / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cells = []
    for dataset in ("m1", "h1", "m2"):
        for arm in ARMS:
            run = dest / dataset / arm / "s42" if dataset == "m1" else dest / dataset / arm
            cells.append({"dataset": dataset, "arm": arm, "run": str(run), "status": "WAITING_REUSE" if dataset == "m2" else "WAITING_PREFLIGHT", "pid": None})
    ledger = {"schema": "chronological_last2_program_v1", "status": "RUNNING", "created_at": now(),
              "controller_pid": os.getpid(), "protocol_sha256": sha(dest / "protocol.json"),
              "scheduler_sha256": sha(Path(__file__)), "cells": cells, "dataset_code_sha256": {},
              "h1_target_prepare": {"status": "WAITING_SOURCE_SELECTIONS"},
              "concurrency": {"max_per_gpu": 2, "max_h1_per_gpu": 1, "gpus": [0, 1]}}
    active = {}
    prepare_process = None
    if args.resume_score_fix_handoff:
        previous = read(ledger_path)
        handoff_path = dest / "controller_score_fix_handoff.json"
        handoff = read(handoff_path)
        archive = dest / "program_before_score_comparison_fix.json"
        if (handoff.get("status") != "SCORE_COMPARISON_AND_RNG_RESTORE_VERIFIED_FOR_CHECKPOINT_RESUME"
                or sha(archive) != handoff["old_program_sha256"] or read(archive) != previous
                or Path(f"/proc/{previous['controller_pid']}").exists()
                or previous.get("status") != "RUNNING"
                or any(c["status"] not in ("TRAINING", "WAITING_PREFLIGHT", "COMPLETED") for c in previous["cells"])):
            raise RuntimeError("invalid verified checkpoint handoff")
        score_path = str(HERE / "m1_score.py")
        train_path = str(HERE / "m1_train.py")
        bindings = previous["dataset_code_sha256"]["m1"]
        if bindings[score_path] != handoff["m1_score_before_sha256"] or sha(Path(score_path)) != handoff["m1_score_after_sha256"]:
            raise RuntimeError("score comparison correction is not bound to the handoff")
        if bindings[train_path] != handoff["m1_train_before_sha256"] or sha(Path(train_path)) != handoff["m1_train_after_sha256"]:
            raise RuntimeError("RNG restoration correction is not bound to the handoff")
        for raw, digest in bindings.items():
            if raw not in (score_path, train_path) and sha(Path(raw)) != digest:
                raise RuntimeError("source science changed outside the two reviewed restoration/validation fixes")
        ledger = previous
        cells = ledger["cells"]
        ledger["dataset_code_sha256"]["m1"][score_path] = handoff["m1_score_after_sha256"]
        ledger["dataset_code_sha256"]["m1"][train_path] = handoff["m1_train_after_sha256"]
        ledger["controller_handoff"] = {"receipt": str(handoff_path), "sha256": sha(handoff_path), "old_controller_pid": ledger["controller_pid"], "old_scheduler_sha256": ledger["scheduler_sha256"]}
        ledger.update(controller_pid=os.getpid(), scheduler_sha256=sha(Path(__file__)))
        for cell in cells:
            if cell["status"] == "TRAINING":
                if cell["dataset"] != "m1" or Path(f"/proc/{cell['pid']}").exists():
                    raise RuntimeError("this explicit handoff is restricted to the three original M1 source fits")
                checkpoint = handoff["resume_checkpoints"][cell["arm"]]
                if sha(Path(checkpoint["path"])) != checkpoint["sha256"]:
                    raise RuntimeError("source checkpoint changed before recovery")
                cell.update(status="WAITING_PREFLIGHT", pid=None, resume=True, resume_checkpoint=checkpoint)

    def save():
        ledger["updated_at"] = now()
        ledger["counts"] = dict(Counter(c["status"] for c in cells))
        atomic_json(ledger_path, ledger)

    def bind_dataset(dataset):
        files = [HERE / f"{dataset}_{suffix}.py" for suffix in (("data", "train", "score") if dataset == "m1" else ("prepare", "train", "audit"))]
        files += [HERE / "protocol.py", ROOT / f"src/btransform_unified_v2/cross_session_{dataset}_model.py"]
        if dataset == "h1":
            files += [ROOT / "scripts/cross_session_v1/h1_prepare.py", ROOT / "scripts/cross_session_v1/h1_train.py"]
        current = {str(p): sha(p) for p in files}
        previous = ledger["dataset_code_sha256"].setdefault(dataset, current)
        if current != previous:
            raise RuntimeError(f"science code changed after source launch: {dataset}")

    def fresh_ready(cell):
        path = Path(cell["run"]) / "preflight.json"
        if not path.is_file():
            return False
        row = read(path)
        if row.get("status") != "PASSED" or row.get("arm") != cell["arm"] or row.get("source_only") is not True:
            raise RuntimeError(f"invalid preflight: {path}")
        ds = DATASETS[cell["dataset"]]
        if cell["dataset"] == "m1":
            if row.get("split_id") != ds["split_id"] or row.get("sources") != ds["source_sessions"] or row.get("targets") != ds["target_sessions"]:
                raise RuntimeError("M1 source preflight roster mismatch")
        else:
            split = row.get("split", {})
            if split.get("split_id") != ds["split_id"] or split.get("source_sessions") != ds["source_sessions"] or split.get("target_sessions") != ds["target_sessions"] or row.get("target_records_opened") != 0:
                raise RuntimeError("H1 source preflight roster mismatch")
            if row.get("source_manifest_sha256") != sha(dest / "h1/prepared/source/manifest.json"):
                raise RuntimeError("H1 preflight source manifest drift")
        for raw, digest in row.get("source_code_sha256", {}).items():
            if sha(Path(raw)) != digest:
                raise RuntimeError("preflight code changed")
        cell["preflight_sha256"] = sha(path)
        return True

    def command(cell, phase):
        run = Path(cell["run"])
        if cell["dataset"] == "m1":
            cmd = [args.python, "-u", str(HERE / ("m1_train.py" if phase == "train" else "m1_score.py")), "--dest", str(run), "--arm", cell["arm"], "--seed", "42", "--device", "cuda:0"]
            if phase == "train":
                cmd += ["--stage", "train"]
                if cell.get("resume"):
                    cmd += ["--resume"]
            return cmd
        return [args.python, "-u", str(HERE / "h1_train.py"), "--prepared", str(dest / "h1/prepared"), "--dest", str(run), "--arm", cell["arm"], "--stage", phase, "--seed", "42", "--device", "cuda:0", "--cpu-threads", "2"]

    def launch(cell, phase, gpu):
        bind_dataset(cell["dataset"])
        suffix = "_resumed" if phase == "train" and cell.get("resume") else ""
        logfile = logs / f"{cell['dataset']}_{cell['arm']}_{phase}{suffix}.log"
        handle = logfile.open("x")
        cmd = command(cell, phase)
        process = subprocess.Popen(cmd, cwd=ROOT.parent, env=environment(gpu), stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        active[process.pid] = (process, cell, phase, handle)
        cell.update(status="TRAINING" if phase == "train" else "SCORING", pid=process.pid, gpu=gpu, command=cmd, log=str(logfile), phase_started_at=now())
        print(json.dumps({"event": "started", "dataset": cell["dataset"], "arm": cell["arm"], "phase": phase, "gpu": gpu, "pid": process.pid}), flush=True)
        save()

    def finish(cell, phase):
        run = Path(cell["run"])
        if phase == "train":
            receipt = read(run / "train_receipt.json")
            if cell["dataset"] == "m1":
                if receipt.get("status") != "COMPLETED" or receipt.get("epochs") != 24 or receipt.get("split_id") != DATASETS["m1"]["split_id"]:
                    raise RuntimeError("M1 training did not complete fixed temporal run")
            elif receipt.get("schema") != "h1_chronological_last2_train_receipt_v1" or len(receipt.get("curve", [])) != 32:
                raise RuntimeError("H1 training did not complete 32 epochs")
            cell.update(status="TRAINED", train_receipt_sha256=sha(run / "train_receipt.json"), source_selection=receipt["selected"])
        else:
            paths = [run / f"score_{s}/score_receipt.json" for s in DATASETS["m1"]["target_sessions"]] if cell["dataset"] == "m1" else [run / "target_score.json"]
            for path in paths:
                row = read(path)
                if row.get("arm") != cell["arm"]:
                    raise RuntimeError("target score arm mismatch")
                if cell["dataset"] == "m1" and (row.get("status") != "COMPLETED" or row.get("target_optimizer_steps") != 0 or row.get("target_labels_used_for_selection") is not False):
                    raise RuntimeError("M1 target scoring scope mismatch")
                if cell["dataset"] == "h1" and (row.get("target_query_labels_used_for_selection") is not False or row.get("target_query_labels_used_for_gradients") is not False):
                    raise RuntimeError("H1 target scoring scope mismatch")
            cell.update(status="COMPLETED", score_receipt_sha256={str(p): sha(p) for p in paths})
        cell.update(pid=None, phase_finished_at=now())

    save()
    try:
        while True:
            for pid, (process, cell, phase, handle) in list(active.items()):
                rc = process.poll()
                if rc is None:
                    continue
                if handle is not None:
                    handle.close()
                del active[pid]
                if rc != 0:
                    cell.update(status="FAILED", returncode=rc, pid=None)
                    raise RuntimeError(f"{cell['dataset']}/{cell['arm']}/{phase} failed with {rc}; inspect {cell['log']}")
                finish(cell, phase)
                save()
            reuse = dest / "m2/m2_reuse_receipt.json"
            if reuse.is_file() and any(c["dataset"] == "m2" and c["status"] != "COMPLETED" for c in cells):
                row = read(reuse)
                if row.get("status") != "COMPLETED" or row.get("schema") != "chronological_m2_reuse_v1":
                    raise RuntimeError("invalid M2 reuse receipt")
                for cell in cells:
                    if cell["dataset"] == "m2":
                        cell.update(status="COMPLETED", reuse_receipt=str(reuse), reuse_receipt_sha256=sha(reuse), completed_at=now())
                save()
            for cell in cells:
                if cell["status"] == "WAITING_PREFLIGHT" and fresh_ready(cell):
                    cell["status"] = "PENDING_TRAIN"
            h1 = [c for c in cells if c["dataset"] == "h1"]
            if all(c["status"] in ("TRAINED", "SCORING", "COMPLETED") for c in h1):
                phase = ledger["h1_target_prepare"]
                if phase["status"] == "WAITING_SOURCE_SELECTIONS":
                    bind_dataset("h1")
                    log = logs / "h1_target_prepare.log"
                    handle = log.open("x")
                    cmd = [args.python, "-u", str(HERE / "h1_prepare.py"), "--dest", str(dest / "h1/prepared"), "--surface", "target"]
                    process = subprocess.Popen(cmd, cwd=ROOT.parent, env=environment(None), stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    prepare_process = (process, handle)
                    phase.update(status="PREPARING", pid=process.pid, command=cmd, log=str(log), started_at=now())
                    save()
                if prepare_process is not None and prepare_process[0].poll() is not None:
                    process, handle = prepare_process
                    handle.close()
                    if process.returncode != 0:
                        raise RuntimeError("H1 target preparation failed; inspect its log")
                    phase.update(status="COMPLETED", pid=None, completed_at=now(), target_manifest_sha256=sha(dest / "h1/prepared/target/manifest.json"))
                    prepare_process = None
                    save()
            for phase in ("score", "train"):
                for arm in ARMS:
                    for dataset in ("m1", "h1"):
                        cell = next(c for c in cells if c["dataset"] == dataset and c["arm"] == arm)
                        if phase == "train" and cell["status"] != "PENDING_TRAIN":
                            continue
                        if phase == "score":
                            group = [c for c in cells if c["dataset"] == dataset]
                            if cell["status"] != "TRAINED" or not all(c["status"] in ("TRAINED", "SCORING", "COMPLETED") for c in group):
                                continue
                            if dataset == "h1" and ledger["h1_target_prepare"]["status"] != "COMPLETED":
                                continue
                        counts = Counter(c["gpu"] for _, c, _, _ in active.values())
                        h1_counts = Counter(c["gpu"] for _, c, _, _ in active.values() if c["dataset"] == "h1")
                        available = [g for g in (0, 1) if counts[g] < 2 and (dataset != "h1" or h1_counts[g] == 0)]
                        if available:
                            launch(cell, phase, min(available, key=lambda g: (counts[g], g)))
            save()
            if all(c["status"] == "COMPLETED" for c in cells):
                ledger.update(status="COMPLETED", completed_at=now())
                save()
                print("All 9 chronological cells completed; final paired audits and paper integration remain.", flush=True)
                return
            time.sleep(5)
    except BaseException as exc:
        ledger.update(status="FAILED", error=str(exc), failed_at=now())
        for process, cell, phase, handle in active.values():
            process.terminate()
            cell.update(status="STOPPED_AFTER_PROGRAM_FAILURE", pid=None)
        if prepare_process is not None:
            prepare_process[0].terminate()
        save()
        raise


if __name__ == "__main__":
    main()
