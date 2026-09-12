"""Immutable-plan launcher for the post-Base19 Flat follow-up.

This module intentionally has no import-time training or final-scoring action.
``plan`` freezes a source/dev-only plan, and ``start`` is the sole training
launcher.  The later ``complete`` phase is deliberately explicit and lazily
imports the Flat finalizer so training workers never depend on it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from ..common import jsonable, sha256
from ..protocol import DEV_SESSIONS, FINAL_SESSIONS, TRAIN_SESSIONS, protocol_dict
from .contract import FLAT_RECIPE, FLAT_SCHEMA, flat_config_metadata, flat_training_source_hashes

SCHEMA = "dandi688_flat_followup_plan_v1"
STATUS = "FROZEN_FLAT_FOLLOWUP_PLAN"
PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parents[1]
DEFAULT_BASE_CAMPAIGN = PACKAGE / "results/formal_campaign_concat_2015_m33_allseeds_20260912"
DEFAULT_CACHE = PACKAGE / "results/prepared_2015_m33_v2"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(jsonable(dict(value)), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _file_binding(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256(path)}


def _encoder_binding(base_campaign: Path, representation: str) -> dict[str, str]:
    return _file_binding(base_campaign / f"pretrain_{representation}_s42/encoder.pt")


def _tasks(seed42_only: bool, base_campaign: Path) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    main: list[str] = []
    supplemental: list[str] = []
    tasks: list[dict[str, Any]] = []
    for representation in ("sua", "pmua"):
        sequence = (("full", 42), ("activity", 42), ("raw_set", 42))
        if not seed42_only:
            sequence += (("full", 43), ("full", 44))
        for arm, seed in sequence:
            cell = f"{arm}_flat_{representation}" + ("" if seed == 42 else f"_s{seed}")
            (main if seed == 42 else supplemental).append(cell)
            task = {"cell": cell, "run": f"train_{cell}", "arm": arm,
                    "representation": representation, "seed": seed}
            if arm == "full":
                task["encoder_path"] = str((base_campaign / f"pretrain_{representation}_s42/encoder.pt").resolve())
            else:
                task["encoder_path"] = None
            tasks.append(task)
    return main, supplemental, tasks


def create_plan(root: Path, *, cache: Path = DEFAULT_CACHE, base_campaign: Path = DEFAULT_BASE_CAMPAIGN,
                seed42_only: bool = False) -> dict[str, Any]:
    """Create an exclusive, self-hashing Flat training plan; never launches work."""
    root, cache, base_campaign = Path(root).resolve(), Path(cache).resolve(), Path(base_campaign).resolve()
    root.mkdir(parents=True, exist_ok=False)
    try:
        main, supplemental, tasks = _tasks(seed42_only, base_campaign)
        final_receipt = _file_binding(base_campaign / "final_concat_19/final_score_receipt.json")
        prepared = _file_binding(cache / "prepared_receipt.json")
        encoders = {rep: _encoder_binding(base_campaign, rep) for rep in ("sua", "pmua")}
        plan: dict[str, Any] = {
            "schema": SCHEMA, "status": STATUS, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base_results_already_observed": True, "planned_cells": main + supplemental,
            "flat_final_cells": main + supplemental,
            "main_cells": main, "supplemental_cells": supplemental, "training_tasks": tasks,
            "recipe": FLAT_RECIPE, "budget": {"segments": 24, "updates_per_segment": 3165, "batch": 32, "total_steps": 75960},
            "flat_training_source_hashes": flat_training_source_hashes(), "base_final_receipt": final_receipt,
            "prepared_receipt": prepared, "encoder_checkpoints": encoders, "cache": str(cache),
            "base_campaign": str(base_campaign), "final_roster": list(FINAL_SESSIONS),
        }
        plan["sha256"] = _digest(plan)
        _write(root / "plan.json", plan)
        return plan
    except Exception:
        # A failed plan must not leave a usable plan.json or partially declared root.
        (root / "plan.json").unlink(missing_ok=True)
        raise


def _load_plan(root: Path) -> dict[str, Any]:
    root = Path(root).resolve(); plan = _read(root / "plan.json")
    recorded = plan.pop("sha256", None)
    if plan.get("schema") != SCHEMA or plan.get("status") != STATUS or not isinstance(recorded, str) or _digest(plan) != recorded:
        raise ValueError("Flat plan schema/status/self-hash mismatch")
    plan["sha256"] = recorded
    base_campaign = Path(plan.get("base_campaign", "")).resolve()
    seed42_only = len(plan.get("planned_cells", ())) == 6
    expected_main, expected_supplemental, expected_tasks = _tasks(seed42_only, base_campaign)
    if (plan.get("base_results_already_observed") is not True or plan.get("recipe") != FLAT_RECIPE
            or plan.get("budget") != {"segments": 24, "updates_per_segment": 3165, "batch": 32, "total_steps": 75960}
            or plan.get("final_roster") != list(FINAL_SESSIONS)
            or plan.get("main_cells") != expected_main or plan.get("supplemental_cells") != expected_supplemental
            or plan.get("planned_cells") != expected_main + expected_supplemental
            or plan.get("flat_final_cells") != expected_main + expected_supplemental
            or plan.get("training_tasks") != expected_tasks):
        raise ValueError("Flat plan scientific scope, recipe, budget, roster, or task mapping mismatch")
    return plan


def _verify_plan_dependencies(plan: Mapping[str, Any]) -> None:
    for name in ("base_final_receipt", "prepared_receipt"):
        binding = plan[name]
        if sha256(Path(binding["path"])) != binding["sha256"]:
            raise ValueError(f"Flat plan dependency changed: {name}")
    base_final = _read(Path(plan["base_final_receipt"]["path"]))
    if base_final.get("schema") != "dandi688_v2_final_score" or base_final.get("status") != "FINAL_SCORED" or base_final.get("final_sessions_opened") != 6:
        raise ValueError("base final receipt is not the completed Base19 final score")
    prepared = _read(Path(plan["prepared_receipt"]["path"]))
    sessions = prepared.get("sessions", {})
    if (prepared.get("schema") != "dandi688_bench_v2_prepared_v1" or prepared.get("final_sessions_opened") != 0
            or not isinstance(sessions, dict) or set(sessions) != set(TRAIN_SESSIONS) | set(DEV_SESSIONS)
            or sum(item.get("split") == "train" for item in sessions.values()) != 18
            or sum(item.get("split") == "dev" for item in sessions.values()) != 6):
        raise ValueError("prepared receipt is not the locked 18-source/6-development no-final cache")
    encoders = plan.get("encoder_checkpoints")
    if not isinstance(encoders, Mapping) or set(encoders) != {"sua", "pmua"}:
        raise ValueError("Flat plan must bind exactly SUA and PMUA base encoders")
    for representation, binding in encoders.items():
        if sha256(Path(binding["path"])) != binding["sha256"]:
            raise ValueError(f"Flat plan encoder changed: {representation}")
    if plan.get("flat_training_source_hashes") != flat_training_source_hashes():
        raise ValueError("Flat training sources changed after plan freeze")


def _cuda_available() -> int:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Flat launcher refuses CPU fallback")
    return int(torch.cuda.device_count())


def _claim(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        with Path(path).open("x", encoding="utf-8") as handle:
            json.dump(jsonable(dict(payload)), handle, sort_keys=True)
    except FileExistsError as error:
        raise PermissionError(f"exclusive Flat claim already exists: {path}") from error


def _worker_command(root: Path, representation: str, gpu: int) -> list[str]:
    return [sys.executable, "-m", "btransform_unified_v2.dandi688_bench_v2.flat_extension.run_campaign",
            "_worker", "--root", str(Path(root).resolve()), "--representation", representation, "--gpu", str(gpu)]


def start(root: Path, *, gpus: tuple[int, int] = (0, 1)) -> dict[str, Any]:
    """Detach exactly two representation-serial CUDA queues after exclusive validation."""
    root = Path(root).resolve(); plan = _load_plan(root); _verify_plan_dependencies(plan); available = _cuda_available()
    if len(gpus) != 2 or gpus[0] == gpus[1] or any(not isinstance(gpu, int) or gpu < 0 or gpu >= available for gpu in gpus):
        raise ValueError("start requires two distinct valid CUDA GPU indices")
    _claim(root / ".flat_training_claim", {"plan_sha256": plan["sha256"], "status": "CLAIMED"})
    children = {}
    for representation, gpu in zip(("sua", "pmua"), gpus, strict=True):
        log = root / f"{representation}.flat_training.stdout.log"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        with log.open("xb") as handle:
            child = subprocess.Popen(_worker_command(root, representation, gpu), cwd=WORKSPACE, env=env,
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        children[representation] = {"pid": child.pid, "gpu": gpu, "log": str(log), "status": "RUNNING"}
    state = {"status": "RUNNING", "plan_sha256": plan["sha256"], "children": children,
             "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write(root / "launcher.json", state)
    return state


def _validate_task_receipt(task: Mapping[str, Any], destination: Path, plan: Mapping[str, Any]) -> None:
    receipt = _read(destination / "receipt.json")
    budget = plan["budget"]
    if (receipt.get("schema") != FLAT_SCHEMA + "_training" or receipt.get("status") != "FORMAL" or receipt.get("completed") is not True or receipt.get("stage") != "train"
            or receipt.get("arm") != task["arm"] or receipt.get("representation") != task["representation"]
            or receipt.get("seed") != task["seed"] or receipt.get("global_step") != budget["total_steps"]
            or receipt.get("actual_budget") != {k: budget[k] for k in ("segments", "updates_per_segment", "batch")}
            or receipt.get("final_sessions_opened") != 0 or receipt.get("device") != "cuda"
            or receipt.get("recipe") != FLAT_RECIPE or receipt.get("flat_recipe") != FLAT_RECIPE
            or receipt.get("flat_config") != flat_config_metadata()
            or receipt.get("code_hashes") != plan["flat_training_source_hashes"]):
        raise ValueError(f"Flat task receipt failed validation: {task['cell']}")
    proof = receipt.get("zero_slope_proof", {})
    if proof.get("slopes_shape") != [4, 8] or proof.get("slopes_zero_count") != 32 or proof.get("learnable_recency_parameters") != []:
        raise ValueError(f"Flat task lacks zero-slope proof: {task['cell']}")
    curve = receipt.get("segments")
    if not isinstance(curve, list) or len(curve) != 24:
        raise ValueError(f"Flat task has incomplete checkpoint curve: {task['cell']}")
    for index, row in enumerate(curve, 1):
        checkpoint = destination / f"segment_{index:02d}.pt"
        if (not checkpoint.is_file() or not isinstance(row, dict) or row.get("segment") != index
                or row.get("global_step") != index * 3165 or row.get("checkpoint") != checkpoint.name
                or row.get("checkpoint_sha256") != sha256(checkpoint)):
            raise ValueError(f"Flat checkpoint curve mismatch: {task['cell']} segment {index}")
    selection = _read(destination / "selection.json")
    selected = Path(selection.get("checkpoint", ""))
    if (selection.get("schema") != FLAT_SCHEMA + "_selection" or selection.get("status") != "FORMAL"
            or not selected.is_file() or selection.get("checkpoint_sha256") != sha256(selected)
            or selected.name not in {row["checkpoint"] for row in curve} or selection.get("final_sessions_opened") != 0):
        raise ValueError(f"Flat selection is incomplete: {task['cell']}")


def worker(root: Path, *, representation: str, gpu: int) -> dict[str, Any]:
    """Run one representation queue synchronously; failure stops its remaining tasks."""
    root = Path(root).resolve(); state_path = root / f"{representation}.worker.json"
    state: dict[str, Any] = {"status": "RUNNING", "representation": representation, "gpu": gpu,
                             "worker_pid": os.getpid(), "tasks": [], "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        plan = _load_plan(root); state["plan_sha256"] = plan["sha256"]; _verify_plan_dependencies(plan)
        tasks = [task for task in plan["training_tasks"] if task["representation"] == representation]
        if representation not in {"sua", "pmua"} or not tasks:
            raise ValueError("worker requires a registered representation queue")
        _claim(root / f".{representation}.worker_claim", {"plan_sha256": plan["sha256"], "gpu": gpu})
        _write(state_path, state)
        for task in tasks:
            _verify_plan_dependencies(plan)
            destination = root / task["run"]
            command = [sys.executable, "-m", "btransform_unified_v2.dandi688_bench_v2.flat_extension.training",
                       "--cache", plan["cache"], "--dest", str(destination), "--representation", representation,
                       "--arm", task["arm"], "--seed", str(task["seed"]), "--device", "cuda"]
            if task["encoder_path"] is not None:
                command.extend(("--encoder", task["encoder_path"]))
            task_state = {"cell": task["cell"], "command": command, "status": "RUNNING"}; state["tasks"].append(task_state); _write(state_path, state)
            child = subprocess.Popen(command, cwd=WORKSPACE, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
            task_state["child_pid"] = child.pid; _write(state_path, state)
            task_state["returncode"] = child.wait()
            if task_state["returncode"] != 0:
                raise RuntimeError(f"Flat training subprocess failed: {task['cell']}")
            _validate_task_receipt(task, destination, plan)
            task_state["status"] = "SUCCEEDED"; _write(state_path, state)
        state["status"] = "SUCCEEDED"
    except Exception as error:
        if state["tasks"] and state["tasks"][-1].get("status") == "RUNNING":
            state["tasks"][-1]["status"] = "FAILED"
        state["status"] = "FAILED"; state["exception"] = str(error); state["phase"] = "training"
    state["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); _write(state_path, state)
    return state


def status(root: Path) -> dict[str, Any]:
    root = Path(root).resolve(); plan = _load_plan(root)
    workers = {}
    for representation in ("sua", "pmua"):
        path = root / f"{representation}.worker.json"
        workers[representation] = _read(path) if path.is_file() else {"status": "NOT_STARTED"}
    return {"plan_sha256": plan["sha256"], "launcher": _read(root / "launcher.json") if (root / "launcher.json").is_file() else None,
            "workers": workers}


def complete(root: Path) -> dict[str, Any]:
    """Explicit post-training seal/score/report phase; never called by ``start``."""
    root = Path(root).resolve(); plan = _load_plan(root); _verify_plan_dependencies(plan); current = status(root)
    if any(value.get("status") != "SUCCEEDED" for value in current["workers"].values()):
        raise RuntimeError("Flat complete requires both representation queues to succeed")
    _claim(root / ".flat_completion_claim", {"plan_sha256": plan["sha256"], "status": "CLAIMED"})
    state: dict[str, Any] = {"status": "RUNNING_SEAL", "plan_sha256": plan["sha256"],
                             "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "automatic_retry": False}
    _write(root / "completion.json", state)
    try:
        import torch
        torch.set_num_threads(2)
        from . import finalize as flat_finalize  # intentionally unavailable until the finalizer worker lands
        selections = {task["cell"]: root / task["run"] / "selection.json" for task in plan["training_tasks"]}
        seal = flat_finalize.seal_selection(root / "selection_seal", prepared_receipt=Path(plan["prepared_receipt"]["path"]),
                                            plan_path=root / "plan.json", selections=selections,
                                            base_final_receipt=Path(plan["base_final_receipt"]["path"]),
                                            encoder_checkpoints={rep: Path(row["path"]) for rep, row in plan["encoder_checkpoints"].items()})
        seal_path = Path(seal.get("path", root / "selection_seal" / "selection_seal.json")) if isinstance(seal, Mapping) else root / "selection_seal" / "selection_seal.json"
        state.update(status="RUNNING_FINAL", seal=jsonable(seal)); _write(root / "completion.json", state)
        final = flat_finalize.score_final(seal_path, root / "final", device="cuda")
        state.update(status="RUNNING_REPORT", final=jsonable(final)); _write(root / "completion.json", state)
        from . import report as flat_report
        report = flat_report.build_report(Path(plan["base_final_receipt"]["path"]), root / "final" / "final_score_receipt.json", root / "report")
        state.update(status="SUCCEEDED", report=jsonable(report))
    except Exception as error:
        state.update(status="FAILED", exception=str(error), failed_phase=state["status"])
    state["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); _write(root / "completion.json", state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan"); plan.add_argument("--root", required=True, type=Path); plan.add_argument("--cache", type=Path, default=DEFAULT_CACHE); plan.add_argument("--base-campaign", type=Path, default=DEFAULT_BASE_CAMPAIGN); plan.add_argument("--seed42-only", action="store_true")
    launch = sub.add_parser("start"); launch.add_argument("--root", required=True, type=Path); launch.add_argument("--gpus", nargs=2, type=int, default=(0, 1))
    check = sub.add_parser("status"); check.add_argument("--root", required=True, type=Path)
    work = sub.add_parser("_worker"); work.add_argument("--root", required=True, type=Path); work.add_argument("--representation", choices=("sua", "pmua"), required=True); work.add_argument("--gpu", type=int, required=True)
    finish = sub.add_parser("complete"); finish.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "plan": result = create_plan(args.root, cache=args.cache, base_campaign=args.base_campaign, seed42_only=args.seed42_only)
    elif args.command == "start": result = start(args.root, gpus=tuple(args.gpus))
    elif args.command == "status": result = status(args.root)
    elif args.command == "_worker": result = worker(args.root, representation=args.representation, gpu=args.gpu)
    else: result = complete(args.root)
    print(json.dumps({"status": result.get("status"), "plan_sha256": result.get("sha256", result.get("plan_sha256"))}, sort_keys=True))
    if result.get("status") == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
