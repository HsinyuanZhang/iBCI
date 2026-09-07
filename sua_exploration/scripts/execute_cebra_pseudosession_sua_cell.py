#!/usr/bin/env python3
"""Execute one fresh CEBRA pseudo-session source-training cell."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


SUA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUA_ROOT.parent
sys.path.insert(0, str(SUA_ROOT))
sys.path.insert(0, str(SUA_ROOT / "scripts"))

from preflight_cebra_pseudosession_sua import write_immutable_pair  # noqa: E402
from train_cebra_pseudosession_sua import (  # noqa: E402
    EXPECTED_PREFLIGHT_SHA256,
    PREFLIGHT,
    SCREEN_ID,
    load_preflight,
    validate_argv,
)


PYTHON = Path("/home/xinyuan/miniconda3/envs/spint/bin/python")
TRAINER = SUA_ROOT / "scripts" / "train_cebra_pseudosession_sua.py"
RESULT_ROOT = SUA_ROOT / "results" / SCREEN_ID
CHECKPOINT_ROOT = SUA_ROOT / "checkpoints"


class ExecuteError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecuteError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def live_bindings() -> dict[str, str]:
    paths = {
        "executor": Path(__file__).resolve(),
        "pseudo_trainer": TRAINER,
        "pseudo_datamodule": SUA_ROOT / "mc_maze" / "cebra_pseudosession.py",
        "base_trainer": SUA_ROOT / "scripts" / "train_variant_dandi688.py",
        "base_datamodule": SUA_ROOT / "mc_maze" / "multisession_datamodule.py",
        "unit_side_features": SUA_ROOT / "mc_maze" / "unit_side_features.py",
        "streaming_module": REPO_ROOT / "streaming_calibration_exp/src/models/streaming_calibration_module.py",
        "streaming_spint": REPO_ROOT / "streaming_calibration_exp/src/models/components/streaming_spint.py",
        "streaming_encoders": REPO_ROOT / "streaming_calibration_exp/src/models/components/streaming_encoders.py",
        "spint": REPO_ROOT / "streaming_calibration_exp/src/models/components/spint.py",
        "run_artifacts": REPO_ROOT / "streaming_calibration_exp/src/metrics/run_artifacts.py",
        "config": SUA_ROOT / f"configs/{SCREEN_ID}.json",
        "contract": SUA_ROOT / "docs/CEBRA_PSEUDOSESSION_SUA_CONTRACT_20260814.md",
        "constructibility_preflight": PREFLIGHT,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def paths_for(arm: str, seed: int) -> dict[str, Path]:
    name = f"{SCREEN_ID}_mix_{arm}_s{seed}"
    return {
        "run": CHECKPOINT_ROOT / name,
        "log": RESULT_ROOT / "logs" / f"mix_{arm}_s{seed}.log",
        "launch": RESULT_ROOT / f"cell_launch_mix_{arm}_s{seed}.json",
        "completion": RESULT_ROOT / f"cell_completion_mix_{arm}_s{seed}.json",
        "global_summary": SUA_ROOT / "results" / f"p3_{name}_seed{seed}.json",
    }


def train_argv(arm: str, seed: int) -> list[str]:
    name = f"{SCREEN_ID}_mix_{arm}_s{seed}"
    return [
        "--teacher_ckpt", str(SUA_ROOT / "checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"),
        "--variant", "B3S", "--side_features", arm, "--side_feature_pool_size", "30",
        "--calibration_n_trials", "30", "--chronological_calibration", "--out_name", name,
        "--data_dir", str(SUA_ROOT / "data/dandi_000688/sub-C"),
        "--cache_dir", str(SUA_ROOT / "cache/dandi688_subc_co_v1"),
        "--train_val_manifest", str(SUA_ROOT / "configs/subc_co_27_6_strict_train_val_manifest.json"),
        "--signal_view", "sua", "--task", "CO", "--split_counts", "27,6,6",
        "--max_units_exclusive", "100", "--max_epochs", "12", "--no_early_stopping",
        "--checkpoint_every_epoch", "--lr", "0.0001", "--batch_size", "32",
        "--num_workers", "4", "--seed", str(seed), "--loss_mode", "task_only",
        "--identity_mode", "calibrated", "--require_gpu", "--disable_progress_bar",
    ]


def validate_fresh(targets: dict[str, Path]) -> None:
    occupied = []
    for name, path in targets.items():
        if path.exists() or Path(str(path) + ".sha256").exists():
            occupied.append(f"{name}={path}")
    require(not occupied, "cell is not fresh: " + "; ".join(occupied))


def runtime_probe(gpu: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": str(gpu)})
    code = (
        "import json,site,sys,torch; "
        "print(json.dumps({'executable':sys.executable,'user_site':site.ENABLE_USER_SITE,"
        "'torch':torch.__version__,'torch_file':torch.__file__,'cuda_build':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available(),'count':torch.cuda.device_count(),"
        "'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    )
    process = subprocess.run([str(PYTHON), "-c", code], env=env, check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    payload = json.loads(process.stdout)
    require(payload == {
        **payload,
        "executable": str(PYTHON),
        "user_site": False,
        "cuda_available": True,
        "count": 1,
    }, "isolated runtime probe drift")
    require(str(payload["torch"]).startswith("2.5.1.post303"), "Torch version drift")
    require(payload["cuda_build"] == "11.8", "Torch CUDA build drift")
    require("RTX 3090" in str(payload["name"]), "GPU class drift")
    return payload


def plan(arm: str, seed: int, gpu: int) -> dict[str, Any]:
    require(arm in {"t4", "z4"}, "arm must be t4 or z4")
    require(seed in {42, 43, 44}, "seed drift")
    require(gpu in {0, 1}, "physical GPU must be 0 or 1")
    load_preflight()
    targets = paths_for(arm, seed)
    validate_fresh(targets)
    argv = train_argv(arm, seed)
    validate_argv(argv)
    bindings = live_bindings()
    return {
        "schema_version": 1,
        "receipt_kind": "cebra_pseudosession_sua_cell_launch",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screen_id": SCREEN_ID,
        "arm": f"mix_{arm}",
        "side_feature_group": arm,
        "seed": seed,
        "physical_gpu": gpu,
        "constructibility_preflight_path": str(PREFLIGHT.resolve()),
        "constructibility_preflight_sha256": EXPECTED_PREFLIGHT_SHA256,
        "implementation_bindings": bindings,
        "implementation_bindings_sha256": hashlib.sha256(
            json.dumps(bindings, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "command": [str(PYTHON), "-u", str(TRAINER), *argv],
        "working_directory": str(REPO_ROOT.resolve()),
        "environment": {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "",
            "CUDA_VISIBLE_DEVICES": str(gpu),
        },
        "targets": {name: str(path.resolve()) for name, path in targets.items()},
        "formal_subc_test_nwb_opened": False,
        "training_started": False,
    }


def execute(arm: str, seed: int, gpu: int) -> int:
    payload = plan(arm, seed, gpu)
    targets = paths_for(arm, seed)
    runtime = runtime_probe(gpu)
    payload["runtime"] = runtime
    payload["training_started"] = True
    payload["training_start_boundary"] = datetime.now(timezone.utc).isoformat()
    launch_sha = write_immutable_pair(targets["launch"], payload)
    # Recheck all bytes immediately before the irreversible subprocess start.
    require(payload["implementation_bindings"] == live_bindings(),
            "implementation drifted after launch receipt")
    env = os.environ.copy()
    env.update(payload["environment"])
    targets["log"].parent.mkdir(parents=True, exist_ok=True)
    with targets["log"].open("xb") as log:
        process = subprocess.run(payload["command"], cwd=payload["working_directory"],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
        log.flush()
        os.fsync(log.fileno())
    completion = {
        "schema_version": 1,
        "receipt_kind": "cebra_pseudosession_sua_cell_completion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screen_id": SCREEN_ID,
        "arm": f"mix_{arm}",
        "seed": seed,
        "launch_receipt_path": str(targets["launch"].resolve()),
        "launch_receipt_sha256": launch_sha,
        "subprocess_started": True,
        "return_code": process.returncode,
        "successful": process.returncode == 0,
        "log_path": str(targets["log"].resolve()),
        "log_sha256": sha256_file(targets["log"]),
        "run_dir_exists": targets["run"].is_dir(),
        "formal_subc_test_nwb_opened": False,
    }
    write_immutable_pair(targets["completion"], completion)
    return int(process.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("t4", "z4"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise ExecuteError("PYTHONNOUSERSITE=1 required")
    if not args.execute:
        print(json.dumps(plan(args.arm, args.seed, args.gpu), indent=2, sort_keys=True))
        return 0
    return execute(args.arm, args.seed, args.gpu)


if __name__ == "__main__":
    raise SystemExit(main())
