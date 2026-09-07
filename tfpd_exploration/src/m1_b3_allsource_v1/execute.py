"""Public-executable drivers for all-source B3 training and cached-identity packaging."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1

from . import gpu as gpu_module
from . import plan
from . import receipts


class ExecuteError(RuntimeError):
    """Fail closed for public execute."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecuteError(message)


def _bind_thread_env() -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    for name, value in v1.THREAD_ENVIRONMENT.items():
        os.environ[name] = value


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_parents(repo_root: Path, mode: str) -> None:
    result = Path(repo_root) / plan.RESULT_ROOT_RELATIVE
    result.mkdir(parents=True, exist_ok=True)
    extra = plan.parent_relative_for_stage(mode)
    if extra is not None:
        (Path(repo_root) / extra).mkdir(parents=True, exist_ok=True)


def execute(
    repo_root: Path,
    *,
    mode: str,
    arm: str,
    gpu_index: int | None,
    gpu_authorized: bool,
    pack: bool = False,
) -> tuple[dict[str, str], str | None, str | None]:
    repo_root = Path(repo_root).absolute()
    _require(os.environ.get("PYTHONNOUSERSITE") == "1", "execute requires PYTHONNOUSERSITE=1")
    plan.validate_mode_arm(mode, arm)
    _require(type(pack) is bool, "pack flag")
    receipts.refuse_sealed_roots(repo_root / plan.RESULT_ROOT_RELATIVE)
    if mode == "train":
        _require(gpu_index is not None, "train requires --gpu-index")
        gpu_info = gpu_module.require_launch(gpu_authorized, int(gpu_index), pack=pack)
        _bind_thread_env()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    else:
        gpu_info = {
            "gpu_index": None,
            "pack": False,
            "note": "package is CPU-only; CUDA_VISIBLE_DEVICES must stay empty",
        }
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        _bind_thread_env()
    _ensure_parents(repo_root, mode)
    relative = plan.stage_relative(mode, arm)
    progress_state: dict[str, object] = {"mode": mode, "arm": arm, "gpu_index": gpu_index}

    def launch_builder() -> dict[str, object]:
        return {
            "schema": "m1_b3_allsource_launch_v1",
            "mode": mode,
            "arm": arm,
            "status": "LAUNCHED",
            "gpu": gpu_info,
            "pack": pack,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "teacher_sha256": plan.TEACHER_SHA256,
            "nwb_or_checkpoint_opened": False,
            "cuda_initialized": False,
        }

    def body_publisher(artifact: Any) -> dict[str, str]:
        if mode == "train":
            return _bodies_train(
                repo_root, arm=arm, gpu_index=int(gpu_index), artifact=artifact,
                progress=progress_state,
            )
        from . import package as package_module

        return package_module.publish_package(
            repo_root, arm=arm, artifact=artifact, progress=progress_state,
        )

    def terminal_builder(shas: Mapping[str, str]) -> dict[str, object]:
        return {
            "schema": "m1_b3_allsource_terminal_v1",
            "status": "COMPLETE",
            "mode": mode,
            "arm": arm,
            "gpu_index": gpu_index,
            "pack": pack,
            "formal_benchmark_verdict": False,
            "local_20120924_is_not_official_heldout": True,
            "official_original_heldout_r2": plan.OFFICIAL_ORIGINAL_HELDOUT_R2,
            "bodies": {key: value for key, value in shas.items() if not str(key).startswith("_")},
        }

    return receipts.run_stage0(
        repo_root / plan.RESULT_ROOT_RELATIVE,
        relative=relative,
        attempt_payload={
            "schema": "m1_b3_allsource_attempt_v1",
            "cell": plan.CELL,
            "phase": plan.PHASE,
            "mode": mode,
            "arm": arm,
            "status": "ATTEMPT_RESERVED",
            "gpu_index": gpu_index,
            "pack": pack,
            "pair_spec_sha256": plan.PairSpec().sha256,
            "data_or_model_accessed": False,
        },
        launch_builder=launch_builder,
        body_publisher=body_publisher,
        terminal_builder=terminal_builder,
        progress=lambda: dict(progress_state),
    )


def _bodies_train(
    repo_root: Path,
    *,
    arm: str,
    gpu_index: int,
    artifact: Any,
    progress: dict[str, object],
) -> dict[str, str]:
    spec = plan.ARM_SPECS[arm]
    teacher = repo_root / plan.TEACHER_CHECKPOINT_RELATIVE
    _require(teacher.is_file() and not teacher.is_symlink(), "teacher checkpoint missing")
    teacher_sha = _sha_file(teacher)
    _require(teacher_sha == plan.TEACHER_SHA256, "teacher checkpoint sha256 drift")
    streaming = repo_root / "streaming_calibration_exp"
    spint_main = repo_root / "SPINT-main"
    _require((streaming / "src" / "train.py").is_file(), "hydra train entry missing")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S-%f")
    run_id = f"m1_b3_allsource_v1_{arm}_s42_e12"
    hydra_dir = streaming / "logs" / "m1_b3_allsource_v1" / arm / "runs" / (
        f"{stamp}_rid-{run_id}_fNone_s{plan.SEED}"
    )
    hydra_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = hydra_dir.parent / f"{hydra_dir.name}.launch.log"
    python = sys.executable
    command = [
        python,
        str(streaming / "src" / "train.py"),
        f"experiment={spec['experiment']}",
        f"task_name=m1_b3_allsource_v1_{arm}",
        f"run_id={run_id}",
        f"seed={plan.SEED}",
        f"hydra.run.dir={hydra_dir}",
        f"+pack_token={plan.OWN_TOKEN}",
    ]
    env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONPATH": os.pathsep.join(
            [str(repo_root), str(streaming), str(spint_main), os.environ.get("PYTHONPATH", "")]
        ),
        "HYDRA_FULL_ERROR": "1",
        **v1.THREAD_ENVIRONMENT,
    }
    progress["hydra_dir"] = str(hydra_dir)
    progress["command"] = command
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=str(streaming),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            timeout=plan.HARD_TIMEOUT_SECONDS_PER_ARM,
            check=False,
        )
    progress["returncode"] = completed.returncode
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8")[-4000:]
        raise ExecuteError(f"hydra train failed rc={completed.returncode}: {tail}")
    ckpt_dir = hydra_dir / "checkpoints" / "best_ckpt"
    epoch_ckpts = sorted(path for path in ckpt_dir.glob("epoch_*.ckpt"))
    _require(bool(epoch_ckpts), f"no epoch checkpoint under {ckpt_dir}")
    student_ckpt = epoch_ckpts[-1]
    last_ckpt = ckpt_dir / "last.ckpt"
    manifest = hydra_dir / "all_source_train_manifest.json"
    hydra_payload = {
        "schema": "m1_b3_allsource_hydra_run_v1",
        "arm": arm,
        "experiment": spec["experiment"],
        "variant": spec["variant"],
        "side_feature_group": spec["side_feature_group"],
        "freeze_decoder": spec["freeze_decoder"],
        "loss_mode": spec["loss_mode"],
        "activity_prefix_cycle": (
            list(spec["activity_prefix_cycle"])
            if spec.get("activity_prefix_cycle") is not None
            else None
        ),
        "seed": plan.SEED,
        "epochs": plan.EPOCHS,
        "adam_lr": plan.ADAM_LR,
        "hydra_output_dir": str(hydra_dir),
        "launch_log": str(log_path),
        "student_checkpoint": str(student_ckpt),
        "student_checkpoint_sha256": _sha_file(student_ckpt),
        "last_checkpoint": str(last_ckpt) if last_ckpt.is_file() else None,
        "last_checkpoint_sha256": _sha_file(last_ckpt) if last_ckpt.is_file() else None,
        "teacher_checkpoint": str(teacher),
        "teacher_sha256": teacher_sha,
        "source_manifest": str(manifest) if manifest.is_file() else None,
        "source_manifest_sha256": _sha_file(manifest) if manifest.is_file() else None,
        "command": command,
        "pack_token": plan.OWN_TOKEN,
        "formal_benchmark_verdict": False,
        "local_20120924_is_not_official_heldout": True,
    }
    shas = {
        "hydra_run.json": artifact.publish_json("hydra_run.json", hydra_payload),
    }
    if manifest.is_file():
        shas["all_source_train_manifest.json"] = artifact.publish_bytes(
            "all_source_train_manifest.json", manifest.read_bytes()
        )
    progress["student_checkpoint_sha256"] = hydra_payload["student_checkpoint_sha256"]
    return shas
