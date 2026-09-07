"""Route-owned Docker CLI argv for V6's reviewed repository-root context."""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

from . import plan
from . import staged_context


class DockerArgvError(RuntimeError):
    pass


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise DockerArgvError(message)


def _run(argv: Sequence[str], *, runner: Runner = subprocess.run) -> subprocess.CompletedProcess[str]:
    command = list(argv)
    _need(command and command[0] == plan.DOCKER_EXECUTABLE, "AOF-S V6 requires exact local docker executable")
    _need(all(isinstance(item, str) and item for item in command), "AOF-S V6 malformed Docker argv")
    _need(not {"login", "push", "pull"}.intersection(command[1:]), "AOF-S V6 registry command prohibited")
    result = runner(command, shell=False, check=False, capture_output=True, text=True)
    _need(result.returncode == 0, f"AOF-S V6 Docker command failed: {result.stderr[:300]}")
    return result


def build_and_validate_container(*, repo_root: Path, artifact_root: Path, runner: Runner = subprocess.run,
                                 context_repo_root: Path | None = None) -> dict[str, Any]:
    data_root, output = repo_root / "SPINT-main/data", artifact_root / "container_minival"
    _need(data_root.is_dir() and not data_root.is_symlink(), "AOF-S V6 local minival data root invalid")
    _need(not os.path.lexists(output), "AOF-S V6 container minival output must be fresh")
    executable = Path(plan.DOCKER_EXECUTABLE)
    _need(executable.is_file() and not executable.is_symlink() and os.access(executable, os.X_OK), "AOF-S V6 local docker executable unavailable")
    base = _run([plan.DOCKER_EXECUTABLE, "image", "inspect", "--format", "{{.Id}}", plan.LOCAL_DOCKER_BASE_TAG], runner=runner)
    _need(base.stdout.strip() == plan.LOCAL_DOCKER_BASE_ID, "AOF-S V6 local base image identity drift")
    source_root = repo_root if context_repo_root is None else context_repo_root
    context_evidence = staged_context.validate_staged_context(repo_root=source_root, artifact_root=artifact_root)
    context = Path(context_evidence["root"])
    dockerfile = context / "Dockerfile"
    _run([plan.DOCKER_EXECUTABLE, "build", "--network=none", "--pull=false", "--file", str(dockerfile), "--tag", plan.IMAGE_TAG, str(context)], runner=runner)
    image = _run([plan.DOCKER_EXECUTABLE, "image", "inspect", "--format", "{{.Id}}", plan.IMAGE_TAG], runner=runner)
    image_id = image.stdout.strip()
    _need(image_id.startswith("sha256:") and len(image_id) == 71, "AOF-S V6 built image identity malformed")
    output.mkdir(mode=0o755)
    prediction, target = output / "prediction.pkl", output / "ground_truth.pkl"
    run = _run([plan.DOCKER_EXECUTABLE, "run", "--network=none", "--pull=never", "--rm",
                "--env", "EVAL_DATA_PATH=/input", "--env", "PREDICTION_PATH_LOCAL=/output/prediction.pkl",
                "--env", "GT_PATH=/output/ground_truth.pkl", "--env", "CUDA_VISIBLE_DEVICES=",
                "--volume", f"{data_root}:/input:ro", "--volume", f"{output}:/output:rw", plan.IMAGE_TAG,
                "--evaluation", "local", "--split", "m2", "--phase", "minival", "--batch-size", "7"], runner=runner)
    _need(prediction.is_file() and target.is_file(), "AOF-S V6 container minival outputs missing")
    return {"image_id": image_id, "image_tag": plan.IMAGE_TAG, "base_image_id": plan.LOCAL_DOCKER_BASE_ID,
            "network_disabled": True, "pull": False, "container_removed": True,
            "command": ["--evaluation", "local", "--split", "m2", "--phase", "minival", "--batch-size", "7"],
            "stdout_sha256": hashlib.sha256(run.stdout.encode()).hexdigest(), "prediction_path": str(prediction),
            "prediction_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(), "target_path": str(target),
            "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
