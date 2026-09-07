"""V7 exact networkless Docker argv over its minimal staged context."""
from __future__ import annotations
import hashlib, os, subprocess
from pathlib import Path
from typing import Any, Callable, Sequence
from . import plan, staged_context
class DockerArgvError(RuntimeError): pass
Runner = Callable[..., subprocess.CompletedProcess[str]]
def _need(v: bool, m: str) -> None:
    if not v: raise DockerArgvError(m)
def _run(argv: Sequence[str], *, runner: Runner = subprocess.run) -> subprocess.CompletedProcess[str]:
    cmd = list(argv); _need(cmd and cmd[0] == plan.DOCKER_EXECUTABLE and all(isinstance(x, str) and x for x in cmd), "V7 malformed/nonlocal Docker argv"); _need(not {"login", "push", "pull"}.intersection(cmd[1:]), "V7 registry command prohibited"); result = runner(cmd, shell=False, check=False, capture_output=True, text=True); _need(result.returncode == 0, f"V7 Docker command failed: {result.stderr[:300]}"); return result
def build_and_validate_container(*, repo_root: Path, artifact_root: Path, runner: Runner = subprocess.run, context_repo_root: Path | None = None) -> dict[str, Any]:
    data, output = repo_root / "SPINT-main/data", artifact_root / "container_minival"; _need(data.is_dir() and not data.is_symlink() and not os.path.lexists(output), "V7 data/output admission drift"); exe = Path(plan.DOCKER_EXECUTABLE); _need(exe.is_file() and not exe.is_symlink() and os.access(exe, os.X_OK), "V7 docker unavailable")
    base = _run([plan.DOCKER_EXECUTABLE, "image", "inspect", "--format", "{{.Id}}", plan.LOCAL_DOCKER_BASE_TAG], runner=runner); _need(base.stdout.strip() == plan.LOCAL_DOCKER_BASE_ID, "V7 base drift")
    evidence = staged_context.validate_staged_context(repo_root=repo_root if context_repo_root is None else context_repo_root, artifact_root=artifact_root); context = Path(evidence["root"]); _run([plan.DOCKER_EXECUTABLE, "build", "--network=none", "--pull=false", "--file", str(context / "Dockerfile"), "--tag", plan.IMAGE_TAG, str(context)], runner=runner)
    image = _run([plan.DOCKER_EXECUTABLE, "image", "inspect", "--format", "{{.Id}}", plan.IMAGE_TAG], runner=runner); image_id = image.stdout.strip(); _need(image_id.startswith("sha256:") and len(image_id) == 71, "V7 image malformed"); output.mkdir(mode=0o755); pred, target = output / "prediction.pkl", output / "ground_truth.pkl"
    run = _run([plan.DOCKER_EXECUTABLE, "run", "--network=none", "--pull=never", "--rm", "--env", "EVAL_DATA_PATH=/input", "--env", "PREDICTION_PATH_LOCAL=/output/prediction.pkl", "--env", "GT_PATH=/output/ground_truth.pkl", "--env", "CUDA_VISIBLE_DEVICES=", "--volume", f"{data}:/input:ro", "--volume", f"{output}:/output:rw", plan.IMAGE_TAG, "--evaluation", "local", "--split", "m2", "--phase", "minival", "--batch-size", "7"], runner=runner); _need(pred.is_file() and target.is_file(), "V7 outputs missing")
    return {"image_id": image_id, "image_tag": plan.IMAGE_TAG, "base_image_id": plan.LOCAL_DOCKER_BASE_ID, "network_disabled": True, "pull": False, "container_removed": True, "command": ["--evaluation", "local", "--split", "m2", "--phase", "minival", "--batch-size", "7"], "stdout_sha256": hashlib.sha256(run.stdout.encode()).hexdigest(), "prediction_path": str(pred), "prediction_sha256": hashlib.sha256(pred.read_bytes()).hexdigest(), "target_path": str(target), "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
