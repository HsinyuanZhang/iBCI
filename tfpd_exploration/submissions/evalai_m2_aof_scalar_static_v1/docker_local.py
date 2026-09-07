"""Offline Docker admission helpers for a separately reviewed local build."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib


class DockerAdmissionError(RuntimeError):
    pass


BASE_IMAGE_TAG = "spint-m2:e8-epoch027-76f0fb2"
BASE_IMAGE_ID = "sha256:b179efcba8e36202aec688b3104397919576e30d4344c309febcf692d4d7aaf8"


def verify_present_base_image(client: Any, reviewed_image_id: str = BASE_IMAGE_ID) -> dict[str, str]:
    """Require an exact already-local base image; this function never pulls."""
    if reviewed_image_id != BASE_IMAGE_ID:
        raise DockerAdmissionError("AOF-S Docker base ID is not the frozen workorder literal")
    image = client.images.get(BASE_IMAGE_TAG)
    image_id = str(image.attrs.get("Id", ""))
    if image_id != reviewed_image_id:
        raise DockerAdmissionError("local Docker base image ID drift")
    return {"base_image_tag": BASE_IMAGE_TAG, "base_image_id": image_id, "pull": "false", "network": "none"}


def build_network_none(
    client: Any,
    package_root: Path,
    reviewed_base_image_id: str,
    tag: str,
    *,
    dockerfile: str = "Dockerfile",
) -> dict[str, str]:
    """Build locally with pull/network disabled; caller must never push/register."""
    evidence = verify_present_base_image(client, reviewed_base_image_id)
    image, logs = client.images.build(path=str(package_root), dockerfile=dockerfile, tag=tag, pull=False, network_mode="none")
    return {**evidence, "image_id": str(image.attrs.get("Id", "")), "image_tag": tag, "dockerfile": dockerfile, "log_entries": str(len(list(logs))), "pushed": "false", "network_submission": "false"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_container_minival(
    client: Any,
    *,
    built_image: dict[str, str],
    repo_root: Path,
    artifact_root: Path,
) -> dict[str, object]:
    """Run the just-built image against local minival with Docker networking off.

    The image is never pulled, pushed, registered, or logged into.  The only
    mounted input is the host's local M2 data tree read-only; results are
    written into a newly created artifact subdirectory and immediately
    rehashed before the route may publish its ``build.json`` receipt.
    """
    required = {"base_image_id", "image_id", "image_tag", "pull", "network"}
    if set(built_image) < required:
        raise DockerAdmissionError("built-image evidence missing container admission keys")
    if (built_image["base_image_id"] != BASE_IMAGE_ID or built_image["pull"] != "false"
            or built_image["network"] != "none" or not built_image["image_id"]):
        raise DockerAdmissionError("built-image evidence drift before container validation")
    data_root = repo_root / "SPINT-main" / "data"
    if not data_root.is_dir() or data_root.is_symlink():
        raise DockerAdmissionError("local minival source data root unavailable/symlinked")
    output_root = artifact_root / "container_minival"
    if output_root.exists() or output_root.is_symlink():
        raise DockerAdmissionError("container minival output root is not fresh")
    output_root.mkdir(mode=0o755)
    try:
        logs = client.containers.run(
            built_image["image_id"],
            command=["--evaluation", "local", "--split", "m2", "--phase", "minival", "--batch-size", "7"],
            network_disabled=True,
            detach=False,
            remove=True,
            environment={
                "EVAL_DATA_PATH": "/input",
                "PREDICTION_PATH_LOCAL": "/output/prediction.pkl",
                "GT_PATH": "/output/ground_truth.pkl",
                "CUDA_VISIBLE_DEVICES": "",
            },
            volumes={
                str(data_root): {"bind": "/input", "mode": "ro"},
                str(output_root): {"bind": "/output", "mode": "rw"},
            },
        )
    except Exception as error:
        raise DockerAdmissionError(f"offline AOF-S container minival failed: {type(error).__name__}: {str(error)[:300]}") from error
    prediction = output_root / "prediction.pkl"
    target = output_root / "ground_truth.pkl"
    if not prediction.is_file() or prediction.is_symlink() or not target.is_file() or target.is_symlink():
        raise DockerAdmissionError("offline AOF-S container minival did not publish both local artifacts")
    return {
        "image_id": built_image["image_id"],
        "image_tag": built_image["image_tag"],
        "base_image_id": built_image["base_image_id"],
        "network_disabled": True,
        "pull": False,
        "container_removed": True,
        "command": ["--evaluation", "local", "--split", "m2", "--phase", "minival", "--batch-size", "7"],
        "stdout_sha256": hashlib.sha256(bytes(logs)).hexdigest() if isinstance(logs, bytes) else hashlib.sha256(str(logs).encode()).hexdigest(),
        "prediction_path": str(prediction),
        "prediction_sha256": _sha256_file(prediction),
        "target_path": str(target),
        "target_sha256": _sha256_file(target),
    }
