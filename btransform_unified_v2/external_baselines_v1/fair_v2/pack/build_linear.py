#!/usr/bin/env python3
"""Build and host-verify one fair_v2 linear EvalAI context. Does not push or submit."""
from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

PACK = Path(__file__).resolve().parent
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from common import (
    CHALLENGE_ID,
    LOCAL_SCORES,
    PHASE_ID,
    PHASE_SLUG,
    SELECTION_BUDGET,
    TASKS,
    TEAM,
    TEAM_ID,
    image_tag,
    method_description,
    method_name,
    package_name,
    sha256_file,
)
from falcon_decoder import FairV2LinearFalconDecoder


def _write_dockerfile(dest: Path, task: str, method: str, payload_sha: str, decoder_sha: str) -> None:
    text = (PACK / "Dockerfile.template").read_text()
    text = text.replace("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}", "FROM " + TASKS[task]["base_image"])
    image = image_tag(task, method)
    text += (
        f'\nLABEL ai.eval.task="{task}" ai.eval.method="fair-v2-{method}" '
        f'ai.eval.payload.sha256="{payload_sha}" ai.eval.decoder.sha256="{decoder_sha}" '
        f'ai.eval.model.sha256="{payload_sha}" ai.eval.checkpoint.sha256="" ai.eval.cpu_only="true" '
        f'org.opencontainers.image.title="{image}" '
        f'org.opencontainers.image.description="Fair v2 CPU NumPy {method} Wiener-filter linear decoder" '
        f'org.opencontainers.image.revision="fair-v2-linear"\n'
        f"ENV TASK={task} BATCH_SIZE={TASKS[task]['max_batch']} PHASE=test EVALUATION_LOC=remote\n"
    )
    if task == "h1":
        text += (
            'LABEL ibci.h1.calibration.trials="" ibci.h1.checkpoint.sha256="" '
            'ibci.h1.epoch.zero_based="" ibci.h1.film_state.sha256="" '
            'ibci.h1.model_state.sha256="" ibci.h1.package.sha256="" '
            'ibci.h1.readout="" ibci.h1.selection.surface=""\n'
        )
    (dest / "Dockerfile").write_text(text)


def _write_readme(dest: Path, task: str, method: str) -> None:
    image = image_tag(task, method)
    (dest / "README.md").write_text(
        "\n".join(
            (
                f"# Fair v2 {task.upper()} {method}",
                "",
                "Self-contained CPU EvalAI context. Numerical inference uses NumPy only.",
                "The payload is numeric: no source query labels, raw NWB files, or training pickles.",
                "",
                "## Protocol",
                "",
                method_description(task, method, json.loads((dest / "manifest.json").read_text()).get("selected")),
                "",
                "## Build",
                "",
                f"```bash\ndocker build -t {image} .\n```",
                "",
                "Do not docker push or EvalAI-submit from this directory without the parent queue.",
                "",
            )
        )
        + "\n"
    )


def evaluator_constructor(dest: Path) -> dict:
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.evaluator import FalconEvaluator

    manifest = json.loads((dest / "manifest.json").read_text())
    task = manifest["task"]
    config = FalconConfig(task=getattr(FalconTask, task))
    decoder = FairV2LinearFalconDecoder(
        config,
        str(dest / "payload.npz"),
        manifest_path=str(dest / "manifest.json"),
        batch_size=int(manifest["max_batch"]),
    )
    kwargs = {"eval_remote": False, "split": task}
    if "dataloader_workers" in inspect.signature(FalconEvaluator).parameters:
        kwargs["dataloader_workers"] = 0
    evaluator = FalconEvaluator(**kwargs)
    return {
        "official_evaluator_constructor": True,
        "decoder_class": type(decoder).__name__,
        "evaluator_class": type(evaluator).__name__,
        "batch_size": int(manifest["max_batch"]),
        "n_channels": int(config.n_channels),
        "out_dim": int(config.out_dim),
    }


def all_sessions_partial_batches(dest: Path) -> dict:
    from falcon_challenge.config import FalconConfig, FalconTask

    manifest = json.loads((dest / "manifest.json").read_text())
    task = manifest["task"]
    config = FalconConfig(task=getattr(FalconTask, task))
    max_batch = int(manifest["max_batch"])
    stems = [row["raw_basename"] for row in manifest["sessions"].values()]
    decoder = FairV2LinearFalconDecoder(
        config,
        str(dest / "payload.npz"),
        manifest_path=str(dest / "manifest.json"),
        batch_size=max_batch,
    )
    chunks = []
    channels = TASKS[task]["channels"]
    for offset in range(0, len(stems), max_batch):
        chunk = stems[offset : offset + max_batch]
        decoder.reset([Path(name) for name in chunk])
        pred = decoder.predict(np.zeros((len(chunk), channels), dtype=np.float32))
        if pred.shape != (len(chunk), TASKS[task]["outputs"]) or not np.isfinite(pred).all():
            raise RuntimeError(f"partial-batch predict failed for {chunk}")
        chunks.append([len(chunk), len(stems)])
        decoder.set_batch_size(max_batch)
    if len(stems) != TASKS[task]["roster"]:
        raise RuntimeError(f"roster {len(stems)} != {TASKS[task]['roster']}")
    return {
        "all_sessions_and_partial_batches": True,
        "roster": len(stems),
        "chunks": chunks,
        "dtype": "float32",
        "finite": True,
    }


def build_image(dest: Path, task: str, method: str) -> dict:
    image = image_tag(task, method)
    subprocess.run(
        ["docker", "build", "-t", image, str(dest)],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "DOCKER_BUILDKIT": "0"},
    )
    inspect = json.loads(
        subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{json .}}"], text=True)
    )
    return {"image_tag": image, "image_id": inspect["Id"], "image_size": int(inspect["Size"])}


def image_files_match(dest: Path, image: str) -> dict:
    names = ("payload.npz", "manifest.json", "falcon_decoder.py", "decode.py")
    mapping = {
        "payload.npz": "/data/payload.npz",
        "manifest.json": "/data/manifest.json",
        "falcon_decoder.py": "/workspace/falcon_decoder.py",
        "decode.py": "/workspace/decode.py",
    }
    rows = {}
    for name in names:
        host = sha256_file(dest / name)
        inside = subprocess.check_output(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "-e",
                "CUDA_VISIBLE_DEVICES=",
                image,
                "python",
                "-c",
                (
                    "import hashlib,sys; p=sys.argv[1]; h=hashlib.sha256(); "
                    "f=open(p,'rb');\n"
                    "import itertools\n"
                    "h.update(f.read()); print(h.hexdigest())"
                ),
                mapping[name],
            ],
            text=True,
        ).strip()
        rows[name] = {"host": host, "image": inside, "match": host == inside}
        if host != inside:
            raise RuntimeError(f"image embedded {name} hash mismatch")
    return {"image_embedded_files_match_context": True, "files": rows}


def write_candidate(dest: Path, task: str, method: str, image_info: dict, checks: dict) -> dict:
    manifest = json.loads((dest / "manifest.json").read_text())
    selected = manifest.get("selected")
    payload_sha = sha256_file(dest / "payload.npz")
    image = image_info["image_tag"]
    command = (
        f"CUDA_VISIBLE_DEVICES='' /usr/bin/python3 "
        f"{PACK / 'submit.py'} "
        f"--manifest {dest / 'evalai_candidate.json'} "
        f"--execute --confirm-image-id {image_info['image_id']} "
        f"--confirm-payload-sha256 {payload_sha}"
    )
    eligible = all(checks.values()) if isinstance(list(checks.values())[0], bool) else all(
        bool(v) for v in checks.values() if isinstance(v, bool)
    )
    candidate = {
        "schema": "fair_v2_evalai_candidate_v1",
        "status": "PACKED_HOST_AND_CONTAINER_VERIFIED" if eligible else "PACKED_INELIGIBLE",
        "submission_eligible": bool(eligible),
        "challenge_id": CHALLENGE_ID,
        "phase_id": PHASE_ID,
        "phase_slug": PHASE_SLUG,
        "evalai_team": TEAM,
        "evalai_team_id": TEAM_ID,
        "task": task,
        "method": method,
        "image_tag": image,
        "image_id": image_info["image_id"],
        "image_size": image_info.get("image_size"),
        "payload_sha256": payload_sha,
        "manifest_sha256": sha256_file(dest / "manifest.json"),
        "method_name": method_name(task, method),
        "method_description": method_description(task, method, selected),
        "selection_budget_disclosure": SELECTION_BUDGET,
        "local_public_calibration_scores": LOCAL_SCORES[(task, method)],
        "selected_hyperparameters": selected,
        "budget_disclosure": (
            "Linear arm used source-only selection. Local scores are public-calibration "
            "diagnostics, not official held-out results. No remote push or EvalAI "
            "submission has been performed for this package."
        ),
        "submission_attributes": {
            "IsHeldOutZeroShot": False,
            "IsTestTimeAdaptive": False,
            "IsPretrained": False,
        },
        "submission_command": command,
        "remote_push_performed": False,
        "evalai_registration_performed": False,
        "evalai_submission_performed": False,
        "checks": checks,
    }
    (dest / "evalai_candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    return candidate


def materialize_context(dest: Path, task: str, method: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("falcon_decoder.py", "decode.py"):
        shutil.copy2(PACK / name, dest / name)
    if not (dest / "payload.npz").is_file() or not (dest / "manifest.json").is_file():
        raise FileNotFoundError("export payload/manifest before build: " + str(dest))
    payload_sha = sha256_file(dest / "payload.npz")
    decoder_sha = sha256_file(PACK / "falcon_decoder.py")
    _write_dockerfile(dest, task, method, payload_sha, decoder_sha)
    _write_readme(dest, task, method)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("m1", "m2", "h1"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--skip-docker", action="store_true")
    args = parser.parse_args()
    materialize_context(args.dest, args.task, args.method)
    ctor = evaluator_constructor(args.dest)
    roster = all_sessions_partial_batches(args.dest)
    checks = {
        "official_evaluator_constructor": ctor["official_evaluator_constructor"],
        "all_sessions_and_partial_batches": roster["all_sessions_and_partial_batches"],
    }
    image_info = {"image_tag": image_tag(args.task, args.method), "image_id": None}
    if not args.skip_docker:
        image_info = build_image(args.dest, args.task, args.method)
        match = image_files_match(args.dest, image_info["image_tag"])
        checks["image_embedded_files_match_context"] = match["image_embedded_files_match_context"]
        (args.dest / "container_smoke.json").write_text(
            json.dumps(
                {
                    "schema": "fair_v2_container_smoke_v1",
                    "image": image_info["image_tag"],
                    "image_id": image_info["image_id"],
                    "roster": roster["roster"],
                    "chunks": roster["chunks"],
                    "falcon_evaluator_constructed": True,
                    "dtype": "float32",
                    "finite": True,
                    "cuda_visible_devices": "",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    candidate = write_candidate(args.dest, args.task, args.method, image_info, checks)
    print(json.dumps({"package": str(args.dest), "candidate": candidate, "ctor": ctor, "roster": roster}, indent=2))


if __name__ == "__main__":
    main()
