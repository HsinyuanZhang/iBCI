#!/usr/bin/env python3
"""Build one fair_v2 static-RIFT EvalAI context. Does not push or submit."""
from __future__ import annotations

import argparse
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
    sha256_file,
)


def materialize(dest: Path, task: str, method: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PACK / "static_falcon_decoder.py", dest / "static_falcon_decoder.py")
    shutil.copy2(PACK / "static_decode.py", dest / "static_decode.py")
    shutil.copy2(PACK / ".dockerignore.static", dest / ".dockerignore")
    payload_sha = sha256_file(dest / "artifacts" / "decoder.pkl")
    text = (PACK / "Dockerfile.static.template").read_text()
    text = text.replace("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}", "FROM " + TASKS[task]["base_image"])
    image = image_tag(task, method)
    text += (
        f'\nLABEL ai.eval.task="{task}" ai.eval.method="fair-v2-{method}" '
        f'ai.eval.payload.sha256="{payload_sha}" ai.eval.cpu_only="true" '
        f'ai.eval.architecture="RIFT" ai.eval.backend="cached" ai.eval.identity="static" '
        f'org.opencontainers.image.title="{image}"\n'
        f"ENV TASK={task} BATCH_SIZE={TASKS[task]['max_batch']} PHASE=test EVALUATION_LOC=remote\n"
    )
    (dest / "Dockerfile").write_text(text)
    (dest / "README.md").write_text(
        f"# Fair v2 {task.upper()} {method}\n\n"
        "Same-capacity static-RIFT CPU image. Network weights are the sealed "
        "frozen EMA checkpoint; only the pre-local_conv neural frontend changes. "
        "No WF smoothing. Payload has numeric frontend maps and EMA weights only.\n\n"
        f"```bash\ndocker build -t {image} .\n```\n"
    )


def evaluator_and_roster(dest: Path) -> dict:
    sys.path.insert(0, str(dest / "artifacts" / "pkg"))
    sys.path.insert(0, str(dest))
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.evaluator import FalconEvaluator
    from static_falcon_decoder import FairV2StaticRiftFalconDecoder

    manifest = json.loads((dest / "manifest.json").read_text())
    task = manifest["task"]
    config = FalconConfig(task=getattr(FalconTask, task))
    decoder = FairV2StaticRiftFalconDecoder(
        config, str(dest / "artifacts" / "decoder.pkl"), batch_size=int(manifest["max_batch"])
    )
    FalconEvaluator(eval_remote=False, split=task, dataloader_workers=0)
    stems = [row["raw_basename"] for row in manifest["sessions"].values()]
    max_batch = int(manifest["max_batch"])
    chunks = []
    for offset in range(0, len(stems), max_batch):
        chunk = stems[offset : offset + max_batch]
        decoder.reset([Path(name) for name in chunk])
        pred = decoder.predict(np.zeros((len(chunk), TASKS[task]["channels"]), dtype=np.float32))
        if pred.shape != (len(chunk), TASKS[task]["outputs"]) or not np.isfinite(pred).all():
            raise RuntimeError("static partial-batch predict failed")
        chunks.append([len(chunk), len(stems)])
        decoder.set_batch_size(max_batch)
    return {
        "official_evaluator_constructor": True,
        "all_sessions_and_partial_batches": True,
        "roster": len(stems),
        "chunks": chunks,
    }


def build_image(dest: Path, task: str, method: str) -> dict:
    image = image_tag(task, method)
    subprocess.run(
        ["docker", "build", "--build-arg", f"PAYLOAD_SHA256={sha256_file(dest / 'artifacts' / 'decoder.pkl')}", "-t", image, str(dest)],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "DOCKER_BUILDKIT": "0"},
    )
    inspect = json.loads(
        subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{json .}}"], text=True)
    )
    return {"image_tag": image, "image_id": inspect["Id"], "image_size": int(inspect["Size"])}


def image_files_match(dest: Path, image: str) -> dict:
    pairs = {
        "artifacts/decoder.pkl": "/data/decoder.pkl",
        "static_falcon_decoder.py": "/workspace/static_falcon_decoder.py",
        "static_decode.py": "/workspace/decode.py",
    }
    rows = {}
    for host_name, inside_path in pairs.items():
        host = sha256_file(dest / host_name)
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
                "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())",
                inside_path,
            ],
            text=True,
        ).strip()
        rows[host_name] = {"host": host, "image": inside, "match": host == inside}
        if host != inside:
            raise RuntimeError(f"image embedded {host_name} hash mismatch")
    return {"image_embedded_files_match_context": True, "files": rows}


def write_candidate(dest: Path, task: str, method: str, image_info: dict, checks: dict) -> dict:
    payload_sha = sha256_file(dest / "artifacts" / "decoder.pkl")
    command = (
        f"CUDA_VISIBLE_DEVICES='' /usr/bin/python3 {PACK / 'submit.py'} "
        f"--manifest {dest / 'evalai_candidate.json'} --execute "
        f"--confirm-image-id {image_info['image_id']} --confirm-payload-sha256 {payload_sha}"
    )
    eligible = all(bool(v) for v in checks.values() if isinstance(v, bool))
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
        "image_tag": image_info["image_tag"],
        "image_id": image_info["image_id"],
        "image_size": image_info.get("image_size"),
        "payload_sha256": payload_sha,
        "manifest_sha256": sha256_file(dest / "manifest.json"),
        "method_name": method_name(task, method),
        "method_description": method_description(task, method),
        "selection_budget_disclosure": SELECTION_BUDGET,
        "local_public_calibration_scores": LOCAL_SCORES[(task, method)],
        "selected_hyperparameters": {
            "frontend": "diag_z" if method.endswith("diag_z") else "coral",
            "coral": {"shrinkage": 0.1, "ridge": 0.001, "kind": "fixed_in_advance_diagonal"},
            "checkpoint_sha256": json.loads((dest / "manifest.json").read_text())["checkpoint_sha256"],
        },
        "budget_disclosure": (
            "Static frontend settings were fixed in advance for CORAL "
            "(diagonal shrinkage 0.1, ridge 0.001), not source-selected. "
            "Existing official RIFT candidates used public calibration query "
            "labels for epoch pick; this package uses a sealed fixed-final EMA "
            "checkpoint and does not replace those official candidates. "
            "No remote push or EvalAI submission has been performed."
        ),
        "submission_attributes": {
            "IsHeldOutZeroShot": False,
            "IsTestTimeAdaptive": False,
            "IsPretrained": False,
        },
        "submission_command": command,
        "state_path": str(dest / "artifacts" / "evalai_push_state.json"),
        "remote_push_performed": False,
        "evalai_registration_performed": False,
        "evalai_submission_performed": False,
        "checks": checks,
    }
    (dest / "evalai_candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("m1", "m2", "h1"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--skip-docker", action="store_true")
    args = parser.parse_args()
    materialize(args.dest, args.task, args.method)
    roster = evaluator_and_roster(args.dest)
    checks = {
        "official_evaluator_constructor": roster["official_evaluator_constructor"],
        "all_sessions_and_partial_batches": roster["all_sessions_and_partial_batches"],
    }
    image_info = {"image_tag": image_tag(args.task, args.method), "image_id": None}
    if not args.skip_docker:
        image_info = build_image(args.dest, args.task, args.method)
        match = image_files_match(args.dest, image_info["image_tag"])
        checks["image_embedded_files_match_context"] = match["image_embedded_files_match_context"]
        (args.dest / "container_smoke.json").write_text(
            json.dumps({"schema": "fair_v2_container_smoke_v1", **image_info, **roster, "cuda_visible_devices": ""}, indent=2)
            + "\n"
        )
    print(json.dumps(write_candidate(args.dest, args.task, args.method, image_info, checks), indent=2))


if __name__ == "__main__":
    main()
