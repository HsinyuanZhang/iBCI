#!/usr/bin/env python3
"""Deprecated builder for the frozen external GF archive.

The six v1 packages are protocol-invalid for comparison and must not be
silently regenerated as submission candidates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = Path(__file__).resolve().parent
TASKS = {"m2": ("spint-m2:e8-epoch027-76f0fb2", 7), "m1": ("spint-original-m1:e9-epoch019-052e9ea", 4), "h1": ("h1-epfilm-c1:no-readout-v1-523d3d2e", 8)}
METHODS = ("coral", "aligned_fa")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_dir(payload_root: Path, task: str, method: str) -> Path:
    choices = (payload_root / f"{task}_{method}_v1", payload_root / f"{task}_{method}")
    matches = [candidate for candidate in choices if (candidate / "payload.npz").is_file() and (candidate / "manifest.json").is_file()]
    if len(matches) != 1:
        raise FileNotFoundError("expected one payload directory among: " + ", ".join(str(x) for x in choices))
    return matches[0]


def build_context(dest: Path, source: Path, task: str, method: str) -> dict:
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("task") != task or manifest.get("method") != method:
        raise ValueError("payload manifest task/method mismatch for " + str(source))
    if int(manifest.get("max_batch", -1)) != TASKS[task][1]:
        raise ValueError("payload max_batch mismatch for " + str(source))
    expected = manifest.get("expected_prediction_fields", {})
    if int(expected.get("history", -1)) != 10:
        raise ValueError("payload history must be 10")
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("falcon_decoder.py", "decode.py"):
        shutil.copy2(TEMPLATE_DIR / name, dest / name)
    # The exporter normally deposits payload/manifest directly in ``dest``.
    # Never replace those immutable artifacts when source and destination match.
    if source.resolve() != dest.resolve():
        shutil.copy2(source / "payload.npz", dest / "payload.npz")
        shutil.copy2(source / "manifest.json", dest / "manifest.json")
    dockerfile = (TEMPLATE_DIR / "Dockerfile.template").read_text()
    dockerfile = dockerfile.replace("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}", "FROM " + TASKS[task][0])
    payload_digest = sha256(dest / "payload.npz")
    decoder_digest = sha256(TEMPLATE_DIR / "falcon_decoder.py")
    dockerfile += ("\nLABEL ai.eval.task=\"%s\" ai.eval.method=\"external-gf-%s\" ai.eval.payload.sha256=\"%s\" ai.eval.decoder.sha256=\"%s\" ai.eval.model.sha256=\"%s\" ai.eval.checkpoint.sha256=\"\" ai.eval.cpu_only=\"true\" "
                   "org.opencontainers.image.title=\"external-gf-%s-%s-v1\" org.opencontainers.image.description=\"Frozen CPU NumPy %s plus Wiener FALCON decoder\" org.opencontainers.image.revision=\"external-gf-runtime-v1\"\n"
                   "ENV TASK=%s BATCH_SIZE=%d PHASE=test EVALUATION_LOC=remote\n"
                   % (task, method, payload_digest, decoder_digest, payload_digest, task, method, method, task, TASKS[task][1]))
    if task == "h1":
        dockerfile += ("LABEL ibci.h1.calibration.trials=\"\" ibci.h1.checkpoint.sha256=\"\" ibci.h1.epoch.zero_based=\"\" "
                       "ibci.h1.film_state.sha256=\"\" ibci.h1.model_state.sha256=\"\" ibci.h1.package.sha256=\"\" "
                       "ibci.h1.readout=\"\" ibci.h1.selection.surface=\"\"\n")
    (dest / "Dockerfile").write_text(dockerfile)
    image = f"external-gf-{task}-{method}-v1:cpu"
    readme = "\n".join((
        f"# External GF {task.upper()} {method}", "",
        "Self-contained CPU EvalAI context. Numerical inference uses NumPy only.",
        "It contains the frozen numeric payload and no source query labels, raw NWB files, or training pickles.", "",
        "## Build", "",
        f"```bash\ndocker build -t {image} .\n```", "",
        "## Frozen archive", "",
        "`FROZEN_PROTOCOL_INVALID_FOR_COMPARISON`: this single-source, `ridge=1` package lacks the official preprocessing control and is not a fair RIFT comparison or EvalAI submission candidate.", "",
        "Keep its payload, manifest, predictions, audits, and image ID for traceability. Do not push or register this image.", "",
        "See `../READINESS.json` for the archive status and `../../fair_v2/` for the replacement plan.", "",
    ))
    (dest / "README.md").write_text(readme)
    prior_candidate = json.loads((dest / "evalai_candidate.json").read_text()) if (dest / "evalai_candidate.json").is_file() else {}
    candidate = {
        "schema": "external_gf_evalai_candidate_v1", "status": "FROZEN_PROTOCOL_INVALID_FOR_COMPARISON",
        "submission_eligible": False,
        "freeze_reason": "Original single-source ridge=1 protocol lacks the official preprocessing control, so it is not a fair RIFT comparison or submission candidate.",
        "challenge_id": 2319, "phase_id": 4599, "phase_slug": "few-shot-test-2319",
        "task": task, "method": method, "image_tag": image, "image_id": prior_candidate.get("image_id"),
        "payload_sha256": payload_digest, "manifest_sha256": sha256(dest / "manifest.json"),
        "method_name": f"External GF {task.upper()} {method}",
        "method_description": "Frozen CPU NumPy streaming decoder: session-specific neural-only %s plus source Wiener readout." % method,
        "budget_disclosure": "Frozen archive; no remote push, registration, or submission has been performed.",
        "submission_attributes": {"IsHeldOutZeroShot": False, "IsTestTimeAdaptive": False, "IsPretrained": False},
        "submission_command": None,
    }
    (dest / "evalai_candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    return {"package": str(dest), "image": image, "payload_sha256": payload_digest, "manifest_sha256": sha256(dest / "manifest.json")}


def record_image(path: Path) -> dict:
    candidate_path = path / "evalai_candidate.json"
    candidate = json.loads(candidate_path.read_text())
    try:
        image_id = subprocess.check_output(["docker", "image", "inspect", candidate["image_tag"], "--format", "{{.Id}}"], text=True).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Docker image is not built for " + candidate["image_tag"]) from exc
    candidate["image_id"] = image_id
    candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    return {"package": str(path), "image": candidate["image_tag"], "image_id": image_id}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-root", type=Path, default=ROOT / "submissions")
    parser.add_argument("--dest-root", type=Path, default=ROOT / "submissions")
    parser.add_argument("--record-images", action="store_true", help="record already-built local image IDs only")
    parser.add_argument("--rebuild-frozen-archive", action="store_true", help="explicitly regenerate the frozen archive; never restores submission eligibility")
    args = parser.parse_args()
    if not args.rebuild_frozen_archive:
        raise RuntimeError("external GF v1 packages are frozen: use the fair_v2 workflow; --rebuild-frozen-archive is archival maintenance only")
    records = []
    for task in TASKS:
        for method in METHODS:
            dest = args.dest_root / f"{task}_{method}_v1"
            records.append(build_context(dest, payload_dir(args.payload_root, task, method), task, method))
            if args.record_images:
                records.append(record_image(dest))
    print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
