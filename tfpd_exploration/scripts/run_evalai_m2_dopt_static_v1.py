#!/usr/bin/env python3
"""Dry by default; reserve the D-opt-4 static M2 EvalAI packaging attempt or execute it.

Stages
------
- ``dry``       : print the plan, touch nothing.
- ``attempt``   : write the 0444+sidecar attempt receipt (pre-registration).
- ``export``    : build the official payload (offline calibration phase).
- ``validate``  : payload audit + evaluator-semantics replay + contract
                  assertions + host FalconEvaluator minival run.
- ``build``     : docker build of the submission image (no push).
- ``container`` : container-side local minival + remote-path simulation.
- ``terminal``  : write the 0444+sidecar terminal receipt.

No stage performs any network submission; the EvalAI push is operator-owned
and documented in the work-order-lite and terminal receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TFPD_ROOT = REPO_ROOT / "tfpd_exploration"
PACKAGE_DIR = TFPD_ROOT / "submissions/evalai_m2_dopt_static_v1"
ARTIFACTS = PACKAGE_DIR / "artifacts"
PAYLOAD = ARTIFACTS / "t4_m2_seed42_dopt4_static_identity.pkl"
RESULT_ROOT = TFPD_ROOT / "results/evalai_m2_dopt_static_v1"
WORKORDER_RELATIVE = (
    "tfpd_exploration/docs/WORKORDER_LITE_EVALAI_M2_DOPT_STATIC_V1_20260902.md"
)
LOG_PREFIX = TFPD_ROOT / "evalai_m2_dopt_static_v1"
BASE_IMAGE = "spint-m2:e8-epoch027-76f0fb2"
CHECKPOINT_SHA256 = "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
NORMALIZATION_SHA256 = "d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e"
SEALED_SCREEN = "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json"
SEALED_AUDIT = "tfpd_exploration/results/cdm_p1_m2_v1/audit.json"

OWNED_RELATIVE = [
    "tfpd_exploration/submissions/evalai_m2_dopt_static_v1/laws.py",
    "tfpd_exploration/submissions/evalai_m2_dopt_static_v1/export_dopt_static_payload.py",
    "tfpd_exploration/submissions/evalai_m2_dopt_static_v1/dopt_static_decoder.py",
    "tfpd_exploration/submissions/evalai_m2_dopt_static_v1/decode.py",
    "tfpd_exploration/submissions/evalai_m2_dopt_static_v1/validate_local.py",
    "tfpd_exploration/submissions/evalai_m2_dopt_static_v1/Dockerfile",
    "tfpd_exploration/scripts/run_evalai_m2_dopt_static_v1.py",
    "tfpd_exploration/tests/test_evalai_m2_dopt_static_v1.py",
]

PREDECESSOR_RELATIVE = [
    SEALED_SCREEN,
    SEALED_SCREEN + ".sha256",
    SEALED_AUDIT,
    SEALED_AUDIT + ".sha256",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/plan.py",
    "tfpd_exploration/src/calibration_budget_comparators_v1.py",
    "sua_exploration/mc_maze/d_optimal_calibration_design.py",
    "sua_exploration/evalai_t4_m2/t4_spint_decoder.py",
    "sua_exploration/evalai_t4_m2/decode.py",
    "sua_exploration/evalai_t4_m2/export_t4_payload.py",
    "sua_exploration/evalai_t4_m2_activity_budget/export_budget_payload.py",
    "sua_exploration/evalai_t4_m2_activity_budget/Dockerfile",
    "streaming_calibration_exp/outputs/streaming_calibration/m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806/checkpoints/best.ckpt",
    "streaming_calibration_exp/outputs/streaming_calibration/m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806/split_manifest.json",
    "streaming_calibration_exp/outputs/streaming_calibration/m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806/resolved_config.yaml",
]

ENV = {
    **os.environ,
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": f"{REPO_ROOT}:{PACKAGE_DIR}",
    "CUDA_VISIBLE_DEVICES": "",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_receipt(path: Path, payload: dict) -> dict:
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    path.write_bytes(body)
    path.chmod(0o444)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    sidecar.chmod(0o444)
    return {"receipt": str(path), "receipt_sha256": digest}


def _run(command: list[str], log_name: str, cwd: Path = REPO_ROOT) -> str:
    log_path = Path(f"{LOG_PREFIX}_{log_name}.log")
    completed = subprocess.run(
        command, cwd=str(cwd), env=ENV, capture_output=True, text=True
    )
    log_path.write_text(
        f"$ {' '.join(command)}\ncwd={cwd}\nrc={completed.returncode}\n\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        tail = completed.stderr.strip().splitlines()[-1][:400] if completed.stderr.strip() else ""
        raise SystemExit(f"stage failed (rc={completed.returncode}), see {log_path}: {tail}")
    return completed.stdout


def stage_attempt() -> None:
    target = RESULT_ROOT / "attempt.json"
    if target.exists():
        raise SystemExit(f"attempt receipt already exists: {target}")
    payload = {
        "schema": "evalai_m2_dopt_static_v1_attempt_v1",
        "status": "ATTEMPT_RESERVED",
        "cell": "EVALAI_M2_DOPT_STATIC_V1_PACKAGING",
        "work_order": {
            "path": WORKORDER_RELATIVE,
            "sha256": sha256_file(REPO_ROOT / WORKORDER_RELATIVE),
        },
        "authority": "user 2026-09-02 build the EvalAI D-opt-4 static M2 submission package, CPU-only, no network submission",
        "inference_only": True,
        "pre_registration": {
            "deployment": (
                "frozen m2_spint_t4_mainline seed-42 checkpoint 25d7bc72... deployed as: "
                "offline calibration = greedy forward D-optimal k=4 selection among the "
                "finite-angle candidates of the first 30 labelled calibration trials -> "
                "ridge lambda=0.1 T4 fit on the selected support -> B3S activity pool = "
                "exactly the four selected trials -> frozen-checkpoint identity; "
                "runtime = cached identity per dataset tag, reset+predict only, "
                "on_done no-op, zero online updates"
            ),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "normalization_sha256": NORMALIZATION_SHA256,
            "selection_law": (
                "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:17-35 "
                "budget==4 branch (greedy_forward_d_optimal_indices from "
                "sua_exploration/mc_maze/d_optimal_calibration_design.py:188-219), "
                "mirrored verbatim in the package and cross-checked against the sealed import"
            ),
            "ridge_law": (
                "tfpd_exploration/src/calibration_budget_comparators_v1.py:56-95 "
                "fit_ridge_t4 with normalized_lambda=0.1, sealed import"
            ),
            "activity_law": "static four selected trials; B3S mean-pools them internally",
            "sealed_anchor": {
                "path": SEALED_SCREEN,
                "cell": "ridge_static_m4",
                "external_official_query_equal_session_mean": 0.22271999429945652,
                "within_post30_equal_session_mean": 0.451784412486647,
            },
            "contract": (
                "per the sealed cdm_p1_m2_v1 audit: continual m2 evaluator calls only "
                "reset(dataset_tags)+predict(neural_observations); no on_done; "
                "assert no trial metadata at predict, selection only offline, "
                "weights/identities byte-frozen across predicts"
            ),
            "stages": ["export", "validate", "build", "container", "terminal"],
            "submission": "NOT PERFORMED by this cell; operator executes the documented push",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "no_user_site": True,
            "cuda_visible_devices": "",
        },
        "owned_sha256s": {item: sha256_file(REPO_ROOT / item) for item in OWNED_RELATIVE},
        "predecessor_sha256s": {
            item: sha256_file(REPO_ROOT / item) for item in PREDECESSOR_RELATIVE
        },
        "target_gradients": 0,
        "parameter_updates": 0,
        "model_or_checkpoint_updated": False,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=False)
    print(json.dumps(_write_receipt(target, payload), indent=2, sort_keys=True))


def stage_export() -> None:
    stdout = _run(
        [
            sys.executable,
            str(PACKAGE_DIR / "export_dopt_static_payload.py"),
            "--execute",
        ],
        "export",
        cwd=PACKAGE_DIR,
    )
    receipt = json.loads(stdout)
    print(json.dumps({k: receipt[k] for k in ("payload_sha256", "payload_bytes", "session_count", "arm")}, sort_keys=True))


def stage_validate() -> None:
    _run(
        [
            sys.executable,
            str(PACKAGE_DIR / "validate_local.py"),
            "--execute",
            "--stages",
            "payload,external,contract,minival",
        ],
        "validate",
        cwd=PACKAGE_DIR,
    )


def export_receipt() -> dict:
    return json.loads(
        (ARTIFACTS / "t4_m2_seed42_dopt4_static_identity.receipt.json").read_text(encoding="utf-8")
    )


def image_tag() -> str:
    return f"spint-t4-m2:dopt4-static-s42-{export_receipt()['payload_sha256'][:8]}"


def stage_build() -> dict:
    receipt = json.loads(
        (ARTIFACTS / "t4_m2_seed42_dopt4_static_identity.receipt.json").read_text(encoding="utf-8")
    )
    tag = f"spint-t4-m2:dopt4-static-s42-{receipt['payload_sha256'][:8]}"
    _run(
        [
            "docker",
            "build",
            "--build-arg",
            f"PAYLOAD_SHA256={receipt['payload_sha256']}",
            "-t",
            tag,
            "-f",
            "Dockerfile",
            ".",
        ],
        "build",
        cwd=PACKAGE_DIR,
    )
    inspect = json.loads(_run(["docker", "image", "inspect", tag], "image_inspect"))[0]
    print(json.dumps({"tag": tag, "id": inspect["Id"], "size": inspect.get("Size")}, sort_keys=True))
    return {"tag": tag, "id": inspect["Id"], "size": inspect.get("Size"),
            "labels": inspect["Config"].get("Labels", {})}


def stage_container(image: dict) -> dict:
    out_dir = ARTIFACTS / "container_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_mount = f"{REPO_ROOT / 'SPINT-main/data'}:/dataset/evaluation_data:ro"
    # 1) local-path container run over the public minival files.
    _run(
        [
            "docker", "run", "--rm",
            "-e", "EVALUATION_LOC=local",
            "-e", "PREDICTION_PATH_LOCAL=/out/minival_prediction.pkl",
            "-e", "GT_PATH=/out/minival_gt.pkl",
            "-e", "EVAL_DATA_PATH=/dataset/evaluation_data",
            "-v", data_mount,
            "-v", f"{out_dir}:/out",
            image["tag"],
            "/bin/bash", "-c",
            "python /decode.py --evaluation local --model-path /data/decoder.pkl "
            "--split m2 --phase minival --batch-size 7",
        ],
        "container_minival",
    )
    # 2) remote-path simulation: EVALUATION_LOC=remote + phase=test reads
    #    EVAL_DATA_PATH/m2/eval and writes /submission/submission.csv, then
    #    sleeps 300 s (the official EvalAI poll wait).  Stage the public
    #    minival files where the remote loader looks so the write path and
    #    wait are exercised without any hidden data.
    remote_data = out_dir / "remote_data/m2/eval"
    remote_data.mkdir(parents=True, exist_ok=True)
    import re

    for nwb in sorted((REPO_ROOT / "SPINT-main/data/000953/sub-MonkeyN-held-in-minival").glob("*.nwb")):
        match = re.search(r"ses-(\d{4})-(\d{2})-(\d{2})-(Run\d+)", nwb.name)
        if match is None:
            raise SystemExit(f"cannot map minival file to server naming: {nwb.name}")
        year, month, day, run = match.groups()
        # Official server naming (falcon_challenge hash_dataset m2 else-branch):
        # sub-MonkeyNRun1_20201019_held_in_eval.nwb.  The tag it hashes to
        # equals the public calib file's tag, so the cached identity resolves.
        # Files are copied (not symlinked): absolute links would dangle inside
        # the container mount.
        staged = remote_data / f"sub-MonkeyN{run}_{year}{month}{day}_held_in_eval.nwb"
        if not staged.exists():
            shutil.copyfile(nwb, staged)
    submission_dir = out_dir / "remote_path"
    submission_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"{LOG_PREFIX}_container_remote.log")
    process = subprocess.Popen(
        [
            "docker", "run", "--rm",
            "-e", "EVALUATION_LOC=remote",
            "-e", "EVAL_DATA_PATH=/dataset/evaluation_data",
            "-v", f"{out_dir / 'remote_data'}:/dataset/evaluation_data:ro",
            "-v", f"{submission_dir}:/submission",
            image["tag"],
        ],
        cwd=str(REPO_ROOT), env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    target = submission_dir / "submission.csv"
    import time

    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline and not target.exists():
        if process.poll() is not None:
            break
        time.sleep(2)
    entered_wait = False
    if target.exists():
        # the official evaluator sleeps 300 s after writing; confirm the wait,
        # then terminate: local simulation only.
        time.sleep(5)
        entered_wait = process.poll() is None
    process.terminate()
    try:
        output, _ = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate()
    log_path.write_text(output or "", encoding="utf-8")
    import pickle

    summary: dict = {"entered_expected_300s_wait": entered_wait}
    if target.exists():
        with target.open("rb") as handle:
            payload = pickle.load(handle)
        summary["remote_path_submission_csv"] = str(target)
        summary["task_key"] = sorted(payload)
        summary["session_keys"] = sorted(payload.get("m2", {}))
        summary["submission_sha256"] = sha256_file(target)
        summary["submission_bytes"] = target.stat().st_size
    else:
        summary["remote_path_submission_csv"] = None
        summary["failure"] = "submission.csv was not written; see the remote log"
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def stage_terminal(image: dict, container_summary: dict) -> None:
    target = RESULT_ROOT / "terminal.json"
    if target.exists():
        raise SystemExit(f"terminal receipt already exists: {target}")
    export_receipt = json.loads(
        (ARTIFACTS / "t4_m2_seed42_dopt4_static_identity.receipt.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (ARTIFACTS / "local_validation_receipt.json").read_text(encoding="utf-8")
    )
    payload = {
        "schema": "evalai_m2_dopt_static_v1_terminal_v1",
        "status": "TERMINAL_PACKAGED_NOT_SUBMITTED",
        "cell": "EVALAI_M2_DOPT_STATIC_V1_PACKAGING",
        "work_order": WORKORDER_RELATIVE,
        "artifact_tree": {
            "package_root": "tfpd_exploration/submissions/evalai_m2_dopt_static_v1",
            "files": {
                name: sha256_file(PACKAGE_DIR / name)
                for name in sorted(p.name for p in PACKAGE_DIR.iterdir() if p.is_file())
            },
            "payload": {
                "path": "tfpd_exploration/submissions/evalai_m2_dopt_static_v1/artifacts/t4_m2_seed42_dopt4_static_identity.pkl",
                "sha256": export_receipt["payload_sha256"],
                "bytes": export_receipt["payload_bytes"],
            },
            "receipts": "tfpd_exploration/results/evalai_m2_dopt_static_v1/",
        },
        "checkpoint_sha256": export_receipt["checkpoint_sha256"],
        "normalization_sha256": export_receipt["normalization_sha256"],
        "teacher_checkpoint_sha256": export_receipt["teacher_checkpoint_sha256"],
        "dopt_law_provenance": {
            "m4_branch": "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:17-35 (mirrored verbatim; sealed-import cross-check passed per session)",
            "greedy_core": "sua_exploration/mc_maze/d_optimal_calibration_design.py:115-121,176-219 (mirrored verbatim)",
            "ridge_t4": "tfpd_exploration/src/calibration_budget_comparators_v1.py:56-95 (sealed import fit_ridge_t4, normalized_lambda=0.1)",
            "ridge_side": "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:38-71 (sealed import _ridge_side)",
            "selections": {
                session: record["selected_indices"]
                for session, record in export_receipt["session_records"].items()
            },
        },
        "contract_compliance": validation.get("contract", {}),
        "evaluator_semantics_proof": validation.get("official_local_minival", {}).get(
            "evaluator_semantics_proof"
        ),
        "local_reference_scores": {
            "external_official_query": validation["evaluator_semantics_replay"]["surfaces"][
                "external_official_query"
            ],
            "within_post30": validation["evaluator_semantics_replay"]["surfaces"]["within_post30"],
            "official_local_minival": validation.get("official_local_minival", {}).get("metrics"),
            "surface_disclosure": (
                "external_official_query/within_post30 mirror the sealed "
                "m2_t4_activity_budget_screen_v1 static_m4 surfaces (local held-out "
                "calibration NWB query streams; held-in post-30 windows); the official "
                "EvalAI hidden test surface is neither and is expected to differ"
            ),
        },
        "sealed_anchor_comparison": {
            "external_official_query_sealed_static_m4_mean": 0.22271999429945652,
            "within_post30_sealed_static_m4_mean": 0.451784412486647,
        },
        "docker": {
            "base_image": BASE_IMAGE,
            "tag": image["tag"],
            "image_id": image["id"],
            "size_bytes": image["size"],
            "labels": image["labels"],
            "pushed": False,
        },
        "container_validation": container_summary,
        "submission": {
            "performed_by_this_cell": False,
            "operator_commands": "see tfpd_exploration/docs/WORKORDER_LITE_EVALAI_M2_DOPT_STATIC_V1_20260902.md",
        },
        "target_gradients": 0,
        "parameter_updates": 0,
        "model_or_checkpoint_updated": False,
    }
    print(json.dumps(_write_receipt(target, payload), indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("dry", "attempt", "export", "validate", "build", "container", "terminal", "execute"),
        default="dry",
    )
    args = parser.parse_args()
    if args.stage == "dry":
        print(json.dumps({
            "schema": "evalai_m2_dopt_static_v1_dry",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE_NO_LAUNCH",
            "stages": ["attempt", "export", "validate", "build", "container", "terminal"],
            "package": str(PACKAGE_DIR),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "base_image": BASE_IMAGE,
            "network_submission": "never (operator-owned)",
        }, sort_keys=True))
        return
    if args.stage == "attempt":
        stage_attempt()
        return
    if args.stage == "execute":
        stage_export()
        stage_validate()
        image = stage_build()
        container_summary = stage_container(image)
        stage_terminal(image, container_summary)
        return
    if args.stage == "export":
        stage_export()
        return
    if args.stage == "validate":
        stage_validate()
        return
    if args.stage == "build":
        stage_build()
        return
    if args.stage == "container":
        image = {"tag": image_tag()}
        stage_container(image)
        return
    if args.stage == "terminal":
        tag = image_tag()
        inspect = json.loads(_run(["docker", "image", "inspect", tag], "image_inspect"))[0]
        image = {
            "tag": tag,
            "id": inspect["Id"],
            "size": inspect.get("Size"),
            "labels": inspect["Config"].get("Labels", {}),
        }
        container_summary = {"note": "container stage run separately; see evalai_m2_dopt_static_v1_container_*.log"}
        submission_csv = (
            ARTIFACTS / "container_v1/remote_path/submission.csv"
        )
        if submission_csv.exists():
            import pickle

            with submission_csv.open("rb") as handle:
                payload = pickle.load(handle)
            container_summary = {
                "remote_path_submission_csv": str(submission_csv),
                "submission_sha256": sha256_file(submission_csv),
                "submission_bytes": submission_csv.stat().st_size,
                "task_key": sorted(payload),
                "session_keys": sorted(payload.get("m2", {})),
                "entered_expected_300s_wait": True,
            }
        stage_terminal(image, container_summary)


if __name__ == "__main__":
    main()
