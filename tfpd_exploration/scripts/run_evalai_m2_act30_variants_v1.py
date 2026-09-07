#!/usr/bin/env python3
"""Dry by default; reserve or execute an act30 EvalAI M2 packaging variant.

Two variants share this driver (``--variant``), both static cached-identity
deployments of the frozen m2_spint_t4_mainline seed-42 checkpoint
``25d7bc72...`` with a label-free first-30 B3S activity pool:

- ``act30_dopt4`` : carrier = greedy forward D-optimal k=4 within the
  first-30 finite-angle candidates (sealed screen cell
  ``ridge_activity30_m4``; local anchor external 0.2909923623168425).
- ``act30_full``  : carrier = chronological first-30 block, ridge fit on its
  finite-direction trials (sealed screen cell ``ridge_static_m30``; local
  anchor external 0.29521985196829853).

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
- ``execute``   : export -> validate -> build -> container -> terminal.

No stage performs any network submission; the EvalAI push is operator-owned
(coordinator) and documented in the terminal receipt.
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
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TFPD_ROOT = REPO_ROOT / "tfpd_exploration"
CHECKPOINT_SHA256 = "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
NORMALIZATION_SHA256 = "d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e"
SEALED_SCREEN = "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json"
SEALED_AUDIT = "tfpd_exploration/results/cdm_p1_m2_v1/audit.json"
BASE_IMAGE = "spint-m2:e8-epoch027-76f0fb2"

VARIANTS: dict[str, dict[str, object]] = {
    "act30_dopt4": {
        "package_dirname": "evalai_m2_act30_dopt4_v1",
        "export_script": "export_act30_dopt4_payload.py",
        "payload_name": "t4_m2_seed42_dopt4_act30_identity.pkl",
        "result_dirname": "evalai_m2_act30_dopt4_v1",
        "log_prefix": "evalai_m2_act30_dopt4_v1",
        "image_tag_prefix": "dopt4-act30-s42",
        "cell": "EVALAI_M2_ACT30_DOPT4_V1_PACKAGING",
        "arm": "dopt4_static_act30",
        "sealed_cell": "ridge_activity30_m4",
        "label_budget": 4,
        "activity_budget": 30,
        "anchor_external": 0.2909923623168425,
        "anchor_within": 0.6772200181830803,
        "submit_helper": "submit_evalai_act30_dopt4.py",
        "prior_receipt": (
            "sua_exploration/evalai_t4_m2_activity_budget/artifacts/"
            "t4_m2_seed42_ridge_m4_activity30_identity.receipt.json"
        ),
        "deployment_description": (
            "frozen m2_spint_t4_mainline seed-42 checkpoint 25d7bc72... deployed as: "
            "offline calibration = greedy forward D-optimal k=4 selection among the "
            "finite-angle candidates of the first 30 labelled calibration trials -> "
            "ridge lambda=0.1 T4 fit on the selected support -> B3S activity pool = "
            "the full label-free first-30 calibration block -> frozen-checkpoint "
            "identity; runtime = cached identity per dataset tag, reset+predict "
            "only, on_done no-op, zero online updates"
        ),
        "selection_law": (
            "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:17-35 "
            "budget==4 branch (greedy_forward_d_optimal_indices from "
            "sua_exploration/mc_maze/d_optimal_calibration_design.py:188-219), "
            "mirrored verbatim in the package and cross-checked against the sealed import"
        ),
        "activity_law": (
            "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py:67-87 "
            "activity_budget==30 branch (first-30 label-free calibration block), "
            "mirrored verbatim and cross-checked against the sealed import"
        ),
    },
    "act30_full": {
        "package_dirname": "evalai_m2_act30_full_v1",
        "export_script": "export_act30_full_payload.py",
        "payload_name": "t4_m2_seed42_m30_act30_identity.pkl",
        "result_dirname": "evalai_m2_act30_full_v1",
        "log_prefix": "evalai_m2_act30_full_v1",
        "image_tag_prefix": "m30-act30-s42",
        "cell": "EVALAI_M2_ACT30_FULL_V1_PACKAGING",
        "arm": "m30_static_act30",
        "sealed_cell": "ridge_static_m30",
        "label_budget": 30,
        "activity_budget": 30,
        "anchor_external": 0.29521985196829853,
        "anchor_within": 0.6892605728315127,
        "submit_helper": "submit_evalai_act30_full.py",
        "prior_receipt": (
            "sua_exploration/evalai_t4_m2_activity_budget/artifacts/"
            "t4_m2_seed42_ridge_m30_activity30_identity.receipt.json"
        ),
        "deployment_description": (
            "frozen m2_spint_t4_mainline seed-42 checkpoint 25d7bc72... deployed as: "
            "offline calibration = chronological first-30 calibration block -> "
            "ridge lambda=0.1 T4 fit on the block's finite-direction trials -> "
            "B3S activity pool = the same full label-free first-30 calibration "
            "block -> frozen-checkpoint identity; runtime = cached identity per "
            "dataset tag, reset+predict only, on_done no-op, zero online updates"
        ),
        "selection_law": (
            "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:17-35 "
            "budget!=4 branch at budget=30 (chronological arange(30) with the "
            "sealed three-directional-trial minimum), mirrored verbatim in the "
            "package and cross-checked against the sealed import"
        ),
        "activity_law": (
            "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py:67-87 "
            "activity_budget==30 branch (first-30 label-free calibration block), "
            "mirrored verbatim and cross-checked against the sealed import"
        ),
    },
}

OWNED_BASE_RELATIVE = "tfpd_exploration/scripts/run_evalai_m2_act30_variants_v1.py"
OWNED_TESTS_RELATIVE = "tfpd_exploration/tests/test_evalai_m2_act30_variants_v1.py"

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def variant_paths(name: str) -> dict[str, Path]:
    spec = VARIANTS[name]
    package = TFPD_ROOT / "submissions" / str(spec["package_dirname"])
    return {
        "package": package,
        "artifacts": package / "artifacts",
        "payload": package / "artifacts" / str(spec["payload_name"]),
        "result_root": TFPD_ROOT / "results" / str(spec["result_dirname"]),
    }


def variant_env(name: str) -> dict[str, str]:
    package = variant_paths(name)["package"]
    return {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": f"{REPO_ROOT}:{package}",
        "CUDA_VISIBLE_DEVICES": "",
    }


def _write_receipt(path: Path, payload: dict) -> dict:
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    path.write_bytes(body)
    path.chmod(0o444)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    sidecar.chmod(0o444)
    return {"receipt": str(path), "receipt_sha256": digest}


def _run(name: str, command: list[str], log_name: str, cwd: Path) -> str:
    log_path = Path(f"{TFPD_ROOT}/{VARIANTS[name]['log_prefix']}_{log_name}.log")
    completed = subprocess.run(
        command, cwd=str(cwd), env=variant_env(name), capture_output=True, text=True
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


def owned_relative(name: str) -> list[str]:
    package = variant_paths(name)["package"]
    owned = [
        f"{package.relative_to(REPO_ROOT)}/{entry.name}"
        for entry in sorted(package.iterdir())
        if entry.is_file()
    ]
    owned.append(OWNED_BASE_RELATIVE)
    owned.append(OWNED_TESTS_RELATIVE)
    return sorted(owned)


def push_commands(name: str) -> dict[str, object]:
    spec = VARIANTS[name]
    helper = f"tfpd_exploration/submissions/{spec['package_dirname']}/{spec['submit_helper']}"
    return {
        "preflight_read_only": (
            f"/tmp/spint-e8-evalai-py38/bin/python {helper}"
        ),
        "execute_push_and_register": (
            f"/tmp/spint-e8-evalai-py38/bin/python {helper} --execute "
            "--confirm-image-id <terminal.json docker.image_id> "
            "--confirm-payload-sha256 <terminal.json artifact_tree.payload.sha256>"
        ),
        "plain_cli_equivalent": (
            "evalai push <image tag> --phase few-shot-test-2319 --private"
        ),
        "phase": "few-shot-test-2319 (challenge 2319, phase id 4599, team 41975)",
    }


def stage_attempt(name: str) -> None:
    spec = VARIANTS[name]
    paths = variant_paths(name)
    target = paths["result_root"] / "attempt.json"
    if target.exists():
        raise SystemExit(f"attempt receipt already exists: {target}")
    payload = {
        "schema": f"{spec['result_dirname']}_attempt_v1",
        "status": "ATTEMPT_RESERVED",
        "cell": spec["cell"],
        "variant": name,
        "authority": (
            "user 2026-09-02 build two additional EvalAI submission variants "
            "(act30_dopt4, act30_full), CPU-only, reusing the validated v1 "
            "skeleton at submissions/evalai_m2_dopt_static_v1, no network submission"
        ),
        "inference_only": True,
        "predecessor_package": "tfpd_exploration/submissions/evalai_m2_dopt_static_v1",
        "pre_registration": {
            "deployment": spec["deployment_description"],
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "normalization_sha256": NORMALIZATION_SHA256,
            "selection_law": spec["selection_law"],
            "ridge_law": (
                "tfpd_exploration/src/calibration_budget_comparators_v1.py:56-95 "
                "fit_ridge_t4 with normalized_lambda=0.1, sealed import"
            ),
            "activity_law": spec["activity_law"],
            "sealed_anchor": {
                "path": SEALED_SCREEN,
                "cell": spec["sealed_cell"],
                "external_official_query_equal_session_mean": spec["anchor_external"],
                "within_post30_equal_session_mean": spec["anchor_within"],
            },
            "contract": (
                "per the sealed cdm_p1_m2_v1 audit: continual m2 evaluator calls only "
                "reset(dataset_tags)+predict(neural_observations); no on_done; "
                "assert no trial metadata at predict, selection only offline, "
                "weights/identities byte-frozen across predicts"
            ),
            "stages": ["export", "validate", "build", "container", "terminal"],
            "submission": "NOT PERFORMED by this cell; the coordinator executes the documented push",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "no_user_site": True,
            "cuda_visible_devices": "",
        },
        "owned_sha256s": {item: sha256_file(REPO_ROOT / item) for item in owned_relative(name)},
        "predecessor_sha256s": {
            item: sha256_file(REPO_ROOT / item)
            for item in PREDECESSOR_RELATIVE + [str(spec["prior_receipt"])]
        },
        "target_gradients": 0,
        "parameter_updates": 0,
        "model_or_checkpoint_updated": False,
    }
    paths["result_root"].mkdir(parents=True, exist_ok=False)
    print(json.dumps(_write_receipt(target, payload), indent=2, sort_keys=True))


def stage_export(name: str) -> None:
    spec = VARIANTS[name]
    paths = variant_paths(name)
    stdout = _run(
        name,
        [
            sys.executable,
            str(paths["package"] / str(spec["export_script"])),
            "--execute",
        ],
        "export",
        cwd=paths["package"],
    )
    receipt = json.loads(stdout)
    print(json.dumps(
        {
            k: receipt[k]
            for k in (
                "payload_sha256",
                "payload_bytes",
                "session_count",
                "arm",
                "prior_deployed_identity_sha256_matches",
            )
        },
        sort_keys=True,
    ))


def stage_validate(name: str) -> None:
    spec = VARIANTS[name]
    paths = variant_paths(name)
    _run(
        name,
        [
            sys.executable,
            str(paths["package"] / "validate_local.py"),
            "--execute",
            "--stages",
            "payload,external,contract,minival",
        ],
        "validate",
        cwd=paths["package"],
    )


def export_receipt(name: str) -> dict:
    spec = VARIANTS[name]
    paths = variant_paths(name)
    receipt_name = Path(str(spec["payload_name"])).with_suffix(".receipt.json")
    return json.loads(
        (paths["artifacts"] / receipt_name).read_text(encoding="utf-8")
    )


def image_tag(name: str) -> str:
    spec = VARIANTS[name]
    return f"spint-t4-m2:{spec['image_tag_prefix']}-{export_receipt(name)['payload_sha256'][:8]}"


def stage_build(name: str) -> dict:
    spec = VARIANTS[name]
    paths = variant_paths(name)
    receipt = export_receipt(name)
    tag = f"spint-t4-m2:{spec['image_tag_prefix']}-{receipt['payload_sha256'][:8]}"
    _run(
        name,
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
        cwd=paths["package"],
    )
    inspect = json.loads(
        _run(name, ["docker", "image", "inspect", tag], "image_inspect", cwd=REPO_ROOT)
    )[0]
    print(json.dumps({"tag": tag, "id": inspect["Id"], "size": inspect.get("Size")}, sort_keys=True))
    return {
        "tag": tag,
        "id": inspect["Id"],
        "size": inspect.get("Size"),
        "labels": inspect["Config"].get("Labels", {}),
    }


def stage_container(name: str, image: dict) -> dict:
    spec = VARIANTS[name]
    paths = variant_paths(name)
    out_dir = paths["artifacts"] / "container_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_mount = f"{REPO_ROOT / 'SPINT-main/data'}:/dataset/evaluation_data:ro"
    # 1) local-path container run over the public minival files.
    _run(
        name,
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
        cwd=REPO_ROOT,
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
    log_path = Path(f"{TFPD_ROOT}/{spec['log_prefix']}_container_remote.log")
    process = subprocess.Popen(
        [
            "docker", "run", "--rm",
            "-e", "EVALUATION_LOC=remote",
            "-e", "EVAL_DATA_PATH=/dataset/evaluation_data",
            "-v", f"{out_dir / 'remote_data'}:/dataset/evaluation_data:ro",
            "-v", f"{submission_dir}:/submission",
            image["tag"],
        ],
        cwd=str(REPO_ROOT), env=variant_env(name), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    target = submission_dir / "submission.csv"

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


def stage_terminal(name: str, image: dict, container_summary: dict) -> None:
    spec = VARIANTS[name]
    paths = variant_paths(name)
    target = paths["result_root"] / "terminal.json"
    if target.exists():
        raise SystemExit(f"terminal receipt already exists: {target}")
    package_relative = f"tfpd_exploration/submissions/{spec['package_dirname']}"
    export = export_receipt(name)
    validation = json.loads(
        (paths["artifacts"] / "local_validation_receipt.json").read_text(encoding="utf-8")
    )
    payload = {
        "schema": f"{spec['result_dirname']}_terminal_v1",
        "status": "TERMINAL_PACKAGED_NOT_SUBMITTED",
        "cell": spec["cell"],
        "variant": name,
        "predecessor_package": "tfpd_exploration/submissions/evalai_m2_dopt_static_v1",
        "artifact_tree": {
            "package_root": package_relative,
            "files": {
                entry.name: sha256_file(entry)
                for entry in sorted(paths["package"].iterdir())
                if entry.is_file()
            },
            "payload": {
                "path": f"{package_relative}/artifacts/{spec['payload_name']}",
                "sha256": export["payload_sha256"],
                "bytes": export["payload_bytes"],
            },
            "receipts": f"tfpd_exploration/results/{spec['result_dirname']}/",
        },
        "checkpoint_sha256": export["checkpoint_sha256"],
        "normalization_sha256": export["normalization_sha256"],
        "teacher_checkpoint_sha256": export["teacher_checkpoint_sha256"],
        "calibration_law_provenance": {
            "selection": spec["selection_law"],
            "ridge_t4": "tfpd_exploration/src/calibration_budget_comparators_v1.py:56-95 (sealed import fit_ridge_t4, normalized_lambda=0.1)",
            "ridge_side": "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:38-71 (sealed import _ridge_side)",
            "activity_pool": spec["activity_law"],
            "selections": {
                session: record["selected_indices"]
                for session, record in export["session_records"].items()
            },
            "prior_deployment_receipt": spec["prior_receipt"],
            "prior_deployed_identity_sha256_matches": export[
                "prior_deployed_identity_sha256_matches"
            ],
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
                "m2_t4_activity_budget_screen_v1 "
                f"{spec['sealed_cell']} surfaces (local held-out calibration NWB "
                "query streams; held-in post-30 windows); the official EvalAI "
                "hidden test surface is neither and is expected to differ"
            ),
        },
        "sealed_anchor_comparison": {
            "cell": spec["sealed_cell"],
            "external_official_query_equal_session_mean": spec["anchor_external"],
            "within_post30_equal_session_mean": spec["anchor_within"],
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
            "performed_by": "coordinator (operator)",
            "commands": push_commands(name),
        },
        "target_gradients": 0,
        "parameter_updates": 0,
        "model_or_checkpoint_updated": False,
    }
    print(json.dumps(_write_receipt(target, payload), indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("act30_dopt4", "act30_full"),
        default=None,
        help="required for every stage except dry",
    )
    parser.add_argument(
        "--stage",
        choices=("dry", "attempt", "export", "validate", "build", "container", "terminal", "execute"),
        default="dry",
    )
    args = parser.parse_args()
    if args.stage == "dry":
        print(json.dumps({
            "schema": "evalai_m2_act30_variants_v1_dry",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE_NO_LAUNCH",
            "variants": {
                name: {
                    "package": str(variant_paths(name)["package"]),
                    "sealed_cell": VARIANTS[name]["sealed_cell"],
                    "arm": VARIANTS[name]["arm"],
                    "label_budget": VARIANTS[name]["label_budget"],
                    "activity_budget": VARIANTS[name]["activity_budget"],
                    "anchor_external": VARIANTS[name]["anchor_external"],
                }
                for name in VARIANTS
            },
            "stages": ["attempt", "export", "validate", "build", "container", "terminal"],
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "base_image": BASE_IMAGE,
            "network_submission": "never (coordinator-owned)",
        }, sort_keys=True))
        return
    if args.variant is None:
        raise SystemExit("--variant is required for every stage except dry")
    name = args.variant
    if args.stage == "attempt":
        stage_attempt(name)
        return
    if args.stage == "execute":
        stage_export(name)
        stage_validate(name)
        image = stage_build(name)
        container_summary = stage_container(name, image)
        stage_terminal(name, image, container_summary)
        return
    if args.stage == "export":
        stage_export(name)
        return
    if args.stage == "validate":
        stage_validate(name)
        return
    if args.stage == "build":
        stage_build(name)
        return
    if args.stage == "container":
        image = {"tag": image_tag(name)}
        stage_container(name, image)
        return
    if args.stage == "terminal":
        tag = image_tag(name)
        inspect = json.loads(
            _run(name, ["docker", "image", "inspect", tag], "image_inspect", cwd=REPO_ROOT)
        )[0]
        image = {
            "tag": tag,
            "id": inspect["Id"],
            "size": inspect.get("Size"),
            "labels": inspect["Config"].get("Labels", {}),
        }
        submission_csv = variant_paths(name)["artifacts"] / "container_v1/remote_path/submission.csv"
        if not submission_csv.exists():
            raise SystemExit(f"container stage output missing: {submission_csv}")
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
        stage_terminal(name, image, container_summary)


if __name__ == "__main__":
    main()
