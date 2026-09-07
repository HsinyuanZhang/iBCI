"""V2 repaired executor; V1 science functions are reused unchanged."""
from __future__ import annotations

import copy
import os
import tempfile
import time
from pathlib import Path

from mc_maze.dandi688_cp_film_postpool_v1 import runner as v1_runner
from mc_maze.dandi688_cp_film_v1 import core as base_core
from mc_maze.dandi688_cp_film_v1 import data as base_data
from mc_maze.dandi688_cp_film_v1 import plan as base_plan
from mc_maze.dandi688_cp_film_v1 import runner as base_runner

from . import core, plan


def _verify(repo_root: Path, seed: int):
    checks = {
        plan.DESIGN_RELATIVE: plan.DESIGN_SHA256,
        plan.WORKORDER_RELATIVE: plan.WORKORDER_SHA256,
        plan.INCIDENT_RELATIVE: plan.INCIDENT_SHA256,
        base_plan.MANIFEST_RELATIVE: base_plan.MANIFEST_SHA256,
        base_plan.TEACHER_RELATIVE: base_plan.TEACHER_SHA256,
        base_plan.ANCHOR_RELATIVE[seed]: base_plan.ANCHOR_SHA256[seed],
    }
    for relative, expected in checks.items():
        path = repo_root / relative
        base_core.require(path.is_file(), f"missing frozen input {relative}")
        base_core.require(base_core.sha256_file(path) == expected, f"frozen input SHA drift {relative}")
    v1_root = repo_root / plan.V1_ROOT_RELATIVE
    for name, expected in plan.V1_BODY_SHA256.items():
        path = v1_root / name
        base_core.require(path.is_file(), f"missing V1 incident body {name}")
        base_core.require(base_core.sha256_file(path) == expected, f"V1 incident body drift {name}")
    return checks


def execute(repo_root: Path, seed: int):
    repo_root = Path(repo_root).resolve()
    frozen = _verify(repo_root, seed)
    base_core.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "requires isolated GPU0")
    base_core.require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "CUBLAS contract drift")
    root = plan.result_root(repo_root, seed)
    base_core.require(root.parent.is_dir(), "result parent missing")
    base_core.require(not root.exists(), "result root exists")
    root.mkdir(mode=0o755)
    attempt_sha = base_runner._publish_json(root, "attempt.json", {
        "schema": f"{plan.SCHEMA}_attempt", "seed": seed, "frozen_inputs": frozen,
        "v1_incident_root": plan.V1_ROOT_RELATIVE, "v1_incident_body_sha256": plan.V1_BODY_SHA256,
        "formal_test_files_opened": False, "gpu_initialized": False, "started_at_unix": time.time(),
    })
    progress = {"stage": "attempt", "published": ["attempt.json"], "formal_test_files_opened": False}
    try:
        launch_sha = base_runner._publish_json(root, "launch.json", {
            "schema": f"{plan.SCHEMA}_launch", "attempt_sha256": attempt_sha, "seed": seed,
            "gpu": base_runner._gpu0_attestation(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        })
        progress.update(stage="launch", published=["attempt.json", "launch.json"])
        dm = base_data.prepare_datamodule(repo_root)
        materials = base_data.materialize(repo_root, dm)
        authority_sha = base_runner._publish_json(root, "source_authority.json", {
            "schema": f"{plan.SCHEMA}_source_authority", "attempt_sha256": attempt_sha,
            "launch_sha256": launch_sha,
            "train_sessions": sorted(key for key, row in materials.items() if row.split == "train"),
            "validation_sessions": sorted(key for key, row in materials.items() if row.split == "val"),
            "formal_test_names_only": list(dm.session_splits["test"]), "formal_test_files_opened": False,
            "profiles": {key: {
                "m10_sha256": base_core.array_sha256(row.profile10),
                "m30_sha256": base_core.array_sha256(row.profile30),
                "activity30_sha256": base_core.array_sha256(row.record.calib_trials),
                "t4_sha256": base_core.array_sha256(row.record.side_features),
                "q50_sha256": base_core.array_sha256(row.q50_starts),
            } for key, row in sorted(materials.items())},
        })
        progress.update(stage="source_authority", published=["attempt.json", "launch.json", "source_authority.json"])

        import torch
        base_core.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "CUDA geometry drift")
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        device = torch.device("cuda:0")
        student = base_runner._prepare_student(repo_root, seed, device)
        substrate_before = base_runner._state_sha(student)
        native_post_pool_before = base_runner._state_sha(student.id_encoder.post_pool)
        cache = base_runner._session_cache(materials, student, device, seed)
        template = core.build_arm(student.id_encoder.post_pool).to(device)
        arms = {name: copy.deepcopy(template).to(device) for name in plan.ARMS}
        trainable_names = {
            name: sorted(key for key, value in arm.named_parameters() if value.requires_grad)
            for name, arm in arms.items()
        }
        base_core.require(len({tuple(value) for value in trainable_names.values()}) == 1, "trainable names drift")
        for arm in arms.values():
            base_core.require(
                sum(value.numel() for value in arm.parameters() if value.requires_grad) == plan.TRAINABLE_PARAMETERS,
                "requires_grad parameter count drift",
            )
            base_core.require(base_runner._state_sha(arm.post_pool) == native_post_pool_before, "post-pool init drift")

        training, averaged = v1_runner._train(materials, cache, student, arms, seed, device)
        substrate_after = base_runner._state_sha(student)
        base_core.require(substrate_after == substrate_before, "frozen native substrate changed")
        final_post_pool = {name: base_runner._state_sha(arm.post_pool) for name, arm in arms.items()}
        averaged_post_pool = {name: base_runner._state_sha(arm.post_pool) for name, arm in averaged.items()}
        base_core.require(all(value != native_post_pool_before for value in final_post_pool.values()), "post-pool did not update")
        base_core.require(all(value != native_post_pool_before for value in averaged_post_pool.values()), "averaged post-pool unchanged")

        checkpoints = {}
        for name in plan.ARMS:
            buffer = tempfile.SpooledTemporaryFile(max_size=1 << 20)
            torch.save({"arm": name, "seed": seed, "final": arms[name].state_dict(),
                        "average_9_12": averaged[name].state_dict()}, buffer)
            buffer.seek(0)
            checkpoints[name] = base_runner._publish_bytes(root, f"coadapt_{name.lower()}.pt", buffer.read())
        training.update({
            "schema": f"{plan.SCHEMA}_training", "seed": seed, "attempt_sha256": attempt_sha,
            "source_authority_sha256": authority_sha, "requires_grad_parameter_count": plan.TRAINABLE_PARAMETERS,
            "trainable_names": trainable_names, "average_epochs": list(plan.AVERAGE_EPOCHS),
            "native_post_pool_before_sha256": native_post_pool_before,
            "final_post_pool_sha256": final_post_pool, "averaged_post_pool_sha256": averaged_post_pool,
            "substrate_before_sha256": substrate_before, "substrate_after_sha256": substrate_after,
            "checkpoints": checkpoints,
        })
        training_sha = base_runner._publish_json(root, "training.json", training)
        progress.update(stage="training", published=["attempt.json", "launch.json", "source_authority.json", "training.json"])
        score = v1_runner._evaluate(materials, cache, student, arms, averaged, device)
        score.update({"schema": f"{plan.SCHEMA}_score", "seed": seed, "attempt_sha256": attempt_sha,
                      "source_authority_sha256": authority_sha, "training_sha256": training_sha})
        score_sha = base_runner._publish_json(root, "score.json", score)
        terminal = {
            "schema": f"{plan.SCHEMA}_terminal", "status": "PASS", "attempt_sha256": attempt_sha,
            "launch_sha256": launch_sha, "source_authority_sha256": authority_sha,
            "training_sha256": training_sha, "score_sha256": score_sha, "decision": score["decision"],
            "formal_test_files_opened": False, "target_session_updates": 0, "evalai_push": False,
        }
        base_runner._publish_json(root, "terminal.json", terminal)
        return terminal
    except Exception as exc:
        failure = {
            "schema": f"{plan.SCHEMA}_failure", "status": "FAIL", "attempt_sha256": attempt_sha,
            "progress": progress, "exception_type": type(exc).__name__, "exception_message": str(exc),
            "formal_test_files_opened": False, "target_session_updates": 0, "evalai_push": False,
        }
        if not (root / "failure.json").exists() and not (root / "terminal.json").exists():
            base_runner._publish_json(root, "failure.json", failure)
        raise

