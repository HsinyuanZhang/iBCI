"""One-GPU runner for post-pool co-adaptation and matched controls."""
from __future__ import annotations

import copy
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from mc_maze.dandi688_cp_film_v1 import core as parent_core
from mc_maze.dandi688_cp_film_v1 import data as parent_data
from mc_maze.dandi688_cp_film_v1 import plan as parent_plan
from mc_maze.dandi688_cp_film_v1 import runner as parent_runner

from . import core, plan


def _verify(repo_root: Path, seed: int):
    parent_core.require(seed in plan.SEEDS, "unsupported seed")
    checks = {
        plan.DESIGN_RELATIVE: plan.DESIGN_SHA256,
        plan.WORKORDER_RELATIVE: plan.WORKORDER_SHA256,
        parent_plan.MANIFEST_RELATIVE: parent_plan.MANIFEST_SHA256,
        parent_plan.TEACHER_RELATIVE: parent_plan.TEACHER_SHA256,
        parent_plan.ANCHOR_RELATIVE[seed]: parent_plan.ANCHOR_SHA256[seed],
    }
    for relative, expected in checks.items():
        path = repo_root / relative
        parent_core.require(path.is_file(), f"missing input {relative}")
        parent_core.require(parent_core.sha256_file(path) == expected, f"input SHA drift {relative}")
    return checks


def _arm_profile_key(arm: str) -> str:
    return {"CP10": "M10", "CP30": "M30", "SHUFFLE10": "SHUFFLED10", "EMPTY": "ZERO"}[arm]


def _train(materials, cache, student, arms, seed, device):
    import torch

    optimizers = {
        name: torch.optim.Adam(arm.parameters(), lr=plan.LEARNING_RATE, weight_decay=0.0)
        for name, arm in arms.items()
    }
    train_sessions = sorted(session for session, material in materials.items() if material.split == "train")
    first_identity_equal, first_prediction_equal = {}, {}
    histories, snapshots = [], {name: [] for name in plan.ARMS}
    steps = {name: 0 for name in plan.ARMS}
    first = True
    for epoch in range(plan.EPOCHS):
        losses = {name: [] for name in plan.ARMS}
        order_rng = np.random.Generator(np.random.PCG64(seed * 1000 + epoch))
        for index in order_rng.permutation(len(train_sessions)):
            session = train_sessions[int(index)]
            material = materials[session]
            starts = parent_data.epoch_starts(material, seed, epoch)
            state = cache[session]
            for offset in range(0, starts.size, plan.BATCH_SIZE):
                chunk = starts[offset : offset + plan.BATCH_SIZE]
                neural_np, target_np = parent_runner._batch_arrays(material.record, chunk)
                neural = torch.from_numpy(neural_np).to(device)
                target = torch.from_numpy(target_np).to(device)
                if first:
                    with torch.no_grad():
                        native = student.decode_with_identity(
                            neural, state["native_identity"].expand(neural.shape[0], -1, -1)
                        )[:, -1, :] / plan.BEHAVIOR_SCALE
                        for name, arm in arms.items():
                            identity = arm.identity(
                                state["mean_feature"], state["carrier"],
                                state["profiles"][_arm_profile_key(name)],
                            )
                            prediction = student.decode_with_identity(
                                neural, identity.expand(neural.shape[0], -1, -1)
                            )[:, -1, :] / plan.BEHAVIOR_SCALE
                            first_identity_equal[name] = bool(torch.equal(identity, state["native_identity"]))
                            first_prediction_equal[name] = bool(torch.equal(prediction, native))
                    parent_core.require(
                        all(first_identity_equal.values()) and all(first_prediction_equal.values()),
                        "zero-init parity failed",
                    )
                    first = False
                for name, arm in arms.items():
                    optimizer = optimizers[name]
                    optimizer.zero_grad(set_to_none=True)
                    identity = arm.identity(
                        state["mean_feature"], state["carrier"],
                        state["profiles"][_arm_profile_key(name)],
                    )
                    prediction = student.decode_with_identity(
                        neural, identity.expand(neural.shape[0], -1, -1)
                    )[:, -1, :] / plan.BEHAVIOR_SCALE
                    loss = torch.mean(torch.square(prediction - target))
                    parent_core.require(bool(torch.isfinite(loss)), f"{name}: loss nonfinite")
                    loss.backward()
                    optimizer.step()
                    parent_core.require(
                        all(bool(torch.isfinite(parameter).all()) for parameter in arm.parameters()),
                        f"{name}: parameter nonfinite",
                    )
                    losses[name].append(float(loss.detach().cpu()))
                    steps[name] += 1
        epoch_number = epoch + 1
        state_digests = {name: parent_runner._state_sha(arm) for name, arm in arms.items()}
        histories.append(
            {
                "epoch": epoch_number,
                "mean_last_bin_mse": {name: float(np.mean(value)) for name, value in losses.items()},
                "batch_count": {name: len(value) for name, value in losses.items()},
                "state_sha256": state_digests,
            }
        )
        if epoch_number in plan.AVERAGE_EPOCHS:
            for name, arm in arms.items():
                snapshots[name].append({key: value.detach().cpu().clone() for key, value in arm.state_dict().items()})
        print(f"epoch={epoch_number} losses={histories[-1]['mean_last_bin_mse']}", flush=True)
    averaged = {}
    for name, arm in arms.items():
        candidate = copy.deepcopy(arm).cpu()
        candidate.load_state_dict(core.averaged_state(snapshots[name]), strict=True)
        averaged[name] = candidate.to(device).eval()
        arm.eval()
    return {
        "history": histories,
        "gradient_steps": steps,
        "first_identity_bitwise_equal": first_identity_equal,
        "first_prediction_bitwise_equal": first_prediction_equal,
        "final_state_sha256": {name: parent_runner._state_sha(arm) for name, arm in arms.items()},
        "averaged_state_sha256": {name: parent_runner._state_sha(arm) for name, arm in averaged.items()},
    }, averaged


def _evaluate(materials, cache, student, final_arms, averaged_arms, device):
    import torch

    states = {"AVG": averaged_arms, "FINAL": final_arms}
    specs = {
        "CP10@M10": ("CP10", "M10"),
        "CP30@M30": ("CP30", "M30"),
        "SHUFFLE10@M10": ("SHUFFLE10", "SHUFFLED10"),
        "EMPTY@ZERO": ("EMPTY", "ZERO"),
    }
    by_cell = {"NATIVE": {}}
    for stage in states:
        for cell in specs:
            by_cell[f"{stage}/{cell}"] = {}
    rows = []
    validation = sorted(session for session, material in materials.items() if material.split == "val")
    parent_core.require(len(validation) == plan.VALIDATION_SESSIONS, "validation roster drift")
    for session in validation:
        material, state = materials[session], cache[session]
        identities = {"NATIVE": state["native_identity"]}
        with torch.no_grad():
            for stage, arm_map in states.items():
                for cell, (name, profile_key) in specs.items():
                    identities[f"{stage}/{cell}"] = arm_map[name].identity(
                        state["mean_feature"], state["carrier"], state["profiles"][profile_key]
                    )
            predictions = {name: [] for name in identities}
            targets = []
            for offset in range(0, material.q50_starts.size, 256):
                starts = material.q50_starts[offset : offset + 256]
                neural_np, target_np = parent_runner._batch_arrays(material.record, starts)
                neural = torch.from_numpy(neural_np).to(device)
                targets.append(target_np)
                for name, identity in identities.items():
                    prediction = student.decode_with_identity(
                        neural, identity.expand(neural.shape[0], -1, -1)
                    )[:, -1, :] / plan.BEHAVIOR_SCALE
                    predictions[name].append(prediction.detach().cpu().numpy().astype(np.float32))
        target = np.ascontiguousarray(np.concatenate(targets), dtype=np.float32)
        row = {"session": session, "window_count": int(target.shape[0]), "cells": {}}
        for name, chunks in predictions.items():
            prediction = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
            value = parent_runner._r2(target, prediction)
            by_cell[name][session] = value
            row["cells"][name] = {"r2": value, "prediction_sha256": parent_core.array_sha256(prediction)}
        rows.append(row)
    contrasts = {
        name: parent_core.paired_summary(values, by_cell["NATIVE"])
        for name, values in by_cell.items() if name != "NATIVE"
    }
    direct = {}
    for stage in states:
        empty = by_cell[f"{stage}/EMPTY@ZERO"]
        for profile in ("CP10@M10", "CP30@M30", "SHUFFLE10@M10"):
            direct[f"{stage}/{profile}-minus-EMPTY"] = parent_core.paired_summary(
                by_cell[f"{stage}/{profile}"], empty
            )
    decision = core.decide(
        contrasts["AVG/CP10@M10"], contrasts["AVG/CP30@M30"],
        direct["AVG/CP10@M10-minus-EMPTY"], direct["AVG/CP30@M30-minus-EMPTY"],
    )
    return {
        "rows": rows,
        "equal_session_mean_r2": {name: float(np.mean(list(values.values()))) for name, values in by_cell.items()},
        "contrasts_vs_native": contrasts,
        "direct_control_contrasts": direct,
        "decision": decision,
    }


def execute(repo_root: Path, seed: int):
    repo_root = Path(repo_root).resolve()
    frozen = _verify(repo_root, seed)
    parent_core.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "requires isolated GPU0")
    parent_core.require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "CUBLAS contract drift")
    root = plan.result_root(repo_root, seed)
    parent_core.require(root.parent.is_dir(), "result parent missing")
    parent_core.require(not root.exists(), "result root exists")
    root.mkdir(mode=0o755)
    attempt_sha = parent_runner._publish_json(root, "attempt.json", {
        "schema": f"{plan.SCHEMA}_attempt", "seed": seed, "frozen_inputs": frozen,
        "formal_test_files_opened": False, "gpu_initialized": False, "started_at_unix": time.time(),
    })
    progress = {"stage": "attempt", "published": ["attempt.json"], "formal_test_files_opened": False}
    try:
        launch_sha = parent_runner._publish_json(root, "launch.json", {
            "schema": f"{plan.SCHEMA}_launch", "attempt_sha256": attempt_sha, "seed": seed,
            "gpu": parent_runner._gpu0_attestation(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        })
        progress.update(stage="launch", published=["attempt.json", "launch.json"])
        dm = parent_data.prepare_datamodule(repo_root)
        materials = parent_data.materialize(repo_root, dm)
        authority_sha = parent_runner._publish_json(root, "source_authority.json", {
            "schema": f"{plan.SCHEMA}_source_authority", "attempt_sha256": attempt_sha,
            "launch_sha256": launch_sha,
            "train_sessions": sorted(key for key, row in materials.items() if row.split == "train"),
            "validation_sessions": sorted(key for key, row in materials.items() if row.split == "val"),
            "formal_test_names_only": list(dm.session_splits["test"]), "formal_test_files_opened": False,
            "profiles": {key: {
                "m10_sha256": parent_core.array_sha256(row.profile10),
                "m30_sha256": parent_core.array_sha256(row.profile30),
                "activity30_sha256": parent_core.array_sha256(row.record.calib_trials),
                "t4_sha256": parent_core.array_sha256(row.record.side_features),
                "q50_sha256": parent_core.array_sha256(row.q50_starts),
            } for key, row in sorted(materials.items())},
        })
        progress.update(stage="source_authority", published=["attempt.json", "launch.json", "source_authority.json"])

        import torch
        parent_core.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "CUDA geometry drift")
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        device = torch.device("cuda:0")
        student = parent_runner._prepare_student(repo_root, seed, device)
        substrate_before = parent_runner._state_sha(student)
        cache = parent_runner._session_cache(materials, student, device, seed)
        template = core.build_arm(student.id_encoder.post_pool).to(device)
        arms = {name: copy.deepcopy(template).to(device) for name in plan.ARMS}
        parent_core.require(len({parent_runner._state_sha(arm) for arm in arms.values()}) == 1, "arm init drift")
        trainable_names = {name: sorted(key for key, value in arm.named_parameters() if value.requires_grad) for name, arm in arms.items()}
        parent_core.require(len({tuple(value) for value in trainable_names.values()}) == 1, "trainable names drift")
        training, averaged = _train(materials, cache, student, arms, seed, device)
        substrate_after = parent_runner._state_sha(student)
        parent_core.require(substrate_after == substrate_before, "frozen substrate changed")

        checkpoints = {}
        for name in plan.ARMS:
            buffer = tempfile.SpooledTemporaryFile(max_size=1 << 20)
            torch.save({"arm": name, "seed": seed, "final": arms[name].state_dict(), "average_9_12": averaged[name].state_dict()}, buffer)
            buffer.seek(0)
            checkpoints[name] = parent_runner._publish_bytes(root, f"coadapt_{name.lower()}.pt", buffer.read())
        training.update({
            "schema": f"{plan.SCHEMA}_training", "seed": seed, "attempt_sha256": attempt_sha,
            "source_authority_sha256": authority_sha, "trainable_parameter_count": plan.TRAINABLE_PARAMETERS,
            "trainable_names": trainable_names, "average_epochs": list(plan.AVERAGE_EPOCHS),
            "substrate_before_sha256": substrate_before, "substrate_after_sha256": substrate_after,
            "checkpoints": checkpoints,
        })
        training_sha = parent_runner._publish_json(root, "training.json", training)
        progress.update(stage="training", published=["attempt.json", "launch.json", "source_authority.json", "training.json"])
        score = _evaluate(materials, cache, student, arms, averaged, device)
        score.update({"schema": f"{plan.SCHEMA}_score", "seed": seed, "attempt_sha256": attempt_sha,
                      "source_authority_sha256": authority_sha, "training_sha256": training_sha})
        score_sha = parent_runner._publish_json(root, "score.json", score)
        terminal = {
            "schema": f"{plan.SCHEMA}_terminal", "status": "PASS", "attempt_sha256": attempt_sha,
            "launch_sha256": launch_sha, "source_authority_sha256": authority_sha,
            "training_sha256": training_sha, "score_sha256": score_sha, "decision": score["decision"],
            "formal_test_files_opened": False, "target_session_updates": 0, "evalai_push": False,
        }
        parent_runner._publish_json(root, "terminal.json", terminal)
        return terminal
    except Exception as exc:
        failure = {
            "schema": f"{plan.SCHEMA}_failure", "status": "FAIL", "attempt_sha256": attempt_sha,
            "progress": progress, "exception_type": type(exc).__name__, "exception_message": str(exc),
            "formal_test_files_opened": False, "target_session_updates": 0, "evalai_push": False,
        }
        if not (root / "failure.json").exists() and not (root / "terminal.json").exists():
            parent_runner._publish_json(root, "failure.json", failure)
        raise

