"""Public-executable drivers for chunk-CDM scoring, packed training, and rescoring.

Unlike sealed T0/C1, ``--execute`` on the public CLI actually runs. GPU
capability is still fail-closed: ``--execute-gpu`` cannot mint one, and
``--execute`` requires ``--gpu-authorized`` plus a live occupancy check.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1

from . import gpu as gpu_module
from . import plan
from . import receipts


class ExecuteError(RuntimeError):
    """Fail closed for public execute."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecuteError(message)


def _bind_thread_env() -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    for name, value in v1.THREAD_ENVIRONMENT.items():
        os.environ[name] = value


def _strip_private(payload: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if not str(key).startswith("_")}


def _ensure_parents(repo_root: Path, mode: str) -> None:
    result = Path(repo_root) / plan.RESULT_ROOT_RELATIVE
    result.mkdir(parents=True, exist_ok=True)
    extra = plan.parent_relative_for_stage(mode)
    if extra is not None:
        (Path(repo_root) / extra).mkdir(parents=True, exist_ok=True)


def execute(
    repo_root: Path,
    *,
    mode: str,
    arm: str,
    gpu_index: int,
    gpu_authorized: bool,
    pack: bool = False,
    source_root: Path | None = None,
    held_in: bool = True,
) -> tuple[dict[str, str], str | None, str | None]:
    repo_root = Path(repo_root).absolute()
    source_root = Path(source_root).absolute() if source_root is not None else repo_root
    _require(os.environ.get("PYTHONNOUSERSITE") == "1", "execute requires PYTHONNOUSERSITE=1")
    plan.validate_mode_arm(mode, arm)
    _require(type(pack) is bool, "pack flag")
    receipts.refuse_sealed_roots(repo_root / plan.RESULT_ROOT_RELATIVE)
    gpu_info = gpu_module.require_launch(gpu_authorized, gpu_index, pack=pack)
    _bind_thread_env()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    _ensure_parents(repo_root, mode)
    relative = plan.stage_relative(mode, arm)
    progress_state: dict[str, object] = {"mode": mode, "arm": arm, "gpu_index": gpu_index}

    def launch_builder() -> dict[str, object]:
        return {
            "schema": "m1_tier12_pack_launch_v1",
            "mode": mode,
            "arm": arm,
            "status": "LAUNCHED",
            "device": "cuda:0",
            "gpu": gpu_info,
            "pack": pack,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cycle": list(plan.CYCLE),
            "cosine_50ep_forbidden": plan.COSINE_50EP_FORBIDDEN,
            "nwb_or_checkpoint_opened": False,
            "cuda_initialized": False,
        }

    def body_publisher(artifact: Any) -> dict[str, str]:
        if mode == "chunk_cdm":
            return _bodies_chunk_cdm(
                repo_root, source_root=source_root, arm=arm, artifact=artifact,
                progress=progress_state, held_in=held_in,
            )
        if mode == "train":
            return _bodies_train(
                repo_root, source_root=source_root, arm=arm, gpu_index=gpu_index,
                artifact=artifact, progress=progress_state,
            )
        return _bodies_score_trained(
            repo_root, source_root=source_root, arm=arm, artifact=artifact,
            progress=progress_state, held_in=held_in,
        )

    def terminal_builder(shas: Mapping[str, str]) -> dict[str, object]:
        return {
            "schema": "m1_tier12_pack_terminal_v1",
            "status": "COMPLETE",
            "mode": mode,
            "arm": arm,
            "gpu_index": gpu_index,
            "pack": pack,
            "formal_benchmark_verdict": False,
            "bodies": {key: value for key, value in shas.items() if not str(key).startswith("_")},
        }

    return receipts.run_stage0(
        repo_root / plan.RESULT_ROOT_RELATIVE,
        relative=relative,
        attempt_payload={
            "schema": "m1_tier12_pack_attempt_v1",
            "cell": plan.CELL,
            "phase": plan.PHASE,
            "mode": mode,
            "arm": arm,
            "status": "ATTEMPT_RESERVED",
            "gpu_index": gpu_index,
            "pack": pack,
            "pair_spec_sha256": plan.PairSpec().sha256,
            "data_or_model_accessed": False,
        },
        launch_builder=launch_builder,
        body_publisher=body_publisher,
        terminal_builder=terminal_builder,
        progress=lambda: dict(progress_state),
    )


def _bodies_chunk_cdm(
    repo_root: Path, *, source_root: Path, arm: str, artifact: Any,
    progress: dict[str, object], held_in: bool,
) -> dict[str, str]:
    from . import score as score_module

    binding = score_module.load_arm_binding(Path(repo_root), arm)
    model, state_sha = score_module.strict_load_arm_model(
        Path(repo_root), binding, device="cuda:0")
    sessions = list(plan.SCORE_ORDER) if held_in else list(plan.HELDOUT_FOLD_SESSIONS)
    if "20120924" in sessions:
        sessions = ["20120924", *[item for item in sessions if item != "20120924"]]
    opened: dict[str, Any] = {}
    cells: dict[str, dict[str, object]] = {}
    for session_id in sessions:
        opened[session_id] = score_module.open_session_dataset(
            Path(repo_root), Path(source_root), session_id)
        deployments = score_module.score_three_deployments(
            model, opened[session_id], device="cuda:0")
        for deployment, cell in deployments.items():
            key = f"{arm}_{deployment}_{session_id}"
            cell = dict(cell)
            cell.update({
                "schema": "m1_tier12_pack_score_cell_v1",
                "arm": arm, "deployment": deployment, "session_id": session_id,
                "arm_checkpoint_state_sha256": state_sha,
                "predecessor": "sealed_m1_t0c1_prefix_v1",
            })
            cells[key] = cell
            progress["cells_completed"] = sorted(cells)
    shas = score_module.publish_score_cells(artifact, arm=arm, cells=cells)
    table = score_module.build_arm_score_table(arm, cells)
    shas["table.json"] = artifact.publish_json("table.json", table)
    if arm == "t0":
        anchors = score_module.load_sealed_t0_anchors(Path(repo_root))
        shas["anchors.json"] = artifact.publish_json("anchors.json", {
            key: value for key, value in anchors.items() if key != "sealed_cells"
        })
        # Fail closed after the score JSON (and table/anchors) are published.
        score_module.check_t0_anchors(cells, anchors)
        rules = score_module.evaluate_read_rules(sealed=anchors, cells=cells)
        shas["read_rules.json"] = artifact.publish_json("read_rules.json", rules)
    return shas


def _bodies_train(
    repo_root: Path, *, source_root: Path, arm: str, gpu_index: int,
    artifact: Any, progress: dict[str, object],
) -> dict[str, str]:
    import torch

    from . import train as train_module

    spec = plan.ARM_SPECS[arm]
    profile = train_module.live_device_profile(torch, gpu_index)
    runner = train_module.ArmedM1Trainer(
        Path(repo_root), Path(source_root), arm, "cuda:0", profile, gpu_index=gpu_index,
    )
    source_authority = runner.prepare()
    progress["prepared"] = True
    prepared_steps = runner._prepared.paired_steps_per_epoch
    _require(prepared_steps == plan.STEPS_PER_EPOCH,
             "armed trainer paired steps/epoch drifted from the frozen 4951")
    result = runner.run(
        epoch_count=plan.EPOCHS, steps_per_epoch=prepared_steps, record_steps=plan.RECORD_STEPS,
    )
    progress["optimizer_steps_completed"] = result["optimizer_steps"]
    shas: dict[str, str] = {
        "source_authority.json": artifact.publish_json("source_authority.json", dict(source_authority)),
        "stream_head.json": artifact.publish_json("stream_head.json", result["stream_records"]),
        "training.json": artifact.publish_json("training.json", _strip_private(result)),
    }
    for epoch_row in result["epochs"]:
        shas[f"epoch_{epoch_row['epoch_index']:02d}.json"] = artifact.publish_json(
            f"epoch_{epoch_row['epoch_index']:02d}.json", epoch_row)
        _require("epoch_mean_source_train_loss" in epoch_row, "epoch_mean_source_train_loss missing")
    bodies = result["_checkpoint_bodies"]
    best_sha = artifact.publish_bytes("checkpoint_best_source_train_loss.pt", bodies["best"])
    last_sha = artifact.publish_bytes("checkpoint_last.pt", bodies["last"])
    checkpoints: dict[str, object] = {
        "best_source_train_loss": {
            "filename": "checkpoint_best_source_train_loss.pt", "sha256": best_sha,
            "state_sha256": result["best_checkpoint_state_sha256"], "strict_reload": True,
        },
        "last": {
            "filename": "checkpoint_last.pt", "sha256": last_sha,
            "state_sha256": result["final_model_state_sha256"], "strict_reload": True,
        },
    }
    if spec["swa"]:
        for index in plan.SWA_EPOCH_INDICES:
            name = f"epoch_{index}.pt"
            shas[name] = artifact.publish_bytes(name, bodies[f"epoch_{index}"])
            checkpoints[f"epoch_{index}"] = {"filename": name, "sha256": shas[name]}
        shas["swa_final4.pt"] = artifact.publish_bytes("swa_final4.pt", bodies["swa_final4"])
        checkpoints["swa_final4"] = {"filename": "swa_final4.pt", "sha256": shas["swa_final4.pt"]}
        if result.get("swa_manifest") is not None:
            shas["swa_manifest.json"] = artifact.publish_json("swa_manifest.json", result["swa_manifest"])
    manifest = {
        "schema": "m1_tier12_arm_checkpoint_manifest_v1",
        "arm": arm,
        "prefix_kind": spec["prefix_kind"],
        "identity_width": spec["identity_width"],
        "swa_enabled": bool(spec["swa"]),
        "checkpoints": checkpoints,
        "best_epoch_index": result["best_source_train_loss_epoch_index"],
        "last_epoch_index": plan.EPOCHS - 1,
        "swa_epoch_indices_0based": list(plan.SWA_EPOCH_INDICES) if spec["swa"] else [],
        "cosine_50ep_forbidden": True,
        "scheduler": None,
    }
    shas["checkpoint_manifest.json"] = artifact.publish_json("checkpoint_manifest.json", manifest)
    return shas


def _bodies_score_trained(
    repo_root: Path, *, source_root: Path, arm: str, artifact: Any,
    progress: dict[str, object], held_in: bool,
) -> dict[str, str]:
    from . import score as score_module

    path = score_module.trained_arm_checkpoint_path(Path(repo_root), arm)
    _require(path.is_file(), f"trained checkpoint absent: {path}")
    body = path.read_bytes()
    model, state_sha = score_module.strict_load_bytes(
        Path(repo_root), arm, body, device="cuda:0")
    sessions = list(plan.SCORE_ORDER) if held_in else list(plan.HELDOUT_FOLD_SESSIONS)
    if "20120924" in sessions:
        sessions = ["20120924", *[item for item in sessions if item != "20120924"]]
    cells: dict[str, dict[str, object]] = {}
    for session_id in sessions:
        opened = score_module.open_session_dataset(Path(repo_root), Path(source_root), session_id)
        deployments = score_module.score_three_deployments(model, opened, device="cuda:0")
        for deployment, cell in deployments.items():
            key = f"{arm}_{deployment}_{session_id}"
            cell = dict(cell)
            cell.update({
                "schema": "m1_tier12_pack_score_cell_v1",
                "arm": arm, "deployment": deployment, "session_id": session_id,
                "query_window": "full_session",
                "arm_checkpoint_state_sha256": state_sha,
                "checkpoint_filename": score_module.trained_checkpoint_filename(arm),
            })
            cells[key] = cell
            progress["cells_completed"] = sorted(cells)
        if session_id in plan.QUERY_Q10_END_SESSIONS:
            opened_q = score_module.restrict_opened_to_query(
                opened, start=plan.QUERY_Q10_END[0], stop=plan.QUERY_Q10_END[1])
            deployments_q = score_module.score_three_deployments(
                model, opened_q, device="cuda:0")
            for deployment, cell in deployments_q.items():
                key = f"{arm}_{deployment}_q10_end_{session_id}"
                cell = dict(cell)
                cell.update({
                    "schema": "m1_tier12_pack_score_cell_v1",
                    "arm": arm, "deployment": deployment, "session_id": session_id,
                    "query_window": "q10_end",
                    "query": list(opened_q["query"]),
                    "n_windows_full": opened_q["n_windows_full"],
                    "n_windows_query": opened_q["n_windows_query"],
                    "output_trial_min": opened_q["output_trial_min"],
                    "output_trial_max": opened_q["output_trial_max"],
                    "governing": False,
                    "arm_checkpoint_state_sha256": state_sha,
                    "checkpoint_filename": score_module.trained_checkpoint_filename(arm),
                })
                cells[key] = cell
                progress["cells_completed"] = sorted(cells)
    shas = score_module.publish_score_cells(artifact, arm=arm, cells=cells)
    table = score_module.build_arm_score_table(arm, cells)
    shas["table.json"] = artifact.publish_json("table.json", table)
    anchors = score_module.load_sealed_t0_anchors(Path(repo_root))
    rules = score_module.evaluate_read_rules(sealed=anchors, cells=cells)
    shas["read_rules.json"] = artifact.publish_json("read_rules.json", rules)
    return shas
