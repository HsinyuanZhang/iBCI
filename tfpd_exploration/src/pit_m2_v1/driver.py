"""Root-reviewed in-process stage drivers: smoke, t0m, c1m, phase3.

The public CLI cannot reach these; a root reviewer calls them from one
process.  Each stage reserves its own fresh immutable root and publishes
``attempt`` before any NWB open, checkpoint load, model construction, or
forward.  The train and phase3 drivers re-check the GPU authorization law
(the same refusal the CLI enforces before importing this module).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

from . import phase3, plan, receipts, smoke, trainer


class DriverError(RuntimeError):
    """Fail closed for stage drivers."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DriverError(message)


def _identity_payload(stage: str, root: Path) -> dict[str, object]:
    closure = plan.implementation_closure(Path(root))
    spec = plan.PairSpec()
    return {
        "pair_spec": spec.payload(),
        "pair_spec_sha256": spec.sha256,
        "closure": closure,
        "stage": stage,
    }


def _launch_payload(stage: str, device: str) -> dict[str, object]:
    return {
        "schema": "pit_m2_stage_launch_v1",
        "stage": stage,
        "status": "LAUNCHED",
        "device": device,
        "cycle_law": plan.CYCLE_LAW,
        "dropout_proof": plan.DROPOUT_PROOF,
        "gates": plan.GATES,
        "smoke_equality_contract": plan.SMOKE_EQUALITY_CONTRACT,
        "nwb_or_checkpoint_opened": False,
        "cuda_initialized": False,
    }


def _strip_private(payload: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def _verify_sealed_smoke(root: Path) -> None:
    """Fail closed unless the sealed CPU-smoke predecessor terminal matches."""
    import hashlib
    import json
    import os as _os
    import stat as _stat

    directory = Path(root).absolute() / plan.SEALED_SMOKE_ROOT_RELATIVE
    terminal = directory / "terminal.json"
    try:
        info = _os.lstat(terminal)
    except OSError as error:
        raise DriverError("pit m2 sealed smoke predecessor terminal is absent") from error
    body = terminal.read_bytes()
    sidecar = (directory / "terminal.json.sha256").read_text(encoding="ascii")
    _require(_stat.S_ISREG(info.st_mode) and _stat.S_IMODE(info.st_mode) == 0o444
             and hashlib.sha256(body).hexdigest() == plan.SEALED_SMOKE_TERMINAL_SHA256
             and sidecar == f"{plan.SEALED_SMOKE_TERMINAL_SHA256}  terminal.json\n",
             "pit m2 sealed smoke predecessor terminal digest drift")
    payload = json.loads(body.decode("utf-8"))
    _require(payload.get("status") == plan.SEALED_SMOKE_STATUS,
             "pit m2 sealed smoke predecessor status drift")


def execute_smoke(root: Path) -> tuple[str | None, str | None]:
    """The CPU matched-pair smoke (no CUDA at all)."""
    identity = _identity_payload("smoke", Path(root))
    receipts.ensure_lane_root(Path(root))
    progress_state: dict[str, object] = {"stages_completed": []}

    def bodies(artifact) -> dict[str, str]:
        progress_state["stages_completed"] = ["prepare"]
        t0m = smoke.run_arm_smoke(Path(root), "t0m", steps=plan.SMOKE_STEPS)
        progress_state["stages_completed"] = ["prepare", "t0m_run"]
        c1m = smoke.run_arm_smoke(Path(root), "c1m", steps=plan.SMOKE_STEPS)
        progress_state["stages_completed"] = ["prepare", "t0m_run", "c1m_run"]
        equality = smoke.validate_smoke_equality(t0m, c1m)
        return {
            "t0m_smoke.json": artifact.publish_json("t0m_smoke.json", dict(t0m)),
            "c1m_smoke.json": artifact.publish_json("c1m_smoke.json", dict(c1m)),
            "equality.json": artifact.publish_json("equality.json", equality),
        }

    def terminal(shas: Mapping[str, str]) -> dict[str, object]:
        return {
            "schema": "pit_m2_smoke_terminal_v1",
            "status": "COMPLETE_PIT_M2_MATCHED_SMOKE_EQUALITY",
            "dropout_p_stream_identical": True,
            "batch_order_identical": True,
            "t4_bytes_unchanged": True,
            "cuda_initialized": False,
            "target_path_resolved": False,
            "t0m_smoke_sha256": shas["t0m_smoke.json"],
            "c1m_smoke_sha256": shas["c1m_smoke.json"],
            "equality_sha256": shas["equality.json"],
        }

    attempt = receipts.stage_attempt_payload("smoke", plan.PairSpec().sha256,
                                             identity["closure"])
    attempt["identity"] = identity
    shas, terminal_sha, failure_sha = receipts.run_stage(
        Path(root), relative=plan.SMOKE_ROOT_RELATIVE, attempt_payload=attempt,
        launch_builder=lambda: _launch_payload("smoke", "cpu"), body_publisher=bodies,
        terminal_builder=terminal, expected_terminal_names=receipts.smoke_stage_names,
        progress=lambda: dict(progress_state),
    )
    return terminal_sha, failure_sha


def execute_arm(root: Path, *, arm: str, gpu_index: int,
                gpu_authorized: bool,
                cmdline_runner: Callable[[list[str]], str] | None = None
                ) -> tuple[str | None, str | None]:
    """One full 12-epoch arm of the matched pair on the authorized GPU."""
    _require(arm in plan.ARMS, "driver arm drift")
    _verify_sealed_smoke(Path(root))
    authorization = trainer.require_train_authorization(
        gpu_authorized=gpu_authorized, gpu_index=gpu_index, cmdline_runner=cmdline_runner)
    identity = _identity_payload(arm, Path(root))
    receipts.ensure_lane_root(Path(root))
    progress_state: dict[str, object] = {"arm": arm, "optimizer_steps_completed": 0}

    def bodies(artifact) -> dict[str, str]:
        import torch

        profile = trainer.live_device_profile(torch, gpu_index)
        progress_state["device_profile"] = profile
        runner = trainer.PitM2ArmedRunner(Path(root), arm, device="cuda:0")
        authority = runner.prepare()
        progress_state["prepared"] = True
        result = runner.run_full()
        progress_state["optimizer_steps_completed"] = result["optimizer_steps"]
        shas: dict[str, str] = {
            "source_authority.json": artifact.publish_json("source_authority.json", authority),
            "stream_head.json": artifact.publish_json("stream_head.json",
                                                      result["stream_records"]),
            "training.json": artifact.publish_json("training.json", _strip_private(result)),
        }
        for epoch_row in result["epochs"]:
            shas[f"epoch_{epoch_row['epoch_index']:02d}.json"] = artifact.publish_json(
                f"epoch_{epoch_row['epoch_index']:02d}.json", epoch_row)
        checkpoint_bodies = result["_checkpoint_bodies"]
        _require(result["best_epoch_index"] is not None,
                 "pit m2 arm produced no validation epoch row (metric tracker drift)")
        best_sha = artifact.publish_bytes("checkpoint_best.pt", checkpoint_bodies["best"])
        last_sha = artifact.publish_bytes("checkpoint_last.pt", checkpoint_bodies["last"])
        manifest = {
            "schema": "pit_m2_arm_checkpoint_manifest_v1",
            "arm": arm,
            "selected_by": plan.CHECKPOINT_MONITOR,
            "mode": plan.CHECKPOINT_MODE,
            "checkpoints": {
                "best": {
                    "filename": "checkpoint_best.pt", "sha256": best_sha,
                    "state_sha256": result["epochs"][int(result["best_epoch_index"])]
                    ["student_state_sha256"],
                    "epoch_index": result["best_epoch_index"],
                },
                "last": {
                    "filename": "checkpoint_last.pt", "sha256": last_sha,
                    "state_sha256": result["final_student_state_sha256"],
                    "epoch_index": plan.EPOCHS - 1,
                },
            },
            "swa_artifact_forbidden": True,
        }
        shas["checkpoint_manifest.json"] = artifact.publish_json(
            "checkpoint_manifest.json", manifest)
        shas["_manifest"] = manifest
        return shas

    def terminal(shas: Mapping[str, str]) -> dict[str, object]:
        return {
            "schema": "pit_m2_arm_terminal_v1",
            "status": "COMPLETE_PIT_M2_MATCHED_ARM_TRAINING",
            "arm": arm,
            "arm_role": plan.ARM_ROLES[arm],
            "checkpoint_manifest": dict(shas["_manifest"]),  # type: ignore[index]
            "cycle_law": plan.CYCLE_LAW,
            "dropout_proof": plan.DROPOUT_PROOF,
            "authorization": authorization,
        }

    attempt = receipts.stage_attempt_payload(arm, plan.PairSpec().sha256,
                                             identity["closure"])
    attempt["identity"] = identity
    shas, terminal_sha, failure_sha = receipts.run_stage(
        Path(root), relative=plan.ARM_ROOT_RELATIVE[arm], attempt_payload=attempt,
        launch_builder=lambda: {**_launch_payload(arm, "cuda:0"),
                                "gpu_authorization": authorization},
        body_publisher=bodies, terminal_builder=terminal,
        expected_terminal_names=receipts.arm_stage_names,
        progress=lambda: dict(progress_state),
    )
    return terminal_sha, failure_sha


def execute_phase3(root: Path, *, gpu_index: int, gpu_authorized: bool,
                   cmdline_runner: Callable[[list[str]], str] | None = None
                   ) -> tuple[str | None, str | None]:
    """The 2x2 scoring: fresh arms on both paths, sealed rows referenced."""
    _verify_sealed_smoke(Path(root))
    authorization = trainer.require_train_authorization(
        gpu_authorized=gpu_authorized, gpu_index=gpu_index, cmdline_runner=cmdline_runner)
    identity = _identity_payload("phase3", Path(root))
    receipts.ensure_lane_root(Path(root))
    progress_state: dict[str, object] = {"cells_completed": []}

    def bodies(artifact) -> dict[str, str]:
        import json

        import numpy as np

        import torch

        from . import phase3 as scoring

        trainer.ensure_streaming_paths(Path(root))
        profile = trainer.live_device_profile(torch, gpu_index)
        device = torch.device("cuda:0")
        anchors = scoring.load_sealed_anchors(Path(root))
        shas: dict[str, str] = {}
        fresh: dict[str, dict[str, object]] = {}
        for arm in plan.ARMS:
            manifest_path = (Path(root).absolute() / plan.ARM_ROOT_RELATIVE[arm]
                             / "checkpoint_manifest.json")
            _require(manifest_path.is_file(), f"pit m2 phase3 missing arm manifest: {arm}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            best_relative = (plan.ARM_ROOT_RELATIVE[arm] + "/"
                             + manifest["checkpoints"]["best"]["filename"])
            best_path = Path(root).absolute() / best_relative
            digest = plan.sha256_file(best_path)
            _require(digest == manifest["checkpoints"]["best"]["sha256"],
                     f"pit m2 phase3 arm checkpoint drift: {arm}")
            runner = trainer.PitM2ArmedRunner(Path(root), arm, device="cuda:0")
            runner.prepare(attach_operator=False)
            state = torch.load(best_path, map_location="cpu", weights_only=True)
            runner._litmodule.load_state_dict(state, strict=True)
            litmodule = runner._litmodule.to(device).eval()
            for parameter in litmodule.parameters():
                parameter.requires_grad_(False)
            datamodule = runner._datamodule
            datamodule.setup("test")
            datasets = {
                "within_post30": datamodule.train_dataset,
                "external_official_query": datamodule.val_heldout_dataset,
                "external_post30_local": datamodule.val_heldout_dataset,
            }
            raw_sessions = {
                "within_post30": datamodule.train_calib_heldin_sessions,
                "external_official_query": datamodule.val_calib_heldout_sessions,
                "external_post30_local": datamodule.val_calib_heldout_sessions,
            }
            normalization = datamodule.native_t4_normalization
            side_mean = np.asarray(normalization["mean"], dtype=np.float32)
            side_std = np.asarray(normalization["std"], dtype=np.float32)
            for path, scorer in (("static", scoring.score_static_session),
                                 ("fifo", scoring.score_fifo_session)):
                for surface in plan.PHASE3_SURFACES[path]:
                    dataset = datasets[surface]
                    _require(dataset is not None, f"pit m2 phase3 dataset missing: {surface}")
                    observed = tuple(sorted(dataset.calib_trialized_neural_features))
                    _require(observed == plan.SURFACE_SESSION_ROSTERS[surface],
                             f"pit m2 phase3 {surface} roster drift: {observed}")
                    rows: dict[str, float] = {}
                    for session in observed:
                        if path == "static":
                            row = scorer(torch=torch, student=litmodule.student,
                                         dataset=dataset, session=session, surface=surface,
                                         device=device)
                        else:
                            row = scorer(torch=torch, litmodule=litmodule,
                                         dataset=dataset,
                                         raw_session=raw_sessions[surface][session],
                                         session=session, side_mean=side_mean,
                                         side_std=side_std, device=device)
                        row.update({"arm": arm, "surface": surface})
                        # ONE leaf per (arm, path, surface, session): the
                        # attempt-1 bug reused the surface-level leaf across
                        # sessions and collided O_EXCL on the second session.
                        name = f"score_{arm}_{path}_{surface}_{session}.json"
                        shas[name] = artifact.publish_json(name, row)
                        rows[session] = float(row["r2"])
                        progress_state["cells_completed"] = sorted(
                            set(list(progress_state["cells_completed"]) + [name]))
                    summary = scoring.summarize_cells(rows)
                    # Key by (arm, path, surface): the attempt-3 bug keyed by
                    # (arm, path) alone and the within summary clobbered the
                    # external one before build_table.
                    fresh[f"{arm}_{path}_{surface}"] = summary
        table = scoring.build_table(anchors, fresh)
        table["device_profile"] = profile
        gates = scoring.evaluate_gates(table)
        shas["table.json"] = artifact.publish_json("table.json", table)
        shas["_gates"] = gates
        return shas

    def terminal(shas: Mapping[str, str]) -> dict[str, object]:
        return {
            "schema": "pit_m2_phase3_terminal_v1",
            "status": "COMPLETE_PIT_M2_PHASE3_TABLE",
            "table_sha256": shas["table.json"],
            "gates": dict(shas["_gates"]),  # type: ignore[index]
            "formal_benchmark_verdict": False,
        }

    attempt = receipts.stage_attempt_payload("phase3", plan.PairSpec().sha256,
                                             identity["closure"])
    attempt["identity"] = identity
    shas, terminal_sha, failure_sha = receipts.run_stage(
        Path(root), relative=plan.PHASE3_ROOT_RELATIVE, attempt_payload=attempt,
        launch_builder=lambda: {**_launch_payload("phase3", "cuda:0"),
                                "gpu_authorization": authorization},
        body_publisher=bodies, terminal_builder=terminal,
        expected_terminal_names=receipts.phase3_stage_names,
        progress=lambda: dict(progress_state),
    )
    return terminal_sha, failure_sha


__all__ = ("DriverError", "execute_smoke", "execute_arm", "execute_phase3")
