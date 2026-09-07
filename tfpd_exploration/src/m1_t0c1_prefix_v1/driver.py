"""Root-reviewed in-process stage drivers: smoke, t0, c1, phase3.

The public CLI cannot reach these; a root reviewer calls them from one
detached process, serially, on the single bound GPU.  Each stage reserves its
own fresh immutable root and publishes ``attempt`` before any NWB open,
checkpoint load, model construction, or CUDA forward.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from . import hook as hook_module
from . import phase3, plan, receipts, schedule, trainer


class DriverError(RuntimeError):
    """Fail closed for stage drivers."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DriverError(message)


def _json_bytes(value: object) -> bytes:
    return plan.canonical_json_bytes(value)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        "schema": "m1_t0c1_stage_launch_v1",
        "stage": stage,
        "status": "LAUNCHED",
        "device": device,
        "gpu_binding": dict(plan.BOUND_GPU),
        "dropout_proof": plan.DROPOUT_PROOF,
        "cycle_law": plan.CYCLE_LAW,
        "operator_resolution": plan.OPERATOR_RESOLUTION,
        "smoke_equality_contract": plan.SMOKE_EQUALITY_CONTRACT,
        "nwb_or_checkpoint_opened": False,
        "cuda_initialized": False,
    }


def _strip_private(payload: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def _verify_sealed_smoke(root: Path) -> None:
    """Fail closed unless the sealed smoke predecessor terminal matches."""
    import os as _os

    directory = Path(root).absolute() / plan.SMOKE_ROOT_RELATIVE
    terminal = directory / "terminal.json"
    try:
        info = _os.lstat(terminal)
    except OSError as error:
        raise DriverError("m1 t0c1 sealed smoke predecessor terminal is absent") from error
    import stat as _stat

    body = terminal.read_bytes()
    sidecar = (directory / "terminal.json.sha256").read_text(encoding="ascii")
    _require(_stat.S_ISREG(info.st_mode) and _stat.S_IMODE(info.st_mode) == 0o444
             and _sha(body) == plan.SEALED_SMOKE_TERMINAL_SHA256
             and sidecar == f"{plan.SEALED_SMOKE_TERMINAL_SHA256}  terminal.json\n",
             "m1 t0c1 sealed smoke predecessor terminal digest drift")
    payload = json.loads(body.decode("utf-8"))
    _require(payload.get("status") == plan.SEALED_SMOKE_STATUS,
             "m1 t0c1 sealed smoke predecessor status drift")


def _validate_smoke_equality(t0: Mapping[str, object], c1: Mapping[str, object]) -> dict[str, object]:
    t0_stream = dict(t0["stream_records"])
    c1_stream = dict(c1["stream_records"])
    t0_operator = dict(t0["operator_snapshot"])
    c1_operator = dict(c1["operator_snapshot"])
    checks = {
        "initial_model_state_sha256": t0["initial_model_state_sha256"] == c1["initial_model_state_sha256"],
        "rng_stream_digest": t0_stream["rng_stream_digest"] == c1_stream["rng_stream_digest"],
        "per_step_rng_digests": [
            row_t["rng_state_after_sha256"] == row_c["rng_state_after_sha256"]
            for row_t, row_c in zip(t0_stream["rows"], c1_stream["rows"], strict=True)
        ],
        "batch_stream_digest": t0_stream["batch_stream_digest"] == c1_stream["batch_stream_digest"],
        "t0_effective_prefixes_all_full": all(item == 10 for item in t0_operator["effective_prefixes"]),
        "c1_prefix_sequence_is_cycle": c1_operator["recorded_prefix_sequence"]
        == schedule.sequence(len(c1_operator["recorded_prefix_sequence"])),
        "t0_eval_dropout_inactive": t0["eval_dropout_inactive_proof"]["bit_identical"] is True,
        "c1_eval_dropout_inactive": c1["eval_dropout_inactive_proof"]["bit_identical"] is True,
        "optimizer_steps": t0["optimizer_steps"] == c1["optimizer_steps"] == plan.SMOKE_STEPS,
    }
    def _all_true(value: object) -> bool:
        if isinstance(value, list):
            return all(item is True for item in value)
        return value is True

    _require(all(_all_true(value) for value in checks.values()),
             f"m1 t0c1 smoke equality failed: {checks}")
    return {
        "schema": "m1_t0c1_smoke_equality_v1",
        "checks": checks,
        "t0_initial_model_state_sha256": t0["initial_model_state_sha256"],
        "c1_initial_model_state_sha256": c1["initial_model_state_sha256"],
        "shared_rng_stream_digest": t0_stream["rng_stream_digest"],
        "shared_batch_stream_digest": t0_stream["batch_stream_digest"],
        "c1_recorded_prefix_sequence": list(c1_operator["recorded_prefix_sequence"]),
        "c1_visible_slice_digests_first_step": (
            c1_operator["records"][0]["visible_slice_sha256"] if c1_operator["records"] else None),
        "dropout_p_stream_identical": True,
        "batch_order_identical": True,
        "cal_aug_discipline_reference": plan.SMOKE_EQUALITY_CONTRACT["cal_aug_discipline_reference"],
    }


def execute_smoke(root: Path, *, source_root: Path, device: str) -> tuple[str | None, str | None]:
    identity = _identity_payload("smoke", Path(root))
    progress_state: dict[str, object] = {"stages_completed": []}

    def bodies(artifact: Any) -> dict[str, str]:
        import torch

        profile = trainer.live_device_profile(torch)
        t0 = trainer.ArmedM1Trainer(Path(root), Path(source_root), "t0", device, profile)
        c1 = trainer.ArmedM1Trainer(Path(root), Path(source_root), "c1", device, profile)
        t0.prepare()
        progress_state["stages_completed"] = ["t0_prepare"]
        t0_result = t0.run(epoch_count=1, steps_per_epoch=plan.SMOKE_STEPS, record_steps=plan.SMOKE_STEPS)
        progress_state["stages_completed"] = ["t0_prepare", "t0_run"]
        c1.prepare()
        c1_result = c1.run(epoch_count=1, steps_per_epoch=plan.SMOKE_STEPS, record_steps=plan.SMOKE_STEPS)
        progress_state["stages_completed"] = ["t0_prepare", "t0_run", "c1_prepare", "c1_run"]
        equality = _validate_smoke_equality(t0_result, c1_result)
        return {
            "t0_smoke.json": artifact.publish_json("t0_smoke.json", _strip_private(t0_result)),
            "c1_smoke.json": artifact.publish_json("c1_smoke.json", _strip_private(c1_result)),
            "equality.json": artifact.publish_json("equality.json", equality),
        }

    def terminal(shas: Mapping[str, str]) -> dict[str, object]:
        return {
            "schema": "m1_t0c1_smoke_terminal_v1",
            "status": "COMPLETE_MATCHED_SMOKE_EQUALITY",
            "dropout_p_stream_identical": True,
            "batch_order_identical": True,
            "phase1_motivation": plan.PHASE1_MOTIVATION,
            "t0_smoke_sha256": shas["t0_smoke.json"],
            "c1_smoke_sha256": shas["c1_smoke.json"],
            "equality_sha256": shas["equality.json"],
        }

    attempt = receipts.stage_attempt_payload("smoke", plan.PairSpec().sha256, identity["closure"])
    attempt["identity"] = identity
    shas, terminal_sha, failure_sha = receipts.run_stage(
        Path(root), relative=plan.SMOKE_ROOT_RELATIVE, attempt_payload=attempt,
        launch_builder=lambda: _launch_payload("smoke", device), body_publisher=bodies,
        terminal_builder=terminal, expected_terminal_names=receipts.smoke_stage_names,
        progress=lambda: dict(progress_state),
    )
    return terminal_sha, failure_sha


def execute_arm(root: Path, *, source_root: Path, device: str, arm: str) -> tuple[str | None, str | None]:
    _require(arm in plan.ARMS, "driver arm drift")
    _verify_sealed_smoke(Path(root))
    identity = _identity_payload(arm, Path(root))
    progress_state: dict[str, object] = {"arm": arm, "optimizer_steps_completed": 0}

    def bodies(artifact: Any) -> dict[str, str]:
        import torch

        profile = trainer.live_device_profile(torch)
        runner = trainer.ArmedM1Trainer(Path(root), Path(source_root), arm, device, profile)
        source_authority = runner.prepare()
        progress_state["prepared"] = True
        prepared_steps = runner._prepared.paired_steps_per_epoch
        _require(prepared_steps == plan.STEPS_PER_EPOCH,
                 "armed trainer paired steps/epoch drifted from the frozen 4951")
        result = runner.run(epoch_count=plan.EPOCHS, steps_per_epoch=prepared_steps,
                            record_steps=plan.RECORD_STEPS)
        progress_state["optimizer_steps_completed"] = result["optimizer_steps"]
        shas: dict[str, str] = {
            "source_authority.json": artifact.publish_json("source_authority.json", source_authority),
            "stream_head.json": artifact.publish_json("stream_head.json", result["stream_records"]),
            "training.json": artifact.publish_json("training.json", _strip_private(result)),
        }
        for epoch_row in result["epochs"]:
            shas[f"epoch_{epoch_row['epoch_index']:02d}.json"] = artifact.publish_json(
                f"epoch_{epoch_row['epoch_index']:02d}.json", epoch_row)
        checkpoint_bodies = result["_checkpoint_bodies"]
        best_sha = artifact.publish_bytes(
            "checkpoint_best_source_train_loss.pt", checkpoint_bodies["best"])
        last_sha = artifact.publish_bytes("checkpoint_last.pt", checkpoint_bodies["last"])
        manifest = {
            "schema": "m1_t0c1_arm_checkpoint_manifest_v1",
            "arm": arm,
            "checkpoints": {
                "best_source_train_loss": {
                    "filename": "checkpoint_best_source_train_loss.pt", "sha256": best_sha,
                    "state_sha256": result["best_checkpoint_state_sha256"], "strict_reload": True,
                },
                "last": {
                    "filename": "checkpoint_last.pt", "sha256": last_sha,
                    "state_sha256": result["final_model_state_sha256"], "strict_reload": True,
                },
            },
            "best_epoch_index": result["best_source_train_loss_epoch_index"],
            "last_epoch_index": plan.EPOCHS - 1,
            "swa_enabled": False, "swa_artifact_forbidden": True,
        }
        shas["checkpoint_manifest.json"] = artifact.publish_json("checkpoint_manifest.json", manifest)
        shas["_manifest"] = manifest
        shas["_best_epoch_index"] = result["best_source_train_loss_epoch_index"]
        return shas

    def terminal(shas: Mapping[str, str]) -> dict[str, object]:
        manifest = dict(shas["_manifest"])  # type: ignore[index]
        return {
            "schema": "m1_t0c1_arm_terminal_v1",
            "status": "COMPLETE_MATCHED_ARM_TRAINING",
            "arm": arm,
            "arm_role": plan.ARM_ROLES[arm],
            "checkpoint_manifest": manifest,
            "best_epoch_index": shas["_best_epoch_index"],
            "dropout_proof": plan.DROPOUT_PROOF,
            "cycle_law": plan.CYCLE_LAW,
            "phase1_motivation": plan.PHASE1_MOTIVATION,
            "source_only": True,
            "swa_enabled": False,
            "target_optimizer_backward_update": 0,
        }

    attempt = receipts.stage_attempt_payload(arm, plan.PairSpec().sha256, identity["closure"])
    attempt["identity"] = identity
    shas, terminal_sha, failure_sha = receipts.run_stage(
        Path(root), relative=plan.ARM_ROOT_RELATIVE[arm], attempt_payload=attempt,
        launch_builder=lambda: _launch_payload(arm, device), body_publisher=bodies,
        terminal_builder=terminal, expected_terminal_names=receipts.arm_stage_names,
        progress=lambda: dict(progress_state),
    )
    return terminal_sha, failure_sha


def execute_phase3(root: Path, *, source_root: Path, device: str) -> tuple[str | None, str | None]:
    _verify_sealed_smoke(Path(root))
    identity = _identity_payload("phase3", Path(root))
    progress_state: dict[str, object] = {"cells_completed": []}

    def bodies(artifact: Any) -> dict[str, str]:
        bindings = {arm: phase3.load_arm_binding(Path(root), arm) for arm in plan.ARMS}
        opened = {}
        for session_id in plan.SCORE_ORDER:
            opened[session_id] = phase3.open_session_dataset(Path(root), Path(source_root), session_id)
        cells: dict[str, dict[str, object]] = {}
        shas: dict[str, str] = {}
        for arm in plan.ARMS:
            binding = bindings[arm]
            model, state_sha = phase3.strict_load_arm_model(
                Path(root), binding, device=device)
            for deployment, scorer in (("static_m10", phase3.score_static),
                                       ("cdm_activity_fifo_m10", phase3.score_cdm_fifo)):
                for session_id in plan.SCORE_ORDER:
                    cell = scorer(model, opened[session_id], device=device)
                    key = f"{arm}_{deployment}_{session_id}"
                    cell.update({
                        "schema": "m1_t0c1_phase3_cell_v1",
                        "arm": arm, "deployment": deployment, "session_id": session_id,
                        "arm_checkpoint_state_sha256": state_sha,
                        "arm_terminal_sha256": binding.terminal_sha256,
                    })
                    cells[key] = cell
                    shas[f"score_{key}.json"] = artifact.publish_json(f"score_{key}.json", cell)
                    progress_state["cells_completed"] = sorted(cells)
        table = phase3.build_table(cells)
        shas["table.json"] = artifact.publish_json("table.json", table)
        shas["_table"] = table
        return shas

    def terminal(shas: Mapping[str, str]) -> dict[str, object]:
        table = dict(shas["_table"])  # type: ignore[index]
        return {
            "schema": "m1_t0c1_phase3_terminal_v1",
            "status": "COMPLETE_PHASE3_TABLE",
            "table_sha256": shas["table.json"],
            "phase1_motivation": plan.PHASE1_MOTIVATION,
            "motivation_note": plan.PHASE1_MOTIVATION["note"],
            "metric": plan.METRIC_LABEL,
            "formal_benchmark_verdict": False,
        }

    attempt = receipts.stage_attempt_payload("phase3", plan.PairSpec().sha256, identity["closure"])
    attempt["identity"] = identity
    shas, terminal_sha, failure_sha = receipts.run_stage(
        Path(root), relative=plan.PHASE3_ROOT_RELATIVE, attempt_payload=attempt,
        launch_builder=lambda: _launch_payload("phase3", device), body_publisher=bodies,
        terminal_builder=terminal, expected_terminal_names=receipts.phase3_stage_names,
        progress=lambda: dict(progress_state),
    )
    return terminal_sha, failure_sha


__all__ = ("DriverError", "execute_smoke", "execute_arm", "execute_phase3")
