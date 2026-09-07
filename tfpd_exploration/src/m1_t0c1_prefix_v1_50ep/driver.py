"""Root-reviewed in-process stage drivers for the 50-epoch pair.

The public CLI cannot reach these; a root reviewer calls them from a
detached process on physical GPU 1.  Stages t0 and c1 may share the card
with each other; smoke, probe, and phase3 still require an exclusive card.
Each stage reserves its own fresh immutable root and publishes ``attempt``
before any NWB open, checkpoint load, model construction, or CUDA forward.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tfpd_exploration.src.m1_t0c1_prefix_v1 import schedule

from . import phase3, plan, probe, receipts, trainer


class DriverError(RuntimeError):
    """Fail closed for 50-epoch stage drivers."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DriverError(message)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity_payload(stage: str, root: Path) -> dict[str, object]:
    closure = plan.implementation_closure(Path(root))
    spec = plan.PairSpec50()
    return {
        "pair_spec": spec.payload(),
        "pair_spec_sha256": spec.sha256,
        "closure": closure,
        "stage": stage,
    }


def _launch_payload(
    stage: str, device: str, *, concurrent_sibling_stage: str | None = None,
) -> dict[str, object]:
    payload = {
        "schema": "m1_t0c1_prefix_v1_50ep_stage_launch_v1",
        "stage": stage,
        "status": "LAUNCHED",
        "device": device,
        "gpu_binding": dict(plan.BOUND_GPU),
        "dropout_proof": plan.DROPOUT_PROOF,
        "cycle_law": plan.CYCLE_LAW,
        "lr_schedule_law": dict(plan.LR_SCHEDULE_LAW),
        "launch_envelope": dict(plan.LAUNCH_ENVELOPE),
        "device_identity_law": dict(plan.DEVICE_IDENTITY_LAW),
        "operator_resolution": plan.OPERATOR_RESOLUTION,
        "arm_concurrency_law": dict(plan.ARM_CONCURRENCY_LAW),
        "spint_resource_parity": dict(plan.SPINT_RESOURCE_PARITY),
        "smoke_equality_contract": plan.SMOKE_EQUALITY_CONTRACT,
        "nwb_or_checkpoint_opened": False,
        "cuda_initialized": False,
    }
    if stage in plan.CONCURRENT_ARM_STAGES:
        payload.update(receipts.arm_concurrency_receipt(
            stage, concurrent_sibling_stage))
    if stage in {"probe", "phase3"}:
        payload["val_heldout_access_law"] = dict(plan.VAL_HELDOUT_ACCESS_LAW)
        payload["frozen_val_heldout_body_sha256"] = dict(plan.VAL_HELDOUT_BODY_SHA256)
        payload["observed_val_heldout_body_sha256"] = None
        payload["observed_body_sha256_published_after_open"] = True
        payload["labels_used_for"] = "metric only"
        payload["training_use"] = False
        payload["gradient_updates"] = 0
        payload["optimizer_steps"] = 0
    return payload


def require_gpu1_profile() -> dict[str, object]:
    """Query physical GPU 1 via this lane's profiler; refuse GPU 0.

    Smoke, arms, probe, and phase3 all call this so every GPU-touching stage
    records the GPU-1 UUID/PCI.  CPU-only tests must monkeypatch
    ``trainer.live_device_profile`` rather than invoking the real nvidia-smi.
    """
    import torch

    profile = trainer.live_device_profile(torch)
    payload = dict(profile.payload())
    _require(payload.get("uuid") == plan.DEVICE_IDENTITY_LAW["required_uuid"],
             "gpu1 profile uuid drifted from DEVICE_IDENTITY_LAW")
    _require(payload.get("uuid") != plan.DEVICE_IDENTITY_LAW["refused_uuid_gpu0"],
             "gpu1 profile resolved the refused GPU-0 uuid")
    _require(payload.get("pci_bus_id") == plan.DEVICE_IDENTITY_LAW["required_pci_bus_id"],
             "gpu1 profile pci.bus_id drifted")
    return payload


def overlay_source_authority(fragment: Mapping[str, object]) -> dict[str, object]:
    """Shadow the inherited 20-epoch prepare() fields with the 50-epoch law.

    ``prepare()`` is inherited from the sealed trainer and still reports
    ``epoch_budget=20``, ``scheduler=None``, ``lr=1e-5``.  Those numbers do
    not describe this lane; the overlay names each stale key and its true
    value so a reader cannot quote them as this run's recipe.
    """
    _require(isinstance(fragment, Mapping), "source_authority fragment must be a mapping")
    payload = dict(fragment)
    overlay = dict(plan.SUPERSEDED_BY_50EP_LAW)
    stale = overlay["stale_keys"]
    _require(isinstance(stale, Mapping)
             and set(stale) == {"epoch_budget", "scheduler", "adam_lr"},
             "superseded_by_50ep_law stale-key topology drifted")
    _require(stale["epoch_budget"]["inherited_value"] == 20
             and stale["epoch_budget"]["true_value"] == plan.EPOCHS == 50,
             "epoch_budget overlay drifted from the 50-epoch law")
    _require(stale["scheduler"]["inherited_value"] == "None"
             and stale["scheduler"]["true_value"] == plan.LR_SCHEDULE_KIND,
             "scheduler overlay drifted from the 50-epoch law")
    _require(stale["adam_lr"]["inherited_value"] == 1e-5,
             "adam_lr overlay inherited_value drifted from 1e-5")
    _require(overlay["true_epochs"] == 50
             and overlay["true_total_optimizer_steps"] == 247550
             and overlay["true_lr_schedule_kind"] == plan.LR_SCHEDULE_KIND,
             "superseded_by_50ep_law true values drifted")
    payload["superseded_by_50ep_law"] = overlay
    _require(payload["superseded_by_50ep_law"]["stale_keys"]["epoch_budget"]["true_value"] == 50,
             "source_authority overlay failed to shadow epoch_budget")
    return payload


def _strip_private(payload: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def _verify_this_lane_terminal(
    root: Path, relative: str, *, expected_status: str, label: str,
) -> str:
    """Fail closed unless this lane's terminal is a 0444 sidecar-consistent OK file.

    Unlike the 20-epoch driver, there is no plan.py hex literal to compare
    against: this lane's digest is discovered at run time.
    """
    import os as _os
    import stat as _stat

    directory = Path(root).absolute() / relative
    terminal = directory / "terminal.json"
    try:
        info = _os.lstat(terminal)
    except OSError as error:
        raise DriverError(f"m1 t0c1 50ep {label} terminal is absent") from error
    body = terminal.read_bytes()
    sidecar = (directory / "terminal.json.sha256").read_text(encoding="ascii")
    digest = _sha(body)
    _require(_stat.S_ISREG(info.st_mode) and _stat.S_IMODE(info.st_mode) == 0o444
             and sidecar == f"{digest}  terminal.json\n",
             f"m1 t0c1 50ep {label} terminal digest/mode/sidecar drift")
    payload = json.loads(body.decode("utf-8"))
    _require(payload.get("status") == expected_status,
             f"m1 t0c1 50ep {label} terminal status drift")
    return digest


def _verify_this_lane_smoke(root: Path) -> str:
    return _verify_this_lane_terminal(
        Path(root), plan.SMOKE_ROOT_RELATIVE,
        expected_status=plan.THIS_LANE_SMOKE_STATUS, label="smoke",
    )


def _verify_this_lane_arms(root: Path) -> dict[str, str]:
    return {
        arm: _verify_this_lane_terminal(
            Path(root), plan.ARM_ROOT_RELATIVE[arm],
            expected_status=plan.THIS_LANE_ARM_STATUS, label=f"arm {arm}",
        )
        for arm in plan.ARMS
    }


def _verify_this_lane_probe(root: Path) -> str:
    return _verify_this_lane_terminal(
        Path(root), plan.PROBE_ROOT_RELATIVE,
        expected_status=plan.THIS_LANE_PROBE_STATUS, label="probe",
    )


def _validate_smoke_equality(t0: Mapping[str, object], c1: Mapping[str, object]) -> dict[str, object]:
    from . import lr_schedule

    t0_stream = dict(t0["stream_records"])
    c1_stream = dict(c1["stream_records"])
    t0_operator = dict(t0["operator_snapshot"])
    c1_operator = dict(c1["operator_snapshot"])
    expected_lr = [lr_schedule.lr_at(index) for index in range(plan.SMOKE_STEPS)]
    t0_lr = [float(item) for item in t0["lr_per_step"]]  # type: ignore[arg-type]
    c1_lr = [float(item) for item in c1["lr_per_step"]]  # type: ignore[arg-type]
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
        "lr_sequence_identical_across_arms": t0_lr == c1_lr,
        "lr_sequence_equals_law": t0_lr == expected_lr and c1_lr == expected_lr,
        "lr_stream_digest_identical": t0["lr_stream_digest"] == c1["lr_stream_digest"]
        == lr_schedule.sequence_digest(expected_lr),
    }

    def _all_true(value: object) -> bool:
        if isinstance(value, list):
            return all(item is True for item in value)
        return value is True

    _require(all(_all_true(value) for value in checks.values()),
             f"m1 t0c1 50ep smoke equality failed: {checks}")
    return {
        "schema": "m1_t0c1_prefix_v1_50ep_smoke_equality_v1",
        "checks": checks,
        "t0_initial_model_state_sha256": t0["initial_model_state_sha256"],
        "c1_initial_model_state_sha256": c1["initial_model_state_sha256"],
        "shared_rng_stream_digest": t0_stream["rng_stream_digest"],
        "shared_batch_stream_digest": t0_stream["batch_stream_digest"],
        "shared_lr_stream_digest": t0["lr_stream_digest"],
        "shared_lr_per_step": expected_lr,
        "c1_recorded_prefix_sequence": list(c1_operator["recorded_prefix_sequence"]),
        "c1_visible_slice_digests_first_step": (
            c1_operator["records"][0]["visible_slice_sha256"] if c1_operator["records"] else None),
        "dropout_p_stream_identical": True,
        "batch_order_identical": True,
        "lr_sequence_identical": True,
        "cal_aug_discipline_reference": plan.SMOKE_EQUALITY_CONTRACT["cal_aug_discipline_reference"],
    }


def execute_smoke(root: Path, *, source_root: Path, device: str) -> tuple[str | None, str | None]:
    identity = _identity_payload("smoke", Path(root))
    progress_state: dict[str, object] = {"stages_completed": []}

    def bodies(artifact: Any) -> dict[str, str]:
        import torch

        profile = trainer.live_device_profile(torch)
        t0 = trainer.ArmedM1Trainer50(Path(root), Path(source_root), "t0", device, profile)
        c1 = trainer.ArmedM1Trainer50(Path(root), Path(source_root), "c1", device, profile)
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
            "schema": "m1_t0c1_prefix_v1_50ep_smoke_terminal_v1",
            "status": plan.THIS_LANE_SMOKE_STATUS,
            "dropout_p_stream_identical": True,
            "batch_order_identical": True,
            "lr_sequence_identical": True,
            "phase1_motivation": plan.PHASE1_MOTIVATION,
            "t0_smoke_sha256": shas["t0_smoke.json"],
            "c1_smoke_sha256": shas["c1_smoke.json"],
            "equality_sha256": shas["equality.json"],
        }

    attempt = receipts.stage_attempt_payload("smoke", plan.PairSpec50().sha256, identity["closure"])
    attempt["identity"] = identity
    _shas, terminal_sha, failure_sha = receipts.run_stage(
        Path(root), relative=plan.SMOKE_ROOT_RELATIVE, attempt_payload=attempt,
        launch_builder=lambda: _launch_payload("smoke", device), body_publisher=bodies,
        terminal_builder=terminal, expected_terminal_names=receipts.smoke_stage_names,
        progress=lambda: dict(progress_state),
    )
    return terminal_sha, failure_sha


def execute_arm(
    root: Path, *, source_root: Path, device: str, arm: str,
    concurrent_sibling_stage: str | None = None,
) -> tuple[str | None, str | None]:
    _require(arm in plan.ARMS, "driver arm drift")
    _verify_this_lane_smoke(Path(root))
    identity = _identity_payload(arm, Path(root))
    concurrency = receipts.arm_concurrency_receipt(arm, concurrent_sibling_stage)
    progress_state: dict[str, object] = {"arm": arm, "optimizer_steps_completed": 0}

    def bodies(artifact: Any) -> dict[str, str]:
        import torch

        profile = trainer.live_device_profile(torch)
        runner = trainer.ArmedM1Trainer50(Path(root), Path(source_root), arm, device, profile)
        source_authority = overlay_source_authority(runner.prepare())
        _require("superseded_by_50ep_law" in source_authority
                 and source_authority["superseded_by_50ep_law"]["stale_keys"]["epoch_budget"]["true_value"]
                 == 50,
                 "source_authority missing superseded_by_50ep_law overlay")
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
        epoch_by_index = {int(row["epoch_index"]): row for row in result["epochs"]}  # type: ignore[arg-type]
        for epoch_row in result["epochs"]:  # type: ignore[union-attr]
            shas[f"epoch_{epoch_row['epoch_index']:02d}.json"] = artifact.publish_json(
                f"epoch_{epoch_row['epoch_index']:02d}.json", epoch_row)
        checkpoint_bodies = result["_checkpoint_bodies"]
        epoch_bodies = result["_checkpoint_epoch_bodies"]
        epoch_state = result["checkpoint_epoch_state_sha256"]
        best_sha = artifact.publish_bytes(
            plan.BEST_CHECKPOINT_FILENAME, checkpoint_bodies["best"])
        checkpoints: dict[str, object] = {
            "best_source_train_loss": {
                "filename": plan.BEST_CHECKPOINT_FILENAME, "sha256": best_sha,
                "state_sha256": result["best_checkpoint_state_sha256"], "strict_reload": True,
            },
        }
        for epoch_index in plan.CHECKPOINT_EPOCH_INDICES:
            filename = plan.epoch_checkpoint_filename(epoch_index)
            file_sha = artifact.publish_bytes(filename, epoch_bodies[epoch_index])
            epoch_row = epoch_by_index[epoch_index]
            checkpoints[f"epoch_{epoch_index:02d}"] = {
                "filename": filename, "sha256": file_sha,
                "state_sha256": epoch_state[epoch_index],
                "epoch_index_0based": epoch_index,
                "epoch_index_1based": epoch_index + 1,
                "epoch_mean_source_train_loss": epoch_row["epoch_mean_source_train_loss"],
                "lr_at_epoch_final_step": epoch_row["lr_at_epoch_final_step"],
                "strict_reload": True,
            }
        manifest = {
            "schema": "m1_t0c1_prefix_v1_50ep_arm_checkpoint_manifest_v1",
            "arm": arm,
            "checkpoints": checkpoints,
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
            "schema": "m1_t0c1_prefix_v1_50ep_arm_terminal_v1",
            "status": plan.THIS_LANE_ARM_STATUS,
            "arm": arm,
            "arm_role": plan.ARM_ROLES[arm],
            "checkpoint_manifest": manifest,
            "best_epoch_index": shas["_best_epoch_index"],
            "dropout_proof": plan.DROPOUT_PROOF,
            "cycle_law": plan.CYCLE_LAW,
            "lr_schedule_law": dict(plan.LR_SCHEDULE_LAW),
            "phase1_motivation": plan.PHASE1_MOTIVATION,
            "source_only": True,
            "swa_enabled": False,
            "target_optimizer_backward_update": 0,
            **concurrency,
        }

    attempt = receipts.stage_attempt_payload(
        arm, plan.PairSpec50().sha256, identity["closure"],
        concurrent_sibling_stage=concurrent_sibling_stage,
    )
    attempt["identity"] = identity
    _shas, terminal_sha, failure_sha = receipts.run_stage(
        Path(root), relative=plan.ARM_ROOT_RELATIVE[arm], attempt_payload=attempt,
        launch_builder=lambda: _launch_payload(
            arm, device, concurrent_sibling_stage=concurrent_sibling_stage),
        body_publisher=bodies,
        terminal_builder=terminal, expected_terminal_names=receipts.arm_stage_names,
        progress=lambda: dict(progress_state),
    )
    return terminal_sha, failure_sha


def execute_probe(root: Path, *, source_root: Path, device: str) -> tuple[str | None, str | None]:
    _verify_this_lane_smoke(Path(root))
    _verify_this_lane_arms(Path(root))
    identity = _identity_payload("probe", Path(root))
    progress_state: dict[str, object] = {"cells_completed": []}

    def bodies(artifact: Any) -> dict[str, str]:
        device_profile = require_gpu1_profile()
        result = probe.run_probe(Path(root), source_root=Path(source_root), device=device)
        result["device_profile"] = device_profile
        shas: dict[str, str] = {}
        for key, cell in result["cells"].items():
            name = f"score_{key}.json"
            shas[name] = artifact.publish_json(name, cell)
            progress_state["cells_completed"] = sorted(result["cells"])
        shas["opened_sessions.json"] = artifact.publish_json(
            "opened_sessions.json", result["opened_sessions"])
        shas["curve.json"] = artifact.publish_json("curve.json", result["curve"])
        shas["verdict.json"] = artifact.publish_json("verdict.json", result["verdict_by_arm"])
        shas["selection.json"] = artifact.publish_json("selection.json", result["selection"])
        shas["_result"] = result
        return shas

    def terminal(shas: Mapping[str, str]) -> dict[str, object]:
        result = dict(shas["_result"])  # type: ignore[index]
        return {
            "schema": "m1_t0c1_prefix_v1_50ep_probe_terminal_v1",
            "status": plan.THIS_LANE_PROBE_STATUS,
            "curve_sha256": shas["curve.json"],
            "verdict_sha256": shas["verdict.json"],
            "selection_sha256": shas["selection.json"],
            "opened_sessions_sha256": shas["opened_sessions.json"],
            "selected_epoch_index_by_arm": result["selection"]["selected_epoch_index_by_arm"],
            "selection_surface": plan.SELECTION_SURFACE,
            "verdict_by_arm": {
                arm: {
                    "val_heldout": payload["val_heldout"]["verdict"],
                    "test_fold": payload["test_fold"]["verdict"],
                }
                for arm, payload in result["verdict_by_arm"].items()
            },
            "val_heldout_access_law_sha256": plan.VAL_HELDOUT_ACCESS_LAW["law_sha256"],
            "frozen_val_heldout_body_sha256": result["frozen_val_heldout_body_sha256"],
            "observed_val_heldout_body_sha256": result["observed_val_heldout_body_sha256"],
            "device_profile": result["device_profile"],
            "phase1_motivation": plan.PHASE1_MOTIVATION,
            "metric": plan.METRIC_LABEL,
        }

    attempt = receipts.stage_attempt_payload("probe", plan.PairSpec50().sha256, identity["closure"])
    attempt["identity"] = identity
    _shas, terminal_sha, failure_sha = receipts.run_stage(
        Path(root), relative=plan.PROBE_ROOT_RELATIVE, attempt_payload=attempt,
        launch_builder=lambda: _launch_payload("probe", device), body_publisher=bodies,
        terminal_builder=terminal, expected_terminal_names=receipts.probe_stage_names,
        progress=lambda: dict(progress_state),
    )
    return terminal_sha, failure_sha


def execute_phase3(root: Path, *, source_root: Path, device: str) -> tuple[str | None, str | None]:
    _verify_this_lane_smoke(Path(root))
    _verify_this_lane_arms(Path(root))
    _verify_this_lane_probe(Path(root))
    identity = _identity_payload("phase3", Path(root))
    progress_state: dict[str, object] = {"cells_completed": []}

    def bodies(artifact: Any) -> dict[str, str]:
        device_profile = require_gpu1_profile()
        selected = probe.load_selection(Path(root))
        result = phase3.run_phase3(
            Path(root), source_root=Path(source_root), device=device, selected_by_arm=selected,
        )
        result["device_profile"] = device_profile
        shas: dict[str, str] = {}
        for label, cells in (("epoch50", result["cells_epoch50"]),
                             ("selected", result["cells_selected"])):
            for key, cell in cells.items():
                name = f"score_{label}_{key}.json"
                shas[name] = artifact.publish_json(name, cell)
                progress_state["cells_completed"] = list(progress_state["cells_completed"]) + [name]  # type: ignore[operator]
        shas["table_epoch50.json"] = artifact.publish_json("table_epoch50.json", result["table_epoch50"])
        shas["table_selected.json"] = artifact.publish_json("table_selected.json", result["table_selected"])
        shas["reference_delta.json"] = artifact.publish_json("reference_delta.json", result["reference_delta"])
        shas["_result"] = result
        return shas

    def terminal(shas: Mapping[str, str]) -> dict[str, object]:
        result = dict(shas["_result"])  # type: ignore[index]
        return {
            "schema": "m1_t0c1_prefix_v1_50ep_phase3_terminal_v1",
            "status": plan.THIS_LANE_PHASE3_STATUS,
            "table_epoch50_sha256": shas["table_epoch50.json"],
            "table_selected_sha256": shas["table_selected.json"],
            "reference_delta_sha256": shas["reference_delta.json"],
            "alias_of_epoch50": result["alias_of_epoch50"],
            "selected_by_arm": result["selected_by_arm"],
            "selection_surface": plan.SELECTION_SURFACE,
            "device_profile": result["device_profile"],
            "phase1_motivation": plan.PHASE1_MOTIVATION,
            "motivation_note": plan.PHASE1_MOTIVATION["note"],
            "cross_recipe_note": plan.CROSS_RECIPE_NOTE,
            "metric": plan.METRIC_LABEL,
            "formal_benchmark_verdict": False,
        }

    attempt = receipts.stage_attempt_payload("phase3", plan.PairSpec50().sha256, identity["closure"])
    attempt["identity"] = identity
    _shas, terminal_sha, failure_sha = receipts.run_stage(
        Path(root), relative=plan.PHASE3_ROOT_RELATIVE, attempt_payload=attempt,
        launch_builder=lambda: _launch_payload("phase3", device), body_publisher=bodies,
        terminal_builder=terminal, expected_terminal_names=receipts.phase3_stage_names,
        progress=lambda: dict(progress_state),
    )
    return terminal_sha, failure_sha


__all__ = (
    "DriverError", "execute_smoke", "execute_arm", "execute_probe", "execute_phase3",
    "overlay_source_authority", "require_gpu1_profile",
)
