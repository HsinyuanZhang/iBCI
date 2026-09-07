"""Fold-local Stage-1: train one arm, then score Static and CDM-A."""
from __future__ import annotations

import io
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import training as rsyn3_training
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as parent_data

from . import carrier_bank
from . import gpu as stage1_gpu
from . import plan
from . import prefix as stage1_prefix
from . import receipts as fold_receipts
from . import score as stage1_score
from . import stage0 as fold_stage0


class Stage1Error(RuntimeError):
    """Fail closed for Stage-1 training/scoring."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage1Error(message)


def ensure_streaming_path(repo_root: Path) -> None:
    experiment = str(Path(repo_root) / "streaming_calibration_exp")
    if experiment not in sys.path:
        sys.path.insert(0, experiment)


def _arm_slug(arm: str) -> str:
    return {"Z-Fix": "z_fix", "S-Fix": "s_fix", "S-Acyc": "s_acyc"}[arm]


def _stage_relative(arm: str) -> str:
    return str(Path(plan.STAGE1_ARM_ROOT_RELATIVE[arm]).relative_to(plan.RESULT_ROOT_RELATIVE))


def _lightning_callback(operator: stage1_prefix.ActivityPrefixOperator):
    import lightning.pytorch as pl

    class PrefixCallback(pl.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.operator = operator

        def on_fit_start(self, trainer, pl_module) -> None:
            self.operator.attach(pl_module.student)

    return PrefixCallback()


def execute_arm(
    repo_root: Path,
    *,
    arm: str,
    gpu_index: int,
    gpu_authorized: bool,
    pack: bool = False,
) -> tuple[dict[str, str], str | None, str | None]:
    repo_root = Path(repo_root)
    _require(arm in plan.STAGE1_ARMS, f"unknown arm {arm}")
    _require(type(pack) is bool, "pack flag")
    plan.verify_bound_documents(repo_root)
    fold_stage0.verify_sealed_roots(repo_root)
    relative = _stage_relative(arm)
    fold_receipts.refuse_sealed_roots(repo_root / plan.STAGE1_ARM_ROOT_RELATIVE[arm])
    (repo_root / plan.STAGE1_ROOT_RELATIVE).mkdir(parents=True, exist_ok=True)
    bank = carrier_bank.build_fold0_carrier_bank(repo_root)
    gpu_info = stage1_gpu.require_launch(gpu_authorized, gpu_index, pack=pack)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    teacher = repo_root / plan.TEACHER_CKPT_RELATIVE
    teacher_sha = parent_data.file_sha256(teacher)
    _require(teacher_sha == plan.TEACHER_CKPT_SHA256, "teacher sha drift")

    def launch_builder() -> dict[str, object]:
        return {
            "schema": "m1_emg_rsyn3_fold_local_stage1_launch_v1",
            "arm": arm,
            "gpu": gpu_info,
            "teacher_ckpt_relative": plan.TEACHER_CKPT_RELATIVE,
            "teacher_ckpt_sha256": teacher_sha,
            "epochs": plan.STAGE1_EPOCHS,
            "fixed_last_epoch_index": plan.FIXED_LAST_EPOCH_INDEX,
            "lr": plan.STAGE1_LR,
            "batch_size": plan.STAGE1_BATCH_SIZE,
            "variant": "B3",
            "side_dim": 0,
            "ls4_enabled": False,
            "pack": pack,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    def body_publisher(artifact) -> dict[str, str]:
        return _train_and_score(repo_root, arm=arm, gpu_index=gpu_index, artifact=artifact, bank=bank)

    def terminal_builder(shas: Mapping[str, str]) -> dict[str, object]:
        return {
            "schema": "m1_emg_rsyn3_fold_local_stage1_terminal_v1",
            "status": "COMPLETE",
            "arm": arm,
            "bodies": dict(shas),
            "ls4_enabled_in_stage1_pilot": False,
            "gpu_capability_issued": True,
        }

    return fold_receipts.run_stage0(
        repo_root / plan.RESULT_ROOT_RELATIVE,
        attempt_payload={
            "schema": "m1_emg_rsyn3_fold_local_stage1_attempt_v1",
            "arm": arm,
            "gpu_index": gpu_index,
            "pack": pack,
            "fold": 0,
        },
        launch_builder=launch_builder,
        body_publisher=body_publisher,
        terminal_builder=terminal_builder,
        relative=relative,
    )


def resolve_fixed_last_checkpoint(directory: Path, selected: str, torch_module) -> tuple[Path, dict[str, Any]]:
    """Resolve the fixed-last checkpoint without assuming Lightning's filename law.

    Lightning interpolates ``filename="epoch_{epoch:03d}"`` into
    ``epoch_epoch=011.ckpt``, so the selection token cannot be used as a path.
    Selection stays target-blind: the epoch index is the only authority.
    """
    directory = Path(directory)
    expected = int(plan.FIXED_LAST_EPOCH_INDEX)
    _require(selected == f"epoch_{expected:03d}", f"fixed-last selection token drift {selected!r}")
    candidates = sorted(directory.glob("*.ckpt"))
    _require(
        len(candidates) == 1,
        f"expected exactly one checkpoint in {directory}, found {[path.name for path in candidates]}",
    )
    path = candidates[0]
    digits = re.findall(r"\d+", path.stem)
    _require(bool(digits), f"checkpoint name carries no epoch digits: {path.name}")
    _require(
        int(digits[-1]) == expected,
        f"checkpoint name epoch {digits[-1]} != fixed-last {expected} ({path.name})",
    )
    payload = torch_module.load(path, map_location="cpu", weights_only=False)
    _require("epoch" in payload, "checkpoint payload has no epoch counter")
    _require(
        int(payload["epoch"]) == expected,
        f"checkpoint payload epoch {payload['epoch']} != fixed-last {expected}",
    )
    _require("state_dict" in payload, "checkpoint payload has no state_dict")
    return path, payload


def _train_and_score(repo_root: Path, *, arm: str, gpu_index: int, artifact, bank: dict[str, Any]) -> dict[str, str]:
    import lightning.pytorch as pl
    import torch

    ensure_streaming_path(repo_root)
    from lightning.pytorch.callbacks import ModelCheckpoint

    from . import datamodule as stage1_data
    from . import module as stage1_module

    pl.seed_everything(plan.SEED, workers=True)
    data = stage1_data.make_datamodule(repo_root, bank, arm)
    lit = stage1_module.make_module(repo_root)
    operator = stage1_prefix.ActivityPrefixOperator(arm)
    with tempfile.TemporaryDirectory(prefix=f"rsyn3_{_arm_slug(arm)}_") as tmp:
        ckpt_cb = ModelCheckpoint(
            dirpath=tmp,
            filename="epoch_{epoch:03d}",
            monitor=None,
            save_top_k=-1,
            every_n_epochs=plan.STAGE1_EPOCHS,
            save_last=False,
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=plan.STAGE1_EPOCHS,
            min_epochs=plan.STAGE1_EPOCHS,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=True,
            callbacks=[ckpt_cb, _lightning_callback(operator)],
            deterministic=True,
        )
        trainer.fit(lit, datamodule=data)
        selected = rsyn3_training.select_fixed_last(plan.FIXED_LAST_EPOCH_INDEX)
        ckpt_path, payload = resolve_fixed_last_checkpoint(Path(tmp), selected, torch)
        if ckpt_cb.best_model_path:
            _require(
                Path(ckpt_cb.best_model_path) == ckpt_path,
                f"callback path {ckpt_cb.best_model_path} disagrees with resolved {ckpt_path}",
            )
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        ckpt_sha = artifact.publish_bytes("epoch_011.pt", buffer.getvalue())
        lit.load_state_dict(payload["state_dict"], strict=True)
    device = "cuda:0"
    lit.to(device)
    lit.eval()
    student = lit.student
    session = plan.FOLD0_TARGET_SESSION
    carrier = carrier_bank.carrier_for_arm(bank, session, arm)
    dataset = data.val_heldin_dataset.base
    static = stage1_score.score_static(
        student, dataset, session_id=session, carrier=carrier, device=device,
    )
    cdm = stage1_score.score_cdm_fifo(
        student, dataset, session_id=session, carrier=carrier, device=device,
    )
    static_sha = artifact.publish_json("score_static.json", static)
    cdm_sha = artifact.publish_json("score_cdm_a.json", cdm)
    table_sha = artifact.publish_json("arm_table.json", {
        "arm": arm,
        "static_r2": static["governing_r2"],
        "cdm_a_r2": cdm["governing_r2"],
        "delta_cdm_minus_static": cdm["governing_r2"] - static["governing_r2"],
        "fixed_last": selected,
        "ls4_enabled": False,
        "target_session": session,
        "query": [plan.QUERY_START, plan.QUERY_STOP_EXCLUSIVE],
    })
    operator.detach()
    return {
        "epoch_011.pt": ckpt_sha,
        "score_static.json": static_sha,
        "score_cdm_a.json": cdm_sha,
        "arm_table.json": table_sha,
    }
