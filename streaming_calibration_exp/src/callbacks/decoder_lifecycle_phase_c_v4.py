"""Capture pretrain/posttrain decoder closure stages in the owned T4 cell."""
from __future__ import annotations

import json
from pathlib import Path

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.core import LightningModule

from src.utils.decoder_lifecycle_phase_c_v4 import (
    finalize_lifecycle_evidence,
    write_stage_exclusive,
)


class DecoderLifecycleTrainStagesV4(Callback):
    def __init__(
        self,
        *,
        stage_dir: str,
        cell_owner_path: str,
        owner_token: str,
        fold: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.stage_dir = Path(stage_dir).resolve()
        self.cell_owner_path = Path(cell_owner_path).resolve()
        self.owner_token = owner_token
        self.fold = fold
        self.seed = seed

    def _identity(self):
        return {
            "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
            "phase_id": "PHASE_C_V4",
            "arm": "t4",
            "fold": self.fold,
            "seed": self.seed,
        }

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "fit":
            return
        owner = json.loads(self.cell_owner_path.read_text(encoding="utf-8"))
        if owner.get("owner_token") != self.owner_token or any(
            owner.get(field) != expected for field, expected in self._identity().items()
        ):
            raise PermissionError("decoder lifecycle callback ownership mismatch")
        cell = self.cell_owner_path.parent.parent.resolve()
        if self.stage_dir != (cell / "run/decoder_lifecycle_stages").resolve():
            raise ValueError("decoder lifecycle stages are outside the owned run")

    @staticmethod
    def _decoder(pl_module: LightningModule):
        pl_module.setup("fit")
        student = getattr(pl_module, "student", None)
        if student is None:
            raise RuntimeError("T4 student is unavailable for decoder lifecycle capture")
        return student.decoder

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        write_stage_exclusive(
            self.stage_dir,
            stage="pretrain",
            decoder=self._decoder(pl_module),
            optimizer=trainer.optimizers[0] if trainer.optimizers else None,
            cell_identity=self._identity(),
        )

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        write_stage_exclusive(
            self.stage_dir,
            stage="posttrain",
            decoder=self._decoder(pl_module),
            optimizer=trainer.optimizers[0] if trainer.optimizers else None,
            cell_identity=self._identity(),
        )


class DecoderLifecycleEvalStagesV4(Callback):
    """Capture the checkpoint-reloaded decoder immediately before outer query."""

    def __init__(
        self,
        *,
        stage_dir: str,
        evidence_output: str,
        cell_owner_path: str,
        owner_token: str,
        fold: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.stage_dir = Path(stage_dir).resolve()
        self.evidence_output = Path(evidence_output).resolve()
        self.cell_owner_path = Path(cell_owner_path).resolve()
        self.owner_token = owner_token
        self.fold = fold
        self.seed = seed

    def _identity(self):
        return {
            "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
            "phase_id": "PHASE_C_V4",
            "arm": "t4",
            "fold": self.fold,
            "seed": self.seed,
        }

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "test":
            return
        owner = json.loads(self.cell_owner_path.read_text(encoding="utf-8"))
        if owner.get("owner_token") != self.owner_token or any(
            owner.get(field) != expected for field, expected in self._identity().items()
        ):
            raise PermissionError("decoder evaluator lifecycle ownership mismatch")
        cell = self.cell_owner_path.parent.parent.resolve()
        if self.stage_dir != (cell / "run/decoder_lifecycle_stages").resolve():
            raise ValueError("decoder evaluator lifecycle stages are outside the owned run")
        if self.evidence_output != (cell / "run/decoder_lifecycle_evidence.json").resolve():
            raise ValueError("decoder lifecycle evidence output is outside the owned run")

    @staticmethod
    def _decoder(pl_module: LightningModule):
        student = getattr(pl_module, "student", None)
        if student is None:
            raise RuntimeError("T4 student is unavailable after checkpoint reload")
        return student.decoder

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        decoder = self._decoder(pl_module)
        for stage in ("reload", "prequery"):
            write_stage_exclusive(
                self.stage_dir,
                stage=stage,
                decoder=decoder,
                optimizer=None,
                cell_identity=self._identity(),
            )
        finalize_lifecycle_evidence(
            self.stage_dir,
            self.evidence_output,
            cell_identity=self._identity(),
        )
