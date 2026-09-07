"""Lightning selector for the factorized, baseline-preserving T4 logit residual."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch

from src.models.components.streaming_spint_t4_logit_residual_adapter import (
    CoupledT4LogitResidualStreamingSpint,
)
from src.models.streaming_calibration_module import StreamingCalibrationLitModule


class T4LogitResidualLitModule(StreamingCalibrationLitModule):
    """Restore selected T4, freeze it, and train only two low-rank factors."""

    def __init__(
        self,
        *,
        residual_mode: Literal["aligned", "shuffled"],
        interaction_mode: Literal["attention_logit", "additive_control"],
        residual_rank: int = 8,
        residual_permutation_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        if residual_mode not in {"aligned", "shuffled"}:
            raise ValueError("residual_mode must be 'aligned' or 'shuffled'")
        if interaction_mode not in {"attention_logit", "additive_control"}:
            raise ValueError("unsupported interaction_mode")
        if residual_rank <= 0:
            raise ValueError("residual_rank must be positive")
        if residual_mode == "shuffled" and residual_permutation_seed is None:
            raise ValueError("shuffled residual requires a permutation seed")
        if residual_mode == "aligned" and residual_permutation_seed is not None:
            raise ValueError("aligned residual forbids a permutation seed")
        if kwargs.get("variant") != "B3S":
            raise ValueError("T4 logit residual requires variant='B3S'")
        if int(kwargs.get("side_dim", 0)) != 4:
            raise ValueError("T4 logit residual requires side_dim=4")
        if kwargs.get("identity_mode", "calibrated") != "calibrated":
            raise ValueError("T4 logit residual requires calibrated identity")
        if int(kwargs.get("fixed_slot_count", 0)) != 0:
            raise ValueError("T4 logit residual forbids fixed slots")
        if kwargs.get("decoder_mode", "coupled") != "coupled":
            raise ValueError("T4 logit residual requires the coupled teacher")
        if bool(kwargs.get("compile", False)):
            raise ValueError("T4 logit residual does not support compile")
        if bool(kwargs.get("freeze_decoder", False)):
            raise ValueError("T4 logit residual owns the exact backbone freeze policy")
        selected_anchor = kwargs.get("encoder_warmstart_path")
        if not selected_anchor:
            raise ValueError("T4 logit residual requires a selected full T4 warm-start checkpoint")

        kwargs["decoder_mode"] = "coupled"
        kwargs["freeze_decoder"] = False
        super().__init__(**kwargs)
        self._residual_mode = residual_mode
        self._interaction_mode = interaction_mode
        self._residual_rank = int(residual_rank)
        self._residual_permutation_seed = residual_permutation_seed
        self._pending_t4_logit_receipt: dict[str, object] | None = None
        self.save_hyperparameters(
            {
                "t4_logit_residual": True,
                "residual_mode": residual_mode,
                "interaction_mode": interaction_mode,
                "residual_rank": self._residual_rank,
                "residual_permutation_seed": residual_permutation_seed,
                "residual_initialization": "zero_query_factors",
            }
        )

    def setup(self, stage: str) -> None:
        if isinstance(self.student, CoupledT4LogitResidualStreamingSpint):
            return
        trainer = getattr(self, "_trainer", None)
        if int(getattr(trainer, "world_size", 1)) != 1:
            raise RuntimeError("T4 logit residual setup is world_size=1")
        super().setup(stage)
        if self.student is None:
            raise RuntimeError("parent setup did not construct a student")
        substrate = self.student
        adapter = CoupledT4LogitResidualStreamingSpint(
            decoder=substrate.decoder,
            id_encoder=substrate.id_encoder,
            residual_mode=self._residual_mode,
            interaction_mode=self._interaction_mode,
            residual_rank=self._residual_rank,
            residual_permutation_seed=self._residual_permutation_seed,
        )
        adapter.freeze_backbone_for_residual_pilot()
        self.student = adapter

    def active_t4_logit_residual_receipt(self) -> dict[str, object]:
        self.setup("fit")
        assert isinstance(self.student, CoupledT4LogitResidualStreamingSpint)
        initialization = self.student.logit_residual_receipt
        teacher_path = self._teacher_ckpt_path
        teacher_sha = (
            self.teacher_sha256(teacher_path)
            if teacher_path and Path(teacher_path).is_file()
            else None
        )
        anchor_path = Path(self._encoder_warmstart_path).expanduser()
        anchor_sha = self.teacher_sha256(str(anchor_path)) if anchor_path.is_file() else None
        return {
            "schema_version": 1,
            "module": "T4LogitResidualLitModule",
            "residual_mode": self._residual_mode,
            "interaction_mode": self._interaction_mode,
            "residual_rank": self._residual_rank,
            "residual_permutation_seed": self._residual_permutation_seed,
            "teacher_checkpoint_sha256": teacher_sha,
            "selected_t4_anchor_path": str(anchor_path.resolve()),
            "selected_t4_anchor_sha256": anchor_sha,
            "initial_factor_sha256": initialization["initial_factor_sha256"],
            "active_factor_sha256": initialization["active_factor_sha256"],
            "zero_initialized": initialization["query_factors_zero_initialized"],
            "backbone_frozen": initialization["backbone_frozen_for_residual_pilot"],
            "cached_state": "per-unit rank factor only",
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["t4_logit_residual_receipt"] = self.active_t4_logit_residual_receipt()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        receipt = checkpoint.get("t4_logit_residual_receipt")
        if not isinstance(receipt, dict):
            raise ValueError("checkpoint is missing the T4 logit-residual receipt")
        expected = {
            "residual_mode": self._residual_mode,
            "interaction_mode": self._interaction_mode,
            "residual_rank": self._residual_rank,
            "residual_permutation_seed": self._residual_permutation_seed,
        }
        for name, value in expected.items():
            if receipt.get(name) != value:
                raise ValueError(
                    f"checkpoint {name}={receipt.get(name)!r} does not match constructor {value!r}"
                )
        self._pending_t4_logit_receipt = dict(receipt)

    def validate_loaded_t4_logit_residual_receipt(self) -> None:
        if self._pending_t4_logit_receipt is None:
            return
        active = self.active_t4_logit_residual_receipt()
        pending = self._pending_t4_logit_receipt
        if active["active_factor_sha256"] != pending.get("active_factor_sha256"):
            raise ValueError("restored active T4 logit-residual hash does not match checkpoint receipt")
        if active["selected_t4_anchor_sha256"] != pending.get("selected_t4_anchor_sha256"):
            raise ValueError("selected T4 anchor hash does not match checkpoint receipt")
        self._pending_t4_logit_receipt = None

    def on_fit_start(self) -> None:
        self.validate_loaded_t4_logit_residual_receipt()

    def decoder_key_features(self, side_features: torch.Tensor | None) -> None:
        if side_features is None or side_features.ndim != 3 or side_features.shape[-1] != 4:
            raise ValueError("T4 logit residual requires aligned T4 with shape [B,N,4]")
        return None

    def decoupled_cost_receipt(self, *, batch_size: int, num_neurons: int) -> dict[str, object]:
        self.setup("fit")
        assert isinstance(self.student, CoupledT4LogitResidualStreamingSpint)
        return self.student.residual_cost_receipt(batch_size=batch_size, num_units=num_neurons)

    @property
    def t4_logit_residual_initialization_receipt(self) -> dict[str, object]:
        self.setup("fit")
        assert isinstance(self.student, CoupledT4LogitResidualStreamingSpint)
        return self.student.logit_residual_receipt
