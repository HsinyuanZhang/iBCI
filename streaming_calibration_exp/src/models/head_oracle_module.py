"""Isolated Lightning selector for the teacher-head-preserving K/V oracle."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.components.streaming_spint_head_oracle_adapter import (
    TeacherHeadOracleStreamingSpint,
)
from src.models.streaming_calibration_module import (
    StreamingCalibrationLitModule,
)


class TeacherHeadOracleLitModule(StreamingCalibrationLitModule):
    """Select the exact-head diagnostic without changing the common substrate."""

    def __init__(
        self,
        *,
        oracle_key_mode: str,
        oracle_key_permutation_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        if oracle_key_mode not in {"e_t4", "e_ts4"}:
            raise ValueError("oracle_key_mode must be 'e_t4' or 'e_ts4'")
        if (
            oracle_key_mode == "e_ts4"
            and oracle_key_permutation_seed is None
        ):
            raise ValueError("oracle e_ts4 requires a permutation seed")
        if (
            oracle_key_mode == "e_t4"
            and oracle_key_permutation_seed is not None
        ):
            raise ValueError("oracle e_t4 forbids a permutation seed")
        if kwargs.get("variant") != "B3S":
            raise ValueError("head oracle requires variant='B3S'")
        if int(kwargs.get("side_dim", 0)) != 4:
            raise ValueError("head oracle requires side_dim=4")
        if kwargs.get("identity_mode", "calibrated") != "calibrated":
            raise ValueError("head oracle requires calibrated identity")
        if int(kwargs.get("fixed_slot_count", 0)) != 0:
            raise ValueError("head oracle forbids fixed slots")
        if kwargs.get("encoder_warmstart_path") is not None:
            raise ValueError("head oracle requires a fresh common-teacher fit")
        if bool(kwargs.get("compile", False)):
            raise ValueError("head oracle selector does not support compile")
        if kwargs.get("decoder_mode", "coupled") != "coupled":
            raise ValueError(
                "head oracle owns decoder selection; base mode must be coupled"
            )

        kwargs["decoder_mode"] = "coupled"
        super().__init__(**kwargs)
        self._oracle_key_mode = oracle_key_mode
        self._oracle_key_permutation_seed = (
            oracle_key_permutation_seed
        )
        self._pending_oracle_checkpoint_receipt: (
            dict[str, object] | None
        ) = None
        self.save_hyperparameters({
            "oracle_key_mode": oracle_key_mode,
            "oracle_key_permutation_seed": (
                oracle_key_permutation_seed
            ),
            "oracle_initialization": (
                "exact_teacher_head_projection_copy"
            ),
        })

    def setup(self, stage: str) -> None:
        if isinstance(self.student, TeacherHeadOracleStreamingSpint):
            return
        trainer = getattr(self, "_trainer", None)
        if int(getattr(trainer, "world_size", 1)) != 1:
            raise RuntimeError(
                "head oracle setup is world_size=1 until copied-factor "
                "broadcast and post-broadcast hashing are implemented"
            )
        super().setup(stage)
        if self.student is None:
            raise RuntimeError("parent setup did not construct a student")
        substrate = self.student
        adapter = TeacherHeadOracleStreamingSpint(
            decoder=substrate.decoder,
            id_encoder=substrate.id_encoder,
            key_mode=self._oracle_key_mode,
            key_permutation_seed=self._oracle_key_permutation_seed,
        )
        if self._freeze_decoder:
            adapter.freeze_decoder()
        self.student = adapter

    def active_oracle_checkpoint_receipt(self) -> dict[str, object]:
        self.setup("fit")
        assert isinstance(self.student, TeacherHeadOracleStreamingSpint)
        initialization = self.student.oracle_initialization_receipt
        teacher_path = self._teacher_ckpt_path
        teacher_sha = (
            self.teacher_sha256(teacher_path)
            if teacher_path and Path(teacher_path).is_file()
            else None
        )
        return {
            "schema_version": 1,
            "module": "TeacherHeadOracleLitModule",
            "oracle_key_mode": self._oracle_key_mode,
            "oracle_key_permutation_seed": (
                self._oracle_key_permutation_seed
            ),
            "teacher_checkpoint_sha256": teacher_sha,
            "initial_factor_sha256": initialization[
                "initial_factor_sha256"
            ],
            "active_factor_sha256": initialization[
                "active_factor_sha256"
            ],
            "initialization_strategy": initialization["strategy"],
            "teacher_head_count": initialization[
                "teacher_head_count"
            ],
            "teacher_headwise_softmax_preserved": initialization[
                "teacher_headwise_softmax_preserved"
            ],
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["teacher_head_oracle_receipt"] = (
            self.active_oracle_checkpoint_receipt()
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        receipt = checkpoint.get("teacher_head_oracle_receipt")
        if not isinstance(receipt, dict):
            raise ValueError("checkpoint is missing the head-oracle receipt")
        expected = {
            "oracle_key_mode": self._oracle_key_mode,
            "oracle_key_permutation_seed": (
                self._oracle_key_permutation_seed
            ),
        }
        for name, value in expected.items():
            if receipt.get(name) != value:
                raise ValueError(
                    f"checkpoint {name}={receipt.get(name)!r} does not "
                    f"match constructor {value!r}"
                )
        self._pending_oracle_checkpoint_receipt = dict(receipt)

    def validate_loaded_oracle_checkpoint_receipt(self) -> None:
        if self._pending_oracle_checkpoint_receipt is None:
            return
        active = self.active_oracle_checkpoint_receipt()
        if active["active_factor_sha256"] != (
            self._pending_oracle_checkpoint_receipt.get(
                "active_factor_sha256"
            )
        ):
            raise ValueError(
                "restored active head-oracle hash does not match receipt"
            )
        self._pending_oracle_checkpoint_receipt = None

    def on_fit_start(self) -> None:
        self.validate_loaded_oracle_checkpoint_receipt()

    def decoder_key_features(
        self, side_features: torch.Tensor | None
    ) -> None:
        if (
            side_features is None
            or side_features.ndim != 3
            or side_features.shape[-1] != 4
        ):
            raise ValueError(
                "head oracle requires aligned T4 with shape [B,N,4]"
            )
        # T4 always reaches the encoder unmodified. The adapter permutes E
        # rows only for the decoder-K content control.
        return None

    def decoupled_cost_receipt(
        self,
        *,
        batch_size: int,
        num_neurons: int,
    ) -> dict[str, object]:
        self.setup("fit")
        assert isinstance(self.student, TeacherHeadOracleStreamingSpint)
        return self.student.oracle_cost_receipt(
            batch_size=batch_size, num_units=num_neurons
        )

    @property
    def oracle_initialization_receipt(self) -> dict[str, object]:
        self.setup("fit")
        assert isinstance(self.student, TeacherHeadOracleStreamingSpint)
        return self.student.oracle_initialization_receipt
