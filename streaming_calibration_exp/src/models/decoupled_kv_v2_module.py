"""Isolated Lightning wrapper for the teacher-readin decoupled K/V v2.

The active v1 runner does not import this module.  Production integration can
select it after the v1 result diagnosis without changing the v1
``StreamingCalibrationLitModule`` behavior or checkpoint topology.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.components.streaming_spint_v2_adapter import (
    TeacherReadinDecoupledStreamingSpint,
)
from src.models.streaming_calibration_module import (
    StreamingCalibrationLitModule,
)


class TeacherReadinDecoupledLitModule(StreamingCalibrationLitModule):
    """Lightning module with explicit v2 key semantics and optimizer wiring."""

    def __init__(
        self,
        *,
        v2_key_mode: str,
        v2_key_dim: int = 48,
        v2_value_dim: int = 64,
        v2_key_permutation_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        if v2_key_mode not in {"e_t4", "e_ts4", "e_only", "x_only"}:
            raise ValueError(
                "v2_key_mode must be one of {'e_t4','e_ts4','e_only','x_only'}"
            )
        if v2_key_dim <= 0 or v2_value_dim <= 0:
            raise ValueError("v2 key/value dimensions must be positive")
        if v2_key_mode == "e_ts4" and v2_key_permutation_seed is None:
            raise ValueError("v2 e_ts4 requires a permutation seed")
        if kwargs.get("variant") != "B3S":
            raise ValueError("teacher-readin decoupled v2 requires variant='B3S'")
        if int(kwargs.get("side_dim", 0)) != 4:
            raise ValueError("teacher-readin decoupled v2 requires T4 side_dim=4")
        if kwargs.get("identity_mode", "calibrated") != "calibrated":
            raise ValueError("teacher-readin decoupled v2 requires calibrated identity")
        if int(kwargs.get("fixed_slot_count", 0)) != 0:
            raise ValueError("teacher-readin decoupled v2 forbids fixed slots")
        if kwargs.get("encoder_warmstart_path") is not None:
            raise ValueError("v2 is a fresh common-teacher fit")
        if bool(kwargs.get("compile", False)):
            raise ValueError("v2 selector does not support torch.compile")
        if kwargs.get("decoder_mode", "coupled") != "coupled":
            raise ValueError(
                "v2 wrapper owns decoder selection; decoder_mode must be coupled"
            )

        # The parent constructs the common teacher/encoder substrate only.
        # setup() replaces its coupled student with the isolated v2 adapter.
        kwargs["decoder_mode"] = "coupled"
        super().__init__(**kwargs)
        self._v2_key_mode = v2_key_mode
        self._v2_key_dim = int(v2_key_dim)
        self._v2_value_dim = int(v2_value_dim)
        self._v2_key_permutation_seed = v2_key_permutation_seed
        self._pending_v2_checkpoint_receipt: dict[str, object] | None = None
        self.save_hyperparameters({
            "v2_key_mode": v2_key_mode,
            "v2_key_dim": self._v2_key_dim,
            "v2_value_dim": self._v2_value_dim,
            "v2_key_permutation_seed": v2_key_permutation_seed,
            "v2_initialization": "teacher_affine_proxy_global_bilinear_svd",
        })

    def setup(self, stage: str) -> None:
        if isinstance(
            self.student, TeacherReadinDecoupledStreamingSpint
        ):
            return
        trainer = getattr(self, "_trainer", None)
        world_size = int(getattr(trainer, "world_size", 1))
        if world_size != 1:
            raise RuntimeError(
                "v2 CPU-SVD initialization is world_size=1 only until "
                "rank-0 factor broadcast and post-broadcast hashing are implemented"
            )
        super().setup(stage)
        if self.student is None:
            raise RuntimeError("parent setup did not construct a student")
        substrate = self.student
        adapter = TeacherReadinDecoupledStreamingSpint(
            decoder=substrate.decoder,
            id_encoder=substrate.id_encoder,
            key_mode=self._v2_key_mode,
            key_dim=self._v2_key_dim,
            value_dim=self._v2_value_dim,
            direct_feature_dim=4,
        )
        if self._freeze_decoder:
            adapter.freeze_decoder()
        self.student = adapter

    def active_v2_checkpoint_receipt(self) -> dict[str, object]:
        self.setup("fit")
        assert isinstance(
            self.student, TeacherReadinDecoupledStreamingSpint
        )
        initialization = self.student.v2_initialization_receipt
        teacher_path = self._teacher_ckpt_path
        teacher_sha = (
            self.teacher_sha256(teacher_path)
            if teacher_path and Path(teacher_path).is_file()
            else None
        )
        return {
            "schema_version": 1,
            "module": "TeacherReadinDecoupledLitModule",
            "v2_key_mode": self._v2_key_mode,
            "v2_key_dim": self._v2_key_dim,
            "v2_value_dim": self._v2_value_dim,
            "v2_key_permutation_seed": self._v2_key_permutation_seed,
            "teacher_checkpoint_sha256": teacher_sha,
            "active_factor_sha256": initialization[
                "active_factor_sha256"
            ],
            "initial_factor_sha256": initialization[
                "initial_factor_sha256"
            ],
            "initialization_strategy": initialization["strategy"],
            "bias_policy": initialization["bias_policy"],
            "teacher_value_bias_fold_exactness": initialization[
                "teacher_value_bias_fold_exactness"
            ],
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["teacher_readin_decoupled_v2_receipt"] = (
            self.active_v2_checkpoint_receipt()
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        receipt = checkpoint.get("teacher_readin_decoupled_v2_receipt")
        if not isinstance(receipt, dict):
            raise ValueError("checkpoint is missing the v2 provenance receipt")
        expected = {
            "v2_key_mode": self._v2_key_mode,
            "v2_key_dim": self._v2_key_dim,
            "v2_value_dim": self._v2_value_dim,
            "v2_key_permutation_seed": self._v2_key_permutation_seed,
        }
        for name, value in expected.items():
            if receipt.get(name) != value:
                raise ValueError(
                    f"checkpoint {name}={receipt.get(name)!r} does not "
                    f"match constructor {value!r}"
                )
        self._pending_v2_checkpoint_receipt = dict(receipt)

    def validate_loaded_v2_checkpoint_receipt(self) -> None:
        """Validate active weights after strict state restoration."""
        if self._pending_v2_checkpoint_receipt is None:
            return
        active = self.active_v2_checkpoint_receipt()
        expected_hash = self._pending_v2_checkpoint_receipt.get(
            "active_factor_sha256"
        )
        if active["active_factor_sha256"] != expected_hash:
            raise ValueError(
                "restored active v2 factor hash does not match checkpoint receipt"
            )
        self._pending_v2_checkpoint_receipt = None

    def on_fit_start(self) -> None:
        self.validate_loaded_v2_checkpoint_receipt()

    def decoder_key_features(
        self, side_features: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Keep encoder T4 aligned; change only the direct decoder key input."""
        if self._v2_key_mode == "x_only":
            return None
        if (
            side_features is None
            or side_features.ndim != 3
            or side_features.shape[-1] != 4
        ):
            raise ValueError(
                "teacher-readin decoupled v2 requires T4 shape [B,N,4]"
            )
        if self._v2_key_mode == "e_t4":
            return side_features
        if self._v2_key_mode == "e_ts4":
            assert self._v2_key_permutation_seed is not None
            order = np.random.RandomState(
                self._v2_key_permutation_seed
            ).permutation(side_features.shape[1])
            index = torch.as_tensor(order, device=side_features.device)
            return side_features.index_select(1, index)
        # e_only must pass None so the adapter itself constructs exact zeros
        # and fail-closes if a caller tries to inject a direct feature.
        return None

    def decoupled_cost_receipt(
        self,
        *,
        batch_size: int,
        num_neurons: int,
    ) -> dict[str, object]:
        self.setup("fit")
        assert isinstance(
            self.student, TeacherReadinDecoupledStreamingSpint
        )
        return self.student.v2_cost_receipt(
            batch_size=batch_size, num_units=num_neurons
        )

    @property
    def v2_initialization_receipt(self) -> dict[str, object]:
        self.setup("fit")
        assert isinstance(
            self.student, TeacherReadinDecoupledStreamingSpint
        )
        return self.student.v2_initialization_receipt
