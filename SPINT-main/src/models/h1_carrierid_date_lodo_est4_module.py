"""Fresh source-only trainer wrapper for H1-EST4-SLODO.

The wrapper has no validation or target path.  It binds every source
checkpoint to the completed five-date H-S/H-C screen, an EST4 source
preflight, a fixed terminal epoch, and zero target-session updates.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Tuple

import torch

from src.data.h1_carrierid_date_lodo_est4 import EST4_ARMS
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256, state_hash
from src.models.components.h1_carrierid_est4_spint import EST4_ADDED_LEARNED_PARAMETERS, H1CarrierIdEst4Spint
from src.models.falcon_module import FalconLitModule


EST4_CHECKPOINT_SCHEMA = "h1_carrierid_date_lodo_est4_terminal_checkpoint_v1"


class Est4ModuleError(ValueError):
    pass


class H1CarrierIdDateLodoEst4LitModule(FalconLitModule):
    """One locked B/L arm; no target score can select or mutate it."""

    def __init__(
        self, *, arm: str, outer_date: str, fixed_seed: int = 42,
        est4_preflight_path: str, five_date_aggregate_path: str, frozen_plan_path: str, **kwargs: Any,
    ) -> None:
        normalized = str(arm).upper()
        if normalized not in EST4_ARMS or str(outer_date) not in CONFIRMATORY_DATES or int(fixed_seed) != 42:
            raise Est4ModuleError("EST4 fixes a declared arm, confirmatory source date, and seed=42")
        super().__init__(**kwargs)
        if not isinstance(self.net, H1CarrierIdEst4Spint):
            raise Est4ModuleError("EST4 module requires H1CarrierIdEst4Spint")
        expected_mode = "baseline" if normalized.startswith("B-") else "learned"
        if self.net.estimator_mode != expected_mode:
            raise Est4ModuleError("EST4 arm/model estimator-mode mismatch")
        if bool(self.net.zero_carrier) != (normalized == "L-C0") or bool(self.net.row_shuffle_output) != (normalized == "L-RS"):
            raise Est4ModuleError("EST4 model-bound C0/RS intervention mismatch")
        self.arm, self.outer_date, self.fixed_seed = normalized, str(outer_date), int(fixed_seed)
        self.est4_preflight_path = Path(est4_preflight_path).resolve()
        self.five_date_aggregate_path = Path(five_date_aggregate_path).resolve()
        self.frozen_plan_path = Path(frozen_plan_path).resolve()
        self._initial_state_sha256: str | None = None
        self._component_initial_state_sha256: str | None = None
        self._shared_backbone_initial_state_sha256: str | None = None
        self._source_binding_sha256: str | None = None
        self._base_source_binding_sha256: str | None = None
        self._source_manifest_sha256: str | None = None
        self._preflight_sha256: str | None = None
        self._est4_preflight_sha256: str | None = None
        self._five_date_aggregate_sha256: str | None = None
        self._config_sha256: str | None = None
        self._source_normalizer: float | None = None

    def forward(self, x: torch.Tensor, calib_trialized_neural_features: torch.Tensor, carrier: torch.Tensor,
                calibration_rates: torch.Tensor, calibration_labels: torch.Tensor,
                calibration_mask: torch.Tensor, row_permutation: torch.Tensor) -> torch.Tensor:
        return self.net(
            x, calib_trialized_neural_features=calib_trialized_neural_features, carrier=carrier,
            calibration_rates=calibration_rates, calibration_labels=calibration_labels,
            calibration_mask=calibration_mask, row_permutation=row_permutation,
        )

    def model_step(self, batch: Tuple[Any, ...]):
        if len(batch) != 9:
            raise Est4ModuleError("EST4 source batch must have nine source-only elements")
        neural, behavior_target, identity, session_name, carrier, rates, labels, mask, permutation = batch
        behavior_pred = self.forward(neural, identity, carrier, rates, labels, mask, permutation)
        if self.hparams.decode_last_timestep_only:
            behavior_pred, behavior_target = behavior_pred[:, -1:, :], behavior_target[:, -1:, :]
        if self.hparams.predict_scaled_behavior:
            behavior_pred = behavior_pred / self.hparams.behavior_scaling_factor
        return self.mse_loss(behavior_pred, behavior_target), behavior_pred, behavior_target, session_name

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        del checkpoint
        raise Est4ModuleError("EST4 checkpoint restore/warm start is forbidden")

    def _materialize_and_bind(self, batch: Tuple[Any, ...]) -> None:
        if self._initial_state_sha256 is not None:
            return
        trainer = self.trainer
        if int(trainer.max_epochs) != 50 or int(trainer.min_epochs) != 50 or getattr(trainer, "ckpt_path", None) not in (None, ""):
            raise Est4ModuleError("EST4 requires fresh fixed 50-epoch source training")
        datamodule = trainer.datamodule
        if not hasattr(datamodule, "phase2_source_manifest") or not hasattr(datamodule, "binding"):
            raise Est4ModuleError("EST4 needs a verified source-only DataModule")
        if Path(str(datamodule.frozen_plan_path)).resolve() != self.frozen_plan_path:
            raise Est4ModuleError("EST4 model/DataModule frozen-plan path mismatch")
        self._source_normalizer = float(datamodule.binding.normalizer.denominator)
        self.net.bind_source_normalizer(self._source_normalizer)
        source = datamodule.phase2_source_manifest()
        if (source.get("est4_arm") != self.arm or source.get("target_recordings_opened") != 0
                or source.get("target_bytes_read") != 0 or source.get("deployment_target_optimizer_steps") != 0
                or source.get("deployment_target_backward_steps") != 0):
            raise Est4ModuleError("EST4 source/target closure drift")
        if (Path(str(source.get("est4_preflight_path", ""))).resolve() != self.est4_preflight_path
                or Path(str(source.get("five_date_aggregate_path", ""))).resolve() != self.five_date_aggregate_path
                or Path(str(source.get("frozen_plan_path", ""))).resolve() != self.frozen_plan_path):
            raise Est4ModuleError("EST4 source receipt path mismatch")
        self._initial_state_sha256 = state_hash(self.state_dict())
        self._component_initial_state_sha256 = state_hash(self.net.state_dict())
        self._shared_backbone_initial_state_sha256 = self.net.shared_backbone_state_hash()
        self._source_binding_sha256 = canonical_sha256(source)
        self._base_source_binding_sha256 = str(source.get("phase2_base_source_binding_sha256", ""))
        self._source_manifest_sha256 = str(datamodule.phase1_manifest_sha256)
        self._preflight_sha256 = str(source.get("preflight_sha256", ""))
        self._est4_preflight_sha256 = str(source.get("est4_preflight_sha256", ""))
        self._five_date_aggregate_sha256 = str(source.get("five_date_aggregate_sha256", ""))
        if any(len(str(value)) != 64 for value in (
            self._source_binding_sha256, self._base_source_binding_sha256, self._source_manifest_sha256,
            self._preflight_sha256, self._est4_preflight_sha256, self._five_date_aggregate_sha256,
        )):
            raise Est4ModuleError("EST4 source receipt SHA closure is incomplete")
        config = Path(trainer.default_root_dir).resolve() / ".hydra/config.yaml"
        if not config.is_file():
            raise Est4ModuleError("EST4 saved Hydra config is missing")
        self._config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()

    def on_train_batch_start(self, batch: Tuple[Any, ...], batch_idx: int) -> None:
        if self.current_epoch == 0 and batch_idx == 0:
            self._materialize_and_bind(batch)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if int(checkpoint.get("epoch", -1)) != 49:
            raise Est4ModuleError("EST4 checkpoint must be fixed terminal e49")
        values = (
            self._initial_state_sha256, self._component_initial_state_sha256, self._shared_backbone_initial_state_sha256,
            self._source_binding_sha256, self._base_source_binding_sha256, self._source_manifest_sha256,
            self._preflight_sha256, self._est4_preflight_sha256, self._five_date_aggregate_sha256,
            self._config_sha256, self._source_normalizer,
        )
        if any(value is None for value in values):
            raise Est4ModuleError("EST4 checkpoint lacks source-only provenance")
        checkpoint["h1_carrierid_date_lodo_est4"] = {
            "schema": EST4_CHECKPOINT_SCHEMA,
            "arm": self.arm, "outer_date": self.outer_date, "fresh_seed": self.fixed_seed,
            "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
            "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
            "initial_state_sha256": self._initial_state_sha256,
            "component_initial_state_sha256": self._component_initial_state_sha256,
            "shared_backbone_initial_state_sha256": self._shared_backbone_initial_state_sha256,
            "est4_source_binding_sha256": self._source_binding_sha256,
            "phase2_base_source_binding_sha256": self._base_source_binding_sha256,
            "phase1_source_manifest_sha256": self._source_manifest_sha256,
            "phase1_preflight_sha256": self._preflight_sha256,
            "est4_preflight_sha256": self._est4_preflight_sha256,
            "five_date_aggregate_sha256": self._five_date_aggregate_sha256,
            "config_sha256": self._config_sha256,
            "frozen_plan_path": str(self.frozen_plan_path),
            "frozen_plan_sha256": hashlib.sha256(self.frozen_plan_path.read_bytes()).hexdigest(),
            "source_normalizer": self._source_normalizer,
            "estimator_mode": self.net.estimator_mode,
            "estimator_added_learned_parameters": self.net.estimator_parameter_count(),
            "expected_estimator_added_learned_parameters": (
                EST4_ADDED_LEARNED_PARAMETERS if self.net.estimator_mode == "learned" else 0
            ),
            "target_optimizer_steps": 0, "target_backward_steps": 0, "checkpoint_warm_start": False,
            "target_evaluator_status": "IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED",
        }
