"""Fresh source-only trainer wrapper for the H1 CI32/CI64 control family."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Tuple

import torch

from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256, state_hash
from src.models.falcon_module import FalconLitModule


CI_CHECKPOINT_SCHEMA = "h1_carrierid_date_lodo_ci_terminal_checkpoint_v1"
CI_ARMS = ("CI32-FULL", "CI64-FULL", "CI64-C0", "CI64-LS", "CI64-RS")


class CarrierIdDateLodoCiModelError(ValueError):
    """Fail-closed violation of the fresh CI source-training contract."""


class H1CarrierIdDateLodoCiLitModule(FalconLitModule):
    """One fresh, source-only CI arm; target evaluation is a separate gated program."""

    def __init__(
        self, *, arm: str, outer_date: str, fixed_seed: int = 42,
        ci_preflight_path: str, five_date_aggregate_path: str, **kwargs: Any,
    ) -> None:
        normalized_arm = str(arm).upper()
        if normalized_arm not in CI_ARMS:
            raise CarrierIdDateLodoCiModelError(f"CI arm must be one of {CI_ARMS}")
        if str(outer_date) not in CONFIRMATORY_DATES or int(fixed_seed) != 42:
            raise CarrierIdDateLodoCiModelError("CI source training fixes a confirmatory outer date and fresh seed=42")
        super().__init__(**kwargs)
        self.arm, self.outer_date, self.fixed_seed = normalized_arm, str(outer_date), int(fixed_seed)
        self.ci_preflight_path = Path(ci_preflight_path).resolve()
        self.five_date_aggregate_path = Path(five_date_aggregate_path).resolve()
        self._initial_state_sha256: str | None = None
        self._component_initial_state_sha256: str | None = None
        self._shared_backbone_state_sha256: str | None = None
        self._source_binding_sha256: str | None = None
        self._base_source_binding_sha256: str | None = None
        self._source_manifest_sha256: str | None = None
        self._preflight_sha256: str | None = None
        self._ci_preflight_sha256: str | None = None
        self._five_date_aggregate_sha256: str | None = None
        self._config_sha256: str | None = None

    def forward(self, x: torch.Tensor, calib_trialized_neural_features: torch.Tensor, carrier: torch.Tensor):
        return self.net(x, calib_trialized_neural_features=calib_trialized_neural_features, carrier=carrier)

    def model_step(self, batch: Tuple[Any, ...]):
        if len(batch) != 5:
            raise CarrierIdDateLodoCiModelError("CI source batch must have five elements")
        neural, behavior_target, identity, session_name, carrier = batch
        behavior_pred = self.forward(neural, calib_trialized_neural_features=identity, carrier=carrier)
        if self.hparams.decode_last_timestep_only:
            behavior_pred, behavior_target = behavior_pred[:, -1:, :], behavior_target[:, -1:, :]
        if self.hparams.predict_scaled_behavior:
            behavior_pred = behavior_pred / self.hparams.behavior_scaling_factor
        return self.mse_loss(behavior_pred, behavior_target), behavior_pred, behavior_target, session_name

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        del checkpoint
        raise CarrierIdDateLodoCiModelError("CI arms require fresh initialization; checkpoint loading is forbidden")

    def _materialize_and_bind(self, batch: Tuple[Any, ...]) -> None:
        if self._initial_state_sha256 is not None:
            return
        trainer = self.trainer
        if int(trainer.max_epochs) != 50 or int(trainer.min_epochs) != 50 or getattr(trainer, "ckpt_path", None) not in (None, ""):
            raise CarrierIdDateLodoCiModelError("CI arms require fresh fixed 50-epoch source training")
        if len(batch) != 5:
            raise CarrierIdDateLodoCiModelError("cannot materialize malformed CI source batch")
        neural, _target, identity, _session, carrier = batch
        was_training = self.net.training
        self.net.eval()
        try:
            with torch.no_grad():
                self.forward(neural, calib_trialized_neural_features=identity, carrier=carrier)
        finally:
            self.net.train(was_training)
        datamodule = trainer.datamodule
        if not hasattr(datamodule, "phase2_source_manifest") or not hasattr(datamodule, "phase1_manifest_sha256"):
            raise CarrierIdDateLodoCiModelError("CI trainer lacks a verified source-only DataModule")
        source = datamodule.phase2_source_manifest()
        expected_intervention = self.arm.split("-", 1)[1].lower()
        if expected_intervention == "c0":
            expected_intervention = "c0"
        if source.get("carrier_intervention") != expected_intervention:
            raise CarrierIdDateLodoCiModelError("CI data intervention does not match model arm")
        if source.get("outer_date") != self.outer_date or source.get("target_recordings_opened") != 0 or source.get("target_bytes_read") != 0:
            raise CarrierIdDateLodoCiModelError("CI source/target boundary drift")
        self._initial_state_sha256 = state_hash(self.state_dict())
        self._component_initial_state_sha256 = state_hash(self.net.state_dict())
        self._shared_backbone_state_sha256 = self.net.shared_backbone_state_hash()
        self._source_binding_sha256 = canonical_sha256(source)
        self._base_source_binding_sha256 = str(source.get("phase2_base_source_binding_sha256", ""))
        if len(self._base_source_binding_sha256) != 64:
            raise CarrierIdDateLodoCiModelError("CI DataModule lacks base source-binding SHA")
        self._source_manifest_sha256 = str(datamodule.phase1_manifest_sha256)
        self._preflight_sha256 = str(source["preflight_sha256"])
        if (self.ci_preflight_path != Path(str(source.get("ci_preflight_path", ""))).resolve()
                or self.five_date_aggregate_path
                != Path(str(source.get("five_date_aggregate_path", ""))).resolve()):
            raise CarrierIdDateLodoCiModelError("CI model config receipt paths differ from DataModule receipt gate")
        self._ci_preflight_sha256 = str(getattr(datamodule, "ci_preflight_sha256"))
        self._five_date_aggregate_sha256 = str(getattr(datamodule, "five_date_aggregate_sha256"))
        if (source.get("ci_preflight_sha256") != self._ci_preflight_sha256
                or source.get("five_date_aggregate_sha256") != self._five_date_aggregate_sha256
                or source.get("five_date_source_date_screen_complete") is not True
                or source.get("five_date_automatic_route_selection_forbidden") is not True):
            raise CarrierIdDateLodoCiModelError("CI DataModule did not bind both receipt gates")
        config_path = Path(trainer.default_root_dir).resolve() / ".hydra" / "config.yaml"
        if not config_path.is_file():
            raise CarrierIdDateLodoCiModelError("resolved CI Hydra config is missing")
        self._config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

    def on_train_batch_start(self, batch: Tuple[Any, ...], batch_idx: int) -> None:
        if self.current_epoch == 0 and batch_idx == 0:
            self._materialize_and_bind(batch)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if int(checkpoint.get("epoch", -1)) != 49:
            raise CarrierIdDateLodoCiModelError("CI checkpoint must be the fixed terminal e49")
        values = (self._initial_state_sha256, self._component_initial_state_sha256, self._shared_backbone_state_sha256, self._source_binding_sha256,
                  self._base_source_binding_sha256,
                  self._source_manifest_sha256, self._preflight_sha256, self._ci_preflight_sha256,
                  self._five_date_aggregate_sha256, self._config_sha256)
        if any(value is None for value in values):
            raise CarrierIdDateLodoCiModelError("CI terminal checkpoint lacks fresh source provenance")
        checkpoint["h1_carrierid_date_lodo_ci"] = {
            "schema": CI_CHECKPOINT_SCHEMA,
            "arm": self.arm, "outer_date": self.outer_date, "fresh_seed": self.fixed_seed,
            "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
            "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
            "initial_state_sha256": self._initial_state_sha256,
            "component_initial_state_sha256": self._component_initial_state_sha256,
            "shared_backbone_initial_state_sha256": self._shared_backbone_state_sha256,
            "ci_source_binding_sha256": self._source_binding_sha256,
            "phase2_base_source_binding_sha256": self._base_source_binding_sha256,
            "phase1_source_manifest_sha256": self._source_manifest_sha256,
            "phase1_preflight_sha256": self._preflight_sha256,
            "ci_preflight_sha256": self._ci_preflight_sha256,
            "five_date_aggregate_sha256": self._five_date_aggregate_sha256,
            "config_sha256": self._config_sha256,
            "target_optimizer_steps": 0, "target_backward_steps": 0, "checkpoint_warm_start": False,
        }
