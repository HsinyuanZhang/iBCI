"""Terminal-provenance wrapper for the separately trained H1 RS/LS controls."""
from __future__ import annotations

from typing import Any

from src.h1_m4_eb_normalized_v2_contract import NormalizedV2ContractError
from src.models.h1_carrierid_module import H1CarrierIdLitModule


class H1CarrierIdShuffleLitModule(H1CarrierIdLitModule):
    """CarrierID with unchanged Full architecture and an auditable source control."""

    def __init__(self, *, pilot_arm: str, **kwargs: Any) -> None:
        if pilot_arm not in {"rs", "ls"}:
            raise NormalizedV2ContractError("H1 CarrierID shuffle pilot_arm must be rs or ls")
        self.shuffle_arm = pilot_arm
        # The parent validates the unmodified, nonzero CarrierID architecture.
        super().__init__(pilot_arm="full", **kwargs)
        self.pilot_arm = pilot_arm

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Retain the parent's strict fixed-epoch and source-provenance checks,
        # but identify this separately trained source intervention correctly.
        self.pilot_arm = "full"
        try:
            super().on_save_checkpoint(checkpoint)
        finally:
            self.pilot_arm = self.shuffle_arm
        datamodule = self.trainer.datamodule
        if getattr(datamodule, "carrier_intervention", None) != self.shuffle_arm:
            raise NormalizedV2ContractError("H1 CarrierID shuffle checkpoint/datamodule arm mismatch")
        effective_hash = getattr(datamodule.train_dataset, "effective_source_carriers_sha256", None)
        effective_shape = getattr(datamodule.train_dataset, "effective_source_carriers_shape", None)
        effective_count = getattr(datamodule.train_dataset, "effective_source_carriers_count", None)
        nonidentity_all = getattr(datamodule.train_dataset, "effective_source_carriers_nonidentity_all", None)
        if not isinstance(effective_hash, str) or len(effective_hash) != 64 or effective_count != 116 or nonidentity_all is not True:
            raise NormalizedV2ContractError("H1 CarrierID shuffle checkpoint lacks complete effective-carrier binding")
        metadata = checkpoint["h1_carrierid"]
        metadata.update({
            "schema": "h1_carrierid_h32_shuffle_terminal_checkpoint_v1",
            "arm": self.shuffle_arm,
            "carrier_mode": f"source_{self.shuffle_arm}_shuffle_training_input",
            "carrier_intervention": self.shuffle_arm,
            "effective_source_carriers_sha256": effective_hash,
            "effective_source_carriers_shape": effective_shape,
            "effective_source_carriers_count": effective_count,
            "effective_source_carriers_nonidentity_all": nonidentity_all,
            "deployment_target_optimizer_steps": 0,
            "deployment_target_backward_steps": 0,
        })
