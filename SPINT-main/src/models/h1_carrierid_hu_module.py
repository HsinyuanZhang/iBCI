"""Terminal-provenance wrapper for the H-U label-free identity arm."""
from __future__ import annotations

from typing import Any

from src.h1_m4_eb_normalized_v2_contract import NormalizedV2ContractError
from src.models.h1_carrierid_module import H1CarrierIdLitModule


class H1CarrierIdHuLitModule(H1CarrierIdLitModule):
    """Identical compact CarrierID consumer; H-U is a source-input intervention."""

    def __init__(self, *, pilot_arm: str, **kwargs: Any) -> None:
        if pilot_arm != "hu":
            raise NormalizedV2ContractError("H-U Lightning module requires pilot_arm='hu'")
        self.hu_arm = "hu"
        super().__init__(pilot_arm="full", **kwargs)
        self.pilot_arm = "hu"

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.pilot_arm = "full"
        try:
            super().on_save_checkpoint(checkpoint)
        finally:
            self.pilot_arm = self.hu_arm
        datamodule = self.trainer.datamodule
        if getattr(datamodule, "carrier_intervention", None) != "hu":
            raise NormalizedV2ContractError("H-U checkpoint/datamodule arm mismatch")
        dataset = datamodule.train_dataset
        effective_hash = getattr(dataset, "effective_source_carriers_sha256", None)
        if not isinstance(effective_hash, str) or len(effective_hash) != 64:
            raise NormalizedV2ContractError("H-U checkpoint lacks effective-carrier binding")
        metadata = checkpoint["h1_carrierid"]
        metadata.update(
            {
                "schema": "h1_carrierid_h32_hu_terminal_checkpoint_v1",
                "arm": "hu",
                "carrier_mode": "source_hu_label_free_descriptor",
                "carrier_intervention": "hu",
                "hu_version": getattr(dataset, "hu_version", None),
                "hu_feature_names": list(getattr(dataset, "hu_feature_names", ())),
                "hu_raw_sha256": getattr(dataset, "hu_raw_sha256", None),
                "effective_source_carriers_sha256": effective_hash,
                "effective_source_carriers_shape": getattr(dataset, "effective_source_carriers_shape", None),
                "effective_source_carriers_count": getattr(dataset, "effective_source_carriers_count", None),
                "label_free": True,
                "used_principal_components": False,
                "deployment_target_optimizer_steps": 0,
                "deployment_target_backward_steps": 0,
            }
        )
