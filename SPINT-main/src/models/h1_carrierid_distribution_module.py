"""Checkpoint provenance wrapper for fresh H1 D-S4/D-Q4 consumer training.

The network itself is the existing CarrierID h=32 network.  Only source-side
data construction and checkpoint labels live in this additive namespace.
"""
from __future__ import annotations

from typing import Any

from src.h1_m4_eb_normalized_v2_contract import NormalizedV2ContractError
from src.models.h1_carrierid_module import H1CarrierIdLitModule


class H1CarrierIdDistributionLitModule(H1CarrierIdLitModule):
    """H1 CarrierID h=32 with a matched source carrier-distribution contract."""

    def __init__(self, *, pilot_arm: str, **kwargs: Any) -> None:
        if pilot_arm not in {"s4", "q4"}:
            raise NormalizedV2ContractError("fresh distribution pilot_arm must be s4 or q4")
        self.distribution_arm = pilot_arm
        # Parent validates the same nonzero h=32 architecture and literal-zero
        # initial carrier columns; ``full`` is only its pre-existing spelling.
        super().__init__(pilot_arm="full", **kwargs)
        self.pilot_arm = pilot_arm

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Preserve all parent source/cache/initial-state accounting while
        # producing a distinct, auditable arm label in the new checkpoint.
        self.pilot_arm = "full"
        try:
            super().on_save_checkpoint(checkpoint)
        finally:
            self.pilot_arm = self.distribution_arm
        datamodule = self.trainer.datamodule
        if getattr(datamodule.hparams, "carrier_distribution_arm", None) != self.distribution_arm:
            raise NormalizedV2ContractError("fresh distribution checkpoint/datamodule arm mismatch")
        manifest = datamodule.pilot_manifest()
        if manifest.get("same_arm_inputs_except_carrier") is not True:
            raise NormalizedV2ContractError("fresh distribution source-pair contract missing")
        metadata = checkpoint["h1_carrierid"]
        metadata.update({
            "schema": "h1_carrierid_h32_fresh_distribution_terminal_checkpoint_v1",
            "arm": f"D-{self.distribution_arm.upper()}",
            "carrier_mode": (
                "standard_support_t_to_t_plus_3" if self.distribution_arm == "s4"
                else "source_query_local_t_plus_4_to_t_plus_7_LEAKAGE_DIAGNOSTIC_ONLY"
            ),
            "source_schedule_sha256": manifest["source_schedule_sha256"],
            "common_query_samples_sha256": manifest["common_query_samples_sha256"],
            "identity_schedule_sha256": manifest["identity_schedule_sha256"],
            "batch_order_sha256": manifest["batch_order_sha256"],
            "s4_effective_carriers_sha256": manifest["s4_effective_carriers_sha256"],
            "q4_effective_carriers_sha256": manifest["q4_effective_carriers_sha256"],
            "q4_label_scope": manifest["q4_label_scope"],
            "leakage_diagnostic_only": True,
            "not_for_selection_or_paper_main_result": True,
            "deployment_target_optimizer_steps": 0,
            "deployment_target_backward_steps": 0,
        })
