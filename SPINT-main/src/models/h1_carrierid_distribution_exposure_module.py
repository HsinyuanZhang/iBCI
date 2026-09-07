"""Terminal-checkpoint provenance for H1 exposure-matched D-S4e/D-Q4e."""
from __future__ import annotations

from typing import Any

from src.h1_m4_eb_normalized_v2_contract import NormalizedV2ContractError
from src.models.h1_carrierid_distribution_module import H1CarrierIdDistributionLitModule


class H1CarrierIdDistributionExposureLitModule(H1CarrierIdDistributionLitModule):
    """The unchanged h=32 consumer with an exposure-matched source receipt."""

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        datamodule = self.trainer.datamodule
        manifest = datamodule.pilot_manifest()
        required = {
            "exposure_matched_to_h_c": True,
            "schedule_count": 72,
            "sample_count": 115_520,
            "scheduled_samples_per_epoch": 115_520,
            "batches_per_epoch": 3_610,
            "epochs": 50,
            "same_arm_inputs_except_carrier": True,
        }
        for key, expected in required.items():
            if manifest.get(key) != expected:
                raise NormalizedV2ContractError(f"exposure terminal checkpoint contract drift at {key}")
        arm = str(getattr(datamodule.hparams, "carrier_distribution_arm", ""))
        if arm != self.distribution_arm:
            raise NormalizedV2ContractError("exposure checkpoint/datamodule arm mismatch")
        metadata = checkpoint["h1_carrierid"]
        metadata.update({
            "schema": "h1_carrierid_h32_fresh_distribution_exposure_terminal_checkpoint_v1",
            "arm": f"D-{arm.upper()}E",
            "carrier_mode": (
                "standard_support_t_to_t_plus_3" if arm == "s4"
                else "source_query_local_t_plus_4_to_t_plus_7_LEAKAGE_DIAGNOSTIC_ONLY"
            ),
            "source_schedule_sha256": manifest["source_schedule_sha256"],
            "common_query_samples_sha256": manifest["common_query_samples_sha256"],
            "identity_schedule_sha256": manifest["identity_schedule_sha256"],
            "batch_order_sha256": manifest["batch_order_sha256"],
            "s4_effective_carriers_sha256": manifest["s4_effective_carriers_sha256"],
            "q4_effective_carriers_sha256": manifest["q4_effective_carriers_sha256"],
            "exposure_sample_plan": manifest["exposure_sample_plan"],
            "exposure_matched_to_h_c": True,
            "q4_label_scope": manifest["q4_label_scope"],
            "leakage_diagnostic_only": True,
            "not_for_selection_or_paper_main_result": True,
            "deployment_target_optimizer_steps": 0,
            "deployment_target_backward_steps": 0,
        })
