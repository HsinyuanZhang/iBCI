"""RT-only nested-LOSO DataModule for the L-D three-arm pilot."""
from __future__ import annotations

from typing import Any

from src.data.rt_ld_adapter import RtLdDualCarrierDataset
from src.data.rt_nested_loso_datamodule import RtNestedLossoDataModule


class RtLdNestedLossoDataModule(RtNestedLossoDataModule):
  """Use Full AFC4 for E and a separately selected Full/XLSv2 gain carrier."""

  def __init__(self, *args: Any, rt_ld_gain_source: str, **kwargs: Any) -> None:
    if str(kwargs.get("side_feature_group", "")).lower() not in {"afc4_vel", "k4"}:
      raise ValueError("RT L-D requires side_feature_group='afc4_vel' for the complete Full identity")
    source = str(rt_ld_gain_source).lower()
    if source not in {"full", "xls_v2"}:
      raise ValueError("rt_ld_gain_source must be 'full' or 'xls_v2'")
    if source == "xls_v2" and not kwargs.get("xls_v2_support_audit_path"):
      raise ValueError("RT L-D G-XLS requires xls_v2_support_audit_path")
    self._rt_ld_gain_source = source
    super().__init__(*args, **kwargs)

  def setup(self, stage: str | None = None) -> None:
    if self._setup_complete:
      return
    super().setup(stage)
    self.train_dataset = RtLdDualCarrierDataset(
      self.train_dataset,
      gain_source=self._rt_ld_gain_source,
      xls_v2_support_audit_path=self.hparams.xls_v2_support_audit_path,
    )
    self.val_inner_dataset = RtLdDualCarrierDataset(
      self.val_inner_dataset,
      gain_source=self._rt_ld_gain_source,
      xls_v2_support_audit_path=self.hparams.xls_v2_support_audit_path,
    )

  def get_split_manifest(self) -> dict[str, Any]:
    manifest = super().get_split_manifest()
    # The generic manifest's arm remains Full because it describes identity.
    # This addendum names the isolated consumer-only XLS branch explicitly.
    manifest["rt_ld"] = {
      "enabled": True,
      "identity_carrier": "aligned_full_afc4",
      "gain_carrier": "aligned_full_afc4" if self._rt_ld_gain_source == "full" else "strong_xls_v2",
      "gain_source": self._rt_ld_gain_source,
      "gain_input_only": True,
      "identity_never_receives_xls_v2": True,
      "common_inverse_or_alignment_map": False,
      "xls_v2_support_audit_sha256": (
        self.train_dataset.xls_v2_audit_sha256 if self._rt_ld_gain_source == "xls_v2" else None
      ),
      "per_session_xls_v2_permutation_sha256": (
        {
          **{
            name: audit["label_permutation_sha256"]
            for name, audit in self.train_dataset.xls_v2_audits.items()
          },
          **{
            name: audit["label_permutation_sha256"]
            for name, audit in self.val_inner_dataset.xls_v2_audits.items()
          },
        }
        if self._rt_ld_gain_source == "xls_v2" else {}
      ),
    }
    return manifest


RtLdNestedLOSODataModule = RtLdNestedLossoDataModule
