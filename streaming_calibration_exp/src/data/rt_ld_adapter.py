"""Dual-carrier adapter for the RT L-D pilot.

The wrapped dataset remains an ordinary aligned Full AFC4 dataset.  This
adapter derives an audited XLSv2 descriptor from exactly the same M24 support
and normalises it with the already-fitted Full source normaliser.  Its output
is a six-item batch where Full and gain carriers occupy separate positions;
there is no path that can substitute XLSv2 into identity construction.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from torch.utils.data import Dataset

from src.data.afc4_xls_v2_adapter import (
  AUDIT_SHA256,
  afc4_xls_v2_from_support,
  load_immutable_xls_v2_audit,
)
from src.data.falcon_datamodule import FalconDataset


class RtLdDualCarrierDataset(Dataset):
  """Expose `(Full identity carrier, selected gain carrier)` separately."""

  def __init__(self, base: FalconDataset, *, gain_source: str, xls_v2_support_audit_path: str | None) -> None:
    if str(base.side_feature_group).lower() not in {"afc4_vel", "k4"}:
      raise ValueError("RT L-D base dataset must expose aligned Full AFC4")
    source = str(gain_source).lower()
    if source not in {"full", "xls_v2"}:
      raise ValueError("RT L-D gain_source must be 'full' or 'xls_v2'")
    if base.side_feature_mean is None or base.side_feature_std is None:
      raise ValueError("RT L-D requires Full's source-only carrier normalizer")
    self.base = base
    self.gain_source = source
    self.xls_v2_audit_sha256: str | None = None
    self.xls_v2_audits: dict[str, dict[str, Any]] = {}
    self._xls_v2_normalized: dict[str, np.ndarray] = {}
    if source == "xls_v2":
      if not xls_v2_support_audit_path:
        raise ValueError("RT L-D G-XLS requires an immutable XLSv2 support audit")
      audit, digest = load_immutable_xls_v2_audit(xls_v2_support_audit_path)
      if digest != AUDIT_SHA256:
        raise RuntimeError("RT L-D XLSv2 audit digest drift")
      self.xls_v2_audit_sha256 = digest
      for session_name, raw in base.k4_raw_calibration.items():
        segment_ids = raw.get("segment_ids")
        if segment_ids is None:
          raise ValueError(f"RT L-D XLSv2 requires RT segment IDs for {session_name}")
        values, descriptor_audit = afc4_xls_v2_from_support(
          raw["neural"], raw["covariates"], raw["trial_change"],
          segment_ids=segment_ids,
          session_name=session_name,
          audit_receipt=audit,
          audit_receipt_sha256=digest,
          calibration_n_trials=int(base.calibration_n_trials),
          seed=int(base.side_feature_shuffle_seed),
        )
        self._xls_v2_normalized[session_name] = (
          (values - base.side_feature_mean) / base.side_feature_std
        ).astype(np.float32)
        self.xls_v2_audits[session_name] = descriptor_audit.as_dict()

  def __getattr__(self, name: str) -> Any:
    # Samplers and RT receipts depend on FalconDataset metadata.  Delegate it
    # verbatim; only `__getitem__` changes the batch contract.
    return getattr(self.base, name)

  def __len__(self) -> int:
    return len(self.base)

  def __getitem__(self, index: int) -> tuple[Any, ...]:
    neural, target, calib, session_name, full = self.base[index]
    full = np.asarray(full, dtype=np.float32)
    if self.gain_source == "full":
      gain = full
    else:
      gain = self._xls_v2_normalized[str(session_name)]
    if full.shape != gain.shape or full.ndim != 2 or full.shape[-1] != 4:
      raise RuntimeError("RT L-D adapter emitted misaligned non-four-wide carriers")
    return neural, target, calib, session_name, full, gain
