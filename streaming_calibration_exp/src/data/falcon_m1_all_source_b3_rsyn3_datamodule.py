"""All-source B3 windows with EMG-rSyn3 carriers attached as B3S side features.

The neural loader stays ``side_feature_group=none``. The 4-d rSyn3 carrier is
the encoder side write, replacing native T4 ``tgt_loc`` features.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.data.falcon_m1_all_source_b3_datamodule import M1AllSourceB3DataModule
from tfpd_exploration.src.m1_b3_allsource_v1 import rsyn3_bank


class RSyn3SideDataset:
    """4-tuple Falcon windows plus the frozen per-session rSyn3 carrier."""

    def __init__(self, base: Any, carriers: dict[str, np.ndarray]) -> None:
        self.base = base
        self.carriers = {
            name: np.ascontiguousarray(value, dtype=np.float32) for name, value in carriers.items()
        }

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        neural, target, calib, session = self.base[index][:4]
        if isinstance(session, bytes):
            name = session.decode("ascii")
        else:
            name = str(session)
        if not name.startswith("ses-"):
            raise ValueError(f"unrecognized session name {session!r}")
        carrier = self.carriers[name]
        n_units = int(np.asarray(calib).shape[-1])
        if carrier.shape != (n_units, 4):
            raise ValueError(f"rSyn3 carrier/unit mismatch {carrier.shape} vs {n_units}")
        return neural, target, calib, session, carrier

    def __getattr__(self, name: str):
        return getattr(self.base, name)


class M1AllSourceB3RSyn3DataModule(M1AllSourceB3DataModule):
    """Fit-only all-source M1 with B3S side = source-frozen EMG-rSyn3."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        group = str(kwargs.get("side_feature_group", "none")).lower()
        if group not in {"none", "rsyn3"}:
            raise ValueError("all-source B3S-rSyn3 requires side_feature_group none/rsyn3")
        kwargs["side_feature_group"] = "none"
        super().__init__(*args, **kwargs)
        self._rsyn3_wrapped = False
        self.rsyn3_manifest: dict[str, Any] | None = None

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        if self._rsyn3_wrapped:
            return
        bank = rsyn3_bank.build_all_source_bank(self.source_paths)
        self.rsyn3_manifest = rsyn3_bank.manifest_payload(bank)
        self.train_dataset = RSyn3SideDataset(self.train_dataset, bank["normalized"])
        self._rsyn3_wrapped = True

    def get_split_manifest(self) -> dict[str, Any]:
        manifest = super().get_split_manifest()
        manifest["side_feature_group"] = "rsyn3"
        manifest["t4_labels"] = "none"
        manifest["rsyn3_carrier"] = "emg_rsyn3_relu_nmf3_unit_ridge_m10"
        if self.rsyn3_manifest is not None:
            manifest["rsyn3_bank"] = dict(self.rsyn3_manifest)
        return manifest
