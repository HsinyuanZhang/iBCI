"""Source-only fold0 windows: 26/27/28. Outer 20120924 is not resolved."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch

from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import plan as fold_plan
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import M1Bank

from . import bank as v2_bank
from . import calibration
from . import plan


def _ensure_experiment_path() -> None:
    experiment = str(plan.REPO_ROOT / "streaming_calibration_exp")
    if experiment not in sys.path:
        sys.path.insert(0, experiment)


def _session_name(session: Any) -> str:
    text = session.decode("ascii") if isinstance(session, bytes) else str(session)
    if not text.startswith("ses-"):
        raise RuntimeError(f"unrecognized session {session!r}")
    return text


class SealedSourceCarrierDataset:
    """Attach the sealed per-session rSyn3 vector. Load-only; no NMF."""

    def __init__(self, base: Any, source_bank: dict[str, Any]) -> None:
        self.base = base
        self.source_bank = source_bank

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        neural, target, calib, session = self.base[index][:4]
        name = _session_name(session)
        carrier = v2_bank.carrier_for_session(self.source_bank, name)
        n_units = int(np.asarray(calib).shape[-1])
        if tuple(carrier.shape) != (n_units, 4):
            raise RuntimeError(f"carrier/unit mismatch {carrier.shape} vs {n_units}")
        return neural, target, calib, session, np.ascontiguousarray(carrier, dtype=np.float32)

    def __getattr__(self, name: str):
        return getattr(self.base, name)


def build_source_only_datamodule(source_bank: dict[str, Any] | None = None):
    _ensure_experiment_path()
    from src.data.m1_version_b_source_loso_datamodule import M1VersionBSourceOnlyFitDataModule

    loaded = source_bank or v2_bank.load_source_bank()
    data_dir = plan.DATA_DIR
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    datamodule = M1VersionBSourceOnlyFitDataModule(
        task="m1",
        data_dir=str(data_dir),
        source_session_names=list(plan.SOURCE_SESSIONS),
        heldin_session_names=list(plan.SOURCE_SESSIONS),
        batch_size=fold_plan.STAGE1_BATCH_SIZE,
        window_size=fold_plan.WINDOW_SIZE,
        calibration_n_trials=plan.SUPPORT_TRIALS,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=fold_plan.TRIAL_LENGTH,
        standardize_covariates=False,
        use_intertrials=True,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        pad_value=-1.0,
        validation_protocol="loso",
        loso_fold=0,
        include_heldout_in_fit=False,
        include_heldout_in_test=False,
        query_start_trial=0,
        heldin_query_start_trial=fold_plan.QUERY_START,
        heldin_query_end_trial=fold_plan.QUERY_STOP_EXCLUSIVE,
        allow_empty_heldout_query=False,
        num_workers=0,
        pin_memory=False,
        sampler_seed=plan.SEED,
        balance_session_batches=False,
        reshuffle_train_sampler_each_epoch=False,
        afc4_arm="none",
    )
    datamodule.setup("fit")
    if getattr(datamodule, "target_path", None) is not None:
        raise RuntimeError("source-only fit resolved a target path")
    if datamodule.val_heldin_dataset is not None or datamodule.val_heldout_dataset is not None:
        raise RuntimeError("source-only fit built a query validation set")
    if list(datamodule.train_session_names) != list(plan.SOURCE_SESSIONS):
        raise RuntimeError(f"train sessions drifted: {datamodule.train_session_names}")
    datamodule.train_dataset = SealedSourceCarrierDataset(datamodule.train_dataset, loaded)
    return datamodule


def materialize_source_banks() -> dict[str, M1Bank]:
    mat = calibration.load_frozen_m1_materializer()
    data = build_source_only_datamodule(mat.bank)
    dataset = data.train_dataset
    banks: dict[str, M1Bank] = {}
    for session in plan.SOURCE_SESSIONS:
        calib = dataset.base.calib_trialized_neural_features[session][:10]
        tensor = torch.as_tensor(calib[None, ...], dtype=torch.float32)
        banks[session] = mat.materialize_bank(tensor, session)
    return banks


def isolation_receipt(datamodule: Any) -> dict[str, Any]:
    manifest = datamodule.get_split_manifest()
    return {
        "revision": plan.REVISION,
        "carrier_name": plan.CARRIER_NAME,
        "train_sessions": list(datamodule.train_session_names),
        "target_path_resolved": getattr(datamodule, "target_path", None) is not None,
        "val_heldin_dataset": datamodule.val_heldin_dataset is not None,
        "outer_left_out_name_only": getattr(datamodule, "outer_left_out", None),
        "manifest": manifest,
    }
